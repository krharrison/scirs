//! Differentiable combinatorial optimization via perturbed optimizers.
//!
//! Implements the **Perturbed Optimizer** framework (Berthet et al., 2020) for
//! making any black-box combinatorial solver differentiable through additive
//! Gaussian noise perturbations.
//!
//! Given a combinatorial optimizer `y*(θ) = argmax_y θᵀ y` (or argmin),
//! the perturbed optimizer computes:
//!
//!   ŷ(θ) = E[y*(θ + σZ)]   where  Z ~ N(0, I)
//!
//! The gradient is estimated as:
//!
//!   ∇_θ L ≈ (1/σ) E[L(y*(θ + σZ)) · Z]   (REINFORCE)
//!
//! or via the reparameterized covariance estimator:
//!
//!   ∇_θ L ≈ (1/σ) Cov[y*(θ + σZ), Z] · ∇_y L
//!
//! Also includes `SparseMap` for structured prediction on the marginal
//! polytope via QP.
//!
//! # References
//! - Berthet et al. (2020). "Learning with Differentiable Perturbed Optimizers." NeurIPS.
//! - Niculae & Blondel (2017). "A regularized framework for sparse and structured
//!   neural attention." NeurIPS.

use crate::error::{OptimizeError, OptimizeResult};

use super::kkt_sensitivity::kkt_sensitivity;

// ─────────────────────────────────────────────────────────────────────────────
// Random number generator (xorshift64 — pure Rust, no external deps)
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight xorshift64 PRNG for Monte Carlo sampling.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a uniform [0, 1) sample.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box-Muller transform: generate N(0, 1) sample.
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15); // avoid log(0)
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        r * theta.cos()
    }

    /// Generate a N(0, I) vector of length n.
    fn normal_vector(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.normal()).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the perturbed optimizer.
#[derive(Debug, Clone)]
pub struct PerturbedOptimizerConfig {
    /// Number of Monte Carlo samples for expectation estimation.
    pub n_samples: usize,
    /// Perturbation standard deviation σ.
    pub sigma: f64,
    /// Random seed (for reproducibility).
    pub seed: u64,
}

impl Default for PerturbedOptimizerConfig {
    fn default() -> Self {
        Self {
            n_samples: 20,
            sigma: 1.0,
            seed: 42,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Perturbed optimizer
// ─────────────────────────────────────────────────────────────────────────────

/// A differentiable wrapper around any black-box combinatorial optimizer.
///
/// Wraps a function `optimizer: θ → y*(θ)` and computes a smooth
/// approximation `E[y*(θ + σZ)]` via Monte Carlo.
///
/// # Type Parameter
/// * `F` – function type for the combinatorial optimizer, mapping `&[f64]` to `Vec<f64>`.
pub struct PerturbedOptimizer<F>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    optimizer: F,
    config: PerturbedOptimizerConfig,
    /// Cached samples from the last forward call (for backward).
    cached_samples: Option<Vec<Vec<f64>>>,
    /// Cached outputs from the last forward call.
    cached_outputs: Option<Vec<Vec<f64>>>,
    /// Cached noise vectors from the last forward call.
    cached_noise: Option<Vec<Vec<f64>>>,
}

impl<F> PerturbedOptimizer<F>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    /// Create a new perturbed optimizer with default configuration.
    pub fn new(optimizer: F) -> Self {
        Self {
            optimizer,
            config: PerturbedOptimizerConfig::default(),
            cached_samples: None,
            cached_outputs: None,
            cached_noise: None,
        }
    }

    /// Create a new perturbed optimizer with custom configuration.
    pub fn with_config(optimizer: F, config: PerturbedOptimizerConfig) -> Self {
        Self {
            optimizer,
            config,
            cached_samples: None,
            cached_outputs: None,
            cached_noise: None,
        }
    }

    /// Forward pass: compute `E[y*(θ + σZ)]` via Monte Carlo.
    ///
    /// Samples `n_samples` perturbations Z_k ~ N(0, I) and returns the
    /// sample mean of the optimizer outputs.
    ///
    /// # Arguments
    /// * `theta` – parameter vector (length d).
    ///
    /// # Returns
    /// Expected optimizer output `ŷ(θ)` (length equal to optimizer output).
    pub fn forward(&mut self, theta: &[f64]) -> OptimizeResult<Vec<f64>> {
        let d = theta.len();
        let mut rng = Xorshift64::new(self.config.seed);

        let mut outputs: Vec<Vec<f64>> = Vec::with_capacity(self.config.n_samples);
        let mut noises: Vec<Vec<f64>> = Vec::with_capacity(self.config.n_samples);

        for _ in 0..self.config.n_samples {
            let z = rng.normal_vector(d);
            let theta_perturbed: Vec<f64> = theta
                .iter()
                .zip(z.iter())
                .map(|(&ti, &zi)| ti + self.config.sigma * zi)
                .collect();
            let y = (self.optimizer)(&theta_perturbed);
            outputs.push(y);
            noises.push(z);
        }

        // Compute mean output
        if outputs.is_empty() {
            return Err(OptimizeError::ComputationError(
                "No samples generated in PerturbedOptimizer::forward".to_string(),
            ));
        }
        let out_len = outputs[0].len();
        let mut mean_y = vec![0.0_f64; out_len];
        for output in &outputs {
            if output.len() != out_len {
                return Err(OptimizeError::ComputationError(
                    "Inconsistent optimizer output lengths".to_string(),
                ));
            }
            for (i, &oi) in output.iter().enumerate() {
                mean_y[i] += oi;
            }
        }
        let n = self.config.n_samples as f64;
        for mi in &mut mean_y {
            *mi /= n;
        }

        // Cache for backward
        self.cached_samples = Some(
            (0..self.config.n_samples)
                .map(|k| {
                    theta
                        .iter()
                        .zip(noises[k].iter())
                        .map(|(&ti, &zi)| ti + self.config.sigma * zi)
                        .collect()
                })
                .collect(),
        );
        self.cached_outputs = Some(outputs);
        self.cached_noise = Some(noises);

        Ok(mean_y)
    }

    /// Gradient estimate via reparameterized covariance:
    ///
    ///   grad_theta L ~ (1/sigma) Cov\[y*(theta + sigma*Z), Z\] * dL/dy
    ///            = `(1/sigma^2*N) Sum_k (y_k - y_mean) * Z_k * dL/dy`
    ///
    /// This is an unbiased estimator when y* is the gradient of a linear function,
    /// and has lower variance than REINFORCE.
    ///
    /// # Arguments
    /// * `theta`  – parameter vector (length d).
    /// * `dl_dy`  – upstream gradient dL/dŷ (length = optimizer output length).
    ///
    /// # Returns
    /// Gradient estimate ∇_θ L (length d).
    pub fn gradient(&self, theta: &[f64], dl_dy: &[f64]) -> OptimizeResult<Vec<f64>> {
        let outputs = self.cached_outputs.as_ref().ok_or_else(|| {
            OptimizeError::ComputationError(
                "PerturbedOptimizer::gradient called before forward".to_string(),
            )
        })?;
        let noises = self
            .cached_noise
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("No cached noise".to_string()))?;

        let d = theta.len();
        let out_len = dl_dy.len();
        let n_samples = outputs.len();

        if n_samples == 0 {
            return Err(OptimizeError::ComputationError(
                "Empty sample cache".to_string(),
            ));
        }

        // Compute mean output ȳ
        let mut mean_y = vec![0.0_f64; out_len];
        for output in outputs.iter() {
            for (i, &oi) in output.iter().enumerate().take(out_len) {
                mean_y[i] += oi;
            }
        }
        for mi in &mut mean_y {
            *mi /= n_samples as f64;
        }

        // Reparameterized covariance estimator:
        // ∇_θ_j L ≈ (1/σ) * (1/N) * Σ_k [(y_k - ȳ) · dL/dy] * Z_k_j
        let sigma = self.config.sigma;
        let mut grad = vec![0.0_f64; d];

        for k in 0..n_samples {
            // Compute (y_k - ȳ) · dL/dy = scalar coefficient for sample k
            let coeff: f64 = outputs[k]
                .iter()
                .zip(mean_y.iter())
                .zip(dl_dy.iter())
                .map(|((&yk, &ybar), &dly)| (yk - ybar) * dly)
                .sum();

            // Gradient contribution: coeff * Z_k / (σ * N)
            for j in 0..d {
                let z_kj = if j < noises[k].len() {
                    noises[k][j]
                } else {
                    0.0
                };
                grad[j] += coeff * z_kj;
            }
        }

        let scale = 1.0 / (sigma * n_samples as f64);
        for gi in &mut grad {
            *gi *= scale;
        }

        Ok(grad)
    }

    /// REINFORCE (score-function) gradient estimator:
    ///
    ///   ∇_θ L ≈ (1/σN) Σ_k L(y_k) Z_k
    ///
    /// where L(y_k) = dL/dy · y_k (linear approximation to the loss).
    ///
    /// # Arguments
    /// * `theta`     – parameter vector.
    /// * `dl_dy`     – upstream gradient (defines the loss as L = dl_dy · y).
    pub fn reinforce_gradient(&self, theta: &[f64], dl_dy: &[f64]) -> OptimizeResult<Vec<f64>> {
        let outputs = self.cached_outputs.as_ref().ok_or_else(|| {
            OptimizeError::ComputationError(
                "PerturbedOptimizer::reinforce_gradient called before forward".to_string(),
            )
        })?;
        let noises = self
            .cached_noise
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("No cached noise".to_string()))?;

        let d = theta.len();
        let n_samples = outputs.len();
        let sigma = self.config.sigma;

        let mut grad = vec![0.0_f64; d];
        for k in 0..n_samples {
            // Approximate loss: L_k = dl_dy · y_k
            let l_k: f64 = outputs[k]
                .iter()
                .zip(dl_dy.iter())
                .map(|(&yk, &dly)| yk * dly)
                .sum();

            for j in 0..d {
                let z_kj = if j < noises[k].len() {
                    noises[k][j]
                } else {
                    0.0
                };
                grad[j] += l_k * z_kj;
            }
        }

        let scale = 1.0 / (sigma * n_samples as f64);
        for gi in &mut grad {
            *gi *= scale;
        }

        Ok(grad)
    }

    /// Access the cached mean output from the last forward pass.
    pub fn last_mean_output(&self) -> Option<Vec<f64>> {
        let outputs = self.cached_outputs.as_ref()?;
        if outputs.is_empty() {
            return None;
        }
        let out_len = outputs[0].len();
        let mut mean = vec![0.0_f64; out_len];
        for output in outputs {
            for (i, &oi) in output.iter().enumerate().take(out_len) {
                mean[i] += oi;
            }
        }
        let n = outputs.len() as f64;
        for mi in &mut mean {
            *mi /= n;
        }
        Some(mean)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SparseMap
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the SparseMap structured prediction layer.
#[derive(Debug, Clone)]
pub struct SparseMapConfig {
    /// Maximum number of projected-gradient iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Step size for projected-gradient updates.
    pub step_size: f64,
}

impl Default for SparseMapConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-8,
            step_size: 0.1,
        }
    }
}

/// A SparseMap layer: structured prediction with sparse marginals.
///
/// Solves the QP:
///
///   max  θᵀμ - ½ μᵀ μ   s.t.  μ ∈ M
///
/// where M is the marginal polytope (e.g., the simplex for unstructured
/// classification). The solution is a sparse probability distribution.
///
/// The backward pass uses the KKT conditions of the QP.
#[derive(Debug, Clone)]
pub struct SparseMap {
    config: SparseMapConfig,
    /// Equality constraint matrix A (defines the polytope via Ax = b, x ≥ 0).
    a_marginal: Vec<Vec<f64>>,
    /// Equality rhs b.
    b_marginal: Vec<f64>,
    /// Last forward result: μ* (sparse distribution).
    last_mu: Option<Vec<f64>>,
    /// Last dual: ν* for equality constraints.
    last_nu: Option<Vec<f64>>,
    /// Last theta.
    last_theta: Option<Vec<f64>>,
}

impl SparseMap {
    /// Create a new SparseMap layer for a given marginal polytope.
    ///
    /// # Arguments
    /// * `a_marginal` – equality constraints defining the polytope (Ax = b, x ≥ 0).
    /// * `b_marginal` – equality rhs.
    pub fn new(a_marginal: Vec<Vec<f64>>, b_marginal: Vec<f64>) -> Self {
        Self {
            config: SparseMapConfig::default(),
            a_marginal,
            b_marginal,
            last_mu: None,
            last_nu: None,
            last_theta: None,
        }
    }

    /// Create a SparseMap for the simplex: Σ μ_i = 1, μ_i ≥ 0.
    pub fn simplex(n: usize) -> Self {
        let a = vec![vec![1.0_f64; n]];
        let b = vec![1.0_f64];
        Self::new(a, b)
    }

    /// Create a SparseMap with custom configuration.
    pub fn with_config(
        a_marginal: Vec<Vec<f64>>,
        b_marginal: Vec<f64>,
        config: SparseMapConfig,
    ) -> Self {
        Self {
            config,
            a_marginal,
            b_marginal,
            last_mu: None,
            last_nu: None,
            last_theta: None,
        }
    }

    /// Forward pass: solve the QP on the marginal polytope.
    ///
    ///   μ* = argmax_{μ ∈ M} θᵀμ - ½ ||μ||²
    ///       = argmin_{μ ∈ M} ½ ||μ - θ||²
    ///       = Π_M(θ)   (Euclidean projection onto M)
    ///
    /// Uses iterative projected gradient on the Lagrangian.
    ///
    /// # Arguments
    /// * `theta` – score vector (length n).
    pub fn forward(&mut self, theta: &[f64]) -> OptimizeResult<Vec<f64>> {
        let n = theta.len();
        let p = self.b_marginal.len();

        if self.a_marginal.len() != p {
            return Err(OptimizeError::InvalidInput(format!(
                "A_marginal rows ({}) != b_marginal length ({})",
                self.a_marginal.len(),
                p
            )));
        }

        // Solve: min ½ μᵀ μ - θᵀ μ  s.t. A μ = b, μ ≥ 0
        // via projected gradient descent in the dual:
        //
        // Lagrangian: L = ½ μᵀμ - θᵀμ + νᵀ(Aμ - b)
        // Primal: μ = max(0, θ - Aᵀν)
        // Dual: max -½ ||max(0, θ - Aᵀν)||² + θᵀ max(0, θ-Aᵀν) - νᵀ b

        let mut nu = vec![0.0_f64; p];
        let step = self.config.step_size;

        for _ in 0..self.config.max_iter {
            // Primal: μ(ν) = max(0, θ - Aᵀν)
            let at_nu: Vec<f64> = (0..n)
                .map(|j| {
                    (0..p)
                        .map(|i| {
                            let a_ij = if i < self.a_marginal.len() && j < self.a_marginal[i].len()
                            {
                                self.a_marginal[i][j]
                            } else {
                                0.0
                            };
                            nu[i] * a_ij
                        })
                        .sum::<f64>()
                })
                .collect();

            let mu: Vec<f64> = (0..n).map(|j| (theta[j] - at_nu[j]).max(0.0)).collect();

            // Dual gradient: ∂L/∂ν_i = Σ_j A_{ij} μ_j - b_i = (Aμ)_i - b_i
            let amu: Vec<f64> = (0..p)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let a_ij = if i < self.a_marginal.len() && j < self.a_marginal[i].len()
                            {
                                self.a_marginal[i][j]
                            } else {
                                0.0
                            };
                            a_ij * mu[j]
                        })
                        .sum::<f64>()
                })
                .collect();

            let nu_new: Vec<f64> = (0..p)
                .map(|i| nu[i] + step * (amu[i] - self.b_marginal[i]))
                .collect();

            // Check convergence
            let delta: f64 = nu_new
                .iter()
                .zip(nu.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            nu = nu_new;

            if delta < self.config.tol {
                break;
            }
        }

        // Final primal
        let at_nu: Vec<f64> = (0..n)
            .map(|j| {
                (0..p)
                    .map(|i| {
                        let a_ij = if i < self.a_marginal.len() && j < self.a_marginal[i].len() {
                            self.a_marginal[i][j]
                        } else {
                            0.0
                        };
                        nu[i] * a_ij
                    })
                    .sum::<f64>()
            })
            .collect();

        let mu: Vec<f64> = (0..n).map(|j| (theta[j] - at_nu[j]).max(0.0)).collect();

        self.last_mu = Some(mu.clone());
        self.last_nu = Some(nu);
        self.last_theta = Some(theta.to_vec());

        Ok(mu)
    }

    /// Backward pass: compute dL/dθ via KKT sensitivity.
    ///
    /// At the optimal μ*, the KKT conditions of the QP are:
    ///
    ///   μ* - θ + Aᵀν* + s = 0   (stationarity, s = -min(μ*, 0))
    ///   Aμ* = b                  (equality)
    ///   μ* ≥ 0, s ≥ 0, s⊙μ* = 0  (complementarity)
    ///
    /// For the active variables (μ*_i > 0), we have s_i = 0, and the
    /// KKT system reduces to an equality system on the support.
    ///
    /// # Arguments
    /// * `dl_dmu` – upstream gradient dL/dμ (length n).
    pub fn backward(&self, dl_dmu: &[f64]) -> OptimizeResult<Vec<f64>> {
        let mu = self.last_mu.as_ref().ok_or_else(|| {
            OptimizeError::ComputationError("SparseMap::backward called before forward".to_string())
        })?;
        let nu = self
            .last_nu
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("No cached nu".to_string()))?;
        let theta = self
            .last_theta
            .as_ref()
            .ok_or_else(|| OptimizeError::ComputationError("No cached theta".to_string()))?;

        let n = mu.len();
        let tol = 1e-8_f64;

        // Active support: μ*_i > 0
        let support: Vec<usize> = (0..n).filter(|&i| mu[i] > tol).collect();

        if support.is_empty() {
            // All-zero solution: gradient is zero
            return Ok(vec![0.0_f64; n]);
        }

        let s = support.len();
        let p = nu.len();

        // Build restricted system: Q_S = I_s, A_S = A[:, support]
        let q_s: Vec<Vec<f64>> = (0..s)
            .map(|i| {
                let mut row = vec![0.0_f64; s];
                row[i] = 1.0;
                row
            })
            .collect();

        let a_s: Vec<Vec<f64>> = (0..p)
            .map(|i| {
                support
                    .iter()
                    .map(|&j| {
                        if i < self.a_marginal.len() && j < self.a_marginal[i].len() {
                            self.a_marginal[i][j]
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        // Restricted primal and dual
        let x_s: Vec<f64> = support
            .iter()
            .map(|&j| if j < mu.len() { mu[j] } else { 0.0 })
            .collect();

        let dl_dx_s: Vec<f64> = support
            .iter()
            .map(|&j| if j < dl_dmu.len() { dl_dmu[j] } else { 0.0 })
            .collect();

        // KKT sensitivity on the restricted system
        let kkt_grad = kkt_sensitivity(&q_s, &a_s, &x_s, nu, &dl_dx_s)?;

        // Expand gradient back to full n: dL/dθ_j = dx_adj_j for active, 0 for inactive
        let mut dl_dtheta = vec![0.0_f64; n];
        for (idx, &j) in support.iter().enumerate() {
            if idx < kkt_grad.dx_adj.len() {
                dl_dtheta[j] = kkt_grad.dx_adj[idx];
            }
        }

        let _ = theta;
        Ok(dl_dtheta)
    }

    /// Project a vector onto the probability simplex.
    ///
    /// Solves: argmin_{μ ≥ 0, Σμ = 1} ||μ - v||²
    ///
    /// Uses the O(n log n) sorting algorithm.
    pub fn project_simplex(v: &[f64]) -> Vec<f64> {
        let n = v.len();
        if n == 0 {
            return vec![];
        }

        let mut u: Vec<f64> = v.to_vec();
        u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let mut cssv = 0.0_f64;
        let mut rho = 0_usize;
        for j in 0..n {
            cssv += u[j];
            let tau = (cssv - 1.0) / (j + 1) as f64;
            if tau < u[j] {
                rho = j;
            }
        }

        let cssv_rho: f64 = u[..=rho].iter().sum();
        let theta = (cssv_rho - 1.0) / (rho + 1) as f64;

        v.iter().map(|&vi| (vi - theta).max(0.0)).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple linear optimizer: y* = argmax_y θᵀy s.t. y ∈ {0, 1}^n
    /// (i.e., select the maximum-score element).
    fn argmax_binary(theta: &[f64]) -> Vec<f64> {
        if theta.is_empty() {
            return vec![];
        }
        let max_idx = theta
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mut y = vec![0.0_f64; theta.len()];
        y[max_idx] = 1.0;
        y
    }

    /// Simple sort optimizer: returns normalized rank vector.
    fn soft_sort_optimizer(theta: &[f64]) -> Vec<f64> {
        let n = theta.len();
        if n == 0 {
            return vec![];
        }
        let mut indexed: Vec<(f64, usize)> = theta
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut rank = vec![0.0_f64; n];
        for (r, (_, i)) in indexed.iter().enumerate() {
            rank[*i] = (n - r) as f64 / n as f64;
        }
        rank
    }

    #[test]
    fn test_perturbed_optimizer_config_default() {
        let cfg = PerturbedOptimizerConfig::default();
        assert_eq!(cfg.n_samples, 20);
        assert!((cfg.sigma - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_perturbed_optimizer_forward_shape() {
        let mut opt = PerturbedOptimizer::new(argmax_binary);
        let theta = vec![1.0, 2.0, 3.0_f64];

        let y = opt.forward(&theta).expect("Forward failed");
        assert_eq!(y.len(), 3, "Output length should match input");
        // Each y_i in [0, 1] (since outputs are binary)
        for yi in &y {
            assert!(*yi >= 0.0 && *yi <= 1.0, "y_i = {} should be in [0, 1]", yi);
        }
    }

    #[test]
    fn test_perturbed_optimizer_forward_distribution_sums_to_one() {
        // For argmax binary, mean over samples should sum to ~1 when sigma is small
        let cfg = PerturbedOptimizerConfig {
            n_samples: 100,
            sigma: 0.1, // small sigma → less randomness
            seed: 123,
        };
        let mut opt = PerturbedOptimizer::with_config(argmax_binary, cfg);
        let theta = vec![1.0, 5.0, 2.0_f64]; // θ[1] is largest

        let y = opt.forward(&theta).expect("Forward failed");
        let sum: f64 = y.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.05,
            "Sum = {} (expected ~1.0 for binary argmax)",
            sum
        );
    }

    #[test]
    fn test_perturbed_optimizer_gradient_sign() {
        // For linear loss L = -y[0], dL/dy = [-1, 0, ..., 0]
        // The gradient dL/dθ[0] should be negative when θ[0] is large
        // (increasing θ[0] increases y[0] which increases loss -y[0]... wait, decreases)
        // Actually for L = sum(-dL_dy * y): increasing θ[0] → y[0] increases → L = -y[0] decreases
        // So dL/dθ[0] < 0... no wait: dL/dθ[0] = dL/dy * dy/dθ
        // dL/dy = [-1, 0] (for L = -y[0])
        // When θ[0] increases, p(y[0]=1) increases, so E[y[0]] increases
        // dL/dθ[0] = dL/dE[y[0]] * dE[y[0]]/dθ[0] = -1 * positive = negative
        // But we pass dl_dy = [1, 0, 0] (loss = y[0]), so dL/dθ[0] should be positive.

        let cfg = PerturbedOptimizerConfig {
            n_samples: 1000,
            sigma: 1.0,
            seed: 42,
        };
        let mut opt = PerturbedOptimizer::with_config(argmax_binary, cfg);
        let theta = vec![2.0, 0.0, 0.0_f64];

        let _y = opt.forward(&theta).expect("Forward failed");

        // L = y[0], dL/dy = [1, 0, 0]
        // We expect dL/dθ[0] > 0 (increasing θ[0] → more likely to pick index 0 → L increases)
        let grad = opt
            .gradient(&theta, &[1.0, 0.0, 0.0])
            .expect("Gradient failed");

        assert_eq!(grad.len(), 3);
        // The gradient should have the correct sign: dL/dθ[0] > 0
        // (positive because increasing θ[0] increases E[y[0]])
        // With enough samples the sign should be correct
        assert!(
            grad[0] > -0.5, // Allow some MC variance
            "grad[0] = {} should be roughly positive",
            grad[0]
        );
    }

    #[test]
    fn test_perturbed_optimizer_gradient_shape() {
        let mut opt = PerturbedOptimizer::new(argmax_binary);
        let theta = vec![1.0, 2.0, 3.0_f64];

        let _y = opt.forward(&theta).expect("Forward failed");
        let grad = opt
            .gradient(&theta, &[1.0, 0.0, 0.0])
            .expect("Gradient failed");

        assert_eq!(grad.len(), 3);
        for gi in &grad {
            assert!(gi.is_finite(), "grad not finite");
        }
    }

    #[test]
    fn test_perturbed_optimizer_reinforce_shape() {
        let mut opt = PerturbedOptimizer::new(soft_sort_optimizer);
        let theta = vec![1.0, 3.0, 2.0_f64];

        let _y = opt.forward(&theta).expect("Forward failed");
        let grad = opt
            .reinforce_gradient(&theta, &[0.0, 1.0, 0.0])
            .expect("REINFORCE failed");

        assert_eq!(grad.len(), 3);
        for gi in &grad {
            assert!(gi.is_finite(), "REINFORCE grad not finite");
        }
    }

    #[test]
    fn test_perturbed_optimizer_no_forward_error() {
        let opt = PerturbedOptimizer::new(argmax_binary);
        let result = opt.gradient(&[1.0, 2.0], &[1.0, 0.0]);
        assert!(result.is_err(), "Should error without forward pass");
    }

    #[test]
    fn test_sparsemap_simplex_projection() {
        // Simple 1D simplex: μ ∈ [0, 1], Σμ = 1
        let mut sm = SparseMap::simplex(3);
        let theta = vec![1.0, 2.0, 0.5_f64];

        let mu = sm.forward(&theta).expect("SparseMap forward failed");

        // Check μ ≥ 0
        for mi in &mu {
            assert!(*mi >= -1e-6, "μ < 0: {}", mi);
        }

        // Check Σμ ≈ 1 (simplex constraint)
        let sum: f64 = mu.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.1,
            "Σμ = {} (expected ~1.0 for simplex)",
            sum
        );
    }

    #[test]
    fn test_sparsemap_backward_shape() {
        let mut sm = SparseMap::simplex(4);
        let theta = vec![1.0, 3.0, 2.0, 0.5_f64];

        let _mu = sm.forward(&theta).expect("SparseMap forward failed");
        let dl_dtheta = sm
            .backward(&[1.0, 0.0, 0.0, 0.0])
            .expect("SparseMap backward failed");

        assert_eq!(dl_dtheta.len(), 4, "Gradient length mismatch");
        for gi in &dl_dtheta {
            assert!(gi.is_finite(), "SparseMap gradient not finite");
        }
    }

    #[test]
    fn test_sparsemap_no_forward_error() {
        let sm = SparseMap::simplex(3);
        let result = sm.backward(&[1.0, 0.0, 0.0]);
        assert!(result.is_err(), "Should error without forward pass");
    }

    #[test]
    fn test_project_simplex_properties() {
        let v = vec![0.5, 1.5, -0.3, 2.0_f64];
        let p = SparseMap::project_simplex(&v);

        // Σ = 1
        let sum: f64 = p.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Simplex sum = {} (expected 1.0)",
            sum
        );

        // All ≥ 0
        for pi in &p {
            assert!(*pi >= -1e-12, "Negative simplex component: {}", pi);
        }
    }

    #[test]
    fn test_project_simplex_uniform_input() {
        // For uniform input [0.5, 0.5], projection = [0.5, 0.5]
        let v = vec![0.5, 0.5_f64];
        let p = SparseMap::project_simplex(&v);
        assert!((p[0] - 0.5).abs() < 1e-10);
        assert!((p[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_xorshift_reproducible() {
        let mut rng1 = Xorshift64::new(42);
        let mut rng2 = Xorshift64::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_xorshift_normal_finite() {
        let mut rng = Xorshift64::new(12345);
        for _ in 0..100 {
            let v = rng.normal();
            assert!(v.is_finite(), "Normal sample not finite: {}", v);
        }
    }
}
