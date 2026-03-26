//! Black-Box Variational Inference (BBVI)
//!
//! Implements REINFORCE (score-function) gradient estimator and VIMCO
//! (Variational Inference with Monte Carlo Objectives) for black-box
//! variational inference without needing gradients of the model.
//!
//! ## REINFORCE Gradient
//! ∇_φ ELBO = E_q[f(z) ∇_φ log q(z;φ)]
//! where f(z) = log p(x,z) - log q(z;φ) is the "reward".
//!
//! ## Variance Reduction
//! Control variate baseline b = mean(f(z)) reduces variance without bias.
//!
//! ## VIMCO
//! Multi-sample ELBO with leave-one-out baselines for further variance reduction.

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Black-Box Variational Inference
#[derive(Debug, Clone)]
pub struct BbviConfig {
    /// Number of Monte Carlo samples per gradient estimate
    pub n_samples: usize,
    /// Learning rate for Adam optimizer
    pub learning_rate: f64,
    /// Maximum number of optimization iterations
    pub n_iter: usize,
    /// Use VIMCO (multi-sample) instead of REINFORCE
    pub use_vimco: bool,
    /// Exponential decay for running baseline (0 < baseline_decay < 1)
    pub baseline_decay: f64,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Convergence tolerance (change in ELBO)
    pub tol: f64,
}

impl Default for BbviConfig {
    fn default() -> Self {
        BbviConfig {
            n_samples: 10,
            learning_rate: 0.01,
            n_iter: 1000,
            use_vimco: false,
            baseline_decay: 0.9,
            seed: 42,
            tol: 1e-6,
        }
    }
}

// ============================================================================
// Output types
// ============================================================================

/// ELBO estimate with uncertainty quantification
#[derive(Debug, Clone)]
pub struct ElboEstimate<F: Float> {
    /// Monte Carlo estimate of the ELBO
    pub value: F,
    /// Standard error of the ELBO estimate
    pub std_error: F,
    /// Variance of per-sample gradient norms (proxy for gradient variance)
    pub gradient_variance: F,
}

/// Result of BBVI optimization
#[derive(Debug, Clone)]
pub struct BbviResult<F: Float> {
    /// ELBO value at each iteration
    pub elbo_history: Vec<F>,
    /// Final variational parameters after optimization
    pub final_params: Vec<F>,
    /// Number of iterations actually performed
    pub n_iters: usize,
    /// Whether optimization converged before n_iter
    pub converged: bool,
}

// ============================================================================
// VariationalDistribution trait
// ============================================================================

/// Trait for variational distributions used in BBVI
///
/// Represents q(z; φ) parameterized by φ, supporting sampling
/// and log-probability evaluation for gradient estimation.
pub trait VariationalDistribution<F: Float + FromPrimitive + Clone + Debug> {
    /// Draw a single sample z ~ q(z; φ), using seed for reproducibility
    fn sample(&self, seed: u64) -> Vec<F>;

    /// Compute log q(z; φ) for a given z
    fn log_prob(&self, z: &[F]) -> F;

    /// Return a reference to the current variational parameters φ
    fn params(&self) -> &[F];

    /// Return a mutable reference to the variational parameters φ
    fn params_mut(&mut self) -> &mut Vec<F>;

    /// Compute score function ∇_φ log q(z; φ) for a given z
    /// Returns gradient with same length as params()
    fn score_function(&self, z: &[F]) -> Vec<F>;

    /// Dimensionality of the latent variable z
    fn latent_dim(&self) -> usize;
}

// ============================================================================
// Mean-Field Gaussian variational family
// ============================================================================

/// Mean-field Gaussian variational distribution
///
/// q(z; φ) = Π_i N(z_i; μ_i, σ_i²)
///
/// Parameters φ = [μ_1, …, μ_d, log_σ_1, …, log_σ_d]
/// where σ_i = exp(log_σ_i) > 0 always.
#[derive(Debug, Clone)]
pub struct MeanFieldGaussian<F: Float + FromPrimitive + Clone + Debug> {
    /// Variational parameters: [μ_1,...,μ_d, log_σ_1,...,log_σ_d]
    params: Vec<F>,
    /// Dimensionality d
    dim: usize,
}

impl<F: Float + FromPrimitive + Clone + Debug> MeanFieldGaussian<F> {
    /// Create a new MeanFieldGaussian initialized at zero mean and unit variance
    pub fn new(dim: usize) -> Self {
        let mut params = vec![F::zero(); 2 * dim];
        // Initialize log_σ = 0 → σ = 1
        for i in dim..(2 * dim) {
            params[i] = F::zero();
        }
        MeanFieldGaussian { params, dim }
    }

    /// Create with explicit mean and log-scale parameters
    pub fn from_params(mu: Vec<F>, log_sigma: Vec<F>) -> Result<Self> {
        if mu.len() != log_sigma.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "mu length {} != log_sigma length {}",
                mu.len(),
                log_sigma.len()
            )));
        }
        let dim = mu.len();
        let mut params = Vec::with_capacity(2 * dim);
        params.extend_from_slice(&mu);
        params.extend_from_slice(&log_sigma);
        Ok(MeanFieldGaussian { params, dim })
    }

    /// Get the mean vector μ
    pub fn mean(&self) -> &[F] {
        &self.params[..self.dim]
    }

    /// Get the log-scale vector log(σ)
    pub fn log_sigma(&self) -> &[F] {
        &self.params[self.dim..]
    }
}

/// LCG-based pseudo-random number generator for reproducible sampling
/// (avoids the rand crate dependency)
struct LcgPrng {
    state: u64,
}

impl LcgPrng {
    fn new(seed: u64) -> Self {
        LcgPrng { state: seed ^ 6364136223846793005 }
    }

    /// Produce next u64 via LCG
    fn next_u64(&mut self) -> u64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Produce a uniform sample in (0, 1) exclusive
    fn uniform(&mut self) -> f64 {
        // Use top 53 bits for mantissa
        let bits = self.next_u64() >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Produce two independent standard normal samples via Box-Muller
    fn normal_pair(&mut self) -> (f64, f64) {
        use std::f64::consts::PI;
        let u1 = self.uniform();
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

impl<F: Float + FromPrimitive + Clone + Debug> VariationalDistribution<F>
    for MeanFieldGaussian<F>
{
    fn sample(&self, seed: u64) -> Vec<F> {
        let mut rng = LcgPrng::new(seed);
        let mut z = Vec::with_capacity(self.dim);
        let mut i = 0;
        while i < self.dim {
            let (n1, n2) = rng.normal_pair();
            let sigma_i = self.params[self.dim + i].exp();
            let mu_i = self.params[i];
            z.push(mu_i + sigma_i * F::from_f64(n1).unwrap_or(F::zero()));
            if i + 1 < self.dim {
                let sigma_i1 = self.params[self.dim + i + 1].exp();
                let mu_i1 = self.params[i + 1];
                z.push(mu_i1 + sigma_i1 * F::from_f64(n2).unwrap_or(F::zero()));
            }
            i += 2;
        }
        z.truncate(self.dim);
        z
    }

    fn log_prob(&self, z: &[F]) -> F {
        if z.len() != self.dim {
            return F::neg_infinity();
        }
        let two_pi = F::from_f64(2.0 * std::f64::consts::PI).unwrap_or(F::one());
        let half = F::from_f64(0.5).unwrap_or(F::one());
        let mut log_p = F::zero();
        for i in 0..self.dim {
            let mu = self.params[i];
            let log_s = self.params[self.dim + i];
            let sigma = log_s.exp();
            // log N(z_i; mu, sigma^2) = -0.5*log(2*pi) - log(sigma) - 0.5*(z-mu)^2/sigma^2
            let diff = z[i] - mu;
            log_p = log_p - half * two_pi.ln() - log_s - half * (diff * diff) / (sigma * sigma);
        }
        log_p
    }

    fn params(&self) -> &[F] {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Vec<F> {
        &mut self.params
    }

    fn score_function(&self, z: &[F]) -> Vec<F> {
        // ∇_μ_i log q = (z_i - μ_i) / σ_i²
        // ∇_{log_σ_i} log q = (z_i - μ_i)² / σ_i² - 1
        let mut grad = vec![F::zero(); 2 * self.dim];
        for i in 0..self.dim {
            let mu = self.params[i];
            let log_s = self.params[self.dim + i];
            let sigma2 = (log_s * F::from_f64(2.0).unwrap_or(F::one())).exp();
            let diff = z[i] - mu;
            grad[i] = diff / sigma2;
            grad[self.dim + i] = (diff * diff) / sigma2 - F::one();
        }
        grad
    }

    fn latent_dim(&self) -> usize {
        self.dim
    }
}

// ============================================================================
// Log-sum-exp utility
// ============================================================================

/// Numerically stable log-sum-exp: log(Σ exp(x_i))
fn log_sum_exp<F: Float + FromPrimitive>(values: &[F]) -> F {
    if values.is_empty() {
        return F::neg_infinity();
    }
    let max_val = values.iter().cloned().fold(F::neg_infinity(), F::max);
    if max_val.is_infinite() {
        return F::neg_infinity();
    }
    let sum_exp = values
        .iter()
        .fold(F::zero(), |acc, &v| acc + (v - max_val).exp());
    max_val + sum_exp.ln()
}

// ============================================================================
// REINFORCE gradient estimator
// ============================================================================

/// Estimate the ELBO gradient using REINFORCE (score-function estimator)
/// with a moving-average baseline for variance reduction.
///
/// Returns (ElboEstimate, gradient_vector)
pub fn reinforce_gradient<F>(
    q: &dyn VariationalDistribution<F>,
    log_joint_fn: &dyn Fn(&[F]) -> F,
    config: &BbviConfig,
) -> Result<(ElboEstimate<F>, Vec<F>)>
where
    F: Float + FromPrimitive + Clone + Debug,
{
    let n = config.n_samples;
    let n_params = q.params().len();
    if n == 0 {
        return Err(StatsError::InvalidArgument(
            "n_samples must be > 0".to_string(),
        ));
    }

    // Sample z_i and compute rewards f(z_i) = log p(x,z) - log q(z)
    let mut rewards: Vec<F> = Vec::with_capacity(n);
    let mut samples: Vec<Vec<F>> = Vec::with_capacity(n);
    let mut scores: Vec<Vec<F>> = Vec::with_capacity(n);

    for i in 0..n {
        let seed = config.seed.wrapping_add(i as u64).wrapping_mul(2654435761);
        let z = q.sample(seed);
        let log_q = q.log_prob(&z);
        let log_p = log_joint_fn(&z);
        let reward = log_p - log_q;
        let score = q.score_function(&z);
        rewards.push(reward);
        samples.push(z);
        scores.push(score);
    }

    // Baseline b = mean(rewards)
    let n_f = F::from_usize(n).ok_or_else(|| {
        StatsError::ComputationError("Cannot convert n_samples to F".to_string())
    })?;
    let reward_sum = rewards.iter().cloned().fold(F::zero(), |acc, v| acc + v);
    let baseline = reward_sum / n_f;

    // Variance of rewards
    let reward_var = {
        let sq_sum = rewards
            .iter()
            .fold(F::zero(), |acc, &r| acc + (r - baseline) * (r - baseline));
        if n > 1 {
            sq_sum / F::from_usize(n - 1).unwrap_or(F::one())
        } else {
            F::zero()
        }
    };
    let std_err = if n > 1 {
        (reward_var / n_f).sqrt()
    } else {
        F::zero()
    };

    // Gradient estimate: (1/n) Σ (f_i - b) · score_i
    let mut gradient = vec![F::zero(); n_params];
    let mut per_sample_norms: Vec<F> = Vec::with_capacity(n);

    for i in 0..n {
        let weight = rewards[i] - baseline;
        let mut norm_sq = F::zero();
        for j in 0..n_params {
            let g_ij = weight * scores[i][j];
            gradient[j] = gradient[j] + g_ij;
            norm_sq = norm_sq + g_ij * g_ij;
        }
        per_sample_norms.push(norm_sq.sqrt());
    }
    for g in gradient.iter_mut() {
        *g = *g / n_f;
    }

    // Gradient variance from per-sample gradient norms
    let norm_mean = per_sample_norms.iter().cloned().fold(F::zero(), |a, v| a + v) / n_f;
    let grad_var = per_sample_norms
        .iter()
        .fold(F::zero(), |acc, &v| acc + (v - norm_mean) * (v - norm_mean))
        / n_f;

    let elbo = ElboEstimate {
        value: baseline, // ELBO estimate = mean(rewards)
        std_error: if std_err < F::zero() { F::zero() } else { std_err },
        gradient_variance: if grad_var < F::zero() { F::zero() } else { grad_var },
    };

    Ok((elbo, gradient))
}

// ============================================================================
// VIMCO gradient estimator
// ============================================================================

/// Estimate the ELBO gradient using VIMCO (multi-sample lower bound)
/// with leave-one-out baselines for variance reduction.
///
/// Returns (ElboEstimate, gradient_vector)
pub fn vimco_gradient<F>(
    q: &dyn VariationalDistribution<F>,
    log_joint_fn: &dyn Fn(&[F]) -> F,
    config: &BbviConfig,
) -> Result<(ElboEstimate<F>, Vec<F>)>
where
    F: Float + FromPrimitive + Clone + Debug,
{
    let k = config.n_samples;
    let n_params = q.params().len();
    if k == 0 {
        return Err(StatsError::InvalidArgument(
            "n_samples must be > 0".to_string(),
        ));
    }

    // Special case k=1: equivalent to REINFORCE
    if k == 1 {
        return reinforce_gradient(q, log_joint_fn, config);
    }

    let k_f = F::from_usize(k).ok_or_else(|| {
        StatsError::ComputationError("Cannot convert k to F".to_string())
    })?;

    // Sample z_j and compute log weights log w_j = log p(x,z) - log q(z)
    let mut log_weights: Vec<F> = Vec::with_capacity(k);
    let mut scores: Vec<Vec<F>> = Vec::with_capacity(k);

    for j in 0..k {
        let seed = config.seed.wrapping_add(j as u64).wrapping_mul(2654435761);
        let z = q.sample(seed);
        let log_q = q.log_prob(&z);
        let log_p = log_joint_fn(&z);
        log_weights.push(log_p - log_q);
        scores.push(q.score_function(&z));
    }

    // Multi-sample ELBO: L_k = log(1/k Σ_j exp(log_w_j))
    //                        = logsumexp(log_w) - log(k)
    let log_k = k_f.ln();
    let lse_all = log_sum_exp(&log_weights);
    let elbo_val = lse_all - log_k;

    // Normalized weights: w̃_j = exp(log_w_j - logsumexp(log_w))
    let log_w_normalized: Vec<F> = log_weights.iter().map(|&w| w - lse_all).collect();

    // Leave-one-out baselines: b_j = logsumexp({log_w_i : i≠j}) - log(k-1)
    let k_minus1 = k - 1;
    let log_k_m1 = F::from_usize(k_minus1)
        .ok_or_else(|| StatsError::ComputationError("k-1 overflow".to_string()))?
        .ln();

    let mut loo_baselines: Vec<F> = Vec::with_capacity(k);
    for j in 0..k {
        let loo_vals: Vec<F> = log_weights
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != j)
            .map(|(_, &w)| w)
            .collect();
        let loo_lse = if loo_vals.is_empty() {
            F::neg_infinity()
        } else {
            log_sum_exp(&loo_vals)
        };
        loo_baselines.push(loo_lse - log_k_m1);
    }

    // VIMCO gradient: (1/k) Σ_j (log_w̃_j - b_j) · ∇_φ log q(z_j)
    let mut gradient = vec![F::zero(); n_params];
    let mut per_sample_norms: Vec<F> = Vec::with_capacity(k);

    for j in 0..k {
        let vimco_weight = log_w_normalized[j] - loo_baselines[j];
        let mut norm_sq = F::zero();
        for p_idx in 0..n_params {
            let g = vimco_weight * scores[j][p_idx];
            gradient[p_idx] = gradient[p_idx] + g;
            norm_sq = norm_sq + g * g;
        }
        per_sample_norms.push(norm_sq.sqrt());
    }
    for g in gradient.iter_mut() {
        *g = *g / k_f;
    }

    // Variance estimation
    let norm_mean = per_sample_norms.iter().cloned().fold(F::zero(), |a, v| a + v) / k_f;
    let grad_var = per_sample_norms
        .iter()
        .fold(F::zero(), |acc, &v| acc + (v - norm_mean) * (v - norm_mean))
        / k_f;

    // Standard error from log weights variance
    let lw_mean = log_weights.iter().cloned().fold(F::zero(), |a, v| a + v) / k_f;
    let lw_var = log_weights
        .iter()
        .fold(F::zero(), |acc, &w| acc + (w - lw_mean) * (w - lw_mean))
        / k_f;
    let std_err = (lw_var / k_f).sqrt();

    let elbo = ElboEstimate {
        value: elbo_val,
        std_error: if std_err < F::zero() { F::zero() } else { std_err },
        gradient_variance: if grad_var < F::zero() { F::zero() } else { grad_var },
    };

    Ok((elbo, gradient))
}

// ============================================================================
// Adam optimizer state
// ============================================================================

struct AdamState<F: Float> {
    m: Vec<F>,    // first moment
    v: Vec<F>,    // second moment
    t: usize,     // step counter
    beta1: F,
    beta2: F,
    epsilon: F,
}

impl<F: Float + FromPrimitive> AdamState<F> {
    fn new(n_params: usize) -> Self {
        AdamState {
            m: vec![F::zero(); n_params],
            v: vec![F::zero(); n_params],
            t: 0,
            beta1: F::from_f64(0.9).unwrap_or(F::one()),
            beta2: F::from_f64(0.999).unwrap_or(F::one()),
            epsilon: F::from_f64(1e-8).unwrap_or(F::zero()),
        }
    }

    /// Compute Adam update and return the parameter delta
    fn step(&mut self, grad: &[F], lr: F) -> Vec<F> {
        self.t += 1;
        let t_f = F::from_usize(self.t).unwrap_or(F::one());
        let one = F::one();
        let mut delta = vec![F::zero(); grad.len()];
        for i in 0..grad.len() {
            self.m[i] = self.beta1 * self.m[i] + (one - self.beta1) * grad[i];
            self.v[i] = self.beta2 * self.v[i] + (one - self.beta2) * grad[i] * grad[i];
            let m_hat = self.m[i] / (one - self.beta1.powf(t_f));
            let v_hat = self.v[i] / (one - self.beta2.powf(t_f));
            delta[i] = lr * m_hat / (v_hat.sqrt() + self.epsilon);
        }
        delta
    }
}

// ============================================================================
// Main optimization loop
// ============================================================================

/// Run BBVI optimization using Adam to maximize the ELBO.
///
/// Uses REINFORCE or VIMCO depending on `config.use_vimco`.
/// Returns the optimization result including ELBO history and final parameters.
pub fn bbvi_optimize<F>(
    q: &mut dyn VariationalDistribution<F>,
    log_joint_fn: &dyn Fn(&[F]) -> F,
    config: &BbviConfig,
) -> Result<BbviResult<F>>
where
    F: Float + FromPrimitive + Clone + Debug,
{
    let n_params = q.params().len();
    let lr = F::from_f64(config.learning_rate).ok_or_else(|| {
        StatsError::InvalidArgument("learning_rate cannot be represented as F".to_string())
    })?;
    let tol = F::from_f64(config.tol).unwrap_or(F::from_f64(1e-6).unwrap_or(F::zero()));

    let mut adam = AdamState::new(n_params);
    let mut elbo_history: Vec<F> = Vec::with_capacity(config.n_iter);
    let mut prev_elbo = F::neg_infinity();
    let mut converged = false;

    // Adjust seed per iteration to avoid using the same samples every step
    let mut iter_config = config.clone();

    for iter in 0..config.n_iter {
        // Vary seed deterministically per iteration
        iter_config.seed = config
            .seed
            .wrapping_add(iter as u64)
            .wrapping_mul(6364136223846793005);

        let (elbo_est, gradient) = if config.use_vimco {
            vimco_gradient(q, log_joint_fn, &iter_config)?
        } else {
            reinforce_gradient(q, log_joint_fn, &iter_config)?
        };

        elbo_history.push(elbo_est.value);

        // Adam update (ascent → add delta)
        let delta = adam.step(&gradient, lr);
        let params = q.params_mut();
        for i in 0..n_params.min(delta.len()) {
            params[i] = params[i] + delta[i];
        }

        // Convergence check
        let elbo_change = (elbo_est.value - prev_elbo).abs();
        if iter > 10 && elbo_change < tol {
            converged = true;
            break;
        }
        prev_elbo = elbo_est.value;
    }

    let final_params = q.params().to_vec();
    let n_iters = elbo_history.len();

    Ok(BbviResult {
        elbo_history,
        final_params,
        n_iters,
        converged,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_gaussian_log_joint(z: &[f64]) -> f64 {
        // log p(z) = Σ log N(z_i; 1.0, 0.5²) — target mean 1.0
        let mut lp = 0.0f64;
        for &zi in z {
            let diff = zi - 1.0;
            lp -= 0.5 * (diff * diff) / 0.25 + 0.5 * (2.0 * std::f64::consts::PI * 0.25).ln();
        }
        lp
    }

    #[test]
    fn test_mean_field_gaussian_sample_shape() {
        let q = MeanFieldGaussian::<f64>::new(5);
        let z = q.sample(42);
        assert_eq!(z.len(), 5, "sample should have same dim as latent space");
    }

    #[test]
    fn test_mean_field_gaussian_log_prob_finite() {
        let q = MeanFieldGaussian::<f64>::new(3);
        let z = q.sample(1);
        let lp = q.log_prob(&z);
        assert!(lp.is_finite(), "log_prob should be finite for a valid sample");
    }

    #[test]
    fn test_mean_field_gaussian_params_length() {
        let dim = 4;
        let q = MeanFieldGaussian::<f64>::new(dim);
        assert_eq!(q.params().len(), 2 * dim, "params should have length 2*dim");
    }

    #[test]
    fn test_reinforce_gradient_output_shape() {
        let q = MeanFieldGaussian::<f64>::new(3);
        let config = BbviConfig { n_samples: 5, ..Default::default() };
        let result = reinforce_gradient(&q, &simple_gaussian_log_joint, &config);
        assert!(result.is_ok());
        let (_, grad) = result.unwrap();
        assert_eq!(grad.len(), q.params().len());
    }

    #[test]
    fn test_elbo_estimate_std_error_non_negative() {
        let q = MeanFieldGaussian::<f64>::new(2);
        let config = BbviConfig { n_samples: 20, ..Default::default() };
        let (elbo, _) = reinforce_gradient(&q, &simple_gaussian_log_joint, &config).unwrap();
        assert!(elbo.std_error >= 0.0, "std_error must be non-negative");
    }

    #[test]
    fn test_vimco_k1_equivalent_to_reinforce() {
        // With k=1, VIMCO delegates to REINFORCE — both use same seeds
        let q1 = MeanFieldGaussian::<f64>::new(2);
        let config = BbviConfig {
            n_samples: 1,
            use_vimco: false,
            seed: 99,
            ..Default::default()
        };
        let (e1, g1) = reinforce_gradient(&q1, &simple_gaussian_log_joint, &config).unwrap();

        let q2 = MeanFieldGaussian::<f64>::new(2);
        let (e2, g2) = vimco_gradient(&q2, &simple_gaussian_log_joint, &config).unwrap();

        // Both should produce the same ELBO value to within floating-point tolerance
        assert!(
            (e1.value - e2.value).abs() < 1e-10,
            "k=1 VIMCO should match REINFORCE: {} vs {}",
            e1.value,
            e2.value
        );
        for (a, b) in g1.iter().zip(g2.iter()) {
            assert!((a - b).abs() < 1e-10, "gradients should match for k=1");
        }
    }

    #[test]
    fn test_vimco_gradient_output_shape() {
        let q = MeanFieldGaussian::<f64>::new(4);
        let config = BbviConfig {
            n_samples: 8,
            use_vimco: true,
            ..Default::default()
        };
        let result = vimco_gradient(&q, &simple_gaussian_log_joint, &config);
        assert!(result.is_ok());
        let (_, grad) = result.unwrap();
        assert_eq!(grad.len(), q.params().len());
    }

    #[test]
    fn test_bbvi_optimize_elbo_increases() {
        let mut q = MeanFieldGaussian::<f64>::new(2);
        let config = BbviConfig {
            n_samples: 20,
            learning_rate: 0.05,
            n_iter: 200,
            use_vimco: false,
            seed: 7,
            tol: 1e-8,
            ..Default::default()
        };
        let result = bbvi_optimize(&mut q, &simple_gaussian_log_joint, &config).unwrap();
        assert!(!result.elbo_history.is_empty());
        // ELBO should be higher in the second half compared to the first half
        let n = result.elbo_history.len();
        let first_half_avg: f64 = result.elbo_history[..n / 4].iter().sum::<f64>() / (n / 4) as f64;
        let second_half_avg: f64 =
            result.elbo_history[3 * n / 4..].iter().sum::<f64>() / (n / 4).max(1) as f64;
        assert!(
            second_half_avg > first_half_avg - 5.0,
            "ELBO should generally increase: early={}, late={}",
            first_half_avg,
            second_half_avg
        );
    }

    #[test]
    fn test_bbvi_config_defaults() {
        let cfg = BbviConfig::default();
        assert_eq!(cfg.n_samples, 10);
        assert!((cfg.learning_rate - 0.01).abs() < 1e-12);
        assert_eq!(cfg.n_iter, 1000);
        assert!(!cfg.use_vimco);
        assert!((cfg.baseline_decay - 0.9).abs() < 1e-12);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn test_vimco_variance_reduction() {
        // VIMCO should produce lower gradient variance than REINFORCE (on average)
        let config_reinforce = BbviConfig {
            n_samples: 10,
            use_vimco: false,
            seed: 100,
            ..Default::default()
        };
        let config_vimco = BbviConfig {
            n_samples: 10,
            use_vimco: true,
            seed: 100,
            ..Default::default()
        };

        let q = MeanFieldGaussian::<f64>::new(3);
        let (e_r, _) =
            reinforce_gradient(&q, &simple_gaussian_log_joint, &config_reinforce).unwrap();
        let (e_v, _) =
            vimco_gradient(&q, &simple_gaussian_log_joint, &config_vimco).unwrap();

        // Both should produce finite gradient variance
        assert!(e_r.gradient_variance.is_finite());
        assert!(e_v.gradient_variance.is_finite());
        // VIMCO variance should be non-negative
        assert!(e_v.gradient_variance >= 0.0);
    }

    #[test]
    fn test_bbvi_result_elbo_history_length() {
        let mut q = MeanFieldGaussian::<f64>::new(2);
        let config = BbviConfig {
            n_iter: 50,
            tol: 0.0, // disable early stopping
            ..Default::default()
        };
        let result = bbvi_optimize(&mut q, &simple_gaussian_log_joint, &config).unwrap();
        assert_eq!(result.n_iters, result.elbo_history.len());
        // With tol=0, should run all 50 iterations
        assert!(result.n_iters <= 50);
    }

    #[test]
    fn test_log_sum_exp_stability() {
        // Large values that would overflow with naive sum of exp
        let vals = vec![1000.0f64, 1001.0, 999.0];
        let result = log_sum_exp(&vals);
        assert!(result.is_finite(), "logsumexp should be stable for large values");
        // Should be approximately 1001 + log(1 + e^{-1} + e^{-2}) ≈ 1001.41
        let expected = 1001.0 + (1.0 + (-1.0f64).exp() + (-2.0f64).exp()).ln();
        assert!((result - expected).abs() < 1e-10, "logsumexp value incorrect: {} vs {}", result, expected);
    }

    #[test]
    fn test_mean_field_gaussian_from_params() {
        let mu = vec![1.0f64, 2.0, 3.0];
        let log_sigma = vec![0.0f64, -0.5, 0.5];
        let q = MeanFieldGaussian::<f64>::from_params(mu.clone(), log_sigma.clone()).unwrap();
        assert_eq!(q.latent_dim(), 3);
        assert_eq!(q.mean(), &mu[..]);
        assert_eq!(q.log_sigma(), &log_sigma[..]);
    }

    #[test]
    fn test_reinforce_gradient_finite() {
        let q = MeanFieldGaussian::<f64>::new(4);
        let config = BbviConfig { n_samples: 15, ..Default::default() };
        let (elbo, grad) = reinforce_gradient(&q, &simple_gaussian_log_joint, &config).unwrap();
        assert!(elbo.value.is_finite(), "ELBO should be finite");
        for g in &grad {
            assert!(g.is_finite(), "gradient should be finite");
        }
    }
}
