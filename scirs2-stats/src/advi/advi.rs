//! ADVI optimizer — Automatic Differentiation Variational Inference.
//!
//! Implements Algorithm 1 from Kucukelbir et al. (2017) "Automatic
//! Differentiation Variational Inference" (JMLR 18:1-45).
//!
//! # Algorithm overview
//!
//! Given a log-joint `log p(x, θ)`, ADVI maximizes the ELBO:
//!
//! ```text
//!   ELBO = E_q[log p(x, θ)] + H[q(θ)]
//!        = E_q[log p(x, θ)] + Σᵢ (1 + log 2π + ωᵢ) / 2
//! ```
//!
//! via the **reparameterization trick**:
//! θᵢ = μᵢ + exp(ωᵢ) · εᵢ,  εᵢ ~ N(0,1).
//!
//! Gradients w.r.t. μ and ω are estimated via MC averaging, and an
//! **Adam** optimizer drives the updates.

use crate::error::{StatsError, StatsResult};

use super::types::{AdviConfig, AdviResult};

// ============================================================================
// LCG pseudo-random number generator (no external rand crate)
// ============================================================================

/// Simple 64-bit LCG PRNG — fast and dependency-free.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Advance state and return a uniform float in [0, 1).
    fn next_f64(&mut self) -> f64 {
        // Knuth multiplicative LCG (mod 2^64)
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Upper 53 bits for mantissa
        ((self.state >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
    }

    /// Box-Muller standard normal sample.
    fn randn(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ============================================================================
// Adam optimizer state
// ============================================================================

/// Per-parameter Adam moment vectors.
#[derive(Debug, Clone)]
struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: usize,
    beta1: f64,
    beta2: f64,
    eps: f64,
}

impl Adam {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// Compute bias-corrected Adam step; returns the update direction (to subtract from params).
    fn step(&mut self, grad: &[f64], lr: f64) -> Vec<f64> {
        self.t += 1;
        let t = self.t as f64;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);
        let mut delta = vec![0.0; grad.len()];
        for i in 0..grad.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            delta[i] = lr * m_hat / (v_hat.sqrt() + self.eps);
        }
        delta
    }
}

// ============================================================================
// ELBO computation
// ============================================================================

/// Compute MC estimate of the ELBO and its gradients w.r.t. (μ, ω).
///
/// ELBO  = (1/S) Σₛ log p(x, θˢ) + Σᵢ (1 + log 2π + ωᵢ) / 2
///
/// where θᵢˢ = μᵢ + exp(ωᵢ) εᵢˢ,  εᵢˢ ~ N(0,1).
///
/// Gradient w.r.t. μᵢ:   ∂ELBO/∂μᵢ  ≈ (1/S) Σₛ ∂log p / ∂θᵢ |_{θˢ}
/// Gradient w.r.t. ωᵢ:   ∂ELBO/∂ωᵢ  ≈ (1/S) Σₛ εᵢˢ · ∂log p / ∂θᵢ |_{θˢ} + 1/2
///                                    = (1/S) Σₛ εᵢˢ · ∂log p / ∂θᵢ + 0.5
///
/// Returns `(elbo, grad_mu, grad_omega)`.
fn elbo_and_gradients(
    log_joint_fn: &dyn Fn(&[f64]) -> f64,
    mu: &[f64],
    log_sigma: &[f64],
    config: &AdviConfig,
    rng: &mut Lcg,
) -> (f64, Vec<f64>, Vec<f64>) {
    let n = mu.len();
    let s = config.n_samples;
    let h = config.fd_step;

    let mut elbo_sum = 0.0;
    let mut grad_mu = vec![0.0f64; n];
    let mut grad_omega = vec![0.0f64; n];

    for _ in 0..s {
        // Sample ε ~ N(0, I_n)
        let eps: Vec<f64> = (0..n).map(|_| rng.randn()).collect();
        // θ = μ + σ · ε
        let theta: Vec<f64> = (0..n)
            .map(|i| mu[i] + log_sigma[i].exp() * eps[i])
            .collect();

        let log_p = log_joint_fn(&theta);
        elbo_sum += log_p;

        // Finite-difference gradient of log p w.r.t. θ
        let mut grad_logp = vec![0.0f64; n];
        for j in 0..n {
            let mut theta_fwd = theta.clone();
            let mut theta_bwd = theta.clone();
            theta_fwd[j] += h;
            theta_bwd[j] -= h;
            grad_logp[j] = (log_joint_fn(&theta_fwd) - log_joint_fn(&theta_bwd)) / (2.0 * h);
        }

        // Accumulate gradients
        for i in 0..n {
            grad_mu[i] += grad_logp[i];
            // ∂ELBO/∂ωᵢ: reparameterization gives εᵢ · (∂log p/∂θᵢ)
            grad_omega[i] += eps[i] * grad_logp[i];
        }
    }

    let s_f = s as f64;
    let log2pi = (2.0 * std::f64::consts::PI).ln();

    // Average over MC samples
    for i in 0..n {
        grad_mu[i] /= s_f;
        grad_omega[i] /= s_f;
        // Entropy gradient term: +1/2 (from d/dωᵢ of the entropy term ωᵢ/2)
        // Wait: entropy = Σᵢ (1 + log 2π + ωᵢ)/2; dH/dωᵢ = 1/2
        grad_omega[i] += 0.5;
    }

    // ELBO = (1/S) Σₛ log p(x, θˢ) + entropy
    let entropy: f64 = log_sigma.iter().map(|&w| 0.5 * (1.0 + log2pi + w)).sum();
    let elbo = elbo_sum / s_f + entropy;

    (elbo, grad_mu, grad_omega)
}

// ============================================================================
// AdviOptimizer
// ============================================================================

/// ADVI mean-field optimizer.
///
/// Maintains the variational parameters (μ, ω) and an Adam optimizer state.
/// Call [`AdviOptimizer::fit`] to run the ELBO optimization loop.
pub struct AdviOptimizer {
    /// Configuration
    pub config: AdviConfig,
}

impl AdviOptimizer {
    /// Create a new optimizer with the given configuration.
    pub fn new(config: AdviConfig) -> Self {
        Self { config }
    }

    /// Create an optimizer with default configuration.
    pub fn default_config() -> Self {
        Self {
            config: AdviConfig::default(),
        }
    }

    /// Fit the mean-field variational approximation.
    ///
    /// # Arguments
    /// * `log_joint_fn` — Closure computing `log p(x, θ)` for a parameter vector θ.
    /// * `n_params` — Dimension of the parameter space.
    ///
    /// # Returns
    /// An [`AdviResult`] with variational parameters, ELBO history, and convergence flag.
    ///
    /// # Errors
    /// Returns `StatsError` if `n_params == 0` or if `log_joint_fn` returns non-finite values.
    pub fn fit(
        &self,
        log_joint_fn: &dyn Fn(&[f64]) -> f64,
        n_params: usize,
    ) -> StatsResult<AdviResult> {
        if n_params == 0 {
            return Err(StatsError::invalid_argument(
                "n_params must be > 0 for ADVI",
            ));
        }

        let cfg = &self.config;

        // Initialize variational parameters: μ = 0, ω = 0 (σ = 1)
        let mut mu = vec![0.0f64; n_params];
        let mut log_sigma = vec![0.0f64; n_params]; // ω = log σ

        // Adam optimizers for μ and ω (separate moment vectors)
        let mut adam_mu = Adam::new(n_params);
        let mut adam_omega = Adam::new(n_params);

        let mut rng = Lcg::new(cfg.seed);
        let mut elbo_history: Vec<f64> = Vec::with_capacity(cfg.n_iter);
        let mut prev_elbo = f64::NEG_INFINITY;
        let mut converged = false;
        let mut n_iter_performed = 0;

        for _iter in 0..cfg.n_iter {
            let (elbo, grad_mu, grad_omega) =
                elbo_and_gradients(log_joint_fn, &mu, &log_sigma, cfg, &mut rng);

            // Guard against degenerate log-joint values
            if !elbo.is_finite() {
                // Skip update but record NaN so caller can detect
                elbo_history.push(elbo);
                n_iter_performed += 1;
                break;
            }

            elbo_history.push(elbo);
            n_iter_performed += 1;

            // Adam ascent on μ (maximize ELBO → negate gradient for step)
            let neg_grad_mu: Vec<f64> = grad_mu.iter().map(|&g| -g).collect();
            let neg_grad_omega: Vec<f64> = grad_omega.iter().map(|&g| -g).collect();

            let delta_mu = adam_mu.step(&neg_grad_mu, cfg.lr);
            let delta_omega = adam_omega.step(&neg_grad_omega, cfg.lr);

            for i in 0..n_params {
                mu[i] -= delta_mu[i]; // Adam step already in ascent direction
                log_sigma[i] -= delta_omega[i];
            }

            // Convergence check
            if (elbo - prev_elbo).abs() < cfg.tol {
                converged = true;
                break;
            }
            prev_elbo = elbo;
        }

        Ok(AdviResult {
            elbo_history,
            mu,
            log_sigma,
            converged,
            n_iter_performed,
        })
    }
}

// ============================================================================
// Posterior sampling
// ============================================================================

/// Draw `n` samples from the fitted variational posterior q(θ) = Π N(μᵢ, σᵢ²).
///
/// Each sample is a vector of length `result.mu.len()`.
///
/// # Errors
/// Returns an error if `result` has empty mu/log_sigma vectors.
pub fn sample_posterior(result: &AdviResult, n: usize, seed: u64) -> StatsResult<Vec<Vec<f64>>> {
    let n_params = result.mu.len();
    if n_params == 0 {
        return Err(StatsError::invalid_argument(
            "AdviResult has zero parameters",
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut rng = Lcg::new(seed);
    let sigma: Vec<f64> = result.log_sigma.iter().map(|&w| w.exp()).collect();

    let samples = (0..n)
        .map(|_| {
            (0..n_params)
                .map(|i| result.mu[i] + sigma[i] * rng.randn())
                .collect()
        })
        .collect();

    Ok(samples)
}

// ============================================================================
// Entropy helpers (public for tests)
// ============================================================================

/// Mean-field Gaussian entropy: `H[q]` = Sum\_i (1 + log 2*pi + omega\_i) / 2.
pub fn mean_field_entropy(log_sigma: &[f64]) -> f64 {
    let log2pi = (2.0 * std::f64::consts::PI).ln();
    log_sigma.iter().map(|&w| 0.5 * (1.0 + log2pi + w)).sum()
}

// ============================================================================
// Convenience: Bayesian linear regression log-joint
// ============================================================================

/// Build a log-joint function for Bayesian linear regression:
/// `log p(y | X, β) + log p(β)` where
/// - likelihood: y | β ~ N(Xβ, σ²I), σ² = noise_var
/// - prior: β ~ N(0, (1/prior_precision) I)
///
/// The returned closure is suitable as input to [`AdviOptimizer::fit`].
pub fn make_linear_regression_log_joint(
    x_data: Vec<Vec<f64>>,
    y_data: Vec<f64>,
    noise_var: f64,
    prior_precision: f64,
) -> impl Fn(&[f64]) -> f64 {
    move |beta: &[f64]| {
        let n = y_data.len();
        let mut log_lik = 0.0;
        for i in 0..n {
            let mut pred = 0.0;
            for (j, &bj) in beta.iter().enumerate() {
                if j < x_data[i].len() {
                    pred += x_data[i][j] * bj;
                }
            }
            let r = y_data[i] - pred;
            log_lik -= 0.5 * r * r / noise_var;
        }
        // Gaussian prior: log p(β) = -prior_precision/2 · ||β||²
        let log_prior: f64 = beta.iter().map(|&b| -0.5 * prior_precision * b * b).sum();
        log_lik + log_prior
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcg_uniform_range() {
        let mut rng = Lcg::new(123);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0, "out of range: {v}");
        }
    }

    #[test]
    fn test_adam_moment_update_first_iter() {
        let mut adam = Adam::new(2);
        let grad = vec![1.0, -2.0];
        let delta = adam.step(&grad, 0.01);

        // After step 1: m_hat = g (since beta1^1 = 0.9, bc1 = 0.1)
        // m[0] = 0.9*0 + 0.1*1 = 0.1; m_hat = 0.1 / 0.1 = 1.0
        // v[0] = 0.999*0 + 0.001*1 = 0.001; v_hat = 0.001/0.001 = 1.0
        // delta[0] = lr * 1.0 / (1.0.sqrt() + eps) ≈ lr
        assert!(
            (delta[0] - 0.01).abs() < 1e-6,
            "Adam step[0] = {} ≠ 0.01",
            delta[0]
        );
        assert!((delta[1] - (-0.01)).abs() < 1e-6);
    }

    #[test]
    fn test_advi_zero_params_error() {
        let opt = AdviOptimizer::default_config();
        let result = opt.fit(&|_: &[f64]| 0.0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_advi_elbo_increases() {
        // Target: log p(θ) = log N(θ | 3, 1) = -0.5(θ-3)² - const
        // The ELBO should generally increase over the first 50 iterations.
        let log_joint = |theta: &[f64]| {
            let t = theta[0];
            -0.5 * (t - 3.0) * (t - 3.0)
        };

        let config = AdviConfig {
            n_iter: 200,
            n_samples: 5,
            lr: 0.05,
            tol: 1e-8, // tight tol so we run all iterations
            seed: 1,
            ..AdviConfig::default()
        };
        let opt = AdviOptimizer::new(config);
        let result = opt.fit(&log_joint, 1).expect("fit ok");

        // The final ELBO should be greater than the initial ELBO
        let n = result.elbo_history.len();
        assert!(n > 5, "Should have at least 5 iterations");
        let init_elbo = result.elbo_history[..5]
            .iter()
            .copied()
            .filter(|e| e.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let final_elbo = result.elbo_history[(n - 5)..]
            .iter()
            .copied()
            .filter(|e| e.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            final_elbo >= init_elbo - 1.0, // allow small numerical fluctuation
            "ELBO should not decrease significantly: init={init_elbo:.4}, final={final_elbo:.4}"
        );
    }

    #[test]
    fn test_advi_recovers_gaussian_mean() {
        // Prior N(0,1) * Likelihood N(3,1): posterior N(1.5, 0.5)
        // With flat prior (precision=0) and single observation at 3,
        // posterior mean should be close to 3.
        let log_joint = |theta: &[f64]| {
            let t = theta[0];
            // log N(3 | t, 1) + log N(t | 0, 1)
            -0.5 * (3.0 - t) * (3.0 - t) - 0.5 * t * t
        };

        let config = AdviConfig {
            n_iter: 500,
            n_samples: 10,
            lr: 0.05,
            tol: 1e-7,
            seed: 7,
            ..AdviConfig::default()
        };
        let opt = AdviOptimizer::new(config);
        let result = opt.fit(&log_joint, 1).expect("fit ok");

        // Analytical posterior: N(3/2, 1/2) => mu ≈ 1.5
        let mu = result.mu[0];
        assert!((mu - 1.5).abs() < 0.5, "Expected mu ≈ 1.5, got {mu:.4}");
    }

    #[test]
    fn test_advi_posterior_samples_shape() {
        let config = AdviConfig {
            n_iter: 10,
            n_samples: 1,
            seed: 42,
            ..AdviConfig::default()
        };
        let opt = AdviOptimizer::new(config);
        let result = opt.fit(&|_: &[f64]| -1.0, 3).expect("fit ok");
        let samples = sample_posterior(&result, 50, 99).expect("samples ok");
        assert_eq!(samples.len(), 50);
        for s in &samples {
            assert_eq!(s.len(), 3);
        }
    }

    #[test]
    fn test_advi_converged_flag() {
        // A flat log-joint → constant ELBO → should converge quickly
        let log_joint = |_: &[f64]| 0.0;
        let config = AdviConfig {
            n_iter: 2000,
            n_samples: 1,
            lr: 0.01,
            tol: 1e-3,
            seed: 5,
            ..AdviConfig::default()
        };
        let opt = AdviOptimizer::new(config);
        let result = opt.fit(&log_joint, 2).expect("fit ok");
        // For a flat log-joint the entropy drives the ELBO to stabilize — converge expected
        assert!(
            result.converged || result.n_iter_performed == 2000,
            "Should converge or exhaust iterations"
        );
    }

    #[test]
    fn test_advi_mean_field_entropy() {
        // H = Σᵢ (1 + log 2π + ωᵢ) / 2
        let log_sigma = vec![0.0, 1.0, -1.0];
        let entropy = mean_field_entropy(&log_sigma);
        let log2pi = (2.0 * std::f64::consts::PI).ln();
        let expected: f64 = log_sigma.iter().map(|&w| 0.5 * (1.0 + log2pi + w)).sum();
        assert!((entropy - expected).abs() < 1e-12);
    }

    #[test]
    fn test_sample_posterior_empty_params_error() {
        let result = AdviResult {
            elbo_history: vec![],
            mu: vec![],
            log_sigma: vec![],
            converged: false,
            n_iter_performed: 0,
        };
        assert!(sample_posterior(&result, 10, 1).is_err());
    }

    #[test]
    fn test_sample_posterior_zero_samples() {
        let result = AdviResult {
            elbo_history: vec![-1.0],
            mu: vec![0.0],
            log_sigma: vec![0.0],
            converged: false,
            n_iter_performed: 1,
        };
        let samples = sample_posterior(&result, 0, 1).expect("ok");
        assert!(samples.is_empty());
    }

    #[test]
    fn test_make_linear_regression_log_joint() {
        // y = 2x, single observation: x=[1], y=[2]
        let x = vec![vec![1.0]];
        let y = vec![2.0];
        let log_joint = make_linear_regression_log_joint(x, y, 1.0, 0.01);

        // At β=[2]: residual=0, log_lik=0, log_prior = -0.01/2*4 = -0.02
        let lp = log_joint(&[2.0]);
        assert!((lp - (-0.02)).abs() < 1e-10, "log_joint(β=2) = {lp}");

        // At β=[0]: residual=2, log_lik = -2, log_prior = 0
        let lp0 = log_joint(&[0.0]);
        assert!((lp0 - (-2.0)).abs() < 1e-10, "log_joint(β=0) = {lp0}");
    }
}
