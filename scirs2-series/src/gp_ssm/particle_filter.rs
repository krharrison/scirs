//! Bootstrap Particle Filter for GP-State-Space Models
//!
//! Implements a Sequential Importance Resampler (SIR / bootstrap particle
//! filter) where the latent transition `f(x)` is a sparse Gaussian Process
//! with a squared-exponential (RBF) kernel.
//!
//! # Algorithm sketch
//!
//! ```text
//! for t = 1 … T:
//!   1. Propagate particles:  x̃_t^(i) ~ p(x_t | x_{t-1}^(i))   [GP transition]
//!   2. Weight:               w_t^(i) ∝ p(y_t | x̃_t^(i))       [Gaussian likelihood]
//!   3. Normalise weights.
//!   4. If ESS < N/2: systematic resample, reset weights to 1/N.
//! ```

use crate::error::{Result, TimeSeriesError};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the [`GpSsm`] bootstrap particle filter.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct GpSsmConfig {
    /// Number of particles. Default: `100`.
    pub n_particles: usize,
    /// Dimensionality of the latent state. Default: `2`.
    pub latent_dim: usize,
    /// Dimensionality of the observations. Default: `1`.
    pub obs_dim: usize,
    /// Process noise standard deviation for the GP transition. Default: `0.1`.
    pub transition_noise: f64,
    /// Observation noise standard deviation. Default: `0.1`.
    pub obs_noise: f64,
    /// RBF kernel lengthscale `l`. Default: `1.0`.
    pub kernel_lengthscale: f64,
    /// RBF kernel variance `σ²`. Default: `1.0`.
    pub kernel_variance: f64,
    /// Jitter added to the GP kernel matrix diagonal for numerical stability.
    /// Default: `1e-6`.
    pub jitter: f64,
}

impl Default for GpSsmConfig {
    fn default() -> Self {
        Self {
            n_particles: 100,
            latent_dim: 2,
            obs_dim: 1,
            transition_noise: 0.1,
            obs_noise: 0.1,
            kernel_lengthscale: 1.0,
            kernel_variance: 1.0,
            jitter: 1e-6,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result
// ─────────────────────────────────────────────────────────────────────────────

/// Output of the GP-SSM particle filter.
#[derive(Debug, Clone)]
pub struct GpSsmResult {
    /// Particle-weighted mean of the latent state at each time step.
    /// Shape: `[T][latent_dim]`.
    pub filtered_states: Vec<Vec<f64>>,
    /// Flattened (row-major) particle-weighted covariance matrix of the latent
    /// state at each time step.  Shape: `[T][latent_dim²]`.
    pub filtered_cov: Vec<Vec<f64>>,
    /// Approximation to the log marginal likelihood `log p(y_{1:T})`.
    pub log_likelihood: f64,
    /// Number of times systematic resampling was performed.
    pub n_resamples: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// GpSsm
// ─────────────────────────────────────────────────────────────────────────────

/// Bootstrap Particle Filter GP-State-Space Model.
///
/// The GP transition function is parameterised by a set of inducing inputs
/// **Z** ∈ ℝ^{M × d} and associated inducing-output weights **w** ∈ ℝ^{M}.
/// The posterior predictive mean at a test point `x*` is:
///
/// ```text
/// f*(x*) = k(x*, Z) α,   α = (K(Z,Z) + σ²_n I)^{-1} w
/// ```
///
/// where the kernel is the RBF / squared-exponential kernel.
pub struct GpSsm {
    config: GpSsmConfig,
    /// Inducing inputs: flat row-major [M × latent_dim].
    inducing_inputs: Vec<Vec<f64>>,
    /// GP weight vector α: [M × latent_dim] (one α per latent dimension).
    gp_weights: Vec<Vec<f64>>,
}

impl GpSsm {
    /// Create a new `GpSsm` with the given configuration.
    /// The GP inducing points are initialised on a regular grid over `[-1, 1]`.
    pub fn new(config: GpSsmConfig) -> Self {
        let n_inducing = 5.max(config.latent_dim * 3);
        let step = 2.0 / (n_inducing as f64 - 1.0).max(1.0);
        let inducing_inputs: Vec<Vec<f64>> = (0..n_inducing)
            .map(|i| {
                let base = -1.0 + i as f64 * step;
                (0..config.latent_dim).map(|_| base).collect()
            })
            .collect();
        let gp_weights = vec![vec![0.0_f64; n_inducing]; config.latent_dim];
        Self {
            config,
            inducing_inputs,
            gp_weights,
        }
    }

    // ─── public API ─────────────────────────────────────────────────────────

    /// Run the bootstrap particle filter on a sequence of observations.
    ///
    /// # Arguments
    ///
    /// * `observations` – sequence of observations, each of length `obs_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`TimeSeriesError::InvalidInput`] when an observation vector
    /// has wrong length.
    pub fn filter(&self, observations: &[Vec<f64>]) -> Result<GpSsmResult> {
        let n = self.config.n_particles;
        let d = self.config.latent_dim;
        let obs_dim = self.config.obs_dim;
        let t_max = observations.len();

        if t_max == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "observations must be non-empty".into(),
            ));
        }
        for (t, obs) in observations.iter().enumerate() {
            if obs.len() != obs_dim {
                return Err(TimeSeriesError::InvalidInput(format!(
                    "observation at t={t} has length {} != obs_dim {}",
                    obs.len(),
                    obs_dim
                )));
            }
        }

        // Initialise particles uniformly in [-1, 1]^d
        let mut particles: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..d)
                    .map(|j| {
                        // Deterministic quasi-random grid for reproducibility
                        let frac = (i * d + j) as f64 / (n * d) as f64;
                        -1.0 + 2.0 * frac
                    })
                    .collect()
            })
            .collect();
        let mut weights = vec![1.0 / n as f64; n];

        let mut filtered_states = Vec::with_capacity(t_max);
        let mut filtered_cov = Vec::with_capacity(t_max);
        let mut log_likelihood = 0.0_f64;
        let mut n_resamples = 0_usize;

        let mut rng_state: u64 = 12345;

        for (t, obs) in observations.iter().enumerate() {
            // 1. Propagate
            let mut new_particles: Vec<Vec<f64>> = Vec::with_capacity(n);
            for p in &particles {
                let mean = self.transition(p);
                let noisy: Vec<f64> = mean
                    .iter()
                    .enumerate()
                    .map(|(j, &m)| {
                        let noise = self.lcg_normal(&mut rng_state, j as u64 + t as u64 * 997);
                        m + self.config.transition_noise * noise
                    })
                    .collect();
                new_particles.push(noisy);
            }
            particles = new_particles;

            // 2. Weight by observation likelihood
            let mut sum_w = 0.0_f64;
            for (i, p) in particles.iter().enumerate() {
                let lhood = self.obs_likelihood(obs, p);
                weights[i] *= lhood;
                sum_w += weights[i];
            }

            // Accumulate log-likelihood
            if sum_w > 0.0 {
                log_likelihood += sum_w.ln();
            } else {
                log_likelihood += f64::NEG_INFINITY;
            }

            // 3. Normalise
            if sum_w > 1e-300 {
                for w in weights.iter_mut() {
                    *w /= sum_w;
                }
            } else {
                // Weight degeneracy – reset
                let unif = 1.0 / n as f64;
                for w in weights.iter_mut() {
                    *w = unif;
                }
            }

            // 4. Compute weighted mean and covariance
            let mean_state = Self::weighted_mean(&particles, &weights, d);
            let cov_state = Self::weighted_cov(&particles, &weights, &mean_state, d);
            filtered_states.push(mean_state);
            filtered_cov.push(cov_state);

            // 5. Resample if ESS < N/2
            let ess = Self::effective_sample_size(&weights);
            if ess < (n as f64) / 2.0 {
                let seed = rng_state.wrapping_add(t as u64 * 31337);
                let indices = Self::systematic_resample(&weights, n, seed);
                let old_particles = particles.clone();
                for (i, &idx) in indices.iter().enumerate() {
                    particles[i] = old_particles[idx].clone();
                }
                let unif = 1.0 / n as f64;
                for w in weights.iter_mut() {
                    *w = unif;
                }
                n_resamples += 1;
                rng_state = rng_state.wrapping_add(1);
            }
        }

        Ok(GpSsmResult {
            filtered_states,
            filtered_cov,
            log_likelihood,
            n_resamples,
        })
    }

    /// Fit inducing points from data (a simple subsample / k-means lite).
    ///
    /// Uses `n_inducing` evenly spaced data points as inducing inputs and
    /// sets the GP weights to zero (prior mean zero).
    pub fn fit_inducing(&mut self, data: &[Vec<f64>], n_inducing: usize) -> Result<()> {
        if data.is_empty() {
            return Err(TimeSeriesError::InvalidInput("data is empty".into()));
        }
        let latent_dim = self.config.latent_dim;
        let n_inducing = n_inducing.max(2);
        let step = (data.len() - 1) as f64 / (n_inducing - 1) as f64;
        self.inducing_inputs = (0..n_inducing)
            .map(|i| {
                let idx = ((i as f64 * step).round() as usize).min(data.len() - 1);
                let row = &data[idx];
                (0..latent_dim)
                    .map(|j| *row.get(j).unwrap_or(&0.0))
                    .collect()
            })
            .collect();
        self.gp_weights = vec![vec![0.0_f64; n_inducing]; latent_dim];
        Ok(())
    }

    // ─── GP utilities ────────────────────────────────────────────────────────

    /// GP transition: compute f(x) using the sparse GP posterior mean.
    ///
    /// `f_d(x) = Σ_m k(x, z_m) α_{d,m}`
    fn transition(&self, x: &[f64]) -> Vec<f64> {
        let d = self.config.latent_dim;
        (0..d)
            .map(|dim| {
                self.inducing_inputs
                    .iter()
                    .enumerate()
                    .map(|(m, z)| self.kernel(x, z) * self.gp_weights[dim][m])
                    .sum::<f64>()
            })
            .collect()
    }

    /// RBF (squared-exponential) kernel.
    ///
    /// `k(x, y) = σ² exp(-‖x-y‖² / (2 l²))`
    pub fn kernel(&self, x: &[f64], y: &[f64]) -> f64 {
        let len_sq = self.config.kernel_lengthscale.powi(2);
        let dist_sq: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - yi).powi(2))
            .sum();
        self.config.kernel_variance * (-dist_sq / (2.0 * len_sq)).exp()
    }

    /// Observation likelihood `p(y | x) = N(y ; C x, σ_obs² I)`.
    ///
    /// The observation matrix C is the first `obs_dim` rows of the identity
    /// (i.e., we observe the first `obs_dim` latent dimensions directly).
    fn obs_likelihood(&self, y: &[f64], x: &[f64]) -> f64 {
        let obs_dim = self.config.obs_dim;
        let sigma = self.config.obs_noise;
        let sigma_sq = sigma * sigma;
        let log_normaliser =
            -(obs_dim as f64) * (0.5 * (2.0 * std::f64::consts::PI * sigma_sq).ln());
        let log_exp: f64 = y
            .iter()
            .enumerate()
            .map(|(d, &y_d)| {
                let cx_d = *x.get(d).unwrap_or(&0.0);
                -0.5 * (y_d - cx_d).powi(2) / sigma_sq
            })
            .sum();
        (log_normaliser + log_exp).exp().max(1e-300)
    }

    // ─── particle utilities ──────────────────────────────────────────────────

    /// Compute the particle-weighted mean.
    fn weighted_mean(particles: &[Vec<f64>], weights: &[f64], d: usize) -> Vec<f64> {
        let mut mean = vec![0.0_f64; d];
        for (p, &w) in particles.iter().zip(weights.iter()) {
            for (j, &pj) in p.iter().enumerate().take(d) {
                mean[j] += w * pj;
            }
        }
        mean
    }

    /// Compute the flattened (row-major) particle-weighted covariance.
    fn weighted_cov(particles: &[Vec<f64>], weights: &[f64], mean: &[f64], d: usize) -> Vec<f64> {
        let mut cov = vec![0.0_f64; d * d];
        for (p, &w) in particles.iter().zip(weights.iter()) {
            for i in 0..d {
                let di = p.get(i).copied().unwrap_or(0.0) - mean[i];
                for j in 0..d {
                    let dj = p.get(j).copied().unwrap_or(0.0) - mean[j];
                    cov[i * d + j] += w * di * dj;
                }
            }
        }
        cov
    }

    /// Effective Sample Size = 1 / Σ w_i².
    pub fn effective_sample_size(weights: &[f64]) -> f64 {
        let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
        if sum_sq < 1e-300 {
            0.0
        } else {
            1.0 / sum_sq
        }
    }

    /// Systematic resampling.
    ///
    /// Returns a vector of `n` indices sampled proportional to `weights`.
    pub fn systematic_resample(weights: &[f64], n: usize, seed: u64) -> Vec<usize> {
        // Use a deterministic LCG offset for the single uniform draw
        let u0 = (lcg_u64(seed) as f64) / (u64::MAX as f64) / n as f64;
        let mut indices = Vec::with_capacity(n);
        let mut cumsum = 0.0_f64;
        let mut j = 0_usize;
        let wn = weights.len();
        for i in 0..n {
            let target = u0 + i as f64 / n as f64;
            while j < wn - 1 {
                cumsum += weights[j];
                if cumsum > target {
                    break;
                }
                j += 1;
            }
            indices.push(j.min(wn - 1));
        }
        indices
    }

    // ─── PRNG helpers (no external RNG deps) ────────────────────────────────

    /// Generate a standard-normal sample using Box-Muller (deterministic LCG).
    fn lcg_normal(&self, state: &mut u64, extra: u64) -> f64 {
        let u1 = lcg_unit(state, extra);
        let u2 = lcg_unit(state, extra.wrapping_add(1));
        let r = (-2.0 * u1.max(1e-15).ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        r * theta.cos()
    }
}

// ─── LCG helper functions (module-level, not methods) ───────────────────────

fn lcg_u64(seed: u64) -> u64 {
    seed.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

fn lcg_unit(state: &mut u64, extra: u64) -> f64 {
    *state = lcg_u64(state.wrapping_add(extra));
    (*state as f64) / (u64::MAX as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let cfg = GpSsmConfig::default();
        assert_eq!(cfg.n_particles, 100);
        assert_eq!(cfg.latent_dim, 2);
        assert_eq!(cfg.obs_dim, 1);
        assert!((cfg.transition_noise - 0.1).abs() < 1e-12);
        assert!((cfg.obs_noise - 0.1).abs() < 1e-12);
        assert!((cfg.kernel_lengthscale - 1.0).abs() < 1e-12);
        assert!((cfg.kernel_variance - 1.0).abs() < 1e-12);
        assert!(cfg.jitter < 1e-5);
    }

    #[test]
    fn test_rbf_kernel_self() {
        let cfg = GpSsmConfig::default();
        let model = GpSsm::new(cfg.clone());
        let x = vec![0.5, -0.3];
        let k_self = model.kernel(&x, &x);
        // k(x,x) = σ² exp(0) = σ²
        assert!(
            (k_self - cfg.kernel_variance).abs() < 1e-12,
            "k(x,x) should equal kernel_variance, got {k_self}"
        );
    }

    #[test]
    fn test_rbf_kernel_decreasing() {
        let cfg = GpSsmConfig::default();
        let model = GpSsm::new(cfg);
        let x = vec![0.0, 0.0];
        let y_near = vec![0.1, 0.0];
        let y_far = vec![2.0, 0.0];
        let k_near = model.kernel(&x, &y_near);
        let k_far = model.kernel(&x, &y_far);
        assert!(k_near > k_far, "kernel should decrease with distance");
    }

    #[test]
    fn test_filter_output_shapes() {
        let cfg = GpSsmConfig {
            n_particles: 20,
            latent_dim: 2,
            obs_dim: 1,
            ..Default::default()
        };
        let model = GpSsm::new(cfg.clone());
        let observations: Vec<Vec<f64>> = (0..10).map(|t| vec![t as f64 * 0.1]).collect();
        let res = model.filter(&observations).expect("filter should succeed");
        assert_eq!(res.filtered_states.len(), 10, "one state per time step");
        assert_eq!(res.filtered_cov.len(), 10);
        for s in &res.filtered_states {
            assert_eq!(s.len(), cfg.latent_dim);
        }
        for c in &res.filtered_cov {
            assert_eq!(c.len(), cfg.latent_dim * cfg.latent_dim);
        }
    }

    #[test]
    fn test_filter_log_likelihood_finite() {
        let cfg = GpSsmConfig {
            n_particles: 30,
            latent_dim: 2,
            obs_dim: 1,
            ..Default::default()
        };
        let model = GpSsm::new(cfg);
        let obs: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0]).collect();
        let res = model.filter(&obs).expect("filter should succeed");
        assert!(
            res.log_likelihood.is_finite(),
            "log_likelihood must be finite"
        );
    }

    #[test]
    fn test_systematic_resample_range() {
        let weights = vec![0.1_f64, 0.4, 0.2, 0.3];
        let n = 4;
        let indices = GpSsm::systematic_resample(&weights, n, 42);
        assert_eq!(indices.len(), n);
        for &idx in &indices {
            assert!(idx < weights.len(), "index {idx} out of range");
        }
    }

    #[test]
    fn test_effective_sample_size_uniform() {
        let n = 50_usize;
        let weights = vec![1.0 / n as f64; n];
        let ess = GpSsm::effective_sample_size(&weights);
        // For uniform weights ESS ≈ n
        assert!(
            (ess - n as f64).abs() < 1.0,
            "ESS of uniform weights should be ~n, got {ess}"
        );
    }

    #[test]
    fn test_fit_inducing() {
        let cfg = GpSsmConfig {
            latent_dim: 2,
            ..Default::default()
        };
        let mut model = GpSsm::new(cfg);
        let data: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, -(i as f64)]).collect();
        model.fit_inducing(&data, 5).expect("fit should succeed");
        assert_eq!(model.inducing_inputs.len(), 5);
    }

    #[test]
    fn test_empty_observations_error() {
        let cfg = GpSsmConfig::default();
        let model = GpSsm::new(cfg);
        let obs: Vec<Vec<f64>> = vec![];
        assert!(model.filter(&obs).is_err());
    }

    #[test]
    fn test_wrong_obs_dim_error() {
        let cfg = GpSsmConfig {
            obs_dim: 1,
            ..Default::default()
        };
        let model = GpSsm::new(cfg);
        let obs = vec![vec![1.0, 2.0]]; // wrong dim
        assert!(model.filter(&obs).is_err());
    }
}
