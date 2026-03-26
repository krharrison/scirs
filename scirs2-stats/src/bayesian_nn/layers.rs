//! Variational Bayesian linear layers for Bayesian Neural Networks.
//!
//! Implements Bayes-by-Backprop (Blundell et al. 2015) local reparameterization:
//! weights are parameterized by means and log-standard-deviations, with MC sampling
//! at forward pass time and KL divergence for the ELBO objective.
//!
//! Each weight w ~ q(w) = N(w_mu, exp(w_log_sigma)^2).
//! The prior is p(w) = N(0, prior_std^2).
//!
//! KL[q || p] = -0.5 * sum(1 + 2*log_sigma - log(prior_std^2) - mu^2/prior_std^2
//!                          - exp(2*log_sigma)/prior_std^2)

use crate::error::StatsError;

/// Configuration for variational Bayesian neural network layers.
#[derive(Debug, Clone)]
pub struct BnnConfig {
    /// Standard deviation of the weight prior N(0, prior_std^2)
    pub prior_std: f64,
    /// Scaling factor on the KL term in the ELBO (default 1.0)
    pub kl_weight: f64,
    /// Number of Monte Carlo forward-pass samples for stochastic gradient estimation
    pub n_samples_mc: usize,
}

impl Default for BnnConfig {
    fn default() -> Self {
        Self {
            prior_std: 1.0,
            kl_weight: 1.0,
            n_samples_mc: 10,
        }
    }
}

/// A single variational Bayesian linear layer.
///
/// Weights: w_{ij} ~ N(w_mu_{ij}, exp(w_log_sigma_{ij})^2)
/// Biases:  b_j   ~ N(b_mu_j,   exp(b_log_sigma_j)^2)
///
/// Stored as flat row-major vectors of length `out_features * in_features`.
#[derive(Debug, Clone)]
pub struct BayesianLinear {
    /// Number of input features
    pub in_features: usize,
    /// Number of output features
    pub out_features: usize,
    /// Weight posterior means, length `out_features * in_features`
    pub w_mu: Vec<f64>,
    /// Weight posterior log-std, length `out_features * in_features`
    pub w_log_sigma: Vec<f64>,
    /// Bias posterior means, length `out_features`
    pub b_mu: Vec<f64>,
    /// Bias posterior log-std, length `out_features`
    pub b_log_sigma: Vec<f64>,
    /// Prior standard deviation
    pub prior_std: f64,
}

impl BayesianLinear {
    /// Create a new `BayesianLinear` layer.
    ///
    /// Weights are initialized from N(0, 0.1) and log-sigma initialized to -3.0
    /// (corresponding to sigma ≈ 0.05, tight initial posterior).
    ///
    /// # Arguments
    /// * `in_features` - Input dimensionality
    /// * `out_features` - Output dimensionality
    /// * `prior_std` - Standard deviation of the weight prior
    ///
    /// # Errors
    /// Returns an error if `in_features` or `out_features` is zero.
    pub fn new(
        in_features: usize,
        out_features: usize,
        prior_std: f64,
    ) -> Result<Self, StatsError> {
        if in_features == 0 {
            return Err(StatsError::InvalidArgument(
                "in_features must be > 0".to_string(),
            ));
        }
        if out_features == 0 {
            return Err(StatsError::InvalidArgument(
                "out_features must be > 0".to_string(),
            ));
        }
        if prior_std <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "prior_std must be positive".to_string(),
            ));
        }

        let n_weights = out_features * in_features;

        // Initialize w_mu ~ N(0, 0.1) using a deterministic pseudo-random scheme
        // (Lehmer LCG seeded by size for reproducibility without external RNG dependency)
        let mut w_mu = vec![0.0f64; n_weights];
        let mut state: u64 = (n_weights as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        for wm in w_mu.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map u64 to [-0.1, 0.1]
            let u = (state >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
            *wm = (u - 0.5) * 0.2; // [-0.1, 0.1]
        }

        let w_log_sigma = vec![-3.0f64; n_weights];
        let b_mu = vec![0.0f64; out_features];
        let b_log_sigma = vec![-3.0f64; out_features];

        Ok(Self {
            in_features,
            out_features,
            w_mu,
            w_log_sigma,
            b_mu,
            b_log_sigma,
            prior_std,
        })
    }

    /// Forward pass with sampled weights (reparameterization trick).
    ///
    /// Samples w = w_mu + eps * exp(w_log_sigma) for each weight and bias,
    /// then computes the matrix-vector product.
    ///
    /// # Arguments
    /// * `x` - Input vector of length `in_features`
    /// * `rng` - Closure producing standard normal samples N(0,1)
    ///
    /// # Returns
    /// Output vector of length `out_features`
    ///
    /// # Errors
    /// Returns an error if `x` has incorrect length.
    pub fn forward_sample(
        &self,
        x: &[f64],
        rng: &mut impl FnMut() -> f64,
    ) -> Result<Vec<f64>, StatsError> {
        if x.len() != self.in_features {
            return Err(StatsError::DimensionMismatch(format!(
                "input length {} != in_features {}",
                x.len(),
                self.in_features
            )));
        }

        let mut out = vec![0.0f64; self.out_features];
        for o in 0..self.out_features {
            // Sampled bias
            let eps_b = rng();
            let b_sigma = self.b_log_sigma[o].exp();
            let b_sample = self.b_mu[o] + eps_b * b_sigma;

            let mut acc = b_sample;
            for i in 0..self.in_features {
                let idx = o * self.in_features + i;
                let eps_w = rng();
                let w_sigma = self.w_log_sigma[idx].exp();
                let w_sample = self.w_mu[idx] + eps_w * w_sigma;
                acc += w_sample * x[i];
            }
            out[o] = acc;
        }
        Ok(out)
    }

    /// Deterministic forward pass using posterior means only.
    ///
    /// Computes output = W_mu @ x + b_mu. Useful for fast predictive mean.
    ///
    /// # Arguments
    /// * `x` - Input vector of length `in_features`
    ///
    /// # Errors
    /// Returns an error if `x` has incorrect length.
    pub fn forward_mean(&self, x: &[f64]) -> Result<Vec<f64>, StatsError> {
        if x.len() != self.in_features {
            return Err(StatsError::DimensionMismatch(format!(
                "input length {} != in_features {}",
                x.len(),
                self.in_features
            )));
        }

        let mut out = vec![0.0f64; self.out_features];
        for o in 0..self.out_features {
            let mut acc = self.b_mu[o];
            for i in 0..self.in_features {
                acc += self.w_mu[o * self.in_features + i] * x[i];
            }
            out[o] = acc;
        }
        Ok(out)
    }

    /// Compute the KL divergence KL[q(w) || p(w)] for all weights and biases.
    ///
    /// For q = N(mu, sigma^2) and p = N(0, prior_std^2):
    /// KL = -0.5 * sum(1 + 2*log_sigma - log(prior_std^2) - mu^2/prior_std^2
    ///                   - sigma^2/prior_std^2)
    ///
    /// # Arguments
    /// * `prior_std` - Prior standard deviation (can differ from initialization value)
    pub fn kl_divergence(&self, prior_std: f64) -> f64 {
        let log_prior_var = (prior_std * prior_std).ln();
        let prior_var = prior_std * prior_std;
        let mut kl = 0.0;

        // Weights
        for i in 0..(self.out_features * self.in_features) {
            let mu = self.w_mu[i];
            let log_sigma = self.w_log_sigma[i];
            let sigma_sq = (2.0 * log_sigma).exp();
            kl += -0.5
                * (1.0 + 2.0 * log_sigma
                    - log_prior_var
                    - mu * mu / prior_var
                    - sigma_sq / prior_var);
        }

        // Biases
        for o in 0..self.out_features {
            let mu = self.b_mu[o];
            let log_sigma = self.b_log_sigma[o];
            let sigma_sq = (2.0 * log_sigma).exp();
            kl += -0.5
                * (1.0 + 2.0 * log_sigma
                    - log_prior_var
                    - mu * mu / prior_var
                    - sigma_sq / prior_var);
        }

        kl
    }

    /// Apply a gradient step (SGD) to the variational parameters.
    ///
    /// # Arguments
    /// * `grad_w_mu`        - Gradient of loss w.r.t. w_mu, length `out*in`
    /// * `grad_w_log_sigma` - Gradient of loss w.r.t. w_log_sigma, length `out*in`
    /// * `grad_b_mu`        - Gradient of loss w.r.t. b_mu, length `out`
    /// * `grad_b_log_sigma` - Gradient of loss w.r.t. b_log_sigma, length `out`
    /// * `lr`               - Learning rate
    ///
    /// # Errors
    /// Returns an error if gradient dimensions are inconsistent.
    pub fn update(
        &mut self,
        grad_w_mu: &[f64],
        grad_w_log_sigma: &[f64],
        grad_b_mu: &[f64],
        grad_b_log_sigma: &[f64],
        lr: f64,
    ) -> Result<(), StatsError> {
        let n_weights = self.out_features * self.in_features;
        if grad_w_mu.len() != n_weights {
            return Err(StatsError::DimensionMismatch(format!(
                "grad_w_mu length {} != {}",
                grad_w_mu.len(),
                n_weights
            )));
        }
        if grad_w_log_sigma.len() != n_weights {
            return Err(StatsError::DimensionMismatch(format!(
                "grad_w_log_sigma length {} != {}",
                grad_w_log_sigma.len(),
                n_weights
            )));
        }
        if grad_b_mu.len() != self.out_features {
            return Err(StatsError::DimensionMismatch(format!(
                "grad_b_mu length {} != {}",
                grad_b_mu.len(),
                self.out_features
            )));
        }
        if grad_b_log_sigma.len() != self.out_features {
            return Err(StatsError::DimensionMismatch(format!(
                "grad_b_log_sigma length {} != {}",
                grad_b_log_sigma.len(),
                self.out_features
            )));
        }

        for i in 0..n_weights {
            self.w_mu[i] -= lr * grad_w_mu[i];
            self.w_log_sigma[i] -= lr * grad_w_log_sigma[i];
        }
        for o in 0..self.out_features {
            self.b_mu[o] -= lr * grad_b_mu[o];
            self.b_log_sigma[o] -= lr * grad_b_log_sigma[o];
        }
        Ok(())
    }

    /// Total number of variational parameters (for KL / ELBO scaling)
    pub fn n_params(&self) -> usize {
        2 * (self.out_features * self.in_features + self.out_features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_normal_rng() -> impl FnMut() -> f64 {
        // Box-Muller for standard normal without external deps
        let mut state: u64 = 12345678901234567;
        let mut cached: Option<f64> = None;
        move || {
            if let Some(v) = cached.take() {
                return v;
            }
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = (state >> 11) as f64 / (1u64 << 53) as f64 + 1e-15;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            cached = Some(r * theta.sin());
            r * theta.cos()
        }
    }

    #[test]
    fn test_bayesian_linear_new() {
        let layer = BayesianLinear::new(3, 4, 1.0).expect("creation should succeed");
        assert_eq!(layer.in_features, 3);
        assert_eq!(layer.out_features, 4);
        assert_eq!(layer.w_mu.len(), 12);
        assert_eq!(layer.w_log_sigma.len(), 12);
        assert_eq!(layer.b_mu.len(), 4);
        assert_eq!(layer.b_log_sigma.len(), 4);
        // All log-sigma should be -3.0
        for &ls in &layer.w_log_sigma {
            assert!((ls - (-3.0)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_forward_mean_shape() {
        let layer = BayesianLinear::new(5, 3, 1.0).expect("creation");
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let out = layer.forward_mean(&x).expect("forward_mean");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_forward_sample_shape() {
        let layer = BayesianLinear::new(4, 2, 1.0).expect("creation");
        let x = vec![1.0, 0.0, -1.0, 0.5];
        let mut rng = make_normal_rng();
        let out = layer.forward_sample(&x, &mut rng).expect("forward_sample");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_kl_divergence_positive() {
        // With non-zero means, KL should be > 0
        let mut layer = BayesianLinear::new(2, 2, 1.0).expect("creation");
        layer.w_mu[0] = 1.0;
        layer.w_mu[1] = -0.5;
        let kl = layer.kl_divergence(1.0);
        assert!(
            kl > 0.0,
            "KL divergence should be positive with non-zero means, got {}",
            kl
        );
    }

    #[test]
    fn test_kl_zero_with_prior_params() {
        // When mu=0 and sigma=prior_std, KL should be 0
        let prior_std = 1.0;
        let mut layer = BayesianLinear::new(2, 1, prior_std).expect("creation");
        // Set mu=0, log_sigma = log(prior_std) = 0.0
        for w in layer.w_mu.iter_mut() {
            *w = 0.0;
        }
        for ls in layer.w_log_sigma.iter_mut() {
            *ls = prior_std.ln();
        } // = 0.0
        for b in layer.b_mu.iter_mut() {
            *b = 0.0;
        }
        for ls in layer.b_log_sigma.iter_mut() {
            *ls = prior_std.ln();
        }
        let kl = layer.kl_divergence(prior_std);
        assert!(kl.abs() < 1e-10, "KL should be ~0 when q=p, got {}", kl);
    }

    #[test]
    fn test_update_step() {
        let mut layer = BayesianLinear::new(2, 2, 1.0).expect("creation");
        let w_mu_before = layer.w_mu.clone();
        let grad_w_mu = vec![1.0, 0.0, -1.0, 0.5];
        let grad_w_ls = vec![0.1, 0.2, 0.3, 0.4];
        let grad_b_mu = vec![0.5, -0.5];
        let grad_b_ls = vec![0.1, 0.1];
        layer
            .update(&grad_w_mu, &grad_w_ls, &grad_b_mu, &grad_b_ls, 0.01)
            .expect("update");
        // w_mu should have changed
        assert!((layer.w_mu[0] - (w_mu_before[0] - 0.01 * 1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_dimension_errors() {
        assert!(BayesianLinear::new(0, 3, 1.0).is_err());
        assert!(BayesianLinear::new(3, 0, 1.0).is_err());
        assert!(BayesianLinear::new(3, 3, -1.0).is_err());

        let layer = BayesianLinear::new(3, 2, 1.0).expect("creation");
        assert!(layer.forward_mean(&[1.0, 2.0]).is_err()); // wrong input size
    }
}
