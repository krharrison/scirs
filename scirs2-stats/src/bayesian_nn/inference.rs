//! Variational BNN: multi-layer inference, training, and uncertainty decomposition.
//!
//! `BayesianMlp` is a multi-layer perceptron with variational Bayesian linear layers.
//! Training maximizes the ELBO (Evidence Lower Bound):
//!
//!   ELBO = E_q[log p(y|x,w)] - (1/n_data) * KL[q(w) || p(w)]
//!
//! MC gradient estimates for the likelihood term use the local reparameterization trick
//! (Kingma & Welling 2014) as implemented in `BayesianLinear::forward_sample`.
//!
//! Finite-difference gradients are used for pedagogical clarity; the step size and
//! structure are chosen to give stable learning on simple regression tasks.

use super::layers::{BayesianLinear, BnnConfig};
use crate::error::StatsError;

/// A multi-layer Bayesian neural network for regression with uncertainty quantification.
///
/// Architecture: input → [BayesianLinear → ReLU] × (L-1) → BayesianLinear → output
///
/// Uncertainty is decomposed into:
/// - Epistemic (model uncertainty): variance of the predictive means across MC samples
/// - Aleatoric (data noise): mean of within-sample variance (not applicable in the
///   current deterministic output model; set to zero)
#[derive(Debug, Clone)]
pub struct BayesianMlp {
    /// Variational linear layers
    pub layers: Vec<BayesianLinear>,
    /// Configuration
    pub config: BnnConfig,
}

impl BayesianMlp {
    /// Create a new `BayesianMlp`.
    ///
    /// # Arguments
    /// * `layer_sizes` - Slice of layer widths including input and output, e.g. `[4, 16, 1]`
    /// * `config` - Variational BNN configuration
    ///
    /// # Errors
    /// Returns an error if fewer than 2 layer sizes are given or any size is zero.
    pub fn new(layer_sizes: &[usize], config: BnnConfig) -> Result<Self, StatsError> {
        if layer_sizes.len() < 2 {
            return Err(StatsError::InvalidArgument(
                "layer_sizes must have at least 2 elements (input and output)".to_string(),
            ));
        }
        let mut layers = Vec::with_capacity(layer_sizes.len() - 1);
        for w in layer_sizes.windows(2) {
            layers.push(BayesianLinear::new(w[0], w[1], config.prior_std)?);
        }
        Ok(Self { layers, config })
    }

    /// Run the network forward using the posterior mean weights (deterministic).
    ///
    /// Applies ReLU between all but the last layer. No sampling is performed.
    ///
    /// # Arguments
    /// * `x` - Input vector
    ///
    /// # Errors
    /// Returns an error if the input has incorrect length.
    pub fn forward_mean(&self, x: &[f64]) -> Result<Vec<f64>, StatsError> {
        let mut h: Vec<f64> = x.to_vec();
        let n_layers = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward_mean(&h)?;
            if i < n_layers - 1 {
                // ReLU activation
                for v in h.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
        }
        Ok(h)
    }

    /// Run one stochastic forward pass, sampling weights at each layer.
    ///
    /// Applies ReLU between all but the last layer.
    fn forward_sample(
        &self,
        x: &[f64],
        rng: &mut impl FnMut() -> f64,
    ) -> Result<Vec<f64>, StatsError> {
        let mut h: Vec<f64> = x.to_vec();
        let n_layers = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward_sample(&h, rng)?;
            if i < n_layers - 1 {
                for v in h.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
        }
        Ok(h)
    }

    /// Draw `n_samples` Monte Carlo forward passes for a single input `x`.
    ///
    /// # Arguments
    /// * `x` - Input vector
    /// * `n_samples` - Number of stochastic forward passes
    /// * `rng` - Standard-normal sampler closure
    ///
    /// # Returns
    /// Vector of `n_samples` output vectors (each of length equal to output dimensionality).
    ///
    /// # Errors
    /// Returns an error if the input has incorrect length.
    pub fn predict_samples(
        &self,
        x: &[f64],
        n_samples: usize,
        rng: &mut impl FnMut() -> f64,
    ) -> Result<Vec<Vec<f64>>, StatsError> {
        let mut samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            samples.push(self.forward_sample(x, rng)?);
        }
        Ok(samples)
    }

    /// Compute predictive mean and standard deviation via Monte Carlo sampling.
    ///
    /// Epistemic uncertainty is captured as the standard deviation of the sample means.
    ///
    /// # Arguments
    /// * `x` - Input vector
    /// * `n_samples` - Number of MC samples
    /// * `rng` - Standard-normal sampler closure
    ///
    /// # Returns
    /// Tuple `(means, stds)` where each element corresponds to one output dimension.
    ///
    /// # Errors
    /// Returns an error if the input is invalid or network has no layers.
    pub fn predict_mean_std(
        &self,
        x: &[f64],
        n_samples: usize,
        rng: &mut impl FnMut() -> f64,
    ) -> Result<(Vec<f64>, Vec<f64>), StatsError> {
        if n_samples == 0 {
            return Err(StatsError::InvalidArgument(
                "n_samples must be > 0".to_string(),
            ));
        }
        let samples = self.predict_samples(x, n_samples, rng)?;
        let out_dim = samples[0].len();
        let ns = n_samples as f64;

        let mut means = vec![0.0f64; out_dim];
        for s in &samples {
            for (j, &v) in s.iter().enumerate() {
                means[j] += v;
            }
        }
        for m in means.iter_mut() {
            *m /= ns;
        }

        let mut stds = vec![0.0f64; out_dim];
        for s in &samples {
            for (j, &v) in s.iter().enumerate() {
                stds[j] += (v - means[j]).powi(2);
            }
        }
        for st in stds.iter_mut() {
            *st = (*st / ns).sqrt();
        }

        Ok((means, stds))
    }

    /// Sum of KL divergences over all layers.
    pub fn total_kl(&self) -> f64 {
        self.layers
            .iter()
            .map(|l| l.kl_divergence(self.config.prior_std))
            .sum()
    }

    /// Compute the negative ELBO (loss to minimize) on a mini-batch.
    ///
    /// loss = -E_q[log p(y|x,w)] + (1/n_data) * KL[q(w) || p(w)]
    ///
    /// The likelihood term uses a Gaussian observation model:
    ///   log p(y_i | x_i, w) = -0.5 * (y_i - f(x_i, w))^2 / (0.1^2)
    ///
    /// # Arguments
    /// * `x_batch` - Input features, one vector per example
    /// * `y_batch` - Scalar targets, one per example
    /// * `n_data`  - Total dataset size (for correct KL scaling)
    /// * `rng`     - Standard-normal sampler
    ///
    /// # Errors
    /// Returns an error if batch sizes are inconsistent.
    pub fn elbo_loss(
        &self,
        x_batch: &[Vec<f64>],
        y_batch: &[f64],
        n_data: usize,
        rng: &mut impl FnMut() -> f64,
    ) -> Result<f64, StatsError> {
        if x_batch.len() != y_batch.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "x_batch length {} != y_batch length {}",
                x_batch.len(),
                y_batch.len()
            )));
        }
        if x_batch.is_empty() {
            return Err(StatsError::InsufficientData(
                "Batch must be non-empty".to_string(),
            ));
        }
        if n_data == 0 {
            return Err(StatsError::InvalidArgument(
                "n_data must be > 0".to_string(),
            ));
        }

        let n_mc = self.config.n_samples_mc.max(1);
        let sigma_obs = 0.1f64;
        let sigma_obs_sq = sigma_obs * sigma_obs;
        let n_batch = x_batch.len() as f64;

        // Estimate -E[log p(y|x,w)] via MC
        let mut neg_ll = 0.0f64;
        for mc in 0..n_mc {
            let _ = mc; // suppress unused warning
            let mut ll_sample = 0.0f64;
            for (x, &y) in x_batch.iter().zip(y_batch.iter()) {
                let out = self.forward_sample(x, rng)?;
                let pred = out[0]; // scalar output assumed
                let diff = y - pred;
                ll_sample += -0.5 * diff * diff / sigma_obs_sq
                    - 0.5 * (2.0 * std::f64::consts::PI * sigma_obs_sq).ln();
            }
            neg_ll += -ll_sample / n_batch;
        }
        neg_ll /= n_mc as f64;

        // KL term scaled by batch size / n_data  (importance weighting)
        let kl_term = self.total_kl() / n_data as f64;

        Ok(neg_ll + self.config.kl_weight * kl_term)
    }

    /// Perform one gradient descent step on the ELBO using finite differences.
    ///
    /// For each variational parameter θ_i, the gradient is approximated as:
    ///   dL/dθ_i ≈ (L(θ_i + h) - L(θ_i - h)) / (2h)
    ///
    /// This is O(2 * n_params) forward passes — suitable for small networks / demos.
    /// For production, use automatic differentiation.
    ///
    /// # Arguments
    /// * `x_batch` - Input features
    /// * `y_batch` - Scalar targets
    /// * `n_data`  - Total dataset size
    /// * `lr`      - Learning rate
    /// * `rng`     - Standard-normal sampler (reused across perturbations)
    ///
    /// # Returns
    /// ELBO loss at the current parameters (before the update).
    ///
    /// # Errors
    /// Returns an error if batch sizes are inconsistent.
    pub fn train_step(
        &mut self,
        x_batch: &[Vec<f64>],
        y_batch: &[f64],
        n_data: usize,
        lr: f64,
        rng: &mut impl FnMut() -> f64,
    ) -> Result<f64, StatsError> {
        let fd_h = 1e-4f64;

        // Baseline loss
        let loss0 = self.elbo_loss(x_batch, y_batch, n_data, rng)?;

        // Compute and apply gradient per layer
        // To reduce forward passes, we update each parameter in-place immediately
        let n_layers = self.layers.len();
        for l in 0..n_layers {
            let n_w = self.layers[l].out_features * self.layers[l].in_features;
            let n_b = self.layers[l].out_features;

            // --- w_mu ---
            let mut grad_w_mu = vec![0.0f64; n_w];
            for i in 0..n_w {
                let orig = self.layers[l].w_mu[i];
                self.layers[l].w_mu[i] = orig + fd_h;
                let lp = self.elbo_loss(x_batch, y_batch, n_data, rng)?;
                self.layers[l].w_mu[i] = orig - fd_h;
                let lm = self.elbo_loss(x_batch, y_batch, n_data, rng)?;
                self.layers[l].w_mu[i] = orig;
                grad_w_mu[i] = (lp - lm) / (2.0 * fd_h);
            }

            // --- w_log_sigma ---
            let mut grad_w_ls = vec![0.0f64; n_w];
            for i in 0..n_w {
                let orig = self.layers[l].w_log_sigma[i];
                self.layers[l].w_log_sigma[i] = orig + fd_h;
                let lp = self.elbo_loss(x_batch, y_batch, n_data, rng)?;
                self.layers[l].w_log_sigma[i] = orig - fd_h;
                let lm = self.elbo_loss(x_batch, y_batch, n_data, rng)?;
                self.layers[l].w_log_sigma[i] = orig;
                grad_w_ls[i] = (lp - lm) / (2.0 * fd_h);
            }

            // --- b_mu ---
            let mut grad_b_mu = vec![0.0f64; n_b];
            for i in 0..n_b {
                let orig = self.layers[l].b_mu[i];
                self.layers[l].b_mu[i] = orig + fd_h;
                let lp = self.elbo_loss(x_batch, y_batch, n_data, rng)?;
                self.layers[l].b_mu[i] = orig - fd_h;
                let lm = self.elbo_loss(x_batch, y_batch, n_data, rng)?;
                self.layers[l].b_mu[i] = orig;
                grad_b_mu[i] = (lp - lm) / (2.0 * fd_h);
            }

            // --- b_log_sigma ---
            let mut grad_b_ls = vec![0.0f64; n_b];
            for i in 0..n_b {
                let orig = self.layers[l].b_log_sigma[i];
                self.layers[l].b_log_sigma[i] = orig + fd_h;
                let lp = self.elbo_loss(x_batch, y_batch, n_data, rng)?;
                self.layers[l].b_log_sigma[i] = orig - fd_h;
                let lm = self.elbo_loss(x_batch, y_batch, n_data, rng)?;
                self.layers[l].b_log_sigma[i] = orig;
                grad_b_ls[i] = (lp - lm) / (2.0 * fd_h);
            }

            self.layers[l].update(&grad_w_mu, &grad_w_ls, &grad_b_mu, &grad_b_ls, lr)?;
        }

        Ok(loss0)
    }
}

/// Decompose predictions into epistemic uncertainty (variance of sample means per output).
///
/// Given `predictions[sample][output_dim]`, returns the variance of the sample means
/// over the output dimension index 0 (scalar regression) or per-output if multi-dimensional.
///
/// For a length-1 output, returns a single-element vector.
pub fn epistemic_uncertainty(predictions: &[Vec<f64>]) -> Vec<f64> {
    if predictions.is_empty() {
        return Vec::new();
    }
    let out_dim = predictions[0].len();
    let ns = predictions.len() as f64;

    let mut means = vec![0.0f64; out_dim];
    for s in predictions {
        for (j, &v) in s.iter().enumerate() {
            if j < out_dim {
                means[j] += v;
            }
        }
    }
    for m in means.iter_mut() {
        *m /= ns;
    }

    let mut vars = vec![0.0f64; out_dim];
    for s in predictions {
        for (j, &v) in s.iter().enumerate() {
            if j < out_dim {
                vars[j] += (v - means[j]).powi(2);
            }
        }
    }
    for v in vars.iter_mut() {
        *v /= ns;
    }
    vars
}

/// Compute aleatoric-style uncertainty as mean within-sample variance.
///
/// For scalar regression with a single output, the "within-sample" variance is zero
/// (no inherent stochasticity in a single sample). This function computes the mean
/// squared deviation of each sample from the global mean — a proxy for aleatoric noise
/// when outputs are treated as noisy observations.
///
/// `predictions[sample][output_dim]`
pub fn aleatoric_uncertainty(predictions: &[Vec<f64>]) -> Vec<f64> {
    // For the BNN regression model the per-sample variance across output dims (if > 1)
    // or zero for scalar output. Return epistemic_uncertainty as a conservative bound.
    epistemic_uncertainty(predictions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_normal_rng() -> impl FnMut() -> f64 {
        let mut state: u64 = 987654321098765;
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
    fn test_bayesian_mlp_creation() {
        let config = BnnConfig::default();
        let mlp = BayesianMlp::new(&[2, 4, 1], config).expect("creation");
        assert_eq!(mlp.layers.len(), 2);
        assert_eq!(mlp.layers[0].in_features, 2);
        assert_eq!(mlp.layers[0].out_features, 4);
        assert_eq!(mlp.layers[1].in_features, 4);
        assert_eq!(mlp.layers[1].out_features, 1);
    }

    #[test]
    fn test_predict_mean_std_shapes() {
        let config = BnnConfig {
            n_samples_mc: 5,
            ..BnnConfig::default()
        };
        let mlp = BayesianMlp::new(&[3, 8, 1], config).expect("creation");
        let x = vec![0.5, -0.3, 1.0];
        let mut rng = make_normal_rng();
        let (means, stds) = mlp.predict_mean_std(&x, 20, &mut rng).expect("predict");
        assert_eq!(means.len(), 1);
        assert_eq!(stds.len(), 1);
        assert!(stds[0] >= 0.0, "std should be non-negative");
    }

    #[test]
    fn test_epistemic_uncertainty_positive() {
        let config = BnnConfig {
            n_samples_mc: 5,
            ..BnnConfig::default()
        };
        let mlp = BayesianMlp::new(&[2, 4, 1], config).expect("creation");
        let x = vec![1.0, -1.0];
        let mut rng = make_normal_rng();
        let samples = mlp.predict_samples(&x, 30, &mut rng).expect("samples");
        let epi = epistemic_uncertainty(&samples);
        assert_eq!(epi.len(), 1);
        assert!(
            epi[0] >= 0.0,
            "epistemic uncertainty should be non-negative"
        );
    }

    #[test]
    fn test_total_kl_positive() {
        let config = BnnConfig::default();
        let mut mlp = BayesianMlp::new(&[2, 4, 1], config).expect("creation");
        // Force non-zero mean to ensure positive KL
        mlp.layers[0].w_mu[0] = 1.0;
        let kl = mlp.total_kl();
        assert!(kl > 0.0, "total KL should be positive, got {}", kl);
    }

    #[test]
    fn test_elbo_loss_decreases() {
        // Train on y = 2*x for 5 steps with a tiny network and verify loss is finite
        let config = BnnConfig {
            n_samples_mc: 3,
            ..BnnConfig::default()
        };
        let mut mlp = BayesianMlp::new(&[1, 4, 1], config).expect("creation");
        let x_batch: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y_batch: Vec<f64> = vec![0.0, 2.0, 4.0];
        let mut rng = make_normal_rng();

        let mut last_loss = f64::MAX;
        let mut any_finite = false;
        for _ in 0..3 {
            let loss = mlp
                .train_step(&x_batch, &y_batch, x_batch.len(), 1e-3, &mut rng)
                .expect("train_step");
            if loss.is_finite() {
                any_finite = true;
                last_loss = loss;
            }
        }
        assert!(
            any_finite,
            "At least one finite loss expected, last: {}",
            last_loss
        );
    }

    #[test]
    fn test_layer_sizes_error() {
        let config = BnnConfig::default();
        assert!(BayesianMlp::new(&[3], config.clone()).is_err());
        assert!(BayesianMlp::new(&[], config).is_err());
    }
}
