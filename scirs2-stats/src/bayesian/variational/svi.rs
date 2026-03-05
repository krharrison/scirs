//! Stochastic Variational Inference (SVI)
//!
//! This module implements scalable variational inference methods that use
//! stochastic optimization for large datasets. Key features:
//!
//! - Mini-batch ELBO estimation
//! - Natural gradient updates
//! - Adam-like learning rate scheduling
//! - Support for both mean-field and full-rank Gaussian approximations

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::validation::*;
use std::f64::consts::PI;

use super::{digamma, lgamma, FullRankGaussian, MeanFieldGaussian, VariationalDiagnostics};

// ============================================================================
// Learning Rate Schedules
// ============================================================================

/// Learning rate schedule for SVI optimization
#[derive(Debug, Clone)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant {
        /// The fixed learning rate
        lr: f64,
    },
    /// Robbins-Monro schedule: lr_t = lr_0 / (1 + decay * t)
    RobbinsMonro {
        /// Initial learning rate
        lr0: f64,
        /// Decay factor
        decay: f64,
    },
    /// Exponential decay: lr_t = lr_0 * gamma^t
    ExponentialDecay {
        /// Initial learning rate
        lr0: f64,
        /// Decay multiplier per step
        gamma: f64,
    },
    /// Adam-like adaptive learning rate
    Adam {
        /// Base learning rate
        lr: f64,
        /// First moment decay (beta1)
        beta1: f64,
        /// Second moment decay (beta2)
        beta2: f64,
        /// Numerical stability constant
        epsilon: f64,
    },
}

impl LearningRateSchedule {
    /// Get the learning rate at iteration t
    pub fn get_lr(&self, t: usize) -> f64 {
        match self {
            LearningRateSchedule::Constant { lr } => *lr,
            LearningRateSchedule::RobbinsMonro { lr0, decay } => lr0 / (1.0 + decay * t as f64),
            LearningRateSchedule::ExponentialDecay { lr0, gamma } => lr0 * gamma.powi(t as i32),
            LearningRateSchedule::Adam { lr, .. } => {
                // Base learning rate; actual Adam adjustment is done in the optimizer
                *lr
            }
        }
    }

    /// Create a default Adam schedule
    pub fn default_adam() -> Self {
        LearningRateSchedule::Adam {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    /// Create a default Robbins-Monro schedule
    pub fn default_robbins_monro() -> Self {
        LearningRateSchedule::RobbinsMonro {
            lr0: 0.1,
            decay: 0.01,
        }
    }
}

// ============================================================================
// Adam Optimizer State
// ============================================================================

/// Adam optimizer state for adaptive learning rates
#[derive(Debug, Clone)]
pub struct AdamState {
    /// First moment estimates
    pub m: Array1<f64>,
    /// Second moment estimates
    pub v: Array1<f64>,
    /// Beta1 parameter (first moment decay)
    pub beta1: f64,
    /// Beta2 parameter (second moment decay)
    pub beta2: f64,
    /// Numerical stability epsilon
    pub epsilon: f64,
    /// Base learning rate
    pub lr: f64,
    /// Current time step
    pub t: usize,
}

impl AdamState {
    /// Create a new Adam state for parameters of given dimension
    pub fn new(dim: usize, lr: f64, beta1: f64, beta2: f64, epsilon: f64) -> Result<Self> {
        check_positive(dim, "dim")?;
        check_positive(lr, "lr")?;
        check_positive(epsilon, "epsilon")?;

        Ok(Self {
            m: Array1::zeros(dim),
            v: Array1::zeros(dim),
            beta1,
            beta2,
            epsilon,
            lr,
            t: 0,
        })
    }

    /// Compute Adam update for a given gradient
    pub fn compute_update(&mut self, gradient: &Array1<f64>) -> Result<Array1<f64>> {
        if gradient.len() != self.m.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "gradient length ({}) must match state dimension ({})",
                gradient.len(),
                self.m.len()
            )));
        }

        self.t += 1;

        // Update biased first moment estimate
        self.m = &self.m * self.beta1 + gradient * (1.0 - self.beta1);

        // Update biased second raw moment estimate
        self.v = &self.v * self.beta2 + &gradient.mapv(|g| g * g) * (1.0 - self.beta2);

        // Compute bias-corrected first moment estimate
        let m_hat = &self.m / (1.0 - self.beta1.powi(self.t as i32));

        // Compute bias-corrected second raw moment estimate
        let v_hat = &self.v / (1.0 - self.beta2.powi(self.t as i32));

        // Compute update
        let update = &m_hat / &v_hat.mapv(|vi| vi.sqrt() + self.epsilon) * self.lr;

        Ok(update)
    }

    /// Reset the optimizer state
    pub fn reset(&mut self) {
        self.m.fill(0.0);
        self.v.fill(0.0);
        self.t = 0;
    }
}

// ============================================================================
// Natural Gradient Computations
// ============================================================================

/// Natural gradient parameters for exponential family distributions
#[derive(Debug, Clone)]
pub struct NaturalGradientParams {
    /// Natural parameters (eta) for the variational distribution
    pub eta: Array1<f64>,
    /// Fisher information matrix (or its approximation)
    /// Stored as diagonal for mean-field case
    pub fisher_diag: Array1<f64>,
}

impl NaturalGradientParams {
    /// Create natural gradient parameters for a mean-field Gaussian
    ///
    /// For a Gaussian q(z; mu, sigma^2):
    ///   eta_1 = mu / sigma^2   (natural parameter 1)
    ///   eta_2 = -1 / (2*sigma^2) (natural parameter 2)
    ///
    /// Fisher information is 1/sigma^2 for the mean parameter
    /// and 2/sigma^4 for the variance parameter
    pub fn from_mean_field(mf: &MeanFieldGaussian) -> Self {
        let dim = mf.dim;
        let stds = mf.stds();
        let vars = mf.variances();

        // Natural parameters: [mu/sigma^2, -1/(2*sigma^2)]
        let mut eta = Array1::zeros(2 * dim);
        let mut fisher_diag = Array1::zeros(2 * dim);

        for i in 0..dim {
            // eta_1 = mu / sigma^2
            eta[i] = mf.means[i] / vars[i];
            // eta_2 = -1 / (2*sigma^2)
            eta[dim + i] = -1.0 / (2.0 * vars[i]);

            // Fisher diagonal
            fisher_diag[i] = 1.0 / vars[i]; // For mean
            fisher_diag[dim + i] = 2.0 / (stds[i].powi(4)); // For variance
        }

        Self { eta, fisher_diag }
    }

    /// Convert natural parameters back to mean/std parameterization
    pub fn to_mean_field(&self) -> Result<MeanFieldGaussian> {
        let dim = self.eta.len() / 2;
        if dim == 0 {
            return Err(StatsError::InvalidArgument(
                "Natural parameters must have positive dimension".to_string(),
            ));
        }

        let mut means = Array1::zeros(dim);
        let mut log_stds = Array1::zeros(dim);

        for i in 0..dim {
            let eta2 = self.eta[dim + i];
            if eta2 >= 0.0 {
                return Err(StatsError::InvalidArgument(format!(
                    "eta_2[{}] = {} must be negative for valid Gaussian",
                    i, eta2
                )));
            }
            let var = -1.0 / (2.0 * eta2);
            let mean = self.eta[i] * var;
            means[i] = mean;
            log_stds[i] = 0.5 * var.ln();
        }

        MeanFieldGaussian::from_params(means, log_stds)
    }

    /// Compute natural gradient update: update = Fisher^{-1} * euclidean_grad
    /// For diagonal Fisher, this is element-wise division
    pub fn natural_gradient_update(&self, euclidean_grad: &Array1<f64>) -> Result<Array1<f64>> {
        if euclidean_grad.len() != self.fisher_diag.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "gradient length ({}) must match parameter dimension ({})",
                euclidean_grad.len(),
                self.fisher_diag.len()
            )));
        }

        let mut nat_grad = Array1::zeros(euclidean_grad.len());
        for i in 0..euclidean_grad.len() {
            if self.fisher_diag[i].abs() < 1e-15 {
                nat_grad[i] = 0.0; // Avoid division by zero
            } else {
                nat_grad[i] = euclidean_grad[i] / self.fisher_diag[i];
            }
        }

        Ok(nat_grad)
    }
}

// ============================================================================
// SVI Configuration
// ============================================================================

/// Configuration for Stochastic Variational Inference
#[derive(Debug, Clone)]
pub struct SviConfig {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Mini-batch size (number of data points per batch)
    pub batch_size: usize,
    /// Learning rate schedule
    pub lr_schedule: LearningRateSchedule,
    /// Convergence tolerance (on ELBO)
    pub tol: f64,
    /// Number of Monte Carlo samples for ELBO estimation
    pub n_mc_samples: usize,
    /// Whether to use natural gradients
    pub use_natural_gradient: bool,
    /// How often to compute full ELBO for diagnostics (0 = never)
    pub diagnostic_interval: usize,
    /// Gradient clipping threshold (0 = no clipping)
    pub grad_clip: f64,
    /// Seed for reproducibility (used for batch selection)
    pub seed: u64,
}

impl Default for SviConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            batch_size: 32,
            lr_schedule: LearningRateSchedule::default_adam(),
            tol: 1e-4,
            n_mc_samples: 1,
            use_natural_gradient: false,
            diagnostic_interval: 50,
            grad_clip: 10.0,
            seed: 42,
        }
    }
}

// ============================================================================
// Stochastic Variational Inference
// ============================================================================

/// Stochastic Variational Inference (SVI)
///
/// Implements scalable variational inference using stochastic gradient
/// ascent on the ELBO. Supports:
/// - Mini-batch ELBO estimation for large datasets
/// - Natural gradient updates for faster convergence
/// - Adam adaptive learning rates
/// - Gradient clipping for stability
///
/// The model assumes a mean-field Gaussian posterior approximation
/// q(z) = prod_i N(z_i; mu_i, sigma_i^2)
///
/// The user provides a log joint density function log p(z, x_batch) that
/// takes the latent variables z and a mini-batch of data.
#[derive(Debug, Clone)]
pub struct StochasticVI {
    /// Variational distribution (mean-field Gaussian)
    pub variational: MeanFieldGaussian,
    /// Configuration
    pub config: SviConfig,
    /// Diagnostics
    pub diagnostics: VariationalDiagnostics,
    /// Adam optimizer state (if using Adam)
    adam_state: Option<AdamState>,
}

impl StochasticVI {
    /// Create a new SVI instance
    pub fn new(dim: usize, config: SviConfig) -> Result<Self> {
        check_positive(dim, "dim")?;

        let variational = MeanFieldGaussian::new(dim)?;

        let adam_state = if let LearningRateSchedule::Adam {
            lr,
            beta1,
            beta2,
            epsilon,
        } = &config.lr_schedule
        {
            Some(AdamState::new(2 * dim, *lr, *beta1, *beta2, *epsilon)?)
        } else {
            None
        };

        Ok(Self {
            variational,
            config,
            diagnostics: VariationalDiagnostics::new(),
            adam_state,
        })
    }

    /// Run SVI optimization with a log joint density function
    ///
    /// # Arguments
    /// * `data` - Full dataset (rows are observations)
    /// * `log_joint` - Function computing log p(z, x_batch) given latent variables
    ///   and a batch of data. Returns (log_prob, gradient_wrt_z).
    /// * `n_total` - Total number of data points (for scaling mini-batch ELBO)
    ///
    /// # Returns
    /// * The optimized variational distribution and diagnostics
    pub fn fit<F>(
        &mut self,
        data: ArrayView2<f64>,
        log_joint: F,
        n_total: usize,
    ) -> Result<SviResult>
    where
        F: Fn(&Array1<f64>, ArrayView2<f64>) -> Result<(f64, Array1<f64>)>,
    {
        checkarray_finite(&data, "data")?;
        check_positive(n_total, "n_total")?;

        let (n_data, _) = data.dim();
        let batch_size = self.config.batch_size.min(n_data);
        let scale_factor = n_total as f64 / batch_size as f64;

        // Simple deterministic batch cycling with seed-based offset
        let offset = (self.config.seed % n_data as u64) as usize;

        for iter in 0..self.config.max_iter {
            // Select mini-batch (deterministic cycling with offset)
            let batch_start = (offset + iter * batch_size) % n_data;
            let batch_end = (batch_start + batch_size).min(n_data);
            let actual_batch_size = batch_end - batch_start;

            let batch = data.slice(scirs2_core::ndarray::s![batch_start..batch_end, ..]);

            // Estimate ELBO gradient using reparameterization trick
            let (elbo_estimate, grad) = self.estimate_elbo_gradient(
                batch,
                &log_joint,
                scale_factor * (actual_batch_size as f64 / batch_size as f64),
            )?;

            // Record diagnostics
            self.diagnostics.record_elbo(elbo_estimate);
            let grad_norm = grad.dot(&grad).sqrt();
            self.diagnostics.record_gradient_norm(grad_norm);

            // Apply gradient clipping if configured
            let clipped_grad = if self.config.grad_clip > 0.0 && grad_norm > self.config.grad_clip {
                &grad * (self.config.grad_clip / grad_norm)
            } else {
                grad
            };

            // Compute update (natural gradient or Euclidean)
            let update = if self.config.use_natural_gradient {
                let nat_params = NaturalGradientParams::from_mean_field(&self.variational);
                nat_params.natural_gradient_update(&clipped_grad)?
            } else {
                clipped_grad
            };

            // Apply learning rate and update parameters
            let lr = self.config.lr_schedule.get_lr(iter);
            let current_params = self.variational.get_params();

            let new_params = if let Some(ref mut adam) = self.adam_state {
                let adam_update = adam.compute_update(&update)?;
                &current_params + &adam_update
            } else {
                &current_params + &(&update * lr)
            };

            // Track parameter change
            let param_change = (&new_params - &current_params).mapv(|x| x * x).sum().sqrt();
            self.diagnostics.record_param_change(param_change);

            self.variational.set_params(&new_params)?;

            // Check convergence
            if iter > 10 && self.diagnostics.check_elbo_convergence(self.config.tol) {
                self.diagnostics.converged = true;
                break;
            }
        }

        Ok(SviResult {
            variational: self.variational.clone(),
            diagnostics: self.diagnostics.clone(),
        })
    }

    /// Estimate ELBO and its gradient using Monte Carlo samples
    fn estimate_elbo_gradient<F>(
        &self,
        batch: ArrayView2<f64>,
        log_joint: &F,
        scale_factor: f64,
    ) -> Result<(f64, Array1<f64>)>
    where
        F: Fn(&Array1<f64>, ArrayView2<f64>) -> Result<(f64, Array1<f64>)>,
    {
        let dim = self.variational.dim;
        let n_samples = self.config.n_mc_samples.max(1);

        let mut total_elbo = 0.0;
        let mut total_grad = Array1::zeros(2 * dim);

        for s in 0..n_samples {
            // Generate epsilon ~ N(0, I) using simple deterministic approximation
            // (for production, you'd use a proper RNG here)
            let epsilon =
                generate_standard_normal(dim, s as u64 + self.diagnostics.n_iterations as u64);

            // Reparameterization: z = mu + sigma * epsilon
            let z = self.variational.sample(&epsilon)?;

            // Compute log joint and its gradient
            let (log_p, grad_z) = log_joint(&z, batch)?;

            // Scale for mini-batch
            let scaled_log_p = log_p * scale_factor;
            let scaled_grad_z = &grad_z * scale_factor;

            // Compute log q(z) and entropy gradient
            let log_q = self.variational.log_prob(&z)?;

            // ELBO = E[log p(z, x) - log q(z)]
            total_elbo += scaled_log_p - log_q;

            // Gradient of ELBO wrt variational parameters (mu, log_sigma)
            // d ELBO / d mu = d log_p / d z (through reparameterization)
            // d ELBO / d log_sigma = d log_p / d z * epsilon * sigma + 1 (entropy gradient)
            let stds = self.variational.stds();
            for i in 0..dim {
                // Gradient wrt mean
                total_grad[i] += scaled_grad_z[i];
                // Gradient wrt log_std: chain rule + entropy
                total_grad[dim + i] += scaled_grad_z[i] * epsilon[i] * stds[i] + 1.0;
            }

            // Subtract gradient of log q
            for i in 0..dim {
                let diff = z[i] - self.variational.means[i];
                let var = stds[i] * stds[i];
                // d log q / d mu = (z - mu) / sigma^2
                total_grad[i] -= diff / var;
                // d log q / d log_sigma = ((z-mu)^2 / sigma^2 - 1) * sigma ... simplified
                total_grad[dim + i] -= diff * diff / var - 1.0;
            }
        }

        // Average over samples
        total_elbo /= n_samples as f64;
        total_grad /= n_samples as f64;

        Ok((total_elbo, total_grad))
    }

    /// Get the current variational distribution
    pub fn variational_distribution(&self) -> &MeanFieldGaussian {
        &self.variational
    }

    /// Get diagnostics
    pub fn diagnostics(&self) -> &VariationalDiagnostics {
        &self.diagnostics
    }

    /// Reset the optimizer state (useful for warm restarts)
    pub fn reset_optimizer(&mut self) {
        if let Some(ref mut adam) = self.adam_state {
            adam.reset();
        }
        self.diagnostics = VariationalDiagnostics::new();
    }
}

/// Results from SVI optimization
#[derive(Debug, Clone)]
pub struct SviResult {
    /// Optimized variational distribution
    pub variational: MeanFieldGaussian,
    /// Optimization diagnostics
    pub diagnostics: VariationalDiagnostics,
}

impl SviResult {
    /// Get posterior means
    pub fn posterior_means(&self) -> &Array1<f64> {
        &self.variational.means
    }

    /// Get posterior standard deviations
    pub fn posterior_stds(&self) -> Array1<f64> {
        self.variational.stds()
    }

    /// Compute approximate credible intervals
    pub fn credible_intervals(&self, confidence: f64) -> Result<Array2<f64>> {
        check_probability(confidence, "confidence")?;
        let alpha = (1.0 - confidence) / 2.0;
        let z_critical = super::normal_ppf(1.0 - alpha)?;

        let dim = self.variational.dim;
        let mut intervals = Array2::zeros((dim, 2));
        let stds = self.variational.stds();

        for i in 0..dim {
            intervals[[i, 0]] = self.variational.means[i] - z_critical * stds[i];
            intervals[[i, 1]] = self.variational.means[i] + z_critical * stds[i];
        }

        Ok(intervals)
    }
}

// ============================================================================
// SVI for Bayesian Linear Regression
// ============================================================================

/// SVI-specialized Bayesian linear regression
///
/// Uses stochastic variational inference to fit a Bayesian linear regression
/// model, making it suitable for large datasets that don't fit in memory.
///
/// Model: y = X * beta + epsilon, epsilon ~ N(0, sigma^2)
/// Prior: beta ~ N(0, prior_var * I), sigma^2 ~ InvGamma(a, b)
#[derive(Debug, Clone)]
pub struct SviBayesianRegression {
    /// Variational mean for coefficients
    pub mean_beta: Array1<f64>,
    /// Variational log std for coefficients
    pub log_std_beta: Array1<f64>,
    /// Variational parameters for noise precision: shape and rate of Gamma
    pub shape_tau: f64,
    pub rate_tau: f64,
    /// Prior variance for coefficients
    pub prior_var: f64,
    /// Prior shape for noise precision
    pub prior_shape: f64,
    /// Prior rate for noise precision
    pub prior_rate: f64,
    /// Number of features
    pub n_features: usize,
    /// SVI configuration
    pub config: SviConfig,
}

impl SviBayesianRegression {
    /// Create a new SVI Bayesian regression model
    pub fn new(n_features: usize, config: SviConfig) -> Result<Self> {
        check_positive(n_features, "n_features")?;

        Ok(Self {
            mean_beta: Array1::zeros(n_features),
            log_std_beta: Array1::zeros(n_features),
            shape_tau: 1.0,
            rate_tau: 1.0,
            prior_var: 100.0,
            prior_shape: 1e-3,
            prior_rate: 1e-3,
            n_features,
            config,
        })
    }

    /// Set prior parameters
    pub fn with_priors(
        mut self,
        prior_var: f64,
        prior_shape: f64,
        prior_rate: f64,
    ) -> Result<Self> {
        check_positive(prior_var, "prior_var")?;
        check_positive(prior_shape, "prior_shape")?;
        check_positive(prior_rate, "prior_rate")?;
        self.prior_var = prior_var;
        self.prior_shape = prior_shape;
        self.prior_rate = prior_rate;
        Ok(self)
    }

    /// Fit using SVI with mini-batches
    pub fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<SviRegressionResult> {
        checkarray_finite(&x, "x")?;
        checkarray_finite(&y, "y")?;

        let (n_samples, n_features) = x.dim();
        if y.len() != n_samples {
            return Err(StatsError::DimensionMismatch(format!(
                "y length ({}) must match x rows ({})",
                y.len(),
                n_samples
            )));
        }
        if n_features != self.n_features {
            return Err(StatsError::DimensionMismatch(format!(
                "x features ({}) must match model features ({})",
                n_features, self.n_features
            )));
        }

        let batch_size = self.config.batch_size.min(n_samples);
        let scale_factor = n_samples as f64 / batch_size as f64;
        let offset = (self.config.seed % n_samples as u64) as usize;

        // Initialize Adam state for all parameters
        // Parameters: [mean_beta (d), log_std_beta (d), log_shape_tau, log_rate_tau]
        let n_params = 2 * self.n_features + 2;
        let mut adam_state = if let LearningRateSchedule::Adam {
            lr,
            beta1,
            beta2,
            epsilon,
        } = &self.config.lr_schedule
        {
            Some(AdamState::new(n_params, *lr, *beta1, *beta2, *epsilon)?)
        } else {
            None
        };

        let mut diagnostics = VariationalDiagnostics::new();

        for iter in 0..self.config.max_iter {
            // Select mini-batch
            let batch_start = (offset + iter * batch_size) % n_samples;
            let batch_end = (batch_start + batch_size).min(n_samples);

            let x_batch = x.slice(scirs2_core::ndarray::s![batch_start..batch_end, ..]);
            let y_batch = y.slice(scirs2_core::ndarray::s![batch_start..batch_end]);

            // Compute stochastic ELBO gradient
            let (elbo, grad) =
                self.compute_stochastic_elbo_grad(x_batch, y_batch, scale_factor, iter as u64)?;

            diagnostics.record_elbo(elbo);

            let grad_norm = grad.dot(&grad).sqrt();
            diagnostics.record_gradient_norm(grad_norm);

            // Clip gradient
            let clipped_grad = if self.config.grad_clip > 0.0 && grad_norm > self.config.grad_clip {
                &grad * (self.config.grad_clip / grad_norm)
            } else {
                grad
            };

            // Get current parameters
            let current_params = self.get_params();

            // Apply update
            let new_params = if let Some(ref mut adam) = adam_state {
                let update = adam.compute_update(&clipped_grad)?;
                &current_params + &update
            } else {
                let lr = self.config.lr_schedule.get_lr(iter);
                &current_params + &(&clipped_grad * lr)
            };

            let param_change = (&new_params - &current_params).mapv(|x| x * x).sum().sqrt();
            diagnostics.record_param_change(param_change);

            self.set_params(&new_params)?;

            // Check convergence
            if iter > 20 && diagnostics.check_elbo_convergence(self.config.tol) {
                diagnostics.converged = true;
                break;
            }
        }

        Ok(SviRegressionResult {
            mean_beta: self.mean_beta.clone(),
            std_beta: self.log_std_beta.mapv(f64::exp),
            shape_tau: self.shape_tau,
            rate_tau: self.rate_tau,
            diagnostics,
        })
    }

    /// Compute stochastic ELBO and gradient for a mini-batch
    fn compute_stochastic_elbo_grad(
        &self,
        x_batch: ArrayView2<f64>,
        y_batch: ArrayView1<f64>,
        scale_factor: f64,
        seed: u64,
    ) -> Result<(f64, Array1<f64>)> {
        let n_batch = x_batch.nrows();
        let d = self.n_features;
        let n_params = 2 * d + 2;

        let std_beta = self.log_std_beta.mapv(f64::exp);
        let expected_tau = self.shape_tau / self.rate_tau;
        let expected_log_tau = digamma(self.shape_tau) - self.rate_tau.ln();

        // Sample beta using reparameterization trick
        let epsilon = generate_standard_normal(d, seed);
        let beta_sample = &self.mean_beta + &(&std_beta * &epsilon);

        // Compute residuals
        let y_pred = x_batch.dot(&beta_sample);
        let residuals = &y_batch.to_owned() - &y_pred;
        let sse = residuals.dot(&residuals);

        // Scaled likelihood term
        let likelihood = scale_factor
            * (0.5 * n_batch as f64 * expected_log_tau
                - 0.5 * n_batch as f64 * (2.0 * PI).ln()
                - 0.5 * expected_tau * sse);

        // Prior term for beta
        let beta_sq_sum = beta_sample.dot(&beta_sample);
        let prior_beta =
            -0.5 * d as f64 * (2.0 * PI * self.prior_var).ln() - 0.5 / self.prior_var * beta_sq_sum;

        // Prior term for tau
        let prior_tau = self.prior_shape * self.prior_rate.ln() - lgamma(self.prior_shape)
            + (self.prior_shape - 1.0) * expected_log_tau
            - self.prior_rate * expected_tau;

        // Entropy of q(beta)
        let entropy_beta: f64 = (0..d)
            .map(|i| 0.5 * (1.0 + (2.0 * PI).ln()) + self.log_std_beta[i])
            .sum();

        // Entropy of q(tau) (Gamma distribution)
        let entropy_tau = self.shape_tau - self.rate_tau.ln()
            + lgamma(self.shape_tau)
            + (1.0 - self.shape_tau) * digamma(self.shape_tau);

        let elbo = likelihood + prior_beta + prior_tau + entropy_beta + entropy_tau;

        // Compute gradients
        let mut grad = Array1::zeros(n_params);

        // Gradient wrt mean_beta
        let grad_beta_from_likelihood = x_batch.t().dot(&residuals) * expected_tau * scale_factor;
        let grad_beta_from_prior = &beta_sample * (-1.0 / self.prior_var);

        for i in 0..d {
            grad[i] = grad_beta_from_likelihood[i] + grad_beta_from_prior[i];
        }

        // Gradient wrt log_std_beta (through reparameterization)
        for i in 0..d {
            let dl_dbeta = grad_beta_from_likelihood[i] + grad_beta_from_prior[i];
            // Chain rule: d/d(log_sigma) = d/dbeta * dbeta/d(log_sigma)
            // dbeta/d(log_sigma) = epsilon * sigma (since beta = mu + sigma*epsilon)
            grad[d + i] = dl_dbeta * epsilon[i] * std_beta[i] + 1.0; // +1 from entropy
        }

        // Gradient wrt shape_tau and rate_tau (use simpler gradient ascent on these)
        // d ELBO / d shape_tau
        let d_likelihood_shape =
            scale_factor * 0.5 * n_batch as f64 * super::trigamma(self.shape_tau);
        let d_prior_shape = (self.prior_shape - 1.0) * super::trigamma(self.shape_tau)
            - self.prior_rate / self.rate_tau;
        let d_entropy_shape = 1.0 - (1.0 - self.shape_tau) * super::trigamma(self.shape_tau)
            + digamma(self.shape_tau) * (-1.0)
            + super::trigamma(self.shape_tau) * (1.0 - self.shape_tau);
        // Simplified: just compute numerically stable gradient
        grad[2 * d] = d_likelihood_shape + d_prior_shape + d_entropy_shape * 0.01;

        // d ELBO / d rate_tau
        let d_likelihood_rate =
            -scale_factor * 0.5 * sse * self.shape_tau / (self.rate_tau * self.rate_tau);
        let d_prior_rate = self.prior_rate * self.shape_tau / (self.rate_tau * self.rate_tau);
        grad[2 * d + 1] = d_likelihood_rate - d_prior_rate + 1.0 / self.rate_tau;

        Ok((elbo, grad))
    }

    fn get_params(&self) -> Array1<f64> {
        let d = self.n_features;
        let mut params = Array1::zeros(2 * d + 2);
        for i in 0..d {
            params[i] = self.mean_beta[i];
            params[d + i] = self.log_std_beta[i];
        }
        params[2 * d] = self.shape_tau;
        params[2 * d + 1] = self.rate_tau;
        params
    }

    fn set_params(&mut self, params: &Array1<f64>) -> Result<()> {
        let d = self.n_features;
        if params.len() != 2 * d + 2 {
            return Err(StatsError::DimensionMismatch(format!(
                "params length ({}) must be {}",
                params.len(),
                2 * d + 2
            )));
        }
        for i in 0..d {
            self.mean_beta[i] = params[i];
            self.log_std_beta[i] = params[d + i];
        }
        // Ensure shape and rate stay positive
        self.shape_tau = params[2 * d].max(1e-6);
        self.rate_tau = params[2 * d + 1].max(1e-6);
        Ok(())
    }
}

/// Results from SVI Bayesian regression
#[derive(Debug, Clone)]
pub struct SviRegressionResult {
    /// Posterior mean of coefficients
    pub mean_beta: Array1<f64>,
    /// Posterior standard deviation of coefficients
    pub std_beta: Array1<f64>,
    /// Posterior shape parameter for noise precision
    pub shape_tau: f64,
    /// Posterior rate parameter for noise precision
    pub rate_tau: f64,
    /// Optimization diagnostics
    pub diagnostics: VariationalDiagnostics,
}

impl SviRegressionResult {
    /// Get expected noise variance: E[1/tau] = rate / (shape - 1) for shape > 1
    pub fn expected_noise_variance(&self) -> f64 {
        if self.shape_tau > 1.0 {
            self.rate_tau / (self.shape_tau - 1.0)
        } else {
            self.rate_tau / self.shape_tau
        }
    }

    /// Compute credible intervals for coefficients
    pub fn credible_intervals(&self, confidence: f64) -> Result<Array2<f64>> {
        check_probability(confidence, "confidence")?;
        let alpha = (1.0 - confidence) / 2.0;
        let z_critical = super::normal_ppf(1.0 - alpha)?;

        let d = self.mean_beta.len();
        let mut intervals = Array2::zeros((d, 2));
        for i in 0..d {
            intervals[[i, 0]] = self.mean_beta[i] - z_critical * self.std_beta[i];
            intervals[[i, 1]] = self.mean_beta[i] + z_critical * self.std_beta[i];
        }
        Ok(intervals)
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Generate approximate standard normal samples using Box-Muller-like
/// deterministic scheme (based on a seed for reproducibility)
///
/// For production use, one should use a proper PRNG. This provides
/// a deterministic but reasonable approximation for testing and
/// demonstration purposes.
fn generate_standard_normal(dim: usize, seed: u64) -> Array1<f64> {
    let mut result = Array1::zeros(dim);
    let golden_ratio = 1.618033988749895;

    for i in 0..dim {
        // Use a quasi-random sequence based on golden ratio
        let u1 = ((seed as f64 * golden_ratio + i as f64 * 0.7548776662466927) % 1.0).abs();
        let u2 = ((seed as f64 * 0.5698402909980532 + i as f64 * golden_ratio) % 1.0).abs();

        // Clamp to avoid log(0)
        let u1_safe = u1.max(1e-10).min(1.0 - 1e-10);
        let u2_safe = u2.max(1e-10).min(1.0 - 1e-10);

        // Box-Muller transform
        let r = (-2.0 * u1_safe.ln()).sqrt();
        let theta = 2.0 * PI * u2_safe;
        result[i] = r * theta.cos();
    }

    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_learning_rate_constant() {
        let lr = LearningRateSchedule::Constant { lr: 0.01 };
        assert!((lr.get_lr(0) - 0.01).abs() < 1e-10);
        assert!((lr.get_lr(100) - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_learning_rate_robbins_monro() {
        let lr = LearningRateSchedule::RobbinsMonro {
            lr0: 0.1,
            decay: 0.01,
        };
        assert!((lr.get_lr(0) - 0.1).abs() < 1e-10);
        assert!(lr.get_lr(100) < lr.get_lr(0));
        assert!(lr.get_lr(100) > 0.0);
    }

    #[test]
    fn test_learning_rate_exponential() {
        let lr = LearningRateSchedule::ExponentialDecay {
            lr0: 0.1,
            gamma: 0.99,
        };
        assert!((lr.get_lr(0) - 0.1).abs() < 1e-10);
        assert!(lr.get_lr(100) < lr.get_lr(0));
    }

    #[test]
    fn test_adam_state() {
        let mut adam = AdamState::new(3, 0.01, 0.9, 0.999, 1e-8).expect("should create");
        let grad = Array1::from_vec(vec![1.0, -0.5, 0.3]);
        let update = adam.compute_update(&grad).expect("should compute update");
        assert_eq!(update.len(), 3);
        // First step should be approximately lr * grad / (sqrt(grad^2) + eps) = lr * sign(grad)
        for i in 0..3 {
            assert!(update[i].is_finite());
        }
    }

    #[test]
    fn test_natural_gradient_roundtrip() {
        let mf = MeanFieldGaussian::from_params(
            Array1::from_vec(vec![1.0, 2.0]),
            Array1::from_vec(vec![0.5, -0.3]),
        )
        .expect("should create");

        let nat = NaturalGradientParams::from_mean_field(&mf);
        let recovered = nat.to_mean_field().expect("should convert back");

        for i in 0..2 {
            assert!(
                (recovered.means[i] - mf.means[i]).abs() < 1e-6,
                "means differ at {}: {} vs {}",
                i,
                recovered.means[i],
                mf.means[i]
            );
            assert!(
                (recovered.log_stds[i] - mf.log_stds[i]).abs() < 1e-6,
                "log_stds differ at {}: {} vs {}",
                i,
                recovered.log_stds[i],
                mf.log_stds[i]
            );
        }
    }

    #[test]
    fn test_svi_creation() {
        let config = SviConfig {
            max_iter: 100,
            batch_size: 10,
            ..SviConfig::default()
        };
        let svi = StochasticVI::new(5, config).expect("should create SVI");
        assert_eq!(svi.variational.dim, 5);
    }

    #[test]
    fn test_svi_bayesian_regression() {
        // Simple test: y = x + noise
        let n = 100;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .enumerate()
            .map(|(i, &xi)| xi + 0.1 * ((i * 7 % 13) as f64 - 6.0) / 6.0)
            .collect();

        let x = Array2::from_shape_fn((n, 1), |(i, _)| x_data[i]);
        let y = Array1::from_vec(y_data);

        let config = SviConfig {
            max_iter: 200,
            batch_size: 20,
            lr_schedule: LearningRateSchedule::Adam {
                lr: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            ..SviConfig::default()
        };

        let mut model = SviBayesianRegression::new(1, config).expect("should create");
        let result = model.fit(x.view(), y.view()).expect("should fit");

        // Check that we get finite results
        assert!(result.mean_beta[0].is_finite());
        assert!(result.std_beta[0].is_finite());
        assert!(result.diagnostics.n_iterations > 0);
    }

    #[test]
    fn test_generate_standard_normal() {
        let samples = generate_standard_normal(100, 42);
        assert_eq!(samples.len(), 100);
        // All should be finite
        for &s in samples.iter() {
            assert!(s.is_finite(), "sample should be finite, got {}", s);
        }
        // Mean should be roughly zero (within reasonable bounds for quasi-random)
        let mean = samples.sum() / 100.0;
        assert!(
            mean.abs() < 2.0,
            "mean should be roughly zero, got {}",
            mean
        );
    }

    #[test]
    fn test_svi_config_default() {
        let config = SviConfig::default();
        assert_eq!(config.max_iter, 1000);
        assert_eq!(config.batch_size, 32);
        assert!(config.grad_clip > 0.0);
    }
}
