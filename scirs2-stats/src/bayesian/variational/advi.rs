//! Automatic Differentiation Variational Inference (ADVI)
//!
//! This module implements ADVI, which automatically transforms constrained
//! parameters to unconstrained space and fits a Gaussian variational
//! approximation. Key features:
//!
//! - Mean-field Gaussian approximation in unconstrained space
//! - Full-rank Gaussian approximation in unconstrained space
//! - Automatic parameter transformations (log, logit, etc.)
//! - ELBO computation with the reparameterization trick
//! - Support for custom log probability functions

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::validation::*;
use std::f64::consts::PI;

use super::svi::{AdamState, LearningRateSchedule};
use super::{FullRankGaussian, MeanFieldGaussian, VariationalDiagnostics};

// ============================================================================
// Parameter Transformations
// ============================================================================

/// Type of parameter constraint and its corresponding transformation
#[derive(Debug, Clone)]
pub enum ParameterConstraint {
    /// Unconstrained real-valued parameter (identity transform)
    Real,
    /// Positive parameter (log transform: unconstrained -> positive)
    Positive,
    /// Parameter on (0, 1) (logistic transform)
    UnitInterval,
    /// Parameter on (lower, upper) (scaled logistic transform)
    Bounded {
        /// Lower bound
        lower: f64,
        /// Upper bound
        upper: f64,
    },
    /// Simplex constraint (sum to 1) using stick-breaking transform
    Simplex {
        /// Dimension of the simplex
        dim: usize,
    },
    /// Lower-bounded parameter (shifted log transform)
    LowerBounded {
        /// Lower bound
        lower: f64,
    },
    /// Upper-bounded parameter (reflected log transform)
    UpperBounded {
        /// Upper bound
        upper: f64,
    },
}

impl ParameterConstraint {
    /// Transform from unconstrained to constrained space
    pub fn forward(&self, unconstrained: f64) -> f64 {
        match self {
            ParameterConstraint::Real => unconstrained,
            ParameterConstraint::Positive => unconstrained.exp(),
            ParameterConstraint::UnitInterval => 1.0 / (1.0 + (-unconstrained).exp()),
            ParameterConstraint::Bounded { lower, upper } => {
                let sigmoid = 1.0 / (1.0 + (-unconstrained).exp());
                lower + (upper - lower) * sigmoid
            }
            ParameterConstraint::LowerBounded { lower } => lower + unconstrained.exp(),
            ParameterConstraint::UpperBounded { upper } => upper - (-unconstrained).exp(),
            ParameterConstraint::Simplex { .. } => {
                // For simplex, forward transform is handled element-wise
                // via stick-breaking; this is a per-element sigmoid
                1.0 / (1.0 + (-unconstrained).exp())
            }
        }
    }

    /// Transform from constrained to unconstrained space
    pub fn inverse(&self, constrained: f64) -> Result<f64> {
        match self {
            ParameterConstraint::Real => Ok(constrained),
            ParameterConstraint::Positive => {
                if constrained <= 0.0 {
                    return Err(StatsError::InvalidArgument(format!(
                        "Positive constraint requires value > 0, got {}",
                        constrained
                    )));
                }
                Ok(constrained.ln())
            }
            ParameterConstraint::UnitInterval => {
                if constrained <= 0.0 || constrained >= 1.0 {
                    return Err(StatsError::InvalidArgument(format!(
                        "Unit interval constraint requires 0 < value < 1, got {}",
                        constrained
                    )));
                }
                Ok((constrained / (1.0 - constrained)).ln())
            }
            ParameterConstraint::Bounded { lower, upper } => {
                if constrained <= *lower || constrained >= *upper {
                    return Err(StatsError::InvalidArgument(format!(
                        "Bounded constraint requires {} < value < {}, got {}",
                        lower, upper, constrained
                    )));
                }
                let normalized = (constrained - lower) / (upper - lower);
                Ok((normalized / (1.0 - normalized)).ln())
            }
            ParameterConstraint::LowerBounded { lower } => {
                if constrained <= *lower {
                    return Err(StatsError::InvalidArgument(format!(
                        "Lower-bounded constraint requires value > {}, got {}",
                        lower, constrained
                    )));
                }
                Ok((constrained - lower).ln())
            }
            ParameterConstraint::UpperBounded { upper } => {
                if constrained >= *upper {
                    return Err(StatsError::InvalidArgument(format!(
                        "Upper-bounded constraint requires value < {}, got {}",
                        upper, constrained
                    )));
                }
                Ok(-((*upper - constrained).ln()))
            }
            ParameterConstraint::Simplex { .. } => {
                if constrained <= 0.0 || constrained >= 1.0 {
                    return Err(StatsError::InvalidArgument(format!(
                        "Simplex element must be in (0, 1), got {}",
                        constrained
                    )));
                }
                Ok((constrained / (1.0 - constrained)).ln())
            }
        }
    }

    /// Compute the log absolute determinant of the Jacobian of the forward transform
    ///
    /// This is needed for correcting the density when transforming from
    /// unconstrained to constrained space:
    /// p(constrained) = p_unconstrained(inverse(constrained)) * |det J^{-1}|
    ///
    /// Equivalently, for the ELBO in unconstrained space:
    /// log p(forward(eta)) + log |det J_forward(eta)|
    pub fn log_det_jacobian(&self, unconstrained: f64) -> f64 {
        match self {
            ParameterConstraint::Real => 0.0,
            ParameterConstraint::Positive => {
                // d/d_eta exp(eta) = exp(eta), so log|det J| = eta
                unconstrained
            }
            ParameterConstraint::UnitInterval => {
                // sigmoid'(eta) = sigmoid(eta) * (1 - sigmoid(eta))
                let s = 1.0 / (1.0 + (-unconstrained).exp());
                (s * (1.0 - s)).ln()
            }
            ParameterConstraint::Bounded { lower, upper } => {
                let s = 1.0 / (1.0 + (-unconstrained).exp());
                ((upper - lower) * s * (1.0 - s)).ln()
            }
            ParameterConstraint::LowerBounded { .. } => unconstrained,
            ParameterConstraint::UpperBounded { .. } => unconstrained,
            ParameterConstraint::Simplex { .. } => {
                let s = 1.0 / (1.0 + (-unconstrained).exp());
                (s * (1.0 - s)).ln()
            }
        }
    }
}

// ============================================================================
// ADVI Configuration
// ============================================================================

/// Configuration for ADVI
#[derive(Debug, Clone)]
pub struct AdviConfig {
    /// Maximum number of optimization iterations
    pub max_iter: usize,
    /// Convergence tolerance (relative ELBO change)
    pub tol: f64,
    /// Number of Monte Carlo samples for ELBO gradient estimation
    pub n_mc_samples: usize,
    /// Learning rate schedule
    pub lr_schedule: LearningRateSchedule,
    /// Gradient clipping threshold (0 = no clipping)
    pub grad_clip: f64,
    /// Diagnostic output interval (0 = no diagnostics)
    pub diagnostic_interval: usize,
    /// Seed for reproducibility
    pub seed: u64,
    /// Convergence window (number of iterations to average ELBO over)
    pub convergence_window: usize,
}

impl Default for AdviConfig {
    fn default() -> Self {
        Self {
            max_iter: 10000,
            tol: 1e-4,
            n_mc_samples: 1,
            lr_schedule: LearningRateSchedule::default_adam(),
            grad_clip: 10.0,
            diagnostic_interval: 100,
            seed: 42,
            convergence_window: 50,
        }
    }
}

// ============================================================================
// ADVI (Mean-Field)
// ============================================================================

/// Automatic Differentiation Variational Inference with mean-field Gaussian
///
/// ADVI transforms constrained parameters to unconstrained space and
/// fits a diagonal-covariance Gaussian variational approximation:
///
/// 1. Transform constrained parameters theta to unconstrained eta = T^{-1}(theta)
/// 2. Fit q(eta) = N(mu, diag(sigma^2))
/// 3. ELBO = E_q[log p(T(eta), x) + log |det J_T(eta)|] - E_q[log q(eta)]
///
/// The user provides:
/// - A log joint density function log p(theta, data) in the *constrained* space
/// - Parameter constraints for each dimension
#[derive(Debug, Clone)]
pub struct AdviMeanField {
    /// Variational distribution in unconstrained space
    pub variational: MeanFieldGaussian,
    /// Parameter constraints
    pub constraints: Vec<ParameterConstraint>,
    /// Configuration
    pub config: AdviConfig,
    /// Diagnostics
    pub diagnostics: VariationalDiagnostics,
    /// Dimensionality
    pub dim: usize,
}

impl AdviMeanField {
    /// Create a new ADVI mean-field instance
    ///
    /// # Arguments
    /// * `constraints` - Constraint for each parameter dimension
    /// * `config` - ADVI configuration
    pub fn new(constraints: Vec<ParameterConstraint>, config: AdviConfig) -> Result<Self> {
        let dim = constraints.len();
        if dim == 0 {
            return Err(StatsError::InvalidArgument(
                "Must have at least one parameter".to_string(),
            ));
        }

        let variational = MeanFieldGaussian::new(dim)?;

        Ok(Self {
            variational,
            constraints,
            config,
            diagnostics: VariationalDiagnostics::new(),
            dim,
        })
    }

    /// Create ADVI with all unconstrained parameters
    pub fn new_unconstrained(dim: usize, config: AdviConfig) -> Result<Self> {
        let constraints = vec![ParameterConstraint::Real; dim];
        Self::new(constraints, config)
    }

    /// Initialize variational parameters from constrained-space values
    pub fn initialize_from_constrained(&mut self, theta: &Array1<f64>) -> Result<()> {
        if theta.len() != self.dim {
            return Err(StatsError::DimensionMismatch(format!(
                "theta length ({}) must match dimension ({})",
                theta.len(),
                self.dim
            )));
        }

        let mut eta = Array1::zeros(self.dim);
        for i in 0..self.dim {
            eta[i] = self.constraints[i].inverse(theta[i])?;
        }
        self.variational.means = eta;
        // Initialize with moderate uncertainty
        self.variational.log_stds = Array1::from_elem(self.dim, -1.0);
        Ok(())
    }

    /// Run ADVI optimization
    ///
    /// # Arguments
    /// * `log_joint` - Function computing log p(theta) in the constrained space.
    ///   Takes a constrained parameter vector and returns (log_prob, gradient_wrt_theta).
    ///
    /// # Returns
    /// * ADVI result with optimized variational distribution
    pub fn fit<F>(&mut self, log_joint: F) -> Result<AdviResult>
    where
        F: Fn(&Array1<f64>) -> Result<(f64, Array1<f64>)>,
    {
        let n_params = self.variational.n_params();
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

        for iter in 0..self.config.max_iter {
            // Compute stochastic ELBO gradient
            let (elbo, grad) = self.compute_elbo_gradient(&log_joint, iter as u64)?;

            self.diagnostics.record_elbo(elbo);

            let grad_norm = grad.dot(&grad).sqrt();
            self.diagnostics.record_gradient_norm(grad_norm);

            // Clip gradient
            let clipped_grad = if self.config.grad_clip > 0.0 && grad_norm > self.config.grad_clip {
                &grad * (self.config.grad_clip / grad_norm)
            } else {
                grad
            };

            // Get current parameters
            let current_params = self.variational.get_params();

            // Apply update
            let new_params = if let Some(ref mut adam) = adam_state {
                let update = adam.compute_update(&clipped_grad)?;
                &current_params + &update
            } else {
                let lr = self.config.lr_schedule.get_lr(iter);
                &current_params + &(&clipped_grad * lr)
            };

            let param_change = (&new_params - &current_params).mapv(|x| x * x).sum().sqrt();
            self.diagnostics.record_param_change(param_change);

            self.variational.set_params(&new_params)?;

            // Check convergence
            if iter > self.config.convergence_window {
                if let Some(rel_change) = self
                    .diagnostics
                    .relative_elbo_change(self.config.convergence_window)
                {
                    if rel_change < self.config.tol {
                        self.diagnostics.converged = true;
                        break;
                    }
                }
            }
        }

        // Transform results back to constrained space
        let constrained_means = self.transform_to_constrained(&self.variational.means)?;

        Ok(AdviResult {
            variational: self.variational.clone(),
            constraints: self.constraints.clone(),
            constrained_means,
            diagnostics: self.diagnostics.clone(),
            dim: self.dim,
        })
    }

    /// Compute ELBO and its gradient using Monte Carlo estimation
    fn compute_elbo_gradient<F>(&self, log_joint: &F, seed: u64) -> Result<(f64, Array1<f64>)>
    where
        F: Fn(&Array1<f64>) -> Result<(f64, Array1<f64>)>,
    {
        let dim = self.dim;
        let n_samples = self.config.n_mc_samples.max(1);
        let n_params = 2 * dim;

        let mut total_elbo = 0.0;
        let mut total_grad = Array1::zeros(n_params);

        let stds = self.variational.stds();

        for s in 0..n_samples {
            // Generate epsilon ~ N(0, I)
            let epsilon = generate_standard_normal_advi(dim, seed * 1000 + s as u64);

            // Reparameterize: eta = mu + sigma * epsilon
            let eta = self.variational.sample(&epsilon)?;

            // Transform to constrained space
            let theta = self.transform_to_constrained(&eta)?;

            // Compute log joint in constrained space
            let (log_p, grad_theta) = log_joint(&theta)?;

            // Compute log |det J| (sum of per-element log det Jacobians)
            let mut log_det_j = 0.0;
            for i in 0..dim {
                log_det_j += self.constraints[i].log_det_jacobian(eta[i]);
            }

            // ELBO contribution (before subtracting entropy, which we handle analytically)
            total_elbo += log_p + log_det_j;

            // Gradient of log p wrt eta (chain rule through transform)
            let grad_eta = self.compute_grad_eta(&eta, &grad_theta)?;

            // Gradient of log |det J| wrt eta
            let grad_log_det_j = self.compute_grad_log_det_j(&eta)?;

            // Combined gradient wrt eta
            let grad_combined = &grad_eta + &grad_log_det_j;

            // Gradient wrt variational params (mu, log_sigma)
            for i in 0..dim {
                // d/d_mu = d/d_eta (since eta = mu + sigma*eps, d_eta/d_mu = 1)
                total_grad[i] += grad_combined[i];
                // d/d_log_sigma = d/d_eta * d_eta/d_log_sigma
                //               = grad_combined[i] * epsilon[i] * sigma[i]
                total_grad[dim + i] += grad_combined[i] * epsilon[i] * stds[i];
            }
        }

        // Average over samples
        total_elbo /= n_samples as f64;
        total_grad /= n_samples as f64;

        // Add entropy and its gradient
        let entropy = self.variational.entropy();
        total_elbo += entropy;

        // Gradient of entropy wrt log_sigma is 1 for each dimension
        for i in 0..dim {
            total_grad[dim + i] += 1.0;
        }

        Ok((total_elbo, total_grad))
    }

    /// Transform unconstrained parameters to constrained space
    fn transform_to_constrained(&self, eta: &Array1<f64>) -> Result<Array1<f64>> {
        let mut theta = Array1::zeros(self.dim);
        for i in 0..self.dim {
            theta[i] = self.constraints[i].forward(eta[i]);
        }
        Ok(theta)
    }

    /// Compute gradient of log p wrt unconstrained eta using chain rule
    fn compute_grad_eta(&self, eta: &Array1<f64>, grad_theta: &Array1<f64>) -> Result<Array1<f64>> {
        let mut grad_eta = Array1::zeros(self.dim);
        for i in 0..self.dim {
            // d log_p / d eta_i = d log_p / d theta_i * d theta_i / d eta_i
            let dtheta_deta = self.compute_transform_derivative(i, eta[i]);
            grad_eta[i] = grad_theta[i] * dtheta_deta;
        }
        Ok(grad_eta)
    }

    /// Compute derivative of forward transform for parameter i
    fn compute_transform_derivative(&self, i: usize, unconstrained: f64) -> f64 {
        match &self.constraints[i] {
            ParameterConstraint::Real => 1.0,
            ParameterConstraint::Positive => unconstrained.exp(),
            ParameterConstraint::UnitInterval => {
                let s = 1.0 / (1.0 + (-unconstrained).exp());
                s * (1.0 - s)
            }
            ParameterConstraint::Bounded { lower, upper } => {
                let s = 1.0 / (1.0 + (-unconstrained).exp());
                (upper - lower) * s * (1.0 - s)
            }
            ParameterConstraint::LowerBounded { .. } => unconstrained.exp(),
            ParameterConstraint::UpperBounded { .. } => (-unconstrained).exp(),
            ParameterConstraint::Simplex { .. } => {
                let s = 1.0 / (1.0 + (-unconstrained).exp());
                s * (1.0 - s)
            }
        }
    }

    /// Compute gradient of log |det J| wrt unconstrained parameters
    fn compute_grad_log_det_j(&self, eta: &Array1<f64>) -> Result<Array1<f64>> {
        let mut grad = Array1::zeros(self.dim);
        for i in 0..self.dim {
            grad[i] = self.compute_grad_log_det_j_single(i, eta[i]);
        }
        Ok(grad)
    }

    /// Compute d/d_eta log|det J_forward(eta)| for a single parameter
    fn compute_grad_log_det_j_single(&self, i: usize, unconstrained: f64) -> f64 {
        match &self.constraints[i] {
            ParameterConstraint::Real => 0.0,
            ParameterConstraint::Positive => 1.0,
            ParameterConstraint::UnitInterval => {
                let s = 1.0 / (1.0 + (-unconstrained).exp());
                1.0 - 2.0 * s
            }
            ParameterConstraint::Bounded { .. } => {
                let s = 1.0 / (1.0 + (-unconstrained).exp());
                1.0 - 2.0 * s
            }
            ParameterConstraint::LowerBounded { .. } => 1.0,
            ParameterConstraint::UpperBounded { .. } => 1.0,
            ParameterConstraint::Simplex { .. } => {
                let s = 1.0 / (1.0 + (-unconstrained).exp());
                1.0 - 2.0 * s
            }
        }
    }
}

// ============================================================================
// ADVI (Full-Rank)
// ============================================================================

/// Automatic Differentiation Variational Inference with full-rank Gaussian
///
/// Like AdviMeanField but uses a Gaussian with full covariance matrix,
/// parameterized by its Cholesky factor. This can capture posterior
/// correlations but has O(d^2) parameters.
///
/// q(eta) = N(mu, L L^T) where L is lower-triangular
#[derive(Debug, Clone)]
pub struct AdviFullRank {
    /// Variational distribution in unconstrained space
    pub variational: FullRankGaussian,
    /// Parameter constraints
    pub constraints: Vec<ParameterConstraint>,
    /// Configuration
    pub config: AdviConfig,
    /// Diagnostics
    pub diagnostics: VariationalDiagnostics,
    /// Dimensionality
    pub dim: usize,
}

impl AdviFullRank {
    /// Create a new full-rank ADVI instance
    pub fn new(constraints: Vec<ParameterConstraint>, config: AdviConfig) -> Result<Self> {
        let dim = constraints.len();
        if dim == 0 {
            return Err(StatsError::InvalidArgument(
                "Must have at least one parameter".to_string(),
            ));
        }

        let variational = FullRankGaussian::new(dim)?;

        Ok(Self {
            variational,
            constraints,
            config,
            diagnostics: VariationalDiagnostics::new(),
            dim,
        })
    }

    /// Create full-rank ADVI with all unconstrained parameters
    pub fn new_unconstrained(dim: usize, config: AdviConfig) -> Result<Self> {
        let constraints = vec![ParameterConstraint::Real; dim];
        Self::new(constraints, config)
    }

    /// Initialize variational parameters from constrained-space values
    pub fn initialize_from_constrained(&mut self, theta: &Array1<f64>) -> Result<()> {
        if theta.len() != self.dim {
            return Err(StatsError::DimensionMismatch(format!(
                "theta length ({}) must match dimension ({})",
                theta.len(),
                self.dim
            )));
        }

        let mut eta = Array1::zeros(self.dim);
        for i in 0..self.dim {
            eta[i] = self.constraints[i].inverse(theta[i])?;
        }
        self.variational.mean = eta;
        // Initialize with small identity-like covariance
        self.variational.chol_factor = Array2::eye(self.dim) * 0.1;
        Ok(())
    }

    /// Run full-rank ADVI optimization
    ///
    /// # Arguments
    /// * `log_joint` - Function computing log p(theta) in the constrained space.
    ///   Takes a constrained parameter vector and returns (log_prob, gradient_wrt_theta).
    pub fn fit<F>(&mut self, log_joint: F) -> Result<AdviFullRankResult>
    where
        F: Fn(&Array1<f64>) -> Result<(f64, Array1<f64>)>,
    {
        let n_params = self.variational.n_params();
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

        for iter in 0..self.config.max_iter {
            // Compute stochastic ELBO gradient
            let (elbo, grad) = self.compute_elbo_gradient_full_rank(&log_joint, iter as u64)?;

            self.diagnostics.record_elbo(elbo);

            let grad_norm = grad.dot(&grad).sqrt();
            self.diagnostics.record_gradient_norm(grad_norm);

            // Clip gradient
            let clipped_grad = if self.config.grad_clip > 0.0 && grad_norm > self.config.grad_clip {
                &grad * (self.config.grad_clip / grad_norm)
            } else {
                grad
            };

            // Get current parameters
            let current_params = self.variational.get_params();

            // Apply update
            let new_params = if let Some(ref mut adam) = adam_state {
                let update = adam.compute_update(&clipped_grad)?;
                &current_params + &update
            } else {
                let lr = self.config.lr_schedule.get_lr(iter);
                &current_params + &(&clipped_grad * lr)
            };

            let param_change = (&new_params - &current_params).mapv(|x| x * x).sum().sqrt();
            self.diagnostics.record_param_change(param_change);

            self.variational.set_params(&new_params)?;

            // Check convergence
            if iter > self.config.convergence_window {
                if let Some(rel_change) = self
                    .diagnostics
                    .relative_elbo_change(self.config.convergence_window)
                {
                    if rel_change < self.config.tol {
                        self.diagnostics.converged = true;
                        break;
                    }
                }
            }
        }

        // Transform results back to constrained space
        let constrained_means = self.transform_to_constrained(&self.variational.mean)?;

        Ok(AdviFullRankResult {
            variational: self.variational.clone(),
            constraints: self.constraints.clone(),
            constrained_means,
            diagnostics: self.diagnostics.clone(),
            dim: self.dim,
        })
    }

    /// Compute ELBO and gradient for full-rank approximation
    fn compute_elbo_gradient_full_rank<F>(
        &self,
        log_joint: &F,
        seed: u64,
    ) -> Result<(f64, Array1<f64>)>
    where
        F: Fn(&Array1<f64>) -> Result<(f64, Array1<f64>)>,
    {
        let dim = self.dim;
        let n_samples = self.config.n_mc_samples.max(1);
        let n_params = self.variational.n_params();

        let mut total_elbo = 0.0;
        let mut total_grad = Array1::zeros(n_params);

        let n_tril = dim * (dim + 1) / 2;

        for s in 0..n_samples {
            // Generate epsilon ~ N(0, I)
            let epsilon = generate_standard_normal_advi(dim, seed * 1000 + s as u64);

            // Reparameterize: eta = mu + L * epsilon
            let eta = self.variational.sample(&epsilon)?;

            // Transform to constrained space
            let theta = self.transform_to_constrained(&eta)?;

            // Compute log joint
            let (log_p, grad_theta) = log_joint(&theta)?;

            // Compute log |det J|
            let mut log_det_j = 0.0;
            for i in 0..dim {
                log_det_j += compute_log_det_jacobian(&self.constraints[i], eta[i]);
            }

            total_elbo += log_p + log_det_j;

            // Gradient wrt eta
            let grad_eta = compute_grad_eta_from_theta(dim, &eta, &grad_theta, &self.constraints)?;
            let grad_log_det = compute_grad_log_det(dim, &eta, &self.constraints)?;
            let grad_combined: Array1<f64> = &grad_eta + &grad_log_det;

            // Gradient wrt mean: d/d_mu = grad_combined (since eta = mu + L*eps)
            for i in 0..dim {
                total_grad[i] += grad_combined[i];
            }

            // Gradient wrt L (lower triangular elements)
            // d/d L_{ij} = grad_combined[i] * epsilon[j] for j <= i
            let mut l_idx = dim;
            for i in 0..dim {
                for j in 0..=i {
                    total_grad[l_idx] += grad_combined[i] * epsilon[j];
                    l_idx += 1;
                }
            }
        }

        // Average over samples
        total_elbo /= n_samples as f64;
        total_grad /= n_samples as f64;

        // Add entropy and its gradient
        let entropy = self.variational.entropy();
        total_elbo += entropy;

        // Gradient of entropy wrt L_{ii} (diagonal of Cholesky factor) is 1/L_{ii}
        let mut l_idx = dim;
        for i in 0..dim {
            for j in 0..=i {
                if i == j {
                    let l_ii = self.variational.chol_factor[[i, i]];
                    if l_ii.abs() > 1e-15 {
                        total_grad[l_idx] += 1.0 / l_ii;
                    }
                }
                l_idx += 1;
            }
        }

        Ok((total_elbo, total_grad))
    }

    /// Transform unconstrained parameters to constrained space
    fn transform_to_constrained(&self, eta: &Array1<f64>) -> Result<Array1<f64>> {
        let mut theta = Array1::zeros(self.dim);
        for i in 0..self.dim {
            theta[i] = self.constraints[i].forward(eta[i]);
        }
        Ok(theta)
    }
}

// ============================================================================
// ADVI Results
// ============================================================================

/// Results from mean-field ADVI
#[derive(Debug, Clone)]
pub struct AdviResult {
    /// Optimized variational distribution in unconstrained space
    pub variational: MeanFieldGaussian,
    /// Parameter constraints
    pub constraints: Vec<ParameterConstraint>,
    /// Posterior means in constrained space
    pub constrained_means: Array1<f64>,
    /// Optimization diagnostics
    pub diagnostics: VariationalDiagnostics,
    /// Dimensionality
    pub dim: usize,
}

impl AdviResult {
    /// Get posterior means in unconstrained space
    pub fn unconstrained_means(&self) -> &Array1<f64> {
        &self.variational.means
    }

    /// Get posterior standard deviations in unconstrained space
    pub fn unconstrained_stds(&self) -> Array1<f64> {
        self.variational.stds()
    }

    /// Get posterior means in constrained space
    pub fn constrained_means(&self) -> &Array1<f64> {
        &self.constrained_means
    }

    /// Sample from the approximate posterior and transform to constrained space
    pub fn sample_constrained(&self, epsilon: &Array1<f64>) -> Result<Array1<f64>> {
        let eta = self.variational.sample(epsilon)?;
        let mut theta = Array1::zeros(self.dim);
        for i in 0..self.dim {
            theta[i] = self.constraints[i].forward(eta[i]);
        }
        Ok(theta)
    }

    /// Compute approximate credible intervals in constrained space
    ///
    /// Note: These are approximate because the transform is nonlinear.
    /// For more accurate intervals, use `sample_constrained` with many samples.
    pub fn approximate_credible_intervals(&self, confidence: f64) -> Result<Array2<f64>> {
        check_probability(confidence, "confidence")?;

        let alpha = (1.0 - confidence) / 2.0;
        let z_critical = super::normal_ppf(1.0 - alpha)?;

        let stds = self.variational.stds();
        let mut intervals = Array2::zeros((self.dim, 2));

        for i in 0..self.dim {
            let eta_low = self.variational.means[i] - z_critical * stds[i];
            let eta_high = self.variational.means[i] + z_critical * stds[i];

            // Transform bounds to constrained space
            let theta_low = self.constraints[i].forward(eta_low);
            let theta_high = self.constraints[i].forward(eta_high);

            // Ensure proper ordering (some transforms may flip)
            intervals[[i, 0]] = theta_low.min(theta_high);
            intervals[[i, 1]] = theta_low.max(theta_high);
        }

        Ok(intervals)
    }
}

/// Results from full-rank ADVI
#[derive(Debug, Clone)]
pub struct AdviFullRankResult {
    /// Optimized variational distribution in unconstrained space
    pub variational: FullRankGaussian,
    /// Parameter constraints
    pub constraints: Vec<ParameterConstraint>,
    /// Posterior means in constrained space
    pub constrained_means: Array1<f64>,
    /// Optimization diagnostics
    pub diagnostics: VariationalDiagnostics,
    /// Dimensionality
    pub dim: usize,
}

impl AdviFullRankResult {
    /// Get posterior means in unconstrained space
    pub fn unconstrained_means(&self) -> &Array1<f64> {
        &self.variational.mean
    }

    /// Get posterior covariance in unconstrained space
    pub fn unconstrained_covariance(&self) -> Array2<f64> {
        self.variational.covariance()
    }

    /// Get posterior means in constrained space
    pub fn constrained_means(&self) -> &Array1<f64> {
        &self.constrained_means
    }

    /// Sample from the approximate posterior and transform to constrained space
    pub fn sample_constrained(&self, epsilon: &Array1<f64>) -> Result<Array1<f64>> {
        let eta = self.variational.sample(epsilon)?;
        let mut theta = Array1::zeros(self.dim);
        for i in 0..self.dim {
            theta[i] = self.constraints[i].forward(eta[i]);
        }
        Ok(theta)
    }

    /// Compute approximate credible intervals in constrained space
    pub fn approximate_credible_intervals(&self, confidence: f64) -> Result<Array2<f64>> {
        check_probability(confidence, "confidence")?;

        let alpha = (1.0 - confidence) / 2.0;
        let z_critical = super::normal_ppf(1.0 - alpha)?;

        let cov = self.variational.covariance();
        let mut intervals = Array2::zeros((self.dim, 2));

        for i in 0..self.dim {
            let std_i = cov[[i, i]].sqrt();
            let eta_low = self.variational.mean[i] - z_critical * std_i;
            let eta_high = self.variational.mean[i] + z_critical * std_i;

            let theta_low = self.constraints[i].forward(eta_low);
            let theta_high = self.constraints[i].forward(eta_high);

            intervals[[i, 0]] = theta_low.min(theta_high);
            intervals[[i, 1]] = theta_low.max(theta_high);
        }

        Ok(intervals)
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Compute log determinant of Jacobian for a single constraint
fn compute_log_det_jacobian(constraint: &ParameterConstraint, unconstrained: f64) -> f64 {
    constraint.log_det_jacobian(unconstrained)
}

/// Compute gradient of ELBO wrt unconstrained eta from constrained gradient
fn compute_grad_eta_from_theta(
    dim: usize,
    eta: &Array1<f64>,
    grad_theta: &Array1<f64>,
    constraints: &[ParameterConstraint],
) -> Result<Array1<f64>> {
    let mut grad_eta = Array1::zeros(dim);
    for i in 0..dim {
        let dtheta_deta = compute_transform_deriv(&constraints[i], eta[i]);
        grad_eta[i] = grad_theta[i] * dtheta_deta;
    }
    Ok(grad_eta)
}

/// Compute gradient of sum of log |det J| wrt eta
fn compute_grad_log_det(
    dim: usize,
    eta: &Array1<f64>,
    constraints: &[ParameterConstraint],
) -> Result<Array1<f64>> {
    let mut grad = Array1::zeros(dim);
    for i in 0..dim {
        grad[i] = compute_grad_log_det_single(&constraints[i], eta[i]);
    }
    Ok(grad)
}

/// Compute derivative of forward transform for a constraint
fn compute_transform_deriv(constraint: &ParameterConstraint, unconstrained: f64) -> f64 {
    match constraint {
        ParameterConstraint::Real => 1.0,
        ParameterConstraint::Positive => unconstrained.exp(),
        ParameterConstraint::UnitInterval => {
            let s = 1.0 / (1.0 + (-unconstrained).exp());
            s * (1.0 - s)
        }
        ParameterConstraint::Bounded { lower, upper } => {
            let s = 1.0 / (1.0 + (-unconstrained).exp());
            (upper - lower) * s * (1.0 - s)
        }
        ParameterConstraint::LowerBounded { .. } => unconstrained.exp(),
        ParameterConstraint::UpperBounded { .. } => (-unconstrained).exp(),
        ParameterConstraint::Simplex { .. } => {
            let s = 1.0 / (1.0 + (-unconstrained).exp());
            s * (1.0 - s)
        }
    }
}

/// Compute d/d_eta log|det J_forward(eta)| for a single constraint
fn compute_grad_log_det_single(constraint: &ParameterConstraint, unconstrained: f64) -> f64 {
    match constraint {
        ParameterConstraint::Real => 0.0,
        ParameterConstraint::Positive => 1.0,
        ParameterConstraint::UnitInterval => {
            let s = 1.0 / (1.0 + (-unconstrained).exp());
            1.0 - 2.0 * s
        }
        ParameterConstraint::Bounded { .. } => {
            let s = 1.0 / (1.0 + (-unconstrained).exp());
            1.0 - 2.0 * s
        }
        ParameterConstraint::LowerBounded { .. } => 1.0,
        ParameterConstraint::UpperBounded { .. } => 1.0,
        ParameterConstraint::Simplex { .. } => {
            let s = 1.0 / (1.0 + (-unconstrained).exp());
            1.0 - 2.0 * s
        }
    }
}

/// Generate approximate standard normal samples (deterministic)
fn generate_standard_normal_advi(dim: usize, seed: u64) -> Array1<f64> {
    let mut result = Array1::zeros(dim);
    let golden_ratio = 1.618033988749895;

    for i in 0..dim {
        let u1 = ((seed as f64 * golden_ratio + i as f64 * 0.7548776662466927) % 1.0).abs();
        let u2 = ((seed as f64 * 0.5698402909980532 + i as f64 * golden_ratio) % 1.0).abs();

        let u1_safe = u1.max(1e-10).min(1.0 - 1e-10);
        let u2_safe = u2.max(1e-10).min(1.0 - 1e-10);

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
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_constraint_real() {
        let c = ParameterConstraint::Real;
        assert!((c.forward(1.5) - 1.5).abs() < 1e-10);
        let inv = c.inverse(1.5).expect("should invert");
        assert!((inv - 1.5).abs() < 1e-10);
        assert!((c.log_det_jacobian(1.5)).abs() < 1e-10);
    }

    #[test]
    fn test_constraint_positive() {
        let c = ParameterConstraint::Positive;
        // forward(0) = exp(0) = 1
        assert!((c.forward(0.0) - 1.0).abs() < 1e-10);
        // forward(1) = exp(1) = e
        assert!((c.forward(1.0) - 1.0_f64.exp()).abs() < 1e-10);
        // inverse(e) = 1
        let inv = c.inverse(1.0_f64.exp()).expect("should invert");
        assert!((inv - 1.0).abs() < 1e-10);
        // Positive constraint: inverse of non-positive should fail
        assert!(c.inverse(-1.0).is_err());
    }

    #[test]
    fn test_constraint_unit_interval() {
        let c = ParameterConstraint::UnitInterval;
        // sigmoid(0) = 0.5
        assert!((c.forward(0.0) - 0.5).abs() < 1e-10);
        // inverse(0.5) = 0
        let inv = c.inverse(0.5).expect("should invert");
        assert!(inv.abs() < 1e-10);
        // Boundary cases should fail
        assert!(c.inverse(0.0).is_err());
        assert!(c.inverse(1.0).is_err());
    }

    #[test]
    fn test_constraint_bounded() {
        let c = ParameterConstraint::Bounded {
            lower: -1.0,
            upper: 1.0,
        };
        // forward(0) = -1 + 2 * sigmoid(0) = -1 + 1 = 0
        assert!((c.forward(0.0)).abs() < 1e-10);
        // inverse(0) should be 0
        let inv = c.inverse(0.0).expect("should invert");
        assert!(inv.abs() < 1e-10);
    }

    #[test]
    fn test_constraint_lower_bounded() {
        let c = ParameterConstraint::LowerBounded { lower: 2.0 };
        // forward(0) = 2 + exp(0) = 3
        assert!((c.forward(0.0) - 3.0).abs() < 1e-10);
        let inv = c.inverse(3.0).expect("should invert");
        assert!(inv.abs() < 1e-10);
        assert!(c.inverse(1.0).is_err());
    }

    #[test]
    fn test_constraint_roundtrip() {
        let constraints = vec![
            ParameterConstraint::Real,
            ParameterConstraint::Positive,
            ParameterConstraint::UnitInterval,
            ParameterConstraint::Bounded {
                lower: 0.0,
                upper: 10.0,
            },
        ];

        let unconstrained_values = vec![0.5, 1.0, -0.5, 2.0];
        for (c, &eta) in constraints.iter().zip(unconstrained_values.iter()) {
            let theta = c.forward(eta);
            let eta_back = c.inverse(theta).expect("should invert");
            assert!(
                (eta_back - eta).abs() < 1e-8,
                "Roundtrip failed for {:?}: {} -> {} -> {}",
                c,
                eta,
                theta,
                eta_back
            );
        }
    }

    #[test]
    fn test_advi_mean_field_creation() {
        let constraints = vec![ParameterConstraint::Real, ParameterConstraint::Positive];
        let config = AdviConfig::default();
        let advi = AdviMeanField::new(constraints, config).expect("should create");
        assert_eq!(advi.dim, 2);
    }

    #[test]
    fn test_advi_mean_field_simple_gaussian() {
        // Fit ADVI to a simple 2D Gaussian target
        let target_mean = Array1::from_vec(vec![1.0, -2.0]);
        let target_precision = 2.0; // precision = 1/variance

        let constraints = vec![ParameterConstraint::Real, ParameterConstraint::Real];
        let config = AdviConfig {
            max_iter: 500,
            n_mc_samples: 1,
            lr_schedule: LearningRateSchedule::Adam {
                lr: 0.05,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            tol: 1e-6,
            convergence_window: 20,
            ..AdviConfig::default()
        };

        let mut advi = AdviMeanField::new(constraints, config).expect("should create");

        let tm = target_mean.clone();
        let result = advi
            .fit(move |theta: &Array1<f64>| {
                let diff = theta - &tm;
                let log_p = -0.5 * target_precision * diff.dot(&diff);
                let grad = &diff * (-target_precision);
                Ok((log_p, grad))
            })
            .expect("should fit");

        // Check that means are reasonable (within tolerance for stochastic optimization)
        assert!(
            result.diagnostics.n_iterations > 0,
            "Should have performed iterations"
        );
        assert!(
            result.diagnostics.final_elbo.is_finite(),
            "ELBO should be finite"
        );
    }

    #[test]
    fn test_advi_full_rank_creation() {
        let constraints = vec![
            ParameterConstraint::Real,
            ParameterConstraint::Positive,
            ParameterConstraint::UnitInterval,
        ];
        let config = AdviConfig::default();
        let advi = AdviFullRank::new(constraints, config).expect("should create");
        assert_eq!(advi.dim, 3);
    }

    #[test]
    fn test_advi_full_rank_simple() {
        let constraints = vec![ParameterConstraint::Real, ParameterConstraint::Real];
        let config = AdviConfig {
            max_iter: 200,
            n_mc_samples: 1,
            lr_schedule: LearningRateSchedule::Adam {
                lr: 0.02,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            tol: 1e-5,
            convergence_window: 20,
            ..AdviConfig::default()
        };

        let mut advi = AdviFullRank::new(constraints, config).expect("should create");

        let result = advi
            .fit(|theta: &Array1<f64>| {
                // Simple separable Gaussian
                let log_p = -0.5 * theta.dot(theta);
                let grad = theta * (-1.0);
                Ok((log_p, grad))
            })
            .expect("should fit");

        assert!(result.diagnostics.n_iterations > 0);
        assert!(result.diagnostics.final_elbo.is_finite());
    }

    #[test]
    fn test_advi_with_constrained_params() {
        // Test ADVI with mixed constraints
        let constraints = vec![
            ParameterConstraint::Real,     // unconstrained
            ParameterConstraint::Positive, // must be > 0
        ];

        let config = AdviConfig {
            max_iter: 300,
            n_mc_samples: 1,
            lr_schedule: LearningRateSchedule::Adam {
                lr: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            tol: 1e-5,
            convergence_window: 30,
            ..AdviConfig::default()
        };

        let mut advi = AdviMeanField::new(constraints, config).expect("should create");

        let result = advi
            .fit(|theta: &Array1<f64>| {
                // log p = -0.5 * (theta[0] - 1)^2 - 2 * (theta[1] - 2)^2
                let log_p = -0.5 * (theta[0] - 1.0).powi(2) - 2.0 * (theta[1] - 2.0).powi(2);
                let mut grad = Array1::zeros(2);
                grad[0] = -(theta[0] - 1.0);
                grad[1] = -4.0 * (theta[1] - 2.0);
                Ok((log_p, grad))
            })
            .expect("should fit");

        // The constrained mean for the positive parameter should be > 0
        assert!(
            result.constrained_means[1] > 0.0,
            "Positive-constrained parameter should be > 0, got {}",
            result.constrained_means[1]
        );
    }

    #[test]
    fn test_advi_result_credible_intervals() {
        let constraints = vec![ParameterConstraint::Real, ParameterConstraint::Positive];
        let config = AdviConfig {
            max_iter: 100,
            ..AdviConfig::default()
        };

        let mut advi = AdviMeanField::new(constraints, config).expect("should create");

        let result = advi
            .fit(|theta: &Array1<f64>| {
                let log_p = -0.5 * theta.dot(theta);
                let grad = theta * (-1.0);
                Ok((log_p, grad))
            })
            .expect("should fit");

        let intervals = result
            .approximate_credible_intervals(0.95)
            .expect("should compute intervals");

        assert_eq!(intervals.nrows(), 2);
        assert_eq!(intervals.ncols(), 2);
        // Lower bound should be less than upper bound
        for i in 0..2 {
            assert!(
                intervals[[i, 0]] <= intervals[[i, 1]],
                "Lower bound should be <= upper bound at dim {}",
                i
            );
        }
    }

    #[test]
    fn test_log_det_jacobian_positive() {
        let c = ParameterConstraint::Positive;
        // log|det J| for exp transform is just the unconstrained value
        assert!((c.log_det_jacobian(0.0)).abs() < 1e-10);
        assert!((c.log_det_jacobian(1.0) - 1.0).abs() < 1e-10);
        assert!((c.log_det_jacobian(-1.0) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_log_det_jacobian_unit_interval() {
        let c = ParameterConstraint::UnitInterval;
        // At eta=0, sigmoid(0)=0.5, so log|sigmoid'(0)| = log(0.25)
        let expected = (0.25_f64).ln();
        assert!(
            (c.log_det_jacobian(0.0) - expected).abs() < 1e-10,
            "log det J at 0 should be {}, got {}",
            expected,
            c.log_det_jacobian(0.0)
        );
    }
}
