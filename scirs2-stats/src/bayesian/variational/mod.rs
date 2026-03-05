//! Variational inference for Bayesian models
//!
//! This module implements variational inference methods as alternatives to MCMC
//! for approximate Bayesian inference. It provides:
//!
//! - **Coordinate Ascent VI (CAVI)**: Classical variational inference for Bayesian
//!   linear regression and Automatic Relevance Determination (ARD)
//! - **Stochastic Variational Inference (SVI)**: Scalable VI with mini-batch
//!   ELBO estimation, natural gradient updates, and learning rate scheduling
//! - **ADVI**: Automatic Differentiation Variational Inference with mean-field
//!   and full-rank Gaussian approximations
//! - **Variational Families**: Mean-field Gaussian, full-rank Gaussian, and
//!   normalizing flow placeholders
//! - **Diagnostics**: ELBO trace, gradient norm monitoring, convergence checks

mod advi;
mod families;
mod svi;

pub use advi::*;
pub use families::*;
pub use svi::*;

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::validation::*;
use statrs::statistics::Statistics;
use std::f64::consts::PI;

// ============================================================================
// Variational Diagnostics
// ============================================================================

/// Diagnostics for monitoring variational inference convergence
#[derive(Debug, Clone)]
pub struct VariationalDiagnostics {
    /// ELBO values at each iteration
    pub elbo_trace: Vec<f64>,
    /// Gradient norms at each iteration (if tracked)
    pub gradient_norms: Vec<f64>,
    /// Parameter change norms at each iteration (if tracked)
    pub param_change_norms: Vec<f64>,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Number of iterations performed
    pub n_iterations: usize,
    /// Final ELBO value
    pub final_elbo: f64,
}

impl VariationalDiagnostics {
    /// Create new diagnostics tracker
    pub fn new() -> Self {
        Self {
            elbo_trace: Vec::new(),
            gradient_norms: Vec::new(),
            param_change_norms: Vec::new(),
            converged: false,
            n_iterations: 0,
            final_elbo: f64::NEG_INFINITY,
        }
    }

    /// Record an ELBO value
    pub fn record_elbo(&mut self, elbo: f64) {
        self.elbo_trace.push(elbo);
        self.final_elbo = elbo;
        self.n_iterations = self.elbo_trace.len();
    }

    /// Record a gradient norm
    pub fn record_gradient_norm(&mut self, norm: f64) {
        self.gradient_norms.push(norm);
    }

    /// Record a parameter change norm
    pub fn record_param_change(&mut self, norm: f64) {
        self.param_change_norms.push(norm);
    }

    /// Check whether parameters have converged based on ELBO change
    pub fn check_elbo_convergence(&self, tol: f64) -> bool {
        if self.elbo_trace.len() < 2 {
            return false;
        }
        let n = self.elbo_trace.len();
        (self.elbo_trace[n - 1] - self.elbo_trace[n - 2]).abs() < tol
    }

    /// Check whether gradient norms indicate convergence
    pub fn check_gradient_convergence(&self, tol: f64) -> bool {
        if let Some(&last_norm) = self.gradient_norms.last() {
            last_norm < tol
        } else {
            false
        }
    }

    /// Check whether parameter changes indicate convergence
    pub fn check_param_convergence(&self, tol: f64) -> bool {
        if let Some(&last_change) = self.param_change_norms.last() {
            last_change < tol
        } else {
            false
        }
    }

    /// Compute the relative ELBO change over recent iterations
    pub fn relative_elbo_change(&self, window: usize) -> Option<f64> {
        let n = self.elbo_trace.len();
        if n < window + 1 {
            return None;
        }
        let recent = self.elbo_trace[n - 1];
        let earlier = self.elbo_trace[n - 1 - window];
        if earlier.abs() < 1e-15 {
            return Some(f64::INFINITY);
        }
        Some((recent - earlier).abs() / earlier.abs())
    }

    /// Get summary statistics for the ELBO trace
    pub fn elbo_summary(&self) -> ElboSummary {
        let n = self.elbo_trace.len();
        if n == 0 {
            return ElboSummary {
                min: f64::NAN,
                max: f64::NAN,
                final_value: f64::NAN,
                mean_change: f64::NAN,
                monotonic: true,
            };
        }

        let min = self
            .elbo_trace
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .elbo_trace
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut monotonic = true;
        let mut total_change = 0.0;
        for i in 1..n {
            let change = self.elbo_trace[i] - self.elbo_trace[i - 1];
            total_change += change.abs();
            if change < -1e-10 {
                monotonic = false;
            }
        }

        let mean_change = if n > 1 {
            total_change / (n - 1) as f64
        } else {
            0.0
        };

        ElboSummary {
            min,
            max,
            final_value: self.elbo_trace[n - 1],
            mean_change,
            monotonic,
        }
    }
}

impl Default for VariationalDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for the ELBO trace
#[derive(Debug, Clone)]
pub struct ElboSummary {
    /// Minimum ELBO observed
    pub min: f64,
    /// Maximum ELBO observed
    pub max: f64,
    /// Final ELBO value
    pub final_value: f64,
    /// Mean absolute change between consecutive iterations
    pub mean_change: f64,
    /// Whether the ELBO was monotonically increasing
    pub monotonic: bool,
}

// ============================================================================
// Variational Families
// ============================================================================

/// Mean-field Gaussian variational family
///
/// Approximates the posterior with a factorized Gaussian:
/// q(z) = prod_i N(z_i; mu_i, sigma_i^2)
///
/// This is the simplest variational family but cannot capture correlations.
#[derive(Debug, Clone)]
pub struct MeanFieldGaussian {
    /// Variational means
    pub means: Array1<f64>,
    /// Variational log standard deviations (unconstrained parameterization)
    pub log_stds: Array1<f64>,
    /// Dimensionality
    pub dim: usize,
}

impl MeanFieldGaussian {
    /// Create a new mean-field Gaussian with given dimension
    pub fn new(dim: usize) -> Result<Self> {
        check_positive(dim, "dim")?;
        Ok(Self {
            means: Array1::zeros(dim),
            log_stds: Array1::zeros(dim), // std = 1.0 initially
            dim,
        })
    }

    /// Create from specific parameters
    pub fn from_params(means: Array1<f64>, log_stds: Array1<f64>) -> Result<Self> {
        if means.len() != log_stds.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "means length ({}) must match log_stds length ({})",
                means.len(),
                log_stds.len()
            )));
        }
        checkarray_finite(&means, "means")?;
        checkarray_finite(&log_stds, "log_stds")?;
        let dim = means.len();
        Ok(Self {
            means,
            log_stds,
            dim,
        })
    }

    /// Get the standard deviations
    pub fn stds(&self) -> Array1<f64> {
        self.log_stds.mapv(f64::exp)
    }

    /// Get the variances
    pub fn variances(&self) -> Array1<f64> {
        self.log_stds.mapv(|ls| (2.0 * ls).exp())
    }

    /// Sample from the variational distribution using reparameterization trick
    ///
    /// z = mu + sigma * epsilon, where epsilon ~ N(0, I)
    pub fn sample(&self, epsilon: &Array1<f64>) -> Result<Array1<f64>> {
        if epsilon.len() != self.dim {
            return Err(StatsError::DimensionMismatch(format!(
                "epsilon length ({}) must match dimension ({})",
                epsilon.len(),
                self.dim
            )));
        }
        let stds = self.stds();
        Ok(&self.means + &(&stds * epsilon))
    }

    /// Compute the entropy of the mean-field Gaussian
    /// H\[q\] = sum_i 0.5 * (1 + log(2*pi) + 2*log_std_i)
    pub fn entropy(&self) -> f64 {
        let base = 0.5 * (1.0 + (2.0 * PI).ln());
        self.log_stds.iter().map(|&ls| base + ls).sum::<f64>()
    }

    /// Compute log q(z) for a given z
    pub fn log_prob(&self, z: &Array1<f64>) -> Result<f64> {
        if z.len() != self.dim {
            return Err(StatsError::DimensionMismatch(format!(
                "z length ({}) must match dimension ({})",
                z.len(),
                self.dim
            )));
        }
        let stds = self.stds();
        let mut log_prob = 0.0;
        for i in 0..self.dim {
            let diff = z[i] - self.means[i];
            log_prob += -0.5 * (2.0 * PI).ln() - self.log_stds[i] - 0.5 * (diff / stds[i]).powi(2);
        }
        Ok(log_prob)
    }

    /// Get total number of variational parameters
    pub fn n_params(&self) -> usize {
        2 * self.dim
    }

    /// Get all variational parameters as a flat vector
    pub fn get_params(&self) -> Array1<f64> {
        let mut params = Array1::zeros(2 * self.dim);
        for i in 0..self.dim {
            params[i] = self.means[i];
            params[self.dim + i] = self.log_stds[i];
        }
        params
    }

    /// Set variational parameters from a flat vector
    pub fn set_params(&mut self, params: &Array1<f64>) -> Result<()> {
        if params.len() != 2 * self.dim {
            return Err(StatsError::DimensionMismatch(format!(
                "params length ({}) must be 2 * dim ({})",
                params.len(),
                2 * self.dim
            )));
        }
        for i in 0..self.dim {
            self.means[i] = params[i];
            self.log_stds[i] = params[self.dim + i];
        }
        Ok(())
    }
}

/// Full-rank Gaussian variational family
///
/// Approximates the posterior with a Gaussian with full covariance:
/// q(z) = N(z; mu, Sigma) where Sigma = L L^T (Cholesky parameterization)
///
/// This can capture correlations but has O(d^2) parameters.
#[derive(Debug, Clone)]
pub struct FullRankGaussian {
    /// Variational mean
    pub mean: Array1<f64>,
    /// Lower triangular Cholesky factor of the covariance
    /// Stored as a flattened lower-triangular matrix
    pub chol_factor: Array2<f64>,
    /// Dimensionality
    pub dim: usize,
}

impl FullRankGaussian {
    /// Create a new full-rank Gaussian with given dimension
    pub fn new(dim: usize) -> Result<Self> {
        check_positive(dim, "dim")?;
        Ok(Self {
            mean: Array1::zeros(dim),
            chol_factor: Array2::eye(dim), // Identity = unit covariance
            dim,
        })
    }

    /// Create from specific parameters
    pub fn from_params(mean: Array1<f64>, chol_factor: Array2<f64>) -> Result<Self> {
        let dim = mean.len();
        if chol_factor.nrows() != dim || chol_factor.ncols() != dim {
            return Err(StatsError::DimensionMismatch(format!(
                "chol_factor shape ({},{}) must be ({},{})",
                chol_factor.nrows(),
                chol_factor.ncols(),
                dim,
                dim
            )));
        }
        checkarray_finite(&mean, "mean")?;
        checkarray_finite(&chol_factor, "chol_factor")?;
        Ok(Self {
            mean,
            chol_factor,
            dim,
        })
    }

    /// Get the covariance matrix: Sigma = L * L^T
    pub fn covariance(&self) -> Array2<f64> {
        self.chol_factor.dot(&self.chol_factor.t())
    }

    /// Get the precision matrix (inverse covariance)
    pub fn precision(&self) -> Result<Array2<f64>> {
        let cov = self.covariance();
        scirs2_linalg::inv(&cov.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to invert covariance: {}", e))
        })
    }

    /// Sample from the variational distribution using reparameterization trick
    ///
    /// z = mu + L * epsilon, where epsilon ~ N(0, I)
    pub fn sample(&self, epsilon: &Array1<f64>) -> Result<Array1<f64>> {
        if epsilon.len() != self.dim {
            return Err(StatsError::DimensionMismatch(format!(
                "epsilon length ({}) must match dimension ({})",
                epsilon.len(),
                self.dim
            )));
        }
        Ok(&self.mean + &self.chol_factor.dot(epsilon))
    }

    /// Compute the entropy of the full-rank Gaussian
    /// H\[q\] = 0.5 * d * (1 + log(2*pi)) + sum_i log(L_ii)
    pub fn entropy(&self) -> f64 {
        let base = 0.5 * self.dim as f64 * (1.0 + (2.0 * PI).ln());
        let log_det: f64 = (0..self.dim)
            .map(|i| self.chol_factor[[i, i]].abs().ln())
            .sum();
        base + log_det
    }

    /// Compute log q(z) for a given z
    pub fn log_prob(&self, z: &Array1<f64>) -> Result<f64> {
        if z.len() != self.dim {
            return Err(StatsError::DimensionMismatch(format!(
                "z length ({}) must match dimension ({})",
                z.len(),
                self.dim
            )));
        }
        let precision = self.precision()?;
        let diff = z - &self.mean;
        let mahal = diff.dot(&precision.dot(&diff));
        let log_det: f64 = (0..self.dim)
            .map(|i| self.chol_factor[[i, i]].abs().ln())
            .sum();
        Ok(-0.5 * self.dim as f64 * (2.0 * PI).ln() - log_det - 0.5 * mahal)
    }

    /// Get total number of variational parameters
    pub fn n_params(&self) -> usize {
        self.dim + self.dim * (self.dim + 1) / 2
    }

    /// Get all variational parameters as a flat vector
    /// Layout: [mean, lower-triangular elements of L]
    pub fn get_params(&self) -> Array1<f64> {
        let n_tril = self.dim * (self.dim + 1) / 2;
        let mut params = Array1::zeros(self.dim + n_tril);
        // Mean
        for i in 0..self.dim {
            params[i] = self.mean[i];
        }
        // Lower triangular
        let mut idx = self.dim;
        for i in 0..self.dim {
            for j in 0..=i {
                params[idx] = self.chol_factor[[i, j]];
                idx += 1;
            }
        }
        params
    }

    /// Set variational parameters from a flat vector
    pub fn set_params(&mut self, params: &Array1<f64>) -> Result<()> {
        let n_tril = self.dim * (self.dim + 1) / 2;
        let expected = self.dim + n_tril;
        if params.len() != expected {
            return Err(StatsError::DimensionMismatch(format!(
                "params length ({}) must be {}",
                params.len(),
                expected
            )));
        }
        // Mean
        for i in 0..self.dim {
            self.mean[i] = params[i];
        }
        // Lower triangular
        let mut idx = self.dim;
        self.chol_factor = Array2::zeros((self.dim, self.dim));
        for i in 0..self.dim {
            for j in 0..=i {
                self.chol_factor[[i, j]] = params[idx];
                idx += 1;
            }
        }
        Ok(())
    }
}

/// Normalizing flow variational family (placeholder/scaffold)
///
/// This provides a framework for flow-based variational inference where
/// the posterior is represented as a composition of invertible transformations
/// applied to a base distribution (typically a standard Gaussian).
///
/// q(z) = q_0(f^{-1}(z)) * |det(df^{-1}/dz)|
///
/// Currently supports:
/// - Planar flows: f(z) = z + u * h(w^T z + b)
/// - Radial flows: f(z) = z + beta * h(alpha, r)(z - z0)
#[derive(Debug, Clone)]
pub struct NormalizingFlowVI {
    /// Base distribution (mean-field Gaussian)
    pub base: MeanFieldGaussian,
    /// Flow layers
    pub flows: Vec<FlowLayer>,
    /// Dimensionality
    pub dim: usize,
}

/// A single flow layer (invertible transformation)
#[derive(Debug, Clone)]
pub enum FlowLayer {
    /// Planar flow: f(z) = z + u * tanh(w^T z + b)
    Planar {
        /// Direction of perturbation
        u: Array1<f64>,
        /// Weight vector
        w: Array1<f64>,
        /// Bias
        b: f64,
    },
    /// Radial flow: f(z) = z + beta * h(alpha, r)(z - z0)
    Radial {
        /// Center point
        z0: Array1<f64>,
        /// Scale parameter (log-parameterized for positivity)
        log_alpha: f64,
        /// Contraction/expansion parameter
        beta: f64,
    },
}

impl NormalizingFlowVI {
    /// Create a new normalizing flow VI with a base mean-field Gaussian
    pub fn new(dim: usize, n_flows: usize) -> Result<Self> {
        check_positive(dim, "dim")?;
        let base = MeanFieldGaussian::new(dim)?;

        // Initialize with identity-like planar flows
        let mut flows = Vec::with_capacity(n_flows);
        for _ in 0..n_flows {
            let u = Array1::from_elem(dim, 0.01);
            let w = Array1::from_elem(dim, 0.01);
            flows.push(FlowLayer::Planar { u, w, b: 0.0 });
        }

        Ok(Self { base, flows, dim })
    }

    /// Add a planar flow layer
    pub fn add_planar_flow(&mut self, u: Array1<f64>, w: Array1<f64>, b: f64) -> Result<()> {
        if u.len() != self.dim || w.len() != self.dim {
            return Err(StatsError::DimensionMismatch(format!(
                "u ({}) and w ({}) must have dimension {}",
                u.len(),
                w.len(),
                self.dim
            )));
        }
        self.flows.push(FlowLayer::Planar { u, w, b });
        Ok(())
    }

    /// Add a radial flow layer
    pub fn add_radial_flow(&mut self, z0: Array1<f64>, log_alpha: f64, beta: f64) -> Result<()> {
        if z0.len() != self.dim {
            return Err(StatsError::DimensionMismatch(format!(
                "z0 ({}) must have dimension {}",
                z0.len(),
                self.dim
            )));
        }
        self.flows.push(FlowLayer::Radial {
            z0,
            log_alpha,
            beta,
        });
        Ok(())
    }

    /// Transform a sample through all flow layers, returning the transformed
    /// sample and the sum of log-abs-det-Jacobians
    pub fn transform(&self, z0: &Array1<f64>) -> Result<(Array1<f64>, f64)> {
        if z0.len() != self.dim {
            return Err(StatsError::DimensionMismatch(format!(
                "z0 length ({}) must match dimension ({})",
                z0.len(),
                self.dim
            )));
        }
        let mut z = z0.clone();
        let mut sum_log_det_jac = 0.0;

        for flow in &self.flows {
            let (z_new, log_det) = apply_flow_layer(flow, &z)?;
            z = z_new;
            sum_log_det_jac += log_det;
        }

        Ok((z, sum_log_det_jac))
    }

    /// Sample from the flow-transformed distribution
    pub fn sample(&self, epsilon: &Array1<f64>) -> Result<(Array1<f64>, f64)> {
        let z0 = self.base.sample(epsilon)?;
        let (z_k, sum_log_det) = self.transform(&z0)?;
        let log_q0 = self.base.log_prob(&z0)?;
        // log q_K(z_K) = log q_0(z_0) - sum log|det J_k|
        let log_q_k = log_q0 - sum_log_det;
        Ok((z_k, log_q_k))
    }

    /// Get the number of flow parameters
    pub fn n_flow_params(&self) -> usize {
        self.flows
            .iter()
            .map(|f| match f {
                FlowLayer::Planar { u, w, .. } => u.len() + w.len() + 1,
                FlowLayer::Radial { z0, .. } => z0.len() + 2,
            })
            .sum()
    }
}

/// Apply a single flow layer to a point z
fn apply_flow_layer(flow: &FlowLayer, z: &Array1<f64>) -> Result<(Array1<f64>, f64)> {
    match flow {
        FlowLayer::Planar { u, w, b } => {
            // f(z) = z + u * tanh(w^T z + b)
            let activation = w.dot(z) + b;
            let tanh_val = activation.tanh();
            let z_new = z + &(u * tanh_val);

            // log|det J| = log|1 + u^T * h'(w^T z + b) * w|
            let dtanh = 1.0 - tanh_val * tanh_val;
            let psi = w * dtanh;
            let det = 1.0 + u.dot(&psi);
            let log_det = det.abs().ln();

            Ok((z_new, log_det))
        }
        FlowLayer::Radial {
            z0,
            log_alpha,
            beta,
        } => {
            let alpha = log_alpha.exp();
            let diff = z - z0;
            let r = diff.dot(&diff).sqrt();
            let h = 1.0 / (alpha + r);
            let z_new = z + &(&diff * (*beta * h));

            // log|det J| for radial flow
            let d = z.len() as f64;
            let h_prime = -1.0 / ((alpha + r) * (alpha + r));
            let term1 = (1.0 + beta * h).powi(d as i32 - 1);
            let term2 = 1.0 + beta * h + beta * h_prime * r;
            let det = term1 * term2;
            let log_det = det.abs().ln();

            Ok((z_new, log_det))
        }
    }
}

// ============================================================================
// Bayesian Linear Regression (existing)
// ============================================================================

/// Mean-field variational inference for Bayesian linear regression
///
/// Approximates the posterior with a factorized normal distribution:
/// q(beta, tau) = q(beta)q(tau) where q(beta) ~ N(mu_beta, Sigma_beta) and q(tau) ~ Gamma(a_tau, b_tau)
#[derive(Debug, Clone)]
pub struct VariationalBayesianRegression {
    /// Variational mean for coefficients
    pub mean_beta: Array1<f64>,
    /// Variational covariance for coefficients
    pub cov_beta: Array2<f64>,
    /// Variational shape parameter for precision
    pub shape_tau: f64,
    /// Variational rate parameter for precision
    pub rate_tau: f64,
    /// Prior parameters
    pub prior_mean_beta: Array1<f64>,
    pub prior_cov_beta: Array2<f64>,
    pub priorshape_tau: f64,
    pub prior_rate_tau: f64,
    /// Model dimensionality
    pub n_features: usize,
    /// Whether to fit intercept
    pub fit_intercept: bool,
}

impl VariationalBayesianRegression {
    /// Create a new variational Bayesian regression model
    pub fn new(n_features: usize, fit_intercept: bool) -> Result<Self> {
        check_positive(n_features, "n_features")?;

        // Initialize with weakly informative priors
        let prior_mean_beta = Array1::zeros(n_features);
        let prior_cov_beta = Array2::eye(n_features) * 100.0; // Large variance = weak prior
        let priorshape_tau = 1e-3;
        let prior_rate_tau = 1e-3;

        Ok(Self {
            mean_beta: prior_mean_beta.clone(),
            cov_beta: prior_cov_beta.clone(),
            shape_tau: priorshape_tau,
            rate_tau: prior_rate_tau,
            prior_mean_beta,
            prior_cov_beta,
            priorshape_tau,
            prior_rate_tau,
            n_features,
            fit_intercept,
        })
    }

    /// Set custom priors
    pub fn with_priors(
        mut self,
        prior_mean_beta: Array1<f64>,
        prior_cov_beta: Array2<f64>,
        priorshape_tau: f64,
        prior_rate_tau: f64,
    ) -> Result<Self> {
        checkarray_finite(&prior_mean_beta, "prior_mean_beta")?;
        checkarray_finite(&prior_cov_beta, "prior_cov_beta")?;
        check_positive(priorshape_tau, "priorshape_tau")?;
        check_positive(prior_rate_tau, "prior_rate_tau")?;

        self.prior_mean_beta = prior_mean_beta.clone();
        self.prior_cov_beta = prior_cov_beta.clone();
        self.priorshape_tau = priorshape_tau;
        self.prior_rate_tau = prior_rate_tau;
        self.mean_beta = prior_mean_beta;
        self.cov_beta = prior_cov_beta;
        self.shape_tau = priorshape_tau;
        self.rate_tau = prior_rate_tau;

        Ok(self)
    }

    /// Fit the model using coordinate ascent variational inference
    pub fn fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        max_iter: usize,
        tol: f64,
    ) -> Result<VariationalRegressionResult> {
        checkarray_finite(&x, "x")?;
        checkarray_finite(&y, "y")?;
        check_positive(max_iter, "max_iter")?;
        check_positive(tol, "tol")?;

        let (n_samples_, n_features) = x.dim();
        if y.len() != n_samples_ {
            return Err(StatsError::DimensionMismatch(format!(
                "y length ({}) must match x rows ({})",
                y.len(),
                n_samples_
            )));
        }

        if n_features != self.n_features {
            return Err(StatsError::DimensionMismatch(format!(
                "x features ({}) must match model features ({})",
                n_features, self.n_features
            )));
        }

        // Center data if fitting intercept
        let (x_centered, y_centered, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = x.mean_axis(Axis(0)).expect("Operation failed");
            let y_mean = y.mean();

            let mut x_centered = x.to_owned();
            for mut row in x_centered.rows_mut() {
                row -= &x_mean;
            }
            let y_centered = &y.to_owned() - y_mean;

            (x_centered, y_centered, Some(x_mean), Some(y_mean))
        } else {
            (x.to_owned(), y.to_owned(), None, None)
        };

        // Precompute matrices
        let xtx = x_centered.t().dot(&x_centered);
        let xty = x_centered.t().dot(&y_centered);
        let yty = y_centered.dot(&y_centered);

        // Prior precision matrix
        let prior_precision =
            scirs2_linalg::inv(&self.prior_cov_beta.view(), None).map_err(|e| {
                StatsError::ComputationError(format!("Failed to invert prior covariance: {}", e))
            })?;

        let mut prev_elbo = f64::NEG_INFINITY;
        let mut elbo_history = Vec::new();

        for _iter in 0..max_iter {
            // Update q(beta)
            self.update_beta_variational(&xtx, &xty, &prior_precision)?;

            // Update q(tau)
            self.update_tau_variational(n_samples_ as f64, &xtx, yty)?;

            // Compute ELBO
            let elbo = self.compute_elbo(n_samples_ as f64, &xtx, &xty, yty, &prior_precision)?;
            elbo_history.push(elbo);

            // Check convergence
            if _iter > 0 && (elbo - prev_elbo).abs() < tol {
                break;
            }

            prev_elbo = elbo;
        }

        Ok(VariationalRegressionResult {
            mean_beta: self.mean_beta.clone(),
            cov_beta: self.cov_beta.clone(),
            shape_tau: self.shape_tau,
            rate_tau: self.rate_tau,
            elbo: prev_elbo,
            elbo_history: elbo_history.clone(),
            n_samples_,
            n_features: self.n_features,
            x_mean,
            y_mean,
            converged: elbo_history.len() < max_iter,
        })
    }

    /// Update variational distribution for beta
    fn update_beta_variational(
        &mut self,
        xtx: &Array2<f64>,
        xty: &Array1<f64>,
        prior_precision: &Array2<f64>,
    ) -> Result<()> {
        // Expected precision: E[tau] = shape / rate
        let expected_tau = self.shape_tau / self.rate_tau;

        // Posterior precision
        let precision_beta = prior_precision + &(xtx * expected_tau);

        // Posterior covariance
        self.cov_beta = scirs2_linalg::inv(&precision_beta.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to invert precision: {}", e))
        })?;

        // Posterior mean
        let prior_contrib = prior_precision.dot(&self.prior_mean_beta);
        let data_contrib = xty * expected_tau;
        self.mean_beta = self.cov_beta.dot(&(prior_contrib + data_contrib));

        Ok(())
    }

    /// Update variational distribution for tau
    fn update_tau_variational(
        &mut self,
        n_samples_: f64,
        xtx: &Array2<f64>,
        yty: f64,
    ) -> Result<()> {
        // Shape parameter
        self.shape_tau = self.priorshape_tau + n_samples_ / 2.0;

        // Rate parameter
        let expected_beta_outer = &self.cov_beta + outer_product(&self.mean_beta);
        let trace_term = (xtx * &expected_beta_outer).sum();
        let quadratic_term = 2.0 * self.mean_beta.dot(&xtx.dot(&self.mean_beta));

        self.rate_tau = self.prior_rate_tau + 0.5 * (yty - quadratic_term + trace_term);

        Ok(())
    }

    /// Compute Evidence Lower BOund (ELBO)
    fn compute_elbo(
        &self,
        n_samples_: f64,
        xtx: &Array2<f64>,
        xty: &Array1<f64>,
        yty: f64,
        prior_precision: &Array2<f64>,
    ) -> Result<f64> {
        let expected_tau = self.shape_tau / self.rate_tau;
        let expected_log_tau = digamma(self.shape_tau) - self.rate_tau.ln();

        // E[log p(y|X,beta,tau)]
        let diff =
            yty - 2.0 * self.mean_beta.dot(xty) + self.mean_beta.dot(&xtx.dot(&self.mean_beta));
        let trace_term = (xtx * &self.cov_beta).sum();
        let likelihood_term = 0.5 * n_samples_ * expected_log_tau
            - 0.5 * n_samples_ * (2.0_f64 * PI).ln()
            - 0.5 * expected_tau * (diff + trace_term);

        // E[log p(beta)]
        let beta_diff = &self.mean_beta - &self.prior_mean_beta;
        let beta_quad = beta_diff.dot(&prior_precision.dot(&beta_diff));
        let beta_trace = (prior_precision * &self.cov_beta).sum();

        let prior_det = scirs2_linalg::det(&prior_precision.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to compute determinant: {}", e))
        })?;

        let beta_prior_term = 0.5 * prior_det.ln()
            - 0.5 * self.n_features as f64 * (2.0_f64 * PI).ln()
            - 0.5 * (beta_quad + beta_trace);

        // E[log p(tau)]
        let tau_prior_term = self.priorshape_tau * self.prior_rate_tau.ln()
            - lgamma(self.priorshape_tau)
            + (self.priorshape_tau - 1.0) * expected_log_tau
            - self.prior_rate_tau * expected_tau;

        // -E[log q(beta)]
        let var_det = scirs2_linalg::det(&self.cov_beta.view(), None).map_err(|e| {
            StatsError::ComputationError(format!("Failed to compute determinant: {}", e))
        })?;
        let beta_entropy =
            0.5 * self.n_features as f64 * (1.0 + (2.0_f64 * PI).ln()) + 0.5 * var_det.ln();

        // -E[log q(tau)]
        let tau_entropy = self.shape_tau - self.rate_tau.ln()
            + lgamma(self.shape_tau)
            + (1.0 - self.shape_tau) * digamma(self.shape_tau);

        Ok(likelihood_term + beta_prior_term + tau_prior_term + beta_entropy + tau_entropy)
    }

    /// Predict on new data
    pub fn predict(
        &self,
        x: ArrayView2<f64>,
        result: &VariationalRegressionResult,
    ) -> Result<VariationalPredictionResult> {
        checkarray_finite(&x, "x")?;
        let (n_test, n_features) = x.dim();

        if n_features != result.n_features {
            return Err(StatsError::DimensionMismatch(format!(
                "x has {} features, expected {}",
                n_features, result.n_features
            )));
        }

        // Center test data if model was fit with intercept
        let x_centered = if let Some(ref x_mean) = result.x_mean {
            let mut x_c = x.to_owned();
            for mut row in x_c.rows_mut() {
                row -= x_mean;
            }
            x_c
        } else {
            x.to_owned()
        };

        // Predictive mean
        let y_pred_centered = x_centered.dot(&result.mean_beta);
        let y_pred = if let Some(y_mean) = result.y_mean {
            &y_pred_centered + y_mean
        } else {
            y_pred_centered.clone()
        };

        // Predictive variance
        let expected_noise_variance = result.rate_tau / result.shape_tau;
        let mut predictive_variance = Array1::zeros(n_test);

        for i in 0..n_test {
            let x_row = x_centered.row(i);
            let model_variance = x_row.dot(&result.cov_beta.dot(&x_row));
            predictive_variance[i] = expected_noise_variance + model_variance;
        }

        Ok(VariationalPredictionResult {
            mean: y_pred,
            variance: predictive_variance.clone(),
            model_uncertainty: predictive_variance.mapv(|v| (v - expected_noise_variance).max(0.0)),
            noise_variance: expected_noise_variance,
        })
    }
}

/// Results from variational Bayesian regression
#[derive(Debug, Clone)]
pub struct VariationalRegressionResult {
    /// Posterior mean of coefficients
    pub mean_beta: Array1<f64>,
    /// Posterior covariance of coefficients
    pub cov_beta: Array2<f64>,
    /// Posterior shape parameter for precision
    pub shape_tau: f64,
    /// Posterior rate parameter for precision
    pub rate_tau: f64,
    /// Final ELBO value
    pub elbo: f64,
    /// ELBO history during optimization
    pub elbo_history: Vec<f64>,
    /// Number of training samples
    pub n_samples_: usize,
    /// Number of features
    pub n_features: usize,
    /// Training data mean (for centering)
    pub x_mean: Option<Array1<f64>>,
    /// Training target mean (for centering)
    pub y_mean: Option<f64>,
    /// Whether optimization converged
    pub converged: bool,
}

impl VariationalRegressionResult {
    /// Get posterior standard deviations of coefficients
    pub fn std_beta(&self) -> Array1<f64> {
        self.cov_beta.diag().mapv(f64::sqrt)
    }

    /// Get posterior mean and standard deviation of noise precision
    pub fn precision_stats(&self) -> (f64, f64) {
        let mean = self.shape_tau / self.rate_tau;
        let variance = self.shape_tau / (self.rate_tau * self.rate_tau);
        (mean, variance.sqrt())
    }

    /// Compute credible intervals for coefficients
    pub fn credible_intervals(&self, confidence: f64) -> Result<Array2<f64>> {
        check_probability(confidence, "confidence")?;

        let n_features = self.mean_beta.len();
        let mut intervals = Array2::zeros((n_features, 2));
        let alpha = (1.0 - confidence) / 2.0;

        // Use normal approximation for coefficients
        for i in 0..n_features {
            let mean = self.mean_beta[i];
            let std = self.cov_beta[[i, i]].sqrt();

            // Using standard normal quantiles (approximate)
            let z_critical = normal_ppf(1.0 - alpha)?;
            intervals[[i, 0]] = mean - z_critical * std;
            intervals[[i, 1]] = mean + z_critical * std;
        }

        Ok(intervals)
    }
}

/// Results from variational prediction
#[derive(Debug, Clone)]
pub struct VariationalPredictionResult {
    /// Predictive mean
    pub mean: Array1<f64>,
    /// Total predictive variance (model + noise)
    pub variance: Array1<f64>,
    /// Model uncertainty component
    pub model_uncertainty: Array1<f64>,
    /// Noise variance
    pub noise_variance: f64,
}

impl VariationalPredictionResult {
    /// Get predictive standard deviations
    pub fn std(&self) -> Array1<f64> {
        self.variance.mapv(f64::sqrt)
    }

    /// Compute predictive credible intervals
    pub fn credible_intervals(&self, confidence: f64) -> Result<Array2<f64>> {
        check_probability(confidence, "confidence")?;

        let n_predictions = self.mean.len();
        let mut intervals = Array2::zeros((n_predictions, 2));
        let alpha = (1.0 - confidence) / 2.0;

        let z_critical = normal_ppf(1.0 - alpha)?;

        for i in 0..n_predictions {
            let mean = self.mean[i];
            let std = self.variance[i].sqrt();
            intervals[[i, 0]] = mean - z_critical * std;
            intervals[[i, 1]] = mean + z_critical * std;
        }

        Ok(intervals)
    }
}

// ============================================================================
// Automatic Relevance Determination (existing)
// ============================================================================

/// Automatic Relevance Determination with Variational Inference
///
/// Uses sparse priors to perform automatic feature selection
#[derive(Debug, Clone)]
pub struct VariationalARD {
    /// Variational mean for coefficients
    pub mean_beta: Array1<f64>,
    /// Variational variance for coefficients (diagonal)
    pub var_beta: Array1<f64>,
    /// Variational parameters for precision (alpha)
    pub shape_alpha: Array1<f64>,
    pub rate_alpha: Array1<f64>,
    /// Variational parameters for noise precision
    pub shape_tau: f64,
    pub rate_tau: f64,
    /// Prior parameters
    pub priorshape_alpha: f64,
    pub prior_rate_alpha: f64,
    pub priorshape_tau: f64,
    pub prior_rate_tau: f64,
    /// Model parameters
    pub n_features: usize,
    pub fit_intercept: bool,
}

impl VariationalARD {
    /// Create new Variational ARD model
    pub fn new(n_features: usize, fit_intercept: bool) -> Result<Self> {
        check_positive(n_features, "n_features")?;

        // Weakly informative priors
        let priorshape_alpha = 1e-3;
        let prior_rate_alpha = 1e-3;
        let priorshape_tau = 1e-3;
        let prior_rate_tau = 1e-3;

        Ok(Self {
            mean_beta: Array1::zeros(n_features),
            var_beta: Array1::from_elem(n_features, 1.0),
            shape_alpha: Array1::from_elem(n_features, priorshape_alpha),
            rate_alpha: Array1::from_elem(n_features, prior_rate_alpha),
            shape_tau: priorshape_tau,
            rate_tau: prior_rate_tau,
            priorshape_alpha,
            prior_rate_alpha,
            priorshape_tau,
            prior_rate_tau,
            n_features,
            fit_intercept,
        })
    }

    /// Fit ARD model using variational inference
    pub fn fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        max_iter: usize,
        tol: f64,
    ) -> Result<VariationalARDResult> {
        checkarray_finite(&x, "x")?;
        checkarray_finite(&y, "y")?;
        check_positive(max_iter, "max_iter")?;
        check_positive(tol, "tol")?;

        let (n_samples_, n_features) = x.dim();
        if y.len() != n_samples_ {
            return Err(StatsError::DimensionMismatch(format!(
                "y length ({}) must match x rows ({})",
                y.len(),
                n_samples_
            )));
        }

        // Center data if fitting intercept
        let (x_centered, y_centered, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = x.mean_axis(Axis(0)).expect("Operation failed");
            let y_mean = y.mean();

            let mut x_centered = x.to_owned();
            for mut row in x_centered.rows_mut() {
                row -= &x_mean;
            }
            let y_centered = &y.to_owned() - y_mean;

            (x_centered, y_centered, Some(x_mean), Some(y_mean))
        } else {
            (x.to_owned(), y.to_owned(), None, None)
        };

        // Precompute matrices
        let xtx = x_centered.t().dot(&x_centered);
        let xty = x_centered.t().dot(&y_centered);
        let yty = y_centered.dot(&y_centered);

        let mut prev_elbo = f64::NEG_INFINITY;
        let mut elbo_history = Vec::new();

        for _iter in 0..max_iter {
            // Update q(beta)
            self.update_beta_ard(&xtx, &xty)?;

            // Update q(alpha)
            self.update_alpha_ard()?;

            // Update q(tau)
            self.update_tau_ard(n_samples_ as f64, &xtx, yty)?;

            // Compute ELBO
            let elbo = self.compute_elbo_ard(n_samples_ as f64, &xtx, &xty, yty)?;
            elbo_history.push(elbo);

            // Check convergence
            if _iter > 0 && (elbo - prev_elbo).abs() < tol {
                break;
            }

            // Prune irrelevant features
            if _iter % 10 == 0 {
                self.prune_features()?;
            }

            prev_elbo = elbo;
        }

        Ok(VariationalARDResult {
            mean_beta: self.mean_beta.clone(),
            var_beta: self.var_beta.clone(),
            shape_alpha: self.shape_alpha.clone(),
            rate_alpha: self.rate_alpha.clone(),
            shape_tau: self.shape_tau,
            rate_tau: self.rate_tau,
            elbo: prev_elbo,
            elbo_history: elbo_history.clone(),
            n_samples_,
            n_features: self.n_features,
            x_mean,
            y_mean,
            converged: elbo_history.len() < max_iter,
        })
    }

    /// Update variational distribution for beta in ARD model
    fn update_beta_ard(&mut self, xtx: &Array2<f64>, xty: &Array1<f64>) -> Result<()> {
        let expected_tau = self.shape_tau / self.rate_tau;
        let expected_alpha = &self.shape_alpha / &self.rate_alpha;

        // Update variance (diagonal approximation)
        for i in 0..self.n_features {
            let precision_i = expected_alpha[i] + expected_tau * xtx[[i, i]];
            self.var_beta[i] = 1.0 / precision_i;
        }

        // Update mean
        for i in 0..self.n_features {
            let sum_j = (0..self.n_features)
                .filter(|&j| j != i)
                .map(|j| xtx[[i, j]] * self.mean_beta[j])
                .sum::<f64>();

            self.mean_beta[i] = expected_tau * self.var_beta[i] * (xty[i] - sum_j);
        }

        Ok(())
    }

    /// Update variational distribution for alpha (precision parameters)
    fn update_alpha_ard(&mut self) -> Result<()> {
        for i in 0..self.n_features {
            self.shape_alpha[i] = self.priorshape_alpha + 0.5;
            self.rate_alpha[i] =
                self.prior_rate_alpha + 0.5 * (self.mean_beta[i].powi(2) + self.var_beta[i]);
        }

        Ok(())
    }

    /// Update variational distribution for tau (noise precision)
    fn update_tau_ard(&mut self, n_samples_: f64, xtx: &Array2<f64>, yty: f64) -> Result<()> {
        self.shape_tau = self.priorshape_tau + n_samples_ / 2.0;

        let mut quadratic_term = 0.0;
        for i in 0..self.n_features {
            for j in 0..self.n_features {
                if i == j {
                    quadratic_term += xtx[[i, j]] * (self.mean_beta[i].powi(2) + self.var_beta[i]);
                } else {
                    quadratic_term += xtx[[i, j]] * self.mean_beta[i] * self.mean_beta[j];
                }
            }
        }

        self.rate_tau = self.prior_rate_tau
            + 0.5 * (yty - 2.0 * self.mean_beta.dot(&xtx.dot(&self.mean_beta)) + quadratic_term);

        Ok(())
    }

    /// Compute ELBO for ARD model
    fn compute_elbo_ard(
        &self,
        n_samples_: f64,
        xtx: &Array2<f64>,
        xty: &Array1<f64>,
        yty: f64,
    ) -> Result<f64> {
        let expected_tau = self.shape_tau / self.rate_tau;
        let expected_log_tau = digamma(self.shape_tau) - self.rate_tau.ln();

        // Likelihood term
        let mut quadratic_form = yty - 2.0 * self.mean_beta.dot(xty);
        for i in 0..self.n_features {
            for j in 0..self.n_features {
                if i == j {
                    quadratic_form += xtx[[i, j]] * (self.mean_beta[i].powi(2) + self.var_beta[i]);
                } else {
                    quadratic_form += xtx[[i, j]] * self.mean_beta[i] * self.mean_beta[j];
                }
            }
        }

        let likelihood_term = 0.5 * n_samples_ * expected_log_tau
            - 0.5 * n_samples_ * (2.0_f64 * PI).ln()
            - 0.5 * expected_tau * quadratic_form;

        // Prior terms
        let mut prior_term = 0.0;
        for i in 0..self.n_features {
            let expected_alpha_i = self.shape_alpha[i] / self.rate_alpha[i];
            let expected_log_alpha_i = digamma(self.shape_alpha[i]) - self.rate_alpha[i].ln();

            prior_term += 0.5 * expected_log_alpha_i
                - 0.5 * (2.0_f64 * PI).ln()
                - 0.5 * expected_alpha_i * (self.mean_beta[i].powi(2) + self.var_beta[i]);
        }

        // Entropy terms
        let mut entropy_term = 0.0;
        for i in 0..self.n_features {
            entropy_term += 0.5 * (1.0 + (2.0 * PI * self.var_beta[i]).ln());
        }

        Ok(likelihood_term + prior_term + entropy_term)
    }

    /// Prune features with small precision (large variance in prior)
    fn prune_features(&mut self) -> Result<()> {
        let threshold = 1e12; // Large precision = irrelevant feature

        for i in 0..self.n_features {
            let expected_alpha = self.shape_alpha[i] / self.rate_alpha[i];
            if expected_alpha > threshold {
                // Feature is irrelevant, set to zero
                self.mean_beta[i] = 0.0;
                self.var_beta[i] = 1e-12;
            }
        }

        Ok(())
    }

    /// Get relevance scores for features
    pub fn feature_relevance(&self) -> Array1<f64> {
        let expected_alpha = &self.shape_alpha / &self.rate_alpha;
        // Relevance is inverse of precision (features with low precision are more relevant)
        expected_alpha.mapv(|alpha| 1.0 / alpha)
    }
}

/// Results from Variational ARD
#[derive(Debug, Clone)]
pub struct VariationalARDResult {
    /// Posterior mean of coefficients
    pub mean_beta: Array1<f64>,
    /// Posterior variance of coefficients
    pub var_beta: Array1<f64>,
    /// Posterior shape parameters for feature precisions
    pub shape_alpha: Array1<f64>,
    /// Posterior rate parameters for feature precisions
    pub rate_alpha: Array1<f64>,
    /// Posterior shape parameter for noise precision
    pub shape_tau: f64,
    /// Posterior rate parameter for noise precision
    pub rate_tau: f64,
    /// Final ELBO value
    pub elbo: f64,
    /// ELBO history
    pub elbo_history: Vec<f64>,
    /// Number of training samples
    pub n_samples_: usize,
    /// Number of features
    pub n_features: usize,
    /// Training data mean
    pub x_mean: Option<Array1<f64>>,
    /// Training target mean
    pub y_mean: Option<f64>,
    /// Whether optimization converged
    pub converged: bool,
}

impl VariationalARDResult {
    /// Get selected features based on relevance threshold
    pub fn selected_features(&self, threshold: f64) -> Vec<usize> {
        let expected_alpha = &self.shape_alpha / &self.rate_alpha;
        expected_alpha
            .iter()
            .enumerate()
            .filter(|(_, &alpha)| alpha < threshold) // Low precision = high relevance
            .map(|(i, _)| i)
            .collect()
    }

    /// Get feature importance scores
    pub fn feature_importance(&self) -> Array1<f64> {
        self.mean_beta.mapv(f64::abs)
    }
}

// ============================================================================
// Shared helper functions
// ============================================================================

/// Compute outer product of a vector with itself
pub(crate) fn outer_product(v: &Array1<f64>) -> Array2<f64> {
    let n = v.len();
    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = v[i] * v[j];
        }
    }
    result
}

/// Approximate normal PPF using rational approximation (Beasley-Springer-Moro)
pub(crate) fn normal_ppf(p: f64) -> Result<f64> {
    if p <= 0.0 || p >= 1.0 {
        return Err(StatsError::InvalidArgument(
            "p must be between 0 and 1".to_string(),
        ));
    }

    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];

    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];

    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];

    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        Ok(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0),
        )
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        Ok(
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0),
        )
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        Ok(
            (-((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0),
        )
    }
}

/// Digamma function (approximate)
pub(crate) fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    if x < 8.0 {
        return digamma(x + 1.0) - 1.0 / x;
    }

    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;

    x.ln() - 0.5 * inv_x - inv_x2 / 12.0 + inv_x2 * inv_x2 / 120.0
        - inv_x2 * inv_x2 * inv_x2 / 252.0
}

/// Log gamma function (approximate using Stirling's series)
pub(crate) fn lgamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    // Use reflection formula for x < 0.5 to improve accuracy
    if x < 0.5 {
        // Reflection: lgamma(x) = ln(pi/sin(pi*x)) - lgamma(1-x)
        return (PI / (PI * x).sin()).ln() - lgamma(1.0 - x);
    }

    // Lanczos approximation (g=7, n=9) -- accurate to ~15 significant digits
    // Coefficients from Paul Godfrey's implementation
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, &c) in C[1..].iter().enumerate() {
        a += c / (x + (i as f64 + 1.0));
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Trigamma function (derivative of digamma)
pub(crate) fn trigamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    if x < 8.0 {
        return trigamma(x + 1.0) + 1.0 / (x * x);
    }

    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;

    inv_x + 0.5 * inv_x2 + inv_x2 * inv_x / 6.0 - inv_x2 * inv_x2 * inv_x / 30.0
        + inv_x2 * inv_x2 * inv_x2 * inv_x / 42.0
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_mean_field_gaussian_creation() {
        let mf = MeanFieldGaussian::new(5).expect("should create mean-field Gaussian");
        assert_eq!(mf.dim, 5);
        assert_eq!(mf.means.len(), 5);
        assert_eq!(mf.log_stds.len(), 5);
        assert_eq!(mf.n_params(), 10);
    }

    #[test]
    fn test_mean_field_gaussian_entropy() {
        let mf = MeanFieldGaussian::new(2).expect("should create");
        let entropy = mf.entropy();
        // For standard normal in 2D: 2 * 0.5 * (1 + log(2*pi))
        let expected = 2.0 * 0.5 * (1.0 + (2.0 * PI).ln());
        assert!((entropy - expected).abs() < 1e-10);
    }

    #[test]
    fn test_mean_field_gaussian_sample() {
        let mf = MeanFieldGaussian::new(3).expect("should create");
        let epsilon = Array1::from_vec(vec![0.5, -0.3, 1.0]);
        let sample = mf.sample(&epsilon).expect("should sample");
        assert_eq!(sample.len(), 3);
        // With mean=0 and std=1, sample should equal epsilon
        for i in 0..3 {
            assert!((sample[i] - epsilon[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mean_field_gaussian_params_roundtrip() {
        let mut mf = MeanFieldGaussian::new(3).expect("should create");
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0, 0.5, -0.3, 0.1]);
        mf.set_params(&params).expect("should set params");
        let retrieved = mf.get_params();
        for i in 0..6 {
            assert!((retrieved[i] - params[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_full_rank_gaussian_creation() {
        let fr = FullRankGaussian::new(3).expect("should create full-rank Gaussian");
        assert_eq!(fr.dim, 3);
        assert_eq!(fr.mean.len(), 3);
        // n_params = d + d*(d+1)/2 = 3 + 6 = 9
        assert_eq!(fr.n_params(), 9);
    }

    #[test]
    fn test_full_rank_gaussian_entropy() {
        let fr = FullRankGaussian::new(2).expect("should create");
        let entropy = fr.entropy();
        // For identity covariance: 0.5*d*(1+log(2*pi)) + 0 = d * 0.5*(1+log(2pi))
        let expected = 2.0 * 0.5 * (1.0 + (2.0 * PI).ln());
        assert!((entropy - expected).abs() < 1e-10);
    }

    #[test]
    fn test_full_rank_gaussian_sample() {
        let fr = FullRankGaussian::new(2).expect("should create");
        let epsilon = Array1::from_vec(vec![1.0, -1.0]);
        let sample = fr.sample(&epsilon).expect("should sample");
        assert_eq!(sample.len(), 2);
        // With identity chol factor and zero mean, sample = epsilon
        for i in 0..2 {
            assert!((sample[i] - epsilon[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normalizing_flow_creation() {
        let nf = NormalizingFlowVI::new(3, 2).expect("should create");
        assert_eq!(nf.dim, 3);
        assert_eq!(nf.flows.len(), 2);
    }

    #[test]
    fn test_normalizing_flow_transform() {
        let nf = NormalizingFlowVI::new(2, 1).expect("should create");
        let z0 = Array1::from_vec(vec![0.5, -0.5]);
        let (z_k, log_det) = nf.transform(&z0).expect("should transform");
        assert_eq!(z_k.len(), 2);
        assert!(log_det.is_finite());
    }

    #[test]
    fn test_diagnostics() {
        let mut diag = VariationalDiagnostics::new();
        diag.record_elbo(-100.0);
        diag.record_elbo(-90.0);
        diag.record_elbo(-85.0);
        diag.record_gradient_norm(10.0);
        diag.record_gradient_norm(5.0);

        assert_eq!(diag.n_iterations, 3);
        assert!(!diag.check_elbo_convergence(1.0));
        assert!(diag.check_elbo_convergence(10.0));

        let summary = diag.elbo_summary();
        assert!((summary.min - (-100.0)).abs() < 1e-10);
        assert!((summary.max - (-85.0)).abs() < 1e-10);
        assert!(summary.monotonic);
    }

    #[test]
    fn test_variational_bayesian_regression() {
        // Simple regression: y = 2*x + 1 + noise
        let n = 50;
        let mut x_data = Vec::with_capacity(n);
        let mut y_data = Vec::with_capacity(n);

        for i in 0..n {
            let xi = i as f64 / n as f64;
            x_data.push(xi);
            y_data.push(2.0 * xi + 1.0 + 0.1 * ((i * 7 % 13) as f64 - 6.0) / 6.0);
        }

        let x = Array2::from_shape_fn((n, 1), |(i, _)| x_data[i]);
        let y = Array1::from_vec(y_data);

        let mut model = VariationalBayesianRegression::new(1, true).expect("should create model");
        let result = model
            .fit(x.view(), y.view(), 100, 1e-6)
            .expect("should fit");

        // Check that coefficient is close to 2.0
        assert!(
            (result.mean_beta[0] - 2.0).abs() < 0.5,
            "beta should be close to 2.0, got {}",
            result.mean_beta[0]
        );
    }

    #[test]
    fn test_trigamma() {
        // trigamma(1) = pi^2/6
        let expected = PI * PI / 6.0;
        let computed = trigamma(1.0);
        assert!(
            (computed - expected).abs() < 0.01,
            "trigamma(1) should be close to pi^2/6, got {}",
            computed
        );
    }
}
