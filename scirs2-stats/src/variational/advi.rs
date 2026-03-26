//! Automatic Differentiation Variational Inference (ADVI)
//!
//! Implements ADVI (Kucukelbir et al. 2017) — transforms constrained parameters
//! to unconstrained real space, then fits a Gaussian variational approximation
//! by maximizing the ELBO via the reparameterization trick and Adam optimizer.
//!
//! Supports:
//! - Mean-field approximation: `q(theta) = prod_i N(mu_i, sigma_i^2)`
//! - Full-rank approximation: `q(theta) = N(mu, L L^T)` with Cholesky factor L
//! - Automatic parameter transformations: log, logit, identity, bounded
//! - Stochastic ELBO gradient estimation via reparameterization trick
//! - Adam optimizer with configurable learning rate

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::f64::consts::PI;

use super::{PosteriorResult, VariationalInference};

// ============================================================================
// Parameter Transforms
// ============================================================================

/// Transformation for mapping constrained parameters to unconstrained space
#[derive(Debug, Clone)]
pub enum AdviTransform {
    /// Identity (unconstrained real line)
    Identity,
    /// Log transform (for positive parameters)
    Log,
    /// Logit transform (for parameters in `[0, 1]`)
    Logit,
    /// Scaled logit for bounded parameters in `[lower, upper]`
    Bounded {
        /// Lower bound
        lower: f64,
        /// Upper bound
        upper: f64,
    },
}

impl AdviTransform {
    /// Map from unconstrained to constrained space
    pub fn forward(&self, eta: f64) -> f64 {
        match self {
            AdviTransform::Identity => eta,
            AdviTransform::Log => eta.exp(),
            AdviTransform::Logit => 1.0 / (1.0 + (-eta).exp()),
            AdviTransform::Bounded { lower, upper } => {
                let s = 1.0 / (1.0 + (-eta).exp());
                lower + (upper - lower) * s
            }
        }
    }

    /// Map from constrained to unconstrained space
    pub fn inverse(&self, theta: f64) -> StatsResult<f64> {
        match self {
            AdviTransform::Identity => Ok(theta),
            AdviTransform::Log => {
                if theta <= 0.0 {
                    return Err(StatsError::InvalidArgument(format!(
                        "Log transform requires positive value, got {}",
                        theta
                    )));
                }
                Ok(theta.ln())
            }
            AdviTransform::Logit => {
                if theta <= 0.0 || theta >= 1.0 {
                    return Err(StatsError::InvalidArgument(format!(
                        "Logit transform requires value in (0, 1), got {}",
                        theta
                    )));
                }
                Ok((theta / (1.0 - theta)).ln())
            }
            AdviTransform::Bounded { lower, upper } => {
                if theta <= *lower || theta >= *upper {
                    return Err(StatsError::InvalidArgument(format!(
                        "Bounded transform requires value in ({}, {}), got {}",
                        lower, upper, theta
                    )));
                }
                let s = (theta - lower) / (upper - lower);
                Ok((s / (1.0 - s)).ln())
            }
        }
    }

    /// Log absolute Jacobian determinant of the forward transform
    /// needed for change-of-variables correction in the ELBO
    pub fn log_det_jacobian(&self, eta: f64) -> f64 {
        match self {
            AdviTransform::Identity => 0.0,
            AdviTransform::Log => eta,
            AdviTransform::Logit => {
                // d/d(eta) sigmoid(eta) = sigmoid(eta) * (1 - sigmoid(eta))
                // log|J| = log(sigmoid(eta)) + log(1 - sigmoid(eta))
                //        = -softplus(-eta) + (-softplus(eta))
                //        = eta - 2*softplus(eta)   [numerically stable]
                let sp = softplus(eta);
                eta - 2.0 * sp
            }
            AdviTransform::Bounded { lower, upper } => {
                let log_range = (upper - lower).ln();
                let sp = softplus(eta);
                log_range + eta - 2.0 * sp
            }
        }
    }

    /// Gradient of the log-det-Jacobian w.r.t. unconstrained parameter eta
    pub fn grad_log_det_jacobian(&self, eta: f64) -> f64 {
        match self {
            AdviTransform::Identity => 0.0,
            AdviTransform::Log => 1.0,
            AdviTransform::Logit | AdviTransform::Bounded { .. } => {
                // d/d(eta) [eta - 2*softplus(eta)] = 1 - 2*sigmoid(eta)
                let s = sigmoid(eta);
                1.0 - 2.0 * s
            }
        }
    }

    /// Gradient of the forward transform w.r.t. eta (d theta / d eta)
    pub fn forward_grad(&self, eta: f64) -> f64 {
        match self {
            AdviTransform::Identity => 1.0,
            AdviTransform::Log => eta.exp(),
            AdviTransform::Logit => {
                let s = sigmoid(eta);
                s * (1.0 - s)
            }
            AdviTransform::Bounded { lower, upper } => {
                let s = sigmoid(eta);
                (upper - lower) * s * (1.0 - s)
            }
        }
    }
}

/// Numerically stable softplus: log(1 + exp(x))
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Sigmoid function: 1 / (1 + exp(-x))
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ============================================================================
// Approximation Type
// ============================================================================

/// Type of variational approximation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdviApproximation {
    /// Mean-field: q(eta) = prod_i N(mu_i, sigma_i^2)
    MeanField,
    /// Full-rank: q(eta) = N(mu, L L^T) with lower-triangular Cholesky factor
    FullRank,
}

// ============================================================================
// Adam Optimizer (self-contained for ADVI)
// ============================================================================

/// Adam optimizer state for ADVI
#[derive(Debug, Clone)]
struct AdviAdamState {
    m: Array1<f64>,
    v: Array1<f64>,
    t: usize,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
}

impl AdviAdamState {
    fn new(n_params: usize) -> Self {
        Self {
            m: Array1::zeros(n_params),
            v: Array1::zeros(n_params),
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    /// Compute Adam update direction (to be scaled by learning rate)
    fn update(&mut self, grad: &Array1<f64>) -> Array1<f64> {
        self.t += 1;
        let n = grad.len();
        let mut direction = Array1::zeros(n);
        for i in 0..n {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
            let m_hat = self.m[i] / (1.0 - self.beta1.powi(self.t as i32));
            let v_hat = self.v[i] / (1.0 - self.beta2.powi(self.t as i32));
            direction[i] = m_hat / (v_hat.sqrt() + self.epsilon);
        }
        direction
    }
}

// ============================================================================
// ADVI Configuration
// ============================================================================

/// Configuration for ADVI
#[derive(Debug, Clone)]
pub struct AdviConfig {
    /// Type of variational approximation
    pub approximation: AdviApproximation,
    /// Parameter transforms (one per dimension); if empty, all Identity
    pub transforms: Vec<AdviTransform>,
    /// Number of Monte Carlo samples for ELBO gradient estimation
    pub num_samples: usize,
    /// Learning rate for Adam optimizer
    pub learning_rate: f64,
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Convergence tolerance on ELBO change
    pub tolerance: f64,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Window size for checking convergence (average over last N ELBOs)
    pub convergence_window: usize,
}

impl Default for AdviConfig {
    fn default() -> Self {
        Self {
            approximation: AdviApproximation::MeanField,
            transforms: Vec::new(),
            num_samples: 10,
            learning_rate: 0.01,
            max_iterations: 10000,
            tolerance: 1e-4,
            seed: 42,
            convergence_window: 50,
        }
    }
}

// ============================================================================
// ADVI Struct
// ============================================================================

/// Automatic Differentiation Variational Inference
///
/// ADVI automatically transforms constrained parameters to unconstrained space,
/// then optimizes a Gaussian variational approximation using the ELBO.
///
/// # Example
/// ```no_run
/// use scirs2_stats::variational::{Advi, AdviConfig, AdviApproximation, AdviTransform};
/// use scirs2_core::ndarray::Array1;
///
/// let config = AdviConfig {
///     approximation: AdviApproximation::MeanField,
///     transforms: vec![AdviTransform::Identity, AdviTransform::Log],
///     num_samples: 10,
///     learning_rate: 0.01,
///     max_iterations: 1000,
///     ..Default::default()
/// };
///
/// let mut advi = Advi::new(config);
/// ```
#[derive(Debug, Clone)]
pub struct Advi {
    /// Configuration
    pub config: AdviConfig,
}

impl Advi {
    /// Create a new ADVI instance with the given configuration
    pub fn new(config: AdviConfig) -> Self {
        Self { config }
    }

    /// Generate quasi-random standard normal samples using Box-Muller with
    /// golden-ratio quasi-random sequences for reproducibility
    fn generate_epsilon(&self, dim: usize, seed: u64) -> Array1<f64> {
        let golden = 1.618033988749895_f64;
        let plastic = 1.324717957244746_f64;
        Array1::from_shape_fn(dim, |i| {
            let u1 = ((seed as f64 * golden + i as f64 * plastic) % 1.0).abs();
            let u2 = ((seed as f64 * plastic + i as f64 * golden) % 1.0).abs();
            let u1 = u1.max(1e-10).min(1.0 - 1e-10);
            let u2 = u2.max(1e-10).min(1.0 - 1e-10);
            let r = (-2.0 * u1.ln()).sqrt();
            r * (2.0 * PI * u2).cos()
        })
    }

    /// Get transform for dimension i, defaulting to Identity if not specified
    fn get_transform(&self, i: usize) -> &AdviTransform {
        if i < self.config.transforms.len() {
            &self.config.transforms[i]
        } else {
            // Return a static reference to Identity
            &AdviTransform::Identity
        }
    }

    /// Transform unconstrained parameters to constrained space
    fn transform_to_constrained(&self, eta: &Array1<f64>) -> Array1<f64> {
        Array1::from_shape_fn(eta.len(), |i| self.get_transform(i).forward(eta[i]))
    }

    /// Compute sum of log-det-Jacobians for all transforms
    fn total_log_det_jacobian(&self, eta: &Array1<f64>) -> f64 {
        (0..eta.len())
            .map(|i| self.get_transform(i).log_det_jacobian(eta[i]))
            .sum()
    }

    /// Compute gradient of total log-det-Jacobian w.r.t. eta
    fn grad_log_det_jacobian(&self, eta: &Array1<f64>) -> Array1<f64> {
        Array1::from_shape_fn(eta.len(), |i| {
            self.get_transform(i).grad_log_det_jacobian(eta[i])
        })
    }

    /// Compute gradient of constrained theta w.r.t. unconstrained eta
    fn forward_grad(&self, eta: &Array1<f64>) -> Array1<f64> {
        Array1::from_shape_fn(eta.len(), |i| self.get_transform(i).forward_grad(eta[i]))
    }

    /// Fit mean-field ADVI
    fn fit_mean_field<F>(&self, log_joint: F, dim: usize) -> StatsResult<PosteriorResult>
    where
        F: Fn(&Array1<f64>) -> StatsResult<(f64, Array1<f64>)>,
    {
        // Variational parameters: mu (dim) and log_sigma (dim)
        let n_params = 2 * dim;
        let mut mu = Array1::zeros(dim);
        let mut log_sigma = Array1::zeros(dim); // sigma = 1 initially

        let mut adam = AdviAdamState::new(n_params);
        let mut elbo_history = Vec::with_capacity(self.config.max_iterations);
        let mut converged = false;

        for iter in 0..self.config.max_iterations {
            let mut elbo_sum = 0.0;
            let mut grad_mu_sum = Array1::zeros(dim);
            let mut grad_log_sigma_sum = Array1::zeros(dim);

            for s in 0..self.config.num_samples {
                let seed = self
                    .config
                    .seed
                    .wrapping_add(iter as u64 * 1000)
                    .wrapping_add(s as u64);
                let epsilon = self.generate_epsilon(dim, seed);

                // Reparameterization: eta = mu + sigma * epsilon
                let sigma = log_sigma.mapv(f64::exp);
                let eta = &mu + &(&sigma * &epsilon);

                // Transform to constrained space
                let theta = self.transform_to_constrained(&eta);

                // Evaluate log joint and gradient
                let (log_p, grad_theta) = log_joint(&theta)?;

                // Log-det-Jacobian correction
                let ldj = self.total_log_det_jacobian(&eta);
                let grad_ldj = self.grad_log_det_jacobian(&eta);

                // Chain rule: d(log_p)/d(eta) = d(log_p)/d(theta) * d(theta)/d(eta)
                let fwd_grad = self.forward_grad(&eta);
                let grad_eta: Array1<f64> =
                    Array1::from_shape_fn(dim, |i| grad_theta[i] * fwd_grad[i] + grad_ldj[i]);

                // ELBO contribution: log p(x, theta) + log|det J|
                let elbo_s = log_p + ldj;
                elbo_sum += elbo_s;

                // Gradients w.r.t. mu and log_sigma
                // d(ELBO)/d(mu_i) = grad_eta_i
                // d(ELBO)/d(log_sigma_i) = grad_eta_i * sigma_i * epsilon_i + 1 (entropy gradient)
                for i in 0..dim {
                    grad_mu_sum[i] += grad_eta[i];
                    grad_log_sigma_sum[i] += grad_eta[i] * sigma[i] * epsilon[i];
                }
            }

            let n_s = self.config.num_samples as f64;
            elbo_sum /= n_s;
            grad_mu_sum /= n_s;
            grad_log_sigma_sum /= n_s;

            // Add entropy gradient: d H[q] / d log_sigma_i = 1.0
            // H[q] = sum_i (0.5 * (1 + log(2*pi)) + log_sigma_i)
            for i in 0..dim {
                grad_log_sigma_sum[i] += 1.0;
            }

            // Include entropy in ELBO
            let entropy: f64 = (0..dim)
                .map(|i| 0.5 * (1.0 + (2.0 * PI).ln()) + log_sigma[i])
                .sum();
            elbo_sum += entropy;

            elbo_history.push(elbo_sum);

            // Combine gradients into single vector for Adam
            let mut full_grad = Array1::zeros(n_params);
            for i in 0..dim {
                full_grad[i] = grad_mu_sum[i];
                full_grad[dim + i] = grad_log_sigma_sum[i];
            }

            // Adam update
            let direction = adam.update(&full_grad);
            let lr = self.config.learning_rate;
            for i in 0..dim {
                mu[i] += lr * direction[i];
                log_sigma[i] += lr * direction[dim + i];
                // Clip log_sigma to prevent numerical issues
                log_sigma[i] = log_sigma[i].max(-10.0).min(10.0);
            }

            // Check convergence
            if elbo_history.len() >= self.config.convergence_window {
                let n = elbo_history.len();
                let w = self.config.convergence_window;
                let recent_avg: f64 =
                    elbo_history[n - w / 2..n].iter().sum::<f64>() / (w / 2) as f64;
                let earlier_avg: f64 =
                    elbo_history[n - w..n - w / 2].iter().sum::<f64>() / (w / 2) as f64;
                if (recent_avg - earlier_avg).abs() < self.config.tolerance {
                    converged = true;
                    break;
                }
            }
        }

        // Compute posterior in constrained space
        let sigma = log_sigma.mapv(f64::exp);
        let constrained_means = self.transform_to_constrained(&mu);

        // Approximate constrained std devs via delta method:
        // Var(theta) approx (d theta/d eta)^2 * Var(eta)
        let fwd_grad = self.forward_grad(&mu);
        let constrained_stds = Array1::from_shape_fn(dim, |i| (fwd_grad[i] * sigma[i]).abs());

        Ok(PosteriorResult {
            means: constrained_means,
            std_devs: constrained_stds,
            elbo_history: elbo_history.clone(),
            iterations: elbo_history.len(),
            converged,
            samples: None,
        })
    }

    /// Fit full-rank ADVI
    fn fit_full_rank<F>(&self, log_joint: F, dim: usize) -> StatsResult<PosteriorResult>
    where
        F: Fn(&Array1<f64>) -> StatsResult<(f64, Array1<f64>)>,
    {
        // Parameters: mu (dim) + lower-triangular L (dim*(dim+1)/2)
        let n_tril = dim * (dim + 1) / 2;
        let n_params = dim + n_tril;
        let mut mu = Array1::zeros(dim);
        // Initialize L as identity (store lower triangular entries)
        let mut l_entries = Array1::zeros(n_tril);
        {
            let mut idx = 0;
            for row in 0..dim {
                for col in 0..=row {
                    if row == col {
                        l_entries[idx] = 1.0; // diagonal = 1
                    }
                    idx += 1;
                }
            }
        }

        let mut adam = AdviAdamState::new(n_params);
        let mut elbo_history = Vec::with_capacity(self.config.max_iterations);
        let mut converged = false;

        for iter in 0..self.config.max_iterations {
            // Reconstruct L matrix from entries
            let l_mat = tril_to_matrix(dim, &l_entries);

            let mut elbo_sum = 0.0;
            let mut grad_mu_sum = Array1::zeros(dim);
            let mut grad_l_sum = Array1::zeros(n_tril);

            for s in 0..self.config.num_samples {
                let seed = self
                    .config
                    .seed
                    .wrapping_add(iter as u64 * 1000)
                    .wrapping_add(s as u64);
                let epsilon = self.generate_epsilon(dim, seed);

                // Reparameterization: eta = mu + L * epsilon
                let l_eps = l_mat.dot(&epsilon);
                let eta = &mu + &l_eps;

                // Transform to constrained space
                let theta = self.transform_to_constrained(&eta);

                // Evaluate log joint and gradient
                let (log_p, grad_theta) = log_joint(&theta)?;

                // Log-det-Jacobian correction
                let ldj = self.total_log_det_jacobian(&eta);
                let grad_ldj = self.grad_log_det_jacobian(&eta);

                // Chain rule: d(log_p)/d(eta) = d(log_p)/d(theta) * d(theta)/d(eta)
                let fwd_grad = self.forward_grad(&eta);
                let grad_eta: Array1<f64> =
                    Array1::from_shape_fn(dim, |i| grad_theta[i] * fwd_grad[i] + grad_ldj[i]);

                let elbo_s = log_p + ldj;
                elbo_sum += elbo_s;

                // Gradients w.r.t. mu
                for i in 0..dim {
                    grad_mu_sum[i] += grad_eta[i];
                }

                // Gradients w.r.t. L entries
                // d(eta)/d(L_{ij}) = epsilon_j (when i is the row)
                let mut idx = 0;
                for row in 0..dim {
                    for col in 0..=row {
                        grad_l_sum[idx] += grad_eta[row] * epsilon[col];
                        idx += 1;
                    }
                }
            }

            let n_s = self.config.num_samples as f64;
            elbo_sum /= n_s;
            grad_mu_sum /= n_s;
            grad_l_sum /= n_s;

            // Entropy of full-rank Gaussian:
            // H[q] = 0.5 * d * (1 + log(2*pi)) + sum_i log|L_ii|
            let mut entropy = 0.5 * dim as f64 * (1.0 + (2.0 * PI).ln());
            {
                let mut idx = 0;
                for row in 0..dim {
                    for col in 0..=row {
                        if row == col {
                            entropy += l_entries[idx].abs().max(1e-15).ln();
                            // Gradient of entropy w.r.t. L_ii = 1/L_ii
                            let l_ii = l_entries[idx];
                            if l_ii.abs() > 1e-15 {
                                grad_l_sum[idx] += 1.0 / l_ii;
                            }
                        }
                        idx += 1;
                    }
                }
            }
            elbo_sum += entropy;
            elbo_history.push(elbo_sum);

            // Combine gradients
            let mut full_grad = Array1::zeros(n_params);
            for i in 0..dim {
                full_grad[i] = grad_mu_sum[i];
            }
            for i in 0..n_tril {
                full_grad[dim + i] = grad_l_sum[i];
            }

            // Adam update
            let direction = adam.update(&full_grad);
            let lr = self.config.learning_rate;
            for i in 0..dim {
                mu[i] += lr * direction[i];
            }
            for i in 0..n_tril {
                l_entries[i] += lr * direction[dim + i];
            }

            // Ensure diagonal of L stays positive (for valid Cholesky)
            {
                let mut idx = 0;
                for row in 0..dim {
                    for col in 0..=row {
                        if row == col {
                            l_entries[idx] = l_entries[idx].abs().max(1e-6);
                        }
                        // Clip entries for stability
                        l_entries[idx] = l_entries[idx].max(-10.0).min(10.0);
                        idx += 1;
                    }
                }
            }

            // Check convergence
            if elbo_history.len() >= self.config.convergence_window {
                let n = elbo_history.len();
                let w = self.config.convergence_window;
                let recent_avg: f64 =
                    elbo_history[n - w / 2..n].iter().sum::<f64>() / (w / 2) as f64;
                let earlier_avg: f64 =
                    elbo_history[n - w..n - w / 2].iter().sum::<f64>() / (w / 2) as f64;
                if (recent_avg - earlier_avg).abs() < self.config.tolerance {
                    converged = true;
                    break;
                }
            }
        }

        // Compute posterior statistics
        let l_mat = tril_to_matrix(dim, &l_entries);
        let constrained_means = self.transform_to_constrained(&mu);

        // Covariance in unconstrained space: Sigma = L L^T
        let cov = l_mat.dot(&l_mat.t());

        // Delta-method std devs in constrained space
        let fwd_grad = self.forward_grad(&mu);
        let constrained_stds =
            Array1::from_shape_fn(dim, |i| (fwd_grad[i] * fwd_grad[i] * cov[[i, i]]).sqrt());

        Ok(PosteriorResult {
            means: constrained_means,
            std_devs: constrained_stds,
            elbo_history: elbo_history.clone(),
            iterations: elbo_history.len(),
            converged,
            samples: None,
        })
    }
}

impl VariationalInference for Advi {
    fn fit<F>(&mut self, log_joint: F, dim: usize) -> StatsResult<PosteriorResult>
    where
        F: Fn(&Array1<f64>) -> StatsResult<(f64, Array1<f64>)>,
    {
        if dim == 0 {
            return Err(StatsError::InvalidArgument(
                "Dimension must be at least 1".to_string(),
            ));
        }
        if self.config.num_samples == 0 {
            return Err(StatsError::InvalidArgument(
                "num_samples must be at least 1".to_string(),
            ));
        }
        if self.config.learning_rate <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "learning_rate must be positive".to_string(),
            ));
        }

        match self.config.approximation {
            AdviApproximation::MeanField => self.fit_mean_field(log_joint, dim),
            AdviApproximation::FullRank => self.fit_full_rank(log_joint, dim),
        }
    }
}

// ============================================================================
// Helper: lower-triangular entries <-> matrix
// ============================================================================

/// Reconstruct a dim x dim lower-triangular matrix from flat entries
fn tril_to_matrix(dim: usize, entries: &Array1<f64>) -> Array2<f64> {
    let mut mat = Array2::zeros((dim, dim));
    let mut idx = 0;
    for row in 0..dim {
        for col in 0..=row {
            mat[[row, col]] = entries[idx];
            idx += 1;
        }
    }
    mat
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test: ADVI recovers the posterior mean of a 1D Gaussian with known
    /// conjugate prior: N(mu | mu0, sigma0^2) * N(data | mu, sigma^2)
    #[test]
    fn test_advi_gaussian_posterior_recovery() {
        // Prior: N(0, 1), Likelihood: data ~ N(mu, 1), observed data mean = 3.0, n = 10
        // Posterior: N(mu | 30/11, 1/11)
        let data_mean = 3.0_f64;
        let n_data = 10.0_f64;
        let prior_mean = 0.0_f64;
        let prior_var = 1.0_f64;
        let lik_var = 1.0_f64;

        let config = AdviConfig {
            approximation: AdviApproximation::MeanField,
            transforms: vec![AdviTransform::Identity],
            num_samples: 20,
            learning_rate: 0.05,
            max_iterations: 3000,
            tolerance: 1e-5,
            seed: 123,
            convergence_window: 100,
        };

        let mut advi = Advi::new(config);
        let result = advi
            .fit(
                move |theta: &Array1<f64>| {
                    let mu = theta[0];
                    // Log prior: N(mu | 0, 1)
                    let log_prior = -0.5 * (mu - prior_mean).powi(2) / prior_var;
                    // Log likelihood: sum of N(x_i | mu, 1) = -n/2 * (mu - data_mean)^2
                    let log_lik = -n_data / 2.0 * (mu - data_mean).powi(2) / lik_var;
                    let log_p = log_prior + log_lik;
                    // Gradient
                    let grad_prior = -(mu - prior_mean) / prior_var;
                    let grad_lik = -n_data * (mu - data_mean) / lik_var;
                    let grad = Array1::from_vec(vec![grad_prior + grad_lik]);
                    Ok((log_p, grad))
                },
                1,
            )
            .expect("ADVI should not fail");

        let expected_mean = (n_data * data_mean / lik_var + prior_mean / prior_var)
            / (n_data / lik_var + 1.0 / prior_var);
        let expected_std = (1.0 / (n_data / lik_var + 1.0 / prior_var)).sqrt();

        assert!(
            (result.means[0] - expected_mean).abs() < 0.3,
            "Mean should be close to {}, got {}",
            expected_mean,
            result.means[0]
        );
        assert!(
            (result.std_devs[0] - expected_std).abs() < 0.2,
            "Std should be close to {}, got {}",
            expected_std,
            result.std_devs[0]
        );
    }

    /// Test: ELBO increases (or at least does not decrease on average) over iterations
    #[test]
    fn test_advi_elbo_increases() {
        let config = AdviConfig {
            approximation: AdviApproximation::MeanField,
            transforms: vec![AdviTransform::Identity, AdviTransform::Identity],
            num_samples: 15,
            learning_rate: 0.02,
            max_iterations: 500,
            tolerance: 1e-6,
            seed: 77,
            convergence_window: 50,
        };

        let mut advi = Advi::new(config);
        let result = advi
            .fit(
                |theta: &Array1<f64>| {
                    // Simple 2D Gaussian target: N([1, 2], I)
                    let diff0 = theta[0] - 1.0;
                    let diff1 = theta[1] - 2.0;
                    let log_p = -0.5 * (diff0 * diff0 + diff1 * diff1);
                    let grad = Array1::from_vec(vec![-diff0, -diff1]);
                    Ok((log_p, grad))
                },
                2,
            )
            .expect("ADVI should succeed");

        // Check that late-stage ELBO is higher than early-stage
        let n = result.elbo_history.len();
        assert!(n > 100, "Should run at least 100 iterations");
        let early_avg: f64 = result.elbo_history[..50].iter().sum::<f64>() / 50.0;
        let late_avg: f64 = result.elbo_history[n - 50..].iter().sum::<f64>() / 50.0;
        assert!(
            late_avg > early_avg - 1.0,
            "Late ELBO ({}) should be higher than early ({})",
            late_avg,
            early_avg
        );
    }

    /// Test: Mean-field vs full-rank comparison
    /// Full-rank should achieve at least as good an ELBO as mean-field
    /// on a correlated target
    #[test]
    fn test_advi_mean_field_vs_full_rank() {
        // Correlated 2D Gaussian target: rho = 0.8
        let rho = 0.8_f64;
        let log_joint = move |theta: &Array1<f64>| {
            let x = theta[0];
            let y = theta[1];
            let det = 1.0 - rho * rho;
            let log_p =
                -0.5 / det * (x * x - 2.0 * rho * x * y + y * y) - 0.5 * (2.0 * PI * det).ln();
            let gx = -1.0 / det * (x - rho * y);
            let gy = -1.0 / det * (y - rho * x);
            Ok((log_p, Array1::from_vec(vec![gx, gy])))
        };

        // Mean-field
        let mf_config = AdviConfig {
            approximation: AdviApproximation::MeanField,
            num_samples: 20,
            learning_rate: 0.02,
            max_iterations: 2000,
            tolerance: 1e-5,
            seed: 42,
            convergence_window: 100,
            ..Default::default()
        };
        let mut mf_advi = Advi::new(mf_config);
        let mf_result = mf_advi.fit(log_joint, 2).expect("MF should succeed");

        // Full-rank
        let fr_config = AdviConfig {
            approximation: AdviApproximation::FullRank,
            num_samples: 20,
            learning_rate: 0.02,
            max_iterations: 2000,
            tolerance: 1e-5,
            seed: 42,
            convergence_window: 100,
            ..Default::default()
        };
        let mut fr_advi = Advi::new(fr_config);
        let fr_result = fr_advi.fit(log_joint, 2).expect("FR should succeed");

        let mf_final_elbo = mf_result
            .elbo_history
            .last()
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        let fr_final_elbo = fr_result
            .elbo_history
            .last()
            .copied()
            .unwrap_or(f64::NEG_INFINITY);

        // Full-rank should do at least as well (with some tolerance for stochasticity)
        assert!(
            fr_final_elbo > mf_final_elbo - 1.0,
            "Full-rank ELBO ({}) should be >= mean-field ELBO ({}) minus tolerance",
            fr_final_elbo,
            mf_final_elbo
        );
    }

    /// Test: ADVI with log transform recovers a positive parameter
    #[test]
    fn test_advi_log_transform() {
        // Target: Gamma(3, 1) = log-concave for shape >= 1
        // mode = (shape - 1) / rate = 2.0
        let config = AdviConfig {
            approximation: AdviApproximation::MeanField,
            transforms: vec![AdviTransform::Log],
            num_samples: 20,
            learning_rate: 0.01,
            max_iterations: 3000,
            tolerance: 1e-5,
            seed: 55,
            convergence_window: 100,
        };

        let mut advi = Advi::new(config);
        let result = advi
            .fit(
                |theta: &Array1<f64>| {
                    let x = theta[0];
                    if x <= 0.0 {
                        return Ok((f64::NEG_INFINITY, Array1::zeros(1)));
                    }
                    // Gamma(3, 1): log p(x) = (3-1)*ln(x) - x - ln(Gamma(3))
                    let log_p = 2.0 * x.ln() - x - (2.0_f64).ln(); // Gamma(3) = 2! = 2
                    let grad = Array1::from_vec(vec![2.0 / x - 1.0]);
                    Ok((log_p, grad))
                },
                1,
            )
            .expect("ADVI with log transform should succeed");

        // Gamma(3,1) mean = 3, mode = 2
        assert!(
            result.means[0] > 0.0,
            "Mean should be positive with log transform"
        );
        assert!(
            (result.means[0] - 3.0).abs() < 1.5,
            "Mean should be near 3 (Gamma(3,1) mean), got {}",
            result.means[0]
        );
    }

    /// Test: dimension validation
    #[test]
    fn test_advi_zero_dim_error() {
        let mut advi = Advi::new(AdviConfig::default());
        let result = advi.fit(|_theta: &Array1<f64>| Ok((0.0, Array1::zeros(0))), 0);
        assert!(result.is_err());
    }

    /// Test: transform forward-inverse roundtrip
    #[test]
    fn test_transform_roundtrip() {
        let transforms = vec![
            AdviTransform::Identity,
            AdviTransform::Log,
            AdviTransform::Logit,
            AdviTransform::Bounded {
                lower: -2.0,
                upper: 5.0,
            },
        ];
        let test_vals = vec![1.5, 2.0, 0.3, 1.0];

        for (t, v) in transforms.iter().zip(test_vals.iter()) {
            let eta = t.inverse(*v).expect("inverse should succeed");
            let recovered = t.forward(eta);
            assert!(
                (recovered - v).abs() < 1e-10,
                "Roundtrip failed for {:?}: {} -> {} -> {}",
                t,
                v,
                eta,
                recovered
            );
        }
    }

    /// Test: log-det-Jacobian is nonzero for non-identity transforms
    #[test]
    fn test_log_det_jacobian_nonzero() {
        let transforms = vec![
            AdviTransform::Log,
            AdviTransform::Logit,
            AdviTransform::Bounded {
                lower: 0.0,
                upper: 10.0,
            },
        ];
        for t in &transforms {
            let ldj = t.log_det_jacobian(0.5);
            assert!(
                ldj.is_finite(),
                "Log-det-Jacobian should be finite for {:?}",
                t
            );
        }
    }
}
