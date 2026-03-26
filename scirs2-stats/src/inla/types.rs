//! Types for Integrated Nested Laplace Approximation (INLA)
//!
//! This module defines the core data structures for latent Gaussian models,
//! INLA configuration, and result types.

use scirs2_core::ndarray::{Array1, Array2};

/// Likelihood family for the observed data given the latent field
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum LikelihoodFamily {
    /// Gaussian likelihood: y_i ~ N(eta_i, sigma^2)
    Gaussian,
    /// Poisson likelihood: y_i ~ Poisson(exp(eta_i))
    Poisson,
    /// Binomial likelihood: y_i ~ Binomial(n_i, logistic(eta_i))
    Binomial,
    /// Negative Binomial likelihood: y_i ~ NegBin(r, p_i) with log link
    NegativeBinomial,
}

/// A latent Gaussian model specification
///
/// The model is:
///   y | x, theta ~ product of p(y_i | eta_i, theta)
///   x | theta ~ N(0, Q(theta)^{-1})
///   eta = A * x  (linear predictor, where A is the design matrix)
#[derive(Debug, Clone)]
pub struct LatentGaussianModel {
    /// Observed data vector (n x 1)
    pub y: Array1<f64>,
    /// Fixed effects design matrix (n x p), maps latent field to linear predictor
    pub design_matrix: Array2<f64>,
    /// GMRF precision matrix Q(theta) for the latent field (p x p)
    pub precision_matrix: Array2<f64>,
    /// Likelihood family for the observations
    pub likelihood: LikelihoodFamily,
    /// Number of trials for Binomial likelihood (one per observation).
    /// Ignored for other likelihood families.
    pub n_trials: Option<Array1<f64>>,
    /// Observation precision (1/sigma^2) for Gaussian likelihood.
    /// Ignored for other likelihood families.
    pub observation_precision: Option<f64>,
}

impl LatentGaussianModel {
    /// Create a new latent Gaussian model
    ///
    /// # Arguments
    /// * `y` - Observation vector
    /// * `design_matrix` - Design matrix mapping latent field to linear predictor
    /// * `precision_matrix` - GMRF precision matrix Q(theta)
    /// * `likelihood` - Likelihood family
    pub fn new(
        y: Array1<f64>,
        design_matrix: Array2<f64>,
        precision_matrix: Array2<f64>,
        likelihood: LikelihoodFamily,
    ) -> Self {
        Self {
            y,
            design_matrix,
            precision_matrix,
            likelihood,
            n_trials: None,
            observation_precision: None,
        }
    }

    /// Set the number of trials for Binomial likelihood
    pub fn with_n_trials(mut self, n_trials: Array1<f64>) -> Self {
        self.n_trials = Some(n_trials);
        self
    }

    /// Set the observation precision for Gaussian likelihood
    pub fn with_observation_precision(mut self, precision: f64) -> Self {
        self.observation_precision = Some(precision);
        self
    }
}

/// Integration strategy for marginal computation
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum IntegrationStrategy {
    /// Full grid-based integration over hyperparameter space
    Grid,
    /// Central Composite Design for efficient integration
    CCD,
    /// Simplified Laplace approximation (fastest but least accurate)
    SimplifiedLaplace,
}

/// Configuration for the INLA algorithm
#[derive(Debug, Clone)]
pub struct INLAConfig {
    /// Number of grid points per hyperparameter dimension
    pub n_hyperparameter_grid: usize,
    /// Integration strategy for marginalizing over hyperparameters
    pub integration_strategy: IntegrationStrategy,
    /// Maximum Newton-Raphson iterations for mode finding
    pub max_newton_iter: usize,
    /// Convergence tolerance for Newton-Raphson
    pub newton_tol: f64,
    /// Whether to use simplified Laplace for conditional marginals
    pub simplified_laplace: bool,
    /// Step size damping factor for Newton-Raphson (0 < damping <= 1)
    pub newton_damping: f64,
    /// Hyperparameter prior log-density (if None, flat prior is used)
    pub hyperparameter_range: Option<(f64, f64)>,
}

impl Default for INLAConfig {
    fn default() -> Self {
        Self {
            n_hyperparameter_grid: 25,
            integration_strategy: IntegrationStrategy::Grid,
            max_newton_iter: 100,
            newton_tol: 1e-8,
            simplified_laplace: false,
            newton_damping: 1.0,
            hyperparameter_range: None,
        }
    }
}

/// Posterior distribution of a single hyperparameter
#[derive(Debug, Clone)]
pub struct HyperparameterPosterior {
    /// Grid points at which the posterior was evaluated
    pub grid_points: Vec<f64>,
    /// Log-density values at each grid point (unnormalized)
    pub log_densities: Vec<f64>,
    /// Posterior mean (computed from normalized density)
    pub mean: f64,
    /// Posterior variance (computed from normalized density)
    pub variance: f64,
}

/// Result of the INLA algorithm
#[derive(Debug, Clone)]
pub struct INLAResult {
    /// Posterior marginal means for each latent field component
    pub marginal_means: Array1<f64>,
    /// Posterior marginal variances for each latent field component
    pub marginal_variances: Array1<f64>,
    /// Posterior distributions of hyperparameters
    pub hyperparameter_posteriors: Vec<HyperparameterPosterior>,
    /// Log marginal likelihood estimate log p(y)
    pub log_marginal_likelihood: f64,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Number of Newton-Raphson iterations used at the mode
    pub newton_iterations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_default_config() {
        let config = INLAConfig::default();
        assert_eq!(config.n_hyperparameter_grid, 25);
        assert_eq!(config.max_newton_iter, 100);
        assert!((config.newton_tol - 1e-8).abs() < 1e-15);
        assert!(!config.simplified_laplace);
        assert_eq!(config.integration_strategy, IntegrationStrategy::Grid);
    }

    #[test]
    fn test_latent_gaussian_model_new() {
        let y = array![1.0, 2.0, 3.0];
        let design = Array2::eye(3);
        let precision = Array2::eye(3);
        let model =
            LatentGaussianModel::new(y.clone(), design, precision, LikelihoodFamily::Gaussian);
        assert_eq!(model.y, y);
        assert_eq!(model.likelihood, LikelihoodFamily::Gaussian);
        assert!(model.n_trials.is_none());
        assert!(model.observation_precision.is_none());
    }

    #[test]
    fn test_model_with_builders() {
        let y = array![1.0, 0.0, 1.0];
        let design = Array2::eye(3);
        let precision = Array2::eye(3);
        let n_trials = array![10.0, 10.0, 10.0];
        let model = LatentGaussianModel::new(y, design, precision, LikelihoodFamily::Binomial)
            .with_n_trials(n_trials.clone())
            .with_observation_precision(1.0);

        assert_eq!(model.likelihood, LikelihoodFamily::Binomial);
        assert!(model.n_trials.is_some());
        assert_eq!(model.n_trials.as_ref().map(|t| t.len()), Some(3));
        assert!((model.observation_precision.unwrap_or(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_likelihood_variants() {
        let variants = [
            LikelihoodFamily::Gaussian,
            LikelihoodFamily::Poisson,
            LikelihoodFamily::Binomial,
            LikelihoodFamily::NegativeBinomial,
        ];
        for v in &variants {
            // Ensure Debug works
            let _ = format!("{:?}", v);
        }
    }

    #[test]
    fn test_integration_strategy_variants() {
        let variants = [
            IntegrationStrategy::Grid,
            IntegrationStrategy::CCD,
            IntegrationStrategy::SimplifiedLaplace,
        ];
        for v in &variants {
            let _ = format!("{:?}", v);
        }
    }
}
