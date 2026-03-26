//! Type definitions for Bayesian Neural Network approximations.
//!
//! Provides core data structures for Laplace approximation and SWAG
//! posterior inference over neural network weights.

use scirs2_core::ndarray::{Array1, Array2};

/// Type of uncertainty to quantify.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UncertaintyType {
    /// Irreducible noise in the data
    Aleatoric,
    /// Model uncertainty due to limited data
    Epistemic,
    /// Total uncertainty (aleatoric + epistemic)
    Total,
}

/// Posterior approximation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ApproximationMethod {
    /// Laplace approximation on last layer weights only
    LastLayerLaplace,
    /// Laplace approximation on all network weights
    FullLaplace,
    /// Full-rank SWAG (diagonal + low-rank)
    SWAG,
    /// Diagonal-only SWAG
    SWAGDiag,
}

/// Configuration for Bayesian neural network approximations.
#[derive(Debug, Clone)]
pub struct BNNConfig {
    /// Approximation method to use
    pub method: ApproximationMethod,
    /// Number of Monte Carlo samples for predictive distribution (default 30)
    pub n_samples: usize,
    /// Prior precision: weights ~ N(0, (1/prior_precision) * I) (default 1.0)
    pub prior_precision: f64,
    /// Number of low-rank deviation columns for SWAG (default 20)
    pub swag_rank: usize,
    /// Start collecting weight snapshots after this many SGD steps (default 0)
    pub swag_collection_start: usize,
    /// Collect a weight snapshot every N SGD steps (default 1)
    pub swag_collection_freq: usize,
}

impl Default for BNNConfig {
    fn default() -> Self {
        Self {
            method: ApproximationMethod::FullLaplace,
            n_samples: 30,
            prior_precision: 1.0,
            swag_rank: 20,
            swag_collection_start: 0,
            swag_collection_freq: 1,
        }
    }
}

/// Structure of the posterior covariance matrix.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum CovarianceType {
    /// Full dense covariance matrix
    Full(Array2<f64>),
    /// Diagonal covariance (variance vector)
    Diagonal(Array1<f64>),
    /// Low-rank plus diagonal: Sigma = diag(d_diag) + deviation * deviation^T / (K-1)
    LowRankPlusDiagonal {
        /// Diagonal component
        d_diag: Array1<f64>,
        /// Low-rank deviation matrix, columns are (theta_i - theta_bar)
        deviation: Array2<f64>,
    },
    /// Kronecker-factored covariance: Sigma approx A kron B
    KroneckerFactored {
        /// Activation covariance factor
        a_factor: Array2<f64>,
        /// Gradient covariance factor
        b_factor: Array2<f64>,
    },
}

/// Posterior distribution over neural network weights.
#[derive(Debug, Clone)]
pub struct BNNPosterior {
    /// Posterior mean (MAP estimate)
    pub mean: Array1<f64>,
    /// Covariance structure
    pub covariance_type: CovarianceType,
    /// Log marginal likelihood estimate
    pub log_marginal_likelihood: f64,
}

/// Predictive distribution at test points.
#[derive(Debug, Clone)]
pub struct PredictiveDistribution {
    /// Predictive mean
    pub mean: Array1<f64>,
    /// Predictive variance
    pub variance: Array1<f64>,
    /// Optional matrix of prediction samples, shape \[n_samples x n_outputs\]
    pub samples: Option<Array2<f64>>,
}

/// A single bin in a reliability (calibration) diagram.
#[derive(Debug, Clone)]
pub struct ReliabilityBin {
    /// Mean predicted probability in this bin
    pub mean_predicted: f64,
    /// Mean observed frequency (fraction of positives) in this bin
    pub mean_observed: f64,
    /// Number of predictions that fell in this bin
    pub count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = BNNConfig::default();
        assert_eq!(cfg.n_samples, 30);
        assert!((cfg.prior_precision - 1.0).abs() < 1e-12);
        assert_eq!(cfg.swag_rank, 20);
        assert_eq!(cfg.method, ApproximationMethod::FullLaplace);
    }

    #[test]
    fn test_uncertainty_type_variants() {
        let u = UncertaintyType::Total;
        assert_eq!(u, UncertaintyType::Total);
        assert_ne!(UncertaintyType::Aleatoric, UncertaintyType::Epistemic);
    }

    #[test]
    fn test_covariance_type_diagonal() {
        let diag = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let cov = CovarianceType::Diagonal(diag.clone());
        match &cov {
            CovarianceType::Diagonal(d) => assert_eq!(d.len(), 3),
            _ => panic!("Expected Diagonal variant"),
        }
    }

    #[test]
    fn test_predictive_distribution_creation() {
        let pd = PredictiveDistribution {
            mean: Array1::from_vec(vec![1.0, 2.0]),
            variance: Array1::from_vec(vec![0.1, 0.2]),
            samples: None,
        };
        assert_eq!(pd.mean.len(), 2);
        assert!(pd.samples.is_none());
    }

    #[test]
    fn test_reliability_bin() {
        let bin = ReliabilityBin {
            mean_predicted: 0.5,
            mean_observed: 0.48,
            count: 100,
        };
        assert_eq!(bin.count, 100);
        assert!((bin.mean_predicted - 0.5).abs() < 1e-12);
    }
}
