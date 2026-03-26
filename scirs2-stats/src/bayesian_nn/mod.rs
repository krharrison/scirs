//! Bayesian Neural Network approximations.
//!
//! This module provides post-hoc Bayesian approximations for neural network
//! weights, enabling uncertainty quantification without modifying the training
//! procedure. Two families of methods are implemented:
//!
//! - **Laplace approximation**: Fits a Gaussian posterior at the MAP estimate
//!   using the curvature (Hessian) of the loss surface. Supports full, diagonal,
//!   and Kronecker-factored (KFAC) covariance structures.
//!
//! - **SWAG** (Stochastic Weight Averaging Gaussian): Collects weight snapshots
//!   during SGD training and fits a Gaussian with diagonal + low-rank covariance.
//!
//! Both methods support Monte Carlo predictive inference, calibration diagnostics
//! (ECE, reliability diagrams), and uncertainty decomposition.

pub mod laplace;
pub mod prediction;
pub mod swag;
pub mod types;

// Variational BNN (Wave 25, WS148)
pub mod inference;
pub mod layers;

// Re-exports (Laplace / SWAG posterior approximation)
pub use laplace::{kfac_factors, LaplaceApproximation};
pub use prediction::{
    decompose_uncertainty, expected_calibration_error, gaussian_nll, mc_predictive,
    reliability_diagram,
};
pub use swag::{multi_swag_predict, SWAGCollector};
pub use types::{
    ApproximationMethod, BNNConfig, BNNPosterior, CovarianceType, PredictiveDistribution,
    ReliabilityBin, UncertaintyType,
};

// Re-exports (variational BNN)
pub use inference::{aleatoric_uncertainty, epistemic_uncertainty, BayesianMlp};
pub use layers::{BayesianLinear, BnnConfig};
