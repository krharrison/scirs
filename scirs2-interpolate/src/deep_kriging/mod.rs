//! Deep Kriging and Gaussian Process Surrogate
//!
//! This module provides two advanced spatial interpolation and surrogate
//! modelling techniques:
//!
//! ## Neural Basis Kriging (Deep Kriging)
//!
//! An MLP learns nonlinear basis functions φ(x) that map raw coordinates
//! into a latent feature space where ordinary kriging is performed.  This
//! allows the model to capture non-stationary and nonlinear spatial patterns
//! that conventional kriging with a fixed variogram cannot represent.
//!
//! ## Gaussian Process Surrogate
//!
//! Full Gaussian process regression with:
//! - Cholesky-based inference
//! - Predictive mean and variance
//! - Marginal log-likelihood for model selection
//! - Hyperparameter optimisation
//! - Acquisition functions (EI, PI, UCB, LCB) for Bayesian optimisation
//!
//! ## Example
//!
//! ```rust,ignore
//! use scirs2_interpolate::deep_kriging::{
//!     GaussianProcessSurrogate, GPSurrogateConfig, KernelType,
//!     AcquisitionFunction,
//! };
//!
//! let train_x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
//! let train_y = vec![0.0, 0.84, 0.91, 0.14];
//! let config = GPSurrogateConfig {
//!     kernel: KernelType::SquaredExponential { lengthscale: 1.0, variance: 1.0 },
//!     noise: 1e-6,
//!     ..GPSurrogateConfig::default()
//! };
//!
//! let gp = GaussianProcessSurrogate::fit(train_x, train_y, config)
//!     .expect("doc example: fit should succeed");
//! let (mean, var) = gp.predict(&[1.5]).expect("doc example: predict should succeed");
//! ```

pub mod gp_surrogate;
pub mod neural_basis;
pub mod types;

// Re-exports
pub use gp_surrogate::GaussianProcessSurrogate;
pub use neural_basis::NeuralBasisKriging;
pub use types::{
    AcquisitionFunction, Activation, DeepKrigingConfig, GPSurrogateConfig, KernelType,
    SurrogateResult,
};
