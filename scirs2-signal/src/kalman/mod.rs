//! Kalman filter variants for state estimation and tracking.
//!
//! This module provides a comprehensive suite of Kalman filter algorithms,
//! covering linear and nonlinear systems, as well as alternative representations:
//!
//! | Filter | System Type | Key Property |
//! |--------|-------------|--------------|
//! | [`KalmanFilter`] | Linear | Optimal for linear Gaussian |
//! | [`ExtendedKalmanFilter`] | Nonlinear (smooth) | Linearises via Jacobian |
//! | [`UnscentedKalmanFilter`] | Nonlinear (general) | Sigma-point propagation |
//! | [`EnsembleKalmanFilter`] | High-dimensional | Monte Carlo ensemble |
//! | [`InformationFilter`] | Linear | Dual/information form |
//! | [`particle::ParticleFilter`] | Nonlinear / Non-Gaussian | Sequential Monte Carlo |
//!
//! # Quick Start
//!
//! ```
//! use scirs2_signal::kalman::KalmanFilter;
//!
//! // Track a 1D position with a constant-velocity model
//! let mut kf = KalmanFilter::new(2, 1);
//! kf.set_F(vec![vec![1.0, 1.0], vec![0.0, 1.0]]).expect("operation should succeed");
//! kf.set_H(vec![vec![1.0, 0.0]]).expect("operation should succeed");
//! kf.set_Q(vec![vec![0.01, 0.0], vec![0.0, 0.01]]).expect("operation should succeed");
//! kf.set_R(vec![vec![0.1]]).expect("operation should succeed");
//! kf.set_initial_state(&[0.0, 1.0]).expect("operation should succeed");
//! kf.predict().expect("operation should succeed");
//! kf.update(&[1.05]).expect("operation should succeed");
//! ```

pub mod ensemble;
pub mod extended;
pub mod information;
pub mod legacy;
pub mod matrix_utils;
pub mod particle;
pub mod standard;
pub mod unscented;

pub use ensemble::EnsembleKalmanFilter;
pub use extended::{numerical_jacobian, ExtendedKalmanFilter};
pub use information::InformationFilter;
pub use particle::{
    effective_sample_size, GaussianLikelihood, Likelihood, ParticleFilter, ResamplingStrategy,
    StudentTLikelihood,
};
pub use standard::KalmanFilter;
pub use unscented::UnscentedKalmanFilter;

// Re-export legacy API for backwards compatibility
pub use legacy::{
    adaptive_kalman_filter, kalman_filter, robust_kalman_filter, unscented_kalman_filter,
    KalmanConfig, UkfConfig,
};
