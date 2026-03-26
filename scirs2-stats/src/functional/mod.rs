//! Functional data analysis and functional regression.
//!
//! This module provides tools for analyzing data where each observation is a
//! function (curve) rather than a finite-dimensional vector. Key capabilities:
//!
//! - **Basis function expansion**: B-spline, Fourier, and polynomial bases
//! - **Curve smoothing**: Penalized least squares with GCV-based parameter selection
//! - **Functional PCA (fPCA)**: Extract principal modes of variation across curves
//! - **Scalar-on-function regression**: Predict a scalar from a functional predictor
//! - **Function-on-function regression**: Predict a curve from a functional predictor
//!
//! # Example
//!
//! ```rust
//! use scirs2_stats::functional::{
//!     FunctionalData, FunctionalConfig, BasisType,
//!     functional_pca, ScalarOnFunctionRegression,
//! };
//!
//! // Create some functional data (5 curves on a grid of 50 points)
//! let grid: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
//! let observations: Vec<Vec<f64>> = (0..5)
//!     .map(|i| {
//!         grid.iter()
//!             .map(|&t| (i as f64 * 0.5).sin() * t + (i as f64 * 0.3).cos() * t * t)
//!             .collect()
//!     })
//!     .collect();
//!
//! let data = FunctionalData::new(grid, observations).expect("valid data");
//! let config = FunctionalConfig::default();
//!
//! // Functional PCA
//! let fpca_result = functional_pca(&data, &config).expect("fPCA succeeds");
//! ```

pub mod basis;
pub mod fpca;
pub mod regression;
pub mod types;

// Re-export main types and functions
pub use basis::{evaluate_basis, gcv_select_lambda, smooth_curve};
pub use fpca::{functional_pca, reconstruct_from_scores};
pub use regression::{r_squared, FunctionOnFunctionRegression, ScalarOnFunctionRegression};
pub use types::{BasisType, FPCAResult, FoFResult, FunctionalConfig, FunctionalData, SoFResult};
