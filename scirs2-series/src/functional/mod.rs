//! Functional Data Analysis (FDA) for time series
//!
//! This module provides a comprehensive suite of methods for functional data analysis,
//! treating time series observations as realizations of smooth random functions.
//!
//! # Sub-modules
//!
//! - [`basis`]: Basis function systems (B-splines, Fourier, Wavelets, Monomials)
//! - [`smoothing`]: Functional data smoothing (penalized LS, kernel, spline)
//! - [`fpca`]: Functional Principal Component Analysis and variants
//! - [`regression`]: Functional regression models (FLM, FoS, Concurrent, ANOVA)
//!
//! # Key Concepts
//!
//! In FDA, each observation is treated as a function x_i: [a,b] → ℝ rather than a
//! finite-dimensional vector. The workflow typically consists of:
//!
//! 1. **Basis expansion**: Represent each curve as c_i^T Φ(t) using a basis system Φ
//! 2. **Smoothing**: Estimate c_i from noisy discrete observations via penalized LS
//! 3. **Analysis**: Apply FPCA, regression, or ANOVA to the smooth curves
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use scirs2_series::functional::basis::BSplineBasis;
//! use scirs2_series::functional::smoothing::SplineSmoother;
//! use scirs2_series::functional::fpca::{FPCA, FPCAConfig};
//! use scirs2_core::ndarray::Array1;
//!
//! // Create observation data
//! let t = Array1::from_vec((0..50).map(|i| i as f64 / 49.0).collect());
//! let y = t.mapv(|x| x.sin() + 0.1 * x.cos());
//!
//! // Smooth with cubic spline
//! let smoother = SplineSmoother::default();
//! let fd = smoother.fit(&t, &y).expect("smoothing failed");
//!
//! // Evaluate at any point
//! let val = fd.eval(0.5).expect("evaluation failed");
//!
//! // Perform FPCA on a collection of curves
//! let t_list = vec![t.clone()];
//! let y_list = vec![y.clone()];
//! // (in practice, use many observations)
//! ```

pub mod basis;
pub mod fpca;
pub mod regression;
pub mod smoothing;

// Re-export key types for convenient access

pub use basis::{
    evaluate_basis_matrix, evaluate_deriv_matrix, BSplineBasis, BasisSystem, FourierBasis,
    MonomialBasis, WaveletBasis, WaveletType,
};

pub use smoothing::{FunctionalData, GCV, KernelSmoother, KernelType, LocalPolynomialOrder,
    PenalizedLeastSquares, SplineSmoother};

pub use fpca::{
    BivariateFPCA, BivariateFPCAResult, FPCAConfig, FPCAResult, MultilevelFPCA,
    MultilevelFPCAResult, VarianceExplained, FPCA,
};

pub use regression::{
    ConcurrentModel, ConcurrentResult, FLMResult, FoSResult, FunctionalANOVA,
    FunctionalANOVAResult, FunctionalLinearModel, FunctionOnScalarRegression,
};
