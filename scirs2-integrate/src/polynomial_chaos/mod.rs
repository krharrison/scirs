//! Polynomial Chaos Expansion (PCE) for Uncertainty Quantification
//!
//! Implements generalized Polynomial Chaos (gPC) following the Wiener-Askey scheme:
//! - Hermite polynomials <-> Gaussian distributions
//! - Legendre polynomials <-> Uniform distributions
//! - Laguerre polynomials <-> Exponential distributions
//! - Jacobi polynomials <-> Beta distributions
//!
//! ## Features
//! - PCE coefficient computation via projection (quadrature) and regression (least-squares)
//! - Multi-dimensional PCE with total-degree and hyperbolic truncation
//! - Sobol sensitivity indices directly from PCE coefficients
//! - Stochastic Galerkin method for ODEs with random parameters
//!
//! ## References
//! - Xiu & Karniadakis (2002). "The Wiener-Askey Polynomial Chaos for Stochastic DEs"
//! - Sudret (2008). "Global sensitivity analysis using polynomial chaos expansions"

pub mod basis;
pub mod expansion;
pub mod galerkin;
pub mod statistics;
pub mod types;

pub use basis::{evaluate_basis_1d, evaluate_basis_nd, generate_multi_indices};
pub use expansion::PolynomialChaosExpansion;
pub use galerkin::{StochasticGalerkinConfig, StochasticGalerkinSolver};
pub use statistics::{pce_mean, pce_variance, sobol_indices, total_sobol_indices};
pub use types::*;
