//! Uncertainty Quantification methods for numerical integration.
//!
//! This module provides tools for propagating uncertainty through computational models:
//!
//! - **Polynomial Chaos Expansion (PCE)**: Non-intrusive spectral projection for
//!   quantifying how input uncertainty propagates to model outputs. Supports:
//!   - Hermite polynomials (Gaussian inputs)
//!   - Legendre polynomials (uniform inputs)
//!   - Laguerre polynomials (exponential inputs)
//!   - Sobol' sensitivity indices from PCE coefficients
//!   - Smolyak sparse grid for high-dimensional problems

pub mod pce;

pub use pce::{PceConfig, PceResult, PolynomialChaos, PolynomialFamily};
