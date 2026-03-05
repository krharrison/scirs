//! Extended orthogonal polynomial functionality
//!
//! This module provides additional orthogonal polynomial features beyond the core
//! evaluators in `orthogonal.rs`:
//!
//! - **Derivatives**: Derivative computation for all polynomial families
//! - **Quadrature**: Gauss quadrature nodes and weights (Gauss-Legendre, Gauss-Hermite,
//!   Gauss-Laguerre, Gauss-Chebyshev, Gauss-Jacobi)
//! - **SciPy-compatible wrappers**: `eval_legendre`, `eval_chebyt`, etc.
//! - **Zernike polynomials**: Used in optics for wavefront aberration analysis

pub mod derivatives;
pub mod quadrature;
pub mod zernike;

// Re-export for convenience
pub use derivatives::*;
pub use quadrature::*;
pub use zernike::*;
