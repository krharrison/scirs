//! Sparse grid interpolation and quadrature methods.
//!
//! - `core`: hierarchical hat-function sparse grid (existing)
//! - `smolyak`: Smolyak construction with Clenshaw-Curtis, Gauss-Legendre, and Gauss-Patterson rules

pub mod core;
pub mod smolyak;

pub use core::*;
pub use smolyak::{
    smolyak_grid, smolyak_interpolant, smolyak_quadrature, SmolyakConfig, SmolyakGrid, SmolyakRule,
};
