//! Stefan (free-boundary / phase-change) problem solvers.
//!
//! This module implements numerical methods for the one-phase Stefan problem,
//! which models the melting or solidification of a material with a moving
//! phase-change interface.
//!
//! ## Problem Statement
//!
//! The one-phase Stefan problem in 1D:
//!
//! ```text
//! ∂u/∂t = α ∂²u/∂x²  for  0 < x < s(t)
//! u(0,t) = T_wall
//! u(s(t),t) = T_m
//! ds/dt = -(α / St) ∂u/∂x|_{x=s(t)⁻}   (Stefan condition)
//! s(0) = 0
//! ```
//!
//! The analytical solution is `s(t) = 2λ√(αt)` where `λ` satisfies
//! `λ exp(λ²) erf(λ) = St / √π`.

pub mod solver;
pub mod types;

pub use solver::StefanSolver;
pub use types::{StefanConfig, StefanResult};
