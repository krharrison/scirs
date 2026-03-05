//! Decomposition methods for large-scale optimization
//!
//! This module provides classical decomposition approaches that exploit
//! problem structure to break large optimization problems into manageable
//! subproblems.
//!
//! # Methods
//!
//! - [`BendersDecomposition`]: Decomposes mixed problems into master + subproblems
//! - [`DantzigWolfe`]: Column generation for structured LP/NLP
//! - [`Admm`]: Alternating direction method of multipliers for consensus problems
//! - [`ProximalBundle`]: Bundle method for nonsmooth convex optimization
//!
//! # Example
//!
//! ```no_run
//! use scirs2_optimize::decomposition::{Admm, AdmmOptions};
//!
//! // Solve: min x^2 + ||z-1||^2 s.t. x = z
//! let result = Admm::default().solve(
//!     |x: &[f64]| x[0].powi(2),
//!     |v: &[f64], rho: f64| {
//!         vec![(1.0 + rho * 0.5 * v[0]) / (1.0 + rho * 0.5)]
//!     },
//!     &[1.0],
//! ).expect("valid input");
//! println!("x* = {:?}, f = {}", result.x, result.fun);
//! ```

pub mod benders;

pub use benders::{
    Admm, AdmmOptions, AdmmResult, BendersDecomposition, BendersOptions, BendersResult,
    DantzigWolfe, DantzigWolfeOptions, DantzigWolfeResult, ProximalBundle, ProximalBundleOptions,
    ProximalBundleResult,
};
