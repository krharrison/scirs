//! Ensemble ODE integration: batched parallel solving of many IVPs.
//!
//! This module provides tools for integrating a large ensemble of ODE
//! initial-value problems in parallel, each with potentially different
//! initial conditions or parameter sets.
//!
//! ## Algorithm
//!
//! Each ensemble member is solved independently using the Dormand-Prince
//! RK45 method (with FSAL — First Same As Last — optimisation).
//! Members are distributed across threads via `std::thread::scope`.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use scirs2_integrate::ode::ensemble::{EnsembleConfig, OdeEnsembleSolver};
//!
//! let cfg = EnsembleConfig { n_ensemble: 10, ..Default::default() };
//! let solver = OdeEnsembleSolver::new(cfg);
//! // f(t, y, &param) = -param * y  (exponential decay)
//! let params: Vec<f64> = (0..10).map(|i| 1.0 + i as f64 * 0.1).collect();
//! let y0s: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0]).collect();
//! let result = solver.solve(|t, y, &p| vec![-p * y[0]], &params, &y0s, &cfg).unwrap();
//! ```

pub mod solver;
pub mod types;

pub use solver::OdeEnsembleSolver;
pub use types::{EnsembleConfig, EnsembleResult};
