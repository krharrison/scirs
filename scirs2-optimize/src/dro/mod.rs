//! Distributionally Robust Optimization (DRO).
//!
//! This module provides optimization algorithms that are robust against
//! distributional uncertainty.  Rather than assuming that the true data
//! distribution is exactly the empirical distribution P_N, DRO seeks a
//! decision that minimises the worst-case expected loss over an
//! *ambiguity set* of plausible distributions.
//!
//! # Sub-modules
//!
//! - [`types`]: shared configuration and result types
//! - [`wasserstein_dro`]: Wasserstein-ball DRO (Esfahani & Kuhn 2018)
//! - [`cvar_dro`]: CVaR-based DRO (Rockafellar & Uryasev 2000)
//!
//! # Quick start
//!
//! ```rust
//! use scirs2_optimize::dro::{portfolio_dro, solve_cvar_dro, DroConfig};
//!
//! // Synthetic return data (3 assets, 20 observations).
//! let returns: Vec<Vec<f64>> = (0..20)
//!     .map(|i| vec![0.01 * i as f64, 0.02, -0.005 * i as f64])
//!     .collect();
//!
//! // Wasserstein DRO portfolio (ε = 0.05).
//! let result = portfolio_dro(&returns, 0.05, None).expect("dro ok");
//! println!("Robust weights: {:?}", result.optimal_weights);
//! println!("Worst-case loss: {:.4}", result.worst_case_loss);
//!
//! // CVaR-DRO with α = 0.9.
//! let samples: Vec<Vec<f64>> = (0..20).map(|i| vec![0.01 * i as f64, 0.02]).collect();
//! let cvar_result = solve_cvar_dro(2, &samples, 0.9, 0.1, None).expect("cvar dro ok");
//! println!("CVaR-DRO weights: {:?}", cvar_result.optimal_weights);
//! ```
//!
//! # References
//!
//! - Esfahani, P. M. & Kuhn, D. (2018). "Data-driven distributionally robust
//!   optimization using the Wasserstein metric." *Mathematical Programming*.
//! - Rockafellar, R. T. & Uryasev, S. (2000). "Optimization of conditional
//!   value-at-risk." *Journal of Risk*.

pub mod cvar_dro;
#[cfg(test)]
mod tests;
pub mod types;
pub mod wasserstein_dro;

// Re-exports
pub use cvar_dro::{solve_cvar_dro, CvarDro, CvarEstimator};
pub use types::{DroConfig, DroResult, DroSolver, RobustObjective, WassersteinBall};
pub use wasserstein_dro::{portfolio_dro, portfolio_erm, WassersteinDro};
