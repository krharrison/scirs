//! Multi-fidelity optimization: Hyperband and Successive Halving.
//!
//! This module provides resource-efficient hyperparameter optimization via
//! early-stopping strategies.
//!
//! - [`SuccessiveHalving`] allocates a fixed budget across configurations and
//!   iteratively discards the worst performers.
//! - [`Hyperband`] runs multiple brackets of Successive Halving with varying
//!   aggressiveness to hedge against the unknown ideal trade-off.
//!
//! # Quick start
//!
//! ```rust
//! use scirs2_optimize::multi_fidelity::{
//!     Hyperband, MultiFidelityConfig, ConfigSampler,
//! };
//!
//! let config = MultiFidelityConfig {
//!     max_budget: 81.0,
//!     min_budget: 1.0,
//!     eta: 3,
//!     ..Default::default()
//! };
//!
//! let hb = Hyperband::new(config).expect("valid config");
//! let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
//! let mut rng = 42u64;
//!
//! let result = hb.run(
//!     &|x: &[f64], _budget: f64| Ok(x.iter().map(|v| v * v).sum()),
//!     &bounds,
//!     &ConfigSampler::Random,
//!     &mut rng,
//! ).expect("optimization ok");
//!
//! println!("Best: {:?} -> {:.6}", result.best_config, result.best_objective);
//! ```

pub mod hyperband;
pub mod successive_halving;
pub mod types;

// Re-exports
pub use hyperband::Hyperband;
pub use successive_halving::SuccessiveHalving;
pub use types::{ConfigSampler, EvaluationResult, MultiFidelityConfig, MultiFidelityResult};
