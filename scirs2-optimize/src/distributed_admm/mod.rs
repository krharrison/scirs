//! Distributed ADMM, PDMM, and EXTRA algorithms.
//!
//! This module implements distributed optimization methods based on the
//! Alternating Direction Method of Multipliers (ADMM) and related algorithms:
//!
//! - **Consensus ADMM** (Boyd et al. 2011): Scalable distributed optimization
//!   using augmented Lagrangian decomposition with warm-starting support.
//! - **Lasso via ADMM**: Canonical example with closed-form x/z updates.
//! - **PDMM** (Chang et al. 2015): Decentralized ADMM over general network topologies.
//! - **EXTRA** (Shi et al. 2015): Gradient-tracking algorithm for exact decentralized
//!   consensus without diminishing step sizes.
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_optimize::distributed_admm::{solve_lasso_admm, AdmmConfig};
//!
//! let a = vec![
//!     vec![1.0, 0.0],
//!     vec![0.0, 1.0],
//!     vec![1.0, 1.0],
//! ];
//! let b = vec![1.0, 0.0, 1.0];
//! let config = AdmmConfig::default();
//! let result = solve_lasso_admm(&a, &b, 0.1, &config).expect("Lasso ADMM failed");
//! println!("Solution: {:?}", result.x);
//! println!("Converged: {}", result.converged);
//! ```
//!
//! # References
//! - Boyd et al. (2011). "Distributed Optimization and Statistical Learning via
//!   the Alternating Direction Method of Multipliers." Foundations and Trends in ML.
//! - Chang et al. (2015). "Multi-Agent Distributed Optimization via Inexact
//!   Consensus ADMM." IEEE Trans. Signal Processing.
//! - Shi et al. (2015). "EXTRA: An Exact First-Order Algorithm for Decentralized
//!   Consensus Optimization." SIAM J. Optim.

pub mod admm;
pub mod pdmm_extra;
pub mod types;

pub use admm::{consensus_admm, solve_lasso_admm, AdmmSolver};
pub use pdmm_extra::{metropolis_hastings_weights, ring_topology, ExtraSolver, PdmmSolver};
pub use types::{AdmmConfig, AdmmResult, ConsensusNode, ExtraConfig, PdmmConfig};
