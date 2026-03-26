//! Second-order stochastic optimization algorithms.
//!
//! This module provides second-order quasi-Newton methods suitable for both
//! deterministic and stochastic optimization:
//!
//! - **L-BFGS-B** (`lbfgsb`): Limited-memory BFGS with box constraints,
//!   featuring strong Wolfe line search and Cauchy-point computation.
//! - **SR1** (`sr1`): Symmetric rank-1 quasi-Newton with trust-region
//!   globalization and limited-memory compact representation.
//! - **S-L-BFGS** (`slbfgs`): Stochastic L-BFGS combining mini-batch
//!   gradients, curvature pairs from large batches, and optional SVRG
//!   variance reduction.
//!
//! ## References
//!
//! - Byrd et al. (1995). "A limited memory algorithm for bound constrained
//!   optimization." SIAM J. Sci. Comput.
//! - Byrd, Khalfan & Schnabel (1994). "Analysis of a symmetric rank-one
//!   trust region method." SIAM J. Optim.
//! - Moritz, Nishihara & Jordan (2016). "A linearly-convergent stochastic
//!   L-BFGS algorithm." AISTATS.

pub mod lbfgsb;
pub mod slbfgs;
pub mod sr1;
pub mod types;

// Re-exports for convenient access
pub use lbfgsb::{
    cauchy_point, hv_product, project, projected_grad_norm, wolfe_line_search, LbfgsBOptimizer,
};
pub use slbfgs::{Lcg, SlbfgsOptimizer};
pub use sr1::{lsr1_hv_product, sr1_update_dense, trust_region_step, Sr1Optimizer};
pub use types::{HessianApprox, LbfgsBConfig, OptResult, SlbfgsConfig, Sr1Config};
