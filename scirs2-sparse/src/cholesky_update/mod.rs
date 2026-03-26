//! Sparse Cholesky modifications (rank-1 updates and downdates).
//!
//! This module implements efficient rank-1 and low-rank modifications to
//! an existing Cholesky factorization, avoiding the cost of a full
//! re-factorization when a symmetric positive-definite matrix A = L L^T
//! is perturbed by a low-rank term.
//!
//! # Supported operations
//!
//! | Operation | Function | Formula |
//! |-----------|----------|---------|
//! | Rank-1 update | [`cholesky_rank1_update`] | A' = A + α u u^T |
//! | Rank-1 downdate | [`cholesky_rank1_downdate`] | A' = A − α u u^T |
//! | Multi-rank update | [`cholesky_rank_k_update`] | A' = A + Σ w_i v_i v_i^T |
//! | Low-rank W D W^T | [`cholesky_low_rank_update`] | A' = A + W D W^T |
//!
//! # Example
//!
//! ```
//! use scirs2_sparse::cholesky_update::{cholesky_factorize, cholesky_rank1_update};
//!
//! let a = vec![
//!     vec![4.0, 2.0],
//!     vec![2.0, 5.0],
//! ];
//! let l = cholesky_factorize(&a).expect("factorize");
//! let u = vec![1.0, 0.5];
//! let l_new = cholesky_rank1_update(&l, &u, 1.0).expect("update");
//! ```
//!
//! # References
//!
//! - Gill, Golub, Murray, Saunders (1974). "Methods for modifying matrix
//!   factorizations." *Math. Comp.* 28(126).
//! - Stewart (1979). "The effects of rounding error on an algorithm for
//!   downdating a Cholesky factorization." *IMA J. Appl. Math.* 23(2).

/// Types and configuration for Cholesky modifications.
pub mod types;

/// Rank-1 Cholesky update via Givens rotations.
pub mod rank1;

/// Rank-1 Cholesky downdate via hyperbolic Givens rotations.
pub mod downdate;

/// Multiple rank-1 and low-rank Cholesky updates.
pub mod multiple;

// Re-exports for convenience
pub use downdate::{
    cholesky_rank1_downdate, cholesky_rank1_downdate_result, cholesky_rank1_downdate_with_config,
};
pub use multiple::{
    cholesky_low_rank_update, cholesky_low_rank_update_with_config, cholesky_rank_k_update,
    cholesky_rank_k_update_with_config,
};
pub use rank1::{
    cholesky_factorize, cholesky_rank1_update, cholesky_rank1_update_result,
    cholesky_rank1_update_with_config,
};
pub use types::{CholUpdateConfig, CholUpdateResult, UpdateType};
