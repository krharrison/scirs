//! Low-rank updates to sparse factorizations (LU, QR, sketched).
//!
//! This module implements efficient low-rank modifications to existing matrix
//! factorizations, avoiding the cost of a full re-factorization when a matrix
//! is perturbed by a low-rank term.
//!
//! # Supported operations
//!
//! | Operation | Function | Formula |
//! |-----------|----------|---------|
//! | Sherman-Morrison-Woodbury | [`sherman_morrison_woodbury`] | (A + UCV)^{-1} |
//! | LU rank-1 update | [`lu_rank1_update`] | PA = LU, A' = A + uv^T |
//! | LU column replace | [`lu_column_replace`] | Replace column k of A |
//! | QR rank-1 update | [`qr_rank1_update`] | A = QR, A' = A + uv^T |
//! | QR column insert | [`qr_column_insert`] | Insert column at position k |
//! | QR column delete | [`qr_column_delete`] | Delete column at position k |
//! | Nystrom update | [`nystrom_update`] | Approximate A + UV^T via sketch |
//! | Randomized update | [`randomized_low_rank_update`] | Randomized range approximation |
//!
//! # References
//!
//! - Golub, Van Loan (2013). *Matrix Computations*, 4th ed., Johns Hopkins.
//! - Halko, Martinsson, Tropp (2011). "Finding structure with randomness."
//!   *SIAM Review* 53(2).

/// Types and configuration for low-rank factorization updates.
pub mod types;

/// LU factorization updates (Sherman-Morrison-Woodbury, rank-1, column replace).
pub mod lu_update;

/// QR factorization updates (rank-1, column insert/delete).
pub mod qr_update;

/// Sketched / randomized low-rank updates (Nystrom, randomized).
pub mod sketched_update;

// Re-exports for convenience
pub use lu_update::{lu_column_replace, lu_rank1_update, sherman_morrison_woodbury};
pub use qr_update::{qr_column_delete, qr_column_insert, qr_rank1_update};
pub use sketched_update::{nystrom_update, randomized_low_rank_update, update_error_bound};
pub use types::{
    FactorizationType, LUUpdateResult, LowRankUpdateConfig, QRUpdateResult, SketchConfig,
};
