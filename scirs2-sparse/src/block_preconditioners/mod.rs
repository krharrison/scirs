//! Advanced block-level preconditioners for sparse linear systems
//!
//! This module provides preconditioners that operate on block structure,
//! complementing the scalar preconditioners in `crate::preconditioners`.
//!
//! # Available Preconditioners
//!
//! | Type | Module | Description |
//! |------|--------|-------------|
//! | [`BlockJacobiPreconditioner`] | [`block_jacobi`] | Block Jacobi — inverts diagonal blocks independently |
//! | [`BlockILU0`] | [`block_ilu`] | Block ILU(0) — incomplete LU on BSR block structure |
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_sparse::block_preconditioners::{BlockJacobiPreconditioner};
//!
//! let n = 4;
//! let indptr: Vec<usize> = (0..=n).collect();
//! let indices: Vec<usize> = (0..n).collect();
//! let data: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0];
//!
//! let mut prec = BlockJacobiPreconditioner::<f64>::new(2).expect("new failed");
//! prec.setup(n, &indptr, &indices, &data).expect("setup failed");
//! let y = prec.apply(&[2.0, 4.0, 6.0, 8.0]).expect("apply failed");
//! ```

pub mod block_jacobi;
pub mod block_ilu;

pub use block_jacobi::BlockJacobiPreconditioner;
pub use block_ilu::{BlockILU0, apply_block_ilu};
