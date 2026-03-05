//! Advanced matrix decomposition algorithms organised as a dedicated submodule.
//!
//! This module groups specialised decompositions that complement the standard
//! factorizations available through the top-level `scirs2_linalg` API.
//!
//! # Submodules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`gsvd`] | Generalized Singular Value Decomposition |
//!
//! # Quick Start
//!
//! ```
//! use scirs2_core::ndarray::array;
//! use scirs2_linalg::factorization::gsvd::gsvd;
//!
//! let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
//! let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
//! let result = gsvd(&a.view(), &b.view()).expect("gsvd");
//!
//! // alpha² + beta² = 1
//! for i in 0..result.alpha.len() {
//!     let sq = result.alpha[i].powi(2) + result.beta[i].powi(2);
//!     assert!((sq - 1.0).abs() < 1e-10);
//! }
//! ```

pub mod gsvd;

pub use gsvd::{generalized_singular_values, gsvd, GsvdResult};
