//! Least-squares iterative solvers for sparse (and general) linear systems
//!
//! This module provides three production-quality iterative methods for
//! solving sparse least-squares problems of the form
//!
//!   min ||Ax - b||_2     (or the damped/regularised variant)
//!
//! All solvers expose a closure-based interface (matvec / rmatvec) so they
//! can be used with any matrix representation, as well as a convenience
//! wrapper that accepts a [`crate::csr::CsrMatrix`] directly.
//!
//! # Solvers
//!
//! | Method | Description |
//! |--------|-------------|
//! | [`lsqr`] | Paige & Saunders (1982): bidiagonalisation + Givens QR. |
//! | [`lsmr`] | Fong & Saunders (2011): like LSQR but with smoother convergence. |
//! | [`cgls`] | Conjugate Gradient Least Squares (CG on normal equations). |

pub mod cgls;
pub mod lsmr;
pub mod lsqr;

pub use cgls::{cgls, cgls_sparse, CGLSResult};
pub use lsmr::{lsmr, lsmr_sparse, LSMRConfig, LSMRResult};
pub use lsqr::{lsqr, lsqr_sparse, LSQRConfig, LSQRResult};
