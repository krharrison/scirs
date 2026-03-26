//! Tensor-train decomposition and TT-cross approximation for high-dimensional functions.
//!
//! This module provides:
//! - `TensorTrain`: TT format with TT-SVD compression (Oseledets 2011)
//! - TT-cross (DMRG-cross): adaptive cross approximation without materialising the full tensor
//! - `tt_interp`: evaluate TT interpolant at arbitrary real-valued points (linear interp)
//!
//! # Mathematical Background
//!
//! A tensor `A[i1,...,id]` is approximated in TT format as the product of matrices:
//!
//! ```text
//! A[i1,...,id] ≈ G1[i1] * G2[i2] * ... * Gd[id]
//! ```
//!
//! where each `Gk[ik]` is an `(r_{k-1} × r_k)` matrix (with `r_0 = r_d = 1`).
//! The cores are stored as 3-D arrays of shape `[r_{k-1}, n_k, r_k]`.

pub mod tt_cross;
pub mod tt_decomp;

pub use tt_cross::*;
pub use tt_decomp::*;
