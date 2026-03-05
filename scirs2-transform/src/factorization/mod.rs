//! Matrix factorization methods
//!
//! This module provides a collection of NMF variants and related matrix
//! factorization algorithms for data analysis and feature learning.

/// NMF variants: standard NMF, Semi-NMF, Convex NMF, Robust NMF, Deep NMF.
pub mod nmf_variants;

pub use nmf_variants::{
    nmf_quality, ConvexNMF, DeepNMF, NmfDivergence, RobustNMF, SemiNMF, NMF,
};
