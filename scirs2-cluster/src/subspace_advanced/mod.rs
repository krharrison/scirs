//! Advanced subspace clustering methods.
//!
//! This module provides state-of-the-art algorithms for clustering data that
//! lies on a *union of linear subspaces*:
//!
//! - **SSC** (Sparse Subspace Clustering, Elhamifar & Vidal 2013): finds a
//!   sparse self-expression matrix via LASSO / proximal gradient and then
//!   applies spectral clustering on the resulting affinity graph.
//! - **LRR** (Low-Rank Representation, Liu et al. 2013): recovers the global
//!   low-rank structure via nuclear-norm minimisation, yielding a robust
//!   affinity matrix.
//! - **ORSC** (Ordered Robust Subspace Clustering): extends LRR with an
//!   ordered/rank-constrained formulation for improved robustness.
//!
//! # References
//!
//! * Elhamifar, E. & Vidal, R. (2013). Sparse Subspace Clustering: Algorithm,
//!   Theory, and Applications. *TPAMI*.
//! * Liu, G. et al. (2013). Robust Recovery of Subspace Structures by
//!   Low-Rank Representation. *TPAMI*.

pub mod lrr;
pub mod orsc;
pub mod ssc;

pub use lrr::LowRankRepresentation;
pub use orsc::OrderedRobustSC;
pub use ssc::SparseSubspaceClustering;
