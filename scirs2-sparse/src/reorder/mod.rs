//! Sparse matrix reordering algorithms
//!
//! This module provides algorithms for reordering sparse matrices to reduce
//! bandwidth, minimize fill-in during factorization, or enable parallel
//! computation through graph coloring.
//!
//! # Algorithms
//!
//! ## Bandwidth reduction
//!
//! - [`cuthill_mckee()`] / [`reverse_cuthill_mckee`]: BFS-based bandwidth reduction.
//!   RCM is one of the most widely used reordering algorithms and typically
//!   produces good bandwidth and profile reduction.
//!
//! ## Fill-reducing orderings
//!
//! - [`amd()`]: Approximate Minimum Degree ordering using a quotient graph
//!   representation. Produces orderings with low fill-in for sparse Cholesky
//!   and LU factorization.
//! - [`nested_dissection()`]: Recursive graph bisection. Near-optimal fill for
//!   2D/3D mesh-based problems. Supports multi-level coarsening.
//!
//! ## Graph coloring
//!
//! - [`greedy_color`]: Greedy coloring with configurable vertex orderings
//!   (natural, largest-first, smallest-last).
//! - [`dsatur_color`]: Degree-of-saturation coloring, which often produces
//!   fewer colors than simple greedy.
//! - [`distance2_color`]: Distance-2 coloring for sparse Jacobian computation.
//!
//! # Utilities
//!
//! - [`AdjacencyGraph`]: Lightweight undirected graph built from CSR/CSC matrices.
//! - [`apply_permutation`] / [`apply_permutation_csr_array`]: Apply a symmetric
//!   permutation P*A*P^T to a sparse matrix.
//! - [`bandwidth`] / [`profile`]: Compute bandwidth/profile metrics.

pub mod adjacency;
pub mod amd;
pub mod cuthill_mckee;
pub mod graph_coloring;
pub mod nested_dissection;

// Re-export primary types and functions
pub use adjacency::{apply_permutation, apply_permutation_csr_array, AdjacencyGraph};

pub use cuthill_mckee::{
    bandwidth, cuthill_mckee, cuthill_mckee_full, profile, reverse_cuthill_mckee,
    reverse_cuthill_mckee_full, CuthillMcKeeResult,
};

pub use amd::{amd, amd_simple, AmdResult};

pub use nested_dissection::{
    nested_dissection, nested_dissection_full, nested_dissection_with_config,
    NestedDissectionConfig, NestedDissectionResult,
};

pub use graph_coloring::{
    distance2_color, dsatur_color, greedy_color, verify_coloring, verify_distance2_coloring,
    ColoringResult, GreedyOrdering,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reorder_smoke() {
        let adj = vec![vec![1], vec![0]];
        let graph = AdjacencyGraph::from_adjacency_list(adj);
        let perm = cuthill_mckee(&graph).expect("CM");
        assert_eq!(perm.len(), 2);
    }
}
