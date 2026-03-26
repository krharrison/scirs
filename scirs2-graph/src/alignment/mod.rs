//! Network alignment algorithms for finding correspondences between graphs.
//!
//! This module provides algorithms for aligning nodes across two different graphs,
//! finding mappings that preserve topological structure. Applications include
//! protein interaction network alignment, social network de-anonymization,
//! and ontology matching.
//!
//! # Algorithms
//!
//! - **IsoRank** (Singh et al., 2008): Spectral method using power iteration
//!   on the Kronecker product of adjacency matrices.
//! - **GRASP**: Meta-heuristic combining greedy randomized construction
//!   with local search refinement.
//!
//! # Example
//!
//! ```rust
//! use scirs2_graph::alignment::{isorank, AlignmentConfig};
//! use scirs2_core::ndarray::Array2;
//!
//! // Create two small graphs (path graphs)
//! let mut adj1 = Array2::zeros((3, 3));
//! adj1[[0, 1]] = 1.0; adj1[[1, 0]] = 1.0;
//! adj1[[1, 2]] = 1.0; adj1[[2, 1]] = 1.0;
//!
//! let adj2 = adj1.clone();
//! let config = AlignmentConfig::default();
//!
//! let result = isorank(&adj1, &adj2, None, &config).expect("alignment should succeed");
//! assert!(!result.mapping.is_empty());
//! ```

pub mod evaluation;
pub mod grasp;
pub mod isorank;
pub mod types;

// Re-export primary types and functions
pub use evaluation::{
    edge_conservation, induced_conserved_structure, node_correctness, symmetric_substructure_score,
};
pub use grasp::grasp_alignment;
pub use isorank::isorank;
pub use types::{AlignmentConfig, AlignmentResult, SimilarityMatrix};
