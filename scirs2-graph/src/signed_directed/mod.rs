//! Signed and directed graph learning with specialised embeddings.
//!
//! This module provides:
//! - [`types`]: Core data structures (`SignedGraph`, `DirectedGraph`, config/result types).
//! - [`signed_spectral`]: Signed Laplacian, SPONGE embedding, ratio-cut clustering,
//!   and status-theory scores.
//! - [`directed_embedding`]: HOPE and APP directed graph embeddings.
//! - [`signed_gcn`]: Balance-theory Signed GCN (Derr et al. 2018).

pub mod directed_embedding;
pub mod signed_gcn;
pub mod signed_spectral;
pub mod types;

// Convenience re-exports
pub use directed_embedding::{app_embedding, hope_embedding, stationary_distribution};
pub use signed_gcn::{predict_sign, SgcnLayer, SgcnModel};
pub use signed_spectral::{
    negative_laplacian, positive_laplacian, signed_laplacian, signed_ratio_cut, sponge_embedding,
    status_score,
};
pub use types::{
    DirectedEdge, DirectedEmbedConfig, DirectedGraph, EmbeddingResult, SignedEdge,
    SignedEmbedConfig, SignedGraph,
};
