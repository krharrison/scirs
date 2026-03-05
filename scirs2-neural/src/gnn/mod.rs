//! Graph Neural Networks for learning on graph-structured data.
//!
//! This module provides:
//!
//! - **Graph data structures**: Dense [`Graph`] and sparse CSR [`SparseGraph`].
//! - **GCN**: Spectral graph convolution (Kipf & Welling, 2017) — [`GCNLayer`].
//! - **GraphSAGE**: Inductive neighbourhood aggregation (Hamilton et al., 2017) — [`GraphSAGELayer`].
//! - **GAT**: Multi-head graph attention (Veličković et al., 2018) — [`GATLayer`].
//! - **GIN**: Graph Isomorphism Network (Xu et al., 2019) — [`GINLayer`].
//! - **Pooling**: Global and hierarchical graph pooling — [`GlobalMeanPool`],
//!   [`GlobalMaxPool`], [`GlobalAddPool`], [`DiffPool`].
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_neural::gnn::{Graph, GCNLayer};
//! use scirs2_neural::gnn::gcn::Activation;
//!
//! // Build a small graph
//! let mut g = Graph::new(4, 8);
//! g.add_undirected_edge(0, 1).expect("operation should succeed");
//! g.add_undirected_edge(1, 2).expect("operation should succeed");
//! g.add_undirected_edge(2, 3).expect("operation should succeed");
//! for i in 0..4 { g.set_node_features(i, vec![1.0; 8]).expect("operation should succeed"); }
//!
//! // Apply a GCN layer
//! let layer = GCNLayer::new(8, 16, Activation::ReLU);
//! let out = layer.forward(&g, &g.node_features).expect("forward ok");
//! assert_eq!(out.len(), 4);
//! assert_eq!(out[0].len(), 16);
//! ```

pub mod gat;
pub mod gcn;
pub mod gin;
pub mod graph;
pub mod pooling;
pub mod sage;

pub use gat::GATLayer;
pub use gcn::GCNLayer;
pub use gin::GINLayer;
pub use graph::{Graph, SparseGraph};
pub use pooling::{DiffPool, GlobalAddPool, GlobalMaxPool, GlobalMeanPool};
pub use sage::GraphSAGELayer;
