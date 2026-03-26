//! Temporal Graph Neural Networks: TGAT and TGN.
//!
//! This module provides two complementary temporal GNN architectures:
//!
//! | Model | Paper | Core Idea |
//! |-------|-------|-----------|
//! | [`TgatModel`] | Xu et al. (ICLR 2020) | Temporal attention over causal neighborhoods |
//! | [`TgnModel`] | Rossi et al. (ICML 2020) | Per-node memory + GRU update + temporal attention |
//!
//! Both models use the [`TimeEncode`] functional time encoding to capture
//! temporal dynamics without handcrafted features.
//!
//! ## Quick Start
//!
//! ### TGAT
//! ```rust
//! use scirs2_graph::temporal::tgnn::{TgatModel, TgatConfig, TgnnGraph, TgnnEdge};
//!
//! let config = TgatConfig { num_heads: 2, time_dim: 8, head_dim: 8, num_layers: 1, dropout: 0.0 };
//! let model = TgatModel::new(&config, 4).expect("model");
//!
//! let mut graph = TgnnGraph::with_zero_features(4, 4);
//! graph.add_edge(TgnnEdge::no_feat(0, 1, 1.0));
//! graph.add_edge(TgnnEdge::no_feat(1, 2, 2.0));
//!
//! let embeddings = model.forward(&graph, 5.0).expect("forward");
//! assert_eq!(embeddings.len(), 4);
//! ```
//!
//! ### TGN
//! ```rust
//! use scirs2_graph::temporal::tgnn::{TgnModel, TgnConfig, TgnnGraph, TgnnEdge};
//!
//! let config = TgnConfig { memory_dim: 8, message_dim: 8, time_dim: 8, node_feat_dim: 4, embedding_dim: 8 };
//! let mut model = TgnModel::new(&config, 4, 0).expect("model");
//!
//! let mut graph = TgnnGraph::with_zero_features(4, 4);
//! graph.add_edge(TgnnEdge::no_feat(0, 1, 1.0));
//!
//! model.process_events(&graph.edges).expect("process");
//! let embeddings = model.get_embeddings(&[0, 1], 5.0, &graph).expect("embed");
//! assert_eq!(embeddings.len(), 2);
//! ```

pub mod tgat;
pub mod tgn;
pub mod time_encoding;
pub mod types;

// Re-export primary types
pub use tgat::{TgatLayer, TgatModel};
pub use tgn::{TgnEmbedding, TgnMemory, TgnMemoryUpdater, TgnMessageModule, TgnModel};
pub use time_encoding::{PositionalTimeEncoding, TimeEncode};
pub use types::{TgatConfig, TgnnEdge, TgnnGraph, TgnConfig, TemporalPrediction};
