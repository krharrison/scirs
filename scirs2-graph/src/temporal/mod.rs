//! Temporal graph analysis algorithms (stream model, f64 timestamps)
//!
//! This module provides a self-contained temporal network analysis framework
//! built around the **stream-of-interactions** model where edges carry
//! real-valued timestamps.  It is focused on composable algorithm modules:
//!
//! | Sub-module   | Contents                                                        |
//! |--------------|----------------------------------------------------------------|
//! | `graph`      | `TemporalEdge`, `TemporalGraph` data structures                |
//! | `centrality` | Temporal closeness, betweenness, and PageRank                  |
//! | `motifs`     | 3-node/3-edge δ-temporal motif counting (8 types)             |
//! | `community`  | Evolutionary clustering (dynamic community detection)          |
//!
//! # Quick start
//!
//! ```rust
//! use scirs2_graph::temporal::{TemporalGraph, TemporalEdge};
//! use scirs2_graph::temporal::centrality::temporal_betweenness;
//!
//! let mut tg = TemporalGraph::new(4);
//! tg.add_edge(TemporalEdge::new(0, 1, 1.0));
//! tg.add_edge(TemporalEdge::new(1, 2, 2.0));
//! tg.add_edge(TemporalEdge::new(2, 3, 3.0));
//!
//! let bet = temporal_betweenness(&mut tg);
//! assert_eq!(bet.len(), 4);
//! ```

pub mod centrality;
pub mod community;
pub mod graph;
pub mod motifs;

// Re-export the most commonly used types
pub use centrality::{temporal_betweenness, temporal_closeness, temporal_pagerank};
pub use community::{evolutionary_clustering, DynamicCommunity};
pub use graph::{TemporalEdge, TemporalGraph};
pub use motifs::{count_temporal_triangles, temporal_motif_count, TemporalMotifCounts};
