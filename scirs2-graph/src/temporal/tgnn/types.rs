//! Core types for Temporal Graph Neural Networks (TGAT, TGN).
//!
//! This module defines the fundamental data structures used by both TGAT
//! (Temporal Graph Attention Network, Xu et al. 2020) and TGN
//! (Temporal Graph Network, Rossi et al. 2020).

use crate::error::{GraphError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// TemporalEdge (TGNN version with feature vectors)
// ─────────────────────────────────────────────────────────────────────────────

/// A temporal edge with optional feature vector, used in TGAT/TGN.
///
/// Different from `crate::temporal::TemporalEdge` in that it carries an
/// arbitrary-length feature vector alongside the timestamp.
#[derive(Debug, Clone, PartialEq)]
pub struct TgnnEdge {
    /// Source node index (0-based)
    pub src: usize,
    /// Destination node index (0-based)
    pub dst: usize,
    /// Interaction timestamp (continuous, monotonically increasing per event stream)
    pub timestamp: f64,
    /// Edge feature vector (may be empty if no edge features)
    pub features: Vec<f64>,
}

impl TgnnEdge {
    /// Create a new temporal edge with a feature vector.
    pub fn new(src: usize, dst: usize, timestamp: f64, features: Vec<f64>) -> Self {
        TgnnEdge {
            src,
            dst,
            timestamp,
            features,
        }
    }

    /// Create a temporal edge with no features.
    pub fn no_feat(src: usize, dst: usize, timestamp: f64) -> Self {
        TgnnEdge {
            src,
            dst,
            timestamp,
            features: Vec::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TgnnGraph
// ─────────────────────────────────────────────────────────────────────────────

/// Temporal graph for TGAT/TGN: nodes with feature vectors + temporal edges.
///
/// Edges are stored sorted by timestamp to enable efficient causal
/// (time-respecting) neighborhood lookups.
#[derive(Debug, Clone)]
pub struct TgnnGraph {
    /// Number of nodes
    pub n_nodes: usize,
    /// Per-node feature vectors (n_nodes × node_feat_dim). May be empty rows.
    pub node_features: Vec<Vec<f64>>,
    /// Edge stream, maintained in timestamp order
    pub edges: Vec<TgnnEdge>,
    /// Node feature dimensionality (0 if no node features provided)
    pub node_feat_dim: usize,
    /// Edge feature dimensionality (0 if no edge features)
    pub edge_feat_dim: usize,
}

impl TgnnGraph {
    /// Create an empty `TgnnGraph` with `n_nodes` nodes and optional node features.
    ///
    /// If `node_features` is empty, each node gets a zero feature vector of
    /// dimension `default_node_dim`.
    pub fn new(n_nodes: usize, node_features: Vec<Vec<f64>>) -> Result<Self> {
        let node_feat_dim = if node_features.is_empty() {
            0
        } else {
            let dim = node_features[0].len();
            if node_features.len() != n_nodes {
                return Err(GraphError::InvalidParameter {
                    param: "node_features".to_string(),
                    value: format!("len={}", node_features.len()),
                    expected: format!("len={}", n_nodes),
                    context: "TgnnGraph::new".to_string(),
                });
            }
            dim
        };
        Ok(TgnnGraph {
            n_nodes,
            node_features,
            edges: Vec::new(),
            node_feat_dim,
            edge_feat_dim: 0,
        })
    }

    /// Create a graph with zero node features of given dimensionality.
    pub fn with_zero_features(n_nodes: usize, node_feat_dim: usize) -> Self {
        let node_features = vec![vec![0.0f64; node_feat_dim]; n_nodes];
        TgnnGraph {
            n_nodes,
            node_features,
            edges: Vec::new(),
            node_feat_dim,
            edge_feat_dim: 0,
        }
    }

    /// Add a temporal edge, updating `edge_feat_dim` if needed.
    pub fn add_edge(&mut self, edge: TgnnEdge) {
        if self.edge_feat_dim == 0 && !edge.features.is_empty() {
            self.edge_feat_dim = edge.features.len();
        }
        self.edges.push(edge);
    }

    /// Return neighbors of `node` with timestamp strictly less than `t`.
    ///
    /// Each entry is `(neighbor_idx, edge_timestamp, edge_features)`.
    /// Causality: only edges observed *before* query time `t` are returned.
    pub fn neighbors_before(
        &self,
        node: usize,
        t: f64,
    ) -> Vec<(usize, f64, &[f64])> {
        let mut result = Vec::new();
        for edge in &self.edges {
            if edge.timestamp >= t {
                // Since edges may not be sorted in general, we check all
                continue;
            }
            if edge.src == node {
                result.push((edge.dst, edge.timestamp, edge.features.as_slice()));
            } else if edge.dst == node {
                result.push((edge.src, edge.timestamp, edge.features.as_slice()));
            }
        }
        // Sort by timestamp descending — most recent first for attention
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Return the feature vector for node `i`, or an empty slice if out of bounds.
    pub fn node_feat(&self, i: usize) -> &[f64] {
        self.node_features.get(i).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Sort edges by timestamp (in-place). Algorithms that need ordered processing call this.
    pub fn sort_edges(&mut self) {
        self.edges.sort_by(|a, b| {
            a.timestamp
                .partial_cmp(&b.timestamp)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TgatConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Temporal Graph Attention Network (TGAT).
///
/// Follows Xu et al. (2020) "Inductive Representation Learning on Temporal Graphs".
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TgatConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimensionality of the time encoding φ(t)
    pub time_dim: usize,
    /// Output embedding dimension per head (total output = num_heads × head_dim)
    pub head_dim: usize,
    /// Number of TGAT layers (hops)
    pub num_layers: usize,
    /// Dropout rate (applied to attention weights; 0.0 = no dropout)
    pub dropout: f64,
}

impl Default for TgatConfig {
    fn default() -> Self {
        TgatConfig {
            num_heads: 2,
            time_dim: 8,
            head_dim: 16,
            num_layers: 2,
            dropout: 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TgnConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Temporal Graph Network (TGN).
///
/// Follows Rossi et al. (2020) "Temporal Graph Networks for Deep Learning on Dynamic Graphs".
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TgnConfig {
    /// Per-node memory vector dimensionality
    pub memory_dim: usize,
    /// Intermediate message dimensionality (output of message module)
    pub message_dim: usize,
    /// Time encoding dimension
    pub time_dim: usize,
    /// Node feature dimension for embedding computation
    pub node_feat_dim: usize,
    /// Output embedding dimension
    pub embedding_dim: usize,
}

impl Default for TgnConfig {
    fn default() -> Self {
        TgnConfig {
            memory_dim: 32,
            message_dim: 32,
            time_dim: 8,
            node_feat_dim: 16,
            embedding_dim: 32,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TemporalPrediction
// ─────────────────────────────────────────────────────────────────────────────

/// Output of temporal GNN: a node embedding at a given query time.
#[derive(Debug, Clone)]
pub struct TemporalPrediction {
    /// Node index
    pub node_id: usize,
    /// Computed embedding vector
    pub embedding: Vec<f64>,
    /// Query timestamp at which the embedding was computed
    pub timestamp: f64,
}

impl TemporalPrediction {
    /// Construct a new prediction record.
    pub fn new(node_id: usize, embedding: Vec<f64>, timestamp: f64) -> Self {
        TemporalPrediction {
            node_id,
            embedding,
            timestamp,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_graph_creation() {
        let feats = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let g = TgnnGraph::new(3, feats).expect("should create graph");
        assert_eq!(g.n_nodes, 3);
        assert_eq!(g.node_feat_dim, 2);
    }

    #[test]
    fn test_temporal_graph_neighbors_before_t() {
        let mut g = TgnnGraph::with_zero_features(4, 2);
        g.add_edge(TgnnEdge::no_feat(0, 1, 1.0));
        g.add_edge(TgnnEdge::no_feat(0, 2, 3.0));
        g.add_edge(TgnnEdge::no_feat(0, 3, 5.0));

        // Query at t=4.0 → should only see edges at t=1 and t=3
        let nbrs = g.neighbors_before(0, 4.0);
        let nbr_ids: Vec<usize> = nbrs.iter().map(|(n, _, _)| *n).collect();
        assert!(nbr_ids.contains(&1));
        assert!(nbr_ids.contains(&2));
        assert!(!nbr_ids.contains(&3), "future edge should not be visible");
    }

    #[test]
    fn test_temporal_graph_edge_features() {
        let mut g = TgnnGraph::with_zero_features(3, 2);
        g.add_edge(TgnnEdge::new(0, 1, 1.0, vec![0.5, 0.3]));
        g.add_edge(TgnnEdge::new(1, 2, 2.0, vec![0.1, 0.9]));

        let nbrs = g.neighbors_before(1, 3.0);
        // Node 1 has neighbor 0 at t=1 and neighbor 2 at t=2 (but directed: 1→2 not 2→1)
        // with undirected lookup: both (0→1) and (1→2) should appear
        assert!(!nbrs.is_empty());
        // Check features are accessible
        let first = &nbrs[0];
        assert!(!first.2.is_empty());
    }

    #[test]
    fn test_tgat_config_default() {
        let cfg = TgatConfig::default();
        assert_eq!(cfg.num_heads, 2);
        assert_eq!(cfg.time_dim, 8);
        assert_eq!(cfg.head_dim, 16);
    }

    #[test]
    fn test_tgn_config_default() {
        let cfg = TgnConfig::default();
        assert_eq!(cfg.memory_dim, 32);
        assert_eq!(cfg.message_dim, 32);
    }

    #[test]
    fn test_tgnn_graph_with_wrong_features_returns_error() {
        let feats = vec![vec![1.0, 2.0]]; // only 1 row for 3 nodes
        let result = TgnnGraph::new(3, feats);
        assert!(result.is_err());
    }
}
