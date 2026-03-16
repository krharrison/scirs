//! Graph Neural Network (GNN) layers and message-passing framework
//!
//! This module implements core GNN building blocks following the message-passing
//! neural network (MPNN) paradigm:
//!
//! - **Message Passing**: flexible aggregation over neighborhoods
//! - **GCNLayer** (legacy Vec-based API): Graph Convolutional Network (Kipf & Welling 2017)
//! - **GraphSAGELayer** (legacy Vec-based API): Sample-and-aggregate (Hamilton et al. 2017)
//! - **GATLayer** (legacy Vec-based API): Graph Attention Network (Veličković et al. 2018)
//!
//! ### Array2-based (ndarray) API (new in 0.3.0)
//!
//! - [`gcn`] – `GcnLayer`, `Gcn`, `gcn_forward`, `add_self_loops`, `CsrMatrix`
//! - [`sage`] – `GraphSageLayer`, `GraphSage`, `sage_aggregate`, `SageAggregation`
//! - [`gat`] – `GraphAttentionLayer`, `gat_forward`

// --- Sub-modules with the new Array2-based API ---
pub mod gcn;
pub mod gat;
pub mod sage;

// --- Re-export new Array2 API types ---
pub use gcn::{CsrMatrix, GcnLayer, Gcn, gcn_forward, add_self_loops, symmetric_normalize};
pub use sage::{GraphSageLayer, GraphSage, SageAggregation, sage_aggregate, sample_neighbors};
pub use gat::{GraphAttentionLayer, gat_forward};

// --- Legacy Vec-based message-passing API (kept for backward compatibility) ---
// The following items are re-exported from the inline implementation below so
// that the existing `lib.rs` re-exports continue to work without modification.

use std::collections::HashMap;

use scirs2_core::random::{Rng, RngExt};

use crate::base::{EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

// ============================================================================
// Message aggregation types (legacy)
// ============================================================================

/// Aggregation strategy for collecting neighbor messages (legacy Vec-based API)
#[derive(Debug, Clone, PartialEq)]
pub enum MessagePassing {
    /// Sum all neighbor messages
    Sum,
    /// Arithmetic mean of neighbor messages
    Mean,
    /// Element-wise maximum
    Max,
    /// Element-wise minimum
    Min,
    /// Attention-weighted mean (weights computed internally)
    Attention,
}

impl Default for MessagePassing {
    fn default() -> Self {
        MessagePassing::Mean
    }
}

// ============================================================================
// MessagePassingLayer trait (legacy)
// ============================================================================

/// Core trait for GNN layers following the message-passing paradigm
///
/// Implementors must provide `aggregate` (neighbourhood → message) and
/// `update` (message + self → new embedding) methods.
pub trait MessagePassingLayer {
    /// Aggregate messages from the neighborhood of each node
    fn aggregate(
        &self,
        node_features: &[Vec<f64>],
        adjacency: &[(usize, usize, f64)],
        n_nodes: usize,
    ) -> Result<Vec<Vec<f64>>>;

    /// Update node embeddings using aggregated messages and self-features
    fn update(
        &self,
        aggregated: &[Vec<f64>],
        node_features: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>>;

    /// Run one full forward pass: aggregate + update
    fn forward(
        &self,
        node_features: &[Vec<f64>],
        adjacency: &[(usize, usize, f64)],
    ) -> Result<Vec<Vec<f64>>> {
        let n = node_features.len();
        let aggregated = self.aggregate(node_features, adjacency, n)?;
        self.update(&aggregated, node_features)
    }
}

// ============================================================================
// Helper utilities (legacy)
// ============================================================================

fn validate_features(features: &[Vec<f64>]) -> Result<usize> {
    if features.is_empty() {
        return Ok(0);
    }
    let dim = features[0].len();
    for (i, row) in features.iter().enumerate() {
        if row.len() != dim {
            return Err(GraphError::InvalidParameter {
                param: "node_features".to_string(),
                value: format!("row {} has {} dims, expected {}", i, row.len(), dim),
                expected: format!("all rows must have {} dimensions", dim),
                context: "GNN feature validation".to_string(),
            });
        }
    }
    Ok(dim)
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn softmax_vec(xs: &[f64]) -> Vec<f64> {
    if xs.is_empty() {
        return Vec::new();
    }
    let max_val = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = xs.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum::<f64>().max(1e-10);
    exps.iter().map(|e| e / sum).collect()
}

fn matvec(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter().map(|row| dot(row, x)).collect()
}

/// Convert graph to sparse adjacency (src, dst, weight) triples
pub fn graph_to_adjacency<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
) -> (Vec<N>, Vec<(usize, usize, f64)>)
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Clone + Into<f64>,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let node_to_idx: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    let mut adjacency = Vec::new();
    for edge in graph.edges() {
        if let (Some(&si), Some(&ti)) = (
            node_to_idx.get(&edge.source),
            node_to_idx.get(&edge.target),
        ) {
            let w: f64 = edge.weight.clone().into();
            adjacency.push((si, ti, w));
            adjacency.push((ti, si, w)); // undirected
        }
    }

    (nodes, adjacency)
}

// ============================================================================
// GCNLayer (legacy Vec-based)
// ============================================================================

/// Graph Convolutional Network layer (Kipf & Welling, 2017) - legacy Vec API
///
/// Forward pass:
/// ```text
///   H' = σ( D̃^{-1/2} Ã D̃^{-1/2} H W )
/// ```
#[derive(Debug, Clone)]
pub struct GCNLayer {
    /// Weight matrix (out_dim × in_dim)
    pub weights: Vec<Vec<f64>>,
    /// Bias vector (out_dim)
    pub bias: Vec<f64>,
    /// Output dimension
    pub out_dim: usize,
    /// Aggregation strategy
    pub aggregation: MessagePassing,
    /// Whether to apply ReLU activation
    pub use_activation: bool,
}

impl GCNLayer {
    /// Create a new GCN layer
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let scale = (2.0 / (in_dim + out_dim) as f64).sqrt();
        let mut weights = vec![vec![0.0f64; in_dim]; out_dim];
        for (i, row) in weights.iter_mut().enumerate() {
            for (j, w) in row.iter_mut().enumerate() {
                *w = if i == j {
                    scale
                } else {
                    scale * 0.01 * ((i as f64 - j as f64).sin())
                };
            }
        }
        GCNLayer {
            weights,
            bias: vec![0.0; out_dim],
            out_dim,
            aggregation: MessagePassing::Mean,
            use_activation: true,
        }
    }

    /// Set custom weights
    pub fn with_weights(mut self, weights: Vec<Vec<f64>>) -> Result<Self> {
        if weights.len() != self.out_dim {
            return Err(GraphError::InvalidParameter {
                param: "weights".to_string(),
                value: format!("rows={}", weights.len()),
                expected: format!("rows={}", self.out_dim),
                context: "GCNLayer::with_weights".to_string(),
            });
        }
        self.weights = weights;
        Ok(self)
    }
}

impl MessagePassingLayer for GCNLayer {
    fn aggregate(
        &self,
        node_features: &[Vec<f64>],
        adjacency: &[(usize, usize, f64)],
        n_nodes: usize,
    ) -> Result<Vec<Vec<f64>>> {
        let in_dim = validate_features(node_features)?;
        if in_dim == 0 {
            return Ok(Vec::new());
        }

        let mut deg = vec![1.0f64; n_nodes];
        for &(src, dst, _) in adjacency {
            deg[src] += 1.0;
            let _ = dst;
        }

        let mut agg: Vec<Vec<f64>> = (0..n_nodes).map(|_| vec![0.0f64; in_dim]).collect();

        for i in 0..n_nodes {
            let d_inv = 1.0 / deg[i].sqrt();
            for k in 0..in_dim {
                agg[i][k] += d_inv * node_features[i][k] * d_inv;
            }
        }

        for &(src, dst, w) in adjacency {
            if src < n_nodes && dst < n_nodes {
                let norm = w / (deg[src].sqrt() * deg[dst].sqrt());
                for k in 0..in_dim {
                    agg[dst][k] += norm * node_features[src][k];
                }
            }
        }

        Ok(agg)
    }

    fn update(
        &self,
        aggregated: &[Vec<f64>],
        _node_features: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>> {
        let mut result = Vec::with_capacity(aggregated.len());
        for agg in aggregated {
            let mut h = matvec(&self.weights, agg);
            for (hi, bi) in h.iter_mut().zip(self.bias.iter()) {
                *hi += bi;
                if self.use_activation {
                    *hi = relu(*hi);
                }
            }
            result.push(h);
        }
        Ok(result)
    }
}

// ============================================================================
// GraphSAGELayer (legacy Vec-based)
// ============================================================================

/// GraphSAGE layer - legacy Vec API
#[derive(Debug, Clone)]
pub struct GraphSAGELayer {
    /// Weight matrix for concatenated [self || neighbor_agg] (out × 2*in)
    pub weights: Vec<Vec<f64>>,
    /// Bias (out_dim)
    pub bias: Vec<f64>,
    /// Output dimension
    pub out_dim: usize,
    /// Aggregation strategy
    pub aggregation: MessagePassing,
    /// Whether to apply ReLU
    pub use_activation: bool,
}

impl GraphSAGELayer {
    /// Create a new GraphSAGE layer
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let concat_dim = 2 * in_dim;
        let scale = (2.0 / (concat_dim + out_dim) as f64).sqrt();
        let mut weights = vec![vec![0.0f64; concat_dim]; out_dim];
        for (i, row) in weights.iter_mut().enumerate() {
            for (j, w) in row.iter_mut().enumerate() {
                *w = if i == j % out_dim {
                    scale
                } else {
                    scale * 0.01 * ((i as f64 - j as f64).cos())
                };
            }
        }
        GraphSAGELayer {
            weights,
            bias: vec![0.0; out_dim],
            out_dim,
            aggregation: MessagePassing::Mean,
            use_activation: true,
        }
    }
}

impl MessagePassingLayer for GraphSAGELayer {
    fn aggregate(
        &self,
        node_features: &[Vec<f64>],
        adjacency: &[(usize, usize, f64)],
        n_nodes: usize,
    ) -> Result<Vec<Vec<f64>>> {
        let in_dim = validate_features(node_features)?;
        if in_dim == 0 {
            return Ok(Vec::new());
        }

        let mut neighbor_sums: Vec<Vec<f64>> =
            (0..n_nodes).map(|_| vec![0.0f64; in_dim]).collect();
        let mut neighbor_counts: Vec<f64> = vec![0.0; n_nodes];
        let mut neighbor_max: Vec<Vec<f64>> =
            (0..n_nodes).map(|_| vec![f64::NEG_INFINITY; in_dim]).collect();
        let mut neighbor_min: Vec<Vec<f64>> =
            (0..n_nodes).map(|_| vec![f64::INFINITY; in_dim]).collect();

        for &(src, dst, _) in adjacency {
            if src < n_nodes && dst < n_nodes {
                neighbor_counts[dst] += 1.0;
                for k in 0..in_dim {
                    neighbor_sums[dst][k] += node_features[src][k];
                    if node_features[src][k] > neighbor_max[dst][k] {
                        neighbor_max[dst][k] = node_features[src][k];
                    }
                    if node_features[src][k] < neighbor_min[dst][k] {
                        neighbor_min[dst][k] = node_features[src][k];
                    }
                }
            }
        }

        let agg_neighbor: Vec<Vec<f64>> = (0..n_nodes)
            .map(|i| {
                let count = neighbor_counts[i].max(1.0);
                match &self.aggregation {
                    MessagePassing::Sum => neighbor_sums[i].clone(),
                    MessagePassing::Mean => {
                        neighbor_sums[i].iter().map(|s| s / count).collect()
                    }
                    MessagePassing::Max => neighbor_max[i]
                        .iter()
                        .map(|&v| if v == f64::NEG_INFINITY { 0.0 } else { v })
                        .collect(),
                    MessagePassing::Min => neighbor_min[i]
                        .iter()
                        .map(|&v| if v == f64::INFINITY { 0.0 } else { v })
                        .collect(),
                    MessagePassing::Attention => {
                        neighbor_sums[i].iter().map(|s| s / count).collect()
                    }
                }
            })
            .collect();

        let concat: Vec<Vec<f64>> = node_features
            .iter()
            .zip(agg_neighbor.iter())
            .map(|(self_feat, nbr)| {
                let mut cat = self_feat.clone();
                cat.extend_from_slice(nbr);
                cat
            })
            .collect();

        Ok(concat)
    }

    fn update(
        &self,
        aggregated: &[Vec<f64>],
        _node_features: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>> {
        let mut result = Vec::with_capacity(aggregated.len());
        for agg in aggregated {
            let mut h = matvec(&self.weights, agg);
            for (hi, bi) in h.iter_mut().zip(self.bias.iter()) {
                *hi += bi;
                if self.use_activation {
                    *hi = relu(*hi);
                }
            }
            let norm: f64 = h.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);
            h.iter_mut().for_each(|x| *x /= norm);
            result.push(h);
        }
        Ok(result)
    }
}

// ============================================================================
// GATLayer (legacy Vec-based)
// ============================================================================

/// Graph Attention Network layer - legacy Vec API
#[derive(Debug, Clone)]
pub struct GATLayer {
    /// Node transformation matrix W (out_dim × in_dim)
    pub weights: Vec<Vec<f64>>,
    /// Attention vector a (2 * out_dim)
    pub attention_weights: Vec<f64>,
    /// Output dimension
    pub out_dim: usize,
    /// LeakyReLU negative slope
    pub negative_slope: f64,
    /// Whether to apply ELU activation on output
    pub use_activation: bool,
}

impl GATLayer {
    /// Create a new GAT layer
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let scale = (2.0 / (in_dim + out_dim) as f64).sqrt();
        let mut weights = vec![vec![0.0f64; in_dim]; out_dim];
        for (i, row) in weights.iter_mut().enumerate() {
            for (j, w) in row.iter_mut().enumerate() {
                *w = if i == j { scale } else { scale * 0.01 };
            }
        }
        let attention_weights: Vec<f64> = (0..2 * out_dim)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        GATLayer {
            weights,
            attention_weights,
            out_dim,
            negative_slope: 0.2,
            use_activation: true,
        }
    }

    fn leaky_relu(&self, x: f64) -> f64 {
        if x >= 0.0 { x } else { self.negative_slope * x }
    }
}

impl MessagePassingLayer for GATLayer {
    fn aggregate(
        &self,
        node_features: &[Vec<f64>],
        adjacency: &[(usize, usize, f64)],
        n_nodes: usize,
    ) -> Result<Vec<Vec<f64>>> {
        let _in_dim = validate_features(node_features)?;

        let transformed: Vec<Vec<f64>> = node_features
            .iter()
            .map(|h| matvec(&self.weights, h))
            .collect();

        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for &(src, dst, _) in adjacency {
            if src < n_nodes && dst < n_nodes {
                neighbors[dst].push(src);
            }
        }
        for i in 0..n_nodes {
            if !neighbors[i].contains(&i) {
                neighbors[i].push(i);
            }
        }

        let mut aggregated: Vec<Vec<f64>> = vec![vec![0.0; self.out_dim]; n_nodes];

        for i in 0..n_nodes {
            let nbrs = &neighbors[i];
            if nbrs.is_empty() {
                continue;
            }

            let scores: Vec<f64> = nbrs
                .iter()
                .map(|&j| {
                    let mut concat = transformed[i].clone();
                    concat.extend_from_slice(&transformed[j]);
                    let e = dot(&self.attention_weights, &concat);
                    self.leaky_relu(e)
                })
                .collect();

            let alphas = softmax_vec(&scores);

            for (k, &j) in nbrs.iter().enumerate() {
                let alpha = alphas[k];
                for d in 0..self.out_dim {
                    aggregated[i][d] += alpha * transformed[j][d];
                }
            }
        }

        Ok(aggregated)
    }

    fn update(
        &self,
        aggregated: &[Vec<f64>],
        _node_features: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>> {
        if !self.use_activation {
            return Ok(aggregated.to_vec());
        }
        let result: Vec<Vec<f64>> = aggregated
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&x| if x >= 0.0 { x } else { x.exp() - 1.0 })
                    .collect()
            })
            .collect();
        Ok(result)
    }
}

// ============================================================================
// NodeEmbedding: high-level embedding container (legacy)
// ============================================================================

/// Container for node feature embeddings
#[derive(Debug, Clone)]
pub struct NodeEmbeddings {
    /// Node index to name mapping (optional)
    pub node_names: Vec<String>,
    /// Feature matrix (n_nodes × embedding_dim)
    pub embeddings: Vec<Vec<f64>>,
    /// Embedding dimensionality
    pub dim: usize,
}

impl NodeEmbeddings {
    /// Create embeddings from raw feature matrix
    pub fn new(embeddings: Vec<Vec<f64>>) -> Result<Self> {
        let dim = validate_features(&embeddings)?;
        let n = embeddings.len();
        Ok(NodeEmbeddings {
            node_names: (0..n).map(|i| i.to_string()).collect(),
            embeddings,
            dim,
        })
    }

    /// Create random initial embeddings
    pub fn random(n_nodes: usize, dim: usize) -> Self {
        let mut rng = scirs2_core::random::rng();
        let embeddings: Vec<Vec<f64>> = (0..n_nodes)
            .map(|_| (0..dim).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect())
            .collect();
        NodeEmbeddings {
            node_names: (0..n_nodes).map(|i| i.to_string()).collect(),
            embeddings,
            dim,
        }
    }

    /// Create one-hot embeddings
    pub fn one_hot(n_nodes: usize) -> Self {
        let embeddings: Vec<Vec<f64>> = (0..n_nodes)
            .map(|i| {
                let mut v = vec![0.0f64; n_nodes];
                v[i] = 1.0;
                v
            })
            .collect();
        NodeEmbeddings {
            node_names: (0..n_nodes).map(|i| i.to_string()).collect(),
            embeddings,
            dim: n_nodes,
        }
    }

    /// Get the number of nodes
    pub fn n_nodes(&self) -> usize {
        self.embeddings.len()
    }

    /// Get embedding for node i
    pub fn get(&self, i: usize) -> Option<&Vec<f64>> {
        self.embeddings.get(i)
    }

    /// Apply a GNN layer to these embeddings
    pub fn apply_layer<L: MessagePassingLayer>(
        &self,
        layer: &L,
        adjacency: &[(usize, usize, f64)],
    ) -> Result<NodeEmbeddings> {
        let new_embeddings = layer.forward(&self.embeddings, adjacency)?;
        let dim = validate_features(&new_embeddings)?;
        Ok(NodeEmbeddings {
            node_names: self.node_names.clone(),
            embeddings: new_embeddings,
            dim,
        })
    }
}

/// Build a GNN pipeline and apply it to a graph
pub fn run_gnn_pipeline<N, E, Ix, L>(
    graph: &Graph<N, E, Ix>,
    initial_features: Option<NodeEmbeddings>,
    layers: &[L],
) -> Result<NodeEmbeddings>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Clone + Into<f64>,
    Ix: petgraph::graph::IndexType,
    L: MessagePassingLayer,
{
    let (_, adjacency) = graph_to_adjacency(graph);
    let n = graph.nodes().len();

    let mut embeddings = match initial_features {
        Some(e) => e,
        None => NodeEmbeddings::one_hot(n),
    };

    for layer in layers {
        embeddings = embeddings.apply_layer(layer, &adjacency)?;
    }

    Ok(embeddings)
}

// ============================================================================
// Tests (legacy API)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::Graph;

    fn make_triangle_graph() -> (Graph<usize, f64>, Vec<(usize, usize, f64)>) {
        let mut g: Graph<usize, f64> = Graph::new();
        let _ = g.add_edge(0, 1, 1.0);
        let _ = g.add_edge(1, 2, 1.0);
        let _ = g.add_edge(0, 2, 1.0);
        let (_, adj) = graph_to_adjacency(&g);
        (g, adj)
    }

    fn make_features(n: usize, dim: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| (0..dim).map(|j| (i * dim + j) as f64 / 10.0).collect())
            .collect()
    }

    #[test]
    fn test_gcn_layer_output_shape() {
        let (_, adj) = make_triangle_graph();
        let features = make_features(3, 4);
        let layer = GCNLayer::new(4, 8);
        let out = layer.forward(&features, &adj).expect("GCN forward failed");
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_graphsage_layer_output_shape() {
        let (_, adj) = make_triangle_graph();
        let features = make_features(3, 4);
        let layer = GraphSAGELayer::new(4, 6);
        let out = layer.forward(&features, &adj).expect("SAGE forward failed");
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), 6);
    }

    #[test]
    fn test_gat_layer_output_shape() {
        let (_, adj) = make_triangle_graph();
        let features = make_features(3, 4);
        let layer = GATLayer::new(4, 8);
        let out = layer.forward(&features, &adj).expect("GAT forward failed");
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_node_embeddings_one_hot() {
        let emb = NodeEmbeddings::one_hot(3);
        assert_eq!(emb.n_nodes(), 3);
        assert_eq!(emb.dim, 3);
        let row0 = emb.get(0).expect("No embedding for node 0");
        assert!((row0[0] - 1.0).abs() < 1e-10);
        assert!((row0[1]).abs() < 1e-10);
    }

    #[test]
    fn test_run_gnn_pipeline() {
        let mut g: Graph<usize, f64> = Graph::new();
        let _ = g.add_edge(0, 1, 1.0);
        let _ = g.add_edge(1, 2, 1.0);
        let _ = g.add_edge(2, 3, 1.0);
        let layers = vec![GCNLayer::new(4, 4), GCNLayer::new(4, 4)];
        let features = NodeEmbeddings::new(make_features(4, 4)).expect("Features");
        let result = run_gnn_pipeline(&g, Some(features), &layers).expect("Pipeline");
        assert_eq!(result.n_nodes(), 4);
    }
}
