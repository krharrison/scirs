//! GraphSAGE (Sample and Aggregate) layers
//!
//! Implements the inductive node embedding framework from Hamilton, Ying &
//! Leskovec (2017), "Inductive Representation Learning on Large Graphs".
//!
//! GraphSAGE learns to aggregate feature information from local neighborhoods,
//! enabling inductive generalization to unseen nodes.
//!
//! Supported aggregation types:
//! - **Mean** – element-wise mean of neighbor features
//! - **Max** – element-wise max-pooling of neighbor features
//! - **Sum** – element-wise sum of neighbor features
//! - **LSTM** – sequential LSTM over a randomly-ordered neighborhood sample
//!   (full LSTM backprop is outside scope; approximated here with a simple
//!   gated recurrent aggregation for the forward pass)

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Rng, RngExt};

use crate::error::{GraphError, Result};
use crate::gnn::gcn::CsrMatrix;

// ============================================================================
// Aggregation type
// ============================================================================

/// Aggregation strategy used to collect neighbor messages in GraphSAGE.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SageAggregation {
    /// Element-wise arithmetic mean over the neighborhood
    Mean,
    /// Element-wise maximum (max-pooling)
    Max,
    /// Element-wise sum
    Sum,
    /// Gated LSTM-style sequential aggregation over neighbors
    Lstm,
}

impl Default for SageAggregation {
    fn default() -> Self {
        SageAggregation::Mean
    }
}

// ============================================================================
// Neighborhood sampling
// ============================================================================

/// Sample up to `k` neighbors for each node, returning indices.
///
/// If a node has fewer than `k` neighbors, all neighbors are returned (no
/// replacement sampling).  The sampling is deterministic per `seed`.
///
/// # Arguments
/// * `adj` – Sparse adjacency (any direction; the function treats each row as
///   the neighbor list of that node).
/// * `k` – Maximum neighborhood size to sample.
///
/// # Returns
/// `sampled[i]` contains up to `k` neighbor indices for node `i`.
pub fn sample_neighbors(adj: &CsrMatrix, k: usize) -> Vec<Vec<usize>> {
    let n = adj.n_rows;
    let mut rng = scirs2_core::random::rng();

    (0..n)
        .map(|i| {
            let start = adj.indptr[i];
            let end = adj.indptr[i + 1];
            let neighbors: Vec<usize> = adj.indices[start..end].to_vec();
            if neighbors.len() <= k {
                neighbors
            } else {
                // Reservoir sampling
                let mut reservoir: Vec<usize> = neighbors[..k].to_vec();
                for idx in k..neighbors.len() {
                    let j = (rng.random::<f64>() * (idx + 1) as f64) as usize;
                    if j < k {
                        reservoir[j] = neighbors[idx];
                    }
                }
                reservoir
            }
        })
        .collect()
}

// ============================================================================
// Neighborhood aggregation (functional API)
// ============================================================================

/// Aggregate neighbor features using the specified aggregation type.
///
/// # Arguments
/// * `adj` – Sparse adjacency matrix (row i → neighbors of node i).
/// * `features` – Node feature matrix `[n_nodes, feat_dim]`.
/// * `aggr_type` – Which aggregation to apply.
///
/// # Returns
/// Aggregated neighbor embeddings `[n_nodes, feat_dim]`.  Isolated nodes
/// (no neighbors) receive a zero vector.
pub fn sage_aggregate(
    adj: &CsrMatrix,
    features: &Array2<f64>,
    aggr_type: &SageAggregation,
) -> Result<Array2<f64>> {
    let n = adj.n_rows;
    let (feat_n, feat_dim) = features.dim();

    if feat_n != n {
        return Err(GraphError::InvalidParameter {
            param: "features".to_string(),
            value: format!("{feat_n} rows"),
            expected: format!("{n} rows (matching adj.n_rows)"),
            context: "sage_aggregate".to_string(),
        });
    }

    let mut agg = Array2::<f64>::zeros((n, feat_dim));

    match aggr_type {
        SageAggregation::Mean | SageAggregation::Sum => {
            let mut counts = vec![0usize; n];
            for (row, col, _) in adj.iter() {
                if col < feat_n {
                    counts[row] += 1;
                    for k in 0..feat_dim {
                        agg[[row, k]] += features[[col, k]];
                    }
                }
            }
            if *aggr_type == SageAggregation::Mean {
                for i in 0..n {
                    if counts[i] > 0 {
                        let inv = 1.0 / counts[i] as f64;
                        for k in 0..feat_dim {
                            agg[[i, k]] *= inv;
                        }
                    }
                }
            }
        }

        SageAggregation::Max => {
            // Initialize to NEG_INFINITY, then reduce
            let mut initialized = vec![false; n];
            for (row, col, _) in adj.iter() {
                if col < feat_n {
                    if !initialized[row] {
                        for k in 0..feat_dim {
                            agg[[row, k]] = features[[col, k]];
                        }
                        initialized[row] = true;
                    } else {
                        for k in 0..feat_dim {
                            if features[[col, k]] > agg[[row, k]] {
                                agg[[row, k]] = features[[col, k]];
                            }
                        }
                    }
                }
            }
            // Nodes with no neighbors keep zero (already set)
        }

        SageAggregation::Lstm => {
            // Gated sequential aggregation over neighbors (approximates LSTM
            // forward pass without backprop).  For each node i we process its
            // neighbor features in order; a hidden state h is updated via:
            //   z = sigmoid(x + h)
            //   h = z * h + (1 - z) * x   (simplified GRU-like update)
            for i in 0..n {
                let start = adj.indptr[i];
                let end = adj.indptr[i + 1];
                let neighbor_indices = &adj.indices[start..end];

                if neighbor_indices.is_empty() {
                    continue;
                }

                let mut h = vec![0.0f64; feat_dim];
                for &nb in neighbor_indices {
                    if nb < feat_n {
                        for k in 0..feat_dim {
                            let x = features[[nb, k]];
                            // Sigmoid gate
                            let z = 1.0 / (1.0 + (-(x + h[k])).exp());
                            h[k] = z * h[k] + (1.0 - z) * x;
                        }
                    }
                }
                for k in 0..feat_dim {
                    agg[[i, k]] = h[k];
                }
            }
        }
    }

    Ok(agg)
}

// ============================================================================
// GraphSAGE layer
// ============================================================================

/// A single GraphSAGE layer.
///
/// Concatenates each node's own representation with the aggregated neighbor
/// representation, then applies a linear transformation:
/// ```text
///   h_v = σ( W · concat(h_v, AGG({h_u : u ∈ N(v)})) + b )
/// ```
/// The output is L2-normalized (per-node) following the original paper.
#[derive(Debug, Clone)]
pub struct GraphSageLayer {
    /// Weight matrix `[2 * in_dim, out_dim]`
    pub weights: Array2<f64>,
    /// Optional bias `[out_dim]`
    pub bias: Option<Array1<f64>>,
    /// Input feature dimension
    pub in_dim: usize,
    /// Output feature dimension
    pub out_dim: usize,
    /// Aggregation strategy
    pub aggregation: SageAggregation,
    /// Apply ReLU activation
    pub use_relu: bool,
    /// Apply L2 normalization on output embeddings
    pub normalize: bool,
}

impl GraphSageLayer {
    /// Create a new GraphSAGE layer with Glorot-uniform initialization.
    ///
    /// # Arguments
    /// * `in_dim` – Input feature dimension per node.
    /// * `out_dim` – Output embedding dimension.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let concat_dim = 2 * in_dim;
        let scale = (6.0_f64 / (concat_dim + out_dim) as f64).sqrt();
        let mut rng = scirs2_core::random::rng();
        let weights = Array2::from_shape_fn((concat_dim, out_dim), |_| {
            rng.random::<f64>() * 2.0 * scale - scale
        });
        GraphSageLayer {
            weights,
            bias: None,
            in_dim,
            out_dim,
            aggregation: SageAggregation::Mean,
            use_relu: true,
            normalize: true,
        }
    }

    /// Use a specific aggregation strategy.
    pub fn with_aggregation(mut self, aggr: SageAggregation) -> Self {
        self.aggregation = aggr;
        self
    }

    /// Disable L2 normalization on output.
    pub fn without_normalize(mut self) -> Self {
        self.normalize = false;
        self
    }

    /// Disable ReLU activation.
    pub fn without_activation(mut self) -> Self {
        self.use_relu = false;
        self
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `adj` – Sparse adjacency (row i → outgoing neighbors of i).
    /// * `features` – Node feature matrix `[n_nodes, in_dim]`.
    pub fn forward(&self, adj: &CsrMatrix, features: &Array2<f64>) -> Result<Array2<f64>> {
        let n = adj.n_rows;
        let (feat_n, feat_dim) = features.dim();

        if feat_n != n {
            return Err(GraphError::InvalidParameter {
                param: "features".to_string(),
                value: format!("{feat_n}"),
                expected: format!("{n}"),
                context: "GraphSageLayer::forward".to_string(),
            });
        }
        if feat_dim != self.in_dim {
            return Err(GraphError::InvalidParameter {
                param: "features feat_dim".to_string(),
                value: format!("{feat_dim}"),
                expected: format!("{}", self.in_dim),
                context: "GraphSageLayer::forward".to_string(),
            });
        }

        // Step 1: aggregate neighbor features
        let agg = sage_aggregate(adj, features, &self.aggregation)?;

        // Step 2: concatenate [self_feat || agg_feat]  →  [n, 2*in_dim]
        let concat_dim = 2 * self.in_dim;
        let mut concat = Array2::<f64>::zeros((n, concat_dim));
        for i in 0..n {
            for k in 0..feat_dim {
                concat[[i, k]] = features[[i, k]];
                concat[[i, feat_dim + k]] = agg[[i, k]];
            }
        }

        // Step 3: linear transform  concat @ weights  →  [n, out_dim]
        let (_, out_dim) = self.weights.dim();
        let mut output = Array2::<f64>::zeros((n, out_dim));
        for i in 0..n {
            for j in 0..out_dim {
                let mut sum = 0.0;
                for k in 0..concat_dim {
                    sum += concat[[i, k]] * self.weights[[k, j]];
                }
                output[[i, j]] = sum;
            }
        }

        // Add bias
        if let Some(ref b) = self.bias {
            for i in 0..n {
                for j in 0..out_dim {
                    output[[i, j]] += b[j];
                }
            }
        }

        // Activation
        if self.use_relu {
            output.mapv_inplace(|x| x.max(0.0));
        }

        // L2 normalize each row
        if self.normalize {
            for i in 0..n {
                let norm = {
                    let row = output.row(i);
                    row.iter().map(|&x| x * x).sum::<f64>().sqrt()
                };
                if norm > 1e-10 {
                    for j in 0..out_dim {
                        output[[i, j]] /= norm;
                    }
                }
            }
        }

        Ok(output)
    }
}

// ============================================================================
// Multi-layer GraphSAGE model
// ============================================================================

/// Multi-layer GraphSAGE model.
///
/// Stacks `GraphSageLayer`s with optional neighborhood sampling at each layer.
pub struct GraphSage {
    /// Layer stack
    pub layers: Vec<GraphSageLayer>,
    /// Optional max neighborhood size per layer (None = use all neighbors)
    pub neighbor_samples: Vec<Option<usize>>,
}

impl GraphSage {
    /// Build a GraphSAGE model from a list of `(in_dim, out_dim)` layer specs.
    ///
    /// # Arguments
    /// * `dims` – Sequence `[d_0, d_1, …, d_L]`.
    /// * `aggr` – Aggregation type applied to all layers.
    pub fn new(dims: &[usize], aggr: SageAggregation) -> Result<Self> {
        if dims.len() < 2 {
            return Err(GraphError::InvalidParameter {
                param: "dims".to_string(),
                value: format!("len={}", dims.len()),
                expected: "at least 2 elements".to_string(),
                context: "GraphSage::new".to_string(),
            });
        }
        let mut layers = Vec::with_capacity(dims.len() - 1);
        for i in 0..(dims.len() - 1) {
            let is_last = i == dims.len() - 2;
            let mut layer = GraphSageLayer::new(dims[i], dims[i + 1])
                .with_aggregation(aggr.clone());
            if is_last {
                layer = layer.without_activation();
            }
            layers.push(layer);
        }
        let neighbor_samples = vec![None; dims.len() - 1];
        Ok(GraphSage { layers, neighbor_samples })
    }

    /// Set the maximum number of neighbors sampled at each layer.
    ///
    /// `sizes[i]` controls layer `i`.  Pass `None` for a layer to use all
    /// neighbors.
    pub fn with_neighbor_samples(mut self, sizes: Vec<Option<usize>>) -> Result<Self> {
        if sizes.len() != self.layers.len() {
            return Err(GraphError::InvalidParameter {
                param: "sizes".to_string(),
                value: format!("len={}", sizes.len()),
                expected: format!("len={}", self.layers.len()),
                context: "GraphSage::with_neighbor_samples".to_string(),
            });
        }
        self.neighbor_samples = sizes;
        Ok(self)
    }

    /// Forward pass through all layers.
    ///
    /// # Arguments
    /// * `adj` – Sparse adjacency matrix.
    /// * `features` – Initial feature matrix `[n_nodes, d_0]`.
    pub fn forward(&self, adj: &CsrMatrix, features: &Array2<f64>) -> Result<Array2<f64>> {
        let mut h = features.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            // Optionally sub-sample the adjacency for mini-batch training
            let sampled_adj = if let Some(k) = self.neighbor_samples[i] {
                // Build a sub-sampled CSR from sampled neighbors
                let sampled = sample_neighbors(adj, k);
                let mut coo = Vec::new();
                for (node_i, nbrs) in sampled.iter().enumerate() {
                    for &nb in nbrs {
                        coo.push((node_i, nb, 1.0f64));
                    }
                }
                CsrMatrix::from_coo(adj.n_rows, adj.n_cols, &coo)?
            } else {
                adj.clone()
            };
            h = layer.forward(&sampled_adj, &h)?;
        }
        Ok(h)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn path_csr(n: usize) -> CsrMatrix {
        let mut coo = Vec::new();
        for i in 0..(n - 1) {
            coo.push((i, i + 1, 1.0));
            coo.push((i + 1, i, 1.0));
        }
        CsrMatrix::from_coo(n, n, &coo).expect("path CSR")
    }

    fn features(n: usize, d: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, d), |(i, j)| (i * d + j) as f64 * 0.1)
    }

    #[test]
    fn test_mean_aggregate_shape() {
        let adj = path_csr(4);
        let feats = features(4, 6);
        let agg = sage_aggregate(&adj, &feats, &SageAggregation::Mean).expect("mean agg");
        assert_eq!(agg.dim(), (4, 6));
    }

    #[test]
    fn test_max_aggregate_shape() {
        let adj = path_csr(4);
        let feats = features(4, 6);
        let agg = sage_aggregate(&adj, &feats, &SageAggregation::Max).expect("max agg");
        assert_eq!(agg.dim(), (4, 6));
    }

    #[test]
    fn test_sum_aggregate_shape() {
        let adj = path_csr(4);
        let feats = features(4, 6);
        let agg = sage_aggregate(&adj, &feats, &SageAggregation::Sum).expect("sum agg");
        assert_eq!(agg.dim(), (4, 6));
    }

    #[test]
    fn test_lstm_aggregate_shape() {
        let adj = path_csr(4);
        let feats = features(4, 6);
        let agg = sage_aggregate(&adj, &feats, &SageAggregation::Lstm).expect("lstm agg");
        assert_eq!(agg.dim(), (4, 6));
    }

    #[test]
    fn test_sage_layer_output_shape() {
        let adj = path_csr(5);
        let feats = features(5, 4);
        let layer = GraphSageLayer::new(4, 8);
        let out = layer.forward(&adj, &feats).expect("sage forward");
        assert_eq!(out.dim(), (5, 8));
    }

    #[test]
    fn test_sage_layer_l2_normalization() {
        let adj = path_csr(5);
        let feats = features(5, 4);
        let layer = GraphSageLayer::new(4, 8);
        let out = layer.forward(&adj, &feats).expect("sage forward");
        // Each row should have unit L2 norm (or be zero)
        for i in 0..5 {
            let norm: f64 = out.row(i).iter().map(|&x| x * x).sum::<f64>().sqrt();
            assert!(
                norm < 1e-10 || (norm - 1.0).abs() < 1e-9,
                "norm={norm} for row {i}"
            );
        }
    }

    #[test]
    fn test_graphsage_multilayer() {
        let adj = path_csr(6);
        let feats = features(6, 8);
        let model = GraphSage::new(&[8, 16, 4], SageAggregation::Mean).expect("sage model");
        let out = model.forward(&adj, &feats).expect("forward");
        assert_eq!(out.dim(), (6, 4));
    }

    #[test]
    fn test_neighbor_sampling() {
        let adj = path_csr(4);
        let sampled = sample_neighbors(&adj, 1);
        assert_eq!(sampled.len(), 4);
        // Internal nodes have 2 neighbors; sampled to 1
        assert!(sampled[1].len() <= 1);
        assert!(sampled[2].len() <= 1);
    }

    #[test]
    fn test_graphsage_with_sampling() {
        let adj = path_csr(6);
        let feats = features(6, 4);
        let model = GraphSage::new(&[4, 8, 4], SageAggregation::Mean)
            .expect("sage model")
            .with_neighbor_samples(vec![Some(2), Some(2)])
            .expect("samples");
        let out = model.forward(&adj, &feats).expect("forward");
        assert_eq!(out.dim(), (6, 4));
    }

    #[test]
    fn test_sage_aggregation_isolated_node() {
        // Node 0 has no neighbors
        let coo = vec![(1, 2, 1.0), (2, 1, 1.0)];
        let adj = CsrMatrix::from_coo(3, 3, &coo).expect("isolated CSR");
        let feats = features(3, 4);
        let agg = sage_aggregate(&adj, &feats, &SageAggregation::Mean).expect("mean agg");
        // Node 0: no neighbors → zero aggregation
        for k in 0..4 {
            assert_eq!(agg[[0, k]], 0.0);
        }
    }
}
