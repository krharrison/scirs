//! Graph Pooling Layers
//!
//! This module provides global and hierarchical graph pooling operations:
//!
//! - [`global_add_pool`]  — sum node features within each graph in a batch
//! - [`global_mean_pool`] — mean node features within each graph in a batch
//! - [`global_max_pool`]  — element-wise max over nodes within each graph
//! - [`DiffPool`]         — differentiable pooling (Ying et al., 2018) that
//!   learns a soft cluster assignment matrix `S` via a GNN and uses it to
//!   hierarchically coarsen the graph:
//!
//! ```text
//! S          = softmax( GNN_pool(X, A) )   [N, K]
//! X_pooled   = S^T X                        [K, F]
//! A_pooled   = S^T A S                      [K, K]
//! ```
//!
//! All functions operate on dense node-feature matrices (`N × F`) and dense
//! adjacency matrices (`N × N`).

use crate::error::{NeuralError, Result};
use crate::layers::graph_conv::{GraphActivation, GraphConvLayer};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, Array1, Array2, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

// ──────────────────────────────────────────────────────────────────────────────
// Global pooling helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Global add (sum) pooling over a batched set of graphs.
///
/// For each unique graph id `g` in `batch_assignments`, sum all node feature
/// rows that belong to `g`.
///
/// # Arguments
/// * `node_features`    — `[N, F]` node feature matrix (all graphs concatenated)
/// * `batch_assignments` — length-`N` vector mapping each node to a graph id
///   (graph ids must be dense integers starting at 0)
///
/// # Returns
/// `[num_graphs, F]` matrix, one row per graph.
///
/// # Examples
/// ```rust
/// use scirs2_neural::layers::graph_pooling::global_add_pool;
/// use scirs2_core::ndarray::Array2;
///
/// // 4 nodes across 2 graphs (nodes 0,1 → graph 0; nodes 2,3 → graph 1)
/// let features = Array2::<f64>::from_elem((4, 3), 1.0);
/// let batch = vec![0usize, 0, 1, 1];
/// let pooled = global_add_pool(&features, &batch).expect("pool failed");
/// assert_eq!(pooled.shape(), &[2, 3]);
/// // Each graph has 2 nodes each with value 1 → row sum = 2
/// assert!((pooled[[0, 0]] - 2.0).abs() < 1e-10);
/// ```
pub fn global_add_pool<F>(
    node_features: &Array2<F>,
    batch_assignments: &[usize],
) -> Result<Array2<F>>
where
    F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static,
{
    let n = node_features.nrows();
    let feat_dim = node_features.ncols();

    if batch_assignments.len() != n {
        return Err(NeuralError::InvalidArgument(format!(
            "batch_assignments length ({}) must equal number of nodes ({})",
            batch_assignments.len(),
            n
        )));
    }

    let num_graphs = batch_assignments
        .iter()
        .copied()
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    if num_graphs == 0 {
        return Ok(Array2::zeros((0, feat_dim)));
    }

    let mut out = Array2::<F>::zeros((num_graphs, feat_dim));
    for (node_idx, &graph_id) in batch_assignments.iter().enumerate() {
        if graph_id >= num_graphs {
            return Err(NeuralError::InvalidArgument(format!(
                "batch_assignments[{node_idx}] = {graph_id} is >= num_graphs ({num_graphs})"
            )));
        }
        for f in 0..feat_dim {
            out[[graph_id, f]] += node_features[[node_idx, f]];
        }
    }

    Ok(out)
}

/// Global mean pooling over a batched set of graphs.
///
/// For each unique graph id `g` in `batch_assignments`, average all node
/// feature rows that belong to `g`.
///
/// # Arguments
/// * `node_features`    — `[N, F]` node feature matrix
/// * `batch_assignments` — length-`N` vector mapping each node to a graph id
///
/// # Returns
/// `[num_graphs, F]` matrix with mean-pooled node features.
///
/// # Examples
/// ```rust
/// use scirs2_neural::layers::graph_pooling::global_mean_pool;
/// use scirs2_core::ndarray::Array2;
///
/// let features = Array2::<f64>::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
///     .expect("shape");
/// let batch = vec![0usize, 0, 1, 1];
/// let pooled = global_mean_pool(&features, &batch).expect("pool failed");
/// assert_eq!(pooled.shape(), &[2, 2]);
/// assert!((pooled[[0, 0]] - 2.0).abs() < 1e-10);  // mean of 1.0 and 3.0
/// ```
pub fn global_mean_pool<F>(
    node_features: &Array2<F>,
    batch_assignments: &[usize],
) -> Result<Array2<F>>
where
    F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static,
{
    let n = node_features.nrows();
    let feat_dim = node_features.ncols();

    if batch_assignments.len() != n {
        return Err(NeuralError::InvalidArgument(format!(
            "batch_assignments length ({}) must equal number of nodes ({})",
            batch_assignments.len(),
            n
        )));
    }

    let num_graphs = batch_assignments
        .iter()
        .copied()
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    if num_graphs == 0 {
        return Ok(Array2::zeros((0, feat_dim)));
    }

    let mut out = Array2::<F>::zeros((num_graphs, feat_dim));
    let mut counts = vec![0usize; num_graphs];

    for (node_idx, &graph_id) in batch_assignments.iter().enumerate() {
        if graph_id >= num_graphs {
            return Err(NeuralError::InvalidArgument(format!(
                "batch_assignments[{node_idx}] = {graph_id} is >= num_graphs ({num_graphs})"
            )));
        }
        for f in 0..feat_dim {
            out[[graph_id, f]] += node_features[[node_idx, f]];
        }
        counts[graph_id] += 1;
    }

    // Normalise by node count
    for g in 0..num_graphs {
        if counts[g] > 0 {
            let inv_count = F::from(1.0 / counts[g] as f64).unwrap_or(F::one());
            for f in 0..feat_dim {
                out[[g, f]] *= inv_count;
            }
        }
    }

    Ok(out)
}

/// Global max pooling over a batched set of graphs.
///
/// For each unique graph id `g` in `batch_assignments`, take the element-wise
/// maximum of all node feature rows that belong to `g`.
///
/// # Arguments
/// * `node_features`    — `[N, F]` node feature matrix
/// * `batch_assignments` — length-`N` vector mapping each node to a graph id
///
/// # Returns
/// `[num_graphs, F]` matrix with max-pooled node features.
///
/// # Examples
/// ```rust
/// use scirs2_neural::layers::graph_pooling::global_max_pool;
/// use scirs2_core::ndarray::Array2;
///
/// let features = Array2::<f64>::from_shape_vec((4, 2), vec![1.0, 5.0, 3.0, 2.0, 4.0, 1.0, 2.0, 8.0])
///     .expect("shape");
/// let batch = vec![0usize, 0, 1, 1];
/// let pooled = global_max_pool(&features, &batch).expect("pool failed");
/// assert_eq!(pooled.shape(), &[2, 2]);
/// assert!((pooled[[0, 0]] - 3.0).abs() < 1e-10);  // max(1.0, 3.0) = 3.0
/// assert!((pooled[[0, 1]] - 5.0).abs() < 1e-10);  // max(5.0, 2.0) = 5.0
/// ```
pub fn global_max_pool<F>(
    node_features: &Array2<F>,
    batch_assignments: &[usize],
) -> Result<Array2<F>>
where
    F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static,
{
    let n = node_features.nrows();
    let feat_dim = node_features.ncols();

    if batch_assignments.len() != n {
        return Err(NeuralError::InvalidArgument(format!(
            "batch_assignments length ({}) must equal number of nodes ({})",
            batch_assignments.len(),
            n
        )));
    }

    let num_graphs = batch_assignments
        .iter()
        .copied()
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    if num_graphs == 0 {
        return Ok(Array2::zeros((0, feat_dim)));
    }

    let mut out = Array2::from_elem((num_graphs, feat_dim), F::neg_infinity());
    let mut has_node = vec![false; num_graphs];

    for (node_idx, &graph_id) in batch_assignments.iter().enumerate() {
        if graph_id >= num_graphs {
            return Err(NeuralError::InvalidArgument(format!(
                "batch_assignments[{node_idx}] = {graph_id} is >= num_graphs ({num_graphs})"
            )));
        }
        has_node[graph_id] = true;
        for f in 0..feat_dim {
            let v = node_features[[node_idx, f]];
            if v > out[[graph_id, f]] {
                out[[graph_id, f]] = v;
            }
        }
    }

    // Replace -inf with 0 for empty graphs (should not normally happen)
    for g in 0..num_graphs {
        if !has_node[g] {
            for f in 0..feat_dim {
                out[[g, f]] = F::zero();
            }
        }
    }

    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────────────
// DiffPool
// ──────────────────────────────────────────────────────────────────────────────

/// Differentiable Pooling (DiffPool) layer — Ying et al. (2018)
///
/// DiffPool learns a soft cluster-assignment matrix `S ∈ ℝ^{N×K}` (where `K`
/// is the target number of clusters) using a GNN, and uses it to hierarchically
/// coarsen the graph:
///
/// ```text
/// S          = softmax_rows( GNN_pool(X, A) )       [N, K]
/// X_pooled   = S^T X                                 [K, F]
/// A_pooled   = S^T A S                               [K, K]
/// ```
///
/// An auxiliary GNN (`GNN_embed`) simultaneously refines the node features
/// before pooling:
///
/// ```text
/// Z          = GNN_embed(X, A)                       [N, F_out]
/// X_pooled   = S^T Z                                 [K, F_out]
/// ```
///
/// Two auxiliary losses are accumulated during the forward pass to encourage
/// the assignment matrix to be sparse and the pooled graph to preserve the
/// original cluster structure:
/// * `link_prediction_loss`: `‖ A - S S^T ‖_F²`
/// * `entropy_loss`: `Σ_{i,k} −s_{ik} ln(s_{ik} + ε)`
///
/// # Shape
/// * `node_feat` — `[N, in_features]`
/// * `adj`       — `[N, N]`
/// * returns     — `([K, out_features], [K, K])` — (pooled features, pooled adjacency)
///
/// # Examples
/// ```rust
/// use scirs2_neural::layers::graph_pooling::DiffPool;
/// use scirs2_core::ndarray::Array2;
/// use scirs2_core::random::rng;
///
/// let mut rng = rng();
/// let layer = DiffPool::<f64>::new(8, 8, 4, &mut rng).expect("DiffPool creation");
///
/// let node_feat = Array2::<f64>::from_elem((10, 8), 0.1);
/// let adj = Array2::<f64>::eye(10);
/// let (x_pool, a_pool) = layer.forward_graph(&node_feat, &adj)
///     .expect("Forward pass failed");
/// assert_eq!(x_pool.shape(), &[4, 8]);
/// assert_eq!(a_pool.shape(), &[4, 4]);
/// ```
#[derive(Debug, Clone)]
pub struct DiffPool<F: Float + Debug + Send + Sync + NumAssign> {
    /// GNN that produces the soft assignment matrix `S`: in_features → n_clusters
    gnn_pool: GraphConvLayer<F>,
    /// GNN that refines node features: in_features → out_features
    gnn_embed: GraphConvLayer<F>,
    /// Number of input node features
    in_features: usize,
    /// Number of output node features after the embed GNN
    out_features: usize,
    /// Target number of clusters
    n_clusters: usize,
    /// Accumulated link-prediction loss from the last forward pass
    last_link_loss: Arc<RwLock<F>>,
    /// Accumulated entropy regularisation loss from the last forward pass
    last_entropy_loss: Arc<RwLock<F>>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> DiffPool<F> {
    /// Create a new DiffPool layer.
    ///
    /// # Arguments
    /// * `in_features`  — input node feature dimension
    /// * `out_features` — output node feature dimension (embed GNN output)
    /// * `n_clusters`   — number of coarsened nodes in the pooled graph
    /// * `rng`          — random number generator
    pub fn new<R: scirs2_core::random::Rng>(
        in_features: usize,
        out_features: usize,
        n_clusters: usize,
        rng: &mut R,
    ) -> Result<Self> {
        if in_features == 0 || out_features == 0 || n_clusters == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "in_features, out_features, and n_clusters must all be > 0".to_string(),
            ));
        }

        let gnn_pool = GraphConvLayer::new(
            in_features,
            n_clusters,
            true,
            GraphActivation::None, // softmax applied manually
            rng,
        )?;

        let gnn_embed = GraphConvLayer::new(
            in_features,
            out_features,
            true,
            GraphActivation::ReLU,
            rng,
        )?;

        Ok(Self {
            gnn_pool,
            gnn_embed,
            in_features,
            out_features,
            n_clusters,
            last_link_loss: Arc::new(RwLock::new(F::zero())),
            last_entropy_loss: Arc::new(RwLock::new(F::zero())),
        })
    }

    /// Row-wise softmax of a matrix.
    fn row_softmax(x: &Array2<F>) -> Array2<F> {
        let (n, k) = (x.nrows(), x.ncols());
        let mut out = Array2::<F>::zeros((n, k));
        for i in 0..n {
            let max_val = (0..k)
                .map(|j| x[[i, j]])
                .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });
            let exps: Vec<F> = (0..k).map(|j| (x[[i, j]] - max_val).exp()).collect();
            let sum: F = exps.iter().copied().fold(F::zero(), |a, b| a + b);
            let inv_sum = if sum > F::zero() {
                F::one() / sum
            } else {
                F::one()
            };
            for j in 0..k {
                out[[i, j]] = exps[j] * inv_sum;
            }
        }
        out
    }

    /// Compute `S^T M` — left-multiply `M` by the transposed assignment matrix.
    ///
    /// `s` is `[N, K]`, `m` is `[N, F]` → returns `[K, F]`
    fn s_transpose_times(s: &Array2<F>, m: &Array2<F>) -> Array2<F> {
        let (n, k) = (s.nrows(), s.ncols());
        let f = m.ncols();
        let mut out = Array2::<F>::zeros((k, f));
        for cluster in 0..k {
            for node in 0..n {
                let s_val = s[[node, cluster]];
                if s_val == F::zero() {
                    continue;
                }
                for feat in 0..f {
                    out[[cluster, feat]] += s_val * m[[node, feat]];
                }
            }
        }
        out
    }

    /// Compute `S^T A S` — the coarsened adjacency matrix.
    ///
    /// `s` is `[N, K]`, `a` is `[N, N]` → returns `[K, K]`
    fn coarsen_adjacency(s: &Array2<F>, a: &Array2<F>) -> Array2<F> {
        let (n, k) = (s.nrows(), s.ncols());
        // First compute tmp = A @ S: [N, K]
        let mut tmp = Array2::<F>::zeros((n, k));
        for i in 0..n {
            for j in 0..n {
                let a_ij = a[[i, j]];
                if a_ij == F::zero() {
                    continue;
                }
                for c in 0..k {
                    tmp[[i, c]] += a_ij * s[[j, c]];
                }
            }
        }
        // Then compute S^T @ tmp = S^T A S: [K, K]
        let mut out = Array2::<F>::zeros((k, k));
        for c1 in 0..k {
            for i in 0..n {
                let s_val = s[[i, c1]];
                if s_val == F::zero() {
                    continue;
                }
                for c2 in 0..k {
                    out[[c1, c2]] += s_val * tmp[[i, c2]];
                }
            }
        }
        out
    }

    /// Compute the link-prediction auxiliary loss: `‖ A − S S^T ‖_F² / (N²)`
    fn link_prediction_loss(s: &Array2<F>, a: &Array2<F>) -> F {
        let n = s.nrows();
        let k = s.ncols();
        // Compute S S^T: [N, N]
        let mut ss_t = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut dot = F::zero();
                for c in 0..k {
                    dot += s[[i, c]] * s[[j, c]];
                }
                ss_t[[i, j]] = dot;
            }
        }
        // Frobenius norm squared of (A - S S^T)
        let mut loss = F::zero();
        for i in 0..n {
            for j in 0..n {
                let diff = a[[i, j]] - ss_t[[i, j]];
                loss += diff * diff;
            }
        }
        let n2 = F::from(n * n).unwrap_or(F::one());
        if n2 > F::zero() {
            loss / n2
        } else {
            loss
        }
    }

    /// Compute the entropy regularisation loss: `(1/N) Σ_{i,k} −s_{ik} ln(s_{ik} + ε)`
    fn entropy_loss(s: &Array2<F>) -> F {
        let n = s.nrows();
        let k = s.ncols();
        let eps = F::from(1e-10_f64).unwrap_or_else(F::zero);
        let mut loss = F::zero();
        for i in 0..n {
            for c in 0..k {
                let p = s[[i, c]];
                loss -= p * (p + eps).ln();
            }
        }
        let n_f = F::from(n).unwrap_or(F::one());
        if n_f > F::zero() {
            loss / n_f
        } else {
            loss
        }
    }

    /// Forward pass of DiffPool.
    ///
    /// # Arguments
    /// * `node_feat` — `[N, in_features]` input node features
    /// * `adj`       — `[N, N]` adjacency matrix
    ///
    /// # Returns
    /// `([K, out_features], [K, K])` — (pooled node features, pooled adjacency)
    ///
    /// Call [`link_prediction_loss_value`] and [`entropy_loss_value`] after
    /// the forward pass to retrieve the auxiliary losses.
    pub fn forward_graph(
        &self,
        node_feat: &Array2<F>,
        adj: &Array2<F>,
    ) -> Result<(Array2<F>, Array2<F>)> {
        let n = node_feat.nrows();
        if node_feat.ncols() != self.in_features {
            return Err(NeuralError::DimensionMismatch(format!(
                "Expected in_features={}, got {}",
                self.in_features,
                node_feat.ncols()
            )));
        }
        if adj.nrows() != n || adj.ncols() != n {
            return Err(NeuralError::InvalidArgument(format!(
                "Adjacency must be {}×{} but got {}×{}",
                n,
                n,
                adj.nrows(),
                adj.ncols()
            )));
        }
        if self.n_clusters > n {
            return Err(NeuralError::InvalidArgument(format!(
                "n_clusters ({}) must be ≤ n ({})",
                self.n_clusters, n
            )));
        }

        // Assignment GNN: [N, n_clusters]
        let s_logits = self.gnn_pool.forward_graph(node_feat, adj)?;
        // Apply row-wise softmax to get a proper stochastic assignment
        let s = Self::row_softmax(&s_logits);

        // Embed GNN: [N, out_features]
        let z = self.gnn_embed.forward_graph(node_feat, adj)?;

        // Coarsen: X_pooled = S^T Z  [K, out_features]
        let x_pooled = Self::s_transpose_times(&s, &z);

        // Coarsen adjacency: A_pooled = S^T A S  [K, K]
        let a_pooled = Self::coarsen_adjacency(&s, adj);

        // Compute and store auxiliary losses
        let lp_loss = Self::link_prediction_loss(&s, adj);
        let ent_loss = Self::entropy_loss(&s);

        {
            let mut ll = self
                .last_link_loss
                .write()
                .map_err(|e| NeuralError::ComputationError(format!("RwLock poisoned: {e}")))?;
            *ll = lp_loss;
        }
        {
            let mut el = self
                .last_entropy_loss
                .write()
                .map_err(|e| NeuralError::ComputationError(format!("RwLock poisoned: {e}")))?;
            *el = ent_loss;
        }

        Ok((x_pooled, a_pooled))
    }

    /// Retrieve the link-prediction auxiliary loss from the most recent forward pass.
    pub fn link_prediction_loss_value(&self) -> Result<F> {
        self.last_link_loss
            .read()
            .map(|v| *v)
            .map_err(|e| NeuralError::ComputationError(format!("RwLock poisoned: {e}")))
    }

    /// Retrieve the entropy regularisation loss from the most recent forward pass.
    pub fn entropy_loss_value(&self) -> Result<F> {
        self.last_entropy_loss
            .read()
            .map(|v| *v)
            .map_err(|e| NeuralError::ComputationError(format!("RwLock poisoned: {e}")))
    }

    /// Number of trainable parameters (sum of both GNNs)
    pub fn num_parameters(&self) -> usize {
        self.gnn_pool.num_parameters() + self.gnn_embed.num_parameters()
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for DiffPool<F>
{
    /// Generic [`Layer`] forward: uses identity adjacency (self-loops only).
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let f2 = input
            .clone()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|e| NeuralError::DimensionMismatch(format!("Expected 2D input: {e}")))?;
        let n = f2.nrows();
        if self.n_clusters > n {
            return Err(NeuralError::InvalidArgument(format!(
                "n_clusters ({}) must be ≤ n ({})",
                self.n_clusters, n
            )));
        }
        let adj = Array2::<F>::eye(n);
        self.forward_graph(&f2, &adj)
            .map(|(x_pool, _)| x_pool.into_dyn())
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad.clone())
    }

    fn update(&mut self, lr: F) -> Result<()> {
        self.gnn_pool.update(lr)?;
        self.gnn_embed.update(lr)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn layer_type(&self) -> &str {
        "DiffPool"
    }
    fn parameter_count(&self) -> usize {
        self.num_parameters()
    }
    fn layer_description(&self) -> String {
        format!(
            "type:DiffPool in:{} out:{} clusters:{}",
            self.in_features, self.out_features, self.n_clusters
        )
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut v = self.gnn_pool.params();
        v.extend(self.gnn_embed.params());
        v
    }
}

// Explicit Send + Sync implementations
unsafe impl<F: Float + Debug + Send + Sync + NumAssign> Send for DiffPool<F> {}
unsafe impl<F: Float + Debug + Send + Sync + NumAssign> Sync for DiffPool<F> {}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use scirs2_core::random::rng;

    // ── global_add_pool ───────────────────────────────────────────────────────

    #[test]
    fn test_global_add_pool_shape() {
        let features = Array2::<f64>::from_elem((6, 4), 1.0);
        let batch = vec![0usize, 0, 0, 1, 1, 1];
        let pooled = global_add_pool(&features, &batch).expect("pool failed");
        assert_eq!(pooled.shape(), &[2, 4]);
    }

    #[test]
    fn test_global_add_pool_values() {
        let features = Array2::<f64>::from_elem((4, 3), 2.0);
        let batch = vec![0usize, 0, 1, 1];
        let pooled = global_add_pool(&features, &batch).expect("pool failed");
        // Each graph has 2 nodes with value 2 → sum = 4
        assert!((pooled[[0, 0]] - 4.0).abs() < 1e-10);
        assert!((pooled[[1, 2]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_global_add_pool_single_graph() {
        let features = Array2::<f64>::from_elem((5, 2), 1.0);
        let batch = vec![0usize; 5];
        let pooled = global_add_pool(&features, &batch).expect("pool failed");
        assert_eq!(pooled.shape(), &[1, 2]);
        assert!((pooled[[0, 0]] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_global_add_pool_length_mismatch_error() {
        let features = Array2::<f64>::from_elem((4, 3), 1.0);
        let batch = vec![0usize, 1, 2]; // length 3 ≠ 4
        assert!(global_add_pool(&features, &batch).is_err());
    }

    // ── global_mean_pool ──────────────────────────────────────────────────────

    #[test]
    fn test_global_mean_pool_shape() {
        let features = Array2::<f64>::from_elem((6, 4), 1.0);
        let batch = vec![0usize, 0, 0, 1, 1, 1];
        let pooled = global_mean_pool(&features, &batch).expect("pool failed");
        assert_eq!(pooled.shape(), &[2, 4]);
    }

    #[test]
    fn test_global_mean_pool_values() {
        // rows: [1,2,3,4,5,6,7,8] split into 2 graphs, 2 nodes each
        let features = Array2::<f64>::from_shape_vec(
            (4, 2),
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .expect("shape");
        let batch = vec![0usize, 0, 1, 1];
        let pooled = global_mean_pool(&features, &batch).expect("pool failed");
        assert!((pooled[[0, 0]] - 2.0).abs() < 1e-10); // mean(1,3)=2
        assert!((pooled[[0, 1]] - 3.0).abs() < 1e-10); // mean(2,4)=3
        assert!((pooled[[1, 0]] - 6.0).abs() < 1e-10); // mean(5,7)=6
        assert!((pooled[[1, 1]] - 7.0).abs() < 1e-10); // mean(6,8)=7
    }

    // ── global_max_pool ───────────────────────────────────────────────────────

    #[test]
    fn test_global_max_pool_shape() {
        let features = Array2::<f64>::from_elem((5, 3), 1.0);
        let batch = vec![0usize, 0, 1, 1, 1];
        let pooled = global_max_pool(&features, &batch).expect("pool failed");
        assert_eq!(pooled.shape(), &[2, 3]);
    }

    #[test]
    fn test_global_max_pool_values() {
        let features = Array2::<f64>::from_shape_vec(
            (4, 2),
            vec![1.0_f64, 5.0, 3.0, 2.0, 4.0, 1.0, 2.0, 8.0],
        )
        .expect("shape");
        let batch = vec![0usize, 0, 1, 1];
        let pooled = global_max_pool(&features, &batch).expect("pool failed");
        assert!((pooled[[0, 0]] - 3.0).abs() < 1e-10); // max(1,3) = 3
        assert!((pooled[[0, 1]] - 5.0).abs() < 1e-10); // max(5,2) = 5
        assert!((pooled[[1, 0]] - 4.0).abs() < 1e-10); // max(4,2) = 4
        assert!((pooled[[1, 1]] - 8.0).abs() < 1e-10); // max(1,8) = 8
    }

    // ── DiffPool ──────────────────────────────────────────────────────────────

    #[test]
    fn test_diffpool_output_shapes() {
        let mut rng = rng();
        let layer = DiffPool::<f64>::new(8, 8, 4, &mut rng).expect("DiffPool creation");
        let node_feat = Array2::<f64>::from_elem((10, 8), 0.1);
        let adj = Array2::<f64>::eye(10);
        let (x_pool, a_pool) = layer
            .forward_graph(&node_feat, &adj)
            .expect("Forward pass failed");
        assert_eq!(x_pool.shape(), &[4, 8]);
        assert_eq!(a_pool.shape(), &[4, 4]);
    }

    #[test]
    fn test_diffpool_adjacency_is_finite() {
        let mut rng = rng();
        let layer = DiffPool::<f64>::new(4, 6, 3, &mut rng).expect("DiffPool creation");
        let node_feat = Array2::<f64>::from_elem((8, 4), 0.2);
        // Ring graph adjacency
        let mut adj = Array2::<f64>::zeros((8, 8));
        for i in 0..8 {
            adj[[i, (i + 1) % 8]] = 1.0;
            adj[[(i + 1) % 8, i]] = 1.0;
        }
        let (x_pool, a_pool) = layer
            .forward_graph(&node_feat, &adj)
            .expect("Forward pass failed");
        assert!(x_pool.iter().all(|&v| v.is_finite()));
        assert!(a_pool.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_diffpool_auxiliary_losses() {
        let mut rng = rng();
        let layer = DiffPool::<f64>::new(4, 4, 2, &mut rng).expect("DiffPool creation");
        let node_feat = Array2::<f64>::from_elem((6, 4), 0.3);
        let adj = Array2::<f64>::eye(6);
        layer.forward_graph(&node_feat, &adj).expect("fwd failed");
        let lp = layer.link_prediction_loss_value().expect("lp loss");
        let ent = layer.entropy_loss_value().expect("ent loss");
        assert!(lp.is_finite());
        assert!(ent >= 0.0);
    }

    #[test]
    fn test_diffpool_n_clusters_gt_n_error() {
        let mut rng = rng();
        let layer = DiffPool::<f64>::new(4, 4, 5, &mut rng).expect("DiffPool creation");
        let node_feat = Array2::<f64>::from_elem((3, 4), 0.1); // only 3 nodes but 5 clusters
        let adj = Array2::<f64>::eye(3);
        assert!(layer.forward_graph(&node_feat, &adj).is_err());
    }

    #[test]
    fn test_diffpool_layer_trait_forward() {
        let mut rng = rng();
        let layer = DiffPool::<f64>::new(4, 4, 3, &mut rng).expect("DiffPool creation");
        let input = Array2::<f64>::from_elem((5, 4), 0.1).into_dyn();
        let out = layer.forward(&input).expect("Layer trait forward");
        assert_eq!(out.shape(), &[3, 4]);
    }

    #[test]
    fn test_diffpool_parameter_count() {
        let mut rng = rng();
        let layer = DiffPool::<f64>::new(4, 6, 3, &mut rng).expect("DiffPool creation");
        // pool GNN: 4→3 with bias = 4*3+3 = 15
        // embed GNN: 4→6 with bias = 4*6+6 = 30
        assert_eq!(layer.num_parameters(), (4 * 3 + 3) + (4 * 6 + 6));
    }

    // ── Additional edge-case tests ────────────────────────────────────────────

    #[test]
    fn test_global_pools_empty_features() {
        let features = Array2::<f64>::zeros((0, 3));
        let batch: Vec<usize> = vec![];
        let add = global_add_pool(&features, &batch).expect("add pool");
        let mean = global_mean_pool(&features, &batch).expect("mean pool");
        let max = global_max_pool(&features, &batch).expect("max pool");
        assert_eq!(add.shape(), &[0, 3]);
        assert_eq!(mean.shape(), &[0, 3]);
        assert_eq!(max.shape(), &[0, 3]);
    }

    #[test]
    fn test_global_pool_non_contiguous_batch() {
        // Nodes interleaved across graphs: 0→g0, 1→g1, 2→g0, 3→g1
        let features = Array2::<f64>::from_shape_vec(
            (4, 2),
            vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        )
        .expect("shape");
        let batch = vec![0usize, 1, 0, 1];
        let add = global_add_pool(&features, &batch).expect("add pool");
        // g0 = node0+node2 = [2,0], g1 = node1+node3 = [0,2]
        assert!((add[[0, 0]] - 2.0).abs() < 1e-10);
        assert!((add[[0, 1]]).abs() < 1e-10);
        assert!((add[[1, 0]]).abs() < 1e-10);
        assert!((add[[1, 1]] - 2.0).abs() < 1e-10);
    }

    /// Verify that the pooled adjacency from DiffPool is symmetric when the
    /// input adjacency is symmetric (holds up to floating-point noise).
    #[test]
    fn test_diffpool_symmetric_adjacency() {
        let mut rng = rng();
        let layer = DiffPool::<f64>::new(4, 4, 3, &mut rng).expect("DiffPool creation");
        let node_feat = Array2::<f64>::from_elem((6, 4), 0.1);
        let mut adj = Array2::<f64>::zeros((6, 6));
        // Fully connected symmetric graph
        for i in 0..6 {
            for j in 0..6 {
                if i != j {
                    adj[[i, j]] = 1.0;
                }
            }
        }
        let (_, a_pool) = layer.forward_graph(&node_feat, &adj).expect("fwd");
        let k = a_pool.nrows();
        for i in 0..k {
            for j in 0..k {
                let diff = (a_pool[[i, j]] - a_pool[[j, i]]).abs();
                assert!(
                    diff < 1e-8,
                    "a_pool[{i},{j}]={} != a_pool[{j},{i}]={}",
                    a_pool[[i, j]],
                    a_pool[[j, i]]
                );
            }
        }
    }

    /// Sanity-check that DiffPool preserves non-negativity when all inputs
    /// are non-negative (ReLU activations pass non-negatives through).
    #[test]
    fn test_diffpool_non_negative_features() {
        let mut rng = rng();
        let layer = DiffPool::<f64>::new(4, 4, 2, &mut rng).expect("DiffPool creation");
        let node_feat = Array2::<f64>::from_elem((5, 4), 0.5);
        let adj = Array2::<f64>::eye(5);
        let (x_pool, _) = layer.forward_graph(&node_feat, &adj).expect("fwd");
        assert!(
            x_pool.iter().all(|&v| v >= 0.0),
            "Expected non-negative pooled features"
        );
    }
}
