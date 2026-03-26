//! DARTS: Differentiable Architecture Search (Liu et al., ICLR 2019)
//!
//! Relaxes the discrete architecture choice over a set of candidate operations to a
//! continuous mixing via softmax weights. During search both the network weights and
//! architecture weights are optimised in a bi-level fashion.  After search the discrete
//! architecture is recovered by taking argmax per edge.
//!
//! ## References
//!
//! - Liu, H., Simonyan, K. and Yang, Y. (2019). "DARTS: Differentiable Architecture
//!   Search". ICLR 2019.

use crate::error::{OptimizeError, OptimizeResult};

// ───────────────────────────────────────────────────────────────── Operations ──

/// Candidate primitive operations for DARTS cells.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    /// Identity (skip connection).
    Identity,
    /// Zero (no information flow).
    Zero,
    /// 3×3 separable convolution.
    Conv3x3,
    /// 5×5 separable convolution.
    Conv5x5,
    /// 3×3 max pooling.
    MaxPool,
    /// 3×3 average pooling.
    AvgPool,
    /// Skip connection (same as Identity but conceptually distinct).
    SkipConnect,
}

impl Operation {
    /// Rough FLOP estimate for a single forward pass through this operation.
    ///
    /// Formula used: conv FLOPs ≈ 2 * kernel² * channels² (assuming spatial size = 1
    /// for simplicity; callers may scale by H*W).
    pub fn cost_flops(&self, channels: usize) -> f64 {
        let c = channels as f64;
        match self {
            Operation::Identity => 0.0,
            Operation::Zero => 0.0,
            Operation::Conv3x3 => 2.0 * 9.0 * c * c,
            Operation::Conv5x5 => 2.0 * 25.0 * c * c,
            Operation::MaxPool => c, // negligible — just comparisons
            Operation::AvgPool => c, // one add per element
            Operation::SkipConnect => 0.0,
        }
    }

    /// Human-readable name used in diagnostic output.
    pub fn name(&self) -> &'static str {
        match self {
            Operation::Identity => "identity",
            Operation::Zero => "zero",
            Operation::Conv3x3 => "conv3x3",
            Operation::Conv5x5 => "conv5x5",
            Operation::MaxPool => "max_pool",
            Operation::AvgPool => "avg_pool",
            Operation::SkipConnect => "skip_connect",
        }
    }

    /// All primitive operations in a fixed canonical order.
    pub fn all() -> &'static [Operation] {
        &[
            Operation::Identity,
            Operation::Zero,
            Operation::Conv3x3,
            Operation::Conv5x5,
            Operation::MaxPool,
            Operation::AvgPool,
        ]
    }
}

// ────────────────────────────────────────────────────────────── DartsConfig ──

/// Configuration for a DARTS architecture search experiment.
#[derive(Debug, Clone)]
pub struct DartsConfig {
    /// Number of cells stacked in the super-network.
    pub n_cells: usize,
    /// Number of candidate operations per edge.
    pub n_operations: usize,
    /// Number of feature channels (used for FLOP estimation).
    pub channels: usize,
    /// Number of intermediate nodes per cell.
    pub n_nodes: usize,
    /// Learning rate for architecture parameter updates.
    pub arch_lr: f64,
    /// Learning rate for network weight updates.
    pub weight_lr: f64,
    /// Softmax temperature (lower → sharper distribution).
    pub temperature: f64,
}

impl Default for DartsConfig {
    fn default() -> Self {
        Self {
            n_cells: 4,
            n_operations: 6,
            channels: 16,
            n_nodes: 4,
            arch_lr: 3e-4,
            weight_lr: 3e-4,
            temperature: 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────── MixedOperation ──

/// One mixed operation on a directed edge in the DARTS cell DAG.
///
/// Maintains per-operation un-normalised log-weights (architecture parameters).
#[derive(Debug, Clone)]
pub struct MixedOperation {
    /// Un-normalised architecture parameters α_k, one per operation.
    pub arch_params: Vec<f64>,
    /// Cached per-operation outputs from the last `forward` call.
    pub operation_outputs: Option<Vec<Vec<f64>>>,
}

impl MixedOperation {
    /// Create a new `MixedOperation` with `n_ops` operations, initialised to
    /// uniform architecture weights (all log-weights = 0).
    pub fn new(n_ops: usize) -> Self {
        Self {
            arch_params: vec![0.0_f64; n_ops],
            operation_outputs: None,
        }
    }

    /// Compute softmax-normalised operation weights at the given temperature.
    ///
    /// `weights[k] = exp(α_k / T) / Σ_j exp(α_j / T)`
    pub fn weights(&self, temperature: f64) -> Vec<f64> {
        let t = temperature.max(1e-8); // guard against divide-by-zero
        let scaled: Vec<f64> = self.arch_params.iter().map(|a| a / t).collect();
        // numerically-stable softmax
        let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scaled.iter().map(|s| (s - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum == 0.0 {
            vec![1.0 / self.arch_params.len() as f64; self.arch_params.len()]
        } else {
            exps.iter().map(|e| e / sum).collect()
        }
    }

    /// Forward pass: weighted sum Σ_k w_k · op_k(x).
    ///
    /// `op_fn(k, x)` returns the output of operation k applied to input x.
    pub fn forward(
        &mut self,
        x: &[f64],
        op_fn: impl Fn(usize, &[f64]) -> Vec<f64>,
        temperature: f64,
    ) -> Vec<f64> {
        let w = self.weights(temperature);
        let n_ops = self.arch_params.len();
        // collect individual op outputs
        let op_outputs: Vec<Vec<f64>> = (0..n_ops).map(|k| op_fn(k, x)).collect();
        // weighted sum
        let out_len = op_outputs.first().map(|v| v.len()).unwrap_or(x.len());
        let mut result = vec![0.0_f64; out_len];
        for (k, out) in op_outputs.iter().enumerate() {
            for (r, o) in result.iter_mut().zip(out.iter()) {
                *r += w[k] * o;
            }
        }
        self.operation_outputs = Some(op_outputs);
        result
    }

    /// Index of the operation with the highest architecture weight.
    pub fn argmax_op(&self) -> usize {
        self.arch_params
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ──────────────────────────────────────────────────────────────── DartsCell ──

/// A DARTS cell: a DAG with `n_input_nodes` fixed inputs and `n_nodes`
/// intermediate nodes.  Each node aggregates outputs from all prior nodes via
/// `MixedOperation` edges.
#[derive(Debug, Clone)]
pub struct DartsCell {
    /// Number of intermediate (learnable) nodes.
    pub n_nodes: usize,
    /// Number of fixed input nodes (typically 2 in the standard DARTS setup).
    pub n_input_nodes: usize,
    /// `edges[i][j]` is the `MixedOperation` from node j to intermediate node i.
    /// Node indices: 0..n_input_nodes are inputs; n_input_nodes..n_input_nodes+n_nodes are
    /// intermediate nodes.
    pub edges: Vec<Vec<MixedOperation>>,
}

impl DartsCell {
    /// Create a new DARTS cell.
    ///
    /// # Arguments
    /// - `n_input_nodes`: Number of input nodes (preceding-cell outputs).
    /// - `n_intermediate_nodes`: Number of intermediate nodes to build.
    /// - `n_ops`: Number of candidate operations per edge.
    pub fn new(n_input_nodes: usize, n_intermediate_nodes: usize, n_ops: usize) -> Self {
        // edges[i] = mixed operations coming into intermediate node i
        // node i receives edges from all n_input_nodes + i prior nodes
        let edges: Vec<Vec<MixedOperation>> = (0..n_intermediate_nodes)
            .map(|i| {
                let n_predecessors = n_input_nodes + i;
                (0..n_predecessors)
                    .map(|_| MixedOperation::new(n_ops))
                    .collect()
            })
            .collect();

        Self {
            n_nodes: n_intermediate_nodes,
            n_input_nodes,
            edges,
        }
    }

    /// Forward pass through the cell.
    ///
    /// Each intermediate node output = Σ_{j < i} mixed_op_{ij}(node_j_output).
    /// Final cell output = concatenation of all intermediate node outputs.
    ///
    /// # Arguments
    /// - `inputs`: Outputs of the n_input_nodes preceding this cell.
    /// - `temperature`: Softmax temperature forwarded to each `MixedOperation`.
    pub fn forward(&mut self, inputs: &[Vec<f64>], temperature: f64) -> Vec<f64> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let feature_len = inputs[0].len();
        // All node outputs (inputs first, then intermediate).
        let mut node_outputs: Vec<Vec<f64>> = inputs.to_vec();

        for i in 0..self.n_nodes {
            let n_prev = self.n_input_nodes + i;
            let mut node_out = vec![0.0_f64; feature_len];
            for j in 0..n_prev {
                let src = node_outputs[j].clone();
                let edge_out = self.edges[i][j].forward(&src, default_op_fn, temperature);
                for (no, eo) in node_out.iter_mut().zip(edge_out.iter()) {
                    *no += eo;
                }
            }
            node_outputs.push(node_out);
        }

        // Concatenate intermediate node outputs (skip the input nodes).
        let mut result = Vec::with_capacity(self.n_nodes * feature_len);
        for node_out in node_outputs.iter().skip(self.n_input_nodes) {
            result.extend_from_slice(node_out);
        }
        result
    }

    /// Collect all architecture parameters from every edge in this cell, flattened.
    pub fn arch_parameters(&self) -> Vec<f64> {
        self.edges
            .iter()
            .flat_map(|row| row.iter().flat_map(|mo| mo.arch_params.iter().cloned()))
            .collect()
    }

    /// Apply gradient updates to architecture parameters using a gradient slice.
    ///
    /// `grads` must have the same length as `arch_parameters()`.
    pub fn update_arch_params(&mut self, grads: &[f64], lr: f64) -> OptimizeResult<()> {
        let n_params: usize = self
            .edges
            .iter()
            .flat_map(|row| row.iter())
            .map(|mo| mo.arch_params.len())
            .sum();
        if grads.len() != n_params {
            return Err(OptimizeError::InvalidInput(format!(
                "Expected {} gradient values, got {}",
                n_params,
                grads.len()
            )));
        }
        let mut idx = 0;
        for row in self.edges.iter_mut() {
            for mo in row.iter_mut() {
                for p in mo.arch_params.iter_mut() {
                    *p -= lr * grads[idx];
                    idx += 1;
                }
            }
        }
        Ok(())
    }

    /// Derive the discrete architecture for this cell: argmax per edge.
    ///
    /// Returns a `Vec<Vec<usize>>` with the same shape as `edges`, where each
    /// entry is the index of the best operation.
    pub fn derive_discrete(&self) -> Vec<Vec<usize>> {
        self.edges
            .iter()
            .map(|row| row.iter().map(|mo| mo.argmax_op()).collect())
            .collect()
    }
}

/// Default operation function used inside cells: identity (returns x unchanged).
fn default_op_fn(_k: usize, x: &[f64]) -> Vec<f64> {
    x.to_vec()
}

// ────────────────────────────────────────────────────────────── DartsSearch ──

/// Top-level DARTS search controller.
///
/// Manages a stack of `DartsCell`s and implements the bi-level optimisation loop.
#[derive(Debug, Clone)]
pub struct DartsSearch {
    /// Stack of cells forming the super-network.
    pub cells: Vec<DartsCell>,
    /// Configuration.
    pub config: DartsConfig,
    /// Flat network weights (shared across all cells for this toy model).
    weights: Vec<f64>,
}

impl DartsSearch {
    /// Construct a `DartsSearch` from the given config.
    pub fn new(config: DartsConfig) -> Self {
        let cells: Vec<DartsCell> = (0..config.n_cells)
            .map(|_| DartsCell::new(2, config.n_nodes, config.n_operations))
            .collect();
        // Simple weight vector (one scalar weight per cell for the toy model).
        let weights = vec![0.01_f64; config.n_cells];
        Self {
            cells,
            config,
            weights,
        }
    }

    /// Return all architecture parameters across all cells, flattened.
    pub fn arch_parameters(&self) -> Vec<f64> {
        self.cells
            .iter()
            .flat_map(|c| c.arch_parameters())
            .collect()
    }

    /// Total number of architecture parameters.
    pub fn n_arch_params(&self) -> usize {
        self.cells.iter().map(|c| c.arch_parameters().len()).sum()
    }

    /// Apply a gradient step to architecture parameters.
    ///
    /// `grads` must be the same length as `arch_parameters()`.
    pub fn update_arch_params(&mut self, grads: &[f64], lr: f64) -> OptimizeResult<()> {
        let total = self.n_arch_params();
        if grads.len() != total {
            return Err(OptimizeError::InvalidInput(format!(
                "Expected {} arch-param grads, got {}",
                total,
                grads.len()
            )));
        }
        let mut offset = 0;
        for cell in self.cells.iter_mut() {
            let n = cell.arch_parameters().len();
            cell.update_arch_params(&grads[offset..offset + n], lr)?;
            offset += n;
        }
        Ok(())
    }

    /// Derive the discrete architecture: for each cell, argmax op per edge.
    ///
    /// Returns `Vec<Vec<Vec<usize>>>` — `[cell][intermediate_node][predecessor]`.
    pub fn derive_discrete_arch_indices(&self) -> Vec<Vec<Vec<usize>>> {
        self.cells.iter().map(|c| c.derive_discrete()).collect()
    }

    /// Derive the discrete architecture as a vec of `Operation` vectors.
    ///
    /// Uses `Operation::all()` to map operation index to `Operation`.  If
    /// `n_operations` exceeds the canonical list length, `Operation::Identity`
    /// is used as a fallback.
    pub fn derive_discrete_arch(&self) -> Vec<Vec<Operation>> {
        let ops = Operation::all();
        self.derive_discrete_arch_indices()
            .iter()
            .map(|cell_disc| {
                cell_disc
                    .iter()
                    .flat_map(|node_edges| {
                        node_edges.iter().map(|&idx| {
                            if idx < ops.len() {
                                ops[idx]
                            } else {
                                Operation::Identity
                            }
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Compute a simple regression loss (MSE) over a dataset.
    ///
    /// The toy model prediction is: y_hat_i = w · mean(x_i), where w is the
    /// sum of `self.weights`.  This is purely for exercising the bi-level loop.
    fn compute_loss(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {
        if x.is_empty() || y.is_empty() {
            return 0.0;
        }
        let w_sum: f64 = self.weights.iter().sum();
        let mut loss = 0.0_f64;
        let n = x.len().min(y.len());
        for i in 0..n {
            let x_mean = if x[i].is_empty() {
                0.0
            } else {
                x[i].iter().sum::<f64>() / x[i].len() as f64
            };
            let pred = w_sum * x_mean;
            let diff = pred - y[i];
            loss += diff * diff;
        }
        loss / n as f64
    }

    /// Compute MSE gradient with respect to `self.weights` (one grad per cell weight).
    fn weight_grads(&self, x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
        let n = x.len().min(y.len());
        if n == 0 {
            return vec![0.0_f64; self.weights.len()];
        }
        let w_sum: f64 = self.weights.iter().sum();
        let mut grad_sum = 0.0_f64;
        for i in 0..n {
            let x_mean = if x[i].is_empty() {
                0.0
            } else {
                x[i].iter().sum::<f64>() / x[i].len() as f64
            };
            let pred = w_sum * x_mean;
            let diff = pred - y[i];
            // d(loss)/d(w_sum) = 2 * diff * x_mean / n
            grad_sum += 2.0 * diff * x_mean / n as f64;
        }
        // Each cell weight contributes equally to w_sum.
        vec![grad_sum; self.weights.len()]
    }

    /// Compute approximate gradient of loss with respect to architecture params
    /// via finite differences (central differences, step = 1e-4).
    fn arch_grads_fd(&self, x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
        let n = self.n_arch_params();
        if n == 0 {
            return Vec::new();
        }
        let mut grads = vec![0.0_f64; n];
        let h = 1e-4;
        let mut offset = 0;
        for cell_idx in 0..self.cells.len() {
            let cell_n = self.cells[cell_idx].arch_parameters().len();
            for local_j in 0..cell_n {
                let global_j = offset + local_j;
                // +h
                let mut search_plus = self.clone();
                let params_plus = search_plus.cells[cell_idx].arch_parameters();
                let mut p_plus = params_plus.clone();
                p_plus[local_j] += h;
                // Rebuild cell arch params from the modified flat vector
                let _ = search_plus.cells[cell_idx].set_arch_params(&p_plus);
                let loss_plus = search_plus.compute_loss(x, y);

                // -h
                let mut search_minus = self.clone();
                let params_minus = search_minus.cells[cell_idx].arch_parameters();
                let mut p_minus = params_minus.clone();
                p_minus[local_j] -= h;
                let _ = search_minus.cells[cell_idx].set_arch_params(&p_minus);
                let loss_minus = search_minus.compute_loss(x, y);

                grads[global_j] = (loss_plus - loss_minus) / (2.0 * h);
            }
            offset += cell_n;
        }
        grads
    }

    /// One bilevel optimisation step (approximate first-order DARTS).
    ///
    /// Inner step: update network weights on `train_x`/`train_y`.
    /// Outer step: update architecture params on `val_x`/`val_y`.
    ///
    /// Returns `(train_loss, val_loss)` before the update.
    pub fn bilevel_step(
        &mut self,
        train_x: &[Vec<f64>],
        train_y: &[f64],
        val_x: &[Vec<f64>],
        val_y: &[f64],
    ) -> (f64, f64) {
        let train_loss = self.compute_loss(train_x, train_y);
        let val_loss = self.compute_loss(val_x, val_y);

        // Inner: gradient step on network weights using train data.
        let w_grads = self.weight_grads(train_x, train_y);
        let lr_w = self.config.weight_lr;
        for (w, g) in self.weights.iter_mut().zip(w_grads.iter()) {
            *w -= lr_w * g;
        }

        // Outer: gradient step on architecture params using val data.
        let a_grads = self.arch_grads_fd(val_x, val_y);
        let lr_a = self.config.arch_lr;
        if !a_grads.is_empty() {
            let _ = self.update_arch_params(&a_grads, lr_a);
        }

        (train_loss, val_loss)
    }
}

// ──────────────────────────────────────── helper: set_arch_params on DartsCell ──

impl DartsCell {
    /// Replace all architecture parameters with values from the flat slice.
    pub fn set_arch_params(&mut self, params: &[f64]) -> OptimizeResult<()> {
        let total: usize = self
            .edges
            .iter()
            .flat_map(|r| r.iter())
            .map(|m| m.arch_params.len())
            .sum();
        if params.len() != total {
            return Err(OptimizeError::InvalidInput(format!(
                "set_arch_params: expected {total} values, got {}",
                params.len()
            )));
        }
        let mut idx = 0;
        for row in self.edges.iter_mut() {
            for mo in row.iter_mut() {
                for p in mo.arch_params.iter_mut() {
                    *p = params[idx];
                    idx += 1;
                }
            }
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════ tests ═══

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_operation_weights_sum_to_one() {
        let mo = MixedOperation::new(6);
        let w = mo.weights(1.0);
        assert_eq!(w.len(), 6);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "weights sum = {sum}");
    }

    #[test]
    fn mixed_operation_weights_temperature_effect() {
        // Low temperature should sharpen the distribution.
        let mut mo = MixedOperation::new(4);
        mo.arch_params = vec![1.0, 0.5, 0.3, 0.2];
        let w_hot = mo.weights(10.0);
        let w_cold = mo.weights(0.1);
        // Highest-weight op (index 0) should have larger weight at low temp.
        assert!(w_cold[0] > w_hot[0], "cold should be sharper");
    }

    #[test]
    fn mixed_operation_forward_correct_shape() {
        let mut mo = MixedOperation::new(3);
        let x = vec![1.0_f64; 8];
        let out = mo.forward(&x, |_k, v| v.to_vec(), 1.0);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn darts_cell_forward_output_shape() {
        let mut cell = DartsCell::new(2, 3, 4);
        let inputs = vec![vec![1.0_f64; 8], vec![0.5_f64; 8]];
        let out = cell.forward(&inputs, 1.0);
        // Output should be n_nodes * feature_len = 3 * 8 = 24.
        assert_eq!(out.len(), 24);
    }

    #[test]
    fn derive_discrete_arch_returns_ops() {
        let config = DartsConfig {
            n_cells: 2,
            n_operations: 6,
            n_nodes: 3,
            ..Default::default()
        };
        let search = DartsSearch::new(config);
        let arch = search.derive_discrete_arch();
        assert_eq!(arch.len(), 2, "one vec per cell");
        // Each cell has n_nodes intermediate nodes, each receiving 2+(0..n-1) edges.
        // Total edges per cell = 2+3+4 = 9 for n_nodes=3, n_input_nodes=2.
        for cell_ops in &arch {
            assert!(!cell_ops.is_empty());
        }
    }

    #[test]
    fn bilevel_step_runs_without_error() {
        let config = DartsConfig::default();
        let mut search = DartsSearch::new(config);
        let train_x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let train_y = vec![1.5, 3.5];
        let val_x = vec![vec![0.5, 1.5]];
        let val_y = vec![1.0];
        let (tl, vl) = search.bilevel_step(&train_x, &train_y, &val_x, &val_y);
        assert!(tl.is_finite());
        assert!(vl.is_finite());
    }

    #[test]
    fn arch_parameters_length_consistent() {
        let config = DartsConfig {
            n_cells: 3,
            n_operations: 5,
            n_nodes: 2,
            ..Default::default()
        };
        let search = DartsSearch::new(config);
        let params = search.arch_parameters();
        assert_eq!(params.len(), search.n_arch_params());
    }

    #[test]
    fn update_arch_params_wrong_length_errors() {
        let mut search = DartsSearch::new(DartsConfig::default());
        let result = search.update_arch_params(&[1.0, 2.0], 0.01);
        assert!(result.is_err());
    }
}
