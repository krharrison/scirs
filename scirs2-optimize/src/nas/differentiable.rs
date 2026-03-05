//! DARTS: Differentiable Architecture Search (Liu et al., ICLR 2019).
//!
//! Relaxes the discrete architecture choice over a set of candidate
//! operations to a continuous mixing via softmax weights α.  During
//! search, both the network weights w and architecture weights α are
//! optimised (bi-level).  After search the discrete architecture is
//! recovered by taking argmax(α) per edge.
//!
//! This module provides the core data structure and the derive-discrete
//! step.  Actual gradient-based updates require coupling with a neural
//! network trainer; `update_alpha` provides a hook for applying
//! externally-computed gradients.

use crate::error::OptimizeError;
use crate::nas::search_space::{ArchEdge, ArchNode, Architecture, OpType, SearchSpace};

/// DARTS continuous architecture parameterisation.
///
/// `alpha[e][k]` is the (un-normalised) log-weight for operation `k` on
/// edge `e`.  Normalised weights are obtained via softmax.
#[derive(Debug, Clone)]
pub struct DARTSSearch {
    /// Number of intermediate nodes in the cell
    pub n_nodes: usize,
    /// Number of candidate operations
    pub n_ops: usize,
    /// Architecture weights: shape `[n_edges, n_ops]`
    pub alpha: Vec<Vec<f64>>,
    /// Learning rate for architecture weight updates
    pub learning_rate: f64,
    /// Number of input nodes (from previous cells)
    pub n_input_nodes: usize,
}

impl DARTSSearch {
    /// Initialise DARTS with uniform architecture weights.
    ///
    /// # Arguments
    /// - `n_nodes`: Number of intermediate nodes per cell.
    /// - `operations`: Slice of candidate operations.
    /// - `n_input_nodes`: Number of fixed input nodes (e.g., 2 for DARTS).
    pub fn new(n_nodes: usize, operations: &[OpType], n_input_nodes: usize) -> Self {
        let n_ops = operations.len();
        // In DARTS each intermediate node i receives edges from all
        // previous nodes (i nodes including the n_input_nodes inputs).
        // Total edges = sum_{i=0}^{n_nodes-1} (n_input_nodes + i)
        let n_edges: usize = (0..n_nodes).map(|i| n_input_nodes + i).sum();
        let init_weight = if n_ops > 0 { 1.0 / n_ops as f64 } else { 0.0 };
        let alpha = vec![vec![init_weight; n_ops]; n_edges.max(1)];

        Self {
            n_nodes,
            n_ops,
            alpha,
            learning_rate: 3e-4,
            n_input_nodes,
        }
    }

    /// Number of edges in the DARTS cell.
    pub fn n_edges(&self) -> usize {
        self.alpha.len()
    }

    /// Softmax-normalised operation weights for a given edge.
    ///
    /// Returns a zero vector if `edge_idx` is out of range.
    pub fn get_op_weights(&self, edge_idx: usize) -> Vec<f64> {
        if edge_idx >= self.alpha.len() {
            return vec![0.0; self.n_ops];
        }
        let raw = &self.alpha[edge_idx];
        let max = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = raw.iter().map(|x| (x - max).exp()).collect();
        let sum: f64 = exp.iter().sum();
        if sum == 0.0 {
            return vec![1.0 / self.n_ops as f64; self.n_ops];
        }
        exp.iter().map(|e| e / sum).collect()
    }

    /// Derive a discrete architecture by taking argmax(α) per edge.
    ///
    /// Returns an `Architecture` whose edges carry the selected operations.
    pub fn derive_architecture(
        &self,
        space: &SearchSpace,
        n_cells: usize,
        channels: usize,
        n_classes: usize,
    ) -> Architecture {
        let mut arch = Architecture::new(n_cells, channels, n_classes);

        // Add input nodes (two previous cell outputs)
        for i in 0..self.n_input_nodes {
            arch.nodes.push(ArchNode {
                id: i,
                name: format!("input{}", i),
                output_channels: channels,
            });
        }

        // Add intermediate nodes
        let mut edge_idx = 0usize;
        for i in 0..self.n_nodes {
            let node_id = self.n_input_nodes + i;
            arch.nodes.push(ArchNode {
                id: node_id,
                name: format!("node{}", i),
                output_channels: channels,
            });

            // Edges from all previous nodes to this one
            let n_prev = self.n_input_nodes + i;
            for from_id in 0..n_prev {
                let weights = self.get_op_weights(edge_idx);
                let best_op_idx = weights
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                let op = space
                    .operations
                    .get(best_op_idx)
                    .cloned()
                    .unwrap_or(OpType::Skip);

                arch.edges.push(ArchEdge {
                    from: from_id,
                    to: node_id,
                    op,
                });
                edge_idx += 1;
            }
        }

        arch
    }

    /// Apply an external gradient update to a single (edge, op) weight.
    ///
    /// Typical usage: call after computing `∂L/∂α[edge_idx][op_idx]`
    /// from a validation loss.
    pub fn update_alpha(
        &mut self,
        edge_idx: usize,
        op_idx: usize,
        grad: f64,
    ) -> Result<(), OptimizeError> {
        if edge_idx >= self.alpha.len() {
            return Err(OptimizeError::InvalidParameter(format!(
                "edge_idx {} out of range (n_edges = {})",
                edge_idx,
                self.alpha.len()
            )));
        }
        if op_idx >= self.n_ops {
            return Err(OptimizeError::InvalidParameter(format!(
                "op_idx {} out of range (n_ops = {})",
                op_idx, self.n_ops
            )));
        }
        self.alpha[edge_idx][op_idx] += self.learning_rate * grad;
        Ok(())
    }

    /// Batch update: apply gradient matrix `grads[edge][op]` to all weights.
    ///
    /// `grads` must have shape `[n_edges, n_ops]`.
    pub fn update_alpha_batch(&mut self, grads: &[Vec<f64>]) -> Result<(), OptimizeError> {
        if grads.len() != self.alpha.len() {
            return Err(OptimizeError::InvalidParameter(format!(
                "grads has {} rows but alpha has {}",
                grads.len(),
                self.alpha.len()
            )));
        }
        for (e, row) in grads.iter().enumerate() {
            if row.len() != self.n_ops {
                return Err(OptimizeError::InvalidParameter(format!(
                    "grads[{}] has {} columns but n_ops = {}",
                    e,
                    row.len(),
                    self.n_ops
                )));
            }
            for (k, &g) in row.iter().enumerate() {
                self.alpha[e][k] += self.learning_rate * g;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::search_space::SearchSpace;

    fn make_darts() -> DARTSSearch {
        let space = SearchSpace::darts_like(4);
        DARTSSearch::new(4, &space.operations, 2)
    }

    #[test]
    fn test_get_op_weights_sum_to_one() {
        let darts = make_darts();
        for e in 0..darts.n_edges() {
            let w = darts.get_op_weights(e);
            let sum: f64 = w.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "weights do not sum to 1: {}",
                sum
            );
        }
    }

    #[test]
    fn test_derive_architecture_correct_structure() {
        let space = SearchSpace::darts_like(4);
        let darts = DARTSSearch::new(4, &space.operations, 2);
        let arch = darts.derive_architecture(&space, 2, 64, 10);

        // Should have n_input_nodes + n_nodes nodes
        assert_eq!(arch.nodes.len(), 2 + 4);
        // All edges should have valid from/to indices
        for e in &arch.edges {
            assert!(e.from < arch.nodes.len());
            assert!(e.to < arch.nodes.len());
        }
    }

    #[test]
    fn test_update_alpha_changes_weights() {
        let mut darts = make_darts();
        let before = darts.alpha[0][0];
        darts.update_alpha(0, 0, 1.0).expect("update failed");
        assert!(
            (darts.alpha[0][0] - before).abs() > 1e-12,
            "alpha did not change"
        );
    }

    #[test]
    fn test_update_alpha_out_of_range_errors() {
        let mut darts = make_darts();
        assert!(darts.update_alpha(9999, 0, 1.0).is_err());
        assert!(darts.update_alpha(0, 9999, 1.0).is_err());
    }

    #[test]
    fn test_update_alpha_batch_correct_shape() {
        let mut darts = make_darts();
        let n_e = darts.n_edges();
        let n_o = darts.n_ops;
        let grads = vec![vec![0.1; n_o]; n_e];
        darts
            .update_alpha_batch(&grads)
            .expect("batch update failed");
    }

    #[test]
    fn test_update_alpha_batch_wrong_shape_errors() {
        let mut darts = make_darts();
        let grads = vec![vec![0.1; darts.n_ops]; darts.n_edges() + 1];
        assert!(darts.update_alpha_batch(&grads).is_err());
    }

    #[test]
    fn test_argmax_selects_highest_weight() {
        let space = SearchSpace::darts_like(2);
        let mut darts = DARTSSearch::new(2, &space.operations, 2);
        // Manually set edge 0 to strongly prefer op 3
        let n_ops = darts.n_ops;
        for k in 0..n_ops {
            darts.alpha[0][k] = 0.0;
        }
        darts.alpha[0][3] = 10.0;

        let arch = darts.derive_architecture(&space, 1, 32, 10);
        // The op on the first edge should be space.operations[3]
        if let Some(e) = arch.edges.first() {
            assert_eq!(e.op, space.operations[3]);
        }
    }
}
