//! OptNet layer abstraction for embedding differentiable optimization
//! in neural network pipelines.
//!
//! The `OptNetLayer` trait provides a uniform interface for forward (solve)
//! and backward (gradient) passes, mirroring the pattern used in deep
//! learning frameworks.

use super::diff_qp::DifferentiableQP;
use super::types::{DiffQPConfig, DiffQPResult, ImplicitGradient};
use crate::error::OptimizeResult;

/// Trait for a differentiable optimization layer.
///
/// Implementations wrap a parametric optimization problem and expose
/// forward/backward methods suitable for integration into gradient-based
/// training pipelines.
pub trait OptNetLayer {
    /// The result type returned by the forward pass.
    type ForwardResult;

    /// Solve the optimization problem (forward pass).
    fn forward(&self) -> OptimizeResult<Self::ForwardResult>;

    /// Compute parameter gradients (backward pass).
    ///
    /// # Arguments
    /// * `result` – the result from a preceding `forward()` call.
    /// * `dl_dx` – upstream gradient of the loss w.r.t. the optimal solution.
    fn backward(
        &self,
        result: &Self::ForwardResult,
        dl_dx: &[f64],
    ) -> OptimizeResult<ImplicitGradient>;
}

/// A standard OptNet layer wrapping a differentiable QP.
#[derive(Debug, Clone)]
pub struct StandardOptNetLayer {
    /// The underlying differentiable QP.
    pub qp: DifferentiableQP,
    /// Configuration for forward/backward passes.
    pub config: DiffQPConfig,
}

impl StandardOptNetLayer {
    /// Create a new OptNet layer from a differentiable QP and config.
    pub fn new(qp: DifferentiableQP, config: DiffQPConfig) -> Self {
        Self { qp, config }
    }

    /// Solve a batch of QPs sharing the same config.
    pub fn forward_batch(
        qps: &[DifferentiableQP],
        config: &DiffQPConfig,
    ) -> OptimizeResult<Vec<DiffQPResult>> {
        DifferentiableQP::batched_forward(qps, config)
    }
}

impl OptNetLayer for StandardOptNetLayer {
    type ForwardResult = DiffQPResult;

    fn forward(&self) -> OptimizeResult<DiffQPResult> {
        self.qp.forward(&self.config)
    }

    fn backward(&self, result: &DiffQPResult, dl_dx: &[f64]) -> OptimizeResult<ImplicitGradient> {
        self.qp.backward(result, dl_dx, &self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_trait_dispatch() {
        let qp = DifferentiableQP::new(
            vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            vec![1.0, 2.0],
            vec![],
            vec![],
            vec![],
            vec![],
        )
        .expect("QP creation failed");

        let layer = StandardOptNetLayer::new(qp, DiffQPConfig::default());

        let result = layer.forward().expect("Forward failed");
        assert!(result.converged);

        let dl_dx = vec![1.0, 0.0];
        let grad = layer.backward(&result, &dl_dx).expect("Backward failed");
        assert_eq!(grad.dl_dc.len(), 2);
    }

    #[test]
    fn test_layer_batch_interface() {
        let qp1 = DifferentiableQP::new(vec![vec![2.0]], vec![1.0], vec![], vec![], vec![], vec![])
            .expect("QP1 creation failed");
        let qp2 = DifferentiableQP::new(vec![vec![4.0]], vec![2.0], vec![], vec![], vec![], vec![])
            .expect("QP2 creation failed");

        let config = DiffQPConfig::default();
        let results =
            StandardOptNetLayer::forward_batch(&[qp1, qp2], &config).expect("Batch failed");

        assert_eq!(results.len(), 2);
        // QP1: min x^2 + x → x* = -0.5
        assert!(
            (results[0].optimal_x[0] - (-0.5)).abs() < 1e-3,
            "batch[0].x = {}",
            results[0].optimal_x[0]
        );
        // QP2: min 2x^2 + 2x → x* = -0.5
        assert!(
            (results[1].optimal_x[0] - (-0.5)).abs() < 1e-2,
            "batch[1].x = {}",
            results[1].optimal_x[0]
        );
    }
}
