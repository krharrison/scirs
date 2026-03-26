//! Types and configuration for learned smoothers.
//!
//! Defines configuration structures, smoother weight storage, training
//! parameters, and convergence metrics used across all smoother variants.

use crate::error::SparseResult;

// ---------------------------------------------------------------------------
// Smoother type selection
// ---------------------------------------------------------------------------

/// Selects the learned smoother variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum SmootherType {
    /// Parametric linear smoother: x_{k+1} = x_k + W · r_k
    #[default]
    Linear,
    /// Per-node 2-layer MLP with shared weights (GNN-style).
    MLP,
    /// Chebyshev polynomial smoother with learned coefficients.
    Chebyshev,
}

// ---------------------------------------------------------------------------
// Main configuration
// ---------------------------------------------------------------------------

/// Configuration for a learned smoother.
#[derive(Debug, Clone)]
pub struct LearnedSmootherConfig {
    /// Which smoother variant to use.
    pub smoother_type: SmootherType,
    /// Learning rate for weight updates.
    pub learning_rate: f64,
    /// Maximum number of training steps.
    pub max_training_steps: usize,
    /// Convergence tolerance for training.
    pub convergence_tol: f64,
    /// Relaxation parameter (omega) for initialisation.
    pub omega: f64,
    /// Number of pre-smoothing sweeps in a V-cycle.
    pub pre_sweeps: usize,
    /// Number of post-smoothing sweeps in a V-cycle.
    pub post_sweeps: usize,
}

impl Default for LearnedSmootherConfig {
    fn default() -> Self {
        Self {
            smoother_type: SmootherType::Linear,
            learning_rate: 0.01,
            max_training_steps: 1000,
            convergence_tol: 1e-6,
            omega: 2.0 / 3.0,
            pre_sweeps: 2,
            post_sweeps: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Training configuration
// ---------------------------------------------------------------------------

/// Training hyper-parameters for gradient-descent-based weight learning.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate for SGD / Adam.
    pub learning_rate: f64,
    /// Maximum number of training epochs.
    pub max_epochs: usize,
    /// Mini-batch size (number of right-hand-side vectors per step).
    pub batch_size: usize,
    /// Early-stopping tolerance on loss decrease.
    pub convergence_tol: f64,
    /// Momentum coefficient (0 = vanilla SGD).
    pub momentum: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_epochs: 100,
            batch_size: 16,
            convergence_tol: 1e-8,
            momentum: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Smoother weights
// ---------------------------------------------------------------------------

/// Stores the trainable parameters of a learned smoother.
///
/// For a linear smoother the weights are a diagonal (or dense) matrix W;
/// for an MLP smoother the weights are the layer parameters.
#[derive(Debug, Clone)]
pub struct SmootherWeights {
    /// Layer-wise weight matrices stored row-major.
    pub matrices: Vec<Vec<f64>>,
    /// Layer-wise bias vectors.
    pub biases: Vec<Vec<f64>>,
    /// Dimensions of each layer: `(input_dim, output_dim)`.
    pub layer_dims: Vec<(usize, usize)>,
}

impl SmootherWeights {
    /// Create an empty weight container with no layers.
    pub fn empty() -> Self {
        Self {
            matrices: Vec::new(),
            biases: Vec::new(),
            layer_dims: Vec::new(),
        }
    }

    /// Create weights for a single diagonal layer of size `n`.
    pub fn diagonal(diag: Vec<f64>) -> Self {
        let n = diag.len();
        Self {
            matrices: vec![diag],
            biases: vec![vec![0.0; n]],
            layer_dims: vec![(n, n)],
        }
    }

    /// Total number of trainable scalar parameters.
    pub fn num_parameters(&self) -> usize {
        let mat_params: usize = self.matrices.iter().map(|m| m.len()).sum();
        let bias_params: usize = self.biases.iter().map(|b| b.len()).sum();
        mat_params + bias_params
    }
}

// ---------------------------------------------------------------------------
// Convergence metrics
// ---------------------------------------------------------------------------

/// Metrics produced after a smoothing / solve run.
#[derive(Debug, Clone)]
pub struct SmootherMetrics {
    /// Estimated spectral radius of the error propagation operator (I - WA).
    pub spectral_radius_reduction: f64,
    /// Convergence factor: ratio ‖e_{k+1}‖ / ‖e_k‖ averaged over iterations.
    pub convergence_factor: f64,
    /// Ratio ‖r_final‖ / ‖r_0‖ of the residual norms.
    pub residual_reduction: f64,
    /// Per-epoch (or per-step) training loss history.
    pub training_loss_history: Vec<f64>,
}

impl SmootherMetrics {
    /// Create a default/empty metrics struct.
    pub fn new() -> Self {
        Self {
            spectral_radius_reduction: 1.0,
            convergence_factor: 1.0,
            residual_reduction: 1.0,
            training_loss_history: Vec::new(),
        }
    }
}

impl Default for SmootherMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Smoother trait
// ---------------------------------------------------------------------------

/// Trait for smoothers that can be plugged into a multigrid cycle.
pub trait Smoother {
    /// Apply `n_sweeps` smoothing iterations: update `x` in-place.
    ///
    /// The matrix is given in raw CSR form (values, row_ptr, col_idx).
    fn smooth(
        &self,
        a_values: &[f64],
        a_row_ptr: &[usize],
        a_col_idx: &[usize],
        x: &mut [f64],
        b: &[f64],
        n_sweeps: usize,
    ) -> SparseResult<()>;

    /// Perform one training step and return the loss.
    fn train_step(
        &mut self,
        a_values: &[f64],
        a_row_ptr: &[usize],
        a_col_idx: &[usize],
        x: &mut [f64],
        b: &[f64],
        x_exact: &[f64],
        lr: f64,
    ) -> SparseResult<f64>;
}
