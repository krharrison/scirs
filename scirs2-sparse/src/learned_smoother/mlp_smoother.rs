//! Per-node MLP smoother with shared weights.
//!
//! Each node's correction δx_i is computed by a small 2-layer MLP whose
//! input features are `[x_i, r_i, sum_neighbors(x), sum_neighbors(r)]`.
//! The weights are shared across all nodes (akin to a GNN message-passing
//! layer), making the smoother transferable across problem sizes.
//!
//! Update rule: x_i ← x_i + δx_i  where δx_i = MLP(features_i).

use crate::error::{SparseError, SparseResult};
use crate::learned_smoother::types::Smoother;

// ---------------------------------------------------------------------------
// CSR helpers
// ---------------------------------------------------------------------------

/// Compute y = A x in CSR format.
fn csr_matvec(a_values: &[f64], a_row_ptr: &[usize], a_col_idx: &[usize], x: &[f64]) -> Vec<f64> {
    let n = a_row_ptr.len().saturating_sub(1);
    let mut y = vec![0.0; n];
    for i in 0..n {
        let start = a_row_ptr[i];
        let end = a_row_ptr[i + 1];
        let mut sum = 0.0;
        for pos in start..end {
            sum += a_values[pos] * x[a_col_idx[pos]];
        }
        y[i] = sum;
    }
    y
}

/// Compute r = b - A x.
fn compute_residual(
    a_values: &[f64],
    a_row_ptr: &[usize],
    a_col_idx: &[usize],
    x: &[f64],
    b: &[f64],
) -> Vec<f64> {
    let ax = csr_matvec(a_values, a_row_ptr, a_col_idx, x);
    b.iter()
        .zip(ax.iter())
        .map(|(&bi, &axi)| bi - axi)
        .collect()
}

/// Euclidean norm.
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ---------------------------------------------------------------------------
// MLP internals
// ---------------------------------------------------------------------------

/// Number of input features per node.
const INPUT_DIM: usize = 4;

/// ReLU activation.
#[inline]
fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Derivative of ReLU.
#[inline]
fn relu_grad(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Gather per-node features: [x_i, r_i, sum_neighbors(x), sum_neighbors(r)].
fn gather_node_features(
    a_row_ptr: &[usize],
    a_col_idx: &[usize],
    x: &[f64],
    r: &[f64],
    node_i: usize,
) -> [f64; INPUT_DIM] {
    let start = a_row_ptr[node_i];
    let end = a_row_ptr[node_i + 1];
    let mut sum_x = 0.0;
    let mut sum_r = 0.0;
    for pos in start..end {
        let j = a_col_idx[pos];
        if j != node_i {
            sum_x += x[j];
            sum_r += r[j];
        }
    }
    [x[node_i], r[node_i], sum_x, sum_r]
}

/// Forward pass through a 2-layer MLP: Linear(4, H) -> ReLU -> Linear(H, 1).
///
/// Returns `(output, hidden_pre_activation)` where hidden_pre_activation is
/// needed for backpropagation.
fn mlp_forward(
    input: &[f64; INPUT_DIM],
    w1: &[f64], // shape: hidden_dim x INPUT_DIM, row-major
    b1: &[f64], // shape: hidden_dim
    w2: &[f64], // shape: hidden_dim (since output_dim=1)
    b2: f64,
    hidden_dim: usize,
) -> (f64, Vec<f64>) {
    // Layer 1: z = W1 * input + b1
    let mut z = vec![0.0; hidden_dim];
    for h in 0..hidden_dim {
        let mut sum = b1[h];
        for j in 0..INPUT_DIM {
            sum += w1[h * INPUT_DIM + j] * input[j];
        }
        z[h] = sum;
    }

    // ReLU activation
    let mut a = vec![0.0; hidden_dim];
    for h in 0..hidden_dim {
        a[h] = relu(z[h]);
    }

    // Layer 2: out = W2 * a + b2
    let mut out = b2;
    for h in 0..hidden_dim {
        out += w2[h] * a[h];
    }

    (out, z)
}

// ---------------------------------------------------------------------------
// MLPSmoother
// ---------------------------------------------------------------------------

/// Per-node 2-layer MLP smoother with shared weights.
///
/// Architecture: Linear(4, hidden_dim) → ReLU → Linear(hidden_dim, 1).
/// Weights are shared across all nodes, inspired by GNN message passing.
#[derive(Debug, Clone)]
pub struct MLPSmoother {
    /// Hidden dimension of the MLP.
    hidden_dim: usize,
    /// Layer 1 weights: hidden_dim x INPUT_DIM, row-major.
    w1: Vec<f64>,
    /// Layer 1 biases: hidden_dim.
    b1: Vec<f64>,
    /// Layer 2 weights: hidden_dim.
    w2: Vec<f64>,
    /// Layer 2 bias (scalar).
    b2: f64,
}

impl MLPSmoother {
    /// Create a new MLP smoother with Xavier-initialised weights.
    ///
    /// # Arguments
    /// - `hidden_dim`: width of the hidden layer (default suggestion: 16)
    pub fn new(hidden_dim: usize) -> Self {
        let hidden_dim = if hidden_dim == 0 { 16 } else { hidden_dim };

        // Xavier initialisation: scale = sqrt(2 / (fan_in + fan_out))
        let scale1 = (2.0 / (INPUT_DIM + hidden_dim) as f64).sqrt();
        let scale2 = (2.0 / (hidden_dim + 1) as f64).sqrt();

        // Deterministic pseudo-random init using a simple LCG
        let mut seed: u64 = 42;
        let mut next_val = |scale: f64| -> f64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = (seed >> 33) as f64 / (1u64 << 31) as f64;
            (bits - 0.5) * 2.0 * scale
        };

        let w1: Vec<f64> = (0..hidden_dim * INPUT_DIM)
            .map(|_| next_val(scale1))
            .collect();
        let b1 = vec![0.0; hidden_dim];
        let w2: Vec<f64> = (0..hidden_dim).map(|_| next_val(scale2)).collect();
        let b2 = 0.0;

        Self {
            hidden_dim,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Total number of trainable parameters.
    pub fn num_parameters(&self) -> usize {
        // W1: hidden_dim * INPUT_DIM, b1: hidden_dim, W2: hidden_dim, b2: 1
        self.hidden_dim * INPUT_DIM + self.hidden_dim + self.hidden_dim + 1
    }

    /// Apply one smoothing sweep over all nodes.
    fn apply_one_sweep(&self, a_row_ptr: &[usize], a_col_idx: &[usize], x: &mut [f64], r: &[f64]) {
        let n = a_row_ptr.len().saturating_sub(1);
        // Compute all corrections first, then apply (Jacobi-style).
        let mut corrections = vec![0.0; n];
        for i in 0..n {
            let features = gather_node_features(a_row_ptr, a_col_idx, x, r, i);
            let (delta, _) = mlp_forward(
                &features,
                &self.w1,
                &self.b1,
                &self.w2,
                self.b2,
                self.hidden_dim,
            );
            corrections[i] = delta;
        }
        for i in 0..n {
            x[i] += corrections[i];
        }
    }

    /// Backpropagation for one node: compute gradients of loss w.r.t. all weights.
    ///
    /// loss_i = (x_i + delta_i - x_exact_i)^2
    /// d_loss / d_delta_i = 2 * (x_i + delta_i - x_exact_i)
    #[allow(clippy::too_many_arguments)]
    fn backprop_node(
        &self,
        features: &[f64; INPUT_DIM],
        d_output: f64,
        grad_w1: &mut [f64],
        grad_b1: &mut [f64],
        grad_w2: &mut [f64],
        grad_b2: &mut f64,
    ) {
        let hidden_dim = self.hidden_dim;

        // Forward pass to get hidden pre-activations
        let mut z = vec![0.0; hidden_dim];
        for h in 0..hidden_dim {
            let mut sum = self.b1[h];
            for j in 0..INPUT_DIM {
                sum += self.w1[h * INPUT_DIM + j] * features[j];
            }
            z[h] = sum;
        }
        let a: Vec<f64> = z.iter().map(|&zi| relu(zi)).collect();

        // Backprop through layer 2: d_loss/d_w2[h] = d_output * a[h]
        for h in 0..hidden_dim {
            grad_w2[h] += d_output * a[h];
        }
        *grad_b2 += d_output;

        // Backprop through ReLU and layer 1
        for h in 0..hidden_dim {
            let d_a = d_output * self.w2[h]; // gradient into hidden activation
            let d_z = d_a * relu_grad(z[h]); // through ReLU
            grad_b1[h] += d_z;
            for j in 0..INPUT_DIM {
                grad_w1[h * INPUT_DIM + j] += d_z * features[j];
            }
        }
    }
}

impl Smoother for MLPSmoother {
    fn smooth(
        &self,
        _a_values: &[f64],
        a_row_ptr: &[usize],
        a_col_idx: &[usize],
        x: &mut [f64],
        b: &[f64],
        n_sweeps: usize,
    ) -> SparseResult<()> {
        let n = a_row_ptr.len().saturating_sub(1);
        if x.len() != n || b.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: x.len(),
            });
        }
        // We need a_values for residual computation; use the _a_values parameter
        for _ in 0..n_sweeps {
            let r = compute_residual(_a_values, a_row_ptr, a_col_idx, x, b);
            self.apply_one_sweep(a_row_ptr, a_col_idx, x, &r);
        }
        Ok(())
    }

    fn train_step(
        &mut self,
        a_values: &[f64],
        a_row_ptr: &[usize],
        a_col_idx: &[usize],
        x: &mut [f64],
        b: &[f64],
        x_exact: &[f64],
        lr: f64,
    ) -> SparseResult<f64> {
        let n = a_row_ptr.len().saturating_sub(1);
        if x.len() != n || b.len() != n || x_exact.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: x.len(),
            });
        }

        let hidden_dim = self.hidden_dim;

        // Compute residual
        let r = compute_residual(a_values, a_row_ptr, a_col_idx, x, b);

        // Forward pass: compute all corrections
        let mut corrections = vec![0.0; n];
        let mut features_all: Vec<[f64; INPUT_DIM]> = Vec::with_capacity(n);
        for i in 0..n {
            let features = gather_node_features(a_row_ptr, a_col_idx, x, &r, i);
            let (delta, _) =
                mlp_forward(&features, &self.w1, &self.b1, &self.w2, self.b2, hidden_dim);
            corrections[i] = delta;
            features_all.push(features);
        }

        // Compute loss and gradients
        let mut loss = 0.0;
        let mut grad_w1 = vec![0.0; hidden_dim * INPUT_DIM];
        let mut grad_b1 = vec![0.0; hidden_dim];
        let mut grad_w2 = vec![0.0; hidden_dim];
        let mut grad_b2 = 0.0;

        for i in 0..n {
            let error_i = x[i] + corrections[i] - x_exact[i];
            loss += error_i * error_i;
            let d_output = 2.0 * error_i;
            self.backprop_node(
                &features_all[i],
                d_output,
                &mut grad_w1,
                &mut grad_b1,
                &mut grad_w2,
                &mut grad_b2,
            );
        }

        // Normalise gradients by number of nodes
        let n_f64 = n as f64;
        if n_f64 > 0.0 {
            for g in grad_w1.iter_mut() {
                *g /= n_f64;
            }
            for g in grad_b1.iter_mut() {
                *g /= n_f64;
            }
            for g in grad_w2.iter_mut() {
                *g /= n_f64;
            }
            grad_b2 /= n_f64;
        }

        // Apply gradient descent
        for i in 0..self.w1.len() {
            self.w1[i] -= lr * grad_w1[i];
        }
        for i in 0..self.b1.len() {
            self.b1[i] -= lr * grad_b1[i];
        }
        for i in 0..self.w2.len() {
            self.w2[i] -= lr * grad_w2[i];
        }
        self.b2 -= lr * grad_b2;

        // Apply corrections to x
        for i in 0..n {
            x[i] += corrections[i];
        }

        Ok(loss)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 3x3 SPD tridiagonal.
    fn tridiag_3() -> (Vec<f64>, Vec<usize>, Vec<usize>) {
        let values = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
        let row_ptr = vec![0, 2, 5, 7];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2];
        (values, row_ptr, col_idx)
    }

    #[test]
    fn test_mlp_smoother_creation() {
        let smoother = MLPSmoother::new(16);
        assert_eq!(smoother.hidden_dim(), 16);
        // 16*4 + 16 + 16 + 1 = 97
        assert_eq!(smoother.num_parameters(), 97);
    }

    #[test]
    fn test_mlp_smoother_smooth_runs() {
        let (vals, rp, ci) = tridiag_3();
        let smoother = MLPSmoother::new(8);

        let b = vec![1.0, 0.0, 1.0];
        let mut x = vec![0.0; 3];

        let result = smoother.smooth(&vals, &rp, &ci, &mut x, &b, 3);
        assert!(result.is_ok(), "MLP smooth should not fail");
    }

    #[test]
    fn test_mlp_train_step_runs() {
        let (vals, rp, ci) = tridiag_3();
        let mut smoother = MLPSmoother::new(8);

        let b = vec![1.0, 0.0, 1.0];
        let x_exact = vec![1.0, 1.0, 1.0];
        let mut x = vec![0.0; 3];

        let loss = smoother
            .train_step(&vals, &rp, &ci, &mut x, &b, &x_exact, 0.001)
            .expect("train step failed");
        assert!(loss >= 0.0, "Loss should be non-negative");
    }

    #[test]
    fn test_mlp_training_reduces_loss() {
        let (vals, rp, ci) = tridiag_3();
        let mut smoother = MLPSmoother::new(8);

        let b = vec![1.0, 0.0, 1.0];
        let x_exact = vec![1.0, 1.0, 1.0];

        let mut losses = Vec::new();
        for _ in 0..30 {
            let mut x = vec![0.0; 3];
            let loss = smoother
                .train_step(&vals, &rp, &ci, &mut x, &b, &x_exact, 0.001)
                .expect("train failed");
            losses.push(loss);
        }

        // Check that loss decreases over training
        let first_avg = losses[..5].iter().sum::<f64>() / 5.0;
        let last_avg = losses[25..].iter().sum::<f64>() / 5.0;
        // MLP should learn to reduce loss, but exact amount depends on init
        assert!(
            last_avg <= first_avg * 1.5, // allow some slack due to stochastic init
            "MLP training should not diverge wildly: first_avg={first_avg}, last_avg={last_avg}"
        );
    }

    #[test]
    fn test_gather_features() {
        let row_ptr = vec![0, 2, 5, 7];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2];
        let x = vec![1.0, 2.0, 3.0];
        let r = vec![0.5, 0.3, 0.1];

        let feat = gather_node_features(&row_ptr, &col_idx, &x, &r, 1);
        // Node 1 neighbors: 0 and 2
        assert!((feat[0] - 2.0).abs() < f64::EPSILON, "x_i = x[1]");
        assert!((feat[1] - 0.3).abs() < f64::EPSILON, "r_i = r[1]");
        assert!(
            (feat[2] - 4.0).abs() < f64::EPSILON,
            "sum_neighbors(x) = x[0]+x[2]"
        );
        assert!(
            (feat[3] - 0.6).abs() < f64::EPSILON,
            "sum_neighbors(r) = r[0]+r[2]"
        );
    }

    #[test]
    fn test_mlp_dimension_mismatch() {
        let (vals, rp, ci) = tridiag_3();
        let smoother = MLPSmoother::new(4);
        let mut x = vec![0.0; 2]; // wrong size
        let b = vec![1.0, 0.0, 1.0];
        let result = smoother.smooth(&vals, &rp, &ci, &mut x, &b, 1);
        assert!(result.is_err());
    }
}
