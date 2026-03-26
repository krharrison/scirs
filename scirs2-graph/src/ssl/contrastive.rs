//! Graph Contrastive Learning: GraphCL (You et al. 2020) and SimGRACE (Xia 2022).
//!
//! ## GraphCL Augmentations
//!
//! GraphCL creates two augmented views of each graph and trains an encoder to
//! maximise agreement between the views using the **NT-Xent** (normalised
//! temperature-scaled cross-entropy) loss:
//!
//! ```text
//! L = -(1/2N) Σ_i [ log exp(s(z_i, z_i') / τ) / Σ_{k≠i} exp(s(z_i, z_k) / τ) ]
//! ```
//!
//! Supported augmentations:
//! - **Feature masking**: zero out a random fraction of node feature dimensions.
//! - **Edge perturbation**: randomly drop existing edges and/or insert new ones.
//!
//! ## SimGRACE
//!
//! SimGRACE (Xia et al. 2022) avoids explicit augmentations by creating the
//! second view through small Gaussian perturbations of the encoder weights.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{Rng, RngExt, SeedableRng};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for GraphCL-style contrastive learning.
#[derive(Debug, Clone)]
pub struct GraphClConfig {
    /// Temperature parameter τ in the NT-Xent loss.
    pub temperature: f64,
    /// Output dimension of the projection head.
    pub proj_dim: usize,
    /// Fraction of node feature dimensions to zero out in each view.
    pub mask_feature_rate: f64,
    /// Fraction of edges to randomly drop from the adjacency matrix.
    pub drop_edge_rate: f64,
    /// Fraction of non-edges to randomly add to the adjacency matrix.
    pub add_edge_rate: f64,
}

impl Default for GraphClConfig {
    fn default() -> Self {
        Self {
            temperature: 0.5,
            proj_dim: 128,
            mask_feature_rate: 0.1,
            drop_edge_rate: 0.1,
            add_edge_rate: 0.0,
        }
    }
}

// ============================================================================
// Augmentation functions
// ============================================================================

/// Feature-masking augmentation (GraphCL).
///
/// Independently zeros out each feature dimension of every node with
/// probability `mask_rate`.  When `mask_rate = 0.0` the input is returned
/// unchanged; when `mask_rate = 1.0` a zero matrix is returned.
///
/// # Arguments
/// * `features`  – node feature matrix `[n_nodes × feature_dim]`
/// * `mask_rate` – probability of zeroing each feature dimension
/// * `seed`      – RNG seed for reproducibility
pub fn augment_features(features: &Array2<f64>, mask_rate: f64, seed: u64) -> Array2<f64> {
    if mask_rate <= 0.0 {
        return features.clone();
    }
    if mask_rate >= 1.0 {
        return Array2::zeros(features.dim());
    }

    let mut rng = scirs2_core::random::ChaCha20Rng::seed_from_u64(seed);
    let mut out = features.clone();
    let (n_nodes, feat_dim) = features.dim();

    for i in 0..n_nodes {
        for j in 0..feat_dim {
            if rng.random::<f64>() < mask_rate {
                out[[i, j]] = 0.0;
            }
        }
    }
    out
}

/// Edge-perturbation augmentation (GraphCL).
///
/// For each existing edge, drops it with probability `drop_rate`.
/// For each non-edge, adds it with probability `add_rate`.
///
/// The returned matrix is forced to be symmetric (undirected graph).
///
/// # Arguments
/// * `adj`       – adjacency matrix `[n × n]` (any non-zero entry = edge)
/// * `drop_rate` – probability of removing an existing edge
/// * `add_rate`  – probability of adding a new edge between non-adjacent nodes
/// * `seed`      – RNG seed
pub fn augment_edges(adj: &Array2<f64>, drop_rate: f64, add_rate: f64, seed: u64) -> Array2<f64> {
    let n = adj.dim().0;
    let mut rng = scirs2_core::random::ChaCha20Rng::seed_from_u64(seed);
    let mut out = adj.clone();

    for i in 0..n {
        for j in (i + 1)..n {
            if adj[[i, j]] > 0.0 {
                // Existing edge: maybe drop
                if drop_rate > 0.0 && rng.random::<f64>() < drop_rate {
                    out[[i, j]] = 0.0;
                    out[[j, i]] = 0.0;
                }
            } else {
                // Non-edge: maybe add
                if add_rate > 0.0 && rng.random::<f64>() < add_rate {
                    out[[i, j]] = 1.0;
                    out[[j, i]] = 1.0;
                }
            }
        }
    }
    out
}

// ============================================================================
// NT-Xent loss
// ============================================================================

/// NT-Xent (normalised temperature-scaled cross-entropy) contrastive loss.
///
/// Given two batches of projected representations `z1` and `z2` (each of
/// shape `[batch × proj_dim]`), where `(z1[i], z2[i])` are positive pairs,
/// computes the symmetric InfoNCE loss over all `2N` samples.
///
/// # Arguments
/// * `z1`         – first-view projections `[N × D]`
/// * `z2`         – second-view projections `[N × D]`
/// * `temperature` – temperature τ > 0; lower values create sharper distributions
///
/// # Returns
/// Scalar loss value.
pub fn nt_xent_loss(z1: &Array2<f64>, z2: &Array2<f64>, temperature: f64) -> f64 {
    let (n, _d) = z1.dim();
    assert_eq!(z1.dim(), z2.dim(), "z1 and z2 must have the same shape");
    assert!(temperature > 0.0, "temperature must be positive");

    // L2-normalise each row
    let norm_z1 = l2_normalise_rows(z1);
    let norm_z2 = l2_normalise_rows(z2);

    // Stack: rows 0..N from z1, rows N..2N from z2  →  [2N × D]
    let mut stacked = Array2::zeros((2 * n, z1.dim().1));
    for i in 0..n {
        for d in 0..z1.dim().1 {
            stacked[[i, d]] = norm_z1[[i, d]];
            stacked[[i + n, d]] = norm_z2[[i, d]];
        }
    }

    // Compute full cosine similarity matrix [2N × 2N] / tau
    let two_n = 2 * n;
    let mut sim = Array2::zeros((two_n, two_n));
    for i in 0..two_n {
        for j in 0..two_n {
            let mut dot = 0.0;
            for d in 0..stacked.dim().1 {
                dot += stacked[[i, d]] * stacked[[j, d]];
            }
            sim[[i, j]] = dot / temperature;
        }
    }

    // Mask diagonal (self-similarity) with large negative value
    for i in 0..two_n {
        sim[[i, i]] = f64::NEG_INFINITY;
    }

    // Positive pair indices:
    //   for i in 0..N: positive = i+N
    //   for i in N..2N: positive = i-N
    let mut loss = 0.0;
    for i in 0..two_n {
        let pos_j = if i < n { i + n } else { i - n };
        let pos_score = sim[[i, pos_j]];

        // log-sum-exp over all non-self entries
        let row_scores: Vec<f64> = (0..two_n)
            .filter(|&j| j != i)
            .map(|j| sim[[i, j]])
            .collect();
        let max_s = row_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let log_sum_exp = max_s
            + row_scores
                .iter()
                .map(|&s| (s - max_s).exp())
                .sum::<f64>()
                .ln();

        loss += -(pos_score - log_sum_exp);
    }

    loss / two_n as f64
}

/// L2-normalise each row of a 2-D array.
fn l2_normalise_rows(x: &Array2<f64>) -> Array2<f64> {
    let norms: Array1<f64> = x.map_axis(Axis(1), |row| {
        let s: f64 = row.iter().map(|&v| v * v).sum();
        s.sqrt().max(1e-12)
    });
    let mut out = x.clone();
    let (n, _d) = x.dim();
    for i in 0..n {
        for d in 0.._d {
            out[[i, d]] /= norms[i];
        }
    }
    out
}

// ============================================================================
// Projection head
// ============================================================================

/// Two-layer MLP projection head used in contrastive learning.
///
/// Architecture: `in_dim → hidden_dim (ReLU) → out_dim`
pub struct ProjectionHead {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
}

impl ProjectionHead {
    /// Construct a new projection head with Xavier-uniform initialised weights.
    ///
    /// # Arguments
    /// * `in_dim`     – input dimension (encoder output size)
    /// * `hidden_dim` – hidden layer dimension
    /// * `out_dim`    – projection output dimension
    /// * `seed`       – RNG seed
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize, seed: u64) -> Self {
        let mut rng = scirs2_core::random::ChaCha20Rng::seed_from_u64(seed);

        let s1 = (6.0 / (in_dim + hidden_dim) as f64).sqrt();
        let w1 = Array2::from_shape_fn((in_dim, hidden_dim), |_| {
            rng.random::<f64>() * 2.0 * s1 - s1
        });
        let b1 = Array1::zeros(hidden_dim);

        let s2 = (6.0 / (hidden_dim + out_dim) as f64).sqrt();
        let w2 = Array2::from_shape_fn((hidden_dim, out_dim), |_| {
            rng.random::<f64>() * 2.0 * s2 - s2
        });
        let b2 = Array1::zeros(out_dim);

        ProjectionHead { w1, b1, w2, b2 }
    }

    /// Forward pass: `x → W1 x + b1 → ReLU → W2 x + b2`
    ///
    /// # Arguments
    /// * `x` – input `[batch × in_dim]`
    ///
    /// # Returns
    /// Projected representations `[batch × out_dim]`
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let batch = x.dim().0;
        let hidden_dim = self.w1.dim().1;
        let out_dim = self.w2.dim().1;

        // First linear layer + ReLU
        let mut h = Array2::zeros((batch, hidden_dim));
        for i in 0..batch {
            for j in 0..hidden_dim {
                let mut val = self.b1[j];
                for d in 0..x.dim().1 {
                    val += x[[i, d]] * self.w1[[d, j]];
                }
                h[[i, j]] = if val > 0.0 { val } else { 0.0 };
            }
        }

        // Second linear layer
        let mut out = Array2::zeros((batch, out_dim));
        for i in 0..batch {
            for k in 0..out_dim {
                let mut val = self.b2[k];
                for j in 0..hidden_dim {
                    val += h[[i, j]] * self.w2[[j, k]];
                }
                out[[i, k]] = val;
            }
        }

        out
    }

    /// Input dimension.
    pub fn in_dim(&self) -> usize {
        self.w1.dim().0
    }

    /// Output projection dimension.
    pub fn out_dim(&self) -> usize {
        self.w2.dim().1
    }
}

// ============================================================================
// SimGRACE perturbation
// ============================================================================

/// SimGRACE weight perturbation.
///
/// Creates a second view by adding Gaussian noise to a weight matrix,
/// effectively simulating a slightly different encoder without explicit
/// graph augmentation.
///
/// # Arguments
/// * `weights`     – weight matrix to perturb `[r × c]`
/// * `noise_scale` – standard deviation of the Gaussian perturbation
/// * `seed`        – RNG seed
///
/// # Returns
/// Perturbed weight matrix of the same shape.
pub fn simgrace_perturb(weights: &Array2<f64>, noise_scale: f64, seed: u64) -> Array2<f64> {
    let mut rng = scirs2_core::random::ChaCha20Rng::seed_from_u64(seed);
    weights.mapv(|v| {
        // Box-Muller for Gaussian noise
        let u1: f64 = rng.random::<f64>().max(1e-12);
        let u2: f64 = rng.random::<f64>();
        let noise = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
        v + noise_scale * noise
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_augment_features_zero_rate_identity() {
        let x = Array2::from_shape_vec((3, 4), (0..12).map(|v| v as f64).collect()).expect("ok");
        let out = augment_features(&x, 0.0, 0);
        for (a, b) in x.iter().zip(out.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_augment_features_full_rate_zeros() {
        let x = Array2::ones((5, 8));
        let out = augment_features(&x, 1.0, 0);
        for v in out.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_nt_xent_identical_views_low_loss() {
        // Identical (normalised) views → minimum possible loss (all positives perfectly aligned)
        let z = Array2::from_shape_fn((8, 16), |(i, j)| if i == j { 1.0 } else { 0.0 });
        let loss = nt_xent_loss(&z, &z, 0.5);
        // With perfectly aligned views the loss should be near -log(1/(2N-1)) / 2N
        // In practice it should be strictly positive (negatives still contribute)
        assert!(loss >= 0.0, "loss should be non-negative, got {loss}");
        // And lower than random baseline
        let mut rng = scirs2_core::random::ChaCha20Rng::seed_from_u64(0);
        let z_rand = Array2::from_shape_fn((8, 16), |_| rng.random::<f64>() - 0.5);
        let loss_rand = nt_xent_loss(&z_rand, &z_rand, 0.5);
        // Identical views have lower loss than random
        assert!(loss <= loss_rand + 1e-6);
    }

    #[test]
    fn test_nt_xent_random_views_positive_loss() {
        let mut rng = scirs2_core::random::ChaCha20Rng::seed_from_u64(42);
        let z1 = Array2::from_shape_fn((6, 8), |_| rng.random::<f64>() - 0.5);
        let z2 = Array2::from_shape_fn((6, 8), |_| rng.random::<f64>() - 0.5);
        let loss = nt_xent_loss(&z1, &z2, 0.5);
        assert!(
            loss > 0.0,
            "loss with random views should be positive, got {loss}"
        );
    }

    #[test]
    fn test_projection_head_output_shape() {
        let head = ProjectionHead::new(32, 64, 16, 0);
        let x = Array2::ones((10, 32));
        let out = head.forward(&x);
        assert_eq!(out.dim(), (10, 16));
    }

    #[test]
    fn test_projection_head_dims() {
        let head = ProjectionHead::new(32, 64, 16, 0);
        assert_eq!(head.in_dim(), 32);
        assert_eq!(head.out_dim(), 16);
    }

    #[test]
    fn test_simgrace_perturb_changes_weights() {
        let w = Array2::ones((8, 8));
        let perturbed = simgrace_perturb(&w, 0.1, 99);
        let diff: f64 = w
            .iter()
            .zip(perturbed.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-10,
            "perturbed weights should differ from original"
        );
    }

    #[test]
    fn test_simgrace_zero_noise_preserves_weights() {
        let w = Array2::ones((4, 4));
        let perturbed = simgrace_perturb(&w, 0.0, 0);
        for (a, b) in w.iter().zip(perturbed.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_augment_edges_symmetry() {
        // Build a symmetric adjacency matrix (path graph 0-1-2-3)
        let mut adj = Array2::zeros((4, 4));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 1]] = 1.0;
        adj[[2, 3]] = 1.0;
        adj[[3, 2]] = 1.0;

        let aug = augment_edges(&adj, 0.3, 0.1, 7);
        let n = 4;
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    aug[[i, j]],
                    aug[[j, i]],
                    "augmented adjacency must remain symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_temperature_sensitivity() {
        // Lower temperature → sharper distribution → different (usually lower) loss for aligned views
        let z = Array2::from_shape_fn((4, 8), |(i, j)| if i == j { 1.0 } else { 0.0 });
        let loss_low_t = nt_xent_loss(&z, &z, 0.1);
        let loss_high_t = nt_xent_loss(&z, &z, 2.0);
        // Both should be non-negative; they should differ
        assert!(loss_low_t >= 0.0);
        assert!(loss_high_t >= 0.0);
        assert!(
            (loss_low_t - loss_high_t).abs() > 1e-6,
            "temperature should affect loss magnitude"
        );
    }
}
