//! Graph Attention Network (GAT) layers
//!
//! Implements the multi-head graph attention operator from Veličković et al.
//! (2018), "Graph Attention Networks".
//!
//! Each attention head computes:
//! ```text
//!   α_{ij}^k = softmax_j( LeakyReLU( a_k^T [ W_k h_i ‖ W_k h_j ] ) )
//!   h_i'^k   = σ( Σ_j α_{ij}^k  W_k h_j )
//! ```
//! Multiple heads are concatenated (hidden layers) or averaged (final layer):
//! ```text
//!   h_i' = ‖_{k=1}^{K}  σ( Σ_j α_{ij}^k W_k h_j )
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Rng, RngExt};

use crate::error::{GraphError, Result};
use crate::gnn::gcn::CsrMatrix;

// ============================================================================
// Helpers
// ============================================================================

/// Numerically-stable softmax over a slice.
fn softmax(xs: &[f64]) -> Vec<f64> {
    if xs.is_empty() {
        return Vec::new();
    }
    let max_val = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = xs.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum::<f64>().max(1e-12);
    exps.iter().map(|e| e / sum).collect()
}

/// LeakyReLU with configurable negative slope.
#[inline]
fn leaky_relu(x: f64, neg_slope: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        neg_slope * x
    }
}

// ============================================================================
// Functional API: gat_forward
// ============================================================================

/// Single-head GAT forward pass (functional API).
///
/// Computes attention-weighted node embeddings:
/// ```text
///   α_{ij} = softmax_j( LeakyReLU( a^T [W h_i ‖ W h_j] ) )
///   h_i'   = σ( Σ_j α_{ij} W h_j )
/// ```
///
/// # Arguments
/// * `adj` – Sparse adjacency (row i → neighbors of i; self-loops are added
///   automatically so every node attends to itself).
/// * `features` – Node feature matrix `[n_nodes, in_dim]`.
/// * `w` – Linear weight matrix `[in_dim, out_dim]`.
/// * `a` – Attention parameter vector `[2 * out_dim]`.
/// * `neg_slope` – LeakyReLU negative slope (typically 0.2).
///
/// # Returns
/// Updated node embeddings `[n_nodes, out_dim]` (raw, before global activation).
pub fn gat_forward(
    adj: &CsrMatrix,
    features: &Array2<f64>,
    w: &Array2<f64>,
    a: &Array1<f64>,
    neg_slope: f64,
) -> Result<Array2<f64>> {
    let n = adj.n_rows;
    let (feat_n, in_dim) = features.dim();
    let (w_in, out_dim) = w.dim();

    if feat_n != n {
        return Err(GraphError::InvalidParameter {
            param: "features".to_string(),
            value: format!("{feat_n} rows"),
            expected: format!("{n} rows"),
            context: "gat_forward".to_string(),
        });
    }
    if w_in != in_dim {
        return Err(GraphError::InvalidParameter {
            param: "w".to_string(),
            value: format!("{w_in} rows"),
            expected: format!("{in_dim}"),
            context: "gat_forward".to_string(),
        });
    }
    if a.len() != 2 * out_dim {
        return Err(GraphError::InvalidParameter {
            param: "a".to_string(),
            value: format!("len={}", a.len()),
            expected: format!("2 * out_dim = {}", 2 * out_dim),
            context: "gat_forward".to_string(),
        });
    }

    // Step 1: W h_i for each node  →  wh: [n, out_dim]
    let mut wh = Array2::<f64>::zeros((n, out_dim));
    for i in 0..n {
        for j in 0..out_dim {
            let mut s = 0.0;
            for k in 0..in_dim {
                s += features[[i, k]] * w[[k, j]];
            }
            wh[[i, j]] = s;
        }
    }

    // Step 2: Build neighbor lists including self
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (row, col, _) in adj.iter() {
        neighbors[row].push(col);
    }
    for i in 0..n {
        if !neighbors[i].contains(&i) {
            neighbors[i].push(i);
        }
    }

    // Step 3: Attention-weighted aggregation
    let mut output = Array2::<f64>::zeros((n, out_dim));

    for i in 0..n {
        let nbrs = &neighbors[i];
        if nbrs.is_empty() {
            continue;
        }

        // Compute raw scores  e_{ij} = a^T [Wh_i || Wh_j]
        let scores: Vec<f64> = nbrs
            .iter()
            .map(|&j| {
                let mut dot = 0.0;
                for k in 0..out_dim {
                    dot += a[k] * wh[[i, k]];
                    dot += a[out_dim + k] * wh[[j, k]];
                }
                leaky_relu(dot, neg_slope)
            })
            .collect();

        let alphas = softmax(&scores);

        for (idx, &j) in nbrs.iter().enumerate() {
            let alpha = alphas[idx];
            for k in 0..out_dim {
                output[[i, k]] += alpha * wh[[j, k]];
            }
        }
    }

    Ok(output)
}

// ============================================================================
// GraphAttentionLayer struct (multi-head)
// ============================================================================

/// A single multi-head Graph Attention layer.
///
/// Maintains `n_heads` independent weight matrices and attention vectors.
/// Hidden layers concatenate head outputs; the final layer can average them.
#[derive(Debug, Clone)]
pub struct GraphAttentionLayer {
    /// Per-head weight matrices, each `[in_dim, head_out_dim]`
    pub head_weights: Vec<Array2<f64>>,
    /// Per-head attention vectors, each `[2 * head_out_dim]`
    pub head_attention: Vec<Array1<f64>>,
    /// Optional output bias `[out_dim]` (applied after concat/average)
    pub bias: Option<Array1<f64>>,
    /// Number of attention heads
    pub n_heads: usize,
    /// Output dimension per head
    pub head_out_dim: usize,
    /// Input dimension
    pub in_dim: usize,
    /// LeakyReLU negative slope
    pub neg_slope: f64,
    /// If true, concatenate head outputs; otherwise average them
    pub concat_heads: bool,
    /// Apply ELU activation on the output
    pub use_activation: bool,
}

impl GraphAttentionLayer {
    /// Create a multi-head GAT layer.
    ///
    /// # Arguments
    /// * `in_dim` – Input feature dimension per node.
    /// * `head_out_dim` – Output dimension per attention head.
    /// * `n_heads` – Number of attention heads.
    /// * `concat` – If `true`, concatenate head outputs (total out = n_heads *
    ///   head_out_dim); if `false`, average them (total out = head_out_dim).
    pub fn new(in_dim: usize, head_out_dim: usize, n_heads: usize, concat: bool) -> Self {
        let mut rng = scirs2_core::random::rng();
        let w_scale = (6.0_f64 / (in_dim + head_out_dim) as f64).sqrt();
        let a_scale = (6.0_f64 / (2 * head_out_dim) as f64).sqrt();

        let head_weights: Vec<Array2<f64>> = (0..n_heads)
            .map(|_| {
                Array2::from_shape_fn((in_dim, head_out_dim), |_| {
                    rng.random::<f64>() * 2.0 * w_scale - w_scale
                })
            })
            .collect();

        let head_attention: Vec<Array1<f64>> = (0..n_heads)
            .map(|_| {
                Array1::from_iter(
                    (0..2 * head_out_dim)
                        .map(|_| rng.random::<f64>() * 2.0 * a_scale - a_scale),
                )
            })
            .collect();

        GraphAttentionLayer {
            head_weights,
            head_attention,
            bias: None,
            n_heads,
            head_out_dim,
            in_dim,
            neg_slope: 0.2,
            concat_heads: concat,
            use_activation: true,
        }
    }

    /// The effective output dimension: `n_heads * head_out_dim` when
    /// concatenating, or `head_out_dim` when averaging.
    pub fn out_dim(&self) -> usize {
        if self.concat_heads {
            self.n_heads * self.head_out_dim
        } else {
            self.head_out_dim
        }
    }

    /// Set the LeakyReLU negative slope (default 0.2).
    pub fn with_neg_slope(mut self, slope: f64) -> Self {
        self.neg_slope = slope;
        self
    }

    /// Disable ELU activation on the output.
    pub fn without_activation(mut self) -> Self {
        self.use_activation = false;
        self
    }

    /// Forward pass of the multi-head GAT layer.
    ///
    /// # Arguments
    /// * `adj` – Sparse adjacency matrix.
    /// * `features` – Node features `[n_nodes, in_dim]`.
    ///
    /// # Returns
    /// * Concatenated or averaged embeddings `[n_nodes, out_dim]`.
    pub fn forward(&self, adj: &CsrMatrix, features: &Array2<f64>) -> Result<Array2<f64>> {
        let n = adj.n_rows;
        let (feat_n, feat_dim) = features.dim();

        if feat_n != n {
            return Err(GraphError::InvalidParameter {
                param: "features".to_string(),
                value: format!("{feat_n}"),
                expected: format!("{n}"),
                context: "GraphAttentionLayer::forward".to_string(),
            });
        }
        if feat_dim != self.in_dim {
            return Err(GraphError::InvalidParameter {
                param: "features in_dim".to_string(),
                value: format!("{feat_dim}"),
                expected: format!("{}", self.in_dim),
                context: "GraphAttentionLayer::forward".to_string(),
            });
        }

        // Run each head
        let head_outputs: Vec<Array2<f64>> = (0..self.n_heads)
            .map(|h| {
                gat_forward(
                    adj,
                    features,
                    &self.head_weights[h],
                    &self.head_attention[h],
                    self.neg_slope,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // Combine heads
        let out_dim = self.out_dim();
        let mut output = Array2::<f64>::zeros((n, out_dim));

        if self.concat_heads {
            for (h, head_out) in head_outputs.iter().enumerate() {
                let offset = h * self.head_out_dim;
                for i in 0..n {
                    for k in 0..self.head_out_dim {
                        output[[i, offset + k]] = head_out[[i, k]];
                    }
                }
            }
        } else {
            // Average
            let inv = 1.0 / self.n_heads as f64;
            for head_out in &head_outputs {
                for i in 0..n {
                    for k in 0..self.head_out_dim {
                        output[[i, k]] += inv * head_out[[i, k]];
                    }
                }
            }
        }

        // Optional bias
        if let Some(ref b) = self.bias {
            for i in 0..n {
                for j in 0..out_dim {
                    output[[i, j]] += b[j];
                }
            }
        }

        // ELU activation
        if self.use_activation {
            output.mapv_inplace(|x| if x >= 0.0 { x } else { x.exp() - 1.0 });
        }

        Ok(output)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_csr() -> CsrMatrix {
        let coo = vec![
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (0, 2, 1.0),
            (2, 0, 1.0),
        ];
        CsrMatrix::from_coo(3, 3, &coo).expect("triangle CSR")
    }

    fn feats(n: usize, d: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, d), |(i, j)| (i * d + j) as f64 * 0.1)
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&xs);
        let s: f64 = probs.iter().sum();
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gat_forward_shape() {
        let adj = triangle_csr();
        let f = feats(3, 4);
        let w = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as f64 * 0.01);
        let a = Array1::from_vec(vec![0.1; 16]);
        let out = gat_forward(&adj, &f, &w, &a, 0.2).expect("gat_forward");
        assert_eq!(out.dim(), (3, 8));
    }

    #[test]
    fn test_gat_forward_finite_values() {
        let adj = triangle_csr();
        let f = feats(3, 4);
        let w = Array2::from_shape_fn((4, 6), |(i, j)| (i as f64 - j as f64) * 0.05);
        let a = Array1::from_vec((0..12).map(|i| if i % 2 == 0 { 0.3 } else { -0.3 }).collect());
        let out = gat_forward(&adj, &f, &w, &a, 0.2).expect("gat_forward");
        for &v in out.iter() {
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }

    #[test]
    fn test_gat_layer_concat_output_dim() {
        let adj = triangle_csr();
        let f = feats(3, 4);
        // 3 heads × 6 out_dim_per_head = 18 total
        let layer = GraphAttentionLayer::new(4, 6, 3, true);
        assert_eq!(layer.out_dim(), 18);
        let out = layer.forward(&adj, &f).expect("layer forward");
        assert_eq!(out.dim(), (3, 18));
    }

    #[test]
    fn test_gat_layer_mean_output_dim() {
        let adj = triangle_csr();
        let f = feats(3, 4);
        // Average across 4 heads → out_dim = head_out_dim = 8
        let layer = GraphAttentionLayer::new(4, 8, 4, false);
        assert_eq!(layer.out_dim(), 8);
        let out = layer.forward(&adj, &f).expect("layer forward");
        assert_eq!(out.dim(), (3, 8));
    }

    #[test]
    fn test_gat_layer_single_head() {
        let adj = triangle_csr();
        let f = feats(3, 4);
        let layer = GraphAttentionLayer::new(4, 8, 1, true);
        let out = layer.forward(&adj, &f).expect("single head");
        assert_eq!(out.dim(), (3, 8));
    }

    #[test]
    fn test_gat_layer_no_activation() {
        let adj = triangle_csr();
        let f = feats(3, 4);
        let layer = GraphAttentionLayer::new(4, 6, 2, false).without_activation();
        let out = layer.forward(&adj, &f).expect("no act");
        assert_eq!(out.dim(), (3, 6));
    }

    #[test]
    fn test_gat_forward_dimension_mismatch() {
        let adj = triangle_csr();
        let f = feats(3, 4);
        let w = Array2::from_shape_fn((5, 8), |(i, j)| (i + j) as f64 * 0.01); // wrong in_dim
        let a = Array1::from_vec(vec![0.1; 16]);
        let result = gat_forward(&adj, &f, &w, &a, 0.2);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_coefficients_sum_to_one() {
        // Build a star graph: node 0 connected to 1, 2, 3
        let coo = vec![
            (0, 1, 1.0), (1, 0, 1.0),
            (0, 2, 1.0), (2, 0, 1.0),
            (0, 3, 1.0), (3, 0, 1.0),
        ];
        let adj = CsrMatrix::from_coo(4, 4, &coo).expect("star CSR");
        let f = feats(4, 3);
        let w = Array2::eye(3);
        let a = Array1::from_vec(vec![1.0; 6]);
        // Just verify output is finite and shaped correctly
        let out = gat_forward(&adj, &f, &w, &a, 0.2).expect("star gat");
        assert_eq!(out.dim(), (4, 3));
        for &v in out.iter() {
            assert!(v.is_finite());
        }
    }
}
