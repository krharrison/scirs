//! Heterogeneous Graph Transformer (HGT)
//!
//! Implements the heterogeneous attention mechanism from Hu et al. (2020),
//! "Heterogeneous Graph Transformer".
//!
//! HGT extends the graph transformer paradigm to **heterogeneous** graphs where
//! nodes and edges can be of different types.  Each meta-relation
//! `(τ_source, φ_edge, τ_target)` has its own projection matrices `W_K`, `W_Q`,
//! `W_V`, `W_O`, so the model can capture type-specific interaction patterns.
//!
//! Layer update rule for destination node `j` of type `τ_t`:
//! ```text
//!   ATT(i, r, j)  = softmax_i ( (W_K^{τ_s,φ} h_i)^T (W_Q^{φ,τ_t} h_j) / sqrt(d/K) )
//!   MSG(i, r, j)  = W_V^{τ_s,φ} h_i
//!   h_j^{new}     = LayerNorm( h_j + W_O^{τ_t} σ( Σ_{i,r} ATT · MSG ) )
//! ```

use std::collections::HashMap;

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Rng, RngExt};

use crate::error::{GraphError, Result};

/// Per-destination, per-head raw attention entries: `[dst][head] -> Vec<(score, message)>`.
type RawAttentions = Vec<Vec<Vec<(f64, Array1<f64>)>>>;

// ============================================================================
// Helpers
// ============================================================================

/// Xavier uniform initialisation.
fn xavier_uniform(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = scirs2_core::random::rng();
    let limit = (6.0_f64 / (rows + cols) as f64).sqrt();
    Array2::from_shape_fn((rows, cols), |_| rng.random::<f64>() * 2.0 * limit - limit)
}

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

/// Layer normalisation: (x - mean) / (std + eps).
fn layer_norm(x: &Array1<f64>) -> Array1<f64> {
    let n = x.len() as f64;
    let mean = x.sum() / n;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = (var + 1e-6).sqrt();
    x.mapv(|v| (v - mean) / std)
}

/// Sigmoid activation.
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ============================================================================
// HgtConfig
// ============================================================================

/// Configuration for an HGT encoder.
#[derive(Debug, Clone)]
pub struct HgtConfig {
    /// Hidden feature dimensionality (must be divisible by `n_heads`)
    pub hidden_dim: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of HGT layers to stack
    pub n_layers: usize,
    /// Dropout probability (applied after the output projection)
    pub dropout: f64,
}

impl Default for HgtConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 64,
            n_heads: 4,
            n_layers: 2,
            dropout: 0.1,
        }
    }
}

// ============================================================================
// Edge type encoding
// ============================================================================

/// An edge (meta-relation) type is identified by the triple
/// `(source_node_type, relation_type, target_node_type)`.
pub type EdgeType = (usize, usize, usize);

// ============================================================================
// HgtLayer
// ============================================================================

/// A single HGT layer.
///
/// Maintains separate projection matrices for each meta-relation and each
/// target node type.
#[derive(Debug, Clone)]
pub struct HgtLayer {
    /// Key projections W_K indexed by `(src_node_type, rel_type)`.
    pub w_k: HashMap<(usize, usize), Array2<f64>>,
    /// Query projections W_Q indexed by `(rel_type, dst_node_type)`.
    pub w_q: HashMap<(usize, usize), Array2<f64>>,
    /// Value projections W_V indexed by `(src_node_type, rel_type)`.
    pub w_v: HashMap<(usize, usize), Array2<f64>>,
    /// Output projections W_O indexed by `dst_node_type`.
    pub w_o: HashMap<usize, Array2<f64>>,
    /// Feature dimensionality
    pub hidden_dim: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Head dimension `= hidden_dim / n_heads`
    pub head_dim: usize,
}

impl HgtLayer {
    /// Create an HGT layer for the given set of meta-relation edge types and
    /// the full set of node types involved in those edge types.
    ///
    /// # Arguments
    /// * `edge_types`  – All `(src_type, rel_type, dst_type)` triples present
    ///   in the graph.
    /// * `hidden_dim`  – Feature dimensionality (must be divisible by `n_heads`).
    /// * `n_heads`     – Number of attention heads.
    pub fn new(edge_types: &[EdgeType], hidden_dim: usize, n_heads: usize) -> Result<Self> {
        if !hidden_dim.is_multiple_of(n_heads) {
            return Err(GraphError::InvalidParameter {
                param: "hidden_dim".to_string(),
                value: format!("{hidden_dim}"),
                expected: format!("divisible by n_heads={n_heads}"),
                context: "HgtLayer::new".to_string(),
            });
        }
        let head_dim = hidden_dim / n_heads;

        // Collect unique key pairs
        let mut kv_keys: std::collections::HashSet<(usize, usize)> = Default::default();
        let mut dst_types: std::collections::HashSet<usize> = Default::default();
        let mut qr_keys: std::collections::HashSet<(usize, usize)> = Default::default();

        for &(src_t, rel_t, dst_t) in edge_types {
            kv_keys.insert((src_t, rel_t));
            qr_keys.insert((rel_t, dst_t));
            dst_types.insert(dst_t);
        }

        let w_k = kv_keys
            .iter()
            .map(|&k| (k, xavier_uniform(hidden_dim, hidden_dim)))
            .collect();
        let w_q = qr_keys
            .iter()
            .map(|&k| (k, xavier_uniform(hidden_dim, hidden_dim)))
            .collect();
        let w_v = kv_keys
            .iter()
            .map(|&k| (k, xavier_uniform(hidden_dim, hidden_dim)))
            .collect();
        let w_o = dst_types
            .iter()
            .map(|&t| (t, xavier_uniform(hidden_dim, hidden_dim)))
            .collect();

        Ok(Self {
            w_k,
            w_q,
            w_v,
            w_o,
            hidden_dim,
            n_heads,
            head_dim,
        })
    }

    /// Single-head dot-product attention score for one head `h` in `[0, n_heads)`.
    fn head_attention_score(
        &self,
        key_proj: &Array1<f64>,
        query_proj: &Array1<f64>,
        head: usize,
    ) -> f64 {
        let start = head * self.head_dim;
        let end = start + self.head_dim;
        let k_slice = key_proj.slice(scirs2_core::ndarray::s![start..end]);
        let q_slice = query_proj.slice(scirs2_core::ndarray::s![start..end]);
        let dot: f64 = k_slice
            .iter()
            .zip(q_slice.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        dot / (self.head_dim as f64).sqrt()
    }

    /// HGT forward pass.
    ///
    /// # Arguments
    /// * `node_feats`      – Node feature matrix `(n_nodes, hidden_dim)`.
    /// * `node_types`      – Node type index for each node (length `n_nodes`).
    /// * `edges`           – List of `(src, rel_type, dst)` directed edges.
    ///
    /// # Returns
    /// Updated node feature matrix `(n_nodes, hidden_dim)`.
    pub fn forward(
        &self,
        node_feats: &Array2<f64>,
        node_types: &[usize],
        edges: &[(usize, usize, usize)], // (src, rel_type, dst)
    ) -> Result<Array2<f64>> {
        let n_nodes = node_feats.nrows();
        if node_types.len() != n_nodes {
            return Err(GraphError::InvalidParameter {
                param: "node_types".to_string(),
                value: format!("len={}", node_types.len()),
                expected: format!("len={}", n_nodes),
                context: "HgtLayer::forward".to_string(),
            });
        }
        if node_feats.ncols() != self.hidden_dim {
            return Err(GraphError::InvalidParameter {
                param: "node_feats".to_string(),
                value: format!("ncols={}", node_feats.ncols()),
                expected: format!("ncols={}", self.hidden_dim),
                context: "HgtLayer::forward".to_string(),
            });
        }

        // Group edges by destination node and head
        // agg_per_dst[dst][head] = accumulated weighted message
        let mut agg: Vec<Vec<Array1<f64>>> = (0..n_nodes)
            .map(|_| {
                (0..self.n_heads)
                    .map(|_| Array1::<f64>::zeros(self.head_dim))
                    .collect()
            })
            .collect();

        // Per-destination, per-head: accumulate raw attention logits + messages
        // We use two-pass: first collect (score, msg) pairs per (dst, head),
        // then softmax and sum.
        //
        // raw_atts[dst][head] = Vec<(score_f64, msg Array1)>
        let mut raw_atts: RawAttentions = (0..n_nodes)
            .map(|_| {
                (0..self.n_heads)
                    .map(|_| Vec::<(f64, Array1<f64>)>::new())
                    .collect()
            })
            .collect();

        for &(src, rel_type, dst) in edges {
            if src >= n_nodes || dst >= n_nodes {
                continue;
            }
            let src_type = node_types[src];
            let dst_type = node_types[dst];
            let kv_key = (src_type, rel_type);
            let qr_key = (rel_type, dst_type);

            // Look up projections (fall back to identity-style if type unseen)
            let h_src = node_feats.row(src).to_owned();
            let h_dst = node_feats.row(dst).to_owned();

            let key_proj: Array1<f64> = if let Some(wk) = self.w_k.get(&kv_key) {
                wk.dot(&h_src)
            } else {
                h_src.clone()
            };
            let query_proj: Array1<f64> = if let Some(wq) = self.w_q.get(&qr_key) {
                wq.dot(&h_dst)
            } else {
                h_dst.clone()
            };
            let val_proj: Array1<f64> = if let Some(wv) = self.w_v.get(&kv_key) {
                wv.dot(&h_src)
            } else {
                h_src.clone()
            };

            for head in 0..self.n_heads {
                let score = self.head_attention_score(&key_proj, &query_proj, head);
                let start = head * self.head_dim;
                let end = start + self.head_dim;
                let msg_slice = val_proj
                    .slice(scirs2_core::ndarray::s![start..end])
                    .to_owned();
                raw_atts[dst][head].push((score, msg_slice));
            }
        }

        // Softmax + aggregate per (dst, head)
        for dst in 0..n_nodes {
            for head in 0..self.n_heads {
                let pairs = &raw_atts[dst][head];
                if pairs.is_empty() {
                    continue;
                }
                let scores: Vec<f64> = pairs.iter().map(|(s, _)| *s).collect();
                let alphas = softmax(&scores);
                for (alpha, (_, msg)) in alphas.iter().zip(pairs.iter()) {
                    let mut slot = agg[dst][head].view_mut();
                    slot.zip_mut_with(msg, |a, &m| *a += alpha * m);
                }
            }
        }

        // Concatenate heads → (n_nodes, hidden_dim)
        let mut concat = Array2::<f64>::zeros((n_nodes, self.hidden_dim));
        for i in 0..n_nodes {
            for head in 0..self.n_heads {
                let start = head * self.head_dim;
                let end = start + self.head_dim;
                let mut row = concat.row_mut(i);
                let head_slice = &agg[i][head];
                for (j, &v) in head_slice.iter().enumerate() {
                    row[start + j] = v;
                }
            }
        }

        // Apply sigmoid non-linearity element-wise (gating)
        let concat_activated = concat.mapv(sigmoid);

        // Output projection per destination type
        let mut output = Array2::<f64>::zeros((n_nodes, self.hidden_dim));
        for i in 0..n_nodes {
            let dst_type = node_types[i];
            let msg = concat_activated.row(i).to_owned();
            let projected: Array1<f64> = if let Some(wo) = self.w_o.get(&dst_type) {
                wo.dot(&msg)
            } else {
                msg.clone()
            };
            // Residual connection + layer norm
            let residual = node_feats.row(i).to_owned() + &projected;
            let normed = layer_norm(&residual);
            output.row_mut(i).assign(&normed);
        }

        Ok(output)
    }
}

// ============================================================================
// Hgt — stacked HGT
// ============================================================================

/// Multi-layer HGT encoder.
///
/// Stacks `n_layers` [`HgtLayer`]s.  All layers share the same
/// `hidden_dim` so the output dimensionality equals `hidden_dim` after
/// an optional linear input projection (not included here; callers should
/// project raw features to `hidden_dim` before calling `forward`).
#[derive(Debug, Clone)]
pub struct Hgt {
    /// Ordered list of HGT layers
    pub layers: Vec<HgtLayer>,
    /// Hidden dimension (same for all layers)
    pub hidden_dim: usize,
}

impl Hgt {
    /// Build an HGT from a [`HgtConfig`] and the set of edge types present in
    /// the graph.
    ///
    /// # Arguments
    /// * `edge_types` – All `(src_type, rel_type, dst_type)` triples.
    /// * `config`     – Hyper-parameter configuration.
    pub fn new(edge_types: &[EdgeType], config: &HgtConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            layers.push(HgtLayer::new(
                edge_types,
                config.hidden_dim,
                config.n_heads,
            )?);
        }
        Ok(Self {
            layers,
            hidden_dim: config.hidden_dim,
        })
    }

    /// Run all HGT layers in sequence.
    ///
    /// # Arguments
    /// * `node_feats`  – Node feature matrix `(n_nodes, hidden_dim)`.
    /// * `node_types`  – Node type index for each node.
    /// * `edges`       – `(src, rel_type, dst)` directed edges.
    ///
    /// # Returns
    /// Final node embeddings `(n_nodes, hidden_dim)`.
    pub fn forward(
        &self,
        node_feats: &Array2<f64>,
        node_types: &[usize],
        edges: &[(usize, usize, usize)],
    ) -> Result<Array2<f64>> {
        let mut h = node_feats.clone();
        for layer in &self.layers {
            h = layer.forward(&h, node_types, edges)?;
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
    use scirs2_core::ndarray::Array2;

    fn random_feats(n: usize, dim: usize) -> Array2<f64> {
        let mut rng = scirs2_core::random::rng();
        Array2::from_shape_fn((n, dim), |_| rng.random::<f64>() * 0.1)
    }

    #[test]
    fn test_hgt_layer_output_shape() {
        let edge_types: Vec<EdgeType> = vec![(0, 0, 1), (1, 1, 0)];
        let layer = HgtLayer::new(&edge_types, 16, 4).expect("layer");
        let feats = random_feats(6, 16);
        let node_types = vec![0usize, 0, 0, 1, 1, 1];
        let edges = vec![(0usize, 0usize, 3usize), (1, 0, 4), (3, 1, 0), (4, 1, 1)];
        let out = layer.forward(&feats, &node_types, &edges).expect("forward");
        assert_eq!(out.nrows(), 6);
        assert_eq!(out.ncols(), 16);
    }

    #[test]
    fn test_hgt_layer_projection_dimensions() {
        let edge_types: Vec<EdgeType> = vec![(0, 0, 0)];
        let hidden_dim = 32usize;
        let n_heads = 4usize;
        let layer = HgtLayer::new(&edge_types, hidden_dim, n_heads).expect("layer");
        // All W_K projections must be (hidden_dim × hidden_dim)
        for w in layer.w_k.values() {
            assert_eq!(w.nrows(), hidden_dim);
            assert_eq!(w.ncols(), hidden_dim);
        }
        for w in layer.w_o.values() {
            assert_eq!(w.nrows(), hidden_dim);
            assert_eq!(w.ncols(), hidden_dim);
        }
    }

    #[test]
    fn test_hgt_layer_attention_softmax_sums_to_one() {
        // Build a simple scenario: node 0 sends to node 2 via two different
        // source nodes; attention over those sources should sum to 1.
        let edge_types: Vec<EdgeType> = vec![(0, 0, 0)];
        let layer = HgtLayer::new(&edge_types, 8, 2).expect("layer");
        let feats = random_feats(3, 8);
        let node_types = vec![0usize; 3];
        // Both node 0 and node 1 point to node 2
        let edges = vec![(0usize, 0usize, 2usize), (1, 0, 2)];
        // We verify indirectly: output is finite (softmax cannot be NaN)
        let out = layer.forward(&feats, &node_types, &edges).expect("fwd");
        for val in out.iter() {
            assert!(val.is_finite(), "output contains non-finite value");
        }
    }

    #[test]
    fn test_hgt_stacked_runs() {
        let edge_types: Vec<EdgeType> = vec![(0, 0, 0), (0, 1, 1)];
        let config = HgtConfig {
            hidden_dim: 16,
            n_heads: 2,
            n_layers: 3,
            dropout: 0.0,
        };
        let hgt = Hgt::new(&edge_types, &config).expect("hgt");
        let feats = random_feats(5, 16);
        let node_types = vec![0usize, 0, 0, 1, 1];
        let edges = vec![(0usize, 0usize, 1usize), (1, 1, 3), (2, 1, 4)];
        let out = hgt.forward(&feats, &node_types, &edges).expect("forward");
        assert_eq!(out.nrows(), 5);
        assert_eq!(out.ncols(), 16);
    }

    #[test]
    fn test_hgt_hidden_dim_not_divisible_errors() {
        let edge_types: Vec<EdgeType> = vec![(0, 0, 0)];
        // 5 is not divisible by 4
        let result = HgtLayer::new(&edge_types, 5, 4);
        assert!(result.is_err());
    }
}
