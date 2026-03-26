//! GraphGPS: General, Powerful, Scalable Graph Transformers
//!
//! Implements Rampasek et al. 2022 "Recipe for a General, Powerful, Scalable
//! Graph Transformer".  Each GPS layer combines:
//! - **Local MPNN**: message-passing over edges (mean aggregation + linear)
//! - **Global Transformer**: full self-attention over all nodes with PE bias
//! - **Combination gate**: learned α balances local vs. global branch
//! - **FFN**: two-layer MLP with GELU activation

use crate::error::{GraphError, Result};

use super::types::{GraphForTransformer, GraphTransformerConfig, GraphTransformerOutput};

// ============================================================================
// Activation helpers
// ============================================================================

/// GELU activation (tanh approximation)
#[inline]
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x)).tanh())
}

/// Layer normalisation: (x - mean) / std with learned γ=1, β=0
fn layer_norm(x: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    if n == 0.0 {
        return Vec::new();
    }
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let std = (var + 1e-6).sqrt();
    x.iter().map(|v| (v - mean) / std).collect()
}

/// Softmax over a slice
fn softmax(xs: &[f64]) -> Vec<f64> {
    if xs.is_empty() {
        return Vec::new();
    }
    let max_v = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = xs.iter().map(|&v| (v - max_v).exp()).collect();
    let sum = exps.iter().sum::<f64>().max(1e-15);
    exps.iter().map(|e| e / sum).collect()
}

/// Dense matrix–vector multiply
fn mv(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Element-wise vector addition
fn vadd(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

// ============================================================================
// Learnable weight matrices — initialised with deterministic LCG values
// ============================================================================

/// A simple LCG pseudo-random number generator for weight initialisation.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x5851_f42d_4c95_7f2d,
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.state >> 33) as i32;
        (bits as f64) / (i32::MAX as f64)
    }

    /// He initialisation scale: sqrt(2 / fan_in)
    fn he_matrix(&mut self, rows: usize, cols: usize) -> Vec<Vec<f64>> {
        let scale = (2.0 / cols as f64).sqrt();
        (0..rows)
            .map(|_| (0..cols).map(|_| self.next_f64() * scale).collect())
            .collect()
    }
}

// ============================================================================
// GpsLayer
// ============================================================================

/// A single GPS layer: parallel MPNN + Transformer + FFN.
struct GpsLayer {
    hidden_dim: usize,
    n_heads: usize,
    pe_dim: usize,

    // Local MPNN branch
    w_msg: Vec<Vec<f64>>, // hidden_dim × hidden_dim

    // Global Transformer: Q, K, V projections per head
    wq: Vec<Vec<Vec<f64>>>, // n_heads × head_dim × (hidden_dim + pe_dim)
    wk: Vec<Vec<Vec<f64>>>,
    wv: Vec<Vec<Vec<f64>>>,
    w_out: Vec<Vec<f64>>, // hidden_dim × hidden_dim  (concat head outputs → hidden)

    // Combination gate parameter (α initialised to 0.5)
    alpha: f64,

    // FFN
    w_ff1: Vec<Vec<f64>>, // 4*hidden_dim × hidden_dim
    w_ff2: Vec<Vec<f64>>, // hidden_dim × 4*hidden_dim
}

impl GpsLayer {
    fn new(hidden_dim: usize, n_heads: usize, pe_dim: usize, seed: u64) -> Self {
        let head_dim = (hidden_dim / n_heads).max(1);
        let in_dim = hidden_dim + pe_dim;
        let mut lcg = Lcg::new(seed);

        let wq = (0..n_heads)
            .map(|_| lcg.he_matrix(head_dim, in_dim))
            .collect();
        let wk = (0..n_heads)
            .map(|_| lcg.he_matrix(head_dim, in_dim))
            .collect();
        let wv = (0..n_heads)
            .map(|_| lcg.he_matrix(head_dim, hidden_dim))
            .collect();
        let w_out = lcg.he_matrix(hidden_dim, hidden_dim);
        let w_msg = lcg.he_matrix(hidden_dim, hidden_dim);
        let w_ff1 = lcg.he_matrix(4 * hidden_dim, hidden_dim);
        let w_ff2 = lcg.he_matrix(hidden_dim, 4 * hidden_dim);

        Self {
            hidden_dim,
            n_heads,
            pe_dim,
            w_msg,
            wq,
            wk,
            wv,
            w_out,
            alpha: 0.5,
            w_ff1,
            w_ff2,
        }
    }

    /// Local MPNN: for each node aggregate neighbour features then apply W_msg + ReLU.
    fn local_mpnn(&self, h: &[Vec<f64>], adj: &[Vec<usize>]) -> Vec<Vec<f64>> {
        let n = h.len();
        let mut out = vec![vec![0.0_f64; self.hidden_dim]; n];
        for i in 0..n {
            let nbrs = &adj[i];
            let agg = if nbrs.is_empty() {
                h[i].clone()
            } else {
                // Mean aggregation
                let mut sum = vec![0.0_f64; self.hidden_dim];
                for &j in nbrs {
                    for d in 0..self.hidden_dim.min(h[j].len()) {
                        sum[d] += h[j][d];
                    }
                }
                let cnt = nbrs.len() as f64;
                sum.iter().map(|v| v / cnt).collect()
            };
            let msg = mv(&self.w_msg, &agg);
            out[i] = msg.into_iter().map(|v| v.max(0.0)).collect(); // ReLU
        }
        out
    }

    /// Global Transformer: full self-attention with PE bias on Q and K.
    ///
    /// Returns node embeddings and a flat attention weight vector (for testing).
    fn global_transformer(&self, h: &[Vec<f64>], pe: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = h.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let head_dim = (self.hidden_dim / self.n_heads).max(1);
        let scale = (head_dim as f64).sqrt().max(1e-6);

        // Augmented input: concat h[i] and pe[i]
        let aug: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let hi = if i < h.len() { &h[i] } else { &h[0] };
                let pi = if i < pe.len() { &pe[i] } else { &pe[0] };
                let mut v = hi.clone();
                v.extend_from_slice(pi);
                v
            })
            .collect();

        // Multi-head attention
        let mut head_outputs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(self.n_heads);
        let mut all_attn: Vec<f64> = Vec::new();

        for hd in 0..self.n_heads {
            // Compute Q, K for augmented input; V for h only
            let q: Vec<Vec<f64>> = aug.iter().map(|a| mv(&self.wq[hd], a)).collect();
            let k: Vec<Vec<f64>> = aug.iter().map(|a| mv(&self.wk[hd], a)).collect();
            let v: Vec<Vec<f64>> = h.iter().map(|hi| mv(&self.wv[hd], hi)).collect();

            // Attention matrix n×n
            let mut attn_logits = vec![vec![0.0_f64; n]; n];
            for i in 0..n {
                for j in 0..n {
                    let dot: f64 = q[i].iter().zip(k[j].iter()).map(|(a, b)| a * b).sum();
                    attn_logits[i][j] = dot / scale;
                }
            }

            // Softmax over each row
            let attn_weights: Vec<Vec<f64>> = attn_logits.iter().map(|row| softmax(row)).collect();

            if hd == 0 {
                // Collect first-head weights for test inspection
                for row in &attn_weights {
                    all_attn.extend_from_slice(row);
                }
            }

            // Weighted sum of V
            let mut head_out = vec![vec![0.0_f64; head_dim.min(v[0].len())]; n];
            for i in 0..n {
                for j in 0..n {
                    let vj_len = v[j].len().min(head_dim);
                    for d in 0..vj_len {
                        head_out[i][d] += attn_weights[i][j] * v[j][d];
                    }
                }
            }
            head_outputs.push(head_out);
        }

        // Concatenate heads (take first hidden_dim dims from each head output)
        let head_dim_out = (self.hidden_dim / self.n_heads).max(1);
        let mut concat = vec![vec![0.0_f64; self.hidden_dim]; n];
        for i in 0..n {
            for hd in 0..self.n_heads {
                let start = hd * head_dim_out;
                let end = (start + head_dim_out).min(self.hidden_dim);
                for d in start..end {
                    let local_d = d - start;
                    if local_d < head_outputs[hd][i].len() {
                        concat[i][d] = head_outputs[hd][i][local_d];
                    }
                }
            }
        }

        // Final output projection
        let out: Vec<Vec<f64>> = concat.iter().map(|c| mv(&self.w_out, c)).collect();
        (out, all_attn)
    }

    /// FFN: 2-layer MLP with GELU.
    fn ffn(&self, h: &[Vec<f64>]) -> Vec<Vec<f64>> {
        h.iter()
            .map(|x| {
                let mid: Vec<f64> = mv(&self.w_ff1, x).into_iter().map(gelu).collect();
                mv(&self.w_ff2, &mid)
            })
            .collect()
    }

    /// Full GPS layer forward pass.
    pub fn forward(
        &self,
        h: &[Vec<f64>],
        adj: &[Vec<usize>],
        pe: &[Vec<f64>],
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = h.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }

        // Ensure h has the right dimensionality (pad if needed)
        let h_norm: Vec<Vec<f64>> = h
            .iter()
            .map(|row| {
                let mut r = row.clone();
                r.resize(self.hidden_dim, 0.0);
                r
            })
            .collect();

        // Ensure pe has the right dimensionality
        let pe_norm: Vec<Vec<f64>> = pe
            .iter()
            .map(|row| {
                let mut r = row.clone();
                r.resize(self.pe_dim, 0.0);
                r
            })
            .collect();

        let h_local = self.local_mpnn(&h_norm, adj);
        let (h_global, attn_weights) = self.global_transformer(&h_norm, &pe_norm);

        // Combine: h_out = LN(h + α·h_local + (1-α)·h_global)
        let alpha = self.alpha.clamp(0.0, 1.0);
        let combined: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let combined_raw: Vec<f64> = (0..self.hidden_dim)
                    .map(|d| h_norm[i][d] + alpha * h_local[i][d] + (1.0 - alpha) * h_global[i][d])
                    .collect();
                layer_norm(&combined_raw)
            })
            .collect();

        // FFN + residual + LayerNorm
        let h_ffn = self.ffn(&combined);
        let h_out: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let res = vadd(&combined[i], &h_ffn[i]);
                layer_norm(&res)
            })
            .collect();

        (h_out, attn_weights)
    }
}

// ============================================================================
// GpsModel
// ============================================================================

/// Full GPS model: stack of GPS layers + mean-pooling.
pub struct GpsModel {
    layers: Vec<GpsLayer>,
    hidden_dim: usize,
    pe_dim: usize,
    /// Input projection: hidden_dim × feat_dim  (built lazily on first forward)
    w_in: Option<Vec<Vec<f64>>>,
    feat_dim: usize,
}

impl GpsModel {
    /// Create a new GPS model from configuration.
    pub fn new(config: &GraphTransformerConfig) -> Self {
        let pe_dim = config.pe_dim;
        let hidden_dim = config.hidden_dim;
        let n_heads = config.n_heads.max(1);
        let layers: Vec<GpsLayer> = (0..config.n_layers)
            .map(|i| {
                let seed = (i as u64)
                    .wrapping_add(1)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15_u64);
                GpsLayer::new(hidden_dim, n_heads, pe_dim, seed)
            })
            .collect();

        Self {
            layers,
            hidden_dim,
            pe_dim,
            w_in: None,
            feat_dim: 0,
        }
    }

    /// Build or return the input-projection matrix.
    fn ensure_w_in(&mut self, feat_dim: usize) {
        if self.w_in.is_none() || self.feat_dim != feat_dim {
            let mut lcg = Lcg::new(0xdead_beef_cafe_babe);
            self.w_in = Some(lcg.he_matrix(self.hidden_dim, feat_dim.max(1)));
            self.feat_dim = feat_dim;
        }
    }

    /// Run a forward pass through the GPS model.
    ///
    /// `pe` should be an `n × pe_dim` matrix produced by `laplacian_pe` or `rwpe`.
    pub fn forward(
        &mut self,
        graph: &GraphForTransformer,
        pe: &[Vec<f64>],
    ) -> Result<(GraphTransformerOutput, Vec<f64>)> {
        let n = graph.n_nodes;
        if n == 0 {
            return Ok((
                GraphTransformerOutput {
                    node_embeddings: Vec::new(),
                    graph_embedding: vec![0.0; self.hidden_dim],
                },
                Vec::new(),
            ));
        }

        let feat_dim = graph
            .node_features
            .first()
            .map(|r| r.len())
            .unwrap_or(1)
            .max(1);
        self.ensure_w_in(feat_dim);

        let w_in = self
            .w_in
            .as_ref()
            .ok_or_else(|| GraphError::InvalidParameter {
                param: "w_in".to_string(),
                value: "None".to_string(),
                expected: "initialised weight matrix".to_string(),
                context: "GpsModel forward".to_string(),
            })?;

        // Project features to hidden_dim
        let mut h: Vec<Vec<f64>> = graph.node_features.iter().map(|f| mv(w_in, f)).collect();

        let mut last_attn: Vec<f64> = Vec::new();
        for layer in &self.layers {
            let (h_new, attn) = layer.forward(&h, &graph.adjacency, pe);
            h = h_new;
            last_attn = attn;
        }

        // Mean-pool for graph embedding
        let mut graph_emb = vec![0.0_f64; self.hidden_dim];
        for row in &h {
            for (d, &v) in row.iter().enumerate() {
                if d < self.hidden_dim {
                    graph_emb[d] += v;
                }
            }
        }
        let inv_n = 1.0 / n as f64;
        for v in graph_emb.iter_mut() {
            *v *= inv_n;
        }

        Ok((
            GraphTransformerOutput {
                node_embeddings: h,
                graph_embedding: graph_emb,
            },
            last_attn,
        ))
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::positional_encoding::{laplacian_pe, rwpe};
    use super::super::types::{GraphForTransformer, GraphTransformerConfig};
    use super::*;

    fn triangle_graph() -> GraphForTransformer {
        GraphForTransformer::new(
            vec![vec![1, 2], vec![0, 2], vec![0, 1]],
            vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
        )
        .expect("valid graph")
    }

    fn single_node_graph() -> GraphForTransformer {
        GraphForTransformer::new(vec![vec![]], vec![vec![1.0]]).expect("valid graph")
    }

    fn default_config() -> GraphTransformerConfig {
        GraphTransformerConfig {
            n_heads: 2,
            hidden_dim: 8,
            n_layers: 1,
            dropout: 0.0,
            pe_type: super::super::types::PeType::LapPE,
            pe_dim: 4,
        }
    }

    fn two_layer_config() -> GraphTransformerConfig {
        GraphTransformerConfig {
            n_layers: 2,
            ..default_config()
        }
    }

    #[test]
    fn test_gps_output_shape() {
        let g = triangle_graph();
        let pe = laplacian_pe(&g.adjacency, 4);
        let cfg = default_config();
        let mut model = GpsModel::new(&cfg);
        let (out, _) = model.forward(&g, &pe).expect("forward ok");
        assert_eq!(out.node_embeddings.len(), 3);
        for row in &out.node_embeddings {
            assert_eq!(row.len(), 8);
        }
    }

    #[test]
    fn test_gps_graph_embedding_shape() {
        let g = triangle_graph();
        let pe = laplacian_pe(&g.adjacency, 4);
        let cfg = default_config();
        let mut model = GpsModel::new(&cfg);
        let (out, _) = model.forward(&g, &pe).expect("forward ok");
        assert_eq!(out.graph_embedding.len(), 8);
    }

    #[test]
    fn test_gps_single_node() {
        let g = single_node_graph();
        let pe = laplacian_pe(&g.adjacency, 4);
        let cfg = default_config();
        let mut model = GpsModel::new(&cfg);
        let (out, _) = model.forward(&g, &pe).expect("forward ok");
        assert_eq!(out.node_embeddings.len(), 1);
        assert_eq!(out.graph_embedding.len(), 8);
    }

    #[test]
    fn test_gps_no_edges() {
        // 3 isolated nodes
        let g = GraphForTransformer::new(
            vec![vec![], vec![], vec![]],
            vec![vec![1.0], vec![2.0], vec![3.0]],
        )
        .expect("valid");
        let pe = rwpe(&g.adjacency, 4);
        let cfg = default_config();
        let mut model = GpsModel::new(&cfg);
        let (out, _) = model.forward(&g, &pe).expect("forward ok");
        assert_eq!(out.node_embeddings.len(), 3);
    }

    #[test]
    fn test_gps_attention_softmax() {
        // Attention weights returned for first head; each row should sum ≈ 1
        let g = triangle_graph();
        let pe = laplacian_pe(&g.adjacency, 4);
        let cfg = default_config();
        let layer = GpsLayer::new(cfg.hidden_dim, cfg.n_heads, cfg.pe_dim, 42);
        let h: Vec<Vec<f64>> = g
            .node_features
            .iter()
            .map(|f| {
                let mut r = f.clone();
                r.resize(cfg.hidden_dim, 0.0);
                r
            })
            .collect();
        let pe_norm: Vec<Vec<f64>> = pe
            .iter()
            .map(|p| {
                let mut r = p.clone();
                r.resize(cfg.pe_dim, 0.0);
                r
            })
            .collect();
        let (_out, attn) = layer.global_transformer(&h, &pe_norm);
        // attn contains n*n values for first head; each row of n should sum to ~1
        let n = g.n_nodes;
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| attn[i * n + j]).sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "row {i} sum={row_sum}");
        }
    }

    #[test]
    fn test_gps_layers_stack() {
        let g = triangle_graph();
        let pe = laplacian_pe(&g.adjacency, 4);
        let cfg = two_layer_config();
        let mut model = GpsModel::new(&cfg);
        let (out, _) = model.forward(&g, &pe).expect("2-layer forward ok");
        assert_eq!(out.node_embeddings.len(), 3);
        for row in &out.node_embeddings {
            assert_eq!(row.len(), 8);
            // Values should be finite
            for &v in row {
                assert!(v.is_finite(), "non-finite value in output");
            }
        }
    }
}
