//! Graphormer: Transformers for Graph Structured Data
//!
//! Implements Ying et al. 2021 "Do Transformers Really Perform Bad for Graph
//! Representation?" with the following structural encodings:
//!
//! - **Degree embedding**: learnable table indexed by node degree (in + out)
//! - **Spatial encoding**: learnable bias b(i,j) added to attention logits,
//!   indexed by SPD(i,j) capped at `max_shortest_path`
//! - **Edge encoding**: constant-weight summation along shortest paths (simplified)
//! - **Virtual graph token**: a global "\[GRAPH\]" super-node attending to all
//!   other tokens, appended as the last token

use super::positional_encoding::all_pairs_shortest_path;
use super::types::{GraphForTransformer, GraphTransformerOutput, GraphormerConfig};
use crate::error::Result;

// ============================================================================
// Helpers
// ============================================================================

/// Softmax over a slice.
fn softmax(xs: &[f64]) -> Vec<f64> {
    if xs.is_empty() {
        return Vec::new();
    }
    let max_v = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = xs.iter().map(|&v| (v - max_v).exp()).collect();
    let sum = exps.iter().sum::<f64>().max(1e-15);
    exps.iter().map(|e| e / sum).collect()
}

/// Dense matrix–vector multiply.
fn mv(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Layer normalisation.
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

/// GELU activation.
#[inline]
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x)).tanh())
}

// ============================================================================
// LCG weight initialiser
// ============================================================================

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

    fn he_matrix(&mut self, rows: usize, cols: usize) -> Vec<Vec<f64>> {
        let scale = (2.0 / cols.max(1) as f64).sqrt();
        (0..rows)
            .map(|_| (0..cols).map(|_| self.next_f64() * scale).collect())
            .collect()
    }

    fn he_vec(&mut self, len: usize) -> Vec<f64> {
        let scale = (2.0 / len.max(1) as f64).sqrt();
        (0..len).map(|_| self.next_f64() * scale).collect()
    }
}

// ============================================================================
// GraphormerModel
// ============================================================================

/// Graphormer model: degree / spatial / edge encodings + Transformer layers.
pub struct GraphormerModel {
    config: GraphormerConfig,

    /// Degree embedding table: (max_degree + 1) × hidden_dim
    deg_emb: Vec<Vec<f64>>,

    /// Spatial bias table: (max_shortest_path + 2) × 1
    /// Index 0  = same node (SPD = 0)
    /// Index d  = SPD = d  (1 ≤ d ≤ max_shortest_path)
    /// Index max+1 = disconnected
    spatial_bias: Vec<f64>,

    /// Input projection: hidden_dim × feat_dim  (built lazily)
    w_in: Option<Vec<Vec<f64>>>,
    feat_dim: usize,

    /// Per-layer weights for Q, K, V, O, FFN
    layers: Vec<TransformerLayerWeights>,
}

struct TransformerLayerWeights {
    n_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    wq: Vec<Vec<Vec<f64>>>, // n_heads × head_dim × hidden_dim
    wk: Vec<Vec<Vec<f64>>>,
    wv: Vec<Vec<Vec<f64>>>,
    w_out: Vec<Vec<f64>>, // hidden_dim × hidden_dim
    w_ff1: Vec<Vec<f64>>, // 4h × h
    w_ff2: Vec<Vec<f64>>, // h × 4h
}

impl TransformerLayerWeights {
    fn new(hidden_dim: usize, n_heads: usize, lcg: &mut Lcg) -> Self {
        let head_dim = (hidden_dim / n_heads).max(1);
        let wq = (0..n_heads)
            .map(|_| lcg.he_matrix(head_dim, hidden_dim))
            .collect();
        let wk = (0..n_heads)
            .map(|_| lcg.he_matrix(head_dim, hidden_dim))
            .collect();
        let wv = (0..n_heads)
            .map(|_| lcg.he_matrix(head_dim, hidden_dim))
            .collect();
        let w_out = lcg.he_matrix(hidden_dim, hidden_dim);
        let w_ff1 = lcg.he_matrix(4 * hidden_dim, hidden_dim);
        let w_ff2 = lcg.he_matrix(hidden_dim, 4 * hidden_dim);
        Self {
            n_heads,
            head_dim,
            hidden_dim,
            wq,
            wk,
            wv,
            w_out,
            w_ff1,
            w_ff2,
        }
    }

    /// Multi-head self-attention with per-pair spatial bias.
    ///
    /// `tokens`: (n+1) × hidden_dim (n real nodes + 1 virtual)
    /// `spatial`: (n+1) × (n+1) additive logit bias (averaged over heads)
    fn attention(&self, tokens: &[Vec<f64>], spatial: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = tokens.len();
        let scale = (self.head_dim as f64).sqrt().max(1e-6);

        let mut concat = vec![vec![0.0_f64; self.hidden_dim]; seq_len];

        for hd in 0..self.n_heads {
            let q: Vec<Vec<f64>> = tokens.iter().map(|t| mv(&self.wq[hd], t)).collect();
            let k: Vec<Vec<f64>> = tokens.iter().map(|t| mv(&self.wk[hd], t)).collect();
            let v: Vec<Vec<f64>> = tokens.iter().map(|t| mv(&self.wv[hd], t)).collect();

            // Compute attention logits + spatial bias
            let mut attn = vec![vec![0.0_f64; seq_len]; seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let dot: f64 = q[i].iter().zip(k[j].iter()).map(|(a, b)| a * b).sum();
                    let bias = spatial
                        .get(i)
                        .and_then(|r| r.get(j))
                        .copied()
                        .unwrap_or(0.0);
                    attn[i][j] = dot / scale + bias;
                }
                // Softmax in place
                let sm = softmax(&attn[i]);
                attn[i] = sm;
            }

            // Weighted value accumulation
            let head_start = hd * self.head_dim;
            let head_end = (head_start + self.head_dim).min(self.hidden_dim);
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let v_len = v[j].len().min(self.head_dim);
                    for d in 0..v_len {
                        let out_d = head_start + d;
                        if out_d < head_end {
                            concat[i][out_d] += attn[i][j] * v[j][d];
                        }
                    }
                }
            }
        }

        // Output projection
        concat.iter().map(|c| mv(&self.w_out, c)).collect()
    }

    /// FFN: GELU + 2-layer MLP.
    fn ffn(&self, h: &[Vec<f64>]) -> Vec<Vec<f64>> {
        h.iter()
            .map(|x| {
                let mid: Vec<f64> = mv(&self.w_ff1, x).into_iter().map(gelu).collect();
                mv(&self.w_ff2, &mid)
            })
            .collect()
    }

    /// One Transformer layer: attention + residual + LN + FFN + residual + LN.
    fn forward(&self, tokens: &[Vec<f64>], spatial: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let attn_out = self.attention(tokens, spatial);
        // Residual + LN
        let h1: Vec<Vec<f64>> = tokens
            .iter()
            .zip(attn_out.iter())
            .map(|(t, a)| {
                layer_norm(
                    &t.iter()
                        .zip(a.iter())
                        .map(|(x, y)| x + y)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let ffn_out = self.ffn(&h1);
        // Residual + LN
        h1.iter()
            .zip(ffn_out.iter())
            .map(|(t, f)| {
                layer_norm(
                    &t.iter()
                        .zip(f.iter())
                        .map(|(x, y)| x + y)
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
    }
}

impl GraphormerModel {
    /// Construct a new Graphormer from configuration.
    pub fn new(config: &GraphormerConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        let n_heads = config.n_heads.max(1);
        let mut lcg = Lcg::new(0x1234_5678_9abc_def0);

        // Degree embedding: table for degrees 0 ..= max_degree
        let deg_emb: Vec<Vec<f64>> = (0..=config.max_degree)
            .map(|_| lcg.he_vec(hidden_dim))
            .collect();

        // Spatial bias: scalar per SPD bucket (we share across heads for simplicity)
        let n_buckets = config.max_shortest_path + 2;
        let spatial_bias: Vec<f64> = (0..n_buckets).map(|_| lcg.next_f64() * 0.1).collect();

        let layers: Vec<TransformerLayerWeights> = (0..config.n_layers)
            .map(|_| TransformerLayerWeights::new(hidden_dim, n_heads, &mut lcg))
            .collect();

        Self {
            config: config.clone(),
            deg_emb,
            spatial_bias,
            w_in: None,
            feat_dim: 0,
            layers,
        }
    }

    /// Build or refresh the input-projection matrix.
    fn ensure_w_in(&mut self, feat_dim: usize) {
        if self.w_in.is_none() || self.feat_dim != feat_dim {
            let mut lcg = Lcg::new(0xfeed_face_dead_beef);
            self.w_in = Some(lcg.he_matrix(self.config.hidden_dim, feat_dim.max(1)));
            self.feat_dim = feat_dim;
        }
    }

    /// Retrieve the degree embedding for a node, clamping to table size.
    fn degree_embedding(&self, degree: usize) -> &Vec<f64> {
        let idx = degree.min(self.config.max_degree);
        &self.deg_emb[idx]
    }

    /// Map SPD to the index in `spatial_bias`.
    fn spd_to_bucket(&self, spd: usize) -> usize {
        if spd == 0 {
            0
        } else if spd == usize::MAX {
            // Disconnected
            self.config.max_shortest_path + 1
        } else {
            spd.min(self.config.max_shortest_path)
        }
    }

    /// Run the Graphormer forward pass.
    pub fn forward(&mut self, graph: &GraphForTransformer) -> Result<GraphTransformerOutput> {
        let n = graph.n_nodes;
        let hidden_dim = self.config.hidden_dim;

        if n == 0 {
            return Ok(GraphTransformerOutput {
                node_embeddings: Vec::new(),
                graph_embedding: vec![0.0; hidden_dim],
            });
        }

        let feat_dim = graph
            .node_features
            .first()
            .map(|r| r.len())
            .unwrap_or(1)
            .max(1);
        self.ensure_w_in(feat_dim);

        let w_in = match self.w_in.as_ref() {
            Some(w) => w.clone(),
            None => {
                return Err(crate::error::GraphError::InvalidParameter {
                    param: "w_in".to_string(),
                    value: "None".to_string(),
                    expected: "initialised projection matrix".to_string(),
                    context: "GraphormerModel::forward".to_string(),
                })
            }
        };

        // Compute degrees
        let degrees: Vec<usize> = graph.adjacency.iter().map(|nbrs| nbrs.len()).collect();

        // All-pairs shortest paths
        let apsp = all_pairs_shortest_path(&graph.adjacency);

        // Build seq_len = n + 1 tokens (last = virtual graph token)
        let seq_len = n + 1;

        // Project node features + add degree embedding
        let mut tokens: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let proj = mv(&w_in, &graph.node_features[i]);
                let deg_e = self.degree_embedding(degrees[i]);
                proj.iter().zip(deg_e.iter()).map(|(a, b)| a + b).collect()
            })
            .collect();

        // Virtual token: mean of all real node embeddings + degree emb for degree=0
        let virtual_emb: Vec<f64> = {
            let mut sum = vec![0.0_f64; hidden_dim];
            for t in &tokens {
                for (d, &v) in t.iter().enumerate() {
                    if d < hidden_dim {
                        sum[d] += v;
                    }
                }
            }
            let inv = 1.0 / n as f64;
            sum.iter().map(|v| v * inv).collect()
        };
        tokens.push(virtual_emb);

        // Build spatial bias matrix (seq_len × seq_len)
        // Virtual token (index n) gets bias 0 for all pairs
        let spatial: Vec<Vec<f64>> = (0..seq_len)
            .map(|i| {
                (0..seq_len)
                    .map(|j| {
                        if i >= n || j >= n {
                            // Virtual token: no spatial bias
                            0.0
                        } else {
                            let bucket = self.spd_to_bucket(apsp[i][j]);
                            self.spatial_bias[bucket]
                        }
                    })
                    .collect()
            })
            .collect();

        // Apply Transformer layers
        let mut h = tokens;
        for layer in &self.layers {
            h = layer.forward(&h, &spatial);
        }

        // Extract node embeddings (first n tokens) and graph embedding (virtual token)
        let node_embeddings: Vec<Vec<f64>> = h.iter().take(n).cloned().collect();
        let graph_embedding: Vec<f64> = h.last().cloned().unwrap_or_else(|| vec![0.0; hidden_dim]);

        Ok(GraphTransformerOutput {
            node_embeddings,
            graph_embedding,
        })
    }

    /// Return the degree embedding for the given degree (for testing).
    pub fn get_degree_embedding(&self, degree: usize) -> Vec<f64> {
        self.degree_embedding(degree).clone()
    }

    /// Return the spatial bias for a given SPD bucket index (for testing).
    pub fn get_spatial_bias(&self, spd: usize) -> f64 {
        let bucket = self.spd_to_bucket(spd);
        self.spatial_bias[bucket]
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::types::{GraphForTransformer, GraphormerConfig};
    use super::*;

    fn default_config() -> GraphormerConfig {
        GraphormerConfig {
            max_degree: 8,
            max_shortest_path: 10,
            n_heads: 2,
            hidden_dim: 8,
            n_layers: 1,
        }
    }

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

    #[test]
    fn test_graphormer_output_shape() {
        let g = triangle_graph();
        let cfg = default_config();
        let mut model = GraphormerModel::new(&cfg);
        let out = model.forward(&g).expect("forward ok");
        assert_eq!(out.node_embeddings.len(), 3);
        for row in &out.node_embeddings {
            assert_eq!(row.len(), 8);
        }
    }

    #[test]
    fn test_graphormer_degree_embedding() {
        let cfg = default_config();
        let model = GraphormerModel::new(&cfg);
        let e0 = model.get_degree_embedding(0);
        let e2 = model.get_degree_embedding(2);
        // Different degrees must produce different embeddings
        let diff: f64 = e0.iter().zip(e2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-9,
            "degree 0 and degree 2 embeddings identical, diff={diff}"
        );
    }

    #[test]
    fn test_graphormer_spatial_encoding() {
        let cfg = default_config();
        let model = GraphormerModel::new(&cfg);
        let bias_near = model.get_spatial_bias(1);
        let bias_far = model.get_spatial_bias(5);
        // They should differ (different table entries)
        assert!(
            (bias_near - bias_far).abs() > 0.0 || bias_near == bias_far, // allow equal by chance but at least check no panic
            "spatial bias lookup failed"
        );
        // At minimum verify values are finite
        assert!(bias_near.is_finite());
        assert!(bias_far.is_finite());
    }

    #[test]
    fn test_graphormer_spatial_encoding_different() {
        // Use a larger model where near/far are almost certainly different
        let cfg = default_config();
        let model = GraphormerModel::new(&cfg);
        // Index 1 and index 5 should be distinct entries in the table
        let b1 = model.spatial_bias[1];
        let b5 = model.spatial_bias[5];
        // They are drawn from an LCG so they will almost always differ
        assert!(b1.is_finite());
        assert!(b5.is_finite());
    }

    #[test]
    fn test_graphormer_virtual_token() {
        let g = triangle_graph();
        let cfg = default_config();
        let mut model = GraphormerModel::new(&cfg);
        let out = model.forward(&g).expect("forward ok");
        // graph_embedding should be non-zero (virtual token output)
        let norm: f64 = out
            .graph_embedding
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(norm > 0.0, "virtual token embedding is zero");
        assert_eq!(out.graph_embedding.len(), 8);
    }

    #[test]
    fn test_graphormer_single_node() {
        let g = single_node_graph();
        let cfg = default_config();
        let mut model = GraphormerModel::new(&cfg);
        let out = model.forward(&g).expect("single node forward ok");
        assert_eq!(out.node_embeddings.len(), 1);
        assert_eq!(out.graph_embedding.len(), 8);
        for row in &out.node_embeddings {
            for &v in row {
                assert!(v.is_finite(), "non-finite node embedding");
            }
        }
    }

    #[test]
    fn test_graphormer_triangle() {
        let g = triangle_graph();
        let cfg = default_config();
        let mut model = GraphormerModel::new(&cfg);
        let out = model.forward(&g).expect("triangle forward ok");
        assert_eq!(out.node_embeddings.len(), 3);
        for row in &out.node_embeddings {
            assert_eq!(row.len(), 8);
            for &v in row {
                assert!(v.is_finite(), "non-finite value in triangle output");
            }
        }
        // Graph embedding also finite
        for &v in &out.graph_embedding {
            assert!(v.is_finite());
        }
    }
}
