//! Temporal Graph Attention Network (TGAT).
//!
//! Implements the TGAT model from:
//! > Xu, D., Ruan, C., Korpeoglu, E., Kumar, S., & Achan, K. (2020).
//! > "Inductive Representation Learning on Temporal Graphs."
//! > ICLR 2020. <https://arxiv.org/abs/2002.07962>
//!
//! ## Architecture
//!
//! For each node `i` queried at time `t`:
//! 1. Gather temporal neighbors `{(j, t_ij)}` with `t_ij < t` (causal masking).
//! 2. Compute time-aware key/query/value attention across the neighborhood.
//!    - Query: `h_i ⊕ φ(0)` (self at query time)
//!    - Key:   `h_j ⊕ φ(t - t_ij)` (neighbor + time-since-interaction)
//!    - Value: `h_j ⊕ φ(t - t_ij)`
//! 3. Multi-head attention: `head_k = softmax(Q_k K_k^T / √d) V_k`
//! 4. Concatenate heads, project through a linear layer + residual connection.
//!
//! The model can be stacked for multi-hop temporal attention.

use super::time_encoding::{concat, matvec, scaled_dot_product, softmax, xavier_init, relu_vec};
use super::types::{TgatConfig, TgnnGraph, TemporalPrediction};
use crate::error::{GraphError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// TgatLayer
// ─────────────────────────────────────────────────────────────────────────────

/// One layer of Temporal Graph Attention (multi-head).
///
/// Each head operates on a projected space of dimension `head_dim`.
/// The input to each attention head is a concatenation of the node feature
/// and its time encoding, so the actual input width is
/// `node_feat_dim + time_dim`.
#[derive(Debug, Clone)]
pub struct TgatLayer {
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Time encoding dimension (must match the TimeEncode used)
    pub time_dim: usize,
    /// Input dimension = node_feat_dim + time_dim
    pub input_dim: usize,
    /// Query projection matrices W_Q^k, one per head (head_dim × input_dim)
    w_q: Vec<Vec<Vec<f64>>>,
    /// Key projection matrices W_K^k (head_dim × input_dim)
    w_k: Vec<Vec<Vec<f64>>>,
    /// Value projection matrices W_V^k (head_dim × input_dim)
    w_v: Vec<Vec<Vec<f64>>>,
    /// Output projection W_O (output_dim × (num_heads * head_dim))
    w_o: Vec<Vec<f64>>,
    /// Output bias b_O (output_dim)
    b_o: Vec<f64>,
    /// Output dimension (= num_heads * head_dim)
    pub output_dim: usize,
}

impl TgatLayer {
    /// Create a new TGAT layer.
    ///
    /// `node_feat_dim` is the dimension of the incoming node feature,
    /// `time_dim` is the dimension of the time encoding output (must be even).
    pub fn new(
        node_feat_dim: usize,
        time_dim: usize,
        num_heads: usize,
        head_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        if num_heads == 0 || head_dim == 0 {
            return Err(GraphError::InvalidParameter {
                param: "num_heads/head_dim".to_string(),
                value: format!("{}/{}", num_heads, head_dim),
                expected: "both > 0".to_string(),
                context: "TgatLayer::new".to_string(),
            });
        }
        let input_dim = node_feat_dim + time_dim;
        let output_dim = num_heads * head_dim;

        let mut w_q = Vec::with_capacity(num_heads);
        let mut w_k = Vec::with_capacity(num_heads);
        let mut w_v = Vec::with_capacity(num_heads);

        for h in 0..num_heads {
            w_q.push(xavier_init(head_dim, input_dim, seed.wrapping_add(h as u64)));
            w_k.push(xavier_init(head_dim, input_dim, seed.wrapping_add(1000 + h as u64)));
            w_v.push(xavier_init(head_dim, input_dim, seed.wrapping_add(2000 + h as u64)));
        }
        let w_o = xavier_init(output_dim, output_dim, seed.wrapping_add(3000));
        let b_o = vec![0.0f64; output_dim];

        Ok(TgatLayer {
            num_heads,
            head_dim,
            time_dim,
            input_dim,
            w_q,
            w_k,
            w_v,
            w_o,
            b_o,
            output_dim,
        })
    }

    /// Forward pass for a single node `i` at query time `t`.
    ///
    /// `h_self` is the current feature of node `i` (length = node_feat_dim).
    /// `neighbors` is a list of `(h_neighbor, t_interaction)` pairs where
    ///   each `h_neighbor` has the same length as `h_self`.
    /// `time_enc` is the time-encoding function to use.
    ///
    /// Returns the updated embedding of shape `output_dim`.
    pub fn forward_node(
        &self,
        h_self: &[f64],
        neighbors: &[(Vec<f64>, f64)],
        query_time: f64,
        time_enc: &super::time_encoding::TimeEncode,
    ) -> Vec<f64> {
        // Query input: self feature + time encoding at Δt=0
        let phi_self = time_enc.encode(0.0);
        let q_input = concat(h_self, &phi_self);

        // If no neighbors, return zero vector
        if neighbors.is_empty() {
            return vec![0.0f64; self.output_dim];
        }

        // Build key/value inputs for each neighbor
        // key_input = h_nbr ‖ φ(t - t_nbr)
        let kv_inputs: Vec<Vec<f64>> = neighbors
            .iter()
            .map(|(h_nbr, t_nbr)| {
                let phi = time_enc.encode_delta(query_time, *t_nbr);
                concat(h_nbr, &phi)
            })
            .collect();

        // Multi-head attention
        let mut head_outputs: Vec<f64> = Vec::with_capacity(self.output_dim);

        for head in 0..self.num_heads {
            // Project query
            let q = matvec(&self.w_q[head], &q_input);

            // Project keys and values
            let keys: Vec<Vec<f64>> = kv_inputs
                .iter()
                .map(|kv| matvec(&self.w_k[head], kv))
                .collect();
            let values: Vec<Vec<f64>> = kv_inputs
                .iter()
                .map(|kv| matvec(&self.w_v[head], kv))
                .collect();

            // Compute attention scores and apply softmax
            let logits = scaled_dot_product(&q, &keys);
            let alphas = softmax(&logits);

            // Weighted sum of values
            let mut attended = vec![0.0f64; self.head_dim];
            for (alpha, val) in alphas.iter().zip(values.iter()) {
                for (a, v) in attended.iter_mut().zip(val.iter()) {
                    *a += alpha * v;
                }
            }
            head_outputs.extend(attended);
        }

        // Output projection + bias + ReLU
        let projected = matvec(&self.w_o, &head_outputs);
        let mut out: Vec<f64> = projected
            .iter()
            .zip(self.b_o.iter())
            .map(|(p, b)| p + b)
            .collect();
        out = relu_vec(&out);

        // Residual: if self embedding and output have the same dim, add self projection
        // (simplified: when dims match, add h_self padded/truncated)
        if h_self.len() == out.len() {
            for (o, s) in out.iter_mut().zip(h_self.iter()) {
                *o += s;
            }
        }

        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TgatModel
// ─────────────────────────────────────────────────────────────────────────────

/// Full TGAT model: stack of `TgatLayer`s with shared time encoding.
///
/// ## Usage
///
/// ```rust,no_run
/// use scirs2_graph::temporal::tgnn::{TgatModel, TgatConfig, TgnnGraph};
///
/// let config = TgatConfig::default();
/// let model = TgatModel::new(&config, 4).expect("model");
/// let mut graph = TgnnGraph::with_zero_features(5, 4);
/// let embeddings = model.forward(&graph, 10.0).expect("forward");
/// assert_eq!(embeddings.len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct TgatModel {
    /// Attention layers
    pub layers: Vec<TgatLayer>,
    /// Time encoder shared across layers
    pub time_enc: super::time_encoding::TimeEncode,
    /// Configuration used at construction
    pub config: TgatConfig,
    /// Output embedding dimension
    pub output_dim: usize,
}

impl TgatModel {
    /// Create a new TGAT model.
    ///
    /// `node_feat_dim` is the raw feature dimension; if 0, one-hot style
    /// features of size `config.head_dim` are used.
    pub fn new(config: &TgatConfig, node_feat_dim: usize) -> Result<Self> {
        let eff_feat_dim = if node_feat_dim == 0 {
            config.head_dim
        } else {
            node_feat_dim
        };

        let time_enc = super::time_encoding::TimeEncode::new(config.time_dim)?;

        let mut layers = Vec::with_capacity(config.num_layers);
        let output_dim = config.num_heads * config.head_dim;

        // Layer 0 takes raw node features; subsequent layers take previous output
        let first_layer = TgatLayer::new(
            eff_feat_dim,
            config.time_dim,
            config.num_heads,
            config.head_dim,
            12345,
        )?;
        let first_output = first_layer.output_dim;
        layers.push(first_layer);

        for layer_idx in 1..config.num_layers {
            let layer = TgatLayer::new(
                first_output, // subsequent layers consume previous layer's output
                config.time_dim,
                config.num_heads,
                config.head_dim,
                12345 + layer_idx as u64 * 999,
            )?;
            layers.push(layer);
        }

        Ok(TgatModel {
            layers,
            time_enc,
            config: config.clone(),
            output_dim,
        })
    }

    /// Compute embeddings for all nodes at query time `t`.
    ///
    /// Returns `Vec<Vec<f64>>` of length `n_nodes`, each of length `output_dim`.
    ///
    /// Causal masking: only edges with `timestamp < query_time` are used.
    pub fn forward(&self, graph: &TgnnGraph, query_time: f64) -> Result<Vec<Vec<f64>>> {
        let n = graph.n_nodes;
        if n == 0 {
            return Ok(Vec::new());
        }

        // Initialise with node features, padding/truncating to eff_feat_dim
        let eff_feat_dim = if graph.node_feat_dim == 0 {
            self.config.head_dim
        } else {
            graph.node_feat_dim
        };

        let mut current_embeddings: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let feat = graph.node_feat(i);
                if feat.is_empty() {
                    vec![0.0f64; eff_feat_dim]
                } else if feat.len() == eff_feat_dim {
                    feat.to_vec()
                } else {
                    // Pad or truncate
                    let mut v = vec![0.0f64; eff_feat_dim];
                    let copy_len = feat.len().min(eff_feat_dim);
                    v[..copy_len].copy_from_slice(&feat[..copy_len]);
                    v
                }
            })
            .collect();

        // Apply each layer
        for layer in &self.layers {
            let prev_embeddings = current_embeddings.clone();
            let mut next_embeddings = Vec::with_capacity(n);

            for i in 0..n {
                // Gather temporal neighbors before query_time
                let nbr_tuples = graph.neighbors_before(i, query_time);

                // Build (h_neighbor, t_edge) pairs using previous layer embeddings
                let neighbors: Vec<(Vec<f64>, f64)> = nbr_tuples
                    .iter()
                    .map(|(j, t_edge, _edge_feat)| {
                        let h_nbr = prev_embeddings
                            .get(*j)
                            .cloned()
                            .unwrap_or_else(|| vec![0.0f64; prev_embeddings[0].len()]);
                        (h_nbr, *t_edge)
                    })
                    .collect();

                let h_self = &prev_embeddings[i];
                let new_h = layer.forward_node(h_self, &neighbors, query_time, &self.time_enc);
                next_embeddings.push(new_h);
            }
            current_embeddings = next_embeddings;
        }

        Ok(current_embeddings)
    }

    /// Compute `TemporalPrediction` records for all nodes at time `t`.
    pub fn predict(&self, graph: &TgnnGraph, query_time: f64) -> Result<Vec<TemporalPrediction>> {
        let embeddings = self.forward(graph, query_time)?;
        Ok(embeddings
            .into_iter()
            .enumerate()
            .map(|(i, emb)| TemporalPrediction::new(i, emb, query_time))
            .collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::types::{TgatConfig, TgnnEdge, TgnnGraph};
    use super::super::time_encoding::TimeEncode;

    fn simple_graph() -> TgnnGraph {
        let mut g = TgnnGraph::with_zero_features(5, 4);
        // Past edges (t < 10.0)
        g.add_edge(TgnnEdge::no_feat(0, 1, 1.0));
        g.add_edge(TgnnEdge::no_feat(1, 2, 2.0));
        g.add_edge(TgnnEdge::no_feat(2, 3, 3.0));
        g.add_edge(TgnnEdge::no_feat(3, 4, 4.0));
        // Future edge (t > 10.0) — must not be attended to
        g.add_edge(TgnnEdge::no_feat(0, 4, 15.0));
        g
    }

    #[test]
    fn test_tgat_output_shape() {
        let config = TgatConfig {
            num_heads: 2,
            time_dim: 8,
            head_dim: 8,
            num_layers: 1,
            dropout: 0.0,
        };
        let model = TgatModel::new(&config, 4).expect("model creation");
        let graph = simple_graph();
        let embeddings = model.forward(&graph, 10.0).expect("forward pass");

        assert_eq!(embeddings.len(), 5, "must produce one embedding per node");
        let expected_dim = config.num_heads * config.head_dim;
        for emb in &embeddings {
            assert_eq!(emb.len(), expected_dim, "each embedding has wrong dim");
        }
    }

    #[test]
    fn test_tgat_causal_masking() {
        // The future edge (0→4, t=15) must not influence embeddings at t=10
        let config = TgatConfig {
            num_heads: 1,
            time_dim: 8,
            head_dim: 8,
            num_layers: 1,
            dropout: 0.0,
        };
        let model = TgatModel::new(&config, 4).expect("model");

        let mut g_with_future = TgnnGraph::with_zero_features(5, 4);
        g_with_future.add_edge(TgnnEdge::no_feat(0, 1, 1.0));
        g_with_future.add_edge(TgnnEdge::no_feat(0, 4, 15.0)); // future

        let mut g_no_future = TgnnGraph::with_zero_features(5, 4);
        g_no_future.add_edge(TgnnEdge::no_feat(0, 1, 1.0));

        let emb_future = model.forward(&g_with_future, 10.0).expect("forward");
        let emb_no_future = model.forward(&g_no_future, 10.0).expect("forward");

        // Embeddings must be identical since future edge is masked
        for (ef, en) in emb_future.iter().zip(emb_no_future.iter()) {
            for (a, b) in ef.iter().zip(en.iter()) {
                assert!(
                    (a - b).abs() < 1e-10,
                    "future edge must not influence embeddings"
                );
            }
        }
    }

    #[test]
    fn test_tgat_attention_softmax_sums_one() {
        // Verify the attention weight computation via a unit test on the layer
        let layer = TgatLayer::new(4, 8, 1, 8, 42).expect("layer");
        let time_enc = TimeEncode::new(8).expect("enc");

        let h_self = vec![1.0, 0.0, 0.0, 0.0];
        let neighbors = vec![
            (vec![0.0, 1.0, 0.0, 0.0], 1.0_f64),
            (vec![0.0, 0.0, 1.0, 0.0], 2.0_f64),
            (vec![0.0, 0.0, 0.0, 1.0], 3.0_f64),
        ];

        // Manually compute attention weights to verify softmax property
        let phi_self = time_enc.encode(0.0);
        let q_input = concat(&h_self, &phi_self);
        let q = matvec(&layer.w_q[0], &q_input);

        let keys: Vec<Vec<f64>> = neighbors
            .iter()
            .map(|(h_nbr, t_nbr)| {
                let phi = time_enc.encode_delta(10.0, *t_nbr);
                let kv = concat(h_nbr, &phi);
                matvec(&layer.w_k[0], &kv)
            })
            .collect();

        let logits = scaled_dot_product(&q, &keys);
        let alphas = softmax(&logits);
        let sum: f64 = alphas.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "attention weights must sum to 1, got {}", sum);
        for &a in &alphas {
            assert!(a >= 0.0, "attention weight must be non-negative");
        }
    }

    #[test]
    fn test_tgat_with_no_neighbors() {
        // Node with no past neighbors must produce a zero embedding
        let config = TgatConfig {
            num_heads: 2,
            time_dim: 8,
            head_dim: 8,
            num_layers: 1,
            dropout: 0.0,
        };
        let model = TgatModel::new(&config, 4).expect("model");

        // Graph with only a future edge (so all nodes have no past neighbors at t=0.5)
        let mut g = TgnnGraph::with_zero_features(3, 4);
        g.add_edge(TgnnEdge::no_feat(0, 1, 5.0)); // future relative to t=0.5

        let embeddings = model.forward(&g, 0.5).expect("forward");
        // All nodes have zero features and no neighbors → embeddings should be zero
        for emb in &embeddings {
            let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
            // Without self-loops or non-zero features, the output from attention
            // (no neighbors → zero attended output) + residual (h_self all zeros)
            // should produce zero or near-zero
            assert!(
                norm < 1e-10,
                "node with no neighbors and zero features should produce ~zero embedding, got norm={}",
                norm
            );
        }
    }

    #[test]
    fn test_tgat_multi_head_concat() {
        // Verify that multi-head output has dim = num_heads * head_dim
        let config = TgatConfig {
            num_heads: 4,
            time_dim: 8,
            head_dim: 6,
            num_layers: 1,
            dropout: 0.0,
        };
        let model = TgatModel::new(&config, 4).expect("model");
        let graph = simple_graph();
        let embeddings = model.forward(&graph, 5.0).expect("forward");
        let expected_dim = 4 * 6; // num_heads * head_dim
        for emb in &embeddings {
            assert_eq!(emb.len(), expected_dim, "multi-head concat size wrong");
        }
    }

    #[test]
    fn test_tgat_two_layers() {
        let config = TgatConfig {
            num_heads: 2,
            time_dim: 8,
            head_dim: 8,
            num_layers: 2,
            dropout: 0.0,
        };
        let model = TgatModel::new(&config, 4).expect("model");
        assert_eq!(model.layers.len(), 2);
        let graph = simple_graph();
        let embeddings = model.forward(&graph, 10.0).expect("2-layer forward");
        assert_eq!(embeddings.len(), 5);
    }
}
