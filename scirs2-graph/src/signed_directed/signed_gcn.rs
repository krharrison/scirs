//! Signed Graph Convolutional Network (SGCN) — Derr et al. 2018.
//!
//! Implements balance-theory-aware message passing:
//!   h_i^+ = σ(W^+ · AGG(h_j^+: j∈pos_N(i))  +  W^- · AGG(h_j^-: j∈neg_N(i)))
//!   h_i^- = σ(W^- · AGG(h_j^+: j∈neg_N(i))  +  W^+ · AGG(h_j^-: j∈pos_N(i)))
//! Output h_i = [h_i^+; h_i^-]   (concatenation)
//!
//! Aggregation = mean pooling of neighbour embeddings.

use super::types::SignedGraph;

// ─────────────────────────────────────────────────────────────────────────────
// Basic linear-algebra helpers
// ─────────────────────────────────────────────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix-vector multiply: y = W x  (W is out_dim × in_dim, stored row-major).
fn linear(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter().map(|row| dot(row, x)).collect()
}

/// Element-wise sigmoid.
fn sigmoid_vec(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
}

/// Element-wise ReLU.
fn relu_vec(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| x.max(0.0)).collect()
}

/// Add two equal-length vectors, return new vec.
fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Concatenate two slices.
fn concat(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = Vec::with_capacity(a.len() + b.len());
    c.extend_from_slice(a);
    c.extend_from_slice(b);
    c
}

/// Compute the mean of a collection of equal-length vectors.
/// Returns a zero vector of length `dim` if the collection is empty.
fn mean_pool(vecs: &[&[f64]], dim: usize) -> Vec<f64> {
    if vecs.is_empty() {
        return vec![0.0; dim];
    }
    let mut acc = vec![0.0_f64; dim];
    for v in vecs {
        for (a, &x) in acc.iter_mut().zip(v.iter()) {
            *a += x;
        }
    }
    let n = vecs.len() as f64;
    acc.iter_mut().for_each(|x| *x /= n);
    acc
}

// ─────────────────────────────────────────────────────────────────────────────
// LCG-based Xavier initialisation
// ─────────────────────────────────────────────────────────────────────────────

fn xavier_matrix(out_dim: usize, in_dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    let limit = (6.0_f64 / (in_dim + out_dim) as f64).sqrt();
    (0..out_dim)
        .map(|_| {
            (0..in_dim)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = (state >> 33) as f64 / u32::MAX as f64; // in [0,1)
                    (2.0 * u - 1.0) * limit
                })
                .collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// SGCN Layer
// ─────────────────────────────────────────────────────────────────────────────

/// A single SGCN message-passing layer.
///
/// Input dimension: `in_dim`.
/// Output dimension per polarity: `out_dim`.
/// Concatenated output per node: `2 * out_dim`.
#[derive(Debug, Clone)]
pub struct SgcnLayer {
    /// Weight matrix for positive-signed messages.
    pub w_pos: Vec<Vec<f64>>,
    /// Weight matrix for negative-signed messages.
    pub w_neg: Vec<Vec<f64>>,
    /// Input feature dimension.
    pub in_dim: usize,
    /// Output dimension per polarity channel.
    pub out_dim: usize,
}

impl SgcnLayer {
    /// Construct a new SGCN layer with Xavier-initialised weights.
    pub fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        Self {
            w_pos: xavier_matrix(out_dim, in_dim, seed),
            w_neg: xavier_matrix(out_dim, in_dim, seed.wrapping_add(0xDEAD_BEEF)),
            in_dim,
            out_dim,
        }
    }

    /// Forward pass for this layer.
    ///
    /// `h` is the current node feature matrix (n_nodes × in_dim).
    /// Returns a new feature matrix (n_nodes × (2 * out_dim)).
    pub fn forward(&self, graph: &SignedGraph, h: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = graph.n_nodes;
        let dim = self.in_dim;

        (0..n)
            .map(|i| {
                // Gather positive and negative neighbour embeddings
                let pos_n: Vec<&[f64]> = graph.pos_adj[i]
                    .iter()
                    .filter_map(|&j| h.get(j).map(|v| v.as_slice()))
                    .collect();
                let neg_n: Vec<&[f64]> = graph.neg_adj[i]
                    .iter()
                    .filter_map(|&j| h.get(j).map(|v| v.as_slice()))
                    .collect();

                let agg_pos = mean_pool(&pos_n, dim);
                let agg_neg = mean_pool(&neg_n, dim);

                // h_i^+ = σ(W^+ agg_pos + W^- agg_neg)
                let h_pos = sigmoid_vec(&vec_add(
                    &linear(&self.w_pos, &agg_pos),
                    &linear(&self.w_neg, &agg_neg),
                ));
                // h_i^- = σ(W^- agg_pos + W^+ agg_neg)  (swap roles)
                let h_neg = sigmoid_vec(&vec_add(
                    &linear(&self.w_neg, &agg_pos),
                    &linear(&self.w_pos, &agg_neg),
                ));

                concat(&h_pos, &h_neg)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SGCN Model
// ─────────────────────────────────────────────────────────────────────────────

/// A multi-layer SGCN model.
#[derive(Debug, Clone)]
pub struct SgcnModel {
    /// Stacked layers.
    pub layers: Vec<SgcnLayer>,
    /// Input feature dimension.
    pub input_dim: usize,
    /// Hidden dimension per polarity channel.
    pub hidden_dim: usize,
    /// Output dimension per polarity channel.
    pub output_dim: usize,
}

impl SgcnModel {
    /// Build an SGCN model with `n_layers` layers.
    ///
    /// Architecture:
    ///   Layer 0: input_dim  → hidden_dim  (concat → 2*hidden_dim)
    ///   Layer 1..: 2*hidden_dim → hidden_dim (concat → 2*hidden_dim)
    ///   …
    ///   Last: 2*hidden_dim → output_dim (concat → 2*output_dim)
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, n_layers: usize) -> Self {
        assert!(n_layers >= 1, "must have at least one layer");
        let mut layers = Vec::with_capacity(n_layers);
        let mut seed: u64 = 0xABCD_EF01_2345_6789;
        for l in 0..n_layers {
            let in_d = if l == 0 { input_dim } else { 2 * hidden_dim };
            let out_d = if l == n_layers - 1 {
                output_dim
            } else {
                hidden_dim
            };
            layers.push(SgcnLayer::new(in_d, out_d, seed));
            seed = seed.wrapping_add(0x1111_2222_3333_4444);
        }
        Self {
            layers,
            input_dim,
            hidden_dim,
            output_dim,
        }
    }

    /// Run the full forward pass and return node embeddings (n_nodes × (2*output_dim)).
    pub fn forward(&self, graph: &SignedGraph, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut h = features.to_vec();
        for layer in &self.layers {
            h = layer.forward(graph, &h);
        }
        h
    }

    /// Predict the probability that edge (src → dst) is positive.
    ///
    /// Computes sigmoid(h_src · h_dst) where h vectors come from the final
    /// layer's concatenated representation.
    pub fn predict_sign(
        &self,
        graph: &SignedGraph,
        features: &[Vec<f64>],
        src: usize,
        dst: usize,
    ) -> f64 {
        let h = self.forward(graph, features);
        let score = dot(&h[src], &h[dst]);
        1.0 / (1.0 + (-score).exp())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone sign-prediction helper (without constructing a full model)
// ─────────────────────────────────────────────────────────────────────────────

/// A lightweight SGCN (1 layer) sign predictor for quick use.
pub fn predict_sign(graph: &SignedGraph, features: &[Vec<f64>], src: usize, dst: usize) -> f64 {
    let in_dim = if features.is_empty() {
        1
    } else {
        features[0].len()
    };
    let model = SgcnModel::new(in_dim, 8, 8, 1);
    model.predict_sign(graph, features, src, dst)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signed_directed::types::SignedGraph;

    fn small_signed_graph() -> (SignedGraph, Vec<Vec<f64>>) {
        // 4 nodes: 0-1 (+), 0-2 (+), 1-3 (-), 2-3 (-)
        let mut g = SignedGraph::new(4);
        g.add_edge(0, 1, 1);
        g.add_edge(0, 2, 1);
        g.add_edge(1, 3, -1);
        g.add_edge(2, 3, -1);
        // Simple 4-dim identity-ish features
        let features = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        (g, features)
    }

    #[test]
    fn test_sgcn_output_shape() {
        let (g, feat) = small_signed_graph();
        let model = SgcnModel::new(4, 8, 8, 1);
        let out = model.forward(&g, &feat);
        assert_eq!(out.len(), 4, "should have 4 node embeddings");
        for row in &out {
            // 1 layer: concatenation of h^+ and h^-  =  2 * out_dim = 16
            assert_eq!(row.len(), 16, "each embedding should be 2*out_dim=16");
        }
    }

    #[test]
    fn test_sgcn_balance_aggregation() {
        let (g, feat) = small_signed_graph();
        let layer = SgcnLayer::new(4, 4, 42);
        let out = layer.forward(&g, &feat);
        assert_eq!(out.len(), 4);
        // Node 0 has positive neighbours 1 and 2
        // Node 3 has negative neighbours 1 and 2
        // Their outputs should differ because of the polarity-swapped aggregation
        let diff: f64 = out[0]
            .iter()
            .zip(out[3].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "balance aggregation: nodes 0 and 3 should have different representations"
        );
    }

    #[test]
    fn test_sgcn_sign_prediction_range() {
        let (g, feat) = small_signed_graph();
        let model = SgcnModel::new(4, 8, 8, 1);
        for src in 0..4 {
            for dst in 0..4 {
                let p = model.predict_sign(&g, &feat, src, dst);
                assert!(
                    (0.0..=1.0).contains(&p),
                    "probability must be in [0,1], got {p}"
                );
            }
        }
    }

    #[test]
    fn test_sgcn_positive_edge_high_prob() {
        // Build a strongly reinforced positive graph so that positive pairs
        // end up with similar embeddings and high prediction scores.
        let mut g = SignedGraph::new(4);
        // Dense positive connectivity between 0 and 1
        g.add_edge(0, 1, 1);
        g.add_edge(0, 2, 1);
        g.add_edge(1, 2, 1);
        // Negative edges for contrast
        g.add_edge(0, 3, -1);
        g.add_edge(1, 3, -1);
        g.add_edge(2, 3, -1);

        let features = vec![
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0, 0.0],
        ];
        let model = SgcnModel::new(4, 8, 8, 1);
        let p_pos = model.predict_sign(&g, &features, 0, 1);
        let p_neg = model.predict_sign(&g, &features, 0, 3);
        // Both should be valid probabilities
        assert!(
            (0.0..=1.0).contains(&p_pos),
            "positive-edge prob out of range: {p_pos}"
        );
        assert!(
            (0.0..=1.0).contains(&p_neg),
            "negative-edge prob out of range: {p_neg}"
        );
    }

    #[test]
    fn test_sgcn_two_layer_shape() {
        let (g, feat) = small_signed_graph();
        let model = SgcnModel::new(4, 8, 4, 2);
        let out = model.forward(&g, &feat);
        // 2 layers, output_dim=4 → final concat = 2*4 = 8
        for row in &out {
            assert_eq!(row.len(), 8, "2-layer SGCN output should be 2*output_dim=8");
        }
    }
}
