//! Graph pre-training strategies for self-supervised learning.
//!
//! This module provides three complementary pre-training objectives:
//!
//! | Struct | Strategy | Reference |
//! |---|---|---|
//! | [`NodeMaskingPretrainer`] | BERT-style node attribute masking | Hu et al. 2020 (Strategies for Pre-training Graph Neural Networks) |
//! | [`GraphContextPretrainer`] | Subgraph-context InfoNCE contrastive | Hu et al. 2020 |
//! | [`AttributeReconstructionObjective`] | MAE-style attribute reconstruction | Hou et al. 2022 (GraphMAE) |

use crate::error::{GraphError, Result};

// ── Tiny LCG RNG (no external crate) ─────────────────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0xdeadbeefcafe1234)
    }
    /// Return next pseudo-random f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Use high 53 bits for double mantissa
        let bits = self.0 >> 11;
        bits as f64 / (1u64 << 53) as f64
    }
    /// Return next usize in 0..bound.
    fn next_usize(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % bound
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// A.  NodeMaskingPretrainer
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for BERT-style node attribute masking.
#[derive(Debug, Clone)]
pub struct NodeMaskingConfig {
    /// Fraction of nodes to mask. Default 0.15.
    pub mask_rate: f64,
    /// Of the masked nodes, fraction replaced with a random feature vector
    /// instead of a zero mask vector. Default 0.1.
    pub replace_rate: f64,
    /// Number of BFS hops used for context features (informational only).
    /// Default 2.
    pub n_neighbors: usize,
    /// Feature dimensionality. Default 64.
    pub feature_dim: usize,
}

impl Default for NodeMaskingConfig {
    fn default() -> Self {
        Self {
            mask_rate: 0.15,
            replace_rate: 0.1,
            n_neighbors: 2,
            feature_dim: 64,
        }
    }
}

/// BERT-style node attribute masking pre-trainer.
///
/// # Usage
/// ```rust,no_run
/// use scirs2_graph::ssl::pretrain::{NodeMaskingPretrainer, NodeMaskingConfig};
///
/// let cfg = NodeMaskingConfig { feature_dim: 4, ..Default::default() };
/// let pretrainer = NodeMaskingPretrainer::new(cfg);
/// let features = vec![vec![1.0, 2.0, 3.0, 4.0]; 10];
/// let (masked, indices) = pretrainer.mask_nodes(&features, 42).unwrap();
/// ```
pub struct NodeMaskingPretrainer {
    config: NodeMaskingConfig,
}

impl NodeMaskingPretrainer {
    /// Create a new pretrainer with the given configuration.
    pub fn new(config: NodeMaskingConfig) -> Self {
        Self { config }
    }

    /// Apply attribute masking to the node feature matrix.
    ///
    /// # Arguments
    /// * `features` – `n_nodes × feature_dim` feature matrix.
    /// * `rng_seed` – seed for reproducibility.
    ///
    /// # Returns
    /// `(masked_features, masked_indices)` where `masked_features` has the
    /// selected nodes zeroed out (or replaced with random vectors), and
    /// `masked_indices` is the sorted list of masked node indices.
    ///
    /// # Errors
    /// Returns `GraphError::InvalidParameter` if feature vectors have
    /// inconsistent lengths or if `mask_rate` is outside (0, 1].
    pub fn mask_nodes(
        &self,
        features: &[Vec<f64>],
        rng_seed: u64,
    ) -> Result<(Vec<Vec<f64>>, Vec<usize>)> {
        let n = features.len();
        if n == 0 {
            return Ok((vec![], vec![]));
        }
        let dim = features[0].len();
        if dim == 0 {
            return Err(GraphError::InvalidParameter {
                param: "features".to_string(),
                value: "empty feature vectors".to_string(),
                expected: "non-empty feature vectors".to_string(),
                context: "NodeMaskingPretrainer::mask_nodes".to_string(),
            });
        }
        for (i, f) in features.iter().enumerate() {
            if f.len() != dim {
                return Err(GraphError::InvalidParameter {
                    param: format!("features[{i}]"),
                    value: format!("length {}", f.len()),
                    expected: format!("length {dim}"),
                    context: "NodeMaskingPretrainer::mask_nodes".to_string(),
                });
            }
        }
        if !(0.0 < self.config.mask_rate && self.config.mask_rate <= 1.0) {
            return Err(GraphError::InvalidParameter {
                param: "mask_rate".to_string(),
                value: format!("{}", self.config.mask_rate),
                expected: "value in (0, 1]".to_string(),
                context: "NodeMaskingPretrainer::mask_nodes".to_string(),
            });
        }

        let k = ((n as f64 * self.config.mask_rate).ceil() as usize).min(n);
        let mut rng = Lcg::new(rng_seed);

        // Randomly sample k distinct node indices (partial Fisher-Yates)
        let mut indices: Vec<usize> = (0..n).collect();
        for i in (n - k..n).rev() {
            let j = rng.next_usize(i + 1);
            indices.swap(i, j);
        }
        let mut masked_indices: Vec<usize> = indices[n - k..].to_vec();
        masked_indices.sort_unstable();

        // Build masked feature matrix
        let mut masked = features.to_vec();
        let masked_set: std::collections::HashSet<usize> = masked_indices.iter().cloned().collect();
        for &node in &masked_indices {
            let replace = rng.next_f64() < self.config.replace_rate;
            masked[node] = if replace {
                // Replace with random vector sampled from U(-1, 1)
                (0..dim).map(|_| rng.next_f64() * 2.0 - 1.0).collect()
            } else {
                // Standard BERT masking: zero out
                vec![0.0; dim]
            };
        }
        // Suppress unused variable warning
        let _ = masked_set;

        Ok((masked, masked_indices))
    }

    /// Compute MSE reconstruction loss only on the masked nodes.
    ///
    /// # Arguments
    /// * `predicted`     – reconstructed features (same shape as `original`).
    /// * `original`      – ground-truth features.
    /// * `masked_indices`– which node indices to include in the loss.
    ///
    /// # Errors
    /// Returns `GraphError::InvalidParameter` if any index is out of range or
    /// vectors have mismatched lengths.
    pub fn reconstruction_loss(
        &self,
        predicted: &[Vec<f64>],
        original: &[Vec<f64>],
        masked_indices: &[usize],
    ) -> Result<f64> {
        if predicted.len() != original.len() {
            return Err(GraphError::InvalidParameter {
                param: "predicted / original".to_string(),
                value: format!("lengths {} vs {}", predicted.len(), original.len()),
                expected: "equal lengths".to_string(),
                context: "NodeMaskingPretrainer::reconstruction_loss".to_string(),
            });
        }
        if masked_indices.is_empty() {
            return Ok(0.0);
        }
        let n = predicted.len();
        let mut total = 0.0_f64;
        let mut count = 0usize;
        for &idx in masked_indices {
            if idx >= n {
                return Err(GraphError::InvalidParameter {
                    param: "masked_indices".to_string(),
                    value: format!("{idx}"),
                    expected: format!("index < {n}"),
                    context: "NodeMaskingPretrainer::reconstruction_loss".to_string(),
                });
            }
            let p = &predicted[idx];
            let o = &original[idx];
            if p.len() != o.len() {
                return Err(GraphError::InvalidParameter {
                    param: format!("predicted[{idx}]"),
                    value: format!("length {}", p.len()),
                    expected: format!("length {}", o.len()),
                    context: "NodeMaskingPretrainer::reconstruction_loss".to_string(),
                });
            }
            for (a, b) in p.iter().zip(o.iter()) {
                let diff = a - b;
                total += diff * diff;
                count += 1;
            }
        }
        Ok(if count > 0 { total / count as f64 } else { 0.0 })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// B.  GraphContextPretrainer
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for subgraph-context contrastive pre-training.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct GraphContextConfig {
    /// Maximum number of nodes to include in the context subgraph (BFS size).
    /// Default 8.
    pub context_size: usize,
    /// Number of negative context samples per positive pair. Default 4.
    pub negative_samples: usize,
    /// Feature dimensionality. Default 64.
    pub feature_dim: usize,
    /// Temperature for InfoNCE loss. Default 0.07.
    pub temperature: f64,
}

impl Default for GraphContextConfig {
    fn default() -> Self {
        Self {
            context_size: 8,
            negative_samples: 4,
            feature_dim: 64,
            temperature: 0.07,
        }
    }
}

/// Graph-context contrastive pre-trainer.
///
/// Samples context subgraphs via BFS and maximises InfoNCE between a center
/// node embedding and its positive context while pushing away negative samples.
pub struct GraphContextPretrainer {
    config: GraphContextConfig,
}

impl GraphContextPretrainer {
    /// Create a new pretrainer.
    pub fn new(config: GraphContextConfig) -> Self {
        Self { config }
    }

    /// Sample a context subgraph around `center` using BFS, limited to
    /// `config.context_size` nodes (including the center).
    ///
    /// # Arguments
    /// * `adj`    – undirected edge list as `(src, dst)` pairs.
    /// * `center` – starting node.
    /// * `n_nodes`– total number of nodes.
    /// * `seed`   – RNG seed (used to break BFS frontier ties randomly).
    ///
    /// # Returns
    /// A vector of node indices (sorted) in the context subgraph.
    ///
    /// # Errors
    /// Returns `GraphError::InvalidParameter` if `center >= n_nodes`.
    pub fn sample_context_subgraph(
        &self,
        adj: &[(usize, usize)],
        center: usize,
        n_nodes: usize,
        seed: u64,
    ) -> Result<Vec<usize>> {
        if n_nodes == 0 {
            return Ok(vec![]);
        }
        if center >= n_nodes {
            return Err(GraphError::InvalidParameter {
                param: "center".to_string(),
                value: format!("{center}"),
                expected: format!("index < {n_nodes}"),
                context: "GraphContextPretrainer::sample_context_subgraph".to_string(),
            });
        }

        // Build adjacency lists
        let mut lists: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for &(u, v) in adj {
            if u < n_nodes && v < n_nodes && u != v {
                lists[u].push(v);
                lists[v].push(u);
            }
        }

        let max_ctx = self.config.context_size.max(1);
        let mut visited = vec![false; n_nodes];
        let mut result = Vec::with_capacity(max_ctx);
        let mut queue = std::collections::VecDeque::new();
        let mut rng = Lcg::new(seed);

        visited[center] = true;
        queue.push_back(center);
        result.push(center);

        while let Some(v) = queue.pop_front() {
            if result.len() >= max_ctx {
                break;
            }
            // Shuffle neighbors to avoid deterministic bias
            let mut nbrs = lists[v].clone();
            for i in (1..nbrs.len()).rev() {
                let j = rng.next_usize(i + 1);
                nbrs.swap(i, j);
            }
            for w in nbrs {
                if result.len() >= max_ctx {
                    break;
                }
                if !visited[w] {
                    visited[w] = true;
                    result.push(w);
                    queue.push_back(w);
                }
            }
        }

        result.sort_unstable();
        Ok(result)
    }

    /// Compute InfoNCE (noise-contrastive estimation) loss.
    ///
    /// ```text
    /// L = -log [ exp(sim(a, p) / τ) / (exp(sim(a, p) / τ) + Σ_i exp(sim(a, nᵢ) / τ)) ]
    /// ```
    ///
    /// where `sim` is cosine similarity.
    ///
    /// # Arguments
    /// * `anchor`      – anchor embedding vector.
    /// * `positive`    – positive (context) embedding vector.
    /// * `negatives`   – list of negative embedding vectors.
    /// * `temperature` – temperature τ.
    ///
    /// # Errors
    /// Returns `GraphError::InvalidParameter` if vectors have mismatched dims.
    pub fn contrastive_loss(
        &self,
        anchor: &[f64],
        positive: &[f64],
        negatives: &[Vec<f64>],
        temperature: f64,
    ) -> Result<f64> {
        infonce_loss(anchor, positive, negatives, temperature)
    }
}

/// Cosine similarity between two equal-length vectors. Returns 0 if either
/// vector has zero norm.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Standalone InfoNCE loss function.
///
/// # Errors
/// Returns [`GraphError::InvalidParameter`] if dimension mismatches are found
/// or `temperature <= 0`.
pub fn infonce_loss(
    anchor: &[f64],
    positive: &[f64],
    negatives: &[Vec<f64>],
    temperature: f64,
) -> Result<f64> {
    let dim = anchor.len();
    if dim == 0 {
        return Err(GraphError::InvalidParameter {
            param: "anchor".to_string(),
            value: "empty".to_string(),
            expected: "non-empty embedding vector".to_string(),
            context: "infonce_loss".to_string(),
        });
    }
    if positive.len() != dim {
        return Err(GraphError::InvalidParameter {
            param: "positive".to_string(),
            value: format!("length {}", positive.len()),
            expected: format!("length {dim}"),
            context: "infonce_loss".to_string(),
        });
    }
    if temperature <= 0.0 {
        return Err(GraphError::InvalidParameter {
            param: "temperature".to_string(),
            value: format!("{temperature}"),
            expected: "positive value".to_string(),
            context: "infonce_loss".to_string(),
        });
    }
    for (i, neg) in negatives.iter().enumerate() {
        if neg.len() != dim {
            return Err(GraphError::InvalidParameter {
                param: format!("negatives[{i}]"),
                value: format!("length {}", neg.len()),
                expected: format!("length {dim}"),
                context: "infonce_loss".to_string(),
            });
        }
    }

    let sim_pos = cosine_similarity(anchor, positive) / temperature;
    // Numerically stable: subtract max before exp
    let mut sims: Vec<f64> = std::iter::once(sim_pos)
        .chain(
            negatives
                .iter()
                .map(|n| cosine_similarity(anchor, n) / temperature),
        )
        .collect();
    let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for s in sims.iter_mut() {
        *s = (*s - max_sim).exp();
    }
    let denom: f64 = sims.iter().sum();
    let loss = -(sims[0].ln() - denom.ln());
    Ok(loss)
}

// ─────────────────────────────────────────────────────────────────────────────
// C.  AttributeReconstructionObjective
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the attribute reconstruction MLP.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct AttrReconConfig {
    /// Number of MLP layers. Default 2.
    pub n_layers: usize,
    /// Hidden layer dimension. Default 128.
    pub hidden_dim: usize,
    /// Dropout rate (stored for future training use). Default 0.1.
    pub dropout: f64,
}

impl Default for AttrReconConfig {
    fn default() -> Self {
        Self {
            n_layers: 2,
            hidden_dim: 128,
            dropout: 0.1,
        }
    }
}

/// A weight matrix and bias vector for a single linear layer.
#[derive(Debug, Clone)]
struct LinearLayer {
    /// Weight matrix: `out_dim × in_dim`.
    weights: Vec<Vec<f64>>,
    /// Bias: `out_dim`.
    bias: Vec<f64>,
}

impl LinearLayer {
    /// Initialise with Xavier-uniform-like weights using the given seed.
    fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let mut rng = Lcg::new(seed);
        let scale = (6.0 / (in_dim + out_dim) as f64).sqrt();
        let weights = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| (rng.next_f64() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();
        let bias = vec![0.0f64; out_dim];
        Self { weights, bias }
    }

    /// Apply this layer with tanh activation.  Returns `out_dim` values.
    fn forward_tanh(&self, x: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(self.bias.iter())
            .map(|(row, b)| {
                let pre: f64 = row.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f64>() + b;
                pre.tanh()
            })
            .collect()
    }

    /// Apply this layer with no activation (output layer).
    fn forward_linear(&self, x: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(self.bias.iter())
            .map(|(row, b)| row.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f64>() + b)
            .collect()
    }
}

/// A simple multi-layer perceptron for attribute reconstruction.
///
/// Architecture: `input_dim → hidden_dim → … → input_dim` (tanh between
/// hidden layers, linear output).  No back-propagation is implemented here;
/// this struct provides the forward pass for inference / loss computation.
pub struct AttributeReconstructionObjective {
    config: AttrReconConfig,
    layers: Vec<LinearLayer>,
    input_dim: usize,
}

impl AttributeReconstructionObjective {
    /// Build a new MLP for `input_dim`-dimensional features.
    ///
    /// Layers are randomly initialised.  Use the same `seed` for
    /// reproducibility.
    ///
    /// # Errors
    /// Returns `GraphError::InvalidParameter` if `input_dim == 0` or
    /// `n_layers == 0`.
    pub fn new(input_dim: usize, config: AttrReconConfig, seed: u64) -> Result<Self> {
        if input_dim == 0 {
            return Err(GraphError::InvalidParameter {
                param: "input_dim".to_string(),
                value: "0".to_string(),
                expected: "positive dimension".to_string(),
                context: "AttributeReconstructionObjective::new".to_string(),
            });
        }
        if config.n_layers == 0 {
            return Err(GraphError::InvalidParameter {
                param: "n_layers".to_string(),
                value: "0".to_string(),
                expected: "at least 1 layer".to_string(),
                context: "AttributeReconstructionObjective::new".to_string(),
            });
        }
        let hidden = config.hidden_dim.max(1);
        let mut layers = Vec::with_capacity(config.n_layers);

        // First layer: input → hidden
        layers.push(LinearLayer::new(input_dim, hidden, seed));

        // Intermediate hidden layers
        for i in 1..config.n_layers.saturating_sub(1) {
            layers.push(LinearLayer::new(
                hidden,
                hidden,
                seed.wrapping_add(i as u64),
            ));
        }

        // Final layer: hidden → input (reconstruction)
        if config.n_layers > 1 {
            layers.push(LinearLayer::new(
                hidden,
                input_dim,
                seed.wrapping_add(config.n_layers as u64),
            ));
        }

        Ok(Self {
            config,
            layers,
            input_dim,
        })
    }

    /// Run a forward pass for each node's feature vector.
    ///
    /// # Arguments
    /// * `features` – `n_nodes × input_dim` feature matrix.
    ///
    /// # Returns
    /// Reconstructed feature matrix of the same shape.
    ///
    /// # Errors
    /// Returns `GraphError::InvalidParameter` if any feature vector has
    /// length ≠ `input_dim`.
    pub fn forward(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        features
            .iter()
            .enumerate()
            .map(|(i, f)| {
                if f.len() != self.input_dim {
                    return Err(GraphError::InvalidParameter {
                        param: format!("features[{i}]"),
                        value: format!("length {}", f.len()),
                        expected: format!("length {}", self.input_dim),
                        context: "AttributeReconstructionObjective::forward".to_string(),
                    });
                }
                let mut h = f.clone();
                let last = self.layers.len().saturating_sub(1);
                for (j, layer) in self.layers.iter().enumerate() {
                    h = if j < last {
                        layer.forward_tanh(&h)
                    } else {
                        layer.forward_linear(&h)
                    };
                }
                Ok(h)
            })
            .collect()
    }

    /// Mean squared error between `predicted` and `target`.
    ///
    /// # Errors
    /// Returns `GraphError::InvalidParameter` if shapes differ.
    pub fn mse_loss(&self, predicted: &[Vec<f64>], target: &[Vec<f64>]) -> Result<f64> {
        if predicted.len() != target.len() {
            return Err(GraphError::InvalidParameter {
                param: "predicted".to_string(),
                value: format!("length {}", predicted.len()),
                expected: format!("length {}", target.len()),
                context: "AttributeReconstructionObjective::mse_loss".to_string(),
            });
        }
        if predicted.is_empty() {
            return Ok(0.0);
        }
        let mut total = 0.0_f64;
        let mut count = 0usize;
        for (p_row, t_row) in predicted.iter().zip(target.iter()) {
            if p_row.len() != t_row.len() {
                return Err(GraphError::InvalidParameter {
                    param: "predicted row".to_string(),
                    value: format!("length {}", p_row.len()),
                    expected: format!("length {}", t_row.len()),
                    context: "AttributeReconstructionObjective::mse_loss".to_string(),
                });
            }
            for (a, b) in p_row.iter().zip(t_row.iter()) {
                let diff = a - b;
                total += diff * diff;
                count += 1;
            }
        }
        Ok(if count > 0 { total / count as f64 } else { 0.0 })
    }

    /// Access the underlying config.
    pub fn config(&self) -> &AttrReconConfig {
        &self.config
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── NodeMaskingPretrainer ─────────────────────────────────────────────────

    #[test]
    fn test_masking_correct_fraction() {
        let n = 100;
        let dim = 8;
        let features: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64; dim]).collect();
        let cfg = NodeMaskingConfig {
            mask_rate: 0.15,
            replace_rate: 0.0,
            feature_dim: dim,
            ..Default::default()
        };
        let pretrainer = NodeMaskingPretrainer::new(cfg);
        let (_, indices) = pretrainer.mask_nodes(&features, 7).unwrap();
        // ceil(100 * 0.15) = 15
        assert_eq!(indices.len(), 15, "should mask exactly 15 nodes");
    }

    #[test]
    fn test_masking_features_differ() {
        let n = 20;
        let dim = 4;
        let features: Vec<Vec<f64>> = (0..n).map(|i| vec![(i + 1) as f64; dim]).collect();
        let cfg = NodeMaskingConfig {
            mask_rate: 0.5,
            replace_rate: 0.0,
            feature_dim: dim,
            ..Default::default()
        };
        let pretrainer = NodeMaskingPretrainer::new(cfg);
        let (masked, indices) = pretrainer.mask_nodes(&features, 99).unwrap();
        // Masked nodes should be all zeros
        for &idx in &indices {
            assert_eq!(masked[idx], vec![0.0; dim], "node {idx} should be zeroed");
        }
        // Unmasked nodes should be identical to originals
        for i in 0..n {
            if !indices.contains(&i) {
                assert_eq!(masked[i], features[i], "node {i} should be unchanged");
            }
        }
    }

    #[test]
    fn test_reconstruction_loss_finite_positive() {
        let n = 10;
        let dim = 6;
        let original: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64; dim]).collect();
        let cfg = NodeMaskingConfig {
            mask_rate: 0.3,
            feature_dim: dim,
            ..Default::default()
        };
        let pretrainer = NodeMaskingPretrainer::new(cfg);
        let (masked, indices) = pretrainer.mask_nodes(&original, 11).unwrap();
        let loss = pretrainer
            .reconstruction_loss(&masked, &original, &indices)
            .unwrap();
        assert!(loss.is_finite(), "loss should be finite");
        assert!(loss >= 0.0, "loss should be non-negative");
    }

    // ── GraphContextPretrainer ────────────────────────────────────────────────

    #[test]
    fn test_context_subgraph_bounded() {
        let edges: Vec<(usize, usize)> = (0..9).map(|i| (i, i + 1)).collect();
        let cfg = GraphContextConfig {
            context_size: 4,
            ..Default::default()
        };
        let pretrainer = GraphContextPretrainer::new(cfg.clone());
        let ctx = pretrainer
            .sample_context_subgraph(&edges, 5, 10, 42)
            .unwrap();
        assert!(
            ctx.len() <= cfg.context_size,
            "context size {} should be ≤ {}",
            ctx.len(),
            cfg.context_size
        );
    }

    #[test]
    fn test_context_subgraph_contains_center() {
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let cfg = GraphContextConfig {
            context_size: 3,
            ..Default::default()
        };
        let pretrainer = GraphContextPretrainer::new(cfg);
        let ctx = pretrainer.sample_context_subgraph(&edges, 1, 4, 0).unwrap();
        assert!(ctx.contains(&1), "context should include center node 1");
    }

    #[test]
    fn test_contrastive_loss_pos_closer_lower_loss() {
        // Anchor is [1,0], positive is also [1,0] (cos sim = 1).
        // Negatives are [-1, 0] (cos sim = -1). Loss should be very small.
        let anchor = vec![1.0_f64, 0.0];
        let positive = vec![1.0_f64, 0.0];
        let negatives = vec![vec![-1.0_f64, 0.0]; 4];
        let cfg = GraphContextConfig {
            temperature: 0.07,
            ..Default::default()
        };
        let pretrainer = GraphContextPretrainer::new(cfg.clone());
        let loss = pretrainer
            .contrastive_loss(&anchor, &positive, &negatives, cfg.temperature)
            .unwrap();
        // Now make negative similar to anchor
        let far_negatives = vec![vec![1.0_f64, 0.0]; 4];
        let high_loss = pretrainer
            .contrastive_loss(&anchor, &positive, &far_negatives, cfg.temperature)
            .unwrap();
        assert!(
            loss < high_loss,
            "loss with far negatives ({loss}) should be lower than loss with close negatives ({high_loss})"
        );
    }

    #[test]
    fn test_contrastive_loss_finite() {
        let anchor = vec![0.5, 0.3, 0.2];
        let positive = vec![0.4, 0.4, 0.2];
        let negatives = vec![vec![0.1, 0.1, 0.8], vec![-0.1, 0.5, 0.4]];
        let loss = infonce_loss(&anchor, &positive, &negatives, 0.1).unwrap();
        assert!(loss.is_finite(), "InfoNCE loss should be finite");
        assert!(loss >= 0.0, "InfoNCE loss should be non-negative");
    }

    // ── AttributeReconstructionObjective ─────────────────────────────────────

    #[test]
    fn test_attr_recon_forward_shape() {
        let cfg = AttrReconConfig {
            n_layers: 2,
            hidden_dim: 16,
            dropout: 0.0,
        };
        let obj = AttributeReconstructionObjective::new(8, cfg, 123).unwrap();
        let features: Vec<Vec<f64>> = (0..5).map(|_| vec![1.0; 8]).collect();
        let out = obj.forward(&features).unwrap();
        assert_eq!(out.len(), 5, "output should have same number of nodes");
        for row in &out {
            assert_eq!(row.len(), 8, "each output vector should have dim 8");
        }
    }

    #[test]
    fn test_config_defaults() {
        let pr = NodeMaskingConfig::default();
        assert!((pr.mask_rate - 0.15).abs() < 1e-9);
        assert!((pr.replace_rate - 0.1).abs() < 1e-9);
        assert_eq!(pr.n_neighbors, 2);
        assert_eq!(pr.feature_dim, 64);

        let gc = GraphContextConfig::default();
        assert_eq!(gc.context_size, 8);
        assert_eq!(gc.negative_samples, 4);
        assert!((gc.temperature - 0.07).abs() < 1e-9);

        let ar = AttrReconConfig::default();
        assert_eq!(ar.n_layers, 2);
        assert_eq!(ar.hidden_dim, 128);
        assert!((ar.dropout - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_empty_graph_handling() {
        // NodeMaskingPretrainer on empty features
        let cfg = NodeMaskingConfig::default();
        let pretrainer = NodeMaskingPretrainer::new(cfg);
        let (masked, indices) = pretrainer.mask_nodes(&[], 0).unwrap();
        assert!(masked.is_empty());
        assert!(indices.is_empty());

        // GraphContextPretrainer on empty graph
        let cfg2 = GraphContextConfig::default();
        let pretrainer2 = GraphContextPretrainer::new(cfg2);
        let ctx = pretrainer2.sample_context_subgraph(&[], 0, 0, 0).unwrap();
        assert!(ctx.is_empty());
    }
}
