//! Graphormer-style Graph Transformer
//!
//! Implements the Graphormer architecture from Ying et al. (2021),
//! "Do Transformers Really Perform Bad for Graph Representation Learning?"
//!
//! Key components:
//! - **Centrality encoding**: learnable embeddings based on node in-degree and out-degree
//! - **Spatial encoding**: shortest-path distance (SPD) between node pairs as attention bias
//! - **Edge encoding**: learnable edge feature encoding along shortest paths
//! - **Multi-head self-attention with graph structural bias**:
//!   `A_ij = softmax((Q_i * K_j / sqrt(d)) + spatial_bias[SPD(i,j)] + edge_bias(i,j))`
//! - **Full Graphormer layer**: attention -> FFN -> LayerNorm (pre-norm)

use std::collections::VecDeque;

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Rng, RngExt};

use crate::error::{GraphError, Result};
use crate::gnn::gcn::CsrMatrix;

// ============================================================================
// Centrality Encoding
// ============================================================================

/// Learnable centrality encoding based on node in-degree and out-degree.
///
/// Adds `z_in[deg_in(v)] + z_out[deg_out(v)]` to each node embedding,
/// where `z_in` and `z_out` are learnable embedding tables indexed by degree.
#[derive(Debug, Clone)]
pub struct CentralityEncoding {
    /// Embedding table for in-degree: `[max_degree + 1, hidden_dim]`
    pub in_degree_embed: Array2<f64>,
    /// Embedding table for out-degree: `[max_degree + 1, hidden_dim]`
    pub out_degree_embed: Array2<f64>,
    /// Maximum degree supported
    pub max_degree: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
}

impl CentralityEncoding {
    /// Create a new centrality encoding module.
    ///
    /// # Arguments
    /// * `max_degree` - Maximum node degree to encode (higher degrees are clamped)
    /// * `hidden_dim` - Dimension of the centrality embedding
    pub fn new(max_degree: usize, hidden_dim: usize) -> Self {
        let mut rng = scirs2_core::random::rng();
        let scale = (1.0 / hidden_dim as f64).sqrt();

        let in_degree_embed = Array2::from_shape_fn((max_degree + 1, hidden_dim), |_| {
            (rng.random::<f64>() * 2.0 - 1.0) * scale
        });
        let out_degree_embed = Array2::from_shape_fn((max_degree + 1, hidden_dim), |_| {
            (rng.random::<f64>() * 2.0 - 1.0) * scale
        });

        CentralityEncoding {
            in_degree_embed,
            out_degree_embed,
            max_degree,
            hidden_dim,
        }
    }

    /// Compute in-degree and out-degree for each node from adjacency.
    pub fn compute_degrees(&self, adj: &CsrMatrix) -> (Vec<usize>, Vec<usize>) {
        let n = adj.n_rows;
        let mut in_deg = vec![0usize; n];
        let mut out_deg = vec![0usize; n];

        for (row, col, _) in adj.iter() {
            out_deg[row] += 1;
            if col < n {
                in_deg[col] += 1;
            }
        }

        (in_deg, out_deg)
    }

    /// Encode centrality information and add to node features.
    ///
    /// # Arguments
    /// * `features` - Node features `[n_nodes, hidden_dim]`
    /// * `adj` - Adjacency matrix
    ///
    /// # Returns
    /// Updated features with centrality encoding added
    pub fn forward(&self, features: &Array2<f64>, adj: &CsrMatrix) -> Result<Array2<f64>> {
        let (n, dim) = features.dim();
        if dim != self.hidden_dim {
            return Err(GraphError::InvalidParameter {
                param: "features".to_string(),
                value: format!("dim={dim}"),
                expected: format!("dim={}", self.hidden_dim),
                context: "CentralityEncoding::forward".to_string(),
            });
        }

        let (in_deg, out_deg) = self.compute_degrees(adj);
        let mut output = features.clone();

        for i in 0..n {
            let in_d = in_deg[i].min(self.max_degree);
            let out_d = out_deg[i].min(self.max_degree);
            for j in 0..dim {
                output[[i, j]] +=
                    self.in_degree_embed[[in_d, j]] + self.out_degree_embed[[out_d, j]];
            }
        }

        Ok(output)
    }
}

// ============================================================================
// Spatial Encoding
// ============================================================================

/// Spatial encoding using shortest-path distances (SPD) between node pairs.
///
/// Computes the all-pairs shortest path distance matrix via BFS on unweighted
/// graphs, then provides learnable bias terms indexed by distance.
#[derive(Debug, Clone)]
pub struct SpatialEncoding {
    /// Learnable bias for each distance value: `[max_distance + 1]`
    /// Index 0 = self-loop, index k = distance k
    pub spatial_bias: Array1<f64>,
    /// Maximum distance to encode (larger distances use `max_distance` bias)
    pub max_distance: usize,
}

impl SpatialEncoding {
    /// Create a new spatial encoding module.
    ///
    /// # Arguments
    /// * `max_distance` - Maximum SPD to encode distinctly
    pub fn new(max_distance: usize) -> Self {
        let mut rng = scirs2_core::random::rng();
        let spatial_bias =
            Array1::from_iter((0..=max_distance).map(|_| (rng.random::<f64>() * 2.0 - 1.0) * 0.1));

        SpatialEncoding {
            spatial_bias,
            max_distance,
        }
    }

    /// Compute all-pairs shortest path distances via BFS.
    ///
    /// Returns a matrix `[n, n]` where entry `(i, j)` is the shortest path
    /// distance from node `i` to node `j`. Unreachable pairs get distance
    /// `max_distance + 1`.
    pub fn compute_spd_matrix(&self, adj: &CsrMatrix) -> Array2<usize> {
        let n = adj.n_rows;
        let unreachable = self.max_distance + 1;
        let mut spd = Array2::from_elem((n, n), unreachable);

        // Build adjacency list for BFS
        let mut adj_list: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (row, col, _) in adj.iter() {
            adj_list[row].push(col);
        }

        // BFS from each node
        for src in 0..n {
            spd[[src, src]] = 0;
            let mut queue = VecDeque::new();
            queue.push_back(src);
            let mut visited = vec![false; n];
            visited[src] = true;

            while let Some(u) = queue.pop_front() {
                let dist = spd[[src, u]];
                if dist >= self.max_distance {
                    continue;
                }
                for &v in &adj_list[u] {
                    if !visited[v] {
                        visited[v] = true;
                        spd[[src, v]] = dist + 1;
                        queue.push_back(v);
                    }
                }
            }
        }

        spd
    }

    /// Get the spatial bias matrix `[n, n]` for attention.
    ///
    /// # Arguments
    /// * `adj` - Adjacency matrix
    ///
    /// # Returns
    /// Bias matrix where entry `(i, j)` is the learnable bias for SPD(i, j)
    pub fn forward(&self, adj: &CsrMatrix) -> Array2<f64> {
        let spd = self.compute_spd_matrix(adj);
        let n = adj.n_rows;
        let mut bias = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let d = spd[[i, j]].min(self.max_distance);
                bias[[i, j]] = self.spatial_bias[d];
            }
        }

        bias
    }
}

// ============================================================================
// Edge Encoding
// ============================================================================

/// Edge encoding along shortest paths between node pairs.
///
/// For each pair `(i, j)`, takes the edges along the shortest path from `i` to `j`
/// and averages the learnable edge embeddings. This provides additional structural
/// information to the attention mechanism.
#[derive(Debug, Clone)]
pub struct EdgeEncoding {
    /// Learnable edge embedding: `[max_edge_types, hidden_dim]`
    pub edge_embed: Array2<f64>,
    /// Projection from hidden_dim to scalar bias
    pub projection: Array1<f64>,
    /// Maximum number of edge types
    pub max_edge_types: usize,
    /// Hidden dimension for edge embedding
    pub hidden_dim: usize,
}

impl EdgeEncoding {
    /// Create a new edge encoding module.
    ///
    /// # Arguments
    /// * `max_edge_types` - Maximum number of distinct edge feature types
    /// * `hidden_dim` - Dimension for edge embeddings
    pub fn new(max_edge_types: usize, hidden_dim: usize) -> Self {
        let mut rng = scirs2_core::random::rng();
        let scale = (1.0 / hidden_dim as f64).sqrt();

        let edge_embed = Array2::from_shape_fn((max_edge_types, hidden_dim), |_| {
            (rng.random::<f64>() * 2.0 - 1.0) * scale
        });
        let projection =
            Array1::from_iter((0..hidden_dim).map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale));

        EdgeEncoding {
            edge_embed,
            projection,
            max_edge_types,
            hidden_dim,
        }
    }

    /// Compute edge bias matrix.
    ///
    /// For simplicity, uses edge weights discretized to integer types.
    /// Each edge on the shortest path contributes its embedding, which is
    /// averaged and projected to a scalar.
    ///
    /// # Arguments
    /// * `adj` - Adjacency matrix with edge weights
    /// * `spd` - Shortest path distance matrix from `SpatialEncoding::compute_spd_matrix`
    ///
    /// # Returns
    /// Edge bias matrix `[n, n]`
    pub fn forward(&self, adj: &CsrMatrix, spd: &Array2<usize>) -> Array2<f64> {
        let n = adj.n_rows;
        let mut bias = Array2::zeros((n, n));

        // Build adjacency list with edge types for path reconstruction
        let mut adj_list: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        for (row, col, val) in adj.iter() {
            // Discretize edge weight to type index
            let edge_type = (val.abs() as usize).min(self.max_edge_types - 1);
            adj_list[row].push((col, edge_type));
        }

        // For each pair, reconstruct shortest path edges and compute embedding
        for src in 0..n {
            // BFS parent tracking from src
            let mut parent: Vec<Option<(usize, usize)>> = vec![None; n]; // (parent_node, edge_type)
            let mut visited = vec![false; n];
            visited[src] = true;
            let mut queue = VecDeque::new();
            queue.push_back(src);

            while let Some(u) = queue.pop_front() {
                for &(v, etype) in &adj_list[u] {
                    if !visited[v] {
                        visited[v] = true;
                        parent[v] = Some((u, etype));
                        queue.push_back(v);
                    }
                }
            }

            // For each target, trace back and compute average edge embedding
            for dst in 0..n {
                if src == dst || spd[[src, dst]] == 0 {
                    continue;
                }
                if parent[dst].is_none() {
                    continue; // unreachable
                }

                // Trace path and accumulate edge embeddings
                let mut avg_embed = vec![0.0f64; self.hidden_dim];
                let mut path_len = 0usize;
                let mut cur = dst;

                while let Some((p, etype)) = parent[cur] {
                    for k in 0..self.hidden_dim {
                        avg_embed[k] += self.edge_embed[[etype, k]];
                    }
                    path_len += 1;
                    cur = p;
                    if cur == src {
                        break;
                    }
                }

                if path_len > 0 {
                    let inv = 1.0 / path_len as f64;
                    let mut scalar = 0.0f64;
                    for k in 0..self.hidden_dim {
                        scalar += avg_embed[k] * inv * self.projection[k];
                    }
                    bias[[src, dst]] = scalar;
                }
            }
        }

        bias
    }
}

// ============================================================================
// Multi-head Self-Attention with Graph Structural Bias
// ============================================================================

/// Numerically-stable softmax over a row.
fn softmax_row(row: &[f64]) -> Vec<f64> {
    if row.is_empty() {
        return Vec::new();
    }
    let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = row.iter().map(|x| (x - max_val).exp()).collect();
    let sum = exps.iter().sum::<f64>().max(1e-12);
    exps.iter().map(|e| e / sum).collect()
}

/// Layer normalization over the feature dimension.
fn layer_norm(x: &mut [f64], eps: f64) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    let inv_std = 1.0 / (var + eps).sqrt();
    for v in x.iter_mut() {
        *v = (*v - mean) * inv_std;
    }
}

// ============================================================================
// Graphormer Configuration
// ============================================================================

/// Configuration for the Graphormer model.
#[derive(Debug, Clone)]
pub struct GraphormerConfig {
    /// Input feature dimension
    pub in_dim: usize,
    /// Hidden dimension (also the dimension for Q, K, V)
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of Graphormer layers
    pub num_layers: usize,
    /// FFN intermediate dimension (typically 4 * hidden_dim)
    pub ffn_dim: usize,
    /// Maximum SPD for spatial encoding
    pub max_distance: usize,
    /// Maximum node degree for centrality encoding
    pub max_degree: usize,
    /// Maximum edge types for edge encoding
    pub max_edge_types: usize,
    /// Dropout rate (stored for reference, applied stochastically)
    pub dropout: f64,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
}

impl Default for GraphormerConfig {
    fn default() -> Self {
        GraphormerConfig {
            in_dim: 64,
            hidden_dim: 64,
            num_heads: 4,
            num_layers: 3,
            ffn_dim: 256,
            max_distance: 10,
            max_degree: 50,
            max_edge_types: 4,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }
}

// ============================================================================
// Graphormer Layer
// ============================================================================

/// A single Graphormer transformer layer.
///
/// Implements pre-norm architecture:
/// ```text
/// x = x + MHA(LayerNorm(x), spatial_bias, edge_bias)
/// x = x + FFN(LayerNorm(x))
/// ```
#[derive(Debug, Clone)]
pub struct GraphormerLayer {
    /// Query projection: `[hidden_dim, hidden_dim]`
    pub w_q: Array2<f64>,
    /// Key projection: `[hidden_dim, hidden_dim]`
    pub w_k: Array2<f64>,
    /// Value projection: `[hidden_dim, hidden_dim]`
    pub w_v: Array2<f64>,
    /// Output projection: `[hidden_dim, hidden_dim]`
    pub w_o: Array2<f64>,
    /// FFN first linear: `[hidden_dim, ffn_dim]`
    pub ffn_w1: Array2<f64>,
    /// FFN second linear: `[ffn_dim, hidden_dim]`
    pub ffn_w2: Array2<f64>,
    /// FFN biases
    pub ffn_b1: Array1<f64>,
    /// FFN output bias
    pub ffn_b2: Array1<f64>,
    /// Number of attention heads
    pub num_heads: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
}

impl GraphormerLayer {
    /// Create a new Graphormer layer.
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        ffn_dim: usize,
        layer_norm_eps: f64,
    ) -> Result<Self> {
        if !hidden_dim.is_multiple_of(num_heads) {
            return Err(GraphError::InvalidParameter {
                param: "hidden_dim".to_string(),
                value: format!("{hidden_dim}"),
                expected: format!("divisible by num_heads={num_heads}"),
                context: "GraphormerLayer::new".to_string(),
            });
        }

        let head_dim = hidden_dim / num_heads;
        let mut rng = scirs2_core::random::rng();
        let w_scale = (6.0_f64 / (hidden_dim + hidden_dim) as f64).sqrt();
        let ffn_scale = (6.0_f64 / (hidden_dim + ffn_dim) as f64).sqrt();

        let mut init_w = |rows: usize, cols: usize, scale: f64| -> Array2<f64> {
            Array2::from_shape_fn((rows, cols), |_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
        };

        Ok(GraphormerLayer {
            w_q: init_w(hidden_dim, hidden_dim, w_scale),
            w_k: init_w(hidden_dim, hidden_dim, w_scale),
            w_v: init_w(hidden_dim, hidden_dim, w_scale),
            w_o: init_w(hidden_dim, hidden_dim, w_scale),
            ffn_w1: init_w(hidden_dim, ffn_dim, ffn_scale),
            ffn_w2: init_w(ffn_dim, hidden_dim, ffn_scale),
            ffn_b1: Array1::zeros(ffn_dim),
            ffn_b2: Array1::zeros(hidden_dim),
            num_heads,
            hidden_dim,
            head_dim,
            layer_norm_eps,
        })
    }

    /// Multi-head self-attention with spatial and edge bias.
    ///
    /// # Arguments
    /// * `x` - Input features `[n, hidden_dim]`
    /// * `spatial_bias` - Spatial bias matrix `[n, n]`
    /// * `edge_bias` - Edge bias matrix `[n, n]`
    ///
    /// # Returns
    /// Attention output `[n, hidden_dim]`
    fn multi_head_attention(
        &self,
        x: &Array2<f64>,
        spatial_bias: &Array2<f64>,
        edge_bias: &Array2<f64>,
    ) -> Array2<f64> {
        let n = x.dim().0;
        let d = self.hidden_dim;
        let h = self.num_heads;
        let dk = self.head_dim;
        let scale = 1.0 / (dk as f64).sqrt();

        // Compute Q, K, V: [n, hidden_dim]
        let mut q = Array2::zeros((n, d));
        let mut k = Array2::zeros((n, d));
        let mut v = Array2::zeros((n, d));

        for i in 0..n {
            for j in 0..d {
                let mut sq = 0.0;
                let mut sk = 0.0;
                let mut sv = 0.0;
                for m in 0..d {
                    let xi = x[[i, m]];
                    sq += xi * self.w_q[[m, j]];
                    sk += xi * self.w_k[[m, j]];
                    sv += xi * self.w_v[[m, j]];
                }
                q[[i, j]] = sq;
                k[[i, j]] = sk;
                v[[i, j]] = sv;
            }
        }

        // Multi-head attention with structural bias
        let mut output = Array2::<f64>::zeros((n, d));

        for head in 0..h {
            let offset = head * dk;

            // Compute attention scores for this head: [n, n]
            let mut scores = vec![vec![0.0f64; n]; n];
            for i in 0..n {
                for j in 0..n {
                    let mut dot = 0.0;
                    for m in 0..dk {
                        dot += q[[i, offset + m]] * k[[j, offset + m]];
                    }
                    // Add graph structural bias (shared across heads)
                    scores[i][j] = dot * scale + spatial_bias[[i, j]] + edge_bias[[i, j]];
                }
            }

            // Softmax per row and aggregate values
            for i in 0..n {
                let alphas = softmax_row(&scores[i]);
                for j in 0..n {
                    let a = alphas[j];
                    for m in 0..dk {
                        output[[i, offset + m]] += a * v[[j, offset + m]];
                    }
                }
            }
        }

        // Output projection
        let mut projected = Array2::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                let mut s = 0.0;
                for m in 0..d {
                    s += output[[i, m]] * self.w_o[[m, j]];
                }
                projected[[i, j]] = s;
            }
        }

        projected
    }

    /// Feed-forward network with GELU activation.
    fn ffn(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.dim().0;
        let ffn_dim = self.ffn_w1.dim().1;
        let d = self.hidden_dim;

        // First linear + GELU
        let mut h = Array2::zeros((n, ffn_dim));
        for i in 0..n {
            for j in 0..ffn_dim {
                let mut s = self.ffn_b1[j];
                for m in 0..d {
                    s += x[[i, m]] * self.ffn_w1[[m, j]];
                }
                // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let x3 = s * s * s;
                let inner = std::f64::consts::FRAC_2_PI.sqrt() * (s + 0.044715 * x3);
                h[[i, j]] = 0.5 * s * (1.0 + inner.tanh());
            }
        }

        // Second linear
        let mut out = Array2::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                let mut s = self.ffn_b2[j];
                for m in 0..ffn_dim {
                    s += h[[i, m]] * self.ffn_w2[[m, j]];
                }
                out[[i, j]] = s;
            }
        }

        out
    }

    /// Forward pass of one Graphormer layer.
    ///
    /// Pre-norm architecture:
    /// ```text
    /// x = x + MHA(LayerNorm(x))
    /// x = x + FFN(LayerNorm(x))
    /// ```
    ///
    /// # Arguments
    /// * `x` - Input features `[n, hidden_dim]`
    /// * `spatial_bias` - SPD-based attention bias `[n, n]`
    /// * `edge_bias` - Edge encoding bias `[n, n]`
    pub fn forward(
        &self,
        x: &Array2<f64>,
        spatial_bias: &Array2<f64>,
        edge_bias: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let (n, d) = x.dim();
        if d != self.hidden_dim {
            return Err(GraphError::InvalidParameter {
                param: "x".to_string(),
                value: format!("dim={d}"),
                expected: format!("dim={}", self.hidden_dim),
                context: "GraphormerLayer::forward".to_string(),
            });
        }

        // Pre-norm for attention
        let mut normed = x.clone();
        for i in 0..n {
            let mut row: Vec<f64> = (0..d).map(|j| normed[[i, j]]).collect();
            layer_norm(&mut row, self.layer_norm_eps);
            for j in 0..d {
                normed[[i, j]] = row[j];
            }
        }

        // Multi-head attention + residual
        let attn_out = self.multi_head_attention(&normed, spatial_bias, edge_bias);
        let mut out = x.clone();
        for i in 0..n {
            for j in 0..d {
                out[[i, j]] += attn_out[[i, j]];
            }
        }

        // Pre-norm for FFN
        let mut normed2 = out.clone();
        for i in 0..n {
            let mut row: Vec<f64> = (0..d).map(|j| normed2[[i, j]]).collect();
            layer_norm(&mut row, self.layer_norm_eps);
            for j in 0..d {
                normed2[[i, j]] = row[j];
            }
        }

        // FFN + residual
        let ffn_out = self.ffn(&normed2);
        for i in 0..n {
            for j in 0..d {
                out[[i, j]] += ffn_out[[i, j]];
            }
        }

        Ok(out)
    }
}

// ============================================================================
// Graphormer Model
// ============================================================================

/// Full Graphormer model stacking multiple transformer layers with
/// centrality, spatial, and edge encodings.
#[derive(Debug, Clone)]
pub struct GraphormerModel {
    /// Input projection: `[in_dim, hidden_dim]`
    pub input_proj: Array2<f64>,
    /// Centrality encoding module
    pub centrality_encoding: CentralityEncoding,
    /// Spatial encoding module
    pub spatial_encoding: SpatialEncoding,
    /// Edge encoding module
    pub edge_encoding: EdgeEncoding,
    /// Stack of Graphormer layers
    pub layers: Vec<GraphormerLayer>,
    /// Configuration
    pub config: GraphormerConfig,
}

impl GraphormerModel {
    /// Create a new Graphormer model from configuration.
    pub fn new(config: GraphormerConfig) -> Result<Self> {
        let mut rng = scirs2_core::random::rng();
        let proj_scale = (6.0_f64 / (config.in_dim + config.hidden_dim) as f64).sqrt();
        let input_proj = Array2::from_shape_fn((config.in_dim, config.hidden_dim), |_| {
            (rng.random::<f64>() * 2.0 - 1.0) * proj_scale
        });

        let centrality_encoding = CentralityEncoding::new(config.max_degree, config.hidden_dim);
        let spatial_encoding = SpatialEncoding::new(config.max_distance);
        let edge_encoding = EdgeEncoding::new(config.max_edge_types, config.hidden_dim);

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(GraphormerLayer::new(
                config.hidden_dim,
                config.num_heads,
                config.ffn_dim,
                config.layer_norm_eps,
            )?);
        }

        Ok(GraphormerModel {
            input_proj,
            centrality_encoding,
            spatial_encoding,
            edge_encoding,
            layers,
            config,
        })
    }

    /// Forward pass of the full Graphormer model.
    ///
    /// # Arguments
    /// * `features` - Input node features `[n_nodes, in_dim]`
    /// * `adj` - Sparse adjacency matrix
    ///
    /// # Returns
    /// Node embeddings `[n_nodes, hidden_dim]`
    pub fn forward(&self, features: &Array2<f64>, adj: &CsrMatrix) -> Result<Array2<f64>> {
        let (n, in_dim) = features.dim();
        if in_dim != self.config.in_dim {
            return Err(GraphError::InvalidParameter {
                param: "features".to_string(),
                value: format!("in_dim={in_dim}"),
                expected: format!("in_dim={}", self.config.in_dim),
                context: "GraphormerModel::forward".to_string(),
            });
        }
        if adj.n_rows != n {
            return Err(GraphError::InvalidParameter {
                param: "adj".to_string(),
                value: format!("n_rows={}", adj.n_rows),
                expected: format!("n_rows={n}"),
                context: "GraphormerModel::forward".to_string(),
            });
        }

        // Project input to hidden dim
        let d = self.config.hidden_dim;
        let mut h = Array2::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                let mut s = 0.0;
                for m in 0..in_dim {
                    s += features[[i, m]] * self.input_proj[[m, j]];
                }
                h[[i, j]] = s;
            }
        }

        // Add centrality encoding
        h = self.centrality_encoding.forward(&h, adj)?;

        // Compute structural biases
        let spatial_bias = self.spatial_encoding.forward(adj);
        let spd = self.spatial_encoding.compute_spd_matrix(adj);
        let edge_bias = self.edge_encoding.forward(adj, &spd);

        // Apply Graphormer layers
        for layer in &self.layers {
            h = layer.forward(&h, &spatial_bias, &edge_bias)?;
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

    fn path_csr() -> CsrMatrix {
        // Path graph: 0 -- 1 -- 2 -- 3
        let coo = vec![
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 3, 1.0),
            (3, 2, 1.0),
        ];
        CsrMatrix::from_coo(4, 4, &coo).expect("path CSR")
    }

    fn feats(n: usize, d: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, d), |(i, j)| (i * d + j) as f64 * 0.1)
    }

    #[test]
    fn test_spatial_encoding_spd_matrix() {
        let adj = path_csr();
        let se = SpatialEncoding::new(10);
        let spd = se.compute_spd_matrix(&adj);

        // Self-distances should be 0
        for i in 0..4 {
            assert_eq!(spd[[i, i]], 0, "self-distance should be 0 for node {i}");
        }

        // Adjacent nodes should have distance 1
        assert_eq!(spd[[0, 1]], 1);
        assert_eq!(spd[[1, 2]], 1);
        assert_eq!(spd[[2, 3]], 1);

        // Path distances
        assert_eq!(spd[[0, 2]], 2);
        assert_eq!(spd[[0, 3]], 3);
        assert_eq!(spd[[1, 3]], 2);

        // Symmetry
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(spd[[i, j]], spd[[j, i]], "SPD should be symmetric");
            }
        }
    }

    #[test]
    fn test_centrality_encoding_degrees() {
        let adj = triangle_csr();
        let ce = CentralityEncoding::new(10, 8);
        let (in_deg, out_deg) = ce.compute_degrees(&adj);

        // Triangle: each node has degree 2 (2 outgoing edges in the symmetric representation)
        for i in 0..3 {
            assert_eq!(in_deg[i], 2, "in-degree of node {i}");
            assert_eq!(out_deg[i], 2, "out-degree of node {i}");
        }
    }

    #[test]
    fn test_centrality_encoding_forward_shape() {
        let adj = triangle_csr();
        let ce = CentralityEncoding::new(10, 8);
        let features = feats(3, 8);
        let result = ce.forward(&features, &adj).expect("centrality forward");
        assert_eq!(result.dim(), (3, 8));

        // Output should differ from input (centrality added)
        let mut differs = false;
        for i in 0..3 {
            for j in 0..8 {
                if (result[[i, j]] - features[[i, j]]).abs() > 1e-12 {
                    differs = true;
                }
            }
        }
        assert!(differs, "centrality encoding should modify features");
    }

    #[test]
    fn test_graphormer_attention_with_bias_output_shape() {
        let adj = triangle_csr();
        let config = GraphormerConfig {
            in_dim: 4,
            hidden_dim: 8,
            num_heads: 2,
            num_layers: 1,
            ffn_dim: 16,
            max_distance: 5,
            max_degree: 10,
            max_edge_types: 2,
            ..Default::default()
        };

        let layer = GraphormerLayer::new(8, 2, 16, 1e-5).expect("layer");
        let x = feats(3, 8);
        let se = SpatialEncoding::new(5);
        let spatial_bias = se.forward(&adj);
        let edge_bias = Array2::zeros((3, 3));

        let out = layer
            .forward(&x, &spatial_bias, &edge_bias)
            .expect("forward");
        assert_eq!(out.dim(), (3, 8));
        for &v in out.iter() {
            assert!(v.is_finite(), "output should be finite, got {v}");
        }
    }

    #[test]
    fn test_graphormer_model_forward() {
        let adj = triangle_csr();
        let config = GraphormerConfig {
            in_dim: 4,
            hidden_dim: 8,
            num_heads: 2,
            num_layers: 2,
            ffn_dim: 16,
            max_distance: 5,
            max_degree: 10,
            max_edge_types: 2,
            ..Default::default()
        };

        let model = GraphormerModel::new(config).expect("model");
        let features = feats(3, 4);
        let out = model.forward(&features, &adj).expect("forward");
        assert_eq!(out.dim(), (3, 8));
        for &v in out.iter() {
            assert!(v.is_finite(), "output should be finite, got {v}");
        }
    }

    #[test]
    fn test_graphormer_edge_encoding() {
        let adj = path_csr();
        let se = SpatialEncoding::new(5);
        let spd = se.compute_spd_matrix(&adj);
        let ee = EdgeEncoding::new(2, 4);
        let bias = ee.forward(&adj, &spd);

        assert_eq!(bias.dim(), (4, 4));
        // Diagonal should be 0 (no self-path edges)
        for i in 0..4 {
            assert!(bias[[i, i]].abs() < 1e-12, "self edge bias should be 0");
        }
        // Off-diagonal should have values for connected pairs
        for &v in bias.iter() {
            assert!(v.is_finite(), "edge bias should be finite");
        }
    }

    #[test]
    fn test_graphormer_invalid_hidden_dim() {
        // hidden_dim=7 not divisible by num_heads=2
        let result = GraphormerLayer::new(7, 2, 16, 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn test_spatial_encoding_disconnected() {
        // Two disconnected components: {0, 1} and {2, 3}
        let coo = vec![(0, 1, 1.0), (1, 0, 1.0), (2, 3, 1.0), (3, 2, 1.0)];
        let adj = CsrMatrix::from_coo(4, 4, &coo).expect("disconnected CSR");
        let se = SpatialEncoding::new(5);
        let spd = se.compute_spd_matrix(&adj);

        // Within-component distances
        assert_eq!(spd[[0, 1]], 1);
        assert_eq!(spd[[2, 3]], 1);

        // Cross-component: should be max_distance + 1 = 6
        assert_eq!(spd[[0, 2]], 6);
        assert_eq!(spd[[0, 3]], 6);
        assert_eq!(spd[[1, 2]], 6);
    }
}
