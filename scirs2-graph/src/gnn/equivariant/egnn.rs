//! E(n)-Equivariant Graph Neural Network (EGNN).
//!
//! Implements the EGNN architecture from Satorras et al. (2021):
//! "E(n) Equivariant Graph Neural Networks". ICML 2021.
//!
//! ## Key Properties
//!
//! - **Translation equivariance**: shifting all positions by t shifts output positions by t
//! - **Rotation equivariance**: rotating all positions by R rotates output positions by R
//! - **Reflection equivariance**: included in the E(n) group
//!
//! ## Architecture
//!
//! For each layer:
//! 1. **Message**: `m_ij = φ_m([h_i, h_j, ||x_i - x_j||², e_ij])`
//! 2. **Aggregate**: `M_i = Σ_j m_ij`
//! 3. **Coord update**: `x_i' = x_i + (1/|N(i)|) Σ_j (x_i - x_j) * φ_x(m_ij)`
//! 4. **Feature update**: `h_i' = φ_h([h_i, M_i])` (residual)

use crate::error::{GraphError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Rng, RngExt};

// ============================================================================
// Activation function
// ============================================================================

/// Activation function choices for EGNN layers.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[derive(Default)]
pub enum Activation {
    /// SiLU / Swish: x * sigmoid(x), smooth and effective for GNNs
    #[default]
    Silu,
    /// Rectified Linear Unit: max(0, x)
    Relu,
    /// Scaled Exponential Linear Unit
    Selu,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Identity (no activation)
    Identity,
}

impl Activation {
    /// Apply activation to a single value.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::Silu => x / (1.0 + (-x).exp()),
            Activation::Relu => x.max(0.0),
            Activation::Selu => {
                const ALPHA: f64 = 1.6732_6324_3226_023;
                const SCALE: f64 = 1.0507_0098_6234_957;
                if x >= 0.0 {
                    SCALE * x
                } else {
                    SCALE * ALPHA * (x.exp() - 1.0)
                }
            }
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Identity => x,
        }
    }

    /// Apply activation element-wise to a slice.
    pub fn apply_slice(&self, xs: &mut [f64]) {
        for x in xs.iter_mut() {
            *x = self.apply(*x);
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for an EGNN model.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EgnnConfig {
    /// Hidden feature dimension.
    pub hidden_dim: usize,
    /// Number of EGNN layers.
    pub n_layers: usize,
    /// Activation function.
    pub act: Activation,
    /// Whether to include edge attributes.
    pub use_edge_attr: bool,
    /// Edge attribute dimension (only used when `use_edge_attr = true`).
    pub edge_attr_dim: usize,
    /// Whether to normalise coordinate updates by neighbour count.
    pub normalize_coords: bool,
}

impl Default for EgnnConfig {
    fn default() -> Self {
        EgnnConfig {
            hidden_dim: 64,
            n_layers: 4,
            act: Activation::Silu,
            use_edge_attr: false,
            edge_attr_dim: 0,
            normalize_coords: true,
        }
    }
}

// ============================================================================
// Linear layer helper
// ============================================================================

/// A simple linear layer: y = W x + b.
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix `[out_dim x in_dim]`.
    pub weight: Vec<Vec<f64>>,
    /// Bias vector `[out_dim]`.
    pub bias: Vec<f64>,
    /// Output dimension.
    pub out_dim: usize,
    /// Input dimension.
    pub in_dim: usize,
}

impl Linear {
    /// Create a new linear layer with Kaiming (He) initialisation.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let scale = (2.0 / in_dim as f64).sqrt();
        let mut rng = scirs2_core::random::rng();
        let weight: Vec<Vec<f64>> = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();
        Linear {
            weight,
            bias: vec![0.0; out_dim],
            out_dim,
            in_dim,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut out = self.bias.clone();
        for (i, row) in self.weight.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                out[i] += w * x[j];
            }
        }
        out
    }
}

// ============================================================================
// EgnnLayer
// ============================================================================

/// A single E(n)-equivariant GNN layer.
///
/// The layer maintains equivariance by only using pairwise distances (not
/// relative vectors) in the message network, and by updating coordinates
/// with a radial scalar that multiplies the direction vector.
#[derive(Debug, Clone)]
pub struct EgnnLayer {
    /// Message network φ_m: [h_i, h_j, dist², e_ij] → message
    pub phi_m: Vec<Linear>,
    /// Coordinate network φ_x: message → scalar weight
    pub phi_x: Vec<Linear>,
    /// Feature update network φ_h: [h_i, M_i] → h_i'
    pub phi_h: Vec<Linear>,
    /// Input feature dimension.
    pub input_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Output feature dimension.
    pub output_dim: usize,
    /// Edge attribute dimension (0 if not used).
    pub edge_attr_dim: usize,
    /// Activation function.
    pub act: Activation,
}

impl EgnnLayer {
    /// Create a new EGNN layer with Kaiming initialisation.
    ///
    /// # Arguments
    /// - `input_dim`: node feature dimension (input)
    /// - `hidden_dim`: internal message/coordinate network width
    /// - `output_dim`: node feature dimension (output)
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self::new_with_edge_attr(input_dim, hidden_dim, output_dim, 0)
    }

    /// Create a new EGNN layer that also accepts edge attributes.
    pub fn new_with_edge_attr(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        edge_attr_dim: usize,
    ) -> Self {
        // phi_m: [h_i (input_dim) | h_j (input_dim) | dist² (1) | edge_attr (edge_attr_dim)] → hidden_dim
        let msg_in = 2 * input_dim + 1 + edge_attr_dim;

        let phi_m = vec![
            Linear::new(msg_in, hidden_dim),
            Linear::new(hidden_dim, hidden_dim),
        ];

        // phi_x: hidden_dim → 1  (scalar multiplier for coord update)
        let phi_x = vec![
            Linear::new(hidden_dim, hidden_dim),
            Linear::new(hidden_dim, 1),
        ];

        // phi_h: [h_i (input_dim) | M_i (hidden_dim)] → output_dim
        let phi_h = vec![
            Linear::new(input_dim + hidden_dim, hidden_dim),
            Linear::new(hidden_dim, output_dim),
        ];

        EgnnLayer {
            phi_m,
            phi_x,
            phi_h,
            input_dim,
            hidden_dim,
            output_dim,
            edge_attr_dim,
            act: Activation::default(),
        }
    }

    /// Run the message MLP.
    fn run_phi_m(&self, x: &[f64]) -> Vec<f64> {
        let mut h = x.to_vec();
        for (i, layer) in self.phi_m.iter().enumerate() {
            h = layer.forward(&h);
            if i < self.phi_m.len() - 1 {
                self.act.apply_slice(&mut h);
            }
        }
        h
    }

    /// Run the coordinate MLP.
    fn run_phi_x(&self, m: &[f64]) -> f64 {
        let mut h = m.to_vec();
        for (i, layer) in self.phi_x.iter().enumerate() {
            h = layer.forward(&h);
            if i < self.phi_x.len() - 1 {
                self.act.apply_slice(&mut h);
            }
        }
        // Final output: scalar, tanh-bounded to prevent explosion
        h[0].tanh()
    }

    /// Run the feature update MLP.
    fn run_phi_h(&self, x: &[f64]) -> Vec<f64> {
        let mut h = x.to_vec();
        for (i, layer) in self.phi_h.iter().enumerate() {
            h = layer.forward(&h);
            if i < self.phi_h.len() - 1 {
                self.act.apply_slice(&mut h);
            }
        }
        h
    }

    /// Forward pass of one EGNN layer.
    ///
    /// # Arguments
    /// - `h`: node features, shape [n_nodes × input_dim]
    /// - `x`: node coordinates, shape [n_nodes × 3]
    /// - `edges`: list of (i, j) directed edges
    /// - `edge_attr`: optional edge attributes, shape [n_edges × edge_attr_dim]
    ///
    /// # Returns
    /// `(new_h, new_x)` with shapes [n_nodes × output_dim] and [n_nodes × 3].
    pub fn forward(
        &self,
        h: &Array2<f64>,
        x: &Array2<f64>,
        edges: &[(usize, usize)],
        edge_attr: Option<&Array2<f64>>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let n_nodes = h.nrows();
        if x.nrows() != n_nodes {
            return Err(GraphError::InvalidParameter {
                param: "x".to_string(),
                value: format!("nrows={}", x.nrows()),
                expected: format!("nrows={n_nodes}"),
                context: "EgnnLayer::forward".to_string(),
            });
        }
        if h.ncols() != self.input_dim {
            return Err(GraphError::InvalidParameter {
                param: "h".to_string(),
                value: format!("ncols={}", h.ncols()),
                expected: format!("ncols={}", self.input_dim),
                context: "EgnnLayer::forward".to_string(),
            });
        }
        if x.ncols() != 3 {
            return Err(GraphError::InvalidParameter {
                param: "x".to_string(),
                value: format!("ncols={}", x.ncols()),
                expected: "ncols=3".to_string(),
                context: "EgnnLayer::forward".to_string(),
            });
        }

        // ── Step 1: compute messages for each edge ──────────────────────────
        let n_edges = edges.len();
        let mut messages: Vec<Vec<f64>> = Vec::with_capacity(n_edges);
        let mut coord_weights: Vec<f64> = Vec::with_capacity(n_edges);

        for (edge_idx, &(i, j)) in edges.iter().enumerate() {
            if i >= n_nodes || j >= n_nodes {
                return Err(GraphError::InvalidParameter {
                    param: "edges".to_string(),
                    value: format!("({i},{j})"),
                    expected: format!("indices < {n_nodes}"),
                    context: "EgnnLayer::forward".to_string(),
                });
            }

            // Squared distance
            let diff: Vec<f64> = (0..3).map(|k| x[[i, k]] - x[[j, k]]).collect();
            let dist_sq: f64 = diff.iter().map(|d| d * d).sum();

            // Message input: [h_i, h_j, dist², edge_attr?]
            let mut msg_in: Vec<f64> = h.row(i).to_vec();
            msg_in.extend(h.row(j).iter());
            msg_in.push(dist_sq);
            if let Some(ea) = edge_attr {
                if edge_idx < ea.nrows() {
                    msg_in.extend(ea.row(edge_idx).iter());
                } else {
                    msg_in.extend(std::iter::repeat_n(0.0_f64, self.edge_attr_dim));
                }
            }

            let m_ij = self.run_phi_m(&msg_in);
            let w_ij = self.run_phi_x(&m_ij);
            messages.push(m_ij);
            coord_weights.push(w_ij);
        }

        // ── Step 2: aggregate messages and update coordinates ────────────────
        let mut msg_agg: Vec<Vec<f64>> = vec![vec![0.0; self.hidden_dim]; n_nodes];
        let mut coord_update: Vec<[f64; 3]> = vec![[0.0; 3]; n_nodes];
        let mut neighbor_count: Vec<usize> = vec![0; n_nodes];

        for (edge_idx, &(i, j)) in edges.iter().enumerate() {
            let m_ij = &messages[edge_idx];
            let w_ij = coord_weights[edge_idx];

            // Aggregate messages to node i
            for (k, &m) in m_ij.iter().enumerate() {
                msg_agg[i][k] += m;
            }
            neighbor_count[i] += 1;

            // Coordinate update: (x_i - x_j) * w_ij
            for k in 0..3 {
                coord_update[i][k] += (x[[i, k]] - x[[j, k]]) * w_ij;
            }
        }

        // ── Step 3: update features and positions ───────────────────────────
        let mut new_h = Array2::zeros((n_nodes, self.output_dim));
        let mut new_x = x.clone();

        for node_i in 0..n_nodes {
            // Feature update with residual connection (when dims match)
            let mut feat_in: Vec<f64> = h.row(node_i).to_vec();
            feat_in.extend(msg_agg[node_i].iter());
            let mut h_new_i = self.run_phi_h(&feat_in);

            // Residual connection if input and output dims match
            if self.input_dim == self.output_dim {
                for (k, hk) in h_new_i.iter_mut().enumerate() {
                    *hk += h[[node_i, k]];
                }
            }

            for k in 0..self.output_dim {
                new_h[[node_i, k]] = h_new_i[k];
            }

            // Coordinate update, normalised by neighbour count
            let count = (neighbor_count[node_i].max(1)) as f64;
            for k in 0..3 {
                new_x[[node_i, k]] += coord_update[node_i][k] / count;
            }
        }

        Ok((new_h, new_x))
    }
}

// ============================================================================
// Egnn (stacked layers)
// ============================================================================

/// E(n)-equivariant Graph Neural Network with stacked EGNN layers.
#[derive(Debug, Clone)]
pub struct Egnn {
    /// Stacked EGNN layers.
    pub layers: Vec<EgnnLayer>,
    /// Model configuration.
    pub config: EgnnConfig,
    /// Output MLP for energy/property prediction (optional).
    pub output_mlp: Option<Vec<Linear>>,
}

impl Egnn {
    /// Build an EGNN model from a configuration.
    ///
    /// # Arguments
    /// - `in_dim`: input node feature dimension
    /// - `config`: model configuration
    pub fn new(in_dim: usize, config: EgnnConfig) -> Self {
        let hidden = config.hidden_dim;
        let n_layers = config.n_layers;
        let edge_attr_dim = if config.use_edge_attr {
            config.edge_attr_dim
        } else {
            0
        };

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let layer_in = if i == 0 { in_dim } else { hidden };
            layers.push(EgnnLayer::new_with_edge_attr(
                layer_in,
                hidden,
                hidden,
                edge_attr_dim,
            ));
        }

        // Energy MLP: mean pool → hidden → 1
        let output_mlp = Some(vec![Linear::new(hidden, hidden), Linear::new(hidden, 1)]);

        Egnn {
            layers,
            config,
            output_mlp,
        }
    }

    /// Forward pass through all EGNN layers.
    ///
    /// Returns updated `(node_features, coordinates)`.
    pub fn forward(
        &self,
        h: &Array2<f64>,
        x: &Array2<f64>,
        edges: &[(usize, usize)],
        edge_attr: Option<&Array2<f64>>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let mut h_cur = h.clone();
        let mut x_cur = x.clone();

        for layer in &self.layers {
            let (h_next, x_next) = layer.forward(&h_cur, &x_cur, edges, edge_attr)?;
            h_cur = h_next;
            x_cur = x_next;
        }

        Ok((h_cur, x_cur))
    }

    /// Predict a scalar energy/property for the whole graph.
    ///
    /// Uses mean-pooling over node features, then passes through the output MLP.
    pub fn predict_energy(&self, h: &Array2<f64>) -> f64 {
        let n = h.nrows();
        if n == 0 {
            return 0.0;
        }

        // Mean pool
        let mut pooled = vec![0.0_f64; h.ncols()];
        for i in 0..n {
            for j in 0..h.ncols() {
                pooled[j] += h[[i, j]];
            }
        }
        let inv_n = 1.0 / n as f64;
        for p in pooled.iter_mut() {
            *p *= inv_n;
        }

        if let Some(mlp) = &self.output_mlp {
            let mut out = pooled;
            for (i, layer) in mlp.iter().enumerate() {
                out = layer.forward(&out);
                if i < mlp.len() - 1 {
                    Activation::Silu.apply_slice(&mut out);
                }
            }
            out[0]
        } else {
            pooled.iter().sum::<f64>() * inv_n
        }
    }
}

// ============================================================================
// Equivariance check helper
// ============================================================================

/// Check that a model is translation-equivariant on coordinates.
///
/// Verifies: `model(x + t, h).x ≈ model(x, h).x + t`
/// where `t` is a translation vector.
pub fn check_translation_equivariance(
    model: &Egnn,
    h: &Array2<f64>,
    x: &Array2<f64>,
    edges: &[(usize, usize)],
    t: &[f64; 3],
    tol: f64,
) -> Result<bool> {
    // Compute model(x, h)
    let (_, x_out) = model.forward(h, x, edges, None)?;

    // Compute model(x + t, h)
    let mut x_shifted = x.clone();
    for i in 0..x_shifted.nrows() {
        for k in 0..3 {
            x_shifted[[i, k]] += t[k];
        }
    }
    let (_, x_out_shifted) = model.forward(h, &x_shifted, edges, None)?;

    // Check x_out_shifted ≈ x_out + t
    let mut max_err = 0.0_f64;
    for i in 0..x_out.nrows() {
        for k in 0..3 {
            let err = (x_out_shifted[[i, k]] - (x_out[[i, k]] + t[k])).abs();
            max_err = max_err.max(err);
        }
    }

    Ok(max_err < tol)
}

/// Check that a model is rotation-equivariant on coordinates.
///
/// Verifies: `model(Rx, h).x ≈ R * model(x, h).x`
/// where `R` is a 3×3 rotation matrix.
pub fn check_rotation_equivariance(
    model: &Egnn,
    h: &Array2<f64>,
    x: &Array2<f64>,
    edges: &[(usize, usize)],
    r_mat: &[[f64; 3]; 3],
    tol: f64,
) -> Result<bool> {
    // Rotate input
    let mut x_rot = Array2::zeros((x.nrows(), 3));
    for i in 0..x.nrows() {
        for k in 0..3 {
            let mut val = 0.0;
            for l in 0..3 {
                val += r_mat[k][l] * x[[i, l]];
            }
            x_rot[[i, k]] = val;
        }
    }

    // model(x, h) then rotate output
    let (_, x_out) = model.forward(h, x, edges, None)?;
    let mut x_out_then_rot = Array2::zeros((x_out.nrows(), 3));
    for i in 0..x_out.nrows() {
        for k in 0..3 {
            let mut val = 0.0;
            for l in 0..3 {
                val += r_mat[k][l] * x_out[[i, l]];
            }
            x_out_then_rot[[i, k]] = val;
        }
    }

    // model(Rx, h)
    let (_, x_rot_out) = model.forward(h, &x_rot, edges, None)?;

    // Compare
    let mut max_err = 0.0_f64;
    for i in 0..x_out.nrows() {
        for k in 0..3 {
            let err = (x_rot_out[[i, k]] - x_out_then_rot[[i, k]]).abs();
            max_err = max_err.max(err);
        }
    }

    Ok(max_err < tol)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_simple_graph() -> (Array2<f64>, Array2<f64>, Vec<(usize, usize)>) {
        // 4 nodes with 2D features and 3D coords
        let h = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        )
        .expect("array creation");
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .expect("array creation");
        let edges = vec![(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)];
        (h, x, edges)
    }

    #[test]
    fn test_egnn_layer_output_shapes() {
        let layer = EgnnLayer::new(4, 8, 8);
        let (h, x, edges) = make_simple_graph();
        let (h_out, x_out) = layer.forward(&h, &x, &edges, None).expect("forward pass");
        assert_eq!(h_out.nrows(), 4, "output node count");
        assert_eq!(h_out.ncols(), 8, "output feature dim");
        assert_eq!(x_out.nrows(), 4, "output coord node count");
        assert_eq!(x_out.ncols(), 3, "output coord dim");
    }

    #[test]
    fn test_egnn_stacked_shapes() {
        let config = EgnnConfig {
            hidden_dim: 8,
            n_layers: 3,
            ..Default::default()
        };
        let model = Egnn::new(4, config);
        let (h, x, edges) = make_simple_graph();
        let (h_out, x_out) = model.forward(&h, &x, &edges, None).expect("forward");
        assert_eq!(h_out.nrows(), 4);
        assert_eq!(h_out.ncols(), 8);
        assert_eq!(x_out.shape(), &[4, 3]);
    }

    #[test]
    fn test_egnn_translation_equivariance() {
        let config = EgnnConfig {
            hidden_dim: 8,
            n_layers: 2,
            ..Default::default()
        };
        let model = Egnn::new(4, config);
        let (h, x, edges) = make_simple_graph();
        let t = [1.5, -2.3, 0.7];
        let ok = check_translation_equivariance(&model, &h, &x, &edges, &t, 1e-9)
            .expect("equivariance check");
        assert!(ok, "model must be translation equivariant");
    }

    #[test]
    fn test_egnn_rotation_equivariance() {
        let config = EgnnConfig {
            hidden_dim: 8,
            n_layers: 2,
            ..Default::default()
        };
        let model = Egnn::new(4, config);
        let (h, x, edges) = make_simple_graph();
        // 90-degree rotation around z-axis: [[0,-1,0],[1,0,0],[0,0,1]]
        let r_mat = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let ok = check_rotation_equivariance(&model, &h, &x, &edges, &r_mat, 1e-9)
            .expect("rotation check");
        assert!(ok, "model must be rotation equivariant");
    }

    #[test]
    fn test_activation_silu() {
        let act = Activation::Silu;
        // SiLU(0) = 0
        assert!((act.apply(0.0)).abs() < 1e-12);
        // SiLU(x) > 0 for x > 0
        assert!(act.apply(1.0) > 0.0);
        // SiLU(-10) is close to 0 (near 0 from below)
        assert!(act.apply(-10.0).abs() < 0.001);
    }

    #[test]
    fn test_egnn_energy_prediction() {
        let config = EgnnConfig {
            hidden_dim: 8,
            n_layers: 2,
            ..Default::default()
        };
        let model = Egnn::new(4, config);
        let (h, x, edges) = make_simple_graph();
        let (h_out, _) = model.forward(&h, &x, &edges, None).expect("forward");
        let energy = model.predict_energy(&h_out);
        // Energy should be finite
        assert!(energy.is_finite(), "energy must be finite");
    }

    #[test]
    fn test_egnn_single_node() {
        // Single node, no edges
        let layer = EgnnLayer::new(4, 8, 8);
        let h = Array2::from_shape_vec((1, 4), vec![1.0, 0.5, -0.5, 0.0]).expect("h");
        let x = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).expect("x");
        let edges: Vec<(usize, usize)> = vec![];
        let (h_out, x_out) = layer.forward(&h, &x, &edges, None).expect("forward");
        assert_eq!(h_out.shape(), &[1, 8]);
        assert_eq!(x_out.shape(), &[1, 3]);
    }
}
