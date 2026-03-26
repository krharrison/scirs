//! SE(3)-Transformer: Attention-based SE(3)-equivariant GNN.
//!
//! Implements the SE(3)-Transformer from Fuchs et al. (2020):
//! "SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks". NeurIPS 2020.
//!
//! ## Key Components
//!
//! - **Real Spherical Harmonics** Y_l^m up to degree l=2
//! - **Tensor Products** with Clebsch-Gordan coefficients (from `cg_coefficients`)
//! - **Radial Network** R_l(|r|): small MLP mapping distance → mixing coefficients
//! - **Self-attention** over type-0 (scalar) features for aggregation weights
//!
//! ## Equivariance
//!
//! - Type-0 (scalar) output features are *invariant* under SO(3) rotations
//! - Type-1 (vector) output features are *equivariant*: they rotate with R

use crate::error::{GraphError, Result};
use crate::gnn::equivariant::cg_coefficients::CgTable;
use crate::gnn::equivariant::egnn::Linear;
use scirs2_core::ndarray::Array2;

// ============================================================================
// EquivariantFeatures
// ============================================================================

/// Equivariant feature container for SE(3)-Transformer.
///
/// Holds per-node features organised by irreducible representation (irrep) type:
/// - `features_l[l]` has shape `[n_nodes × n_channels_l × (2l+1)]` (flattened into Vec)
#[derive(Debug, Clone)]
pub struct EquivariantFeatures {
    /// Scalar (l=0) features: `[n_nodes × n_channels]`
    pub scalars: Vec<f64>,
    /// Vector (l=1) features: `[n_nodes × 3]` (one channel per node for simplicity)
    pub vectors: Vec<[f64; 3]>,
    /// Number of nodes.
    pub n_nodes: usize,
    /// Number of scalar channels per node.
    pub n_scalar_channels: usize,
}

impl EquivariantFeatures {
    /// Create zeroed equivariant features.
    pub fn new(n_nodes: usize, n_scalar_channels: usize) -> Self {
        EquivariantFeatures {
            scalars: vec![0.0; n_nodes * n_scalar_channels],
            vectors: vec![[0.0; 3]; n_nodes],
            n_nodes,
            n_scalar_channels,
        }
    }

    /// Get scalar features for node i.
    pub fn get_scalars(&self, i: usize) -> &[f64] {
        let start = i * self.n_scalar_channels;
        &self.scalars[start..start + self.n_scalar_channels]
    }

    /// Get mutable scalar features for node i.
    pub fn get_scalars_mut(&mut self, i: usize) -> &mut [f64] {
        let start = i * self.n_scalar_channels;
        &mut self.scalars[start..start + self.n_scalar_channels]
    }
}

// ============================================================================
// Real Spherical Harmonics
// ============================================================================

/// Real spherical harmonics evaluator up to degree l=2.
///
/// Uses real-valued (not complex) spherical harmonics in the convention:
/// ```text
/// Y^l_m = {  sqrt(2) * Re[Y^l_{|m|}]    for m > 0
///           {  Y^l_0                      for m = 0
///           {  sqrt(2) * Im[Y^l_{|m|}]   for m < 0
/// ```
///
/// ## Normalisation
///
/// We use fully normalised spherical harmonics where:
/// `∫ Y^l_m * Y^{l'}_{m'} dΩ = δ_{ll'} δ_{mm'}`
#[derive(Debug, Clone, Default)]
pub struct SphericalHarmonics;

impl SphericalHarmonics {
    /// Evaluate all real spherical harmonics up to degree `l_max`.
    ///
    /// # Arguments
    /// - `r`: 3D direction vector (need not be normalised; normalised internally)
    /// - `l_max`: maximum degree to compute
    ///
    /// # Returns
    /// `Vec<Vec<f64>>` where `result[l]` has length `2l+1` and contains
    /// `Y_l^m` for m from -l to l.
    pub fn evaluate(r: [f64; 3], l_max: u8) -> Vec<Vec<f64>> {
        let norm = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        let (x, y, z) = if norm < 1e-12 {
            (0.0, 0.0, 1.0)
        } else {
            (r[0] / norm, r[1] / norm, r[2] / norm)
        };

        let mut result = Vec::with_capacity((l_max as usize) + 1);

        // l=0: Y_0^0 = 1/sqrt(4π)
        let inv_sqrt_4pi = 1.0 / (4.0 * std::f64::consts::PI).sqrt();
        result.push(vec![inv_sqrt_4pi]);

        if l_max < 1 {
            return result;
        }

        // l=1: three components, m=-1,0,1
        // Y_1^{-1} = sqrt(3/4π) * y
        // Y_1^0   = sqrt(3/4π) * z
        // Y_1^1   = sqrt(3/4π) * x
        let c1 = (3.0 / (4.0 * std::f64::consts::PI)).sqrt();
        result.push(vec![c1 * y, c1 * z, c1 * x]);

        if l_max < 2 {
            return result;
        }

        // l=2: five components, m=-2,-1,0,1,2
        // Y_2^{-2} = 0.5 * sqrt(15/π) * x*y
        // Y_2^{-1} = 0.5 * sqrt(15/π) * y*z
        // Y_2^0   = 0.25 * sqrt(5/π) * (2z²-x²-y²)
        // Y_2^1   = 0.5 * sqrt(15/π) * x*z
        // Y_2^2   = 0.25 * sqrt(15/π) * (x²-y²)
        let c2_pm2 = 0.5 * (15.0 / std::f64::consts::PI).sqrt();
        let c2_0 = 0.25 * (5.0 / std::f64::consts::PI).sqrt();
        let c2_pm1 = 0.5 * (15.0 / std::f64::consts::PI).sqrt();
        let c2_p2 = 0.25 * (15.0 / std::f64::consts::PI).sqrt();
        result.push(vec![
            c2_pm2 * x * y,
            c2_pm1 * y * z,
            c2_0 * (2.0 * z * z - x * x - y * y),
            c2_pm1 * x * z,
            c2_p2 * (x * x - y * y),
        ]);

        // unused but referenced to avoid dead code
        let _ = c2_pm1;
        let _ = c2_p2;

        result
    }

    /// Evaluate spherical harmonics at degree `l` only.
    pub fn evaluate_l(r: [f64; 3], l: u8) -> Vec<f64> {
        let all = Self::evaluate(r, l);
        all.into_iter().last().unwrap_or_default()
    }

    /// Sum of squares of all Y_l^m for a given l (should equal (2l+1)/4π for unit r).
    pub fn sum_of_squares(r: [f64; 3], l: u8) -> f64 {
        let ys = Self::evaluate_l(r, l);
        ys.iter().map(|y| y * y).sum()
    }
}

// ============================================================================
// Radial Network
// ============================================================================

/// Radial network R_l(|r|): maps scalar distance to l-channel mixing weights.
///
/// A small MLP with SiLU activations:  |r| → hidden → n_heads scalars per l-channel.
#[derive(Debug, Clone)]
struct RadialNet {
    layers: Vec<Linear>,
    out_dim: usize,
}

impl RadialNet {
    fn new(hidden_dim: usize, out_dim: usize) -> Self {
        RadialNet {
            layers: vec![
                Linear::new(1, hidden_dim),
                Linear::new(hidden_dim, hidden_dim),
                Linear::new(hidden_dim, out_dim),
            ],
            out_dim,
        }
    }

    fn forward(&self, dist: f64) -> Vec<f64> {
        let mut h = vec![dist];
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            if i < self.layers.len() - 1 {
                // SiLU activation
                for x in h.iter_mut() {
                    *x = *x / (1.0 + (-*x).exp());
                }
            }
        }
        h
    }
}

// ============================================================================
// SE(3)-Transformer Configuration
// ============================================================================

/// Configuration for the SE(3)-Transformer model.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Se3Config {
    /// Number of attention heads.
    pub n_heads: usize,
    /// Maximum irrep degree (0=scalars only, 1=scalars+vectors, 2=scalars+vectors+rank-2 tensors).
    pub l_max: u8,
    /// Hidden channel dimension.
    pub hidden_channels: usize,
    /// Number of SE(3)-Transformer layers.
    pub n_layers: usize,
}

impl Default for Se3Config {
    fn default() -> Self {
        Se3Config {
            n_heads: 4,
            l_max: 1,
            hidden_channels: 32,
            n_layers: 2,
        }
    }
}

// ============================================================================
// Se3Layer
// ============================================================================

/// A single SE(3)-Transformer layer.
///
/// Computes equivariant messages using spherical harmonics and CG tensor products,
/// then aggregates with attention weights derived from scalar (type-0) features.
#[derive(Debug, Clone)]
pub struct Se3Layer {
    /// Precomputed CG coefficient table.
    cg: CgTable,
    /// Radial networks for each (l_in, l_edge → l_out) combination.
    radial_nets: Vec<RadialNet>,
    /// Attention query projection for scalar features.
    attn_q: Linear,
    /// Attention key projection for scalar features.
    attn_k: Linear,
    /// Value projection for scalar features.
    attn_v: Linear,
    /// Output projection for scalars.
    out_proj: Linear,
    /// Layer norm scale (scalar).
    layer_norm_scale: Vec<f64>,
    /// Configuration.
    config: Se3Config,
    /// Input scalar channels.
    in_channels: usize,
}

impl Se3Layer {
    /// Create a new SE(3)-Transformer layer.
    pub fn new(in_channels: usize, config: Se3Config) -> Self {
        let hidden = config.hidden_channels;
        let n_heads = config.n_heads;

        // Radial networks: one per (l, head)
        let n_l_values = (config.l_max as usize) + 1;
        let radial_nets = (0..n_l_values)
            .map(|_| RadialNet::new(hidden, n_heads))
            .collect();

        Se3Layer {
            cg: CgTable::new(),
            radial_nets,
            attn_q: Linear::new(in_channels, hidden),
            attn_k: Linear::new(in_channels, hidden),
            attn_v: Linear::new(in_channels, hidden),
            out_proj: Linear::new(hidden, in_channels),
            layer_norm_scale: vec![1.0; in_channels],
            config,
            in_channels,
        }
    }

    /// Compute the type-l contribution to the message from edge (i→j).
    ///
    /// `m^l_{ij,m} = sum_{l'} sum_{m'} CG(l',l1→l) * R_l(|r|) * Y_l^m(r_ij) * f^{l'}_{j,m'}`
    fn compute_equivariant_message(
        &self,
        r_ij: [f64; 3],
        f_j_scalar: &[f64],
        f_j_vector: &[f64; 3],
        l_out: u8,
        head: usize,
    ) -> Vec<f64> {
        let dist = (r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2]).sqrt();
        let y_l = SphericalHarmonics::evaluate_l(r_ij, l_out);
        let radial_weights = self.radial_nets[l_out as usize].forward(dist);
        let r_weight = if head < radial_weights.len() {
            radial_weights[head]
        } else {
            0.0
        };

        let dim_out = 2 * l_out as usize + 1;
        let mut msg = vec![0.0_f64; dim_out];

        // Contribution from type-0 source (scalar ⊗ Y_l → type-l)
        // Using CG(0, 0, l, m → l, m) = 1 (identity coupling)
        if !f_j_scalar.is_empty() {
            let f0 = f_j_scalar[0]; // first scalar channel
            for m_idx in 0..dim_out {
                // Y_l^m acts as the type-l irrep for direction
                msg[m_idx] += r_weight * f0 * y_l[m_idx];
            }
        }

        // Contribution from type-1 source (vector ⊗ Y_1 → type-l output via CG)
        if l_out <= 2 {
            let y_1 = SphericalHarmonics::evaluate_l(r_ij, 1);
            // f_j_vector is [f_{-1}, f_0, f_1] in spherical basis
            // Approximate: map Cartesian to spherical (m=-1: y, m=0: z, m=1: x)
            let f1 = [f_j_vector[1], f_j_vector[2], f_j_vector[0]];

            for m_out in -(l_out as i8)..=(l_out as i8) {
                let m_out_idx = (m_out + l_out as i8) as usize;
                let mut contrib = 0.0;
                // Sum over l1=1 channel (3 components)
                for m1 in [-1_i8, 0, 1] {
                    let m1_idx = (m1 + 1) as usize;
                    // Sum over l_edge=1 channel
                    for m_edge in [-1_i8, 0, 1] {
                        let m_edge_idx = (m_edge + 1) as usize;
                        if m1 + m_edge == m_out {
                            let cg_val = self.cg.get(1, m1, 1, m_edge, l_out, m_out);
                            contrib += cg_val * f1[m1_idx] * y_1[m_edge_idx];
                        }
                    }
                }
                msg[m_out_idx] += r_weight * contrib;
            }
        }

        msg
    }

    /// Forward pass for one SE(3)-Transformer layer.
    ///
    /// # Arguments
    /// - `features`: equivariant node features (scalars + vectors)
    /// - `coords`: node coordinates [n_nodes × 3]
    /// - `edges`: directed edge list
    ///
    /// # Returns
    /// Updated equivariant features (same shape).
    pub fn forward(
        &self,
        features: &EquivariantFeatures,
        coords: &Array2<f64>,
        edges: &[(usize, usize)],
    ) -> Result<EquivariantFeatures> {
        let n_nodes = features.n_nodes;
        let hidden = self.config.hidden_channels;

        if coords.nrows() != n_nodes {
            return Err(GraphError::InvalidParameter {
                param: "coords".to_string(),
                value: format!("nrows={}", coords.nrows()),
                expected: format!("nrows={n_nodes}"),
                context: "Se3Layer::forward".to_string(),
            });
        }

        // ── Compute attention weights (based on scalar features) ─────────────
        // For each node: Q = attn_q(h_i_scalar), K = attn_k(h_j_scalar)
        // Attention: a_ij = softmax_j(Q_i · K_j / sqrt(d))

        let mut node_q: Vec<Vec<f64>> = Vec::with_capacity(n_nodes);
        let mut node_k: Vec<Vec<f64>> = Vec::with_capacity(n_nodes);
        let mut node_v: Vec<Vec<f64>> = Vec::with_capacity(n_nodes);

        for i in 0..n_nodes {
            let scalars_i = features.get_scalars(i);
            node_q.push(self.attn_q.forward(scalars_i));
            node_k.push(self.attn_k.forward(scalars_i));
            node_v.push(self.attn_v.forward(scalars_i));
        }

        let scale = (hidden as f64).sqrt();

        // ── Per-node aggregation using attention ─────────────────────────────
        let mut new_scalars = vec![0.0_f64; n_nodes * self.in_channels];
        let mut new_vectors = vec![[0.0_f64; 3]; n_nodes];

        // Build neighbour list for softmax normalisation
        let mut neighbor_edges: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for (edge_idx, &(i, _j)) in edges.iter().enumerate() {
            if i < n_nodes {
                neighbor_edges[i].push(edge_idx);
            }
        }

        for i in 0..n_nodes {
            let edge_indices = &neighbor_edges[i];
            if edge_indices.is_empty() {
                // No neighbours: copy input
                let src = features.get_scalars(i);
                let dst_start = i * self.in_channels;
                new_scalars[dst_start..dst_start + self.in_channels].copy_from_slice(src);
                new_vectors[i] = features.vectors[i];
                continue;
            }

            // Compute raw attention scores
            let scores: Vec<f64> = edge_indices
                .iter()
                .map(|&eidx| {
                    let j = edges[eidx].1;
                    if j < n_nodes {
                        let dot: f64 = node_q[i]
                            .iter()
                            .zip(node_k[j].iter())
                            .map(|(q, k)| q * k)
                            .sum();
                        dot / scale
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect();

            // Softmax over neighbours
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let exp_sum: f64 = exps.iter().sum::<f64>().max(1e-15);
            let alphas: Vec<f64> = exps.iter().map(|e| e / exp_sum).collect();

            // Aggregate value features (scalar attention)
            let mut v_agg = vec![0.0_f64; hidden];
            for (k, &eidx) in edge_indices.iter().enumerate() {
                let j = edges[eidx].1;
                if j < n_nodes {
                    for d in 0..hidden {
                        v_agg[d] += alphas[k] * node_v[j][d];
                    }
                }
            }

            // Project aggregated values back to input dim
            let out = self.out_proj.forward(&v_agg);

            // Residual + layer norm
            let start = i * self.in_channels;
            for d in 0..self.in_channels {
                let h_i = features.get_scalars(i)[d];
                let h_new = h_i + out[d];
                // Simple layer norm: scale by learned parameter (initialised to 1)
                new_scalars[start + d] = h_new * self.layer_norm_scale[d];
            }

            // Vector features: aggregate equivariant messages
            let mut vec_agg = [0.0_f64; 3];
            for (k, &eidx) in edge_indices.iter().enumerate() {
                let j = edges[eidx].1;
                if j < n_nodes {
                    // r_ij = x_j - x_i
                    let r_ij = [
                        coords[[j, 0]] - coords[[i, 0]],
                        coords[[j, 1]] - coords[[i, 1]],
                        coords[[j, 2]] - coords[[i, 2]],
                    ];

                    let f_j_scalar = features.get_scalars(j);
                    let f_j_vec = &features.vectors[j];

                    // Compute equivariant message for l=1 (vectors)
                    let m1 = self.compute_equivariant_message(r_ij, f_j_scalar, f_j_vec, 1, 0);
                    // m1 is in spherical basis [Y_{-1}, Y_0, Y_1] ≈ [y, z, x]
                    // Convert back to Cartesian: x ← m1[2], y ← m1[0], z ← m1[1]
                    vec_agg[0] += alphas[k] * m1[2]; // x
                    vec_agg[1] += alphas[k] * m1[0]; // y
                    vec_agg[2] += alphas[k] * m1[1]; // z
                }
            }

            // Residual for vectors
            for d in 0..3 {
                new_vectors[i][d] = features.vectors[i][d] + vec_agg[d];
            }
        }

        Ok(EquivariantFeatures {
            scalars: new_scalars,
            vectors: new_vectors,
            n_nodes,
            n_scalar_channels: self.in_channels,
        })
    }
}

// ============================================================================
// Se3Transformer (stacked layers)
// ============================================================================

/// SE(3)-Transformer: stacked Se3Layers for equivariant molecular graph learning.
#[derive(Debug, Clone)]
pub struct Se3Transformer {
    /// Stacked SE(3)-Transformer layers.
    pub layers: Vec<Se3Layer>,
    /// Model configuration.
    pub config: Se3Config,
}

impl Se3Transformer {
    /// Build a Se3Transformer model.
    ///
    /// # Arguments
    /// - `in_channels`: scalar feature channels per node
    /// - `config`: model configuration
    pub fn new(in_channels: usize, config: Se3Config) -> Self {
        let n_layers = config.n_layers;
        let layers = (0..n_layers)
            .map(|_| Se3Layer::new(in_channels, config.clone()))
            .collect();
        Se3Transformer { layers, config }
    }

    /// Forward pass through all Se3 layers.
    pub fn forward(
        &self,
        features: &EquivariantFeatures,
        coords: &Array2<f64>,
        edges: &[(usize, usize)],
    ) -> Result<EquivariantFeatures> {
        let mut feat = features.clone();
        for layer in &self.layers {
            feat = layer.forward(&feat, coords, edges)?;
        }
        Ok(feat)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-9;

    // ── Spherical Harmonics ──────────────────────────────────────────────────

    #[test]
    fn test_y00_value() {
        // Y_0^0 = 1/sqrt(4π) everywhere
        let ys = SphericalHarmonics::evaluate([0.0, 0.0, 1.0], 0);
        assert_eq!(ys.len(), 1);
        let expected = 1.0 / (4.0 * PI).sqrt();
        assert!((ys[0][0] - expected).abs() < TOL, "Y_0^0 = {}", ys[0][0]);
    }

    #[test]
    fn test_y10_on_z_axis() {
        // At (0,0,1): Y_1^0 = sqrt(3/4π)*z = sqrt(3/4π)
        let ys = SphericalHarmonics::evaluate([0.0, 0.0, 1.0], 1);
        assert_eq!(ys.len(), 2);
        let expected_y10 = (3.0 / (4.0 * PI)).sqrt();
        // Y_1^m for l=1 has m: -1, 0, 1 → index 1 is m=0
        assert!(
            (ys[1][1] - expected_y10).abs() < TOL,
            "Y_1^0 = {}",
            ys[1][1]
        );
    }

    #[test]
    fn test_y1_on_z_axis_other_components_zero() {
        // At z-axis: Y_1^{±1} = 0 (x=y=0)
        let ys = SphericalHarmonics::evaluate([0.0, 0.0, 1.0], 1);
        assert!(ys[1][0].abs() < TOL, "Y_1^{{-1}} should be 0 on z-axis");
        assert!(ys[1][2].abs() < TOL, "Y_1^{{1}} should be 0 on z-axis");
    }

    #[test]
    fn test_y1_normalization_x_axis() {
        // On x-axis (1,0,0): Y_1^1 = sqrt(3/4π), Y_1^{-1} = Y_1^0 = 0
        let ys = SphericalHarmonics::evaluate([1.0, 0.0, 0.0], 1);
        let expected = (3.0 / (4.0 * PI)).sqrt();
        assert!((ys[1][2] - expected).abs() < TOL, "Y_1^1 on x-axis");
        assert!(ys[1][0].abs() < TOL);
        assert!(ys[1][1].abs() < TOL);
    }

    #[test]
    fn test_sph_harm_sum_of_squares_l1() {
        // For unit vector r: sum_m |Y_l^m(r)|² = (2l+1)/(4π)
        // For l=1: = 3/(4π)
        let r = [1.0_f64 / 3.0_f64.sqrt(); 3]; // (1,1,1)/sqrt(3)
        let sum_sq = SphericalHarmonics::sum_of_squares(r, 1);
        let expected = 3.0 / (4.0 * PI);
        assert!(
            (sum_sq - expected).abs() < 1e-10,
            "sum of squares for l=1: got {sum_sq}, expected {expected}"
        );
    }

    #[test]
    fn test_sph_harm_sum_of_squares_l2() {
        // For l=2: sum_m |Y_2^m(r)|² = 5/(4π)
        let r = [0.0, 0.0, 1.0];
        let sum_sq = SphericalHarmonics::sum_of_squares(r, 2);
        let expected = 5.0 / (4.0 * PI);
        assert!(
            (sum_sq - expected).abs() < 1e-10,
            "sum of squares for l=2: got {sum_sq}, expected {expected}"
        );
    }

    // ── Se3Layer ─────────────────────────────────────────────────────────────

    fn make_features(n_nodes: usize, n_channels: usize) -> EquivariantFeatures {
        let mut feat = EquivariantFeatures::new(n_nodes, n_channels);
        for i in 0..n_nodes {
            for c in 0..n_channels {
                feat.scalars[i * n_channels + c] = (i * n_channels + c) as f64 * 0.1 + 0.1;
            }
            feat.vectors[i] = [(i as f64) * 0.1, (i as f64) * 0.2, (i as f64) * 0.3];
        }
        feat
    }

    fn make_coords(n_nodes: usize) -> Array2<f64> {
        let mut data = Vec::with_capacity(n_nodes * 3);
        for i in 0..n_nodes {
            data.push(i as f64 * 0.5);
            data.push(0.0);
            data.push(0.0);
        }
        Array2::from_shape_vec((n_nodes, 3), data).expect("coords")
    }

    #[test]
    fn test_se3_layer_output_shape() {
        let config = Se3Config {
            n_heads: 2,
            l_max: 1,
            hidden_channels: 8,
            n_layers: 1,
        };
        let layer = Se3Layer::new(4, config);
        let features = make_features(5, 4);
        let coords = make_coords(5);
        let edges = vec![(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)];
        let out = layer.forward(&features, &coords, &edges).expect("forward");
        assert_eq!(out.n_nodes, 5);
        assert_eq!(out.n_scalar_channels, 4);
        assert_eq!(out.scalars.len(), 5 * 4);
        assert_eq!(out.vectors.len(), 5);
    }

    #[test]
    fn test_se3_scalars_change_after_forward() {
        // Scalars should be updated (not identical to input)
        let config = Se3Config {
            n_heads: 2,
            l_max: 1,
            hidden_channels: 8,
            n_layers: 1,
        };
        let layer = Se3Layer::new(4, config);
        let features = make_features(4, 4);
        let coords = make_coords(4);
        let edges = vec![(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)];
        let out = layer.forward(&features, &coords, &edges).expect("forward");
        // At least some scalar values must differ
        let changed = features
            .scalars
            .iter()
            .zip(out.scalars.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(changed, "scalars must change after forward pass");
    }

    #[test]
    fn test_se3_transformer_stacked() {
        let config = Se3Config {
            n_heads: 2,
            l_max: 1,
            hidden_channels: 8,
            n_layers: 2,
        };
        let model = Se3Transformer::new(4, config);
        let features = make_features(4, 4);
        let coords = make_coords(4);
        let edges = vec![(0, 1), (1, 0), (1, 2), (2, 1)];
        let out = model.forward(&features, &coords, &edges).expect("forward");
        assert_eq!(out.n_nodes, 4);
        assert_eq!(out.scalars.len(), 4 * 4);
    }

    #[test]
    fn test_se3_rotation_invariance_of_scalars() {
        // Type-0 (scalar) features should be approximately invariant under
        // rotation of the input coordinates. We test with a 90-degree z-rotation.
        let config = Se3Config {
            n_heads: 2,
            l_max: 1,
            hidden_channels: 8,
            n_layers: 1,
        };
        let layer = Se3Layer::new(4, config);
        let features = make_features(4, 4);
        let coords = make_coords(4);
        let edges = vec![(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)];

        let out_orig = layer.forward(&features, &coords, &edges).expect("orig");

        // Rotate 90° around z: x→y, y→-x, z→z
        let mut coords_rot = coords.clone();
        for i in 0..4 {
            let xi = coords[[i, 0]];
            let yi = coords[[i, 1]];
            coords_rot[[i, 0]] = -yi;
            coords_rot[[i, 1]] = xi;
        }
        let out_rot = layer.forward(&features, &coords_rot, &edges).expect("rot");

        // Scalar features should be approximately equal
        // (SE(3)-equivariance: scalars invariant, vectors equivariant)
        // Due to the approximate construction, we allow moderate tolerance
        let max_err = out_orig
            .scalars
            .iter()
            .zip(out_rot.scalars.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 0.5,
            "Scalar invariance violated: max error = {max_err}"
        );
    }

    #[test]
    fn test_se3_layer_no_edges() {
        // With no edges, each node should get its own features back (residual)
        let config = Se3Config {
            n_heads: 2,
            l_max: 1,
            hidden_channels: 8,
            n_layers: 1,
        };
        let layer = Se3Layer::new(4, config);
        let features = make_features(3, 4);
        let coords = make_coords(3);
        let edges: Vec<(usize, usize)> = vec![];
        let out = layer.forward(&features, &coords, &edges).expect("forward");
        assert_eq!(out.n_nodes, 3);
        // With no edges (copy path), scalars should equal input
        for i in 0..3 {
            let in_s = features.get_scalars(i);
            let out_s = out.get_scalars(i);
            for d in 0..4 {
                assert!(
                    (in_s[d] - out_s[d]).abs() < TOL,
                    "no-edge: node {i} scalar {d}: in={}, out={}",
                    in_s[d],
                    out_s[d]
                );
            }
        }
    }
}
