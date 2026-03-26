//! Set Abstraction layer for PointNet++.
//!
//! Each Set Abstraction layer:
//! 1. Sub-samples the point cloud using Farthest Point Sampling (FPS) to select
//!    M centroids.
//! 2. Groups neighbours of each centroid via Ball Query.
//! 3. Applies a shared MLP (with ReLU) to each neighbourhood point and
//!    max-pools the responses, yielding one feature vector per centroid.
//!
//! The `group_all` mode treats the entire input as a single group (used in the
//! final global-feature SA layer).

use crate::error::{Result, VisionError};
use crate::pointnet::sampling::{ball_query, euclidean_dist_sq, farthest_point_sampling};
use scirs2_core::ndarray::{Array1, Array2};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a single Set Abstraction layer.
#[derive(Debug, Clone)]
pub struct SAConfig {
    /// Number of centroids selected by FPS (`group_all` overrides this).
    pub n_points: usize,
    /// Ball-query radius (single-scale SA).
    pub radius: f64,
    /// Maximum number of points per ball group.
    pub max_group_size: usize,
    /// MLP channel sizes `[C1, C2, …, C_out]`.
    pub mlp_channels: Vec<usize>,
    /// If `true`, use all N points as one single group (global pooling mode).
    pub group_all: bool,
}

impl Default for SAConfig {
    fn default() -> Self {
        Self {
            n_points: 512,
            radius: 0.2,
            max_group_size: 32,
            mlp_channels: vec![64, 64, 128],
            group_all: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MLP helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a sequence of linear layers with ReLU activations.
///
/// `weights[l]` has shape `[C_in, C_out]` and `biases[l]` has length `C_out`.
pub(crate) fn mlp_forward(
    weights: &[Array2<f64>],
    biases: &[Array1<f64>],
    x: &Array1<f64>,
) -> Array1<f64> {
    let mut h = x.clone();
    for (w, b) in weights.iter().zip(biases.iter()) {
        let c_out = b.len();
        let mut out = Array1::zeros(c_out);
        for j in 0..c_out {
            let mut val = b[j];
            for i in 0..h.len() {
                val += h[i] * w[[i, j]];
            }
            // ReLU
            out[j] = val.max(0.0);
        }
        h = out;
    }
    h
}

/// Xavier (Glorot) uniform initialisation for a weight matrix `[fan_in, fan_out]`.
fn xavier_init(fan_in: usize, fan_out: usize) -> Array2<f64> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
    // Deterministic pseudo-random using a simple LCG for reproducibility and
    // no external dependency (no rand).
    let mut state: u64 = 0x_dead_beef_cafe_babe;
    let mut w = Array2::zeros((fan_in, fan_out));
    for i in 0..fan_in {
        for j in 0..fan_out {
            // LCG step
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Map to [-limit, limit]
            let t = (state as f64) / (u64::MAX as f64); // [0, 1)
            w[[i, j]] = (2.0 * t - 1.0) * limit;
        }
    }
    w
}

// ─────────────────────────────────────────────────────────────────────────────
// SetAbstraction
// ─────────────────────────────────────────────────────────────────────────────

/// A single Set Abstraction (SA) layer.
///
/// Holds the MLP weight matrices and biases; does not own any data arrays.
pub struct SetAbstraction {
    config: SAConfig,
    /// `weights[l]` has shape `[C_in_l, C_out_l]`
    weights: Vec<Array2<f64>>,
    /// `biases[l]` has length `C_out_l`
    biases: Vec<Array1<f64>>,
}

impl SetAbstraction {
    /// Construct a new SA layer.
    ///
    /// `in_channels` is the number of per-point *feature* channels in the
    /// input (0 if the input is coordinates only).  The MLP input dimension
    /// is `3 + in_channels` (relative xyz coordinates concatenated with
    /// features).
    pub fn new(config: SAConfig, in_channels: usize) -> Result<Self> {
        if config.mlp_channels.is_empty() {
            return Err(VisionError::InvalidParameter(
                "SetAbstraction: mlp_channels must not be empty".to_string(),
            ));
        }

        // MLP input = relative xyz (3 dims) + feature dims
        let first_in = 3 + in_channels;
        let mut dims = vec![first_in];
        dims.extend_from_slice(&config.mlp_channels);

        let mut weights = Vec::with_capacity(dims.len() - 1);
        let mut biases = Vec::with_capacity(dims.len() - 1);

        for pair in dims.windows(2) {
            let fan_in = pair[0];
            let fan_out = pair[1];
            weights.push(xavier_init(fan_in, fan_out));
            biases.push(Array1::zeros(fan_out));
        }

        Ok(Self {
            config,
            weights,
            biases,
        })
    }

    /// Output feature dimension (last channel in `mlp_channels`).
    pub fn out_channels(&self) -> usize {
        self.config.mlp_channels.last().copied().unwrap_or(0)
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `points`: `[N, 3]` XYZ coordinates.
    /// - `features`: optional `[N, C_in]` per-point features.
    ///
    /// # Returns
    /// `(new_xyz [M, 3], new_features [M, C_out])` where M = `n_points`
    /// (or 1 in `group_all` mode).
    pub fn forward(
        &self,
        points: &Array2<f64>,
        features: Option<&Array2<f64>>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let n = points.nrows();
        if points.ncols() != 3 {
            return Err(VisionError::InvalidParameter(
                "SetAbstraction::forward: points must have 3 columns".to_string(),
            ));
        }
        if let Some(f) = features {
            if f.nrows() != n {
                return Err(VisionError::DimensionMismatch(format!(
                    "SetAbstraction::forward: features.nrows ({}) != points.nrows ({})",
                    f.nrows(),
                    n
                )));
            }
        }

        let c_out = self.out_channels();

        if self.config.group_all {
            return self.forward_group_all(points, features, c_out);
        }

        // ── 1. Farthest Point Sampling ────────────────────────────────────────
        let m = self.config.n_points.min(n);
        let centroid_indices = farthest_point_sampling(points, m)?;

        // Build centroid coordinate array [M, 3]
        let mut new_xyz = Array2::zeros((m, 3));
        for (ci, &src) in centroid_indices.iter().enumerate() {
            for d in 0..3 {
                new_xyz[[ci, d]] = points[[src, d]];
            }
        }

        // ── 2. Ball Query ─────────────────────────────────────────────────────
        let groups = ball_query(
            points,
            &new_xyz,
            self.config.radius,
            self.config.max_group_size,
        );

        // ── 3. MLP + max-pool ─────────────────────────────────────────────────
        let mut new_features = Array2::zeros((m, c_out));

        for ci in 0..m {
            let cx = [new_xyz[[ci, 0]], new_xyz[[ci, 1]], new_xyz[[ci, 2]]];
            let mut pool: Option<Array1<f64>> = None;

            for &pi in &groups[ci] {
                let feat = self.point_feature(points, features, pi, &cx)?;
                let h = mlp_forward(&self.weights, &self.biases, &feat);
                pool = Some(match pool {
                    None => h,
                    Some(prev) => {
                        Array1::from_iter(prev.iter().zip(h.iter()).map(|(&a, &b)| a.max(b)))
                    }
                });
            }

            if let Some(p) = pool {
                for d in 0..c_out {
                    new_features[[ci, d]] = p[d];
                }
            }
        }

        Ok((new_xyz, new_features))
    }

    // ── group_all forward (single global group) ───────────────────────────────

    fn forward_group_all(
        &self,
        points: &Array2<f64>,
        features: Option<&Array2<f64>>,
        c_out: usize,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let n = points.nrows();
        // Single centroid at the origin (0,0,0) as convention.
        let new_xyz = Array2::zeros((1, 3));
        let cx = [0.0f64, 0.0, 0.0];

        let mut pool: Option<Array1<f64>> = None;
        for pi in 0..n {
            let feat = self.point_feature(points, features, pi, &cx)?;
            let h = mlp_forward(&self.weights, &self.biases, &feat);
            pool = Some(match pool {
                None => h,
                Some(prev) => Array1::from_iter(prev.iter().zip(h.iter()).map(|(&a, &b)| a.max(b))),
            });
        }

        let mut new_features = Array2::zeros((1, c_out));
        if let Some(p) = pool {
            for d in 0..c_out {
                new_features[[0, d]] = p[d];
            }
        }

        Ok((new_xyz, new_features))
    }

    // ── Build the per-point MLP input: relative coords + optional features ────

    fn point_feature(
        &self,
        points: &Array2<f64>,
        features: Option<&Array2<f64>>,
        pi: usize,
        centroid: &[f64; 3],
    ) -> Result<Array1<f64>> {
        let rel_x = points[[pi, 0]] - centroid[0];
        let rel_y = points[[pi, 1]] - centroid[1];
        let rel_z = points[[pi, 2]] - centroid[2];

        match features {
            None => Ok(Array1::from_vec(vec![rel_x, rel_y, rel_z])),
            Some(f) => {
                let c = f.ncols();
                let mut v = Vec::with_capacity(3 + c);
                v.push(rel_x);
                v.push(rel_y);
                v.push(rel_z);
                for d in 0..c {
                    v.push(f[[pi, d]]);
                }
                Ok(Array1::from_vec(v))
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn random_cloud(n: usize) -> Array2<f64> {
        let mut pts = Array2::zeros((n, 3));
        // Simple deterministic pseudo-random values.
        let mut state: u64 = 42;
        for i in 0..n {
            for j in 0..3 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                pts[[i, j]] = (state as f64) / (u64::MAX as f64) * 2.0 - 1.0;
            }
        }
        pts
    }

    #[test]
    fn test_sa_config_default() {
        let cfg = SAConfig::default();
        assert_eq!(cfg.n_points, 512);
        assert!(!cfg.group_all);
    }

    #[test]
    fn test_mlp_forward_shape() {
        let cfg = SAConfig {
            mlp_channels: vec![16, 32],
            ..Default::default()
        };
        let sa = SetAbstraction::new(cfg, 0).expect("SA construction failed");
        let x = Array1::zeros(3_usize);
        let out = mlp_forward(&sa.weights, &sa.biases, &x);
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn test_sa_layer_output_shape() {
        let pts = random_cloud(64);
        let cfg = SAConfig {
            n_points: 16,
            radius: 0.5,
            max_group_size: 8,
            mlp_channels: vec![32, 64],
            group_all: false,
        };
        let sa = SetAbstraction::new(cfg, 0).expect("SA construction failed");
        let (new_xyz, new_feat) = sa.forward(&pts, None).expect("SA forward failed");
        assert_eq!(new_xyz.nrows(), 16);
        assert_eq!(new_xyz.ncols(), 3);
        assert_eq!(new_feat.nrows(), 16);
        assert_eq!(new_feat.ncols(), 64);
    }

    #[test]
    fn test_sa_group_all_mode() {
        let pts = random_cloud(32);
        let cfg = SAConfig {
            group_all: true,
            mlp_channels: vec![32, 64],
            ..Default::default()
        };
        let sa = SetAbstraction::new(cfg, 0).expect("SA construction failed");
        let (new_xyz, new_feat) = sa.forward(&pts, None).expect("SA forward failed");
        assert_eq!(new_xyz.nrows(), 1, "group_all should produce 1 centroid");
        assert_eq!(new_feat.nrows(), 1);
        assert_eq!(new_feat.ncols(), 64);
    }

    #[test]
    fn test_sa_with_features_input() {
        let pts = random_cloud(32);
        let in_c = 4usize;
        let mut feats = Array2::zeros((32, in_c));
        // Fill with some values.
        for i in 0..32 {
            for j in 0..in_c {
                feats[[i, j]] = (i + j) as f64 * 0.1;
            }
        }
        let cfg = SAConfig {
            n_points: 8,
            radius: 0.8,
            max_group_size: 4,
            mlp_channels: vec![32, 64],
            group_all: false,
        };
        let sa = SetAbstraction::new(cfg, in_c).expect("SA construction failed");
        let (new_xyz, new_feat) = sa.forward(&pts, Some(&feats)).expect("SA forward failed");
        assert_eq!(new_xyz.nrows(), 8);
        assert_eq!(new_feat.ncols(), 64);
    }
}
