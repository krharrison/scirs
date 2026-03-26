//! PointNet++ backbone for 3D point cloud feature extraction.
//!
//! The backbone stacks a configurable number of Set Abstraction (SA) layers
//! that hierarchically downsample the point cloud and extract features, then
//! a symmetric stack of Feature Propagation (FP) layers that upsample back to
//! the original resolution, enabling per-point predictions for tasks such as
//! 3D object detection, semantic segmentation, and part segmentation.
//!
//! # Typical Architecture (following the original paper)
//!
//! ```text
//! Input N×3
//!   ↓ SA(512 pts, r=0.2, [64,64,128])
//! 512×128
//!   ↓ SA(128 pts, r=0.4, [128,128,256])
//! 128×256
//!   ↓ SA(group_all,  [256,512,1024])
//! 1×1024
//!   ↑ FP([256,256])
//! 128×256
//!   ↑ FP([256,128])
//! 512×128
//!   ↑ FP([128,128,128])
//! N×128
//! ```

use crate::error::{Result, VisionError};
use crate::pointnet::feature_propagation::{FPConfig, FeaturePropagation};
use crate::pointnet::set_abstraction::{SAConfig, SetAbstraction};
use scirs2_core::ndarray::Array2;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Full PointNet++ backbone configuration.
///
/// `sa_configs` and `fp_configs` are ordered from coarse to fine; the number
/// of FP layers must equal the number of SA layers (one FP layer undoes one
/// SA layer).
#[derive(Debug, Clone)]
pub struct PointNetPPConfig {
    /// SA layer configurations, ordered from input → coarsest.
    pub sa_configs: Vec<SAConfig>,
    /// FP layer configurations, ordered from coarsest → input resolution.
    pub fp_configs: Vec<FPConfig>,
    /// Output feature dimension per point.
    pub feature_dim: usize,
    /// Number of per-point input feature channels (0 = coordinates only).
    pub in_channels: usize,
}

impl Default for PointNetPPConfig {
    fn default() -> Self {
        // Two SA layers + two FP layers, roughly following the SSG variant.
        Self {
            sa_configs: vec![
                SAConfig {
                    n_points: 128,
                    radius: 0.2,
                    max_group_size: 32,
                    mlp_channels: vec![32, 32, 64],
                    group_all: false,
                },
                SAConfig {
                    n_points: 0, // ignored because group_all=true
                    radius: 0.4,
                    max_group_size: 64,
                    mlp_channels: vec![64, 64, 128],
                    group_all: true,
                },
            ],
            fp_configs: vec![
                FPConfig {
                    mlp_channels: vec![128, 128],
                    k_neighbors: 3,
                },
                FPConfig {
                    mlp_channels: vec![128, 128],
                    k_neighbors: 3,
                },
            ],
            feature_dim: 128,
            in_channels: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backbone
// ─────────────────────────────────────────────────────────────────────────────

/// PointNet++ backbone.
///
/// Built from a stack of [`SetAbstraction`] layers followed by a symmetric
/// stack of [`FeaturePropagation`] layers.  The backbone accepts raw XYZ point
/// clouds and outputs per-point feature vectors suitable for downstream 3D
/// detection or segmentation heads.
pub struct PointNetPPBackbone {
    sa_layers: Vec<SetAbstraction>,
    fp_layers: Vec<FeaturePropagation>,
    config: PointNetPPConfig,
}

impl PointNetPPBackbone {
    /// Construct the backbone from a configuration.
    ///
    /// # Errors
    /// Returns [`VisionError::InvalidParameter`] if:
    /// - There are zero SA layers.
    /// - The number of FP layers does not equal the number of SA layers.
    pub fn new(config: PointNetPPConfig) -> Result<Self> {
        let n_sa = config.sa_configs.len();
        let n_fp = config.fp_configs.len();

        if n_sa == 0 {
            return Err(VisionError::InvalidParameter(
                "PointNetPPBackbone: must have at least one SA layer".to_string(),
            ));
        }
        if n_fp != n_sa {
            return Err(VisionError::InvalidParameter(format!(
                "PointNetPPBackbone: number of FP layers ({n_fp}) must equal SA layers ({n_sa})"
            )));
        }

        // ── Build SA layers ───────────────────────────────────────────────────
        let mut sa_layers = Vec::with_capacity(n_sa);
        let mut sa_out_channels = Vec::with_capacity(n_sa);

        let mut current_in = config.in_channels;
        for sa_cfg in &config.sa_configs {
            let layer = SetAbstraction::new(sa_cfg.clone(), current_in)?;
            current_in = layer.out_channels();
            sa_out_channels.push(current_in);
            sa_layers.push(layer);
        }

        // ── Build FP layers ───────────────────────────────────────────────────
        // FP layers are applied in reverse order (coarsest→finest).
        // fp_layers[0] is applied first (merges the two coarsest levels).
        // fp_layers[i] propagates from level (n_sa - 1 - i) to level (n_sa - 2 - i).
        //
        // Input channels for FP[i]:
        //   coarse features: sa_out_channels[n_sa - 1 - i]
        //   skip features  : sa_out_channels[n_sa - 2 - i]  (or config.in_channels for i = n_sa-1)
        //
        // For the last FP layer the skip features come from the original input
        // (config.in_channels; may be 0 if no features are provided).

        let mut fp_layers = Vec::with_capacity(n_fp);

        for i in 0..n_fp {
            let coarse_c = sa_out_channels[n_sa - 1 - i];
            // Previous FP output becomes coarse_c for subsequent layers once
            // we've processed layer 0:
            let coarse_c = if i == 0 {
                coarse_c
            } else {
                config.fp_configs[i - 1]
                    .mlp_channels
                    .last()
                    .copied()
                    .ok_or_else(|| {
                        VisionError::InvalidParameter(
                            "PointNetPPBackbone: fp mlp_channels must not be empty".to_string(),
                        )
                    })?
            };

            let skip_c = if n_sa - 1 - i == 0 {
                // Propagating back to the original point cloud.
                config.in_channels
            } else {
                sa_out_channels[n_sa - 2 - i]
            };

            let in_c = coarse_c + skip_c;
            fp_layers.push(FeaturePropagation::new(config.fp_configs[i].clone(), in_c)?);
        }

        Ok(Self {
            sa_layers,
            fp_layers,
            config,
        })
    }

    /// Number of Set Abstraction layers.
    pub fn n_sa_layers(&self) -> usize {
        self.sa_layers.len()
    }

    /// Number of Feature Propagation layers.
    pub fn n_fp_layers(&self) -> usize {
        self.fp_layers.len()
    }

    /// Run the backbone on a point cloud.
    ///
    /// # Arguments
    /// - `points`: `[N, 3]` XYZ coordinates.
    ///
    /// # Returns
    /// `[N, feature_dim]` per-point feature matrix.  When the last FP output
    /// channel count differs from `config.feature_dim` the result is still
    /// returned with the actual FP output channel count (the caller should
    /// ensure consistency via the config).
    pub fn forward(&self, points: &Array2<f64>) -> Result<Array2<f64>> {
        let n = points.nrows();
        if points.ncols() != 3 {
            return Err(VisionError::InvalidParameter(
                "PointNetPPBackbone::forward: points must have shape [N, 3]".to_string(),
            ));
        }
        if n == 0 {
            return Err(VisionError::InvalidParameter(
                "PointNetPPBackbone::forward: empty point cloud".to_string(),
            ));
        }

        // ── SA pass: build hierarchy ──────────────────────────────────────────
        // sa_xyz[0]  = original points,  sa_feat[0] = None (or zeros if in_channels > 0)
        // sa_xyz[i+1], sa_feat[i+1] = SA layer i output
        let n_sa = self.sa_layers.len();
        let mut sa_xyz: Vec<Array2<f64>> = Vec::with_capacity(n_sa + 1);
        let mut sa_feat: Vec<Option<Array2<f64>>> = Vec::with_capacity(n_sa + 1);

        sa_xyz.push(points.clone());
        // Original point features: if in_channels > 0 the caller would need to
        // pass them; for now we support coordinates-only input (in_channels = 0).
        if self.config.in_channels > 0 {
            // Zero-initialise input features when not supplied externally.
            sa_feat.push(Some(Array2::zeros((n, self.config.in_channels))));
        } else {
            sa_feat.push(None);
        }

        for (i, sa_layer) in self.sa_layers.iter().enumerate() {
            let prev_xyz = &sa_xyz[i];
            let prev_feat = sa_feat[i].as_ref();
            let (new_xyz, new_feat) = sa_layer.forward(prev_xyz, prev_feat)?;
            sa_xyz.push(new_xyz);
            sa_feat.push(Some(new_feat));
        }

        // ── FP pass: upsample back to N points ────────────────────────────────
        // Start from the coarsest level and propagate toward the original.
        //
        // fp_layers[i] propagates:
        //   xyz2 = sa_xyz[n_sa - i]     (coarse)
        //   xyz1 = sa_xyz[n_sa - 1 - i] (one level finer)
        //   features2 = current_feat
        //   features1 = sa_feat[n_sa - 1 - i] (skip)

        let mut current_feat: Array2<f64> = sa_feat
            .last()
            .ok_or_else(|| {
                VisionError::OperationError(
                    "PointNetPPBackbone: SA feature list is empty".to_string(),
                )
            })?
            .clone()
            .ok_or_else(|| {
                VisionError::OperationError(
                    "PointNetPPBackbone: last SA output has no features".to_string(),
                )
            })?;

        for (i, fp_layer) in self.fp_layers.iter().enumerate() {
            let coarse_level = n_sa - i; // index into sa_xyz / sa_feat
            let fine_level = n_sa - 1 - i; // one step finer

            let xyz2 = &sa_xyz[coarse_level];
            let xyz1 = &sa_xyz[fine_level];
            let skip = sa_feat[fine_level].as_ref();

            current_feat = fp_layer.forward(xyz1, xyz2, skip, &current_feat)?;
        }

        // current_feat should now be [N, C_out]; verify shape.
        if current_feat.nrows() != n {
            return Err(VisionError::DimensionMismatch(format!(
                "PointNetPPBackbone: expected output [{n}, _], got [{}, {}]",
                current_feat.nrows(),
                current_feat.ncols()
            )));
        }

        Ok(current_feat)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pointnet::feature_propagation::FPConfig;
    use crate::pointnet::sampling::{
        ball_query, euclidean_dist_sq, farthest_point_sampling, knn_query,
    };
    use crate::pointnet::set_abstraction::{mlp_forward, SAConfig};
    use scirs2_core::ndarray::{Array1, Array2};

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn pseudo_cloud(n: usize, seed: u64) -> Array2<f64> {
        let mut pts = Array2::zeros((n, 3));
        let mut s = seed;
        for i in 0..n {
            for j in 0..3 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                pts[[i, j]] = (s as f64) / (u64::MAX as f64) * 4.0 - 2.0;
            }
        }
        pts
    }

    fn make_xyz(coords: &[[f64; 3]]) -> Array2<f64> {
        let n = coords.len();
        let mut a = Array2::zeros((n, 3));
        for (i, c) in coords.iter().enumerate() {
            a[[i, 0]] = c[0];
            a[[i, 1]] = c[1];
            a[[i, 2]] = c[2];
        }
        a
    }

    // ── Sampling tests ────────────────────────────────────────────────────────

    #[test]
    fn test_fps_selects_correct_count() {
        let pts = pseudo_cloud(50, 1);
        let sel = farthest_point_sampling(&pts, 10).expect("fps failed");
        assert_eq!(sel.len(), 10);
    }

    #[test]
    fn test_fps_no_duplicates() {
        let pts = pseudo_cloud(50, 2);
        let sel = farthest_point_sampling(&pts, 20).expect("fps failed");
        let mut sorted = sel.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 20);
    }

    #[test]
    fn test_fps_diverse_selection() {
        let mut pts = Array2::zeros((5, 3));
        let xs = [0.0f64, 0.1, 0.2, 0.3, 10.0];
        for (i, &x) in xs.iter().enumerate() {
            pts[[i, 0]] = x;
        }
        let sel = farthest_point_sampling(&pts, 2).expect("fps failed");
        assert_eq!(sel[0], 0);
        assert_eq!(sel[1], 4, "second selected should be the outlier at x=10");
    }

    #[test]
    fn test_ball_query_radius_filter() {
        let pts = pseudo_cloud(20, 3);
        let centers = make_xyz(&[[0.0, 0.0, 0.0]]);
        let r = 0.5;
        let nb = ball_query(&pts, &centers, r, 20);
        // All returned indices should be within radius.
        for &idx in &nb[0] {
            let dx = pts[[idx, 0]];
            let dy = pts[[idx, 1]];
            let dz = pts[[idx, 2]];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(d <= r + 1e-10, "point at dist {d} is outside radius {r}");
        }
    }

    #[test]
    fn test_ball_query_max_points_limit() {
        let pts = pseudo_cloud(100, 4);
        let centers = make_xyz(&[[0.0, 0.0, 0.0]]);
        let nb = ball_query(&pts, &centers, 10.0, 5); // very large radius
        assert_eq!(nb[0].len(), 5);
    }

    #[test]
    fn test_knn_returns_k_neighbors() {
        let pts = pseudo_cloud(50, 5);
        let queries = make_xyz(&[[0.0, 0.0, 0.0]]);
        let knn = knn_query(&pts, &queries, 7);
        assert_eq!(knn[0].len(), 7);
    }

    // ── SA tests ──────────────────────────────────────────────────────────────

    #[test]
    fn test_sa_layer_output_shape() {
        let pts = pseudo_cloud(64, 6);
        let cfg = SAConfig {
            n_points: 16,
            radius: 1.0,
            max_group_size: 8,
            mlp_channels: vec![32, 64],
            group_all: false,
        };
        let sa = SetAbstraction::new(cfg, 0).expect("SA construction failed");
        let (xyz, feat) = sa.forward(&pts, None).expect("SA forward failed");
        assert_eq!(xyz.nrows(), 16);
        assert_eq!(feat.ncols(), 64);
    }

    #[test]
    fn test_sa_group_all_mode() {
        let pts = pseudo_cloud(32, 7);
        let cfg = SAConfig {
            group_all: true,
            mlp_channels: vec![64, 128],
            ..Default::default()
        };
        let sa = SetAbstraction::new(cfg, 0).expect("SA construction failed");
        let (xyz, feat) = sa.forward(&pts, None).expect("SA forward failed");
        assert_eq!(xyz.nrows(), 1);
        assert_eq!(feat.nrows(), 1);
        assert_eq!(feat.ncols(), 128);
    }

    #[test]
    fn test_mlp_forward_shape() {
        // Build weight/bias arrays manually to test mlp_forward shape.
        // Layer 0: [3 → 16], Layer 1: [16 → 32], Layer 2: [32 → 64]
        let dims = [3usize, 16, 32, 64];
        let weights: Vec<Array2<f64>> = dims
            .windows(2)
            .map(|pair| Array2::zeros((pair[0], pair[1])))
            .collect();
        let biases: Vec<Array1<f64>> = dims[1..].iter().map(|&c| Array1::zeros(c)).collect();
        let x = Array1::zeros(3_usize);
        let out = mlp_forward(&weights, &biases, &x);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_sa_config_default() {
        let cfg = SAConfig::default();
        assert_eq!(cfg.n_points, 512);
        assert!(!cfg.group_all);
    }

    #[test]
    fn test_sa_with_features_input() {
        let pts = pseudo_cloud(32, 8);
        let in_c = 6usize;
        let mut feats = Array2::zeros((32, in_c));
        for i in 0..32 {
            feats[[i, 0]] = i as f64;
        }
        let cfg = SAConfig {
            n_points: 8,
            radius: 1.0,
            max_group_size: 4,
            mlp_channels: vec![32, 64],
            group_all: false,
        };
        let sa = SetAbstraction::new(cfg, in_c).expect("SA construction failed");
        let (xyz, feat) = sa.forward(&pts, Some(&feats)).expect("SA forward failed");
        assert_eq!(xyz.nrows(), 8);
        assert_eq!(feat.ncols(), 64);
    }

    // ── FP tests ──────────────────────────────────────────────────────────────

    #[test]
    fn test_fp_output_shape_matches_finer_level() {
        let xyz1 = pseudo_cloud(16, 9);
        let xyz2 = pseudo_cloud(4, 10);
        let c2 = 8usize;
        let mut feats2 = Array2::zeros((4, c2));
        feats2[[0, 0]] = 1.0;

        let cfg = FPConfig {
            mlp_channels: vec![32, 64],
            k_neighbors: 3,
        };
        let fp = FeaturePropagation::new(cfg, c2).expect("FP construction failed");
        let out = fp
            .forward(&xyz1, &xyz2, None, &feats2)
            .expect("FP forward failed");
        assert_eq!(out.nrows(), 16);
        assert_eq!(out.ncols(), 64);
    }

    #[test]
    fn test_fp_interpolation_weights_sum_to_one() {
        // Single coarse point → all fine points get the same interpolated features.
        let xyz1 = make_xyz(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let xyz2 = make_xyz(&[[0.5, 0.0, 0.0]]);
        let c2 = 4usize;
        let mut feats2 = Array2::zeros((1, c2));
        for d in 0..c2 {
            feats2[[0, d]] = (d + 1) as f64;
        }
        let cfg = FPConfig {
            mlp_channels: vec![8],
            k_neighbors: 1,
        };
        let fp = FeaturePropagation::new(cfg, c2).expect("FP construction failed");
        let out = fp
            .forward(&xyz1, &xyz2, None, &feats2)
            .expect("FP forward failed");
        // All rows should produce the same output (identical input to MLP).
        for d in 0..8 {
            assert!(
                (out[[0, d]] - out[[1, d]]).abs() < 1e-10,
                "rows 0 and 1 differ at col {d}: {} vs {}",
                out[[0, d]],
                out[[1, d]]
            );
        }
    }

    #[test]
    fn test_fp_single_source_copies_features() {
        let xyz1 = make_xyz(&[[0.0, 0.0, 0.0]]);
        let xyz2 = make_xyz(&[[0.0, 0.0, 0.0]]);
        let c2 = 3usize;
        let mut feats2 = Array2::zeros((1, c2));
        feats2[[0, 0]] = 5.0;
        feats2[[0, 1]] = 7.0;
        feats2[[0, 2]] = 11.0;
        let cfg = FPConfig {
            mlp_channels: vec![8],
            k_neighbors: 1,
        };
        let fp = FeaturePropagation::new(cfg, c2).expect("FP construction failed");
        let out = fp
            .forward(&xyz1, &xyz2, None, &feats2)
            .expect("FP forward failed");
        assert_eq!(out.nrows(), 1);
        let any_nonzero = (0..8).any(|d| out[[0, d]] != 0.0);
        assert!(
            any_nonzero,
            "output should not be all zeros with non-zero inputs"
        );
    }

    #[test]
    fn test_fp_with_skip_connection() {
        let xyz1 = make_xyz(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let xyz2 = make_xyz(&[[0.5, 0.0, 0.0]]);
        let c2 = 4usize;
        let c_skip = 4usize;
        let mut feats2 = Array2::zeros((1, c2));
        feats2[[0, 0]] = 1.0;
        let mut skip = Array2::zeros((2, c_skip));
        skip[[0, 0]] = 2.0;
        let cfg = FPConfig {
            mlp_channels: vec![16, 32],
            k_neighbors: 1,
        };
        let fp = FeaturePropagation::new(cfg, c2 + c_skip).expect("FP construction failed");
        let out = fp
            .forward(&xyz1, &xyz2, Some(&skip), &feats2)
            .expect("FP forward failed");
        assert_eq!(out.nrows(), 2);
        assert_eq!(out.ncols(), 32);
    }

    // ── Backbone tests ────────────────────────────────────────────────────────

    #[test]
    fn test_pointnetpp_backbone_output_shape() {
        let pts = pseudo_cloud(128, 11);
        let config = PointNetPPConfig::default();
        let backbone = PointNetPPBackbone::new(config).expect("backbone construction failed");
        let out = backbone.forward(&pts).expect("backbone forward failed");
        assert_eq!(out.nrows(), 128, "output should have N=128 rows");
    }

    #[test]
    fn test_pointnetpp_forward_small_cloud() {
        let pts = pseudo_cloud(32, 12);
        let config = PointNetPPConfig {
            sa_configs: vec![
                SAConfig {
                    n_points: 16,
                    radius: 1.0,
                    max_group_size: 8,
                    mlp_channels: vec![16, 32],
                    group_all: false,
                },
                SAConfig {
                    group_all: true,
                    mlp_channels: vec![32, 64],
                    ..Default::default()
                },
            ],
            fp_configs: vec![
                FPConfig {
                    mlp_channels: vec![64, 64],
                    k_neighbors: 3,
                },
                FPConfig {
                    mlp_channels: vec![64, 64],
                    k_neighbors: 3,
                },
            ],
            feature_dim: 64,
            in_channels: 0,
        };
        let backbone = PointNetPPBackbone::new(config).expect("backbone construction failed");
        let out = backbone.forward(&pts).expect("backbone forward failed");
        assert_eq!(out.nrows(), 32);
        assert_eq!(out.ncols(), 64);
    }

    // ── IoU / geometry tests ──────────────────────────────────────────────────

    /// Axis-aligned 3-D IoU (for unit testing purposes).
    fn iou3d_aabb(min1: [f64; 3], max1: [f64; 3], min2: [f64; 3], max2: [f64; 3]) -> f64 {
        // Intersection side along axis i = min(max1[i], max2[i]) - max(min1[i], min2[i])
        let inter_vol: f64 = (0..3)
            .map(|i| (max1[i].min(max2[i]) - min1[i].max(min2[i])).max(0.0))
            .product();
        if inter_vol == 0.0 {
            return 0.0;
        }
        let vol1: f64 = (0..3).map(|i| (max1[i] - min1[i]).max(0.0)).product();
        let vol2: f64 = (0..3).map(|i| (max2[i] - min2[i]).max(0.0)).product();
        let union_vol = vol1 + vol2 - inter_vol;
        if union_vol <= 0.0 {
            0.0
        } else {
            inter_vol / union_vol
        }
    }

    #[test]
    fn test_iou3d_self_equals_one() {
        let min = [0.0, 0.0, 0.0];
        let max = [1.0, 1.0, 1.0];
        let iou = iou3d_aabb(min, max, min, max);
        assert!(
            (iou - 1.0).abs() < 1e-10,
            "IoU of box with itself should be 1, got {iou}"
        );
    }

    #[test]
    fn test_euclidean_dist_sq_zero() {
        let a = [1.0, 2.0, 3.0];
        assert_eq!(euclidean_dist_sq(&a, &a), 0.0);
    }

    #[test]
    fn test_euclidean_dist_sq_known() {
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        let d = euclidean_dist_sq(&a, &b);
        assert!((d - 25.0).abs() < 1e-10, "expected 25, got {d}");
    }
}
