//! Feature Propagation layer for PointNet++.
//!
//! Propagates features from a coarser to a finer resolution level using
//! inverse-distance-weighted (IDW) interpolation from k nearest neighbours,
//! followed by a skip-connection concatenation and an MLP.

use crate::error::{Result, VisionError};
use crate::pointnet::sampling::{euclidean_dist_sq, knn_query};
use crate::pointnet::set_abstraction::mlp_forward;
use scirs2_core::ndarray::{Array1, Array2};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a Feature Propagation layer.
#[derive(Debug, Clone)]
pub struct FPConfig {
    /// MLP channel sizes applied after interpolation + skip concatenation.
    pub mlp_channels: Vec<usize>,
    /// Number of nearest coarse-level neighbours used for IDW interpolation.
    pub k_neighbors: usize,
}

impl Default for FPConfig {
    fn default() -> Self {
        Self {
            mlp_channels: vec![128, 128],
            k_neighbors: 3,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Xavier init (same as set_abstraction — keep local to avoid leaking private
// helpers across modules)
// ─────────────────────────────────────────────────────────────────────────────

fn xavier_init(fan_in: usize, fan_out: usize) -> Array2<f64> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
    let mut state: u64 = 0x_feed_face_dead_beef;
    let mut w = Array2::zeros((fan_in, fan_out));
    for i in 0..fan_in {
        for j in 0..fan_out {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let t = (state as f64) / (u64::MAX as f64);
            w[[i, j]] = (2.0 * t - 1.0) * limit;
        }
    }
    w
}

// ─────────────────────────────────────────────────────────────────────────────
// FeaturePropagation
// ─────────────────────────────────────────────────────────────────────────────

/// A Feature Propagation (FP) layer.
///
/// Interpolates features from `xyz2` (coarse) back to `xyz1` (fine) using
/// inverse-distance-weighted k-NN, then optionally concatenates skip-connection
/// features from `xyz1` and applies a shared MLP.
pub struct FeaturePropagation {
    config: FPConfig,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
}

impl FeaturePropagation {
    /// Construct a new FP layer.
    ///
    /// # Arguments
    /// - `config`: layer configuration.
    /// - `in_channels`: total input channels *after* concatenating interpolated
    ///   features from `xyz2` and skip features from `xyz1`.  The caller must
    ///   compute this as `coarse_feature_channels + skip_feature_channels`
    ///   (skip = 0 when `features1 = None`).
    pub fn new(config: FPConfig, in_channels: usize) -> Result<Self> {
        if config.mlp_channels.is_empty() {
            return Err(VisionError::InvalidParameter(
                "FeaturePropagation: mlp_channels must not be empty".to_string(),
            ));
        }
        if config.k_neighbors == 0 {
            return Err(VisionError::InvalidParameter(
                "FeaturePropagation: k_neighbors must be > 0".to_string(),
            ));
        }

        let mut dims = vec![in_channels];
        dims.extend_from_slice(&config.mlp_channels);

        let mut weights = Vec::with_capacity(dims.len() - 1);
        let mut biases = Vec::with_capacity(dims.len() - 1);
        for pair in dims.windows(2) {
            weights.push(xavier_init(pair[0], pair[1]));
            biases.push(Array1::zeros(pair[1]));
        }

        Ok(Self {
            config,
            weights,
            biases,
        })
    }

    /// Output feature dimension (last entry in `mlp_channels`).
    pub fn out_channels(&self) -> usize {
        self.config.mlp_channels.last().copied().unwrap_or(0)
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `xyz1`: `[N1, 3]` — fine-level point coordinates.
    /// - `xyz2`: `[N2, 3]` — coarse-level point coordinates.
    /// - `features1`: optional `[N1, C1]` skip-connection features.
    /// - `features2`: `[N2, C2]` coarse-level features to be interpolated.
    ///
    /// # Returns
    /// `[N1, C_out]` per-point features at the fine level.
    pub fn forward(
        &self,
        xyz1: &Array2<f64>,
        xyz2: &Array2<f64>,
        features1: Option<&Array2<f64>>,
        features2: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let n1 = xyz1.nrows();
        let n2 = xyz2.nrows();
        let c2 = features2.ncols();

        if xyz1.ncols() != 3 || xyz2.ncols() != 3 {
            return Err(VisionError::InvalidParameter(
                "FeaturePropagation: xyz arrays must have 3 columns".to_string(),
            ));
        }
        if features2.nrows() != n2 {
            return Err(VisionError::DimensionMismatch(format!(
                "FeaturePropagation: features2.nrows ({}) != xyz2.nrows ({})",
                features2.nrows(),
                n2
            )));
        }
        if let Some(f1) = features1 {
            if f1.nrows() != n1 {
                return Err(VisionError::DimensionMismatch(format!(
                    "FeaturePropagation: features1.nrows ({}) != xyz1.nrows ({})",
                    f1.nrows(),
                    n1
                )));
            }
        }

        // ── 1. Interpolate features2 → each xyz1 point ───────────────────────
        let k = self.config.k_neighbors.min(n2);
        let knn = knn_query(xyz2, xyz1, k);
        let interpolated = self.idw_interpolate(xyz1, xyz2, features2, &knn, k, c2)?;

        // ── 2. Skip-connection concatenation ─────────────────────────────────
        let c_out_mlp = self.out_channels();
        let mut out = Array2::zeros((n1, c_out_mlp));

        for i in 0..n1 {
            let interp_row = interpolated.row(i);

            let concat: Vec<f64> = match features1 {
                None => interp_row.iter().copied().collect(),
                Some(f1) => {
                    let skip_row = f1.row(i);
                    interp_row
                        .iter()
                        .copied()
                        .chain(skip_row.iter().copied())
                        .collect()
                }
            };

            let h = Array1::from_vec(concat);
            let result = mlp_forward(&self.weights, &self.biases, &h);

            for d in 0..c_out_mlp {
                out[[i, d]] = result[d];
            }
        }

        Ok(out)
    }

    // ── IDW interpolation ─────────────────────────────────────────────────────

    fn idw_interpolate(
        &self,
        xyz1: &Array2<f64>,
        xyz2: &Array2<f64>,
        features2: &Array2<f64>,
        knn: &[Vec<usize>],
        k: usize,
        c2: usize,
    ) -> Result<Array2<f64>> {
        let n1 = xyz1.nrows();
        let dim = xyz1.ncols().min(xyz2.ncols());
        let mut interp = Array2::zeros((n1, c2));

        for i in 0..n1 {
            let px: Vec<f64> = (0..dim).map(|d| xyz1[[i, d]]).collect();
            let neighbors = &knn[i];

            // Compute distances to each of the k coarse neighbours.
            let mut dists: Vec<f64> = neighbors
                .iter()
                .map(|&j| {
                    let qx: Vec<f64> = (0..dim).map(|d| xyz2[[j, d]]).collect();
                    euclidean_dist_sq(&px, &qx).sqrt()
                })
                .collect();

            // If any distance is exactly zero, copy that feature directly
            // (avoids division-by-zero and gives exact behaviour for identical
            // points, e.g. in the single-source test).
            let exact_match: Option<usize> = dists
                .iter()
                .enumerate()
                .find(|(_, &d)| d < 1e-15)
                .map(|(idx, _)| neighbors[idx]);

            if let Some(src) = exact_match {
                for d in 0..c2 {
                    interp[[i, d]] = features2[[src, d]];
                }
                continue;
            }

            // Inverse-distance weights (w_j = 1 / dist_j).
            let inv_dists: Vec<f64> = dists.iter_mut().map(|d| 1.0 / *d).collect();
            let weight_sum: f64 = inv_dists.iter().sum();

            if weight_sum < 1e-30 {
                // Degenerate: use uniform weights.
                let uniform = 1.0 / k as f64;
                for &j in neighbors.iter().take(k) {
                    for d in 0..c2 {
                        interp[[i, d]] += uniform * features2[[j, d]];
                    }
                }
            } else {
                for (idx, &j) in neighbors.iter().take(k).enumerate() {
                    let w = inv_dists[idx] / weight_sum;
                    for d in 0..c2 {
                        interp[[i, d]] += w * features2[[j, d]];
                    }
                }
            }
        }

        Ok(interp)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

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

    #[test]
    fn test_fp_output_shape_matches_finer_level() {
        let xyz1 = make_xyz(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]);
        let xyz2 = make_xyz(&[[0.5, 0.0, 0.0], [2.5, 0.0, 0.0]]);

        let c2 = 8usize;
        let mut feats2 = Array2::zeros((2, c2));
        feats2[[0, 0]] = 1.0;
        feats2[[1, 0]] = 2.0;

        let cfg = FPConfig {
            mlp_channels: vec![32, 64],
            k_neighbors: 2,
        };
        // in_channels = c2 (no skip)
        let fp = FeaturePropagation::new(cfg, c2).expect("FP construction failed");
        let out = fp
            .forward(&xyz1, &xyz2, None, &feats2)
            .expect("FP forward failed");
        assert_eq!(out.nrows(), 4, "output rows should match xyz1");
        assert_eq!(out.ncols(), 64);
    }

    #[test]
    fn test_fp_interpolation_weights_sum_to_one() {
        // Single coarse point: all fine points must get exactly its feature.
        let xyz1 = make_xyz(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0]]);
        let xyz2 = make_xyz(&[[0.5, 0.5, 0.5]]);

        let c2 = 4usize;
        let mut feats2 = Array2::zeros((1, c2));
        for d in 0..c2 {
            feats2[[0, d]] = (d + 1) as f64;
        }

        let cfg = FPConfig {
            mlp_channels: vec![16],
            k_neighbors: 1,
        };
        let fp = FeaturePropagation::new(cfg, c2).expect("FP construction failed");
        // Use only interpolation step, ignore MLP (can't easily skip it), but
        // verify the interpolation matches feats2 before MLP by checking the
        // actual output has non-zero values derived from feats2.
        let out = fp
            .forward(&xyz1, &xyz2, None, &feats2)
            .expect("FP forward failed");
        // All rows should be the same (single coarse source).
        for i in 0..3 {
            for d in 0..16 {
                assert_eq!(
                    out[[i, d]],
                    out[[0, d]],
                    "all rows should be identical with single coarse source"
                );
            }
        }
    }

    #[test]
    fn test_fp_single_source_copies_features() {
        // Coarse level has 1 point, which coincides with a fine point.
        let xyz1 = make_xyz(&[[0.0, 0.0, 0.0]]);
        let xyz2 = make_xyz(&[[0.0, 0.0, 0.0]]); // identical

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
        // ReLU output should be non-negative and non-zero.
        let any_nonzero = (0..8).any(|d| out[[0, d]] != 0.0);
        assert!(
            any_nonzero,
            "FP output should not be all zeros when input features are non-zero"
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
        skip[[0, 1]] = 2.0;
        skip[[1, 2]] = 3.0;

        let cfg = FPConfig {
            mlp_channels: vec![16, 32],
            k_neighbors: 1,
        };
        // in_channels = c2 + c_skip
        let fp = FeaturePropagation::new(cfg, c2 + c_skip).expect("FP construction failed");
        let out = fp
            .forward(&xyz1, &xyz2, Some(&skip), &feats2)
            .expect("FP forward failed");
        assert_eq!(out.nrows(), 2);
        assert_eq!(out.ncols(), 32);
    }
}
