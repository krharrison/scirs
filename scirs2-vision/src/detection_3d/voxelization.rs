//! Voxelization and pillar feature extraction for 3D detection.

use std::collections::HashMap;

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{Result, VisionError};

use super::types::VoxelConfig;

// ---------------------------------------------------------------------------
// Voxel
// ---------------------------------------------------------------------------

/// A single voxel (or pillar) containing accumulated points.
#[derive(Debug, Clone)]
pub struct Voxel {
    /// Points inside this voxel. Each entry is \[x, y, z, intensity\].
    pub points: Vec<[f64; 4]>,
    /// Centre of the voxel in world coordinates.
    pub center: [f64; 3],
}

// ---------------------------------------------------------------------------
// VoxelGrid
// ---------------------------------------------------------------------------

/// Sparse voxel grid built from a `VoxelConfig`.
#[derive(Debug, Clone)]
pub struct VoxelGrid {
    config: VoxelConfig,
    /// Number of grid cells along each axis.
    grid_dims: [usize; 3],
}

impl VoxelGrid {
    /// Create a new voxel grid from configuration.
    pub fn new(config: &VoxelConfig) -> Self {
        let gx = ((config.x_range[1] - config.x_range[0]) / config.voxel_size[0]).ceil() as usize;
        let gy = ((config.y_range[1] - config.y_range[0]) / config.voxel_size[1]).ceil() as usize;
        let gz = ((config.z_range[1] - config.z_range[0]) / config.voxel_size[2]).ceil() as usize;
        Self {
            config: config.clone(),
            grid_dims: [gx, gy, gz],
        }
    }

    /// Grid dimensions \[nx, ny, nz\].
    pub fn dims(&self) -> [usize; 3] {
        self.grid_dims
    }
}

// ---------------------------------------------------------------------------
// Free-standing voxelization
// ---------------------------------------------------------------------------

/// Discretise a set of points (each `[x, y, z, intensity]`) into voxels.
///
/// Points outside the configured range are silently dropped. Voxels are capped
/// at `config.max_points_per_voxel` and the total number of returned voxels is
/// capped at `config.max_voxels`.
pub fn voxelize(points: &[[f64; 4]], config: &VoxelConfig) -> Vec<Voxel> {
    let gx = ((config.x_range[1] - config.x_range[0]) / config.voxel_size[0]).ceil() as usize;
    let gy = ((config.y_range[1] - config.y_range[0]) / config.voxel_size[1]).ceil() as usize;

    // Map: (ix, iy, iz) -> list of points.
    let mut map: HashMap<(usize, usize, usize), Vec<[f64; 4]>> = HashMap::new();

    for p in points {
        let x = p[0];
        let y = p[1];
        let z = p[2];
        if x < config.x_range[0]
            || x >= config.x_range[1]
            || y < config.y_range[0]
            || y >= config.y_range[1]
            || z < config.z_range[0]
            || z >= config.z_range[1]
        {
            continue;
        }
        let ix = ((x - config.x_range[0]) / config.voxel_size[0]) as usize;
        let iy = ((y - config.y_range[0]) / config.voxel_size[1]) as usize;
        let iz = ((z - config.z_range[0]) / config.voxel_size[2]) as usize;
        let entry = map.entry((ix, iy, iz)).or_default();
        if entry.len() < config.max_points_per_voxel {
            entry.push(*p);
        }
    }

    // Build voxel vec (sorted for determinism, capped at max_voxels).
    let mut keys: Vec<_> = map.keys().copied().collect();
    keys.sort();
    keys.truncate(config.max_voxels);

    keys.into_iter()
        .filter_map(|k| {
            let pts = map.remove(&k)?;
            let cx = config.x_range[0] + (k.0 as f64 + 0.5) * config.voxel_size[0];
            let cy = config.y_range[0] + (k.1 as f64 + 0.5) * config.voxel_size[1];
            let cz = config.z_range[0] + (k.2 as f64 + 0.5) * config.voxel_size[2];
            Some(Voxel {
                points: pts,
                center: [cx, cy, cz],
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// PillarFeatureNet
// ---------------------------------------------------------------------------

/// Extracts fixed-length feature vectors from point-cloud pillars.
///
/// Each pillar is encoded with the following per-point features:
/// `[x, y, z, intensity, x_offset, y_offset, z_offset, pillar_cx, pillar_cy, pillar_cz]`
///
/// The mean of these features across all points in the pillar is taken, then
/// linearly projected to `n_features` dimensions.
#[derive(Debug, Clone)]
pub struct PillarFeatureNet {
    /// Linear projection weights: (10, n_features).
    weights: Array2<f64>,
    /// Bias: (n_features,).
    bias: Array1<f64>,
}

/// Number of raw per-point features before projection.
const RAW_FEAT_DIM: usize = 10;

impl PillarFeatureNet {
    /// Create a new pillar feature network with Xavier-initialised weights.
    pub fn new(n_features: usize) -> Self {
        // Simple deterministic initialisation (not random – tests stay deterministic).
        let scale = (2.0 / (RAW_FEAT_DIM + n_features) as f64).sqrt();
        let mut weights = Array2::zeros((RAW_FEAT_DIM, n_features));
        for i in 0..RAW_FEAT_DIM {
            for j in 0..n_features {
                // Deterministic pseudo-random via hashing indices.
                let val = ((i * 7 + j * 13 + 3) as f64).sin() * scale;
                weights[[i, j]] = val;
            }
        }
        let bias = Array1::zeros(n_features);
        Self { weights, bias }
    }

    /// Encode a slice of `Voxel`s into pillar feature vectors.
    ///
    /// Returns `(n_pillars, n_features)`.
    pub fn forward(&self, pillars: &[Voxel]) -> Result<Array2<f64>> {
        let n = pillars.len();
        let n_features = self.weights.ncols();
        let mut out = Array2::zeros((n, n_features));

        for (pi, voxel) in pillars.iter().enumerate() {
            if voxel.points.is_empty() {
                continue;
            }
            // Compute mean raw features.
            let mut mean_feat = [0.0f64; RAW_FEAT_DIM];
            let count = voxel.points.len() as f64;
            for pt in &voxel.points {
                mean_feat[0] += pt[0];
                mean_feat[1] += pt[1];
                mean_feat[2] += pt[2];
                mean_feat[3] += pt[3]; // intensity
                mean_feat[4] += pt[0] - voxel.center[0];
                mean_feat[5] += pt[1] - voxel.center[1];
                mean_feat[6] += pt[2] - voxel.center[2];
                mean_feat[7] += voxel.center[0];
                mean_feat[8] += voxel.center[1];
                mean_feat[9] += voxel.center[2];
            }
            for v in &mut mean_feat {
                *v /= count;
            }

            // Linear projection: feat @ weights + bias  → ReLU.
            for j in 0..n_features {
                let mut val = self.bias[j];
                for (k, &mf) in mean_feat.iter().enumerate().take(RAW_FEAT_DIM) {
                    val += mf * self.weights[[k, j]];
                }
                out[[pi, j]] = val.max(0.0); // ReLU
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Scatter to BEV
// ---------------------------------------------------------------------------

/// Scatter pillar feature vectors onto a 2D bird's-eye-view grid.
///
/// The returned array has shape `(grid_h * grid_w, n_features)`.
/// Each pillar's features are placed at its (ix, iy) position.
pub fn scatter_to_bev(
    pillar_features: &Array2<f64>,
    pillars: &[Voxel],
    config: &VoxelConfig,
) -> Result<Array2<f64>> {
    let gx = ((config.x_range[1] - config.x_range[0]) / config.voxel_size[0]).ceil() as usize;
    let gy = ((config.y_range[1] - config.y_range[0]) / config.voxel_size[1]).ceil() as usize;
    let n_features = pillar_features.ncols();

    if pillar_features.nrows() != pillars.len() {
        return Err(VisionError::DimensionMismatch(format!(
            "pillar_features rows ({}) != pillars len ({})",
            pillar_features.nrows(),
            pillars.len()
        )));
    }

    let grid_size = gx * gy;
    let mut bev = Array2::zeros((grid_size, n_features));

    for (pi, voxel) in pillars.iter().enumerate() {
        let ix = ((voxel.center[0] - config.x_range[0]) / config.voxel_size[0]) as usize;
        let iy = ((voxel.center[1] - config.y_range[0]) / config.voxel_size[1]) as usize;
        if ix < gx && iy < gy {
            let idx = iy * gx + ix;
            for j in 0..n_features {
                bev[[idx, j]] = pillar_features[[pi, j]];
            }
        }
    }
    Ok(bev)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_points() -> Vec<[f64; 4]> {
        vec![
            [1.0, 2.0, 0.0, 0.5],
            [1.1, 2.0, 0.1, 0.6],
            [1.0, 2.1, -0.1, 0.4],
            [-60.0, 0.0, 0.0, 0.0], // out of range
        ]
    }

    #[test]
    fn voxelize_basic() {
        let cfg = VoxelConfig::default();
        let voxels = voxelize(&sample_points(), &cfg);
        // The three in-range points should land in one or more voxels.
        let total: usize = voxels.iter().map(|v| v.points.len()).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn voxel_grid_dims() {
        let cfg = VoxelConfig::default();
        let grid = VoxelGrid::new(&cfg);
        assert!(grid.dims()[0] > 0);
        assert!(grid.dims()[1] > 0);
        assert!(grid.dims()[2] > 0);
    }

    #[test]
    fn pillar_feature_net_forward() {
        let cfg = VoxelConfig::default();
        let voxels = voxelize(&sample_points(), &cfg);
        let net = PillarFeatureNet::new(16);
        let feats = net.forward(&voxels).expect("forward failed");
        assert_eq!(feats.nrows(), voxels.len());
        assert_eq!(feats.ncols(), 16);
    }

    #[test]
    fn scatter_roundtrip() {
        let cfg = VoxelConfig::default();
        let voxels = voxelize(&sample_points(), &cfg);
        let net = PillarFeatureNet::new(8);
        let feats = net.forward(&voxels).expect("forward failed");
        let bev = scatter_to_bev(&feats, &voxels, &cfg).expect("scatter failed");
        let gx = ((cfg.x_range[1] - cfg.x_range[0]) / cfg.voxel_size[0]).ceil() as usize;
        let gy = ((cfg.y_range[1] - cfg.y_range[0]) / cfg.voxel_size[1]).ceil() as usize;
        assert_eq!(bev.nrows(), gx * gy);
        assert_eq!(bev.ncols(), 8);
    }
}
