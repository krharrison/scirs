//! LiDAR point cloud processing for 3D object detection.
//!
//! Provides voxelization (PointVoxel-style), pillar feature extraction
//! (PointPillars-style), and point normal estimation.

use std::collections::HashMap;

use crate::error::{Result, VisionError};

// ---------------------------------------------------------------------------
// VoxelConfig
// ---------------------------------------------------------------------------

/// Configuration for voxel grid construction.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct VoxelConfig {
    /// Voxel dimensions `[voxel_x, voxel_y, voxel_z]` in metres.
    pub voxel_size: [f64; 3],
    /// Spatial extent of the point cloud
    /// `[x_min, y_min, z_min, x_max, y_max, z_max]`.
    pub point_cloud_range: [f64; 6],
    /// Maximum number of points stored per voxel.
    pub max_points_per_voxel: usize,
    /// Maximum number of voxels created.
    pub max_voxels: usize,
}

impl Default for VoxelConfig {
    fn default() -> Self {
        Self {
            voxel_size: [0.1, 0.1, 0.2],
            point_cloud_range: [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
            max_points_per_voxel: 32,
            max_voxels: 20_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Voxel / VoxelizationResult
// ---------------------------------------------------------------------------

/// A single voxel holding a subset of the input point cloud.
#[derive(Debug, Clone)]
pub struct Voxel {
    /// Voxel grid coordinates `(ix, iy, iz)`.
    pub indices: [usize; 3],
    /// XYZ points inside this voxel (up to `max_points_per_voxel`).
    pub points: Vec<[f64; 3]>,
    /// Centroid of all `points`.
    pub mean_point: [f64; 3],
    /// Number of points actually stored in `points`.
    pub n_points: usize,
}

/// Result of the voxelization step.
#[derive(Debug, Clone)]
pub struct VoxelizationResult {
    /// Non-empty voxels.
    pub voxels: Vec<Voxel>,
    /// Total number of non-empty voxels.
    pub n_voxels: usize,
    /// Shape of the voxel grid `(nx, ny, nz)`.
    pub grid_shape: [usize; 3],
}

// ---------------------------------------------------------------------------
// voxelize
// ---------------------------------------------------------------------------

/// Discretize a point cloud `points` (each point `[x, y, z]`) into voxels.
///
/// Points outside `config.point_cloud_range` are silently dropped.
/// Each voxel accumulates at most `config.max_points_per_voxel` points, and
/// the total number of non-empty voxels is capped at `config.max_voxels`.
pub fn voxelize(points: &[[f64; 3]], config: &VoxelConfig) -> Result<VoxelizationResult> {
    let [vx, vy, vz] = config.voxel_size;
    if vx <= 0.0 || vy <= 0.0 || vz <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "voxel_size dimensions must be positive".into(),
        ));
    }

    let [x_min, y_min, z_min, x_max, y_max, z_max] = config.point_cloud_range;

    let nx = ((x_max - x_min) / vx).ceil() as usize;
    let ny = ((y_max - y_min) / vy).ceil() as usize;
    let nz = ((z_max - z_min) / vz).ceil() as usize;

    if nx == 0 || ny == 0 || nz == 0 {
        return Err(VisionError::InvalidParameter(
            "point_cloud_range produces a zero-dimension grid".into(),
        ));
    }

    // Group points by voxel index.
    let mut map: HashMap<[usize; 3], Vec<[f64; 3]>> = HashMap::new();

    for &[x, y, z] in points {
        // Filter out-of-range points.
        if x < x_min || x >= x_max || y < y_min || y >= y_max || z < z_min || z >= z_max {
            continue;
        }
        let ix = ((x - x_min) / vx).floor() as usize;
        let iy = ((y - y_min) / vy).floor() as usize;
        let iz = ((z - z_min) / vz).floor() as usize;

        // Guard against floating-point edge cases.
        if ix >= nx || iy >= ny || iz >= nz {
            continue;
        }

        let entry = map.entry([ix, iy, iz]).or_default();
        if entry.len() < config.max_points_per_voxel {
            entry.push([x, y, z]);
        }
    }

    // Build sorted list capped at max_voxels.
    let mut voxel_keys: Vec<[usize; 3]> = map.keys().copied().collect();
    voxel_keys.sort_unstable(); // deterministic order

    let n_kept = voxel_keys.len().min(config.max_voxels);
    voxel_keys.truncate(n_kept);

    let mut voxels = Vec::with_capacity(n_kept);
    for key in voxel_keys {
        let pts = map.remove(&key).unwrap_or_default();
        let n = pts.len();
        let mean = centroid(&pts);
        voxels.push(Voxel {
            indices: key,
            points: pts,
            mean_point: mean,
            n_points: n,
        });
    }

    let n_voxels = voxels.len();
    Ok(VoxelizationResult {
        voxels,
        n_voxels,
        grid_shape: [nx, ny, nz],
    })
}

// ---------------------------------------------------------------------------
// PillarConfig / PillarFeatures
// ---------------------------------------------------------------------------

/// Configuration for the PointPillars pillar-feature extraction step.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct PillarConfig {
    /// Voxel/pillar grid configuration (Z extent treated as one pillar column).
    pub voxel_cfg: VoxelConfig,
    /// Maximum number of points stored per pillar.
    pub max_points_per_pillar: usize,
    /// Number of features per point.
    ///
    /// The default of **9** uses `(x, y, z, r, xc, yc, zc, xp, yp)`:
    /// - `r`: reflectance (4th input channel)
    /// - `xc, yc, zc`: offset from pillar centroid
    /// - `xp, yp`: offset from pillar centre in XY
    pub n_features: usize,
}

impl Default for PillarConfig {
    fn default() -> Self {
        Self {
            voxel_cfg: VoxelConfig::default(),
            max_points_per_pillar: 32,
            n_features: 9,
        }
    }
}

/// Pillar feature tensor ready for a PillarFeatureNet encoder.
#[derive(Debug, Clone)]
pub struct PillarFeatures {
    /// `[n_pillars][max_pts * n_features]` — row-major feature matrix.
    pub features: Vec<Vec<f64>>,
    /// `(col_idx, row_idx)` = `(ix, iy)` for each pillar (BEV coordinates).
    pub coords: Vec<[usize; 2]>,
}

// ---------------------------------------------------------------------------
// extract_pillar_features
// ---------------------------------------------------------------------------

/// Extract PointPillars-style features from a reflective point cloud.
///
/// `points` — each row is `[x, y, z, reflectance]`.
///
/// Returns a [`PillarFeatures`] struct whose `features` matrix has shape
/// `[n_pillars][max_points_per_pillar * n_features]` (zero-padded).
pub fn extract_pillar_features(
    points: &[[f64; 4]],
    config: &PillarConfig,
) -> Result<PillarFeatures> {
    let vcfg = &config.voxel_cfg;
    let [vx, vy, _vz] = vcfg.voxel_size;

    if vx <= 0.0 || vy <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "voxel_size x/y must be positive".into(),
        ));
    }

    let [x_min, y_min, z_min, x_max, y_max, z_max] = vcfg.point_cloud_range;
    let nx = ((x_max - x_min) / vx).ceil() as usize;
    let ny = ((y_max - y_min) / vy).ceil() as usize;

    if nx == 0 || ny == 0 {
        return Err(VisionError::InvalidParameter(
            "point_cloud_range produces a zero-dimension grid".into(),
        ));
    }

    // Group points by (ix, iy) — pillars collapse the Z dimension.
    let mut pillar_map: HashMap<[usize; 2], Vec<[f64; 4]>> = HashMap::new();

    for &[x, y, z, r] in points {
        if x < x_min || x >= x_max || y < y_min || y >= y_max || z < z_min || z >= z_max {
            continue;
        }
        let ix = ((x - x_min) / vx).floor() as usize;
        let iy = ((y - y_min) / vy).floor() as usize;
        if ix >= nx || iy >= ny {
            continue;
        }
        let entry = pillar_map.entry([ix, iy]).or_default();
        if entry.len() < config.max_points_per_pillar {
            entry.push([x, y, z, r]);
        }
    }

    // Sort keys for determinism and cap.
    let mut keys: Vec<[usize; 2]> = pillar_map.keys().copied().collect();
    keys.sort_unstable();
    let n_kept = keys.len().min(vcfg.max_voxels);
    keys.truncate(n_kept);

    let feat_len = config.max_points_per_pillar * config.n_features;
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(n_kept);
    let mut coords: Vec<[usize; 2]> = Vec::with_capacity(n_kept);

    for key in keys {
        let pts = pillar_map.remove(&key).unwrap_or_default();

        // Compute pillar centroid (mean over actual points).
        let n = pts.len() as f64;
        let (sum_x, sum_y, sum_z) = pts.iter().fold((0.0, 0.0, 0.0), |acc, p| {
            (acc.0 + p[0], acc.1 + p[1], acc.2 + p[2])
        });
        let (cx, cy, cz) = if n > 0.0 {
            (sum_x / n, sum_y / n, sum_z / n)
        } else {
            (0.0, 0.0, 0.0)
        };

        // Pillar XY centre in world coordinates.
        let [ix, iy] = key;
        let px = x_min + (ix as f64 + 0.5) * vx;
        let py = y_min + (iy as f64 + 0.5) * vy;

        let mut row = vec![0.0_f64; feat_len];
        for (i, &[x, y, z, r]) in pts.iter().enumerate() {
            let base = i * config.n_features;
            if base + config.n_features > row.len() {
                break;
            }
            // x, y, z, r
            row[base] = x;
            row[base + 1] = y;
            row[base + 2] = z;
            row[base + 3] = r;
            // offsets from centroid
            row[base + 4] = x - cx;
            row[base + 5] = y - cy;
            row[base + 6] = z - cz;
            // offsets from pillar XY centre (no z offset for pillar)
            row[base + 7] = x - px;
            row[base + 8] = y - py;
        }

        features.push(row);
        coords.push(key);
    }

    Ok(PillarFeatures { features, coords })
}

// ---------------------------------------------------------------------------
// Normal estimation
// ---------------------------------------------------------------------------

/// Estimate point normals using PCA on k-nearest neighbours.
///
/// For each point the k nearest neighbours (including the point itself) are
/// found via brute-force L2 search. A 3×3 covariance matrix is computed from
/// the centred neighbourhood, and the smallest eigenvector (from Jacobi
/// iteration) is returned as the surface normal.
///
/// The returned normals are unit vectors; orientation is not disambiguated.
pub fn estimate_normals(points: &[[f64; 3]], k_neighbors: usize) -> Result<Vec<[f64; 3]>> {
    if k_neighbors < 3 {
        return Err(VisionError::InvalidParameter(
            "k_neighbors must be at least 3 to define a plane".into(),
        ));
    }
    let n = points.len();
    if n < k_neighbors {
        return Err(VisionError::InvalidParameter(format!(
            "Not enough points ({n}) for k_neighbors={k_neighbors}"
        )));
    }

    let mut normals = Vec::with_capacity(n);

    for i in 0..n {
        let pi = points[i];

        // Brute-force k-NN (excluding self — we include self separately).
        let mut dists: Vec<(f64, usize)> = points
            .iter()
            .enumerate()
            .map(|(j, &pj)| {
                let dx = pi[0] - pj[0];
                let dy = pi[1] - pj[1];
                let dz = pi[2] - pj[2];
                (dx * dx + dy * dy + dz * dz, j)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let neighbors: Vec<[f64; 3]> = dists[..k_neighbors]
            .iter()
            .map(|&(_, j)| points[j])
            .collect();

        // Centroid of neighbourhood.
        let mean = centroid(&neighbors);

        // 3×3 covariance matrix (upper triangle only, then symmetrise).
        let mut cov = [[0.0_f64; 3]; 3];
        for &nb in &neighbors {
            let d = [nb[0] - mean[0], nb[1] - mean[1], nb[2] - mean[2]];
            for r in 0..3 {
                for c in 0..3 {
                    cov[r][c] += d[r] * d[c];
                }
            }
        }
        let scale = 1.0 / neighbors.len() as f64;
        for cov_row in &mut cov {
            for cov_val in cov_row.iter_mut() {
                *cov_val *= scale;
            }
        }

        // Jacobi iteration for 3×3 symmetric matrix eigenvectors.
        let (eigenvalues, eigenvectors) = jacobi_eigen_3x3(cov)?;

        // Normal = eigenvector corresponding to smallest eigenvalue.
        let min_idx = eigenvalues
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let n_vec = eigenvectors[min_idx];
        let len = (n_vec[0] * n_vec[0] + n_vec[1] * n_vec[1] + n_vec[2] * n_vec[2]).sqrt();
        let normal = if len > 1e-12 {
            [n_vec[0] / len, n_vec[1] / len, n_vec[2] / len]
        } else {
            [0.0, 0.0, 1.0]
        };

        normals.push(normal);
    }

    Ok(normals)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the centroid of a slice of 3D points.
fn centroid(pts: &[[f64; 3]]) -> [f64; 3] {
    if pts.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let n = pts.len() as f64;
    let (sx, sy, sz) = pts.iter().fold((0.0, 0.0, 0.0), |acc, &p| {
        (acc.0 + p[0], acc.1 + p[1], acc.2 + p[2])
    });
    [sx / n, sy / n, sz / n]
}

/// Jacobi eigenvalue decomposition for a 3×3 symmetric matrix.
///
/// Returns `(eigenvalues, eigenvectors)` where each row of `eigenvectors` is
/// one eigenvector corresponding to the eigenvalue at the same index.
fn jacobi_eigen_3x3(a_in: [[f64; 3]; 3]) -> Result<([f64; 3], [[f64; 3]; 3])> {
    let mut a = a_in;
    // Eigenvector matrix initialised to identity.
    let mut v = [[0.0_f64; 3]; 3];
    for (i, v_row) in v.iter_mut().enumerate() {
        v_row[i] = 1.0;
    }

    const MAX_ITER: usize = 100;
    const EPS: f64 = 1e-12;

    for _ in 0..MAX_ITER {
        // Find largest off-diagonal element.
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for (r, a_row) in a.iter().enumerate() {
            for (c, &a_val) in a_row.iter().enumerate().skip(r + 1) {
                if a_val.abs() > max_val {
                    max_val = a_val.abs();
                    p = r;
                    q = c;
                }
            }
        }

        if max_val < EPS {
            break;
        }

        // Jacobi rotation to zero out a[p][q].
        let theta = if (a[q][q] - a[p][p]).abs() < EPS {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * a[p][q]) / (a[q][q] - a[p][p])).atan()
        };

        let (s, cs) = theta.sin_cos();

        // Apply rotation: A' = R^T A R
        let mut a2 = a;
        a2[p][p] = cs * cs * a[p][p] - 2.0 * s * cs * a[p][q] + s * s * a[q][q];
        a2[q][q] = s * s * a[p][p] + 2.0 * s * cs * a[p][q] + cs * cs * a[q][q];
        a2[p][q] = 0.0;
        a2[q][p] = 0.0;

        let third = 3 - p - q; // index of the third axis (0+1+2=3)
        let r = third;
        a2[p][r] = cs * a[p][r] - s * a[q][r];
        a2[r][p] = a2[p][r];
        a2[q][r] = s * a[p][r] + cs * a[q][r];
        a2[r][q] = a2[q][r];

        a = a2;

        // Update eigenvector matrix.
        for v_row in &mut v {
            let vi_p = v_row[p];
            let vi_q = v_row[q];
            v_row[p] = cs * vi_p - s * vi_q;
            v_row[q] = s * vi_p + cs * vi_q;
        }
    }

    Ok(([a[0][0], a[1][1], a[2][2]], v))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform_cloud(n: usize) -> Vec<[f64; 3]> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                [t * 0.5 - 10.0, t * 0.3 - 5.0, t * 0.1 - 1.0]
            })
            .collect()
    }

    #[test]
    fn test_voxel_config_defaults() {
        let cfg = VoxelConfig::default();
        assert_eq!(cfg.voxel_size, [0.1, 0.1, 0.2]);
        assert_eq!(cfg.max_points_per_voxel, 32);
        assert_eq!(cfg.max_voxels, 20_000);
    }

    #[test]
    fn test_pillar_config_defaults() {
        let cfg = PillarConfig::default();
        assert_eq!(cfg.max_points_per_pillar, 32);
        assert_eq!(cfg.n_features, 9);
    }

    #[test]
    fn test_voxelization_basic() {
        let pts: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [0.05, 0.05, 0.05], // same voxel as previous
            [1.0, 0.0, 0.0],    // different voxel
        ];
        let cfg = VoxelConfig::default();
        let res = voxelize(&pts, &cfg).expect("voxelize should succeed");
        // First two points share a voxel; third is separate.
        assert_eq!(res.n_voxels, 2);
    }

    #[test]
    fn test_voxelization_filters_out_of_range() {
        let pts: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],   // in range
            [100.0, 0.0, 0.0], // out of range
        ];
        let cfg = VoxelConfig::default();
        let res = voxelize(&pts, &cfg).expect("voxelize should succeed");
        assert_eq!(res.n_voxels, 1);
    }

    #[test]
    fn test_voxelization_max_voxels_cap() {
        let n = 500;
        let pts: Vec<[f64; 3]> = (0..n).map(|i| [i as f64 * 0.2 - 10.0, 0.0, 0.0]).collect();
        let cfg = VoxelConfig {
            max_voxels: 10,
            ..Default::default()
        };
        let res = voxelize(&pts, &cfg).expect("voxelize should succeed");
        assert!(res.n_voxels <= 10);
    }

    #[test]
    fn test_voxelization_max_points_per_voxel() {
        // Pack many points into a single voxel.
        let pts: Vec<[f64; 3]> = (0..100).map(|_| [0.0, 0.0, 0.0]).collect();
        let cfg = VoxelConfig {
            max_points_per_voxel: 5,
            ..Default::default()
        };
        let res = voxelize(&pts, &cfg).expect("voxelize should succeed");
        assert_eq!(res.n_voxels, 1);
        assert_eq!(res.voxels[0].n_points, 5);
    }

    #[test]
    fn test_pillar_feature_shape() {
        let pts: Vec<[f64; 4]> = (0..50)
            .map(|i| {
                let t = i as f64 * 0.3 - 5.0;
                [t, t * 0.5, 0.0, 0.8]
            })
            .collect();
        let cfg = PillarConfig::default();
        let pf = extract_pillar_features(&pts, &cfg).expect("pillar extraction should succeed");
        // Each feature vector has length max_points_per_pillar * n_features.
        let expected_len = cfg.max_points_per_pillar * cfg.n_features;
        for row in &pf.features {
            assert_eq!(row.len(), expected_len);
        }
        assert_eq!(pf.features.len(), pf.coords.len());
    }

    #[test]
    fn test_pillar_n_pillars_bounded() {
        // Generate a dense cloud that would exceed max_voxels.
        let pts: Vec<[f64; 4]> = (0..1000)
            .map(|i| {
                let x = (i % 50) as f64 * 0.5 - 12.0;
                let y = (i / 50) as f64 * 0.5 - 5.0;
                [x, y, 0.0, 1.0]
            })
            .collect();
        let cfg = PillarConfig {
            voxel_cfg: VoxelConfig {
                max_voxels: 50,
                ..Default::default()
            },
            ..Default::default()
        };
        let pf = extract_pillar_features(&pts, &cfg).expect("pillar extraction should succeed");
        assert!(pf.features.len() <= 50);
    }

    #[test]
    fn test_normals_unit_length() {
        // Simple flat patch in the XY plane — normals should be (0, 0, ±1).
        let pts: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.2, 0.8, 0.0],
        ];
        let normals = estimate_normals(&pts, 4).expect("normal estimation should succeed");
        assert_eq!(normals.len(), pts.len());
        for n in &normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-9, "normal length = {len}");
        }
    }

    #[test]
    fn test_normals_z_axis() {
        // Flat patch in XY plane — dominant normal component should be along Z.
        let pts: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ];
        let normals = estimate_normals(&pts, 4).expect("normal estimation should succeed");
        for n in &normals {
            assert!(n[2].abs() > 0.9, "expected Z-dominant normal, got {n:?}");
        }
    }

    #[test]
    fn test_voxel_grid_shape() {
        let cfg = VoxelConfig {
            voxel_size: [1.0, 1.0, 1.0],
            point_cloud_range: [0.0, 0.0, 0.0, 10.0, 10.0, 5.0],
            ..Default::default()
        };
        let pts: Vec<[f64; 3]> = vec![[5.0, 5.0, 2.0]];
        let res = voxelize(&pts, &cfg).expect("voxelize should succeed");
        assert_eq!(res.grid_shape, [10, 10, 5]);
    }

    #[test]
    fn test_centroid_mean_point() {
        let pts: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]];
        let cfg = VoxelConfig {
            voxel_size: [5.0, 5.0, 5.0],
            point_cloud_range: [-5.0, -5.0, -5.0, 5.0, 5.0, 5.0],
            ..Default::default()
        };
        let res = voxelize(&pts, &cfg).expect("voxelize should succeed");
        assert_eq!(res.n_voxels, 1);
        let mean = res.voxels[0].mean_point;
        // mean x = 1.0, mean y = 2/3 ≈ 0.666...
        assert!((mean[0] - 1.0).abs() < 1e-9);
        assert!((mean[1] - 2.0 / 3.0).abs() < 1e-9);
    }
}
