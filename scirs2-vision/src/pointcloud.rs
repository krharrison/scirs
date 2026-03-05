//! 3-D Point Cloud Processing.
//!
//! Provides [`PointCloud3D`] — a lightweight, self-contained point cloud
//! container with per-point normals, colours and intensities — together with
//! the following operations:
//!
//! | Operation | Function / Method |
//! |-----------|-------------------|
//! | Statistical outlier removal | [`PointCloud3D::remove_statistical_outliers`] |
//! | Voxel grid downsampling | [`PointCloud3D::voxel_downsample`] |
//! | PCA normal estimation | [`PointCloud3D::estimate_normals`] |
//! | ICP rigid registration | [`PointCloud3D::icp_align`] |
//! | RANSAC plane fitting | [`ransac_plane_fit`] |
//! | PLY import / export | [`PointCloud3D::to_ply_string`], [`PointCloud3D::from_ply_str`] |

use crate::camera::CameraIntrinsics;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// PointCloud3D
// ─────────────────────────────────────────────────────────────────────────────

/// A 3-D point cloud.
///
/// Points are stored as `[f32; 3]` (x, y, z).  Optional per-point attributes
/// (normals, colours, intensities) are stored as parallel `Vec`s.
#[derive(Debug, Clone)]
pub struct PointCloud3D {
    /// `[x, y, z]` positions.
    pub points: Vec<[f32; 3]>,
    /// Unit surface normals — populated by [`Self::estimate_normals`].
    pub normals: Option<Vec<[f32; 3]>>,
    /// Per-point RGB colour in `[0, 255]`.
    pub colors: Option<Vec<[u8; 3]>>,
    /// Per-point scalar intensity.
    pub intensities: Option<Vec<f32>>,
}

impl PointCloud3D {
    // ── Constructors ─────────────────────────────────────────────────────

    /// Create a new cloud from a `Vec` of `[x, y, z]` points.
    pub fn new(points: Vec<[f32; 3]>) -> Self {
        Self {
            points,
            normals: None,
            colors: None,
            intensities: None,
        }
    }

    /// Unproject a depth map into a point cloud using pinhole intrinsics.
    ///
    /// Pixels with depth ≤ 0 are skipped.  The resulting cloud is in the
    /// camera frame.
    ///
    /// # Arguments
    /// * `depth`      – `[row][col]` depth map in metres.
    /// * `intrinsics` – Camera intrinsics for unprojection.
    pub fn from_depth_map(depth: &[Vec<f32>], intrinsics: &CameraIntrinsics) -> Self {
        let rows = depth.len();
        let mut pts = Vec::new();
        for (r, row) in depth.iter().enumerate() {
            let cols = row.len();
            for (c, &d) in row.iter().enumerate() {
                if d <= 0.0 {
                    continue;
                }
                let xn = (c as f64 - intrinsics.cx) / intrinsics.fx;
                let yn = (r as f64 - intrinsics.cy) / intrinsics.fy;
                let z = d as f64;
                pts.push([(xn * z) as f32, (yn * z) as f32, z as f32]);
            }
            let _ = rows; // suppress unused warning
        }
        Self::new(pts)
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Number of points in the cloud.
    #[inline]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` when the cloud contains no points.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Centroid (mean position) of the cloud.
    pub fn centroid(&self) -> [f32; 3] {
        let n = self.points.len();
        if n == 0 {
            return [0.0; 3];
        }
        let mut sum = [0.0f64; 3];
        for p in &self.points {
            sum[0] += p[0] as f64;
            sum[1] += p[1] as f64;
            sum[2] += p[2] as f64;
        }
        let nf = n as f64;
        [
            (sum[0] / nf) as f32,
            (sum[1] / nf) as f32,
            (sum[2] / nf) as f32,
        ]
    }

    /// Axis-aligned bounding box: `(min_xyz, max_xyz)`.
    pub fn bounding_box(&self) -> ([f32; 3], [f32; 3]) {
        if self.points.is_empty() {
            return ([0.0; 3], [0.0; 3]);
        }
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for p in &self.points {
            for k in 0..3 {
                if p[k] < mn[k] {
                    mn[k] = p[k];
                }
                if p[k] > mx[k] {
                    mx[k] = p[k];
                }
            }
        }
        (mn, mx)
    }

    // ── Outlier removal ───────────────────────────────────────────────────

    /// Remove statistical outliers (SOR filter).
    ///
    /// Each point's mean distance to its `k` nearest neighbours is computed.
    /// Points whose mean distance exceeds `mean + std_ratio * std_dev` of the
    /// global distribution are removed.
    ///
    /// `self` is mutated in place.  All optional attribute arrays (normals,
    /// colors, intensities) are filtered to match.
    pub fn remove_statistical_outliers(&mut self, k: usize, std_ratio: f64) {
        let n = self.points.len();
        if n == 0 || k == 0 {
            return;
        }

        // Compute mean k-NN distance for every point
        let mut mean_dists = Vec::with_capacity(n);
        for i in 0..n {
            let pi = self.points[i];
            // Collect distances to all other points, keep k smallest
            let mut dists: Vec<f32> = (0..n)
                .filter(|&j| j != i)
                .map(|j| dist3(pi, self.points[j]))
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let k_actual = k.min(dists.len());
            let mean: f64 =
                dists[..k_actual].iter().map(|&v| v as f64).sum::<f64>() / k_actual as f64;
            mean_dists.push(mean);
        }

        // Global statistics
        let global_mean = mean_dists.iter().sum::<f64>() / n as f64;
        let variance = mean_dists
            .iter()
            .map(|&v| (v - global_mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let global_std = variance.sqrt();
        let threshold = global_mean + std_ratio * global_std;

        // Keep points below threshold
        let keep: Vec<bool> = mean_dists.iter().map(|&d| d <= threshold).collect();
        self.filter_by_mask(&keep);
    }

    // ── Voxel downsampling ────────────────────────────────────────────────

    /// Downsample the cloud by replacing all points inside each voxel cell
    /// with their centroid.
    ///
    /// Returns a new cloud.  Optional attributes are NOT preserved (they would
    /// require averaging, which depends on the attribute semantics).
    pub fn voxel_downsample(&self, voxel_size: f32) -> Self {
        if self.points.is_empty() || voxel_size <= 0.0 {
            return self.clone();
        }

        // Map voxel key → accumulated sum + count
        let mut voxels: HashMap<(i64, i64, i64), ([f64; 3], usize)> = HashMap::new();

        for p in &self.points {
            let key = (
                (p[0] / voxel_size).floor() as i64,
                (p[1] / voxel_size).floor() as i64,
                (p[2] / voxel_size).floor() as i64,
            );
            let entry = voxels.entry(key).or_insert(([0.0; 3], 0));
            entry.0[0] += p[0] as f64;
            entry.0[1] += p[1] as f64;
            entry.0[2] += p[2] as f64;
            entry.1 += 1;
        }

        let pts: Vec<[f32; 3]> = voxels
            .values()
            .map(|(sum, cnt)| {
                let n = *cnt as f64;
                [
                    (sum[0] / n) as f32,
                    (sum[1] / n) as f32,
                    (sum[2] / n) as f32,
                ]
            })
            .collect();

        Self::new(pts)
    }

    // ── Normal estimation ─────────────────────────────────────────────────

    /// Estimate per-point surface normals by PCA on the `k`-nearest neighbour
    /// neighbourhood.
    ///
    /// The smallest eigenvector of the 3×3 covariance matrix of the
    /// neighbourhood is taken as the normal.  Orientation is made consistent
    /// by flipping to point "upward" (positive Z component).
    ///
    /// `self.normals` is populated (or replaced) in place.
    pub fn estimate_normals(&mut self, k: usize) {
        let n = self.points.len();
        if n == 0 {
            return;
        }

        let mut normals = Vec::with_capacity(n);

        for i in 0..n {
            let pi = self.points[i];
            let neighbours = self.k_nearest_neighbors(pi, k.min(n - 1).max(1));

            if neighbours.is_empty() {
                normals.push([0.0f32, 0.0, 1.0]);
                continue;
            }

            // Neighbourhood centroid
            let mut cx = 0.0f64;
            let mut cy = 0.0f64;
            let mut cz = 0.0f64;
            for &idx in &neighbours {
                let q = self.points[idx];
                cx += q[0] as f64;
                cy += q[1] as f64;
                cz += q[2] as f64;
            }
            let m = neighbours.len() as f64;
            cx /= m;
            cy /= m;
            cz /= m;

            // 3×3 covariance
            let mut cov = [[0.0f64; 3]; 3];
            for &idx in &neighbours {
                let q = self.points[idx];
                let dx = q[0] as f64 - cx;
                let dy = q[1] as f64 - cy;
                let dz = q[2] as f64 - cz;
                let diff = [dx, dy, dz];
                for a in 0..3 {
                    for b in 0..3 {
                        cov[a][b] += diff[a] * diff[b];
                    }
                }
            }

            let normal = smallest_eigenvector_3x3(&cov);
            // Ensure normal points toward positive Z (viewpoint consistency)
            let nf = if normal[2] < 0.0 {
                [-normal[0] as f32, -normal[1] as f32, -normal[2] as f32]
            } else {
                [normal[0] as f32, normal[1] as f32, normal[2] as f32]
            };
            normals.push(nf);
        }

        self.normals = Some(normals);
    }

    // ── ICP registration ──────────────────────────────────────────────────

    /// Align `source` to `target` using point-to-point ICP.
    ///
    /// Returns the aligned source cloud and the 4×4 cumulative transform.
    ///
    /// The algorithm:
    /// 1. For each source point find the nearest target point.
    /// 2. Compute the optimal rigid transform via SVD of the cross-covariance.
    /// 3. Apply the transform and update the cumulative matrix.
    /// 4. Stop when the mean correspondence distance change is < `tolerance`
    ///    or `max_iterations` is reached.
    pub fn icp_align(
        source: &PointCloud3D,
        target: &PointCloud3D,
        max_iterations: usize,
        tolerance: f64,
    ) -> (PointCloud3D, [[f64; 4]; 4]) {
        if source.is_empty() || target.is_empty() {
            return (source.clone(), identity4());
        }

        let mut current_pts = source.points.clone();
        let mut cumulative = identity4();

        let mut prev_mean_dist = f64::INFINITY;

        for _iter in 0..max_iterations {
            // Step 1: correspondences (nearest-target for each source point)
            let mut src_corr = Vec::with_capacity(current_pts.len());
            let mut tgt_corr = Vec::with_capacity(current_pts.len());
            let mut total_dist = 0.0f64;

            for &sp in &current_pts {
                let (nn_idx, dist) = nearest_in_slice(&target.points, sp);
                src_corr.push(sp);
                tgt_corr.push(target.points[nn_idx]);
                total_dist += dist as f64;
            }

            let mean_dist = total_dist / current_pts.len() as f64;

            // Step 2: optimal rigid transform via SVD of cross-covariance
            let (r, t) = optimal_rigid_transform(&src_corr, &tgt_corr);

            // Step 3: apply transform
            for p in current_pts.iter_mut() {
                *p = apply_rigid(r, t, *p);
            }

            // Accumulate
            cumulative = compose_mat4(mat4_from_rt(r, t), cumulative);

            if (prev_mean_dist - mean_dist).abs() < tolerance {
                break;
            }
            prev_mean_dist = mean_dist;
        }

        let aligned = PointCloud3D::new(current_pts);
        (aligned, cumulative)
    }

    // ── k-NN ─────────────────────────────────────────────────────────────

    /// Return the indices of the `k` nearest points (excluding self).
    pub fn k_nearest_neighbors(&self, query: [f32; 3], k: usize) -> Vec<usize> {
        k_nearest_in_slice(&self.points, query, k)
    }

    // ── PLY I/O ───────────────────────────────────────────────────────────

    /// Serialise the cloud to an ASCII PLY string.
    pub fn to_ply_string(&self) -> String {
        let n = self.points.len();
        let has_normals = self.normals.as_ref().map(|v| v.len() == n).unwrap_or(false);
        let has_colors = self.colors.as_ref().map(|v| v.len() == n).unwrap_or(false);

        let mut s = String::with_capacity(512 + n * 64);
        s.push_str("ply\nformat ascii 1.0\n");
        s.push_str(&format!("element vertex {}\n", n));
        s.push_str("property float x\nproperty float y\nproperty float z\n");
        if has_normals {
            s.push_str("property float nx\nproperty float ny\nproperty float nz\n");
        }
        if has_colors {
            s.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\n");
        }
        s.push_str("end_header\n");

        for i in 0..n {
            let p = self.points[i];
            s.push_str(&format!("{} {} {}", p[0], p[1], p[2]));
            if has_normals {
                let nm = self
                    .normals
                    .as_ref()
                    .expect("normals present - guarded by has_normals check")[i];
                s.push_str(&format!(" {} {} {}", nm[0], nm[1], nm[2]));
            }
            if has_colors {
                let c = self
                    .colors
                    .as_ref()
                    .expect("colors present - guarded by has_colors check")[i];
                s.push_str(&format!(" {} {} {}", c[0], c[1], c[2]));
            }
            s.push('\n');
        }
        s
    }

    /// Parse an ASCII PLY string into a `PointCloud3D`.
    ///
    /// Supports `property float x/y/z`, `property float nx/ny/nz`, and
    /// `property uchar red/green/blue`.
    pub fn from_ply_str(s: &str) -> Result<Self, String> {
        let mut lines = s.lines();

        // Parse header
        let first = lines.next().ok_or("Empty PLY")?;
        if first.trim() != "ply" {
            return Err(format!("Not a PLY file (got '{}')", first));
        }

        let mut n_vertices: Option<usize> = None;
        let mut has_nx = false;
        let mut has_red = false;

        loop {
            let line = lines.next().ok_or("Truncated PLY header")?;
            let line = line.trim();
            if line == "end_header" {
                break;
            }
            if let Some(rest) = line.strip_prefix("element vertex ") {
                n_vertices = Some(
                    rest.trim()
                        .parse::<usize>()
                        .map_err(|e| format!("Bad vertex count: {}", e))?,
                );
            } else if line == "property float nx" {
                has_nx = true;
            } else if line == "property uchar red" {
                has_red = true;
            }
        }

        let n_vertices = n_vertices.ok_or("Missing 'element vertex' in PLY header")?;
        let mut points = Vec::with_capacity(n_vertices);
        let mut normals_v = if has_nx {
            Some(Vec::with_capacity(n_vertices))
        } else {
            None
        };
        let mut colors_v = if has_red {
            Some(Vec::with_capacity(n_vertices))
        } else {
            None
        };

        for line in lines.take(n_vertices) {
            let mut toks = line.split_whitespace();
            let x: f32 = toks
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| format!("Bad x in '{}'", line))?;
            let y: f32 = toks
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| format!("Bad y in '{}'", line))?;
            let z: f32 = toks
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| format!("Bad z in '{}'", line))?;
            points.push([x, y, z]);

            if let Some(ref mut nv) = normals_v {
                let nx: f32 = toks
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| format!("Bad nx in '{}'", line))?;
                let ny: f32 = toks
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| format!("Bad ny in '{}'", line))?;
                let nz: f32 = toks
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| format!("Bad nz in '{}'", line))?;
                nv.push([nx, ny, nz]);
            }

            if let Some(ref mut cv) = colors_v {
                let r: u8 = toks
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| format!("Bad red in '{}'", line))?;
                let g: u8 = toks
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| format!("Bad green in '{}'", line))?;
                let b: u8 = toks
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| format!("Bad blue in '{}'", line))?;
                cv.push([r, g, b]);
            }
        }

        Ok(Self {
            normals: normals_v,
            colors: colors_v,
            intensities: None,
            points,
        })
    }

    // ── Private helpers ───────────────────────────────────────────────────

    fn filter_by_mask(&mut self, keep: &[bool]) {
        let pts: Vec<[f32; 3]> = self
            .points
            .iter()
            .zip(keep.iter())
            .filter(|(_, &k)| k)
            .map(|(&p, _)| p)
            .collect();

        if let Some(ref mut norms) = self.normals {
            *norms = norms
                .iter()
                .zip(keep.iter())
                .filter(|(_, &k)| k)
                .map(|(&n, _)| n)
                .collect();
        }
        if let Some(ref mut cols) = self.colors {
            *cols = cols
                .iter()
                .zip(keep.iter())
                .filter(|(_, &k)| k)
                .map(|(&c, _)| c)
                .collect();
        }
        if let Some(ref mut ints) = self.intensities {
            *ints = ints
                .iter()
                .zip(keep.iter())
                .filter(|(_, &k)| k)
                .map(|(&iv, _)| iv)
                .collect();
        }
        self.points = pts;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RANSAC plane fitting
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a plane to a point cloud using RANSAC.
///
/// # Returns
/// `Some(([a, b, c, d], inlier_indices))` where `a·x + b·y + c·z + d = 0`
/// and the normal `[a, b, c]` has unit length.  Returns `None` when the cloud
/// has fewer than 3 points or RANSAC fails to find a valid hypothesis.
///
/// # Arguments
/// * `cloud`              – Input point cloud.
/// * `distance_threshold` – Points within this distance are counted as inliers.
/// * `max_iterations`     – Number of RANSAC trials.
///
/// # Example
/// ```
/// use scirs2_vision::pointcloud::{PointCloud3D, ransac_plane_fit};
///
/// let pts: Vec<[f32; 3]> = (0..20).flat_map(|i| {
///     (0..20).map(move |j| [i as f32, j as f32, 0.0f32])
/// }).collect();
/// let cloud = PointCloud3D::new(pts);
/// let result = ransac_plane_fit(&cloud, 0.01, 100);
/// assert!(result.is_some());
/// let (plane, _inliers) = result.unwrap();
/// // Plane should be z=0, i.e. normal ≈ (0,0,1), d ≈ 0
/// assert!(plane[2].abs() > 0.9, "plane={:?}", plane);
/// ```
pub fn ransac_plane_fit(
    cloud: &PointCloud3D,
    distance_threshold: f32,
    max_iterations: usize,
) -> Option<([f32; 4], Vec<usize>)> {
    let n = cloud.points.len();
    if n < 3 {
        return None;
    }

    // Simple LCG RNG (no external deps)
    let mut rng_state = 12345u64;
    let mut rng_next = |max: usize| -> usize {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng_state >> 33) as usize) % max
    };

    let mut best_inliers: Vec<usize> = Vec::new();
    let mut best_plane = [0.0f32; 4];

    for _ in 0..max_iterations {
        // Sample 3 distinct points
        let i0 = rng_next(n);
        let mut i1 = rng_next(n);
        while i1 == i0 {
            i1 = rng_next(n);
        }
        let mut i2 = rng_next(n);
        while i2 == i0 || i2 == i1 {
            i2 = rng_next(n);
        }

        let p0 = cloud.points[i0];
        let p1 = cloud.points[i1];
        let p2 = cloud.points[i2];

        // Plane normal = (p1-p0) × (p2-p0)
        let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let nx = v1[1] * v2[2] - v1[2] * v2[1];
        let ny = v1[2] * v2[0] - v1[0] * v2[2];
        let nz = v1[0] * v2[1] - v1[1] * v2[0];
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len < 1e-6 {
            continue; // Degenerate
        }
        let (nx, ny, nz) = (nx / len, ny / len, nz / len);
        let d = -(nx * p0[0] + ny * p0[1] + nz * p0[2]);

        // Count inliers
        let inliers: Vec<usize> = (0..n)
            .filter(|&i| {
                let p = cloud.points[i];
                (nx * p[0] + ny * p[1] + nz * p[2] + d).abs() <= distance_threshold
            })
            .collect();

        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_plane = [nx, ny, nz, d];
        }
    }

    if best_inliers.is_empty() {
        None
    } else {
        Some((best_plane, best_inliers))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private math helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Return the index and distance of the nearest point in a slice.
fn nearest_in_slice(pts: &[[f32; 3]], query: [f32; 3]) -> (usize, f32) {
    let mut best_idx = 0;
    let mut best_dist = f32::INFINITY;
    for (i, &p) in pts.iter().enumerate() {
        let d = dist3(p, query);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    (best_idx, best_dist)
}

/// Return indices of k nearest neighbours in a slice (excluding exact query match).
fn k_nearest_in_slice(pts: &[[f32; 3]], query: [f32; 3], k: usize) -> Vec<usize> {
    if pts.is_empty() || k == 0 {
        return Vec::new();
    }
    let mut indexed: Vec<(usize, f32)> = pts
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, dist3(p, query)))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    // Skip distance-0 (the point itself)
    indexed
        .iter()
        .filter(|&&(_, d)| d > 0.0)
        .take(k)
        .map(|&(i, _)| i)
        .collect()
}

/// Identity 4×4 matrix.
fn identity4() -> [[f64; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn mat4_from_rt(r: [[f64; 3]; 3], t: [f64; 3]) -> [[f64; 4]; 4] {
    [
        [r[0][0], r[0][1], r[0][2], t[0]],
        [r[1][0], r[1][1], r[1][2], t[1]],
        [r[2][0], r[2][1], r[2][2], t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn compose_mat4(a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut c = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn apply_rigid(r: [[f64; 3]; 3], t: [f64; 3], p: [f32; 3]) -> [f32; 3] {
    let x = p[0] as f64;
    let y = p[1] as f64;
    let z = p[2] as f64;
    [
        (r[0][0] * x + r[0][1] * y + r[0][2] * z + t[0]) as f32,
        (r[1][0] * x + r[1][1] * y + r[1][2] * z + t[1]) as f32,
        (r[2][0] * x + r[2][1] * y + r[2][2] * z + t[2]) as f32,
    ]
}

/// Compute optimal rigid transform (R, t) minimising sum of squared distances
/// between corresponding point sets using SVD via Jacobi iterations.
fn optimal_rigid_transform(src: &[[f32; 3]], tgt: &[[f32; 3]]) -> ([[f64; 3]; 3], [f64; 3]) {
    let n = src.len();
    if n == 0 {
        return (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
        );
    }

    // Centroids
    let mut cs = [0.0f64; 3];
    let mut ct = [0.0f64; 3];
    for i in 0..n {
        for k in 0..3 {
            cs[k] += src[i][k] as f64;
        }
        for k in 0..3 {
            ct[k] += tgt[i][k] as f64;
        }
    }
    let nf = n as f64;
    for k in 0..3 {
        cs[k] /= nf;
        ct[k] /= nf;
    }

    // Cross-covariance H = sum( (src_i - cs)^T * (tgt_i - ct) )
    let mut h = [[0.0f64; 3]; 3];
    for i in 0..n {
        let ds = [
            src[i][0] as f64 - cs[0],
            src[i][1] as f64 - cs[1],
            src[i][2] as f64 - cs[2],
        ];
        let dt = [
            tgt[i][0] as f64 - ct[0],
            tgt[i][1] as f64 - ct[1],
            tgt[i][2] as f64 - ct[2],
        ];
        for a in 0..3 {
            for b in 0..3 {
                h[a][b] += ds[a] * dt[b];
            }
        }
    }

    // SVD of H via Jacobi (H = U * S * V^T → R = V * U^T)
    let (u, _s, v) = svd3x3_jacobi(h);
    let r = mat3_mul(v, mat3_transpose(u));

    // Ensure proper rotation (det = +1)
    let r = ensure_rotation(r);

    // t = ct - R * cs
    let rcs = mat3_vec(r, cs);
    let t = [ct[0] - rcs[0], ct[1] - rcs[1], ct[2] - rcs[2]];

    (r, t)
}

/// Estimate the smallest eigenvector of a symmetric 3×3 matrix using the
/// power iteration on the *inverse* (shifted) matrix (Jacobi eigendecomposition).
fn smallest_eigenvector_3x3(cov: &[[f64; 3]; 3]) -> [f64; 3] {
    // Use Jacobi eigen decomposition
    let (_, vecs) = jacobi_eigen3(cov);
    // Find column with smallest eigenvalue
    // jacobi_eigen3 returns (eigenvalues, eigenvectors-as-columns)
    let (evals, evecs) = jacobi_eigen3(cov);
    let mut min_idx = 0;
    let mut min_val = evals[0];
    for (i, &ev) in evals.iter().enumerate().skip(1) {
        if ev < min_val {
            min_val = ev;
            min_idx = i;
        }
    }
    let _ = vecs;
    [evecs[0][min_idx], evecs[1][min_idx], evecs[2][min_idx]]
}

/// Jacobi eigendecomposition for a symmetric 3×3 matrix.
/// Returns (eigenvalues, eigenvectors as columns: evecs[row][col]).
fn jacobi_eigen3(a: &[[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut m = *a;
    let mut v = [[1.0f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; // eigenvectors

    for _ in 0..50 {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0usize;
        let mut q = 1usize;
        #[allow(clippy::needless_range_loop)]
        for i in 0..3 {
            for j in (i + 1)..3 {
                if m[i][j].abs() > max_val {
                    max_val = m[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-12 {
            break;
        }

        // Compute Jacobi rotation
        let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            1.0 / (theta - (1.0 + theta * theta).sqrt())
        };
        let cos = 1.0 / (1.0 + t * t).sqrt();
        let sin = t * cos;
        let tau = sin / (1.0 + cos);

        // Update m
        let mpq = m[p][q];
        m[p][p] -= t * mpq;
        m[q][q] += t * mpq;
        m[p][q] = 0.0;
        m[q][p] = 0.0;

        #[allow(clippy::needless_range_loop)]
        for r in 0..3 {
            if r != p && r != q {
                let mrp = m[r][p];
                let mrq = m[r][q];
                m[r][p] = mrp - sin * (mrq + tau * mrp);
                m[p][r] = m[r][p];
                m[r][q] = mrq + sin * (mrp - tau * mrq);
                m[q][r] = m[r][q];
            }
        }

        // Update eigenvectors
        #[allow(clippy::needless_range_loop)]
        for r in 0..3 {
            let vp = v[r][p];
            let vq = v[r][q];
            v[r][p] = vp - sin * (vq + tau * vp);
            v[r][q] = vq + sin * (vp - tau * vq);
        }
    }

    ([m[0][0], m[1][1], m[2][2]], v)
}

/// Jacobi SVD for a general (not necessarily symmetric) 3×3 matrix.
/// Returns (U, S_diag, V) such that A = U * diag(S) * V^T.
fn svd3x3_jacobi(a: [[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3], [[f64; 3]; 3]) {
    // Compute A^T * A (symmetric) and decompose
    let mut ata = [[0.0f64; 3]; 3];
    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                ata[i][j] += a[k][i] * a[k][j];
            }
        }
    }
    let (evals, v) = jacobi_eigen3(&ata);

    // S = sqrt(evals)  (ensure non-negative)
    let s = [
        evals[0].max(0.0).sqrt(),
        evals[1].max(0.0).sqrt(),
        evals[2].max(0.0).sqrt(),
    ];

    // U_i = A * V_i / sigma_i
    let mut u = [[0.0f64; 3]; 3];
    for j in 0..3 {
        if s[j] > 1e-10 {
            for i in 0..3 {
                u[i][j] = (a[i][0] * v[0][j] + a[i][1] * v[1][j] + a[i][2] * v[2][j]) / s[j];
            }
        } else {
            // Handle zero singular value: pick arbitrary orthogonal vector
            if j == 0 {
                u[0][j] = 1.0;
            } else if j == 1 {
                u[1][j] = 1.0;
            } else {
                u[2][j] = 1.0;
            }
        }
    }

    (u, s, v)
}

fn mat3_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn mat3_transpose(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn mat3_vec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn ensure_rotation(r: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // det(R): if < 0, flip last column
    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    if det < 0.0 {
        let mut r2 = r;
        r2[0][2] = -r2[0][2];
        r2[1][2] = -r2[1][2];
        r2[2][2] = -r2[2][2];
        r2
    } else {
        r
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn plane_cloud() -> PointCloud3D {
        // 10×10 grid on z=0 plane
        let pts: Vec<[f32; 3]> = (0..10)
            .flat_map(|i| (0..10).map(move |j| [i as f32, j as f32, 0.0f32]))
            .collect();
        PointCloud3D::new(pts)
    }

    #[test]
    fn test_centroid_simple() {
        let cloud = PointCloud3D::new(vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let c = cloud.centroid();
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1]).abs() < 1e-6);
        assert!((c[2]).abs() < 1e-6);
    }

    #[test]
    fn test_bounding_box() {
        let cloud = PointCloud3D::new(vec![[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
        let (mn, mx) = cloud.bounding_box();
        assert!((mn[0] - (-1.0)).abs() < 1e-6);
        assert!((mx[0] - 1.0).abs() < 1e-6);
        assert!((mn[2] - (-3.0)).abs() < 1e-6);
        assert!((mx[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_voxel_downsample_reduces_count() {
        let cloud = plane_cloud();
        let orig_n = cloud.len();
        let down = cloud.voxel_downsample(2.0);
        assert!(
            down.len() < orig_n,
            "downsampled: {}, orig: {}",
            down.len(),
            orig_n
        );
    }

    #[test]
    fn test_voxel_downsample_empty() {
        let cloud = PointCloud3D::new(Vec::new());
        let down = cloud.voxel_downsample(1.0);
        assert!(down.is_empty());
    }

    #[test]
    fn test_sor_removes_outliers() {
        let mut cloud = plane_cloud();
        // Add an obvious outlier far from the plane
        cloud.points.push([100.0, 100.0, 100.0]);
        let n_before = cloud.len();
        cloud.remove_statistical_outliers(5, 1.0);
        let n_after = cloud.len();
        assert!(n_after < n_before, "before={}, after={}", n_before, n_after);
    }

    #[test]
    fn test_estimate_normals() {
        let mut cloud = plane_cloud();
        cloud.estimate_normals(5);
        let normals = cloud
            .normals
            .as_ref()
            .expect("normals should be populated after estimate_normals");
        assert_eq!(normals.len(), cloud.len());
        // Z component should be dominant (plane at z=0 → normal ≈ (0,0,1))
        for n in normals {
            assert!(n[2].abs() > 0.5, "normal z component too small: {:?}", n);
        }
    }

    #[test]
    fn test_ransac_plane_fit_z0() {
        let cloud = plane_cloud();
        let result = ransac_plane_fit(&cloud, 0.01, 200);
        assert!(result.is_some(), "RANSAC found no plane");
        let (plane, inliers) = result.expect("RANSAC plane fit should find a plane");
        // Normal z component should be dominant
        assert!(plane[2].abs() > 0.9, "plane={:?}", plane);
        // Most points should be inliers
        assert!(inliers.len() > cloud.len() / 2);
    }

    #[test]
    fn test_ply_roundtrip() {
        let pts = vec![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let cloud = PointCloud3D::new(pts);
        let ply = cloud.to_ply_string();
        let loaded =
            PointCloud3D::from_ply_str(&ply).expect("from_ply_str should succeed on valid PLY");
        assert_eq!(loaded.len(), 2);
        assert!((loaded.points[0][0] - 1.0).abs() < 1e-5);
        assert!((loaded.points[1][2] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_ply_with_normals_roundtrip() {
        let pts = vec![[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let norms = vec![[0.0f32, 0.0, 1.0], [0.0, 0.0, 1.0]];
        let cloud = PointCloud3D {
            points: pts,
            normals: Some(norms),
            colors: None,
            intensities: None,
        };
        let ply = cloud.to_ply_string();
        let loaded = PointCloud3D::from_ply_str(&ply)
            .expect("from_ply_str should succeed on PLY with normals");
        assert!(loaded.normals.is_some());
        let nv = loaded
            .normals
            .expect("normals should be present after round-trip with normals");
        assert!((nv[0][2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_from_depth_map() {
        use crate::camera::CameraIntrinsics;
        let intrinsics = CameraIntrinsics::ideal(100.0, 100.0, 4.0, 4.0);
        let depth = vec![
            vec![0.0f32, 0.0, 1.0, 0.0, 0.0], // only col 2 has depth
            vec![0.0f32; 5],
        ];
        let cloud = PointCloud3D::from_depth_map(&depth, &intrinsics);
        assert_eq!(cloud.len(), 1);
        assert!((cloud.points[0][2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_icp_identity() {
        // Source = target → transform should be identity
        let cloud = plane_cloud();
        let (aligned, _tf) = PointCloud3D::icp_align(&cloud, &cloud, 5, 1e-6);
        // Aligned should be approximately the same as original
        for (a, b) in aligned.points.iter().zip(cloud.points.iter()) {
            let d = dist3(*a, *b);
            assert!(d < 0.5, "dist={}", d);
        }
    }

    #[test]
    fn test_k_nearest_neighbors() {
        let cloud = PointCloud3D::new(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]);
        let nn = cloud.k_nearest_neighbors([0.5, 0.0, 0.0], 2);
        assert_eq!(nn.len(), 2);
        // Nearest should be index 0 and 1 (distance 0.5 each from query 0.5)
        assert!(nn.contains(&0) || nn.contains(&1));
    }
}
