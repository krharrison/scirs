//! 3D reconstruction from stereo disparity maps.
//!
//! Provides [`PointCloud`] with per-point colour and normal estimation,
//! voxel-grid downsampling, and PLY serialisation.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// PointCloud
// ─────────────────────────────────────────────────────────────────────────────

/// A 3-D point cloud with optional per-point colour and surface normals.
///
/// Points are stored in metric units (usually metres), in camera-space
/// coordinates: +X right, +Y down, +Z forward.
#[derive(Debug, Clone)]
pub struct PointCloud {
    /// `[x, y, z]` positions.
    pub points: Vec<[f32; 3]>,
    /// `[r, g, b]` colours (same length as `points`, or empty).
    pub colors: Option<Vec<[u8; 3]>>,
    /// Unit surface normals (same length as `points`, or empty).
    pub normals: Option<Vec<[f32; 3]>>,
}

impl PointCloud {
    /// Create an empty point cloud.
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            colors: None,
            normals: None,
        }
    }

    /// Back-project a depth map to 3-D using a pinhole camera model.
    ///
    /// Pixels with depth ≤ 0 are silently skipped.
    ///
    /// # Parameters
    /// - `depth`:  row-major `[H × W]` depth values in the same units as `z`.
    /// - `fx/fy`:  focal lengths in pixels.
    /// - `cx/cy`:  principal point in pixels.
    pub fn from_depth_map(
        depth: &[f32],
        width: usize,
        height: usize,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
    ) -> Self {
        let mut points = Vec::with_capacity(width * height / 4);
        for row in 0..height {
            for col in 0..width {
                let d = depth[row * width + col];
                if d > 0.0 {
                    let x = (col as f32 - cx) * d / fx;
                    let y = (row as f32 - cy) * d / fy;
                    points.push([x, y, d]);
                }
            }
        }
        Self {
            points,
            colors: None,
            normals: None,
        }
    }

    /// Attach per-point colours from a `[H × W × 3]` RGB image.
    ///
    /// Only pixels that produced a valid point (depth > 0) are included.
    pub fn with_colors(
        mut self,
        depth: &[f32],
        image: &[u8],
        width: usize,
        height: usize,
        _fx: f32,
        _fy: f32,
        _cx: f32,
        _cy: f32,
    ) -> Self {
        let mut colors = Vec::with_capacity(self.points.len());
        for row in 0..height {
            for col in 0..width {
                let d = depth[row * width + col];
                if d > 0.0 {
                    let idx = (row * width + col) * 3;
                    if idx + 2 < image.len() {
                        colors.push([image[idx], image[idx + 1], image[idx + 2]]);
                    }
                }
            }
        }
        self.colors = Some(colors);
        self
    }

    /// Estimate surface normals via PCA on the `k` nearest neighbours.
    ///
    /// The orientation is heuristically aligned toward the viewer
    /// (i.e. negative Z component is flipped).
    pub fn estimate_normals(&mut self, k: usize) {
        let n = self.points.len();
        if n < 3 {
            self.normals = Some(vec![[0.0_f32, 0.0, 1.0]; n]);
            return;
        }

        let k_actual = k.min(n - 1).max(2);
        let mut normals = vec![[0.0_f32, 0.0, 1.0]; n];

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let pi = self.points[i];

            // Find k nearest neighbours (brute-force O(n²); sufficient for
            // moderate cloud sizes; an octree would improve large datasets).
            let mut dists: Vec<(f32, usize)> = self
                .points
                .iter()
                .enumerate()
                .map(|(j, pj)| {
                    let dx = pi[0] - pj[0];
                    let dy = pi[1] - pj[1];
                    let dz = pi[2] - pj[2];
                    (dx * dx + dy * dy + dz * dz, j)
                })
                .collect();

            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Skip self (index 0, distance 0).
            let neighbours: Vec<[f32; 3]> = dists[1..=k_actual]
                .iter()
                .map(|(_, j)| self.points[*j])
                .collect();

            // Centroid.
            let inv_n = 1.0 / neighbours.len() as f32;
            let gcx = neighbours.iter().map(|p| p[0]).sum::<f32>() * inv_n;
            let gcy = neighbours.iter().map(|p| p[1]).sum::<f32>() * inv_n;
            let gcz = neighbours.iter().map(|p| p[2]).sum::<f32>() * inv_n;

            // 3×3 covariance matrix (upper triangle).
            let mut cov = [[0.0f32; 3]; 3];
            for p in &neighbours {
                let dx = p[0] - gcx;
                let dy = p[1] - gcy;
                let dz = p[2] - gcz;
                cov[0][0] += dx * dx;
                cov[0][1] += dx * dy;
                cov[0][2] += dx * dz;
                cov[1][1] += dy * dy;
                cov[1][2] += dy * dz;
                cov[2][2] += dz * dz;
            }
            cov[1][0] = cov[0][1];
            cov[2][0] = cov[0][2];
            cov[2][1] = cov[1][2];

            // The surface normal is the eigenvector of the *smallest* eigenvalue.
            // Approximate via power-iteration on (λ_max * I - C) to find the
            // smallest eigenvector without external dependencies.
            let normal = smallest_eigenvector_3x3(&cov);

            // Orient toward viewer (negative Z means away from camera in standard
            // camera frame — flip if pointing away).
            let sign = if normal[2] > 0.0 { -1.0_f32 } else { 1.0 };
            normals[i] = [normal[0] * sign, normal[1] * sign, normal[2] * sign];
        }

        self.normals = Some(normals);
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// True if the cloud contains no points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Voxel-grid downsampling.
    ///
    /// Each occupied voxel is replaced by the centroid of its contained points.
    /// Colour and normals are **not** propagated (they become `None`).
    pub fn voxel_downsample(&self, voxel_size: f32) -> Self {
        if voxel_size <= 0.0 || self.is_empty() {
            return self.clone();
        }

        let inv = 1.0 / voxel_size;
        #[allow(clippy::type_complexity)]
        let mut voxels: HashMap<(i32, i32, i32), (f32, f32, f32, u32)> = HashMap::new();

        for p in &self.points {
            let key = (
                (p[0] * inv).floor() as i32,
                (p[1] * inv).floor() as i32,
                (p[2] * inv).floor() as i32,
            );
            let entry = voxels.entry(key).or_insert((0.0, 0.0, 0.0, 0));
            entry.0 += p[0];
            entry.1 += p[1];
            entry.2 += p[2];
            entry.3 += 1;
        }

        let points: Vec<[f32; 3]> = voxels
            .values()
            .map(|(sx, sy, sz, cnt)| {
                let n = *cnt as f32;
                [sx / n, sy / n, sz / n]
            })
            .collect();

        Self {
            points,
            colors: None,
            normals: None,
        }
    }

    /// Radius outlier removal: keep only points that have ≥ `min_neighbors`
    /// neighbours within `radius` distance.
    pub fn remove_outliers(&self, radius: f32, min_neighbors: usize) -> Self {
        let r2 = radius * radius;
        let points: Vec<[f32; 3]> = self
            .points
            .iter()
            .enumerate()
            .filter(|(i, pi)| {
                let count = self
                    .points
                    .iter()
                    .enumerate()
                    .filter(|(j, pj)| {
                        if *j == *i {
                            return false;
                        }
                        let dx = pi[0] - pj[0];
                        let dy = pi[1] - pj[1];
                        let dz = pi[2] - pj[2];
                        dx * dx + dy * dy + dz * dz <= r2
                    })
                    .count();
                count >= min_neighbors
            })
            .map(|(_, p)| *p)
            .collect();

        Self {
            points,
            colors: None,
            normals: None,
        }
    }

    /// Serialise the point cloud to an ASCII PLY string.
    pub fn to_ply_string(&self) -> String {
        let has_color = self.colors.is_some();
        let has_normal = self.normals.is_some();

        let mut out = format!(
            "ply\nformat ascii 1.0\nelement vertex {}\n\
             property float x\nproperty float y\nproperty float z\n",
            self.len()
        );
        if has_normal {
            out += "property float nx\nproperty float ny\nproperty float nz\n";
        }
        if has_color {
            out += "property uchar red\nproperty uchar green\nproperty uchar blue\n";
        }
        out += "end_header\n";

        for i in 0..self.points.len() {
            let p = self.points[i];
            out += &format!("{} {} {}", p[0], p[1], p[2]);
            if has_normal {
                if let Some(ref ns) = self.normals {
                    let n = ns[i];
                    out += &format!(" {} {} {}", n[0], n[1], n[2]);
                }
            }
            if has_color {
                if let Some(ref cs) = self.colors {
                    let c = cs[i];
                    out += &format!(" {} {} {}", c[0], c[1], c[2]);
                }
            }
            out += "\n";
        }
        out
    }

    /// Merge two point clouds.
    pub fn merge(&mut self, other: &PointCloud) {
        self.points.extend_from_slice(&other.points);
        // Drop optional channels if they are incompatible.
        if self.colors.is_none() || other.colors.is_none() {
            self.colors = None;
        } else {
            if let (Some(ref mut cs), Some(ref os)) = (&mut self.colors, &other.colors) {
                cs.extend_from_slice(os);
            }
        }
        self.normals = None;
    }
}

impl Default for PointCloud {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Approximate the eigenvector of the *smallest* eigenvalue of a 3×3 symmetric
/// positive semi-definite matrix via the inverse power method.
fn smallest_eigenvector_3x3(m: &[[f32; 3]; 3]) -> [f32; 3] {
    // Shift: subtract a small multiple of I to avoid singularity during inversion.
    let trace = m[0][0] + m[1][1] + m[2][2];
    let shift = trace * 1e-3 + 1e-12;

    // (M + shift·I)^-1 · v  via Cramer's rule (always well-defined for PSD + shift).
    let a = [
        [m[0][0] + shift, m[0][1], m[0][2]],
        [m[1][0], m[1][1] + shift, m[1][2]],
        [m[2][0], m[2][1], m[2][2] + shift],
    ];

    // Start with a random-like vector.
    let mut v = [1.0_f32, 0.7, 0.5];
    normalise_vec3(&mut v);

    for _ in 0..20 {
        // w = A^{-1} · v  (solve A·w = v).
        let w = solve_3x3(&a, v);
        let len = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
        if len < 1e-12 {
            break;
        }
        v = [w[0] / len, w[1] / len, w[2] / len];
    }
    v
}

fn normalise_vec3(v: &mut [f32; 3]) {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-12 {
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }
}

/// Solve 3×3 linear system A·x = b via Cramer's rule.
fn solve_3x3(a: &[[f32; 3]; 3], b: [f32; 3]) -> [f32; 3] {
    let det = det_3x3(a);
    if det.abs() < 1e-30 {
        return b; // fallback
    }
    let inv_det = 1.0 / det;
    let mut res = [0.0f32; 3];
    for i in 0..3 {
        let mut ai = *a;
        ai[0][i] = b[0];
        ai[1][i] = b[1];
        ai[2][i] = b[2];
        res[i] = det_3x3(&ai) * inv_det;
    }
    res
}

fn det_3x3(m: &[[f32; 3]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_cloud_default_empty() {
        let pc = PointCloud::default();
        assert!(pc.is_empty());
        assert_eq!(pc.len(), 0);
    }

    #[test]
    fn test_from_depth_map_basic() {
        let width = 4;
        let height = 4;
        let mut depth = vec![0.0_f32; width * height];
        depth[width + 1] = 5.0;
        depth[2 * width + 2] = 3.0;

        let pc = PointCloud::from_depth_map(&depth, width, height, 100.0, 100.0, 2.0, 2.0);
        assert_eq!(pc.len(), 2);
        // Point at (col=1, row=1): x = (1-2)*5/100 = -0.05
        assert!(
            (pc.points[0][0] - (-0.05)).abs() < 1e-4,
            "x={}",
            pc.points[0][0]
        );
    }

    #[test]
    fn test_voxel_downsample_reduces_count() {
        let mut pc = PointCloud::new();
        // 8 points in a 2-unit cube — with voxel size 1 they should map to at
        // most 8 voxels, but with voxel size 4 they all collapse to 1.
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    pc.points.push([i as f32, j as f32, k as f32]);
                }
            }
        }
        let down = pc.voxel_downsample(4.0);
        assert_eq!(down.len(), 1, "all 8 points should merge into 1 voxel");

        let down2 = pc.voxel_downsample(0.5);
        assert_eq!(down2.len(), 8, "0.5-voxels should keep 8 separate points");
    }

    #[test]
    fn test_ply_output_format() {
        let mut pc = PointCloud::new();
        pc.points.push([1.0, 2.0, 3.0]);
        pc.colors = Some(vec![[255, 0, 128]]);

        let ply = pc.to_ply_string();
        assert!(ply.starts_with("ply\n"), "PLY header missing");
        assert!(ply.contains("element vertex 1"), "vertex count wrong");
        assert!(
            ply.contains("property uchar red"),
            "colour property missing"
        );
        assert!(ply.contains("end_header"), "end_header missing");
        assert!(ply.contains("1 2 3"), "point coords missing");
        assert!(ply.contains("255 0 128"), "colour missing");
    }

    #[test]
    fn test_ply_output_no_color() {
        let mut pc = PointCloud::new();
        pc.points.push([0.0, 0.0, 0.0]);
        let ply = pc.to_ply_string();
        assert!(
            !ply.contains("uchar red"),
            "should not contain colour property"
        );
    }

    #[test]
    fn test_estimate_normals_does_not_panic() {
        let mut pc = PointCloud::new();
        for i in 0..20 {
            pc.points.push([i as f32, 0.0, 0.0]);
        }
        pc.estimate_normals(5);
        assert!(pc.normals.is_some());
        let ns = pc.normals.as_ref().expect("normals should exist");
        assert_eq!(ns.len(), 20);
    }

    #[test]
    fn test_remove_outliers() {
        let mut pc = PointCloud::new();
        // Dense cluster at origin.
        for _ in 0..20 {
            pc.points.push([0.0, 0.0, 0.0]);
        }
        // Single isolated outlier.
        pc.points.push([1000.0, 1000.0, 1000.0]);

        let filtered = pc.remove_outliers(1.0, 5);
        assert!(filtered.len() < pc.len(), "outlier should be removed");
    }

    #[test]
    fn test_merge() {
        let mut a = PointCloud::new();
        a.points.push([1.0, 0.0, 0.0]);
        let mut b = PointCloud::new();
        b.points.push([2.0, 0.0, 0.0]);
        b.points.push([3.0, 0.0, 0.0]);
        a.merge(&b);
        assert_eq!(a.len(), 3);
    }
}
