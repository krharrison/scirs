//! 3D Point Cloud Processing
//!
//! Provides data structures and algorithms for working with 3D point clouds,
//! including nearest-neighbor queries via a K-D tree, voxel downsampling,
//! outlier removal, normal estimation, and registration algorithms (ICP and
//! RANSAC).

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::Array2;
use scirs2_linalg::svd;

// ─────────────────────────────────────────────────────────────────────────────
// PointCloud
// ─────────────────────────────────────────────────────────────────────────────

/// 3D point cloud: n×3 matrix of (x, y, z) points.
///
/// Optionally carries per-point RGB colours (in \[0,1\]) and unit surface
/// normals.
#[derive(Debug, Clone)]
pub struct PointCloud {
    /// Shape (n, 3) – each row is `[x, y, z]`.
    pub points: Array2<f64>,
    /// Shape (n, 3) – RGB values in \[0, 1\].  `None` when unavailable.
    pub colors: Option<Array2<f64>>,
    /// Shape (n, 3) – unit surface normals.  `None` until
    /// [`PointCloud::estimate_normals`] is called.
    pub normals: Option<Array2<f64>>,
}

impl PointCloud {
    /// Create a new point cloud from an existing (n, 3) array.
    ///
    /// Returns `Err` when the second dimension is not 3.
    pub fn new(points: Array2<f64>) -> Result<Self> {
        let shape = points.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(VisionError::InvalidParameter(
                "PointCloud::new expects an (n, 3) array".to_string(),
            ));
        }
        Ok(Self {
            points,
            colors: None,
            normals: None,
        })
    }

    /// Convenience constructor from a `Vec` of `[f64; 3]` triples.
    pub fn from_vec(points: Vec<[f64; 3]>) -> Self {
        let n = points.len();
        let mut arr = Array2::zeros((n, 3));
        for (i, p) in points.iter().enumerate() {
            arr[[i, 0]] = p[0];
            arr[[i, 1]] = p[1];
            arr[[i, 2]] = p[2];
        }
        Self {
            points: arr,
            colors: None,
            normals: None,
        }
    }

    /// Number of points.
    #[inline]
    pub fn n_points(&self) -> usize {
        self.points.nrows()
    }

    /// Centroid (mean position) of the point cloud.
    pub fn centroid(&self) -> [f64; 3] {
        let n = self.n_points();
        if n == 0 {
            return [0.0; 3];
        }
        let mut sum = [0.0f64; 3];
        for i in 0..n {
            sum[0] += self.points[[i, 0]];
            sum[1] += self.points[[i, 1]];
            sum[2] += self.points[[i, 2]];
        }
        let n_f = n as f64;
        [sum[0] / n_f, sum[1] / n_f, sum[2] / n_f]
    }

    /// Axis-aligned bounding box: `(min_xyz, max_xyz)`.
    pub fn bounding_box(&self) -> ([f64; 3], [f64; 3]) {
        let n = self.n_points();
        if n == 0 {
            return ([0.0; 3], [0.0; 3]);
        }
        let mut mn = [f64::INFINITY; 3];
        let mut mx = [f64::NEG_INFINITY; 3];
        for i in 0..n {
            for k in 0..3 {
                let v = self.points[[i, k]];
                if v < mn[k] {
                    mn[k] = v;
                }
                if v > mx[k] {
                    mx[k] = v;
                }
            }
        }
        (mn, mx)
    }

    /// Voxel grid downsampling.
    ///
    /// Each voxel of side `voxel_size` retains the centroid of the points
    /// that fall into it.  Colours and normals are averaged when present.
    pub fn voxel_downsample(&self, voxel_size: f64) -> Result<PointCloud> {
        if voxel_size <= 0.0 {
            return Err(VisionError::InvalidParameter(
                "voxel_size must be positive".to_string(),
            ));
        }
        let n = self.n_points();
        if n == 0 {
            return PointCloud::new(Array2::zeros((0, 3)));
        }

        // Map each point to its voxel key.
        use std::collections::HashMap;
        // key → (sum_xyz, count, sum_rgb, sum_nrm)
        #[allow(clippy::type_complexity)]
        let mut voxels: HashMap<
            (i64, i64, i64),
            (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<[f64; 3]>),
        > = HashMap::new();

        for i in 0..n {
            let x = self.points[[i, 0]];
            let y = self.points[[i, 1]];
            let z = self.points[[i, 2]];
            let key = (
                (x / voxel_size).floor() as i64,
                (y / voxel_size).floor() as i64,
                (z / voxel_size).floor() as i64,
            );
            let entry = voxels
                .entry(key)
                .or_insert_with(|| (Vec::new(), Vec::new(), Vec::new()));
            entry.0.push([x, y, z]);
            if let Some(ref c) = self.colors {
                entry.1.push([c[[i, 0]], c[[i, 1]], c[[i, 2]]]);
            }
            if let Some(ref nrm) = self.normals {
                entry.2.push([nrm[[i, 0]], nrm[[i, 1]], nrm[[i, 2]]]);
            }
        }

        let m = voxels.len();
        let mut pts = Array2::zeros((m, 3));
        let has_colors = self.colors.is_some();
        let has_normals = self.normals.is_some();
        let mut cols: Option<Array2<f64>> = if has_colors {
            Some(Array2::zeros((m, 3)))
        } else {
            None
        };
        let mut nrms: Option<Array2<f64>> = if has_normals {
            Some(Array2::zeros((m, 3)))
        } else {
            None
        };

        for (idx, (_, (ps, cs, ns))) in voxels.iter().enumerate() {
            let cnt = ps.len() as f64;
            for k in 0..3 {
                pts[[idx, k]] = ps.iter().map(|p| p[k]).sum::<f64>() / cnt;
            }
            if let Some(ref mut ca) = cols {
                for k in 0..3 {
                    ca[[idx, k]] = cs.iter().map(|c| c[k]).sum::<f64>() / cnt;
                }
            }
            if let Some(ref mut na) = nrms {
                let mut nv = [0.0f64; 3];
                for k in 0..3 {
                    nv[k] = ns.iter().map(|n| n[k]).sum::<f64>() / cnt;
                }
                let len = (nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]).sqrt();
                if len > 1e-12 {
                    for k in 0..3 {
                        na[[idx, k]] = nv[k] / len;
                    }
                }
            }
        }

        Ok(PointCloud {
            points: pts,
            colors: cols,
            normals: nrms,
        })
    }

    /// Statistical outlier removal.
    ///
    /// For each point, computes the mean distance to its `nb_neighbors`
    /// nearest neighbours.  Points whose mean distance exceeds
    /// `global_mean + std_ratio * global_std` are removed.
    pub fn remove_statistical_outliers(
        &self,
        nb_neighbors: usize,
        std_ratio: f64,
    ) -> Result<PointCloud> {
        if nb_neighbors == 0 {
            return Err(VisionError::InvalidParameter(
                "nb_neighbors must be > 0".to_string(),
            ));
        }
        let n = self.n_points();
        if n <= nb_neighbors {
            // Not enough points; return a clone.
            return Ok(self.clone());
        }

        let tree = PointKdTree::build(self);
        let mut mean_dists = Vec::with_capacity(n);

        for i in 0..n {
            let q = [
                self.points[[i, 0]],
                self.points[[i, 1]],
                self.points[[i, 2]],
            ];
            // k+1 because the point itself is the first result.
            let nn = tree.knn(&q, nb_neighbors + 1);
            // Skip self (dist² == 0).
            let valid: Vec<f64> = nn
                .iter()
                .filter(|(_, d2)| *d2 > 1e-20)
                .map(|(_, d2)| d2.sqrt())
                .collect();
            if valid.is_empty() {
                mean_dists.push(0.0f64);
            } else {
                mean_dists.push(valid.iter().sum::<f64>() / valid.len() as f64);
            }
        }

        let global_mean = mean_dists.iter().sum::<f64>() / n as f64;
        let variance = mean_dists
            .iter()
            .map(|d| (d - global_mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let global_std = variance.sqrt();
        let threshold = global_mean + std_ratio * global_std;

        let keep: Vec<usize> = mean_dists
            .iter()
            .enumerate()
            .filter(|(_, d)| **d <= threshold)
            .map(|(i, _)| i)
            .collect();

        Ok(self.select_indices(&keep))
    }

    /// Radius outlier removal.
    ///
    /// Removes points that have fewer than `nb_points` neighbours within
    /// `radius`.
    pub fn remove_radius_outliers(&self, nb_points: usize, radius: f64) -> Result<PointCloud> {
        if radius <= 0.0 {
            return Err(VisionError::InvalidParameter(
                "radius must be positive".to_string(),
            ));
        }
        let n = self.n_points();
        let tree = PointKdTree::build(self);
        let mut keep = Vec::with_capacity(n);

        for i in 0..n {
            let q = [
                self.points[[i, 0]],
                self.points[[i, 1]],
                self.points[[i, 2]],
            ];
            let neighbours = tree.radius_search(&q, radius);
            // Subtract 1 for the point itself.
            let cnt = neighbours.iter().filter(|(_, d2)| *d2 > 1e-20).count();
            if cnt >= nb_points {
                keep.push(i);
            }
        }

        Ok(self.select_indices(&keep))
    }

    /// Estimate surface normals via PCA on each point's k-nearest-neighbour
    /// neighbourhood.
    ///
    /// Normals are oriented towards the origin (simple heuristic).
    pub fn estimate_normals(&mut self, k: usize) -> Result<()> {
        if k < 3 {
            return Err(VisionError::InvalidParameter(
                "k must be >= 3 for normal estimation".to_string(),
            ));
        }
        let n = self.n_points();
        let tree = PointKdTree::build(self);
        let mut normals = Array2::zeros((n, 3));

        for i in 0..n {
            let q = [
                self.points[[i, 0]],
                self.points[[i, 1]],
                self.points[[i, 2]],
            ];
            let mut nn = tree.knn(&q, k + 1);
            // Include the point itself by keeping up to k neighbours.
            nn.truncate(k + 1);
            let nbrs: Vec<usize> = nn.iter().map(|(idx, _)| *idx).collect();

            // Compute centroid of the neighbourhood.
            let mut cen = [0.0f64; 3];
            for &j in &nbrs {
                cen[0] += self.points[[j, 0]];
                cen[1] += self.points[[j, 1]];
                cen[2] += self.points[[j, 2]];
            }
            let cnt = nbrs.len() as f64;
            cen[0] /= cnt;
            cen[1] /= cnt;
            cen[2] /= cnt;

            // Build 3×m centered matrix.
            let m_pts = nbrs.len();
            let mut mat = Array2::zeros((m_pts, 3));
            for (row, &j) in nbrs.iter().enumerate() {
                mat[[row, 0]] = self.points[[j, 0]] - cen[0];
                mat[[row, 1]] = self.points[[j, 1]] - cen[1];
                mat[[row, 2]] = self.points[[j, 2]] - cen[2];
            }

            // Covariance (3×3) = matT × mat.
            let cov = mat.t().dot(&mat);

            // SVD on the 3×3 covariance – the normal is the right singular
            // vector corresponding to the smallest singular value.
            let normal = match svd(&cov.view(), false, None) {
                Ok((_u, s, vt)) => {
                    // Singular values are in descending order; smallest is last.
                    let min_idx = if s[0] <= s[1] && s[0] <= s[2] {
                        0
                    } else if s[1] <= s[2] {
                        1
                    } else {
                        2
                    };
                    [vt[[min_idx, 0]], vt[[min_idx, 1]], vt[[min_idx, 2]]]
                }
                Err(_) => {
                    // Fall back to z-axis normal.
                    [0.0, 0.0, 1.0]
                }
            };

            // Orient towards origin.
            let dot = normal[0] * (-cen[0]) + normal[1] * (-cen[1]) + normal[2] * (-cen[2]);
            let sign = if dot < 0.0 { -1.0 } else { 1.0 };
            let len =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            if len > 1e-12 {
                normals[[i, 0]] = sign * normal[0] / len;
                normals[[i, 1]] = sign * normal[1] / len;
                normals[[i, 2]] = sign * normal[2] / len;
            } else {
                normals[[i, 0]] = 0.0;
                normals[[i, 1]] = 0.0;
                normals[[i, 2]] = 1.0;
            }
        }

        self.normals = Some(normals);
        Ok(())
    }

    /// Apply a 4×4 homogeneous transformation matrix.
    ///
    /// The input array must have shape (4, 4).
    pub fn transform(&self, t: &Array2<f64>) -> Result<PointCloud> {
        if t.shape() != [4, 4] {
            return Err(VisionError::InvalidParameter(
                "Transformation matrix must be 4×4".to_string(),
            ));
        }
        let n = self.n_points();
        let mut new_pts = Array2::zeros((n, 3));
        for i in 0..n {
            let x = self.points[[i, 0]];
            let y = self.points[[i, 1]];
            let z = self.points[[i, 2]];
            new_pts[[i, 0]] = t[[0, 0]] * x + t[[0, 1]] * y + t[[0, 2]] * z + t[[0, 3]];
            new_pts[[i, 1]] = t[[1, 0]] * x + t[[1, 1]] * y + t[[1, 2]] * z + t[[1, 3]];
            new_pts[[i, 2]] = t[[2, 0]] * x + t[[2, 1]] * y + t[[2, 2]] * z + t[[2, 3]];
        }
        let new_nrm = self.normals.as_ref().map(|nrm| {
            let mut nn = Array2::zeros((n, 3));
            for i in 0..n {
                let nx = nrm[[i, 0]];
                let ny = nrm[[i, 1]];
                let nz = nrm[[i, 2]];
                let rx = t[[0, 0]] * nx + t[[0, 1]] * ny + t[[0, 2]] * nz;
                let ry = t[[1, 0]] * nx + t[[1, 1]] * ny + t[[1, 2]] * nz;
                let rz = t[[2, 0]] * nx + t[[2, 1]] * ny + t[[2, 2]] * nz;
                let len = (rx * rx + ry * ry + rz * rz).sqrt();
                if len > 1e-12 {
                    nn[[i, 0]] = rx / len;
                    nn[[i, 1]] = ry / len;
                    nn[[i, 2]] = rz / len;
                }
            }
            nn
        });
        Ok(PointCloud {
            points: new_pts,
            colors: self.colors.clone(),
            normals: new_nrm,
        })
    }

    /// Axis-aligned bounding-box crop.
    ///
    /// Returns a new cloud containing only the points strictly inside
    /// `[min_pt, max_pt]` (inclusive bounds).
    pub fn crop_box(&self, min_pt: &[f64; 3], max_pt: &[f64; 3]) -> PointCloud {
        let n = self.n_points();
        let mut keep = Vec::with_capacity(n);
        for i in 0..n {
            let x = self.points[[i, 0]];
            let y = self.points[[i, 1]];
            let z = self.points[[i, 2]];
            if x >= min_pt[0]
                && x <= max_pt[0]
                && y >= min_pt[1]
                && y <= max_pt[1]
                && z >= min_pt[2]
                && z <= max_pt[2]
            {
                keep.push(i);
            }
        }
        self.select_indices(&keep)
    }

    // ─────────────────── internal helpers ───────────────────

    /// Build a sub-cloud from an index list.
    fn select_indices(&self, indices: &[usize]) -> PointCloud {
        let m = indices.len();
        let mut pts = Array2::zeros((m, 3));
        let mut cols: Option<Array2<f64>> = self.colors.as_ref().map(|_| Array2::zeros((m, 3)));
        let mut nrms: Option<Array2<f64>> = self.normals.as_ref().map(|_| Array2::zeros((m, 3)));

        for (row, &i) in indices.iter().enumerate() {
            for k in 0..3 {
                pts[[row, k]] = self.points[[i, k]];
            }
            if let (Some(ref mut ca), Some(ref c)) = (&mut cols, &self.colors) {
                for k in 0..3 {
                    ca[[row, k]] = c[[i, k]];
                }
            }
            if let (Some(ref mut na), Some(ref nrm)) = (&mut nrms, &self.normals) {
                for k in 0..3 {
                    na[[row, k]] = nrm[[i, k]];
                }
            }
        }

        PointCloud {
            points: pts,
            colors: cols,
            normals: nrms,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PointKdTree
// ─────────────────────────────────────────────────────────────────────────────

/// A simple recursive K-D tree for 3-D nearest-neighbour queries.
///
/// Built once from a [`PointCloud`], then used for multiple queries.
pub struct PointKdTree {
    nodes: Vec<KdNode>,
    points: Vec<[f64; 3]>,
}

#[derive(Clone)]
struct KdNode {
    /// Index into `PointKdTree::points`.
    point_idx: usize,
    /// Split axis (0 = x, 1 = y, 2 = z).
    axis: usize,
    left: Option<usize>,  // child node index
    right: Option<usize>, // child node index
}

impl PointKdTree {
    /// Build the tree from a point cloud.
    pub fn build(cloud: &PointCloud) -> Self {
        let n = cloud.n_points();
        let pts: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                [
                    cloud.points[[i, 0]],
                    cloud.points[[i, 1]],
                    cloud.points[[i, 2]],
                ]
            })
            .collect();
        let mut indices: Vec<usize> = (0..n).collect();
        let mut nodes: Vec<KdNode> = Vec::with_capacity(n);
        Self::build_recursive(&pts, &mut indices, 0, &mut nodes);
        PointKdTree { nodes, points: pts }
    }

    fn build_recursive(
        pts: &[[f64; 3]],
        indices: &mut [usize],
        depth: usize,
        nodes: &mut Vec<KdNode>,
    ) -> Option<usize> {
        if indices.is_empty() {
            return None;
        }
        let axis = depth % 3;
        indices.sort_unstable_by(|&a, &b| {
            pts[a][axis]
                .partial_cmp(&pts[b][axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mid = indices.len() / 2;
        let node_idx = nodes.len();
        nodes.push(KdNode {
            point_idx: indices[mid],
            axis,
            left: None,
            right: None,
        });
        let left_child = Self::build_recursive(pts, &mut indices[..mid], depth + 1, nodes);
        nodes[node_idx].left = left_child;
        let right_child = Self::build_recursive(pts, &mut indices[mid + 1..], depth + 1, nodes);
        nodes[node_idx].right = right_child;
        Some(node_idx)
    }

    /// Return the `k` nearest neighbours of `query` as `(index, dist²)` pairs,
    /// sorted by ascending distance squared.
    pub fn knn(&self, query: &[f64; 3], k: usize) -> Vec<(usize, f64)> {
        if self.nodes.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut result: Vec<(usize, f64)> = Vec::with_capacity(k);

        // Bounded best list: (dist², idx), kept sorted descending so best[0] is worst.
        let mut best: Vec<(f64, usize)> = Vec::with_capacity(k + 1); // (dist², idx)

        self.knn_recursive(0, query, k, &mut best);

        best.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (d2, idx) in best {
            result.push((idx, d2));
        }
        result
    }

    fn knn_recursive(
        &self,
        node_idx: usize,
        query: &[f64; 3],
        k: usize,
        best: &mut Vec<(f64, usize)>,
    ) {
        if node_idx >= self.nodes.len() {
            return;
        }
        let node = &self.nodes[node_idx];
        let pt = self.points[node.point_idx];
        let d2 = dist2(query, &pt);

        // Attempt to insert into best list.
        if best.len() < k {
            best.push((d2, node.point_idx));
            // Keep sorted by dist² descending so best[0] is worst.
            best.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            let worst_d2 = best[0].0;
            if d2 < worst_d2 {
                best[0] = (d2, node.point_idx);
                best.sort_unstable_by(|a, b| {
                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        let axis = node.axis;
        let diff = query[axis] - pt[axis];
        let (near_child, far_child) = if diff <= 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        if let Some(nc) = near_child {
            self.knn_recursive(nc, query, k, best);
        }
        // Recompute worst after near subtree.
        let current_worst = if best.len() < k {
            f64::INFINITY
        } else {
            best[0].0
        };
        // Only visit the far subtree if the splitting plane is within range.
        if diff * diff < current_worst {
            if let Some(fc) = far_child {
                self.knn_recursive(fc, query, k, best);
            }
        }
    }

    /// Return all points within squared distance ≤ `radius²` from `query`.
    ///
    /// Returns `(index, dist²)` pairs, unsorted.
    pub fn radius_search(&self, query: &[f64; 3], radius: f64) -> Vec<(usize, f64)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let r2 = radius * radius;
        let mut result = Vec::new();
        self.radius_recursive(0, query, r2, &mut result);
        result
    }

    fn radius_recursive(
        &self,
        node_idx: usize,
        query: &[f64; 3],
        r2: f64,
        result: &mut Vec<(usize, f64)>,
    ) {
        if node_idx >= self.nodes.len() {
            return;
        }
        let node = &self.nodes[node_idx];
        let pt = self.points[node.point_idx];
        let d2 = dist2(query, &pt);
        if d2 <= r2 {
            result.push((node.point_idx, d2));
        }
        let axis = node.axis;
        let diff = query[axis] - pt[axis];
        if let Some(nc) = if diff <= 0.0 { node.left } else { node.right } {
            self.radius_recursive(nc, query, r2, result);
        }
        if diff * diff <= r2 {
            if let Some(fc) = if diff <= 0.0 { node.right } else { node.left } {
                self.radius_recursive(fc, query, r2, result);
            }
        }
    }
}

#[inline]
fn dist2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

// ─────────────────────────────────────────────────────────────────────────────
// ICP result
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a point cloud registration algorithm.
#[derive(Debug, Clone)]
pub struct IcpResult {
    /// 4×4 homogeneous rigid-body transformation (source → target space).
    pub transformation: Array2<f64>,
    /// Fraction of source points that found a close-enough correspondence.
    pub fitness: f64,
    /// Root-mean-square error of inlier correspondences.
    pub inlier_rmse: f64,
    /// Whether the algorithm converged within `max_iter` iterations.
    pub converged: bool,
    /// Number of iterations executed.
    pub n_iterations: usize,
}

impl IcpResult {
    fn identity() -> Self {
        let mut t = Array2::zeros((4, 4));
        t[[0, 0]] = 1.0;
        t[[1, 1]] = 1.0;
        t[[2, 2]] = 1.0;
        t[[3, 3]] = 1.0;
        IcpResult {
            transformation: t,
            fitness: 0.0,
            inlier_rmse: 0.0,
            converged: false,
            n_iterations: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ICP
// ─────────────────────────────────────────────────────────────────────────────

/// Point-to-point Iterative Closest Point (ICP) registration.
///
/// Aligns `source` into the coordinate frame of `target` by iteratively
/// finding nearest-neighbour correspondences and solving the SVD-based
/// optimal rigid-body transform.
///
/// # Arguments
/// * `source` – the cloud to be aligned.
/// * `target` – the reference cloud.
/// * `max_correspondence_dist` – only correspondences with distance ≤ this
///   value contribute to the estimate.
/// * `max_iter` – maximum number of ICP iterations.
/// * `tolerance` – convergence threshold: stop when the change in RMSE
///   between two consecutive iterations is smaller than this value.
pub fn icp(
    source: &PointCloud,
    target: &PointCloud,
    max_correspondence_dist: f64,
    max_iter: usize,
    tolerance: f64,
) -> Result<IcpResult> {
    if source.n_points() == 0 || target.n_points() == 0 {
        return Err(VisionError::InvalidParameter(
            "Source and target clouds must be non-empty".to_string(),
        ));
    }

    let target_tree = PointKdTree::build(target);
    // Running transformation (accumulated across iterations).
    let mut accum = identity4();
    // Working copy of source that we transform at each step.
    let mut src = source.clone();
    let mut prev_rmse = f64::INFINITY;
    let mcd2 = max_correspondence_dist * max_correspondence_dist;

    for iter in 0..max_iter {
        let ns = src.n_points();
        let nt = target.n_points();

        // Find correspondences.
        let mut src_pts: Vec<[f64; 3]> = Vec::with_capacity(ns);
        let mut tgt_pts: Vec<[f64; 3]> = Vec::with_capacity(ns);

        for i in 0..ns {
            let q = [src.points[[i, 0]], src.points[[i, 1]], src.points[[i, 2]]];
            let nn = target_tree.knn(&q, 1);
            if let Some(&(j, d2)) = nn.first() {
                if d2 <= mcd2 && j < nt {
                    src_pts.push(q);
                    tgt_pts.push([
                        target.points[[j, 0]],
                        target.points[[j, 1]],
                        target.points[[j, 2]],
                    ]);
                }
            }
        }

        let n_corr = src_pts.len();
        if n_corr == 0 {
            break;
        }

        // Compute centroids.
        let mu_s = centroid_of(&src_pts);
        let mu_t = centroid_of(&tgt_pts);

        // Cross-covariance.
        let mut h = [[0.0f64; 3]; 3];
        for k in 0..n_corr {
            for r in 0..3 {
                for c in 0..3 {
                    h[r][c] += (src_pts[k][r] - mu_s[r]) * (tgt_pts[k][c] - mu_t[c]);
                }
            }
        }

        let h_arr = arr2_3x3(&h);
        let (u, _s, vt) =
            svd(&h_arr.view(), false, None).map_err(|e| VisionError::LinAlgError(e.to_string()))?;

        // R = V × Uᵀ
        let vt_t = vt.t().to_owned();
        let u_t = u.t().to_owned();
        let r = vt_t.dot(&u_t);

        // Fix reflection if det(R) < 0.
        let det_r = mat3_det(&r);
        let r_fixed = if det_r < 0.0 {
            // Flip the last column of V.
            let mut vt2 = vt.t().to_owned();
            vt2[[0, 2]] = -vt2[[0, 2]];
            vt2[[1, 2]] = -vt2[[1, 2]];
            vt2[[2, 2]] = -vt2[[2, 2]];
            vt2.dot(&u_t)
        } else {
            r
        };

        // t = mu_t - R × mu_s
        let t_vec = [
            mu_t[0]
                - (r_fixed[[0, 0]] * mu_s[0]
                    + r_fixed[[0, 1]] * mu_s[1]
                    + r_fixed[[0, 2]] * mu_s[2]),
            mu_t[1]
                - (r_fixed[[1, 0]] * mu_s[0]
                    + r_fixed[[1, 1]] * mu_s[1]
                    + r_fixed[[1, 2]] * mu_s[2]),
            mu_t[2]
                - (r_fixed[[2, 0]] * mu_s[0]
                    + r_fixed[[2, 1]] * mu_s[1]
                    + r_fixed[[2, 2]] * mu_s[2]),
        ];

        // Build 4×4 transform.
        let step_t = build_transform(&r_fixed, &t_vec);

        // Apply step to the working source.
        src = src.transform(&step_t)?;

        // Accumulate.
        accum = compose_transforms(&step_t, &accum);

        // RMSE.
        let rmse: f64 = src_pts
            .iter()
            .zip(tgt_pts.iter())
            .map(|(s, t2)| {
                let dx = s[0] - t2[0];
                let dy = s[1] - t2[1];
                let dz = s[2] - t2[2];
                dx * dx + dy * dy + dz * dz
            })
            .sum::<f64>()
            / n_corr as f64;
        let rmse = rmse.sqrt();

        if (prev_rmse - rmse).abs() < tolerance {
            return Ok(IcpResult {
                transformation: accum,
                fitness: n_corr as f64 / ns as f64,
                inlier_rmse: rmse,
                converged: true,
                n_iterations: iter + 1,
            });
        }
        prev_rmse = rmse;
    }

    let ns = src.n_points();
    let nt = target.n_points();
    let mut final_corr = 0usize;
    let mut final_rmse_sum = 0.0f64;

    for i in 0..ns {
        let q = [src.points[[i, 0]], src.points[[i, 1]], src.points[[i, 2]]];
        let nn = target_tree.knn(&q, 1);
        if let Some(&(j, d2)) = nn.first() {
            if d2 <= mcd2 && j < nt {
                final_corr += 1;
                final_rmse_sum += d2;
            }
        }
    }

    let fitness = final_corr as f64 / ns.max(1) as f64;
    let inlier_rmse = if final_corr > 0 {
        (final_rmse_sum / final_corr as f64).sqrt()
    } else {
        0.0
    };

    Ok(IcpResult {
        transformation: accum,
        fitness,
        inlier_rmse,
        converged: false,
        n_iterations: max_iter,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// RANSAC Registration
// ─────────────────────────────────────────────────────────────────────────────

/// RANSAC-based rigid registration.
///
/// Randomly samples 3-point minimal subsets, estimates a rigid transform, and
/// counts inliers.  Returns the best-scoring transform, refined with a final
/// ICP pass.
pub fn ransac_registration(
    source: &PointCloud,
    target: &PointCloud,
    max_correspondence_dist: f64,
    n_iterations: usize,
    seed: u64,
) -> Result<IcpResult> {
    if source.n_points() < 3 || target.n_points() < 3 {
        return Err(VisionError::InvalidParameter(
            "Both clouds must have at least 3 points for RANSAC".to_string(),
        ));
    }

    let mcd2 = max_correspondence_dist * max_correspondence_dist;
    let ns = source.n_points();
    let nt = target.n_points();
    let target_tree = PointKdTree::build(target);

    let mut best_inliers = 0usize;
    let mut best_t = identity4();

    // Minimal lcg-based RNG (no external deps).
    let mut rng = LcgRng::new(seed ^ 0xDEAD_BEEF_1234_5678);

    for _ in 0..n_iterations {
        // Sample 3 source indices.
        let i0 = rng.next_usize(ns);
        let i1 = rng.next_usize(ns);
        let i2 = rng.next_usize(ns);
        if i0 == i1 || i1 == i2 || i0 == i2 {
            continue;
        }

        // Their nearest neighbours in target.
        let s_pts = [
            [
                source.points[[i0, 0]],
                source.points[[i0, 1]],
                source.points[[i0, 2]],
            ],
            [
                source.points[[i1, 0]],
                source.points[[i1, 1]],
                source.points[[i1, 2]],
            ],
            [
                source.points[[i2, 0]],
                source.points[[i2, 1]],
                source.points[[i2, 2]],
            ],
        ];
        let mut t_pts = [[0.0f64; 3]; 3];
        let mut valid = true;
        for (k, sp) in s_pts.iter().enumerate() {
            let nn = target_tree.knn(sp, 1);
            match nn.first() {
                Some(&(j, d2)) if d2 <= mcd2 && j < nt => {
                    t_pts[k] = [
                        target.points[[j, 0]],
                        target.points[[j, 1]],
                        target.points[[j, 2]],
                    ];
                }
                _ => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid {
            continue;
        }

        // Estimate transform from 3 correspondences.
        let s_slice = s_pts.to_vec();
        let t_slice = t_pts.to_vec();
        let candidate = match estimate_rigid_transform_3pts(&s_slice, &t_slice) {
            Some(t) => t,
            None => continue,
        };

        // Count inliers.
        let mut inliers = 0usize;
        for i in 0..ns {
            let x = source.points[[i, 0]];
            let y = source.points[[i, 1]];
            let z = source.points[[i, 2]];
            let tx = candidate[[0, 0]] * x
                + candidate[[0, 1]] * y
                + candidate[[0, 2]] * z
                + candidate[[0, 3]];
            let ty = candidate[[1, 0]] * x
                + candidate[[1, 1]] * y
                + candidate[[1, 2]] * z
                + candidate[[1, 3]];
            let tz = candidate[[2, 0]] * x
                + candidate[[2, 1]] * y
                + candidate[[2, 2]] * z
                + candidate[[2, 3]];
            let nn = target_tree.knn(&[tx, ty, tz], 1);
            if let Some(&(_, d2)) = nn.first() {
                if d2 <= mcd2 {
                    inliers += 1;
                }
            }
        }

        if inliers > best_inliers {
            best_inliers = inliers;
            best_t = candidate;
        }
    }

    // Refine with ICP.
    let transformed = source.transform(&best_t)?;
    let refined = icp(&transformed, target, max_correspondence_dist, 50, 1e-6)?;
    let final_t = compose_transforms(&refined.transformation, &best_t);

    Ok(IcpResult {
        transformation: final_t,
        fitness: refined.fitness,
        inlier_rmse: refined.inlier_rmse,
        converged: refined.converged,
        n_iterations: n_iterations + refined.n_iterations,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn identity4() -> Array2<f64> {
    let mut t = Array2::zeros((4, 4));
    t[[0, 0]] = 1.0;
    t[[1, 1]] = 1.0;
    t[[2, 2]] = 1.0;
    t[[3, 3]] = 1.0;
    t
}

fn arr2_3x3(m: &[[f64; 3]; 3]) -> Array2<f64> {
    let mut a = Array2::zeros((3, 3));
    for r in 0..3 {
        for c in 0..3 {
            a[[r, c]] = m[r][c];
        }
    }
    a
}

fn mat3_det(m: &Array2<f64>) -> f64 {
    m[[0, 0]] * (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]])
        - m[[0, 1]] * (m[[1, 0]] * m[[2, 2]] - m[[1, 2]] * m[[2, 0]])
        + m[[0, 2]] * (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]])
}

fn centroid_of(pts: &[[f64; 3]]) -> [f64; 3] {
    let n = pts.len() as f64;
    let mut s = [0.0f64; 3];
    for p in pts {
        s[0] += p[0];
        s[1] += p[1];
        s[2] += p[2];
    }
    [s[0] / n, s[1] / n, s[2] / n]
}

fn build_transform(r: &Array2<f64>, t: &[f64; 3]) -> Array2<f64> {
    let mut m = Array2::zeros((4, 4));
    for row in 0..3 {
        for col in 0..3 {
            m[[row, col]] = r[[row, col]];
        }
        m[[row, 3]] = t[row];
    }
    m[[3, 3]] = 1.0;
    m
}

/// Compose T_new = T_step × T_old (4×4 matrix multiply).
fn compose_transforms(step: &Array2<f64>, old: &Array2<f64>) -> Array2<f64> {
    step.dot(old)
}

/// Estimate rigid transform from exactly 3 point correspondences using SVD.
fn estimate_rigid_transform_3pts(src: &[[f64; 3]], tgt: &[[f64; 3]]) -> Option<Array2<f64>> {
    let mu_s = centroid_of(src);
    let mu_t = centroid_of(tgt);
    let mut h = [[0.0f64; 3]; 3];
    for k in 0..3 {
        for r in 0..3 {
            for c in 0..3 {
                h[r][c] += (src[k][r] - mu_s[r]) * (tgt[k][c] - mu_t[c]);
            }
        }
    }
    let h_arr = arr2_3x3(&h);
    let (u, _s, vt) = svd(&h_arr.view(), false, None).ok()?;
    let vt_t = vt.t().to_owned();
    let u_t = u.t().to_owned();
    let mut r = vt_t.dot(&u_t);
    if mat3_det(&r) < 0.0 {
        let mut vt2 = vt.t().to_owned();
        vt2[[0, 2]] = -vt2[[0, 2]];
        vt2[[1, 2]] = -vt2[[1, 2]];
        vt2[[2, 2]] = -vt2[[2, 2]];
        r = vt2.dot(&u_t);
    }
    let t_vec = [
        mu_t[0] - (r[[0, 0]] * mu_s[0] + r[[0, 1]] * mu_s[1] + r[[0, 2]] * mu_s[2]),
        mu_t[1] - (r[[1, 0]] * mu_s[0] + r[[1, 1]] * mu_s[1] + r[[1, 2]] * mu_s[2]),
        mu_t[2] - (r[[2, 0]] * mu_s[0] + r[[2, 1]] * mu_s[1] + r[[2, 2]] * mu_s[2]),
    ];
    Some(build_transform(&r, &t_vec))
}

// Simple Linear Congruential Generator (period 2^64) for reproducible RANSAC.
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        LcgRng { state: seed | 1 }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }
    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Build a simple cube-like point cloud for testing.
    fn make_cube(n: usize) -> PointCloud {
        let step = 1.0 / (n as f64 - 1.0).max(1.0);
        let mut pts = Vec::with_capacity(n * n * n);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    pts.push([i as f64 * step, j as f64 * step, k as f64 * step]);
                }
            }
        }
        PointCloud::from_vec(pts)
    }

    #[test]
    fn test_pointcloud_new_valid() {
        let arr = Array2::zeros((10, 3));
        let pc = PointCloud::new(arr);
        assert!(pc.is_ok());
        assert_eq!(pc.map(|p| p.n_points()).unwrap_or(0), 10);
    }

    #[test]
    fn test_pointcloud_new_invalid_shape() {
        let arr = Array2::zeros((10, 2));
        let pc = PointCloud::new(arr);
        assert!(pc.is_err());
    }

    #[test]
    fn test_from_vec_and_n_points() {
        let pts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let pc = PointCloud::from_vec(pts);
        assert_eq!(pc.n_points(), 3);
    }

    #[test]
    fn test_centroid() {
        let pts = vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
        ];
        let pc = PointCloud::from_vec(pts);
        let c = pc.centroid();
        assert_abs_diff_eq!(c[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[2], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_bounding_box() {
        let pts = vec![[0.0, 1.0, -1.0], [3.0, -2.0, 5.0]];
        let pc = PointCloud::from_vec(pts);
        let (mn, mx) = pc.bounding_box();
        assert_abs_diff_eq!(mn[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(mn[1], -2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(mn[2], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(mx[0], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(mx[1], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(mx[2], 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_voxel_downsample_reduces_count() {
        let cloud = make_cube(5); // 125 points on a 1×1×1 grid
        let downsampled = cloud.voxel_downsample(0.5).expect("downsample failed");
        assert!(
            downsampled.n_points() < cloud.n_points(),
            "Downsampled count {} should be < original {}",
            downsampled.n_points(),
            cloud.n_points()
        );
    }

    #[test]
    fn test_voxel_downsample_empty() {
        let cloud = PointCloud::from_vec(vec![]);
        let down = cloud
            .voxel_downsample(0.1)
            .expect("downsample of empty failed");
        assert_eq!(down.n_points(), 0);
    }

    #[test]
    fn test_statistical_outlier_removal() {
        // Cloud with one outlier far away.
        let mut pts: Vec<[f64; 3]> = (0..20)
            .flat_map(|i| (0..20).map(move |j| [i as f64 * 0.1, j as f64 * 0.1, 0.0]))
            .collect();
        pts.push([100.0, 100.0, 100.0]); // clear outlier
        let cloud = PointCloud::from_vec(pts.clone());
        let n_before = cloud.n_points();
        let filtered = cloud
            .remove_statistical_outliers(5, 1.0)
            .expect("filter failed");
        assert!(
            filtered.n_points() < n_before,
            "outlier removal should reduce count"
        );
    }

    #[test]
    fn test_normal_estimation_unit_length() {
        let mut cloud = make_cube(4); // 64 points
        cloud.estimate_normals(6).expect("normal estimation failed");
        let nrm = cloud.normals.as_ref().expect("normals should be set");
        let n = cloud.n_points();
        for i in 0..n {
            let nx = nrm[[i, 0]];
            let ny = nrm[[i, 1]];
            let nz = nrm[[i, 2]];
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            assert_abs_diff_eq!(len, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_kdtree_knn() {
        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ];
        let cloud = PointCloud::from_vec(pts);
        let tree = PointKdTree::build(&cloud);
        let result = tree.knn(&[0.5, 0.0, 0.0], 2);
        assert_eq!(result.len(), 2);
        // Nearest should be index 0 or 1 (both at dist² 0.25).
        let indices: Vec<usize> = result.iter().map(|(i, _)| *i).collect();
        assert!(indices.contains(&0) || indices.contains(&1));
    }

    #[test]
    fn test_kdtree_radius_search() {
        let pts = vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let cloud = PointCloud::from_vec(pts);
        let tree = PointKdTree::build(&cloud);
        let result = tree.radius_search(&[0.0, 0.0, 0.0], 1.0);
        // Points at 0.0 and 0.5 should be within radius 1.0.
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_icp_identical_clouds_converges() {
        let cloud = make_cube(3); // 27 points
        let result = icp(&cloud, &cloud, 1.0, 50, 1e-6).expect("ICP failed");
        // On identical clouds the very first iteration should find near-zero error
        // and converge. Fitness should be 1.0 (all inliers).
        assert!(result.converged || result.inlier_rmse < 1e-6);
        // The transformation should be close to identity.
        let t = &result.transformation;
        assert_abs_diff_eq!(t[[0, 0]], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(t[[1, 1]], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(t[[2, 2]], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(t[[0, 3]], 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_ransac_registration_recovers_translation() {
        // Source shifted by (1, 0, 0).
        let source = make_cube(3);
        let target_pts: Vec<[f64; 3]> = (0..source.n_points())
            .map(|i| {
                [
                    source.points[[i, 0]] + 1.0,
                    source.points[[i, 1]],
                    source.points[[i, 2]],
                ]
            })
            .collect();
        let target = PointCloud::from_vec(target_pts);
        let result = ransac_registration(&source, &target, 0.2, 100, 42).expect("RANSAC failed");
        // Translation x component should be close to +1.0.
        let tx = result.transformation[[0, 3]];
        assert!((tx - 1.0).abs() < 0.5, "Expected tx ≈ 1.0, got {tx}");
    }

    #[test]
    fn test_crop_box() {
        let pts = vec![[0.5, 0.5, 0.5], [2.0, 0.5, 0.5], [-1.0, 0.5, 0.5]];
        let cloud = PointCloud::from_vec(pts);
        let cropped = cloud.crop_box(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]);
        assert_eq!(cropped.n_points(), 1);
    }

    #[test]
    fn test_transform_identity() {
        let cloud = make_cube(2); // 8 points
        let t = identity4();
        let transformed = cloud.transform(&t).expect("transform failed");
        for i in 0..cloud.n_points() {
            for k in 0..3 {
                assert_abs_diff_eq!(
                    cloud.points[[i, k]],
                    transformed.points[[i, k]],
                    epsilon = 1e-10
                );
            }
        }
    }
}
