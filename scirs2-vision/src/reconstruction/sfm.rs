//! Structure from Motion (SfM) algorithms
//!
//! Provides fundamental matrix/essential matrix estimation, triangulation,
//! bundle adjustment, and an incremental SfM pipeline with track management.

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// PointCloud
// ─────────────────────────────────────────────────────────────────────────────

/// Reconstructed 3D point cloud with optional colour and confidence fields.
#[derive(Debug, Clone)]
pub struct PointCloud {
    /// Shape (n, 3) – each row is `[x, y, z]` in world coordinates.
    pub points: Array2<f64>,
    /// Shape (n, 3) – RGB values in \[0, 1\].  `None` when unavailable.
    pub colors: Option<Array2<f64>>,
    /// Per-point confidence in \[0, 1\].  `None` when unavailable.
    pub confidence: Option<Array1<f64>>,
}

impl PointCloud {
    /// Create an empty point cloud.
    pub fn empty() -> Self {
        Self {
            points: Array2::zeros((0, 3)),
            colors: None,
            confidence: None,
        }
    }

    /// Create from a Vec of `[x, y, z]` triples.
    pub fn from_vec(pts: Vec<[f64; 3]>) -> Self {
        let n = pts.len();
        let mut arr = Array2::zeros((n, 3));
        for (i, p) in pts.iter().enumerate() {
            arr[[i, 0]] = p[0];
            arr[[i, 1]] = p[1];
            arr[[i, 2]] = p[2];
        }
        Self {
            points: arr,
            colors: None,
            confidence: None,
        }
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.points.nrows()
    }

    /// Whether the point cloud contains any points.
    pub fn is_empty(&self) -> bool {
        self.points.nrows() == 0
    }

    /// Merge another PointCloud into this one (ignores color/confidence for simplicity).
    pub fn merge(&mut self, other: &PointCloud) {
        let n0 = self.points.nrows();
        let n1 = other.points.nrows();
        if n1 == 0 {
            return;
        }
        let mut new_pts = Array2::zeros((n0 + n1, 3));
        for i in 0..n0 {
            for j in 0..3 {
                new_pts[[i, j]] = self.points[[i, j]];
            }
        }
        for i in 0..n1 {
            for j in 0..3 {
                new_pts[[n0 + i, j]] = other.points[[i, j]];
            }
        }
        self.points = new_pts;
        self.colors = None;
        self.confidence = None;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-product matrix (skew-symmetric) for vector `v = [v0, v1, v2]`.
#[inline]
fn cross_mat(v: &ArrayView1<f64>) -> Array2<f64> {
    let mut m = Array2::zeros((3, 3));
    m[[0, 1]] = -v[2];
    m[[0, 2]] = v[1];
    m[[1, 0]] = v[2];
    m[[1, 2]] = -v[0];
    m[[2, 0]] = -v[1];
    m[[2, 1]] = v[0];
    m
}

/// Dot product of two 1-D views.
#[inline]
fn dot1(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a 1-D view.
#[inline]
fn norm1(v: &ArrayView1<f64>) -> f64 {
    dot1(v, v).sqrt()
}

/// Multiply a 3×3 matrix by a 3-vector.
fn mat3_vec3(m: &Array2<f64>, v: &Array1<f64>) -> Array1<f64> {
    let mut out = Array1::zeros(3);
    for i in 0..3 {
        for j in 0..3 {
            out[i] += m[[i, j]] * v[j];
        }
    }
    out
}

/// 3×3 matrix product.
fn mat3_mul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let mut c = Array2::zeros((3, 3));
    for i in 0..3 {
        for k in 0..3 {
            for j in 0..3 {
                c[[i, j]] += a[[i, k]] * b[[k, j]];
            }
        }
    }
    c
}

/// 3×3 matrix transpose.
fn mat3_t(a: &Array2<f64>) -> Array2<f64> {
    let mut t = Array2::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            t[[i, j]] = a[[j, i]];
        }
    }
    t
}

/// Minimal Golub-Reinsch SVD for an m×n matrix (m >= n).
/// Returns (U, S, Vt) where A = U * diag(S) * Vt.
/// Uses Jacobi iterations for small matrices.
fn svd_small(a: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let m = a.nrows();
    let n = a.ncols();
    // Use one-sided Jacobi SVD for small matrices
    // ATA decomposition for Vt
    let mut ata = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..m {
                ata[[i, j]] += a[[k, i]] * a[[k, j]];
            }
        }
    }
    let (eigenvalues, v) = symmetric_eigen(&ata);
    let mut s = Array1::zeros(n);
    for i in 0..n {
        s[i] = eigenvalues[i].max(0.0).sqrt();
    }
    // U = A * V * Sigma^{-1}
    let mut u = Array2::<f64>::zeros((m, n.min(m)));
    for j in 0..n.min(m) {
        if s[j] > 1e-14 {
            for i in 0..m {
                for k in 0..n {
                    u[[i, j]] += a[[i, k]] * v[[k, j]];
                }
                u[[i, j]] /= s[j];
            }
        }
    }
    // Vt = V^T
    let mut vt = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            vt[[i, j]] = v[[j, i]];
        }
    }
    (u, s, vt)
}

/// Jacobi eigendecomposition of a symmetric matrix.
/// Returns (eigenvalues, eigenvectors) sorted by descending eigenvalue.
fn symmetric_eigen(a: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut d: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();
    let mut v: Array2<f64> = Array2::eye(n);
    let mut off_diag = a.clone();
    // Zero the diagonal of off_diag
    for i in 0..n {
        off_diag[[i, i]] = 0.0;
    }

    for _ in 0..100 {
        // Find max off-diagonal element
        let mut max_val = 0.0f64;
        let (mut p, mut q) = (0, 1);
        for i in 0..n {
            for j in (i + 1)..n {
                let v_abs = off_diag[[i, j]].abs();
                if v_abs > max_val {
                    max_val = v_abs;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }
        // Compute Jacobi rotation
        let theta = if (d[p] - d[q]).abs() < 1e-14 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * off_diag[[p, q]]) / (d[p] - d[q])).atan()
        };
        let c = theta.cos();
        let s = theta.sin();
        // Update diagonal
        let dp_new = c * c * d[p] + 2.0 * c * s * off_diag[[p, q]] + s * s * d[q];
        let dq_new = s * s * d[p] - 2.0 * c * s * off_diag[[p, q]] + c * c * d[q];
        d[p] = dp_new;
        d[q] = dq_new;
        // Update off_diag
        for r in 0..n {
            if r != p && r != q {
                let apr = off_diag[[r.min(p), r.max(p)]];
                let aqr = off_diag[[r.min(q), r.max(q)]];
                let new_apr = c * apr + s * aqr;
                let new_aqr = -s * apr + c * aqr;
                off_diag[[r.min(p), r.max(p)]] = new_apr;
                off_diag[[p.min(r), p.max(r)]] = new_apr;
                off_diag[[r.min(q), r.max(q)]] = new_aqr;
                off_diag[[q.min(r), q.max(r)]] = new_aqr;
            }
        }
        off_diag[[p, q]] = 0.0;
        off_diag[[q, p]] = 0.0;
        // Update eigenvectors
        for r in 0..n {
            let vr_p = v[[r, p]];
            let vr_q = v[[r, q]];
            v[[r, p]] = c * vr_p + s * vr_q;
            v[[r, q]] = -s * vr_p + c * vr_q;
        }
    }

    // Sort by descending eigenvalue
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| d[b].partial_cmp(&d[a]).unwrap_or(std::cmp::Ordering::Equal));
    let sorted_d: Vec<f64> = idx.iter().map(|&i| d[i]).collect();
    let mut sorted_v = Array2::<f64>::zeros((n, n));
    for (j, &i) in idx.iter().enumerate() {
        for r in 0..n {
            sorted_v[[r, j]] = v[[r, i]];
        }
    }
    (sorted_d, sorted_v)
}

/// Normalize 2D point correspondences for numerical conditioning.
/// Returns (normalized_pts, transform_matrix).
fn normalize_points(pts: &[[f64; 2]]) -> (Vec<[f64; 2]>, Array2<f64>) {
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p[0]).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p[1]).sum::<f64>() / n;
    let scale = {
        let mean_dist = pts
            .iter()
            .map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt())
            .sum::<f64>()
            / n;
        if mean_dist < 1e-10 {
            1.0
        } else {
            std::f64::consts::SQRT_2 / mean_dist
        }
    };
    let norm: Vec<[f64; 2]> = pts
        .iter()
        .map(|p| [(p[0] - cx) * scale, (p[1] - cy) * scale])
        .collect();
    let mut t = Array2::<f64>::zeros((3, 3));
    t[[0, 0]] = scale;
    t[[1, 1]] = scale;
    t[[0, 2]] = -cx * scale;
    t[[1, 2]] = -cy * scale;
    t[[2, 2]] = 1.0;
    (norm, t)
}

// ─────────────────────────────────────────────────────────────────────────────
// FundamentalMatrix
// ─────────────────────────────────────────────────────────────────────────────

/// Fundamental matrix estimation from point correspondences.
///
/// Supports the normalised 8-point algorithm and RANSAC robust fitting.
#[derive(Debug, Clone)]
pub struct FundamentalMatrix {
    /// The 3×3 fundamental matrix F such that x'^T F x = 0.
    pub matrix: Array2<f64>,
}

impl FundamentalMatrix {
    /// Estimate F using the normalised 8-point algorithm.
    ///
    /// `pts1` and `pts2` must contain at least 8 corresponding `[x, y]` points.
    pub fn from_eight_point(pts1: &[[f64; 2]], pts2: &[[f64; 2]]) -> Result<Self> {
        if pts1.len() < 8 || pts1.len() != pts2.len() {
            return Err(VisionError::InvalidParameter(
                "FundamentalMatrix requires at least 8 correspondences".to_string(),
            ));
        }
        let n = pts1.len();
        let (norm1, t1) = normalize_points(pts1);
        let (norm2, t2) = normalize_points(pts2);

        // Build the constraint matrix A (n×9)
        let mut a = Array2::<f64>::zeros((n, 9));
        for i in 0..n {
            let (x1, y1) = (norm1[i][0], norm1[i][1]);
            let (x2, y2) = (norm2[i][0], norm2[i][1]);
            a[[i, 0]] = x2 * x1;
            a[[i, 1]] = x2 * y1;
            a[[i, 2]] = x2;
            a[[i, 3]] = y2 * x1;
            a[[i, 4]] = y2 * y1;
            a[[i, 5]] = y2;
            a[[i, 6]] = x1;
            a[[i, 7]] = y1;
            a[[i, 8]] = 1.0;
        }
        let (_u, _s, vt) = svd_small(&a);
        // Last row of Vt is the null vector → reshape to 3×3
        let mut f_norm = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                f_norm[[i, j]] = vt[[vt.nrows() - 1, i * 3 + j]];
            }
        }
        // Enforce rank-2 constraint
        let f_rank2 = enforce_rank2(&f_norm);
        // Denormalize: F = T2^T * F_norm * T1
        let t2t = mat3_t(&t2);
        let matrix = mat3_mul(&mat3_mul(&t2t, &f_rank2), &t1);
        Ok(Self { matrix })
    }

    /// RANSAC-robust fundamental matrix estimation.
    ///
    /// - `threshold`: inlier reprojection threshold in pixels.
    /// - `confidence`: desired probability that the result has no outliers (e.g. 0.99).
    /// - `max_iterations`: upper bound on RANSAC iterations.
    pub fn from_ransac(
        pts1: &[[f64; 2]],
        pts2: &[[f64; 2]],
        threshold: f64,
        confidence: f64,
        max_iterations: usize,
    ) -> Result<(Self, Vec<bool>)> {
        if pts1.len() < 8 || pts1.len() != pts2.len() {
            return Err(VisionError::InvalidParameter(
                "RANSAC fundamental matrix requires at least 8 correspondences".to_string(),
            ));
        }
        let n = pts1.len();
        let mut best_inliers = vec![false; n];
        let mut best_count = 0usize;
        let mut rng_state: u64 = 12345;

        // Adaptive RANSAC: update max_iter based on inlier fraction
        let mut iterations = max_iterations;
        let mut iter = 0usize;

        while iter < iterations {
            // Sample 8 random indices
            let mut sample = Vec::with_capacity(8);
            while sample.len() < 8 {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let idx = ((rng_state >> 33) as usize) % n;
                if !sample.contains(&idx) {
                    sample.push(idx);
                }
            }
            let s1: Vec<[f64; 2]> = sample.iter().map(|&i| pts1[i]).collect();
            let s2: Vec<[f64; 2]> = sample.iter().map(|&i| pts2[i]).collect();
            let f_candidate = match FundamentalMatrix::from_eight_point(&s1, &s2) {
                Ok(f) => f,
                Err(_) => {
                    iter += 1;
                    continue;
                }
            };
            // Count inliers via Sampson distance
            let inliers: Vec<bool> = (0..n)
                .map(|i| f_candidate.sampson_distance(&pts1[i], &pts2[i]) < threshold * threshold)
                .collect();
            let count = inliers.iter().filter(|&&b| b).count();
            if count > best_count {
                best_count = count;
                best_inliers = inliers;
                // Update adaptive iterations
                if best_count < n {
                    let w = best_count as f64 / n as f64;
                    if w > 1.0 - 1e-10 {
                        iterations = 1;
                    } else {
                        let log_c = (1.0 - confidence).ln();
                        let log_w = (1.0 - w.powi(8)).ln();
                        if log_w.abs() > 1e-14 {
                            let new_iters = (log_c / log_w).ceil() as usize;
                            iterations = new_iters.min(max_iterations);
                        }
                    }
                }
            }
            iter += 1;
        }

        if best_count < 8 {
            return Err(VisionError::OperationError(
                "RANSAC found fewer than 8 inliers".to_string(),
            ));
        }

        // Re-estimate with all inliers
        let in1: Vec<[f64; 2]> = (0..n)
            .filter(|&i| best_inliers[i])
            .map(|i| pts1[i])
            .collect();
        let in2: Vec<[f64; 2]> = (0..n)
            .filter(|&i| best_inliers[i])
            .map(|i| pts2[i])
            .collect();
        let f_final = FundamentalMatrix::from_eight_point(&in1, &in2)?;
        Ok((f_final, best_inliers))
    }

    /// Sampson distance for a point correspondence.
    pub fn sampson_distance(&self, p1: &[f64; 2], p2: &[f64; 2]) -> f64 {
        let x1 = Array1::from(vec![p1[0], p1[1], 1.0]);
        let x2 = Array1::from(vec![p2[0], p2[1], 1.0]);
        let fx = mat3_vec3(&self.matrix, &x1);
        let ftx = mat3_vec3(&mat3_t(&self.matrix), &x2);
        let num = dot1(&x2.view(), &fx.view());
        let denom = fx[0] * fx[0] + fx[1] * fx[1] + ftx[0] * ftx[0] + ftx[1] * ftx[1];
        if denom.abs() < 1e-14 {
            f64::MAX
        } else {
            (num * num) / denom
        }
    }
}

/// Enforce rank-2 constraint on a 3×3 matrix via SVD zeroing.
fn enforce_rank2(m: &Array2<f64>) -> Array2<f64> {
    let (u, mut s, vt) = svd_small(m);
    // Zero the smallest singular value
    s[2] = 0.0;
    // Reconstruct: U * diag(s) * Vt
    let mut result = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                result[[i, j]] += u[[i, k]] * s[k] * vt[[k, j]];
            }
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// EssentialMatrix
// ─────────────────────────────────────────────────────────────────────────────

/// Camera intrinsic matrix (simplified).
#[derive(Debug, Clone)]
pub struct IntrinsicMatrix {
    /// Focal length x
    pub fx: f64,
    /// Focal length y
    pub fy: f64,
    /// Principal point x
    pub cx: f64,
    /// Principal point y
    pub cy: f64,
}

impl IntrinsicMatrix {
    /// Build the 3×3 K matrix.
    pub fn to_matrix(&self) -> Array2<f64> {
        let mut k = Array2::<f64>::zeros((3, 3));
        k[[0, 0]] = self.fx;
        k[[1, 1]] = self.fy;
        k[[0, 2]] = self.cx;
        k[[1, 2]] = self.cy;
        k[[2, 2]] = 1.0;
        k
    }

    /// Inverse of K.
    pub fn to_inverse(&self) -> Array2<f64> {
        let mut ki = Array2::<f64>::zeros((3, 3));
        ki[[0, 0]] = 1.0 / self.fx;
        ki[[1, 1]] = 1.0 / self.fy;
        ki[[0, 2]] = -self.cx / self.fx;
        ki[[1, 2]] = -self.cy / self.fy;
        ki[[2, 2]] = 1.0;
        ki
    }
}

/// Essential matrix and its decomposition into rotation + translation.
#[derive(Debug, Clone)]
pub struct EssentialMatrix {
    /// The 3×3 essential matrix E.
    pub matrix: Array2<f64>,
}

impl EssentialMatrix {
    /// Compute E from a fundamental matrix and camera intrinsics.
    /// E = K2^T * F * K1
    pub fn from_fundamental(
        f: &FundamentalMatrix,
        k1: &IntrinsicMatrix,
        k2: &IntrinsicMatrix,
    ) -> Self {
        let k1m = k1.to_matrix();
        let k2m = k2.to_matrix();
        let k2t = mat3_t(&k2m);
        let matrix = mat3_mul(&mat3_mul(&k2t, &f.matrix), &k1m);
        Self { matrix }
    }

    /// Estimate E directly from normalised image coordinates (5-point algorithm, simplified).
    ///
    /// This implementation uses the normalised 8-point solver on calibrated coordinates
    /// as a practical alternative to the full 5-point Nistér algorithm.
    pub fn from_five_point(pts1_norm: &[[f64; 2]], pts2_norm: &[[f64; 2]]) -> Result<Self> {
        if pts1_norm.len() < 5 || pts1_norm.len() != pts2_norm.len() {
            return Err(VisionError::InvalidParameter(
                "EssentialMatrix requires at least 5 calibrated correspondences".to_string(),
            ));
        }
        // Use 8-point solver on normalised coordinates, then enforce E constraints
        let f = FundamentalMatrix::from_eight_point(pts1_norm, pts2_norm)?;
        let (u, mut s, vt) = svd_small(&f.matrix);
        // Essential matrix has two equal singular values; set both to mean, zero third
        let mean_s = (s[0] + s[1]) * 0.5;
        s[0] = mean_s;
        s[1] = mean_s;
        s[2] = 0.0;
        let mut matrix = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    matrix[[i, j]] += u[[i, k]] * s[k] * vt[[k, j]];
                }
            }
        }
        Ok(Self { matrix })
    }

    /// Decompose E into the four candidate (R, t) pairs.
    ///
    /// Returns a list of `(R, t)` pairs where R is a 3×3 rotation matrix
    /// and t is a unit 3-vector.
    pub fn decompose(&self) -> Vec<(Array2<f64>, Array1<f64>)> {
        let (u, _s, vt) = svd_small(&self.matrix);
        // W = [[0,-1,0],[1,0,0],[0,0,1]]
        let mut w = Array2::<f64>::zeros((3, 3));
        w[[0, 1]] = -1.0;
        w[[1, 0]] = 1.0;
        w[[2, 2]] = 1.0;
        let wt = mat3_t(&w);

        let u_det = det3(&u);
        let vt_det = det3(&vt);
        // Ensure proper rotation (det = 1)
        let sign_u = if u_det < 0.0 { -1.0 } else { 1.0 };
        let sign_v = if vt_det < 0.0 { -1.0 } else { 1.0 };

        let mut u_fix = u.clone();
        let mut vt_fix = vt.clone();
        if u_det < 0.0 {
            for val in u_fix.iter_mut() {
                *val *= sign_u;
            }
        }
        if vt_det < 0.0 {
            for val in vt_fix.iter_mut() {
                *val *= sign_v;
            }
        }

        // R1 = U * W * Vt,  R2 = U * W^T * Vt
        let r1 = mat3_mul(&mat3_mul(&u_fix, &w), &vt_fix);
        let r2 = mat3_mul(&mat3_mul(&u_fix, &wt), &vt_fix);
        // t = last column of U (translation up to sign)
        let t_pos = Array1::from(vec![u_fix[[0, 2]], u_fix[[1, 2]], u_fix[[2, 2]]]);
        let t_neg = Array1::from(vec![-u_fix[[0, 2]], -u_fix[[1, 2]], -u_fix[[2, 2]]]);

        vec![
            (r1.clone(), t_pos.clone()),
            (r1, t_neg.clone()),
            (r2.clone(), t_pos),
            (r2, t_neg),
        ]
    }

    /// Choose the correct (R, t) pair by the cheirality check.
    ///
    /// - `pts1_norm`, `pts2_norm`: normalised calibrated coordinates.
    ///
    /// Returns the pair where the most points triangulate with positive depth.
    pub fn recover_pose(
        &self,
        pts1_norm: &[[f64; 2]],
        pts2_norm: &[[f64; 2]],
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let candidates = self.decompose();
        let mut best_idx = 0usize;
        let mut best_count = 0usize;

        for (idx, (r, t)) in candidates.iter().enumerate() {
            let count = count_positive_depth(pts1_norm, pts2_norm, r, t);
            if count > best_count {
                best_count = count;
                best_idx = idx;
            }
        }
        let (r, t) = &candidates[best_idx];
        Ok((r.clone(), t.clone()))
    }
}

/// 3×3 determinant.
fn det3(m: &Array2<f64>) -> f64 {
    m[[0, 0]] * (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]])
        - m[[0, 1]] * (m[[1, 0]] * m[[2, 2]] - m[[1, 2]] * m[[2, 0]])
        + m[[0, 2]] * (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]])
}

/// Count correspondences with positive depth under (R, t) via DLT triangulation.
fn count_positive_depth(
    pts1: &[[f64; 2]],
    pts2: &[[f64; 2]],
    r: &Array2<f64>,
    t: &Array1<f64>,
) -> usize {
    // Camera 1: P1 = [I | 0], Camera 2: P2 = [R | t]
    let p1 = build_projection_matrix(&Array2::eye(3), &Array1::zeros(3));
    let p2 = build_projection_matrix(r, t);
    let n = pts1.len();
    let mut count = 0usize;
    for i in 0..n {
        if let Ok(pt) = triangulate_dlt_single(&pts1[i], &pts2[i], &p1, &p2) {
            // Check positive depth in both cameras
            let d1 = pt[2];
            let rt_pt = mat3_vec3(r, &pt);
            let d2 = rt_pt[2] + t[2];
            if d1 > 0.0 && d2 > 0.0 {
                count += 1;
            }
        }
    }
    count
}

/// Build a 3×4 projection matrix from R (3×3) and t (3-vector).
pub fn build_projection_matrix(r: &Array2<f64>, t: &Array1<f64>) -> Array2<f64> {
    let mut p = Array2::<f64>::zeros((3, 4));
    for i in 0..3 {
        for j in 0..3 {
            p[[i, j]] = r[[i, j]];
        }
        p[[i, 3]] = t[i];
    }
    p
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangulation
// ─────────────────────────────────────────────────────────────────────────────

/// Triangulation algorithms for recovering 3D points from 2D observations.
pub struct Triangulation;

impl Triangulation {
    /// Linear triangulation (DLT) for a batch of correspondences.
    ///
    /// - `pts1`, `pts2`: arrays of `[x, y]` image coordinates.
    /// - `p1`, `p2`: 3×4 projection matrices.
    ///
    /// Returns a PointCloud with the triangulated 3D points.
    pub fn linear(
        pts1: &[[f64; 2]],
        pts2: &[[f64; 2]],
        p1: &ArrayView2<f64>,
        p2: &ArrayView2<f64>,
    ) -> Result<PointCloud> {
        if pts1.len() != pts2.len() {
            return Err(VisionError::DimensionMismatch(
                "Triangulation: point arrays must have the same length".to_string(),
            ));
        }
        let n = pts1.len();
        let mut pts3d = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let pt = triangulate_dlt_single(&pts1[i], &pts2[i], &p1.to_owned(), &p2.to_owned())?;
            pts3d[[i, 0]] = pt[0];
            pts3d[[i, 1]] = pt[1];
            pts3d[[i, 2]] = pt[2];
        }
        Ok(PointCloud {
            points: pts3d,
            colors: None,
            confidence: None,
        })
    }

    /// Optimal triangulation correcting for epipolar constraint.
    ///
    /// Implements the Hartley–Sturm correction then DLT.
    pub fn optimal(
        pts1: &[[f64; 2]],
        pts2: &[[f64; 2]],
        f: &FundamentalMatrix,
        p1: &ArrayView2<f64>,
        p2: &ArrayView2<f64>,
    ) -> Result<PointCloud> {
        if pts1.len() != pts2.len() {
            return Err(VisionError::DimensionMismatch(
                "Optimal triangulation: point arrays must have the same length".to_string(),
            ));
        }
        let mut corrected1 = pts1.to_vec();
        let mut corrected2 = pts2.to_vec();
        for i in 0..pts1.len() {
            let (c1, c2) = correct_correspondences(&pts1[i], &pts2[i], f);
            corrected1[i] = c1;
            corrected2[i] = c2;
        }
        Self::linear(&corrected1, &corrected2, p1, p2)
    }
}

/// DLT triangulation for a single correspondence.
pub fn triangulate_dlt_single(
    p1_img: &[f64; 2],
    p2_img: &[f64; 2],
    proj1: &Array2<f64>,
    proj2: &Array2<f64>,
) -> Result<Array1<f64>> {
    // Build 4×4 system A * X = 0
    let mut a = Array2::<f64>::zeros((4, 4));
    for j in 0..4 {
        a[[0, j]] = p1_img[0] * proj1[[2, j]] - proj1[[0, j]];
        a[[1, j]] = p1_img[1] * proj1[[2, j]] - proj1[[1, j]];
        a[[2, j]] = p2_img[0] * proj2[[2, j]] - proj2[[0, j]];
        a[[3, j]] = p2_img[1] * proj2[[2, j]] - proj2[[1, j]];
    }
    // Solve via SVD: null vector of A
    let (_u, _s, vt) = svd_small(&a);
    let last_row = vt.nrows() - 1;
    let w = vt[[last_row, 3]];
    if w.abs() < 1e-14 {
        return Err(VisionError::OperationError(
            "DLT triangulation: degenerate configuration".to_string(),
        ));
    }
    let x = vt[[last_row, 0]] / w;
    let y = vt[[last_row, 1]] / w;
    let z = vt[[last_row, 2]] / w;
    Ok(Array1::from(vec![x, y, z]))
}

/// Correct a point correspondence to lie on the epipolar line (Hartley-Sturm, first-order).
fn correct_correspondences(
    p1: &[f64; 2],
    p2: &[f64; 2],
    f: &FundamentalMatrix,
) -> ([f64; 2], [f64; 2]) {
    let x1 = Array1::from(vec![p1[0], p1[1], 1.0]);
    let x2 = Array1::from(vec![p2[0], p2[1], 1.0]);
    // Epipolar lines
    let l2 = mat3_vec3(&f.matrix, &x1);
    let l1 = mat3_vec3(&mat3_t(&f.matrix), &x2);
    // First-order correction: project p onto its epipolar line
    let lam1 = dot1(&l1.view(), &x1.view()) / (l1[0] * l1[0] + l1[1] * l1[1]).max(1e-14);
    let lam2 = dot1(&l2.view(), &x2.view()) / (l2[0] * l2[0] + l2[1] * l2[1]).max(1e-14);
    (
        [p1[0] - lam1 * l1[0], p1[1] - lam1 * l1[1]],
        [p2[0] - lam2 * l2[0], p2[1] - lam2 * l2[1]],
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// BundleAdjustment
// ─────────────────────────────────────────────────────────────────────────────

/// Bundle adjustment via Levenberg-Marquardt reprojection error minimization.
pub struct BundleAdjustment {
    /// Maximum number of LM iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on the update norm.
    pub tolerance: f64,
    /// Initial damping factor λ.
    pub lambda: f64,
}

impl Default for BundleAdjustment {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
            lambda: 1e-3,
        }
    }
}

/// A single camera with rotation vector (Rodrigues) and translation.
#[derive(Debug, Clone)]
pub struct Camera {
    /// Rodrigues rotation vector (3 components).
    pub rvec: Array1<f64>,
    /// Translation vector (3 components).
    pub tvec: Array1<f64>,
    /// Camera intrinsics.
    pub intrinsics: IntrinsicMatrix,
}

impl Camera {
    /// Project a 3D world point to image coordinates.
    pub fn project(&self, pt3d: &Array1<f64>) -> [f64; 2] {
        let r = rodrigues_to_rotation(&self.rvec);
        let pt_cam = mat3_vec3(&r, pt3d);
        let xc = pt_cam[0] + self.tvec[0];
        let yc = pt_cam[1] + self.tvec[1];
        let zc = pt_cam[2] + self.tvec[2];
        if zc.abs() < 1e-14 {
            return [f64::NAN, f64::NAN];
        }
        let x_norm = xc / zc;
        let y_norm = yc / zc;
        [
            self.intrinsics.fx * x_norm + self.intrinsics.cx,
            self.intrinsics.fy * y_norm + self.intrinsics.cy,
        ]
    }
}

/// An observation of a 3D point in a camera.
#[derive(Debug, Clone)]
pub struct Observation {
    /// Camera index.
    pub camera_idx: usize,
    /// 3D point index.
    pub point_idx: usize,
    /// Observed 2D image coordinate.
    pub observed: [f64; 2],
}

impl BundleAdjustment {
    /// Run bundle adjustment on cameras and 3D points.
    ///
    /// Minimises the sum of squared reprojection errors over all observations
    /// using the Levenberg-Marquardt algorithm.
    ///
    /// Returns `(cameras, points)` after optimisation.
    pub fn run(
        &self,
        mut cameras: Vec<Camera>,
        mut points: Vec<Array1<f64>>,
        observations: &[Observation],
    ) -> Result<(Vec<Camera>, Vec<Array1<f64>>)> {
        // Parameter count: 6 per camera (rvec 3 + tvec 3) + 3 per point
        let nc = cameras.len();
        let np = points.len();
        let total_params = nc * 6 + np * 3;
        let num_obs = observations.len();

        if num_obs < 1 {
            return Err(VisionError::InvalidParameter(
                "BundleAdjustment requires at least one observation".to_string(),
            ));
        }

        let mut lambda = self.lambda;

        for _iter in 0..self.max_iterations {
            // Compute residuals and Jacobian (sparse approximation via numerical diff)
            let residuals = compute_residuals(&cameras, &points, observations);
            let cost: f64 = residuals.iter().map(|r| r * r).sum::<f64>();

            // Numerical Jacobian (simplified – full sparse BA is complex)
            let eps = 1e-6;
            let mut j = Array2::<f64>::zeros((num_obs * 2, total_params));
            // Perturb camera parameters
            for ci in 0..nc {
                for pi in 0..6 {
                    let param_idx = ci * 6 + pi;
                    let orig = if pi < 3 {
                        cameras[ci].rvec[pi]
                    } else {
                        cameras[ci].tvec[pi - 3]
                    };
                    if pi < 3 {
                        cameras[ci].rvec[pi] += eps;
                    } else {
                        cameras[ci].tvec[pi - 3] += eps;
                    }
                    let res_p = compute_residuals(&cameras, &points, observations);
                    if pi < 3 {
                        cameras[ci].rvec[pi] = orig;
                    } else {
                        cameras[ci].tvec[pi - 3] = orig;
                    }
                    for row in 0..(num_obs * 2) {
                        j[[row, param_idx]] = (res_p[row] - residuals[row]) / eps;
                    }
                }
            }
            // Perturb 3D point parameters
            for pi in 0..np {
                for coord in 0..3 {
                    let param_idx = nc * 6 + pi * 3 + coord;
                    let orig = points[pi][coord];
                    points[pi][coord] += eps;
                    let res_p = compute_residuals(&cameras, &points, observations);
                    points[pi][coord] = orig;
                    for row in 0..(num_obs * 2) {
                        j[[row, param_idx]] = (res_p[row] - residuals[row]) / eps;
                    }
                }
            }

            // Normal equations: (J^T J + lambda * I) delta = -J^T r
            let jt_j = compute_jtj(&j);
            let jt_r = compute_jtr(&j, &residuals);
            let delta = solve_normal_equations(&jt_j, &jt_r, lambda, total_params)?;

            // Test update
            let (cams_new, pts_new) = apply_delta(&cameras, &points, &delta, nc, np);
            let res_new = compute_residuals(&cams_new, &pts_new, observations);
            let cost_new: f64 = res_new.iter().map(|r| r * r).sum::<f64>();

            if cost_new < cost {
                cameras = cams_new;
                points = pts_new;
                lambda /= 10.0;
            } else {
                lambda *= 10.0;
            }

            // Convergence check
            let delta_norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
            if delta_norm < self.tolerance {
                break;
            }
        }
        Ok((cameras, points))
    }
}

/// Convert a Rodrigues rotation vector to a 3×3 rotation matrix.
///
/// The angle of rotation is the norm of `rvec`; the axis is `rvec / ‖rvec‖`.
pub fn rodrigues_to_rotation(rvec: &Array1<f64>) -> Array2<f64> {
    let theta = norm1(&rvec.view());
    if theta < 1e-14 {
        return Array2::eye(3);
    }
    let axis = rvec.mapv(|v| v / theta);
    let kx = axis[0];
    let ky = axis[1];
    let kz = axis[2];
    let c = theta.cos();
    let s = theta.sin();
    let t = 1.0 - c;
    let mut r = Array2::<f64>::zeros((3, 3));
    r[[0, 0]] = t * kx * kx + c;
    r[[0, 1]] = t * kx * ky - s * kz;
    r[[0, 2]] = t * kx * kz + s * ky;
    r[[1, 0]] = t * ky * kx + s * kz;
    r[[1, 1]] = t * ky * ky + c;
    r[[1, 2]] = t * ky * kz - s * kx;
    r[[2, 0]] = t * kz * kx - s * ky;
    r[[2, 1]] = t * kz * ky + s * kx;
    r[[2, 2]] = t * kz * kz + c;
    r
}

fn compute_residuals(
    cameras: &[Camera],
    points: &[Array1<f64>],
    observations: &[Observation],
) -> Vec<f64> {
    let mut res = Vec::with_capacity(observations.len() * 2);
    for obs in observations {
        if obs.camera_idx < cameras.len() && obs.point_idx < points.len() {
            let proj = cameras[obs.camera_idx].project(&points[obs.point_idx]);
            res.push(proj[0] - obs.observed[0]);
            res.push(proj[1] - obs.observed[1]);
        } else {
            res.push(0.0);
            res.push(0.0);
        }
    }
    res
}

fn compute_jtj(j: &Array2<f64>) -> Array2<f64> {
    let n = j.ncols();
    let m = j.nrows();
    let mut jtj = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for k in 0..n {
            let mut val = 0.0f64;
            for r in 0..m {
                val += j[[r, i]] * j[[r, k]];
            }
            jtj[[i, k]] = val;
        }
    }
    jtj
}

fn compute_jtr(j: &Array2<f64>, r: &[f64]) -> Vec<f64> {
    let n = j.ncols();
    let m = j.nrows();
    let mut jtr = vec![0.0f64; n];
    for i in 0..n {
        for row in 0..m {
            jtr[i] += j[[row, i]] * r[row];
        }
    }
    jtr
}

fn solve_normal_equations(
    jtj: &Array2<f64>,
    jtr: &[f64],
    lambda: f64,
    n: usize,
) -> Result<Vec<f64>> {
    // (J^T J + lambda I) delta = -J^T r
    let mut a = jtj.clone();
    for i in 0..n {
        a[[i, i]] += lambda;
    }
    // Cholesky-like solve (Gaussian elimination)
    let b: Vec<f64> = jtr.iter().map(|v| -v).collect();
    gaussian_elimination(&a, &b)
}

fn gaussian_elimination(a: &Array2<f64>, b: &[f64]) -> Result<Vec<f64>> {
    let n = a.nrows();
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = (0..n).map(|j| a[[i, j]]).collect();
            row.push(b[i]);
            row
        })
        .collect();
    #[allow(clippy::needless_range_loop)]
    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-14 {
            // Singular – return zeros
            return Ok(vec![0.0; n]);
        }
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for k in col..(n + 1) {
                let val = aug[col][k];
                aug[row][k] -= factor * val;
            }
        }
    }
    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }
    Ok(x)
}

fn apply_delta(
    cameras: &[Camera],
    points: &[Array1<f64>],
    delta: &[f64],
    nc: usize,
    np: usize,
) -> (Vec<Camera>, Vec<Array1<f64>>) {
    let mut new_cameras = cameras.to_vec();
    let mut new_points = points.to_vec();
    #[allow(clippy::needless_range_loop)]
    for ci in 0..nc {
        let base = ci * 6;
        for k in 0..3 {
            new_cameras[ci].rvec[k] += delta[base + k];
            new_cameras[ci].tvec[k] += delta[base + 3 + k];
        }
    }
    #[allow(clippy::needless_range_loop)]
    for pi in 0..np {
        let base = nc * 6 + pi * 3;
        for k in 0..3 {
            new_points[pi][k] += delta[base + k];
        }
    }
    (new_cameras, new_points)
}

// ─────────────────────────────────────────────────────────────────────────────
// Track Management
// ─────────────────────────────────────────────────────────────────────────────

/// A feature track: the same scene point seen across multiple frames.
#[derive(Debug, Clone)]
pub struct Track {
    /// Unique track identifier.
    pub id: usize,
    /// (frame_index, [x, y]) observations.
    pub observations: Vec<(usize, [f64; 2])>,
    /// Reconstructed 3D point (if triangulated).
    pub point3d: Option<Array1<f64>>,
}

impl Track {
    /// Create a new track.
    pub fn new(id: usize) -> Self {
        Self {
            id,
            observations: Vec::new(),
            point3d: None,
        }
    }

    /// Add an observation.
    pub fn add_observation(&mut self, frame_idx: usize, pt: [f64; 2]) {
        self.observations.push((frame_idx, pt));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SFMPipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Result of the SfM pipeline.
#[derive(Debug, Clone)]
pub struct SFMResult {
    /// Reconstructed cameras (one per frame).
    pub cameras: Vec<Camera>,
    /// Reconstructed 3D point cloud.
    pub point_cloud: PointCloud,
    /// Feature tracks.
    pub tracks: Vec<Track>,
}

/// Incremental Structure-from-Motion pipeline.
///
/// Processes a sequence of frames, each described by a set of keypoints and
/// descriptor matches, and reconstructs camera poses + 3D structure.
pub struct SFMPipeline {
    /// Camera intrinsics (assumed fixed).
    pub intrinsics: IntrinsicMatrix,
    /// Bundle adjustment configuration.
    pub bundle_adjustment: BundleAdjustment,
    /// RANSAC threshold for F/E estimation (pixels).
    pub ransac_threshold: f64,
    /// Minimum number of inlier matches to accept a frame pair.
    pub min_inliers: usize,
    /// Internal cameras accumulated so far.
    cameras: Vec<Camera>,
    /// Internal tracks.
    tracks: Vec<Track>,
    /// Next track ID.
    next_track_id: usize,
    /// Accumulated 3D points.
    points3d: Vec<Array1<f64>>,
}

impl SFMPipeline {
    /// Create a new SFM pipeline.
    pub fn new(intrinsics: IntrinsicMatrix) -> Self {
        Self {
            intrinsics,
            bundle_adjustment: BundleAdjustment::default(),
            ransac_threshold: 2.0,
            min_inliers: 20,
            cameras: Vec::new(),
            tracks: Vec::new(),
            next_track_id: 0,
            points3d: Vec::new(),
        }
    }

    /// Process a new frame given feature points and matches with the previous frame.
    ///
    /// `keypoints`: detected `[x, y]` keypoints in this frame.
    /// `matches`: pairs `(prev_kp_idx, curr_kp_idx)` of matched keypoints with the previous frame.
    /// Returns the estimated camera pose if successful.
    pub fn add_frame(
        &mut self,
        keypoints: &[[f64; 2]],
        matches: &[(usize, usize)],
        prev_keypoints: Option<&[[f64; 2]]>,
    ) -> Result<Option<Camera>> {
        if self.cameras.is_empty() {
            // First frame: canonical pose
            let cam = Camera {
                rvec: Array1::zeros(3),
                tvec: Array1::zeros(3),
                intrinsics: self.intrinsics.clone(),
            };
            self.cameras.push(cam.clone());
            // Create tracks for all keypoints
            for (i, kp) in keypoints.iter().enumerate() {
                let mut track = Track::new(self.next_track_id);
                self.next_track_id += 1;
                track.add_observation(0, *kp);
                // Store track index in matches later
                let _ = i; // silence unused warning
                self.tracks.push(track);
            }
            return Ok(Some(cam));
        }

        let prev_kps = match prev_keypoints {
            Some(p) => p,
            None => return Ok(None),
        };

        if matches.len() < self.min_inliers {
            return Err(VisionError::OperationError(format!(
                "SFMPipeline: not enough matches ({} < {})",
                matches.len(),
                self.min_inliers
            )));
        }

        // Collect matched point pairs in normalised coordinates
        let ki = self.intrinsics.to_inverse();
        let normalise = |p: &[f64; 2]| -> [f64; 2] {
            let v = Array1::from(vec![p[0], p[1], 1.0]);
            let n = mat3_vec3(&ki, &v);
            [n[0] / n[2].max(1e-14), n[1] / n[2].max(1e-14)]
        };

        let pts1_norm: Vec<[f64; 2]> = matches
            .iter()
            .map(|&(pi, _)| normalise(&prev_kps[pi]))
            .collect();
        let pts2_norm: Vec<[f64; 2]> = matches
            .iter()
            .map(|&(_, ci)| normalise(&keypoints[ci]))
            .collect();
        let pts1_px: Vec<[f64; 2]> = matches.iter().map(|&(pi, _)| prev_kps[pi]).collect();
        let pts2_px: Vec<[f64; 2]> = matches.iter().map(|&(_, ci)| keypoints[ci]).collect();

        // Estimate essential matrix with RANSAC
        let (f, inliers) =
            FundamentalMatrix::from_ransac(&pts1_px, &pts2_px, self.ransac_threshold, 0.99, 1000)?;
        let e = EssentialMatrix::from_fundamental(&f, &self.intrinsics, &self.intrinsics);

        // Filter to inlier normalised points
        let pts1_in: Vec<[f64; 2]> = inliers
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| pts1_norm[i])
            .collect();
        let pts2_in: Vec<[f64; 2]> = inliers
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| pts2_norm[i])
            .collect();

        let (r, t) = e.recover_pose(&pts1_in, &pts2_in)?;

        let frame_idx = self.cameras.len();
        let rvec = rotation_to_rodrigues(&r);
        let cam = Camera {
            rvec,
            tvec: t,
            intrinsics: self.intrinsics.clone(),
        };
        self.cameras.push(cam.clone());

        // Triangulate inlier points
        let prev_cam = &self.cameras[frame_idx - 1];
        let p1 = {
            let r_prev = rodrigues_to_rotation(&prev_cam.rvec);
            build_projection_matrix(&r_prev, &prev_cam.tvec)
        };
        let r_curr = rodrigues_to_rotation(&cam.rvec);
        let p2 = build_projection_matrix(&r_curr, &cam.tvec);

        let cloud = Triangulation::linear(&pts1_in, &pts2_in, &p1.view(), &p2.view())?;

        // Add new tracks
        for (i, (&(pi, ci), &inlier)) in matches.iter().zip(inliers.iter()).enumerate() {
            if inlier {
                let mut track = Track::new(self.next_track_id);
                self.next_track_id += 1;
                track.add_observation(frame_idx - 1, prev_kps[pi]);
                track.add_observation(frame_idx, keypoints[ci]);
                let pt_idx = i; // rough mapping
                if pt_idx < cloud.points.nrows() {
                    let pt = Array1::from(vec![
                        cloud.points[[pt_idx, 0]],
                        cloud.points[[pt_idx, 1]],
                        cloud.points[[pt_idx, 2]],
                    ]);
                    track.point3d = Some(pt.clone());
                    self.points3d.push(pt);
                }
                self.tracks.push(track);
            }
        }

        Ok(Some(cam))
    }

    /// Finalise the reconstruction.
    pub fn finalize(self) -> SFMResult {
        let n = self.points3d.len();
        let mut pts_arr = Array2::<f64>::zeros((n, 3));
        for (i, pt) in self.points3d.iter().enumerate() {
            pts_arr[[i, 0]] = pt[0];
            pts_arr[[i, 1]] = pt[1];
            pts_arr[[i, 2]] = pt[2];
        }
        SFMResult {
            cameras: self.cameras,
            point_cloud: PointCloud {
                points: pts_arr,
                colors: None,
                confidence: None,
            },
            tracks: self.tracks,
        }
    }
}

/// Convert a rotation matrix to a Rodrigues vector.
fn rotation_to_rodrigues(r: &Array2<f64>) -> Array1<f64> {
    // tr(R) = 1 + 2 cos(theta)
    let trace = r[[0, 0]] + r[[1, 1]] + r[[2, 2]];
    let cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    if theta.abs() < 1e-10 {
        return Array1::zeros(3);
    }
    let scale = theta / (2.0 * theta.sin());
    Array1::from(vec![
        (r[[2, 1]] - r[[1, 2]]) * scale,
        (r[[0, 2]] - r[[2, 0]]) * scale,
        (r[[1, 0]] - r[[0, 1]]) * scale,
    ])
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_correspondences(n: usize) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        // Simulated points on a plane with a small rotation
        let mut pts1 = Vec::with_capacity(n);
        let mut pts2 = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i as f64 * 37.0) % 300.0 + 50.0;
            let y = (i as f64 * 53.0) % 200.0 + 50.0;
            pts1.push([x, y]);
            // Small translation in x
            pts2.push([x + 20.0 + (i as f64 * 0.1), y + (i as f64 * 0.05)]);
        }
        (pts1, pts2)
    }

    #[test]
    fn test_fundamental_matrix_eight_point() {
        let (pts1, pts2) = synthetic_correspondences(15);
        let result = FundamentalMatrix::from_eight_point(&pts1, &pts2);
        assert!(result.is_ok(), "Eight-point algorithm should succeed");
        let f = result.expect("FundamentalMatrix::from_eight_point should succeed");
        assert_eq!(f.matrix.shape(), [3, 3]);
    }

    #[test]
    fn test_fundamental_matrix_ransac() {
        let (pts1, pts2) = synthetic_correspondences(30);
        let result = FundamentalMatrix::from_ransac(&pts1, &pts2, 5.0, 0.99, 100);
        assert!(result.is_ok(), "RANSAC F estimation should succeed");
    }

    #[test]
    fn test_point_cloud_empty() {
        let pc = PointCloud::empty();
        assert_eq!(pc.len(), 0);
        assert!(pc.is_empty());
    }

    #[test]
    fn test_point_cloud_from_vec() {
        let pts = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let pc = PointCloud::from_vec(pts);
        assert_eq!(pc.len(), 2);
        assert!((pc.points[[0, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangulation_dlt() {
        // Camera 1: P = [I | 0], Camera 2: P = [I | -1,0,0]
        let mut p1 = Array2::<f64>::zeros((3, 4));
        p1[[0, 0]] = 1.0;
        p1[[1, 1]] = 1.0;
        p1[[2, 2]] = 1.0;
        let mut p2 = p1.clone();
        p2[[0, 3]] = -1.0; // translate by -1 in x

        // A point at (0, 0, 5) projects to (0,0) in cam1 and (0.2, 0) in cam2
        let pt3d = [0.0f64, 0.0, 5.0];
        let proj1 = [
            (p1[[0, 0]] * pt3d[0] + p1[[0, 2]] * pt3d[2])
                / (p1[[2, 0]] * pt3d[0] + p1[[2, 2]] * pt3d[2]),
            0.0,
        ];
        let proj2 = [
            (p2[[0, 0]] * pt3d[0] + p2[[0, 3]]) / (p2[[2, 0]] * pt3d[0] + p2[[2, 2]] * pt3d[2]),
            0.0,
        ];
        let result = triangulate_dlt_single(&proj1, &proj2, &p1, &p2);
        assert!(result.is_ok());
    }
}
