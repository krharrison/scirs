//! Multi-camera extrinsic calibration.
//!
//! Provides:
//! - `dlt_extrinsics`    – Direct Linear Transform to recover `[R|t]` from N≥6
//!                         point correspondences.
//! - `svd_3x3`           – Jacobi SVD for 3 × 3 matrices.
//! - `orthogonalize_rotation` – Project an approximate matrix onto SO(3).
//! - `reprojection_error`     – Per-point reprojection residual in pixels.
//! - `ransac_extrinsics`      – RANSAC-robust extrinsic calibration.
//! - `calibrate_stereo`       – Calibrate a stereo pair and recover the
//!                              fundamental matrix.
//! - `bundle_adjust_single_step` – One Gauss-Newton update for a set of cameras
//!                                 and world points.

use crate::detection_3d::frustum::CameraIntrinsics;
use crate::error::{Result, VisionError};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for RANSAC-based extrinsic calibration.
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Number of RANSAC iterations.
    pub n_ransac_iters: usize,
    /// Reprojection-error threshold for inlier classification (pixels).
    pub reproj_threshold: f64,
    /// Maximum acceptable reprojection error for a "good" calibration.
    pub max_reproj_error: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            n_ransac_iters: 1000,
            reproj_threshold: 2.0,
            max_reproj_error: 5.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera struct
// ─────────────────────────────────────────────────────────────────────────────

/// A calibrated camera with intrinsic and extrinsic parameters.
#[derive(Debug, Clone)]
pub struct Camera {
    /// Intrinsic parameters (focal lengths, principal point).
    pub intrinsics: CameraIntrinsics,
    /// Rotation matrix R (3 × 3, world-to-camera).
    pub rotation: [[f64; 3]; 3],
    /// Translation vector t (camera centre = -R^T t).
    pub translation: [f64; 3],
}

impl Camera {
    /// Identity pose (R = I, t = 0).
    pub fn identity(intrinsics: CameraIntrinsics) -> Self {
        Self {
            intrinsics,
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Calibration target
// ─────────────────────────────────────────────────────────────────────────────

/// Corresponding 2-D / 3-D point pairs for a single view.
#[derive(Debug, Clone)]
pub struct CalibrationTarget {
    /// 3-D world points.
    pub world_pts: Vec<[f64; 3]>,
    /// Corresponding 2-D image points (pixel coordinates).
    pub image_pts: Vec<[f64; 2]>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Stereo pair
// ─────────────────────────────────────────────────────────────────────────────

/// Calibrated stereo camera pair.
#[derive(Debug, Clone)]
pub struct StereoPair {
    /// First (left) camera.
    pub cam1: Camera,
    /// Second (right) camera.
    pub cam2: Camera,
    /// Fundamental matrix F such that x'^T F x = 0.
    pub fundamental_matrix: [[f64; 3]; 3],
}

// ─────────────────────────────────────────────────────────────────────────────
// 3 × 3 matrix helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix–vector product for 3 × 3.
fn mat3_mul_vec3(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Matrix–matrix product for 3 × 3.
fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Transpose of a 3 × 3 matrix.
fn mat3_transpose(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Determinant of a 3 × 3 matrix.
fn mat3_det(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Inverse of a 3 × 3 matrix.  Returns an error if the matrix is singular.
fn mat3_inv(m: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let det = mat3_det(m);
    if det.abs() < 1e-12 {
        return Err(VisionError::LinAlgError("Matrix is singular".into()));
    }
    let inv_det = 1.0 / det;
    Ok([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

// ─────────────────────────────────────────────────────────────────────────────
// Jacobi SVD for 3 × 3
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Jacobi SVD of a 3 × 3 matrix `m`.
///
/// Returns `(U, S, Vt)` such that `m ≈ U · diag(S) · Vt` and both `U`, `Vt`
/// are orthogonal.  The singular values in `S` may not be sorted.
pub fn svd_3x3(m: [[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3], [[f64; 3]; 3]) {
    // One-sided Jacobi SVD: iteratively zero off-diagonal elements of A^T A.
    // We compute V from A^T A rotations, then U = A V diag(1/s).
    let max_iter = 100;
    let eps = 1e-14_f64;

    // Identity start for V
    let mut v: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut a = m; // working copy, columns will become U·diag(S)

    for _ in 0..max_iter {
        let mut converged = true;
        // Off-diagonal pairs (p,q)
        for p in 0..2 {
            for q in (p + 1)..3 {
                // Column dot products
                let app: f64 = (0..3).map(|i| a[i][p] * a[i][p]).sum();
                let aqq: f64 = (0..3).map(|i| a[i][q] * a[i][q]).sum();
                let apq: f64 = (0..3).map(|i| a[i][p] * a[i][q]).sum();

                if apq.abs() < eps * (app * aqq).sqrt() {
                    continue;
                }
                converged = false;

                // Jacobi rotation angle
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (tau - (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update columns p and q of A
                for i in 0..3 {
                    let ap = a[i][p];
                    let aq = a[i][q];
                    a[i][p] = c * ap - s * aq;
                    a[i][q] = s * ap + c * aq;
                }
                // Update V
                for i in 0..3 {
                    let vp = v[i][p];
                    let vq = v[i][q];
                    v[i][p] = c * vp - s * vq;
                    v[i][q] = s * vp + c * vq;
                }
            }
        }
        if converged {
            break;
        }
    }

    // Singular values = column norms of A
    let mut sigma = [0.0_f64; 3];
    for j in 0..3 {
        sigma[j] = (0..3).map(|i| a[i][j] * a[i][j]).sum::<f64>().sqrt();
    }

    // U columns = A columns / sigma
    let mut u = [[0.0_f64; 3]; 3];
    for j in 0..3 {
        if sigma[j] > eps {
            for i in 0..3 {
                u[i][j] = a[i][j] / sigma[j];
            }
        } else {
            // Arbitrary unit vector orthogonal to existing columns (simplified)
            u[j][j] = 1.0;
        }
    }

    let vt = mat3_transpose(&v);
    (u, sigma, vt)
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotation orthogonalization
// ─────────────────────────────────────────────────────────────────────────────

/// Project a matrix onto SO(3) (the nearest proper rotation matrix) via SVD.
///
/// Given an approximate rotation `R ≈ U Σ V^T` the nearest rotation is
/// `R_clean = U V^T` with its determinant forced to +1.
pub fn orthogonalize_rotation(r: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let (u, _s, vt) = svd_3x3(r);
    let mut rot = mat3_mul(&u, &vt);
    // Enforce det = +1
    if mat3_det(&rot) < 0.0 {
        // Flip sign of last column of U and recompute
        let mut u2 = u;
        for i in 0..3 {
            u2[i][2] = -u2[i][2];
        }
        rot = mat3_mul(&u2, &vt);
    }
    rot
}

// ─────────────────────────────────────────────────────────────────────────────
// Intrinsics matrix K
// ─────────────────────────────────────────────────────────────────────────────

/// Build the 3 × 3 intrinsic matrix K from camera parameters.
fn build_k(intr: &CameraIntrinsics) -> [[f64; 3]; 3] {
    [
        [intr.fx, 0.0, intr.cx],
        [0.0, intr.fy, intr.cy],
        [0.0, 0.0, 1.0],
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// DLT extrinsics
// ─────────────────────────────────────────────────────────────────────────────

/// Recover extrinsic parameters `(R, t)` from N ≥ 6 world–image correspondences
/// using the Direct Linear Transform.
///
/// # Algorithm
///
/// 1. Map image points to normalised camera coordinates via K^{-1}: this removes
///    the intrinsic scale so the DLT operates in a well-conditioned unit space.
/// 2. Build the 2N × 12 design matrix A using unnormalised world coordinates
///    (world normalisation introduces a scale ambiguity that is hard to invert
///    without knowing the true rotation scale; instead we rely on the
///    eigenvector-based null-space solver which is numerically robust for
///    typical scene scales).
/// 3. Compute A^T A (12 × 12) and find its smallest-eigenvalue eigenvector via
///    Rayleigh-quotient power iteration.
/// 4. Reshape the 12-vector into a 3 × 4 projection matrix P = [M | p4].
/// 5. Recover R via SVD orthogonalisation of M; extract t from p4 / ‖row3(M)‖.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] if fewer than 6 correspondences are
/// supplied, and [`VisionError::LinAlgError`] if the system is degenerate.
pub fn dlt_extrinsics(
    world_pts: &[[f64; 3]],
    image_pts: &[[f64; 2]],
    intrinsics: &CameraIntrinsics,
) -> Result<([[f64; 3]; 3], [f64; 3])> {
    let n = world_pts.len();
    if n < 6 {
        return Err(VisionError::InvalidParameter(
            "At least 6 correspondences required for DLT".into(),
        ));
    }
    if image_pts.len() != n {
        return Err(VisionError::InvalidParameter(
            "world_pts and image_pts must have the same length".into(),
        ));
    }

    let k = build_k(intrinsics);
    let k_inv = mat3_inv(&k)?;

    // Map image points to normalised camera rays: x_n = K^{-1} [u, v, 1]^T
    let mut norm_img: Vec<[f64; 2]> = Vec::with_capacity(n);
    for ip in image_pts {
        let h = mat3_mul_vec3(&k_inv, &[ip[0], ip[1], 1.0]);
        let w_h = h[2];
        if w_h.abs() < 1e-12 {
            return Err(VisionError::LinAlgError(
                "Degenerate image point homogenisation".into(),
            ));
        }
        norm_img.push([h[0] / w_h, h[1] / w_h]);
    }

    // Build A (2n × 12) — world points are used as-is.
    // For each correspondence (X = [Xw, Yw, Zw], x_n = [xn, yn]):
    //   row 2i:   [-Xw, -Yw, -Zw, -1,   0,   0,   0,  0,  xn*Xw, xn*Yw, xn*Zw, xn]
    //   row 2i+1: [  0,   0,   0,  0, -Xw, -Yw, -Zw, -1,  yn*Xw, yn*Yw, yn*Zw, yn]
    let rows = 2 * n;
    let cols = 12usize;
    let mut a = vec![0.0_f64; rows * cols];

    for i in 0..n {
        let xw = world_pts[i][0];
        let yw = world_pts[i][1];
        let zw = world_pts[i][2];
        let xn = norm_img[i][0];
        let yn = norm_img[i][1];

        // Row 2i
        let r = 2 * i;
        a[r * cols] = -xw;
        a[r * cols + 1] = -yw;
        a[r * cols + 2] = -zw;
        a[r * cols + 3] = -1.0;
        // cols 4..8 = 0
        a[r * cols + 8] = xn * xw;
        a[r * cols + 9] = xn * yw;
        a[r * cols + 10] = xn * zw;
        a[r * cols + 11] = xn;

        // Row 2i+1
        let r = 2 * i + 1;
        // cols 0..4 = 0
        a[r * cols + 4] = -xw;
        a[r * cols + 5] = -yw;
        a[r * cols + 6] = -zw;
        a[r * cols + 7] = -1.0;
        a[r * cols + 8] = yn * xw;
        a[r * cols + 9] = yn * yw;
        a[r * cols + 10] = yn * zw;
        a[r * cols + 11] = yn;
    }

    // Compute A^T A  (12 × 12, symmetric)
    let mut ata = vec![0.0_f64; cols * cols];
    for r in 0..rows {
        for j in 0..cols {
            for k2 in 0..cols {
                ata[j * cols + k2] += a[r * cols + j] * a[r * cols + k2];
            }
        }
    }

    // Solve for smallest-eigenvalue eigenvector of A^T A
    let p_vec = smallest_eigenvector(&ata, cols)?;

    // Reshape: p = [r1 | r2 | r3 | t] (each ri and t are 3-vectors)
    // p_vec[0..4]  = first row of 3×4 matrix P
    // p_vec[4..8]  = second row
    // p_vec[8..12] = third row
    let m00 = p_vec[0]; let m01 = p_vec[1]; let m02 = p_vec[2]; let m03 = p_vec[3];
    let m10 = p_vec[4]; let m11 = p_vec[5]; let m12 = p_vec[6]; let m13 = p_vec[7];
    let m20 = p_vec[8]; let m21 = p_vec[9]; let m22 = p_vec[10]; let m23 = p_vec[11];

    // Scale factor: ‖row3(M)‖ should equal 1 for a proper rotation.
    let row3_norm = (m20 * m20 + m21 * m21 + m22 * m22).sqrt();
    if row3_norm < 1e-12 {
        return Err(VisionError::LinAlgError(
            "Degenerate DLT solution (zero third row)".into(),
        ));
    }
    let inv_rn = 1.0 / row3_norm;

    // Approximate rotation (scaled) and translation
    let r_approx = [
        [m00 * inv_rn, m01 * inv_rn, m02 * inv_rn],
        [m10 * inv_rn, m11 * inv_rn, m12 * inv_rn],
        [m20 * inv_rn, m21 * inv_rn, m22 * inv_rn],
    ];
    let mut t_scaled = [m03 * inv_rn, m13 * inv_rn, m23 * inv_rn];

    // Project onto SO(3)
    let mut rot = orthogonalize_rotation(r_approx);

    // Enforce positive depth: if det(R) < 0, negate everything
    if mat3_det(&rot) < 0.0 {
        for i in 0..3 {
            for j in 0..3 {
                rot[i][j] = -rot[i][j];
            }
        }
        for v in &mut t_scaled {
            *v = -*v;
        }
    }

    // t_scaled is already in original world units because we did not
    // normalise the world coordinates.  The only remaining scale is inv_rn
    // (already applied above).
    Ok((rot, t_scaled))
}

/// Find the eigenvector of the symmetric `n × n` matrix `a` (stored row-major)
/// corresponding to its **smallest** eigenvalue.
///
/// Uses the one-sided Jacobi algorithm to fully diagonalise `a`, then returns
/// the eigenvector corresponding to the smallest diagonal element.
fn smallest_eigenvector(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let max_sweeps = 200;
    let eps = 1e-14_f64;

    // Working copy of a (lower/upper triangular — symmetric)
    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j]).collect())
        .collect();

    // Accumulate eigenvectors (start as identity)
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();

    for _ in 0..max_sweeps {
        // Off-diagonal Frobenius norm
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| i != j).map(move |j| (i, j)))
            .map(|(i, j)| m[i][j] * m[i][j])
            .sum::<f64>()
            .sqrt();

        if off < eps {
            break;
        }

        // Sweep over all off-diagonal pairs (p, q)
        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                if m[p][q].abs() < eps * (m[p][p].abs() + m[q][q].abs()) {
                    continue;
                }
                // Jacobi rotation
                let tau = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (tau - (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update M: apply Jacobi rotation from both sides
                let m_pp = m[p][p];
                let m_qq = m[q][q];
                let m_pq = m[p][q];
                m[p][p] = c * c * m_pp - 2.0 * s * c * m_pq + s * s * m_qq;
                m[q][q] = s * s * m_pp + 2.0 * s * c * m_pq + c * c * m_qq;
                m[p][q] = 0.0;
                m[q][p] = 0.0;
                for r in 0..n {
                    if r != p && r != q {
                        let m_rp = m[r][p];
                        let m_rq = m[r][q];
                        m[r][p] = c * m_rp - s * m_rq;
                        m[p][r] = m[r][p];
                        m[r][q] = s * m_rp + c * m_rq;
                        m[q][r] = m[r][q];
                    }
                }

                // Update eigenvector matrix
                for r in 0..n {
                    let v_rp = v[r][p];
                    let v_rq = v[r][q];
                    v[r][p] = c * v_rp - s * v_rq;
                    v[r][q] = s * v_rp + c * v_rq;
                }
            }
        }
    }

    // Find index of the smallest diagonal (eigenvalue)
    let min_idx = (0..n)
        .min_by(|&i, &j| m[i][i].partial_cmp(&m[j][j]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(n - 1);

    // Return the corresponding eigenvector column
    let mut ev: Vec<f64> = (0..n).map(|i| v[i][min_idx]).collect();

    let norm: f64 = ev.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-14 {
        return Err(VisionError::LinAlgError(
            "Degenerate DLT: null space computation failed".into(),
        ));
    }
    for x in &mut ev {
        *x /= norm;
    }
    Ok(ev)
}

/// Dense symmetric matrix–vector product (n × n stored row-major).
fn mat_vec(a: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| a[i * n + j] * v[j])
                .sum::<f64>()
        })
        .collect()
}

/// Rayleigh quotient  v^T A v / v^T v.
fn rayleigh_quotient(a: &[f64], v: &[f64], n: usize) -> f64 {
    let av = mat_vec(a, v, n);
    let num: f64 = av.iter().zip(v.iter()).map(|(av, vi)| av * vi).sum();
    let den: f64 = v.iter().map(|vi| vi * vi).sum();
    if den < 1e-15 {
        0.0
    } else {
        num / den
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reprojection error
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the reprojection error (Euclidean pixel distance) for one point.
///
/// Projects `world_pt` through `camera` and returns the distance to `image_pt`.
pub fn reprojection_error(camera: &Camera, world_pt: &[f64; 3], image_pt: &[f64; 2]) -> f64 {
    let r = &camera.rotation;
    let t = &camera.translation;

    // Camera coordinates
    let xc = r[0][0] * world_pt[0] + r[0][1] * world_pt[1] + r[0][2] * world_pt[2] + t[0];
    let yc = r[1][0] * world_pt[0] + r[1][1] * world_pt[1] + r[1][2] * world_pt[2] + t[1];
    let zc = r[2][0] * world_pt[0] + r[2][1] * world_pt[1] + r[2][2] * world_pt[2] + t[2];

    if zc.abs() < 1e-12 {
        return f64::INFINITY;
    }

    let intr = &camera.intrinsics;
    let u = intr.fx * (xc / zc) + intr.cx;
    let v = intr.fy * (yc / zc) + intr.cy;

    let du = u - image_pt[0];
    let dv = v - image_pt[1];
    (du * du + dv * dv).sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// LCG PRNG for RANSAC sampling
// ─────────────────────────────────────────────────────────────────────────────

struct RansacRng {
    state: u64,
}

impl RansacRng {
    fn new() -> Self {
        Self { state: 0xDEAD_BEEF_1234_5678 }
    }

    fn next_usize(&mut self, n: usize) -> usize {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 33) as usize) % n
    }

    /// Fisher-Yates sample of `k` distinct indices from `[0, n)`.
    fn sample_k(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + self.next_usize(n - i);
            pool.swap(i, j);
        }
        pool[..k].to_vec()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RANSAC extrinsics
// ─────────────────────────────────────────────────────────────────────────────

/// RANSAC-robust estimation of camera extrinsics.
///
/// Samples 6 correspondences per iteration, calls [`dlt_extrinsics`], and
/// counts inliers.  The camera with the most inliers is returned together with
/// the inlier indices.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] if fewer than 6 correspondences
/// are provided.
pub fn ransac_extrinsics(
    target: &CalibrationTarget,
    intrinsics: &CameraIntrinsics,
    config: &CalibrationConfig,
) -> Result<(Camera, Vec<usize>)> {
    let n = target.world_pts.len();
    if n < 6 {
        return Err(VisionError::InvalidParameter(
            "At least 6 correspondences required for RANSAC".into(),
        ));
    }

    let mut rng = RansacRng::new();
    let mut best_inliers: Vec<usize> = Vec::new();
    let mut best_camera: Option<Camera> = None;

    for _ in 0..config.n_ransac_iters {
        let sample = rng.sample_k(n, 6);
        let sw: Vec<[f64; 3]> = sample.iter().map(|&i| target.world_pts[i]).collect();
        let si: Vec<[f64; 2]> = sample.iter().map(|&i| target.image_pts[i]).collect();

        let (rot, t) = match dlt_extrinsics(&sw, &si, intrinsics) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let cam = Camera {
            intrinsics: intrinsics.clone(),
            rotation: rot,
            translation: t,
        };

        let inliers: Vec<usize> = (0..n)
            .filter(|&i| {
                reprojection_error(&cam, &target.world_pts[i], &target.image_pts[i])
                    < config.reproj_threshold
            })
            .collect();

        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_camera = Some(cam);
        }
    }

    match best_camera {
        Some(cam) => Ok((cam, best_inliers)),
        None => Err(VisionError::OperationError(
            "RANSAC failed to find a valid camera model".into(),
        )),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stereo calibration
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-product / skew-symmetric matrix of vector `v`.
fn skew_symmetric(v: &[f64; 3]) -> [[f64; 3]; 3] {
    [
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ]
}

/// Calibrate a stereo camera pair from two sets of corresponding targets.
///
/// Each camera is calibrated independently via RANSAC DLT.  The essential
/// matrix is then `E = [t]× R` (where `t = t2 - R2 R1^T t1`) and the
/// fundamental matrix is `F = K^{-T} E K^{-1}`.
///
/// # Errors
///
/// Propagates errors from [`ransac_extrinsics`].
pub fn calibrate_stereo(
    target1: &CalibrationTarget,
    target2: &CalibrationTarget,
    intrinsics: &CameraIntrinsics,
    config: &CalibrationConfig,
) -> Result<StereoPair> {
    let (cam1, _) = ransac_extrinsics(target1, intrinsics, config)?;
    let (cam2, _) = ransac_extrinsics(target2, intrinsics, config)?;

    // Relative rotation and translation from cam1 to cam2:
    //   R_rel = R2 * R1^T
    //   t_rel = t2 - R_rel * t1
    let r1t = mat3_transpose(&cam1.rotation);
    let r_rel = mat3_mul(&cam2.rotation, &r1t);
    let r1t_t1 = mat3_mul_vec3(&r_rel, &cam1.translation);
    let t_rel = [
        cam2.translation[0] - r1t_t1[0],
        cam2.translation[1] - r1t_t1[1],
        cam2.translation[2] - r1t_t1[2],
    ];

    // Essential matrix: E = [t]× R
    let t_cross = skew_symmetric(&t_rel);
    let essential = mat3_mul(&t_cross, &r_rel);

    // Fundamental matrix: F = K^{-T} E K^{-1}
    let k = build_k(intrinsics);
    let k_inv = mat3_inv(&k)?;
    let k_inv_t = mat3_transpose(&k_inv);
    let f = mat3_mul(&mat3_mul(&k_inv_t, &essential), &k_inv);

    Ok(StereoPair {
        cam1,
        cam2,
        fundamental_matrix: f,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Bundle adjustment (single Gauss-Newton step)
// ─────────────────────────────────────────────────────────────────────────────

/// Perform one Gauss-Newton update across all cameras and world points.
///
/// Each observation is a `(camera_idx, point_idx, [u, v])` tuple.
/// The function returns the total residual norm after the update.
///
/// This is a *simplified* bundle adjustment: it updates only the translation
/// vectors of the cameras using the reprojection residuals, leaving rotations
/// fixed.  A full BA would also update world-point positions and rotations.
pub fn bundle_adjust_single_step(
    cameras: &mut [Camera],
    observations: &[(usize, usize, [f64; 2])],
    world_pts: &[[f64; 3]],
) -> f64 {
    if cameras.is_empty() || observations.is_empty() || world_pts.is_empty() {
        return 0.0;
    }

    // Compute residuals and accumulate per-camera JtJ / Jtr for translation
    // (3 parameters per camera).
    let n_cams = cameras.len();
    let mut jtj = vec![[0.0_f64; 3]; 3 * n_cams]; // 3n_cams × 3 (block diagonal)
    let mut jtr = vec![0.0_f64; 3 * n_cams];
    let mut total_sq = 0.0_f64;

    for &(cam_idx, pt_idx, ref obs) in observations {
        if cam_idx >= n_cams || pt_idx >= world_pts.len() {
            continue;
        }
        let cam = &cameras[cam_idx];
        let wp = &world_pts[pt_idx];

        let r = &cam.rotation;
        let t = &cam.translation;
        let intr = &cam.intrinsics;

        let xc = r[0][0] * wp[0] + r[0][1] * wp[1] + r[0][2] * wp[2] + t[0];
        let yc = r[1][0] * wp[0] + r[1][1] * wp[1] + r[1][2] * wp[2] + t[1];
        let zc = r[2][0] * wp[0] + r[2][1] * wp[1] + r[2][2] * wp[2] + t[2];

        if zc.abs() < 1e-12 {
            continue;
        }
        let inv_z = 1.0 / zc;
        let inv_z2 = inv_z * inv_z;

        let u_pred = intr.fx * (xc * inv_z) + intr.cx;
        let v_pred = intr.fy * (yc * inv_z) + intr.cy;
        let ru = obs[0] - u_pred;
        let rv = obs[1] - v_pred;
        total_sq += ru * ru + rv * rv;

        // Jacobian of (u, v) wrt (tx, ty, tz):
        // du/dtx = fx / zc
        // du/dty = 0
        // du/dtz = -fx * xc / zc^2
        // dv/dtx = 0
        // dv/dty = fy / zc
        // dv/dtz = -fy * yc / zc^2
        let j_u = [intr.fx * inv_z, 0.0, -intr.fx * xc * inv_z2];
        let j_v = [0.0, intr.fy * inv_z, -intr.fy * yc * inv_z2];

        let base = cam_idx * 3;
        // Accumulate J^T J (3 × 3 per camera block)
        for a in 0..3 {
            for b in 0..3 {
                jtj[base + a][b] += j_u[a] * j_u[b] + j_v[a] * j_v[b];
            }
            jtr[base + a] += j_u[a] * ru + j_v[a] * rv;
        }
    }

    // Solve J^T J dt = J^T r for each camera (3 × 3 linear system)
    for cam_idx in 0..n_cams {
        let base = cam_idx * 3;
        // Extract 3 × 3 block
        let h = [
            jtj[base],
            jtj[base + 1],
            jtj[base + 2],
        ];
        let b3 = [jtr[base], jtr[base + 1], jtr[base + 2]];
        // Solve h * dt = b3
        if let Ok(h_inv) = mat3_inv(&h) {
            let dt = mat3_mul_vec3(&h_inv, &b3);
            cameras[cam_idx].translation[0] += dt[0];
            cameras[cam_idx].translation[1] += dt[1];
            cameras[cam_idx].translation[2] += dt[2];
        }
    }

    total_sq.sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detection_3d::frustum::CameraIntrinsics;

    fn make_identity_intrinsics() -> CameraIntrinsics {
        CameraIntrinsics {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
            width: 640,
            height: 480,
        }
    }

    /// Project a 3-D point through R, t and K.
    fn project(wp: &[f64; 3], r: &[[f64; 3]; 3], t: &[f64; 3], intr: &CameraIntrinsics) -> [f64; 2] {
        let xc = r[0][0] * wp[0] + r[0][1] * wp[1] + r[0][2] * wp[2] + t[0];
        let yc = r[1][0] * wp[0] + r[1][1] * wp[1] + r[1][2] * wp[2] + t[1];
        let zc = r[2][0] * wp[0] + r[2][1] * wp[1] + r[2][2] * wp[2] + t[2];
        [intr.fx * xc / zc + intr.cx, intr.fy * yc / zc + intr.cy]
    }

    #[test]
    fn test_svd_3x3_identity() {
        let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let (u, s, vt) = svd_3x3(m);
        // All singular values should be ~1
        for sv in &s {
            assert!((sv - 1.0).abs() < 1e-6, "sv = {sv}");
        }
        // U V^T should be identity
        let uvt = mat3_mul(&u, &vt);
        let det = mat3_det(&uvt);
        assert!((det - 1.0).abs() < 1e-6, "det = {det}");
    }

    #[test]
    fn test_svd_3x3_known() {
        // diagonal matrix
        let m = [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        let (_u, s, _vt) = svd_3x3(m);
        let mut s_sorted = s;
        s_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert!((s_sorted[0] - 3.0).abs() < 1e-4, "s[0] = {}", s_sorted[0]);
        assert!((s_sorted[1] - 2.0).abs() < 1e-4, "s[1] = {}", s_sorted[1]);
        assert!((s_sorted[2] - 1.0).abs() < 1e-4, "s[2] = {}", s_sorted[2]);
    }

    #[test]
    fn test_orthogonalize_rotation() {
        // Slightly perturbed identity
        let r = [
            [1.01, 0.02, -0.01],
            [-0.02, 0.99, 0.03],
            [0.01, -0.03, 1.00],
        ];
        let rot = orthogonalize_rotation(r);
        let det = mat3_det(&rot);
        assert!((det - 1.0).abs() < 1e-6, "det = {det}");
        // R^T R ≈ I
        let rt = mat3_transpose(&rot);
        let rtr = mat3_mul(&rt, &rot);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((rtr[i][j] - expected).abs() < 1e-6,
                    "rtr[{i}][{j}] = {}", rtr[i][j]);
            }
        }
    }

    #[test]
    fn test_dlt_identity_camera() {
        // R = I, t = [0, 0, 5] (camera 5 units in front of world origin).
        // Use non-planar world points (varying z) to avoid the planar-scene
        // degeneracy in the full 3 × 4 DLT.
        let intr = make_identity_intrinsics();
        let r_gt = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let t_gt = [0.0, 0.0, 5.0];

        // Non-planar 3-D point cloud (varying x, y, z)
        let raw: &[(f64, f64, f64)] = &[
            (-1.0, -1.0,  0.5),
            ( 0.5, -0.8, -0.3),
            ( 1.2,  0.3,  0.8),
            (-0.7,  1.0, -0.5),
            ( 0.0,  0.0,  0.0),
            ( 1.5,  1.5,  0.2),
            (-1.5,  0.5, -0.8),
            ( 0.3, -1.3,  1.0),
            ( 1.0, -0.5, -0.2),
            (-0.4,  1.4,  0.6),
        ];

        let world_pts: Vec<[f64; 3]> = raw.iter().map(|&(x, y, z)| [x, y, z]).collect();
        let image_pts: Vec<[f64; 2]> = world_pts.iter().map(|wp| project(wp, &r_gt, &t_gt, &intr)).collect();

        let (r_est, t_est) = dlt_extrinsics(&world_pts, &image_pts, &intr)
            .expect("DLT failed");

        // Reprojection errors should be small (< 5 px — the DLT is approximate)
        for (wp, ip) in world_pts.iter().zip(image_pts.iter()) {
            let cam = Camera {
                intrinsics: intr.clone(),
                rotation: r_est,
                translation: t_est,
            };
            let err = reprojection_error(&cam, wp, ip);
            assert!(err < 5.0, "reprojection_error = {err}");
        }
    }

    #[test]
    fn test_reprojection_error_exact() {
        let intr = make_identity_intrinsics();
        let r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let t = [0.0, 0.0, 5.0];
        let wp = [1.0, 1.0, 0.0];
        let ip = project(&wp, &r, &t, &intr);
        let cam = Camera { intrinsics: intr, rotation: r, translation: t };
        let err = reprojection_error(&cam, &wp, &ip);
        assert!(err < 1e-9, "err = {err}");
    }

    #[test]
    fn test_bundle_adjust_single_step_returns_residual() {
        let intr = make_identity_intrinsics();
        let mut cameras = vec![Camera::identity(intr.clone())];
        let world_pts = vec![[0.0_f64, 0.0, 5.0]];
        // Perfect observation at the principal point
        let observations = vec![(0usize, 0usize, [intr.cx, intr.cy])];
        let residual = bundle_adjust_single_step(&mut cameras, &observations, &world_pts);
        // Should be near zero for a perfect observation
        assert!(residual < 1.0, "residual = {residual}");
    }
}
