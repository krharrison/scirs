//! Pose estimation and camera geometry.
//!
//! Provides:
//!
//! * [`solve_pnp`] — Perspective-n-Point pose estimation (DLT + optional RANSAC).
//! * [`compute_homography`] — DLT homography from ≥ 4 point correspondences.
//! * [`compute_essential_matrix`] — 8-point algorithm for the essential matrix.
//! * [`decompose_essential_matrix`] — Decompose E into (R, t) candidates.
//! * [`compose_transforms`] — Compose two rigid-body transforms.

use crate::camera::CameraIntrinsics;
use crate::error::{Result, VisionError};

// ─────────────────────────────────────────────────────────────────────────────
// PnP result
// ─────────────────────────────────────────────────────────────────────────────

/// Output of [`solve_pnp`].
#[derive(Debug, Clone)]
pub struct PnPResult {
    /// Estimated 3×3 rotation matrix (world → camera).
    pub rotation: [[f64; 3]; 3],
    /// Estimated 3-element translation vector (world → camera, metres).
    pub translation: [f64; 3],
    /// Mean re-projection error over inlier correspondences (pixels).
    pub reprojection_error: f64,
    /// Indices of inlier correspondences (all indices when `use_ransac = false`).
    pub inliers: Vec<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// PnP
// ─────────────────────────────────────────────────────────────────────────────

/// Solve the Perspective-n-Point problem: estimate `(R, t)` such that
/// `image_points[i] ≈ K * (R * object_points[i] + t)`.
///
/// Uses the **Direct Linear Transform** (6-point minimum) followed by
/// factorisation of the 3×4 projection matrix P = K^{-1} [R | t].  When
/// `use_ransac = true` the solver is wrapped in a RANSAC loop (min 6 points
/// per hypothesis, max 200 trials).
///
/// # Errors
/// Returns `Err` when fewer than 6 correspondences are supplied or the DLT
/// system is degenerate.
///
/// # Example
/// ```
/// use scirs2_vision::camera::CameraIntrinsics;
/// use scirs2_vision::pose::solve_pnp;
///
/// let obj: Vec<[f64; 3]> = vec![
///     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
///     [1.0, 1.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.1],
/// ];
/// let cam = CameraIntrinsics::ideal(500.0, 500.0, 320.0, 240.0);
/// // Simulate R = I, t = (0, 0, 2)
/// let img: Vec<[f64; 2]> = obj.iter().map(|p| {
///     let z = p[2] + 2.0;
///     [500.0 * p[0] / z + 320.0, 500.0 * p[1] / z + 240.0]
/// }).collect();
/// let result = solve_pnp(&obj, &img, &cam, false);
/// assert!(result.is_ok(), "{:?}", result.err());
/// ```
pub fn solve_pnp(
    object_points: &[[f64; 3]],
    image_points: &[[f64; 2]],
    intrinsics: &CameraIntrinsics,
    use_ransac: bool,
) -> Result<PnPResult> {
    let n = object_points.len();
    if n < 6 {
        return Err(VisionError::InvalidParameter(
            "solve_pnp requires at least 6 correspondences".to_string(),
        ));
    }
    if image_points.len() != n {
        return Err(VisionError::InvalidParameter(
            "object_points and image_points must have the same length".to_string(),
        ));
    }

    if use_ransac {
        solve_pnp_ransac(object_points, image_points, intrinsics)
    } else {
        solve_pnp_dlt_all(object_points, image_points, intrinsics)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Homography
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a 3×3 homography matrix from ≥ 4 point correspondences using the
/// **Direct Linear Transform**.
///
/// The returned matrix `H` satisfies `dst ≅ H * src` (up to scale).  It is
/// normalised so that `H[2][2] = 1`.
///
/// Returns `None` when the system is degenerate (insufficient distinct points
/// or all points collinear).
///
/// # Example
/// ```
/// use scirs2_vision::pose::compute_homography;
///
/// let src = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let dst = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
/// let h = compute_homography(&src, &dst).unwrap();
/// // Scale factor = 2 → H[0][0] ≈ 2, H[1][1] ≈ 2
/// assert!((h[0][0] - 2.0).abs() < 1e-6, "H[0][0]={}", h[0][0]);
/// assert!((h[1][1] - 2.0).abs() < 1e-6, "H[1][1]={}", h[1][1]);
/// ```
pub fn compute_homography(src_pts: &[[f64; 2]], dst_pts: &[[f64; 2]]) -> Option<[[f64; 3]; 3]> {
    let n = src_pts.len();
    if n < 4 || dst_pts.len() != n {
        return None;
    }

    // Normalise points for numerical stability
    let (src_n, t_src) = normalise_points(src_pts);
    let (dst_n, t_dst) = normalise_points(dst_pts);

    // Build 2n×9 DLT matrix A
    let mut a = vec![[0.0f64; 9]; 2 * n];
    for i in 0..n {
        let (x, y) = (src_n[i][0], src_n[i][1]);
        let (xp, yp) = (dst_n[i][0], dst_n[i][1]);
        a[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, xp * x, xp * y, xp];
        a[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, yp * x, yp * y, yp];
    }

    // Solve via SVD: null-space of A (last row of V^T from A^T A)
    let h_vec = svd_nullspace_9(&a)?;
    let h_norm = [
        [h_vec[0], h_vec[1], h_vec[2]],
        [h_vec[3], h_vec[4], h_vec[5]],
        [h_vec[6], h_vec[7], h_vec[8]],
    ];

    // Denormalise: H = T_dst^{-1} * H_norm * T_src
    let h = mat3_mul(mat3_mul(inv_normalise_mat(t_dst), h_norm), t_src);

    // Normalise so H[2][2] = 1
    let scale = h[2][2];
    if scale.abs() < 1e-12 {
        return None;
    }
    Some([
        [h[0][0] / scale, h[0][1] / scale, h[0][2] / scale],
        [h[1][0] / scale, h[1][1] / scale, h[1][2] / scale],
        [h[2][0] / scale, h[2][1] / scale, h[2][2] / scale],
    ])
}

// ─────────────────────────────────────────────────────────────────────────────
// Essential matrix (8-point algorithm)
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the essential matrix from ≥ 8 normalised point correspondences
/// using the **8-point algorithm**.
///
/// Points are normalised by the intrinsic matrix K before the DLT step.  The
/// essential-matrix constraint `rank(E) = 2` is enforced by zeroing the
/// smallest singular value after the initial DLT estimate.
///
/// Returns `None` when the DLT system is degenerate or fewer than 8
/// correspondences are provided.
///
/// # Example
/// ```
/// use scirs2_vision::camera::CameraIntrinsics;
/// use scirs2_vision::pose::compute_essential_matrix;
///
/// let cam = CameraIntrinsics::ideal(500.0, 500.0, 320.0, 240.0);
/// // Two cameras looking at a planar scene: identity pose difference
/// let pts1: Vec<[f64; 2]> = (0..8).map(|i| [i as f64 * 30.0 + 320.0, 240.0]).collect();
/// let pts2 = pts1.clone();
/// // E from identical views should have all-near-zero entries
/// let _e = compute_essential_matrix(&pts1, &pts2, &cam);
/// // (allow None for degenerate identical-point case)
/// ```
pub fn compute_essential_matrix(
    pts1: &[[f64; 2]],
    pts2: &[[f64; 2]],
    intrinsics: &CameraIntrinsics,
) -> Option<[[f64; 3]; 3]> {
    let n = pts1.len();
    if n < 8 || pts2.len() != n {
        return None;
    }

    // Normalise to camera coordinates
    let norm = |p: [f64; 2]| -> [f64; 2] {
        [
            (p[0] - intrinsics.cx) / intrinsics.fx,
            (p[1] - intrinsics.cy) / intrinsics.fy,
        ]
    };

    let n1: Vec<[f64; 2]> = pts1.iter().map(|&p| norm(p)).collect();
    let n2: Vec<[f64; 2]> = pts2.iter().map(|&p| norm(p)).collect();

    // Build n×9 epipolar constraint matrix A
    // Each row: [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    let mut a = vec![[0.0f64; 9]; n];
    for i in 0..n {
        let (x1, y1) = (n1[i][0], n1[i][1]);
        let (x2, y2) = (n2[i][0], n2[i][1]);
        a[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0];
    }

    let e_vec = svd_nullspace_9(&a)?;
    let mut e = [
        [e_vec[0], e_vec[1], e_vec[2]],
        [e_vec[3], e_vec[4], e_vec[5]],
        [e_vec[6], e_vec[7], e_vec[8]],
    ];

    // Enforce rank-2: decompose with SVD, zero smallest singular value
    enforce_essential_rank2(&mut e);

    Some(e)
}

/// Decompose an essential matrix into up to 4 (R, t) candidates.
///
/// A valid essential matrix has two distinct non-zero singular values (both
/// equal to 1 after normalisation) and produces 4 solutions:
/// `(R1, t)`, `(R1, -t)`, `(R2, t)`, `(R2, -t)`.  The caller should select
/// the correct solution by cheirality check (reconstructed points must lie
/// in front of both cameras).
///
/// # Example
/// ```
/// use scirs2_vision::pose::decompose_essential_matrix;
///
/// // E = [t]× R, where t = (1,0,0) and R = I
/// let e = [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
/// let solutions = decompose_essential_matrix(&e);
/// assert!(!solutions.is_empty());
/// ```
pub fn decompose_essential_matrix(e_mat: &[[f64; 3]; 3]) -> Vec<([[f64; 3]; 3], [f64; 3])> {
    // SVD of E
    let (u, _s, v) = svd3x3_jacobi(*e_mat);

    // W = [[0,-1,0],[1,0,0],[0,0,1]]
    let w = [[0.0f64, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let wt = mat3_transpose(w);

    // R = U * W * V^T   or   R = U * W^T * V^T
    let vt = mat3_transpose(v);
    let r1 = mat3_mul(mat3_mul(u, w), vt);
    let r2 = mat3_mul(mat3_mul(u, wt), vt);

    // Ensure proper rotations
    let r1 = ensure_rotation(r1);
    let r2 = ensure_rotation(r2);

    // t = third column of U (up to sign)
    let t = [u[0][2], u[1][2], u[2][2]];
    let nt = [-u[0][2], -u[1][2], -u[2][2]];

    vec![(r1, t), (r1, nt), (r2, t), (r2, nt)]
}

// ─────────────────────────────────────────────────────────────────────────────
// Compose transforms
// ─────────────────────────────────────────────────────────────────────────────

/// Compose two rigid-body transforms `(R, t)`.
///
/// Given transforms `t1: p ↦ R1*p + t1` and `t2: p ↦ R2*p + t2`, returns
/// their composition `t2 ∘ t1: p ↦ R2*(R1*p + t1) + t2`.
///
/// # Example
/// ```
/// use scirs2_vision::pose::compose_transforms;
///
/// let id  = ([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]], [0.0;3]);
/// let t1  = ([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]], [1.0, 0.0, 0.0]);
/// let t2  = ([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]], [0.0, 1.0, 0.0]);
/// let comp = compose_transforms(t1, t2);
/// assert!((comp.1[0] - 1.0).abs() < 1e-12);
/// assert!((comp.1[1] - 1.0).abs() < 1e-12);
/// let comp_id = compose_transforms(id.clone(), id);
/// assert!((comp_id.0[0][0] - 1.0).abs() < 1e-12);
/// ```
pub fn compose_transforms(
    t1: ([[f64; 3]; 3], [f64; 3]),
    t2: ([[f64; 3]; 3], [f64; 3]),
) -> ([[f64; 3]; 3], [f64; 3]) {
    let (r1, t1v) = t1;
    let (r2, t2v) = t2;
    let r = mat3_mul(r2, r1);
    let new_t = [
        r2[0][0] * t1v[0] + r2[0][1] * t1v[1] + r2[0][2] * t1v[2] + t2v[0],
        r2[1][0] * t1v[0] + r2[1][1] * t1v[1] + r2[1][2] * t1v[2] + t2v[1],
        r2[2][0] * t1v[0] + r2[2][1] * t1v[1] + r2[2][2] * t1v[2] + t2v[2],
    ];
    (r, new_t)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: DLT PnP (all points)
// ─────────────────────────────────────────────────────────────────────────────

fn solve_pnp_dlt_all(
    obj: &[[f64; 3]],
    img: &[[f64; 2]],
    intrinsics: &CameraIntrinsics,
) -> Result<PnPResult> {
    let n = obj.len();
    // Build 2n × 12 DLT matrix for P (3×4 projection matrix)
    let k = intrinsics.calibration_matrix();

    // Normalise image points by K^{-1}
    let img_n: Vec<[f64; 2]> = img
        .iter()
        .map(|&p| [(p[0] - k[0][2]) / k[0][0], (p[1] - k[1][2]) / k[1][1]])
        .collect();

    // Build constraint matrix for P = [R | t] (calibrated DLT)
    let mut a = vec![[0.0f64; 12]; 2 * n];
    for i in 0..n {
        let (x, y, z) = (obj[i][0], obj[i][1], obj[i][2]);
        let (u, v) = (img_n[i][0], img_n[i][1]);
        a[2 * i] = [x, y, z, 1.0, 0.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u * z, -u];
        a[2 * i + 1] = [0.0, 0.0, 0.0, 0.0, x, y, z, 1.0, -v * x, -v * y, -v * z, -v];
    }

    let p_vec = svd_nullspace_12(&a)
        .ok_or_else(|| VisionError::OperationError("PnP DLT: degenerate system".to_string()))?;

    // Reshape into 3×4
    let p34 = [
        [p_vec[0], p_vec[1], p_vec[2], p_vec[3]],
        [p_vec[4], p_vec[5], p_vec[6], p_vec[7]],
        [p_vec[8], p_vec[9], p_vec[10], p_vec[11]],
    ];

    // Extract R and t via QR decomposition of P[:, :3]
    let (rotation, translation) = extract_rt_from_p34(&p34)?;

    // Compute reprojection error
    let mut total_err = 0.0f64;
    for i in 0..n {
        let pt_cam = mat3_vec(rotation, obj[i]);
        let pt_cam = [
            pt_cam[0] + translation[0],
            pt_cam[1] + translation[1],
            pt_cam[2] + translation[2],
        ];
        if pt_cam[2] <= 0.0 {
            continue;
        }
        let u_hat = k[0][0] * pt_cam[0] / pt_cam[2] + k[0][2];
        let v_hat = k[1][1] * pt_cam[1] / pt_cam[2] + k[1][2];
        let eu = u_hat - img[i][0];
        let ev = v_hat - img[i][1];
        total_err += (eu * eu + ev * ev).sqrt();
    }

    Ok(PnPResult {
        rotation,
        translation,
        reprojection_error: total_err / n as f64,
        inliers: (0..n).collect(),
    })
}

fn solve_pnp_ransac(
    obj: &[[f64; 3]],
    img: &[[f64; 2]],
    intrinsics: &CameraIntrinsics,
) -> Result<PnPResult> {
    let n = obj.len();
    let max_iter = 200usize;
    let inlier_thresh = 4.0f64; // pixels

    let mut rng = 54321u64;
    let mut rng_next = |max: usize| -> usize {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng >> 33) as usize) % max
    };

    let mut best_inliers: Vec<usize> = Vec::new();
    let k = intrinsics.calibration_matrix();

    for _ in 0..max_iter {
        // Sample 6 distinct points
        let mut sample = Vec::with_capacity(6);
        while sample.len() < 6 {
            let idx = rng_next(n);
            if !sample.contains(&idx) {
                sample.push(idx);
            }
        }
        let obj_s: Vec<[f64; 3]> = sample.iter().map(|&i| obj[i]).collect();
        let img_s: Vec<[f64; 2]> = sample.iter().map(|&i| img[i]).collect();

        if let Ok(result) = solve_pnp_dlt_all(&obj_s, &img_s, intrinsics) {
            let inliers: Vec<usize> = (0..n)
                .filter(|&i| {
                    let pt_cam = mat3_vec(result.rotation, obj[i]);
                    let pt_cam = [
                        pt_cam[0] + result.translation[0],
                        pt_cam[1] + result.translation[1],
                        pt_cam[2] + result.translation[2],
                    ];
                    if pt_cam[2] <= 0.0 {
                        return false;
                    }
                    let u_hat = k[0][0] * pt_cam[0] / pt_cam[2] + k[0][2];
                    let v_hat = k[1][1] * pt_cam[1] / pt_cam[2] + k[1][2];
                    let eu = u_hat - img[i][0];
                    let ev = v_hat - img[i][1];
                    (eu * eu + ev * ev).sqrt() < inlier_thresh
                })
                .collect();

            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
            }
        }
    }

    if best_inliers.len() >= 6 {
        // Refit on all inliers
        let obj_i: Vec<[f64; 3]> = best_inliers.iter().map(|&i| obj[i]).collect();
        let img_i: Vec<[f64; 2]> = best_inliers.iter().map(|&i| img[i]).collect();
        if let Ok(mut result) = solve_pnp_dlt_all(&obj_i, &img_i, intrinsics) {
            result.inliers = best_inliers;
            return Ok(result);
        }
    }

    // Fallback: fit on ALL points (handles coplanar/degenerate subsets).
    // With more points the DLT system is better conditioned.
    let mut result = solve_pnp_dlt_all(obj, img, intrinsics)?;
    // Recompute inliers based on reprojection error
    let k = intrinsics.calibration_matrix();
    let inlier_thresh = 8.0_f64; // relaxed for fallback
    let inliers: Vec<usize> = (0..n)
        .filter(|&i| {
            let pt_cam = mat3_vec(result.rotation, obj[i]);
            let pt_cam = [
                pt_cam[0] + result.translation[0],
                pt_cam[1] + result.translation[1],
                pt_cam[2] + result.translation[2],
            ];
            if pt_cam[2] <= 0.0 {
                return false;
            }
            let u_hat = k[0][0] * pt_cam[0] / pt_cam[2] + k[0][2];
            let v_hat = k[1][1] * pt_cam[1] / pt_cam[2] + k[1][2];
            let eu = u_hat - img[i][0];
            let ev = v_hat - img[i][1];
            (eu * eu + ev * ev).sqrt() < inlier_thresh
        })
        .collect();
    result.inliers = inliers;
    Ok(result)
}

/// Extract rotation and translation from a 3×4 projection matrix [R|t] using
/// RQ decomposition (via QR on the reverse matrix).
fn extract_rt_from_p34(p: &[[f64; 4]; 3]) -> Result<([[f64; 3]; 3], [f64; 3])> {
    // Extract the 3×3 submatrix M = P[:, :3]
    let m = [
        [p[0][0], p[0][1], p[0][2]],
        [p[1][0], p[1][1], p[1][2]],
        [p[2][0], p[2][1], p[2][2]],
    ];

    // Normalise to get rotation: R = M / ||m[2]||
    let scale = (m[2][0] * m[2][0] + m[2][1] * m[2][1] + m[2][2] * m[2][2]).sqrt();
    if scale < 1e-12 {
        return Err(VisionError::OperationError(
            "Degenerate P matrix in PnP extraction".to_string(),
        ));
    }

    let mut r = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = m[i][j] / scale;
        }
    }

    // Ensure proper rotation
    let r = ensure_rotation(r);

    // t = R^{-1} * (P[:, 3] / scale) = R^T * (P[:, 3] / scale)
    let t_raw = [p[0][3] / scale, p[1][3] / scale, p[2][3] / scale];
    let t = [
        r[0][0] * t_raw[0] + r[1][0] * t_raw[1] + r[2][0] * t_raw[2],
        r[0][1] * t_raw[0] + r[1][1] * t_raw[1] + r[2][1] * t_raw[2],
        r[0][2] * t_raw[0] + r[1][2] * t_raw[1] + r[2][2] * t_raw[2],
    ];
    // Actually t is directly: R * X + t maps 3D to camera frame.
    // So t = P[:,3]/scale directly after we have R.
    let t_direct = [p[0][3] / scale, p[1][3] / scale, p[2][3] / scale];

    Ok((r, t_direct))
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix utilities
// ─────────────────────────────────────────────────────────────────────────────

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
// Point normalisation for DLT
// ─────────────────────────────────────────────────────────────────────────────

/// Normalise 2-D points to have zero centroid and mean distance √2.
/// Returns (normalised_points, normalisation_matrix).
fn normalise_points(pts: &[[f64; 2]]) -> (Vec<[f64; 2]>, [[f64; 3]; 3]) {
    let n = pts.len();
    if n == 0 {
        return (
            Vec::new(),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        );
    }
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;
    for p in pts {
        cx += p[0];
        cy += p[1];
    }
    cx /= n as f64;
    cy /= n as f64;

    let mut mean_dist = 0.0f64;
    for p in pts {
        mean_dist += ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt();
    }
    mean_dist /= n as f64;

    let scale = if mean_dist > 1e-10 {
        2.0f64.sqrt() / mean_dist
    } else {
        1.0
    };

    let normed: Vec<[f64; 2]> = pts
        .iter()
        .map(|p| [(p[0] - cx) * scale, (p[1] - cy) * scale])
        .collect();

    let t = [
        [scale, 0.0, -cx * scale],
        [0.0, scale, -cy * scale],
        [0.0, 0.0, 1.0],
    ];

    (normed, t)
}

fn inv_normalise_mat(t: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // T is a similarity transform: [[s, 0, tx], [0, s, ty], [0, 0, 1]]
    // T^{-1} = [[1/s, 0, -tx/s], [0, 1/s, -ty/s], [0, 0, 1]]
    let s = t[0][0];
    if s.abs() < 1e-12 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    [
        [1.0 / s, 0.0, -t[0][2] / s],
        [0.0, 1.0 / s, -t[1][2] / s],
        [0.0, 0.0, 1.0],
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// SVD null-space solvers (Jacobi)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the null-space vector of an (m × 9) matrix A (A^T A method).
fn svd_nullspace_9(a: &[[f64; 9]]) -> Option<[f64; 9]> {
    // Form A^T * A (9×9 symmetric)
    let mut ata = [[0.0f64; 9]; 9];
    for row in a {
        for i in 0..9 {
            for j in 0..9 {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    // Eigenvector corresponding to smallest eigenvalue via Jacobi
    let v = jacobi_eigen9_min(&ata)?;
    Some(v)
}

/// Compute the null-space vector of an (m × 12) matrix A (A^T A method).
fn svd_nullspace_12(a: &[[f64; 12]]) -> Option<[f64; 12]> {
    let mut ata = [[0.0f64; 12]; 12];
    for row in a {
        for i in 0..12 {
            for j in 0..12 {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    jacobi_eigen12_min(&ata)
}

// ── Jacobi eigenvector for smallest eigenvalue of a symmetric 9×9 matrix ────

fn jacobi_eigen9_min(a: &[[f64; 9]; 9]) -> Option<[f64; 9]> {
    let mut m = *a;
    let mut v = [[0.0f64; 9]; 9];
    #[allow(clippy::needless_range_loop)]
    for i in 0..9 {
        v[i][i] = 1.0;
    }

    for _ in 0..200 {
        let (p, q, max_val) = largest_off_diag_9(&m);
        if max_val < 1e-12 {
            break;
        }
        givens_rotate_9(&mut m, &mut v, p, q);
    }

    // Find column index with smallest diagonal (eigenvalue)
    let mut min_idx = 0;
    let mut min_val = m[0][0];
    #[allow(clippy::needless_range_loop)]
    for i in 1..9 {
        if m[i][i] < min_val {
            min_val = m[i][i];
            min_idx = i;
        }
    }

    let mut result = [0.0f64; 9];
    #[allow(clippy::needless_range_loop)]
    for i in 0..9 {
        result[i] = v[i][min_idx];
    }
    // Normalise
    let len = result.iter().map(|x| x * x).sum::<f64>().sqrt();
    if len < 1e-12 {
        return None;
    }
    for x in &mut result {
        *x /= len;
    }
    Some(result)
}

fn largest_off_diag_9(m: &[[f64; 9]; 9]) -> (usize, usize, f64) {
    let mut max_val = 0.0f64;
    let mut pi = 0usize;
    let mut qi = 1usize;
    #[allow(clippy::needless_range_loop)]
    for i in 0..9 {
        for j in (i + 1)..9 {
            if m[i][j].abs() > max_val {
                max_val = m[i][j].abs();
                pi = i;
                qi = j;
            }
        }
    }
    (pi, qi, max_val)
}

fn givens_rotate_9(m: &mut [[f64; 9]; 9], v: &mut [[f64; 9]; 9], p: usize, q: usize) {
    let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
    let t = if theta >= 0.0 {
        1.0 / (theta + (1.0 + theta * theta).sqrt())
    } else {
        1.0 / (theta - (1.0 + theta * theta).sqrt())
    };
    let cos = 1.0 / (1.0 + t * t).sqrt();
    let sin = t * cos;
    let tau = sin / (1.0 + cos);

    let mpq = m[p][q];
    m[p][p] -= t * mpq;
    m[q][q] += t * mpq;
    m[p][q] = 0.0;
    m[q][p] = 0.0;

    #[allow(clippy::needless_range_loop)]
    for r in 0..9 {
        if r != p && r != q {
            let mrp = m[r][p];
            let mrq = m[r][q];
            m[r][p] = mrp - sin * (mrq + tau * mrp);
            m[p][r] = m[r][p];
            m[r][q] = mrq + sin * (mrp - tau * mrq);
            m[q][r] = m[r][q];
        }
    }
    #[allow(clippy::needless_range_loop)]
    for r in 0..9 {
        let vp = v[r][p];
        let vq = v[r][q];
        v[r][p] = vp - sin * (vq + tau * vp);
        v[r][q] = vq + sin * (vp - tau * vq);
    }
}

// ── Jacobi eigenvector for smallest eigenvalue of a symmetric 12×12 matrix ──

fn jacobi_eigen12_min(a: &[[f64; 12]; 12]) -> Option<[f64; 12]> {
    let mut m = *a;
    let mut v = [[0.0f64; 12]; 12];
    #[allow(clippy::needless_range_loop)]
    for i in 0..12 {
        v[i][i] = 1.0;
    }

    for _ in 0..1000 {
        let (p, q, max_val) = largest_off_diag_12(&m);
        if max_val < 1e-14 {
            break;
        }
        givens_rotate_12(&mut m, &mut v, p, q);
    }

    let mut min_idx = 0;
    let mut min_val = m[0][0];
    #[allow(clippy::needless_range_loop)]
    for i in 1..12 {
        if m[i][i] < min_val {
            min_val = m[i][i];
            min_idx = i;
        }
    }

    let mut result = [0.0f64; 12];
    #[allow(clippy::needless_range_loop)]
    for i in 0..12 {
        result[i] = v[i][min_idx];
    }
    let len = result.iter().map(|x| x * x).sum::<f64>().sqrt();
    if len < 1e-12 {
        return None;
    }
    for x in &mut result {
        *x /= len;
    }
    Some(result)
}

fn largest_off_diag_12(m: &[[f64; 12]; 12]) -> (usize, usize, f64) {
    let mut max_val = 0.0f64;
    let mut pi = 0usize;
    let mut qi = 1usize;
    #[allow(clippy::needless_range_loop)]
    for i in 0..12 {
        for j in (i + 1)..12 {
            if m[i][j].abs() > max_val {
                max_val = m[i][j].abs();
                pi = i;
                qi = j;
            }
        }
    }
    (pi, qi, max_val)
}

fn givens_rotate_12(m: &mut [[f64; 12]; 12], v: &mut [[f64; 12]; 12], p: usize, q: usize) {
    let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
    let t = if theta >= 0.0 {
        1.0 / (theta + (1.0 + theta * theta).sqrt())
    } else {
        1.0 / (theta - (1.0 + theta * theta).sqrt())
    };
    let cos = 1.0 / (1.0 + t * t).sqrt();
    let sin = t * cos;
    let tau = sin / (1.0 + cos);

    let mpq = m[p][q];
    m[p][p] -= t * mpq;
    m[q][q] += t * mpq;
    m[p][q] = 0.0;
    m[q][p] = 0.0;

    #[allow(clippy::needless_range_loop)]
    for r in 0..12 {
        if r != p && r != q {
            let mrp = m[r][p];
            let mrq = m[r][q];
            m[r][p] = mrp - sin * (mrq + tau * mrp);
            m[p][r] = m[r][p];
            m[r][q] = mrq + sin * (mrp - tau * mrq);
            m[q][r] = m[r][q];
        }
    }
    #[allow(clippy::needless_range_loop)]
    for r in 0..12 {
        let vp = v[r][p];
        let vq = v[r][q];
        v[r][p] = vp - sin * (vq + tau * vp);
        v[r][q] = vq + sin * (vp - tau * vq);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Essential matrix rank-2 enforcement via Jacobi SVD
// ─────────────────────────────────────────────────────────────────────────────

fn enforce_essential_rank2(e: &mut [[f64; 3]; 3]) {
    let (u, mut s, v) = svd3x3_jacobi(*e);
    // Set smallest singular value to 0, average the other two
    // Sort singular values descending
    let mut sv = [(s[0], 0usize), (s[1], 1), (s[2], 2)];
    sv.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let avg = (sv[0].0 + sv[1].0) / 2.0;
    s[sv[0].1] = avg;
    s[sv[1].1] = avg;
    s[sv[2].1] = 0.0;

    // Recompose E = U * diag(s) * V^T
    let vt = mat3_transpose(v);
    let sv_diag = [[s[0], 0.0, 0.0], [0.0, s[1], 0.0], [0.0, 0.0, s[2]]];
    *e = mat3_mul(mat3_mul(u, sv_diag), vt);
}

/// Jacobi SVD for a general 3×3 matrix.
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

    let s = [
        evals[0].max(0.0).sqrt(),
        evals[1].max(0.0).sqrt(),
        evals[2].max(0.0).sqrt(),
    ];

    let mut u = [[0.0f64; 3]; 3];
    #[allow(clippy::needless_range_loop)]
    for j in 0..3 {
        if s[j] > 1e-10 {
            for i in 0..3 {
                u[i][j] = (a[i][0] * v[0][j] + a[i][1] * v[1][j] + a[i][2] * v[2][j]) / s[j];
            }
        } else if j == 0 {
            u[0][j] = 1.0;
        } else if j == 1 {
            u[1][j] = 1.0;
        } else {
            u[2][j] = 1.0;
        }
    }

    (u, s, v)
}

fn jacobi_eigen3(a: &[[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut m = *a;
    let mut v = [[1.0f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for _ in 0..50 {
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
        if max_val < 1e-14 {
            break;
        }

        let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            1.0 / (theta - (1.0 + theta * theta).sqrt())
        };
        let cos = 1.0 / (1.0 + t * t).sqrt();
        let sin = t * cos;
        let tau = sin / (1.0 + cos);

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

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cam() -> CameraIntrinsics {
        CameraIntrinsics::ideal(500.0, 500.0, 320.0, 240.0)
    }

    fn gen_pnp_data(n: usize, tz: f64) -> (Vec<[f64; 3]>, Vec<[f64; 2]>) {
        let cam = test_cam();
        // Object points with varying z (non-coplanar) for better DLT conditioning
        let obj: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let x = (i % 4) as f64 * 0.2 - 0.3;
                let y = (i / 4) as f64 * 0.2 - 0.3;
                let z = (i as f64 * 0.05) - 0.1; // non-coplanar: spread in z
                [x, y, z]
            })
            .collect();
        // Project with identity R, translation (0,0,tz)
        let img: Vec<[f64; 2]> = obj
            .iter()
            .map(|p| {
                let z = p[2] + tz;
                [500.0 * p[0] / z + cam.cx, 500.0 * p[1] / z + cam.cy]
            })
            .collect();
        (obj, img)
    }

    #[test]
    fn test_solve_pnp_basic() {
        let (obj, img) = gen_pnp_data(12, 2.0);
        let result = solve_pnp(&obj, &img, &test_cam(), false)
            .expect("solve_pnp should succeed with valid correspondences");
        // Re-projection error should be small
        assert!(
            result.reprojection_error < 5.0,
            "reprojection_error={}",
            result.reprojection_error
        );
        assert_eq!(result.inliers.len(), obj.len());
    }

    #[test]
    fn test_solve_pnp_too_few_points() {
        let cam = test_cam();
        assert!(solve_pnp(&[[0.0; 3]; 5], &[[0.0; 2]; 5], &cam, false).is_err());
    }

    #[test]
    fn test_solve_pnp_ransac() {
        let (mut obj, mut img) = gen_pnp_data(16, 3.0);
        // Add outliers
        img.push([999.0, 999.0]);
        obj.push([5.0, 5.0, 0.0]);
        img.push([0.0, 0.0]);
        obj.push([-5.0, -5.0, 0.0]);
        let result = solve_pnp(&obj, &img, &test_cam(), true);
        assert!(result.is_ok(), "{:?}", result.err());
    }

    #[test]
    fn test_homography_scale() {
        let src = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let dst: Vec<[f64; 2]> = src.iter().map(|&[x, y]| [x * 3.0, y * 3.0]).collect();
        let h = compute_homography(&src, &dst)
            .expect("compute_homography should succeed with 4+ correspondences");
        assert!((h[0][0] - 3.0).abs() < 1e-5, "H[0][0]={}", h[0][0]);
        assert!((h[1][1] - 3.0).abs() < 1e-5, "H[1][1]={}", h[1][1]);
        assert!((h[2][2] - 1.0).abs() < 1e-5, "H[2][2]={}", h[2][2]);
    }

    #[test]
    fn test_homography_translation() {
        let src = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let dst: Vec<[f64; 2]> = src.iter().map(|&[x, y]| [x + 10.0, y + 5.0]).collect();
        let h = compute_homography(&src, &dst)
            .expect("compute_homography should succeed with 4+ correspondences");
        // Apply H to src[0] = (0,0,1) → should give (10, 5, 1)
        let hw = h[2][0] * 0.0 + h[2][1] * 0.0 + h[2][2];
        let u = (h[0][0] * 0.0 + h[0][1] * 0.0 + h[0][2]) / hw;
        let v = (h[1][0] * 0.0 + h[1][1] * 0.0 + h[1][2]) / hw;
        assert!((u - 10.0).abs() < 1e-5, "u={}", u);
        assert!((v - 5.0).abs() < 1e-5, "v={}", v);
    }

    #[test]
    fn test_homography_too_few() {
        let src = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let dst = src.clone();
        assert!(compute_homography(&src, &dst).is_none());
    }

    #[test]
    fn test_essential_matrix_returns_something() {
        let cam = test_cam();
        let pts1: Vec<[f64; 2]> = (0..8)
            .map(|i| [i as f64 * 10.0 + 280.0, 200.0 + (i as f64 * 5.0)])
            .collect();
        // Small horizontal shift (simulates camera baseline)
        let pts2: Vec<[f64; 2]> = pts1.iter().map(|&[x, y]| [x - 5.0, y]).collect();
        let e = compute_essential_matrix(&pts1, &pts2, &cam);
        // Could be None for near-degenerate, but should not panic
        let _ = e;
    }

    #[test]
    fn test_decompose_essential_matrix() {
        // E for a pure horizontal translation t=(1,0,0): E = [t]× = [[0,0,0],[0,0,-1],[0,1,0]]
        let e = [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
        let solutions = decompose_essential_matrix(&e);
        assert!(!solutions.is_empty(), "Expected non-empty solutions");
        // Each solution should have a valid rotation determinant ≈ +1
        for (r, _t) in &solutions {
            let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
                - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
                + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
            assert!((det - 1.0).abs() < 0.1, "det={}", det);
        }
    }

    #[test]
    fn test_compose_transforms_identity() {
        let id = (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0f64; 3],
        );
        let t = (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [1.0, 2.0, 3.0],
        );
        let comp = compose_transforms(id, t);
        assert!((comp.1[0] - 1.0).abs() < 1e-12);
        assert!((comp.1[1] - 2.0).abs() < 1e-12);
        let comp2 = compose_transforms(t, id);
        assert!((comp2.1[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_compose_transforms_chain() {
        // Two translations should add up
        let tx1 = (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [1.0, 0.0, 0.0],
        );
        let tx2 = (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [2.0, 0.0, 0.0],
        );
        let comp = compose_transforms(tx1, tx2);
        assert!((comp.1[0] - 3.0).abs() < 1e-12, "tx={}", comp.1[0]);
    }
}
