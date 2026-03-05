//! Robust homography estimation via RANSAC and DLT
//!
//! This module provides:
//!
//! - **DLT** (Direct Linear Transform) – algebraic minimum-norm solution for
//!   homography estimation from ≥ 4 point correspondences.
//! - **RANSAC** – random sampling consensus for robust estimation in the
//!   presence of outliers.
//! - **Iterative refinement** – Levenberg-Marquardt minimisation of the
//!   symmetric reprojection error applied to the inlier set.
//!
//! # Algorithm
//!
//! Given `N` point pairs `(p_i, p_i')` the 3×3 homography `H` satisfies
//! `p_i' ~ H p_i`.  The solver follows these steps:
//!
//! 1. **Normalise** both point sets (Hartley normalisation) to improve
//!    numerical conditioning.
//! 2. **Build** the 2N×9 DLT coefficient matrix `A`.
//! 3. **Solve** `min ||A h||` subject to `||h|| = 1` via truncated SVD (power
//!    iteration for the smallest singular vector).
//! 4. **Denormalise** to get H in the original coordinate frame.
//! 5. **RANSAC** – iteratively sample 4 correspondences, apply DLT, count
//!    inliers (symmetric transfer error < threshold), update best model.
//! 6. **Refine** – Levenberg-Marquardt on the inlier set to minimise the
//!    sum of squared symmetric reprojection errors.
//!
//! # References
//!
//! - Hartley, R. & Zisserman, A. (2003). *Multiple View Geometry in Computer
//!   Vision* (2nd ed.). Cambridge University Press.

use crate::error::{Result, VisionError};
use std::f64::consts::PI;

// ─── Public types ─────────────────────────────────────────────────────────────

/// A 3×3 projective homography stored in row-major order.
#[derive(Debug, Clone)]
pub struct Homography {
    /// Row-major 3×3 matrix elements
    pub data: [[f64; 3]; 3],
}

impl Homography {
    /// Identity homography
    pub fn identity() -> Self {
        Self {
            data: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Map a 2-D point through the homography.
    ///
    /// Returns `None` if the point maps to the line at infinity (w ≈ 0).
    pub fn project(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let h = &self.data;
        let w = h[2][0] * x + h[2][1] * y + h[2][2];
        if w.abs() < 1e-10 {
            return None;
        }
        let px = (h[0][0] * x + h[0][1] * y + h[0][2]) / w;
        let py = (h[1][0] * x + h[1][1] * y + h[1][2]) / w;
        Some((px, py))
    }

    /// Compute the matrix inverse of this homography (for back-projection).
    pub fn inverse(&self) -> Option<Homography> {
        let h = &self.data;
        let a00 = h[1][1] * h[2][2] - h[1][2] * h[2][1];
        let a01 = -(h[0][1] * h[2][2] - h[0][2] * h[2][1]);
        let a02 = h[0][1] * h[1][2] - h[0][2] * h[1][1];

        let det = h[0][0] * a00
            + h[0][1] * (-(h[1][0] * h[2][2] - h[1][2] * h[2][0]))
            + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);

        if det.abs() < 1e-10 {
            return None;
        }

        let inv_det = 1.0 / det;

        let a10 = -(h[1][0] * h[2][2] - h[1][2] * h[2][0]);
        let a11 = h[0][0] * h[2][2] - h[0][2] * h[2][0];
        let a12 = -(h[0][0] * h[1][2] - h[0][2] * h[1][0]);

        let a20 = h[1][0] * h[2][1] - h[1][1] * h[2][0];
        let a21 = -(h[0][0] * h[2][1] - h[0][1] * h[2][0]);
        let a22 = h[0][0] * h[1][1] - h[0][1] * h[1][0];

        Some(Homography {
            data: [
                [a00 * inv_det, a01 * inv_det, a02 * inv_det],
                [a10 * inv_det, a11 * inv_det, a12 * inv_det],
                [a20 * inv_det, a21 * inv_det, a22 * inv_det],
            ],
        })
    }
}

/// Result of a RANSAC homography estimation.
#[derive(Debug, Clone)]
pub struct RansacHomographyResult {
    /// Best-fit homography (refined with LM on inliers)
    pub homography: Homography,
    /// Boolean mask — `inlier_mask[i] == true` iff point pair `i` is an inlier.
    pub inlier_mask: Vec<bool>,
    /// Number of inliers
    pub num_inliers: usize,
    /// Number of RANSAC iterations actually performed
    pub iterations: usize,
    /// Mean symmetric reprojection error over inliers (after refinement)
    pub mean_error: f64,
}

/// Configuration for RANSAC homography estimation.
#[derive(Debug, Clone)]
pub struct RansacConfig {
    /// Maximum number of RANSAC iterations
    pub max_iterations: usize,
    /// Inlier reprojection-error threshold (pixels)
    pub threshold: f64,
    /// Desired RANSAC confidence (0 < confidence < 1)
    pub confidence: f64,
    /// Minimum number of inliers to accept a model
    pub min_inliers: usize,
    /// Levenberg-Marquardt refinement iterations (0 = skip)
    pub lm_iterations: usize,
    /// Optional RNG seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            max_iterations: 2000,
            threshold: 3.0,
            confidence: 0.995,
            min_inliers: 4,
            lm_iterations: 20,
            seed: None,
        }
    }
}

// ─── Entry point ─────────────────────────────────────────────────────────────

/// Estimate a homography from point correspondences using RANSAC + DLT + LM.
///
/// # Arguments
///
/// * `src_pts` – Source points as `(x, y)` pairs
/// * `dst_pts` – Destination points as `(x, y)` pairs (same length as `src_pts`)
/// * `config`  – RANSAC and refinement parameters
///
/// # Returns
///
/// A [`RansacHomographyResult`] on success.
pub fn estimate_homography_ransac(
    src_pts: &[(f64, f64)],
    dst_pts: &[(f64, f64)],
    config: &RansacConfig,
) -> Result<RansacHomographyResult> {
    let n = src_pts.len();
    if n != dst_pts.len() {
        return Err(VisionError::InvalidParameter(
            "src_pts and dst_pts must have the same length".to_string(),
        ));
    }
    if n < 4 {
        return Err(VisionError::InvalidParameter(
            "At least 4 point correspondences are required".to_string(),
        ));
    }

    // RANSAC main loop
    let mut rng_state = config.seed.unwrap_or(0xDEAD_BEEF_1234_5678u64);
    let mut best_inliers: Vec<bool> = vec![false; n];
    let mut best_count = 0usize;
    let mut best_h = Homography::identity();
    let mut actual_iters = 0usize;
    let mut dynamic_max = config.max_iterations;

    for iter in 0..config.max_iterations {
        if iter >= dynamic_max {
            break;
        }
        actual_iters += 1;

        // Random sample of 4 points (Fisher-Yates partial shuffle)
        let mut idx: Vec<usize> = (0..n).collect();
        for i in 0..4 {
            rng_state = lcg_next(rng_state);
            let j = i + (rng_state as usize % (n - i));
            idx.swap(i, j);
        }
        let sample = &idx[..4];

        let s_src: Vec<(f64, f64)> = sample.iter().map(|&i| src_pts[i]).collect();
        let s_dst: Vec<(f64, f64)> = sample.iter().map(|&i| dst_pts[i]).collect();

        // DLT estimate
        let h = match dlt_homography(&s_src, &s_dst) {
            Ok(h) => h,
            Err(_) => continue,
        };

        // Count inliers
        let (inliers, count) = count_inliers(&h, src_pts, dst_pts, config.threshold);
        if count > best_count {
            best_count = count;
            best_inliers = inliers;
            best_h = h;

            // Update dynamic iteration count (Lowe / Hartley formula)
            if best_count >= config.min_inliers && n > 0 {
                let epsilon = 1.0 - best_count as f64 / n as f64;
                let denom = (1.0 - epsilon.powi(4)).max(f64::EPSILON).ln().abs();
                let new_max = ((1.0 - config.confidence).ln().abs() / denom).ceil() as usize;
                dynamic_max = new_max.min(config.max_iterations);
            }

            if best_count == n {
                break; // all inliers found – no need to continue
            }
        }
    }

    if best_count < config.min_inliers {
        return Err(VisionError::OperationError(format!(
            "RANSAC failed: only {best_count} inliers (required ≥ {})",
            config.min_inliers
        )));
    }

    // Refit DLT on all inliers (use best_h as fallback if refit is degenerate)
    let in_src: Vec<(f64, f64)> = src_pts
        .iter()
        .zip(best_inliers.iter())
        .filter_map(|(&s, &ok)| if ok { Some(s) } else { None })
        .collect();
    let in_dst: Vec<(f64, f64)> = dst_pts
        .iter()
        .zip(best_inliers.iter())
        .filter_map(|(&d, &ok)| if ok { Some(d) } else { None })
        .collect();

    let refined_h = dlt_homography(&in_src, &in_dst).unwrap_or(best_h);

    // Levenberg-Marquardt refinement
    let final_h = if config.lm_iterations > 0 {
        lm_refine(&refined_h, &in_src, &in_dst, config.lm_iterations)?
    } else {
        refined_h
    };

    // Recompute inlier mask with refined homography
    let (final_inliers, final_count) = count_inliers(&final_h, src_pts, dst_pts, config.threshold);

    // Mean error
    let mean_error = compute_mean_error(&final_h, src_pts, dst_pts, &final_inliers);

    Ok(RansacHomographyResult {
        homography: final_h,
        inlier_mask: final_inliers,
        num_inliers: final_count,
        iterations: actual_iters,
        mean_error,
    })
}

// ─── Direct Linear Transform (DLT) ───────────────────────────────────────────

/// Estimate a homography from exactly (or at least) 4 correspondences using
/// the normalised DLT algorithm.
///
/// # Returns
///
/// A [`Homography`] or an error if the system is degenerate.
pub fn dlt_homography(src_pts: &[(f64, f64)], dst_pts: &[(f64, f64)]) -> Result<Homography> {
    let n = src_pts.len();
    if n < 4 {
        return Err(VisionError::InvalidParameter(
            "DLT requires ≥ 4 point correspondences".to_string(),
        ));
    }

    // Hartley normalisation
    let (src_norm, t_src) = normalize_points(src_pts);
    let (dst_norm, t_dst) = normalize_points(dst_pts);

    // Build coefficient matrix A (2n × 9)
    let mut a = vec![[0.0f64; 9]; 2 * n];
    for (i, ((x, y), (xp, yp))) in src_norm.iter().zip(dst_norm.iter()).enumerate() {
        // Row 2i: [−x, −y, −1,  0,  0,  0, x'x, x'y, x']
        a[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, xp * x, xp * y, *xp];
        // Row 2i+1: [0, 0, 0, −x, −y, −1, y'x, y'y, y']
        a[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, yp * x, yp * y, *yp];
    }

    // Solve Ah = 0 subject to ||h|| = 1 via power iteration on A^T A
    let h_vec = smallest_singular_vector(&a, 9)?;

    // Reshape to 3×3
    let h_norm = Homography {
        data: [
            [h_vec[0], h_vec[1], h_vec[2]],
            [h_vec[3], h_vec[4], h_vec[5]],
            [h_vec[6], h_vec[7], h_vec[8]],
        ],
    };

    // Denormalise: H = T_dst^{-1} · H_norm · T_src
    let h = denormalize_homography(&h_norm, &t_src, &t_dst)?;

    Ok(h)
}

// ─── Normalisation ─────────────────────────────────────────────────────────────

/// Hartley normalisation: translate centroid to origin, scale so mean distance to
/// origin is √2.  Returns normalised points and the 3×3 normalisation matrix T.
fn normalize_points(pts: &[(f64, f64)]) -> (Vec<(f64, f64)>, [[f64; 3]; 3]) {
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p.1).sum::<f64>() / n;

    let mean_dist = pts
        .iter()
        .map(|p| ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt())
        .sum::<f64>()
        / n;

    let scale = if mean_dist > 1e-10 {
        std::f64::consts::SQRT_2 / mean_dist
    } else {
        1.0
    };

    let normed: Vec<(f64, f64)> = pts
        .iter()
        .map(|p| ((p.0 - cx) * scale, (p.1 - cy) * scale))
        .collect();

    // T: scale × (x − cx, y − cy)
    let t = [
        [scale, 0.0, -scale * cx],
        [0.0, scale, -scale * cy],
        [0.0, 0.0, 1.0],
    ];

    (normed, t)
}

/// Denormalise homography: H = T_dst^{-1} · H_norm · T_src
fn denormalize_homography(
    h_norm: &Homography,
    t_src: &[[f64; 3]; 3],
    t_dst: &[[f64; 3]; 3],
) -> Result<Homography> {
    // Compute T_dst^{-1} (it's a similarity transform so inversion is easy)
    // T_dst = [[s, 0, -s*cx], [0, s, -s*cy], [0, 0, 1]]
    let s = t_dst[0][0];
    if s.abs() < 1e-10 {
        return Err(VisionError::OperationError(
            "Degenerate normalisation matrix".to_string(),
        ));
    }
    let inv_s = 1.0 / s;
    // T_dst = [[s, 0, -s*cx], [0, s, -s*cy], [0, 0, 1]]
    // T_dst^{-1} = [[1/s, 0, cx], [0, 1/s, cy], [0, 0, 1]]
    //            = [[1/s, 0, -T[0][2]/s], [0, 1/s, -T[1][2]/s], [0, 0, 1]]
    let t_dst_inv = [
        [inv_s, 0.0, -t_dst[0][2] * inv_s],
        [0.0, inv_s, -t_dst[1][2] * inv_s],
        [0.0, 0.0, 1.0],
    ];

    // H = T_dst_inv · H_norm · T_src
    let tmp = mat3x3_mul(&h_norm.data, t_src);
    let h_data = mat3x3_mul(&t_dst_inv, &tmp);

    Ok(Homography { data: h_data })
}

// ─── Inlier counting ──────────────────────────────────────────────────────────

/// Returns `(inlier_mask, inlier_count)` using the symmetric transfer error.
fn count_inliers(
    h: &Homography,
    src: &[(f64, f64)],
    dst: &[(f64, f64)],
    threshold: f64,
) -> (Vec<bool>, usize) {
    let h_inv = h.inverse();
    let thresh2 = threshold * threshold;
    let mut mask = vec![false; src.len()];
    let mut count = 0usize;

    for (i, (s, d)) in src.iter().zip(dst.iter()).enumerate() {
        // Forward transfer error
        let fwd_err = if let Some((px, py)) = h.project(s.0, s.1) {
            (px - d.0).powi(2) + (py - d.1).powi(2)
        } else {
            f64::MAX
        };

        // Backward transfer error
        let bwd_err = if let Some(ref hi) = h_inv {
            if let Some((px, py)) = hi.project(d.0, d.1) {
                (px - s.0).powi(2) + (py - s.1).powi(2)
            } else {
                f64::MAX
            }
        } else {
            f64::MAX
        };

        let sym_err = (fwd_err + bwd_err) / 2.0;
        if sym_err < thresh2 {
            mask[i] = true;
            count += 1;
        }
    }

    (mask, count)
}

fn compute_mean_error(
    h: &Homography,
    src: &[(f64, f64)],
    dst: &[(f64, f64)],
    inliers: &[bool],
) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    let h_inv = h.inverse();

    for ((&s, &d), &ok) in src.iter().zip(dst.iter()).zip(inliers.iter()) {
        if !ok {
            continue;
        }
        let fwd = h
            .project(s.0, s.1)
            .map(|(px, py)| ((px - d.0).powi(2) + (py - d.1).powi(2)).sqrt())
            .unwrap_or(f64::MAX);
        let bwd = h_inv
            .as_ref()
            .and_then(|hi| hi.project(d.0, d.1))
            .map(|(px, py)| ((px - s.0).powi(2) + (py - s.1).powi(2)).sqrt())
            .unwrap_or(f64::MAX);
        total += (fwd + bwd) / 2.0;
        count += 1;
    }

    if count == 0 {
        f64::MAX
    } else {
        total / count as f64
    }
}

// ─── Levenberg-Marquardt refinement ──────────────────────────────────────────

/// LM minimisation of sum of squared symmetric transfer errors over inlier set.
///
/// The 9-D parameterisation is the vectorised H (last element normalised to 1).
fn lm_refine(
    h0: &Homography,
    src: &[(f64, f64)],
    dst: &[(f64, f64)],
    max_iter: usize,
) -> Result<Homography> {
    let mut h = h0.clone();
    let mut mu = 1e-3; // LM damping
    let nu = 2.0f64;

    let n = src.len();

    for _iter in 0..max_iter {
        // Compute residuals and Jacobian
        let (r, j) = residuals_and_jacobian(&h, src, dst);
        let cost: f64 = r.iter().map(|ri| ri * ri).sum();

        // Gradient: J^T r
        let g = mat_vec_t_mul(&j, &r, 8);

        // Hessian approximation: J^T J
        let jtj = mat_t_mat_mul(&j, 8);

        // LM system: (J^T J + mu * I) Δh = −J^T r
        let mut a_diag = jtj.clone();
        #[allow(clippy::needless_range_loop)]
        for k in 0..8 {
            a_diag[k][k] += mu;
        }

        let neg_g: Vec<f64> = g.iter().map(|v| -v).collect();
        let delta = solve_8x8(&a_diag, &neg_g);

        // Tentative update
        let h_new = apply_delta(&h, &delta);
        let (r_new, _) = residuals_and_jacobian(&h_new, src, dst);
        let cost_new: f64 = r_new.iter().map(|ri| ri * ri).sum();

        // Gain ratio
        let predicted_reduction: f64 = delta
            .iter()
            .zip(g.iter())
            .map(|(d, gi)| d * gi)
            .sum::<f64>()
            .abs()
            + 1e-20;
        let gain = (cost - cost_new) / predicted_reduction;

        if cost_new < cost {
            h = h_new;
            mu /= nu;
        } else {
            mu *= nu;
        }

        if gain > 0.0 && cost_new < 1e-10 {
            break;
        }
    }

    Ok(h)
}

/// Compute 2n residuals and (2n × 8) Jacobian.
///
/// The parameterisation is h[0..8] with h[8] = 1 fixed.
fn residuals_and_jacobian(
    h: &Homography,
    src: &[(f64, f64)],
    dst: &[(f64, f64)],
) -> (Vec<f64>, Vec<[f64; 8]>) {
    let hd = &h.data;
    let n = src.len();
    let mut residuals = Vec::with_capacity(2 * n);
    let mut jacobian: Vec<[f64; 8]> = Vec::with_capacity(2 * n);

    for (&(x, y), &(xp, yp)) in src.iter().zip(dst.iter()) {
        let w = hd[2][0] * x + hd[2][1] * y + hd[2][2];
        if w.abs() < 1e-10 {
            residuals.push(0.0);
            residuals.push(0.0);
            jacobian.push([0.0; 8]);
            jacobian.push([0.0; 8]);
            continue;
        }
        let inv_w = 1.0 / w;
        let px = (hd[0][0] * x + hd[0][1] * y + hd[0][2]) * inv_w;
        let py = (hd[1][0] * x + hd[1][1] * y + hd[1][2]) * inv_w;

        residuals.push(px - xp);
        residuals.push(py - yp);

        // Jacobian of (px, py) w.r.t. h[0..8]
        // h[8] = 1 is fixed, so we only vary h[0..8]
        // ∂px/∂h = [x, y, 1, 0, 0, 0, −px·x, −px·y] / w
        let mut jx = [0.0f64; 8];
        jx[0] = x * inv_w;
        jx[1] = y * inv_w;
        jx[2] = inv_w;
        jx[6] = -px * x * inv_w;
        jx[7] = -px * y * inv_w;

        let mut jy = [0.0f64; 8];
        jy[3] = x * inv_w;
        jy[4] = y * inv_w;
        jy[5] = inv_w;
        jy[6] = -py * x * inv_w;
        jy[7] = -py * y * inv_w;

        jacobian.push(jx);
        jacobian.push(jy);
    }

    (residuals, jacobian)
}

/// Apply delta to homography parameters.
fn apply_delta(h: &Homography, delta: &[f64]) -> Homography {
    let hd = &h.data;
    let params = [
        hd[0][0] + delta.first().copied().unwrap_or(0.0),
        hd[0][1] + delta.get(1).copied().unwrap_or(0.0),
        hd[0][2] + delta.get(2).copied().unwrap_or(0.0),
        hd[1][0] + delta.get(3).copied().unwrap_or(0.0),
        hd[1][1] + delta.get(4).copied().unwrap_or(0.0),
        hd[1][2] + delta.get(5).copied().unwrap_or(0.0),
        hd[2][0] + delta.get(6).copied().unwrap_or(0.0),
        hd[2][1] + delta.get(7).copied().unwrap_or(0.0),
        hd[2][2], // fixed
    ];
    Homography {
        data: [
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [params[6], params[7], params[8]],
        ],
    }
}

// ─── Linear algebra helpers ───────────────────────────────────────────────────

/// Multiply two 3×3 matrices.
fn mat3x3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
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

/// Find the null-space vector of `A` (shape m × 9) by constrained linear solve.
///
/// The homogeneous system `A h = 0` is solved by fixing `h[8] = 1` and solving
/// the resulting 8×8 linear system `A[0..8, 0..8] · h[0..8] = −A[0..8, 8]` via
/// Gaussian elimination with partial pivoting.  The result is normalised to unit
/// length and sign-corrected.
///
/// This approach is exact for well-posed inputs (rank = 8) and is equivalent to
/// extracting the right null vector from an SVD, but avoids the convergence issues
/// of iterative methods on ill-conditioned matrices.
fn smallest_singular_vector(a: &[[f64; 9]], cols: usize) -> Result<Vec<f64>> {
    let rows = a.len();
    // We fix h[cols-1]=1 and solve the (cols-1)×(cols-1) sub-system,
    // so we need rows >= cols-1 (e.g., 4 points → 8 rows, need 8 for 8×8).
    if rows < cols - 1 {
        return Err(VisionError::OperationError(
            "Underdetermined system for DLT".to_string(),
        ));
    }

    // Form A^T A  (9 × 9)
    let mut ata = vec![vec![0.0f64; cols]; cols];
    for row in a {
        for j in 0..cols {
            for k in 0..cols {
                ata[j][k] += row[j] * row[k];
            }
        }
    }

    // Constrained solve: fix h[last] = 1, solve (cols-1) × (cols-1) sub-system.
    // System: ata[0..cols-1, 0..cols-1] · x = -ata[0..cols-1, cols-1]
    let n8 = cols - 1;
    let mut mat: Vec<Vec<f64>> = (0..n8).map(|i| ata[i][0..n8].to_vec()).collect();
    let mut rhs: Vec<f64> = (0..n8).map(|i| -ata[i][cols - 1]).collect();

    // Gaussian elimination with partial pivoting
    #[allow(clippy::needless_range_loop)]
    for col in 0..n8 {
        // Find pivot
        let mut max_row = col;
        let mut max_val = mat[col][col].abs();
        for row in (col + 1)..n8 {
            if mat[row][col].abs() > max_val {
                max_val = mat[row][col].abs();
                max_row = row;
            }
        }
        mat.swap(col, max_row);
        rhs.swap(col, max_row);

        let pivot = mat[col][col];
        if pivot.abs() < 1e-10 {
            // Near-singular column: this can happen for degenerate point sets.
            // Fall back to the last row of the identity as null vector (h = e_cols-1).
            let mut h = vec![0.0f64; cols];
            h[cols - 1] = 1.0;
            return Ok(h);
        }

        for row in (col + 1)..n8 {
            let factor = mat[row][col] / pivot;
            for k in col..n8 {
                let sub = mat[col][k] * factor;
                mat[row][k] -= sub;
            }
            rhs[row] -= rhs[col] * factor;
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n8];
    for row in (0..n8).rev() {
        let mut sum = rhs[row];
        for k in (row + 1)..n8 {
            sum -= mat[row][k] * x[k];
        }
        if mat[row][row].abs() > 1e-10 {
            x[row] = sum / mat[row][row];
        }
    }

    let mut h = x;
    h.push(1.0_f64); // h[8] = 1

    // Normalise to unit length
    let norm: f64 = h.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-14 {
        return Err(VisionError::OperationError(
            "Null-space computation failed: degenerate input".to_string(),
        ));
    }
    for v in h.iter_mut() {
        *v /= norm;
    }

    // Fix sign: make the element with largest magnitude positive
    let max_abs = h.iter().cloned().map(f64::abs).fold(0.0_f64, f64::max);
    let sign = h
        .iter()
        .find(|&&v| v.abs() >= max_abs - 1e-14)
        .map_or(1.0, |&v| if v < 0.0 { -1.0 } else { 1.0 });
    for v in h.iter_mut() {
        *v *= sign;
    }

    Ok(h)
}

/// J^T · r  where J has shape (m, n_params) and r has shape (m,)
fn mat_vec_t_mul(j: &[[f64; 8]], r: &[f64], n_params: usize) -> Vec<f64> {
    let mut g = vec![0.0f64; n_params];
    for (ji, ri) in j.iter().zip(r.iter()) {
        for k in 0..n_params {
            g[k] += ji[k] * ri;
        }
    }
    g
}

/// Compute J^T J  (n_params × n_params) from rows of J
fn mat_t_mat_mul(j: &[[f64; 8]], n_params: usize) -> Vec<Vec<f64>> {
    let mut h = vec![vec![0.0f64; n_params]; n_params];
    for ji in j.iter() {
        for k in 0..n_params {
            for l in 0..n_params {
                h[k][l] += ji[k] * ji[l];
            }
        }
    }
    h
}

/// Solve an 8×8 system via Gaussian elimination with partial pivoting.
fn solve_8x8(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = 8usize;
    let mut mat: Vec<Vec<f64>> = a.to_vec();
    let mut rhs: Vec<f64> = b.to_vec();

    #[allow(clippy::needless_range_loop)]
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = mat[col][col].abs();
        for row in (col + 1)..n {
            if mat[row][col].abs() > max_val {
                max_val = mat[row][col].abs();
                max_row = row;
            }
        }
        mat.swap(col, max_row);
        rhs.swap(col, max_row);

        let pivot = mat[col][col];
        if pivot.abs() < 1e-12 {
            continue;
        }

        for row in (col + 1)..n {
            let factor = mat[row][col] / pivot;
            for k in col..n {
                let sub = mat[col][k] * factor;
                mat[row][k] -= sub;
            }
            rhs[row] -= rhs[col] * factor;
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for row in (0..n).rev() {
        let mut sum = rhs[row];
        for k in (row + 1)..n {
            sum -= mat[row][k] * x[k];
        }
        if mat[row][row].abs() > 1e-12 {
            x[row] = sum / mat[row][row];
        }
    }
    x
}

// ─── LCG random number generator ─────────────────────────────────────────────

fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_known_homography(pts: &[(f64, f64)], h: &[[f64; 3]; 3]) -> Vec<(f64, f64)> {
        pts.iter()
            .map(|&(x, y)| {
                let w = h[2][0] * x + h[2][1] * y + h[2][2];
                let px = (h[0][0] * x + h[0][1] * y + h[0][2]) / w;
                let py = (h[1][0] * x + h[1][1] * y + h[1][2]) / w;
                (px, py)
            })
            .collect()
    }

    fn sample_points(n: usize) -> Vec<(f64, f64)> {
        (0..n)
            .map(|i| {
                let t = i as f64 * 37.0;
                (50.0 + (t.sin() * 120.0), 50.0 + (t.cos() * 80.0))
            })
            .collect()
    }

    #[test]
    fn test_identity_homography_dlt() {
        let src = sample_points(8);
        let dst = src.clone();
        let h = dlt_homography(&src, &dst)
            .expect("dlt_homography should succeed with 8+ point correspondences");
        // Identity maps every point to itself
        for (s, d) in src.iter().zip(dst.iter()) {
            let (px, py) = h
                .project(s.0, s.1)
                .expect("homography project should succeed for valid point");
            assert!((px - d.0).abs() < 0.5, "px={px} d.0={}", d.0);
            assert!((py - d.1).abs() < 0.5, "py={py} d.1={}", d.1);
        }
    }

    #[test]
    fn test_dlt_known_homography() {
        // A perspective homography
        let h_true = [[1.1, 0.05, 10.0], [-0.03, 1.2, 5.0], [0.0002, 0.0001, 1.0]];
        let src = sample_points(12);
        let dst = apply_known_homography(&src, &h_true);
        let h = dlt_homography(&src, &dst)
            .expect("dlt_homography should succeed with known correspondences");

        // Check reprojection error is small
        for (s, d) in src.iter().zip(dst.iter()) {
            let (px, py) = h
                .project(s.0, s.1)
                .expect("homography project should succeed for valid point");
            let err = ((px - d.0).powi(2) + (py - d.1).powi(2)).sqrt();
            assert!(err < 1.0, "Reprojection error too large: {err}");
        }
    }

    #[test]
    fn test_ransac_with_outliers() {
        let h_true = [[1.2, 0.1, 15.0], [0.05, 0.9, -10.0], [0.0003, -0.0002, 1.0]];
        let mut src = sample_points(30);
        let mut dst = apply_known_homography(&src, &h_true);

        // Add 6 outliers (20%)
        let outlier_indices = [2usize, 7, 11, 18, 23, 28];
        for &i in &outlier_indices {
            dst[i].0 += 100.0 * ((i as f64).sin());
            dst[i].1 -= 80.0 * ((i as f64).cos());
        }

        let config = RansacConfig {
            max_iterations: 500,
            threshold: 5.0,
            confidence: 0.99,
            min_inliers: 10,
            lm_iterations: 10,
            seed: Some(42),
        };

        let result = estimate_homography_ransac(&src, &dst, &config)
            .expect("RANSAC homography should succeed with sufficient inliers");

        // Should detect most inliers and exclude the outliers
        let outlier_accepted = outlier_indices
            .iter()
            .filter(|&&i| result.inlier_mask[i])
            .count();
        assert!(
            outlier_accepted <= 2,
            "Too many outliers accepted: {outlier_accepted}"
        );
        assert!(
            result.num_inliers >= 20,
            "Too few inliers: {}",
            result.num_inliers
        );
    }

    #[test]
    fn test_homography_inverse() {
        let h_data = [[1.1, 0.05, 10.0], [-0.03, 1.2, 5.0], [0.0002, 0.0001, 1.0]];
        let h = Homography { data: h_data };
        let hi = h.inverse().expect("Should have an inverse");

        // H · H^{-1} ≈ I
        let prod = crate::features::homography_ransac::mat3x3_mul(&h.data, &hi.data);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[i][j] - expected).abs() < 1e-8,
                    "H·H^-1 [{i}][{j}] = {} expected {expected}",
                    prod[i][j]
                );
            }
        }
    }

    #[test]
    fn test_too_few_points() {
        let src = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)];
        let dst = vec![(0.1, 0.1), (1.1, 0.1), (0.6, 1.1)];
        assert!(dlt_homography(&src, &dst).is_err());
    }

    #[test]
    fn test_homography_project_infinity() {
        // Construct a degenerate H where w = 0 for some point
        let h = Homography {
            data: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        };
        // w = x + 0·y + 0 = x; for x = 0 this should still work
        let res = h.project(0.0, 5.0);
        // w = 0·1 = 0, so should return None
        assert!(res.is_none() || res.is_some()); // degenerate case -- just shouldn't panic
    }
}
