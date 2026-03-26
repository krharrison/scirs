//! Procrustes analysis for aligning geometric configurations.
//!
//! ## Overview
//!
//! Procrustes analysis finds the optimal orthogonal transformation (rotation and
//! optionally reflection and scaling) that maps one matrix onto another in the
//! Frobenius-norm sense.
//!
//! ### Orthogonal Procrustes Problem
//!
//! Given matrices **A** (n × d) and **B** (n × d), find:
//!
//! ```text
//! min_{R: Rᵀ R = I}  ||s · A R + 1 tᵀ − B||_F
//! ```
//!
//! **Solution via SVD** of Bᵀ A = U Σ Vᵀ:
//! - R = V Uᵀ  (or V diag(1,…,det(VUᵀ)) Uᵀ to prevent reflections)
//! - Optimal scale s = trace(Σ) / ||A||_F²  (when centering and scaling enabled)
//!
//! ### Generalized Procrustes Analysis
//!
//! Aligns multiple matrices to a common mean (consensus) shape via iterative
//! pairwise Procrustes alignment, similar to the GPA algorithm of Gower (1975).
//!
//! ## References
//!
//! - Schönemann (1966): A generalized solution of the orthogonal Procrustes problem
//! - Gower (1975): Generalized Procrustes analysis
//! - Golub & Van Loan (1996): Matrix Computations, §12.4

use scirs2_core::ndarray::{Array1, Array2, Axis};

use crate::error::{Result, TransformError};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Procrustes alignment.
#[derive(Debug, Clone)]
pub struct ProcrustesConfig {
    /// Allow reflections (orthogonal group O(d)) in addition to rotations SO(d).
    /// Default: `false` (rotation only, det(R) = +1).
    pub allow_reflection: bool,
    /// Find the optimal isotropic scale factor.
    /// Default: `true`.
    pub scaling: bool,
    /// Center both matrices before solving.
    /// Default: `true`.
    pub centering: bool,
}

impl Default for ProcrustesConfig {
    fn default() -> Self {
        Self {
            allow_reflection: false,
            scaling: true,
            centering: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result of a Procrustes alignment.
#[derive(Debug, Clone)]
pub struct ProcrustesResult {
    /// Optimal orthogonal rotation matrix R (d × d), with det(R) = +1 unless
    /// `allow_reflection = true`.
    pub rotation: Array2<f64>,
    /// Optimal isotropic scale factor s (1.0 when `scaling = false`).
    pub scale: f64,
    /// Translation vector t (d-dimensional) applied *after* rotation.
    pub translation: Array1<f64>,
    /// Frobenius-norm residual ‖s·A·R + 1·tᵀ − B‖_F after alignment.
    pub disparity: f64,
    /// Aligned version of A: s·(A_centred · R) + centroid_B.
    pub transformed: Array2<f64>,
}

// ---------------------------------------------------------------------------
// Orthogonal Procrustes
// ---------------------------------------------------------------------------

/// Solve the orthogonal Procrustes problem.
///
/// Finds the best-fitting orthogonal transformation (rotation, optional scale,
/// and translation) mapping **A** onto **B**:
///
/// ```text
/// min_{R: Rᵀ R = I, s > 0, t}  ||s · A R + 1 tᵀ − B||_F
/// ```
///
/// # Arguments
/// * `a`      – Source matrix (n × d).
/// * `b`      – Target matrix (n × d).
/// * `config` – Alignment options.
///
/// # Errors
/// Returns [`TransformError::InvalidInput`] when shapes are incompatible, or
/// [`TransformError::ComputationError`] on numerical failure.
///
/// # Example
/// ```rust
/// use scirs2_transform::alignment::procrustes::{orthogonal_procrustes, ProcrustesConfig};
/// use scirs2_core::ndarray::array;
///
/// // A 90° rotation of a simple triangle
/// let a = array![[1.0_f64, 0.0], [0.0, 1.0], [0.0, 0.0]];
/// let b = array![[0.0_f64, 1.0], [-1.0, 0.0], [0.0, 0.0]];
/// let config = ProcrustesConfig { scaling: false, ..Default::default() };
/// let result = orthogonal_procrustes(&a, &b, &config).expect("should succeed");
/// assert!(result.disparity < 1e-6);
/// ```
pub fn orthogonal_procrustes(
    a: &Array2<f64>,
    b: &Array2<f64>,
    config: &ProcrustesConfig,
) -> Result<ProcrustesResult> {
    let (n, d) = a.dim();
    if b.dim() != (n, d) {
        return Err(TransformError::InvalidInput(format!(
            "Shape mismatch: A is ({n}×{d}) but B is ({}×{})",
            b.nrows(),
            b.ncols()
        )));
    }
    if n == 0 || d == 0 {
        return Err(TransformError::InvalidInput(
            "Matrices must be non-empty".to_string(),
        ));
    }

    // ----------------------------------------------------------------
    // 1. Center both matrices
    // ----------------------------------------------------------------
    let centroid_a: Array1<f64> = if config.centering {
        a.mean_axis(Axis(0)).ok_or_else(|| {
            TransformError::ComputationError("Failed to compute centroid of A".to_string())
        })?
    } else {
        Array1::zeros(d)
    };

    let centroid_b: Array1<f64> = if config.centering {
        b.mean_axis(Axis(0)).ok_or_else(|| {
            TransformError::ComputationError("Failed to compute centroid of B".to_string())
        })?
    } else {
        Array1::zeros(d)
    };

    // Centered matrices
    let a_c: Array2<f64> = a - &centroid_a.view().insert_axis(Axis(0));
    let b_c: Array2<f64> = b - &centroid_b.view().insert_axis(Axis(0));

    // ----------------------------------------------------------------
    // 2. Frobenius norm of centered A
    // ----------------------------------------------------------------
    let norm_a_sq: f64 = a_c.iter().map(|&x| x * x).sum();

    if norm_a_sq < f64::EPSILON {
        // A is (approximately) zero — can't define rotation; return identity
        let rotation = Array2::eye(d);
        let translation = centroid_b.clone();
        let zeros_plus_cb: Array2<f64> =
            Array2::from_shape_fn((n, d), |_| 0.0) + &centroid_b.view().insert_axis(Axis(0));
        let disparity = b_c.iter().map(|&x| x * x).sum::<f64>().sqrt();
        return Ok(ProcrustesResult {
            rotation,
            scale: 1.0,
            translation,
            disparity,
            transformed: zeros_plus_cb,
        });
    }

    // ----------------------------------------------------------------
    // 3. Compute M = Bᵀ A  (d × d)
    //    The Procrustes solution uses SVD of M = B_cᵀ A_c
    // ----------------------------------------------------------------
    let m = b_c.t().dot(&a_c); // d × d

    // ----------------------------------------------------------------
    // 4. SVD of M: M = U Σ Vᵀ  using Jacobi SVD
    // ----------------------------------------------------------------
    let (u_mat, sigma_vec, vt_mat) = jacobi_svd_square(&m)?;
    // u_mat : d×d,  sigma_vec: d,  vt_mat: d×d  (rows are right singular vectors)
    // So M = U diag(σ) Vᵀ

    // ----------------------------------------------------------------
    // 5. Construct candidate R = V Uᵀ
    // ----------------------------------------------------------------
    let v_mat = vt_mat.t().to_owned(); // V: d×d  (columns are right singular vectors)
    let ut_mat = u_mat.t().to_owned(); // Uᵀ: d×d
    let mut r = v_mat.dot(&ut_mat); // R = V Uᵀ

    // ----------------------------------------------------------------
    // 6. Enforce det(R) = +1 if reflections are not allowed
    // ----------------------------------------------------------------
    if !config.allow_reflection {
        let det_r = mat_det(&r);
        if det_r < 0.0 {
            // Flip sign of the last column of V (associated with smallest σ)
            // so that det(R) = +1: R = V diag(1,…,1,−1) Uᵀ
            let mut v_adj = v_mat.clone();
            for row in 0..d {
                v_adj[[row, d - 1]] *= -1.0;
            }
            r = v_adj.dot(&ut_mat);
        }
    }

    // ----------------------------------------------------------------
    // 7. Optimal scale  s = trace(Σ_adj) / ‖A_c‖²_F
    //    Σ_adj accounts for the possible sign-flip of the last singular value.
    // ----------------------------------------------------------------
    let (scale, _sigma_trace) = if config.scaling {
        let sigma_sum_raw: f64 = sigma_vec.iter().sum();
        // If we flipped the last singular value to fix det:
        let det_r = mat_det(&r);
        let sigma_adj = if !config.allow_reflection && det_r > 0.0 {
            // Check if we needed to flip (by comparing with raw det before flip)
            // The raw sigma_sum is correct if we flipped, need to subtract 2*sigma_last
            // But since `r` is already the corrected rotation, we re-check det
            // The correction happened above: if original det < 0, we flipped.
            // We always stored corrected `r`, so compare det of corrected r.
            // If det(r) = +1, no flip was needed OR flip was applied.
            // Easier: just recompute via V Uᵀ to see if flip happened.
            let r_uncorrected = v_mat.dot(&ut_mat);
            let det_uncorrected = mat_det(&r_uncorrected);
            if det_uncorrected < 0.0 && !config.allow_reflection {
                // Flip was applied → adjusted sigma
                sigma_sum_raw - 2.0 * sigma_vec[d - 1]
            } else {
                sigma_sum_raw
            }
        } else {
            sigma_sum_raw
        };
        let s = (sigma_adj / norm_a_sq).max(0.0);
        (s, sigma_adj)
    } else {
        (1.0, sigma_vec.iter().sum::<f64>())
    };

    // ----------------------------------------------------------------
    // 8. Translation: t = centroid_B − s · (centroid_A · R)
    // ----------------------------------------------------------------
    let ca_r: Array1<f64> = centroid_a
        .view()
        .insert_axis(Axis(0))
        .dot(&r)
        .row(0)
        .to_owned();
    let translation: Array1<f64> = &centroid_b - &(ca_r * scale);

    // ----------------------------------------------------------------
    // 9. Apply transformation: T(A) = s · A_c · R + centroid_B
    // ----------------------------------------------------------------
    let a_c_r = a_c.dot(&r);
    let transformed: Array2<f64> = a_c_r * scale + &centroid_b.view().insert_axis(Axis(0));

    // ----------------------------------------------------------------
    // 10. Disparity = ‖T(A) − B‖_F
    // ----------------------------------------------------------------
    let diff = &transformed - b;
    let disparity: f64 = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();

    Ok(ProcrustesResult {
        rotation: r,
        scale,
        translation,
        disparity,
        transformed,
    })
}

// ---------------------------------------------------------------------------
// Generalized Procrustes Analysis
// ---------------------------------------------------------------------------

/// Generalized Procrustes Analysis (GPA): align multiple matrices to a common mean.
///
/// Iteratively aligns each matrix to the current consensus (mean) shape using
/// [`orthogonal_procrustes`] until convergence or `max_iter` is reached.
///
/// # Arguments
/// * `matrices` – Slice of matrices, each (n × d), representing the same n landmarks.
/// * `max_iter` – Maximum number of GPA sweeps. Default suggestion: 100.
/// * `tol`      – Convergence tolerance on the total disparity change. Default: 1e-8.
///
/// # Returns
/// One [`ProcrustesResult`] per input matrix (aligned to consensus).
///
/// # Errors
/// Returns an error if fewer than 2 matrices are provided or shapes differ.
pub fn generalized_procrustes(
    matrices: &[Array2<f64>],
    max_iter: usize,
    tol: f64,
) -> Result<Vec<ProcrustesResult>> {
    let k = matrices.len();
    if k < 2 {
        return Err(TransformError::InvalidInput(
            "Generalized Procrustes requires at least 2 matrices".to_string(),
        ));
    }

    let (n, d) = matrices[0].dim();
    for (idx, m) in matrices.iter().enumerate() {
        if m.dim() != (n, d) {
            return Err(TransformError::InvalidInput(format!(
                "Matrix {idx} has shape ({},{}) but expected ({n},{d})",
                m.nrows(),
                m.ncols()
            )));
        }
    }

    let config = ProcrustesConfig {
        allow_reflection: false,
        scaling: true,
        centering: true,
    };

    // Initialise: copies of original matrices as "aligned" versions
    let mut aligned: Vec<Array2<f64>> = matrices.to_vec();

    let mut prev_disparity = f64::INFINITY;

    for _iter in 0..max_iter {
        // Compute consensus (mean shape)
        let consensus = compute_mean_shape(&aligned);

        // Align each matrix to the consensus
        let mut total_disparity = 0.0_f64;
        for m in aligned.iter_mut() {
            let result = orthogonal_procrustes(m, &consensus, &config)?;
            total_disparity += result.disparity;
            *m = result.transformed;
        }

        // Check convergence
        let change = (prev_disparity - total_disparity).abs();
        prev_disparity = total_disparity;
        if change < tol {
            break;
        }
    }

    // Final pass: compute ProcrustesResult for each original matrix against consensus
    let consensus = compute_mean_shape(&aligned);
    let mut results = Vec::with_capacity(k);
    for orig in matrices.iter() {
        let result = orthogonal_procrustes(orig, &consensus, &config)?;
        results.push(result);
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the element-wise mean of a collection of matrices.
fn compute_mean_shape(matrices: &[Array2<f64>]) -> Array2<f64> {
    let k = matrices.len() as f64;
    let (n, d) = matrices[0].dim();
    let mut mean = Array2::<f64>::zeros((n, d));
    for m in matrices {
        mean = mean + m;
    }
    mean / k
}

/// Jacobi one-sided SVD for a square d×d matrix.
///
/// Computes M = U Σ Vᵀ using Givens rotations on Mᵀ M (Golub-Reinsch variant).
/// Returns (U, σ, Vᵀ) where Vᵀ has rows that are the right singular vectors.
fn jacobi_svd_square(m: &Array2<f64>) -> Result<(Array2<f64>, Vec<f64>, Array2<f64>)> {
    let d = m.nrows();
    if m.ncols() != d {
        return Err(TransformError::ComputationError(
            "jacobi_svd_square requires square matrix".to_string(),
        ));
    }
    if d == 0 {
        return Err(TransformError::ComputationError(
            "jacobi_svd_square requires non-empty matrix".to_string(),
        ));
    }

    // Work on B = Mᵀ M (symmetric PSD), accumulate V
    let mut b = m.t().dot(m); // d×d
    let mut v = Array2::<f64>::eye(d);

    let max_sweeps = 200;
    let eps = 1e-14_f64;

    for _ in 0..max_sweeps {
        let mut converged = true;
        for p in 0..d {
            for q in (p + 1)..d {
                let bpq = b[[p, q]];
                if bpq.abs() < eps * (b[[p, p]].abs().max(b[[q, q]].abs()).max(1.0)) {
                    continue;
                }
                converged = false;

                // 2×2 Jacobi rotation to zero b[p,q]
                let bpp = b[[p, p]];
                let bqq = b[[q, q]];
                let tau = (bqq - bpp) / (2.0 * bpq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (tau - (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update diagonal first
                b[[p, p]] = bpp - t * bpq;
                b[[q, q]] = bqq + t * bpq;
                b[[p, q]] = 0.0;
                b[[q, p]] = 0.0;

                // Update off-diagonal elements
                for i in 0..d {
                    if i != p && i != q {
                        let bip = b[[i, p]];
                        let biq = b[[i, q]];
                        b[[i, p]] = c * bip - s * biq;
                        b[[i, q]] = s * bip + c * biq;
                        b[[p, i]] = b[[i, p]];
                        b[[q, i]] = b[[i, q]];
                    }
                }

                // Accumulate V: V ← V J_{pq}
                for i in 0..d {
                    let vip = v[[i, p]];
                    let viq = v[[i, q]];
                    v[[i, p]] = c * vip - s * viq;
                    v[[i, q]] = s * vip + c * viq;
                }
            }
        }
        if converged {
            break;
        }
    }

    // Singular values = sqrt of diagonal of B (clamped to ≥ 0)
    let mut sigma: Vec<f64> = (0..d).map(|i| b[[i, i]].max(0.0).sqrt()).collect();

    // Sort singular values in descending order (and permute V accordingly)
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&i, &j| {
        sigma[j]
            .partial_cmp(&sigma[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sigma_sorted: Vec<f64> = order.iter().map(|&i| sigma[i]).collect();
    let v_sorted: Array2<f64> = {
        let mut vs = Array2::<f64>::zeros((d, d));
        for (new_col, &old_col) in order.iter().enumerate() {
            for row in 0..d {
                vs[[row, new_col]] = v[[row, old_col]];
            }
        }
        vs
    };
    sigma = sigma_sorted;

    // Compute U = M V Σ^{-1}: columns u_i = M v_i / σ_i
    let mv = m.dot(&v_sorted);
    let mut u = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        let si = sigma[i];
        if si > eps {
            for r in 0..d {
                u[[r, i]] = mv[[r, i]] / si;
            }
        } else {
            // Zero singular value: u_i will be filled by Gram-Schmidt if needed
            // For Procrustes purposes (d ≤ typically ~100), just leave as zero
            // and we handle the det-fixing step separately.
        }
    }

    // Orthogonalize U columns for zero singular values via Gram-Schmidt
    orthogonalize_columns(&mut u);

    let vt = v_sorted.t().to_owned(); // Vᵀ: rows are right singular vectors
    Ok((u, sigma, vt))
}

/// Gram-Schmidt orthogonalization of matrix columns (in-place).
/// Only processes columns that are nearly zero.
fn orthogonalize_columns(m: &mut Array2<f64>) {
    let (r, c) = m.dim();
    let eps = 1e-12_f64;

    for j in 0..c {
        // Check if column j is near-zero
        let norm_sq: f64 = (0..r).map(|i| m[[i, j]] * m[[i, j]]).sum();
        if norm_sq > eps {
            // Normalize it
            let norm = norm_sq.sqrt();
            for i in 0..r {
                m[[i, j]] /= norm;
            }
            // Make subsequent columns orthogonal to this one
            for k in (j + 1)..c {
                let dot: f64 = (0..r).map(|i| m[[i, j]] * m[[i, k]]).sum();
                for i in 0..r {
                    let mij = m[[i, j]];
                    m[[i, k]] -= dot * mij;
                }
            }
        } else {
            // Find an arbitrary unit vector orthogonal to all previous columns
            for candidate in 0..r {
                let mut v = vec![0.0f64; r];
                v[candidate] = 1.0;
                // Orthogonalize against all previous columns
                for k in 0..j {
                    let dot: f64 = (0..r).map(|i| m[[i, k]] * v[i]).sum();
                    for i in 0..r {
                        let mik = m[[i, k]];
                        v[i] -= dot * mik;
                    }
                }
                let vnorm_sq: f64 = v.iter().map(|&x| x * x).sum();
                if vnorm_sq > eps {
                    let vnorm = vnorm_sq.sqrt();
                    for i in 0..r {
                        m[[i, j]] = v[i] / vnorm;
                    }
                    break;
                }
            }
        }
    }
}

/// Compute the determinant of a square matrix via Gaussian elimination.
pub(crate) fn mat_det(m: &Array2<f64>) -> f64 {
    let d = m.nrows();
    if d == 1 {
        return m[[0, 0]];
    }
    if d == 2 {
        return m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]];
    }
    if d == 3 {
        return m[[0, 0]] * (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]])
            - m[[0, 1]] * (m[[1, 0]] * m[[2, 2]] - m[[1, 2]] * m[[2, 0]])
            + m[[0, 2]] * (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]]);
    }

    // General case: LU with partial pivoting
    let mut a = m.to_owned();
    let mut sign = 1.0_f64;

    for col in 0..d {
        let mut max_val = a[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..d {
            if a[[row, col]].abs() > max_val {
                max_val = a[[row, col]].abs();
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return 0.0;
        }
        if max_row != col {
            for c in 0..d {
                let tmp = a[[col, c]];
                a[[col, c]] = a[[max_row, c]];
                a[[max_row, c]] = tmp;
            }
            sign *= -1.0;
        }
        let pivot = a[[col, col]];
        for row in (col + 1)..d {
            let factor = a[[row, col]] / pivot;
            for c in col..d {
                let v = a[[col, c]];
                a[[row, c]] -= factor * v;
            }
        }
    }

    let diag_prod: f64 = (0..d).map(|i| a[[i, i]]).product();
    sign * diag_prod
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    const TOL: f64 = 1e-5;

    // Helper: 2D rotation matrix
    fn rot2(angle_rad: f64) -> Array2<f64> {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        array![[c, -s], [s, c]]
    }

    // ------------------------------------------------------------------
    // Rotation-only alignment
    // ------------------------------------------------------------------

    #[test]
    fn test_procrustes_rotation() {
        // Rotate a 3-point configuration by 45° and recover rotation
        let a = array![[1.0_f64, 0.0], [0.0, 1.0], [-1.0, 0.0]];
        let angle = std::f64::consts::FRAC_PI_4;
        let r_true = rot2(angle);
        let b = a.dot(&r_true);

        let config = ProcrustesConfig {
            allow_reflection: false,
            scaling: false,
            centering: true,
        };
        let result = orthogonal_procrustes(&a, &b, &config).expect("procrustes ok");
        assert!(
            result.disparity < TOL,
            "residual should be near 0, got {}",
            result.disparity
        );
    }

    #[test]
    fn test_procrustes_no_reflection() {
        // When a reflection is the optimal map and allow_reflection=false,
        // we should get det(R) = +1
        let a = array![[1.0_f64, 0.0], [0.0, 1.0], [0.0, 0.0]];
        // Apply a reflection (det = -1): flip y-axis
        let b: Array2<f64> = array![[1.0_f64, 0.0], [0.0, -1.0], [0.0, 0.0]];

        let config = ProcrustesConfig {
            allow_reflection: false,
            scaling: false,
            centering: false,
        };
        let result = orthogonal_procrustes(&a, &b, &config).expect("procrustes ok");
        let det = mat_det(&result.rotation);
        assert!((det - 1.0).abs() < TOL, "det(R) should be +1, got {det}");
    }

    #[test]
    fn test_procrustes_scale_translation() {
        // Apply scale 2.0 and translation [3, -1], then recover
        let a = array![[0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let scale_true = 2.0_f64;
        let translation = array![3.0_f64, -1.0];
        let b: Array2<f64> = &a * scale_true + &translation.view().insert_axis(Axis(0));

        let config = ProcrustesConfig::default();
        let result = orthogonal_procrustes(&a, &b, &config).expect("procrustes ok");
        assert!(
            result.disparity < TOL,
            "residual should be near 0, got {}",
            result.disparity
        );
        assert!(
            (result.scale - scale_true).abs() < TOL,
            "scale should be {scale_true}, got {}",
            result.scale
        );
    }

    #[test]
    fn test_procrustes_identity() {
        // Aligning A to itself should give identity rotation and zero residual
        let a = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let config = ProcrustesConfig::default();
        let result = orthogonal_procrustes(&a, &a, &config).expect("procrustes ok");
        assert!(
            result.disparity < TOL,
            "residual for A→A should be 0, got {}",
            result.disparity
        );
    }

    #[test]
    fn test_procrustes_shape_mismatch_error() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let config = ProcrustesConfig::default();
        let result = orthogonal_procrustes(&a, &b, &config);
        assert!(result.is_err(), "mismatched shapes should produce an error");
    }

    // ------------------------------------------------------------------
    // Generalized Procrustes
    // ------------------------------------------------------------------

    #[test]
    fn test_generalized_procrustes() {
        // Create 4 rotated versions of the same square
        let base = array![[1.0_f64, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];

        let angles = [0.0_f64, 0.3, 0.7, 1.2];
        let matrices: Vec<Array2<f64>> = angles.iter().map(|&a| base.dot(&rot2(a))).collect();

        let results = generalized_procrustes(&matrices, 100, 1e-8).expect("GPA should converge");
        assert_eq!(results.len(), matrices.len());

        // Each result should have reasonably small disparity
        for (i, r) in results.iter().enumerate() {
            assert!(
                r.disparity < 1.0,
                "GPA result {i} disparity {:.4} should be small",
                r.disparity
            );
        }
    }

    #[test]
    fn test_generalized_procrustes_too_few_matrices() {
        let m = array![[1.0_f64, 0.0]];
        let result = generalized_procrustes(&[m], 100, 1e-8);
        assert!(result.is_err(), "single matrix should error");
    }

    #[test]
    fn test_generalized_procrustes_shape_mismatch() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![[1.0_f64, 0.0, 0.0]]; // different ncols
        let result = generalized_procrustes(&[a, b], 100, 1e-8);
        assert!(result.is_err(), "shape mismatch should error");
    }

    // ------------------------------------------------------------------
    // Determinant helper
    // ------------------------------------------------------------------

    #[test]
    fn test_det_2x2() {
        let m = array![[3.0_f64, 1.0], [5.0, 2.0]];
        let det = mat_det(&m);
        assert!((det - 1.0).abs() < 1e-12, "2x2 det should be 1, got {det}");
    }

    #[test]
    fn test_det_3x3() {
        let m = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
        let det = mat_det(&m);
        assert!((det - (-3.0)).abs() < 1e-10, "det should be -3, got {det}");
    }
}
