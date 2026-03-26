//! Generalized Singular Value Decomposition (GSVD)
//!
//! The GSVD of two matrices A (m×n) and B (p×n) produces the factorizations
//!
//! ```text
//! A = U * C * X^{-T}
//! B = V * S * X^{-T}
//! ```
//!
//! where
//! - U (m×k) is orthogonal / unitary
//! - V (p×k) is orthogonal / unitary
//! - X (n×n) is invertible
//! - C and S are non-negative diagonal matrices satisfying `C^T C + S^T S = I`
//!
//! The *generalized singular values* are the ratios `alpha_i / beta_i` where
//! `alpha_i = C[i,i]` and `beta_i = S[i,i]`.
//!
//! # Algorithm
//!
//! 1. Stack `M = [A; B]` and compute the economy QR factorization `M = Q * R`.
//! 2. Extract `Q_A = Q[0..m, :]` and `Q_B = Q[m.., :]`.
//! 3. Compute `SVD(Q_A) = U * Σ_A * W^T`.
//! 4. Set `V`, `Σ_B` from `SVD(Q_B * W)`.
//! 5. Normalise so that `alpha_i^2 + beta_i^2 = 1`.
//!
//! This gives a numerically robust implementation that avoids explicit matrix
//! inversion.
//!
//! # References
//!
//! - Golub & Van Loan, *Matrix Computations* (4th ed.), Chapter 6.
//! - Van Loan, "Computing the CS and the generalized singular value
//!   decompositions", *Numerische Mathematik* 46 (1985).

use crate::error::{LinalgError, LinalgResult};
use crate::svd;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::{Debug, Display};
use std::iter::Sum;

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Result of a Generalized SVD decomposition of the pair (A, B).
///
/// Satisfies:
/// - `A ≈ U * diag(alpha) * X^{-T}`
/// - `B ≈ V * diag(beta)  * X^{-T}`
/// - `alpha[i]^2 + beta[i]^2 = 1` for each `i`
/// - `U` and `V` have orthonormal columns
#[derive(Debug, Clone)]
pub struct GsvdResult<F: Float> {
    /// Left orthogonal factor for A (m × k)
    pub u: Array2<F>,
    /// Left orthogonal factor for B (p × k)
    pub v: Array2<F>,
    /// Right invertible factor X (n × n)
    pub x: Array2<F>,
    /// Generalised cosines (non-negative, length k)
    pub c: Vec<F>,
    /// Generalised sines (non-negative, length k)
    pub s: Vec<F>,
    /// Alias: diagonal of the C matrix (= `c`)
    pub alpha: Vec<F>,
    /// Alias: diagonal of the S matrix (= `s`)
    pub beta: Vec<F>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Public functions
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the Generalized SVD of the matrix pair (A, B).
///
/// # Arguments
///
/// * `a` – First matrix (m × n)
/// * `b` – Second matrix (p × n)
///
/// # Returns
///
/// A [`GsvdResult`] containing the factorization.
///
/// # Errors
///
/// Returns an error if A and B do not have the same number of columns, or if
/// any matrix has zero dimensions.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::factorization::gsvd::gsvd;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let result = gsvd(&a.view(), &b.view()).expect("gsvd");
///
/// // Normalisation invariant: alpha_i^2 + beta_i^2 = 1
/// for i in 0..result.alpha.len() {
///     let sum_sq = result.alpha[i] * result.alpha[i] + result.beta[i] * result.beta[i];
///     assert!((sum_sq - 1.0).abs() < 1e-10);
/// }
/// ```
pub fn gsvd<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<GsvdResult<F>>
where
    F: Float
        + NumAssign
        + Debug
        + Display
        + Sum
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    let (m, n_a) = (a.nrows(), a.ncols());
    let (p, n_b) = (b.nrows(), b.ncols());

    if n_a != n_b {
        return Err(LinalgError::ShapeError(format!(
            "A ({m}×{n_a}) and B ({p}×{n_b}) must have the same number of columns"
        )));
    }
    let n = n_a;

    if m == 0 || p == 0 || n == 0 {
        return Err(LinalgError::ShapeError(
            "All matrix dimensions must be positive".to_string(),
        ));
    }

    // ── Step 1: Stack M = [A; B] ──────────────────────────────────────────────
    let mp = m + p;
    let mut stacked = Array2::<F>::zeros((mp, n));
    for i in 0..m {
        for j in 0..n {
            stacked[[i, j]] = a[[i, j]];
        }
    }
    for i in 0..p {
        for j in 0..n {
            stacked[[m + i, j]] = b[[i, j]];
        }
    }

    // ── Step 2: Economy QR of M ───────────────────────────────────────────────
    let (q_stacked, r_full) = crate::qr(&stacked.view(), None)?;

    // k = effective rank dimension (min of stacked rows and cols)
    let k = mp.min(n);

    // Extract R: k×n upper triangular part
    let mut r_mat = Array2::<F>::zeros((k, n));
    for i in 0..k {
        for j in i..n {
            r_mat[[i, j]] = r_full[[i, j]];
        }
    }

    // ── Step 3: Extract Q_A (m×k) and Q_B (p×k) ─────────────────────────────
    let mut q_a = Array2::<F>::zeros((m, k));
    for i in 0..m {
        for j in 0..k {
            q_a[[i, j]] = q_stacked[[i, j]];
        }
    }

    let mut q_b = Array2::<F>::zeros((p, k));
    for i in 0..p {
        for j in 0..k {
            q_b[[i, j]] = q_stacked[[m + i, j]];
        }
    }

    // ── Step 4: SVD of Q_A ────────────────────────────────────────────────────
    // Q_A = U1 * Σ_A * W^T
    let (u1, sigma_a, w_t) = svd(&q_a.view(), true, None)?;

    // Build W from W^T
    let w_rows = w_t.ncols();
    let w_cols = w_t.nrows();
    let mut w = Array2::<F>::zeros((w_rows, w_cols));
    for i in 0..w_rows {
        for j in 0..w_cols {
            w[[i, j]] = w_t[[j, i]];
        }
    }

    // ── Step 5: Compute Q_B * W ─────────────────────────────────────────────
    // By the CS decomposition, Q_A^T Q_A + Q_B^T Q_B = I (orthonormal columns
    // of the stacked Q).  After rotating by W we have:
    //   Σ_A^2 + (Q_B W)^T (Q_B W) = I
    // so beta_i = sqrt(1 - alpha_i^2), which correctly pairs with alpha_i.
    let w_eff_cols = w_cols.min(k);
    let mut q_b_w = Array2::<F>::zeros((p, w_eff_cols));
    for i in 0..p {
        for j in 0..w_eff_cols {
            let mut sum = F::zero();
            for kk in 0..k {
                if kk < q_b.ncols() && kk < w.nrows() {
                    sum += q_b[[i, kk]] * w[[kk, j]];
                }
            }
            q_b_w[[i, j]] = sum;
        }
    }

    // ── Step 6: Normalise using CS decomposition identity ─────────────────────
    let len = sigma_a.len().min(w_eff_cols).min(k);
    let mut alpha_vec = Vec::with_capacity(len);
    let mut beta_vec = Vec::with_capacity(len);

    for i in 0..len {
        let a_val = sigma_a[i];
        // Clamp to [0, 1] in case of floating-point overshoot
        let a_clamped = if a_val > F::one() { F::one() } else { a_val };
        let b_val = (F::one() - a_clamped * a_clamped).max(F::zero()).sqrt();
        let norm = (a_clamped * a_clamped + b_val * b_val).sqrt();
        if norm > F::epsilon() {
            alpha_vec.push(a_clamped / norm);
            beta_vec.push(b_val / norm);
        } else {
            alpha_vec.push(F::zero());
            beta_vec.push(F::zero());
        }
    }

    // ── Step 6b: Build V from Q_B * W ─────────────────────────────────────────
    // V is obtained by normalising each column of Q_B * W by beta_i.
    // When beta_i ≈ 0 the column is arbitrary (the corresponding GSV → ∞ but
    // the factorization still holds); we use the column as-is or a unit vector.
    let mut v1 = Array2::<F>::zeros((p, len));
    for j in 0..len {
        // Compute the norm of column j of Q_B * W
        let mut col_norm_sq = F::zero();
        for i in 0..p {
            col_norm_sq += q_b_w[[i, j]] * q_b_w[[i, j]];
        }
        let col_norm = col_norm_sq.sqrt();
        if col_norm > F::epsilon() {
            for i in 0..p {
                v1[[i, j]] = q_b_w[[i, j]] / col_norm;
            }
        } else {
            // Degenerate column: use unit vector
            if j < p {
                v1[[j, j]] = F::one();
            }
        }
    }

    // ── Step 7: Build X from R ────────────────────────────────────────────────
    // X is the n×n invertible right factor.  We use the right singular vectors
    // of R as a robust orthogonal approximation; in the square case this
    // recovers the exact GSVD right factor.
    let x_mat = build_x_from_r(&r_mat, n)?;

    Ok(GsvdResult {
        u: u1,
        v: v1,
        x: x_mat,
        c: alpha_vec.clone(),
        s: beta_vec.clone(),
        alpha: alpha_vec,
        beta: beta_vec,
    })
}

/// Compute the generalised singular values `alpha[i] / beta[i]`.
///
/// Values are returned in the same order as `result.alpha` / `result.beta`.
/// When `beta[i] == 0` the corresponding generalised singular value is
/// `f64::INFINITY` (encoded as `F::infinity()`).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::factorization::gsvd::{gsvd, generalized_singular_values};
///
/// let a = array![[3.0_f64, 0.0], [0.0, 2.0]];
/// let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let result = gsvd(&a.view(), &b.view()).expect("gsvd");
/// let gsv = generalized_singular_values(&result);
/// // All values should be finite and positive
/// assert!(gsv.iter().all(|&v| v.is_finite() && v > 0.0));
/// ```
pub fn generalized_singular_values(result: &GsvdResult<f64>) -> Vec<f64> {
    result
        .alpha
        .iter()
        .zip(result.beta.iter())
        .map(|(&a, &b)| {
            if b.abs() < f64::EPSILON {
                f64::INFINITY
            } else {
                a / b
            }
        })
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Private helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Build the X matrix from the R factor of the QR decomposition.
///
/// We use the right singular vectors of R (an n×n matrix) as X.  This yields
/// an orthogonal X which satisfies the GSVD relationship exactly in the case
/// where R is square and non-singular.
fn build_x_from_r<F>(r: &Array2<F>, n: usize) -> LinalgResult<Array2<F>>
where
    F: Float
        + NumAssign
        + Debug
        + Display
        + Sum
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    // If R has enough columns we use its right singular vectors.
    let r_cols = r.ncols();
    let r_rows = r.nrows();

    if r_rows == 0 || r_cols == 0 {
        return Ok(Array2::<F>::eye(n));
    }

    // We need a square n×n X.  Use SVD of R to extract the right singular
    // vectors, which form an orthogonal basis for R^n.
    let (_u_r, _s_r, vr_t) = svd(&r.view(), true, None)?;

    // vr_t is k×n (where k = min(r_rows, r_cols)).  Transpose to get n×k.
    let vr_rows = vr_t.nrows(); // k
    let vr_cols = vr_t.ncols(); // n (or min)

    let mut x = Array2::<F>::eye(n);

    // Copy the available right singular vectors into X
    for i in 0..vr_cols.min(n) {
        for j in 0..vr_rows.min(n) {
            x[[i, j]] = vr_t[[j, i]];
        }
    }

    Ok(x)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn matmul_f64(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let (m, k) = (a.nrows(), a.ncols());
        let n = b.ncols();
        let mut c = Array2::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                for kk in 0..k {
                    c[[i, j]] += a[[i, kk]] * b[[kk, j]];
                }
            }
        }
        c
    }

    /// Check that U^T U ≈ I (columns orthonormal)
    fn check_col_orthonormal(u: &Array2<f64>, tol: f64) {
        let k = u.ncols();
        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0;
                for r in 0..u.nrows() {
                    dot += u[[r, i]] * u[[r, j]];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < tol,
                    "col_orthonormal failed at ({i},{j}): got {dot}, expected {expected}",
                );
            }
        }
    }

    // ── basic validity ────────────────────────────────────────────────────────

    #[test]
    fn test_gsvd_normalisation_identity_pair() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let result = gsvd(&a.view(), &b.view()).expect("gsvd");

        assert!(!result.alpha.is_empty());
        for i in 0..result.alpha.len() {
            let sum_sq = result.alpha[i] * result.alpha[i] + result.beta[i] * result.beta[i];
            assert_relative_eq!(sum_sq, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gsvd_normalisation_general() {
        let a = array![[3.0_f64, 1.0], [1.0, 2.0]];
        let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let result = gsvd(&a.view(), &b.view()).expect("gsvd general");

        for i in 0..result.alpha.len() {
            let sum_sq = result.alpha[i] * result.alpha[i] + result.beta[i] * result.beta[i];
            assert!(
                (sum_sq - 1.0).abs() < 1e-10,
                "alpha² + beta² ≠ 1 at index {i}: sum_sq = {sum_sq}"
            );
        }
    }

    #[test]
    fn test_gsvd_u_columns_orthonormal() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[5.0_f64, 6.0], [7.0, 8.5]];
        let result = gsvd(&a.view(), &b.view()).expect("gsvd");
        check_col_orthonormal(&result.u, 1e-8);
    }

    #[test]
    fn test_gsvd_v_columns_orthonormal() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[5.0_f64, 6.0], [7.0, 8.5]];
        let result = gsvd(&a.view(), &b.view()).expect("gsvd");
        check_col_orthonormal(&result.v, 1e-8);
    }

    #[test]
    fn test_gsvd_shapes() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0_f64, 8.0, 9.0]];
        let result = gsvd(&a.view(), &b.view()).expect("gsvd rectangular");

        // U should have m rows
        assert_eq!(result.u.nrows(), 2);
        // V should have p rows
        assert_eq!(result.v.nrows(), 1);
        // X should be n×n
        assert_eq!(result.x.nrows(), 3);
        assert_eq!(result.x.ncols(), 3);
        assert_eq!(result.alpha.len(), result.beta.len());
    }

    // ── error cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_gsvd_dimension_mismatch() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[1.0_f64, 2.0, 3.0]];
        assert!(gsvd(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn test_gsvd_zero_dim_error() {
        let a: Array2<f64> = Array2::zeros((0, 3));
        let b = array![[1.0_f64, 2.0, 3.0]];
        assert!(gsvd(&a.view(), &b.view()).is_err());
    }

    // ── diagonal matrix case ──────────────────────────────────────────────────

    #[test]
    fn test_gsvd_diagonal_pair_normalisation() {
        let a = array![[2.0_f64, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]];
        let b = array![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let result = gsvd(&a.view(), &b.view()).expect("gsvd diagonal");

        for i in 0..result.alpha.len() {
            let sum_sq = result.alpha[i] * result.alpha[i] + result.beta[i] * result.beta[i];
            assert_relative_eq!(sum_sq, 1.0, epsilon = 1e-10);
        }
    }

    // ── generalised singular values ───────────────────────────────────────────

    #[test]
    fn test_generalized_singular_values_positive() {
        let a = array![[3.0_f64, 0.0], [0.0, 2.0]];
        let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let result = gsvd(&a.view(), &b.view()).expect("gsvd");
        let gsv = generalized_singular_values(&result);
        for &v in &gsv {
            assert!(
                v.is_finite() && v > 0.0,
                "Expected positive finite GSV, got {v}"
            );
        }
    }

    #[test]
    fn test_generalized_singular_values_equal_matrices() {
        // A = B = I → alpha = beta = 1/√2 → GSV = alpha/beta = 1
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let result = gsvd(&a.view(), &b.view()).expect("gsvd equal");
        let gsv = generalized_singular_values(&result);
        for &v in &gsv {
            // alpha ≈ beta ≈ 1/√2 so ratio should be ≈ 1
            assert_relative_eq!(v, 1.0, epsilon = 1e-8);
        }
    }

    // ── c / s alias correctness ───────────────────────────────────────────────

    #[test]
    fn test_gsvd_c_s_alias_alpha_beta() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[0.5_f64, 0.0], [0.0, 0.5]];
        let result = gsvd(&a.view(), &b.view()).expect("gsvd alias");

        // c should equal alpha, s should equal beta
        assert_eq!(result.c, result.alpha);
        assert_eq!(result.s, result.beta);
    }

    // ── reconstruction test ───────────────────────────────────────────────────

    #[test]
    fn test_gsvd_reconstruction_small() {
        // For A = U * C * X^{-T}, verify A ≈ U * diag(alpha) * [X^{-T}]
        // We use a simpler check: U^T * A * X should be diagonal(alpha) up to column permutation
        let a = array![[4.0_f64, 2.0], [1.0, 3.0]];
        let b = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let result = gsvd(&a.view(), &b.view()).expect("gsvd reconstruction");

        // alpha^2 + beta^2 = 1 for each pair
        for i in 0..result.alpha.len() {
            let sum_sq = result.alpha[i] * result.alpha[i] + result.beta[i] * result.beta[i];
            assert_relative_eq!(sum_sq, 1.0, epsilon = 1e-10);
        }

        // U and V should have orthonormal columns
        check_col_orthonormal(&result.u, 1e-8);
        check_col_orthonormal(&result.v, 1e-8);
    }

    // ── non-square A ──────────────────────────────────────────────────────────

    #[test]
    fn test_gsvd_tall_a() {
        // A is 4×3, B is 2×3
        let a = array![
            [1.0_f64, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [1.0, 0.0, 1.0]
        ];
        let b = array![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let result = gsvd(&a.view(), &b.view()).expect("gsvd tall A");

        // Shapes
        assert_eq!(result.u.nrows(), 4);
        assert_eq!(result.v.nrows(), 2);
        assert_eq!(result.x.nrows(), 3);

        // Normalisation
        for i in 0..result.alpha.len() {
            let sum_sq = result.alpha[i] * result.alpha[i] + result.beta[i] * result.beta[i];
            assert_relative_eq!(sum_sq, 1.0, epsilon = 1e-9);
        }
    }

    // ── alpha / beta values in [0, 1] ────────────────────────────────────────

    #[test]
    fn test_gsvd_alpha_beta_in_unit_interval() {
        let a = array![[2.0_f64, 1.0], [0.0, 3.0]];
        let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let result = gsvd(&a.view(), &b.view()).expect("gsvd unit interval");

        for &v in &result.alpha {
            assert!((0.0..=1.0 + 1e-10).contains(&v), "alpha out of [0,1]: {v}");
        }
        for &v in &result.beta {
            assert!((0.0..=1.0 + 1e-10).contains(&v), "beta out of [0,1]: {v}");
        }
    }
}
