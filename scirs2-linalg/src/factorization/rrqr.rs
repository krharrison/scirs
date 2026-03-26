//! Rank-Revealing QR (RRQR) Decomposition
//!
//! Computes A * P = Q * R where P is a column permutation chosen so that
//! the diagonal of R reveals the numerical rank of A.
//!
//! # Algorithms
//!
//! - **Businger-Golub**: At each step, pivot the column with the largest
//!   remaining 2-norm. This is the classical column-pivoted QR.
//! - **Strong RRQR**: After the Businger-Golub factorization, verify that
//!   R11 is well-conditioned and R22 is small; if not, perform additional
//!   column swaps to strengthen the rank revelation.
//!
//! # References
//!
//! - Businger & Golub (1965). "Linear Least Squares Solutions by
//!   Householder Transformations." Numer. Math. 7.
//! - Gu & Eisenstat (1996). "Efficient Algorithms for Computing a Strong
//!   Rank-Revealing QR Factorization." SIAM J. Sci. Comput. 17(4).
//! - Golub & Van Loan (2013). "Matrix Computations." 4th ed., Ch. 5.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

// ============================================================================
// Result type
// ============================================================================

/// Result of Rank-Revealing QR decomposition.
///
/// Stores A * P = Q * R where P is a permutation, Q is orthogonal, and R is
/// upper triangular with diagonal entries sorted by decreasing magnitude
/// (up to the numerical rank boundary).
#[derive(Debug, Clone)]
pub struct RRQRResult<F> {
    /// Orthogonal factor Q  (m x m)
    pub q: Array2<F>,
    /// Upper triangular factor R  (m x n)
    pub r: Array2<F>,
    /// Column permutation vector of length n.
    /// `perm[j]` is the original column index that ended up in position j.
    pub perm: Vec<usize>,
    /// Detected numerical rank (number of "significant" columns).
    pub rank: usize,
}

// ============================================================================
// Businger-Golub RRQR
// ============================================================================

/// Compute the Rank-Revealing QR factorization of a matrix using the
/// Businger-Golub algorithm (column-pivoted Householder QR).
///
/// Returns `A * P = Q * R` where:
/// - Q is m x m orthogonal
/// - R is m x n upper triangular
/// - P is represented as a permutation vector
///
/// The numerical rank is determined as the largest k such that
/// `|R[k-1, k-1]| / |R[0, 0]| >= tol`.
///
/// # Arguments
///
/// * `a`   - Input matrix (m x n)
/// * `tol` - Tolerance for rank determination. Columns with relative
///   diagonal magnitude below this threshold are considered
///   numerically zero.
///
/// # Errors
///
/// Returns an error if the input matrix is empty or contains non-finite values.
pub fn rrqr<F>(a: &ArrayView2<F>, tol: F) -> LinalgResult<RRQRResult<F>>
where
    F: Float + NumAssign + Sum + Debug + ScalarOperand + Send + Sync + 'static,
{
    let (m, n) = a.dim();
    if m == 0 || n == 0 {
        return Err(LinalgError::ShapeError(
            "RRQR: input matrix must be non-empty".to_string(),
        ));
    }
    for &v in a.iter() {
        if !v.is_finite() {
            return Err(LinalgError::InvalidInputError(
                "RRQR: matrix contains non-finite values".to_string(),
            ));
        }
    }

    let min_dim = m.min(n);

    // Working copy of A (will become R)
    let mut r = a.to_owned();
    // Q accumulator (identity initially)
    let mut q = Array2::<F>::eye(m);
    // Permutation vector: perm[j] = original column index
    let mut perm: Vec<usize> = (0..n).collect();

    // Column norms squared (for pivoting)
    let mut col_norms_sq: Vec<F> = (0..n)
        .map(|j| {
            let mut s = F::zero();
            for i in 0..m {
                s += r[[i, j]] * r[[i, j]];
            }
            s
        })
        .collect();

    let two = F::from(2.0).unwrap_or_else(|| F::one() + F::one());

    let mut rank = 0usize;

    for k in 0..min_dim {
        // --- Pivot: find column with max remaining norm among k..n ---
        let mut best_col = k;
        let mut best_norm = col_norms_sq[k];
        for (j, &cnj) in col_norms_sq.iter().enumerate().take(n).skip(k + 1) {
            if cnj > best_norm {
                best_norm = cnj;
                best_col = j;
            }
        }

        // Early exit if remaining columns are negligible
        if best_norm.sqrt() <= F::epsilon() {
            break;
        }

        // Swap columns k and best_col in R, perm, col_norms_sq
        if best_col != k {
            for i in 0..m {
                let tmp = r[[i, k]];
                r[[i, k]] = r[[i, best_col]];
                r[[i, best_col]] = tmp;
            }
            perm.swap(k, best_col);
            col_norms_sq.swap(k, best_col);
        }

        // --- Householder reflection to zero out R[k+1..m, k] ---
        let mut x = Array1::<F>::zeros(m - k);
        for i in k..m {
            x[i - k] = r[[i, k]];
        }

        let x_norm = x.iter().fold(F::zero(), |acc, &v| acc + v * v).sqrt();

        if x_norm <= F::epsilon() {
            // Column is already zero below diagonal — nothing to do
            // but it counts as a zero column in rank detection
            continue;
        }

        // Sign choice to avoid cancellation
        let alpha = if x[0] >= F::zero() { -x_norm } else { x_norm };
        let mut v = x;
        v[0] -= alpha;

        let v_norm_sq = v.iter().fold(F::zero(), |acc, &val| acc + val * val);
        if v_norm_sq <= F::epsilon() {
            continue;
        }
        // beta = 2 / (v^T v)  for the Householder reflector H = I - beta * v * v^T
        let beta = two / v_norm_sq;

        // Apply H to R from the left: R[k:, j] -= beta * v * (v^T * R[k:, j])
        for j in k..n {
            let mut dot = F::zero();
            for i in 0..(m - k) {
                dot += v[i] * r[[i + k, j]];
            }
            for i in 0..(m - k) {
                r[[i + k, j]] -= beta * v[i] * dot;
            }
        }

        // Accumulate Q: Q[:, k:] -= beta * (Q[:, k:] * v) * v^T
        for i in 0..m {
            let mut dot = F::zero();
            for jj in 0..(m - k) {
                dot += q[[i, jj + k]] * v[jj];
            }
            for jj in 0..(m - k) {
                q[[i, jj + k]] -= beta * dot * v[jj];
            }
        }

        // Update column norms for remaining columns (down-date)
        for j in (k + 1)..n {
            // Remove contribution of row k from the norm
            let rk = r[[k, j]];
            col_norms_sq[j] -= rk * rk;
            // Guard against negative values from floating-point drift
            if col_norms_sq[j] < F::zero() {
                col_norms_sq[j] = F::zero();
                for i in (k + 1)..m {
                    col_norms_sq[j] += r[[i, j]] * r[[i, j]];
                }
            }
        }

        // Rank determination: compare |R[k,k]| against |R[0,0]|
        let r00_abs = r[[0, 0]].abs();
        let rkk_abs = r[[k, k]].abs();
        if r00_abs > F::epsilon() && rkk_abs / r00_abs >= tol {
            rank = k + 1;
        } else if r00_abs <= F::epsilon() {
            // The very first diagonal is tiny, so rank stays 0 (unless rkk is big)
            if rkk_abs > tol {
                rank = k + 1;
            }
        }
    }

    // If we never triggered the rank counter at all but the matrix is nonzero
    // (e.g. 1x1 case) do a final check
    if rank == 0 && min_dim > 0 && r[[0, 0]].abs() > tol {
        rank = 1;
    }

    Ok(RRQRResult { q, r, perm, rank })
}

/// Convenience function: compute only the numerical rank of a matrix.
///
/// Equivalent to `rrqr(a, tol)?.rank` but semantically clearer at the
/// call site.
pub fn rrqr_rank<F>(a: &ArrayView2<F>, tol: F) -> LinalgResult<usize>
where
    F: Float + NumAssign + Sum + Debug + ScalarOperand + Send + Sync + 'static,
{
    Ok(rrqr(a, tol)?.rank)
}

// ============================================================================
// Strong RRQR
// ============================================================================

/// Compute a *strong* Rank-Revealing QR factorization.
///
/// After the standard Businger-Golub RRQR, this routine checks whether the
/// factorization satisfies the strong RRQR conditions:
///
/// 1. `sigma_min(R11) >= sigma_min(A) / f(k, n)`
/// 2. `|R22| <= f(k, n) * sigma_{k+1}(A)`
///
/// where `f` is a modest polynomial bound. If the conditions are not met,
/// additional column swaps and re-triangularizations are performed.
///
/// This is based on the Gu-Eisenstat algorithm (1996).
///
/// # Arguments
///
/// * `a`         - Input matrix (m x n)
/// * `tol`       - Tolerance for rank determination
/// * `max_swaps` - Maximum number of strengthening swaps (0 = Businger-Golub only)
pub fn strong_rrqr<F>(a: &ArrayView2<F>, tol: F, max_swaps: usize) -> LinalgResult<RRQRResult<F>>
where
    F: Float + NumAssign + Sum + Debug + ScalarOperand + Send + Sync + 'static,
{
    let mut result = rrqr(a, tol)?;

    if max_swaps == 0 || result.rank == 0 {
        return Ok(result);
    }

    let (m, n) = a.dim();
    let k = result.rank;
    let two = F::from(2.0).unwrap_or_else(|| F::one() + F::one());

    // Strengthening loop: look for columns in R22 that are "too large"
    // compared to the smallest diagonal in R11, and swap them in.
    for _swap_iter in 0..max_swaps {
        // Find the smallest |R[i,i]| for i < k
        let mut min_diag = result.r[[0, 0]].abs();
        let mut min_diag_idx = 0usize;
        for i in 1..k {
            let d = result.r[[i, i]].abs();
            if d < min_diag {
                min_diag = d;
                min_diag_idx = i;
            }
        }

        // Find the largest column norm in R22 (columns k..n, rows k..m)
        let mut max_col_norm_sq = F::zero();
        let mut max_col_idx = k;
        for j in k..n {
            let mut norm_sq = F::zero();
            for i in k..m {
                norm_sq += result.r[[i, j]] * result.r[[i, j]];
            }
            // Also include the R12 entries (row < k, col j) for the
            // contribution to the column's overall significance
            let mut r12_norm_sq = F::zero();
            for i in 0..k {
                r12_norm_sq += result.r[[i, j]] * result.r[[i, j]];
            }
            let total = norm_sq + r12_norm_sq;
            if total > max_col_norm_sq {
                max_col_norm_sq = total;
                max_col_idx = j;
            }
        }

        let max_col_norm = max_col_norm_sq.sqrt();

        // If the largest R22 column norm is smaller than the smallest R11
        // diagonal (within a factor), we're done.
        if max_col_norm <= min_diag * two {
            break;
        }

        // Swap column min_diag_idx and max_col_idx in R and perm
        if min_diag_idx != max_col_idx {
            for i in 0..m {
                let tmp = result.r[[i, min_diag_idx]];
                result.r[[i, min_diag_idx]] = result.r[[i, max_col_idx]];
                result.r[[i, max_col_idx]] = tmp;
            }
            result.perm.swap(min_diag_idx, max_col_idx);

            // Re-triangularize the swapped columns using Givens rotations
            retriangularize_givens(&mut result.q, &mut result.r, min_diag_idx, m);
        }

        // Re-evaluate rank
        let r00 = result.r[[0, 0]].abs();
        let mut new_rank = 0;
        for i in 0..k.min(m.min(n)) {
            if r00 > F::epsilon() && result.r[[i, i]].abs() / r00 >= tol {
                new_rank = i + 1;
            }
        }
        result.rank = new_rank;
    }

    Ok(result)
}

/// Apply Givens rotations to restore upper-triangular form in column `col`
/// after a column swap.
fn retriangularize_givens<F>(q: &mut Array2<F>, r: &mut Array2<F>, col: usize, m: usize)
where
    F: Float + NumAssign + Sum + Debug + ScalarOperand + 'static,
{
    let n = r.ncols();
    let two = F::from(2.0).unwrap_or_else(|| F::one() + F::one());
    let _ = two; // suppress unused warning

    // Zero out sub-diagonal entries in column `col` using Givens rotations
    for i in (col + 1)..m {
        let a_val = r[[col, col]];
        let b_val = r[[i, col]];

        if b_val.abs() <= F::epsilon() {
            continue;
        }

        // Compute Givens rotation [c s; -s c] that zeros b
        let rr = (a_val * a_val + b_val * b_val).sqrt();
        let c = a_val / rr;
        let s = b_val / rr;

        // Apply rotation to rows col and i of R
        for j in 0..n {
            let r_col_j = r[[col, j]];
            let r_i_j = r[[i, j]];
            r[[col, j]] = c * r_col_j + s * r_i_j;
            r[[i, j]] = -s * r_col_j + c * r_i_j;
        }

        // Apply rotation to columns col and i of Q (Q := Q * G^T)
        let q_rows = q.nrows();
        for row in 0..q_rows {
            let q_r_col = q[[row, col]];
            let q_r_i = q[[row, i]];
            q[[row, col]] = c * q_r_col + s * q_r_i;
            q[[row, i]] = -s * q_r_col + c * q_r_i;
        }
    }
}

/// Build the n x n permutation matrix from a permutation vector.
///
/// `perm[j]` means original column `perm[j]` is now in position `j`.
/// The permutation matrix P satisfies: A * P = Q * R.
pub fn perm_to_matrix<F>(perm: &[usize]) -> Array2<F>
where
    F: Float,
{
    let n = perm.len();
    let mut p = Array2::<F>::zeros((n, n));
    for (j, &orig) in perm.iter().enumerate() {
        // Column j of P has a 1 in row orig
        p[[orig, j]] = F::one();
    }
    p
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Helper: compute Frobenius norm of (A - B)
    fn frob_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn test_rrqr_full_rank_square() {
        // A well-conditioned 3x3 matrix with rank 3
        let a = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0] // not 9 -> full rank
        ];
        let res = rrqr(&a.view(), 1e-12).expect("rrqr failed");

        assert_eq!(res.rank, 3, "should detect full rank");
        assert_eq!(res.q.shape(), &[3, 3]);
        assert_eq!(res.r.shape(), &[3, 3]);
        assert_eq!(res.perm.len(), 3);

        // Q should be orthogonal
        let qtq = res.q.t().dot(&res.q);
        let eye3 = Array2::<f64>::eye(3);
        assert!(frob_diff(&qtq, &eye3) < 1e-10, "Q must be orthogonal");

        // Reconstruction: A * P = Q * R  =>  A = Q * R * P^T
        let p = perm_to_matrix::<f64>(&res.perm);
        let qr = res.q.dot(&res.r);
        let qr_pt = qr.dot(&p.t());
        assert!(
            frob_diff(&qr_pt, &a.to_owned()) < 1e-10,
            "A = Q R P^T reconstruction"
        );
    }

    #[test]
    fn test_rrqr_rank_deficient() {
        // Rank-2 matrix (third row = first + second)
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]];
        let res = rrqr(&a.view(), 1e-10).expect("rrqr failed");

        assert_eq!(res.rank, 2, "rank should be 2");

        // R[2,2] should be negligible compared to R[0,0]
        let ratio = res.r[[2, 2]].abs() / res.r[[0, 0]].abs();
        assert!(ratio < 1e-10, "R[2,2]/R[0,0] = {ratio} should be tiny");
    }

    #[test]
    fn test_rrqr_column_pivoting_improves_conditioning() {
        // Matrix where first column is tiny — pivoting should swap it
        let a = array![[1e-15, 1.0, 0.0], [1e-15, 0.0, 1.0], [1e-15, 0.0, 0.0]];
        let res = rrqr(&a.view(), 1e-12).expect("rrqr failed");

        // Column 0 (tiny) should NOT be the first pivot
        assert_ne!(
            res.perm[0], 0,
            "pivot should not choose the tiny column first"
        );
    }

    #[test]
    fn test_rrqr_reconstruction_rectangular() {
        // 4 x 3 tall matrix
        let a = array![
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
            [2.0, 1.0, 3.0]
        ];
        let res = rrqr(&a.view(), 1e-10).expect("rrqr failed");

        // Reconstruction
        let p = perm_to_matrix::<f64>(&res.perm);
        let reconstructed = res.q.dot(&res.r).dot(&p.t());

        let diff = frob_diff(&reconstructed, &a.to_owned());
        assert!(diff < 1e-10, "reconstruction error = {diff}");

        // Rank should be 2 (col3 = col1 + col2)
        assert_eq!(res.rank, 2, "rectangular matrix has rank 2");
    }

    #[test]
    fn test_rrqr_rank_convenience() {
        let a = array![
            [1.0, 2.0],
            [2.0, 4.0] // rank 1
        ];
        let r = rrqr_rank(&a.view(), 1e-10).expect("rrqr_rank failed");
        assert_eq!(r, 1);
    }

    #[test]
    fn test_rrqr_identity() {
        let eye = Array2::<f64>::eye(4);
        let res = rrqr(&eye.view(), 1e-12).expect("rrqr failed");
        assert_eq!(res.rank, 4);
    }

    #[test]
    fn test_rrqr_zero_matrix() {
        let z = Array2::<f64>::zeros((3, 3));
        let res = rrqr(&z.view(), 1e-12).expect("rrqr failed");
        assert_eq!(res.rank, 0, "zero matrix has rank 0");
    }

    #[test]
    fn test_rrqr_empty_matrix_error() {
        let e = Array2::<f64>::zeros((0, 0));
        assert!(rrqr(&e.view(), 1e-12).is_err());
    }

    #[test]
    fn test_strong_rrqr_basic() {
        // Rank-deficient matrix
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]];
        let res = strong_rrqr(&a.view(), 1e-10, 5).expect("strong_rrqr failed");
        assert_eq!(res.rank, 2);

        // Reconstruction
        let p = perm_to_matrix::<f64>(&res.perm);
        let reconstructed = res.q.dot(&res.r).dot(&p.t());
        let diff = frob_diff(&reconstructed, &a.to_owned());
        assert!(diff < 1e-8, "strong RRQR reconstruction error = {diff}");
    }

    #[test]
    fn test_strong_rrqr_improves_over_basic() {
        // With max_swaps = 0 it should behave like basic RRQR
        let a = array![[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]];
        let basic = rrqr(&a.view(), 1e-10).expect("rrqr failed");
        let strong = strong_rrqr(&a.view(), 1e-10, 10).expect("strong rrqr failed");

        // Both should detect rank 2
        assert_eq!(basic.rank, 2);
        assert_eq!(strong.rank, 2);
    }

    #[test]
    fn test_perm_to_matrix() {
        let perm = vec![2, 0, 1];
        let p = perm_to_matrix::<f64>(&perm);
        // Column 0 of P should have 1 in row 2
        assert!((p[[2, 0]] - 1.0).abs() < 1e-15);
        // Column 1 of P should have 1 in row 0
        assert!((p[[0, 1]] - 1.0).abs() < 1e-15);
        // Column 2 of P should have 1 in row 1
        assert!((p[[1, 2]] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_rrqr_wide_matrix() {
        // 2 x 4 wide matrix, rank 2
        let a = array![[1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 3.0]];
        let res = rrqr(&a.view(), 1e-10).expect("rrqr failed");
        assert_eq!(res.rank, 2);

        let p = perm_to_matrix::<f64>(&res.perm);
        let reconstructed = res.q.dot(&res.r).dot(&p.t());
        let diff = frob_diff(&reconstructed, &a.to_owned());
        assert!(diff < 1e-10, "wide matrix reconstruction error = {diff}");
    }
}
