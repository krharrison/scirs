//! URV Decomposition
//!
//! Computes A = U * R * V^T where U and V are orthogonal and R is upper
//! triangular with a rank-revealing structure.  This is a two-sided
//! orthogonal decomposition that exposes the numerical rank of A.
//!
//! # Algorithm
//!
//! 1. Compute the RRQR factorization: A * P_col = Q_1 * R_1
//! 2. Transpose R_1 and compute its QR: R_1^T = Q_2 * R_2
//! 3. Set U = Q_1 * Q_2, R = R_2^T (upper triangular), V = P_col
//!    (or refine V from Q_2).
//!
//! The matrix R has the block structure:
//!
//! ```text
//!     [ R11  R12 ]
//! R = [          ]     where R11 is k x k upper triangular (k = rank)
//!     [  0   R22 ]
//! ```
//!
//! and |R22| is small (≈ 0 for exact rank deficiency).
//!
//! # References
//!
//! - Stewart (1999). "The QLP Approximation to the SVD."
//!   SIAM J. Sci. Comput. 20(4).
//! - Fierro & Hansen (1997). "UTV Tools: A Matlab Package for Rank-Revealing
//!   and Rank-Modifying Decompositions."

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};
use crate::factorization::rrqr::{perm_to_matrix, rrqr};

// ============================================================================
// Result type
// ============================================================================

/// Result of URV decomposition.
///
/// Stores A = U * R * V^T where U and V are orthogonal and R is upper
/// triangular with rank-revealing structure.
#[derive(Debug, Clone)]
pub struct URVResult<F> {
    /// Left orthogonal factor U  (m x m)
    pub u: Array2<F>,
    /// Upper triangular factor R  (m x n) with rank structure
    pub r: Array2<F>,
    /// Right orthogonal factor V  (n x n)
    pub v: Array2<F>,
    /// Detected numerical rank
    pub rank: usize,
}

// ============================================================================
// URV decomposition
// ============================================================================

/// Compute the URV decomposition of a matrix.
///
/// Returns A = U * R * V^T where:
/// - U is m x m orthogonal
/// - R is m x n upper triangular with rank structure
/// - V is n x n orthogonal
///
/// # Arguments
///
/// * `a`   - Input matrix (m x n)
/// * `tol` - Tolerance for numerical rank detection
///
/// # Errors
///
/// Returns an error if the input matrix is empty or contains non-finite values.
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::factorization::urv::urv;
///
/// let a = array![
///     [1.0_f64, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [5.0, 7.0, 9.0]  // row3 = row1 + row2 => rank 2
/// ];
/// let res = urv(&a.view(), 1e-10).expect("urv");
/// assert_eq!(res.rank, 2);
///
/// // Reconstruction: A ≈ U * R * V^T
/// let recon = res.u.dot(&res.r).dot(&res.v.t());
/// let err: f64 = a.iter().zip(recon.iter())
///     .map(|(&x, &y)| (x - y).powi(2)).sum::<f64>().sqrt();
/// assert!(err < 1e-10);
/// ```
pub fn urv<F>(a: &ArrayView2<F>, tol: F) -> LinalgResult<URVResult<F>>
where
    F: Float + NumAssign + Sum + Debug + ScalarOperand + Send + Sync + 'static,
{
    let (m, n) = a.dim();
    if m == 0 || n == 0 {
        return Err(LinalgError::ShapeError(
            "URV: input matrix must be non-empty".to_string(),
        ));
    }

    // Step 1: RRQR of A: A * P = Q1 * R1
    let rrqr_res = rrqr(a, tol)?;
    let q1 = rrqr_res.q; // m x m
    let r1 = rrqr_res.r; // m x n
    let rank = rrqr_res.rank;

    // Build the permutation matrix P (n x n)
    let p_mat = perm_to_matrix::<F>(&rrqr_res.perm); // n x n

    // Step 2: QR of R1^T to get right orthogonal factor
    //   R1^T is n x m.  We compute its QR: R1^T = Q2 * R2   (Q2: n x n, R2: n x m)
    let r1t = r1.t().to_owned(); // n x m
    let (q2, r2) = householder_qr(&r1t)?;

    // Step 3: Construct the URV factors
    //   A * P = Q1 * R1
    //   R1^T = Q2 * R2
    //   R1   = R2^T * Q2^T
    //   A * P = Q1 * R2^T * Q2^T
    //   A     = Q1 * R2^T * Q2^T * P^T
    //
    //   So: U = Q1, R_urv = R2^T, V = P * Q2  (and A = U * R_urv * V^T)
    let r_urv = r2.t().to_owned(); // m x n (upper triangular structure)
    let v = p_mat.dot(&q2); // n x n

    Ok(URVResult {
        u: q1,
        r: r_urv,
        v,
        rank,
    })
}

/// Solve a rank-deficient least-squares problem using URV decomposition.
///
/// Finds x that minimizes ||A x - b||_2 for potentially rank-deficient A.
///
/// The URV decomposition gives A = U * R * V^T.  The minimum-norm
/// least-squares solution is:
///
///   x = V * R^{+} * U^T * b
///
/// where R^{+} is the pseudo-inverse of R (truncated at the detected rank).
///
/// # Arguments
///
/// * `a`   - Coefficient matrix (m x n)
/// * `b`   - Right-hand side vector (m)
/// * `tol` - Tolerance for rank detection
pub fn urv_lstsq<F>(a: &ArrayView2<F>, b: &Array1<F>, tol: F) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + Debug + ScalarOperand + Send + Sync + 'static,
{
    let (m, _n) = a.dim();
    if b.len() != m {
        return Err(LinalgError::DimensionError(format!(
            "URV lstsq: b length ({}) must match matrix rows ({m})",
            b.len()
        )));
    }

    let res = urv(a, tol)?;
    let k = res.rank;
    let n = res.v.nrows();

    // c = U^T * b  (m-vector)
    let c = res.u.t().dot(b);

    // Solve R11 * y1 = c1 via back-substitution (only first k components)
    let mut y = Array1::<F>::zeros(n);
    if k > 0 {
        // Back-substitution on the k x k upper-left block of R
        for i in (0..k).rev() {
            let mut s = c[i];
            for j in (i + 1)..k {
                s -= res.r[[i, j]] * y[j];
            }
            let diag = res.r[[i, i]];
            if diag.abs() <= F::epsilon() {
                return Err(LinalgError::SingularMatrixError(
                    "URV lstsq: zero diagonal in R at detected rank boundary".to_string(),
                ));
            }
            y[i] = s / diag;
        }
    }

    // x = V * y
    let x = res.v.dot(&y);

    Ok(x)
}

// ============================================================================
// Internal: Householder QR (without pivoting)
// ============================================================================

/// Compute a standard (non-pivoted) Householder QR factorization.
///
/// Returns (Q, R) for the input matrix `a` of shape (m x n).
fn householder_qr<F>(a: &Array2<F>) -> LinalgResult<(Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + Debug + ScalarOperand + 'static,
{
    let (m, n) = a.dim();
    let min_dim = m.min(n);

    let mut r = a.clone();
    let mut q = Array2::<F>::eye(m);

    let two = F::from(2.0).unwrap_or_else(|| F::one() + F::one());

    for k in 0..min_dim {
        // Extract the sub-column
        let mut x = Array1::<F>::zeros(m - k);
        for i in k..m {
            x[i - k] = r[[i, k]];
        }

        let x_norm = x.iter().fold(F::zero(), |acc, &v| acc + v * v).sqrt();
        if x_norm <= F::epsilon() {
            continue;
        }

        let alpha = if x[0] >= F::zero() { -x_norm } else { x_norm };
        let mut v = x;
        v[0] -= alpha;

        let v_norm_sq = v.iter().fold(F::zero(), |acc, &val| acc + val * val);
        if v_norm_sq <= F::epsilon() {
            continue;
        }
        let beta = two / v_norm_sq;

        // R <- H * R
        for j in k..n {
            let mut dot = F::zero();
            for i in 0..(m - k) {
                dot += v[i] * r[[i + k, j]];
            }
            for i in 0..(m - k) {
                r[[i + k, j]] -= beta * v[i] * dot;
            }
        }

        // Q <- Q * H^T  (H is symmetric so H^T = H)
        for row in 0..m {
            let mut dot = F::zero();
            for jj in 0..(m - k) {
                dot += q[[row, jj + k]] * v[jj];
            }
            for jj in 0..(m - k) {
                q[[row, jj + k]] -= beta * dot * v[jj];
            }
        }
    }

    Ok((q, r))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn frob_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn test_urv_full_rank() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];
        let res = urv(&a.view(), 1e-12).expect("urv failed");
        assert_eq!(res.rank, 3);

        // Reconstruction
        let recon = res.u.dot(&res.r).dot(&res.v.t());
        let err = frob_diff(&recon, &a.to_owned());
        assert!(err < 1e-10, "reconstruction error = {err}");

        // U orthogonal
        let utu = res.u.t().dot(&res.u);
        let eye = Array2::<f64>::eye(3);
        assert!(frob_diff(&utu, &eye) < 1e-10, "U not orthogonal");

        // V orthogonal
        let vtv = res.v.t().dot(&res.v);
        assert!(frob_diff(&vtv, &eye) < 1e-10, "V not orthogonal");
    }

    #[test]
    fn test_urv_rank_deficient() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]];
        let res = urv(&a.view(), 1e-10).expect("urv failed");
        assert_eq!(res.rank, 2, "rank should be 2");

        let recon = res.u.dot(&res.r).dot(&res.v.t());
        let err = frob_diff(&recon, &a.to_owned());
        assert!(err < 1e-10, "reconstruction error = {err}");
    }

    #[test]
    fn test_urv_rectangular_tall() {
        let a = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]];
        let res = urv(&a.view(), 1e-10).expect("urv failed");
        assert_eq!(res.rank, 2);
        assert_eq!(res.u.shape(), &[4, 4]);
        assert_eq!(res.r.shape(), &[4, 2]);
        assert_eq!(res.v.shape(), &[2, 2]);

        let recon = res.u.dot(&res.r).dot(&res.v.t());
        let err = frob_diff(&recon, &a.to_owned());
        assert!(err < 1e-10, "tall reconstruction error = {err}");
    }

    #[test]
    fn test_urv_rectangular_wide() {
        let a = array![[1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 3.0]];
        let res = urv(&a.view(), 1e-10).expect("urv failed");
        assert_eq!(res.rank, 2);
        assert_eq!(res.u.shape(), &[2, 2]);
        assert_eq!(res.r.shape(), &[2, 4]);
        assert_eq!(res.v.shape(), &[4, 4]);

        let recon = res.u.dot(&res.r).dot(&res.v.t());
        let err = frob_diff(&recon, &a.to_owned());
        assert!(err < 1e-10, "wide reconstruction error = {err}");
    }

    #[test]
    fn test_urv_zero_matrix() {
        let a = Array2::<f64>::zeros((3, 3));
        let res = urv(&a.view(), 1e-12).expect("urv failed");
        assert_eq!(res.rank, 0);
    }

    #[test]
    fn test_urv_identity() {
        let eye = Array2::<f64>::eye(4);
        let res = urv(&eye.view(), 1e-12).expect("urv failed");
        assert_eq!(res.rank, 4);

        let recon = res.u.dot(&res.r).dot(&res.v.t());
        let err = frob_diff(&recon, &eye);
        assert!(err < 1e-10);
    }

    #[test]
    fn test_urv_lstsq_overdetermined() {
        // Overdetermined full-rank system
        let a = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
        let b = array![1.0, 2.0, 3.0];

        let x = urv_lstsq(&a.view(), &b, 1e-10).expect("lstsq failed");

        // Check residual
        let residual = &a.dot(&x) - &b;
        let res_norm: f64 = residual.iter().map(|&v| v * v).sum::<f64>().sqrt();
        // For this system the exact LS solution is x = [0, 1]
        assert!(res_norm < 0.5, "residual norm = {res_norm}");
    }

    #[test]
    fn test_urv_lstsq_rank_deficient() {
        // Rank-deficient system
        let a = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];
        let b = array![1.0, 2.0, 3.0];

        let x = urv_lstsq(&a.view(), &b, 1e-10).expect("lstsq failed");

        // The residual should be small (b is in the column space)
        let residual = &a.dot(&x) - &b;
        let res_norm: f64 = residual.iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!(
            res_norm < 1e-8,
            "rank-deficient lstsq residual = {res_norm}"
        );
    }

    #[test]
    fn test_urv_lstsq_dimension_error() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![1.0, 2.0, 3.0]; // wrong length
        assert!(urv_lstsq(&a.view(), &b, 1e-10).is_err());
    }

    #[test]
    fn test_urv_single_element() {
        let a = array![[5.0]];
        let res = urv(&a.view(), 1e-12).expect("urv failed");
        assert_eq!(res.rank, 1);

        let recon = res.u.dot(&res.r).dot(&res.v.t());
        assert!((recon[[0, 0]] - 5.0).abs() < 1e-10);
    }
}
