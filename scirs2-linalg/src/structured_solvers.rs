//! Structured matrix solvers for specialized linear systems
//!
//! This module provides efficient solvers for linear systems with structured
//! coefficient matrices, including banded, symmetric, triangular, and Toeplitz
//! systems. These exploit the matrix structure for significant speedups and
//! reduced memory usage compared to general solvers.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

/// Solve a banded linear system Ab * x = b.
///
/// The banded matrix is stored in the LAPACK banded format: a 2D array with
/// (lower + upper + 1) rows and n columns, where row (upper + i - j) stores
/// element `A[i,j]` for the band.
///
/// Alternatively, (l, u) specifies the lower and upper bandwidths and `ab` is
/// the compact banded storage.
///
/// # Arguments
///
/// * `l` - Number of lower diagonals
/// * `u` - Number of upper diagonals
/// * `ab` - Banded matrix in compact storage, shape ((l + u + 1), n)
/// * `b` - Right-hand side vector of length n
///
/// # Returns
///
/// * Solution vector x of length n
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_linalg::structured_solvers::solve_banded;
///
/// // Tridiagonal system: [-1, 2, -1] * x = [1, 0, 1]
/// // Compact form: rows = [upper, main, lower]
/// let ab = Array2::from_shape_vec((3, 3), vec![
///     0.0, -1.0, -1.0,  // upper diagonal
///     2.0,  2.0,  2.0,  // main diagonal
///    -1.0, -1.0,  0.0,  // lower diagonal
/// ]).expect("shape");
/// let b = array![1.0_f64, 0.0, 1.0];
///
/// let x = solve_banded(1, 1, &ab.view(), &b.view()).expect("ok");
/// assert!(x.len() == 3);
/// ```
pub fn solve_banded<F>(
    l: usize,
    u: usize,
    ab: &ArrayView2<F>,
    b: &ArrayView1<F>,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let bandwidth = l + u + 1;
    let (ab_rows, n) = (ab.nrows(), ab.ncols());

    if ab_rows != bandwidth {
        return Err(LinalgError::ShapeError(format!(
            "Banded matrix should have {} rows (l+u+1), got {}",
            bandwidth, ab_rows
        )));
    }
    if b.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "RHS length {} does not match matrix size {}",
            b.len(),
            n
        )));
    }
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    // Extract the full banded matrix into a dense format for LU-like factoring
    // For efficiency, use a banded LU factorization (column-based Gaussian elimination)

    // Build working copy of the banded storage (we'll modify it)
    let mut ab_work = ab.to_owned();
    let mut x = b.to_owned();

    // Forward elimination with partial pivoting within the band
    for k in 0..n {
        // Find the pivot (largest in the column within the lower band)
        let mut max_val = ab_work[[u, k]].abs();
        let mut max_idx = 0_usize;

        let band_end = (l + 1).min(n - k);
        for i in 1..band_end {
            let val = ab_work[[u + i, k]].abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        if max_val < F::epsilon() * F::from(100.0).unwrap_or(F::one()) {
            return Err(LinalgError::SingularMatrixError(
                "Banded matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows if needed (within the banded structure)
        if max_idx != 0 {
            // Swap entries in ab_work and x
            // This is tricky in banded format; for simplicity, swap the
            // relevant portions
            let pivot_row = u + max_idx;
            for j in k..n.min(k + l + u + 1) {
                let col_offset_orig = j as isize - k as isize;
                let col_offset_pivot = col_offset_orig - max_idx as isize;

                if col_offset_orig >= 0
                    && (col_offset_orig as usize) < bandwidth
                    && col_offset_pivot >= 0
                    && (col_offset_pivot as usize) < bandwidth
                {
                    let r1 = u.wrapping_add_signed(-(col_offset_orig - u as isize));
                    let r2 = pivot_row.wrapping_add_signed(-(col_offset_orig - u as isize));
                    if r1 < bandwidth && r2 < bandwidth {
                        let tmp = ab_work[[r1, j]];
                        ab_work[[r1, j]] = ab_work[[r2, j]];
                        ab_work[[r2, j]] = tmp;
                    }
                }
            }
            // Swap RHS
            let tmp = x[k];
            x[k] = x[k + max_idx];
            x[k + max_idx] = tmp;
        }

        let pivot = ab_work[[u, k]];

        // Eliminate below the pivot
        for i in 1..band_end {
            let factor = ab_work[[u + i, k]] / pivot;
            ab_work[[u + i, k]] = F::zero();

            // Update the remaining entries in the row
            for j in (k + 1)..n.min(k + u + 1) {
                let row_in_band = u + i - (j - k);
                if row_in_band < bandwidth {
                    ab_work[[row_in_band, j]] =
                        ab_work[[row_in_band, j]] - factor * ab_work[[u - (j - k), j]];
                }
            }

            x[k + i] = x[k + i] - factor * x[k];
        }
    }

    // Back substitution
    for k in (0..n).rev() {
        let pivot = ab_work[[u, k]];
        if pivot.abs() < F::epsilon() * F::from(100.0).unwrap_or(F::one()) {
            return Err(LinalgError::SingularMatrixError(
                "Banded matrix is singular in back substitution".to_string(),
            ));
        }

        let mut sum = x[k];
        for j in (k + 1)..n.min(k + u + 1) {
            let row_in_band = u - (j - k);
            sum -= ab_work[[row_in_band, j]] * x[j];
        }
        x[k] = sum / pivot;
    }

    Ok(x)
}

/// Solve a symmetric positive-definite linear system A * x = b using Cholesky
/// decomposition.
///
/// This is more efficient than general solve for SPD matrices, using
/// L * L^T = A, then forward/back substitution.
///
/// # Arguments
///
/// * `a` - Symmetric positive-definite matrix (n x n)
/// * `b` - Right-hand side vector of length n
///
/// # Returns
///
/// * Solution vector x
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::structured_solvers::solve_symmetric;
///
/// let a = array![[4.0_f64, 2.0], [2.0, 3.0]];
/// let b = array![6.0_f64, 7.0];
/// let x = solve_symmetric(&a.view(), &b.view()).expect("ok");
/// // Check: A * x ~ b
/// ```
pub fn solve_symmetric<F>(a: &ArrayView2<F>, b: &ArrayView1<F>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + std::fmt::Display + 'static,
{
    let (m, n) = (a.nrows(), a.ncols());
    if m != n {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square, got {}x{}",
            m, n
        )));
    }
    if n != b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix size {} does not match vector length {}",
            n,
            b.len()
        )));
    }

    // Check symmetry
    let sym_tol = F::epsilon() * F::from(n as f64 * 100.0).unwrap_or(F::one());
    for i in 0..n {
        for j in (i + 1)..n {
            if (a[[i, j]] - a[[j, i]]).abs() > sym_tol {
                return Err(LinalgError::InvalidInputError(format!(
                    "Matrix is not symmetric: a[{},{}]={} vs a[{},{}]={}",
                    i,
                    j,
                    a[[i, j]],
                    j,
                    i,
                    a[[j, i]]
                )));
            }
        }
    }

    // Cholesky decomposition: A = L * L^T
    let l = crate::cholesky(a, None)?;

    // Forward substitution: L * y = b
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        if l[[i, i]].abs() < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Zero diagonal in Cholesky factor".to_string(),
            ));
        }
        y[i] = sum / l[[i, i]];
    }

    // Back substitution: L^T * x = y
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[[j, i]] * x[j]; // L^T[i,j] = L[j,i]
        }
        if l[[i, i]].abs() < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Zero diagonal in Cholesky factor".to_string(),
            ));
        }
        x[i] = sum / l[[i, i]];
    }

    Ok(x)
}

/// Solve a symmetric positive-definite system with multiple right-hand sides.
///
/// # Arguments
///
/// * `a` - SPD matrix (n x n)
/// * `b` - Matrix of RHS vectors (n x m)
///
/// # Returns
///
/// * Solution matrix X (n x m)
pub fn solve_symmetric_multiple<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + std::fmt::Display + 'static,
{
    let (n, _) = (a.nrows(), a.ncols());
    let (_, nrhs) = (b.nrows(), b.ncols());

    let mut result = Array2::zeros((n, nrhs));
    for j in 0..nrhs {
        let b_col = b.column(j);
        let x = solve_symmetric(a, &b_col)?;
        for i in 0..n {
            result[[i, j]] = x[i];
        }
    }

    Ok(result)
}

/// Solve a Toeplitz system T * x = b using the Levinson-Durbin recursion.
///
/// The Toeplitz matrix is defined by its first row r and first column c.
/// This is O(n^2) instead of O(n^3) for general solvers.
///
/// # Arguments
///
/// * `c` - First column of the Toeplitz matrix (length n)
/// * `r` - First row of the Toeplitz matrix (length n)
/// * `b` - Right-hand side vector (length n)
///
/// # Returns
///
/// * Solution vector x
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::structured_solvers::solve_toeplitz_levinson;
///
/// let c = array![2.0_f64, 1.0, 0.5];
/// let r = array![2.0_f64, 1.0, 0.5];
/// let b = array![1.0_f64, 2.0, 3.0];
/// let x = solve_toeplitz_levinson(&c.view(), &r.view(), &b.view()).expect("ok");
/// ```
pub fn solve_toeplitz_levinson<F>(
    c: &ArrayView1<F>,
    r: &ArrayView1<F>,
    b: &ArrayView1<F>,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = c.len();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }
    if r.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "Column length {} does not match row length {}",
            n,
            r.len()
        )));
    }
    if b.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "RHS length {} does not match matrix size {}",
            b.len(),
            n
        )));
    }
    if c[0] != r[0] {
        return Err(LinalgError::InvalidInputError(
            "First column and first row must have the same first element".to_string(),
        ));
    }

    let t0 = c[0];
    if t0.abs() < F::epsilon() {
        return Err(LinalgError::SingularMatrixError(
            "Toeplitz matrix has zero diagonal".to_string(),
        ));
    }

    if n == 1 {
        return Ok(Array1::from_vec(vec![b[0] / t0]));
    }

    // Helper: T[i,j] element of the Toeplitz matrix
    // T[i,j] = r[j-i] if j >= i, else c[i-j]
    let t_elem = |i: usize, j: usize| -> F {
        if j >= i {
            r[j - i]
        } else {
            c[i - j]
        }
    };

    // Zohar's adaptation of Levinson-Durbin for general (non-symmetric) Toeplitz.
    // Reference: Golub & Van Loan, "Matrix Computations", Section 4.7.
    //
    // We maintain:
    //   a[0..m] = forward LP coefficients of order m (T_m * [1, a_1, ..., a_m]^T = [e_f, 0,...,0]^T)
    //   aa[0..m] = backward LP coefficients of order m
    //   x[0..m+1] = solution of T_{m+1} x = b_{m+1}
    //   e_f, e_b = forward/backward prediction errors

    let mut a = vec![F::zero(); n]; // forward LP coefficients
    let mut aa = vec![F::zero(); n]; // backward LP coefficients

    // Order 0: a is empty, e_f = e_b = t0
    let mut e_f = t0;
    let mut e_b = t0;

    // x = solution of T_1 * x = b_1 => x[0] = b[0] / t0
    let mut x = vec![F::zero(); n];
    x[0] = b[0] / t0;

    for m in 0..(n - 1) {
        // m is the current order; we're extending to order m+1

        // Compute forward prediction error coefficient
        // e_f_num = c[m+1] + sum_{j=0}^{m-1} a[j] * c[m-j]
        //         = sum_{j=0}^{m} T[m+1, m-j] * (1 if j==m, else a[j]) ... hmm
        // Actually: the forward prediction from the bottom row of T_{m+2}
        // alpha_f = (c[m+1] + a[0]*c[m] + a[1]*c[m-1] + ... + a[m-1]*c[1]) / e_f
        let mut alpha_f = c[m + 1];
        for j in 0..m {
            alpha_f += a[j] * c[m - j];
        }
        alpha_f /= e_f;

        // Compute backward prediction error coefficient
        // alpha_b = (r[m+1] + aa[0]*r[m] + aa[1]*r[m-1] + ... + aa[m-1]*r[1]) / e_b
        let mut alpha_b = r[m + 1];
        for j in 0..m {
            alpha_b += aa[j] * r[m - j];
        }
        alpha_b /= e_b;

        // Update forward and backward LP vectors
        // a_new[j] = a[j] - alpha_f * aa[m-1-j]  for j=0..m-1
        // a_new[m] = -alpha_f
        // aa_new[j] = aa[j] - alpha_b * a[m-1-j]  for j=0..m-1
        // aa_new[m] = -alpha_b
        let a_old = a.clone();
        let aa_old = aa.clone();

        for j in 0..m {
            a[j] = a_old[j] - alpha_f * aa_old[m - 1 - j];
            aa[j] = aa_old[j] - alpha_b * a_old[m - 1 - j];
        }
        a[m] = -alpha_f;
        aa[m] = -alpha_b;

        // Update prediction errors
        e_f *= F::one() - alpha_f * alpha_b;
        e_b = e_f; // For consistent Toeplitz: e_f == e_b after update

        if e_f.abs() < F::epsilon() * F::from(100.0).unwrap_or(F::one()) {
            return Err(LinalgError::SingularMatrixError(
                "Toeplitz matrix is singular (prediction error became zero)".to_string(),
            ));
        }

        // Now extend the solution to order m+2:
        // Compute residual: r_m = b[m+1] - sum_{j=0}^{m} T[m+1, j] * x[j]
        let mut residual = b[m + 1];
        for (j, &xj) in x.iter().enumerate().take(m + 1) {
            residual -= t_elem(m + 1, j) * xj;
        }
        let gamma = residual / e_f;

        // Update x: x_new[j] = x_old[j] + gamma * aa_new[m-j]  for j=0..m
        //           x_new[m+1] = gamma
        // The backward LP vector aa of order m+1 is [aa[0], ..., aa[m], 1] reversed
        // i.e., the "backward" vector is b_rev = [1, aa[m-1], ..., aa[0]] (the reversed coefficients)
        // Actually: x_new = x_old + gamma * backward_vector
        // where backward_vector = [aa[m], aa[m-1], ..., aa[0], 1] for indices 0..m+1
        // Wait, let me think about this more carefully.
        //
        // The backward LP vector of order m+1 is: bb = [aa[0], aa[1], ..., aa[m], 1]
        // (with 1 appended, since backward LP is: T_{m+2} * bb = [0,...,0, e_b])
        //
        // But for the solution update: x_new = x_old_extended + gamma * bb_reversed
        // bb_reversed = [1, aa[m], aa[m-1], ..., aa[0]]
        // No wait. The standard Trench algorithm says:
        // x_new[j] = x_old[j] + gamma * backward_lp_reversed[j]
        // backward_lp_reversed = [aa[m], aa[m-1], ..., aa[0], 1]

        // Let me use the simplest correct formulation:
        // The backward polynomial b(z) = 1 + aa[0]*z + aa[1]*z^2 + ... + aa[m]*z^(m+1)
        // reversed is: z^(m+1) * b(1/z) = aa[m] + aa[m-1]*z + ... + aa[0]*z^m + z^(m+1)
        // So backward_reversed[j] = aa[m-j] for j=0..m, and backward_reversed[m+1] = 1

        for j in 0..=m {
            x[j] += gamma * aa[m - j];
        }
        x[m + 1] = gamma;
    }

    Ok(Array1::from_vec(x))
}

/// Solve a tridiagonal system using the Thomas algorithm (O(n)).
///
/// The system is defined by three diagonals: sub (lower), main (diagonal),
/// and sup (upper).
///
/// # Arguments
///
/// * `sub` - Sub-diagonal, length n-1
/// * `main` - Main diagonal, length n
/// * `sup` - Super-diagonal, length n-1
/// * `b` - Right-hand side vector, length n
///
/// # Returns
///
/// * Solution vector x
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::structured_solvers::solve_tridiagonal;
///
/// // System: [2,-1,0; -1,2,-1; 0,-1,2] * x = [1, 0, 1]
/// let sub = array![-1.0_f64, -1.0];
/// let main = array![2.0_f64, 2.0, 2.0];
/// let sup = array![-1.0_f64, -1.0];
/// let b = array![1.0_f64, 0.0, 1.0];
/// let x = solve_tridiagonal(&sub.view(), &main.view(), &sup.view(), &b.view()).expect("ok");
/// ```
pub fn solve_tridiagonal<F>(
    sub: &ArrayView1<F>,
    main: &ArrayView1<F>,
    sup: &ArrayView1<F>,
    b: &ArrayView1<F>,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let n = main.len();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }
    if sub.len() != n - 1 || sup.len() != n - 1 {
        return Err(LinalgError::ShapeError(format!(
            "Sub/super diagonal lengths must be {}, got sub={}, sup={}",
            n - 1,
            sub.len(),
            sup.len()
        )));
    }
    if b.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "RHS length {} does not match matrix size {}",
            b.len(),
            n
        )));
    }

    if n == 1 {
        if main[0].abs() < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Tridiagonal matrix has zero pivot".to_string(),
            ));
        }
        return Ok(Array1::from_vec(vec![b[0] / main[0]]));
    }

    // Thomas algorithm: forward sweep
    let mut c_prime = vec![F::zero(); n];
    let mut d_prime = vec![F::zero(); n];

    // First row
    if main[0].abs() < F::epsilon() {
        return Err(LinalgError::SingularMatrixError(
            "Tridiagonal matrix has zero pivot at row 0".to_string(),
        ));
    }
    c_prime[0] = sup[0] / main[0];
    d_prime[0] = b[0] / main[0];

    for i in 1..n {
        let denom = main[i] - sub[i - 1] * c_prime[i - 1];
        if denom.abs() < F::epsilon() * F::from(100.0).unwrap_or(F::one()) {
            return Err(LinalgError::SingularMatrixError(format!(
                "Tridiagonal matrix has zero pivot at row {}",
                i
            )));
        }
        if i < n - 1 {
            c_prime[i] = sup[i] / denom;
        }
        d_prime[i] = (b[i] - sub[i - 1] * d_prime[i - 1]) / denom;
    }

    // Back substitution
    let mut x = vec![F::zero(); n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(Array1::from_vec(x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    // Helper: compute dense A*x for verification
    fn dense_matvec(a: &Array2<f64>, x: &Array1<f64>) -> Array1<f64> {
        let n = a.nrows();
        let mut y = Array1::zeros(n);
        for i in 0..n {
            for j in 0..a.ncols() {
                y[i] += a[[i, j]] * x[j];
            }
        }
        y
    }

    // --- solve_symmetric tests ---

    #[test]
    fn test_solve_symmetric_identity() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![3.0_f64, 4.0];
        let x = solve_symmetric(&a.view(), &b.view()).expect("ok");
        assert_relative_eq!(x[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_symmetric_spd() {
        let a = array![[4.0_f64, 2.0], [2.0, 3.0]];
        let b = array![6.0_f64, 7.0];
        let x = solve_symmetric(&a.view(), &b.view()).expect("ok");
        // Verify: A * x = b
        let ax = dense_matvec(&a, &x);
        assert_relative_eq!(ax[0], b[0], epsilon = 1e-10);
        assert_relative_eq!(ax[1], b[1], epsilon = 1e-10);
    }

    #[test]
    fn test_solve_symmetric_3x3() {
        let a = array![[5.0_f64, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 3.0]];
        let b = array![6.0_f64, 5.0, 4.0];
        let x = solve_symmetric(&a.view(), &b.view()).expect("ok");
        let ax = dense_matvec(&a, &x);
        for i in 0..3 {
            assert_relative_eq!(ax[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_solve_symmetric_nonsquare_error() {
        let a = Array2::<f64>::zeros((2, 3));
        let b = array![1.0_f64, 2.0];
        assert!(solve_symmetric(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn test_solve_symmetric_nonsymmetric_error() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![1.0_f64, 2.0];
        assert!(solve_symmetric(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn test_solve_symmetric_dim_mismatch() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![1.0_f64, 2.0, 3.0];
        assert!(solve_symmetric(&a.view(), &b.view()).is_err());
    }

    // --- solve_tridiagonal tests ---

    #[test]
    fn test_solve_tridiagonal_simple() {
        // [2,-1; -1,2] * [x1,x2] = [1, 1]
        let sub = array![-1.0_f64];
        let main = array![2.0_f64, 2.0];
        let sup = array![-1.0_f64];
        let b = array![1.0_f64, 1.0];
        let x = solve_tridiagonal(&sub.view(), &main.view(), &sup.view(), &b.view()).expect("ok");
        // A*x = b check
        assert_relative_eq!(2.0 * x[0] - x[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(-x[0] + 2.0 * x[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_tridiagonal_3x3() {
        let sub = array![-1.0_f64, -1.0];
        let main = array![2.0_f64, 2.0, 2.0];
        let sup = array![-1.0_f64, -1.0];
        let b = array![1.0_f64, 0.0, 1.0];
        let x = solve_tridiagonal(&sub.view(), &main.view(), &sup.view(), &b.view()).expect("ok");
        // Verify
        assert_relative_eq!(2.0 * x[0] - x[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(-x[0] + 2.0 * x[1] - x[2], 0.0, epsilon = 1e-10);
        assert_relative_eq!(-x[1] + 2.0 * x[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_tridiagonal_diagonal() {
        let sub = array![0.0_f64, 0.0];
        let main = array![3.0_f64, 2.0, 5.0];
        let sup = array![0.0_f64, 0.0];
        let b = array![6.0_f64, 8.0, 15.0];
        let x = solve_tridiagonal(&sub.view(), &main.view(), &sup.view(), &b.view()).expect("ok");
        assert_relative_eq!(x[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 4.0, epsilon = 1e-10);
        assert_relative_eq!(x[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_tridiagonal_singular() {
        let sub = array![1.0_f64];
        let main = array![0.0_f64, 0.0];
        let sup = array![1.0_f64];
        let b = array![1.0_f64, 1.0];
        assert!(solve_tridiagonal(&sub.view(), &main.view(), &sup.view(), &b.view()).is_err());
    }

    #[test]
    fn test_solve_tridiagonal_size_mismatch() {
        let sub = array![1.0_f64];
        let main = array![2.0_f64, 2.0, 2.0];
        let sup = array![1.0_f64];
        let b = array![1.0_f64, 1.0, 1.0];
        assert!(solve_tridiagonal(&sub.view(), &main.view(), &sup.view(), &b.view()).is_err());
    }

    #[test]
    fn test_solve_tridiagonal_single() {
        let sub = Array1::<f64>::zeros(0);
        let main = array![5.0_f64];
        let sup = Array1::<f64>::zeros(0);
        let b = array![10.0_f64];
        let x = solve_tridiagonal(&sub.view(), &main.view(), &sup.view(), &b.view()).expect("ok");
        assert_relative_eq!(x[0], 2.0, epsilon = 1e-10);
    }

    // --- solve_banded tests ---

    #[test]
    fn test_solve_banded_diagonal() {
        // Diagonal matrix (l=0, u=0)
        let ab = Array2::from_shape_vec((1, 3), vec![2.0, 3.0, 4.0]).expect("shape");
        let b = array![4.0_f64, 9.0, 16.0];
        let x = solve_banded(0, 0, &ab.view(), &b.view()).expect("ok");
        assert_relative_eq!(x[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(x[2], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_banded_tridiagonal() {
        // Tridiagonal: [[2,-1,0],[-1,2,-1],[0,-1,2]]
        // In banded format (l=1, u=1): 3 rows x 3 cols
        // row 0 (upper): [0, -1, -1]
        // row 1 (main): [2, 2, 2]
        // row 2 (lower): [-1, -1, 0]
        let ab = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, -1.0, -1.0, 2.0, 2.0, 2.0, -1.0, -1.0, 0.0],
        )
        .expect("shape");
        let b = array![1.0_f64, 0.0, 1.0];
        let x = solve_banded(1, 1, &ab.view(), &b.view()).expect("ok");
        // Verify
        assert_relative_eq!(2.0 * x[0] - x[1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(-x[0] + 2.0 * x[1] - x[2], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_solve_banded_shape_error() {
        let ab = Array2::<f64>::zeros((2, 3));
        let b = array![1.0_f64, 2.0, 3.0];
        assert!(solve_banded(1, 1, &ab.view(), &b.view()).is_err()); // expects 3 rows
    }

    #[test]
    fn test_solve_banded_rhs_mismatch() {
        let ab = Array2::<f64>::zeros((3, 3));
        let b = array![1.0_f64, 2.0];
        assert!(solve_banded(1, 1, &ab.view(), &b.view()).is_err());
    }

    #[test]
    fn test_solve_banded_singular() {
        let ab = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).expect("shape");
        let b = array![1.0_f64, 2.0];
        assert!(solve_banded(0, 0, &ab.view(), &b.view()).is_err());
    }

    // --- solve_toeplitz_levinson tests ---

    #[test]
    fn test_solve_toeplitz_identity() {
        // Toeplitz identity: c = [1,0,0], r = [1,0,0]
        let c = array![1.0_f64, 0.0, 0.0];
        let r = array![1.0_f64, 0.0, 0.0];
        let b = array![3.0_f64, 5.0, 7.0];
        let x = solve_toeplitz_levinson(&c.view(), &r.view(), &b.view()).expect("ok");
        assert_relative_eq!(x[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 5.0, epsilon = 1e-10);
        assert_relative_eq!(x[2], 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_toeplitz_symmetric() {
        // Symmetric Toeplitz: [2, 1, 0.5] same for row and col
        let c = array![2.0_f64, 1.0, 0.5];
        let r = array![2.0_f64, 1.0, 0.5];
        let b = array![1.0_f64, 2.0, 3.0];
        let x = solve_toeplitz_levinson(&c.view(), &r.view(), &b.view()).expect("ok");
        // Verify: T * x = b
        let t = array![[2.0, 1.0, 0.5], [1.0, 2.0, 1.0], [0.5, 1.0, 2.0]];
        let tx = dense_matvec(&t, &x);
        for i in 0..3 {
            assert_relative_eq!(tx[i], b[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_solve_toeplitz_nonsymmetric() {
        let c = array![3.0_f64, 1.0];
        let r = array![3.0_f64, 2.0];
        let b = array![5.0_f64, 7.0];
        let x = solve_toeplitz_levinson(&c.view(), &r.view(), &b.view()).expect("ok");
        // T = [[3, 2], [1, 3]], det = 7
        let t = array![[3.0, 2.0], [1.0, 3.0]];
        let tx = dense_matvec(&t, &x);
        assert_relative_eq!(tx[0], b[0], epsilon = 1e-8);
        assert_relative_eq!(tx[1], b[1], epsilon = 1e-8);
    }

    #[test]
    fn test_solve_toeplitz_length_mismatch() {
        let c = array![1.0_f64, 2.0];
        let r = array![1.0_f64, 2.0, 3.0];
        let b = array![1.0_f64, 2.0];
        assert!(solve_toeplitz_levinson(&c.view(), &r.view(), &b.view()).is_err());
    }

    #[test]
    fn test_solve_toeplitz_first_element_mismatch() {
        let c = array![1.0_f64, 2.0];
        let r = array![3.0_f64, 2.0];
        let b = array![1.0_f64, 2.0];
        assert!(solve_toeplitz_levinson(&c.view(), &r.view(), &b.view()).is_err());
    }

    #[test]
    fn test_solve_toeplitz_single() {
        let c = array![5.0_f64];
        let r = array![5.0_f64];
        let b = array![10.0_f64];
        let x = solve_toeplitz_levinson(&c.view(), &r.view(), &b.view()).expect("ok");
        assert_relative_eq!(x[0], 2.0, epsilon = 1e-10);
    }

    // --- solve_symmetric_multiple tests ---

    #[test]
    fn test_solve_symmetric_multiple_identity() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![[3.0_f64, 4.0], [5.0, 6.0]];
        let x = solve_symmetric_multiple(&a.view(), &b.view()).expect("ok");
        assert_relative_eq!(x[[0, 0]], 3.0, epsilon = 1e-10);
        assert_relative_eq!(x[[0, 1]], 4.0, epsilon = 1e-10);
        assert_relative_eq!(x[[1, 0]], 5.0, epsilon = 1e-10);
        assert_relative_eq!(x[[1, 1]], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_symmetric_multiple_spd() {
        let a = array![[4.0_f64, 2.0], [2.0, 3.0]];
        let b = array![[6.0_f64, 4.0], [7.0, 5.0]];
        let x = solve_symmetric_multiple(&a.view(), &b.view()).expect("ok");
        // Check first column
        let ax0 = dense_matvec(&a, &Array1::from_vec(vec![x[[0, 0]], x[[1, 0]]]));
        assert_relative_eq!(ax0[0], 6.0, epsilon = 1e-10);
        assert_relative_eq!(ax0[1], 7.0, epsilon = 1e-10);
    }
}
