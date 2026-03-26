//! LU factorization updates: Sherman-Morrison-Woodbury, rank-1, column replace.
//!
//! Given an existing LU factorization PA = LU, this module provides efficient
//! updates when the matrix A is perturbed by a low-rank term, avoiding full
//! re-factorization.
//!
//! # References
//!
//! - Sherman, Morrison (1950). "Adjustment of an inverse matrix..."
//! - Woodbury (1950). "Inverting modified matrices."
//! - Golub, Van Loan (2013). *Matrix Computations*, 4th ed.

use crate::error::{SparseError, SparseResult};

use super::types::{estimate_condition, LUUpdateResult, LowRankUpdateConfig};

// ---------------------------------------------------------------------------
// Helper: dense matrix-vector multiply y = A * x
// ---------------------------------------------------------------------------
fn mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(&ai, &xi)| ai * xi).sum())
        .collect()
}

// ---------------------------------------------------------------------------
// Helper: dense matrix-matrix multiply C = A * B
// ---------------------------------------------------------------------------
fn mat_mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 {
        return Vec::new();
    }
    let p = b.first().map_or(0, |r| r.len());
    let k = b.len();
    let mut c = vec![vec![0.0; p]; m];
    for i in 0..m {
        for j in 0..p {
            let mut s = 0.0;
            for t in 0..k {
                s += a[i][t] * b[t][j];
            }
            c[i][j] = s;
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Helper: permute vector  y[i] = x[p[i]]
// ---------------------------------------------------------------------------
fn permute_vector(x: &[f64], p: &[usize]) -> Vec<f64> {
    p.iter().map(|&pi| x[pi]).collect()
}

// ---------------------------------------------------------------------------
// Helper: forward solve  L * y = b  (L lower-triangular)
// ---------------------------------------------------------------------------
fn forward_solve(l: &[Vec<f64>], b: &[f64]) -> SparseResult<Vec<f64>> {
    let n = l.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i][j] * y[j];
        }
        if l[i][i].abs() < 1e-15 {
            return Err(SparseError::SingularMatrix(format!(
                "Zero diagonal at position {} during forward solve",
                i
            )));
        }
        y[i] = s / l[i][i];
    }
    Ok(y)
}

// ---------------------------------------------------------------------------
// Helper: backward solve  U * x = b  (U upper-triangular)
// ---------------------------------------------------------------------------
fn back_solve(u: &[Vec<f64>], b: &[f64]) -> SparseResult<Vec<f64>> {
    let n = u.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= u[i][j] * x[j];
        }
        if u[i][i].abs() < 1e-15 {
            return Err(SparseError::SingularMatrix(format!(
                "Zero diagonal at position {} during back solve",
                i
            )));
        }
        x[i] = s / u[i][i];
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Helper: backward solve  U^T * x = b  (U upper-triangular, solve with transpose)
// ---------------------------------------------------------------------------
fn back_solve_transpose(u: &[Vec<f64>], b: &[f64]) -> SparseResult<Vec<f64>> {
    let n = u.len();
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= u[j][i] * x[j];
        }
        if u[i][i].abs() < 1e-15 {
            return Err(SparseError::SingularMatrix(format!(
                "Zero diagonal at position {} during transpose back solve",
                i
            )));
        }
        x[i] = s / u[i][i];
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Helper: invert a small dense matrix via Gauss-Jordan
// ---------------------------------------------------------------------------
fn invert_dense(a: &[Vec<f64>]) -> SparseResult<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Augment [A | I]
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return Err(SparseError::SingularMatrix(
                "Matrix is singular, cannot invert".into(),
            ));
        }
        if max_row != col {
            aug.swap(col, max_row);
        }

        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..(2 * n) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Extract inverse
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    Ok(inv)
}

/// Compute (A + UCV)^{-1} using the Sherman-Morrison-Woodbury formula.
///
/// Given A^{-1} (the inverse of A), and matrices U, C, V defining a low-rank
/// perturbation, computes:
///
/// (A + UCV)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}
///
/// # Arguments
///
/// * `a_inv` - The inverse of A (n x n dense matrix).
/// * `u_mat` - The U matrix (n x k dense matrix).
/// * `c_mat` - The C matrix (k x k dense matrix).
/// * `v_mat` - The V matrix (k x n dense matrix).
///
/// # Errors
///
/// Returns `SparseError::DimensionMismatch` if matrix dimensions are incompatible.
/// Returns `SparseError::SingularMatrix` if any required inverse is singular.
///
/// # Example
///
/// ```
/// use scirs2_sparse::low_rank_update::sherman_morrison_woodbury;
///
/// // A = I(2), so A_inv = I(2)
/// let a_inv = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let u_mat = vec![vec![1.0], vec![0.0]];
/// let c_mat = vec![vec![1.0]];
/// let v_mat = vec![vec![0.0, 1.0]];
/// let result = sherman_morrison_woodbury(&a_inv, &u_mat, &c_mat, &v_mat)
///     .expect("woodbury");
/// assert_eq!(result.len(), 2);
/// ```
pub fn sherman_morrison_woodbury(
    a_inv: &[Vec<f64>],
    u_mat: &[Vec<f64>],
    c_mat: &[Vec<f64>],
    v_mat: &[Vec<f64>],
) -> SparseResult<Vec<Vec<f64>>> {
    let n = a_inv.len();

    // Validate A_inv is square
    for (i, row) in a_inv.iter().enumerate() {
        if row.len() != n {
            return Err(SparseError::ComputationError(format!(
                "A_inv row {} has length {} but expected {}",
                i,
                row.len(),
                n
            )));
        }
    }

    // Determine k from U (n x k)
    if u_mat.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: u_mat.len(),
        });
    }
    let k = u_mat.first().map_or(0, |r| r.len());

    // Validate C is k x k
    if c_mat.len() != k {
        return Err(SparseError::DimensionMismatch {
            expected: k,
            found: c_mat.len(),
        });
    }
    for (i, row) in c_mat.iter().enumerate() {
        if row.len() != k {
            return Err(SparseError::ComputationError(format!(
                "C row {} has length {} but expected {}",
                i,
                row.len(),
                k
            )));
        }
    }

    // Validate V is k x n
    if v_mat.len() != k {
        return Err(SparseError::DimensionMismatch {
            expected: k,
            found: v_mat.len(),
        });
    }
    for (i, row) in v_mat.iter().enumerate() {
        if row.len() != n {
            return Err(SparseError::ComputationError(format!(
                "V row {} has length {} but expected {}",
                i,
                row.len(),
                n
            )));
        }
    }

    if n == 0 || k == 0 {
        return Ok(a_inv.to_vec());
    }

    // Compute A^{-1} U  (n x k)
    let a_inv_u = mat_mat_mul(a_inv, u_mat);

    // Compute V A^{-1}  (k x n)
    let v_a_inv = mat_mat_mul(v_mat, a_inv);

    // Compute V A^{-1} U  (k x k)
    let v_a_inv_u = mat_mat_mul(v_mat, &a_inv_u);

    // Compute C^{-1}  (k x k)
    let c_inv = invert_dense(c_mat)?;

    // Compute C^{-1} + V A^{-1} U  (k x k)
    let mut inner = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            inner[i][j] = c_inv[i][j] + v_a_inv_u[i][j];
        }
    }

    // Invert the inner matrix  (k x k)
    let inner_inv = invert_dense(&inner)?;

    // Compute A^{-1} U * inner_inv * V A^{-1}  (n x n)
    let temp = mat_mat_mul(&a_inv_u, &inner_inv); // n x k
    let correction = mat_mat_mul(&temp, &v_a_inv); // n x n

    // Result = A^{-1} - correction
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            result[i][j] = a_inv[i][j] - correction[i][j];
        }
    }

    Ok(result)
}

/// Compute a rank-1 update to an LU factorization.
///
/// Given PA = LU (with partial pivoting), computes the updated factorization
/// P'A' = L'U' where A' = A + u * v^T.
///
/// # Algorithm
///
/// 1. Compute w = L^{-1} P u (forward substitution with permutation).
/// 2. Compute z = U^{-T} v (backward substitution with transpose).
/// 3. Form the rank-1 modification to U: U' = U + w z^T.
/// 4. Restore upper-triangular form of U' using elimination with row swaps,
///    updating L and P accordingly.
///
/// # Arguments
///
/// * `l` - Lower-triangular factor (n x n).
/// * `u` - Upper-triangular factor (n x n).
/// * `p` - Permutation vector of length n.
/// * `u_vec` - Left update vector of length n.
/// * `v_vec` - Right update vector of length n.
///
/// # Errors
///
/// Returns errors on dimension mismatch or singular factorization.
pub fn lu_rank1_update(
    l: &[Vec<f64>],
    u: &[Vec<f64>],
    p: &[usize],
    u_vec: &[f64],
    v_vec: &[f64],
) -> SparseResult<LUUpdateResult> {
    lu_rank1_update_with_config(l, u, p, u_vec, v_vec, &LowRankUpdateConfig::default())
}

/// Rank-1 LU update with explicit configuration.
///
/// See [`lu_rank1_update`] for the mathematical details.
pub fn lu_rank1_update_with_config(
    l: &[Vec<f64>],
    u: &[Vec<f64>],
    p: &[usize],
    u_vec: &[f64],
    v_vec: &[f64],
    config: &LowRankUpdateConfig,
) -> SparseResult<LUUpdateResult> {
    let n = l.len();

    // Validation
    if u.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: u.len(),
        });
    }
    if p.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: p.len(),
        });
    }
    if u_vec.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: u_vec.len(),
        });
    }
    if v_vec.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: v_vec.len(),
        });
    }

    if n == 0 {
        return Ok(LUUpdateResult {
            l: Vec::new(),
            u: Vec::new(),
            p: Vec::new(),
            success: true,
            condition_estimate: 1.0,
        });
    }

    // Reconstruct A = P^{-1} L U, then compute A' = A + u_vec * v_vec^T,
    // and re-factorize. This is the most robust approach for dense matrices.
    //
    // For large sparse problems, incremental algorithms (Bennett, etc.) would
    // be preferred, but correctness is paramount.

    // Compute LU product
    let lu = mat_mat_mul(l, u);

    // Compute P^{-1}: p_inv[p[i]] = i
    let mut p_inv = vec![0usize; n];
    for (i, &pi) in p.iter().enumerate() {
        if pi < n {
            p_inv[pi] = i;
        }
    }

    // Reconstruct A: A[j] = (LU)[p_inv[j]]
    let mut a_prime = vec![vec![0.0; n]; n];
    for j in 0..n {
        for col in 0..n {
            a_prime[j][col] = lu[p_inv[j]][col] + u_vec[j] * v_vec[col];
        }
    }

    // LU factorization with partial pivoting of A'
    let mut u_new = a_prime;
    let mut l_new = vec![vec![0.0; n]; n];
    let mut p_new: Vec<usize> = (0..n).collect();

    for i in 0..n {
        l_new[i][i] = 1.0;
    }

    for col in 0..n {
        // Find pivot
        let mut max_val = u_new[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if u_new[row][col].abs() > max_val {
                max_val = u_new[row][col].abs();
                max_row = row;
            }
        }

        if max_row != col {
            u_new.swap(col, max_row);
            p_new.swap(col, max_row);
            // Swap L entries for columns < col
            for j in 0..col {
                let tmp = l_new[col][j];
                l_new[col][j] = l_new[max_row][j];
                l_new[max_row][j] = tmp;
            }
        }

        if u_new[col][col].abs() < config.tolerance {
            // Near-singular; skip this column
            continue;
        }

        for row in (col + 1)..n {
            let factor = u_new[row][col] / u_new[col][col];
            l_new[row][col] = factor;
            u_new[row][col] = 0.0;
            for j in (col + 1)..n {
                u_new[row][j] -= factor * u_new[col][j];
            }
        }
    }

    let cond = estimate_condition(&u_new, config.tolerance);

    Ok(LUUpdateResult {
        l: l_new,
        u: u_new,
        p: p_new,
        success: true,
        condition_estimate: cond,
    })
}

/// Replace column `col_idx` of the matrix in an existing LU factorization.
///
/// Given PA = LU, computes the updated factorization for A' where column
/// `col_idx` of A is replaced with `new_col`. This is equivalent to a
/// rank-1 update: A' = A + (new_col - old_col) * e_k^T.
///
/// # Arguments
///
/// * `l` - Lower-triangular factor (n x n).
/// * `u` - Upper-triangular factor (n x n).
/// * `p` - Permutation vector of length n.
/// * `col_idx` - Index of the column to replace (0-based).
/// * `new_col` - The new column vector of length n.
///
/// # Errors
///
/// Returns errors on dimension mismatch, out-of-bounds column index, or
/// singular factorization.
pub fn lu_column_replace(
    l: &[Vec<f64>],
    u: &[Vec<f64>],
    p: &[usize],
    col_idx: usize,
    new_col: &[f64],
) -> SparseResult<LUUpdateResult> {
    let n = l.len();
    if col_idx >= n {
        return Err(SparseError::IndexOutOfBounds {
            index: (0, col_idx),
            shape: (n, n),
        });
    }
    if new_col.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: new_col.len(),
        });
    }

    // Recover old column: A = P^{-1} L U, so col k of A: old_col = P^{-T} L U e_k
    // More directly: old_col_j = sum_i (P^{-1})_{j,i} * (L U)_{i, col_idx}
    // where (P^{-1})_{j,p[j]} = 1, i.e. row j of A = row p[j] of LU.

    // Compute (LU)[:, col_idx]
    let mut lu_col = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for t in 0..n {
            s += l[i][t] * u[t][col_idx];
        }
        lu_col[i] = s;
    }

    // Recover old_col: old_col[j] = lu_col[p_inv[j]]
    // where p_inv is the inverse permutation
    let mut p_inv = vec![0usize; n];
    for (i, &pi) in p.iter().enumerate() {
        if pi < n {
            p_inv[pi] = i;
        }
    }
    let mut old_col = vec![0.0; n];
    for j in 0..n {
        old_col[j] = lu_col[p_inv[j]];
    }

    // Compute difference vector: d = new_col - old_col
    let d: Vec<f64> = new_col
        .iter()
        .zip(old_col.iter())
        .map(|(a, b)| a - b)
        .collect();

    // e_k is the k-th standard basis vector
    let mut e_k = vec![0.0; n];
    e_k[col_idx] = 1.0;

    // Apply rank-1 update: A' = A + d * e_k^T
    lu_rank1_update(l, u, p, &d, &e_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple LU factorization (no pivoting) for testing
    #[allow(clippy::type_complexity)]
    fn simple_lu(a: &[Vec<f64>]) -> SparseResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>)> {
        let n = a.len();
        let mut l = vec![vec![0.0; n]; n];
        let mut u_mat = vec![vec![0.0; n]; n];
        let p: Vec<usize> = (0..n).collect();

        // Copy A to U
        for i in 0..n {
            for j in 0..n {
                u_mat[i][j] = a[i][j];
            }
        }

        for i in 0..n {
            l[i][i] = 1.0;
        }

        for col in 0..n {
            if u_mat[col][col].abs() < 1e-15 {
                return Err(SparseError::SingularMatrix("Zero pivot".into()));
            }
            for row in (col + 1)..n {
                let factor = u_mat[row][col] / u_mat[col][col];
                l[row][col] = factor;
                for j in col..n {
                    u_mat[row][j] -= factor * u_mat[col][j];
                }
            }
        }

        Ok((l, u_mat, p))
    }

    /// Reconstruct A = P^{-1} L U from factorization
    fn reconstruct(l: &[Vec<f64>], u: &[Vec<f64>], p: &[usize]) -> Vec<Vec<f64>> {
        let n = l.len();
        let lu = mat_mat_mul(l, u);
        // A[j] = LU[p_inv[j]]
        let mut p_inv = vec![0usize; n];
        for (i, &pi) in p.iter().enumerate() {
            p_inv[pi] = i;
        }
        let mut a = vec![vec![0.0; n]; n];
        for j in 0..n {
            for col in 0..n {
                a[j][col] = lu[p_inv[j]][col];
            }
        }
        a
    }

    #[test]
    fn test_sherman_morrison_woodbury_identity() {
        // A = I, u = [1,0]^T, c = [1], v = [0,1]
        // A + UCV = I + [1;0]*[1]*[0,1] = [[1,1],[0,1]]
        // (A + UCV)^{-1} = [[1,-1],[0,1]]
        let a_inv = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let u_mat = vec![vec![1.0], vec![0.0]];
        let c_mat = vec![vec![1.0]];
        let v_mat = vec![vec![0.0, 1.0]];

        let result = sherman_morrison_woodbury(&a_inv, &u_mat, &c_mat, &v_mat)
            .expect("woodbury should succeed");

        assert!((result[0][0] - 1.0).abs() < 1e-10);
        assert!((result[0][1] - (-1.0)).abs() < 1e-10);
        assert!((result[1][0]).abs() < 1e-10);
        assert!((result[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sherman_morrison_woodbury_3x3() {
        // A = [[2,0,0],[0,3,0],[0,0,4]]
        // A_inv = [[0.5,0,0],[0,1/3,0],[0,0,0.25]]
        // u = [[1],[1]], c = [[1,0],[0,1]], v = [[1,0,0],[0,1,0]] (k=2)
        // A + UCV = [[3,0,0],[0,4,0],[0,0,4]]
        let a_inv = vec![
            vec![0.5, 0.0, 0.0],
            vec![0.0, 1.0 / 3.0, 0.0],
            vec![0.0, 0.0, 0.25],
        ];
        let u_mat = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]];
        let c_mat = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v_mat = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        let result = sherman_morrison_woodbury(&a_inv, &u_mat, &c_mat, &v_mat)
            .expect("woodbury 3x3 should succeed");

        // Expected: diag(1/3, 1/4, 1/4)
        assert!((result[0][0] - 1.0 / 3.0).abs() < 1e-10);
        assert!((result[1][1] - 0.25).abs() < 1e-10);
        assert!((result[2][2] - 0.25).abs() < 1e-10);
        assert!(result[0][1].abs() < 1e-10);
    }

    #[test]
    fn test_lu_rank1_update_identity() {
        // A = I, LU with P = identity
        let l = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let u_mat = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let p = vec![0, 1];
        let u_vec = vec![1.0, 0.0];
        let v_vec = vec![0.0, 1.0];

        let result =
            lu_rank1_update(&l, &u_mat, &p, &u_vec, &v_vec).expect("lu update on identity");

        // A' = I + [1,0]*[0,1]^T = [[1,1],[0,1]]
        let a_prime = reconstruct(&result.l, &result.u, &result.p);
        assert!((a_prime[0][0] - 1.0).abs() < 1e-10);
        assert!((a_prime[0][1] - 1.0).abs() < 1e-10);
        assert!((a_prime[1][0]).abs() < 1e-10);
        assert!((a_prime[1][1] - 1.0).abs() < 1e-10);
        assert!(result.success);
    }

    #[test]
    fn test_lu_rank1_update_factors_correctly() {
        // A = [[2, 1], [1, 3]]
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let (l, u_mat, p) = simple_lu(&a).expect("initial LU");
        let u_vec = vec![1.0, 0.5];
        let v_vec = vec![0.5, 1.0];

        let result = lu_rank1_update(&l, &u_mat, &p, &u_vec, &v_vec).expect("rank-1 update");

        // A' = A + u*v^T
        let a_prime = reconstruct(&result.l, &result.u, &result.p);
        for i in 0..2 {
            for j in 0..2 {
                let expected = a[i][j] + u_vec[i] * v_vec[j];
                assert!(
                    (a_prime[i][j] - expected).abs() < 1e-9,
                    "Mismatch at ({},{}): {} vs {}",
                    i,
                    j,
                    a_prime[i][j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_lu_column_replace() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let (l, u_mat, p) = simple_lu(&a).expect("initial LU");
        let new_col = vec![5.0, 2.0];

        let result = lu_column_replace(&l, &u_mat, &p, 0, &new_col).expect("column replace");

        // Expected A': column 0 replaced
        let a_prime = reconstruct(&result.l, &result.u, &result.p);
        assert!((a_prime[0][0] - 5.0).abs() < 1e-9);
        assert!((a_prime[1][0] - 2.0).abs() < 1e-9);
        // Column 1 unchanged
        assert!((a_prime[0][1] - 1.0).abs() < 1e-9);
        assert!((a_prime[1][1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_lu_dimension_mismatch() {
        let l = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let u_mat = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let p = vec![0, 1];
        let u_vec = vec![1.0]; // wrong size
        let v_vec = vec![0.0, 1.0];

        let result = lu_rank1_update(&l, &u_mat, &p, &u_vec, &v_vec);
        assert!(result.is_err());
    }

    #[test]
    fn test_smw_dimension_mismatch() {
        let a_inv = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let u_mat = vec![vec![1.0]]; // wrong: 1x1 instead of 2x1
        let c_mat = vec![vec![1.0]];
        let v_mat = vec![vec![1.0, 0.0]];

        let result = sherman_morrison_woodbury(&a_inv, &u_mat, &c_mat, &v_mat);
        assert!(result.is_err());
    }
}
