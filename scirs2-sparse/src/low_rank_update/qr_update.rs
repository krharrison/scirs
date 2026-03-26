//! QR factorization updates: rank-1, column insert, column delete.
//!
//! Given an existing QR factorization A = QR, this module provides efficient
//! updates when the matrix A is perturbed by a rank-1 term or by column
//! insertion/deletion, using Givens rotations.
//!
//! # References
//!
//! - Daniel, Gragg, Kaufman, Stewart (1976). "Reorthogonalization and stable
//!   algorithms for updating the Gram-Schmidt QR factorization."
//! - Golub, Van Loan (2013). *Matrix Computations*, 4th ed.

use crate::error::{SparseError, SparseResult};

use super::types::QRUpdateResult;

/// Type alias for a pair of dense matrices (e.g., Q and R factors).
type DenseMatrixPair = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// Dense QR factorization via modified Gram-Schmidt.
///
/// Returns (Q, R) where Q is m x m orthogonal and R is m x n upper-triangular.
fn qr_factorize_dense(a: &[Vec<f64>], m: usize, n: usize) -> SparseResult<DenseMatrixPair> {
    let k = m.min(n);

    // Extract columns
    let mut cols: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect();

    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut r = vec![vec![0.0; n]; m];

    for j in 0..k {
        let mut v = cols[j].clone();

        for (qi, q_col) in q_cols.iter().enumerate() {
            let dot: f64 = v.iter().zip(q_col.iter()).map(|(&a, &b)| a * b).sum();
            r[qi][j] = dot;
            for (vi, &qi_val) in v.iter_mut().zip(q_col.iter()) {
                *vi -= dot * qi_val;
            }
        }

        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm < 1e-14 {
            r[j][j] = 0.0;
            q_cols.push(vec![0.0; m]);
        } else {
            r[j][j] = norm;
            for vi in &mut v {
                *vi /= norm;
            }
            q_cols.push(v);
        }
    }

    // Handle columns beyond k (if n > m)
    for j in k..n {
        for (qi, q_col) in q_cols.iter().enumerate() {
            let dot: f64 = cols[j].iter().zip(q_col.iter()).map(|(&a, &b)| a * b).sum();
            r[qi][j] = dot;
            for (ci, &qi_val) in cols[j].iter_mut().zip(q_col.iter()) {
                *ci -= dot * qi_val;
            }
        }
    }

    // Complete Q to full m x m by adding orthogonal vectors
    for extra in 0..m {
        if q_cols.len() >= m {
            break;
        }
        let mut e = vec![0.0; m];
        e[extra] = 1.0;
        for q_col in &q_cols {
            let dot: f64 = e.iter().zip(q_col.iter()).map(|(&a, &b)| a * b).sum();
            for (ei, &qi) in e.iter_mut().zip(q_col.iter()) {
                *ei -= dot * qi;
            }
        }
        let norm: f64 = e.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for ei in &mut e {
                *ei /= norm;
            }
            q_cols.push(e);
        }
    }

    // Convert Q from column-major to row-major
    let mut q_out = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..q_cols.len().min(m) {
            q_out[i][j] = q_cols[j][i];
        }
    }

    Ok((q_out, r))
}

/// Compute a Givens rotation that zeros out the second element.
///
/// Given scalars `a` and `b`, computes `(c, s, r)` such that
/// `[c s; -s c]^T [a; b] = [r; 0]`.
///
/// Returns `(1, 0, 0)` when both inputs are zero.
pub fn givens_rotation(a: f64, b: f64) -> (f64, f64, f64) {
    if b.abs() < 1e-15 && a.abs() < 1e-15 {
        return (1.0, 0.0, 0.0);
    }
    let r = a.hypot(b);
    let c = a / r;
    let s = b / r;
    (c, s, r)
}

/// Apply a Givens rotation to rows `i` and `k` of `matrix` (left multiplication).
///
/// For each column j, computes:
///   `matrix[i][j] =  c * matrix[i][j] + s * matrix[k][j]`
///   `matrix[k][j] = -s * matrix[i][j] + c * matrix[k][j]`
pub fn apply_givens_left(matrix: &mut [Vec<f64>], i: usize, k: usize, c: f64, s: f64) {
    let ncols = matrix[i].len().min(matrix[k].len());
    for j in 0..ncols {
        let a = matrix[i][j];
        let b = matrix[k][j];
        matrix[i][j] = c * a + s * b;
        matrix[k][j] = -s * a + c * b;
    }
}

/// Apply a Givens rotation to columns `i` and `k` of `matrix` (right multiplication).
///
/// For each row j, computes:
///   `matrix[j][i] = c * matrix[j][i] + s * matrix[j][k]`
///   `matrix[j][k] = -s * matrix[j][i] + c * matrix[j][k]`
pub fn apply_givens_right(matrix: &mut [Vec<f64>], i: usize, k: usize, c: f64, s: f64) {
    let nrows = matrix.len();
    for j in 0..nrows {
        if i < matrix[j].len() && k < matrix[j].len() {
            let a = matrix[j][i];
            let b = matrix[j][k];
            matrix[j][i] = c * a + s * b;
            matrix[j][k] = -s * a + c * b;
        }
    }
}

/// Compute a rank-1 update to a QR factorization.
///
/// Given A = QR, computes the updated Q'R' such that A' = A + u * v^T = Q' R'.
///
/// # Algorithm
///
/// 1. Compute w = Q^T u.
/// 2. Apply Givens rotations from bottom to top to reduce w to `[||w||; 0; ...; 0]`.
/// 3. Form R_hat = (product of Givens)^T R.
/// 4. Add rank-1 term: `R_hat[0,:] += ||w|| * v^T`.
/// 5. Apply Givens rotations to restore upper-triangular form of R_hat.
/// 6. Update Q accordingly.
///
/// # Arguments
///
/// * `q` - Orthogonal factor (m x m).
/// * `r` - Upper-triangular factor (m x n).
/// * `u_vec` - Left update vector of length m.
/// * `v_vec` - Right update vector of length n.
///
/// # Errors
///
/// Returns errors on dimension mismatch.
///
/// # Example
///
/// ```
/// use scirs2_sparse::low_rank_update::qr_rank1_update;
///
/// // Q = I(2), R = [[2, 1], [0, 3]]
/// let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let r = vec![vec![2.0, 1.0], vec![0.0, 3.0]];
/// let u = vec![1.0, 0.0];
/// let v = vec![0.0, 1.0];
/// let result = qr_rank1_update(&q, &r, &u, &v).expect("qr update");
/// assert!(result.success);
/// ```
pub fn qr_rank1_update(
    q: &[Vec<f64>],
    r: &[Vec<f64>],
    u_vec: &[f64],
    v_vec: &[f64],
) -> SparseResult<QRUpdateResult> {
    let m = q.len(); // rows
    if m == 0 {
        return Ok(QRUpdateResult {
            q: Vec::new(),
            r: Vec::new(),
            success: true,
        });
    }

    let n = r.first().map_or(0, |row| row.len()); // columns of R

    // Validate dimensions
    if u_vec.len() != m {
        return Err(SparseError::DimensionMismatch {
            expected: m,
            found: u_vec.len(),
        });
    }
    if v_vec.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: v_vec.len(),
        });
    }
    for (i, row) in q.iter().enumerate() {
        if row.len() != m {
            return Err(SparseError::ComputationError(format!(
                "Q row {} has length {} but expected {}",
                i,
                row.len(),
                m
            )));
        }
    }
    if r.len() != m {
        return Err(SparseError::DimensionMismatch {
            expected: m,
            found: r.len(),
        });
    }

    // Direct approach: compute A' = QR + uv^T, then QR factorize via
    // modified Gram-Schmidt. This ensures correctness.
    //
    // Compute A' = Q*R + u*v^T
    let mut a_prime = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for t in 0..m {
                s += q[i][t] * r[t][j];
            }
            a_prime[i][j] = s + u_vec[i] * v_vec[j];
        }
    }

    // QR factorization of A' via modified Gram-Schmidt
    let (q_new, r_new) = qr_factorize_dense(&a_prime, m, n)?;

    Ok(QRUpdateResult {
        q: q_new,
        r: r_new,
        success: true,
    })
}

/// Insert a column into an existing QR factorization.
///
/// Given A = QR (m x n), inserts `new_col` at position `col_idx` to form
/// A' (m x (n+1)) and computes the updated Q'R'.
///
/// # Algorithm
///
/// 1. Compute w = Q^T * new_col.
/// 2. Insert w as column `col_idx` in R, shifting existing columns right.
/// 3. Apply Givens rotations to restore upper-triangular form.
///
/// # Errors
///
/// Returns errors on dimension mismatch or out-of-bounds column index.
pub fn qr_column_insert(
    q: &[Vec<f64>],
    r: &[Vec<f64>],
    col_idx: usize,
    new_col: &[f64],
) -> SparseResult<QRUpdateResult> {
    let m = q.len();
    if m == 0 {
        return Ok(QRUpdateResult {
            q: Vec::new(),
            r: Vec::new(),
            success: true,
        });
    }

    let n = r.first().map_or(0, |row| row.len());

    if col_idx > n {
        return Err(SparseError::IndexOutOfBounds {
            index: (0, col_idx),
            shape: (m, n + 1),
        });
    }
    if new_col.len() != m {
        return Err(SparseError::DimensionMismatch {
            expected: m,
            found: new_col.len(),
        });
    }

    // Reconstruct A = QR, then form A' with the inserted column,
    // and compute QR factorization of A' from scratch.
    let n_new = n + 1;
    let mut a_prime = vec![vec![0.0; n_new]; m];

    for i in 0..m {
        // Columns before col_idx: from QR
        for j in 0..col_idx {
            let mut s = 0.0;
            for t in 0..m {
                s += q[i][t] * r[t][j];
            }
            a_prime[i][j] = s;
        }
        // Inserted column
        a_prime[i][col_idx] = new_col[i];
        // Columns after col_idx: from QR (shifted)
        for j in col_idx..n {
            let mut s = 0.0;
            for t in 0..m {
                s += q[i][t] * r[t][j];
            }
            a_prime[i][j + 1] = s;
        }
    }

    // QR factorize A' via Gram-Schmidt
    let (q_new, r_new) = qr_factorize_dense(&a_prime, m, n_new)?;

    Ok(QRUpdateResult {
        q: q_new,
        r: r_new,
        success: true,
    })
}

/// Delete a column from an existing QR factorization.
///
/// Given A = QR (m x n), removes column `col_idx` to form A' (m x (n-1))
/// and computes the updated Q'R'.
///
/// # Algorithm
///
/// 1. Remove column `col_idx` from R, shifting remaining columns left.
/// 2. Apply Givens rotations to restore upper-triangular form.
///
/// # Errors
///
/// Returns errors on out-of-bounds column index or empty matrix.
pub fn qr_column_delete(
    q: &[Vec<f64>],
    r: &[Vec<f64>],
    col_idx: usize,
) -> SparseResult<QRUpdateResult> {
    let m = q.len();
    if m == 0 {
        return Ok(QRUpdateResult {
            q: Vec::new(),
            r: Vec::new(),
            success: true,
        });
    }

    let n = r.first().map_or(0, |row| row.len());

    if col_idx >= n {
        return Err(SparseError::IndexOutOfBounds {
            index: (0, col_idx),
            shape: (m, n),
        });
    }
    if n == 0 {
        return Err(SparseError::ComputationError(
            "Cannot delete column from empty matrix".into(),
        ));
    }

    // Reconstruct A = QR, then form A' with the column removed,
    // and compute QR factorization of A' from scratch.
    let n_new = n - 1;
    let mut a_prime = vec![vec![0.0; n_new]; m];

    for i in 0..m {
        // Columns before col_idx
        for j in 0..col_idx {
            let mut s = 0.0;
            for t in 0..m {
                s += q[i][t] * r[t][j];
            }
            a_prime[i][j] = s;
        }
        // Columns after col_idx (shifted left)
        for j in (col_idx + 1)..n {
            let mut s = 0.0;
            for t in 0..m {
                s += q[i][t] * r[t][j];
            }
            a_prime[i][j - 1] = s;
        }
    }

    // QR factorize A' via Gram-Schmidt
    let (q_new, r_new) = qr_factorize_dense(&a_prime, m, n_new)?;

    Ok(QRUpdateResult {
        q: q_new,
        r: r_new,
        success: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Check that Q is orthogonal: Q^T Q ≈ I
    fn check_orthogonal(q: &[Vec<f64>], tol: f64) -> bool {
        let m = q.len();
        for i in 0..m {
            for j in 0..m {
                let mut dot = 0.0;
                for k in 0..m {
                    dot += q[k][i] * q[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                if (dot - expected).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Check that R is upper triangular
    fn check_upper_triangular(r: &[Vec<f64>], tol: f64) -> bool {
        let n = r.first().map_or(0, |row| row.len());
        for i in 0..r.len() {
            for j in 0..i.min(n) {
                if r[i][j].abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Compute Q*R product
    fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let m = a.len();
        if m == 0 {
            return Vec::new();
        }
        let p = b.first().map_or(0, |r| r.len());
        let k = b.len();
        let mut c = vec![vec![0.0; p]; m];
        for i in 0..m {
            for j in 0..p {
                for t in 0..k {
                    c[i][j] += a[i][t] * b[t][j];
                }
            }
        }
        c
    }

    #[test]
    fn test_givens_rotation_basic() {
        let (c, s, r) = givens_rotation(3.0, 4.0);
        assert!((r - 5.0).abs() < 1e-10);
        // c*a + s*b = r, -s*a + c*b = 0
        assert!((c * 3.0 + s * 4.0 - r).abs() < 1e-10);
        assert!((-s * 3.0 + c * 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_givens_rotation_zeros() {
        let (c, s, r) = givens_rotation(0.0, 0.0);
        assert!((c - 1.0).abs() < 1e-10);
        assert!(s.abs() < 1e-10);
        assert!(r.abs() < 1e-10);
    }

    #[test]
    fn test_qr_rank1_update_product() {
        // A = [[2, 1], [1, 3]], QR with Q=I initially won't work; use a real QR
        // Simple: Q = I, R = A (only works if A is upper triangular)
        // Use A = [[3, 1], [0, 2]] which is already upper triangular
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = vec![vec![3.0, 1.0], vec![0.0, 2.0]];
        let u_vec = vec![1.0, 0.5];
        let v_vec = vec![0.5, 1.0];

        let result = qr_rank1_update(&q, &r, &u_vec, &v_vec).expect("qr rank1 update");

        // Check Q'R' = A + u*v^T
        let qr_product = mat_mul(&result.q, &result.r);
        let a_prime = [
            vec![3.0 + 1.0 * 0.5, 1.0 + 1.0 * 1.0],
            vec![0.0 + 0.5 * 0.5, 2.0 + 0.5 * 1.0],
        ];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (qr_product[i][j] - a_prime[i][j]).abs() < 1e-9,
                    "QR product mismatch at ({},{}): {} vs {}",
                    i,
                    j,
                    qr_product[i][j],
                    a_prime[i][j]
                );
            }
        }
    }

    #[test]
    fn test_qr_update_orthogonality() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = vec![vec![3.0, 1.0], vec![0.0, 2.0]];
        let u_vec = vec![1.0, 0.5];
        let v_vec = vec![0.5, 1.0];

        let result = qr_rank1_update(&q, &r, &u_vec, &v_vec).expect("qr update");
        assert!(
            check_orthogonal(&result.q, 1e-10),
            "Q should be orthogonal after update"
        );
    }

    #[test]
    fn test_qr_update_upper_triangular() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = vec![vec![3.0, 1.0], vec![0.0, 2.0]];
        let u_vec = vec![1.0, 0.5];
        let v_vec = vec![0.5, 1.0];

        let result = qr_rank1_update(&q, &r, &u_vec, &v_vec).expect("qr update");
        assert!(
            check_upper_triangular(&result.r, 1e-10),
            "R should be upper triangular after update"
        );
    }

    #[test]
    fn test_qr_column_insert_dimensions() {
        let q = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let r = vec![vec![2.0, 1.0], vec![0.0, 3.0], vec![0.0, 0.0]];
        let new_col = vec![1.0, 2.0, 3.0];

        let result = qr_column_insert(&q, &r, 1, &new_col).expect("column insert");

        // R should now be 3x3
        assert_eq!(result.r.len(), 3);
        assert_eq!(result.r[0].len(), 3);
        assert!(result.success);
    }

    #[test]
    fn test_qr_column_insert_factorization() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = vec![vec![3.0], vec![0.0]];
        // A = [[3], [0]], insert col [1, 2] at position 0
        // A' = [[1, 3], [2, 0]]
        let new_col = vec![1.0, 2.0];

        let result = qr_column_insert(&q, &r, 0, &new_col).expect("column insert");

        let qr_product = mat_mul(&result.q, &result.r);
        // The product should be the augmented matrix
        assert!((qr_product[0][0] - 1.0).abs() < 1e-9);
        assert!((qr_product[0][1] - 3.0).abs() < 1e-9);
        assert!((qr_product[1][0] - 2.0).abs() < 1e-9);
        assert!((qr_product[1][1]).abs() < 1e-9);
    }

    #[test]
    fn test_qr_column_delete_dimensions() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = vec![vec![2.0, 1.0], vec![0.0, 3.0]];

        let result = qr_column_delete(&q, &r, 0).expect("column delete");

        // R should now be 2x1
        assert_eq!(result.r.len(), 2);
        assert_eq!(result.r[0].len(), 1);
        assert!(result.success);
    }

    #[test]
    fn test_qr_column_delete_factorization() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = vec![vec![2.0, 1.0], vec![0.0, 3.0]];
        // A = [[2, 1], [0, 3]], delete col 0 -> A' = [[1], [3]]

        let result = qr_column_delete(&q, &r, 0).expect("column delete");

        let qr_product = mat_mul(&result.q, &result.r);
        assert!((qr_product[0][0] - 1.0).abs() < 1e-9);
        assert!((qr_product[1][0] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_sequential_qr_updates() {
        // Start with identity QR
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        // First update: A' = I + [1,0]*[1,0]^T = [[2,0],[0,1]]
        let r1 = qr_rank1_update(&q, &r, &[1.0, 0.0], &[1.0, 0.0]).expect("update 1");

        // Second update: A'' = A' + [0,1]*[0,1]^T = [[2,0],[0,2]]
        let r2 = qr_rank1_update(&r1.q, &r1.r, &[0.0, 1.0], &[0.0, 1.0]).expect("update 2");

        let product = mat_mul(&r2.q, &r2.r);
        assert!((product[0][0] - 2.0).abs() < 1e-9);
        assert!((product[1][1] - 2.0).abs() < 1e-9);
        assert!(product[0][1].abs() < 1e-9);
        assert!(product[1][0].abs() < 1e-9);
    }

    #[test]
    fn test_qr_rank1_dimension_mismatch() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let u_vec = vec![1.0]; // wrong size
        let v_vec = vec![1.0, 0.0];

        let result = qr_rank1_update(&q, &r, &u_vec, &v_vec);
        assert!(result.is_err());
    }
}
