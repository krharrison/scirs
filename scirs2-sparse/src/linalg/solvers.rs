//! Direct solvers and basic operations for sparse matrices

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::iter::Sum;

/// Solve a sparse linear system Ax = b using Gaussian elimination with
/// partial pivoting on the dense representation.
///
/// For small-to-medium systems this is reliable. Large systems should
/// prefer iterative solvers (CG, GMRES, BiCGSTAB) from the
/// `iterative` sub-module.
#[allow(dead_code)]
pub fn spsolve<F>(a: &CsrMatrix<F>, b: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static + std::fmt::Debug,
{
    if a.rows() != a.cols() {
        return Err(SparseError::ValueError(format!(
            "Matrix must be square, got {}x{}",
            a.rows(),
            a.cols()
        )));
    }
    if a.rows() != b.len() {
        return Err(SparseError::DimensionMismatch {
            expected: a.rows(),
            found: b.len(),
        });
    }
    let a_dense = a.to_dense();
    gaussian_elimination(&a_dense, b)
}

/// Solve a sparse linear system using direct methods.
///
/// Accepts hints for symmetry and positive-definiteness (currently both
/// paths fall through to Gaussian elimination with partial pivoting).
#[allow(dead_code)]
pub fn sparse_direct_solve<F>(
    a: &CsrMatrix<F>,
    b: &[F],
    _symmetric: bool,
    _positive_definite: bool,
) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static + std::fmt::Debug,
{
    spsolve(a, b)
}

/// Solve a least squares problem
#[allow(dead_code)]
pub fn sparse_lstsq<F>(a: &CsrMatrix<F>, b: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static + std::fmt::Debug,
{
    // For now, solve normal equations: A^T * A * x = A^T * b
    let at = a.transpose();
    let ata = matmul(&at, a)?;
    // Compute A^T * b
    let mut atb = vec![F::sparse_zero(); at.rows()];
    for (row, atb_val) in atb.iter_mut().enumerate().take(at.rows()) {
        let row_range = at.row_range(row);
        let row_indices = &at.indices[row_range.clone()];
        let row_data = &at.data[row_range];

        let mut sum = F::sparse_zero();
        for (col_idx, &col) in row_indices.iter().enumerate() {
            sum += row_data[col_idx] * b[col];
        }
        *atb_val = sum;
    }
    spsolve(&ata, &atb)
}

/// Compute matrix norm using the sparse CSR structure directly.
///
///  - `"1"`   : 1-norm (maximum absolute column sum)
///  - `"inf"` : infinity-norm (maximum absolute row sum)
///  - `"fro"` : Frobenius norm (sqrt of sum of squared entries)
#[allow(dead_code)]
pub fn norm<F>(a: &CsrMatrix<F>, ord: &str) -> SparseResult<F>
where
    F: Float + NumAssign + Sum + SparseElement + 'static + std::fmt::Debug,
{
    match ord {
        "1" => {
            // 1-norm: single pass over non-zeros to accumulate column sums
            let mut col_sums = vec![F::sparse_zero(); a.cols()];
            for i in 0..a.rows() {
                let range = a.row_range(i);
                let row_indices = &a.indices[range.clone()];
                let row_data = &a.data[range];
                for (idx, &col) in row_indices.iter().enumerate() {
                    col_sums[col] += row_data[idx].abs();
                }
            }
            let max_sum = col_sums
                .into_iter()
                .fold(F::sparse_zero(), |mx, v| if v > mx { v } else { mx });
            Ok(max_sum)
        }
        "inf" => {
            // Infinity norm: single pass over rows
            let mut max_sum = F::sparse_zero();
            for i in 0..a.rows() {
                let range = a.row_range(i);
                let row_data = &a.data[range];
                let row_sum: F = row_data.iter().map(|v| v.abs()).sum();
                if row_sum > max_sum {
                    max_sum = row_sum;
                }
            }
            Ok(max_sum)
        }
        "fro" => {
            let sum_squares: F = a.data.iter().map(|v| *v * *v).sum();
            Ok(sum_squares.sqrt())
        }
        _ => Err(SparseError::ValueError(format!("Unknown norm: {ord}"))),
    }
}

/// Sparse matrix multiplication C = A * B using CSR row-by-row accumulation.
///
/// For each row i of A, scatters `a[i,k] * B[k,:]` into a workspace.
/// Operates in O(nnz(A) * avg_nnz_per_row(B)) time.
#[allow(dead_code)]
pub fn matmul<F>(a: &CsrMatrix<F>, b: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static + std::fmt::Debug,
{
    if a.cols() != b.rows() {
        return Err(SparseError::DimensionMismatch {
            expected: a.cols(),
            found: b.rows(),
        });
    }

    let nrows = a.rows();
    let ncols = b.cols();
    let mut result_rows = Vec::new();
    let mut result_cols = Vec::new();
    let mut result_data = Vec::new();

    let mut workspace = vec![F::sparse_zero(); ncols];
    let mut marker = vec![false; ncols];

    for i in 0..nrows {
        let a_range = a.row_range(i);
        let a_indices = &a.indices[a_range.clone()];
        let a_data_row = &a.data[a_range];
        let mut touched_cols: Vec<usize> = Vec::new();

        for (idx, &k) in a_indices.iter().enumerate() {
            let a_ik = a_data_row[idx];
            if a_ik == F::sparse_zero() {
                continue;
            }
            let b_range = b.row_range(k);
            let b_indices = &b.indices[b_range.clone()];
            let b_data_row = &b.data[b_range];
            for (bidx, &j) in b_indices.iter().enumerate() {
                workspace[j] += a_ik * b_data_row[bidx];
                if !marker[j] {
                    marker[j] = true;
                    touched_cols.push(j);
                }
            }
        }

        touched_cols.sort_unstable();
        for &j in &touched_cols {
            let val = workspace[j];
            if val != F::sparse_zero() {
                result_rows.push(i);
                result_cols.push(j);
                result_data.push(val);
            }
            workspace[j] = F::sparse_zero();
            marker[j] = false;
        }
    }

    CsrMatrix::new(result_data, result_rows, result_cols, (nrows, ncols))
}

/// Sparse matrix addition using a merge of the two CSR row structures.
///
/// Runs in O(nnz(A) + nnz(B)) time without converting to dense form.
#[allow(dead_code)]
pub fn add<F>(a: &CsrMatrix<F>, b: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static + std::fmt::Debug,
{
    if a.shape() != b.shape() {
        return Err(SparseError::ShapeMismatch {
            expected: a.shape(),
            found: b.shape(),
        });
    }

    let (nrows, ncols) = a.shape();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for i in 0..nrows {
        let a_range = a.row_range(i);
        let b_range = b.row_range(i);
        let a_cols = &a.indices[a_range.clone()];
        let a_data = &a.data[a_range];
        let b_cols = &b.indices[b_range.clone()];
        let b_data = &b.data[b_range];

        let mut ai = 0usize;
        let mut bi = 0usize;
        while ai < a_cols.len() && bi < b_cols.len() {
            if a_cols[ai] < b_cols[bi] {
                let val = a_data[ai];
                if val != F::sparse_zero() {
                    rows.push(i);
                    cols.push(a_cols[ai]);
                    data.push(val);
                }
                ai += 1;
            } else if a_cols[ai] > b_cols[bi] {
                let val = b_data[bi];
                if val != F::sparse_zero() {
                    rows.push(i);
                    cols.push(b_cols[bi]);
                    data.push(val);
                }
                bi += 1;
            } else {
                let val = a_data[ai] + b_data[bi];
                if val != F::sparse_zero() {
                    rows.push(i);
                    cols.push(a_cols[ai]);
                    data.push(val);
                }
                ai += 1;
                bi += 1;
            }
        }
        while ai < a_cols.len() {
            let val = a_data[ai];
            if val != F::sparse_zero() {
                rows.push(i);
                cols.push(a_cols[ai]);
                data.push(val);
            }
            ai += 1;
        }
        while bi < b_cols.len() {
            let val = b_data[bi];
            if val != F::sparse_zero() {
                rows.push(i);
                cols.push(b_cols[bi]);
                data.push(val);
            }
            bi += 1;
        }
    }

    CsrMatrix::new(data, rows, cols, (nrows, ncols))
}

/// Element-wise multiplication (Hadamard product) using sorted CSR row merge.
///
/// Only produces non-zeros where both A and B have non-zeros in the same
/// position, so runs in O(nnz(A) + nnz(B)) time.
#[allow(dead_code)]
pub fn multiply<F>(a: &CsrMatrix<F>, b: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static + std::fmt::Debug,
{
    if a.shape() != b.shape() {
        return Err(SparseError::ShapeMismatch {
            expected: a.shape(),
            found: b.shape(),
        });
    }

    let (nrows, _ncols) = a.shape();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for i in 0..nrows {
        let a_range = a.row_range(i);
        let b_range = b.row_range(i);
        let a_cols = &a.indices[a_range.clone()];
        let a_data = &a.data[a_range];
        let b_cols = &b.indices[b_range.clone()];
        let b_data = &b.data[b_range];

        let mut ai = 0usize;
        let mut bi = 0usize;
        while ai < a_cols.len() && bi < b_cols.len() {
            if a_cols[ai] < b_cols[bi] {
                ai += 1;
            } else if a_cols[ai] > b_cols[bi] {
                bi += 1;
            } else {
                let val = a_data[ai] * b_data[bi];
                if val != F::sparse_zero() {
                    rows.push(i);
                    cols.push(a_cols[ai]);
                    data.push(val);
                }
                ai += 1;
                bi += 1;
            }
        }
    }

    CsrMatrix::new(data, rows, cols, a.shape())
}

/// Create a diagonal matrix
#[allow(dead_code)]
pub fn diag_matrix<F>(diag: &[F], n: Option<usize>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static + std::fmt::Debug,
{
    let size = n.unwrap_or(diag.len());
    if size < diag.len() {
        return Err(SparseError::ValueError(
            "Size must be at least as large as diagonal".to_string(),
        ));
    }

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for (i, &val) in diag.iter().enumerate() {
        if val != F::sparse_zero() {
            rows.push(i);
            cols.push(i);
            data.push(val);
        }
    }

    CsrMatrix::new(data, rows, cols, (size, size))
}

/// Create an identity matrix
#[allow(dead_code)]
pub fn eye<F>(n: usize) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static + std::fmt::Debug,
{
    let diag = vec![F::sparse_one(); n];
    diag_matrix(&diag, Some(n))
}

/// Matrix inverse
#[allow(dead_code)]
pub fn inv<F>(a: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static + std::fmt::Debug,
{
    if a.rows() != a.cols() {
        return Err(SparseError::ValueError(
            "Matrix must be square for inverse".to_string(),
        ));
    }

    let n = a.rows();

    // Solve A * X = I for X
    let mut inv_cols = Vec::new();

    for j in 0..n {
        // Get column j from identity matrix
        let mut col_vec = vec![F::sparse_zero(); n];
        col_vec[j] = F::sparse_one();
        let x = spsolve(a, &col_vec)?;
        inv_cols.push(x);
    }

    // Construct the inverse matrix from columns
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for (j, col) in inv_cols.iter().enumerate() {
        for (i, &val) in col.iter().enumerate() {
            if val.abs() > F::epsilon() {
                rows.push(i);
                cols.push(j);
                data.push(val);
            }
        }
    }

    CsrMatrix::new(data, rows, cols, (n, n))
}

// Matrix exponential functionality is now available in linalg/expm.rs module

/// Matrix power
#[allow(dead_code)]
pub fn matrix_power<F>(a: &CsrMatrix<F>, power: i32) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static + std::fmt::Debug,
{
    if a.rows() != a.cols() {
        return Err(SparseError::ValueError(
            "Matrix must be square for power".to_string(),
        ));
    }

    match power {
        0 => eye(a.rows()),
        1 => Ok(a.clone()),
        p if p > 0 => {
            let mut result = a.clone();
            for _ in 1..p {
                result = matmul(&result, a)?;
            }
            Ok(result)
        }
        p => {
            // Negative power: compute inverse and then positive power
            let inv_a = inv(a)?;
            matrix_power(&inv_a, -p)
        }
    }
}

// Helper functions

#[allow(dead_code)]
fn gaussian_elimination<F>(a: &[Vec<F>], b: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + SparseElement,
{
    let n = a.len();
    let mut aug = vec![vec![F::sparse_zero(); n + 1]; n];

    // Create augmented matrix
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    // Forward elimination
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        for i in (k + 1)..n {
            if aug[i][k].abs() > aug[max_idx][k].abs() {
                max_idx = i;
            }
        }
        aug.swap(k, max_idx);

        // Check for zero pivot
        if aug[k][k].abs() < F::epsilon() {
            return Err(SparseError::SingularMatrix(
                "Matrix is singular".to_string(),
            ));
        }

        // Eliminate column
        for i in (k + 1)..n {
            let factor = aug[i][k] / aug[k][k];
            for j in k..=n {
                aug[i][j] = aug[i][j] - factor * aug[k][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![F::sparse_zero(); n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] = x[i] - aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Ok(x)
}

// Helper functions for matrix exponential have been moved to linalg/expm.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eye_matrix() {
        let eye_matrix = eye::<f64>(3).expect("Operation failed");
        assert_eq!(eye_matrix.shape(), (3, 3));
        assert_eq!(eye_matrix.get(0, 0), 1.0);
        assert_eq!(eye_matrix.get(1, 1), 1.0);
        assert_eq!(eye_matrix.get(2, 2), 1.0);
        assert_eq!(eye_matrix.get(0, 1), 0.0);
    }

    #[test]
    fn test_diag_matrix() {
        let diag = vec![2.0, 3.0, 4.0];
        let diag_matrix = diag_matrix(&diag, None).expect("Operation failed");
        assert_eq!(diag_matrix.shape(), (3, 3));
        assert_eq!(diag_matrix.get(0, 0), 2.0);
        assert_eq!(diag_matrix.get(1, 1), 3.0);
        assert_eq!(diag_matrix.get(2, 2), 4.0);
    }

    #[test]
    fn test_matrix_power() {
        let diag = vec![2.0, 3.0];
        let matrix = diag_matrix(&diag, None).expect("Operation failed");

        // Test power 2
        let matrix2 = matrix_power(&matrix, 2).expect("Operation failed");
        assert_eq!(matrix2.get(0, 0), 4.0);
        assert_eq!(matrix2.get(1, 1), 9.0);

        // Test power 0 (identity)
        let matrix0 = matrix_power(&matrix, 0).expect("Operation failed");
        assert_eq!(matrix0.get(0, 0), 1.0);
        assert_eq!(matrix0.get(1, 1), 1.0);
    }
}
