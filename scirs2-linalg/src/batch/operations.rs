//! Batch BLAS-like matrix operations
//!
//! Provides batched versions of common linear algebra operations for processing
//! multiple matrices simultaneously, optimized for throughput in ML workloads.

use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView3, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

/// Perform pairwise matrix multiplication on two batches of matrices.
///
/// Given batches A and B each of shape (batch_size, m, k) and (batch_size, k, n),
/// computes `C[i] = A[i] * B[i]` for each batch element.
///
/// # Arguments
///
/// * `batch_a` - 3D array of shape (batch_size, m, k)
/// * `batch_b` - 3D array of shape (batch_size, k, n)
///
/// # Returns
///
/// * 3D array of shape (batch_size, m, n)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array3;
/// use scirs2_linalg::batch::batch_matmul_pairwise;
///
/// let a = Array3::from_shape_vec((2, 2, 2), vec![
///     1.0, 2.0, 3.0, 4.0,
///     5.0, 6.0, 7.0, 8.0,
/// ]).expect("shape");
/// let b = Array3::from_shape_vec((2, 2, 2), vec![
///     1.0, 0.0, 0.0, 1.0,
///     2.0, 0.0, 0.0, 2.0,
/// ]).expect("shape");
///
/// let c = batch_matmul_pairwise(&a.view(), &b.view()).expect("ok");
/// assert_eq!(c.shape(), &[2, 2, 2]);
/// ```
pub fn batch_matmul_pairwise<F>(
    batch_a: &ArrayView3<F>,
    batch_b: &ArrayView3<F>,
) -> LinalgResult<Array3<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (bs_a, m, k1) = batch_a.dim();
    let (bs_b, k2, n) = batch_b.dim();

    if bs_a != bs_b {
        return Err(LinalgError::ShapeError(format!(
            "Batch sizes mismatch: {} vs {}",
            bs_a, bs_b
        )));
    }
    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Inner dimensions mismatch: {} vs {}",
            k1, k2
        )));
    }

    let batch_size = bs_a;
    let mut result = Array3::zeros((batch_size, m, n));

    // Threshold for BLAS acceleration
    let work_size = m * k1 * n;
    if work_size >= 64 {
        for b in 0..batch_size {
            let a_slice = batch_a.slice(scirs2_core::ndarray::s![b, .., ..]);
            let b_slice = batch_b.slice(scirs2_core::ndarray::s![b, .., ..]);
            let c = crate::blas_accelerated::matmul(&a_slice, &b_slice)?;
            result
                .slice_mut(scirs2_core::ndarray::s![b, .., ..])
                .assign(&c);
        }
    } else {
        for b in 0..batch_size {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = F::zero();
                    for kk in 0..k1 {
                        sum += batch_a[[b, i, kk]] * batch_b[[b, kk, j]];
                    }
                    result[[b, i, j]] = sum;
                }
            }
        }
    }

    Ok(result)
}

/// Solve a batch of linear systems `A[i] x[i] = b[i]`.
///
/// For each batch element, solves the linear system using LU decomposition
/// with partial pivoting.
///
/// # Arguments
///
/// * `batch_a` - 3D array of shape (batch_size, n, n), batch of square coefficient matrices
/// * `batch_b` - 2D array of shape (batch_size, n), batch of right-hand side vectors
///
/// # Returns
///
/// * 2D array of shape (batch_size, n), the solution vectors
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{Array3, Array2};
/// use scirs2_linalg::batch::batch_solve;
///
/// // Identity matrices => x = b
/// let a = Array3::<f64>::from_shape_vec((2, 2, 2), vec![
///     1.0, 0.0, 0.0, 1.0,
///     1.0, 0.0, 0.0, 1.0,
/// ]).expect("shape");
/// let b = Array2::<f64>::from_shape_vec((2, 2), vec![
///     3.0, 4.0,
///     5.0, 6.0,
/// ]).expect("shape");
///
/// let x = batch_solve(&a.view(), &b.view()).expect("ok");
/// assert!((x[[0, 0]] - 3.0_f64).abs() < 1e-10);
/// ```
pub fn batch_solve<F>(
    batch_a: &ArrayView3<F>,
    batch_b: &scirs2_core::ndarray::ArrayView2<F>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (bs_a, rows, cols) = batch_a.dim();
    let (bs_b, b_len) = batch_b.dim();

    if bs_a != bs_b {
        return Err(LinalgError::ShapeError(format!(
            "Batch sizes mismatch: {} vs {}",
            bs_a, bs_b
        )));
    }
    if rows != cols {
        return Err(LinalgError::ShapeError(format!(
            "Matrices must be square, got {}x{}",
            rows, cols
        )));
    }
    if rows != b_len {
        return Err(LinalgError::ShapeError(format!(
            "Matrix size {} does not match vector length {}",
            rows, b_len
        )));
    }

    let batch_size = bs_a;
    let n = rows;
    let mut result = Array2::zeros((batch_size, n));

    for b in 0..batch_size {
        let a_slice = batch_a.slice(scirs2_core::ndarray::s![b, .., ..]);
        let b_vec = batch_b.slice(scirs2_core::ndarray::s![b, ..]);
        let x = crate::solve(&a_slice, &b_vec, None)?;
        for j in 0..n {
            result[[b, j]] = x[j];
        }
    }

    Ok(result)
}

/// Solve a batch of linear systems `A[i] X[i] = B[i]` with matrix right-hand sides.
///
/// For each batch element, solves the linear system where B is a matrix,
/// yielding a matrix solution X.
///
/// # Arguments
///
/// * `batch_a` - 3D array of shape (batch_size, n, n)
/// * `batch_b` - 3D array of shape (batch_size, n, m)
///
/// # Returns
///
/// * 3D array of shape (batch_size, n, m)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array3;
/// use scirs2_linalg::batch::batch_solve_matrix;
///
/// let a = Array3::<f64>::from_shape_vec((1, 2, 2), vec![
///     1.0, 0.0, 0.0, 1.0,
/// ]).expect("shape");
/// let b = Array3::<f64>::from_shape_vec((1, 2, 2), vec![
///     3.0, 4.0, 5.0, 6.0,
/// ]).expect("shape");
///
/// let x = batch_solve_matrix(&a.view(), &b.view()).expect("ok");
/// assert!((x[[0, 0, 0]] - 3.0_f64).abs() < 1e-10);
/// ```
pub fn batch_solve_matrix<F>(
    batch_a: &ArrayView3<F>,
    batch_b: &ArrayView3<F>,
) -> LinalgResult<Array3<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (bs_a, rows_a, cols_a) = batch_a.dim();
    let (bs_b, rows_b, cols_b) = batch_b.dim();

    if bs_a != bs_b {
        return Err(LinalgError::ShapeError(format!(
            "Batch sizes mismatch: {} vs {}",
            bs_a, bs_b
        )));
    }
    if rows_a != cols_a {
        return Err(LinalgError::ShapeError(format!(
            "Matrices must be square, got {}x{}",
            rows_a, cols_a
        )));
    }
    if rows_a != rows_b {
        return Err(LinalgError::ShapeError(format!(
            "Matrix rows {} do not match RHS rows {}",
            rows_a, rows_b
        )));
    }

    let batch_size = bs_a;
    let n = rows_a;
    let nrhs = cols_b;
    let mut result = Array3::zeros((batch_size, n, nrhs));

    for b in 0..batch_size {
        let a_slice = batch_a.slice(scirs2_core::ndarray::s![b, .., ..]);
        let b_slice = batch_b.slice(scirs2_core::ndarray::s![b, .., ..]);
        let x = crate::solve_multiple(&a_slice, &b_slice, None)?;
        result
            .slice_mut(scirs2_core::ndarray::s![b, .., ..])
            .assign(&x);
    }

    Ok(result)
}

/// Compute the inverse of each matrix in a batch.
///
/// # Arguments
///
/// * `batch_a` - 3D array of shape (batch_size, n, n)
///
/// # Returns
///
/// * 3D array of shape (batch_size, n, n), the inverses
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array3;
/// use scirs2_linalg::batch::batch_inverse;
///
/// let a = Array3::<f64>::from_shape_vec((2, 2, 2), vec![
///     1.0, 0.0, 0.0, 1.0,
///     2.0, 0.0, 0.0, 0.5,
/// ]).expect("shape");
///
/// let inv = batch_inverse(&a.view()).expect("ok");
/// assert!((inv[[0, 0, 0]] - 1.0_f64).abs() < 1e-10);
/// assert!((inv[[1, 0, 0]] - 0.5_f64).abs() < 1e-10);
/// ```
pub fn batch_inverse<F>(batch_a: &ArrayView3<F>) -> LinalgResult<Array3<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (batch_size, rows, cols) = batch_a.dim();

    if rows != cols {
        return Err(LinalgError::ShapeError(format!(
            "Matrices must be square, got {}x{}",
            rows, cols
        )));
    }

    let n = rows;
    let mut result = Array3::zeros((batch_size, n, n));

    for b in 0..batch_size {
        let a_slice = batch_a.slice(scirs2_core::ndarray::s![b, .., ..]);
        let inv = crate::inv(&a_slice, None)?;
        result
            .slice_mut(scirs2_core::ndarray::s![b, .., ..])
            .assign(&inv);
    }

    Ok(result)
}

/// Compute the determinant of each matrix in a batch.
///
/// # Arguments
///
/// * `batch_a` - 3D array of shape (batch_size, n, n)
///
/// # Returns
///
/// * 1D array of shape (batch_size,), the determinants
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array3;
/// use scirs2_linalg::batch::batch_det;
///
/// let a = Array3::<f64>::from_shape_vec((2, 2, 2), vec![
///     1.0, 0.0, 0.0, 1.0,  // det = 1
///     2.0, 3.0, 1.0, 4.0,  // det = 2*4 - 3*1 = 5
/// ]).expect("shape");
///
/// let dets = batch_det(&a.view()).expect("ok");
/// assert!((dets[0] - 1.0_f64).abs() < 1e-10);
/// assert!((dets[1] - 5.0_f64).abs() < 1e-10);
/// ```
pub fn batch_det<F>(batch_a: &ArrayView3<F>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (batch_size, rows, cols) = batch_a.dim();

    if rows != cols {
        return Err(LinalgError::ShapeError(format!(
            "Matrices must be square, got {}x{}",
            rows, cols
        )));
    }

    let mut result = Array1::zeros(batch_size);

    for b in 0..batch_size {
        let a_slice = batch_a.slice(scirs2_core::ndarray::s![b, .., ..]);
        let d = crate::det(&a_slice, None)?;
        result[b] = d;
    }

    Ok(result)
}

/// Compute the trace of each matrix in a batch.
///
/// # Arguments
///
/// * `batch_a` - 3D array of shape (batch_size, n, n)
///
/// # Returns
///
/// * 1D array of shape (batch_size,), the traces
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array3;
/// use scirs2_linalg::batch::batch_trace;
///
/// let a = Array3::from_shape_vec((2, 2, 2), vec![
///     1.0, 2.0, 3.0, 4.0,  // trace = 1+4 = 5
///     5.0, 6.0, 7.0, 8.0,  // trace = 5+8 = 13
/// ]).expect("shape");
///
/// let traces = batch_trace(&a.view()).expect("ok");
/// assert!((traces[0] - 5.0).abs() < 1e-10);
/// assert!((traces[1] - 13.0).abs() < 1e-10);
/// ```
pub fn batch_trace<F>(batch_a: &ArrayView3<F>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (batch_size, rows, cols) = batch_a.dim();

    if rows != cols {
        return Err(LinalgError::ShapeError(format!(
            "Matrices must be square, got {}x{}",
            rows, cols
        )));
    }

    let n = rows;
    let mut result = Array1::zeros(batch_size);

    for b in 0..batch_size {
        let mut tr = F::zero();
        for i in 0..n {
            tr += batch_a[[b, i, i]];
        }
        result[b] = tr;
    }

    Ok(result)
}

/// Compute the Frobenius norm of each matrix in a batch.
///
/// # Arguments
///
/// * `batch_a` - 3D array of shape (batch_size, m, n)
///
/// # Returns
///
/// * 1D array of shape (batch_size,), the Frobenius norms
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array3;
/// use scirs2_linalg::batch::batch_frobenius_norm;
///
/// let a = Array3::from_shape_vec((1, 2, 2), vec![
///     3.0, 4.0, 0.0, 0.0,
/// ]).expect("shape");
///
/// let norms = batch_frobenius_norm(&a.view()).expect("ok");
/// assert!((norms[0] - 5.0).abs() < 1e-10);
/// ```
pub fn batch_frobenius_norm<F>(batch_a: &ArrayView3<F>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (batch_size, m, n) = batch_a.dim();
    let mut result = Array1::zeros(batch_size);

    for b in 0..batch_size {
        let mut sum_sq = F::zero();
        for i in 0..m {
            for j in 0..n {
                let val = batch_a[[b, i, j]];
                sum_sq += val * val;
            }
        }
        result[b] = sum_sq.sqrt();
    }

    Ok(result)
}

/// Compute the transpose of each matrix in a batch.
///
/// # Arguments
///
/// * `batch_a` - 3D array of shape (batch_size, m, n)
///
/// # Returns
///
/// * 3D array of shape (batch_size, n, m)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array3;
/// use scirs2_linalg::batch::batch_transpose;
///
/// let a = Array3::from_shape_vec((1, 2, 3), vec![
///     1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
/// ]).expect("shape");
///
/// let at = batch_transpose(&a.view());
/// assert_eq!(at.shape(), &[1, 3, 2]);
/// assert_eq!(at[[0, 0, 0]], 1.0);
/// assert_eq!(at[[0, 0, 1]], 4.0);
/// ```
pub fn batch_transpose<F>(batch_a: &ArrayView3<F>) -> Array3<F>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (batch_size, m, n) = batch_a.dim();
    let mut result = Array3::zeros((batch_size, n, m));

    for b in 0..batch_size {
        for i in 0..m {
            for j in 0..n {
                result[[b, j, i]] = batch_a[[b, i, j]];
            }
        }
    }

    result
}

/// Compute Cholesky factorization of each matrix in a batch.
///
/// Each matrix must be symmetric positive-definite. Returns lower-triangular
/// factors L such that `A[i] = L[i] * L[i]^T`.
///
/// # Arguments
///
/// * `batch_a` - 3D array of shape (batch_size, n, n)
///
/// # Returns
///
/// * 3D array of shape (batch_size, n, n), the lower-triangular Cholesky factors
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array3;
/// use scirs2_linalg::batch::batch_cholesky;
///
/// let a = Array3::from_shape_vec((1, 2, 2), vec![
///     4.0, 2.0, 2.0, 3.0,
/// ]).expect("shape");
///
/// let l = batch_cholesky(&a.view()).expect("ok");
/// assert!((l[[0, 0, 0]] - 2.0).abs() < 1e-10);
/// ```
pub fn batch_cholesky<F>(batch_a: &ArrayView3<F>) -> LinalgResult<Array3<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (batch_size, rows, cols) = batch_a.dim();

    if rows != cols {
        return Err(LinalgError::ShapeError(format!(
            "Matrices must be square, got {}x{}",
            rows, cols
        )));
    }

    let n = rows;
    let mut result = Array3::zeros((batch_size, n, n));

    for b in 0..batch_size {
        let a_slice = batch_a.slice(scirs2_core::ndarray::s![b, .., ..]);
        let l = crate::cholesky(&a_slice, None)?;
        result
            .slice_mut(scirs2_core::ndarray::s![b, .., ..])
            .assign(&l);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::{Array2, Array3};

    // --- batch_matmul_pairwise tests ---

    #[test]
    fn test_batch_matmul_pairwise_identity() {
        let a = Array3::from_shape_vec((2, 2, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("shape");
        let eye = Array3::from_shape_vec((2, 2, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
            .expect("shape");
        let c = batch_matmul_pairwise(&a.view(), &eye.view()).expect("ok");
        for b in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    assert_relative_eq!(c[[b, i, j]], a[[b, i, j]], epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_batch_matmul_pairwise_known() {
        // A[0] = [[1,2],[3,4]], B[0] = [[5,6],[7,8]]
        // C[0] = [[19,22],[43,50]]
        let a = Array3::from_shape_vec((1, 2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("shape");
        let b = Array3::from_shape_vec((1, 2, 2), vec![5.0, 6.0, 7.0, 8.0]).expect("shape");
        let c = batch_matmul_pairwise(&a.view(), &b.view()).expect("ok");
        assert_relative_eq!(c[[0, 0, 0]], 19.0, epsilon = 1e-12);
        assert_relative_eq!(c[[0, 0, 1]], 22.0, epsilon = 1e-12);
        assert_relative_eq!(c[[0, 1, 0]], 43.0, epsilon = 1e-12);
        assert_relative_eq!(c[[0, 1, 1]], 50.0, epsilon = 1e-12);
    }

    #[test]
    fn test_batch_matmul_pairwise_rectangular() {
        // (2, 2, 3) * (2, 3, 1) => (2, 2, 1)
        let a = Array3::from_shape_vec(
            (2, 2, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("shape");
        let b =
            Array3::from_shape_vec((2, 3, 1), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).expect("shape");
        let c = batch_matmul_pairwise(&a.view(), &b.view()).expect("ok");
        assert_eq!(c.shape(), &[2, 2, 1]);
        // First batch: [1+2+3, 4+5+6] = [6, 15]
        assert_relative_eq!(c[[0, 0, 0]], 6.0, epsilon = 1e-12);
        assert_relative_eq!(c[[0, 1, 0]], 15.0, epsilon = 1e-12);
        // Second batch: [7*2+8*2+9*2, 10*2+11*2+12*2] = [48, 66]
        assert_relative_eq!(c[[1, 0, 0]], 48.0, epsilon = 1e-12);
        assert_relative_eq!(c[[1, 1, 0]], 66.0, epsilon = 1e-12);
    }

    #[test]
    fn test_batch_matmul_pairwise_batch_size_mismatch() {
        let a = Array3::<f64>::zeros((2, 2, 2));
        let b = Array3::<f64>::zeros((3, 2, 2));
        assert!(batch_matmul_pairwise(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn test_batch_matmul_pairwise_dim_mismatch() {
        let a = Array3::<f64>::zeros((2, 2, 3));
        let b = Array3::<f64>::zeros((2, 4, 2));
        assert!(batch_matmul_pairwise(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn test_batch_matmul_pairwise_large() {
        // Use larger matrices to trigger BLAS path
        let n = 16;
        let a = Array3::from_shape_fn(
            (2, n, n),
            |(b, i, j)| {
                if i == j {
                    (b + 1) as f64
                } else {
                    0.0
                }
            },
        );
        let b = Array3::from_shape_fn((2, n, n), |(_, i, j)| if i == j { 1.0 } else { 0.0 });
        let c = batch_matmul_pairwise(&a.view(), &b.view()).expect("ok");
        for i in 0..n {
            assert_relative_eq!(c[[0, i, i]], 1.0, epsilon = 1e-12);
            assert_relative_eq!(c[[1, i, i]], 2.0, epsilon = 1e-12);
        }
    }

    // --- batch_solve tests ---

    #[test]
    fn test_batch_solve_identity() {
        let a = Array3::from_shape_vec((2, 2, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
            .expect("shape");
        let b = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 5.0, 6.0]).expect("shape");
        let x = batch_solve(&a.view(), &b.view()).expect("ok");
        assert_relative_eq!(x[[0, 0]], 3.0, epsilon = 1e-10);
        assert_relative_eq!(x[[0, 1]], 4.0, epsilon = 1e-10);
        assert_relative_eq!(x[[1, 0]], 5.0, epsilon = 1e-10);
        assert_relative_eq!(x[[1, 1]], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_solve_known_system() {
        // A = [[2, 1], [1, 3]], b = [5, 7] => x = [8/5, 9/5]
        let a = Array3::from_shape_vec((1, 2, 2), vec![2.0, 1.0, 1.0, 3.0]).expect("shape");
        let b = Array2::from_shape_vec((1, 2), vec![5.0, 7.0]).expect("shape");
        let x = batch_solve(&a.view(), &b.view()).expect("ok");
        assert_relative_eq!(x[[0, 0]], 8.0 / 5.0, epsilon = 1e-10);
        assert_relative_eq!(x[[0, 1]], 9.0 / 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_solve_nonsquare_error() {
        let a = Array3::<f64>::zeros((1, 2, 3));
        let b = Array2::<f64>::zeros((1, 2));
        assert!(batch_solve(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn test_batch_solve_batch_mismatch() {
        let a = Array3::<f64>::zeros((2, 2, 2));
        let b = Array2::<f64>::zeros((3, 2));
        assert!(batch_solve(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn test_batch_solve_dim_mismatch() {
        let a = Array3::<f64>::zeros((1, 3, 3));
        let b = Array2::<f64>::zeros((1, 2));
        assert!(batch_solve(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn test_batch_solve_3x3() {
        // A = [[1,0,0],[0,2,0],[0,0,3]], b = [1,4,9] => x = [1,2,3]
        let a =
            Array3::from_shape_vec((1, 3, 3), vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])
                .expect("shape");
        let b = Array2::from_shape_vec((1, 3), vec![1.0, 4.0, 9.0]).expect("shape");
        let x = batch_solve(&a.view(), &b.view()).expect("ok");
        assert_relative_eq!(x[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[[0, 1]], 2.0, epsilon = 1e-10);
        assert_relative_eq!(x[[0, 2]], 3.0, epsilon = 1e-10);
    }

    // --- batch_solve_matrix tests ---

    #[test]
    fn test_batch_solve_matrix_identity() {
        let a = Array3::from_shape_vec((1, 2, 2), vec![1.0, 0.0, 0.0, 1.0]).expect("shape");
        let b = Array3::from_shape_vec((1, 2, 2), vec![3.0, 4.0, 5.0, 6.0]).expect("shape");
        let x = batch_solve_matrix(&a.view(), &b.view()).expect("ok");
        assert_relative_eq!(x[[0, 0, 0]], 3.0, epsilon = 1e-10);
        assert_relative_eq!(x[[0, 0, 1]], 4.0, epsilon = 1e-10);
        assert_relative_eq!(x[[0, 1, 0]], 5.0, epsilon = 1e-10);
        assert_relative_eq!(x[[0, 1, 1]], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_solve_matrix_nonsquare_error() {
        let a = Array3::<f64>::zeros((1, 2, 3));
        let b = Array3::<f64>::zeros((1, 2, 2));
        assert!(batch_solve_matrix(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn test_batch_solve_matrix_batch_mismatch() {
        let a = Array3::<f64>::zeros((2, 2, 2));
        let b = Array3::<f64>::zeros((3, 2, 2));
        assert!(batch_solve_matrix(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn test_batch_solve_matrix_rows_mismatch() {
        let a = Array3::<f64>::zeros((1, 3, 3));
        let b = Array3::<f64>::zeros((1, 2, 2));
        assert!(batch_solve_matrix(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn test_batch_solve_matrix_known() {
        // A = [[2, 0], [0, 3]], B = [[4, 6], [9, 12]] => X = [[2, 3], [3, 4]]
        let a = Array3::from_shape_vec((1, 2, 2), vec![2.0, 0.0, 0.0, 3.0]).expect("shape");
        let b = Array3::from_shape_vec((1, 2, 2), vec![4.0, 6.0, 9.0, 12.0]).expect("shape");
        let x = batch_solve_matrix(&a.view(), &b.view()).expect("ok");
        assert_relative_eq!(x[[0, 0, 0]], 2.0, epsilon = 1e-10);
        assert_relative_eq!(x[[0, 0, 1]], 3.0, epsilon = 1e-10);
        assert_relative_eq!(x[[0, 1, 0]], 3.0, epsilon = 1e-10);
        assert_relative_eq!(x[[0, 1, 1]], 4.0, epsilon = 1e-10);
    }

    // --- batch_inverse tests ---

    #[test]
    fn test_batch_inverse_identity() {
        let a = Array3::from_shape_vec((2, 2, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
            .expect("shape");
        let inv = batch_inverse(&a.view()).expect("ok");
        for b in 0..2 {
            assert_relative_eq!(inv[[b, 0, 0]], 1.0, epsilon = 1e-10);
            assert_relative_eq!(inv[[b, 0, 1]], 0.0, epsilon = 1e-10);
            assert_relative_eq!(inv[[b, 1, 0]], 0.0, epsilon = 1e-10);
            assert_relative_eq!(inv[[b, 1, 1]], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_batch_inverse_diagonal() {
        let a = Array3::from_shape_vec((1, 2, 2), vec![2.0, 0.0, 0.0, 4.0]).expect("shape");
        let inv = batch_inverse(&a.view()).expect("ok");
        assert_relative_eq!(inv[[0, 0, 0]], 0.5, epsilon = 1e-10);
        assert_relative_eq!(inv[[0, 1, 1]], 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_inverse_roundtrip() {
        let a = Array3::from_shape_vec((1, 2, 2), vec![4.0, 7.0, 2.0, 6.0]).expect("shape");
        let inv = batch_inverse(&a.view()).expect("ok");
        let c = batch_matmul_pairwise(&a.view(), &inv.view()).expect("ok");
        // Should be approximately identity
        assert_relative_eq!(c[[0, 0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(c[[0, 0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(c[[0, 1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(c[[0, 1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_inverse_nonsquare() {
        let a = Array3::<f64>::zeros((1, 2, 3));
        assert!(batch_inverse(&a.view()).is_err());
    }

    #[test]
    fn test_batch_inverse_multiple() {
        // Two different matrices
        let a = Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, // det = -2
                5.0, 6.0, 7.0, 8.5, // det = 5*8.5 - 6*7 = 42.5-42 = 0.5
            ],
        )
        .expect("shape");
        let inv = batch_inverse(&a.view()).expect("ok");
        let c = batch_matmul_pairwise(&a.view(), &inv.view()).expect("ok");
        for b in 0..2 {
            assert_relative_eq!(c[[b, 0, 0]], 1.0, epsilon = 1e-8);
            assert_relative_eq!(c[[b, 0, 1]], 0.0, epsilon = 1e-8);
            assert_relative_eq!(c[[b, 1, 0]], 0.0, epsilon = 1e-8);
            assert_relative_eq!(c[[b, 1, 1]], 1.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_batch_inverse_3x3() {
        let a =
            Array3::from_shape_vec((1, 3, 3), vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0])
                .expect("shape");
        let inv = batch_inverse(&a.view()).expect("ok");
        assert_relative_eq!(inv[[0, 0, 0]], 0.5, epsilon = 1e-10);
        assert_relative_eq!(inv[[0, 1, 1]], 1.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(inv[[0, 2, 2]], 0.2, epsilon = 1e-10);
    }

    // --- batch_det tests ---

    #[test]
    fn test_batch_det_identity() {
        let a = Array3::from_shape_vec((2, 2, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
            .expect("shape");
        let dets = batch_det(&a.view()).expect("ok");
        assert_relative_eq!(dets[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(dets[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_det_known() {
        let a = Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                2.0, 3.0, 1.0, 4.0, // det = 2*4 - 3*1 = 5
                1.0, 2.0, 3.0, 4.0, // det = 1*4 - 2*3 = -2
            ],
        )
        .expect("shape");
        let dets = batch_det(&a.view()).expect("ok");
        assert_relative_eq!(dets[0], 5.0, epsilon = 1e-10);
        assert_relative_eq!(dets[1], -2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_det_nonsquare() {
        let a = Array3::<f64>::zeros((1, 2, 3));
        assert!(batch_det(&a.view()).is_err());
    }

    #[test]
    fn test_batch_det_3x3() {
        // [[1,0,0],[0,2,0],[0,0,3]] => det = 6
        let a =
            Array3::from_shape_vec((1, 3, 3), vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])
                .expect("shape");
        let dets = batch_det(&a.view()).expect("ok");
        assert_relative_eq!(dets[0], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_det_scalar_property() {
        // det(cA) = c^n * det(A) for n x n matrix
        let a = Array3::from_shape_vec((1, 2, 2), vec![2.0, 3.0, 1.0, 4.0]).expect("shape");
        let det_a = batch_det(&a.view()).expect("ok")[0];

        let scaled: Array3<f64> = a.mapv(|x| x * 2.0);
        let det_2a = batch_det(&scaled.view()).expect("ok")[0];
        // det(2A) = 2^2 * det(A) = 4 * det(A)
        assert_relative_eq!(det_2a, 4.0 * det_a, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_det_singular() {
        // [[1, 2], [2, 4]] has det = 0
        let a = Array3::from_shape_vec((1, 2, 2), vec![1.0, 2.0, 2.0, 4.0]).expect("shape");
        let dets = batch_det(&a.view()).expect("ok");
        assert!(dets[0].abs() < 1e-10);
    }

    // --- batch_trace tests ---

    #[test]
    fn test_batch_trace_identity() {
        let a =
            Array3::from_shape_vec((1, 3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
                .expect("shape");
        let tr = batch_trace(&a.view()).expect("ok");
        assert_relative_eq!(tr[0], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_batch_trace_known() {
        let a = Array3::from_shape_vec((2, 2, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("shape");
        let tr = batch_trace(&a.view()).expect("ok");
        assert_relative_eq!(tr[0], 5.0, epsilon = 1e-12);
        assert_relative_eq!(tr[1], 13.0, epsilon = 1e-12);
    }

    #[test]
    fn test_batch_trace_nonsquare() {
        let a = Array3::<f64>::zeros((1, 2, 3));
        assert!(batch_trace(&a.view()).is_err());
    }

    #[test]
    fn test_batch_trace_linearity() {
        let a = Array3::from_shape_vec((1, 2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("shape");
        let b = Array3::from_shape_vec((1, 2, 2), vec![5.0, 6.0, 7.0, 8.0]).expect("shape");
        let tr_a = batch_trace(&a.view()).expect("ok")[0];
        let tr_b = batch_trace(&b.view()).expect("ok")[0];

        // tr(A + B) = tr(A) + tr(B)
        let sum: Array3<f64> =
            Array3::from_shape_fn((1, 2, 2), |(bi, i, j)| a[[bi, i, j]] + b[[bi, i, j]]);
        let tr_sum = batch_trace(&sum.view()).expect("ok")[0];
        assert_relative_eq!(tr_sum, tr_a + tr_b, epsilon = 1e-12);
    }

    #[test]
    fn test_batch_trace_zero_matrix() {
        let a = Array3::<f64>::zeros((2, 3, 3));
        let tr = batch_trace(&a.view()).expect("ok");
        assert_relative_eq!(tr[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(tr[1], 0.0, epsilon = 1e-12);
    }

    // --- batch_frobenius_norm tests ---

    #[test]
    fn test_batch_frobenius_norm_known() {
        let a = Array3::from_shape_vec((1, 2, 2), vec![3.0, 4.0, 0.0, 0.0]).expect("shape");
        let norms = batch_frobenius_norm(&a.view()).expect("ok");
        assert_relative_eq!(norms[0], 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_batch_frobenius_norm_identity() {
        let a =
            Array3::from_shape_vec((1, 3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
                .expect("shape");
        let norms = batch_frobenius_norm(&a.view()).expect("ok");
        assert_relative_eq!(norms[0], 3.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn test_batch_frobenius_norm_zero_matrix() {
        let a = Array3::<f64>::zeros((2, 3, 3));
        let norms = batch_frobenius_norm(&a.view()).expect("ok");
        assert_relative_eq!(norms[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(norms[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_batch_frobenius_norm_multiple() {
        let a = Array3::from_shape_vec((2, 1, 2), vec![3.0, 4.0, 5.0, 12.0]).expect("shape");
        let norms = batch_frobenius_norm(&a.view()).expect("ok");
        assert_relative_eq!(norms[0], 5.0, epsilon = 1e-12);
        assert_relative_eq!(norms[1], 13.0, epsilon = 1e-12);
    }

    #[test]
    fn test_batch_frobenius_norm_scaling() {
        let a = Array3::from_shape_vec((1, 2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("shape");
        let norm_a = batch_frobenius_norm(&a.view()).expect("ok")[0];

        let scaled: Array3<f64> = a.mapv(|x| x * 3.0);
        let norm_3a = batch_frobenius_norm(&scaled.view()).expect("ok")[0];
        assert_relative_eq!(norm_3a, 3.0 * norm_a, epsilon = 1e-10);
    }

    // --- batch_transpose tests ---

    #[test]
    fn test_batch_transpose_identity() {
        let a = Array3::from_shape_vec((1, 2, 2), vec![1.0, 0.0, 0.0, 1.0]).expect("shape");
        let at = batch_transpose(&a.view());
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(at[[0, i, j]], a[[0, i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_batch_transpose_rectangular() {
        let a =
            Array3::from_shape_vec((1, 2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("shape");
        let at = batch_transpose(&a.view());
        assert_eq!(at.shape(), &[1, 3, 2]);
        assert_eq!(at[[0, 0, 0]], 1.0);
        assert_eq!(at[[0, 1, 0]], 2.0);
        assert_eq!(at[[0, 2, 0]], 3.0);
        assert_eq!(at[[0, 0, 1]], 4.0);
        assert_eq!(at[[0, 1, 1]], 5.0);
        assert_eq!(at[[0, 2, 1]], 6.0);
    }

    #[test]
    fn test_batch_transpose_double() {
        let a = Array3::from_shape_vec(
            (2, 2, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("shape");
        let att = batch_transpose(&batch_transpose(&a.view()).view());
        assert_eq!(att.shape(), a.shape());
        for b in 0..2 {
            for i in 0..2 {
                for j in 0..3 {
                    assert_relative_eq!(att[[b, i, j]], a[[b, i, j]], epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_batch_transpose_symmetric() {
        let a = Array3::from_shape_vec((1, 2, 2), vec![1.0, 2.0, 2.0, 3.0]).expect("shape");
        let at = batch_transpose(&a.view());
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(at[[0, i, j]], a[[0, i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_batch_transpose_multiple() {
        let a =
            Array3::from_shape_vec((3, 1, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("shape");
        let at = batch_transpose(&a.view());
        assert_eq!(at.shape(), &[3, 2, 1]);
        assert_eq!(at[[0, 0, 0]], 1.0);
        assert_eq!(at[[0, 1, 0]], 2.0);
        assert_eq!(at[[1, 0, 0]], 3.0);
        assert_eq!(at[[1, 1, 0]], 4.0);
        assert_eq!(at[[2, 0, 0]], 5.0);
        assert_eq!(at[[2, 1, 0]], 6.0);
    }

    // --- batch_cholesky tests ---

    #[test]
    fn test_batch_cholesky_identity() {
        let a = Array3::from_shape_vec((1, 2, 2), vec![1.0, 0.0, 0.0, 1.0]).expect("shape");
        let l = batch_cholesky(&a.view()).expect("ok");
        assert_relative_eq!(l[[0, 0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(l[[0, 1, 1]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(l[[0, 0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(l[[0, 1, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_cholesky_spd() {
        // [[4, 2], [2, 3]] => L = [[2, 0], [1, sqrt(2)]]
        let a = Array3::from_shape_vec((1, 2, 2), vec![4.0, 2.0, 2.0, 3.0]).expect("shape");
        let l = batch_cholesky(&a.view()).expect("ok");
        assert_relative_eq!(l[[0, 0, 0]], 2.0, epsilon = 1e-10);
        assert_relative_eq!(l[[0, 1, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(l[[0, 1, 1]], 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_batch_cholesky_nonsquare() {
        let a = Array3::<f64>::zeros((1, 2, 3));
        assert!(batch_cholesky(&a.view()).is_err());
    }

    #[test]
    fn test_batch_cholesky_multiple() {
        let a = Array3::from_shape_vec((2, 2, 2), vec![1.0, 0.0, 0.0, 1.0, 9.0, 0.0, 0.0, 4.0])
            .expect("shape");
        let l = batch_cholesky(&a.view()).expect("ok");
        assert_relative_eq!(l[[0, 0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(l[[1, 0, 0]], 3.0, epsilon = 1e-10);
        assert_relative_eq!(l[[1, 1, 1]], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_cholesky_roundtrip() {
        // L * L^T should give back A
        let a = Array3::from_shape_vec((1, 2, 2), vec![4.0, 2.0, 2.0, 3.0]).expect("shape");
        let l = batch_cholesky(&a.view()).expect("ok");
        let lt = batch_transpose(&l.view());
        let reconstructed = batch_matmul_pairwise(&l.view(), &lt.view()).expect("ok");
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(reconstructed[[0, i, j]], a[[0, i, j]], epsilon = 1e-10);
            }
        }
    }
}
