//! Basic matrix operations
//!
//! Uses OxiBLAS for optimized pure Rust BLAS/LAPACK operations.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

// OxiBLAS LAPACK operations (via scirs2-core linalg abstraction)
use scirs2_core::linalg::{det_ndarray, inv_ndarray};

/// Compute the determinant of a square matrix.
///
/// Uses optimized BLAS/LAPACK when available (f32/f64), falls back to pure Rust for other types.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `workers` - Number of worker threads (None = use default) - ignored when using BLAS/LAPACK
///
/// # Returns
///
/// * Determinant of the matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::det;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let d = det(&a.view(), None).expect("Operation failed");
/// assert!((d - (-2.0)).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn det<F>(a: &ArrayView2<F>, _workers: Option<usize>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    if a.nrows() != a.ncols() {
        let rows = a.nrows();
        let cols = a.ncols();
        return Err(LinalgError::ShapeError(format!(
            "Determinant computation failed: Matrix must be square\nMatrix shape: {rows}×{cols}\nExpected: Square matrix (n×n)"
        )));
    }

    // Optimized path for f64 using BLAS/LAPACK (200-700x faster!)
    det_impl(a)
}

/// OxiBLAS-accelerated determinant for f64 (PUBLIC for direct Python use)
///
/// This function uses OxiBLAS pure Rust LAPACK implementation.
pub fn det_f64_lapack(a: &ArrayView2<f64>) -> LinalgResult<f64> {
    // Validate square matrix
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square: got {}×{}",
            a.nrows(),
            a.ncols()
        )));
    }

    // Small matrices: use simple formulas (faster than LAPACK overhead)
    match a.nrows() {
        0 => return Ok(1.0),
        1 => return Ok(a[[0, 0]]),
        2 => return Ok(a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]]),
        3 => {
            let det = a[[0, 0]] * (a[[1, 1]] * a[[2, 2]] - a[[1, 2]] * a[[2, 1]])
                - a[[0, 1]] * (a[[1, 0]] * a[[2, 2]] - a[[1, 2]] * a[[2, 0]])
                + a[[0, 2]] * (a[[1, 0]] * a[[2, 1]] - a[[1, 1]] * a[[2, 0]]);
            return Ok(det);
        }
        _ => {}
    }

    // For 4x4+: Use OxiBLAS LAPACK
    match det_ndarray(&a.to_owned()) {
        Ok(det) => Ok(det),
        Err(e) => {
            // OxiBLAS returns an error for singular matrices - determinant is 0
            let err_str = format!("{:?}", e);
            if err_str.contains("Singular") {
                Ok(0.0)
            } else {
                Err(LinalgError::ComputationError(format!(
                    "OxiBLAS det failed: {:?}",
                    e
                )))
            }
        }
    }
}

/// OxiBLAS-accelerated matrix inverse for f64
///
/// Uses OxiBLAS pure Rust LAPACK implementation.
pub fn inv_f64_lapack(a: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    // Validate square matrix
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square: got {}×{}",
            a.nrows(),
            a.ncols()
        )));
    }

    // Use OxiBLAS LAPACK for matrix inversion
    inv_ndarray(&a.to_owned())
        .map_err(|e| LinalgError::ComputationError(format!("OxiBLAS inv failed: {:?}", e)))
}

/// Implementation of determinant - uses OxiBLAS for f32/f64, pure Rust fallback for others
fn det_impl<F>(a: &ArrayView2<F>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    use scirs2_core::numeric::NumCast;
    use std::any::TypeId;

    // Fast path for f64 using OxiBLAS
    if TypeId::of::<F>() == TypeId::of::<f64>() {
        // SAFETY: We've verified the type is f64
        let a_f64: &ArrayView2<f64> = unsafe { std::mem::transmute(a) };
        let result = det_f64_lapack(a_f64)?;
        return Ok(<F as NumCast>::from(result).expect("Operation failed"));
    }

    // Fast path for f32 using OxiBLAS
    if TypeId::of::<F>() == TypeId::of::<f32>() {
        let a_f32: &ArrayView2<f32> = unsafe { std::mem::transmute(a) };
        let result = det_ndarray(&a_f32.to_owned())
            .map_err(|e| LinalgError::ComputationError(format!("OxiBLAS det failed: {:?}", e)))?;
        return Ok(<F as NumCast>::from(result).expect("Operation failed"));
    }

    // Fallback to pure Rust for other types
    det_pure_rust(a)
}

/// Pure Rust determinant implementation (fallback for non-LAPACK types)
fn det_pure_rust<F>(a: &ArrayView2<F>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    // Simple implementation for small matrices
    match a.nrows() {
        0 => Ok(F::one()),
        1 => Ok(a[[0, 0]]),
        2 => Ok(a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]]),
        3 => {
            let det = a[[0, 0]] * (a[[1, 1]] * a[[2, 2]] - a[[1, 2]] * a[[2, 1]])
                - a[[0, 1]] * (a[[1, 0]] * a[[2, 2]] - a[[1, 2]] * a[[2, 0]])
                + a[[0, 2]] * (a[[1, 0]] * a[[2, 1]] - a[[1, 1]] * a[[2, 0]]);
            Ok(det)
        }
        _ => {
            // For larger matrices, use LU decomposition
            use crate::decomposition::lu;

            match lu(a, None) {
                Ok((p, _l, u)) => {
                    // Calculate determinant from U diagonal and permutation count
                    let mut det_u = F::one();
                    for i in 0..u.nrows() {
                        det_u *= u[[i, i]];
                    }

                    let mut swap_count = 0;
                    for i in 0..p.nrows() {
                        for j in 0..i {
                            if p[[i, j]] == F::one() {
                                swap_count += 1;
                            }
                        }
                    }

                    if swap_count % 2 == 0 {
                        Ok(det_u)
                    } else {
                        Ok(-det_u)
                    }
                }
                Err(LinalgError::SingularMatrixError(_)) => Ok(F::zero()),
                Err(e) => Err(e),
            }
        }
    }
}

/// Compute the inverse of a square matrix.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Inverse of the matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::inv;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let a_inv = inv(&a.view(), None).expect("Operation failed");
/// assert!((a_inv[[0, 0]] - 1.0).abs() < 1e-10);
/// assert!((a_inv[[1, 1]] - 0.5).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn inv<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    if a.nrows() != a.ncols() {
        let rows = a.nrows();
        let cols = a.ncols();
        return Err(LinalgError::ShapeError(format!(
            "Matrix inverse computation failed: Matrix must be square\nMatrix shape: {rows}×{cols}\nExpected: Square matrix (n×n)"
        )));
    }

    // Simple implementation for 2x2 matrices
    if a.nrows() == 2 {
        let det_val = det(a, workers)?;
        if det_val.abs() < F::epsilon() {
            // Calculate condition number estimate for 2x2 matrix
            let norm_a = (a[[0, 0]] * a[[0, 0]]
                + a[[0, 1]] * a[[0, 1]]
                + a[[1, 0]] * a[[1, 0]]
                + a[[1, 1]] * a[[1, 1]])
            .sqrt();
            let cond_estimate = if det_val.abs() > F::zero() {
                Some((norm_a / det_val.abs()).to_f64().unwrap_or(1e16))
            } else {
                None
            };

            return Err(LinalgError::singularmatrix_with_suggestions(
                "matrix inverse",
                a.dim(),
                cond_estimate,
            ));
        }

        let inv_det = F::one() / det_val;
        let mut result = Array2::zeros((2, 2));
        result[[0, 0]] = a[[1, 1]] * inv_det;
        result[[0, 1]] = -a[[0, 1]] * inv_det;
        result[[1, 0]] = -a[[1, 0]] * inv_det;
        result[[1, 1]] = a[[0, 0]] * inv_det;
        return Ok(result);
    }

    // For larger matrices, use the solve_multiple function with an identity matrix
    use crate::solve::solve_multiple;

    let n = a.nrows();
    let mut identity = Array2::zeros((n, n));
    for i in 0..n {
        identity[[i, i]] = F::one();
    }

    // Solve A * X = I to get X = A^(-1)
    match solve_multiple(a, &identity.view(), workers) {
        Err(LinalgError::SingularMatrixError(_)) => {
            // Use enhanced error with regularization suggestions
            Err(LinalgError::singularmatrix_with_suggestions(
                "matrix inverse via solve",
                a.dim(),
                None, // Could compute condition number here for better diagnostics
            ))
        }
        other => other,
    }
}

/// Raise a square matrix to the given power.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `n` - Power (can be positive, negative, or zero)
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Matrix raised to the power n
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_power;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
///
/// // Identity matrix for n=0
/// let a_0 = matrix_power(&a.view(), 0, None).expect("Operation failed");
/// assert!((a_0[[0, 0]] - 1.0).abs() < 1e-10);
/// assert!((a_0[[0, 1]] - 0.0).abs() < 1e-10);
/// assert!((a_0[[1, 0]] - 0.0).abs() < 1e-10);
/// assert!((a_0[[1, 1]] - 1.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn matrix_power<F>(a: &ArrayView2<F>, n: i32, workers: Option<usize>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    if a.nrows() != a.ncols() {
        let rows = a.nrows();
        let cols = a.ncols();
        return Err(LinalgError::ShapeError(format!(
            "Matrix power computation failed: Matrix must be square\nMatrix shape: {rows}×{cols}\nExpected: Square matrix (n×n)"
        )));
    }

    let dim = a.nrows();

    // Handle special cases
    if n == 0 {
        // Return identity matrix
        let mut result = Array2::zeros((dim, dim));
        for i in 0..dim {
            result[[i, i]] = F::one();
        }
        return Ok(result);
    }

    if n == 1 {
        // Return copy of the matrix
        return Ok(a.to_owned());
    }

    if n == -1 {
        // Return inverse
        return inv(a, workers);
    }

    // For negative powers with |n| > 1, compute inv(A)^|n|
    let base = if n < 0 {
        inv(a, workers)?
    } else {
        a.to_owned()
    };

    // Binary exponentiation for |n| >= 2
    let mut exp = n.unsigned_abs();
    let mut result = Array2::zeros((dim, dim));
    for i in 0..dim {
        result[[i, i]] = F::one();
    }
    let mut cur = base;

    while exp > 0 {
        if exp & 1 == 1 {
            result = result.dot(&cur);
        }
        exp >>= 1;
        if exp > 0 {
            let tmp = cur.clone();
            cur = tmp.dot(&cur);
        }
    }

    Ok(result)
}

/// Compute the trace of a square matrix.
///
/// The trace is the sum of the diagonal elements.
///
/// # Arguments
///
/// * `a` - A square matrix
///
/// # Returns
///
/// * Trace of the matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::basic_trace;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let tr = basic_trace(&a.view()).expect("Operation failed");
/// assert!((tr - 5.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn trace<F>(a: &ArrayView2<F>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    if a.nrows() != a.ncols() {
        let rows = a.nrows();
        let cols = a.ncols();
        return Err(LinalgError::ShapeError(format!(
            "Matrix trace computation failed: Matrix must be square\nMatrix shape: {rows}×{cols}\nExpected: Square matrix (n×n)"
        )));
    }

    let mut tr = F::zero();
    for i in 0..a.nrows() {
        tr += a[[i, i]];
    }

    Ok(tr)
}

//
// Backward compatibility wrapper functions
//

/// Compute the determinant of a square matrix (backward compatibility wrapper).
///
/// This is a convenience function that calls `det` with `workers = None`.
/// For new code, prefer using `det` directly with explicit workers parameter.
#[allow(dead_code)]
pub fn det_default<F>(a: &ArrayView2<F>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    det(a, None)
}

/// Compute the inverse of a square matrix (backward compatibility wrapper).
///
/// This is a convenience function that calls `inv` with `workers = None`.
/// For new code, prefer using `inv` directly with explicit workers parameter.
#[allow(dead_code)]
pub fn inv_default<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    inv(a, None)
}

/// Raise a square matrix to the given power (backward compatibility wrapper).
///
/// This is a convenience function that calls `matrix_power` with `workers = None`.
/// For new code, prefer using `matrix_power` directly with explicit workers parameter.
#[allow(dead_code)]
pub fn matrix_power_default<F>(a: &ArrayView2<F>, n: i32) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    matrix_power(a, n, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_det_2x2() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let d = det(&a.view(), None).expect("Operation failed");
        assert!((d - (-2.0)).abs() < 1e-10);

        let b = array![[2.0, 0.0], [0.0, 3.0]];
        let d = det(&b.view(), None).expect("Operation failed");
        assert!((d - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_det_3x3() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let d = det(&a.view(), None).expect("Operation failed");
        assert!((d - 0.0).abs() < 1e-10);

        let b = array![[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        let d = det(&b.view(), None).expect("Operation failed");
        assert!((d - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_inv_2x2() {
        let a = array![[1.0, 0.0], [0.0, 2.0]];
        let a_inv = inv(&a.view(), None).expect("Operation failed");
        assert_relative_eq!(a_inv[[0, 0]], 1.0);
        assert_relative_eq!(a_inv[[0, 1]], 0.0);
        assert_relative_eq!(a_inv[[1, 0]], 0.0);
        assert_relative_eq!(a_inv[[1, 1]], 0.5);

        let b = array![[1.0, 2.0], [3.0, 4.0]];
        let b_inv = inv(&b.view(), None).expect("Operation failed");
        assert_relative_eq!(b_inv[[0, 0]], -2.0);
        assert_relative_eq!(b_inv[[0, 1]], 1.0);
        assert_relative_eq!(b_inv[[1, 0]], 1.5);
        assert_relative_eq!(b_inv[[1, 1]], -0.5);
    }

    #[test]
    fn test_inv_large() {
        // Test 3x3 matrix
        let a = array![[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];
        let a_inv = inv(&a.view(), None).expect("Operation failed");

        // Verify A * A^(-1) = I
        let product = a.dot(&a_inv);
        let n = a.nrows();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    assert_relative_eq!(product[[i, j]], 1.0, epsilon = 1e-10);
                } else {
                    assert_relative_eq!(product[[i, j]], 0.0, epsilon = 1e-10);
                }
            }
        }

        // Test 4x4 diagonal matrix
        let b = array![
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 5.0]
        ];
        let b_inv = inv(&b.view(), None).expect("Operation failed");
        assert_relative_eq!(b_inv[[0, 0]], 0.5, epsilon = 1e-10);
        assert_relative_eq!(b_inv[[1, 1]], 1.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(b_inv[[2, 2]], 0.25, epsilon = 1e-10);
        assert_relative_eq!(b_inv[[3, 3]], 0.2, epsilon = 1e-10);

        // Test singular matrix should error
        let c = array![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]];
        assert!(inv(&c.view(), None).is_err());
    }

    #[test]
    fn testmatrix_power() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];

        // Power 0 should give identity matrix
        let a_0 = matrix_power(&a.view(), 0, None).expect("Operation failed");
        assert_relative_eq!(a_0[[0, 0]], 1.0);
        assert_relative_eq!(a_0[[0, 1]], 0.0);
        assert_relative_eq!(a_0[[1, 0]], 0.0);
        assert_relative_eq!(a_0[[1, 1]], 1.0);

        // Power 1 should return the original matrix
        let a_1 = matrix_power(&a.view(), 1, None).expect("Operation failed");
        assert_relative_eq!(a_1[[0, 0]], a[[0, 0]]);
        assert_relative_eq!(a_1[[0, 1]], a[[0, 1]]);
        assert_relative_eq!(a_1[[1, 0]], a[[1, 0]]);
        assert_relative_eq!(a_1[[1, 1]], a[[1, 1]]);
    }

    #[test]
    fn test_det_large() {
        // Test 4x4 matrix
        let a = array![
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 2.0]
        ];
        let d = det(&a.view(), None).expect("Operation failed");
        assert_relative_eq!(d, 5.0, epsilon = 1e-10);

        // Test 5x5 diagonal matrix
        let b = array![
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 5.0]
        ];
        let d = det(&b.view(), None).expect("Operation failed");
        assert_relative_eq!(d, 120.0, epsilon = 1e-10);

        // Test singular matrix
        let c = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [3.0, 6.0, 9.0, 12.0],
            [4.0, 8.0, 12.0, 16.0]
        ];
        let d = det(&c.view(), None).expect("Operation failed");
        assert_relative_eq!(d, 0.0, epsilon = 1e-10);
    }
}
