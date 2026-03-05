//! Matrix multiplication CPU fallback implementations.
//!
//! Provides GEMM (General Matrix Multiplication) helper functions
//! used by the [`super::GpuContext`] CPU fallback operations.

use super::traits::NumericOps;

/// Reinterpret a Vec<T> as Vec<U> when we know T and U have the same size.
/// This is only safe when T and U are the same type (verified by TypeId check).
pub(super) unsafe fn reinterpret_vec<T, U>(v: Vec<T>) -> Vec<U> {
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    debug_assert_eq!(std::mem::align_of::<T>(), std::mem::align_of::<U>());
    let mut v = std::mem::ManuallyDrop::new(v);
    Vec::from_raw_parts(v.as_mut_ptr() as *mut U, v.len(), v.capacity())
}

// =============================================================================
// GEMM (General Matrix Multiplication)
// =============================================================================

/// CPU fallback for GEMM: C = A * B
/// A is m x k, B is k x n, C is m x n (row-major layout)
pub(super) fn gemm_cpu<T: NumericOps>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    let mut c = vec![T::zero(); m * n];

    // Standard triple-loop matrix multiplication with row-major layout
    // Using i-k-j order for better cache locality on row-major data
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                let b_val = b[p * n + j];
                c[i * n + j] = c[i * n + j].add(a_val.mul(b_val));
            }
        }
    }

    c
}

/// CPU fallback for GEMM with transposed B: C = A * B^T
/// A is m x k, B is n x k (stored row-major, but used as k x n transposed), C is m x n
pub(super) fn gemm_transpose_b_cpu<T: NumericOps>(
    a: &[T],
    b: &[T],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<T> {
    let mut c = vec![T::zero(); m * n];

    // B^T[p][j] = B[j][p], B is stored as n x k
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for p in 0..k {
                sum = sum.add(a[i * k + p].mul(b[j * k + p]));
            }
            c[i * n + j] = sum;
        }
    }

    c
}

/// CPU fallback for GEMM with transposed A: C = A^T * B
/// A is k x m (stored row-major, used as m x k transposed), B is k x n, C is m x n
pub(super) fn gemm_transpose_a_cpu<T: NumericOps>(
    a: &[T],
    b: &[T],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<T> {
    let mut c = vec![T::zero(); m * n];

    // A^T[i][p] = A[p][i], A is stored as k x m
    for i in 0..m {
        for p in 0..k {
            let a_val = a[p * m + i]; // A^T[i][p] = A[p][i]
            for j in 0..n {
                c[i * n + j] = c[i * n + j].add(a_val.mul(b[p * n + j]));
            }
        }
    }

    c
}
