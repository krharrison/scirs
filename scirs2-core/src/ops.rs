//! # Ergonomic Matrix Operations
//!
//! This module provides shorthand functions for the most common matrix and
//! vector operations in scientific computing, making it easy to write concise
//! numerical code without importing many different traits.
//!
//! All functions are generic over `f64`/`f32` where practical, or carry their
//! own trait bounds so IDEs can provide accurate type hints.
//!
//! ## Operations
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`dot`] | Matrix–matrix multiply (`C = A · B`) |
//! | [`outer`] | Outer (tensor) product of two 1D arrays |
//! | [`kron`] | Kronecker product of two 2D arrays |
//! | [`vstack`] | Vertical concatenation of 2D arrays (add rows) |
//! | [`hstack`] | Horizontal concatenation of 2D arrays (add columns) |
//! | [`block_diag`] | Build a block-diagonal matrix from a sequence of 2D blocks |
//!
//! ## Design Notes
//!
//! - **No unwrap** — every fallible function returns [`CoreResult`].
//! - **Generic** — all functions are parameterised over numeric type `T`.
//! - The implementations deliberately avoid pulling in full BLAS; they use pure
//!   ndarray loops. For production workloads that require maximum matrix-multiply
//!   throughput, use `scirs2-linalg` which delegates to OxiBLAS.

use crate::error::{CoreError, CoreResult, ErrorContext};
use ::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::Zero;
use std::ops::{Add, Mul};

// ============================================================================
// dot — Matrix–Matrix Multiplication
// ============================================================================

/// Compute the matrix product `C = A · B`.
///
/// This is a convenient shorthand for the ndarray `.dot()` method, useful when
/// the full method syntax is overly verbose or when chaining with other ops
/// from this module.
///
/// # Type Parameters
///
/// `T` must support addition (with identity `Zero`) and multiplication.
///
/// # Arguments
///
/// * `a` — Left operand, shape `(m, k)`
/// * `b` — Right operand, shape `(k, n)`
///
/// # Returns
///
/// A new `Array2<T>` of shape `(m, n)`.
///
/// # Panics
///
/// The inner dimensions of `a` and `b` must match (`a.ncols() == b.nrows()`);
/// ndarray will panic if they do not. This matches the standard contract for
/// `.dot()`.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ops::dot;
/// use ndarray::array;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1.0]]; // Identity
/// let b = array![[3.0_f64, 4.0], [5.0, 6.0]];
/// let c = dot(&a.view(), &b.view());
/// assert_eq!(c, b);
/// ```
pub fn dot<T>(a: &ArrayView2<T>, b: &ArrayView2<T>) -> Array2<T>
where
    T: Clone + Zero + Add<Output = T> + Mul<Output = T>,
{
    let (m, k) = (a.nrows(), a.ncols());
    let (k2, n) = (b.nrows(), b.ncols());
    debug_assert_eq!(k, k2, "dot: inner dimensions must match");

    let mut result = Array2::<T>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for l in 0..k {
                sum = sum + a[[i, l]].clone() * b[[l, j]].clone();
            }
            result[[i, j]] = sum;
        }
    }
    result
}

// ============================================================================
// outer — Outer Product
// ============================================================================

/// Compute the outer product of two 1D arrays.
///
/// Given vectors `u` of length `m` and `v` of length `n`, returns an `m × n`
/// matrix `M` where `M[i, j] = u[i] * v[j]`.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ops::outer;
/// use ndarray::array;
///
/// let u = array![1.0_f64, 2.0, 3.0];
/// let v = array![4.0_f64, 5.0];
/// let m = outer(&u.view(), &v.view());
/// assert_eq!(m.shape(), &[3, 2]);
/// assert_eq!(m[[0, 0]], 4.0);
/// assert_eq!(m[[2, 1]], 15.0);
/// ```
pub fn outer<T>(u: &ArrayView1<T>, v: &ArrayView1<T>) -> Array2<T>
where
    T: Clone + Zero + Mul<Output = T>,
{
    let m = u.len();
    let n = v.len();
    Array2::from_shape_fn((m, n), |(i, j)| u[i].clone() * v[j].clone())
}

// ============================================================================
// kron — Kronecker Product
// ============================================================================

/// Compute the Kronecker product of two 2D arrays.
///
/// If `A` has shape `(p, q)` and `B` has shape `(r, s)`, the result has shape
/// `(p·r, q·s)` where block `(i, j)` equals `A[i,j] * B`.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ops::kron;
/// use ndarray::array;
///
/// let a = array![[1_i32, 0], [0, 1]]; // 2×2 identity
/// let b = array![[1_i32, 2], [3, 4]];
/// let k = kron(&a.view(), &b.view());
/// assert_eq!(k.shape(), &[4, 4]);
/// // Top-left block should be `1 * b`
/// assert_eq!(k[[0, 0]], 1);
/// assert_eq!(k[[0, 1]], 2);
/// ```
pub fn kron<T>(a: &ArrayView2<T>, b: &ArrayView2<T>) -> Array2<T>
where
    T: Clone + Zero + Mul<Output = T>,
{
    let (p, q) = (a.nrows(), a.ncols());
    let (r, s) = (b.nrows(), b.ncols());

    Array2::from_shape_fn((p * r, q * s), |(i, j)| {
        let ai = i / r;
        let bi = i % r;
        let aj = j / s;
        let bj = j % s;
        a[[ai, aj]].clone() * b[[bi, bj]].clone()
    })
}

// ============================================================================
// vstack — Vertical Stack
// ============================================================================

/// Stack a sequence of 2D arrays vertically (concatenate rows).
///
/// All arrays must have the same number of columns. Returns an error if the
/// slice is empty or if any column-count mismatches exist.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ops::vstack;
/// use ndarray::array;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let b = array![[5.0_f64, 6.0]];
/// let s = vstack(&[a.view(), b.view()]).expect("same column count");
/// assert_eq!(s.shape(), &[3, 2]);
/// assert_eq!(s[[2, 0]], 5.0);
/// ```
pub fn vstack<T>(arrays: &[ArrayView2<T>]) -> CoreResult<Array2<T>>
where
    T: Clone + Zero,
{
    if arrays.is_empty() {
        return Err(CoreError::InvalidInput(ErrorContext::new(
            "vstack: cannot stack an empty slice of arrays",
        )));
    }

    let ncols = arrays[0].ncols();
    for (idx, arr) in arrays.iter().enumerate().skip(1) {
        if arr.ncols() != ncols {
            return Err(CoreError::InvalidInput(ErrorContext::new(format!(
                "vstack: array at index {idx} has {cols} columns, expected {ncols}",
                cols = arr.ncols()
            ))));
        }
    }

    let total_rows: usize = arrays.iter().map(|a| a.nrows()).sum();
    let mut result = Array2::<T>::zeros((total_rows, ncols));

    let mut row_offset = 0;
    for arr in arrays {
        let nrows = arr.nrows();
        for r in 0..nrows {
            for c in 0..ncols {
                result[[row_offset + r, c]] = arr[[r, c]].clone();
            }
        }
        row_offset += nrows;
    }

    Ok(result)
}

// ============================================================================
// hstack — Horizontal Stack
// ============================================================================

/// Stack a sequence of 2D arrays horizontally (concatenate columns).
///
/// All arrays must have the same number of rows. Returns an error if the
/// slice is empty or if any row-count mismatches exist.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ops::hstack;
/// use ndarray::array;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let b = array![[5.0_f64], [6.0]];
/// let s = hstack(&[a.view(), b.view()]).expect("same row count");
/// assert_eq!(s.shape(), &[2, 3]);
/// assert_eq!(s[[0, 2]], 5.0);
/// assert_eq!(s[[1, 2]], 6.0);
/// ```
pub fn hstack<T>(arrays: &[ArrayView2<T>]) -> CoreResult<Array2<T>>
where
    T: Clone + Zero,
{
    if arrays.is_empty() {
        return Err(CoreError::InvalidInput(ErrorContext::new(
            "hstack: cannot stack an empty slice of arrays",
        )));
    }

    let nrows = arrays[0].nrows();
    for (idx, arr) in arrays.iter().enumerate().skip(1) {
        if arr.nrows() != nrows {
            return Err(CoreError::InvalidInput(ErrorContext::new(format!(
                "hstack: array at index {idx} has {r} rows, expected {nrows}",
                r = arr.nrows()
            ))));
        }
    }

    let total_cols: usize = arrays.iter().map(|a| a.ncols()).sum();
    let mut result = Array2::<T>::zeros((nrows, total_cols));

    let mut col_offset = 0;
    for arr in arrays {
        let ncols = arr.ncols();
        for r in 0..nrows {
            for c in 0..ncols {
                result[[r, col_offset + c]] = arr[[r, c]].clone();
            }
        }
        col_offset += ncols;
    }

    Ok(result)
}

// ============================================================================
// block_diag — Block Diagonal Matrix
// ============================================================================

/// Build a block-diagonal matrix from a sequence of 2D blocks.
///
/// Given blocks `B₀`, `B₁`, …, `Bₙ` with shapes `(r₀, c₀)`, `(r₁, c₁)`, …,
/// the result is a `(Σrᵢ) × (Σcᵢ)` matrix with each block placed on the
/// diagonal and zeros elsewhere.
///
/// Returns an empty `0×0` matrix when given an empty slice.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ops::block_diag;
/// use ndarray::array;
///
/// let a = array![[1_i32, 2], [3, 4]];
/// let b = array![[5_i32]];
/// let bd = block_diag(&[a.view(), b.view()]);
/// assert_eq!(bd.shape(), &[3, 3]);
/// assert_eq!(bd[[0, 0]], 1);
/// assert_eq!(bd[[2, 2]], 5);
/// assert_eq!(bd[[0, 2]], 0); // off-block element
/// ```
pub fn block_diag<T>(blocks: &[ArrayView2<T>]) -> Array2<T>
where
    T: Clone + Zero,
{
    if blocks.is_empty() {
        return Array2::<T>::zeros((0, 0));
    }

    let total_rows: usize = blocks.iter().map(|b| b.nrows()).sum();
    let total_cols: usize = blocks.iter().map(|b| b.ncols()).sum();

    let mut result = Array2::<T>::zeros((total_rows, total_cols));

    let mut row_off = 0;
    let mut col_off = 0;
    for block in blocks {
        let (br, bc) = (block.nrows(), block.ncols());
        for r in 0..br {
            for c in 0..bc {
                result[[row_off + r, col_off + c]] = block[[r, c]].clone();
            }
        }
        row_off += br;
        col_off += bc;
    }

    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ::ndarray::array;
    use approx::assert_abs_diff_eq;

    // --- dot ---

    #[test]
    fn test_dot_identity() {
        let eye = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![[3.0_f64, 4.0], [5.0, 6.0]];
        let c = dot(&eye.view(), &b.view());
        assert_abs_diff_eq!(c[[0, 0]], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[1, 1]], 6.0, epsilon = 1e-12);
    }

    #[test]
    fn test_dot_rectangular() {
        // (2×3) · (3×2) → (2×2)
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0_f64, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let c = dot(&a.view(), &b.view());
        assert_eq!(c.shape(), &[2, 2]);
        // Row 0: [1·7+2·9+3·11, 1·8+2·10+3·12] = [58, 64]
        assert_abs_diff_eq!(c[[0, 0]], 58.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[0, 1]], 64.0, epsilon = 1e-12);
        // Row 1: [4·7+5·9+6·11, 4·8+5·10+6·12] = [139, 154]
        assert_abs_diff_eq!(c[[1, 0]], 139.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[[1, 1]], 154.0, epsilon = 1e-12);
    }

    #[test]
    fn test_dot_integers() {
        let a = array![[1_i32, 2], [3, 4]];
        let b = array![[5_i32, 6], [7, 8]];
        let c = dot(&a.view(), &b.view());
        assert_eq!(c[[0, 0]], 19); // 1·5+2·7
        assert_eq!(c[[1, 1]], 50); // 3·6+4·8
    }

    // --- outer ---

    #[test]
    fn test_outer_basic() {
        let u = array![1.0_f64, 2.0, 3.0];
        let v = array![4.0_f64, 5.0];
        let m = outer(&u.view(), &v.view());
        assert_eq!(m.shape(), &[3, 2]);
        assert_abs_diff_eq!(m[[0, 0]], 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(m[[1, 1]], 10.0, epsilon = 1e-12);
        assert_abs_diff_eq!(m[[2, 0]], 12.0, epsilon = 1e-12);
        assert_abs_diff_eq!(m[[2, 1]], 15.0, epsilon = 1e-12);
    }

    #[test]
    fn test_outer_integers() {
        let u = array![1_i32, 2];
        let v = array![3_i32, 4, 5];
        let m = outer(&u.view(), &v.view());
        assert_eq!(m.shape(), &[2, 3]);
        assert_eq!(m[[0, 0]], 3);
        assert_eq!(m[[1, 2]], 10);
    }

    // --- kron ---

    #[test]
    fn test_kron_identity_identity() {
        let eye2 = array![[1_i32, 0], [0, 1]];
        let eye3 = array![[1_i32, 0, 0], [0, 1, 0], [0, 0, 1]];
        let k = kron(&eye2.view(), &eye3.view());
        assert_eq!(k.shape(), &[6, 6]);
        // Should also be the 6×6 identity
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(k[[i, j]], if i == j { 1 } else { 0 });
            }
        }
    }

    #[test]
    fn test_kron_scalar() {
        let two = array![[2_i32]];
        let b = array![[1_i32, 2], [3, 4]];
        let k = kron(&two.view(), &b.view());
        assert_eq!(k.shape(), &[2, 2]);
        assert_eq!(k[[0, 0]], 2);
        assert_eq!(k[[1, 1]], 8);
    }

    #[test]
    fn test_kron_matches_expected() {
        // From NumPy docs:
        // A = [[1, 2], [3, 4]]
        // B = [[0, 5], [6, 7]]
        let a = array![[1_i32, 2], [3, 4]];
        let b = array![[0_i32, 5], [6, 7]];
        let k = kron(&a.view(), &b.view());
        assert_eq!(k.shape(), &[4, 4]);
        // Row 0: 1*[0,5] ++ 2*[0,5] = [0,5,0,10]
        assert_eq!(k[[0, 0]], 0);
        assert_eq!(k[[0, 1]], 5);
        assert_eq!(k[[0, 2]], 0);
        assert_eq!(k[[0, 3]], 10);
        // Row 3: 3*[6,7] ++ 4*[6,7] = [18,21,24,28]
        assert_eq!(k[[3, 0]], 18);
        assert_eq!(k[[3, 1]], 21);
        assert_eq!(k[[3, 2]], 24);
        assert_eq!(k[[3, 3]], 28);
    }

    // --- vstack ---

    #[test]
    fn test_vstack_basic() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[5.0_f64, 6.0]];
        let s = vstack(&[a.view(), b.view()]).expect("same cols");
        assert_eq!(s.shape(), &[3, 2]);
        assert_abs_diff_eq!(s[[2, 0]], 5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(s[[2, 1]], 6.0, epsilon = 1e-12);
    }

    #[test]
    fn test_vstack_three_arrays() {
        let a = array![[1_i32, 2]];
        let b = array![[3_i32, 4]];
        let c = array![[5_i32, 6], [7, 8]];
        let s = vstack(&[a.view(), b.view(), c.view()]).expect("same cols");
        assert_eq!(s.shape(), &[4, 2]);
        assert_eq!(s[[0, 0]], 1);
        assert_eq!(s[[1, 1]], 4);
        assert_eq!(s[[2, 0]], 5);
        assert_eq!(s[[3, 1]], 8);
    }

    #[test]
    fn test_vstack_mismatch_error() {
        let a = array![[1.0_f64, 2.0, 3.0]]; // 3 cols
        let b = array![[4.0_f64, 5.0]]; // 2 cols
        assert!(vstack(&[a.view(), b.view()]).is_err());
    }

    #[test]
    fn test_vstack_empty_error() {
        let empty: &[ArrayView2<f64>] = &[];
        assert!(vstack(empty).is_err());
    }

    // --- hstack ---

    #[test]
    fn test_hstack_basic() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[5.0_f64], [6.0]];
        let s = hstack(&[a.view(), b.view()]).expect("same rows");
        assert_eq!(s.shape(), &[2, 3]);
        assert_abs_diff_eq!(s[[0, 2]], 5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(s[[1, 2]], 6.0, epsilon = 1e-12);
    }

    #[test]
    fn test_hstack_three_arrays() {
        let a = array![[1_i32], [2]];
        let b = array![[3_i32], [4]];
        let c = array![[5_i32, 6], [7, 8]];
        let s = hstack(&[a.view(), b.view(), c.view()]).expect("same rows");
        assert_eq!(s.shape(), &[2, 4]);
        assert_eq!(s[[0, 0]], 1);
        assert_eq!(s[[0, 1]], 3);
        assert_eq!(s[[1, 3]], 8);
    }

    #[test]
    fn test_hstack_mismatch_error() {
        let a = array![[1.0_f64], [2.0], [3.0]]; // 3 rows
        let b = array![[4.0_f64], [5.0]]; // 2 rows
        assert!(hstack(&[a.view(), b.view()]).is_err());
    }

    #[test]
    fn test_hstack_empty_error() {
        let empty: &[ArrayView2<f64>] = &[];
        assert!(hstack(empty).is_err());
    }

    // --- block_diag ---

    #[test]
    fn test_block_diag_square_blocks() {
        let a = array![[1_i32, 2], [3, 4]];
        let b = array![[5_i32, 6], [7, 8]];
        let bd = block_diag(&[a.view(), b.view()]);
        assert_eq!(bd.shape(), &[4, 4]);
        assert_eq!(bd[[0, 0]], 1);
        assert_eq!(bd[[1, 1]], 4);
        assert_eq!(bd[[2, 2]], 5);
        assert_eq!(bd[[3, 3]], 8);
        // Off-diagonal blocks should be zero
        assert_eq!(bd[[0, 2]], 0);
        assert_eq!(bd[[3, 0]], 0);
    }

    #[test]
    fn test_block_diag_rectangular_blocks() {
        let a = array![[1_i32, 2, 3]]; // 1×3
        let b = array![[4_i32], [5]]; // 2×1
        let bd = block_diag(&[a.view(), b.view()]);
        assert_eq!(bd.shape(), &[3, 4]);
        // a block at rows 0, cols 0..3
        assert_eq!(bd[[0, 2]], 3);
        // b block at rows 1..3, col 3
        assert_eq!(bd[[1, 3]], 4);
        assert_eq!(bd[[2, 3]], 5);
        // zeros
        assert_eq!(bd[[1, 0]], 0);
    }

    #[test]
    fn test_block_diag_single() {
        let a = array![[9_i32]];
        let bd = block_diag(&[a.view()]);
        assert_eq!(bd.shape(), &[1, 1]);
        assert_eq!(bd[[0, 0]], 9);
    }

    #[test]
    fn test_block_diag_empty() {
        let empty: &[ArrayView2<i32>] = &[];
        let bd = block_diag(empty);
        assert_eq!(bd.shape(), &[0, 0]);
    }

    #[test]
    fn test_block_diag_three_blocks() {
        let a = array![[1_i32, 2], [3, 4]];
        let b = array![[5_i32]];
        let c = array![[6_i32, 7, 8]];
        let bd = block_diag(&[a.view(), b.view(), c.view()]);
        assert_eq!(bd.shape(), &[4, 6]);
        // Check corners of each block
        assert_eq!(bd[[0, 0]], 1);
        assert_eq!(bd[[1, 1]], 4);
        assert_eq!(bd[[2, 2]], 5);
        assert_eq!(bd[[3, 3]], 6);
        assert_eq!(bd[[3, 5]], 8);
        // Verify zeros outside blocks
        assert_eq!(bd[[0, 3]], 0);
        assert_eq!(bd[[3, 0]], 0);
    }
}
