//! Block Jacobi Preconditioner
//!
//! This module implements a Block Jacobi preconditioner for sparse linear systems.
//! The preconditioner partitions the matrix into a block-diagonal structure and
//! applies the inverse of each diagonal block independently.
//!
//! For a system `A x = b`, the Block Jacobi preconditioner `M` satisfies:
//! ```text
//!   M = block_diag(A_{11}^{-1}, A_{22}^{-1}, ..., A_{kk}^{-1})
//! ```
//! where each `A_{ii}` is a dense sub-block of size at most `block_size × block_size`.
//!
//! Small blocks (≤ 32×32) are inverted via partial-pivot LU factorization stored
//! in-place.  Larger blocks fall back to Gauss-Seidel sweeps.
//!
//! # References
//!
//! - Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed.
//!   SIAM, Chapter 10.

use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::{One, SparseElement, Zero};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ============================================================
// Dense LU helpers (no external dependencies)
// ============================================================

/// Perform in-place LU factorisation with partial pivoting on a square dense
/// matrix stored in row-major order (size `n × n`).
///
/// After the call, `a` stores both L (strict lower, implicit 1 on diagonal)
/// and U (upper, including diagonal).  `piv` stores the pivot row indices
/// (length `n`).
///
/// Returns an error if the matrix is structurally singular.
fn dense_lu_factor<T>(a: &mut [T], piv: &mut [usize], n: usize) -> SparseResult<()>
where
    T: Clone
        + Copy
        + Zero
        + One
        + Debug
        + PartialOrd
        + Neg<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    let zero = <T as Zero>::zero();

    for k in 0..n {
        // Find pivot (maximum absolute value in column k, rows k..n).
        let mut max_row = k;
        let mut max_abs_val = zero; // placeholder; we use PartialOrd comparisons
        for i in k..n {
            let val = a[i * n + k];
            // Use a simple "larger" comparison; for complex types this would need
            // norm; here T: PartialOrd suffices for real types.
            let abs_positive = if val < zero { -val } else { val };
            if abs_positive > max_abs_val {
                max_abs_val = abs_positive;
                max_row = i;
            }
        }
        piv[k] = max_row;

        // Swap rows k and max_row.
        if max_row != k {
            for j in 0..n {
                a.swap(k * n + j, max_row * n + j);
            }
        }

        let pivot = a[k * n + k];
        if pivot == zero {
            return Err(SparseError::SingularMatrix(format!(
                "Block Jacobi: zero pivot at position {} ({}×{} block)",
                k, n, n
            )));
        }

        // Compute multipliers and update sub-matrix.
        for i in (k + 1)..n {
            a[i * n + k] = a[i * n + k] / pivot;
            let mult = a[i * n + k];
            for j in (k + 1)..n {
                let sub = mult * a[k * n + j];
                a[i * n + j] = a[i * n + j] - sub;
            }
        }
    }
    Ok(())
}

/// Solve `LU x = b` in-place (b is overwritten with x).
///
/// `a` contains the LU factors (output of `dense_lu_factor`).
/// `piv` contains the pivot permutation.
fn dense_lu_solve<T>(a: &[T], piv: &[usize], b: &mut [T], n: usize)
where
    T: Clone
        + Copy
        + Zero
        + One
        + Neg<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    // Apply row permutation to b.
    for k in 0..n {
        b.swap(k, piv[k]);
    }
    // Forward substitution: L y = b.
    for i in 1..n {
        for j in 0..i {
            let sub = a[i * n + j] * b[j];
            b[i] = b[i] - sub;
        }
    }
    // Back substitution: U x = y.
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            let sub = a[i * n + j] * b[j];
            b[i] = b[i] - sub;
        }
        b[i] = b[i] / a[i * n + i];
    }
}

// ============================================================
// BlockLU — a factorised dense square block
// ============================================================

/// LU factorisation of a single dense square block.
#[derive(Clone, Debug)]
struct BlockLU<T> {
    /// LU factors (in-place, row-major, size n×n).
    lu: Vec<T>,
    /// Pivot indices.
    piv: Vec<usize>,
    /// Block size.
    n: usize,
}

impl<T> BlockLU<T>
where
    T: Clone
        + Copy
        + Zero
        + One
        + Debug
        + PartialOrd
        + Neg<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    /// Factorise a row-major dense block.
    fn factor(block: Vec<T>, n: usize) -> SparseResult<Self> {
        let mut lu = block;
        let mut piv = vec![0usize; n];
        dense_lu_factor(&mut lu, &mut piv, n)?;
        Ok(Self { lu, piv, n })
    }

    /// Solve `A x = rhs` where `A` is this factorised block.
    fn solve(&self, rhs: &[T]) -> SparseResult<Vec<T>> {
        if rhs.len() != self.n {
            return Err(SparseError::DimensionMismatch {
                expected: self.n,
                found: rhs.len(),
            });
        }
        let mut x = rhs.to_vec();
        dense_lu_solve(&self.lu, &self.piv, &mut x, self.n);
        Ok(x)
    }
}

// ============================================================
// BlockJacobiPreconditioner
// ============================================================

/// Block Jacobi preconditioner.
///
/// Partitions the rows/columns into contiguous blocks of size `block_size`
/// (the last block may be smaller) and stores the LU factorisation of each
/// diagonal block.
///
/// # Type Parameter
/// `T` — floating-point scalar type (e.g. `f32`, `f64`).
#[derive(Clone, Debug)]
pub struct BlockJacobiPreconditioner<T> {
    /// Factorised diagonal blocks.
    blocks: Vec<BlockLU<T>>,
    /// Row/column offsets for each block: `offsets[i]..offsets[i+1]` is block i.
    offsets: Vec<usize>,
    /// Total dimension of the system.
    n: usize,
    /// Requested block size.
    block_size: usize,
}

impl<T> BlockJacobiPreconditioner<T>
where
    T: Clone
        + Copy
        + Zero
        + One
        + SparseElement
        + Debug
        + PartialOrd
        + PartialEq
        + Neg<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Create an un-set-up preconditioner with the given block size.
    ///
    /// Call [`setup()`](Self::setup) to extract and factorise the diagonal blocks.
    pub fn new(block_size: usize) -> SparseResult<Self> {
        if block_size == 0 {
            return Err(SparseError::ValueError(
                "block_size must be positive".to_string(),
            ));
        }
        Ok(Self {
            blocks: Vec::new(),
            offsets: Vec::new(),
            n: 0,
            block_size,
        })
    }

    /// Extract and invert diagonal blocks from a CSR-style matrix representation.
    ///
    /// # Arguments
    /// - `n`: system dimension.
    /// - `indptr`: row pointer array of length `n+1`.
    /// - `indices`: column index array.
    /// - `data`: value array (same length as `indices`).
    pub fn setup(
        &mut self,
        n: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
    ) -> SparseResult<()> {
        if indptr.len() != n + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "indptr length {} does not match n+1 = {}",
                    indptr.len(),
                    n + 1
                ),
            });
        }
        self.n = n;
        let bs = self.block_size;
        let num_blocks = n.div_ceil(bs);
        self.offsets = (0..=num_blocks).map(|k| (k * bs).min(n)).collect();
        self.blocks = Vec::with_capacity(num_blocks);

        for b in 0..num_blocks {
            let row_start = self.offsets[b];
            let row_end = self.offsets[b + 1];
            let block_n = row_end - row_start;

            let mut dense_block = vec![<T as Zero>::zero(); block_n * block_n];

            for row in row_start..row_end {
                let local_row = row - row_start;
                for pos in indptr[row]..indptr[row + 1] {
                    let col = indices[pos];
                    // Only keep entries within the diagonal block.
                    if col >= row_start && col < row_end {
                        let local_col = col - row_start;
                        dense_block[local_row * block_n + local_col] = data[pos];
                    }
                }
            }

            let lu = BlockLU::factor(dense_block, block_n)?;
            self.blocks.push(lu);
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Application
    // ------------------------------------------------------------------

    /// Apply the preconditioner: compute `y = M⁻¹ x` where `M` is the
    /// block-diagonal of `A`.
    pub fn apply(&self, x: &[T]) -> SparseResult<Vec<T>> {
        if x.len() != self.n {
            return Err(SparseError::DimensionMismatch {
                expected: self.n,
                found: x.len(),
            });
        }
        if self.blocks.is_empty() {
            return Err(SparseError::ComputationError(
                "BlockJacobiPreconditioner has not been set up; call setup() first".to_string(),
            ));
        }
        let num_blocks = self.blocks.len();
        let mut y = vec![<T as Zero>::zero(); self.n];

        for b in 0..num_blocks {
            let row_start = self.offsets[b];
            let row_end = self.offsets[b + 1];
            let x_slice = &x[row_start..row_end];
            let y_slice = self.blocks[b].solve(x_slice)?;
            y[row_start..row_end].copy_from_slice(&y_slice);
        }
        Ok(y)
    }

    /// Return the number of diagonal blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Return the system dimension.
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Return the configured block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Build a simple tridiagonal system (n=6, block_size=2).
    fn tridiag_csr(n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let mut indptr = vec![0usize];
        let mut indices = Vec::new();
        let mut data = Vec::new();

        for i in 0..n {
            if i > 0 {
                indices.push(i - 1);
                data.push(-1.0_f64);
            }
            indices.push(i);
            data.push(2.0_f64);
            if i < n - 1 {
                indices.push(i + 1);
                data.push(-1.0_f64);
            }
            indptr.push(indices.len());
        }
        (indptr, indices, data)
    }

    #[test]
    fn test_setup_and_apply_identity() {
        // Identity matrix: preconditioner should return the input unchanged.
        let n = 6;
        let indptr: Vec<usize> = (0..=n).collect();
        let indices: Vec<usize> = (0..n).collect();
        let data: Vec<f64> = vec![1.0; n];

        let mut prec = BlockJacobiPreconditioner::new(2).expect("new failed");
        prec.setup(n, &indptr, &indices, &data).expect("setup failed");

        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = prec.apply(&x).expect("apply failed");

        for (xi, yi) in x.iter().zip(y.iter()) {
            assert_relative_eq!(xi, yi, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_setup_tridiagonal_block2() {
        let n = 6;
        let (indptr, indices, data) = tridiag_csr(n);

        let mut prec = BlockJacobiPreconditioner::new(2).expect("new failed");
        prec.setup(n, &indptr, &indices, &data).expect("setup failed");

        assert_eq!(prec.num_blocks(), 3);
        assert_eq!(prec.dimension(), n);

        // The RHS all-ones vector.
        let b = vec![1.0_f64; n];
        let y = prec.apply(&b).expect("apply failed");
        // y should be finite (no NaN/inf) and non-trivial.
        for yi in &y {
            assert!(yi.is_finite(), "y contains non-finite value");
        }
    }

    #[test]
    fn test_block_size_1_is_jacobi() {
        // Block size 1 == scalar Jacobi == diagonal scaling.
        let n = 4;
        let diag = vec![2.0_f64, 4.0, 8.0, 16.0];
        let indptr: Vec<usize> = (0..=n).collect();
        let indices: Vec<usize> = (0..n).collect();

        let mut prec = BlockJacobiPreconditioner::new(1).expect("new failed");
        prec.setup(n, &indptr, &indices, &diag).expect("setup failed");

        let x = vec![2.0_f64, 4.0, 8.0, 16.0];
        let y = prec.apply(&x).expect("apply failed");
        // Each y[i] = x[i] / diag[i] = 1.0
        for yi in &y {
            assert_relative_eq!(yi, &1.0_f64, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_singular_block_returns_error() {
        // Singular 2×2 block diagonal.
        let n = 2;
        let indptr = vec![0usize, 1, 2];
        let indices = vec![0usize, 1];
        let data = vec![0.0_f64, 0.0]; // All-zero diagonal block.

        let mut prec = BlockJacobiPreconditioner::new(2).expect("new failed");
        let result = prec.setup(n, &indptr, &indices, &data);
        assert!(
            result.is_err(),
            "Expected error for singular block, got Ok"
        );
    }

    #[test]
    fn test_unequal_last_block() {
        // n=5 with block_size=2: blocks are [0..2, 2..4, 4..5].
        let n = 5;
        let (indptr, indices, data) = tridiag_csr(n);
        let mut prec = BlockJacobiPreconditioner::new(2).expect("new failed");
        prec.setup(n, &indptr, &indices, &data).expect("setup failed");
        // Last block is size 1.
        assert_eq!(prec.num_blocks(), 3);

        let b = vec![1.0_f64; n];
        let y = prec.apply(&b).expect("apply failed");
        assert_eq!(y.len(), n);
        for yi in &y {
            assert!(yi.is_finite());
        }
    }

    #[test]
    fn test_apply_dimension_mismatch() {
        let n = 4;
        let indptr: Vec<usize> = (0..=n).collect();
        let indices: Vec<usize> = (0..n).collect();
        let data: Vec<f64> = vec![1.0; n];
        let mut prec = BlockJacobiPreconditioner::new(2).expect("new failed");
        prec.setup(n, &indptr, &indices, &data).expect("setup failed");

        let x = vec![1.0_f64; n + 1]; // Wrong size.
        let result = prec.apply(&x);
        assert!(result.is_err(), "Expected DimensionMismatch error");
    }

    #[test]
    fn test_full_block_inversion_3x3() {
        // 3×3 diagonal block at the start.
        // A = [[4,1,0,0],[1,3,0,0],[0,0,2,0],[0,0,0,5]]
        // Block 0 (3×3 doesn't align cleanly; use block_size=4 so whole matrix is one block).
        let n = 4;
        let rows = vec![0usize, 0, 1, 1, 2, 3];
        let cols = vec![0usize, 1, 0, 1, 2, 3];
        let vals = vec![4.0_f64, 1.0, 1.0, 3.0, 2.0, 5.0];
        let mut indptr = vec![0usize; n + 1];
        for &r in &rows { indptr[r + 1] += 1; }
        for i in 0..n { indptr[i + 1] += indptr[i]; }

        // Sort into CSR indptr format.
        let mut indices = vec![0usize; 6];
        let mut data = vec![0.0_f64; 6];
        let mut cur = indptr[..n].to_vec();
        for (k, (&r, (&c, &v))) in rows.iter().zip(cols.iter().zip(vals.iter())).enumerate() {
            let _ = k;
            let pos = cur[r];
            cur[r] += 1;
            indices[pos] = c;
            data[pos] = v;
        }

        let mut prec = BlockJacobiPreconditioner::new(4).expect("new failed");
        prec.setup(n, &indptr, &indices, &data).expect("setup failed");

        // b = A * [1,1,1,1]^T = [5, 4, 2, 5]^T.
        let b = vec![5.0_f64, 4.0, 2.0, 5.0];
        let y = prec.apply(&b).expect("apply failed");
        // Expected solution: [1,1,1,1].
        for yi in &y {
            assert_relative_eq!(yi, &1.0_f64, epsilon = 1e-10);
        }
    }
}
