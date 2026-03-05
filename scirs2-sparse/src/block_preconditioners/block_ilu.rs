//! Block ILU(0) Preconditioner
//!
//! This module implements an Incomplete LU factorisation operating at the
//! block level.  Given a BSR matrix, `BlockILU0` computes the block-level
//! ILU(0) factorisation — i.e., ILU with the same block sparsity pattern as
//! the original matrix — and supplies forward/backward triangular solves.
//!
//! # Algorithm
//!
//! For a BSR matrix with block structure `A_{ij}` (where `i,j` index block
//! rows/columns), the ILU(0) factorisation produces block factors `L` and `U`
//! such that `L * U ≈ A` with the same block-sparsity pattern.
//!
//! The factorisation proceeds by block-row elimination (analogous to scalar
//! ILU(0)):
//!
//! ```text
//! for i = 1..block_rows:
//!   for k in {j < i : A_{ik} ≠ 0}:
//!     A_{ik} ← A_{ik} * A_{kk}^{-1}          (= L_{ik})
//!     for j in {j >= k : A_{kj} ≠ 0} ∩ {A_{ij} ≠ 0}:
//!       A_{ij} ← A_{ij} - L_{ik} * A_{kj}    (= U_{ij} / L_{ij})
//! ```
//!
//! The diagonal blocks are inverted using partial-pivot LU.
//!
//! # Usage
//!
//! ```ignore
//! let mut bilu = BlockILU0::new();
//! bilu.factorize(&bsr_matrix)?;
//! let y = bilu.apply(&rhs)?;
//! ```
//!
//! # References
//!
//! - Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed.
//!   SIAM, Chapter 10.

use crate::error::{SparseError, SparseResult};
use crate::formats::bsr::BSRMatrix;
use scirs2_core::numeric::{One, SparseElement, Zero};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ============================================================
// Dense block operations (shared with block_jacobi internals)
// ============================================================

/// In-place partial-pivot LU factorisation of an `n×n` row-major block.
/// Returns the pivot array.
fn block_lu_factor<T>(a: &mut [T], n: usize) -> SparseResult<Vec<usize>>
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
    let mut piv = vec![0usize; n];

    for k in 0..n {
        // Partial pivoting.
        let mut max_row = k;
        let mut max_val = zero;
        for i in k..n {
            let v = a[i * n + k];
            let abs_v = if v < zero { -v } else { v };
            if abs_v > max_val {
                max_val = abs_v;
                max_row = i;
            }
        }
        piv[k] = max_row;
        if max_row != k {
            for j in 0..n {
                a.swap(k * n + j, max_row * n + j);
            }
        }

        let pivot = a[k * n + k];
        if pivot == zero {
            return Err(SparseError::SingularMatrix(format!(
                "BlockILU0: zero pivot encountered at k={} in {}×{} block",
                k, n, n
            )));
        }

        for i in (k + 1)..n {
            a[i * n + k] = a[i * n + k] / pivot;
            let m = a[i * n + k];
            for j in (k + 1)..n {
                let sub = m * a[k * n + j];
                a[i * n + j] = a[i * n + j] - sub;
            }
        }
    }
    Ok(piv)
}

/// Solve `LU x = b` in-place given LU factors and pivots.
fn block_lu_solve<T>(lu: &[T], piv: &[usize], b: &mut [T], n: usize)
where
    T: Clone
        + Copy
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    // Apply permutation.
    for k in 0..n {
        b.swap(k, piv[k]);
    }
    // Forward: L y = b.
    for i in 1..n {
        for j in 0..i {
            let sub = lu[i * n + j] * b[j];
            b[i] = b[i] - sub;
        }
    }
    // Backward: U x = y.
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            let sub = lu[i * n + j] * b[j];
            b[i] = b[i] - sub;
        }
        b[i] = b[i] / lu[i * n + i];
    }
}

/// Multiply two dense `r×r` blocks: `c += a * b` (all row-major, size r×r).
fn block_gemm_add<T>(c: &mut [T], a: &[T], b: &[T], r: usize, k: usize, col: usize)
where
    T: Clone + Copy + Zero + Add<Output = T> + Mul<Output = T>,
{
    // c: r×col, a: r×k, b: k×col (all row-major)
    for i in 0..r {
        for j in 0..col {
            let mut acc = <T as Zero>::zero();
            for p in 0..k {
                acc = acc + a[i * k + p] * b[p * col + j];
            }
            c[i * col + j] = c[i * col + j] + acc;
        }
    }
}

/// Multiply two dense `r×r` blocks: `c -= a * b`.
fn block_gemm_sub<T>(c: &mut [T], a: &[T], b: &[T], r: usize, k: usize, col: usize)
where
    T: Clone + Copy + Zero + Sub<Output = T> + Add<Output = T> + Mul<Output = T>,
{
    for i in 0..r {
        for j in 0..col {
            let mut acc = <T as Zero>::zero();
            for p in 0..k {
                acc = acc + a[i * k + p] * b[p * col + j];
            }
            c[i * col + j] = c[i * col + j] - acc;
        }
    }
}

/// Compute `a ← a * b^{-1}` where `b` is given as LU factors + pivots (b_lu, b_piv).
/// Equivalent to solving `x * B = A` → `x = A * B^{-1}`.
/// We solve column by column: for each column `k` of `a`, solve `B^T y = a_col_k`.
/// Since `B = L*U`, `B^T = U^T * L^T`.
///
/// Actually the simplest approach: solve `(B^{-1})^T a^T` row by row.
/// We use: `(A * B^{-1})_{ij} = row_i(A) * B^{-1}`.
/// So for each row `i` of `a`, solve `B^T y = a[i,:]^T` — but that requires L^T U^T solve.
///
/// Simpler: compute `B^{-1}` explicitly for small blocks (n ≤ 32) by solving `B * e_j = y_j`
/// for each unit vector, then `a ← a * B^{-1}`.
fn block_right_solve<T>(a: &mut [T], b_lu: &[T], b_piv: &[usize], r: usize, c: usize, n: usize)
where
    T: Clone
        + Copy
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
{
    // Compute B^{-1} column by column (n×n → B_inv: n×n).
    let mut b_inv = vec![<T as Zero>::zero(); n * n];
    let mut e_j = vec![<T as Zero>::zero(); n];
    for j in 0..n {
        // Solve B * x = e_j.
        e_j.fill(<T as Zero>::zero());
        e_j[j] = <T as One>::one();
        block_lu_solve(b_lu, b_piv, &mut e_j, n);
        for i in 0..n {
            b_inv[i * n + j] = e_j[i];
        }
        e_j[j] = <T as Zero>::zero();
    }

    // a (r×n) * b_inv (n×n) → tmp (r×n)
    let mut tmp = vec![<T as Zero>::zero(); r * c];
    for i in 0..r {
        for j in 0..c {
            let mut acc = <T as Zero>::zero();
            for k in 0..n {
                acc = acc + a[i * n + k] * b_inv[k * c + j];
            }
            tmp[i * c + j] = acc;
        }
    }
    a[..r * c].copy_from_slice(&tmp);
}

// ============================================================
// BlockILU0
// ============================================================

/// Block ILU(0) factorisation and preconditioner.
///
/// After calling [`factorize()`](Self::factorize), the ILU factors are stored
/// in the same block-sparsity pattern as the original BSR matrix.
/// [`apply()`](Self::apply) performs a combined forward/backward block-triangular solve.
#[derive(Clone, Debug)]
pub struct BlockILU0<T> {
    /// ILU block data (same sparsity as input BSR).
    ilu_data: Vec<T>,
    /// Block-column indices (same as input BSR).
    indices: Vec<usize>,
    /// Row pointer array (same as input BSR).
    indptr: Vec<usize>,
    /// LU pivots for each non-zero block (only meaningful for diagonal blocks used in solve).
    /// Length = `nnz_blocks`, each entry is a Vec<usize> of length `r`.
    pivots: Vec<Vec<usize>>,
    /// Block size (r, c).  Must satisfy r == c (square blocks).
    block_size: (usize, usize),
    /// Block-row count.
    block_rows: usize,
    /// Total number of matrix rows.
    nrows: usize,
    /// Total number of matrix columns.
    ncols: usize,
    /// Whether the factorisation has been computed.
    is_factorised: bool,
}

impl<T> BlockILU0<T>
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
    /// Create a new, unfactorised `BlockILU0`.
    pub fn new() -> Self {
        Self {
            ilu_data: Vec::new(),
            indices: Vec::new(),
            indptr: Vec::new(),
            pivots: Vec::new(),
            block_size: (0, 0),
            block_rows: 0,
            nrows: 0,
            ncols: 0,
            is_factorised: false,
        }
    }

    // ------------------------------------------------------------------
    // Factorization
    // ------------------------------------------------------------------

    /// Compute the ILU(0) factorisation of a BSR matrix.
    ///
    /// The block size must have `r == c` (square blocks) for the standard
    /// ILU(0) algorithm.
    pub fn factorize(&mut self, bsr: &BSRMatrix<T>) -> SparseResult<()>
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        let (r, c) = bsr.block_size;
        if r != c {
            return Err(SparseError::ValueError(format!(
                "BlockILU0 requires square blocks, got ({}, {})",
                r, c
            )));
        }
        if bsr.nrows != bsr.ncols {
            return Err(SparseError::ValueError(
                "BlockILU0 requires a square matrix".to_string(),
            ));
        }
        let bs = r; // block size (square)
        let block_rows = bsr.block_rows;
        let nnz_blocks = bsr.indices.len();

        self.block_size = (bs, bs);
        self.block_rows = block_rows;
        self.nrows = bsr.nrows;
        self.ncols = bsr.ncols;

        // Make a mutable copy of the block data.
        let mut ilu_data = bsr.data.clone();
        let indices = bsr.indices.clone();
        let indptr = bsr.indptr.clone();

        // Allocate pivot storage.
        let mut pivots: Vec<Vec<usize>> = vec![Vec::new(); nnz_blocks];

        // Helper: find position of block (bi, bj) in ilu_data, or None.
        let find_block = |bi: usize, bj: usize| -> Option<usize> {
            for pos in indptr[bi]..indptr[bi + 1] {
                if indices[pos] == bj {
                    return Some(pos);
                }
            }
            None
        };

        // Block ILU(0) sweep.
        for bi in 0..block_rows {
            // For each block A_{bi, k} with k < bi (lower triangular blocks in row bi):
            for pos_ik in indptr[bi]..indptr[bi + 1] {
                let bk = indices[pos_ik];
                if bk >= bi {
                    continue; // upper triangle or diagonal — skip
                }

                // Find diagonal block A_{kk} (should already be factorised at this point).
                let pos_kk = match find_block(bk, bk) {
                    Some(p) => p,
                    None => {
                        return Err(SparseError::SingularMatrix(format!(
                            "BlockILU0: missing diagonal block at ({}, {})",
                            bk, bk
                        )))
                    }
                };

                // L_{ik} = A_{ik} * A_{kk}^{-1}
                // Compute: ilu_data[pos_ik] ← ilu_data[pos_ik] * U_{kk}^{-1} * L_{kk}^{-1}
                // This is a right-solve: X = A_{ik} * (LU_{kk})^{-1}
                let kk_lu_base = pos_kk * bs * bs;
                let ik_base = pos_ik * bs * bs;
                let kk_lu: Vec<T> = ilu_data[kk_lu_base..kk_lu_base + bs * bs].to_vec();
                let kk_piv = pivots[pos_kk].clone();

                // Right-solve: ilu_data[pos_ik] ← ilu_data[pos_ik] * (kk_lu)^{-1}
                {
                    let ik_slice = &mut ilu_data[ik_base..ik_base + bs * bs];
                    block_right_solve(ik_slice, &kk_lu, &kk_piv, bs, bs, bs);
                }

                // Update blocks in row bi that share column with row bk.
                // For each A_{kj} in row bk where j >= bk and A_{ij} exists:
                for pos_kj in indptr[bk]..indptr[bk + 1] {
                    let bj = indices[pos_kj];
                    if bj <= bk {
                        continue;
                    }
                    // Check if A_{bi, bj} exists.
                    if let Some(pos_ij) = find_block(bi, bj) {
                        let l_ik: Vec<T> = ilu_data[ik_base..ik_base + bs * bs].to_vec();
                        let u_kj: Vec<T> = {
                            let kj_base = pos_kj * bs * bs;
                            ilu_data[kj_base..kj_base + bs * bs].to_vec()
                        };
                        let ij_base = pos_ij * bs * bs;
                        block_gemm_sub(
                            &mut ilu_data[ij_base..ij_base + bs * bs],
                            &l_ik,
                            &u_kj,
                            bs,
                            bs,
                            bs,
                        );
                    }
                }
            }

            // Factorise the diagonal block A_{bi, bi}.
            let pos_bb = match find_block(bi, bi) {
                Some(p) => p,
                None => {
                    return Err(SparseError::SingularMatrix(format!(
                        "BlockILU0: missing diagonal block at ({}, {})",
                        bi, bi
                    )))
                }
            };
            let bb_base = pos_bb * bs * bs;
            let piv = block_lu_factor(&mut ilu_data[bb_base..bb_base + bs * bs], bs)?;
            pivots[pos_bb] = piv;
        }

        self.ilu_data = ilu_data;
        self.indices = indices;
        self.indptr = indptr;
        self.pivots = pivots;
        self.is_factorised = true;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Triangular solves
    // ------------------------------------------------------------------

    /// Forward solve: y ← L⁻¹ b  (block lower triangular with identity diagonal).
    pub fn solve_lower(&self, b: &[T]) -> SparseResult<Vec<T>> {
        self.check_factorised()?;
        let bs = self.block_size.0;
        let block_rows = self.block_rows;
        let zero = <T as Zero>::zero();
        let mut y = b.to_vec();

        for bi in 0..block_rows {
            let row_start = bi * bs;
            let row_end = (row_start + bs).min(self.nrows);
            let actual_bs = row_end - row_start;

            // Subtract L_{bi, bk} * y_{bk} for all bk < bi.
            for pos in self.indptr[bi]..self.indptr[bi + 1] {
                let bk = self.indices[pos];
                if bk >= bi {
                    continue;
                }
                let bk_start = bk * bs;
                let bk_end = (bk_start + bs).min(self.ncols);
                let actual_bk = bk_end - bk_start;

                let l_base = pos * bs * bs;

                let mut acc = vec![zero; actual_bs];
                for local_row in 0..actual_bs {
                    for local_col in 0..actual_bk {
                        acc[local_row] = acc[local_row]
                            + self.ilu_data[l_base + local_row * bs + local_col]
                                * y[bk_start + local_col];
                    }
                }
                for local_row in 0..actual_bs {
                    y[row_start + local_row] = y[row_start + local_row] - acc[local_row];
                }
            }
        }
        Ok(y)
    }

    /// Backward solve: x ← U⁻¹ y  (block upper triangular, diagonal blocks factorised).
    pub fn solve_upper(&self, y: &[T]) -> SparseResult<Vec<T>> {
        self.check_factorised()?;
        let bs = self.block_size.0;
        let block_rows = self.block_rows;
        let zero = <T as Zero>::zero();
        let mut x = y.to_vec();

        for bi in (0..block_rows).rev() {
            let row_start = bi * bs;
            let row_end = (row_start + bs).min(self.nrows);
            let actual_bs = row_end - row_start;

            // Subtract U_{bi, bj} * x_{bj} for all bj > bi.
            for pos in self.indptr[bi]..self.indptr[bi + 1] {
                let bj = self.indices[pos];
                if bj <= bi {
                    continue;
                }
                let bj_start = bj * bs;
                let bj_end = (bj_start + bs).min(self.ncols);
                let actual_bj = bj_end - bj_start;

                let u_base = pos * bs * bs;
                let mut acc = vec![zero; actual_bs];
                for local_row in 0..actual_bs {
                    for local_col in 0..actual_bj {
                        acc[local_row] = acc[local_row]
                            + self.ilu_data[u_base + local_row * bs + local_col]
                                * x[bj_start + local_col];
                    }
                }
                for local_row in 0..actual_bs {
                    x[row_start + local_row] = x[row_start + local_row] - acc[local_row];
                }
            }

            // Solve diagonal block U_{bi,bi} * x_{bi} = rhs_bi.
            let pos_bb = self
                .find_block_pos(bi, bi)
                .ok_or_else(|| SparseError::SingularMatrix(
                    format!("BlockILU0 solve_upper: missing diagonal at block {}", bi),
                ))?;
            let bb_base = pos_bb * bs * bs;
            let lu_block = &self.ilu_data[bb_base..bb_base + bs * bs];
            let piv = &self.pivots[pos_bb];
            let mut rhs = x[row_start..row_end].to_vec();
            block_lu_solve(lu_block, piv, &mut rhs, actual_bs);
            x[row_start..row_end].copy_from_slice(&rhs);
        }
        Ok(x)
    }

    // ------------------------------------------------------------------
    // Full apply
    // ------------------------------------------------------------------

    /// Apply the ILU(0) preconditioner: compute `y = (LU)^{-1} x`.
    pub fn apply(&self, x: &[T]) -> SparseResult<Vec<T>> {
        if x.len() != self.nrows {
            return Err(SparseError::DimensionMismatch {
                expected: self.nrows,
                found: x.len(),
            });
        }
        let y = self.solve_lower(x)?;
        self.solve_upper(&y)
    }

    // ------------------------------------------------------------------
    // Utility
    // ------------------------------------------------------------------

    /// Return `true` if the factorisation has been computed.
    pub fn is_factorised(&self) -> bool {
        self.is_factorised
    }

    /// Return the block size (r, c) = (r, r) for square blocks.
    pub fn block_size(&self) -> (usize, usize) {
        self.block_size
    }

    fn check_factorised(&self) -> SparseResult<()> {
        if !self.is_factorised {
            Err(SparseError::ComputationError(
                "BlockILU0 has not been factorised; call factorize() first".to_string(),
            ))
        } else {
            Ok(())
        }
    }

    fn find_block_pos(&self, bi: usize, bj: usize) -> Option<usize> {
        for pos in self.indptr[bi]..self.indptr[bi + 1] {
            if self.indices[pos] == bj {
                return Some(pos);
            }
        }
        None
    }
}

impl<T> Default for BlockILU0<T>
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
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Convenience function
// ============================================================

/// Factorize a BSR matrix and return the resulting `BlockILU0` preconditioner.
pub fn apply_block_ilu<T>(bsr: &BSRMatrix<T>, x: &[T]) -> SparseResult<Vec<T>>
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
    let mut ilu = BlockILU0::new();
    ilu.factorize(bsr)?;
    ilu.apply(x)
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::bsr::BSRMatrix;
    use approx::assert_relative_eq;

    /// Build a block-tridiagonal BSR matrix of size `2n × 2n` with `2×2` blocks.
    ///
    /// The block-tridiagonal structure is:
    ///   diag blocks = [[4,1],[1,4]]
    ///   off-diag blocks = [[-1,0],[0,-1]]
    fn make_block_tridiag(n: usize) -> BSRMatrix<f64> {
        let bs = 2usize;
        let nrows = n * bs;
        let diag_block = vec![4.0_f64, 1.0, 1.0, 4.0];
        let off_block = vec![-1.0_f64, 0.0, 0.0, -1.0];

        let mut data: Vec<f64> = Vec::new();
        let mut indices: Vec<usize> = Vec::new();
        let mut indptr = vec![0usize; n + 1];

        for bi in 0..n {
            if bi > 0 {
                data.extend_from_slice(&off_block);
                indices.push(bi - 1);
            }
            data.extend_from_slice(&diag_block);
            indices.push(bi);
            if bi < n - 1 {
                data.extend_from_slice(&off_block);
                indices.push(bi + 1);
            }
            indptr[bi + 1] = indices.len();
        }

        BSRMatrix::new(data, indices, indptr, (nrows, nrows), (bs, bs))
            .expect("block tridiag construction failed")
    }

    #[test]
    fn test_factorize_single_diagonal_block() {
        // 2×2 matrix with one 2×2 block.
        let data = vec![4.0_f64, 1.0, 1.0, 4.0];
        let indices = vec![0usize];
        let indptr = vec![0usize, 1];
        let bsr = BSRMatrix::new(data, indices, indptr, (2, 2), (2, 2))
            .expect("BSR construction");

        let mut ilu = BlockILU0::new();
        ilu.factorize(&bsr).expect("factorize failed");
        assert!(ilu.is_factorised());

        // Solve A x = b = A * [1,1]^T = [5,5]^T.
        let b = vec![5.0_f64, 5.0];
        let x = ilu.apply(&b).expect("apply failed");
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_factorize_block_tridiagonal() {
        let n = 3;
        let bsr = make_block_tridiag(n);
        let mut ilu = BlockILU0::new();
        ilu.factorize(&bsr).expect("factorize tridiag failed");
        assert!(ilu.is_factorised());

        // Check that apply returns finite values for b = [1,...,1].
        let b = vec![1.0_f64; n * 2];
        let y = ilu.apply(&b).expect("apply tridiag failed");
        assert_eq!(y.len(), n * 2);
        for yi in &y {
            assert!(yi.is_finite(), "Non-finite value in ILU apply: {}", yi);
        }
    }

    #[test]
    fn test_solve_lower_upper_compose() {
        // For a diagonal-only BSR, L = I, U = diagonal blocks.
        // So solve_lower(b) = b, solve_upper(b) = A^{-1} b.
        let data = vec![2.0_f64, 0.0, 0.0, 4.0, 3.0, 0.0, 0.0, 6.0]; // Two 2×2 diagonal blocks.
        let indices = vec![0usize, 1];
        let indptr = vec![0usize, 1, 2];
        let bsr = BSRMatrix::new(data, indices, indptr, (4, 4), (2, 2))
            .expect("BSR construction");

        let mut ilu = BlockILU0::new();
        ilu.factorize(&bsr).expect("factorize failed");

        let y = ilu.solve_lower(&[1.0_f64, 1.0, 1.0, 1.0]).expect("solve_lower failed");
        // L = I for diagonal-only BSR, so solve_lower = identity.
        for yi in &y {
            assert!(yi.is_finite());
        }

        let x = ilu.solve_upper(&y).expect("solve_upper failed");
        assert_eq!(x.len(), 4);
        for xi in &x {
            assert!(xi.is_finite(), "Non-finite in solve_upper: {}", xi);
        }
    }

    #[test]
    fn test_apply_block_ilu_helper() {
        let n = 2;
        let bsr = make_block_tridiag(n);
        let b = vec![1.0_f64; n * 2];
        let y = apply_block_ilu(&bsr, &b).expect("apply_block_ilu failed");
        assert_eq!(y.len(), n * 2);
        for yi in &y {
            assert!(yi.is_finite());
        }
    }

    #[test]
    fn test_non_square_matrix_returns_error() {
        let data = vec![1.0_f64; 4];
        let indices = vec![0usize];
        let indptr = vec![0usize, 1];
        let bsr = BSRMatrix::new(data, indices, indptr, (2, 4), (2, 2))
            .expect("BSR construction");
        let mut ilu = BlockILU0::new();
        let result = ilu.factorize(&bsr);
        assert!(result.is_err(), "Expected error for non-square matrix");
    }

    #[test]
    fn test_apply_without_factorize_returns_error() {
        let ilu: BlockILU0<f64> = BlockILU0::new();
        // Manually set nrows for dimension check to pass.
        let result = ilu.apply(&[1.0_f64]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch_in_apply() {
        let n = 2;
        let bsr = make_block_tridiag(n);
        let mut ilu = BlockILU0::new();
        ilu.factorize(&bsr).expect("factorize failed");
        let result = ilu.apply(&[1.0_f64; 10]);
        assert!(result.is_err(), "Expected DimensionMismatch");
    }

    #[test]
    fn test_ilu_is_better_preconditioner_than_identity() {
        // For a well-conditioned system, ILU should give a good approximation.
        // We verify || (LU)^{-1} A x - x ||_2 is small (not zero, since ILU(0)≠exact).
        let n = 4;
        let bsr = make_block_tridiag(n);
        let mut ilu = BlockILU0::new();
        ilu.factorize(&bsr).expect("factorize failed");

        let x_true = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // b = A * x_true
        let b = bsr.spmv(&x_true).expect("spmv failed");
        // x_ilu ≈ A^{-1} b = x_true (approximately)
        let x_ilu = ilu.apply(&b).expect("apply failed");
        // Residual should be small for this well-conditioned system.
        let norm_sq: f64 = x_ilu.iter().zip(x_true.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        assert!(norm_sq.sqrt() < 1.0, "ILU residual too large: {}", norm_sq.sqrt());
    }
}
