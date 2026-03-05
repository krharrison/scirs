//! Block Sparse Column (BSC) format
//!
//! BSC is the column-major counterpart of BSR: blocks are indexed by their
//! block-row within each block-column.  The layout is:
//!
//! ```text
//! data[k * r * c .. (k+1) * r * c]  ← k-th non-zero block (row-major within block)
//! indices[k]                          ← block-row index of k-th block
//! indptr[j] .. indptr[j+1]            ← range of non-zero blocks in block-column j
//! ```
//!
//! Key operations:
//! - `from_bsr()` — build BSC by transposing a BSR matrix.
//! - `spmv()` — SpMV y = A·x operating block-column wise.
//! - `to_dense()`, `get()`, `shape()`.

use crate::error::{SparseError, SparseResult};
use crate::formats::bsr::BSRMatrix;
use scirs2_core::numeric::{One, SparseElement, Zero};
use std::fmt::Debug;
use std::ops::{Add, Mul};

// ============================================================
// BSCMatrix
// ============================================================

/// Block Sparse Column matrix with flat block storage.
///
/// Each non-zero block is stored contiguously in `data` (row-major within the
/// block).  `block_size = (r, c)` means each block has `r` rows and `c`
/// columns.
#[derive(Debug, Clone)]
pub struct BSCMatrix<T> {
    /// Total number of matrix rows.
    pub nrows: usize,
    /// Total number of matrix columns.
    pub ncols: usize,
    /// Block dimensions (rows per block, cols per block).
    pub block_size: (usize, usize),
    /// Number of block-rows.
    pub block_rows: usize,
    /// Number of block-columns.
    pub block_cols: usize,
    /// Flat block storage: length = `nnz_blocks * r * c`.
    pub data: Vec<T>,
    /// Block-row indices: length = `nnz_blocks`.
    pub indices: Vec<usize>,
    /// Column pointer array: length = `block_cols + 1`.
    pub indptr: Vec<usize>,
}

impl<T> BSCMatrix<T>
where
    T: Clone + Copy + Zero + One + SparseElement + Debug + PartialEq,
{
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Create a BSCMatrix from pre-built flat data.
    ///
    /// # Arguments
    /// - `data`: flat block data, length must equal `indices.len() * r * c`.
    /// - `indices`: block-row indices.
    /// - `indptr`: column pointer, length `block_cols + 1`.
    /// - `shape`: `(nrows, ncols)` of the full matrix.
    /// - `block_size`: `(r, c)` block dimensions.
    pub fn new(
        data: Vec<T>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        shape: (usize, usize),
        block_size: (usize, usize),
    ) -> SparseResult<Self> {
        let (nrows, ncols) = shape;
        let (r, c) = block_size;
        if r == 0 || c == 0 {
            return Err(SparseError::ValueError(
                "BSC block dimensions must be positive".to_string(),
            ));
        }
        let block_rows = nrows.div_ceil(r);
        let block_cols = ncols.div_ceil(c);

        if indptr.len() != block_cols + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "indptr length {} does not match block_cols+1 {}",
                    indptr.len(),
                    block_cols + 1
                ),
            });
        }
        let nnz_blocks = indices.len();
        if data.len() != nnz_blocks * r * c {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "data length {} does not match nnz_blocks*r*c = {}*{}*{} = {}",
                    data.len(),
                    nnz_blocks,
                    r,
                    c,
                    nnz_blocks * r * c
                ),
            });
        }
        let last_ptr = *indptr.last().ok_or_else(|| SparseError::InconsistentData {
            reason: "indptr is empty".to_string(),
        })?;
        if last_ptr != nnz_blocks {
            return Err(SparseError::InconsistentData {
                reason: "indptr last element must equal nnz_blocks".to_string(),
            });
        }
        for bj in 0..block_cols {
            if indptr[bj] > indptr[bj + 1] {
                return Err(SparseError::InconsistentData {
                    reason: format!("indptr is not non-decreasing at position {}", bj),
                });
            }
        }
        for &bi in &indices {
            if bi >= block_rows {
                return Err(SparseError::IndexOutOfBounds {
                    index: (bi, 0),
                    shape: (block_rows, block_cols),
                });
            }
        }

        Ok(Self {
            nrows,
            ncols,
            block_size,
            block_rows,
            block_cols,
            data,
            indices,
            indptr,
        })
    }

    /// Create an empty BSCMatrix (all-zero) of the given shape and block size.
    pub fn zeros(shape: (usize, usize), block_size: (usize, usize)) -> SparseResult<Self> {
        let (_nrows, ncols) = shape;
        let (_r, c) = block_size;
        if _r == 0 || c == 0 {
            return Err(SparseError::ValueError(
                "BSC block dimensions must be positive".to_string(),
            ));
        }
        let block_cols = ncols.div_ceil(c);
        Self::new(vec![], vec![], vec![0usize; block_cols + 1], shape, block_size)
    }

    /// Convert a BSRMatrix to BSCMatrix (essentially a block-level transpose and re-index).
    ///
    /// The resulting BSCMatrix represents the same matrix but stored column-major.
    pub fn from_bsr(bsr: &BSRMatrix<T>) -> SparseResult<Self>
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        // BSC of A = transposed index structure.  We keep the same block data but
        // reorganise indptr/indices to be column-based rather than row-based.
        let nrows = bsr.nrows;
        let ncols = bsr.ncols;
        let (r, c) = bsr.block_size;
        let block_rows = bsr.block_rows;
        let block_cols = bsr.block_cols;
        let nnz_blocks = bsr.indices.len();

        // Count blocks per block-column.
        let mut col_counts = vec![0usize; block_cols];
        for &bj in &bsr.indices {
            col_counts[bj] += 1;
        }
        let mut bsc_indptr = vec![0usize; block_cols + 1];
        for j in 0..block_cols {
            bsc_indptr[j + 1] = bsc_indptr[j] + col_counts[j];
        }

        let mut bsc_indices = vec![0usize; nnz_blocks];
        let mut bsc_data = vec![<T as Zero>::zero(); nnz_blocks * r * c];
        let mut cur = bsc_indptr[..block_cols].to_vec();

        for bi in 0..block_rows {
            for pos in bsr.indptr[bi]..bsr.indptr[bi + 1] {
                let bj = bsr.indices[pos];
                let dst = cur[bj];
                cur[bj] += 1;
                bsc_indices[dst] = bi;
                let src_base = pos * r * c;
                let dst_base = dst * r * c;
                // Copy block verbatim (no transpose — BSC stores same orientation as BSR).
                bsc_data[dst_base..dst_base + r * c]
                    .copy_from_slice(&bsr.data[src_base..src_base + r * c]);
            }
        }

        Self::new(bsc_data, bsc_indices, bsc_indptr, (nrows, ncols), (r, c))
    }

    /// Build a BSCMatrix from a row-major dense slice.
    pub fn from_dense(
        dense: &[T],
        nrows: usize,
        ncols: usize,
        block_size: (usize, usize),
    ) -> SparseResult<Self> {
        if dense.len() != nrows * ncols {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "dense slice length {} does not match nrows*ncols = {}",
                    dense.len(),
                    nrows * ncols
                ),
            });
        }
        let (r, c) = block_size;
        if r == 0 || c == 0 {
            return Err(SparseError::ValueError(
                "Block dimensions must be positive".to_string(),
            ));
        }
        let block_rows = nrows.div_ceil(r);
        let block_cols = ncols.div_ceil(c);
        let zero = <T as Zero>::zero();

        let mut data: Vec<T> = Vec::new();
        let mut indices: Vec<usize> = Vec::new();
        let mut indptr = vec![0usize; block_cols + 1];

        // Iterate column-first.
        for bj in 0..block_cols {
            let col_start = bj * c;
            let col_end = col_start + c;
            for bi in 0..block_rows {
                let row_start = bi * r;
                let row_end = row_start + r;
                let mut block = Vec::with_capacity(r * c);
                let mut all_zero = true;
                for row in row_start..row_end {
                    for col in col_start..col_end {
                        let val = if row < nrows && col < ncols {
                            dense[row * ncols + col]
                        } else {
                            zero
                        };
                        if val != zero {
                            all_zero = false;
                        }
                        block.push(val);
                    }
                }
                if !all_zero {
                    data.extend_from_slice(&block);
                    indices.push(bi);
                }
            }
            indptr[bj + 1] = indices.len();
        }

        Self::new(data, indices, indptr, (nrows, ncols), block_size)
    }

    // ------------------------------------------------------------------
    // Conversion
    // ------------------------------------------------------------------

    /// Convert the BSCMatrix to a row-major dense vector (nrows × ncols).
    pub fn to_dense(&self) -> Vec<T> {
        let zero = <T as Zero>::zero();
        let mut dense = vec![zero; self.nrows * self.ncols];
        let (r, c) = self.block_size;

        for bj in 0..self.block_cols {
            let col_start = bj * c;
            for pos in self.indptr[bj]..self.indptr[bj + 1] {
                let bi = self.indices[pos];
                let row_start = bi * r;
                let base = pos * r * c;
                for local_row in 0..r {
                    let matrix_row = row_start + local_row;
                    if matrix_row >= self.nrows {
                        break;
                    }
                    for local_col in 0..c {
                        let matrix_col = col_start + local_col;
                        if matrix_col >= self.ncols {
                            break;
                        }
                        dense[matrix_row * self.ncols + matrix_col] =
                            self.data[base + local_row * c + local_col];
                    }
                }
            }
        }
        dense
    }

    /// Convert to a BSRMatrix (re-index as row-based).
    pub fn to_bsr(&self) -> SparseResult<BSRMatrix<T>>
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        // BSR of A = BSC with row/col swapped index.
        let (r, c) = self.block_size;
        let nnz_blocks = self.indices.len();

        // Count blocks per block-row.
        let mut row_counts = vec![0usize; self.block_rows];
        for &bi in &self.indices {
            row_counts[bi] += 1;
        }
        let mut bsr_indptr = vec![0usize; self.block_rows + 1];
        for i in 0..self.block_rows {
            bsr_indptr[i + 1] = bsr_indptr[i] + row_counts[i];
        }

        let mut bsr_indices = vec![0usize; nnz_blocks];
        let mut bsr_data = vec![<T as Zero>::zero(); nnz_blocks * r * c];
        let mut cur = bsr_indptr[..self.block_rows].to_vec();

        for bj in 0..self.block_cols {
            for pos in self.indptr[bj]..self.indptr[bj + 1] {
                let bi = self.indices[pos];
                let dst = cur[bi];
                cur[bi] += 1;
                bsr_indices[dst] = bj;
                let src_base = pos * r * c;
                let dst_base = dst * r * c;
                bsr_data[dst_base..dst_base + r * c]
                    .copy_from_slice(&self.data[src_base..src_base + r * c]);
            }
        }

        BSRMatrix::new(
            bsr_data,
            bsr_indices,
            bsr_indptr,
            (self.nrows, self.ncols),
            self.block_size,
        )
    }

    // ------------------------------------------------------------------
    // SpMV
    // ------------------------------------------------------------------

    /// Sparse matrix-vector product: y = A * x.
    ///
    /// Iterates over block-columns and scatters partial sums back to y.
    pub fn spmv(&self, x: &[T]) -> SparseResult<Vec<T>>
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        if x.len() != self.ncols {
            return Err(SparseError::DimensionMismatch {
                expected: self.ncols,
                found: x.len(),
            });
        }
        let zero = <T as Zero>::zero();
        let mut y = vec![zero; self.nrows];
        let (r, c) = self.block_size;

        for bj in 0..self.block_cols {
            let col_start = bj * c;
            let col_end = (col_start + c).min(self.ncols);

            for pos in self.indptr[bj]..self.indptr[bj + 1] {
                let bi = self.indices[pos];
                let row_start = bi * r;
                let row_end = (row_start + r).min(self.nrows);
                let base = pos * r * c;

                for local_row in 0..(row_end - row_start) {
                    let mut acc = zero;
                    for local_col in 0..(col_end - col_start) {
                        acc = acc
                            + self.data[base + local_row * c + local_col]
                                * x[col_start + local_col];
                    }
                    y[row_start + local_row] = y[row_start + local_row] + acc;
                }
            }
        }
        Ok(y)
    }

    // ------------------------------------------------------------------
    // Transpose → BSR
    // ------------------------------------------------------------------

    /// Compute the transpose as a BSRMatrix with swapped block_size.
    pub fn transpose_to_bsr(&self) -> SparseResult<BSRMatrix<T>>
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        // Convert self → BSR first, then transpose.
        let bsr = self.to_bsr()?;
        bsr.transpose()
    }

    // ------------------------------------------------------------------
    // Utility
    // ------------------------------------------------------------------

    /// Return the number of non-zero blocks.
    pub fn nnz_blocks(&self) -> usize {
        self.indices.len()
    }

    /// Return the total number of stored scalar values.
    pub fn nnz(&self) -> usize {
        let (r, c) = self.block_size;
        self.indices.len() * r * c
    }

    /// Return (nrows, ncols).
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Get the scalar value at position (row, col).
    pub fn get(&self, row: usize, col: usize) -> T {
        if row >= self.nrows || col >= self.ncols {
            return <T as Zero>::zero();
        }
        let (r, c) = self.block_size;
        let bi = row / r;
        let bj = col / c;
        let local_row = row % r;
        let local_col = col % c;

        for pos in self.indptr[bj]..self.indptr[bj + 1] {
            if self.indices[pos] == bi {
                let base = pos * r * c;
                return self.data[base + local_row * c + local_col];
            }
        }
        <T as Zero>::zero()
    }
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::bsr::BSRMatrix;
    use approx::assert_relative_eq;

    fn make_4x4_bsr() -> BSRMatrix<f64> {
        let data = vec![
            1.0_f64, 2.0, 3.0, 4.0, // block (0,0)
            5.0, 6.0, 7.0, 8.0,     // block (1,1)
        ];
        let indices = vec![0, 1];
        let indptr = vec![0, 1, 2];
        BSRMatrix::new(data, indices, indptr, (4, 4), (2, 2)).expect("BSR construction failed")
    }

    #[test]
    fn test_from_bsr_roundtrip() {
        let bsr = make_4x4_bsr();
        let bsc = BSCMatrix::from_bsr(&bsr).expect("from_bsr failed");
        let bsr2 = bsc.to_bsr().expect("to_bsr failed");

        let dense_bsr = bsr.to_dense();
        let dense_bsr2 = bsr2.to_dense();
        for (a, b) in dense_bsr.iter().zip(dense_bsr2.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_from_dense() {
        let dense = vec![
            1.0_f64, 2.0, 0.0, 0.0,
            3.0, 4.0, 0.0, 0.0,
            0.0, 0.0, 5.0, 6.0,
            0.0, 0.0, 7.0, 8.0,
        ];
        let bsc = BSCMatrix::from_dense(&dense, 4, 4, (2, 2)).expect("from_dense failed");
        assert_eq!(bsc.nnz_blocks(), 2);
        assert_eq!(bsc.get(0, 0), 1.0);
        assert_eq!(bsc.get(3, 3), 8.0);
        assert_eq!(bsc.get(0, 2), 0.0);
    }

    #[test]
    fn test_spmv_matches_bsr() {
        let bsr = make_4x4_bsr();
        let bsc = BSCMatrix::from_bsr(&bsr).expect("from_bsr failed");
        let x = vec![1.0_f64, 2.0, 3.0, 4.0];
        let y_bsr = bsr.spmv(&x).expect("bsr spmv failed");
        let y_bsc = bsc.spmv(&x).expect("bsc spmv failed");
        for (a, b) in y_bsr.iter().zip(y_bsc.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_to_dense_consistent() {
        let dense_orig = vec![
            1.0_f64, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let bsc = BSCMatrix::from_dense(&dense_orig, 4, 4, (2, 2)).expect("from_dense failed");
        let recovered = bsc.to_dense();
        for (a, b) in recovered.iter().zip(dense_orig.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_shape_and_nnz() {
        let bsr = make_4x4_bsr();
        let bsc = BSCMatrix::from_bsr(&bsr).expect("from_bsr failed");
        assert_eq!(bsc.shape(), (4, 4));
        assert_eq!(bsc.nnz_blocks(), 2);
        assert_eq!(bsc.nnz(), 8);
    }

    #[test]
    fn test_get_consistency_with_to_dense() {
        let bsr = make_4x4_bsr();
        let bsc = BSCMatrix::from_bsr(&bsr).expect("from_bsr failed");
        let dense = bsc.to_dense();
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(bsc.get(i, j), dense[i * 4 + j], epsilon = 1e-12);
            }
        }
    }
}
