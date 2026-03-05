//! Block Sparse Row (BSR) format
//!
//! This module provides a flat-flattened-block BSR implementation where the
//! block data is stored as a contiguous `Vec<T>` with stride `r*c` per block,
//! enabling efficient BLAS-style SpMV without any intermediate allocation.
//!
//! # Layout
//!
//! ```text
//! data[k * r * c .. (k+1) * r * c]  ← k-th non-zero block (row-major within block)
//! indices[k]                          ← block-column index of k-th block
//! indptr[i] .. indptr[i+1]            ← range of non-zero blocks in block-row i
//! ```
//!
//! The full matrix has `block_rows = ceil(nrows/r)` block-rows and
//! `block_cols = ceil(ncols/c)` block-columns.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::{One, SparseElement, Zero};
use std::fmt::Debug;
use std::ops::{Add, Mul, Neg, Sub};

// ============================================================
// BSRMatrix
// ============================================================

/// Block Sparse Row matrix with flat block storage.
///
/// Each non-zero block is stored contiguously in `data` (row-major within the
/// block).  `block_size = (r, c)` means each block has `r` rows and `c`
/// columns.
#[derive(Debug, Clone)]
pub struct BSRMatrix<T> {
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
    /// Block-column indices: length = `nnz_blocks`.
    pub indices: Vec<usize>,
    /// Row pointer array: length = `block_rows + 1`.
    pub indptr: Vec<usize>,
}

impl<T> BSRMatrix<T>
where
    T: Clone + Copy + Zero + One + SparseElement + Debug + PartialEq,
{
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Create a BSRMatrix from pre-built flat data.
    ///
    /// # Arguments
    /// - `data`: flat block data, length must equal `indices.len() * r * c`.
    /// - `indices`: block-column indices.
    /// - `indptr`: row pointer, length `block_rows + 1`.
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
                "BSR block dimensions must be positive".to_string(),
            ));
        }
        let block_rows = nrows.div_ceil(r);
        let block_cols = ncols.div_ceil(c);

        if indptr.len() != block_rows + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "indptr length {} does not match block_rows+1 {}",
                    indptr.len(),
                    block_rows + 1
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
        // Validate indptr is non-decreasing and in range.
        if *indptr.last().ok_or_else(|| {
            SparseError::InconsistentData {
                reason: "indptr is empty".to_string(),
            }
        })? != nnz_blocks
        {
            return Err(SparseError::InconsistentData {
                reason: "indptr last element must equal nnz_blocks".to_string(),
            });
        }
        for bi in 0..block_rows {
            if indptr[bi] > indptr[bi + 1] {
                return Err(SparseError::InconsistentData {
                    reason: format!(
                        "indptr is not non-decreasing at position {}",
                        bi
                    ),
                });
            }
        }
        for &bc in &indices {
            if bc >= block_cols {
                return Err(SparseError::IndexOutOfBounds {
                    index: (0, bc),
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

    /// Create an empty BSRMatrix (all-zero) of the given shape and block size.
    pub fn zeros(shape: (usize, usize), block_size: (usize, usize)) -> SparseResult<Self> {
        let (nrows, _ncols) = shape;
        let (r, c) = block_size;
        if r == 0 || c == 0 {
            return Err(SparseError::ValueError(
                "BSR block dimensions must be positive".to_string(),
            ));
        }
        let block_rows = nrows.div_ceil(r);
        Self::new(vec![], vec![], vec![0usize; block_rows + 1], shape, block_size)
    }

    /// Build a BSRMatrix from a row-major dense matrix.
    ///
    /// Blocks whose all entries are zero are omitted.
    pub fn from_dense(dense: &[T], nrows: usize, ncols: usize, block_size: (usize, usize)) -> SparseResult<Self>
    where
        T: PartialEq + Zero,
    {
        if dense.len() != nrows * ncols {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "dense slice length {} does not match nrows*ncols = {}*{} = {}",
                    dense.len(), nrows, ncols, nrows * ncols
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

        let mut data: Vec<T> = Vec::new();
        let mut indices: Vec<usize> = Vec::new();
        let mut indptr: Vec<usize> = vec![0usize; block_rows + 1];

        let zero = <T as Zero>::zero();

        for bi in 0..block_rows {
            let row_start = bi * r;
            let row_end = row_start + r;

            for bj in 0..block_cols {
                let col_start = bj * c;
                let col_end = col_start + c;

                // Extract block and check if non-zero.
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
                    indices.push(bj);
                }
            }
            indptr[bi + 1] = indices.len();
        }

        Self::new(data, indices, indptr, (nrows, ncols), block_size)
    }

    /// Build a BSRMatrix from a CsrMatrix with the given block size.
    ///
    /// The block size must exactly divide the matrix dimensions (padding rows/cols
    /// with zero is handled automatically when neither divides evenly).
    pub fn from_csr(csr: &CsrMatrix<T>, block_size: (usize, usize)) -> SparseResult<Self>
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        let (nrows, ncols) = csr.shape();
        let (r, c) = block_size;
        if r == 0 || c == 0 {
            return Err(SparseError::ValueError(
                "Block dimensions must be positive".to_string(),
            ));
        }
        let block_rows = nrows.div_ceil(r);
        let block_cols = ncols.div_ceil(c);

        // We collect (bi, bj, flat_offset, value) then fold into blocks.
        // Use a temporary 2-D structure: Vec<Vec<Option<Vec<T>>>>.
        // For memory efficiency, collect (bi, bj) → block accumulator.

        // Build a map from (bi, bj) to flat block buffer.
        // Since we iterate CSR in row order, group by bi.
        let zero = <T as Zero>::zero();

        // temporary storage: for each block_row, a HashMap-like map from bj → block_buf.
        // We use a Vec<(bj, Vec<T>)> per block-row and sort/dedup at the end.
        // For simplicity, use a flat map over usize (block_cols small enough).

        struct BlockAccum<U> {
            blocks: Vec<Option<Vec<U>>>, // indexed by bj, length = block_cols
        }
        impl<U: Copy + Clone + Zero> BlockAccum<U> {
            fn new(bc: usize, r: usize, c: usize) -> Self {
                Self {
                    blocks: vec![None; bc],
                }
            }
            fn accumulate(&mut self, bj: usize, local_row: usize, local_col: usize, val: U, r: usize, c: usize) {
                if self.blocks[bj].is_none() {
                    self.blocks[bj] = Some(vec![U::zero(); r * c]);
                }
                if let Some(buf) = &mut self.blocks[bj] {
                    buf[local_row * c + local_col] = val;
                }
            }
        }

        let mut data: Vec<T> = Vec::new();
        let mut indices: Vec<usize> = Vec::new();
        let mut indptr: Vec<usize> = vec![0usize; block_rows + 1];

        // Walk CSR data block-row by block-row.
        for bi in 0..block_rows {
            let mut accum: BlockAccum<T> = BlockAccum::new(block_cols, r, c);

            let row_start = bi * r;
            let row_end = (row_start + r).min(nrows);

            for matrix_row in row_start..row_end {
                let local_row = matrix_row - row_start;
                for pos in csr.indptr[matrix_row]..csr.indptr[matrix_row + 1] {
                    let col = csr.indices[pos];
                    let val = csr.data[pos];
                    let bj = col / c;
                    let local_col = col % c;
                    accum.accumulate(bj, local_row, local_col, val, r, c);
                }
            }

            // Emit non-zero blocks for this block-row (in column order).
            for (bj, maybe_block) in accum.blocks.into_iter().enumerate() {
                if let Some(block) = maybe_block {
                    // Check all-zero (shouldn't happen but guard).
                    let non_zero = block.iter().any(|&v| v != zero);
                    if non_zero {
                        data.extend_from_slice(&block);
                        indices.push(bj);
                    }
                }
            }
            indptr[bi + 1] = indices.len();
        }

        Self::new(data, indices, indptr, (nrows, ncols), block_size)
    }

    // ------------------------------------------------------------------
    // Conversion
    // ------------------------------------------------------------------

    /// Convert the BSRMatrix to a row-major dense vector (nrows × ncols).
    pub fn to_dense(&self) -> Vec<T> {
        let zero = <T as Zero>::zero();
        let mut dense = vec![zero; self.nrows * self.ncols];
        let (r, c) = self.block_size;

        for bi in 0..self.block_rows {
            for pos in self.indptr[bi]..self.indptr[bi + 1] {
                let bj = self.indices[pos];
                let base = pos * r * c;
                let row_start = bi * r;
                let col_start = bj * c;
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

    /// Convert the BSRMatrix to a CsrMatrix.
    pub fn to_csr(&self) -> SparseResult<CsrMatrix<T>>
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        let dense = self.to_dense();
        // Build CSR from dense.
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        let zero = <T as Zero>::zero();
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                let v = dense[i * self.ncols + j];
                if v != zero {
                    row_indices.push(i);
                    col_indices.push(j);
                    values.push(v);
                }
            }
        }
        CsrMatrix::new(values, row_indices, col_indices, (self.nrows, self.ncols))
    }

    // ------------------------------------------------------------------
    // SpMV
    // ------------------------------------------------------------------

    /// Sparse matrix-vector product: y = A * x.
    ///
    /// Iterates over block-rows and applies a small dense GEMV per non-zero block.
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

        for bi in 0..self.block_rows {
            let row_start = bi * r;
            let row_end = (row_start + r).min(self.nrows);

            for pos in self.indptr[bi]..self.indptr[bi + 1] {
                let bj = self.indices[pos];
                let col_start = bj * c;
                let col_end = (col_start + c).min(self.ncols);
                let base = pos * r * c;

                for local_row in 0..(row_end - row_start) {
                    let mut acc = zero;
                    for local_col in 0..(col_end - col_start) {
                        acc = acc + self.data[base + local_row * c + local_col] * x[col_start + local_col];
                    }
                    y[row_start + local_row] = y[row_start + local_row] + acc;
                }
            }
        }
        Ok(y)
    }

    // ------------------------------------------------------------------
    // Transpose
    // ------------------------------------------------------------------

    /// Compute the transpose of this BSRMatrix (returns a new BSRMatrix).
    pub fn transpose(&self) -> SparseResult<BSRMatrix<T>>
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        let (r, c) = self.block_size;
        // Transposed matrix has block_size (c, r), shape (ncols, nrows).
        let t_nrows = self.ncols;
        let t_ncols = self.nrows;
        let t_block_size = (c, r);
        let t_block_rows = t_nrows.div_ceil(c);
        let t_block_cols = t_ncols.div_ceil(r);
        let nnz_blocks = self.indices.len();

        // Count blocks per transposed block-row (= original block-col).
        let mut t_indptr = vec![0usize; t_block_rows + 1];
        for &bj in &self.indices {
            t_indptr[bj + 1] += 1;
        }
        for i in 0..t_block_rows {
            t_indptr[i + 1] += t_indptr[i];
        }

        let mut t_indices = vec![0usize; nnz_blocks];
        let mut t_data = vec![<T as Zero>::zero(); nnz_blocks * c * r];
        let mut cur = t_indptr[..t_block_rows].to_vec();

        for bi in 0..self.block_rows {
            for pos in self.indptr[bi]..self.indptr[bi + 1] {
                let bj = self.indices[pos];
                let dst = cur[bj];
                cur[bj] += 1;
                t_indices[dst] = bi;
                let src_base = pos * r * c;
                let dst_base = dst * c * r;
                // Transpose the block: src[lr*c + lc] → dst[lc*r + lr]
                for lr in 0..r {
                    for lc in 0..c {
                        t_data[dst_base + lc * r + lr] = self.data[src_base + lr * c + lc];
                    }
                }
            }
        }
        let _ = t_block_cols; // suppress lint
        BSRMatrix::new(t_data, t_indices, t_indptr, (t_nrows, t_ncols), t_block_size)
    }

    // ------------------------------------------------------------------
    // Arithmetic
    // ------------------------------------------------------------------

    /// Element-wise addition of two BSRMatrices with the same shape and block size.
    pub fn add(&self, other: &BSRMatrix<T>) -> SparseResult<BSRMatrix<T>>
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        self.elementwise_op(other, |a, b| a + b, "add")
    }

    /// Element-wise subtraction: `self - other`.
    pub fn sub(&self, other: &BSRMatrix<T>) -> SparseResult<BSRMatrix<T>>
    where
        T: Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
    {
        self.elementwise_op(other, |a, b| a - b, "sub")
    }

    /// Block-level matrix multiplication: `self * other`.
    ///
    /// Requires `self.ncols == other.nrows` and compatible block sizes
    /// (`self.block_size.1 == other.block_size.0`).
    pub fn multiply_bsr(&self, other: &BSRMatrix<T>) -> SparseResult<BSRMatrix<T>>
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        if self.ncols != other.nrows {
            return Err(SparseError::DimensionMismatch {
                expected: self.ncols,
                found: other.nrows,
            });
        }
        let (r, k) = self.block_size;
        let (k2, c) = other.block_size;
        if k != k2 {
            return Err(SparseError::ValueError(format!(
                "Incompatible block sizes for multiplication: self block_cols {} != other block_rows {}",
                k, k2
            )));
        }

        let out_nrows = self.nrows;
        let out_ncols = other.ncols;
        let out_block_size = (r, c);
        let out_block_rows = out_nrows.div_ceil(r);
        let out_block_cols = out_ncols.div_ceil(c);
        let zero = <T as Zero>::zero();

        // Temporary accumulators: out_block_rows × out_block_cols optional blocks.
        let mut accum: Vec<Vec<Option<Vec<T>>>> = (0..out_block_rows)
            .map(|_| vec![None; out_block_cols])
            .collect();

        // Build a column-indexed view of `other` for fast lookup:
        // other_col_view[bj] = list of (bi_other, block_pos_in_other)
        let other_block_rows = other.block_rows;
        let mut other_by_col: Vec<Vec<(usize, usize)>> = vec![Vec::new(); other.block_cols];
        for bi_other in 0..other_block_rows {
            for pos in other.indptr[bi_other]..other.indptr[bi_other + 1] {
                let bj_other = other.indices[pos];
                other_by_col[bj_other].push((bi_other, pos));
            }
        }

        for bi in 0..self.block_rows {
            for pos_a in self.indptr[bi]..self.indptr[bi + 1] {
                let bk = self.indices[pos_a]; // block-col of A = block-row of B
                let base_a = pos_a * r * k;

                // Look up blocks in B that are in block-row bk.
                for pos_b in other.indptr[bk]..other.indptr[bk + 1] {
                    let bj = other.indices[pos_b];
                    let base_b = pos_b * k * c;

                    // Ensure accumulator block exists.
                    if accum[bi][bj].is_none() {
                        accum[bi][bj] = Some(vec![zero; r * c]);
                    }
                    let buf = accum[bi][bj].as_mut().expect("just initialised");

                    // Block multiply: buf[lr, lc] += A_block[lr, lk] * B_block[lk, lc]
                    for lr in 0..r {
                        for lk in 0..k {
                            let a_val = self.data[base_a + lr * k + lk];
                            if a_val == zero {
                                continue;
                            }
                            for lc in 0..c {
                                buf[lr * c + lc] = buf[lr * c + lc]
                                    + a_val * other.data[base_b + lk * c + lc];
                            }
                        }
                    }
                }
            }
        }

        // Flatten accumulators into BSR format.
        let mut out_data: Vec<T> = Vec::new();
        let mut out_indices: Vec<usize> = Vec::new();
        let mut out_indptr = vec![0usize; out_block_rows + 1];

        for bi in 0..out_block_rows {
            for bj in 0..out_block_cols {
                if let Some(block) = &accum[bi][bj] {
                    let non_zero = block.iter().any(|&v| v != zero);
                    if non_zero {
                        out_data.extend_from_slice(block);
                        out_indices.push(bj);
                    }
                }
            }
            out_indptr[bi + 1] = out_indices.len();
        }
        let _ = other_by_col; // suppress lint

        BSRMatrix::new(out_data, out_indices, out_indptr, (out_nrows, out_ncols), out_block_size)
    }

    // ------------------------------------------------------------------
    // Utility
    // ------------------------------------------------------------------

    /// Return the number of non-zero blocks.
    pub fn nnz_blocks(&self) -> usize {
        self.indices.len()
    }

    /// Return the total number of stored non-zero scalar values.
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

        for pos in self.indptr[bi]..self.indptr[bi + 1] {
            if self.indices[pos] == bj {
                let base = pos * r * c;
                return self.data[base + local_row * c + local_col];
            }
        }
        <T as Zero>::zero()
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Helper for element-wise binary operations on matching-structure BSR matrices.
    fn elementwise_op<F>(&self, other: &BSRMatrix<T>, op: F, _op_name: &str) -> SparseResult<BSRMatrix<T>>
    where
        F: Fn(T, T) -> T,
        T: Add<Output = T> + Mul<Output = T>,
    {
        if self.shape() != other.shape() {
            return Err(SparseError::ShapeMismatch {
                expected: self.shape(),
                found: other.shape(),
            });
        }
        if self.block_size != other.block_size {
            return Err(SparseError::ValueError(format!(
                "Block sizes differ: {:?} vs {:?}",
                self.block_size, other.block_size
            )));
        }
        let (r, c) = self.block_size;
        let zero = <T as Zero>::zero();

        // Merge: iterate block rows; for each row do sorted merge of indices.
        let mut out_data: Vec<T> = Vec::new();
        let mut out_indices: Vec<usize> = Vec::new();
        let mut out_indptr = vec![0usize; self.block_rows + 1];

        for bi in 0..self.block_rows {
            let a_start = self.indptr[bi];
            let a_end = self.indptr[bi + 1];
            let b_start = other.indptr[bi];
            let b_end = other.indptr[bi + 1];

            let mut ai = a_start;
            let mut bi_idx = b_start;

            while ai < a_end || bi_idx < b_end {
                let a_col = if ai < a_end { self.indices[ai] } else { usize::MAX };
                let b_col = if bi_idx < b_end { other.indices[bi_idx] } else { usize::MAX };

                if a_col < b_col {
                    // Only in A; apply op with zero.
                    let base = ai * r * c;
                    let mut block = vec![zero; r * c];
                    for k in 0..r * c {
                        block[k] = op(self.data[base + k], zero);
                    }
                    let non_zero = block.iter().any(|&v| v != zero);
                    if non_zero {
                        out_data.extend_from_slice(&block);
                        out_indices.push(a_col);
                    }
                    ai += 1;
                } else if b_col < a_col {
                    // Only in B; apply op with zero (A side).
                    let base = bi_idx * r * c;
                    let mut block = vec![zero; r * c];
                    for k in 0..r * c {
                        block[k] = op(zero, other.data[base + k]);
                    }
                    let non_zero = block.iter().any(|&v| v != zero);
                    if non_zero {
                        out_data.extend_from_slice(&block);
                        out_indices.push(b_col);
                    }
                    bi_idx += 1;
                } else {
                    // Same block column — combine.
                    let base_a = ai * r * c;
                    let base_b = bi_idx * r * c;
                    let mut block = vec![zero; r * c];
                    for k in 0..r * c {
                        block[k] = op(self.data[base_a + k], other.data[base_b + k]);
                    }
                    let non_zero = block.iter().any(|&v| v != zero);
                    if non_zero {
                        out_data.extend_from_slice(&block);
                        out_indices.push(a_col);
                    }
                    ai += 1;
                    bi_idx += 1;
                }
            }
            out_indptr[bi + 1] = out_indices.len();
        }

        BSRMatrix::new(out_data, out_indices, out_indptr, self.shape(), self.block_size)
    }
}

// ============================================================
// Scale by scalar
// ============================================================

impl<T> BSRMatrix<T>
where
    T: Clone + Copy + Zero + One + SparseElement + Debug + PartialEq + Mul<Output = T>,
{
    /// Multiply all entries by a scalar.
    pub fn scale(&self, alpha: T) -> BSRMatrix<T> {
        BSRMatrix {
            nrows: self.nrows,
            ncols: self.ncols,
            block_size: self.block_size,
            block_rows: self.block_rows,
            block_cols: self.block_cols,
            data: self.data.iter().map(|&v| v * alpha).collect(),
            indices: self.indices.clone(),
            indptr: self.indptr.clone(),
        }
    }
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_4x4_bsr() -> BSRMatrix<f64> {
        // 4×4 matrix with 2×2 blocks:
        // [ 1 2 | 0 0 ]
        // [ 3 4 | 0 0 ]
        // [ 0 0 | 5 6 ]
        // [ 0 0 | 7 8 ]
        let data = vec![
            1.0_f64, 2.0, 3.0, 4.0, // block (0,0)
            5.0, 6.0, 7.0, 8.0,     // block (1,1)
        ];
        let indices = vec![0, 1];
        let indptr = vec![0, 1, 2];
        BSRMatrix::new(data, indices, indptr, (4, 4), (2, 2)).expect("construction failed")
    }

    #[test]
    fn test_from_dense_to_dense_roundtrip() {
        let bsr = make_4x4_bsr();
        let dense = bsr.to_dense();
        let expected = vec![
            1.0, 2.0, 0.0, 0.0,
            3.0, 4.0, 0.0, 0.0,
            0.0, 0.0, 5.0, 6.0,
            0.0, 0.0, 7.0, 8.0,
        ];
        for (a, b) in dense.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_from_dense_constructor() {
        let dense = vec![
            1.0_f64, 2.0, 0.0, 0.0,
            3.0, 4.0, 0.0, 0.0,
            0.0, 0.0, 5.0, 6.0,
            0.0, 0.0, 7.0, 8.0,
        ];
        let bsr = BSRMatrix::from_dense(&dense, 4, 4, (2, 2)).expect("from_dense failed");
        assert_eq!(bsr.nnz_blocks(), 2);
        assert_eq!(bsr.get(0, 0), 1.0);
        assert_eq!(bsr.get(2, 2), 5.0);
        assert_eq!(bsr.get(0, 2), 0.0);
    }

    #[test]
    fn test_spmv() {
        let bsr = make_4x4_bsr();
        let x = vec![1.0_f64, 1.0, 1.0, 1.0];
        let y = bsr.spmv(&x).expect("spmv failed");
        // Row 0: 1+2 = 3, row 1: 3+4 = 7, row 2: 5+6 = 11, row 3: 7+8 = 15
        assert_relative_eq!(y[0], 3.0, epsilon = 1e-12);
        assert_relative_eq!(y[1], 7.0, epsilon = 1e-12);
        assert_relative_eq!(y[2], 11.0, epsilon = 1e-12);
        assert_relative_eq!(y[3], 15.0, epsilon = 1e-12);
    }

    #[test]
    fn test_transpose() {
        let bsr = make_4x4_bsr();
        let bsrt = bsr.transpose().expect("transpose failed");
        let dense_t = bsrt.to_dense();
        // Transposed: T[i,j] = original[j,i]
        let orig = bsr.to_dense();
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(dense_t[i * 4 + j], orig[j * 4 + i], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_add() {
        let bsr = make_4x4_bsr();
        let result = bsr.add(&bsr).expect("add failed");
        let dense = result.to_dense();
        let orig = bsr.to_dense();
        for (a, b) in dense.iter().zip(orig.iter()) {
            assert_relative_eq!(a, &(b * 2.0), epsilon = 1e-12);
        }
    }

    #[test]
    fn test_multiply_bsr() {
        let bsr = make_4x4_bsr();
        let result = bsr.multiply_bsr(&bsr).expect("multiply_bsr failed");
        let dense_r = result.to_dense();
        // Manual: A^2 for block-diagonal matrix → each block squared
        // Block (0,0) = [[1,2],[3,4]]^2 = [[7,10],[15,22]]
        // Block (1,1) = [[5,6],[7,8]]^2 = [[67,78],[83,96]] (wait recalc)
        // Actually [[5,6],[7,8]] * [[5,6],[7,8]] = [[5*5+6*7, 5*6+6*8],[7*5+8*7, 7*6+8*8]]
        //  = [[25+42, 30+48],[35+56, 42+64]] = [[67,78],[91,106]]
        assert_relative_eq!(dense_r[0 * 4 + 0], 7.0, epsilon = 1e-12);
        assert_relative_eq!(dense_r[0 * 4 + 1], 10.0, epsilon = 1e-12);
        assert_relative_eq!(dense_r[1 * 4 + 0], 15.0, epsilon = 1e-12);
        assert_relative_eq!(dense_r[1 * 4 + 1], 22.0, epsilon = 1e-12);
        assert_relative_eq!(dense_r[2 * 4 + 2], 67.0, epsilon = 1e-12);
        assert_relative_eq!(dense_r[2 * 4 + 3], 78.0, epsilon = 1e-12);
        assert_relative_eq!(dense_r[3 * 4 + 2], 91.0, epsilon = 1e-12);
        assert_relative_eq!(dense_r[3 * 4 + 3], 106.0, epsilon = 1e-12);
    }

    #[test]
    fn test_from_csr() {
        use crate::csr::CsrMatrix;
        let rows = vec![0usize, 0, 1, 1, 2, 2, 3, 3];
        let cols = vec![0usize, 1, 0, 1, 2, 3, 2, 3];
        let vals = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let csr = CsrMatrix::new(vals, rows, cols, (4, 4)).expect("csr failed");
        let bsr = BSRMatrix::from_csr(&csr, (2, 2)).expect("from_csr failed");
        assert_eq!(bsr.nnz_blocks(), 2);
        assert_eq!(bsr.get(0, 0), 1.0);
        assert_eq!(bsr.get(1, 1), 4.0);
        assert_eq!(bsr.get(2, 3), 6.0);
    }

    #[test]
    fn test_get_out_of_bounds_returns_zero() {
        let bsr = make_4x4_bsr();
        assert_eq!(bsr.get(10, 10), 0.0);
        assert_eq!(bsr.get(0, 3), 0.0);
    }

    #[test]
    fn test_non_square_blocks() {
        // 4×6 matrix with 2×3 blocks
        let dense = vec![
            1.0_f64, 2.0, 3.0, 0.0, 0.0, 0.0,
            4.0, 5.0, 6.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 7.0, 8.0, 9.0,
            0.0, 0.0, 0.0, 10.0, 11.0, 12.0,
        ];
        let bsr = BSRMatrix::from_dense(&dense, 4, 6, (2, 3)).expect("from_dense non-square");
        assert_eq!(bsr.nnz_blocks(), 2);
        let x = vec![1.0_f64; 6];
        let y = bsr.spmv(&x).expect("spmv non-square");
        assert_relative_eq!(y[0], 6.0, epsilon = 1e-12);
        assert_relative_eq!(y[1], 15.0, epsilon = 1e-12);
        assert_relative_eq!(y[2], 24.0, epsilon = 1e-12);
        assert_relative_eq!(y[3], 33.0, epsilon = 1e-12);
    }
}
