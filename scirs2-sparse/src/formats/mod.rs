//! Additional sparse matrix storage formats
//!
//! This module provides GPU-friendly and bandwidth-optimized sparse formats:
//!
//! - [`EllpackMatrix`]: ELLPACK/ITPACK format — fixed non-zeros per row, excellent for GPU.
//! - [`DiaMatrix`]: Diagonal storage — highly efficient for banded / tridiagonal matrices.
//! - [`BlockCSR`]: Block CSR (BCSR) — exploits dense block sub-structures.
//! - [`HybridMatrix`]: ELL+COO hybrid — handles irregular sparsity for GPU-friendly storage.
//! - [`format_convert`] functions — zero-copy or minimal-copy conversions between all formats.

pub mod bsc;
pub mod bsr;
pub mod csf;
pub mod csr5;
pub mod poly_precond;
pub mod sell;

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::{Float, SparseElement, Zero};
use std::fmt::Debug;

// ============================================================
// EllpackMatrix
// ============================================================

/// ELLPACK (ITPACK) sparse matrix format.
///
/// Stores the matrix as a dense 2-D array where every row has the same width
/// (`max_nnz_per_row`). Shorter rows are padded with `(invalid_col, 0.0)`.
///
/// # Layout
///
/// ```text
/// col_indices: [row0_col0, row0_col1, ..., row0_col_{max-1},
///               row1_col0, ..., ]        -- shape (nrows, max_nnz_per_row)
/// values:      same shape
/// ```
///
/// This layout enables coalesced GPU memory access during SpMV.
#[derive(Debug, Clone)]
pub struct EllpackMatrix<T> {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Maximum non-zeros per row (padding width).
    pub max_nnz_per_row: usize,
    /// Column indices, row-major; length = `nrows * max_nnz_per_row`.
    /// Padding entries use `ELLPACK_PADDING_COL`.
    pub col_indices: Vec<usize>,
    /// Values, row-major; padding entries are zero.
    pub values: Vec<T>,
}

/// Sentinel column index used for ELLPACK padding.
pub const ELLPACK_PADDING_COL: usize = usize::MAX;

impl<T> EllpackMatrix<T>
where
    T: Clone + Copy + Zero + SparseElement + Debug,
{
    /// Construct an ELLPACK matrix from a CSR matrix.
    ///
    /// The `max_nnz_per_row` is determined automatically as the maximum row
    /// length in the CSR matrix.
    pub fn from_csr(csr: &CsrMatrix<T>) -> SparseResult<Self> {
        let (nrows, ncols) = csr.shape();
        // Compute max NNZ per row.
        let max_nnz_per_row = (0..nrows)
            .map(|r| csr.indptr[r + 1] - csr.indptr[r])
            .max()
            .unwrap_or(0);

        let total = nrows * max_nnz_per_row;
        let mut col_indices = vec![ELLPACK_PADDING_COL; total];
        let mut values = vec![T::sparse_zero(); total];

        for row in 0..nrows {
            let row_start = csr.indptr[row];
            let row_end = csr.indptr[row + 1];
            for (k, j) in (row_start..row_end).enumerate() {
                let pos = row * max_nnz_per_row + k;
                col_indices[pos] = csr.indices[j];
                values[pos] = csr.data[j];
            }
        }

        Ok(Self {
            nrows,
            ncols,
            max_nnz_per_row,
            col_indices,
            values,
        })
    }

    /// Perform SpMV: `y = self * x`.
    pub fn spmv(&self, x: &[T]) -> SparseResult<Vec<T>>
    where
        T: std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        if x.len() != self.ncols {
            return Err(SparseError::DimensionMismatch {
                expected: self.ncols,
                found: x.len(),
            });
        }
        let mut y = vec![T::sparse_zero(); self.nrows];
        for row in 0..self.nrows {
            let base = row * self.max_nnz_per_row;
            for k in 0..self.max_nnz_per_row {
                let col = self.col_indices[base + k];
                if col == ELLPACK_PADDING_COL {
                    break; // Padding begins here (columns are stored in order).
                }
                y[row] = y[row] + self.values[base + k] * x[col];
            }
        }
        Ok(y)
    }

    /// Convert back to CSR format.
    pub fn to_csr(&self) -> SparseResult<CsrMatrix<T>>
    where
        T: std::cmp::PartialEq,
    {
        let mut row_indices: Vec<usize> = Vec::new();
        let mut col_indices: Vec<usize> = Vec::new();
        let mut data: Vec<T> = Vec::new();

        for row in 0..self.nrows {
            let base = row * self.max_nnz_per_row;
            for k in 0..self.max_nnz_per_row {
                let col = self.col_indices[base + k];
                if col == ELLPACK_PADDING_COL {
                    break;
                }
                let val = self.values[base + k];
                if val != T::sparse_zero() || col != ELLPACK_PADDING_COL {
                    row_indices.push(row);
                    col_indices.push(col);
                    data.push(val);
                }
            }
        }
        CsrMatrix::new(data, row_indices, col_indices, (self.nrows, self.ncols))
    }

    /// Number of non-zeros (excluding padding).
    pub fn nnz(&self) -> usize {
        self.col_indices
            .iter()
            .filter(|&&c| c != ELLPACK_PADDING_COL)
            .count()
    }
}

// ============================================================
// DiaMatrix (diagonal storage for banded matrices)
// ============================================================

/// Diagonal storage format for banded / multi-diagonal sparse matrices.
///
/// Stores explicitly named diagonals. Diagonal `offset = 0` is the main
/// diagonal, `offset > 0` is superdiagonal, `offset < 0` is subdiagonal.
///
/// # Layout
///
/// Each diagonal `d` with offset `k` has length `min(nrows, ncols, nrows - max(0,-k), ncols - max(0,k))`.
/// They are stored as flat `Vec<T>` with the first valid element at position 0.
#[derive(Debug, Clone)]
pub struct DiaMatrix<T> {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Diagonal offsets (may be negative).
    pub offsets: Vec<i64>,
    /// Diagonal data; `diags[i]` has length equal to the diagonal with `offsets[i]`.
    pub diags: Vec<Vec<T>>,
}

impl<T> DiaMatrix<T>
where
    T: Clone + Copy + Zero + SparseElement + Debug,
{
    /// Create a `DiaMatrix` from explicit diagonals.
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of rows.
    /// * `ncols` - Number of columns.
    /// * `offsets` - Diagonal offsets.
    /// * `diags` - One `Vec<T>` per offset.
    pub fn new(
        nrows: usize,
        ncols: usize,
        offsets: Vec<i64>,
        diags: Vec<Vec<T>>,
    ) -> SparseResult<Self> {
        if offsets.len() != diags.len() {
            return Err(SparseError::DimensionMismatch {
                expected: offsets.len(),
                found: diags.len(),
            });
        }
        // Validate diagonal lengths.
        for (i, &k) in offsets.iter().enumerate() {
            let expected_len = diagonal_length(nrows, ncols, k);
            if diags[i].len() != expected_len {
                return Err(SparseError::ValueError(format!(
                    "Diagonal at offset {} has length {} but expected {}",
                    k,
                    diags[i].len(),
                    expected_len
                )));
            }
        }
        Ok(Self {
            nrows,
            ncols,
            offsets,
            diags,
        })
    }

    /// Construct from a CSR matrix by extracting its diagonals.
    pub fn from_csr(csr: &CsrMatrix<T>) -> SparseResult<Self> {
        let (nrows, ncols) = csr.shape();
        // Discover all occupied diagonals.
        let mut offset_set: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        for row in 0..nrows {
            for j in csr.indptr[row]..csr.indptr[row + 1] {
                let col = csr.indices[j];
                offset_set.insert(col as i64 - row as i64);
            }
        }
        let offsets: Vec<i64> = offset_set.into_iter().collect();
        let mut diag_map: std::collections::HashMap<i64, Vec<T>> = std::collections::HashMap::new();
        for &k in &offsets {
            let len = diagonal_length(nrows, ncols, k);
            diag_map.insert(k, vec![T::sparse_zero(); len]);
        }

        for row in 0..nrows {
            for j in csr.indptr[row]..csr.indptr[row + 1] {
                let col = csr.indices[j];
                let k = col as i64 - row as i64;
                let diag_idx = if k >= 0 {
                    row // element index in the diagonal
                } else {
                    col
                };
                if let Some(d) = diag_map.get_mut(&k) {
                    if diag_idx < d.len() {
                        d[diag_idx] = csr.data[j];
                    }
                }
            }
        }

        let diags: Vec<Vec<T>> = offsets.iter().map(|k| diag_map[k].clone()).collect();
        Ok(Self {
            nrows,
            ncols,
            offsets,
            diags,
        })
    }

    /// SpMV: `y = self * x`.
    pub fn spmv(&self, x: &[T]) -> SparseResult<Vec<T>>
    where
        T: std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        if x.len() != self.ncols {
            return Err(SparseError::DimensionMismatch {
                expected: self.ncols,
                found: x.len(),
            });
        }
        let mut y = vec![T::sparse_zero(); self.nrows];
        for (idx, &k) in self.offsets.iter().enumerate() {
            let d = &self.diags[idx];
            // If k >= 0: element d[i] is at (i, i + k) for i in 0..min(nrows, ncols - k).
            // If k <  0: element d[i] is at (i - k, i) for i in 0..min(ncols, nrows + k).
            if k >= 0 {
                let ku = k as usize;
                for (i, &v) in d.iter().enumerate() {
                    let col = i + ku;
                    if col < self.ncols {
                        y[i] = y[i] + v * x[col];
                    }
                }
            } else {
                let km = (-k) as usize;
                for (i, &v) in d.iter().enumerate() {
                    let row = i + km;
                    if row < self.nrows {
                        y[row] = y[row] + v * x[i];
                    }
                }
            }
        }
        Ok(y)
    }

    /// Convert to CSR format.
    pub fn to_csr(&self) -> SparseResult<CsrMatrix<T>> {
        let mut row_idx: Vec<usize> = Vec::new();
        let mut col_idx: Vec<usize> = Vec::new();
        let mut data: Vec<T> = Vec::new();

        for (idx, &k) in self.offsets.iter().enumerate() {
            let d = &self.diags[idx];
            if k >= 0 {
                let ku = k as usize;
                for (i, &v) in d.iter().enumerate() {
                    let col = i + ku;
                    if col < self.ncols && v != T::sparse_zero() {
                        row_idx.push(i);
                        col_idx.push(col);
                        data.push(v);
                    }
                }
            } else {
                let km = (-k) as usize;
                for (i, &v) in d.iter().enumerate() {
                    let row = i + km;
                    if row < self.nrows && v != T::sparse_zero() {
                        row_idx.push(row);
                        col_idx.push(i);
                        data.push(v);
                    }
                }
            }
        }

        CsrMatrix::new(data, row_idx, col_idx, (self.nrows, self.ncols))
    }

    /// Number of non-zeros (sum of diagonal lengths).
    pub fn nnz(&self) -> usize {
        self.diags.iter().map(|d| d.len()).sum()
    }
}

/// Compute the length of diagonal `k` in an `nrows × ncols` matrix.
fn diagonal_length(nrows: usize, ncols: usize, k: i64) -> usize {
    if k >= 0 {
        let ku = k as usize;
        if ku >= ncols {
            0
        } else {
            nrows.min(ncols - ku)
        }
    } else {
        let km = (-k) as usize;
        if km >= nrows {
            0
        } else {
            ncols.min(nrows - km)
        }
    }
}

// ============================================================
// BlockCSR (BCSR)
// ============================================================

/// Block Compressed Sparse Row (BCSR / BSR) format.
///
/// The matrix is divided into `(r × c)` blocks. Only blocks containing at least
/// one non-zero are stored. Internally each stored block is a dense `r * c` array.
///
/// # When to use
///
/// Use `BlockCSR` when the sparse pattern has a natural block structure (e.g.,
/// FEM stiffness matrices, graph problems with vector unknowns per node). Dense
/// block kernels (`GEMV`) run much faster than scalar scatter operations.
#[derive(Debug, Clone)]
pub struct BlockCSR<T> {
    /// Matrix rows.
    pub nrows: usize,
    /// Matrix cols.
    pub ncols: usize,
    /// Block row size.
    pub block_rows: usize,
    /// Block col size.
    pub block_cols: usize,
    /// Number of block rows = ceil(nrows / block_rows).
    pub num_block_rows: usize,
    /// Number of block cols = ceil(ncols / block_cols).
    pub num_block_cols: usize,
    /// CSR-like indptr over block rows; length = `num_block_rows + 1`.
    pub indptr: Vec<usize>,
    /// Block column indices; length = number of stored blocks.
    pub block_col_indices: Vec<usize>,
    /// Dense block data; length = `#blocks * block_rows * block_cols`.
    /// Block `k` occupies `data[k * block_rows * block_cols .. (k+1) * block_rows * block_cols]`
    /// in row-major order.
    pub data: Vec<T>,
}

impl<T> BlockCSR<T>
where
    T: Clone + Copy + Zero + SparseElement + Debug + std::cmp::PartialEq,
{
    /// Build a `BlockCSR` from a CSR matrix with given block dimensions.
    ///
    /// # Arguments
    ///
    /// * `csr` - Source CSR matrix.
    /// * `block_rows` - Block row size (must be ≥ 1).
    /// * `block_cols` - Block col size (must be ≥ 1).
    pub fn from_csr(
        csr: &CsrMatrix<T>,
        block_rows: usize,
        block_cols: usize,
    ) -> SparseResult<Self> {
        if block_rows == 0 || block_cols == 0 {
            return Err(SparseError::ValueError(
                "Block dimensions must be at least 1".to_string(),
            ));
        }
        let (nrows, ncols) = csr.shape();
        let num_block_rows = nrows.div_ceil(block_rows);
        let num_block_cols = ncols.div_ceil(block_cols);
        let block_size = block_rows * block_cols;

        // For each block position, collect non-zeros.
        // Use a HashMap (brow, bcol) -> dense block.
        let mut block_map: std::collections::BTreeMap<(usize, usize), Vec<T>> =
            std::collections::BTreeMap::new();

        for row in 0..nrows {
            let brow = row / block_rows;
            let local_row = row % block_rows;
            for j in csr.indptr[row]..csr.indptr[row + 1] {
                let col = csr.indices[j];
                let bcol = col / block_cols;
                let local_col = col % block_cols;
                let block = block_map
                    .entry((brow, bcol))
                    .or_insert_with(|| vec![T::sparse_zero(); block_size]);
                block[local_row * block_cols + local_col] = csr.data[j];
            }
        }

        // Build BCSR arrays.
        let mut indptr = vec![0usize; num_block_rows + 1];
        let mut block_col_indices: Vec<usize> = Vec::new();
        let mut data: Vec<T> = Vec::new();

        // Iterate block rows in order.
        let mut current_brow = 0usize;
        for (&(brow, bcol), block_data) in &block_map {
            while current_brow < brow {
                indptr[current_brow + 1] = block_col_indices.len();
                current_brow += 1;
            }
            block_col_indices.push(bcol);
            data.extend_from_slice(block_data);
        }
        // Fill remaining indptr entries.
        while current_brow < num_block_rows {
            indptr[current_brow + 1] = block_col_indices.len();
            current_brow += 1;
        }
        indptr[num_block_rows] = block_col_indices.len();

        Ok(Self {
            nrows,
            ncols,
            block_rows,
            block_cols,
            num_block_rows,
            num_block_cols,
            indptr,
            block_col_indices,
            data,
        })
    }

    /// SpMV: `y = self * x`.
    pub fn spmv(&self, x: &[T]) -> SparseResult<Vec<T>>
    where
        T: std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        if x.len() != self.ncols {
            return Err(SparseError::DimensionMismatch {
                expected: self.ncols,
                found: x.len(),
            });
        }
        let mut y = vec![T::sparse_zero(); self.nrows];
        let bs = self.block_rows * self.block_cols;

        for brow in 0..self.num_block_rows {
            let row_base = brow * self.block_rows;
            for kb in self.indptr[brow]..self.indptr[brow + 1] {
                let bcol = self.block_col_indices[kb];
                let col_base = bcol * self.block_cols;
                let block = &self.data[kb * bs..(kb + 1) * bs];
                for lr in 0..self.block_rows {
                    let row = row_base + lr;
                    if row >= self.nrows {
                        break;
                    }
                    for lc in 0..self.block_cols {
                        let col = col_base + lc;
                        if col >= self.ncols {
                            break;
                        }
                        y[row] = y[row] + block[lr * self.block_cols + lc] * x[col];
                    }
                }
            }
        }
        Ok(y)
    }

    /// Convert back to CSR.
    pub fn to_csr(&self) -> SparseResult<CsrMatrix<T>> {
        let bs = self.block_rows * self.block_cols;
        let mut row_idx: Vec<usize> = Vec::new();
        let mut col_idx: Vec<usize> = Vec::new();
        let mut data_out: Vec<T> = Vec::new();

        for brow in 0..self.num_block_rows {
            let row_base = brow * self.block_rows;
            for kb in self.indptr[brow]..self.indptr[brow + 1] {
                let bcol = self.block_col_indices[kb];
                let col_base = bcol * self.block_cols;
                let block = &self.data[kb * bs..(kb + 1) * bs];
                for lr in 0..self.block_rows {
                    let row = row_base + lr;
                    if row >= self.nrows {
                        break;
                    }
                    for lc in 0..self.block_cols {
                        let col = col_base + lc;
                        if col >= self.ncols {
                            break;
                        }
                        let v = block[lr * self.block_cols + lc];
                        if v != T::sparse_zero() {
                            row_idx.push(row);
                            col_idx.push(col);
                            data_out.push(v);
                        }
                    }
                }
            }
        }
        CsrMatrix::new(data_out, row_idx, col_idx, (self.nrows, self.ncols))
    }

    /// Number of stored blocks.
    pub fn num_blocks(&self) -> usize {
        self.block_col_indices.len()
    }
}

// ============================================================
// HybridMatrix (ELL + COO)
// ============================================================

/// ELL+COO hybrid sparse matrix format.
///
/// The matrix is split into two components:
///
/// - **ELL part**: rows with NNZ ≤ `ell_width` are stored in ELLPACK format.
/// - **COO part**: remaining non-zeros from rows exceeding `ell_width` are
///   stored in coordinate (COO) format.
///
/// This hybrid strikes a balance between regular GPU memory access (ELL) and
/// full generality (COO) for matrices with irregular degree distribution.
#[derive(Debug, Clone)]
pub struct HybridMatrix<T> {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// ELL width (maximum NNZ per row in the ELL part).
    pub ell_width: usize,
    /// ELL column indices: `nrows * ell_width` entries.
    pub ell_col: Vec<usize>,
    /// ELL values: `nrows * ell_width` entries.
    pub ell_val: Vec<T>,
    /// COO row indices (overflow entries).
    pub coo_row: Vec<usize>,
    /// COO column indices (overflow entries).
    pub coo_col: Vec<usize>,
    /// COO values (overflow entries).
    pub coo_val: Vec<T>,
}

impl<T> HybridMatrix<T>
where
    T: Clone + Copy + Zero + SparseElement + Debug,
{
    /// Build from CSR using the given ELL width.
    ///
    /// Rows with more than `ell_width` non-zeros contribute their first
    /// `ell_width` entries to the ELL part and remaining entries to COO.
    pub fn from_csr(csr: &CsrMatrix<T>, ell_width: usize) -> SparseResult<Self> {
        let (nrows, ncols) = csr.shape();
        let total_ell = nrows * ell_width;

        let mut ell_col = vec![ELLPACK_PADDING_COL; total_ell];
        let mut ell_val = vec![T::sparse_zero(); total_ell];
        let mut coo_row: Vec<usize> = Vec::new();
        let mut coo_col: Vec<usize> = Vec::new();
        let mut coo_val: Vec<T> = Vec::new();

        for row in 0..nrows {
            let row_start = csr.indptr[row];
            let row_end = csr.indptr[row + 1];
            let base = row * ell_width;

            for (k, j) in (row_start..row_end).enumerate() {
                let col = csr.indices[j];
                let val = csr.data[j];
                if k < ell_width {
                    ell_col[base + k] = col;
                    ell_val[base + k] = val;
                } else {
                    coo_row.push(row);
                    coo_col.push(col);
                    coo_val.push(val);
                }
            }
        }

        Ok(Self {
            nrows,
            ncols,
            ell_width,
            ell_col,
            ell_val,
            coo_row,
            coo_col,
            coo_val,
        })
    }

    /// Auto-select `ell_width` as the median row NNZ (a common heuristic).
    pub fn from_csr_auto(csr: &CsrMatrix<T>) -> SparseResult<Self> {
        let (nrows, _) = csr.shape();
        let mut row_nnz: Vec<usize> = (0..nrows)
            .map(|r| csr.indptr[r + 1] - csr.indptr[r])
            .collect();
        row_nnz.sort_unstable();
        let ell_width = row_nnz.get(nrows / 2).copied().unwrap_or(0);
        Self::from_csr(csr, ell_width)
    }

    /// SpMV: `y = self * x`.
    pub fn spmv(&self, x: &[T]) -> SparseResult<Vec<T>>
    where
        T: std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        if x.len() != self.ncols {
            return Err(SparseError::DimensionMismatch {
                expected: self.ncols,
                found: x.len(),
            });
        }
        let mut y = vec![T::sparse_zero(); self.nrows];

        // ELL part.
        for row in 0..self.nrows {
            let base = row * self.ell_width;
            for k in 0..self.ell_width {
                let col = self.ell_col[base + k];
                if col == ELLPACK_PADDING_COL {
                    break;
                }
                y[row] = y[row] + self.ell_val[base + k] * x[col];
            }
        }

        // COO part.
        for ((&row, &col), &val) in self
            .coo_row
            .iter()
            .zip(self.coo_col.iter())
            .zip(self.coo_val.iter())
        {
            y[row] = y[row] + val * x[col];
        }

        Ok(y)
    }

    /// Convert to CSR.
    pub fn to_csr(&self) -> SparseResult<CsrMatrix<T>>
    where
        T: std::cmp::PartialEq,
    {
        let mut row_idx: Vec<usize> = Vec::new();
        let mut col_idx: Vec<usize> = Vec::new();
        let mut data: Vec<T> = Vec::new();

        // ELL part.
        for row in 0..self.nrows {
            let base = row * self.ell_width;
            for k in 0..self.ell_width {
                let col = self.ell_col[base + k];
                if col == ELLPACK_PADDING_COL {
                    break;
                }
                let val = self.ell_val[base + k];
                row_idx.push(row);
                col_idx.push(col);
                data.push(val);
            }
        }

        // COO part.
        for ((&row, &col), &val) in self
            .coo_row
            .iter()
            .zip(self.coo_col.iter())
            .zip(self.coo_val.iter())
        {
            row_idx.push(row);
            col_idx.push(col);
            data.push(val);
        }

        CsrMatrix::new(data, row_idx, col_idx, (self.nrows, self.ncols))
    }

    /// Total number of non-zeros.
    pub fn nnz(&self) -> usize {
        let ell_nnz = self
            .ell_col
            .iter()
            .filter(|&&c| c != ELLPACK_PADDING_COL)
            .count();
        ell_nnz + self.coo_val.len()
    }
}

// ============================================================
// format_convert utilities
// ============================================================

/// Conversion utilities between all sparse formats.
pub mod format_convert {
    use super::*;

    /// Convert CSR → ELLPACK.
    pub fn csr_to_ellpack<T>(csr: &CsrMatrix<T>) -> SparseResult<EllpackMatrix<T>>
    where
        T: Clone + Copy + Zero + SparseElement + Debug,
    {
        EllpackMatrix::from_csr(csr)
    }

    /// Convert CSR → Diagonal (DIA).
    pub fn csr_to_dia<T>(csr: &CsrMatrix<T>) -> SparseResult<DiaMatrix<T>>
    where
        T: Clone + Copy + Zero + SparseElement + Debug + std::cmp::PartialEq,
    {
        DiaMatrix::from_csr(csr)
    }

    /// Convert CSR → Block CSR with given block dimensions.
    pub fn csr_to_bcsr<T>(
        csr: &CsrMatrix<T>,
        block_rows: usize,
        block_cols: usize,
    ) -> SparseResult<BlockCSR<T>>
    where
        T: Clone + Copy + Zero + SparseElement + Debug + std::cmp::PartialEq,
    {
        BlockCSR::from_csr(csr, block_rows, block_cols)
    }

    /// Convert CSR → Hybrid (ELL+COO) with automatic ELL width selection.
    pub fn csr_to_hybrid<T>(csr: &CsrMatrix<T>) -> SparseResult<HybridMatrix<T>>
    where
        T: Clone + Copy + Zero + SparseElement + Debug,
    {
        HybridMatrix::from_csr_auto(csr)
    }

    /// Convert ELLPACK → CSR.
    pub fn ellpack_to_csr<T>(ell: &EllpackMatrix<T>) -> SparseResult<CsrMatrix<T>>
    where
        T: Clone + Copy + Zero + SparseElement + Debug + std::cmp::PartialEq,
    {
        ell.to_csr()
    }

    /// Convert DIA → CSR.
    pub fn dia_to_csr<T>(dia: &DiaMatrix<T>) -> SparseResult<CsrMatrix<T>>
    where
        T: Clone + Copy + Zero + SparseElement + Debug + std::cmp::PartialEq,
    {
        dia.to_csr()
    }

    /// Convert BCSR → CSR.
    pub fn bcsr_to_csr<T>(bcsr: &BlockCSR<T>) -> SparseResult<CsrMatrix<T>>
    where
        T: Clone + Copy + Zero + SparseElement + Debug + std::cmp::PartialEq,
    {
        bcsr.to_csr()
    }

    /// Convert Hybrid → CSR.
    pub fn hybrid_to_csr<T>(hyb: &HybridMatrix<T>) -> SparseResult<CsrMatrix<T>>
    where
        T: Clone + Copy + Zero + SparseElement + Debug + std::cmp::PartialEq,
    {
        hyb.to_csr()
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        // Tridiagonal matrix: -1 2 -1
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(2.0);
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                vals.push(-1.0);
            }
            if i + 1 < n {
                rows.push(i);
                cols.push(i + 1);
                vals.push(-1.0);
            }
        }
        CsrMatrix::new(vals, rows, cols, (n, n)).expect("laplacian_1d")
    }

    fn check_spmv(csr: &CsrMatrix<f64>, y_ref: &[f64], label: &str) {
        let n = csr.cols();
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        // Reference
        let mut ref_y = vec![0.0f64; csr.rows()];
        for row in 0..csr.rows() {
            for j in csr.indptr[row]..csr.indptr[row + 1] {
                ref_y[row] += csr.data[j] * x[csr.indices[j]];
            }
        }
        for (i, (&got, &exp)) in y_ref.iter().zip(ref_y.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "{}[{}] mismatch: got={} exp={}",
                label,
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_ellpack_spmv_roundtrip() {
        let csr = laplacian_1d(6);
        let n = csr.cols();
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let ell = EllpackMatrix::from_csr(&csr).expect("ellpack");
        let y = ell.spmv(&x).expect("spmv");
        check_spmv(&csr, &y, "ELLPACK");

        // Roundtrip.
        let csr2 = ell.to_csr().expect("ell->csr");
        assert_eq!(csr2.nnz(), csr.nnz());
    }

    #[test]
    fn test_dia_spmv_roundtrip() {
        let csr = laplacian_1d(6);
        let n = csr.cols();
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let dia = DiaMatrix::from_csr(&csr).expect("dia");
        let y = dia.spmv(&x).expect("spmv");
        check_spmv(&csr, &y, "DIA");

        let csr2 = dia.to_csr().expect("dia->csr");
        assert_eq!(csr2.nnz(), csr.nnz());
    }

    #[test]
    fn test_bcsr_spmv_roundtrip() {
        let csr = laplacian_1d(6);
        let n = csr.cols();
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let bcsr = BlockCSR::from_csr(&csr, 2, 2).expect("bcsr");
        let y = bcsr.spmv(&x).expect("spmv");
        check_spmv(&csr, &y, "BCSR");

        let csr2 = bcsr.to_csr().expect("bcsr->csr");
        assert_eq!(csr2.nnz(), csr.nnz());
    }

    #[test]
    fn test_hybrid_spmv_roundtrip() {
        let csr = laplacian_1d(6);
        let n = csr.cols();
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let hyb = HybridMatrix::from_csr_auto(&csr).expect("hybrid");
        let y = hyb.spmv(&x).expect("spmv");
        check_spmv(&csr, &y, "Hybrid");

        let csr2 = hyb.to_csr().expect("hyb->csr");
        assert_eq!(csr2.nnz(), csr.nnz());
    }

    #[test]
    fn test_format_convert_roundtrip() {
        let csr = laplacian_1d(8);
        let n = csr.cols();
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let ell = format_convert::csr_to_ellpack(&csr).expect("csr_to_ellpack");
        let y_ell = ell.spmv(&x).expect("ell spmv");

        let hyb = format_convert::csr_to_hybrid(&csr).expect("csr_to_hybrid");
        let y_hyb = hyb.spmv(&x).expect("hyb spmv");

        for i in 0..csr.rows() {
            assert_relative_eq!(y_ell[i], y_hyb[i], epsilon = 1e-12);
        }
    }
}
