//! Advanced sparse matrix storage formats
//!
//! This module provides GPU-friendly and cache-optimized sparse matrix formats:
//!
//! - [`BlockCsrMatrix`] — Block Compressed Sparse Row (BCSR): stores `r×r` dense
//!   blocks, reducing index overhead and enabling BLAS-level micro-kernels.
//! - [`EllpackMatrix`] — ELLPACK / ITPACK: each row stores exactly `max_nnz`
//!   entries (padded with −1 in column indices), enabling coalesced GPU access.
//! - [`HybMatrix`] — Hybrid ELL+COO: uses ELLPACK for rows with ≤ `ell_width`
//!   non-zeros and COO overflow for the rest, combining GPU throughput with
//!   irregular-sparsity flexibility.
//!
//! All types implement `matvec` (sparse matrix-vector product) and conversions
//! to/from plain CSR.
//!
//! ## References
//!
//! - Bell & Garland (2009). "Implementing sparse matrix-vector multiplication on
//!   throughput-oriented processors." SC'09.
//! - Vázquez et al. (2011). "ELLR-T: Improving performance of sparse matrix-
//!   vector product with ELLPACK via threads." PPAM.

use crate::error::{SparseError, SparseResult};

// ---------------------------------------------------------------------------
// Block CSR (BCSR) matrix
// ---------------------------------------------------------------------------

/// Block Compressed Sparse Row (BCSR) matrix.
///
/// The matrix is partitioned into `block_size × block_size` dense blocks.
/// Only the blocks that contain at least one non-zero entry are stored.
/// Rows and columns must be divisible by `block_size`; remainders are zero-padded
/// conceptually but not stored.
///
/// # Layout
///
/// ```text
/// block_row_ptr[br + 1] − block_row_ptr[br]   = number of non-zero blocks in block-row br
/// block_col_ind[p]                             = block-column index of the p-th block
/// block_val[p][r][c]                           = value at local position (r, c) of block p
/// ```
#[derive(Debug, Clone)]
pub struct BlockCsrMatrix {
    /// Dense block size (both row and column).
    pub block_size: usize,
    /// Block values: `block_val[p]` is an `r×r` dense matrix (stored as `Vec<Vec<f64>>`).
    pub block_val: Vec<Vec<Vec<f64>>>,
    /// Block column indices (one per stored block).
    pub block_col_ind: Vec<usize>,
    /// Block row pointer (CSR-style, length = `m + 1` where `m` = number of block rows).
    pub block_row_ptr: Vec<usize>,
    /// Number of block rows.
    pub m: usize,
    /// Number of block columns.
    pub n: usize,
}

impl BlockCsrMatrix {
    /// Convert a CSR matrix into BCSR format.
    ///
    /// # Arguments
    ///
    /// * `val`, `row_ptr`, `col_ind` – CSR input.
    /// * `nrows`, `ncols`            – Matrix dimensions.
    /// * `block_size`                – Dense block dimension `r`.
    ///
    /// Rows/columns are padded to the nearest multiple of `block_size`;
    /// values in padded positions are zero.
    pub fn from_csr(
        val: &[f64],
        row_ptr: &[usize],
        col_ind: &[usize],
        nrows: usize,
        ncols: usize,
        block_size: usize,
    ) -> SparseResult<Self> {
        if block_size == 0 {
            return Err(SparseError::InvalidArgument(
                "block_size must be positive".to_string(),
            ));
        }
        if row_ptr.len() != nrows + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "row_ptr.len()={} != nrows+1={}",
                    row_ptr.len(),
                    nrows + 1
                ),
            });
        }

        let r = block_size;
        // Number of block rows and block columns (ceil division)
        let m = (nrows + r - 1) / r;
        let n = (ncols + r - 1) / r;

        // For each CSR row, scatter values into blocks
        // We build a map: block_row → { block_col → dense r×r block }
        use std::collections::HashMap;
        let mut block_map: Vec<HashMap<usize, Vec<Vec<f64>>>> =
            (0..m).map(|_| HashMap::new()).collect();

        for i in 0..nrows {
            let br = i / r;
            let local_row = i % r;
            for pos in row_ptr[i]..row_ptr[i + 1] {
                let j = col_ind[pos];
                if j >= ncols {
                    continue; // skip out-of-range indices
                }
                let bc = j / r;
                let local_col = j % r;
                let block = block_map[br]
                    .entry(bc)
                    .or_insert_with(|| vec![vec![0.0f64; r]; r]);
                block[local_row][local_col] += val[pos];
            }
        }

        // Build sorted BCSR from the map
        let mut all_block_val = Vec::new();
        let mut all_block_col_ind = Vec::new();
        let mut all_block_row_ptr = vec![0usize; m + 1];

        for br in 0..m {
            let mut cols: Vec<usize> = block_map[br].keys().copied().collect();
            cols.sort_unstable();
            for bc in &cols {
                all_block_col_ind.push(*bc);
                all_block_val.push(block_map[br][bc].clone());
            }
            all_block_row_ptr[br + 1] = all_block_val.len();
        }

        Ok(Self {
            block_size: r,
            block_val: all_block_val,
            block_col_ind: all_block_col_ind,
            block_row_ptr: all_block_row_ptr,
            m,
            n,
        })
    }

    /// Sparse matrix-vector product: y = A x.
    ///
    /// `x` must have length `n * block_size`; `y` has length `m * block_size`.
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let r = self.block_size;
        let nrows = self.m * r;
        let mut y = vec![0.0f64; nrows];

        for br in 0..self.m {
            let row_base = br * r;
            for p in self.block_row_ptr[br]..self.block_row_ptr[br + 1] {
                let bc = self.block_col_ind[p];
                let col_base = bc * r;
                let block = &self.block_val[p];
                for local_row in 0..r {
                    let global_row = row_base + local_row;
                    if global_row >= y.len() {
                        break;
                    }
                    let mut acc = 0.0f64;
                    for local_col in 0..r {
                        let global_col = col_base + local_col;
                        if global_col < x.len() {
                            acc += block[local_row][local_col] * x[global_col];
                        }
                    }
                    y[global_row] += acc;
                }
            }
        }
        y
    }

    /// Convert back to plain CSR format.
    ///
    /// Returns `(val, row_ptr, col_ind, nrows_actual, ncols_actual)` where
    /// `nrows_actual = m * block_size` and `ncols_actual = n * block_size`.
    pub fn to_csr(&self) -> (Vec<f64>, Vec<usize>, Vec<usize>, usize, usize) {
        let r = self.block_size;
        let nrows = self.m * r;
        let ncols = self.n * r;

        let mut val_out = Vec::new();
        let mut col_ind_out = Vec::new();
        let mut row_ptr_out = vec![0usize; nrows + 1];

        // First pass: count non-zeros per row
        for br in 0..self.m {
            for p in self.block_row_ptr[br]..self.block_row_ptr[br + 1] {
                let bc = self.block_col_ind[p];
                let block = &self.block_val[p];
                for lr in 0..r {
                    let global_row = br * r + lr;
                    for lc in 0..r {
                        if block[lr][lc] != 0.0 {
                            let global_col = bc * r + lc;
                            if global_col < ncols {
                                row_ptr_out[global_row + 1] += 1;
                            }
                        }
                    }
                }
            }
        }

        // Prefix sum
        for i in 0..nrows {
            row_ptr_out[i + 1] += row_ptr_out[i];
        }

        // Second pass: fill values
        let nnz = row_ptr_out[nrows];
        val_out.resize(nnz, 0.0);
        col_ind_out.resize(nnz, 0);
        let mut row_pos = row_ptr_out[..nrows].to_vec();

        for br in 0..self.m {
            for p in self.block_row_ptr[br]..self.block_row_ptr[br + 1] {
                let bc = self.block_col_ind[p];
                let block = &self.block_val[p];
                for lr in 0..r {
                    let global_row = br * r + lr;
                    for lc in 0..r {
                        if block[lr][lc] != 0.0 {
                            let global_col = bc * r + lc;
                            if global_col < ncols {
                                let pos = row_pos[global_row];
                                val_out[pos] = block[lr][lc];
                                col_ind_out[pos] = global_col;
                                row_pos[global_row] += 1;
                            }
                        }
                    }
                }
            }
        }

        (val_out, row_ptr_out, col_ind_out, nrows, ncols)
    }

    /// Return the number of non-zero blocks.
    pub fn num_blocks(&self) -> usize {
        self.block_val.len()
    }

    /// Return the total number of stored values (including zeros in blocks).
    pub fn stored_values(&self) -> usize {
        self.num_blocks() * self.block_size * self.block_size
    }
}

// ---------------------------------------------------------------------------
// ELLPACK matrix
// ---------------------------------------------------------------------------

/// ELLPACK / ITPACK sparse matrix format.
///
/// Every row stores exactly `max_nnz` `(column, value)` pairs.  Rows with
/// fewer than `max_nnz` non-zeros are padded: column index `−1` (stored as
/// `i64`) signals padding, and the corresponding value is `0`.
///
/// This format enables fully coalesced memory access patterns on GPU hardware
/// when iterating over the matrix in column-major order.
#[derive(Debug, Clone)]
pub struct EllpackMatrix {
    /// Values: `values[row][k]` is the k-th stored value of `row`.
    pub values: Vec<Vec<f64>>,
    /// Column indices: `col_ind[row][k]` is the column of the k-th entry.
    /// Padding entries use `−1`.
    pub col_ind: Vec<Vec<i64>>,
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Maximum non-zeros per row (padding width).
    pub max_nnz: usize,
}

impl EllpackMatrix {
    /// Convert a CSR matrix to ELLPACK format.
    ///
    /// `max_nnz` is automatically computed as the maximum number of
    /// non-zeros in any row.
    pub fn from_csr(
        val: &[f64],
        row_ptr: &[usize],
        col_ind: &[usize],
        nrows: usize,
        ncols: usize,
    ) -> SparseResult<Self> {
        if row_ptr.len() != nrows + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "row_ptr.len()={} != nrows+1={}",
                    row_ptr.len(),
                    nrows + 1
                ),
            });
        }

        // Compute max non-zeros per row
        let max_nnz = (0..nrows)
            .map(|i| row_ptr[i + 1] - row_ptr[i])
            .max()
            .unwrap_or(0);

        let mut values = vec![vec![0.0f64; max_nnz]; nrows];
        let mut col_ind_out = vec![vec![-1i64; max_nnz]; nrows];

        for i in 0..nrows {
            let start = row_ptr[i];
            let end = row_ptr[i + 1];
            for (k, pos) in (start..end).enumerate() {
                values[i][k] = val[pos];
                col_ind_out[i][k] = col_ind[pos] as i64;
            }
        }

        Ok(Self {
            values,
            col_ind: col_ind_out,
            nrows,
            ncols,
            max_nnz,
        })
    }

    /// Sparse matrix-vector product: y = A x.
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0f64; self.nrows];
        for i in 0..self.nrows {
            let mut acc = 0.0f64;
            for k in 0..self.max_nnz {
                let col = self.col_ind[i][k];
                if col < 0 {
                    break; // padding
                }
                let c = col as usize;
                if c < x.len() {
                    acc += self.values[i][k] * x[c];
                }
            }
            y[i] = acc;
        }
        y
    }

    /// Return the number of non-padding entries.
    pub fn nnz(&self) -> usize {
        self.values
            .iter()
            .flat_map(|row| row.iter())
            .zip(self.col_ind.iter().flat_map(|row| row.iter()))
            .filter(|(_, &c)| c >= 0)
            .count()
    }
}

// ---------------------------------------------------------------------------
// HYB (Hybrid ELL + COO) matrix
// ---------------------------------------------------------------------------

/// Hybrid ELL+COO sparse matrix.
///
/// Rows with at most `ell_width` non-zeros are stored in the ELLPACK part.
/// The remaining entries (from rows that exceed `ell_width`) are stored in a
/// coordinate (COO) list.
///
/// This provides GPU-friendly access for the bulk of the matrix while
/// gracefully handling highly irregular rows via COO.
#[derive(Debug, Clone)]
pub struct HybMatrix {
    /// ELLPACK part (truncated to `ell_width` per row).
    ell: EllpackMatrix,
    /// COO overflow values.
    coo_val: Vec<f64>,
    /// COO overflow row indices.
    coo_row: Vec<usize>,
    /// COO overflow column indices.
    coo_col: Vec<usize>,
    /// Number of rows in the global matrix.
    nrows: usize,
    /// Number of columns in the global matrix.
    ncols: usize,
}

impl HybMatrix {
    /// Convert a CSR matrix to the hybrid ELL+COO format.
    ///
    /// # Arguments
    ///
    /// * `val`, `row_ptr`, `col_ind` – CSR input.
    /// * `nrows`, `ncols`            – Matrix dimensions.
    ///
    /// The ELLPACK width is automatically chosen as the median row
    /// non-zero count to balance the two parts.
    pub fn from_csr(
        val: &[f64],
        row_ptr: &[usize],
        col_ind: &[usize],
        nrows: usize,
        ncols: usize,
    ) -> SparseResult<Self> {
        if row_ptr.len() != nrows + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "row_ptr.len()={} != nrows+1={}",
                    row_ptr.len(),
                    nrows + 1
                ),
            });
        }

        // Compute row lengths and determine ELL width (median)
        let row_lengths: Vec<usize> = (0..nrows)
            .map(|i| row_ptr[i + 1] - row_ptr[i])
            .collect();
        let ell_width = if row_lengths.is_empty() {
            0
        } else {
            let mut sorted = row_lengths.clone();
            sorted.sort_unstable();
            sorted[sorted.len() / 2]
        };

        // Build ELLPACK part (each row: first `ell_width` entries)
        // and COO overflow
        let max_nnz = ell_width;
        let mut ell_values = vec![vec![0.0f64; max_nnz]; nrows];
        let mut ell_col_ind = vec![vec![-1i64; max_nnz]; nrows];
        let mut coo_val = Vec::new();
        let mut coo_row = Vec::new();
        let mut coo_col_list = Vec::new();

        for i in 0..nrows {
            let start = row_ptr[i];
            let end = row_ptr[i + 1];
            for (k, pos) in (start..end).enumerate() {
                if k < max_nnz {
                    ell_values[i][k] = val[pos];
                    ell_col_ind[i][k] = col_ind[pos] as i64;
                } else {
                    coo_val.push(val[pos]);
                    coo_row.push(i);
                    coo_col_list.push(col_ind[pos]);
                }
            }
        }

        let ell = EllpackMatrix {
            values: ell_values,
            col_ind: ell_col_ind,
            nrows,
            ncols,
            max_nnz,
        };

        Ok(Self {
            ell,
            coo_val,
            coo_row,
            coo_col: coo_col_list,
            nrows,
            ncols,
        })
    }

    /// Sparse matrix-vector product: y = A x.
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        // ELL part
        let mut y = self.ell.matvec(x);
        // COO part
        for k in 0..self.coo_val.len() {
            let r = self.coo_row[k];
            let c = self.coo_col[k];
            if c < x.len() && r < y.len() {
                y[r] += self.coo_val[k] * x[c];
            }
        }
        y
    }

    /// Return the number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Return the number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Return the ELL width (max entries per row in ELLPACK part).
    pub fn ell_width(&self) -> usize {
        self.ell.max_nnz
    }

    /// Return the number of COO overflow entries.
    pub fn coo_nnz(&self) -> usize {
        self.coo_val.len()
    }

    /// Return the total number of non-zeros (ELL + COO).
    pub fn nnz(&self) -> usize {
        self.ell.nnz() + self.coo_nnz()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple 4×4 tridiagonal CSR: diag=2, off=-1.
    fn tridiag4_csr() -> (Vec<f64>, Vec<usize>, Vec<usize>, usize, usize) {
        let n = 4usize;
        let val = vec![
            2.0f64, -1.0,        // row 0
            -1.0, 2.0, -1.0,     // row 1
            -1.0, 2.0, -1.0,     // row 2
            -1.0, 2.0,           // row 3
        ];
        let row_ptr = vec![0, 2, 5, 8, 10];
        let col_ind = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        (val, row_ptr, col_ind, n, n)
    }

    // Reference matvec on CSR
    fn csr_matvec_ref(val: &[f64], rp: &[usize], ci: &[usize], x: &[f64], n: usize) -> Vec<f64> {
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            for pos in rp[i]..rp[i + 1] {
                y[i] += val[pos] * x[ci[pos]];
            }
        }
        y
    }

    // -------------------------------------------------------------------------
    // BlockCsrMatrix tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bcsr_from_csr_block2() {
        let (val, rp, ci, nrows, ncols) = tridiag4_csr();
        let bcsr = BlockCsrMatrix::from_csr(&val, &rp, &ci, nrows, ncols, 2)
            .expect("BlockCsrMatrix::from_csr");
        assert_eq!(bcsr.m, 2); // 4 rows / 2 = 2 block rows
        assert_eq!(bcsr.n, 2); // 4 cols / 2 = 2 block cols
        assert_eq!(bcsr.block_size, 2);
    }

    #[test]
    fn test_bcsr_matvec_correctness() {
        let (val, rp, ci, nrows, ncols) = tridiag4_csr();
        let bcsr = BlockCsrMatrix::from_csr(&val, &rp, &ci, nrows, ncols, 2)
            .expect("from_csr");
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y_bcsr = bcsr.matvec(&x);
        let y_ref = csr_matvec_ref(&val, &rp, &ci, &x, nrows);
        for i in 0..nrows {
            assert!(
                (y_bcsr[i] - y_ref[i]).abs() < 1e-12,
                "row {i}: bcsr={} ref={}",
                y_bcsr[i],
                y_ref[i]
            );
        }
    }

    #[test]
    fn test_bcsr_to_csr_roundtrip() {
        let (val, rp, ci, nrows, ncols) = tridiag4_csr();
        let bcsr = BlockCsrMatrix::from_csr(&val, &rp, &ci, nrows, ncols, 2)
            .expect("from_csr");
        let (val2, rp2, ci2, nr2, nc2) = bcsr.to_csr();
        assert_eq!(nr2, nrows);
        assert_eq!(nc2, ncols);
        // Verify matvec with recovered CSR
        let x = vec![1.0, 0.0, 1.0, 0.0];
        let y1 = csr_matvec_ref(&val, &rp, &ci, &x, nrows);
        let y2 = csr_matvec_ref(&val2, &rp2, &ci2, &x, nr2);
        for i in 0..nrows {
            assert!((y1[i] - y2[i]).abs() < 1e-12, "row {i}");
        }
    }

    #[test]
    fn test_bcsr_block_size_1_identity() {
        // Block size 1 should behave identically to plain CSR
        let (val, rp, ci, nrows, ncols) = tridiag4_csr();
        let bcsr = BlockCsrMatrix::from_csr(&val, &rp, &ci, nrows, ncols, 1)
            .expect("from_csr block_size=1");
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y_bcsr = bcsr.matvec(&x);
        let y_ref = csr_matvec_ref(&val, &rp, &ci, &x, nrows);
        for i in 0..nrows {
            assert!(
                (y_bcsr[i] - y_ref[i]).abs() < 1e-12,
                "row {i}: bcsr={} ref={}",
                y_bcsr[i],
                y_ref[i]
            );
        }
    }

    #[test]
    fn test_bcsr_zero_block_size_error() {
        let (val, rp, ci, nrows, ncols) = tridiag4_csr();
        let r = BlockCsrMatrix::from_csr(&val, &rp, &ci, nrows, ncols, 0);
        assert!(r.is_err());
    }

    // -------------------------------------------------------------------------
    // EllpackMatrix tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ellpack_from_csr() {
        let (val, rp, ci, nrows, ncols) = tridiag4_csr();
        let ell = EllpackMatrix::from_csr(&val, &rp, &ci, nrows, ncols)
            .expect("EllpackMatrix::from_csr");
        assert_eq!(ell.nrows, nrows);
        assert_eq!(ell.ncols, ncols);
        // Max nnz per row in tridiag: 3 (for rows 1 and 2)
        assert_eq!(ell.max_nnz, 3);
    }

    #[test]
    fn test_ellpack_matvec_correctness() {
        let (val, rp, ci, nrows, ncols) = tridiag4_csr();
        let ell = EllpackMatrix::from_csr(&val, &rp, &ci, nrows, ncols)
            .expect("from_csr");
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y_ell = ell.matvec(&x);
        let y_ref = csr_matvec_ref(&val, &rp, &ci, &x, nrows);
        for i in 0..nrows {
            assert!(
                (y_ell[i] - y_ref[i]).abs() < 1e-12,
                "row {i}: ell={} ref={}",
                y_ell[i],
                y_ref[i]
            );
        }
    }

    #[test]
    fn test_ellpack_nnz() {
        let (val, rp, ci, nrows, ncols) = tridiag4_csr();
        let ell = EllpackMatrix::from_csr(&val, &rp, &ci, nrows, ncols)
            .expect("from_csr");
        let orig_nnz = val.len();
        assert_eq!(ell.nnz(), orig_nnz, "nnz mismatch");
    }

    #[test]
    fn test_ellpack_col_padding() {
        let (val, rp, ci, nrows, ncols) = tridiag4_csr();
        let ell = EllpackMatrix::from_csr(&val, &rp, &ci, nrows, ncols)
            .expect("from_csr");
        // Row 0 has 2 entries, row 2 has 3; padding should be -1
        // Row 0, column 2 (0-indexed) should be -1
        assert_eq!(ell.col_ind[0][2], -1, "padding expected at row 0 pos 2");
        assert_eq!(ell.col_ind[3][2], -1, "padding expected at row 3 pos 2");
    }

    // -------------------------------------------------------------------------
    // HybMatrix tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_hyb_from_csr() {
        let (val, rp, ci, nrows, ncols) = tridiag4_csr();
        let hyb = HybMatrix::from_csr(&val, &rp, &ci, nrows, ncols)
            .expect("HybMatrix::from_csr");
        assert_eq!(hyb.nrows(), nrows);
        assert_eq!(hyb.ncols(), ncols);
    }

    #[test]
    fn test_hyb_matvec_correctness() {
        let (val, rp, ci, nrows, ncols) = tridiag4_csr();
        let hyb = HybMatrix::from_csr(&val, &rp, &ci, nrows, ncols)
            .expect("from_csr");
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y_hyb = hyb.matvec(&x);
        let y_ref = csr_matvec_ref(&val, &rp, &ci, &x, nrows);
        for i in 0..nrows {
            assert!(
                (y_hyb[i] - y_ref[i]).abs() < 1e-12,
                "row {i}: hyb={} ref={}",
                y_hyb[i],
                y_ref[i]
            );
        }
    }

    #[test]
    fn test_hyb_total_nnz() {
        let (val, rp, ci, nrows, ncols) = tridiag4_csr();
        let hyb = HybMatrix::from_csr(&val, &rp, &ci, nrows, ncols)
            .expect("from_csr");
        // Total nnz should equal original
        assert_eq!(hyb.nnz(), val.len());
    }

    #[test]
    fn test_hyb_irregular_matrix() {
        // Build a matrix with very irregular row lengths: row 0 has 1 entry,
        // row 1 has 5 entries, rest have 1 entry each.
        let nrows = 4;
        let ncols = 6;
        // Row 0: col 0
        // Row 1: cols 0-4
        // Row 2: col 5
        // Row 3: col 0
        let val = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0];
        let rp = vec![0, 1, 6, 7, 9];
        let ci = vec![0usize, 0, 1, 2, 3, 4, 5, 0, 2];
        let hyb = HybMatrix::from_csr(&val, &rp, &ci, nrows, ncols)
            .expect("from_csr irregular");
        let x = vec![1.0f64; ncols];
        let y_hyb = hyb.matvec(&x);
        let y_ref = csr_matvec_ref(&val, &rp, &ci, &x, nrows);
        for i in 0..nrows {
            assert!(
                (y_hyb[i] - y_ref[i]).abs() < 1e-12,
                "irregular row {i}: hyb={} ref={}",
                y_hyb[i],
                y_ref[i]
            );
        }
        // Verify overflow captured in COO
        assert!(hyb.coo_nnz() > 0 || hyb.ell_width() >= 5,
            "either COO has overflow or ELL width covers all");
    }

    #[test]
    fn test_formats_dimension_error() {
        let val = vec![1.0];
        let rp = vec![0, 1]; // n=1
        let ci = vec![0];
        // Pass wrong nrows
        let r = EllpackMatrix::from_csr(&val, &rp, &ci, 3, 3);
        assert!(r.is_err());
        let r2 = BlockCsrMatrix::from_csr(&val, &rp, &ci, 3, 3, 2);
        assert!(r2.is_err());
        let r3 = HybMatrix::from_csr(&val, &rp, &ci, 5, 5);
        assert!(r3.is_err());
    }
}
