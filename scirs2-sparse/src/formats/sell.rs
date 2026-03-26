//! Sliced ELLPACK (SELL-C-sigma) sparse matrix format
//!
//! SELL-C-sigma partitions rows into slices of `C` rows, sorts rows within
//! each slice by their number of non-zeros (sigma-sorting) to minimise padding,
//! and stores each slice in ELLPACK (dense column-index + value) layout.
//!
//! This format is cache-friendly and SIMD-friendly because:
//! - Each slice has a uniform width (max NNZ within that slice, not global max).
//! - sigma-sorting ensures rows with similar NNZ share a slice, reducing padding.
//! - Slice width `C` can be aligned to SIMD register width (32, 64, 128).
//!
//! # References
//!
//! - Kreutzer et al. (2014). "A Unified Sparse Matrix Data Format for Efficient
//!   General SpMV on Modern Processors with Wide SIMD Units." SIAM J. Sci. Comput.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::{SparseElement, Zero};
use std::fmt::Debug;

/// Sentinel column index used for SELL padding entries.
pub const SELL_PADDING_COL: usize = usize::MAX;

/// Sliced ELLPACK (SELL-C-sigma) sparse matrix format.
///
/// The matrix rows are partitioned into slices of `slice_width` rows.
/// Within each slice, rows are optionally sigma-sorted by NNZ to reduce padding.
/// Each slice is stored as a dense block of `(slice_height x max_nnz_in_slice)`
/// entries.
///
/// # Layout
///
/// ```text
/// slice_ptr[s]   = offset into values/col_indices where slice s begins
/// slice_ptr[s+1] = offset where slice s ends
/// max_nnz[s]     = maximum row NNZ in slice s (= "width" of the ELL block)
/// ```
///
/// Within slice `s`, element at local row `r`, position `k` is at:
///   `slice_ptr[s] + k * actual_slice_height + r`
/// where `actual_slice_height = min(slice_width, nrows - s * slice_width)`.
#[derive(Debug, Clone)]
pub struct SellMatrix<T> {
    /// Number of rows in the original matrix.
    pub nrows: usize,
    /// Number of columns in the original matrix.
    pub ncols: usize,
    /// Slice width C (number of rows per slice).
    pub slice_width: usize,
    /// Whether sigma-sorting is applied.
    pub sigma_sorted: bool,
    /// Number of slices.
    pub num_slices: usize,
    /// Slice pointers: `slice_ptr[s]` is the starting offset of slice `s`
    /// in `values` and `col_indices`. Length = `num_slices + 1`.
    pub slice_ptr: Vec<usize>,
    /// Maximum NNZ per row within each slice. Length = `num_slices`.
    pub max_nnz: Vec<usize>,
    /// Column indices stored in slice-major order (padded with `SELL_PADDING_COL`).
    pub col_indices: Vec<usize>,
    /// Values stored in slice-major order (padded with zero).
    pub values: Vec<T>,
    /// Row permutation: `perm[i]` = original row index for local row `i`.
    /// When sigma_sorted is false, this is the identity permutation.
    pub perm: Vec<usize>,
    /// Inverse permutation: `inv_perm[original_row]` = local row index.
    pub inv_perm: Vec<usize>,
}

impl<T> SellMatrix<T>
where
    T: Clone + Copy + Zero + SparseElement + Debug,
{
    /// Construct a SELL-C-sigma matrix from a CSR matrix.
    ///
    /// # Arguments
    ///
    /// * `csr` - Source CSR matrix.
    /// * `slice_width` - Number of rows per slice (C). Common values: 32, 64, 128.
    /// * `sigma_sort` - Whether to sigma-sort rows by NNZ within each slice.
    pub fn from_csr(
        csr: &CsrMatrix<T>,
        slice_width: usize,
        sigma_sort: bool,
    ) -> SparseResult<Self> {
        if slice_width == 0 {
            return Err(SparseError::ValueError(
                "slice_width must be at least 1".to_string(),
            ));
        }

        let (nrows, ncols) = csr.shape();
        let num_slices = nrows.div_ceil(slice_width);

        // Compute row NNZ counts
        let row_nnz: Vec<usize> = (0..nrows)
            .map(|r| csr.indptr[r + 1] - csr.indptr[r])
            .collect();

        // Build permutation
        let mut perm: Vec<usize> = (0..nrows).collect();

        if sigma_sort {
            // Sort rows within each slice by NNZ (descending) to minimize padding
            for s in 0..num_slices {
                let start = s * slice_width;
                let end = nrows.min(start + slice_width);
                let slice_range = &mut perm[start..end];
                slice_range.sort_by(|&a, &b| row_nnz[b].cmp(&row_nnz[a]));
            }
        }

        // Build inverse permutation
        let mut inv_perm = vec![0usize; nrows];
        for (local, &original) in perm.iter().enumerate() {
            inv_perm[original] = local;
        }

        // Build slice data
        let mut slice_ptr = Vec::with_capacity(num_slices + 1);
        let mut max_nnz_vec = Vec::with_capacity(num_slices);
        let mut col_indices_out: Vec<usize> = Vec::new();
        let mut values_out: Vec<T> = Vec::new();

        slice_ptr.push(0usize);

        for s in 0..num_slices {
            let start = s * slice_width;
            let end = nrows.min(start + slice_width);
            let actual_height = end - start;

            // Find max NNZ in this slice
            let slice_max_nnz = perm[start..end]
                .iter()
                .map(|&orig_row| row_nnz[orig_row])
                .max()
                .unwrap_or(0);
            max_nnz_vec.push(slice_max_nnz);

            // Store in column-major order within the slice:
            // For position k (0..slice_max_nnz), for local row r (0..actual_height):
            //   index = base + k * actual_height + r
            let base = col_indices_out.len();
            let slice_size = slice_max_nnz * actual_height;
            col_indices_out.resize(base + slice_size, SELL_PADDING_COL);
            values_out.resize(base + slice_size, T::sparse_zero());

            for (local_r, &orig_row) in perm[start..end].iter().enumerate() {
                let row_start = csr.indptr[orig_row];
                let row_end = csr.indptr[orig_row + 1];
                for (k, j) in (row_start..row_end).enumerate() {
                    let idx = base + k * actual_height + local_r;
                    col_indices_out[idx] = csr.indices[j];
                    values_out[idx] = csr.data[j];
                }
            }

            slice_ptr.push(col_indices_out.len());
        }

        Ok(Self {
            nrows,
            ncols,
            slice_width,
            sigma_sorted: sigma_sort,
            num_slices,
            slice_ptr,
            max_nnz: max_nnz_vec,
            col_indices: col_indices_out,
            values: values_out,
            perm,
            inv_perm,
        })
    }

    /// Perform SpMV: `y = self * x`.
    ///
    /// The result `y` is in the original row ordering (not the permuted order).
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

        for s in 0..self.num_slices {
            let start = s * self.slice_width;
            let end = self.nrows.min(start + self.slice_width);
            let actual_height = end - start;
            let base = self.slice_ptr[s];
            let slice_max = self.max_nnz[s];

            // Vectorizable inner loop: iterate over positions k, then rows
            for k in 0..slice_max {
                for r in 0..actual_height {
                    let idx = base + k * actual_height + r;
                    let col = self.col_indices[idx];
                    if col == SELL_PADDING_COL {
                        continue;
                    }
                    let orig_row = self.perm[start + r];
                    y[orig_row] = y[orig_row] + self.values[idx] * x[col];
                }
            }
        }

        Ok(y)
    }

    /// Convert back to CSR format (original row ordering).
    pub fn to_csr(&self) -> SparseResult<CsrMatrix<T>>
    where
        T: std::cmp::PartialEq,
    {
        let mut row_indices: Vec<usize> = Vec::new();
        let mut col_indices: Vec<usize> = Vec::new();
        let mut data: Vec<T> = Vec::new();

        for s in 0..self.num_slices {
            let start = s * self.slice_width;
            let end = self.nrows.min(start + self.slice_width);
            let actual_height = end - start;
            let base = self.slice_ptr[s];
            let slice_max = self.max_nnz[s];

            for r in 0..actual_height {
                let orig_row = self.perm[start + r];
                for k in 0..slice_max {
                    let idx = base + k * actual_height + r;
                    let col = self.col_indices[idx];
                    if col == SELL_PADDING_COL {
                        continue;
                    }
                    let val = self.values[idx];
                    row_indices.push(orig_row);
                    col_indices.push(col);
                    data.push(val);
                }
            }
        }

        CsrMatrix::new(data, row_indices, col_indices, (self.nrows, self.ncols))
    }

    /// Number of stored non-zeros (excluding padding).
    pub fn nnz(&self) -> usize {
        self.col_indices
            .iter()
            .filter(|&&c| c != SELL_PADDING_COL)
            .count()
    }

    /// Total storage (including padding) in number of elements.
    pub fn storage_size(&self) -> usize {
        self.values.len()
    }

    /// Padding overhead ratio: (storage - nnz) / nnz.
    /// Returns 0.0 for empty matrices.
    pub fn padding_ratio(&self) -> f64 {
        let nnz = self.nnz();
        if nnz == 0 {
            return 0.0;
        }
        (self.storage_size() as f64 - nnz as f64) / nnz as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_test_csr() -> CsrMatrix<f64> {
        // 6x6 tridiagonal: 2 on diagonal, -1 on off-diagonals
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..6 {
            rows.push(i);
            cols.push(i);
            vals.push(2.0);
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                vals.push(-1.0);
            }
            if i + 1 < 6 {
                rows.push(i);
                cols.push(i + 1);
                vals.push(-1.0);
            }
        }
        CsrMatrix::new(vals, rows, cols, (6, 6)).expect("csr")
    }

    fn csr_spmv(csr: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
        let (nrows, _) = csr.shape();
        let mut y = vec![0.0f64; nrows];
        for row in 0..nrows {
            for j in csr.indptr[row]..csr.indptr[row + 1] {
                y[row] += csr.data[j] * x[csr.indices[j]];
            }
        }
        y
    }

    #[test]
    fn test_sell_spmv_matches_csr() {
        let csr = make_test_csr();
        let x: Vec<f64> = (0..6).map(|i| (i + 1) as f64).collect();
        let y_ref = csr_spmv(&csr, &x);

        for &c in &[2usize, 4, 32] {
            let sell = SellMatrix::from_csr(&csr, c, true).expect("sell");
            let y_sell = sell.spmv(&x).expect("spmv");
            for i in 0..6 {
                assert_relative_eq!(y_sell[i], y_ref[i], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_sell_preserves_nnz() {
        let csr = make_test_csr();
        let sell = SellMatrix::from_csr(&csr, 2, true).expect("sell");
        assert_eq!(sell.nnz(), csr.nnz());
    }

    #[test]
    fn test_sell_roundtrip() {
        let csr = make_test_csr();
        let sell = SellMatrix::from_csr(&csr, 4, true).expect("sell");
        let csr2 = sell.to_csr().expect("to_csr");
        assert_eq!(csr2.nnz(), csr.nnz());
        let x: Vec<f64> = (0..6).map(|i| (i + 1) as f64).collect();
        let y1 = csr_spmv(&csr, &x);
        let y2 = csr_spmv(&csr2, &x);
        for i in 0..6 {
            assert_relative_eq!(y1[i], y2[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_sell_no_sigma_sort() {
        let csr = make_test_csr();
        let sell = SellMatrix::from_csr(&csr, 2, false).expect("sell");
        let x: Vec<f64> = (0..6).map(|i| (i + 1) as f64).collect();
        let y_ref = csr_spmv(&csr, &x);
        let y_sell = sell.spmv(&x).expect("spmv");
        for i in 0..6 {
            assert_relative_eq!(y_sell[i], y_ref[i], epsilon = 1e-12);
        }
        assert!(!sell.sigma_sorted);
    }

    #[test]
    fn test_sell_padding_ratio() {
        let csr = make_test_csr();
        let sell = SellMatrix::from_csr(&csr, 2, true).expect("sell");
        let ratio = sell.padding_ratio();
        assert!(ratio >= 0.0);
    }

    #[test]
    fn test_sell_irregular_matrix() {
        // Matrix with irregular row lengths
        let rows = vec![0, 0, 0, 0, 1, 2, 2, 3, 3, 3];
        let cols = vec![0, 1, 2, 3, 0, 0, 3, 1, 2, 3];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let csr = CsrMatrix::new(vals, rows, cols, (4, 4)).expect("csr");
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y_ref = csr_spmv(&csr, &x);

        let sell = SellMatrix::from_csr(&csr, 2, true).expect("sell");
        let y_sell = sell.spmv(&x).expect("spmv");
        for i in 0..4 {
            assert_relative_eq!(y_sell[i], y_ref[i], epsilon = 1e-12);
        }
        assert_eq!(sell.nnz(), csr.nnz());
    }

    #[test]
    fn test_sell_zero_slice_width_error() {
        let csr = make_test_csr();
        assert!(SellMatrix::<f64>::from_csr(&csr, 0, true).is_err());
    }
}
