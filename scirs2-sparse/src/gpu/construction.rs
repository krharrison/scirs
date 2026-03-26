//! GPU-accelerated COO/CSR/CSC construction from triplets.
//!
//! Provides standalone `CooMatrix`, `CsrMatrix`, and `CscMatrix` types that
//! are designed for GPU-friendly construction workflows (sort-then-scan
//! patterns matching a compute shader pipeline).

use crate::error::{SparseError, SparseResult};
use crate::gpu::spmv::{csr_spmv, GpuSpMvConfig};
use scirs2_core::ndarray::Array2;

// ============================================================
// CooMatrix — coordinate format
// ============================================================

/// Sparse matrix in coordinate (COO) format.
///
/// Entries can be pushed in arbitrary order; call `to_csr` or
/// `to_csc` to produce a compressed representation.  Duplicate
/// `(row, col)` entries are **summed** during conversion.
#[derive(Debug, Clone)]
pub struct GpuCooMatrix {
    /// Row indices of non-zero entries.
    pub row_idx: Vec<usize>,
    /// Column indices of non-zero entries.
    pub col_idx: Vec<usize>,
    /// Values of non-zero entries.
    pub values: Vec<f64>,
    /// Number of rows.
    pub n_rows: usize,
    /// Number of columns.
    pub n_cols: usize,
}

impl GpuCooMatrix {
    /// Create an empty COO matrix of the given shape.
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            row_idx: Vec::new(),
            col_idx: Vec::new(),
            values: Vec::new(),
            n_rows,
            n_cols,
        }
    }

    /// Append a single triplet `(row, col, val)`.
    ///
    /// Panics in debug mode when indices are out of bounds.
    pub fn push(&mut self, row: usize, col: usize, val: f64) {
        debug_assert!(row < self.n_rows, "row index out of bounds");
        debug_assert!(col < self.n_cols, "col index out of bounds");
        self.row_idx.push(row);
        self.col_idx.push(col);
        self.values.push(val);
    }

    /// Construct a COO matrix from parallel triplet slices.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InconsistentData`] when slice lengths differ, or
    /// [`SparseError::IndexOutOfBounds`] when any index exceeds the declared
    /// dimensions.
    pub fn from_triplets(
        n_rows: usize,
        n_cols: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
    ) -> SparseResult<Self> {
        if rows.len() != cols.len() || cols.len() != vals.len() {
            return Err(SparseError::InconsistentData {
                reason: format!(
                    "triplet slice lengths do not match: rows={} cols={} vals={}",
                    rows.len(),
                    cols.len(),
                    vals.len()
                ),
            });
        }
        for (i, (&r, &c)) in rows.iter().zip(cols.iter()).enumerate() {
            if r >= n_rows || c >= n_cols {
                return Err(SparseError::IndexOutOfBounds {
                    index: (r, c),
                    shape: (n_rows, n_cols),
                });
            }
            let _ = i; // suppress unused warning
        }
        Ok(Self {
            row_idx: rows.to_vec(),
            col_idx: cols.to_vec(),
            values: vals.to_vec(),
            n_rows,
            n_cols,
        })
    }

    /// Convert to CSR by sorting triplets in row-major order and summing
    /// duplicates.
    pub fn to_csr(&self) -> GpuCsrMatrix {
        let nnz = self.row_idx.len();
        // Build sort order: sort by (row, col)
        let mut order: Vec<usize> = (0..nnz).collect();
        order.sort_unstable_by_key(|&k| (self.row_idx[k], self.col_idx[k]));

        // Merge duplicates
        let mut merged_rows: Vec<usize> = Vec::with_capacity(nnz);
        let mut merged_cols: Vec<usize> = Vec::with_capacity(nnz);
        let mut merged_vals: Vec<f64> = Vec::with_capacity(nnz);

        for &k in &order {
            let r = self.row_idx[k];
            let c = self.col_idx[k];
            let v = self.values[k];
            if let Some(&last_r) = merged_rows.last() {
                let last_c = *merged_cols.last().expect("cols non-empty");
                if last_r == r && last_c == c {
                    *merged_vals.last_mut().expect("vals non-empty") += v;
                    continue;
                }
            }
            merged_rows.push(r);
            merged_cols.push(c);
            merged_vals.push(v);
        }

        // Build row_ptr via prefix sum
        let merged_nnz = merged_rows.len();
        let mut row_ptr = vec![0usize; self.n_rows + 1];
        for &r in &merged_rows {
            row_ptr[r + 1] += 1;
        }
        for i in 0..self.n_rows {
            row_ptr[i + 1] += row_ptr[i];
        }

        GpuCsrMatrix {
            row_ptr,
            col_idx: merged_cols,
            values: merged_vals,
            n_rows: self.n_rows,
            n_cols: self.n_cols,
        }
    }

    /// Convert to CSC by sorting triplets in column-major order and summing
    /// duplicates.
    pub fn to_csc(&self) -> GpuCscMatrix {
        let nnz = self.row_idx.len();
        let mut order: Vec<usize> = (0..nnz).collect();
        order.sort_unstable_by_key(|&k| (self.col_idx[k], self.row_idx[k]));

        let mut merged_rows: Vec<usize> = Vec::with_capacity(nnz);
        let mut merged_cols: Vec<usize> = Vec::with_capacity(nnz);
        let mut merged_vals: Vec<f64> = Vec::with_capacity(nnz);

        for &k in &order {
            let r = self.row_idx[k];
            let c = self.col_idx[k];
            let v = self.values[k];
            if let Some(&last_c) = merged_cols.last() {
                let last_r = *merged_rows.last().expect("rows non-empty");
                if last_c == c && last_r == r {
                    *merged_vals.last_mut().expect("vals non-empty") += v;
                    continue;
                }
            }
            merged_rows.push(r);
            merged_cols.push(c);
            merged_vals.push(v);
        }

        // Build col_ptr via prefix sum
        let mut col_ptr = vec![0usize; self.n_cols + 1];
        for &c in &merged_cols {
            col_ptr[c + 1] += 1;
        }
        for i in 0..self.n_cols {
            col_ptr[i + 1] += col_ptr[i];
        }

        GpuCscMatrix {
            col_ptr,
            row_idx: merged_rows,
            values: merged_vals,
            n_rows: self.n_rows,
            n_cols: self.n_cols,
        }
    }
}

// ============================================================
// GpuCsrMatrix — compressed sparse row
// ============================================================

/// Sparse matrix in compressed sparse row (CSR) format.
///
/// This is a lightweight, GPU-friendly structure with direct access to the
/// three CSR arrays.  For richer functionality use the existing
/// [`crate::csr::CsrMatrix`].
#[derive(Debug, Clone)]
pub struct GpuCsrMatrix {
    /// Row pointer array of length `n_rows + 1`.
    pub row_ptr: Vec<usize>,
    /// Column indices of non-zero entries.
    pub col_idx: Vec<usize>,
    /// Values of non-zero entries.
    pub values: Vec<f64>,
    /// Number of rows.
    pub n_rows: usize,
    /// Number of columns.
    pub n_cols: usize,
}

impl GpuCsrMatrix {
    /// Return the number of stored non-zeros.
    pub fn n_nnz(&self) -> usize {
        self.values.len()
    }

    /// Return the fill density: `nnz / (n_rows * n_cols)`.
    pub fn density(&self) -> f64 {
        let total = self.n_rows * self.n_cols;
        if total == 0 {
            return 0.0;
        }
        self.n_nnz() as f64 / total as f64
    }

    /// Compute `y = A * x`.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] when `x.len() != n_cols`.
    pub fn spmv(&self, x: &[f64]) -> SparseResult<Vec<f64>> {
        if x.len() != self.n_cols {
            return Err(SparseError::DimensionMismatch {
                expected: self.n_cols,
                found: x.len(),
            });
        }
        csr_spmv(
            &self.row_ptr,
            &self.col_idx,
            &self.values,
            x,
            &GpuSpMvConfig::default(),
        )
    }

    /// Compute the transpose `A^T`.
    pub fn transpose(&self) -> GpuCsrMatrix {
        // Build COO for A^T by swapping row and col, then convert.
        let mut coo = GpuCooMatrix::new(self.n_cols, self.n_rows);
        for row in 0..self.n_rows {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            for k in start..end {
                coo.push(self.col_idx[k], row, self.values[k]);
            }
        }
        coo.to_csr()
    }

    /// Convert to a dense `n_rows × n_cols` matrix.
    pub fn to_dense(&self) -> Array2<f64> {
        let mut dense = Array2::zeros((self.n_rows, self.n_cols));
        for row in 0..self.n_rows {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            for k in start..end {
                dense[[row, self.col_idx[k]]] = self.values[k];
            }
        }
        dense
    }
}

// ============================================================
// GpuCscMatrix — compressed sparse column
// ============================================================

/// Sparse matrix in compressed sparse column (CSC) format.
#[derive(Debug, Clone)]
pub struct GpuCscMatrix {
    /// Column pointer array of length `n_cols + 1`.
    pub col_ptr: Vec<usize>,
    /// Row indices of non-zero entries.
    pub row_idx: Vec<usize>,
    /// Values of non-zero entries.
    pub values: Vec<f64>,
    /// Number of rows.
    pub n_rows: usize,
    /// Number of columns.
    pub n_cols: usize,
}

impl GpuCscMatrix {
    /// Return the number of stored non-zeros.
    pub fn n_nnz(&self) -> usize {
        self.values.len()
    }

    /// Convert to CSR format.
    pub fn to_csr(&self) -> GpuCsrMatrix {
        // Build COO and delegate.
        let mut coo = GpuCooMatrix::new(self.n_rows, self.n_cols);
        for col in 0..self.n_cols {
            let start = self.col_ptr[col];
            let end = self.col_ptr[col + 1];
            for k in start..end {
                coo.push(self.row_idx[k], col, self.values[k]);
            }
        }
        coo.to_csr()
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_to_csr_basic() {
        // 3×3 matrix
        // [1 0 2]
        // [0 3 0]
        // [4 5 0]
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 1];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let coo =
            GpuCooMatrix::from_triplets(3, 3, &rows, &cols, &vals).expect("from_triplets failed");
        let csr = coo.to_csr();
        assert_eq!(csr.n_rows, 3);
        assert_eq!(csr.n_cols, 3);
        assert_eq!(csr.n_nnz(), 5);
        assert_eq!(csr.row_ptr, vec![0, 2, 3, 5]);
    }

    #[test]
    fn test_coo_duplicate_sum() {
        // Duplicate (0,0) entries → should be summed
        let rows = vec![0, 0];
        let cols = vec![0, 0];
        let vals = vec![1.0, 2.0];
        let coo =
            GpuCooMatrix::from_triplets(2, 2, &rows, &cols, &vals).expect("from_triplets failed");
        let csr = coo.to_csr();
        assert_eq!(csr.n_nnz(), 1);
        assert!((csr.values[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_csr_spmv_identity() {
        let n = 4;
        let row_ptr: Vec<usize> = (0..=n).collect();
        let col_idx: Vec<usize> = (0..n).collect();
        let values = vec![1.0; n];
        let mat = GpuCsrMatrix {
            row_ptr,
            col_idx,
            values,
            n_rows: n,
            n_cols: n,
        };
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = mat.spmv(&x).expect("spmv failed");
        assert_eq!(y, x);
    }

    #[test]
    fn test_csr_transpose_round_trip() {
        // (A^T)^T = A
        let rows = vec![0, 0, 1];
        let cols = vec![0, 1, 2];
        let vals = vec![1.0, 2.0, 3.0];
        let coo =
            GpuCooMatrix::from_triplets(2, 3, &rows, &cols, &vals).expect("from_triplets failed");
        let csr = coo.to_csr();
        let csr_tt = csr.transpose().transpose();
        assert_eq!(csr.row_ptr, csr_tt.row_ptr);
        assert_eq!(csr.col_idx, csr_tt.col_idx);
        for (a, b) in csr.values.iter().zip(csr_tt.values.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_coo_from_triplets_sorted_csr() {
        // Out-of-order triplets → CSR should be sorted
        let rows = vec![2, 0, 1];
        let cols = vec![0, 0, 0];
        let vals = vec![3.0, 1.0, 2.0];
        let coo =
            GpuCooMatrix::from_triplets(3, 2, &rows, &cols, &vals).expect("from_triplets failed");
        let csr = coo.to_csr();
        // row_ptr must be monotonically non-decreasing
        for w in csr.row_ptr.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn test_coo_to_csc() {
        let rows = vec![0, 1, 0];
        let cols = vec![0, 0, 1];
        let vals = vec![1.0, 2.0, 3.0];
        let coo =
            GpuCooMatrix::from_triplets(2, 2, &rows, &cols, &vals).expect("from_triplets failed");
        let csc = coo.to_csc();
        assert_eq!(csc.n_nnz(), 3);
        // col_ptr[1] should be 2 (two entries in col 0)
        assert_eq!(csc.col_ptr[1], 2);
    }

    #[test]
    fn test_density() {
        let rows = vec![0, 1];
        let cols = vec![0, 1];
        let vals = vec![1.0, 1.0];
        let coo =
            GpuCooMatrix::from_triplets(2, 2, &rows, &cols, &vals).expect("from_triplets failed");
        let csr = coo.to_csr();
        assert!((csr.density() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_to_dense() {
        let rows = vec![0, 1];
        let cols = vec![1, 0];
        let vals = vec![5.0, 7.0];
        let coo =
            GpuCooMatrix::from_triplets(2, 2, &rows, &cols, &vals).expect("from_triplets failed");
        let csr = coo.to_csr();
        let dense = csr.to_dense();
        assert!((dense[[0, 1]] - 5.0).abs() < 1e-12);
        assert!((dense[[1, 0]] - 7.0).abs() < 1e-12);
        assert!((dense[[0, 0]]).abs() < 1e-12);
    }

    #[test]
    fn test_empty_matrix() {
        let coo = GpuCooMatrix::new(0, 0);
        let csr = coo.to_csr();
        assert_eq!(csr.n_nnz(), 0);
        assert_eq!(csr.density(), 0.0);
    }
}
