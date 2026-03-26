//! CSR5 sparse matrix format for balanced SpMV
//!
//! CSR5 is a 2D tiling of the CSR format designed for balanced parallel SpMV.
//! The non-zero elements are partitioned into tiles of configurable width,
//! and a tile descriptor array enables segmented scan within each tile.
//!
//! The key idea: instead of assigning rows to threads (which can be imbalanced),
//! CSR5 assigns equal-sized tiles of non-zeros to threads and uses the tile
//! descriptors to handle row boundaries within tiles via segmented reduction.
//!
//! # References
//!
//! - Liu, W. & Vinter, B. (2015). "CSR5: An Efficient Storage Format for
//!   Cross-Platform Sparse Matrix-Vector Multiplication." ICS'15.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::{SparseElement, Zero};
use std::fmt::Debug;

/// Tile descriptor for CSR5 segmented scan.
///
/// Each tile descriptor records:
/// - Whether the tile contains any row boundaries (segment starts).
/// - For each column within the tile, the row index of the first element.
/// - Whether each column within the tile starts a new segment.
#[derive(Debug, Clone)]
pub struct TileDescriptor {
    /// Row index that the first element of this tile belongs to.
    pub first_row: usize,
    /// Whether this tile has any segment boundaries (row transitions).
    pub has_segment_boundary: bool,
    /// Number of complete rows that start within this tile.
    pub num_complete_rows: usize,
    /// For each element position in the tile, the row it belongs to.
    pub row_ids: Vec<usize>,
    /// Bit-vector: `is_segment_start[i]` is true if position `i` starts a new row.
    pub is_segment_start: Vec<bool>,
}

/// CSR5 sparse matrix format.
///
/// Tiles the CSR non-zeros into 2D tiles for balanced SpMV.
/// Each tile contains `tile_width` non-zeros.
#[derive(Debug, Clone)]
pub struct Csr5Matrix<T> {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Tile width (number of non-zeros per tile).
    pub tile_width: usize,
    /// Number of tiles.
    pub num_tiles: usize,
    /// Column indices (same as CSR).
    pub col_indices: Vec<usize>,
    /// Values (same as CSR).
    pub values: Vec<T>,
    /// Row pointers (same as CSR).
    pub row_ptr: Vec<usize>,
    /// Tile descriptors for segmented scan.
    pub tile_desc: Vec<TileDescriptor>,
    /// Tile pointers: `tile_ptr[t]` = starting offset of tile `t` in col_indices/values.
    pub tile_ptr: Vec<usize>,
}

impl<T> Csr5Matrix<T>
where
    T: Clone + Copy + Zero + SparseElement + Debug,
{
    /// Construct a CSR5 matrix from a CSR matrix.
    ///
    /// # Arguments
    ///
    /// * `csr` - Source CSR matrix.
    /// * `tile_width` - Number of non-zeros per tile. Typical values: 16, 32, 64.
    pub fn from_csr(csr: &CsrMatrix<T>, tile_width: usize) -> SparseResult<Self> {
        if tile_width == 0 {
            return Err(SparseError::ValueError(
                "tile_width must be at least 1".to_string(),
            ));
        }

        let (nrows, ncols) = csr.shape();
        let nnz = csr.nnz();

        // Copy CSR data
        let col_indices = csr.indices.clone();
        let values = csr.data.clone();
        let row_ptr = csr.indptr.clone();

        // Compute number of tiles
        let num_tiles = if nnz == 0 {
            0
        } else {
            nnz.div_ceil(tile_width)
        };

        // Build tile pointers
        let mut tile_ptr = Vec::with_capacity(num_tiles + 1);
        for t in 0..=num_tiles {
            tile_ptr.push((t * tile_width).min(nnz));
        }

        // Build tile descriptors using calibration phase
        let tile_desc = Self::calibrate(&row_ptr, nrows, nnz, tile_width, num_tiles);

        Ok(Self {
            nrows,
            ncols,
            tile_width,
            num_tiles,
            col_indices,
            values,
            row_ptr,
            tile_desc,
            tile_ptr,
        })
    }

    /// Calibration phase: build tile descriptors.
    ///
    /// For each tile, determine which rows its elements belong to and
    /// where segment boundaries (row transitions) occur.
    fn calibrate(
        row_ptr: &[usize],
        nrows: usize,
        nnz: usize,
        tile_width: usize,
        num_tiles: usize,
    ) -> Vec<TileDescriptor> {
        let mut descriptors = Vec::with_capacity(num_tiles);

        for t in 0..num_tiles {
            let tile_start = t * tile_width;
            let tile_end = nnz.min(tile_start + tile_width);
            let tile_len = tile_end - tile_start;

            // Find which row the first element belongs to
            let first_row = Self::find_row(row_ptr, nrows, tile_start);

            // Build row IDs and segment start flags
            let mut row_ids = Vec::with_capacity(tile_len);
            let mut is_segment_start = Vec::with_capacity(tile_len);
            let mut current_row = first_row;
            let mut num_complete_rows = 0usize;
            let mut has_boundary = false;

            for pos in tile_start..tile_end {
                // Advance current_row until row_ptr[current_row + 1] > pos
                while current_row < nrows && row_ptr[current_row + 1] <= pos {
                    current_row += 1;
                }

                let is_start = if pos == tile_start {
                    // First element in tile: it's a segment start if it's also
                    // the first element of its row
                    pos == row_ptr[current_row]
                } else {
                    pos == row_ptr[current_row]
                };

                if is_start && pos != tile_start {
                    has_boundary = true;
                    num_complete_rows += 1;
                }

                row_ids.push(current_row);
                is_segment_start.push(is_start);
            }

            descriptors.push(TileDescriptor {
                first_row,
                has_segment_boundary: has_boundary,
                num_complete_rows,
                row_ids,
                is_segment_start,
            });
        }

        descriptors
    }

    /// Binary search to find which row a given NNZ position belongs to.
    fn find_row(row_ptr: &[usize], nrows: usize, pos: usize) -> usize {
        // row_ptr[row] <= pos < row_ptr[row+1]
        let mut lo = 0usize;
        let mut hi = nrows;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if row_ptr[mid + 1] <= pos {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Perform SpMV: `y = self * x`.
    ///
    /// Two-phase SpMV:
    /// 1. Each tile computes partial sums using segmented reduction.
    /// 2. Partial sums for rows spanning multiple tiles are merged.
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

        if self.num_tiles == 0 {
            return Ok(y);
        }

        // Phase 1: per-tile segmented reduction
        // For each tile, accumulate partial sums and write completed rows to y.
        // Carry-over partial sums are collected for cross-tile merging.

        // carry[t] = (row, partial_sum) for the last segment in tile t
        // that may continue into the next tile.
        let mut carries: Vec<Option<(usize, T)>> = vec![None; self.num_tiles];

        for t in 0..self.num_tiles {
            let desc = &self.tile_desc[t];
            let tile_start = self.tile_ptr[t];
            let tile_end = self.tile_ptr[t + 1];
            let tile_len = tile_end - tile_start;

            if tile_len == 0 {
                continue;
            }

            // Segmented scan within the tile
            let mut acc = T::sparse_zero();
            let mut current_row = desc.first_row;

            for i in 0..tile_len {
                let pos = tile_start + i;
                let row = desc.row_ids[i];

                if row != current_row {
                    // Row boundary: flush acc to y or carry
                    if i == 0 {
                        // The very first position already switched — means
                        // previous tile's carry goes to current_row
                    } else {
                        // current_row's segment is complete within this tile
                        y[current_row] = y[current_row] + acc;
                    }
                    acc = T::sparse_zero();
                    current_row = row;
                }

                acc = acc + self.values[pos] * x[self.col_indices[pos]];
            }

            // Remaining acc is carry for this tile's last row
            carries[t] = Some((current_row, acc));
        }

        // Phase 2: merge carries
        // Process carries from the first tile forward.
        // If tile t's first row equals tile (t-1)'s carry row, accumulate.
        // Otherwise, flush the previous carry.

        for t in 0..self.num_tiles {
            if let Some((row, val)) = carries[t] {
                // Check if the next tile continues this row
                let continues = if t + 1 < self.num_tiles {
                    let next_desc = &self.tile_desc[t + 1];
                    next_desc.first_row == row
                } else {
                    false
                };

                if continues {
                    // Propagate carry to the next tile
                    if let Some((_, ref mut next_val)) = carries[t + 1] {
                        // The next tile's first partial sum needs this carry added
                        // But the next tile's first_row = row, so the next carry
                        // already includes partial sums. We add to y and let next
                        // tile's carry handle the rest.
                        y[row] = y[row] + val;
                    } else {
                        y[row] = y[row] + val;
                    }
                } else {
                    y[row] = y[row] + val;
                }
            }
        }

        Ok(y)
    }

    /// Convert back to CSR format.
    pub fn to_csr(&self) -> SparseResult<CsrMatrix<T>>
    where
        T: std::cmp::PartialEq,
    {
        // The CSR data is already stored internally; just reconstruct triplets.
        let mut row_indices: Vec<usize> = Vec::with_capacity(self.values.len());
        let mut col_indices: Vec<usize> = Vec::with_capacity(self.values.len());
        let mut data: Vec<T> = Vec::with_capacity(self.values.len());

        for row in 0..self.nrows {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            for pos in start..end {
                row_indices.push(row);
                col_indices.push(self.col_indices[pos]);
                data.push(self.values[pos]);
            }
        }

        CsrMatrix::new(data, row_indices, col_indices, (self.nrows, self.ncols))
    }

    /// Number of non-zeros.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Return the tile width.
    pub fn get_tile_width(&self) -> usize {
        self.tile_width
    }

    /// Return the number of tiles.
    pub fn get_num_tiles(&self) -> usize {
        self.num_tiles
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_tridiag_csr(n: usize) -> CsrMatrix<f64> {
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
        CsrMatrix::new(vals, rows, cols, (n, n)).expect("csr")
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
    fn test_csr5_spmv_matches_csr() {
        let csr = make_tridiag_csr(8);
        let x: Vec<f64> = (0..8).map(|i| (i + 1) as f64).collect();
        let y_ref = csr_spmv(&csr, &x);

        for &tw in &[4usize, 8, 16, 32] {
            let csr5 = Csr5Matrix::from_csr(&csr, tw).expect("csr5");
            let y_csr5 = csr5.spmv(&x).expect("spmv");
            for i in 0..8 {
                assert_relative_eq!(y_csr5[i], y_ref[i], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_csr5_preserves_nnz() {
        let csr = make_tridiag_csr(10);
        let csr5 = Csr5Matrix::from_csr(&csr, 4).expect("csr5");
        assert_eq!(csr5.nnz(), csr.nnz());
    }

    #[test]
    fn test_csr5_roundtrip() {
        let csr = make_tridiag_csr(6);
        let csr5 = Csr5Matrix::from_csr(&csr, 4).expect("csr5");
        let csr2 = csr5.to_csr().expect("to_csr");
        assert_eq!(csr2.nnz(), csr.nnz());
        let x: Vec<f64> = (0..6).map(|i| (i + 1) as f64).collect();
        let y1 = csr_spmv(&csr, &x);
        let y2 = csr_spmv(&csr2, &x);
        for i in 0..6 {
            assert_relative_eq!(y1[i], y2[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_csr5_irregular_matrix() {
        // Matrix with varying row lengths
        let rows = vec![0, 0, 0, 0, 1, 2, 2, 3, 3, 3];
        let cols = vec![0, 1, 2, 3, 0, 0, 3, 1, 2, 3];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let csr = CsrMatrix::new(vals, rows, cols, (4, 4)).expect("csr");

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y_ref = csr_spmv(&csr, &x);

        let csr5 = Csr5Matrix::from_csr(&csr, 3).expect("csr5");
        let y_csr5 = csr5.spmv(&x).expect("spmv");

        for i in 0..4 {
            assert_relative_eq!(y_csr5[i], y_ref[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_csr5_empty_matrix() {
        let csr = CsrMatrix::<f64>::new(vec![], vec![], vec![], (3, 3)).expect("csr");
        let csr5 = Csr5Matrix::from_csr(&csr, 4).expect("csr5");
        assert_eq!(csr5.nnz(), 0);
        assert_eq!(csr5.num_tiles, 0);
        let y = csr5.spmv(&[0.0, 0.0, 0.0]).expect("spmv");
        assert_eq!(y, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_csr5_tile_width_error() {
        let csr = make_tridiag_csr(4);
        assert!(Csr5Matrix::<f64>::from_csr(&csr, 0).is_err());
    }

    #[test]
    fn test_csr5_single_row() {
        let csr =
            CsrMatrix::new(vec![1.0, 2.0, 3.0], vec![0, 0, 0], vec![0, 1, 2], (1, 3)).expect("csr");
        let x = vec![1.0, 2.0, 3.0];
        let y_ref = csr_spmv(&csr, &x);
        let csr5 = Csr5Matrix::from_csr(&csr, 2).expect("csr5");
        let y = csr5.spmv(&x).expect("spmv");
        assert_relative_eq!(y[0], y_ref[0], epsilon = 1e-12);
    }

    #[test]
    fn test_csr5_large_tile() {
        // Tile larger than nnz
        let csr = make_tridiag_csr(4);
        let x: Vec<f64> = (0..4).map(|i| (i + 1) as f64).collect();
        let y_ref = csr_spmv(&csr, &x);
        let csr5 = Csr5Matrix::from_csr(&csr, 100).expect("csr5");
        let y = csr5.spmv(&x).expect("spmv");
        for i in 0..4 {
            assert_relative_eq!(y[i], y_ref[i], epsilon = 1e-12);
        }
    }
}
