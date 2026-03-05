//! Enhanced Diagonal (DIA) format with efficient banded matrix operations
//!
//! This module provides enhanced DIA format operations including:
//! - Efficient banded matrix-vector multiplication
//! - Banded LU solve (Thomas algorithm for tridiagonal, general banded)
//! - Direct conversion to/from CsrArray and CscArray
//! - Banded matrix arithmetic

use crate::csc_array::CscArray;
use crate::csr_array::CsrArray;
use crate::dia_array::DiaArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, SparseElement};
use std::fmt::Debug;
use std::ops::Div;

/// Enhanced DIA matrix with banded operation support
///
/// Wraps a standard DIA format and adds efficient banded algorithms.
/// The internal representation stores diagonals indexed by offset:
///   offset > 0: super-diagonal
///   offset = 0: main diagonal
///   offset < 0: sub-diagonal
#[derive(Debug, Clone)]
pub struct EnhancedDia<T>
where
    T: Float + SparseElement + Debug + Copy + 'static,
{
    /// Diagonal data: `diags[d][i]` is the element at position
    /// `(i, i + offsets[d])` if offset >= 0, or `(i - offsets[d], i)` if offset < 0.
    /// Each diagonal has length `max(nrows, ncols)` (padded with zeros for out-of-range).
    diags: Vec<Vec<T>>,
    /// Diagonal offsets (sorted)
    offsets: Vec<isize>,
    /// Number of rows
    nrows: usize,
    /// Number of columns
    ncols: usize,
}

impl<T> EnhancedDia<T>
where
    T: Float + SparseElement + Debug + Copy + 'static,
{
    /// Create a new EnhancedDia from diagonal data and offsets.
    ///
    /// # Arguments
    /// * `diags` - Diagonal vectors. Each must have length `max(nrows, ncols)`.
    /// * `offsets` - Diagonal offsets.
    /// * `nrows` - Number of rows.
    /// * `ncols` - Number of columns.
    pub fn new(
        diags: Vec<Vec<T>>,
        offsets: Vec<isize>,
        nrows: usize,
        ncols: usize,
    ) -> SparseResult<Self> {
        if diags.len() != offsets.len() {
            return Err(SparseError::DimensionMismatch {
                expected: offsets.len(),
                found: diags.len(),
            });
        }
        let max_dim = nrows.max(ncols);
        for (d, diag) in diags.iter().enumerate() {
            if diag.len() != max_dim {
                return Err(SparseError::DimensionMismatch {
                    expected: max_dim,
                    found: diag.len(),
                });
            }
        }
        if nrows == 0 || ncols == 0 {
            return Err(SparseError::ValueError(
                "Matrix dimensions must be positive".to_string(),
            ));
        }

        // Sort by offsets for efficient access
        let mut indexed: Vec<(isize, Vec<T>)> = offsets.into_iter().zip(diags).collect();
        indexed.sort_by_key(|&(off, _)| off);

        let sorted_offsets: Vec<isize> = indexed.iter().map(|(off, _)| *off).collect();
        let sorted_diags: Vec<Vec<T>> = indexed.into_iter().map(|(_, d)| d).collect();

        Ok(Self {
            diags: sorted_diags,
            offsets: sorted_offsets,
            nrows,
            ncols,
        })
    }

    /// Create from a DiaArray
    pub fn from_dia_array(dia: &DiaArray<T>) -> SparseResult<Self>
    where
        T: Float + SparseElement + Div<Output = T> + std::ops::AddAssign + 'static,
    {
        let (nrows, ncols) = dia.shape();
        // Extract non-zero elements via to_array
        let dense = dia.to_array();
        let max_dim = nrows.max(ncols);

        // Determine which diagonals are present
        let mut diag_map: std::collections::BTreeMap<isize, Vec<T>> =
            std::collections::BTreeMap::new();

        for i in 0..nrows {
            for j in 0..ncols {
                let v = dense[[i, j]];
                if !SparseElement::is_zero(&v) {
                    let offset = j as isize - i as isize;
                    let diag_vec = diag_map
                        .entry(offset)
                        .or_insert_with(|| vec![T::sparse_zero(); max_dim]);
                    // For offset >= 0: element at (i, i+offset) -> diag index i
                    // For offset < 0: element at (i, j) where i = j - offset -> diag index j
                    if offset >= 0 {
                        diag_vec[i] = v;
                    } else {
                        diag_vec[j] = v;
                    }
                }
            }
        }

        let offsets: Vec<isize> = diag_map.keys().copied().collect();
        let diags: Vec<Vec<T>> = diag_map.into_values().collect();

        Self::new(diags, offsets, nrows, ncols)
    }

    /// Create a tridiagonal matrix with given diagonals.
    ///
    /// # Arguments
    /// * `lower` - Sub-diagonal (length n-1)
    /// * `main` - Main diagonal (length n)
    /// * `upper` - Super-diagonal (length n-1)
    pub fn tridiagonal(lower: &[T], main: &[T], upper: &[T]) -> SparseResult<Self> {
        let n = main.len();
        if lower.len() != n.saturating_sub(1) || upper.len() != n.saturating_sub(1) {
            return Err(SparseError::ValueError(
                "Tridiagonal: lower and upper must have length n-1".to_string(),
            ));
        }
        if n == 0 {
            return Err(SparseError::ValueError(
                "Matrix dimension must be positive".to_string(),
            ));
        }

        let max_dim = n;
        let mut main_diag = vec![T::sparse_zero(); max_dim];
        let mut lower_diag = vec![T::sparse_zero(); max_dim];
        let mut upper_diag = vec![T::sparse_zero(); max_dim];

        main_diag[..n].copy_from_slice(&main[..n]);
        let m = n.saturating_sub(1);
        lower_diag[..m].copy_from_slice(&lower[..m]); // a_{i+1, i} stored at index i
        upper_diag[..m].copy_from_slice(&upper[..m]); // a_{i, i+1} stored at index i

        Self::new(
            vec![lower_diag, main_diag, upper_diag],
            vec![-1, 0, 1],
            n,
            n,
        )
    }

    /// Get the shape
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Get the bandwidth: (lower bandwidth, upper bandwidth)
    pub fn bandwidth(&self) -> (usize, usize) {
        let lower = self
            .offsets
            .iter()
            .filter(|&&o| o < 0)
            .map(|&o| (-o) as usize)
            .max()
            .unwrap_or(0);
        let upper = self
            .offsets
            .iter()
            .filter(|&&o| o > 0)
            .map(|&o| o as usize)
            .max()
            .unwrap_or(0);
        (lower, upper)
    }

    /// Get element at (i, j)
    pub fn get(&self, i: usize, j: usize) -> T {
        if i >= self.nrows || j >= self.ncols {
            return T::sparse_zero();
        }
        let offset = j as isize - i as isize;
        if let Ok(idx) = self.offsets.binary_search(&offset) {
            // For offset >= 0: diag index is i (row index)
            // For offset < 0: diag index is j (col index)
            let diag_idx = if offset >= 0 { i } else { j };
            if diag_idx < self.diags[idx].len() {
                return self.diags[idx][diag_idx];
            }
        }
        T::sparse_zero()
    }

    /// Number of non-zero elements
    pub fn nnz(&self) -> usize {
        let mut count = 0usize;
        for (d, &offset) in self.offsets.iter().enumerate() {
            let diag = &self.diags[d];
            // Determine valid range for this diagonal
            let (start, len) = diagonal_range(self.nrows, self.ncols, offset);
            for k in 0..len {
                let idx = start + k;
                if idx < diag.len() && !SparseElement::is_zero(&diag[idx]) {
                    count += 1;
                }
            }
        }
        count
    }

    /// Banded matrix-vector multiplication: y = A * x
    pub fn matvec(&self, x: &[T]) -> SparseResult<Vec<T>> {
        if x.len() != self.ncols {
            return Err(SparseError::DimensionMismatch {
                expected: self.ncols,
                found: x.len(),
            });
        }

        let mut y = vec![T::sparse_zero(); self.nrows];

        for (d, &offset) in self.offsets.iter().enumerate() {
            let diag = &self.diags[d];
            // Iterate over the valid range of this diagonal
            let (diag_start, diag_len) = diagonal_range(self.nrows, self.ncols, offset);
            for k in 0..diag_len {
                let diag_idx = diag_start + k;
                let (row, col) = if offset >= 0 {
                    (diag_idx, diag_idx + offset as usize)
                } else {
                    (diag_idx + (-offset) as usize, diag_idx)
                };
                if row < self.nrows && col < self.ncols && diag_idx < diag.len() {
                    y[row] = y[row] + diag[diag_idx] * x[col];
                }
            }
        }

        Ok(y)
    }

    /// Convert to CsrArray
    pub fn to_csr(&self) -> SparseResult<CsrArray<T>>
    where
        T: Float + SparseElement + Div<Output = T> + 'static,
    {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for (d, &offset) in self.offsets.iter().enumerate() {
            let diag = &self.diags[d];
            let (diag_start, diag_len) = diagonal_range(self.nrows, self.ncols, offset);
            for k in 0..diag_len {
                let diag_idx = diag_start + k;
                let (row, col) = if offset >= 0 {
                    (diag_idx, diag_idx + offset as usize)
                } else {
                    (diag_idx + (-offset) as usize, diag_idx)
                };
                if row < self.nrows && col < self.ncols && diag_idx < diag.len() {
                    let v = diag[diag_idx];
                    if !SparseElement::is_zero(&v) {
                        rows.push(row);
                        cols.push(col);
                        vals.push(v);
                    }
                }
            }
        }

        CsrArray::from_triplets(&rows, &cols, &vals, (self.nrows, self.ncols), false)
    }

    /// Convert to CscArray
    pub fn to_csc(&self) -> SparseResult<CscArray<T>>
    where
        T: Float + SparseElement + Div<Output = T> + 'static,
    {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for (d, &offset) in self.offsets.iter().enumerate() {
            let diag = &self.diags[d];
            let (diag_start, diag_len) = diagonal_range(self.nrows, self.ncols, offset);
            for k in 0..diag_len {
                let diag_idx = diag_start + k;
                let (row, col) = if offset >= 0 {
                    (diag_idx, diag_idx + offset as usize)
                } else {
                    (diag_idx + (-offset) as usize, diag_idx)
                };
                if row < self.nrows && col < self.ncols && diag_idx < diag.len() {
                    let v = diag[diag_idx];
                    if !SparseElement::is_zero(&v) {
                        rows.push(row);
                        cols.push(col);
                        vals.push(v);
                    }
                }
            }
        }

        CscArray::from_triplets(&rows, &cols, &vals, (self.nrows, self.ncols), false)
    }

    /// Create from a CsrArray (extracting diagonals from the CSR data)
    pub fn from_csr(csr: &CsrArray<T>) -> SparseResult<Self>
    where
        T: Float + SparseElement + Div<Output = T> + 'static,
    {
        let (nrows, ncols) = csr.shape();
        let (row_arr, col_arr, val_arr) = csr.find();
        let max_dim = nrows.max(ncols);

        let mut diag_map: std::collections::BTreeMap<isize, Vec<T>> =
            std::collections::BTreeMap::new();

        for idx in 0..row_arr.len() {
            let r = row_arr[idx];
            let c = col_arr[idx];
            let v = val_arr[idx];
            let offset = c as isize - r as isize;

            let diag_vec = diag_map
                .entry(offset)
                .or_insert_with(|| vec![T::sparse_zero(); max_dim]);

            let diag_idx = if offset >= 0 { r } else { c };
            if diag_idx < max_dim {
                diag_vec[diag_idx] = v;
            }
        }

        let offsets: Vec<isize> = diag_map.keys().copied().collect();
        let diags: Vec<Vec<T>> = diag_map.into_values().collect();

        Self::new(diags, offsets, nrows, ncols)
    }

    /// Convert to dense Array2
    pub fn to_dense(&self) -> scirs2_core::ndarray::Array2<T> {
        let mut result = scirs2_core::ndarray::Array2::zeros((self.nrows, self.ncols));
        for (d, &offset) in self.offsets.iter().enumerate() {
            let diag = &self.diags[d];
            let (diag_start, diag_len) = diagonal_range(self.nrows, self.ncols, offset);
            for k in 0..diag_len {
                let diag_idx = diag_start + k;
                let (row, col) = if offset >= 0 {
                    (diag_idx, diag_idx + offset as usize)
                } else {
                    (diag_idx + (-offset) as usize, diag_idx)
                };
                if row < self.nrows && col < self.ncols && diag_idx < diag.len() {
                    result[[row, col]] = diag[diag_idx];
                }
            }
        }
        result
    }

    /// Add two banded matrices (must have the same shape)
    pub fn add(&self, other: &Self) -> SparseResult<Self> {
        if self.nrows != other.nrows || self.ncols != other.ncols {
            return Err(SparseError::ShapeMismatch {
                expected: (self.nrows, self.ncols),
                found: (other.nrows, other.ncols),
            });
        }

        let max_dim = self.nrows.max(self.ncols);
        let mut diag_map: std::collections::BTreeMap<isize, Vec<T>> =
            std::collections::BTreeMap::new();

        // Add self's diagonals
        for (d, &off) in self.offsets.iter().enumerate() {
            let entry = diag_map
                .entry(off)
                .or_insert_with(|| vec![T::sparse_zero(); max_dim]);
            for i in 0..max_dim {
                entry[i] = entry[i] + self.diags[d][i];
            }
        }

        // Add other's diagonals
        for (d, &off) in other.offsets.iter().enumerate() {
            let entry = diag_map
                .entry(off)
                .or_insert_with(|| vec![T::sparse_zero(); max_dim]);
            for i in 0..max_dim {
                entry[i] = entry[i] + other.diags[d][i];
            }
        }

        let offsets: Vec<isize> = diag_map.keys().copied().collect();
        let diags: Vec<Vec<T>> = diag_map.into_values().collect();

        Self::new(diags, offsets, self.nrows, self.ncols)
    }

    /// Scale the matrix by a scalar
    pub fn scale(&self, alpha: T) -> Self {
        let diags: Vec<Vec<T>> = self
            .diags
            .iter()
            .map(|d| d.iter().map(|&v| v * alpha).collect())
            .collect();
        Self {
            diags,
            offsets: self.offsets.clone(),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

/// Compute the starting index and length of a diagonal given matrix dimensions and offset.
fn diagonal_range(nrows: usize, ncols: usize, offset: isize) -> (usize, usize) {
    if offset >= 0 {
        let off = offset as usize;
        let len = if off < ncols {
            nrows.min(ncols - off)
        } else {
            0
        };
        (0, len)
    } else {
        let off = (-offset) as usize;
        let len = if off < nrows {
            ncols.min(nrows - off)
        } else {
            0
        };
        (0, len)
    }
}

// ---------------------------------------------------------------------------
// Banded solvers
// ---------------------------------------------------------------------------

/// Solve a tridiagonal system Ax = b using the Thomas algorithm.
///
/// The matrix is defined by three diagonals:
/// * `lower` - Sub-diagonal (length n-1)
/// * `main` - Main diagonal (length n)
/// * `upper` - Super-diagonal (length n-1)
/// * `b` - Right-hand side vector (length n)
///
/// Returns the solution vector x.
pub fn tridiagonal_solve<T>(lower: &[T], main: &[T], upper: &[T], b: &[T]) -> SparseResult<Vec<T>>
where
    T: Float + SparseElement + Debug + Copy + 'static,
{
    let n = main.len();
    if lower.len() != n.saturating_sub(1) || upper.len() != n.saturating_sub(1) || b.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: b.len(),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        let d = main[0];
        if d.abs() < T::from(1e-14).unwrap_or(T::sparse_zero()) {
            return Err(SparseError::SingularMatrix(
                "Zero diagonal in tridiagonal solve".to_string(),
            ));
        }
        return Ok(vec![b[0] / d]);
    }

    // Forward sweep
    let mut c_prime = vec![T::sparse_zero(); n];
    let mut d_prime = vec![T::sparse_zero(); n];

    let m0 = main[0];
    if m0.abs() < T::from(1e-14).unwrap_or(T::sparse_zero()) {
        return Err(SparseError::SingularMatrix(
            "Zero pivot in Thomas algorithm at row 0".to_string(),
        ));
    }
    c_prime[0] = upper[0] / m0;
    d_prime[0] = b[0] / m0;

    for i in 1..n {
        let denom = main[i] - lower[i - 1] * c_prime[i - 1];
        if denom.abs() < T::from(1e-14).unwrap_or(T::sparse_zero()) {
            return Err(SparseError::SingularMatrix(format!(
                "Zero pivot in Thomas algorithm at row {i}"
            )));
        }
        if i < n - 1 {
            c_prime[i] = upper[i] / denom;
        }
        d_prime[i] = (b[i] - lower[i - 1] * d_prime[i - 1]) / denom;
    }

    // Back substitution
    let mut x = vec![T::sparse_zero(); n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

/// Solve a general banded system Ax = b.
///
/// Uses banded LU decomposition (without pivoting) for matrices stored in
/// banded form.
///
/// # Arguments
/// * `dia` - The banded matrix in EnhancedDia format (must be square)
/// * `b` - Right-hand side vector
pub fn banded_solve<T>(dia: &EnhancedDia<T>, b: &[T]) -> SparseResult<Vec<T>>
where
    T: Float + SparseElement + Debug + Copy + 'static,
{
    let (nrows, ncols) = dia.shape();
    if nrows != ncols {
        return Err(SparseError::ValueError(
            "Banded solve requires a square matrix".to_string(),
        ));
    }
    if b.len() != nrows {
        return Err(SparseError::DimensionMismatch {
            expected: nrows,
            found: b.len(),
        });
    }

    let n = nrows;
    let (kl, ku) = dia.bandwidth();

    // Check if tridiagonal (special fast path)
    if kl <= 1 && ku <= 1 {
        let mut main_diag = vec![T::sparse_zero(); n];
        let mut lower_diag = vec![T::sparse_zero(); n.saturating_sub(1)];
        let mut upper_diag = vec![T::sparse_zero(); n.saturating_sub(1)];

        for i in 0..n {
            main_diag[i] = dia.get(i, i);
        }
        for i in 0..n.saturating_sub(1) {
            lower_diag[i] = dia.get(i + 1, i);
            upper_diag[i] = dia.get(i, i + 1);
        }

        return tridiagonal_solve(&lower_diag, &main_diag, &upper_diag, b);
    }

    // General banded LU (no pivoting)
    // Store the matrix densely for the factorization (only band entries)
    let mut a = vec![vec![T::sparse_zero(); n]; n];
    for i in 0..n {
        let j_start = i.saturating_sub(kl);
        let j_end = (i + ku + 1).min(n);
        for j in j_start..j_end {
            a[i][j] = dia.get(i, j);
        }
    }

    // LU factorization in place (band-aware)
    for k in 0..n {
        let pivot = a[k][k];
        if pivot.abs() < T::from(1e-14).unwrap_or(T::sparse_zero()) {
            return Err(SparseError::SingularMatrix(format!(
                "Zero pivot at row {k} in banded LU"
            )));
        }
        let i_end = (k + kl + 1).min(n);
        for i in (k + 1)..i_end {
            let factor = a[i][k] / pivot;
            a[i][k] = factor; // Store L factor
            let j_end = (k + ku + 1).min(n);
            for j in (k + 1)..j_end {
                a[i][j] = a[i][j] - factor * a[k][j];
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = b.to_vec();
    for i in 1..n {
        let j_start = i.saturating_sub(kl);
        for j in j_start..i {
            y[i] = y[i] - a[i][j] * y[j];
        }
    }

    // Back substitution: U x = y
    let mut x = y;
    for i in (0..n).rev() {
        let j_end = (i + ku + 1).min(n);
        for j in (i + 1)..j_end {
            x[i] = x[i] - a[i][j] * x[j];
        }
        let d = a[i][i];
        if d.abs() < T::from(1e-14).unwrap_or(T::sparse_zero()) {
            return Err(SparseError::SingularMatrix(format!(
                "Zero diagonal at row {i} in back substitution"
            )));
        }
        x[i] = x[i] / d;
    }

    Ok(x)
}

/// Solve a tridiagonal system stored in EnhancedDia format.
///
/// This is a convenience function that extracts the three diagonals and
/// calls the Thomas algorithm.
pub fn dia_tridiagonal_solve<T>(dia: &EnhancedDia<T>, b: &[T]) -> SparseResult<Vec<T>>
where
    T: Float + SparseElement + Debug + Copy + 'static,
{
    let (nrows, ncols) = dia.shape();
    if nrows != ncols {
        return Err(SparseError::ValueError(
            "Tridiagonal solve requires a square matrix".to_string(),
        ));
    }
    let n = nrows;
    if b.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: b.len(),
        });
    }

    let mut main_diag = vec![T::sparse_zero(); n];
    let mut lower_diag = vec![T::sparse_zero(); n.saturating_sub(1)];
    let mut upper_diag = vec![T::sparse_zero(); n.saturating_sub(1)];

    for i in 0..n {
        main_diag[i] = dia.get(i, i);
    }
    for i in 0..n.saturating_sub(1) {
        lower_diag[i] = dia.get(i + 1, i);
        upper_diag[i] = dia.get(i, i + 1);
    }

    tridiagonal_solve(&lower_diag, &main_diag, &upper_diag, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_enhanced_dia_tridiagonal() {
        // Create tridiagonal matrix:
        // [2, -1,  0]
        // [-1, 2, -1]
        // [0, -1,  2]
        let lower = vec![-1.0, -1.0];
        let main = vec![2.0, 2.0, 2.0];
        let upper = vec![-1.0, -1.0];

        let dia = EnhancedDia::tridiagonal(&lower, &main, &upper).expect("should succeed");

        assert_eq!(dia.shape(), (3, 3));
        assert_eq!(dia.bandwidth(), (1, 1));
        assert_relative_eq!(dia.get(0, 0), 2.0);
        assert_relative_eq!(dia.get(0, 1), -1.0);
        assert_relative_eq!(dia.get(1, 0), -1.0);
        assert_relative_eq!(dia.get(1, 1), 2.0);
        assert_relative_eq!(dia.get(2, 1), -1.0);
        assert_relative_eq!(dia.get(2, 2), 2.0);
        assert_relative_eq!(dia.get(0, 2), 0.0);
    }

    #[test]
    fn test_enhanced_dia_matvec() {
        let lower = vec![-1.0, -1.0];
        let main = vec![2.0, 2.0, 2.0];
        let upper = vec![-1.0, -1.0];
        let dia = EnhancedDia::tridiagonal(&lower, &main, &upper).expect("should succeed");

        let x = vec![1.0, 2.0, 3.0];
        let y = dia.matvec(&x).expect("matvec");

        // y[0] = 2*1 + (-1)*2 = 0
        // y[1] = (-1)*1 + 2*2 + (-1)*3 = 0
        // y[2] = (-1)*2 + 2*3 = 4
        assert_relative_eq!(y[0], 0.0);
        assert_relative_eq!(y[1], 0.0);
        assert_relative_eq!(y[2], 4.0);
    }

    #[test]
    fn test_tridiagonal_solve() {
        // 2 -1  0
        // -1  2 -1
        // 0 -1  2
        let lower = vec![-1.0, -1.0];
        let main = vec![2.0, 2.0, 2.0];
        let upper = vec![-1.0, -1.0];
        let b = vec![1.0, 0.0, 1.0];

        let x = tridiagonal_solve(&lower, &main, &upper, &b).expect("solve");

        // Verify: A x = b
        let dia = EnhancedDia::tridiagonal(&lower, &main, &upper).expect("dia");
        let ax = dia.matvec(&x).expect("matvec");

        for i in 0..3 {
            assert_relative_eq!(ax[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_banded_solve_pentadiagonal() {
        // 5-point stencil (pentadiagonal):
        // [4, -1, -1,  0,  0]
        // [-1, 4,  0, -1,  0]
        // [-1, 0,  4,  0, -1]
        // [0, -1,  0,  4, -1]
        // [0,  0, -1, -1,  4]
        let n = 5;
        let max_dim = n;

        let mut d_m2 = vec![0.0f64; max_dim]; // offset -2
        let mut d_m1 = vec![0.0f64; max_dim]; // offset -1
        let mut d_0 = vec![0.0f64; max_dim]; // offset 0
        let mut d_p1 = vec![0.0f64; max_dim]; // offset +1
        let mut d_p2 = vec![0.0f64; max_dim]; // offset +2

        // Main diagonal
        for i in 0..n {
            d_0[i] = 4.0;
        }
        // offset -1
        d_m1[0] = -1.0; // a[1,0]
        d_m1[3] = -1.0; // a[4,3]
                        // offset +1
        d_p1[0] = -1.0; // a[0,1]
        d_p1[3] = -1.0; // a[3,4]
                        // offset -2
        d_m2[0] = -1.0; // a[2,0]
        d_m2[1] = -1.0; // a[3,1]
        d_m2[2] = -1.0; // a[4,2]
                        // offset +2
        d_p2[0] = -1.0; // a[0,2]
        d_p2[1] = -1.0; // a[1,3]
        d_p2[2] = -1.0; // a[2,4]

        let dia = EnhancedDia::new(
            vec![d_m2, d_m1, d_0, d_p1, d_p2],
            vec![-2, -1, 0, 1, 2],
            n,
            n,
        )
        .expect("dia");

        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = banded_solve(&dia, &b).expect("banded_solve");

        // Verify: A x = b
        let ax = dia.matvec(&x).expect("matvec");
        for i in 0..n {
            assert_relative_eq!(ax[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_enhanced_dia_csr_roundtrip() {
        let lower = vec![-1.0f64, -1.0];
        let main = vec![2.0, 3.0, 4.0];
        let upper = vec![0.5, 0.5];
        let dia = EnhancedDia::tridiagonal(&lower, &main, &upper).expect("dia");

        let csr = dia.to_csr().expect("to_csr");
        let dia2 = EnhancedDia::from_csr(&csr).expect("from_csr");

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(dia.get(i, j), dia2.get(i, j), epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_enhanced_dia_csc_conversion() {
        let lower = vec![-1.0f64, -1.0];
        let main = vec![2.0, 2.0, 2.0];
        let upper = vec![-1.0, -1.0];
        let dia = EnhancedDia::tridiagonal(&lower, &main, &upper).expect("dia");

        let csc = dia.to_csc().expect("to_csc");
        assert_eq!(csc.shape(), (3, 3));
        assert_eq!(csc.nnz(), 7); // 3 diag + 2 sub + 2 super

        let dense_dia = dia.to_dense();
        let dense_csc = csc.to_array();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(dense_dia[[i, j]], dense_csc[[i, j]], epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_enhanced_dia_add() {
        let lower_a = vec![1.0f64, 1.0];
        let main_a = vec![2.0, 2.0, 2.0];
        let upper_a = vec![1.0, 1.0];
        let a = EnhancedDia::tridiagonal(&lower_a, &main_a, &upper_a).expect("a");

        let lower_b = vec![0.5, 0.5];
        let main_b = vec![1.0, 1.0, 1.0];
        let upper_b = vec![0.5, 0.5];
        let b_mat = EnhancedDia::tridiagonal(&lower_b, &main_b, &upper_b).expect("b");

        let c = a.add(&b_mat).expect("add");

        assert_relative_eq!(c.get(0, 0), 3.0);
        assert_relative_eq!(c.get(0, 1), 1.5);
        assert_relative_eq!(c.get(1, 0), 1.5);
        assert_relative_eq!(c.get(1, 1), 3.0);
        assert_relative_eq!(c.get(2, 2), 3.0);
    }

    #[test]
    fn test_enhanced_dia_scale() {
        let main = vec![2.0f64, 3.0, 4.0];
        let lower = vec![0.0f64, 0.0];
        let upper = vec![0.0f64, 0.0];
        let dia = EnhancedDia::tridiagonal(&lower, &main, &upper).expect("dia");
        let scaled = dia.scale(0.5);

        assert_relative_eq!(scaled.get(0, 0), 1.0);
        assert_relative_eq!(scaled.get(1, 1), 1.5);
        assert_relative_eq!(scaled.get(2, 2), 2.0);
    }

    #[test]
    fn test_banded_solve_diagonal() {
        // Diagonal system
        let n = 4;
        let max_dim = n;
        let mut d = vec![0.0f64; max_dim];
        d[0] = 2.0;
        d[1] = 3.0;
        d[2] = 4.0;
        d[3] = 5.0;

        let dia = EnhancedDia::new(vec![d], vec![0], n, n).expect("dia");
        let b = vec![4.0, 9.0, 12.0, 25.0];
        let x = banded_solve(&dia, &b).expect("solve");

        assert_relative_eq!(x[0], 2.0, epsilon = 1e-12);
        assert_relative_eq!(x[1], 3.0, epsilon = 1e-12);
        assert_relative_eq!(x[2], 3.0, epsilon = 1e-12);
        assert_relative_eq!(x[3], 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_enhanced_dia_to_dense() {
        let lower = vec![-1.0f64, -1.0];
        let main = vec![2.0, 2.0, 2.0];
        let upper = vec![-1.0, -1.0];
        let dia = EnhancedDia::tridiagonal(&lower, &main, &upper).expect("dia");

        let dense = dia.to_dense();
        assert_relative_eq!(dense[[0, 0]], 2.0);
        assert_relative_eq!(dense[[0, 1]], -1.0);
        assert_relative_eq!(dense[[1, 0]], -1.0);
        assert_relative_eq!(dense[[1, 1]], 2.0);
        assert_relative_eq!(dense[[2, 2]], 2.0);
        assert_relative_eq!(dense[[0, 2]], 0.0);
    }
}
