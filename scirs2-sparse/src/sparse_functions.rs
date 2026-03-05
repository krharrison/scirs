//! Sparse matrix utility functions
//!
//! This module provides convenience functions that return concrete typed sparse
//! matrices (`CsrArray<T>`) instead of trait objects, plus additional utility
//! functions for common sparse matrix constructions.
//!
//! Functions provided:
//! - `sparse_eye(n)` - Sparse identity matrix
//! - `sparse_random(m, n, density)` - Random sparse matrix
//! - `sparse_kron(A, B)` - Kronecker product
//! - `sparse_hstack(arrays)` - Horizontal stacking
//! - `sparse_vstack(arrays)` - Vertical stacking
//! - `sparse_block_diag(arrays)` - Block diagonal construction
//! - `sparse_diags(diags, offsets, shape)` - Construct from diagonals

use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::numeric::{Float, SparseElement};
use std::fmt::Debug;
use std::ops::Div;

/// Create a sparse identity matrix of size n x n in CSR format.
///
/// # Arguments
/// * `n` - Matrix dimension
///
/// # Examples
/// ```
/// use scirs2_sparse::sparse_functions::sparse_eye;
/// use scirs2_sparse::sparray::SparseArray;
///
/// let eye = sparse_eye::<f64>(3).expect("should succeed");
/// assert_eq!(eye.shape(), (3, 3));
/// assert_eq!(eye.nnz(), 3);
/// assert_eq!(eye.get(0, 0), 1.0);
/// assert_eq!(eye.get(0, 1), 0.0);
/// ```
pub fn sparse_eye<T>(n: usize) -> SparseResult<CsrArray<T>>
where
    T: Float + SparseElement + Div<Output = T> + 'static,
{
    if n == 0 {
        return Err(SparseError::ValueError(
            "Matrix dimension must be positive".to_string(),
        ));
    }

    let rows: Vec<usize> = (0..n).collect();
    let cols: Vec<usize> = (0..n).collect();
    let data: Vec<T> = vec![T::sparse_one(); n];

    CsrArray::from_triplets(&rows, &cols, &data, (n, n), true)
}

/// Create a rectangular sparse identity-like matrix of size m x n in CSR format.
///
/// Places 1s on the main diagonal (the first min(m, n) diagonal entries).
///
/// # Arguments
/// * `m` - Number of rows
/// * `n` - Number of columns
pub fn sparse_eye_rect<T>(m: usize, n: usize) -> SparseResult<CsrArray<T>>
where
    T: Float + SparseElement + Div<Output = T> + 'static,
{
    if m == 0 || n == 0 {
        return Err(SparseError::ValueError(
            "Matrix dimensions must be positive".to_string(),
        ));
    }

    let diag_len = m.min(n);
    let rows: Vec<usize> = (0..diag_len).collect();
    let cols: Vec<usize> = (0..diag_len).collect();
    let data: Vec<T> = vec![T::sparse_one(); diag_len];

    CsrArray::from_triplets(&rows, &cols, &data, (m, n), true)
}

/// Create a random sparse matrix in CSR format.
///
/// Generates a sparse matrix where approximately `density * m * n` elements are
/// non-zero, with values drawn uniformly from [0, 1).
///
/// # Arguments
/// * `m` - Number of rows
/// * `n` - Number of columns
/// * `density` - Density of non-zero elements (0.0 to 1.0)
/// * `seed` - Optional random seed for reproducibility
pub fn sparse_random(
    m: usize,
    n: usize,
    density: f64,
    seed: Option<u64>,
) -> SparseResult<CsrArray<f64>> {
    if m == 0 || n == 0 {
        return Err(SparseError::ValueError(
            "Matrix dimensions must be positive".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&density) {
        return Err(SparseError::ValueError(
            "Density must be between 0.0 and 1.0".to_string(),
        ));
    }

    let total_elements = m * n;
    let nnz_target = (density * total_elements as f64).round() as usize;

    if nnz_target == 0 {
        // Return empty sparse matrix
        let rows: Vec<usize> = Vec::new();
        let cols: Vec<usize> = Vec::new();
        let data: Vec<f64> = Vec::new();
        return CsrArray::from_triplets(&rows, &cols, &data, (m, n), false);
    }

    use scirs2_core::random::{Rng, SeedableRng};
    let mut rng = match seed {
        Some(s) => scirs2_core::random::StdRng::seed_from_u64(s),
        None => scirs2_core::random::StdRng::seed_from_u64(42),
    };

    // Generate random positions
    let mut positions: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

    // For low density, random sampling is efficient
    // For high density, it's better to sample from all positions and reject
    if density < 0.5 {
        while positions.len() < nnz_target {
            let r = rng.random_range(0..m);
            let c = rng.random_range(0..n);
            positions.insert((r, c));
        }
    } else {
        // Generate all positions and shuffle-select
        let mut all_positions: Vec<(usize, usize)> = Vec::with_capacity(total_elements);
        for r in 0..m {
            for c in 0..n {
                all_positions.push((r, c));
            }
        }
        // Partial Fisher-Yates shuffle
        for i in 0..nnz_target.min(all_positions.len()) {
            let j = rng.random_range(i..all_positions.len());
            all_positions.swap(i, j);
            positions.insert(all_positions[i]);
        }
    }

    let mut rows: Vec<usize> = Vec::with_capacity(nnz_target);
    let mut cols: Vec<usize> = Vec::with_capacity(nnz_target);
    let mut data: Vec<f64> = Vec::with_capacity(nnz_target);

    for (r, c) in positions {
        rows.push(r);
        cols.push(c);
        data.push(rng.random::<f64>());
    }

    CsrArray::from_triplets(&rows, &cols, &data, (m, n), false)
}

/// Compute the Kronecker product of two sparse matrices.
///
/// If A is m x n and B is p x q, the result is (m*p) x (n*q).
///
/// # Arguments
/// * `a` - First sparse matrix
/// * `b` - Second sparse matrix
///
/// # Examples
/// ```
/// use scirs2_sparse::sparse_functions::{sparse_eye, sparse_kron};
/// use scirs2_sparse::sparray::SparseArray;
///
/// let i2 = sparse_eye::<f64>(2).expect("eye");
/// let result = sparse_kron(&i2, &i2).expect("kron");
/// assert_eq!(result.shape(), (4, 4));
/// assert_eq!(result.nnz(), 4);
/// ```
pub fn sparse_kron<T>(a: &CsrArray<T>, b: &CsrArray<T>) -> SparseResult<CsrArray<T>>
where
    T: Float + SparseElement + Div<Output = T> + Debug + Copy + 'static,
{
    let (m, n) = a.shape();
    let (p, q) = b.shape();
    let result_rows = m * p;
    let result_cols = n * q;

    let (a_rows, a_cols, a_vals) = a.find();
    let (b_rows, b_cols, b_vals) = b.find();

    let a_nnz = a_vals.len();
    let b_nnz = b_vals.len();

    let mut rows = Vec::with_capacity(a_nnz * b_nnz);
    let mut cols = Vec::with_capacity(a_nnz * b_nnz);
    let mut data = Vec::with_capacity(a_nnz * b_nnz);

    for i in 0..a_nnz {
        let ar = a_rows[i];
        let ac = a_cols[i];
        let av = a_vals[i];

        for j in 0..b_nnz {
            let br = b_rows[j];
            let bc = b_cols[j];
            let bv = b_vals[j];

            rows.push(ar * p + br);
            cols.push(ac * q + bc);
            data.push(av * bv);
        }
    }

    CsrArray::from_triplets(&rows, &cols, &data, (result_rows, result_cols), false)
}

/// Stack sparse matrices horizontally (column-wise).
///
/// All matrices must have the same number of rows.
///
/// # Arguments
/// * `arrays` - Slice of references to CsrArray matrices
pub fn sparse_hstack<T>(arrays: &[&CsrArray<T>]) -> SparseResult<CsrArray<T>>
where
    T: Float + SparseElement + Div<Output = T> + Debug + Copy + 'static,
{
    if arrays.is_empty() {
        return Err(SparseError::ValueError(
            "Cannot stack empty list of arrays".to_string(),
        ));
    }

    let m = arrays[0].shape().0;
    for (idx, &arr) in arrays.iter().enumerate().skip(1) {
        if arr.shape().0 != m {
            return Err(SparseError::DimensionMismatch {
                expected: m,
                found: arr.shape().0,
            });
        }
    }

    let total_cols: usize = arrays.iter().map(|a| a.shape().1).sum();

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    let mut col_offset = 0usize;
    for &arr in arrays {
        let (ar, ac, av) = arr.find();
        for i in 0..av.len() {
            rows.push(ar[i]);
            cols.push(ac[i] + col_offset);
            data.push(av[i]);
        }
        col_offset += arr.shape().1;
    }

    CsrArray::from_triplets(&rows, &cols, &data, (m, total_cols), false)
}

/// Stack sparse matrices vertically (row-wise).
///
/// All matrices must have the same number of columns.
///
/// # Arguments
/// * `arrays` - Slice of references to CsrArray matrices
pub fn sparse_vstack<T>(arrays: &[&CsrArray<T>]) -> SparseResult<CsrArray<T>>
where
    T: Float + SparseElement + Div<Output = T> + Debug + Copy + 'static,
{
    if arrays.is_empty() {
        return Err(SparseError::ValueError(
            "Cannot stack empty list of arrays".to_string(),
        ));
    }

    let n = arrays[0].shape().1;
    for (idx, &arr) in arrays.iter().enumerate().skip(1) {
        if arr.shape().1 != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: arr.shape().1,
            });
        }
    }

    let total_rows: usize = arrays.iter().map(|a| a.shape().0).sum();

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    let mut row_offset = 0usize;
    for &arr in arrays {
        let (ar, ac, av) = arr.find();
        for i in 0..av.len() {
            rows.push(ar[i] + row_offset);
            cols.push(ac[i]);
            data.push(av[i]);
        }
        row_offset += arr.shape().0;
    }

    CsrArray::from_triplets(&rows, &cols, &data, (total_rows, n), false)
}

/// Construct a block diagonal sparse matrix from a list of sub-matrices.
///
/// The resulting matrix has shape (sum of rows, sum of cols) where each
/// sub-matrix appears along the diagonal.
///
/// # Arguments
/// * `arrays` - Slice of references to CsrArray matrices
///
/// # Examples
/// ```
/// use scirs2_sparse::sparse_functions::{sparse_eye, sparse_block_diag};
/// use scirs2_sparse::sparray::SparseArray;
///
/// let a = sparse_eye::<f64>(2).expect("eye");
/// let b = sparse_eye::<f64>(3).expect("eye");
/// let bd = sparse_block_diag(&[&a, &b]).expect("block_diag");
/// assert_eq!(bd.shape(), (5, 5));
/// assert_eq!(bd.nnz(), 5);
/// ```
pub fn sparse_block_diag<T>(arrays: &[&CsrArray<T>]) -> SparseResult<CsrArray<T>>
where
    T: Float + SparseElement + Div<Output = T> + Debug + Copy + 'static,
{
    if arrays.is_empty() {
        return Err(SparseError::ValueError(
            "Cannot create block diagonal from empty list".to_string(),
        ));
    }

    let total_rows: usize = arrays.iter().map(|a| a.shape().0).sum();
    let total_cols: usize = arrays.iter().map(|a| a.shape().1).sum();

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    let mut row_offset = 0usize;
    let mut col_offset = 0usize;

    for &arr in arrays {
        let (ar, ac, av) = arr.find();
        for i in 0..av.len() {
            rows.push(ar[i] + row_offset);
            cols.push(ac[i] + col_offset);
            data.push(av[i]);
        }
        row_offset += arr.shape().0;
        col_offset += arr.shape().1;
    }

    CsrArray::from_triplets(&rows, &cols, &data, (total_rows, total_cols), false)
}

/// Construct a sparse matrix from diagonals.
///
/// # Arguments
/// * `diags` - Slice of diagonal vectors
/// * `offsets` - Diagonal offsets (0 = main, positive = super, negative = sub)
/// * `shape` - (nrows, ncols)
///
/// # Examples
/// ```
/// use scirs2_sparse::sparse_functions::sparse_diags;
/// use scirs2_sparse::sparray::SparseArray;
///
/// let main = vec![2.0, 2.0, 2.0];
/// let upper = vec![-1.0, -1.0];
/// let lower = vec![-1.0, -1.0];
/// let a = sparse_diags(&[&lower, &main, &upper], &[-1, 0, 1], (3, 3)).expect("diags");
/// assert_eq!(a.shape(), (3, 3));
/// assert_eq!(a.get(0, 0), 2.0);
/// assert_eq!(a.get(0, 1), -1.0);
/// assert_eq!(a.get(1, 0), -1.0);
/// ```
pub fn sparse_diags<T>(
    diags: &[&[T]],
    offsets: &[isize],
    shape: (usize, usize),
) -> SparseResult<CsrArray<T>>
where
    T: Float + SparseElement + Div<Output = T> + Debug + Copy + 'static,
{
    if diags.len() != offsets.len() {
        return Err(SparseError::DimensionMismatch {
            expected: offsets.len(),
            found: diags.len(),
        });
    }

    let (nrows, ncols) = shape;
    if nrows == 0 || ncols == 0 {
        return Err(SparseError::ValueError(
            "Matrix dimensions must be positive".to_string(),
        ));
    }

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for (d, &offset) in offsets.iter().enumerate() {
        let diag = diags[d];
        if offset >= 0 {
            let off = offset as usize;
            let diag_len = nrows.min(ncols.saturating_sub(off));
            if diag.len() < diag_len {
                return Err(SparseError::DimensionMismatch {
                    expected: diag_len,
                    found: diag.len(),
                });
            }
            for k in 0..diag_len {
                let v = diag[k];
                if !SparseElement::is_zero(&v) {
                    rows.push(k);
                    cols.push(k + off);
                    data.push(v);
                }
            }
        } else {
            let off = (-offset) as usize;
            let diag_len = ncols.min(nrows.saturating_sub(off));
            if diag.len() < diag_len {
                return Err(SparseError::DimensionMismatch {
                    expected: diag_len,
                    found: diag.len(),
                });
            }
            for k in 0..diag_len {
                let v = diag[k];
                if !SparseElement::is_zero(&v) {
                    rows.push(k + off);
                    cols.push(k);
                    data.push(v);
                }
            }
        }
    }

    CsrArray::from_triplets(&rows, &cols, &data, shape, false)
}

/// Create a sparse matrix with given values on the specified diagonal.
///
/// # Arguments
/// * `diag` - Values for the diagonal
/// * `offset` - Diagonal offset (0 = main, positive = super, negative = sub)
/// * `shape` - (nrows, ncols). If None, inferred from diag length and offset.
pub fn sparse_diag_matrix<T>(
    diag: &[T],
    offset: isize,
    shape: Option<(usize, usize)>,
) -> SparseResult<CsrArray<T>>
where
    T: Float + SparseElement + Div<Output = T> + Debug + Copy + 'static,
{
    let n = diag.len();
    let (nrows, ncols) = shape.unwrap_or_else(|| {
        if offset >= 0 {
            (n, n + offset as usize)
        } else {
            (n + (-offset) as usize, n)
        }
    });

    sparse_diags(&[diag], &[offset], (nrows, ncols))
}

/// Kronecker sum of two sparse matrices: A (x) I_q + I_p (x) B
///
/// If A is p x p and B is q x q, the result is (p*q) x (p*q).
///
/// # Arguments
/// * `a` - First sparse matrix (must be square)
/// * `b` - Second sparse matrix (must be square)
pub fn sparse_kronsum<T>(a: &CsrArray<T>, b: &CsrArray<T>) -> SparseResult<CsrArray<T>>
where
    T: Float + SparseElement + Div<Output = T> + Debug + Copy + 'static,
{
    let (p, pa) = a.shape();
    let (q, qb) = b.shape();

    if p != pa {
        return Err(SparseError::ValueError(
            "First matrix must be square for Kronecker sum".to_string(),
        ));
    }
    if q != qb {
        return Err(SparseError::ValueError(
            "Second matrix must be square for Kronecker sum".to_string(),
        ));
    }

    let iq = sparse_eye::<T>(q)?;
    let ip = sparse_eye::<T>(p)?;

    let a_kron_iq = sparse_kron(a, &iq)?;
    let ip_kron_b = sparse_kron(&ip, b)?;

    // Add the two Kronecker products
    let result = a_kron_iq.add(&ip_kron_b)?;

    // Convert result back to CsrArray
    let (rr, rc, rv) = result.find();
    let rows_vec: Vec<usize> = rr.to_vec();
    let cols_vec: Vec<usize> = rc.to_vec();
    let vals_vec: Vec<T> = rv.to_vec();
    let shape = result.shape();

    CsrArray::from_triplets(&rows_vec, &cols_vec, &vals_vec, shape, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sparse_eye() {
        let eye = sparse_eye::<f64>(4).expect("eye");
        assert_eq!(eye.shape(), (4, 4));
        assert_eq!(eye.nnz(), 4);
        for i in 0..4 {
            assert_relative_eq!(eye.get(i, i), 1.0);
            if i > 0 {
                assert_relative_eq!(eye.get(i, i - 1), 0.0);
            }
        }
    }

    #[test]
    fn test_sparse_eye_rect() {
        let eye = sparse_eye_rect::<f64>(3, 5).expect("eye_rect");
        assert_eq!(eye.shape(), (3, 5));
        assert_eq!(eye.nnz(), 3);
        for i in 0..3 {
            assert_relative_eq!(eye.get(i, i), 1.0);
        }
        assert_relative_eq!(eye.get(0, 3), 0.0);
    }

    #[test]
    fn test_sparse_random() {
        let mat = sparse_random(10, 10, 0.3, Some(42)).expect("random");
        assert_eq!(mat.shape(), (10, 10));
        let nnz = mat.nnz();
        // Should have approximately 30 non-zeros (10*10*0.3)
        assert!(nnz > 10 && nnz < 50);
    }

    #[test]
    fn test_sparse_random_empty() {
        let mat = sparse_random(5, 5, 0.0, Some(1)).expect("random empty");
        assert_eq!(mat.nnz(), 0);
    }

    #[test]
    fn test_sparse_random_full() {
        let mat = sparse_random(3, 3, 1.0, Some(1)).expect("random full");
        assert_eq!(mat.shape(), (3, 3));
        assert_eq!(mat.nnz(), 9);
    }

    #[test]
    fn test_sparse_kron_identity() {
        let i2 = sparse_eye::<f64>(2).expect("eye");
        let result = sparse_kron(&i2, &i2).expect("kron");
        assert_eq!(result.shape(), (4, 4));
        assert_eq!(result.nnz(), 4);

        // Should be 4x4 identity
        for i in 0..4 {
            assert_relative_eq!(result.get(i, i), 1.0);
            for j in 0..4 {
                if i != j {
                    assert_relative_eq!(result.get(i, j), 0.0);
                }
            }
        }
    }

    #[test]
    fn test_sparse_kron_general() {
        // A = [[1, 2], [3, 4]], B = [[0, 5], [6, 7]]
        let a = CsrArray::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 2.0, 3.0, 4.0],
            (2, 2),
            false,
        )
        .expect("a");

        let b = CsrArray::from_triplets(&[0, 1, 1], &[1, 0, 1], &[5.0, 6.0, 7.0], (2, 2), false)
            .expect("b");

        let result = sparse_kron(&a, &b).expect("kron");
        assert_eq!(result.shape(), (4, 4));

        // kron(A, B) =
        // [1*[0,5;6,7]  2*[0,5;6,7]]
        // [3*[0,5;6,7]  4*[0,5;6,7]]
        //
        // = [0  5  0 10]
        //   [6  7 12 14]
        //   [0 15  0 20]
        //   [18 21 24 28]
        assert_relative_eq!(result.get(0, 0), 0.0);
        assert_relative_eq!(result.get(0, 1), 5.0);
        assert_relative_eq!(result.get(0, 2), 0.0);
        assert_relative_eq!(result.get(0, 3), 10.0);
        assert_relative_eq!(result.get(1, 0), 6.0);
        assert_relative_eq!(result.get(3, 3), 28.0);
    }

    #[test]
    fn test_sparse_hstack() {
        let a =
            CsrArray::from_triplets(&[0, 1], &[0, 1], &[1.0f64, 2.0], (2, 2), false).expect("a");

        let b =
            CsrArray::from_triplets(&[0, 1], &[0, 0], &[3.0f64, 4.0], (2, 1), false).expect("b");

        let result = sparse_hstack(&[&a, &b]).expect("hstack");
        assert_eq!(result.shape(), (2, 3));
        assert_relative_eq!(result.get(0, 0), 1.0);
        assert_relative_eq!(result.get(1, 1), 2.0);
        assert_relative_eq!(result.get(0, 2), 3.0);
        assert_relative_eq!(result.get(1, 2), 4.0);
    }

    #[test]
    fn test_sparse_vstack() {
        let a =
            CsrArray::from_triplets(&[0, 0], &[0, 1], &[1.0f64, 2.0], (1, 3), false).expect("a");

        let b =
            CsrArray::from_triplets(&[0, 1], &[1, 2], &[3.0f64, 4.0], (2, 3), false).expect("b");

        let result = sparse_vstack(&[&a, &b]).expect("vstack");
        assert_eq!(result.shape(), (3, 3));
        assert_relative_eq!(result.get(0, 0), 1.0);
        assert_relative_eq!(result.get(0, 1), 2.0);
        assert_relative_eq!(result.get(1, 1), 3.0);
        assert_relative_eq!(result.get(2, 2), 4.0);
    }

    #[test]
    fn test_sparse_block_diag() {
        let a = sparse_eye::<f64>(2).expect("eye");
        let b = CsrArray::from_triplets(
            &[0, 0, 1, 1, 2, 2],
            &[0, 1, 0, 1, 0, 1],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            (3, 2),
            false,
        )
        .expect("b");

        let result = sparse_block_diag(&[&a, &b]).expect("block_diag");
        assert_eq!(result.shape(), (5, 4));

        // Top-left 2x2 is identity
        assert_relative_eq!(result.get(0, 0), 1.0);
        assert_relative_eq!(result.get(1, 1), 1.0);
        assert_relative_eq!(result.get(0, 1), 0.0);
        assert_relative_eq!(result.get(1, 0), 0.0);

        // Bottom-right 3x2 is b
        assert_relative_eq!(result.get(2, 2), 1.0);
        assert_relative_eq!(result.get(2, 3), 2.0);
        assert_relative_eq!(result.get(4, 3), 6.0);

        // Off-block should be zero
        assert_relative_eq!(result.get(0, 2), 0.0);
        assert_relative_eq!(result.get(2, 0), 0.0);
    }

    #[test]
    fn test_sparse_diags() {
        let main = vec![2.0f64, 2.0, 2.0];
        let upper = vec![-1.0f64, -1.0];
        let lower = vec![-1.0f64, -1.0];

        let a =
            sparse_diags(&[&lower[..], &main[..], &upper[..]], &[-1, 0, 1], (3, 3)).expect("diags");

        assert_eq!(a.shape(), (3, 3));
        assert_relative_eq!(a.get(0, 0), 2.0);
        assert_relative_eq!(a.get(0, 1), -1.0);
        assert_relative_eq!(a.get(1, 0), -1.0);
        assert_relative_eq!(a.get(1, 1), 2.0);
        assert_relative_eq!(a.get(1, 2), -1.0);
        assert_relative_eq!(a.get(2, 1), -1.0);
        assert_relative_eq!(a.get(2, 2), 2.0);
        assert_relative_eq!(a.get(0, 2), 0.0);
    }

    #[test]
    fn test_sparse_diag_matrix() {
        let diag = vec![3.0f64, 5.0, 7.0];
        let m = sparse_diag_matrix(&diag, 0, None).expect("diag_matrix");
        assert_eq!(m.shape(), (3, 3));
        assert_relative_eq!(m.get(0, 0), 3.0);
        assert_relative_eq!(m.get(1, 1), 5.0);
        assert_relative_eq!(m.get(2, 2), 7.0);

        // Super diagonal
        let sd = vec![1.0f64, 2.0];
        let m2 = sparse_diag_matrix(&sd, 1, None).expect("super_diag");
        assert_eq!(m2.shape(), (2, 3));
        assert_relative_eq!(m2.get(0, 1), 1.0);
        assert_relative_eq!(m2.get(1, 2), 2.0);
    }

    #[test]
    fn test_sparse_kronsum() {
        // A = [[1, 0], [0, 2]], B = [[3, 0], [0, 4]]
        let a =
            CsrArray::from_triplets(&[0, 1], &[0, 1], &[1.0f64, 2.0], (2, 2), false).expect("a");

        let b =
            CsrArray::from_triplets(&[0, 1], &[0, 1], &[3.0f64, 4.0], (2, 2), false).expect("b");

        let result = sparse_kronsum(&a, &b).expect("kronsum");
        assert_eq!(result.shape(), (4, 4));

        // A (x) I2 + I2 (x) B:
        // diag(A (x) I2) = [1,1,2,2]
        // diag(I2 (x) B) = [3,4,3,4]
        // diagonal = [4,5,5,6]
        assert_relative_eq!(result.get(0, 0), 4.0);
        assert_relative_eq!(result.get(1, 1), 5.0);
        assert_relative_eq!(result.get(2, 2), 5.0);
        assert_relative_eq!(result.get(3, 3), 6.0);
    }
}
