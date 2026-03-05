//! Sparse matrix format conversions and low-level CSR operations
//!
//! This module provides:
//!
//! - **Format conversions**: COO ↔ CSR, CSR ↔ CSC, Dense ↔ CSR.
//! - **Core CSR kernels**: matrix-vector multiply, matrix-matrix multiply,
//!   transpose, diagonal extraction.
//! - **Graph utility**: adjacency matrix → graph Laplacian.
//!
//! All functions work on raw slices / `Array2<f64>` and return owned `Vec`
//! or `Array2` – no heap-allocated sparse struct is required.  This makes
//! the module useful as a dependency-free building block for higher-level
//! algorithms.
//!
//! # Conventions
//!
//! - **CSR** (Compressed Sparse Row): `(row_ptrs, col_indices, values)`.
//!   `row_ptrs` has length `nrows + 1`; `row_ptrs[i]..row_ptrs[i+1]` spans
//!   the non-zeros of row `i`.
//! - **CSC** (Compressed Sparse Column): `(col_ptrs, row_indices, values)`.
//!   Analogous to CSR with rows and columns swapped.
//! - **COO** (Coordinate): three parallel arrays `(row_indices, col_indices,
//!   values)` of length `nnz`.  Duplicate entries are summed during
//!   conversion.

use crate::error::{SparseError, SparseResult};
use scirs2_core::ndarray::{Array2, ArrayView2};

// ---------------------------------------------------------------------------
// CSR ↔ CSC
// ---------------------------------------------------------------------------

/// Convert a CSR matrix to CSC format.
///
/// # Arguments
///
/// * `row_ptrs`    – CSR row pointer array of length `nrows + 1`.
/// * `col_indices` – CSR column index array.
/// * `values`      – CSR value array.
/// * `nrows`       – Number of rows.
/// * `ncols`       – Number of columns.
///
/// # Returns
///
/// `(col_ptrs, row_indices, values)` – CSC representation.
///
/// # Example
///
/// ```
/// use scirs2_sparse::sparse_format::csr_to_csc;
///
/// let row_ptrs    = vec![0, 2, 3, 5];
/// let col_indices = vec![0, 2, 2, 0, 1];
/// let values      = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let (cp, ri, vv) = csr_to_csc(&row_ptrs, &col_indices, &values, 3, 3);
/// assert_eq!(cp.len(), 4); // ncols + 1
/// ```
pub fn csr_to_csc(
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[f64],
    nrows: usize,
    ncols: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let nnz = values.len();

    // Count entries per column.
    let mut col_counts = vec![0usize; ncols];
    for &c in col_indices {
        col_counts[c] += 1;
    }

    // Build col_ptrs (prefix sum).
    let mut col_ptrs = vec![0usize; ncols + 1];
    for c in 0..ncols {
        col_ptrs[c + 1] = col_ptrs[c] + col_counts[c];
    }

    // Fill row_indices and values.
    let mut row_indices_csc = vec![0usize; nnz];
    let mut values_csc = vec![0.0f64; nnz];
    let mut cur = col_ptrs[..ncols].to_vec(); // write cursors per column

    for i in 0..nrows {
        for pos in row_ptrs[i]..row_ptrs[i + 1] {
            let c = col_indices[pos];
            let dst = cur[c];
            row_indices_csc[dst] = i;
            values_csc[dst] = values[pos];
            cur[c] += 1;
        }
    }

    (col_ptrs, row_indices_csc, values_csc)
}

/// Convert a CSC matrix to CSR format.
///
/// # Arguments
///
/// * `col_ptrs`    – CSC column pointer array of length `ncols + 1`.
/// * `row_indices` – CSC row index array.
/// * `values`      – CSC value array.
/// * `nrows`       – Number of rows.
/// * `ncols`       – Number of columns.
///
/// # Returns
///
/// `(row_ptrs, col_indices, values)` – CSR representation.
pub fn csc_to_csr(
    col_ptrs: &[usize],
    row_indices: &[usize],
    values: &[f64],
    nrows: usize,
    ncols: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    // CSC → CSR is the same algorithm as CSR → CSC with rows/cols swapped.
    // Re-use csr_to_csc treating the "CSC columns" as rows of the transposed
    // view.
    csr_to_csc(col_ptrs, row_indices, values, ncols, nrows)
}

// ---------------------------------------------------------------------------
// COO ↔ CSR
// ---------------------------------------------------------------------------

/// Convert COO (coordinate) format to CSR.
///
/// Duplicate `(row, col)` entries are **summed**.  The resulting CSR matrix
/// has sorted row order; within each row the column order follows the input
/// order (ties broken by input position).
///
/// # Arguments
///
/// * `row_indices` – COO row indices.
/// * `col_indices` – COO column indices.
/// * `values`      – COO values.
/// * `nrows`       – Number of rows in the matrix.
///
/// # Returns
///
/// `(row_ptrs, col_indices, values)` – CSR representation.
///
/// # Example
///
/// ```
/// use scirs2_sparse::sparse_format::coo_to_csr;
///
/// let rows = vec![0, 0, 1, 2, 2];
/// let cols = vec![0, 2, 2, 0, 1];
/// let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let (rp, ci, vv) = coo_to_csr(&rows, &cols, &vals, 3);
/// assert_eq!(rp.len(), 4); // nrows + 1
/// ```
pub fn coo_to_csr(
    row_indices: &[usize],
    col_indices: &[usize],
    values: &[f64],
    nrows: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let nnz = values.len();

    // Count entries per row (for allocation).
    let mut row_counts = vec![0usize; nrows];
    for &r in row_indices {
        if r < nrows {
            row_counts[r] += 1;
        }
    }

    // Build row_ptrs.
    let mut row_ptrs = vec![0usize; nrows + 1];
    for r in 0..nrows {
        row_ptrs[r + 1] = row_ptrs[r] + row_counts[r];
    }

    // Allocate output.
    let mut out_cols = vec![0usize; nnz];
    let mut out_vals = vec![0.0f64; nnz];
    let mut cur = row_ptrs[..nrows].to_vec();

    for k in 0..nnz {
        let r = row_indices[k];
        if r < nrows {
            let dst = cur[r];
            out_cols[dst] = col_indices[k];
            out_vals[dst] = values[k];
            cur[r] += 1;
        }
    }

    // Sum duplicate (row, col) entries.
    // Build a deduplicated version.
    let mut dedup_cols: Vec<usize> = Vec::with_capacity(nnz);
    let mut dedup_vals: Vec<f64> = Vec::with_capacity(nnz);
    let mut new_row_ptrs = vec![0usize; nrows + 1];

    for r in 0..nrows {
        let start = row_ptrs[r];
        let end = row_ptrs[r + 1];

        // Collect (col, val) pairs for this row, sort by col.
        let mut row_entries: Vec<(usize, f64)> = out_cols[start..end]
            .iter()
            .zip(out_vals[start..end].iter())
            .map(|(&c, &v)| (c, v))
            .collect();
        row_entries.sort_unstable_by_key(|&(c, _)| c);

        // Merge duplicates.
        let row_start = dedup_cols.len();
        for (c, v) in row_entries {
            if let Some(last) = dedup_cols.last().copied() {
                if last == c && dedup_cols.len() > row_start {
                    // Same column as previous entry in this row: sum.
                    if let Some(last_val) = dedup_vals.last_mut() {
                        *last_val += v;
                    }
                    continue;
                }
            }
            dedup_cols.push(c);
            dedup_vals.push(v);
        }
        new_row_ptrs[r + 1] = dedup_cols.len();
    }

    (new_row_ptrs, dedup_cols, dedup_vals)
}

/// Convert CSR to COO format.
///
/// # Arguments
///
/// * `row_ptrs`    – CSR row pointer array of length `nrows + 1`.
/// * `col_indices` – CSR column index array.
/// * `values`      – CSR value array.
///
/// # Returns
///
/// `(row_indices, col_indices, values)` – COO arrays.
pub fn csr_to_coo(
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[f64],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let nrows = row_ptrs.len().saturating_sub(1);
    let nnz = values.len();
    let mut row_indices = vec![0usize; nnz];
    for i in 0..nrows {
        for pos in row_ptrs[i]..row_ptrs[i + 1] {
            row_indices[pos] = i;
        }
    }
    (row_indices, col_indices.to_vec(), values.to_vec())
}

// ---------------------------------------------------------------------------
// Dense ↔ CSR
// ---------------------------------------------------------------------------

/// Convert a dense `Array2<f64>` to CSR format (ignoring exact zeros).
///
/// # Arguments
///
/// * `dense` – Row-major 2-D array.
///
/// # Returns
///
/// `(row_ptrs, col_indices, values)`.
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_sparse::sparse_format::dense_to_csr;
///
/// let d = Array2::<f64>::eye(3);
/// let (rp, ci, vv) = dense_to_csr(&d);
/// assert_eq!(vv.len(), 3);
/// ```
pub fn dense_to_csr(dense: &Array2<f64>) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let (nrows, _ncols) = dense.dim();
    let mut row_ptrs = vec![0usize; nrows + 1];
    let mut col_indices: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();

    for i in 0..nrows {
        for j in 0..dense.ncols() {
            let v = dense[[i, j]];
            if v != 0.0 {
                col_indices.push(j);
                values.push(v);
            }
        }
        row_ptrs[i + 1] = col_indices.len();
    }

    (row_ptrs, col_indices, values)
}

/// Convert CSR to a dense `Array2<f64>`.
///
/// # Arguments
///
/// * `row_ptrs`, `col_indices`, `values` – CSR arrays.
/// * `nrows`, `ncols` – Output matrix dimensions.
///
/// # Returns
///
/// Dense `Array2<f64>` of shape `(nrows, ncols)`.
pub fn csr_to_dense(
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[f64],
    nrows: usize,
    ncols: usize,
) -> Array2<f64> {
    let mut dense = Array2::<f64>::zeros((nrows, ncols));
    for i in 0..nrows {
        for pos in row_ptrs[i]..row_ptrs[i + 1] {
            dense[[i, col_indices[pos]]] += values[pos];
        }
    }
    dense
}

// ---------------------------------------------------------------------------
// CSR Matrix-Vector Multiply
// ---------------------------------------------------------------------------

/// Compute y = A x where A is given in CSR format.
///
/// # Arguments
///
/// * `row_ptrs`, `col_indices`, `values` – CSR arrays of A.
/// * `x` – Input vector of length `ncols`.
///
/// # Returns
///
/// Output vector of length `nrows`.
///
/// # Example
///
/// ```
/// use scirs2_sparse::sparse_format::csr_matvec;
///
/// let row_ptrs    = vec![0, 2, 3, 5];
/// let col_indices = vec![0, 2, 2, 0, 1];
/// let values      = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let x = vec![1.0, 2.0, 3.0];
/// let y = csr_matvec(&row_ptrs, &col_indices, &values, &x);
/// // row 0: 1*1 + 2*3 = 7
/// assert_eq!(y[0], 7.0);
/// ```
pub fn csr_matvec(
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[f64],
    x: &[f64],
) -> Vec<f64> {
    let nrows = row_ptrs.len().saturating_sub(1);
    let mut y = vec![0.0f64; nrows];
    for i in 0..nrows {
        let start = row_ptrs[i];
        let end = row_ptrs[i + 1];
        let mut acc = 0.0f64;
        for pos in start..end {
            acc += values[pos] * x[col_indices[pos]];
        }
        y[i] = acc;
    }
    y
}

// ---------------------------------------------------------------------------
// CSR Transpose
// ---------------------------------------------------------------------------

/// Compute the CSR representation of Aᵀ.
///
/// # Arguments
///
/// * `row_ptrs`, `col_indices`, `values` – CSR arrays of A (shape nrows × ncols).
/// * `nrows`, `ncols` – Dimensions of A.
///
/// # Returns
///
/// `(row_ptrs_t, col_indices_t, values_t)` – CSR of Aᵀ (shape ncols × nrows).
pub fn csr_transpose(
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[f64],
    nrows: usize,
    ncols: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    // Aᵀ in CSR == A in CSC.  Use the csr_to_csc function which produces
    // (col_ptrs, row_indices, values) of A = row_ptrs of Aᵀ.
    csr_to_csc(row_ptrs, col_indices, values, nrows, ncols)
}

// ---------------------------------------------------------------------------
// CSR Matrix-Matrix Multiply
// ---------------------------------------------------------------------------

/// Compute C = A * B for two sparse matrices in CSR format.
///
/// Uses the row-by-row "hash-map" accumulator approach: for each row i of A
/// we accumulate contributions from rows of B into a dense temporary, then
/// compress into the sparse result.
///
/// # Arguments
///
/// * `a_row_ptrs`, `a_col_indices`, `a_values` – CSR of A (nrows_a × ncols_a).
/// * `b_row_ptrs`, `b_col_indices`, `b_values` – CSR of B (ncols_a × ncols_b).
/// * `nrows_a`, `ncols_a`, `ncols_b` – Dimensions.
///
/// # Returns
///
/// `(c_row_ptrs, c_col_indices, c_values)` – CSR of C (nrows_a × ncols_b).
///
/// # Example
///
/// ```
/// use scirs2_sparse::sparse_format::{dense_to_csr, csr_matmat, csr_to_dense};
/// use scirs2_core::ndarray::Array2;
///
/// let a = Array2::<f64>::eye(3);
/// let (rpa, cia, va) = dense_to_csr(&a);
/// let (rpc, cic, vc) = csr_matmat(&rpa, &cia, &va, &rpa, &cia, &va, 3, 3, 3);
/// let c = csr_to_dense(&rpc, &cic, &vc, 3, 3);
/// assert_eq!(c, a);
/// ```
pub fn csr_matmat(
    a_row_ptrs: &[usize],
    a_col_indices: &[usize],
    a_values: &[f64],
    b_row_ptrs: &[usize],
    b_col_indices: &[usize],
    b_values: &[f64],
    nrows_a: usize,
    _ncols_a: usize,
    ncols_b: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut c_row_ptrs = vec![0usize; nrows_a + 1];
    let mut c_col_indices: Vec<usize> = Vec::new();
    let mut c_values: Vec<f64> = Vec::new();

    // Dense accumulator for one row of C.
    let mut acc = vec![0.0f64; ncols_b];
    // Flags to track which columns are non-zero (for efficient scatter).
    let mut touched: Vec<usize> = Vec::new();

    for i in 0..nrows_a {
        // Accumulate row i of C.
        for pos_a in a_row_ptrs[i]..a_row_ptrs[i + 1] {
            let k = a_col_indices[pos_a];
            let a_ik = a_values[pos_a];
            for pos_b in b_row_ptrs[k]..b_row_ptrs[k + 1] {
                let j = b_col_indices[pos_b];
                if acc[j] == 0.0 {
                    touched.push(j);
                }
                acc[j] += a_ik * b_values[pos_b];
            }
        }

        // Sort touched columns for canonical output order.
        touched.sort_unstable();

        for &j in &touched {
            let v = acc[j];
            if v != 0.0 {
                c_col_indices.push(j);
                c_values.push(v);
            }
            acc[j] = 0.0;
        }
        touched.clear();
        c_row_ptrs[i + 1] = c_col_indices.len();
    }

    (c_row_ptrs, c_col_indices, c_values)
}

// ---------------------------------------------------------------------------
// Sparse Diagonal Extraction
// ---------------------------------------------------------------------------

/// Extract the main diagonal of a sparse CSR matrix.
///
/// Returns a `Vec<f64>` of length `n = min(nrows, ncols)`.  Missing diagonal
/// entries are returned as `0.0`.
///
/// # Arguments
///
/// * `row_ptrs`, `col_indices`, `values` – CSR arrays.
/// * `n` – Size of the diagonal (= `min(nrows, ncols)`).
///
/// # Example
///
/// ```
/// use scirs2_sparse::sparse_format::csr_diagonal;
///
/// let row_ptrs    = vec![0, 2, 4, 5];
/// let col_indices = vec![0, 1, 0, 1, 2];
/// let values      = vec![3.0, 7.0, 2.0, 5.0, 1.0];
/// let diag = csr_diagonal(&row_ptrs, &col_indices, &values, 3);
/// assert_eq!(diag, vec![3.0, 5.0, 1.0]);
/// ```
pub fn csr_diagonal(
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[f64],
    n: usize,
) -> Vec<f64> {
    let mut diag = vec![0.0f64; n];
    let nrows = row_ptrs.len().saturating_sub(1).min(n);
    for i in 0..nrows {
        for pos in row_ptrs[i]..row_ptrs[i + 1] {
            if col_indices[pos] == i {
                diag[i] += values[pos];
            }
        }
    }
    diag
}

// ---------------------------------------------------------------------------
// Graph Laplacian
// ---------------------------------------------------------------------------

/// Convert a sparse adjacency matrix (CSR) to its graph Laplacian L = D − A.
///
/// For undirected graphs the adjacency matrix should be symmetric.  The
/// degree of node `i` is the sum of all non-zero entries in row `i`:
/// `D[i,i] = Σⱼ A[i,j]`.
///
/// The resulting Laplacian satisfies `L 1 = 0` (all row-sums are zero).
///
/// # Arguments
///
/// * `row_ptrs`, `col_indices`, `values` – CSR of the adjacency matrix A.
/// * `n` – Dimension (number of nodes).
///
/// # Returns
///
/// `(row_ptrs, col_indices, values)` – CSR of the Laplacian L.
///
/// # Example
///
/// ```
/// use scirs2_sparse::sparse_format::adjacency_to_laplacian;
///
/// // Triangle graph (3-cycle): A has 6 off-diagonal entries.
/// let row_ptrs    = vec![0, 2, 4, 6];
/// let col_indices = vec![1, 2,  0, 2,  0, 1];
/// let values      = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
/// let (lp, li, lv) = adjacency_to_laplacian(&row_ptrs, &col_indices, &values, 3);
/// // Each row of L sums to 0.
/// let y = scirs2_sparse::sparse_format::csr_matvec(&lp, &li, &lv, &[1.0, 1.0, 1.0]);
/// for &yi in &y { assert!((yi).abs() < 1e-12); }
/// ```
pub fn adjacency_to_laplacian(
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[f64],
    n: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    // Compute degree of each node = row sum of A.
    let mut degree = vec![0.0f64; n];
    let nrows = row_ptrs.len().saturating_sub(1).min(n);
    for i in 0..nrows {
        for pos in row_ptrs[i]..row_ptrs[i + 1] {
            degree[i] += values[pos];
        }
    }

    // L = D − A.  We build L row by row.
    // For each row i of A (with possibly missing diagonal) we output:
    //   - off-diagonal: -A[i,j] for j ≠ i
    //   - diagonal: degree[i] − A[i,i]
    let mut l_row_ptrs = vec![0usize; n + 1];
    let mut l_col_indices: Vec<usize> = Vec::new();
    let mut l_values: Vec<f64> = Vec::new();

    for i in 0..n {
        if i >= nrows {
            // Row not present in input (degree = 0); just a zero row.
            l_row_ptrs[i + 1] = l_col_indices.len();
            continue;
        }

        let start = row_ptrs[i];
        let end = row_ptrs[i + 1];

        // Check whether diagonal already exists in A.
        let mut diag_in_a = false;
        for pos in start..end {
            if col_indices[pos] == i {
                diag_in_a = true;
                break;
            }
        }

        // Collect (col, value) pairs for L row i.
        // Sort by column for canonical CSR.
        let mut row_entries: Vec<(usize, f64)> = Vec::with_capacity(end - start + 1);
        for pos in start..end {
            let j = col_indices[pos];
            let v = values[pos];
            if j == i {
                // Diagonal: degree[i] − A[i,i]
                row_entries.push((i, degree[i] - v));
            } else {
                // Off-diagonal: −A[i,j]
                row_entries.push((j, -v));
            }
        }
        if !diag_in_a {
            row_entries.push((i, degree[i]));
        }
        row_entries.sort_unstable_by_key(|&(c, _)| c);

        for (j, v) in row_entries {
            if v != 0.0 {
                l_col_indices.push(j);
                l_values.push(v);
            }
        }
        l_row_ptrs[i + 1] = l_col_indices.len();
    }

    (l_row_ptrs, l_col_indices, l_values)
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Return `Err` if the CSR row_ptrs / col_indices / values are inconsistent.
pub fn validate_csr(
    row_ptrs: &[usize],
    col_indices: &[usize],
    values: &[f64],
    nrows: usize,
    ncols: usize,
) -> SparseResult<()> {
    if row_ptrs.len() != nrows + 1 {
        return Err(SparseError::InconsistentData {
            reason: format!(
                "row_ptrs length {} != nrows+1 = {}",
                row_ptrs.len(),
                nrows + 1
            ),
        });
    }
    if col_indices.len() != values.len() {
        return Err(SparseError::InconsistentData {
            reason: format!(
                "col_indices length {} != values length {}",
                col_indices.len(),
                values.len()
            ),
        });
    }
    let nnz = *row_ptrs.last().unwrap_or(&0);
    if nnz != col_indices.len() {
        return Err(SparseError::InconsistentData {
            reason: format!(
                "row_ptrs last entry {} != nnz = {}",
                nnz,
                col_indices.len()
            ),
        });
    }
    for &c in col_indices {
        if c >= ncols {
            return Err(SparseError::IndexOutOfBounds {
                index: (0, c),
                shape: (nrows, ncols),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    // Helper: dense matrix from row-major slice.
    fn mat3x3(v: &[f64]) -> Array2<f64> {
        Array2::from_shape_vec((3, 3), v.to_vec()).expect("shape ok")
    }

    // CSR of the 3×3 matrix
    //   [1 0 2]
    //   [0 3 0]
    //   [4 5 0]
    fn sample_csr() -> (Vec<usize>, Vec<usize>, Vec<f64>, usize, usize) {
        let rp = vec![0, 2, 3, 5];
        let ci = vec![0, 2, 1, 0, 1];
        let vv = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        (rp, ci, vv, 3, 3)
    }

    // -------------------------------------------------------------------
    // COO → CSR round-trip
    // -------------------------------------------------------------------
    #[test]
    fn test_coo_to_csr_round_trip() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 1];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (rp, ci, vv) = coo_to_csr(&rows, &cols, &vals, 3);

        // Convert back to COO.
        let (r2, c2, v2) = csr_to_coo(&rp, &ci, &vv);

        // Sort both by (row, col) before comparing.
        let mut orig: Vec<(usize, usize, f64)> = rows
            .iter()
            .zip(cols.iter())
            .zip(vals.iter())
            .map(|((&r, &c), &v)| (r, c, v))
            .collect();
        let mut result: Vec<(usize, usize, f64)> = r2
            .iter()
            .zip(c2.iter())
            .zip(v2.iter())
            .map(|((&r, &c), &v)| (r, c, v))
            .collect();
        orig.sort_unstable_by_key(|&(r, c, _)| (r, c));
        result.sort_unstable_by_key(|&(r, c, _)| (r, c));
        assert_eq!(orig.len(), result.len());
        for (a, b) in orig.iter().zip(result.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
            assert!((a.2 - b.2).abs() < 1e-12, "value mismatch {a:?} vs {b:?}");
        }
    }

    #[test]
    fn test_coo_to_csr_duplicates_summed() {
        // Two entries at (0,0): should sum to 3.0.
        let rows = vec![0, 0, 1];
        let cols = vec![0, 0, 1];
        let vals = vec![1.0, 2.0, 5.0];
        let (rp, ci, vv) = coo_to_csr(&rows, &cols, &vals, 2);
        let dense = csr_to_dense(&rp, &ci, &vv, 2, 2);
        assert!((dense[[0, 0]] - 3.0).abs() < 1e-12);
        assert!((dense[[1, 1]] - 5.0).abs() < 1e-12);
    }

    // -------------------------------------------------------------------
    // CSR → CSC → CSR identity
    // -------------------------------------------------------------------
    #[test]
    fn test_csr_to_csc_then_csc_to_csr_identity() {
        let (rp, ci, vv, nr, nc) = sample_csr();

        let (cp, ri, cv) = csr_to_csc(&rp, &ci, &vv, nr, nc);
        let (rp2, ci2, vv2) = csc_to_csr(&cp, &ri, &cv, nr, nc);

        // Both CSR representations should give the same dense matrix.
        let d1 = csr_to_dense(&rp, &ci, &vv, nr, nc);
        let d2 = csr_to_dense(&rp2, &ci2, &vv2, nr, nc);
        for i in 0..nr {
            for j in 0..nc {
                assert!((d1[[i, j]] - d2[[i, j]]).abs() < 1e-12,
                    "mismatch at ({i},{j}): {} vs {}", d1[[i,j]], d2[[i,j]]);
            }
        }
    }

    // -------------------------------------------------------------------
    // Dense → CSR → Dense identity
    // -------------------------------------------------------------------
    #[test]
    fn test_dense_to_csr_then_csr_to_dense_identity() {
        let d = mat3x3(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0]);
        let (rp, ci, vv) = dense_to_csr(&d);
        let d2 = csr_to_dense(&rp, &ci, &vv, 3, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!((d[[i, j]] - d2[[i, j]]).abs() < 1e-12,
                    "mismatch at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_dense_to_csr_zero_matrix() {
        let d = Array2::<f64>::zeros((3, 3));
        let (rp, ci, vv) = dense_to_csr(&d);
        assert_eq!(vv.len(), 0);
        assert_eq!(ci.len(), 0);
        assert_eq!(rp, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_dense_to_csr_identity() {
        let d = Array2::<f64>::eye(4);
        let (rp, ci, vv) = dense_to_csr(&d);
        assert_eq!(vv.len(), 4);
        let d2 = csr_to_dense(&rp, &ci, &vv, 4, 4);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((d2[[i, j]] - expected).abs() < 1e-12);
            }
        }
    }

    // -------------------------------------------------------------------
    // CSR matvec matches dense
    // -------------------------------------------------------------------
    #[test]
    fn test_csr_matvec_matches_dense() {
        let d = mat3x3(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0]);
        let (rp, ci, vv) = dense_to_csr(&d);
        let x = vec![1.0, 2.0, 3.0];

        let y_sparse = csr_matvec(&rp, &ci, &vv, &x);

        // Dense multiply.
        let mut y_dense = vec![0.0f64; 3];
        for i in 0..3 {
            for j in 0..3 {
                y_dense[i] += d[[i, j]] * x[j];
            }
        }

        for i in 0..3 {
            assert!((y_sparse[i] - y_dense[i]).abs() < 1e-12,
                "matvec mismatch at {i}: {} vs {}", y_sparse[i], y_dense[i]);
        }
    }

    #[test]
    fn test_csr_matvec_identity() {
        let d = Array2::<f64>::eye(5);
        let (rp, ci, vv) = dense_to_csr(&d);
        let x: Vec<f64> = (1..=5).map(|i| i as f64).collect();
        let y = csr_matvec(&rp, &ci, &vv, &x);
        for i in 0..5 {
            assert!((y[i] - x[i]).abs() < 1e-12);
        }
    }

    // -------------------------------------------------------------------
    // CSR matmat matches dense matmat
    // -------------------------------------------------------------------
    #[test]
    fn test_csr_matmat_matches_dense() {
        let a_d = mat3x3(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 5.0, 0.0]);
        let b_d = mat3x3(&[1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0]);
        let (rpa, cia, va) = dense_to_csr(&a_d);
        let (rpb, cib, vb) = dense_to_csr(&b_d);

        let (rpc, cic, vc) = csr_matmat(&rpa, &cia, &va, &rpb, &cib, &vb, 3, 3, 3);
        let c_sparse = csr_to_dense(&rpc, &cic, &vc, 3, 3);

        // Dense C = A * B.
        let mut c_dense = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for k in 0..3 {
                if a_d[[i, k]] == 0.0 { continue; }
                for j in 0..3 {
                    c_dense[[i, j]] += a_d[[i, k]] * b_d[[k, j]];
                }
            }
        }

        for i in 0..3 {
            for j in 0..3 {
                assert!((c_sparse[[i, j]] - c_dense[[i, j]]).abs() < 1e-12,
                    "matmat mismatch at ({i},{j}): {} vs {}", c_sparse[[i,j]], c_dense[[i,j]]);
            }
        }
    }

    #[test]
    fn test_csr_matmat_identity_property() {
        let d = Array2::<f64>::eye(4);
        let (rp, ci, vv) = dense_to_csr(&d);
        // I * I = I
        let (rpc, cic, vc) = csr_matmat(&rp, &ci, &vv, &rp, &ci, &vv, 4, 4, 4);
        let c = csr_to_dense(&rpc, &cic, &vc, 4, 4);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((c[[i, j]] - expected).abs() < 1e-12);
            }
        }
    }

    // -------------------------------------------------------------------
    // CSR diagonal extraction
    // -------------------------------------------------------------------
    #[test]
    fn test_csr_diagonal() {
        let (rp, ci, vv, nr, nc) = sample_csr();
        let diag = csr_diagonal(&rp, &ci, &vv, nr.min(nc));
        // Matrix:  [1 0 2] → diag[0] = 1
        //          [0 3 0] → diag[1] = 3
        //          [4 5 0] → diag[2] = 0
        assert!((diag[0] - 1.0).abs() < 1e-12);
        assert!((diag[1] - 3.0).abs() < 1e-12);
        assert!((diag[2] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_csr_diagonal_identity() {
        let d = Array2::<f64>::eye(5);
        let (rp, ci, vv) = dense_to_csr(&d);
        let diag = csr_diagonal(&rp, &ci, &vv, 5);
        for i in 0..5 {
            assert!((diag[i] - 1.0).abs() < 1e-12);
        }
    }

    // -------------------------------------------------------------------
    // CSR transpose is its own inverse (for square matrices)
    // -------------------------------------------------------------------
    #[test]
    fn test_csr_transpose_involution() {
        let (rp, ci, vv, nr, nc) = sample_csr();
        let (rpt, cit, vvt) = csr_transpose(&rp, &ci, &vv, nr, nc);
        // Transpose twice → original.
        let (rp2, ci2, vv2) = csr_transpose(&rpt, &cit, &vvt, nc, nr);
        let d1 = csr_to_dense(&rp, &ci, &vv, nr, nc);
        let d2 = csr_to_dense(&rp2, &ci2, &vv2, nr, nc);
        for i in 0..nr {
            for j in 0..nc {
                assert!((d1[[i, j]] - d2[[i, j]]).abs() < 1e-12,
                    "transpose involution failed at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_csr_transpose_correctness() {
        // Non-square matrix 2×3.
        //   [1 2 0]
        //   [0 0 3]
        let rp = vec![0, 2, 3];
        let ci = vec![0, 1, 2];
        let vv = vec![1.0, 2.0, 3.0];
        let (rpt, cit, vvt) = csr_transpose(&rp, &ci, &vv, 2, 3);
        let d_t = csr_to_dense(&rpt, &cit, &vvt, 3, 2);
        // Expected Aᵀ 3×2:  [1 0; 2 0; 0 3]
        assert!((d_t[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((d_t[[1, 0]] - 2.0).abs() < 1e-12);
        assert!((d_t[[2, 1]] - 3.0).abs() < 1e-12);
        assert!((d_t[[0, 1]] - 0.0).abs() < 1e-12);
    }

    // -------------------------------------------------------------------
    // Adjacency → Laplacian: row sums = 0
    // -------------------------------------------------------------------
    #[test]
    fn test_adjacency_to_laplacian_row_sums_zero() {
        // Complete graph K₄.
        let n = 4usize;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    rows.push(i);
                    cols.push(j);
                    vals.push(1.0);
                }
            }
        }
        let (rp, ci, vv) = coo_to_csr(&rows, &cols, &vals, n);
        let (lp, li, lv) = adjacency_to_laplacian(&rp, &ci, &vv, n);

        let ones = vec![1.0f64; n];
        let y = csr_matvec(&lp, &li, &lv, &ones);
        for i in 0..n {
            assert!(y[i].abs() < 1e-10, "L*1 != 0 at row {i}: {}", y[i]);
        }
    }

    #[test]
    fn test_adjacency_to_laplacian_triangle() {
        // 3-cycle: A = [[0,1,1],[1,0,1],[1,1,0]]
        let rp = vec![0, 2, 4, 6];
        let ci = vec![1, 2, 0, 2, 0, 1];
        let vv = vec![1.0; 6];
        let (lp, li, lv) = adjacency_to_laplacian(&rp, &ci, &vv, 3);

        let ones = vec![1.0f64; 3];
        let y = csr_matvec(&lp, &li, &lv, &ones);
        for i in 0..3 {
            assert!(y[i].abs() < 1e-10, "L*1 != 0 at row {i}");
        }

        // Diagonal should be degree = 2.
        let diag = csr_diagonal(&lp, &li, &lv, 3);
        for d in &diag {
            assert!((*d - 2.0).abs() < 1e-12, "degree != 2: {d}");
        }
    }

    // -------------------------------------------------------------------
    // validate_csr
    // -------------------------------------------------------------------
    #[test]
    fn test_validate_csr_ok() {
        let (rp, ci, vv, nr, nc) = sample_csr();
        validate_csr(&rp, &ci, &vv, nr, nc).expect("should be valid");
    }

    #[test]
    fn test_validate_csr_wrong_row_ptrs_length() {
        let rp = vec![0, 2, 3]; // length 3 instead of 4 for nrows=3
        let ci = vec![0, 1, 1];
        let vv = vec![1.0, 2.0, 3.0];
        let result = validate_csr(&rp, &ci, &vv, 3, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_csr_out_of_bounds_col() {
        let rp = vec![0, 1, 2, 3];
        let ci = vec![0, 5, 1]; // col 5 out of bounds for ncols=3
        let vv = vec![1.0, 2.0, 3.0];
        let result = validate_csr(&rp, &ci, &vv, 3, 3);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------
    // CSR → CSC explicit column check
    // -------------------------------------------------------------------
    #[test]
    fn test_csr_to_csc_column_access() {
        let (rp, ci, vv, nr, nc) = sample_csr();
        let (cp, ri, cv) = csr_to_csc(&rp, &ci, &vv, nr, nc);
        // Column 1 of the matrix contains: row1→3, row2→5.
        let col1_start = cp[1];
        let col1_end = cp[2];
        let mut col1: Vec<(usize, f64)> = ri[col1_start..col1_end]
            .iter()
            .zip(cv[col1_start..col1_end].iter())
            .map(|(&r, &v)| (r, v))
            .collect();
        col1.sort_unstable_by_key(|&(r, _)| r);
        assert_eq!(col1, vec![(1, 3.0), (2, 5.0)]);
    }

    // -------------------------------------------------------------------
    // Additional round-trip for rectangular matrices
    // -------------------------------------------------------------------
    #[test]
    fn test_rectangular_csr_csc_round_trip() {
        // 2 × 4 matrix.
        let rp = vec![0, 3, 5];
        let ci = vec![0, 2, 3, 1, 3];
        let vv = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (cp, ri, cv) = csr_to_csc(&rp, &ci, &vv, 2, 4);
        let (rp2, ci2, vv2) = csc_to_csr(&cp, &ri, &cv, 2, 4);
        let d1 = csr_to_dense(&rp, &ci, &vv, 2, 4);
        let d2 = csr_to_dense(&rp2, &ci2, &vv2, 2, 4);
        for i in 0..2 {
            for j in 0..4 {
                assert!((d1[[i, j]] - d2[[i, j]]).abs() < 1e-12);
            }
        }
    }
}
