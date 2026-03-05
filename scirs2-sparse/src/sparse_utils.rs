//! Sparse matrix utility operations
//!
//! This module provides fundamental sparse matrix operations including:
//!
//! - **Norms**: 1-norm, infinity-norm, Frobenius norm
//! - **SpGEMM**: Sparse matrix-matrix multiplication
//! - **Arithmetic**: Sparse addition, subtraction, scaling
//! - **Kronecker product**: Sparse Kronecker (tensor) product
//! - **Reordering**: Reverse Cuthill-McKee bandwidth reduction
//! - **Condition number estimate**: Cheap 1-norm-based condition estimate

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::collections::VecDeque;
use std::fmt::Debug;
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Sparse matrix norms
// ---------------------------------------------------------------------------

/// Type of matrix norm to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseNorm {
    /// 1-norm: maximum absolute column sum.
    One,
    /// Infinity norm: maximum absolute row sum.
    Inf,
    /// Frobenius norm: sqrt of sum of squared elements.
    Frobenius,
}

/// Compute a matrix norm of a sparse CSR matrix.
///
/// - `One`: max over columns of sum of absolute values (||A||_1)
/// - `Inf`: max over rows of sum of absolute values (||A||_inf)
/// - `Frobenius`: sqrt(sum(|a_ij|^2))
pub fn sparse_matrix_norm<F>(a: &CsrMatrix<F>, norm_type: SparseNorm) -> SparseResult<F>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let (m, n_cols) = a.shape();
    match norm_type {
        SparseNorm::Inf => {
            let mut max_row_sum = F::sparse_zero();
            for i in 0..m {
                let range = a.row_range(i);
                let vals = &a.data[range];
                let row_sum: F = vals.iter().map(|v| v.abs()).sum();
                if row_sum > max_row_sum {
                    max_row_sum = row_sum;
                }
            }
            Ok(max_row_sum)
        }
        SparseNorm::One => {
            let mut col_sums = vec![F::sparse_zero(); n_cols];
            for i in 0..m {
                let range = a.row_range(i);
                let indices = &a.indices[range.clone()];
                let vals = &a.data[range];
                for (idx, &col) in indices.iter().enumerate() {
                    col_sums[col] += vals[idx].abs();
                }
            }
            let max_col =
                col_sums
                    .iter()
                    .copied()
                    .fold(F::sparse_zero(), |acc, x| if x > acc { x } else { acc });
            Ok(max_col)
        }
        SparseNorm::Frobenius => {
            let mut sum_sq = F::sparse_zero();
            for val in &a.data {
                sum_sq += *val * *val;
            }
            Ok(sum_sq.sqrt())
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse matrix-matrix multiplication (SpGEMM)
// ---------------------------------------------------------------------------

/// Sparse matrix-matrix multiplication: C = A * B.
///
/// Both A and B are in CSR format. The result is also in CSR format.
/// Uses a symbolic + numeric two-phase approach for efficiency.
pub fn spgemm<F>(a: &CsrMatrix<F>, b: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let (m, ka) = a.shape();
    let (kb, n) = b.shape();
    if ka != kb {
        return Err(SparseError::ShapeMismatch {
            expected: (m, ka),
            found: (kb, n),
        });
    }

    // Dense accumulator approach (scatter-gather): for each row of C,
    // scatter contributions into a dense workspace, then gather non-zeros.
    let mut values = vec![F::sparse_zero(); n];
    let mut active = vec![false; n];
    let mut rows_out = Vec::new();
    let mut cols_out = Vec::new();
    let mut data_out = Vec::new();

    for i in 0..m {
        let a_range = a.row_range(i);
        let a_cols = &a.indices[a_range.clone()];
        let a_vals = &a.data[a_range];

        // Scatter: accumulate row i of C = sum_k a_ik * b_row_k
        let mut col_list: Vec<usize> = Vec::new();
        for (a_idx, &k_col) in a_cols.iter().enumerate() {
            let a_ik = a_vals[a_idx];
            let b_range = b.row_range(k_col);
            let b_cols = &b.indices[b_range.clone()];
            let b_vals = &b.data[b_range];

            for (b_idx, &j) in b_cols.iter().enumerate() {
                values[j] += a_ik * b_vals[b_idx];
                if !active[j] {
                    active[j] = true;
                    col_list.push(j);
                }
            }
        }

        // Gather: collect non-zero entries in sorted column order
        col_list.sort_unstable();
        for &j in &col_list {
            let val = values[j];
            if val.abs() > F::epsilon() * F::from(0.01).unwrap_or(F::sparse_zero()) {
                rows_out.push(i);
                cols_out.push(j);
                data_out.push(val);
            }
            // Reset workspace
            values[j] = F::sparse_zero();
            active[j] = false;
        }
    }

    CsrMatrix::new(data_out, rows_out, cols_out, (m, n))
}

// ---------------------------------------------------------------------------
// Sparse matrix addition / subtraction
// ---------------------------------------------------------------------------

/// Sparse matrix addition: C = alpha * A + beta * B.
///
/// Both A and B must have the same shape. The result is in CSR format.
pub fn sparse_add<F>(
    a: &CsrMatrix<F>,
    b: &CsrMatrix<F>,
    alpha: F,
    beta: F,
) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let (ma, na) = a.shape();
    let (mb, nb) = b.shape();
    if ma != mb || na != nb {
        return Err(SparseError::ShapeMismatch {
            expected: (ma, na),
            found: (mb, nb),
        });
    }

    let mut rows_out = Vec::new();
    let mut cols_out = Vec::new();
    let mut data_out = Vec::new();

    let mut b_vals = vec![F::sparse_zero(); na]; // workspace for one row of B
    let mut b_flags = vec![false; na];

    for i in 0..ma {
        // Load row i of B into workspace
        let b_range = b.row_range(i);
        let b_cols = &b.indices[b_range.clone()];
        let b_data = &b.data[b_range];
        for (idx, &col) in b_cols.iter().enumerate() {
            b_vals[col] = b_data[idx];
            b_flags[col] = true;
        }

        // Process row i of A
        let a_range = a.row_range(i);
        let a_cols = &a.indices[a_range.clone()];
        let a_data = &a.data[a_range];
        let mut used_cols: Vec<usize> = Vec::new();

        for (idx, &col) in a_cols.iter().enumerate() {
            let val = alpha * a_data[idx]
                + if b_flags[col] {
                    beta * b_vals[col]
                } else {
                    F::sparse_zero()
                };
            if val.abs() > F::epsilon() {
                rows_out.push(i);
                cols_out.push(col);
                data_out.push(val);
            }
            if b_flags[col] {
                b_flags[col] = false;
                b_vals[col] = F::sparse_zero();
            }
            used_cols.push(col);
        }

        // Remaining entries from B not in A
        for (idx, &col) in b_cols.iter().enumerate() {
            if b_flags[col] {
                let val = beta * b_data[idx];
                if val.abs() > F::epsilon() {
                    rows_out.push(i);
                    cols_out.push(col);
                    data_out.push(val);
                }
                b_flags[col] = false;
                b_vals[col] = F::sparse_zero();
            }
        }
    }

    CsrMatrix::new(data_out, rows_out, cols_out, (ma, na))
}

/// Sparse matrix subtraction: C = A - B.
pub fn sparse_sub<F>(a: &CsrMatrix<F>, b: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    sparse_add(a, b, F::sparse_one(), -F::sparse_one())
}

/// Scale a sparse matrix: C = alpha * A.
pub fn sparse_scale<F>(a: &CsrMatrix<F>, alpha: F) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let (m, n) = a.shape();
    let (rows_in, cols_in, data_in) = a.get_triplets();
    let data_out: Vec<F> = data_in.iter().map(|&v| alpha * v).collect();
    CsrMatrix::new(data_out, rows_in, cols_in, (m, n))
}

// ---------------------------------------------------------------------------
// Sparse Kronecker product
// ---------------------------------------------------------------------------

/// Compute the Kronecker product C = A kron B.
///
/// If A is (m x n) and B is (p x q), then C is (mp x nq).
pub fn sparse_kronecker<F>(a: &CsrMatrix<F>, b: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let (ma, na) = a.shape();
    let (mb, nb) = b.shape();
    let out_rows = ma * mb;
    let out_cols = na * nb;

    let mut rows_out = Vec::new();
    let mut cols_out = Vec::new();
    let mut data_out = Vec::new();

    for ia in 0..ma {
        let a_range = a.row_range(ia);
        let a_cols = &a.indices[a_range.clone()];
        let a_vals = &a.data[a_range];

        for ib in 0..mb {
            let b_range = b.row_range(ib);
            let b_cols = &b.indices[b_range.clone()];
            let b_vals = &b.data[b_range];

            let out_row = ia * mb + ib;

            for (a_idx, &ja) in a_cols.iter().enumerate() {
                for (b_idx, &jb) in b_cols.iter().enumerate() {
                    let out_col = ja * nb + jb;
                    let val = a_vals[a_idx] * b_vals[b_idx];
                    if val.abs() > F::epsilon() {
                        rows_out.push(out_row);
                        cols_out.push(out_col);
                        data_out.push(val);
                    }
                }
            }
        }
    }

    CsrMatrix::new(data_out, rows_out, cols_out, (out_rows, out_cols))
}

// ---------------------------------------------------------------------------
// Reverse Cuthill-McKee (RCM) reordering
// ---------------------------------------------------------------------------

/// Result of the Reverse Cuthill-McKee algorithm.
#[derive(Debug, Clone)]
pub struct RcmResult {
    /// The permutation vector: `new_index[i]` = old_index.
    pub permutation: Vec<usize>,
    /// The inverse permutation: `old_index[i]` = new_index.
    pub inverse_permutation: Vec<usize>,
    /// Original bandwidth of the matrix.
    pub original_bandwidth: usize,
    /// Bandwidth after reordering.
    pub new_bandwidth: usize,
}

/// Compute the bandwidth of a sparse matrix (max |i - j| for non-zero a_ij).
fn compute_bandwidth<F>(a: &CsrMatrix<F>) -> usize
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let m = a.rows();
    let mut bw = 0usize;
    for i in 0..m {
        let range = a.row_range(i);
        for &col in &a.indices[range] {
            let diff = i.abs_diff(col);
            if diff > bw {
                bw = diff;
            }
        }
    }
    bw
}

/// Compute the degree of node `i` in the adjacency graph (number of off-diagonal entries in row i).
fn node_degree<F>(a: &CsrMatrix<F>, i: usize) -> usize
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let range = a.row_range(i);
    a.indices[range].iter().filter(|&&col| col != i).count()
}

/// Find a pseudo-peripheral node (good starting node for RCM).
fn find_pseudo_peripheral<F>(a: &CsrMatrix<F>) -> usize
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let n = a.rows();
    if n == 0 {
        return 0;
    }

    // Start from the node with minimum degree
    let mut start = 0;
    let mut min_deg = usize::MAX;
    for i in 0..n {
        let deg = node_degree(a, i);
        if deg < min_deg {
            min_deg = deg;
            start = i;
        }
    }

    // BFS to find a peripheral node
    for _ in 0..5 {
        let levels = bfs_levels(a, start);
        let max_level = levels.iter().copied().max().unwrap_or(0);
        if max_level == 0 {
            break;
        }
        // Among the nodes at the last level, pick the one with minimum degree
        let mut best = start;
        let mut best_deg = usize::MAX;
        for i in 0..n {
            if levels[i] == max_level {
                let deg = node_degree(a, i);
                if deg < best_deg {
                    best_deg = deg;
                    best = i;
                }
            }
        }
        if best == start {
            break;
        }
        start = best;
    }

    start
}

/// BFS from `start`, returning level numbers for each node.
fn bfs_levels<F>(a: &CsrMatrix<F>, start: usize) -> Vec<usize>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let n = a.rows();
    let mut levels = vec![usize::MAX; n];
    let mut queue = VecDeque::new();
    levels[start] = 0;
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        let range = a.row_range(node);
        for &neighbor in &a.indices[range] {
            if levels[neighbor] == usize::MAX {
                levels[neighbor] = levels[node] + 1;
                queue.push_back(neighbor);
            }
        }
    }

    levels
}

/// Compute the Reverse Cuthill-McKee permutation of a sparse matrix.
///
/// The RCM algorithm reduces the bandwidth of a sparse matrix by
/// reordering its rows and columns. This can significantly improve
/// the performance of direct solvers and incomplete factorizations.
///
/// # Arguments
///
/// * `a` - Square sparse matrix in CSR format
///
/// # Returns
///
/// An `RcmResult` containing the permutation and bandwidth information.
pub fn reverse_cuthill_mckee<F>(a: &CsrMatrix<F>) -> SparseResult<RcmResult>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let (m, n) = a.shape();
    if m != n {
        return Err(SparseError::ValueError(
            "RCM requires a square matrix".to_string(),
        ));
    }

    let original_bandwidth = compute_bandwidth(a);

    if m == 0 {
        return Ok(RcmResult {
            permutation: Vec::new(),
            inverse_permutation: Vec::new(),
            original_bandwidth: 0,
            new_bandwidth: 0,
        });
    }

    // Cuthill-McKee ordering
    let mut visited = vec![false; m];
    let mut cm_order = Vec::with_capacity(m);

    // Handle potentially disconnected graphs
    while cm_order.len() < m {
        // Find starting node for next component
        let start = if cm_order.is_empty() {
            find_pseudo_peripheral(a)
        } else {
            // Find first unvisited node
            let mut s = 0;
            for i in 0..m {
                if !visited[i] {
                    s = i;
                    break;
                }
            }
            s
        };

        if visited[start] {
            break;
        }

        visited[start] = true;
        cm_order.push(start);
        let mut queue_start = cm_order.len() - 1;

        while queue_start < cm_order.len() {
            let node = cm_order[queue_start];
            queue_start += 1;

            // Get neighbors sorted by degree (ascending)
            let range = a.row_range(node);
            let mut neighbors: Vec<usize> = a.indices[range]
                .iter()
                .copied()
                .filter(|&nb| !visited[nb])
                .collect();
            neighbors.sort_by_key(|&nb| node_degree(a, nb));

            for nb in neighbors {
                if !visited[nb] {
                    visited[nb] = true;
                    cm_order.push(nb);
                }
            }
        }
    }

    // Reverse the ordering for RCM
    cm_order.reverse();

    // Compute inverse permutation
    let mut inv_perm = vec![0usize; m];
    for (new_idx, &old_idx) in cm_order.iter().enumerate() {
        inv_perm[old_idx] = new_idx;
    }

    // Compute new bandwidth
    let mut new_bw = 0usize;
    for i in 0..m {
        let range = a.row_range(i);
        let new_i = inv_perm[i];
        for &col in &a.indices[range] {
            let new_j = inv_perm[col];
            let diff = new_i.abs_diff(new_j);
            if diff > new_bw {
                new_bw = diff;
            }
        }
    }

    Ok(RcmResult {
        permutation: cm_order,
        inverse_permutation: inv_perm,
        original_bandwidth,
        new_bandwidth: new_bw,
    })
}

/// Apply a permutation to a sparse matrix: P * A * P^T.
///
/// This reorders both rows and columns using the given permutation vector
/// where `perm[new_i] = old_i`.
pub fn permute_matrix<F>(
    a: &CsrMatrix<F>,
    perm: &[usize],
    inv_perm: &[usize],
) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let (m, n) = a.shape();
    if perm.len() != m || inv_perm.len() != n {
        return Err(SparseError::ValueError(
            "Permutation size mismatch".to_string(),
        ));
    }

    let mut rows_out = Vec::new();
    let mut cols_out = Vec::new();
    let mut data_out = Vec::new();

    for new_i in 0..m {
        let old_i = perm[new_i];
        let range = a.row_range(old_i);
        let old_cols = &a.indices[range.clone()];
        let vals = &a.data[range];

        for (idx, &old_j) in old_cols.iter().enumerate() {
            let new_j = inv_perm[old_j];
            rows_out.push(new_i);
            cols_out.push(new_j);
            data_out.push(vals[idx]);
        }
    }

    CsrMatrix::new(data_out, rows_out, cols_out, (m, n))
}

// ---------------------------------------------------------------------------
// Condition number estimate
// ---------------------------------------------------------------------------

/// Estimate the 1-norm condition number of a sparse matrix.
///
/// Uses Hager's algorithm (1-norm estimation) combined with a simple
/// triangular solve estimate. This is much cheaper than computing the
/// actual condition number via SVD.
///
/// Returns `None` if the matrix appears singular.
pub fn condest_1norm<F>(a: &CsrMatrix<F>) -> SparseResult<Option<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let (m, n) = a.shape();
    if m != n || m == 0 {
        return Err(SparseError::ValueError(
            "condest requires a non-empty square matrix".to_string(),
        ));
    }

    let a_norm = sparse_matrix_norm(a, SparseNorm::One)?;
    if a_norm < F::epsilon() {
        return Ok(None); // Essentially zero matrix
    }

    // Estimate ||A^{-1}||_1 using Hager's algorithm
    // Start with x = (1/n, 1/n, ..., 1/n)
    let inv_n = F::sparse_one()
        / F::from(n as f64)
            .ok_or_else(|| SparseError::ValueError("Failed to convert n".to_string()))?;

    let mut x = vec![inv_n; n];
    let max_iter = 5;
    let mut gamma = F::sparse_zero();

    for _ in 0..max_iter {
        // y = A^{-1} x  (approximate via iterative refinement)
        let y = approximate_solve(a, &x)?;

        // gamma = ||y||_1
        let new_gamma: F = y.iter().map(|v| v.abs()).sum();
        if new_gamma <= gamma {
            break;
        }
        gamma = new_gamma;

        // z = A^{-T} sign(y)
        let sign_y: Vec<F> = y
            .iter()
            .map(|&v| {
                if v >= F::sparse_zero() {
                    F::sparse_one()
                } else {
                    -F::sparse_one()
                }
            })
            .collect();

        let at = a.transpose();
        let z = approximate_solve(&at, &sign_y)?;

        // Find the index of maximum |z_j|
        let mut max_abs = F::sparse_zero();
        let mut max_idx = 0;
        for (j, &zj) in z.iter().enumerate() {
            if zj.abs() > max_abs {
                max_abs = zj.abs();
                max_idx = j;
            }
        }

        // Check if we can improve: ||z||_inf <= z^T x
        let ztx: F = z.iter().zip(x.iter()).map(|(&zi, &xi)| zi * xi).sum();
        if max_abs <= ztx {
            break;
        }

        // x = e_{max_idx}
        for xi in x.iter_mut() {
            *xi = F::sparse_zero();
        }
        x[max_idx] = F::sparse_one();
    }

    if gamma < F::epsilon() {
        return Ok(None);
    }

    Ok(Some(a_norm * gamma))
}

/// Approximate solve of A * x = b using a few Jacobi iterations.
/// This is used internally for condition number estimation and does NOT
/// need to be highly accurate.
fn approximate_solve<F>(a: &CsrMatrix<F>, b: &[F]) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let n = b.len();
    let (m, _) = a.shape();
    if m != n {
        return Err(SparseError::DimensionMismatch {
            expected: m,
            found: n,
        });
    }

    // Extract diagonal
    let mut diag = vec![F::sparse_one(); n];
    for i in 0..n {
        let d = a.get(i, i);
        if d.abs() > F::epsilon() {
            diag[i] = d;
        }
    }

    // A few Jacobi iterations: x_{k+1} = D^{-1} (b - (A - D) x_k)
    let mut x = vec![F::sparse_zero(); n];
    for _ in 0..10 {
        let mut x_new = vec![F::sparse_zero(); n];
        for i in 0..n {
            let range = a.row_range(i);
            let cols = &a.indices[range.clone()];
            let vals = &a.data[range];
            let mut sum = b[i];
            for (idx, &col) in cols.iter().enumerate() {
                if col != i {
                    sum -= vals[idx] * x[col];
                }
            }
            x_new[i] = sum / diag[i];
        }
        x = x_new;
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Sparse matrix transpose
// ---------------------------------------------------------------------------

/// Compute the transpose of a sparse CSR matrix (returns a new CSR matrix).
pub fn sparse_transpose<F>(a: &CsrMatrix<F>) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    Ok(a.transpose())
}

// ---------------------------------------------------------------------------
// Sparse diagonal extraction
// ---------------------------------------------------------------------------

/// Extract the diagonal of a sparse CSR matrix as a dense vector.
pub fn sparse_extract_diagonal<F>(a: &CsrMatrix<F>) -> Vec<F>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let n = a.rows().min(a.cols());
    let mut diag = vec![F::sparse_zero(); n];
    for i in 0..n {
        diag[i] = a.get(i, i);
    }
    diag
}

/// Compute the trace of a sparse matrix (sum of diagonal elements).
pub fn sparse_matrix_trace<F>(a: &CsrMatrix<F>) -> F
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let diag = sparse_extract_diagonal(a);
    diag.iter().copied().sum()
}

// ---------------------------------------------------------------------------
// Sparse identity matrix construction
// ---------------------------------------------------------------------------

/// Create an n x n sparse identity matrix in CSR format.
pub fn sparse_identity<F>(n: usize) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let rows: Vec<usize> = (0..n).collect();
    let cols: Vec<usize> = (0..n).collect();
    let data: Vec<F> = vec![F::sparse_one(); n];
    CsrMatrix::new(data, rows, cols, (n, n))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_matrix() -> CsrMatrix<f64> {
        // 3x3 matrix:
        // [1  2  0]
        // [3  4  5]
        // [0  6  7]
        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 1, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        CsrMatrix::new(data, rows, cols, (3, 3)).expect("valid matrix")
    }

    fn build_identity(n: usize) -> CsrMatrix<f64> {
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        let data = vec![1.0; n];
        CsrMatrix::new(data, rows, cols, (n, n)).expect("valid identity")
    }

    fn build_tridiag(n: usize) -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                data.push(-1.0);
            }
            rows.push(i);
            cols.push(i);
            data.push(2.0);
            if i + 1 < n {
                rows.push(i);
                cols.push(i + 1);
                data.push(-1.0);
            }
        }
        CsrMatrix::new(data, rows, cols, (n, n)).expect("valid matrix")
    }

    #[test]
    fn test_frobenius_norm() {
        let a = build_test_matrix();
        let nrm = sparse_matrix_norm(&a, SparseNorm::Frobenius).expect("frobenius norm");
        // sqrt(1 + 4 + 9 + 16 + 25 + 36 + 49) = sqrt(140)
        let expected = (140.0_f64).sqrt();
        assert!(
            (nrm - expected).abs() < 1e-10,
            "Expected {expected}, got {nrm}"
        );
    }

    #[test]
    fn test_one_norm() {
        let a = build_test_matrix();
        let nrm = sparse_matrix_norm(&a, SparseNorm::One).expect("1-norm");
        // Column sums: col0=|1|+|3|=4, col1=|2|+|4|+|6|=12, col2=|5|+|7|=12
        assert!((nrm - 12.0).abs() < 1e-10, "Expected 12.0, got {nrm}");
    }

    #[test]
    fn test_inf_norm() {
        let a = build_test_matrix();
        let nrm = sparse_matrix_norm(&a, SparseNorm::Inf).expect("inf-norm");
        // Row sums: row0=1+2=3, row1=3+4+5=12, row2=6+7=13
        assert!((nrm - 13.0).abs() < 1e-10, "Expected 13.0, got {nrm}");
    }

    #[test]
    fn test_spgemm_identity() {
        let a = build_test_matrix();
        let eye = build_identity(3);
        let c = spgemm(&a, &eye).expect("spgemm A*I");
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (c.get(i, j) - a.get(i, j)).abs() < 1e-10,
                    "Mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_spgemm_square() {
        // A = [[1,2],[3,4]]
        // A^2 = [[7,10],[15,22]]
        let rows = vec![0, 0, 1, 1];
        let cols = vec![0, 1, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let a = CsrMatrix::new(data, rows, cols, (2, 2)).expect("valid matrix");
        let c = spgemm(&a, &a).expect("spgemm A*A");
        assert!((c.get(0, 0) - 7.0).abs() < 1e-10);
        assert!((c.get(0, 1) - 10.0).abs() < 1e-10);
        assert!((c.get(1, 0) - 15.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 22.0).abs() < 1e-10);
    }

    #[test]
    fn test_spgemm_dimension_mismatch() {
        let a = CsrMatrix::new(vec![1.0], vec![0], vec![0], (1, 2)).expect("valid");
        let b = CsrMatrix::new(vec![1.0], vec![0], vec![0], (3, 1)).expect("valid");
        assert!(spgemm(&a, &b).is_err());
    }

    #[test]
    fn test_sparse_add() {
        let a = build_identity(3);
        let b = build_identity(3);
        let c = sparse_add(&a, &b, 2.0, 3.0).expect("sparse add");
        // C = 2I + 3I = 5I
        for i in 0..3 {
            assert!((c.get(i, i) - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sparse_sub() {
        let a = build_test_matrix();
        let c = sparse_sub(&a, &a).expect("sparse sub");
        // A - A should be zero
        for i in 0..3 {
            for j in 0..3 {
                assert!(c.get(i, j).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_sparse_scale() {
        let a = build_identity(3);
        let c = sparse_scale(&a, 5.0).expect("sparse scale");
        for i in 0..3 {
            assert!((c.get(i, i) - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sparse_kronecker_identity() {
        let i2 = build_identity(2);
        let i3 = build_identity(3);
        let c = sparse_kronecker(&i2, &i3).expect("kronecker I2 x I3");
        // Result should be I6
        let (m, n) = c.shape();
        assert_eq!(m, 6);
        assert_eq!(n, 6);
        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (c.get(i, j) - expected).abs() < 1e-10,
                    "Kronecker I2xI3 mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_sparse_kronecker_small() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // A kron B = [[5,6,10,12],[7,8,14,16],[15,18,20,24],[21,24,28,32]]
        let a = CsrMatrix::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            (2, 2),
        )
        .expect("valid");
        let b = CsrMatrix::new(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            (2, 2),
        )
        .expect("valid");
        let c = sparse_kronecker(&a, &b).expect("kronecker");
        assert!((c.get(0, 0) - 5.0).abs() < 1e-10);
        assert!((c.get(0, 2) - 10.0).abs() < 1e-10);
        assert!((c.get(3, 3) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_rcm_tridiagonal() {
        let n = 10;
        let a = build_tridiag(n);
        let result = reverse_cuthill_mckee(&a).expect("rcm");
        assert_eq!(result.permutation.len(), n);
        assert_eq!(result.inverse_permutation.len(), n);
        // Tridiagonal already has bandwidth 1, RCM shouldn't make it worse
        assert!(result.new_bandwidth <= result.original_bandwidth + 1);
    }

    #[test]
    fn test_rcm_sparse_matrix() {
        // A banded matrix with wider bandwidth
        let n = 8;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            data.push(4.0);
            if i + 1 < n {
                rows.push(i);
                cols.push(i + 1);
                data.push(-1.0);
                rows.push(i + 1);
                cols.push(i);
                data.push(-1.0);
            }
            if i + 3 < n {
                rows.push(i);
                cols.push(i + 3);
                data.push(-0.5);
                rows.push(i + 3);
                cols.push(i);
                data.push(-0.5);
            }
        }
        let a = CsrMatrix::new(data, rows, cols, (n, n)).expect("valid matrix");
        let result = reverse_cuthill_mckee(&a).expect("rcm");
        // Just verify it produces a valid permutation
        let mut sorted_perm = result.permutation.clone();
        sorted_perm.sort();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(sorted_perm, expected);
    }

    #[test]
    fn test_rcm_identity() {
        let eye = build_identity(5);
        let result = reverse_cuthill_mckee(&eye).expect("rcm identity");
        assert_eq!(result.original_bandwidth, 0);
        assert_eq!(result.new_bandwidth, 0);
    }

    #[test]
    fn test_rcm_error_non_square() {
        let a = CsrMatrix::new(vec![1.0, 2.0], vec![0, 1], vec![0, 1], (2, 3)).expect("valid");
        assert!(reverse_cuthill_mckee(&a).is_err());
    }

    #[test]
    fn test_permute_matrix() {
        // A = [[1,2],[3,4]]
        // Permutation [1,0] => P*A*P^T = [[4,3],[2,1]]
        let a = CsrMatrix::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            (2, 2),
        )
        .expect("valid");
        let perm = vec![1, 0];
        let inv_perm = vec![1, 0];
        let b = permute_matrix(&a, &perm, &inv_perm).expect("permute");
        assert!((b.get(0, 0) - 4.0).abs() < 1e-10);
        assert!((b.get(0, 1) - 3.0).abs() < 1e-10);
        assert!((b.get(1, 0) - 2.0).abs() < 1e-10);
        assert!((b.get(1, 1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_condest_identity() {
        let eye = build_identity(5);
        let cond = condest_1norm(&eye).expect("condest");
        // Condition number of identity is 1
        if let Some(c) = cond {
            assert!((c - 1.0).abs() < 1.0, "Expected cond(I) ~ 1, got {c}");
        }
    }

    #[test]
    fn test_condest_diagonal() {
        // diag(1, 100) => cond_1 = 100
        let a = CsrMatrix::new(vec![1.0, 100.0], vec![0, 1], vec![0, 1], (2, 2)).expect("valid");
        let cond = condest_1norm(&a).expect("condest");
        if let Some(c) = cond {
            // Should be around 100
            assert!(c > 10.0 && c < 1000.0, "Expected cond ~ 100, got {c}");
        }
    }

    #[test]
    fn test_condest_error_non_square() {
        let a = CsrMatrix::new(vec![1.0], vec![0], vec![0], (1, 2)).expect("valid");
        assert!(condest_1norm(&a).is_err());
    }

    #[test]
    fn test_sparse_extract_diagonal() {
        let a = build_test_matrix();
        let diag = sparse_extract_diagonal(&a);
        assert_eq!(diag.len(), 3);
        assert!((diag[0] - 1.0).abs() < 1e-10);
        assert!((diag[1] - 4.0).abs() < 1e-10);
        assert!((diag[2] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_matrix_trace() {
        let a = build_test_matrix();
        let tr = sparse_matrix_trace(&a);
        assert!((tr - 12.0).abs() < 1e-10); // 1 + 4 + 7 = 12
    }

    #[test]
    fn test_sparse_identity() {
        let eye: CsrMatrix<f64> = sparse_identity(4).expect("sparse identity");
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (eye.get(i, j) - expected).abs() < 1e-10,
                    "Identity mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_sparse_transpose() {
        let a = build_test_matrix();
        let at = sparse_transpose(&a).expect("transpose");
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (at.get(i, j) - a.get(j, i)).abs() < 1e-10,
                    "Transpose mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_compute_bandwidth() {
        let a = build_tridiag(5);
        assert_eq!(compute_bandwidth(&a), 1);
    }
}
