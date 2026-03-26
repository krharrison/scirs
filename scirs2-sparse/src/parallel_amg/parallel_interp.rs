//! Parallel Interpolation Operator Construction for AMG
//!
//! This module constructs the prolongation operator P (fine → coarse mapping)
//! and restriction operator R = P^T, as well as the Galerkin coarse operator
//! A_c = R A P.
//!
//! # Interpolation Methods
//!
//! 1. **Direct interpolation**: For each F-node, interpolate from C-nodes
//!    in its strong neighborhood. Parallelized row-by-row.
//!
//! 2. **Smoothed aggregation (SA) interpolation**: First builds a tentative
//!    prolongator (nearest-C aggregation), then applies one Jacobi smoothing
//!    step to improve the approximation property.
//!
//! # Galerkin Operator
//!
//! The coarse-grid operator is A_c = R A P = P^T A P. This is computed
//! row-by-row in parallel.
//!
//! # References
//!
//! - De Sterck, H., et al. (2006). "Reducing complexity in parallel algebraic
//!   multigrid preconditioners." SIAM J. Matrix Anal. Appl.
//! - Vaněk, P., et al. (1996). "Algebraic multigrid by smoothed aggregation."
//!   Computing, 56, 179-196.

use crate::csr::CsrMatrix;
use crate::error::SparseResult;
use crate::parallel_amg::strength::StrengthGraph;
use std::sync::Arc;

// ============================================================================
// Utility functions
// ============================================================================

/// Sparse matrix-vector multiply: y = A * x
fn spmv_internal(a: &CsrMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let (n_rows, _) = a.shape();
    let mut y = vec![0.0f64; n_rows];
    for i in 0..n_rows {
        let mut acc = 0.0f64;
        for pos in a.row_range(i) {
            acc += a.data[pos] * x[a.indices[pos]];
        }
        y[i] = acc;
    }
    y
}

/// Build CSR matrix from triplets (row, col, val) for a given shape.
fn build_csr_from_triplets(
    rows_t: Vec<usize>,
    cols_t: Vec<usize>,
    vals_t: Vec<f64>,
    shape: (usize, usize),
) -> SparseResult<CsrMatrix<f64>> {
    CsrMatrix::new(vals_t, rows_t, cols_t, shape)
}

// ============================================================================
// Direct interpolation
// ============================================================================

/// Compute direct interpolation for a range of rows [row_start, row_end).
///
/// Returns triplets (fine_row, coarse_col, value) for rows in that range.
fn direct_interp_row_range(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    splitting: &[u8],
    c_map: &[usize], // c_map[i] = coarse index of C-node i (usize::MAX if F)
    row_start: usize,
    row_end: usize,
) -> Vec<(usize, usize, f64)> {
    let mut triplets = Vec::new();
    let n = splitting.len();

    for i in row_start..row_end {
        if i >= n {
            break;
        }

        if splitting[i] == 1 {
            // C-node: identity row in P (maps to its own coarse index)
            let ci = c_map[i];
            triplets.push((i, ci, 1.0f64));
        } else {
            // F-node: interpolate from C-neighbors in strong set
            // P_{i,j} = -a_{ij} / (a_{ii} * Σ_{k∈C_i} a_{ik}/a_{kk})  for j ∈ C_i
            //
            // Simplified direct interpolation:
            // Find diagonal a_ii
            let mut diag = 0.0f64;
            let row_start_ptr = indptr[i];
            let row_end_ptr = indptr[i + 1];
            for pos in row_start_ptr..row_end_ptr {
                if indices[pos] == i {
                    diag = data[pos];
                    break;
                }
            }

            if diag.abs() < f64::EPSILON {
                continue;
            }

            // Find C-neighbors
            let mut sum_neg_a_ij_over_diag = 0.0f64;
            for pos in row_start_ptr..row_end_ptr {
                let j = indices[pos];
                if j != i && j < splitting.len() && splitting[j] == 1 {
                    sum_neg_a_ij_over_diag += -data[pos];
                }
            }

            if sum_neg_a_ij_over_diag.abs() < f64::EPSILON {
                continue;
            }

            // Assign weights
            for pos in row_start_ptr..row_end_ptr {
                let j = indices[pos];
                if j != i && j < splitting.len() && splitting[j] == 1 {
                    let cj = c_map[j];
                    // w_{ij} = -a_ij / (sum of -a_ik for C-neighbors k)
                    let weight = -data[pos] / sum_neg_a_ij_over_diag;
                    triplets.push((i, cj, weight));
                }
            }
        }
    }

    triplets
}

/// Parallel direct interpolation operator.
///
/// For each C-node: `P[i, c_map[i]] = 1` (identity).
/// For each F-node: `P[i, j] = -a_ij / sum_{k in C_i} (-a_ik)` for each C-neighbor j.
///
/// # Arguments
///
/// * `a` - System matrix
/// * `splitting` - C/F splitting array (1 = C, 0 = F)
/// * `n_threads` - Number of parallel threads
///
/// # Returns
///
/// Prolongation matrix P of shape (n_fine, n_coarse).
pub fn parallel_direct_interpolation(
    a: &CsrMatrix<f64>,
    splitting: &[u8],
    n_threads: usize,
) -> SparseResult<CsrMatrix<f64>> {
    let n = a.shape().0;
    let n_threads = n_threads.max(1);

    // Build coarse index map
    let mut c_map = vec![usize::MAX; n];
    let mut n_coarse = 0usize;
    for i in 0..n {
        if i < splitting.len() && splitting[i] == 1 {
            c_map[i] = n_coarse;
            n_coarse += 1;
        }
    }

    if n_coarse == 0 {
        return CsrMatrix::new(Vec::new(), Vec::new(), Vec::new(), (n, 0));
    }

    let indptr = Arc::new(a.indptr.clone());
    let indices_arc = Arc::new(a.indices.clone());
    let data_arc = Arc::new(a.data.clone());
    let splitting_arc = Arc::new(splitting.to_vec());
    let c_map_arc = Arc::new(c_map);

    let chunk_size = (n + n_threads - 1) / n_threads;
    let mut all_triplets: Vec<Vec<(usize, usize, f64)>> = Vec::new();

    std::thread::scope(|s| {
        let mut handles = Vec::new();

        for t in 0..n_threads {
            let row_start = t * chunk_size;
            let row_end = ((t + 1) * chunk_size).min(n);
            if row_start >= row_end {
                continue;
            }

            let indptr_ref = Arc::clone(&indptr);
            let indices_ref = Arc::clone(&indices_arc);
            let data_ref = Arc::clone(&data_arc);
            let splitting_ref = Arc::clone(&splitting_arc);
            let c_map_ref = Arc::clone(&c_map_arc);

            let handle = s.spawn(move || {
                direct_interp_row_range(
                    &indptr_ref,
                    &indices_ref,
                    &data_ref,
                    &splitting_ref,
                    &c_map_ref,
                    row_start,
                    row_end,
                )
            });
            handles.push(handle);
        }

        for h in handles {
            if let Ok(triplets) = h.join() {
                all_triplets.push(triplets);
            }
        }
    });

    // Merge all triplets
    let mut rows_t = Vec::new();
    let mut cols_t = Vec::new();
    let mut vals_t = Vec::new();
    for chunk in all_triplets {
        for (r, c, v) in chunk {
            rows_t.push(r);
            cols_t.push(c);
            vals_t.push(v);
        }
    }

    build_csr_from_triplets(rows_t, cols_t, vals_t, (n, n_coarse))
}

// ============================================================================
// Smoothed Aggregation interpolation
// ============================================================================

/// Build tentative prolongator P_0 for SA:
/// Each F-node is aggregated to its nearest (first encountered) C-node.
/// C-nodes map to themselves.
fn build_tentative_prolongator(
    strength: &StrengthGraph,
    splitting: &[u8],
    c_map: &[usize],
    n_coarse: usize,
) -> SparseResult<CsrMatrix<f64>> {
    let n = strength.n;
    let mut rows_t = Vec::new();
    let mut cols_t = Vec::new();
    let mut vals_t = Vec::new();

    for i in 0..n {
        if i >= splitting.len() {
            break;
        }
        if splitting[i] == 1 {
            // C-node: maps to its own coarse dof
            rows_t.push(i);
            cols_t.push(c_map[i]);
            vals_t.push(1.0f64);
        } else {
            // F-node: find nearest C-node in strong neighborhood
            let nearest_c = strength.strong_influencers[i]
                .iter()
                .find(|&&j| j < splitting.len() && splitting[j] == 1)
                .or_else(|| {
                    strength.strong_neighbors[i]
                        .iter()
                        .find(|&&j| j < splitting.len() && splitting[j] == 1)
                })
                .copied();

            if let Some(c) = nearest_c {
                rows_t.push(i);
                cols_t.push(c_map[c]);
                vals_t.push(1.0f64);
            }
            // If no C-neighbor found, F-node is not interpolated (rare edge case)
        }
    }

    build_csr_from_triplets(rows_t, cols_t, vals_t, (n, n_coarse))
}

/// Parallel smoothed aggregation interpolation.
///
/// Algorithm:
/// 1. Build tentative prolongator P_0 (nearest-C aggregation).
/// 2. Apply one Jacobi smoothing step: P = (I - ω D^{-1} A) P_0.
///
/// # Arguments
///
/// * `a` - System matrix
/// * `strength` - Strength graph (for aggregation)
/// * `n_threads` - Number of parallel threads
/// * `splitting` - C/F splitting
/// * `omega` - Jacobi smoothing weight (typically 4/3)
///
/// # Returns
///
/// Smoothed prolongation matrix P of shape (n_fine, n_coarse).
pub fn parallel_sa_interpolation(
    a: &CsrMatrix<f64>,
    strength: &StrengthGraph,
    splitting: &[u8],
    n_threads: usize,
    omega: f64,
) -> SparseResult<CsrMatrix<f64>> {
    let n = a.shape().0;
    let n_threads = n_threads.max(1);

    // Build coarse index map
    let mut c_map = vec![usize::MAX; n];
    let mut n_coarse = 0usize;
    for i in 0..n {
        if i < splitting.len() && splitting[i] == 1 {
            c_map[i] = n_coarse;
            n_coarse += 1;
        }
    }

    if n_coarse == 0 {
        return CsrMatrix::new(Vec::new(), Vec::new(), Vec::new(), (n, 0));
    }

    // Build tentative P_0
    let p0 = build_tentative_prolongator(strength, splitting, &c_map, n_coarse)?;

    // Compute diagonal inverse D^{-1}
    let mut diag_inv = vec![0.0f64; n];
    for i in 0..n {
        for pos in a.row_range(i) {
            if a.indices[pos] == i {
                let d = a.data[pos];
                if d.abs() > f64::EPSILON {
                    diag_inv[i] = 1.0 / d;
                }
                break;
            }
        }
    }

    // Jacobi smoothing: P = P_0 - omega * D^{-1} * A * P_0
    // We compute this column-by-column via A*p0_col for each coarse dof
    // Then assemble the smoothed P as triplets.

    let p0_arc = Arc::new(p0);
    let a_arc = Arc::new(a.clone());
    let diag_inv_arc = Arc::new(diag_inv.clone());

    // Process coarse dofs in parallel chunks
    let chunk_size = (n_coarse + n_threads - 1) / n_threads;
    let mut all_cols: Vec<Vec<(usize, usize, f64)>> = Vec::new();

    std::thread::scope(|s| {
        let mut handles = Vec::new();

        for t in 0..n_threads {
            let col_start = t * chunk_size;
            let col_end = ((t + 1) * chunk_size).min(n_coarse);
            if col_start >= col_end {
                continue;
            }

            let p0_ref = Arc::clone(&p0_arc);
            let a_ref = Arc::clone(&a_arc);
            let diag_inv_ref = Arc::clone(&diag_inv_arc);

            let handle = s.spawn(move || {
                let mut col_triplets = Vec::new();
                let n_fine = p0_ref.shape().0;

                for c in col_start..col_end {
                    // Extract column c of P0 as dense vector
                    let mut p0_col = vec![0.0f64; n_fine];
                    for i in 0..n_fine {
                        for pos in p0_ref.row_range(i) {
                            if p0_ref.indices[pos] == c {
                                p0_col[i] = p0_ref.data[pos];
                                break;
                            }
                        }
                    }

                    // Compute A * p0_col
                    let ap0_col = spmv_internal(&a_ref, &p0_col);

                    // Smoothed: p_col = p0_col - omega * D^{-1} * ap0_col
                    for i in 0..n_fine {
                        let val = p0_col[i] - omega * diag_inv_ref[i] * ap0_col[i];
                        if val.abs() > f64::EPSILON * 10.0 {
                            col_triplets.push((i, c, val));
                        }
                    }
                }
                col_triplets
            });
            handles.push(handle);
        }

        for h in handles {
            if let Ok(triplets) = h.join() {
                all_cols.push(triplets);
            }
        }
    });

    // Merge all triplets
    let mut rows_t = Vec::new();
    let mut cols_t = Vec::new();
    let mut vals_t = Vec::new();
    for chunk in all_cols {
        for (r, c, v) in chunk {
            rows_t.push(r);
            cols_t.push(c);
            vals_t.push(v);
        }
    }

    build_csr_from_triplets(rows_t, cols_t, vals_t, (n, n_coarse))
}

// ============================================================================
// Galerkin coarse operator: A_c = R A P = P^T A P
// ============================================================================

/// Compute the Galerkin coarse-grid operator A_c = R * A * P.
///
/// This is computed as: first B = A * P, then A_c = R * B = P^T * B.
/// Parallelized row-by-row: each thread handles a block of coarse rows.
///
/// # Arguments
///
/// * `a` - Fine-grid system matrix
/// * `p` - Prolongation operator (n_fine × n_coarse)
///
/// # Returns
///
/// Coarse-grid matrix A_c of shape (n_coarse × n_coarse).
pub fn galerkin_coarse_operator(
    a: &CsrMatrix<f64>,
    p: &CsrMatrix<f64>,
) -> SparseResult<CsrMatrix<f64>> {
    // Step 1: B = A * P  (n_fine × n_coarse)
    let b = a.matmul(p)?;

    // Step 2: R = P^T  (n_coarse × n_fine)
    let r = p.transpose();

    // Step 3: A_c = R * B = P^T * B  (n_coarse × n_coarse)
    r.matmul(&b)
}

/// Parallel Galerkin operator: splits coarse rows across threads.
///
/// Each thread computes a set of rows of R*A*P using direct triple product
/// row-by-row. This avoids the full intermediate matrix storage when
/// n_coarse is large.
///
/// # Arguments
///
/// * `a` - Fine-grid system matrix (n_fine × n_fine)
/// * `p` - Prolongation operator (n_fine × n_coarse)
/// * `n_threads` - Number of parallel threads
///
/// # Returns
///
/// Coarse-grid matrix A_c of shape (n_coarse × n_coarse).
pub fn parallel_galerkin_coarse_operator(
    a: &CsrMatrix<f64>,
    p: &CsrMatrix<f64>,
    n_threads: usize,
) -> SparseResult<CsrMatrix<f64>> {
    let n_fine = a.shape().0;
    let n_coarse = p.shape().1;
    let n_threads = n_threads.max(1);

    if n_coarse == 0 {
        return CsrMatrix::new(Vec::new(), Vec::new(), Vec::new(), (0, 0));
    }

    // Compute R = P^T
    let r = p.transpose();
    let a_arc = Arc::new(a.clone());
    let p_arc = Arc::new(p.clone());
    let r_arc = Arc::new(r);

    let chunk_size = (n_coarse + n_threads - 1) / n_threads;
    let mut all_triplets: Vec<Vec<(usize, usize, f64)>> = Vec::new();

    std::thread::scope(|s| {
        let mut handles = Vec::new();

        for t in 0..n_threads {
            let row_start = t * chunk_size;
            let row_end = ((t + 1) * chunk_size).min(n_coarse);
            if row_start >= row_end {
                continue;
            }

            let a_ref = Arc::clone(&a_arc);
            let p_ref = Arc::clone(&p_arc);
            let r_ref = Arc::clone(&r_arc);

            let handle = s.spawn(move || {
                // For each coarse row ci in [row_start, row_end):
                // (R A P)[ci, cj] = r_row_ci dot A * p_col_cj
                // = sum_{i} R[ci, i] * sum_{j} A[i, j] * P[j, cj]
                //
                // We compute row ci of R*A first, then multiply by P.
                let mut triplets = Vec::new();

                for ci in row_start..row_end {
                    // Row ci of R = column ci of P (transposed)
                    // r_row: sparse vector of (fine_idx, value) for R[ci, :]
                    let mut r_row: Vec<(usize, f64)> = Vec::new();
                    for pos in r_ref.row_range(ci) {
                        r_row.push((r_ref.indices[pos], r_ref.data[pos]));
                    }

                    // Compute row ci of R*A: (R*A)[ci, :] = sum_i R[ci,i] * A[i,:]
                    let mut ra_row = vec![0.0f64; n_fine];
                    for &(fi, rval) in &r_row {
                        for pos in a_ref.row_range(fi) {
                            ra_row[a_ref.indices[pos]] += rval * a_ref.data[pos];
                        }
                    }

                    // Multiply ra_row by P to get row ci of R*A*P
                    let mut rac_row = vec![0.0f64; n_coarse];
                    for (fi, &raval) in ra_row.iter().enumerate() {
                        if raval.abs() < f64::EPSILON * 1e-6 {
                            continue;
                        }
                        for pos in p_ref.row_range(fi) {
                            rac_row[p_ref.indices[pos]] += raval * p_ref.data[pos];
                        }
                    }

                    for (cj, &val) in rac_row.iter().enumerate() {
                        if val.abs() > f64::EPSILON * 1e-8 {
                            triplets.push((ci, cj, val));
                        }
                    }
                }

                triplets
            });
            handles.push(handle);
        }

        for h in handles {
            if let Ok(triplets) = h.join() {
                all_triplets.push(triplets);
            }
        }
    });

    // Merge all triplets
    let mut rows_t = Vec::new();
    let mut cols_t = Vec::new();
    let mut vals_t = Vec::new();
    for chunk in all_triplets {
        for (r, c, v) in chunk {
            rows_t.push(r);
            cols_t.push(c);
            vals_t.push(v);
        }
    }

    build_csr_from_triplets(rows_t, cols_t, vals_t, (n_coarse, n_coarse))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel_amg::parallel_rs::pmis_coarsening;
    use crate::parallel_amg::strength::serial_strength_of_connection;

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(2.0f64);
        }
        for i in 0..n - 1 {
            rows.push(i);
            cols.push(i + 1);
            vals.push(-1.0f64);
            rows.push(i + 1);
            cols.push(i);
            vals.push(-1.0f64);
        }
        CsrMatrix::new(vals, rows, cols, (n, n)).expect("valid Laplacian")
    }

    #[test]
    fn test_direct_interp_c_node_identity() {
        let n = 8;
        let a = laplacian_1d(n);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);
        let p = parallel_direct_interpolation(&a, &result.cf_splitting, 1)
            .expect("direct interpolation");

        // For each C-node: P has exactly one entry of 1.0 in that row
        let mut c_col = 0usize;
        for i in 0..n {
            if result.cf_splitting[i] == 1 {
                // Find the row i entries in P
                let mut found = false;
                for pos in p.row_range(i) {
                    if p.indices[pos] == c_col {
                        found = true;
                        assert!(
                            (p.data[pos] - 1.0).abs() < 1e-10,
                            "C-node {i} should map to coarse col {c_col} with weight 1.0"
                        );
                    }
                }
                assert!(found, "C-node {i} should have identity entry in P");
                c_col += 1;
            }
        }
    }

    #[test]
    fn test_direct_interp_f_node_has_c_parents() {
        let n = 12;
        let a = laplacian_1d(n);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);
        let p = parallel_direct_interpolation(&a, &result.cf_splitting, 1)
            .expect("direct interpolation");
        let n_coarse = result.c_nodes.len();

        // All column indices of P must be valid coarse indices
        for pos in 0..p.nnz() {
            assert!(
                p.indices[pos] < n_coarse,
                "P column index {} out of range [0, {})",
                p.indices[pos],
                n_coarse
            );
        }
    }

    #[test]
    fn test_direct_interp_parallel() {
        let n = 16;
        let a = laplacian_1d(n);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);

        let p_serial = parallel_direct_interpolation(&a, &result.cf_splitting, 1)
            .expect("serial direct interpolation");
        let p_parallel = parallel_direct_interpolation(&a, &result.cf_splitting, 4)
            .expect("parallel direct interpolation");

        assert_eq!(p_serial.shape(), p_parallel.shape());
        assert_eq!(p_serial.nnz(), p_parallel.nnz(), "NNZ should match");
    }

    #[test]
    fn test_sa_interp_shape() {
        let n = 16;
        let a = laplacian_1d(n);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);
        let n_coarse = result.c_nodes.len();

        let p = parallel_sa_interpolation(&a, &g, &result.cf_splitting, 2, 4.0 / 3.0)
            .expect("SA interpolation");

        let (rows, cols) = p.shape();
        assert_eq!(rows, n, "P should have n_fine rows");
        assert_eq!(cols, n_coarse, "P should have n_coarse columns");
    }

    #[test]
    fn test_galerkin_operator_size() {
        let n = 12;
        let a = laplacian_1d(n);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);
        let n_coarse = result.c_nodes.len();

        let p = parallel_direct_interpolation(&a, &result.cf_splitting, 1)
            .expect("direct interpolation");
        let ac = galerkin_coarse_operator(&a, &p).expect("galerkin operator");

        let (rows_c, cols_c) = ac.shape();
        assert_eq!(rows_c, n_coarse, "A_c should have n_coarse rows");
        assert_eq!(cols_c, n_coarse, "A_c should have n_coarse columns");
    }

    #[test]
    fn test_galerkin_spd_preserved() {
        // The 1D Laplacian is SPD (positive semi-definite with Dirichlet boundary)
        // The Galerkin operator should also be SPD
        let n = 10;
        let a = laplacian_1d(n);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);

        let p = parallel_direct_interpolation(&a, &result.cf_splitting, 1)
            .expect("direct interpolation");
        let ac = galerkin_coarse_operator(&a, &p).expect("galerkin operator");

        // Check diagonal dominance (sufficient for SPD in this context)
        let (nc, _) = ac.shape();
        for i in 0..nc {
            let mut diag = 0.0f64;
            let mut off_diag_sum = 0.0f64;
            for pos in ac.row_range(i) {
                if ac.indices[pos] == i {
                    diag = ac.data[pos];
                } else {
                    off_diag_sum += ac.data[pos].abs();
                }
            }
            assert!(
                diag > 0.0,
                "Coarse diagonal should be positive (got {diag} at row {i})"
            );
            // Diagonal dominance: diag >= off_diag_sum (or close)
            assert!(
                diag >= off_diag_sum - 1e-10,
                "Coarse matrix not diagonally dominant at row {i}: diag={diag}, off={off_diag_sum}"
            );
        }
    }
}
