//! Smoothed Aggregation Algebraic Multigrid (SA-AMG)
//!
//! This module implements the Smoothed Aggregation (SA) variant of Algebraic
//! Multigrid (AMG), suitable for symmetric positive definite (SPD) systems.
//!
//! # Algorithm Overview
//!
//! SA-AMG proceeds in two phases:
//!
//! **Setup phase** (`sa_setup`):
//! 1. Compute strength-of-connection graph from A
//! 2. Aggregate nodes using a greedy algorithm
//! 3. Construct tentative prolongation P₀ from aggregates + near-nullspace
//! 4. Smooth P₀ with a Jacobi step to get final prolongation P
//! 5. Compute coarse-grid operator A_c = P^T A P
//! 6. Recurse to build the full multigrid hierarchy
//!
//! **Solve phase** (`sa_vcycle`):
//! - Pre-smooth with Gauss-Seidel or Jacobi
//! - Restrict residual to coarse grid
//! - Solve (recursively or exactly) on coarse grid
//! - Prolongate and correct
//! - Post-smooth
//!
//! # References
//!
//! - Vaněk, Mandel, Brezina (1996): "Algebraic multigrid based on smoothed aggregation"
//! - Stuben (2001): "A review of algebraic multigrid"

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use crate::iterative_solvers::{IterativeSolverConfig, Preconditioner, SolverResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::fmt::Debug;
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single level in the multigrid hierarchy.
#[derive(Debug)]
pub struct AmgLevel {
    /// System matrix on this level.
    pub a: CsrMatrix<f64>,
    /// Prolongation operator P (fine → coarse) — None on coarsest level.
    pub p: Option<CsrMatrix<f64>>,
    /// Restriction operator R = P^T (coarse → fine) — None on coarsest level.
    pub r: Option<CsrMatrix<f64>>,
    /// Pre-computed inverse diagonal of A for Jacobi smoother.
    pub diag_inv: Array1<f64>,
    /// Size of this level.
    pub size: usize,
}

/// Full SA-AMG hierarchy.
#[derive(Debug)]
pub struct AmgHierarchy {
    /// All levels from finest (index 0) to coarsest (index n_levels-1).
    pub levels: Vec<AmgLevel>,
}

// ---------------------------------------------------------------------------
// Strength-of-connection and aggregation
// ---------------------------------------------------------------------------

/// Compute the strength-of-connection graph.
///
/// Node `i` strongly influences node `j` when:
/// `|A_{ij}| >= epsilon * sqrt(|A_{ii}| * |A_{jj}|)`
///
/// # Returns
/// For each node i, a list of j indices that are strongly connected to i.
fn strength_of_connection(a: &CsrMatrix<f64>, epsilon_strong: f64) -> Vec<Vec<usize>> {
    let n = a.rows();
    let mut strong = vec![Vec::new(); n];

    // Precompute |A_{ii}|
    let diag: Vec<f64> = (0..n).map(|i| a.get(i, i).abs()).collect();

    for i in 0..n {
        let range = a.row_range(i);
        let d_ii = diag[i];
        for pos in range {
            let j = a.indices[pos];
            if j == i {
                continue;
            }
            let a_ij = a.data[pos].abs();
            let threshold = epsilon_strong * (d_ii * diag[j]).sqrt();
            if a_ij >= threshold {
                strong[i].push(j);
            }
        }
    }
    strong
}

/// Greedy aggregation algorithm.
///
/// Returns a mapping from node index to aggregate index.
/// Unassigned nodes are pulled into the nearest aggregate.
fn greedy_aggregate(n: usize, strong: &[Vec<usize>]) -> Vec<usize> {
    let mut agg_id = vec![usize::MAX; n];
    let mut next_agg = 0usize;

    // Phase 1: Seed aggregates from unaggregated nodes with strong neighbours
    for i in 0..n {
        if agg_id[i] != usize::MAX {
            continue;
        }
        // Check whether i has any unassigned strong neighbour
        let has_free_neighbour = strong[i].iter().any(|&j| agg_id[j] == usize::MAX);
        if !has_free_neighbour && !strong[i].is_empty() {
            continue; // defer
        }
        // Start a new aggregate: assign i and all unassigned strong neighbours
        agg_id[i] = next_agg;
        for &j in &strong[i] {
            if agg_id[j] == usize::MAX {
                agg_id[j] = next_agg;
            }
        }
        next_agg += 1;
    }

    // Phase 2: Assign remaining unassigned nodes
    for i in 0..n {
        if agg_id[i] != usize::MAX {
            continue;
        }
        // Find a strongly connected node that IS assigned
        let mut found = false;
        for &j in &strong[i] {
            if agg_id[j] != usize::MAX {
                agg_id[i] = agg_id[j];
                found = true;
                break;
            }
        }
        if !found {
            // Isolated node: start its own aggregate
            agg_id[i] = next_agg;
            next_agg += 1;
        }
    }

    agg_id
}

/// Convert aggregate assignment vector to Vec<Vec<usize>>.
///
/// `aggregates[k]` contains the fine-grid node indices belonging to aggregate k.
fn build_aggregate_sets(n: usize, agg_id: &[usize]) -> Vec<Vec<usize>> {
    let num_agg = agg_id.iter().copied().max().map(|v| v + 1).unwrap_or(0);
    let mut aggregates = vec![Vec::new(); num_agg];
    for i in 0..n {
        if agg_id[i] < num_agg {
            aggregates[agg_id[i]].push(i);
        }
    }
    aggregates
}

/// Public aggregation entry point.
///
/// Performs strength-of-connection analysis followed by greedy aggregation.
///
/// # Arguments
/// * `a` - System matrix (must be square)
/// * `epsilon_strong` - Strength threshold in (0, 1); 0.25 is typical
///
/// # Returns
/// A list of aggregate sets; each element lists the fine-grid nodes in that aggregate.
pub fn aggregate(a: &CsrMatrix<f64>, epsilon_strong: f64) -> SparseResult<Vec<Vec<usize>>> {
    if a.rows() != a.cols() {
        return Err(SparseError::ValueError(
            "Matrix must be square for aggregation".to_string(),
        ));
    }
    let n = a.rows();
    let strong = strength_of_connection(a, epsilon_strong);
    let agg_id = greedy_aggregate(n, &strong);
    Ok(build_aggregate_sets(n, &agg_id))
}

// ---------------------------------------------------------------------------
// Tentative prolongation
// ---------------------------------------------------------------------------

/// Build the tentative prolongation P₀.
///
/// For each aggregate `k`, the column block of P₀ is the local near-nullspace
/// vector, normalised so that P₀ P₀^T = I restricted to aggregate support.
///
/// With a single near-nullspace vector `v` (length n), the tentative prolongation
/// is the n × num_agg sparse matrix:
/// ```text
/// P₀[i, k] = v[i] / ||v|_agg_k||   if node i belongs to aggregate k
///           = 0                      otherwise
/// ```
///
/// # Arguments
/// * `aggregates` - List of aggregate sets from `aggregate()`
/// * `near_nullspace` - Near-nullspace vector of length n (often the constant vector 1)
pub fn tentative_prolongation(
    aggregates: &[Vec<usize>],
    near_nullspace: &Array1<f64>,
) -> SparseResult<CsrMatrix<f64>> {
    let n = near_nullspace.len();
    let num_agg = aggregates.len();

    // Validate aggregate coverage
    let mut covered = vec![false; n];
    for (k, agg) in aggregates.iter().enumerate() {
        for &i in agg {
            if i >= n {
                return Err(SparseError::ValueError(format!(
                    "Aggregate {k} contains node {i} which is out of range (n={n})"
                )));
            }
            if covered[i] {
                return Err(SparseError::ValueError(format!(
                    "Node {i} appears in more than one aggregate"
                )));
            }
            covered[i] = true;
        }
    }
    if covered.iter().any(|&c| !c) {
        return Err(SparseError::ValueError(
            "Some nodes are not covered by any aggregate".to_string(),
        ));
    }

    // Compute per-aggregate norm of the near-nullspace
    let mut agg_norms = vec![0.0_f64; num_agg];
    for (k, agg) in aggregates.iter().enumerate() {
        let sq: f64 = agg.iter().map(|&i| near_nullspace[i] * near_nullspace[i]).sum();
        agg_norms[k] = sq.sqrt();
    }

    // Assemble P₀ as a sparse matrix
    let nnz = aggregates.iter().map(|a| a.len()).sum();
    let mut rows = Vec::with_capacity(nnz);
    let mut cols = Vec::with_capacity(nnz);
    let mut vals = Vec::with_capacity(nnz);

    for (k, agg) in aggregates.iter().enumerate() {
        let norm = agg_norms[k];
        let scale = if norm > 1e-14 { 1.0 / norm } else { 1.0 };
        for &i in agg {
            rows.push(i);
            cols.push(k);
            vals.push(near_nullspace[i] * scale);
        }
    }

    CsrMatrix::from_triplets(n, num_agg, rows, cols, vals)
}

// ---------------------------------------------------------------------------
// Smoothed prolongation (Jacobi smoothing step)
// ---------------------------------------------------------------------------

/// Apply one step of Jacobi smoothing to the tentative prolongation P₀.
///
/// The smoothed prolongation is:
/// ```text
/// P = (I - omega * D^{-1} A) P₀
/// ```
/// where `D = diag(A)` and `omega in (0, 4/3)` (commonly 4/3).
///
/// # Arguments
/// * `p_tent` - Tentative prolongation (n × num_agg)
/// * `a` - System matrix (n × n)
/// * `omega` - Damping parameter (default 4/3)
pub fn smooth_prolongation(
    p_tent: &CsrMatrix<f64>,
    a: &CsrMatrix<f64>,
    omega: f64,
) -> SparseResult<CsrMatrix<f64>> {
    let n = a.rows();
    if p_tent.rows() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: p_tent.rows(),
        });
    }

    let num_agg = p_tent.cols();

    // Compute D^{-1} from the diagonal of A
    let mut d_inv = vec![1.0_f64; n];
    for i in 0..n {
        let d = a.get(i, i);
        if d.abs() > 1e-14 {
            d_inv[i] = 1.0 / d;
        }
    }

    // Build P = P₀ - omega * D^{-1} A P₀  using sparse column-by-column operations
    // We compute this as P_ij = P₀_ij - omega * sum_k D^{-1}_{ii} A_{ik} P₀_{kj}

    // Convert P_tent to dense for simplicity (it's n × num_agg, potentially large)
    // For large problems one would keep this sparse, but correctness first.
    let p0_dense = csr_to_dense(p_tent)?;

    // Compute A * P₀ (dense result)
    let mut ap0 = vec![vec![0.0_f64; num_agg]; n];
    for i in 0..n {
        let range = a.row_range(i);
        for pos in range {
            let k = a.indices[pos];
            let a_ik = a.data[pos];
            for j in 0..num_agg {
                ap0[i][j] += a_ik * p0_dense[k][j];
            }
        }
    }

    // P = P₀ - omega * D^{-1} (A P₀)
    let mut p_rows = Vec::new();
    let mut p_cols = Vec::new();
    let mut p_vals = Vec::new();

    const DROP_TOL: f64 = 1e-14;
    for i in 0..n {
        for j in 0..num_agg {
            let val = p0_dense[i][j] - omega * d_inv[i] * ap0[i][j];
            if val.abs() > DROP_TOL {
                p_rows.push(i);
                p_cols.push(j);
                p_vals.push(val);
            }
        }
    }

    CsrMatrix::from_triplets(n, num_agg, p_rows, p_cols, p_vals)
}

// ---------------------------------------------------------------------------
// Galerkin coarse-grid operator
// ---------------------------------------------------------------------------

/// Compute the coarse-grid operator A_c = P^T A P.
///
/// This is the standard Galerkin coarse-grid operator. The result is an
/// (num_agg × num_agg) symmetric sparse matrix.
fn galerkin_coarse(a: &CsrMatrix<f64>, p: &CsrMatrix<f64>) -> SparseResult<CsrMatrix<f64>> {
    let n = a.rows();
    let nc = p.cols();

    if p.rows() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: p.rows(),
        });
    }

    // Compute A P first (n × nc dense)
    let mut ap = vec![vec![0.0_f64; nc]; n];
    let p_dense = csr_to_dense(p)?;

    for i in 0..n {
        let range = a.row_range(i);
        for pos in range {
            let k = a.indices[pos];
            let a_ik = a.data[pos];
            for j in 0..nc {
                ap[i][j] += a_ik * p_dense[k][j];
            }
        }
    }

    // Now compute A_c = P^T (AP)  (nc × nc dense)
    let mut ac = vec![vec![0.0_f64; nc]; nc];
    for i in 0..n {
        for j in 0..nc {
            let p_ij = p_dense[i][j];
            if p_ij.abs() < 1e-14 {
                continue;
            }
            for k in 0..nc {
                ac[j][k] += p_ij * ap[i][k];
            }
        }
    }

    // Convert to CSR
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    const DROP_TOL: f64 = 1e-14;
    for i in 0..nc {
        for j in 0..nc {
            if ac[i][j].abs() > DROP_TOL {
                rows.push(i);
                cols.push(j);
                vals.push(ac[i][j]);
            }
        }
    }

    CsrMatrix::from_triplets(nc, nc, rows, cols, vals)
}

// ---------------------------------------------------------------------------
// Pre-compute diagonal inverse for a CsrMatrix
// ---------------------------------------------------------------------------

fn compute_diag_inv(a: &CsrMatrix<f64>) -> Array1<f64> {
    let n = a.rows();
    let mut d = Array1::ones(n);
    for i in 0..n {
        let v = a.get(i, i);
        if v.abs() > 1e-14 {
            d[i] = 1.0 / v;
        }
    }
    d
}

// ---------------------------------------------------------------------------
// Full SA-AMG setup
// ---------------------------------------------------------------------------

/// Perform the full SA-AMG setup phase.
///
/// Constructs a multigrid hierarchy from finest to coarsest level.
///
/// # Arguments
/// * `a` - System matrix (SPD, n×n)
/// * `near_nullspace` - Near-nullspace vector of length n (pass ones if unsure)
/// * `n_levels` - Maximum number of levels (including fine and coarse)
/// * `epsilon_strong` - Strength-of-connection threshold (0.25 is typical)
/// * `omega` - Jacobi smoothing parameter for prolongation (4/3 typical)
///
/// # Returns
/// An `AmgHierarchy` ready to be used in V-cycle iterations.
pub fn sa_setup(
    a: &CsrMatrix<f64>,
    near_nullspace: &Array1<f64>,
    n_levels: usize,
    epsilon_strong: f64,
    omega: f64,
) -> SparseResult<AmgHierarchy> {
    if a.rows() != a.cols() {
        return Err(SparseError::ValueError(
            "Matrix must be square for SA-AMG setup".to_string(),
        ));
    }
    if n_levels < 2 {
        return Err(SparseError::ValueError(
            "n_levels must be at least 2".to_string(),
        ));
    }
    if near_nullspace.len() != a.rows() {
        return Err(SparseError::DimensionMismatch {
            expected: a.rows(),
            found: near_nullspace.len(),
        });
    }

    let coarsest_size = 4; // stop coarsening below this threshold

    let mut levels: Vec<AmgLevel> = Vec::with_capacity(n_levels);

    let mut a_current = a.clone();
    let mut ns_current = near_nullspace.clone();

    for _level in 0..n_levels - 1 {
        let n_current = a_current.rows();
        if n_current <= coarsest_size {
            break;
        }

        // Aggregation
        let aggregates = aggregate(&a_current, epsilon_strong)?;
        let num_agg = aggregates.len();

        // Safety: if aggregation produced only one aggregate, stop coarsening
        if num_agg >= n_current || num_agg == 0 {
            break;
        }

        // Tentative prolongation
        let p_tent = tentative_prolongation(&aggregates, &ns_current)?;

        // Smoothed prolongation
        let p = smooth_prolongation(&p_tent, &a_current, omega)?;

        // Restriction = P^T
        let r = p.transpose();

        // Coarse-grid near-nullspace: restrict the fine-level constant
        // The coarse near-nullspace is R * ns_current, then normalise
        let nc = num_agg;
        let mut ns_coarse = Array1::zeros(nc);
        let r_dense = csr_to_dense(&r)?;
        for j in 0..nc {
            let mut acc = 0.0_f64;
            for i in 0..n_current {
                acc += r_dense[j][i] * ns_current[i];
            }
            ns_coarse[j] = acc;
        }
        // Normalise
        let norm: f64 = ns_coarse.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-14 {
            ns_coarse.mapv_inplace(|v| v / norm);
        }

        // Galerkin coarse operator
        let a_coarse = galerkin_coarse(&a_current, &p)?;

        let diag_inv = compute_diag_inv(&a_current);
        let size = n_current;

        levels.push(AmgLevel {
            a: a_current.clone(),
            p: Some(p),
            r: Some(r),
            diag_inv,
            size,
        });

        a_current = a_coarse;
        ns_current = ns_coarse;

        if a_current.rows() <= coarsest_size {
            break;
        }
    }

    // Push coarsest level (no P/R)
    let diag_inv = compute_diag_inv(&a_current);
    let size = a_current.rows();
    levels.push(AmgLevel {
        a: a_current,
        p: None,
        r: None,
        diag_inv,
        size,
    });

    if levels.is_empty() {
        return Err(SparseError::ComputationError(
            "SA-AMG setup produced no levels".to_string(),
        ));
    }

    Ok(AmgHierarchy { levels })
}

// ---------------------------------------------------------------------------
// V-cycle
// ---------------------------------------------------------------------------

/// Apply n_smooth steps of weighted Jacobi smoothing.
fn jacobi_smooth(
    a: &CsrMatrix<f64>,
    diag_inv: &Array1<f64>,
    b: &Array1<f64>,
    x: &mut Array1<f64>,
    n_smooth: usize,
    omega: f64,
) -> SparseResult<()> {
    let n = a.rows();
    let mut ax = Array1::zeros(n);
    for _ in 0..n_smooth {
        // ax = A * x
        for i in 0..n {
            let range = a.row_range(i);
            let mut acc = 0.0_f64;
            for pos in range {
                acc += a.data[pos] * x[a.indices[pos]];
            }
            ax[i] = acc;
        }
        // x = x + omega * D^{-1} (b - ax)
        for i in 0..n {
            x[i] += omega * diag_inv[i] * (b[i] - ax[i]);
        }
    }
    Ok(())
}

/// Compute residual r = b - A x.
fn residual(a: &CsrMatrix<f64>, b: &Array1<f64>, x: &Array1<f64>) -> SparseResult<Array1<f64>> {
    let n = a.rows();
    let mut r = b.clone();
    for i in 0..n {
        let range = a.row_range(i);
        let mut acc = 0.0_f64;
        for pos in range {
            acc += a.data[pos] * x[a.indices[pos]];
        }
        r[i] -= acc;
    }
    Ok(r)
}

/// Restrict a vector from fine to coarse grid using R = P^T.
fn restrict(r_mat: &CsrMatrix<f64>, v: &Array1<f64>) -> SparseResult<Array1<f64>> {
    let (m, n) = r_mat.shape();
    if v.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: v.len(),
        });
    }
    let mut result = Array1::zeros(m);
    for i in 0..m {
        let range = r_mat.row_range(i);
        let mut acc = 0.0_f64;
        for pos in range {
            acc += r_mat.data[pos] * v[r_mat.indices[pos]];
        }
        result[i] = acc;
    }
    Ok(result)
}

/// Prolongate a vector from coarse to fine grid using P.
fn prolongate(p_mat: &CsrMatrix<f64>, v: &Array1<f64>) -> SparseResult<Array1<f64>> {
    let (n, nc) = p_mat.shape();
    if v.len() != nc {
        return Err(SparseError::DimensionMismatch {
            expected: nc,
            found: v.len(),
        });
    }
    let mut result = Array1::zeros(n);
    for i in 0..n {
        let range = p_mat.row_range(i);
        let mut acc = 0.0_f64;
        for pos in range {
            acc += p_mat.data[pos] * v[p_mat.indices[pos]];
        }
        result[i] = acc;
    }
    Ok(result)
}

/// Apply a dense exact direct solve on the coarsest grid using Gaussian elimination.
fn coarsest_solve(a: &CsrMatrix<f64>, b: &Array1<f64>) -> SparseResult<Array1<f64>> {
    let n = a.rows();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    // Convert to dense
    let mut mat = vec![vec![0.0_f64; n]; n];
    let mut rhs = b.to_vec();

    for i in 0..n {
        let range = a.row_range(i);
        for pos in range {
            mat[i][a.indices[pos]] = a.data[pos];
        }
    }

    // Gaussian elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut pivot_row = col;
        let mut pivot_val = mat[col][col].abs();
        for row in (col + 1)..n {
            if mat[row][col].abs() > pivot_val {
                pivot_val = mat[row][col].abs();
                pivot_row = row;
            }
        }
        if pivot_val < 1e-14 {
            // Singular or near-singular: fall back to Jacobi iterate
            let mut x = Array1::zeros(n);
            for i in 0..n {
                let d = a.get(i, i);
                if d.abs() > 1e-14 {
                    x[i] = b[i] / d;
                }
            }
            return Ok(x);
        }
        // Swap rows
        if pivot_row != col {
            mat.swap(col, pivot_row);
            rhs.swap(col, pivot_row);
        }
        // Eliminate
        let pivot = mat[col][col];
        for row in (col + 1)..n {
            let factor = mat[row][col] / pivot;
            for k in col..n {
                let v = mat[col][k];
                mat[row][k] -= factor * v;
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    // Back-substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum -= mat[i][j] * x[j];
        }
        if mat[i][i].abs() < 1e-14 {
            x[i] = 0.0;
        } else {
            x[i] = sum / mat[i][i];
        }
    }
    Ok(Array1::from_vec(x))
}

/// Perform one SA-AMG V-cycle.
///
/// # Arguments
/// * `hierarchy` - The AMG hierarchy from `sa_setup`
/// * `b` - Right-hand side vector (length = fine-grid size)
/// * `x` - Current iterate (modified in-place); should have length = fine-grid size
/// * `n_pre` - Number of pre-smoothing steps
/// * `n_post` - Number of post-smoothing steps
/// * `level` - Current level index (0 = finest); called recursively
fn vcycle_recursive(
    hierarchy: &AmgHierarchy,
    b: &Array1<f64>,
    x: &mut Array1<f64>,
    n_pre: usize,
    n_post: usize,
    level: usize,
) -> SparseResult<()> {
    let lv = &hierarchy.levels[level];
    let is_coarsest = level == hierarchy.levels.len() - 1 || lv.p.is_none();

    if is_coarsest {
        // Direct solve on coarsest grid
        let sol = coarsest_solve(&lv.a, b)?;
        for i in 0..sol.len() {
            x[i] = sol[i];
        }
        return Ok(());
    }

    // Pre-smooth
    let smooth_omega = 2.0 / 3.0;
    jacobi_smooth(&lv.a, &lv.diag_inv, b, x, n_pre, smooth_omega)?;

    // Compute residual
    let r = residual(&lv.a, b, x)?;

    // Restrict to coarse grid
    let r_mat = lv.r.as_ref().ok_or_else(|| {
        SparseError::ComputationError("No restriction operator at non-coarsest level".to_string())
    })?;
    let r_coarse = restrict(r_mat, &r)?;

    // Recursive coarse-grid correction
    let nc = r_coarse.len();
    let mut e_coarse = Array1::zeros(nc);
    vcycle_recursive(hierarchy, &r_coarse, &mut e_coarse, n_pre, n_post, level + 1)?;

    // Prolongate correction
    let p_mat = lv.p.as_ref().ok_or_else(|| {
        SparseError::ComputationError("No prolongation operator at non-coarsest level".to_string())
    })?;
    let e_fine = prolongate(p_mat, &e_coarse)?;

    // Correct: x = x + e_fine
    for i in 0..x.len() {
        x[i] += e_fine[i];
    }

    // Post-smooth
    jacobi_smooth(&lv.a, &lv.diag_inv, b, x, n_post, smooth_omega)?;

    Ok(())
}

/// Apply one SA-AMG V-cycle starting from the finest level.
///
/// # Arguments
/// * `hierarchy` - AMG hierarchy from `sa_setup`
/// * `b` - Right-hand side vector (length n)
/// * `x` - Initial guess vector (length n); updated in place
/// * `n_pre` - Number of pre-smoothing Jacobi steps per level
/// * `n_post` - Number of post-smoothing Jacobi steps per level
///
/// # Returns
/// The updated solution vector.
pub fn sa_vcycle(
    hierarchy: &AmgHierarchy,
    b: &Array1<f64>,
    x: &Array1<f64>,
    n_pre: usize,
    n_post: usize,
) -> SparseResult<Array1<f64>> {
    if hierarchy.levels.is_empty() {
        return Err(SparseError::ComputationError(
            "Empty AMG hierarchy".to_string(),
        ));
    }
    let n = hierarchy.levels[0].size;
    if b.len() != n || x.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: if b.len() != n { b.len() } else { x.len() },
        });
    }
    let mut x_out = x.clone();
    vcycle_recursive(hierarchy, b, &mut x_out, n_pre, n_post, 0)?;
    Ok(x_out)
}

// ---------------------------------------------------------------------------
// AMG preconditioner wrapper (implements Preconditioner trait)
// ---------------------------------------------------------------------------

/// SA-AMG preconditioner wrapping one V-cycle application.
pub struct SaAmgPreconditioner<'a> {
    hierarchy: &'a AmgHierarchy,
    n_pre: usize,
    n_post: usize,
}

impl<'a> SaAmgPreconditioner<'a> {
    /// Create from a pre-built hierarchy.
    pub fn new(hierarchy: &'a AmgHierarchy, n_pre: usize, n_post: usize) -> Self {
        Self {
            hierarchy,
            n_pre,
            n_post,
        }
    }
}

impl<'a> Preconditioner<f64> for SaAmgPreconditioner<'a> {
    fn apply(&self, r: &Array1<f64>) -> crate::error::SparseResult<Array1<f64>> {
        let x0 = Array1::zeros(r.len());
        sa_vcycle(self.hierarchy, r, &x0, self.n_pre, self.n_post)
    }
}

// ---------------------------------------------------------------------------
// Full SA-AMG solver
// ---------------------------------------------------------------------------

/// Solve `A x = b` using SA-AMG as a preconditioner for PCG.
///
/// # Algorithm
/// 1. Build the SA-AMG hierarchy from `A` and a constant near-nullspace vector
/// 2. Run Preconditioned Conjugate Gradient with one V-cycle per PCG step
///
/// # Arguments
/// * `a` - SPD system matrix (n×n)
/// * `b` - Right-hand side vector (length n)
/// * `tol` - Relative convergence tolerance (e.g. 1e-8)
/// * `max_iter` - Maximum number of PCG iterations
///
/// # Returns
/// A `SolverResult` with the computed solution.
pub fn sa_solve(
    a: &CsrMatrix<f64>,
    b: &Array1<f64>,
    tol: f64,
    max_iter: usize,
) -> SparseResult<SolverResult<f64>> {
    let n = a.rows();
    if a.cols() != n {
        return Err(SparseError::ValueError(
            "Matrix must be square".to_string(),
        ));
    }
    if b.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: b.len(),
        });
    }

    // Build near-nullspace (constant vector, normalised)
    let scale = 1.0 / (n as f64).sqrt();
    let near_nullspace = Array1::from_elem(n, scale);

    // Build AMG hierarchy
    let hierarchy = sa_setup(a, &near_nullspace, 10, 0.25, 4.0 / 3.0)?;

    // Preconditioned CG
    let amg_pc = SaAmgPreconditioner::new(&hierarchy, 2, 2);

    let config = IterativeSolverConfig {
        max_iter,
        tol,
        verbose: false,
    };

    pcg(a, b, &config, Some(&amg_pc as &dyn Preconditioner<f64>))
}

// ---------------------------------------------------------------------------
// Minimal PCG implementation
// ---------------------------------------------------------------------------

fn pcg(
    a: &CsrMatrix<f64>,
    b: &Array1<f64>,
    config: &IterativeSolverConfig,
    precond: Option<&dyn Preconditioner<f64>>,
) -> SparseResult<SolverResult<f64>> {
    let n = a.rows();
    let mut x = Array1::zeros(n);

    let mut r = b.clone();
    let bnorm = norm2_arr(&r);
    if bnorm < 1e-30 {
        return Ok(SolverResult {
            solution: x,
            n_iter: 0,
            residual_norm: 0.0,
            converged: true,
        });
    }

    let tol_abs = config.tol * bnorm;

    let mut z = match precond {
        Some(pc) => pc.apply(&r)?,
        None => r.clone(),
    };

    let mut p = z.clone();
    let mut rz = dot_arr(&r, &z);

    for k in 0..config.max_iter {
        let ap = csr_matvec_arr(a, &p)?;
        let pap = dot_arr(&p, &ap);
        if pap.abs() < 1e-30 {
            break;
        }
        let alpha = rz / pap;
        axpy_mut_arr(&mut x, alpha, &p);
        axpy_mut_arr(&mut r, -alpha, &ap);

        let rnorm = norm2_arr(&r);
        if rnorm <= tol_abs {
            return Ok(SolverResult {
                solution: x,
                n_iter: k + 1,
                residual_norm: rnorm,
                converged: true,
            });
        }

        z = match precond {
            Some(pc) => pc.apply(&r)?,
            None => r.clone(),
        };

        let rz_new = dot_arr(&r, &z);
        let beta = rz_new / rz;
        for (pi, &zi) in p.iter_mut().zip(z.iter()) {
            *pi = zi + beta * *pi;
        }
        rz = rz_new;
    }

    let rnorm = norm2_arr(&r);
    Ok(SolverResult {
        solution: x,
        n_iter: config.max_iter,
        residual_norm: rnorm,
        converged: rnorm <= tol_abs,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn csr_matvec_arr(a: &CsrMatrix<f64>, x: &Array1<f64>) -> SparseResult<Array1<f64>> {
    let (m, n) = a.shape();
    if x.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: x.len(),
        });
    }
    let mut y = Array1::zeros(m);
    for i in 0..m {
        let range = a.row_range(i);
        let mut acc = 0.0_f64;
        for pos in range {
            acc += a.data[pos] * x[a.indices[pos]];
        }
        y[i] = acc;
    }
    Ok(y)
}

#[inline]
fn dot_arr(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

#[inline]
fn norm2_arr(v: &Array1<f64>) -> f64 {
    dot_arr(v, v).sqrt()
}

#[inline]
fn axpy_mut_arr(y: &mut Array1<f64>, alpha: f64, x: &Array1<f64>) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// Convert a CSR matrix to a dense Vec<Vec<f64>> (rows × cols).
fn csr_to_dense(a: &CsrMatrix<f64>) -> SparseResult<Vec<Vec<f64>>> {
    let (m, n) = a.shape();
    let mut dense = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        let range = a.row_range(i);
        for pos in range {
            dense[i][a.indices[pos]] = a.data[pos];
        }
    }
    Ok(dense)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Build a simple 1D Laplacian as a CSR matrix.
    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(2.0_f64);
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                vals.push(-1.0_f64);
                rows.push(i - 1);
                cols.push(i);
                vals.push(-1.0_f64);
            }
        }
        CsrMatrix::from_triplets(n, n, rows, cols, vals).expect("valid test setup")
    }

    #[test]
    fn test_aggregate_small() {
        let a = laplacian_1d(8);
        let aggs = aggregate(&a, 0.25).expect("valid test setup");
        // Every node should be in exactly one aggregate
        let mut covered = vec![false; 8];
        for agg in &aggs {
            for &i in agg {
                assert!(!covered[i], "Node {i} covered twice");
                covered[i] = true;
            }
        }
        assert!(covered.iter().all(|&c| c));
    }

    #[test]
    fn test_tentative_prolongation_partition_of_unity() {
        let a = laplacian_1d(8);
        let aggs = aggregate(&a, 0.25).expect("valid test setup");
        let ns = Array1::ones(8);
        let p = tentative_prolongation(&aggs, &ns).expect("valid test setup");

        // Each row of P should have exactly one non-zero (the aggregate it belongs to)
        assert_eq!(p.rows(), 8);
        assert_eq!(p.cols(), aggs.len());

        // Sum of squares in each column = 1 (each aggregate column is normalised)
        for k in 0..p.cols() {
            let mut sq_sum = 0.0_f64;
            for i in 0..8 {
                let v = p.get(i, k);
                sq_sum += v * v;
            }
            assert_relative_eq!(sq_sum, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_smooth_prolongation_shape() {
        let a = laplacian_1d(8);
        let aggs = aggregate(&a, 0.25).expect("valid test setup");
        let ns = Array1::ones(8);
        let p_tent = tentative_prolongation(&aggs, &ns).expect("valid test setup");
        let p = smooth_prolongation(&p_tent, &a, 4.0 / 3.0).expect("valid test setup");
        assert_eq!(p.rows(), 8);
        assert_eq!(p.cols(), aggs.len());
    }

    #[test]
    fn test_sa_setup_levels() {
        let n = 32;
        let a = laplacian_1d(n);
        let ns = Array1::ones(n);
        let hierarchy = sa_setup(&a, &ns, 5, 0.25, 4.0 / 3.0).expect("valid test setup");
        assert!(hierarchy.levels.len() >= 2);
        // Coarsest level must be smaller than finest
        assert!(hierarchy.levels.last().expect("hierarchy has at least one level").size < n);
    }

    #[test]
    fn test_sa_vcycle_reduces_residual() {
        let n = 16;
        let a = laplacian_1d(n);
        let ns = Array1::ones(n);
        let b = Array1::ones(n);
        let hierarchy = sa_setup(&a, &ns, 5, 0.25, 4.0 / 3.0).expect("valid test setup");

        let x0 = Array1::zeros(n);
        let x1 = sa_vcycle(&hierarchy, &b, &x0, 2, 2).expect("valid test setup");

        // Compute residual norms
        let r0: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        let r1 = residual(&a, &b, &x1).expect("valid test setup");
        let r1_norm: f64 = r1.iter().map(|v| v * v).sum::<f64>().sqrt();

        assert!(
            r1_norm < r0,
            "V-cycle should reduce residual: {r1_norm} >= {r0}"
        );
    }

    #[test]
    fn test_sa_solve_convergence() {
        let n = 16;
        let a = laplacian_1d(n);
        // Create an RHS consistent with a known solution x* = [1,1,...,1]
        // A * 1 = [1, 0, ..., 0, 1] for a 1D Laplacian (boundary effects)
        // Use a simple RHS
        let b = Array1::ones(n);
        let result = sa_solve(&a, &b, 1e-8, 200).expect("valid test setup");
        assert!(
            result.converged,
            "sa_solve did not converge: residual={}",
            result.residual_norm
        );
        assert_eq!(result.solution.len(), n);
    }
}
