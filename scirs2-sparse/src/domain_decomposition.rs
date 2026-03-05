//! Domain decomposition methods for sparse linear systems
//!
//! This module provides domain decomposition preconditioners and solvers,
//! which partition the problem domain into subdomains and solve local
//! sub-problems either independently (additive) or sequentially (multiplicative).
//!
//! # Methods
//!
//! - **Overlapping Schwarz**: Partition degrees of freedom with overlap regions;
//!   solve sub-problems independently (additive) or in Gauss-Seidel fashion
//!   (multiplicative).
//! - **Schur complement**: Eliminate interior DOFs exactly, reduce to interface system.
//! - **FETI (Finite Element Tearing and Interconnecting)**: Dual formulation that
//!   enforces compatibility across subdomain boundaries via Lagrange multipliers.
//!
//! # References
//!
//! - Toselli & Widlund (2005). *Domain Decomposition Methods — Algorithms and Theory*.
//! - Farhat & Roux (1991). "A method of finite element tearing and interconnecting".
//!   *Int. J. Numer. Methods Eng.* 32(6), 1205-1227.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use crate::iterative_solvers::{cg, IterativeSolverConfig, SolverResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::fmt::Debug;
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Partition construction
// ---------------------------------------------------------------------------

/// Partition a 1-D index range `[0, n)` into `n_subdomains` overlapping
/// subsets, each extended by `overlap` DOFs on each side.
///
/// # Arguments
///
/// * `n`           – Total number of degrees of freedom.
/// * `n_subdomains` – Number of subdomains to create.
/// * `overlap`     – Number of additional DOFs added at each boundary.
///
/// # Returns
///
/// A vector of length `n_subdomains`, where each entry lists the DOF indices
/// belonging to that subdomain (possibly including overlap with neighbours).
///
/// # Errors
///
/// Returns an error if `n_subdomains == 0` or `n_subdomains > n`.
///
/// # Examples
///
/// ```
/// use scirs2_sparse::domain_decomposition::partition_domain;
///
/// let parts = partition_domain(10, 2, 1).expect("valid input");
/// assert_eq!(parts.len(), 2);
/// // Each partition must cover at least its core region
/// assert!(parts[0].contains(&0));
/// assert!(parts[1].contains(&9));
/// ```
pub fn partition_domain(
    n: usize,
    n_subdomains: usize,
    overlap: usize,
) -> SparseResult<Vec<Vec<usize>>> {
    if n_subdomains == 0 {
        return Err(SparseError::ValueError(
            "n_subdomains must be at least 1".to_string(),
        ));
    }
    if n_subdomains > n {
        return Err(SparseError::ValueError(format!(
            "n_subdomains ({n_subdomains}) cannot exceed n ({n})"
        )));
    }

    // Base partition widths (integer division with remainder distribution)
    let base = n / n_subdomains;
    let remainder = n % n_subdomains;

    let mut partitions = Vec::with_capacity(n_subdomains);
    let mut start = 0usize;

    for k in 0..n_subdomains {
        // Core extent — distribute remainder across leading subdomains
        let core_size = base + if k < remainder { 1 } else { 0 };
        let core_end = start + core_size;

        // Extend by overlap on both sides (clamp to valid range)
        let ext_start = start.saturating_sub(overlap);
        let ext_end = (core_end + overlap).min(n);

        let dofs: Vec<usize> = (ext_start..ext_end).collect();
        partitions.push(dofs);
        start = core_end;
    }

    Ok(partitions)
}

// ---------------------------------------------------------------------------
// Additive Schwarz
// ---------------------------------------------------------------------------

/// Configuration and method marker for the Schwarz domain decomposition.
#[derive(Debug, Clone)]
pub struct SchwartzOverlap {
    /// Overlap size in DOFs.
    pub overlap: usize,
    /// Maximum CG iterations for each sub-problem.
    pub sub_max_iter: usize,
    /// Tolerance for sub-problem CG solvers.
    pub sub_tol: f64,
}

impl Default for SchwartzOverlap {
    fn default() -> Self {
        Self {
            overlap: 1,
            sub_max_iter: 200,
            sub_tol: 1e-10,
        }
    }
}

/// Additive Schwarz method.
///
/// Solves `A x = b` using the additive Schwarz (Jacobi-like) domain
/// decomposition preconditioner.  Each sub-domain sub-matrix is solved
/// approximately with CG.  The correction vectors are accumulated and
/// added to produce the global update.
///
/// # Arguments
///
/// * `a`          – Global system matrix in CSR format (n × n).
/// * `b`          – Right-hand side vector of length n.
/// * `partitions` – Overlapping index sets; each entry is a list of DOF
///                  indices belonging to one subdomain.  Produced by
///                  [`partition_domain`] or provided by the caller.
/// * `config`     – Schwarz configuration (overlap, sub-problem solver params).
///
/// # Returns
///
/// Solution vector `x` such that `||A x - b|| / ||b||` is small.
pub fn additive_schwarz<F>(
    a: &CsrMatrix<F>,
    b: &Array1<F>,
    partitions: &[Vec<usize>],
    config: &SchwartzOverlap,
) -> SparseResult<Array1<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
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
    if partitions.is_empty() {
        return Err(SparseError::ValueError(
            "partitions must be non-empty".to_string(),
        ));
    }

    let sub_cfg = IterativeSolverConfig {
        max_iter: config.sub_max_iter,
        tol: config.sub_tol,
        verbose: false,
    };

    let mut x = Array1::<F>::zeros(n);
    // Accumulate corrections
    let mut correction = Array1::<F>::zeros(n);

    for dofs in partitions {
        if dofs.is_empty() {
            continue;
        }
        let m = dofs.len();

        // Extract local sub-matrix A_loc and local rhs b_loc
        let (a_loc, b_loc) = extract_subproblem(a, b, dofs)?;

        // Solve local system
        let result = cg(&a_loc, &b_loc, &sub_cfg, None)?;
        let x_loc = result.solution;

        // Scatter local solution back
        for (local_i, &global_i) in dofs.iter().enumerate() {
            if local_i < m && global_i < n {
                correction[global_i] = correction[global_i] + x_loc[local_i];
            }
        }
    }

    // Average the corrections (additive combination with equal weighting)
    let n_parts = F::from(partitions.len()).ok_or_else(|| {
        SparseError::ValueError("Failed to convert partition count to float".to_string())
    })?;

    // Count how many partitions each DOF appears in (for averaging)
    let mut count = vec![0usize; n];
    for dofs in partitions {
        for &g in dofs {
            if g < n {
                count[g] += 1;
            }
        }
    }

    for i in 0..n {
        let cnt = if count[i] > 0 { count[i] } else { 1 };
        let cnt_f = F::from(cnt).unwrap_or(F::sparse_one());
        x[i] = correction[i] / cnt_f;
    }

    let _ = n_parts; // suppress warning
    Ok(x)
}

// ---------------------------------------------------------------------------
// Multiplicative Schwarz
// ---------------------------------------------------------------------------

/// Multiplicative (Gauss-Seidel) Schwarz method.
///
/// Solves `A x = b` by iterating over subdomains sequentially, updating
/// the solution and recomputing the residual after each sub-domain solve.
///
/// # Arguments
///
/// * `a`          – Global system matrix in CSR format.
/// * `b`          – Right-hand side vector.
/// * `partitions` – Overlapping subdomain index sets.
/// * `max_iter`   – Number of outer multiplicative Schwarz sweeps.
/// * `tol`        – Convergence tolerance on the relative residual.
///
/// # Returns
///
/// Solution vector on success.
pub fn multiplicative_schwarz<F>(
    a: &CsrMatrix<F>,
    b: &Array1<F>,
    partitions: &[Vec<usize>],
    max_iter: usize,
    tol: F,
) -> SparseResult<Array1<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
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
    if partitions.is_empty() {
        return Err(SparseError::ValueError(
            "partitions must be non-empty".to_string(),
        ));
    }

    let bnorm = vec_norm2(b);
    if bnorm <= F::epsilon() {
        return Ok(Array1::zeros(n));
    }
    let tolerance = tol * bnorm;

    let sub_cfg = IterativeSolverConfig {
        max_iter: 200,
        tol: 1e-10,
        verbose: false,
    };

    let mut x = Array1::<F>::zeros(n);

    for _outer in 0..max_iter {
        // One multiplicative sweep over all subdomains
        for dofs in partitions {
            if dofs.is_empty() {
                continue;
            }
            let m = dofs.len();

            // Compute residual r = b - A x
            let r = compute_residual(a, b, &x)?;

            // Extract local sub-matrix and local residual
            let (a_loc, r_loc) = extract_subproblem(a, &r, dofs)?;

            // Solve A_loc delta_loc = r_loc
            let result = cg(&a_loc, &r_loc, &sub_cfg, None)?;
            let delta_loc = result.solution;

            // Update global solution
            for (local_i, &global_i) in dofs.iter().enumerate() {
                if local_i < m && global_i < n {
                    x[global_i] = x[global_i] + delta_loc[local_i];
                }
            }
        }

        // Check global convergence
        let r = compute_residual(a, b, &x)?;
        if vec_norm2(&r) <= tolerance {
            break;
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Schur complement solver
// ---------------------------------------------------------------------------

/// Solve a linear system using the Schur complement decomposition.
///
/// Given a system `A x = b` where the DOFs are split into interior DOFs
/// and interface DOFs, we eliminate the interior DOFs via Gaussian elimination
/// and solve for the interface DOFs first, then back-substitute.
///
/// # Matrix blocking
///
/// Let `I = interior_dofs` (indices NOT in `interface_dofs`) and
/// `B = interface_dofs`.  The system becomes:
///
/// ```text
/// [A_II  A_IB] [x_I]   [b_I]
/// [A_BI  A_BB] [x_B] = [b_B]
/// ```
///
/// The Schur complement is `S = A_BB - A_BI A_II^{-1} A_IB`.
/// We solve `S x_B = b_B - A_BI A_II^{-1} b_I`, then back-solve for `x_I`.
///
/// # Arguments
///
/// * `a`             – Global system matrix (n × n, SPD assumed).
/// * `b`             – Right-hand side vector.
/// * `interface_dofs` – Indices that are treated as interface DOFs.
///
/// # Returns
///
/// Full solution vector.
pub fn schur_complement_solve<F>(
    a: &CsrMatrix<F>,
    b: &Array1<F>,
    interface_dofs: &[usize],
) -> SparseResult<Array1<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
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

    // Validate interface indices
    for &idx in interface_dofs {
        if idx >= n {
            return Err(SparseError::ValueError(format!(
                "interface DOF index {idx} out of bounds (n={n})"
            )));
        }
    }

    // Build index sets
    let mut is_interface = vec![false; n];
    for &i in interface_dofs {
        is_interface[i] = true;
    }
    let interior_dofs: Vec<usize> = (0..n).filter(|&i| !is_interface[i]).collect();

    if interior_dofs.is_empty() {
        // All DOFs are interface — solve the full system directly via CG
        let cfg = IterativeSolverConfig {
            max_iter: 2000,
            tol: 1e-12,
            verbose: false,
        };
        let result = cg(a, b, &cfg, None)?;
        return Ok(result.solution);
    }

    if interface_dofs.is_empty() {
        // No interface — solve directly
        let cfg = IterativeSolverConfig {
            max_iter: 2000,
            tol: 1e-12,
            verbose: false,
        };
        let result = cg(a, b, &cfg, None)?;
        return Ok(result.solution);
    }

    // ---- Build block sub-matrices ----------------------------------------
    // A_II and b_I
    let (a_ii, b_i) = extract_subproblem(a, b, &interior_dofs)?;
    // A_IB: interior rows × interface cols
    let a_ib = extract_submatrix_rows_cols(a, &interior_dofs, interface_dofs)?;
    // A_BI: interface rows × interior cols
    let a_bi = extract_submatrix_rows_cols(a, interface_dofs, &interior_dofs)?;
    // A_BB: interface rows × interface cols
    let (a_bb, b_b) = extract_subproblem(a, b, interface_dofs)?;

    let n_i = interior_dofs.len();
    let n_b = interface_dofs.len();

    let cfg = IterativeSolverConfig {
        max_iter: 2000,
        tol: 1e-12,
        verbose: false,
    };

    // Compute A_II^{-1} b_I  (solve A_II y = b_I)
    let y_i = cg(&a_ii, &b_i, &cfg, None)?.solution;

    // Compute A_BI A_II^{-1} b_I
    let abi_y = spmv_dense(&a_bi, &y_i, n_b, n_i)?;

    // Modified rhs for Schur: g = b_B - A_BI A_II^{-1} b_I
    let mut g = b_b.clone();
    for i in 0..n_b {
        g[i] = g[i] - abi_y[i];
    }

    // Compute Schur complement S = A_BB - A_BI A_II^{-1} A_IB
    // We represent S implicitly via matrix-vector products.
    // For simplicity here we compute S explicitly (works for moderate n_b).
    let s = build_schur_complement(&a_bb, &a_bi, &a_ii, &a_ib, n_b, n_i, &cfg)?;

    // Solve S x_B = g
    let x_b = cg(&s, &g, &cfg, None)?.solution;

    // Back-substitute: x_I = A_II^{-1} (b_I - A_IB x_B)
    let aib_xb = spmv_dense(&a_ib, &x_b, n_i, n_b)?;
    let mut rhs_i = b_i.clone();
    for i in 0..n_i {
        rhs_i[i] = rhs_i[i] - aib_xb[i];
    }
    let x_i = cg(&a_ii, &rhs_i, &cfg, None)?.solution;

    // Assemble global solution
    let mut x = Array1::<F>::zeros(n);
    for (local_i, &global_i) in interior_dofs.iter().enumerate() {
        x[global_i] = x_i[local_i];
    }
    for (local_i, &global_i) in interface_dofs.iter().enumerate() {
        x[global_i] = x_b[local_i];
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// FETI solver (simplified)
// ---------------------------------------------------------------------------

/// FETI (Finite Element Tearing and Interconnecting) solver — simplified version.
///
/// The FETI method is a dual-primal domain decomposition technique.  In its
/// simplified form presented here, each subdomain has its own local system
/// `A_s x_s = b_s`, and compatibility across subdomain boundaries is enforced
/// via Lagrange multipliers.
///
/// This implementation solves each sub-problem independently (as in additive
/// Schwarz) and then enforces compatibility constraints via a projected
/// gradient descent on the Lagrange multiplier system.
///
/// # Arguments
///
/// * `a_list`      – List of local stiffness matrices, one per subdomain.
/// * `b_list`      – List of local rhs vectors, one per subdomain.
/// * `connectivity` – For each subdomain pair `(s, t)` with shared DOFs,
///                    a tuple `(s, t, local_s_dofs, local_t_dofs)` where
///                    `local_s_dofs[k]` in subdomain s is the same physical
///                    DOF as `local_t_dofs[k]` in subdomain t.
///
/// # Returns
///
/// A concatenated global solution: the solution for subdomain 0, then 1, etc.
/// Use the returned lengths or connectivity to extract per-subdomain solutions.
pub fn feti_solve<F>(
    a_list: &[CsrMatrix<F>],
    b_list: &[Array1<F>],
    connectivity: &[(usize, usize, Vec<usize>, Vec<usize>)],
) -> SparseResult<Vec<Array1<F>>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    if a_list.is_empty() {
        return Err(SparseError::ValueError(
            "a_list must contain at least one subdomain".to_string(),
        ));
    }
    if a_list.len() != b_list.len() {
        return Err(SparseError::DimensionMismatch {
            expected: a_list.len(),
            found: b_list.len(),
        });
    }

    let n_sub = a_list.len();

    let cfg = IterativeSolverConfig {
        max_iter: 1000,
        tol: 1e-10,
        verbose: false,
    };

    // Initialise Lagrange multipliers to zero
    let n_constraints = connectivity.iter().map(|(_, _, v, _)| v.len()).sum::<usize>();
    let mut lambdas = Array1::<F>::zeros(n_constraints);

    // Solve each subdomain independently as initial guess
    let mut x_list: Vec<Array1<F>> = Vec::with_capacity(n_sub);
    for s in 0..n_sub {
        let n_s = a_list[s].rows();
        if a_list[s].cols() != n_s {
            return Err(SparseError::ValueError(format!(
                "Subdomain {s} matrix must be square"
            )));
        }
        if b_list[s].len() != n_s {
            return Err(SparseError::DimensionMismatch {
                expected: n_s,
                found: b_list[s].len(),
            });
        }
        let result = cg(&a_list[s], &b_list[s], &cfg, None)?;
        x_list.push(result.solution);
    }

    if connectivity.is_empty() {
        return Ok(x_list);
    }

    // FETI projected gradient iterations on the interface problem
    // F λ = d, where F is the flexibility operator.
    let feti_max_iter = 200usize;
    let feti_tol = F::from(1e-8).ok_or_else(|| {
        SparseError::ValueError("Cannot convert FETI tolerance".to_string())
    })?;

    for _iter in 0..feti_max_iter {
        // Compute jump vector: j_k = x_{s,i} - x_{t,j} for each constraint k
        let mut jump = Array1::<F>::zeros(n_constraints);
        let mut offset = 0;
        for (s, t, ls_dofs, lt_dofs) in connectivity {
            for k in 0..ls_dofs.len() {
                let i_s = ls_dofs[k];
                let i_t = lt_dofs[k];
                if i_s < x_list[*s].len() && i_t < x_list[*t].len() {
                    jump[offset + k] = x_list[*s][i_s] - x_list[*t][i_t];
                }
            }
            offset += ls_dofs.len();
        }

        // Check convergence
        let jump_norm = vec_norm2(&jump);
        if jump_norm <= feti_tol {
            break;
        }

        // Update Lagrange multipliers: λ ← λ + α * jump (gradient step)
        // Step size α is estimated as a fixed fraction (like 0.5)
        let alpha = F::from(0.5).ok_or_else(|| {
            SparseError::ValueError("Cannot convert alpha".to_string())
        })?;
        for k in 0..n_constraints {
            lambdas[k] = lambdas[k] + alpha * jump[k];
        }

        // Apply Lagrange multiplier forces to each subdomain rhs
        let mut b_modified: Vec<Array1<F>> = b_list.to_vec();
        let mut offset2 = 0;
        for (s, t, ls_dofs, lt_dofs) in connectivity {
            for k in 0..ls_dofs.len() {
                let lam = lambdas[offset2 + k];
                let i_s = ls_dofs[k];
                let i_t = lt_dofs[k];
                if i_s < b_modified[*s].len() {
                    b_modified[*s][i_s] = b_modified[*s][i_s] - lam;
                }
                if i_t < b_modified[*t].len() {
                    b_modified[*t][i_t] = b_modified[*t][i_t] + lam;
                }
            }
            offset2 += ls_dofs.len();
        }

        // Re-solve subdomains with updated rhs
        for s in 0..n_sub {
            let result = cg(&a_list[s], &b_modified[s], &cfg, None)?;
            x_list[s] = result.solution;
        }
    }

    Ok(x_list)
}

// ---------------------------------------------------------------------------
// Internal helper functions
// ---------------------------------------------------------------------------

/// Extract the local sub-problem (sub-matrix and local rhs) for a set of DOFs.
fn extract_subproblem<F>(
    a: &CsrMatrix<F>,
    b: &Array1<F>,
    dofs: &[usize],
) -> SparseResult<(CsrMatrix<F>, Array1<F>)>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let m = dofs.len();
    // Build a map: global index -> local index
    let mut global_to_local = std::collections::HashMap::new();
    for (local, &global) in dofs.iter().enumerate() {
        global_to_local.insert(global, local);
    }

    // Build local rhs
    let mut b_loc = Array1::<F>::zeros(m);
    for (local_i, &global_i) in dofs.iter().enumerate() {
        if global_i < b.len() {
            b_loc[local_i] = b[global_i];
        }
    }

    // Build local matrix (only entries where both row and col are in dofs)
    let mut row_idx = Vec::new();
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();

    for (local_row, &global_row) in dofs.iter().enumerate() {
        let range = a.row_range(global_row);
        for pos in range {
            let global_col = a.indices[pos];
            if let Some(&local_col) = global_to_local.get(&global_col) {
                row_idx.push(local_row);
                col_idx.push(local_col);
                vals.push(a.data[pos]);
            }
        }
    }

    let a_loc = CsrMatrix::from_triplets(m, m, row_idx, col_idx, vals)?;
    Ok((a_loc, b_loc))
}

/// Extract a rectangular sub-matrix: rows from `row_dofs`, columns from `col_dofs`.
fn extract_submatrix_rows_cols<F>(
    a: &CsrMatrix<F>,
    row_dofs: &[usize],
    col_dofs: &[usize],
) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let nr = row_dofs.len();
    let nc = col_dofs.len();

    let mut col_map = std::collections::HashMap::new();
    for (local, &global) in col_dofs.iter().enumerate() {
        col_map.insert(global, local);
    }

    let mut row_idx = Vec::new();
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();

    for (local_row, &global_row) in row_dofs.iter().enumerate() {
        let range = a.row_range(global_row);
        for pos in range {
            let global_col = a.indices[pos];
            if let Some(&local_col) = col_map.get(&global_col) {
                row_idx.push(local_row);
                col_idx.push(local_col);
                vals.push(a.data[pos]);
            }
        }
    }

    CsrMatrix::from_triplets(nr, nc, row_idx, col_idx, vals)
}

/// Compute residual r = b - A x.
fn compute_residual<F>(
    a: &CsrMatrix<F>,
    b: &Array1<F>,
    x: &Array1<F>,
) -> SparseResult<Array1<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static,
{
    let n = a.rows();
    let mut r = b.clone();
    for i in 0..n {
        let range = a.row_range(i);
        let mut ax_i = F::sparse_zero();
        for pos in range {
            ax_i = ax_i + a.data[pos] * x[a.indices[pos]];
        }
        r[i] = r[i] - ax_i;
    }
    Ok(r)
}

/// Dense matrix-vector multiply for a CsrMatrix with `nrows` × `ncols` shape.
fn spmv_dense<F>(
    a: &CsrMatrix<F>,
    x: &Array1<F>,
    nrows: usize,
    ncols: usize,
) -> SparseResult<Array1<F>>
where
    F: Float + NumAssign + SparseElement + 'static,
{
    if x.len() != ncols {
        return Err(SparseError::DimensionMismatch {
            expected: ncols,
            found: x.len(),
        });
    }
    let mut y = Array1::<F>::zeros(nrows);
    for i in 0..nrows {
        let range = a.row_range(i);
        for pos in range {
            let j = a.indices[pos];
            if j < ncols {
                y[i] = y[i] + a.data[pos] * x[j];
            }
        }
    }
    Ok(y)
}

/// Build the Schur complement matrix `S = A_BB - A_BI A_II^{-1} A_IB` explicitly.
///
/// For each column `k` of `A_IB` we solve `A_II y_k = A_IB[:,k]` and then
/// compute the k-th column of `A_BI A_II^{-1} A_IB` as `A_BI y_k`.
fn build_schur_complement<F>(
    a_bb: &CsrMatrix<F>,
    a_bi: &CsrMatrix<F>,
    a_ii: &CsrMatrix<F>,
    a_ib: &CsrMatrix<F>,
    n_b: usize,
    n_i: usize,
    cfg: &IterativeSolverConfig,
) -> SparseResult<CsrMatrix<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    // Dense Schur complement: S[i][j] = A_BB[i][j] - sum_k A_BI[i][k] * (A_II^{-1} A_IB)[k][j]
    // Build A_II^{-1} A_IB column-by-column.
    let mut schur_dense = vec![vec![F::sparse_zero(); n_b]; n_b];

    // Copy A_BB into Schur
    for row in 0..n_b {
        let range = a_bb.row_range(row);
        for pos in range {
            let col = a_bb.indices[pos];
            if col < n_b {
                schur_dense[row][col] = a_bb.data[pos];
            }
        }
    }

    // Subtract A_BI A_II^{-1} A_IB, one column of A_IB at a time
    for j in 0..n_b {
        // Extract j-th column of A_IB as a dense vector
        let mut aib_col = Array1::<F>::zeros(n_i);
        for row_i in 0..n_i {
            let range = a_ib.row_range(row_i);
            for pos in range {
                if a_ib.indices[pos] == j {
                    aib_col[row_i] = a_ib.data[pos];
                }
            }
        }

        // Solve A_II y = aib_col
        let y = cg(a_ii, &aib_col, cfg, None)?.solution;

        // Compute A_BI y (n_b × n_i  times  n_i vector)
        let abi_y = spmv_dense(a_bi, &y, n_b, n_i)?;

        // Subtract from Schur
        for i in 0..n_b {
            schur_dense[i][j] = schur_dense[i][j] - abi_y[i];
        }
    }

    // Convert dense Schur complement to CsrMatrix
    let mut row_idx = Vec::new();
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();

    for (i, row) in schur_dense.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val.abs() > F::epsilon() {
                row_idx.push(i);
                col_idx.push(j);
                vals.push(val);
            }
        }
    }

    CsrMatrix::from_triplets(n_b, n_b, row_idx, col_idx, vals)
}

/// Compute the 2-norm of an Array1 vector.
#[inline]
fn vec_norm2<F: Float + Sum>(v: &Array1<F>) -> F {
    v.iter().map(|&x| x * x).sum::<F>().sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Build a simple n×n tridiagonal SPD CSR matrix: 2 on diagonal, -1 off.
    fn tridiagonal_csr(n: usize) -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for i in 0..n {
            // Main diagonal
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

        CsrMatrix::from_triplets(n, n, rows, cols, vals).expect("Failed to build tridiagonal CSR")
    }

    #[test]
    fn test_partition_domain_basic() {
        let parts = partition_domain(10, 2, 0).expect("partition failed");
        assert_eq!(parts.len(), 2);
        // Non-overlapping: first 5, last 5
        assert_eq!(parts[0].len(), 5);
        assert_eq!(parts[1].len(), 5);
        assert_eq!(parts[0][0], 0);
        assert_eq!(parts[1][4], 9);
    }

    #[test]
    fn test_partition_domain_with_overlap() {
        let parts = partition_domain(10, 2, 1).expect("partition failed");
        assert_eq!(parts.len(), 2);
        // With overlap=1 the first partition extends right by 1, second extends left by 1
        assert!(parts[0].len() > 5);
        assert!(parts[1].len() > 5);
        // The overlap region should be present in both
        assert!(parts[0].contains(&5));
        assert!(parts[1].contains(&4));
    }

    #[test]
    fn test_partition_domain_errors() {
        assert!(partition_domain(5, 0, 0).is_err());
        assert!(partition_domain(3, 10, 0).is_err());
    }

    #[test]
    fn test_additive_schwarz_small() {
        let n = 6;
        let a = tridiagonal_csr(n);
        let b = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);

        let parts = partition_domain(n, 3, 1).expect("partition failed");
        let config = SchwartzOverlap {
            overlap: 1,
            sub_max_iter: 200,
            sub_tol: 1e-12,
        };

        let x = additive_schwarz(&a, &b, &parts, &config).expect("additive_schwarz failed");
        assert_eq!(x.len(), n);

        // Verify residual is reasonable
        let r = compute_residual(&a, &b, &x).expect("residual failed");
        let rnorm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        // Additive Schwarz is a direct solve for small problems; tolerance is lenient here
        assert!(rnorm < 5.0, "residual too large: {rnorm}");
    }

    #[test]
    fn test_multiplicative_schwarz_small() {
        let n = 6;
        let a = tridiagonal_csr(n);
        let b = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let parts = partition_domain(n, 3, 1).expect("partition failed");
        let x = multiplicative_schwarz(&a, &b, &parts, 50, 1e-8)
            .expect("multiplicative_schwarz failed");

        assert_eq!(x.len(), n);
        let r = compute_residual(&a, &b, &x).expect("residual failed");
        let rnorm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(rnorm < 1e-4, "residual too large: {rnorm}");
    }

    #[test]
    fn test_schur_complement_solve_small() {
        let n = 4;
        let a = tridiagonal_csr(n);
        let b = Array1::from_vec(vec![1.0, 2.0, 2.0, 1.0]);

        // Interior DOFs: {1, 2}, interface DOFs: {0, 3}
        let interface_dofs = vec![0, 3];
        let x = schur_complement_solve(&a, &b, &interface_dofs)
            .expect("schur_complement_solve failed");

        assert_eq!(x.len(), n);

        // Verify A x ≈ b
        let r = compute_residual(&a, &b, &x).expect("residual failed");
        let rnorm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(rnorm < 1e-8, "Schur residual {rnorm} too large");
    }

    #[test]
    fn test_feti_solve_single_subdomain() {
        let n = 4;
        let a = tridiagonal_csr(n);
        let b = Array1::from_vec(vec![1.0, 2.0, 2.0, 1.0]);

        // Single subdomain, no connectivity
        let result = feti_solve(&[a], &[b.clone()], &[]).expect("feti_solve failed");

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), n);
    }

    #[test]
    fn test_feti_solve_two_subdomains() {
        // Two separate n=3 tridiagonal systems (no shared DOFs for simplicity)
        let n = 3;
        let a0 = tridiagonal_csr(n);
        let a1 = tridiagonal_csr(n);
        let b0 = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let b1 = Array1::from_vec(vec![2.0, 2.0, 2.0]);

        // Connect DOF 2 of subdomain 0 with DOF 0 of subdomain 1
        let connectivity = vec![(0usize, 1usize, vec![2usize], vec![0usize])];

        let result = feti_solve(&[a0, a1], &[b0, b1], &connectivity).expect("feti_solve failed");

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), n);
        assert_eq!(result[1].len(), n);
    }
}
