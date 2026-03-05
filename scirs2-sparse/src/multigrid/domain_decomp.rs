//! Domain Decomposition Methods for Sparse Linear Systems
//!
//! This module provides advanced domain decomposition preconditioners and solvers:
//!
//! 1. **Additive Schwarz Method (ASM)** – Overlapping subdomain decomposition with
//!    independent parallel local solves.
//!
//! 2. **Multiplicative Schwarz Method (MSM)** – Sequential subdomain updates that
//!    guarantee convergence for SPD systems.
//!
//! 3. **FETI (Finite Element Tearing and Interconnecting)** – Dual substructuring
//!    approach that formulates and solves an interface compatibility problem.
//!
//! 4. **Neumann-Neumann Balancing (NNB)** – Balanced domain decomposition with
//!    Neumann boundary conditions on subdomains for improved parallel scalability.
//!
//! # References
//!
//! - Toselli & Widlund (2005). *Domain Decomposition Methods — Algorithms and Theory*.
//! - Farhat & Roux (1991). "A method of finite element tearing and interconnecting."
//! - Mandel (1993). "Balancing domain decomposition." *Commun. Numer. Methods Eng.*
//! - Smith, Bjørstad & Gropp (1996). *Domain Decomposition*. Cambridge University Press.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use crate::iterative_solvers::{IterativeSolverConfig, SolverResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::fmt::Debug;
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

/// Sparse matrix-vector product: y = A * x  (f64 specialisation)
fn spmv_f64(a: &CsrMatrix<f64>, x: &[f64]) -> SparseResult<Vec<f64>> {
    a.dot(x)
}

/// Extract a principal submatrix A[dofs, dofs].
fn extract_submatrix(a: &CsrMatrix<f64>, dofs: &[usize]) -> CsrMatrix<f64> {
    let m = dofs.len();
    // Build reverse mapping: global dof → local index
    let n_global = a.shape().0;
    let mut local_idx = vec![usize::MAX; n_global];
    for (loc, &g) in dofs.iter().enumerate() {
        if g < n_global {
            local_idx[g] = loc;
        }
    }

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for (loc_i, &g_i) in dofs.iter().enumerate() {
        for pos in a.row_range(g_i) {
            let g_j = a.indices[pos];
            let loc_j = local_idx[g_j];
            if loc_j != usize::MAX {
                rows.push(loc_i);
                cols.push(loc_j);
                vals.push(a.data[pos]);
            }
        }
    }

    CsrMatrix::from_triplets(m, m, rows, cols, vals).unwrap_or_else(|_| {
        // Fallback: diagonal approximation
        let d: Vec<f64> = dofs.iter().map(|&g| a.get(g, g)).collect();
        let r: Vec<usize> = (0..m).collect();
        let c: Vec<usize> = (0..m).collect();
        CsrMatrix::from_triplets(m, m, r, c, d).unwrap_or_else(|_| {
            CsrMatrix::from_triplets(m, m, vec![], vec![], vec![]).expect("empty matrix")
        })
    })
}

/// Gaussian elimination with partial pivoting for a small dense system.
fn dense_solve(a: &[Vec<f64>], b: &[f64]) -> SparseResult<Vec<f64>> {
    let n = b.len();
    if a.len() != n {
        return Err(SparseError::DimensionMismatch { expected: n, found: a.len() });
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivot
        let (mut max_row, mut max_val) = (col, aug[col][col].abs());
        for row in (col + 1)..n {
            let v = aug[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return Err(SparseError::SingularMatrix(
                "singular local submatrix in domain decomposition".to_string(),
            ));
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for k in col..=n {
                let v = aug[col][k];
                aug[row][k] -= factor * v;
            }
        }
    }

    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = aug[i][n];
        for j in (i + 1)..n {
            s -= aug[i][j] * x[j];
        }
        x[i] = s / aug[i][i];
    }
    Ok(x)
}

/// Convert CSR to dense (for small local systems).
fn to_dense(a: &CsrMatrix<f64>) -> Vec<Vec<f64>> {
    let (rows, cols) = a.shape();
    let mut d = vec![vec![0.0f64; cols]; rows];
    for i in 0..rows {
        for pos in a.row_range(i) {
            let j = a.indices[pos];
            d[i][j] = a.data[pos];
        }
    }
    d
}

/// Solve A x = b for a CSR system using dense Gaussian elimination.
/// Only suitable for small local systems.
fn local_solve(a: &CsrMatrix<f64>, b: &[f64]) -> SparseResult<Vec<f64>> {
    let dense = to_dense(a);
    dense_solve(&dense, b)
}

// ---------------------------------------------------------------------------
// Subdomain partition
// ---------------------------------------------------------------------------

/// Partition `[0, n)` into `n_subdomains` overlapping subsets.
///
/// Each partition core has approximately `n / n_subdomains` DOFs,
/// extended by `overlap` additional DOFs on each boundary side.
pub fn partition_overlapping(
    n: usize,
    n_subdomains: usize,
    overlap: usize,
) -> SparseResult<Vec<Vec<usize>>> {
    if n_subdomains == 0 {
        return Err(SparseError::ValueError("n_subdomains must be >= 1".to_string()));
    }
    if n_subdomains > n {
        return Err(SparseError::ValueError(format!(
            "n_subdomains ({n_subdomains}) > n ({n})"
        )));
    }

    let base = n / n_subdomains;
    let rem = n % n_subdomains;
    let mut parts = Vec::with_capacity(n_subdomains);
    let mut start = 0usize;

    for k in 0..n_subdomains {
        let core = base + if k < rem { 1 } else { 0 };
        let end = start + core;
        let ext_start = start.saturating_sub(overlap);
        let ext_end = (end + overlap).min(n);
        parts.push((ext_start..ext_end).collect());
        start = end;
    }
    Ok(parts)
}

// ===========================================================================
// Part 1: Additive Schwarz Method (ASM)
// ===========================================================================

/// Configuration for Schwarz domain decomposition.
#[derive(Debug, Clone)]
pub struct SchwarzConfig {
    /// Number of subdomains.
    pub n_subdomains: usize,
    /// Overlap width (number of DOFs shared across subdomain boundaries).
    pub overlap: usize,
    /// Maximum outer CG/GMRES iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Whether to print convergence information.
    pub verbose: bool,
}

impl Default for SchwarzConfig {
    fn default() -> Self {
        Self {
            n_subdomains: 4,
            overlap: 1,
            max_iter: 100,
            tol: 1e-8,
            verbose: false,
        }
    }
}

/// Pre-built Additive Schwarz preconditioner.
///
/// Stores the local submatrices and subdomain DOF lists for fast application.
pub struct AdditiveSchwarzPreconditioner {
    /// DOF lists for each subdomain (including overlap).
    pub subdomains: Vec<Vec<usize>>,
    /// Dense local matrices A_i (for small subdomains).
    local_dense: Vec<Vec<Vec<f64>>>,
    /// Total problem size.
    pub n: usize,
}

impl AdditiveSchwarzPreconditioner {
    /// Build the ASM preconditioner from the global matrix and a partition.
    ///
    /// # Arguments
    ///
    /// * `a` – Global system matrix.
    /// * `config` – Schwarz configuration.
    pub fn new(a: &CsrMatrix<f64>, config: &SchwarzConfig) -> SparseResult<Self> {
        let n = a.shape().0;
        let subdomains = partition_overlapping(n, config.n_subdomains, config.overlap)?;

        let local_dense: Vec<Vec<Vec<f64>>> = subdomains
            .iter()
            .map(|dofs| {
                let local_a = extract_submatrix(a, dofs);
                to_dense(&local_a)
            })
            .collect();

        Ok(Self { subdomains, local_dense, n })
    }

    /// Apply the ASM preconditioner: z = M^{-1} r.
    ///
    /// Computes z = sum_i R_i^T A_i^{-1} R_i r, where R_i is the restriction
    /// operator for subdomain i.
    pub fn apply(&self, r: &[f64]) -> SparseResult<Vec<f64>> {
        if r.len() != self.n {
            return Err(SparseError::DimensionMismatch {
                expected: self.n,
                found: r.len(),
            });
        }
        let mut z = vec![0.0f64; self.n];

        for (k, dofs) in self.subdomains.iter().enumerate() {
            // Restrict r to subdomain k
            let r_local: Vec<f64> = dofs.iter().map(|&g| r[g]).collect();

            // Solve local system
            let z_local = dense_solve(&self.local_dense[k], &r_local)?;

            // Extend (add) local solution back to global
            for (loc, &g) in dofs.iter().enumerate() {
                z[g] += z_local[loc];
            }
        }
        Ok(z)
    }
}

/// Solve A x = b using the Additive Schwarz Method as a preconditioner for CG.
///
/// Uses ASM as a left-preconditioner inside a Preconditioned Conjugate Gradient
/// iteration, which is suitable for SPD systems.
///
/// # Arguments
///
/// * `a` – Global system matrix (SPD).
/// * `b` – Right-hand side vector.
/// * `config` – Schwarz configuration (subdomains, overlap, tolerance).
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csr::CsrMatrix;
/// use scirs2_sparse::multigrid::domain_decomp::{additive_schwarz_solve, SchwarzConfig};
///
/// let n = 8usize;
/// let mut rows = Vec::new(); let mut cols = Vec::new(); let mut vals = Vec::new();
/// for i in 0..n { rows.push(i); cols.push(i); vals.push(2.0f64); }
/// for i in 0..n-1 {
///     rows.push(i); cols.push(i+1); vals.push(-1.0f64);
///     rows.push(i+1); cols.push(i); vals.push(-1.0f64);
/// }
/// let a = CsrMatrix::from_triplets(&rows, &cols, &vals, (n, n)).expect("valid input");
/// let b: Vec<f64> = vec![1.0; n];
/// let config = SchwarzConfig { n_subdomains: 2, overlap: 1, max_iter: 100, tol: 1e-8, verbose: false };
/// let result = additive_schwarz_solve(&a, &b, &config).expect("valid input");
/// assert_eq!(result.solution.len(), n);
/// ```
pub fn additive_schwarz_solve(
    a: &CsrMatrix<f64>,
    b: &[f64],
    config: &SchwarzConfig,
) -> SparseResult<SolverResult<f64>> {
    let n = a.shape().0;
    if b.len() != n {
        return Err(SparseError::DimensionMismatch { expected: n, found: b.len() });
    }

    let prec = AdditiveSchwarzPreconditioner::new(a, config)?;
    let mut x = vec![0.0f64; n];

    // Preconditioned CG
    let ax0 = spmv_f64(a, &x)?;
    let mut r: Vec<f64> = b.iter().zip(ax0.iter()).map(|(bi, axi)| bi - axi).collect();
    let mut z = prec.apply(&r)?;
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(z.iter()).map(|(ri, zi)| ri * zi).sum();

    let b_norm: f64 = b.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
    let mut converged = false;
    let mut n_iter = 0usize;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;
        let ap = spmv_f64(a, &p)?;
        let pap: f64 = p.iter().zip(ap.iter()).map(|(pi, api)| pi * api).sum();

        if pap.abs() < f64::EPSILON {
            break;
        }
        let alpha = rz / pap;

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let res_norm: f64 = r.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
        if config.verbose {
            println!("ASM-CG iter {}: rel_res = {:.3e}", iter, res_norm / b_norm.max(1e-15));
        }

        if res_norm / b_norm.max(1e-15) < config.tol {
            converged = true;
            break;
        }

        z = prec.apply(&r)?;
        let rz_new: f64 = r.iter().zip(z.iter()).map(|(ri, zi)| ri * zi).sum();
        let beta = rz_new / rz.max(f64::EPSILON);
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }

    let ax_final = spmv_f64(a, &x)?;
    let res_final: f64 = b
        .iter()
        .zip(ax_final.iter())
        .map(|(bi, axi)| (bi - axi).powi(2))
        .sum::<f64>()
        .sqrt();

    Ok(SolverResult {
        solution: Array1::from_vec(x),
        n_iter,
        residual_norm: res_final,
        converged,
    })
}

// ===========================================================================
// Part 2: Multiplicative Schwarz Method (MSM)
// ===========================================================================

/// Solve A x = b using the Multiplicative (Sequential) Schwarz Method.
///
/// Performs `n_cycles` sweeps through all subdomains in sequence, updating
/// the global solution with each local solve. Converges for SPD matrices.
///
/// # Arguments
///
/// * `a` – Global system matrix (SPD).
/// * `b` – Right-hand side.
/// * `config` – Schwarz configuration.
/// * `n_cycles` – Number of full sweeps through all subdomains.
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csr::CsrMatrix;
/// use scirs2_sparse::multigrid::domain_decomp::{multiplicative_schwarz_solve, SchwarzConfig};
///
/// let n = 8usize;
/// let mut rows = Vec::new(); let mut cols = Vec::new(); let mut vals = Vec::new();
/// for i in 0..n { rows.push(i); cols.push(i); vals.push(2.0f64); }
/// for i in 0..n-1 {
///     rows.push(i); cols.push(i+1); vals.push(-1.0f64);
///     rows.push(i+1); cols.push(i); vals.push(-1.0f64);
/// }
/// let a = CsrMatrix::from_triplets(&rows, &cols, &vals, (n, n)).expect("valid input");
/// let b: Vec<f64> = vec![1.0; n];
/// let config = SchwarzConfig::default();
/// let result = multiplicative_schwarz_solve(&a, &b, &config, 30).expect("valid input");
/// assert_eq!(result.solution.len(), n);
/// ```
pub fn multiplicative_schwarz_solve(
    a: &CsrMatrix<f64>,
    b: &[f64],
    config: &SchwarzConfig,
    n_cycles: usize,
) -> SparseResult<SolverResult<f64>> {
    let n = a.shape().0;
    if b.len() != n {
        return Err(SparseError::DimensionMismatch { expected: n, found: b.len() });
    }

    let subdomains = partition_overlapping(n, config.n_subdomains, config.overlap)?;
    let local_dense: Vec<Vec<Vec<f64>>> = subdomains
        .iter()
        .map(|dofs| to_dense(&extract_submatrix(a, dofs)))
        .collect();

    let mut x = vec![0.0f64; n];
    let b_norm: f64 = b.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
    let mut converged = false;
    let mut n_iter = 0usize;

    'outer: for cycle in 0..n_cycles {
        n_iter = cycle + 1;
        // Sequential sweep through all subdomains
        for (k, dofs) in subdomains.iter().enumerate() {
            // Compute current residual restricted to subdomain k
            let r_local: Vec<f64> = dofs
                .iter()
                .map(|&g| {
                    let ax_g: f64 = a
                        .row_range(g)
                        .map(|pos| a.data[pos] * x[a.indices[pos]])
                        .sum();
                    b[g] - ax_g
                })
                .collect();

            // Solve local correction
            let dz_local = dense_solve(&local_dense[k], &r_local)?;

            // Update global solution
            for (loc, &g) in dofs.iter().enumerate() {
                x[g] += dz_local[loc];
            }
        }

        // Check global convergence
        let ax = spmv_f64(a, &x)?;
        let res: f64 = b
            .iter()
            .zip(ax.iter())
            .map(|(bi, axi)| (bi - axi).powi(2))
            .sum::<f64>()
            .sqrt();
        let rel_res = res / b_norm.max(1e-15);
        if config.verbose {
            println!("Multiplicative Schwarz cycle {}: rel_res = {:.3e}", cycle, rel_res);
        }
        if rel_res < config.tol {
            converged = true;
            break 'outer;
        }
    }

    let ax_final = spmv_f64(a, &x)?;
    let res_final: f64 = b
        .iter()
        .zip(ax_final.iter())
        .map(|(bi, axi)| (bi - axi).powi(2))
        .sum::<f64>()
        .sqrt();

    Ok(SolverResult {
        solution: Array1::from_vec(x),
        n_iter,
        residual_norm: res_final,
        converged,
    })
}

// ===========================================================================
// Part 3: FETI — Finite Element Tearing and Interconnecting
// ===========================================================================

/// Interface between two subdomains: lists of DOFs on each side.
#[derive(Debug, Clone)]
pub struct FetiInterface {
    /// DOFs on the left side of the interface (subdomain i).
    pub left_dofs: Vec<usize>,
    /// DOFs on the right side of the interface (subdomain j).
    pub right_dofs: Vec<usize>,
}

/// FETI configuration.
#[derive(Debug, Clone)]
pub struct FetiConfig {
    /// Number of subdomains (must be ≥ 2).
    pub n_subdomains: usize,
    /// Maximum iterations for the interface CG.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Verbose output.
    pub verbose: bool,
}

impl Default for FetiConfig {
    fn default() -> Self {
        Self {
            n_subdomains: 2,
            max_iter: 100,
            tol: 1e-8,
            verbose: false,
        }
    }
}

/// FETI solver state (pre-assembled).
pub struct FetiSolver {
    /// Subdomain DOF lists (non-overlapping partitions).
    pub subdomains: Vec<Vec<usize>>,
    /// Local stiffness matrices (dense).
    local_dense: Vec<Vec<Vec<f64>>>,
    /// Interface DOF pairs (one interface between each pair of adjacent subdomains).
    interfaces: Vec<FetiInterface>,
    /// Total problem size.
    pub n: usize,
    /// FETI configuration.
    pub config: FetiConfig,
}

impl FetiSolver {
    /// Build a FETI solver for the 1D partitioning case.
    ///
    /// Partitions `[0, n)` into `n_subdomains` non-overlapping blocks.
    /// Interfaces are identified by boundary DOFs between adjacent blocks.
    ///
    /// # Arguments
    ///
    /// * `a` – Global stiffness matrix (assembled from finite elements).
    /// * `config` – FETI configuration.
    pub fn new(a: &CsrMatrix<f64>, config: FetiConfig) -> SparseResult<Self> {
        let n = a.shape().0;
        if config.n_subdomains < 2 {
            return Err(SparseError::ValueError("FETI requires at least 2 subdomains".to_string()));
        }

        // Non-overlapping partition
        let subdomains = partition_overlapping(n, config.n_subdomains, 0)?;

        // Extract local stiffness matrices
        let local_dense: Vec<Vec<Vec<f64>>> = subdomains
            .iter()
            .map(|dofs| to_dense(&extract_submatrix(a, dofs)))
            .collect();

        // Identify interfaces between adjacent subdomains
        let mut interfaces = Vec::new();
        for k in 0..subdomains.len().saturating_sub(1) {
            let right_of_left = *subdomains[k].last().unwrap_or(&0);
            let left_of_right = *subdomains[k + 1].first().unwrap_or(&0);
            interfaces.push(FetiInterface {
                left_dofs: vec![right_of_left],
                right_dofs: vec![left_of_right],
            });
        }

        Ok(Self { subdomains, local_dense, interfaces, n, config })
    }

    /// Apply local subdomain solver: solve A_k * x_k = f_k
    fn local_solve_k(&self, k: usize, f_local: &[f64]) -> SparseResult<Vec<f64>> {
        dense_solve(&self.local_dense[k], f_local)
    }

    /// Apply the FETI interface operator F to Lagrange multipliers lambda.
    ///
    /// F = sum_k B_k A_k^{-1} B_k^T
    ///
    /// where B_k is the signed Boolean connectivity matrix for subdomain k.
    fn apply_interface_operator(&self, lambda: &[f64], a: &CsrMatrix<f64>) -> SparseResult<Vec<f64>> {
        let n_lambda = lambda.len();
        let mut f_lambda = vec![0.0f64; n_lambda];

        for (iface_idx, iface) in self.interfaces.iter().enumerate() {
            // Construct local forces from lambda: B_k^T * lambda on interface iface_idx
            let lambda_val = lambda[iface_idx];

            // Left subdomain: +lambda on left_dofs
            let k_left = iface_idx;
            if k_left < self.subdomains.len() {
                let dofs_left = &self.subdomains[k_left];
                let mut f_left = vec![0.0f64; dofs_left.len()];
                for &g in &iface.left_dofs {
                    if let Some(loc) = dofs_left.iter().position(|&d| d == g) {
                        f_left[loc] += lambda_val;
                    }
                }
                // Solve A_left u = f_left
                let u_left = self.local_solve_k(k_left, &f_left)?;
                // B_left * u_left contribution
                for &g in &iface.left_dofs {
                    if let Some(loc) = dofs_left.iter().position(|&d| d == g) {
                        f_lambda[iface_idx] += u_left[loc];
                    }
                }
            }

            // Right subdomain: -lambda on right_dofs
            let k_right = iface_idx + 1;
            if k_right < self.subdomains.len() {
                let dofs_right = &self.subdomains[k_right];
                let mut f_right = vec![0.0f64; dofs_right.len()];
                for &g in &iface.right_dofs {
                    if let Some(loc) = dofs_right.iter().position(|&d| d == g) {
                        f_right[loc] -= lambda_val;
                    }
                }
                let u_right = self.local_solve_k(k_right, &f_right)?;
                // -B_right * u_right contribution
                for &g in &iface.right_dofs {
                    if let Some(loc) = dofs_right.iter().position(|&d| d == g) {
                        f_lambda[iface_idx] -= u_right[loc];
                    }
                }
            }
        }

        Ok(f_lambda)
    }

    /// Compute the interface right-hand side: d = sum_k B_k A_k^{-1} f_k
    fn compute_interface_rhs(&self, b: &[f64]) -> SparseResult<Vec<f64>> {
        let n_lambda = self.interfaces.len();
        let mut d = vec![0.0f64; n_lambda];

        for (iface_idx, iface) in self.interfaces.iter().enumerate() {
            // Left subdomain
            let k_left = iface_idx;
            if k_left < self.subdomains.len() {
                let dofs_left = &self.subdomains[k_left];
                let f_left: Vec<f64> = dofs_left.iter().map(|&g| b[g]).collect();
                let u_left = self.local_solve_k(k_left, &f_left)?;
                for &g in &iface.left_dofs {
                    if let Some(loc) = dofs_left.iter().position(|&d| d == g) {
                        d[iface_idx] += u_left[loc];
                    }
                }
            }

            // Right subdomain (subtract)
            let k_right = iface_idx + 1;
            if k_right < self.subdomains.len() {
                let dofs_right = &self.subdomains[k_right];
                let f_right: Vec<f64> = dofs_right.iter().map(|&g| b[g]).collect();
                let u_right = self.local_solve_k(k_right, &f_right)?;
                for &g in &iface.right_dofs {
                    if let Some(loc) = dofs_right.iter().position(|&d| d == g) {
                        d[iface_idx] -= u_right[loc];
                    }
                }
            }
        }

        Ok(d)
    }

    /// Solve the FETI interface problem via Conjugate Gradient on lambda.
    ///
    /// Solves F λ = d, then recovers u_k = A_k^{-1}(f_k - B_k^T λ) for each k.
    pub fn solve(&self, a: &CsrMatrix<f64>, b: &[f64]) -> SparseResult<SolverResult<f64>> {
        let n = a.shape().0;
        if b.len() != n {
            return Err(SparseError::DimensionMismatch { expected: n, found: b.len() });
        }

        let n_lambda = self.interfaces.len();

        if n_lambda == 0 {
            // Single subdomain: direct solve
            let u = local_solve(a, b)?;
            let ax = spmv_f64(a, &u)?;
            let res: f64 = b.iter().zip(ax.iter()).map(|(bi, axi)| (bi - axi).powi(2)).sum::<f64>().sqrt();
            return Ok(SolverResult {
                solution: Array1::from_vec(u),
                n_iter: 1,
                residual_norm: res,
                converged: true,
            });
        }

        // Compute interface RHS d
        let d = self.compute_interface_rhs(b)?;

        // CG on interface problem F λ = d
        let mut lambda = vec![0.0f64; n_lambda];
        let mut r_iface = d.clone();
        let mut p_iface = r_iface.clone();
        let mut rr: f64 = r_iface.iter().map(|v| v.powi(2)).sum();
        let d_norm: f64 = d.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();

        let mut converged = false;
        let mut n_iter = 0usize;

        for iter in 0..self.config.max_iter {
            n_iter = iter + 1;
            let fp = self.apply_interface_operator(&p_iface, a)?;
            let pfp: f64 = p_iface.iter().zip(fp.iter()).map(|(pi, fpi)| pi * fpi).sum();

            if pfp.abs() < f64::EPSILON {
                break;
            }
            let alpha = rr / pfp;
            for i in 0..n_lambda {
                lambda[i] += alpha * p_iface[i];
                r_iface[i] -= alpha * fp[i];
            }
            let rr_new: f64 = r_iface.iter().map(|v| v.powi(2)).sum();
            let rel_res = rr_new.sqrt() / d_norm.max(1e-15);

            if self.config.verbose {
                println!("FETI CG iter {}: rel_res = {:.3e}", iter, rel_res);
            }
            if rel_res < self.config.tol {
                converged = true;
                break;
            }
            let beta = rr_new / rr.max(f64::EPSILON);
            for i in 0..n_lambda {
                p_iface[i] = r_iface[i] + beta * p_iface[i];
            }
            rr = rr_new;
        }

        // Recover primal solution: u_k = A_k^{-1}(f_k - B_k^T lambda)
        let mut x = vec![0.0f64; n];
        for (k, dofs) in self.subdomains.iter().enumerate() {
            let mut f_local: Vec<f64> = dofs.iter().map(|&g| b[g]).collect();
            // Subtract B_k^T lambda contributions
            for (iface_idx, iface) in self.interfaces.iter().enumerate() {
                let lv = lambda[iface_idx];
                if k == iface_idx {
                    for &g in &iface.left_dofs {
                        if let Some(loc) = dofs.iter().position(|&d| d == g) {
                            f_local[loc] -= lv;
                        }
                    }
                } else if k == iface_idx + 1 {
                    for &g in &iface.right_dofs {
                        if let Some(loc) = dofs.iter().position(|&d| d == g) {
                            f_local[loc] += lv;
                        }
                    }
                }
            }
            let u_local = self.local_solve_k(k, &f_local)?;
            for (loc, &g) in dofs.iter().enumerate() {
                x[g] = u_local[loc];
            }
        }

        let ax_final = spmv_f64(a, &x)?;
        let res_final: f64 = b
            .iter()
            .zip(ax_final.iter())
            .map(|(bi, axi)| (bi - axi).powi(2))
            .sum::<f64>()
            .sqrt();

        Ok(SolverResult {
            solution: Array1::from_vec(x),
            n_iter,
            residual_norm: res_final,
            converged,
        })
    }
}

/// Build and solve using the FETI method.
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csr::CsrMatrix;
/// use scirs2_sparse::multigrid::domain_decomp::{feti_solve, FetiConfig};
///
/// let n = 8usize;
/// let mut rows = Vec::new(); let mut cols = Vec::new(); let mut vals = Vec::new();
/// for i in 0..n { rows.push(i); cols.push(i); vals.push(2.0f64); }
/// for i in 0..n-1 {
///     rows.push(i); cols.push(i+1); vals.push(-1.0f64);
///     rows.push(i+1); cols.push(i); vals.push(-1.0f64);
/// }
/// let a = CsrMatrix::from_triplets(&rows, &cols, &vals, (n, n)).expect("valid input");
/// let b: Vec<f64> = vec![1.0; n];
/// let config = FetiConfig { n_subdomains: 2, max_iter: 50, tol: 1e-8, verbose: false };
/// let result = feti_solve(&a, &b, config).expect("valid input");
/// assert_eq!(result.solution.len(), n);
/// ```
pub fn feti_solve(
    a: &CsrMatrix<f64>,
    b: &[f64],
    config: FetiConfig,
) -> SparseResult<SolverResult<f64>> {
    let solver = FetiSolver::new(a, config)?;
    solver.solve(a, b)
}

// ===========================================================================
// Part 4: Neumann-Neumann Balancing (BNN)
// ===========================================================================

/// Configuration for Neumann-Neumann Balancing.
#[derive(Debug, Clone)]
pub struct NeumannNeumannConfig {
    /// Number of subdomains.
    pub n_subdomains: usize,
    /// Maximum CG iterations on the interface problem.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Scaling: `true` = use multiplicity scaling, `false` = unity.
    pub use_multiplicity_scaling: bool,
    /// Verbose output.
    pub verbose: bool,
}

impl Default for NeumannNeumannConfig {
    fn default() -> Self {
        Self {
            n_subdomains: 4,
            max_iter: 100,
            tol: 1e-8,
            use_multiplicity_scaling: true,
            verbose: false,
        }
    }
}

/// Neumann-Neumann Balancing preconditioner / solver.
///
/// The Balanced Neumann-Neumann method applies a two-level preconditioner:
///
/// M_BNN^{-1} = P_I + (I - P_I) M_NN^{-1} (I - P_I)^T
///
/// where M_NN^{-1} is the Neumann-Neumann preconditioner and P_I is the
/// coarse-grid (interface) projector.
///
/// In this implementation we use the simplified one-level Neumann-Neumann
/// preconditioner (the balancing projection requires a coarse solve which
/// for the general case is complex; it is incorporated as an additional
/// deflation step after each CG iteration).
pub struct NeumannNeumannSolver {
    /// Subdomain DOF lists (non-overlapping).
    subdomains: Vec<Vec<usize>>,
    /// Local Dirichlet matrices (for primal solve): dense.
    local_dirichlet_dense: Vec<Vec<Vec<f64>>>,
    /// Local Neumann matrices (for dual preconditioner): dense.
    /// For Neumann BCs, the diagonal is modified to be positive-definite.
    local_neumann_dense: Vec<Vec<Vec<f64>>>,
    /// Multiplicity of each DOF (number of subdomains it belongs to).
    multiplicity: Vec<f64>,
    /// Total problem size.
    n: usize,
    /// Configuration.
    config: NeumannNeumannConfig,
    /// Interface DOFs (shared between subdomains).
    interface_dofs: Vec<usize>,
}

impl NeumannNeumannSolver {
    /// Build the Neumann-Neumann solver.
    pub fn new(a: &CsrMatrix<f64>, config: NeumannNeumannConfig) -> SparseResult<Self> {
        let n = a.shape().0;
        let subdomains = partition_overlapping(n, config.n_subdomains, 1)?;

        // Count multiplicity of each DOF
        let mut mult = vec![0.0f64; n];
        for dofs in &subdomains {
            for &g in dofs {
                mult[g] += 1.0;
            }
        }

        // Identify interface DOFs (multiplicity > 1)
        let interface_dofs: Vec<usize> = (0..n).filter(|&i| mult[i] > 1.0).collect();

        // Local Dirichlet matrices: exact submatrix extraction
        let local_dirichlet_dense: Vec<Vec<Vec<f64>>> = subdomains
            .iter()
            .map(|dofs| to_dense(&extract_submatrix(a, dofs)))
            .collect();

        // Local Neumann matrices: same as Dirichlet but with regularisation on
        // interface DOFs to handle the null space (floating subdomain problem)
        let local_neumann_dense: Vec<Vec<Vec<f64>>> = subdomains
            .iter()
            .map(|dofs| {
                let mut d = to_dense(&extract_submatrix(a, dofs));
                let m = dofs.len();
                // Add a small stabilization on interface rows
                // (Neumann BC: zero-flux, regularise by adding epsilon * I)
                for i in 0..m {
                    let g = dofs[i];
                    if mult[g] > 1.0 {
                        d[i][i] += 1e-10;
                    }
                }
                d
            })
            .collect();

        Ok(Self {
            subdomains,
            local_dirichlet_dense,
            local_neumann_dense,
            multiplicity: mult,
            n,
            config,
            interface_dofs,
        })
    }

    /// Apply the Neumann-Neumann preconditioner: M_NN^{-1} r.
    ///
    /// M_NN^{-1} = sum_k D_k^{-1} R_k^T A_k^{†} R_k D_k^{-1}
    ///
    /// where D_k is the partition-of-unity scaling (multiplicity-based)
    /// and A_k^{†} is the local Neumann solve (possibly pseudo-inverse).
    fn apply_nn_preconditioner(&self, r: &[f64]) -> SparseResult<Vec<f64>> {
        let mut z = vec![0.0f64; self.n];

        for (k, dofs) in self.subdomains.iter().enumerate() {
            // Scale by D^{-1} (partition of unity)
            let r_scaled: Vec<f64> = if self.config.use_multiplicity_scaling {
                dofs.iter().map(|&g| r[g] / self.multiplicity[g]).collect()
            } else {
                dofs.iter().map(|&g| r[g]).collect()
            };

            // Neumann solve
            let z_local = match dense_solve(&self.local_neumann_dense[k], &r_scaled) {
                Ok(v) => v,
                Err(_) => {
                    // Fallback: Jacobi approximation
                    r_scaled
                        .iter()
                        .enumerate()
                        .map(|(i, &ri)| {
                            let d = self.local_neumann_dense[k][i][i];
                            if d.abs() > 1e-15 { ri / d } else { 0.0 }
                        })
                        .collect()
                }
            };

            // Scale by D^{-1} again (symmetric) and accumulate
            for (loc, &g) in dofs.iter().enumerate() {
                let scale = if self.config.use_multiplicity_scaling {
                    1.0 / self.multiplicity[g]
                } else {
                    1.0
                };
                z[g] += scale * z_local[loc];
            }
        }

        Ok(z)
    }

    /// Coarse-grid correction: project out the component in the kernel of the
    /// Neumann problem (rigid-body modes for structural problems; constant for Laplacian).
    fn coarse_correction(&self, r: &[f64]) -> Vec<f64> {
        // For each subdomain, compute the mean of r on that subdomain
        // and subtract the subdomain-mean to enforce compatibility
        let mut r_proj = r.to_vec();
        for dofs in &self.subdomains {
            if dofs.is_empty() {
                continue;
            }
            let mean: f64 = dofs.iter().map(|&g| r_proj[g]).sum::<f64>() / dofs.len() as f64;
            for &g in dofs {
                r_proj[g] -= mean;
            }
        }
        r_proj
    }

    /// Full BNN solve: PCG with Neumann-Neumann preconditioner.
    ///
    /// The balancing correction is applied after each preconditioner step
    /// to handle floating subdomains and improve parallel scalability.
    pub fn solve(&self, a: &CsrMatrix<f64>, b: &[f64]) -> SparseResult<SolverResult<f64>> {
        let n = a.shape().0;
        if b.len() != n {
            return Err(SparseError::DimensionMismatch { expected: n, found: b.len() });
        }

        let mut x = vec![0.0f64; n];
        let ax0 = spmv_f64(a, &x)?;
        let mut r: Vec<f64> = b.iter().zip(ax0.iter()).map(|(bi, axi)| bi - axi).collect();

        // Apply balancing projection to initial residual
        r = self.coarse_correction(&r);

        let mut z = self.apply_nn_preconditioner(&r)?;
        let mut p = z.clone();
        let mut rz: f64 = r.iter().zip(z.iter()).map(|(ri, zi)| ri * zi).sum();

        let b_norm: f64 = b.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
        let mut converged = false;
        let mut n_iter = 0usize;

        for iter in 0..self.config.max_iter {
            n_iter = iter + 1;
            let ap = spmv_f64(a, &p)?;
            let pap: f64 = p.iter().zip(ap.iter()).map(|(pi, api)| pi * api).sum();

            if pap.abs() < f64::EPSILON {
                break;
            }
            let alpha = rz / pap;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }

            // Balancing coarse correction on residual
            r = self.coarse_correction(&r);

            let res_norm: f64 = r.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
            let rel_res = res_norm / b_norm.max(1e-15);

            if self.config.verbose {
                println!("BNN iter {}: rel_res = {:.3e}", iter, rel_res);
            }
            if rel_res < self.config.tol {
                converged = true;
                break;
            }

            z = self.apply_nn_preconditioner(&r)?;
            let rz_new: f64 = r.iter().zip(z.iter()).map(|(ri, zi)| ri * zi).sum();
            let beta = if rz.abs() > f64::EPSILON { rz_new / rz } else { 0.0 };
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz = rz_new;
        }

        let ax_final = spmv_f64(a, &x)?;
        let res_final: f64 = b
            .iter()
            .zip(ax_final.iter())
            .map(|(bi, axi)| (bi - axi).powi(2))
            .sum::<f64>()
            .sqrt();

        Ok(SolverResult {
            solution: Array1::from_vec(x),
            n_iter,
            residual_norm: res_final,
            converged,
        })
    }
}

/// Solve A x = b using the Neumann-Neumann Balancing method.
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csr::CsrMatrix;
/// use scirs2_sparse::multigrid::domain_decomp::{neumann_neumann_solve, NeumannNeumannConfig};
///
/// let n = 8usize;
/// let mut rows = Vec::new(); let mut cols = Vec::new(); let mut vals = Vec::new();
/// for i in 0..n { rows.push(i); cols.push(i); vals.push(2.0f64); }
/// for i in 0..n-1 {
///     rows.push(i); cols.push(i+1); vals.push(-1.0f64);
///     rows.push(i+1); cols.push(i); vals.push(-1.0f64);
/// }
/// let a = CsrMatrix::from_triplets(&rows, &cols, &vals, (n, n)).expect("valid input");
/// let b: Vec<f64> = vec![1.0; n];
/// let config = NeumannNeumannConfig { n_subdomains: 2, max_iter: 100, tol: 1e-8,
///                                     use_multiplicity_scaling: true, verbose: false };
/// let result = neumann_neumann_solve(&a, &b, config).expect("valid input");
/// assert_eq!(result.solution.len(), n);
/// ```
pub fn neumann_neumann_solve(
    a: &CsrMatrix<f64>,
    b: &[f64],
    config: NeumannNeumannConfig,
) -> SparseResult<SolverResult<f64>> {
    let solver = NeumannNeumannSolver::new(a, config)?;
    solver.solve(a, b)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i); cols.push(i); vals.push(2.0f64);
        }
        for i in 0..n - 1 {
            rows.push(i); cols.push(i + 1); vals.push(-1.0f64);
            rows.push(i + 1); cols.push(i); vals.push(-1.0f64);
        }
        CsrMatrix::from_triplets(n, n, rows, cols, vals).expect("laplacian_1d")
    }

    #[test]
    fn test_partition_overlapping() {
        let parts = partition_overlapping(10, 3, 1).expect("partition");
        assert_eq!(parts.len(), 3);
        assert!(parts[0].contains(&0));
        assert!(parts[2].contains(&9));
        // Verify overlap
        let s0: std::collections::HashSet<_> = parts[0].iter().collect();
        let s1: std::collections::HashSet<_> = parts[1].iter().collect();
        let shared: Vec<_> = s0.intersection(&s1).collect();
        assert!(!shared.is_empty(), "adjacent subdomains must share DOFs with overlap=1");
    }

    #[test]
    fn test_additive_schwarz_solve() {
        let n = 8;
        let a = laplacian_1d(n);
        let b: Vec<f64> = vec![1.0f64; n];
        let config = SchwarzConfig {
            n_subdomains: 2,
            overlap: 1,
            max_iter: 100,
            tol: 1e-8,
            verbose: false,
        };
        let result = additive_schwarz_solve(&a, &b, &config).expect("ASM solve");
        assert_eq!(result.solution.len(), n);

        // Verify residual
        let x: Vec<f64> = result.solution.to_vec();
        let ax = a.dot(&x).expect("spmv");
        let res: f64 = b.iter().zip(ax.iter()).map(|(bi, axi)| (bi - axi).powi(2)).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
        assert!(res / b_norm < 1e-5, "ASM residual too large: {}", res / b_norm);
    }

    #[test]
    fn test_multiplicative_schwarz_solve() {
        let n = 8;
        let a = laplacian_1d(n);
        let b: Vec<f64> = vec![1.0f64; n];
        let config = SchwarzConfig {
            n_subdomains: 2,
            overlap: 1,
            max_iter: 100,
            tol: 1e-8,
            verbose: false,
        };
        let result = multiplicative_schwarz_solve(&a, &b, &config, 30).expect("MSM solve");
        assert_eq!(result.solution.len(), n);
    }

    #[test]
    fn test_feti_solve() {
        let n = 8;
        let a = laplacian_1d(n);
        let b: Vec<f64> = vec![1.0f64; n];
        let config = FetiConfig {
            n_subdomains: 2,
            max_iter: 50,
            tol: 1e-8,
            verbose: false,
        };
        let result = feti_solve(&a, &b, config).expect("FETI solve");
        assert_eq!(result.solution.len(), n);
    }

    #[test]
    fn test_neumann_neumann_solve() {
        let n = 8;
        let a = laplacian_1d(n);
        let b: Vec<f64> = vec![1.0f64; n];
        let config = NeumannNeumannConfig {
            n_subdomains: 2,
            max_iter: 100,
            tol: 1e-8,
            use_multiplicity_scaling: true,
            verbose: false,
        };
        let result = neumann_neumann_solve(&a, &b, config).expect("BNN solve");
        assert_eq!(result.solution.len(), n);
    }

    #[test]
    fn test_extract_submatrix() {
        let a = laplacian_1d(5);
        let dofs = vec![1, 2, 3];
        let sub = extract_submatrix(&a, &dofs);
        assert_eq!(sub.shape(), (3, 3));
        // Diagonal should be 2.0
        assert!((sub.get(0, 0) - 2.0).abs() < 1e-12);
        assert!((sub.get(1, 1) - 2.0).abs() < 1e-12);
        // Off-diagonal should be -1.0
        assert!((sub.get(0, 1) - (-1.0)).abs() < 1e-12);
        assert!((sub.get(1, 0) - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_dense_solve() {
        // 3×3 test
        let a = vec![
            vec![4.0, -1.0, 0.0],
            vec![-1.0, 4.0, -1.0],
            vec![0.0, -1.0, 4.0],
        ];
        let b = vec![1.0, 0.0, 1.0];
        let x = dense_solve(&a, &b).expect("dense solve");
        for (i, row) in a.iter().enumerate() {
            let sum: f64 = row.iter().zip(x.iter()).map(|(ai, xi)| ai * xi).sum();
            assert!((sum - b[i]).abs() < 1e-10, "residual at row {}: {}", i, (sum - b[i]).abs());
        }
    }

    #[test]
    fn test_asm_preconditioner_apply() {
        let n = 6;
        let a = laplacian_1d(n);
        let config = SchwarzConfig { n_subdomains: 2, overlap: 1, max_iter: 50, tol: 1e-8, verbose: false };
        let prec = AdditiveSchwarzPreconditioner::new(&a, &config).expect("ASM preconditioner");
        let r = vec![1.0f64; n];
        let z = prec.apply(&r).expect("apply ASM");
        assert_eq!(z.len(), n);
        // z should not be identically zero
        assert!(z.iter().any(|&v| v.abs() > 1e-15), "preconditioner must produce non-trivial output");
    }

    #[test]
    fn test_nn_solver_build() {
        let n = 8;
        let a = laplacian_1d(n);
        let config = NeumannNeumannConfig::default();
        let solver = NeumannNeumannSolver::new(&a, config).expect("NN solver build");
        assert_eq!(solver.n, n);
        assert!(!solver.interface_dofs.is_empty(), "must identify interface DOFs");
    }
}
