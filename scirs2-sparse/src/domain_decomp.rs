//! Domain decomposition methods for sparse linear systems
//!
//! This module provides high-level domain decomposition solvers and
//! preconditioners that complement the existing `domain_decomposition` module.
//!
//! ## Available Methods
//!
//! | Type | Description |
//! |------|-------------|
//! | [`SchwarzSolver`] | Additive overlapping Schwarz method |
//! | [`SchurComplementSolver`] | Exact block 2×2 Schur complement elimination |
//!
//! ## References
//!
//! - Toselli & Widlund (2005). *Domain Decomposition Methods — Algorithms and Theory*.
//! - Saad (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed.

use crate::error::{SparseError, SparseResult};

// ---------------------------------------------------------------------------
// Internal dense LU (partial pivoting)
// ---------------------------------------------------------------------------

fn dense_lu_factor(a: &mut Vec<Vec<f64>>, n: usize) -> Option<Vec<usize>> {
    let mut perm: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let mut max_val = a[k][k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            if a[i][k].abs() > max_val {
                max_val = a[i][k].abs();
                max_row = i;
            }
        }
        if max_val < 1e-300 {
            return None;
        }
        a.swap(k, max_row);
        perm.swap(k, max_row);
        let piv = a[k][k];
        for i in (k + 1)..n {
            a[i][k] /= piv;
            for j in (k + 1)..n {
                let l = a[i][k];
                a[i][j] -= l * a[k][j];
            }
        }
    }
    Some(perm)
}

fn dense_lu_solve(lu: &[Vec<f64>], perm: &[usize], b: &[f64], n: usize) -> Vec<f64> {
    let mut y: Vec<f64> = perm.iter().map(|&p| b[p]).collect();
    for i in 0..n {
        for j in 0..i {
            y[i] -= lu[i][j] * y[j];
        }
    }
    let mut x = y;
    for ii in 0..n {
        let i = n - 1 - ii;
        for j in (i + 1)..n {
            x[i] -= lu[i][j] * x[j];
        }
        x[i] /= lu[i][i];
    }
    x
}

// ---------------------------------------------------------------------------
// Sparse CSR helpers
// ---------------------------------------------------------------------------

fn csr_matvec(
    row_ptr: &[usize],
    col_ind: &[usize],
    val: &[f64],
    x: &[f64],
    n: usize,
) -> Vec<f64> {
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0f64;
        for pos in row_ptr[i]..row_ptr[i + 1] {
            if col_ind[pos] < x.len() {
                acc += val[pos] * x[col_ind[pos]];
            }
        }
        y[i] = acc;
    }
    y
}

/// Extract the sub-matrix A[rows, rows] from a CSR matrix (global → local indices).
/// Returns (local_val, local_row_ptr, local_col_ind, local_n).
fn extract_submatrix_square(
    global_row_ptr: &[usize],
    global_col_ind: &[usize],
    global_val: &[f64],
    rows: &[usize],
) -> (Vec<f64>, Vec<usize>, Vec<usize>, usize) {
    let m = rows.len();
    // Build inverse mapping global → local
    let global_n = global_row_ptr.len().saturating_sub(1);
    let mut g2l = vec![usize::MAX; global_n];
    for (local, &global) in rows.iter().enumerate() {
        g2l[global] = local;
    }

    let mut val = Vec::new();
    let mut col_ind = Vec::new();
    let mut row_ptr = vec![0usize; m + 1];

    for (local_row, &global_row) in rows.iter().enumerate() {
        for pos in global_row_ptr[global_row]..global_row_ptr[global_row + 1] {
            let gc = global_col_ind[pos];
            if gc < global_n && g2l[gc] != usize::MAX {
                col_ind.push(g2l[gc]);
                val.push(global_val[pos]);
            }
        }
        row_ptr[local_row + 1] = val.len();
    }
    (val, row_ptr, col_ind, m)
}

/// Extract the rectangular sub-matrix A[rows, cols] in dense form (nrows × ncols).
#[allow(dead_code)]
fn extract_submatrix_dense(
    global_row_ptr: &[usize],
    global_col_ind: &[usize],
    global_val: &[f64],
    rows: &[usize],
    cols: &[usize],
) -> Vec<Vec<f64>> {
    let nr = rows.len();
    let nc = cols.len();
    let mut dense = vec![vec![0.0f64; nc]; nr];
    // Map cols to local column indices
    let max_col = global_col_ind.iter().copied().max().unwrap_or(0) + 1;
    let mut c2l = vec![usize::MAX; max_col.max(1)];
    for (lc, &gc) in cols.iter().enumerate() {
        if gc < c2l.len() {
            c2l[gc] = lc;
        }
    }
    for (lr, &gr) in rows.iter().enumerate() {
        for pos in global_row_ptr[gr]..global_row_ptr[gr + 1] {
            let gc = global_col_ind[pos];
            if gc < c2l.len() && c2l[gc] != usize::MAX {
                dense[lr][c2l[gc]] = global_val[pos];
            }
        }
    }
    dense
}

// ---------------------------------------------------------------------------
// Conjugate Gradient (internal, used by SchwarzSolver::solve)
// ---------------------------------------------------------------------------

fn cg_internal<F>(
    matvec: F,
    b: &[f64],
    n: usize,
    max_iter: usize,
    tol: f64,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut x = vec![0.0f64; n];
    let mut r: Vec<f64> = b.to_vec(); // r = b - A*0 = b
    let mut p = r.clone();
    let mut rr: f64 = r.iter().map(|v| v * v).sum();
    let tol_sq = tol * tol * b.iter().map(|v| v * v).sum::<f64>();

    for _ in 0..max_iter {
        if rr <= tol_sq {
            break;
        }
        let ap = matvec(&p);
        let pap: f64 = p.iter().zip(ap.iter()).map(|(a, b)| a * b).sum();
        if pap.abs() < 1e-300 {
            break;
        }
        let alpha = rr / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rr_new: f64 = r.iter().map(|v| v * v).sum();
        let beta = rr_new / rr;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rr = rr_new;
    }
    x
}

// ---------------------------------------------------------------------------
// Sub-domain solver (used internally by SchwarzSolver)
// ---------------------------------------------------------------------------

struct SubdomainSolver {
    /// Global indices belonging to this subdomain.
    global_indices: Vec<usize>,
    /// Dense LU factors of the local problem.
    lu_factor: Vec<Vec<f64>>,
    /// LU permutation.
    lu_perm: Vec<usize>,
    /// Local dimension.
    size: usize,
}

impl SubdomainSolver {
    /// Build a SubdomainSolver from the global CSR matrix and a list of DOFs.
    fn new(
        global_row_ptr: &[usize],
        global_col_ind: &[usize],
        global_val: &[f64],
        global_indices: Vec<usize>,
    ) -> SparseResult<Self> {
        let m = global_indices.len();
        if m == 0 {
            return Err(SparseError::InvalidArgument(
                "subdomain must have at least one DOF".to_string(),
            ));
        }
        let (lval, lrp, lci, ln) = extract_submatrix_square(
            global_row_ptr, global_col_ind, global_val, &global_indices,
        );

        // Build dense copy for LU
        let mut dense = vec![vec![0.0f64; ln]; ln];
        for i in 0..ln {
            for pos in lrp[i]..lrp[i + 1] {
                dense[i][lci[pos]] = lval[pos];
            }
        }

        let perm = dense_lu_factor(&mut dense, ln).ok_or_else(|| {
            SparseError::SingularMatrix("subdomain local matrix is singular".to_string())
        })?;

        Ok(Self {
            global_indices,
            lu_factor: dense,
            lu_perm: perm,
            size: ln,
        })
    }

    /// Solve the local system: given global rhs `b` and current global iterate
    /// `x`, compute the local correction and scatter it back into the global
    /// solution (additive Schwarz update).
    fn apply_additive(
        &self,
        x_global: &[f64],
        b_global: &[f64],
        global_row_ptr: &[usize],
        global_col_ind: &[usize],
        global_val: &[f64],
        global_n: usize,
        result: &mut Vec<f64>,
    ) {
        let m = self.size;
        let idx = &self.global_indices;

        // Local residual: r_loc = b_loc - A_loc_full * x_global
        // A_loc_full includes contributions from ALL columns (not just local ones)
        let mut r_loc = vec![0.0f64; m];
        for (li, &gi) in idx.iter().enumerate() {
            let mut acc = b_global[gi];
            if gi < global_n {
                for pos in global_row_ptr[gi]..global_row_ptr[gi + 1] {
                    let gc = global_col_ind[pos];
                    if gc < x_global.len() {
                        acc -= global_val[pos] * x_global[gc];
                    }
                }
            }
            r_loc[li] = acc;
        }

        // Solve local system: A_loc * delta = r_loc
        let delta = dense_lu_solve(&self.lu_factor, &self.lu_perm, &r_loc, m);

        // Scatter: result[gi] += delta[li]
        for (li, &gi) in idx.iter().enumerate() {
            result[gi] += delta[li];
        }
    }
}

// ---------------------------------------------------------------------------
// Schwarz Solver
// ---------------------------------------------------------------------------

/// Additive overlapping Schwarz method.
///
/// Partitions `[0, n)` into `n_subdomains` overlapping subsets and uses the
/// additive Schwarz preconditioner inside a Conjugate Gradient iteration.
///
/// # Construction
///
/// ```rust
/// use scirs2_sparse::domain_decomp::SchwarzSolver;
///
/// // 5×5 tridiagonal system: 2 on diagonal, -1 off-diagonal
/// let n = 5usize;
/// let row_ptr: Vec<usize> = vec![0, 2, 5, 8, 11, 13];
/// let col_ind = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
/// let val = vec![2.0f64, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
/// let solver = SchwarzSolver::new(&val, &row_ptr, &col_ind, n, 2, 1)
///     .expect("build");
/// let b = vec![1.0f64; n];
/// let x = solver.solve(&b, 100, 1e-10);
/// assert_eq!(x.len(), n);
/// ```
pub struct SchwarzSolver {
    subdomains: Vec<SubdomainSolver>,
    n: usize,
    row_ptr: Vec<usize>,
    col_ind: Vec<usize>,
    val: Vec<f64>,
}

impl SchwarzSolver {
    /// Build a Schwarz solver for the CSR matrix A.
    ///
    /// # Arguments
    ///
    /// * `csr_val`, `csr_row_ptr`, `csr_col_ind` – Matrix in CSR format.
    /// * `n`             – Dimension of A (must be square).
    /// * `n_subdomains`  – Number of subdomains to create.
    /// * `overlap`       – Number of DOFs to add at subdomain boundaries.
    pub fn new(
        csr_val: &[f64],
        csr_row_ptr: &[usize],
        csr_col_ind: &[usize],
        n: usize,
        n_subdomains: usize,
        overlap: usize,
    ) -> SparseResult<Self> {
        if n_subdomains == 0 || n_subdomains > n {
            return Err(SparseError::InvalidArgument(format!(
                "n_subdomains={n_subdomains} must be in [1, {n}]"
            )));
        }

        // Build overlapping partitions of [0, n)
        let base = n / n_subdomains;
        let remainder = n % n_subdomains;
        let mut partitions: Vec<Vec<usize>> = Vec::with_capacity(n_subdomains);
        let mut start = 0;
        for s in 0..n_subdomains {
            let extra = if s < remainder { 1 } else { 0 };
            let end = (start + base + extra).min(n);
            // Add overlap on the left and right
            let lo = start.saturating_sub(overlap);
            let hi = (end + overlap).min(n);
            partitions.push((lo..hi).collect());
            start = end;
        }

        // Build subdomain solvers
        let mut subdomains = Vec::with_capacity(n_subdomains);
        for indices in partitions {
            let sd = SubdomainSolver::new(csr_row_ptr, csr_col_ind, csr_val, indices)?;
            subdomains.push(sd);
        }

        Ok(Self {
            subdomains,
            n,
            row_ptr: csr_row_ptr.to_vec(),
            col_ind: csr_col_ind.to_vec(),
            val: csr_val.to_vec(),
        })
    }

    /// Apply one additive Schwarz step: compute the sum of local corrections.
    ///
    /// Given the current iterate `x` and right-hand side `b`, returns a
    /// new iterate incorporating local subdomain corrections.
    pub fn apply(&self, x: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = x.to_vec();
        let n = self.n;
        let mut delta = vec![0.0f64; n];

        for sd in &self.subdomains {
            sd.apply_additive(
                x,
                b,
                &self.row_ptr,
                &self.col_ind,
                &self.val,
                n,
                &mut delta,
            );
        }

        for i in 0..n {
            result[i] += delta[i];
        }
        result
    }

    /// Solve A x = b using a preconditioned Conjugate Gradient method where
    /// the additive Schwarz operator is the preconditioner.
    pub fn solve(&self, b: &[f64], max_iter: usize, tol: f64) -> Vec<f64> {
        let n = self.n;
        let rp = &self.row_ptr;
        let ci = &self.col_ind;
        let vl = &self.val;

        // PCG with additive Schwarz preconditioner
        let mut x = vec![0.0f64; n];
        let mut r: Vec<f64> = b.to_vec(); // r = b - A*0 = b
        let mut z = self.precondition(&r); // M^{-1} r
        let mut p = z.clone();
        let mut rz: f64 = r.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
        let b_norm_sq: f64 = b.iter().map(|v| v * v).sum();
        let tol_sq = tol * tol * b_norm_sq;

        for _ in 0..max_iter {
            if r.iter().map(|v| v * v).sum::<f64>() <= tol_sq {
                break;
            }
            let ap = csr_matvec(rp, ci, vl, &p, n);
            let pap: f64 = p.iter().zip(ap.iter()).map(|(a, b)| a * b).sum();
            if pap.abs() < 1e-300 {
                break;
            }
            let alpha = rz / pap;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            z = self.precondition(&r);
            let rz_new: f64 = r.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
            let beta = rz_new / rz;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz = rz_new;
        }
        x
    }

    /// Apply the additive Schwarz preconditioner: M^{-1} r.
    ///
    /// This sums up the local corrections: M^{-1} r = Σ_s Rₛᵀ Aₛ⁻¹ Rₛ r
    fn precondition(&self, r: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut result = vec![0.0f64; n];
        // Each subdomain solves its local system for the local residual
        for sd in &self.subdomains {
            let m = sd.size;
            let idx = &sd.global_indices;
            // Local rhs: restrict r to this subdomain
            let r_loc: Vec<f64> = idx.iter().map(|&gi| r[gi]).collect();
            // Solve A_loc * z_loc = r_loc
            let z_loc = dense_lu_solve(&sd.lu_factor, &sd.lu_perm, &r_loc, m);
            // Scatter (prolongate) z_loc back: result[gi] += z_loc[li]
            for (li, &gi) in idx.iter().enumerate() {
                result[gi] += z_loc[li];
            }
        }
        result
    }

    /// Return the global dimension.
    pub fn size(&self) -> usize {
        self.n
    }

    /// Return the number of subdomains.
    pub fn num_subdomains(&self) -> usize {
        self.subdomains.len()
    }
}

// ---------------------------------------------------------------------------
// Schur Complement Solver
// ---------------------------------------------------------------------------

/// Schur complement method for block 2×2 systems.
///
/// Solves the system:
///
/// ```text
/// [ A  B ] [ x ]   [ f ]
/// [ C  D ] [ y ] = [ g ]
/// ```
///
/// via the Schur complement S = D − C A⁻¹ B.  The algorithm is:
///
/// 1. Solve A x̂ = f for x̂ (temporary RHS).
/// 2. Form g̃ = g − C x̂.
/// 3. Solve S y = g̃ for y via CG applied to the implicit Schur operator.
/// 4. Solve A x = f − B y for x.
///
/// The Schur complement is applied matrix-free; A is solved via an internal
/// dense LU (for moderate n1) or CG (for large n1).
pub struct SchurComplementSolver {
    n1: usize,
    n2: usize,
}

impl SchurComplementSolver {
    /// Solve the 2×2 block system and return `[x; y]` as a single vector.
    ///
    /// # Arguments
    ///
    /// All matrix arguments are in CSR format with the indicated block sizes.
    ///
    /// * `n1` – Number of rows/columns in block A (rows 0..n1 of the global system).
    /// * `n2` – Number of rows/columns in block D (rows n1..n1+n2).
    /// * `f`  – Right-hand side for the first block (length n1).
    /// * `g`  – Right-hand side for the second block (length n2).
    #[allow(clippy::too_many_arguments)]
    pub fn solve(
        a_val: &[f64],
        a_row_ptr: &[usize],
        a_col_ind: &[usize],
        n1: usize,
        b_val: &[f64],
        b_row_ptr: &[usize],
        b_col_ind: &[usize],
        c_val: &[f64],
        c_row_ptr: &[usize],
        c_col_ind: &[usize],
        d_val: &[f64],
        d_row_ptr: &[usize],
        d_col_ind: &[usize],
        n2: usize,
        f: &[f64],
        g: &[f64],
        max_iter: usize,
        tol: f64,
    ) -> SparseResult<Vec<f64>> {
        // --- 1. Build dense LU of A for moderate n1 ---
        let a_lu_opt: Option<(Vec<Vec<f64>>, Vec<usize>)> = if n1 <= 512 {
            let mut a_dense = vec![vec![0.0f64; n1]; n1];
            for i in 0..n1 {
                for pos in a_row_ptr[i]..a_row_ptr[i + 1] {
                    if a_col_ind[pos] < n1 {
                        a_dense[i][a_col_ind[pos]] = a_val[pos];
                    }
                }
            }
            match dense_lu_factor(&mut a_dense, n1) {
                Some(perm) => Some((a_dense, perm)),
                None => {
                    return Err(SparseError::SingularMatrix(
                        "block A is singular".to_string(),
                    ))
                }
            }
        } else {
            None
        };

        // Solve A z = rhs using LU (if available) or CG fallback.
        let solve_a = |rhs: &[f64]| -> Vec<f64> {
            match &a_lu_opt {
                Some((lu, perm)) => dense_lu_solve(lu, perm, rhs, n1),
                None => cg_internal(
                    |v| csr_matvec(a_row_ptr, a_col_ind, a_val, v, n1),
                    rhs,
                    n1,
                    max_iter,
                    tol,
                ),
            }
        };

        // --- 2. Temporary: solve A x̂ = f ---
        let x_hat = solve_a(f);

        // --- 3. g̃ = g − C x̂ ---
        let cx_hat = csr_matvec(c_row_ptr, c_col_ind, c_val, &x_hat, n2);
        let g_tilde: Vec<f64> = (0..n2).map(|i| g[i] - cx_hat[i]).collect();

        // --- 4. Solve S y = g̃ where S y = D y − C A^{-1} B y ---
        let schur_mv = |y: &[f64]| -> Vec<f64> {
            let by = csr_matvec(b_row_ptr, b_col_ind, b_val, y, n1);
            let a_inv_by = solve_a(&by);
            let dy = csr_matvec(d_row_ptr, d_col_ind, d_val, y, n2);
            let ca_inv_by = csr_matvec(c_row_ptr, c_col_ind, c_val, &a_inv_by, n2);
            (0..n2).map(|i| dy[i] - ca_inv_by[i]).collect()
        };

        let y = cg_internal(schur_mv, &g_tilde, n2, max_iter, tol);

        // --- 5. Solve A x = f − B y ---
        let by = csr_matvec(b_row_ptr, b_col_ind, b_val, &y, n1);
        let f_minus_by: Vec<f64> = (0..n1).map(|i| f[i] - by[i]).collect();
        let x = solve_a(&f_minus_by);

        // Concatenate [x; y]
        let mut result = x;
        result.extend_from_slice(&y);
        Ok(result)
    }

    /// Return n1 (size of the A block).
    pub fn n1(&self) -> usize {
        self.n1
    }

    /// Return n2 (size of the D block).
    pub fn n2(&self) -> usize {
        self.n2
    }

    /// Create a bookkeeping handle (sizes only; use `solve` for actual computation).
    pub fn new(n1: usize, n2: usize) -> Self {
        Self { n1, n2 }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build CSR for the n×n tridiagonal matrix with `diag` on main diagonal
    /// and `off` on super- and sub-diagonals.
    fn tridiag_csr(n: usize, diag: f64, off: f64) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
        let mut val = Vec::new();
        let mut col_ind = Vec::new();
        let mut row_ptr = vec![0usize; n + 1];
        for i in 0..n {
            if i > 0 {
                col_ind.push(i - 1);
                val.push(off);
            }
            col_ind.push(i);
            val.push(diag);
            if i + 1 < n {
                col_ind.push(i + 1);
                val.push(off);
            }
            row_ptr[i + 1] = val.len();
        }
        (val, row_ptr, col_ind)
    }

    #[test]
    fn test_schwarz_solver_constructs() {
        let n = 6;
        let (val, rp, ci) = tridiag_csr(n, 4.0, -1.0);
        let solver = SchwarzSolver::new(&val, &rp, &ci, n, 3, 1)
            .expect("SchwarzSolver::new");
        assert_eq!(solver.size(), n);
        assert_eq!(solver.num_subdomains(), 3);
    }

    #[test]
    fn test_schwarz_solver_apply() {
        let n = 4;
        let (val, rp, ci) = tridiag_csr(n, 4.0, -1.0);
        let solver = SchwarzSolver::new(&val, &rp, &ci, n, 2, 1)
            .expect("SchwarzSolver::new");
        let x = vec![0.0f64; n];
        let b = vec![1.0f64; n];
        let x_new = solver.apply(&x, &b);
        assert_eq!(x_new.len(), n);
        // After one Schwarz step starting from x=0, x_new should be non-zero
        let norm: f64 = x_new.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm > 1e-12, "Schwarz apply should give non-zero update");
    }

    #[test]
    fn test_schwarz_solver_solve_converges() {
        let n = 8;
        let (val, rp, ci) = tridiag_csr(n, 4.0, -1.0);
        let solver = SchwarzSolver::new(&val, &rp, &ci, n, 4, 1)
            .expect("SchwarzSolver::new");
        // b = e_0 (unit vector)
        let mut b = vec![0.0f64; n];
        b[0] = 1.0;
        let x = solver.solve(&b, 200, 1e-10);
        // Verify: A x ≈ b
        let ax = csr_matvec(&rp, &ci, &val, &x, n);
        let res: f64 = b.iter().zip(ax.iter()).map(|(bi, axi)| (bi - axi).powi(2)).sum::<f64>().sqrt();
        assert!(res < 1e-8, "Schwarz PCG residual {res}");
    }

    #[test]
    fn test_schwarz_invalid_args() {
        let n = 4;
        let (val, rp, ci) = tridiag_csr(n, 2.0, -0.5);
        // Zero subdomains
        let r = SchwarzSolver::new(&val, &rp, &ci, n, 0, 0);
        assert!(r.is_err());
        // Too many subdomains
        let r2 = SchwarzSolver::new(&val, &rp, &ci, n, n + 1, 0);
        assert!(r2.is_err());
    }

    #[test]
    fn test_schur_complement_2x2() {
        // System: A = [3 0; 0 3], B = [-1 0; 0 -1], C = B^T, D = [2 0; 0 2]
        // Full system: [3  0 -1  0] [x0]   [1]
        //              [0  3  0 -1] [x1] = [0]
        //              [-1 0  2  0] [y0]   [0]
        //              [0 -1  0  2] [y1]   [1]
        let n1 = 2;
        let n2 = 2;

        // A = 3*I_{2}
        let a_rp = vec![0, 1, 2];
        let a_ci = vec![0, 1];
        let a_v = vec![3.0, 3.0];

        // B = -I_{2}
        let b_rp = vec![0, 1, 2];
        let b_ci = vec![0, 1];
        let b_v = vec![-1.0, -1.0];

        // C = -I_{2}
        let c_rp = vec![0, 1, 2];
        let c_ci = vec![0, 1];
        let c_v = vec![-1.0, -1.0];

        // D = 2*I_{2}
        let d_rp = vec![0, 1, 2];
        let d_ci = vec![0, 1];
        let d_v = vec![2.0, 2.0];

        let f = vec![1.0, 0.0];
        let g = vec![0.0, 1.0];

        let sol = SchurComplementSolver::solve(
            &a_v, &a_rp, &a_ci, n1,
            &b_v, &b_rp, &b_ci,
            &c_v, &c_rp, &c_ci,
            &d_v, &d_rp, &d_ci, n2,
            &f, &g,
            100, 1e-12,
        ).expect("Schur solve");

        assert_eq!(sol.len(), n1 + n2);

        // Verify by substitution: full system A_{full} sol ≈ [f; g]
        // Row 0: 3*sol[0] - 1*sol[2] = 1
        let r0 = 3.0 * sol[0] - sol[2] - 1.0;
        let r1 = 3.0 * sol[1] - sol[3] - 0.0;
        let r2 = -sol[0] + 2.0 * sol[2] - 0.0;
        let r3 = -sol[1] + 2.0 * sol[3] - 1.0;
        for (i, r) in [r0, r1, r2, r3].iter().enumerate() {
            assert!(r.abs() < 1e-8, "residual[{i}] = {r}");
        }
    }
}
