//! Hybrid multigrid solver with learned smoother integration.
//!
//! Implements a simple 2-level multigrid V-cycle where the fine-level
//! smoother is replaced with a learned smoother (linear or MLP).
//! Falls back to classical Jacobi if the learned smoother increases
//! the residual.

use crate::error::{SparseError, SparseResult};
use crate::learned_smoother::types::{LearnedSmootherConfig, Smoother, SmootherMetrics};

/// Type alias for geometric coarsening result:
/// (R_values, R_col_indices, R_row_ptr, P_values, P_col_indices, P_row_ptr, coarse_n).
type CoarseningResult = (
    Vec<f64>,
    Vec<usize>,
    Vec<usize>,
    Vec<f64>,
    Vec<usize>,
    Vec<usize>,
    usize,
);

// ---------------------------------------------------------------------------
// CSR helpers
// ---------------------------------------------------------------------------

/// y = A x (CSR).
fn csr_matvec(a_values: &[f64], a_row_ptr: &[usize], a_col_idx: &[usize], x: &[f64]) -> Vec<f64> {
    let n = a_row_ptr.len().saturating_sub(1);
    let mut y = vec![0.0; n];
    for i in 0..n {
        let start = a_row_ptr[i];
        let end = a_row_ptr[i + 1];
        let mut sum = 0.0;
        for pos in start..end {
            sum += a_values[pos] * x[a_col_idx[pos]];
        }
        y[i] = sum;
    }
    y
}

/// r = b - A x.
fn compute_residual(
    a_values: &[f64],
    a_row_ptr: &[usize],
    a_col_idx: &[usize],
    x: &[f64],
    b: &[f64],
) -> Vec<f64> {
    let ax = csr_matvec(a_values, a_row_ptr, a_col_idx, x);
    b.iter()
        .zip(ax.iter())
        .map(|(&bi, &axi)| bi - axi)
        .collect()
}

/// Euclidean norm.
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Extract diagonal of A (CSR).
fn extract_diagonal(
    a_values: &[f64],
    a_row_ptr: &[usize],
    a_col_idx: &[usize],
    n: usize,
) -> Vec<f64> {
    let mut diag = vec![0.0; n];
    for i in 0..n {
        let start = a_row_ptr[i];
        let end = a_row_ptr[i + 1];
        for pos in start..end {
            if a_col_idx[pos] == i {
                diag[i] = a_values[pos];
                break;
            }
        }
    }
    diag
}

/// Weighted Jacobi smoother: x += omega * D^{-1} * r.
fn jacobi_smooth(
    a_values: &[f64],
    a_row_ptr: &[usize],
    a_col_idx: &[usize],
    x: &mut [f64],
    b: &[f64],
    diag_inv: &[f64],
    omega: f64,
    n_sweeps: usize,
) {
    let n = x.len();
    for _ in 0..n_sweeps {
        let r = compute_residual(a_values, a_row_ptr, a_col_idx, x, b);
        for i in 0..n {
            x[i] += omega * diag_inv[i] * r[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Simple coarsening
// ---------------------------------------------------------------------------

/// Build a simple geometric coarsening: keep every other node.
/// Returns (restriction, prolongation, coarse_n).
///
/// Restriction R is n_c x n injection: R[c, 2c] = 1.
/// Prolongation P is n x n_c linear interpolation.
fn build_coarsening(n: usize) -> CoarseningResult {
    let n_c = (n + 1) / 2;

    // --- Restriction (n_c x n) in CSR ---
    let mut r_values = Vec::new();
    let mut r_col_idx = Vec::new();
    let mut r_row_ptr = vec![0usize];

    for c in 0..n_c {
        let fine = 2 * c;
        if fine > 0 {
            r_values.push(0.25);
            r_col_idx.push(fine - 1);
        }
        r_values.push(0.5);
        r_col_idx.push(fine);
        if fine + 1 < n {
            r_values.push(0.25);
            r_col_idx.push(fine + 1);
        }
        r_row_ptr.push(r_values.len());
    }

    // --- Prolongation (n x n_c) in CSR ---
    let mut p_values = Vec::new();
    let mut p_col_idx = Vec::new();
    let mut p_row_ptr = vec![0usize];

    for i in 0..n {
        if i % 2 == 0 {
            // Coarse point: direct injection
            let c = i / 2;
            p_values.push(1.0);
            p_col_idx.push(c);
        } else {
            // Fine point: average of two coarse neighbors
            let c_left = i / 2;
            let c_right = c_left + 1;
            p_values.push(0.5);
            p_col_idx.push(c_left);
            if c_right < n_c {
                p_values.push(0.5);
                p_col_idx.push(c_right);
            }
        }
        p_row_ptr.push(p_values.len());
    }

    (
        r_values, r_row_ptr, r_col_idx, p_values, p_row_ptr, p_col_idx, n_c,
    )
}

/// Compute A_c = R A P (Galerkin coarse-grid operator) using raw CSR operations.
///
/// This is a simplified implementation for the 2-level multigrid.
fn compute_coarse_operator(
    a_values: &[f64],
    a_row_ptr: &[usize],
    a_col_idx: &[usize],
    n: usize,
    r_values: &[f64],
    r_row_ptr: &[usize],
    r_col_idx: &[usize],
    p_values: &[f64],
    p_row_ptr: &[usize],
    p_col_idx: &[usize],
    n_c: usize,
) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    // Build A_c row by row using dense accumulation (fine for small coarse grids)
    let mut ac_dense = vec![vec![0.0; n_c]; n_c];

    for ic in 0..n_c {
        // e_c = unit vector in coarse space
        let mut e_c = vec![0.0; n_c];
        e_c[ic] = 1.0;

        // p_e = P * e_c (prolongation)
        let p_e = csr_matvec(p_values, p_row_ptr, p_col_idx, &e_c);

        // a_p_e = A * P * e_c
        let a_p_e = csr_matvec(a_values, a_row_ptr, a_col_idx, &p_e);

        // r_a_p_e = R * A * P * e_c
        let r_a_p_e = csr_matvec(r_values, r_row_ptr, r_col_idx, &a_p_e);

        for jc in 0..n_c {
            ac_dense[jc][ic] = r_a_p_e[jc];
        }
    }

    // Convert dense to CSR
    let mut ac_values = Vec::new();
    let mut ac_col_idx = Vec::new();
    let mut ac_row_ptr = vec![0usize];

    for i in 0..n_c {
        for j in 0..n_c {
            if ac_dense[i][j].abs() > 1e-14 {
                ac_values.push(ac_dense[i][j]);
                ac_col_idx.push(j);
            }
        }
        ac_row_ptr.push(ac_values.len());
    }

    (ac_values, ac_row_ptr, ac_col_idx)
}

/// Direct solve for small dense system (Gaussian elimination).
fn direct_solve(
    a_values: &[f64],
    a_row_ptr: &[usize],
    a_col_idx: &[usize],
    b: &[f64],
    n: usize,
) -> SparseResult<Vec<f64>> {
    if n == 0 {
        return Ok(Vec::new());
    }

    // Convert CSR to dense
    let mut dense = vec![vec![0.0; n]; n];
    for i in 0..n {
        let start = a_row_ptr[i];
        let end = a_row_ptr[i + 1];
        for pos in start..end {
            dense[i][a_col_idx[pos]] = a_values[pos];
        }
    }

    // Augment with RHS
    let mut aug: Vec<Vec<f64>> = dense
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(SparseError::SingularMatrix(
                "Coarse grid operator is singular".to_string(),
            ));
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// HybridMultigridSolver
// ---------------------------------------------------------------------------

/// Hybrid 2-level multigrid solver with a learned smoother at the fine level.
///
/// The solver performs V-cycles:
/// 1. Pre-smooth (learned smoother)
/// 2. Restrict residual to coarse grid
/// 3. Solve coarse system (direct solve)
/// 4. Prolongate correction to fine grid
/// 5. Post-smooth (learned smoother)
///
/// If the learned smoother increases the residual, the solver falls back
/// to classical weighted Jacobi.
pub struct HybridMultigridSolver {
    // Fine grid operator (CSR)
    a_values: Vec<f64>,
    a_row_ptr: Vec<usize>,
    a_col_idx: Vec<usize>,
    n: usize,

    // Coarse grid operator (CSR)
    ac_values: Vec<f64>,
    ac_row_ptr: Vec<usize>,
    ac_col_idx: Vec<usize>,
    n_c: usize,

    // Transfer operators (CSR)
    r_values: Vec<f64>,
    r_row_ptr: Vec<usize>,
    r_col_idx: Vec<usize>,
    p_values: Vec<f64>,
    p_row_ptr: Vec<usize>,
    p_col_idx: Vec<usize>,

    // Smoother
    smoother: Box<dyn Smoother>,

    // Fallback diagonal inverse for Jacobi
    diag_inv: Vec<f64>,

    // Configuration
    config: LearnedSmootherConfig,
}

impl HybridMultigridSolver {
    /// Create a new hybrid multigrid solver.
    ///
    /// # Arguments
    /// - `a_values`, `a_row_ptr`, `a_col_idx`: fine-grid operator in CSR
    /// - `smoother`: a trained (or untrained) learned smoother
    /// - `config`: solver configuration
    pub fn new(
        a_values: Vec<f64>,
        a_row_ptr: Vec<usize>,
        a_col_idx: Vec<usize>,
        smoother: Box<dyn Smoother>,
        config: LearnedSmootherConfig,
    ) -> SparseResult<Self> {
        let n = a_row_ptr.len().saturating_sub(1);
        if n == 0 {
            return Err(SparseError::ValueError(
                "Matrix dimension must be positive".to_string(),
            ));
        }

        // Build coarsening operators
        let (r_values, r_row_ptr, r_col_idx, p_values, p_row_ptr, p_col_idx, n_c) =
            build_coarsening(n);

        // Build coarse operator
        let (ac_values, ac_row_ptr, ac_col_idx) = compute_coarse_operator(
            &a_values, &a_row_ptr, &a_col_idx, n, &r_values, &r_row_ptr, &r_col_idx, &p_values,
            &p_row_ptr, &p_col_idx, n_c,
        );

        // Compute diagonal inverse for Jacobi fallback
        let diag = extract_diagonal(&a_values, &a_row_ptr, &a_col_idx, n);
        let diag_inv: Vec<f64> = diag
            .iter()
            .map(|&d| if d.abs() > f64::EPSILON { 1.0 / d } else { 1.0 })
            .collect();

        Ok(Self {
            a_values,
            a_row_ptr,
            a_col_idx,
            n,
            ac_values,
            ac_row_ptr,
            ac_col_idx,
            n_c,
            r_values,
            r_row_ptr,
            r_col_idx,
            p_values,
            p_row_ptr,
            p_col_idx,
            smoother,
            diag_inv,
            config,
        })
    }

    /// Perform one V-cycle: pre-smooth → restrict → coarse-solve → prolongate → post-smooth.
    fn v_cycle(&self, x: &mut [f64], b: &[f64]) -> SparseResult<()> {
        // 1. Pre-smooth
        self.smoother.smooth(
            &self.a_values,
            &self.a_row_ptr,
            &self.a_col_idx,
            x,
            b,
            self.config.pre_sweeps,
        )?;

        // 2. Compute fine-grid residual
        let r_fine = compute_residual(&self.a_values, &self.a_row_ptr, &self.a_col_idx, x, b);

        // 3. Restrict residual to coarse grid: r_c = R * r_fine
        let r_coarse = csr_matvec(&self.r_values, &self.r_row_ptr, &self.r_col_idx, &r_fine);

        // 4. Solve coarse system: A_c * e_c = r_c
        let e_coarse = direct_solve(
            &self.ac_values,
            &self.ac_row_ptr,
            &self.ac_col_idx,
            &r_coarse,
            self.n_c,
        )?;

        // 5. Prolongate correction: e_fine = P * e_c
        let e_fine = csr_matvec(&self.p_values, &self.p_row_ptr, &self.p_col_idx, &e_coarse);

        // 6. Apply correction
        for i in 0..self.n {
            x[i] += e_fine[i];
        }

        // 7. Post-smooth
        self.smoother.smooth(
            &self.a_values,
            &self.a_row_ptr,
            &self.a_col_idx,
            x,
            b,
            self.config.post_sweeps,
        )?;

        Ok(())
    }

    /// Perform one V-cycle using classical Jacobi (fallback).
    fn v_cycle_jacobi(&self, x: &mut [f64], b: &[f64]) -> SparseResult<()> {
        let omega = self.config.omega;
        let pre = self.config.pre_sweeps;
        let post = self.config.post_sweeps;

        // Pre-smooth with Jacobi
        jacobi_smooth(
            &self.a_values,
            &self.a_row_ptr,
            &self.a_col_idx,
            x,
            b,
            &self.diag_inv,
            omega,
            pre,
        );

        // Restrict
        let r_fine = compute_residual(&self.a_values, &self.a_row_ptr, &self.a_col_idx, x, b);
        let r_coarse = csr_matvec(&self.r_values, &self.r_row_ptr, &self.r_col_idx, &r_fine);

        // Coarse solve
        let e_coarse = direct_solve(
            &self.ac_values,
            &self.ac_row_ptr,
            &self.ac_col_idx,
            &r_coarse,
            self.n_c,
        )?;

        // Prolongate
        let e_fine = csr_matvec(&self.p_values, &self.p_row_ptr, &self.p_col_idx, &e_coarse);
        for i in 0..self.n {
            x[i] += e_fine[i];
        }

        // Post-smooth with Jacobi
        jacobi_smooth(
            &self.a_values,
            &self.a_row_ptr,
            &self.a_col_idx,
            x,
            b,
            &self.diag_inv,
            omega,
            post,
        );

        Ok(())
    }

    /// Solve A x = b using iterative V-cycles.
    ///
    /// Returns the solution vector. Falls back to Jacobi smoothing if the
    /// learned smoother causes divergence.
    pub fn solve(&self, b: &[f64]) -> SparseResult<Vec<f64>> {
        if b.len() != self.n {
            return Err(SparseError::DimensionMismatch {
                expected: self.n,
                found: b.len(),
            });
        }

        let mut x = vec![0.0; self.n];
        let tol = self.config.convergence_tol;
        let max_iter = self.config.max_training_steps;

        let r0 = compute_residual(&self.a_values, &self.a_row_ptr, &self.a_col_idx, &x, b);
        let norm0 = vec_norm(&r0);
        if norm0 < tol {
            return Ok(x);
        }

        let mut use_learned = true;
        let mut prev_norm = norm0;

        for _iter in 0..max_iter {
            if use_learned {
                let x_backup: Vec<f64> = x.clone();
                let result = self.v_cycle(&mut x, b);

                let r = compute_residual(&self.a_values, &self.a_row_ptr, &self.a_col_idx, &x, b);
                let norm_r = vec_norm(&r);

                // Check for divergence: if residual increased, fall back to Jacobi
                if result.is_err() || norm_r > prev_norm * 2.0 {
                    x.copy_from_slice(&x_backup);
                    use_learned = false;
                    self.v_cycle_jacobi(&mut x, b)?;
                    let r2 =
                        compute_residual(&self.a_values, &self.a_row_ptr, &self.a_col_idx, &x, b);
                    prev_norm = vec_norm(&r2);
                } else {
                    prev_norm = norm_r;
                }
            } else {
                self.v_cycle_jacobi(&mut x, b)?;
                let r = compute_residual(&self.a_values, &self.a_row_ptr, &self.a_col_idx, &x, b);
                prev_norm = vec_norm(&r);
            }

            if prev_norm < tol * norm0 {
                break;
            }
        }

        Ok(x)
    }

    /// Compare learned smoother convergence with classical Jacobi.
    ///
    /// Runs both for `n_iterations` V-cycles and returns metrics.
    pub fn compare_with_jacobi(
        &self,
        b: &[f64],
        n_iterations: usize,
    ) -> SparseResult<SmootherMetrics> {
        if b.len() != self.n {
            return Err(SparseError::DimensionMismatch {
                expected: self.n,
                found: b.len(),
            });
        }

        // --- Run learned smoother ---
        let mut x_learned = vec![0.0; self.n];
        let r0 = compute_residual(
            &self.a_values,
            &self.a_row_ptr,
            &self.a_col_idx,
            &x_learned,
            b,
        );
        let norm0 = vec_norm(&r0);

        let mut learned_history = Vec::with_capacity(n_iterations);
        for _ in 0..n_iterations {
            let _ = self.v_cycle(&mut x_learned, b);
            let r = compute_residual(
                &self.a_values,
                &self.a_row_ptr,
                &self.a_col_idx,
                &x_learned,
                b,
            );
            learned_history.push(vec_norm(&r));
        }

        // --- Run Jacobi ---
        let mut x_jacobi = vec![0.0; self.n];
        let mut jacobi_final_norm = norm0;
        for _ in 0..n_iterations {
            self.v_cycle_jacobi(&mut x_jacobi, b)?;
            let r = compute_residual(
                &self.a_values,
                &self.a_row_ptr,
                &self.a_col_idx,
                &x_jacobi,
                b,
            );
            jacobi_final_norm = vec_norm(&r);
        }

        let learned_final = learned_history.last().copied().unwrap_or(norm0);

        // Convergence factor: geometric mean of reduction per step
        let convergence_factor = if n_iterations > 0 && norm0 > f64::EPSILON {
            (learned_final / norm0).powf(1.0 / n_iterations as f64)
        } else {
            1.0
        };

        let residual_reduction = if norm0 > f64::EPSILON {
            learned_final / norm0
        } else {
            0.0
        };

        let spectral_radius_reduction = if jacobi_final_norm > f64::EPSILON {
            learned_final / jacobi_final_norm
        } else {
            1.0
        };

        Ok(SmootherMetrics {
            spectral_radius_reduction,
            convergence_factor,
            residual_reduction,
            training_loss_history: learned_history,
        })
    }

    /// Problem dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Coarse grid dimension.
    pub fn coarse_dim(&self) -> usize {
        self.n_c
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned_smoother::linear_smoother::LinearSmoother;

    /// Build 1D Poisson matrix of size n: tridiag(-1, 2, -1).
    fn poisson_1d(n: usize) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
        let mut values = Vec::new();
        let mut col_idx = Vec::new();
        let mut row_ptr = vec![0usize];

        for i in 0..n {
            if i > 0 {
                values.push(-1.0);
                col_idx.push(i - 1);
            }
            values.push(2.0);
            col_idx.push(i);
            if i + 1 < n {
                values.push(-1.0);
                col_idx.push(i + 1);
            }
            row_ptr.push(values.len());
        }
        (values, row_ptr, col_idx)
    }

    #[test]
    fn test_coarsening_dimensions() {
        let (_, _, _, _, _, _, n_c) = build_coarsening(7);
        assert_eq!(n_c, 4);
        let (_, _, _, _, _, _, n_c2) = build_coarsening(8);
        assert_eq!(n_c2, 4);
    }

    #[test]
    fn test_hybrid_solver_solves_poisson() {
        let n = 7;
        let (vals, rp, ci) = poisson_1d(n);

        let smoother = LinearSmoother::from_csr(&vals, &rp, &ci, 2.0 / 3.0);
        let config = LearnedSmootherConfig {
            max_training_steps: 200,
            convergence_tol: 1e-8,
            ..LearnedSmootherConfig::default()
        };

        let solver = HybridMultigridSolver::new(
            vals.clone(),
            rp.clone(),
            ci.clone(),
            Box::new(smoother),
            config,
        )
        .expect("solver creation");

        // RHS: b = [1, 0, 0, ..., 0, 1]
        let mut b = vec![0.0; n];
        b[0] = 1.0;
        b[n - 1] = 1.0;

        let x = solver.solve(&b).expect("solve");

        // Verify: ‖b - Ax‖ / ‖b‖ should be small
        let r = compute_residual(&vals, &rp, &ci, &x, &b);
        let rel_res = vec_norm(&r) / vec_norm(&b);
        assert!(
            rel_res < 1e-4,
            "Relative residual should be small, got {rel_res}"
        );
    }

    #[test]
    fn test_compare_with_jacobi() {
        let n = 7;
        let (vals, rp, ci) = poisson_1d(n);

        let smoother = LinearSmoother::from_csr(&vals, &rp, &ci, 2.0 / 3.0);
        let config = LearnedSmootherConfig::default();

        let solver = HybridMultigridSolver::new(
            vals.clone(),
            rp.clone(),
            ci.clone(),
            Box::new(smoother),
            config,
        )
        .expect("solver creation");

        let mut b = vec![1.0; n];
        b[0] = 2.0;

        let metrics = solver.compare_with_jacobi(&b, 10).expect("compare");

        assert!(
            metrics.convergence_factor < 1.0,
            "Convergence factor should be < 1, got {}",
            metrics.convergence_factor
        );
        assert_eq!(metrics.training_loss_history.len(), 10);
    }

    #[test]
    fn test_solver_dimension_mismatch() {
        let n = 5;
        let (vals, rp, ci) = poisson_1d(n);
        let smoother = LinearSmoother::from_csr(&vals, &rp, &ci, 2.0 / 3.0);
        let config = LearnedSmootherConfig::default();

        let solver = HybridMultigridSolver::new(vals, rp, ci, Box::new(smoother), config)
            .expect("solver creation");

        let b = vec![1.0; 3]; // wrong size
        assert!(solver.solve(&b).is_err());
    }

    #[test]
    fn test_solver_coarse_dim() {
        let n = 9;
        let (vals, rp, ci) = poisson_1d(n);
        let smoother = LinearSmoother::from_csr(&vals, &rp, &ci, 2.0 / 3.0);
        let config = LearnedSmootherConfig::default();

        let solver = HybridMultigridSolver::new(vals, rp, ci, Box::new(smoother), config)
            .expect("solver creation");

        assert_eq!(solver.dim(), 9);
        assert_eq!(solver.coarse_dim(), 5);
    }

    #[test]
    fn test_direct_solve_small_system() {
        // 2x2 system: [2 1; 1 3] x = [5; 7] => x = [8/5, 9/5] = [1.6, 1.8]
        let vals = vec![2.0, 1.0, 1.0, 3.0];
        let rp = vec![0, 2, 4];
        let ci = vec![0, 1, 0, 1];
        let b = vec![5.0, 7.0];

        let x = direct_solve(&vals, &rp, &ci, &b, 2).expect("direct solve");
        assert!((x[0] - 1.6).abs() < 1e-10);
        assert!((x[1] - 1.8).abs() < 1e-10);
    }
}
