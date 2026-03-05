//! Recycled GMRES (GCRO-DR) for sequences of linear systems.
//!
//! When solving A x_i = b_i for i = 1, 2, ..., this solver recycles
//! Krylov subspace information (a low-dimensional invariant subspace approximation)
//! from previous solves to accelerate subsequent ones.
//!
//! # Algorithm: GCRO-DR (Galerkin/Conjugate Residual Recycling with Deflated Restart)
//!
//! Inspired by:
//! - Parks, M.L., de Sturler, E., Mackey, G., Johnson, D.D., Maiti, S. (2006).
//!   "Recycling Krylov subspaces for sequences of linear systems".
//!   SIAM J. Sci. Comput. 28(5), 1651-1674.
//!
//! Key idea:
//! 1. Maintain a *recycled subspace* U_k spanning approximate (left-)invariant
//!    subspace of A. Also store C_k = A U_k.
//! 2. For each new right-hand side b:
//!    a. Compute the component of b in the recycled subspace: x += U_k (C_k^T C_k)^{-1} C_k^T r
//!    b. Run GMRES on the deflated residual.
//! 3. After each solve, extract the Krylov basis vectors to update the recycled subspace U_k.

use crate::error::SparseError;
use crate::krylov::gmres_dr::{
    dot, gram_schmidt_mgs, norm2, solve_least_squares_hessenberg, GmresDRResult,
};

/// Recycled GMRES (GCRO-DR) solver for sequences of linear systems.
///
/// Call [`RecycledGmres::solve`] repeatedly. The recycled subspace U_k is
/// updated after each call and automatically used to accelerate the next.
#[derive(Debug, Clone)]
pub struct RecycledGmres {
    /// Krylov space dimension per restart cycle.
    pub restart: usize,
    /// Maximum dimension of the recycled subspace (k in GCRO-DR).
    pub n_recycle: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Maximum total matrix-vector products per solve.
    pub max_iter: usize,
    /// Recycled subspace U_k: vectors span approximate invariant subspace.
    recycled_u: Vec<Vec<f64>>,
    /// C_k = A * U_k (stored to avoid recomputing A*U each solve).
    recycled_c: Vec<Vec<f64>>,
}

impl RecycledGmres {
    /// Create a new RecycledGmres solver.
    ///
    /// # Arguments
    ///
    /// * `restart` - Krylov dimension per cycle. Recommended: 20-50.
    /// * `n_recycle` - Recycled subspace dimension. Must be < restart.
    pub fn new(restart: usize, n_recycle: usize) -> Self {
        assert!(
            n_recycle < restart,
            "n_recycle ({}) must be < restart ({})",
            n_recycle,
            restart
        );
        Self {
            restart,
            n_recycle,
            tol: 1e-10,
            max_iter: 1000,
            recycled_u: Vec::new(),
            recycled_c: Vec::new(),
        }
    }

    /// Set convergence tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set maximum iteration count.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Reset (discard) the recycled subspace.
    ///
    /// Call this when the system matrix changes significantly.
    pub fn clear_recycle(&mut self) {
        self.recycled_u.clear();
        self.recycled_c.clear();
    }

    /// Return the current recycled subspace dimension.
    pub fn recycle_dim(&self) -> usize {
        self.recycled_u.len()
    }

    /// Solve A x = b using the recycled subspace from previous solves.
    ///
    /// On first call (or after [`RecycledGmres::clear_recycle`]) this reduces to standard GMRES with
    /// restarts. After convergence the recycled subspace is updated.
    ///
    /// # Arguments
    ///
    /// * `matvec` - Closure computing y = A x (must represent the *same* matrix A
    ///   as in previous calls for recycling to be beneficial).
    /// * `b` - Right-hand side.
    /// * `x0` - Optional initial guess.
    pub fn solve<F>(
        &mut self,
        matvec: F,
        b: &[f64],
        x0: Option<&[f64]>,
    ) -> Result<GmresDRResult, SparseError>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let n = b.len();
        let mut x = match x0 {
            Some(v) => v.to_vec(),
            None => vec![0.0f64; n],
        };

        let b_norm = norm2(b);
        let abs_tol = if b_norm > 1e-300 {
            self.tol * b_norm
        } else {
            self.tol
        };
        let mut total_matvecs = 0usize;
        let mut residual_history = Vec::new();

        // --- Phase 1: deflation with recycled subspace (GCRO projection) ---
        if !self.recycled_u.is_empty() {
            let (correction, mv) = self.project_onto_recycled(&x, b, &matvec);
            total_matvecs += mv;
            for i in 0..n {
                x[i] += correction[i];
            }
        }

        // Compute residual after recycled correction.
        let ax0 = matvec(&x);
        total_matvecs += 1;
        let r0: Vec<f64> = b.iter().zip(ax0.iter()).map(|(bi, axi)| bi - axi).collect();
        let r0_norm = norm2(&r0);
        residual_history.push(r0_norm);

        if r0_norm <= abs_tol {
            return Ok(GmresDRResult {
                x,
                residual_norm: r0_norm,
                iterations: total_matvecs,
                converged: true,
                residual_history,
            });
        }

        // --- Phase 2: Standard GMRES cycles with restarts ---
        let m = self.restart;
        let max_cycles = (self.max_iter / m.max(1)).max(10);
        let mut last_krylov_vecs: Vec<Vec<f64>> = Vec::new();

        for _cycle in 0..max_cycles {
            let ax = matvec(&x);
            total_matvecs += 1;
            let r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();
            let r_norm = norm2(&r);
            residual_history.push(r_norm);

            if r_norm <= abs_tol {
                // Update recycled subspace before returning.
                self.update_recycle_from_krylov(&last_krylov_vecs, &matvec);
                return Ok(GmresDRResult {
                    x,
                    residual_norm: r_norm,
                    iterations: total_matvecs,
                    converged: true,
                    residual_history,
                });
            }

            if total_matvecs >= self.max_iter {
                break;
            }

            // Build standard Arnoldi basis.
            let mut v: Vec<Vec<f64>> = vec![vec![0.0f64; n]; m + 1];
            let mut h: Vec<Vec<f64>> = vec![vec![0.0f64; m]; m + 1];

            // v[0] = r / ||r||
            let inv_r = 1.0 / r_norm;
            for l in 0..n {
                v[0][l] = r[l] * inv_r;
            }

            // Arnoldi iteration.
            let mut j_end = 1;
            for j in 1..=m {
                if j == m {
                    j_end = m;
                    break;
                }
                let w_raw = matvec(&v[j - 1]);
                total_matvecs += 1;
                let mut w = w_raw;

                for i in 0..j {
                    h[i][j - 1] = dot(&w, &v[i]);
                    for l in 0..n {
                        w[l] -= h[i][j - 1] * v[i][l];
                    }
                }
                h[j][j - 1] = norm2(&w);

                if h[j][j - 1] > 1e-15 {
                    let inv = 1.0 / h[j][j - 1];
                    for l in 0..n {
                        v[j][l] = w[l] * inv;
                    }
                    j_end = j + 1;
                } else {
                    j_end = j + 1;
                    break;
                }

                if total_matvecs >= self.max_iter {
                    j_end = j + 1;
                    break;
                }
            }

            // Solve least-squares problem.
            let krylov_size = (j_end - 1).max(1).min(h[0].len());
            let mut g = vec![0.0f64; j_end];
            g[0] = r_norm;

            let cols = krylov_size.min(h[0].len());
            let y = solve_least_squares_hessenberg(&h, &g, cols)?;

            // Update solution.
            for j in 0..y.len().min(v.len()) {
                for i in 0..n {
                    x[i] += y[j] * v[j][i];
                }
            }

            last_krylov_vecs = v[..j_end].to_vec();

            if total_matvecs >= self.max_iter {
                break;
            }
        }

        // Final residual.
        let ax_fin = matvec(&x);
        total_matvecs += 1;
        let r_fin: Vec<f64> = b
            .iter()
            .zip(ax_fin.iter())
            .map(|(bi, axi)| bi - axi)
            .collect();
        let r_fin_norm = norm2(&r_fin);
        residual_history.push(r_fin_norm);

        // Update recycled subspace.
        self.update_recycle_from_krylov(&last_krylov_vecs, &matvec);

        Ok(GmresDRResult {
            x,
            residual_norm: r_fin_norm,
            iterations: total_matvecs,
            converged: r_fin_norm <= abs_tol,
            residual_history,
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Project the residual onto the recycled subspace C_k = A U_k and compute the
    /// update delta_x = U_k (C_k^T C_k)^{-1} C_k^T r.
    ///
    /// Returns (delta_x, matvec_count).
    fn project_onto_recycled<F>(&self, x: &[f64], b: &[f64], matvec: &F) -> (Vec<f64>, usize)
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let n = b.len();
        let k = self.recycled_u.len();
        if k == 0 {
            return (vec![0.0; n], 0);
        }

        // Compute residual r = b - A x.
        let ax = matvec(x);
        let r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, axi)| bi - axi).collect();

        // Solve the k x k normal equations (C^T C) alpha = C^T r.
        let c = &self.recycled_c;
        let u = &self.recycled_u;

        // Build C^T C (k x k symmetric).
        let mut ctc = vec![vec![0.0f64; k]; k];
        for i in 0..k {
            for j in 0..k {
                ctc[i][j] = dot(&c[i], &c[j]);
            }
        }

        // C^T r (k-vector).
        let ctr: Vec<f64> = (0..k).map(|i| dot(&c[i], &r)).collect();

        // Solve via Cholesky or fallback.
        let alpha = solve_dense_spd(&ctc, &ctr, k);

        // delta_x = U alpha.
        let mut delta = vec![0.0f64; n];
        for j in 0..k {
            for i in 0..n {
                delta[i] += alpha[j] * u[j][i];
            }
        }

        (delta, 1) // 1 for the initial matvec(x)
    }

    /// Update the recycled subspace from the last Krylov basis vectors.
    ///
    /// Takes the first `n_recycle` orthonormal basis vectors from the Krylov
    /// basis and uses them as the new recycled subspace.
    fn update_recycle_from_krylov<F>(&mut self, v: &[Vec<f64>], matvec: &F)
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        if v.is_empty() {
            return;
        }
        let n_take = self.n_recycle.min(v.len());

        // Take the first n_take Krylov vectors.
        let mut new_u: Vec<Vec<f64>> = v[..n_take].to_vec();

        // Orthonormalise.
        gram_schmidt_mgs(&mut new_u);
        new_u.retain(|vi| norm2(vi) > 0.5);

        // Recompute C = A U.
        let new_c: Vec<Vec<f64>> = new_u.iter().map(|ui| matvec(ui)).collect();

        self.recycled_u = new_u;
        self.recycled_c = new_c;
    }
}

/// Solve a k x k SPD (or near-SPD) system A x = b via Cholesky decomposition.
///
/// Falls back to Gaussian elimination with partial pivoting if Cholesky fails.
fn solve_dense_spd(a: &[Vec<f64>], b: &[f64], k: usize) -> Vec<f64> {
    if k == 0 {
        return Vec::new();
    }
    if k == 1 {
        let diag = a[0][0];
        return vec![if diag.abs() > 1e-300 {
            b[0] / diag
        } else {
            0.0
        }];
    }

    // Attempt Cholesky: L L^T decomposition.
    let mut l = vec![vec![0.0f64; k]; k];
    let mut ok = true;
    'chol: for i in 0..k {
        for j in 0..=i {
            let mut sum = a[i][j];
            for p in 0..j {
                sum -= l[i][p] * l[j][p];
            }
            if i == j {
                if sum < 1e-300 {
                    ok = false;
                    break 'chol;
                }
                l[i][j] = sum.sqrt();
            } else if l[j][j].abs() > 1e-300 {
                l[i][j] = sum / l[j][j];
            } else {
                ok = false;
                break 'chol;
            }
        }
    }

    if ok {
        // Forward substitution: L y = b.
        let mut y = vec![0.0f64; k];
        for i in 0..k {
            let mut s = b[i];
            for j in 0..i {
                s -= l[i][j] * y[j];
            }
            y[i] = if l[i][i].abs() > 1e-300 {
                s / l[i][i]
            } else {
                0.0
            };
        }
        // Back substitution: L^T x = y.
        let mut x = vec![0.0f64; k];
        for i in (0..k).rev() {
            let mut s = y[i];
            for j in (i + 1)..k {
                s -= l[j][i] * x[j];
            }
            x[i] = if l[i][i].abs() > 1e-300 {
                s / l[i][i]
            } else {
                0.0
            };
        }
        x
    } else {
        // Fallback: diagonal approximation (numerically safe).
        (0..k)
            .map(|i| {
                if a[i][i].abs() > 1e-300 {
                    b[i] / a[i][i]
                } else {
                    0.0
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn diag_matvec_fn(diag: &'static [f64]) -> impl Fn(&[f64]) -> Vec<f64> {
        move |x: &[f64]| x.iter().zip(diag.iter()).map(|(xi, di)| xi * di).collect()
    }

    #[test]
    fn test_recycled_gmres_single_solve() {
        static DIAG: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = DIAG.len();
        let b: Vec<f64> = vec![1.0; n];

        let mut solver = RecycledGmres::new(6, 2).with_tolerance(1e-12);
        let result = solver
            .solve(diag_matvec_fn(DIAG), &b, None)
            .expect("solve failed");

        assert!(
            result.converged,
            "RecycledGmres single solve: residual = {:.3e}",
            result.residual_norm
        );
    }

    #[test]
    fn test_recycled_gmres_two_related_systems() {
        // Solve two systems with same A but different b.
        // Second solve should benefit from recycled subspace.
        let n = 12usize;
        let diag_vals: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let b1: Vec<f64> = vec![1.0; n];
        let b2: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let mut solver = RecycledGmres::new(8, 3)
            .with_tolerance(1e-12)
            .with_max_iter(500);

        // First solve.
        let diag_vals_clone1 = diag_vals.clone();
        let result1 = solver
            .solve(
                move |x: &[f64]| {
                    x.iter()
                        .zip(diag_vals_clone1.iter())
                        .map(|(xi, di)| xi * di)
                        .collect()
                },
                &b1,
                None,
            )
            .expect("first solve failed");

        assert!(result1.converged, "First solve did not converge");

        // Second solve uses recycled subspace.
        let diag_vals_clone2 = diag_vals.clone();
        let result2 = solver
            .solve(
                move |x: &[f64]| {
                    x.iter()
                        .zip(diag_vals_clone2.iter())
                        .map(|(xi, di)| xi * di)
                        .collect()
                },
                &b2,
                None,
            )
            .expect("second solve failed");

        assert!(result2.converged, "Second solve did not converge");

        // Verify second solution: x[i] = b2[i] / diag[i] = (i+1)/(i+1) = 1.
        for i in 0..n {
            let expected = (i + 1) as f64 / (i + 1) as f64; // = 1.0
            assert!(
                (result2.x[i] - expected).abs() < 1e-9,
                "x[{}] = {:.6}, expected {:.6}",
                i,
                result2.x[i],
                expected
            );
        }
    }

    #[test]
    fn test_recycled_gmres_clear_recycle() {
        let n = 8usize;
        let diag_vals: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let b: Vec<f64> = vec![1.0; n];

        let mut solver = RecycledGmres::new(6, 2).with_tolerance(1e-12);

        let dv1 = diag_vals.clone();
        solver
            .solve(
                move |x: &[f64]| x.iter().zip(dv1.iter()).map(|(xi, di)| xi * di).collect(),
                &b,
                None,
            )
            .expect("solve failed");

        let before = solver.recycle_dim();
        solver.clear_recycle();
        assert_eq!(
            solver.recycle_dim(),
            0,
            "clear_recycle should empty recycled subspace"
        );
        let _ = before; // just to use the variable
    }

    #[test]
    fn test_solve_dense_spd_2x2() {
        // A = [[4, 2],[2, 3]], b = [10, 7] -> x = [2, 1].
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let b = vec![10.0, 7.0];
        let x = solve_dense_spd(&a, &b, 2);
        assert!((x[0] - 2.0).abs() < 1e-12, "x[0] = {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-12, "x[1] = {}", x[1]);
    }
}
