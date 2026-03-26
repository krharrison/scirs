//! Flexible GMRES (FGMRES) solver for variable preconditioners.
//!
//! This submodule implements the FGMRES algorithm (Saad, 1993), which extends
//! GMRES to allow a different preconditioner at each inner iteration.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};
use crate::validation::{validate_iteration_parameters, validate_linear_system};

use super::{diagonal_condition_estimate, dot_vec, mv, vec_norm, SolveResult};

// ─────────────────────────────────────────────────────────────────────────────
// FGMRES — Flexible GMRES (variable preconditioner)
// ─────────────────────────────────────────────────────────────────────────────

/// Flexible GMRES solver for general linear systems with variable preconditioners.
///
/// FGMRES (Saad, 1993) extends GMRES to allow a different preconditioner at
/// each inner iteration. This is essential when the preconditioner itself is
/// iterative (e.g., inner GMRES, AMG V-cycle, or any method that does not
/// produce exactly the same result every time).
///
/// Unlike standard right-preconditioned GMRES where `M⁻¹` is fixed, FGMRES
/// stores the preconditioned vectors `z_j = M_j⁻¹ v_j` separately so that
/// the solution can be recovered even when the preconditioner changes.
///
/// # Algorithm
///
/// 1. Compute r = b - Ax₀, β = ‖r‖, v₁ = r/β
/// 2. For j = 1, ..., m (restart size):
///    a. z_j = M_j⁻¹ v_j  (preconditioner may vary)
///    b. w = A z_j
///    c. Modified Gram-Schmidt: orthogonalize w against v₁,...,v_j
///    d. Givens rotations on Hessenberg column
///    e. Check convergence
/// 3. Solve least-squares: y = H⁻¹ (β e₁)
/// 4. Update: x = x₀ + Z_m y  (using stored preconditioned vectors)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::krylov::Fgmres;
///
/// let a = array![[4.0_f64, 1.0], [1.0, 3.0]];
/// let b = array![5.0_f64, 7.0];
///
/// // Jacobi preconditioner
/// let diag = [4.0_f64, 3.0];
/// let precond = move |v: &scirs2_core::ndarray::Array1<f64>| -> scirs2_core::ndarray::Array1<f64> {
///     scirs2_core::ndarray::Array1::from_iter(v.iter().enumerate().map(|(i, &vi)| vi / diag[i]))
/// };
///
/// let result = Fgmres::new()
///     .tol(1e-12)
///     .max_iter(100)
///     .restart(10)
///     .solve_preconditioned(&a, &b.view(), &precond)
///     .expect("FGMRES failed");
/// assert!(result.converged);
/// ```
///
/// # References
///
/// - Saad, Y. (1993). *A flexible inner-outer preconditioned GMRES algorithm.*
///   SIAM J. Sci. Comput., 14(2), 461–469.
#[derive(Debug, Clone)]
pub struct Fgmres<F> {
    tol: F,
    max_iter: usize,
    restart: usize,
}

impl<F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + std::fmt::Debug + 'static> Default
    for Fgmres<F>
{
    fn default() -> Self {
        Self {
            tol: F::from(1e-10_f64).unwrap_or(F::epsilon()),
            max_iter: 200,
            restart: 30,
        }
    }
}

impl<F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + std::fmt::Debug + 'static>
    Fgmres<F>
{
    /// Create a new FGMRES builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set convergence tolerance (relative residual).
    pub fn tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set maximum number of outer restart iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the Krylov subspace dimension before restart.
    pub fn restart(mut self, restart: usize) -> Self {
        self.restart = restart;
        self
    }

    /// Solve A x = b without a preconditioner (equivalent to standard GMRES).
    pub fn solve(
        &self,
        a: &Array2<F>,
        b: &ArrayView1<F>,
    ) -> LinalgResult<SolveResult<F>> {
        // Identity preconditioner
        let identity = |v: &Array1<F>| v.clone();
        self.solve_preconditioned(a, b, &identity)
    }

    /// Solve A x = b with a (possibly variable) right preconditioner.
    ///
    /// The preconditioner `precond(v)` should approximate `M⁻¹ v`.
    /// It may be iterative or change between calls — this is the key
    /// advantage of FGMRES over standard preconditioned GMRES.
    pub fn solve_preconditioned(
        &self,
        a: &Array2<F>,
        b: &ArrayView1<F>,
        precond: &dyn Fn(&Array1<F>) -> Array1<F>,
    ) -> LinalgResult<SolveResult<F>> {
        validate_linear_system(&a.view(), b, "FGMRES")?;
        validate_iteration_parameters(self.max_iter, self.tol, "FGMRES")?;

        if self.restart == 0 {
            return Err(LinalgError::InvalidInputError(
                "FGMRES restart parameter must be positive".to_string(),
            ));
        }

        let n = a.nrows();
        let b_owned = b.to_owned();
        let b_norm = vec_norm(&b_owned);

        if b_norm <= F::epsilon() {
            return Ok(SolveResult {
                x: Array1::zeros(n),
                iterations: 0,
                residual_norm: F::zero(),
                converged: true,
                residual_history: vec![F::zero()],
                condition_estimate: None,
            });
        }

        let abs_tol = self.tol * b_norm;
        let mut x: Array1<F> = Array1::zeros(n);
        let mut total_iters = 0usize;
        let mut residual_norm = F::zero();
        let mut history = Vec::new();

        for _outer in 0..self.max_iter {
            // Compute residual r = b - Ax
            let ax = mv(a, &x);
            let r: Array1<F> = Array1::from_iter(
                b_owned.iter().zip(ax.iter()).map(|(&bi, &ai)| bi - ai),
            );
            let beta = vec_norm(&r);
            residual_norm = beta;
            history.push(residual_norm);

            if beta <= abs_tol {
                return Ok(SolveResult {
                    x,
                    iterations: total_iters,
                    residual_norm,
                    converged: true,
                    residual_history: history,
                    condition_estimate: Some(diagonal_condition_estimate(a)),
                });
            }

            let m = self.restart;

            // Arnoldi basis V (columns are v_1, ..., v_{m+1})
            let mut v_basis: Vec<Array1<F>> = Vec::with_capacity(m + 1);
            // Preconditioned vectors Z (z_j = M_j^{-1} v_j)
            let mut z_basis: Vec<Array1<F>> = Vec::with_capacity(m);

            // v_1 = r / beta
            v_basis.push(r.mapv(|vi| vi / beta));

            // Hessenberg matrix H (stored as column vectors of length m+1)
            let mut hess: Vec<Vec<F>> = Vec::with_capacity(m);

            // Givens rotations
            let mut cs: Vec<F> = Vec::with_capacity(m);
            let mut sn: Vec<F> = Vec::with_capacity(m);

            // Right-hand side g = beta * e_1
            let mut g: Vec<F> = Vec::with_capacity(m + 1);
            g.push(beta);

            let mut inner_iters = 0usize;

            for j in 0..m {
                total_iters += 1;

                // z_j = M_j^{-1} v_j  (the preconditioner may vary!)
                let z_j = precond(&v_basis[j]);
                z_basis.push(z_j.clone());

                // w = A z_j
                let mut w = mv(a, &z_j);

                // Modified Gram-Schmidt orthogonalization
                let mut h_col: Vec<F> = vec![F::zero(); j + 2];
                for (i, vi) in v_basis.iter().enumerate().take(j + 1) {
                    let h_ij = dot_vec(&w, vi);
                    h_col[i] = h_ij;
                    for k in 0..n {
                        w[k] = w[k] - h_ij * vi[k];
                    }
                }
                let h_j1j = vec_norm(&w);
                h_col[j + 1] = h_j1j;

                // Apply previous Givens rotations
                for i in 0..j {
                    let h0 = h_col[i];
                    let h1 = h_col[i + 1];
                    h_col[i] = cs[i] * h0 + sn[i] * h1;
                    h_col[i + 1] = -sn[i] * h0 + cs[i] * h1;
                }

                // Compute new Givens rotation
                let t = (h_col[j] * h_col[j] + h_col[j + 1] * h_col[j + 1]).sqrt();
                let (c, s) = if t < F::epsilon() {
                    (F::one(), F::zero())
                } else {
                    (h_col[j] / t, h_col[j + 1] / t)
                };
                cs.push(c);
                sn.push(s);

                // Apply new rotation
                h_col[j] = c * h_col[j] + s * h_col[j + 1];
                h_col[j + 1] = F::zero();

                let g_j = g[j];
                g.push(-sn[j] * g_j);
                g[j] = cs[j] * g_j;

                hess.push(h_col);

                residual_norm = g[j + 1].abs();
                inner_iters = j + 1;

                // Add next basis vector
                if h_j1j > F::epsilon() {
                    let v_next: Array1<F> = w.mapv(|vi| vi / h_j1j);
                    v_basis.push(v_next);
                } else {
                    // Lucky breakdown
                    break;
                }

                if residual_norm <= abs_tol {
                    break;
                }
            }

            // Back-substitution: solve Ry = g[0..inner_iters]
            let mut y: Vec<F> = vec![F::zero(); inner_iters];
            for i in (0..inner_iters).rev() {
                let mut sum = g[i];
                for k in (i + 1)..inner_iters {
                    sum = sum - hess[k][i] * y[k];
                }
                let diag = hess[i][i];
                if diag.abs() < F::epsilon() {
                    y[i] = F::zero();
                } else {
                    y[i] = sum / diag;
                }
            }

            // Update solution: x = x + Z_m * y
            // This is the key difference from GMRES: we use Z (preconditioned vectors)
            // instead of V (Krylov basis vectors)
            for (i, &yi) in y.iter().enumerate() {
                for k in 0..n {
                    x[k] = x[k] + yi * z_basis[i][k];
                }
            }

            history.push(residual_norm);

            if residual_norm <= abs_tol {
                return Ok(SolveResult {
                    x,
                    iterations: total_iters,
                    residual_norm,
                    converged: true,
                    residual_history: history,
                    condition_estimate: Some(diagonal_condition_estimate(a)),
                });
            }
        }

        Ok(SolveResult {
            x,
            iterations: total_iters,
            residual_norm,
            converged: false,
            residual_history: history,
            condition_estimate: Some(diagonal_condition_estimate(a)),
        })
    }
}

/// Solve A x = b with Flexible GMRES and return a rich [`SolveResult`].
///
/// Convenience wrapper around the [`Fgmres`] builder without preconditioner.
pub fn fgmres_solve<F>(
    a: &Array2<F>,
    b: &ArrayView1<F>,
    tol: F,
    max_iter: usize,
    restart: usize,
) -> LinalgResult<SolveResult<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + std::fmt::Debug + 'static,
{
    Fgmres::new()
        .tol(tol)
        .max_iter(max_iter)
        .restart(restart)
        .solve(a, b)
}
