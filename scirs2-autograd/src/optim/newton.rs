//! Newton-CG (Truncated Newton) optimizer.
//!
//! Approximately solves the Newton system `H d = -g` using the conjugate
//! gradient method, terminating early when negative curvature is encountered
//! (Steihaug stopping criterion, 1983) or when the CG residual is sufficiently
//! small (Eisenstat-Walker forcing sequence).
//!
//! The Hessian–vector product `H(x) v` is supplied by the caller and can be
//! computed via forward-over-reverse autodiff, finite differences, or any
//! other approach.
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::optim::NewtonCG;
//!
//! // Minimise  f(x) = x^T A x / 2,  A = diag(3, 7)
//! // Exact minimiser at (0, 0).
//! let a = vec![3.0_f64, 7.0];
//! let grad_fn = {
//!     let a = a.clone();
//!     move |x: &[f64]| -> (f64, Vec<f64>) {
//!         let f: f64 = x.iter().zip(a.iter()).map(|(xi, ai)| 0.5 * ai * xi * xi).sum();
//!         let g: Vec<f64> = x.iter().zip(a.iter()).map(|(xi, ai)| ai * xi).collect();
//!         (f, g)
//!     }
//! };
//! let hvp_fn = move |_x: &[f64], v: &[f64]| -> Vec<f64> {
//!     v.iter().zip(a.iter()).map(|(vi, ai)| ai * vi).collect()
//! };
//! let ncg = NewtonCG::new().with_max_iter(50);
//! let result = ncg.minimize(grad_fn, hvp_fn, vec![5.0, -3.0]).expect("Newton-CG error");
//! assert!(result.converged, "did not converge");
//! for xi in &result.x { assert!(xi.abs() < 1e-5, "xi={}", xi); }
//! ```

use crate::error::AutogradError;
use crate::optim::lbfgs::LBFGSResult;

/// Truncated Newton optimizer using conjugate gradients to solve the Newton system.
pub struct NewtonCG {
    /// Maximum number of outer (Newton) iterations.
    pub max_iter: usize,
    /// Maximum number of inner CG iterations per Newton step.
    pub cg_max_iter: usize,
    /// Gradient-norm convergence tolerance.
    pub tol: f64,
    /// Forcing sequence parameter (Eisenstat-Walker): inner CG tolerance
    /// is `min(0.5, eta * ‖g‖)`.
    pub eta: f64,
    /// Armijo constant for the backtracking line search.
    pub armijo_c1: f64,
}

impl NewtonCG {
    /// Create a NewtonCG optimizer with sensible defaults.
    pub fn new() -> Self {
        Self {
            max_iter: 200,
            cg_max_iter: 100,
            tol: 1e-5,
            eta: 0.1,
            armijo_c1: 1e-4,
        }
    }

    /// Override the maximum outer iteration count.
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Override gradient-norm convergence tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Override the CG forcing-sequence parameter.
    pub fn with_eta(mut self, eta: f64) -> Self {
        self.eta = eta;
        self
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public interface
    // ─────────────────────────────────────────────────────────────────────────

    /// Minimise `f(x)` using the Newton-CG method.
    ///
    /// * `grad_fn(x)` — returns `(f_value, gradient)`.
    /// * `hvp_fn(x, v)` — returns the Hessian–vector product `H(x) v`.
    pub fn minimize<F, H>(
        &self,
        grad_fn: F,
        hvp_fn: H,
        x0: Vec<f64>,
    ) -> Result<LBFGSResult, AutogradError>
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>),
        H: Fn(&[f64], &[f64]) -> Vec<f64>,
    {
        let mut x = x0;
        let (mut f, mut g) = grad_fn(&x);
        let mut loss_history = vec![f];

        for iter in 0..self.max_iter {
            let grad_norm = super::lbfgs::l2_norm(&g);
            if grad_norm < self.tol {
                return Ok(LBFGSResult {
                    x,
                    f,
                    grad_norm,
                    iterations: iter,
                    converged: true,
                    loss_history,
                });
            }

            // Inner CG tolerance via Eisenstat-Walker forcing sequence.
            let cg_tol = (self.eta * grad_norm).min(0.5).max(1e-10);

            // Solve H d = -g with CG; stop on negative curvature.
            let d = self.cg_solve(&g, |v| hvp_fn(&x, v), cg_tol);

            // Verify descent: dᵀ g must be negative.
            let dg = super::lbfgs::dot(&d, &g);
            let d = if dg >= 0.0 {
                // Negative curvature or CG degeneration — fall back to -g.
                g.iter().map(|gi| -gi).collect::<Vec<f64>>()
            } else {
                d
            };

            let dg = super::lbfgs::dot(&d, &g);

            // Backtracking line search (Armijo).
            let mut alpha = 1.0_f64;
            for _ in 0..30 {
                let x_new: Vec<f64> =
                    x.iter().zip(d.iter()).map(|(xi, di)| xi + alpha * di).collect();
                let (f_new, _) = grad_fn(&x_new);
                if f_new <= f + self.armijo_c1 * alpha * dg {
                    break;
                }
                alpha *= 0.5;
            }

            x = x.iter().zip(d.iter()).map(|(xi, di)| xi + alpha * di).collect();
            let (new_f, new_g) = grad_fn(&x);
            f = new_f;
            g = new_g;
            loss_history.push(f);
        }

        let grad_norm = super::lbfgs::l2_norm(&g);
        Ok(LBFGSResult {
            x,
            f,
            grad_norm,
            iterations: self.max_iter,
            converged: grad_norm < self.tol,
            loss_history,
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Solve `A x = b` via CG where `A = matvec`.
    ///
    /// Terminates when `‖r‖ < tol`, when negative curvature is detected
    /// (`pᵀ A p ≤ 0`), or when `cg_max_iter` is reached.
    fn cg_solve<MV>(&self, rhs: &[f64], matvec: MV, tol: f64) -> Vec<f64>
    where
        MV: Fn(&[f64]) -> Vec<f64>,
    {
        let n = rhs.len();
        let mut sol = vec![0.0_f64; n];
        // Solve for the descent direction: A sol = -rhs  (rhs is the gradient).
        let mut r: Vec<f64> = rhs.iter().map(|ri| -ri).collect();
        let mut p = r.clone();
        let mut rr: f64 = super::lbfgs::dot(&r, &r);

        for _ in 0..self.cg_max_iter {
            if rr.sqrt() < tol {
                break;
            }
            let ap = matvec(&p);
            let pap: f64 = super::lbfgs::dot(&p, &ap);

            // Negative curvature: the quadratic model is unbounded below along p.
            if pap <= 0.0 {
                // If we haven't moved yet, fall back to -gradient direction.
                if super::lbfgs::l2_norm(&sol) < 1e-14 {
                    return rhs.iter().map(|ri| -ri).collect();
                }
                break;
            }

            let alpha = rr / pap.max(1e-20);
            for i in 0..n {
                sol[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let rr_new: f64 = super::lbfgs::dot(&r, &r);
            let beta = rr_new / rr.max(1e-20);
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rr = rr_new;
        }

        sol
    }
}

impl Default for NewtonCG {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Diagonal quadratic: f(x) = Σ a_i x_i^2 / 2.  H = diag(a).
    #[test]
    fn test_newton_cg_diagonal_quadratic() {
        let a: Vec<f64> = vec![3.0, 7.0, 2.0, 5.0];
        let n = a.len();
        let a_grad = a.clone();
        let a_hvp = a.clone();

        let grad_fn = move |x: &[f64]| -> (f64, Vec<f64>) {
            let f: f64 = x.iter().zip(a_grad.iter()).map(|(xi, ai)| 0.5 * ai * xi * xi).sum();
            let g: Vec<f64> = x.iter().zip(a_grad.iter()).map(|(xi, ai)| ai * xi).collect();
            (f, g)
        };
        let hvp_fn = move |_x: &[f64], v: &[f64]| -> Vec<f64> {
            v.iter().zip(a_hvp.iter()).map(|(vi, ai)| ai * vi).collect()
        };

        let x0 = vec![5.0_f64; n];
        let ncg = NewtonCG::new().with_max_iter(50).with_tolerance(1e-8);
        let result = ncg.minimize(grad_fn, hvp_fn, x0).expect("Newton-CG should not error");

        assert!(result.converged, "Newton-CG did not converge; grad_norm={}", result.grad_norm);
        for (i, xi) in result.x.iter().enumerate() {
            assert!(xi.abs() < 1e-5, "x[{i}] = {} expected ~0", xi);
        }
    }

    /// Verify that Newton-CG handles negative curvature gracefully (does not panic).
    #[test]
    fn test_newton_cg_negative_curvature() {
        // Concave function: f(x) = -x^2.  Gradient = -2x, H = -2 (negative definite).
        let grad_fn = |x: &[f64]| -> (f64, Vec<f64>) { (-x[0] * x[0], vec![-2.0 * x[0]]) };
        let hvp_fn = |_x: &[f64], v: &[f64]| -> Vec<f64> { vec![-2.0 * v[0]] };

        let ncg = NewtonCG::new().with_max_iter(5);
        // Should not panic; convergence is not expected for a concave function.
        let _ = ncg.minimize(grad_fn, hvp_fn, vec![1.0]);
    }
}
