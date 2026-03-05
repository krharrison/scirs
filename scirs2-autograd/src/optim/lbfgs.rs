//! L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) optimizer.
//!
//! L-BFGS is a quasi-Newton method that approximates the inverse Hessian using
//! the last `m` (s, y) pairs via the two-loop recursion (Nocedal, 1980).
//! Line search satisfies the strong Wolfe conditions.
//!
//! # Convergence
//!
//! Superlinear convergence on smooth, strongly-convex problems; practically
//! excellent on non-convex problems such as neural network training.
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::optim::LBFGS;
//!
//! // Minimise the Rosenbrock function  f(x,y) = (1-x)^2 + 100(y-x^2)^2
//! let lbfgs = LBFGS::new(10).with_max_iter(500).with_tolerance(1e-7);
//! let result = lbfgs.minimize(
//!     |x: &[f64]| {
//!         let f = (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
//!         let g0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2));
//!         let g1 = 200.0 * (x[1] - x[0].powi(2));
//!         (f, vec![g0, g1])
//!     },
//!     vec![-1.0, 1.0],
//! ).expect("L-BFGS failed");
//! assert!(result.converged, "did not converge");
//! assert!((result.x[0] - 1.0).abs() < 1e-4, "x[0]={}", result.x[0]);
//! assert!((result.x[1] - 1.0).abs() < 1e-4, "x[1]={}", result.x[1]);
//! ```

use crate::error::AutogradError;
use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// Internal state
// ─────────────────────────────────────────────────────────────────────────────

/// One element of the L-BFGS history.
#[derive(Debug, Clone)]
struct HistoryPair {
    /// s_k = x_{k+1} - x_k  (parameter difference)
    s: Vec<f64>,
    /// y_k = g_{k+1} - g_k  (gradient difference)
    y: Vec<f64>,
    /// rho_k = 1 / (y_k^T s_k)
    rho: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Public result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result returned by [`LBFGS::minimize`].
#[derive(Debug, Clone)]
pub struct LBFGSResult {
    /// Final parameter vector.
    pub x: Vec<f64>,
    /// Final objective value.
    pub f: f64,
    /// L2 norm of the final gradient.
    pub grad_norm: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the gradient-norm convergence criterion was satisfied.
    pub converged: bool,
    /// Objective value at each accepted iterate.
    pub loss_history: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Optimizer
// ─────────────────────────────────────────────────────────────────────────────

/// Limited-memory BFGS optimizer with strong-Wolfe line search.
pub struct LBFGS {
    /// Number of (s, y) pairs to retain in the curvature history.
    pub m: usize,
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// Gradient-norm convergence tolerance.
    pub tol_grad: f64,
    /// Function-change convergence tolerance.
    pub tol_change: f64,
    /// Maximum number of line-search iterations.
    pub line_search_max_iter: usize,
    /// Armijo (sufficient-decrease) constant.
    pub c1: f64,
    /// Curvature (Wolfe) constant.
    pub c2: f64,
}

impl LBFGS {
    /// Create an L-BFGS optimizer with history size `m` and default settings.
    pub fn new(m: usize) -> Self {
        Self {
            m,
            max_iter: 1000,
            tol_grad: 1e-7,
            tol_change: 1e-9,
            line_search_max_iter: 40,
            c1: 1e-4,
            c2: 0.9,
        }
    }

    /// Override the maximum iteration count.
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Override the gradient-norm convergence tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol_grad = tol;
        self
    }

    /// Override Armijo and Wolfe constants.
    pub fn with_wolfe(mut self, c1: f64, c2: f64) -> Self {
        self.c1 = c1;
        self.c2 = c2;
        self
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public interface
    // ─────────────────────────────────────────────────────────────────────────

    /// Minimise `f(x)` starting from `x0`.
    ///
    /// `grad_fn(x)` must return `(f_value, gradient_vector)`.
    pub fn minimize<F>(&self, grad_fn: F, x0: Vec<f64>) -> Result<LBFGSResult, AutogradError>
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>),
    {
        let n = x0.len();
        let mut x = x0;
        let (mut f, mut g) = grad_fn(&x);

        let mut history: VecDeque<HistoryPair> = VecDeque::with_capacity(self.m + 1);
        let mut loss_history = vec![f];

        for iter in 0..self.max_iter {
            let grad_norm = l2_norm(&g);
            if grad_norm < self.tol_grad {
                return Ok(LBFGSResult {
                    x,
                    f,
                    grad_norm,
                    iterations: iter,
                    converged: true,
                    loss_history,
                });
            }

            // Compute descent direction via two-loop recursion.
            let d = self.two_loop_recursion(&g, &history, n);

            // Ensure d is a descent direction; fall back to steepest descent.
            let dg0 = dot(&d, &g);
            let d = if dg0 >= 0.0 {
                // L-BFGS direction has wrong sign (degenerate curvature) — reset.
                history.clear();
                g.iter().map(|gi| -gi).collect::<Vec<f64>>()
            } else {
                d
            };

            // Wolfe line search.
            let (x_new, f_new, g_new) =
                self.wolfe_line_search(&x, &g, &d, f, dg0.min(-dot(&d, &g)), &grad_fn)?;

            // Build (s, y) pair and check curvature condition.
            let s: Vec<f64> = x_new.iter().zip(x.iter()).map(|(nx, xk)| nx - xk).collect();
            let y: Vec<f64> = g_new.iter().zip(g.iter()).map(|(ng, gk)| ng - gk).collect();
            let sy: f64 = dot(&s, &y);

            if sy > 1e-10 * dot(&y, &y).sqrt() * dot(&s, &s).sqrt() {
                if history.len() >= self.m {
                    history.pop_front();
                }
                history.push_back(HistoryPair { s, y, rho: 1.0 / sy });
            }

            let f_change = (f - f_new).abs();
            x = x_new;
            f = f_new;
            g = g_new;
            loss_history.push(f);

            if f_change < self.tol_change {
                break;
            }
        }

        let grad_norm = l2_norm(&g);
        Ok(LBFGSResult {
            x,
            f,
            grad_norm,
            iterations: self.max_iter,
            converged: grad_norm < self.tol_grad,
            loss_history,
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Two-loop L-BFGS recursion: returns the search direction `d = -H_k g_k`.
    fn two_loop_recursion(&self, g: &[f64], history: &VecDeque<HistoryPair>, n: usize) -> Vec<f64> {
        let mut q = g.to_vec();
        let m = history.len();
        let mut alphas = vec![0.0_f64; m];

        // Backward loop.
        for (i, pair) in history.iter().enumerate().rev() {
            let alpha = pair.rho * dot(&pair.s, &q);
            alphas[i] = alpha;
            for j in 0..n {
                q[j] -= alpha * pair.y[j];
            }
        }

        // Initial Hessian scaling H_0 = gamma * I.
        let gamma = history
            .back()
            .map(|pair| {
                let sy = dot(&pair.s, &pair.y);
                let yy = dot(&pair.y, &pair.y);
                if yy > 1e-20 { sy / yy } else { 1.0 }
            })
            .unwrap_or(1.0);

        let mut r: Vec<f64> = q.iter().map(|qi| gamma * qi).collect();

        // Forward loop.
        for (i, pair) in history.iter().enumerate() {
            let beta = pair.rho * dot(&pair.y, &r);
            for j in 0..n {
                r[j] += pair.s[j] * (alphas[i] - beta);
            }
        }

        // Negate to get descent direction.
        r.iter().map(|ri| -ri).collect()
    }

    /// Line search satisfying strong Wolfe conditions using bisection zoom.
    fn wolfe_line_search<F>(
        &self,
        x: &[f64],
        g: &[f64],
        d: &[f64],
        f0: f64,
        dg0: f64,
        grad_fn: &F,
    ) -> Result<(Vec<f64>, f64, Vec<f64>), AutogradError>
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>),
    {
        let n = x.len();
        let mut alpha = 1.0_f64;
        let mut alpha_lo = 0.0_f64;
        let mut alpha_hi = f64::INFINITY;
        let mut f_lo = f0;

        // Phase 1: bracket.
        for _i in 0..self.line_search_max_iter {
            let x_new: Vec<f64> = x.iter().zip(d.iter()).map(|(xi, di)| xi + alpha * di).collect();
            let (f_new, g_new) = grad_fn(&x_new);

            if f_new > f0 + self.c1 * alpha * dg0 || (f_new >= f_lo && _i > 0) {
                // Zoom between alpha_lo and alpha.
                return self.zoom(x, g, d, f0, dg0, alpha_lo, f_lo, alpha, f_new, n, grad_fn);
            }

            let dg_new = dot(d, &g_new);
            if dg_new.abs() <= -self.c2 * dg0 {
                return Ok((x_new, f_new, g_new));
            }

            if dg_new >= 0.0 {
                return self.zoom(x, g, d, f0, dg0, alpha, f_new, alpha_lo, f_lo, n, grad_fn);
            }

            f_lo = f_new;
            alpha_lo = alpha;
            alpha = if alpha_hi.is_infinite() { alpha * 2.0 } else { (alpha + alpha_hi) / 2.0 };
        }

        // Fallback: accept whatever we have.
        let x_new: Vec<f64> = x.iter().zip(d.iter()).map(|(xi, di)| xi + alpha * di).collect();
        let (f_new, g_new) = grad_fn(&x_new);
        Ok((x_new, f_new, g_new))
    }

    /// Bisection zoom phase of Wolfe line search.
    #[allow(clippy::too_many_arguments)]
    fn zoom<F>(
        &self,
        x: &[f64],
        _g: &[f64],
        d: &[f64],
        f0: f64,
        dg0: f64,
        mut alpha_lo: f64,
        mut f_lo: f64,
        mut alpha_hi: f64,
        _f_hi: f64,
        _n: usize,
        grad_fn: &F,
    ) -> Result<(Vec<f64>, f64, Vec<f64>), AutogradError>
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>),
    {
        for _ in 0..self.line_search_max_iter {
            let alpha = (alpha_lo + alpha_hi) / 2.0;
            let x_new: Vec<f64> = x.iter().zip(d.iter()).map(|(xi, di)| xi + alpha * di).collect();
            let (f_new, g_new) = grad_fn(&x_new);

            if f_new > f0 + self.c1 * alpha * dg0 || f_new >= f_lo {
                alpha_hi = alpha;
            } else {
                let dg_new = dot(d, &g_new);
                if dg_new.abs() <= -self.c2 * dg0 {
                    return Ok((x_new, f_new, g_new));
                }
                f_lo = f_new;
                alpha_lo = alpha;
                if dg_new * (alpha_hi - alpha_lo) >= 0.0 {
                    alpha_hi = alpha_lo;
                }
            }
        }

        // Accept the best found point.
        let alpha = (alpha_lo + alpha_hi) / 2.0;
        let x_new: Vec<f64> = x.iter().zip(d.iter()).map(|(xi, di)| xi + alpha * di).collect();
        let (f_new, g_new) = grad_fn(&x_new);
        Ok((x_new, f_new, g_new))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Small numeric helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
pub(crate) fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|vi| vi * vi).sum::<f64>().sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2; minimum at (1,1).
    fn rosenbrock(x: &[f64]) -> (f64, Vec<f64>) {
        let f = (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
        let g0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2));
        let g1 = 200.0 * (x[1] - x[0].powi(2));
        (f, vec![g0, g1])
    }

    #[test]
    fn test_lbfgs_rosenbrock() {
        let lbfgs = LBFGS::new(10).with_max_iter(2000).with_tolerance(1e-7);
        let result = lbfgs.minimize(rosenbrock, vec![-1.2, 1.0]).expect("L-BFGS should not error");
        assert!(result.converged, "L-BFGS did not converge; grad_norm={}", result.grad_norm);
        assert!(
            (result.x[0] - 1.0).abs() < 1e-4,
            "x[0] = {} expected ~1.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 1e-4,
            "x[1] = {} expected ~1.0",
            result.x[1]
        );
    }

    /// Simple quadratic: f(x) = x^T A x / 2; gradient = A x.
    #[test]
    fn test_lbfgs_quadratic() {
        // A = diag(1, 2, 3, 4)
        let a = vec![1.0_f64, 2.0, 3.0, 4.0];
        let grad_fn = move |x: &[f64]| -> (f64, Vec<f64>) {
            let f: f64 = x.iter().zip(a.iter()).map(|(xi, ai)| 0.5 * ai * xi * xi).sum();
            let g: Vec<f64> = x.iter().zip(a.iter()).map(|(xi, ai)| ai * xi).collect();
            (f, g)
        };
        let lbfgs = LBFGS::new(5).with_max_iter(200).with_tolerance(1e-8);
        let result = lbfgs.minimize(grad_fn, vec![4.0, -3.0, 2.0, -1.0]).expect("L-BFGS error");
        assert!(result.grad_norm < 1e-6, "grad_norm={}", result.grad_norm);
        for xi in &result.x {
            assert!(xi.abs() < 1e-5, "xi={}", xi);
        }
    }

    #[test]
    fn test_dot_helper() {
        assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-12);
    }
}
