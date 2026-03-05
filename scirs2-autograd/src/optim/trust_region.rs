//! Trust Region Methods using the Steihaug–Toint CG approach.
//!
//! At each iteration a quadratic model
//! `m(p) = f + gᵀ p + ½ pᵀ H p`
//! is minimised approximately subject to the trust-region constraint `‖p‖ ≤ Δ`
//! using the truncated conjugate-gradient method of Steihaug (1983).  The
//! trust-region radius `Δ` is updated based on the ratio of actual to
//! predicted reduction.
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::optim::TrustRegionOptimizer;
//!
//! // Minimise the sphere function  f(x) = Σ x_i^2.
//! let tr = TrustRegionOptimizer::new().with_max_iter(100).with_tolerance(1e-8);
//! let result = tr.minimize(
//!     |x: &[f64]| {
//!         let f: f64 = x.iter().map(|xi| xi * xi).sum();
//!         let g: Vec<f64> = x.iter().map(|xi| 2.0 * xi).collect();
//!         (f, g)
//!     },
//!     |_x: &[f64], v: &[f64]| v.iter().map(|vi| 2.0 * vi).collect(),
//!     vec![3.0, -2.0, 1.0],
//! ).expect("trust region error");
//! assert!(result.converged, "did not converge");
//! for xi in &result.x { assert!(xi.abs() < 1e-5, "xi={}", xi); }
//! ```

use crate::error::AutogradError;
use crate::optim::lbfgs::{dot, l2_norm, LBFGSResult};

/// Trust-region optimizer using Steihaug–Toint truncated CG.
pub struct TrustRegionOptimizer {
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// Gradient-norm convergence tolerance.
    pub tol: f64,
    /// Initial trust-region radius.
    pub initial_radius: f64,
    /// Maximum trust-region radius.
    pub max_radius: f64,
    /// Minimum acceptance ratio ρ; steps with ρ < η are rejected.
    pub eta: f64,
    /// Maximum Steihaug CG iterations per outer step.
    pub cg_max_iter: usize,
}

impl TrustRegionOptimizer {
    /// Create a `TrustRegionOptimizer` with sensible defaults.
    pub fn new() -> Self {
        Self {
            max_iter: 500,
            tol: 1e-6,
            initial_radius: 1.0,
            max_radius: 100.0,
            eta: 0.1,
            cg_max_iter: 0, // 0 means auto (= n)
        }
    }

    /// Override the maximum outer iteration count.
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Override the gradient-norm convergence tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Override the initial and maximum trust-region radii.
    pub fn with_radius(mut self, initial: f64, max: f64) -> Self {
        self.initial_radius = initial;
        self.max_radius = max;
        self
    }

    /// Override the acceptance threshold η.
    pub fn with_eta(mut self, eta: f64) -> Self {
        self.eta = eta;
        self
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public interface
    // ─────────────────────────────────────────────────────────────────────────

    /// Minimise `f(x)` using trust-region iterations.
    ///
    /// * `obj_grad(x)` — returns `(f_value, gradient)`.
    /// * `hvp(x, v)` — returns the Hessian–vector product `H(x) v`.
    pub fn minimize<F, H>(
        &self,
        obj_grad: F,
        hvp: H,
        x0: Vec<f64>,
    ) -> Result<LBFGSResult, AutogradError>
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>),
        H: Fn(&[f64], &[f64]) -> Vec<f64>,
    {
        let n = x0.len();
        let cg_max = if self.cg_max_iter == 0 { n + 1 } else { self.cg_max_iter };
        let mut x = x0;
        let (mut f, mut g) = obj_grad(&x);
        let mut delta = self.initial_radius;
        let mut loss_history = vec![f];

        for iter in 0..self.max_iter {
            let grad_norm = l2_norm(&g);
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

            // Steihaug CG: minimise m(p) s.t. ‖p‖ ≤ delta.
            let p = self.steihaug_cg(&g, |v| hvp(&x, v), delta, n, cg_max);
            let p_norm = l2_norm(&p);

            // Compute actual and predicted reduction.
            let x_new: Vec<f64> = x.iter().zip(p.iter()).map(|(xi, pi)| xi + pi).collect();
            let (f_new, _) = obj_grad(&x_new);

            let hp: Vec<f64> = hvp(&x, &p);
            let predicted = -(dot(&g, &p) + 0.5 * dot(&p, &hp));
            let actual = f - f_new;

            let rho = if predicted.abs() < 1e-14 {
                if actual > 0.0 { 1.0 } else { 0.0 }
            } else {
                actual / predicted
            };

            // Accept or reject the step.
            if rho > self.eta {
                x = x_new;
                let (new_f, new_g) = obj_grad(&x);
                f = new_f;
                g = new_g;
                loss_history.push(f);
            }

            // Update trust-region radius.
            if rho < 0.25 {
                delta *= 0.25;
                // Safeguard against collapse.
                delta = delta.max(1e-10 * l2_norm(&g));
            } else if rho > 0.75 && (p_norm - delta).abs() < 1e-8 * delta.max(1.0) {
                delta = (2.0 * delta).min(self.max_radius);
            }
        }

        let grad_norm = l2_norm(&g);
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

    /// Steihaug–Toint CG: minimise `½ pᵀ H p + gᵀ p` s.t. `‖p‖ ≤ delta`.
    ///
    /// Returns the step `p`.
    fn steihaug_cg<MV>(&self, g: &[f64], hvp: MV, delta: f64, n: usize, max_cg: usize) -> Vec<f64>
    where
        MV: Fn(&[f64]) -> Vec<f64>,
    {
        let mut z = vec![0.0_f64; n];
        let mut r = g.to_vec();
        let mut d: Vec<f64> = r.iter().map(|ri| -ri).collect();
        let mut rr: f64 = dot(&r, &r);
        let tol = 1e-10 * rr.sqrt().max(1.0);

        for _ in 0..max_cg {
            if rr.sqrt() < tol {
                break;
            }

            let hd = hvp(&d);
            let dhd: f64 = dot(&d, &hd);

            if dhd <= 0.0 {
                // Negative/zero curvature: find intersection with trust-region boundary.
                return boundary_step(&z, &d, delta, n);
            }

            let alpha = rr / dhd;
            let z_new: Vec<f64> = z.iter().zip(d.iter()).map(|(zi, di)| zi + alpha * di).collect();

            let z_norm = l2_norm(&z_new);
            if z_norm >= delta {
                // Step would leave the trust region: intersect.
                return boundary_step(&z, &d, delta, n);
            }

            let r_new: Vec<f64> =
                r.iter().zip(hd.iter()).map(|(ri, hdi)| ri + alpha * hdi).collect();
            let rr_new: f64 = dot(&r_new, &r_new);

            if rr_new.sqrt() < tol {
                return z_new;
            }

            let beta = rr_new / rr.max(1e-20);
            d = r_new.iter().zip(d.iter()).map(|(ri, di)| -ri + beta * di).collect();
            z = z_new;
            r = r_new;
            rr = rr_new;
        }

        z
    }
}

/// Find the point on the ray `z + τ d` that lies exactly on the sphere `‖·‖ = delta`.
///
/// Solves `‖z + τ d‖² = delta²` for the positive root τ.
fn boundary_step(z: &[f64], d: &[f64], delta: f64, n: usize) -> Vec<f64> {
    let a: f64 = dot(d, d);
    let b: f64 = 2.0 * dot(z, d);
    let c: f64 = dot(z, z) - delta * delta;

    // Quadratic formula; take the positive root.
    let disc = (b * b - 4.0 * a * c).max(0.0);
    let tau = if a.abs() < 1e-20 {
        0.0
    } else {
        (-b + disc.sqrt()) / (2.0 * a)
    };
    let tau = tau.max(0.0);

    (0..n).map(|i| z[i] + tau * d[i]).collect()
}

impl Default for TrustRegionOptimizer {
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

    /// Sphere function: f(x) = Σ x_i^2; minimum at 0.
    fn sphere(x: &[f64]) -> (f64, Vec<f64>) {
        let f: f64 = x.iter().map(|xi| xi * xi).sum();
        let g: Vec<f64> = x.iter().map(|xi| 2.0 * xi).collect();
        (f, g)
    }

    fn sphere_hvp(_x: &[f64], v: &[f64]) -> Vec<f64> {
        v.iter().map(|vi| 2.0 * vi).collect()
    }

    #[test]
    fn test_trust_region_sphere_3d() {
        let tr = TrustRegionOptimizer::new().with_max_iter(200).with_tolerance(1e-8);
        let result =
            tr.minimize(sphere, sphere_hvp, vec![3.0_f64, -2.0, 1.0]).expect("TR error");
        assert!(result.converged, "TR did not converge; grad_norm={}", result.grad_norm);
        for (i, xi) in result.x.iter().enumerate() {
            assert!(xi.abs() < 1e-5, "x[{i}] = {xi} expected ~0");
        }
    }

    #[test]
    fn test_trust_region_diagonal_quadratic() {
        // f(x) = 2x^2 + 5y^2; minimum at (0, 0).
        let obj = |x: &[f64]| -> (f64, Vec<f64>) {
            (2.0 * x[0] * x[0] + 5.0 * x[1] * x[1], vec![4.0 * x[0], 10.0 * x[1]])
        };
        let hvp = |_x: &[f64], v: &[f64]| -> Vec<f64> { vec![4.0 * v[0], 10.0 * v[1]] };

        let tr = TrustRegionOptimizer::new().with_max_iter(100).with_tolerance(1e-8);
        let result = tr.minimize(obj, hvp, vec![4.0_f64, -3.0]).expect("TR error");
        assert!(result.converged, "grad_norm={}", result.grad_norm);
        assert!(result.x[0].abs() < 1e-5, "x[0]={}", result.x[0]);
        assert!(result.x[1].abs() < 1e-5, "x[1]={}", result.x[1]);
    }

    #[test]
    fn test_boundary_step_on_sphere() {
        // z = 0, d = [1, 0], delta = 2.0 => tau = 2.0.
        let z = vec![0.0_f64, 0.0];
        let d = vec![1.0_f64, 0.0];
        let p = boundary_step(&z, &d, 2.0, 2);
        assert!((p[0] - 2.0).abs() < 1e-10, "p[0]={}", p[0]);
        assert!(p[1].abs() < 1e-10, "p[1]={}", p[1]);
    }

    #[test]
    fn test_trust_region_negative_curvature_does_not_panic() {
        // Concave objective: should step to boundary without panicking.
        let obj = |x: &[f64]| -> (f64, Vec<f64>) { (-x[0] * x[0] - x[1] * x[1], vec![-2.0 * x[0], -2.0 * x[1]]) };
        let hvp = |_x: &[f64], v: &[f64]| -> Vec<f64> { vec![-2.0 * v[0], -2.0 * v[1]] };
        let tr = TrustRegionOptimizer::new().with_max_iter(5);
        let _ = tr.minimize(obj, hvp, vec![0.5_f64, 0.5]);
    }
}
