//! Natural gradient descent using the Fisher Information Matrix (FIM).
//!
//! The natural gradient `F⁻¹ g` replaces the Euclidean gradient `g` with a
//! direction that accounts for the curvature of the statistical manifold of the
//! model's output distribution (Amari, 1998).
//!
//! This implementation estimates the FIM empirically from a mini-batch of
//! per-sample gradients and applies a damped conjugate-gradient solve
//! `(F̂ + λ I) d = g` to obtain the natural gradient step.
//!
//! # K-FAC approximation
//!
//! For large networks the full FIM is intractable.  The [`KFACLayer`] struct
//! stores the Kronecker factors `(A, G)` for a single linear layer and
//! provides an efficient preconditioned update without forming the full matrix.
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::optim::NaturalGradient;
//!
//! let mut params = vec![2.0_f64, -3.0];
//! let gradient = vec![1.0_f64, -1.0];
//!
//! // Two per-sample gradients (empirical Fisher).
//! let samples = vec![
//!     vec![0.8_f64, -0.6],
//!     vec![1.2_f64, -1.4],
//! ];
//!
//! let ng = NaturalGradient::new(0.1);
//! let prev_params = params.clone();
//! ng.step(&mut params, &gradient, &samples);
//!
//! // Parameters should have changed.
//! assert_ne!(params, prev_params);
//! ```

use crate::error::AutogradError;
use crate::optim::lbfgs::{dot, l2_norm};

/// Natural gradient descent optimizer.
///
/// At each step the empirical Fisher matrix is estimated from `fisher_samples`
/// (per-sample log-likelihood gradients) and the parameter update is
///
/// ```text
/// θ ← θ - lr · (F̂ + λI)⁻¹ g
/// ```
///
/// where the linear system is solved via conjugate gradients.
pub struct NaturalGradient {
    /// Learning rate (step size).
    pub learning_rate: f64,
    /// Tikhonov damping `λ` for numerical stability.
    pub damping: f64,
    /// Maximum CG iterations for the inner solve.
    pub cg_max_iter: usize,
    /// CG convergence tolerance.
    pub cg_tol: f64,
}

impl NaturalGradient {
    /// Create a `NaturalGradient` optimizer with the given learning rate.
    pub fn new(lr: f64) -> Self {
        Self {
            learning_rate: lr,
            damping: 1e-4,
            cg_max_iter: 50,
            cg_tol: 1e-8,
        }
    }

    /// Override the damping parameter.
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping;
        self
    }

    /// Override the CG iteration limit.
    pub fn with_cg_max_iter(mut self, n: usize) -> Self {
        self.cg_max_iter = n;
        self
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public interface
    // ─────────────────────────────────────────────────────────────────────────

    /// Perform one natural gradient update step.
    ///
    /// * `params` — current parameter vector (modified in-place).
    /// * `grad` — gradient of the loss w.r.t. `params`.
    /// * `fisher_samples` — per-sample gradients used to estimate the FIM.
    ///   Pass an empty slice to fall back to ordinary gradient descent.
    pub fn step(&self, params: &mut Vec<f64>, grad: &[f64], fisher_samples: &[Vec<f64>]) {
        let m = fisher_samples.len();

        if m == 0 {
            // Degenerate case: plain gradient descent.
            for (p, g) in params.iter_mut().zip(grad.iter()) {
                *p -= self.learning_rate * g;
            }
            return;
        }

        let n = params.len();
        let nat_grad = self.solve_fisher_cg(grad, fisher_samples, n, m);

        for (p, ng) in params.iter_mut().zip(nat_grad.iter()) {
            *p -= self.learning_rate * ng;
        }
    }

    /// Perform a full natural gradient optimization run.
    ///
    /// `grad_and_samples_fn(x)` returns `(loss, gradient, per_sample_gradients)`.
    pub fn minimize<F>(
        &self,
        grad_and_samples_fn: F,
        mut x: Vec<f64>,
        max_iter: usize,
        tol: f64,
    ) -> Result<NaturalGradientResult, AutogradError>
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>),
    {
        let mut loss_history = Vec::with_capacity(max_iter + 1);

        for iter in 0..max_iter {
            let (f, g, samples) = grad_and_samples_fn(&x);
            loss_history.push(f);

            let grad_norm = l2_norm(&g);
            if grad_norm < tol {
                return Ok(NaturalGradientResult {
                    x,
                    f,
                    grad_norm,
                    iterations: iter,
                    converged: true,
                    loss_history,
                });
            }

            self.step(&mut x, &g, &samples);
        }

        let (f, g, _) = grad_and_samples_fn(&x);
        let grad_norm = l2_norm(&g);
        loss_history.push(f);

        Ok(NaturalGradientResult {
            x,
            f,
            grad_norm,
            iterations: max_iter,
            converged: grad_norm < tol,
            loss_history,
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Matrix-free CG solve for `(F̂ + λI) d = g`.
    ///
    /// The empirical Fisher matrix-vector product is:
    /// `F̂ v = (1/m) Σ_i (gᵢᵀ v) gᵢ`
    fn solve_fisher_cg(
        &self,
        g: &[f64],
        samples: &[Vec<f64>],
        n: usize,
        m: usize,
    ) -> Vec<f64> {
        let m_f = m as f64;
        let damping = self.damping;

        // Matrix-free Fisher+damping matvec.
        let fv = |v: &[f64]| -> Vec<f64> {
            let mut result = vec![0.0_f64; n];
            for s in samples {
                let sv: f64 = dot(s, v);
                for j in 0..n {
                    result[j] += sv * s[j] / m_f;
                }
            }
            for j in 0..n {
                result[j] += damping * v[j];
            }
            result
        };

        // CG: solve (F + λI) x = g.
        let mut x = vec![0.0_f64; n];
        let mut r = g.to_vec(); // residual = g - (F+λI)*0 = g
        let mut p = r.clone();
        let mut rr: f64 = dot(&r, &r);

        for _ in 0..self.cg_max_iter {
            if rr < self.cg_tol * self.cg_tol {
                break;
            }
            let ap = fv(&p);
            let pap: f64 = dot(&p, &ap);
            if pap < 1e-20 {
                break;
            }
            let alpha = rr / pap;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let rr_new: f64 = dot(&r, &r);
            let beta = rr_new / rr.max(1e-20);
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rr = rr_new;
        }

        x
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// K-FAC layer approximation
// ─────────────────────────────────────────────────────────────────────────────

/// K-FAC curvature factors for a single fully-connected layer.
///
/// The FIM block for layer `l` is approximated as
/// `F_l ≈ A_l ⊗ G_l`
/// where `A_l = E[a aᵀ]` (input covariance) and `G_l = E[δ δᵀ]`
/// (pre-activation gradient covariance).  The inverse then factors as
/// `F_l⁻¹ ≈ A_l⁻¹ ⊗ G_l⁻¹`.
#[derive(Debug, Clone)]
pub struct KFACLayer {
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
    /// Running estimate of `A = E[a aᵀ]`, stored row-major, size `in_dim × in_dim`.
    pub a_factor: Vec<f64>,
    /// Running estimate of `G = E[δ δᵀ]`, stored row-major, size `out_dim × out_dim`.
    pub g_factor: Vec<f64>,
    /// Damping.
    pub damping: f64,
}

impl KFACLayer {
    /// Create a new K-FAC layer tracker.
    pub fn new(in_dim: usize, out_dim: usize, damping: f64) -> Self {
        Self {
            in_dim,
            out_dim,
            a_factor: vec![0.0_f64; in_dim * in_dim],
            g_factor: vec![0.0_f64; out_dim * out_dim],
            damping,
        }
    }

    /// Accumulate one mini-batch into the running covariance estimates.
    ///
    /// * `activations` — flattened `(batch × in_dim)` matrix (row-major).
    /// * `grad_outputs` — flattened `(batch × out_dim)` pre-activation gradient matrix.
    /// * `momentum` — weight for the new batch statistics; use `1.0` for a
    ///   simple running average (no smoothing), `0.0` to keep the old estimate.
    pub fn update(&mut self, activations: &[f64], grad_outputs: &[f64], batch: usize, momentum: f64) {
        let batch_f = batch as f64;
        let d = self.damping;

        // Update A factor: A = (1 - momentum) * A_old + momentum * A_new + d*I
        for i in 0..self.in_dim {
            for j in 0..self.in_dim {
                let mut aij = 0.0_f64;
                for b in 0..batch {
                    aij += activations[b * self.in_dim + i] * activations[b * self.in_dim + j];
                }
                aij /= batch_f;
                self.a_factor[i * self.in_dim + j] =
                    (1.0 - momentum) * self.a_factor[i * self.in_dim + j] + momentum * aij;
            }
            // Damping on diagonal.
            self.a_factor[i * self.in_dim + i] += d;
        }

        // Update G factor: G = (1 - momentum) * G_old + momentum * G_new + d*I
        for i in 0..self.out_dim {
            for j in 0..self.out_dim {
                let mut gij = 0.0_f64;
                for b in 0..batch {
                    gij +=
                        grad_outputs[b * self.out_dim + i] * grad_outputs[b * self.out_dim + j];
                }
                gij /= batch_f;
                self.g_factor[i * self.out_dim + j] =
                    (1.0 - momentum) * self.g_factor[i * self.out_dim + j] + momentum * gij;
            }
            self.g_factor[i * self.out_dim + i] += d;
        }
    }

    /// Apply the K-FAC preconditioner to a weight gradient matrix.
    ///
    /// The gradient `dW` has shape `(out_dim × in_dim)` stored row-major.
    /// Returns `G⁻¹ dW A⁻¹` (same shape) via Cholesky-based solves.
    pub fn precondition(&self, dw: &[f64]) -> Result<Vec<f64>, AutogradError> {
        let a_inv = cholesky_invert(&self.a_factor, self.in_dim)?;
        let g_inv = cholesky_invert(&self.g_factor, self.out_dim)?;

        // Result = G_inv * dW * A_inv
        // First: tmp = dW * A_inv
        let mut tmp = vec![0.0_f64; self.out_dim * self.in_dim];
        for i in 0..self.out_dim {
            for j in 0..self.in_dim {
                let mut v = 0.0_f64;
                for k in 0..self.in_dim {
                    v += dw[i * self.in_dim + k] * a_inv[k * self.in_dim + j];
                }
                tmp[i * self.in_dim + j] = v;
            }
        }

        // Then: result = G_inv * tmp
        let mut result = vec![0.0_f64; self.out_dim * self.in_dim];
        for i in 0..self.out_dim {
            for j in 0..self.in_dim {
                let mut v = 0.0_f64;
                for k in 0..self.out_dim {
                    v += g_inv[i * self.out_dim + k] * tmp[k * self.in_dim + j];
                }
                result[i * self.in_dim + j] = v;
            }
        }

        Ok(result)
    }
}

/// Result returned by [`NaturalGradient::minimize`].
#[derive(Debug, Clone)]
pub struct NaturalGradientResult {
    /// Final parameter vector.
    pub x: Vec<f64>,
    /// Final objective value.
    pub f: f64,
    /// L2 norm of the final gradient.
    pub grad_norm: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the gradient-norm tolerance was satisfied.
    pub converged: bool,
    /// Objective value at each iteration.
    pub loss_history: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Numeric helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Invert a symmetric positive-definite matrix `A` (size `n × n`, row-major)
/// using Cholesky decomposition followed by back-substitution.
fn cholesky_invert(a: &[f64], n: usize) -> Result<Vec<f64>, AutogradError> {
    // Cholesky: A = L Lᵀ
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(AutogradError::OperationError(
                        "Cholesky: matrix is not positive definite".to_string(),
                    ));
                }
                l[i * n + i] = sum.sqrt();
            } else {
                let lii = l[j * n + j];
                if lii.abs() < 1e-20 {
                    return Err(AutogradError::OperationError(
                        "Cholesky: near-zero diagonal".to_string(),
                    ));
                }
                l[i * n + j] = sum / lii;
            }
        }
    }

    // Invert L via forward substitution: solve L Y = I.
    let mut y = vec![0.0_f64; n * n];
    for col in 0..n {
        for row in 0..n {
            let mut val = if row == col { 1.0_f64 } else { 0.0_f64 };
            for k in 0..row {
                val -= l[row * n + k] * y[k * n + col];
            }
            let lrr = l[row * n + row];
            if lrr.abs() < 1e-20 {
                return Err(AutogradError::OperationError(
                    "Cholesky invert: singular L".to_string(),
                ));
            }
            y[row * n + col] = val / lrr;
        }
    }

    // A⁻¹ = (Lᵀ)⁻¹ Yᵀ  — via back substitution on Lᵀ x = y_col.
    let mut inv = vec![0.0_f64; n * n];
    for col in 0..n {
        // Back-substitute Lᵀ x = y_col.
        let mut x = vec![0.0_f64; n];
        for row in (0..n).rev() {
            let mut val = y[row * n + col];
            for k in (row + 1)..n {
                val -= l[k * n + row] * x[k];
            }
            let lrr = l[row * n + row];
            x[row] = val / lrr.max(1e-20);
        }
        for row in 0..n {
            inv[row * n + col] = x[row];
        }
    }

    Ok(inv)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_gradient_step_reduces_gradient_norm() {
        // Quadratic: f(x) = x^2 + y^2; g = [2x, 2y]; Fisher = I (isotropic).
        let params_start = vec![2.0_f64, -3.0];
        let gradient = vec![4.0_f64, -6.0]; // g = 2*params

        // Fisher samples: use the gradient itself as the single sample.
        let fisher_samples = vec![gradient.clone()];

        let ng = NaturalGradient::new(0.1).with_damping(1e-3);

        let mut params = params_start.clone();
        ng.step(&mut params, &gradient, &fisher_samples);

        let grad_norm_before = l2_norm(&gradient);
        // After a natural gradient step the params should have moved.
        let grad_after: Vec<f64> = params.iter().map(|p| 2.0 * p).collect();
        let grad_norm_after = l2_norm(&grad_after);

        assert!(
            grad_norm_after < grad_norm_before,
            "grad norm did not decrease: before={grad_norm_before} after={grad_norm_after}"
        );
    }

    #[test]
    fn test_natural_gradient_fallback_no_samples() {
        let mut params = vec![1.0_f64, 2.0];
        let grad = vec![0.5_f64, -1.0];
        let ng = NaturalGradient::new(0.1);
        ng.step(&mut params, &grad, &[]);
        // Plain gradient descent: params -= lr * grad
        assert!((params[0] - (1.0 - 0.1 * 0.5)).abs() < 1e-10);
        assert!((params[1] - (2.0 + 0.1 * 1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_invert_identity() {
        // A = I_3; A^{-1} should be I_3.
        let mut a = vec![0.0_f64; 9];
        a[0] = 1.0;
        a[4] = 1.0;
        a[8] = 1.0;
        let inv = cholesky_invert(&a, 3).expect("cholesky invert failed");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0_f64 } else { 0.0_f64 };
                assert!((inv[i * 3 + j] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_kfac_layer_update() {
        let mut layer = KFACLayer::new(2, 2, 1e-3);
        // Batch of 1: activations = [1, 0], grad_outputs = [0, 1].
        let acts = vec![1.0_f64, 0.0];
        let grads = vec![0.0_f64, 1.0];
        layer.update(&acts, &grads, 1, 1.0);
        // A factor should have a[0,0] = 1, others near 0.
        assert!((layer.a_factor[0] - (1.0 + 1e-3)).abs() < 1e-8);
    }
}
