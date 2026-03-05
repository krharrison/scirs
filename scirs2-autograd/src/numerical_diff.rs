//! Numerical differentiation utilities
//!
//! This module provides finite-difference, Richardson extrapolation, and
//! complex-step methods for computing derivatives numerically.  All methods
//! operate on plain Rust slices and closures, making them useful for:
//!
//! - Gradient-checking analytic autodiff results against numerical estimates
//! - Differentiating black-box functions that have no analytic gradient
//! - Building higher-order derivative approximations via Richardson extrapolation
//!
//! # Methods overview
//!
//! | Method | Order | Notes |
//! |--------|-------|-------|
//! | Forward difference | O(h) | Cheapest; 1 extra eval per variable |
//! | Backward difference | O(h) | Identical cost; useful for boundary checks |
//! | Central difference | O(h²) | 2 extra evals; recommended default |
//! | Five-point stencil | O(h⁴) | 4 extra evals; high accuracy |
//! | Richardson extrapolation | O(h^(2n)) | n steps; near-machine-precision |
//! | Complex-step | ~ε_mach | Requires analytic extension to ℂ |
//!
//! # Examples
//!
//! ```rust
//! use scirs2_autograd::numerical_diff::{
//!     numerical_gradient, FiniteDiffMethod, check_gradient,
//! };
//!
//! let f = |xs: &[f64]| xs[0].powi(3) + xs[1].powi(2);
//! let grad_f = |xs: &[f64]| vec![3.0 * xs[0].powi(2), 2.0 * xs[1]];
//!
//! let x = vec![2.0_f64, 3.0];
//! let g = numerical_gradient(&f, &x, FiniteDiffMethod::Central, 1e-6);
//! assert!((g[0] - 12.0).abs() < 1e-5);
//! assert!((g[1] -  6.0).abs() < 1e-5);
//!
//! assert!(check_gradient(&f, &grad_f, &x, 1e-6, 1e-4, 1e-4));
//! ```

/// Finite-difference method selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiniteDiffMethod {
    /// `(f(x+h) - f(x)) / h`  — first-order accuracy O(h)
    Forward,
    /// `(f(x) - f(x-h)) / h`  — first-order accuracy O(h)
    Backward,
    /// `(f(x+h) - f(x-h)) / (2h)` — second-order accuracy O(h²)
    Central,
    /// `(-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)` — fourth-order O(h⁴)
    FivePoint,
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar-valued gradient
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the gradient of a scalar-valued function `f : ℝⁿ → ℝ` at `x`.
///
/// # Arguments
/// * `f` - scalar-valued function
/// * `x` - point at which to evaluate the gradient
/// * `method` - finite-difference stencil to use
/// * `eps` - step size (h)
///
/// # Returns
/// Gradient vector `∂f/∂xᵢ` for each `i ∈ 0..n`.
///
/// # Example
/// ```rust
/// use scirs2_autograd::numerical_diff::{numerical_gradient, FiniteDiffMethod};
///
/// let f = |xs: &[f64]| xs[0] * xs[0] + xs[1] * xs[1];
/// let g = numerical_gradient(&f, &[3.0, 4.0], FiniteDiffMethod::Central, 1e-6);
/// assert!((g[0] - 6.0).abs() < 1e-4);
/// assert!((g[1] - 8.0).abs() < 1e-4);
/// ```
pub fn numerical_gradient(
    f: &impl Fn(&[f64]) -> f64,
    x: &[f64],
    method: FiniteDiffMethod,
    eps: f64,
) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0_f64; n];
    let mut xp = x.to_vec();

    for i in 0..n {
        let orig = xp[i];
        grad[i] = match method {
            FiniteDiffMethod::Forward => {
                xp[i] = orig + eps;
                let fp = f(&xp);
                xp[i] = orig;
                let f0 = f(&xp);
                (fp - f0) / eps
            }
            FiniteDiffMethod::Backward => {
                xp[i] = orig;
                let f0 = f(&xp);
                xp[i] = orig - eps;
                let fm = f(&xp);
                xp[i] = orig;
                (f0 - fm) / eps
            }
            FiniteDiffMethod::Central => {
                xp[i] = orig + eps;
                let fp = f(&xp);
                xp[i] = orig - eps;
                let fm = f(&xp);
                xp[i] = orig;
                (fp - fm) / (2.0 * eps)
            }
            FiniteDiffMethod::FivePoint => {
                xp[i] = orig + 2.0 * eps;
                let fp2 = f(&xp);
                xp[i] = orig + eps;
                let fp1 = f(&xp);
                xp[i] = orig - eps;
                let fm1 = f(&xp);
                xp[i] = orig - 2.0 * eps;
                let fm2 = f(&xp);
                xp[i] = orig;
                (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * eps)
            }
        };
        xp[i] = orig; // restore (belt-and-suspenders)
    }
    grad
}

// ─────────────────────────────────────────────────────────────────────────────
// Jacobian of vector-valued function
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Jacobian matrix of `f : ℝⁿ → ℝᵐ` at `x`.
///
/// The Jacobian `J ∈ ℝ^{m×n}` is laid out row-major: `J[i][j] = ∂fᵢ/∂xⱼ`.
///
/// # Arguments
/// * `f` - vector-valued function returning a `Vec<f64>` of length `m`
/// * `x` - point at which to evaluate the Jacobian (length `n`)
/// * `method` - finite-difference stencil
/// * `eps` - step size
///
/// # Returns
/// `Vec<Vec<f64>>` of shape `(m, n)`.
///
/// # Example
/// ```rust
/// use scirs2_autograd::numerical_diff::{numerical_jacobian, FiniteDiffMethod};
///
/// // f(x) = [x0 + x1, x0 * x1]  =>  J = [[1, 1], [x1, x0]]
/// let f = |xs: &[f64]| vec![xs[0] + xs[1], xs[0] * xs[1]];
/// let j = numerical_jacobian(&f, &[2.0, 3.0], FiniteDiffMethod::Central, 1e-6);
/// assert!((j[0][0] - 1.0).abs() < 1e-4); // ∂f0/∂x0
/// assert!((j[0][1] - 1.0).abs() < 1e-4); // ∂f0/∂x1
/// assert!((j[1][0] - 3.0).abs() < 1e-4); // ∂f1/∂x0 = x1
/// assert!((j[1][1] - 2.0).abs() < 1e-4); // ∂f1/∂x1 = x0
/// ```
pub fn numerical_jacobian(
    f: &impl Fn(&[f64]) -> Vec<f64>,
    x: &[f64],
    method: FiniteDiffMethod,
    eps: f64,
) -> Vec<Vec<f64>> {
    let n = x.len();
    // Determine output dimension by one evaluation at the current point.
    let f0 = f(x);
    let m = f0.len();

    // J[i][j] = ∂fᵢ/∂xⱼ — initialise from f0 columns to reuse f0 below.
    let mut jac = vec![vec![0.0_f64; n]; m];
    let mut xp = x.to_vec();

    for j in 0..n {
        let orig = xp[j];

        match method {
            FiniteDiffMethod::Forward => {
                xp[j] = orig + eps;
                let fp = f(&xp);
                xp[j] = orig;
                for i in 0..m {
                    jac[i][j] = (fp[i] - f0[i]) / eps;
                }
            }
            FiniteDiffMethod::Backward => {
                xp[j] = orig - eps;
                let fm = f(&xp);
                xp[j] = orig;
                for i in 0..m {
                    jac[i][j] = (f0[i] - fm[i]) / eps;
                }
            }
            FiniteDiffMethod::Central => {
                xp[j] = orig + eps;
                let fp = f(&xp);
                xp[j] = orig - eps;
                let fm = f(&xp);
                xp[j] = orig;
                for i in 0..m {
                    jac[i][j] = (fp[i] - fm[i]) / (2.0 * eps);
                }
            }
            FiniteDiffMethod::FivePoint => {
                xp[j] = orig + 2.0 * eps;
                let fp2 = f(&xp);
                xp[j] = orig + eps;
                let fp1 = f(&xp);
                xp[j] = orig - eps;
                let fm1 = f(&xp);
                xp[j] = orig - 2.0 * eps;
                let fm2 = f(&xp);
                xp[j] = orig;
                for i in 0..m {
                    jac[i][j] =
                        (-fp2[i] + 8.0 * fp1[i] - 8.0 * fm1[i] + fm2[i]) / (12.0 * eps);
                }
            }
        }
        xp[j] = orig; // restore
    }
    jac
}

// ─────────────────────────────────────────────────────────────────────────────
// Hessian
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Hessian matrix of a scalar-valued function `f : ℝⁿ → ℝ` at `x`.
///
/// Uses the standard second-order central-difference formula:
///
/// ```text
/// H[i,j] = (f(x + hᵢ + hⱼ) - f(x + hᵢ - hⱼ) - f(x - hᵢ + hⱼ) + f(x - hᵢ - hⱼ)) / (4h²)
/// ```
///
/// For diagonal entries a simpler formula is used:
/// ```text
/// H[i,i] = (f(x + hᵢ) - 2f(x) + f(x - hᵢ)) / h²
/// ```
///
/// # Example
/// ```rust
/// use scirs2_autograd::numerical_diff::numerical_hessian;
///
/// // f(x) = x0² + x0*x1 + 2*x1²
/// // H = [[2, 1], [1, 4]]  (constant Hessian)
/// let f = |xs: &[f64]| xs[0].powi(2) + xs[0]*xs[1] + 2.0*xs[1].powi(2);
/// let h = numerical_hessian(&f, &[1.0, 1.0], 1e-4);
/// assert!((h[0][0] - 2.0).abs() < 1e-3);
/// assert!((h[0][1] - 1.0).abs() < 1e-3);
/// assert!((h[1][0] - 1.0).abs() < 1e-3);
/// assert!((h[1][1] - 4.0).abs() < 1e-3);
/// ```
pub fn numerical_hessian(f: &impl Fn(&[f64]) -> f64, x: &[f64], eps: f64) -> Vec<Vec<f64>> {
    let n = x.len();
    let f0 = f(x);
    let mut hess = vec![vec![0.0_f64; n]; n];
    let mut xp = x.to_vec();

    for i in 0..n {
        // Diagonal: (f(x+eᵢ) - 2f(x) + f(x-eᵢ)) / h²
        let orig_i = xp[i];
        xp[i] = orig_i + eps;
        let fp = f(&xp);
        xp[i] = orig_i - eps;
        let fm = f(&xp);
        xp[i] = orig_i;
        hess[i][i] = (fp - 2.0 * f0 + fm) / (eps * eps);

        // Off-diagonal (upper triangle, symmetrize)
        for j in (i + 1)..n {
            let orig_j = xp[j];

            xp[i] = orig_i + eps;
            xp[j] = orig_j + eps;
            let fpp = f(&xp);

            xp[i] = orig_i + eps;
            xp[j] = orig_j - eps;
            let fpm = f(&xp);

            xp[i] = orig_i - eps;
            xp[j] = orig_j + eps;
            let fmp = f(&xp);

            xp[i] = orig_i - eps;
            xp[j] = orig_j - eps;
            let fmm = f(&xp);

            xp[i] = orig_i;
            xp[j] = orig_j;

            let hij = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps);
            hess[i][j] = hij;
            hess[j][i] = hij; // symmetry
        }
    }
    hess
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient checker
// ─────────────────────────────────────────────────────────────────────────────

/// Check an analytic gradient against a numerical central-difference estimate.
///
/// Returns `true` if every component satisfies the relative+absolute tolerance:
/// `|analytic - numeric| <= atol + rtol * max(|analytic|, |numeric|)`.
///
/// # Arguments
/// * `f` - scalar-valued function
/// * `grad` - analytic gradient function
/// * `x` - evaluation point
/// * `eps` - step size for numerical estimate
/// * `rtol` - relative tolerance
/// * `atol` - absolute tolerance
///
/// # Example
/// ```rust
/// use scirs2_autograd::numerical_diff::{check_gradient, FiniteDiffMethod};
///
/// let f = |xs: &[f64]| xs[0].sin() + xs[1].cos();
/// let grad_f = |xs: &[f64]| vec![xs[0].cos(), -xs[1].sin()];
///
/// assert!(check_gradient(&f, &grad_f, &[0.5, 1.2], 1e-6, 1e-4, 1e-6));
/// ```
pub fn check_gradient(
    f: &impl Fn(&[f64]) -> f64,
    grad: &impl Fn(&[f64]) -> Vec<f64>,
    x: &[f64],
    eps: f64,
    rtol: f64,
    atol: f64,
) -> bool {
    let analytic = grad(x);
    let numeric = numerical_gradient(f, x, FiniteDiffMethod::Central, eps);

    if analytic.len() != numeric.len() {
        return false;
    }

    analytic.iter().zip(numeric.iter()).all(|(&a, &n)| {
        let diff = (a - n).abs();
        let scale = a.abs().max(n.abs());
        diff <= atol + rtol * scale
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Richardson extrapolation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the gradient of `f` using Richardson extrapolation for improved accuracy.
///
/// Starting from step size `eps0`, each successive column of the Romberg table
/// halves the step and the extrapolated value achieves order O(h^(2n)).
///
/// # Arguments
/// * `f` - scalar-valued function
/// * `x` - evaluation point
/// * `n_steps` - number of Richardson levels (typically 4–6)
/// * `eps0` - initial step size (will be successively halved)
///
/// # Example
/// ```rust
/// use scirs2_autograd::numerical_diff::richardson_gradient;
///
/// // sin(x): analytic grad = cos(x)
/// let f = |xs: &[f64]| xs[0].sin();
/// let g = richardson_gradient(&f, &[1.0], 5, 0.1);
/// assert!((g[0] - 1.0_f64.cos()).abs() < 1e-12, "g[0]={}", g[0]);
/// ```
pub fn richardson_gradient(
    f: &impl Fn(&[f64]) -> f64,
    x: &[f64],
    n_steps: usize,
    eps0: f64,
) -> Vec<f64> {
    let n_steps = n_steps.max(1);
    let n = x.len();
    let mut grad = vec![0.0_f64; n];
    let mut xp = x.to_vec();

    for k in 0..n {
        let orig = xp[k];
        // Build Romberg table column by column.
        // D[i] holds the current-level estimate with step eps0 / 2^i.
        let mut d = Vec::with_capacity(n_steps);

        let mut h = eps0;
        // First column: central differences with decreasing h
        for _ in 0..n_steps {
            xp[k] = orig + h;
            let fp = f(&xp);
            xp[k] = orig - h;
            let fm = f(&xp);
            xp[k] = orig;
            d.push((fp - fm) / (2.0 * h));
            h *= 0.5;
        }

        // Richardson extrapolation: combine adjacent estimates.
        // For central difference (order 2): factor = 4^p / (4^p - 1)
        let mut power = 4.0_f64; // 4 for central difference (p=2)
        for _level in 1..n_steps {
            let len = d.len() - 1;
            for i in 0..len {
                d[i] = (power * d[i + 1] - d[i]) / (power - 1.0);
            }
            d.pop();
            power *= 4.0; // next level: 4^(level+1)
        }

        grad[k] = d.first().copied().unwrap_or(0.0);
        xp[k] = orig; // restore
    }
    grad
}

// ─────────────────────────────────────────────────────────────────────────────
// Complex-step differentiation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the gradient of `f` using the complex-step method.
///
/// For an analytic function extended to the complex plane the identity
/// `Im(f(x + ih)) / h ≈ f'(x)` holds to machine precision because there is
/// no cancellation error (unlike finite differences).
///
/// The closure receives `(re_inputs, im_inputs)` — real and imaginary parts of
/// each input — and must return `(re_out, im_out)`.  For a real-valued function
/// the imaginary part propagates linearly through every elementary operation.
///
/// # Arguments
/// * `f_complex` - complex extension of `f`; signature `(re: &[f64], im: &[f64]) -> (f64, f64)`
/// * `x_real` - real evaluation point
/// * `eps` - imaginary perturbation (typically 1e-20 to 1e-100)
///
/// # Example
/// ```rust
/// use scirs2_autograd::numerical_diff::complex_step_gradient;
///
/// // f(x) = x[0]^3 + x[1]^2  =>  df/dx0 = 3*x0^2, df/dx1 = 2*x1
/// // Complex extension: (re + i*im)^n  ≈ re^n + i*n*re^(n-1)*im  (first order)
/// let f_cx = |re: &[f64], im: &[f64]| {
///     // re part: re0^3 + re1^2
///     let re_out = re[0].powi(3) + re[1].powi(2);
///     // im part: 3*re0^2*im0 + 2*re1*im1
///     let im_out = 3.0 * re[0].powi(2) * im[0] + 2.0 * re[1] * im[1];
///     (re_out, im_out)
/// };
///
/// let x = vec![2.0_f64, 3.0];
/// let g = complex_step_gradient(&f_cx, &x, 1e-20);
/// assert!((g[0] - 12.0).abs() < 1e-10, "df/dx0={}", g[0]);
/// assert!((g[1] -  6.0).abs() < 1e-10, "df/dx1={}", g[1]);
/// ```
pub fn complex_step_gradient(
    f_complex: &impl Fn(&[f64], &[f64]) -> (f64, f64),
    x_real: &[f64],
    eps: f64,
) -> Vec<f64> {
    let n = x_real.len();
    let mut grad = vec![0.0_f64; n];
    // im_inputs: all-zero except the perturbed component
    let mut im_inputs = vec![0.0_f64; n];

    for k in 0..n {
        im_inputs[k] = eps;
        let (_, im_out) = f_complex(x_real, &im_inputs);
        grad[k] = im_out / eps;
        im_inputs[k] = 0.0; // restore
    }
    grad
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── numerical_gradient ────────────────────────────────────────────────

    #[test]
    fn test_numerical_gradient_forward_quadratic() {
        let f = |xs: &[f64]| xs[0] * xs[0] + xs[1] * xs[1];
        let g = numerical_gradient(&f, &[3.0, 4.0], FiniteDiffMethod::Forward, 1e-6);
        assert!((g[0] - 6.0).abs() < 1e-3, "g[0]={}", g[0]);
        assert!((g[1] - 8.0).abs() < 1e-3, "g[1]={}", g[1]);
    }

    #[test]
    fn test_numerical_gradient_backward_quadratic() {
        let f = |xs: &[f64]| xs[0] * xs[0] + xs[1] * xs[1];
        let g = numerical_gradient(&f, &[3.0, 4.0], FiniteDiffMethod::Backward, 1e-6);
        assert!((g[0] - 6.0).abs() < 1e-3, "g[0]={}", g[0]);
        assert!((g[1] - 8.0).abs() < 1e-3, "g[1]={}", g[1]);
    }

    #[test]
    fn test_numerical_gradient_central_quadratic() {
        let f = |xs: &[f64]| xs[0] * xs[0] + xs[1] * xs[1];
        let g = numerical_gradient(&f, &[3.0, 4.0], FiniteDiffMethod::Central, 1e-6);
        assert!((g[0] - 6.0).abs() < 1e-5, "g[0]={}", g[0]);
        assert!((g[1] - 8.0).abs() < 1e-5, "g[1]={}", g[1]);
    }

    #[test]
    fn test_numerical_gradient_five_point_trig() {
        let f = |xs: &[f64]| xs[0].sin();
        // df/dx0 = cos(x0) at x0 = 1.0
        let g = numerical_gradient(&f, &[1.0], FiniteDiffMethod::FivePoint, 1e-5);
        assert!((g[0] - 1.0_f64.cos()).abs() < 1e-9, "g[0]={}", g[0]);
    }

    // ── numerical_jacobian ────────────────────────────────────────────────

    #[test]
    fn test_numerical_jacobian_linear() {
        // f(x) = [x0 + x1, 2*x0 - x1]
        // J = [[1, 1], [2, -1]]
        let f = |xs: &[f64]| vec![xs[0] + xs[1], 2.0 * xs[0] - xs[1]];
        let j = numerical_jacobian(&f, &[1.0, 2.0], FiniteDiffMethod::Central, 1e-6);
        assert!((j[0][0] - 1.0).abs() < 1e-5);
        assert!((j[0][1] - 1.0).abs() < 1e-5);
        assert!((j[1][0] - 2.0).abs() < 1e-5);
        assert!((j[1][1] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_numerical_jacobian_nonlinear() {
        // f(x) = [x0 + x1, x0 * x1]  =>  J at (2,3) = [[1,1],[3,2]]
        let f = |xs: &[f64]| vec![xs[0] + xs[1], xs[0] * xs[1]];
        let j = numerical_jacobian(&f, &[2.0, 3.0], FiniteDiffMethod::Central, 1e-6);
        assert!((j[1][0] - 3.0).abs() < 1e-4, "j[1][0]={}", j[1][0]);
        assert!((j[1][1] - 2.0).abs() < 1e-4, "j[1][1]={}", j[1][1]);
    }

    // ── numerical_hessian ─────────────────────────────────────────────────

    #[test]
    fn test_numerical_hessian_quadratic() {
        // f(x) = x0^2 + x0*x1 + 2*x1^2  =>  H = [[2,1],[1,4]]
        let f = |xs: &[f64]| xs[0].powi(2) + xs[0] * xs[1] + 2.0 * xs[1].powi(2);
        let h = numerical_hessian(&f, &[1.0, 1.0], 1e-4);
        assert!((h[0][0] - 2.0).abs() < 1e-3, "H[0,0]={}", h[0][0]);
        assert!((h[0][1] - 1.0).abs() < 1e-3, "H[0,1]={}", h[0][1]);
        assert!((h[1][0] - 1.0).abs() < 1e-3, "H[1,0]={}", h[1][0]);
        assert!((h[1][1] - 4.0).abs() < 1e-3, "H[1,1]={}", h[1][1]);
    }

    #[test]
    fn test_numerical_hessian_symmetric() {
        let f = |xs: &[f64]| xs[0].powi(3) * xs[1] + xs[1].powi(3);
        let h = numerical_hessian(&f, &[1.0, 2.0], 1e-4);
        assert!(
            (h[0][1] - h[1][0]).abs() < 1e-3,
            "H not symmetric: {} vs {}",
            h[0][1],
            h[1][0]
        );
    }

    // ── check_gradient ────────────────────────────────────────────────────

    #[test]
    fn test_check_gradient_correct() {
        let f = |xs: &[f64]| xs[0].sin() + xs[1].cos();
        let grad_f = |xs: &[f64]| vec![xs[0].cos(), -xs[1].sin()];
        assert!(check_gradient(&f, &grad_f, &[0.5, 1.2], 1e-6, 1e-4, 1e-6));
    }

    #[test]
    fn test_check_gradient_wrong_fails() {
        let f = |xs: &[f64]| xs[0].sin();
        let correct_grad = |xs: &[f64]| vec![xs[0].cos()];
        let wrong_grad = |xs: &[f64]| vec![xs[0].sin()]; // deliberately wrong
        // Correct gradient should pass
        let correct = check_gradient(&f, &correct_grad, &[0.5], 1e-6, 1e-3, 1e-6);
        assert!(correct, "correct gradient should pass check");
        // Wrong gradient should fail
        let wrong = check_gradient(&f, &wrong_grad, &[0.5], 1e-6, 1e-4, 1e-6);
        assert!(!wrong, "wrong gradient should fail check");
    }

    // ── richardson_gradient ───────────────────────────────────────────────

    #[test]
    fn test_richardson_gradient_sin() {
        let f = |xs: &[f64]| xs[0].sin();
        let g = richardson_gradient(&f, &[1.0], 5, 0.1);
        let expected = 1.0_f64.cos();
        assert!((g[0] - expected).abs() < 1e-11, "g[0]={}", g[0]);
    }

    #[test]
    fn test_richardson_gradient_poly() {
        // f(x) = x^5, f'(x) = 5x^4 at x=2 => 80
        let f = |xs: &[f64]| xs[0].powi(5);
        let g = richardson_gradient(&f, &[2.0], 6, 0.5);
        assert!((g[0] - 80.0).abs() < 1e-8, "g[0]={}", g[0]);
    }

    // ── complex_step_gradient ─────────────────────────────────────────────

    #[test]
    fn test_complex_step_gradient_poly() {
        // f(x) = x0^3 + x1^2, cx extension first-order in im
        let f_cx = |re: &[f64], im: &[f64]| {
            let re_out = re[0].powi(3) + re[1].powi(2);
            let im_out = 3.0 * re[0].powi(2) * im[0] + 2.0 * re[1] * im[1];
            (re_out, im_out)
        };
        let x = vec![2.0_f64, 3.0];
        let g = complex_step_gradient(&f_cx, &x, 1e-20);
        assert!((g[0] - 12.0).abs() < 1e-10, "df/dx0={}", g[0]);
        assert!((g[1] - 6.0).abs() < 1e-10, "df/dx1={}", g[1]);
    }

    #[test]
    fn test_complex_step_gradient_exp() {
        // f(x) = exp(x), cx: re_out = exp(re), im_out = exp(re)*im
        let f_cx = |re: &[f64], im: &[f64]| {
            let e = re[0].exp();
            (e, e * im[0])
        };
        let g = complex_step_gradient(&f_cx, &[1.0], 1e-20);
        assert!((g[0] - 1.0_f64.exp()).abs() < 1e-12, "g[0]={}", g[0]);
    }
}
