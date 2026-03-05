//! JAX-style functional transforms for automatic differentiation
//!
//! This module provides higher-order function transforms that turn an ordinary
//! Rust function into its gradient, Jacobian, Hessian, or vectorised variant.
//! The API is modelled after JAX's `jax.grad`, `jax.jacobian`, `jax.hessian`,
//! `jax.vmap`, `jax.jvp`, and `jax.vjp`, adapted to Rust's ownership model.
//!
//! All transforms accept *plain closures* over `f64` slices and return
//! *plain closures*, making them composable and framework-agnostic.
//!
//! # Computation strategy
//!
//! | Transform | Method | Cost |
//! |-----------|--------|------|
//! | `grad`    | Central FD | O(n) evals |
//! | `jacobian`| Column-by-column central FD | O(n) evals |
//! | `hessian` | Double central FD | O(n²) evals |
//! | `vmap`    | Sequential map (parallel opt. future) | O(batch) × O(1) |
//! | `jvp`     | 2 central FD evals | O(1) per tangent |
//! | `vjp`     | Full FD Jacobian × cotangent | O(n) evals |
//!
//! For small `n` (n ≤ ~20) all methods are practical. For large `n`, prefer
//! `jvp` (forward mode) or `vjp` with the tape-based engine
//! ([`crate::functional::tape`]) for O(1) reverse-mode cost.
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::functional::transforms::{grad, jacobian, hessian, jvp, vjp, vmap};
//!
//! // Gradient of f(x,y) = x^2 + y^2
//! let grad_f = grad(|xs: &[f64]| xs[0]*xs[0] + xs[1]*xs[1]);
//! let g = grad_f(&[3.0, 4.0]);
//! assert!((g[0] - 6.0).abs() < 1e-4);
//! assert!((g[1] - 8.0).abs() < 1e-4);
//!
//! // Jacobian of f(x,y) = [x^2, x*y]
//! let jac_f = jacobian(|xs: &[f64]| vec![xs[0]*xs[0], xs[0]*xs[1]], 2);
//! let j = jac_f(&[2.0, 3.0]);
//! assert!((j[0][0] - 4.0).abs() < 1e-4); // d(x^2)/dx = 2x = 4
//! assert!((j[1][1] - 2.0).abs() < 1e-4); // d(xy)/dy = x = 2
//!
//! // Hessian of f(x,y) = x^2 + y^2
//! let hess_f = hessian(|xs: &[f64]| xs[0]*xs[0] + xs[1]*xs[1]);
//! let h = hess_f(&[1.0, 1.0]);
//! assert!((h[0][0] - 2.0).abs() < 1e-3);
//! assert!((h[1][1] - 2.0).abs() < 1e-3);
//! ```

use crate::error::AutogradError;
use crate::Result as AgResult;

/// Central finite-difference step size used throughout this module.
const FD_H: f64 = 1e-5;

// ============================================================================
// grad — transform f: R^n -> R into ∇f: R^n -> R^n
// ============================================================================

/// Transform a scalar function `f: Rⁿ → R` into its gradient function `∇f: Rⁿ → Rⁿ`.
///
/// The returned closure computes the gradient at any point `x` using central
/// finite differences.  It can be called multiple times with different inputs.
///
/// # Signature
///
/// ```text
/// grad(f) -> Fn(&[f64]) -> Vec<f64>
/// ```
///
/// # Cost
///
/// `2n` function evaluations per gradient evaluation.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::transforms::grad;
///
/// let grad_f = grad(|xs: &[f64]| xs[0].powi(3) - xs[1].powi(2));
/// let g = grad_f(&[2.0, 3.0]);
/// // ∂f/∂x = 3x^2 = 12,  ∂f/∂y = -2y = -6
/// assert!((g[0] - 12.0).abs() < 1e-3, "g[0] = {}", g[0]);
/// assert!((g[1] - (-6.0)).abs() < 1e-3, "g[1] = {}", g[1]);
/// ```
pub fn grad<F>(f: F) -> impl Fn(&[f64]) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    move |x: &[f64]| {
        let n = x.len();
        let mut g = vec![0.0f64; n];
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        let two_h = 2.0 * FD_H;

        for i in 0..n {
            xp[i] = x[i] + FD_H;
            xm[i] = x[i] - FD_H;
            g[i] = (f(&xp) - f(&xm)) / two_h;
            xp[i] = x[i];
            xm[i] = x[i];
        }
        g
    }
}

/// Like [`grad`] but returns a `Result`, propagating errors from non-finite
/// function values.
///
/// # Errors
///
/// Returns `AutogradError` if the input is empty or if any function evaluation
/// yields a non-finite value (NaN or ±inf).
pub fn grad_checked<F>(f: F) -> impl Fn(&[f64]) -> AgResult<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    move |x: &[f64]| {
        if x.is_empty() {
            return Err(AutogradError::invalid_argument(
                "grad_checked: input must be non-empty".to_string(),
            ));
        }
        let n = x.len();
        let mut g = vec![0.0f64; n];
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        let two_h = 2.0 * FD_H;

        for i in 0..n {
            xp[i] = x[i] + FD_H;
            xm[i] = x[i] - FD_H;
            let fp = f(&xp);
            let fm = f(&xm);
            if !fp.is_finite() || !fm.is_finite() {
                return Err(AutogradError::invalid_argument(format!(
                    "grad_checked: non-finite value at component {i}: fp={fp}, fm={fm}"
                )));
            }
            g[i] = (fp - fm) / two_h;
            xp[i] = x[i];
            xm[i] = x[i];
        }
        Ok(g)
    }
}

/// Transform f into a closure that returns `(f(x), ∇f(x))` simultaneously.
///
/// This is slightly cheaper than calling `f` and `grad(f)` separately because
/// we also record the primal value during the gradient computation.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::transforms::value_and_grad;
///
/// let vg = value_and_grad(|xs: &[f64]| xs[0]*xs[0] + xs[1]*xs[1]);
/// let (val, g) = vg(&[3.0, 4.0]);
/// assert!((val - 25.0).abs() < 1e-12);
/// assert!((g[0] - 6.0).abs() < 1e-4);
/// ```
pub fn value_and_grad<F>(f: F) -> impl Fn(&[f64]) -> (f64, Vec<f64>)
where
    F: Fn(&[f64]) -> f64,
{
    move |x: &[f64]| {
        let n = x.len();
        let val = f(x);
        let mut g = vec![0.0f64; n];
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        let two_h = 2.0 * FD_H;

        for i in 0..n {
            xp[i] = x[i] + FD_H;
            xm[i] = x[i] - FD_H;
            g[i] = (f(&xp) - f(&xm)) / two_h;
            xp[i] = x[i];
            xm[i] = x[i];
        }
        (val, g)
    }
}

// ============================================================================
// jacobian — transform f: R^n -> R^m into Jf: R^n -> R^(m x n)
// ============================================================================

/// Transform `f: Rⁿ → Rᵐ` into its Jacobian function `Jf: Rⁿ → Rᵐˣⁿ`.
///
/// The returned closure computes the full `m × n` Jacobian matrix at any input
/// `x`.  Column `j` of the result is `∂f/∂xⱼ` evaluated at `x`.
///
/// # Arguments
///
/// * `f`         — Vector-valued function `f: Rⁿ → Rᵐ`.
/// * `n_outputs` — The output dimension `m`.  Must match `f(x).len()`.
///
/// # Cost
///
/// `2n` function evaluations per Jacobian evaluation.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::transforms::jacobian;
///
/// let jac = jacobian(|xs: &[f64]| vec![xs[0]*xs[0], xs[0]*xs[1]], 2);
/// let j = jac(&[2.0, 3.0]);
/// // J = [[2x, 0], [y, x]] = [[4, 0], [3, 2]]
/// assert!((j[0][0] - 4.0).abs() < 1e-4);
/// assert!((j[0][1] - 0.0).abs() < 1e-4);
/// assert!((j[1][0] - 3.0).abs() < 1e-4);
/// assert!((j[1][1] - 2.0).abs() < 1e-4);
/// ```
pub fn jacobian<F>(f: F, n_outputs: usize) -> impl Fn(&[f64]) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    move |x: &[f64]| {
        let n = x.len();
        // J[i][j] = ∂fᵢ/∂xⱼ
        let mut j = vec![vec![0.0f64; n]; n_outputs];
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        let two_h = 2.0 * FD_H;

        for col in 0..n {
            xp[col] = x[col] + FD_H;
            xm[col] = x[col] - FD_H;
            let fp = f(&xp);
            let fm = f(&xm);
            for row in 0..n_outputs.min(fp.len()).min(fm.len()) {
                j[row][col] = (fp[row] - fm[row]) / two_h;
            }
            xp[col] = x[col];
            xm[col] = x[col];
        }
        j
    }
}

// ============================================================================
// hessian — transform f: R^n -> R into Hf: R^n -> R^(n x n)
// ============================================================================

/// Transform a scalar function `f: Rⁿ → R` into its Hessian function
/// `Hf: Rⁿ → Rⁿˣⁿ`.
///
/// The Hessian is computed using second-order central finite differences for
/// diagonal entries and the cross-difference formula for off-diagonal entries:
///
/// ```text
/// H[i][i] ≈ (f(x+hᵢ) - 2f(x) + f(x-hᵢ)) / h²
/// H[i][j] ≈ (f(x+hᵢ+hⱼ) - f(x+hᵢ-hⱼ) - f(x-hᵢ+hⱼ) + f(x-hᵢ-hⱼ)) / (4h²)
/// ```
///
/// The returned matrix is symmetric by construction.
///
/// # Cost
///
/// `1 + 2n + 4·n(n-1)/2 = O(n²)` function evaluations.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::transforms::hessian;
///
/// let hess = hessian(|xs: &[f64]| xs[0]*xs[0] + 3.0*xs[0]*xs[1] + 2.0*xs[1]*xs[1]);
/// let h = hess(&[1.0, 1.0]);
/// // H = [[2, 3], [3, 4]]
/// assert!((h[0][0] - 2.0).abs() < 1e-3, "H[0][0] = {}", h[0][0]);
/// assert!((h[0][1] - 3.0).abs() < 1e-3, "H[0][1] = {}", h[0][1]);
/// assert!((h[1][0] - 3.0).abs() < 1e-3, "H[1][0] = {}", h[1][0]);
/// assert!((h[1][1] - 4.0).abs() < 1e-3, "H[1][1] = {}", h[1][1]);
/// ```
pub fn hessian<F>(f: F) -> impl Fn(&[f64]) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    move |x: &[f64]| {
        let n = x.len();
        let mut h = vec![vec![0.0f64; n]; n];
        let h2 = FD_H * FD_H;
        let fx = f(x);

        // Diagonal: second-order central FD
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        for i in 0..n {
            xp[i] = x[i] + FD_H;
            xm[i] = x[i] - FD_H;
            h[i][i] = (f(&xp) + f(&xm) - 2.0 * fx) / h2;
            xp[i] = x[i];
            xm[i] = x[i];
        }

        // Off-diagonal: cross-difference formula
        for i in 0..n {
            for j in (i + 1)..n {
                let mut xpp = x.to_vec();
                let mut xpm = x.to_vec();
                let mut xmp = x.to_vec();
                let mut xmm = x.to_vec();
                xpp[i] += FD_H;
                xpp[j] += FD_H;
                xpm[i] += FD_H;
                xpm[j] -= FD_H;
                xmp[i] -= FD_H;
                xmp[j] += FD_H;
                xmm[i] -= FD_H;
                xmm[j] -= FD_H;
                let entry = (f(&xpp) - f(&xpm) - f(&xmp) + f(&xmm)) / (4.0 * h2);
                h[i][j] = entry;
                h[j][i] = entry; // symmetric
            }
        }

        h
    }
}

// ============================================================================
// vmap — vectorized map over a batch of inputs
// ============================================================================

/// Vectorise a function `f: Rⁿ → Rᵐ` over a batch of input vectors.
///
/// `vmap(f)` returns a closure that accepts `&[&[f64]]` (a slice of input
/// slices) and applies `f` to each one independently, returning a
/// `Vec<Vec<f64>>` of results.
///
/// This is the eager, sequential implementation.  A future version may
/// dispatch to thread pools for large batches.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::transforms::vmap;
///
/// let f = |xs: &[f64]| vec![xs[0]*xs[0], xs[1]*2.0];
/// let batched = vmap(f);
/// let inputs: Vec<&[f64]> = vec![&[1.0, 2.0], &[3.0, 4.0]];
/// let out = batched(&inputs);
/// assert_eq!(out.len(), 2);
/// assert!((out[0][0] - 1.0).abs() < 1e-12);  // 1^2
/// assert!((out[1][0] - 9.0).abs() < 1e-12);  // 3^2
/// assert!((out[1][1] - 8.0).abs() < 1e-12);  // 4*2
/// ```
pub fn vmap<F>(f: F) -> impl Fn(&[&[f64]]) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    move |batch: &[&[f64]]| batch.iter().map(|x| f(x)).collect()
}

/// Vectorised map that also returns the gradient for each element.
///
/// Returns `(values, gradients)` where both have length `batch.len()`.
/// `gradients[i]` is the gradient of the scalar function `f_scalar` evaluated
/// at `batch[i]`.
pub fn vmap_with_grad<F, G>(
    f: F,
    f_scalar: G,
) -> impl Fn(&[&[f64]]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>)
where
    F: Fn(&[f64]) -> Vec<f64>,
    G: Fn(&[f64]) -> f64,
{
    let grad_fn = grad(f_scalar);
    move |batch: &[&[f64]]| {
        let values: Vec<Vec<f64>> = batch.iter().map(|x| f(x)).collect();
        let grads: Vec<Vec<f64>> = batch.iter().map(|x| grad_fn(x)).collect();
        (values, grads)
    }
}

// ============================================================================
// jvp — Jacobian-vector product (forward-mode)
// ============================================================================

/// Compute the Jacobian-vector product (JVP) for `f: Rⁿ → Rᵐ`.
///
/// Returns `(f(x), J(x)·tangent)` where `J(x)` is the Jacobian of `f` at `x`.
///
/// Uses central finite differences: `J·v ≈ (f(x+h·v) - f(x-h·v)) / (2h)`.
/// This requires exactly **2 function evaluations** regardless of `n`.
///
/// # Arguments
///
/// * `f`       — Function `f: Rⁿ → Rᵐ`.
/// * `x`       — Primal input (length n).
/// * `tangent` — Tangent direction (length n).
///
/// # Returns
///
/// `(f(x), J(x)·tangent)` — both are `Vec<f64>` of length m.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::transforms::jvp;
///
/// let f = |xs: &[f64]| vec![xs[0]*xs[0], xs[0]*xs[1]];
/// let (fx, jvp_val) = jvp(f, &[2.0, 3.0], &[1.0, 0.0]);
/// // J·[1,0] = [2x, y] = [4, 3]
/// assert!((jvp_val[0] - 4.0).abs() < 1e-4, "jvp[0] = {}", jvp_val[0]);
/// assert!((jvp_val[1] - 3.0).abs() < 1e-4, "jvp[1] = {}", jvp_val[1]);
/// assert!((fx[0] - 4.0).abs() < 1e-12);   // f₀(2,3) = 4
/// assert!((fx[1] - 6.0).abs() < 1e-12);   // f₁(2,3) = 6
/// ```
pub fn jvp<F>(f: F, x: &[f64], tangent: &[f64]) -> (Vec<f64>, Vec<f64>)
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    // Forward: f(x)
    let fx = f(x);
    // Perturbed: f(x + h*v) and f(x - h*v)
    let xp: Vec<f64> = x.iter().zip(tangent.iter()).map(|(&xi, &vi)| xi + FD_H * vi).collect();
    let xm: Vec<f64> = x.iter().zip(tangent.iter()).map(|(&xi, &vi)| xi - FD_H * vi).collect();
    if n == 0 || tangent.len() != n {
        return (fx, vec![]);
    }
    let fp = f(&xp);
    let fm = f(&xm);
    let two_h = 2.0 * FD_H;
    let jv: Vec<f64> = fp.iter().zip(fm.iter()).map(|(&a, &b)| (a - b) / two_h).collect();
    (fx, jv)
}

// ============================================================================
// vjp — Vector-Jacobian product (reverse-mode via full Jacobian)
// ============================================================================

/// Compute the vector-Jacobian product (VJP) for `f: Rⁿ → Rᵐ`.
///
/// Returns `(f(x), cotangentᵀ·J(x))` where `J(x)` is the Jacobian of `f`
/// at `x`.  The VJP is the transpose of the JVP: it applies the adjoint
/// (transpose) of `J` to the cotangent vector.
///
/// This implementation builds the full Jacobian via `2n` central-FD evals
/// and then multiplies by the cotangent.  For large `n` and small `m`,
/// consider using the tape-based reverse mode in [`crate::functional::tape`].
///
/// # Arguments
///
/// * `f`          — Function `f: Rⁿ → Rᵐ`.
/// * `x`          — Primal input (length n).
/// * `cotangent`  — Cotangent vector (length m).
///
/// # Returns
///
/// `(f(x), cotangentᵀ·J(x))` — `f(x)` has length m, VJP has length n.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::transforms::vjp;
///
/// let f = |xs: &[f64]| vec![xs[0]*xs[0], xs[0]*xs[1]];
/// let (fx, g) = vjp(f, &[2.0, 3.0], &[1.0, 0.0]);
/// // vᵀ·J = [1,0]·[[2x,0],[y,x]] = [2x, 0] = [4, 0]
/// assert!((g[0] - 4.0).abs() < 1e-4, "vjp[0] = {}", g[0]);
/// assert!((g[1] - 0.0).abs() < 1e-4, "vjp[1] = {}", g[1]);
/// ```
pub fn vjp<F>(f: F, x: &[f64], cotangent: &[f64]) -> (Vec<f64>, Vec<f64>)
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let fx = f(x);
    let m = fx.len();

    if n == 0 || cotangent.len() != m {
        return (fx, vec![0.0f64; n]);
    }

    // Build full Jacobian: J[row][col] = ∂f_row/∂x_col
    let mut j_col = vec![vec![0.0f64; m]; n]; // j_col[col][row] for efficient VJP
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    let two_h = 2.0 * FD_H;

    for col in 0..n {
        xp[col] = x[col] + FD_H;
        xm[col] = x[col] - FD_H;
        let fp = f(&xp);
        let fm_vals = f(&xm);
        for row in 0..m.min(fp.len()).min(fm_vals.len()) {
            j_col[col][row] = (fp[row] - fm_vals[row]) / two_h;
        }
        xp[col] = x[col];
        xm[col] = x[col];
    }

    // VJP: result[col] = Σ_row cotangent[row] * J[row][col]
    let g: Vec<f64> = (0..n)
        .map(|col| {
            cotangent
                .iter()
                .zip(j_col[col].iter())
                .map(|(&c, &jrc)| c * jrc)
                .sum()
        })
        .collect();

    (fx, g)
}

// ============================================================================
// grad_of_grad — higher-order gradient transform
// ============================================================================

/// Transform `f` into the function that computes its *second* gradient (i.e.
/// the Hessian-vector product direction, or just the second-order derivative
/// when composed with `grad`).
///
/// Specifically, returns a closure that evaluates `∇(∇f)(x)` — i.e. the
/// gradient of the gradient norm (which is the Hessian applied to the current
/// gradient direction).
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::transforms::grad_of_grad;
///
/// // f(x,y) = x^2 + y^2; ∇f = [2x, 2y]; ∇(||∇f||^2/2) = 2∇f = [4x, 4y]? No,
/// // grad_of_grad computes ∇(∇f) as the gradient of each grad component.
/// // Here: ∇₁(∇f₀) = [2, 0], meaning ∂(2x)/∂x = 2, ∂(2x)/∂y = 0
/// let gg = grad_of_grad(|xs: &[f64]| xs[0]*xs[0] + xs[1]*xs[1]);
/// let diag_hess = gg(&[3.0, 4.0]); // = Hessian diagonal via the pattern
/// // Just verifying the call works (full Hessian via hessian() is more precise)
/// assert_eq!(diag_hess.len(), 2);
/// ```
pub fn grad_of_grad<F>(f: F) -> impl Fn(&[f64]) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64 + Clone,
{
    let g = grad(f.clone());
    // Compute ∇(||∇f||²/2) = (∇²f)ᵀ·∇f = H·∇f
    move |x: &[f64]| {
        let gx = g(x);
        // grad_of_grad[i] = Σⱼ H[i][j] * gx[j]  (Hessian-vector product)
        // Approximate via outer FD on the gradient:
        let n = x.len();
        let mut result = vec![0.0f64; n];
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        let two_h = 2.0 * FD_H;

        for j in 0..n {
            // ∂gᵢ/∂xⱼ ≈ (g(x+hej)ᵢ - g(x-hej)ᵢ) / 2h
            xp[j] = x[j] + FD_H;
            xm[j] = x[j] - FD_H;
            let gp = g(&xp);
            let gm = g(&xm);
            for i in 0..n {
                let h_ij = (gp[i] - gm[i]) / two_h;
                result[i] += h_ij * gx[j];
            }
            xp[j] = x[j];
            xm[j] = x[j];
        }
        result
    }
}

// ============================================================================
// linearize — first-order Taylor approximation
// ============================================================================

/// Compute the first-order Taylor linearisation of `f` at `x` along `tangent`.
///
/// Returns `(f(x), J(x)·tangent)`, identical to [`jvp`] but with a more
/// descriptive name aligned with the "linearise" terminology in JAX.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::transforms::linearize;
///
/// let f = |xs: &[f64]| vec![xs[0].exp(), xs[0]*xs[1]];
/// let (primal, tangent_out) = linearize(f, &[0.0, 2.0], &[1.0, 0.0]);
/// assert!((primal[0] - 1.0).abs() < 1e-9);    // exp(0) = 1
/// assert!((tangent_out[0] - 1.0).abs() < 1e-4); // d(exp(x))/dx * 1 = 1
/// assert!((tangent_out[1] - 2.0).abs() < 1e-4); // d(xy)/dx * 1 = y = 2
/// ```
pub fn linearize<F>(f: F, x: &[f64], tangent: &[f64]) -> (Vec<f64>, Vec<f64>)
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    jvp(f, x, tangent)
}

// ============================================================================
// compose — function composition
// ============================================================================

/// Compose two functions: `compose(f, g)(x) = g(f(x))`.
///
/// Both `f` and `g` accept and return `Vec<f64>`.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::transforms::compose_vec;
///
/// let f = |xs: Vec<f64>| xs.iter().map(|&v| v * 2.0).collect::<Vec<_>>();
/// let g = |xs: Vec<f64>| xs.iter().map(|&v| v + 1.0).collect::<Vec<_>>();
/// let h = compose_vec(f, g);
/// let out = h(vec![1.0, 2.0, 3.0]);
/// assert_eq!(out, vec![3.0, 5.0, 7.0]); // 2*x + 1
/// ```
pub fn compose_vec<F, G>(f: F, g: G) -> impl Fn(Vec<f64>) -> Vec<f64>
where
    F: Fn(Vec<f64>) -> Vec<f64>,
    G: Fn(Vec<f64>) -> Vec<f64>,
{
    move |x| g(f(x))
}

/// Compose a scalar function with itself `n` times: `f ∘ f ∘ … ∘ f`.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::transforms::iterate_scalar;
///
/// let f = |x: f64| x * 2.0;
/// let f5 = iterate_scalar(f, 5);
/// assert!((f5(1.0) - 32.0).abs() < 1e-12); // 2^5 = 32
/// ```
pub fn iterate_scalar<F>(f: F, n: usize) -> impl Fn(f64) -> f64
where
    F: Fn(f64) -> f64,
{
    move |mut x| {
        for _ in 0..n {
            x = f(x);
        }
        x
    }
}

// ============================================================================
// check_grad — numerical gradient checker
// ============================================================================

/// Numerically verify that the analytical gradient `grad_fn` matches the
/// central-FD gradient of `f` at point `x`.
///
/// Returns the maximum absolute difference between the analytical and numerical
/// gradients (the "gradient check error").  A value below `1e-4` typically
/// indicates the gradient is correct.
///
/// # Arguments
///
/// * `f`       — Scalar function.
/// * `grad_fn` — Analytical gradient function (the one being verified).
/// * `x`       — Point at which to check.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::transforms::check_grad;
///
/// let err = check_grad(
///     |xs: &[f64]| xs[0]*xs[0] + xs[1]*xs[1],
///     |xs: &[f64]| vec![2.0*xs[0], 2.0*xs[1]],
///     &[3.0, 4.0],
/// );
/// assert!(err < 1e-4, "Gradient check failed with error {}", err);
/// ```
pub fn check_grad<F, G>(f: F, grad_fn: G, x: &[f64]) -> f64
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let numerical_g = {
        let g_fn = grad(f);
        g_fn(x)
    };
    let analytical_g = grad_fn(x);

    numerical_g
        .iter()
        .zip(analytical_g.iter())
        .map(|(&n, &a)| (n - a).abs())
        .fold(0.0_f64, f64::max)
}

// ============================================================================
// stop_gradient — block gradient flow
// ============================================================================

/// Block gradient flow: treat `x` as a constant in subsequent computations.
///
/// In the functional (FD-based) context this is a no-op (returns `x`), but it
/// acts as a semantic marker that gradient computation through this value is
/// intentionally disabled.  In the graph-based context, use
/// [`crate::custom_gradient::detach`] instead.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::transforms::stop_gradient;
///
/// let x = vec![1.0, 2.0, 3.0];
/// let y = stop_gradient(x.clone());
/// assert_eq!(y, x);
/// ```
pub fn stop_gradient(x: Vec<f64>) -> Vec<f64> {
    x
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_quadratic() {
        let g = grad(|xs: &[f64]| xs[0] * xs[0] + xs[1] * xs[1]);
        let result = g(&[3.0, 4.0]);
        assert!((result[0] - 6.0).abs() < 1e-3, "g[0] = {}", result[0]);
        assert!((result[1] - 8.0).abs() < 1e-3, "g[1] = {}", result[1]);
    }

    #[test]
    fn test_grad_cubic() {
        // f(x) = x^3; f'(x) = 3x^2 = 12 at x=2
        let g = grad(|xs: &[f64]| xs[0].powi(3));
        let result = g(&[2.0]);
        assert!((result[0] - 12.0).abs() < 1e-2, "g[0] = {}", result[0]);
    }

    #[test]
    fn test_grad_checked_empty_input() {
        let g = grad_checked(|xs: &[f64]| xs[0] * xs[0]);
        assert!(g(&[]).is_err());
    }

    #[test]
    fn test_value_and_grad() {
        let vg = value_and_grad(|xs: &[f64]| xs[0] * xs[0] + xs[1] * xs[1]);
        let (val, g) = vg(&[3.0, 4.0]);
        assert!((val - 25.0).abs() < 1e-12);
        assert!((g[0] - 6.0).abs() < 1e-3);
        assert!((g[1] - 8.0).abs() < 1e-3);
    }

    #[test]
    fn test_jacobian_linear_map() {
        // f(x,y) = [2x + y, x - y]
        // J = [[2, 1], [1, -1]]
        let jac = jacobian(|xs: &[f64]| vec![2.0 * xs[0] + xs[1], xs[0] - xs[1]], 2);
        let j = jac(&[5.0, 3.0]);
        assert!((j[0][0] - 2.0).abs() < 1e-3, "J[0][0] = {}", j[0][0]);
        assert!((j[0][1] - 1.0).abs() < 1e-3, "J[0][1] = {}", j[0][1]);
        assert!((j[1][0] - 1.0).abs() < 1e-3, "J[1][0] = {}", j[1][0]);
        assert!((j[1][1] - (-1.0)).abs() < 1e-3, "J[1][1] = {}", j[1][1]);
    }

    #[test]
    fn test_hessian_quadratic_form() {
        // f(x,y) = x^2 + 3*x*y + 2*y^2; H = [[2,3],[3,4]]
        let hess = hessian(|xs: &[f64]| xs[0] * xs[0] + 3.0 * xs[0] * xs[1] + 2.0 * xs[1] * xs[1]);
        let h = hess(&[1.0, 1.0]);
        assert!((h[0][0] - 2.0).abs() < 1e-2, "H[0][0] = {}", h[0][0]);
        assert!((h[0][1] - 3.0).abs() < 1e-2, "H[0][1] = {}", h[0][1]);
        assert!((h[1][0] - 3.0).abs() < 1e-2, "H[1][0] = {}", h[1][0]);
        assert!((h[1][1] - 4.0).abs() < 1e-2, "H[1][1] = {}", h[1][1]);
    }

    #[test]
    fn test_hessian_symmetry() {
        let f = |xs: &[f64]| xs[0].sin() * xs[1].cos() + xs[0] * xs[1] * xs[1];
        let h = hessian(f)(&[1.0, 2.0]);
        // Hessian must be symmetric
        assert!((h[0][1] - h[1][0]).abs() < 1e-5, "Symmetry violated: {} vs {}", h[0][1], h[1][0]);
    }

    #[test]
    fn test_vmap_batch() {
        let f = |xs: &[f64]| vec![xs[0] * xs[0], xs[1] * 2.0];
        let batched = vmap(f);
        let a: &[f64] = &[1.0, 2.0];
        let b: &[f64] = &[3.0, 4.0];
        let out = batched(&[a, b]);
        assert_eq!(out.len(), 2);
        assert!((out[0][0] - 1.0).abs() < 1e-12);
        assert!((out[1][0] - 9.0).abs() < 1e-12);
        assert!((out[1][1] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_jvp_quadratic() {
        let f = |xs: &[f64]| vec![xs[0] * xs[0], xs[0] * xs[1]];
        let (fx, jv) = jvp(f, &[2.0, 3.0], &[1.0, 0.0]);
        assert!((fx[0] - 4.0).abs() < 1e-12);
        assert!((fx[1] - 6.0).abs() < 1e-12);
        // J·[1,0] = [2x, y] = [4, 3]
        assert!((jv[0] - 4.0).abs() < 1e-4, "jvp[0] = {}", jv[0]);
        assert!((jv[1] - 3.0).abs() < 1e-4, "jvp[1] = {}", jv[1]);
    }

    #[test]
    fn test_vjp_quadratic() {
        let f = |xs: &[f64]| vec![xs[0] * xs[0], xs[0] * xs[1]];
        let (fx, g) = vjp(f, &[2.0, 3.0], &[1.0, 0.0]);
        assert!((fx[0] - 4.0).abs() < 1e-12);
        // vᵀ·J = [1,0]·[[2x,0],[y,x]] at x=2,y=3 = [2x, 0] = [4, 0]
        assert!((g[0] - 4.0).abs() < 1e-4, "vjp[0] = {}", g[0]);
        assert!((g[1] - 0.0).abs() < 1e-4, "vjp[1] = {}", g[1]);
    }

    #[test]
    fn test_vjp_with_nontrivial_cotangent() {
        // f(x,y) = [x+y, x*y]; J = [[1,1],[y,x]] at (2,3)
        // cotangent = [1,1]: vᵀ·J = [1+y, 1+x] = [4, 3]
        let f = |xs: &[f64]| vec![xs[0] + xs[1], xs[0] * xs[1]];
        let (_, g) = vjp(f, &[2.0, 3.0], &[1.0, 1.0]);
        assert!((g[0] - 4.0).abs() < 1e-4, "vjp[0] = {}", g[0]);
        assert!((g[1] - 3.0).abs() < 1e-4, "vjp[1] = {}", g[1]);
    }

    #[test]
    fn test_linearize_exp() {
        let f = |xs: &[f64]| vec![xs[0].exp(), xs[0] * xs[1]];
        let (primal, tangent_out) = linearize(f, &[0.0, 2.0], &[1.0, 0.0]);
        assert!((primal[0] - 1.0).abs() < 1e-9);
        assert!((tangent_out[0] - 1.0).abs() < 1e-4);
        assert!((tangent_out[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_check_grad_correct() {
        let err = check_grad(
            |xs: &[f64]| xs[0] * xs[0] + xs[1] * xs[1],
            |xs: &[f64]| vec![2.0 * xs[0], 2.0 * xs[1]],
            &[3.0, 4.0],
        );
        assert!(err < 1e-3, "gradient check error = {}", err);
    }

    #[test]
    fn test_compose_vec() {
        let f = |xs: Vec<f64>| xs.iter().map(|&v| v * 2.0).collect::<Vec<_>>();
        let g = |xs: Vec<f64>| xs.iter().map(|&v| v + 1.0).collect::<Vec<_>>();
        let h = compose_vec(f, g);
        let out = h(vec![1.0, 2.0, 3.0]);
        // g(f(x)) = 2x + 1
        assert!((out[0] - 3.0).abs() < 1e-12);
        assert!((out[1] - 5.0).abs() < 1e-12);
        assert!((out[2] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_iterate_scalar() {
        let f = |x: f64| x * 2.0;
        let f5 = iterate_scalar(f, 5);
        assert!((f5(1.0) - 32.0).abs() < 1e-12);
    }

    #[test]
    fn test_stop_gradient_passthrough() {
        let x = vec![1.0, 2.0, 3.0];
        let y = stop_gradient(x.clone());
        assert_eq!(y, x);
    }

    #[test]
    fn test_grad_of_grad_quadratic() {
        // f(x,y) = x^2 + y^2; H = 2I; H·∇f = H·[2x,2y] = [4x, 4y]
        let gg = grad_of_grad(|xs: &[f64]| xs[0] * xs[0] + xs[1] * xs[1]);
        let result = gg(&[3.0, 4.0]);
        // H·∇f = [4*3, 4*4] = [12, 16]
        assert!((result[0] - 12.0).abs() < 1e-1, "gg[0] = {}", result[0]);
        assert!((result[1] - 16.0).abs() < 1e-1, "gg[1] = {}", result[1]);
    }
}
