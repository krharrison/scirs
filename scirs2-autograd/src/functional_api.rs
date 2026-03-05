//! Functional differentiation API (JAX-style, higher-order functions)
//!
//! This module provides a higher-order-function API for automatic differentiation
//! that returns **function-level transforms** rather than point-wise values.
//! The design mirrors JAX's `jax.grad`, `jax.value_and_grad`, `jax.vmap`, and
//! `jax.lax.scan`.
//!
//! All operations use **central finite differences** (step `h = 1e-7`) and
//! are purely functional — no computation-graph tensors are required.
//!
//! # Overview
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`grad`] | Transform `f: Rⁿ→R` into its gradient function `∇f: Rⁿ→Rⁿ` |
//! | [`value_and_grad`] | Transform into `(f(x), ∇f(x))` function |
//! | [`vmap`] | Vectorise a function over a batch of inputs |
//! | [`scan`] | Sequential scan with carry state (cumulative reduction) |
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::functional_api::{grad, value_and_grad, vmap, scan};
//!
//! // Build a gradient function
//! let grad_f = grad(|xs: &[f64]| xs[0]*xs[0] + xs[1]*xs[1]);
//! let g = grad_f(&[3.0, 4.0]).expect("grad");
//! assert!((g[0] - 6.0).abs() < 1e-4);
//! assert!((g[1] - 8.0).abs() < 1e-4);
//!
//! // Value + gradient simultaneously
//! let vg = value_and_grad(|xs: &[f64]| xs[0]*xs[0] + xs[1]*xs[1]);
//! let (val, g2) = vg(&[3.0, 4.0]).expect("value_and_grad");
//! assert!((val - 25.0).abs() < 1e-9);
//! assert!((g2[0] - 6.0).abs() < 1e-4);
//! ```

use crate::error::AutogradError;
use crate::Result;

/// Central finite-difference step size (matches `jvp_vjp` module).
const FD_H: f64 = 1e-7;

// ============================================================================
// grad — transform f into its gradient function
// ============================================================================

/// Transform a scalar function `f: Rⁿ → R` into its gradient function `∇f`.
///
/// The returned closure accepts a slice reference `&[f64]` and returns
/// `Result<Vec<f64>>` containing the gradient at that point.
///
/// Gradients are computed via **central finite differences**:
/// `∂f/∂xᵢ ≈ (f(x + hᵢ) − f(x − hᵢ)) / (2h)`
///
/// # Arguments
///
/// * `f` — A scalar function.  The closure is captured by value and may be
///   called multiple times, so it must be `Clone + 'static`.
///
/// # Returns
///
/// A closure `impl Fn(&[f64]) -> Result<Vec<f64>>`.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional_api::grad;
///
/// let grad_f = grad(|xs: &[f64]| xs[0]*xs[0] + xs[1]*xs[1]);
/// let g = grad_f(&[3.0, 4.0]).expect("grad at (3,4)");
/// assert!((g[0] - 6.0).abs() < 1e-4);
/// assert!((g[1] - 8.0).abs() < 1e-4);
/// ```
pub fn grad<F>(f: F) -> impl Fn(&[f64]) -> Result<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    move |x: &[f64]| {
        let n = x.len();
        if n == 0 {
            return Err(AutogradError::invalid_argument(
                "grad: input must be non-empty".to_string(),
            ));
        }

        let mut g = vec![0.0f64; n];
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        let two_h = 2.0 * FD_H;

        for i in 0..n {
            xp[i] += FD_H;
            xm[i] -= FD_H;
            g[i] = (f(&xp) - f(&xm)) / two_h;
            xp[i] = x[i];
            xm[i] = x[i];
        }

        Ok(g)
    }
}

// ============================================================================
// value_and_grad — transform f into (f(x), ∇f(x)) function
// ============================================================================

/// Transform a scalar function into a function that returns both the value
/// and the gradient simultaneously.
///
/// This is more efficient than calling `f(x)` and `grad(f)(x)` separately
/// because it avoids one extra function evaluation.
///
/// # Arguments
///
/// * `f` — A scalar function `f: Rⁿ → R`.
///
/// # Returns
///
/// A closure `impl Fn(&[f64]) -> Result<(f64, Vec<f64>)>` that returns
/// `(f(x), ∇f(x))`.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional_api::value_and_grad;
///
/// let vg = value_and_grad(|xs: &[f64]| xs[0].powi(2) + xs[1].powi(2));
/// let (val, g) = vg(&[3.0, 4.0]).expect("value_and_grad at (3,4)");
/// assert!((val - 25.0).abs() < 1e-12);
/// assert!((g[0] - 6.0).abs() < 1e-4);
/// assert!((g[1] - 8.0).abs() < 1e-4);
/// ```
pub fn value_and_grad<F>(f: F) -> impl Fn(&[f64]) -> Result<(f64, Vec<f64>)>
where
    F: Fn(&[f64]) -> f64,
{
    move |x: &[f64]| {
        let n = x.len();
        if n == 0 {
            return Err(AutogradError::invalid_argument(
                "value_and_grad: input must be non-empty".to_string(),
            ));
        }

        let val = f(x);
        let mut g = vec![0.0f64; n];
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        let two_h = 2.0 * FD_H;

        for i in 0..n {
            xp[i] += FD_H;
            xm[i] -= FD_H;
            g[i] = (f(&xp) - f(&xm)) / two_h;
            xp[i] = x[i];
            xm[i] = x[i];
        }

        Ok((val, g))
    }
}

// ============================================================================
// vmap — vectorised map over a batch
// ============================================================================

/// Vectorise a function `f: Rⁿ → Rᵐ` over a batch of inputs.
///
/// `vmap(f)` returns a closure that applies `f` independently to each row of
/// a `batch × n` matrix (represented as `Vec<Vec<f64>>`) and stacks the
/// results into a `batch × m` output.
///
/// This is semantically equivalent to:
/// ```text
/// outputs[i] = f(inputs[i])  for each i in 0..batch
/// ```
///
/// # Arguments
///
/// * `f` — The function to vectorise.  It is called once per row.
///
/// # Returns
///
/// A closure `impl Fn(Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>>`.
///
/// # Errors
///
/// The returned closure returns `AutogradError` if:
/// - `inputs` is empty
/// - Any row has length 0
/// - The function produces inconsistent output sizes across rows
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional_api::vmap;
///
/// let double = vmap(|xs: &[f64]| vec![xs[0] * 2.0, xs[1] * 3.0]);
/// let batch = vec![
///     vec![1.0, 2.0],
///     vec![3.0, 4.0],
///     vec![5.0, 6.0],
/// ];
/// let result = double(batch).expect("vmap result");
/// assert!((result[0][0] - 2.0).abs() < 1e-12);
/// assert!((result[1][1] - 12.0).abs() < 1e-12);
/// assert!((result[2][0] - 10.0).abs() < 1e-12);
/// ```
pub fn vmap<F>(f: F) -> impl Fn(Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    move |inputs: Vec<Vec<f64>>| {
        if inputs.is_empty() {
            return Err(AutogradError::invalid_argument(
                "vmap: input batch is empty".to_string(),
            ));
        }

        // Probe output dimension from first row
        let first_row = &inputs[0];
        if first_row.is_empty() {
            return Err(AutogradError::invalid_argument(
                "vmap: input rows must be non-empty".to_string(),
            ));
        }
        let out0 = f(first_row);
        let m = out0.len();
        if m == 0 {
            return Err(AutogradError::invalid_argument(
                "vmap: function output must be non-empty".to_string(),
            ));
        }

        let batch = inputs.len();
        let mut outputs = Vec::with_capacity(batch);
        outputs.push(out0);

        for row in inputs.iter().skip(1) {
            if row.is_empty() {
                return Err(AutogradError::invalid_argument(
                    "vmap: all input rows must be non-empty".to_string(),
                ));
            }
            let out = f(row);
            if out.len() != m {
                return Err(AutogradError::ShapeMismatch(format!(
                    "vmap: inconsistent output lengths: expected {m}, got {}",
                    out.len()
                )));
            }
            outputs.push(out);
        }

        Ok(outputs)
    }
}

// ============================================================================
// scan — sequential scan with carry state
// ============================================================================

/// Sequential scan with a carry state.
///
/// `scan(f, init, xs)` iterates a function `f(carry, x) → (new_carry, y)` over
/// a sequence `xs`, threading the carry state through each step and collecting
/// the outputs.
///
/// This is analogous to a left fold that also returns all intermediate values:
/// ```text
/// carry₀ = init
/// (carry₁, y₀) = f(carry₀, xs[0])
/// (carry₂, y₁) = f(carry₁, xs[1])
/// ...
/// (carryₙ, yₙ₋₁) = f(carryₙ₋₁, xs[n-1])
/// returns (carryₙ, [y₀, y₁, ..., yₙ₋₁])
/// ```
///
/// # Arguments
///
/// * `f`    — Step function `f(carry: &[f64], x: &[f64]) -> (Vec<f64>, Vec<f64>)`
///   returning `(new_carry, output_y)`.
/// * `init` — Initial carry state.
/// * `xs`   — Sequence of inputs to scan over.
///
/// # Returns
///
/// `(final_carry, outputs)` where `outputs[i]` is the output produced at step `i`.
///
/// # Errors
///
/// Returns `AutogradError` if the step function returns an empty carry or output.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional_api::scan;
///
/// // Cumulative sum: carry = running sum, y = current element doubled
/// let (final_carry, ys) = scan(
///     |carry: &[f64], x: &[f64]| {
///         let new_carry = vec![carry[0] + x[0]];
///         let y = vec![x[0] * 2.0];
///         (new_carry, y)
///     },
///     vec![0.0],
///     &[vec![1.0], vec![2.0], vec![3.0]],
/// ).expect("scan");
///
/// assert!((final_carry[0] - 6.0).abs() < 1e-12);  // 0+1+2+3
/// assert!((ys[0][0] - 2.0).abs() < 1e-12);         // 1*2
/// assert!((ys[1][0] - 4.0).abs() < 1e-12);         // 2*2
/// assert!((ys[2][0] - 6.0).abs() < 1e-12);         // 3*2
/// ```
pub fn scan<F>(
    f: F,
    init: Vec<f64>,
    xs: &[Vec<f64>],
) -> Result<(Vec<f64>, Vec<Vec<f64>>)>
where
    F: Fn(&[f64], &[f64]) -> (Vec<f64>, Vec<f64>),
{
    if init.is_empty() {
        return Err(AutogradError::invalid_argument(
            "scan: initial carry must be non-empty".to_string(),
        ));
    }

    let mut carry = init;
    let mut outputs = Vec::with_capacity(xs.len());

    for (step, x) in xs.iter().enumerate() {
        if x.is_empty() {
            return Err(AutogradError::invalid_argument(format!(
                "scan: input xs[{step}] is empty"
            )));
        }

        let (new_carry, y) = f(&carry, x);

        if new_carry.is_empty() {
            return Err(AutogradError::invalid_argument(format!(
                "scan: step function returned empty carry at step {step}"
            )));
        }
        if y.is_empty() {
            return Err(AutogradError::invalid_argument(format!(
                "scan: step function returned empty output at step {step}"
            )));
        }

        carry = new_carry;
        outputs.push(y);
    }

    Ok((carry, outputs))
}

// ============================================================================
// grad_of_grad — higher-order gradient (Jacobian of the gradient)
// ============================================================================

/// Compute the gradient of the gradient: `∇(∇f)` evaluated at `x`.
///
/// This is the Hessian diagonal obtained by differentiating each partial
/// derivative once more.  It is numerically equivalent to the diagonal of the
/// Hessian matrix but avoids computing the full Hessian.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional_api::grad_of_grad;
///
/// // f(x, y) = x^3 + y^3; ∇f = [3x^2, 3y^2]; ∇(∇f)[0] = 6x at (2, 1) => 12
/// let hdiag = grad_of_grad(|xs: &[f64]| xs[0].powi(3) + xs[1].powi(3), &[2.0, 1.0])
///     .expect("grad_of_grad");
/// assert!((hdiag[0] - 12.0).abs() < 1e-2);
/// assert!((hdiag[1] -  6.0).abs() < 1e-2);
/// ```
pub fn grad_of_grad<F>(f: F, x: &[f64]) -> Result<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    if n == 0 {
        return Err(AutogradError::invalid_argument(
            "grad_of_grad: input must be non-empty".to_string(),
        ));
    }

    // Second-order diagonal via central differences: f''(xᵢ) ≈ (f(x+hᵢ) - 2f(x) + f(x-hᵢ)) / h²
    // Optimal step for 2nd derivative FD: h ≈ eps^(1/4) ≈ 1.2e-4 (avoids catastrophic cancellation)
    const H2: f64 = 1e-4;
    let f0 = f(x);
    let h2_sq = H2 * H2;
    let mut hdiag = vec![0.0f64; n];
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();

    for i in 0..n {
        xp[i] += H2;
        xm[i] -= H2;
        hdiag[i] = (f(&xp) - 2.0 * f0 + f(&xm)) / h2_sq;
        xp[i] = x[i];
        xm[i] = x[i];
    }

    Ok(hdiag)
}

// ============================================================================
// compose — function composition
// ============================================================================

/// Compose two functions: `compose(g, f)(x) = g(f(x))`.
///
/// Returns a closure that first applies `f`, then `g`.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional_api::compose;
///
/// let double = |xs: &[f64]| xs.iter().map(|&x| x * 2.0).collect::<Vec<_>>();
/// let plus_one = |xs: &[f64]| xs.iter().map(|&x| x + 1.0).collect::<Vec<_>>();
/// let composed = compose(plus_one, double);  // x -> x*2 -> x*2+1
/// let result = composed(&[3.0, 5.0]);
/// assert!((result[0] - 7.0).abs() < 1e-12);  // 3*2+1
/// assert!((result[1] - 11.0).abs() < 1e-12); // 5*2+1
/// ```
pub fn compose<F, G>(g: G, f: F) -> impl Fn(&[f64]) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
    G: Fn(&[f64]) -> Vec<f64>,
{
    move |x: &[f64]| {
        let fx = f(x);
        g(&fx)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // grad
    // -----------------------------------------------------------------------

    #[test]
    fn test_grad_quadratic() {
        // f(x, y) = x^2 + y^2; grad = [2x, 2y] at (3, 4) => [6, 8]
        let gf = grad(|xs: &[f64]| xs[0] * xs[0] + xs[1] * xs[1]);
        let g = gf(&[3.0, 4.0]).expect("grad quadratic");
        assert!((g[0] - 6.0).abs() < 1e-4, "g[0] = {}", g[0]);
        assert!((g[1] - 8.0).abs() < 1e-4, "g[1] = {}", g[1]);
    }

    #[test]
    fn test_grad_single_input() {
        // f(x) = x^3; grad = 3x^2 at x=2 => 12
        let gf = grad(|xs: &[f64]| xs[0] * xs[0] * xs[0]);
        let g = gf(&[2.0]).expect("grad single");
        assert!((g[0] - 12.0).abs() < 1e-3, "g[0] = {}", g[0]);
    }

    #[test]
    fn test_grad_empty_input_error() {
        let gf = grad(|_xs: &[f64]| 0.0_f64);
        assert!(gf(&[]).is_err());
    }

    #[test]
    fn test_grad_exp() {
        // f(x) = exp(x); grad = exp(x) at x=0 => 1
        let gf = grad(|xs: &[f64]| xs[0].exp());
        let g = gf(&[0.0]).expect("grad exp");
        assert!((g[0] - 1.0).abs() < 1e-4, "g[0] = {}", g[0]);
    }

    // -----------------------------------------------------------------------
    // value_and_grad
    // -----------------------------------------------------------------------

    #[test]
    fn test_value_and_grad_basic() {
        let vg = value_and_grad(|xs: &[f64]| xs[0] * xs[0] + xs[1] * xs[1]);
        let (val, g) = vg(&[3.0, 4.0]).expect("value_and_grad");
        assert!((val - 25.0).abs() < 1e-12, "val = {val}");
        assert!((g[0] - 6.0).abs() < 1e-4, "g[0] = {}", g[0]);
        assert!((g[1] - 8.0).abs() < 1e-4, "g[1] = {}", g[1]);
    }

    #[test]
    fn test_value_and_grad_empty_error() {
        let vg = value_and_grad(|_: &[f64]| 0.0_f64);
        assert!(vg(&[]).is_err());
    }

    #[test]
    fn test_value_matches_direct_eval() {
        let f = |xs: &[f64]| xs[0].sin() + xs[1].cos();
        let vg = value_and_grad(|xs: &[f64]| xs[0].sin() + xs[1].cos());
        let (val, _g) = vg(&[1.0, 2.0]).expect("value_and_grad sin/cos");
        let expected = f(&[1.0, 2.0]);
        assert!((val - expected).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // vmap
    // -----------------------------------------------------------------------

    #[test]
    fn test_vmap_basic() {
        let double = vmap(|xs: &[f64]| vec![xs[0] * 2.0, xs[1] * 3.0]);
        let batch = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let result = double(batch).expect("vmap basic");
        assert_eq!(result.len(), 3);
        assert!((result[0][0] - 2.0).abs() < 1e-12);
        assert!((result[0][1] - 6.0).abs() < 1e-12);
        assert!((result[1][0] - 6.0).abs() < 1e-12);
        assert!((result[1][1] - 12.0).abs() < 1e-12);
        assert!((result[2][0] - 10.0).abs() < 1e-12);
        assert!((result[2][1] - 18.0).abs() < 1e-12);
    }

    #[test]
    fn test_vmap_empty_batch_error() {
        let f_vm = vmap(|xs: &[f64]| xs.to_vec());
        assert!(f_vm(vec![]).is_err());
    }

    #[test]
    fn test_vmap_single_input() {
        let sq = vmap(|xs: &[f64]| vec![xs[0] * xs[0]]);
        let result = sq(vec![vec![2.0], vec![3.0]]).expect("vmap sq");
        assert!((result[0][0] - 4.0).abs() < 1e-12);
        assert!((result[1][0] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_vmap_grad_over_batch() {
        // Vectorise a gradient computation over a batch
        let gf = |xs: &[f64]| {
            // ∇(x^2 + y^2) = [2x, 2y]
            let n = xs.len();
            let mut g = vec![0.0f64; n];
            let mut xp = xs.to_vec();
            let mut xm = xs.to_vec();
            let f = |v: &[f64]| v.iter().map(|&vi| vi * vi).sum::<f64>();
            for i in 0..n {
                xp[i] += FD_H;
                xm[i] -= FD_H;
                g[i] = (f(&xp) - f(&xm)) / (2.0 * FD_H);
                xp[i] = xs[i];
                xm[i] = xs[i];
            }
            g
        };

        let batched_grad = vmap(gf);
        let batch = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let grads = batched_grad(batch).expect("batched grad");

        assert!((grads[0][0] - 2.0).abs() < 1e-3, "g[0][0] = {}", grads[0][0]);
        assert!((grads[0][1] - 4.0).abs() < 1e-3, "g[0][1] = {}", grads[0][1]);
        assert!((grads[1][0] - 6.0).abs() < 1e-3, "g[1][0] = {}", grads[1][0]);
        assert!((grads[1][1] - 8.0).abs() < 1e-3, "g[1][1] = {}", grads[1][1]);
    }

    // -----------------------------------------------------------------------
    // scan
    // -----------------------------------------------------------------------

    #[test]
    fn test_scan_cumulative_sum() {
        // carry = running sum; y = x * 2
        let (final_carry, ys) = scan(
            |carry: &[f64], x: &[f64]| {
                let new_carry = vec![carry[0] + x[0]];
                let y = vec![x[0] * 2.0];
                (new_carry, y)
            },
            vec![0.0],
            &[vec![1.0], vec![2.0], vec![3.0]],
        )
        .expect("scan cumulative sum");

        assert!((final_carry[0] - 6.0).abs() < 1e-12); // 1+2+3
        assert_eq!(ys.len(), 3);
        assert!((ys[0][0] - 2.0).abs() < 1e-12);
        assert!((ys[1][0] - 4.0).abs() < 1e-12);
        assert!((ys[2][0] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_scan_empty_sequence() {
        let (final_carry, ys) = scan(
            |carry: &[f64], x: &[f64]| (carry.to_vec(), x.to_vec()),
            vec![1.0, 2.0],
            &[],
        )
        .expect("scan empty");
        assert_eq!(final_carry, vec![1.0, 2.0]);
        assert!(ys.is_empty());
    }

    #[test]
    fn test_scan_single_step() {
        let (c, ys) = scan(
            |carry: &[f64], x: &[f64]| {
                let nc = vec![carry[0] * x[0]];
                let y = vec![carry[0] + x[0]];
                (nc, y)
            },
            vec![3.0],
            &[vec![2.0]],
        )
        .expect("scan single");
        assert!((c[0] - 6.0).abs() < 1e-12); // 3*2
        assert!((ys[0][0] - 5.0).abs() < 1e-12); // 3+2
    }

    #[test]
    fn test_scan_multi_dim_carry() {
        // Carry = [sum, product]; y = carry[0] (running sum)
        let (final_carry, ys) = scan(
            |carry: &[f64], x: &[f64]| {
                let s = carry[0] + x[0];
                let p = carry[1] * x[0];
                let y = vec![s];
                (vec![s, p], y)
            },
            vec![0.0, 1.0],
            &[vec![2.0], vec![3.0], vec![4.0]],
        )
        .expect("scan multi-dim carry");

        assert!((final_carry[0] - 9.0).abs() < 1e-12); // 0+2+3+4
        assert!((final_carry[1] - 24.0).abs() < 1e-12); // 1*2*3*4
        assert!((ys[0][0] - 2.0).abs() < 1e-12);
        assert!((ys[1][0] - 5.0).abs() < 1e-12);
        assert!((ys[2][0] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_scan_empty_carry_error() {
        let result = scan(
            |carry: &[f64], x: &[f64]| (carry.to_vec(), x.to_vec()),
            vec![],
            &[vec![1.0]],
        );
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // grad_of_grad
    // -----------------------------------------------------------------------

    #[test]
    fn test_grad_of_grad_cubic() {
        // f = x^3; f'' = 6x; at x=2 => 12
        let hd = grad_of_grad(|xs: &[f64]| xs[0].powi(3), &[2.0]).expect("grad_of_grad cubic");
        assert!((hd[0] - 12.0).abs() < 1e-1, "hd[0] = {}", hd[0]);
    }

    #[test]
    fn test_grad_of_grad_quadratic() {
        // f = x^2 + y^2; f'' diagonal = [2, 2]
        let hd =
            grad_of_grad(|xs: &[f64]| xs[0].powi(2) + xs[1].powi(2), &[1.0, 1.0])
                .expect("grad_of_grad quad");
        assert!((hd[0] - 2.0).abs() < 1e-2, "hd[0] = {}", hd[0]);
        assert!((hd[1] - 2.0).abs() < 1e-2, "hd[1] = {}", hd[1]);
    }

    // -----------------------------------------------------------------------
    // compose
    // -----------------------------------------------------------------------

    #[test]
    fn test_compose_double_then_plus_one() {
        let double = |xs: &[f64]| xs.iter().map(|&x| x * 2.0).collect::<Vec<_>>();
        let plus_one = |xs: &[f64]| xs.iter().map(|&x| x + 1.0).collect::<Vec<_>>();
        let f = compose(plus_one, double); // first double, then +1
        let r = f(&[3.0, 5.0]);
        assert!((r[0] - 7.0).abs() < 1e-12); // 3*2+1
        assert!((r[1] - 11.0).abs() < 1e-12); // 5*2+1
    }

    #[test]
    fn test_compose_identity() {
        let id: fn(&[f64]) -> Vec<f64> = |xs| xs.to_vec();
        let f = compose(id, id);
        let r = f(&[1.0, 2.0, 3.0]);
        assert_eq!(r, vec![1.0, 2.0, 3.0]);
    }
}
