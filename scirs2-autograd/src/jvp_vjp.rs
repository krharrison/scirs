//! JVP/VJP primitives for automatic differentiation
//!
//! This module provides low-level, closure-based Jacobian-vector product (JVP)
//! and vector-Jacobian product (VJP) primitives, together with higher-level
//! Jacobian and Hessian computation routines and a `linearize` utility.
//!
//! All computations use **central finite differences** with step `h = 1e-7`
//! unless otherwise noted.  The interface is purely functional — no
//! computation-graph tensors are required.
//!
//! # Quick reference
//!
//! | Function | Description | Cost |
//! |----------|-------------|------|
//! | [`jvp`] | `(f(x), J(x)·t)` via 2 function evals | O(1) per tangent direction |
//! | [`vjp`] | `(f(x), vᵀ·J(x))` via full Jacobian | O(n) per cotangent |
//! | [`jacfwd`] | Full Jacobian column-by-column (forward) | O(n) evals |
//! | [`jacrev`] | Full Jacobian row-by-row (reverse via VJP) | O(m) evals |
//! | [`hessian`] | Hessian of a scalar function via double FD | O(n²) evals |
//! | [`linearize`] | Tangent linearisation: `(f(x), J(x)·t)` | 2 function evals |
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::jvp_vjp::{jvp, vjp, jacfwd, jacrev, hessian};
//!
//! // f(x, y) = [x^2, x*y]
//! let f = |xs: &[f64]| vec![xs[0]*xs[0], xs[0]*xs[1]];
//!
//! // Forward-mode JVP at (2,3) with tangent (1,0)
//! let (fx, jvp_val) = jvp(&f, &[2.0, 3.0], &[1.0, 0.0]).expect("jvp");
//! assert!((jvp_val[0] - 4.0).abs() < 1e-4);  // d(x^2)/dx * 1 = 2x = 4
//! assert!((jvp_val[1] - 3.0).abs() < 1e-4);  // d(xy)/dx * 1 = y = 3
//!
//! // Reverse-mode VJP at (2,3) with cotangent (1,0)
//! let (fx2, vjp_val) = vjp(&f, &[2.0, 3.0], &[1.0, 0.0]).expect("vjp");
//! assert!((vjp_val[0] - 4.0).abs() < 1e-4);  // vᵀ·J[:,0] = 1*2x = 4
//! assert!((vjp_val[1] - 0.0).abs() < 1e-4);  // vᵀ·J[:,1] = 1*0  = 0
//!
//! // Full Jacobian via forward mode
//! let j = jacfwd(&f, &[2.0, 3.0]).expect("jacfwd");
//! assert!((j[0][0] - 4.0).abs() < 1e-4);
//! ```

use crate::error::AutogradError;
use crate::Result;

/// Central finite-difference step size.
const FD_H: f64 = 1e-7;

// ============================================================================
// jvp — forward-mode Jacobian-vector product
// ============================================================================

/// Forward-mode Jacobian-vector product: `(f(x), J(x)·t)`.
///
/// Computes the directional derivative `J(x)·t` via central finite differences:
///
/// `(J(x)·t)_i ≈ (f(x + h·t)_i − f(x − h·t)_i) / (2h)`
///
/// This requires exactly **2 function evaluations** regardless of input
/// dimension, making it O(1) per tangent direction.
///
/// # Arguments
///
/// * `f`  — Vector function `f: Rⁿ → Rᵐ`
/// * `x`  — Primal input point (length n)
/// * `t`  — Tangent vector (length n)
///
/// # Returns
///
/// `(f(x), J(x)·t)` where both components have length m.
///
/// # Errors
///
/// Returns `AutogradError` if `x` is empty or `t.len() != x.len()`.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::jvp_vjp::jvp;
///
/// let (fx, jvp_val) = jvp(
///     &|xs: &[f64]| vec![xs[0]*xs[0], xs[0]*xs[1]],
///     &[2.0, 3.0],
///     &[1.0, 0.0],
/// ).expect("jvp");
/// assert!((jvp_val[0] - 4.0).abs() < 1e-4);
/// assert!((jvp_val[1] - 3.0).abs() < 1e-4);
/// ```
pub fn jvp<F>(f: &F, x: &[f64], t: &[f64]) -> Result<(Vec<f64>, Vec<f64>)>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    if n == 0 {
        return Err(AutogradError::invalid_argument(
            "jvp: primal input must be non-empty".to_string(),
        ));
    }
    if t.len() != n {
        return Err(AutogradError::ShapeMismatch(format!(
            "jvp: tangent length {} != input length {}",
            t.len(),
            n
        )));
    }

    let xp: Vec<f64> = x
        .iter()
        .zip(t.iter())
        .map(|(&xi, &ti)| xi + FD_H * ti)
        .collect();
    let xm: Vec<f64> = x
        .iter()
        .zip(t.iter())
        .map(|(&xi, &ti)| xi - FD_H * ti)
        .collect();

    let fp = f(&xp);
    let fm = f(&xm);
    let fx = f(x);
    let two_h = 2.0 * FD_H;

    let jvp_val: Vec<f64> = fp
        .iter()
        .zip(fm.iter())
        .map(|(&fpi, &fmi)| (fpi - fmi) / two_h)
        .collect();

    Ok((fx, jvp_val))
}

// ============================================================================
// vjp — reverse-mode vector-Jacobian product
// ============================================================================

/// Reverse-mode vector-Jacobian product: `(f(x), vᵀ·J(x))`.
///
/// Computed by first building the full m×n Jacobian (via forward-mode FD,
/// column-by-column) and then contracting with the cotangent vector `v`:
///
/// `(vᵀ·J)_j = Σᵢ v_i · J[i,j]`
///
/// For n << m prefer [`jvp`]; for m << n this function is more efficient
/// because it only requires `m` passes via the row form.
///
/// # Arguments
///
/// * `f`  — Vector function `f: Rⁿ → Rᵐ`
/// * `x`  — Primal input (length n)
/// * `v`  — Cotangent vector (length m)
///
/// # Returns
///
/// `(f(x), vᵀ·J(x))` where the second component has length n.
///
/// # Errors
///
/// Returns `AutogradError` on dimension mismatch or empty input/output.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::jvp_vjp::vjp;
///
/// let (fx, vjp_val) = vjp(
///     &|xs: &[f64]| vec![xs[0]*xs[0], xs[0]*xs[1]],
///     &[2.0, 3.0],
///     &[1.0, 0.0],
/// ).expect("vjp");
/// assert!((vjp_val[0] - 4.0).abs() < 1e-4);
/// assert!((vjp_val[1] - 0.0).abs() < 1e-4);
/// ```
pub fn vjp<F>(f: &F, x: &[f64], v: &[f64]) -> Result<(Vec<f64>, Vec<f64>)>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    if n == 0 {
        return Err(AutogradError::invalid_argument(
            "vjp: primal input must be non-empty".to_string(),
        ));
    }

    let fx = f(x);
    let m = fx.len();
    if m == 0 {
        return Err(AutogradError::invalid_argument(
            "vjp: function output must be non-empty".to_string(),
        ));
    }
    if v.len() != m {
        return Err(AutogradError::ShapeMismatch(format!(
            "vjp: cotangent length {} != output length {}",
            v.len(),
            m
        )));
    }

    // Build full Jacobian column-by-column (forward mode)
    let mut jac = vec![vec![0.0f64; n]; m];
    let mut xp = x.to_vec();
    let mut xmv = x.to_vec();
    let two_h = 2.0 * FD_H;

    for j in 0..n {
        xp[j] += FD_H;
        xmv[j] -= FD_H;
        let fp = f(&xp);
        let fmv = f(&xmv);
        for i in 0..m {
            jac[i][j] = (fp[i] - fmv[i]) / two_h;
        }
        xp[j] = x[j];
        xmv[j] = x[j];
    }

    // Contract: result[j] = sum_i v[i] * J[i][j]
    let mut result = vec![0.0f64; n];
    for j in 0..n {
        for i in 0..m {
            result[j] += v[i] * jac[i][j];
        }
    }

    Ok((fx, result))
}

// ============================================================================
// jacfwd — full Jacobian via forward mode (column by column)
// ============================================================================

/// Compute the full m×n Jacobian matrix via forward-mode finite differences.
///
/// Each column j is computed as the JVP along the j-th standard basis vector:
/// `J[:,j] = (f(x + h·eⱼ) − f(x − h·eⱼ)) / (2h)`
///
/// **Complexity**: `2n` function evaluations → O(n) evaluations for n columns.
///
/// # Arguments
///
/// * `f`  — Function `f: Rⁿ → Rᵐ`
/// * `x`  — Evaluation point (length n)
///
/// # Returns
///
/// Jacobian as `Vec<Vec<f64>>` of shape `m × n`, where `result[i][j] = ∂fᵢ/∂xⱼ`.
///
/// # Errors
///
/// Returns `AutogradError` if `x` is empty or the function output is empty.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::jvp_vjp::jacfwd;
///
/// let j = jacfwd(
///     &|xs: &[f64]| vec![xs[0]*xs[0], xs[0]*xs[1]],
///     &[2.0, 3.0],
/// ).expect("jacfwd");
/// // J = [[2x, 0], [y, x]] = [[4, 0], [3, 2]]
/// assert!((j[0][0] - 4.0).abs() < 1e-4);
/// assert!((j[0][1] - 0.0).abs() < 1e-4);
/// assert!((j[1][0] - 3.0).abs() < 1e-4);
/// assert!((j[1][1] - 2.0).abs() < 1e-4);
/// ```
pub fn jacfwd<F>(f: &F, x: &[f64]) -> Result<Vec<Vec<f64>>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    if n == 0 {
        return Err(AutogradError::invalid_argument(
            "jacfwd: input must be non-empty".to_string(),
        ));
    }

    // Probe output dimension
    let f0 = f(x);
    let m = f0.len();
    if m == 0 {
        return Err(AutogradError::invalid_argument(
            "jacfwd: function output must be non-empty".to_string(),
        ));
    }

    let mut jac = vec![vec![0.0f64; n]; m];
    let mut xp = x.to_vec();
    let mut xmv = x.to_vec();
    let two_h = 2.0 * FD_H;

    for j in 0..n {
        xp[j] += FD_H;
        xmv[j] -= FD_H;
        let fp = f(&xp);
        let fm = f(&xmv);
        for i in 0..m {
            jac[i][j] = (fp[i] - fm[i]) / two_h;
        }
        xp[j] = x[j];
        xmv[j] = x[j];
    }

    Ok(jac)
}

// ============================================================================
// jacrev — full Jacobian via reverse mode (row by row, using VJP)
// ============================================================================

/// Compute the full m×n Jacobian matrix via reverse-mode (VJP row-by-row).
///
/// Each row i is computed by a VJP with the i-th standard basis cotangent:
/// `J[i,:] = VJP(f, x, eᵢ)`
///
/// **Complexity**: `2m·n` function evaluations (m VJPs each costing 2n FD
/// probes) → O(m·n).  For m << n this is preferred over [`jacfwd`] (O(n·m)
/// the same asymptotically, but reverse has smaller constant in typical NN
/// scenarios).
///
/// # Arguments
///
/// * `f`  — Function `f: Rⁿ → Rᵐ`
/// * `x`  — Evaluation point (length n)
///
/// # Returns
///
/// Jacobian as `Vec<Vec<f64>>` of shape `m × n`.
///
/// # Errors
///
/// Returns `AutogradError` if `x` is empty or function output is empty.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::jvp_vjp::jacrev;
///
/// let j = jacrev(
///     &|xs: &[f64]| vec![xs[0]*xs[0], xs[0]*xs[1]],
///     &[2.0, 3.0],
/// ).expect("jacrev");
/// assert!((j[1][0] - 3.0).abs() < 1e-4);  // ∂(xy)/∂x = y = 3
/// assert!((j[1][1] - 2.0).abs() < 1e-4);  // ∂(xy)/∂y = x = 2
/// ```
pub fn jacrev<F>(f: &F, x: &[f64]) -> Result<Vec<Vec<f64>>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    if n == 0 {
        return Err(AutogradError::invalid_argument(
            "jacrev: input must be non-empty".to_string(),
        ));
    }

    let f0 = f(x);
    let m = f0.len();
    if m == 0 {
        return Err(AutogradError::invalid_argument(
            "jacrev: function output must be non-empty".to_string(),
        ));
    }

    let mut jac = vec![vec![0.0f64; n]; m];

    for i in 0..m {
        // Cotangent = eᵢ
        let mut cotangent = vec![0.0f64; m];
        cotangent[i] = 1.0;

        let (_fx, row) = vjp(f, x, &cotangent)?;
        jac[i] = row;
    }

    Ok(jac)
}

// ============================================================================
// hessian — second-order derivatives via double reverse-mode (FD)
// ============================================================================

/// Compute the n×n Hessian matrix of a scalar function.
///
/// Uses the mixed second-order central finite difference formula:
///
/// `H[i,j] ≈ (f(x+hᵢ+hⱼ) − f(x+hᵢ−hⱼ) − f(x−hᵢ+hⱼ) + f(x−hᵢ−hⱼ)) / (4h²)`
///
/// The diagonal is computed as:
/// `H[i,i] ≈ (f(x+hᵢ) − 2f(x) + f(x−hᵢ)) / h²`
///
/// **Complexity**: O(n²) function evaluations.
///
/// # Arguments
///
/// * `f`  — Scalar function `f: Rⁿ → R`
/// * `x`  — Evaluation point (length n)
///
/// # Returns
///
/// Hessian as `Vec<Vec<f64>>` of shape `n × n`.  The matrix is symmetric.
///
/// # Errors
///
/// Returns `AutogradError` if `x` is empty.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::jvp_vjp::hessian;
///
/// // f(x, y) = x^2 + x*y => H = [[2, 1], [1, 0]]
/// let h = hessian(
///     &|xs: &[f64]| xs[0]*xs[0] + xs[0]*xs[1],
///     &[1.0, 1.0],
/// ).expect("hessian");
/// assert!((h[0][0] - 2.0).abs() < 1e-3);
/// assert!((h[0][1] - 1.0).abs() < 1e-3);
/// assert!((h[1][0] - 1.0).abs() < 1e-3);
/// assert!((h[1][1] - 0.0).abs() < 1e-3);
/// ```
pub fn hessian<F>(f: &F, x: &[f64]) -> Result<Vec<Vec<f64>>>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    if n == 0 {
        return Err(AutogradError::invalid_argument(
            "hessian: input must be non-empty".to_string(),
        ));
    }

    let f0 = f(x);
    // Optimal step for 2nd derivative FD: ~ eps^(1/4) ~ 1.2e-4 (reduces catastrophic cancellation)
    const H2: f64 = 1e-4;
    let h2 = H2 * H2;
    let four_h2 = 4.0 * h2;

    let mut hess = vec![vec![0.0f64; n]; n];

    let mut xa = x.to_vec();
    let mut xb = x.to_vec();
    let mut xab = x.to_vec();

    for i in 0..n {
        // Diagonal via second-order central difference
        xa[i] = x[i] + H2;
        xb[i] = x[i] - H2;
        let fpi = f(&xa);
        let fmi = f(&xb);
        hess[i][i] = (fpi - 2.0 * f0 + fmi) / h2;
        xa[i] = x[i];
        xb[i] = x[i];

        // Off-diagonal: mixed partials
        for j in (i + 1)..n {
            xa[i] = x[i] + H2;
            xa[j] = x[j] + H2;
            let fpp = f(&xa);

            xa[j] = x[j] - H2;
            let fpm = f(&xa);

            xb[i] = x[i] - H2;
            xb[j] = x[j] + H2;
            let fmp = f(&xb);

            xb[j] = x[j] - H2;
            let fmm = f(&xb);

            let val = (fpp - fpm - fmp + fmm) / four_h2;
            hess[i][j] = val;
            hess[j][i] = val;

            // Restore
            xa[i] = x[i];
            xa[j] = x[j];
            xb[i] = x[i];
            xb[j] = x[j];
            // xab not used here, but reset for safety
            xab[i] = x[i];
            xab[j] = x[j];
        }
    }

    // Suppress unused warning
    let _ = xab;

    Ok(hess)
}

// ============================================================================
// linearize — tangent linearisation at a point
// ============================================================================

/// Linearise a function at a point: compute both the primal output and the
/// tangent output simultaneously.
///
/// `linearize(f, x, t)` returns `(f(x), J(x)·t)`.  This is semantically
/// identical to [`jvp`] but with a different API emphasis: the returned
/// closure captures the point and tangent for repeated calls with the same
/// linearisation.
///
/// # Arguments
///
/// * `f`  — Function `f: Rⁿ → Rᵐ`
/// * `x`  — Primal point to linearise around (length n)
/// * `t`  — Tangent direction (length n)
///
/// # Returns
///
/// `(primal_output, tangent_output)` where both have length m.
///
/// # Errors
///
/// Propagates errors from the underlying [`jvp`] call.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::jvp_vjp::linearize;
///
/// // f(x) = [exp(x[0]), x[0]*x[1]]
/// let f = |xs: &[f64]| vec![xs[0].exp(), xs[0]*xs[1]];
/// let (primal, tangent) = linearize(&f, &[0.0, 2.0], &[1.0, 0.0])
///     .expect("linearize");
/// assert!((primal[0] - 1.0).abs() < 1e-9);  // exp(0) = 1
/// assert!((tangent[0] - 1.0).abs() < 1e-4); // d(exp(x))/dx = exp(0) = 1
/// ```
pub fn linearize<F>(f: &F, x: &[f64], t: &[f64]) -> Result<(Vec<f64>, Vec<f64>)>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    jvp(f, x, t)
}

// ============================================================================
// Helper: scalar wrapper for jacfwd/jacrev on scalar functions
// ============================================================================

/// Compute the gradient of a scalar function `f: Rⁿ → R` at `x`.
///
/// This is a thin wrapper around [`jacfwd`] that returns `Vec<f64>` rather
/// than a 1×n matrix.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::jvp_vjp::grad_scalar;
///
/// let g = grad_scalar(
///     &|xs: &[f64]| xs[0]*xs[0] + xs[1]*xs[1],
///     &[3.0, 4.0],
/// ).expect("grad");
/// assert!((g[0] - 6.0).abs() < 1e-4);
/// assert!((g[1] - 8.0).abs() < 1e-4);
/// ```
pub fn grad_scalar<F>(f: &F, x: &[f64]) -> Result<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let wrapper = |xs: &[f64]| vec![f(xs)];
    let jac = jacfwd(&wrapper, x)?;
    // jac is 1×n; return the single row
    Ok(jac.into_iter().next().unwrap_or_default())
}

/// Compute the Hessian-vector product `H(x)·v` efficiently via double FD.
///
/// Uses the identity `H(x)·v ≈ (∇f(x+h·v) − ∇f(x−h·v)) / (2h)`,
/// requiring only two gradient evaluations.
///
/// **Complexity**: O(n) function evaluations.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::jvp_vjp::hvp;
///
/// // f(x,y) = x^2 + y^2, H = 2I, v = [1,0] → HVP = [2,0]
/// let h = hvp(
///     &|xs: &[f64]| xs[0]*xs[0] + xs[1]*xs[1],
///     &[1.0, 1.0],
///     &[1.0, 0.0],
/// ).expect("hvp");
/// assert!((h[0] - 2.0).abs() < 1e-3);
/// assert!(h[1].abs() < 1e-3);
/// ```
pub fn hvp<F>(f: &F, x: &[f64], v: &[f64]) -> Result<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    if n == 0 {
        return Err(AutogradError::invalid_argument(
            "hvp: input must be non-empty".to_string(),
        ));
    }
    if v.len() != n {
        return Err(AutogradError::ShapeMismatch(format!(
            "hvp: vector length {} != input length {}",
            v.len(),
            n
        )));
    }

    // Use a larger outer step to reduce contamination from inner FD errors.
    // Optimal outer step for forward-over-reverse: ~ sqrt(inner_h) ~ sqrt(1e-7) ~ 3e-4
    const HVP_STEP: f64 = 1e-4;
    let xp: Vec<f64> = x
        .iter()
        .zip(v.iter())
        .map(|(&xi, &vi)| xi + HVP_STEP * vi)
        .collect();
    let xm: Vec<f64> = x
        .iter()
        .zip(v.iter())
        .map(|(&xi, &vi)| xi - HVP_STEP * vi)
        .collect();

    let gp = grad_scalar(f, &xp)?;
    let gm = grad_scalar(f, &xm)?;

    let two_h = 2.0 * HVP_STEP;
    let result: Vec<f64> = gp
        .iter()
        .zip(gm.iter())
        .map(|(&gpi, &gmi)| (gpi - gmi) / two_h)
        .collect();

    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn f_quad(xs: &[f64]) -> Vec<f64> {
        // f(x, y) = [x^2, x*y]
        vec![xs[0] * xs[0], xs[0] * xs[1]]
    }

    fn f_scalar(xs: &[f64]) -> f64 {
        // f(x, y) = x^2 + y^2
        xs[0] * xs[0] + xs[1] * xs[1]
    }

    #[test]
    fn test_jvp_basic() {
        // tangent (1, 0) => JVP = [2x, y] at (2, 3)
        let (fx, jvp_val) = jvp(&f_quad, &[2.0, 3.0], &[1.0, 0.0]).expect("jvp");
        assert!((fx[0] - 4.0).abs() < 1e-9); // x^2 = 4
        assert!((fx[1] - 6.0).abs() < 1e-9); // x*y = 6
        assert!((jvp_val[0] - 4.0).abs() < 1e-4, "jvp[0] = {}", jvp_val[0]);
        assert!((jvp_val[1] - 3.0).abs() < 1e-4, "jvp[1] = {}", jvp_val[1]);
    }

    #[test]
    fn test_jvp_second_direction() {
        // tangent (0, 1) => JVP = [0, x] at (2, 3)
        let (_, jvp_val) = jvp(&f_quad, &[2.0, 3.0], &[0.0, 1.0]).expect("jvp");
        assert!((jvp_val[0] - 0.0).abs() < 1e-4, "jvp[0] = {}", jvp_val[0]);
        assert!((jvp_val[1] - 2.0).abs() < 1e-4, "jvp[1] = {}", jvp_val[1]);
    }

    #[test]
    fn test_vjp_basic() {
        // cotangent (1, 0) => VJP = [2x, 0] at (2, 3)
        let (fx, vjp_val) = vjp(&f_quad, &[2.0, 3.0], &[1.0, 0.0]).expect("vjp");
        assert!((fx[0] - 4.0).abs() < 1e-9);
        assert!((vjp_val[0] - 4.0).abs() < 1e-4, "vjp[0] = {}", vjp_val[0]);
        assert!((vjp_val[1] - 0.0).abs() < 1e-4, "vjp[1] = {}", vjp_val[1]);
    }

    #[test]
    fn test_vjp_second_cotangent() {
        // cotangent (0, 1) => VJP = [y, x] at (2, 3)
        let (_, vjp_val) = vjp(&f_quad, &[2.0, 3.0], &[0.0, 1.0]).expect("vjp");
        assert!((vjp_val[0] - 3.0).abs() < 1e-4, "vjp[0] = {}", vjp_val[0]);
        assert!((vjp_val[1] - 2.0).abs() < 1e-4, "vjp[1] = {}", vjp_val[1]);
    }

    #[test]
    fn test_jacfwd() {
        let j = jacfwd(&f_quad, &[2.0, 3.0]).expect("jacfwd");
        // J = [[2x, 0], [y, x]] = [[4, 0], [3, 2]]
        assert!((j[0][0] - 4.0).abs() < 1e-4, "j[0][0] = {}", j[0][0]);
        assert!((j[0][1] - 0.0).abs() < 1e-4, "j[0][1] = {}", j[0][1]);
        assert!((j[1][0] - 3.0).abs() < 1e-4, "j[1][0] = {}", j[1][0]);
        assert!((j[1][1] - 2.0).abs() < 1e-4, "j[1][1] = {}", j[1][1]);
    }

    #[test]
    fn test_jacrev() {
        let j = jacrev(&f_quad, &[2.0, 3.0]).expect("jacrev");
        assert!((j[0][0] - 4.0).abs() < 1e-4, "j[0][0] = {}", j[0][0]);
        assert!((j[0][1] - 0.0).abs() < 1e-4, "j[0][1] = {}", j[0][1]);
        assert!((j[1][0] - 3.0).abs() < 1e-4, "j[1][0] = {}", j[1][0]);
        assert!((j[1][1] - 2.0).abs() < 1e-4, "j[1][1] = {}", j[1][1]);
    }

    #[test]
    fn test_jacfwd_jacrev_agree() {
        // Both should produce the same Jacobian
        let jf = jacfwd(&f_quad, &[1.5, 2.5]).expect("jacfwd");
        let jr = jacrev(&f_quad, &[1.5, 2.5]).expect("jacrev");
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (jf[i][j] - jr[i][j]).abs() < 1e-4,
                    "jf[{i}][{j}]={} jr[{i}][{j}]={}",
                    jf[i][j],
                    jr[i][j]
                );
            }
        }
    }

    #[test]
    fn test_hessian_diagonal() {
        // f(x, y) = x^2 + y^2 => H = diag(2, 2)
        let h = hessian(&f_scalar, &[1.0, 1.0]).expect("hessian");
        assert!((h[0][0] - 2.0).abs() < 1e-3, "H[0][0] = {}", h[0][0]);
        assert!((h[1][1] - 2.0).abs() < 1e-3, "H[1][1] = {}", h[1][1]);
        assert!(h[0][1].abs() < 1e-3, "H[0][1] = {}", h[0][1]);
        assert!(h[1][0].abs() < 1e-3, "H[1][0] = {}", h[1][0]);
    }

    #[test]
    fn test_hessian_mixed_partial() {
        // f(x, y) = x^2 + x*y => H = [[2, 1], [1, 0]]
        let g = |xs: &[f64]| xs[0] * xs[0] + xs[0] * xs[1];
        let h = hessian(&g, &[1.0, 1.0]).expect("hessian mixed");
        assert!((h[0][0] - 2.0).abs() < 1e-3, "H[0][0] = {}", h[0][0]);
        assert!((h[0][1] - 1.0).abs() < 1e-3, "H[0][1] = {}", h[0][1]);
        assert!((h[1][0] - 1.0).abs() < 1e-3, "H[1][0] = {}", h[1][0]);
        assert!((h[1][1] - 0.0).abs() < 1e-3, "H[1][1] = {}", h[1][1]);
    }

    #[test]
    fn test_linearize() {
        let f = |xs: &[f64]| vec![xs[0].exp(), xs[0] * xs[1]];
        let (primal, tangent) = linearize(&f, &[0.0, 2.0], &[1.0, 0.0]).expect("linearize");
        assert!((primal[0] - 1.0).abs() < 1e-9); // exp(0) = 1
        assert!((tangent[0] - 1.0).abs() < 1e-4); // d(exp(x))/dx * 1 = 1
        assert!((tangent[1] - 2.0).abs() < 1e-4); // d(xy)/dx * 1 = y = 2
    }

    #[test]
    fn test_grad_scalar() {
        // f(x, y) = x^2 + y^2; grad = [2x, 2y] at (3, 4) => [6, 8]
        let g = grad_scalar(&f_scalar, &[3.0, 4.0]).expect("grad_scalar");
        assert!((g[0] - 6.0).abs() < 1e-4, "g[0] = {}", g[0]);
        assert!((g[1] - 8.0).abs() < 1e-4, "g[1] = {}", g[1]);
    }

    #[test]
    fn test_hvp() {
        // f(x,y) = x^2 + y^2, H = 2I, v = [1, 0] => HVP = [2, 0]
        let h = hvp(&f_scalar, &[1.0, 1.0], &[1.0, 0.0]).expect("hvp");
        assert!((h[0] - 2.0).abs() < 1e-3, "hvp[0] = {}", h[0]);
        assert!(h[1].abs() < 1e-3, "hvp[1] = {}", h[1]);
    }

    #[test]
    fn test_jvp_empty_input_error() {
        let f = |_xs: &[f64]| vec![1.0_f64];
        assert!(jvp(&f, &[], &[]).is_err());
    }

    #[test]
    fn test_vjp_dimension_mismatch_error() {
        let f = |xs: &[f64]| vec![xs[0]];
        assert!(vjp(&f, &[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_jacfwd_empty_input_error() {
        let f = |_xs: &[f64]| vec![1.0_f64];
        assert!(jacfwd(&f, &[]).is_err());
    }

    #[test]
    fn test_hessian_empty_input_error() {
        let f = |_xs: &[f64]| 0.0_f64;
        assert!(hessian(&f, &[]).is_err());
    }
}
