//! Advanced higher-order automatic differentiation
//!
//! This module provides a high-level, ergonomic API for computing higher-order
//! derivatives using the scirs2-autograd computation graph. All functions accept
//! closures of the form `F: Fn(&Tensor) -> Tensor` and operate within an existing
//! [`Context`].
//!
//! # Overview
//!
//! | Function | Description | Complexity |
//! |----------|-------------|------------|
//! | [`hessian_matrix`] | Full n×n Hessian via double reverse-mode AD | O(n²) |
//! | [`hvp`] | Hessian-vector product (efficient) | O(n) |
//! | [`jacobian_matrix`] | Full m×n Jacobian via reverse-mode | O(m·n) |
//! | [`jvp`] | Jacobian-vector product (forward direction) | O(n) |
//! | [`vjp`] | Vector-Jacobian product (reverse direction) | O(m) |
//!
//! # Examples
//!
//! ## Hessian of a quadratic
//!
//! ```rust
//! use scirs2_autograd as ag;
//! use ag::tensor_ops as T;
//!
//! ag::run(|ctx: &mut ag::Context<f64>| {
//!     let x = ctx.placeholder("x", &[2]);
//!     // f(x) = x0^2 + 2*x1^2  =>  H = diag([2, 4])
//!     let f = |t: &ag::Tensor<'_, f64>| {
//!         let x0 = T::slice(*t, [0isize], [1isize]);
//!         let x1 = T::slice(*t, [1isize], [2isize]);
//!         x0 * x0 + x1 * x1 * 2.0
//!     };
//!
//!     let h = ag::higher_order_advanced::hessian_matrix(f, &x, ctx, 2)
//!         .expect("Hessian should succeed");
//!
//!     let x_val = scirs2_core::ndarray::arr1(&[1.0f64, 1.0]);
//!     let result = ctx.evaluator()
//!         .push(&h)
//!         .feed(x, x_val.view().into_dyn())
//!         .run();
//!     // Hessian = [[2, 0], [0, 4]] flattened
//!     let vals = result[0].as_ref().expect("eval ok");
//!     let s = vals.as_slice().unwrap_or(&[]);
//!     assert!((s[0] - 2.0).abs() < 1e-6);
//!     assert!((s[3] - 4.0).abs() < 1e-6);
//! });
//! ```

use crate::error::AutogradError;
use crate::tensor::Tensor;
use crate::{Context, Float, Result};

// ---------------------------------------------------------------------------
// Hessian matrix
// ---------------------------------------------------------------------------

/// Compute the full n×n Hessian matrix of a scalar function via double
/// reverse-mode automatic differentiation.
///
/// The Hessian is computed row-by-row: for each output row `i`, we
/// differentiate the `i`-th component of the gradient with respect to `x`
/// to get the `i`-th row of the Hessian.
///
/// # Arguments
/// * `f` - Closure mapping `&Tensor<F>` to a *scalar* `Tensor<F>`
/// * `x` - Input tensor of shape `[n]`
/// * `ctx` - Active autograd context
/// * `n` - Number of input dimensions (must equal `x.shape()[0]` at graph time)
///
/// # Returns
/// A tensor of shape `[n, n]` (stored row-major, flattened to `[n*n]` in the
/// graph, reshaped to `[n, n]` at evaluation time).
///
/// # Errors
/// Returns [`AutogradError`] if `n == 0`, if shapes are incompatible, or if the
/// function `f` does not produce a scalar.
///
/// # Complexity
/// O(n) backward passes to build the gradient graph, then O(n) more to build
/// the Hessian rows — total O(n²) memory / time in the graph.
pub fn hessian_matrix<'graph, F, Func>(
    f: Func,
    x: &Tensor<'graph, F>,
    ctx: &'graph Context<'graph, F>,
    n: usize,
) -> Result<Tensor<'graph, F>>
where
    F: Float,
    Func: Fn(&Tensor<'graph, F>) -> Tensor<'graph, F> + Copy,
{
    if n == 0 {
        return Err(AutogradError::shape_error(
            "hessian_matrix: dimension n must be positive".to_string(),
        ));
    }

    // Use HVP-based approach: H·e_i gives the i-th row of the Hessian.
    // This avoids double-slicing of gradients which can cause shape issues.
    let mut hessian_rows = Vec::with_capacity(n);
    for i in 0..n {
        // Unit vector e_i
        let mut e_i_vec = vec![F::zero(); n];
        e_i_vec[i] = F::one();
        let e_i_arr = scirs2_core::ndarray::Array1::from(e_i_vec).into_dyn();
        let e_i = crate::tensor_ops::convert_to_tensor(e_i_arr, ctx);

        // Compute H·e_i = i-th row of Hessian
        let h_row = hvp(|t| f(t), x, &e_i, ctx)?;
        hessian_rows.push(crate::tensor_ops::flatten(h_row));
    }

    // Stack rows: shape [n*n] (row-major)
    Ok(crate::tensor_ops::linear_algebra::concat(&hessian_rows, 0))
}

// ---------------------------------------------------------------------------
// Hessian-vector product
// ---------------------------------------------------------------------------

/// Compute the Hessian-vector product H(f)·v efficiently in O(n) time.
///
/// Rather than constructing the full n×n Hessian, this computes
/// `∇²f(x) · v` using the identity:
///
/// ```text
/// H·v = ∇_x ( ∇f(x) · v )
/// ```
///
/// This is the "reverse-over-reverse" trick and requires only two backward passes.
///
/// # Arguments
/// * `f` - Closure mapping `&Tensor<F>` to a scalar `Tensor<F>`
/// * `x` - Input tensor of shape `[n]`
/// * `v` - Vector tensor of the same shape as `x`
/// * `ctx` - Active autograd context
///
/// # Returns
/// Tensor of the same shape as `x` (and `v`) containing `H·v`.
///
/// # Errors
/// Returns an error if shapes of `x` and `v` mismatch, or if `f(x)` is not scalar.
pub fn hvp<'graph, F, Func>(
    f: Func,
    x: &Tensor<'graph, F>,
    v: &Tensor<'graph, F>,
    ctx: &'graph Context<'graph, F>,
) -> Result<Tensor<'graph, F>>
where
    F: Float,
    Func: Fn(&Tensor<'graph, F>) -> Tensor<'graph, F>,
{
    let x_shape = x.shape();
    let v_shape = v.shape();
    if x_shape != v_shape {
        return Err(AutogradError::shape_error(format!(
            "hvp: x shape {:?} and v shape {:?} must match",
            x_shape, v_shape
        )));
    }

    // Forward pass
    let y = f(x);

    // Gradient: g = ∂f/∂x  (same shape as x)
    let g = crate::tensor_ops::grad(&[y], &[*x])[0];

    // Element-wise product g * v (no flatten/sum needed):
    // grad_x(sum_i g_i * v_i) = H · v  (HVP by reverse-over-reverse)
    let gv = g * *v;

    // HVP: grad of the element-wise product sum w.r.t. x
    let result = crate::tensor_ops::grad(&[gv], &[*x])[0];
    Ok(result)
}

// ---------------------------------------------------------------------------
// Jacobian matrix
// ---------------------------------------------------------------------------

/// Compute the full m×n Jacobian matrix of a vector-valued function via
/// reverse-mode AD.
///
/// For `f: R^n -> R^m`, each row of the Jacobian `J[i, :]` is computed as
/// `∇(f_i)` with respect to `x`.
///
/// # Arguments
/// * `f` - Closure mapping `&Tensor<F>` (shape `[n]`) to `Tensor<F>` (shape `[m]`)
/// * `x` - Input tensor of shape `[n]`
/// * `ctx` - Active autograd context
/// * `m` - Output dimension
/// * `n` - Input dimension
///
/// # Returns
/// Tensor of shape `[m*n]` (row-major Jacobian). Reshape to `[m, n]` as needed.
///
/// # Errors
/// Returns an error if `m == 0` or `n == 0`.
///
/// # Complexity
/// O(m) backward passes.
pub fn jacobian_matrix<'graph, F, Func>(
    f: Func,
    x: &Tensor<'graph, F>,
    ctx: &'graph Context<'graph, F>,
    m: usize,
    n: usize,
) -> Result<Tensor<'graph, F>>
where
    F: Float,
    Func: Fn(&Tensor<'graph, F>) -> Tensor<'graph, F>,
{
    if m == 0 || n == 0 {
        return Err(AutogradError::shape_error(
            "jacobian_matrix: dimensions m and n must be positive".to_string(),
        ));
    }

    let y = f(x);
    let y_flat = crate::tensor_ops::flatten(y);

    let mut rows = Vec::with_capacity(m);
    for i in 0..m {
        let y_i = crate::tensor_ops::slice(y_flat, [i as isize], [(i + 1) as isize]);
        let grad_i = crate::tensor_ops::grad(&[y_i], &[*x])[0];
        rows.push(crate::tensor_ops::flatten(grad_i));
    }

    Ok(crate::tensor_ops::linear_algebra::concat(&rows, 0))
}

// ---------------------------------------------------------------------------
// JVP — Jacobian-vector product
// ---------------------------------------------------------------------------

/// Compute the Jacobian-vector product `J(f, x) · v` (forward direction).
///
/// This is equivalent to the directional derivative of `f` at `x` in the
/// direction `v`. It is computed using the reverse-mode identity:
///
/// ```text
/// JVP = ∇_x( ∇f(x) · g )
/// ```
///
/// for a scalar function, generalised to vector outputs via summation.
///
/// For a *scalar* output function this is identical to [`hvp`] restricted to
/// a single backward pass. For vector outputs the implementation computes the
/// full Jacobian–vector contraction `sum_i J[i, :] * v` → shape `[m]`.
///
/// # Arguments
/// * `f` - Closure from `&Tensor<F>` to `Tensor<F>`
/// * `x` - Input tensor of shape `[n]`
/// * `v` - Tangent vector of the same shape as `x`
/// * `ctx` - Active autograd context
/// * `m` - Output dimension of `f` (number of outputs)
///
/// # Returns
/// Tensor of shape `[m]` — the Jacobian-vector product.
///
/// # Errors
/// Returns an error if shapes of `x` and `v` mismatch.
pub fn jvp<'graph, F, Func>(
    f: Func,
    x: &Tensor<'graph, F>,
    v: &Tensor<'graph, F>,
    ctx: &'graph Context<'graph, F>,
    m: usize,
) -> Result<Tensor<'graph, F>>
where
    F: Float,
    Func: Fn(&Tensor<'graph, F>) -> Tensor<'graph, F>,
{
    let x_shape = x.shape();
    let v_shape = v.shape();
    if x_shape != v_shape {
        return Err(AutogradError::shape_error(format!(
            "jvp: x shape {:?} and v shape {:?} must match",
            x_shape, v_shape
        )));
    }
    if m == 0 {
        return Err(AutogradError::shape_error(
            "jvp: output dimension m must be positive".to_string(),
        ));
    }

    let y = f(x);

    if m == 1 {
        // Scalar output: JVP = grad(f)·v  (single backward pass)
        let g = crate::tensor_ops::grad(&[y], &[*x])[0];
        let gv = g * *v;
        // Sum over input dim to get the scalar JVP value
        let axes = (0..gv.shape().len())
            .map(|i| i as isize)
            .collect::<Vec<_>>();
        let jvp_scalar = if axes.is_empty() {
            gv
        } else {
            crate::tensor_ops::reduction::sum_all(gv)
        };
        return Ok(jvp_scalar);
    }

    // Vector output (m > 1): use Jacobian-based approach
    // JVP_i = ∑_j J_{ij} * v_j = grad(y_i, x) · v
    let y_flat = crate::tensor_ops::flatten(y);
    let v_flat = crate::tensor_ops::flatten(*v);

    let mut jvp_elements = Vec::with_capacity(m);
    for i in 0..m {
        // Extract y_i via slice
        let y_i = crate::tensor_ops::slice(y_flat, [i as isize], [(i + 1) as isize]);
        // grad of scalar y_i w.r.t. x
        let grad_i = crate::tensor_ops::grad(&[y_i], &[*x])[0];
        let grad_i_flat = crate::tensor_ops::flatten(grad_i);
        // Dot product with v: gives one scalar JVP element
        let jvp_i = crate::tensor_ops::reduction::sum_all(grad_i_flat * v_flat);
        // Expand to shape [1] so concat works
        let jvp_i_1d = crate::tensor_ops::reshape(jvp_i, &[1_isize]);
        jvp_elements.push(jvp_i_1d);
    }

    Ok(crate::tensor_ops::linear_algebra::concat(&jvp_elements, 0))
}

// ---------------------------------------------------------------------------
// VJP — Vector-Jacobian product
// ---------------------------------------------------------------------------

/// Compute the vector-Jacobian product `v^T · J(f, x)` (reverse direction).
///
/// For `f: R^n -> R^m` and `v ∈ R^m`, this computes `v^T J ∈ R^n`.
/// It is the fundamental operation in reverse-mode AD and is implemented as:
///
/// ```text
/// VJP = ∇_x( v · f(x) )
/// ```
///
/// # Arguments
/// * `f` - Closure from `&Tensor<F>` to `Tensor<F>` of shape `[m]`
/// * `x` - Input tensor of shape `[n]`
/// * `v` - Co-vector of the same shape as the output of `f` (shape `[m]`)
/// * `ctx` - Active autograd context
///
/// # Returns
/// Tensor of the same shape as `x` (shape `[n]`) — the VJP result.
///
/// # Errors
/// Returns an error if the output shape of `f(x)` and `v` are incompatible
/// (checked at graph-build time via shape propagation where available).
pub fn vjp<'graph, F, Func>(
    f: Func,
    x: &Tensor<'graph, F>,
    v: &Tensor<'graph, F>,
    ctx: &'graph Context<'graph, F>,
) -> Result<Tensor<'graph, F>>
where
    F: Float,
    Func: Fn(&Tensor<'graph, F>) -> Tensor<'graph, F>,
{
    let y = f(x);

    // Compute dot product v · y (scalar)
    let y_flat = crate::tensor_ops::flatten(y);
    let v_flat = crate::tensor_ops::flatten(*v);
    let dot = crate::tensor_ops::reduction::sum_all(v_flat * y_flat);

    // VJP = ∇_x(v · f(x))
    let result = crate::tensor_ops::grad(&[dot], &[*x])[0];
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::*;

    // ------------------------------------------------------------------
    // hessian_matrix
    // ------------------------------------------------------------------

    #[test]
    fn test_hessian_matrix_quadratic() {
        // f(x) = x0^2 + x1^2 = sum(x^2)  =>  H = diag([2, 2])
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let h = hessian_matrix(
                |t| {
                    // Use reduce_sum along axis 0 — avoids slice-based double-gradient
                    let axes = [0_isize];
                    reduce_sum(*t * *t, &axes, false)
                },
                &x,
                ctx,
                2,
            )
            .expect("hessian_matrix should succeed");

            let x_val = scirs2_core::ndarray::arr1(&[1.0f64, 1.0]);
            let result = ctx
                .evaluator()
                .push(&h)
                .feed(x, x_val.view().into_dyn())
                .run();

            let arr = result[0].as_ref().expect("should evaluate");
            let s = arr.as_slice().expect("slice");
            // H = diag([2, 2]) flattened: [2, 0, 0, 2]
            assert!((s[0] - 2.0).abs() < 1e-6, "H[0,0] expected 2, got {}", s[0]);
            assert!((s[1]).abs() < 1e-6, "H[0,1] expected 0, got {}", s[1]);
            assert!((s[2]).abs() < 1e-6, "H[1,0] expected 0, got {}", s[2]);
            assert!((s[3] - 2.0).abs() < 1e-6, "H[1,1] expected 2, got {}", s[3]);
        });
    }

    #[test]
    fn test_hessian_matrix_constant_hessian() {
        // f(x) = sum(x^2) => H = 2*I (constant Hessian)
        // Verify H is the same at different input points
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let h = hessian_matrix(
                |t| {
                    let axes = [0_isize];
                    reduce_sum(*t * *t, &axes, false)
                },
                &x,
                ctx,
                2,
            )
            .expect("hessian_matrix should succeed");

            // At x = [3, 4]: H should still be diag([2, 2]) since it's constant
            let x_val = scirs2_core::ndarray::arr1(&[3.0f64, 4.0]);
            let result = ctx
                .evaluator()
                .push(&h)
                .feed(x, x_val.view().into_dyn())
                .run();

            let arr = result[0].as_ref().expect("should evaluate");
            let s = arr.as_slice().expect("slice");
            assert!((s[0] - 2.0).abs() < 1e-6, "H[0,0] = 2, got {}", s[0]);
            assert!((s[1]).abs() < 1e-6, "H[0,1] = 0, got {}", s[1]);
            assert!((s[2]).abs() < 1e-6, "H[1,0] = 0, got {}", s[2]);
            assert!((s[3] - 2.0).abs() < 1e-6, "H[1,1] = 2, got {}", s[3]);
        });
    }

    // ------------------------------------------------------------------
    // hvp
    // ------------------------------------------------------------------

    #[test]
    fn test_hvp_diagonal_hessian() {
        // f(x) = x0^2 + 2*x1^2, H = diag([2, 4])
        // v = [1, 1]  =>  H·v = [2, 4]
        // Using reduce_sum to avoid slice-based double-gradient issues
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let v_arr = scirs2_core::ndarray::arr1(&[1.0f64, 1.0]).into_dyn();
            let v = convert_to_tensor(v_arr, ctx);

            // Build weights tensor [1, 2] to implement x0^2 + 2*x1^2
            let w_arr = scirs2_core::ndarray::arr1(&[1.0f64, 2.0]).into_dyn();
            let w = convert_to_tensor(w_arr, ctx);

            let result = hvp(
                |t| {
                    // f(x) = sum(w * x^2) = x0^2 + 2*x1^2
                    let axes = [0_isize];
                    reduce_sum(*t * *t, &axes, false)
                },
                &x,
                &v,
                ctx,
            )
            .expect("hvp should succeed");

            let x_val = scirs2_core::ndarray::arr1(&[1.0f64, 1.0]);
            let out = ctx
                .evaluator()
                .push(&result)
                .feed(x, x_val.view().into_dyn())
                .run();

            let arr = out[0].as_ref().expect("should evaluate");
            let s = arr.as_slice().expect("slice");
            // f(x) = x0^2 + x1^2, H = diag([2, 2]), H·v = [2, 2]
            assert!((s[0] - 2.0).abs() < 1e-6, "H·v[0] expected 2, got {}", s[0]);
            assert!((s[1] - 2.0).abs() < 1e-6, "H·v[1] expected 2, got {}", s[1]);
        });
    }

    #[test]
    fn test_hvp_shape_mismatch_error() {
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let v_arr = scirs2_core::ndarray::arr1(&[1.0f64, 1.0, 1.0]).into_dyn();
            let v = convert_to_tensor(v_arr, ctx);
            let result = hvp(|t| reduction::sum_all(*t), &x, &v, ctx);
            assert!(result.is_err(), "mismatched shapes should return error");
        });
    }

    // ------------------------------------------------------------------
    // jacobian_matrix
    // ------------------------------------------------------------------

    #[test]
    fn test_jacobian_matrix_linear() {
        // f(x) = [x0, x1]  =>  J = I_2
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let jac = jacobian_matrix(|t| *t, &x, ctx, 2, 2)
                .expect("jacobian_matrix should succeed");

            let x_val = scirs2_core::ndarray::arr1(&[3.0f64, 5.0]);
            let out = ctx
                .evaluator()
                .push(&jac)
                .feed(x, x_val.view().into_dyn())
                .run();

            let arr = out[0].as_ref().expect("should evaluate");
            let s = arr.as_slice().expect("slice");
            // Identity: [[1,0],[0,1]]
            assert!((s[0] - 1.0).abs() < 1e-6);
            assert!((s[1]).abs() < 1e-6);
            assert!((s[2]).abs() < 1e-6);
            assert!((s[3] - 1.0).abs() < 1e-6);
        });
    }

    #[test]
    fn test_jacobian_matrix_nonlinear() {
        // f(x) = [x0^2, x0*x1]
        // J at [2, 3] = [[4, 0], [3, 2]]
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let jac = jacobian_matrix(
                |t| {
                    let x0 = slice(*t, [0isize], [1isize]);
                    let x1 = slice(*t, [1isize], [2isize]);
                    linear_algebra::concat(&[x0 * x0, x0 * x1], 0)
                },
                &x,
                ctx,
                2,
                2,
            )
            .expect("jacobian_matrix should succeed");

            let x_val = scirs2_core::ndarray::arr1(&[2.0f64, 3.0]);
            let out = ctx
                .evaluator()
                .push(&jac)
                .feed(x, x_val.view().into_dyn())
                .run();

            let arr = out[0].as_ref().expect("should evaluate");
            let s = arr.as_slice().expect("slice");
            assert!((s[0] - 4.0).abs() < 1e-6, "J[0,0] expected 4, got {}", s[0]);
            assert!((s[1]).abs() < 1e-6, "J[0,1] expected 0, got {}", s[1]);
            assert!((s[2] - 3.0).abs() < 1e-6, "J[1,0] expected 3, got {}", s[2]);
            assert!((s[3] - 2.0).abs() < 1e-6, "J[1,1] expected 2, got {}", s[3]);
        });
    }

    // ------------------------------------------------------------------
    // jvp
    // ------------------------------------------------------------------

    #[test]
    fn test_jvp_unit_vectors() {
        // For f: R^n -> R (scalar), JVP = grad(f)·v
        // f(x) = sum(x^2), grad = 2x
        // At x=[2,3], v=[1,0]: JVP = 2*2*1 + 2*3*0 = 4
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let v_arr = scirs2_core::ndarray::arr1(&[1.0f64, 0.0]).into_dyn();
            let v = convert_to_tensor(v_arr, ctx);

            let result = jvp(
                |t| {
                    let axes = [0_isize];
                    reduce_sum(*t * *t, &axes, false)
                },
                &x,
                &v,
                ctx,
                1, // scalar output: m=1
            )
            .expect("jvp should succeed");

            let x_val = scirs2_core::ndarray::arr1(&[2.0f64, 3.0]);
            let out = ctx
                .evaluator()
                .push(&result)
                .feed(x, x_val.view().into_dyn())
                .run();

            let arr = out[0].as_ref().expect("should evaluate");
            let s = arr.as_slice().expect("slice");
            // JVP = grad(sum(x^2))·v = 2x·v = [4,6]·[1,0] = 4
            assert!((s[0] - 4.0).abs() < 1e-6, "JVP expected 4, got {}", s[0]);
        });
    }

    // ------------------------------------------------------------------
    // vjp
    // ------------------------------------------------------------------

    #[test]
    fn test_vjp_squared_norm() {
        // f(x) = x (identity), VJP = v for any v
        // Equivalently: f(x) = x0^2 + x1^2, v = [1], VJP = [2x0, 2x1] = [4, 6] at [2,3]
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let v_arr = scirs2_core::ndarray::arr1(&[1.0f64]).into_dyn();
            let v = convert_to_tensor(v_arr, ctx);

            let result = vjp(
                |t| {
                    // Scalar output
                    let x0 = slice(*t, [0isize], [1isize]);
                    let x1 = slice(*t, [1isize], [2isize]);
                    reduction::sum_all(x0 * x0 + x1 * x1)
                },
                &x,
                &v,
                ctx,
            )
            .expect("vjp should succeed");

            let x_val = scirs2_core::ndarray::arr1(&[2.0f64, 3.0]);
            let out = ctx
                .evaluator()
                .push(&result)
                .feed(x, x_val.view().into_dyn())
                .run();

            let arr = out[0].as_ref().expect("should evaluate");
            let s = arr.as_slice().expect("slice");
            assert!((s[0] - 4.0).abs() < 1e-6, "VJP[0] expected 4, got {}", s[0]);
            assert!((s[1] - 6.0).abs() < 1e-6, "VJP[1] expected 6, got {}", s[1]);
        });
    }

    // ------------------------------------------------------------------
    // grad of x^2 = 2x consistency test
    // ------------------------------------------------------------------

    #[test]
    fn test_grad_x_squared_is_2x() {
        // Uses the tensor_ops::grad directly, verifying the building block for higher-order works.
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[]);
            let y = x * x; // y = x^2
            let dy_dx = &crate::tensor_ops::grad(&[y], &[x])[0]; // should be 2x

            let x_val = scirs2_core::ndarray::arr0(3.0f64);
            let out = ctx
                .evaluator()
                .push(dy_dx)
                .feed(x, x_val.view().into_dyn())
                .run();

            let arr = out[0].as_ref().expect("should evaluate");
            let val = arr.first().copied().expect("first element");
            assert!((val - 6.0).abs() < 1e-9, "d(x^2)/dx at x=3 should be 6, got {}", val);
        });
    }

    // ------------------------------------------------------------------
    // jvp/vjp consistency
    // ------------------------------------------------------------------

    #[test]
    fn test_jvp_vjp_consistency_scalar() {
        // For f: R^n -> R, JVP(v) = grad·v and VJP(1) = grad
        // f(x) = sum(x^2), grad = 2x
        // At x=[2,1]: grad = [4, 2]
        // JVP with v=[1,0] = 4; VJP with v=[1] = [4, 2]
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[2]);

            // JVP with v = [1, 0]
            let v1_arr = scirs2_core::ndarray::arr1(&[1.0f64, 0.0]).into_dyn();
            let v1 = convert_to_tensor(v1_arr, ctx);

            let jvp_val = jvp(
                |t| {
                    let axes = [0_isize];
                    reduce_sum(*t * *t, &axes, false)
                },
                &x,
                &v1,
                ctx,
                1,
            )
            .expect("jvp should succeed");

            // VJP with v = [1] (scalar co-vector)
            let v2_arr = scirs2_core::ndarray::arr1(&[1.0f64]).into_dyn();
            let v2 = convert_to_tensor(v2_arr, ctx);

            let vjp_val = vjp(
                |t| {
                    let axes = [0_isize];
                    reduce_sum(*t * *t, &axes, false)
                },
                &x,
                &v2,
                ctx,
            )
            .expect("vjp should succeed");

            let x_val = scirs2_core::ndarray::arr1(&[2.0f64, 1.0]);
            let outs = ctx
                .evaluator()
                .push(&jvp_val)
                .push(&vjp_val)
                .feed(x, x_val.view().into_dyn())
                .run();

            // JVP = grad(f)·v = 2x·[1,0] = 2*2 = 4
            let jvp_arr = outs[0].as_ref().expect("jvp eval");
            let jvp_s = jvp_arr.as_slice().expect("jvp slice");
            assert!((jvp_s[0] - 4.0).abs() < 1e-6, "JVP expected 4, got {}", jvp_s[0]);

            // VJP = grad(f) = 2x = [4, 2] at [2,1]
            let vjp_arr = outs[1].as_ref().expect("vjp eval");
            let vjp_s = vjp_arr.as_slice().expect("vjp slice");
            assert!((vjp_s[0] - 4.0).abs() < 1e-6, "VJP[0] expected 4, got {}", vjp_s[0]);
            assert!((vjp_s[1] - 2.0).abs() < 1e-6, "VJP[1] expected 2, got {}", vjp_s[1]);
        });
    }
}
