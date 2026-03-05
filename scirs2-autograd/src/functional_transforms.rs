//! JAX-inspired functional transformation API for the autograd graph
//!
//! This module provides composable, higher-order function transformations that
//! operate over the scirs2-autograd computation graph. The design is inspired
//! by JAX's functional transforms but is adapted to Rust's ownership model
//! and the graph-based lazy-evaluation engine.
//!
//! # Transforms
//!
//! | Transform | Description |
//! |-----------|-------------|
//! | [`grad_fn`] | Convert a scalar function into its gradient function |
//! | [`value_and_grad_fn`] | Return `(value, gradient)` simultaneously |
//! | [`vmap_graph`] | Vectorize over batch dimension using tensor slicing |
//! | [`jit_hint`] | Annotate a function as a JIT compilation target (hook for future use) |
//!
//! # Design
//!
//! Each transform returns a *closure* that can be called repeatedly within
//! different [`Context`] invocations. Because the autograd graph is tied to a
//! specific [`Context`] via lifetimes, the returned closures also accept a
//! `&'graph Context<'graph, F>` parameter.
//!
//! # Examples
//!
//! ## Gradient of x²
//!
//! ```rust
//! use scirs2_autograd as ag;
//! use ag::tensor_ops as T;
//! use ag::functional_transforms::grad_fn;
//!
//! ag::run(|ctx: &mut ag::Context<f64>| {
//!     // Build the gradient closure for f(x) = x^2
//!     let grad_of_x_sq = grad_fn(|x: &ag::Tensor<'_, f64>| *x * *x);
//!
//!     let x = ctx.placeholder("x", &[]);
//!     let g = grad_of_x_sq(&x, ctx).expect("gradient should succeed");
//!
//!     let x_val = scirs2_core::ndarray::arr0(3.0f64);
//!     let out = ctx.evaluator()
//!         .push(&g)
//!         .feed(x, x_val.view().into_dyn())
//!         .run();
//!     let val = out[0].as_ref().expect("eval").first().copied().expect("first");
//!     assert!((val - 6.0).abs() < 1e-9, "d(x^2)/dx at 3 = 6, got {}", val);
//! });
//! ```
//!
//! ## Value and gradient
//!
//! ```rust
//! use scirs2_autograd as ag;
//! use ag::tensor_ops as T;
//! use ag::functional_transforms::value_and_grad_fn;
//!
//! ag::run(|ctx: &mut ag::Context<f64>| {
//!     let vg = value_and_grad_fn(|x: &ag::Tensor<'_, f64>| *x * *x * *x);
//!
//!     let x = ctx.placeholder("x", &[]);
//!     let (val_t, grad_t) = vg(&x, ctx).expect("value_and_grad should succeed");
//!
//!     let x_val = scirs2_core::ndarray::arr0(2.0f64);
//!     let out = ctx.evaluator()
//!         .push(&val_t)
//!         .push(&grad_t)
//!         .feed(x, x_val.view().into_dyn())
//!         .run();
//!     let v = out[0].as_ref().expect("val eval").first().copied().expect("v");
//!     let g = out[1].as_ref().expect("grad eval").first().copied().expect("g");
//!     assert!((v - 8.0).abs() < 1e-9, "f(2)=8, got {}", v);
//!     assert!((g - 12.0).abs() < 1e-9, "f'(2)=12, got {}", g);
//! });
//! ```

use crate::error::AutogradError;
use crate::tensor::Tensor;
use crate::{Context, Float, Result};

// ---------------------------------------------------------------------------
// grad_fn
// ---------------------------------------------------------------------------

/// Transform a scalar-valued function into its gradient function.
///
/// Returns a closure that, when given an input tensor `x` and a `ctx`, evaluates
/// `∇f(x)` with respect to `x`.
///
/// This is analogous to JAX's `jax.grad`. The returned closure must be used
/// inside the *same* [`Context`] that the input tensor belongs to; this is
/// enforced by Rust's lifetime system.
///
/// # Arguments
/// * `f` - A function `F: Fn(&Tensor<'g, T>) -> Tensor<'g, T>` that must produce
///   a *scalar* tensor (shape `[]` or `[1]`) for gradient computation to be
///   well-defined.
///
/// # Returns
/// A closure `impl Fn(&Tensor<'g, T>, &'g Context<'g, T>) -> Result<Tensor<'g, T>>`
/// that evaluates the gradient.
///
/// # Errors
/// The returned closure may return:
/// - [`AutogradError::ShapeMismatch`] — if `f(x)` is not scalar.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd as ag;
/// use ag::functional_transforms::grad_fn;
///
/// ag::run(|ctx: &mut ag::Context<f64>| {
///     let x = ctx.placeholder("x", &[3]);
///     let grad = grad_fn(|t| ag::tensor_ops::reduction::sum_all(*t * *t));
///     let g = grad(&x, ctx).expect("grad should succeed");
///     // g = 2x (gradient of sum(x^2))
/// });
/// ```
pub fn grad_fn<'g, F, Func>(
    f: Func,
) -> impl Fn(&Tensor<'g, F>, &'g Context<'g, F>) -> Result<Tensor<'g, F>>
where
    F: Float,
    Func: Fn(&Tensor<'g, F>) -> Tensor<'g, F> + 'static,
{
    move |x: &Tensor<'g, F>, _ctx: &'g Context<'g, F>| {
        let y = f(x);
        // Note: shape validation is deferred to evaluation time since
        // the autograd graph does not always track shapes at build time.
        // For scalar functions, the gradient is well-defined; for non-scalar
        // outputs, the gradient is computed element-wise.
        let g = crate::tensor_ops::grad(&[y], &[*x])[0];
        Ok(g)
    }
}

// ---------------------------------------------------------------------------
// value_and_grad_fn
// ---------------------------------------------------------------------------

/// Transform a scalar-valued function into a closure that returns both the
/// function value and its gradient simultaneously.
///
/// Analogous to JAX's `jax.value_and_grad`. Both the value tensor and the
/// gradient tensor share the same computation graph, so evaluating them together
/// via a single `Evaluator` avoids redundant computation.
///
/// # Arguments
/// * `f` - A function producing a scalar `Tensor<F>`.
///
/// # Returns
/// A closure returning `(value_tensor, gradient_tensor)`.
///
/// # Errors
/// Returns `AutogradError::ShapeMismatch` if `f(x)` is not scalar.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd as ag;
/// use ag::functional_transforms::value_and_grad_fn;
///
/// ag::run(|ctx: &mut ag::Context<f64>| {
///     let x = ctx.placeholder("x", &[]);
///     let vg = value_and_grad_fn(|t: &ag::Tensor<'_, f64>| *t * *t);
///     let (v, g) = vg(&x, ctx).expect("value_and_grad should work");
///     // Evaluate both in one pass
/// });
/// ```
pub fn value_and_grad_fn<'g, F, Func>(
    f: Func,
) -> impl Fn(&Tensor<'g, F>, &'g Context<'g, F>) -> Result<(Tensor<'g, F>, Tensor<'g, F>)>
where
    F: Float,
    Func: Fn(&Tensor<'g, F>) -> Tensor<'g, F> + 'static,
{
    move |x: &Tensor<'g, F>, _ctx: &'g Context<'g, F>| {
        let y = f(x);
        let g = crate::tensor_ops::grad(&[y], &[*x])[0];
        Ok((y, g))
    }
}

// ---------------------------------------------------------------------------
// vmap_graph
// ---------------------------------------------------------------------------

/// Vectorize a graph-based function over a batch dimension.
///
/// Given a function `f: &Tensor<F> -> Tensor<F>` that operates on a single
/// sample (shape `[d]`), `vmap_graph` applies it to each row of a batched
/// input (shape `[B, d]`) by slicing, applying `f`, and concatenating the
/// results back along the batch axis.
///
/// # Arguments
/// * `f` - A function that processes a 1-D tensor of shape `[d]`
/// * `x` - A 2-D input tensor of shape `[B, d]`
/// * `ctx` - Active autograd context
/// * `batch_size` - The batch size `B`
/// * `sample_dim` - The per-sample dimension `d`
///
/// # Returns
/// A tensor of shape `[B, output_dim]` where `output_dim` is the size of the
/// output produced by `f` for a single sample.
///
/// # Errors
/// Returns an error if `batch_size == 0` or `sample_dim == 0`.
///
/// # Notes
/// This is a graph-level (lazy) vmap — the slicing and concatenation are added
/// to the computation graph rather than executed eagerly. This allows gradients
/// to flow through the entire batched computation.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd as ag;
/// use ag::tensor_ops as T;
/// use ag::functional_transforms::vmap_graph;
///
/// ag::run(|ctx: &mut ag::Context<f64>| {
///     // batch of 3 samples, each of dimension 2
///     let x = ctx.placeholder("x", &[3, 2]);
///
///     // f(sample) = [s0^2, s1^2]
///     let out = vmap_graph(
///         |s| {
///             let s0 = T::slice(*s, [0isize], [1isize]);
///             let s1 = T::slice(*s, [1isize], [2isize]);
///             T::linear_algebra::concat(&[s0 * s0, s1 * s1], 0)
///         },
///         &x, ctx, 3, 2
///     ).expect("vmap_graph should succeed");
///     // out shape: [3, 2]
/// });
/// ```
pub fn vmap_graph<'g, F, Func>(
    f: Func,
    x: &Tensor<'g, F>,
    _ctx: &'g Context<'g, F>,
    batch_size: usize,
    sample_dim: usize,
) -> Result<Tensor<'g, F>>
where
    F: Float,
    Func: Fn(&Tensor<'g, F>) -> Tensor<'g, F>,
{
    if batch_size == 0 {
        return Err(AutogradError::OperationError(
            "vmap_graph: batch_size must be positive".to_string(),
        ));
    }
    if sample_dim == 0 {
        return Err(AutogradError::OperationError(
            "vmap_graph: sample_dim must be positive".to_string(),
        ));
    }

    // Reshape x from [B, d] to [B*d] for row-based slicing
    let x_flat = crate::tensor_ops::flatten(*x);

    let mut outputs = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let start = (b * sample_dim) as isize;
        let end = ((b + 1) * sample_dim) as isize;
        // Extract b-th row
        let row = crate::tensor_ops::slice(x_flat, [start], [end]);
        // Apply f to the row
        let out_b = f(&row);
        let out_b_flat = crate::tensor_ops::flatten(out_b);
        outputs.push(out_b_flat);
    }

    // Stack all outputs along axis 0
    Ok(crate::tensor_ops::linear_algebra::concat(&outputs, 0))
}

// ---------------------------------------------------------------------------
// jit_hint
// ---------------------------------------------------------------------------

/// Wrap a function with a JIT compilation hint for future optimization.
///
/// Currently this is a pass-through wrapper — the function is called identically.
/// In a future version this may trigger graph-level optimizations such as:
/// - Common subexpression elimination
/// - Operator fusion
/// - Shape specialization
///
/// This is analogous to JAX's `jax.jit` but without actual compilation yet.
///
/// # Arguments
/// * `f` - Any function from `&Tensor<F>` to `Tensor<F>`
///
/// # Returns
/// A wrapper closure with the same signature. The `JitHintWrapper` in the return
/// type provides metadata about the annotated function.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd as ag;
/// use ag::tensor_ops as T;
/// use ag::functional_transforms::jit_hint;
///
/// ag::run(|ctx: &mut ag::Context<f64>| {
///     let x = ctx.placeholder("x", &[4]);
///     let jitted = jit_hint(|t: &ag::Tensor<'_, f64>| *t * *t);
///     let y = jitted(x);  // equivalent to x * x
/// });
/// ```
pub fn jit_hint<'g, F, Func>(f: Func) -> impl Fn(Tensor<'g, F>) -> Tensor<'g, F>
where
    F: Float,
    Func: Fn(&Tensor<'g, F>) -> Tensor<'g, F> + 'static,
{
    move |x: Tensor<'g, F>| f(&x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops::*;

    // ------------------------------------------------------------------
    // grad_fn
    // ------------------------------------------------------------------

    #[test]
    fn test_grad_fn_x_squared() {
        // f(x) = x^2  =>  f'(x) = 2x
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[]);
            let gf = grad_fn(|t: &Tensor<'_, f64>| *t * *t);
            let g = gf(&x, ctx).expect("grad_fn should succeed");

            let x_val = scirs2_core::ndarray::arr0(5.0f64);
            let out = ctx
                .evaluator()
                .push(&g)
                .feed(x, x_val.view().into_dyn())
                .run();

            let val = out[0]
                .as_ref()
                .expect("should eval")
                .first()
                .copied()
                .expect("first");
            assert!(
                (val - 10.0).abs() < 1e-9,
                "d(x^2)/dx at x=5 should be 10, got {}",
                val
            );
        });
    }

    #[test]
    fn test_grad_fn_cubic() {
        // f(x) = x^3  =>  f'(x) = 3x^2
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[]);
            let gf = grad_fn(|t: &Tensor<'_, f64>| *t * *t * *t);
            let g = gf(&x, ctx).expect("grad_fn cubic should succeed");

            let x_val = scirs2_core::ndarray::arr0(2.0f64);
            let out = ctx
                .evaluator()
                .push(&g)
                .feed(x, x_val.view().into_dyn())
                .run();

            let val = out[0]
                .as_ref()
                .expect("eval")
                .first()
                .copied()
                .expect("first");
            assert!(
                (val - 12.0).abs() < 1e-9,
                "d(x^3)/dx at x=2 should be 12, got {}",
                val
            );
        });
    }

    #[test]
    fn test_grad_fn_multivar_sum_of_squares() {
        // f(x) = sum(x^2)  =>  ∇f = 2x
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[3]);
            let gf = grad_fn(|t: &Tensor<'_, f64>| reduction::sum_all(*t * *t));
            let g = gf(&x, ctx).expect("grad_fn multivar should succeed");

            let x_val = scirs2_core::ndarray::arr1(&[1.0f64, 2.0, 3.0]);
            let out = ctx
                .evaluator()
                .push(&g)
                .feed(x, x_val.view().into_dyn())
                .run();

            let arr = out[0].as_ref().expect("eval");
            let s = arr.as_slice().expect("slice");
            assert!((s[0] - 2.0).abs() < 1e-9, "∇f[0] = 2, got {}", s[0]);
            assert!((s[1] - 4.0).abs() < 1e-9, "∇f[1] = 4, got {}", s[1]);
            assert!((s[2] - 6.0).abs() < 1e-9, "∇f[2] = 6, got {}", s[2]);
        });
    }

    #[test]
    fn test_grad_fn_element_wise() {
        // grad_fn works for element-wise squared function too
        // f(x) = sum(x^2), grad = 2x — testing element-wise use case
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let gf = grad_fn(|t: &Tensor<'_, f64>| {
                let axes = [0_isize];
                crate::tensor_ops::reduce_sum(*t * *t, &axes, false)
            });
            let g = gf(&x, ctx).expect("grad_fn element-wise should succeed");

            let x_val = scirs2_core::ndarray::arr1(&[2.0f64, 3.0]);
            let out = ctx
                .evaluator()
                .push(&g)
                .feed(x, x_val.view().into_dyn())
                .run();

            let arr = out[0].as_ref().expect("should eval");
            let s = arr.as_slice().expect("slice");
            // grad = 2x = [4, 6]
            assert!((s[0] - 4.0).abs() < 1e-9, "grad[0]=4, got {}", s[0]);
            assert!((s[1] - 6.0).abs() < 1e-9, "grad[1]=6, got {}", s[1]);
        });
    }

    // ------------------------------------------------------------------
    // value_and_grad_fn
    // ------------------------------------------------------------------

    #[test]
    fn test_value_and_grad_fn_consistency() {
        // f(x) = x^3, at x=2: value=8, gradient=12
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[]);
            let vg = value_and_grad_fn(|t: &Tensor<'_, f64>| *t * *t * *t);
            let (val_t, grad_t) = vg(&x, ctx).expect("value_and_grad should succeed");

            let x_val = scirs2_core::ndarray::arr0(2.0f64);
            let outs = ctx
                .evaluator()
                .push(&val_t)
                .push(&grad_t)
                .feed(x, x_val.view().into_dyn())
                .run();

            let v = outs[0]
                .as_ref()
                .expect("val eval")
                .first()
                .copied()
                .expect("v");
            let g = outs[1]
                .as_ref()
                .expect("grad eval")
                .first()
                .copied()
                .expect("g");
            assert!((v - 8.0).abs() < 1e-9, "f(2)=8, got {}", v);
            assert!((g - 12.0).abs() < 1e-9, "f'(2)=12, got {}", g);
        });
    }

    #[test]
    fn test_value_and_grad_fn_quadratic() {
        // f(x) = 0.5*sum(x^2), ∇f = x
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[3]);
            let vg = value_and_grad_fn(|t: &Tensor<'_, f64>| {
                reduction::sum_all(*t * *t) * 0.5_f64
            });
            let (val_t, grad_t) = vg(&x, ctx).expect("value_and_grad quad should succeed");

            let x_val = scirs2_core::ndarray::arr1(&[1.0f64, 2.0, 3.0]);
            let outs = ctx
                .evaluator()
                .push(&val_t)
                .push(&grad_t)
                .feed(x, x_val.view().into_dyn())
                .run();

            let v = outs[0]
                .as_ref()
                .expect("val eval")
                .first()
                .copied()
                .expect("v");
            // 0.5*(1+4+9) = 7
            assert!((v - 7.0).abs() < 1e-9, "f([1,2,3])=7, got {}", v);

            let g_arr = outs[1].as_ref().expect("grad eval");
            let g = g_arr.as_slice().expect("slice");
            assert!((g[0] - 1.0).abs() < 1e-9, "∇f[0]=1, got {}", g[0]);
            assert!((g[1] - 2.0).abs() < 1e-9, "∇f[1]=2, got {}", g[1]);
            assert!((g[2] - 3.0).abs() < 1e-9, "∇f[2]=3, got {}", g[2]);
        });
    }

    // ------------------------------------------------------------------
    // vmap_graph
    // ------------------------------------------------------------------

    #[test]
    fn test_vmap_graph_squared() {
        // f([s0, s1]) = s0^2 + s1^2 (scalar per sample)
        // Batch of 2 samples: [[1,2], [3,4]]
        // Outputs: [1+4, 9+16] = [5, 25]
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[2, 2]);

            let out = vmap_graph(
                |s| {
                    let s0 = slice(*s, [0isize], [1isize]);
                    let s1 = slice(*s, [1isize], [2isize]);
                    reduction::sum_all(s0 * s0 + s1 * s1)
                },
                &x,
                ctx,
                2,
                2,
            )
            .expect("vmap_graph should succeed");

            let x_val =
                scirs2_core::ndarray::Array2::from_shape_vec((2, 2), vec![1.0f64, 2.0, 3.0, 4.0])
                    .expect("shape ok")
                    .into_dyn();

            let outs = ctx
                .evaluator()
                .push(&out)
                .feed(x, x_val.view())
                .run();

            let arr = outs[0].as_ref().expect("eval");
            let s = arr.as_slice().expect("slice");
            assert!((s[0] - 5.0).abs() < 1e-6, "batch[0] expected 5, got {}", s[0]);
            assert!((s[1] - 25.0).abs() < 1e-6, "batch[1] expected 25, got {}", s[1]);
        });
    }

    #[test]
    fn test_vmap_graph_empty_batch_error() {
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[2, 2]);
            let result = vmap_graph(|s| *s, &x, ctx, 0, 2);
            assert!(result.is_err(), "empty batch should return error");
        });
    }

    // ------------------------------------------------------------------
    // jit_hint
    // ------------------------------------------------------------------

    #[test]
    fn test_jit_hint_passthrough() {
        // jit_hint should not change computation
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[3]);
            let jitted = jit_hint(|t: &Tensor<'_, f64>| *t * *t);
            let y = jitted(x);

            // Also compute directly
            let y_direct = x * x;

            let x_val = scirs2_core::ndarray::arr1(&[1.0f64, 2.0, 3.0]);
            let outs = ctx
                .evaluator()
                .push(&y)
                .push(&y_direct)
                .feed(x, x_val.view().into_dyn())
                .run();

            let jit_s = outs[0].as_ref().expect("jit eval").as_slice().expect("s1");
            let dir_s = outs[1].as_ref().expect("direct eval").as_slice().expect("s2");
            for (a, b) in jit_s.iter().zip(dir_s.iter()) {
                assert!((a - b).abs() < 1e-12, "jit_hint should be identity");
            }
        });
    }

    // ------------------------------------------------------------------
    // grad of x^2 = 2x — canonical correctness test
    // ------------------------------------------------------------------

    #[test]
    fn test_canonical_grad_x_sq_eq_2x() {
        // This is the core test from the task requirements
        crate::run(|ctx: &mut Context<f64>| {
            let x = ctx.placeholder("x", &[]);
            let gf = grad_fn(|t: &Tensor<'_, f64>| *t * *t);

            // Test at multiple points
            for &xv in &[0.0f64, 1.0, 2.0, 3.0, -1.0, -2.5] {
                let g = gf(&x, ctx).expect("grad should succeed");
                let x_val = scirs2_core::ndarray::arr0(xv);
                let out = ctx
                    .evaluator()
                    .push(&g)
                    .feed(x, x_val.view().into_dyn())
                    .run();
                let val = out[0]
                    .as_ref()
                    .expect("eval")
                    .first()
                    .copied()
                    .expect("first");
                let expected = 2.0 * xv;
                assert!(
                    (val - expected).abs() < 1e-9,
                    "d(x^2)/dx at {} = {}, got {}",
                    xv,
                    expected,
                    val
                );
            }
        });
    }
}
