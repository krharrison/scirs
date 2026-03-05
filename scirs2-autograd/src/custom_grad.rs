//! Custom gradient registration and decorator-style API
//!
//! This module provides a `HashMap`-backed registry for custom gradient
//! functions, a type-safe `CustomGradientFn<F, GF>` decorator struct, and a
//! `custom_gradient!` declarative macro that bundles a forward and backward
//! closure into a single named computation.
//!
//! It complements the [`custom_gradient`](crate::custom_gradient) module which
//! provides the lower-level `CustomGradientOp` trait and autograd-graph
//! integration. This module operates at the *plain-Rust* level using `Vec<f64>`
//! slices, making it easy to prototype and test gradient rules without depending
//! on the full computation-graph machinery.
//!
//! # Design
//!
//! | Component | Purpose |
//! |-----------|---------|
//! | [`GradFn`] | Type alias for a thread-safe custom gradient closure |
//! | [`CustomGradRegistry`] | Global + per-instance registry of `(name → GradFn)` |
//! | [`CustomGradientFn`] | Decorator pairing a forward `F` with a gradient `GF` |
//! | [`custom_gradient!`] | Macro for defining forward+backward together |
//!
//! # Examples
//!
//! ## Registry
//!
//! ```rust
//! use scirs2_autograd::custom_grad::{CustomGradRegistry, GradFn};
//! use std::sync::Arc;
//!
//! // Register a custom gradient for "my_square"
//! let mut reg = CustomGradRegistry::new();
//! reg.register("my_square", |inputs: &[Vec<f64>], _output_grads: &[Vec<f64>]| {
//!     // d(x²)/dx = 2x, but pass through output_grad via chain rule
//!     let x = &inputs[0];
//!     let og = &_output_grads[0];
//!     vec![x.iter().zip(og.iter()).map(|(&xi, &gi)| 2.0 * xi * gi).collect()]
//! });
//! assert!(reg.contains("my_square"));
//! ```
//!
//! ## Decorator
//!
//! ```rust
//! use scirs2_autograd::custom_grad::CustomGradientFn;
//!
//! let sq = CustomGradientFn::new(
//!     |inputs: &[Vec<f64>]| inputs[0].iter().map(|&v| v * v).collect::<Vec<f64>>(),
//!     |inputs: &[Vec<f64>], og: &[Vec<f64>]| {
//!         vec![inputs[0].iter().zip(og[0].iter()).map(|(&x, &g)| 2.0 * x * g).collect()]
//!     },
//! );
//!
//! let out = sq.call(&[vec![3.0_f64]]);
//! assert!((out[0] - 9.0).abs() < 1e-10);
//!
//! let grads = sq.grad(&[vec![3.0_f64]], &[vec![1.0_f64]]);
//! assert!((grads[0][0] - 6.0).abs() < 1e-10);
//! ```
//!
//! ## Macro
//!
//! ```rust
//! use scirs2_autograd::define_custom_gradient;
//!
//! // Define a "cube" op: z = x^3, dz/dx = 3x^2
//! let cube = define_custom_gradient!(
//!     fwd: |inputs: &[Vec<f64>]| -> Vec<f64> {
//!         inputs[0].iter().map(|&v| v * v * v).collect()
//!     },
//!     bwd: |inputs: &[Vec<f64>], og: &[Vec<f64>]| -> Vec<Vec<f64>> {
//!         vec![inputs[0].iter().zip(og[0].iter())
//!             .map(|(&x, &g)| 3.0 * x * x * g).collect()]
//!     }
//! );
//!
//! let z = cube.call(&[vec![2.0_f64]]);
//! assert!((z[0] - 8.0).abs() < 1e-10);
//! let g = cube.grad(&[vec![2.0_f64]], &[vec![1.0_f64]]);
//! assert!((g[0][0] - 12.0).abs() < 1e-10);
//! ```

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

// ─────────────────────────────────────────────────────────────────────────────
// GradFn type alias
// ─────────────────────────────────────────────────────────────────────────────

/// Type alias for a thread-safe custom gradient function.
///
/// The closure receives:
/// - `inputs: &[Vec<f64>]`  — the primal inputs to the op
/// - `output_grads: &[Vec<f64>]` — the upstream gradients (one per output)
///
/// It must return one `Vec<f64>` per input (the input gradients).
pub type GradFn = Arc<dyn Fn(&[Vec<f64>], &[Vec<f64>]) -> Vec<Vec<f64>> + Send + Sync>;

// ─────────────────────────────────────────────────────────────────────────────
// CustomGradRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// Registry mapping operation names to their custom gradient functions.
///
/// A global singleton is accessible via [`CustomGradRegistry::global`].
pub struct CustomGradRegistry {
    ops: HashMap<String, GradFn>,
}

impl Default for CustomGradRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomGradRegistry {
    /// Create a new, empty registry.
    pub fn new() -> Self {
        Self {
            ops: HashMap::new(),
        }
    }

    /// Register a custom gradient for the given operation name.
    ///
    /// If an entry with the same name already exists it is silently replaced.
    ///
    /// # Arguments
    /// * `op_name` - a unique, stable name for the operation
    /// * `grad_fn` - closure `(inputs, output_grads) -> input_grads`
    ///
    /// # Example
    /// ```rust
    /// use scirs2_autograd::custom_grad::CustomGradRegistry;
    ///
    /// let mut reg = CustomGradRegistry::new();
    /// reg.register("abs", |inputs: &[Vec<f64>], og: &[Vec<f64>]| {
    ///     let x = &inputs[0];
    ///     let g = &og[0];
    ///     vec![x.iter().zip(g.iter()).map(|(&xi, &gi)| gi * xi.signum()).collect()]
    /// });
    /// assert!(reg.contains("abs"));
    /// ```
    pub fn register(
        &mut self,
        op_name: impl Into<String>,
        grad_fn: impl Fn(&[Vec<f64>], &[Vec<f64>]) -> Vec<Vec<f64>> + Send + Sync + 'static,
    ) {
        self.ops
            .insert(op_name.into(), Arc::new(grad_fn));
    }

    /// Look up a registered gradient function by name.
    ///
    /// Returns `None` when no function with that name has been registered.
    pub fn get(&self, op_name: &str) -> Option<&GradFn> {
        self.ops.get(op_name)
    }

    /// Returns `true` if a gradient function for `op_name` is registered.
    pub fn contains(&self, op_name: &str) -> bool {
        self.ops.contains_key(op_name)
    }

    /// Remove the gradient function for `op_name` from the registry.
    ///
    /// Returns the function if it was present, or `None` otherwise.
    pub fn remove(&mut self, op_name: &str) -> Option<GradFn> {
        self.ops.remove(op_name)
    }

    /// Number of registered gradient functions.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Returns `true` if no gradient functions are registered.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Iterator over registered operation names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.ops.keys().map(|s| s.as_str())
    }

    /// Access the process-wide global registry.
    ///
    /// The registry is lazily initialised on first access.  Use the returned
    /// `RwLock` to obtain shared (`read()`) or exclusive (`write()`) access.
    ///
    /// # Example
    /// ```rust
    /// use scirs2_autograd::custom_grad::CustomGradRegistry;
    ///
    /// {
    ///     let mut reg = CustomGradRegistry::global().write().expect("lock poisoned");
    ///     reg.register("global_op", |inputs, og| {
    ///         vec![og[0].clone()] // pass-through gradient
    ///     });
    /// }
    ///
    /// {
    ///     let reg = CustomGradRegistry::global().read().expect("lock poisoned");
    ///     assert!(reg.contains("global_op"));
    /// }
    /// ```
    pub fn global() -> &'static RwLock<CustomGradRegistry> {
        static GLOBAL: OnceLock<RwLock<CustomGradRegistry>> = OnceLock::new();
        GLOBAL.get_or_init(|| RwLock::new(CustomGradRegistry::new()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CustomGradientFn decorator
// ─────────────────────────────────────────────────────────────────────────────

/// A decorator that pairs a forward function `F` with an analytic gradient `GF`.
///
/// This is the plain-Rust equivalent of Python's `@tf.custom_gradient` or
/// JAX's `jax.custom_vjp`: bundle the primal and backward computations in one
/// object so they can be passed around together.
///
/// # Type parameters
/// - `F`: `Fn(&[Vec<f64>]) -> Vec<f64>`
/// - `GF`: `Fn(&[Vec<f64>], &[Vec<f64>]) -> Vec<Vec<f64>>`
///
/// # Example
/// ```rust
/// use scirs2_autograd::custom_grad::CustomGradientFn;
///
/// let relu = CustomGradientFn::new(
///     |inputs: &[Vec<f64>]| inputs[0].iter().map(|&v| v.max(0.0)).collect::<Vec<f64>>(),
///     |inputs: &[Vec<f64>], og: &[Vec<f64>]| {
///         vec![inputs[0].iter().zip(og[0].iter())
///             .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
///             .collect()]
///     },
/// );
///
/// let z = relu.call(&[vec![-1.0_f64, 2.0]]);
/// assert_eq!(z, vec![0.0, 2.0]);
/// let g = relu.grad(&[vec![-1.0_f64, 2.0]], &[vec![1.0_f64, 1.0]]);
/// assert_eq!(g[0], vec![0.0, 1.0]);
/// ```
pub struct CustomGradientFn<F, GF> {
    forward: F,
    gradient: GF,
}

impl<F, GF> CustomGradientFn<F, GF>
where
    F: Fn(&[Vec<f64>]) -> Vec<f64>,
    GF: Fn(&[Vec<f64>], &[Vec<f64>]) -> Vec<Vec<f64>>,
{
    /// Construct a new `CustomGradientFn` from forward and gradient closures.
    pub fn new(forward: F, gradient: GF) -> Self {
        Self { forward, gradient }
    }

    /// Execute the forward computation on `inputs`.
    ///
    /// # Arguments
    /// * `inputs` - slice of input vectors; each inner `Vec<f64>` is one input tensor
    ///
    /// # Returns
    /// The output vector.
    pub fn call(&self, inputs: &[Vec<f64>]) -> Vec<f64> {
        (self.forward)(inputs)
    }

    /// Compute the input gradients given primal `inputs` and upstream `output_grad`.
    ///
    /// # Arguments
    /// * `inputs` - same inputs that were passed to `call`
    /// * `output_grad` - slice with one `Vec<f64>` per output of `call`
    ///
    /// # Returns
    /// One `Vec<f64>` per input, containing the gradient with respect to that input.
    pub fn grad(&self, inputs: &[Vec<f64>], output_grad: &[Vec<f64>]) -> Vec<Vec<f64>> {
        (self.gradient)(inputs, output_grad)
    }

    /// Convenience: run the forward pass and return `(output, backward_closure)`.
    ///
    /// The returned backward closure captures the inputs by cloning them so it
    /// can be called independently of the original `inputs` reference.
    ///
    /// # Example
    /// ```rust
    /// use scirs2_autograd::custom_grad::CustomGradientFn;
    ///
    /// let sq = CustomGradientFn::new(
    ///     |inputs: &[Vec<f64>]| inputs[0].iter().map(|&v| v * v).collect::<Vec<f64>>(),
    ///     |inputs: &[Vec<f64>], og: &[Vec<f64>]| {
    ///         vec![inputs[0].iter().zip(og[0].iter()).map(|(&x,&g)| 2.0*x*g).collect()]
    ///     },
    /// );
    ///
    /// let (out, bwd) = sq.forward_with_backward(&[vec![3.0_f64]]);
    /// assert!((out[0] - 9.0).abs() < 1e-10);
    /// let grads = bwd(&[vec![1.0_f64]]);
    /// assert!((grads[0][0] - 6.0).abs() < 1e-10);
    /// ```
    pub fn forward_with_backward(
        &self,
        inputs: &[Vec<f64>],
    ) -> (Vec<f64>, impl Fn(&[Vec<f64>]) -> Vec<Vec<f64>> + '_) {
        let output = (self.forward)(inputs);
        let inputs_clone = inputs.to_vec();
        let bwd = move |og: &[Vec<f64>]| (self.gradient)(&inputs_clone, og);
        (output, bwd)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Macro: define_custom_gradient
// ─────────────────────────────────────────────────────────────────────────────

/// Define a custom differentiable operation with co-located forward and backward.
///
/// Expands to a [`CustomGradientFn`] expression, so you can bind it to a
/// variable or pass it directly.
///
/// # Syntax
///
/// ```rust
/// use scirs2_autograd::define_custom_gradient;
///
/// let op = define_custom_gradient!(
///     fwd: |inputs: &[Vec<f64>]| -> Vec<f64> {
///         inputs[0].iter().map(|&v| v * v).collect()
///     },
///     bwd: |inputs: &[Vec<f64>], og: &[Vec<f64>]| -> Vec<Vec<f64>> {
///         vec![inputs[0].iter().zip(og[0].iter())
///             .map(|(&x, &g)| 2.0 * x * g)
///             .collect()]
///     }
/// );
///
/// let z = op.call(&[vec![4.0_f64]]);
/// assert!((z[0] - 16.0).abs() < 1e-10);
/// let g = op.grad(&[vec![4.0_f64]], &[vec![1.0_f64]]);
/// assert!((g[0][0] - 8.0).abs() < 1e-10);
/// ```
#[macro_export]
macro_rules! define_custom_gradient {
    (
        fwd: $fwd:expr,
        bwd: $bwd:expr
    ) => {
        $crate::custom_grad::CustomGradientFn::new($fwd, $bwd)
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: register_global_grad
// ─────────────────────────────────────────────────────────────────────────────

/// Register a gradient function in the global [`CustomGradRegistry`].
///
/// Returns `Ok(())` on success, or `Err` if the lock is poisoned.
///
/// # Example
/// ```rust
/// use scirs2_autograd::custom_grad::register_global_grad;
///
/// register_global_grad("clamp_grad", |_inputs, og| vec![og[0].clone()])
///     .expect("lock should not be poisoned");
/// ```
pub fn register_global_grad(
    op_name: impl Into<String>,
    grad_fn: impl Fn(&[Vec<f64>], &[Vec<f64>]) -> Vec<Vec<f64>> + Send + Sync + 'static,
) -> Result<(), String> {
    CustomGradRegistry::global()
        .write()
        .map(|mut reg| reg.register(op_name, grad_fn))
        .map_err(|e| format!("CustomGradRegistry lock poisoned: {e}"))
}

/// Look up a gradient function from the global [`CustomGradRegistry`] and clone it.
///
/// Returns `None` when no function with that name has been registered.
///
/// # Example
/// ```rust
/// use scirs2_autograd::custom_grad::{register_global_grad, lookup_global_grad};
///
/// register_global_grad("my_op_v2", |_inputs, og| vec![og[0].clone()])
///     .expect("registration succeeds");
/// let f = lookup_global_grad("my_op_v2");
/// assert!(f.is_some());
/// ```
pub fn lookup_global_grad(op_name: &str) -> Option<GradFn> {
    CustomGradRegistry::global()
        .read()
        .ok()
        .and_then(|reg| reg.get(op_name).cloned())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CustomGradRegistry ────────────────────────────────────────────────

    #[test]
    fn test_registry_register_and_lookup() {
        let mut reg = CustomGradRegistry::new();
        reg.register("sq_grad", |inputs: &[Vec<f64>], og: &[Vec<f64>]| {
            vec![inputs[0]
                .iter()
                .zip(og[0].iter())
                .map(|(&x, &g)| 2.0 * x * g)
                .collect()]
        });
        assert!(reg.contains("sq_grad"));
        assert!(!reg.contains("nonexistent"));
    }

    #[test]
    fn test_registry_get_and_call() {
        let mut reg = CustomGradRegistry::new();
        reg.register("double_grad", |_inputs: &[Vec<f64>], og: &[Vec<f64>]| {
            vec![og[0].iter().map(|&g| 2.0 * g).collect()]
        });

        let f = reg.get("double_grad").expect("should be registered");
        let inputs = vec![vec![1.0_f64, 2.0]];
        let og = vec![vec![1.0_f64, 1.0]];
        let grads = f(&inputs, &og);
        assert_eq!(grads[0], vec![2.0, 2.0]);
    }

    #[test]
    fn test_registry_replace_existing() {
        let mut reg = CustomGradRegistry::new();
        reg.register("op", |_i: &[Vec<f64>], og: &[Vec<f64>]| {
            vec![og[0].clone()]
        });
        reg.register("op", |_i: &[Vec<f64>], og: &[Vec<f64>]| {
            vec![og[0].iter().map(|&g| g * 3.0).collect()]
        });
        let f = reg.get("op").expect("should exist");
        let og = vec![vec![2.0_f64]];
        let grads = f(&[], &og);
        // new version multiplies by 3
        assert!((grads[0][0] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = CustomGradRegistry::new();
        reg.register("removable", |_i: &[Vec<f64>], og: &[Vec<f64>]| {
            vec![og[0].clone()]
        });
        assert!(reg.contains("removable"));
        let removed = reg.remove("removable");
        assert!(removed.is_some());
        assert!(!reg.contains("removable"));
    }

    #[test]
    fn test_registry_len_and_is_empty() {
        let mut reg = CustomGradRegistry::new();
        assert!(reg.is_empty());
        reg.register("a", |_i, og| vec![og[0].clone()]);
        reg.register("b", |_i, og| vec![og[0].clone()]);
        assert_eq!(reg.len(), 2);
    }

    // ── CustomGradientFn ──────────────────────────────────────────────────

    #[test]
    fn test_decorator_forward_square() {
        let sq = CustomGradientFn::new(
            |inputs: &[Vec<f64>]| inputs[0].iter().map(|&v| v * v).collect::<Vec<f64>>(),
            |inputs: &[Vec<f64>], og: &[Vec<f64>]| {
                vec![inputs[0]
                    .iter()
                    .zip(og[0].iter())
                    .map(|(&x, &g)| 2.0 * x * g)
                    .collect()]
            },
        );

        let out = sq.call(&[vec![3.0_f64, 4.0]]);
        assert!((out[0] - 9.0).abs() < 1e-10, "out[0]={}", out[0]);
        assert!((out[1] - 16.0).abs() < 1e-10, "out[1]={}", out[1]);
    }

    #[test]
    fn test_decorator_gradient_square() {
        let sq = CustomGradientFn::new(
            |inputs: &[Vec<f64>]| inputs[0].iter().map(|&v| v * v).collect::<Vec<f64>>(),
            |inputs: &[Vec<f64>], og: &[Vec<f64>]| {
                vec![inputs[0]
                    .iter()
                    .zip(og[0].iter())
                    .map(|(&x, &g)| 2.0 * x * g)
                    .collect()]
            },
        );

        let grads = sq.grad(&[vec![3.0_f64]], &[vec![1.0_f64]]);
        assert!((grads[0][0] - 6.0).abs() < 1e-10, "grad={}", grads[0][0]);
    }

    #[test]
    fn test_decorator_forward_with_backward() {
        let sq = CustomGradientFn::new(
            |inputs: &[Vec<f64>]| inputs[0].iter().map(|&v| v * v).collect::<Vec<f64>>(),
            |inputs: &[Vec<f64>], og: &[Vec<f64>]| {
                vec![inputs[0]
                    .iter()
                    .zip(og[0].iter())
                    .map(|(&x, &g)| 2.0 * x * g)
                    .collect()]
            },
        );

        let (out, bwd) = sq.forward_with_backward(&[vec![5.0_f64]]);
        assert!((out[0] - 25.0).abs() < 1e-10);
        let grads = bwd(&[vec![1.0_f64]]);
        assert!((grads[0][0] - 10.0).abs() < 1e-10);
    }

    // ── define_custom_gradient! macro ─────────────────────────────────────

    #[test]
    fn test_macro_cube() {
        let cube = define_custom_gradient!(
            fwd: |inputs: &[Vec<f64>]| -> Vec<f64> {
                inputs[0].iter().map(|&v| v * v * v).collect()
            },
            bwd: |inputs: &[Vec<f64>], og: &[Vec<f64>]| -> Vec<Vec<f64>> {
                vec![inputs[0].iter().zip(og[0].iter())
                    .map(|(&x, &g)| 3.0 * x * x * g).collect()]
            }
        );

        let z = cube.call(&[vec![2.0_f64]]);
        assert!((z[0] - 8.0).abs() < 1e-10, "z[0]={}", z[0]);

        let g = cube.grad(&[vec![2.0_f64]], &[vec![1.0_f64]]);
        assert!((g[0][0] - 12.0).abs() < 1e-10, "g[0][0]={}", g[0][0]);
    }

    #[test]
    fn test_macro_log_grad() {
        // log(x): fwd = ln(x), bwd = 1/x * g
        let logop = define_custom_gradient!(
            fwd: |inputs: &[Vec<f64>]| -> Vec<f64> {
                inputs[0].iter().map(|&v| v.ln()).collect()
            },
            bwd: |inputs: &[Vec<f64>], og: &[Vec<f64>]| -> Vec<Vec<f64>> {
                vec![inputs[0].iter().zip(og[0].iter())
                    .map(|(&x, &g)| g / x).collect()]
            }
        );

        let z = logop.call(&[vec![std::f64::consts::E]]);
        assert!((z[0] - 1.0).abs() < 1e-10);

        let g = logop.grad(&[vec![2.0_f64]], &[vec![1.0_f64]]);
        assert!((g[0][0] - 0.5).abs() < 1e-10);
    }

    // ── Global registry ───────────────────────────────────────────────────

    #[test]
    fn test_global_registry_register_and_lookup() {
        // Use unique name to avoid cross-test pollution
        register_global_grad("test_global_abs_grad", |inputs: &[Vec<f64>], og: &[Vec<f64>]| {
            vec![inputs[0]
                .iter()
                .zip(og[0].iter())
                .map(|(&x, &g)| g * x.signum())
                .collect()]
        })
        .expect("lock should not be poisoned");

        let f = lookup_global_grad("test_global_abs_grad")
            .expect("should be registered");

        let inputs = vec![vec![-2.0_f64, 3.0]];
        let og = vec![vec![1.0_f64, 1.0]];
        let grads = f(&inputs, &og);
        assert!((grads[0][0] + 1.0).abs() < 1e-10); // signum(-2) = -1
        assert!((grads[0][1] - 1.0).abs() < 1e-10); // signum(3) = 1
    }

    #[test]
    fn test_global_registry_lookup_missing() {
        let f = lookup_global_grad("definitely_not_registered_xyz_abc");
        assert!(f.is_none());
    }
}
