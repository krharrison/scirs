//! Comprehensive functional automatic differentiation API for scirs2-autograd
//!
//! This module provides a self-contained, graph-free automatic differentiation
//! API built on three complementary foundations:
//!
//! | Submodule | Mechanism | Best for |
//! |-----------|-----------|----------|
//! | [`dual`] | Dual-number forward-mode AD | Exact derivatives, small n |
//! | [`tape`] | Wengert-list reverse-mode AD | Gradient of scalar losses |
//! | [`transforms`] | JAX-style functional transforms (FD) | Composable closures |
//! | [`higher_order`] | Higher-order derivatives and specialised operators | Hessians, Laplacians, Taylor series |
//! | [`legacy`] | Original point-wise API returning `Result` | Backward compat |
//!
//! # Quick Start
//!
//! ## Gradient via transform closure
//!
//! ```rust
//! use scirs2_autograd::functional::transforms::grad;
//!
//! let grad_f = grad(|xs: &[f64]| xs[0].powi(2) + xs[1].powi(2));
//! let g = grad_f(&[3.0, 4.0]);
//! assert!((g[0] - 6.0).abs() < 1e-3);
//! assert!((g[1] - 8.0).abs() < 1e-3);
//! ```
//!
//! ## Gradient via dual numbers (exact)
//!
//! ```rust
//! use scirs2_autograd::functional::dual::{Dual, eval_gradient};
//!
//! let (val, grad) = eval_gradient(
//!     |xs: &[scirs2_autograd::functional::dual::Dual]| xs[0] * xs[0] + xs[1] * xs[1],
//!     &[3.0, 4.0],
//! );
//! assert!((val - 25.0).abs() < 1e-12);
//! assert!((grad[0] - 6.0).abs() < 1e-12);
//! ```
//!
//! ## Gradient via Wengert tape (reverse-mode)
//!
//! ```rust
//! use scirs2_autograd::functional::tape::tape_grad;
//!
//! let g = tape_grad(
//!     |tape, xs| {
//!         let x2 = tape.powi(xs[0], 2);
//!         let y2 = tape.powi(xs[1], 2);
//!         tape.add(x2, y2)
//!     },
//!     &[3.0, 4.0],
//! ).expect("tape gradient");
//! assert!((g[0] - 6.0).abs() < 1e-12);
//! assert!((g[1] - 8.0).abs() < 1e-12);
//! ```
//!
//! ## JVP and VJP
//!
//! ```rust
//! use scirs2_autograd::functional::transforms::{jvp, vjp};
//!
//! let f = |xs: &[f64]| vec![xs[0]*xs[0], xs[0]*xs[1]];
//!
//! // Forward-mode JVP
//! let (fx, jvp_val) = jvp(f, &[2.0, 3.0], &[1.0, 0.0]);
//! assert!((jvp_val[0] - 4.0).abs() < 1e-4); // d(x^2)/dx * 1 = 4
//!
//! // Reverse-mode VJP
//! let f2 = |xs: &[f64]| vec![xs[0]*xs[0], xs[0]*xs[1]];
//! let (_, g) = vjp(f2, &[2.0, 3.0], &[1.0, 0.0]);
//! assert!((g[0] - 4.0).abs() < 1e-4);
//! ```
//!
//! ## Hessian
//!
//! ```rust
//! use scirs2_autograd::functional::transforms::hessian;
//!
//! let hess = hessian(|xs: &[f64]| xs[0]*xs[0] + 3.0*xs[0]*xs[1] + 2.0*xs[1]*xs[1]);
//! let h = hess(&[1.0, 1.0]);
//! assert!((h[0][0] - 2.0).abs() < 1e-2);
//! assert!((h[0][1] - 3.0).abs() < 1e-2);
//! ```
//!
//! ## Laplacian
//!
//! ```rust
//! use scirs2_autograd::functional::higher_order::laplacian;
//!
//! // f = x^2 + y^2 + z^2; Δf = 6
//! let lap = laplacian(
//!     |xs: &[f64]| xs.iter().map(|v| v*v).sum::<f64>(),
//!     &[1.0, 2.0, 3.0],
//! ).expect("laplacian");
//! assert!((lap - 6.0).abs() < 0.5);
//! ```
//!
//! # Design Notes
//!
//! ## Three AD backends
//!
//! This module deliberately provides **three distinct backends** for gradient
//! computation, each with different trade-offs:
//!
//! 1. **Dual numbers** (`dual` module): Forward-mode AD with *exact* chain-rule
//!    propagation.  Cost: `n` forward passes for the full gradient, one pass per
//!    partial.  Best for small `n` where exactness matters (e.g. inside numerical
//!    solvers, verification tests).
//!
//! 2. **Wengert tape** (`tape` module): Reverse-mode AD that records the forward
//!    computation and replays it backwards.  Cost: one forward pass + one backward
//!    sweep (O(n) in graph size).  Best for large `n` (many inputs, one output
//!    loss), as in machine learning.
//!
//! 3. **Finite differences** (`transforms` module): Numerical approximation via
//!    central differences.  Cost: `2n` evaluations per gradient; `O(n²)` for
//!    Hessian.  Best for *any* function (no instrumentation required) when `n`
//!    is small and accuracy requirements are modest.
//!
//! ## JAX compatibility
//!
//! The function signatures in [`transforms`] deliberately match JAX conventions:
//!
//! | JAX | scirs2 |
//! |-----|--------|
//! | `jax.grad(f)` | `transforms::grad(f)` |
//! | `jax.jacobian(f)` | `transforms::jacobian(f, m)` |
//! | `jax.hessian(f)` | `transforms::hessian(f)` |
//! | `jax.vmap(f)` | `transforms::vmap(f)` |
//! | `jax.jvp(f, x, v)` | `transforms::jvp(f, x, v)` |
//! | `jax.vjp(f, x)` | `transforms::vjp(f, x, cotangent)` |

// New advanced submodules
pub mod dual;
pub mod higher_order;
pub mod tape;
pub mod transforms;

// Backward-compatible legacy API (preserved from functional.rs)
pub mod legacy;

// ============================================================================
// Convenience re-exports — new advanced API
// ============================================================================

// Dual number types and helpers
pub use dual::{eval_gradient, eval_hessian, Dual, HyperDual};

// Tape-based reverse-mode AD
pub use tape::{
    backward, backward_with_seed, tape_grad, tape_jacobian, GradientTape, Tape, TapeVar,
};

// Functional transforms (JAX-style)
pub use transforms::{
    check_grad, compose_vec, grad, grad_checked, grad_of_grad, hessian, iterate_scalar, jacobian,
    jvp, linearize, stop_gradient, value_and_grad, vjp, vmap, vmap_with_grad,
};

// Higher-order derivatives
pub use higher_order::{
    hessian_diagonal, hvp, iterated_gradient, jacobian_sequence, laplacian, laplacian_stochastic,
    mixed_partial, nth_derivative_scalar, taylor_coefficients,
};

// ============================================================================
// Module-level tests (integration tests spanning multiple submodules)
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Verify that all three AD backends agree on a simple quadratic gradient.
    #[test]
    fn test_three_backends_agree_gradient() {
        let x = &[3.0_f64, 4.0_f64];

        // 1. Dual numbers (exact)
        let (_, dual_g) = eval_gradient(|xs: &[Dual]| xs[0] * xs[0] + xs[1] * xs[1], x);

        // 2. Wengert tape (reverse-mode)
        let tape_g = tape_grad(
            |tape, xs| {
                let x2 = tape.powi(xs[0], 2);
                let y2 = tape.powi(xs[1], 2);
                tape.add(x2, y2)
            },
            x,
        )
        .expect("tape gradient");

        // 3. Finite differences
        let fd_g_fn = grad(|xs: &[f64]| xs[0] * xs[0] + xs[1] * xs[1]);
        let fd_g = fd_g_fn(x);

        // All should agree to at least 1e-3
        for i in 0..2 {
            assert!(
                (dual_g[i] - tape_g[i]).abs() < 1e-3,
                "dual vs tape mismatch at {i}: {} vs {}",
                dual_g[i],
                tape_g[i]
            );
            assert!(
                (dual_g[i] - fd_g[i]).abs() < 1e-3,
                "dual vs FD mismatch at {i}: {} vs {}",
                dual_g[i],
                fd_g[i]
            );
        }
    }

    /// Verify JVP and VJP are transposes of each other.
    #[test]
    fn test_jvp_vjp_duality() {
        // For any f, u, v: <u, J·v> = <Jᵀ·u, v>
        let x = &[2.0_f64, 3.0_f64];
        let v_vec = &[1.0_f64, 0.5_f64];
        let u = &[0.5_f64, 1.0_f64, 0.25_f64];

        let (_, jvp_val) = jvp(
            |xs: &[f64]| vec![xs[0] * xs[0], xs[0] * xs[1], xs[1] * xs[1]],
            x,
            v_vec,
        );
        let (_, vjp_val) = vjp(
            |xs: &[f64]| vec![xs[0] * xs[0], xs[0] * xs[1], xs[1] * xs[1]],
            x,
            u,
        );

        // <u, J·v>
        let u_jv: f64 = u.iter().zip(jvp_val.iter()).map(|(&a, &b)| a * b).sum();
        // <Jᵀ·u, v>
        let jtu_v: f64 = vjp_val.iter().zip(v_vec.iter()).map(|(&a, &b)| a * b).sum();

        assert!((u_jv - jtu_v).abs() < 1e-3, "<u,Jv>={u_jv}, <Jtu,v>={jtu_v}");
    }

    /// Verify that the Hessian from transforms matches the Hessian from dual numbers.
    #[test]
    fn test_hessian_dual_vs_fd() {
        let x = &[1.0_f64, 1.0_f64];

        // Dual-number Hessian (exact)
        let dual_h = eval_hessian(
            |xs: &[HyperDual]| {
                xs[0].powi(2)
                    + HyperDual::constant(3.0) * xs[0] * xs[1]
                    + HyperDual::constant(2.0) * xs[1].powi(2)
            },
            x,
        );

        // FD-based Hessian
        let fd_hess = hessian(
            |xs: &[f64]| xs[0] * xs[0] + 3.0 * xs[0] * xs[1] + 2.0 * xs[1] * xs[1],
        );
        let fd_h = fd_hess(x);

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (dual_h[i][j] - fd_h[i][j]).abs() < 1e-2,
                    "H[{i}][{j}]: dual={}, fd={}",
                    dual_h[i][j],
                    fd_h[i][j]
                );
            }
        }
    }

    /// Verify the Laplacian equals the trace of the Hessian.
    #[test]
    fn test_laplacian_equals_hessian_trace() {
        let x = &[2.0_f64, 3.0_f64];

        let h = hessian(
            |xs: &[f64]| xs[0].sin() * xs[1].cos() + xs[0] * xs[1] * xs[1],
        )(x);
        let hess_trace = h[0][0] + h[1][1];

        let lap = laplacian(
            |xs: &[f64]| xs[0].sin() * xs[1].cos() + xs[0] * xs[1] * xs[1],
            x,
        )
        .expect("laplacian");

        assert!(
            (lap - hess_trace).abs() < 0.1,
            "Laplacian={lap}, tr(H)={hess_trace}"
        );
    }

    /// Verify that grad_of_grad computes H·∇f correctly.
    #[test]
    fn test_grad_of_grad_is_hvp_at_grad() {
        // f(x,y) = x^2 + y^2; H = 2I; ∇f(3,4) = [6,8]; H·∇f = [12, 16]
        let gg = grad_of_grad(|xs: &[f64]| xs[0] * xs[0] + xs[1] * xs[1]);
        let result = gg(&[3.0, 4.0]);
        assert!((result[0] - 12.0).abs() < 1.0, "gg[0] = {}", result[0]);
        assert!((result[1] - 16.0).abs() < 1.0, "gg[1] = {}", result[1]);
    }
}
