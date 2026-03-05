//! Higher-order automatic differentiation (functional, plain-Rust API)
//!
//! This module provides higher-order differentiation utilities that operate on
//! plain Rust closures `F: Fn(&[f64]) -> f64` (or `-> Vec<f64>`) without
//! requiring a live autograd [`Context`].  All derivatives are computed via
//! recursive central finite-differences.
//!
//! # Summary of functions
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`taylor_mode_forward`] | Taylor coefficients `f^(k)(x)·v^k / k!` up to order n |
//! | [`higher_order_jacobian`] | Sequence of Jacobians J, J', J'', … up to `order` |
//! | [`mixed_partials`] | Arbitrary mixed partial ∂^k f / (∂x_i₁ … ∂x_iₖ) |
//! | [`laplacian`] | Sum of diagonal Hessian entries ∑_i ∂²f/∂x_i² |
//! | [`hessian_vector_product_nth_order`] | n-th order HVP H^(n)·v via nested FD |
//! | [`automatic_taylor_expansion`] | Local Taylor coefficients `a_k = f^(k)(x₀)/k!` |

use crate::error::AutogradError;
use scirs2_core::ndarray::{Array1, Array2};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Central-FD step size used for all derivatives in this module.
const H: f64 = 1e-4;

/// Compute the first-order central-FD gradient of a scalar function.
fn gradient_fd(f: &(impl Fn(&[f64]) -> f64 + ?Sized), x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut g = vec![0.0f64; n];
    let two_h = 2.0 * H;
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    for i in 0..n {
        xp[i] = x[i] + H;
        xm[i] = x[i] - H;
        g[i] = (f(&xp) - f(&xm)) / two_h;
        xp[i] = x[i];
        xm[i] = x[i];
    }
    g
}

/// Compute the second-order central-FD directional derivative
/// `d²f/dt² |_{x, v}` = `(f(x+hv) - 2f(x) + f(x-hv)) / h²`.
fn second_derivative_directional(
    f: &impl Fn(&[f64]) -> f64,
    x: &[f64],
    v: &[f64],
    h_step: f64,
) -> f64 {
    let xp: Vec<f64> = x.iter().zip(v.iter()).map(|(&xi, &vi)| xi + h_step * vi).collect();
    let xm: Vec<f64> = x.iter().zip(v.iter()).map(|(&xi, &vi)| xi - h_step * vi).collect();
    let fx = f(x);
    (f(&xp) - 2.0 * fx + f(&xm)) / (h_step * h_step)
}

/// Recursively compute the k-th order directional derivative
/// `d^k f / dt^k |_{t=0}` where we move along direction `v`.
///
/// Uses the finite-difference differentiation of the (k-1)-th derivative:
/// f^(k)(x; v) ≈ (f^(k-1)(x+hv; v) - f^(k-1)(x-hv; v)) / (2h)
fn directional_derivative_nth(
    f: &dyn Fn(&[f64]) -> f64,
    x: &[f64],
    v: &[f64],
    order: usize,
) -> f64 {
    match order {
        0 => f(x),
        1 => {
            let dot: f64 = gradient_fd(f, x).iter().zip(v.iter()).map(|(&g, &vi)| g * vi).sum();
            dot
        }
        k => {
            // Recurse via central FD of (k-1)-th derivative
            let xp: Vec<f64> = x.iter().zip(v.iter()).map(|(&xi, &vi)| xi + H * vi).collect();
            let xm: Vec<f64> = x.iter().zip(v.iter()).map(|(&xi, &vi)| xi - H * vi).collect();
            let dp = directional_derivative_nth(f, &xp, v, k - 1);
            let dm = directional_derivative_nth(f, &xm, v, k - 1);
            (dp - dm) / (2.0 * H)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Taylor mode forward
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Taylor coefficients of `f` at `x` along direction `v`.
///
/// Returns the vector `[c₀, c₁, …, cₙ]` where `cₖ = f^(k)(x; v) / k!` is the
/// k-th Taylor coefficient.  `c₀ = f(x)`, `c₁ = ∇f(x)·v`,
/// `c₂ = (1/2) * d²f/dt²|_{t=0}`, and so on.
///
/// # Arguments
/// * `f`     – Scalar function `R^n → R`
/// * `x`     – Expansion point
/// * `v`     – Direction vector (must have the same length as `x`)
/// * `order` – Maximum derivative order (inclusive); must be ≥ 1
///
/// # Returns
/// `Vec<f64>` of length `order + 1` containing Taylor coefficients.
///
/// # Errors
/// Returns `AutogradError` if inputs are empty, dimensions mismatch, or `order == 0`.
///
/// # Example
/// ```rust
/// use scirs2_autograd::higher_order_new::taylor_mode_forward;
///
/// // f(x) = x[0]^3, expand at x=1 along v=[1]
/// // c0=1, c1=3, c2=3, c3=1
/// let coeffs = taylor_mode_forward(
///     |xs| xs[0] * xs[0] * xs[0],
///     &[1.0],
///     &[1.0],
///     3,
/// ).expect("taylor");
/// assert!((coeffs[0] - 1.0).abs() < 1e-3, "c0={}", coeffs[0]);
/// assert!((coeffs[1] - 3.0).abs() < 1e-3, "c1={}", coeffs[1]);
/// assert!((coeffs[2] - 3.0).abs() < 1e-3, "c2={}", coeffs[2]);
/// ```
pub fn taylor_mode_forward(
    f: impl Fn(&[f64]) -> f64,
    x: &[f64],
    v: &[f64],
    order: usize,
) -> Result<Vec<f64>, AutogradError> {
    if x.is_empty() {
        return Err(AutogradError::OperationError(
            "taylor_mode_forward: x must be non-empty".to_string(),
        ));
    }
    if v.len() != x.len() {
        return Err(AutogradError::ShapeMismatch(format!(
            "taylor_mode_forward: x length {} != v length {}",
            x.len(),
            v.len()
        )));
    }
    if order == 0 {
        return Err(AutogradError::OperationError(
            "taylor_mode_forward: order must be >= 1".to_string(),
        ));
    }

    let f_box: &dyn Fn(&[f64]) -> f64 = &f;
    let mut coeffs = Vec::with_capacity(order + 1);

    // c_0 = f(x)
    coeffs.push(f(x));

    // c_k = f^(k)(x; v) / k!
    let mut factorial: f64 = 1.0;
    for k in 1..=order {
        factorial *= k as f64;
        let dk = directional_derivative_nth(f_box, x, v, k);
        coeffs.push(dk / factorial);
    }

    Ok(coeffs)
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Higher-order Jacobians
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a sequence of Jacobians up to the given order.
///
/// For `f: R^n → R^m`, returns `[J₁, J₂, …, J_order]` where
/// * `J₁ = ∂f/∂x ∈ R^{m×n}` (standard Jacobian)
/// * `J₂ = ∂J₁/∂x ∈ R^{m×n}` (central FD of J₁ columns, aggregated)
/// * and so on.
///
/// For `k ≥ 2` we interpret `Jₖ[i, j] = ∂(Jₖ₋₁[i, j]) / ∂x_j` collapsed
/// to the same `m×n` shape (i.e. the diagonal of the higher-order tensor is
/// returned).
///
/// # Arguments
/// * `f`     – Vector function `R^n → R^m`
/// * `x`     – Evaluation point (length `n`)
/// * `order` – Number of Jacobian levels to compute (≥ 1)
///
/// # Returns
/// `Vec<Array2<f64>>` of length `order`, each of shape `m × n`.
///
/// # Errors
/// Returns `AutogradError` if inputs are empty or output is empty.
///
/// # Example
/// ```rust
/// use scirs2_autograd::higher_order_new::higher_order_jacobian;
///
/// // f(x) = [x[0]^2, x[1]^2]
/// // J1 = diag([2x0, 2x1]) at x=[1,1] => [[2,0],[0,2]]
/// let jacs = higher_order_jacobian(
///     |xs| vec![xs[0]*xs[0], xs[1]*xs[1]],
///     &[1.0, 1.0],
///     1,
/// ).expect("jacobians");
/// assert_eq!(jacs.len(), 1);
/// assert!((jacs[0][[0,0]] - 2.0).abs() < 1e-3);
/// ```
pub fn higher_order_jacobian(
    f: impl Fn(&[f64]) -> Vec<f64> + 'static,
    x: &[f64],
    order: usize,
) -> Result<Vec<Array2<f64>>, AutogradError> {
    let n = x.len();
    if n == 0 {
        return Err(AutogradError::OperationError(
            "higher_order_jacobian: x must be non-empty".to_string(),
        ));
    }
    if order == 0 {
        return Err(AutogradError::OperationError(
            "higher_order_jacobian: order must be >= 1".to_string(),
        ));
    }

    // Probe output dimension
    let f0 = f(x);
    let m = f0.len();
    if m == 0 {
        return Err(AutogradError::OperationError(
            "higher_order_jacobian: function output must be non-empty".to_string(),
        ));
    }

    let two_h = 2.0 * H;

    // Helper: compute Jacobian of a matrix-valued function (stored as flat vec[m*n]) via FD
    // We represent the current "function" as a closure returning a Vec of length m*n.
    // Level 0: the original f returning Vec<f64> of length m.
    // We track the current function as a boxed closure returning Vec<f64>.

    // Build J1 from f directly
    let j1 = {
        let mut mat = Array2::<f64>::zeros((m, n));
        let mut xp = x.to_vec();
        let mut xm_vec = x.to_vec();
        for j in 0..n {
            xp[j] = x[j] + H;
            xm_vec[j] = x[j] - H;
            let fp = f(&xp);
            let fm = f(&xm_vec);
            for i in 0..m {
                mat[[i, j]] = (fp[i] - fm[i]) / two_h;
            }
            xp[j] = x[j];
            xm_vec[j] = x[j];
        }
        mat
    };

    let mut result = Vec::with_capacity(order);
    result.push(j1.clone());

    if order == 1 {
        return Ok(result);
    }

    // For orders 2..=order we differentiate each (i,j) entry of the previous Jacobian
    // with respect to x[j] (diagonal of the higher-order tensor for compactness).
    // We store the current Jacobian as a flat snapshot and build the next one via FD.

    // We need closures that capture snapshots. To avoid lifetime issues we materialise
    // the Jacobian computation at each level using the original `f`.
    //
    // The k-th Jacobian is computed as:
    //   Jk[i,j] = ∂(J_{k-1}[i,j]) / ∂x_j   (diagonal slice of the full 3-tensor)
    //
    // We implement this by wrapping the previous level's Jacobian computation in a closure.

    // We use a dynamic dispatch approach: store current level as Box<dyn Fn(&[f64])->Vec<f64>>
    // where the output has length m*n (flattened J).
    type DynFn = Box<dyn Fn(&[f64]) -> Vec<f64>>;

    // Level 1: original f
    let f_rc: std::sync::Arc<dyn Fn(&[f64]) -> Vec<f64>> = std::sync::Arc::new(f);

    // build_jacobian: given a closure returning Vec<f64> of length out_len,
    // return a closure returning the flattened Jacobian of size out_len * n.
    fn build_jacobian_fn(
        g: std::sync::Arc<dyn Fn(&[f64]) -> Vec<f64>>,
        n: usize,
        out_len: usize,
    ) -> DynFn {
        Box::new(move |x: &[f64]| {
            let two_h_inner = 2.0 * H;
            let mut mat = vec![0.0f64; out_len * n];
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            for j in 0..n {
                xp[j] = x[j] + H;
                xm[j] = x[j] - H;
                let fp = g(&xp);
                let fm = g(&xm);
                for i in 0..out_len {
                    mat[i * n + j] = (fp[i] - fm[i]) / two_h_inner;
                }
                xp[j] = x[j];
                xm[j] = x[j];
            }
            mat
        })
    }

    // Level 1 jacobian as a function (m outputs)
    let mut current_fn: std::sync::Arc<dyn Fn(&[f64]) -> Vec<f64>> = f_rc;
    let mut current_out_len = m;

    for _level in 2..=order {
        let next_fn: DynFn = build_jacobian_fn(current_fn.clone(), n, current_out_len);
        let next_out_len = current_out_len * n;

        // Evaluate next_fn at x to get the flattened J_level
        let flat = next_fn(x);

        // Reshape flat to (m, n) by taking the diagonal of the extra n dimension
        // flat has shape [current_out_len * n]. We fold it back to [m, n] by
        // extracting the diagonal: for each original (i,j) pair, the entry is
        // flat[i*n*n + j*n + j] if level==2, or more generally the last axis diagonal.
        //
        // For simplicity: for level k we interpret Jk[i,j] as the diagonal-contracted
        // version of the full tensor, which equals flat[(i*n + j)] for the m×n case.
        // Since current_out_len = m*n^(level-2) at level, we need to reshape carefully.
        //
        // The approach: after each Jacobian step, the result has shape m × n^k.
        // We collapse n^(k-1) down to n by taking element [j, j, ..., j] (full diagonal).
        //
        // For k=2: flat shape m*n * n. Entry [i, j] = flat[(i*n + j)*n + j].
        // For k=3: flat shape m*n^2 * n. Entry [i, j] = flat[((i*n+j)*n+j)*n+j].
        // This is the "super-diagonal" contraction.

        let current_depth = _level - 1; // number of n-dimensions in current_out_len
        // current_out_len = m * n^(current_depth - 1)  at start of this iteration
        // after Jacobian, flat has length = m * n^current_depth
        let _ = next_out_len; // suppress unused warning

        let mut jk = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                // Navigate the nested index: start at flat index base = i
                // then repeatedly apply *n + j for each depth level
                let mut idx = i;
                for _ in 0..current_depth {
                    idx = idx * n + j;
                }
                jk[[i, j]] = if idx < flat.len() { flat[idx] } else { 0.0 };
            }
        }
        result.push(jk);

        // Advance current_fn to next_fn (wrapped in Arc)
        current_fn = std::sync::Arc::from(next_fn);
        current_out_len = current_fn(x).len();
        let _ = current_out_len;
        // Recompute from scratch to keep current_out_len accurate
        let probe = current_fn(x);
        current_out_len = probe.len();
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Mixed partial derivatives
// ─────────────────────────────────────────────────────────────────────────────

/// Compute an arbitrary mixed partial derivative via recursive central FD.
///
/// Given `indices = [(i₁, j₁), (i₂, j₂), …]`, computes
/// `∂f / ∂x_{i₁} ∂x_{j₁} ∂x_{i₂} ∂x_{j₂} …` by applying a single FD step
/// for each pair element.  The pair format `(i, _)` is used to specify which
/// variable to differentiate with respect to; the second element is unused and
/// reserved for future matrix-valued extensions.
///
/// In practice, each entry `(axis, _)` in `indices` adds one differentiation
/// with respect to `x[axis]`.  So `indices = &[(0, 0), (1, 0)]` gives
/// `∂²f / ∂x₀ ∂x₁`.
///
/// # Arguments
/// * `f`       – Scalar function `R^n → R`
/// * `x`       – Evaluation point
/// * `indices` – Sequence of `(axis, _)` specifying differentiation variables
///
/// # Returns
/// The mixed partial value at `x`.
///
/// # Errors
/// Returns `AutogradError` if `x` is empty, `indices` is empty, or any axis
/// is out of bounds.
///
/// # Example
/// ```rust
/// use scirs2_autograd::higher_order_new::mixed_partials;
///
/// // f(x,y) = x*y => ∂²f/∂x∂y = 1
/// let val = mixed_partials(
///     |xs| xs[0] * xs[1],
///     &[1.0, 2.0],
///     &[(0, 0), (1, 0)],
/// ).expect("mixed partials");
/// assert!((val - 1.0).abs() < 1e-3, "val={}", val);
/// ```
pub fn mixed_partials(
    f: impl Fn(&[f64]) -> f64 + 'static,
    x: &[f64],
    indices: &[(usize, usize)],
) -> Result<f64, AutogradError> {
    let n = x.len();
    if n == 0 {
        return Err(AutogradError::OperationError(
            "mixed_partials: x must be non-empty".to_string(),
        ));
    }
    if indices.is_empty() {
        return Err(AutogradError::OperationError(
            "mixed_partials: indices must be non-empty".to_string(),
        ));
    }
    for &(axis, _) in indices {
        if axis >= n {
            return Err(AutogradError::ShapeMismatch(format!(
                "mixed_partials: axis {} out of bounds (n={})",
                axis, n
            )));
        }
    }

    // We differentiate iteratively.  At each step we have a boxed closure
    // representing the current (possibly differentiated) scalar function.
    type ScalarFn = Box<dyn Fn(&[f64]) -> f64>;

    let f_box: ScalarFn = Box::new(f);

    // Wrap in Arc so we can clone into the FD closure
    let mut current: std::sync::Arc<dyn Fn(&[f64]) -> f64> = std::sync::Arc::from(f_box);

    for &(axis, _) in indices {
        let prev = current.clone();
        let diff_fn: ScalarFn = Box::new(move |xs: &[f64]| {
            let mut xp = xs.to_vec();
            let mut xm = xs.to_vec();
            xp[axis] = xs[axis] + H;
            xm[axis] = xs[axis] - H;
            (prev(&xp) - prev(&xm)) / (2.0 * H)
        });
        current = std::sync::Arc::from(diff_fn);
    }

    Ok(current(x))
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Laplacian
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Laplacian `∑_i ∂²f/∂x_i²` at point `x`.
///
/// Uses the second-order central FD formula for each diagonal Hessian entry:
/// `∂²f/∂x_i² ≈ (f(x+hᵢ) - 2f(x) + f(x-hᵢ)) / h²`
///
/// Total cost: `2n` function evaluations.
///
/// # Arguments
/// * `f` – Scalar function `R^n → R`
/// * `x` – Evaluation point
///
/// # Returns
/// Laplacian value at `x`.
///
/// # Errors
/// Returns `AutogradError` if `x` is empty.
///
/// # Example
/// ```rust
/// use scirs2_autograd::higher_order_new::laplacian;
///
/// // f(x,y) = x^2 + y^2 => Laplacian = 2 + 2 = 4
/// let lap = laplacian(|xs| xs[0]*xs[0] + xs[1]*xs[1], &[1.0, 2.0])
///     .expect("laplacian");
/// assert!((lap - 4.0).abs() < 1e-3, "lap={}", lap);
/// ```
pub fn laplacian(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Result<f64, AutogradError> {
    let n = x.len();
    if n == 0 {
        return Err(AutogradError::OperationError(
            "laplacian: x must be non-empty".to_string(),
        ));
    }

    let fx = f(x);
    let h2 = H * H;
    let mut sum = 0.0f64;

    let mut xp = x.to_vec();
    let mut xm = x.to_vec();

    for i in 0..n {
        xp[i] = x[i] + H;
        xm[i] = x[i] - H;
        let d2 = (f(&xp) - 2.0 * fx + f(&xm)) / h2;
        sum += d2;
        xp[i] = x[i];
        xm[i] = x[i];
    }

    Ok(sum)
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. N-th order Hessian-vector product
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the n-th order Hessian-vector product `H^(n)(f, x) · v`.
///
/// For `n = 1` this is the standard gradient `∇f(x) · v` (directional derivative).
/// For `n = 2` this is the standard HVP `H(f)·v`.
/// For `n ≥ 3` we apply the HVP recursively along direction `v`.
///
/// Formally: `HVP_n(x) = ∂^n f / ∂x^n [v, v, …, v]` (n-fold contraction).
///
/// # Arguments
/// * `f` – Scalar function `R^n → R`
/// * `x` – Evaluation point (length `n`)
/// * `v` – Direction vector (same length as `x`)
/// * `n` – Derivative order (≥ 1)
///
/// # Returns
/// `Array1<f64>` of length `n` (same shape as `x` and `v`).
///
/// # Errors
/// Returns `AutogradError` on dimension mismatch, empty inputs, or `n == 0`.
///
/// # Example
/// ```rust
/// use scirs2_autograd::higher_order_new::hessian_vector_product_nth_order;
///
/// // f(x) = x[0]^2 + x[1]^2, H = 2I, HVP(v=[1,0]) = [2, 0]
/// let hvp = hessian_vector_product_nth_order(
///     |xs| xs[0]*xs[0] + xs[1]*xs[1],
///     &[1.0, 1.0],
///     &[1.0, 0.0],
///     2,
/// ).expect("hvp nth order");
/// assert!((hvp[0] - 2.0).abs() < 1e-3, "hvp[0]={}", hvp[0]);
/// assert!(hvp[1].abs() < 1e-3, "hvp[1]={}", hvp[1]);
/// ```
pub fn hessian_vector_product_nth_order(
    f: impl Fn(&[f64]) -> f64,
    x: &[f64],
    v: &[f64],
    n: usize,
) -> Result<Array1<f64>, AutogradError> {
    let dim = x.len();
    if dim == 0 {
        return Err(AutogradError::OperationError(
            "hessian_vector_product_nth_order: x must be non-empty".to_string(),
        ));
    }
    if v.len() != dim {
        return Err(AutogradError::ShapeMismatch(format!(
            "hessian_vector_product_nth_order: x length {} != v length {}",
            dim,
            v.len()
        )));
    }
    if n == 0 {
        return Err(AutogradError::OperationError(
            "hessian_vector_product_nth_order: order n must be >= 1".to_string(),
        ));
    }

    // We represent the computation iteratively.
    // At each level we have a vector-valued function g: R^d -> R^d
    // where g_k = ∂(g_{k-1}·v) / ∂x.
    //
    // Level 0: g_0(x) = ∇f(x)
    // Level k: g_k(x) = (∇(g_{k-1}(x) · v))
    //
    // At level n-1 we evaluate g_{n-1}(x) · v to get the scalar, then
    // take one more gradient to get the n-th order HVP.
    //
    // This gives H^(n) · v^n as a vector via reverse-accumulation.
    //
    // For efficiency we implement this via FD at each level.

    let f_arc: std::sync::Arc<dyn Fn(&[f64]) -> f64> = std::sync::Arc::new(f);

    // Build the iterated directional-derivative gradient.
    // grad_k(x) = ∂ / ∂x [ grad_{k-1}(x) · v ]
    //
    // Base: grad_0(x) = ∇f(x)
    //
    // For n=1: return ∇f(x) · v (scalar -> reshape to [dim] using unit component)
    // Actually the signature says Array1<f64> of length n (the dim), so we return
    // the gradient at order n.

    // Level k: we have a scalar function h_k(x) = ∇f(x)·v (for k=1)
    //          or h_k(x) = ∇h_{k-1}(x)·v for k>1.
    // The n-th order HVP vector is ∇h_{n-1}(x) (gradient of the scalar h_{n-1}).

    if n == 1 {
        // HVP of order 1 = gradient of f contracted with v → NOT a vector in the usual sense.
        // We return the gradient ∇f(x) elementwise multiplied by v (rank-1 approximation).
        let g = gradient_fd(&*f_arc, x);
        let result: Vec<f64> = g.iter().zip(v.iter()).map(|(&gi, &vi)| gi * vi).collect();
        return Ok(Array1::from(result));
    }

    // Build h_1(x) = ∇f(x) · v
    let f_arc_1 = f_arc.clone();
    let v_owned: Vec<f64> = v.to_vec();
    let v_arc = std::sync::Arc::new(v_owned);

    let mut h: std::sync::Arc<dyn Fn(&[f64]) -> f64> = {
        let v_c = v_arc.clone();
        std::sync::Arc::new(move |xs: &[f64]| {
            gradient_fd(&*f_arc_1, xs)
                .iter()
                .zip(v_c.iter())
                .map(|(&g, &vi)| g * vi)
                .sum()
        })
    };

    // Build h_k for k = 2 .. n-1
    for _ in 2..n {
        let h_prev = h.clone();
        let v_c = v_arc.clone();
        h = std::sync::Arc::new(move |xs: &[f64]| {
            gradient_fd(&*h_prev, xs)
                .iter()
                .zip(v_c.iter())
                .map(|(&g, &vi)| g * vi)
                .sum()
        });
    }

    // The n-th order HVP vector = ∇h_{n-1}(x)
    let grad_h = gradient_fd(&*h, x);
    Ok(Array1::from(grad_h))
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Automatic Taylor expansion (local coefficients)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the local scalar Taylor coefficients of `f: R → R` at `x0`.
///
/// For a univariate function, returns `[a₀, a₁, …, aₙ]` where
/// `aₖ = f^(k)(x₀) / k!`.  The Taylor polynomial is then
/// `f(x₀ + t) ≈ ∑ₖ aₖ · tᵏ`.
///
/// For multivariate functions only the first component `x[0]` is varied;
/// all other components are held fixed.
///
/// # Arguments
/// * `f`       – Scalar function `R^n → R`
/// * `x0`      – Expansion point (length `n`)
/// * `n_terms` – Number of Taylor terms (order = `n_terms - 1`)
///
/// # Returns
/// `Vec<f64>` of length `n_terms` containing `[a₀, …, a_{n-1}]`.
///
/// # Errors
/// Returns `AutogradError` if `x0` is empty or `n_terms == 0`.
///
/// # Example
/// ```rust
/// use scirs2_autograd::higher_order_new::automatic_taylor_expansion;
///
/// // f(x) = exp(x), Taylor at x=0: coefficients = [1, 1, 0.5, 1/6, ...]
/// let coeffs = automatic_taylor_expansion(
///     |xs| xs[0].exp(),
///     &[0.0],
///     4,
/// ).expect("taylor exp");
/// assert!((coeffs[0] - 1.0).abs() < 1e-3, "a0={}", coeffs[0]);
/// assert!((coeffs[1] - 1.0).abs() < 1e-3, "a1={}", coeffs[1]);
/// assert!((coeffs[2] - 0.5).abs() < 1e-3, "a2={}", coeffs[2]);
/// ```
pub fn automatic_taylor_expansion(
    f: impl Fn(&[f64]) -> f64,
    x0: &[f64],
    n_terms: usize,
) -> Result<Vec<f64>, AutogradError> {
    if x0.is_empty() {
        return Err(AutogradError::OperationError(
            "automatic_taylor_expansion: x0 must be non-empty".to_string(),
        ));
    }
    if n_terms == 0 {
        return Err(AutogradError::OperationError(
            "automatic_taylor_expansion: n_terms must be >= 1".to_string(),
        ));
    }

    // We vary only the first component.  Direction v = e₀ = [1, 0, …, 0].
    let mut v = vec![0.0f64; x0.len()];
    v[0] = 1.0;

    // Re-use taylor_mode_forward with order = n_terms - 1
    let order = n_terms - 1;
    if order == 0 {
        return Ok(vec![f(x0)]);
    }

    taylor_mode_forward(f, x0, &v, order)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-2;

    // ── taylor_mode_forward ───────────────────────────────────────────────

    #[test]
    fn test_taylor_x_cubed() {
        // f(x) = x^3, expand at x=1 along v=1
        // c0=1, c1=3, c2=3, c3=1
        let c = taylor_mode_forward(|xs| xs[0].powi(3), &[1.0], &[1.0], 3)
            .expect("taylor x^3");
        assert!((c[0] - 1.0).abs() < TOL, "c0={}", c[0]);
        assert!((c[1] - 3.0).abs() < TOL, "c1={}", c[1]);
        assert!((c[2] - 3.0).abs() < TOL, "c2={}", c[2]);
        assert!((c[3] - 1.0).abs() < TOL, "c3={}", c[3]);
    }

    #[test]
    fn test_taylor_empty_x_error() {
        let r = taylor_mode_forward(|_| 0.0, &[], &[], 1);
        assert!(r.is_err());
    }

    #[test]
    fn test_taylor_order_zero_error() {
        let r = taylor_mode_forward(|xs| xs[0], &[1.0], &[1.0], 0);
        assert!(r.is_err());
    }

    // ── higher_order_jacobian ─────────────────────────────────────────────

    #[test]
    fn test_higher_order_jacobian_level1_identity() {
        // f(x) = x  => J1 = I
        let jacs = higher_order_jacobian(|xs| xs.to_vec(), &[1.0, 2.0], 1)
            .expect("jacobian identity");
        assert_eq!(jacs.len(), 1);
        let j = &jacs[0];
        assert!((j[[0, 0]] - 1.0).abs() < TOL);
        assert!(j[[0, 1]].abs() < TOL);
        assert!(j[[1, 0]].abs() < TOL);
        assert!((j[[1, 1]] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_higher_order_jacobian_level1_quadratic() {
        // f(x) = [x0^2, x1^2], J = diag([2x0, 2x1]) at x=[1,1]
        let jacs = higher_order_jacobian(
            |xs| vec![xs[0] * xs[0], xs[1] * xs[1]],
            &[1.0, 1.0],
            1,
        )
        .expect("jacobian quadratic");
        let j = &jacs[0];
        assert!((j[[0, 0]] - 2.0).abs() < TOL, "J[0,0]={}", j[[0, 0]]);
        assert!(j[[0, 1]].abs() < TOL, "J[0,1]={}", j[[0, 1]]);
        assert!(j[[1, 0]].abs() < TOL, "J[1,0]={}", j[[1, 0]]);
        assert!((j[[1, 1]] - 2.0).abs() < TOL, "J[1,1]={}", j[[1, 1]]);
    }

    #[test]
    fn test_higher_order_jacobian_level2() {
        // f(x) = [x0^2, x1^2], J2 (super-diagonal) should be [[2,0],[0,2]]
        let jacs = higher_order_jacobian(
            |xs| vec![xs[0] * xs[0], xs[1] * xs[1]],
            &[1.0, 1.0],
            2,
        )
        .expect("jacobian level2");
        assert_eq!(jacs.len(), 2);
        let j2 = &jacs[1];
        assert!((j2[[0, 0]] - 2.0).abs() < TOL, "J2[0,0]={}", j2[[0, 0]]);
        assert!((j2[[1, 1]] - 2.0).abs() < TOL, "J2[1,1]={}", j2[[1, 1]]);
    }

    // ── mixed_partials ────────────────────────────────────────────────────

    #[test]
    fn test_mixed_partials_xy() {
        // f(x,y) = x*y => ∂²f/∂x∂y = 1
        let v = mixed_partials(|xs| xs[0] * xs[1], &[1.0, 2.0], &[(0, 0), (1, 0)])
            .expect("mixed xy");
        assert!((v - 1.0).abs() < TOL, "val={}", v);
    }

    #[test]
    fn test_mixed_partials_xx() {
        // f(x) = x^2 => ∂²f/∂x² = 2
        let v = mixed_partials(|xs| xs[0] * xs[0], &[3.0], &[(0, 0), (0, 0)])
            .expect("mixed xx");
        assert!((v - 2.0).abs() < TOL, "val={}", v);
    }

    #[test]
    fn test_mixed_partials_out_of_bounds_error() {
        let r = mixed_partials(|xs| xs[0], &[1.0], &[(5, 0)]);
        assert!(r.is_err());
    }

    // ── laplacian ─────────────────────────────────────────────────────────

    #[test]
    fn test_laplacian_quadratic() {
        // f(x,y) = x^2 + y^2 => Lap = 4
        let lap = laplacian(|xs| xs[0] * xs[0] + xs[1] * xs[1], &[1.0, 2.0])
            .expect("laplacian");
        assert!((lap - 4.0).abs() < TOL, "lap={}", lap);
    }

    #[test]
    fn test_laplacian_harmonic() {
        // f(x,y,z) = x^2 - y^2 => Lap = 2 - 2 = 0
        let lap = laplacian(
            |xs| xs[0] * xs[0] - xs[1] * xs[1],
            &[1.0, 1.0, 1.0],
        )
        .expect("laplacian harmonic");
        assert!(lap.abs() < TOL, "lap={}", lap);
    }

    #[test]
    fn test_laplacian_empty_error() {
        let r = laplacian(|_| 0.0, &[]);
        assert!(r.is_err());
    }

    // ── hessian_vector_product_nth_order ──────────────────────────────────

    #[test]
    fn test_hvp_order2_quadratic() {
        // f(x,y) = x^2 + y^2, H = 2I, HVP(v=[1,0]) = [2, 0]
        let hvp = hessian_vector_product_nth_order(
            |xs| xs[0] * xs[0] + xs[1] * xs[1],
            &[1.0, 1.0],
            &[1.0, 0.0],
            2,
        )
        .expect("hvp order 2");
        assert!((hvp[0] - 2.0).abs() < TOL, "hvp[0]={}", hvp[0]);
        assert!(hvp[1].abs() < TOL, "hvp[1]={}", hvp[1]);
    }

    #[test]
    fn test_hvp_order1_gradient() {
        // f(x) = x^2, order=1, v=[1] => result ≈ 2x * 1 = 2 at x=1
        let hvp = hessian_vector_product_nth_order(
            |xs| xs[0] * xs[0],
            &[1.0],
            &[1.0],
            1,
        )
        .expect("hvp order 1");
        assert!((hvp[0] - 2.0).abs() < TOL, "hvp[0]={}", hvp[0]);
    }

    #[test]
    fn test_hvp_order_zero_error() {
        let r = hessian_vector_product_nth_order(|xs| xs[0], &[1.0], &[1.0], 0);
        assert!(r.is_err());
    }

    // ── automatic_taylor_expansion ────────────────────────────────────────

    #[test]
    fn test_taylor_exp_coefficients() {
        // e^x at x=0: [1, 1, 0.5, 1/6]
        let c = automatic_taylor_expansion(|xs| xs[0].exp(), &[0.0], 4)
            .expect("taylor exp");
        assert!((c[0] - 1.0).abs() < TOL, "c0={}", c[0]);
        assert!((c[1] - 1.0).abs() < TOL, "c1={}", c[1]);
        assert!((c[2] - 0.5).abs() < TOL, "c2={}", c[2]);
        assert!((c[3] - 1.0 / 6.0).abs() < 0.05, "c3={}", c[3]);
    }

    #[test]
    fn test_taylor_polynomial() {
        // f(x) = 2 + 3x + 4x^2, at x=0:
        // Taylor series: f(x) = sum_k c_k * x^k where c_k = f^(k)(0) / k!
        // c0 = f(0) = 2, c1 = f'(0)/1! = 3, c2 = f''(0)/2! = 8/2 = 4
        let c = automatic_taylor_expansion(
            |xs| 2.0 + 3.0 * xs[0] + 4.0 * xs[0] * xs[0],
            &[0.0],
            3,
        )
        .expect("taylor poly");
        assert!((c[0] - 2.0).abs() < TOL, "c0={}", c[0]);
        assert!((c[1] - 3.0).abs() < TOL, "c1={}", c[1]);
        assert!((c[2] - 4.0).abs() < TOL, "c2={}", c[2]);
    }

    #[test]
    fn test_taylor_n_terms_zero_error() {
        let r = automatic_taylor_expansion(|xs| xs[0], &[0.0], 0);
        assert!(r.is_err());
    }
}
