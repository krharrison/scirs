//! Higher-order derivatives for automatic differentiation
//!
//! This module provides utilities for computing derivatives beyond the first
//! order.  It is designed to complement [`crate::functional::transforms`] with
//! specialised, efficient algorithms for:
//!
//! * Mixed partial derivatives of arbitrary order
//! * The Laplacian (trace of the Hessian) computed without the full O(n²) cost
//! * Taylor-mode forward propagation of high-order coefficients
//! * Iterated gradient / Hessian applications
//! * Efficient higher-order Jacobian sequences
//!
//! # Overview
//!
//! | Function | Description | Cost |
//! |----------|-------------|------|
//! | [`mixed_partial`] | `∂ᵏf / (∂xᵢ₁ … ∂xᵢₖ)` | O(2ᵏ) evals |
//! | [`laplacian`] | `Σᵢ ∂²f/∂xᵢ²` | `2n` + 1 evals |
//! | [`laplacian_stochastic`] | Hutchinson estimator of Laplacian | O(m) evals |
//! | [`taylor_coefficients`] | `f⁽ᵏ⁾(x)·vᵏ / k!` up to order n | O(n·2ⁿ) evals |
//! | [`hessian_diagonal`] | Diagonal of Hessian | O(2n) evals |
//! | [`hvp`] | Hessian-vector product H·v | O(n) evals |
//! | [`iterated_gradient`] | Apply `∇` k times | O(nᵏ) evals |
//! | [`jacobian_sequence`] | J, J', J'', … up to `order` | O(n·order) evals |
//! | [`nth_derivative_scalar`] | n-th derivative of univariate f | O(2ⁿ) evals |

use crate::error::AutogradError;
use crate::Result as AgResult;

/// Central finite-difference step.
const H: f64 = 1e-4;

// ============================================================================
// mixed_partial — arbitrary mixed partial derivative
// ============================================================================

/// Compute an arbitrary mixed partial derivative of `f` at `x`.
///
/// Given a list of axis indices `axes = [i₁, i₂, …, iₖ]`, this computes
/// `∂ᵏf / (∂x_{i₁} ∂x_{i₂} … ∂x_{iₖ})` evaluated at `x`.
///
/// The implementation uses the *iterated central finite-difference* approach:
/// the k-th partial is obtained by applying a first-order FD to the (k-1)-th
/// partial along axis `iₖ`.
///
/// # Cost
///
/// `2ᵏ` function evaluations (exponential in the order `k`).
///
/// # Errors
///
/// Returns `AutogradError` if any axis index is out of bounds.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::higher_order::mixed_partial;
///
/// // f(x,y,z) = x^2 * y * z; ∂³f/(∂x ∂y ∂z) = 2x * 1 * 1 = 2x
/// let val = mixed_partial(
///     |xs: &[f64]| xs[0]*xs[0]*xs[1]*xs[2],
///     &[1.0, 2.0, 3.0],
///     &[0, 1, 2],
/// ).expect("mixed partial");
/// // ∂³f/∂x∂y∂z = 2x = 2.0 at x=1
/// assert!((val - 2.0).abs() < 1e-2, "mixed partial = {val}");
/// ```
pub fn mixed_partial<F>(f: F, x: &[f64], axes: &[usize]) -> AgResult<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    for &ax in axes {
        if ax >= n {
            return Err(AutogradError::invalid_argument(format!(
                "mixed_partial: axis {ax} out of bounds for input of length {n}"
            )));
        }
    }

    if axes.is_empty() {
        return Ok(f(x));
    }

    Ok(mixed_partial_impl(&f, x, axes))
}

/// Internal recursive implementation of mixed_partial.
fn mixed_partial_impl(f: &dyn Fn(&[f64]) -> f64, x: &[f64], axes: &[usize]) -> f64 {
    if axes.is_empty() {
        return f(x);
    }

    let ax = axes[0];
    let remaining = &axes[1..];

    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[ax] = x[ax] + H;
    xm[ax] = x[ax] - H;

    let fp = mixed_partial_impl(f, &xp, remaining);
    let fm = mixed_partial_impl(f, &xm, remaining);

    (fp - fm) / (2.0 * H)
}

// ============================================================================
// laplacian — sum of diagonal Hessian entries
// ============================================================================

/// Compute the Laplacian `Δf(x) = Σᵢ ∂²f/∂xᵢ²` at point `x`.
///
/// Uses second-order central finite differences for each diagonal entry:
/// `∂²f/∂xᵢ² ≈ (f(x+hᵢ) - 2f(x) + f(x-hᵢ)) / h²`
///
/// This requires `1 + 2n` function evaluations (much cheaper than the full
/// Hessian which needs O(n²)).
///
/// # Errors
///
/// Returns `AutogradError` if `x` is empty.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::higher_order::laplacian;
///
/// // f(x,y,z) = x^2 + y^2 + z^2; Δf = 2 + 2 + 2 = 6
/// let lap = laplacian(
///     |xs: &[f64]| xs[0]*xs[0] + xs[1]*xs[1] + xs[2]*xs[2],
///     &[1.0, 2.0, 3.0],
/// ).expect("laplacian");
/// assert!((lap - 6.0).abs() < 1e-2, "Δf = {lap}");
/// ```
pub fn laplacian<F>(f: F, x: &[f64]) -> AgResult<f64>
where
    F: Fn(&[f64]) -> f64,
{
    if x.is_empty() {
        return Err(AutogradError::invalid_argument(
            "laplacian: input must be non-empty".to_string(),
        ));
    }

    let fx = f(x);
    let h2 = H * H;
    let n = x.len();
    let mut sum = 0.0f64;

    let mut xp = x.to_vec();
    let mut xm = x.to_vec();

    for i in 0..n {
        xp[i] = x[i] + H;
        xm[i] = x[i] - H;
        sum += (f(&xp) + f(&xm) - 2.0 * fx) / h2;
        xp[i] = x[i];
        xm[i] = x[i];
    }

    Ok(sum)
}

/// Stochastic (Hutchinson) estimator for the Laplacian using `m` random probe
/// vectors.
///
/// Computes `E[vᵀ H v]` where `v ~ Rademacher(±1)`, which is an unbiased
/// estimator of `tr(H) = Δf`.  This reduces cost from O(n) evaluations to
/// O(m) evaluations (where `m << n` is practical when `n` is very large).
///
/// # Arguments
///
/// * `f`       — Scalar function.
/// * `x`       — Evaluation point.
/// * `m`       — Number of probe vectors.
/// * `seed`    — RNG seed for reproducibility.
///
/// # Errors
///
/// Returns `AutogradError` if `x` is empty or `m == 0`.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::higher_order::laplacian_stochastic;
///
/// // f = x^2 + y^2 + z^2; exact Laplacian = 6
/// // With m=200 the estimate should be within ±1
/// let est = laplacian_stochastic(
///     |xs: &[f64]| xs.iter().map(|v| v*v).sum::<f64>(),
///     &[1.0, 2.0, 3.0],
///     200,
///     42,
/// ).expect("hutchinson laplacian");
/// assert!((est - 6.0).abs() < 2.0, "Δf estimate = {est}");
/// ```
pub fn laplacian_stochastic<F>(f: F, x: &[f64], m: usize, seed: u64) -> AgResult<f64>
where
    F: Fn(&[f64]) -> f64,
{
    if x.is_empty() {
        return Err(AutogradError::invalid_argument(
            "laplacian_stochastic: input must be non-empty".to_string(),
        ));
    }
    if m == 0 {
        return Err(AutogradError::invalid_argument(
            "laplacian_stochastic: m must be positive".to_string(),
        ));
    }

    let n = x.len();
    let h2 = H * H;
    let mut total = 0.0f64;

    // Simple LCG PRNG seeded by `seed`
    let mut rng_state = seed.wrapping_add(1);
    let lcg_next = |s: &mut u64| -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s
    };

    for _ in 0..m {
        // Rademacher vector: ±1
        let v: Vec<f64> = (0..n)
            .map(|_| if lcg_next(&mut rng_state) & 1 == 0 { 1.0 } else { -1.0 })
            .collect();

        // Compute vᵀ H v via FD: (f(x+hv) - 2f(x) + f(x-hv)) / h²
        let xp: Vec<f64> = x.iter().zip(v.iter()).map(|(&xi, &vi)| xi + H * vi).collect();
        let xm: Vec<f64> = x.iter().zip(v.iter()).map(|(&xi, &vi)| xi - H * vi).collect();
        let fx = f(x);
        let estimate = (f(&xp) + f(&xm) - 2.0 * fx) / h2;
        total += estimate;
    }

    Ok(total / m as f64)
}

// ============================================================================
// hessian_diagonal — diagonal entries of the Hessian
// ============================================================================

/// Compute only the diagonal entries of the Hessian `H[i][i] = ∂²f/∂xᵢ²`.
///
/// Much cheaper than the full Hessian when only the diagonal is needed (e.g.
/// for diagonal preconditioners or curvature-aware learning rate schedules).
///
/// # Cost
///
/// `1 + 2n` function evaluations.
///
/// # Errors
///
/// Returns `AutogradError` if `x` is empty.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::higher_order::hessian_diagonal;
///
/// // f(x,y) = x^3 + y^2; H_diag = [6x, 2] at (2, 1) = [12, 2]
/// let d = hessian_diagonal(
///     |xs: &[f64]| xs[0].powi(3) + xs[1].powi(2),
///     &[2.0, 1.0],
/// ).expect("hessian diagonal");
/// assert!((d[0] - 12.0).abs() < 1e-1, "H[0][0] = {}", d[0]);
/// assert!((d[1] - 2.0).abs() < 1e-1,  "H[1][1] = {}", d[1]);
/// ```
pub fn hessian_diagonal<F>(f: F, x: &[f64]) -> AgResult<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    if x.is_empty() {
        return Err(AutogradError::invalid_argument(
            "hessian_diagonal: input must be non-empty".to_string(),
        ));
    }

    let n = x.len();
    let fx = f(x);
    let h2 = H * H;
    let mut diag = vec![0.0f64; n];

    let mut xp = x.to_vec();
    let mut xm = x.to_vec();

    for i in 0..n {
        xp[i] = x[i] + H;
        xm[i] = x[i] - H;
        diag[i] = (f(&xp) + f(&xm) - 2.0 * fx) / h2;
        xp[i] = x[i];
        xm[i] = x[i];
    }

    Ok(diag)
}

// ============================================================================
// hvp — Hessian-vector product
// ============================================================================

/// Compute the Hessian-vector product `H(x) · v` efficiently.
///
/// Uses the forward-over-reverse trick via FD:
/// `H·v ≈ (∇f(x + h·v) − ∇f(x − h·v)) / (2h)`
///
/// This requires `4n` function evaluations (two gradient calls, each O(2n)).
///
/// # Errors
///
/// Returns `AutogradError` if `x` is empty or dimensions mismatch.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::higher_order::hvp;
///
/// // f(x,y) = x^2 + y^2; H = 2I; H·[1,2] = [2,4]
/// let hv = hvp(
///     |xs: &[f64]| xs[0]*xs[0] + xs[1]*xs[1],
///     &[3.0, 4.0],
///     &[1.0, 2.0],
/// ).expect("hvp");
/// assert!((hv[0] - 2.0).abs() < 1e-2, "hv[0] = {}", hv[0]);
/// assert!((hv[1] - 4.0).abs() < 1e-2, "hv[1] = {}", hv[1]);
/// ```
pub fn hvp<F>(f: F, x: &[f64], v: &[f64]) -> AgResult<Vec<f64>>
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
        return Err(AutogradError::invalid_argument(format!(
            "hvp: v has length {} but x has length {n}",
            v.len()
        )));
    }

    // Compute gradient via central FD
    let grad_fd = |pt: &[f64]| -> Vec<f64> {
        let mut g = vec![0.0f64; n];
        let mut pp = pt.to_vec();
        let mut pm = pt.to_vec();
        for i in 0..n {
            pp[i] = pt[i] + H;
            pm[i] = pt[i] - H;
            g[i] = (f(&pp) - f(&pm)) / (2.0 * H);
            pp[i] = pt[i];
            pm[i] = pt[i];
        }
        g
    };

    let xp: Vec<f64> = x.iter().zip(v.iter()).map(|(&xi, &vi)| xi + H * vi).collect();
    let xm: Vec<f64> = x.iter().zip(v.iter()).map(|(&xi, &vi)| xi - H * vi).collect();

    let gp = grad_fd(&xp);
    let gm = grad_fd(&xm);

    let hv: Vec<f64> = gp
        .iter()
        .zip(gm.iter())
        .map(|(&a, &b)| (a - b) / (2.0 * H))
        .collect();

    Ok(hv)
}

// ============================================================================
// taylor_coefficients — Taylor expansion coefficients
// ============================================================================

/// Compute Taylor coefficients `cₖ = f⁽ᵏ⁾(x; v) / k!` for `k = 0, …, order`.
///
/// Returns the vector `[c₀, c₁, …, c_order]` where `cₖ` is the k-th
/// normalised Taylor coefficient along direction `v`:
///
/// * `c₀ = f(x)`
/// * `c₁ = ∇f(x) · v`
/// * `c₂ = (1/2) d²f/dt² |_{t=0}` along v
/// * …
///
/// These are the coefficients of the univariate Taylor expansion
/// `f(x + t·v) = Σₖ cₖ tᵏ`.
///
/// # Cost
///
/// `O(2ᵏ)` function evaluations for each coefficient (exponential in order),
/// via nested central FD.  Practical only for small orders (≤ 6).
///
/// # Errors
///
/// Returns `AutogradError` if `x` is empty, `v.len() != x.len()`, or
/// `order > 8` (to prevent exponential blowup).
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::higher_order::taylor_coefficients;
///
/// // f(x) = exp(x); f(x + t) = exp(x) * (1 + t + t²/2 + …)
/// // Taylor coefficients along v=1 at x=0: c_k = 1/k!
/// let cs = taylor_coefficients(
///     |xs: &[f64]| xs[0].exp(),
///     &[0.0],
///     &[1.0],
///     4,
/// ).expect("taylor coefficients");
/// // c0=1, c1=1, c2=0.5, c3=1/6, c4=1/24
/// assert!((cs[0] - 1.0).abs() < 1e-6, "c0 = {}", cs[0]);
/// assert!((cs[1] - 1.0).abs() < 1e-3, "c1 = {}", cs[1]);
/// assert!((cs[2] - 0.5).abs() < 1e-2, "c2 = {}", cs[2]);
/// ```
pub fn taylor_coefficients<F>(
    f: F,
    x: &[f64],
    v: &[f64],
    order: usize,
) -> AgResult<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    if x.is_empty() {
        return Err(AutogradError::invalid_argument(
            "taylor_coefficients: input must be non-empty".to_string(),
        ));
    }
    if v.len() != x.len() {
        return Err(AutogradError::invalid_argument(format!(
            "taylor_coefficients: v has length {} but x has length {}",
            v.len(),
            x.len()
        )));
    }
    if order > 8 {
        return Err(AutogradError::invalid_argument(
            "taylor_coefficients: order > 8 would require too many function evaluations"
                .to_string(),
        ));
    }

    let mut coeffs = vec![0.0f64; order + 1];

    for k in 0..=order {
        let raw = directional_derivative_recursive(&f, x, v, k);
        // Divide by k! to get Taylor coefficient
        let k_factorial: f64 = (1..=k).map(|i| i as f64).product();
        coeffs[k] = raw / k_factorial;
    }

    Ok(coeffs)
}

/// Compute the k-th directional derivative `d^k f(x; v) / dt^k |_{t=0}` via
/// nested central FD.
fn directional_derivative_recursive(
    f: &dyn Fn(&[f64]) -> f64,
    x: &[f64],
    v: &[f64],
    k: usize,
) -> f64 {
    match k {
        0 => f(x),
        1 => {
            // Central FD directional derivative
            let xp: Vec<f64> = x.iter().zip(v.iter()).map(|(&xi, &vi)| xi + H * vi).collect();
            let xm: Vec<f64> = x.iter().zip(v.iter()).map(|(&xi, &vi)| xi - H * vi).collect();
            (f(&xp) - f(&xm)) / (2.0 * H)
        }
        n => {
            // Recurse via FD of (n-1)-th derivative
            let xp: Vec<f64> = x.iter().zip(v.iter()).map(|(&xi, &vi)| xi + H * vi).collect();
            let xm: Vec<f64> = x.iter().zip(v.iter()).map(|(&xi, &vi)| xi - H * vi).collect();
            let dp = directional_derivative_recursive(f, &xp, v, n - 1);
            let dm = directional_derivative_recursive(f, &xm, v, n - 1);
            (dp - dm) / (2.0 * H)
        }
    }
}

// ============================================================================
// nth_derivative_scalar — n-th derivative of univariate function
// ============================================================================

/// Compute the n-th derivative `f⁽ⁿ⁾(x)` of a univariate scalar function.
///
/// Uses the n-point finite-difference formula via the backward difference
/// operator and binomial coefficients:
///
/// `f⁽ⁿ⁾(x) ≈ (1/hⁿ) Σₖ₌₀ⁿ (-1)ᵏ C(n,k) f(x + (n/2 - k)h)`
///
/// # Errors
///
/// Returns `AutogradError` if `n > 10` (to avoid catastrophic cancellation).
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::higher_order::nth_derivative_scalar;
///
/// // f(x) = x^4; f''(x) = 12x^2 = 12 at x=1
/// let d2 = nth_derivative_scalar(|x: f64| x.powi(4), 1.0, 2).expect("2nd deriv");
/// assert!((d2 - 12.0).abs() < 0.5, "f''(1) = {d2}");
///
/// // f(x) = sin(x); f'(0) = 1
/// let d1 = nth_derivative_scalar(|x: f64| x.sin(), 0.0, 1).expect("1st deriv");
/// assert!((d1 - 1.0).abs() < 1e-3, "sin'(0) = {d1}");
/// ```
pub fn nth_derivative_scalar<F>(f: F, x: f64, n: usize) -> AgResult<f64>
where
    F: Fn(f64) -> f64,
{
    if n == 0 {
        return Ok(f(x));
    }
    if n > 10 {
        return Err(AutogradError::invalid_argument(
            "nth_derivative_scalar: n > 10 is numerically unreliable".to_string(),
        ));
    }

    // Use a step size scaled by the order to balance truncation and round-off
    let h_ord = H.powf(1.0 / n as f64).max(1e-6_f64.powf(1.0 / n as f64));
    // For orders 1 and 2 use a fixed reliable step
    let h_step = match n {
        1 => 1e-5_f64,
        2 => 1e-4_f64,
        3 | 4 => 1e-3_f64,
        _ => h_ord,
    };

    // Central finite difference: use the centred n-point stencil
    // Stencil points: x + (n/2 - k) * h for k = 0, ..., n
    // Weights are (-1)^k * C(n,k) / h^n
    let h_n = h_step.powi(n as i32);
    let mut sum = 0.0f64;
    for k in 0..=(n as u32) {
        let binom = binomial(n as u32, k);
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let shift = (n as f64 / 2.0 - k as f64) * h_step;
        sum += sign * binom as f64 * f(x + shift);
    }

    Ok(sum / h_n)
}

/// Compute the binomial coefficient C(n, k).
fn binomial(n: u32, k: u32) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k); // use smaller k for efficiency
    let mut result = 1u64;
    for i in 0..k {
        result = result * (n - i) as u64 / (i + 1) as u64;
    }
    result
}

// ============================================================================
// iterated_gradient — apply ∇ multiple times
// ============================================================================

/// Apply the gradient operator `k` times to `f`, returning the k-th iterated
/// gradient as a flat vector.
///
/// * `k = 0`: returns `f(x)` as a single-element vector.
/// * `k = 1`: returns `∇f(x)` (the standard gradient), length n.
/// * `k = 2`: returns a flattened `n × n` Hessian row-major, length n².
///
/// For `k > 2` the output grows as `nᵏ` entries and computation is expensive.
///
/// # Errors
///
/// Returns `AutogradError` if `x` is empty or `k > 3`.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::higher_order::iterated_gradient;
///
/// // k=1: ∇(x^2+y^2) at (3,4) = [6, 8]
/// let g1 = iterated_gradient(
///     |xs: &[f64]| xs[0]*xs[0] + xs[1]*xs[1],
///     &[3.0, 4.0],
///     1,
/// ).expect("grad");
/// assert!((g1[0] - 6.0).abs() < 1e-3, "∇f[0] = {}", g1[0]);
/// assert!((g1[1] - 8.0).abs() < 1e-3, "∇f[1] = {}", g1[1]);
/// ```
pub fn iterated_gradient<F>(f: F, x: &[f64], k: usize) -> AgResult<Vec<f64>>
where
    F: Fn(&[f64]) -> f64 + Clone,
{
    if x.is_empty() {
        return Err(AutogradError::invalid_argument(
            "iterated_gradient: input must be non-empty".to_string(),
        ));
    }
    if k > 3 {
        return Err(AutogradError::invalid_argument(
            "iterated_gradient: k > 3 would produce n^k outputs and is prohibitively expensive"
                .to_string(),
        ));
    }

    match k {
        0 => Ok(vec![f(x)]),
        1 => {
            let n = x.len();
            let mut g = vec![0.0f64; n];
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            for i in 0..n {
                xp[i] = x[i] + H;
                xm[i] = x[i] - H;
                g[i] = (f(&xp) - f(&xm)) / (2.0 * H);
                xp[i] = x[i];
                xm[i] = x[i];
            }
            Ok(g)
        }
        2 => {
            // Flatten Hessian row-major
            let n = x.len();
            let fx = f(x);
            let h2 = H * H;
            let mut mat = vec![0.0f64; n * n];
            // Diagonal
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            for i in 0..n {
                xp[i] = x[i] + H;
                xm[i] = x[i] - H;
                mat[i * n + i] = (f(&xp) + f(&xm) - 2.0 * fx) / h2;
                xp[i] = x[i];
                xm[i] = x[i];
            }
            // Off-diagonal (symmetric)
            for i in 0..n {
                for j in (i + 1)..n {
                    let mut xpp = x.to_vec();
                    let mut xpm = x.to_vec();
                    let mut xmp = x.to_vec();
                    let mut xmm = x.to_vec();
                    xpp[i] += H;
                    xpp[j] += H;
                    xpm[i] += H;
                    xpm[j] -= H;
                    xmp[i] -= H;
                    xmp[j] += H;
                    xmm[i] -= H;
                    xmm[j] -= H;
                    let val = (f(&xpp) - f(&xpm) - f(&xmp) + f(&xmm)) / (4.0 * h2);
                    mat[i * n + j] = val;
                    mat[j * n + i] = val;
                }
            }
            Ok(mat)
        }
        3 => {
            // Third-order tensor (n³ entries): use FD on Hessian
            let n = x.len();
            let mut tensor = vec![0.0f64; n * n * n];
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            // ∂³f/∂xᵢ∂xⱼ∂xₖ via FD on Hessian along axis k
            for k_ax in 0..n {
                xp[k_ax] = x[k_ax] + H;
                xm[k_ax] = x[k_ax] - H;
                let hess_p = hessian_matrix(&f, &xp);
                let hess_m = hessian_matrix(&f, &xm);
                xp[k_ax] = x[k_ax];
                xm[k_ax] = x[k_ax];
                for i in 0..n {
                    for j in 0..n {
                        tensor[i * n * n + j * n + k_ax] =
                            (hess_p[i][j] - hess_m[i][j]) / (2.0 * H);
                    }
                }
            }
            Ok(tensor)
        }
        _ => unreachable!(),
    }
}

/// Internal helper: compute the full Hessian matrix.
fn hessian_matrix(f: &dyn Fn(&[f64]) -> f64, x: &[f64]) -> Vec<Vec<f64>> {
    let n = x.len();
    let fx = f(x);
    let h2 = H * H;
    let mut hess = vec![vec![0.0f64; n]; n];

    let mut xp = x.to_vec();
    let mut xm = x.to_vec();

    for i in 0..n {
        xp[i] = x[i] + H;
        xm[i] = x[i] - H;
        hess[i][i] = (f(&xp) + f(&xm) - 2.0 * fx) / h2;
        xp[i] = x[i];
        xm[i] = x[i];
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let mut xpp = x.to_vec();
            let mut xpm = x.to_vec();
            let mut xmp = x.to_vec();
            let mut xmm = x.to_vec();
            xpp[i] += H;
            xpp[j] += H;
            xpm[i] += H;
            xpm[j] -= H;
            xmp[i] -= H;
            xmp[j] += H;
            xmm[i] -= H;
            xmm[j] -= H;
            let val = (f(&xpp) - f(&xpm) - f(&xmp) + f(&xmm)) / (4.0 * h2);
            hess[i][j] = val;
            hess[j][i] = val;
        }
    }
    hess
}

// ============================================================================
// jacobian_sequence — sequence of Jacobians up to given order
// ============================================================================

/// Compute the sequence of Jacobians `J, J', J'', …` up to `order`.
///
/// Returns a `Vec` of `order + 1` items.  Item `k` is the k-th Jacobian
/// of `f`, flattened row-major with shape `m × n^(k+1)`.
///
/// * Order 0: `f(x)` itself (length `m`).
/// * Order 1: the standard Jacobian `J(x)` (m × n, flattened to m*n).
/// * Order 2: the Jacobian of the Jacobian (m × n × n, flattened to m*n²).
///
/// # Errors
///
/// Returns `AutogradError` if `x` is empty or `order > 2`.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::functional::higher_order::jacobian_sequence;
///
/// // f(x,y) = [x^2, x*y]; J = [[2x,0],[y,x]] at (2,3) = [[4,0],[3,2]]
/// let seq = jacobian_sequence(
///     |xs: &[f64]| vec![xs[0]*xs[0], xs[0]*xs[1]],
///     &[2.0, 3.0],
///     1,
/// ).expect("jacobian sequence");
/// let j = &seq[1]; // order-1 Jacobian, flattened [m*n]
/// // seq[1] = [J[0][0], J[0][1], J[1][0], J[1][1]] = [4, 0, 3, 2]
/// assert!((j[0] - 4.0).abs() < 1e-3, "J[0][0] = {}", j[0]);
/// assert!((j[2] - 3.0).abs() < 1e-3, "J[1][0] = {}", j[2]);
/// ```
pub fn jacobian_sequence<F>(
    f: F,
    x: &[f64],
    order: usize,
) -> AgResult<Vec<Vec<f64>>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if x.is_empty() {
        return Err(AutogradError::invalid_argument(
            "jacobian_sequence: input must be non-empty".to_string(),
        ));
    }
    if order > 2 {
        return Err(AutogradError::invalid_argument(
            "jacobian_sequence: order > 2 is not supported".to_string(),
        ));
    }

    let n = x.len();
    let fx = f(x);
    let m = fx.len();
    let mut result = Vec::with_capacity(order + 1);

    // Order 0: function values
    result.push(fx.clone());

    if order == 0 {
        return Ok(result);
    }

    // Order 1: Jacobian (m*n entries, row-major)
    let mut j1 = vec![0.0f64; m * n];
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    let two_h = 2.0 * H;

    for col in 0..n {
        xp[col] = x[col] + H;
        xm[col] = x[col] - H;
        let fp = f(&xp);
        let fm = f(&xm);
        for row in 0..m {
            j1[row * n + col] = (fp[row] - fm[row]) / two_h;
        }
        xp[col] = x[col];
        xm[col] = x[col];
    }
    result.push(j1.clone());

    if order == 1 {
        return Ok(result);
    }

    // Order 2: Jacobian of Jacobian (m*n*n entries)
    // ∂²fᵣ/(∂xᵢ∂xⱼ) — compute by FD on J along each axis j
    let mut j2 = vec![0.0f64; m * n * n];

    for j_ax in 0..n {
        xp[j_ax] = x[j_ax] + H;
        xm[j_ax] = x[j_ax] - H;

        // J at x + h*eⱼ
        let mut jp = vec![0.0f64; m * n];
        for col in 0..n {
            let mut xpp = xp.clone();
            let mut xpm = xp.clone();
            xpp[col] += H;
            xpm[col] -= H;
            let fp2 = f(&xpp);
            let fm2 = f(&xpm);
            for row in 0..m {
                jp[row * n + col] = (fp2[row] - fm2[row]) / two_h;
            }
        }

        // J at x - h*eⱼ
        let mut jm = vec![0.0f64; m * n];
        for col in 0..n {
            let mut xmp = xm.clone();
            let mut xmm = xm.clone();
            xmp[col] += H;
            xmm[col] -= H;
            let fp2 = f(&xmp);
            let fm2 = f(&xmm);
            for row in 0..m {
                jm[row * n + col] = (fp2[row] - fm2[row]) / two_h;
            }
        }

        // ∂J/∂x_{j_ax}
        for row in 0..m {
            for col in 0..n {
                j2[row * n * n + col * n + j_ax] =
                    (jp[row * n + col] - jm[row * n + col]) / two_h;
            }
        }

        xp[j_ax] = x[j_ax];
        xm[j_ax] = x[j_ax];
    }
    result.push(j2);

    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_partial_cubic() {
        // f(x,y,z) = x^2 * y * z; ∂³f/(∂x ∂y ∂z) = 2x at x=1
        let val = mixed_partial(
            |xs: &[f64]| xs[0] * xs[0] * xs[1] * xs[2],
            &[1.0, 2.0, 3.0],
            &[0, 1, 2],
        )
        .expect("mixed partial");
        assert!((val - 2.0).abs() < 0.5, "∂³f = {val}");
    }

    #[test]
    fn test_mixed_partial_second_order() {
        // f(x,y) = x*y; ∂²f/∂x∂y = 1
        let val = mixed_partial(
            |xs: &[f64]| xs[0] * xs[1],
            &[1.0, 1.0],
            &[0, 1],
        )
        .expect("2nd order mixed");
        assert!((val - 1.0).abs() < 1e-2, "∂²f/∂x∂y = {val}");
    }

    #[test]
    fn test_mixed_partial_empty_axes() {
        // Empty axes => f(x)
        let val = mixed_partial(|xs: &[f64]| xs[0] * xs[0], &[3.0], &[]).expect("zero order");
        assert!((val - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_mixed_partial_oob_axis() {
        assert!(mixed_partial(|xs: &[f64]| xs[0], &[1.0], &[5]).is_err());
    }

    #[test]
    fn test_laplacian_sphere() {
        // f = x^2 + y^2 + z^2; Δf = 6
        let lap = laplacian(
            |xs: &[f64]| xs[0] * xs[0] + xs[1] * xs[1] + xs[2] * xs[2],
            &[1.0, 2.0, 3.0],
        )
        .expect("laplacian");
        assert!((lap - 6.0).abs() < 1e-1, "Δf = {lap}");
    }

    #[test]
    fn test_laplacian_gaussian() {
        // f = exp(-x^2-y^2); Δf = (4r^2-4) * exp(-r^2) at r^2=x^2+y^2
        // at (0,0): Δf = -4
        let lap = laplacian(
            |xs: &[f64]| (-xs[0] * xs[0] - xs[1] * xs[1]).exp(),
            &[0.0, 0.0],
        )
        .expect("laplacian gaussian");
        assert!((lap - (-4.0)).abs() < 0.5, "Δf(0,0) = {lap}");
    }

    #[test]
    fn test_laplacian_stochastic_sphere() {
        let est = laplacian_stochastic(
            |xs: &[f64]| xs.iter().map(|v| v * v).sum::<f64>(),
            &[1.0, 2.0, 3.0],
            500,
            12345,
        )
        .expect("hutchinson");
        assert!((est - 6.0).abs() < 2.0, "stochastic Δf = {est}");
    }

    #[test]
    fn test_hessian_diagonal_quadratic() {
        // f(x,y) = 3*x^2 + 5*y^2; H_diag = [6, 10]
        let d = hessian_diagonal(
            |xs: &[f64]| 3.0 * xs[0] * xs[0] + 5.0 * xs[1] * xs[1],
            &[1.0, 1.0],
        )
        .expect("hessian diagonal");
        assert!((d[0] - 6.0).abs() < 0.5, "H[0][0] = {}", d[0]);
        assert!((d[1] - 10.0).abs() < 0.5, "H[1][1] = {}", d[1]);
    }

    #[test]
    fn test_hvp_identity_hessian() {
        // f(x,y) = x^2 + y^2; H = 2I; H·v = 2v
        let v = &[1.0, 2.0];
        let hv = hvp(|xs: &[f64]| xs[0] * xs[0] + xs[1] * xs[1], &[3.0, 4.0], v)
            .expect("hvp");
        assert!((hv[0] - 2.0).abs() < 0.5, "hv[0] = {}", hv[0]);
        assert!((hv[1] - 4.0).abs() < 0.5, "hv[1] = {}", hv[1]);
    }

    #[test]
    fn test_hvp_dimension_mismatch() {
        assert!(
            hvp(|xs: &[f64]| xs[0] * xs[0], &[1.0], &[1.0, 2.0]).is_err()
        );
    }

    #[test]
    fn test_taylor_coefficients_exp() {
        // f(x) = exp(x); Taylor at 0 along v=1: c_k = 1/k!
        let cs = taylor_coefficients(|xs: &[f64]| xs[0].exp(), &[0.0], &[1.0], 3)
            .expect("taylor exp");
        assert!((cs[0] - 1.0).abs() < 1e-6, "c0 = {}", cs[0]);
        assert!((cs[1] - 1.0).abs() < 1e-2, "c1 = {}", cs[1]);
        assert!((cs[2] - 0.5).abs() < 0.1, "c2 = {}", cs[2]);
    }

    #[test]
    fn test_taylor_coefficients_too_high_order() {
        assert!(
            taylor_coefficients(|xs: &[f64]| xs[0], &[1.0], &[1.0], 9).is_err()
        );
    }

    #[test]
    fn test_nth_derivative_sin() {
        // sin'(0) = cos(0) = 1
        let d1 = nth_derivative_scalar(|x: f64| x.sin(), 0.0, 1).expect("sin'");
        assert!((d1 - 1.0).abs() < 1e-3, "sin'(0) = {d1}");
    }

    #[test]
    fn test_nth_derivative_polynomial() {
        // f(x) = x^4; f''(1) = 12
        let d2 = nth_derivative_scalar(|x: f64| x.powi(4), 1.0, 2).expect("x^4 d2");
        assert!((d2 - 12.0).abs() < 1.0, "f''(1) = {d2}");
    }

    #[test]
    fn test_nth_derivative_too_high() {
        assert!(nth_derivative_scalar(|x: f64| x, 1.0, 11).is_err());
    }

    #[test]
    fn test_iterated_gradient_order0() {
        let g0 = iterated_gradient(|xs: &[f64]| xs[0] * xs[0], &[3.0], 0).expect("order 0");
        assert!((g0[0] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_iterated_gradient_order1() {
        // ∇(x^2+y^2) at (3,4) = [6, 8]
        let g1 = iterated_gradient(
            |xs: &[f64]| xs[0] * xs[0] + xs[1] * xs[1],
            &[3.0, 4.0],
            1,
        )
        .expect("order 1");
        assert!((g1[0] - 6.0).abs() < 1e-2, "∇f[0] = {}", g1[0]);
        assert!((g1[1] - 8.0).abs() < 1e-2, "∇f[1] = {}", g1[1]);
    }

    #[test]
    fn test_jacobian_sequence_order1() {
        // f(x,y) = [x^2, x*y]; J = [[2x,0],[y,x]] at (2,3) = [[4,0],[3,2]]
        let seq = jacobian_sequence(
            |xs: &[f64]| vec![xs[0] * xs[0], xs[0] * xs[1]],
            &[2.0, 3.0],
            1,
        )
        .expect("jacobian sequence");
        let j = &seq[1];
        assert!((j[0] - 4.0).abs() < 1e-2, "J[0][0] = {}", j[0]);
        assert!((j[1] - 0.0).abs() < 1e-2, "J[0][1] = {}", j[1]);
        assert!((j[2] - 3.0).abs() < 1e-2, "J[1][0] = {}", j[2]);
        assert!((j[3] - 2.0).abs() < 1e-2, "J[1][1] = {}", j[3]);
    }
}
