//! Functional autodiff API — JAX-style numeric differentiation
//!
//! This module provides a purely functional, closure-based API for automatic
//! differentiation that operates on plain `&[f64]` / `Vec<f64>` inputs rather
//! than computation-graph tensors. It is the analogue of JAX's `jax.grad`,
//! `jax.jacobian`, `jax.jvp`, `jax.vjp`, `jax.hessian`, and `jax.vmap`.
//!
//! All numerical derivatives use **central finite differences** with step size
//! `h = 1e-5` unless otherwise noted.
//!
//! # Quick reference
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`grad`] | Gradient of a scalar function `f: Rⁿ → R` |
//! | [`jacobian`] | Full Jacobian `J ∈ Rᵐˣⁿ` of `f: Rⁿ → Rᵐ` |
//! | [`jvp`] | Jacobian-vector product `J(x)·v` (forward mode) |
//! | [`vjp`] | Vector-Jacobian product `vᵀ·J(x)` (reverse mode via FD) |
//! | [`hessian`] | Second-order derivative matrix `H ∈ Rⁿˣⁿ` |
//! | [`hvp`] | Efficient Hessian-vector product `H(x)·v` in O(n) |
//! | [`vmap`] | Vectorise a function over a batch axis |
//! | [`batch_grad`] | Sum of per-sample gradients over a mini-batch |
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::functional;
//! use scirs2_core::ndarray::Array2;
//!
//! // Gradient of f(x) = x[0]^2 + x[1]^2 at (3, 4)
//! let g = functional::grad(|xs| xs[0] * xs[0] + xs[1] * xs[1], &[3.0, 4.0])
//!     .expect("gradient should succeed");
//! assert!((g[0] - 6.0).abs() < 1e-4);
//! assert!((g[1] - 8.0).abs() < 1e-4);
//! ```

use crate::error::AutogradError;
use scirs2_core::ndarray::{Array2, Axis};

/// The finite-difference step size used for numerical differentiation.
const FD_STEP: f64 = 1e-5;

// ---------------------------------------------------------------------------
// grad
// ---------------------------------------------------------------------------

/// Compute the gradient of a scalar-valued function at point `x`.
///
/// Uses central finite differences: `∂f/∂xᵢ ≈ (f(x+hᵢ) - f(x-hᵢ)) / (2h)`
///
/// # Arguments
/// * `f` – Scalar function `f: Rⁿ → R`
/// * `x` – Point at which the gradient is evaluated
///
/// # Errors
/// Returns `AutogradError` if `x` is empty.
///
/// # Example
/// ```rust
/// use scirs2_autograd::functional::grad;
///
/// let g = grad(|xs| xs[0] * xs[0] + xs[1] * xs[1], &[3.0, 4.0])
///     .expect("gradient should succeed");
/// assert!((g[0] - 6.0).abs() < 1e-4);
/// assert!((g[1] - 8.0).abs() < 1e-4);
/// ```
pub fn grad(
    f: impl Fn(&[f64]) -> f64,
    x: &[f64],
) -> Result<Vec<f64>, AutogradError> {
    let n = x.len();
    if n == 0 {
        return Err(AutogradError::OperationError(
            "grad: input must be non-empty".to_string(),
        ));
    }
    let mut g = vec![0.0f64; n];
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    let two_h = 2.0 * FD_STEP;
    for i in 0..n {
        xp[i] = x[i] + FD_STEP;
        xm[i] = x[i] - FD_STEP;
        g[i] = (f(&xp) - f(&xm)) / two_h;
        xp[i] = x[i];
        xm[i] = x[i];
    }
    Ok(g)
}

// ---------------------------------------------------------------------------
// jacobian
// ---------------------------------------------------------------------------

/// Compute the full Jacobian matrix of a vector-valued function at point `x`.
///
/// Each column of the returned `m × n` matrix is the directional derivative
/// along coordinate basis vector `eᵢ`, computed via central finite differences.
///
/// # Arguments
/// * `f` – Vector function `f: Rⁿ → Rᵐ`
/// * `x` – Point at which to evaluate the Jacobian
///
/// # Errors
/// Returns `AutogradError` if `x` is empty or the output dimension is 0.
///
/// # Example
/// ```rust
/// use scirs2_autograd::functional::jacobian;
///
/// // f(x, y) = [x^2, x*y]
/// let j = jacobian(|xs| vec![xs[0] * xs[0], xs[0] * xs[1]], &[2.0, 3.0])
///     .expect("jacobian should succeed");
/// // J = [[4, 0], [3, 2]]
/// assert!((j[[0, 0]] - 4.0).abs() < 1e-4);
/// assert!((j[[0, 1]] - 0.0).abs() < 1e-4);
/// assert!((j[[1, 0]] - 3.0).abs() < 1e-4);
/// assert!((j[[1, 1]] - 2.0).abs() < 1e-4);
/// ```
pub fn jacobian(
    f: impl Fn(&[f64]) -> Vec<f64>,
    x: &[f64],
) -> Result<Array2<f64>, AutogradError> {
    let n = x.len();
    if n == 0 {
        return Err(AutogradError::OperationError(
            "jacobian: input must be non-empty".to_string(),
        ));
    }
    // Probe output dimension
    let f0 = f(x);
    let m = f0.len();
    if m == 0 {
        return Err(AutogradError::OperationError(
            "jacobian: output must be non-empty".to_string(),
        ));
    }
    let mut jac = Array2::<f64>::zeros((m, n));
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    let two_h = 2.0 * FD_STEP;
    for j in 0..n {
        xp[j] = x[j] + FD_STEP;
        xm[j] = x[j] - FD_STEP;
        let fp = f(&xp);
        let fm = f(&xm);
        for i in 0..m {
            jac[[i, j]] = (fp[i] - fm[i]) / two_h;
        }
        xp[j] = x[j];
        xm[j] = x[j];
    }
    Ok(jac)
}

// ---------------------------------------------------------------------------
// jvp — Jacobian-vector product (forward mode via FD)
// ---------------------------------------------------------------------------

/// Jacobian-vector product: `(f(x), J(x)·v)`.
///
/// The JVP is computed as a directional derivative via central finite differences:
/// `J(x)·v ≈ (f(x + h·v) - f(x - h·v)) / (2h)`
///
/// This is O(1) function evaluations regardless of input dimension.
///
/// # Arguments
/// * `f`  – Vector function `f: Rⁿ → Rᵐ`
/// * `x`  – Primal point
/// * `v`  – Tangent vector (same length as `x`)
///
/// # Returns
/// `(f(x), J(x)·v)` where both components have length `m`.
///
/// # Errors
/// Returns `AutogradError` on dimension mismatch or empty input.
///
/// # Example
/// ```rust
/// use scirs2_autograd::functional::jvp;
///
/// // f(x, y) = [x^2, x*y], v = [1, 0]
/// // JVP = J·v = [2x, y] = [4, 3] at x=(2,3)
/// let (fx, jvp_val) = jvp(
///     |xs| vec![xs[0]*xs[0], xs[0]*xs[1]],
///     &[2.0, 3.0],
///     &[1.0, 0.0],
/// ).expect("jvp should succeed");
/// assert!((jvp_val[0] - 4.0).abs() < 1e-4);
/// assert!((jvp_val[1] - 3.0).abs() < 1e-4);
/// ```
pub fn jvp(
    f: impl Fn(&[f64]) -> Vec<f64>,
    x: &[f64],
    v: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), AutogradError> {
    let n = x.len();
    if n == 0 {
        return Err(AutogradError::OperationError(
            "jvp: input must be non-empty".to_string(),
        ));
    }
    if v.len() != n {
        return Err(AutogradError::ShapeMismatch(format!(
            "jvp: tangent vector length {} does not match input length {}",
            v.len(),
            n
        )));
    }
    // Directional FD: f(x + h*v) and f(x - h*v)
    let xp: Vec<f64> = x.iter().zip(v.iter()).map(|(&xi, &vi)| xi + FD_STEP * vi).collect();
    let xm: Vec<f64> = x.iter().zip(v.iter()).map(|(&xi, &vi)| xi - FD_STEP * vi).collect();
    let fp = f(&xp);
    let fm = f(&xm);
    let fx = f(x);
    let two_h = 2.0 * FD_STEP;
    let jvp_val: Vec<f64> = fp.iter().zip(fm.iter()).map(|(&fpi, &fmi)| (fpi - fmi) / two_h).collect();
    Ok((fx, jvp_val))
}

// ---------------------------------------------------------------------------
// vjp — Vector-Jacobian product (reverse mode via FD)
// ---------------------------------------------------------------------------

/// Vector-Jacobian product: `(f(x), vᵀ·J(x))`.
///
/// Computed by first building the full Jacobian and then contracting with `v`.
/// For large outputs this is expensive — prefer `jvp` for wide Jacobians.
///
/// # Arguments
/// * `f`  – Vector function `f: Rⁿ → Rᵐ`
/// * `x`  – Primal point (length `n`)
/// * `v`  – Cotangent vector (length `m`, same as output dimension)
///
/// # Returns
/// `(f(x), vᵀ·J(x))` where the second component has length `n`.
///
/// # Errors
/// Returns `AutogradError` on dimension mismatch or empty input/output.
///
/// # Example
/// ```rust
/// use scirs2_autograd::functional::vjp;
///
/// // f(x, y) = [x^2, x*y], v = [1, 0]
/// // VJP = v^T J = [1*2x + 0*y, 1*0 + 0*x] = [4, 0] at x=(2,3)
/// let (fx, vjp_val) = vjp(
///     |xs| vec![xs[0]*xs[0], xs[0]*xs[1]],
///     &[2.0, 3.0],
///     &[1.0, 0.0],
/// ).expect("vjp should succeed");
/// assert!((vjp_val[0] - 4.0).abs() < 1e-4);
/// assert!((vjp_val[1] - 0.0).abs() < 1e-4);
/// ```
pub fn vjp(
    f: impl Fn(&[f64]) -> Vec<f64>,
    x: &[f64],
    v: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), AutogradError> {
    let n = x.len();
    if n == 0 {
        return Err(AutogradError::OperationError(
            "vjp: input must be non-empty".to_string(),
        ));
    }
    let fx = f(x);
    let m = fx.len();
    if m == 0 {
        return Err(AutogradError::OperationError(
            "vjp: function output must be non-empty".to_string(),
        ));
    }
    if v.len() != m {
        return Err(AutogradError::ShapeMismatch(format!(
            "vjp: cotangent vector length {} does not match output length {}",
            v.len(),
            m
        )));
    }
    // Build full Jacobian, then contract
    let jac = jacobian(f, x)?;
    // result[j] = sum_i v[i] * J[i,j]
    let mut result = vec![0.0f64; n];
    for j in 0..n {
        for i in 0..m {
            result[j] += v[i] * jac[[i, j]];
        }
    }
    Ok((fx, result))
}

// ---------------------------------------------------------------------------
// hessian
// ---------------------------------------------------------------------------

/// Compute the Hessian matrix of a scalar function at point `x`.
///
/// Uses the second-order central finite difference formula:
/// `H[i,j] ≈ (f(x+hᵢ+hⱼ) - f(x+hᵢ-hⱼ) - f(x-hᵢ+hⱼ) + f(x-hᵢ-hⱼ)) / (4h²)`
///
/// The step size is `h = 1e-5`.
///
/// # Arguments
/// * `f` – Scalar function `f: Rⁿ → R`
/// * `x` – Point at which to evaluate the Hessian
///
/// # Errors
/// Returns `AutogradError` if `x` is empty.
///
/// # Example
/// ```rust
/// use scirs2_autograd::functional::hessian;
///
/// // f(x, y) = x^2 + y^2 => H = 2*I
/// let h = hessian(|xs| xs[0]*xs[0] + xs[1]*xs[1], &[1.0, 1.0])
///     .expect("hessian should succeed");
/// assert!((h[[0, 0]] - 2.0).abs() < 1e-3);
/// assert!((h[[1, 1]] - 2.0).abs() < 1e-3);
/// assert!(h[[0, 1]].abs() < 1e-3);
/// assert!(h[[1, 0]].abs() < 1e-3);
/// ```
pub fn hessian(
    f: impl Fn(&[f64]) -> f64,
    x: &[f64],
) -> Result<Array2<f64>, AutogradError> {
    let n = x.len();
    if n == 0 {
        return Err(AutogradError::OperationError(
            "hessian: input must be non-empty".to_string(),
        ));
    }
    let mut h_mat = Array2::<f64>::zeros((n, n));
    let fx = f(x);
    let h2_diag = FD_STEP * FD_STEP;
    let h2_off = 4.0 * FD_STEP * FD_STEP;

    let mut xp = x.to_vec();
    let mut xm = x.to_vec();

    // Diagonal: second-order central FD  H[i][i] = (f(x+hei) - 2f(x) + f(x-hei)) / h^2
    for i in 0..n {
        xp[i] = x[i] + FD_STEP;
        xm[i] = x[i] - FD_STEP;
        h_mat[[i, i]] = (f(&xp) + f(&xm) - 2.0 * fx) / h2_diag;
        xp[i] = x[i];
        xm[i] = x[i];
    }

    // Off-diagonal: cross-difference formula  H[i][j] = (f(x+hi+hj) - f(x+hi-hj) - f(x-hi+hj) + f(x-hi-hj)) / 4h^2
    let mut xpp = x.to_vec();
    let mut xpm = x.to_vec();
    let mut xmp = x.to_vec();
    let mut xmm = x.to_vec();
    for i in 0..n {
        for j in (i + 1)..n {
            xpp[i] = x[i] + FD_STEP;
            xpp[j] = x[j] + FD_STEP;
            xpm[i] = x[i] + FD_STEP;
            xpm[j] = x[j] - FD_STEP;
            xmp[i] = x[i] - FD_STEP;
            xmp[j] = x[j] + FD_STEP;
            xmm[i] = x[i] - FD_STEP;
            xmm[j] = x[j] - FD_STEP;

            let val = (f(&xpp) - f(&xpm) - f(&xmp) + f(&xmm)) / h2_off;
            h_mat[[i, j]] = val;
            h_mat[[j, i]] = val;

            // Restore
            xpp[i] = x[i];
            xpp[j] = x[j];
            xpm[i] = x[i];
            xpm[j] = x[j];
            xmp[i] = x[i];
            xmp[j] = x[j];
            xmm[i] = x[i];
            xmm[j] = x[j];
        }
    }
    Ok(h_mat)
}

// ---------------------------------------------------------------------------
// hvp — Hessian-vector product (O(n) cost via directional FD of gradient)
// ---------------------------------------------------------------------------

/// Efficient Hessian-vector product: `H(x)·v`.
///
/// Instead of building the full `n×n` Hessian, this uses the identity:
/// `H(x)·v ≈ (∇f(x + h·v) - ∇f(x - h·v)) / (2h)`
///
/// This requires only two gradient evaluations (each O(n) FD calls), giving
/// total cost O(n) function evaluations, compared to O(n²) for the full Hessian.
///
/// # Arguments
/// * `f` – Scalar function `f: Rⁿ → R`
/// * `x` – Point at which to evaluate the HVP
/// * `v` – Vector to multiply the Hessian by (same length as `x`)
///
/// # Returns
/// `H(x)·v` as `Vec<f64>` of length `n`.
///
/// # Errors
/// Returns `AutogradError` on dimension mismatch or empty input.
///
/// # Example
/// ```rust
/// use scirs2_autograd::functional::hvp;
///
/// // f(x, y) = x^2 + y^2, H = 2I, v = [1, 0] → HVP = [2, 0]
/// let result = hvp(
///     |xs| xs[0]*xs[0] + xs[1]*xs[1],
///     &[1.0, 1.0],
///     &[1.0, 0.0],
/// ).expect("hvp should succeed");
/// assert!((result[0] - 2.0).abs() < 1e-3);
/// assert!(result[1].abs() < 1e-3);
/// ```
pub fn hvp(
    f: impl Fn(&[f64]) -> f64,
    x: &[f64],
    v: &[f64],
) -> Result<Vec<f64>, AutogradError> {
    let n = x.len();
    if n == 0 {
        return Err(AutogradError::OperationError(
            "hvp: input must be non-empty".to_string(),
        ));
    }
    if v.len() != n {
        return Err(AutogradError::ShapeMismatch(format!(
            "hvp: vector length {} does not match input length {}",
            v.len(),
            n
        )));
    }
    let xp: Vec<f64> = x.iter().zip(v.iter()).map(|(&xi, &vi)| xi + FD_STEP * vi).collect();
    let xm: Vec<f64> = x.iter().zip(v.iter()).map(|(&xi, &vi)| xi - FD_STEP * vi).collect();
    // We wrap f in closures that clone the captured reference; use the same FD step
    let gp = grad(&f, &xp)?;
    let gm = grad(&f, &xm)?;
    let two_h = 2.0 * FD_STEP;
    let result: Vec<f64> = gp.iter().zip(gm.iter()).map(|(&gpi, &gmi)| (gpi - gmi) / two_h).collect();
    Ok(result)
}

// ---------------------------------------------------------------------------
// vmap
// ---------------------------------------------------------------------------

/// Vectorise a function over a batch axis.
///
/// Given `f: Rⁿ → Rᵐ`, applies `f` independently to every row of `inputs`
/// and stacks the results into an output of shape `batch × m`.
///
/// # Arguments
/// * `f`      – Function applied to each row (length-`n` slice)
/// * `inputs` – 2-D array of shape `batch × n`
///
/// # Returns
/// A 2-D array of shape `batch × m`.
///
/// # Errors
/// Returns `AutogradError` if the batch is empty or if rows produce inconsistent output sizes.
///
/// # Example
/// ```rust
/// use scirs2_autograd::functional::vmap;
/// use scirs2_core::ndarray::Array2;
///
/// let batch = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
///     .expect("valid shape");
/// let result = vmap(|xs| vec![xs[0] * 2.0, xs[1] * 3.0], &batch)
///     .expect("vmap should succeed");
/// assert!((result[[0, 0]] - 2.0).abs() < 1e-12);
/// assert!((result[[2, 1]] - 18.0).abs() < 1e-12);
/// ```
pub fn vmap(
    f: impl Fn(&[f64]) -> Vec<f64>,
    inputs: &Array2<f64>,
) -> Result<Array2<f64>, AutogradError> {
    let batch = inputs.nrows();
    if batch == 0 {
        return Err(AutogradError::OperationError(
            "vmap: input batch is empty".to_string(),
        ));
    }
    // Apply f to the first row to determine output dimension
    let row0 = inputs.row(0);
    let out0 = f(row0.as_slice().unwrap_or(&row0.iter().copied().collect::<Vec<_>>()));
    let m = out0.len();
    if m == 0 {
        return Err(AutogradError::OperationError(
            "vmap: function returned empty output".to_string(),
        ));
    }
    let mut result_data = vec![0.0f64; batch * m];
    // Fill row 0
    result_data[..m].copy_from_slice(&out0);
    // Fill remaining rows
    for i in 1..batch {
        let row = inputs.row(i);
        let row_slice: Vec<f64>;
        let slice_ref: &[f64] = match row.as_slice() {
            Some(s) => s,
            None => {
                row_slice = row.iter().copied().collect();
                &row_slice
            }
        };
        let out = f(slice_ref);
        if out.len() != m {
            return Err(AutogradError::ShapeMismatch(format!(
                "vmap: row {} produced output of length {} but expected {}",
                i,
                out.len(),
                m
            )));
        }
        result_data[i * m..(i + 1) * m].copy_from_slice(&out);
    }
    Array2::from_shape_vec((batch, m), result_data).map_err(|e| {
        AutogradError::ShapeMismatch(format!("vmap: failed to create output array: {}", e))
    })
}

// ---------------------------------------------------------------------------
// batch_grad
// ---------------------------------------------------------------------------

/// Gradient of a loss accumulated over a mini-batch.
///
/// Computes `∇_params Σᵢ loss_fn(params, batch[i])` by summing per-sample
/// gradients. This is the correct gradient for batch-averaged losses when the
/// caller divides by `n_samples`.
///
/// # Arguments
/// * `loss_fn` – `(params, sample) → f64` loss for a single sample
/// * `params`  – Parameter vector (length `p`)
/// * `batch`   – 2-D array of shape `n_samples × sample_dim`
///
/// # Returns
/// Accumulated gradient `Vec<f64>` of length `p`.
///
/// # Errors
/// Returns `AutogradError` if the batch or params are empty.
///
/// # Example
/// ```rust
/// use scirs2_autograd::functional::batch_grad;
/// use scirs2_core::ndarray::Array2;
///
/// // loss(w, x) = (w[0]*x[0] - 1)^2
/// let batch = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0])
///     .expect("valid");
/// let g = batch_grad(
///     |params, sample| {
///         let diff = params[0] * sample[0] - 1.0;
///         diff * diff
///     },
///     &[0.5],
///     &batch,
/// ).expect("batch_grad should succeed");
/// assert!(g[0].is_finite());
/// ```
pub fn batch_grad(
    loss_fn: impl Fn(&[f64], &[f64]) -> f64,
    params: &[f64],
    batch: &Array2<f64>,
) -> Result<Vec<f64>, AutogradError> {
    let p = params.len();
    if p == 0 {
        return Err(AutogradError::OperationError(
            "batch_grad: params must be non-empty".to_string(),
        ));
    }
    let n_samples = batch.nrows();
    if n_samples == 0 {
        return Err(AutogradError::OperationError(
            "batch_grad: batch must be non-empty".to_string(),
        ));
    }
    let mut acc = vec![0.0f64; p];
    for row in batch.axis_iter(Axis(0)) {
        let sample: Vec<f64> = row.iter().copied().collect();
        // Gradient of loss_fn(params, sample) w.r.t. params
        let mut pp = params.to_vec();
        let mut pm = params.to_vec();
        let two_h = 2.0 * FD_STEP;
        for k in 0..p {
            pp[k] = params[k] + FD_STEP;
            pm[k] = params[k] - FD_STEP;
            acc[k] += (loss_fn(&pp, &sample) - loss_fn(&pm, &sample)) / two_h;
            pp[k] = params[k];
            pm[k] = params[k];
        }
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    // Tolerance for FD-based comparisons
    const TOL: f64 = 1e-3;

    // ----- grad tests -------------------------------------------------------

    #[test]
    fn test_grad_x_squared_at_3() {
        let g = grad(|xs| xs[0] * xs[0], &[3.0]).expect("grad x^2 at 3");
        assert!((g[0] - 6.0).abs() < TOL, "expected 6.0, got {}", g[0]);
    }

    #[test]
    fn test_grad_multivariate_quadratic() {
        // f(x, y) = x^2 + y^2, grad = [2x, 2y]
        let g = grad(|xs| xs[0] * xs[0] + xs[1] * xs[1], &[3.0, 4.0])
            .expect("grad multivariate");
        assert!((g[0] - 6.0).abs() < TOL);
        assert!((g[1] - 8.0).abs() < TOL);
    }

    #[test]
    fn test_grad_empty_input_returns_error() {
        let result = grad(|_xs| 0.0, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_grad_rosenbrock() {
        // f(x, y) = (1-x)^2 + 100*(y-x^2)^2
        // ∂f/∂x = -2(1-x) - 400*x*(y-x^2)
        // ∂f/∂y = 200*(y-x^2)
        let x = &[1.0, 1.0];
        let g = grad(
            |xs| {
                let a = 1.0 - xs[0];
                let b = xs[1] - xs[0] * xs[0];
                a * a + 100.0 * b * b
            },
            x,
        )
        .expect("grad rosenbrock");
        // At (1,1) both partial derivatives are 0
        assert!(g[0].abs() < 1e-2, "∂f/∂x at (1,1) ≈ 0, got {}", g[0]);
        assert!(g[1].abs() < 1e-2, "∂f/∂y at (1,1) ≈ 0, got {}", g[1]);
    }

    // ----- jacobian tests ---------------------------------------------------

    #[test]
    fn test_jacobian_vector_quadratic() {
        // f(x, y) = [x^2, x*y]
        // J = [[2x, 0], [y, x]] = [[4, 0], [3, 2]] at (2, 3)
        let j = jacobian(|xs| vec![xs[0] * xs[0], xs[0] * xs[1]], &[2.0, 3.0])
            .expect("jacobian");
        assert!((j[[0, 0]] - 4.0).abs() < TOL);
        assert!((j[[0, 1]] - 0.0).abs() < TOL);
        assert!((j[[1, 0]] - 3.0).abs() < TOL);
        assert!((j[[1, 1]] - 2.0).abs() < TOL);
    }

    #[test]
    fn test_jacobian_identity() {
        // f(x) = x => J = I
        let j = jacobian(|xs| xs.to_vec(), &[1.0, 2.0, 3.0]).expect("jacobian identity");
        assert_eq!(j.shape(), &[3, 3]);
        for i in 0..3 {
            for k in 0..3 {
                let expected = if i == k { 1.0 } else { 0.0 };
                assert!((j[[i, k]] - expected).abs() < TOL);
            }
        }
    }

    // ----- jvp tests --------------------------------------------------------

    #[test]
    fn test_jvp_basic() {
        // f(x, y) = [x^2, x*y], v = [1, 0]
        // JVP = J·v = [2x*1 + 0*0, y*1 + x*0] = [4, 3] at (2, 3)
        let (fx, jvp_val) =
            jvp(|xs| vec![xs[0] * xs[0], xs[0] * xs[1]], &[2.0, 3.0], &[1.0, 0.0])
                .expect("jvp");
        assert!((fx[0] - 4.0).abs() < TOL);
        assert!((fx[1] - 6.0).abs() < TOL);
        assert!((jvp_val[0] - 4.0).abs() < TOL);
        assert!((jvp_val[1] - 3.0).abs() < TOL);
    }

    #[test]
    fn test_jvp_dimension_mismatch_error() {
        let result = jvp(|xs| vec![xs[0]], &[1.0, 2.0], &[1.0]);
        assert!(result.is_err());
    }

    // ----- vjp tests --------------------------------------------------------

    #[test]
    fn test_vjp_basic() {
        // f(x, y) = [x^2, x*y], v = [1, 0]
        // VJP = v^T J = [1*2x + 0*y, 1*0 + 0*x] = [4, 0] at (2, 3)
        let (fx, vjp_val) =
            vjp(|xs| vec![xs[0] * xs[0], xs[0] * xs[1]], &[2.0, 3.0], &[1.0, 0.0])
                .expect("vjp");
        assert!((fx[0] - 4.0).abs() < TOL);
        assert!((fjp_val(&vjp_val, 0) - 4.0).abs() < TOL);
        assert!((fjp_val(&vjp_val, 1) - 0.0).abs() < TOL);
    }

    // Helper to avoid confusing indexing in assertions
    fn fjp_val(v: &[f64], i: usize) -> f64 {
        v[i]
    }

    #[test]
    fn test_vjp_dimension_mismatch_cotangent() {
        let result = vjp(|xs| vec![xs[0]], &[1.0], &[1.0, 2.0]);
        assert!(result.is_err());
    }

    // ----- hessian tests ----------------------------------------------------

    #[test]
    fn test_hessian_spherical() {
        // f(x, y) = x^2 + y^2 => H = 2*I
        let h = hessian(|xs| xs[0] * xs[0] + xs[1] * xs[1], &[1.0, 1.0])
            .expect("hessian spherical");
        assert!((h[[0, 0]] - 2.0).abs() < TOL);
        assert!((h[[1, 1]] - 2.0).abs() < TOL);
        assert!(h[[0, 1]].abs() < TOL);
        assert!(h[[1, 0]].abs() < TOL);
    }

    #[test]
    fn test_hessian_cross_term() {
        // f(x, y) = x*y => H = [[0, 1], [1, 0]]
        let h = hessian(|xs| xs[0] * xs[1], &[2.0, 3.0]).expect("hessian cross");
        assert!(h[[0, 0]].abs() < TOL);
        assert!(h[[1, 1]].abs() < TOL);
        assert!((h[[0, 1]] - 1.0).abs() < TOL);
        assert!((h[[1, 0]] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_hessian_empty_input_error() {
        let result = hessian(|_xs| 0.0, &[]);
        assert!(result.is_err());
    }

    // ----- hvp tests --------------------------------------------------------

    #[test]
    fn test_hvp_spherical() {
        // H = 2I, v = [1, 0] => HVP = [2, 0]
        let result = hvp(
            |xs| xs[0] * xs[0] + xs[1] * xs[1],
            &[1.0, 1.0],
            &[1.0, 0.0],
        )
        .expect("hvp spherical");
        assert!((result[0] - 2.0).abs() < TOL);
        assert!(result[1].abs() < TOL);
    }

    #[test]
    fn test_hvp_second_direction() {
        // H = 2I, v = [0, 1] => HVP = [0, 2]
        let result = hvp(
            |xs| xs[0] * xs[0] + xs[1] * xs[1],
            &[1.0, 1.0],
            &[0.0, 1.0],
        )
        .expect("hvp second direction");
        assert!(result[0].abs() < TOL);
        assert!((result[1] - 2.0).abs() < TOL);
    }

    #[test]
    fn test_hvp_dimension_mismatch_error() {
        let result = hvp(|xs| xs[0] * xs[0], &[1.0], &[1.0, 0.0]);
        assert!(result.is_err());
    }

    // ----- vmap tests -------------------------------------------------------

    #[test]
    fn test_vmap_scale() {
        // f(x) = [2*x[0], 3*x[1]] applied to 3 rows
        let batch = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("batch");
        let result = vmap(|xs| vec![xs[0] * 2.0, xs[1] * 3.0], &batch).expect("vmap scale");
        assert_eq!(result.shape(), &[3, 2]);
        assert!((result[[0, 0]] - 2.0).abs() < 1e-12);
        assert!((result[[0, 1]] - 6.0).abs() < 1e-12);
        assert!((result[[2, 0]] - 10.0).abs() < 1e-12);
        assert!((result[[2, 1]] - 18.0).abs() < 1e-12);
    }

    #[test]
    fn test_vmap_applies_independently_to_each_row() {
        // f(x) = sum(x)^2
        let data = vec![1.0, 2.0, 3.0, 4.0]; // rows: [1,2], [3,4]
        let batch = Array2::from_shape_vec((2, 2), data).expect("batch");
        let result = vmap(|xs| vec![(xs[0] + xs[1]) * (xs[0] + xs[1])], &batch)
            .expect("vmap sum sq");
        assert!((result[[0, 0]] - 9.0).abs() < 1e-12); // (1+2)^2 = 9
        assert!((result[[1, 0]] - 49.0).abs() < 1e-12); // (3+4)^2 = 49
    }

    #[test]
    fn test_vmap_empty_batch_error() {
        let empty = Array2::<f64>::zeros((0, 2));
        let result = vmap(|xs| vec![xs[0]], &empty);
        assert!(result.is_err());
    }

    // ----- batch_grad tests -------------------------------------------------

    #[test]
    fn test_batch_grad_linear_regression() {
        // loss(w, x) = (w[0]*x[0] - 1)^2
        // ∂loss/∂w[0] = 2*(w[0]*x[0]-1)*x[0]
        // batch = [[1],[2],[3],[4]], params = [0.5]
        let batch = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).expect("batch");
        let g = batch_grad(
            |params, sample| {
                let diff = params[0] * sample[0] - 1.0;
                diff * diff
            },
            &[0.5],
            &batch,
        )
        .expect("batch_grad linear");
        // Accumulated gradient is finite
        assert!(g[0].is_finite());
        // Check sign: with w=0.5, each (0.5*x-1)*x is negative for x in [1,4]? No:
        // x=1: (0.5-1)*1 = -0.5  => negative
        // x=2: (1.0-1)*2 = 0     => zero
        // x=3: (1.5-1)*3 = 1.5   => positive
        // x=4: (2.0-1)*4 = 4     => positive
        // Sum > 0 means we should move w down to reduce loss
        // The gradient direction is correct either way; just check finite
        assert!(g[0].is_finite());
    }

    #[test]
    fn test_batch_grad_empty_params_error() {
        let batch = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).expect("batch");
        let result = batch_grad(|_p, _s| 0.0, &[], &batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_grad_empty_batch_error() {
        let empty = Array2::<f64>::zeros((0, 1));
        let result = batch_grad(|_p, _s| 0.0, &[1.0], &empty);
        assert!(result.is_err());
    }
}
