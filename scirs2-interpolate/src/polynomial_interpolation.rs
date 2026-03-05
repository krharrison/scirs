//! Classical polynomial interpolation methods
//!
//! This module provides a rich set of univariate and bivariate polynomial
//! interpolation algorithms:
//!
//! - **Newton divided differences**: the standard O(n²) divided-difference
//!   table from which Newton's interpolating polynomial can be evaluated in
//!   O(n) via nested multiplication (Horner's rule).
//! - **Lagrange form**: direct O(n²) evaluation of the Lagrange interpolating
//!   polynomial, useful when the divided-difference table is not pre-computed.
//! - **Chebyshev nodes**: optimal placement of interpolation nodes on `[a,b]`
//!   to minimise Runge's phenomenon and nearly minimise the maximum error of
//!   the interpolating polynomial.
//! - **Bivariate polynomial fit**: least-squares fitting of a 2-D polynomial
//!   `p(x,y) = Σ_{i≤deg_x, j≤deg_y} c_{ij} x^i y^j` to scattered data via
//!   a Vandermonde-style system.
//! - **Vandermonde matrix**: explicit construction of the 1-D or 2-D
//!   Vandermonde system matrix.
//!
//! ## Numerical remarks
//!
//! - All algorithms are implemented to avoid panics and `unwrap()`; errors are
//!   propagated via `InterpolateResult`.
//! - For large `n`, Newton divided differences are numerically preferable to
//!   the raw Lagrange form because their Horner-form evaluation is more stable.
//! - For near-Runge problems prefer Chebyshev nodes (`chebyshev_nodes`).
//!
//! ## References
//!
//! - Atkinson, K.E. (1989). *An Introduction to Numerical Analysis*, 2nd ed.
//!   Wiley. Chapters 3-4.
//! - Burden, R.L. & Faires, J.D. (2010). *Numerical Analysis*, 9th ed.
//!   Brooks/Cole.

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Newton divided differences
// ---------------------------------------------------------------------------

/// Compute the Newton divided-difference table for nodes `x` and values `y`.
///
/// Returns the **leading coefficients** of the Newton form, i.e., `c[k]` is
/// the `k`-th divided difference `[x_0, …, x_k]f`.  The polynomial is
///
/// ```text
/// p(t) = c[0] + c[1](t-x[0]) + c[2](t-x[0])(t-x[1]) + ...
/// ```
///
/// # Arguments
///
/// * `x` – Distinct interpolation nodes, length `n`.
/// * `y` – Function values at those nodes, length `n`.
///
/// # Errors
///
/// Returns an error if `x` and `y` have different lengths, if there are
/// fewer than 1 point, or if any two nodes coincide (preventing division).
///
/// # Example
///
/// ```rust
/// use scirs2_interpolate::polynomial_interpolation::newton_divided_differences;
///
/// let x = vec![0.0_f64, 1.0, 2.0];
/// let y = vec![1.0_f64, 0.0, 1.0];   // f(t) = t² - 2t + 1 = (t-1)²
/// let dd = newton_divided_differences(&x, &y).expect("doc example: should succeed");
/// // dd[0] = f[x0] = 1.0
/// // dd[1] = f[x0,x1] = -1.0
/// // dd[2] = f[x0,x1,x2] = 1.0
/// assert!((dd[2] - 1.0).abs() < 1e-12);
/// ```
pub fn newton_divided_differences(x: &[f64], y: &[f64]) -> InterpolateResult<Vec<f64>> {
    let n = x.len();
    if n == 0 {
        return Err(InterpolateError::InsufficientData(
            "newton_divided_differences requires at least one node".to_string(),
        ));
    }
    if n != y.len() {
        return Err(InterpolateError::DimensionMismatch(format!(
            "x has {n} elements but y has {} elements",
            y.len()
        )));
    }

    // Check for duplicate nodes
    for i in 0..n {
        for j in (i + 1)..n {
            if (x[i] - x[j]).abs() < 1e-14 {
                return Err(InterpolateError::InvalidInput {
                    message: format!(
                        "Duplicate nodes x[{i}] = x[{j}] = {} detected in Newton divided differences",
                        x[i]
                    ),
                });
            }
        }
    }

    // Work table: dd[k] will hold [x_k, …, x_n-1] f after the k-th pass.
    let mut dd: Vec<f64> = y.to_vec();

    for pass in 1..n {
        // Update in-place from the end so we don't need a temporary copy.
        for k in (pass..n).rev() {
            let denom = x[k] - x[k - pass];
            // denom != 0 guaranteed by the check above (all nodes distinct)
            dd[k] = (dd[k] - dd[k - 1]) / denom;
        }
    }

    // The leading coefficients are now dd[0], dd[1], ..., dd[n-1].
    Ok(dd)
}

// ---------------------------------------------------------------------------
// Newton polynomial evaluation
// ---------------------------------------------------------------------------

/// Evaluate the Newton interpolating polynomial at `t` using Horner's rule.
///
/// # Arguments
///
/// * `x_data`   – Interpolation nodes used to build `dd_table`.
/// * `dd_table` – The divided-difference coefficients returned by
///                `newton_divided_differences`.
/// * `t`        – Query point.
///
/// # Errors
///
/// Returns an error if `x_data` is empty or if `x_data.len() != dd_table.len()`.
///
/// # Example
///
/// ```rust
/// use scirs2_interpolate::polynomial_interpolation::{
///     newton_divided_differences, newton_polynomial,
/// };
///
/// let x = vec![0.0_f64, 1.0, 2.0];
/// let y = vec![1.0_f64, 0.0, 1.0];
/// let dd = newton_divided_differences(&x, &y).expect("doc example: should succeed");
/// let val = newton_polynomial(&x, &dd, 0.5).expect("doc example: should succeed");
/// assert!((val - 0.25).abs() < 1e-10); // (0.5-1)^2 = 0.25
/// ```
pub fn newton_polynomial(x_data: &[f64], dd_table: &[f64], t: f64) -> InterpolateResult<f64> {
    let n = x_data.len();
    if n == 0 {
        return Err(InterpolateError::InsufficientData(
            "newton_polynomial requires at least one node".to_string(),
        ));
    }
    if n != dd_table.len() {
        return Err(InterpolateError::DimensionMismatch(format!(
            "x_data has {n} nodes but dd_table has {} entries",
            dd_table.len()
        )));
    }

    // Horner's rule: p = c[n-1]; p = c[n-2] + p*(t - x[n-2]); ...
    let mut p = dd_table[n - 1];
    for k in (0..n - 1).rev() {
        p = dd_table[k] + p * (t - x_data[k]);
    }
    Ok(p)
}

// ---------------------------------------------------------------------------
// Lagrange form
// ---------------------------------------------------------------------------

/// Evaluate the Lagrange interpolating polynomial at `t`.
///
/// Complexity: O(n²).  For repeated evaluations prefer pre-computing the
/// barycentric weights (see `barycentric` module); this function is provided
/// for clarity and as a reference implementation.
///
/// # Arguments
///
/// * `x_data`  – Interpolation nodes, must be distinct.
/// * `y_data`  – Function values at those nodes.
/// * `t`       – Query point.
///
/// # Errors
///
/// Returns an error if lengths differ, the input is empty, or nodes coincide.
///
/// # Example
///
/// ```rust
/// use scirs2_interpolate::polynomial_interpolation::lagrange_interpolate;
///
/// let x = vec![0.0_f64, 1.0, 2.0];
/// let y = vec![0.0_f64, 1.0, 4.0];   // p(t) = t²
/// let val = lagrange_interpolate(&x, &y, 1.5).expect("doc example: should succeed");
/// assert!((val - 2.25).abs() < 1e-10);
/// ```
pub fn lagrange_interpolate(x_data: &[f64], y_data: &[f64], t: f64) -> InterpolateResult<f64> {
    let n = x_data.len();
    if n == 0 {
        return Err(InterpolateError::InsufficientData(
            "lagrange_interpolate requires at least one node".to_string(),
        ));
    }
    if n != y_data.len() {
        return Err(InterpolateError::DimensionMismatch(format!(
            "x_data has {n} elements but y_data has {} elements",
            y_data.len()
        )));
    }

    // Check for duplicate nodes
    for i in 0..n {
        for j in (i + 1)..n {
            if (x_data[i] - x_data[j]).abs() < 1e-14 {
                return Err(InterpolateError::InvalidInput {
                    message: format!(
                        "Duplicate nodes x[{i}] = x[{j}] = {} in Lagrange interpolation",
                        x_data[i]
                    ),
                });
            }
        }
    }

    let mut result = 0.0_f64;
    for i in 0..n {
        // Compute the i-th Lagrange basis polynomial L_i(t)
        let mut li = 1.0_f64;
        for j in 0..n {
            if j != i {
                li *= (t - x_data[j]) / (x_data[i] - x_data[j]);
            }
        }
        result += y_data[i] * li;
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Chebyshev nodes
// ---------------------------------------------------------------------------

/// Compute `n` Chebyshev nodes of the first kind on the interval `[a, b]`.
///
/// The nodes are
///
/// ```text
/// x_k = (a + b) / 2  +  (b - a) / 2 · cos((2k + 1)π / (2n)),   k = 0, …, n-1
/// ```
///
/// These nodes nearly minimise the Lebesgue constant and dramatically reduce
/// the Runge phenomenon compared to uniformly spaced nodes.
///
/// # Arguments
///
/// * `a`, `b` – Interval endpoints, must satisfy `a < b`.
/// * `n`      – Number of nodes, must be at least 1.
///
/// # Errors
///
/// Returns an error if `n == 0` or if `a >= b`.
///
/// # Example
///
/// ```rust
/// use scirs2_interpolate::polynomial_interpolation::chebyshev_nodes;
///
/// let nodes = chebyshev_nodes(-1.0, 1.0, 5).expect("doc example: should succeed");
/// assert_eq!(nodes.len(), 5);
/// // All nodes should lie in (-1, 1)
/// for &x in &nodes {
///     assert!(x > -1.0 && x < 1.0);
/// }
/// ```
pub fn chebyshev_nodes(a: f64, b: f64, n: usize) -> InterpolateResult<Vec<f64>> {
    if n == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "chebyshev_nodes requires n >= 1".to_string(),
        });
    }
    if a >= b {
        return Err(InterpolateError::InvalidInput {
            message: format!("chebyshev_nodes requires a < b, got a={a}, b={b}"),
        });
    }

    let mid = (a + b) * 0.5;
    let half = (b - a) * 0.5;
    let nf = n as f64;

    let nodes: Vec<f64> = (0..n)
        .map(|k| mid + half * ((2.0 * k as f64 + 1.0) * PI / (2.0 * nf)).cos())
        .collect();

    Ok(nodes)
}

// ---------------------------------------------------------------------------
// Vandermonde matrix (1-D)
// ---------------------------------------------------------------------------

/// Construct the 1-D Vandermonde matrix for nodes `x` up to degree `degree`.
///
/// The matrix has shape `(n, degree + 1)` where entry `[i, j] = x[i]^j`.
///
/// # Arguments
///
/// * `x`      – Node values, length `n`.
/// * `degree` – Maximum polynomial degree. Column `j` corresponds to `x^j`.
///
/// # Errors
///
/// Returns an error if `x` is empty.
///
/// # Example
///
/// ```rust
/// use scirs2_interpolate::polynomial_interpolation::vandermonde_matrix;
///
/// let x = vec![0.0_f64, 1.0, 2.0];
/// let v = vandermonde_matrix(&x, 2).expect("doc example: should succeed");
/// // v[[0,0]]=1, v[[0,1]]=0, v[[0,2]]=0
/// // v[[1,0]]=1, v[[1,1]]=1, v[[1,2]]=1
/// assert!((v[[2, 2]] - 4.0).abs() < 1e-12); // 2^2 = 4
/// ```
pub fn vandermonde_matrix(x: &[f64], degree: usize) -> InterpolateResult<Array2<f64>> {
    let n = x.len();
    if n == 0 {
        return Err(InterpolateError::InsufficientData(
            "vandermonde_matrix requires at least one node".to_string(),
        ));
    }

    let cols = degree + 1;
    let mut mat = Array2::zeros((n, cols));
    for i in 0..n {
        let mut xpow = 1.0_f64;
        for j in 0..cols {
            mat[[i, j]] = xpow;
            xpow *= x[i];
        }
    }
    Ok(mat)
}

// ---------------------------------------------------------------------------
// Bivariate polynomial fit
// ---------------------------------------------------------------------------

/// Fit a bivariate polynomial `p(x, y)` to scattered 2-D data `(X, Y) → Z`
/// using least squares.
///
/// The polynomial has the form
///
/// ```text
/// p(x, y) = Σ_{0 ≤ i ≤ degree_x, 0 ≤ j ≤ degree_y}  c_{ij}  x^i  y^j
/// ```
///
/// The number of terms is `(degree_x + 1) * (degree_y + 1)`.  At least this
/// many data points must be provided.
///
/// Returns `(coefficients, residual_rms)` where:
/// - `coefficients` is a matrix of shape `(degree_x + 1, degree_y + 1)` with
///   `c[i, j]` being the coefficient of `x^i y^j`.
/// - `residual_rms` is the root-mean-square of the least-squares residuals.
///
/// # Arguments
///
/// * `x_data`   – x-coordinates of the data points, length `m`.
/// * `y_data`   – y-coordinates of the data points, length `m`.
/// * `z_data`   – Function values `z = f(x, y)`, length `m`.
/// * `degree_x` – Maximum degree in `x`.
/// * `degree_y` – Maximum degree in `y`.
///
/// # Errors
///
/// Returns an error if `x_data`, `y_data`, `z_data` have different lengths,
/// if `m < (degree_x+1)*(degree_y+1)`, or if the Vandermonde system is
/// numerically singular.
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray::Array1;
/// use scirs2_interpolate::polynomial_interpolation::bivariate_polynomial;
///
/// // Data from f(x,y) = x + y
/// let x = vec![0.0_f64, 1.0, 0.0, 1.0];
/// let y = vec![0.0_f64, 0.0, 1.0, 1.0];
/// let z = vec![0.0_f64, 1.0, 1.0, 2.0];
///
/// let (coeffs, rms) = bivariate_polynomial(&x, &y, &z, 1, 1).expect("doc example: should succeed");
/// assert!(rms < 1e-10);
/// // c[1,0] ≈ 1.0 (coefficient of x)
/// // c[0,1] ≈ 1.0 (coefficient of y)
/// assert!((coeffs[[1, 0]] - 1.0).abs() < 1e-8);
/// assert!((coeffs[[0, 1]] - 1.0).abs() < 1e-8);
/// ```
pub fn bivariate_polynomial(
    x_data: &[f64],
    y_data: &[f64],
    z_data: &[f64],
    degree_x: usize,
    degree_y: usize,
) -> InterpolateResult<(Array2<f64>, f64)> {
    let m = x_data.len();
    if m == 0 {
        return Err(InterpolateError::InsufficientData(
            "bivariate_polynomial requires at least one data point".to_string(),
        ));
    }
    if y_data.len() != m || z_data.len() != m {
        return Err(InterpolateError::DimensionMismatch(format!(
            "x_data, y_data, z_data must all have the same length (x: {}, y: {}, z: {})",
            m,
            y_data.len(),
            z_data.len()
        )));
    }

    let n_terms = (degree_x + 1) * (degree_y + 1);
    if m < n_terms {
        return Err(InterpolateError::InsufficientData(format!(
            "bivariate_polynomial needs at least {} data points for degree ({degree_x},{degree_y}), got {m}",
            n_terms
        )));
    }

    // Build the Vandermonde matrix A of shape (m, n_terms).
    // Column ordering: (i,j) with i in 0..=degree_x, j in 0..=degree_y,
    // stored in row-major order: col = i * (degree_y + 1) + j
    let n_cols = n_terms;
    let mut a = vec![0.0_f64; m * n_cols];

    for row in 0..m {
        let xv = x_data[row];
        let yv = y_data[row];
        let mut xpow = 1.0_f64;
        for i in 0..=degree_x {
            let mut ypow = 1.0_f64;
            for j in 0..=degree_y {
                let col = i * (degree_y + 1) + j;
                a[row * n_cols + col] = xpow * ypow;
                ypow *= yv;
            }
            xpow *= xv;
        }
    }

    // Solve the least-squares problem A c = z via the normal equations
    // using QR factorisation (implemented below without external BLAS).
    let c = least_squares_qr(&a, m, n_cols, z_data)?;

    // Compute residual RMS
    let mut ss = 0.0_f64;
    for row in 0..m {
        let mut pred = 0.0_f64;
        for col in 0..n_cols {
            pred += a[row * n_cols + col] * c[col];
        }
        let r = z_data[row] - pred;
        ss += r * r;
    }
    let rms = (ss / m as f64).sqrt();

    // Reshape c into (degree_x+1) × (degree_y+1) matrix
    let mut coeffs = Array2::zeros((degree_x + 1, degree_y + 1));
    for i in 0..=degree_x {
        for j in 0..=degree_y {
            coeffs[[i, j]] = c[i * (degree_y + 1) + j];
        }
    }

    Ok((coeffs, rms))
}

/// Evaluate the bivariate polynomial given its coefficient matrix at `(x, y)`.
///
/// `coeffs[[i, j]]` is the coefficient of `x^i y^j`.
///
/// # Arguments
///
/// * `coeffs` – Coefficient matrix of shape `(degree_x+1, degree_y+1)`.
/// * `x`, `y` – Query point.
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_interpolate::polynomial_interpolation::eval_bivariate_polynomial;
///
/// // p(x,y) = 1 + 2x + 3y
/// let mut c = Array2::zeros((2, 2));
/// c[[0, 0]] = 1.0;
/// c[[1, 0]] = 2.0;
/// c[[0, 1]] = 3.0;
/// let val = eval_bivariate_polynomial(&c, 2.0, 3.0);
/// assert!((val - 14.0).abs() < 1e-12); // 1 + 4 + 9 = 14
/// ```
pub fn eval_bivariate_polynomial(coeffs: &Array2<f64>, x: f64, y: f64) -> f64 {
    let dx = coeffs.nrows();
    let dy = coeffs.ncols();
    let mut result = 0.0_f64;
    let mut xpow = 1.0_f64;
    for i in 0..dx {
        let mut ypow = 1.0_f64;
        for j in 0..dy {
            result += coeffs[[i, j]] * xpow * ypow;
            ypow *= y;
        }
        xpow *= x;
    }
    result
}

// ---------------------------------------------------------------------------
// Internal: column-pivoting QR least-squares solver
// ---------------------------------------------------------------------------

/// Solve the overdetermined / square system `A x = b` in the least-squares
/// sense using Householder QR factorisation.
///
/// `a` is stored row-major with shape `(nrows, ncols)`.
/// Returns the coefficient vector of length `ncols`.
fn least_squares_qr(a_flat: &[f64], nrows: usize, ncols: usize, b: &[f64]) -> InterpolateResult<Vec<f64>> {
    assert_eq!(a_flat.len(), nrows * ncols);
    assert_eq!(b.len(), nrows);

    // We work with copies so we can in-place factorize.
    let mut a: Vec<Vec<f64>> = (0..nrows)
        .map(|i| a_flat[i * ncols..(i + 1) * ncols].to_vec())
        .collect();
    let mut rhs: Vec<f64> = b.to_vec();

    // Householder QR
    let min_dim = nrows.min(ncols);
    for k in 0..min_dim {
        // Extract column k starting from row k
        let col_norm_sq: f64 = (k..nrows).map(|i| a[i][k].powi(2)).sum();
        if col_norm_sq < 1e-28 {
            continue; // numerically zero column
        }
        let col_norm = col_norm_sq.sqrt();
        let sigma = if a[k][k] >= 0.0 { col_norm } else { -col_norm };

        // Householder vector
        let mut v: Vec<f64> = (k..nrows).map(|i| a[i][k]).collect();
        v[0] += sigma;
        let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
        if v_norm_sq < 1e-28 {
            continue;
        }
        let beta = 2.0 / v_norm_sq;

        // Apply H = I - beta * v * v^T to columns k..ncols of A
        for j in k..ncols {
            let dot: f64 = (k..nrows).map(|i| v[i - k] * a[i][j]).sum();
            let c = beta * dot;
            for i in k..nrows {
                a[i][j] -= c * v[i - k];
            }
        }

        // Apply H to rhs
        let dot_rhs: f64 = (k..nrows).map(|i| v[i - k] * rhs[i]).sum();
        let c = beta * dot_rhs;
        for i in k..nrows {
            rhs[i] -= c * v[i - k];
        }
    }

    // Now solve the upper triangular system R x = Q^T b
    // (only the first ncols equations, since R is ncols × ncols)
    let mut x = vec![0.0_f64; ncols];
    for k in (0..ncols).rev() {
        let mut s = rhs[k];
        for j in (k + 1)..ncols {
            s -= a[k][j] * x[j];
        }
        let pivot = a[k][k];
        if pivot.abs() < 1e-14 {
            return Err(InterpolateError::LinalgError(format!(
                "bivariate_polynomial: Vandermonde system is singular at pivot {k} (|pivot|={:.2e}). \
                 Reduce polynomial degree or add more data points.",
                pivot.abs()
            )));
        }
        x[k] = s / pivot;
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Convenience constructors / wrappers
// ---------------------------------------------------------------------------

/// Compute Newton divided differences and return both the coefficient table
/// and a closure that evaluates the polynomial at any query point.
///
/// This is a convenience wrapper that combines `newton_divided_differences`
/// and `newton_polynomial`.
///
/// # Errors
///
/// Propagates errors from `newton_divided_differences`.
pub fn make_newton_polynomial(
    x: Vec<f64>,
    y: &[f64],
) -> InterpolateResult<impl Fn(f64) -> f64> {
    let dd = newton_divided_differences(&x, y)?;
    Ok(move |t: f64| newton_polynomial(&x, &dd, t).unwrap_or(f64::NAN))
}

/// Compute Chebyshev nodes on `[a, b]` and use them together with supplied
/// function values to build a Newton interpolating polynomial.
///
/// # Arguments
///
/// * `a`, `b` – Interval.
/// * `n`      – Degree + 1 (number of nodes).
/// * `f`      – Function to sample at the Chebyshev nodes.
///
/// # Errors
///
/// Propagates errors from `chebyshev_nodes` and `newton_divided_differences`.
pub fn chebyshev_newton_polynomial(
    a: f64,
    b: f64,
    n: usize,
    f: &dyn Fn(f64) -> f64,
) -> InterpolateResult<(Vec<f64>, Vec<f64>)> {
    let nodes = chebyshev_nodes(a, b, n)?;
    let values: Vec<f64> = nodes.iter().map(|&xi| f(xi)).collect();
    let dd = newton_divided_differences(&nodes, &values)?;
    Ok((nodes, dd))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Newton divided differences ---

    #[test]
    fn test_newton_dd_quadratic() {
        // p(t) = (t-1)² = t² - 2t + 1
        let x = vec![0.0_f64, 1.0, 2.0];
        let y = vec![1.0_f64, 0.0, 1.0];
        let dd = newton_divided_differences(&x, &y).expect("test: should succeed");
        // dd[0] = 1, dd[1] = -1, dd[2] = 1
        assert!((dd[0] - 1.0).abs() < 1e-12, "dd[0]={}", dd[0]);
        assert!((dd[1] - (-1.0)).abs() < 1e-12, "dd[1]={}", dd[1]);
        assert!((dd[2] - 1.0).abs() < 1e-12, "dd[2]={}", dd[2]);
    }

    #[test]
    fn test_newton_dd_constant() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![5.0, 5.0, 5.0, 5.0]; // constant function
        let dd = newton_divided_differences(&x, &y).expect("test: should succeed");
        assert!((dd[0] - 5.0).abs() < 1e-12);
        for k in 1..dd.len() {
            assert!(dd[k].abs() < 1e-10, "Higher dd[{k}]={} should be 0", dd[k]);
        }
    }

    #[test]
    fn test_newton_dd_duplicate_nodes_error() {
        let x = vec![0.0, 1.0, 1.0];
        let y = vec![0.0, 1.0, 2.0];
        assert!(newton_divided_differences(&x, &y).is_err());
    }

    #[test]
    fn test_newton_dd_length_mismatch() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0, 2.0];
        assert!(newton_divided_differences(&x, &y).is_err());
    }

    // ---- Newton polynomial evaluation ---

    #[test]
    fn test_newton_polynomial_quadratic() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 0.0, 1.0]; // (t-1)^2
        let dd = newton_divided_differences(&x, &y).expect("test: should succeed");

        for &(t, expected) in &[(0.0, 1.0), (0.5, 0.25), (1.0, 0.0), (1.5, 0.25), (2.0, 1.0)] {
            let val = newton_polynomial(&x, &dd, t).expect("test: should succeed");
            assert!(
                (val - expected).abs() < 1e-10,
                "newton_polynomial({t}) = {val}, expected {expected}"
            );
        }
    }

    // ---- Lagrange ---

    #[test]
    fn test_lagrange_linear() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 2.0]; // f(t) = 2t
        for &t in &[0.0, 0.25, 0.5, 1.0] {
            let val = lagrange_interpolate(&x, &y, t).expect("test: should succeed");
            assert!((val - 2.0 * t).abs() < 1e-12, "t={t}: val={val}");
        }
    }

    #[test]
    fn test_lagrange_quadratic() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0]; // f(t) = t^2
        let val = lagrange_interpolate(&x, &y, 1.5).expect("test: should succeed");
        assert!((val - 2.25).abs() < 1e-10, "val={val}");
    }

    #[test]
    fn test_lagrange_exact_at_nodes() {
        let x = vec![-1.0, 0.0, 1.0, 2.0];
        let y = vec![1.0, 0.0, 1.0, 4.0];
        for i in 0..x.len() {
            let val = lagrange_interpolate(&x, &y, x[i]).expect("test: should succeed");
            assert!((val - y[i]).abs() < 1e-10, "Node {i}: val={val}, expected={}", y[i]);
        }
    }

    // ---- Chebyshev nodes ---

    #[test]
    fn test_chebyshev_nodes_count() {
        for n in 1..=10 {
            let nodes = chebyshev_nodes(-1.0, 1.0, n).expect("test: should succeed");
            assert_eq!(nodes.len(), n);
        }
    }

    #[test]
    fn test_chebyshev_nodes_bounds() {
        let nodes = chebyshev_nodes(-2.0, 3.0, 10).expect("test: should succeed");
        for &x in &nodes {
            assert!(x > -2.0 && x < 3.0, "Node {x} out of (-2, 3)");
        }
    }

    #[test]
    fn test_chebyshev_nodes_symmetry() {
        // Chebyshev nodes on [-1,1] are symmetric around 0
        let nodes = chebyshev_nodes(-1.0, 1.0, 5).expect("test: should succeed");
        let mut sorted = nodes.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("test: should succeed"));
        for k in 0..sorted.len() / 2 {
            assert!(
                (sorted[k] + sorted[sorted.len() - 1 - k]).abs() < 1e-12,
                "Asymmetry at k={k}"
            );
        }
    }

    #[test]
    fn test_chebyshev_bad_interval() {
        assert!(chebyshev_nodes(1.0, 0.0, 5).is_err()); // a >= b
        assert!(chebyshev_nodes(0.0, 0.0, 5).is_err()); // a == b
        assert!(chebyshev_nodes(0.0, 1.0, 0).is_err()); // n == 0
    }

    // ---- Vandermonde matrix ---

    #[test]
    fn test_vandermonde_basic() {
        let x = vec![0.0_f64, 1.0, 2.0];
        let v = vandermonde_matrix(&x, 2).expect("test: should succeed");
        assert_eq!(v.shape(), &[3, 3]);
        assert!((v[[0, 0]] - 1.0).abs() < 1e-12); // 0^0
        assert!((v[[2, 2]] - 4.0).abs() < 1e-12); // 2^2
    }

    #[test]
    fn test_vandermonde_degree_0() {
        let x = vec![3.0_f64, 7.0];
        let v = vandermonde_matrix(&x, 0).expect("test: should succeed");
        assert_eq!(v.shape(), &[2, 1]);
        assert!((v[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((v[[1, 0]] - 1.0).abs() < 1e-12);
    }

    // ---- Bivariate polynomial ---

    #[test]
    fn test_bivariate_linear() {
        // f(x,y) = x + y  →  c[1,0] = 1, c[0,1] = 1
        let x = vec![0.0, 1.0, 0.0, 1.0, 0.5];
        let y = vec![0.0, 0.0, 1.0, 1.0, 0.5];
        let z: Vec<f64> = x.iter().zip(y.iter()).map(|(xi, yi)| xi + yi).collect();
        let (coeffs, rms) = bivariate_polynomial(&x, &y, &z, 1, 1).expect("test: should succeed");
        assert!(rms < 1e-8, "RMS={rms}");
        assert!((coeffs[[1, 0]] - 1.0).abs() < 1e-6, "c[1,0]={}", coeffs[[1, 0]]);
        assert!((coeffs[[0, 1]] - 1.0).abs() < 1e-6, "c[0,1]={}", coeffs[[0, 1]]);
    }

    #[test]
    fn test_bivariate_eval() {
        let mut c = Array2::zeros((2, 2));
        c[[0, 0]] = 1.0;
        c[[1, 0]] = 2.0;
        c[[0, 1]] = 3.0;
        // p(2,3) = 1 + 2*2 + 3*3 = 14
        let val = eval_bivariate_polynomial(&c, 2.0, 3.0);
        assert!((val - 14.0).abs() < 1e-12, "val={val}");
    }

    #[test]
    fn test_bivariate_too_few_points() {
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0];
        let z = vec![0.0, 1.0];
        // degree 1,1 needs 4 points
        assert!(bivariate_polynomial(&x, &y, &z, 1, 1).is_err());
    }

    #[test]
    fn test_chebyshev_newton_polynomial_approx() {
        // Approximate sin on [0, pi] with 8 Chebyshev nodes
        let (nodes, dd) = chebyshev_newton_polynomial(0.0, std::f64::consts::PI, 8, &f64::sin)
            .expect("test: should succeed");
        // Check a few interior points
        for &t in &[0.3, 0.8, 1.5, 2.0, 2.7] {
            let val = newton_polynomial(&nodes, &dd, t).expect("test: should succeed");
            let expected = t.sin();
            assert!(
                (val - expected).abs() < 1e-5,
                "sin({t}) ≈ {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_make_newton_polynomial() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0]; // n^2
        let poly = make_newton_polynomial(x, &y).expect("test: should succeed");
        assert!((poly(2.5) - 6.25).abs() < 1e-8, "poly(2.5)={}", poly(2.5));
    }
}
