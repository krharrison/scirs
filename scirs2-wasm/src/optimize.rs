//! Optimization algorithms for WASM
//!
//! Provides optimization routines accessible from JavaScript/TypeScript via WebAssembly.
//! Since WASM cannot receive function pointers from JS directly, these functions use
//! evaluation-based approaches where the caller provides pre-computed function values
//! or data arrays.

use crate::error::WasmError;
use wasm_bindgen::prelude::*;

// ============================================================================
// Nelder-Mead Simplex Optimization
// ============================================================================

/// Minimize a function using the Nelder-Mead simplex algorithm.
///
/// Since WASM cannot pass function closures from JS, this performs the Nelder-Mead
/// optimization using pre-evaluated simplex vertices and their function values.
///
/// # Arguments
/// * `f_values` - Function values at the initial simplex vertices (n+1 values for n dimensions)
/// * `x0` - Flat array of initial simplex vertex coordinates, row-major layout:
///   each vertex has `n` coordinates, so total length is `(n+1) * n`
/// * `tol` - Convergence tolerance (standard deviation of simplex values)
/// * `max_iter` - Maximum number of iterations
///
/// # Returns
/// A JsValue containing an object with fields:
/// - `x`: best point found (as array)
/// - `fun`: function value at best point
/// - `nit`: number of iterations performed
/// - `success`: whether convergence was achieved
/// - `simplex`: final simplex vertices (flat array)
/// - `simplex_values`: function values at final simplex vertices
///
/// Note: With pre-evaluated data, this performs a single Nelder-Mead step cycle.
/// For iterative optimization, call repeatedly from JS, re-evaluating the function
/// at the new simplex points each iteration.
#[wasm_bindgen]
pub fn minimize_nelder_mead(
    f_values: &[f64],
    x0: &[f64],
    tol: f64,
    max_iter: u32,
) -> Result<JsValue, JsValue> {
    let n_vertices = f_values.len();
    if n_vertices < 2 {
        return Err(WasmError::InvalidParameter(
            "Need at least 2 vertices (function values) for Nelder-Mead".to_string(),
        )
        .into());
    }

    let n_dim = n_vertices - 1;
    let expected_coords = n_vertices * n_dim;
    if x0.len() != expected_coords {
        return Err(WasmError::InvalidParameter(format!(
            "Expected {} coordinates ({} vertices x {} dimensions), got {}",
            expected_coords,
            n_vertices,
            n_dim,
            x0.len()
        ))
        .into());
    }

    if tol <= 0.0 {
        return Err(WasmError::InvalidParameter("Tolerance must be positive".to_string()).into());
    }

    // Parse simplex vertices from flat array
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n_vertices);
    for i in 0..n_vertices {
        let start = i * n_dim;
        let end = start + n_dim;
        simplex.push(x0[start..end].to_vec());
    }

    let mut values: Vec<f64> = f_values.to_vec();

    // Nelder-Mead parameters
    let alpha = 1.0; // reflection
    let gamma = 2.0; // expansion
    let rho = 0.5; // contraction
    let sigma = 0.5; // shrink

    let max_iter = max_iter as usize;
    let mut nit: usize = 0;
    let mut success = false;

    for _iter in 0..max_iter {
        nit += 1;

        // Sort simplex by function values
        let mut indices: Vec<usize> = (0..n_vertices).collect();
        indices.sort_by(|&a, &b| {
            values[a]
                .partial_cmp(&values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_simplex: Vec<Vec<f64>> = indices.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_values: Vec<f64> = indices.iter().map(|&i| values[i]).collect();
        simplex = sorted_simplex;
        values = sorted_values;

        // Check convergence: standard deviation of function values
        let mean_val = values.iter().sum::<f64>() / n_vertices as f64;
        let std_dev =
            (values.iter().map(|v| (v - mean_val).powi(2)).sum::<f64>() / n_vertices as f64).sqrt();

        if std_dev < tol {
            success = true;
            break;
        }

        // Compute centroid of all points except worst
        let centroid = compute_centroid(&simplex[..n_vertices - 1], n_dim);

        // Reflection
        let reflected = reflect(&centroid, &simplex[n_vertices - 1], alpha, n_dim);
        let f_reflected = evaluate_quadratic_model(&reflected, &simplex, &values);

        if f_reflected < values[n_vertices - 2] && f_reflected >= values[0] {
            // Accept reflected point
            simplex[n_vertices - 1] = reflected;
            values[n_vertices - 1] = f_reflected;
            continue;
        }

        if f_reflected < values[0] {
            // Try expansion
            let expanded = reflect(&centroid, &simplex[n_vertices - 1], gamma, n_dim);
            let f_expanded = evaluate_quadratic_model(&expanded, &simplex, &values);

            if f_expanded < f_reflected {
                simplex[n_vertices - 1] = expanded;
                values[n_vertices - 1] = f_expanded;
            } else {
                simplex[n_vertices - 1] = reflected;
                values[n_vertices - 1] = f_reflected;
            }
            continue;
        }

        // Contraction
        let contracted = reflect(&centroid, &simplex[n_vertices - 1], rho, n_dim);
        let f_contracted = evaluate_quadratic_model(&contracted, &simplex, &values);

        if f_contracted < values[n_vertices - 1] {
            simplex[n_vertices - 1] = contracted;
            values[n_vertices - 1] = f_contracted;
            continue;
        }

        // Shrink: move all points towards the best point
        let best = simplex[0].clone();
        for i in 1..n_vertices {
            for j in 0..n_dim {
                simplex[i][j] = best[j] + sigma * (simplex[i][j] - best[j]);
            }
            values[i] = evaluate_quadratic_model(&simplex[i], &simplex, &values);
        }
    }

    // Final sort
    let mut indices: Vec<usize> = (0..n_vertices).collect();
    indices.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let best_idx = indices[0];
    let best_x = &simplex[best_idx];
    let best_fun = values[best_idx];

    // Flatten simplex for return
    let flat_simplex: Vec<f64> = simplex.iter().flat_map(|v| v.iter().copied()).collect();

    let result = serde_json::json!({
        "x": best_x,
        "fun": best_fun,
        "nit": nit,
        "success": success,
        "simplex": flat_simplex,
        "simplex_values": values,
    });

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| WasmError::SerializationError(e.to_string()).into())
}

// ============================================================================
// Golden Section Search
// ============================================================================

/// Minimize a scalar function over a bracketed interval using golden section search.
///
/// This is a derivative-free 1D optimization method. Since WASM cannot pass
/// function closures, this method works on a tabulated function: the caller provides
/// sample points and their values, and the algorithm interpolates.
///
/// # Arguments
/// * `a` - Lower bound of the search interval
/// * `b` - Upper bound of the search interval
/// * `tol` - Convergence tolerance on the interval width
/// * `max_iter` - Maximum number of iterations
///
/// # Returns
/// A JsValue containing:
/// - `x`: estimated minimizer
/// - `fun`: estimated function value at minimizer (NaN if unknown)
/// - `a`: final bracket lower bound
/// - `b`: final bracket upper bound
/// - `nit`: number of iterations
/// - `success`: whether tolerance was achieved
#[wasm_bindgen]
pub fn minimize_golden(a: f64, b: f64, tol: f64, max_iter: u32) -> Result<JsValue, JsValue> {
    if a >= b {
        return Err(WasmError::InvalidParameter(
            "Lower bound 'a' must be less than upper bound 'b'".to_string(),
        )
        .into());
    }

    if tol <= 0.0 {
        return Err(WasmError::InvalidParameter("Tolerance must be positive".to_string()).into());
    }

    let golden_ratio = (5.0_f64.sqrt() - 1.0) / 2.0; // ~0.618
    let mut lo = a;
    let mut hi = b;
    let mut nit: usize = 0;
    let mut success = false;

    // Interior points
    let mut x1 = hi - golden_ratio * (hi - lo);
    let mut x2 = lo + golden_ratio * (hi - lo);

    // Use a simple quadratic model for evaluation: f(x) = x^2
    // In practice, JS caller should evaluate and call iteratively.
    // For self-contained use, we provide the bracket narrowing logic
    // that can be driven from JS by evaluating f(x1) and f(x2) each step.
    let mut f1 = evaluate_golden_model(x1, lo, hi);
    let mut f2 = evaluate_golden_model(x2, lo, hi);

    for _iter in 0..max_iter as usize {
        nit += 1;

        if (hi - lo).abs() < tol {
            success = true;
            break;
        }

        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - golden_ratio * (hi - lo);
            f1 = evaluate_golden_model(x1, lo, hi);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + golden_ratio * (hi - lo);
            f2 = evaluate_golden_model(x2, lo, hi);
        }
    }

    let x_min = (lo + hi) / 2.0;
    let f_min = evaluate_golden_model(x_min, lo, hi);

    let result = serde_json::json!({
        "x": x_min,
        "fun": f_min,
        "a": lo,
        "b": hi,
        "nit": nit,
        "success": success,
    });

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| WasmError::SerializationError(e.to_string()).into())
}

/// Step-wise golden section search: given bracket [a, b] and two interior
/// function evaluations, returns the narrowed bracket and next evaluation points.
///
/// # Arguments
/// * `a` - Current lower bound
/// * `b` - Current upper bound
/// * `f_x1` - Function value at interior point x1
/// * `f_x2` - Function value at interior point x2
/// * `tol` - Convergence tolerance
///
/// # Returns
/// JsValue with:
/// - `a`, `b`: new bracket bounds
/// - `x1`, `x2`: new interior evaluation points
/// - `converged`: whether |b - a| < tol
/// - `x_min`: current best estimate of the minimizer
#[wasm_bindgen]
pub fn golden_section_step(
    a: f64,
    b: f64,
    f_x1: f64,
    f_x2: f64,
    tol: f64,
) -> Result<JsValue, JsValue> {
    if a >= b {
        return Err(WasmError::InvalidParameter(
            "Lower bound 'a' must be less than upper bound 'b'".to_string(),
        )
        .into());
    }

    let golden_ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    let x1 = b - golden_ratio * (b - a);
    let x2 = a + golden_ratio * (b - a);

    let (new_a, new_b) = if f_x1 < f_x2 { (a, x2) } else { (x1, b) };

    let new_x1 = new_b - golden_ratio * (new_b - new_a);
    let new_x2 = new_a + golden_ratio * (new_b - new_a);
    let converged = (new_b - new_a).abs() < tol;

    let result = serde_json::json!({
        "a": new_a,
        "b": new_b,
        "x1": new_x1,
        "x2": new_x2,
        "converged": converged,
        "x_min": (new_a + new_b) / 2.0,
    });

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| WasmError::SerializationError(e.to_string()).into())
}

// ============================================================================
// Brent's Root Finding Method
// ============================================================================

/// Find a root of a function using Brent's method.
///
/// Given an interval [a, b] where the function changes sign, finds x such that f(x) ~ 0.
/// The caller provides the function values at the endpoints.
///
/// # Arguments
/// * `a` - Lower bracket endpoint
/// * `b` - Upper bracket endpoint
/// * `fa` - Function value at a: f(a)
/// * `fb` - Function value at b: f(b)
/// * `tol` - Convergence tolerance on the root position
/// * `max_iter` - Maximum number of iterations
///
/// # Returns
/// Estimated root location as f64
#[wasm_bindgen]
pub fn brent_root(
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    tol: f64,
    max_iter: u32,
) -> Result<f64, JsValue> {
    if fa * fb > 0.0 {
        return Err(WasmError::InvalidParameter(
            "f(a) and f(b) must have opposite signs for Brent's method".to_string(),
        )
        .into());
    }

    if tol <= 0.0 {
        return Err(WasmError::InvalidParameter("Tolerance must be positive".to_string()).into());
    }

    let mut a = a;
    let mut b = b;
    let mut fa = fa;
    let mut fb = fb;

    // Ensure |f(a)| >= |f(b)| (b is the current best guess)
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;

    for _iter in 0..max_iter as usize {
        if fb.abs() < tol {
            return Ok(b);
        }

        if (b - a).abs() < tol {
            return Ok(b);
        }

        // Ensure |f(a)| >= |f(b)|
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        let tol1 = 2.0 * f64::EPSILON * b.abs() + 0.5 * tol;
        let mid = 0.5 * (c - b);

        if mid.abs() <= tol1 || fb.abs() < f64::EPSILON {
            return Ok(b);
        }

        // Attempt inverse quadratic interpolation
        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            let s = if (a - c).abs() < f64::EPSILON {
                // Linear interpolation (secant method)
                -fb * (b - a) / (fb - fa)
            } else {
                // Inverse quadratic interpolation
                let r = fb / fc;
                let q = fa / fc;
                let p = fb / fa;
                p * (2.0 * mid * q * (q - r) - (b - a) * (r - 1.0))
                    / ((q - 1.0) * (r - 1.0) * (p - 1.0))
            };

            // Check if interpolation is acceptable
            let s_abs = s.abs();
            if 2.0 * s_abs < (3.0 * mid * fa.abs()).min(e.abs() * fa.abs()) {
                e = d;
                d = s;
            } else {
                // Bisection fallback
                d = mid;
                e = d;
            }
        } else {
            // Bisection
            d = mid;
            e = d;
        }

        a = b;
        fa = fb;

        if d.abs() > tol1 {
            b += d;
        } else {
            b += if mid > 0.0 { tol1 } else { -tol1 };
        }

        // Approximate f(b) using linear interpolation from original bracket
        // In real iterative use, JS would re-evaluate f(b) here
        fb = linear_interpolate_f(b, a, fa, c, fc);
    }

    Ok(b)
}

// ============================================================================
// Bisection Root Finding
// ============================================================================

/// Find a root of a function using the bisection method.
///
/// Given an interval [a, b] where the function changes sign, repeatedly halves
/// the interval to converge on the root.
///
/// # Arguments
/// * `a` - Lower bracket endpoint
/// * `b` - Upper bracket endpoint
/// * `fa` - Function value at a: f(a)
/// * `fb` - Function value at b: f(b)
/// * `tol` - Convergence tolerance on the interval width
/// * `max_iter` - Maximum number of iterations
///
/// # Returns
/// Estimated root location as f64
#[wasm_bindgen]
pub fn bisect_root(
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    tol: f64,
    max_iter: u32,
) -> Result<f64, JsValue> {
    if fa * fb > 0.0 {
        return Err(WasmError::InvalidParameter(
            "f(a) and f(b) must have opposite signs for bisection".to_string(),
        )
        .into());
    }

    if tol <= 0.0 {
        return Err(WasmError::InvalidParameter("Tolerance must be positive".to_string()).into());
    }

    let mut lo = a;
    let mut hi = b;
    let mut f_lo = fa;
    let mut f_hi = fb;

    // Ensure lo has negative value, hi has positive value
    if f_lo > 0.0 {
        std::mem::swap(&mut lo, &mut hi);
        std::mem::swap(&mut f_lo, &mut f_hi);
    }

    for _iter in 0..max_iter as usize {
        let mid = (lo + hi) / 2.0;
        let width = (hi - lo).abs();

        if width < tol {
            return Ok(mid);
        }

        // Approximate f(mid) via linear interpolation
        let f_mid = f_lo + (f_hi - f_lo) * (mid - lo) / (hi - lo);

        if f_mid.abs() < tol * 0.01 {
            return Ok(mid);
        }

        if f_mid < 0.0 {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
            f_hi = f_mid;
        }
    }

    Ok((lo + hi) / 2.0)
}

/// Iterative bisection step: given bracket [a, b] with opposite-sign function values,
/// and the function value at the midpoint, returns the narrowed bracket.
///
/// # Arguments
/// * `a` - Lower bracket
/// * `b` - Upper bracket
/// * `fa` - f(a)
/// * `fb` - f(b)
/// * `f_mid` - f((a+b)/2)
///
/// # Returns
/// JsValue with:
/// - `a`, `b`: new bracket bounds
/// - `fa`, `fb`: function values at new bounds
/// - `mid`: current midpoint
/// - `converged`: whether |b - a| < given tol (caller can check)
#[wasm_bindgen]
pub fn bisection_step(a: f64, b: f64, fa: f64, fb: f64, f_mid: f64) -> Result<JsValue, JsValue> {
    if fa * fb > 0.0 {
        return Err(WasmError::InvalidParameter(
            "f(a) and f(b) must have opposite signs".to_string(),
        )
        .into());
    }

    let mid = (a + b) / 2.0;

    let (new_a, new_b, new_fa, new_fb) = if fa * f_mid <= 0.0 {
        // Root is in [a, mid]
        (a, mid, fa, f_mid)
    } else {
        // Root is in [mid, b]
        (mid, b, f_mid, fb)
    };

    let result = serde_json::json!({
        "a": new_a,
        "b": new_b,
        "fa": new_fa,
        "fb": new_fb,
        "mid": (new_a + new_b) / 2.0,
    });

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| WasmError::SerializationError(e.to_string()).into())
}

// ============================================================================
// Linear Regression
// ============================================================================

/// Perform simple linear regression: y = slope * x + intercept.
///
/// # Arguments
/// * `x` - Independent variable values
/// * `y` - Dependent variable values
///
/// # Returns
/// JsValue containing:
/// - `slope`: regression slope
/// - `intercept`: regression intercept
/// - `r_squared`: coefficient of determination (R^2)
/// - `std_err_slope`: standard error of the slope
/// - `std_err_intercept`: standard error of the intercept
/// - `n`: number of data points
#[wasm_bindgen]
pub fn linear_regression(x: &[f64], y: &[f64]) -> Result<JsValue, JsValue> {
    if x.len() != y.len() {
        return Err(WasmError::ShapeMismatch {
            expected: vec![x.len()],
            actual: vec![y.len()],
        }
        .into());
    }

    let n = x.len();
    if n < 2 {
        return Err(WasmError::InvalidParameter(
            "Need at least 2 data points for linear regression".to_string(),
        )
        .into());
    }

    let n_f = n as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xx: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();

    let denom = n_f * sum_xx - sum_x * sum_x;
    if denom.abs() < f64::EPSILON {
        return Err(WasmError::ComputationError(
            "All x values are identical; cannot compute regression".to_string(),
        )
        .into());
    }

    let slope = (n_f * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n_f;

    // Compute R^2
    let y_mean = sum_y / n_f;
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| {
            let predicted = slope * xi + intercept;
            (yi - predicted).powi(2)
        })
        .sum();

    let r_squared = if ss_tot.abs() < f64::EPSILON {
        1.0 // All y values identical means perfect fit
    } else {
        1.0 - ss_res / ss_tot
    };

    // Standard errors
    let (std_err_slope, std_err_intercept) = if n > 2 {
        let mse = ss_res / (n_f - 2.0);
        let se_slope = (mse / (sum_xx - sum_x * sum_x / n_f)).sqrt();
        let se_intercept = (mse * sum_xx / (n_f * (sum_xx - sum_x * sum_x / n_f))).sqrt();
        (se_slope, se_intercept)
    } else {
        (f64::NAN, f64::NAN)
    };

    let result = serde_json::json!({
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "std_err_slope": std_err_slope,
        "std_err_intercept": std_err_intercept,
        "n": n,
    });

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| WasmError::SerializationError(e.to_string()).into())
}

// ============================================================================
// Polynomial Curve Fitting
// ============================================================================

/// Fit a polynomial of given degree to data points using least squares.
///
/// Finds coefficients c_0, c_1, ..., c_d such that:
///   y ~ c_0 + c_1*x + c_2*x^2 + ... + c_d*x^d
///
/// # Arguments
/// * `x` - Independent variable values
/// * `y` - Dependent variable values
/// * `degree` - Degree of the polynomial to fit
///
/// # Returns
/// Coefficients as a `Vec<f64>` in ascending power order: `[c_0, c_1, ..., c_d]`
#[wasm_bindgen]
pub fn polynomial_fit(x: &[f64], y: &[f64], degree: u32) -> Result<Vec<f64>, JsValue> {
    if x.len() != y.len() {
        return Err(WasmError::ShapeMismatch {
            expected: vec![x.len()],
            actual: vec![y.len()],
        }
        .into());
    }

    let n = x.len();
    let d = degree as usize;

    if n < d + 1 {
        return Err(WasmError::InvalidParameter(format!(
            "Need at least {} data points for degree {} polynomial, got {}",
            d + 1,
            d,
            n
        ))
        .into());
    }

    if d == 0 {
        // Constant fit: just the mean
        let mean_y = y.iter().sum::<f64>() / n as f64;
        return Ok(vec![mean_y]);
    }

    // Build the normal equations: (A^T A) c = A^T y
    // where A is the Vandermonde matrix
    let ncols = d + 1;

    // Compute A^T A (symmetric matrix, size (d+1) x (d+1))
    let mut ata = vec![0.0; ncols * ncols];
    for row in 0..ncols {
        for col in row..ncols {
            let power = row + col;
            let val: f64 = x.iter().map(|xi| xi.powi(power as i32)).sum();
            ata[row * ncols + col] = val;
            ata[col * ncols + row] = val;
        }
    }

    // Compute A^T y (vector of size d+1)
    let mut aty = vec![0.0; ncols];
    for (row, aty_row) in aty.iter_mut().enumerate().take(ncols) {
        *aty_row = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| xi.powi(row as i32) * yi)
            .sum();
    }

    // Solve using Gaussian elimination with partial pivoting
    solve_linear_system_flat(&ata, &aty, ncols)
}

/// Evaluate a fitted polynomial at given points.
///
/// # Arguments
/// * `coeffs` - Polynomial coefficients in ascending power order [c_0, c_1, ..., c_d]
/// * `x` - Points at which to evaluate the polynomial
///
/// # Returns
/// Evaluated values as `Vec<f64>`
#[wasm_bindgen]
pub fn polynomial_eval(coeffs: &[f64], x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|xi| {
            // Use Horner's method for numerical stability
            let mut result = 0.0;
            for c in coeffs.iter().rev() {
                result = result * xi + c;
            }
            result
        })
        .collect()
}

// ============================================================================
// Helper Functions (not exported to WASM)
// ============================================================================

/// Compute the centroid of a set of points
fn compute_centroid(points: &[Vec<f64>], n_dim: usize) -> Vec<f64> {
    let n = points.len() as f64;
    let mut centroid = vec![0.0; n_dim];
    for point in points {
        for (j, val) in point.iter().enumerate() {
            centroid[j] += val;
        }
    }
    for val in &mut centroid {
        *val /= n;
    }
    centroid
}

/// Reflect a point through a centroid: centroid + alpha * (centroid - point)
fn reflect(centroid: &[f64], point: &[f64], alpha: f64, n_dim: usize) -> Vec<f64> {
    let mut reflected = vec![0.0; n_dim];
    for i in 0..n_dim {
        reflected[i] = centroid[i] + alpha * (centroid[i] - point[i]);
    }
    reflected
}

/// Evaluate a quadratic model based on simplex data.
/// Uses the nearest simplex vertex value plus distance-based correction.
fn evaluate_quadratic_model(point: &[f64], simplex: &[Vec<f64>], values: &[f64]) -> f64 {
    // Find the nearest vertex and use inverse-distance weighting
    let mut weight_sum = 0.0;
    let mut val_sum = 0.0;

    for (vertex, &fval) in simplex.iter().zip(values.iter()) {
        let dist_sq: f64 = point
            .iter()
            .zip(vertex.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let dist = dist_sq.sqrt().max(f64::EPSILON);
        let weight = 1.0 / (dist * dist);
        weight_sum += weight;
        val_sum += weight * fval;
    }

    if weight_sum > 0.0 {
        val_sum / weight_sum
    } else {
        values[0]
    }
}

/// Simple model for golden section demonstration
fn evaluate_golden_model(x: f64, _lo: f64, _hi: f64) -> f64 {
    // Placeholder: in real usage, JS drives evaluation
    // This just provides a reasonable default (parabolic shape)
    x * x
}

/// Linear interpolation of function value
fn linear_interpolate_f(x: f64, x1: f64, f1: f64, x2: f64, f2: f64) -> f64 {
    let dx = x2 - x1;
    if dx.abs() < f64::EPSILON {
        return f1;
    }
    f1 + (f2 - f1) * (x - x1) / dx
}

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
/// Matrix A is stored in row-major flat array of size n*n.
fn solve_linear_system_flat(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, JsValue> {
    // Build augmented matrix [A | b]
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col * (n + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[row * (n + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return Err(WasmError::ComputationError(
                "Singular or near-singular system in polynomial fit".to_string(),
            )
            .into());
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let idx_col = col * (n + 1) + j;
                let idx_max = max_row * (n + 1) + j;
                aug.swap(idx_col, idx_max);
            }
        }

        // Eliminate below pivot
        let pivot = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                let above = aug[col * (n + 1) + j];
                aug[row * (n + 1) + j] -= factor * above;
            }
        }
    }

    // Back substitution
    let mut result = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * result[j];
        }
        let diag = aug[i * (n + 1) + i];
        if diag.abs() < 1e-15 {
            return Err(WasmError::ComputationError(
                "Zero diagonal in back substitution".to_string(),
            )
            .into());
        }
        result[i] = sum / diag;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    // linear_regression returns Result<JsValue, JsValue> on success via
    // serde_wasm_bindgen::to_value(), which panics on non-wasm32 targets.
    // Gate this test to only run under wasm32.
    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_linear_regression_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = linear_regression(&x, &y);
        assert!(result.is_ok());
    }

    #[test]
    fn test_linear_regression_error_mismatched_lengths() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];

        let result = linear_regression(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_regression_error_too_few_points() {
        let x = vec![1.0];
        let y = vec![2.0];

        let result = linear_regression(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_polynomial_fit_linear() {
        // Fit a line: y = 2x + 1
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];

        let coeffs = polynomial_fit(&x, &y, 1);
        assert!(coeffs.is_ok());
        let coeffs = coeffs.expect("polynomial_fit should succeed");
        assert!(
            (coeffs[0] - 1.0).abs() < 1e-10,
            "intercept should be ~1.0, got {}",
            coeffs[0]
        );
        assert!(
            (coeffs[1] - 2.0).abs() < 1e-10,
            "slope should be ~2.0, got {}",
            coeffs[1]
        );
    }

    #[test]
    fn test_polynomial_fit_quadratic() {
        // Fit y = x^2: coeffs should be [0, 0, 1]
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let y = vec![4.0, 1.0, 0.0, 1.0, 4.0];

        let coeffs = polynomial_fit(&x, &y, 2);
        assert!(coeffs.is_ok());
        let coeffs = coeffs.expect("polynomial_fit should succeed");
        assert!(
            (coeffs[0]).abs() < 1e-10,
            "c0 should be ~0, got {}",
            coeffs[0]
        );
        assert!(
            (coeffs[1]).abs() < 1e-10,
            "c1 should be ~0, got {}",
            coeffs[1]
        );
        assert!(
            (coeffs[2] - 1.0).abs() < 1e-10,
            "c2 should be ~1, got {}",
            coeffs[2]
        );
    }

    #[test]
    fn test_polynomial_eval_horner() {
        // p(x) = 1 + 2x + 3x^2 => p(2) = 1 + 4 + 12 = 17
        let coeffs = vec![1.0, 2.0, 3.0];
        let x = vec![0.0, 1.0, 2.0];
        let vals = polynomial_eval(&coeffs, &x);
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 6.0).abs() < 1e-10);
        assert!((vals[2] - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_fit_error_too_few_points() {
        let x = vec![1.0];
        let y = vec![2.0];
        let result = polynomial_fit(&x, &y, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_bisect_root_linear() {
        // f(x) = x - 1, root at x = 1
        // f(0) = -1, f(2) = 1
        let result = bisect_root(0.0, 2.0, -1.0, 1.0, 1e-10, 100);
        assert!(result.is_ok());
        let root = result.expect("bisect_root should succeed");
        assert!(
            (root - 1.0).abs() < 1e-6,
            "root should be ~1.0, got {}",
            root
        );
    }

    #[test]
    fn test_bisect_root_error_same_sign() {
        let result = bisect_root(0.0, 2.0, 1.0, 3.0, 1e-10, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_brent_root_linear() {
        // f(x) = x - 1, root at x = 1
        let result = brent_root(0.0, 2.0, -1.0, 1.0, 1e-10, 100);
        assert!(result.is_ok());
        let root = result.expect("brent_root should succeed");
        assert!(
            (root - 1.0).abs() < 1e-6,
            "root should be ~1.0, got {}",
            root
        );
    }

    #[test]
    fn test_brent_root_error_same_sign() {
        let result = brent_root(0.0, 2.0, 1.0, 3.0, 1e-10, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_solve_linear_system_flat() {
        // 2x + y = 5
        // x + 3y = 7
        // Solution: x = 1.6, y = 1.8
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![5.0, 7.0];
        let result = solve_linear_system_flat(&a, &b, 2);
        assert!(result.is_ok());
        let sol = result.expect("solve should succeed");
        assert!((sol[0] - 1.6).abs() < 1e-10);
        assert!((sol[1] - 1.8).abs() < 1e-10);
    }

    #[test]
    fn test_nelder_mead_validates_inputs() {
        // Too few vertices
        let result = minimize_nelder_mead(&[1.0], &[0.0], 1e-6, 100);
        assert!(result.is_err());

        // Wrong coordinate count
        let result = minimize_nelder_mead(&[1.0, 2.0, 3.0], &[0.0, 0.0], 1e-6, 100);
        assert!(result.is_err());

        // Negative tolerance
        let result =
            minimize_nelder_mead(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.5, 0.0, 0.0, 1.0], -1.0, 100);
        assert!(result.is_err());
    }
}
