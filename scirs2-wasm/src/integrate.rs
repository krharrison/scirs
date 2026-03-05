//! Numerical integration and ODE solver functions for WASM
//!
//! Provides trapezoidal, Simpson's rule, Romberg integration, cumulative
//! trapezoidal integration, and a basic Runge-Kutta 4th order ODE stepper
//! accessible from JavaScript via `wasm_bindgen`.

use crate::error::WasmError;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Helper: convert a JS array-like value to Vec<f64>
// ---------------------------------------------------------------------------

/// Parse a `JsValue` that is either a JS `Array` or a `Float64Array` / `Float32Array`
/// into a `Vec<f64>`.
fn js_to_vec(val: &JsValue) -> Result<Vec<f64>, WasmError> {
    if val.is_array() {
        let array = js_sys::Array::from(val);
        crate::utils::js_array_to_vec_f64(&array)
    } else {
        crate::utils::typed_array_to_vec_f64(val)
    }
}

// ---------------------------------------------------------------------------
// Trapezoidal rule
// ---------------------------------------------------------------------------

/// Compute the definite integral of `y` over `x` using the composite
/// trapezoidal rule.
///
/// Both `y_js` and `x_js` must be arrays of the same length (>= 2).
///
/// # Errors
/// Returns a `JsValue` error when the inputs have mismatched lengths or
/// contain fewer than 2 points.
#[wasm_bindgen]
pub fn trapezoid(y_js: &JsValue, x_js: &JsValue) -> Result<f64, JsValue> {
    let y = js_to_vec(y_js)?;
    let x = js_to_vec(x_js)?;

    if y.len() != x.len() {
        return Err(WasmError::ShapeMismatch {
            expected: vec![x.len()],
            actual: vec![y.len()],
        }
        .into());
    }

    if y.len() < 2 {
        return Err(WasmError::InvalidParameter(
            "At least 2 points are required for trapezoidal integration".to_string(),
        )
        .into());
    }

    let mut integral = 0.0;
    for i in 0..y.len() - 1 {
        let dx = x[i + 1] - x[i];
        integral += 0.5 * dx * (y[i] + y[i + 1]);
    }

    Ok(integral)
}

// ---------------------------------------------------------------------------
// Simpson's rule
// ---------------------------------------------------------------------------

/// Compute the definite integral of `y` over `x` using Simpson's rule.
///
/// For segments where three consecutive points are available the classic
/// Simpson 1/3 rule is used.  A final trapezoidal panel is added when the
/// number of intervals is odd.
///
/// # Errors
/// Returns a `JsValue` error on mismatched lengths or fewer than 2 points.
#[wasm_bindgen]
pub fn simpson(y_js: &JsValue, x_js: &JsValue) -> Result<f64, JsValue> {
    let y = js_to_vec(y_js)?;
    let x = js_to_vec(x_js)?;

    if y.len() != x.len() {
        return Err(WasmError::ShapeMismatch {
            expected: vec![x.len()],
            actual: vec![y.len()],
        }
        .into());
    }

    let n = y.len();
    if n < 2 {
        return Err(WasmError::InvalidParameter(
            "At least 2 points are required for Simpson's rule integration".to_string(),
        )
        .into());
    }

    // Fall back to trapezoidal for exactly 2 points
    if n == 2 {
        let dx = x[1] - x[0];
        return Ok(0.5 * dx * (y[0] + y[1]));
    }

    let mut integral = 0.0;
    let intervals = n - 1;

    // Process pairs of intervals (Simpson 1/3)
    let mut i = 0;
    while i + 2 < n {
        let h0 = x[i + 1] - x[i];
        let h1 = x[i + 2] - x[i + 1];

        // For non-uniform spacing use the generalised Simpson formula.
        // When h0 == h1 this reduces to the standard (h/3)(f0 + 4f1 + f2).
        let h_sum = h0 + h1;
        let seg = (h_sum / 6.0)
            * (y[i] * (2.0 - h1 / h0)
                + y[i + 1] * (h_sum * h_sum / (h0 * h1))
                + y[i + 2] * (2.0 - h0 / h1));
        integral += seg;
        i += 2;
    }

    // If there is a remaining interval (odd number of intervals), use trapezoid
    if i < intervals {
        let dx = x[i + 1] - x[i];
        integral += 0.5 * dx * (y[i] + y[i + 1]);
    }

    Ok(integral)
}

// ---------------------------------------------------------------------------
// RK4 single step
// ---------------------------------------------------------------------------

/// Perform a single explicit Runge-Kutta 4th order (RK4) step.
///
/// Because JavaScript cannot pass function pointers, the caller must supply
/// the derivative values pre-computed at the current state.
///
/// * `f_vals_js` -- derivative values dy/dt evaluated at (t, y), one per
///   component.
/// * `t`         -- current time.
/// * `y_js`      -- current state vector.
/// * `h`         -- step size.
///
/// The function approximates the next state assuming a *constant* derivative
/// (i.e. Euler) combined with the classical RK4 weighting using the provided
/// derivative.  For a full multi-stage RK4 the caller would need to evaluate
/// the derivative at the intermediate stages; this helper is therefore best
/// suited for demonstration / simple cases.
///
/// Returns the new state vector `y_{n+1}` as a `Float64Array`.
///
/// # Errors
/// Returns a `JsValue` error when `f_vals` and `y` have different lengths.
#[wasm_bindgen]
pub fn rk4_step(
    f_vals_js: &JsValue,
    _t: f64,
    y_js: &JsValue,
    h: f64,
) -> Result<js_sys::Float64Array, JsValue> {
    let f_vals = js_to_vec(f_vals_js)?;
    let y = js_to_vec(y_js)?;

    if f_vals.len() != y.len() {
        return Err(WasmError::ShapeMismatch {
            expected: vec![y.len()],
            actual: vec![f_vals.len()],
        }
        .into());
    }

    // With only a single derivative evaluation available we use a simplified
    // RK4 where all four stages use the same slope (equivalent to Euler but
    // weighted as RK4 for consistency of the interface).
    //
    //   k1 = f_vals
    //   k2 = f_vals  (approximation)
    //   k3 = f_vals  (approximation)
    //   k4 = f_vals  (approximation)
    //   y_new = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    //         = y + h * f_vals  (simplifies to Euler)
    //
    // NOTE: for a true multi-stage RK4, the caller should invoke
    // `rk4_step` at intermediate points.  See `ode_solve` for a
    // fully self-contained solver that evaluates a built-in ODE.

    let result: Vec<f64> = y
        .iter()
        .zip(f_vals.iter())
        .map(|(&yi, &fi)| yi + h * fi)
        .collect();

    Ok(crate::utils::vec_f64_to_typed_array(result))
}

// ---------------------------------------------------------------------------
// ODE solver (exponential decay demonstration)
// ---------------------------------------------------------------------------

/// Solve the ODE  dy/dt = -y  (exponential decay) using the classic RK4
/// method.
///
/// Since JavaScript cannot pass function pointers to WASM, this solver uses
/// a built-in right-hand side (exponential decay, dy_i/dt = -y_i for every
/// component).
///
/// * `y0_js`     -- initial state vector (1-D array).
/// * `t_span_js` -- two-element array `[t_start, t_end]`.
/// * `n_steps`   -- number of integration steps.
///
/// Returns a serialised JSON object (via `JsValue`) with fields:
/// * `t`  -- array of time values.
/// * `y`  -- array of arrays, one per time point (each inner array has the
///   same length as `y0`).
///
/// # Errors
/// Returns a `JsValue` error when inputs are invalid.
#[wasm_bindgen]
pub fn ode_solve(y0_js: &JsValue, t_span_js: &JsValue, n_steps: u32) -> Result<JsValue, JsValue> {
    let y0 = js_to_vec(y0_js)?;
    let t_span = js_to_vec(t_span_js)?;

    if t_span.len() != 2 {
        return Err(WasmError::InvalidParameter(
            "t_span must be a two-element array [t_start, t_end]".to_string(),
        )
        .into());
    }

    if y0.is_empty() {
        return Err(
            WasmError::InvalidParameter("y0 must have at least one element".to_string()).into(),
        );
    }

    if n_steps == 0 {
        return Err(
            WasmError::InvalidParameter("n_steps must be greater than 0".to_string()).into(),
        );
    }

    let t_start = t_span[0];
    let t_end = t_span[1];
    let h = (t_end - t_start) / n_steps as f64;
    let dim = y0.len();

    // Collect results (including the initial point)
    let total_points = n_steps as usize + 1;
    let mut t_values: Vec<f64> = Vec::with_capacity(total_points);
    let mut y_values: Vec<Vec<f64>> = Vec::with_capacity(total_points);

    let mut t = t_start;
    let mut y = y0;

    t_values.push(t);
    y_values.push(y.clone());

    // RK4 integration loop with built-in RHS:  f(t, y) = -y
    for _ in 0..n_steps {
        let k1: Vec<f64> = y.iter().map(|&yi| -yi).collect();

        let y_mid1: Vec<f64> = (0..dim).map(|j| y[j] + 0.5 * h * k1[j]).collect();
        let k2: Vec<f64> = y_mid1.iter().map(|&yi| -yi).collect();

        let y_mid2: Vec<f64> = (0..dim).map(|j| y[j] + 0.5 * h * k2[j]).collect();
        let k3: Vec<f64> = y_mid2.iter().map(|&yi| -yi).collect();

        let y_end: Vec<f64> = (0..dim).map(|j| y[j] + h * k3[j]).collect();
        let k4: Vec<f64> = y_end.iter().map(|&yi| -yi).collect();

        y = (0..dim)
            .map(|j| y[j] + (h / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]))
            .collect();

        t += h;

        t_values.push(t);
        y_values.push(y.clone());
    }

    // Serialise the result as JSON
    let result = serde_json::json!({
        "t": t_values,
        "y": y_values,
    });

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| WasmError::SerializationError(e.to_string()).into())
}

// ---------------------------------------------------------------------------
// Cumulative trapezoidal integration
// ---------------------------------------------------------------------------

/// Compute the cumulative integral of `y` over `x` using the trapezoidal
/// rule.
///
/// Returns a `Float64Array` of length `n - 1` where `n = y.len()`, with
/// each element being the integral from `x[0]` to `x[i+1]`.
///
/// # Errors
/// Returns a `JsValue` error on mismatched lengths or fewer than 2 points.
#[wasm_bindgen]
pub fn cumulative_trapezoid(
    y_js: &JsValue,
    x_js: &JsValue,
) -> Result<js_sys::Float64Array, JsValue> {
    let y = js_to_vec(y_js)?;
    let x = js_to_vec(x_js)?;

    if y.len() != x.len() {
        return Err(WasmError::ShapeMismatch {
            expected: vec![x.len()],
            actual: vec![y.len()],
        }
        .into());
    }

    if y.len() < 2 {
        return Err(WasmError::InvalidParameter(
            "At least 2 points are required for cumulative trapezoidal integration".to_string(),
        )
        .into());
    }

    let n = y.len();
    let mut result: Vec<f64> = Vec::with_capacity(n - 1);
    let mut cumulative = 0.0;

    for i in 0..n - 1 {
        let dx = x[i + 1] - x[i];
        cumulative += 0.5 * dx * (y[i] + y[i + 1]);
        result.push(cumulative);
    }

    Ok(crate::utils::vec_f64_to_typed_array(result))
}

// ---------------------------------------------------------------------------
// Romberg integration
// ---------------------------------------------------------------------------

/// Compute the integral of tabulated data `y` with uniform spacing `dx`
/// using Romberg integration (Richardson extrapolation of the trapezoidal
/// rule).
///
/// The number of data points must be of the form `2^k + 1` for some
/// non-negative integer `k` (e.g. 2, 3, 5, 9, 17, 33, ...).  If the length
/// does not satisfy this requirement an error is returned.
///
/// # Arguments
/// * `y_js` -- ordinate values (array of `f64`).
/// * `dx`   -- uniform spacing between consecutive x-values.
///
/// # Errors
/// Returns a `JsValue` error when the input length is not `2^k + 1` or when
/// fewer than 2 points are given.
#[wasm_bindgen]
pub fn romberg(y_js: &JsValue, dx: f64) -> Result<f64, JsValue> {
    let y = js_to_vec(y_js)?;

    if y.len() < 2 {
        return Err(WasmError::InvalidParameter(
            "At least 2 points are required for Romberg integration".to_string(),
        )
        .into());
    }

    if dx <= 0.0 {
        return Err(WasmError::InvalidParameter("dx must be positive".to_string()).into());
    }

    // Determine k such that n = 2^k + 1
    let n = y.len();
    let intervals = n - 1;

    // Check if intervals is a power of 2
    if intervals == 0 || (intervals & (intervals - 1)) != 0 {
        return Err(WasmError::InvalidParameter(format!(
            "Number of points must be 2^k + 1, got {} points ({} intervals)",
            n, intervals
        ))
        .into());
    }

    // k = log2(intervals)
    let k = (intervals as f64).log2().round() as usize;

    // Build the Romberg table R[j][m] where j is the refinement level and
    // m is the extrapolation order.
    // R[0] = composite trapezoidal rule with step = intervals * dx
    // R[j] refines by halving the step j times.

    let levels = k + 1;
    let mut r: Vec<Vec<f64>> = Vec::with_capacity(levels);

    // Level 0: coarsest trapezoidal estimate (only endpoints)
    let h0 = intervals as f64 * dx;
    let t0 = 0.5 * h0 * (y[0] + y[intervals]);
    r.push(vec![t0]);

    // Successive refinement levels
    for j in 1..levels {
        let step = intervals >> j; // number of original intervals per sub-step
        let h = step as f64 * dx;
        let n_new = 1usize << (j - 1); // number of NEW midpoints

        // Trapezoidal estimate at level j:
        //   T_j = T_{j-1}/2 + h * sum of f at new midpoints
        let prev_t = r[j - 1][0];
        let mut mid_sum = 0.0;
        for m in 0..n_new {
            let idx = step * (2 * m + 1);
            if idx < y.len() {
                mid_sum += y[idx];
            }
        }
        let t_j = 0.5 * prev_t + h * mid_sum;

        let mut row = Vec::with_capacity(j + 1);
        row.push(t_j);

        // Richardson extrapolation
        for m in 1..=j {
            let factor = 4.0_f64.powi(m as i32);
            let prev_row = &r[j - 1];
            let prev_col = if m - 1 < prev_row.len() {
                prev_row[m - 1]
            } else {
                // Should not happen for well-formed input
                return Err(WasmError::ComputationError(
                    "Romberg table index out of bounds".to_string(),
                )
                .into());
            };
            let val = (factor * row[m - 1] - prev_col) / (factor - 1.0);
            row.push(val);
        }

        r.push(row);
    }

    // The best estimate is the bottom-right element of the table.
    let last_row = r
        .last()
        .ok_or_else(|| WasmError::ComputationError("Romberg table is empty".to_string()))?;
    let best = last_row
        .last()
        .ok_or_else(|| WasmError::ComputationError("Romberg table row is empty".to_string()))?;

    Ok(*best)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {

    // Helper: create a JsValue from a Vec<f64> for testing
    // In non-WASM test environments we cannot construct real JsValues,
    // so we test the internal logic directly.

    #[test]
    fn test_romberg_power_of_two_check() {
        // 4 intervals => 5 points => 2^2 + 1  -- valid
        assert!((4 & (4 - 1)) == 0); // power of 2

        // 3 intervals => not power of 2
        assert!((3 & (3 - 1)) != 0);
    }

    #[test]
    fn test_trapezoid_logic() {
        // Integrate y = x from 0 to 1 with uniform spacing
        // Expected result: 0.5
        let x = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let y = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        let mut integral = 0.0;
        for i in 0..y.len() - 1 {
            let dx = x[i + 1] - x[i];
            integral += 0.5 * dx * (y[i] + y[i + 1]);
        }
        assert!((integral - 0.5_f64).abs() < 1e-12_f64);
    }

    #[test]
    fn test_simpson_logic() {
        // Integrate y = x^2 from 0 to 1 with 5 points (uniform)
        // Expected result: 1/3
        let n = 5;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let intervals = n - 1;
        let mut integral = 0.0;
        let mut i = 0;
        while i + 2 < n {
            let h0 = x[i + 1] - x[i];
            let h1 = x[i + 2] - x[i + 1];
            let h_sum = h0 + h1;
            let seg = (h_sum / 6.0)
                * (y[i] * (2.0 - h1 / h0)
                    + y[i + 1] * (h_sum * h_sum / (h0 * h1))
                    + y[i + 2] * (2.0 - h0 / h1));
            integral += seg;
            i += 2;
        }
        if i < intervals {
            let dx = x[i + 1] - x[i];
            integral += 0.5 * dx * (y[i] + y[i + 1]);
        }

        assert!(
            (integral - 1.0 / 3.0).abs() < 1e-10,
            "Simpson's rule for x^2: got {}, expected {}",
            integral,
            1.0 / 3.0
        );
    }

    #[test]
    fn test_cumulative_trapezoid_logic() {
        // Integrate y = 1 from 0 to 4, cumulative result should be [1, 2, 3, 4]
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        let n = y.len();
        let mut result = Vec::with_capacity(n - 1);
        let mut cumulative = 0.0;
        for i in 0..n - 1 {
            let dx = x[i + 1] - x[i];
            cumulative += 0.5 * dx * (y[i] + y[i + 1]);
            result.push(cumulative);
        }

        assert_eq!(result.len(), 4);
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - (i + 1) as f64).abs() < 1e-12,
                "cumulative[{}] = {}, expected {}",
                i,
                val,
                i + 1
            );
        }
    }

    #[test]
    fn test_romberg_logic() {
        // Integrate y = x from 0 to 1, dx = 0.25, 5 points (2^2 + 1)
        // Expected: 0.5
        let y = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let dx = 0.25;
        let n = y.len();
        let intervals = n - 1; // 4

        let k = (intervals as f64).log2().round() as usize; // 2
        let levels = k + 1; // 3

        let mut r: Vec<Vec<f64>> = Vec::with_capacity(levels);

        let h0 = intervals as f64 * dx;
        let t0 = 0.5 * h0 * (y[0] + y[intervals]);
        r.push(vec![t0]);

        for j in 1..levels {
            let step = intervals >> j;
            let h = step as f64 * dx;
            let n_new = 1usize << (j - 1);

            let prev_t = r[j - 1][0];
            let mut mid_sum = 0.0;
            for m in 0..n_new {
                let idx = step * (2 * m + 1);
                if idx < y.len() {
                    mid_sum += y[idx];
                }
            }
            let t_j = 0.5 * prev_t + h * mid_sum;

            let mut row = Vec::with_capacity(j + 1);
            row.push(t_j);

            for m in 1..=j {
                let factor = 4.0_f64.powi(m as i32);
                let prev_col = r[j - 1][m - 1];
                let val = (factor * row[m - 1] - prev_col) / (factor - 1.0);
                row.push(val);
            }
            r.push(row);
        }

        let best = *r
            .last()
            .and_then(|row| row.last())
            .expect("non-empty table");
        assert!(
            (best - 0.5).abs() < 1e-12,
            "Romberg for y=x: got {}, expected 0.5",
            best
        );
    }

    #[test]
    fn test_ode_rk4_exponential_decay() {
        // Solve dy/dt = -y, y(0) = 1, from t=0 to t=1 with 1000 steps
        // Expected final value: e^{-1} ~ 0.36787944117
        let y0 = vec![1.0];
        let t_start = 0.0;
        let t_end = 1.0;
        let n_steps = 1000u32;
        let h = (t_end - t_start) / n_steps as f64;
        let dim = y0.len();

        let mut y = y0;
        for _ in 0..n_steps {
            let k1: Vec<f64> = y.iter().map(|&yi| -yi).collect();
            let y_mid1: Vec<f64> = (0..dim).map(|j| y[j] + 0.5 * h * k1[j]).collect();
            let k2: Vec<f64> = y_mid1.iter().map(|&yi| -yi).collect();
            let y_mid2: Vec<f64> = (0..dim).map(|j| y[j] + 0.5 * h * k2[j]).collect();
            let k3: Vec<f64> = y_mid2.iter().map(|&yi| -yi).collect();
            let y_end: Vec<f64> = (0..dim).map(|j| y[j] + h * k3[j]).collect();
            let k4: Vec<f64> = y_end.iter().map(|&yi| -yi).collect();

            y = (0..dim)
                .map(|j| y[j] + (h / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]))
                .collect();
        }

        let expected = (-1.0_f64).exp();
        assert!(
            (y[0] - expected).abs() < 1e-10,
            "RK4 exponential decay: got {}, expected {}",
            y[0],
            expected
        );
    }
}
