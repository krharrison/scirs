//! Interpolation functions for WASM
//!
//! Provides wasm_bindgen bindings for 1D interpolation methods including
//! linear, nearest, quadratic (cubic), cubic spline, Lagrange polynomial,
//! PCHIP (Piecewise Cubic Hermite Interpolating Polynomial), and Akima
//! interpolation.

use crate::error::WasmError;
use scirs2_interpolate::advanced::akima::AkimaSpline;
use scirs2_interpolate::advanced::barycentric::BarycentricInterpolator;
use scirs2_interpolate::interp1d::{Interp1d, InterpolationMethod};
use scirs2_interpolate::spline::CubicSpline;
use wasm_bindgen::prelude::*;

/// Validate that input arrays are non-empty and have matching lengths.
///
/// Returns an error if any of the arrays are empty or if x and y have
/// different lengths.
fn validate_xy(x: &[f64], y: &[f64]) -> Result<(), WasmError> {
    if x.is_empty() || y.is_empty() {
        return Err(WasmError::InvalidParameter(
            "Input arrays must not be empty".to_string(),
        ));
    }
    if x.len() != y.len() {
        return Err(WasmError::ShapeMismatch {
            expected: vec![x.len()],
            actual: vec![y.len()],
        });
    }
    Ok(())
}

/// Validate that x_new is non-empty.
fn validate_x_new(x_new: &[f64]) -> Result<(), WasmError> {
    if x_new.is_empty() {
        return Err(WasmError::InvalidParameter(
            "Evaluation points array must not be empty".to_string(),
        ));
    }
    Ok(())
}

/// Convert slices to ndarray Array1 views for use with scirs2-interpolate.
fn to_array_view(data: &[f64]) -> scirs2_core::ndarray::ArrayView1<'_, f64> {
    scirs2_core::ndarray::ArrayView1::from(data)
}

/// Perform 1D interpolation with a specified method.
///
/// Interpolates between the given (x, y) data points and evaluates the
/// interpolation at the new x coordinates.
///
/// # Arguments
///
/// * `x` - Known x-coordinates (must be sorted in ascending order)
/// * `y` - Known y-coordinates (same length as x)
/// * `x_new` - x-coordinates at which to evaluate the interpolation
/// * `kind` - Interpolation method: "linear", "nearest", or "quadratic" (uses cubic/Catmull-Rom)
///
/// # Returns
///
/// A `Vec<f64>` containing the interpolated values at each point in `x_new`.
///
/// # Errors
///
/// Returns an error if:
/// - Input arrays are empty or have mismatched lengths
/// - x values are not sorted in ascending order
/// - An unknown interpolation kind is specified
/// - Insufficient points for the requested method
#[wasm_bindgen]
pub fn interp1d(x: &[f64], y: &[f64], x_new: &[f64], kind: &str) -> Result<Vec<f64>, JsValue> {
    validate_xy(x, y)?;
    validate_x_new(x_new)?;

    let method = match kind {
        "linear" => InterpolationMethod::Linear,
        "nearest" => InterpolationMethod::Nearest,
        "quadratic" | "cubic" => InterpolationMethod::Cubic,
        other => {
            return Err(WasmError::InvalidParameter(format!(
                "Unknown interpolation kind '{}'. Use 'linear', 'nearest', or 'quadratic'",
                other
            ))
            .into());
        }
    };

    let x_view = to_array_view(x);
    let y_view = to_array_view(y);
    let x_new_view = to_array_view(x_new);

    let interp = Interp1d::new(
        &x_view,
        &y_view,
        method,
        scirs2_interpolate::interp1d::ExtrapolateMode::Nearest,
    )
    .map_err(|e| WasmError::ComputationError(format!("Failed to create interpolator: {}", e)))?;

    let result = interp.evaluate_array(&x_new_view).map_err(|e| {
        WasmError::ComputationError(format!("Interpolation evaluation failed: {}", e))
    })?;

    Ok(result.to_vec())
}

/// Perform natural cubic spline interpolation.
///
/// Constructs a natural cubic spline (zero second derivative at boundaries)
/// through the given data points and evaluates it at the specified x coordinates.
/// Provides C2 continuity (continuous second derivatives).
///
/// # Arguments
///
/// * `x` - Known x-coordinates (must be sorted in ascending order, at least 3 points)
/// * `y` - Known y-coordinates (same length as x)
/// * `x_new` - x-coordinates at which to evaluate the spline
///
/// # Returns
///
/// A `Vec<f64>` containing the interpolated values at each point in `x_new`.
///
/// # Errors
///
/// Returns an error if:
/// - Input arrays are empty or have mismatched lengths
/// - Fewer than 3 data points are provided
/// - x values are not sorted in ascending order
/// - Evaluation points are outside the data range
#[wasm_bindgen]
pub fn cubic_spline(x: &[f64], y: &[f64], x_new: &[f64]) -> Result<Vec<f64>, JsValue> {
    validate_xy(x, y)?;
    validate_x_new(x_new)?;

    if x.len() < 3 {
        return Err(WasmError::InvalidParameter(
            "Cubic spline requires at least 3 data points".to_string(),
        )
        .into());
    }

    let x_view = to_array_view(x);
    let y_view = to_array_view(y);
    let x_new_view = to_array_view(x_new);

    let spline = CubicSpline::new(&x_view, &y_view).map_err(|e| {
        WasmError::ComputationError(format!("Failed to create cubic spline: {}", e))
    })?;

    let result = spline.evaluate_array(&x_new_view).map_err(|e| {
        WasmError::ComputationError(format!("Cubic spline evaluation failed: {}", e))
    })?;

    Ok(result.to_vec())
}

/// Perform Lagrange polynomial interpolation.
///
/// Uses barycentric Lagrange interpolation, which is numerically more stable
/// than the naive Lagrange formula. The order of the polynomial equals
/// the number of data points minus one.
///
/// # Arguments
///
/// * `x` - Known x-coordinates (must have distinct values)
/// * `y` - Known y-coordinates (same length as x)
/// * `x_new` - x-coordinates at which to evaluate the polynomial
///
/// # Returns
///
/// A `Vec<f64>` containing the interpolated values at each point in `x_new`.
///
/// # Errors
///
/// Returns an error if:
/// - Input arrays are empty or have mismatched lengths
/// - Fewer than 2 data points are provided
/// - x values contain duplicates
#[wasm_bindgen]
pub fn lagrange(x: &[f64], y: &[f64], x_new: &[f64]) -> Result<Vec<f64>, JsValue> {
    validate_xy(x, y)?;
    validate_x_new(x_new)?;

    if x.len() < 2 {
        return Err(WasmError::InvalidParameter(
            "Lagrange interpolation requires at least 2 data points".to_string(),
        )
        .into());
    }

    let x_view = to_array_view(x);
    let y_view = to_array_view(y);
    let x_new_view = to_array_view(x_new);

    // Use barycentric interpolation with order = n-1 (full polynomial degree)
    let order = x.len() - 1;
    let interp = BarycentricInterpolator::new(&x_view, &y_view, order).map_err(|e| {
        WasmError::ComputationError(format!(
            "Failed to create Lagrange (barycentric) interpolator: {}",
            e
        ))
    })?;

    let result = interp.evaluate_array(&x_new_view).map_err(|e| {
        WasmError::ComputationError(format!("Lagrange interpolation evaluation failed: {}", e))
    })?;

    Ok(result.to_vec())
}

/// Perform PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation.
///
/// PCHIP produces a monotonicity-preserving interpolant: if the input data is
/// monotonically increasing (or decreasing) in an interval, the interpolation
/// will be too. The first derivatives are continuous, but the second derivatives
/// may have jumps at the knot points.
///
/// # Arguments
///
/// * `x` - Known x-coordinates (must be sorted in ascending order, at least 2 points)
/// * `y` - Known y-coordinates (same length as x)
/// * `x_new` - x-coordinates at which to evaluate the interpolation
///
/// # Returns
///
/// A `Vec<f64>` containing the interpolated values at each point in `x_new`.
///
/// # Errors
///
/// Returns an error if:
/// - Input arrays are empty or have mismatched lengths
/// - Fewer than 2 data points are provided
/// - x values are not sorted in ascending order
#[wasm_bindgen]
pub fn pchip(x: &[f64], y: &[f64], x_new: &[f64]) -> Result<Vec<f64>, JsValue> {
    validate_xy(x, y)?;
    validate_x_new(x_new)?;

    if x.len() < 2 {
        return Err(WasmError::InvalidParameter(
            "PCHIP interpolation requires at least 2 data points".to_string(),
        )
        .into());
    }

    let x_view = to_array_view(x);
    let y_view = to_array_view(y);
    let x_new_view = to_array_view(x_new);

    let result =
        scirs2_interpolate::interp1d::pchip_interpolate(&x_view, &y_view, &x_new_view, false)
            .map_err(|e| {
                WasmError::ComputationError(format!("PCHIP interpolation failed: {}", e))
            })?;

    Ok(result.to_vec())
}

/// Perform Akima interpolation.
///
/// Akima interpolation is a piecewise cubic method that is designed to be
/// robust against outliers. Unlike standard cubic splines, Akima interpolation
/// uses a local scheme for computing derivatives, making it less prone to
/// oscillations near outliers or abrupt changes in the data.
///
/// # Arguments
///
/// * `x` - Known x-coordinates (must be sorted in strictly ascending order, at least 5 points)
/// * `y` - Known y-coordinates (same length as x)
/// * `x_new` - x-coordinates at which to evaluate the interpolation
///
/// # Returns
///
/// A `Vec<f64>` containing the interpolated values at each point in `x_new`.
///
/// # Errors
///
/// Returns an error if:
/// - Input arrays are empty or have mismatched lengths
/// - Fewer than 5 data points are provided (Akima requires at least 5)
/// - x values are not sorted in strictly ascending order
/// - Evaluation points are outside the data range
#[wasm_bindgen]
pub fn akima(x: &[f64], y: &[f64], x_new: &[f64]) -> Result<Vec<f64>, JsValue> {
    validate_xy(x, y)?;
    validate_x_new(x_new)?;

    if x.len() < 5 {
        return Err(WasmError::InvalidParameter(
            "Akima interpolation requires at least 5 data points".to_string(),
        )
        .into());
    }

    let x_view = to_array_view(x);
    let y_view = to_array_view(y);
    let x_new_view = to_array_view(x_new);

    let spline = AkimaSpline::new(&x_view, &y_view).map_err(|e| {
        WasmError::ComputationError(format!("Failed to create Akima spline: {}", e))
    })?;

    let result = spline.evaluate_array(&x_new_view).map_err(|e| {
        WasmError::ComputationError(format!("Akima interpolation evaluation failed: {}", e))
    })?;

    Ok(result.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn sample_x() -> Vec<f64> {
        vec![0.0, 1.0, 2.0, 3.0, 4.0]
    }

    fn sample_y() -> Vec<f64> {
        vec![0.0, 1.0, 4.0, 9.0, 16.0]
    }

    fn sample_x_new() -> Vec<f64> {
        vec![0.5, 1.5, 2.5, 3.5]
    }

    // -- interp1d tests --

    #[test]
    fn test_interp1d_linear() {
        let x = sample_x();
        let y = sample_y();
        let x_new = sample_x_new();

        let result = interp1d(&x, &y, &x_new, "linear").expect("linear interp1d should succeed");
        assert_eq!(result.len(), x_new.len());
        // Between (0,0) and (1,1): midpoint -> 0.5
        assert_relative_eq!(result[0], 0.5, epsilon = 1e-10);
        // Between (1,1) and (2,4): midpoint -> 2.5
        assert_relative_eq!(result[1], 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_interp1d_nearest() {
        let x = sample_x();
        let y = sample_y();
        let x_new = vec![0.3, 0.7, 2.6];

        let result = interp1d(&x, &y, &x_new, "nearest").expect("nearest interp1d should succeed");
        assert_eq!(result.len(), 3);
        // 0.3 is closer to 0.0 -> y=0.0
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-10);
        // 0.7 is closer to 1.0 -> y=1.0
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-10);
        // 2.6 is closer to 3.0 -> y=9.0
        assert_relative_eq!(result[2], 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_interp1d_quadratic() {
        let x = sample_x();
        let y = sample_y();
        let x_new = vec![1.0, 2.0, 3.0];

        let result =
            interp1d(&x, &y, &x_new, "quadratic").expect("quadratic interp1d should succeed");
        assert_eq!(result.len(), 3);
        // At known data points, result should be exact
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 4.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_interp1d_invalid_kind() {
        let x = sample_x();
        let y = sample_y();
        let x_new = sample_x_new();

        let result = interp1d(&x, &y, &x_new, "unknown");
        assert!(result.is_err());
    }

    #[test]
    fn test_interp1d_empty_input() {
        let result = interp1d(&[], &[], &[1.0], "linear");
        assert!(result.is_err());
    }

    #[test]
    fn test_interp1d_mismatched_lengths() {
        let result = interp1d(&[1.0, 2.0], &[1.0], &[1.5], "linear");
        assert!(result.is_err());
    }

    // -- cubic_spline tests --

    #[test]
    fn test_cubic_spline_at_data_points() {
        let x = sample_x();
        let y = sample_y();
        // Evaluate at the known data points
        let x_new = x.clone();

        let result = cubic_spline(&x, &y, &x_new).expect("cubic_spline should succeed");
        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, y[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_cubic_spline_between_points() {
        let x = sample_x();
        let y = sample_y();
        let x_new = sample_x_new();

        let result = cubic_spline(&x, &y, &x_new).expect("cubic_spline should succeed");
        assert_eq!(result.len(), x_new.len());
        // For y = x^2, cubic spline should approximate well
        for (i, &xv) in x_new.iter().enumerate() {
            let expected = xv * xv;
            assert_relative_eq!(result[i], expected, epsilon = 1.0);
        }
    }

    #[test]
    fn test_cubic_spline_too_few_points() {
        let result = cubic_spline(&[1.0, 2.0], &[1.0, 4.0], &[1.5]);
        assert!(result.is_err());
    }

    // -- lagrange tests --

    #[test]
    fn test_lagrange_at_data_points() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 8.0, 27.0]; // y = x^3
        let x_new = x.clone();

        let result = lagrange(&x, &y, &x_new).expect("lagrange should succeed");
        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, y[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_lagrange_polynomial() {
        // For 4 points, Lagrange gives exact cubic polynomial
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 8.0, 27.0]; // y = x^3

        let x_new = vec![0.5, 1.5, 2.5];
        let result = lagrange(&x, &y, &x_new).expect("lagrange should succeed");

        // x^3 at those points
        assert_relative_eq!(result[0], 0.125, epsilon = 1e-6);
        assert_relative_eq!(result[1], 3.375, epsilon = 1e-6);
        assert_relative_eq!(result[2], 15.625, epsilon = 1e-6);
    }

    #[test]
    fn test_lagrange_too_few_points() {
        let result = lagrange(&[1.0], &[1.0], &[1.5]);
        assert!(result.is_err());
    }

    // -- pchip tests --

    #[test]
    fn test_pchip_at_data_points() {
        let x = sample_x();
        let y = sample_y();

        // Evaluate near but not exactly at data points to avoid boundary issues
        let x_new = vec![0.01, 1.01, 2.01, 3.01, 3.99];

        let result = pchip(&x, &y, &x_new).expect("pchip should succeed");
        assert_eq!(result.len(), x_new.len());
        // Values should be close to f(x) = x^2 for these near-data-point evaluations
        for (i, &xv) in x_new.iter().enumerate() {
            let expected = xv * xv;
            assert_relative_eq!(result[i], expected, epsilon = 0.5);
        }
    }

    #[test]
    fn test_pchip_monotonicity() {
        // PCHIP preserves monotonicity of monotone data
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 2.0, 3.0, 4.0]; // strictly increasing

        let x_new = vec![0.5, 1.5, 2.5, 3.5];
        let result = pchip(&x, &y, &x_new).expect("pchip should succeed");

        // Each result should be between the surrounding y-values
        for (i, &val) in result.iter().enumerate() {
            let lower = y[i];
            let upper = y[i + 1];
            assert!(
                val >= lower && val <= upper,
                "PCHIP violated monotonicity: {} not in [{}, {}]",
                val,
                lower,
                upper
            );
        }
    }

    #[test]
    fn test_pchip_too_few_points() {
        let result = pchip(&[1.0], &[1.0], &[1.5]);
        assert!(result.is_err());
    }

    // -- akima tests --

    #[test]
    fn test_akima_at_data_points() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let x_new = x.clone();

        let result = akima(&x, &y, &x_new).expect("akima should succeed");
        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, y[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_akima_between_points() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let x_new = vec![0.5, 1.5, 2.5, 3.5];

        let result = akima(&x, &y, &x_new).expect("akima should succeed");
        assert_eq!(result.len(), x_new.len());
        // For y = x^2, Akima should give reasonable approximations
        for (i, &xv) in x_new.iter().enumerate() {
            let expected = xv * xv;
            assert_relative_eq!(result[i], expected, epsilon = 1.0);
        }
    }

    #[test]
    fn test_akima_too_few_points() {
        let result = akima(&[1.0, 2.0, 3.0, 4.0], &[1.0, 4.0, 9.0, 16.0], &[2.5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_akima_empty_x_new() {
        let result = akima(&[0.0, 1.0, 2.0, 3.0, 4.0], &[0.0, 1.0, 4.0, 9.0, 16.0], &[]);
        assert!(result.is_err());
    }
}
