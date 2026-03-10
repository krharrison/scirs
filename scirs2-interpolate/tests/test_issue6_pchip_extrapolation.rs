//! Tests for PCHIP extrapolation modes (issue #6 / PR #112)
//!
//! Verifies that `PchipExtrapolateMode::Linear` (default) produces bounded
//! growth for far extrapolation, while `PchipExtrapolateMode::Polynomial`
//! reproduces scipy-compatible cubic continuation.

use approx::assert_relative_eq;
use scirs2_core::ndarray::array;
use scirs2_interpolate::interp1d::{
    ExtrapolateMode, Interp1d, InterpolationMethod, PchipExtrapolateMode, PchipInterpolator,
};

// ---------------------------------------------------------------------------
// PchipExtrapolateMode::Linear (default)
// ---------------------------------------------------------------------------

#[test]
fn linear_mode_is_default() {
    let x = array![0.0, 1.0, 2.0, 3.0];
    let y = array![0.0, 1.0, 4.0, 9.0];
    let interp = PchipInterpolator::new(&x.view(), &y.view(), true).unwrap();

    // Default mode should produce linear extrapolation — constant slope beyond
    // the boundary.
    let y4 = interp.evaluate(4.0).unwrap();
    let y5 = interp.evaluate(5.0).unwrap();
    let slope = y5 - y4; // dx = 1

    let last_deriv = interp.evaluate(3.0).unwrap(); // at boundary
    let _ = last_deriv; // used indirectly via the derivative field

    // The slope between any two extrapolated points should be constant
    let y10 = interp.evaluate(10.0).unwrap();
    let y11 = interp.evaluate(11.0).unwrap();
    assert_relative_eq!(y11 - y10, slope, epsilon = 1e-10);
}

#[test]
fn linear_mode_far_extrapolation_is_bounded() {
    let x = array![0.0f64, 1.0, 2.0, 3.0];
    let y = array![0.0f64, 1.0, 4.0, 9.0];
    let interp = PchipInterpolator::new(&x.view(), &y.view(), true).unwrap();

    let y50 = interp.evaluate(50.0).unwrap();
    assert!(y50 > 9.0, "should exceed last data point, got {}", y50);
    assert!(y50 < 1000.0, "should be bounded (linear), got {}", y50);

    let y_neg50 = interp.evaluate(-50.0).unwrap();
    assert!(y_neg50.is_finite());
    assert!(
        y_neg50.abs() < 1000.0,
        "should be bounded (linear), got {}",
        y_neg50
    );
}

// ---------------------------------------------------------------------------
// PchipExtrapolateMode::Polynomial
// ---------------------------------------------------------------------------

#[test]
fn polynomial_mode_uses_cubic_continuation() {
    let x = array![0.0f64, 1.0, 2.0, 3.0];
    let y = array![0.0f64, 1.0, 4.0, 9.0];
    let interp = PchipInterpolator::new(&x.view(), &y.view(), true)
        .unwrap()
        .with_extrapolate_mode(PchipExtrapolateMode::Polynomial);

    // Polynomial continuation should NOT have constant slope — the second
    // difference should be non-zero for a cubic.
    let y4 = interp.evaluate(4.0).unwrap();
    let y5 = interp.evaluate(5.0).unwrap();
    let y6 = interp.evaluate(6.0).unwrap();

    let slope_45 = y5 - y4;
    let slope_56 = y6 - y5;

    // For cubic continuation the slope changes (second difference != 0)
    assert!(
        (slope_56 - slope_45).abs() > 1e-6,
        "polynomial mode should produce non-constant slope"
    );

    // It should still be finite for moderate extrapolation
    assert!(y6.is_finite());
}

#[test]
fn polynomial_mode_matches_boundary_value() {
    let x = array![0.0, 1.0, 2.0, 3.0];
    let y = array![0.0, 1.0, 4.0, 9.0];
    let interp = PchipInterpolator::new(&x.view(), &y.view(), true)
        .unwrap()
        .with_extrapolate_mode(PchipExtrapolateMode::Polynomial);

    // At the boundary itself, both modes must agree with the data
    assert_relative_eq!(interp.evaluate(0.0).unwrap(), 0.0, epsilon = 1e-12);
    assert_relative_eq!(interp.evaluate(3.0).unwrap(), 9.0, epsilon = 1e-12);
}

// ---------------------------------------------------------------------------
// Interp1d integration — cached PCHIP with polynomial extrapolation
// ---------------------------------------------------------------------------

#[test]
fn interp1d_pchip_extrapolate_uses_polynomial() {
    let x = array![0.0f64, 1.0, 2.0, 3.0];
    let y = array![0.0f64, 1.0, 4.0, 9.0];

    let interp = Interp1d::new(
        &x.view(),
        &y.view(),
        InterpolationMethod::Pchip,
        ExtrapolateMode::Extrapolate,
    )
    .unwrap();

    // Should use polynomial continuation (not simple linear from edge segment)
    let y4 = interp.evaluate(4.0).unwrap();
    let y5 = interp.evaluate(5.0).unwrap();
    let y6 = interp.evaluate(6.0).unwrap();

    let slope_45 = y5 - y4;
    let slope_56 = y6 - y5;

    // Cubic continuation => changing slope
    assert!(
        (slope_56 - slope_45).abs() > 1e-6,
        "Interp1d PCHIP extrapolation should use polynomial continuation"
    );
}

#[test]
fn interp1d_pchip_no_extrapolate_errors() {
    let x = array![0.0, 1.0, 2.0, 3.0];
    let y = array![0.0, 1.0, 4.0, 9.0];

    let interp = Interp1d::new(
        &x.view(),
        &y.view(),
        InterpolationMethod::Pchip,
        ExtrapolateMode::Error,
    )
    .unwrap();

    assert!(interp.evaluate(-0.5).is_err());
    assert!(interp.evaluate(3.5).is_err());
}

#[test]
fn interp1d_pchip_in_range_unchanged() {
    let x = array![0.0, 1.0, 2.0, 3.0];
    let y = array![0.0, 1.0, 4.0, 9.0];

    let interp = Interp1d::new(
        &x.view(),
        &y.view(),
        InterpolationMethod::Pchip,
        ExtrapolateMode::Extrapolate,
    )
    .unwrap();

    // In-range values should be identical regardless of extrapolation mode
    assert_relative_eq!(interp.evaluate(0.0).unwrap(), 0.0, epsilon = 1e-12);
    assert_relative_eq!(interp.evaluate(1.0).unwrap(), 1.0, epsilon = 1e-12);
    assert_relative_eq!(interp.evaluate(2.0).unwrap(), 4.0, epsilon = 1e-12);
    assert_relative_eq!(interp.evaluate(3.0).unwrap(), 9.0, epsilon = 1e-12);

    // Monotonicity preserved
    let y15 = interp.evaluate(1.5).unwrap();
    assert!(y15 > 1.0 && y15 < 4.0);
}
