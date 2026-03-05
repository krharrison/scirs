//! Numeric utilities for floating-point comparison, error measurement, and safe clamping.
//!
//! Provides generic functions over `Float` types:
//!
//! - `approx_eq` -- approximate equality within an absolute tolerance
//! - `relative_error` -- relative error between computed and exact values
//! - `ulp_distance` -- ULP (Unit in the Last Place) distance between floats
//! - `is_finite_and_positive` -- combined finiteness + positivity check
//! - `clamp_to_range` -- safe clamping that rejects NaN
//! - `MachineConstants` trait -- machine epsilon, min/max normal, etc.
//!
//! All functions are generic over `num_traits::Float` so they work with
//! both `f32` and `f64`.

use num_traits::Float;
use std::fmt::Debug;

use crate::error::{CoreError, CoreResult, ErrorContext};

// ---------------------------------------------------------------------------
// MachineConstants trait
// ---------------------------------------------------------------------------

/// Machine-specific constants for a floating-point type.
///
/// Provides type-level access to epsilon, minimum/maximum normal values,
/// minimum positive subnormal, and radix information.
pub trait MachineConstants: Float + Debug {
    /// Machine epsilon (difference between 1.0 and the next representable value).
    fn machine_epsilon() -> Self;

    /// Smallest positive normal number.
    fn min_positive_normal() -> Self;

    /// Largest finite number.
    fn max_finite() -> Self;

    /// Smallest positive subnormal number (closest to zero without being zero).
    fn min_positive_subnormal() -> Self;

    /// Number of significand (mantissa) bits.
    fn mantissa_digits() -> u32;

    /// Radix of the floating-point representation (always 2 for IEEE 754).
    fn radix() -> u32 {
        2
    }
}

impl MachineConstants for f32 {
    fn machine_epsilon() -> Self {
        f32::EPSILON
    }

    fn min_positive_normal() -> Self {
        f32::MIN_POSITIVE
    }

    fn max_finite() -> Self {
        f32::MAX
    }

    fn min_positive_subnormal() -> Self {
        // Smallest positive f32 subnormal: 2^(-149)
        // f32::MIN_POSITIVE is the smallest *normal* value
        // The smallest subnormal is: 1.0 * 2^(-126 - 23) = 2^(-149)
        1.401_298_4e-45_f32
    }

    fn mantissa_digits() -> u32 {
        f32::MANTISSA_DIGITS
    }
}

impl MachineConstants for f64 {
    fn machine_epsilon() -> Self {
        f64::EPSILON
    }

    fn min_positive_normal() -> Self {
        f64::MIN_POSITIVE
    }

    fn max_finite() -> Self {
        f64::MAX
    }

    fn min_positive_subnormal() -> Self {
        // Smallest positive f64 subnormal: 2^(-1074)
        5e-324_f64
    }

    fn mantissa_digits() -> u32 {
        f64::MANTISSA_DIGITS
    }
}

// ---------------------------------------------------------------------------
// approx_eq
// ---------------------------------------------------------------------------

/// Check whether two floating-point values are approximately equal within an
/// absolute tolerance.
///
/// Returns `true` if `|a - b| <= tol`.
///
/// # Arguments
///
/// * `a` - First value
/// * `b` - Second value
/// * `tol` - Absolute tolerance (must be non-negative)
///
/// # Edge Cases
///
/// - If either value is NaN the result is `false`.
/// - If both values are the same infinity the result is `true`.
pub fn approx_eq<T: Float>(a: T, b: T, tol: T) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    // Exact equality catches +inf == +inf and -inf == -inf
    if a == b {
        return true;
    }
    (a - b).abs() <= tol
}

/// Check approximate equality using a relative tolerance.
///
/// Returns `true` if `|a - b| <= rel_tol * max(|a|, |b|)`.
///
/// Falls back to absolute tolerance `abs_tol` for values near zero.
pub fn approx_eq_relative<T: Float>(a: T, b: T, rel_tol: T, abs_tol: T) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a == b {
        return true;
    }
    let diff = (a - b).abs();
    let max_abs = a.abs().max(b.abs());
    diff <= rel_tol * max_abs || diff <= abs_tol
}

// ---------------------------------------------------------------------------
// relative_error
// ---------------------------------------------------------------------------

/// Compute the relative error between a computed value and an exact (reference) value.
///
/// `relative_error = |computed - exact| / |exact|`
///
/// # Returns
///
/// - `Ok(rel_err)` on success.
/// - `Err(CoreError::DomainError)` if `exact` is zero (division by zero).
/// - `Err(CoreError::ValueError)` if either input is NaN.
pub fn relative_error<T: Float + Debug>(computed: T, exact: T) -> CoreResult<T> {
    if computed.is_nan() || exact.is_nan() {
        return Err(CoreError::ValueError(ErrorContext::new(
            "relative_error: NaN input is not allowed",
        )));
    }
    if exact.is_zero() {
        return Err(CoreError::DomainError(ErrorContext::new(
            "relative_error: exact value is zero, relative error is undefined",
        )));
    }
    Ok((computed - exact).abs() / exact.abs())
}

/// Compute relative error with a safe fallback for near-zero exact values.
///
/// If `|exact| < floor`, uses `floor` as the denominator instead.
pub fn relative_error_safe<T: Float + Debug>(computed: T, exact: T, floor: T) -> CoreResult<T> {
    if computed.is_nan() || exact.is_nan() || floor.is_nan() {
        return Err(CoreError::ValueError(ErrorContext::new(
            "relative_error_safe: NaN input is not allowed",
        )));
    }
    let denom = exact.abs().max(floor);
    if denom.is_zero() {
        return Err(CoreError::DomainError(ErrorContext::new(
            "relative_error_safe: denominator is zero even with floor",
        )));
    }
    Ok((computed - exact).abs() / denom)
}

// ---------------------------------------------------------------------------
// ulp_distance
// ---------------------------------------------------------------------------

/// Compute the ULP (Unit in the Last Place) distance between two `f64` values.
///
/// ULP distance counts the number of representable floating-point values
/// between `a` and `b`.
///
/// # Returns
///
/// - `Ok(distance)` on success.
/// - `Err` if either input is NaN.
///
/// # Notes
///
/// For `f32`, convert to `f64` first or use [`ulp_distance_f32`].
pub fn ulp_distance(a: f64, b: f64) -> CoreResult<u64> {
    if a.is_nan() || b.is_nan() {
        return Err(CoreError::ValueError(ErrorContext::new(
            "ulp_distance: NaN input is not allowed",
        )));
    }
    if a == b {
        return Ok(0);
    }

    let a_bits = a.to_bits() as i64;
    let b_bits = b.to_bits() as i64;

    // For IEEE 754 doubles, the bit representation is ordered except for
    // the sign bit. We need to map negative numbers to a continuous integer
    // space.
    let a_mapped = if a_bits < 0 {
        i64::MIN - a_bits
    } else {
        a_bits
    };
    let b_mapped = if b_bits < 0 {
        i64::MIN - b_bits
    } else {
        b_bits
    };

    Ok((a_mapped - b_mapped).unsigned_abs())
}

/// Compute ULP distance for `f32` values.
pub fn ulp_distance_f32(a: f32, b: f32) -> CoreResult<u32> {
    if a.is_nan() || b.is_nan() {
        return Err(CoreError::ValueError(ErrorContext::new(
            "ulp_distance_f32: NaN input is not allowed",
        )));
    }
    if a == b {
        return Ok(0);
    }

    let a_bits = a.to_bits() as i32;
    let b_bits = b.to_bits() as i32;

    let a_mapped = if a_bits < 0 {
        i32::MIN - a_bits
    } else {
        a_bits
    };
    let b_mapped = if b_bits < 0 {
        i32::MIN - b_bits
    } else {
        b_bits
    };

    Ok((a_mapped - b_mapped).unsigned_abs())
}

// ---------------------------------------------------------------------------
// is_finite_and_positive
// ---------------------------------------------------------------------------

/// Check that a value is both finite and strictly positive (`> 0`).
///
/// Returns `false` for NaN, infinity, zero, and negative values.
pub fn is_finite_and_positive<T: Float>(x: T) -> bool {
    x.is_finite() && x > T::zero()
}

/// Check that a value is finite and non-negative (`>= 0`).
pub fn is_finite_and_non_negative<T: Float>(x: T) -> bool {
    x.is_finite() && x >= T::zero()
}

// ---------------------------------------------------------------------------
// clamp_to_range
// ---------------------------------------------------------------------------

/// Clamp `x` to the range `[min, max]`, returning an error if `x` is NaN or if
/// `min > max`.
///
/// Unlike `f64::clamp()`, this function does not panic on NaN but returns a
/// proper error.
///
/// # Errors
///
/// - `CoreError::ValueError` if `x` is NaN.
/// - `CoreError::DomainError` if `min > max`.
pub fn clamp_to_range<T: Float + Debug>(x: T, min: T, max: T) -> CoreResult<T> {
    if x.is_nan() {
        return Err(CoreError::ValueError(ErrorContext::new(
            "clamp_to_range: input is NaN",
        )));
    }
    if min > max {
        return Err(CoreError::DomainError(ErrorContext::new(format!(
            "clamp_to_range: min ({min:?}) > max ({max:?})"
        ))));
    }
    if x < min {
        Ok(min)
    } else if x > max {
        Ok(max)
    } else {
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Check if two slices of floats are element-wise approximately equal.
pub fn slices_approx_eq<T: Float>(a: &[T], b: &[T], tol: T) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(&x, &y)| approx_eq(x, y, tol))
}

/// Compute the maximum absolute difference between two slices.
///
/// Returns `Ok(max_diff)` or `Err` if the slices have different lengths.
pub fn max_abs_difference<T: Float>(a: &[T], b: &[T]) -> CoreResult<T> {
    if a.len() != b.len() {
        return Err(CoreError::DimensionError(ErrorContext::new(format!(
            "max_abs_difference: length mismatch ({} vs {})",
            a.len(),
            b.len()
        ))));
    }
    if a.is_empty() {
        return Ok(T::zero());
    }
    let max = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(T::zero(), |acc, d| if d > acc { d } else { acc });
    Ok(max)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- approx_eq tests ----

    #[test]
    fn test_approx_eq_basic() {
        assert!(approx_eq(1.0_f64, 1.0 + 1e-11, 1e-10));
        assert!(!approx_eq(1.0_f64, 1.0 + 1e-9, 1e-10));
    }

    #[test]
    fn test_approx_eq_nan() {
        assert!(!approx_eq(f64::NAN, 1.0, 1e-10));
        assert!(!approx_eq(1.0, f64::NAN, 1e-10));
        assert!(!approx_eq(f64::NAN, f64::NAN, 1e-10));
    }

    #[test]
    fn test_approx_eq_infinity() {
        assert!(approx_eq(f64::INFINITY, f64::INFINITY, 1e-10));
        assert!(approx_eq(f64::NEG_INFINITY, f64::NEG_INFINITY, 1e-10));
        assert!(!approx_eq(f64::INFINITY, f64::NEG_INFINITY, 1e-10));
    }

    #[test]
    fn test_approx_eq_zero() {
        assert!(approx_eq(0.0_f64, 0.0, 0.0));
        assert!(approx_eq(0.0_f64, 1e-16, 1e-15));
    }

    #[test]
    fn test_approx_eq_f32() {
        assert!(approx_eq(1.0_f32, 1.0 + 1e-6, 1e-5));
        assert!(!approx_eq(1.0_f32, 1.0 + 1e-4, 1e-5));
    }

    #[test]
    fn test_approx_eq_relative_basic() {
        assert!(approx_eq_relative(100.0_f64, 100.001, 1e-4, 1e-10));
        assert!(!approx_eq_relative(100.0_f64, 101.0, 1e-4, 1e-10));
    }

    // ---- relative_error tests ----

    #[test]
    fn test_relative_error_basic() {
        let err = relative_error(1.01_f64, 1.0).expect("should succeed");
        assert!(approx_eq(err, 0.01, 1e-14));
    }

    #[test]
    fn test_relative_error_exact() {
        let err = relative_error(3.14_f64, 3.14).expect("should succeed");
        assert_eq!(err, 0.0);
    }

    #[test]
    fn test_relative_error_zero_exact() {
        let result = relative_error(1.0_f64, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_relative_error_nan() {
        assert!(relative_error(f64::NAN, 1.0).is_err());
        assert!(relative_error(1.0, f64::NAN).is_err());
    }

    #[test]
    fn test_relative_error_safe_near_zero() {
        let err = relative_error_safe(0.001_f64, 0.0, 1.0).expect("should succeed");
        assert!(approx_eq(err, 0.001, 1e-14));
    }

    // ---- ulp_distance tests ----

    #[test]
    fn test_ulp_distance_same() {
        assert_eq!(ulp_distance(1.0, 1.0).expect("should succeed"), 0);
    }

    #[test]
    fn test_ulp_distance_adjacent() {
        let a = 1.0_f64;
        let b = f64::from_bits(a.to_bits() + 1);
        assert_eq!(ulp_distance(a, b).expect("should succeed"), 1);
    }

    #[test]
    fn test_ulp_distance_symmetric() {
        let a = 1.0_f64;
        let b = 1.0 + f64::EPSILON;
        let d1 = ulp_distance(a, b).expect("should succeed");
        let d2 = ulp_distance(b, a).expect("should succeed");
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_ulp_distance_nan() {
        assert!(ulp_distance(f64::NAN, 1.0).is_err());
    }

    #[test]
    fn test_ulp_distance_across_zero() {
        // Distance from a small negative to a small positive
        let d = ulp_distance(-0.0_f64, 0.0).expect("should succeed");
        // -0.0 and 0.0 should be 0 ULPs apart because they are equal
        assert_eq!(d, 0);
    }

    #[test]
    fn test_ulp_distance_f32_basic() {
        let a = 1.0_f32;
        let b = f32::from_bits(a.to_bits() + 1);
        assert_eq!(ulp_distance_f32(a, b).expect("should succeed"), 1);
    }

    #[test]
    fn test_ulp_distance_f32_nan() {
        assert!(ulp_distance_f32(f32::NAN, 1.0).is_err());
    }

    // ---- is_finite_and_positive tests ----

    #[test]
    fn test_is_finite_and_positive_basic() {
        assert!(is_finite_and_positive(1.0_f64));
        assert!(is_finite_and_positive(f64::MIN_POSITIVE));
        assert!(!is_finite_and_positive(0.0_f64));
        assert!(!is_finite_and_positive(-1.0_f64));
        assert!(!is_finite_and_positive(f64::INFINITY));
        assert!(!is_finite_and_positive(f64::NAN));
    }

    #[test]
    fn test_is_finite_and_positive_f32() {
        assert!(is_finite_and_positive(0.001_f32));
        assert!(!is_finite_and_positive(f32::NEG_INFINITY));
    }

    #[test]
    fn test_is_finite_and_non_negative() {
        assert!(is_finite_and_non_negative(0.0_f64));
        assert!(is_finite_and_non_negative(1.0_f64));
        assert!(!is_finite_and_non_negative(-0.001_f64));
        assert!(!is_finite_and_non_negative(f64::NAN));
    }

    // ---- clamp_to_range tests ----

    #[test]
    fn test_clamp_to_range_normal() {
        assert_eq!(clamp_to_range(5.0_f64, 0.0, 10.0).expect("ok"), 5.0);
    }

    #[test]
    fn test_clamp_to_range_below() {
        assert_eq!(clamp_to_range(-5.0_f64, 0.0, 10.0).expect("ok"), 0.0);
    }

    #[test]
    fn test_clamp_to_range_above() {
        assert_eq!(clamp_to_range(15.0_f64, 0.0, 10.0).expect("ok"), 10.0);
    }

    #[test]
    fn test_clamp_to_range_nan() {
        let result = clamp_to_range(f64::NAN, 0.0, 10.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_clamp_to_range_inverted() {
        let result = clamp_to_range(5.0_f64, 10.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_clamp_to_range_exact_boundary() {
        assert_eq!(clamp_to_range(0.0_f64, 0.0, 10.0).expect("ok"), 0.0);
        assert_eq!(clamp_to_range(10.0_f64, 0.0, 10.0).expect("ok"), 10.0);
    }

    #[test]
    fn test_clamp_to_range_f32() {
        assert_eq!(clamp_to_range(100.0_f32, -1.0, 1.0).expect("ok"), 1.0);
    }

    // ---- MachineConstants tests ----

    #[test]
    fn test_machine_constants_f64() {
        assert_eq!(f64::machine_epsilon(), f64::EPSILON);
        assert_eq!(f64::min_positive_normal(), f64::MIN_POSITIVE);
        assert_eq!(f64::max_finite(), f64::MAX);
        assert_eq!(f64::mantissa_digits(), 53);
        assert_eq!(f64::radix(), 2);
        assert!(f64::min_positive_subnormal() > 0.0);
        assert!(f64::min_positive_subnormal() < f64::MIN_POSITIVE);
    }

    #[test]
    fn test_machine_constants_f32() {
        assert_eq!(f32::machine_epsilon(), f32::EPSILON);
        assert_eq!(f32::min_positive_normal(), f32::MIN_POSITIVE);
        assert_eq!(f32::max_finite(), f32::MAX);
        assert_eq!(f32::mantissa_digits(), 24);
        assert!(f32::min_positive_subnormal() > 0.0);
        assert!(f32::min_positive_subnormal() < f32::MIN_POSITIVE);
    }

    // ---- slices_approx_eq tests ----

    #[test]
    fn test_slices_approx_eq_basic() {
        let a = [1.0_f64, 2.0, 3.0];
        let b = [1.0 + 1e-12, 2.0 - 1e-12, 3.0];
        assert!(slices_approx_eq(&a, &b, 1e-10));
    }

    #[test]
    fn test_slices_approx_eq_different_lengths() {
        let a = [1.0_f64, 2.0];
        let b = [1.0_f64];
        assert!(!slices_approx_eq(&a, &b, 1e-10));
    }

    #[test]
    fn test_slices_approx_eq_not_equal() {
        let a = [1.0_f64, 2.0];
        let b = [1.0_f64, 3.0];
        assert!(!slices_approx_eq(&a, &b, 0.5));
    }

    // ---- max_abs_difference tests ----

    #[test]
    fn test_max_abs_difference_basic() {
        let a = [1.0_f64, 2.0, 3.0];
        let b = [1.1, 2.0, 2.5];
        let d = max_abs_difference(&a, &b).expect("ok");
        assert!(approx_eq(d, 0.5, 1e-14));
    }

    #[test]
    fn test_max_abs_difference_empty() {
        let a: [f64; 0] = [];
        let b: [f64; 0] = [];
        assert_eq!(max_abs_difference(&a, &b).expect("ok"), 0.0);
    }

    #[test]
    fn test_max_abs_difference_length_mismatch() {
        let a = [1.0_f64];
        let b = [1.0_f64, 2.0];
        assert!(max_abs_difference(&a, &b).is_err());
    }
}
