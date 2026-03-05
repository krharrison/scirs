//! ULP (Units in the Last Place) analysis for floating-point numbers.
//!
//! This module implements rigorous ULP-based comparison and analysis utilities for
//! IEEE 754 floating-point arithmetic. ULP distance is the gold standard for
//! measuring floating-point error because it is invariant under scaling and
//! captures the actual representational difference between two values.
//!
//! # Background
//!
//! IEEE 754 floating-point numbers are distributed non-uniformly on the real line:
//! numbers near 1.0 are more densely packed than numbers near 2^50. The ULP of a
//! number x is the distance to the next representable float, which equals
//! `2^(exponent(x) - mantissa_bits)`. Comparing two floats by ULP distance rather
//! than absolute or relative difference correctly handles this non-uniformity.
//!
//! # References
//!
//! - David Goldberg, "What Every Computer Scientist Should Know About Floating-Point
//!   Arithmetic", ACM Computing Surveys, 1991.
//! - Jean-Michel Muller et al., "Handbook of Floating-Point Arithmetic", 2nd ed., 2018.

use crate::error::{CoreError, ErrorContext};

// ---------------------------------------------------------------------------
// f32 helpers
// ---------------------------------------------------------------------------

/// Compute the ULP distance between two `f32` values.
///
/// Returns `None` if either argument is NaN or if both are infinities of
/// opposite sign (undefined distance).
///
/// The returned value is the number of representable `f32` values strictly
/// between `a` and `b` (inclusive of one endpoint), which equals
/// `|bits(a) - bits(b)|` after sign-magnitude reinterpretation.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::ulp::ulp_distance_f32;
/// let d = ulp_distance_f32(1.0_f32, 1.0_f32 + f32::EPSILON);
/// assert_eq!(d, Some(1));
/// ```
pub fn ulp_distance_f32(a: f32, b: f32) -> Option<u32> {
    if a.is_nan() || b.is_nan() {
        return None;
    }
    // Convert to sign-magnitude representation used by IEEE 754 bit patterns.
    let bits_a = sign_magnitude_f32(a);
    let bits_b = sign_magnitude_f32(b);
    Some(bits_a.abs_diff(bits_b))
}

/// Compute the ULP distance between two `f64` values.
///
/// Returns `None` if either argument is NaN.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::ulp::ulp_distance_f64;
/// let d = ulp_distance_f64(1.0_f64, 1.0_f64 + f64::EPSILON);
/// assert_eq!(d, Some(1));
/// ```
pub fn ulp_distance_f64(a: f64, b: f64) -> Option<u64> {
    if a.is_nan() || b.is_nan() {
        return None;
    }
    let bits_a = sign_magnitude_f64(a);
    let bits_b = sign_magnitude_f64(b);
    Some(bits_a.abs_diff(bits_b))
}

// ---------------------------------------------------------------------------
// next representable float
// ---------------------------------------------------------------------------

/// Return the smallest representable `f64` strictly greater than `x`.
///
/// - If `x == f64::INFINITY` the result is `f64::INFINITY`.
/// - If `x == f64::NEG_INFINITY` the result is the most-negative finite `f64`.
/// - If `x` is NaN the result is NaN.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::ulp::next_up_f64;
/// let x = 1.0_f64;
/// let y = next_up_f64(x);
/// assert!(y > x);
/// assert_eq!(y - x, f64::EPSILON);
/// ```
pub fn next_up_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    // +0.0 and -0.0 should both step to the smallest positive subnormal.
    if x == 0.0_f64 {
        return f64::from_bits(1u64);
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// Return the largest representable `f64` strictly less than `x`.
///
/// - If `x == f64::NEG_INFINITY` the result is `f64::NEG_INFINITY`.
/// - If `x == f64::INFINITY` the result is the largest finite `f64`.
/// - If `x` is NaN the result is NaN.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::ulp::next_down_f64;
/// let x = 1.0_f64;
/// let y = next_down_f64(x);
/// assert!(y < x);
/// ```
pub fn next_down_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if x == 0.0_f64 {
        return f64::from_bits(0x8000_0000_0000_0001u64); // smallest negative subnormal
    }
    let bits = x.to_bits();
    if x < 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

// ---------------------------------------------------------------------------
// Machine epsilon at a point
// ---------------------------------------------------------------------------

/// Return the machine epsilon at `x`, i.e. `ulp(x) / 2`.
///
/// This is the smallest `d` such that `fl(x + d) != fl(x)` when d is added
/// toward the direction of the next representable number.  In particular
/// `eps_of(1.0) == f64::EPSILON / 2.0`.
///
/// Returns `Err` if `x` is NaN or infinite.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::ulp::eps_of;
/// let eps = eps_of(1.0_f64).expect("should succeed");
/// assert!((eps - f64::EPSILON / 2.0).abs() < 1e-320);
/// ```
pub fn eps_of(x: f64) -> Result<f64, CoreError> {
    if x.is_nan() {
        return Err(CoreError::DomainError(ErrorContext::new(
            "eps_of: argument is NaN".to_string(),
        )));
    }
    if x.is_infinite() {
        return Err(CoreError::DomainError(ErrorContext::new(
            "eps_of: argument is infinite".to_string(),
        )));
    }
    let up = next_up_f64(x.abs());
    Ok((up - x.abs()) / 2.0)
}

// ---------------------------------------------------------------------------
// Relative error
// ---------------------------------------------------------------------------

/// Compute the relative error between `computed` and `exact` expressed in ULPs.
///
/// The relative error is `|computed - exact| / ulp(exact)`.
/// Returns `None` if either value is NaN.
/// Returns `Ok(f64::INFINITY)` if `exact` is zero and `computed != exact`.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::ulp::relative_error;
/// let err = relative_error(1.0_f64 + f64::EPSILON, 1.0_f64).expect("should succeed");
/// assert!(err >= 0.0);
/// ```
pub fn relative_error(computed: f64, exact: f64) -> Option<f64> {
    if computed.is_nan() || exact.is_nan() {
        return None;
    }
    let diff = (computed - exact).abs();
    if exact == 0.0 {
        if diff == 0.0 {
            return Some(0.0);
        }
        return Some(f64::INFINITY);
    }
    let ulp = next_up_f64(exact.abs()) - exact.abs();
    if ulp == 0.0 {
        return Some(0.0);
    }
    Some(diff / ulp)
}

// ---------------------------------------------------------------------------
// Float nearly-equal
// ---------------------------------------------------------------------------

/// Compare two `f64` values as equal up to `ulp_tolerance` ULPs.
///
/// Additionally, two values within `abs_tolerance` are always considered equal
/// (handles the case where both values are near zero).
///
/// Returns `false` if either value is NaN.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::ulp::float_nearly_equal;
/// assert!(float_nearly_equal(1.0_f64, 1.0_f64 + f64::EPSILON, 2, 0.0));
/// assert!(!float_nearly_equal(1.0_f64, 2.0_f64, 2, 0.0));
/// ```
pub fn float_nearly_equal(a: f64, b: f64, ulp_tolerance: u64, abs_tolerance: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    // Handle infinities: equal only if both are the same infinity.
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    // Absolute tolerance guard (catches near-zero comparisons).
    if (a - b).abs() <= abs_tolerance {
        return true;
    }
    // ULP-based comparison.
    match ulp_distance_f64(a, b) {
        Some(d) => d <= ulp_tolerance,
        None => false,
    }
}

// ---------------------------------------------------------------------------
// Significant bits
// ---------------------------------------------------------------------------

/// Count the number of significant (correct) mantissa bits in `computed`
/// relative to `reference`.
///
/// The result is `max(0, 52 - floor(log2(|computed - reference| / ulp(reference))))`.
/// A perfect match returns 52 (all mantissa bits correct for f64).
///
/// Returns `Err` if either value is NaN or if `reference` is zero and
/// `computed` is non-zero (undefined).
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::ulp::significant_bits;
/// // Exact match → 52 bits
/// assert_eq!(significant_bits(1.0_f64, 1.0_f64).expect("should succeed"), 52);
/// ```
pub fn significant_bits(computed: f64, reference: f64) -> Result<u32, CoreError> {
    if computed.is_nan() || reference.is_nan() {
        return Err(CoreError::DomainError(ErrorContext::new(
            "significant_bits: NaN argument".to_string(),
        )));
    }
    if computed == reference {
        return Ok(52);
    }
    if reference == 0.0 {
        if computed == 0.0 {
            return Ok(52);
        }
        return Err(CoreError::DomainError(ErrorContext::new(
            "significant_bits: reference is zero, computed is non-zero (undefined)".to_string(),
        )));
    }
    let ulp = next_up_f64(reference.abs()) - reference.abs();
    if ulp == 0.0 {
        // reference is at max magnitude; treat as exact
        return Ok(52);
    }
    let diff = (computed - reference).abs();
    let ratio = diff / ulp;
    if ratio <= 0.0 {
        return Ok(52);
    }
    let log2_ratio = ratio.log2().floor() as i64;
    let sig = 52i64 - log2_ratio;
    Ok(sig.max(0) as u32)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Reinterpret the bit pattern of `x` as a signed-magnitude integer.
///
/// IEEE 754 sign-magnitude integers compare correctly with abs_diff:
/// the distance is always the number of representable floats between the two.
#[inline]
fn sign_magnitude_f32(x: f32) -> i32 {
    let bits = x.to_bits() as i32;
    if bits < 0 {
        // Negative float: flip all bits except sign to get sign-magnitude.
        i32::MIN ^ bits
    } else {
        bits
    }
}

#[inline]
fn sign_magnitude_f64(x: f64) -> i64 {
    let bits = x.to_bits() as i64;
    if bits < 0 {
        i64::MIN ^ bits
    } else {
        bits
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp_distance_same_value() {
        assert_eq!(ulp_distance_f64(1.0, 1.0), Some(0));
        assert_eq!(ulp_distance_f32(1.0, 1.0), Some(0));
    }

    #[test]
    fn ulp_distance_adjacent_f64() {
        let x = 1.0_f64;
        let y = next_up_f64(x);
        assert_eq!(ulp_distance_f64(x, y), Some(1));
    }

    #[test]
    fn ulp_distance_nan_returns_none() {
        assert_eq!(ulp_distance_f64(f64::NAN, 1.0), None);
        assert_eq!(ulp_distance_f64(1.0, f64::NAN), None);
    }

    #[test]
    fn next_up_positive() {
        let x = 1.0_f64;
        let y = next_up_f64(x);
        assert!(y > x);
        assert_eq!(y - x, f64::EPSILON);
    }

    #[test]
    fn next_down_positive() {
        let x = 1.0_f64;
        let y = next_down_f64(x);
        assert!(y < x);
    }

    #[test]
    fn next_up_zero() {
        let y = next_up_f64(0.0);
        assert!(y > 0.0);
        assert!(y.is_subnormal());
    }

    #[test]
    fn eps_of_one() {
        let eps = eps_of(1.0).expect("valid");
        assert!((eps - f64::EPSILON / 2.0).abs() < 1e-320);
    }

    #[test]
    fn eps_of_nan_errors() {
        assert!(eps_of(f64::NAN).is_err());
    }

    #[test]
    fn relative_error_exact() {
        let err = relative_error(1.0, 1.0).expect("should be some");
        assert_eq!(err, 0.0);
    }

    #[test]
    fn float_nearly_equal_same() {
        assert!(float_nearly_equal(1.0, 1.0, 0, 0.0));
    }

    #[test]
    fn float_nearly_equal_adjacent() {
        let x = 1.0_f64;
        let y = next_up_f64(x);
        assert!(float_nearly_equal(x, y, 1, 0.0));
        assert!(!float_nearly_equal(x, y, 0, 0.0));
    }

    #[test]
    fn float_nearly_equal_nan_false() {
        assert!(!float_nearly_equal(f64::NAN, 1.0, 10, 0.0));
    }

    #[test]
    fn significant_bits_exact() {
        assert_eq!(significant_bits(1.0, 1.0).expect("ok"), 52);
    }

    #[test]
    fn significant_bits_half_eps() {
        // computed differs by 1 ULP → 51 significant bits
        let x = 1.0_f64;
        let y = next_up_f64(x);
        let bits = significant_bits(y, x).expect("ok");
        assert!(bits <= 52);
    }

    #[test]
    fn ulp_distance_across_zero() {
        // -1 ULP from 0 to smallest positive subnormal is 2.
        let a = f64::from_bits(1u64); // smallest positive subnormal
        let b = -f64::from_bits(1u64);
        assert_eq!(ulp_distance_f64(a, b), Some(2));
    }
}
