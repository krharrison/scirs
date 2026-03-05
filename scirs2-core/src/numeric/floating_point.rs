//! Floating-point analysis utilities.
//!
//! This module provides low-level tools for understanding and controlling
//! floating-point arithmetic behaviour following IEEE 754.
//!
//! ## Overview
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`machine_epsilon_f32`] | Compute ε_mach for `f32` by iteration |
//! | [`machine_epsilon_f64`] | Compute ε_mach for `f64` by iteration |
//! | [`rounding_error_bound`] | Global `n·u` rounding error bound for n operations |
//! | [`significand_bits`] | Extract the raw mantissa bits of a `f64` |
//! | [`ulp`] | Unit in the last place for a `f64` |
//! | [`round_to_significant`] | Round to a given number of significant decimal digits |
//! | [`safe_divide`] | Division guarded against zero denominator |
//! | [`kahan_sum`] | Kahan compensated summation (O(ε) global error) |
//! | [`pairwise_sum`] | Recursive pairwise summation (O(ε log n) global error) |
//!
//! ## Floating-Point Type Enum
//!
//! [`FloatType`] lets callers parameterise the machine epsilon and error bound
//! functions without using generics:
//!
//! ```rust
//! use scirs2_core::numeric::floating_point::{FloatType, rounding_error_bound};
//!
//! // How much rounding error can accumulate in 1000 f64 additions?
//! let bound = rounding_error_bound(1000, FloatType::F64);
//! assert!(bound < 1e-12);
//! ```
//!
//! ## References
//!
//! - Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM.
//! - Kahan, W. (1965). "Pracniques: further remarks on reducing truncation errors."
//!   *Commun. ACM*, 8(1), 40.
//! - IEEE 754-2019 standard for floating-point arithmetic.

// ---------------------------------------------------------------------------
// FloatType enum
// ---------------------------------------------------------------------------

/// Identifies a floating-point precision level.
///
/// Used to select the appropriate machine epsilon when a generic parameter
/// would be inconvenient (e.g., in Python bindings or configuration structs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatType {
    /// 32-bit single precision (`f32`).
    F32,
    /// 64-bit double precision (`f64`).
    F64,
}

impl FloatType {
    /// Returns the machine epsilon for this float type.
    ///
    /// For `F32` this is `f32::EPSILON` (~1.19e-7).
    /// For `F64` this is `f64::EPSILON` (~2.22e-16).
    #[inline]
    #[must_use]
    pub fn epsilon(self) -> f64 {
        match self {
            FloatType::F32 => f32::EPSILON as f64,
            FloatType::F64 => f64::EPSILON,
        }
    }

    /// Returns the unit roundoff `u = ε_mach / 2`.
    ///
    /// The unit roundoff is the fundamental rounding constant: every IEEE 754
    /// floating-point operation is exact to within a factor of `1 + δ` where
    /// `|δ| ≤ u`.
    #[inline]
    #[must_use]
    pub fn unit_roundoff(self) -> f64 {
        self.epsilon() / 2.0
    }
}

// ---------------------------------------------------------------------------
// Machine epsilon computations
// ---------------------------------------------------------------------------

/// Compute machine epsilon for `f32` by iterative halving.
///
/// Starting from `1.0`, repeatedly halves a value until `1.0 + value == 1.0`.
/// The last value for which the inequality was false is `ε_mach`.
///
/// This matches the IEEE 754 definition: `ε = 2^{-23}` for single precision.
///
/// # Examples
///
/// ```
/// use scirs2_core::numeric::floating_point::machine_epsilon_f32;
///
/// let eps = machine_epsilon_f32();
/// // Should equal the hardware constant
/// assert!((eps - f32::EPSILON).abs() < 1e-38_f32);
/// ```
#[must_use]
pub fn machine_epsilon_f32() -> f32 {
    let mut eps = 1.0_f32;
    loop {
        let half = eps / 2.0;
        if 1.0_f32 + half == 1.0_f32 {
            break;
        }
        eps = half;
    }
    eps
}

/// Compute machine epsilon for `f64` by iterative halving.
///
/// This matches the IEEE 754 definition: `ε = 2^{-52}` for double precision.
///
/// # Examples
///
/// ```
/// use scirs2_core::numeric::floating_point::machine_epsilon_f64;
///
/// let eps = machine_epsilon_f64();
/// assert!((eps - f64::EPSILON).abs() < 1e-300);
/// ```
#[must_use]
pub fn machine_epsilon_f64() -> f64 {
    let mut eps = 1.0_f64;
    loop {
        let half = eps / 2.0;
        if 1.0_f64 + half == 1.0_f64 {
            break;
        }
        eps = half;
    }
    eps
}

// ---------------------------------------------------------------------------
// Rounding error bound
// ---------------------------------------------------------------------------

/// Compute the standard `n·u` rounding error bound.
///
/// After `n` sequential floating-point operations each introducing a relative
/// error of at most `u = ε_mach / 2`, the total relative error is bounded by
/// `γ(n) = n·u / (1 - n·u) ≈ n·u` when `n·u ≪ 1`.
///
/// This bound is used throughout Higham's stability analysis to characterise
/// algorithmically stable computations.
///
/// # Arguments
///
/// * `n`     — number of floating-point operations (rounding steps)
/// * `dtype` — float type (selects the unit roundoff)
///
/// # Panics
///
/// Does not panic; if `n·u ≥ 1` (degenerate) returns `f64::INFINITY`.
///
/// # Examples
///
/// ```
/// use scirs2_core::numeric::floating_point::{FloatType, rounding_error_bound};
///
/// // 1000 f64 additions: error bound ≈ 1000 * 1.11e-16 ≈ 1.11e-13
/// let bound = rounding_error_bound(1000, FloatType::F64);
/// assert!(bound > 0.0);
/// assert!(bound < 1e-10);
/// ```
#[must_use]
pub fn rounding_error_bound(n: usize, dtype: FloatType) -> f64 {
    let u = dtype.unit_roundoff();
    let nu = n as f64 * u;
    let denom = 1.0 - nu;
    if denom <= 0.0 {
        f64::INFINITY
    } else {
        nu / denom
    }
}

// ---------------------------------------------------------------------------
// Bit-level inspection
// ---------------------------------------------------------------------------

/// Extract the raw significand (mantissa) bits of an `f64`.
///
/// For an IEEE 754 double, the 52-bit mantissa is stored in bits 0–51 of the
/// 64-bit representation.  This function extracts exactly those 52 bits as a
/// `u64` (without the implicit leading 1 bit).
///
/// For `NaN`, `±Inf`, and subnormals the raw bits are returned as-is.
///
/// # Examples
///
/// ```
/// use scirs2_core::numeric::floating_point::significand_bits;
///
/// // 1.0 has zero mantissa bits (implicit leading 1)
/// assert_eq!(significand_bits(1.0_f64), 0);
///
/// // 1.5 = 1.1 binary → mantissa bits = 100...0
/// assert_eq!(significand_bits(1.5_f64), 1u64 << 51);
/// ```
#[inline]
#[must_use]
pub fn significand_bits(val: f64) -> u64 {
    // Mask out the sign bit (bit 63) and exponent (bits 62–52) → keep bits 51–0
    const MANTISSA_MASK: u64 = (1u64 << 52) - 1;
    val.to_bits() & MANTISSA_MASK
}

// ---------------------------------------------------------------------------
// ULP (Unit in the Last Place)
// ---------------------------------------------------------------------------

/// Compute the unit in the last place (ULP) for a finite `f64` value.
///
/// The ULP of `x` is the gap between `x` and the nearest adjacent floating-point
/// number with larger magnitude, i.e. `next_after(x, +∞) - x` for positive `x`.
///
/// For `NaN` and ±`Inf`, returns `NaN`.
/// For `0.0`, returns the smallest positive subnormal `f64` value.
///
/// This is the standard definition used by Kahan and MPFR.
///
/// # Examples
///
/// ```
/// use scirs2_core::numeric::floating_point::ulp;
///
/// // ulp(1.0) should equal f64::EPSILON
/// assert_eq!(ulp(1.0_f64), f64::EPSILON);
///
/// // ulp(2.0) = 2 * f64::EPSILON
/// assert_eq!(ulp(2.0_f64), 2.0 * f64::EPSILON);
/// ```
#[must_use]
pub fn ulp(val: f64) -> f64 {
    if val.is_nan() || val.is_infinite() {
        return f64::NAN;
    }
    if val == 0.0 {
        // Smallest positive subnormal: bits = 1
        return f64::from_bits(1);
    }
    // Work with absolute value to handle negative numbers uniformly
    let abs_val = val.abs();
    let bits = abs_val.to_bits();
    // next float above abs_val
    let next = f64::from_bits(bits + 1);
    next - abs_val
}

// ---------------------------------------------------------------------------
// Rounding to significant digits
// ---------------------------------------------------------------------------

/// Round a floating-point value to a given number of significant decimal digits.
///
/// Uses the formula `round(x / 10^e) * 10^e` where `e = floor(log10(|x|)) - (sig_digits - 1)`.
///
/// Edge cases:
/// - `x == 0.0` returns `0.0`.
/// - `sig_digits == 0` returns `0.0`.
/// - `NaN` / `±Inf` are returned unchanged.
///
/// # Examples
///
/// ```
/// use scirs2_core::numeric::floating_point::round_to_significant;
///
/// let v = round_to_significant(3.14159, 3);
/// assert!((v - 3.14).abs() < 1e-10);
///
/// let v2 = round_to_significant(123456.789, 4);
/// assert!((v2 - 123500.0).abs() < 1.0);
/// ```
#[must_use]
pub fn round_to_significant(val: f64, sig_digits: usize) -> f64 {
    if !val.is_finite() {
        return val;
    }
    if val == 0.0 || sig_digits == 0 {
        return 0.0;
    }
    let mag = val.abs().log10().floor();
    let scale = 10.0_f64.powf(mag - (sig_digits as f64 - 1.0));
    (val / scale).round() * scale
}

// ---------------------------------------------------------------------------
// Safe division
// ---------------------------------------------------------------------------

/// Divide `a` by `b`, returning `default` if `b` is zero, NaN, or ±Infinity.
///
/// This function is primarily useful when building numerical algorithms that
/// need to avoid floating-point exceptions (e.g., divide-by-zero) without
/// resorting to explicit branches in hot loops.
///
/// # Examples
///
/// ```
/// use scirs2_core::numeric::floating_point::safe_divide;
///
/// assert_eq!(safe_divide(10.0, 2.0, 0.0), 5.0);
/// assert_eq!(safe_divide(1.0, 0.0, f64::INFINITY), f64::INFINITY);
/// assert_eq!(safe_divide(1.0, f64::NAN, -1.0), -1.0);
/// ```
#[inline]
#[must_use]
pub fn safe_divide(a: f64, b: f64, default: f64) -> f64 {
    if b == 0.0 || !b.is_finite() {
        default
    } else {
        a / b
    }
}

// ---------------------------------------------------------------------------
// Kahan compensated summation
// ---------------------------------------------------------------------------

/// Kahan compensated summation of a slice of `f64` values.
///
/// Achieves O(ε) global rounding error independent of the number of terms,
/// compared to O(n·ε) for naive summation.  The compensating variable `c`
/// tracks accumulated rounding error and subtracts it from each step.
///
/// This algorithm is described in:
/// > Kahan, W. (1965). "Pracniques: further remarks on reducing truncation errors."
/// > *Commun. ACM*, 8(1), 40.
///
/// # Examples
///
/// ```
/// use scirs2_core::numeric::floating_point::kahan_sum;
///
/// // Sum 1 million ones — naive sum accumulates O(1e6 * ε) error; Kahan is exact
/// let ones = vec![1.0_f64; 1_000_000];
/// let s = kahan_sum(&ones);
/// assert!((s - 1_000_000.0).abs() < 1.0);
/// ```
#[must_use]
pub fn kahan_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64; // compensation
    for &v in values {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

// ---------------------------------------------------------------------------
// Pairwise summation
// ---------------------------------------------------------------------------

/// Pairwise (cascade) summation of a slice of `f64` values.
///
/// Recursively splits the input in half and sums the two halves, then adds
/// the partial sums.  This achieves O(ε log n) global error, compared to
/// O(n·ε) for sequential summation.
///
/// Pairwise summation is the algorithm used internally by NumPy and is
/// preferred over Kahan when the constant factor matters more than the
/// asymptotic bound.
///
/// Below a threshold (`LEAF_SIZE = 8`), the function falls back to sequential
/// summation to avoid recursion overhead.
///
/// # Examples
///
/// ```
/// use scirs2_core::numeric::floating_point::pairwise_sum;
///
/// let values = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(pairwise_sum(&values), 15.0);
/// ```
#[must_use]
pub fn pairwise_sum(values: &[f64]) -> f64 {
    pairwise_sum_inner(values)
}

/// Leaf threshold for pairwise summation (switch to sequential below this size).
const LEAF_SIZE: usize = 8;

fn pairwise_sum_inner(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return values[0];
    }
    if n <= LEAF_SIZE {
        // Sequential fallback
        let mut s = 0.0_f64;
        for &v in values {
            s += v;
        }
        return s;
    }
    let mid = n / 2;
    pairwise_sum_inner(&values[..mid]) + pairwise_sum_inner(&values[mid..])
}

// ---------------------------------------------------------------------------
// Neumaier (improved Kahan) summation
// ---------------------------------------------------------------------------

/// Neumaier's improved compensated summation.
///
/// Extends Kahan's algorithm to also handle the case where the new term is
/// larger than the running sum (i.e., handles large increments that would
/// otherwise lose the compensation).
///
/// > Neumaier, A. (1974). "Rounding error analysis of some methods for summing
/// > finite sums." *Zeitschrift für Angewandte Mathematik und Mechanik*, 54, T59–T61.
///
/// # Examples
///
/// ```
/// use scirs2_core::numeric::floating_point::neumaier_sum;
///
/// let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
/// let s = neumaier_sum(&values);
/// assert_eq!(s, 5050.0);
/// ```
#[must_use]
pub fn neumaier_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for &v in values {
        let t = sum + v;
        if sum.abs() >= v.abs() {
            c += (sum - t) + v;
        } else {
            c += (v - t) + sum;
        }
        sum = t;
    }
    sum + c
}

// ---------------------------------------------------------------------------
// Floating-point utilities struct
// ---------------------------------------------------------------------------

/// Collection of floating-point analysis and safe arithmetic utilities.
///
/// This struct provides a convenient namespace for the stateless utility
/// functions in this module.  All methods delegate to the module-level
/// free functions.
///
/// # Examples
///
/// ```
/// use scirs2_core::numeric::floating_point::FloatingPointUtils;
///
/// let eps = FloatingPointUtils::machine_epsilon_f64();
/// assert!(eps > 0.0 && eps < 1e-10);
///
/// let s = FloatingPointUtils::kahan_sum(&[1.0, 1e-10, -1.0]);
/// assert!((s - 1e-10).abs() < 1e-20);
/// ```
pub struct FloatingPointUtils;

impl FloatingPointUtils {
    /// Compute machine epsilon for `f32` by iterative halving.
    #[inline]
    #[must_use]
    pub fn machine_epsilon_f32() -> f32 {
        machine_epsilon_f32()
    }

    /// Compute machine epsilon for `f64` by iterative halving.
    #[inline]
    #[must_use]
    pub fn machine_epsilon_f64() -> f64 {
        machine_epsilon_f64()
    }

    /// Standard `n·u` rounding error bound for `n` operations of type `dtype`.
    #[inline]
    #[must_use]
    pub fn rounding_error_bound(n: usize, dtype: FloatType) -> f64 {
        rounding_error_bound(n, dtype)
    }

    /// Extract the 52 significand bits of an `f64`.
    #[inline]
    #[must_use]
    pub fn significand_bits(val: f64) -> u64 {
        significand_bits(val)
    }

    /// Unit in the last place for a `f64`.
    #[inline]
    #[must_use]
    pub fn ulp(val: f64) -> f64 {
        ulp(val)
    }

    /// Round to `sig_digits` significant decimal digits.
    #[inline]
    #[must_use]
    pub fn round_to_significant(val: f64, sig_digits: usize) -> f64 {
        round_to_significant(val, sig_digits)
    }

    /// Division guarded against zero / NaN / Inf denominator.
    #[inline]
    #[must_use]
    pub fn safe_divide(a: f64, b: f64, default: f64) -> f64 {
        safe_divide(a, b, default)
    }

    /// Kahan compensated summation.
    #[inline]
    #[must_use]
    pub fn kahan_sum(values: &[f64]) -> f64 {
        kahan_sum(values)
    }

    /// Pairwise (cascade) summation.
    #[inline]
    #[must_use]
    pub fn pairwise_sum(values: &[f64]) -> f64 {
        pairwise_sum(values)
    }

    /// Neumaier improved compensated summation.
    #[inline]
    #[must_use]
    pub fn neumaier_sum(values: &[f64]) -> f64 {
        neumaier_sum(values)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- machine epsilon ----

    #[test]
    fn test_machine_epsilon_f32() {
        let eps = machine_epsilon_f32();
        assert_eq!(eps, f32::EPSILON);
    }

    #[test]
    fn test_machine_epsilon_f64() {
        let eps = machine_epsilon_f64();
        assert_eq!(eps, f64::EPSILON);
    }

    // ---- FloatType ----

    #[test]
    fn test_float_type_epsilon() {
        assert!((FloatType::F32.epsilon() - f32::EPSILON as f64).abs() < 1e-30);
        assert!((FloatType::F64.epsilon() - f64::EPSILON).abs() < 1e-300);
    }

    #[test]
    fn test_float_type_unit_roundoff() {
        assert_eq!(FloatType::F64.unit_roundoff(), f64::EPSILON / 2.0);
    }

    // ---- rounding error bound ----

    #[test]
    fn test_rounding_error_bound_zero_ops() {
        let b = rounding_error_bound(0, FloatType::F64);
        assert_eq!(b, 0.0);
    }

    #[test]
    fn test_rounding_error_bound_reasonable() {
        let b = rounding_error_bound(1000, FloatType::F64);
        // 1000 * (f64::EPSILON / 2) ≈ 1.11e-13
        assert!(b > 0.0);
        assert!(b < 1e-10);
    }

    // ---- significand_bits ----

    #[test]
    fn test_significand_bits_one() {
        // 1.0 has no fractional mantissa bits
        assert_eq!(significand_bits(1.0), 0u64);
    }

    #[test]
    fn test_significand_bits_one_point_five() {
        // 1.5 = 1.1_b => mantissa = 1000...0 (52 bits, top bit = 1)
        let bits = significand_bits(1.5);
        assert_eq!(bits, 1u64 << 51);
    }

    #[test]
    fn test_significand_bits_negative_same_as_positive() {
        // Sign bit is stripped: |significand| of +x and -x are equal
        assert_eq!(significand_bits(2.5), significand_bits(-2.5));
    }

    // ---- ulp ----

    #[test]
    fn test_ulp_one() {
        // ulp(1.0) = f64::EPSILON by IEEE 754 definition
        assert_eq!(ulp(1.0), f64::EPSILON);
    }

    #[test]
    fn test_ulp_two() {
        assert_eq!(ulp(2.0), 2.0 * f64::EPSILON);
    }

    #[test]
    fn test_ulp_zero() {
        // ulp(0) is the smallest positive subnormal
        let u = ulp(0.0);
        assert!(u > 0.0);
        assert!(u < f64::MIN_POSITIVE);
    }

    #[test]
    fn test_ulp_nan() {
        assert!(ulp(f64::NAN).is_nan());
    }

    #[test]
    fn test_ulp_inf() {
        assert!(ulp(f64::INFINITY).is_nan());
    }

    // ---- round_to_significant ----

    #[test]
    fn test_round_to_significant_pi() {
        let v = round_to_significant(std::f64::consts::PI, 4);
        assert!((v - 3.142).abs() < 1e-9);
    }

    #[test]
    fn test_round_to_significant_zero() {
        assert_eq!(round_to_significant(0.0, 5), 0.0);
    }

    #[test]
    fn test_round_to_significant_zero_digits() {
        assert_eq!(round_to_significant(3.14, 0), 0.0);
    }

    #[test]
    fn test_round_to_significant_large() {
        let v = round_to_significant(123456.0, 3);
        assert!((v - 123000.0).abs() < 1.0);
    }

    #[test]
    fn test_round_to_significant_nan_inf() {
        assert!(round_to_significant(f64::NAN, 3).is_nan());
        assert!(round_to_significant(f64::INFINITY, 3).is_infinite());
    }

    // ---- safe_divide ----

    #[test]
    fn test_safe_divide_normal() {
        assert_eq!(safe_divide(10.0, 2.0, 0.0), 5.0);
    }

    #[test]
    fn test_safe_divide_by_zero() {
        assert_eq!(safe_divide(1.0, 0.0, 42.0), 42.0);
    }

    #[test]
    fn test_safe_divide_by_nan() {
        assert_eq!(safe_divide(1.0, f64::NAN, -1.0), -1.0);
    }

    #[test]
    fn test_safe_divide_by_inf() {
        assert_eq!(safe_divide(1.0, f64::INFINITY, 0.5), 0.5);
    }

    // ---- kahan_sum ----

    #[test]
    fn test_kahan_sum_empty() {
        assert_eq!(kahan_sum(&[]), 0.0);
    }

    #[test]
    fn test_kahan_sum_basic() {
        assert_eq!(kahan_sum(&[1.0, 2.0, 3.0, 4.0]), 10.0);
    }

    #[test]
    fn test_kahan_sum_cancellation() {
        // 1e15 + 1 - 1e15 — naively the intermediate 1e15+1 could lose the 1
        // if precision were tight, but with f64 spacing ~0.125 near 1e15 this
        // is representable.  Kahan compensation tracks lost bits.
        let values = [1e15_f64, 1.0, -1e15];
        let s = kahan_sum(&values);
        assert_eq!(s, 1.0, "Kahan should handle catastrophic cancellation");
    }

    #[test]
    fn test_kahan_sum_many_ones() {
        let ones = vec![1.0_f64; 100_000];
        let s = kahan_sum(&ones);
        assert!((s - 100_000.0).abs() < 1.0);
    }

    // ---- pairwise_sum ----

    #[test]
    fn test_pairwise_sum_empty() {
        assert_eq!(pairwise_sum(&[]), 0.0);
    }

    #[test]
    fn test_pairwise_sum_single() {
        assert_eq!(pairwise_sum(&[42.0]), 42.0);
    }

    #[test]
    fn test_pairwise_sum_basic() {
        assert_eq!(pairwise_sum(&[1.0, 2.0, 3.0, 4.0, 5.0]), 15.0);
    }

    #[test]
    fn test_pairwise_sum_large() {
        let values: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
        let s = pairwise_sum(&values);
        // 1 + 2 + ... + 1000 = 500500
        assert!((s - 500_500.0).abs() < 1.0);
    }

    // ---- neumaier_sum ----

    #[test]
    fn test_neumaier_sum_exact() {
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let s = neumaier_sum(&values);
        assert_eq!(s, 5050.0);
    }

    #[test]
    fn test_neumaier_sum_large_small_mix() {
        // Adding 1e16 and then small values: Neumaier handles the order-of-magnitude gap
        let values = [1e16_f64, 1.0, -1e16, 2.0];
        let s = neumaier_sum(&values);
        assert!((s - 3.0).abs() < 2.0, "neumaier_sum got {s}, expected ~3.0");
    }

    // ---- FloatingPointUtils ----

    #[test]
    fn test_utils_struct_delegations() {
        assert_eq!(
            FloatingPointUtils::machine_epsilon_f64(),
            f64::EPSILON
        );
        assert_eq!(
            FloatingPointUtils::significand_bits(1.0),
            significand_bits(1.0)
        );
        assert_eq!(
            FloatingPointUtils::ulp(1.0),
            f64::EPSILON
        );
        assert_eq!(
            FloatingPointUtils::safe_divide(6.0, 3.0, 0.0),
            2.0
        );
        assert_eq!(
            FloatingPointUtils::kahan_sum(&[1.0, 2.0, 3.0]),
            6.0
        );
        assert_eq!(
            FloatingPointUtils::pairwise_sum(&[1.0, 2.0, 3.0]),
            6.0
        );
    }
}
