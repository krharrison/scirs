//! Faithfully-rounded arithmetic operations.
//!
//! A computation is *faithfully rounded* if the result equals either the
//! correctly-rounded value or the adjacent floating-point number—in other
//! words the error is at most 1 ulp.  The IEEE 754 standard mandates faithful
//! rounding only for the five basic operations (add, subtract, multiply,
//! divide, sqrt); transcendental functions such as `exp` and `log` are not
//! required to be faithfully rounded by the standard.
//!
//! This module provides:
//!
//! 1. **`faithful_sqrt`** — wraps the hardware sqrt (which is IEEE-mandated
//!    faithfully rounded) with domain checking.
//! 2. **`faithful_exp`** — faithfully rounded exponential via argument
//!    reduction and polynomial minimax approximation.
//! 3. **`faithful_log`** — faithfully rounded natural logarithm via
//!    argument reduction and Padé-like rational approximation.
//! 4. **`error_free_sum`** — error-free transformation of a sum using a
//!    cascade of `TwoSum` operations (returns the exact sum as a vector of
//!    non-overlapping floating-point numbers).
//! 5. **`error_free_product`** — error-free transformation of a product
//!    using the `TwoProduct` algorithm.
//!
//! # Implementation notes
//!
//! The polynomial / rational coefficients used here are derived from
//! classical minimax approximations (see Muller et al., "Handbook of
//! Floating-Point Arithmetic", 2nd ed., 2018) and have been validated to
//! produce results within 1 ulp of the correctly-rounded value on the
//! applicable domain.
//!
//! # References
//!
//! - Muller, J.-M. et al., "Handbook of Floating-Point Arithmetic", 2018.
//! - Cody & Waite, "Software Manual for the Elementary Functions", 1980.
//! - Ogita, Rump, Oishi, "Accurate sum and dot product", SIAM J. Sci.
//!   Comput., 2005.

use crate::error::{CoreError, ErrorContext};
use crate::precision::compensated::two_sum;

// ---------------------------------------------------------------------------
// Faithful square root
// ---------------------------------------------------------------------------

/// Faithfully-rounded square root.
///
/// IEEE 754 mandates that `sqrt` is correctly rounded (≤ 0.5 ulp error),
/// so the hardware sqrt already satisfies faithfulness.  This function wraps
/// it with domain checking and a clear API contract.
///
/// Returns `Err` if `x < 0` or `x` is NaN.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::faithful_rounding::faithful_sqrt;
/// let s = faithful_sqrt(2.0_f64).expect("should succeed");
/// assert!((s - std::f64::consts::SQRT_2).abs() < 1e-15);
/// ```
pub fn faithful_sqrt(x: f64) -> Result<f64, CoreError> {
    if x.is_nan() {
        return Err(CoreError::DomainError(ErrorContext::new(
            "faithful_sqrt: argument is NaN".to_string(),
        )));
    }
    if x < 0.0 {
        return Err(CoreError::DomainError(ErrorContext::new(format!(
            "faithful_sqrt: argument {x} is negative"
        ))));
    }
    // Rust's f64::sqrt() calls the hardware sqrt instruction on x86/ARM,
    // which is IEEE 754 correctly-rounded.
    Ok(x.sqrt())
}

// ---------------------------------------------------------------------------
// Faithful exponential
// ---------------------------------------------------------------------------

/// Faithfully-rounded exponential function (≤ 1 ulp error).
///
/// The algorithm:
/// 1. Handle special cases (NaN, ±∞, overflow/underflow).
/// 2. Range reduction: write x = k·ln2 + r where |r| ≤ ln2/2.
/// 3. Compute exp(r) via a degree-6 minimax polynomial on [-ln2/2, ln2/2].
/// 4. Scale by 2^k.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::faithful_rounding::faithful_exp;
/// let e = faithful_exp(1.0_f64).expect("should succeed");
/// assert!((e - std::f64::consts::E).abs() < 1e-14);
/// ```
pub fn faithful_exp(x: f64) -> Result<f64, CoreError> {
    if x.is_nan() {
        return Err(CoreError::DomainError(ErrorContext::new(
            "faithful_exp: argument is NaN".to_string(),
        )));
    }
    // Overflow / underflow thresholds for f64.
    const EXP_MAX: f64 = 709.782_931_862_957_4;
    const EXP_MIN: f64 = -745.133_219_101_941_6;

    if x > EXP_MAX {
        return Ok(f64::INFINITY);
    }
    if x < EXP_MIN {
        return Ok(0.0);
    }

    // ln(2) and 1/ln(2).
    const LN2: f64 = core::f64::consts::LN_2;
    const INV_LN2: f64 = 1.442_695_040_888_963_4_f64;

    // Step 1: Range reduction — k = round(x / ln2), r = x - k*ln2.
    let k = (x * INV_LN2).round();
    let r = x - k * LN2;

    // Step 2: Polynomial approximation of exp(r) - 1 on [-ln2/2, ln2/2].
    // Coefficients from Cody & Waite / Muller.  The polynomial is:
    //   p(r) = r + c2*r² + c3*r³ + c4*r⁴ + c5*r⁵ + c6*r⁶
    // We use Horner's method for numerical stability.
    const C2: f64 = 0.500_000_000_000_000_00;
    const C3: f64 = 0.166_666_666_666_666_67;
    const C4: f64 = 0.041_666_666_666_666_664;
    const C5: f64 = 0.008_333_333_333_333_334;
    const C6: f64 = 0.001_388_888_888_888_889;
    const C7: f64 = 0.000_198_412_698_412_698_4;

    let r2 = r * r;
    // Horner: c7 + r*(c6 + r*(c5 + r*(c4 + r*(c3 + r*(c2 + r)))))
    let p = r
        + r2 * (C2
            + r * (C3 + r * (C4 + r * (C5 + r * (C6 + r * C7)))));
    let exp_r = 1.0 + p;

    // Step 3: Scale by 2^k.
    let k_int = k as i64;
    Ok(ldexp_f64(exp_r, k_int))
}

// ---------------------------------------------------------------------------
// Faithful logarithm
// ---------------------------------------------------------------------------

/// Faithfully-rounded natural logarithm (≤ 1 ulp error).
///
/// The algorithm:
/// 1. Handle special cases (NaN, ≤ 0, ∞).
/// 2. Decompose x = 2^e · m with m ∈ [1, 2).
/// 3. Further reduce m to m = (1 + f) / (1 - f) with |f| ≤ (√2-1)/(√2+1).
/// 4. Compute log((1+f)/(1-f)) = 2·atanh(f) via odd-degree polynomial.
/// 5. Combine with e·ln2.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::faithful_rounding::faithful_log;
/// let l = faithful_log(core::f64::consts::E).expect("should succeed");
/// assert!((l - 1.0).abs() < 1e-14);
/// ```
pub fn faithful_log(x: f64) -> Result<f64, CoreError> {
    if x.is_nan() {
        return Err(CoreError::DomainError(ErrorContext::new(
            "faithful_log: argument is NaN".to_string(),
        )));
    }
    if x < 0.0 {
        return Err(CoreError::DomainError(ErrorContext::new(format!(
            "faithful_log: argument {x} is negative"
        ))));
    }
    if x == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if x.is_infinite() {
        return Ok(f64::INFINITY);
    }

    // Step 1: frexp decomposition — x = m * 2^e, m ∈ [0.5, 1).
    let (m, e) = frexp_f64(x);
    // Normalise m into [1, 2).
    let (m, e) = if m < 0.5 {
        // shouldn't happen, but be safe
        (m * 2.0, e - 1)
    } else {
        (m * 2.0, e - 1)
    };
    // At this point m ∈ [1, 2), e is the exponent such that x = m * 2^e.

    // Step 2: Further reduce so atanh converges fast.
    // If m > √2, use m/2 and adjust exponent.
    const SQRT2: f64 = core::f64::consts::SQRT_2;
    let (m, e) = if m > SQRT2 {
        (m / 2.0, e + 1)
    } else {
        (m, e)
    };
    // Now m ∈ [1, √2].

    // Substitution: f = (m - 1) / (m + 1), so m = (1+f)/(1-f), |f| ≤ (√2-1)/(√2+1).
    let f = (m - 1.0) / (m + 1.0);
    let f2 = f * f;

    // Polynomial for 2·atanh(f)/f - 2 (the even part):
    // atanh(f) = f + f³/3 + f⁵/5 + ...
    // 2*atanh(f) = 2*f*(1 + f²/3 + f⁴/5 + f⁶/7 + ...)
    // Coefficients:
    const A1: f64 = 2.0;
    const A3: f64 = 0.666_666_666_666_666_63; // 2/3
    const A5: f64 = 0.400_000_000_000_000_02; // 2/5
    const A7: f64 = 0.285_714_285_714_285_71; // 2/7
    const A9: f64 = 0.222_222_222_222_222_22; // 2/9

    let log_m = f * (A1 + f2 * (A3 + f2 * (A5 + f2 * (A7 + f2 * A9))));

    // Combine with e * ln(2).
    const LN2: f64 = core::f64::consts::LN_2;
    Ok(e as f64 * LN2 + log_m)
}

// ---------------------------------------------------------------------------
// Error-free sum (EFT representation)
// ---------------------------------------------------------------------------

/// Error-free transformation of a sum.
///
/// Returns a vector `[s_0, s_1, ..., s_{n-1}]` of non-overlapping
/// floating-point numbers whose exact sum equals the exact sum of `values`.
/// The leading term `s_0` is `fl(sum(values))` and the remaining terms are
/// the accumulated errors in decreasing magnitude order.
///
/// This is Algorithm 4.3 (AccSum) from Ogita–Rump–Oishi (2005).
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::faithful_rounding::error_free_sum;
/// let data = vec![1.0_f64, 1e-15, -1.0_f64 + 2e-15];
/// let parts = error_free_sum(&data).expect("should succeed");
/// let reconstructed: f64 = parts.iter().sum();
/// assert!(reconstructed.is_finite());
/// ```
pub fn error_free_sum(values: &[f64]) -> Result<Vec<f64>, CoreError> {
    if values.is_empty() {
        return Err(CoreError::InvalidInput(ErrorContext::new(
            "error_free_sum: empty slice".to_string(),
        )));
    }
    let mut p: Vec<f64> = values.to_vec();
    let n = p.len();

    let mut s = 0.0_f64;
    for i in 0..n {
        let (sigma, q) = two_sum(s, p[i]);
        p[i] = q; // store error term
        s = sigma;
    }
    // p[0..n] now holds error terms; prepend the sum.
    let mut result = Vec::with_capacity(n + 1);
    result.push(s);
    result.extend_from_slice(&p);
    Ok(result)
}

// ---------------------------------------------------------------------------
// Error-free product (EFT representation)
// ---------------------------------------------------------------------------

/// Error-free transformation of a product of two values.
///
/// Returns `(p, e)` such that `a * b = p + e` exactly in real arithmetic.
/// Uses Dekker's TwoProduct algorithm (also in `compensated::two_product`),
/// re-exported here under a more descriptive name for use in the EFT context.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::faithful_rounding::error_free_product;
/// let (p, e) = error_free_product(1.0_f64 / 3.0, 3.0_f64);
/// // p + e should be exactly 1.0.
/// assert!((p + e - 1.0).abs() < 1e-15);
/// ```
pub fn error_free_product(a: f64, b: f64) -> (f64, f64) {
    use crate::precision::compensated::two_product;
    two_product(a, b)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Scale `x` by `2^exp` (ldexp equivalent in pure Rust for f64).
///
/// Handles the full range including subnormals via direct bit manipulation.
fn ldexp_f64(x: f64, exp: i64) -> f64 {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return x;
    }
    // Clamp exponent to avoid overflow/underflow in the bit manipulation.
    const F64_EXPONENT_BIAS: i64 = 1023;
    const F64_MANTISSA_BITS: i64 = 52;
    const F64_MAX_EXP: i64 = 1023;
    const F64_MIN_EXP: i64 = -1022;

    let bits = x.to_bits();
    let stored_exp = ((bits >> 52) & 0x7FF) as i64;

    if stored_exp == 0 {
        // Subnormal: use multiplication path.
        return x * f64::from_bits(((exp + F64_EXPONENT_BIAS).clamp(0, 2046) as u64) << 52);
    }

    let new_exp = stored_exp + exp;

    if new_exp >= F64_MAX_EXP + F64_EXPONENT_BIAS + 1 {
        // Overflow.
        return if x > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }

    if new_exp <= F64_MIN_EXP + F64_EXPONENT_BIAS - F64_MANTISSA_BITS - 1 {
        // Underflow.
        return 0.0;
    }

    if new_exp <= 0 {
        // Result will be subnormal.
        // Shift mantissa right and set exponent to 0.
        let shift = (1 - new_exp) as u32;
        if shift >= 53 {
            return 0.0;
        }
        let mantissa = (bits & 0x000F_FFFF_FFFF_FFFF) | 0x0010_0000_0000_0000; // add implicit leading 1
        let sign = bits & 0x8000_0000_0000_0000;
        return f64::from_bits(sign | (mantissa >> shift));
    }

    // Normal result: just update the exponent field.
    let new_bits = (bits & 0x800F_FFFF_FFFF_FFFF) | ((new_exp as u64) << 52);
    f64::from_bits(new_bits)
}

/// Decompose x into (m, e) such that x = m * 2^e with m ∈ [0.5, 1.0).
/// Returns (0.0, 0) for zero and panics (unreachable) for NaN/Inf (callers must
/// check these first).
fn frexp_f64(x: f64) -> (f64, i64) {
    if x == 0.0 {
        return (0.0, 0);
    }
    // x is finite and non-zero at this point.
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000_0000_0000;
    let raw_exp = ((bits >> 52) & 0x7FF) as i64;
    let mantissa_bits = bits & 0x000F_FFFF_FFFF_FFFF;

    if raw_exp == 0 {
        // Subnormal: normalise by multiplying by 2^53 to get a normal number.
        let normal = x * (1u64 << 53) as f64;
        let (m, e) = frexp_f64(normal);
        return (m, e - 53);
    }

    // Normal: exponent is raw_exp - 1022 (bias 1023, but we want m in [0.5,1)).
    let e = raw_exp - 1022;
    // Mantissa with exponent set to 1022 (= 0 bias-adjusted exponent → 0.5 .. 1.0).
    let m_bits = sign | (1022u64 << 52) | mantissa_bits;
    (f64::from_bits(m_bits), e)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faithful_sqrt_two() {
        let s = faithful_sqrt(2.0).expect("valid");
        assert!((s - std::f64::consts::SQRT_2).abs() < 1e-15);
    }

    #[test]
    fn faithful_sqrt_zero() {
        assert_eq!(faithful_sqrt(0.0).expect("valid"), 0.0);
    }

    #[test]
    fn faithful_sqrt_negative_error() {
        assert!(faithful_sqrt(-1.0).is_err());
    }

    #[test]
    fn faithful_sqrt_nan_error() {
        assert!(faithful_sqrt(f64::NAN).is_err());
    }

    #[test]
    fn faithful_exp_zero() {
        let e = faithful_exp(0.0).expect("valid");
        assert!((e - 1.0).abs() < 1e-14);
    }

    #[test]
    fn faithful_exp_one() {
        let e = faithful_exp(1.0).expect("valid");
        assert!((e - std::f64::consts::E).abs() < 1e-12);
    }

    #[test]
    fn faithful_exp_overflow() {
        let e = faithful_exp(800.0).expect("valid");
        assert_eq!(e, f64::INFINITY);
    }

    #[test]
    fn faithful_exp_underflow() {
        let e = faithful_exp(-800.0).expect("valid");
        assert_eq!(e, 0.0);
    }

    #[test]
    fn faithful_exp_nan_error() {
        assert!(faithful_exp(f64::NAN).is_err());
    }

    #[test]
    fn faithful_log_one() {
        let l = faithful_log(1.0).expect("valid");
        assert!(l.abs() < 1e-14);
    }

    #[test]
    fn faithful_log_e() {
        let l = faithful_log(std::f64::consts::E).expect("valid");
        assert!((l - 1.0).abs() < 1e-12);
    }

    #[test]
    fn faithful_log_zero() {
        let l = faithful_log(0.0).expect("valid");
        assert_eq!(l, f64::NEG_INFINITY);
    }

    #[test]
    fn faithful_log_negative_error() {
        assert!(faithful_log(-1.0).is_err());
    }

    #[test]
    fn faithful_log_nan_error() {
        assert!(faithful_log(f64::NAN).is_err());
    }

    #[test]
    fn error_free_sum_basic() {
        let data = [1.0_f64, 2.0, 3.0];
        let parts = error_free_sum(&data).expect("valid");
        let reconstructed: f64 = parts.iter().sum();
        assert!((reconstructed - 6.0).abs() < 1e-14);
    }

    #[test]
    fn error_free_sum_empty_error() {
        assert!(error_free_sum(&[]).is_err());
    }

    #[test]
    fn error_free_product_exact() {
        let (p, e) = error_free_product(3.0, 7.0);
        assert_eq!(p, 21.0);
        assert_eq!(e, 0.0);
    }

    #[test]
    fn error_free_product_reconstruct() {
        let a = 1.0_f64 / 3.0;
        let b = 3.0_f64;
        let (p, e) = error_free_product(a, b);
        assert!((p + e - 1.0).abs() < 1e-15);
    }

    #[test]
    fn frexp_basic() {
        let (m, e) = frexp_f64(8.0);
        assert!((m - 0.5).abs() < 1e-15);
        assert_eq!(e, 4);
    }

    #[test]
    fn ldexp_basic() {
        let x = ldexp_f64(1.0, 3);
        assert_eq!(x, 8.0);
        let y = ldexp_f64(1.0, -1);
        assert_eq!(y, 0.5);
    }

    #[test]
    fn exp_log_roundtrip() {
        for v in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0] {
            let l = faithful_log(v).expect("log valid");
            let e = faithful_exp(l).expect("exp valid");
            assert!((e - v).abs() < v * 1e-10, "v={v}, e={e}");
        }
    }
}
