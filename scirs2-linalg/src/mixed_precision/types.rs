//! Pure Rust IEEE 754 half-precision (F16) and bfloat16 (BF16) types
//!
//! This module provides software implementations of 16-bit floating-point formats
//! without any external dependencies. Both types use round-to-nearest-even semantics
//! for conversions from higher precision formats.
//!
//! # F16 (IEEE 754 binary16)
//! - Sign: 1 bit, Exponent: 5 bits (bias 15), Mantissa: 10 bits
//! - Range: ~6.1e-5 to 65504
//! - Precision: ~3 decimal digits
//!
//! # BF16 (bfloat16 / Google Brain float16)
//! - Sign: 1 bit, Exponent: 8 bits (bias 127), Mantissa: 7 bits
//! - Range: same as f32 (~1.2e-38 to ~3.4e38)
//! - Precision: ~2 decimal digits

use std::cmp::Ordering;
use std::fmt;

// ============================================================================
// F16 - IEEE 754 binary16 half-precision float
// ============================================================================

/// IEEE 754 binary16 half-precision floating-point type.
///
/// Layout: `[sign:1][exponent:5][mantissa:10]`
///
/// - Exponent bias: 15
/// - Max finite value: 65504.0
/// - Smallest positive normal: 2^-14 (~6.1e-5)
/// - Smallest positive subnormal: 2^-24 (~5.96e-8)
/// - Machine epsilon: 2^-10 (~9.77e-4)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct F16(u16);

impl F16 {
    /// Positive zero
    pub const ZERO: F16 = F16(0x0000);
    /// Negative zero
    pub const NEG_ZERO: F16 = F16(0x8000);
    /// One (1.0)
    pub const ONE: F16 = F16(0x3C00);
    /// Negative one (-1.0)
    pub const NEG_ONE: F16 = F16(0xBC00);
    /// Maximum finite value (65504.0)
    pub const MAX: F16 = F16(0x7BFF);
    /// Minimum positive normal value (2^-14)
    pub const MIN_POSITIVE: F16 = F16(0x0400);
    /// Machine epsilon (2^-10)
    pub const EPSILON: F16 = F16(0x1400);
    /// Not a Number (quiet NaN)
    pub const NAN: F16 = F16(0x7E00);
    /// Positive infinity
    pub const INFINITY: F16 = F16(0x7C00);
    /// Negative infinity
    pub const NEG_INFINITY: F16 = F16(0xFC00);

    /// Create an F16 from raw bits.
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        F16(bits)
    }

    /// Return the raw bit representation.
    #[inline]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Convert an `f32` to `F16` using round-to-nearest-even.
    ///
    /// # Algorithm
    /// 1. Extract sign, exponent, mantissa from f32
    /// 2. Handle special cases (NaN, Inf, zero)
    /// 3. Rebias exponent: f32_exp - 127 + 15
    /// 4. Underflow to zero or subnormal
    /// 5. Overflow to infinity
    /// 6. Normal: truncate mantissa to 10 bits with round-to-nearest-even
    #[inline]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x007F_FFFF;

        let h_sign = (sign as u16) << 15;

        // Zero (including negative zero)
        if exp == 0 && mantissa == 0 {
            return F16(h_sign);
        }

        // NaN
        if exp == 0xFF && mantissa != 0 {
            // Preserve NaN: set at least one mantissa bit
            let h_mantissa = (mantissa >> 13) as u16;
            let h_mantissa = if h_mantissa == 0 { 1 } else { h_mantissa };
            return F16(h_sign | 0x7C00 | (h_mantissa & 0x03FF));
        }

        // Infinity
        if exp == 0xFF {
            return F16(h_sign | 0x7C00);
        }

        // Rebias exponent
        let new_exp = exp - 127 + 15;

        // Subnormal f32 input
        if exp == 0 {
            // f32 subnormal -> f16 will be zero (too small)
            return F16(h_sign);
        }

        // Underflow: result is too small for f16
        if new_exp < -10 {
            return F16(h_sign);
        }

        // Subnormal f16 result
        if new_exp < 1 {
            // The value is 2^(exp-127) * (1 + mantissa/2^23)
            // In f16 subnormal: 2^(-14) * (m/2^10) where m is the subnormal mantissa
            // So we need: m/2^10 = 2^(exp-127+14) * (1 + mantissa/2^23)
            // Shift = 1 - new_exp = number of positions to shift right
            let shift = (1 - new_exp) as u32;
            // Add implicit leading 1 to mantissa
            let full_mantissa = mantissa | 0x0080_0000;
            // We need to shift right by (13 + shift) from the 23-bit f32 mantissa
            // to get the 10-bit f16 mantissa
            let total_shift = 13 + shift;

            if total_shift >= 24 {
                return F16(h_sign);
            }

            let h_mantissa_raw = full_mantissa >> total_shift;

            // Round-to-nearest-even
            let round_bit = if total_shift > 0 {
                (full_mantissa >> (total_shift - 1)) & 1
            } else {
                0
            };
            let sticky_bits = if total_shift > 1 {
                full_mantissa & ((1 << (total_shift - 1)) - 1)
            } else {
                0
            };

            let h_mantissa = if round_bit != 0 && (sticky_bits != 0 || (h_mantissa_raw & 1) != 0) {
                (h_mantissa_raw + 1) as u16
            } else {
                h_mantissa_raw as u16
            };

            return F16(h_sign | h_mantissa);
        }

        // Overflow: result is too large for f16
        if new_exp > 30 {
            return F16(h_sign | 0x7C00);
        }

        // Normal case: truncate mantissa from 23 bits to 10 bits
        let h_exp = (new_exp as u16) << 10;
        let h_mantissa_raw = (mantissa >> 13) as u16;

        // Round-to-nearest-even
        let round_bit = (mantissa >> 12) & 1;
        let sticky_bits = mantissa & 0x0FFF;

        let h_mantissa = if round_bit != 0 && (sticky_bits != 0 || (h_mantissa_raw & 1) != 0) {
            h_mantissa_raw + 1
        } else {
            h_mantissa_raw
        };

        // Check if rounding caused mantissa overflow (carry into exponent)
        if h_mantissa > 0x03FF {
            // Mantissa overflowed, increment exponent
            let h_exp_inc = h_exp + (1 << 10);
            if h_exp_inc >= 0x7C00 {
                // Overflow to infinity
                return F16(h_sign | 0x7C00);
            }
            return F16(h_sign | h_exp_inc | (h_mantissa & 0x03FF));
        }

        F16(h_sign | h_exp | h_mantissa)
    }

    /// Convert this `F16` to `f32`.
    #[inline]
    pub fn to_f32(self) -> f32 {
        let sign = ((self.0 >> 15) & 1) as u32;
        let exp = ((self.0 >> 10) & 0x1F) as u32;
        let mantissa = (self.0 & 0x03FF) as u32;

        if exp == 0 {
            if mantissa == 0 {
                // Zero
                return f32::from_bits(sign << 31);
            }
            // Subnormal: value = 2^(-14) * (mantissa / 2^10)
            // Normalize by finding leading 1 and shifting it out
            let mut m = mantissa;
            let mut shift: i32 = 0;
            while (m & 0x0400) == 0 {
                m <<= 1;
                shift += 1;
            }
            // Remove the implicit leading 1 bit
            m &= 0x03FF;
            // f32 exponent: the subnormal value is 2^(-14) * (mantissa / 2^10).
            // After normalizing (shifting left by `shift`), the mantissa
            // had its leading 1 at bit position 10, so the true exponent is
            // -14 + (10 - shift - 10) = -14 - shift.
            // But we normalized it to 1.xxxx form, so we add 0 for that.
            // In f32 biased form: (-14 - shift) + 127 = 113 - shift
            let f32_exp = (113 - shift) as u32;
            let f32_mantissa = m << 13;
            return f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mantissa);
        }

        if exp == 0x1F {
            if mantissa == 0 {
                // Infinity
                return f32::from_bits((sign << 31) | 0x7F80_0000);
            }
            // NaN
            return f32::from_bits((sign << 31) | 0x7F80_0000 | (mantissa << 13));
        }

        // Normal number
        // f32 exponent: (exp - 15) + 127 = exp + 112
        let f32_exp = (exp + 112) << 23;
        let f32_mantissa = mantissa << 13;
        f32::from_bits((sign << 31) | f32_exp | f32_mantissa)
    }

    /// Convert an `f64` to `F16`.
    #[inline]
    pub fn from_f64(value: f64) -> Self {
        // Go through f32 as intermediate (f64 -> f32 -> f16)
        // This is correct because f16 has less precision than f32.
        F16::from_f32(value as f32)
    }

    /// Convert this `F16` to `f64`.
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    /// Returns `true` if this value is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        let exp = (self.0 >> 10) & 0x1F;
        let mantissa = self.0 & 0x03FF;
        exp == 0x1F && mantissa != 0
    }

    /// Returns `true` if this value is positive or negative infinity.
    #[inline]
    pub fn is_infinite(self) -> bool {
        let exp = (self.0 >> 10) & 0x1F;
        let mantissa = self.0 & 0x03FF;
        exp == 0x1F && mantissa == 0
    }

    /// Returns `true` if this is a subnormal (denormalized) number.
    #[inline]
    pub fn is_subnormal(self) -> bool {
        let exp = (self.0 >> 10) & 0x1F;
        let mantissa = self.0 & 0x03FF;
        exp == 0 && mantissa != 0
    }

    /// Returns `true` if this value is positive or negative zero.
    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 & 0x7FFF == 0
    }

    /// Returns `true` if this value is finite (not NaN or infinity).
    #[inline]
    pub fn is_finite(self) -> bool {
        (self.0 >> 10) & 0x1F != 0x1F
    }

    /// Returns the absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        F16(self.0 & 0x7FFF)
    }

    /// Returns the negation.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        F16(self.0 ^ 0x8000)
    }
}

impl fmt::Display for F16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl From<f32> for F16 {
    #[inline]
    fn from(value: f32) -> Self {
        F16::from_f32(value)
    }
}

impl From<f64> for F16 {
    #[inline]
    fn from(value: f64) -> Self {
        F16::from_f64(value)
    }
}

impl From<F16> for f32 {
    #[inline]
    fn from(value: F16) -> Self {
        value.to_f32()
    }
}

impl From<F16> for f64 {
    #[inline]
    fn from(value: F16) -> Self {
        value.to_f64()
    }
}

impl PartialOrd for F16 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

impl Default for F16 {
    fn default() -> Self {
        F16::ZERO
    }
}

// ============================================================================
// BF16 - bfloat16 (Google Brain float16)
// ============================================================================

/// bfloat16 floating-point type (Google Brain format).
///
/// Layout: `[sign:1][exponent:8][mantissa:7]`
///
/// - Exponent bias: 127 (same as f32)
/// - Same dynamic range as f32
/// - Reduced precision (~2 decimal digits vs ~7 for f32)
/// - Fast conversion: upper 16 bits of f32
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct BF16(u16);

impl BF16 {
    /// Positive zero
    pub const ZERO: BF16 = BF16(0x0000);
    /// Negative zero
    pub const NEG_ZERO: BF16 = BF16(0x8000);
    /// One (1.0)
    pub const ONE: BF16 = BF16(0x3F80);
    /// Negative one (-1.0)
    pub const NEG_ONE: BF16 = BF16(0xBF80);
    /// Maximum finite value (~3.39e38)
    pub const MAX: BF16 = BF16(0x7F7F);
    /// Minimum positive normal value (~1.18e-38)
    pub const MIN_POSITIVE: BF16 = BF16(0x0080);
    /// Machine epsilon (2^-7 = 0.0078125)
    pub const EPSILON: BF16 = BF16(0x3C00);
    /// Not a Number (quiet NaN)
    pub const NAN: BF16 = BF16(0x7FC0);
    /// Positive infinity
    pub const INFINITY: BF16 = BF16(0x7F80);
    /// Negative infinity
    pub const NEG_INFINITY: BF16 = BF16(0xFF80);

    /// Create a BF16 from raw bits.
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        BF16(bits)
    }

    /// Return the raw bit representation.
    #[inline]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Convert an `f32` to `BF16` with round-to-nearest-even.
    ///
    /// Fast path: BF16 is the upper 16 bits of f32, with rounding.
    /// The rounding formula adds `0x7FFF + ((bits >> 16) & 1)` before
    /// shifting, which implements round-to-nearest-even.
    #[inline]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();

        // Handle NaN: preserve NaN-ness but canonicalize
        if value.is_nan() {
            return BF16::NAN;
        }

        // Round-to-nearest-even: add rounding bias
        // The expression 0x7FFF + ((bits >> 16) & 1) rounds to even
        // by adding half of the truncated portion plus the LSB of the
        // result for tie-breaking.
        let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
        BF16((rounded >> 16) as u16)
    }

    /// Convert this `BF16` to `f32`.
    ///
    /// Fast path: pad the 16-bit value with 16 zero bits to get f32.
    #[inline]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    /// Convert an `f64` to `BF16`.
    #[inline]
    pub fn from_f64(value: f64) -> Self {
        BF16::from_f32(value as f32)
    }

    /// Convert this `BF16` to `f64`.
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    /// Returns `true` if this value is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        let exp = (self.0 >> 7) & 0xFF;
        let mantissa = self.0 & 0x7F;
        exp == 0xFF && mantissa != 0
    }

    /// Returns `true` if this value is positive or negative infinity.
    #[inline]
    pub fn is_infinite(self) -> bool {
        let exp = (self.0 >> 7) & 0xFF;
        let mantissa = self.0 & 0x7F;
        exp == 0xFF && mantissa == 0
    }

    /// Returns `true` if this is a subnormal (denormalized) number.
    #[inline]
    pub fn is_subnormal(self) -> bool {
        let exp = (self.0 >> 7) & 0xFF;
        let mantissa = self.0 & 0x7F;
        exp == 0 && mantissa != 0
    }

    /// Returns `true` if this value is positive or negative zero.
    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 & 0x7FFF == 0
    }

    /// Returns `true` if this value is finite (not NaN or infinity).
    #[inline]
    pub fn is_finite(self) -> bool {
        (self.0 >> 7) & 0xFF != 0xFF
    }

    /// Returns the absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        BF16(self.0 & 0x7FFF)
    }

    /// Returns the negation.
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        BF16(self.0 ^ 0x8000)
    }
}

impl fmt::Display for BF16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl From<f32> for BF16 {
    #[inline]
    fn from(value: f32) -> Self {
        BF16::from_f32(value)
    }
}

impl From<f64> for BF16 {
    #[inline]
    fn from(value: f64) -> Self {
        BF16::from_f64(value)
    }
}

impl From<BF16> for f32 {
    #[inline]
    fn from(value: BF16) -> Self {
        value.to_f32()
    }
}

impl From<BF16> for f64 {
    #[inline]
    fn from(value: BF16) -> Self {
        value.to_f64()
    }
}

impl PartialOrd for BF16 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

impl Default for BF16 {
    fn default() -> Self {
        BF16::ZERO
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- F16 tests ---

    #[test]
    fn test_f16_zero() {
        let z = F16::from_f32(0.0);
        assert_eq!(z.bits(), 0x0000);
        assert_eq!(z.to_f32(), 0.0);
        assert!(z.is_zero());
        assert!(!z.is_nan());
        assert!(!z.is_infinite());
    }

    #[test]
    fn test_f16_neg_zero() {
        let nz = F16::from_f32(-0.0);
        assert_eq!(nz.bits(), 0x8000);
        assert!(nz.is_zero());
        let val = nz.to_f32();
        assert!(val.is_sign_negative());
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_f16_one() {
        let one = F16::from_f32(1.0);
        assert_eq!(one.bits(), F16::ONE.bits());
        assert!((one.to_f32() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_f16_roundtrip_representable() {
        // Values exactly representable in f16
        let values = [0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, 0.25, 65504.0, -65504.0];
        for &v in &values {
            let h = F16::from_f32(v);
            let back = h.to_f32();
            assert_eq!(back, v, "f16 round-trip failed for {v}: got {back}");
        }
    }

    #[test]
    fn test_f16_nan() {
        let nan = F16::from_f32(f32::NAN);
        assert!(nan.is_nan());
        assert!(!nan.is_infinite());
        assert!(!nan.is_finite());
        assert!(nan.to_f32().is_nan());
    }

    #[test]
    fn test_f16_infinity() {
        let pos_inf = F16::from_f32(f32::INFINITY);
        assert!(pos_inf.is_infinite());
        assert!(!pos_inf.is_nan());
        assert_eq!(pos_inf.bits(), F16::INFINITY.bits());
        assert_eq!(pos_inf.to_f32(), f32::INFINITY);

        let neg_inf = F16::from_f32(f32::NEG_INFINITY);
        assert!(neg_inf.is_infinite());
        assert_eq!(neg_inf.bits(), F16::NEG_INFINITY.bits());
        assert_eq!(neg_inf.to_f32(), f32::NEG_INFINITY);
    }

    #[test]
    fn test_f16_overflow() {
        // Values > 65504 should become infinity
        let big = F16::from_f32(100000.0);
        assert!(big.is_infinite());
    }

    #[test]
    fn test_f16_subnormal() {
        // Smallest positive subnormal: 2^(-24)
        let tiny = F16::from_bits(0x0001);
        assert!(tiny.is_subnormal());
        assert!(!tiny.is_zero());
        let val = tiny.to_f32();
        let expected = 2.0f32.powi(-24);
        assert!(
            (val - expected).abs() < 1e-10,
            "smallest subnormal: expected {expected}, got {val}"
        );
    }

    #[test]
    fn test_f16_display() {
        let v = F16::from_f32(3.25);
        let s = format!("{v}");
        // Should display the f32 value (with f16 precision)
        assert!(!s.is_empty());
    }

    #[test]
    fn test_f16_partial_ord() {
        let a = F16::from_f32(1.0);
        let b = F16::from_f32(2.0);
        assert!(a < b);
        assert!(b > a);
        let c = F16::from_f32(1.0);
        assert_eq!(a.partial_cmp(&c), Some(Ordering::Equal));
    }

    #[test]
    fn test_f16_from_f64() {
        let v = F16::from_f64(1.5);
        assert!((v.to_f64() - 1.5).abs() < 1e-3);
    }

    #[test]
    fn test_f16_abs_neg() {
        let pos = F16::from_f32(3.0);
        let neg = pos.neg();
        assert!((neg.to_f32() + 3.0).abs() < 1e-3);
        assert_eq!(neg.abs().bits(), pos.bits());
    }

    // --- BF16 tests ---

    #[test]
    fn test_bf16_zero() {
        let z = BF16::from_f32(0.0);
        assert_eq!(z.bits(), 0x0000);
        assert_eq!(z.to_f32(), 0.0);
        assert!(z.is_zero());
    }

    #[test]
    fn test_bf16_one() {
        let one = BF16::from_f32(1.0);
        assert_eq!(one.bits(), BF16::ONE.bits());
        assert_eq!(one.to_f32(), 1.0);
    }

    #[test]
    fn test_bf16_roundtrip() {
        // bf16 has same exponent range as f32 but fewer mantissa bits
        let values = [0.0f32, 1.0, -1.0, 2.0, -2.0, 0.5, 128.0];
        for &v in &values {
            let b = BF16::from_f32(v);
            let back = b.to_f32();
            assert_eq!(back, v, "bf16 round-trip failed for {v}: got {back}");
        }
    }

    #[test]
    fn test_bf16_precision_loss() {
        // bf16 loses 16 mantissa bits compared to f32
        // 1.0 + 2^-8 should be rounded
        let v = 1.0f32 + (1.0 / 256.0); // 1.00390625
        let b = BF16::from_f32(v);
        let back = b.to_f32();
        // bf16 has 7 mantissa bits, so precision is ~2^-7 = 0.0078125
        // The value 1.00390625 might or might not be representable
        // but the error should be within epsilon
        assert!((back - v).abs() < 0.01);
    }

    #[test]
    fn test_bf16_nan() {
        let nan = BF16::from_f32(f32::NAN);
        assert!(nan.is_nan());
        assert!(nan.to_f32().is_nan());
    }

    #[test]
    fn test_bf16_infinity() {
        let inf = BF16::from_f32(f32::INFINITY);
        assert!(inf.is_infinite());
        assert_eq!(inf.to_f32(), f32::INFINITY);

        let ninf = BF16::from_f32(f32::NEG_INFINITY);
        assert!(ninf.is_infinite());
        assert_eq!(ninf.to_f32(), f32::NEG_INFINITY);
    }

    #[test]
    fn test_bf16_large_range() {
        // BF16 should handle the same exponent range as f32
        // Use a value well within range to avoid rounding to infinity
        let large = BF16::from_f32(1.0e38);
        assert!(large.is_finite());
        let back = large.to_f32();
        assert!(back > 9.0e37);
        assert!(back.is_finite());

        // Verify BF16::MAX is finite and large
        let max_val = BF16::MAX.to_f32();
        assert!(BF16::MAX.is_finite());
        assert!(max_val > 3.0e38, "BF16 max should be > 3e38, got {max_val}");
    }

    #[test]
    fn test_bf16_to_f32_fast_path() {
        // Verify the fast path: BF16 bits << 16 == f32 bits (for exact values)
        let one = BF16::ONE;
        let f = one.to_f32();
        assert_eq!(f, 1.0f32);
        assert_eq!(f.to_bits(), (one.bits() as u32) << 16);
    }

    #[test]
    fn test_bf16_display() {
        let v = BF16::from_f32(42.0);
        let s = format!("{v}");
        assert!(s.contains("42"));
    }

    #[test]
    fn test_bf16_partial_ord() {
        let a = BF16::from_f32(1.0);
        let b = BF16::from_f32(2.0);
        assert!(a < b);
    }

    #[test]
    fn test_bf16_neg_zero() {
        let nz = BF16::from_f32(-0.0);
        assert!(nz.is_zero());
        assert!(nz.to_f32().is_sign_negative());
    }
}
