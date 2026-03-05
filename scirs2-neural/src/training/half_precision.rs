//! Half-precision floating-point types for mixed-precision training.
//!
//! Provides [`Half`] (IEEE 754 FP16) and [`BFloat16`] (Google Brain Float 16)
//! types with full arithmetic, conversion, and utility functions.

// ============================================================================
// Half Precision Tensor Operations
// ============================================================================

/// Half precision (FP16) value representation
///
/// Uses u16 internally to store the IEEE 754 half-precision float
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Half(u16);

impl Half {
    /// Zero value in half precision
    pub const ZERO: Half = Half(0);

    /// One value in half precision
    pub const ONE: Half = Half(0x3c00);

    /// Maximum finite value
    pub const MAX: Half = Half(0x7bff);

    /// Minimum positive normalized value
    pub const MIN_POSITIVE: Half = Half(0x0400);

    /// Infinity
    pub const INFINITY: Half = Half(0x7c00);

    /// Negative infinity
    pub const NEG_INFINITY: Half = Half(0xfc00);

    /// NaN (quiet NaN)
    pub const NAN: Half = Half(0x7e00);

    /// Create a half from raw bits
    pub const fn from_bits(bits: u16) -> Self {
        Half(bits)
    }

    /// Get the raw bits
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert from f32 to half precision
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xff) as i32;
        let frac = bits & 0x7fffff;

        // Handle special cases
        if exp == 0xff {
            // Infinity or NaN
            if frac == 0 {
                // Infinity
                return Half(((sign << 15) | 0x7c00) as u16);
            } else {
                // NaN
                return Half(((sign << 15) | 0x7e00) as u16);
            }
        }

        // Rebias exponent from f32 (bias 127) to f16 (bias 15)
        let new_exp = exp - 127 + 15;

        if new_exp <= 0 {
            // Denormalized or zero
            if new_exp < -10 {
                // Too small, becomes zero
                return Half((sign << 15) as u16);
            }
            // Denormalized
            let shift = 1 - new_exp;
            let frac_with_hidden = frac | 0x800000;
            let frac16 = (frac_with_hidden >> (shift + 13)) as u16;
            return Half(((sign << 15) | frac16 as u32) as u16);
        }

        if new_exp >= 31 {
            // Overflow to infinity
            return Half(((sign << 15) | 0x7c00) as u16);
        }

        // Normal case
        let frac16 = (frac >> 13) as u16;
        Half(((sign << 15) | ((new_exp as u32) << 10) | frac16 as u32) as u16)
    }

    /// Convert from half precision to f32
    pub fn to_f32(self) -> f32 {
        let bits = self.0 as u32;
        let sign = (bits >> 15) & 1;
        let exp = (bits >> 10) & 0x1f;
        let frac = bits & 0x3ff;

        if exp == 0 {
            if frac == 0 {
                // Zero
                return f32::from_bits(sign << 31);
            }
            // Denormalized
            let mut frac = frac;
            let mut e = -14i32;
            while frac & 0x400 == 0 {
                frac <<= 1;
                e -= 1;
            }
            frac &= 0x3ff;
            let exp32 = (e + 127) as u32;
            let frac32 = frac << 13;
            return f32::from_bits((sign << 31) | (exp32 << 23) | frac32);
        }

        if exp == 0x1f {
            // Infinity or NaN
            if frac == 0 {
                return f32::from_bits((sign << 31) | 0x7f800000);
            }
            return f32::from_bits((sign << 31) | 0x7fc00000);
        }

        // Normal case
        let exp32 = (exp as i32 - 15 + 127) as u32;
        let frac32 = frac << 13;
        f32::from_bits((sign << 31) | (exp32 << 23) | frac32)
    }

    /// Check if the value is NaN
    pub fn is_nan(self) -> bool {
        (self.0 & 0x7c00) == 0x7c00 && (self.0 & 0x03ff) != 0
    }

    /// Check if the value is infinite
    pub fn is_infinite(self) -> bool {
        (self.0 & 0x7fff) == 0x7c00
    }

    /// Check if the value is finite
    pub fn is_finite(self) -> bool {
        (self.0 & 0x7c00) != 0x7c00
    }

    /// Check if the value is zero
    pub fn is_zero(self) -> bool {
        (self.0 & 0x7fff) == 0
    }
}

impl From<f32> for Half {
    fn from(value: f32) -> Self {
        Half::from_f32(value)
    }
}

impl From<Half> for f32 {
    fn from(value: Half) -> Self {
        value.to_f32()
    }
}

impl From<f64> for Half {
    fn from(value: f64) -> Self {
        Half::from_f32(value as f32)
    }
}

impl From<Half> for f64 {
    fn from(value: Half) -> Self {
        value.to_f32() as f64
    }
}


// ────────────────────────────────────────────────────────────────────────────
// BFloat16 (Brain Float 16) half-precision type
// ────────────────────────────────────────────────────────────────────────────

/// A BFloat16 (brain float) value stored as raw 16-bit integer bits.
///
/// BFloat16 uses the same 8-bit exponent as `f32` but truncates the 23-bit
/// significand to 7 bits.  This makes conversion trivial (just drop/round the
/// lower 16 bits of the `f32` representation) and gives the same dynamic range
/// as FP32 — which makes BF16 particularly attractive for training stability.
///
/// # Layout
///
/// ```text
/// Bit 15  : sign
/// Bits 14–7 : exponent (same bias 127 as f32)
/// Bits 6–0  : fraction (7 bits instead of f32's 23)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BFloat16(u16);

impl BFloat16 {
    /// Positive zero.
    pub const ZERO: BFloat16 = BFloat16(0);

    /// Positive one.
    pub const ONE: BFloat16 = BFloat16(0x3f80);

    /// Largest finite BF16 value.
    pub const MAX: BFloat16 = BFloat16(0x7f7f);

    /// Smallest positive normalized BF16 value.
    pub const MIN_POSITIVE: BFloat16 = BFloat16(0x0080);

    /// Positive infinity.
    pub const INFINITY: BFloat16 = BFloat16(0x7f80);

    /// Negative infinity.
    pub const NEG_INFINITY: BFloat16 = BFloat16(0xff80);

    /// A quiet NaN.
    pub const NAN: BFloat16 = BFloat16(0x7fc0);

    /// Construct from raw bits.
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        BFloat16(bits)
    }

    /// Return the raw bit pattern.
    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    // ── Conversion from f32 ──────────────────────────────────────────────

    /// Convert an `f32` to `BFloat16` using round-to-nearest-even.
    ///
    /// Special values (NaN, ±∞) are preserved exactly.
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();

        // Propagate NaN, preserve its sign & quiet bit.
        if (bits & 0x7f80_0000) == 0x7f80_0000 && (bits & 0x007f_ffff) != 0 {
            // Set quiet bit to ensure it stays a quiet NaN.
            return BFloat16(((bits >> 16) | 0x0040) as u16);
        }

        // Round the bottom 16 bits using round-to-nearest-even.
        let rounding_bias = 0x0000_7fff_u32 + ((bits >> 16) & 1);
        BFloat16(((bits + rounding_bias) >> 16) as u16)
    }

    /// Convert `BFloat16` back to `f32` (lossless — we just zero-extend).
    #[inline]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    // ── Convenience conversion from/to f64 ──────────────────────────────

    /// Convert an `f64` to `BFloat16` (via `f32`).
    #[inline]
    pub fn from_f64(value: f64) -> Self {
        BFloat16::from_f32(value as f32)
    }

    /// Convert `BFloat16` to `f64`.
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    // ── Classification helpers ───────────────────────────────────────────

    /// Returns `true` if the value is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        (self.0 & 0x7f80) == 0x7f80 && (self.0 & 0x007f) != 0
    }

    /// Returns `true` if the value is ±∞.
    #[inline]
    pub fn is_infinite(self) -> bool {
        (self.0 & 0x7fff) == 0x7f80
    }

    /// Returns `true` if the value is finite (not NaN and not ±∞).
    #[inline]
    pub fn is_finite(self) -> bool {
        (self.0 & 0x7f80) != 0x7f80
    }

    /// Returns `true` if the value is ±0.
    #[inline]
    pub fn is_zero(self) -> bool {
        (self.0 & 0x7fff) == 0
    }

    /// Returns `true` if the value is subnormal (exponent bits all zero,
    /// significand non-zero).
    #[inline]
    pub fn is_subnormal(self) -> bool {
        (self.0 & 0x7f80) == 0 && (self.0 & 0x007f) != 0
    }

    /// Returns the absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        BFloat16(self.0 & 0x7fff)
    }

    /// Returns the negation.
    #[inline]
    pub fn neg(self) -> Self {
        BFloat16(self.0 ^ 0x8000)
    }
}

// ── From trait impls ─────────────────────────────────────────────────────

impl From<f32> for BFloat16 {
    #[inline]
    fn from(value: f32) -> Self {
        BFloat16::from_f32(value)
    }
}

impl From<BFloat16> for f32 {
    #[inline]
    fn from(value: BFloat16) -> Self {
        value.to_f32()
    }
}

impl From<f64> for BFloat16 {
    #[inline]
    fn from(value: f64) -> Self {
        BFloat16::from_f64(value)
    }
}

impl From<BFloat16> for f64 {
    #[inline]
    fn from(value: BFloat16) -> Self {
        value.to_f64()
    }
}

impl From<Half> for BFloat16 {
    /// Lossless round-trip via f32: `Half → f32 → BFloat16`.
    #[inline]
    fn from(value: Half) -> Self {
        BFloat16::from_f32(value.to_f32())
    }
}

impl From<BFloat16> for Half {
    /// Lossless round-trip via f32: `BFloat16 → f32 → Half`.
    #[inline]
    fn from(value: BFloat16) -> Self {
        Half::from_f32(value.to_f32())
    }
}

// ── std::fmt::Display ───────────────────────────────────────────────────────

impl std::fmt::Display for BFloat16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

// ── Arithmetic (via f32) ─────────────────────────────────────────────────────

impl std::ops::Add for BFloat16 {
    type Output = BFloat16;
    fn add(self, rhs: BFloat16) -> BFloat16 {
        BFloat16::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl std::ops::Sub for BFloat16 {
    type Output = BFloat16;
    fn sub(self, rhs: BFloat16) -> BFloat16 {
        BFloat16::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl std::ops::Mul for BFloat16 {
    type Output = BFloat16;
    fn mul(self, rhs: BFloat16) -> BFloat16 {
        BFloat16::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl std::ops::Div for BFloat16 {
    type Output = BFloat16;
    fn div(self, rhs: BFloat16) -> BFloat16 {
        BFloat16::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl std::ops::Neg for BFloat16 {
    type Output = BFloat16;
    fn neg(self) -> BFloat16 {
        self.neg()
    }
}

impl PartialOrd for BFloat16 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

/// Utility: convert a slice of `f32` values to `BFloat16`.
pub fn f32_slice_to_bf16(src: &[f32]) -> Vec<BFloat16> {
    src.iter().copied().map(BFloat16::from_f32).collect()
}

/// Utility: convert a slice of `BFloat16` values back to `f32`.
pub fn bf16_slice_to_f32(src: &[BFloat16]) -> Vec<f32> {
    src.iter().copied().map(BFloat16::to_f32).collect()
}

/// Utility: convert a slice of `f64` values to `BFloat16`.
pub fn f64_slice_to_bf16(src: &[f64]) -> Vec<BFloat16> {
    src.iter().copied().map(BFloat16::from_f64).collect()
}

/// Compute the global L2 norm of a BF16 gradient slice (upcasts to f32 internally).
pub fn bf16_grad_norm(grads: &[BFloat16]) -> f32 {
    grads
        .iter()
        .map(|g| {
            let v = g.to_f32();
            v * v
        })
        .sum::<f32>()
        .sqrt()
}

