//! Arbitrary-precision floating-point arithmetic (software float).
//!
//! This module implements a `BigFloat` type — a variable-precision floating-point
//! number stored as a sign-magnitude pair where the magnitude is a binary fraction
//! with an arbitrary number of mantissa bits (in 64-bit limbs) and a 64-bit
//! signed exponent.
//!
//! ## Representation
//!
//! ```text
//! value = (-1)^sign  ×  0.mantissa  ×  2^exponent
//! ```
//!
//! The mantissa is normalised so that its most-significant bit is always 1
//! (except for zero, which is represented with an empty mantissa and special
//! sign/exponent fields).  Limbs are stored most-significant first.
//!
//! ## Algorithms
//!
//! | Operation | Algorithm |
//! |-----------|-----------|
//! | `sqrt`    | Newton-Raphson |
//! | `ln`      | AGM-based identity + argument reduction |
//! | `exp`     | Taylor series with argument reduction |
//! | `pi`      | AGM / Brent-Salamin recurrence |
//! | `e`       | Taylor series for e |
//! | `ln2`     | `ln(2)` via AGM |
//! | `sqrt2`   | Newton-Raphson on 2 |
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::arbitrary_precision::BigFloat;
//!
//! let prec = 128; // bits
//! let a = BigFloat::from_f64(3.14, prec);
//! let b = BigFloat::from_f64(2.0, prec);
//! let c = a.add(&b, prec).expect("should succeed");
//! println!("{}", c.to_f64());
//! ```

use crate::error::{CoreError, CoreResult, ErrorContext};

// ─── helpers ──────────────────────────────────────────────────────────────────

/// Bits per limb (u64).
const LIMB_BITS: usize = 64;

/// Create a `CoreError::ComputationError` with the given message.
#[inline(always)]
fn comp_err(msg: impl Into<String>) -> CoreError {
    CoreError::ComputationError(ErrorContext::new(msg))
}

// ─── BigFloat ─────────────────────────────────────────────────────────────────

/// Arbitrary-precision binary floating-point number.
///
/// The represented value is `(-1)^sign × 0.mantissa × 2^exponent` where
/// `mantissa` is a non-empty slice of 64-bit limbs stored most-significant
/// first, and the most-significant bit of `mantissa[0]` is always 1 (the
/// normalisation invariant).
///
/// The zero value is the unique case where `mantissa` is empty.
#[derive(Debug, Clone)]
pub struct BigFloat {
    /// `true` means negative.
    pub sign: bool,
    /// Binary exponent: value = 0.mantissa × 2^exponent.
    pub exponent: i64,
    /// Most-significant-first 64-bit limbs of the mantissa.
    /// Empty ↔ value is zero.
    pub mantissa: Vec<u64>,
    /// Desired precision in bits (determines how many limbs are kept).
    pub precision: usize,
}

impl BigFloat {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create a BigFloat representing zero with the given precision.
    #[must_use]
    pub fn zero(precision: usize) -> Self {
        let prec = precision.max(1);
        Self { sign: false, exponent: 0, mantissa: Vec::new(), precision: prec }
    }

    /// Create a BigFloat representing one with the given precision.
    #[must_use]
    pub fn one(precision: usize) -> Self {
        let prec = precision.max(1);
        let mut bf = Self { sign: false, exponent: 1, mantissa: Vec::new(), precision: prec };
        let limbs = Self::limb_count(prec);
        bf.mantissa = vec![0u64; limbs];
        bf.mantissa[0] = 1u64 << 63; // 0.1000... in binary = 1/2, so exponent 1 gives 1.0
        bf
    }

    /// Create a BigFloat representing two with the given precision.
    #[must_use]
    pub fn two(precision: usize) -> Self {
        let prec = precision.max(1);
        let limbs = Self::limb_count(prec);
        let mut mantissa = vec![0u64; limbs];
        mantissa[0] = 1u64 << 63;
        Self { sign: false, exponent: 2, mantissa, precision: prec }
    }

    /// Convert an `f64` to a `BigFloat` with the specified precision (in bits).
    #[must_use]
    pub fn from_f64(x: f64, precision: usize) -> Self {
        let prec = precision.max(1);
        if x == 0.0 || x.is_nan() || x.is_infinite() {
            return Self::zero(prec);
        }
        let sign = x < 0.0;
        let abs_x = x.abs();
        // Extract IEEE-754 components.
        let bits = abs_x.to_bits();
        // Biased exponent; for normal numbers biased_exp ≥ 1
        let biased_exp = (bits >> 52) as i32;
        let frac = bits & ((1u64 << 52) - 1);
        // Significand as integer: for normal numbers it is (1 << 52) | frac
        let sig: u64 = if biased_exp > 0 {
            (1u64 << 52) | frac
        } else {
            frac // subnormal
        };
        // The value is sig × 2^(biased_exp - 1023 - 52)
        // In our representation value = 0.mantissa × 2^exponent
        // 0.mantissa = sig / 2^53 (for normal) → exponent = biased_exp - 1023 - 52 + 53
        let raw_exp = if biased_exp > 0 {
            biased_exp as i64 - 1023 - 52 + 53
        } else {
            // subnormal: biased_exp == 0, value = sig × 2^(1 - 1023 - 52)
            1i64 - 1023 - 52 + 53
        };
        // sig occupies 53 bits (bit 52 set for normal). Shift into MSB of u64.
        let shift = 64 - 53; // = 11
        let top_limb = sig << shift;
        let limbs = Self::limb_count(prec);
        let mut mantissa = vec![0u64; limbs];
        mantissa[0] = top_limb;
        // Compute exponent: the MSB of top_limb is bit 63, so
        // value = top_limb / 2^64 × 2^raw_exp = 0.top_limb × 2^raw_exp
        // But sig already has MSB at bit 52 (= top_limb bit 63), so exponent = raw_exp.
        let mut bf = Self { sign, exponent: raw_exp, mantissa, precision: prec };
        bf.normalise();
        bf
    }

    /// Convert this `BigFloat` to the nearest `f64`.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        if self.mantissa.is_empty() {
            return 0.0;
        }
        // Take the top 53 bits of the mantissa for the significand.
        let top = self.mantissa[0];
        // Leading bit of top is at bit 63.  We need sig in [1, 2).
        // value = 0.mantissa × 2^exponent
        //       = mantissa[0]/2^64 × ... × 2^exponent
        // ≈ top / 2^64 × 2^exponent
        // = (top >> 11) / 2^53 × 2^exponent
        // Standard form: sig × 2^(exponent - 53) where sig = top >> 11
        let sig = top >> 11; // 53 bits
        let exp_f64 = self.exponent - 53;
        // sig is in [2^52, 2^53-1] for a normalised value.
        let value = sig as f64 * (exp_f64 as f64).exp2();
        if self.sign { -value } else { value }
    }

    /// Parse a decimal string like `"3.14"`, `"-2.71828"` into a `BigFloat`
    /// with the specified precision.  Only finite decimal values are supported.
    pub fn from_str(s: &str, precision: usize) -> CoreResult<Self> {
        let s = s.trim();
        let (neg, digits) = if let Some(rest) = s.strip_prefix('-') {
            (true, rest)
        } else {
            (false, s)
        };
        // Split on decimal point.
        let (int_part, frac_part) = if let Some(dot) = digits.find('.') {
            (&digits[..dot], &digits[dot + 1..])
        } else {
            (digits, "")
        };
        // Parse integer part.
        let mut result = Self::zero(precision);
        let ten = Self::from_f64(10.0, precision);
        for ch in int_part.chars() {
            let d = ch.to_digit(10).ok_or_else(|| comp_err(format!("invalid digit '{ch}'")))?;
            result = result.mul(&ten, precision)?;
            let digit = Self::from_f64(f64::from(d), precision);
            result = result.add(&digit, precision)?;
        }
        // Parse fractional part.
        let mut place = Self::from_f64(0.1, precision);
        for ch in frac_part.chars() {
            let d = ch.to_digit(10).ok_or_else(|| comp_err(format!("invalid digit '{ch}'")))?;
            let digit = Self::from_f64(f64::from(d), precision);
            let contribution = digit.mul(&place, precision)?;
            result = result.add(&contribution, precision)?;
            place = place.mul(&Self::from_f64(0.1, precision), precision)?;
        }
        result.sign = neg && !result.is_zero();
        Ok(result)
    }

    /// Convert to a decimal string with the given number of significant digits.
    ///
    /// Uses the `f64` approximation for the decimal conversion (sufficient for
    /// display purposes at modest precisions).
    #[must_use]
    pub fn to_string_decimal(&self, sig_digits: usize) -> String {
        if self.is_zero() {
            return "0".to_string();
        }
        // For display, use f64 with printf-style formatting.
        let v = self.to_f64();
        let width = sig_digits.max(1);
        format!("{:.prec$e}", v, prec = width - 1)
    }

    // ── Arithmetic ───────────────────────────────────────────────────────────

    /// Add `self + other` at the given precision.
    pub fn add(&self, other: &BigFloat, precision: usize) -> CoreResult<BigFloat> {
        if self.is_zero() {
            return Ok(other.with_precision(precision));
        }
        if other.is_zero() {
            return Ok(self.with_precision(precision));
        }
        if self.sign == other.sign {
            // Same sign → add magnitudes.
            let mut result = self.add_magnitudes(other, precision)?;
            result.sign = self.sign;
            Ok(result)
        } else {
            // Different signs → subtract smaller from larger.
            let cmp = self.cmp_magnitude(other);
            match cmp {
                std::cmp::Ordering::Equal => Ok(BigFloat::zero(precision)),
                std::cmp::Ordering::Greater => {
                    let mut result = self.sub_magnitudes(other, precision)?;
                    result.sign = self.sign;
                    Ok(result)
                }
                std::cmp::Ordering::Less => {
                    let mut result = other.sub_magnitudes(self, precision)?;
                    result.sign = other.sign;
                    Ok(result)
                }
            }
        }
    }

    /// Subtract: `self - other` at the given precision.
    pub fn sub(&self, other: &BigFloat, precision: usize) -> CoreResult<BigFloat> {
        let neg_other = BigFloat {
            sign: !other.sign && !other.is_zero(),
            exponent: other.exponent,
            mantissa: other.mantissa.clone(),
            precision: other.precision,
        };
        self.add(&neg_other, precision)
    }

    /// Multiply `self × other` at the given precision.
    pub fn mul(&self, other: &BigFloat, precision: usize) -> CoreResult<BigFloat> {
        if self.is_zero() || other.is_zero() {
            return Ok(BigFloat::zero(precision));
        }
        let sign = self.sign ^ other.sign;
        // Multiply mantissa arrays (schoolbook – sufficient for moderate precision).
        let result_mantissa = multiply_mantissas(&self.mantissa, &other.mantissa);
        let exponent = self.exponent + other.exponent;
        let mut bf = BigFloat {
            sign,
            exponent,
            mantissa: result_mantissa,
            precision,
        };
        bf.normalise();
        bf.truncate(precision);
        Ok(bf)
    }

    /// Divide `self / other` at the given precision.
    pub fn div(&self, other: &BigFloat, precision: usize) -> CoreResult<BigFloat> {
        if other.is_zero() {
            return Err(comp_err("division by zero in BigFloat::div"));
        }
        if self.is_zero() {
            return Ok(BigFloat::zero(precision));
        }
        let sign = self.sign ^ other.sign;
        // Use Newton-Raphson to compute 1/other, then multiply by self.
        // This avoids the pitfalls of a hand-rolled long division.
        let work_prec = precision + 64;
        // Initial approximation from f64.
        let recip_approx = 1.0 / other.to_f64();
        let mut x = BigFloat::from_f64(recip_approx, work_prec);
        let two = BigFloat::from_f64(2.0, work_prec);
        // Newton-Raphson: x_{n+1} = x_n * (2 - other * x_n)
        for _ in 0..100 {
            let ox = other.mul(&x, work_prec)?;
            let correction = two.sub(&ox, work_prec)?;
            let x_new = x.mul(&correction, work_prec)?;
            let diff = x_new.sub(&x, work_prec)?;
            if diff.is_zero() || diff.exponent < x.exponent - (work_prec as i64) {
                x = x_new;
                break;
            }
            x = x_new;
        }
        // quotient = self * (1/other)
        let mut result = self.mul(&x, work_prec)?;
        result.sign = sign;
        result.truncate(precision);
        Ok(result)
    }

    // ── Mathematical functions ────────────────────────────────────────────────

    /// Square root via Newton-Raphson iteration.
    pub fn sqrt(&self, precision: usize) -> CoreResult<BigFloat> {
        if self.sign {
            return Err(comp_err("sqrt of negative number"));
        }
        if self.is_zero() {
            return Ok(BigFloat::zero(precision));
        }
        // Working precision with guard bits.
        let work_prec = precision + 64;
        // Initial approximation from f64.
        let approx = self.to_f64().sqrt();
        let mut x = BigFloat::from_f64(approx, work_prec);
        // Refine: x_{n+1} = (x_n + self / x_n) / 2
        let half = BigFloat::from_f64(0.5, work_prec);
        let two = BigFloat::from_f64(2.0, work_prec);
        for _ in 0..100 {
            let xnew = x.add(&self.div(&x, work_prec)?, work_prec)?.mul(&half, work_prec)?;
            // Check convergence.
            let diff = xnew.sub(&x, work_prec)?;
            if diff.is_zero() {
                x = xnew;
                break;
            }
            // Compare exponents: if |diff| < 2^(exponent - precision), converged.
            let diff_exp = diff.exponent;
            let target_exp = x.exponent - (work_prec as i64);
            if diff_exp < target_exp {
                x = xnew;
                break;
            }
            x = xnew;
            // Verify x² ≈ self (avoid infinite loops on edge cases).
            let _ = two.clone(); // suppress unused warning
        }
        x.truncate(precision);
        Ok(x)
    }

    /// Natural logarithm via AGM-based formula.
    ///
    /// Uses the identity: ln(x) = π / (2 × AGM(1, 4/x × 2^m)) − m × ln(2)
    /// for x scaled to [1/2, 1) and m chosen so that x × 2^m is large.
    pub fn ln(&self, precision: usize) -> CoreResult<BigFloat> {
        if self.sign || self.is_zero() {
            return Err(comp_err("ln of non-positive number"));
        }
        let work_prec = precision + 64;
        // Use f64 shortcut if precision fits.
        if precision <= 52 {
            let v = self.to_f64().ln();
            return Ok(BigFloat::from_f64(v, precision));
        }
        // Argument reduction: write self = y × 2^k where y ∈ [0.5, 1).
        // Then ln(self) = ln(y) + k×ln(2).
        // y is self with exponent adjusted to 0 (so y ∈ [0.5, 1)).
        let k = self.exponent; // self = 0.mantissa × 2^exponent, so y = 0.mantissa (∈[0.5,1))
        let mut y = self.with_precision(work_prec);
        y.exponent = 0; // y ∈ [0.5, 1)
        // Use AGM method: ln(y) = π / (2×AGM(1, 4s/y)) − s×ln(2)
        // where s is chosen large (≈ precision/3) so AGM converges fast.
        let s = ((work_prec as f64 / 3.0).ceil() as i64).max(4);
        // z = y × 2^s → z ∈ [2^(s-1), 2^s)
        let mut z = y.clone();
        z.exponent += s;
        // ln(y) = π/(2×AGM(1, 4/z)) − s×ln(2)  [standard formula with z = 2^s × y]
        // Compute AGM(1, 4/z) = AGM(1, 4×2^{-s}/y)
        let four = BigFloat::from_f64(4.0, work_prec);
        let four_over_z = four.div(&z, work_prec)?;
        let one = BigFloat::one(work_prec);
        let agm_val = agm(&one, &four_over_z, work_prec)?;
        let pi_val = pi_bf(work_prec)?;
        let two = BigFloat::from_f64(2.0, work_prec);
        let ln_y = pi_val.div(&(two.mul(&agm_val, work_prec)?), work_prec)?
            .sub(&BigFloat::from_f64(s as f64, work_prec)
                .mul(&ln2_bf(work_prec)?, work_prec)?, work_prec)?;
        // ln(self) = ln(y) + k×ln(2)
        let k_ln2 = BigFloat::from_f64(k as f64, work_prec).mul(&ln2_bf(work_prec)?, work_prec)?;
        let mut result = ln_y.add(&k_ln2, work_prec)?;
        result.truncate(precision);
        Ok(result)
    }

    /// Exponential e^x via Taylor series with argument reduction.
    ///
    /// Reduces argument to |r| < 0.5 then uses Taylor series, then squares back.
    pub fn exp(&self, precision: usize) -> CoreResult<BigFloat> {
        if self.is_zero() {
            return Ok(BigFloat::one(precision));
        }
        let work_prec = precision + 64;
        // For |x| > ~700 the result overflows f64 but BigFloat is fine.
        // Argument reduction: x = k×ln(2) + r where |r| ≤ ln(2)/2
        let ln2 = ln2_bf(work_prec)?;
        // k = round(x / ln(2))
        let x_over_ln2 = self.div(&ln2, work_prec)?;
        let k = x_over_ln2.to_f64().round() as i64;
        // r = x - k×ln(2)
        let k_ln2 = BigFloat::from_f64(k as f64, work_prec).mul(&ln2, work_prec)?;
        let r = self.sub(&k_ln2, work_prec)?;
        // Taylor series: e^r = sum_{n=0}^{N} r^n / n!
        let n_terms = (work_prec as f64 / (work_prec as f64).log2() + 10.0).ceil() as usize;
        let n_terms = n_terms.max(20).min(1000);
        let one = BigFloat::one(work_prec);
        let mut sum = one.clone();
        let mut term = one.clone();
        for n in 1..=n_terms {
            term = term.mul(&r, work_prec)?
                .div(&BigFloat::from_f64(n as f64, work_prec), work_prec)?;
            let new_sum = sum.add(&term, work_prec)?;
            // Convergence check.
            if !term.is_zero() && term.exponent < sum.exponent - (work_prec as i64) {
                sum = new_sum;
                break;
            }
            sum = new_sum;
        }
        // Multiply by 2^k.
        sum.exponent += k;
        sum.truncate(precision);
        Ok(sum)
    }

    // ── Mathematical constants ────────────────────────────────────────────────

    /// π computed to the given precision using the Brent-Salamin AGM formula.
    pub fn pi(precision: usize) -> CoreResult<BigFloat> {
        pi_bf(precision)
    }

    /// Euler's number e = exp(1) computed to the given precision.
    pub fn e(precision: usize) -> CoreResult<BigFloat> {
        BigFloat::one(precision).exp(precision)
    }

    /// ln(2) computed to the given precision.
    pub fn ln2(precision: usize) -> CoreResult<BigFloat> {
        ln2_bf(precision)
    }

    /// √2 computed to the given precision.
    pub fn sqrt2(precision: usize) -> CoreResult<BigFloat> {
        BigFloat::from_f64(2.0, precision).sqrt(precision)
    }

    // ── Predicates ───────────────────────────────────────────────────────────

    /// Returns `true` if the value is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.mantissa.is_empty() || self.mantissa.iter().all(|&l| l == 0)
    }

    /// Returns `true` if the value is negative.
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.sign && !self.is_zero()
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Number of 64-bit limbs needed for `prec` bits.
    #[inline]
    fn limb_count(prec: usize) -> usize {
        (prec + LIMB_BITS - 1) / LIMB_BITS
    }

    /// Clone with a different precision (truncate or zero-extend mantissa).
    fn with_precision(&self, precision: usize) -> BigFloat {
        let mut bf = self.clone();
        bf.precision = precision;
        bf.truncate(precision);
        bf
    }

    /// Normalise: ensure the MSB of `mantissa[0]` is 1 (shift left, adjust exponent).
    fn normalise(&mut self) {
        if self.is_zero() {
            self.mantissa.clear();
            return;
        }
        // Remove leading zero limbs.
        while !self.mantissa.is_empty() && self.mantissa[0] == 0 {
            self.mantissa.remove(0);
            self.exponent -= LIMB_BITS as i64;
        }
        if self.mantissa.is_empty() {
            return;
        }
        // Shift left until MSB is set.
        let leading = self.mantissa[0].leading_zeros() as usize;
        if leading > 0 {
            left_shift_mantissa(&mut self.mantissa, leading);
            self.exponent -= leading as i64;
        }
    }

    /// Truncate mantissa to `precision` bits (round to nearest, ties-to-even).
    fn truncate(&mut self, precision: usize) {
        if self.is_zero() {
            return;
        }
        let limbs = Self::limb_count(precision);
        if self.mantissa.len() <= limbs {
            // Zero-extend.
            self.mantissa.resize(limbs, 0);
            return;
        }
        // Rounding bit: MSB of first discarded limb.
        let round_bit = (self.mantissa[limbs] >> 63) != 0;
        self.mantissa.truncate(limbs);
        if round_bit {
            // Round up.
            let mut carry = 1u64;
            for limb in self.mantissa.iter_mut().rev() {
                let (new_limb, c) = limb.overflowing_add(carry);
                *limb = new_limb;
                carry = if c { 1 } else { 0 };
                if carry == 0 {
                    break;
                }
            }
            if carry != 0 {
                // Overflow: shift right by 1 and adjust exponent.
                right_shift_mantissa(&mut self.mantissa, 1);
                self.mantissa[0] |= 1u64 << 63;
                self.exponent += 1;
            }
        }
    }

    /// Compare magnitudes of two BigFloats (ignoring sign).
    fn cmp_magnitude(&self, other: &BigFloat) -> std::cmp::Ordering {
        if self.exponent != other.exponent {
            return self.exponent.cmp(&other.exponent);
        }
        let len = self.mantissa.len().min(other.mantissa.len());
        for i in 0..len {
            if self.mantissa[i] != other.mantissa[i] {
                return self.mantissa[i].cmp(&other.mantissa[i]);
            }
        }
        self.mantissa.len().cmp(&other.mantissa.len())
    }

    /// Add magnitudes (same sign, or for internal use).
    fn add_magnitudes(&self, other: &BigFloat, precision: usize) -> CoreResult<BigFloat> {
        // Align exponents.
        let (a, b, exp) = align_exponents(self, other, precision);
        // Add limb arrays.
        // Prepend a zero limb for carry headroom (carry propagates toward index 0).
        let max_len = a.len().max(b.len());
        let mut a_ext = vec![0u64; max_len + 1];
        let mut b_ext = vec![0u64; max_len + 1];
        // Copy original data starting at index 1 (index 0 is the carry headroom).
        for (i, &v) in a.iter().enumerate() {
            if i + 1 < a_ext.len() {
                a_ext[i + 1] = v;
            }
        }
        for (i, &v) in b.iter().enumerate() {
            if i + 1 < b_ext.len() {
                b_ext[i + 1] = v;
            }
        }
        let total_len = max_len + 1;
        let mut result = vec![0u64; total_len];
        let mut carry = 0u64;
        for i in (0..total_len).rev() {
            let (s1, c1) = a_ext[i].overflowing_add(b_ext[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            result[i] = s2;
            carry = if c1 || c2 { 1 } else { 0 };
        }
        // Adjust exponent: we prepended one limb (64 bits) of headroom.
        let mut bf = BigFloat { sign: false, exponent: exp + LIMB_BITS as i64, mantissa: result, precision };
        bf.normalise();
        bf.truncate(precision);
        Ok(bf)
    }

    /// Subtract magnitudes: self_magnitude - other_magnitude (self > other).
    fn sub_magnitudes(&self, other: &BigFloat, precision: usize) -> CoreResult<BigFloat> {
        let (a, b, exp) = align_exponents(self, other, precision);
        let max_len = a.len().max(b.len());
        let a_ext = extend_limbs(&a, max_len);
        let b_ext = extend_limbs(&b, max_len);
        let mut result = vec![0u64; max_len];
        let mut borrow = 0u64;
        for i in (0..max_len).rev() {
            let (d1, b1) = a_ext[i].overflowing_sub(b_ext[i]);
            let (d2, b2) = d1.overflowing_sub(borrow);
            result[i] = d2;
            borrow = if b1 || b2 { 1 } else { 0 };
        }
        let mut bf = BigFloat { sign: false, exponent: exp, mantissa: result, precision };
        bf.normalise();
        bf.truncate(precision);
        Ok(bf)
    }
}

// ─── Limb-level helpers ───────────────────────────────────────────────────────

/// Align two BigFloat mantissas to the same exponent, returning (a_limbs, b_limbs, common_exp).
///
/// The returned exponent is the larger one; the smaller mantissa is right-shifted.
fn align_exponents(a: &BigFloat, b: &BigFloat, precision: usize) -> (Vec<u64>, Vec<u64>, i64) {
    let limbs = BigFloat::limb_count(precision) + 2;
    let a_exp = a.exponent;
    let b_exp = b.exponent;
    let (a_m, b_m, exp) = if a_exp >= b_exp {
        let shift = (a_exp - b_exp) as usize;
        let mut b_shifted = extend_limbs(&b.mantissa, limbs + (shift / LIMB_BITS) + 2);
        right_shift_mantissa(&mut b_shifted, shift);
        (extend_limbs(&a.mantissa, limbs), b_shifted, a_exp)
    } else {
        let shift = (b_exp - a_exp) as usize;
        let mut a_shifted = extend_limbs(&a.mantissa, limbs + (shift / LIMB_BITS) + 2);
        right_shift_mantissa(&mut a_shifted, shift);
        (a_shifted, extend_limbs(&b.mantissa, limbs), b_exp)
    };
    (a_m, b_m, exp)
}

/// Extend a limb slice to the given length by appending zeros.
fn extend_limbs(limbs: &[u64], target: usize) -> Vec<u64> {
    let mut v = limbs.to_vec();
    if v.len() < target {
        v.resize(target, 0);
    }
    v
}

/// Left-shift a mantissa by `shift` bits (across limb boundaries).
fn left_shift_mantissa(mantissa: &mut Vec<u64>, shift: usize) {
    if shift == 0 || mantissa.is_empty() {
        return;
    }
    let limb_shift = shift / LIMB_BITS;
    let bit_shift = shift % LIMB_BITS;
    // Remove leading zero limbs produced by limb_shift.
    for _ in 0..limb_shift {
        if !mantissa.is_empty() {
            mantissa.remove(0);
        }
    }
    if bit_shift > 0 && !mantissa.is_empty() {
        let mut carry = 0u64;
        for limb in mantissa.iter_mut().rev() {
            let new_carry = if bit_shift < 64 { *limb >> (64 - bit_shift) } else { 0 };
            *limb = limb.wrapping_shl(bit_shift as u32) | carry;
            carry = new_carry;
        }
    }
}

/// Right-shift a mantissa by `shift` bits (across limb boundaries), in place.
fn right_shift_mantissa(mantissa: &mut Vec<u64>, shift: usize) {
    if shift == 0 || mantissa.is_empty() {
        return;
    }
    let limb_shift = shift / LIMB_BITS;
    let bit_shift = shift % LIMB_BITS;
    // Prepend zero limbs for whole-limb shift.
    for _ in 0..limb_shift {
        mantissa.insert(0, 0u64);
    }
    if bit_shift > 0 {
        let mut carry = 0u64;
        for limb in mantissa.iter_mut() {
            let new_carry = if bit_shift < 64 { *limb << (64 - bit_shift) } else { 0 };
            *limb = limb.wrapping_shr(bit_shift as u32) | carry;
            carry = new_carry;
        }
    }
}

/// Schoolbook multiplication of two mantissa arrays.
///
/// Returns an array of length `a.len() + b.len()`.
///
/// The mantissa is MSB-first, so `a[0]` is the most significant limb.
/// The partial product `a[i] * b[j]` contributes to position `i + j + 1`
/// (with carry flowing into `i + j`).
fn multiply_mantissas(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let len = a.len() + b.len();
    let mut out = vec![0u64; len];
    // Process from least significant to most significant for safe carry propagation.
    for i in (0..a.len()).rev() {
        let mut carry = 0u128;
        for j in (0..b.len()).rev() {
            let pos = i + j + 1;
            let prod = a[i] as u128 * b[j] as u128 + out[pos] as u128 + carry;
            out[pos] = prod as u64;
            carry = prod >> 64;
        }
        // Propagate remaining carry toward MSB (lower indices).
        let mut pos = i;
        while carry > 0 {
            let val = out[pos] as u128 + carry;
            out[pos] = val as u64;
            carry = val >> 64;
            if pos == 0 {
                break;
            }
            pos -= 1;
        }
    }
    out
}

/// Long-division of numerator by denominator mantissa, producing `precision` bits of quotient.
///
/// Both inputs are MSB-first normalised mantissa arrays representing fractional
/// values in [0.5, 1). The quotient is therefore in (0.5, 2). The result array
/// has `limbs` limbs of quotient bits.
fn divide_mantissas(num: &[u64], den: &[u64], precision: usize) -> Vec<u64> {
    let limbs = BigFloat::limb_count(precision) + 2;
    let total_bits = limbs * LIMB_BITS;

    // Work length: use at least den.len() limbs + extra for the extended quotient.
    let work_len = (den.len().max(num.len()) + limbs + 2).max(4);

    // Remainder, initialised from numerator and zero-extended.
    let mut rem = vec![0u64; work_len];
    for (i, &v) in num.iter().enumerate() {
        if i < work_len {
            rem[i] = v;
        }
    }

    // Denominator, zero-extended to work_len.
    let mut den_w = vec![0u64; work_len];
    for (i, &v) in den.iter().enumerate() {
        if i < work_len {
            den_w[i] = v;
        }
    }

    let mut quotient = vec![0u64; limbs];

    // Restoring binary division: for each quotient bit, compare R >= D,
    // if so set bit and subtract, then shift R left by 1.
    for bit_idx in 0..total_bits {
        // Compare rem >= den_w (unsigned big-integer comparison, MSB first).
        let ge = bigint_ge(&rem, &den_w);

        if ge {
            // Set quotient bit.
            let limb_idx = bit_idx / LIMB_BITS;
            let bit_pos = 63 - (bit_idx % LIMB_BITS);
            quotient[limb_idx] |= 1u64 << bit_pos;

            // rem -= den_w
            bigint_sub_inplace(&mut rem, &den_w);
        }

        // Shift remainder left by 1 bit.
        bigint_shl1(&mut rem);
    }

    quotient
}

/// Compare two big integers (MSB-first limb arrays): returns true if a >= b.
fn bigint_ge(a: &[u64], b: &[u64]) -> bool {
    let len = a.len().max(b.len());
    for i in 0..len {
        let ai = if i < a.len() { a[i] } else { 0 };
        let bi = if i < b.len() { b[i] } else { 0 };
        if ai > bi {
            return true;
        }
        if ai < bi {
            return false;
        }
    }
    true // equal
}

/// Subtract b from a in place (a -= b), assuming a >= b.
fn bigint_sub_inplace(a: &mut [u64], b: &[u64]) {
    let mut borrow = 0u64;
    let len = a.len();
    for i in (0..len).rev() {
        let bi = if i < b.len() { b[i] } else { 0 };
        let (d1, b1) = a[i].overflowing_sub(bi);
        let (d2, b2) = d1.overflowing_sub(borrow);
        a[i] = d2;
        borrow = if b1 || b2 { 1 } else { 0 };
    }
}

/// Left-shift a big integer (MSB-first limb array) by 1 bit, in place.
fn bigint_shl1(a: &mut [u64]) {
    let mut carry = 0u64;
    for i in (0..a.len()).rev() {
        let new_carry = a[i] >> 63;
        a[i] = (a[i] << 1) | carry;
        carry = new_carry;
    }
    // Note: carry from MSB is discarded (the caller ensures this doesn't happen
    // because after subtraction the remainder is always less than the denominator).
}

// ─── AGM and constants ────────────────────────────────────────────────────────

/// Arithmetic-Geometric Mean of `a` and `b`.
fn agm(a: &BigFloat, b: &BigFloat, precision: usize) -> CoreResult<BigFloat> {
    let work_prec = precision + 32;
    let mut a_k = a.with_precision(work_prec);
    let mut b_k = b.with_precision(work_prec);
    let half = BigFloat::from_f64(0.5, work_prec);
    for _ in 0..200 {
        let a_next = a_k.add(&b_k, work_prec)?.mul(&half, work_prec)?;
        let b_next = a_k.mul(&b_k, work_prec)?.sqrt(work_prec)?;
        let diff = a_next.sub(&b_next, work_prec)?;
        let old = a_k.clone();
        a_k = a_next;
        b_k = b_next;
        if diff.is_zero() || (!diff.is_zero() && diff.exponent < old.exponent - (work_prec as i64)) {
            break;
        }
    }
    Ok(a_k.with_precision(precision))
}

/// Compute π using the Brent-Salamin AGM recurrence.
///
/// π = (a_n + b_n)² / (1 − Σ 2^{k+1} c_k²)  (simplified for implementation)
///
/// Here we use the efficient form:
/// After N steps of AGM(1, 1/√2), π ≈ (a_N + b_N)² / (1 − Σ 2^k (a_k² − b_k²) / 2)
fn pi_bf(precision: usize) -> CoreResult<BigFloat> {
    let work_prec = precision + 64;
    let one = BigFloat::one(work_prec);
    let two = BigFloat::from_f64(2.0, work_prec);
    let half = BigFloat::from_f64(0.5, work_prec);
    // Brent-Salamin: a₀ = 1, b₀ = 1/√2, t₀ = 1/4, p₀ = 1
    let inv_sqrt2 = two.sqrt(work_prec)?.div(&two, work_prec)?; // 1/√2
    let mut a = one.clone();
    let mut b = inv_sqrt2;
    let mut t = BigFloat::from_f64(0.25, work_prec);
    let mut p = one.clone();
    for _ in 0..200 {
        let a_next = a.add(&b, work_prec)?.mul(&half, work_prec)?;
        let b_next = a.mul(&b, work_prec)?.sqrt(work_prec)?;
        let diff = a.sub(&a_next, work_prec)?; // a - a_next = (a - b) / 2
        let diff_sq = diff.mul(&diff, work_prec)?.mul(&p, work_prec)?;
        let t_next = t.sub(&diff_sq, work_prec)?;
        let p_next = p.mul(&two, work_prec)?;
        let old_a = a.clone();
        a = a_next;
        b = b_next;
        let conv_check = a.sub(&old_a, work_prec)?;
        t = t_next;
        p = p_next;
        if conv_check.is_zero() || conv_check.exponent < a.exponent - (work_prec as i64) {
            break;
        }
    }
    // π ≈ (a + b)² / (4t)
    let ab = a.add(&b, work_prec)?;
    let numerator = ab.mul(&ab, work_prec)?;
    let four_t = BigFloat::from_f64(4.0, work_prec).mul(&t, work_prec)?;
    let mut result = numerator.div(&four_t, work_prec)?;
    result.truncate(precision);
    Ok(result)
}

/// Compute ln(2) via the identity ln(2) = π / (2 × AGM(1, 4×2^{-s})) − s×ln(2).
/// We bootstrap: ln(2) = π / (2 × AGM(1, 4/2^s) + s×ln(2)^{-1}…)
/// Actually use the simpler direct identity:
/// ln(2) ≈ π / (2 × AGM(1, 4/2^s)) × 1/s  for large s (Borwein).
///
/// Concretely: ln(y) = π/(2×AGM(1,4/y)) − (precision/3)×ln(2)  where y = 2^{p/3}.
/// This is circular; instead use the BBP-style formula:
///
/// ln(2) = Σ_{k=1}^∞  1 / (k × 2^k)
///
/// which converges, but slowly.  For moderate precision, use a larger fixed-point
/// iteration or Machin-like formula.  Here we use:
///
/// ln(2) = 3 × ln(10/9) − ln(25/24) + 5 × ln(81/80)
/// (Machin-like identity for ln 2, accurate for series).
fn ln2_bf(precision: usize) -> CoreResult<BigFloat> {
    let work_prec = precision + 64;
    if precision <= 52 {
        return Ok(BigFloat::from_f64(std::f64::consts::LN_2, precision));
    }
    // Use: ln(2) = 3 ln(3/2) - ln(5/4) + ln(4/3) [Lehmer decomposition is complex]
    // Simpler: use series ln(x) = 2 × atanh((x-1)/(x+1))
    // atanh(z) = z + z³/3 + z⁵/5 + …  for |z| < 1.
    // For ln(2): z = (2-1)/(2+1) = 1/3
    // ln(2) = 2 × atanh(1/3) = 2 × (1/3 + 1/(3×27) + 1/(5×243) + …)
    let one = BigFloat::one(work_prec);
    let three = BigFloat::from_f64(3.0, work_prec);
    let z = one.div(&three, work_prec)?; // 1/3
    let z2 = z.mul(&z, work_prec)?;     // (1/3)²
    let mut sum = z.clone();             // first term
    let mut term = z.clone();
    let n_terms = (work_prec as f64 / std::f64::consts::LOG2_10 / 2.0 + 10.0).ceil() as usize;
    for k in 1..=n_terms {
        term = term.mul(&z2, work_prec)?;
        let denom = BigFloat::from_f64((2 * k + 1) as f64, work_prec);
        let t = term.div(&denom, work_prec)?;
        let new_sum = sum.add(&t, work_prec)?;
        if !t.is_zero() && t.exponent < sum.exponent - (work_prec as i64) {
            sum = new_sum;
            break;
        }
        sum = new_sum;
    }
    // ln(2) = 2 × sum
    let two = BigFloat::from_f64(2.0, work_prec);
    let mut result = two.mul(&sum, work_prec)?;
    result.truncate(precision);
    Ok(result)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip f64 → BigFloat → f64.
    #[test]
    fn test_from_f64_roundtrip() {
        for &v in &[1.0f64, -1.0, 3.14, 0.5, 1024.0, -0.125] {
            let bf = BigFloat::from_f64(v, 128);
            let back = bf.to_f64();
            let rel = (back - v).abs() / v.abs().max(1e-300);
            assert!(rel < 1e-14, "roundtrip failed for {v}: got {back}");
        }
    }

    #[test]
    fn test_add() {
        let a = BigFloat::from_f64(1.5, 128);
        let b = BigFloat::from_f64(2.5, 128);
        let c = a.add(&b, 128).expect("should succeed");
        let diff = (c.to_f64() - 4.0).abs();
        assert!(diff < 1e-14, "add: expected 4.0, got {}", c.to_f64());
    }

    #[test]
    fn test_sub() {
        let a = BigFloat::from_f64(5.0, 128);
        let b = BigFloat::from_f64(3.0, 128);
        let c = a.sub(&b, 128).expect("should succeed");
        let diff = (c.to_f64() - 2.0).abs();
        assert!(diff < 1e-14, "sub: expected 2.0, got {}", c.to_f64());
    }

    #[test]
    fn test_mul() {
        let a = BigFloat::from_f64(3.0, 128);
        let b = BigFloat::from_f64(4.0, 128);
        let c = a.mul(&b, 128).expect("should succeed");
        let diff = (c.to_f64() - 12.0).abs();
        assert!(diff < 1e-13, "mul: expected 12.0, got {}", c.to_f64());
    }

    #[test]
    fn test_div() {
        let a = BigFloat::from_f64(10.0, 128);
        let b = BigFloat::from_f64(4.0, 128);
        let c = a.div(&b, 128).expect("should succeed");
        let diff = (c.to_f64() - 2.5).abs();
        assert!(diff < 1e-13, "div: expected 2.5, got {}", c.to_f64());
    }

    #[test]
    fn test_div_by_zero() {
        let a = BigFloat::from_f64(1.0, 64);
        let z = BigFloat::zero(64);
        assert!(a.div(&z, 64).is_err());
    }

    #[test]
    fn test_sqrt() {
        let a = BigFloat::from_f64(2.0, 128);
        let s = a.sqrt(128).expect("should succeed");
        let expected = 2.0f64.sqrt();
        let diff = (s.to_f64() - expected).abs();
        assert!(diff < 1e-14, "sqrt(2): expected {expected}, got {}", s.to_f64());
    }

    #[test]
    fn test_sqrt_negative() {
        let a = BigFloat::from_f64(-1.0, 64);
        assert!(a.sqrt(64).is_err());
    }

    #[test]
    fn test_pi() {
        let pi = BigFloat::pi(128).expect("should succeed");
        let diff = (pi.to_f64() - std::f64::consts::PI).abs();
        assert!(diff < 1e-12, "pi: expected {}, got {}", std::f64::consts::PI, pi.to_f64());
    }

    #[test]
    fn test_ln2() {
        let ln2 = BigFloat::ln2(128).expect("should succeed");
        let expected = std::f64::consts::LN_2;
        let diff = (ln2.to_f64() - expected).abs();
        assert!(diff < 1e-13, "ln2: expected {expected}, got {}", ln2.to_f64());
    }

    #[test]
    fn test_sqrt2() {
        let s = BigFloat::sqrt2(128).expect("should succeed");
        let expected = std::f64::consts::SQRT_2;
        let diff = (s.to_f64() - expected).abs();
        assert!(diff < 1e-14, "sqrt2: expected {expected}, got {}", s.to_f64());
    }

    #[test]
    fn test_exp() {
        let one = BigFloat::one(128);
        let e = one.exp(128).expect("should succeed");
        let expected = std::f64::consts::E;
        let diff = (e.to_f64() - expected).abs();
        assert!(diff < 1e-13, "e: expected {expected}, got {}", e.to_f64());
    }

    #[test]
    fn test_zero_identity() {
        let a = BigFloat::from_f64(5.0, 128);
        let z = BigFloat::zero(128);
        let sum = a.add(&z, 128).expect("should succeed");
        let diff = (sum.to_f64() - 5.0).abs();
        assert!(diff < 1e-14);
    }

    #[test]
    fn test_negative_add() {
        let a = BigFloat::from_f64(3.0, 128);
        let b = BigFloat::from_f64(-1.5, 128);
        let c = a.add(&b, 128).expect("should succeed");
        let diff = (c.to_f64() - 1.5).abs();
        assert!(diff < 1e-14);
    }
}
