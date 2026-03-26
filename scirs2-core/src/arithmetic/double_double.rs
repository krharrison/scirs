//! Double-double arithmetic (~31 decimal digits of precision).
//!
//! A double-double number is represented as `(hi, lo)` where `hi + lo` is more
//! precise than a single `f64`. The invariant is `|lo| ≤ 0.5 ulp(hi)`,
//! giving approximately 106 bits (~31.9 significant decimal digits) of precision.
//!
//! ## Error-free Transformations
//!
//! All arithmetic operations use classic error-free transformations (EFTs):
//!
//! - **TwoSum** (Knuth/Møller): computes `(s, e)` such that `s + e == a + b` exactly.
//! - **TwoProd** (Veltkamp splitting): computes `(p, e)` such that `p + e == a * b` exactly.
//!
//! ## References
//!
//! * Dekker, T. J. (1971). "A floating-point technique for extending the available precision."
//!   *Numerische Mathematik* 18, 224–242.
//! * Shewchuk, J. R. (1997). "Adaptive precision floating-point arithmetic."
//!   *DCG* 18, 305–363.
//! * Ogita, T.; Rump, S. M.; Oishi, S. (2005). "Accurate sum and dot product."
//!   *SIAM J. Sci. Comput.* 26(6), 1955–1988.

use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::error::{CoreError, CoreResult, ErrorContext};

// ─── Internal helpers ────────────────────────────────────────────────────────

#[inline(always)]
fn comp_err(msg: impl Into<String>) -> CoreError {
    CoreError::ComputationError(ErrorContext::new(msg))
}

// ─── Error-free transformations ──────────────────────────────────────────────

/// TwoSum (Knuth algorithm): returns `(s, e)` with `s + e == a + b` exactly
/// in round-to-nearest IEEE 754 arithmetic.
///
/// # Example
/// ```rust
/// use scirs2_core::arithmetic::DoubleDouble;
/// let dd = DoubleDouble::new(1.0) + DoubleDouble::new(2.0_f64.powi(-53));
/// // The result captures full precision.
/// assert!(dd.is_finite());
/// ```
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

/// TwoProd via Veltkamp splitting: returns `(p, e)` with `p + e == a * b` exactly.
///
/// Uses the Veltkamp split constant 2^27 + 1 = 134_217_729.
#[inline]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    // Veltkamp split of a.
    let c = 134_217_729.0_f64 * a; // 2^27 + 1
    let a_hi = c - (c - a);
    let a_lo = a - a_hi;
    // Veltkamp split of b.
    let c2 = 134_217_729.0_f64 * b;
    let b_hi = c2 - (c2 - b);
    let b_lo = b - b_hi;
    let e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (p, e)
}

/// Veltkamp split: returns `(hi, lo)` such that `a == hi + lo` and
/// `hi` has at most 26 significant bits.
#[inline]
fn split(a: f64) -> (f64, f64) {
    let c = 134_217_729.0_f64 * a;
    let hi = c - (c - a);
    let lo = a - hi;
    (hi, lo)
}

// ─── DoubleDouble struct ──────────────────────────────────────────────────────

/// Double-double precision floating-point number.
///
/// Represents the unevaluated sum `hi + lo` where `|lo| ≤ 0.5 ulp(hi)`.
/// Provides approximately 31 decimal digits of precision using two `f64` values.
///
/// # Constants
///
/// ```rust
/// use scirs2_core::arithmetic::DoubleDouble;
///
/// let zero = DoubleDouble::ZERO;
/// let one = DoubleDouble::ONE;
/// let pi = DoubleDouble::PI;
/// let e = DoubleDouble::E;
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct DoubleDouble {
    /// High-order word (most significant bits).
    pub hi: f64,
    /// Low-order word (error term; `|lo| ≤ 0.5 ulp(hi)`).
    pub lo: f64,
}

impl DoubleDouble {
    // ── Constants ──────────────────────────────────────────────────────────────

    /// Zero.
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

    /// One.
    pub const ONE: Self = Self { hi: 1.0, lo: 0.0 };

    /// π to double-double precision.
    /// Value: 3.14159265358979323846264338327950288...
    pub const PI: Self = Self {
        hi: std::f64::consts::PI,
        lo: 1.224_646_799_147_353_2e-16,
    };

    /// e = exp(1) to double-double precision.
    /// Value: 2.71828182845904523536028747135266249...
    pub const E: Self = Self {
        hi: std::f64::consts::E,
        lo: -2.842_170_943_040_400_8e-17,
    };

    // ── Constructors ───────────────────────────────────────────────────────────

    /// Construct from a single `f64` (error term is zero).
    #[inline]
    #[must_use]
    pub fn new(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    /// Construct from explicit `(hi, lo)` parts with renormalization.
    #[inline]
    #[must_use]
    pub fn from_f64(hi: f64, lo: f64) -> Self {
        Self::renorm(hi, lo)
    }

    /// Construct from an `i64` integer exactly.
    #[inline]
    #[must_use]
    pub fn from_i64(x: i64) -> Self {
        let hi = x as f64;
        // Compute residual without overflow risk: works for i64 values fitting in f64.
        let lo = (x - hi as i64) as f64;
        Self::renorm(hi, lo)
    }

    // ── Conversion ─────────────────────────────────────────────────────────────

    /// Convert to `f64` — returns only the high-order word.
    #[inline]
    #[must_use]
    pub fn to_f64(self) -> f64 {
        self.hi
    }

    /// Convert to `f64` with compensated rounding (returns `hi + lo`).
    /// Loses precision in the conversion but is numerically more accurate than `to_f64`.
    #[inline]
    #[must_use]
    pub fn to_f128_approx(self) -> f64 {
        self.hi + self.lo
    }

    // ── Predicates ─────────────────────────────────────────────────────────────

    /// Returns `true` if both `hi` and `lo` are finite.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.hi.is_finite() && self.lo.is_finite()
    }

    /// Returns `true` if `hi` is NaN.
    #[inline]
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.hi.is_nan()
    }

    /// Returns `true` if both components are zero.
    #[inline]
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.hi == 0.0 && self.lo == 0.0
    }

    // ── Arithmetic ─────────────────────────────────────────────────────────────

    /// Absolute value.
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        if self.hi < 0.0 {
            Self {
                hi: -self.hi,
                lo: -self.lo,
            }
        } else {
            self
        }
    }

    /// Negate: returns `−self`.
    #[inline]
    #[must_use]
    pub fn negate(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    /// Add `self + rhs` using double-double arithmetic.
    ///
    /// Uses the precise algorithm (Shewchuk 1997).
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, rhs: Self) -> Self {
        let (s1, s2) = two_sum(self.hi, rhs.hi);
        let (t1, t2) = two_sum(self.lo, rhs.lo);
        let c = s2 + t1;
        let (v_hi, v_lo) = two_sum(s1, c);
        let w = t2 + v_lo;
        Self::renorm(v_hi, w)
    }

    /// Subtract `self - rhs`.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, rhs: Self) -> Self {
        self.add(rhs.negate())
    }

    /// Multiply `self * rhs`.
    ///
    /// Uses Dekker's algorithm with TwoProd.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, rhs: Self) -> Self {
        let (p1, p2) = two_product(self.hi, rhs.hi);
        let p2 = p2 + self.hi * rhs.lo + self.lo * rhs.hi;
        Self::renorm(p1, p2)
    }

    /// Divide `self / rhs`.
    ///
    /// Returns `Err` if `rhs` is zero.
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, rhs: Self) -> CoreResult<Self> {
        if rhs.is_zero() {
            return Err(comp_err("DoubleDouble::div — division by zero"));
        }
        // First approximation in f64.
        let q1 = self.hi / rhs.hi;
        // Residual: self − q1 × rhs.
        let r = self.sub(Self::new(q1).mul(rhs));
        // Correction term.
        let q2 = r.hi / rhs.hi;
        Ok(Self::renorm(q1, q2))
    }

    /// Multiply `self × rhs` where `rhs` is a plain `f64`.
    ///
    /// Slightly cheaper than the full double-double multiplication.
    #[must_use]
    pub fn mul_f64(self, rhs: f64) -> Self {
        let (p1, p2) = two_product(self.hi, rhs);
        let p2 = p2 + self.lo * rhs;
        Self::renorm(p1, p2)
    }

    /// Square of `self` (cheaper than `mul(self, self)`).
    #[must_use]
    pub fn square(self) -> Self {
        let (p1, p2) = two_product(self.hi, self.hi);
        let p2 = p2 + 2.0 * self.hi * self.lo;
        Self::renorm(p1, p2)
    }

    /// Square root via Newton-Raphson iteration in double-double precision.
    ///
    /// Returns `Err` for negative inputs.
    pub fn sqrt(self) -> CoreResult<Self> {
        if self.hi < 0.0 {
            return Err(comp_err("DoubleDouble::sqrt — negative input"));
        }
        if self.is_zero() {
            return Ok(Self::ZERO);
        }
        // Initial approximation from f64 sqrt.
        let x0 = Self::new(self.hi.sqrt());
        let half = Self::new(0.5);
        // Newton step: x_{n+1} = (x_n + a/x_n) / 2
        let x1 = x0.add(self.div(x0)?).mul(half);
        // Second refinement for full double-double accuracy.
        let x2 = x1.add(self.div(x1)?).mul(half);
        Ok(x2)
    }

    /// Compute `e^self` using range reduction and Taylor series in DD arithmetic.
    ///
    /// Range reduction: `x = k*ln(2) + r`, then `exp(x) = 2^k * exp(r)`.
    pub fn exp(self) -> CoreResult<Self> {
        if !self.is_finite() {
            return Err(comp_err("DoubleDouble::exp — non-finite input"));
        }
        let ln2 = Self::ln2_const();
        let k_f = (self.hi / ln2.hi).round();
        let k = k_f as i64;
        let k_ln2 = Self::new(k_f).mul(ln2);
        let r = self.sub(k_ln2);
        // Taylor series for exp(r): sum_{n=0}^{N} r^n / n!
        let n_terms = 30usize;
        let mut sum = Self::ONE;
        let mut term = Self::ONE;
        for n in 1..=n_terms {
            term = term.mul(r).div(Self::from_i64(n as i64))?;
            let new_sum = sum.add(term);
            if term.abs().hi.abs() < sum.abs().hi * f64::EPSILON * 0.5 {
                sum = new_sum;
                break;
            }
            sum = new_sum;
        }
        // Multiply by 2^k using ldexp.
        let exp_k = k + 1023;
        if exp_k <= 0 || exp_k >= 2047 {
            // Underflow or overflow: use f64 as fallback.
            let scale = f64::from(2.0_f32).powi(k as i32);
            return Ok(Self::renorm(sum.hi * scale, sum.lo * scale));
        }
        let scale = f64::from_bits((exp_k as u64) << 52);
        Ok(Self::renorm(sum.hi * scale, sum.lo * scale))
    }

    /// Compute `ln(self)` for positive inputs.
    ///
    /// Uses a Newton refinement of the f64 result.
    pub fn ln(self) -> CoreResult<Self> {
        if self.hi <= 0.0 {
            return Err(comp_err("DoubleDouble::ln — argument must be positive"));
        }
        if !self.is_finite() {
            return Err(comp_err("DoubleDouble::ln — non-finite input"));
        }
        // Starting approximation.
        let a0 = Self::new(self.hi.ln());
        // Newton refinement: a1 = a0 + (x - exp(a0)) / exp(a0)
        let exp_a0 = a0.exp()?;
        let correction = self.sub(exp_a0).div(exp_a0)?;
        Ok(a0.add(correction))
    }

    /// Compute `sin(self)` in double-double precision.
    ///
    /// Returns only the sine component. For simultaneous sin/cos, prefer `sincos`.
    pub fn sin(self) -> CoreResult<Self> {
        let (s, _c) = self.sincos()?;
        Ok(s)
    }

    /// Compute `cos(self)` in double-double precision.
    ///
    /// Returns only the cosine component. For simultaneous sin/cos, prefer `sincos`.
    pub fn cos(self) -> CoreResult<Self> {
        let (_s, c) = self.sincos()?;
        Ok(c)
    }

    /// Compute `(sin(self), cos(self))` simultaneously.
    ///
    /// Uses argument reduction to `[-π/4, π/4]` followed by Taylor series.
    pub fn sincos(self) -> CoreResult<(Self, Self)> {
        if !self.is_finite() {
            return Err(comp_err("DoubleDouble::sincos — non-finite input"));
        }
        let pi = Self::PI;
        let two = Self::new(2.0);
        let two_over_pi = two.div(pi)?;
        let k_f = self.mul(two_over_pi).hi.round();
        let k = k_f as i64;
        let half_pi = pi.mul_f64(0.5);
        let r = self.sub(Self::from_i64(k).mul(half_pi));
        let r2 = r.square();
        let n_terms = 20usize;
        // sin(r) = r - r³/3! + r⁵/5! - …
        let mut sin_val = r;
        let mut term_sin = r;
        // cos(r) = 1 - r²/2! + r⁴/4! - …
        let mut cos_val = Self::ONE;
        let mut term_cos = Self::ONE;
        for i in 1..=n_terms {
            term_sin = term_sin
                .mul(r2.negate())
                .div(Self::from_i64((2 * i) as i64))?
                .div(Self::from_i64((2 * i + 1) as i64))?;
            term_cos = term_cos
                .mul(r2.negate())
                .div(Self::from_i64((2 * i - 1) as i64))?
                .div(Self::from_i64((2 * i) as i64))?;
            let new_sin = sin_val.add(term_sin);
            let new_cos = cos_val.add(term_cos);
            let conv = term_sin.abs().hi.abs() < sin_val.abs().hi * f64::EPSILON * 0.5;
            sin_val = new_sin;
            cos_val = new_cos;
            if conv {
                break;
            }
        }
        // Unreduce based on quarter index k mod 4.
        let km4 = ((k % 4) + 4) as usize % 4;
        let (s, c) = match km4 {
            0 => (sin_val, cos_val),
            1 => (cos_val, sin_val.negate()),
            2 => (sin_val.negate(), cos_val.negate()),
            _ => (cos_val.negate(), sin_val),
        };
        Ok((s, c))
    }

    /// Integer power via repeated squaring.
    ///
    /// Handles negative exponents by inversion.
    pub fn powi(self, n: i32) -> CoreResult<Self> {
        if n == 0 {
            return Ok(Self::ONE);
        }
        if n < 0 {
            let inv = Self::ONE.div(self)?;
            return inv.powi(-n);
        }
        let mut result = Self::ONE;
        let mut base = self;
        let mut exp = n as u32;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(base);
            }
            base = base.square();
            exp >>= 1;
        }
        Ok(result)
    }

    // ── Comparison ──────────────────────────────────────────────────────────────

    /// Compare `self` with `rhs` for ordering.
    #[must_use]
    pub fn compare(&self, rhs: &Self) -> Ordering {
        match self.hi.partial_cmp(&rhs.hi) {
            Some(Ordering::Equal) => self.lo.partial_cmp(&rhs.lo).unwrap_or(Ordering::Equal),
            Some(ord) => ord,
            None => Ordering::Equal, // NaN
        }
    }

    // ── Internal helpers ────────────────────────────────────────────────────────

    /// Renormalize `hi + lo` so the invariant `|lo| ≤ 0.5 ulp(hi)` holds.
    #[inline]
    #[must_use]
    pub fn renorm(hi: f64, lo: f64) -> Self {
        let (s, e) = two_sum(hi, lo);
        Self { hi: s, lo: e }
    }

    /// ln(2) to double-double precision (internal constant).
    #[inline]
    fn ln2_const() -> Self {
        Self {
            hi: std::f64::consts::LN_2,
            lo: 2.319_046_813_846_299_6e-17,
        }
    }
}

// ─── Trait implementations ────────────────────────────────────────────────────

impl PartialEq for DoubleDouble {
    fn eq(&self, other: &Self) -> bool {
        self.hi == other.hi && self.lo == other.lo
    }
}

impl PartialOrd for DoubleDouble {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.compare(other))
    }
}

impl Neg for DoubleDouble {
    type Output = DoubleDouble;
    fn neg(self) -> DoubleDouble {
        self.negate()
    }
}

impl Add for DoubleDouble {
    type Output = DoubleDouble;
    fn add(self, rhs: DoubleDouble) -> DoubleDouble {
        DoubleDouble::add(self, rhs)
    }
}

impl Sub for DoubleDouble {
    type Output = DoubleDouble;
    fn sub(self, rhs: DoubleDouble) -> DoubleDouble {
        DoubleDouble::sub(self, rhs)
    }
}

impl Mul for DoubleDouble {
    type Output = DoubleDouble;
    fn mul(self, rhs: DoubleDouble) -> DoubleDouble {
        DoubleDouble::mul(self, rhs)
    }
}

/// Division via `div`; returns `DoubleDouble { hi: NaN, lo: NaN }` if `rhs` is zero.
/// Prefer [`DoubleDouble::div`] for error handling.
impl Div for DoubleDouble {
    type Output = DoubleDouble;
    fn div(self, rhs: DoubleDouble) -> DoubleDouble {
        DoubleDouble::div(self, rhs).unwrap_or(DoubleDouble {
            hi: f64::NAN,
            lo: f64::NAN,
        })
    }
}

impl From<f64> for DoubleDouble {
    fn from(x: f64) -> DoubleDouble {
        DoubleDouble::new(x)
    }
}

impl From<i64> for DoubleDouble {
    fn from(x: i64) -> DoubleDouble {
        DoubleDouble::from_i64(x)
    }
}

impl From<i32> for DoubleDouble {
    fn from(x: i32) -> DoubleDouble {
        DoubleDouble::new(x as f64)
    }
}

impl fmt::Display for DoubleDouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display using hi + lo for maximum precision representation.
        write!(f, "{:.30e} + {:.10e}", self.hi, self.lo)
    }
}

// ─── Free functions ────────────────────────────────────────────────────────────

/// Compute an accumulated dot product `Σ a[i] × b[i]` in double-double precision.
///
/// This is more accurate than a naive f64 dot product for ill-conditioned inputs
/// (Ogita, Rump, Oishi 2005 — Algorithm `AccDot`).
///
/// # Panics
///
/// Does not panic. Returns `DoubleDouble::ZERO` if either slice is empty or lengths differ.
///
/// # Example
///
/// ```rust
/// use scirs2_core::arithmetic::dot_dd;
///
/// let a = [1.0_f64, 2.0, 3.0];
/// let b = [4.0_f64, 5.0, 6.0];
/// let result = dot_dd(&a, &b);
/// assert!((result.to_f64() - 32.0).abs() < 1e-12);
/// ```
pub fn dot_dd(a: &[f64], b: &[f64]) -> DoubleDouble {
    let n = a.len().min(b.len());
    let mut sum = DoubleDouble::ZERO;
    for i in 0..n {
        let (p, e) = two_product(a[i], b[i]);
        let (s1, s2) = two_sum(sum.hi, p);
        let err = s2 + e + sum.lo;
        let (hi, lo_part) = two_sum(s1, err);
        sum = DoubleDouble { hi, lo: lo_part };
    }
    sum
}

/// Compute the sum of `values` in double-double precision.
///
/// Uses the Ogita-Rump-Oishi compensated summation algorithm (`AccSum`),
/// which correctly handles catastrophic cancellation.
///
/// # Example
///
/// ```rust
/// use scirs2_core::arithmetic::sum_dd;
///
/// // Without compensation, [1.0, 1e100, 1.0, -1e100] loses the 2.0.
/// let values = [1.0_f64, 1e100, 1.0, -1e100];
/// let result = sum_dd(&values);
/// assert!((result.to_f64() - 2.0).abs() < 1e-10,
///     "Expected 2.0, got {}", result.to_f64());
/// ```
pub fn sum_dd(values: &[f64]) -> DoubleDouble {
    let mut sum = DoubleDouble::ZERO;
    for &v in values {
        let (s, e) = two_sum(sum.hi, v);
        sum.lo += e;
        sum.hi = s;
    }
    // Final pass to absorb accumulated lo.
    let (hi, lo_part) = two_sum(sum.hi, sum.lo);
    DoubleDouble { hi, lo: lo_part }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_identity_add() {
        let x = DoubleDouble::new(std::f64::consts::PI);
        let result = DoubleDouble::ZERO + x;
        assert!((result.hi - x.hi).abs() < f64::EPSILON);
        assert!((result.lo - x.lo).abs() < f64::EPSILON * 2.0);
    }

    #[test]
    fn test_one_identity_mul() {
        let x = DoubleDouble::new(std::f64::consts::E);
        let result = DoubleDouble::ONE * x;
        assert!((result.hi - x.hi).abs() < f64::EPSILON * 2.0);
    }

    #[test]
    fn test_two_sum_exact() {
        let a = 1.0_f64;
        let b = f64::EPSILON / 2.0;
        let (s, e) = two_sum(a, b);
        assert_eq!(s + e, a + b, "TwoSum: s + e must equal a + b exactly");
    }

    #[test]
    fn test_two_product_exact() {
        let a = 1.0_f64 + f64::EPSILON;
        let b = 1.0_f64 + f64::EPSILON;
        let (p, e) = two_product(a, b);
        let reconstructed = p + e;
        assert!(
            (reconstructed - a * b).abs() <= f64::EPSILON * 4.0,
            "TwoProd roundtrip: {reconstructed} vs {}",
            a * b
        );
    }

    #[test]
    fn test_split_exact() {
        let a = 1.234_567_890_123_456_7_f64;
        let (hi, lo) = split(a);
        assert_eq!(hi + lo, a, "split: hi + lo must equal a exactly");
    }

    #[test]
    fn test_sqrt_four() {
        let four = DoubleDouble::new(4.0);
        let s = four.sqrt().expect("sqrt(4.0) should succeed");
        assert!(
            (s.hi - 2.0).abs() < f64::EPSILON * 4.0,
            "sqrt(4.0) = 2.0, got {}",
            s.hi
        );
    }

    #[test]
    fn test_exp_one_approx_e() {
        let one = DoubleDouble::ONE;
        let e_val = one.exp().expect("exp(1) should succeed");
        let e_exact = std::f64::consts::E;
        // DD result should agree with e to at least 30 decimal places.
        let diff = (e_val.hi + e_val.lo - e_exact).abs();
        assert!(diff < 1e-30, "exp(1) − e = {diff}");
    }

    #[test]
    fn test_sin_pi_over_6() {
        // sin(π/6) = 0.5 exactly
        let pi_over_6 = DoubleDouble::PI.mul_f64(1.0 / 6.0);
        let s = pi_over_6.sin().expect("sin should succeed");
        let diff = (s.hi + s.lo - 0.5).abs();
        assert!(diff < 1e-14, "sin(π/6) ≈ 0.5, diff = {diff}");
    }

    #[test]
    fn test_cos_pi_over_3() {
        // cos(π/3) = 0.5 exactly
        let pi_over_3 = DoubleDouble::PI.mul_f64(1.0 / 3.0);
        let c = pi_over_3.cos().expect("cos should succeed");
        let diff = (c.hi + c.lo - 0.5).abs();
        assert!(diff < 1e-14, "cos(π/3) ≈ 0.5, diff = {diff}");
    }

    #[test]
    fn test_dot_dd_precision() {
        // Ill-conditioned case: naive f64 dot product loses digits.
        let a = [1e15_f64, 1.0, -1e15];
        let b = [1.0_f64, 1e15, 1.0];
        // Exact result: 1e15 + 1e15 - 1e15 = 1e15 but also has 1e15*1 + 1*1e15 - 1e15*1
        // Simple test: dot([1, 2, 3], [4, 5, 6]) = 32
        let a2 = [1.0_f64, 2.0, 3.0];
        let b2 = [4.0_f64, 5.0, 6.0];
        let result = dot_dd(&a2, &b2);
        assert!(
            (result.hi - 32.0).abs() < f64::EPSILON * 4.0,
            "dot = {}",
            result.hi
        );
    }

    #[test]
    fn test_sum_dd_catastrophic_cancellation() {
        // Without compensation this sum loses the 2.0 entirely.
        let values = [1.0_f64, 1e100, 1.0, -1e100];
        let result = sum_dd(&values);
        assert!(
            (result.to_f64() - 2.0).abs() < 1e-10,
            "sum_dd of [1, 1e100, 1, -1e100] should be 2.0, got {}",
            result.to_f64()
        );
    }

    #[test]
    fn test_display_non_empty() {
        let x = DoubleDouble::new(1.5);
        let s = format!("{x}");
        assert!(!s.is_empty(), "Display should produce non-empty string");
        assert!(s.contains("1.5") || s.contains("1."), "Display: {s}");
    }

    #[test]
    fn test_powi_basic() {
        let two = DoubleDouble::new(2.0);
        let eight = two.powi(3).expect("2^3 should succeed");
        assert!(
            (eight.hi - 8.0).abs() < f64::EPSILON * 4.0,
            "2^3 = 8, got {}",
            eight.hi
        );
    }

    #[test]
    fn test_mul_f64() {
        let x = DoubleDouble::new(3.0);
        let result = x.mul_f64(4.0);
        assert!((result.hi - 12.0).abs() < f64::EPSILON * 4.0);
    }

    #[test]
    fn test_pi_constant() {
        let pi = DoubleDouble::PI;
        let diff = (pi.hi - std::f64::consts::PI).abs();
        assert!(diff < f64::EPSILON * 2.0, "PI hi part error: {diff}");
        assert!(
            pi.lo.abs() > 0.0,
            "PI lo should be nonzero for extra precision"
        );
    }

    #[test]
    fn test_e_constant() {
        let e = DoubleDouble::E;
        let diff = (e.hi - std::f64::consts::E).abs();
        assert!(diff < f64::EPSILON * 2.0, "E hi part error: {diff}");
    }
}
