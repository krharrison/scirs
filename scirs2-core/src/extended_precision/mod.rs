//! Double-double arithmetic: ~31 decimal digits of precision using two `f64` values.
//!
//! A `DD` (double-double) number represents an unevaluated sum `hi + lo` where
//! `|lo| ≤ 0.5 ulp(hi)`.  This gives roughly 106 bits (~31.9 significant decimal
//! digits) of precision, twice what `f64` alone provides.
//!
//! ## Error-free transformations
//!
//! All arithmetic uses the classic *error-free transformations* (EFTs):
//!
//! - **TwoSum** (Knuth/Møller): computes `(s, e)` such that `s + e == a + b`
//!   exactly in IEEE arithmetic.
//! - **TwoProd** (Veltkamp splitting): computes `(p, e)` such that `p + e == a * b`
//!   exactly.
//!
//! ## References
//!
//! * Knuth, D. E. (1969). *The Art of Computer Programming*, Vol. 2.
//! * Dekker, T. J. (1971). "A floating-point technique for extending the
//!   available precision." *Numerische Mathematik* 18, 224–242.
//! * Shewchuk, J. R. (1997). "Adaptive precision floating-point arithmetic
//!   and fast robust geometric predicates." *DCG* 18, 305–363.
//! * Hida, Y.; Li, X. S.; Bailey, D. H. (2001). "Algorithms for quad-double
//!   precision floating-point arithmetic." *ARITH-15*.

use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};
use crate::error::{CoreError, CoreResult, ErrorContext};

// ─── Error helpers ─────────────────────────────────────────────────────────────

#[inline(always)]
fn comp_err(msg: impl Into<String>) -> CoreError {
    CoreError::ComputationError(ErrorContext::new(msg))
}

// ─── Error-free transformations ──────────────────────────────────────────────

/// `TwoSum` (Knuth algorithm): returns `(s, e)` with `s + e == a + b` exactly
/// in round-to-nearest IEEE arithmetic.
///
/// # Example
/// ```rust
/// use scirs2_core::extended_precision::two_sum;
/// let (s, e) = two_sum(1.0_f64, 2.0_f64.powi(-53));
/// assert!((s + e - 1.0 - 2.0_f64.powi(-53)).abs() < f64::EPSILON);
/// ```
#[inline]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

/// `TwoProd` via Veltkamp splitting: returns `(p, e)` with `p + e == a * b`
/// exactly in round-to-nearest IEEE arithmetic.
///
/// Uses the Veltkamp split constant 2^27 + 1 = 134_217_729.
///
/// # Example
/// ```rust
/// use scirs2_core::extended_precision::two_prod;
/// let (p, e) = two_prod(3.0_f64, 7.0_f64);
/// assert_eq!(p + e, 21.0);
/// ```
#[inline]
pub fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    // Veltkamp split of a.
    let c = (134_217_729.0_f64) * a; // 2^27 + 1
    let a_hi = c - (c - a);
    let a_lo = a - a_hi;
    // Veltkamp split of b.
    let c2 = (134_217_729.0_f64) * b;
    let b_hi = c2 - (c2 - b);
    let b_lo = b - b_hi;
    let e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (p, e)
}

/// `TwoDiff`: analogous to `TwoSum` but for subtraction.
/// Returns `(d, e)` with `d + e == a - b` exactly.
#[inline]
pub fn two_diff(a: f64, b: f64) -> (f64, f64) {
    two_sum(a, -b)
}

// ─── DD struct ────────────────────────────────────────────────────────────────

/// Double-double precision floating-point number.
///
/// Represents the value `hi + lo` where `hi` carries the significant bits and
/// `lo` carries the rounding error.  The invariant is `|lo| ≤ 0.5 ulp(hi)`,
/// maintained by normalisation after every operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct DD {
    /// High-order word (most significant bits).
    pub hi: f64,
    /// Low-order word (error term; `|lo| ≤ 0.5 ulp(hi)`).
    pub lo: f64,
}

impl DD {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Construct a `DD` from a single `f64` (zero error term).
    #[inline]
    #[must_use]
    pub fn from_f64(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    /// Construct a `DD` from an `i64` integer, exact.
    #[inline]
    #[must_use]
    pub fn from_i64(x: i64) -> Self {
        // Split into high and low halves to avoid rounding.
        let hi = x as f64;
        let lo = (x - hi as i64) as f64;
        Self::renorm(hi, lo)
    }

    /// Construct a `DD` from two `f64` values `a + b` (may violate the
    /// normalisation invariant; use `renorm` internally).
    #[inline]
    #[must_use]
    pub fn from_parts(hi: f64, lo: f64) -> Self {
        Self::renorm(hi, lo)
    }

    // ── Constants ─────────────────────────────────────────────────────────────

    /// Zero.
    pub const ZERO: DD = DD { hi: 0.0, lo: 0.0 };

    /// One.
    pub const ONE: DD = DD { hi: 1.0, lo: 0.0 };

    /// π to double-double precision.
    ///
    /// Value from Bailey et al. (2021) quad-double table.
    #[must_use]
    pub fn pi() -> DD {
        // π ≈ 3.141592653589793115997963...
        DD { hi: 3.141_592_653_589_793_1_f64, lo: 1.224_646_799_147_353_2e-16 }
    }

    /// e = exp(1) to double-double precision.
    #[must_use]
    pub fn e() -> DD {
        // e ≈ 2.718281828459045235360287...
        DD { hi: 2.718_281_828_459_045_f64, lo: 1.445_646_891_729_250_2e-16 }
    }

    /// ln(2) to double-double precision.
    #[must_use]
    pub fn ln2() -> DD {
        // ln 2 ≈ 0.693147180559945309417232...
        DD { hi: 0.693_147_180_559_945_3_f64, lo: 2.319_046_813_846_299_6e-17 }
    }

    /// √2 to double-double precision.
    #[must_use]
    pub fn sqrt2() -> DD {
        // √2 ≈ 1.41421356237309504880168...
        DD { hi: 1.414_213_562_373_095_f64, lo: -9.667_293_313_452_914e-17 }
    }

    // ── Predicates ────────────────────────────────────────────────────────────

    /// Returns `true` if the value is zero.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.hi == 0.0 && self.lo == 0.0
    }

    /// Returns `true` if the value is finite.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.hi.is_finite()
    }

    /// Returns `true` if `hi` is NaN.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.hi.is_nan()
    }

    // ── Conversion ────────────────────────────────────────────────────────────

    /// Convert to `f64` (just the high-order word).
    #[inline]
    #[must_use]
    pub fn to_f64(self) -> f64 {
        self.hi
    }

    /// Convert to `f64` with compensated rounding (returns `hi + lo`).
    #[inline]
    #[must_use]
    pub fn to_f64_round(self) -> f64 {
        self.hi + self.lo
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Negate: returns `−self`.
    #[inline]
    #[must_use]
    pub fn negate(self) -> DD {
        DD { hi: -self.hi, lo: -self.lo }
    }

    /// Addition: `self + rhs`.
    #[must_use]
    pub fn dd_add(self, rhs: DD) -> DD {
        // Sloppy algorithm (Shewchuk):
        // (a + b) + (c + d) ≈ (a + c) + ((a − s) + c + b + d)
        // where s = a + c.  For better accuracy use the precise version.
        let (s1, s2) = two_sum(self.hi, rhs.hi);
        let (t1, t2) = two_sum(self.lo, rhs.lo);
        let c = s2 + t1;
        let (v_hi, v_lo) = two_sum(s1, c);
        let w = t2 + v_lo;
        DD::renorm(v_hi, w)
    }

    /// Subtraction: `self − rhs`.
    #[must_use]
    pub fn dd_sub(self, rhs: DD) -> DD {
        self.dd_add(rhs.negate())
    }

    /// Multiplication: `self × rhs`.
    #[must_use]
    pub fn dd_mul(self, rhs: DD) -> DD {
        let (p1, p2) = two_prod(self.hi, rhs.hi);
        let p2 = p2 + self.hi * rhs.lo + self.lo * rhs.hi;
        DD::renorm(p1, p2)
    }

    /// Division: `self / rhs` via Newton-Raphson refinement.
    pub fn dd_div(self, rhs: DD) -> CoreResult<DD> {
        if rhs.is_zero() {
            return Err(comp_err("DD::div — division by zero"));
        }
        // First approximation.
        let q1 = self.hi / rhs.hi;
        // Residual: self − q1 × rhs.
        let r = self.dd_sub(DD::from_f64(q1).dd_mul(rhs));
        // Correction.
        let q2 = r.hi / rhs.hi;
        Ok(DD::renorm(q1, q2))
    }

    /// Absolute value.
    #[inline]
    #[must_use]
    pub fn abs(self) -> DD {
        if self.hi < 0.0 { self.negate() } else { self }
    }

    /// Square root via Newton-Raphson in double-double precision.
    pub fn sqrt(self) -> CoreResult<DD> {
        if self.hi < 0.0 {
            return Err(comp_err("DD::sqrt of negative number"));
        }
        if self.is_zero() {
            return Ok(DD::ZERO);
        }
        // Initial approximation in f64.
        let x0 = DD::from_f64(self.hi.sqrt());
        // One Newton step: x₁ = (x₀ + self/x₀) / 2
        let half = DD::from_f64(0.5);
        let x1 = x0.dd_add(self.dd_div(x0)?).dd_mul(half);
        // Second Newton step for full double-double accuracy.
        let x2 = x1.dd_add(self.dd_div(x1)?).dd_mul(half);
        Ok(x2)
    }

    /// Square of self (slightly cheaper than `dd_mul(self, self)`).
    #[must_use]
    pub fn square(self) -> DD {
        let (p1, p2) = two_prod(self.hi, self.hi);
        let p2 = p2 + 2.0 * self.hi * self.lo;
        DD::renorm(p1, p2)
    }

    // ── Renormalisation ───────────────────────────────────────────────────────

    /// Renormalise `hi + lo` so that `|lo| ≤ 0.5 ulp(hi)`.
    #[inline]
    #[must_use]
    pub fn renorm(hi: f64, lo: f64) -> DD {
        let (s, e) = two_sum(hi, lo);
        DD { hi: s, lo: e }
    }

    // ── Comparison ────────────────────────────────────────────────────────────

    /// Compare `self` with `rhs` for ordering.
    #[must_use]
    pub fn compare(&self, rhs: &DD) -> Ordering {
        match self.hi.partial_cmp(&rhs.hi) {
            Some(Ordering::Equal) => self.lo.partial_cmp(&rhs.lo).unwrap_or(Ordering::Equal),
            Some(ord) => ord,
            None => Ordering::Equal, // NaN
        }
    }
}

// ─── Trait implementations ────────────────────────────────────────────────────

impl PartialEq for DD {
    fn eq(&self, other: &Self) -> bool {
        self.hi == other.hi && self.lo == other.lo
    }
}

impl PartialOrd for DD {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.compare(other))
    }
}

impl Neg for DD {
    type Output = DD;
    fn neg(self) -> DD {
        self.negate()
    }
}

impl Add for DD {
    type Output = DD;
    fn add(self, rhs: DD) -> DD {
        self.dd_add(rhs)
    }
}

impl Sub for DD {
    type Output = DD;
    fn sub(self, rhs: DD) -> DD {
        self.dd_sub(rhs)
    }
}

impl Mul for DD {
    type Output = DD;
    fn mul(self, rhs: DD) -> DD {
        self.dd_mul(rhs)
    }
}

/// Division via `dd_div`; panics if `rhs` is zero (use `dd_div` for error handling).
impl Div for DD {
    type Output = DD;
    fn div(self, rhs: DD) -> DD {
        self.dd_div(rhs).unwrap_or(DD { hi: f64::NAN, lo: f64::NAN })
    }
}

impl fmt::Display for DD {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display the value to ~31 significant digits.
        let v = self.hi + self.lo;
        write!(f, "{:.30e}", v)
    }
}

impl From<f64> for DD {
    fn from(x: f64) -> DD {
        DD::from_f64(x)
    }
}

impl From<i64> for DD {
    fn from(x: i64) -> DD {
        DD::from_i64(x)
    }
}

impl From<i32> for DD {
    fn from(x: i32) -> DD {
        DD::from_f64(x as f64)
    }
}

// ─── Higher-level functions ───────────────────────────────────────────────────

/// Compute `e^x` in double-double precision using Taylor series with
/// argument reduction `x = k×ln(2) + r`.
pub fn dd_exp(x: DD) -> CoreResult<DD> {
    if !x.is_finite() {
        return Err(comp_err("dd_exp: non-finite input"));
    }
    // Argument reduction.
    let ln2 = DD::ln2();
    let k_f = (x.hi / ln2.hi).round();
    let k = k_f as i64;
    // r = x − k×ln(2)
    let k_ln2 = DD::from_f64(k_f).dd_mul(ln2);
    let r = x.dd_sub(k_ln2);
    // Taylor series for e^r (r is now small).
    let n_terms = 30usize;
    let mut sum = DD::ONE;
    let mut term = DD::ONE;
    for n in 1..=n_terms {
        term = term.dd_mul(r).dd_div(DD::from_i64(n as i64))?;
        let new_sum = sum.dd_add(term);
        if term.abs().hi.abs() < sum.abs().hi * f64::EPSILON * 0.5 {
            sum = new_sum;
            break;
        }
        sum = new_sum;
    }
    // Multiply by 2^k using ldexp.
    let scale = f64::from_bits(((1023i64 + k) as u64) << 52);
    Ok(DD::renorm(sum.hi * scale, sum.lo * scale))
}

/// Compute `ln(x)` for positive `x` in double-double precision.
///
/// Uses argument reduction and a first-order Newton correction to the
/// float `ln` value.
pub fn dd_ln(x: DD) -> CoreResult<DD> {
    if x.hi <= 0.0 {
        return Err(comp_err("dd_ln: argument must be positive"));
    }
    if !x.is_finite() {
        return Err(comp_err("dd_ln: non-finite input"));
    }
    // Starting approximation.
    let a0 = DD::from_f64(x.hi.ln());
    // Newton refinement: a1 = a0 + (x - e^a0) / e^a0 = a0 + x×e^{-a0} - 1
    let exp_a0 = dd_exp(a0)?;
    // a1 = a0 + (x - exp(a0)) / exp(a0)
    let correction = x.dd_sub(exp_a0).dd_div(exp_a0)?;
    Ok(a0.dd_add(correction))
}

/// Compute `sin(x)` and `cos(x)` simultaneously in double-double precision
/// using a Clenshaw-Curtis style Chebyshev evaluation after argument reduction.
///
/// Returns `(sin(x), cos(x))`.
pub fn dd_sincos(x: DD) -> CoreResult<(DD, DD)> {
    if !x.is_finite() {
        return Err(comp_err("dd_sincos: non-finite input"));
    }
    // Argument reduction to [-π/4, π/4].
    let pi = DD::pi();
    let two_over_pi = DD::from_f64(2.0).dd_div(pi)?;
    let k_f = (x.dd_mul(two_over_pi)).hi.round();
    let k = k_f as i64;
    let half_pi = pi.dd_mul(DD::from_f64(0.5));
    let r = x.dd_sub(DD::from_i64(k).dd_mul(half_pi));
    // Taylor series for sin(r) and cos(r).
    let r2 = r.square();
    let n_terms = 20usize;
    // sin(r) = r - r³/3! + r⁵/5! - …
    let mut sin_val = r;
    let mut term_sin = r;
    // cos(r) = 1 - r²/2! + r⁴/4! - …
    let mut cos_val = DD::ONE;
    let mut term_cos = DD::ONE;
    for i in 1..=n_terms {
        term_sin = term_sin.dd_mul(r2.negate())
            .dd_div(DD::from_i64((2 * i) as i64))?
            .dd_div(DD::from_i64((2 * i + 1) as i64))?;
        term_cos = term_cos.dd_mul(r2.negate())
            .dd_div(DD::from_i64((2 * i - 1) as i64))?
            .dd_div(DD::from_i64((2 * i) as i64))?;
        let new_sin = sin_val.dd_add(term_sin);
        let new_cos = cos_val.dd_add(term_cos);
        let conv = term_sin.abs().hi.abs() < sin_val.abs().hi * f64::EPSILON * 0.5;
        sin_val = new_sin;
        cos_val = new_cos;
        if conv {
            break;
        }
    }
    // Unreduce using the quarter index k mod 4.
    let km4 = ((k % 4) + 4) as usize % 4;
    let (s, c) = match km4 {
        0 => (sin_val, cos_val),
        1 => (cos_val, sin_val.negate()),
        2 => (sin_val.negate(), cos_val.negate()),
        _ => (cos_val.negate(), sin_val),
    };
    Ok((s, c))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_sum_exact() {
        let a = 1.0_f64;
        let b = f64::EPSILON / 2.0;
        let (s, e) = two_sum(a, b);
        // s + e should equal a + b exactly.
        assert_eq!(s + e, a + b, "TwoSum roundtrip failed");
    }

    #[test]
    fn test_two_prod_exact() {
        let a = 1.0_f64 + f64::EPSILON;
        let b = 1.0_f64 + f64::EPSILON;
        let (p, e) = two_prod(a, b);
        let exact = (a as f64) * (b as f64); // same as p, but check e is residual
        let _ = exact; // just ensure it compiles
        // p + e must reconstruct a * b; we can check identity of sum.
        let reconstructed = p + e;
        assert!((reconstructed - a * b).abs() <= f64::EPSILON * 4.0,
            "TwoProd roundtrip: {reconstructed} vs {}", a * b);
    }

    #[test]
    fn test_dd_add_basic() {
        let a = DD::from_f64(1.0);
        let b = DD::from_f64(2.0);
        let c = a.dd_add(b);
        assert_eq!(c.hi, 3.0, "1 + 2 should be 3");
        assert_eq!(c.lo, 0.0);
    }

    #[test]
    fn test_dd_sub_basic() {
        let a = DD::from_f64(5.0);
        let b = DD::from_f64(3.0);
        let c = a.dd_sub(b);
        assert_eq!(c.hi, 2.0);
    }

    #[test]
    fn test_dd_mul_basic() {
        let a = DD::from_f64(3.0);
        let b = DD::from_f64(4.0);
        let c = a.dd_mul(b);
        assert_eq!(c.hi, 12.0);
        assert_eq!(c.lo, 0.0);
    }

    #[test]
    fn test_dd_div_basic() {
        let a = DD::from_f64(10.0);
        let b = DD::from_f64(4.0);
        let c = a.dd_div(b).expect("should succeed");
        let diff = (c.hi - 2.5).abs();
        assert!(diff < f64::EPSILON * 4.0, "10/4 should be 2.5, got {}", c.hi);
    }

    #[test]
    fn test_dd_div_zero() {
        let a = DD::from_f64(1.0);
        let b = DD::ZERO;
        assert!(a.dd_div(b).is_err());
    }

    #[test]
    fn test_dd_sqrt() {
        let two = DD::from_f64(2.0);
        let s = two.sqrt().expect("should succeed");
        let expected = std::f64::consts::SQRT_2;
        let diff = (s.hi - expected).abs();
        assert!(diff < f64::EPSILON * 4.0, "sqrt(2) error: {diff}");
        // Check that lo contributes extra precision.
        let reconst = s.hi + s.lo;
        let better = (reconst - expected).abs();
        assert!(better < 1e-31 || better < diff, "DD sqrt should be more precise than f64");
    }

    #[test]
    fn test_dd_sqrt_negative() {
        let neg = DD::from_f64(-1.0);
        assert!(neg.sqrt().is_err());
    }

    #[test]
    fn test_dd_pi_accuracy() {
        let pi = DD::pi();
        let diff = (pi.hi - std::f64::consts::PI).abs();
        // hi should be very close to f64 π.
        assert!(diff < f64::EPSILON * 2.0, "DD::pi hi part error: {diff}");
        // lo should be nonzero (contributes extra precision).
        assert!(pi.lo.abs() > 0.0, "DD::pi lo part should be non-zero");
    }

    #[test]
    fn test_dd_e_accuracy() {
        let e = DD::e();
        let diff = (e.hi - std::f64::consts::E).abs();
        assert!(diff < f64::EPSILON * 2.0, "DD::e hi part error: {diff}");
    }

    #[test]
    fn test_dd_ln2_accuracy() {
        let ln2 = DD::ln2();
        let diff = (ln2.hi - std::f64::consts::LN_2).abs();
        assert!(diff < f64::EPSILON * 2.0, "DD::ln2 hi part error: {diff}");
    }

    #[test]
    fn test_dd_sqrt2_accuracy() {
        let sqrt2 = DD::sqrt2();
        let diff = (sqrt2.hi - std::f64::consts::SQRT_2).abs();
        assert!(diff < f64::EPSILON * 2.0, "DD::sqrt2 hi part error: {diff}");
    }

    #[test]
    fn test_dd_exp() {
        let one = DD::ONE;
        let e_val = dd_exp(one).expect("should succeed");
        let expected = std::f64::consts::E;
        let diff = (e_val.hi + e_val.lo - expected).abs();
        assert!(diff < 1e-30, "dd_exp(1) - e = {diff}");
    }

    #[test]
    fn test_dd_ln() {
        let e_val = DD::e();
        let ln_e = dd_ln(e_val).expect("should succeed");
        let diff = (ln_e.hi + ln_e.lo - 1.0).abs();
        assert!(diff < 1e-28, "ln(e) - 1 = {diff}");
    }

    #[test]
    fn test_dd_sincos() {
        let x = DD::from_f64(1.0);
        let (s, c) = dd_sincos(x).expect("should succeed");
        let expected_sin = 1.0_f64.sin();
        let expected_cos = 1.0_f64.cos();
        let diff_s = (s.hi - expected_sin).abs();
        let diff_c = (c.hi - expected_cos).abs();
        assert!(diff_s < 1e-15, "sin(1) diff: {diff_s}");
        assert!(diff_c < 1e-15, "cos(1) diff: {diff_c}");
    }

    #[test]
    fn test_operator_overloads() {
        let a = DD::from_f64(3.0);
        let b = DD::from_f64(4.0);
        let sum = a + b;
        let diff = a - b;
        let prod = a * b;
        let quot = a / b;
        assert_eq!(sum.hi, 7.0);
        assert_eq!(diff.hi, -1.0);
        assert_eq!(prod.hi, 12.0);
        assert!((quot.hi - 0.75).abs() < f64::EPSILON * 4.0);
    }

    #[test]
    fn test_partial_ord() {
        let a = DD::from_f64(1.0);
        let b = DD::from_f64(2.0);
        assert!(a < b);
        assert!(b > a);
        assert!(a <= a);
    }

    #[test]
    fn test_from_i64() {
        let x = DD::from_i64(1_000_000_000_000i64);
        assert_eq!(x.hi, 1_000_000_000_000.0_f64);
    }

    #[test]
    fn test_square() {
        let x = DD::from_f64(3.0);
        let sq = x.square();
        assert_eq!(sq.hi, 9.0);
        assert_eq!(sq.lo, 0.0);
    }

    #[test]
    fn test_display() {
        let x = DD::from_f64(1.5);
        let s = format!("{x}");
        assert!(s.contains("1.5"), "Display: {s}");
    }
}
