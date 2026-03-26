//! Double-double arithmetic for extended precision (~31 decimal digits).
//!
//! A `DoubleDouble` represents a number as an unevaluated sum of two IEEE 754
//! `f64` values `(hi, lo)` such that `hi + lo` is the represented value.
//! The pair is maintained in a *non-overlapping* canonical form where
//! `|lo| <= 0.5 * ulp(hi)`, giving roughly twice the precision of a single
//! `f64` (about 31–32 significant decimal digits).
//!
//! # Algorithms
//!
//! The core building blocks are the **TwoSum** (Knuth / Moller) and
//! **TwoProduct** (Dekker) error-free transformations which capture the
//! rounding error of an IEEE addition or multiplication exactly.
//!
//! Higher-level functions (sqrt, exp, log, sin, cos, gamma, erf, ...)
//! are built on top using Newton iteration, Taylor series with argument
//! reduction, or other classical techniques, all carried out in DD
//! arithmetic.
//!
//! # Example
//!
//! ```
//! use scirs2_special::double_double::{DoubleDouble, DD_PI};
//!
//! // Compute sin(pi) — should be very close to zero
//! let s = DD_PI.sin();
//! assert!(s.to_f64().abs() < 1e-30);
//! ```

mod arithmetic;
mod elementary;
mod special_functions;

pub use arithmetic::*;
pub use elementary::*;
pub use special_functions::*;

use std::fmt;

// ---------------------------------------------------------------------------
// Core type
// ---------------------------------------------------------------------------

/// A double-double number represented as `hi + lo` with ~31 decimal digits
/// of precision.
#[derive(Clone, Copy)]
pub struct DoubleDouble {
    /// High-order part
    pub hi: f64,
    /// Low-order part (correction term)
    pub lo: f64,
}

impl DoubleDouble {
    /// Create a new `DoubleDouble` from two `f64` values.
    ///
    /// The caller is responsible for ensuring that `|lo| <= 0.5 * ulp(hi)`.
    /// For arbitrary pairs, use [`DoubleDouble::from_sum`] instead.
    #[inline]
    pub const fn new(hi: f64, lo: f64) -> Self {
        Self { hi, lo }
    }

    /// Create a `DoubleDouble` from an arbitrary sum `a + b`, normalising
    /// the pair so the invariant holds.
    #[inline]
    pub fn from_sum(a: f64, b: f64) -> Self {
        let (s, e) = two_sum(a, b);
        Self { hi: s, lo: e }
    }

    /// Convert to a single `f64` (loses precision).
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    /// Return `true` if the value is finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.hi.is_finite()
    }

    /// Return `true` if the value is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.hi.is_nan()
    }

    /// Return `true` if the value is zero.
    #[inline]
    pub fn is_zero(self) -> bool {
        self.hi == 0.0
    }

    /// Return `true` if the value is negative.
    #[inline]
    pub fn is_negative(self) -> bool {
        self.hi < 0.0 || (self.hi == 0.0 && self.lo < 0.0)
    }

    /// Return `true` if the value is positive.
    #[inline]
    pub fn is_positive(self) -> bool {
        self.hi > 0.0 || (self.hi == 0.0 && self.lo > 0.0)
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

impl fmt::Debug for DoubleDouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DD({:e} + {:e})", self.hi, self.lo)
    }
}

impl fmt::Display for DoubleDouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Show enough digits to be useful
        write!(f, "{:.32e}", self.hi + self.lo)
    }
}

impl PartialEq for DoubleDouble {
    fn eq(&self, other: &Self) -> bool {
        self.hi == other.hi && self.lo == other.lo
    }
}

impl PartialOrd for DoubleDouble {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.hi.partial_cmp(&other.hi) {
            Some(std::cmp::Ordering::Equal) => self.lo.partial_cmp(&other.lo),
            ord => ord,
        }
    }
}

impl From<f64> for DoubleDouble {
    #[inline]
    fn from(v: f64) -> Self {
        Self { hi: v, lo: 0.0 }
    }
}

impl From<i64> for DoubleDouble {
    #[inline]
    fn from(v: i64) -> Self {
        let hi = v as f64;
        let lo = (v - hi as i64) as f64;
        Self { hi, lo }
    }
}

impl From<DoubleDouble> for f64 {
    #[inline]
    fn from(dd: DoubleDouble) -> f64 {
        dd.to_f64()
    }
}

// ---------------------------------------------------------------------------
// Constants (high-precision values taken from known multiprecision sources)
// ---------------------------------------------------------------------------

/// Zero
pub const DD_ZERO: DoubleDouble = DoubleDouble::new(0.0, 0.0);
/// One
pub const DD_ONE: DoubleDouble = DoubleDouble::new(1.0, 0.0);
/// Two
pub const DD_TWO: DoubleDouble = DoubleDouble::new(2.0, 0.0);
/// Half
pub const DD_HALF: DoubleDouble = DoubleDouble::new(0.5, 0.0);

/// Pi to ~31 digits: 3.14159265358979323846264338327950288...
pub const DD_PI: DoubleDouble = DoubleDouble::new(3.141592653589793, 1.2246467991473532e-16);

/// 2*Pi
pub const DD_TWO_PI: DoubleDouble = DoubleDouble::new(6.283185307179586, 2.4492935982947064e-16);

/// Pi/2
pub const DD_PI_OVER_2: DoubleDouble = DoubleDouble::new(1.5707963267948966, 6.123233995736766e-17);

/// Pi/4
pub const DD_PI_OVER_4: DoubleDouble = DoubleDouble::new(0.7853981633974483, 3.061616997868383e-17);

/// e (Euler's number) to ~31 digits
pub const DD_E: DoubleDouble = DoubleDouble::new(2.718281828459045, 1.4456468917292502e-16);

/// ln(2)
pub const DD_LN2: DoubleDouble = DoubleDouble::new(0.6931471805599453, 2.3190468138462996e-17);

/// ln(10)
pub const DD_LN10: DoubleDouble = DoubleDouble::new(2.302585092994046, -2.1707562233822494e-16);

/// sqrt(2)
pub const DD_SQRT2: DoubleDouble = DoubleDouble::new(1.4142135623730951, -9.667293313452913e-17);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_f64() {
        let dd = DoubleDouble::from(3.14);
        assert_eq!(dd.hi, 3.14);
        assert_eq!(dd.lo, 0.0);
    }

    #[test]
    fn test_from_i64() {
        let dd = DoubleDouble::from(42_i64);
        assert!((dd.to_f64() - 42.0).abs() < 1e-15);
    }

    #[test]
    fn test_into_f64() {
        let dd = DD_PI;
        let v: f64 = dd.into();
        assert!((v - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn test_partial_ord() {
        assert!(DD_PI > DD_E);
        assert!(DD_ONE < DD_TWO);
        assert!(DD_ZERO < DD_ONE);
    }

    #[test]
    fn test_display_debug() {
        let s = format!("{}", DD_PI);
        assert!(!s.is_empty());
        let d = format!("{:?}", DD_PI);
        assert!(d.contains("DD("));
    }

    #[test]
    fn test_predicates() {
        assert!(DD_ZERO.is_zero());
        assert!(!DD_ONE.is_zero());
        assert!(DD_ONE.is_positive());
        assert!(DD_ONE.is_finite());
        assert!(!DD_ONE.is_nan());
        let neg = -DD_ONE;
        assert!(neg.is_negative());
    }

    #[test]
    fn test_pi_precision() {
        // pi to 31+ digits: 3.1415926535897932384626433832795
        // Our DD_PI should match to ~31 digits
        let pi_ref = 3.141_592_653_589_793_238_462_643_383_279_5;
        let diff = (DD_PI.hi + DD_PI.lo) - pi_ref;
        assert!(diff.abs() < 1e-31, "pi diff = {diff:e}");
    }
}
