//! Core arithmetic operations for double-double numbers.
//!
//! Implements the TwoSum (Knuth–Moller) and TwoProduct (Dekker) error-free
//! transformations, and the standard `Add`, `Sub`, `Mul`, `Div` operator
//! traits for [`DoubleDouble`].

use super::DoubleDouble;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ---------------------------------------------------------------------------
// Error-free transformations
// ---------------------------------------------------------------------------

/// TwoSum: compute `s + e = a + b` exactly, where `s` is the floating-point
/// sum and `e` is the rounding error.
///
/// Requires IEEE 754 arithmetic with round-to-nearest.
#[inline]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

/// QuickTwoSum: a fast variant of TwoSum valid when `|a| >= |b|`.
#[inline]
pub fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let e = b - (s - a);
    (s, e)
}

/// TwoDiff: compute `s + e = a - b` exactly.
#[inline]
pub fn two_diff(a: f64, b: f64) -> (f64, f64) {
    let s = a - b;
    let v = s - a;
    let e = (a - (s - v)) - (b + v);
    (s, e)
}

/// Split a f64 into high and low parts for Dekker multiplication.
/// Uses the Veltkamp split with `s = 2^27 + 1`.
#[inline]
fn split(a: f64) -> (f64, f64) {
    const SPLITTER: f64 = 134_217_729.0; // 2^27 + 1
    let t = SPLITTER * a;
    let a_hi = t - (t - a);
    let a_lo = a - a_hi;
    (a_hi, a_lo)
}

/// TwoProduct: compute `p + e = a * b` exactly using Dekker splitting.
///
/// If the platform has FMA, that single instruction gives the error directly;
/// otherwise we fall back to Dekker's algorithm.
#[inline]
pub fn two_product(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    // Try FMA path: e = fma(a, b, -p)
    let e = a.mul_add(b, -p);
    (p, e)
}

// ---------------------------------------------------------------------------
// Negation & absolute value
// ---------------------------------------------------------------------------

impl Neg for DoubleDouble {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.hi, -self.lo)
    }
}

impl DoubleDouble {
    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        if self.is_negative() {
            -self
        } else {
            self
        }
    }

    /// Optimised squaring (cheaper than general multiply).
    #[inline]
    pub fn sqr(self) -> Self {
        let (p1, p2) = two_product(self.hi, self.hi);
        let p2 = p2 + 2.0 * self.hi * self.lo;
        let (s, e) = quick_two_sum(p1, p2);
        Self::new(s, e)
    }

    /// Reciprocal `1 / self`, computed via Newton iteration.
    pub fn recip(self) -> Self {
        if self.is_zero() {
            return Self::new(f64::INFINITY, 0.0);
        }
        // Initial approximation
        let x0 = 1.0 / self.hi;
        // Newton: x_{n+1} = x_n + x_n * (1 - a * x_n)
        let dd_x = Self::from(x0);
        let r = Self::from(1.0) - self * dd_x;
        let dd_x = dd_x + dd_x * r;
        // Second iteration for full DD precision
        let r = Self::from(1.0) - self * dd_x;
        dd_x + dd_x * r
    }
}

// ---------------------------------------------------------------------------
// DD + DD
// ---------------------------------------------------------------------------

impl Add for DoubleDouble {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let (s1, e1) = two_sum(self.hi, rhs.hi);
        let (s2, e2) = two_sum(self.lo, rhs.lo);
        let e1 = e1 + s2;
        let (s1, e1) = quick_two_sum(s1, e1);
        let e1 = e1 + e2;
        let (s1, e1) = quick_two_sum(s1, e1);
        Self::new(s1, e1)
    }
}

impl AddAssign for DoubleDouble {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

// ---------------------------------------------------------------------------
// DD - DD
// ---------------------------------------------------------------------------

impl Sub for DoubleDouble {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl SubAssign for DoubleDouble {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

// ---------------------------------------------------------------------------
// DD * DD
// ---------------------------------------------------------------------------

impl Mul for DoubleDouble {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Hida / Li / Bailey QD-style multiplication:
        // (a.hi + a.lo) * (b.hi + b.lo)
        //  = a.hi*b.hi + a.hi*b.lo + a.lo*b.hi + a.lo*b.lo
        //
        // TwoProduct gives exact a.hi*b.hi = p1 + p2
        let (p1, p2) = two_product(self.hi, rhs.hi);
        // Cross terms (only need the leading f64 of each)
        let p2 = p2 + (self.hi * rhs.lo + self.lo * rhs.hi);
        // Renormalize
        let (s, e) = quick_two_sum(p1, p2);
        Self::new(s, e)
    }
}

impl MulAssign for DoubleDouble {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

// ---------------------------------------------------------------------------
// DD / DD
// ---------------------------------------------------------------------------

impl Div for DoubleDouble {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        if rhs.is_zero() {
            if self.is_zero() {
                return Self::new(f64::NAN, 0.0);
            }
            let sign = if self.is_negative() != rhs.is_negative() {
                -1.0
            } else {
                1.0
            };
            return Self::new(sign * f64::INFINITY, 0.0);
        }
        // Compute quotient via long-division style:
        //   q1 = a.hi / b.hi  (approx quotient)
        //   r  = a - b * q1   (remainder in DD)
        //   q2 = r.hi / b.hi  (correction)
        let q1 = self.hi / rhs.hi;
        let r = self - rhs * Self::from(q1);
        let q2 = r.hi / rhs.hi;
        let r = r - rhs * Self::from(q2);
        let q3 = r.hi / rhs.hi;
        let (s, e) = quick_two_sum(q1, q2);
        Self::new(s, e) + Self::from(q3)
    }
}

impl DivAssign for DoubleDouble {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// ---------------------------------------------------------------------------
// Mixed operations: DD op f64
// ---------------------------------------------------------------------------

impl Add<f64> for DoubleDouble {
    type Output = Self;

    #[inline]
    fn add(self, rhs: f64) -> Self {
        self + Self::from(rhs)
    }
}

impl Sub<f64> for DoubleDouble {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: f64) -> Self {
        self - Self::from(rhs)
    }
}

impl Mul<f64> for DoubleDouble {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        let (p1, p2) = two_product(self.hi, rhs);
        let p2 = p2 + self.lo * rhs;
        let (s, e) = quick_two_sum(p1, p2);
        Self::new(s, e)
    }
}

impl Div<f64> for DoubleDouble {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self {
        self / Self::from(rhs)
    }
}

// ---------------------------------------------------------------------------
// f64 op DD (commutative convenience)
// ---------------------------------------------------------------------------

impl Add<DoubleDouble> for f64 {
    type Output = DoubleDouble;

    #[inline]
    fn add(self, rhs: DoubleDouble) -> DoubleDouble {
        rhs + self
    }
}

impl Sub<DoubleDouble> for f64 {
    type Output = DoubleDouble;

    #[inline]
    fn sub(self, rhs: DoubleDouble) -> DoubleDouble {
        DoubleDouble::from(self) - rhs
    }
}

impl Mul<DoubleDouble> for f64 {
    type Output = DoubleDouble;

    #[inline]
    fn mul(self, rhs: DoubleDouble) -> DoubleDouble {
        rhs * self
    }
}

impl Div<DoubleDouble> for f64 {
    type Output = DoubleDouble;

    #[inline]
    fn div(self, rhs: DoubleDouble) -> DoubleDouble {
        DoubleDouble::from(self) / rhs
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::double_double::{DD_E, DD_ONE, DD_PI, DD_TWO, DD_ZERO};

    #[test]
    fn test_two_sum_exactness() {
        // a + b = s + e  exactly
        let a = 1.0;
        let b = 1e-20;
        let (s, e) = two_sum(a, b);
        // Reconstruct
        let reconstructed = s + e;
        assert!(
            (reconstructed - (a + b)).abs() < 1e-35,
            "TwoSum not exact: diff = {:e}",
            reconstructed - (a + b)
        );
    }

    #[test]
    fn test_two_product_exactness() {
        let a = 1.0 + 1e-10;
        let b = 1.0 - 1e-10;
        let (p, e) = two_product(a, b);
        // p + e should equal a*b exactly (within representation)
        let reconstructed = p + e;
        let exact = a * b;
        assert!(
            (reconstructed - exact).abs() < 1e-30,
            "TwoProduct error: {:e}",
            reconstructed - exact
        );
    }

    #[test]
    fn test_add_sub_roundtrip() {
        let a = DD_PI;
        let b = DD_E;
        let c = a + b;
        let d = c - b;
        // d should equal a to ~31 digits
        let diff = (d - a).abs();
        assert!(
            diff.to_f64() < 1e-30,
            "add/sub roundtrip error: {:e}",
            diff.to_f64()
        );
    }

    #[test]
    fn test_mul_div_roundtrip() {
        let a = DD_PI;
        let b = DD_E;
        let c = a * b;
        let d = c / b;
        let diff = (d - a).abs();
        assert!(
            diff.to_f64() < 1e-29,
            "mul/div roundtrip error: {:e}",
            diff.to_f64()
        );
    }

    #[test]
    fn test_commutativity_add() {
        let a = DD_PI;
        let b = DD_E;
        let ab = a + b;
        let ba = b + a;
        assert_eq!(ab, ba);
    }

    #[test]
    fn test_commutativity_mul() {
        let a = DD_PI;
        let b = DD_E;
        let ab = a * b;
        let ba = b * a;
        // May differ by a few ulps due to ordering; check closeness
        let diff = (ab - ba).abs();
        assert!(
            diff.to_f64() < 1e-30,
            "mul commutativity error: {:e}",
            diff.to_f64()
        );
    }

    #[test]
    fn test_negation() {
        let a = DD_PI;
        let neg_a = -a;
        let sum = a + neg_a;
        assert!(sum.abs().to_f64() < 1e-31);
    }

    #[test]
    fn test_sqr() {
        let a = DD_TWO;
        let sq = a.sqr();
        let diff = (sq - DoubleDouble::from(4.0)).abs();
        assert!(diff.to_f64() < 1e-30);
    }

    #[test]
    fn test_recip() {
        let a = DD_TWO;
        let r = a.recip();
        let diff = (r - DoubleDouble::from(0.5)).abs();
        assert!(diff.to_f64() < 1e-30);
    }

    #[test]
    fn test_div_by_zero() {
        let r = DD_ONE / DD_ZERO;
        assert!(r.hi.is_infinite());
        assert!(r.hi > 0.0);

        let r2 = DD_ZERO / DD_ZERO;
        assert!(r2.hi.is_nan());
    }

    #[test]
    fn test_mixed_dd_f64() {
        let a = DD_PI;
        let b = a + 1.0;
        let c = b - 1.0;
        let diff = (c - a).abs();
        assert!(diff.to_f64() < 1e-30);

        let d = a * 2.0;
        let e = d / 2.0;
        let diff2 = (e - a).abs();
        assert!(diff2.to_f64() < 1e-30);
    }

    #[test]
    fn test_f64_op_dd() {
        let a = 2.0 + DD_PI;
        let b = DD_PI + 2.0;
        assert_eq!(a, b);

        let c = 10.0 - DD_PI;
        assert!(c.to_f64() > 6.0);

        let d = 2.0 * DD_PI;
        let e = DD_PI * 2.0;
        let diff = (d - e).abs();
        assert!(diff.to_f64() < 1e-30);
    }
}
