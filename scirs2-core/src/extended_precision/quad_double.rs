//! Quad-double (QD) arithmetic: ~62 decimal digits of precision using four `f64` values.
//!
//! A `QD` number represents the unevaluated sum `x0 + x1 + x2 + x3` where the
//! components satisfy a non-overlapping, decreasing-magnitude chain:
//!   `|x1| <= ulp(x0)/2`, `|x2| <= ulp(x1)/2`, `|x3| <= ulp(x2)/2`.
//!
//! This gives roughly 212 bits (~62 significant decimal digits) of precision,
//! four times what a single `f64` provides.
//!
//! ## References
//!
//! * Hida, Y.; Li, X. S.; Bailey, D. H. (2001). "Algorithms for quad-double
//!   precision floating-point arithmetic." *ARITH-15*.
//! * Shewchuk, J. R. (1997). "Adaptive precision floating-point arithmetic."

use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::error::{CoreError, CoreResult, ErrorContext};
use super::{two_sum, two_prod, DD};

// ─── Error helpers ─────────────────────────────────────────────────────────────

#[inline(always)]
fn comp_err(msg: impl Into<String>) -> CoreError {
    CoreError::ComputationError(ErrorContext::new(msg))
}

// ─── QD struct ─────────────────────────────────────────────────────────────────

/// Quad-double precision floating-point number.
///
/// Represents the value `x0 + x1 + x2 + x3` where components satisfy a
/// non-overlapping, decreasing-magnitude invariant maintained by renormalization.
#[derive(Debug, Clone, Copy)]
pub struct QD {
    /// Component 0 (most significant).
    pub x0: f64,
    /// Component 1.
    pub x1: f64,
    /// Component 2.
    pub x2: f64,
    /// Component 3 (least significant).
    pub x3: f64,
}

impl Default for QD {
    fn default() -> Self {
        Self::ZERO
    }
}

impl QD {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Construct a `QD` from a single `f64` (remaining components zero).
    #[inline]
    #[must_use]
    pub fn new(a: f64) -> Self {
        Self { x0: a, x1: 0.0, x2: 0.0, x3: 0.0 }
    }

    /// Construct a `QD` from a pair `(hi, lo)`, filling components 2 and 3 with zero.
    #[inline]
    #[must_use]
    pub fn from_pair(hi: f64, lo: f64) -> Self {
        let (s, e) = two_sum(hi, lo);
        Self { x0: s, x1: e, x2: 0.0, x3: 0.0 }
    }

    /// Construct a `QD` from a `DD`.
    #[inline]
    #[must_use]
    pub fn from_dd(d: DD) -> Self {
        Self { x0: d.hi, x1: d.lo, x2: 0.0, x3: 0.0 }
    }

    /// Construct a `QD` from four raw components and renormalize.
    #[must_use]
    pub fn from_components(a: f64, b: f64, c: f64, d: f64) -> Self {
        let mut q = Self { x0: a, x1: b, x2: c, x3: d };
        q.renormalize();
        q
    }

    // ── Constants ─────────────────────────────────────────────────────────────

    /// Zero.
    pub const ZERO: QD = QD { x0: 0.0, x1: 0.0, x2: 0.0, x3: 0.0 };

    /// One.
    pub const ONE: QD = QD { x0: 1.0, x1: 0.0, x2: 0.0, x3: 0.0 };

    /// Two.
    pub const TWO: QD = QD { x0: 2.0, x1: 0.0, x2: 0.0, x3: 0.0 };

    /// One half.
    pub const HALF: QD = QD { x0: 0.5, x1: 0.0, x2: 0.0, x3: 0.0 };

    /// pi to quad-double precision (~62 digits).
    ///
    /// pi = 3.14159265358979323846264338327950288419716939937510582097494...
    ///
    /// Components from Bailey's QD library tables.
    pub const QD_PI: QD = QD {
        x0: 3.141_592_653_589_793_1_f64,
        x1: 1.224_646_799_147_353_2e-16,
        x2: -2.994_769_809_718_339_7e-33,
        x3: 1.112_454_220_863_365_3e-49,
    };

    /// e = exp(1) to quad-double precision.
    ///
    /// e = 2.71828182845904523536028747135266249775724709369995957496697...
    pub const QD_E: QD = QD {
        x0: 2.718_281_828_459_045_f64,
        x1: 1.445_646_891_729_250_2e-16,
        x2: -2.127_717_108_038_176_8e-33,
        x3: 1.515_630_159_841_218_9e-49,
    };

    /// ln(2) to quad-double precision.
    ///
    /// ln(2) = 0.69314718055994530941723212145817656807550013436025525412068...
    pub const QD_LN2: QD = QD {
        x0: 0.693_147_180_559_945_3_f64,
        x1: 2.319_046_813_846_299_6e-17,
        x2: 5.707_708_438_416_212e-34,
        x3: -3.582_432_210_601_811_4e-50,
    };

    /// ln(10) to quad-double precision.
    ///
    /// ln(10) = 2.30258509299404568401799145468436420760110148862877297603333...
    pub const QD_LN10: QD = QD {
        x0: 2.302_585_092_994_046_f64,
        x1: -2.170_756_223_382_249e-16,
        x2: -9.984_699_484_030_25e-33,
        x3: 4.624_180_205_714_528_5e-49,
    };

    // ── Predicates ────────────────────────────────────────────────────────────

    /// Returns `true` if all components are zero.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.x0 == 0.0
    }

    /// Returns `true` if x0 is finite.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.x0.is_finite()
    }

    /// Returns `true` if x0 is NaN.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.x0.is_nan()
    }

    /// Returns `true` if the value is negative.
    #[inline]
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.x0 < 0.0
    }

    /// Returns `true` if the value is positive.
    #[inline]
    #[must_use]
    pub fn is_positive(&self) -> bool {
        self.x0 > 0.0
    }

    // ── Conversion ────────────────────────────────────────────────────────────

    /// Convert to `f64` (just x0).
    #[inline]
    #[must_use]
    pub fn to_f64(self) -> f64 {
        self.x0
    }

    /// Convert to `f64` with full compensation.
    #[inline]
    #[must_use]
    pub fn to_f64_round(self) -> f64 {
        self.x0 + self.x1 + self.x2 + self.x3
    }

    /// Convert to a `DD` (keeps only x0, x1).
    #[inline]
    #[must_use]
    pub fn to_dd(self) -> DD {
        DD::from_parts(self.x0, self.x1)
    }

    // ── Renormalization ──────────────────────────────────────────────────────

    /// Renormalize so that the components form a non-overlapping,
    /// decreasing-magnitude chain.
    ///
    /// Uses the renormalization algorithm from Hida, Li, Bailey (2001).
    pub fn renormalize(&mut self) {
        // First pass: propagate errors from bottom to top via TwoSum.
        let (s, t3) = two_sum(self.x2, self.x3);
        let (s, t2) = two_sum(self.x1, s);
        let (s0, t1) = two_sum(self.x0, s);

        // Second pass: propagate from top to bottom, collecting
        // non-zero components into the output array.
        let mut c = [0.0_f64; 4];
        #[allow(unused_assignments)]
        let mut k = 0usize;
        let mut s_val = s0;
        let ts = [t1, t2, t3];

        for &t in &ts {
            let (s_new, e) = two_sum(s_val, t);
            if e != 0.0 {
                if k < 4 {
                    c[k] = s_new;
                    k += 1;
                }
                s_val = e;
            } else {
                s_val = s_new;
            }
        }

        if k < 4 {
            c[k] = s_val;
        }

        self.x0 = c[0];
        self.x1 = c[1];
        self.x2 = c[2];
        self.x3 = c[3];
    }

    // ── Arithmetic ────────────────────────────────────────────────────────────

    /// Negate.
    #[inline]
    #[must_use]
    pub fn negate(self) -> QD {
        QD { x0: -self.x0, x1: -self.x1, x2: -self.x2, x3: -self.x3 }
    }

    /// Absolute value.
    #[inline]
    #[must_use]
    pub fn abs(self) -> QD {
        if self.x0 < 0.0 { self.negate() } else { self }
    }

    /// Compare `self` with `rhs` for ordering.
    #[must_use]
    pub fn compare(&self, rhs: &QD) -> Ordering {
        if self.x0 < rhs.x0 { return Ordering::Less; }
        if self.x0 > rhs.x0 { return Ordering::Greater; }
        if self.x1 < rhs.x1 { return Ordering::Less; }
        if self.x1 > rhs.x1 { return Ordering::Greater; }
        if self.x2 < rhs.x2 { return Ordering::Less; }
        if self.x2 > rhs.x2 { return Ordering::Greater; }
        if self.x3 < rhs.x3 { return Ordering::Less; }
        if self.x3 > rhs.x3 { return Ordering::Greater; }
        Ordering::Equal
    }
}


// ─── Arithmetic functions ───────────────────────────────────────────────────

/// Add two QD numbers using Shewchuk's algorithm extended to 4 components.
///
/// Merges the 8 components (4 from each operand) in decreasing magnitude
/// and accumulates via `two_sum` to produce a 4-component result.
#[must_use]
pub fn qd_add(a: &QD, b: &QD) -> QD {
    // Merge-based addition following Hida/Li/Bailey.
    // Step 1: pairwise two_sum on corresponding components.
    let (s0, e0) = two_sum(a.x0, b.x0);
    let (s1, e1) = two_sum(a.x1, b.x1);
    let (s2, e2) = two_sum(a.x2, b.x2);
    let (s3, e3) = two_sum(a.x3, b.x3);

    // Step 2: accumulate carries.
    let (s1, e0b) = two_sum(s1, e0);
    let (s2, e1b) = two_sum(s2, e1);
    let (s2, e0c) = two_sum(s2, e0b);
    let (s3, e2b) = two_sum(s3, e2);
    let (s3, e1c) = two_sum(s3, e1b);
    let (s3, e0d) = two_sum(s3, e0c);

    // Collect remaining error terms.
    let t = e3 + e2b + e1c + e0d;

    QD::from_components(s0, s1, s2, s3 + t)
}

/// Subtract two QD numbers: a - b.
#[must_use]
pub fn qd_sub(a: &QD, b: &QD) -> QD {
    qd_add(a, &b.negate())
}

/// Multiply two QD numbers using TwoProd and accumulation.
///
/// Uses the "sloppy" multiplication algorithm from Hida/Li/Bailey (2001)
/// which is sufficient for maintaining ~62 digits of accuracy.
#[must_use]
pub fn qd_mul(a: &QD, b: &QD) -> QD {
    // Compute the cross products that contribute to each "level".
    // Level 0: a.x0 * b.x0
    let (p0, q0) = two_prod(a.x0, b.x0);

    // Level 1: a.x0*b.x1 + a.x1*b.x0 + q0
    let (p1, q1) = two_prod(a.x0, b.x1);
    let (p2, q2) = two_prod(a.x1, b.x0);
    let (t1, e1) = two_sum(q0, p1);
    let (t1, e2) = two_sum(t1, p2);
    let carry1 = e1 + e2 + q1 + q2;

    // Level 2: a.x0*b.x2 + a.x1*b.x1 + a.x2*b.x0 + carry1
    let (p3, q3) = two_prod(a.x0, b.x2);
    let (p4, q4) = two_prod(a.x1, b.x1);
    let (p5, q5) = two_prod(a.x2, b.x0);
    let (t2, e3) = two_sum(carry1, p3);
    let (t2, e4) = two_sum(t2, p4);
    let (t2, e5) = two_sum(t2, p5);
    let carry2 = e3 + e4 + e5 + q3 + q4 + q5;

    // Level 3: a.x0*b.x3 + a.x1*b.x2 + a.x2*b.x1 + a.x3*b.x0 + carry2
    let t3 = a.x0 * b.x3
        + a.x1 * b.x2
        + a.x2 * b.x1
        + a.x3 * b.x0
        + carry2;

    QD::from_components(p0, t1, t2, t3)
}

/// Divide two QD numbers: a / b, using Newton iteration on 1/b then multiply.
///
/// Returns `Err` on division by zero.
pub fn qd_div(a: &QD, b: &QD) -> CoreResult<QD> {
    if b.is_zero() {
        return Err(comp_err("qd_div: division by zero"));
    }

    // Initial approximation: q0 = a.x0 / b.x0
    let q0 = a.x0 / b.x0;

    // Compute residual: r = a - q0 * b
    let q0_qd = QD::new(q0);
    let r = qd_sub(a, &qd_mul(&q0_qd, b));

    // Next approximation
    let q1 = r.x0 / b.x0;
    let q1_qd = QD::new(q1);
    let r = qd_sub(&r, &qd_mul(&q1_qd, b));

    let q2 = r.x0 / b.x0;
    let q2_qd = QD::new(q2);
    let r = qd_sub(&r, &qd_mul(&q2_qd, b));

    let q3 = r.x0 / b.x0;

    let _ = q0_qd;
    let _ = q1_qd;
    let _ = q2_qd;

    Ok(QD::from_components(q0, q1, q2, q3))
}

/// Multiply a QD by a scalar f64.
#[must_use]
pub fn qd_mul_f64(a: &QD, b: f64) -> QD {
    let (p0, q0) = two_prod(a.x0, b);
    let (p1, q1) = two_prod(a.x1, b);
    let (p2, q2) = two_prod(a.x2, b);
    let p3 = a.x3 * b;

    // Accumulate
    let (s1, e0) = two_sum(q0, p1);
    let (s2, e1) = two_sum(q1, p2);
    let (s2, e0b) = two_sum(s2, e0);
    let s3 = q2 + p3 + e1 + e0b;

    QD::from_components(p0, s1, s2, s3)
}

/// Add a scalar f64 to a QD.
#[must_use]
pub fn qd_add_f64(a: &QD, b: f64) -> QD {
    let (s0, e0) = two_sum(a.x0, b);
    let (s1, e1) = two_sum(a.x1, e0);
    let (s2, e2) = two_sum(a.x2, e1);
    let s3 = a.x3 + e2;
    QD::from_components(s0, s1, s2, s3)
}

/// Square a QD number (slightly more efficient than qd_mul(a, a)).
#[must_use]
pub fn qd_square(a: &QD) -> QD {
    let (p0, q0) = two_prod(a.x0, a.x0);

    let (p1, q1) = two_prod(2.0 * a.x0, a.x1);
    let (t1, e1) = two_sum(q0, p1);
    let carry1 = e1 + q1;

    let (p2, q2) = two_prod(2.0 * a.x0, a.x2);
    let (p3, q3) = two_prod(a.x1, a.x1);
    let (t2, e2) = two_sum(carry1, p2);
    let (t2, e3) = two_sum(t2, p3);
    let carry2 = e2 + e3 + q2 + q3;

    let t3 = 2.0 * a.x0 * a.x3 + 2.0 * a.x1 * a.x2 + carry2;

    QD::from_components(p0, t1, t2, t3)
}

// ─── Trait implementations ────────────────────────────────────────────────────

impl PartialEq for QD {
    fn eq(&self, other: &Self) -> bool {
        self.x0 == other.x0 && self.x1 == other.x1
            && self.x2 == other.x2 && self.x3 == other.x3
    }
}

impl PartialOrd for QD {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.compare(other))
    }
}

impl Neg for QD {
    type Output = QD;
    fn neg(self) -> QD {
        self.negate()
    }
}

impl Add for QD {
    type Output = QD;
    fn add(self, rhs: QD) -> QD {
        qd_add(&self, &rhs)
    }
}

impl Sub for QD {
    type Output = QD;
    fn sub(self, rhs: QD) -> QD {
        qd_sub(&self, &rhs)
    }
}

impl Mul for QD {
    type Output = QD;
    fn mul(self, rhs: QD) -> QD {
        qd_mul(&self, &rhs)
    }
}

/// Division via `qd_div`; returns NaN QD if rhs is zero.
impl Div for QD {
    type Output = QD;
    fn div(self, rhs: QD) -> QD {
        qd_div(&self, &rhs).unwrap_or(QD {
            x0: f64::NAN, x1: f64::NAN, x2: f64::NAN, x3: f64::NAN,
        })
    }
}

impl fmt::Display for QD {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format to ~62 significant digits.
        // We use a careful digit-extraction approach.
        if self.is_nan() {
            return write!(f, "NaN");
        }
        if !self.is_finite() {
            if self.x0 > 0.0 {
                return write!(f, "inf");
            } else {
                return write!(f, "-inf");
            }
        }
        if self.is_zero() {
            return write!(f, "0.0");
        }

        // For display, compute the full value through string extraction.
        // Use a simpler approach: format each component contribution.
        let val = *self;
        let sign = if val.is_negative() { "-" } else { "" };
        let abs_val = val.abs();

        // Extract digits by repeated multiplication.
        let exp10 = abs_val.x0.log10().floor() as i32;
        let mut digits = alloc::string::String::with_capacity(70);

        // Scale to [1, 10)
        let mut scaled = abs_val;
        if exp10 != 0 {
            // Compute 10^(-exp10) in QD
            let pow10 = qd_pow10(-exp10);
            scaled = qd_mul(&scaled, &pow10);
        }

        // Extract up to 62 digits
        for i in 0..62 {
            let d = scaled.x0.floor();
            let d_clamped = if d < 0.0 { 0.0 } else if d > 9.0 { 9.0 } else { d };
            let digit = d_clamped as u8;
            digits.push((b'0' + digit) as char);
            if i == 0 {
                digits.push('.');
            }
            scaled = qd_sub(&scaled, &QD::new(d_clamped));
            scaled = qd_mul_f64(&scaled, 10.0);
        }

        write!(f, "{sign}{digits}e{exp10:+}")
    }
}

/// Compute 10^n in QD precision for display purposes.
fn qd_pow10(n: i32) -> QD {
    if n == 0 {
        return QD::ONE;
    }
    let mut result = QD::ONE;
    let mut base = if n > 0 {
        QD::new(10.0)
    } else {
        QD::new(0.1)
    };
    let mut exp = n.unsigned_abs();
    while exp > 0 {
        if exp & 1 == 1 {
            result = qd_mul(&result, &base);
        }
        base = qd_square(&base);
        exp >>= 1;
    }
    result
}

impl From<f64> for QD {
    fn from(x: f64) -> QD {
        QD::new(x)
    }
}

impl From<DD> for QD {
    fn from(d: DD) -> QD {
        QD::from_dd(d)
    }
}

impl From<i32> for QD {
    fn from(x: i32) -> QD {
        QD::new(x as f64)
    }
}

impl From<i64> for QD {
    fn from(x: i64) -> QD {
        let hi = x as f64;
        let lo = (x - hi as i64) as f64;
        QD::from_pair(hi, lo)
    }
}

// We need alloc for String in Display
extern crate alloc;

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qd_new_and_zero() {
        let z = QD::ZERO;
        assert!(z.is_zero());
        let one = QD::ONE;
        assert!(!one.is_zero());
        assert_eq!(one.x0, 1.0);
    }

    #[test]
    fn test_qd_add_basic() {
        let a = QD::new(1.0);
        let b = QD::new(2.0);
        let c = qd_add(&a, &b);
        assert_eq!(c.x0, 3.0);
    }

    #[test]
    fn test_qd_sub_basic() {
        let a = QD::new(5.0);
        let b = QD::new(3.0);
        let c = qd_sub(&a, &b);
        assert_eq!(c.x0, 2.0);
    }

    #[test]
    fn test_qd_mul_basic() {
        let a = QD::new(3.0);
        let b = QD::new(4.0);
        let c = qd_mul(&a, &b);
        assert_eq!(c.x0, 12.0);
    }

    #[test]
    fn test_qd_div_basic() {
        let a = QD::new(10.0);
        let b = QD::new(4.0);
        let c = qd_div(&a, &b).expect("division should succeed");
        let diff = (c.x0 - 2.5).abs();
        assert!(diff < f64::EPSILON * 4.0, "10/4 should be 2.5, got {}", c.x0);
    }

    #[test]
    fn test_qd_div_zero() {
        let a = QD::new(1.0);
        let b = QD::ZERO;
        assert!(qd_div(&a, &b).is_err());
    }

    #[test]
    fn test_qd_one_third_times_three() {
        // (1/3) * 3 should be very close to 1 in QD.
        let one = QD::ONE;
        let three = QD::new(3.0);
        let third = qd_div(&one, &three).expect("division should succeed");
        let result = qd_mul(&third, &three);
        let diff = qd_sub(&result, &one);
        // Should be close to zero to ~60 digits.
        assert!(
            diff.abs().x0 < 1e-60,
            "(1/3)*3 - 1 = {:e}, expected < 1e-60",
            diff.x0
        );
    }

    #[test]
    fn test_qd_operator_overloads() {
        let a = QD::new(3.0);
        let b = QD::new(4.0);
        let sum = a + b;
        let diff = a - b;
        let prod = a * b;
        let quot = a / b;
        assert_eq!(sum.x0, 7.0);
        assert_eq!(diff.x0, -1.0);
        assert_eq!(prod.x0, 12.0);
        assert!((quot.x0 - 0.75).abs() < f64::EPSILON * 4.0);
    }

    #[test]
    fn test_qd_partial_ord() {
        let a = QD::new(1.0);
        let b = QD::new(2.0);
        assert!(a < b);
        assert!(b > a);
        assert!(a <= a);
        assert!(a == a);
    }

    #[test]
    fn test_qd_negate() {
        let a = QD::new(3.14);
        let b = a.negate();
        assert_eq!(b.x0, -3.14);
        let c = qd_add(&a, &b);
        assert!(c.is_zero() || c.x0.abs() < 1e-300);
    }

    #[test]
    fn test_qd_from_dd() {
        let d = DD::from_parts(1.0, 1e-17);
        let q = QD::from_dd(d);
        assert_eq!(q.x0, d.hi);
        assert_eq!(q.x1, d.lo);
    }

    #[test]
    fn test_qd_pi_first_component() {
        let pi = QD::QD_PI;
        let diff = (pi.x0 - core::f64::consts::PI).abs();
        assert!(diff < f64::EPSILON * 2.0, "QD_PI x0 error: {diff}");
        // lo components should be non-zero
        assert!(pi.x1.abs() > 0.0, "QD_PI x1 should be non-zero");
    }

    #[test]
    fn test_qd_display() {
        let x = QD::new(1.5);
        let s = format!("{x}");
        assert!(s.contains('1'), "Display should contain 1: {s}");
    }

    #[test]
    fn test_qd_square() {
        let x = QD::new(3.0);
        let sq = qd_square(&x);
        assert_eq!(sq.x0, 9.0);
    }
}
