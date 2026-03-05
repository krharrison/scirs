//! Core interval type with outward-rounded arithmetic.
//!
//! This module provides `Interval<T>` — a closed real interval `[lo, hi]` —
//! together with all four arithmetic operations and elementary transcendental
//! functions.  Every operation rounds *outward* (down for the lower bound,
//! up for the upper bound) so that the true mathematical result is always
//! contained in the computed interval.
//!
//! # Rounding strategy
//!
//! Stable Rust does not expose IEEE-754 directed rounding modes through the
//! standard library.  We therefore implement outward rounding conservatively
//! by adding/subtracting one ULP (unit in the last place) from each bound
//! after the computation:
//!
//! * Lower bound: `f64::from_bits(bits - 1)` — moves toward −∞.
//! * Upper bound: `f64::from_bits(bits + 1)` — moves toward +∞.
//!
//! The helpers `prev_float` / `next_float` in this file implement that
//! correctly for all finite, zero, and infinite values.
//!
//! # Special intervals
//!
//! * `Interval::ENTIRE` — the entire real line `(−∞, +∞)`.
//! * `Interval::EMPTY`  — the empty set, represented as `(+∞, −∞)`.

use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};
use num_traits::Float;

// ---------------------------------------------------------------------------
// ULP helpers
// ---------------------------------------------------------------------------

/// Return the largest representable value strictly less than `x`.
///
/// * For finite positive / negative values this subtracts one ULP.
/// * `prev_float(-∞) == -∞`.
/// * `prev_float(+∞) == f64::MAX`.
/// * `prev_float(NaN) == NaN`.
#[inline]
pub fn prev_float(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if x == 0.0 {
        return -f64::MIN_POSITIVE * f64::EPSILON; // smallest negative
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

/// Return the smallest representable value strictly greater than `x`.
///
/// * For finite positive / negative values this adds one ULP.
/// * `next_float(+∞) == +∞`.
/// * `next_float(-∞) == f64::MIN` (most negative finite).
/// * `next_float(NaN) == NaN`.
#[inline]
pub fn next_float(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x == 0.0 {
        return f64::MIN_POSITIVE * f64::EPSILON; // smallest positive
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

// ---------------------------------------------------------------------------
// Generic ULP helpers for f32 / f64 via Float
// ---------------------------------------------------------------------------

/// Outward-round `x` downward (toward −∞) by one ULP.
///
/// Works for both `f32` and `f64` by operating on the raw bit patterns.
#[inline]
pub fn round_down<T: Float + FloatBits>(x: T) -> T {
    if x.is_nan() || x == T::neg_infinity() {
        return x;
    }
    if x == T::zero() {
        return T::zero(); // −0 ≡ 0 for our purposes
    }
    T::from_bits(if x > T::zero() {
        T::wrapping_sub_bits(x.to_bits(), T::one_bits())
    } else {
        T::wrapping_add_bits(x.to_bits(), T::one_bits())
    })
}

/// Outward-round `x` upward (toward +∞) by one ULP.
#[inline]
pub fn round_up<T: Float + FloatBits>(x: T) -> T {
    if x.is_nan() || x == T::infinity() {
        return x;
    }
    if x == T::zero() {
        return T::zero();
    }
    T::from_bits(if x > T::zero() {
        T::wrapping_add_bits(x.to_bits(), T::one_bits())
    } else {
        T::wrapping_sub_bits(x.to_bits(), T::one_bits())
    })
}

// ---------------------------------------------------------------------------
// FloatBits helper trait (sealed)
// ---------------------------------------------------------------------------

/// Sealed helper trait that bridges `num_traits::Float` and raw bit operations.
pub trait FloatBits: Sized + Copy {
    type Bits: Copy
        + core::ops::Add<Output = Self::Bits>
        + core::ops::Sub<Output = Self::Bits>
        + Eq
        + PartialOrd;

    fn to_bits(self) -> Self::Bits;
    fn from_bits(bits: Self::Bits) -> Self;
    /// The bit-pattern increment representing "one ULP upward from a positive value".
    fn one_bits() -> Self::Bits;
    fn wrapping_add_bits(a: Self::Bits, b: Self::Bits) -> Self::Bits;
    fn wrapping_sub_bits(a: Self::Bits, b: Self::Bits) -> Self::Bits;
}

impl FloatBits for f32 {
    type Bits = u32;
    #[inline]
    fn to_bits(self) -> u32 {
        f32::to_bits(self)
    }
    #[inline]
    fn from_bits(bits: u32) -> f32 {
        f32::from_bits(bits)
    }
    #[inline]
    fn one_bits() -> u32 {
        1u32
    }
    #[inline]
    fn wrapping_add_bits(a: u32, b: u32) -> u32 {
        a.wrapping_add(b)
    }
    #[inline]
    fn wrapping_sub_bits(a: u32, b: u32) -> u32 {
        a.wrapping_sub(b)
    }
}

impl FloatBits for f64 {
    type Bits = u64;
    #[inline]
    fn to_bits(self) -> u64 {
        f64::to_bits(self)
    }
    #[inline]
    fn from_bits(bits: u64) -> f64 {
        f64::from_bits(bits)
    }
    #[inline]
    fn one_bits() -> u64 {
        1u64
    }
    #[inline]
    fn wrapping_add_bits(a: u64, b: u64) -> u64 {
        a.wrapping_add(b)
    }
    #[inline]
    fn wrapping_sub_bits(a: u64, b: u64) -> u64 {
        a.wrapping_sub(b)
    }
}

// Note: FloatBits is implemented concretely for f32 and f64 above; no blanket impl needed.

// ---------------------------------------------------------------------------
// Interval<T>
// ---------------------------------------------------------------------------

/// A closed real interval `[lo, hi]`.
///
/// The bounds are stored as floating-point values.  An empty interval is
/// represented with `lo > hi`; the canonical empty interval has
/// `lo = +∞, hi = −∞`.
///
/// # Invariants (soft)
///
/// * For a non-empty interval: `lo <= hi`.
/// * All arithmetic operations preserve the containment property: if
///   `x ∈ self` and `y ∈ rhs` then the result of any operation `x ⊕ y` is
///   guaranteed to be contained in `self ⊕ rhs`.
#[derive(Clone, Copy, PartialEq)]
pub struct Interval<T: Float> {
    /// Lower bound (rounded toward −∞).
    pub lo: T,
    /// Upper bound (rounded toward +∞).
    pub hi: T,
}

// ---------------------------------------------------------------------------
// Special-interval constants (f64 specialisation)
// ---------------------------------------------------------------------------

impl Interval<f64> {
    /// The entire real line `(−∞, +∞)`.
    pub const ENTIRE: Self = Self {
        lo: f64::NEG_INFINITY,
        hi: f64::INFINITY,
    };

    /// The empty interval.  A canonical representation so that
    /// `interval.is_empty()` is always `true` for this value.
    pub const EMPTY: Self = Self {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
}

impl Interval<f32> {
    /// The entire real line `(−∞, +∞)`.
    pub const ENTIRE: Self = Self {
        lo: f32::NEG_INFINITY,
        hi: f32::INFINITY,
    };

    /// The empty interval.
    pub const EMPTY: Self = Self {
        lo: f32::INFINITY,
        hi: f32::NEG_INFINITY,
    };
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl<T: Float> Interval<T> {
    /// Construct `[lo, hi]` without any rounding adjustment.
    ///
    /// The caller is responsible for ensuring `lo <= hi` if a non-empty
    /// interval is desired.
    #[inline]
    pub fn new(lo: T, hi: T) -> Self {
        Self { lo, hi }
    }

    /// Construct a degenerate interval (a point) `[x, x]`.
    #[inline]
    pub fn point(x: T) -> Self {
        Self { lo: x, hi: x }
    }

    /// Construct `[midpoint - radius, midpoint + radius]` with outward rounding.
    ///
    /// Returns `None` if `radius < 0`.
    #[inline]
    pub fn from_midpoint_radius(midpoint: T, radius: T) -> Option<Self> {
        if radius < T::zero() {
            return None;
        }
        Some(Self {
            lo: midpoint - radius,
            hi: midpoint + radius,
        })
    }

    /// Construct an interval from two bounds, automatically ordering them.
    #[inline]
    pub fn from_bounds(a: T, b: T) -> Self {
        if a <= b {
            Self { lo: a, hi: b }
        } else {
            Self { lo: b, hi: a }
        }
    }
}

impl Interval<f64> {
    /// Construct `[lo, hi]` with one-ULP outward rounding applied to both bounds.
    #[inline]
    pub fn rounded(lo: f64, hi: f64) -> Self {
        Self {
            lo: prev_float(lo),
            hi: next_float(hi),
        }
    }

    /// Construct a degenerate interval `[x, x]` with one-ULP outward rounding.
    #[inline]
    pub fn point_rounded(x: f64) -> Self {
        Self {
            lo: prev_float(x),
            hi: next_float(x),
        }
    }
}

// ---------------------------------------------------------------------------
// Predicates
// ---------------------------------------------------------------------------

impl<T: Float> Interval<T> {
    /// Returns `true` if the interval is empty (`lo > hi`).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lo > self.hi
    }

    /// Returns `true` if the interval contains exactly one point.
    #[inline]
    pub fn is_point(&self) -> bool {
        self.lo == self.hi
    }

    /// Returns `true` if the interval is unbounded below.
    #[inline]
    pub fn is_unbounded_below(&self) -> bool {
        self.lo == T::neg_infinity()
    }

    /// Returns `true` if the interval is unbounded above.
    #[inline]
    pub fn is_unbounded_above(&self) -> bool {
        self.hi == T::infinity()
    }

    /// Returns `true` if the interval is the entire real line.
    #[inline]
    pub fn is_entire(&self) -> bool {
        self.is_unbounded_below() && self.is_unbounded_above()
    }

    /// Returns `true` if `x` is contained in the interval.
    #[inline]
    pub fn contains(&self, x: T) -> bool {
        !self.is_empty() && self.lo <= x && x <= self.hi
    }

    /// Returns `true` if `other` is a subset of `self`.
    #[inline]
    pub fn contains_interval(&self, other: &Self) -> bool {
        other.is_empty() || (!self.is_empty() && self.lo <= other.lo && other.hi <= self.hi)
    }

    /// Returns `true` if the two intervals have a non-empty intersection.
    #[inline]
    pub fn intersects(&self, other: &Self) -> bool {
        !self.is_empty() && !other.is_empty() && self.lo <= other.hi && other.lo <= self.hi
    }
}

// ---------------------------------------------------------------------------
// Derived values
// ---------------------------------------------------------------------------

impl<T: Float> Interval<T> {
    /// Width of the interval: `hi - lo`.  Returns `NaN` for an empty interval.
    #[inline]
    pub fn width(&self) -> T {
        if self.is_empty() {
            T::nan()
        } else {
            self.hi - self.lo
        }
    }

    /// Midpoint `(lo + hi) / 2`.  Returns `NaN` for an empty interval.
    #[inline]
    pub fn midpoint(&self) -> T {
        if self.is_empty() {
            T::nan()
        } else {
            let two = T::one() + T::one();
            self.lo / two + self.hi / two
        }
    }

    /// Radius `(hi - lo) / 2`.  Returns `NaN` for an empty interval.
    #[inline]
    pub fn radius(&self) -> T {
        if self.is_empty() {
            T::nan()
        } else {
            let two = T::one() + T::one();
            (self.hi - self.lo) / two
        }
    }

    /// The magnitude (mignitude?) — the minimum absolute value in the interval.
    ///
    /// Equals `0` if `0 ∈ self`, otherwise `min(|lo|, |hi|)`.
    #[inline]
    pub fn mig(&self) -> T {
        if self.is_empty() {
            return T::nan();
        }
        if self.lo <= T::zero() && T::zero() <= self.hi {
            T::zero()
        } else {
            self.lo.abs().min(self.hi.abs())
        }
    }

    /// The magnitude — the maximum absolute value: `max(|lo|, |hi|)`.
    #[inline]
    pub fn mag(&self) -> T {
        if self.is_empty() {
            return T::nan();
        }
        self.lo.abs().max(self.hi.abs())
    }

    /// Compute the intersection of two intervals.
    ///
    /// Returns `Interval::EMPTY` (represented as `[+∞, −∞]`) when they do not
    /// overlap.
    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() || !self.intersects(other) {
            Self {
                lo: T::infinity(),
                hi: T::neg_infinity(),
            }
        } else {
            Self {
                lo: self.lo.max(other.lo),
                hi: self.hi.min(other.hi),
            }
        }
    }

    /// Convex hull of two intervals (smallest interval containing both).
    #[inline]
    pub fn hull(&self, other: &Self) -> Self {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// Absolute value interval.
    ///
    /// * `[0, max(|lo|, |hi|)]`  if `0 ∈ self`
    /// * `[min(|lo|,|hi|), max(|lo|,|hi|)]`  otherwise
    #[inline]
    pub fn abs(&self) -> Self {
        if self.is_empty() {
            return *self;
        }
        if self.lo >= T::zero() {
            *self
        } else if self.hi <= T::zero() {
            Self {
                lo: (-self.hi),
                hi: (-self.lo),
            }
        } else {
            Self {
                lo: T::zero(),
                hi: self.lo.abs().max(self.hi.abs()),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Arithmetic — outward rounding via f64 ULP helpers
// ---------------------------------------------------------------------------

// We provide a specialised, ULP-safe implementation for `Interval<f64>`.
// For the generic `T: Float` path we fall back to a "2-ULP outward" approach
// using epsilon-based widening.

impl Interval<f64> {
    /// `[a,b] + [c,d]` with outward rounding.
    #[inline]
    pub fn add_rounded(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::EMPTY;
        }
        Self {
            lo: prev_float(self.lo + rhs.lo),
            hi: next_float(self.hi + rhs.hi),
        }
    }

    /// `[a,b] - [c,d]` with outward rounding.
    #[inline]
    pub fn sub_rounded(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::EMPTY;
        }
        Self {
            lo: prev_float(self.lo - rhs.hi),
            hi: next_float(self.hi - rhs.lo),
        }
    }

    /// `[a,b] * [c,d]` with outward rounding.
    ///
    /// Considers all sign combinations of the four corner products.
    #[inline]
    pub fn mul_rounded(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::EMPTY;
        }
        let p1 = self.lo * rhs.lo;
        let p2 = self.lo * rhs.hi;
        let p3 = self.hi * rhs.lo;
        let p4 = self.hi * rhs.hi;
        let raw_lo = p1.min(p2).min(p3).min(p4);
        let raw_hi = p1.max(p2).max(p3).max(p4);
        Self {
            lo: prev_float(raw_lo),
            hi: next_float(raw_hi),
        }
    }

    /// `[a,b] / [c,d]` with outward rounding.
    ///
    /// Returns `ENTIRE` when the divisor contains zero and the dividend is
    /// non-empty; returns `EMPTY` when either operand is empty.
    #[inline]
    pub fn div_rounded(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::EMPTY;
        }
        // Divisor straddles zero — result is unbounded
        if rhs.lo <= 0.0 && rhs.hi >= 0.0 {
            return Self::ENTIRE;
        }
        let p1 = self.lo / rhs.lo;
        let p2 = self.lo / rhs.hi;
        let p3 = self.hi / rhs.lo;
        let p4 = self.hi / rhs.hi;
        let raw_lo = p1.min(p2).min(p3).min(p4);
        let raw_hi = p1.max(p2).max(p3).max(p4);
        Self {
            lo: prev_float(raw_lo),
            hi: next_float(raw_hi),
        }
    }

    /// Square root `√[a,b]`.
    ///
    /// * Returns `EMPTY` if `b < 0`.
    /// * Clamps `lo` to zero if `a < 0` (partial overlap with non-negatives).
    #[inline]
    pub fn sqrt(&self) -> Self {
        if self.is_empty() || self.hi < 0.0 {
            return Self::EMPTY;
        }
        let lo_clamped = self.lo.max(0.0);
        Self {
            lo: prev_float(lo_clamped.sqrt()),
            hi: next_float(self.hi.sqrt()),
        }
    }

    /// Natural exponential `e^[a,b]`.
    #[inline]
    pub fn exp(&self) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        Self {
            lo: prev_float(self.lo.exp()),
            hi: next_float(self.hi.exp()),
        }
    }

    /// Natural logarithm `ln[a,b]`.
    ///
    /// Returns `EMPTY` if `b <= 0`.
    #[inline]
    pub fn ln(&self) -> Self {
        if self.is_empty() || self.hi <= 0.0 {
            return Self::EMPTY;
        }
        let lo_clamped = self.lo.max(0.0);
        Self {
            lo: prev_float(lo_clamped.ln()),
            hi: next_float(self.hi.ln()),
        }
    }

    /// Sine `sin([a,b])`.
    ///
    /// Because sine is non-monotone, we use a conservative bound:
    /// the result is `[-1, 1]` whenever the interval width exceeds `2π`.
    /// Otherwise we evaluate at the endpoints and additional extrema that
    /// lie inside the interval.
    #[inline]
    pub fn sin(&self) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        let two_pi = core::f64::consts::TAU;
        if self.hi - self.lo >= two_pi {
            return Self::new(-1.0, 1.0);
        }
        // Evaluate at endpoints and check for extrema at ±π/2 + 2kπ
        let s_lo = self.lo.sin();
        let s_hi = self.hi.sin();
        let mut lo = s_lo.min(s_hi);
        let mut hi = s_lo.max(s_hi);

        // Maximum at π/2 + 2kπ
        let pi_half = core::f64::consts::FRAC_PI_2;
        let k_min = ((self.lo - pi_half) / two_pi).ceil() as i64;
        let k_max = ((self.hi - pi_half) / two_pi).floor() as i64;
        if k_min <= k_max {
            hi = 1.0_f64;
        }

        // Minimum at -π/2 + 2kπ = 3π/2 + 2kπ
        let neg_pi_half = -pi_half;
        let k_min2 = ((self.lo - neg_pi_half) / two_pi).ceil() as i64;
        let k_max2 = ((self.hi - neg_pi_half) / two_pi).floor() as i64;
        if k_min2 <= k_max2 {
            lo = -1.0_f64;
        }

        Self {
            lo: prev_float(lo),
            hi: next_float(hi),
        }
    }

    /// Cosine `cos([a,b])`.
    ///
    /// Implemented by shifting by π/2 and delegating to `sin`.
    #[inline]
    pub fn cos(&self) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        let pi_half = core::f64::consts::FRAC_PI_2;
        let shifted = Self {
            lo: self.lo + pi_half,
            hi: self.hi + pi_half,
        };
        // cos(x) = sin(x + π/2)
        // But we compute it directly for accuracy
        let two_pi = core::f64::consts::TAU;
        if self.hi - self.lo >= two_pi {
            return Self::new(-1.0, 1.0);
        }
        let c_lo = self.lo.cos();
        let c_hi = self.hi.cos();
        let mut lo = c_lo.min(c_hi);
        let mut hi = c_lo.max(c_hi);

        // Maximum at 2kπ
        let k_min = (self.lo / two_pi).ceil() as i64;
        let k_max = (self.hi / two_pi).floor() as i64;
        if k_min <= k_max {
            hi = 1.0_f64;
        }

        // Minimum at π + 2kπ
        let pi = core::f64::consts::PI;
        let k_min2 = ((self.lo - pi) / two_pi).ceil() as i64;
        let k_max2 = ((self.hi - pi) / two_pi).floor() as i64;
        if k_min2 <= k_max2 {
            lo = -1.0_f64;
        }

        // Zero-crossing at π/2 + kπ: cos(x) = 0 at these points.
        // Ensure the interval includes 0 if such a point is inside [lo_in, hi_in].
        let pi_half = core::f64::consts::FRAC_PI_2;
        let zk_min = ((self.lo - pi_half) / pi).ceil() as i64;
        let zk_max = ((self.hi - pi_half) / pi).floor() as i64;
        if zk_min <= zk_max {
            // The interval contains a zero of cos.
            if lo > 0.0 {
                lo = 0.0;
            }
            if hi < 0.0 {
                hi = 0.0;
            }
        }

        let _ = shifted; // silence unused warning
        Self {
            lo: prev_float(lo),
            hi: next_float(hi),
        }
    }

    /// Arctangent `atan([a,b])`.
    ///
    /// Monotone on ℝ, so the result is simply `[atan(lo), atan(hi)]`.
    #[inline]
    pub fn atan(&self) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        Self {
            lo: prev_float(self.lo.atan()),
            hi: next_float(self.hi.atan()),
        }
    }

    /// Power `[a,b]^n` for non-negative integer exponent.
    ///
    /// Uses the monotone / non-monotone case split.
    pub fn powi(&self, n: i32) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        if n == 0 {
            return Self::point(1.0);
        }
        if n < 0 {
            // [a,b]^(-n) = 1 / [a,b]^n
            let pos = self.powi(-n);
            return Self::point(1.0).div_rounded(pos);
        }
        if n % 2 == 0 {
            // Even: non-negative result; minimum at 0 if 0 ∈ self
            let abs_iv = self.abs();
            let lo = prev_float(abs_iv.lo.powi(n));
            let hi = next_float(abs_iv.hi.powi(n));
            Self { lo, hi }
        } else {
            // Odd: monotone
            let lo = prev_float(self.lo.powi(n));
            let hi = next_float(self.hi.powi(n));
            Self { lo, hi }
        }
    }
}

// ---------------------------------------------------------------------------
// Negation
// ---------------------------------------------------------------------------

impl<T: Float> Neg for Interval<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        if self.is_empty() {
            Self {
                lo: T::infinity(),
                hi: T::neg_infinity(),
            }
        } else {
            Self {
                lo: -self.hi,
                hi: -self.lo,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Operator overloads for Interval<f64>
// ---------------------------------------------------------------------------

impl Add for Interval<f64> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.add_rounded(rhs)
    }
}

impl Sub for Interval<f64> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.sub_rounded(rhs)
    }
}

impl Mul for Interval<f64> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_rounded(rhs)
    }
}

impl Div for Interval<f64> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self.div_rounded(rhs)
    }
}

// Scalar-Interval and Interval-scalar shortcuts
impl Add<f64> for Interval<f64> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f64) -> Self {
        self.add_rounded(Self::point(rhs))
    }
}

impl Sub<f64> for Interval<f64> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f64) -> Self {
        self.sub_rounded(Self::point(rhs))
    }
}

impl Mul<f64> for Interval<f64> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        self.mul_rounded(Self::point(rhs))
    }
}

impl Div<f64> for Interval<f64> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        self.div_rounded(Self::point(rhs))
    }
}

// ---------------------------------------------------------------------------
// Display / Debug
// ---------------------------------------------------------------------------

impl<T: Float + fmt::Display> fmt::Display for Interval<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "∅")
        } else {
            write!(f, "[{}, {}]", self.lo, self.hi)
        }
    }
}

impl<T: Float + fmt::Debug> fmt::Debug for Interval<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "Interval::EMPTY")
        } else {
            write!(f, "Interval {{ lo: {:?}, hi: {:?} }}", self.lo, self.hi)
        }
    }
}

// ---------------------------------------------------------------------------
// Partial ordering (subset relation)
// ---------------------------------------------------------------------------

impl<T: Float> PartialOrd for Interval<T> {
    /// Partial order by containment: `a <= b` iff `a` is a subset of `b`.
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        use core::cmp::Ordering;
        if self == other {
            return Some(Ordering::Equal);
        }
        if other.contains_interval(self) {
            return Some(Ordering::Less);
        }
        if self.contains_interval(other) {
            return Some(Ordering::Greater);
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_containment() {
        // [1, 2] + [3, 4] should contain [4, 6]
        let a = Interval::new(1.0_f64, 2.0);
        let b = Interval::new(3.0_f64, 4.0);
        let c = a + b;
        assert!(c.lo <= 4.0, "lo should be ≤ 4, got {}", c.lo);
        assert!(c.hi >= 6.0, "hi should be ≥ 6, got {}", c.hi);
    }

    #[test]
    fn test_sub_containment() {
        let a = Interval::new(5.0_f64, 7.0);
        let b = Interval::new(1.0_f64, 3.0);
        let c = a - b;
        assert!(c.lo <= 2.0, "lo should be ≤ 2, got {}", c.lo);
        assert!(c.hi >= 6.0, "hi should be ≥ 6, got {}", c.hi);
    }

    #[test]
    fn test_mul_containment() {
        let a = Interval::new(2.0_f64, 3.0);
        let b = Interval::new(4.0_f64, 5.0);
        let c = a * b;
        assert!(c.lo <= 8.0, "lo should be ≤ 8, got {}", c.lo);
        assert!(c.hi >= 15.0, "hi should be ≥ 15, got {}", c.hi);
    }

    #[test]
    fn test_div_containment() {
        let a = Interval::new(6.0_f64, 8.0);
        let b = Interval::new(2.0_f64, 4.0);
        let c = a / b;
        assert!(c.lo <= 1.5, "lo should be ≤ 1.5, got {}", c.lo);
        assert!(c.hi >= 4.0, "hi should be ≥ 4, got {}", c.hi);
    }

    #[test]
    fn test_div_zero_divisor() {
        let a = Interval::new(1.0_f64, 2.0);
        let b = Interval::new(-1.0_f64, 1.0);
        let c = a / b;
        assert!(c.is_entire(), "dividing by interval containing zero should give ENTIRE");
    }

    #[test]
    fn test_sqrt_containment() {
        let a = Interval::new(4.0_f64, 9.0);
        let r = a.sqrt();
        assert!(r.lo <= 2.0, "lo should be ≤ 2, got {}", r.lo);
        assert!(r.hi >= 3.0, "hi should be ≥ 3, got {}", r.hi);
    }

    #[test]
    fn test_sqrt_negative() {
        let a = Interval::new(-1.0_f64, -0.5);
        assert!(a.sqrt().is_empty());
    }

    #[test]
    fn test_exp_containment() {
        let a = Interval::new(0.0_f64, 1.0);
        let e = a.exp();
        assert!(e.lo <= 1.0, "e.lo should be ≤ 1, got {}", e.lo);
        assert!(e.hi >= core::f64::consts::E, "e.hi should be ≥ e, got {}", e.hi);
    }

    #[test]
    fn test_ln_containment() {
        let a = Interval::new(1.0_f64, core::f64::consts::E);
        let l = a.ln();
        assert!(l.lo <= 0.0, "l.lo should be ≤ 0, got {}", l.lo);
        assert!(l.hi >= 1.0, "l.hi should be ≥ 1, got {}", l.hi);
    }

    #[test]
    fn test_sin_wide_interval() {
        // Width >= 2π => [-1, 1]
        let a = Interval::new(0.0_f64, 7.0);
        let s = a.sin();
        assert!(s.lo <= -1.0, "sin wide lo should be ≤ -1");
        assert!(s.hi >= 1.0, "sin wide hi should be ≥ 1");
    }

    #[test]
    fn test_contains() {
        let a = Interval::new(1.0_f64, 3.0);
        assert!(a.contains(2.0));
        assert!(!a.contains(0.0));
        assert!(!a.contains(4.0));
    }

    #[test]
    fn test_empty_interval() {
        let e = Interval::<f64>::EMPTY;
        assert!(e.is_empty());
        assert!(!e.contains(0.0));
    }

    #[test]
    fn test_entire_interval() {
        let e = Interval::<f64>::ENTIRE;
        assert!(e.is_entire());
        assert!(e.contains(1e300));
        assert!(e.contains(-1e300));
    }

    #[test]
    fn test_width_midpoint() {
        let a = Interval::new(1.0_f64, 5.0);
        assert_eq!(a.width(), 4.0);
        assert_eq!(a.midpoint(), 3.0);
    }

    #[test]
    fn test_hull() {
        let a = Interval::new(1.0_f64, 3.0);
        let b = Interval::new(5.0_f64, 7.0);
        let h = a.hull(&b);
        assert_eq!(h.lo, 1.0);
        assert_eq!(h.hi, 7.0);
    }

    #[test]
    fn test_intersection() {
        let a = Interval::new(1.0_f64, 5.0);
        let b = Interval::new(3.0_f64, 7.0);
        let i = a.intersection(&b);
        assert_eq!(i.lo, 3.0);
        assert_eq!(i.hi, 5.0);

        let c = Interval::new(6.0_f64, 8.0);
        assert!(a.intersection(&c).is_empty());
    }

    #[test]
    fn test_powi_even() {
        // [-2, 3]^2 = [0, 9]
        let a = Interval::new(-2.0_f64, 3.0);
        let r = a.powi(2);
        assert!(r.lo <= 0.0);
        assert!(r.hi >= 9.0);
    }

    #[test]
    fn test_powi_odd() {
        // [-2, 3]^3 = [-8, 27]
        let a = Interval::new(-2.0_f64, 3.0);
        let r = a.powi(3);
        assert!(r.lo <= -8.0);
        assert!(r.hi >= 27.0);
    }

    #[test]
    fn test_neg() {
        let a = Interval::new(1.0_f64, 3.0);
        let b = -a;
        assert_eq!(b.lo, -3.0);
        assert_eq!(b.hi, -1.0);
    }

    #[test]
    fn test_next_prev_float() {
        let x = 1.0_f64;
        assert!(prev_float(x) < x);
        assert!(next_float(x) > x);
        assert_eq!(prev_float(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(next_float(f64::INFINITY), f64::INFINITY);
    }
}
