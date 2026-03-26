//! Elementary functions (sqrt, exp, log, pow, trig) in double-double precision.
//!
//! Each function uses argument reduction to keep intermediate values small,
//! then applies a convergent series or Newton iteration carried out entirely
//! in DD arithmetic.

use super::{
    DoubleDouble, DD_E, DD_HALF, DD_LN2, DD_ONE, DD_PI, DD_PI_OVER_2, DD_PI_OVER_4, DD_TWO,
    DD_TWO_PI, DD_ZERO,
};

// ───────────────────────────────────────────────────────────────────────────
// Square root
// ───────────────────────────────────────────────────────────────────────────

/// Double-double square root using Newton iteration.
///
/// Starts from the hardware `sqrt` and refines with one Newton step in DD.
pub fn dd_sqrt(a: DoubleDouble) -> DoubleDouble {
    if a.is_zero() {
        return DD_ZERO;
    }
    if a.is_negative() {
        return DoubleDouble::new(f64::NAN, 0.0);
    }
    // Initial approximation from f64 sqrt
    let x0 = a.hi.sqrt();
    // Newton: x_{n+1} = (x_n + a/x_n) / 2
    let dd_x = DoubleDouble::from(x0);
    let step = (dd_x + a / dd_x) * 0.5;
    // Second iteration for full precision
    (step + a / step) * 0.5
}

impl DoubleDouble {
    /// Square root of `self`.
    #[inline]
    pub fn sqrt(self) -> Self {
        dd_sqrt(self)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Exponential
// ───────────────────────────────────────────────────────────────────────────

/// Double-double exponential function.
///
/// Uses argument reduction `x = k * ln2 + r` with `|r| < ln2/2`, then
/// evaluates `exp(r)` via a truncated Taylor series with enough terms for
/// ~31 digit accuracy, and reconstructs `exp(x) = 2^k * exp(r)`.
pub fn dd_exp(a: DoubleDouble) -> DoubleDouble {
    if a.is_zero() {
        return DD_ONE;
    }
    if a.hi > 709.0 {
        return DoubleDouble::new(f64::INFINITY, 0.0);
    }
    if a.hi < -709.0 {
        return DD_ZERO;
    }

    // Argument reduction: a = k * ln2 + r
    let k_f = (a / DD_LN2).hi.round();
    let k = k_f as i32;
    let r = a - DD_LN2 * k_f;

    // Taylor series: exp(r) = 1 + r + r^2/2! + r^3/3! + ...
    // We need about 25 terms for ~31 digits when |r| < ln2/2 ≈ 0.347
    let mut term = DD_ONE;
    let mut sum = DD_ONE;
    for i in 1..=25 {
        term = term * r / DoubleDouble::from(i as f64);
        sum = sum + term;
        if term.abs().hi < 1e-35 {
            break;
        }
    }

    // Multiply by 2^k using ldexp
    if k == 0 {
        return sum;
    }
    let scale = (2.0_f64).powi(k);
    sum * scale
}

/// Double-double `exp(x) - 1` for small `x`, avoiding cancellation.
pub fn dd_expm1(a: DoubleDouble) -> DoubleDouble {
    if a.abs().hi < 1e-5 {
        // Taylor: expm1(x) = x + x^2/2 + x^3/6 + ...
        let mut term = a;
        let mut sum = a;
        for i in 2..=25 {
            term = term * a / DoubleDouble::from(i as f64);
            sum = sum + term;
            if term.abs().hi < 1e-35 {
                break;
            }
        }
        sum
    } else {
        dd_exp(a) - DD_ONE
    }
}

impl DoubleDouble {
    /// Exponential function `e^self`.
    #[inline]
    pub fn exp(self) -> Self {
        dd_exp(self)
    }

    /// `exp(self) - 1` with improved accuracy for small values.
    #[inline]
    pub fn expm1(self) -> Self {
        dd_expm1(self)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Logarithm
// ───────────────────────────────────────────────────────────────────────────

/// Double-double natural logarithm using Newton iteration on `exp(y) = x`.
///
/// Starting from the hardware `ln`, performs two Newton steps:
/// `y_{n+1} = y_n + 2 * (x - exp(y_n)) / (x + exp(y_n))`
/// (Halley-like iteration for faster convergence).
pub fn dd_log(a: DoubleDouble) -> DoubleDouble {
    if a.is_zero() {
        return DoubleDouble::new(f64::NEG_INFINITY, 0.0);
    }
    if a.is_negative() {
        return DoubleDouble::new(f64::NAN, 0.0);
    }

    // Initial approximation
    let y0 = a.hi.ln();
    let mut y = DoubleDouble::from(y0);

    // Newton iteration: y = y + (a - exp(y)) / exp(y)
    //   equivalently:   y = y + (a * exp(-y) - 1)
    // Using the Halley-like form for better convergence:
    //   y = y + 2*(a - e^y) / (a + e^y)
    for _ in 0..3 {
        let ey = dd_exp(y);
        let diff = a - ey;
        let sum_ae = a + ey;
        y = y + DD_TWO * diff / sum_ae;
    }

    y
}

/// Double-double `log(1 + x)` for small `x`, avoiding cancellation.
pub fn dd_log1p(a: DoubleDouble) -> DoubleDouble {
    if a.abs().hi < 1e-5 {
        // Taylor: log(1+x) = x - x^2/2 + x^3/3 - ...
        let mut term = a;
        let mut sum = a;
        for i in 2..=30 {
            term = term * a * (-DD_ONE);
            sum = sum + term / DoubleDouble::from(i as f64);
            if term.abs().hi < 1e-35 {
                break;
            }
        }
        sum
    } else {
        dd_log(DD_ONE + a)
    }
}

impl DoubleDouble {
    /// Natural logarithm `ln(self)`.
    #[inline]
    pub fn ln(self) -> Self {
        dd_log(self)
    }

    /// `ln(1 + self)` with improved accuracy for small values.
    #[inline]
    pub fn ln1p(self) -> Self {
        dd_log1p(self)
    }

    /// Base-10 logarithm.
    #[inline]
    pub fn log10(self) -> Self {
        dd_log(self) / super::DD_LN10
    }

    /// Base-2 logarithm.
    #[inline]
    pub fn log2(self) -> Self {
        dd_log(self) / DD_LN2
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Power
// ───────────────────────────────────────────────────────────────────────────

/// Double-double power `x^y = exp(y * ln(x))`.
pub fn dd_pow(x: DoubleDouble, y: DoubleDouble) -> DoubleDouble {
    if x.is_zero() {
        if y.is_positive() {
            return DD_ZERO;
        }
        return DoubleDouble::new(f64::INFINITY, 0.0);
    }
    if y.is_zero() {
        return DD_ONE;
    }

    // Integer power optimisation
    let yi = y.hi.round();
    if (y.hi - yi).abs() < 1e-15 && y.lo.abs() < 1e-30 && yi.abs() < 100.0 {
        return dd_powi(x, yi as i64);
    }

    dd_exp(y * dd_log(x))
}

/// Integer power by repeated squaring.
fn dd_powi(mut base: DoubleDouble, mut n: i64) -> DoubleDouble {
    if n == 0 {
        return DD_ONE;
    }
    let mut invert = false;
    if n < 0 {
        n = -n;
        invert = true;
    }
    let mut result = DD_ONE;
    while n > 0 {
        if n & 1 == 1 {
            result = result * base;
        }
        base = base.sqr();
        n >>= 1;
    }
    if invert {
        DD_ONE / result
    } else {
        result
    }
}

impl DoubleDouble {
    /// `self` raised to the power `y`.
    #[inline]
    pub fn pow(self, y: Self) -> Self {
        dd_pow(self, y)
    }

    /// `self` raised to an integer power (by repeated squaring).
    #[inline]
    pub fn powi(self, n: i64) -> Self {
        dd_powi(self, n)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Trigonometric functions
// ───────────────────────────────────────────────────────────────────────────

/// Reduce angle to `[-pi, pi)` returning `(reduced, quadrant)`.
fn reduce_angle(a: DoubleDouble) -> (DoubleDouble, i32) {
    // Number of half-pi intervals
    let k = (a / DD_PI_OVER_2).hi.round() as i64;
    let reduced = a - DD_PI_OVER_2 * DoubleDouble::from(k as f64);
    // Quadrant mod 4 (can be negative)
    let q = ((k % 4) + 4) % 4;
    (reduced, q as i32)
}

/// Taylor series for sin(x) around 0, assuming |x| is small.
fn sin_taylor(x: DoubleDouble) -> DoubleDouble {
    let x2 = x.sqr();
    let mut term = x;
    let mut sum = x;
    for i in 1..=20 {
        let n = 2 * i;
        term = term * x2 * (-DD_ONE) / DoubleDouble::from((n * (n + 1)) as f64);
        sum = sum + term;
        if term.abs().hi < 1e-35 {
            break;
        }
    }
    sum
}

/// Taylor series for cos(x) around 0, assuming |x| is small.
fn cos_taylor(x: DoubleDouble) -> DoubleDouble {
    let x2 = x.sqr();
    let mut term = DD_ONE;
    let mut sum = DD_ONE;
    for i in 1..=20 {
        let n = 2 * i;
        term = term * x2 * (-DD_ONE) / DoubleDouble::from((n * (n - 1)) as f64);
        sum = sum + term;
        if term.abs().hi < 1e-35 {
            break;
        }
    }
    sum
}

/// Double-double sine.
pub fn dd_sin(a: DoubleDouble) -> DoubleDouble {
    if a.is_zero() {
        return DD_ZERO;
    }
    let (r, q) = reduce_angle(a);
    match q {
        0 => sin_taylor(r),
        1 => cos_taylor(r),
        2 => -sin_taylor(r),
        3 => -cos_taylor(r),
        _ => sin_taylor(r), // unreachable in practice
    }
}

/// Double-double cosine.
pub fn dd_cos(a: DoubleDouble) -> DoubleDouble {
    if a.is_zero() {
        return DD_ONE;
    }
    let (r, q) = reduce_angle(a);
    match q {
        0 => cos_taylor(r),
        1 => -sin_taylor(r),
        2 => -cos_taylor(r),
        3 => sin_taylor(r),
        _ => cos_taylor(r),
    }
}

/// Double-double tangent `sin(x) / cos(x)`.
pub fn dd_tan(a: DoubleDouble) -> DoubleDouble {
    dd_sin(a) / dd_cos(a)
}

impl DoubleDouble {
    /// Sine.
    #[inline]
    pub fn sin(self) -> Self {
        dd_sin(self)
    }
    /// Cosine.
    #[inline]
    pub fn cos(self) -> Self {
        dd_cos(self)
    }
    /// Tangent.
    #[inline]
    pub fn tan(self) -> Self {
        dd_tan(self)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Inverse trigonometric functions
// ───────────────────────────────────────────────────────────────────────────

/// Double-double arctangent via Taylor series with argument reduction.
///
/// For large `|x|`, uses `atan(x) = pi/2 - atan(1/x)`.
/// For medium `|x|`, uses identity `atan(x) = atan(c) + atan((x-c)/(1+xc))`
/// with `c` chosen to keep the argument small.
pub fn dd_atan(a: DoubleDouble) -> DoubleDouble {
    if a.is_zero() {
        return DD_ZERO;
    }

    let abs_a = a.abs();

    // For very large |x|: atan(x) = sign(x) * (pi/2 - atan(1/|x|))
    if abs_a.hi > 1e10 {
        let result = DD_PI_OVER_2 - dd_atan_small(abs_a.recip());
        return if a.is_negative() { -result } else { result };
    }

    // For |x| > 1: atan(x) = sign(x) * (pi/2 - atan(1/|x|))
    if abs_a.hi > 1.0 {
        let result = DD_PI_OVER_2 - dd_atan_small(DD_ONE / abs_a);
        return if a.is_negative() { -result } else { result };
    }

    // For |x| > 0.5: use atan(x) = atan(0.5) + atan((x - 0.5)/(1 + 0.5*x))
    if abs_a.hi > 0.5 {
        let half = DD_HALF;
        // atan(0.5) ≈ 0.46364760900080611621...
        let atan_half = DoubleDouble::new(0.4636476090008061, 2.269877745296453e-17);
        let reduced = (a - half) / (DD_ONE + half * a);
        return atan_half + dd_atan_small(reduced);
    }

    dd_atan_small(a)
}

/// Taylor series for atan(x) with |x| <= 0.5.
fn dd_atan_small(x: DoubleDouble) -> DoubleDouble {
    if x.abs().hi < 1e-15 {
        return x;
    }
    let x2 = x.sqr();
    let mut term = x;
    let mut sum = x;
    for i in 1..=30 {
        let n = 2 * i + 1;
        term = term * x2 * (-DD_ONE);
        sum = sum + term / DoubleDouble::from(n as f64);
        if term.abs().hi < 1e-35 {
            break;
        }
    }
    sum
}

/// Double-double `atan2(y, x)` — full quadrant arctangent.
pub fn dd_atan2(y: DoubleDouble, x: DoubleDouble) -> DoubleDouble {
    if x.is_zero() {
        if y.is_zero() {
            return DD_ZERO;
        }
        return if y.is_positive() {
            DD_PI_OVER_2
        } else {
            -DD_PI_OVER_2
        };
    }

    let a = dd_atan(y / x);

    if x.is_positive() {
        a
    } else if y.is_negative() {
        a - DD_PI
    } else {
        a + DD_PI
    }
}

impl DoubleDouble {
    /// Arctangent.
    #[inline]
    pub fn atan(self) -> Self {
        dd_atan(self)
    }

    /// Two-argument arctangent.
    #[inline]
    pub fn atan2(self, x: Self) -> Self {
        dd_atan2(self, x)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::double_double::{DD_E, DD_LN2, DD_PI, DD_SQRT2};

    #[test]
    fn test_sqrt2_squared() {
        // sqrt(2)^2 should equal 2 to ~30 digits
        let s = DD_SQRT2.sqr();
        let diff = (s - DD_TWO).abs();
        assert!(diff.to_f64() < 1e-30, "sqrt(2)^2 - 2 = {:e}", diff.to_f64());
    }

    #[test]
    fn test_sqrt_of_4() {
        let s = dd_sqrt(DoubleDouble::from(4.0));
        let diff = (s - DD_TWO).abs();
        assert!(diff.to_f64() < 1e-30, "sqrt(4) - 2 = {:e}", diff.to_f64());
    }

    #[test]
    fn test_sqrt_negative() {
        let s = dd_sqrt(DoubleDouble::from(-1.0));
        assert!(s.is_nan());
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let x = DoubleDouble::from(1.5);
        let y = dd_exp(dd_log(x));
        let diff = (y - x).abs();
        assert!(
            diff.to_f64() < 1e-28,
            "exp(log(1.5)) - 1.5 = {:e}",
            diff.to_f64()
        );
    }

    #[test]
    fn test_log_exp_roundtrip() {
        let x = DoubleDouble::from(2.5);
        let y = dd_log(dd_exp(x));
        let diff = (y - x).abs();
        assert!(
            diff.to_f64() < 1e-28,
            "log(exp(2.5)) - 2.5 = {:e}",
            diff.to_f64()
        );
    }

    #[test]
    fn test_exp_zero() {
        let r = dd_exp(DD_ZERO);
        let diff = (r - DD_ONE).abs();
        assert!(diff.to_f64() < 1e-31);
    }

    #[test]
    fn test_exp_one() {
        let r = dd_exp(DD_ONE);
        let diff = (r - DD_E).abs();
        assert!(diff.to_f64() < 1e-28, "exp(1) - e = {:e}", diff.to_f64());
    }

    #[test]
    fn test_log_e() {
        let r = dd_log(DD_E);
        let diff = (r - DD_ONE).abs();
        assert!(diff.to_f64() < 1e-28, "log(e) - 1 = {:e}", diff.to_f64());
    }

    #[test]
    fn test_log_2() {
        let r = dd_log(DD_TWO);
        let diff = (r - DD_LN2).abs();
        assert!(diff.to_f64() < 1e-28, "log(2) - ln2 = {:e}", diff.to_f64());
    }

    #[test]
    fn test_sin_cos_pythagorean() {
        // sin^2(x) + cos^2(x) = 1
        for &x_f in &[0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0] {
            let x = DoubleDouble::from(x_f);
            let s = dd_sin(x);
            let c = dd_cos(x);
            let sum = s.sqr() + c.sqr();
            let diff = (sum - DD_ONE).abs();
            assert!(
                diff.to_f64() < 1e-28,
                "sin^2({x_f}) + cos^2({x_f}) - 1 = {:e}",
                diff.to_f64()
            );
        }
    }

    #[test]
    fn test_sin_zero() {
        let s = dd_sin(DD_ZERO);
        assert!(s.abs().to_f64() < 1e-31);
    }

    #[test]
    fn test_sin_pi() {
        let s = dd_sin(DD_PI);
        assert!(s.abs().to_f64() < 1e-28, "sin(pi) = {:e}", s.to_f64());
    }

    #[test]
    fn test_cos_zero() {
        let c = dd_cos(DD_ZERO);
        let diff = (c - DD_ONE).abs();
        assert!(diff.to_f64() < 1e-31);
    }

    #[test]
    fn test_cos_pi() {
        let c = dd_cos(DD_PI);
        let diff = (c + DD_ONE).abs(); // cos(pi) = -1
        assert!(diff.to_f64() < 1e-28, "cos(pi) + 1 = {:e}", diff.to_f64());
    }

    #[test]
    fn test_tan() {
        // tan(pi/4) = 1
        let t = dd_tan(DD_PI_OVER_4);
        let diff = (t - DD_ONE).abs();
        assert!(diff.to_f64() < 1e-27, "tan(pi/4) - 1 = {:e}", diff.to_f64());
    }

    #[test]
    fn test_atan_one() {
        // atan(1) = pi/4
        let a = dd_atan(DD_ONE);
        let diff = (a - DD_PI_OVER_4).abs();
        assert!(
            diff.to_f64() < 1e-27,
            "atan(1) - pi/4 = {:e}",
            diff.to_f64()
        );
    }

    #[test]
    fn test_atan2_quadrants() {
        // atan2(1, 1) = pi/4
        let a = dd_atan2(DD_ONE, DD_ONE);
        let diff = (a - DD_PI_OVER_4).abs();
        assert!(diff.to_f64() < 1e-27);

        // atan2(0, 1) = 0
        let b = dd_atan2(DD_ZERO, DD_ONE);
        assert!(b.abs().to_f64() < 1e-30);

        // atan2(1, 0) = pi/2
        let c = dd_atan2(DD_ONE, DD_ZERO);
        let diff2 = (c - DD_PI_OVER_2).abs();
        assert!(diff2.to_f64() < 1e-30);
    }

    #[test]
    fn test_pow_integer() {
        let x = DoubleDouble::from(2.0);
        let y = DoubleDouble::from(10.0);
        let r = dd_pow(x, y);
        let diff = (r - DoubleDouble::from(1024.0)).abs();
        assert!(diff.to_f64() < 1e-25, "2^10 - 1024 = {:e}", diff.to_f64());
    }

    #[test]
    fn test_pow_fractional() {
        // 4^0.5 = 2
        let x = DoubleDouble::from(4.0);
        let y = DD_HALF;
        let r = dd_pow(x, y);
        let diff = (r - DD_TWO).abs();
        assert!(diff.to_f64() < 1e-27, "4^0.5 - 2 = {:e}", diff.to_f64());
    }
}
