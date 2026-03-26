//! Transcendental and elementary functions for quad-double (QD) precision.
//!
//! All functions use range reduction followed by Taylor/Newton iteration
//! to achieve ~62 digits of accuracy.

use crate::error::{CoreError, CoreResult, ErrorContext};
use super::quad_double::{QD, qd_add, qd_sub, qd_mul, qd_div, qd_mul_f64, qd_square, qd_add_f64};

// ─── Error helpers ─────────────────────────────────────────────────────────────

#[inline(always)]
fn comp_err(msg: impl Into<String>) -> CoreError {
    CoreError::ComputationError(ErrorContext::new(msg))
}

// ─── Elementary functions ──────────────────────────────────────────────────────

/// Absolute value of a QD number.
#[inline]
#[must_use]
pub fn qd_abs(a: &QD) -> QD {
    a.abs()
}

/// Negate a QD number.
#[inline]
#[must_use]
pub fn qd_neg(a: &QD) -> QD {
    a.negate()
}

/// Square root via Newton iteration: x_{n+1} = (x_n + a/x_n) / 2.
///
/// Starting from the f64 sqrt, performs enough Newton iterations to
/// converge to full QD (~62 digit) accuracy.
pub fn qd_sqrt(a: &QD) -> CoreResult<QD> {
    if a.is_negative() {
        return Err(comp_err("qd_sqrt: argument must be non-negative"));
    }
    if a.is_zero() {
        return Ok(QD::ZERO);
    }
    if !a.is_finite() {
        return Err(comp_err("qd_sqrt: non-finite input"));
    }

    // Initial approximation from f64 sqrt.
    let x0 = QD::new(a.x0.sqrt());

    // Newton iterations: x_{n+1} = (x_n + a/x_n) / 2
    // Each iteration roughly doubles the number of correct digits.
    // f64 gives ~16 digits, so we need: 16 -> 32 -> 64. Three iterations suffice.
    let half = QD::HALF;

    let ax = qd_div(a, &x0)?;
    let x1 = qd_mul(&qd_add(&x0, &ax), &half);

    let ax = qd_div(a, &x1)?;
    let x2 = qd_mul(&qd_add(&x1, &ax), &half);

    let ax = qd_div(a, &x2)?;
    let x3 = qd_mul(&qd_add(&x2, &ax), &half);

    Ok(x3)
}

/// Exponential function exp(x) in QD precision.
///
/// Uses argument reduction `x = k * ln(2) + r` where `|r| < ln(2)/2`,
/// then evaluates a Taylor series for `exp(r)`, and finally multiplies
/// by `2^k`.
pub fn qd_exp(a: &QD) -> CoreResult<QD> {
    if !a.is_finite() {
        return Err(comp_err("qd_exp: non-finite input"));
    }
    if a.is_zero() {
        return Ok(QD::ONE);
    }

    // If the argument is very large, check for overflow.
    if a.x0 > 709.0 {
        return Err(comp_err("qd_exp: argument too large, would overflow"));
    }
    if a.x0 < -745.0 {
        // Underflows to zero.
        return Ok(QD::ZERO);
    }

    let ln2 = QD::QD_LN2;

    // Argument reduction: k = round(x / ln2)
    let k_f = (a.x0 / ln2.x0).round();
    let k = k_f as i64;

    // r = x - k * ln2
    let k_ln2 = qd_mul_f64(&ln2, k_f);
    let r = qd_sub(a, &k_ln2);

    // Further reduce: divide r by 2^m for faster convergence.
    let m = 10; // r/1024 makes Taylor series converge very fast.
    let scale = 1.0_f64 / (1u64 << m) as f64;
    let r_scaled = qd_mul_f64(&r, scale);

    // Taylor series: exp(r_scaled) = sum_{n=0}^{N} r_scaled^n / n!
    let n_terms = 40usize;
    let mut sum = QD::ONE;
    let mut term = QD::ONE;

    for n in 1..=n_terms {
        term = qd_div(&qd_mul(&term, &r_scaled), &QD::new(n as f64))?;
        let new_sum = qd_add(&sum, &term);
        if term.abs().x0.abs() < sum.abs().x0 * 1e-65 {
            sum = new_sum;
            break;
        }
        sum = new_sum;
    }

    // Square m times to undo the scaling: exp(r) = exp(r_scaled)^(2^m)
    let mut result = sum;
    for _ in 0..m {
        result = qd_square(&result);
    }

    // Multiply by 2^k.
    if k == 0 {
        return Ok(result);
    }

    // Handle 2^k carefully to avoid overflow in the scale factor.
    // Split large k into manageable pieces.
    let mut remaining_k = k;
    while remaining_k != 0 {
        let chunk = remaining_k.clamp(-1022, 1023);
        let scale_bits = ((1023i64 + chunk) as u64) << 52;
        let scale_val = f64::from_bits(scale_bits);
        result = QD {
            x0: result.x0 * scale_val,
            x1: result.x1 * scale_val,
            x2: result.x2 * scale_val,
            x3: result.x3 * scale_val,
        };
        remaining_k -= chunk;
    }

    Ok(result)
}

/// Natural logarithm ln(x) in QD precision.
///
/// Uses argument reduction to [1, 2) and Halley's iteration for refinement.
/// Starting from the f64 ln value, converges cubically.
pub fn qd_ln(a: &QD) -> CoreResult<QD> {
    if a.x0 <= 0.0 {
        return Err(comp_err("qd_ln: argument must be positive"));
    }
    if !a.is_finite() {
        return Err(comp_err("qd_ln: non-finite input"));
    }

    // Special case: ln(1) = 0
    if a.x0 == 1.0 && a.x1 == 0.0 && a.x2 == 0.0 && a.x3 == 0.0 {
        return Ok(QD::ZERO);
    }

    // Start with f64 ln as initial approximation.
    let mut y = QD::new(a.x0.ln());

    // Halley's iteration for ln:
    //   exp_y = exp(y)
    //   y_new = y + 2 * (x - exp_y) / (x + exp_y)
    // Cubic convergence: 16 -> 48 -> 144 digits. Two iterations suffice.
    for _ in 0..3 {
        let exp_y = qd_exp(&y)?;
        let x_minus_exp = qd_sub(a, &exp_y);
        let x_plus_exp = qd_add(a, &exp_y);
        let ratio = qd_div(&x_minus_exp, &x_plus_exp)?;
        let correction = qd_mul_f64(&ratio, 2.0);
        y = qd_add(&y, &correction);
    }

    Ok(y)
}

/// Sine function sin(x) in QD precision.
///
/// Uses range reduction mod pi/4 followed by a Taylor series.
pub fn qd_sin(a: &QD) -> CoreResult<QD> {
    let (s, _c) = qd_sincos(a)?;
    Ok(s)
}

/// Cosine function cos(x) in QD precision.
///
/// Uses range reduction mod pi/4 followed by a Taylor series.
pub fn qd_cos(a: &QD) -> CoreResult<QD> {
    let (_s, c) = qd_sincos(a)?;
    Ok(c)
}

/// Compute sin(x) and cos(x) simultaneously in QD precision.
///
/// Uses argument reduction to [-pi/4, pi/4] and Taylor series evaluation.
pub fn qd_sincos(a: &QD) -> CoreResult<(QD, QD)> {
    if !a.is_finite() {
        return Err(comp_err("qd_sincos: non-finite input"));
    }
    if a.is_zero() {
        return Ok((QD::ZERO, QD::ONE));
    }

    let pi = QD::QD_PI;
    let half_pi = qd_mul_f64(&pi, 0.5);

    // Argument reduction: k = round(x / (pi/2))
    let two_over_pi = qd_div(&QD::TWO, &pi)?;
    let k_f = qd_mul(a, &two_over_pi).x0.round();
    let k = k_f as i64;

    // r = x - k * (pi/2)
    let r = qd_sub(a, &qd_mul_f64(&half_pi, k_f));

    // Taylor series for sin(r) and cos(r) where |r| <= pi/4.
    let r2 = qd_square(&r);
    let n_terms = 30usize;

    // sin(r) = r - r^3/3! + r^5/5! - ...
    let mut sin_val = r;
    let mut term_sin = r;

    // cos(r) = 1 - r^2/2! + r^4/4! - ...
    let mut cos_val = QD::ONE;
    let mut term_cos = QD::ONE;

    for i in 1..=n_terms {
        // sin term: *= -r^2 / (2i * (2i+1))
        term_sin = qd_mul(&term_sin, &r2.negate());
        term_sin = qd_div(&term_sin, &QD::new((2 * i) as f64))?;
        term_sin = qd_div(&term_sin, &QD::new((2 * i + 1) as f64))?;

        // cos term: *= -r^2 / ((2i-1) * 2i)
        term_cos = qd_mul(&term_cos, &r2.negate());
        term_cos = qd_div(&term_cos, &QD::new((2 * i - 1) as f64))?;
        term_cos = qd_div(&term_cos, &QD::new((2 * i) as f64))?;

        let new_sin = qd_add(&sin_val, &term_sin);
        let new_cos = qd_add(&cos_val, &term_cos);

        let converged = term_sin.abs().x0.abs() < sin_val.abs().x0 * 1e-65;
        sin_val = new_sin;
        cos_val = new_cos;

        if converged {
            break;
        }
    }

    // Unreduce using quarter index k mod 4.
    let km4 = ((k % 4) + 4) as usize % 4;
    let (s, c) = match km4 {
        0 => (sin_val, cos_val),
        1 => (cos_val, sin_val.negate()),
        2 => (sin_val.negate(), cos_val.negate()),
        _ => (cos_val.negate(), sin_val),
    };

    Ok((s, c))
}

/// Integer power: a^n via binary exponentiation.
pub fn qd_powi(a: &QD, n: i32) -> CoreResult<QD> {
    if n == 0 {
        return Ok(QD::ONE);
    }
    if n == 1 {
        return Ok(*a);
    }
    if n == -1 {
        return qd_div(&QD::ONE, a);
    }

    let mut result = QD::ONE;
    let mut base = if n < 0 {
        qd_div(&QD::ONE, a)?
    } else {
        *a
    };
    let mut exp = n.unsigned_abs();

    while exp > 0 {
        if exp & 1 == 1 {
            result = qd_mul(&result, &base);
        }
        base = qd_square(&base);
        exp >>= 1;
    }

    Ok(result)
}

/// Convert a QD to a `DD` by discarding the lower two components.
///
/// Provided for convenience; same as `a.to_dd()`.
#[inline]
#[must_use]
pub fn qd_to_dd(a: &QD) -> super::DD {
    a.to_dd()
}

/// Promote a DD to QD.
///
/// Provided for convenience; same as `QD::from_dd(d)`.
#[inline]
#[must_use]
pub fn dd_to_qd(d: &super::DD) -> QD {
    QD::from_dd(*d)
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qd_sqrt_of_4() {
        let four = QD::new(4.0);
        let result = qd_sqrt(&four).expect("sqrt should succeed");
        let diff = qd_sub(&result, &QD::TWO);
        assert!(
            diff.abs().x0.abs() < 1e-60,
            "sqrt(4) - 2 = {:e}",
            diff.x0
        );
    }

    #[test]
    fn test_qd_sqrt_of_2() {
        let two = QD::TWO;
        let s = qd_sqrt(&two).expect("sqrt should succeed");
        // sqrt(2)^2 should be 2
        let sq = qd_square(&s);
        let diff = qd_sub(&sq, &two);
        assert!(
            diff.abs().x0.abs() < 1e-58,
            "sqrt(2)^2 - 2 = {:e}",
            diff.x0
        );
    }

    #[test]
    fn test_qd_sqrt_negative() {
        let neg = QD::new(-1.0);
        assert!(qd_sqrt(&neg).is_err());
    }

    #[test]
    fn test_qd_exp_of_zero() {
        let result = qd_exp(&QD::ZERO).expect("exp should succeed");
        assert_eq!(result.x0, 1.0);
    }

    #[test]
    fn test_qd_exp_of_one() {
        let result = qd_exp(&QD::ONE).expect("exp should succeed");
        let expected = QD::QD_E;
        let diff = qd_sub(&result, &expected);
        // Accuracy limited by constant precision; ~46 digits is excellent.
        assert!(
            diff.abs().x0.abs() < 1e-45,
            "exp(1) - e = {:e}",
            diff.x0
        );
    }

    #[test]
    fn test_qd_ln_of_e() {
        let e_val = QD::QD_E;
        let result = qd_ln(&e_val).expect("ln should succeed");
        let diff = qd_sub(&result, &QD::ONE);
        // ~46 digits accuracy from constant precision.
        assert!(
            diff.abs().x0.abs() < 1e-45,
            "ln(e) - 1 = {:e}",
            diff.x0
        );
    }

    #[test]
    fn test_qd_ln_exp_roundtrip() {
        let x = QD::new(2.5);
        let exp_x = qd_exp(&x).expect("exp should succeed");
        let ln_exp_x = qd_ln(&exp_x).expect("ln should succeed");
        let diff = qd_sub(&ln_exp_x, &x);
        assert!(
            diff.abs().x0.abs() < 1e-45,
            "ln(exp(2.5)) - 2.5 = {:e}",
            diff.x0
        );
    }

    #[test]
    fn test_qd_sin_cos_identity() {
        // sin^2(x) + cos^2(x) = 1
        let x = QD::new(1.23456789);
        let (s, c) = qd_sincos(&x).expect("sincos should succeed");
        let s2 = qd_square(&s);
        let c2 = qd_square(&c);
        let sum = qd_add(&s2, &c2);
        let diff = qd_sub(&sum, &QD::ONE);
        // ~50 digits accuracy.
        assert!(
            diff.abs().x0.abs() < 1e-48,
            "sin^2 + cos^2 - 1 = {:e}",
            diff.x0
        );
    }

    #[test]
    fn test_qd_sin_of_pi() {
        let pi = QD::QD_PI;
        let s = qd_sin(&pi).expect("sin should succeed");
        assert!(
            s.abs().x0.abs() < 1e-50,
            "sin(pi) = {:e}, expected ~0",
            s.x0
        );
    }

    #[test]
    fn test_qd_cos_of_zero() {
        let c = qd_cos(&QD::ZERO).expect("cos should succeed");
        let diff = qd_sub(&c, &QD::ONE);
        assert!(
            diff.abs().x0.abs() < 1e-60,
            "cos(0) - 1 = {:e}",
            diff.x0
        );
    }

    #[test]
    fn test_qd_powi_basic() {
        let two = QD::TWO;
        let result = qd_powi(&two, 10).expect("powi should succeed");
        let expected = QD::new(1024.0);
        let diff = qd_sub(&result, &expected);
        assert!(
            diff.abs().x0.abs() < 1e-50,
            "2^10 - 1024 = {:e}",
            diff.x0
        );
    }

    #[test]
    fn test_qd_powi_negative() {
        let two = QD::TWO;
        let result = qd_powi(&two, -2).expect("powi should succeed");
        let expected = QD::new(0.25);
        let diff = qd_sub(&result, &expected);
        assert!(
            diff.abs().x0.abs() < 1e-60,
            "2^(-2) - 0.25 = {:e}",
            diff.x0
        );
    }

    #[test]
    fn test_qd_powi_zero() {
        let x = QD::new(42.0);
        let result = qd_powi(&x, 0).expect("powi should succeed");
        assert_eq!(result.x0, 1.0);
    }

    #[test]
    fn test_dd_to_qd_promotion() {
        let d = super::super::DD::from_parts(3.14, 1e-17);
        let q = dd_to_qd(&d);
        assert_eq!(q.x0, d.hi);
        assert_eq!(q.x1, d.lo);
        assert_eq!(q.x2, 0.0);
        assert_eq!(q.x3, 0.0);
    }

    #[test]
    fn test_qd_abs_and_neg() {
        let x = QD::new(-3.14);
        let a = qd_abs(&x);
        assert!(a.is_positive());
        assert_eq!(a.x0, 3.14);

        let n = qd_neg(&a);
        assert!(n.is_negative());
        assert_eq!(n.x0, -3.14);
    }
}
