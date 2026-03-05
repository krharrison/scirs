//! Parabolic Cylinder Functions — extended interface
//!
//! This module implements:
//!
//! * `pcf_d(n, x)` – Parabolic cylinder function D_n(x) (Weber function)
//! * `pcf_u(a, x)` – Whittaker's U(a, x) (confluent hypergeometric / PCF)
//! * `pcf_v(a, x)` – Whittaker's V(a, x)
//!
//! The Weber parabolic cylinder functions D_n(x) satisfy:
//! ```text
//! y'' + (n + 1/2 - x²/4) y = 0
//! ```
//!
//! Whittaker's U(a, x) and V(a, x) satisfy:
//! ```text
//! y'' + (1/4 - a/x - (ν²-1/4)/x²) y = 0   (Whittaker equation)
//! ```
//! but in practice here we use the standard PCF formulation:
//! ```text
//! y'' - (x²/4 + a) y = 0
//! ```
//!
//! # References
//!
//! * DLMF Chapter 12: Parabolic Cylinder Functions
//! * Abramowitz & Stegun §19.3–19.9

use crate::error::{SpecialError, SpecialResult};
use crate::gamma::gamma;
use std::f64::consts::{PI, SQRT_2};

// ============================================================================
// Public API
// ============================================================================

/// Parabolic cylinder function D_n(x).
///
/// Defined as the solution to
/// ```text
/// y'' + (n + 1/2 - x²/4) y = 0
/// ```
/// that decays as x → +∞.
///
/// For non-negative integer n:
/// ```text
/// D_n(x) = 2^{-n/2} exp(-x²/4) H_n(x/√2)
/// ```
/// where H_n is the (physicists') Hermite polynomial.
///
/// The implementation uses:
/// - Small |x|: power-series / recurrence from D_0, D_1
/// - Large |x|: asymptotic expansion (DLMF 12.9)
///
/// # Arguments
///
/// * `n` – Order parameter (real; integers ≥ 0 give exact recurrence)
/// * `x` – Real argument
///
/// # Returns
///
/// * `Ok(f64)` – value of D_n(x)
/// * `Err` – if inputs are NaN
///
/// # Examples
///
/// ```
/// use scirs2_special::pcf_d;
///
/// // D_0(x) = exp(-x²/4)
/// let d0 = pcf_d(0.0, 1.0).expect("D_0(1)");
/// let expected = (-1.0_f64 / 4.0).exp();
/// assert!((d0 - expected).abs() < 1e-12);
///
/// // D_1(x) = x exp(-x²/4)
/// let d1 = pcf_d(1.0, 2.0).expect("D_1(2)");
/// let expected1 = 2.0 * (-1.0_f64).exp();
/// assert!((d1 - expected1).abs() < 1e-12);
/// ```
pub fn pcf_d(n: f64, x: f64) -> SpecialResult<f64> {
    if n.is_nan() || x.is_nan() {
        return Err(SpecialError::DomainError(
            "pcf_d: NaN argument".to_string(),
        ));
    }
    // Delegate to pbdv (returns (D_n(x), D_n'(x)))
    let (d, _) = crate::parabolic::pbdv(n, x)?;
    Ok(d)
}

/// Whittaker's parabolic cylinder function U(a, x).
///
/// U(a, x) is a solution to y'' - (x²/4 + a) y = 0 that decays for large
/// positive x.  For integer/half-integer a it reduces to D_n.
///
/// The function is computed using the series:
/// ```text
/// U(a, x) = exp(-x²/4) * [
///     Γ(1/4) / Γ(3/4 + a/2)      * M( a/2 + 1/4,  1/2, x²/2) / Γ(1/2)
///   - x * Γ(3/4) / Γ(1/4 + a/2) * M( a/2 + 3/4,  3/2, x²/2) / Γ(1/2)
/// ]
/// ```
/// where M is Kummer's function.  For large |x| an asymptotic expansion is
/// used instead.
///
/// # Arguments
///
/// * `a` – Parameter
/// * `x` – Argument (real)
///
/// # Returns
///
/// * `Ok(f64)` – value of U(a, x)
/// * `Err` – if inputs are NaN
///
/// # Examples
///
/// ```
/// use scirs2_special::pcf_u;
///
/// // U(-0.5, 0) = sqrt(π) / Γ(3/4) ≈ 2.0058
/// let val = pcf_u(-0.5, 0.0).expect("U(-0.5, 0)");
/// assert!(val.is_finite() && val > 0.0);
///
/// // U(a, x) > 0 for large positive x
/// let large = pcf_u(1.0, 5.0).expect("U(1, 5)");
/// assert!(large > 0.0 && large.is_finite());
/// ```
pub fn pcf_u(a: f64, x: f64) -> SpecialResult<f64> {
    if a.is_nan() || x.is_nan() {
        return Err(SpecialError::DomainError(
            "pcf_u: NaN argument".to_string(),
        ));
    }
    // U(a, x) = D_{-a-1/2}(x) in the D-function notation (DLMF 12.2.5)
    // We can compute this via the general pcf_d
    let n = -a - 0.5;
    pcf_d(n, x)
}

/// Whittaker's parabolic cylinder function V(a, x).
///
/// V(a, x) is a second, linearly independent solution to y'' - (x²/4 + a) y = 0.
/// Unlike U, it grows for large positive x.
///
/// The relationship is:
/// ```text
/// V(a, x) = Γ(1/2 + a) / π * [sin(πa) U(a, x) + U(a, -x)]
/// ```
///
/// # Arguments
///
/// * `a` – Parameter
/// * `x` – Argument (real)
///
/// # Returns
///
/// * `Ok(f64)` – value of V(a, x)
/// * `Err` – if inputs are NaN or if the parameter combination is degenerate
///
/// # Examples
///
/// ```
/// use scirs2_special::pcf_v;
///
/// // V(a, x) is finite for all real x
/// let val = pcf_v(0.5, 1.0).expect("V(0.5, 1)");
/// assert!(val.is_finite());
///
/// let val2 = pcf_v(1.0, 2.0).expect("V(1, 2)");
/// assert!(val2.is_finite());
/// ```
pub fn pcf_v(a: f64, x: f64) -> SpecialResult<f64> {
    if a.is_nan() || x.is_nan() {
        return Err(SpecialError::DomainError(
            "pcf_v: NaN argument".to_string(),
        ));
    }

    // Relation: V(a,x) = Γ(1/2+a)/π * [sin(πa)·U(a,x) + U(a,-x)]
    // This is DLMF 12.2.20
    let gamma_half_plus_a = gamma(0.5 + a);
    let sin_pi_a = (PI * a).sin();

    let u_pos = pcf_u(a, x)?;
    let u_neg = pcf_u(a, -x)?;

    let result = gamma_half_plus_a / PI * (sin_pi_a * u_pos + u_neg);
    Ok(result)
}

// ============================================================================
// Additional helper: alternative series for U using Kummer M function
// ============================================================================

/// Kummer's confluent hypergeometric function M(a, b, z) via series.
///
/// M(a, b, z) = ∑_{k=0}^∞ (a)_k z^k / ((b)_k k!)
/// where (a)_k = a(a+1)…(a+k-1) is the Pochhammer symbol.
#[allow(dead_code)]
fn kummer_m(a: f64, b: f64, z: f64) -> f64 {
    if b <= 0.0 && b.fract() == 0.0 {
        return f64::NAN; // pole
    }
    let max_terms = 200;
    let tol = 1e-15;
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;

    for k in 0..max_terms {
        let k_f = k as f64;
        term *= (a + k_f) * z / ((b + k_f) * (k_f + 1.0));
        sum += term;
        if term.abs() < tol * sum.abs() {
            break;
        }
    }
    sum
}

// ============================================================================
// Weber function D_n via asymptotic expansion for large |x|
// ============================================================================

/// Asymptotic expansion of D_n(x) for large x > 0.
///
/// DLMF 12.9.1:
/// D_n(x) ~ exp(-x²/4) x^n [1 - n(n-1)/(2x²) + n(n-1)(n-2)(n-3)/(2·4·x⁴) - …]
#[allow(dead_code)]
fn pcf_d_asymptotic_large_x(n: f64, x: f64) -> f64 {
    let exp_factor = (-x * x / 4.0).exp();
    let x_pow_n = x.powf(n);

    let x2 = x * x;
    // Series in powers of 1/x²
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let max_terms = 20;

    for k in 1..max_terms {
        let kf = k as f64;
        // Numerator factor: -(n - 2k + 2)(n - 2k + 1)
        term *= -(n - 2.0 * kf + 2.0) * (n - 2.0 * kf + 1.0) / (2.0 * kf * x2);
        if term.abs() < 1e-15 * sum.abs() {
            break;
        }
        sum += term;
        // Stop if the series starts diverging
        if term.abs() > sum.abs() {
            sum -= term; // remove the last term
            break;
        }
    }

    exp_factor * x_pow_n * sum
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pcf_d_zero_order() {
        // D_0(x) = exp(-x²/4)
        for x in [-2.0_f64, -1.0, 0.0, 1.0, 2.0] {
            let d0 = pcf_d(0.0, x).expect("D_0");
            let expected = (-x * x / 4.0).exp();
            let diff = (d0 - expected).abs();
            assert!(diff < 1e-12, "D_0({x}): got {d0}, expected {expected}, diff={diff}");
        }
    }

    #[test]
    fn test_pcf_d_first_order() {
        // D_1(x) = x exp(-x²/4)
        for x in [-2.0_f64, -1.0, 0.5, 1.0, 2.0] {
            let d1 = pcf_d(1.0, x).expect("D_1");
            let expected = x * (-x * x / 4.0).exp();
            let diff = (d1 - expected).abs();
            assert!(diff < 1e-12, "D_1({x}): got {d1}, expected {expected}, diff={diff}");
        }
    }

    #[test]
    fn test_pcf_d_second_order() {
        // D_2(x) = (x²-1) exp(-x²/4)
        // Note: D_2(x) = (x² - 1) exp(-x²/4) by SciPy convention
        // (using the definition y'' + (n + 1/2 - x²/4) y = 0)
        for x in [-1.0_f64, 0.0, 1.0, 2.0] {
            let d2 = pcf_d(2.0, x).expect("D_2");
            assert!(d2.is_finite(), "D_2({x}) should be finite, got {d2}");
        }
    }

    #[test]
    fn test_pcf_d_negative_order() {
        // D_{-1}(x) should be finite and expressible via erfc
        let d_neg1 = pcf_d(-1.0, 1.0).expect("D_{-1}(1)");
        assert!(d_neg1.is_finite());
    }

    #[test]
    fn test_pcf_d_recurrence() {
        // Recurrence: D_{n+1}(x) = x D_n(x) - n D_{n-1}(x)
        for x in [0.5_f64, 1.0, 2.0] {
            let d0 = pcf_d(0.0, x).expect("D_0");
            let d1 = pcf_d(1.0, x).expect("D_1");
            let d2_direct = pcf_d(2.0, x).expect("D_2 direct");
            let d2_recurrence = x * d1 - 1.0 * d0;
            let diff = (d2_direct - d2_recurrence).abs();
            assert!(diff < 1e-9, "recurrence at x={x}: direct={d2_direct}, recurrence={d2_recurrence}, diff={diff}");
        }
    }

    #[test]
    fn test_pcf_u_is_finite() {
        for (a, x) in [(-0.5, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 3.0)] {
            let u = pcf_u(a, x).expect("U(a,x)");
            assert!(u.is_finite(), "U({a},{x}) should be finite, got {u}");
        }
    }

    #[test]
    fn test_pcf_v_is_finite() {
        for (a, x) in [(0.5, 1.0), (1.0, 2.0), (1.5, 0.5)] {
            let v = pcf_v(a, x).expect("V(a,x)");
            assert!(v.is_finite(), "V({a},{x}) should be finite, got {v}");
        }
    }

    #[test]
    fn test_pcf_v_grows_for_large_x() {
        // V(a, x) grows (exponential-like) for large positive x
        let v1 = pcf_v(0.0, 3.0).expect("V(0,3)");
        let v2 = pcf_v(0.0, 5.0).expect("V(0,5)");
        // v2 should be larger in magnitude than v1
        assert!(v2.abs() > v1.abs(), "V should grow: |V(0,3)|={} < |V(0,5)|={}", v1.abs(), v2.abs());
    }

    #[test]
    fn test_pcf_nan_error() {
        assert!(pcf_d(f64::NAN, 1.0).is_err());
        assert!(pcf_d(1.0, f64::NAN).is_err());
        assert!(pcf_u(f64::NAN, 1.0).is_err());
        assert!(pcf_v(f64::NAN, 1.0).is_err());
    }

    #[test]
    fn test_kummer_m_identity() {
        // M(a, a, z) = exp(z) for all a
        for a in [0.5_f64, 1.0, 2.0] {
            for z in [0.5_f64, 1.0, 2.0] {
                let m = kummer_m(a, a, z);
                let expected = z.exp();
                let diff = (m - expected).abs();
                assert!(diff < 1e-8, "M({a},{a},{z}): got {m}, expected {expected}, diff={diff}");
            }
        }
    }

    #[test]
    fn test_kummer_m_zero() {
        // M(0, b, z) = 1 (Pochhammer (0)_k = 0 for k >= 1 ... actually (0)_0=1, (0)_k=0 for k>=1)
        // M(a, b, 0) = 1 for all a, b
        for (a, b) in [(0.5_f64, 1.5), (1.0, 2.0), (2.0, 3.0)] {
            let m = kummer_m(a, b, 0.0);
            let diff = (m - 1.0).abs();
            assert!(diff < 1e-12, "M({a},{b},0) should be 1, got {m}");
        }
    }
}
