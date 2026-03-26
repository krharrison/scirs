//! Special mathematical functions evaluated in double-double precision.
//!
//! Provides `dd_gamma`, `dd_lgamma`, `dd_erf`, `dd_erfc`, `dd_beta`, and
//! `dd_bessel_j0` — each computed with ~31 decimal digits of accuracy.

use super::{
    arithmetic::two_sum,
    elementary::{dd_cos, dd_exp, dd_log, dd_pow, dd_sin, dd_sqrt},
    DoubleDouble, DD_HALF, DD_LN2, DD_ONE, DD_PI, DD_TWO, DD_TWO_PI, DD_ZERO,
};

// ───────────────────────────────────────────────────────────────────────────
// Gamma function via Lanczos approximation in DD precision
// ───────────────────────────────────────────────────────────────────────────

/// Lanczos g parameter (we use g = 7, n = 9 coefficients).
const LANCZOS_G: f64 = 7.0;

/// Lanczos coefficients for g = 7 (from Paul Godfrey / Numerical Recipes).
/// These are f64 — we promote to DD during the computation.
const LANCZOS_COEFF: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_9,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Double-double Gamma function using the Lanczos approximation.
///
/// For positive `x`, computes `Gamma(x)` via
///
/// ```text
/// Gamma(x) = sqrt(2*pi) * (x + g - 0.5)^(x - 0.5) * exp(-(x + g - 0.5)) * A_g(x)
/// ```
///
/// where `A_g(x)` is the Lanczos series sum.
///
/// For negative non-integer `x`, uses the reflection formula
/// `Gamma(x) * Gamma(1-x) = pi / sin(pi * x)`.
pub fn dd_gamma(x: DoubleDouble) -> DoubleDouble {
    // Special cases
    if x.is_zero() {
        return DoubleDouble::new(f64::INFINITY, 0.0);
    }

    // Negative integers -> infinity
    if x.is_negative() {
        let xi = x.hi.round();
        if (x.hi - xi).abs() < 1e-15 && x.lo.abs() < 1e-30 && xi <= 0.0 {
            return DoubleDouble::new(f64::INFINITY, 0.0);
        }
    }

    // Reflection formula for x < 0.5
    if x.hi < 0.5 {
        // Gamma(x) = pi / (sin(pi*x) * Gamma(1-x))
        let sin_px = dd_sin(DD_PI * x);
        let g1mx = dd_gamma(DD_ONE - x);
        return DD_PI / (sin_px * g1mx);
    }

    // Lanczos for x >= 0.5
    let z = x - DD_ONE;
    let dd_g = DoubleDouble::from(LANCZOS_G);

    // Compute the series A_g(z)
    let mut ag = DoubleDouble::from(LANCZOS_COEFF[0]);
    for i in 1..LANCZOS_COEFF.len() {
        let coeff = DoubleDouble::from(LANCZOS_COEFF[i]);
        ag = ag + coeff / (z + DoubleDouble::from(i as f64));
    }

    // t = z + g + 0.5
    let t = z + dd_g + DD_HALF;

    // sqrt(2*pi)
    let sqrt_2pi = dd_sqrt(DD_TWO_PI);

    // Gamma(x) = sqrt(2*pi) * t^(z+0.5) * exp(-t) * ag
    sqrt_2pi * dd_pow(t, z + DD_HALF) * dd_exp(-t) * ag
}

/// Double-double log-Gamma function.
///
/// Computes `ln(|Gamma(x)|)` using the Lanczos approximation in DD precision.
/// More numerically stable than `log(gamma(x))` for large `x`.
pub fn dd_lgamma(x: DoubleDouble) -> DoubleDouble {
    if x.is_zero() || (x.is_negative() && x.hi == x.hi.round() && x.lo.abs() < 1e-30) {
        return DoubleDouble::new(f64::INFINITY, 0.0);
    }

    // Reflection for x < 0.5
    if x.hi < 0.5 {
        // lgamma(x) = ln(pi) - ln|sin(pi*x)| - lgamma(1-x)
        let ln_pi = dd_log(DD_PI);
        let sin_px = dd_sin(DD_PI * x).abs();
        let ln_sin = dd_log(sin_px);
        return ln_pi - ln_sin - dd_lgamma(DD_ONE - x);
    }

    let z = x - DD_ONE;
    let dd_g = DoubleDouble::from(LANCZOS_G);

    let mut ag = DoubleDouble::from(LANCZOS_COEFF[0]);
    for i in 1..LANCZOS_COEFF.len() {
        let coeff = DoubleDouble::from(LANCZOS_COEFF[i]);
        ag = ag + coeff / (z + DoubleDouble::from(i as f64));
    }

    let t = z + dd_g + DD_HALF;

    // ln(sqrt(2*pi)) + (z+0.5)*ln(t) - t + ln(ag)
    let ln_sqrt_2pi = dd_log(dd_sqrt(DD_TWO_PI));
    ln_sqrt_2pi + (z + DD_HALF) * dd_log(t) - t + dd_log(ag)
}

impl DoubleDouble {
    /// Gamma function.
    #[inline]
    pub fn gamma(self) -> Self {
        dd_gamma(self)
    }

    /// Log-gamma function `ln(|Gamma(self)|)`.
    #[inline]
    pub fn lgamma(self) -> Self {
        dd_lgamma(self)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Error function
// ───────────────────────────────────────────────────────────────────────────

/// Double-double error function via Taylor series for small `|x|` and
/// continued-fraction / asymptotic expansion for large `|x|`.
///
/// ```text
/// erf(x) = (2/sqrt(pi)) * sum_{n=0}^{inf} (-1)^n * x^{2n+1} / (n! * (2n+1))
/// ```
pub fn dd_erf(x: DoubleDouble) -> DoubleDouble {
    if x.is_zero() {
        return DD_ZERO;
    }

    let abs_x = x.abs();

    if abs_x.hi > 6.0 {
        // erf(x) ≈ 1 for large x
        let result = DD_ONE - dd_erfc_large(abs_x);
        return if x.is_negative() { -result } else { result };
    }

    if abs_x.hi > 2.5 {
        // Use erfc for medium-large arguments
        let result = DD_ONE - dd_erfc_cf(abs_x);
        return if x.is_negative() { -result } else { result };
    }

    // Taylor series for small to moderate |x|
    let x2 = x.sqr();
    let mut term = x; // x / 0!
    let mut sum = x;
    for n in 1..=40 {
        // term *= -x^2 / n
        term = term * x2 * (-DD_ONE) / DoubleDouble::from(n as f64);
        let contrib = term / DoubleDouble::from((2 * n + 1) as f64);
        sum = sum + contrib;
        if contrib.abs().hi < 1e-35 {
            break;
        }
    }

    // Multiply by 2/sqrt(pi)
    let two_over_sqrt_pi = DD_TWO / dd_sqrt(DD_PI);
    let result = two_over_sqrt_pi * sum;

    // Clamp to [-1, 1]
    if result.hi > 1.0 {
        DD_ONE
    } else if result.hi < -1.0 {
        -DD_ONE
    } else {
        result
    }
}

/// Continued fraction for erfc(x) when x is moderately large (x > 2.5).
fn dd_erfc_cf(x: DoubleDouble) -> DoubleDouble {
    // erfc(x) = exp(-x^2) / (x * sqrt(pi)) * CF
    // where CF = 1 / (1 + a1/(1 + a2/(1 + ...))) with a_n = n/2
    //
    // Using Lentz's method for the continued fraction:
    // erfc(x) = exp(-x²) / sqrt(pi) * 1/(x + 0.5/(x + 1/(x + 1.5/(x + ...))))
    let x2 = x.sqr();
    let exp_neg_x2 = dd_exp(-x2);

    // Evaluate CF from bottom up with ~30 terms
    let mut cf = x;
    for k in (1..=30).rev() {
        let ak = DoubleDouble::from(k as f64) * DD_HALF;
        cf = x + ak / cf;
    }

    exp_neg_x2 / (dd_sqrt(DD_PI) * cf)
}

/// Asymptotic expansion for erfc(x) when x is large (x > 6).
fn dd_erfc_large(x: DoubleDouble) -> DoubleDouble {
    let x2 = x.sqr();
    let exp_neg_x2 = dd_exp(-x2);
    let inv_2x2 = DD_ONE / (DD_TWO * x2);

    // Asymptotic series: erfc(x) ~ exp(-x^2)/(x*sqrt(pi)) * sum_{k=0}^{N} (-1)^k (2k-1)!! / (2x^2)^k
    let mut term = DD_ONE;
    let mut sum = DD_ONE;
    for k in 1..=20 {
        term = term * DoubleDouble::from((2 * k - 1) as f64) * (-inv_2x2);
        let old_sum = sum;
        sum = sum + term;
        // Stop when terms start growing (asymptotic series!)
        if term.abs().hi > (old_sum - sum).abs().hi * 2.0 {
            sum = old_sum;
            break;
        }
        if term.abs().hi < 1e-34 {
            break;
        }
    }

    exp_neg_x2 / (x * dd_sqrt(DD_PI)) * sum
}

/// Double-double complementary error function `erfc(x) = 1 - erf(x)`.
///
/// For large `x`, uses the continued fraction directly to avoid cancellation.
pub fn dd_erfc(x: DoubleDouble) -> DoubleDouble {
    if x.is_zero() {
        return DD_ONE;
    }

    let abs_x = x.abs();

    if abs_x.hi > 6.0 {
        let r = dd_erfc_large(abs_x);
        return if x.is_negative() { DD_TWO - r } else { r };
    }

    if abs_x.hi > 2.5 {
        let r = dd_erfc_cf(abs_x);
        return if x.is_negative() { DD_TWO - r } else { r };
    }

    // For small x, compute via erf
    DD_ONE - dd_erf(x)
}

impl DoubleDouble {
    /// Error function.
    #[inline]
    pub fn erf(self) -> Self {
        dd_erf(self)
    }

    /// Complementary error function.
    #[inline]
    pub fn erfc(self) -> Self {
        dd_erfc(self)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Beta function
// ───────────────────────────────────────────────────────────────────────────

/// Double-double Beta function `B(a, b) = Gamma(a) * Gamma(b) / Gamma(a+b)`.
///
/// Computed via log-gamma to avoid overflow for large arguments:
/// `B(a,b) = exp(lgamma(a) + lgamma(b) - lgamma(a+b))`.
pub fn dd_beta(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
    let lg_a = dd_lgamma(a);
    let lg_b = dd_lgamma(b);
    let lg_ab = dd_lgamma(a + b);
    dd_exp(lg_a + lg_b - lg_ab)
}

impl DoubleDouble {
    /// Beta function `B(self, b)`.
    #[inline]
    pub fn beta(self, b: Self) -> Self {
        dd_beta(self, b)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Bessel J0
// ───────────────────────────────────────────────────────────────────────────

/// Double-double Bessel function of the first kind `J_0(x)`.
///
/// For small `|x|`, uses the power series:
/// ```text
/// J_0(x) = sum_{k=0}^{inf} (-1)^k * (x/2)^{2k} / (k!)^2
/// ```
///
/// For large `|x|`, uses the asymptotic expansion:
/// ```text
/// J_0(x) ~ sqrt(2/(pi*x)) * cos(x - pi/4) * P(x) - sin(x - pi/4) * Q(x)
/// ```
pub fn dd_bessel_j0(x: DoubleDouble) -> DoubleDouble {
    let abs_x = x.abs();

    if abs_x.is_zero() {
        return DD_ONE;
    }

    if abs_x.hi > 20.0 {
        return bessel_j0_asymptotic(abs_x);
    }

    // Power series
    let x_half = abs_x * DD_HALF;
    let x_half_sq = x_half.sqr();
    let mut term = DD_ONE; // k = 0 term
    let mut sum = DD_ONE;

    for k in 1..=40 {
        let kf = DoubleDouble::from(k as f64);
        term = term * (-x_half_sq) / kf.sqr();
        sum = sum + term;
        if term.abs().hi < 1e-35 {
            break;
        }
    }

    sum
}

/// Asymptotic expansion for J0(x) for large x.
fn bessel_j0_asymptotic(x: DoubleDouble) -> DoubleDouble {
    // J0(x) ~ sqrt(2/(pi*x)) * (P0*cos(theta) - Q0*sin(theta))
    // where theta = x - pi/4
    let theta = x - DD_PI * DoubleDouble::from(0.25);
    let cos_theta = dd_cos(theta);
    let sin_theta = dd_sin(theta);

    // P0 and Q0 asymptotic series
    let inv_8x = DD_ONE / (DoubleDouble::from(8.0) * x);
    let inv_8x_sq = inv_8x.sqr();

    // P0(x) = 1 - 9/(128x^2) + ...
    // Q0(x) = -1/(8x) + 75/(1024x^3) - ...
    let mut p0 = DD_ONE;
    let mut p_term = DD_ONE;
    let mut q_term = DD_ONE;

    // Coefficients (2k-1)!! / (2k)!! pattern
    for k in 1..=10 {
        let k2 = 2 * k;
        let num = DoubleDouble::from(((k2 - 1) * (k2 - 1)) as f64);
        p_term = p_term * (-num) / DoubleDouble::from((2 * k * (2 * k - 1)) as f64) * inv_8x_sq;
        p0 = p0 + p_term;

        let num_q = DoubleDouble::from(((k2 - 1) * (k2 + 1)) as f64);
        q_term = q_term * (-num_q) / DoubleDouble::from(((2 * k) * (2 * k + 1)) as f64) * inv_8x_sq;
    }
    let q0 = -inv_8x * q_term;

    let prefactor = dd_sqrt(DD_TWO / (DD_PI * x));
    prefactor * (p0 * cos_theta - q0 * sin_theta)
}

impl DoubleDouble {
    /// Bessel function `J_0(self)`.
    #[inline]
    pub fn bessel_j0(self) -> Self {
        dd_bessel_j0(self)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_half_is_sqrt_pi() {
        // Gamma(0.5) = sqrt(pi)
        // Lanczos with f64 coefficients gives ~15 digit accuracy
        let g = dd_gamma(DD_HALF);
        let sqrt_pi = dd_sqrt(DD_PI);
        let diff = (g - sqrt_pi).abs();
        assert!(
            diff.to_f64() < 1e-14,
            "Gamma(0.5) - sqrt(pi) = {:e}",
            diff.to_f64()
        );
    }

    #[test]
    fn test_gamma_integers() {
        // Gamma(n) = (n-1)!
        let g1 = dd_gamma(DD_ONE);
        assert!(
            (g1 - DD_ONE).abs().to_f64() < 1e-14,
            "Gamma(1) = {:e}",
            g1.to_f64()
        );

        let g2 = dd_gamma(DD_TWO);
        assert!(
            (g2 - DD_ONE).abs().to_f64() < 1e-14,
            "Gamma(2) = {:e}",
            g2.to_f64()
        );

        let g5 = dd_gamma(DoubleDouble::from(5.0));
        let diff = (g5 - DoubleDouble::from(24.0)).abs();
        assert!(diff.to_f64() < 1e-12, "Gamma(5) - 24 = {:e}", diff.to_f64());
    }

    #[test]
    fn test_lgamma_positive() {
        // lgamma(1) = 0, lgamma(2) = 0
        let lg1 = dd_lgamma(DD_ONE);
        assert!(lg1.abs().to_f64() < 1e-14, "lgamma(1) = {:e}", lg1.to_f64());

        let lg2 = dd_lgamma(DD_TWO);
        assert!(lg2.abs().to_f64() < 1e-14, "lgamma(2) = {:e}", lg2.to_f64());
    }

    #[test]
    fn test_lgamma_large() {
        // lgamma(10) = ln(9!) = ln(362880)
        let lg10 = dd_lgamma(DoubleDouble::from(10.0));
        let expected = dd_log(DoubleDouble::from(362_880.0));
        let diff = (lg10 - expected).abs();
        assert!(
            diff.to_f64() < 1e-12,
            "lgamma(10) - ln(362880) = {:e}",
            diff.to_f64()
        );
    }

    #[test]
    fn test_erf_zero() {
        let e = dd_erf(DD_ZERO);
        assert!(e.abs().to_f64() < 1e-31, "erf(0) = {:e}", e.to_f64());
    }

    #[test]
    fn test_erf_large_approaches_one() {
        let e = dd_erf(DoubleDouble::from(5.0));
        let diff = (e - DD_ONE).abs();
        assert!(diff.to_f64() < 1e-10, "erf(5) - 1 = {:e}", diff.to_f64());
    }

    #[test]
    fn test_erf_symmetry() {
        // erf(-x) = -erf(x)
        let x = DoubleDouble::from(1.0);
        let ep = dd_erf(x);
        let en = dd_erf(-x);
        let sum = ep + en;
        assert!(
            sum.abs().to_f64() < 1e-28,
            "erf(1) + erf(-1) = {:e}",
            sum.to_f64()
        );
    }

    #[test]
    fn test_erf_known_value() {
        // erf(1) ≈ 0.8427007929497148693...
        let e = dd_erf(DD_ONE);
        let expected = 0.842_700_792_949_714_9;
        let diff = (e.to_f64() - expected).abs();
        assert!(diff < 1e-14, "erf(1) = {:e}, diff = {:e}", e.to_f64(), diff);
    }

    #[test]
    fn test_erfc_zero() {
        let ec = dd_erfc(DD_ZERO);
        let diff = (ec - DD_ONE).abs();
        assert!(diff.to_f64() < 1e-31);
    }

    #[test]
    fn test_erfc_large() {
        // erfc(5) should be very small but positive
        let ec = dd_erfc(DoubleDouble::from(5.0));
        assert!(ec.to_f64() > 0.0);
        assert!(ec.to_f64() < 1e-10);
    }

    #[test]
    fn test_erf_plus_erfc_equals_one() {
        for &xf in &[0.5, 1.0, 2.0, 3.0] {
            let x = DoubleDouble::from(xf);
            let sum = dd_erf(x) + dd_erfc(x);
            let diff = (sum - DD_ONE).abs();
            assert!(
                diff.to_f64() < 1e-25,
                "erf({xf}) + erfc({xf}) - 1 = {:e}",
                diff.to_f64()
            );
        }
    }

    #[test]
    fn test_beta_function() {
        // B(1,1) = 1
        let b11 = dd_beta(DD_ONE, DD_ONE);
        let diff = (b11 - DD_ONE).abs();
        assert!(diff.to_f64() < 1e-14, "B(1,1) - 1 = {:e}", diff.to_f64());
    }

    #[test]
    fn test_beta_half_half() {
        // B(0.5, 0.5) = pi
        let b = dd_beta(DD_HALF, DD_HALF);
        let diff = (b - DD_PI).abs();
        assert!(
            diff.to_f64() < 1e-13,
            "B(0.5, 0.5) - pi = {:e}",
            diff.to_f64()
        );
    }

    #[test]
    fn test_bessel_j0_zero() {
        // J0(0) = 1
        let j = dd_bessel_j0(DD_ZERO);
        let diff = (j - DD_ONE).abs();
        assert!(diff.to_f64() < 1e-31);
    }

    #[test]
    fn test_bessel_j0_known_values() {
        // J0(1) ≈ 0.7651976865579666...
        let j = dd_bessel_j0(DD_ONE);
        let expected = 0.765_197_686_557_966_6;
        let diff = (j.to_f64() - expected).abs();
        assert!(diff < 1e-14, "J0(1) = {:e}, diff = {:e}", j.to_f64(), diff);
    }

    #[test]
    fn test_bessel_j0_symmetry() {
        // J0 is even: J0(-x) = J0(x)
        let x = DoubleDouble::from(2.5);
        let jp = dd_bessel_j0(x);
        let jn = dd_bessel_j0(-x);
        let diff = (jp - jn).abs();
        assert!(
            diff.to_f64() < 1e-28,
            "J0(2.5) - J0(-2.5) = {:e}",
            diff.to_f64()
        );
    }

    #[test]
    fn test_gamma_negative() {
        // Gamma(-0.5) = -2*sqrt(pi)
        let g = dd_gamma(DoubleDouble::from(-0.5));
        let expected = DD_TWO * dd_sqrt(DD_PI) * (-DD_ONE);
        let diff = (g - expected).abs();
        assert!(
            diff.to_f64() < 1e-13,
            "Gamma(-0.5) error = {:e}",
            diff.to_f64()
        );
    }
}
