//! SciPy compatibility functions
//!
//! This module provides convenience aliases and additional functions
//! for full SciPy parity. It includes:
//!
//! - `exp1(x)`: Alias for the exponential integral E_1(x)
//! - `expn(n, x)`: Alias for the generalized exponential integral E_n(x)
//! - `loggamma_sign(x)`: Returns (log|Gamma(x)|, sign(Gamma(x)))
//! - `asindg(x)`: Inverse sine in degrees
//! - `acosdg(x)`: Inverse cosine in degrees
//! - `atandg(x)`: Inverse tangent in degrees
//! - `multinomial(n, ks)`: Multinomial coefficient
//! - `bernoulli_poly(n, x)`: Bernoulli polynomial B_n(x)
//! - `euler_poly(n, x)`: Euler polynomial E_n(x)

use crate::error::{SpecialError, SpecialResult};
use std::f64::consts::PI;

// ============================================================================
// Exponential integral aliases
// ============================================================================

/// Exponential integral E_1(x).
///
/// E_1(x) = integral_1^inf e^{-t}/t dt = integral_x^inf e^{-t}/t dt
///
/// For small x, uses the series: E_1(x) = -gamma - ln(x) - sum_{k=1}^inf (-x)^k / (k * k!)
/// For large x, uses the continued fraction expansion.
///
/// # Arguments
///
/// * `x` - Positive real argument
///
/// # Returns
///
/// * `SpecialResult<f64>` - Value of E_1(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::exp1;
///
/// let val = exp1(1.0).expect("exp1 failed");
/// // E_1(1) ~ 0.21938...
/// assert!((val - 0.21938).abs() < 1e-4);
/// ```
pub fn exp1(x: f64) -> SpecialResult<f64> {
    if x.is_nan() {
        return Err(SpecialError::DomainError("NaN input to exp1".to_string()));
    }
    if x <= 0.0 {
        return Err(SpecialError::DomainError("exp1 requires x > 0".to_string()));
    }
    if !x.is_finite() {
        return Ok(0.0);
    }

    Ok(exp1_impl(x))
}

/// Internal implementation of E_1(x) for x > 0.
fn exp1_impl(x: f64) -> f64 {
    const EULER_MASCHERONI: f64 = 0.5772156649015329;

    if x <= 1.0 {
        // Series expansion: E_1(x) = -gamma - ln(x) - sum_{k=1}^inf (-x)^k / (k * k!)
        let mut sum = -EULER_MASCHERONI - x.ln();
        let mut term = 1.0;

        for k in 1..=100 {
            let k_f = k as f64;
            term *= -x / k_f;
            let contribution = term / k_f;
            sum -= contribution;

            if contribution.abs() < 1e-16 * sum.abs() {
                break;
            }
        }

        sum
    } else {
        // Continued fraction (Lentz's method):
        // E_1(x) = e^{-x} * CF where CF = 1/(x+1-1/(x+3-2/(x+5-3/(...))))
        // Using the form: E_1(x) = e^{-x} / (x + 1/(1 + 1/(x + 2/(1 + 2/(x + 3/(1 + ...))))))
        //
        // Standard CF: a_i, b_i where
        // b_0 = 0, b_i = x for odd i, b_i = 1 for even i
        // a_0 = 1, a_i = ceil(i/2) for i >= 1
        //
        // Actually, the simplest CF for E_1 is:
        // E_1(x) = e^{-x} * 1/(x + 1/(1 + 1/(x + 2/(1 + 2/(x + ...)))))
        //
        // Let's use a well-known CF:
        // E_1(x) = exp(-x) * cfrac{1}{x+} cfrac{1}{1+} cfrac{1}{x+} cfrac{2}{1+} cfrac{2}{x+} ...
        //
        // Modified Lentz's method:
        // Continued fraction (modified Lentz's method):
        // E_1(x) = exp(-x) * CF(x) where
        // CF(x) = 1/(x + 1/(1 + 1/(x + 2/(1 + 2/(x + 3/(1 + ...))))))
        //
        // Using a_n, b_n:
        // a_{2k-1} = k, b_{2k-1} = 1 (odd steps)
        // a_{2k} = k, b_{2k} = x (even steps)
        // b_0 = x (initial)
        let tiny = 1.0e-30;
        let mut f = 1.0 / x;
        let mut c = x;
        let mut d = 1.0 / x;

        for i in 1..=200 {
            let (a_i, b_i) = if i % 2 == 1 {
                (((i + 1) / 2) as f64, 1.0)
            } else {
                ((i / 2) as f64, x)
            };

            d = b_i + a_i * d;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b_i + a_i / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = c * d;
            f *= delta;

            if (delta - 1.0).abs() < 1e-15 {
                break;
            }
        }

        (-x).exp() * f
    }
}

/// Generalized exponential integral E_n(x).
///
/// E_n(x) = integral_1^inf e^{-xt}/t^n dt
///
/// Uses E_1(x) as base case and recurrence for n > 1:
/// E_{n+1}(x) = (e^{-x} - x * E_n(x)) / n
///
/// # Arguments
///
/// * `n` - Order (non-negative integer)
/// * `x` - Positive real argument
///
/// # Returns
///
/// * `SpecialResult<f64>` - Value of E_n(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::expn;
///
/// let val = expn(1, 1.0).expect("expn failed");
/// // E_1(1) ~ 0.21938...
/// assert!((val - 0.21938).abs() < 1e-4);
/// ```
pub fn expn(n: i32, x: f64) -> SpecialResult<f64> {
    if x.is_nan() {
        return Err(SpecialError::DomainError("NaN input to expn".to_string()));
    }
    if n < 0 {
        return Err(SpecialError::DomainError(
            "expn requires n >= 0".to_string(),
        ));
    }
    if x <= 0.0 && n <= 1 {
        return Err(SpecialError::DomainError(
            "expn requires x > 0 for n <= 1".to_string(),
        ));
    }

    if n == 0 {
        // E_0(x) = e^{-x} / x
        if x <= 0.0 {
            return Err(SpecialError::DomainError(
                "expn(0, x) requires x > 0".to_string(),
            ));
        }
        return Ok((-x).exp() / x);
    }

    if n == 1 {
        return exp1(x);
    }

    // For n > 1, use forward recurrence from E_1:
    // E_{k+1}(x) = (e^{-x} - x * E_k(x)) / k
    let mut en = exp1_impl(x);
    let exp_neg_x = (-x).exp();

    for k in 1..n {
        en = (exp_neg_x - x * en) / k as f64;
    }

    Ok(en)
}

// ============================================================================
// Log-gamma with sign
// ============================================================================

/// Logarithm of the absolute value of the gamma function with sign.
///
/// Returns (log|Gamma(x)|, sign(Gamma(x))) where sign is +1 or -1.
///
/// This is useful when you need the log of the gamma function for negative
/// arguments, where Gamma(x) can be negative.
///
/// # Arguments
///
/// * `x` - Real argument (not a non-positive integer)
///
/// # Returns
///
/// * `SpecialResult<(f64, f64)>` - Tuple (log|Gamma(x)|, sign(Gamma(x)))
///
/// # Examples
///
/// ```
/// use scirs2_special::loggamma_sign;
///
/// // For positive x, sign is always +1
/// let (lg, sign) = loggamma_sign(5.0).expect("failed");
/// assert!((lg - 24.0_f64.ln()).abs() < 1e-10);
/// assert_eq!(sign, 1.0);
///
/// // For x = -0.5, Gamma(-0.5) = -2*sqrt(pi) < 0
/// let (lg, sign) = loggamma_sign(-0.5).expect("failed");
/// assert_eq!(sign, -1.0);
/// ```
pub fn loggamma_sign(x: f64) -> SpecialResult<(f64, f64)> {
    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to loggamma_sign".to_string(),
        ));
    }

    if !x.is_finite() {
        return Err(SpecialError::DomainError(
            "Infinite input to loggamma_sign".to_string(),
        ));
    }

    // Check for non-positive integers (poles)
    if x <= 0.0 && x == x.floor() {
        return Err(SpecialError::DomainError(
            "loggamma_sign undefined at non-positive integers".to_string(),
        ));
    }

    if x > 0.0 {
        // For positive x, Gamma(x) > 0 always
        // Use the Lanczos approximation directly for correctness
        let lg = lanczos_lgamma(x);
        return Ok((lg, 1.0));
    }

    // For negative non-integer x, use the reflection formula:
    // Gamma(x) = pi / (sin(pi*x) * Gamma(1-x))
    // log|Gamma(x)| = log(pi) - log|sin(pi*x)| - log(Gamma(1-x))
    // sign(Gamma(x)) = sign(1/sin(pi*x)) since Gamma(1-x) > 0 for x < 0

    let one_minus_x = 1.0 - x;
    let lg_one_minus_x = lanczos_lgamma(one_minus_x);
    let sin_pi_x = (PI * x).sin();

    if sin_pi_x.abs() < 1e-300 {
        return Err(SpecialError::DomainError(
            "loggamma_sign: near a pole".to_string(),
        ));
    }

    let lg = PI.ln() - sin_pi_x.abs().ln() - lg_one_minus_x;

    // sign(Gamma(x)) = sign(1/sin(pi*x))
    let sign = if sin_pi_x > 0.0 { 1.0 } else { -1.0 };

    Ok((lg, sign))
}

/// Lanczos approximation for log(Gamma(x)) for x > 0.
///
/// Uses the Lanczos approximation with g=7, n=9 coefficients.
/// This is accurate to about 15 digits.
fn lanczos_lgamma(x: f64) -> f64 {
    // Lanczos coefficients for g=7, n=9
    const LANCZOS_G: f64 = 7.0;
    const LANCZOS_COEFFS: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        // Use reflection formula
        let sin_pi_x = (PI * x).sin();
        if sin_pi_x.abs() < 1e-300 {
            return f64::INFINITY;
        }
        return PI.ln() - sin_pi_x.abs().ln() - lanczos_lgamma(1.0 - x);
    }

    let z = x - 1.0;
    let mut ag = LANCZOS_COEFFS[0];
    for i in 1..9 {
        ag += LANCZOS_COEFFS[i] / (z + i as f64);
    }

    let tmp = z + LANCZOS_G + 0.5;
    0.5 * (2.0 * PI).ln() + (z + 0.5) * tmp.ln() - tmp + ag.ln()
}

// ============================================================================
// Inverse trigonometric functions in degrees
// ============================================================================

/// Inverse sine in degrees.
///
/// Returns arcsin(x) converted to degrees.
///
/// # Arguments
///
/// * `x` - Input value in [-1, 1]
///
/// # Returns
///
/// * `SpecialResult<f64>` - arcsin(x) in degrees
///
/// # Examples
///
/// ```
/// use scirs2_special::asindg;
///
/// let val = asindg(0.5).expect("asindg failed");
/// assert!((val - 30.0).abs() < 1e-10);
///
/// let val = asindg(1.0).expect("asindg failed");
/// assert!((val - 90.0).abs() < 1e-10);
/// ```
pub fn asindg(x: f64) -> SpecialResult<f64> {
    if x.is_nan() {
        return Err(SpecialError::DomainError("NaN input to asindg".to_string()));
    }

    if !(-1.0..=1.0).contains(&x) {
        return Err(SpecialError::DomainError(
            "asindg requires x in [-1, 1]".to_string(),
        ));
    }

    Ok(x.asin() * 180.0 / PI)
}

/// Inverse cosine in degrees.
///
/// Returns arccos(x) converted to degrees.
///
/// # Arguments
///
/// * `x` - Input value in [-1, 1]
///
/// # Returns
///
/// * `SpecialResult<f64>` - arccos(x) in degrees
///
/// # Examples
///
/// ```
/// use scirs2_special::acosdg;
///
/// let val = acosdg(0.5).expect("acosdg failed");
/// assert!((val - 60.0).abs() < 1e-10);
///
/// let val = acosdg(0.0).expect("acosdg failed");
/// assert!((val - 90.0).abs() < 1e-10);
/// ```
pub fn acosdg(x: f64) -> SpecialResult<f64> {
    if x.is_nan() {
        return Err(SpecialError::DomainError("NaN input to acosdg".to_string()));
    }

    if !(-1.0..=1.0).contains(&x) {
        return Err(SpecialError::DomainError(
            "acosdg requires x in [-1, 1]".to_string(),
        ));
    }

    Ok(x.acos() * 180.0 / PI)
}

/// Inverse tangent in degrees.
///
/// Returns arctan(x) converted to degrees.
///
/// # Arguments
///
/// * `x` - Real input value
///
/// # Returns
///
/// * `SpecialResult<f64>` - arctan(x) in degrees
///
/// # Examples
///
/// ```
/// use scirs2_special::atandg;
///
/// let val = atandg(1.0).expect("atandg failed");
/// assert!((val - 45.0).abs() < 1e-10);
///
/// let val = atandg(0.0).expect("atandg failed");
/// assert!(val.abs() < 1e-10);
/// ```
pub fn atandg(x: f64) -> SpecialResult<f64> {
    if x.is_nan() {
        return Err(SpecialError::DomainError("NaN input to atandg".to_string()));
    }

    Ok(x.atan() * 180.0 / PI)
}

// ============================================================================
// Multinomial coefficient
// ============================================================================

/// Multinomial coefficient.
///
/// Computes the multinomial coefficient:
/// n! / (k_1! * k_2! * ... * k_m!)
///
/// where n = k_1 + k_2 + ... + k_m.
///
/// # Arguments
///
/// * `n` - Total number of items (must equal sum of ks)
/// * `ks` - Slice of group sizes
///
/// # Returns
///
/// * `SpecialResult<f64>` - The multinomial coefficient
///
/// # Examples
///
/// ```
/// use scirs2_special::multinomial;
///
/// // 6! / (2! * 2! * 2!) = 720 / 8 = 90
/// let val = multinomial(6, &[2, 2, 2]).expect("multinomial failed");
/// assert!((val - 90.0).abs() < 1e-10);
///
/// // 4! / (1! * 1! * 2!) = 24 / 2 = 12
/// let val = multinomial(4, &[1, 1, 2]).expect("multinomial failed");
/// assert!((val - 12.0).abs() < 1e-10);
/// ```
pub fn multinomial(n: u32, ks: &[u32]) -> SpecialResult<f64> {
    // Validate: sum of ks must equal n
    let sum_ks: u32 = ks.iter().sum();
    if sum_ks != n {
        return Err(SpecialError::ValueError(format!(
            "Sum of group sizes ({}) must equal n ({})",
            sum_ks, n
        )));
    }

    // Check for zeros and trivial cases
    if n == 0 {
        return Ok(1.0);
    }

    if ks.len() == 1 {
        return Ok(1.0);
    }

    // For moderate n, compute using iterative multiplication
    // to maintain precision: n! / (k1! * k2! * ... * km!)
    // = C(n, k1) * C(n-k1, k2) * C(n-k1-k2, k3) * ...
    let mut result = 1.0;
    let mut remaining = n;

    for &k in ks {
        if k > remaining {
            return Ok(0.0);
        }

        // Compute C(remaining, k)
        let binom = crate::binomial(remaining, k)?;
        result *= binom;
        remaining -= k;
    }

    Ok(result)
}

// ============================================================================
// Bernoulli polynomials
// ============================================================================

/// Bernoulli polynomial B_n(x).
///
/// The Bernoulli polynomials are defined by the generating function:
/// t*e^{xt} / (e^t - 1) = sum_{n=0}^inf B_n(x) * t^n / n!
///
/// They satisfy: B_n(x) = sum_{k=0}^n C(n,k) * B_k * x^{n-k}
/// where B_k are Bernoulli numbers.
///
/// # Arguments
///
/// * `n` - Order (non-negative integer)
/// * `x` - Real argument
///
/// # Returns
///
/// * `SpecialResult<f64>` - Value of B_n(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::bernoulli_poly;
///
/// // B_0(x) = 1
/// let val = bernoulli_poly(0, 0.5).expect("failed");
/// assert!((val - 1.0).abs() < 1e-10);
///
/// // B_1(x) = x - 1/2
/// let val = bernoulli_poly(1, 0.5).expect("failed");
/// assert!(val.abs() < 1e-10); // B_1(0.5) = 0
///
/// // B_2(x) = x^2 - x + 1/6
/// let val = bernoulli_poly(2, 1.0).expect("failed");
/// assert!((val - 1.0/6.0).abs() < 1e-10); // B_2(1) = 1/6
/// ```
pub fn bernoulli_poly(n: u32, x: f64) -> SpecialResult<f64> {
    if n > 50 {
        return Err(SpecialError::ValueError(
            "bernoulli_poly: n too large (max 50)".to_string(),
        ));
    }

    // B_n(x) = sum_{k=0}^n C(n,k) * B_k * x^{n-k}
    let mut result = 0.0;

    for k in 0..=n {
        let bk = crate::bernoulli_number(k)?;
        let binom = crate::binomial(n, k)?;
        let x_power = x.powi((n - k) as i32);
        result += binom * bk * x_power;
    }

    Ok(result)
}

// ============================================================================
// Euler polynomials
// ============================================================================

/// Euler polynomial E_n(x).
///
/// The Euler polynomials are defined by the generating function:
/// 2*e^{xt} / (e^t + 1) = sum_{n=0}^inf E_n(x) * t^n / n!
///
/// They satisfy: E_n(x) = sum_{k=0}^n C(n,k) * E_k/2^k * (x - 1/2)^{n-k}
/// where E_k are Euler numbers.
///
/// Alternatively: E_n(x) = 2/((n+1)) * [B_{n+1}(x) - 2^{n+1} * B_{n+1}(x/2)]
///
/// For a simpler formula:
/// E_n(x) = sum_{k=0}^n C(n,k) * (E_k / 2^k) * (x - 0.5)^{n-k}
///
/// # Arguments
///
/// * `n` - Order (non-negative integer)
/// * `x` - Real argument
///
/// # Returns
///
/// * `SpecialResult<f64>` - Value of E_n(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::euler_poly;
///
/// // E_0(x) = 1
/// let val = euler_poly(0, 0.5).expect("failed");
/// assert!((val - 1.0).abs() < 1e-10);
///
/// // E_1(x) = x - 1/2
/// let val = euler_poly(1, 1.0).expect("failed");
/// assert!((val - 0.5).abs() < 1e-10);
/// ```
pub fn euler_poly(n: u32, x: f64) -> SpecialResult<f64> {
    if n > 50 {
        return Err(SpecialError::ValueError(
            "euler_poly: n too large (max 50)".to_string(),
        ));
    }

    // Use the relation to Bernoulli polynomials:
    // E_n(x) = (2 / (n+1)) * [B_{n+1}(x) - 2^{n+1} * B_{n+1}(x/2)]
    let n_plus_1 = n + 1;
    let bn1_x = bernoulli_poly(n_plus_1, x)?;
    let bn1_x_half = bernoulli_poly(n_plus_1, x / 2.0)?;
    let two_pow = 2.0_f64.powi(n_plus_1 as i32);

    Ok(2.0 / (n_plus_1 as f64) * (bn1_x - two_pow * bn1_x_half))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ============ exp1/expn tests ============

    #[test]
    fn test_exp1_basic() {
        let val = exp1(1.0).expect("exp1 failed");
        // E_1(1) ~ 0.21938393439552...
        assert_relative_eq!(val, 0.21938393439552, epsilon = 1e-6);
    }

    #[test]
    fn test_expn_basic() {
        let val = expn(1, 1.0).expect("expn failed");
        let val2 = exp1(1.0).expect("exp1 failed");
        assert_relative_eq!(val, val2, epsilon = 1e-12);
    }

    // ============ loggamma_sign tests ============

    #[test]
    fn test_loggamma_sign_positive() {
        // Gamma(5) = 24, sign = +1
        let (lg, sign) = loggamma_sign(5.0).expect("failed");
        assert_relative_eq!(lg, 24.0_f64.ln(), epsilon = 1e-10);
        assert_eq!(sign, 1.0);
    }

    #[test]
    fn test_loggamma_sign_half() {
        // Gamma(0.5) = sqrt(pi), sign = +1
        let (lg, sign) = loggamma_sign(0.5).expect("failed");
        assert_relative_eq!(lg, (PI.sqrt()).ln(), epsilon = 1e-10);
        assert_eq!(sign, 1.0);
    }

    #[test]
    fn test_loggamma_sign_negative() {
        // Gamma(-0.5) = -2*sqrt(pi), sign = -1
        let (lg, sign) = loggamma_sign(-0.5).expect("failed");
        assert_eq!(sign, -1.0);
        let expected_lg = (2.0 * PI.sqrt()).ln();
        assert_relative_eq!(lg, expected_lg, epsilon = 1e-8);
    }

    #[test]
    fn test_loggamma_sign_pole() {
        // At non-positive integers, it should error
        assert!(loggamma_sign(0.0).is_err());
        assert!(loggamma_sign(-1.0).is_err());
        assert!(loggamma_sign(-2.0).is_err());
    }

    // ============ Inverse trig in degrees ============

    #[test]
    fn test_asindg() {
        assert_relative_eq!(asindg(0.0).expect("failed"), 0.0, epsilon = 1e-10);
        assert_relative_eq!(asindg(0.5).expect("failed"), 30.0, epsilon = 1e-10);
        assert_relative_eq!(asindg(1.0).expect("failed"), 90.0, epsilon = 1e-10);
        assert_relative_eq!(asindg(-1.0).expect("failed"), -90.0, epsilon = 1e-10);
    }

    #[test]
    fn test_asindg_domain_error() {
        assert!(asindg(1.5).is_err());
        assert!(asindg(-1.5).is_err());
    }

    #[test]
    fn test_acosdg() {
        assert_relative_eq!(acosdg(1.0).expect("failed"), 0.0, epsilon = 1e-10);
        assert_relative_eq!(acosdg(0.5).expect("failed"), 60.0, epsilon = 1e-10);
        assert_relative_eq!(acosdg(0.0).expect("failed"), 90.0, epsilon = 1e-10);
        assert_relative_eq!(acosdg(-1.0).expect("failed"), 180.0, epsilon = 1e-10);
    }

    #[test]
    fn test_acosdg_domain_error() {
        assert!(acosdg(1.5).is_err());
    }

    #[test]
    fn test_atandg() {
        assert_relative_eq!(atandg(0.0).expect("failed"), 0.0, epsilon = 1e-10);
        assert_relative_eq!(atandg(1.0).expect("failed"), 45.0, epsilon = 1e-10);
        assert_relative_eq!(atandg(-1.0).expect("failed"), -45.0, epsilon = 1e-10);
    }

    #[test]
    fn test_atandg_nan() {
        assert!(atandg(f64::NAN).is_err());
    }

    // ============ Multinomial tests ============

    #[test]
    fn test_multinomial_basic() {
        // 4! / (2! * 2!) = 6
        let val = multinomial(4, &[2, 2]).expect("failed");
        assert_relative_eq!(val, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multinomial_triple() {
        // 6! / (2! * 2! * 2!) = 90
        let val = multinomial(6, &[2, 2, 2]).expect("failed");
        assert_relative_eq!(val, 90.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multinomial_trivial() {
        let val = multinomial(5, &[5]).expect("failed");
        assert_relative_eq!(val, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multinomial_zero() {
        let val = multinomial(0, &[]).expect("failed");
        assert_relative_eq!(val, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multinomial_invalid_sum() {
        assert!(multinomial(5, &[2, 2]).is_err());
    }

    #[test]
    fn test_multinomial_with_ones() {
        // 4! / (1! * 1! * 1! * 1!) = 24
        let val = multinomial(4, &[1, 1, 1, 1]).expect("failed");
        assert_relative_eq!(val, 24.0, epsilon = 1e-10);
    }

    // ============ Bernoulli polynomial tests ============

    #[test]
    fn test_bernoulli_poly_b0() {
        // B_0(x) = 1
        assert_relative_eq!(
            bernoulli_poly(0, 0.0).expect("failed"),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            bernoulli_poly(0, 0.5).expect("failed"),
            1.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            bernoulli_poly(0, 1.0).expect("failed"),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_bernoulli_poly_b1() {
        // B_1(x) = x - 1/2
        assert_relative_eq!(
            bernoulli_poly(1, 0.0).expect("failed"),
            -0.5,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            bernoulli_poly(1, 0.5).expect("failed"),
            0.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            bernoulli_poly(1, 1.0).expect("failed"),
            0.5,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_bernoulli_poly_b2() {
        // B_2(x) = x^2 - x + 1/6
        let b2_0 = bernoulli_poly(2, 0.0).expect("failed");
        assert_relative_eq!(b2_0, 1.0 / 6.0, epsilon = 1e-10);

        let b2_1 = bernoulli_poly(2, 1.0).expect("failed");
        assert_relative_eq!(b2_1, 1.0 / 6.0, epsilon = 1e-10);

        let b2_half = bernoulli_poly(2, 0.5).expect("failed");
        assert_relative_eq!(b2_half, -1.0 / 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bernoulli_poly_at_zero() {
        // B_n(0) = B_n (Bernoulli number)
        for n in 0..=8 {
            let bp = bernoulli_poly(n, 0.0).expect("failed");
            let bn = crate::bernoulli_number(n).expect("failed");
            assert_relative_eq!(bp, bn, epsilon = 1e-8,);
        }
    }

    // ============ Euler polynomial tests ============

    #[test]
    fn test_euler_poly_e0() {
        // E_0(x) = 1
        assert_relative_eq!(euler_poly(0, 0.5).expect("failed"), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_euler_poly_e1() {
        // E_1(x) = x - 1/2
        assert_relative_eq!(euler_poly(1, 0.0).expect("failed"), -0.5, epsilon = 1e-10);
        assert_relative_eq!(euler_poly(1, 0.5).expect("failed"), 0.0, epsilon = 1e-10);
        assert_relative_eq!(euler_poly(1, 1.0).expect("failed"), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_euler_poly_e2() {
        // E_2(x) = x^2 - x
        let e2_0 = euler_poly(2, 0.0).expect("failed");
        assert_relative_eq!(e2_0, 0.0, epsilon = 1e-10);

        let e2_1 = euler_poly(2, 1.0).expect("failed");
        assert_relative_eq!(e2_1, 0.0, epsilon = 1e-10);

        let e2_half = euler_poly(2, 0.5).expect("failed");
        assert_relative_eq!(e2_half, -0.25, epsilon = 1e-10);
    }
}
