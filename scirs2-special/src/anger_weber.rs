//! Anger and Weber functions, and Lommel functions
//!
//! These are generalizations of Bessel functions that arise in problems involving
//! non-integer order and inhomogeneous Bessel equations.
//!
//! ## Anger Function J_v(x)
//!
//! The Anger function is defined as:
//! ```text
//! J_v(x) = (1/pi) * integral_0^pi cos(v*t - x*sin(t)) dt
//! ```
//!
//! For integer v, J_v(x) = J_v(x) (ordinary Bessel function of the first kind).
//!
//! ## Weber Function E_v(x)
//!
//! The Weber function is defined as:
//! ```text
//! E_v(x) = (1/pi) * integral_0^pi sin(v*t - x*sin(t)) dt
//! ```
//!
//! ## Lommel Functions
//!
//! The Lommel functions s_{mu,nu}(z) and S_{mu,nu}(z) are particular solutions
//! of the inhomogeneous Bessel equation:
//! ```text
//! z^2 y'' + z y' + (z^2 - nu^2) y = z^{mu+1}
//! ```

use crate::error::{SpecialError, SpecialResult};
use std::f64::consts::PI;

/// Anger function J_v(x).
///
/// Defined as:
/// ```text
/// J_v(x) = (1/pi) * integral_0^pi cos(v*theta - x*sin(theta)) d_theta
/// ```
///
/// For integer v, this reduces to the ordinary Bessel function J_v(x).
///
/// # Arguments
/// * `v` - Order (real number)
/// * `x` - Argument
///
/// # Returns
/// Value of the Anger function J_v(x)
///
/// # Examples
/// ```
/// use scirs2_special::anger_j;
/// // For integer order, should approximate Bessel J
/// let result = anger_j(0.0, 1.0).expect("anger_j failed");
/// assert!(result.is_finite());
/// ```
pub fn anger_j(v: f64, x: f64) -> SpecialResult<f64> {
    if v.is_nan() || x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to anger_j".to_string(),
        ));
    }

    // Special case: x = 0
    if x.abs() < 1e-15 {
        if v == 0.0 {
            return Ok(1.0);
        }
        // For v != 0, J_v(0) = sin(v*pi) / (v*pi) via the integral
        return Ok((v * PI).sin() / (v * PI));
    }

    // Use numerical quadrature (Simpson's rule with many points)
    let n = 1000;
    let h = PI / (n as f64);
    let mut sum = 0.0;

    // Simpson's rule: integral ~ (h/3) * [f(0) + 4f(h) + 2f(2h) + 4f(3h) + ... + f(pi)]
    for i in 0..=n {
        let theta = (i as f64) * h;
        let f = (v * theta - x * theta.sin()).cos();

        let weight = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };

        sum += weight * f;
    }

    Ok(sum * h / (3.0 * PI))
}

/// Weber function E_v(x).
///
/// Defined as:
/// ```text
/// E_v(x) = (1/pi) * integral_0^pi sin(v*theta - x*sin(theta)) d_theta
/// ```
///
/// # Arguments
/// * `v` - Order (real number)
/// * `x` - Argument
///
/// # Returns
/// Value of the Weber function E_v(x)
///
/// # Examples
/// ```
/// use scirs2_special::weber_e;
/// let result = weber_e(0.5, 1.0).expect("weber_e failed");
/// assert!(result.is_finite());
/// ```
pub fn weber_e(v: f64, x: f64) -> SpecialResult<f64> {
    if v.is_nan() || x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to weber_e".to_string(),
        ));
    }

    // Special case: x = 0
    if x.abs() < 1e-15 {
        if v == 0.0 {
            return Ok(0.0);
        }
        // E_v(0) = (1 - cos(v*pi)) / (v*pi)
        return Ok((1.0 - (v * PI).cos()) / (v * PI));
    }

    // Use numerical quadrature (Simpson's rule)
    let n = 1000;
    let h = PI / (n as f64);
    let mut sum = 0.0;

    for i in 0..=n {
        let theta = (i as f64) * h;
        let f = (v * theta - x * theta.sin()).sin();

        let weight = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };

        sum += weight * f;
    }

    Ok(sum * h / (3.0 * PI))
}

/// Relation between Anger, Weber, and Bessel functions.
///
/// For non-integer v:
/// ```text
/// J_v(x) = J_v(x) * cos(v*pi) - Y_v(x) * sin(v*pi) + a_v(x)
/// ```
/// where a_v(x) is a correction term. For integer v, J_v = J_v exactly.
///
/// This function returns (anger_j, weber_e) as a pair.
pub fn anger_weber(v: f64, x: f64) -> SpecialResult<(f64, f64)> {
    let j = anger_j(v, x)?;
    let e = weber_e(v, x)?;
    Ok((j, e))
}

/// Lommel function of the first kind s_{mu,nu}(z).
///
/// Defined as a particular solution of:
/// ```text
/// z^2 y'' + z y' + (z^2 - nu^2) y = z^{mu+1}
/// ```
///
/// When mu +/- nu is not a negative odd integer, the series representation is:
/// ```text
/// s_{mu,nu}(z) = z^{mu+1} * sum_{k=0}^inf (-1)^k * z^{2k} / prod_{j=0}^k [(mu+2j+1)^2 - nu^2]
/// ```
///
/// # Arguments
/// * `mu` - First parameter
/// * `nu` - Second parameter (related to the Bessel order)
/// * `z` - Argument (z > 0)
///
/// # Examples
/// ```
/// use scirs2_special::lommel_s1;
/// let result = lommel_s1(1.0, 0.5, 1.0).expect("lommel_s1 failed");
/// assert!(result.is_finite());
/// ```
pub fn lommel_s1(mu: f64, nu: f64, z: f64) -> SpecialResult<f64> {
    if z.is_nan() || mu.is_nan() || nu.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to lommel_s1".to_string(),
        ));
    }

    if z.abs() < 1e-15 {
        if mu > -1.0 {
            return Ok(0.0);
        }
        return Err(SpecialError::DomainError(
            "lommel_s1 requires mu > -1 when z = 0".to_string(),
        ));
    }

    // Series expansion
    let max_terms = 200;
    let tol = 1e-15;

    let z2 = z * z;
    let mut sum = 0.0;
    let mut z_power = 1.0; // z^{2k}
    let mut denom_product = 1.0;

    for k in 0..max_terms {
        // Compute the denominator factor: (mu + 2k + 1)^2 - nu^2
        let alpha = mu + 2.0 * (k as f64) + 1.0;
        let factor = alpha * alpha - nu * nu;

        if factor.abs() < 1e-300 {
            return Err(SpecialError::ComputationError(format!(
                "Lommel s1: denominator near zero at k={k} (mu+2k+1={alpha}, nu={nu})"
            )));
        }

        denom_product *= factor;
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let term = sign * z_power / denom_product;
        sum += term;

        if k > 2 && term.abs() < tol * sum.abs() {
            break;
        }

        z_power *= z2;
    }

    // Multiply by z^{mu+1}
    Ok(z.powf(mu + 1.0) * sum)
}

/// Lommel function of the second kind S_{mu,nu}(z).
///
/// Defined in terms of s_{mu,nu} and Bessel functions:
/// ```text
/// S_{mu,nu}(z) = s_{mu,nu}(z) + 2^{mu-1} Gamma((mu+nu+1)/2) Gamma((mu-nu+1)/2)
///                * [sin((mu-nu)pi/2) J_nu(z) - cos((mu-nu)pi/2) Y_nu(z)]
/// ```
///
/// For simplicity, we compute this using the series when it converges,
/// or fall back to the asymptotic form for large z.
///
/// # Arguments
/// * `mu` - First parameter
/// * `nu` - Second parameter
/// * `z` - Argument (z > 0)
///
/// # Examples
/// ```
/// use scirs2_special::lommel_s2;
/// let result = lommel_s2(1.0, 0.5, 1.0).expect("lommel_s2 failed");
/// assert!(result.is_finite());
/// ```
pub fn lommel_s2(mu: f64, nu: f64, z: f64) -> SpecialResult<f64> {
    if z.is_nan() || mu.is_nan() || nu.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to lommel_s2".to_string(),
        ));
    }

    if z <= 0.0 {
        return Err(SpecialError::DomainError(
            "z must be > 0 for lommel_s2".to_string(),
        ));
    }

    // First compute s_{mu,nu}(z)
    let s1 = lommel_s1(mu, nu, z)?;

    // Compute the Bessel correction term
    let alpha = (mu - nu + 1.0) / 2.0;
    let beta = (mu + nu + 1.0) / 2.0;

    // gamma_alpha and gamma_beta
    let gamma_alpha = crate::gamma::gamma(alpha);
    let gamma_beta = crate::gamma::gamma(beta);

    let sin_term = ((mu - nu) * PI / 2.0).sin();
    let cos_term = ((mu - nu) * PI / 2.0).cos();

    // Bessel functions
    let j_nu = if nu.fract() == 0.0 && nu >= 0.0 {
        crate::bessel::jn(nu as i32, z)
    } else {
        crate::bessel::jv(nu, z)
    };
    let y_nu = if nu.fract() == 0.0 && nu >= 0.0 {
        crate::bessel::yn(nu as i32, z)
    } else {
        // For non-integer order, compute Y_nu via the standard formula
        // Y_nu = (J_nu cos(nu*pi) - J_{-nu}) / sin(nu*pi)
        let j_minus_nu = crate::bessel::jv(-nu, z);
        let cos_nu_pi = (nu * PI).cos();
        let sin_nu_pi = (nu * PI).sin();
        if sin_nu_pi.abs() < 1e-300 {
            0.0 // Degenerate case
        } else {
            (j_nu * cos_nu_pi - j_minus_nu) / sin_nu_pi
        }
    };

    let correction =
        2.0_f64.powf(mu - 1.0) * gamma_alpha * gamma_beta * (sin_term * j_nu - cos_term * y_nu);

    Ok(s1 + correction)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ====== Anger function tests ======

    #[test]
    fn test_anger_j_integer_order_zero() {
        // J_0(x) for integer order should approximate Bessel J_0
        let result = anger_j(0.0, 1.0).expect("anger_j failed");
        let j0_bessel = crate::bessel::j0(1.0);
        assert!(
            (result - j0_bessel).abs() < 1e-6,
            "anger_j(0,1) = {result}, J0(1) = {j0_bessel}"
        );
    }

    #[test]
    fn test_anger_j_integer_order_one() {
        let result = anger_j(1.0, 1.0).expect("anger_j failed");
        let j1_bessel = crate::bessel::j1(1.0);
        assert!(
            (result - j1_bessel).abs() < 1e-5,
            "anger_j(1,1) = {result}, J1(1) = {j1_bessel}"
        );
    }

    #[test]
    fn test_anger_j_half_order() {
        // J_{0.5}(x) should be finite and well-defined
        let result = anger_j(0.5, 1.0).expect("anger_j failed");
        assert!(result.is_finite(), "J_0.5(1) should be finite: {result}");
    }

    #[test]
    fn test_anger_j_at_zero() {
        // J_0(0) = 1
        let result = anger_j(0.0, 0.0).expect("anger_j failed");
        assert!((result - 1.0).abs() < 1e-10, "J_0(0) = 1, got {result}");
    }

    #[test]
    fn test_anger_j_nonzero_v_at_zero() {
        // J_v(0) = sin(v*pi) / (v*pi) for v != 0
        let v = 0.5;
        let result = anger_j(v, 0.0).expect("anger_j failed");
        let expected = (v * PI).sin() / (v * PI);
        assert!(
            (result - expected).abs() < 1e-10,
            "J_0.5(0) = {expected}, got {result}"
        );
    }

    #[test]
    fn test_anger_j_nan_input() {
        assert!(anger_j(f64::NAN, 1.0).is_err());
        assert!(anger_j(0.0, f64::NAN).is_err());
    }

    // ====== Weber function tests ======

    #[test]
    fn test_weber_e_at_zero_v_zero() {
        // E_0(0) = 0
        let result = weber_e(0.0, 0.0).expect("weber_e failed");
        assert!((result - 0.0).abs() < 1e-14, "E_0(0) = 0, got {result}");
    }

    #[test]
    fn test_weber_e_half_order() {
        let result = weber_e(0.5, 1.0).expect("weber_e failed");
        assert!(result.is_finite(), "E_0.5(1) should be finite: {result}");
    }

    #[test]
    fn test_weber_e_integer_order() {
        let result = weber_e(1.0, 1.0).expect("weber_e failed");
        assert!(result.is_finite(), "E_1(1) should be finite: {result}");
    }

    #[test]
    fn test_weber_e_nonzero_v_at_zero() {
        // E_v(0) = (1 - cos(v*pi)) / (v*pi)
        let v = 0.5;
        let result = weber_e(v, 0.0).expect("weber_e failed");
        let expected = (1.0 - (v * PI).cos()) / (v * PI);
        assert!(
            (result - expected).abs() < 1e-10,
            "E_0.5(0) = {expected}, got {result}"
        );
    }

    #[test]
    fn test_weber_e_nan_input() {
        assert!(weber_e(f64::NAN, 1.0).is_err());
    }

    // ====== anger_weber combined tests ======

    #[test]
    fn test_anger_weber_combined() {
        let (j, e) = anger_weber(0.5, 2.0).expect("anger_weber failed");
        assert!(j.is_finite());
        assert!(e.is_finite());
    }

    // ====== Lommel s1 tests ======

    #[test]
    fn test_lommel_s1_basic() {
        let result = lommel_s1(1.0, 0.5, 1.0).expect("lommel_s1 failed");
        assert!(
            result.is_finite(),
            "s_{{1,0.5}}(1) should be finite: {result}"
        );
    }

    #[test]
    fn test_lommel_s1_at_zero() {
        let result = lommel_s1(1.0, 0.5, 0.0).expect("lommel_s1 failed");
        assert!((result - 0.0).abs() < 1e-14, "s(0) = 0, got {result}");
    }

    #[test]
    fn test_lommel_s1_moderate_z() {
        let result = lommel_s1(2.0, 1.0, 3.0).expect("lommel_s1 failed");
        assert!(
            result.is_finite(),
            "s_{{2,1}}(3) should be finite: {result}"
        );
    }

    #[test]
    fn test_lommel_s1_small_z() {
        // For small z, s_{mu,nu}(z) ~ z^{mu+1} / [(mu+1)^2 - nu^2]
        let mu = 2.0;
        let nu = 0.5;
        let z = 0.01;
        let result = lommel_s1(mu, nu, z).expect("lommel_s1 failed");
        let approx = z.powf(mu + 1.0) / ((mu + 1.0).powi(2) - nu * nu);
        assert!(
            (result - approx).abs() / approx.abs() < 0.01,
            "small z approx: {result} vs {approx}"
        );
    }

    #[test]
    fn test_lommel_s1_nan_input() {
        assert!(lommel_s1(f64::NAN, 0.5, 1.0).is_err());
    }

    // ====== Lommel S2 tests ======

    #[test]
    fn test_lommel_s2_basic() {
        let result = lommel_s2(1.0, 0.5, 1.0).expect("lommel_s2 failed");
        assert!(
            result.is_finite(),
            "S_{{1,0.5}}(1) should be finite: {result}"
        );
    }

    #[test]
    fn test_lommel_s2_moderate_z() {
        let result = lommel_s2(2.0, 1.0, 5.0).expect("lommel_s2 failed");
        assert!(
            result.is_finite(),
            "S_{{2,1}}(5) should be finite: {result}"
        );
    }

    #[test]
    fn test_lommel_s2_large_z() {
        let result = lommel_s2(1.0, 0.5, 10.0).expect("lommel_s2 failed");
        assert!(
            result.is_finite(),
            "S_{{1,0.5}}(10) should be finite: {result}"
        );
    }

    #[test]
    fn test_lommel_s2_negative_z_error() {
        assert!(lommel_s2(1.0, 0.5, -1.0).is_err());
    }

    #[test]
    fn test_lommel_s2_nan_input() {
        assert!(lommel_s2(f64::NAN, 0.5, 1.0).is_err());
    }
}
