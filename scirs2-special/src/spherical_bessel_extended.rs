//! Modified spherical Bessel functions and Riccati-Bessel functions
//!
//! This module provides:
//! - `spherical_in(n, x)`: Modified spherical Bessel function of the first kind i_n(x)
//! - `spherical_kn(n, x)`: Modified spherical Bessel function of the second kind k_n(x)
//! - `spherical_in_derivative(n, x)`: Derivative of i_n(x)
//! - `spherical_kn_derivative(n, x)`: Derivative of k_n(x)
//! - `riccati_jn(n, x)`: Riccati-Bessel function S_n(x) = x * j_n(x)
//! - `riccati_yn(n, x)`: Riccati-Bessel function C_n(x) = -x * y_n(x)
//!
//! ## Mathematical Background
//!
//! ### Modified spherical Bessel functions
//!
//! The modified spherical Bessel functions are related to the modified Bessel functions by:
//! - i_n(x) = sqrt(pi/(2x)) * I_{n+1/2}(x)
//! - k_n(x) = sqrt(2/(pi*x)) * K_{n+1/2}(x)
//!
//! They satisfy the differential equation:
//! x^2 y'' + 2x y' - [x^2 + n(n+1)] y = 0
//!
//! ### Riccati-Bessel functions
//!
//! The Riccati-Bessel functions are defined as:
//! - S_n(x) = x * j_n(x)
//! - C_n(x) = -x * y_n(x)
//!
//! They are useful in electromagnetic scattering theory (Mie scattering).
//!
//! ## References
//!
//! 1. Abramowitz, M. and Stegun, I. A. (1972). Handbook of Mathematical Functions, Ch. 10.
//! 2. NIST Digital Library of Mathematical Functions, Ch. 10.47-10.49.

use crate::bessel::spherical::{spherical_jn, spherical_yn};
use crate::error::{SpecialError, SpecialResult};
use std::f64::consts::PI;

/// Modified spherical Bessel function of the first kind i_n(x).
///
/// Defined as: i_n(x) = sqrt(pi/(2x)) * I_{n+1/2}(x)
///
/// For real x, this can also be expressed as:
/// - i_0(x) = sinh(x)/x
/// - i_1(x) = cosh(x)/x - sinh(x)/x^2
///
/// And for general n: i_n(x) = (-i)^n * j_n(ix)
///
/// # Arguments
///
/// * `n` - Order (non-negative integer)
/// * `x` - Real argument
///
/// # Returns
///
/// * `SpecialResult<f64>` - Value of i_n(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::spherical_in;
///
/// // i_0(1.0) = sinh(1.0) / 1.0
/// let val = spherical_in(0, 1.0).expect("spherical_in failed");
/// assert!((val - 1.0_f64.sinh()).abs() < 1e-10);
/// ```
pub fn spherical_in(n: i32, x: f64) -> SpecialResult<f64> {
    if n < 0 {
        return Err(SpecialError::DomainError(
            "Order n must be non-negative for spherical_in".to_string(),
        ));
    }

    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to spherical_in".to_string(),
        ));
    }

    // Special case x = 0
    if x == 0.0 {
        return Ok(if n == 0 { 1.0 } else { 0.0 });
    }

    // For very small |x|, use series expansion
    if x.abs() < 1e-15 {
        return Ok(if n == 0 { 1.0 } else { 0.0 });
    }

    // Direct formulas for n = 0 and n = 1
    if n == 0 {
        return spherical_i0(x);
    }
    if n == 1 {
        return spherical_i1(x);
    }

    // For small |x|, use power series
    if x.abs() < 0.5 * (n as f64 + 1.0) {
        return spherical_in_series(n, x);
    }

    // For moderate to large arguments, use recurrence relation
    // Forward recurrence: i_{n+1}(x) = -(2n+1)/x * i_n(x) + i_{n-1}(x)
    // This is stable in the forward direction for i_n
    let mut i_prev = spherical_i0(x)?;
    let mut i_curr = spherical_i1(x)?;

    for k in 1..n {
        let i_next = i_prev - (2.0 * k as f64 + 1.0) / x * i_curr;
        i_prev = i_curr;
        i_curr = i_next;

        // Safety check
        if !i_curr.is_finite() {
            return Ok(f64::INFINITY);
        }
    }

    Ok(i_curr)
}

/// i_0(x) = sinh(x)/x
fn spherical_i0(x: f64) -> SpecialResult<f64> {
    if x.abs() < 1e-8 {
        // Series: sinh(x)/x = 1 + x^2/6 + x^4/120 + ...
        let x2 = x * x;
        return Ok(1.0 + x2 / 6.0 + x2 * x2 / 120.0);
    }
    Ok(x.sinh() / x)
}

/// i_1(x) = cosh(x)/x - sinh(x)/x^2 = -sinh(x)/x^2 + cosh(x)/x
fn spherical_i1(x: f64) -> SpecialResult<f64> {
    if x.abs() < 1e-8 {
        // Series: i_1(x) = x/3 + x^3/30 + x^5/840 + ...
        let x2 = x * x;
        return Ok(x / 3.0 + x * x2 / 30.0 + x * x2 * x2 / 840.0);
    }
    Ok(x.cosh() / x - x.sinh() / (x * x))
}

/// Power series for i_n(x) for small arguments
fn spherical_in_series(n: i32, x: f64) -> SpecialResult<f64> {
    // i_n(x) = (x^n) / (2n+1)!! * sum_{k=0}^inf x^{2k} / (2^k * k! * prod_{j=1}^k (2n+2j+1))
    // Simplified: use the relation i_n(x) = (-i)^n * j_n(ix) where j_n is spherical Bessel
    // But that involves complex arithmetic. Instead, use the direct series:
    //
    // i_n(x) = sqrt(pi/(2x)) * I_{n+1/2}(x)
    // I_v(x) = (x/2)^v * sum_{k=0}^inf (x/2)^{2k} / (k! * Gamma(v+k+1))
    //
    // For the modified function, all signs are positive (no alternating).

    let x_half = x / 2.0;
    let v = n as f64 + 0.5;

    // Compute (x/2)^v
    let log_prefix = v * x_half.abs().ln();
    if log_prefix > 700.0 {
        return Ok(f64::INFINITY);
    }
    if log_prefix < -700.0 {
        return Ok(0.0);
    }

    let x_half_sq = x_half * x_half;

    // First term: 1/Gamma(v+1)
    let log_gamma_v_plus_1 = crate::gamma::gammaln(v + 1.0);

    let mut term = (-log_gamma_v_plus_1).exp();
    let mut sum = term;

    for k in 1..100 {
        term *= x_half_sq / (k as f64 * (v + k as f64));
        sum += term;

        if term.abs() < 1e-16 * sum.abs() && k > 3 {
            break;
        }
        if !term.is_finite() {
            break;
        }
    }

    // i_n(x) = sqrt(pi/(2x)) * (x/2)^v * sum
    // But we need to handle x = 0 specially (already done above)
    let prefix = (PI / (2.0 * x.abs())).sqrt() * x_half.abs().powf(v);

    // Handle sign for negative x
    let sign = if x < 0.0 && n % 2 != 0 { -1.0 } else { 1.0 };

    Ok(sign * prefix * sum)
}

/// Modified spherical Bessel function of the second kind k_n(x).
///
/// Defined as: k_n(x) = sqrt(2/(pi*x)) * K_{n+1/2}(x)
///
/// For positive real x:
/// - k_0(x) = (pi/(2x)) * e^(-x)
/// - k_1(x) = (pi/(2x)) * e^(-x) * (1 + 1/x)
///
/// # Arguments
///
/// * `n` - Order (non-negative integer)
/// * `x` - Real argument (must be positive)
///
/// # Returns
///
/// * `SpecialResult<f64>` - Value of k_n(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::spherical_kn;
///
/// // k_0(1.0) = (pi/2) * e^(-1)
/// let val = spherical_kn(0, 1.0).expect("spherical_kn failed");
/// let expected = (std::f64::consts::PI / 2.0) * (-1.0_f64).exp();
/// assert!((val - expected).abs() < 1e-10);
/// ```
pub fn spherical_kn(n: i32, x: f64) -> SpecialResult<f64> {
    if n < 0 {
        return Err(SpecialError::DomainError(
            "Order n must be non-negative for spherical_kn".to_string(),
        ));
    }

    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to spherical_kn".to_string(),
        ));
    }

    if x <= 0.0 {
        return Err(SpecialError::DomainError(
            "spherical_kn requires positive x".to_string(),
        ));
    }

    // k_n(x) has a closed form: k_n(x) = (pi/(2x)) * e^(-x) * sum_{k=0}^n (n+k)! / (k! * (n-k)! * (2x)^k)
    let prefactor = (PI / (2.0 * x)) * (-x).exp();

    if n == 0 {
        return Ok(prefactor);
    }

    // General formula: k_n(x) = (pi/(2x)) * e^(-x) * P_n(1/x)
    // where P_n is a polynomial in 1/x with specific coefficients
    let mut sum = 1.0;
    let two_x_inv = 1.0 / (2.0 * x);
    let mut term = 1.0;

    for k in 1..=n {
        let k_f = k as f64;
        let n_f = n as f64;
        // term_k = (n+k)! / (k! * (n-k)!) * (1/(2x))^k
        // ratio: term_k / term_{k-1} = (n+k)*(n-k+1) / k * (1/(2x))
        // But more directly:
        // coeff = C(n+k, 2k) * (2k)! / k! = prod_{j=1}^k (n+j)*(n-j+1)/j ...
        // Actually simpler: the product form
        // (n+k)! / (k! * (n-k)!) = binomial(n+k, 2k) * ... let me use the simpler recurrence
        term *= (n_f + k_f) * (n_f - k_f + 1.0) / k_f * two_x_inv;
        sum += term;

        if !term.is_finite() {
            break;
        }
    }

    Ok(prefactor * sum)
}

/// Derivative of the modified spherical Bessel function of the first kind.
///
/// Computes d/dx [i_n(x)] using the recurrence relation:
/// i_n'(x) = i_{n-1}(x) - (n+1)/x * i_n(x)  (for n >= 1)
/// i_0'(x) = i_1(x)
///
/// # Arguments
///
/// * `n` - Order (non-negative integer)
/// * `x` - Real argument
///
/// # Returns
///
/// * `SpecialResult<f64>` - Value of i_n'(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::spherical_in_derivative;
///
/// let val = spherical_in_derivative(0, 1.0).expect("spherical_in_derivative failed");
/// assert!(val.is_finite());
/// ```
pub fn spherical_in_derivative(n: i32, x: f64) -> SpecialResult<f64> {
    if n < 0 {
        return Err(SpecialError::DomainError(
            "Order n must be non-negative for spherical_in_derivative".to_string(),
        ));
    }

    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to spherical_in_derivative".to_string(),
        ));
    }

    if n == 0 {
        // i_0'(x) = i_1(x)
        return spherical_in(1, x);
    }

    if x == 0.0 {
        // i_n'(0) = 0 for n >= 2, and i_1'(0) = 1/3
        if n == 1 {
            return Ok(1.0 / 3.0);
        }
        return Ok(0.0);
    }

    // Use recurrence: i_n'(x) = i_{n-1}(x) - (n+1)/x * i_n(x)
    let i_n = spherical_in(n, x)?;
    let i_nm1 = spherical_in(n - 1, x)?;

    Ok(i_nm1 - (n as f64 + 1.0) / x * i_n)
}

/// Derivative of the modified spherical Bessel function of the second kind.
///
/// Computes d/dx [k_n(x)] using the recurrence relation:
/// k_n'(x) = -k_{n-1}(x) - (n+1)/x * k_n(x)
///
/// # Arguments
///
/// * `n` - Order (non-negative integer)
/// * `x` - Real argument (must be positive)
///
/// # Returns
///
/// * `SpecialResult<f64>` - Value of k_n'(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::spherical_kn_derivative;
///
/// let val = spherical_kn_derivative(0, 1.0).expect("spherical_kn_derivative failed");
/// assert!(val.is_finite());
/// ```
pub fn spherical_kn_derivative(n: i32, x: f64) -> SpecialResult<f64> {
    if n < 0 {
        return Err(SpecialError::DomainError(
            "Order n must be non-negative for spherical_kn_derivative".to_string(),
        ));
    }

    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to spherical_kn_derivative".to_string(),
        ));
    }

    if x <= 0.0 {
        return Err(SpecialError::DomainError(
            "spherical_kn_derivative requires positive x".to_string(),
        ));
    }

    if n == 0 {
        // k_0'(x) = -k_1(x) (using the relation k_0'(x) = -k_1(x) + 0 since n=0 => no (n+1)/x correction needed for n-1=-1)
        // Actually k_0'(x) = d/dx[(pi/(2x)) * e^(-x)] = -(pi/(2x))e^(-x) - (pi/(2x^2))e^(-x)
        //                   = -(pi/(2x))e^(-x)(1 + 1/x) = -k_1(x)
        let k1 = spherical_kn(1, x)?;
        return Ok(-k1);
    }

    // General formula: k_n'(x) = -k_{n-1}(x) - (n+1)/x * k_n(x)
    let k_n = spherical_kn(n, x)?;
    let k_nm1 = spherical_kn(n - 1, x)?;

    Ok(-k_nm1 - (n as f64 + 1.0) / x * k_n)
}

/// Riccati-Bessel function S_n(x) = x * j_n(x).
///
/// The Riccati-Bessel functions are particularly important in Mie scattering theory.
/// They satisfy the Riccati-Bessel differential equation:
/// x^2 y'' + [x^2 - n(n+1)] y = 0
///
/// # Arguments
///
/// * `n` - Order (non-negative integer)
/// * `x` - Real argument
///
/// # Returns
///
/// * `SpecialResult<(Vec<f64>, Vec<f64>)>` - Tuple of (S_0...S_n, S_0'...S_n') values
///
/// # Examples
///
/// ```
/// use scirs2_special::riccati_jn;
///
/// let (s_vals, sp_vals) = riccati_jn(3, 1.5).expect("riccati_jn failed");
/// // S_0(x) = x * j_0(x) = sin(x)
/// assert!((s_vals[0] - 1.5_f64.sin()).abs() < 1e-10);
/// ```
pub fn riccati_jn(n: i32, x: f64) -> SpecialResult<(Vec<f64>, Vec<f64>)> {
    if n < 0 {
        return Err(SpecialError::DomainError(
            "Order n must be non-negative for riccati_jn".to_string(),
        ));
    }

    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to riccati_jn".to_string(),
        ));
    }

    let n_usize = n as usize;
    let mut s_vals = vec![0.0; n_usize + 1];
    let mut sp_vals = vec![0.0; n_usize + 1];

    for k in 0..=n_usize {
        let jn_val: f64 = spherical_jn(k as i32, x);

        s_vals[k] = x * jn_val;

        // S_n'(x) = j_n(x) + x * j_n'(x)
        // Using j_n'(x) = j_{n-1}(x) - (n+1)/x * j_n(x)  (for n >= 1)
        // j_0'(x) = -j_1(x)
        if k == 0 {
            if x.abs() < 1e-15 {
                // S_0'(0) = cos(0) = 1
                sp_vals[0] = 1.0;
            } else {
                // S_0(x) = sin(x), S_0'(x) = cos(x)
                sp_vals[0] = x.cos();
            }
        } else if x.abs() < 1e-15 {
            sp_vals[k] = 0.0;
        } else {
            let jn_prev: f64 = spherical_jn(k as i32 - 1, x);
            let jn_prime = jn_prev - (k as f64 + 1.0) / x * jn_val;
            sp_vals[k] = jn_val + x * jn_prime;
        }
    }

    Ok((s_vals, sp_vals))
}

/// Riccati-Bessel function C_n(x) = -x * y_n(x).
///
/// The Riccati-Bessel functions are used in Mie scattering theory.
///
/// # Arguments
///
/// * `n` - Order (non-negative integer)
/// * `x` - Real argument (must be positive)
///
/// # Returns
///
/// * `SpecialResult<(Vec<f64>, Vec<f64>)>` - Tuple of (C_0...C_n, C_0'...C_n') values
///
/// # Examples
///
/// ```
/// use scirs2_special::riccati_yn;
///
/// let (c_vals, cp_vals) = riccati_yn(3, 1.5).expect("riccati_yn failed");
/// // C_0(x) = -x * y_0(x) = -x * (-cos(x)/x) = cos(x)
/// assert!((c_vals[0] - 1.5_f64.cos()).abs() < 1e-10);
/// ```
pub fn riccati_yn(n: i32, x: f64) -> SpecialResult<(Vec<f64>, Vec<f64>)> {
    if n < 0 {
        return Err(SpecialError::DomainError(
            "Order n must be non-negative for riccati_yn".to_string(),
        ));
    }

    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to riccati_yn".to_string(),
        ));
    }

    if x <= 0.0 {
        return Err(SpecialError::DomainError(
            "riccati_yn requires positive x".to_string(),
        ));
    }

    let n_usize = n as usize;
    let mut c_vals = vec![0.0; n_usize + 1];
    let mut cp_vals = vec![0.0; n_usize + 1];

    for k in 0..=n_usize {
        let yn_val: f64 = spherical_yn(k as i32, x);

        c_vals[k] = -x * yn_val;

        // C_n'(x) = -y_n(x) - x * y_n'(x)
        // Using y_n'(x) = y_{n-1}(x) - (n+1)/x * y_n(x) (for n >= 1)
        // y_0'(x) = -y_1(x) ... wait, y_0'(x) = d/dx[-cos(x)/x] = sin(x)/x - cos(x)/x^2
        if k == 0 {
            // C_0(x) = cos(x), C_0'(x) = -sin(x)
            cp_vals[0] = -x.sin();
        } else {
            let yn_prev: f64 = spherical_yn(k as i32 - 1, x);
            let yn_prime = yn_prev - (k as f64 + 1.0) / x * yn_val;
            cp_vals[k] = -yn_val - x * yn_prime;
        }
    }

    Ok((c_vals, cp_vals))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ============ Modified spherical Bessel i_n tests ============

    #[test]
    fn test_spherical_i0() {
        // i_0(x) = sinh(x)/x
        let x = 1.0;
        let val = spherical_in(0, x).expect("spherical_in(0, 1.0) failed");
        let expected = x.sinh() / x;
        assert_relative_eq!(val, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_spherical_i0_at_zero() {
        let val = spherical_in(0, 0.0).expect("spherical_in(0, 0.0) failed");
        assert_relative_eq!(val, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spherical_i1() {
        // i_1(x) = cosh(x)/x - sinh(x)/x^2
        let x = 2.0;
        let val = spherical_in(1, x).expect("spherical_in(1, 2.0) failed");
        let expected = x.cosh() / x - x.sinh() / (x * x);
        assert_relative_eq!(val, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_spherical_in_positive_values() {
        // i_n(x) should be positive for positive x and all n >= 0
        for n in 0..=5 {
            let val = spherical_in(n, 1.5).expect("spherical_in failed");
            assert!(val > 0.0, "i_{}(1.5) should be positive, got {}", n, val);
        }
    }

    #[test]
    fn test_spherical_in_small_x() {
        // For small x, i_0(x) ~ 1
        let val = spherical_in(0, 1e-10).expect("spherical_in failed");
        assert_relative_eq!(val, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_spherical_in_nan_input() {
        assert!(spherical_in(0, f64::NAN).is_err());
    }

    #[test]
    fn test_spherical_in_negative_order() {
        assert!(spherical_in(-1, 1.0).is_err());
    }

    // ============ Modified spherical Bessel k_n tests ============

    #[test]
    fn test_spherical_k0() {
        // k_0(x) = (pi/(2x)) * e^(-x)
        let x = 1.0;
        let val = spherical_kn(0, x).expect("spherical_kn(0, 1.0) failed");
        let expected = (PI / (2.0 * x)) * (-x).exp();
        assert_relative_eq!(val, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_spherical_k1() {
        // k_1(x) = (pi/(2x)) * e^(-x) * (1 + 1/x)
        let x = 1.0;
        let val = spherical_kn(1, x).expect("spherical_kn(1, 1.0) failed");
        let expected = (PI / (2.0 * x)) * (-x).exp() * (1.0 + 1.0 / x);
        assert_relative_eq!(val, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_spherical_kn_positive_values() {
        // k_n(x) should be positive for positive x
        for n in 0..=5 {
            let val = spherical_kn(n, 1.5).expect("spherical_kn failed");
            assert!(val > 0.0, "k_{}(1.5) should be positive, got {}", n, val);
        }
    }

    #[test]
    fn test_spherical_kn_decreasing() {
        // k_n(x) should decrease as x increases for fixed n
        let k0_1 = spherical_kn(0, 1.0).expect("failed");
        let k0_2 = spherical_kn(0, 2.0).expect("failed");
        assert!(k0_1 > k0_2, "k_0 should decrease with x");
    }

    #[test]
    fn test_spherical_kn_nonpositive() {
        assert!(spherical_kn(0, 0.0).is_err());
        assert!(spherical_kn(0, -1.0).is_err());
    }

    // ============ Derivative tests ============

    #[test]
    fn test_spherical_in_derivative_numerical() {
        // Verify derivative using finite differences
        let n = 2;
        let x = 1.5;
        let h = 1e-7;

        let deriv = spherical_in_derivative(n, x).expect("derivative failed");
        let i_plus = spherical_in(n, x + h).expect("failed");
        let i_minus = spherical_in(n, x - h).expect("failed");
        let numerical = (i_plus - i_minus) / (2.0 * h);

        assert_relative_eq!(deriv, numerical, epsilon = 1e-5);
    }

    #[test]
    fn test_spherical_kn_derivative_numerical() {
        let n = 2;
        let x = 1.5;
        let h = 1e-7;

        let deriv = spherical_kn_derivative(n, x).expect("derivative failed");
        let k_plus = spherical_kn(n, x + h).expect("failed");
        let k_minus = spherical_kn(n, x - h).expect("failed");
        let numerical = (k_plus - k_minus) / (2.0 * h);

        assert_relative_eq!(deriv, numerical, epsilon = 1e-5);
    }

    // ============ Riccati-Bessel tests ============

    #[test]
    fn test_riccati_jn_s0() {
        // S_0(x) = x * j_0(x) = sin(x)
        let x = 1.5;
        let (s_vals, sp_vals) = riccati_jn(0, x).expect("riccati_jn failed");
        assert_relative_eq!(s_vals[0], x.sin(), epsilon = 1e-10);
        assert_relative_eq!(sp_vals[0], x.cos(), epsilon = 1e-10);
    }

    #[test]
    fn test_riccati_jn_sequence() {
        let (s_vals, _) = riccati_jn(5, 2.0).expect("riccati_jn failed");
        // All values should be finite
        for (i, val) in s_vals.iter().enumerate() {
            assert!(
                val.is_finite(),
                "S_{}(2.0) should be finite, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_riccati_yn_c0() {
        // C_0(x) = -x * y_0(x) = -x * (-cos(x)/x) = cos(x)
        let x = 1.5;
        let (c_vals, cp_vals) = riccati_yn(0, x).expect("riccati_yn failed");
        assert_relative_eq!(c_vals[0], x.cos(), epsilon = 1e-10);
        assert_relative_eq!(cp_vals[0], -x.sin(), epsilon = 1e-10);
    }

    #[test]
    fn test_riccati_yn_sequence() {
        let (c_vals, _) = riccati_yn(5, 2.0).expect("riccati_yn failed");
        for (i, val) in c_vals.iter().enumerate() {
            assert!(
                val.is_finite(),
                "C_{}(2.0) should be finite, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_riccati_jn_negative_order() {
        assert!(riccati_jn(-1, 1.0).is_err());
    }

    #[test]
    fn test_riccati_yn_negative_x() {
        assert!(riccati_yn(0, -1.0).is_err());
    }

    #[test]
    fn test_riccati_wronskian() {
        // The Wronskian: S_n(x)*C_n'(x) - S_n'(x)*C_n(x) = -1
        // This follows from W[j_n, y_n] = 1/x^2 and the sign of C_n = -x*y_n:
        // W[S_n, C_n] = x^2 * (j_n'*y_n - j_n*y_n') = x^2 * (-1/x^2) = -1
        let x = 2.5;
        let (s_vals, sp_vals) = riccati_jn(3, x).expect("failed");
        let (c_vals, cp_vals) = riccati_yn(3, x).expect("failed");

        for n in 0..=3 {
            let wronskian = s_vals[n] * cp_vals[n] - sp_vals[n] * c_vals[n];
            assert_relative_eq!(wronskian, -1.0, epsilon = 1e-8);
        }
    }
}
