//! Meixner-Pollaczek polynomials and related discrete orthogonal polynomials
//!
//! This module provides implementations of the Meixner-Pollaczek polynomials
//! P_n^{(lambda)}(x; phi) and the related Charlier polynomials.
//!
//! ## Mathematical Background
//!
//! ### Meixner-Pollaczek Polynomials
//!
//! The Meixner-Pollaczek polynomials are a family of orthogonal polynomials
//! defined by the three-term recurrence relation:
//!
//! ```text
//! (n+1) P_{n+1}^{(lambda)}(x; phi) =
//!     2 (x sin(phi) + (n + lambda) cos(phi)) P_n^{(lambda)}(x; phi)
//!     - (n + 2*lambda - 1) P_{n-1}^{(lambda)}(x; phi)
//! ```
//!
//! with initial conditions:
//! ```text
//! P_0^{(lambda)}(x; phi) = 1
//! P_1^{(lambda)}(x; phi) = 2 (lambda cos(phi) + x sin(phi))
//! ```
//!
//! Orthogonality holds on the real line with weight function:
//! ```text
//! w(x) = exp((2*phi - pi)*x) * |Gamma(lambda + ix)|^2 / (2*pi)
//! ```
//! where lambda > 0 and 0 < phi < pi.
//!
//! ### Charlier Polynomials
//!
//! The Charlier polynomials C_n(x; a) are orthogonal with respect to the
//! Poisson distribution with parameter a > 0:
//!
//! ```text
//! C_n(x; a) = 2F0(-n, -x; -; -1/a)
//!           = sum_{k=0}^{n} binom(n,k) binom(x,k) k! (-1/a)^{n-k}
//! ```
//!
//! Three-term recurrence:
//! ```text
//! a C_{n+1}(x; a) = (a - x - n) C_n(x; a) - n C_{n-1}(x; a)
//! ```
//! (taking C_0 = 1, C_1 = 1 - x/a)

use crate::error::{SpecialError, SpecialResult};
use crate::gamma::gamma;

// ============================================================================
// Meixner-Pollaczek Polynomials
// ============================================================================

/// Compute the Meixner-Pollaczek polynomial P_n^{(lambda)}(x; phi)
/// using the three-term recurrence relation.
///
/// The Meixner-Pollaczek polynomial of degree n with parameters lambda and phi
/// is defined via:
///
/// ```text
/// P_0^{(lambda)}(x; phi) = 1
/// P_1^{(lambda)}(x; phi) = 2 * (lambda * cos(phi) + x * sin(phi))
/// (n+1) P_{n+1} = 2*(x*sin(phi) + (n+lambda)*cos(phi)) * P_n
///                 - (n + 2*lambda - 1) * P_{n-1}
/// ```
///
/// # Arguments
/// * `n`      - Degree (non-negative integer)
/// * `lambda` - Shape parameter (must satisfy lambda > 0)
/// * `phi`    - Angle parameter (must satisfy 0 < phi < pi)
/// * `x`      - Evaluation point
///
/// # Returns
/// Value of P_n^{(lambda)}(x; phi)
///
/// # Errors
/// Returns `SpecialError::DomainError` if lambda <= 0 or phi is not in (0, pi).
///
/// # Examples
/// ```
/// use scirs2_special::meixner_pollaczek::meixner_pollaczek;
/// // P_0 = 1 for all x
/// let p0 = meixner_pollaczek(0, 1.0, std::f64::consts::FRAC_PI_2, 0.0).expect("ok");
/// assert!((p0 - 1.0).abs() < 1e-14);
/// // P_1 at phi = pi/2: P_1 = 2*(lambda*0 + x*1) = 2x
/// let p1 = meixner_pollaczek(1, 1.0, std::f64::consts::FRAC_PI_2, 1.5).expect("ok");
/// assert!((p1 - 3.0).abs() < 1e-12);
/// ```
pub fn meixner_pollaczek(n: usize, lambda: f64, phi: f64, x: f64) -> SpecialResult<f64> {
    validate_mp_params(lambda, phi)?;
    Ok(meixner_pollaczek_recur(n, lambda, phi, x))
}

/// Compute all Meixner-Pollaczek polynomials up to degree `n_max`.
///
/// Returns a `Vec<f64>` of length `n_max + 1` containing
/// `P_0^{(lambda)}(x;phi), ..., P_{n_max}^{(lambda)}(x;phi)`.
///
/// Using a single recurrence pass this is more efficient than calling
/// `meixner_pollaczek` individually for each degree.
///
/// # Arguments
/// * `n_max`  - Maximum degree (inclusive)
/// * `lambda` - Shape parameter (lambda > 0)
/// * `phi`    - Angle parameter (0 < phi < pi)
/// * `x`      - Evaluation point
///
/// # Errors
/// Returns `SpecialError::DomainError` if lambda <= 0 or phi is not in (0, pi).
///
/// # Examples
/// ```
/// use scirs2_special::meixner_pollaczek::meixner_pollaczek_array;
/// let vals = meixner_pollaczek_array(3, 1.0, std::f64::consts::FRAC_PI_2, 0.0).expect("ok");
/// assert_eq!(vals.len(), 4);
/// // P_0 = 1, P_1(0) = 0, from recurrence
/// assert!((vals[0] - 1.0).abs() < 1e-14);
/// assert!(vals[1].abs() < 1e-14);
/// ```
pub fn meixner_pollaczek_array(
    n_max: usize,
    lambda: f64,
    phi: f64,
    x: f64,
) -> SpecialResult<Vec<f64>> {
    validate_mp_params(lambda, phi)?;

    let mut result = Vec::with_capacity(n_max + 1);
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();

    // P_0 = 1
    let p0 = 1.0_f64;
    result.push(p0);

    if n_max == 0 {
        return Ok(result);
    }

    // P_1 = 2*(lambda*cos(phi) + x*sin(phi))
    let p1 = 2.0 * (lambda * cos_phi + x * sin_phi);
    result.push(p1);

    if n_max == 1 {
        return Ok(result);
    }

    let mut p_prev = p0;
    let mut p_curr = p1;

    for k in 1..n_max {
        let n_f = k as f64;
        // (n+1) P_{n+1} = 2*(x*sin(phi) + (n+lambda)*cos(phi))*P_n - (n+2*lambda-1)*P_{n-1}
        let coeff_curr = 2.0 * (x * sin_phi + (n_f + lambda) * cos_phi);
        let coeff_prev = n_f + 2.0 * lambda - 1.0;
        let p_next = (coeff_curr * p_curr - coeff_prev * p_prev) / (n_f + 1.0);
        result.push(p_next);
        p_prev = p_curr;
        p_curr = p_next;
    }

    Ok(result)
}

/// Evaluate the Meixner-Pollaczek polynomial at multiple x values.
///
/// # Arguments
/// * `n`      - Degree
/// * `lambda` - Shape parameter (lambda > 0)
/// * `phi`    - Angle parameter (0 < phi < pi)
/// * `x_vals` - Slice of evaluation points
///
/// # Returns
/// Vec of P_n^{(lambda)}(x_k; phi) for each x_k in x_vals.
///
/// # Errors
/// Returns `SpecialError::DomainError` if lambda <= 0 or phi is not in (0, pi).
///
/// # Examples
/// ```
/// use scirs2_special::meixner_pollaczek::meixner_pollaczek_eval;
/// let xs = vec![0.0, 0.5, 1.0, -1.0];
/// let vals = meixner_pollaczek_eval(2, 1.0, std::f64::consts::FRAC_PI_2, &xs).expect("ok");
/// assert_eq!(vals.len(), 4);
/// ```
pub fn meixner_pollaczek_eval(
    n: usize,
    lambda: f64,
    phi: f64,
    x_vals: &[f64],
) -> SpecialResult<Vec<f64>> {
    validate_mp_params(lambda, phi)?;
    let result = x_vals
        .iter()
        .map(|&x| meixner_pollaczek_recur(n, lambda, phi, x))
        .collect();
    Ok(result)
}

/// Compute the weight function for Meixner-Pollaczek orthogonality.
///
/// The weight function is:
/// ```text
/// w(x; lambda, phi) = exp((2*phi - pi)*x) * |Gamma(lambda + ix)|^2 / (2*pi)
/// ```
///
/// This is used to verify orthogonality relations numerically.
///
/// # Arguments
/// * `x`      - Real evaluation point
/// * `lambda` - Shape parameter (lambda > 0)
/// * `phi`    - Angle parameter (0 < phi < pi)
///
/// # Errors
/// Returns `SpecialError::DomainError` for invalid parameters.
///
/// # Examples
/// ```
/// use scirs2_special::meixner_pollaczek::meixner_pollaczek_weight;
/// let w = meixner_pollaczek_weight(0.0, 1.0, std::f64::consts::FRAC_PI_4).expect("ok");
/// assert!(w > 0.0);
/// ```
pub fn meixner_pollaczek_weight(x: f64, lambda: f64, phi: f64) -> SpecialResult<f64> {
    validate_mp_params(lambda, phi)?;

    // Compute |Gamma(lambda + ix)|^2 using the reflection formula and Stirling series.
    // For real lambda > 0 and real x:
    // |Gamma(lambda + ix)|^2 = Gamma(lambda + ix) * Gamma(lambda - ix)
    //
    // We use the log-modulus approach:
    // log|Gamma(lambda + ix)| = Re(log Gamma(lambda + ix))
    // and compute via the Stirling-Lanczos approximation.
    let log_abs_gamma_sq = 2.0 * log_abs_gamma_complex(lambda, x)?;
    let exp_factor = (2.0 * phi - std::f64::consts::PI) * x;
    let result = ((exp_factor + log_abs_gamma_sq) - (2.0 * std::f64::consts::PI).ln()).exp();
    Ok(result)
}

// ============================================================================
// Charlier Polynomials
// ============================================================================

/// Compute the Charlier polynomial C_n(x; a).
///
/// The Charlier polynomials are orthogonal with respect to the Poisson measure
/// on the non-negative integers. They are defined via the hypergeometric
/// representation:
///
/// ```text
/// C_n(x; a) = 2F0(-n, -x; -; -1/a)
/// ```
///
/// and satisfy the recurrence:
/// ```text
/// C_0(x;a) = 1,
/// C_1(x;a) = 1 - x/a
/// a * C_{n+1}(x;a) = (a - n - x) * C_n(x;a) - n * C_{n-1}(x;a)
/// ```
///
/// # Arguments
/// * `n` - Degree (non-negative integer)
/// * `a` - Parameter (a > 0)
/// * `x` - Evaluation point (can be any real number, though classically a non-negative integer)
///
/// # Returns
/// Value of C_n(x; a)
///
/// # Errors
/// Returns `SpecialError::DomainError` if a <= 0.
///
/// # Examples
/// ```
/// use scirs2_special::meixner_pollaczek::charlier_polynomial;
/// // C_0 = 1
/// let c0 = charlier_polynomial(0, 2.0, 3.0).expect("ok");
/// assert!((c0 - 1.0).abs() < 1e-14);
/// // C_1(x; a) = 1 - x/a
/// let c1 = charlier_polynomial(1, 2.0, 4.0).expect("ok");
/// assert!((c1 - (1.0 - 4.0/2.0)).abs() < 1e-14);
/// ```
pub fn charlier_polynomial(n: usize, a: f64, x: f64) -> SpecialResult<f64> {
    if a <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "Charlier polynomial requires a > 0, got a = {a}"
        )));
    }

    match n {
        0 => Ok(1.0),
        1 => Ok(1.0 - x / a),
        _ => {
            let mut c_prev = 1.0_f64; // C_{k-1}
            let mut c_curr = 1.0 - x / a; // C_k

            for k in 1..n {
                let k_f = k as f64;
                // a * C_{k+1} = (a - k - x) * C_k - k * C_{k-1}
                let c_next = ((a - k_f - x) * c_curr - k_f * c_prev) / a;
                c_prev = c_curr;
                c_curr = c_next;
            }

            Ok(c_curr)
        }
    }
}

/// Compute all Charlier polynomials C_0(x;a), ..., C_{n_max}(x;a).
///
/// # Arguments
/// * `n_max` - Maximum degree (inclusive)
/// * `a`     - Parameter (a > 0)
/// * `x`     - Evaluation point
///
/// # Returns
/// Vec of length n_max + 1 containing C_k(x; a) for k = 0, ..., n_max.
///
/// # Errors
/// Returns `SpecialError::DomainError` if a <= 0.
///
/// # Examples
/// ```
/// use scirs2_special::meixner_pollaczek::charlier_polynomial_array;
/// let vals = charlier_polynomial_array(4, 1.0, 2.0).expect("ok");
/// assert_eq!(vals.len(), 5);
/// assert!((vals[0] - 1.0).abs() < 1e-14);
/// ```
pub fn charlier_polynomial_array(n_max: usize, a: f64, x: f64) -> SpecialResult<Vec<f64>> {
    if a <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "Charlier polynomial requires a > 0, got a = {a}"
        )));
    }

    let mut result = Vec::with_capacity(n_max + 1);
    let c0 = 1.0_f64;
    result.push(c0);

    if n_max == 0 {
        return Ok(result);
    }

    let c1 = 1.0 - x / a;
    result.push(c1);

    let mut c_prev = c0;
    let mut c_curr = c1;

    for k in 1..n_max {
        let k_f = k as f64;
        let c_next = ((a - k_f - x) * c_curr - k_f * c_prev) / a;
        result.push(c_next);
        c_prev = c_curr;
        c_curr = c_next;
    }

    Ok(result)
}

// ============================================================================
// Meixner Polynomials (discrete, related to Meixner-Pollaczek)
// ============================================================================

/// Compute the Meixner polynomial M_n(x; beta, c).
///
/// The Meixner polynomials are a family of discrete orthogonal polynomials
/// related to the negative binomial distribution. They satisfy:
///
/// ```text
/// M_0(x; beta, c) = 1
/// M_1(x; beta, c) = 1 - x*(1-c)/(beta*c)  (simplified from recurrence)
/// (n+1) M_{n+1}(x) = ((2n+beta)(1+c) - x*(1-c)*c^{-1} + x*(1-c)) M_n(x)/(1+c)
///                    - ... (full recurrence below)
/// ```
///
/// Full recurrence (the standard form is):
/// ```text
/// (n+1)(1-c) M_{n+1}(x) = [c*(2n+beta) - (1-c)*x - n] (1-c)/c^{1/2} M_n(x)
///                          ... actually using the canonical form:
/// ```
/// The canonical three-term recurrence is:
/// ```text
/// c*(n + beta) M_{n+1}(x) = (c*(2n + beta) + n - x*(1-c)) M_n(x) - n * M_{n-1}(x)
/// ```
/// Wait — the standard form used in numerical literature is:
/// ```text
/// M_0 = 1, M_1 = 1 - x*(1-c)/(beta*c)  ... actually let's use the 2F1 form.
/// M_n(x; beta, c) = 2F1(-n, -x; beta; 1 - 1/c)
/// ```
///
/// # Arguments
/// * `n`    - Degree
/// * `beta` - Shape parameter (beta > 0)
/// * `c`    - Parameter (0 < c < 1)
/// * `x`    - Evaluation point (classically non-negative integer)
///
/// # Returns
/// Value of M_n(x; beta, c)
///
/// # Errors
/// Returns `SpecialError::DomainError` if beta <= 0 or c is not in (0, 1).
///
/// # Examples
/// ```
/// use scirs2_special::meixner_pollaczek::meixner_polynomial;
/// let m0 = meixner_polynomial(0, 1.0, 0.5, 2.0).expect("ok");
/// assert!((m0 - 1.0).abs() < 1e-14);
/// ```
pub fn meixner_polynomial(n: usize, beta: f64, c: f64, x: f64) -> SpecialResult<f64> {
    if beta <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "Meixner polynomial requires beta > 0, got {beta}"
        )));
    }
    if c <= 0.0 || c >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "Meixner polynomial requires 0 < c < 1, got c = {c}"
        )));
    }

    // Use the hypergeometric series representation:
    // M_n(x; beta, c) = 2F1(-n, -x; beta; 1 - 1/c)
    // which is a terminating series of length n+1.
    let z = 1.0 - 1.0 / c;
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;

    for k in 1..=n {
        let k_f = k as f64;
        // term_k = term_{k-1} * (-n + k - 1) * (-x + k - 1) / ((beta + k - 1) * k) * z
        let numer = (-( (n as f64) ) + (k_f - 1.0)) * (-(x) + (k_f - 1.0));
        let denom = (beta + k_f - 1.0) * k_f;
        if denom.abs() < 1e-300 {
            break;
        }
        term *= numer / denom * z;
        sum += term;
    }

    Ok(sum)
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Validate parameters for Meixner-Pollaczek polynomials.
fn validate_mp_params(lambda: f64, phi: f64) -> SpecialResult<()> {
    if lambda <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "Meixner-Pollaczek polynomial requires lambda > 0, got lambda = {lambda}"
        )));
    }
    if phi <= 0.0 || phi >= std::f64::consts::PI {
        return Err(SpecialError::DomainError(format!(
            "Meixner-Pollaczek polynomial requires 0 < phi < pi, got phi = {phi}"
        )));
    }
    Ok(())
}

/// Core recurrence evaluation (no parameter validation).
fn meixner_pollaczek_recur(n: usize, lambda: f64, phi: f64, x: f64) -> f64 {
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();

    match n {
        0 => 1.0,
        1 => 2.0 * (lambda * cos_phi + x * sin_phi),
        _ => {
            let mut p_prev = 1.0_f64;
            let mut p_curr = 2.0 * (lambda * cos_phi + x * sin_phi);

            for k in 1..n {
                let n_f = k as f64;
                let coeff_curr = 2.0 * (x * sin_phi + (n_f + lambda) * cos_phi);
                let coeff_prev = n_f + 2.0 * lambda - 1.0;
                let p_next = (coeff_curr * p_curr - coeff_prev * p_prev) / (n_f + 1.0);
                p_prev = p_curr;
                p_curr = p_next;
            }

            p_curr
        }
    }
}

/// Compute Re(log Gamma(a + ib)) = log|Gamma(a + ib)|.
///
/// Uses the asymptotic Stirling series for large |a + ib|,
/// and the functional equation Gamma(z+1) = z Gamma(z) for small arguments.
fn log_abs_gamma_complex(a: f64, b: f64) -> SpecialResult<f64> {
    // For real b = 0, use standard lgamma.
    if b == 0.0 {
        let g = gamma(a);
        if g <= 0.0 || !g.is_finite() {
            return Err(SpecialError::DomainError(format!(
                "log_abs_gamma_complex: Gamma({a}) = {g} is not positive finite"
            )));
        }
        return Ok(g.ln());
    }

    // Use the Stirling series for |z| >= 8 (with z = a + ib)
    // and the recursion Re(log Gamma(z)) = Re(log Gamma(z+n)) - sum_{k=0}^{n-1} log|z+k|
    // to shift into the large-|z| region.

    let mut a_shifted = a;
    let mut correction = 0.0_f64;

    // Shift until |a_shifted + ib| is large enough for Stirling
    while a_shifted.hypot(b) < 8.0 {
        correction += (a_shifted.hypot(b)).ln();
        a_shifted += 1.0;
    }

    // Stirling series: log Gamma(z) ~ (z - 0.5)*log(z) - z + 0.5*log(2pi)
    //                                 + 1/(12z) - 1/(360z^3) + ...
    // For z = a + ib, Re(log Gamma(z)) = Re((z-0.5)*log(z)) - a + 0.5*log(2pi) + ...
    let r2 = a_shifted * a_shifted + b * b; // |z|^2
    let r = r2.sqrt();
    let log_r = r.ln();
    let theta = b.atan2(a_shifted); // arg(z)

    // Re((z - 0.5) * log(z)):
    //   (z - 0.5) = (a-0.5) + ib
    //   log(z) = log(r) + i*theta
    //   product: Re = (a-0.5)*log(r) - b*theta
    let re_stirling = (a_shifted - 0.5) * log_r
        - b * theta
        - a_shifted
        + 0.5 * (2.0 * std::f64::consts::PI).ln();

    // Add first Stirling correction term: Re(1/(12z)) = a/(12 r^2)
    let stirling_corr1 = a_shifted / (12.0 * r2);

    // Second Stirling term: Re(-1/(360z^3))
    // z^3 = (a^3 - 3ab^2) + i(3a^2*b - b^3)
    // 1/z^3: Re = (a^3 - 3ab^2) / |z|^6
    let re_z3 = a_shifted * a_shifted * a_shifted - 3.0 * a_shifted * b * b;
    let stirling_corr2 = -re_z3 / (360.0 * r2 * r2 * r2);

    let log_abs_gamma_shifted = re_stirling + stirling_corr1 + stirling_corr2;

    // Remove the correction from the shift
    Ok(log_abs_gamma_shifted - correction)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mp_degree_zero() {
        // P_0 = 1 for any valid parameters and x
        let phi = std::f64::consts::PI / 3.0;
        for &x in &[-2.0, 0.0, 1.5, 100.0] {
            let val = meixner_pollaczek(0, 1.0, phi, x).expect("ok");
            assert!((val - 1.0).abs() < 1e-14, "P_0 should be 1, got {val}");
        }
    }

    #[test]
    fn test_mp_degree_one_at_half_pi() {
        // At phi = pi/2: P_1^{(lambda)}(x; pi/2) = 2*(lambda*0 + x*1) = 2x
        let phi = std::f64::consts::FRAC_PI_2;
        for &x in &[0.0, 1.0, -1.5, 3.7] {
            let val = meixner_pollaczek(1, 2.0, phi, x).expect("ok");
            let expected = 2.0 * x;
            assert!(
                (val - expected).abs() < 1e-12,
                "P_1 at phi=pi/2 should be {expected}, got {val}"
            );
        }
    }

    #[test]
    fn test_mp_array_consistency() {
        // meixner_pollaczek_array should agree with repeated meixner_pollaczek calls
        let lambda = 0.5;
        let phi = std::f64::consts::PI / 4.0;
        let x = 1.3;
        let n_max = 6;
        let arr = meixner_pollaczek_array(n_max, lambda, phi, x).expect("ok");
        assert_eq!(arr.len(), n_max + 1);
        for n in 0..=n_max {
            let scalar = meixner_pollaczek(n, lambda, phi, x).expect("ok");
            assert!(
                (arr[n] - scalar).abs() < 1e-12,
                "degree {n}: array={} scalar={scalar}",
                arr[n]
            );
        }
    }

    #[test]
    fn test_mp_domain_errors() {
        // lambda <= 0
        assert!(meixner_pollaczek(2, 0.0, 1.0, 0.0).is_err());
        assert!(meixner_pollaczek(2, -1.0, 1.0, 0.0).is_err());
        // phi out of (0, pi)
        assert!(meixner_pollaczek(2, 1.0, 0.0, 0.0).is_err());
        assert!(meixner_pollaczek(2, 1.0, std::f64::consts::PI, 0.0).is_err());
        assert!(meixner_pollaczek(2, 1.0, -0.1, 0.0).is_err());
        assert!(meixner_pollaczek(2, 1.0, 4.0, 0.0).is_err());
    }

    #[test]
    fn test_mp_recurrence_consistency() {
        // Verify the recurrence relation manually for n=2:
        // 2*P_2 = 2*(x*sin(phi) + (1+lambda)*cos(phi)) * P_1 - 2*lambda * P_0
        let lambda = 1.5;
        let phi = std::f64::consts::PI / 3.0;
        let x = 0.8;
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();

        let p0 = meixner_pollaczek(0, lambda, phi, x).expect("ok");
        let p1 = meixner_pollaczek(1, lambda, phi, x).expect("ok");
        let p2 = meixner_pollaczek(2, lambda, phi, x).expect("ok");

        // For n=1 (k=1):
        // (1+1)*P_2 = 2*(x*sin + (1+lambda)*cos)*P_1 - (1+2*lambda-1)*P_0
        let rhs = (2.0 * (x * sin_phi + (1.0 + lambda) * cos_phi) * p1
            - (1.0 + 2.0 * lambda - 1.0) * p0)
            / 2.0;
        assert!(
            (p2 - rhs).abs() < 1e-12,
            "Recurrence check: P_2 = {p2}, rhs = {rhs}"
        );
    }

    #[test]
    fn test_mp_weight_positive() {
        // Weight function must be positive
        let lambda = 1.0;
        let phi = std::f64::consts::PI / 3.0;
        for &x in &[-3.0, -1.0, 0.0, 1.0, 2.5] {
            let w = meixner_pollaczek_weight(x, lambda, phi).expect("ok");
            assert!(w > 0.0, "Weight should be positive at x={x}, got {w}");
        }
    }

    #[test]
    fn test_charlier_degree_zero() {
        // C_0 = 1
        for &x in &[0.0, 1.0, 5.0] {
            let val = charlier_polynomial(0, 2.0, x).expect("ok");
            assert!((val - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_charlier_degree_one() {
        // C_1(x; a) = 1 - x/a
        let a = 3.0;
        for &x in &[0.0, 1.0, 3.0, 6.0] {
            let val = charlier_polynomial(1, a, x).expect("ok");
            let expected = 1.0 - x / a;
            assert!(
                (val - expected).abs() < 1e-14,
                "C_1({x}; {a}) should be {expected}, got {val}"
            );
        }
    }

    #[test]
    fn test_charlier_recurrence() {
        // Verify: a*C_2 = (a - 1 - x)*C_1 - 1*C_0
        let a = 2.0;
        let x = 1.5;
        let c0 = charlier_polynomial(0, a, x).expect("ok");
        let c1 = charlier_polynomial(1, a, x).expect("ok");
        let c2 = charlier_polynomial(2, a, x).expect("ok");
        let rhs = ((a - 1.0 - x) * c1 - c0) / a;
        assert!(
            (c2 - rhs).abs() < 1e-12,
            "Charlier recurrence: C_2={c2}, rhs={rhs}"
        );
    }

    #[test]
    fn test_charlier_array_consistency() {
        let a = 2.5;
        let x = 2.0;
        let n_max = 5;
        let arr = charlier_polynomial_array(n_max, a, x).expect("ok");
        assert_eq!(arr.len(), n_max + 1);
        for n in 0..=n_max {
            let scalar = charlier_polynomial(n, a, x).expect("ok");
            assert!(
                (arr[n] - scalar).abs() < 1e-12,
                "Charlier degree {n}: array={} scalar={scalar}",
                arr[n]
            );
        }
    }

    #[test]
    fn test_charlier_domain_error() {
        assert!(charlier_polynomial(2, 0.0, 1.0).is_err());
        assert!(charlier_polynomial(2, -1.0, 1.0).is_err());
    }

    #[test]
    fn test_meixner_polynomial_n0() {
        let m0 = meixner_polynomial(0, 1.0, 0.5, 2.0).expect("ok");
        assert!((m0 - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_meixner_polynomial_n1() {
        // M_1(x; beta, c) = 2F1(-1, -x; beta; 1-1/c)
        //                 = 1 + (-1)*(-x)/(beta*1)*(1-1/c)
        //                 = 1 + x*(1-1/c)/beta
        let beta = 2.0;
        let c = 0.5;
        let x = 3.0;
        let m1 = meixner_polynomial(1, beta, c, x).expect("ok");
        let expected = 1.0 + x * (1.0 - 1.0 / c) / beta;
        assert!(
            (m1 - expected).abs() < 1e-12,
            "M_1 should be {expected}, got {m1}"
        );
    }

    #[test]
    fn test_meixner_polynomial_domain_error() {
        assert!(meixner_polynomial(2, 0.0, 0.5, 1.0).is_err());
        assert!(meixner_polynomial(2, 1.0, 0.0, 1.0).is_err());
        assert!(meixner_polynomial(2, 1.0, 1.0, 1.0).is_err());
        assert!(meixner_polynomial(2, 1.0, -0.1, 1.0).is_err());
    }

    #[test]
    fn test_log_abs_gamma_complex_real_case() {
        // For b=0, should match lgamma
        use crate::gamma::gammaln;
        let a = 3.5;
        let result = log_abs_gamma_complex(a, 0.0).expect("ok");
        let expected = gammaln(a);
        assert!(
            (result - expected).abs() < 1e-10,
            "log|Gamma({a})| should match lgamma: {result} vs {expected}"
        );
    }
}
