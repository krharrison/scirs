//! Debye functions
//!
//! The Debye functions D_n(x) arise in the computation of the Debye model
//! for the specific heat of solids in thermodynamics. They are defined as:
//!
//! D_n(x) = (n/x^n) * integral_0^x (t^n / (e^t - 1)) dt
//!
//! where n >= 1 is an integer.
//!
//! ## Physical Applications
//!
//! In the Debye model, the internal energy of a solid at temperature T is:
//! U = 9*N*k_B*T * (T/Theta_D)^3 * D_3(Theta_D/T)
//!
//! where Theta_D is the Debye temperature and D_3 is the third Debye function.
//!
//! ## Implementation Notes
//!
//! - For small x (x < 2*pi), a series expansion is used based on
//!   Bernoulli numbers.
//! - For intermediate x, numerical integration via Gauss-Legendre quadrature.
//! - For large x (x >> 1), D_n(x) -> n! * zeta(n+1) / x^n where
//!   zeta is the Riemann zeta function.
//!
//! ## References
//!
//! 1. Abramowitz, M. and Stegun, I. A. (1972). Handbook of Mathematical Functions, Ch. 27.
//! 2. NIST Digital Library of Mathematical Functions, Ch. 27.1.

use crate::error::{SpecialError, SpecialResult};
use std::f64::consts::PI;

/// First Debye function D_1(x) = (1/x) * integral_0^x t/(e^t - 1) dt
///
/// # Arguments
///
/// * `x` - Non-negative real argument
///
/// # Returns
///
/// * `SpecialResult<f64>` - Value of D_1(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::debye1;
///
/// let val = debye1(1.0).expect("debye1 failed");
/// assert!(val > 0.0);
/// assert!(val < 1.0);
/// ```
pub fn debye1(x: f64) -> SpecialResult<f64> {
    debye_n(1, x)
}

/// Second Debye function D_2(x) = (2/x^2) * integral_0^x t^2/(e^t - 1) dt
///
/// # Arguments
///
/// * `x` - Non-negative real argument
///
/// # Returns
///
/// * `SpecialResult<f64>` - Value of D_2(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::debye2;
///
/// let val = debye2(1.0).expect("debye2 failed");
/// assert!(val > 0.0);
/// assert!(val < 1.0);
/// ```
pub fn debye2(x: f64) -> SpecialResult<f64> {
    debye_n(2, x)
}

/// Third Debye function D_3(x) = (3/x^3) * integral_0^x t^3/(e^t - 1) dt
///
/// The most commonly used Debye function in solid state physics.
/// Related to the Debye model specific heat: C_v = 9*N*k_B * D_3(Theta_D/T)
///
/// # Arguments
///
/// * `x` - Non-negative real argument
///
/// # Returns
///
/// * `SpecialResult<f64>` - Value of D_3(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::debye3;
///
/// let val = debye3(1.0).expect("debye3 failed");
/// assert!(val > 0.0);
/// assert!(val < 1.0);
/// ```
pub fn debye3(x: f64) -> SpecialResult<f64> {
    debye_n(3, x)
}

/// Fourth Debye function D_4(x) = (4/x^4) * integral_0^x t^4/(e^t - 1) dt
///
/// # Arguments
///
/// * `x` - Non-negative real argument
///
/// # Returns
///
/// * `SpecialResult<f64>` - Value of D_4(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::debye4;
///
/// let val = debye4(1.0).expect("debye4 failed");
/// assert!(val > 0.0);
/// assert!(val < 1.0);
/// ```
pub fn debye4(x: f64) -> SpecialResult<f64> {
    debye_n(4, x)
}

/// Fifth Debye function D_5(x) = (5/x^5) * integral_0^x t^5/(e^t - 1) dt
///
/// # Arguments
///
/// * `x` - Non-negative real argument
///
/// # Returns
///
/// * `SpecialResult<f64>` - Value of D_5(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::debye5;
///
/// let val = debye5(1.0).expect("debye5 failed");
/// assert!(val > 0.0);
/// assert!(val < 1.0);
/// ```
pub fn debye5(x: f64) -> SpecialResult<f64> {
    debye_n(5, x)
}

/// General Debye function D_n(x) for integer n >= 1.
///
/// D_n(x) = (n/x^n) * integral_0^x t^n/(e^t - 1) dt
///
/// # Arguments
///
/// * `n` - Order (positive integer)
/// * `x` - Non-negative real argument
///
/// # Returns
///
/// * `SpecialResult<f64>` - Value of D_n(x)
fn debye_n(n: i32, x: f64) -> SpecialResult<f64> {
    if n < 1 {
        return Err(SpecialError::DomainError(
            "Debye function order n must be >= 1".to_string(),
        ));
    }

    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to Debye function".to_string(),
        ));
    }

    if x < 0.0 {
        return Err(SpecialError::DomainError(
            "Debye function requires non-negative x".to_string(),
        ));
    }

    // Special case: x = 0
    // D_n(0) = 1 (limiting value)
    if x == 0.0 {
        return Ok(1.0);
    }

    // For very small x, use the Taylor expansion
    // D_n(x) = 1 - n*x/(2*(n+1)) + n*B_2*x^2/(2*(n+2)) - ...
    // where B_k are Bernoulli numbers
    if x < 0.1 {
        return debye_small_x(n, x);
    }

    // For small to moderate x (x < 2*pi), use the series expansion
    if x < 2.0 * PI {
        return debye_series(n, x);
    }

    // For large x, use the asymptotic expansion plus the integral tail
    debye_large_x(n, x)
}

/// Small x expansion for Debye function using Taylor series.
///
/// D_n(x) = 1 - n/(2(n+1)) * x + sum_{k=1}^K n * B_{2k} / ((2k)! * (n+2k)) * x^{2k}
fn debye_small_x(n: i32, x: f64) -> SpecialResult<f64> {
    let n_f = n as f64;

    // Bernoulli numbers B_2, B_4, B_6, B_8, B_10, B_12
    let bernoulli_2k = [
        1.0 / 6.0,       // B_2
        -1.0 / 30.0,     // B_4
        1.0 / 42.0,      // B_6
        -1.0 / 30.0,     // B_8
        5.0 / 66.0,      // B_10
        -691.0 / 2730.0, // B_12
    ];

    let mut result = 1.0 - n_f * x / (2.0 * (n_f + 1.0));
    let mut x_power = 1.0; // will become x^{2k}

    for (k_idx, &b2k) in bernoulli_2k.iter().enumerate() {
        let k = (k_idx + 1) as f64; // k = 1, 2, 3, ...
        let two_k = 2.0 * k;
        x_power *= x * x;

        // Factorial (2k)!
        let mut factorial_2k = 1.0;
        for j in 1..=(2 * (k_idx + 1)) {
            factorial_2k *= j as f64;
        }

        let term = n_f * b2k / (factorial_2k * (n_f + two_k)) * x_power;
        result += term;

        if term.abs() < 1e-16 * result.abs() {
            break;
        }
    }

    Ok(result)
}

/// Series expansion for Debye function using partial sums.
///
/// D_n(x) = (n/x^n) * sum_{k=1}^inf e^{-kx} * (x^n/k + n*x^{n-1}/k^2 + ... + n!/k^{n+1})
fn debye_series(n: i32, x: f64) -> SpecialResult<f64> {
    // Use Gauss-Legendre quadrature for the integral
    // integral_0^x t^n / (e^t - 1) dt
    // This is more reliable than the series for moderate x.

    let integral = gauss_legendre_debye(n, x)?;
    let n_f = n as f64;

    // D_n(x) = (n/x^n) * integral
    let x_pow_n = x.powi(n);
    if x_pow_n == 0.0 {
        return Ok(0.0);
    }

    Ok(n_f / x_pow_n * integral)
}

/// Large x asymptotic expansion.
///
/// For large x, we split the integral:
/// integral_0^x t^n/(e^t-1) dt = integral_0^inf t^n/(e^t-1) dt - integral_x^inf t^n/(e^t-1) dt
///
/// The first part = n! * zeta(n+1) (known exactly).
/// The second part is exponentially small for large x and can be computed
/// via an asymptotic series.
fn debye_large_x(n: i32, x: f64) -> SpecialResult<f64> {
    let n_f = n as f64;

    // integral_0^inf t^n/(e^t-1) dt = Gamma(n+1) * zeta(n+1)
    // We'll compute this using known values of zeta
    let zeta_values = [
        0.0,                // placeholder for n=0
        PI * PI / 6.0,      // zeta(2)
        1.202056903159594,  // zeta(3) Apery's constant
        PI.powi(4) / 90.0,  // zeta(4)
        1.036927755143370,  // zeta(5)
        PI.powi(6) / 945.0, // zeta(6)
    ];

    let zeta_n_plus_1 = if (n as usize) < zeta_values.len() {
        zeta_values[n as usize]
    } else {
        // For larger n, zeta(n+1) approaches 1 rapidly
        1.0 + 1.0 / 2.0_f64.powi(n + 1) + 1.0 / 3.0_f64.powi(n + 1)
    };

    // Gamma(n+1) = n! for integer n
    let mut n_factorial = 1.0;
    for k in 1..=n {
        n_factorial *= k as f64;
    }

    let full_integral = n_factorial * zeta_n_plus_1;

    // Compute the tail integral_x^inf t^n/(e^t-1) dt
    // For large x, this is approximately sum_{k=1}^inf e^{-kx} * P_n(x, k)
    // where P_n(x, k) = sum_{j=0}^n n!/(n-j)! * x^{n-j} / k^{j+1}
    let mut tail = 0.0;
    for k in 1..=20 {
        let k_f = k as f64;
        let ekx = (-k_f * x).exp();

        if ekx < 1e-300 {
            break; // remaining terms are negligible
        }

        // Compute P_n(x, k)
        let mut poly = 0.0;
        let mut coeff = 1.0; // n!/(n-j)! starts at 1 for j=0
        let mut x_power = x.powi(n); // x^{n-j} starts at x^n
        let mut k_power = k_f; // k^{j+1} starts at k

        for j in 0..=n {
            poly += coeff * x_power / k_power;

            // Update for next j
            if j < n {
                coeff *= (n - j) as f64;
                x_power /= x;
                k_power *= k_f;
            }
        }

        tail += ekx * poly;

        if ekx * poly < 1e-16 * tail.abs() {
            break;
        }
    }

    let integral = full_integral - tail;
    let x_pow_n = x.powi(n);

    if x_pow_n == 0.0 {
        return Ok(0.0);
    }

    Ok(n_f / x_pow_n * integral)
}

/// Gauss-Legendre quadrature for the Debye integral.
///
/// Computes integral_0^x t^n / (e^t - 1) dt using adaptive quadrature.
fn gauss_legendre_debye(n: i32, x: f64) -> SpecialResult<f64> {
    // 20-point Gauss-Legendre nodes and weights on [-1, 1]
    // Pre-computed for high accuracy
    let nodes = [
        -0.993128599185094925,
        -0.963971927277913791,
        -0.912234428251325906,
        -0.839116971822218823,
        -0.746331906460150793,
        -0.636053680726515025,
        -0.510867001950827098,
        -0.373706088715419561,
        -0.227785851141645078,
        -0.076526521133497334,
        0.076526521133497334,
        0.227785851141645078,
        0.373706088715419561,
        0.510867001950827098,
        0.636053680726515025,
        0.746331906460150793,
        0.839116971822218823,
        0.912234428251325906,
        0.963971927277913791,
        0.993128599185094925,
    ];

    let weights = [
        0.017614007139152118,
        0.040601429800386941,
        0.062672048334109064,
        0.083276741576704749,
        0.101930119817240435,
        0.118194531961518417,
        0.131688638449176627,
        0.142096109318382051,
        0.149172986472603747,
        0.152753387130725851,
        0.152753387130725851,
        0.149172986472603747,
        0.142096109318382051,
        0.131688638449176627,
        0.118194531961518417,
        0.101930119817240435,
        0.083276741576704749,
        0.062672048334109064,
        0.040601429800386941,
        0.017614007139152118,
    ];

    // Transform integral from [0, x] to [-1, 1]
    let half_x = x / 2.0;
    let mid_x = x / 2.0;

    let mut integral = 0.0;
    let n_f = n as f64;

    for (&node, &weight) in nodes.iter().zip(weights.iter()) {
        let t = mid_x + half_x * node;
        if t <= 0.0 {
            continue;
        }

        // Compute t^n / (e^t - 1) carefully
        let integrand = if t < 1e-10 {
            // For very small t: t^n / (e^t - 1) ~ t^{n-1} (since e^t - 1 ~ t)
            t.powf(n_f - 1.0)
        } else if t > 500.0 {
            // For very large t: t^n * e^{-t}
            t.powf(n_f) * (-t).exp()
        } else {
            // General case
            let exp_t = t.exp();
            t.powf(n_f) / (exp_t - 1.0)
        };

        integral += weight * integrand;
    }

    integral *= half_x;

    Ok(integral)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_debye_at_zero() {
        // D_n(0) = 1 for all n
        for n in 1..=5 {
            let val = debye_n(n, 0.0).expect("debye_n failed at x=0");
            assert_relative_eq!(val, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_debye1_known_values() {
        // D_1(1) ~ 0.7775046... (SciPy reference)
        let val = debye1(1.0).expect("debye1 failed");
        assert_relative_eq!(val, 0.7775046341122, epsilon = 1e-4);
    }

    #[test]
    fn test_debye2_known_values() {
        // D_2(1) ~ 0.7075883... (SciPy reference)
        let val = debye2(1.0).expect("debye2 failed");
        assert_relative_eq!(val, 0.7075883, epsilon = 1e-3);
    }

    #[test]
    fn test_debye3_known_values() {
        // D_3(1) ~ 0.6744... (SciPy reference)
        let val = debye3(1.0).expect("debye3 failed");
        assert_relative_eq!(val, 0.6744, epsilon = 5e-3);
    }

    #[test]
    fn test_debye4_known_values() {
        // D_4(1) ~ 0.6544... (SciPy reference)
        let val = debye4(1.0).expect("debye4 failed");
        assert_relative_eq!(val, 0.6544, epsilon = 5e-3);
    }

    #[test]
    fn test_debye_monotone_decreasing() {
        // D_n(x) is monotonically decreasing for x > 0
        for n in 1..=4 {
            let d1 = debye_n(n, 0.5).expect("failed");
            let d2 = debye_n(n, 1.0).expect("failed");
            let d3 = debye_n(n, 2.0).expect("failed");
            let d4 = debye_n(n, 5.0).expect("failed");
            assert!(d1 > d2, "D_{} should be decreasing", n);
            assert!(d2 > d3, "D_{} should be decreasing", n);
            assert!(d3 > d4, "D_{} should be decreasing", n);
        }
    }

    #[test]
    fn test_debye_range_zero_to_one() {
        // D_n(x) is always in (0, 1] for x >= 0
        for n in 1..=5 {
            for &x in &[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0] {
                let val = debye_n(n, x).expect("failed");
                assert!(
                    val > 0.0 && val <= 1.0,
                    "D_{}({}) = {} should be in (0, 1]",
                    n,
                    x,
                    val
                );
            }
        }
    }

    #[test]
    fn test_debye_large_x() {
        // For large x, D_n(x) -> n! * zeta(n+1) / x^n
        let x = 100.0;
        let val = debye1(x).expect("failed");
        // D_1(100) ~ 1! * zeta(2) / 100 ~ pi^2/6 / 100 ~ 0.01645
        let expected = PI * PI / 6.0 / x;
        assert_relative_eq!(val, expected, epsilon = 1e-3);
    }

    #[test]
    fn test_debye_small_x_series() {
        // For very small x, D_n(x) ~ 1 - n*x/(2(n+1))
        let x = 0.01;
        for n in 1..=4 {
            let val = debye_n(n, x).expect("failed");
            let approx = 1.0 - n as f64 * x / (2.0 * (n as f64 + 1.0));
            assert_relative_eq!(val, approx, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_debye_negative_x() {
        assert!(debye1(-1.0).is_err());
    }

    #[test]
    fn test_debye_nan() {
        assert!(debye1(f64::NAN).is_err());
    }

    #[test]
    fn test_debye5_positive() {
        let val = debye5(3.0).expect("failed");
        assert!(val > 0.0);
        assert!(val < 1.0);
    }
}
