//! Clausen function
//!
//! The Clausen function Cl_2(x) is defined as:
//! Cl_2(x) = -integral_0^x ln|2 sin(t/2)| dt = sum_{k=1}^inf sin(kx)/k^2
//!
//! It is related to the dilogarithm and appears in various areas of
//! mathematical physics, including quantum field theory and lattice
//! calculations.
//!
//! ## Properties
//!
//! 1. **Periodicity**: Cl_2(x + 2*pi) = Cl_2(x)
//! 2. **Odd symmetry**: Cl_2(-x) = -Cl_2(x)
//! 3. **Maximum**: Cl_2(pi/3) = 1.01494... (Catalan-like constant)
//! 4. **Zeros**: Cl_2(k*pi) = 0 for all integers k
//! 5. **Relation to dilogarithm**: Cl_2(x) = Im(Li_2(e^{ix}))
//!
//! ## References
//!
//! 1. Clausen, T. (1832). "Ueber die Function sin(φ) + sin(2φ)/2^2 + sin(3φ)/3^2 + etc."
//! 2. Lewin, L. (1981). Polylogarithms and Associated Functions.
//! 3. NIST Digital Library of Mathematical Functions, Ch. 25.12.

use crate::error::{SpecialError, SpecialResult};
use std::f64::consts::PI;

/// Clausen function Cl_2(x).
///
/// Defined as: Cl_2(x) = -integral_0^x ln|2 sin(t/2)| dt
///           = sum_{k=1}^inf sin(kx)/k^2
///
/// # Arguments
///
/// * `x` - Real argument
///
/// # Returns
///
/// * `SpecialResult<f64>` - Value of Cl_2(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::clausen;
///
/// // Cl_2(pi/3) is approximately 1.01494
/// let val = clausen(std::f64::consts::PI / 3.0).expect("clausen failed");
/// assert!((val - 1.01494).abs() < 1e-4);
///
/// // Cl_2(k*pi) = 0
/// let val_pi = clausen(std::f64::consts::PI).expect("clausen failed");
/// assert!(val_pi.abs() < 1e-10);
/// ```
pub fn clausen(x: f64) -> SpecialResult<f64> {
    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to clausen".to_string(),
        ));
    }

    if !x.is_finite() {
        return Err(SpecialError::DomainError(
            "Infinite input to clausen".to_string(),
        ));
    }

    // Reduce x to [0, 2*pi) using periodicity
    let two_pi = 2.0 * PI;
    let mut theta = x % two_pi;
    if theta < 0.0 {
        theta += two_pi;
    }

    // Handle the odd symmetry: Cl_2(-x) = -Cl_2(x)
    // After reduction to [0, 2pi), we use Cl_2(2pi - x) = -Cl_2(x)
    let (sign, theta) = if theta > PI {
        (-1.0, two_pi - theta)
    } else {
        (1.0, theta)
    };

    // Now theta is in [0, pi]

    // Special case: theta = 0 or theta = pi
    if theta.abs() < 1e-15 || (theta - PI).abs() < 1e-15 {
        return Ok(0.0);
    }

    // Use the Bernoulli-number expansion which is valid for all theta in (0, 2*pi).
    // This converges well for the entire range.
    Ok(sign * clausen_series(theta))
}

/// Clausen function using Bernoulli-number expansion valid for theta in (0, 2*pi).
///
/// Uses the exact expansion:
/// Cl_2(theta) = theta*(1 - ln(theta)) + sum_{k=1}^N |B_{2k}| / (2k*(2k)!*(2k+1)) * theta^{2k+1}
///
/// This is based on the integral definition and the expansion of ln(2*sin(t/2)):
/// ln(2*sin(t/2)) = ln(t) - sum_{k=1}^inf |B_{2k}| / (2k*(2k)!) * t^{2k}
///
/// The expansion converges for 0 < theta < 2*pi with enough terms.
fn clausen_series(theta: f64) -> f64 {
    let ln_theta = theta.ln();

    let mut result = theta * (1.0 - ln_theta);

    // Bernoulli numbers |B_{2k}| for k = 1..20
    // These are exact rational values.
    let bernoulli_abs: [f64; 20] = [
        1.0 / 6.0,                          // |B_2|
        1.0 / 30.0,                         // |B_4|
        1.0 / 42.0,                         // |B_6|
        1.0 / 30.0,                         // |B_8|
        5.0 / 66.0,                         // |B_10|
        691.0 / 2730.0,                     // |B_12|
        7.0 / 6.0,                          // |B_14|
        3617.0 / 510.0,                     // |B_16|
        43867.0 / 798.0,                    // |B_18|
        174611.0 / 330.0,                   // |B_20|
        854513.0 / 138.0,                   // |B_22|
        236364091.0 / 2730.0,               // |B_24|
        8553103.0 / 6.0,                    // |B_26|
        23749461029.0 / 870.0,              // |B_28|
        8615841276005.0 / 14322.0,          // |B_30|
        7709321041217.0 / 510.0,            // |B_32|
        2577687858367.0 / 6.0,              // |B_34|
        26315271553053477373.0 / 1919190.0, // |B_36|
        2929993913841559.0 / 6.0,           // |B_38|
        261082718496449122051.0 / 13530.0,  // |B_40|
    ];

    let theta_sq = theta * theta;
    let mut theta_power = theta * theta_sq; // theta^3

    // Precompute factorials
    let mut factorial_2k = 1.0_f64; // (2k)! starting with 0! = 1

    for (idx, &bk) in bernoulli_abs.iter().enumerate() {
        let k = idx + 1;
        let two_k = 2 * k;

        // Update factorial: (2k)! = (2k-2)! * (2k-1) * (2k)
        factorial_2k *= (two_k - 1) as f64 * two_k as f64;

        let coeff = bk / (two_k as f64 * factorial_2k * (two_k + 1) as f64);
        let term = coeff * theta_power;
        result += term;

        // Check convergence
        if term.abs() < 1e-16 * result.abs() && k > 3 {
            break;
        }

        theta_power *= theta_sq;

        // Guard against overflow in factorial
        if factorial_2k.is_infinite() {
            break;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_clausen_at_zero() {
        let val = clausen(0.0).expect("clausen(0) failed");
        assert_relative_eq!(val, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_clausen_at_pi() {
        let val = clausen(PI).expect("clausen(pi) failed");
        assert_relative_eq!(val, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_clausen_at_2pi() {
        let val = clausen(2.0 * PI).expect("clausen(2pi) failed");
        assert_relative_eq!(val, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_clausen_maximum() {
        // Cl_2(pi/3) ~ 1.01494160640965362502... (known high-precision value)
        let val = clausen(PI / 3.0).expect("clausen(pi/3) failed");
        assert_relative_eq!(val, 1.01494160640965, epsilon = 1e-6);
    }

    #[test]
    fn test_clausen_odd_symmetry() {
        // Cl_2(-x) = -Cl_2(x)
        let x = 1.5;
        let val_pos = clausen(x).expect("failed");
        let val_neg = clausen(-x).expect("failed");
        assert_relative_eq!(val_pos, -val_neg, epsilon = 1e-10);
    }

    #[test]
    fn test_clausen_periodicity() {
        // Cl_2(x + 2*pi) = Cl_2(x)
        let x = 1.2;
        let val1 = clausen(x).expect("failed");
        let val2 = clausen(x + 2.0 * PI).expect("failed");
        assert_relative_eq!(val1, val2, epsilon = 1e-10);
    }

    #[test]
    fn test_clausen_at_pi_over_2() {
        // Cl_2(pi/2) = Catalan's constant G ~ 0.915966...
        let val = clausen(PI / 2.0).expect("clausen(pi/2) failed");
        assert_relative_eq!(val, 0.915965594177, epsilon = 1e-5);
    }

    #[test]
    fn test_clausen_small_values() {
        // For small x, Cl_2(x) ~ x * (1 - ln|x|)
        let x = 0.01;
        let val = clausen(x).expect("failed");
        let approx = x * (1.0 - x.ln());
        assert_relative_eq!(val, approx, epsilon = 1e-3);
    }

    #[test]
    fn test_clausen_nan() {
        assert!(clausen(f64::NAN).is_err());
    }

    #[test]
    fn test_clausen_infinity() {
        assert!(clausen(f64::INFINITY).is_err());
    }
}
