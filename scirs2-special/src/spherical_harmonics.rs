//! Spherical Harmonics
//!
//! This module provides implementations of spherical harmonic functions Y_l^m(θ, φ)
//! that are important in quantum mechanics, physical chemistry, and solving
//! differential equations in spherical coordinates.

use crate::error::SpecialResult;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::f64;
use std::f64::consts::PI;
use std::fmt::Debug;

/// Computes the value of the real spherical harmonic Y_l^m(θ, φ) function.
///
/// The spherical harmonics Y_l^m(θ, φ) form a complete orthogonal basis for functions
/// defined on the sphere. They appear extensively in solving three-dimensional
/// partial differential equations in spherical coordinates, especially in quantum physics.
///
/// This implementation returns the real form of the spherical harmonic, which is often
/// more convenient for practical applications.
///
/// # Arguments
///
/// * `l` - Degree (non-negative integer)
/// * `m` - Order (integer with |m| ≤ l)
/// * `theta` - Polar angle (in radians, 0 ≤ θ ≤ π)
/// * `phi` - Azimuthal angle (in radians, 0 ≤ φ < 2π)
///
/// # Returns
///
/// * Value of the real spherical harmonic Y_l^m(θ, φ)
///
/// # Examples
///
/// ```
/// use scirs2_special::sph_harm;
/// use std::f64::consts::PI;
///
/// // Y₀⁰(θ, φ) = 1/(2√π)
/// let y00: f64 = sph_harm(0, 0, PI/2.0, 0.0).expect("Operation failed");
/// assert!((y00 - 0.5/f64::sqrt(PI)).abs() < 1e-10);
///
/// // Y₁⁰(θ, φ) = √(3/4π) cos(θ)
/// let y10: f64 = sph_harm(1, 0, PI/4.0, 0.0).expect("Operation failed");
/// let expected = f64::sqrt(3.0/(4.0*PI)) * f64::cos(PI/4.0);
/// assert!((y10 - expected).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn sph_harm<F>(l: usize, m: i32, theta: F, phi: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug,
{
    // Validate that |m| <= l
    let m_abs = m.unsigned_abs() as usize;
    if m_abs > l {
        return Ok(F::zero()); // Y_l^m = 0 if |m| > l
    }

    let cos_theta = theta.cos();
    let x = cos_theta.to_f64().unwrap_or(0.0);

    // Compute P_l^{|m|}(cos theta) using standard recurrence with Condon-Shortley phase
    let plm = associated_legendre_for_sph(l, m_abs, x);
    let plm_f = F::from(plm).unwrap_or(F::zero());

    // Compute normalization: K_l^m = sqrt((2l+1)/(4pi) * (l-|m|)!/(l+|m|)!)
    let mut fact_ratio = 1.0_f64;
    if m_abs > 0 {
        for i in (l - m_abs + 1)..=(l + m_abs) {
            fact_ratio /= i as f64;
        }
    }
    let norm = ((2.0 * l as f64 + 1.0) / (4.0 * std::f64::consts::PI) * fact_ratio).sqrt();
    let norm_f = F::from(norm).unwrap_or(F::zero());

    // Compute angular part for real spherical harmonics
    let angular_part: F;
    if m == 0 {
        angular_part = F::one();
    } else {
        let sqrt2 = F::from(std::f64::consts::SQRT_2).unwrap_or(F::one());
        let m_f = F::from(m_abs).unwrap_or(F::one());
        let m_phi = m_f * phi;

        if m > 0 {
            angular_part = sqrt2 * m_phi.cos();
        } else {
            angular_part = sqrt2 * m_phi.sin();
        }
    }

    Ok(norm_f * plm_f * angular_part)
}

/// Compute the associated Legendre function P_l^m(x) WITHOUT Condon-Shortley phase.
///
/// Uses upward recurrence starting from P_m^m.
/// The Condon-Shortley phase (-1)^m is NOT included, i.e.,
/// P_m^m(x) = (2m-1)!! * (1-x^2)^(m/2) (always non-negative at m=l).
///
/// This convention is used for real spherical harmonics where the standard
/// definition gives Y_1^1(theta, phi) = sqrt(3/(4pi)) * sin(theta) * cos(phi) > 0.
fn associated_legendre_for_sph(l: usize, m: usize, x: f64) -> f64 {
    if m > l {
        return 0.0;
    }

    // Start with P_m^m = (2m-1)!! * (1-x^2)^(m/2)
    // Note: no (-1)^m factor (Condon-Shortley phase excluded)
    let mut pmm = 1.0;
    if m > 0 {
        let somx2 = ((1.0 - x) * (1.0 + x)).sqrt();
        let mut fact = 1.0;
        for _i in 1..=m {
            pmm *= fact * somx2;
            fact += 2.0;
        }
    }

    if l == m {
        return pmm;
    }

    // Compute P_{m+1}^m = x * (2m+1) * P_m^m
    let mut pmmp1 = x * (2 * m + 1) as f64 * pmm;

    if l == m + 1 {
        return pmmp1;
    }

    // Upward recurrence: (l-m)*P_l^m = (2l-1)*x*P_{l-1}^m - (l+m-1)*P_{l-2}^m
    let mut result = 0.0;
    for ll in (m + 2)..=l {
        result = (x * (2 * ll - 1) as f64 * pmmp1 - (ll + m - 1) as f64 * pmm) / (ll - m) as f64;
        pmm = pmmp1;
        pmmp1 = result;
    }

    result
}

/// Computes the value of the complex spherical harmonic Y_l^m(θ, φ) function.
///
/// The complex spherical harmonics are the conventional form found in quantum mechanics.
/// They are eigenfunctions of the angular momentum operators.
///
/// # Arguments
///
/// * `l` - Degree (non-negative integer)
/// * `m` - Order (integer with |m| ≤ l)
/// * `theta` - Polar angle (in radians, 0 ≤ θ ≤ π)
/// * `phi` - Azimuthal angle (in radians, 0 ≤ φ < 2π)
///
/// # Returns
///
/// * Real part of the complex spherical harmonic Y_l^m(θ, φ)
/// * Imaginary part of the complex spherical harmonic Y_l^m(θ, φ)
///
/// # Examples
///
/// ```
/// use scirs2_special::sph_harm_complex;
/// use std::f64::consts::PI;
///
/// // Y₀⁰(θ, φ) = 1/(2√π)
/// let (re, im): (f64, f64) = sph_harm_complex(0, 0, PI/2.0, 0.0).expect("Operation failed");
/// assert!((re - 0.5/f64::sqrt(PI)).abs() < 1e-10);
/// assert!(im.abs() < 1e-10);
///
/// // Y₁¹(θ, φ) = -√(3/8π) sin(θ) e^(iφ)
/// let (re, im): (f64, f64) = sph_harm_complex(1, 1, PI/4.0, PI/3.0).expect("Operation failed");
/// let amplitude = -f64::sqrt(3.0/(8.0*PI)) * f64::sin(PI/4.0);
/// let expected_re = amplitude * f64::cos(PI/3.0);
/// let expected_im = amplitude * f64::sin(PI/3.0);
/// assert!((re - expected_re).abs() < 1e-10);
/// assert!((im - expected_im).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn sph_harm_complex<F>(l: usize, m: i32, theta: F, phi: F) -> SpecialResult<(F, F)>
where
    F: Float + FromPrimitive + Debug,
{
    // Validate that |m| <= l
    let m_abs = m.unsigned_abs() as usize;
    if m_abs > l {
        return Ok((F::zero(), F::zero())); // Y_l^m = 0 if |m| > l
    }

    let cos_theta = theta.cos();
    let x = cos_theta.to_f64().unwrap_or(0.0);

    // Compute P_l^|m|(cos theta) WITHOUT Condon-Shortley phase
    let plm = associated_legendre_for_sph(l, m_abs, x);
    let plm_f = F::from(plm).unwrap_or(F::zero());

    // Compute normalization: K_l^m = sqrt((2l+1)/(4pi) * (l-|m|)!/(l+|m|)!)
    let mut fact_ratio = 1.0_f64;
    if m_abs > 0 {
        for i in (l - m_abs + 1)..=(l + m_abs) {
            fact_ratio /= i as f64;
        }
    }
    let norm = ((2.0 * l as f64 + 1.0) / (4.0 * std::f64::consts::PI) * fact_ratio).sqrt();
    let norm_f = F::from(norm).unwrap_or(F::zero());

    // Physics convention: Y_l^m = (-1)^m * K_l^m * P_l^|m|(cos theta) * e^{im*phi}
    // The (-1)^m is the Condon-Shortley phase applied to the spherical harmonic
    let cs_phase = if m_abs.is_multiple_of(2) {
        F::one()
    } else {
        -F::one()
    };

    // For negative m, use: Y_l^{-|m|} = (-1)^|m| * conj(Y_l^{|m|})
    // Y_l^{|m|} = (-1)^|m| * K * P * e^{i|m|phi}
    // Y_l^{-|m|} = (-1)^|m| * conj((-1)^|m| * K * P * e^{i|m|phi})
    //            = (-1)^{2|m|} * K * P * e^{-i|m|phi}
    //            = K * P * e^{-i|m|phi}
    //
    // For positive m:
    // Y_l^m = (-1)^m * K * P * e^{im*phi}
    //
    // For m = 0:
    // Y_l^0 = K * P

    // Compute e^{im*phi}
    let m_f64 = m as f64;
    let m_phi = F::from(m_f64).unwrap_or(F::zero()) * phi;
    let cos_m_phi = m_phi.cos();
    let sin_m_phi = m_phi.sin();

    let amplitude = if m >= 0 {
        cs_phase * norm_f * plm_f
    } else {
        // Y_l^{-|m|} = (-1)^|m| * conj(Y_l^{|m|})
        // = (-1)^|m| * (-1)^|m| * K * P * e^{-i|m|phi}
        // = (-1)^{2|m|} * K * P * e^{-i|m|phi}  = K * P * e^{-i|m|phi}
        // But e^{im*phi} with m negative already gives e^{-i|m|phi}
        // So amplitude is just K * P (no CS phase for negative m)
        norm_f * plm_f
    };

    let real_part = amplitude * cos_m_phi;
    let imag_part = amplitude * sin_m_phi;

    Ok((real_part, imag_part))
}

/// Compute the normalization factor for spherical harmonics.
///
/// K_l^m = sqrt[(2l+1)/(4pi) * (l-|m|)!/(l+|m|)!]
///
/// This is useful for constructing custom spherical harmonic variants.
///
/// # Examples
/// ```
/// use scirs2_special::sph_harm_normalization;
/// use std::f64::consts::PI;
/// // K_0^0 = 1/(2*sqrt(pi))
/// let k = sph_harm_normalization(0, 0);
/// assert!((k - 0.5 / PI.sqrt()).abs() < 1e-14);
/// ```
pub fn sph_harm_normalization(l: usize, m: i32) -> f64 {
    let m_abs = m.unsigned_abs() as usize;
    if m_abs > l {
        return 0.0;
    }

    let two_l_plus_1 = (2 * l + 1) as f64;
    let four_pi = 4.0 * PI;

    let mut factorial_ratio = 1.0;
    if m_abs > 0 {
        for i in (l - m_abs + 1)..=(l + m_abs) {
            factorial_ratio /= i as f64;
        }
    }

    (two_l_plus_1 / four_pi * factorial_ratio).sqrt()
}

/// Regular solid harmonic R_l^m(r, theta, phi).
///
/// Defined as:
/// ```text
/// R_l^m(r, theta, phi) = sqrt(4*pi/(2l+1)) * r^l * Y_l^m(theta, phi)
/// ```
///
/// Regular solid harmonics are the harmonic polynomial solutions of
/// Laplace's equation that are regular at the origin.
///
/// # Arguments
/// * `l` - Degree
/// * `m` - Order (|m| <= l)
/// * `r` - Radial distance (r >= 0)
/// * `theta` - Polar angle
/// * `phi` - Azimuthal angle
///
/// # Examples
/// ```
/// use scirs2_special::solid_harmonic_regular;
/// use std::f64::consts::PI;
/// // R_0^0 = 1 (independent of r, theta, phi)
/// let r = solid_harmonic_regular(0, 0, 1.0, PI/4.0, 0.0).expect("failed");
/// assert!((r - 1.0).abs() < 1e-10);
/// ```
pub fn solid_harmonic_regular(
    l: usize,
    m: i32,
    r: f64,
    theta: f64,
    phi: f64,
) -> SpecialResult<f64> {
    let y_lm: f64 = sph_harm(l, m, theta, phi)?;
    let factor = (4.0 * PI / (2 * l + 1) as f64).sqrt();
    Ok(factor * r.powi(l as i32) * y_lm)
}

/// Irregular solid harmonic I_l^m(r, theta, phi).
///
/// Defined as:
/// ```text
/// I_l^m(r, theta, phi) = sqrt(4*pi/(2l+1)) * r^{-(l+1)} * Y_l^m(theta, phi)
/// ```
///
/// Irregular solid harmonics are solutions of Laplace's equation that are
/// singular at the origin but regular at infinity.
///
/// # Arguments
/// * `l` - Degree
/// * `m` - Order (|m| <= l)
/// * `r` - Radial distance (r > 0)
/// * `theta` - Polar angle
/// * `phi` - Azimuthal angle
///
/// # Examples
/// ```
/// use scirs2_special::solid_harmonic_irregular;
/// use std::f64::consts::PI;
/// // At r=1: I_0^0 = 1/(2*sqrt(pi)) * sqrt(4*pi) = 1
/// let r = solid_harmonic_irregular(0, 0, 1.0, PI/4.0, 0.0).expect("failed");
/// assert!((r - 1.0).abs() < 1e-10);
/// ```
pub fn solid_harmonic_irregular(
    l: usize,
    m: i32,
    r: f64,
    theta: f64,
    phi: f64,
) -> SpecialResult<f64> {
    if r <= 0.0 {
        return Err(crate::SpecialError::DomainError(
            "r must be > 0 for irregular solid harmonic".to_string(),
        ));
    }

    let y_lm: f64 = sph_harm(l, m, theta, phi)?;
    let factor = (4.0 * PI / (2 * l + 1) as f64).sqrt();
    Ok(factor / r.powi((l + 1) as i32) * y_lm)
}

/// Spherical harmonic addition theorem coefficient.
///
/// The addition theorem states:
/// ```text
/// P_l(cos(gamma)) = (4*pi/(2l+1)) * sum_{m=-l}^{l} Y_l^m*(theta1,phi1) * Y_l^m(theta2,phi2)
/// ```
/// where gamma is the angle between the two directions (theta1,phi1) and (theta2,phi2).
///
/// This function computes cos(gamma) given two directions.
///
/// # Arguments
/// * `theta1`, `phi1` - First direction
/// * `theta2`, `phi2` - Second direction
///
/// # Returns
/// cos(gamma) where gamma is the angle between the two directions
///
/// # Examples
/// ```
/// use scirs2_special::sph_harm_cos_angle;
/// use std::f64::consts::PI;
/// // Same direction: cos(gamma) = 1
/// let cos_g = sph_harm_cos_angle(PI/4.0, 0.0, PI/4.0, 0.0);
/// assert!((cos_g - 1.0).abs() < 1e-14);
/// // Opposite directions: cos(gamma) = -1
/// let cos_g2 = sph_harm_cos_angle(0.0, 0.0, PI, 0.0);
/// assert!((cos_g2 + 1.0).abs() < 1e-14);
/// ```
pub fn sph_harm_cos_angle(theta1: f64, phi1: f64, theta2: f64, phi2: f64) -> f64 {
    theta1.sin() * theta2.sin() * (phi1 - phi2).cos() + theta1.cos() * theta2.cos()
}

/// Verify the spherical harmonic addition theorem.
///
/// Computes P_l(cos(gamma)) and compares with the sum
/// (4pi/(2l+1)) * sum_{m=-l}^{l} Y_l^m(theta1,phi1) * Y_l^m(theta2,phi2)
///
/// Returns both values as (legendre_value, sum_value).
pub fn sph_harm_addition_theorem_check(
    l: usize,
    theta1: f64,
    phi1: f64,
    theta2: f64,
    phi2: f64,
) -> SpecialResult<(f64, f64)> {
    use crate::orthogonal::legendre;

    let cos_gamma = sph_harm_cos_angle(theta1, phi1, theta2, phi2);
    let p_l = legendre(l, cos_gamma);

    let factor = 4.0 * PI / (2 * l + 1) as f64;
    let mut sum = 0.0;

    for m_i in -(l as i32)..=(l as i32) {
        let y1: f64 = sph_harm(l, m_i, theta1, phi1)?;
        let y2: f64 = sph_harm(l, m_i, theta2, phi2)?;
        sum += y1 * y2;
    }

    Ok((p_l, factor * sum))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::{PI, SQRT_2};

    #[test]
    fn test_real_spherical_harmonics() {
        // Test Y₀⁰: Y₀⁰(θ, φ) = 1/(2√π)
        let expected_y00 = 0.5 / f64::sqrt(PI);
        assert_relative_eq!(
            sph_harm(0, 0, 0.0, 0.0).expect("Operation failed"),
            expected_y00,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            sph_harm(0, 0, PI / 2.0, 0.0).expect("Operation failed"),
            expected_y00,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            sph_harm(0, 0, PI, 0.0).expect("Operation failed"),
            expected_y00,
            epsilon = 1e-10
        );

        // Test Y₁⁰: Y₁⁰(θ, φ) = √(3/4π) cos(θ)
        let factor_y10 = f64::sqrt(3.0 / (4.0 * PI));
        assert_relative_eq!(
            sph_harm(1, 0, 0.0, 0.0).expect("Operation failed"),
            factor_y10,
            epsilon = 1e-10
        ); // θ=0, cos(θ)=1
        assert_relative_eq!(
            sph_harm(1, 0, PI / 2.0, 0.0).expect("Operation failed"),
            0.0,
            epsilon = 1e-10
        ); // θ=π/2, cos(θ)=0
        assert_relative_eq!(
            sph_harm(1, 0, PI, 0.0).expect("Operation failed"),
            -factor_y10,
            epsilon = 1e-10
        ); // θ=π, cos(θ)=-1

        // Test Y₁¹: Y₁¹(θ, φ) = √(3/8π) sin(θ) cos(φ) * √2
        let factor_y11 = f64::sqrt(3.0 / (8.0 * PI)) * SQRT_2;

        // At (θ=π/2, φ=0): sin(θ)=1, cos(φ)=1
        assert_relative_eq!(
            sph_harm(1, 1, PI / 2.0, 0.0).expect("Operation failed"),
            factor_y11,
            epsilon = 1e-10
        );

        // At (θ=π/2, φ=π/2): sin(θ)=1, cos(φ)=0
        assert_relative_eq!(
            sph_harm(1, 1, PI / 2.0, PI / 2.0).expect("Operation failed"),
            0.0,
            epsilon = 1e-10
        );

        // Test Y₂⁰: Y₂⁰(θ, φ) = √(5/16π) (3cos²(θ) - 1)
        let factor_y20 = f64::sqrt(5.0 / (16.0 * PI));

        // At θ=0: cos²(θ)=1, Y₂⁰ = √(5/16π) * 2
        assert_relative_eq!(
            sph_harm(2, 0, 0.0, 0.0).expect("Operation failed"),
            factor_y20 * 2.0,
            epsilon = 1e-10
        );

        // Verify that m > l returns zero
        assert_relative_eq!(
            sph_harm(1, 2, PI / 2.0, 0.0).expect("Operation failed"),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_complex_spherical_harmonics() {
        // Test Y₀⁰: Y₀⁰(θ, φ) = 1/(2√π)
        let expected_y00 = 0.5 / f64::sqrt(PI);
        let (re, im) = sph_harm_complex(0, 0, 0.0, 0.0).expect("Operation failed");
        assert_relative_eq!(re, expected_y00, epsilon = 1e-10);
        assert_relative_eq!(im, 0.0, epsilon = 1e-10);

        // Test Y₁⁰: Y₁⁰(θ, φ) = √(3/4π) cos(θ)
        let factor_y10 = f64::sqrt(3.0 / (4.0 * PI));
        let (re, im) = sph_harm_complex(1, 0, 0.0, 0.0).expect("Operation failed");
        assert_relative_eq!(re, factor_y10, epsilon = 1e-10);
        assert_relative_eq!(im, 0.0, epsilon = 1e-10);

        // Test Y₁¹: Y₁¹(θ, φ) = -√(3/8π) sin(θ) e^(iφ)
        let factor_y11 = -f64::sqrt(3.0 / (8.0 * PI));

        // At (θ=π/2, φ=0): sin(θ)=1, e^(iφ)=1
        let (re, im) = sph_harm_complex(1, 1, PI / 2.0, 0.0).expect("Operation failed");
        assert_relative_eq!(re, factor_y11, epsilon = 1e-10);
        assert_relative_eq!(im, 0.0, epsilon = 1e-10);

        // At (θ=π/2, φ=π/2): sin(θ)=1, e^(iφ)=i
        let (re, im) = sph_harm_complex(1, 1, PI / 2.0, PI / 2.0).expect("Operation failed");
        assert_relative_eq!(re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(im, factor_y11, epsilon = 1e-10);

        // Test Y₁⁻¹: Y₁⁻¹(θ, φ) = √(3/8π) sin(θ) e^(-iφ)
        let factor_y1_neg1 = f64::sqrt(3.0 / (8.0 * PI));

        // At (θ=π/2, φ=0): sin(θ)=1, e^(-iφ)=1
        let (re, im) = sph_harm_complex(1, -1, PI / 2.0, 0.0).expect("Operation failed");
        assert_relative_eq!(re, factor_y1_neg1, epsilon = 1e-10);
        assert_relative_eq!(im, 0.0, epsilon = 1e-10);

        // At (θ=π/2, φ=π/2): sin(θ)=1, e^(-iφ)=-i
        let (re, im) = sph_harm_complex(1, -1, PI / 2.0, PI / 2.0).expect("Operation failed");
        assert_relative_eq!(re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(im, -factor_y1_neg1, epsilon = 1e-10);
    }

    // ====== Additional spherical harmonic tests ======

    #[test]
    fn test_sph_harm_orthogonality_same_l() {
        // For the same l, different m should give orthogonal functions
        // Sum over theta grid should vanish for m1 != m2
        // (Crude numerical test)
        let n_theta = 50;
        let n_phi = 50;
        let d_theta = PI / (n_theta as f64);
        let d_phi = 2.0 * PI / (n_phi as f64);

        let mut integral = 0.0;
        for i in 0..n_theta {
            let theta = (i as f64 + 0.5) * d_theta;
            for j in 0..n_phi {
                let phi = (j as f64 + 0.5) * d_phi;
                let y10: f64 = sph_harm(1, 0, theta, phi).expect("failed");
                let y11: f64 = sph_harm(1, 1, theta, phi).expect("failed");
                integral += y10 * y11 * theta.sin() * d_theta * d_phi;
            }
        }
        assert!(
            integral.abs() < 0.05,
            "Y_1^0 and Y_1^1 should be orthogonal, integral = {integral}"
        );
    }

    #[test]
    fn test_sph_harm_normalization_function() {
        let k00 = sph_harm_normalization(0, 0);
        assert_relative_eq!(k00, 0.5 / PI.sqrt(), epsilon = 1e-14);

        let k10 = sph_harm_normalization(1, 0);
        let expected = (3.0 / (4.0 * PI)).sqrt();
        assert_relative_eq!(k10, expected, epsilon = 1e-14);
    }

    #[test]
    fn test_sph_harm_normalization_zero_for_invalid_m() {
        assert_relative_eq!(sph_harm_normalization(1, 3), 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_sph_harm_cos_angle_same_direction() {
        let cos_g = sph_harm_cos_angle(PI / 4.0, PI / 3.0, PI / 4.0, PI / 3.0);
        assert_relative_eq!(cos_g, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_sph_harm_cos_angle_opposite() {
        let cos_g = sph_harm_cos_angle(0.0, 0.0, PI, 0.0);
        assert_relative_eq!(cos_g, -1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_sph_harm_cos_angle_perpendicular() {
        // z-axis and x-axis: cos(gamma) = 0
        let cos_g = sph_harm_cos_angle(0.0, 0.0, PI / 2.0, 0.0);
        assert_relative_eq!(cos_g, 0.0, epsilon = 1e-12);
    }

    // ====== Solid harmonic tests ======

    #[test]
    fn test_solid_harmonic_regular_l0() {
        // R_0^0 = sqrt(4*pi) * 1/(2*sqrt(pi)) = 1
        let r = solid_harmonic_regular(0, 0, 1.0, PI / 4.0, 0.0).expect("failed");
        assert_relative_eq!(r, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solid_harmonic_regular_scales_with_r() {
        // R_l^m scales as r^l
        let r1 = solid_harmonic_regular(2, 0, 1.0, PI / 4.0, 0.0).expect("failed");
        let r2 = solid_harmonic_regular(2, 0, 2.0, PI / 4.0, 0.0).expect("failed");
        assert_relative_eq!(r2, r1 * 4.0, epsilon = 1e-8);
    }

    #[test]
    fn test_solid_harmonic_irregular_l0() {
        // I_0^0(r=1) = sqrt(4pi) * 1/(2*sqrt(pi)) / r = 1
        let r = solid_harmonic_irregular(0, 0, 1.0, PI / 4.0, 0.0).expect("failed");
        assert_relative_eq!(r, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_solid_harmonic_irregular_zero_r_error() {
        assert!(solid_harmonic_irregular(0, 0, 0.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn test_solid_harmonic_irregular_scales_with_r() {
        // I_l^m scales as r^{-(l+1)}
        let r1 = solid_harmonic_irregular(1, 0, 1.0, PI / 4.0, 0.0).expect("failed");
        let r2 = solid_harmonic_irregular(1, 0, 2.0, PI / 4.0, 0.0).expect("failed");
        // r^{-(l+1)} = r^{-2} for l=1, so ratio should be (1/2)^2 = 0.25
        assert_relative_eq!(r2 / r1, 0.25, epsilon = 1e-8);
    }

    // ====== Addition theorem tests ======

    #[test]
    fn test_addition_theorem_l0() {
        // For l=0, P_0(cos gamma) = 1, and the sum should also be 1
        let (p_val, sum_val) =
            sph_harm_addition_theorem_check(0, PI / 4.0, 0.0, PI / 3.0, PI / 6.0).expect("failed");
        assert_relative_eq!(p_val, 1.0, epsilon = 1e-10);
        assert_relative_eq!(sum_val, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_addition_theorem_same_direction() {
        // For same direction, cos(gamma) = 1, P_l(1) = 1
        let (p_val, sum_val) =
            sph_harm_addition_theorem_check(2, PI / 4.0, PI / 3.0, PI / 4.0, PI / 3.0)
                .expect("failed");
        assert_relative_eq!(p_val, 1.0, epsilon = 1e-10);
        assert_relative_eq!(sum_val, 1.0, epsilon = 1e-4);
    }
}
