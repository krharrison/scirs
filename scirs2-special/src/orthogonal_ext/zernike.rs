//! Zernike polynomials
//!
//! Zernike polynomials are a sequence of polynomials orthogonal on the unit disk.
//! They are widely used in optics to describe wavefront aberrations, in computational
//! chemistry for molecular surfaces, and in image processing for shape descriptors.
//!
//! ## Mathematical Definition
//!
//! The Zernike polynomials Z_n^m(rho, theta) are defined as:
//! ```text
//! Z_n^m(rho, theta) = R_n^|m|(rho) * { cos(m*theta) if m >= 0
//!                                       { sin(|m|*theta) if m < 0
//! ```
//!
//! where R_n^m(rho) is the radial polynomial:
//! ```text
//! R_n^m(rho) = sum_{s=0}^{(n-m)/2} (-1)^s * (n-s)! / (s! * ((n+m)/2-s)! * ((n-m)/2-s)!) * rho^{n-2s}
//! ```
//!
//! ## Conventions
//! - n >= 0: radial degree
//! - |m| <= n and n - |m| must be even
//! - 0 <= rho <= 1: radial coordinate
//! - 0 <= theta < 2*pi: azimuthal angle

use crate::error::{SpecialError, SpecialResult};

/// Compute the radial part of the Zernike polynomial R_n^m(rho).
///
/// # Arguments
/// * `n` - Radial degree (non-negative)
/// * `m` - Azimuthal order (non-negative, with n-m even and m <= n)
/// * `rho` - Radial coordinate (0 <= rho <= 1)
///
/// # Returns
/// Value of R_n^m(rho)
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_ext::zernike_radial;
/// // R_0^0(rho) = 1
/// assert!((zernike_radial(0, 0, 0.5).expect("failed") - 1.0).abs() < 1e-14);
/// // R_2^0(rho) = 2*rho^2 - 1
/// assert!((zernike_radial(2, 0, 0.5).expect("failed") - (-0.5)).abs() < 1e-14);
/// ```
pub fn zernike_radial(n: usize, m: usize, rho: f64) -> SpecialResult<f64> {
    if m > n {
        return Err(SpecialError::DomainError(format!(
            "m ({m}) must be <= n ({n})"
        )));
    }

    if (n - m) % 2 != 0 {
        return Err(SpecialError::DomainError(format!(
            "n-m ({}) must be even",
            n - m
        )));
    }

    if rho < 0.0 || rho > 1.0 + 1e-10 {
        return Err(SpecialError::DomainError(format!(
            "rho ({rho}) must be in [0, 1]"
        )));
    }

    let rho = rho.min(1.0); // Clamp to handle floating-point edge cases

    // Special cases
    if n == 0 && m == 0 {
        return Ok(1.0);
    }

    // Number of terms in the sum
    let num_terms = (n - m) / 2 + 1;

    let mut result = 0.0;
    for s in 0..num_terms {
        let sign = if s % 2 == 0 { 1.0 } else { -1.0 };
        let numer = factorial(n - s);
        let denom =
            factorial(s) * factorial((n + m) / 2 - s) * factorial((n - m) / 2 - s);

        if denom == 0.0 {
            return Err(SpecialError::ComputationError(
                "Division by zero in Zernike radial polynomial".to_string(),
            ));
        }

        let power = n - 2 * s;
        let rho_pow = rho.powi(power as i32);

        result += sign * numer / denom * rho_pow;
    }

    Ok(result)
}

/// Compute the Zernike polynomial Z_n^m(rho, theta).
///
/// # Arguments
/// * `n` - Radial degree (non-negative)
/// * `m` - Azimuthal order (integer, |m| <= n, n - |m| even)
/// * `rho` - Radial coordinate (0 <= rho <= 1)
/// * `theta` - Azimuthal angle (radians)
///
/// # Returns
/// Value of Z_n^m(rho, theta)
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_ext::zernike;
/// use std::f64::consts::PI;
/// // Z_0^0 = 1 everywhere on the disk
/// assert!((zernike(0, 0, 0.5, PI/4.0).expect("failed") - 1.0).abs() < 1e-14);
/// ```
pub fn zernike(n: usize, m: i32, rho: f64, theta: f64) -> SpecialResult<f64> {
    let m_abs = m.unsigned_abs() as usize;

    if m_abs > n {
        return Err(SpecialError::DomainError(format!(
            "|m| ({m_abs}) must be <= n ({n})"
        )));
    }

    if (n - m_abs) % 2 != 0 {
        return Err(SpecialError::DomainError(format!(
            "n - |m| ({}) must be even",
            n - m_abs
        )));
    }

    let radial = zernike_radial(n, m_abs, rho)?;

    if m >= 0 {
        Ok(radial * (m as f64 * theta).cos())
    } else {
        Ok(radial * (m_abs as f64 * theta).sin())
    }
}

/// Noll index to (n, m) conversion for Zernike polynomials.
///
/// The Noll single-index ordering is commonly used in optics.
/// j = 1 -> (0, 0), j = 2 -> (1, 1), j = 3 -> (1, -1), etc.
///
/// # Arguments
/// * `j` - Noll index (starting from 1)
///
/// # Returns
/// (n, m) tuple
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_ext::noll_to_nm;
/// assert_eq!(noll_to_nm(1).expect("failed"), (0, 0));
/// assert_eq!(noll_to_nm(2).expect("failed"), (1, 1));
/// assert_eq!(noll_to_nm(3).expect("failed"), (1, -1));
/// assert_eq!(noll_to_nm(4).expect("failed"), (2, 0));
/// ```
pub fn noll_to_nm(j: usize) -> SpecialResult<(usize, i32)> {
    if j == 0 {
        return Err(SpecialError::ValueError(
            "Noll index must be >= 1".to_string(),
        ));
    }

    // Find n such that n*(n+1)/2 < j <= (n+1)*(n+2)/2
    let mut n = 0usize;
    while (n + 1) * (n + 2) / 2 < j {
        n += 1;
    }

    // Position within the row
    let row_start = n * (n + 1) / 2 + 1;
    let pos = j - row_start; // 0-indexed position in this n

    // Determine m from the position
    // For even n: m values are 0, -2, 2, -4, 4, ...
    // For odd n: m values are -1, 1, -3, 3, ...
    let m_abs = if pos % 2 == 0 {
        // Even position: m = n, n-2, n-4, ...
        n - pos
    } else {
        n - pos
    };

    // Actually the Noll ordering is more specific. Let me compute correctly.
    // The number of elements with radial order n is n+1
    // Within a given n, the m values ordered by Noll convention are:
    // If n is even: 0, -2, 2, -4, 4, ..., -n, n
    // If n is odd: -1, 1, -3, 3, ..., -n, n

    let mut m_values: Vec<i32> = Vec::new();
    if n % 2 == 0 {
        m_values.push(0);
        let mut m_val = 2i32;
        while (m_val as usize) <= n {
            m_values.push(-m_val);
            m_values.push(m_val);
            m_val += 2;
        }
    } else {
        let mut m_val = 1i32;
        while (m_val as usize) <= n {
            m_values.push(-m_val);
            m_values.push(m_val);
            m_val += 2;
        }
    }

    if pos < m_values.len() {
        Ok((n, m_values[pos]))
    } else {
        Err(SpecialError::ValueError(format!(
            "Invalid Noll index {j}: position {pos} out of range for n={n}"
        )))
    }
}

/// ANSI/OSA single-index to (n, m) conversion.
///
/// j = 0 -> (0, 0), j = 1 -> (1, -1), j = 2 -> (1, 1), etc.
///
/// Formula: j = n(n+2)/2 + m, n = ceil((-3+sqrt(9+8j))/2)
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_ext::ansi_to_nm;
/// assert_eq!(ansi_to_nm(0).expect("failed"), (0, 0));
/// assert_eq!(ansi_to_nm(1).expect("failed"), (1, -1));
/// assert_eq!(ansi_to_nm(2).expect("failed"), (1, 1));
/// ```
pub fn ansi_to_nm(j: usize) -> SpecialResult<(usize, i32)> {
    let j_f = j as f64;
    let n = ((-3.0 + (9.0 + 8.0 * j_f).sqrt()) / 2.0).ceil() as usize;

    // Ensure n - m is even
    let m_raw = (2 * j) as i32 - (n * (n + 2)) as i32;
    let m = m_raw;

    // Validate
    let m_abs = m.unsigned_abs() as usize;
    if m_abs > n || (n - m_abs) % 2 != 0 {
        return Err(SpecialError::ValueError(format!(
            "Invalid ANSI index {j}: computed (n={n}, m={m})"
        )));
    }

    Ok((n, m))
}

/// Compute factorial as f64
fn factorial(n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    let mut result = 1.0;
    for i in 2..=n {
        result *= i as f64;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ====== Zernike radial polynomial tests ======

    #[test]
    fn test_zernike_radial_r00() {
        // R_0^0(rho) = 1
        let r = zernike_radial(0, 0, 0.5).expect("zernike_radial failed");
        assert!((r - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_zernike_radial_r11() {
        // R_1^1(rho) = rho
        let rho = 0.7;
        let r = zernike_radial(1, 1, rho).expect("zernike_radial failed");
        assert!((r - rho).abs() < 1e-14);
    }

    #[test]
    fn test_zernike_radial_r20() {
        // R_2^0(rho) = 2*rho^2 - 1
        let rho = 0.5;
        let r = zernike_radial(2, 0, rho).expect("zernike_radial failed");
        let expected = 2.0 * rho * rho - 1.0;
        assert!((r - expected).abs() < 1e-14, "got {r}, expected {expected}");
    }

    #[test]
    fn test_zernike_radial_r22() {
        // R_2^2(rho) = rho^2
        let rho = 0.6;
        let r = zernike_radial(2, 2, rho).expect("zernike_radial failed");
        assert!((r - rho * rho).abs() < 1e-14);
    }

    #[test]
    fn test_zernike_radial_r31() {
        // R_3^1(rho) = 3*rho^3 - 2*rho
        let rho = 0.5;
        let r = zernike_radial(3, 1, rho).expect("zernike_radial failed");
        let expected = 3.0 * rho * rho * rho - 2.0 * rho;
        assert!((r - expected).abs() < 1e-14, "got {r}, expected {expected}");
    }

    #[test]
    fn test_zernike_radial_at_boundary() {
        // At rho=1: R_n^m(1) = 1 for all valid (n,m)
        let r = zernike_radial(4, 2, 1.0).expect("zernike_radial failed");
        assert!((r - 1.0).abs() < 1e-14, "R_4^2(1) should be 1, got {r}");
    }

    #[test]
    fn test_zernike_radial_at_center() {
        // At rho=0: R_n^m(0) = 0 for m > 0, R_n^0(0) = (-1)^{n/2}
        let r = zernike_radial(4, 0, 0.0).expect("zernike_radial failed");
        assert!(
            (r - 1.0).abs() < 1e-14,
            "R_4^0(0) should be 1, got {r}"
        );

        let r = zernike_radial(2, 2, 0.0).expect("zernike_radial failed");
        assert!(r.abs() < 1e-14, "R_2^2(0) should be 0, got {r}");
    }

    #[test]
    fn test_zernike_radial_invalid_m() {
        assert!(zernike_radial(2, 3, 0.5).is_err());
    }

    #[test]
    fn test_zernike_radial_odd_nm() {
        // n-m must be even
        assert!(zernike_radial(3, 0, 0.5).is_err());
    }

    // ====== Full Zernike polynomial tests ======

    #[test]
    fn test_zernike_piston() {
        // Z_0^0 = 1 (piston)
        let z = zernike(0, 0, 0.5, PI / 4.0).expect("zernike failed");
        assert!((z - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_zernike_tilt_x() {
        // Z_1^1 = rho * cos(theta) (x-tilt)
        let rho = 0.7;
        let theta = PI / 3.0;
        let z = zernike(1, 1, rho, theta).expect("zernike failed");
        let expected = rho * theta.cos();
        assert!((z - expected).abs() < 1e-14);
    }

    #[test]
    fn test_zernike_tilt_y() {
        // Z_1^{-1} = rho * sin(theta) (y-tilt)
        let rho = 0.7;
        let theta = PI / 3.0;
        let z = zernike(1, -1, rho, theta).expect("zernike failed");
        let expected = rho * theta.sin();
        assert!((z - expected).abs() < 1e-14);
    }

    #[test]
    fn test_zernike_defocus() {
        // Z_2^0 = 2*rho^2 - 1 (defocus)
        let rho = 0.5;
        let theta = PI / 6.0;
        let z = zernike(2, 0, rho, theta).expect("zernike failed");
        let expected = 2.0 * rho * rho - 1.0;
        assert!((z - expected).abs() < 1e-14);
    }

    #[test]
    fn test_zernike_astigmatism() {
        // Z_2^2 = rho^2 * cos(2*theta) (astigmatism)
        let rho = 0.6;
        let theta = PI / 4.0;
        let z = zernike(2, 2, rho, theta).expect("zernike failed");
        let expected = rho * rho * (2.0 * theta).cos();
        assert!((z - expected).abs() < 1e-14);
    }

    // ====== Noll index tests ======

    #[test]
    fn test_noll_to_nm_first() {
        assert_eq!(noll_to_nm(1).expect("failed"), (0, 0));
    }

    #[test]
    fn test_noll_to_nm_second() {
        assert_eq!(noll_to_nm(2).expect("failed"), (1, 1));
    }

    #[test]
    fn test_noll_to_nm_third() {
        assert_eq!(noll_to_nm(3).expect("failed"), (1, -1));
    }

    #[test]
    fn test_noll_to_nm_fourth() {
        assert_eq!(noll_to_nm(4).expect("failed"), (2, 0));
    }

    #[test]
    fn test_noll_to_nm_zero_error() {
        assert!(noll_to_nm(0).is_err());
    }

    // ====== ANSI index tests ======

    #[test]
    fn test_ansi_to_nm_zero() {
        assert_eq!(ansi_to_nm(0).expect("failed"), (0, 0));
    }

    #[test]
    fn test_ansi_to_nm_first() {
        assert_eq!(ansi_to_nm(1).expect("failed"), (1, -1));
    }

    #[test]
    fn test_ansi_to_nm_second() {
        assert_eq!(ansi_to_nm(2).expect("failed"), (1, 1));
    }

    #[test]
    fn test_ansi_to_nm_third() {
        // j=3 -> n=2, m = 2*3 - 2*4 = -2 -> (2, -2)
        assert_eq!(ansi_to_nm(3).expect("failed"), (2, -2));
    }

    #[test]
    fn test_ansi_to_nm_fourth() {
        // j=4 -> n=2, m = 2*4 - 2*4 = 0 -> (2, 0)
        assert_eq!(ansi_to_nm(4).expect("failed"), (2, 0));
    }
}
