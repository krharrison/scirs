//! Legendre elliptic integrals via Carlson symmetric forms
//!
//! This module provides high-accuracy implementations of the classical Legendre
//! elliptic integrals by expressing them in terms of Carlson symmetric forms
//! (RF, RD, RJ). This approach provides better numerical stability and
//! uniform accuracy across the entire parameter range.
//!
//! ## Complete Elliptic Integrals
//!
//! - `ellipk_carlson(m)`: K(m) = RF(0, 1-m, 1)
//! - `ellipe_carlson(m)`: E(m) = RF(0, 1-m, 1) - (m/3) RD(0, 1-m, 1)
//! - `ellippi_carlson(n, m)`: Pi(n, m) via RF and RJ
//!
//! ## Incomplete Elliptic Integrals
//!
//! - `ellipf_carlson(phi, m)`: F(phi, m) via RF
//! - `ellipe_inc_carlson(phi, m)`: E(phi, m) via RF and RD
//! - `ellippi_inc_carlson(n, phi, m)`: Pi(n, phi, m) via RF and RJ
//!
//! ## References
//!
//! - Carlson, B. C. (1995). "Numerical computation of real or complex elliptic integrals."
//!   *Numerical Algorithms*, 10(1), 13--26.
//! - DLMF Chapter 19: Elliptic Integrals.

use crate::carlson::{elliprd, elliprf, elliprj};
use crate::error::{SpecialError, SpecialResult};

/// Complete elliptic integral of the first kind K(m) via Carlson RF.
///
/// K(m) = RF(0, 1-m, 1)
///
/// # Arguments
/// * `m` - Parameter (0 <= m < 1)
///
/// # Returns
/// Value of K(m)
///
/// # Examples
/// ```
/// use scirs2_special::legendre_elliptic::ellipk_carlson;
/// let k = ellipk_carlson(0.0).expect("failed");
/// assert!((k - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
/// ```
pub fn ellipk_carlson(m: f64) -> SpecialResult<f64> {
    if !(0.0..1.0).contains(&m) {
        return Err(SpecialError::DomainError(format!(
            "m must be in [0, 1) for K(m), got {m}"
        )));
    }

    elliprf(0.0, 1.0 - m, 1.0)
}

/// Complete elliptic integral of the second kind E(m) via Carlson RF and RD.
///
/// E(m) = RF(0, 1-m, 1) - (m/3) * RD(0, 1-m, 1)
///
/// # Arguments
/// * `m` - Parameter (0 <= m <= 1)
///
/// # Returns
/// Value of E(m)
///
/// # Examples
/// ```
/// use scirs2_special::legendre_elliptic::ellipe_carlson;
/// let e = ellipe_carlson(0.0).expect("failed");
/// assert!((e - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
/// ```
pub fn ellipe_carlson(m: f64) -> SpecialResult<f64> {
    if !(0.0..=1.0).contains(&m) {
        return Err(SpecialError::DomainError(format!(
            "m must be in [0, 1] for E(m), got {m}"
        )));
    }

    if (m - 1.0).abs() < 1e-15 {
        return Ok(1.0);
    }

    let rf_val = elliprf(0.0, 1.0 - m, 1.0)?;
    let rd_val = elliprd(0.0, 1.0 - m, 1.0)?;
    Ok(rf_val - (m / 3.0) * rd_val)
}

/// Complete elliptic integral of the third kind Pi(n, m) via Carlson RF and RJ.
///
/// Pi(n, m) = RF(0, 1-m, 1) + (n/3) * RJ(0, 1-m, 1, 1-n)
///
/// # Arguments
/// * `n` - Characteristic (n < 1)
/// * `m` - Parameter (0 <= m < 1)
///
/// # Returns
/// Value of Pi(n, m)
///
/// # Examples
/// ```
/// use scirs2_special::legendre_elliptic::ellippi_carlson;
/// // Pi(0, m) = K(m)
/// let pi_val = ellippi_carlson(0.0, 0.5).expect("failed");
/// let k_val = scirs2_special::legendre_elliptic::ellipk_carlson(0.5).expect("failed");
/// assert!((pi_val - k_val).abs() < 1e-10);
/// ```
pub fn ellippi_carlson(n: f64, m: f64) -> SpecialResult<f64> {
    if !(0.0..1.0).contains(&m) {
        return Err(SpecialError::DomainError(format!(
            "m must be in [0, 1) for Pi(n, m), got {m}"
        )));
    }

    if (1.0 - n).abs() < 1e-15 {
        return Err(SpecialError::DomainError(
            "n = 1 gives a singular integral".to_string(),
        ));
    }

    let rf_val = elliprf(0.0, 1.0 - m, 1.0)?;

    if n.abs() < 1e-15 {
        return Ok(rf_val);
    }

    let rj_val = elliprj(0.0, 1.0 - m, 1.0, 1.0 - n)?;
    Ok(rf_val + (n / 3.0) * rj_val)
}

/// Incomplete elliptic integral of the first kind F(phi, m) via Carlson RF.
///
/// F(phi, m) = sin(phi) * RF(cos^2(phi), 1 - m*sin^2(phi), 1)
///
/// # Arguments
/// * `phi` - Amplitude angle in radians
/// * `m` - Parameter (0 <= m <= 1)
///
/// # Returns
/// Value of F(phi, m)
///
/// # Examples
/// ```
/// use scirs2_special::legendre_elliptic::ellipf_carlson;
/// // F(0, m) = 0
/// let f = ellipf_carlson(0.0, 0.5).expect("failed");
/// assert!(f.abs() < 1e-14);
/// ```
pub fn ellipf_carlson(phi: f64, m: f64) -> SpecialResult<f64> {
    if !(0.0..=1.0).contains(&m) {
        return Err(SpecialError::DomainError(format!(
            "m must be in [0, 1] for F(phi, m), got {m}"
        )));
    }

    if phi.abs() < 1e-300 {
        return Ok(0.0);
    }

    // Handle range reduction for |phi| > pi/2
    let sign = if phi < 0.0 { -1.0 } else { 1.0 };
    let phi = phi.abs();
    let half_pi = std::f64::consts::FRAC_PI_2;

    if phi <= half_pi {
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        let sin2 = sin_phi * sin_phi;
        let cos2 = cos_phi * cos_phi;

        let rf_val = elliprf(cos2, 1.0 - m * sin2, 1.0)?;
        return Ok(sign * sin_phi * rf_val);
    }

    // For phi > pi/2, use the identity:
    // F(phi, m) = 2*n*K(m) + F(phi - n*pi, m)  where n = floor(phi/(pi/2) + 1/2) * ...
    // More precisely: F(phi + pi, m) = F(phi, m) + 2*K(m)
    let n_periods = (phi / std::f64::consts::PI).floor() as i64;
    let phi_rem = phi - (n_periods as f64) * std::f64::consts::PI;

    let mut result = 0.0;
    if n_periods > 0 && m < 1.0 {
        let k = ellipk_carlson(m)?;
        result += 2.0 * (n_periods as f64) * k;
    }

    if phi_rem > half_pi {
        // F(pi - x, m) = 2*K(m) - F(x, m)
        let x = std::f64::consts::PI - phi_rem;
        if m < 1.0 {
            let k = ellipk_carlson(m)?;
            result += 2.0 * k;
        }
        let sin_x = x.sin();
        let cos_x = x.cos();
        let sin2 = sin_x * sin_x;
        let cos2 = cos_x * cos_x;
        let rf_val = elliprf(cos2, 1.0 - m * sin2, 1.0)?;
        result -= sin_x * rf_val;
    } else {
        let sin_phi = phi_rem.sin();
        let cos_phi = phi_rem.cos();
        let sin2 = sin_phi * sin_phi;
        let cos2 = cos_phi * cos_phi;
        let rf_val = elliprf(cos2, 1.0 - m * sin2, 1.0)?;
        result += sin_phi * rf_val;
    }

    Ok(sign * result)
}

/// Incomplete elliptic integral of the second kind E(phi, m) via Carlson RF and RD.
///
/// E(phi, m) = sin(phi) * RF(cos^2, 1 - m*sin^2, 1)
///             - (m/3) * sin^3(phi) * RD(cos^2, 1 - m*sin^2, 1)
///
/// # Arguments
/// * `phi` - Amplitude angle in radians
/// * `m` - Parameter (0 <= m <= 1)
///
/// # Returns
/// Value of E(phi, m)
///
/// # Examples
/// ```
/// use scirs2_special::legendre_elliptic::ellipe_inc_carlson;
/// // E(0, m) = 0
/// let e = ellipe_inc_carlson(0.0, 0.5).expect("failed");
/// assert!(e.abs() < 1e-14);
/// ```
pub fn ellipe_inc_carlson(phi: f64, m: f64) -> SpecialResult<f64> {
    if !(0.0..=1.0).contains(&m) {
        return Err(SpecialError::DomainError(format!(
            "m must be in [0, 1] for E(phi, m), got {m}"
        )));
    }

    if phi.abs() < 1e-300 {
        return Ok(0.0);
    }

    let sign = if phi < 0.0 { -1.0 } else { 1.0 };
    let phi = phi.abs();
    let half_pi = std::f64::consts::FRAC_PI_2;

    if phi <= half_pi {
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        let sin2 = sin_phi * sin_phi;
        let cos2 = cos_phi * cos_phi;
        let k2 = 1.0 - m * sin2;

        let rf_val = elliprf(cos2, k2, 1.0)?;
        let rd_val = elliprd(cos2, k2, 1.0)?;

        let result = sin_phi * rf_val - (m / 3.0) * sin_phi * sin2 * rd_val;
        return Ok(sign * result);
    }

    // Range reduction for phi > pi/2
    let n_periods = (phi / std::f64::consts::PI).floor() as i64;
    let phi_rem = phi - (n_periods as f64) * std::f64::consts::PI;

    let mut result = 0.0;
    if n_periods > 0 {
        let e_complete = ellipe_carlson(m)?;
        result += 2.0 * (n_periods as f64) * e_complete;
    }

    if phi_rem > half_pi {
        let x = std::f64::consts::PI - phi_rem;
        let e_complete = ellipe_carlson(m)?;
        result += 2.0 * e_complete;

        let sin_x = x.sin();
        let cos_x = x.cos();
        let sin2 = sin_x * sin_x;
        let cos2 = cos_x * cos_x;
        let k2 = 1.0 - m * sin2;

        let rf_val = elliprf(cos2, k2, 1.0)?;
        let rd_val = elliprd(cos2, k2, 1.0)?;
        result -= sin_x * rf_val - (m / 3.0) * sin_x * sin2 * rd_val;
    } else if phi_rem.abs() > 1e-300 {
        let sin_phi = phi_rem.sin();
        let cos_phi = phi_rem.cos();
        let sin2 = sin_phi * sin_phi;
        let cos2 = cos_phi * cos_phi;
        let k2 = 1.0 - m * sin2;

        let rf_val = elliprf(cos2, k2, 1.0)?;
        let rd_val = elliprd(cos2, k2, 1.0)?;
        result += sin_phi * rf_val - (m / 3.0) * sin_phi * sin2 * rd_val;
    }

    Ok(sign * result)
}

/// Incomplete elliptic integral of the third kind Pi(n, phi, m) via Carlson RF and RJ.
///
/// Pi(n, phi, m) = sin(phi) * RF(cos^2, 1 - m*sin^2, 1)
///                 + (n/3) * sin^3(phi) * RJ(cos^2, 1 - m*sin^2, 1, 1 - n*sin^2)
///
/// # Arguments
/// * `n` - Characteristic
/// * `phi` - Amplitude angle in radians
/// * `m` - Parameter (0 <= m <= 1)
///
/// # Returns
/// Value of Pi(n, phi, m)
///
/// # Examples
/// ```
/// use scirs2_special::legendre_elliptic::ellippi_inc_carlson;
/// // Pi(0, phi, m) = F(phi, m)
/// let pi_val = ellippi_inc_carlson(0.0, 0.5, 0.3).expect("failed");
/// let f_val = scirs2_special::legendre_elliptic::ellipf_carlson(0.5, 0.3).expect("failed");
/// assert!((pi_val - f_val).abs() < 1e-10);
/// ```
pub fn ellippi_inc_carlson(n: f64, phi: f64, m: f64) -> SpecialResult<f64> {
    if !(0.0..=1.0).contains(&m) {
        return Err(SpecialError::DomainError(format!(
            "m must be in [0, 1] for Pi(n, phi, m), got {m}"
        )));
    }

    if phi.abs() < 1e-300 {
        return Ok(0.0);
    }

    let sign = if phi < 0.0 { -1.0 } else { 1.0 };
    let phi = phi.abs();

    let sin_phi = phi.sin();
    let cos_phi = phi.cos();
    let sin2 = sin_phi * sin_phi;
    let cos2 = cos_phi * cos_phi;
    let k2 = 1.0 - m * sin2;
    let p2 = 1.0 - n * sin2;

    if p2.abs() < 1e-15 {
        return Err(SpecialError::DomainError(
            "Pi is singular when 1 - n*sin^2(phi) = 0".to_string(),
        ));
    }

    let rf_val = elliprf(cos2, k2, 1.0)?;
    let mut result = sin_phi * rf_val;

    if n.abs() > 1e-15 {
        let rj_val = elliprj(cos2, k2, 1.0, p2)?;
        result += (n / 3.0) * sin_phi * sin2 * rj_val;
    }

    Ok(sign * result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn test_ellipk_carlson_at_zero() {
        // K(0) = pi/2
        let k = ellipk_carlson(0.0).expect("K(0) failed");
        assert_relative_eq!(k, FRAC_PI_2, epsilon = 1e-12);
    }

    #[test]
    fn test_ellipk_carlson_at_half() {
        // K(0.5) ~ 1.854074677301372
        let k = ellipk_carlson(0.5).expect("K(0.5) failed");
        assert_relative_eq!(k, 1.854_074_677_301_37, epsilon = 1e-10);
    }

    #[test]
    fn test_ellipe_carlson_at_zero() {
        // E(0) = pi/2
        let e = ellipe_carlson(0.0).expect("E(0) failed");
        assert_relative_eq!(e, FRAC_PI_2, epsilon = 1e-12);
    }

    #[test]
    fn test_ellipe_carlson_at_one() {
        // E(1) = 1
        let e = ellipe_carlson(1.0).expect("E(1) failed");
        assert_relative_eq!(e, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_ellipe_carlson_at_half() {
        // E(0.5) ~ 1.350643881047675
        let e = ellipe_carlson(0.5).expect("E(0.5) failed");
        assert_relative_eq!(e, 1.350_643_881_047_67, epsilon = 1e-8);
    }

    #[test]
    fn test_legendre_identity_k_e() {
        // Legendre's relation: K(m)*E(1-m) + E(m)*K(1-m) - K(m)*K(1-m) = pi/2
        let m = 0.3;
        let k_m = ellipk_carlson(m).expect("K(m) failed");
        let e_m = ellipe_carlson(m).expect("E(m) failed");
        let k_1m = ellipk_carlson(1.0 - m).expect("K(1-m) failed");
        let e_1m = ellipe_carlson(1.0 - m).expect("E(1-m) failed");

        let lhs = k_m * e_1m + e_m * k_1m - k_m * k_1m;
        assert_relative_eq!(lhs, FRAC_PI_2, epsilon = 1e-8);
    }

    #[test]
    fn test_ellippi_carlson_n_zero() {
        // Pi(0, m) = K(m)
        let pi_val = ellippi_carlson(0.0, 0.5).expect("Pi(0, 0.5) failed");
        let k_val = ellipk_carlson(0.5).expect("K(0.5) failed");
        assert_relative_eq!(pi_val, k_val, epsilon = 1e-10);
    }

    #[test]
    fn test_ellipf_carlson_at_zero() {
        // F(0, m) = 0
        let f = ellipf_carlson(0.0, 0.5).expect("F(0, m) failed");
        assert!(f.abs() < 1e-14);
    }

    #[test]
    fn test_ellipf_carlson_at_half_pi() {
        // F(pi/2, m) = K(m)
        let f = ellipf_carlson(FRAC_PI_2, 0.5).expect("F(pi/2, m) failed");
        let k = ellipk_carlson(0.5).expect("K(m) failed");
        assert_relative_eq!(f, k, epsilon = 1e-10);
    }

    #[test]
    fn test_ellipe_inc_carlson_at_zero() {
        // E(0, m) = 0
        let e = ellipe_inc_carlson(0.0, 0.5).expect("E(0, m) failed");
        assert!(e.abs() < 1e-14);
    }

    #[test]
    fn test_ellipe_inc_carlson_at_half_pi() {
        // E(pi/2, m) = E(m)
        let e_inc = ellipe_inc_carlson(FRAC_PI_2, 0.5).expect("E(pi/2, m) failed");
        let e_comp = ellipe_carlson(0.5).expect("E(m) failed");
        assert_relative_eq!(e_inc, e_comp, epsilon = 1e-8);
    }

    #[test]
    fn test_ellippi_inc_carlson_n_zero() {
        // Pi(0, phi, m) = F(phi, m)
        let phi = 0.7;
        let m = 0.3;
        let pi_val = ellippi_inc_carlson(0.0, phi, m).expect("Pi(0, phi, m) failed");
        let f_val = ellipf_carlson(phi, m).expect("F(phi, m) failed");
        assert_relative_eq!(pi_val, f_val, epsilon = 1e-10);
    }

    #[test]
    fn test_ellipf_carlson_odd() {
        // F(-phi, m) = -F(phi, m)
        let phi = 0.7;
        let m = 0.3;
        let f_pos = ellipf_carlson(phi, m).expect("failed");
        let f_neg = ellipf_carlson(-phi, m).expect("failed");
        assert_relative_eq!(f_pos, -f_neg, epsilon = 1e-10);
    }

    #[test]
    fn test_ellipk_carlson_increases() {
        // K(m) is monotonically increasing for m in [0, 1)
        let k1 = ellipk_carlson(0.1).expect("failed");
        let k2 = ellipk_carlson(0.5).expect("failed");
        let k3 = ellipk_carlson(0.9).expect("failed");
        assert!(k1 < k2, "K should increase: K(0.1)={k1} vs K(0.5)={k2}");
        assert!(k2 < k3, "K should increase: K(0.5)={k2} vs K(0.9)={k3}");
    }

    #[test]
    fn test_ellipk_carlson_domain() {
        assert!(ellipk_carlson(-0.1).is_err());
        assert!(ellipk_carlson(1.0).is_err());
    }

    #[test]
    fn test_ellipe_carlson_decreases() {
        // E(m) is monotonically decreasing for m in [0, 1]
        let e1 = ellipe_carlson(0.1).expect("failed");
        let e2 = ellipe_carlson(0.5).expect("failed");
        let e3 = ellipe_carlson(0.9).expect("failed");
        assert!(e1 > e2, "E should decrease");
        assert!(e2 > e3, "E should decrease");
    }
}
