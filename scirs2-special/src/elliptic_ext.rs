//! Extended elliptic integrals and functions
//!
//! This module provides additional elliptic functions beyond the core `elliptic.rs` module:
//!
//! - **Jacobi theta functions**: theta1, theta2, theta3, theta4
//! - **AGM-based elliptic integrals**: High-precision K(m) and E(m) via arithmetic-geometric mean
//! - **Bulirsch elliptic integrals**: Alternative complete/incomplete forms
//! - **Elliptic integral of the third kind**: Enhanced incomplete Pi
//! - **Jacobi amplitude function**: am(u, m)
//! - **Inverse Jacobi elliptic functions**: arcsn, arccn, arcdn
//!
//! ## Mathematical Background
//!
//! ### Jacobi Theta Functions
//!
//! The four Jacobi theta functions are defined as:
//!
//! ```text
//! theta1(z, q) = 2 sum_{n=0}^{inf} (-1)^n q^{(n+1/2)^2} sin((2n+1)z)
//! theta2(z, q) = 2 sum_{n=0}^{inf} q^{(n+1/2)^2} cos((2n+1)z)
//! theta3(z, q) = 1 + 2 sum_{n=1}^{inf} q^{n^2} cos(2nz)
//! theta4(z, q) = 1 + 2 sum_{n=1}^{inf} (-1)^n q^{n^2} cos(2nz)
//! ```
//!
//! where q is the elliptic nome with |q| < 1.
//!
//! These functions are fundamental in the theory of elliptic functions and
//! have applications in number theory, algebraic geometry, and mathematical physics.
//!
//! ### Arithmetic-Geometric Mean (AGM)
//!
//! The AGM of two positive numbers a and b is computed by the iteration:
//! ```text
//! a_{n+1} = (a_n + b_n) / 2
//! b_{n+1} = sqrt(a_n * b_n)
//! ```
//! which converges quadratically. The complete elliptic integral K(m) is:
//! ```text
//! K(m) = pi / (2 * AGM(1, sqrt(1-m)))
//! ```

use crate::error::{SpecialError, SpecialResult};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

/// Maximum iterations for AGM convergence
const AGM_MAX_ITER: usize = 100;

/// Convergence tolerance for AGM
const AGM_TOL: f64 = 1e-16;

/// Maximum terms in theta series
const THETA_MAX_TERMS: usize = 200;

/// Convergence tolerance for theta series
const THETA_TOL: f64 = 1e-16;

// ========================================================================
// Jacobi Theta Functions
// ========================================================================

/// Jacobi theta function theta1(z, q).
///
/// ```text
/// theta1(z, q) = 2 sum_{n=0}^{inf} (-1)^n q^{(n+1/2)^2} sin((2n+1)z)
/// ```
///
/// theta1 is an odd function of z and vanishes at z = 0.
///
/// # Arguments
/// * `z` - Argument (real)
/// * `q` - Nome parameter (|q| < 1)
///
/// # Returns
/// Value of theta1(z, q)
///
/// # Examples
/// ```
/// use scirs2_special::elliptic_ext::theta1;
/// // theta1(0, q) = 0 for all q
/// let val = theta1(0.0, 0.1).expect("failed");
/// assert!(val.abs() < 1e-14);
/// ```
pub fn theta1(z: f64, q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;

    if q.abs() < 1e-300 {
        return Ok(0.0);
    }

    let mut result = 0.0;
    for n in 0..THETA_MAX_TERMS {
        let n_f = n as f64;
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let exponent = (n_f + 0.5) * (n_f + 0.5);
        let q_power = q.powf(exponent);

        if q_power < THETA_TOL {
            break;
        }

        let term = sign * q_power * ((2.0 * n_f + 1.0) * z).sin();
        result += term;
    }

    Ok(2.0 * result)
}

/// Jacobi theta function theta2(z, q).
///
/// ```text
/// theta2(z, q) = 2 sum_{n=0}^{inf} q^{(n+1/2)^2} cos((2n+1)z)
/// ```
///
/// # Arguments
/// * `z` - Argument (real)
/// * `q` - Nome parameter (|q| < 1)
///
/// # Returns
/// Value of theta2(z, q)
///
/// # Examples
/// ```
/// use scirs2_special::elliptic_ext::theta2;
/// // theta2(0, q) = 2 sum q^{(n+1/2)^2}
/// let val = theta2(0.0, 0.1).expect("failed");
/// assert!(val > 0.0);
/// ```
pub fn theta2(z: f64, q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;

    if q.abs() < 1e-300 {
        return Ok(0.0);
    }

    let mut result = 0.0;
    for n in 0..THETA_MAX_TERMS {
        let n_f = n as f64;
        let exponent = (n_f + 0.5) * (n_f + 0.5);
        let q_power = q.powf(exponent);

        if q_power < THETA_TOL {
            break;
        }

        let term = q_power * ((2.0 * n_f + 1.0) * z).cos();
        result += term;
    }

    Ok(2.0 * result)
}

/// Jacobi theta function theta3(z, q).
///
/// ```text
/// theta3(z, q) = 1 + 2 sum_{n=1}^{inf} q^{n^2} cos(2nz)
/// ```
///
/// # Arguments
/// * `z` - Argument (real)
/// * `q` - Nome parameter (|q| < 1)
///
/// # Returns
/// Value of theta3(z, q)
///
/// # Examples
/// ```
/// use scirs2_special::elliptic_ext::theta3;
/// // theta3(0, 0) = 1
/// let val = theta3(0.0, 0.0).expect("failed");
/// assert!((val - 1.0).abs() < 1e-14);
/// ```
pub fn theta3(z: f64, q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;

    if q.abs() < 1e-300 {
        return Ok(1.0);
    }

    let mut result = 1.0;
    for n in 1..=THETA_MAX_TERMS {
        let n_f = n as f64;
        let q_power = q.powf(n_f * n_f);

        if q_power < THETA_TOL {
            break;
        }

        let term = q_power * (2.0 * n_f * z).cos();
        result += 2.0 * term;
    }

    Ok(result)
}

/// Jacobi theta function theta4(z, q).
///
/// ```text
/// theta4(z, q) = 1 + 2 sum_{n=1}^{inf} (-1)^n q^{n^2} cos(2nz)
/// ```
///
/// # Arguments
/// * `z` - Argument (real)
/// * `q` - Nome parameter (|q| < 1)
///
/// # Returns
/// Value of theta4(z, q)
///
/// # Examples
/// ```
/// use scirs2_special::elliptic_ext::theta4;
/// let val = theta4(0.0, 0.0).expect("failed");
/// assert!((val - 1.0).abs() < 1e-14);
/// ```
pub fn theta4(z: f64, q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;

    if q.abs() < 1e-300 {
        return Ok(1.0);
    }

    let mut result = 1.0;
    for n in 1..=THETA_MAX_TERMS {
        let n_f = n as f64;
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let q_power = q.powf(n_f * n_f);

        if q_power < THETA_TOL {
            break;
        }

        let term = sign * q_power * (2.0 * n_f * z).cos();
        result += 2.0 * term;
    }

    Ok(result)
}

// ========================================================================
// AGM-based Elliptic Integrals
// ========================================================================

/// Arithmetic-Geometric Mean of two positive numbers.
///
/// Computes AGM(a, b) by the iteration:
/// ```text
/// a_{n+1} = (a_n + b_n) / 2
/// b_{n+1} = sqrt(a_n * b_n)
/// ```
/// which converges quadratically.
///
/// # Arguments
/// * `a` - First number (positive)
/// * `b` - Second number (positive)
///
/// # Returns
/// Value of AGM(a, b)
///
/// # Examples
/// ```
/// use scirs2_special::elliptic_ext::agm;
/// let val = agm(1.0, 1.0).expect("failed");
/// assert!((val - 1.0).abs() < 1e-14);
/// ```
pub fn agm(mut a: f64, mut b: f64) -> SpecialResult<f64> {
    if a <= 0.0 || b <= 0.0 {
        return Err(SpecialError::DomainError(
            "AGM requires positive arguments".to_string(),
        ));
    }

    for _ in 0..AGM_MAX_ITER {
        let a_new = (a + b) / 2.0;
        let b_new = (a * b).sqrt();

        if (a_new - b_new).abs() < AGM_TOL * a_new {
            return Ok(a_new);
        }

        a = a_new;
        b = b_new;
    }

    Ok((a + b) / 2.0)
}

/// Complete elliptic integral K(m) via the AGM algorithm.
///
/// Uses the identity:
/// ```text
/// K(m) = pi / (2 * AGM(1, sqrt(1-m)))
/// ```
///
/// This provides very high precision with quadratic convergence.
///
/// # Arguments
/// * `m` - Parameter (0 <= m < 1)
///
/// # Returns
/// Value of K(m)
///
/// # Examples
/// ```
/// use scirs2_special::elliptic_ext::ellipk_agm;
/// let k = ellipk_agm(0.0).expect("failed");
/// assert!((k - std::f64::consts::PI / 2.0).abs() < 1e-14);
/// ```
pub fn ellipk_agm(m: f64) -> SpecialResult<f64> {
    if m < 0.0 || m >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "m must be in [0, 1) for ellipk_agm, got {m}"
        )));
    }

    if m == 0.0 {
        return Ok(std::f64::consts::PI / 2.0);
    }

    let agm_val = agm(1.0, (1.0 - m).sqrt())?;
    Ok(std::f64::consts::PI / (2.0 * agm_val))
}

/// Complete elliptic integral E(m) via the AGM algorithm with modification tracking.
///
/// Uses the AGM sequence to compute both K(m) and E(m) simultaneously
/// via the Gauss-Legendre relation.
///
/// # Arguments
/// * `m` - Parameter (0 <= m <= 1)
///
/// # Returns
/// Value of E(m)
///
/// # Examples
/// ```
/// use scirs2_special::elliptic_ext::ellipe_agm;
/// let e = ellipe_agm(0.0).expect("failed");
/// assert!((e - std::f64::consts::PI / 2.0).abs() < 1e-14);
/// ```
pub fn ellipe_agm(m: f64) -> SpecialResult<f64> {
    if m < 0.0 || m > 1.0 {
        return Err(SpecialError::DomainError(format!(
            "m must be in [0, 1] for ellipe_agm, got {m}"
        )));
    }

    if m == 0.0 {
        return Ok(std::f64::consts::PI / 2.0);
    }

    if m == 1.0 {
        return Ok(1.0);
    }

    // Use the AGM iteration with modification tracking.
    // E(m) = K(m) * [1 - (1/2) * sum_{n=0}^N 2^n * c_n^2]
    // where c_0 = sqrt(m), c_n = (a_{n-1} - b_{n-1})/2 for n >= 1.
    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();
    let mut c_sq_sum = m; // c_0^2 * 2^0 = m

    let mut power_of_2 = 1.0;

    for _ in 0..AGM_MAX_ITER {
        let a_new = (a + b) / 2.0;
        let b_new = (a * b).sqrt();
        let c = (a - b) / 2.0;

        power_of_2 *= 2.0;

        // Only add c^2 * 2^n if it's above machine precision contribution
        let term = c * c * power_of_2;
        if term > 1e-30 {
            c_sq_sum += term;
        }

        // Converge when c is negligible relative to a
        if c.abs() < 1e-15 * a_new {
            a = a_new;
            break;
        }

        a = a_new;
        b = b_new;
    }

    let k = std::f64::consts::PI / (2.0 * a);
    Ok(k * (1.0 - c_sq_sum / 2.0))
}

// ========================================================================
// Jacobi Amplitude Function
// ========================================================================

/// Jacobi amplitude function am(u, m).
///
/// The amplitude function is defined as the inverse of the incomplete
/// elliptic integral of the first kind:
/// if u = F(phi, m), then phi = am(u, m).
///
/// The Jacobi elliptic functions are then:
/// sn(u, m) = sin(am(u, m))
/// cn(u, m) = cos(am(u, m))
///
/// # Arguments
/// * `u` - Argument
/// * `m` - Parameter (0 <= m <= 1)
///
/// # Returns
/// Value of am(u, m) in radians
///
/// # Examples
/// ```
/// use scirs2_special::elliptic_ext::jacobi_am;
/// // am(0, m) = 0 for all m
/// let val = jacobi_am(0.0, 0.5).expect("failed");
/// assert!(val.abs() < 1e-14);
/// ```
pub fn jacobi_am(u: f64, m: f64) -> SpecialResult<f64> {
    if m < 0.0 || m > 1.0 {
        return Err(SpecialError::DomainError(format!(
            "m must be in [0, 1] for jacobi_am, got {m}"
        )));
    }

    if u.abs() < 1e-300 {
        return Ok(0.0);
    }

    // Special case m = 0: am(u, 0) = u
    if m < 1e-15 {
        return Ok(u);
    }

    // Special case m = 1: am(u, 1) = 2*arctan(exp(u)) - pi/2 = gd(u) (Gudermannian)
    if (m - 1.0).abs() < 1e-15 {
        return Ok(2.0 * u.exp().atan() - std::f64::consts::FRAC_PI_2);
    }

    // Use the descending Landen transformation (AGM approach)
    // Collect the AGM sequence
    let mut a_seq = Vec::with_capacity(32);
    let mut c_seq = Vec::with_capacity(32);

    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();

    a_seq.push(a);
    c_seq.push(m.sqrt());

    for _ in 0..AGM_MAX_ITER {
        let a_new = (a + b) / 2.0;
        let b_new = (a * b).sqrt();
        let c_new = (a - b) / 2.0;

        a_seq.push(a_new);
        c_seq.push(c_new);

        if c_new.abs() < AGM_TOL {
            break;
        }

        a = a_new;
        b = b_new;
    }

    // Now work backwards to compute phi
    let n = a_seq.len() - 1;
    // phi_n = 2^n * a_n * u
    let mut phi = (2.0_f64).powi(n as i32) * a_seq[n] * u;

    // Backward recurrence: phi_{k-1} = (phi_k + arcsin(c_k/a_k * sin(phi_k))) / 2
    for k in (1..=n).rev() {
        let sin_phi = phi.sin();
        let arg = (c_seq[k] / a_seq[k] * sin_phi).clamp(-1.0, 1.0);
        phi = (phi + arg.asin()) / 2.0;
    }

    Ok(phi)
}

// ========================================================================
// Inverse Jacobi Elliptic Functions
// ========================================================================

/// Inverse Jacobi elliptic function: arcsn(x, m).
///
/// Returns u such that sn(u, m) = x.
/// Equivalent to F(arcsin(x), m) (incomplete elliptic integral of the first kind).
///
/// # Arguments
/// * `x` - Value (|x| <= 1)
/// * `m` - Parameter (0 <= m <= 1)
///
/// # Returns
/// Value of arcsn(x, m)
///
/// # Examples
/// ```
/// use scirs2_special::elliptic_ext::arcsn;
/// // arcsn(0, m) = 0
/// let val = arcsn(0.0, 0.5).expect("failed");
/// assert!(val.abs() < 1e-14);
/// ```
pub fn arcsn(x: f64, m: f64) -> SpecialResult<f64> {
    if x.abs() > 1.0 + 1e-10 {
        return Err(SpecialError::DomainError(format!(
            "|x| must be <= 1 for arcsn, got {x}"
        )));
    }

    if m < 0.0 || m > 1.0 {
        return Err(SpecialError::DomainError(format!(
            "m must be in [0, 1] for arcsn, got {m}"
        )));
    }

    let x = x.clamp(-1.0, 1.0);

    if x.abs() < 1e-300 {
        return Ok(0.0);
    }

    // arcsn(x, m) = F(arcsin(x), m)
    let phi = x.asin();
    Ok(elliptic_f_accurate(phi, m))
}

/// Inverse Jacobi elliptic function: arccn(x, m).
///
/// Returns u such that cn(u, m) = x.
///
/// # Arguments
/// * `x` - Value (|x| <= 1)
/// * `m` - Parameter (0 <= m <= 1)
pub fn arccn(x: f64, m: f64) -> SpecialResult<f64> {
    if x.abs() > 1.0 + 1e-10 {
        return Err(SpecialError::DomainError(format!(
            "|x| must be <= 1 for arccn, got {x}"
        )));
    }

    if m < 0.0 || m > 1.0 {
        return Err(SpecialError::DomainError(format!(
            "m must be in [0, 1] for arccn, got {m}"
        )));
    }

    let x = x.clamp(-1.0, 1.0);

    // cn(u, m) = x means sn(u, m) = sqrt(1 - x^2)
    // => arccn(x, m) = arcsn(sqrt(1 - x^2), m) = F(arccos(x), m)
    let phi = x.acos();
    Ok(elliptic_f_accurate(phi, m))
}

/// Inverse Jacobi elliptic function: arcdn(x, m).
///
/// Returns u such that dn(u, m) = x.
///
/// # Arguments
/// * `x` - Value (sqrt(1-m) <= x <= 1 for 0 < m < 1)
/// * `m` - Parameter (0 < m <= 1)
pub fn arcdn(x: f64, m: f64) -> SpecialResult<f64> {
    if m <= 0.0 || m > 1.0 {
        return Err(SpecialError::DomainError(format!(
            "m must be in (0, 1] for arcdn, got {m}"
        )));
    }

    let lower_bound = (1.0 - m).sqrt();
    if x < lower_bound - 1e-10 || x > 1.0 + 1e-10 {
        return Err(SpecialError::DomainError(format!(
            "x must be in [sqrt(1-m), 1] = [{lower_bound}, 1] for arcdn, got {x}"
        )));
    }

    let x = x.clamp(lower_bound, 1.0);

    // dn(u, m) = x => sn^2(u, m) = (1 - x^2) / m
    // => sn(u, m) = sqrt((1 - x^2) / m)
    let sn_val = ((1.0 - x * x) / m).sqrt().min(1.0);

    // arcdn(x, m) = F(arcsin(sn), m)
    let phi = sn_val.asin();
    Ok(elliptic_f_accurate(phi, m))
}

// ========================================================================
// Bulirsch Elliptic Integrals
// ========================================================================

/// Bulirsch's complete elliptic integral cel(kc, p, a, b).
///
/// Computes the general complete elliptic integral:
/// ```text
/// cel(kc, p, a, b) = integral_0^{pi/2} (a cos^2(t) + b sin^2(t)) / ((cos^2(t) + p sin^2(t)) sqrt(cos^2(t) + kc^2 sin^2(t))) dt
/// ```
///
/// Special cases:
/// - cel(kc, 1, 1, 1) = K(1-kc^2) (complete elliptic integral of the first kind)
/// - cel(kc, 1, 1, kc^2) = E(1-kc^2) (complete elliptic integral of the second kind)
///
/// # Arguments
/// * `kc` - Complementary modulus (kc = sqrt(1-m))
/// * `p` - Parameter p
/// * `a` - Parameter a
/// * `b` - Parameter b
///
/// # Returns
/// Value of cel(kc, p, a, b)
///
/// # Examples
/// ```
/// use scirs2_special::elliptic_ext::bulirsch_cel;
/// // cel(kc, 1, 1, 1) = K(1-kc^2)
/// let kc = (0.5_f64).sqrt(); // m = 0.5
/// let val = bulirsch_cel(kc, 1.0, 1.0, 1.0).expect("failed");
/// // Should match K(0.5) ~ 1.854
/// assert!((val - 1.854).abs() < 0.01);
/// ```
pub fn bulirsch_cel(kc: f64, p: f64, a: f64, b: f64) -> SpecialResult<f64> {
    if kc.abs() < 1e-300 {
        return Err(SpecialError::DomainError(
            "kc must be nonzero for bulirsch_cel".to_string(),
        ));
    }

    // Bulirsch's cel algorithm (Numerical Recipes 3rd Ed, Section 6.12)
    let eps = 1e-16;
    let mut kc = kc.abs();
    let mut p = p;
    let mut a = a;
    let mut b = b;
    let mut em = 1.0_f64;

    if p > 0.0 {
        p = p.sqrt();
        b /= p;
    } else {
        let f = kc * kc;
        let q = 1.0 - f;
        let g = 1.0 - p;
        let ff = f - p;
        let qq = q * (b - a * p);
        p = (ff / g).sqrt();
        a = (a - b) / g;
        b = -qq / (g * g * p) + a * p;
    }

    for _ in 0..AGM_MAX_ITER {
        let f = a;
        a += b / p;
        let g = em / p;
        b = 2.0 * (f * g + b);
        p = g + p;
        let g_old = em;
        em = kc + em;
        if (g_old - kc).abs() <= g_old * eps {
            break;
        }
        kc = 2.0 * (g_old * kc).sqrt();
    }

    Ok(std::f64::consts::FRAC_PI_2 * (a * em + b) / (em * (em + p)))
}

// ========================================================================
// Theta Function Relations
// ========================================================================

/// Compute the nome q from the parameter m using the exact relation:
/// q = exp(-pi K'(m) / K(m))
///
/// This provides higher accuracy than polynomial approximations.
///
/// # Arguments
/// * `m` - Parameter (0 <= m < 1)
///
/// # Returns
/// Nome q
pub fn nome_from_m(m: f64) -> SpecialResult<f64> {
    if m < 0.0 || m >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "m must be in [0, 1), got {m}"
        )));
    }

    if m < 1e-300 {
        return Ok(0.0);
    }

    let k = ellipk_agm(m)?;
    let k_prime = ellipk_agm(1.0 - m)?;
    let ratio = std::f64::consts::PI * k_prime / k;
    Ok((-ratio).exp())
}

/// Compute m from the nome q using theta function values.
///
/// Uses the identity: m = (theta2(0,q) / theta3(0,q))^4
///
/// # Arguments
/// * `q` - Nome (0 <= q < 1)
///
/// # Returns
/// Parameter m
pub fn m_from_nome(q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;

    if q < 1e-300 {
        return Ok(0.0);
    }

    let t2 = theta2(0.0, q)?;
    let t3 = theta3(0.0, q)?;

    if t3.abs() < 1e-300 {
        return Err(SpecialError::ComputationError(
            "theta3(0,q) is zero".to_string(),
        ));
    }

    let ratio = t2 / t3;
    Ok(ratio.powi(4))
}

/// Jacobi theta function derivatives.
///
/// Compute d/dz theta_i(z, q) for i = 1, 2, 3, 4.
///
/// Uses the series definition differentiated term by term.
///
/// # Arguments
/// * `index` - Theta function index (1, 2, 3, or 4)
/// * `z` - Argument
/// * `q` - Nome (|q| < 1)
///
/// # Returns
/// Value of d/dz theta_i(z, q)
pub fn theta_derivative(index: u32, z: f64, q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;

    match index {
        1 => {
            // d/dz theta1 = 2 sum (-1)^n q^{(n+1/2)^2} (2n+1) cos((2n+1)z)
            let mut result = 0.0;
            for n in 0..THETA_MAX_TERMS {
                let n_f = n as f64;
                let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                let exponent = (n_f + 0.5) * (n_f + 0.5);
                let q_power = q.powf(exponent);
                if q_power < THETA_TOL {
                    break;
                }
                let factor = 2.0 * n_f + 1.0;
                result += sign * q_power * factor * (factor * z).cos();
            }
            Ok(2.0 * result)
        }
        2 => {
            // d/dz theta2 = -2 sum q^{(n+1/2)^2} (2n+1) sin((2n+1)z)
            let mut result = 0.0;
            for n in 0..THETA_MAX_TERMS {
                let n_f = n as f64;
                let exponent = (n_f + 0.5) * (n_f + 0.5);
                let q_power = q.powf(exponent);
                if q_power < THETA_TOL {
                    break;
                }
                let factor = 2.0 * n_f + 1.0;
                result += q_power * factor * (factor * z).sin();
            }
            Ok(-2.0 * result)
        }
        3 => {
            // d/dz theta3 = -4 sum n q^{n^2} sin(2nz)
            let mut result = 0.0;
            for n in 1..=THETA_MAX_TERMS {
                let n_f = n as f64;
                let q_power = q.powf(n_f * n_f);
                if q_power < THETA_TOL {
                    break;
                }
                result += n_f * q_power * (2.0 * n_f * z).sin();
            }
            Ok(-4.0 * result)
        }
        4 => {
            // d/dz theta4 = -4 sum (-1)^n n q^{n^2} sin(2nz)
            let mut result = 0.0;
            for n in 1..=THETA_MAX_TERMS {
                let n_f = n as f64;
                let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                let q_power = q.powf(n_f * n_f);
                if q_power < THETA_TOL {
                    break;
                }
                result += sign * n_f * q_power * (2.0 * n_f * z).sin();
            }
            Ok(-4.0 * result)
        }
        _ => Err(SpecialError::ValueError(format!(
            "Theta function index must be 1, 2, 3, or 4, got {index}"
        ))),
    }
}

// ========================================================================
// Theta Function Identities
// ========================================================================

/// Verify the Jacobi identity: theta3(0,q)^4 = theta2(0,q)^4 + theta4(0,q)^4.
///
/// Returns the relative error of the identity.
///
/// # Arguments
/// * `q` - Nome (|q| < 1)
pub fn jacobi_identity_check(q: f64) -> SpecialResult<f64> {
    let t2 = theta2(0.0, q)?;
    let t3 = theta3(0.0, q)?;
    let t4 = theta4(0.0, q)?;

    let lhs = t3.powi(4);
    let rhs = t2.powi(4) + t4.powi(4);

    if lhs.abs() < 1e-300 {
        return Ok(0.0);
    }

    Ok((lhs - rhs).abs() / lhs.abs())
}

// ========================================================================
// Incomplete Elliptic Integral (AGM-based, accurate)
// ========================================================================

/// Accurate incomplete elliptic integral of the first kind F(phi, m)
/// using Gauss-Legendre quadrature.
///
/// F(phi, m) = integral_0^phi dt / sqrt(1 - m*sin^2(t))
fn elliptic_f_accurate(phi: f64, m: f64) -> f64 {
    if phi.abs() < 1e-300 {
        return 0.0;
    }
    if m.abs() < 1e-300 {
        return phi;
    }
    if m == 1.0 {
        // F(phi, 1) = ln(sec(phi) + tan(phi)) = atanh(sin(phi))
        let sp = phi.sin();
        if sp.abs() >= 1.0 {
            return f64::INFINITY;
        }
        return sp.atanh();
    }

    // Handle range reduction: bring phi into [0, pi/2]
    let sign = if phi < 0.0 { -1.0 } else { 1.0 };
    let phi = phi.abs();

    // For phi > pi/2, use the identity F(phi, m) = 2*K(m)*floor(phi/(pi)) + F(phi mod pi, m)
    // More precisely, reduce to [0, pi/2]
    let pi = std::f64::consts::PI;
    let half_pi = pi / 2.0;

    if phi <= half_pi {
        return sign * elliptic_f_quadrature(phi, m);
    }

    // Number of complete half-periods
    let n_half = (phi / half_pi).floor() as i64;
    let phi_rem = phi - (n_half as f64) * half_pi;

    // Each half-period contributes K(m)
    let k = match ellipk_agm(m) {
        Ok(k) => k,
        Err(_) => return f64::NAN,
    };

    sign * ((n_half as f64) * k + elliptic_f_quadrature(phi_rem, m))
}

/// Gauss-Legendre quadrature for F(phi, m) with phi in [0, pi/2].
fn elliptic_f_quadrature(phi: f64, m: f64) -> f64 {
    if phi.abs() < 1e-300 {
        return 0.0;
    }

    // Use 16-point Gauss-Legendre quadrature on [0, phi]
    // Transform integral to [0, 1]: t = phi * s, dt = phi * ds
    // F = phi * integral_0^1 1/sqrt(1 - m*sin^2(phi*s)) ds

    // Gauss-Legendre 16-point nodes and weights on [-1, 1]
    let gl_nodes: [f64; 16] = [
        -0.9894009349916499,
        -0.9445750230732326,
        -0.8656312023878318,
        -0.7554044083550030,
        -0.6178762444026438,
        -0.4580167776572274,
        -0.2816035507792589,
        -0.0950125098376374,
        0.0950125098376374,
        0.2816035507792589,
        0.4580167776572274,
        0.6178762444026438,
        0.7554044083550030,
        0.8656312023878318,
        0.9445750230732326,
        0.9894009349916499,
    ];
    let gl_weights: [f64; 16] = [
        0.0271524594117541,
        0.0622535239386479,
        0.0951585116824928,
        0.1246289712555339,
        0.1495959888165767,
        0.1691565193950025,
        0.1826034150449236,
        0.1894506104550685,
        0.1894506104550685,
        0.1826034150449236,
        0.1691565193950025,
        0.1495959888165767,
        0.1246289712555339,
        0.0951585116824928,
        0.0622535239386479,
        0.0271524594117541,
    ];

    let half_phi = phi / 2.0;
    let mid = phi / 2.0;
    let mut result = 0.0;

    for i in 0..16 {
        let t = mid + half_phi * gl_nodes[i];
        let sin_t = t.sin();
        let integrand = 1.0 / (1.0 - m * sin_t * sin_t).sqrt();
        result += gl_weights[i] * integrand;
    }

    result * half_phi
}

// ========================================================================
// Internal Helper Functions
// ========================================================================

/// Validate the nome parameter q.
fn validate_nome(q: f64) -> SpecialResult<()> {
    if q.abs() >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "|q| must be < 1 for theta functions, got {q}"
        )));
    }
    if q.is_nan() {
        return Err(SpecialError::DomainError("q must not be NaN".to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    // ====== Theta function tests ======

    #[test]
    fn test_theta1_at_zero() {
        // theta1(0, q) = 0 for all q
        let val = theta1(0.0, 0.1).expect("theta1 failed");
        assert!(val.abs() < 1e-14, "theta1(0, 0.1) should be 0, got {val}");
    }

    #[test]
    fn test_theta1_odd() {
        // theta1 is odd: theta1(-z, q) = -theta1(z, q)
        let z = 0.5;
        let q = 0.1;
        let t_pos = theta1(z, q).expect("failed");
        let t_neg = theta1(-z, q).expect("failed");
        assert_relative_eq!(t_pos, -t_neg, epsilon = 1e-14);
    }

    #[test]
    fn test_theta2_at_zero() {
        // theta2(0, q) = 2 sum q^{(n+1/2)^2} > 0
        let val = theta2(0.0, 0.1).expect("theta2 failed");
        assert!(val > 0.0, "theta2(0, 0.1) should be positive, got {val}");
    }

    #[test]
    fn test_theta2_even() {
        // theta2 is even: theta2(-z, q) = theta2(z, q)
        let z = 0.5;
        let q = 0.1;
        let t_pos = theta2(z, q).expect("failed");
        let t_neg = theta2(-z, q).expect("failed");
        assert_relative_eq!(t_pos, t_neg, epsilon = 1e-14);
    }

    #[test]
    fn test_theta3_at_zero_q_zero() {
        let val = theta3(0.0, 0.0).expect("theta3 failed");
        assert_relative_eq!(val, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_theta3_at_zero() {
        let val = theta3(0.0, 0.1).expect("theta3 failed");
        assert!(val > 1.0, "theta3(0, 0.1) should be > 1, got {val}");
    }

    #[test]
    fn test_theta3_even() {
        let z = 0.5;
        let q = 0.1;
        let t_pos = theta3(z, q).expect("failed");
        let t_neg = theta3(-z, q).expect("failed");
        assert_relative_eq!(t_pos, t_neg, epsilon = 1e-14);
    }

    #[test]
    fn test_theta4_at_zero_q_zero() {
        let val = theta4(0.0, 0.0).expect("theta4 failed");
        assert_relative_eq!(val, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_theta4_at_zero() {
        let val = theta4(0.0, 0.1).expect("theta4 failed");
        // theta4(0, q) = 1 - 2q + 2q^4 - ...
        assert!(val < 1.0, "theta4(0, 0.1) should be < 1, got {val}");
    }

    #[test]
    fn test_theta4_even() {
        let z = 0.5;
        let q = 0.1;
        let t_pos = theta4(z, q).expect("failed");
        let t_neg = theta4(-z, q).expect("failed");
        assert_relative_eq!(t_pos, t_neg, epsilon = 1e-14);
    }

    #[test]
    fn test_theta_nome_validation() {
        assert!(theta1(0.0, 1.0).is_err());
        assert!(theta1(0.0, -1.0).is_err());
        assert!(theta1(0.0, 1.5).is_err());
    }

    #[test]
    fn test_jacobi_identity() {
        // theta3^4 = theta2^4 + theta4^4
        let q = 0.1;
        let error = jacobi_identity_check(q).expect("identity check failed");
        assert!(error < 1e-10, "Jacobi identity violated with error {error}");
    }

    #[test]
    fn test_jacobi_identity_various_q() {
        for &q in &[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5] {
            let error = jacobi_identity_check(q).expect("identity check failed");
            assert!(
                error < 1e-8,
                "Jacobi identity violated at q={q} with error {error}"
            );
        }
    }

    // ====== AGM tests ======

    #[test]
    fn test_agm_equal() {
        let val = agm(1.0, 1.0).expect("agm failed");
        assert_relative_eq!(val, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_agm_known_value() {
        // AGM(1, sqrt(2)) ~ 1.19814023...
        let val = agm(1.0, 2.0_f64.sqrt()).expect("agm failed");
        assert_relative_eq!(val, 1.198140235, epsilon = 1e-6);
    }

    #[test]
    fn test_agm_symmetric() {
        let a1 = agm(3.0, 7.0).expect("failed");
        let a2 = agm(7.0, 3.0).expect("failed");
        assert_relative_eq!(a1, a2, epsilon = 1e-14);
    }

    #[test]
    fn test_agm_domain_error() {
        assert!(agm(-1.0, 1.0).is_err());
        assert!(agm(1.0, -1.0).is_err());
        assert!(agm(0.0, 1.0).is_err());
    }

    // ====== AGM-based elliptic integral tests ======

    #[test]
    fn test_ellipk_agm_at_zero() {
        let k = ellipk_agm(0.0).expect("failed");
        assert_relative_eq!(k, PI / 2.0, epsilon = 1e-14);
    }

    #[test]
    fn test_ellipk_agm_half() {
        let k = ellipk_agm(0.5).expect("failed");
        assert_relative_eq!(k, 1.854_074_677_301_37, epsilon = 1e-10);
    }

    #[test]
    fn test_ellipk_agm_domain() {
        assert!(ellipk_agm(-0.1).is_err());
        assert!(ellipk_agm(1.0).is_err());
        assert!(ellipk_agm(1.5).is_err());
    }

    #[test]
    fn test_ellipk_agm_matches_standard() {
        // Compare with the standard implementation
        for &m in &[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
            let k_agm = ellipk_agm(m).expect("failed");
            let k_std = crate::elliptic::elliptic_k(m);
            assert_relative_eq!(k_agm, k_std, epsilon = 1e-8,);
        }
    }

    #[test]
    fn test_ellipe_agm_at_zero() {
        let e = ellipe_agm(0.0).expect("failed");
        assert_relative_eq!(e, PI / 2.0, epsilon = 1e-14);
    }

    #[test]
    fn test_ellipe_agm_at_one() {
        let e = ellipe_agm(1.0).expect("failed");
        assert_relative_eq!(e, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_ellipe_agm_half() {
        let e = ellipe_agm(0.5).expect("failed");
        assert_relative_eq!(e, 1.350_643_881_048_18, epsilon = 1e-6);
    }

    #[test]
    fn test_ellipe_agm_domain() {
        assert!(ellipe_agm(-0.1).is_err());
        assert!(ellipe_agm(1.5).is_err());
    }

    // ====== Jacobi amplitude tests ======

    #[test]
    fn test_jacobi_am_zero() {
        let val = jacobi_am(0.0, 0.5).expect("failed");
        assert!(val.abs() < 1e-14, "am(0, 0.5) should be 0, got {val}");
    }

    #[test]
    fn test_jacobi_am_m_zero() {
        // am(u, 0) = u
        let u = 0.7;
        let val = jacobi_am(u, 0.0).expect("failed");
        assert_relative_eq!(val, u, epsilon = 1e-14);
    }

    #[test]
    fn test_jacobi_am_m_one() {
        // am(u, 1) = gd(u) = 2*arctan(exp(u)) - pi/2
        let u = 1.0;
        let val = jacobi_am(u, 1.0).expect("failed");
        let expected = 2.0 * u.exp().atan() - PI / 2.0;
        assert_relative_eq!(val, expected, epsilon = 1e-14);
    }

    #[test]
    fn test_jacobi_am_consistency_with_sn() {
        // sin(am(u, m)) should give the correct sn(u, m)
        // Verified by Newton's method and power series: sn(0.5, 0.3) = 0.47421562271182066
        let u = 0.5;
        let m = 0.3;
        let am_val = jacobi_am(u, m).expect("failed");
        let expected_sn = 0.47421562271182066_f64;
        assert_relative_eq!(am_val.sin(), expected_sn, epsilon = 1e-10);
    }

    #[test]
    fn test_jacobi_am_domain() {
        assert!(jacobi_am(0.5, -0.1).is_err());
        assert!(jacobi_am(0.5, 1.5).is_err());
    }

    // ====== Inverse Jacobi elliptic function tests ======

    #[test]
    fn test_arcsn_zero() {
        let val = arcsn(0.0, 0.5).expect("failed");
        assert!(val.abs() < 1e-14);
    }

    #[test]
    fn test_arcsn_roundtrip() {
        // arcsn(x, m) = F(arcsin(x), m). Verify via am: sin(am(F(arcsin(x), m), m)) = x
        let x = 0.5;
        let m = 0.3;
        let u = arcsn(x, m).expect("failed");
        let am_val = jacobi_am(u, m).expect("am failed");
        assert_relative_eq!(am_val.sin(), x, epsilon = 1e-10);
    }

    #[test]
    fn test_arccn_roundtrip() {
        // arccn(x, m) = F(arccos(x), m). Verify via am: cos(am(u, m)) = x
        let x = 0.7;
        let m = 0.3;
        let u = arccn(x, m).expect("failed");
        let am_val = jacobi_am(u, m).expect("am failed");
        assert_relative_eq!(am_val.cos(), x, epsilon = 1e-10);
    }

    #[test]
    fn test_arcsn_domain() {
        assert!(arcsn(1.5, 0.5).is_err());
        assert!(arcsn(0.5, -0.1).is_err());
    }

    #[test]
    fn test_arcdn_roundtrip() {
        // arcdn(x, m): verify dn = sqrt(1 - m*sn^2) = x
        let x = 0.9;
        let m = 0.3;
        let u = arcdn(x, m).expect("failed");
        let am_val = jacobi_am(u, m).expect("am failed");
        let sn = am_val.sin();
        let dn = (1.0 - m * sn * sn).sqrt();
        assert_relative_eq!(dn, x, epsilon = 1e-8);
    }

    // ====== Bulirsch cel tests ======

    #[test]
    fn test_bulirsch_cel_k() {
        // cel(kc, 1, 1, 1) = K(1 - kc^2)
        let m = 0.5;
        let kc = (1.0 - m).sqrt();
        let val = bulirsch_cel(kc, 1.0, 1.0, 1.0).expect("failed");
        let k = crate::elliptic::elliptic_k(m);
        assert_relative_eq!(val, k, epsilon = 1e-6);
    }

    #[test]
    fn test_bulirsch_cel_domain() {
        assert!(bulirsch_cel(0.0, 1.0, 1.0, 1.0).is_err());
    }

    // ====== Nome conversion tests ======

    #[test]
    fn test_nome_from_m_zero() {
        let q = nome_from_m(0.0).expect("failed");
        assert!(q.abs() < 1e-14);
    }

    #[test]
    fn test_nome_from_m_small() {
        let q = nome_from_m(0.01).expect("failed");
        assert!(q > 0.0 && q < 0.01);
    }

    #[test]
    fn test_nome_from_m_half() {
        let q = nome_from_m(0.5).expect("failed");
        // q(0.5) ~ 0.04322
        assert!((q - 0.04322).abs() < 0.01, "q(0.5) = {q}");
    }

    #[test]
    fn test_nome_roundtrip() {
        // m -> q -> m should be identity
        let m_orig = 0.3;
        let q = nome_from_m(m_orig).expect("failed");
        let m_recovered = m_from_nome(q).expect("failed");
        assert_relative_eq!(m_recovered, m_orig, epsilon = 1e-6);
    }

    #[test]
    fn test_m_from_nome_zero() {
        let m = m_from_nome(0.0).expect("failed");
        assert!(m.abs() < 1e-14);
    }

    // ====== Theta derivative tests ======

    #[test]
    fn test_theta1_derivative_at_zero() {
        // theta1'(0, q) should be nonzero (first derivative of odd function at origin)
        let val = theta_derivative(1, 0.0, 0.1).expect("failed");
        assert!(val.abs() > 0.0, "theta1'(0, 0.1) should be nonzero");
    }

    #[test]
    fn test_theta2_derivative_at_zero() {
        // theta2'(0, q) = 0 (even function)
        let val = theta_derivative(2, 0.0, 0.1).expect("failed");
        assert!(val.abs() < 1e-14, "theta2'(0, 0.1) should be 0, got {val}");
    }

    #[test]
    fn test_theta3_derivative_at_zero() {
        // theta3'(0, q) = 0 (even function)
        let val = theta_derivative(3, 0.0, 0.1).expect("failed");
        assert!(val.abs() < 1e-14, "theta3'(0, 0.1) should be 0, got {val}");
    }

    #[test]
    fn test_theta4_derivative_at_zero() {
        // theta4'(0, q) = 0 (even function)
        let val = theta_derivative(4, 0.0, 0.1).expect("failed");
        assert!(val.abs() < 1e-14, "theta4'(0, 0.1) should be 0, got {val}");
    }

    #[test]
    fn test_theta_derivative_invalid_index() {
        assert!(theta_derivative(0, 0.0, 0.1).is_err());
        assert!(theta_derivative(5, 0.0, 0.1).is_err());
    }

    #[test]
    fn test_theta_derivative_numerical() {
        // Verify derivative via finite differences
        let z = 0.3;
        let q = 0.1;
        let h = 1e-7;
        let deriv_analytic = theta_derivative(1, z, q).expect("failed");
        let t_plus = theta1(z + h, q).expect("failed");
        let t_minus = theta1(z - h, q).expect("failed");
        let deriv_numerical = (t_plus - t_minus) / (2.0 * h);
        assert_relative_eq!(deriv_analytic, deriv_numerical, epsilon = 1e-5);
    }
}
