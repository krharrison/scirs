//! Jacobi Theta Functions
//!
//! This module provides the four classical Jacobi theta functions and related utilities.
//! The theta functions are entire functions (in z) and holomorphic in the upper half-plane
//! (in τ) that play a central role in the theory of elliptic functions, modular forms,
//! string theory, and many areas of mathematical physics.
//!
//! ## Mathematical Definitions
//!
//! With nome `q = exp(iπτ)` where `Im(τ) > 0`, so `0 < q < 1` for real positive τ:
//!
//! ```text
//! θ₁(z|q) = 2 Σ_{n=0}^{∞} (-1)^n  q^{(n+½)²} sin((2n+1)z)
//! θ₂(z|q) = 2 Σ_{n=0}^{∞}          q^{(n+½)²} cos((2n+1)z)
//! θ₃(z|q) = 1 + 2 Σ_{n=1}^{∞}      q^{n²}     cos(2nz)
//! θ₄(z|q) = 1 + 2 Σ_{n=1}^{∞} (-1)^n q^{n²}   cos(2nz)
//! ```
//!
//! ## Symmetry Relations
//!
//! ```text
//! θ₁(z+π, q) = -θ₁(z, q)         (period π)
//! θ₂(z+π, q) = -θ₂(z, q)
//! θ₃(z+π, q) =  θ₃(z, q)
//! θ₄(z+π, q) =  θ₄(z, q)
//! ```
//!
//! ## References
//!
//! - Whittaker & Watson, *A Course of Modern Analysis*, Chapter 21
//! - DLMF §20: Theta Functions  <https://dlmf.nist.gov/20>
//! - Abramowitz & Stegun, Chapter 16

use crate::error::{SpecialError, SpecialResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Convergence tolerance: stop when |term| < THETA_TOL
const THETA_TOL: f64 = 1e-15;

/// Maximum number of summation terms (safety guard; convergence is much earlier)
const THETA_MAX_TERMS: usize = 500;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate that the nome `q` lies in [0, 1).
#[inline]
fn validate_nome(q: f64) -> SpecialResult<()> {
    if q < 0.0 || q >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "theta: nome q must satisfy 0 ≤ q < 1, got q = {q}"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// θ₁
// ---------------------------------------------------------------------------

/// Jacobi theta function θ₁(z, q).
///
/// # Definition
///
/// ```text
/// θ₁(z, q) = 2 Σ_{n=0}^{∞} (-1)^n q^{(n+½)²} sin((2n+1)z)
/// ```
///
/// # Arguments
/// * `z` – complex argument (real for this implementation)
/// * `q` – nome parameter, must satisfy `0 ≤ q < 1`
///
/// # Returns
/// Value of θ₁(z, q) as `f64`.  Returns `f64::NAN` when `q` is out of range.
///
/// # Special Values
/// * θ₁(0, q) = 0 for all q
/// * θ₁(π/2, q) = θ₂(0, q) (relation between theta functions)
///
/// # Examples
/// ```
/// use scirs2_special::theta_functions::theta1;
/// // θ₁(0, q) = 0 for any q
/// assert!((theta1(0.0, 0.3) - 0.0).abs() < 1e-14);
/// ```
pub fn theta1(z: f64, q: f64) -> f64 {
    match theta1_impl(z, q) {
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

fn theta1_impl(z: f64, q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;
    if q < 1e-300 {
        return Ok(0.0);
    }
    let mut result = 0.0_f64;
    for n in 0..THETA_MAX_TERMS {
        let nf = n as f64;
        let exp = (nf + 0.5) * (nf + 0.5);
        let q_pow = q.powf(exp);
        if q_pow < THETA_TOL {
            break;
        }
        let sign = if n % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
        result += sign * q_pow * ((2.0 * nf + 1.0) * z).sin();
    }
    Ok(2.0 * result)
}

// ---------------------------------------------------------------------------
// θ₂
// ---------------------------------------------------------------------------

/// Jacobi theta function θ₂(z, q).
///
/// # Definition
///
/// ```text
/// θ₂(z, q) = 2 Σ_{n=0}^{∞} q^{(n+½)²} cos((2n+1)z)
/// ```
///
/// # Arguments
/// * `z` – real argument
/// * `q` – nome parameter, `0 ≤ q < 1`
///
/// # Returns
/// Value of θ₂(z, q).  Returns `f64::NAN` for invalid `q`.
///
/// # Examples
/// ```
/// use scirs2_special::theta_functions::theta2;
/// let val = theta2(0.0, 0.1);
/// assert!(val > 0.0);
/// ```
pub fn theta2(z: f64, q: f64) -> f64 {
    match theta2_impl(z, q) {
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

fn theta2_impl(z: f64, q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;
    if q < 1e-300 {
        return Ok(0.0);
    }
    let mut result = 0.0_f64;
    for n in 0..THETA_MAX_TERMS {
        let nf = n as f64;
        let exp = (nf + 0.5) * (nf + 0.5);
        let q_pow = q.powf(exp);
        if q_pow < THETA_TOL {
            break;
        }
        result += q_pow * ((2.0 * nf + 1.0) * z).cos();
    }
    Ok(2.0 * result)
}

// ---------------------------------------------------------------------------
// θ₃
// ---------------------------------------------------------------------------

/// Jacobi theta function θ₃(z, q).
///
/// # Definition
///
/// ```text
/// θ₃(z, q) = 1 + 2 Σ_{n=1}^{∞} q^{n²} cos(2nz)
/// ```
///
/// # Arguments
/// * `z` – real argument
/// * `q` – nome parameter, `0 ≤ q < 1`
///
/// # Returns
/// Value of θ₃(z, q).  Returns `f64::NAN` for invalid `q`.
///
/// # Special Values
/// * θ₃(0, 0) = 1
/// * θ₃(z, q) ≥ 1 − 2q/(1−q) for small q
///
/// # Examples
/// ```
/// use scirs2_special::theta_functions::theta3;
/// assert!((theta3(0.0, 0.0) - 1.0).abs() < 1e-14);
/// ```
pub fn theta3(z: f64, q: f64) -> f64 {
    match theta3_impl(z, q) {
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

fn theta3_impl(z: f64, q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;
    if q < 1e-300 {
        return Ok(1.0);
    }
    let mut result = 1.0_f64;
    for n in 1..=THETA_MAX_TERMS {
        let nf = n as f64;
        let q_pow = q.powf(nf * nf);
        if q_pow < THETA_TOL {
            break;
        }
        result += 2.0 * q_pow * (2.0 * nf * z).cos();
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// θ₄
// ---------------------------------------------------------------------------

/// Jacobi theta function θ₄(z, q).
///
/// # Definition
///
/// ```text
/// θ₄(z, q) = 1 + 2 Σ_{n=1}^{∞} (-1)^n q^{n²} cos(2nz)
/// ```
///
/// # Arguments
/// * `z` – real argument
/// * `q` – nome parameter, `0 ≤ q < 1`
///
/// # Returns
/// Value of θ₄(z, q).  Returns `f64::NAN` for invalid `q`.
///
/// # Examples
/// ```
/// use scirs2_special::theta_functions::theta4;
/// assert!((theta4(0.0, 0.0) - 1.0).abs() < 1e-14);
/// ```
pub fn theta4(z: f64, q: f64) -> f64 {
    match theta4_impl(z, q) {
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

fn theta4_impl(z: f64, q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;
    if q < 1e-300 {
        return Ok(1.0);
    }
    let mut result = 1.0_f64;
    for n in 1..=THETA_MAX_TERMS {
        let nf = n as f64;
        let sign = if n % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
        let q_pow = q.powf(nf * nf);
        if q_pow < THETA_TOL {
            break;
        }
        result += 2.0 * sign * q_pow * (2.0 * nf * z).cos();
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Derivatives
// ---------------------------------------------------------------------------

/// Derivative of θ₁ with respect to z: dθ₁/dz.
///
/// # Definition
///
/// ```text
/// dθ₁/dz = 2 Σ_{n=0}^{∞} (-1)^n (2n+1) q^{(n+½)²} cos((2n+1)z)
/// ```
///
/// # Arguments
/// * `z` – real argument
/// * `q` – nome parameter, `0 ≤ q < 1`
///
/// # Returns
/// Value of dθ₁/dz at (z, q).  Returns `f64::NAN` for invalid `q`.
///
/// # Examples
/// ```
/// use scirs2_special::theta_functions::theta1_derivative;
/// // derivative is positive near z=0 for small q
/// let d = theta1_derivative(0.0, 0.1);
/// assert!(d > 0.0);
/// ```
pub fn theta1_derivative(z: f64, q: f64) -> f64 {
    match theta1_derivative_impl(z, q) {
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

fn theta1_derivative_impl(z: f64, q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;
    if q < 1e-300 {
        return Ok(0.0);
    }
    let mut result = 0.0_f64;
    for n in 0..THETA_MAX_TERMS {
        let nf = n as f64;
        let exp = (nf + 0.5) * (nf + 0.5);
        let q_pow = q.powf(exp);
        if q_pow.abs() < THETA_TOL {
            break;
        }
        let sign = if n % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
        let freq = 2.0 * nf + 1.0;
        result += sign * freq * q_pow * (freq * z).cos();
    }
    Ok(2.0 * result)
}

/// Derivative of θ₂ with respect to z: dθ₂/dz.
///
/// # Definition
///
/// ```text
/// dθ₂/dz = -2 Σ_{n=0}^{∞} (2n+1) q^{(n+½)²} sin((2n+1)z)
/// ```
///
/// # Arguments
/// * `z` – real argument
/// * `q` – nome parameter, `0 ≤ q < 1`
///
/// # Returns
/// Value of dθ₂/dz.  Returns `f64::NAN` for invalid `q`.
///
/// # Examples
/// ```
/// use scirs2_special::theta_functions::theta2_derivative;
/// // At z=0, θ₂ is at a maximum so derivative = 0
/// assert!((theta2_derivative(0.0, 0.1)).abs() < 1e-14);
/// ```
pub fn theta2_derivative(z: f64, q: f64) -> f64 {
    match theta2_derivative_impl(z, q) {
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

fn theta2_derivative_impl(z: f64, q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;
    if q < 1e-300 {
        return Ok(0.0);
    }
    let mut result = 0.0_f64;
    for n in 0..THETA_MAX_TERMS {
        let nf = n as f64;
        let exp = (nf + 0.5) * (nf + 0.5);
        let q_pow = q.powf(exp);
        if q_pow < THETA_TOL {
            break;
        }
        let freq = 2.0 * nf + 1.0;
        result += freq * q_pow * (freq * z).sin();
    }
    Ok(-2.0 * result)
}

/// Derivative of θ₃ with respect to z: dθ₃/dz.
///
/// # Definition
///
/// ```text
/// dθ₃/dz = -4 Σ_{n=1}^{∞} n q^{n²} sin(2nz)
/// ```
///
/// # Arguments
/// * `z` – real argument
/// * `q` – nome parameter, `0 ≤ q < 1`
///
/// # Returns
/// Value of dθ₃/dz.  Returns `f64::NAN` for invalid `q`.
///
/// # Examples
/// ```
/// use scirs2_special::theta_functions::theta3_derivative;
/// // At z=0, θ₃ is at a maximum so derivative = 0
/// assert!((theta3_derivative(0.0, 0.2)).abs() < 1e-14);
/// ```
pub fn theta3_derivative(z: f64, q: f64) -> f64 {
    match theta3_derivative_impl(z, q) {
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

fn theta3_derivative_impl(z: f64, q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;
    if q < 1e-300 {
        return Ok(0.0);
    }
    let mut result = 0.0_f64;
    for n in 1..=THETA_MAX_TERMS {
        let nf = n as f64;
        let q_pow = q.powf(nf * nf);
        if q_pow < THETA_TOL {
            break;
        }
        result += nf * q_pow * (2.0 * nf * z).sin();
    }
    Ok(-4.0 * result)
}

/// Derivative of θ₄ with respect to z: dθ₄/dz.
///
/// # Definition
///
/// ```text
/// dθ₄/dz = -4 Σ_{n=1}^{∞} (-1)^n n q^{n²} sin(2nz)
/// ```
///
/// # Arguments
/// * `z` – real argument
/// * `q` – nome parameter, `0 ≤ q < 1`
///
/// # Returns
/// Value of dθ₄/dz.  Returns `f64::NAN` for invalid `q`.
///
/// # Examples
/// ```
/// use scirs2_special::theta_functions::theta4_derivative;
/// // At z=0, θ₄ is at a maximum so derivative = 0
/// assert!((theta4_derivative(0.0, 0.2)).abs() < 1e-14);
/// ```
pub fn theta4_derivative(z: f64, q: f64) -> f64 {
    match theta4_derivative_impl(z, q) {
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

fn theta4_derivative_impl(z: f64, q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;
    if q < 1e-300 {
        return Ok(0.0);
    }
    let mut result = 0.0_f64;
    for n in 1..=THETA_MAX_TERMS {
        let nf = n as f64;
        let sign = if n % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
        let q_pow = q.powf(nf * nf);
        if q_pow < THETA_TOL {
            break;
        }
        result += sign * nf * q_pow * (2.0 * nf * z).sin();
    }
    Ok(-4.0 * result)
}

// ---------------------------------------------------------------------------
// Nome ↔ Modulus conversions
// ---------------------------------------------------------------------------

/// Compute the elliptic nome `q` from the elliptic modulus `k`.
///
/// Uses the AGM (arithmetic-geometric mean) to compute `K(k)` and `K'(k)`,
/// then `q = exp(-π K'/K)`.
///
/// # Arguments
/// * `k` – elliptic modulus, `0 ≤ k < 1`
///
/// # Returns
/// Nome `q = exp(-π K'(k) / K(k))`, in `[0, 1)`.
/// Returns `f64::NAN` for `k` outside `[0, 1)`.
///
/// # Examples
/// ```
/// use scirs2_special::theta_functions::q_from_k;
/// // k=0 → K=π/2, K'=∞ → q=0
/// assert!((q_from_k(0.0) - 0.0).abs() < 1e-14);
/// // k approaching 1 → q approaching 1
/// let q = q_from_k(0.9);
/// assert!(q > 0.0 && q < 1.0);
/// ```
pub fn q_from_k(k: f64) -> f64 {
    if k < 0.0 || k >= 1.0 {
        return f64::NAN;
    }
    if k == 0.0 {
        return 0.0;
    }
    let k_prime = (1.0 - k * k).sqrt();
    let big_k = agm_elliptic_k(k);
    let big_k_prime = agm_elliptic_k(k_prime);
    (-std::f64::consts::PI * big_k_prime / big_k).exp()
}

/// Compute the elliptic modulus `k` from the nome `q`.
///
/// Uses the identity `k = θ₂(0,q)² / θ₃(0,q)²`.
///
/// # Arguments
/// * `q` – nome parameter, `0 ≤ q < 1`
///
/// # Returns
/// Elliptic modulus `k`, in `[0, 1)`.
/// Returns `f64::NAN` for `q` outside `[0, 1)`.
///
/// # Examples
/// ```
/// use scirs2_special::theta_functions::{q_from_k, k_from_q};
/// // Round-trip test
/// let k_orig = 0.7_f64;
/// let q = q_from_k(k_orig);
/// let k_back = k_from_q(q);
/// assert!((k_back - k_orig).abs() < 1e-12);
/// ```
pub fn k_from_q(q: f64) -> f64 {
    if q < 0.0 || q >= 1.0 {
        return f64::NAN;
    }
    if q == 0.0 {
        return 0.0;
    }
    let t2 = theta2(0.0, q);
    let t3 = theta3(0.0, q);
    if t3.abs() < 1e-300 {
        return f64::NAN;
    }
    (t2 / t3) * (t2 / t3)
}

// ---------------------------------------------------------------------------
// Internal: AGM-based complete elliptic integral K(k)
// ---------------------------------------------------------------------------

/// Complete elliptic integral of the first kind K(k) via AGM.
///
/// K(k) = π / (2 · AGM(1, sqrt(1−k²)))
fn agm_elliptic_k(k: f64) -> f64 {
    if k <= 0.0 {
        return std::f64::consts::FRAC_PI_2;
    }
    if k >= 1.0 {
        return f64::INFINITY;
    }
    let k_prime_sq = 1.0 - k * k;
    let b = k_prime_sq.sqrt();
    let agm_val = agm(1.0, b);
    std::f64::consts::FRAC_PI_2 / agm_val
}

/// Arithmetic-Geometric Mean of `a` and `b`.
fn agm(mut a: f64, mut b: f64) -> f64 {
    for _ in 0..100 {
        let a_new = (a + b) * 0.5;
        let b_new = (a * b).sqrt();
        if (a_new - b_new).abs() < 1e-15 * a_new.abs() {
            return a_new;
        }
        a = a_new;
        b = b_new;
    }
    (a + b) * 0.5
}

// ---------------------------------------------------------------------------
// Logarithmic derivative (useful for elliptic functions)
// ---------------------------------------------------------------------------

/// Logarithmic derivative of θ₁: (dθ₁/dz) / θ₁(z, q).
///
/// This is related to the Jacobi zeta function and the Weierstrass ζ function.
///
/// # Arguments
/// * `z` – real argument (must not be at a zero of θ₁)
/// * `q` – nome parameter, `0 ≤ q < 1`
///
/// # Returns
/// Value of (d/dz) ln θ₁(z, q).  Returns `f64::NAN` if θ₁(z,q) ≈ 0 or q is invalid.
///
/// # Examples
/// ```
/// use scirs2_special::theta_functions::theta1_log_derivative;
/// let ld = theta1_log_derivative(0.5, 0.1);
/// assert!(ld.is_finite());
/// ```
pub fn theta1_log_derivative(z: f64, q: f64) -> f64 {
    let t1 = theta1(z, q);
    let dt1 = theta1_derivative(z, q);
    if t1.abs() < 1e-300 {
        return f64::NAN;
    }
    dt1 / t1
}

// ---------------------------------------------------------------------------
// Modular transformation: q → exp(iπ/τ) nome
// ---------------------------------------------------------------------------

/// Compute the complementary nome q' from q.
///
/// If `q = exp(-π K'/K)`, then `q' = exp(-π K/K')`.
/// This satisfies the Landen transformation identity.
///
/// # Arguments
/// * `q` – nome parameter, `0 ≤ q < 1`
///
/// # Returns
/// Complementary nome `q'`.  Returns `f64::NAN` for invalid `q`.
///
/// # Examples
/// ```
/// use scirs2_special::theta_functions::complementary_nome;
/// let q  = 0.1_f64;
/// let qp = complementary_nome(q);
/// assert!(qp > 0.0 && qp < 1.0);
/// ```
pub fn complementary_nome(q: f64) -> f64 {
    if q < 0.0 || q >= 1.0 {
        return f64::NAN;
    }
    if q == 0.0 {
        // q=0 means K'/K → ∞, so complementary nome → 1 (singular)
        return 1.0;
    }
    // k from q, then k' from k, then q' from k'
    let k = k_from_q(q);
    if !k.is_finite() {
        return f64::NAN;
    }
    let k_prime = (1.0 - k * k).max(0.0).sqrt();
    q_from_k(k_prime)
}

// ---------------------------------------------------------------------------
// Jacobi elliptic functions via theta quotients
// ---------------------------------------------------------------------------

/// Jacobi elliptic sn(u, k) via theta function quotients.
///
/// ```text
/// sn(u, k) = (θ₃(0,q) / θ₂(0,q)) · (θ₁(z,q) / θ₄(z,q))
/// ```
/// where `z = u / θ₃(0,q)²` and `q = q_from_k(k)`.
///
/// # Arguments
/// * `u` – real argument
/// * `k` – elliptic modulus, `0 ≤ k < 1`
///
/// # Returns
/// Jacobi elliptic function sn(u, k).
///
/// # Examples
/// ```
/// use scirs2_special::theta_functions::jacobi_sn;
/// // sn(0, k) = 0
/// assert!((jacobi_sn(0.0, 0.5)).abs() < 1e-14);
/// ```
pub fn jacobi_sn(u: f64, k: f64) -> f64 {
    if k < 0.0 || k >= 1.0 {
        return f64::NAN;
    }
    if k == 0.0 {
        return u.sin();
    }
    let q = q_from_k(k);
    let t3_0 = theta3(0.0, q);
    if t3_0.abs() < 1e-300 {
        return f64::NAN;
    }
    // The argument transformation: z = π u / (2 K(k)) = π u / (π θ₃(0,q)²) = u/θ₃(0,q)²
    // Standard formula: K = π/2 · θ₃(0,q)²
    let big_k = std::f64::consts::FRAC_PI_2 * t3_0 * t3_0;
    let z = std::f64::consts::FRAC_PI_2 * u / big_k;

    let t1_z = theta1(z, q);
    let t4_z = theta4(z, q);
    let t2_0 = theta2(0.0, q);

    if t2_0.abs() < 1e-300 || t4_z.abs() < 1e-300 {
        return f64::NAN;
    }

    (t3_0 / t2_0) * (t1_z / t4_z)
}

/// Jacobi elliptic cn(u, k) via theta function quotients.
///
/// ```text
/// cn(u, k) = (θ₄(0,q) / θ₂(0,q)) · (θ₂(z,q) / θ₄(z,q))
/// ```
///
/// # Arguments
/// * `u` – real argument
/// * `k` – elliptic modulus, `0 ≤ k < 1`
///
/// # Returns
/// Jacobi elliptic function cn(u, k).
///
/// # Examples
/// ```
/// use scirs2_special::theta_functions::jacobi_cn;
/// // cn(0, k) = 1
/// assert!((jacobi_cn(0.0, 0.5) - 1.0).abs() < 1e-12);
/// ```
pub fn jacobi_cn(u: f64, k: f64) -> f64 {
    if k < 0.0 || k >= 1.0 {
        return f64::NAN;
    }
    if k == 0.0 {
        return u.cos();
    }
    let q = q_from_k(k);
    let t3_0 = theta3(0.0, q);
    if t3_0.abs() < 1e-300 {
        return f64::NAN;
    }
    let big_k = std::f64::consts::FRAC_PI_2 * t3_0 * t3_0;
    let z = std::f64::consts::FRAC_PI_2 * u / big_k;

    let t2_z = theta2(z, q);
    let t4_z = theta4(z, q);
    let t2_0 = theta2(0.0, q);
    let t4_0 = theta4(0.0, q);

    if t2_0.abs() < 1e-300 || t4_z.abs() < 1e-300 {
        return f64::NAN;
    }

    (t4_0 / t2_0) * (t2_z / t4_z)
}

/// Jacobi elliptic dn(u, k) via theta function quotients.
///
/// ```text
/// dn(u, k) = (θ₄(0,q) / θ₃(0,q)) · (θ₃(z,q) / θ₄(z,q))
/// ```
///
/// # Arguments
/// * `u` – real argument
/// * `k` – elliptic modulus, `0 ≤ k < 1`
///
/// # Returns
/// Jacobi elliptic function dn(u, k).
///
/// # Examples
/// ```
/// use scirs2_special::theta_functions::jacobi_dn;
/// // dn(0, k) = 1
/// assert!((jacobi_dn(0.0, 0.7) - 1.0).abs() < 1e-12);
/// ```
pub fn jacobi_dn(u: f64, k: f64) -> f64 {
    if k < 0.0 || k >= 1.0 {
        return f64::NAN;
    }
    if k == 0.0 {
        return 1.0;
    }
    let q = q_from_k(k);
    let t3_0 = theta3(0.0, q);
    if t3_0.abs() < 1e-300 {
        return f64::NAN;
    }
    let big_k = std::f64::consts::FRAC_PI_2 * t3_0 * t3_0;
    let z = std::f64::consts::FRAC_PI_2 * u / big_k;

    let t3_z = theta3(z, q);
    let t4_z = theta4(z, q);
    let t4_0 = theta4(0.0, q);

    if t3_0.abs() < 1e-300 || t4_z.abs() < 1e-300 {
        return f64::NAN;
    }

    (t4_0 / t3_0) * (t3_z / t4_z)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    // --- θ₁ ---

    #[test]
    fn test_theta1_zero_z() {
        // θ₁(0, q) = 0 by the sine-series (all terms vanish)
        for &q in &[0.0, 0.1, 0.3, 0.5, 0.7] {
            assert!(
                theta1(0.0, q).abs() < EPS,
                "theta1(0, {q}) should be 0, got {}",
                theta1(0.0, q)
            );
        }
    }

    #[test]
    fn test_theta1_odd_symmetry() {
        // θ₁(-z, q) = -θ₁(z, q)
        let q = 0.3;
        for &z in &[0.2, 0.5, 1.0, 1.5] {
            let pos = theta1(z, q);
            let neg = theta1(-z, q);
            assert!(
                (pos + neg).abs() < EPS,
                "theta1 odd symmetry failed at z={z}: {pos} + {neg} = {}",
                pos + neg
            );
        }
    }

    // --- θ₂ ---

    #[test]
    fn test_theta2_even_symmetry() {
        // θ₂(-z, q) = θ₂(z, q)
        let q = 0.2;
        for &z in &[0.1, 0.4, 0.8, 1.2] {
            let pos = theta2(z, q);
            let neg = theta2(-z, q);
            assert!(
                (pos - neg).abs() < EPS,
                "theta2 even symmetry failed at z={z}"
            );
        }
    }

    #[test]
    fn test_theta2_positive_at_zero() {
        for &q in &[0.05, 0.2, 0.5] {
            assert!(theta2(0.0, q) > 0.0, "theta2(0, {q}) should be positive");
        }
    }

    // --- θ₃ ---

    #[test]
    fn test_theta3_q0() {
        assert!((theta3(0.0, 0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_theta3_even_symmetry() {
        let q = 0.15;
        for &z in &[0.3, 0.7, 1.1] {
            assert!(
                (theta3(z, q) - theta3(-z, q)).abs() < EPS,
                "theta3 even symmetry failed at z={z}"
            );
        }
    }

    // --- θ₄ ---

    #[test]
    fn test_theta4_q0() {
        assert!((theta4(0.0, 0.0) - 1.0).abs() < EPS);
    }

    // --- Derivatives ---

    #[test]
    fn test_theta1_derivative_z0() {
        // θ₁'(0,q) > 0 for q > 0
        let d = theta1_derivative(0.0, 0.1);
        assert!(d > 0.0);
    }

    #[test]
    fn test_theta2_derivative_z0() {
        // θ₂'(0,q) = 0 (maximum at z=0)
        assert!(theta2_derivative(0.0, 0.3).abs() < EPS);
    }

    #[test]
    fn test_theta3_derivative_z0() {
        // θ₃'(0,q) = 0 (maximum at z=0)
        assert!(theta3_derivative(0.0, 0.2).abs() < EPS);
    }

    // --- Nome conversion ---

    #[test]
    fn test_q_from_k_zero() {
        assert!((q_from_k(0.0) - 0.0).abs() < EPS);
    }

    #[test]
    fn test_k_from_q_zero() {
        assert!((k_from_q(0.0) - 0.0).abs() < EPS);
    }

    #[test]
    fn test_k_q_roundtrip() {
        for &k in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let q = q_from_k(k);
            let k2 = k_from_q(q);
            assert!(
                (k2 - k).abs() < 1e-10,
                "round-trip failed for k={k}: got {k2}"
            );
        }
    }

    // --- Jacobi elliptic via theta ---

    #[test]
    fn test_jacobi_sn_zero() {
        assert!(jacobi_sn(0.0, 0.5).abs() < EPS);
    }

    #[test]
    fn test_jacobi_cn_zero() {
        assert!((jacobi_cn(0.0, 0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jacobi_dn_zero() {
        assert!((jacobi_dn(0.0, 0.7) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jacobi_identity_sn2_cn2() {
        // sn²(u,k) + cn²(u,k) = 1
        let k = 0.6_f64;
        for &u in &[0.3, 0.8, 1.2] {
            let sn = jacobi_sn(u, k);
            let cn = jacobi_cn(u, k);
            assert!(
                (sn * sn + cn * cn - 1.0).abs() < 1e-10,
                "sn² + cn² ≠ 1 at u={u}, k={k}: sn={sn}, cn={cn}"
            );
        }
    }

    #[test]
    fn test_invalid_q() {
        assert!(theta1(0.0, -0.1).is_nan());
        assert!(theta1(0.0, 1.0).is_nan());
        assert!(theta2(0.0, 1.5).is_nan());
    }
}
