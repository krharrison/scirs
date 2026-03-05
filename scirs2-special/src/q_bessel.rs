//! q-Bessel Functions
//!
//! This module implements q-analogs of classical Bessel functions. The q-Bessel functions
//! arise naturally in the theory of basic hypergeometric series and q-orthogonal polynomials.
//! They have applications in:
//!
//! - q-Fourier analysis and q-integral transforms
//! - Quantum mechanics on q-deformed spaces
//! - Combinatorics of lattice paths
//! - Mathematical physics of quantum groups
//!
//! ## Mathematical Background
//!
//! ### Jackson's q-Bessel Functions
//!
//! Jackson introduced three families of q-Bessel functions. The most commonly used is the
//! second Jackson q-Bessel function (also denoted J_ν^(2)), defined by:
//! ```text
//! J_ν^(2)(x; q) = (q^{ν+1}; q)_∞ / (q; q)_∞
//!                 * sum_{k=0}^∞ (-1)^k * q^{k(k+1)/2} * (x/2)^{ν+2k}
//!                               / ((q; q)_k * (q^{ν+1}; q)_k)
//! ```
//!
//! The factor (q^{ν+1}; q)_∞ / (q; q)_∞ ensures proper normalization.
//!
//! ### Alternative Series (First Jackson q-Bessel, J_ν^(1))
//!
//! The first Jackson q-Bessel function is:
//! ```text
//! J_ν^(1)(x; q) = (q^{ν+1}; q)_∞ / (q; q)_∞ * (x/2)^ν
//!                 * sum_{k=0}^∞ (-1)^k * (x/2)^{2k} / ((q; q)_k * (q^{ν+1}; q)_k)
//! ```
//!
//! Note: J_ν^(1)(x(1-q); q) → J_ν(x) as q → 1.
//!
//! ### Big q-Bessel Function
//!
//! The big q-Bessel function (Stanton's extension) is:
//! ```text
//! J_ν(x; q) = x^ν * sum_{k=0}^∞ (-1)^k * q^{k^2} * x^{2k}
//!                   / ((q^2; q^2)_k * (q^{2ν+2}; q^2)_k)
//! ```
//!
//! ### Limiting Behavior
//!
//! As q → 1:
//! - Jackson J_ν^(2)(x(1-q); q) → J_ν(x)  (classical Bessel J_ν)
//! - The classical Bessel function satisfies J_ν(x) = sum_{k=0}^∞ (-1)^k (x/2)^{ν+2k} / (k! Γ(ν+k+1))
//!
//! ## References
//!
//! - Jackson, F.H. (1905). On q-functions and a certain difference operator. *Trans. Roy. Soc. Edinburgh*, 46, 253-281.
//! - Gasper, G. & Rahman, M. (2004). *Basic Hypergeometric Series*, 2nd ed. Cambridge.
//! - Ismail, M.E.H. (2005). *Classical and Quantum Orthogonal Polynomials in One Variable*. Cambridge.
//! - Koornwinder, T.H. & Swarttouw, R.F. (1992). On q-analogues of the Fourier and Hankel transforms.
//!   *Trans. Amer. Math. Soc.*, 333(1), 445-461.

use crate::error::{SpecialError, SpecialResult};
use crate::q_analogs::{q_pochhammer, q_pochhammer_inf};

/// Maximum number of terms in series summation
const MAX_TERMS: usize = 500;

/// Convergence tolerance
const TOL: f64 = 1e-14;

// ============================================================================
// Utility: q-Pochhammer for finite and infinite products
// ============================================================================

/// Computes (q^{nu+1}; q)_n for given nu and n terms.
fn q_poch_nu1(nu: f64, q: f64, n: usize) -> SpecialResult<f64> {
    let a = q.powf(nu + 1.0);
    q_pochhammer(a, q, n)
}

// ============================================================================
// Jackson's q-Bessel Function J_nu^(2)
// ============================================================================

/// Computes Jackson's second q-Bessel function J_ν^(2)(x; q).
///
/// # Definition
///
/// ```text
/// J_ν^(2)(x; q) = (q^{ν+1}; q)_∞ / (q; q)_∞
///                 * sum_{k=0}^∞ (-1)^k * q^{k(k+1)/2} * (x/2)^{ν+2k}
///                               / ((q; q)_k * (q^{ν+1}; q)_k)
/// ```
///
/// This is related to the classical Bessel function via:
/// ```text
/// lim_{q→1} J_ν^(2)(x(1-q); q) = J_ν(x)
/// ```
///
/// # Arguments
///
/// * `nu` - Order parameter (real, ν > -1 for most convergence results)
/// * `x`  - Argument (x ≥ 0)
/// * `q`  - Deformation parameter (0 < q < 1)
///
/// # Returns
///
/// Value of J_ν^(2)(x; q) or an error.
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_bessel::jackson_j_nu;
///
/// // J_0^(2)(0; q) = 1 for any q ∈ (0,1)
/// let val = jackson_j_nu(0.0, 0.0, 0.5).unwrap();
/// assert!((val - 1.0).abs() < 1e-10);
/// ```
pub fn jackson_j_nu(nu: f64, x: f64, q: f64) -> SpecialResult<f64> {
    if q <= 0.0 || q >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "jackson_j_nu: q must satisfy 0 < q < 1, got q = {q}"
        )));
    }

    if x < 0.0 {
        return Err(SpecialError::DomainError(format!(
            "jackson_j_nu: x must be non-negative, got x = {x}"
        )));
    }

    // Prefactor: (q^{ν+1}; q)_∞ / (q; q)_∞
    let a_nu = q.powf(nu + 1.0);
    let poch_nu = q_pochhammer_inf(a_nu, q)?;
    let poch_q = q_pochhammer_inf(q, q)?;

    if poch_q.abs() < 1e-300 {
        return Err(SpecialError::ComputationError(
            "jackson_j_nu: (q;q)_∞ is too close to zero".to_string(),
        ));
    }

    let prefactor = poch_nu / poch_q;
    let half_x = x / 2.0;

    // Series: sum_{k=0}^∞ (-1)^k * q^{k(k+1)/2} * (x/2)^{ν+2k} / ((q;q)_k * (q^{ν+1};q)_k)
    let mut sum = 0.0f64;
    let mut q_kk1_2 = 1.0f64; // q^{k(k+1)/2}, starts at q^0 = 1 (k=0)
    let mut q_pow = 1.0f64; // q^k
    let mut poch_q_k = 1.0f64; // (q;q)_k, starts at 1
    let mut poch_nu_k = 1.0f64; // (q^{ν+1};q)_k, starts at 1
    let mut half_x_pow = half_x.powf(nu); // (x/2)^{ν+2k}

    for k in 0..MAX_TERMS {
        if k > 0 {
            // Update q^{k(k+1)/2}: at step k, exponent = k(k+1)/2 = (k-1)k/2 + k
            q_kk1_2 *= q.powi(k as i32); // multiply by q^k
            q_pow *= q;

            // Update (q;q)_k = (q;q)_{k-1} * (1 - q^k)
            poch_q_k *= 1.0 - q_pow;

            // Update (q^{ν+1};q)_k = (q^{ν+1};q)_{k-1} * (1 - q^{ν+k})
            poch_nu_k *= 1.0 - a_nu * q.powi((k - 1) as i32);

            // Update (x/2)^{ν+2k}
            half_x_pow *= half_x * half_x;
        }

        let denom = poch_q_k * poch_nu_k;
        if denom.abs() < 1e-300 {
            break;
        }

        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let term = sign * q_kk1_2 * half_x_pow / denom;
        sum += term;

        if term.abs() < TOL * sum.abs().max(1e-300) && k > 2 {
            break;
        }

        if !sum.is_finite() {
            return Err(SpecialError::OverflowError(
                "jackson_j_nu: series diverged".to_string(),
            ));
        }
    }

    let result = prefactor * sum;

    if !result.is_finite() {
        return Err(SpecialError::OverflowError(
            "jackson_j_nu: result is not finite".to_string(),
        ));
    }

    Ok(result)
}

// ============================================================================
// Big q-Bessel Function
// ============================================================================

/// Computes the big q-Bessel function J_ν(x; q) (Stanton's version).
///
/// # Definition
///
/// The big q-Bessel function is defined by:
/// ```text
/// J_ν(x; q) = (q^{ν+1}; q)_∞ / (q; q)_∞ * (x/2)^ν
///             * sum_{k=0}^∞ (-1)^k * (x/2)^{2k} * q^{k(ν+k)}
///                           / ((q; q)_k * (q^{ν+1}; q)_k)
/// ```
///
/// This differs from Jackson's J_ν^(2) in the q-exponent pattern.
///
/// # Arguments
///
/// * `nu` - Order parameter (ν > -1)
/// * `x`  - Argument
/// * `q`  - Deformation parameter (0 < q < 1)
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_bessel::big_q_bessel;
///
/// // big_q_bessel(0, 0, q) = 1
/// let val = big_q_bessel(0.0, 0.0, 0.5).unwrap();
/// assert!((val - 1.0).abs() < 1e-10);
/// ```
pub fn big_q_bessel(nu: f64, x: f64, q: f64) -> SpecialResult<f64> {
    if q <= 0.0 || q >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "big_q_bessel: q must satisfy 0 < q < 1, got q = {q}"
        )));
    }

    if x < 0.0 {
        return Err(SpecialError::DomainError(format!(
            "big_q_bessel: x must be non-negative, got x = {x}"
        )));
    }

    let a_nu = q.powf(nu + 1.0);

    // Prefactor
    let poch_nu = q_pochhammer_inf(a_nu, q)?;
    let poch_q = q_pochhammer_inf(q, q)?;

    if poch_q.abs() < 1e-300 {
        return Err(SpecialError::ComputationError(
            "big_q_bessel: (q;q)_∞ is too close to zero".to_string(),
        ));
    }

    let half_x = x / 2.0;
    let half_x_nu = half_x.powf(nu);
    let prefactor = poch_nu / poch_q * half_x_nu;

    // Series: sum_{k=0}^∞ (-1)^k (x/2)^{2k} q^{k(ν+k)} / ((q;q)_k (q^{ν+1};q)_k)
    let mut sum = 0.0f64;
    let mut poch_q_k = 1.0f64;
    let mut poch_nu_k = 1.0f64;
    let mut half_x2_pow = 1.0f64; // (x/2)^{2k}
    let half_x2 = half_x * half_x;
    let mut q_k_nu_k = 1.0f64; // q^{k(ν+k)}
    let mut q_pow_k = 1.0f64; // q^k
    let mut q_pow_nu = a_nu; // q^{ν+1}

    for k in 0..MAX_TERMS {
        if k > 0 {
            q_pow_k *= q;
            // q^{k(ν+k)} = q^{(k-1)(ν+k-1)} * q^{ν+2k-1}
            q_k_nu_k *= q.powf(nu + (2 * k - 1) as f64);

            // (q;q)_k = (q;q)_{k-1} * (1 - q^k)
            poch_q_k *= 1.0 - q_pow_k;
            // (q^{ν+1};q)_k = (q^{ν+1};q)_{k-1} * (1 - q^{ν+k})
            poch_nu_k *= 1.0 - q_pow_nu * q.powi((k - 1) as i32);

            half_x2_pow *= half_x2;
        }

        let denom = poch_q_k * poch_nu_k;
        if denom.abs() < 1e-300 {
            break;
        }

        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let term = sign * half_x2_pow * q_k_nu_k / denom;
        sum += term;

        if term.abs() < TOL * sum.abs().max(1e-300) && k > 2 {
            break;
        }

        if !sum.is_finite() {
            return Err(SpecialError::OverflowError(
                "big_q_bessel: series diverged".to_string(),
            ));
        }
    }

    let result = prefactor * sum;

    if !result.is_finite() {
        return Err(SpecialError::OverflowError(
            "big_q_bessel: result is not finite".to_string(),
        ));
    }

    Ok(result)
}

// ============================================================================
// General q-Bessel Power Series
// ============================================================================

/// Computes the q-Bessel function via a direct power series with `n_terms` terms.
///
/// # Definition
///
/// The general q-Bessel series of order ν is:
/// ```text
/// J_ν^q-series(x; q) = sum_{k=0}^{n_terms-1}
///     (-1)^k * (x/2)^{ν+2k} / ((q; q)_k * (q^{ν+1}; q)_k)
/// ```
///
/// This is the "bare" series without the infinite product prefactor, useful for
/// controlled truncation and comparison with other implementations.
///
/// # Arguments
///
/// * `nu`     - Order (ν > -1)
/// * `x`      - Argument (x ≥ 0)
/// * `q`      - Deformation parameter (0 < q < 1)
/// * `n_terms` - Number of series terms to include
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_bessel::q_bessel_series;
///
/// // Small x: first term dominates
/// let val = q_bessel_series(0.0, 0.1, 0.5, 20).unwrap();
/// assert!(val.is_finite());
/// ```
pub fn q_bessel_series(nu: f64, x: f64, q: f64, n_terms: usize) -> SpecialResult<f64> {
    if q <= 0.0 || q >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "q_bessel_series: q must satisfy 0 < q < 1, got q = {q}"
        )));
    }

    if x < 0.0 {
        return Err(SpecialError::DomainError(format!(
            "q_bessel_series: x must be non-negative, got x = {x}"
        )));
    }

    if n_terms == 0 {
        return Ok(0.0);
    }

    let half_x = x / 2.0;
    let mut sum = 0.0f64;
    let mut half_x_pow = half_x.powf(nu); // (x/2)^{ν+2k}
    let half_x2 = half_x * half_x;
    let mut q_pow = 1.0f64; // q^k
    let a_nu = q.powf(nu + 1.0);

    for k in 0..n_terms {
        let poch_q_k = q_pochhammer(q, q, k)?;
        let poch_nu_k = q_poch_nu1(nu, q, k)?;

        let denom = poch_q_k * poch_nu_k;
        if denom.abs() < 1e-300 {
            break;
        }

        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let term = sign * half_x_pow / denom;
        sum += term;

        // Advance (x/2)^{ν+2k} and q^k
        half_x_pow *= half_x2;
        q_pow *= q;

        let _ = (a_nu, q_pow); // suppress unused warning

        if term.abs() < TOL * sum.abs().max(1e-300) {
            break;
        }

        if !sum.is_finite() {
            return Err(SpecialError::OverflowError(
                "q_bessel_series: series diverged".to_string(),
            ));
        }
    }

    Ok(sum)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jackson_j_nu_at_zero_order_zero() {
        // J_0^(2)(0; q) = 1 for any q in (0,1)
        // At x=0: only k=0 term contributes -> (x/2)^0 = 1, ratio = (q^1;q)_∞/(q;q)_∞ = 1
        let val = jackson_j_nu(0.0, 0.0, 0.5).expect("jackson j_0(0)");
        assert!((val - 1.0).abs() < 1e-8, "val = {val}");
    }

    #[test]
    fn test_jackson_j_nu_small_x() {
        // For small x and q=0.5, J_0^(2)(x) should be close to 1
        let val = jackson_j_nu(0.0, 0.01, 0.5).expect("jackson j_0 small x");
        assert!((val - 1.0).abs() < 0.01, "val = {val}");
    }

    #[test]
    fn test_jackson_j_nu_order_one_at_zero() {
        // J_1^(2)(0; q) = 0 (since (x/2)^1 -> 0 as x -> 0)
        let val = jackson_j_nu(1.0, 0.0, 0.5).expect("jackson j_1(0)");
        assert!(val.abs() < 1e-14, "val = {val}");
    }

    #[test]
    fn test_big_q_bessel_at_zero_order_zero() {
        // J_0(0; q) = (q;q)_∞/(q;q)_∞ * 1 * [first term = 1] = 1
        let val = big_q_bessel(0.0, 0.0, 0.5).expect("big_q_bessel J_0(0)");
        assert!((val - 1.0).abs() < 1e-8, "val = {val}");
    }

    #[test]
    fn test_big_q_bessel_order_one_at_zero() {
        // J_1(0; q) = 0 (half_x_nu = 0 since half_x = 0, nu = 1)
        let val = big_q_bessel(1.0, 0.0, 0.5).expect("big_q_bessel J_1(0)");
        assert!(val.abs() < 1e-14, "val = {val}");
    }

    #[test]
    fn test_q_bessel_series_small() {
        // Series should be finite for small x
        let val = q_bessel_series(0.0, 0.1, 0.5, 20).expect("q_bessel_series");
        assert!(val.is_finite());
        // First term is 1, so value should be near 1
        assert!((val - 1.0).abs() < 0.1, "val = {val}");
    }

    #[test]
    fn test_q_bessel_series_n1_equals_first_term() {
        // With n_terms=1: sum = (x/2)^ν / ((q;q)_0 * (q^{ν+1};q)_0) = (x/2)^ν
        let nu = 0.5f64;
        let x = 0.4f64;
        let q = 0.7f64;
        let val = q_bessel_series(nu, x, q, 1).expect("q_bessel_series n=1");
        let expected = (x / 2.0).powf(nu);
        assert!((val - expected).abs() < 1e-12, "val = {val}, expected = {expected}");
    }

    #[test]
    fn test_jackson_domain_errors() {
        assert!(jackson_j_nu(0.0, 0.5, 1.5).is_err()); // q > 1
        assert!(jackson_j_nu(0.0, -0.5, 0.5).is_err()); // x < 0
        assert!(jackson_j_nu(0.0, 0.5, 0.0).is_err()); // q = 0
    }
}
