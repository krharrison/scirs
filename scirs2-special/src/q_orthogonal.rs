//! q-Orthogonal Polynomials (Discrete q-Askey Scheme)
//!
//! This module implements q-analogs of classical orthogonal polynomials arranged
//! according to the q-Askey scheme. These polynomials are orthogonal with respect to
//! discrete or absolutely continuous measures involving the q-Pochhammer symbol and
//! play a central role in:
//!
//! - Representation theory of quantum groups
//! - Basic hypergeometric series (via the Askey-Wilson scheme)
//! - q-Fourier analysis
//! - Combinatorics of lattice paths and tilings
//! - Exactly solvable models in statistical mechanics
//!
//! ## Mathematical Background
//!
//! ### q-Askey Scheme Hierarchy
//!
//! The q-Askey scheme organizes q-orthogonal polynomials by their degree of generality:
//! ```text
//! Askey-Wilson / q-Racah (top)
//!     ↓
//! Big q-Jacobi / Little q-Jacobi
//!     ↓
//! q-Laguerre / Al-Salam-Carlitz / Big q-Hermite
//!     ↓
//! Discrete q-Hermite / q-Charlier (bottom)
//! ```
//!
//! ### Little q-Jacobi Polynomials
//!
//! The little q-Jacobi polynomials p_n(x; a, b | q) are defined by the basic
//! hypergeometric series:
//! ```text
//! p_n(x; a,b | q) = ₂φ₁(q^{-n}, abq^{n+1}; aq | q; qx)
//!                 = sum_{k=0}^n C(n,k)_q * (abq^{n+1};q)_k / (aq;q)_k * (-1)^k * q^{k(k-1)/2} * (qx)^k
//! ```
//!
//! They are orthogonal on (0,1) with weight (bqx; q)_∞ / (qx; q)_∞ * (aq)^x.
//!
//! ### Big q-Jacobi Polynomials
//!
//! The big q-Jacobi polynomials P_n(x; a, b, c | q) are defined by:
//! ```text
//! P_n(x; a,b,c | q) = ₃φ₂(q^{-n}, abq^{n+1}, x; aq, cq | q; q)
//! ```
//!
//! Orthogonal on (cq, aq) with respect to a specific q-weight.
//!
//! ### q-Laguerre Polynomials
//!
//! The q-Laguerre polynomials L_n^(α)(x; q) are defined by:
//! ```text
//! L_n^(α)(x; q) = (q^{α+1}; q)_n / (q; q)_n * ₁φ₁(q^{-n}; q^{α+1}; q; -q^{α+n+1}x)
//! ```
//!
//! In the limit q → 1 they reduce to classical Laguerre polynomials L_n^(α)(x).
//!
//! ### Al-Salam-Carlitz Polynomials
//!
//! Two families U_n^(a)(x; q) and V_n^(a)(x; q):
//! ```text
//! U_n^(a)(x; q) = sum_{k=0}^n C(n,k)_q * (a;q)_k * x^k
//! V_n^(a)(x; q) = sum_{k=0}^n C(n,k)_q * (a;q)_k * (-1)^k * q^{k(k-1)/2} * x^{n-k}
//! ```
//!
//! ### Discrete q-Hermite Polynomials
//!
//! There are two families:
//! ```text
//! h_n^I(x; q)  = sum_{k=0}^{⌊n/2⌋} C(n, 2k)_q * (q;q^2)_k * (-1)^k * q^{k(k-1)} * x^{n-2k}
//! h_n^II(x; q) = sum_{k=0}^{⌊n/2⌋} C(n, 2k)_q * (q;q^2)_k * (-1)^k * q^{-2k^2} * x^{n-2k}
//! ```
//!
//! ### q-Charlier Polynomials
//!
//! The q-Charlier polynomials C_n(x; a; q) are defined by:
//! ```text
//! C_n(x; a; q) = ₂φ₁(q^{-n}, x^{-1}; 0; q; -q^{n+1}/a)
//!              = sum_{k=0}^n C(n,k)_q * (x^{-1}; q)_k * (-1)^k * q^{k(k-1)/2+nk} / a^k
//! ```
//!
//! Orthogonal on non-negative integers with Poisson-like q-weight.
//!
//! ## References
//!
//! - Koekoek, R., Lesky, P.A., Swarttouw, R.F. (2010). *Hypergeometric Orthogonal
//!   Polynomials and Their q-Analogues*. Springer.
//! - Gasper, G. & Rahman, M. (2004). *Basic Hypergeometric Series*, 2nd ed. Cambridge.
//! - Ismail, M.E.H. (2005). *Classical and Quantum Orthogonal Polynomials in One Variable*. Cambridge.
//! - Nikiforov, A.F., Suslov, S.K., Uvarov, V.B. (1991). *Classical Orthogonal Polynomials
//!   of a Discrete Variable*. Springer.
//! - DLMF Chapter 18: Orthogonal Polynomials.

use crate::error::{SpecialError, SpecialResult};
use crate::q_analogs::{q_binomial, q_pochhammer};

/// Maximum number of terms in basic hypergeometric summation
const MAX_TERMS: usize = 300;

/// Convergence tolerance
const TOL: f64 = 1e-14;

// ============================================================================
// Internal helpers
// ============================================================================

/// Evaluates a terminating basic hypergeometric series ₂φ₁.
///
/// ₂φ₁(a1, a2; b1; q; z) = sum_{k=0}^n (a1;q)_k (a2;q)_k / ((b1;q)_k (q;q)_k) * z^k
///
/// Here a1 = q^{-n} so the series terminates at k = n.
fn phi2_1_terminating(a1: f64, a2: f64, b1: f64, q: f64, z: f64, n: usize) -> SpecialResult<f64> {
    let mut sum = 0.0f64;
    let mut a1_k = 1.0f64; // (a1;q)_k
    let mut a2_k = 1.0f64; // (a2;q)_k
    let mut b1_k = 1.0f64; // (b1;q)_k
    let mut qq_k = 1.0f64; // (q;q)_k
    let mut z_pow = 1.0f64; // z^k
    let mut q_pow = 1.0f64; // q^k

    for k in 0..=n {
        if k > 0 {
            // Update Pochhammer products
            a1_k *= 1.0 - a1 * q.powi((k - 1) as i32);
            a2_k *= 1.0 - a2 * q.powi((k - 1) as i32);
            b1_k *= 1.0 - b1 * q.powi((k - 1) as i32);
            q_pow *= q;
            qq_k *= 1.0 - q_pow;
            z_pow *= z;
        }

        let denom = b1_k * qq_k;
        if denom.abs() < 1e-300 {
            return Err(SpecialError::ComputationError(format!(
                "phi2_1_terminating: denominator vanished at k = {k}"
            )));
        }

        let term = a1_k * a2_k / denom * z_pow;
        sum += term;

        if !sum.is_finite() {
            return Err(SpecialError::OverflowError(format!(
                "phi2_1_terminating: overflow at k = {k}"
            )));
        }
    }

    Ok(sum)
}

/// Evaluates a terminating basic hypergeometric series ₃φ₂.
///
/// ₃φ₂(a1, a2, a3; b1, b2; q; z) = sum_{k=0}^n (a1;q)_k (a2;q)_k (a3;q)_k
///                                            / ((b1;q)_k (b2;q)_k (q;q)_k) * z^k
fn phi3_2_terminating(
    a1: f64,
    a2: f64,
    a3: f64,
    b1: f64,
    b2: f64,
    q: f64,
    z: f64,
    n: usize,
) -> SpecialResult<f64> {
    let mut sum = 0.0f64;
    let mut a1_k = 1.0f64;
    let mut a2_k = 1.0f64;
    let mut a3_k = 1.0f64;
    let mut b1_k = 1.0f64;
    let mut b2_k = 1.0f64;
    let mut qq_k = 1.0f64;
    let mut z_pow = 1.0f64;
    let mut q_pow = 1.0f64;

    for k in 0..=n {
        if k > 0 {
            let qk1 = q.powi((k - 1) as i32);
            a1_k *= 1.0 - a1 * qk1;
            a2_k *= 1.0 - a2 * qk1;
            a3_k *= 1.0 - a3 * qk1;
            b1_k *= 1.0 - b1 * qk1;
            b2_k *= 1.0 - b2 * qk1;
            q_pow *= q;
            qq_k *= 1.0 - q_pow;
            z_pow *= z;
        }

        let denom = b1_k * b2_k * qq_k;
        if denom.abs() < 1e-300 {
            return Err(SpecialError::ComputationError(format!(
                "phi3_2_terminating: denominator vanished at k = {k}"
            )));
        }

        let term = a1_k * a2_k * a3_k / denom * z_pow;
        sum += term;

        if !sum.is_finite() {
            return Err(SpecialError::OverflowError(format!(
                "phi3_2_terminating: overflow at k = {k}"
            )));
        }
    }

    Ok(sum)
}

/// Evaluates a ₁φ₁ basic hypergeometric series (may be non-terminating).
fn phi1_1(a1: f64, b1: f64, q: f64, z: f64) -> SpecialResult<f64> {
    let mut sum = 0.0f64;
    let mut a1_k = 1.0f64;
    let mut b1_k = 1.0f64;
    let mut qq_k = 1.0f64;
    let mut z_pow = 1.0f64;
    let mut q_pow = 1.0f64;

    for k in 0..MAX_TERMS {
        if k > 0 {
            let qk1 = q.powi((k - 1) as i32);
            a1_k *= 1.0 - a1 * qk1;
            b1_k *= 1.0 - b1 * qk1;
            q_pow *= q;
            qq_k *= 1.0 - q_pow;
            z_pow *= z;
        }

        let denom = b1_k * qq_k;
        if denom.abs() < 1e-300 {
            break;
        }

        let term = a1_k / denom * z_pow;
        sum += term;

        if term.abs() < TOL * sum.abs().max(1e-300) && k > 3 {
            return Ok(sum);
        }

        if !sum.is_finite() {
            return Err(SpecialError::OverflowError(
                "phi1_1: series diverged".to_string(),
            ));
        }
    }

    if sum.is_finite() {
        Ok(sum)
    } else {
        Err(SpecialError::ConvergenceError(
            "phi1_1: series did not converge".to_string(),
        ))
    }
}

// ============================================================================
// Little q-Jacobi Polynomials
// ============================================================================

/// Computes the little q-Jacobi polynomial p_n(x; a, b | q).
///
/// # Definition
///
/// ```text
/// p_n(x; a, b | q) = ₂φ₁(q^{-n}, abq^{n+1}; aq | q; qx)
/// ```
///
/// This is a terminating basic hypergeometric series with n+1 terms.
///
/// # Orthogonality
///
/// On the interval (0, 1) with weight w(x) = (bqx; q)_∞ / (qx; q)_∞ * (aq)^x,
/// restricted to the set {q^k : k = 0, 1, 2, ...}:
/// ```text
/// sum_{k=0}^∞ p_m(q^k) * p_n(q^k) * w(q^k) = 0   for m ≠ n
/// ```
///
/// # Limiting Behavior
///
/// As q → 1, p_n(x; a, b | q) → P_n^{(α,β)}(1-2x) / P_n^{(α,β)}(1) where
/// α = a and β = b are the Jacobi parameters.
///
/// # Arguments
///
/// * `n` - Degree of the polynomial (non-negative integer)
/// * `a` - Parameter a (a > 0 for orthogonality on (0,1))
/// * `b` - Parameter b (b > 0)
/// * `x` - Evaluation point
/// * `q` - Deformation parameter (0 < q < 1)
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_orthogonal::little_q_jacobi;
///
/// // p_0 = 1 for any parameters
/// let val = little_q_jacobi(0, 0.5, 0.5, 0.3, 0.7).unwrap();
/// assert!((val - 1.0).abs() < 1e-12);
/// ```
pub fn little_q_jacobi(n: usize, a: f64, b: f64, x: f64, q: f64) -> SpecialResult<f64> {
    if q <= 0.0 || q >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "little_q_jacobi: q must satisfy 0 < q < 1, got q = {q}"
        )));
    }

    if n == 0 {
        return Ok(1.0);
    }

    // ₂φ₁(q^{-n}, abq^{n+1}; aq | q; qx)
    let a1 = q.powi(-(n as i32)); // q^{-n}
    let a2 = a * b * q.powi((n + 1) as i32); // abq^{n+1}
    let b1 = a * q; // aq

    phi2_1_terminating(a1, a2, b1, q, q * x, n)
}

// ============================================================================
// Big q-Jacobi Polynomials
// ============================================================================

/// Computes the big q-Jacobi polynomial P_n(x; a, b, c | q).
///
/// # Definition
///
/// ```text
/// P_n(x; a, b, c | q) = ₃φ₂(q^{-n}, abq^{n+1}, x; aq, cq | q; q)
/// ```
///
/// This is a terminating ₃φ₂ series with n+1 terms.
///
/// # Orthogonality
///
/// Orthogonal on the set {cq^k : k=0,1,...} ∪ {aq^k : k=0,1,...} with
/// appropriate q-weight function.
///
/// # Arguments
///
/// * `n` - Degree (non-negative integer)
/// * `a` - Parameter a
/// * `b` - Parameter b
/// * `c` - Parameter c
/// * `x` - Evaluation point
/// * `q` - Deformation parameter (0 < q < 1)
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_orthogonal::big_q_jacobi;
///
/// // P_0 = 1
/// let val = big_q_jacobi(0, 0.5, 0.5, 0.1, 0.4, 0.7).unwrap();
/// assert!((val - 1.0).abs() < 1e-12);
/// ```
pub fn big_q_jacobi(n: usize, a: f64, b: f64, c: f64, x: f64, q: f64) -> SpecialResult<f64> {
    if q <= 0.0 || q >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "big_q_jacobi: q must satisfy 0 < q < 1, got q = {q}"
        )));
    }

    if n == 0 {
        return Ok(1.0);
    }

    // ₃φ₂(q^{-n}, abq^{n+1}, x; aq, cq | q; q)
    let a1 = q.powi(-(n as i32));
    let a2 = a * b * q.powi((n + 1) as i32);
    let a3 = x;
    let b1 = a * q;
    let b2 = c * q;

    phi3_2_terminating(a1, a2, a3, b1, b2, q, q, n)
}

// ============================================================================
// q-Laguerre Polynomials
// ============================================================================

/// Computes the q-Laguerre polynomial L_n^(α)(x; q).
///
/// # Definition
///
/// ```text
/// L_n^(α)(x; q) = (q^{α+1}; q)_n / (q; q)_n
///                 * ₁φ₁(q^{-n}; q^{α+1}; q; -q^{α+n+1} * x)
/// ```
///
/// # Properties
///
/// - Orthogonal on (0, ∞) with weight x^α / (-x; q)_∞ (Jackson measure)
/// - Recurrence: L_{n+1}^(α)(x) = ((1 + q^{α+1}(1 - q^n / [n+1]_q)) L_n^(α)(x) - ...
/// - Classical limit: lim_{q→1} L_n^(α)(x(1-q)) = L_n^(α)(x) / L_n^(α)(0)
///
/// # Arguments
///
/// * `n`     - Degree (non-negative integer)
/// * `alpha` - Parameter α (α > -1 for orthogonality)
/// * `x`     - Evaluation point (x ≥ 0 for orthogonality region)
/// * `q`     - Deformation parameter (0 < q < 1)
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_orthogonal::q_laguerre;
///
/// // L_0^(α)(x; q) = 1
/// let val = q_laguerre(0, 0.5, 1.0, 0.7).unwrap();
/// assert!((val - 1.0).abs() < 1e-12);
/// ```
pub fn q_laguerre(n: usize, alpha: f64, x: f64, q: f64) -> SpecialResult<f64> {
    if q <= 0.0 || q >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "q_laguerre: q must satisfy 0 < q < 1, got q = {q}"
        )));
    }

    if n == 0 {
        return Ok(1.0);
    }

    // Compute prefactor (q^{α+1}; q)_n / (q; q)_n
    let qa1 = q.powf(alpha + 1.0); // q^{α+1}
    let poch_qa1_n = q_pochhammer(qa1, q, n)?;
    let poch_q_n = q_pochhammer(q, q, n)?;

    if poch_q_n.abs() < 1e-300 {
        return Err(SpecialError::ComputationError(
            "q_laguerre: (q;q)_n is too close to zero".to_string(),
        ));
    }

    let prefactor = poch_qa1_n / poch_q_n;

    // ₁φ₁(q^{-n}; q^{α+1}; q; -q^{α+n+1} * x)
    let a1 = q.powi(-(n as i32)); // q^{-n}
    let b1 = qa1; // q^{α+1}
    let z = -q.powf(alpha + (n as f64) + 1.0) * x; // -q^{α+n+1} * x

    // Since a1 = q^{-n}, the series terminates at k = n
    let phi_val = phi1_1_terminating(a1, b1, q, z, n)?;

    Ok(prefactor * phi_val)
}

/// Terminating ₁φ₁ series with upper parameter a1 = q^{-n}.
fn phi1_1_terminating(a1: f64, b1: f64, q: f64, z: f64, n: usize) -> SpecialResult<f64> {
    let mut sum = 0.0f64;
    let mut a1_k = 1.0f64;
    let mut b1_k = 1.0f64;
    let mut qq_k = 1.0f64;
    let mut z_pow = 1.0f64;
    let mut q_pow = 1.0f64;

    for k in 0..=n {
        if k > 0 {
            let qk1 = q.powi((k - 1) as i32);
            a1_k *= 1.0 - a1 * qk1;
            b1_k *= 1.0 - b1 * qk1;
            q_pow *= q;
            qq_k *= 1.0 - q_pow;
            z_pow *= z;
        }

        let denom = b1_k * qq_k;
        if denom.abs() < 1e-300 {
            return Err(SpecialError::ComputationError(format!(
                "phi1_1_terminating: denominator vanished at k = {k}"
            )));
        }

        let term = a1_k / denom * z_pow;
        sum += term;

        if !sum.is_finite() {
            return Err(SpecialError::OverflowError(format!(
                "phi1_1_terminating: overflow at k = {k}"
            )));
        }
    }

    Ok(sum)
}

// ============================================================================
// Al-Salam-Carlitz Polynomials
// ============================================================================

/// Computes the Al-Salam-Carlitz polynomial U_n^(a)(x; q).
///
/// # Definition
///
/// The first family is:
/// ```text
/// U_n^(a)(x; q) = sum_{k=0}^n [n choose k]_q * (a; q)_k * x^k
/// ```
///
/// Equivalently:
/// ```text
/// U_n^(a)(x; q) = (-a)^n q^{n(n-1)/2} * ₂φ₀(q^{-n}, x^{-1}; -; q; q^{n}/a)  [formal]
/// ```
///
/// # Properties
///
/// - Orthogonal on {a, aq, aq^2, ...} ∪ {0} for a < 0
/// - U_0^(a)(x; q) = 1
/// - U_1^(a)(x; q) = x - (1+a)
/// - Recurrence: U_{n+1} = (x - (1 + a*q^n)) U_n - a*q^{n-1}*(1-q^n) U_{n-1}
///
/// # Arguments
///
/// * `n` - Degree (non-negative integer)
/// * `a` - Parameter a (a ≠ 0 for non-trivial case)
/// * `x` - Evaluation point
/// * `q` - Deformation parameter (0 < q < 1)
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_orthogonal::al_salam_carlitz;
///
/// // U_0 = 1
/// let val = al_salam_carlitz(0, -1.0, 0.5, 0.7).unwrap();
/// assert!((val - 1.0).abs() < 1e-12);
///
/// // U_1 = x - (1+a)
/// let a = -0.5;
/// let x = 0.3;
/// let val1 = al_salam_carlitz(1, a, x, 0.7).unwrap();
/// assert!((val1 - (x - (1.0 + a))).abs() < 1e-10);
/// ```
pub fn al_salam_carlitz(n: usize, a: f64, x: f64, q: f64) -> SpecialResult<f64> {
    if q <= 0.0 || q >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "al_salam_carlitz: q must satisfy 0 < q < 1, got q = {q}"
        )));
    }

    if n == 0 {
        return Ok(1.0);
    }

    if n == 1 {
        return Ok(x - (1.0 + a));
    }

    // Use three-term recurrence: U_{n+1}(x) = (x - (1 + a*q^n)) U_n(x) - a*q^{n-1}*(1-q^n) U_{n-1}(x)
    let mut u_prev = 1.0f64; // U_0
    let mut u_curr = x - (1.0 + a); // U_1

    for k in 1..n {
        let q_k = q.powi(k as i32); // q^k (this is the "n" in the recurrence at step k)
        let coeff_curr = x - (1.0 + a * q_k);
        let coeff_prev = a * q.powi((k - 1) as i32) * (1.0 - q_k);
        let u_next = coeff_curr * u_curr - coeff_prev * u_prev;
        u_prev = u_curr;
        u_curr = u_next;

        if !u_curr.is_finite() {
            return Err(SpecialError::OverflowError(format!(
                "al_salam_carlitz: overflow at k = {k}"
            )));
        }
    }

    Ok(u_curr)
}

// ============================================================================
// Discrete q-Hermite Polynomials
// ============================================================================

/// Computes the discrete q-Hermite polynomial h_n^I(x; q).
///
/// # Definition
///
/// The first discrete q-Hermite polynomials (type I) are:
/// ```text
/// h_n^I(x; q) = sum_{k=0}^{⌊n/2⌋} [n choose 2k]_q * (q; q^2)_k * (-1)^k * q^{k(k-1)} * x^{n-2k}
/// ```
///
/// where (q; q^2)_k = (1-q)(1-q^3)(1-q^5)...(1-q^{2k-1}).
///
/// # Properties
///
/// - h_0^I = 1
/// - h_1^I(x) = x
/// - Recurrence: h_{n+1}^I(x) = x * h_n^I(x) - q^{n-1} * (1-q^n) * h_{n-1}^I(x)
/// - Orthogonal on (0, 1/(1-q)) with q-weight
///
/// # Arguments
///
/// * `n` - Degree (non-negative integer)
/// * `x` - Evaluation point
/// * `q` - Deformation parameter (0 < q < 1)
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_orthogonal::discrete_q_hermite;
///
/// // h_0 = 1
/// let val = discrete_q_hermite(0, 1.0, 0.7).unwrap();
/// assert!((val - 1.0).abs() < 1e-12);
///
/// // h_1 = x
/// let val1 = discrete_q_hermite(1, 0.5, 0.7).unwrap();
/// assert!((val1 - 0.5).abs() < 1e-12);
/// ```
pub fn discrete_q_hermite(n: usize, x: f64, q: f64) -> SpecialResult<f64> {
    if q <= 0.0 || q >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "discrete_q_hermite: q must satisfy 0 < q < 1, got q = {q}"
        )));
    }

    if n == 0 {
        return Ok(1.0);
    }

    if n == 1 {
        return Ok(x);
    }

    // Three-term recurrence: h_{n+1}^I(x) = x * h_n^I(x) - q^{n-1}*(1-q^n)*h_{n-1}^I(x)
    let mut h_prev = 1.0f64; // h_0
    let mut h_curr = x; // h_1

    for k in 1..n {
        // Recurrence from h_{k} to h_{k+1}:
        // h_{k+1}(x) = x * h_k(x) - q^{k-1}*(1-q^k) * h_{k-1}(x)
        let q_km1 = q.powi((k - 1) as i32);
        let q_k = q.powi(k as i32);
        let coeff = q_km1 * (1.0 - q_k);
        let h_next = x * h_curr - coeff * h_prev;
        h_prev = h_curr;
        h_curr = h_next;

        if !h_curr.is_finite() {
            return Err(SpecialError::OverflowError(format!(
                "discrete_q_hermite: overflow at k = {k}"
            )));
        }
    }

    Ok(h_curr)
}

// ============================================================================
// q-Charlier Polynomials
// ============================================================================

/// Computes the q-Charlier polynomial C_n(q^{-x}; a; q).
///
/// # Definition
///
/// The q-Charlier polynomials are defined by:
/// ```text
/// C_n(x; a; q) = ₂φ₁(q^{-n}, x; 0; q; -q^{n+1}/a)
/// ```
///
/// where the series is evaluated at x representing a discrete variable q^{-s}.
///
/// In direct series form:
/// ```text
/// C_n(x; a; q) = sum_{k=0}^n [n choose k]_q * (x; q)_k * (-1)^k * q^{k(k-1)/2} * q^{nk} / a^k
/// ```
///
/// # Orthogonality
///
/// Orthogonal on {1, q, q^2, ...} (i.e., x = q^{-s}, s = 0, 1, 2, ...) with
/// weight w(q^{-s}) = a^s * q^{s(s-1)/2} / (q; q)_s.
///
/// # Arguments
///
/// * `n` - Degree (non-negative integer)
/// * `a` - Parameter a (a > 0 for orthogonality)
/// * `x` - Evaluation point (often x = q^{-s} for some non-negative integer s)
/// * `q` - Deformation parameter (0 < q < 1)
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_orthogonal::q_charlier;
///
/// // C_0 = 1
/// let val = q_charlier(0, 2.0, 0.5, 0.7).unwrap();
/// assert!((val - 1.0).abs() < 1e-12);
/// ```
pub fn q_charlier(n: usize, a: f64, x: f64, q: f64) -> SpecialResult<f64> {
    if q <= 0.0 || q >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "q_charlier: q must satisfy 0 < q < 1, got q = {q}"
        )));
    }

    if a <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "q_charlier: a must be positive, got a = {a}"
        )));
    }

    if n == 0 {
        return Ok(1.0);
    }

    // Series: sum_{k=0}^n [n choose k]_q * (x;q)_k * (-1)^k * q^{k(k-1)/2} * q^{nk} / a^k
    let mut sum = 0.0f64;
    let mut q_kk1_2 = 1.0f64; // q^{k(k-1)/2}
    let mut q_nk = 1.0f64; // q^{nk}
    let mut a_pow = 1.0f64; // a^k
    let q_n = q.powi(n as i32); // q^n

    for k in 0..=n {
        if k > 0 {
            // q^{k(k-1)/2}: at step k, multiply by q^{k-1}
            q_kk1_2 *= q.powi((k - 1) as i32);
            q_nk *= q_n;
            a_pow *= a;
        }

        let binom = q_binomial(n, k, q)?;
        let poch_x_k = q_pochhammer(x, q, k)?;

        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let term = binom * poch_x_k * sign * q_kk1_2 * q_nk / a_pow;
        sum += term;

        if !sum.is_finite() {
            return Err(SpecialError::OverflowError(format!(
                "q_charlier: overflow at k = {k}"
            )));
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

    // --- Little q-Jacobi ---

    #[test]
    fn test_little_q_jacobi_degree_zero() {
        let val = little_q_jacobi(0, 0.5, 0.5, 0.3, 0.7).expect("little_q_jacobi n=0");
        assert!((val - 1.0).abs() < 1e-12, "val = {val}");
    }

    #[test]
    fn test_little_q_jacobi_degree_one() {
        // p_1(x; a, b | q) = ₂φ₁(q^{-1}, abq^2; aq; q; qx)
        // = 1 + (q^{-1}-1)(abq^2-1)/(aq-1) * qx
        // = 1 - (1-q^{-1})(abq^2-1)/(aq-1) * qx
        let q = 0.6f64;
        let a = 0.5f64;
        let b = 0.4f64;
        let x = 0.3f64;
        let val = little_q_jacobi(1, a, b, x, q).expect("little_q_jacobi n=1");
        // Manual: k=0: 1; k=1: (q^{-1}-1)*(abq^2-1)/(aq-1)/(q-1) * qx
        // = (1/q - 1)*(abq^2 - 1) / (aq - 1) / (q - 1) * qx
        let a1 = q.powi(-1);
        let a2 = a * b * q * q;
        let b1 = a * q;
        let k1_num = (1.0 - a1) * (1.0 - a2);
        let k1_den = (1.0 - b1) * (1.0 - q);
        let expected = 1.0 + k1_num / k1_den * (q * x);
        assert!((val - expected).abs() < 1e-10, "val = {val}, expected = {expected}");
    }

    #[test]
    fn test_little_q_jacobi_symmetry_at_x0() {
        // p_n(0; a, b | q) = 1 for all n (since qx = 0)
        // Actually p_n(0) = ₂φ₁(q^{-n}, abq^{n+1}; aq; q; 0) = 1
        let val = little_q_jacobi(3, 0.5, 0.5, 0.0, 0.7).expect("little_q_jacobi at x=0");
        assert!((val - 1.0).abs() < 1e-12, "val = {val}");
    }

    // --- Big q-Jacobi ---

    #[test]
    fn test_big_q_jacobi_degree_zero() {
        let val = big_q_jacobi(0, 0.5, 0.5, 0.1, 0.4, 0.7).expect("big_q_jacobi n=0");
        assert!((val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_big_q_jacobi_finite() {
        let val = big_q_jacobi(2, 0.5, 0.3, 0.1, 0.4, 0.7).expect("big_q_jacobi n=2");
        assert!(val.is_finite());
    }

    // --- q-Laguerre ---

    #[test]
    fn test_q_laguerre_degree_zero() {
        let val = q_laguerre(0, 0.5, 1.0, 0.7).expect("q_laguerre n=0");
        assert!((val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_q_laguerre_finite() {
        let val = q_laguerre(3, 0.5, 0.5, 0.7).expect("q_laguerre n=3");
        assert!(val.is_finite());
    }

    // --- Al-Salam-Carlitz ---

    #[test]
    fn test_al_salam_carlitz_degree_zero() {
        let val = al_salam_carlitz(0, -1.0, 0.5, 0.7).expect("al_salam_carlitz n=0");
        assert!((val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_al_salam_carlitz_degree_one() {
        let a = -0.5f64;
        let x = 0.3f64;
        let val = al_salam_carlitz(1, a, x, 0.7).expect("al_salam_carlitz n=1");
        let expected = x - (1.0 + a);
        assert!((val - expected).abs() < 1e-10, "val = {val}, expected = {expected}");
    }

    #[test]
    fn test_al_salam_carlitz_recurrence() {
        // Verify 3-term recurrence: U_{n+1}(x) = (x - (1+aq^n)) U_n(x) - aq^{n-1}(1-q^n) U_{n-1}(x)
        let a = -0.3f64;
        let x = 0.5f64;
        let q = 0.6f64;
        let u0 = al_salam_carlitz(0, a, x, q).expect("U_0");
        let u1 = al_salam_carlitz(1, a, x, q).expect("U_1");
        let u2 = al_salam_carlitz(2, a, x, q).expect("U_2");
        // n=1: U_2 = (x - (1+aq^1)) U_1 - a*q^0*(1-q^1) U_0
        let expected = (x - (1.0 + a * q)) * u1 - a * (1.0 - q) * u0;
        assert!((u2 - expected).abs() < 1e-8, "u2 = {u2}, expected = {expected}");
    }

    // --- Discrete q-Hermite ---

    #[test]
    fn test_discrete_q_hermite_degree_zero() {
        let val = discrete_q_hermite(0, 1.0, 0.7).expect("dqh n=0");
        assert!((val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_discrete_q_hermite_degree_one() {
        let x = 0.5f64;
        let val = discrete_q_hermite(1, x, 0.7).expect("dqh n=1");
        assert!((val - x).abs() < 1e-12);
    }

    #[test]
    fn test_discrete_q_hermite_recurrence() {
        // Verify: h_{n+1}^I(x) = x * h_n^I(x) - q^{n-1}*(1-q^n)*h_{n-1}^I(x)
        let x = 0.6f64;
        let q = 0.5f64;
        let h0 = discrete_q_hermite(0, x, q).expect("h0");
        let h1 = discrete_q_hermite(1, x, q).expect("h1");
        let h2 = discrete_q_hermite(2, x, q).expect("h2");
        // n=1: h_2 = x*h_1 - q^0*(1-q^1)*h_0
        let expected = x * h1 - (1.0 - q) * h0;
        assert!((h2 - expected).abs() < 1e-10, "h2 = {h2}, expected = {expected}");
    }

    // --- q-Charlier ---

    #[test]
    fn test_q_charlier_degree_zero() {
        let val = q_charlier(0, 2.0, 0.5, 0.7).expect("q_charlier n=0");
        assert!((val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_q_charlier_finite() {
        let val = q_charlier(3, 1.5, 0.4, 0.6).expect("q_charlier n=3");
        assert!(val.is_finite());
    }

    #[test]
    fn test_q_charlier_domain_error() {
        assert!(q_charlier(2, -1.0, 0.5, 0.7).is_err()); // a <= 0
        assert!(q_charlier(2, 1.0, 0.5, 1.5).is_err()); // q >= 1
    }
}
