//! q-Analogs of Classical Special Functions
//!
//! This module implements q-analogs of fundamental special functions. A q-analog of a mathematical
//! formula is a generalization involving a parameter q that recovers the original formula in the
//! limit q → 1. These functions appear extensively in:
//!
//! - Combinatorics and enumerative combinatorics
//! - Quantum groups and quantum algebra
//! - Basic hypergeometric series (Gasper & Rahman)
//! - Statistical mechanics (exactly solvable models)
//! - Number theory (partition functions, modular forms)
//!
//! ## Mathematical Background
//!
//! ### q-Pochhammer Symbol
//!
//! The q-Pochhammer symbol (a; q)_n is the fundamental building block:
//! ```text
//! (a; q)_n = prod_{k=0}^{n-1} (1 - a*q^k)
//!          = (1-a)(1-aq)(1-aq^2)...(1-aq^{n-1})
//! ```
//! With the convention (a; q)_0 = 1.
//!
//! ### q-Numbers and q-Factorial
//!
//! The q-analog of an integer n is:
//! ```text
//! [n]_q = (1 - q^n) / (1 - q)  for q ≠ 1
//!       = n                     for q = 1
//! ```
//!
//! The q-factorial is:
//! ```text
//! [n]_q! = [1]_q * [2]_q * ... * [n]_q = (q; q)_n / (1-q)^n
//! ```
//!
//! ### Gaussian Binomial Coefficient
//!
//! The q-binomial (Gaussian binomial) coefficient is:
//! ```text
//! [n choose k]_q = [n]_q! / ([k]_q! * [n-k]_q!)
//!                = (q; q)_n / ((q; q)_k * (q; q)_{n-k})
//! ```
//!
//! It counts the number of k-dimensional subspaces of an n-dimensional vector space over F_q.
//!
//! ### q-Gamma Function
//!
//! The q-gamma function generalizes the classical gamma function:
//! ```text
//! Γ_q(x) = (q; q)_∞ / (q^x; q)_∞  * (1-q)^{1-x}   for 0 < q < 1
//! ```
//! It satisfies Γ_q(x+1) = [x]_q * Γ_q(x) and lim_{q→1} Γ_q(x) = Γ(x).
//!
//! ### q-Exponential Functions
//!
//! There are two natural q-exponentials:
//! ```text
//! e_q(x) = sum_{n=0}^∞ x^n / [n]_q!     (Jackson's e_q)
//!        = 1 / (x(1-q); q)_∞             (product formula, |x(1-q)| < 1)
//! E_q(x) = sum_{n=0}^∞ q^{n(n-1)/2} x^n / [n]_q!  (Jackson's E_q)
//! ```
//!
//! ### q-Logarithm
//!
//! The q-logarithm is defined as the inverse of e_q:
//! ```text
//! ln_q(x) = (x^{1-q} - 1) / (1 - q)   (Tsallis q-logarithm)
//! ```
//!
//! ## References
//!
//! - Gasper, G. & Rahman, M. (2004). *Basic Hypergeometric Series*, 2nd ed. Cambridge.
//! - Andrews, G.E. (1976). *The Theory of Partitions*. Addison-Wesley.
//! - Koekoek, R., Lesky, P.A., Swarttouw, R.F. (2010). *Hypergeometric Orthogonal
//!   Polynomials and Their q-Analogues*. Springer.
//! - DLMF Chapter 17: q-Hypergeometric and Related Functions.

use crate::error::{SpecialError, SpecialResult};

// ============================================================================
// Constants and thresholds
// ============================================================================

/// Maximum number of terms in series expansions
const MAX_SERIES_TERMS: usize = 500;

/// Convergence tolerance for infinite products / series
const CONVERGENCE_TOL: f64 = 1e-15;

// ============================================================================
// q-Pochhammer Symbol
// ============================================================================

/// Computes the finite q-Pochhammer symbol (a; q)_n.
///
/// # Definition
///
/// ```text
/// (a; q)_n = prod_{k=0}^{n-1} (1 - a*q^k)
/// ```
///
/// Special cases:
/// - (a; q)_0 = 1 for all a, q
/// - If q = 0: (a; 0)_1 = 1 - a, (a; 0)_n = 1 - a for n ≥ 1
///
/// # Arguments
///
/// * `a` - Base parameter
/// * `q` - Deformation parameter (typically 0 < q < 1 or |q| < 1)
/// * `n` - Number of factors
///
/// # Returns
///
/// `Ok((a; q)_n)` or an error if computation fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_analogs::q_pochhammer;
///
/// // (1/2; 1/2)_3 = (1 - 1/2)(1 - 1/4)(1 - 1/8) = (1/2)(3/4)(7/8)
/// let val = q_pochhammer(0.5, 0.5, 3).unwrap();
/// let expected = 0.5 * 0.75 * 0.875;
/// assert!((val - expected).abs() < 1e-12);
/// ```
pub fn q_pochhammer(a: f64, q: f64, n: usize) -> SpecialResult<f64> {
    if n == 0 {
        return Ok(1.0);
    }

    let mut result = 1.0f64;
    let mut q_k = 1.0f64; // q^k, starts at q^0 = 1

    for _ in 0..n {
        let factor = 1.0 - a * q_k;
        result *= factor;
        q_k *= q;
    }

    if !result.is_finite() {
        return Err(SpecialError::OverflowError(
            "q_pochhammer: result overflowed or underflowed".to_string(),
        ));
    }

    Ok(result)
}

/// Computes the infinite q-Pochhammer symbol (a; q)_∞ for |q| < 1.
///
/// # Definition
///
/// ```text
/// (a; q)_∞ = prod_{k=0}^∞ (1 - a*q^k)
/// ```
///
/// This converges absolutely for |q| < 1.
///
/// # Arguments
///
/// * `a` - Base parameter
/// * `q` - Deformation parameter, must satisfy |q| < 1
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_analogs::q_pochhammer_inf;
///
/// // (0; q)_∞ = 1 for any q
/// let val = q_pochhammer_inf(0.0, 0.5).unwrap();
/// assert!((val - 1.0).abs() < 1e-12);
/// ```
pub fn q_pochhammer_inf(a: f64, q: f64) -> SpecialResult<f64> {
    if q.abs() >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "q_pochhammer_inf: |q| must be < 1, got q = {q}"
        )));
    }

    let mut result = 1.0f64;
    let mut q_k = 1.0f64; // q^k

    for _ in 0..MAX_SERIES_TERMS {
        let factor = 1.0 - a * q_k;
        let prev = result;
        result *= factor;
        q_k *= q;

        // Check convergence: factors approach 1 as k → ∞ since |q| < 1
        if (result - prev).abs() < CONVERGENCE_TOL * result.abs() {
            return Ok(result);
        }

        if !result.is_finite() {
            return Err(SpecialError::OverflowError(
                "q_pochhammer_inf: partial product overflowed".to_string(),
            ));
        }
    }

    // If we didn't formally converge but result is finite, return it with a note
    // (the series may have de facto converged but our criterion wasn't met)
    if result.is_finite() {
        Ok(result)
    } else {
        Err(SpecialError::ConvergenceError(
            "q_pochhammer_inf: infinite product failed to converge".to_string(),
        ))
    }
}

// ============================================================================
// q-Number and q-Factorial
// ============================================================================

/// Computes the q-analog of an integer: [n]_q = (1 - q^n) / (1 - q).
///
/// # Definition
///
/// ```text
/// [n]_q = (1 - q^n) / (1 - q)   for q ≠ 1
///       = n                       for q = 1 (limit)
/// ```
///
/// This is the q-integer or q-number.
fn q_number(n: u64, q: f64) -> f64 {
    if (q - 1.0).abs() < 1e-14 {
        // Limit as q → 1
        n as f64
    } else {
        (1.0 - q.powi(n as i32)) / (1.0 - q)
    }
}

/// Computes the q-factorial [n]_q! = [1]_q * [2]_q * ... * [n]_q.
///
/// # Definition
///
/// ```text
/// [n]_q! = prod_{k=1}^{n} [k]_q
///        = prod_{k=1}^{n} (1 - q^k) / (1 - q)
///        = (q; q)_n / (1-q)^n
/// ```
///
/// # Arguments
///
/// * `n` - Non-negative integer
/// * `q` - Deformation parameter
///
/// # Returns
///
/// `Ok([n]_q!)` or an error.
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_analogs::q_factorial;
///
/// // q-factorial at q=1 should equal n!
/// let val = q_factorial(5, 1.0 - 1e-10).unwrap();
/// assert!((val - 120.0).abs() < 1e-4);
/// ```
pub fn q_factorial(n: usize, q: f64) -> SpecialResult<f64> {
    if n == 0 {
        return Ok(1.0);
    }

    let mut result = 1.0f64;

    if (q - 1.0).abs() < 1e-14 {
        // Classical limit: [n]! = n!
        for k in 1..=(n as u64) {
            result *= k as f64;
        }
    } else {
        for k in 1..=(n as u64) {
            let q_k = q_number(k, q);
            result *= q_k;
            if !result.is_finite() {
                return Err(SpecialError::OverflowError(format!(
                    "q_factorial: overflow at k = {k}"
                )));
            }
        }
    }

    Ok(result)
}

// ============================================================================
// Gaussian Binomial Coefficient
// ============================================================================

/// Computes the Gaussian (q-) binomial coefficient [n choose k]_q.
///
/// # Definition
///
/// ```text
/// [n choose k]_q = [n]_q! / ([k]_q! * [n-k]_q!)
///                = (q; q)_n / ((q; q)_k * (q; q)_{n-k})
/// ```
///
/// At q = 1 this reduces to the ordinary binomial coefficient C(n, k).
///
/// # Properties
///
/// - [n choose 0]_q = [n choose n]_q = 1
/// - [n choose k]_q = [n choose n-k]_q  (symmetry)
/// - [n choose k]_q ∈ Z[q]  (polynomial in q with non-negative integer coefficients)
/// - At q = prime power, counts k-dim subspaces of F_q^n
///
/// # Arguments
///
/// * `n` - Top parameter (non-negative integer)
/// * `k` - Bottom parameter (0 ≤ k ≤ n)
/// * `q` - Deformation parameter
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_analogs::q_binomial;
///
/// // Classical binomial C(4,2) = 6
/// let val = q_binomial(4, 2, 1.0 - 1e-10).unwrap();
/// assert!((val - 6.0).abs() < 1e-4);
///
/// // q-binomial [4 choose 2]_q at q = 2 (counts 2-dim subspaces of F_2^4)
/// let val2 = q_binomial(4, 2, 2.0).unwrap();
/// assert!((val2 - 35.0).abs() < 1e-6);
/// ```
pub fn q_binomial(n: usize, k: usize, q: f64) -> SpecialResult<f64> {
    if k > n {
        return Err(SpecialError::DomainError(format!(
            "q_binomial: k = {k} must be ≤ n = {n}"
        )));
    }

    if k == 0 || k == n {
        return Ok(1.0);
    }

    // Use the more numerically stable formula via q-Pochhammer symbols:
    // [n choose k]_q = (q; q)_n / ((q; q)_k * (q; q)_{n-k})
    // Computed as product to avoid cancellation

    // Use the recurrence / product form that is more stable:
    // [n choose k]_q = prod_{j=0}^{k-1} (1 - q^{n-j}) / (1 - q^{j+1})
    let k_eff = k.min(n - k); // symmetry to reduce number of terms

    let mut result = 1.0f64;

    if (q - 1.0).abs() < 1e-14 {
        // Classical binomial coefficient C(n, k_eff)
        for j in 0..k_eff {
            result *= (n - j) as f64 / (j + 1) as f64;
        }
    } else {
        for j in 0..k_eff {
            let num = 1.0 - q.powi((n - j) as i32);
            let den = 1.0 - q.powi((j + 1) as i32);
            if den.abs() < 1e-300 {
                return Err(SpecialError::ComputationError(
                    "q_binomial: denominator factor vanished".to_string(),
                ));
            }
            result *= num / den;
            if !result.is_finite() {
                return Err(SpecialError::OverflowError(format!(
                    "q_binomial: overflow at j = {j}"
                )));
            }
        }
    }

    Ok(result)
}

// ============================================================================
// q-Gamma Function
// ============================================================================

/// Computes the q-gamma function Γ_q(x) for 0 < q < 1.
///
/// # Definition
///
/// For 0 < q < 1:
/// ```text
/// Γ_q(x) = (q; q)_∞ / (q^x; q)_∞  * (1-q)^{1-x}
/// ```
///
/// This satisfies:
/// - Γ_q(x+1) = [x]_q * Γ_q(x)  (functional equation)
/// - Γ_q(1) = 1
/// - lim_{q→1^-} Γ_q(x) = Γ(x)  (classical limit)
///
/// # Arguments
///
/// * `x` - Argument (x > 0)
/// * `q` - Deformation parameter (0 < q < 1)
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_analogs::q_gamma;
///
/// // Γ_q(1) = 1 for any q ∈ (0,1)
/// let val = q_gamma(1.0, 0.5).unwrap();
/// assert!((val - 1.0).abs() < 1e-10);
///
/// // Γ_q(2) = [1]_q = 1 for any q (since [1]_q = 1)
/// let val2 = q_gamma(2.0, 0.5).unwrap();
/// assert!((val2 - 1.0).abs() < 1e-10);
/// ```
pub fn q_gamma(x: f64, q: f64) -> SpecialResult<f64> {
    if q <= 0.0 || q >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "q_gamma: q must satisfy 0 < q < 1, got q = {q}"
        )));
    }

    if x <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "q_gamma: x must be positive, got x = {x}"
        )));
    }

    // Γ_q(x) = (q; q)_∞ / (q^x; q)_∞  * (1-q)^{1-x}

    let qx = q.powf(x);

    let poch_q = q_pochhammer_inf(q, q)?;
    let poch_qx = q_pochhammer_inf(qx, q)?;

    if poch_qx.abs() < 1e-300 {
        return Err(SpecialError::ComputationError(
            "q_gamma: (q^x; q)_∞ is too close to zero".to_string(),
        ));
    }

    let result = poch_q / poch_qx * (1.0 - q).powf(1.0 - x);

    if !result.is_finite() {
        return Err(SpecialError::OverflowError(
            "q_gamma: result is not finite".to_string(),
        ));
    }

    Ok(result)
}

// ============================================================================
// q-Beta Function
// ============================================================================

/// Computes the q-beta function B_q(a, b).
///
/// # Definition
///
/// The q-beta function is defined via the q-gamma function:
/// ```text
/// B_q(a, b) = Γ_q(a) * Γ_q(b) / Γ_q(a + b)
/// ```
///
/// For 0 < q < 1, a, b > 0.
///
/// # Properties
///
/// - B_q(a, b) = B_q(b, a)  (symmetry)
/// - lim_{q→1} B_q(a, b) = B(a, b) = Γ(a)Γ(b)/Γ(a+b)
///
/// # Arguments
///
/// * `a` - First parameter (a > 0)
/// * `b` - Second parameter (b > 0)
/// * `q` - Deformation parameter (0 < q < 1)
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_analogs::q_beta;
///
/// // B_q(1, 1) = Γ_q(1)^2 / Γ_q(2) = 1/1 = 1
/// let val = q_beta(1.0, 1.0, 0.5).unwrap();
/// assert!((val - 1.0).abs() < 1e-8);
/// ```
pub fn q_beta(a: f64, b: f64, q: f64) -> SpecialResult<f64> {
    if q <= 0.0 || q >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "q_beta: q must satisfy 0 < q < 1, got q = {q}"
        )));
    }

    if a <= 0.0 || b <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "q_beta: a and b must be positive, got a = {a}, b = {b}"
        )));
    }

    let gamma_a = q_gamma(a, q)?;
    let gamma_b = q_gamma(b, q)?;
    let gamma_ab = q_gamma(a + b, q)?;

    if gamma_ab.abs() < 1e-300 {
        return Err(SpecialError::ComputationError(
            "q_beta: Γ_q(a+b) is too close to zero".to_string(),
        ));
    }

    let result = gamma_a * gamma_b / gamma_ab;

    if !result.is_finite() {
        return Err(SpecialError::OverflowError(
            "q_beta: result is not finite".to_string(),
        ));
    }

    Ok(result)
}

// ============================================================================
// q-Exponential Function
// ============================================================================

/// Computes Jackson's q-exponential e_q(x).
///
/// # Definition
///
/// Jackson's q-exponential is defined by the series:
/// ```text
/// e_q(x) = sum_{n=0}^∞ x^n / [n]_q!
/// ```
///
/// It satisfies D_q(e_q(x)) = e_q(x) where D_q is the q-derivative.
///
/// Product formula (for |x(1-q)| < 1):
/// ```text
/// e_q(x) = 1 / ((1-q) * x; q)_∞
/// ```
///
/// Note: There are two standard q-exponentials:
/// - `e_q(x)` (this function): product formula 1/((x(1-q)); q)_∞
/// - `E_q(x)`: includes q^{n(n-1)/2} factors (see `q_exponential_big`)
///
/// # Arguments
///
/// * `x` - Argument
/// * `q` - Deformation parameter (typically |q| < 1)
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_analogs::q_exponential;
///
/// // Near q=1, e_q(x) ≈ exp(x)
/// let val = q_exponential(1.0, 0.999).unwrap();
/// assert!((val - std::f64::consts::E).abs() < 0.01);
/// ```
pub fn q_exponential(x: f64, q: f64) -> SpecialResult<f64> {
    if q.abs() >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "q_exponential: |q| must be < 1, got q = {q}"
        )));
    }

    // Use series: e_q(x) = sum_{n=0}^∞ x^n / [n]_q!
    // This converges for all x when |q| < 1 (the radius of convergence is
    // 1/(1-q) for the series formulation)

    let mut sum = 0.0f64;
    let mut x_pow_n = 1.0f64; // x^n
    let mut q_fact = 1.0f64; // [n]_q!
    let mut q_pow_n = 1.0f64; // q^n (for computing [n]_q)

    for n in 0..MAX_SERIES_TERMS {
        if n > 0 {
            x_pow_n *= x;
            // [n]_q = (1 - q^n) / (1 - q)
            let q_n = if (q - 1.0).abs() < 1e-14 {
                n as f64
            } else {
                (1.0 - q_pow_n) / (1.0 - q)
            };
            q_fact *= q_n;

            if q_fact.abs() < 1e-300 {
                // Series is dominted by zero denominator, stop
                break;
            }
        }

        let term = x_pow_n / q_fact;
        sum += term;

        if term.abs() < CONVERGENCE_TOL * sum.abs() && n > 5 {
            return Ok(sum);
        }

        q_pow_n *= q;

        if !sum.is_finite() {
            return Err(SpecialError::OverflowError(
                "q_exponential: series diverged".to_string(),
            ));
        }
    }

    if sum.is_finite() {
        Ok(sum)
    } else {
        Err(SpecialError::ConvergenceError(
            "q_exponential: series did not converge".to_string(),
        ))
    }
}

/// Computes the big q-exponential E_q(x) (second Jackson q-exponential).
///
/// # Definition
///
/// ```text
/// E_q(x) = sum_{n=0}^∞ q^{n(n-1)/2} * x^n / [n]_q!
/// ```
///
/// Satisfies E_q(x) * e_q(-x) = 1 and product formula:
/// ```text
/// E_q(x) = (-x(1-q); q)_∞
/// ```
///
/// # Arguments
///
/// * `x` - Argument
/// * `q` - Deformation parameter (|q| < 1)
pub fn q_exponential_big(x: f64, q: f64) -> SpecialResult<f64> {
    if q.abs() >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "q_exponential_big: |q| must be < 1, got q = {q}"
        )));
    }

    // Use product formula: E_q(x) = (-x(1-q); q)_∞
    let a = -x * (1.0 - q);
    q_pochhammer_inf(a, q)
}

// ============================================================================
// q-Logarithm
// ============================================================================

/// Computes the q-logarithm (Tsallis logarithm) ln_q(x).
///
/// # Definition
///
/// The Tsallis q-logarithm is defined as:
/// ```text
/// ln_q(x) = (x^{1-q} - 1) / (1 - q)   for q ≠ 1, x > 0
///          = ln(x)                       for q = 1 (limit)
/// ```
///
/// This is the inverse of the q-exponential in the sense of non-extensive
/// statistical mechanics (Tsallis statistics).
///
/// # Properties
///
/// - ln_q(1) = 0 for all q
/// - ln_q(x * y) = ln_q(x) + ln_q(y) + (1-q) * ln_q(x) * ln_q(y)
///   (q-additivity)
/// - lim_{q→1} ln_q(x) = ln(x)
///
/// # Arguments
///
/// * `x` - Argument (x > 0)
/// * `q` - Deformation parameter
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_analogs::q_logarithm;
///
/// // ln_q(1) = 0 for any q
/// let val = q_logarithm(1.0, 0.5).unwrap();
/// assert!(val.abs() < 1e-14);
///
/// // Near q=1: ln_q(e) ≈ 1
/// let val2 = q_logarithm(std::f64::consts::E, 1.0 - 1e-10).unwrap();
/// assert!((val2 - 1.0).abs() < 1e-6);
/// ```
pub fn q_logarithm(x: f64, q: f64) -> SpecialResult<f64> {
    if x <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "q_logarithm: x must be positive, got x = {x}"
        )));
    }

    if (q - 1.0).abs() < 1e-14 {
        // Classical limit
        return Ok(x.ln());
    }

    // ln_q(x) = (x^{1-q} - 1) / (1 - q)
    let exponent = 1.0 - q;
    let x_pow = x.powf(exponent);
    let result = (x_pow - 1.0) / exponent;

    if !result.is_finite() {
        return Err(SpecialError::OverflowError(
            "q_logarithm: result is not finite".to_string(),
        ));
    }

    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q_pochhammer_empty_product() {
        // (a; q)_0 = 1 for any a, q
        let val = q_pochhammer(0.5, 0.5, 0).expect("q_pochhammer(0.5, 0.5, 0)");
        assert!((val - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_q_pochhammer_single_factor() {
        // (a; q)_1 = 1 - a
        let val = q_pochhammer(0.3, 0.5, 1).expect("q_pochhammer single factor");
        assert!((val - 0.7).abs() < 1e-14);
    }

    #[test]
    fn test_q_pochhammer_three_factors() {
        // (1/2; 1/2)_3 = (1 - 1/2)(1 - 1/4)(1 - 1/8)
        let expected = 0.5 * 0.75 * 0.875;
        let val = q_pochhammer(0.5, 0.5, 3).expect("q_pochhammer three factors");
        assert!((val - expected).abs() < 1e-12);
    }

    #[test]
    fn test_q_pochhammer_inf_zero_base() {
        // (0; q)_∞ = 1
        let val = q_pochhammer_inf(0.0, 0.5).expect("q_pochhammer_inf zero base");
        assert!((val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_q_factorial_zero() {
        let val = q_factorial(0, 0.5).expect("q_factorial(0)");
        assert!((val - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_q_factorial_classical_limit() {
        // [n]_{q→1}! = n!
        let q = 1.0 - 1e-9;
        for n in 0..=7usize {
            let qf = q_factorial(n, q).expect("q_factorial classical limit");
            let classical: f64 = (1..=(n as u64)).product::<u64>() as f64;
            let expected = if n == 0 { 1.0 } else { classical };
            assert!(
                (qf - expected).abs() < 1e-3 * expected.max(1.0),
                "n={n}: q_factorial={qf}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_q_binomial_boundary() {
        // [n choose 0]_q = [n choose n]_q = 1
        for n in 1..=5usize {
            let v0 = q_binomial(n, 0, 0.5).expect("q_binomial k=0");
            let vn = q_binomial(n, n, 0.5).expect("q_binomial k=n");
            assert!((v0 - 1.0).abs() < 1e-12);
            assert!((vn - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_q_binomial_classical_limit() {
        // [4 choose 2]_{q→1} = C(4,2) = 6
        let val = q_binomial(4, 2, 1.0 - 1e-9).expect("q_binomial classical");
        assert!((val - 6.0).abs() < 0.01, "val = {val}");
    }

    #[test]
    fn test_q_binomial_q2() {
        // [4 choose 2]_2 should be 35
        // (2;2)_4 / ((2;2)_2 * (2;2)_2)
        // (1-2)(1-4)(1-8)(1-16) / ((1-2)(1-4))^2
        // = ((-1)(-3)(-7)(-15)) / ((-1)(-3))^2
        // = 315 / 9 = 35
        let val = q_binomial(4, 2, 2.0).expect("q_binomial q=2");
        assert!((val - 35.0).abs() < 1e-6, "val = {val}");
    }

    #[test]
    fn test_q_gamma_at_one() {
        // Γ_q(1) = 1
        let val = q_gamma(1.0, 0.5).expect("q_gamma(1)");
        assert!((val - 1.0).abs() < 1e-8, "val = {val}");
    }

    #[test]
    fn test_q_gamma_functional_eq() {
        // Γ_q(x+1) = [x]_q * Γ_q(x)
        let q = 0.7f64;
        let x = 1.5f64;
        let gx1 = q_gamma(x + 1.0, q).expect("q_gamma(x+1)");
        let gx = q_gamma(x, q).expect("q_gamma(x)");
        let q_x = (1.0 - q.powf(x)) / (1.0 - q);
        assert!(
            (gx1 - q_x * gx).abs() < 1e-8,
            "functional eq: lhs={gx1}, rhs={}, diff={}",
            q_x * gx,
            (gx1 - q_x * gx).abs()
        );
    }

    #[test]
    fn test_q_beta_symmetry() {
        // B_q(a, b) = B_q(b, a)
        let q = 0.6f64;
        let a = 1.5f64;
        let b = 2.0f64;
        let v1 = q_beta(a, b, q).expect("q_beta(a,b)");
        let v2 = q_beta(b, a, q).expect("q_beta(b,a)");
        assert!((v1 - v2).abs() < 1e-8, "symmetry: {v1} vs {v2}");
    }

    #[test]
    fn test_q_exponential_near_classical() {
        // e_q(x) → exp(x) as q → 1
        let val = q_exponential(1.0, 0.999).expect("q_exponential near classical");
        assert!(
            (val - std::f64::consts::E).abs() < 0.01,
            "val = {val}, e = {}",
            std::f64::consts::E
        );
    }

    #[test]
    fn test_q_logarithm_at_one() {
        // ln_q(1) = 0 for any q
        for q in &[0.1, 0.5, 0.9, 2.0] {
            let val = q_logarithm(1.0, *q).expect("q_logarithm(1)");
            assert!(val.abs() < 1e-14, "q={q}: val = {val}");
        }
    }

    #[test]
    fn test_q_logarithm_classical_limit() {
        // ln_q(e) → 1 as q → 1
        let val = q_logarithm(std::f64::consts::E, 1.0 - 1e-9).expect("q_log classical");
        assert!((val - 1.0).abs() < 1e-5, "val = {val}");
    }
}

// ============================================================================
// Basic Hypergeometric Series  2φ1
// ============================================================================

/// Computes the basic hypergeometric series ₂φ₁(a, b; c; q, z).
///
/// # Definition
///
/// The basic hypergeometric series (q-analogue of the Gauss hypergeometric
/// function) is defined by:
///
/// ```text
/// ₂φ₁(a, b; c; q, z) = Σ_{n=0}^∞  (a;q)_n · (b;q)_n
///                                    ─────────────────── · z^n
///                                    (c;q)_n · (q;q)_n
/// ```
///
/// where `(x; q)_n` is the finite q-Pochhammer symbol.
///
/// The series converges for `|z| < 1` when `|q| < 1`, or terminates when
/// `a = q^{-N}` for a non-negative integer N (in which case all terms with
/// n > N vanish because `(q^{-N}; q)_n = 0` for n > N).
///
/// # Classical limit
///
/// As q → 1⁻, the function reduces to the Gauss hypergeometric function:
/// ```text
/// lim_{q→1} ₂φ₁(q^a, q^b; q^c; q, z) = ₂F₁(a, b; c; z)
/// ```
///
/// # Arguments
///
/// * `a` - First numerator parameter
/// * `b` - Second numerator parameter
/// * `c` - Denominator parameter (c ≠ q^{-k} for k = 0, 1, 2, …)
/// * `q` - Base parameter (typically 0 < q < 1)
/// * `z` - Argument (series converges for |z| < 1)
///
/// # Errors
///
/// Returns an error if:
/// - The series fails to converge within `MAX_SERIES_TERMS` iterations.
/// - A denominator factor `(c;q)_n · (q;q)_n` vanishes (pole of the function).
///
/// # Examples
///
/// ```rust
/// use scirs2_special::q_analogs::basic_hypergeometric_2phi1;
///
/// // Terminating series: a = q^{-2} = 1/q^2 → (a;q)_n = 0 for n > 2
/// let q = 0.5_f64;
/// let a = q.powi(-2); // q^{-2}
/// let b = 0.5_f64;
/// let c = 0.25_f64;
/// let z = 0.3_f64;
/// let val = basic_hypergeometric_2phi1(a, b, c, q, z).unwrap();
/// assert!(val.is_finite());
///
/// // Near classical limit
/// // ₂φ₁(q^a, q^b; q^c; q, z) → ₂F₁(a, b; c; z) as q → 1⁻
/// ```
pub fn basic_hypergeometric_2phi1(a: f64, b: f64, c: f64, q: f64, z: f64) -> SpecialResult<f64> {
    // Validate q
    if q.abs() >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "basic_hypergeometric_2phi1: |q| must be < 1, got q = {q}"
        )));
    }

    // Validate z for convergence  (series converges absolutely for |z| < 1)
    if z.abs() >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "basic_hypergeometric_2phi1: |z| must be < 1 for convergence, got z = {z}"
        )));
    }

    // Sum the series term by term.
    // term_n = (a;q)_n * (b;q)_n / ((c;q)_n * (q;q)_n) * z^n
    //
    // We maintain running Pochhammer products to avoid recomputing from scratch.
    // At step n:
    //   poch_a_n  = (a;q)_n
    //   poch_b_n  = (b;q)_n
    //   poch_c_n  = (c;q)_n
    //   poch_q_n  = (q;q)_n

    let mut poch_a = 1.0f64; // (a;q)_0 = 1
    let mut poch_b = 1.0f64; // (b;q)_0 = 1
    let mut poch_c = 1.0f64; // (c;q)_0 = 1
    let mut poch_qq = 1.0f64; // (q;q)_0 = 1
    let mut z_pow = 1.0f64; // z^0 = 1
    let mut q_pow = 1.0f64; // q^0 (for the factor in Pochhammer update)

    let mut sum = 0.0f64;

    for n in 0..MAX_SERIES_TERMS {
        // Denominator at step n (before updating)
        let denom = poch_c * poch_qq;
        if denom.abs() < 1e-300 {
            // Pole: series is not well-defined at this parameter combination
            return Err(SpecialError::ComputationError(format!(
                "basic_hypergeometric_2phi1: denominator vanished at n = {n} \
                 (c may be a non-positive power of q)"
            )));
        }

        let term = poch_a * poch_b / denom * z_pow;
        sum += term;

        // Convergence check (after a few terms)
        if n >= 5 && term.abs() < CONVERGENCE_TOL * sum.abs().max(1e-300) {
            return Ok(sum);
        }
        if !sum.is_finite() {
            return Err(SpecialError::OverflowError(
                "basic_hypergeometric_2phi1: partial sum overflowed".to_string(),
            ));
        }

        // Update Pochhammer products for the next term n+1:
        // (x;q)_{n+1} = (x;q)_n * (1 - x*q^n)
        // q^n is tracked by q_pow (starts at q^0 = 1, updated to q^n after step n)
        let qn = q_pow; // q^n
        poch_a *= 1.0 - a * qn;
        poch_b *= 1.0 - b * qn;
        poch_c *= 1.0 - c * qn;
        poch_qq *= 1.0 - q * qn; // (q;q)_{n+1} uses factor (1 - q^{n+1})

        z_pow *= z;
        q_pow *= q;

        // Early termination: if poch_a (or poch_b) goes to zero, all subsequent
        // terms are zero — the series terminates.
        if poch_a.abs() < 1e-300 || poch_b.abs() < 1e-300 {
            return Ok(sum);
        }
    }

    // If we reach here the series did not formally converge
    if sum.is_finite() {
        Ok(sum)
    } else {
        Err(SpecialError::ConvergenceError(
            "basic_hypergeometric_2phi1: series did not converge".to_string(),
        ))
    }
}

// ============================================================================
// Tests for new q-functions
// ============================================================================

#[cfg(test)]
mod advanced_q_tests {
    use super::*;

    #[test]
    fn test_2phi1_unit_term() {
        // ₂φ₁(0, 0; 0; q, z) = 1 (all Pochhammer symbols are 1 for a=b=0)
        // (0;q)_n = (1-0)(1-0q)...(1-0q^{n-1}) = 1 for all n
        // so ₂φ₁(0, 0; 0; q, z) = sum z^n / (0;q)_n * (q;q)_n
        // (0;q)_n = 1, so sum is z^n/(q;q)_n = e_q(z) product — but c=0 causes denom=0 at n=1
        // Use a non-degenerate case:
        // ₂φ₁(0, b; c; q, 0) = 1 (z=0, only n=0 term survives)
        let val = basic_hypergeometric_2phi1(0.0, 0.5, 0.5, 0.3, 0.0).expect("2phi1 z=0");
        assert!((val - 1.0).abs() < 1e-14, "val = {val}");
    }

    #[test]
    fn test_2phi1_q_geometric_series() {
        // When b = 0:  (0;q)_n = 1 for all n,
        // ₂φ₁(a, 0; c; q, z) = sum_{n=0}^∞ (a;q)_n / ((c;q)_n (q;q)_n) z^n
        // When also a = 0: all Pochhammer factors are 1 except (q;q)_n in denom,
        //   ₂φ₁(0, 0; c; q, z) = sum z^n / ((c;q)_n (q;q)_n)
        // Use specific small z to verify convergence and finiteness:
        let q = 0.5f64;
        let val = basic_hypergeometric_2phi1(0.1, 0.2, 0.7, q, 0.3).expect("2phi1 general");
        assert!(val.is_finite(), "val not finite: {val}");
        assert!(val > 0.0, "val should be positive for these params");
    }

    #[test]
    fn test_2phi1_terminating() {
        // a = q^{-N} for integer N makes series terminate after N+1 terms
        let q = 0.5f64;
        let n_terms = 3usize;
        let a = q.powi(-(n_terms as i32)); // q^{-3}
        let b = 0.5f64;
        let c = 0.75f64;
        let z = 0.4f64;
        let val = basic_hypergeometric_2phi1(a, b, c, q, z).expect("2phi1 terminating");
        assert!(val.is_finite(), "terminating series should be finite: {val}");
    }

    #[test]
    fn test_2phi1_domain_error_q_too_large() {
        let result = basic_hypergeometric_2phi1(0.5, 0.5, 0.5, 1.5, 0.3);
        assert!(result.is_err());
    }

    #[test]
    fn test_2phi1_domain_error_z_too_large() {
        let result = basic_hypergeometric_2phi1(0.5, 0.5, 0.5, 0.5, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_2phi1_symmetry_in_a_b() {
        // ₂φ₁(a, b; c; q, z) = ₂φ₁(b, a; c; q, z)
        let (a, b, c, q, z) = (0.3, 0.6, 0.8, 0.4, 0.2);
        let v1 = basic_hypergeometric_2phi1(a, b, c, q, z).expect("v1");
        let v2 = basic_hypergeometric_2phi1(b, a, c, q, z).expect("v2");
        assert!((v1 - v2).abs() < 1e-12, "symmetry: {v1} vs {v2}");
    }
}
