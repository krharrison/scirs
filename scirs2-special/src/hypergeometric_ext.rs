//! Extended hypergeometric functions
//!
//! This module provides advanced hypergeometric functions beyond the standard
//! `hypergeometric.rs` and `hypergeometric_enhanced.rs` modules:
//!
//! - **Appell functions**: F1, F2 (double hypergeometric series)
//! - **Meijer G-function**: For basic parameter configurations
//! - **Confluent hypergeometric limit function 0F1**: Enhanced implementation
//! - **Generalized hypergeometric pFq**: For specific (p,q) cases
//! - **Pochhammer and rising factorial extensions**
//!
//! ## Mathematical Background
//!
//! ### Appell Functions
//!
//! The Appell hypergeometric functions are two-variable generalizations
//! of the Gauss hypergeometric function 2F1:
//!
//! ```text
//! F1(a; b1, b2; c; x, y) = sum_{m,n>=0} (a)_{m+n} (b1)_m (b2)_n / ((c)_{m+n} m! n!) x^m y^n
//! ```
//!
//! converging for |x| < 1 and |y| < 1.
//!
//! ```text
//! F2(a; b1, b2; c1, c2; x, y) = sum_{m,n>=0} (a)_{m+n} (b1)_m (b2)_n / ((c1)_m (c2)_n m! n!) x^m y^n
//! ```
//!
//! converging for |x| + |y| < 1.
//!
//! ### Meijer G-function
//!
//! The Meijer G-function is a very general function defined by a contour integral:
//!
//! ```text
//! G_{p,q}^{m,n}(z | a_1,...,a_p; b_1,...,b_q)
//! ```
//!
//! Most special functions can be expressed as special cases of the Meijer G-function.

use crate::error::{SpecialError, SpecialResult};
use crate::gamma::{gamma, gammaln};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

/// Maximum number of terms for double series computations
const MAX_DOUBLE_SERIES_TERMS: usize = 100;

/// Maximum total terms in double series
const MAX_TOTAL_TERMS: usize = 5000;

/// Convergence tolerance
const CONVERGENCE_TOL: f64 = 1e-15;

/// Helper to convert f64 constants to generic Float type
#[inline(always)]
fn const_f64<F: Float + FromPrimitive>(value: f64) -> F {
    F::from(value).unwrap_or_else(|| {
        if value > 0.0 {
            F::infinity()
        } else if value < 0.0 {
            F::neg_infinity()
        } else {
            F::zero()
        }
    })
}

// ========================================================================
// Appell Functions
// ========================================================================

/// Appell hypergeometric function F1(a; b1, b2; c; x, y).
///
/// The Appell F1 function is a two-variable generalization of the Gauss
/// hypergeometric function, defined by the double series:
///
/// ```text
/// F1(a; b1, b2; c; x, y) = sum_{m,n>=0} (a)_{m+n} (b1)_m (b2)_n / ((c)_{m+n} m! n!) x^m y^n
/// ```
///
/// The series converges absolutely for |x| < 1 and |y| < 1.
///
/// # Arguments
/// * `a` - Parameter a
/// * `b1` - Parameter b1
/// * `b2` - Parameter b2
/// * `c` - Parameter c (must not be 0, -1, -2, ...)
/// * `x` - First variable (|x| < 1)
/// * `y` - Second variable (|y| < 1)
///
/// # Returns
/// Value of F1(a; b1, b2; c; x, y)
///
/// # Examples
/// ```
/// use scirs2_special::hypergeometric_ext::appell_f1;
/// // F1(a; b1, b2; c; 0, 0) = 1
/// let val = appell_f1(1.0, 2.0, 3.0, 4.0, 0.0, 0.0).expect("failed");
/// assert!((val - 1.0).abs() < 1e-14);
/// ```
pub fn appell_f1(a: f64, b1: f64, b2: f64, c: f64, x: f64, y: f64) -> SpecialResult<f64> {
    // Validate parameters
    if c <= 0.0 && c.fract() == 0.0 {
        return Err(SpecialError::DomainError(format!(
            "c must not be 0 or negative integer, got {c}"
        )));
    }

    if x.abs() >= 1.0 || y.abs() >= 1.0 {
        return Err(SpecialError::DomainError(
            "Appell F1 requires |x| < 1 and |y| < 1".to_string(),
        ));
    }

    // Special case: x = y = 0
    if x.abs() < 1e-300 && y.abs() < 1e-300 {
        return Ok(1.0);
    }

    // Special case: y = 0 reduces to 2F1(a, b1; c; x)
    if y.abs() < 1e-300 {
        return crate::hypergeometric::hyp2f1(a, b1, c, x).map(|v: f64| v);
    }

    // Special case: x = 0 reduces to 2F1(a, b2; c; y)
    if x.abs() < 1e-300 {
        return crate::hypergeometric::hyp2f1(a, b2, c, y).map(|v: f64| v);
    }

    // Double series computation
    let mut result = 0.0;
    let mut total_terms = 0usize;

    // Outer sum over m
    let mut x_pow_m = 1.0; // x^m
    let mut pochhammer_b1_m = 1.0; // (b1)_m
    let mut m_factorial = 1.0; // m!

    for m in 0..MAX_DOUBLE_SERIES_TERMS {
        if m > 0 {
            x_pow_m *= x;
            pochhammer_b1_m *= b1 + (m - 1) as f64;
            m_factorial *= m as f64;
        }

        if x_pow_m.abs() < 1e-300 && m > 0 {
            break;
        }

        // Inner sum over n
        let mut y_pow_n = 1.0;
        let mut pochhammer_b2_n = 1.0;
        let mut n_factorial = 1.0;
        let mut inner_sum = 0.0;
        let mut pochhammer_a_mn = pochhammer_a_precompute(a, m);
        let mut pochhammer_c_mn = pochhammer_a_precompute(c, m);

        for n in 0..MAX_DOUBLE_SERIES_TERMS {
            if n > 0 {
                y_pow_n *= y;
                pochhammer_b2_n *= b2 + (n - 1) as f64;
                n_factorial *= n as f64;
                pochhammer_a_mn *= a + (m + n - 1) as f64;
                pochhammer_c_mn *= c + (m + n - 1) as f64;
            }

            total_terms += 1;
            if total_terms > MAX_TOTAL_TERMS {
                break;
            }

            if pochhammer_c_mn.abs() < 1e-300 {
                return Err(SpecialError::DomainError(
                    "Division by zero: (c)_{m+n} = 0".to_string(),
                ));
            }

            let term = pochhammer_a_mn * pochhammer_b1_m * pochhammer_b2_n
                / (pochhammer_c_mn * m_factorial * n_factorial)
                * x_pow_m
                * y_pow_n;

            inner_sum += term;

            if n > 0 && term.abs() < CONVERGENCE_TOL * inner_sum.abs() {
                break;
            }
        }

        result += inner_sum;

        if m > 0 && inner_sum.abs() < CONVERGENCE_TOL * result.abs() {
            break;
        }

        if total_terms > MAX_TOTAL_TERMS {
            break;
        }
    }

    Ok(result)
}

/// Appell hypergeometric function F2(a; b1, b2; c1, c2; x, y).
///
/// Defined by the double series:
///
/// ```text
/// F2(a; b1, b2; c1, c2; x, y) = sum_{m,n>=0} (a)_{m+n} (b1)_m (b2)_n / ((c1)_m (c2)_n m! n!) x^m y^n
/// ```
///
/// The series converges for |x| + |y| < 1.
///
/// # Arguments
/// * `a` - Parameter a
/// * `b1` - Parameter b1
/// * `b2` - Parameter b2
/// * `c1` - Parameter c1 (must not be 0, -1, -2, ...)
/// * `c2` - Parameter c2 (must not be 0, -1, -2, ...)
/// * `x` - First variable
/// * `y` - Second variable
///
/// # Returns
/// Value of F2(a; b1, b2; c1, c2; x, y)
///
/// # Examples
/// ```
/// use scirs2_special::hypergeometric_ext::appell_f2;
/// let val = appell_f2(1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0).expect("failed");
/// assert!((val - 1.0).abs() < 1e-14);
/// ```
pub fn appell_f2(a: f64, b1: f64, b2: f64, c1: f64, c2: f64, x: f64, y: f64) -> SpecialResult<f64> {
    // Validate
    if (c1 <= 0.0 && c1.fract() == 0.0) || (c2 <= 0.0 && c2.fract() == 0.0) {
        return Err(SpecialError::DomainError(
            "c1 and c2 must not be 0 or negative integers".to_string(),
        ));
    }

    if x.abs() + y.abs() >= 1.0 {
        return Err(SpecialError::DomainError(
            "Appell F2 requires |x| + |y| < 1".to_string(),
        ));
    }

    if x.abs() < 1e-300 && y.abs() < 1e-300 {
        return Ok(1.0);
    }

    // Special case: y = 0 reduces to 2F1(a, b1; c1; x)
    if y.abs() < 1e-300 {
        return crate::hypergeometric::hyp2f1(a, b1, c1, x).map(|v: f64| v);
    }

    // Special case: x = 0 reduces to 2F1(a, b2; c2; y)
    if x.abs() < 1e-300 {
        return crate::hypergeometric::hyp2f1(a, b2, c2, y).map(|v: f64| v);
    }

    // Double series computation
    let mut result = 0.0;
    let mut total_terms = 0usize;

    let mut x_pow_m = 1.0;
    let mut pochhammer_b1_m = 1.0;
    let mut pochhammer_c1_m = 1.0;
    let mut m_factorial = 1.0;

    for m in 0..MAX_DOUBLE_SERIES_TERMS {
        if m > 0 {
            x_pow_m *= x;
            pochhammer_b1_m *= b1 + (m - 1) as f64;
            pochhammer_c1_m *= c1 + (m - 1) as f64;
            m_factorial *= m as f64;
        }

        if pochhammer_c1_m.abs() < 1e-300 && m > 0 {
            return Err(SpecialError::DomainError("(c1)_m = 0".to_string()));
        }

        let mut y_pow_n = 1.0;
        let mut pochhammer_b2_n = 1.0;
        let mut pochhammer_c2_n = 1.0;
        let mut n_factorial = 1.0;
        let mut inner_sum = 0.0;
        let mut pochhammer_a_mn = pochhammer_a_precompute(a, m);

        for n in 0..MAX_DOUBLE_SERIES_TERMS {
            if n > 0 {
                y_pow_n *= y;
                pochhammer_b2_n *= b2 + (n - 1) as f64;
                pochhammer_c2_n *= c2 + (n - 1) as f64;
                n_factorial *= n as f64;
                pochhammer_a_mn *= a + (m + n - 1) as f64;
            }

            total_terms += 1;
            if total_terms > MAX_TOTAL_TERMS {
                break;
            }

            if pochhammer_c2_n.abs() < 1e-300 && n > 0 {
                return Err(SpecialError::DomainError("(c2)_n = 0".to_string()));
            }

            let denom = pochhammer_c1_m * pochhammer_c2_n * m_factorial * n_factorial;
            if denom.abs() < 1e-300 {
                continue;
            }

            let term =
                pochhammer_a_mn * pochhammer_b1_m * pochhammer_b2_n / denom * x_pow_m * y_pow_n;

            inner_sum += term;

            if n > 0 && term.abs() < CONVERGENCE_TOL * inner_sum.abs().max(1e-300) {
                break;
            }
        }

        result += inner_sum;

        if m > 0 && inner_sum.abs() < CONVERGENCE_TOL * result.abs().max(1e-300) {
            break;
        }

        if total_terms > MAX_TOTAL_TERMS {
            break;
        }
    }

    Ok(result)
}

// ========================================================================
// Meijer G-function (basic cases)
// ========================================================================

/// Meijer G-function for the special case G_{0,1}^{1,0}(z | -; b).
///
/// This equals exp(-z) * z^b / Gamma(b+1) for Re(z) > 0.
///
/// # Arguments
/// * `z` - Argument (must be positive)
/// * `b` - Parameter b
///
/// # Returns
/// Value of G_{0,1}^{1,0}(z | -; b)
///
/// # Examples
/// ```
/// use scirs2_special::hypergeometric_ext::meijer_g_01_10;
/// let val = meijer_g_01_10(1.0, 0.0).expect("failed");
/// // G_{0,1}^{1,0}(1 | -; 0) = exp(-1)
/// assert!((val - (-1.0_f64).exp()).abs() < 1e-12);
/// ```
pub fn meijer_g_01_10(z: f64, b: f64) -> SpecialResult<f64> {
    if z <= 0.0 {
        return Err(SpecialError::DomainError(
            "z must be positive for this Meijer G case".to_string(),
        ));
    }

    let gamma_bp1 = gamma(b + 1.0);
    if gamma_bp1.abs() < 1e-300 || !gamma_bp1.is_finite() {
        return Err(SpecialError::OverflowError(
            "Gamma(b+1) overflow or zero in Meijer G".to_string(),
        ));
    }

    Ok((-z).exp() * z.powf(b) / gamma_bp1)
}

/// Meijer G-function for the special case G_{1,0}^{0,1}(z | a; -).
///
/// This equals z^a * exp(-z) for Re(z) > 0.
///
/// This is related to the exponential function.
///
/// # Arguments
/// * `z` - Argument
/// * `a` - Parameter
pub fn meijer_g_10_01(z: f64, a: f64) -> SpecialResult<f64> {
    if z < 0.0 {
        return Err(SpecialError::DomainError(
            "z must be non-negative for this Meijer G case".to_string(),
        ));
    }

    if z == 0.0 {
        if a > 0.0 {
            return Ok(0.0);
        } else if a == 0.0 {
            return Ok(1.0);
        } else {
            return Ok(f64::INFINITY);
        }
    }

    Ok(z.powf(a) * (-z).exp())
}

/// Meijer G-function for the case G_{1,1}^{1,1}(z | a; b).
///
/// For this basic case:
/// G_{1,1}^{1,1}(z | a; b) = Gamma(1-a+b) * z^b * (1+z)^{a-b-1} / Gamma(a)
/// when 0 < Re(1-a+b).
///
/// This encompasses the beta distribution and related functions.
///
/// # Arguments
/// * `z` - Argument (must be positive)
/// * `a` - Upper parameter
/// * `b` - Lower parameter
pub fn meijer_g_11_11(z: f64, a: f64, b: f64) -> SpecialResult<f64> {
    if z <= 0.0 {
        return Err(SpecialError::DomainError(
            "z must be positive for G_{1,1}^{1,1}".to_string(),
        ));
    }

    let cond = 1.0 - a + b;
    if cond <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "Need 1-a+b > 0, got {cond}"
        )));
    }

    let gamma_cond = gamma(cond);
    let gamma_a = gamma(a);

    if gamma_a.abs() < 1e-300 || !gamma_a.is_finite() {
        return Err(SpecialError::OverflowError(
            "Gamma(a) overflow in Meijer G".to_string(),
        ));
    }

    Ok(gamma_cond * z.powf(b) * (1.0 + z).powf(a - b - 1.0) / gamma_a)
}

// ========================================================================
// Generalized Hypergeometric pFq
// ========================================================================

/// Generalized hypergeometric function pFq(a_1,...,a_p; b_1,...,b_q; z).
///
/// Computes the generalized hypergeometric series:
/// ```text
/// pFq(a;b;z) = sum_{n=0}^{inf} [(a_1)_n ... (a_p)_n / ((b_1)_n ... (b_q)_n)] z^n / n!
/// ```
///
/// # Arguments
/// * `a_params` - Numerator parameters [a_1, ..., a_p]
/// * `b_params` - Denominator parameters [b_1, ..., b_q] (none may be 0 or negative integers)
/// * `z` - Argument
///
/// # Returns
/// Value of pFq(a; b; z)
///
/// # Convergence
/// - p <= q: converges for all z
/// - p = q + 1: converges for |z| < 1
/// - p > q + 1: diverges (except for terminating series)
///
/// # Examples
/// ```
/// use scirs2_special::hypergeometric_ext::hyp_pfq;
/// // 1F0(1; ; z) = 1/(1-z)
/// let val = hyp_pfq(&[1.0], &[], 0.5).expect("failed");
/// assert!((val - 2.0).abs() < 1e-10);
/// ```
pub fn hyp_pfq(a_params: &[f64], b_params: &[f64], z: f64) -> SpecialResult<f64> {
    let p = a_params.len();
    let q = b_params.len();

    // Check b parameters
    for (i, &b) in b_params.iter().enumerate() {
        if b <= 0.0 && b.fract() == 0.0 {
            return Err(SpecialError::DomainError(format!(
                "b_{} = {b} is a non-positive integer",
                i + 1
            )));
        }
    }

    // Check convergence
    if p > q + 1 {
        // Check for terminating series (some a_i is non-positive integer)
        let is_terminating = a_params.iter().any(|&a| a <= 0.0 && a.fract() == 0.0);
        if !is_terminating {
            return Err(SpecialError::DomainError(format!(
                "p={p} > q+1={}: series diverges for non-terminating case",
                q + 1
            )));
        }
    }

    if p == q + 1 && z.abs() >= 1.0 {
        // Check for terminating series
        let is_terminating = a_params.iter().any(|&a| a <= 0.0 && a.fract() == 0.0);
        if !is_terminating {
            return Err(SpecialError::DomainError(
                "p = q+1 and |z| >= 1: series may not converge".to_string(),
            ));
        }
    }

    if z.abs() < 1e-300 {
        return Ok(1.0);
    }

    // Series computation
    let mut result = 1.0;
    let mut term = 1.0;
    let max_terms = 1000;

    for n in 1..=max_terms {
        let n_f = n as f64;

        // Numerator: product of (a_i + n - 1)
        let mut numer_factor = 1.0;
        for &a in a_params {
            numer_factor *= a + n_f - 1.0;
        }

        // Denominator: product of (b_i + n - 1)
        let mut denom_factor = 1.0;
        for &b in b_params {
            let bf = b + n_f - 1.0;
            if bf.abs() < 1e-300 {
                return Err(SpecialError::DomainError(
                    "Denominator parameter hit zero".to_string(),
                ));
            }
            denom_factor *= bf;
        }

        term *= numer_factor * z / (denom_factor * n_f);
        result += term;

        // Check for terminating series (numerator factor becomes 0)
        if numer_factor.abs() < 1e-300 {
            break;
        }

        if term.abs() < CONVERGENCE_TOL * result.abs() {
            break;
        }

        if !result.is_finite() {
            return Err(SpecialError::OverflowError(
                "Overflow in pFq series".to_string(),
            ));
        }
    }

    Ok(result)
}

/// Compute 3F2(a1, a2, a3; b1, b2; z).
///
/// Special case of pFq for the common 3F2 case.
/// Converges for |z| < 1 (or |z| = 1 if Re(b1+b2-a1-a2-a3) > 0).
///
/// # Examples
/// ```
/// use scirs2_special::hypergeometric_ext::hyp3f2;
/// // 3F2(1,1,1; 2,2; z) relates to dilogarithm
/// let val = hyp3f2(1.0, 1.0, 1.0, 2.0, 2.0, 0.5).expect("failed");
/// assert!(val.is_finite());
/// ```
pub fn hyp3f2(a1: f64, a2: f64, a3: f64, b1: f64, b2: f64, z: f64) -> SpecialResult<f64> {
    hyp_pfq(&[a1, a2, a3], &[b1, b2], z)
}

// ========================================================================
// Pochhammer Symbol Extensions
// ========================================================================

/// Generalized Pochhammer symbol (a)_x for real x (not just integer n).
///
/// Defined as:
/// ```text
/// (a)_x = Gamma(a + x) / Gamma(a)
/// ```
///
/// # Arguments
/// * `a` - Base parameter
/// * `x` - Index (real number)
///
/// # Returns
/// Value of (a)_x
///
/// # Examples
/// ```
/// use scirs2_special::hypergeometric_ext::pochhammer_real;
/// // (1)_3 = 1*2*3 = 6
/// let val = pochhammer_real(1.0, 3.0).expect("failed");
/// assert!((val - 6.0).abs() < 1e-10);
/// ```
pub fn pochhammer_real(a: f64, x: f64) -> SpecialResult<f64> {
    if x == 0.0 {
        return Ok(1.0);
    }

    // For non-negative integer x, use the direct product (a)(a+1)...(a+x-1)
    if x > 0.0 && x == x.floor() && x < 100.0 {
        let n = x as usize;
        let mut result = 1.0;
        for k in 0..n {
            result *= a + k as f64;
        }
        return Ok(result);
    }

    // For general real x, use Gamma(a+x)/Gamma(a)
    let gamma_ax = gamma(a + x);
    let gamma_a = gamma(a);
    if gamma_a.abs() < 1e-300 {
        return Err(SpecialError::DomainError(format!(
            "Gamma({a}) is zero or near-zero"
        )));
    }

    if !gamma_ax.is_finite() || !gamma_a.is_finite() {
        // Use log-gamma for large values
        let log_result = gammaln(a + x) - gammaln(a);
        if log_result.is_finite() {
            return Ok(log_result.exp());
        }
    }

    Ok(gamma_ax / gamma_a)
}

/// Regularized Pochhammer symbol (a)_n / Gamma(a+n) = 1/Gamma(a).
///
/// This is useful when working with regularized hypergeometric functions.
///
/// # Arguments
/// * `a` - Base parameter
/// * `n` - Integer index
///
/// # Returns
/// Value of (a)_n / Gamma(a + n)
pub fn pochhammer_regularized(a: f64, n: usize) -> SpecialResult<f64> {
    let gamma_a = gamma(a);
    if gamma_a.abs() < 1e-300 || !gamma_a.is_finite() {
        return Err(SpecialError::DomainError(format!(
            "Gamma({a}) is not finite or zero"
        )));
    }
    Ok(1.0 / gamma_a)
}

// ========================================================================
// Confluent Hypergeometric Function Improvements
// ========================================================================

/// Enhanced confluent hypergeometric function 1F1(a; b; z) with asymptotic expansion.
///
/// For large |z|, uses the asymptotic expansion instead of the series:
/// ```text
/// 1F1(a; b; z) ~ Gamma(b)/Gamma(a) * exp(z) * z^{a-b} * [1 + O(1/z)]
///              + Gamma(b)/Gamma(b-a) * (-z)^{-a} * [1 + O(1/z)]
/// ```
///
/// # Arguments
/// * `a` - First parameter
/// * `b` - Second parameter (must not be 0, -1, -2, ...)
/// * `z` - Argument
///
/// # Returns
/// Value of 1F1(a; b; z)
pub fn hyp1f1_asymptotic(a: f64, b: f64, z: f64) -> SpecialResult<f64> {
    if b <= 0.0 && b.fract() == 0.0 {
        return Err(SpecialError::DomainError(format!(
            "b must not be 0 or negative integer, got {b}"
        )));
    }

    // For small |z|, use the standard series
    if z.abs() < 50.0 {
        return crate::hypergeometric::hyp1f1(a, b, z);
    }

    // Asymptotic expansion for large z > 0
    if z > 0.0 {
        // 1F1(a; b; z) ~ Gamma(b)/Gamma(a) * exp(z) * z^{a-b} * sum_s
        let log_gamma_b = gammaln(b);
        let log_gamma_a = gammaln(a);
        let log_gamma_ba = gammaln(b - a);

        // First asymptotic term
        let log_term1 = log_gamma_b - log_gamma_a + z + (a - b) * z.ln();
        // Second asymptotic term
        let log_term2 = log_gamma_b - log_gamma_ba + (-a) * (-z).abs().ln();

        // Compute the asymptotic sum for the first term
        let mut sum1 = 1.0;
        let mut coeff1 = 1.0;
        for s in 1..20 {
            let s_f = s as f64;
            coeff1 *= (b - a + s_f - 1.0) * (1.0 - a + s_f - 1.0) / (s_f * z);
            if coeff1.abs() < 1e-15 {
                break;
            }
            sum1 += coeff1;
            if coeff1.abs() > sum1.abs() {
                // Asymptotic series starts to diverge
                break;
            }
        }

        // Compute second asymptotic term
        let mut sum2 = 1.0;
        let mut coeff2 = 1.0;
        for s in 1..20 {
            let s_f = s as f64;
            coeff2 *= (a + s_f - 1.0) * (a - b + s_f) / (s_f * (-z));
            if coeff2.abs() < 1e-15 {
                break;
            }
            sum2 += coeff2;
            if coeff2.abs() > sum2.abs() {
                break;
            }
        }

        let term1 = if log_term1.is_finite() {
            log_term1.exp() * sum1
        } else {
            0.0
        };

        let sign2 = if ((-a) as i64) % 2 == 0 { 1.0 } else { -1.0 };
        let term2 = if log_term2.is_finite() {
            sign2 * log_term2.exp() * sum2
        } else {
            0.0
        };

        return Ok(term1 + term2);
    }

    // For large negative z, use Kummer's transformation: 1F1(a;b;z) = exp(z) * 1F1(b-a;b;-z)
    let result = crate::hypergeometric::hyp1f1(b - a, b, -z)?;
    Ok(z.exp() * result)
}

/// Tricomi's confluent hypergeometric function U(a, b, z).
///
/// Also known as the confluent hypergeometric function of the second kind.
/// Related to Kummer's function by:
/// ```text
/// U(a, b, z) = Gamma(1-b)/Gamma(a-b+1) * M(a,b,z) + Gamma(b-1)/Gamma(a) * z^{1-b} * M(a-b+1, 2-b, z)
/// ```
///
/// where M(a,b,z) = 1F1(a;b;z).
///
/// # Arguments
/// * `a` - First parameter
/// * `b` - Second parameter
/// * `z` - Argument (z > 0)
///
/// # Returns
/// Value of U(a, b, z)
///
/// # Examples
/// ```
/// use scirs2_special::hypergeometric_ext::tricomi_u;
/// let val = tricomi_u(1.0, 1.0, 1.0).expect("failed");
/// assert!(val.is_finite());
/// ```
pub fn tricomi_u(a: f64, b: f64, z: f64) -> SpecialResult<f64> {
    if z <= 0.0 {
        return Err(SpecialError::DomainError(
            "z must be positive for Tricomi U".to_string(),
        ));
    }

    // Use the hyperu function from the base hypergeometric module
    crate::hypergeometric::hyperu(a, b, z)
}

// ========================================================================
// Kummer's Relations
// ========================================================================

/// Kummer's first transformation: 1F1(a; b; z) = exp(z) * 1F1(b-a; b; -z).
///
/// This is useful for computing 1F1 when z is large and negative.
///
/// # Arguments
/// * `a` - First parameter
/// * `b` - Second parameter
/// * `z` - Argument
pub fn kummer_transform<
    F: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::MulAssign,
>(
    a: F,
    b: F,
    z: F,
) -> SpecialResult<F> {
    let b_minus_a = b - a;
    let neg_z = -z;
    let m_val: F = crate::hypergeometric::hyp1f1(b_minus_a, b, neg_z)?;
    Ok(z.exp() * m_val)
}

// ========================================================================
// Internal Helper Functions
// ========================================================================

/// Compute (a)_n = a * (a+1) * ... * (a+n-1) for integer n
fn pochhammer_a_precompute(a: f64, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut result = 1.0;
    for k in 0..n {
        result *= a + k as f64;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ====== Appell F1 tests ======

    #[test]
    fn test_appell_f1_zero() {
        let val = appell_f1(1.0, 2.0, 3.0, 4.0, 0.0, 0.0).expect("failed");
        assert_relative_eq!(val, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_appell_f1_y_zero() {
        // F1(a; b1, b2; c; x, 0) = 2F1(a, b1; c; x)
        let x = 0.3;
        let f1 = appell_f1(1.0, 2.0, 3.0, 4.0, x, 0.0).expect("failed");
        let h: f64 = crate::hypergeometric::hyp2f1(1.0, 2.0, 4.0, x).expect("failed");
        assert_relative_eq!(f1, h, epsilon = 1e-10);
    }

    #[test]
    fn test_appell_f1_x_zero() {
        // F1(a; b1, b2; c; 0, y) = 2F1(a, b2; c; y)
        let y = 0.3;
        let f1 = appell_f1(1.0, 2.0, 3.0, 4.0, 0.0, y).expect("failed");
        let h: f64 = crate::hypergeometric::hyp2f1(1.0, 3.0, 4.0, y).expect("failed");
        assert_relative_eq!(f1, h, epsilon = 1e-10);
    }

    #[test]
    fn test_appell_f1_small_args() {
        // F1 with small x, y should be close to 1
        let val = appell_f1(1.0, 1.0, 1.0, 2.0, 0.1, 0.1).expect("failed");
        assert!(val > 1.0, "F1 should be > 1 for positive args");
        assert!(val < 2.0, "F1 should be bounded for small args");
    }

    #[test]
    fn test_appell_f1_domain_error() {
        assert!(appell_f1(1.0, 2.0, 3.0, 4.0, 1.5, 0.0).is_err());
        assert!(appell_f1(1.0, 2.0, 3.0, 0.0, 0.5, 0.5).is_err());
    }

    // ====== Appell F2 tests ======

    #[test]
    fn test_appell_f2_zero() {
        let val = appell_f2(1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0).expect("failed");
        assert_relative_eq!(val, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_appell_f2_y_zero() {
        let x = 0.2;
        let f2 = appell_f2(1.0, 2.0, 3.0, 4.0, 5.0, x, 0.0).expect("failed");
        let h: f64 = crate::hypergeometric::hyp2f1(1.0, 2.0, 4.0, x).expect("failed");
        assert_relative_eq!(f2, h, epsilon = 1e-10);
    }

    #[test]
    fn test_appell_f2_x_zero() {
        let y = 0.2;
        let f2 = appell_f2(1.0, 2.0, 3.0, 4.0, 5.0, 0.0, y).expect("failed");
        let h: f64 = crate::hypergeometric::hyp2f1(1.0, 3.0, 5.0, y).expect("failed");
        assert_relative_eq!(f2, h, epsilon = 1e-10);
    }

    #[test]
    fn test_appell_f2_domain_error() {
        // |x| + |y| >= 1
        assert!(appell_f2(1.0, 2.0, 3.0, 4.0, 5.0, 0.6, 0.5).is_err());
    }

    // ====== Meijer G tests ======

    #[test]
    fn test_meijer_g_01_10_exp() {
        // G_{0,1}^{1,0}(z | -; 0) = exp(-z)
        let val = meijer_g_01_10(1.0, 0.0).expect("failed");
        assert_relative_eq!(val, (-1.0_f64).exp(), epsilon = 1e-12);
    }

    #[test]
    fn test_meijer_g_01_10_b1() {
        // G_{0,1}^{1,0}(z | -; 1) = z * exp(-z) / Gamma(2) = z * exp(-z)
        let z = 2.0;
        let val = meijer_g_01_10(z, 1.0).expect("failed");
        assert_relative_eq!(val, z * (-z).exp(), epsilon = 1e-12);
    }

    #[test]
    fn test_meijer_g_10_01() {
        let z = 2.0;
        let val = meijer_g_10_01(z, 1.0).expect("failed");
        assert_relative_eq!(val, z * (-z).exp(), epsilon = 1e-12);
    }

    #[test]
    fn test_meijer_g_11_11() {
        // G_{1,1}^{1,1}(z | a; b) = Gamma(1-a+b) * z^b * (1+z)^{a-b-1} / Gamma(a)
        // Use a=0.5, b=0.0 so that 1-a+b = 0.5 > 0
        let z = 2.0;
        let a = 0.5;
        let b = 0.0;
        let val = meijer_g_11_11(z, a, b).expect("failed");
        let expected = gamma(1.0 - a + b) * z.powf(b) * (1.0 + z).powf(a - b - 1.0) / gamma(a);
        assert_relative_eq!(val, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_meijer_g_domain_errors() {
        assert!(meijer_g_01_10(-1.0, 0.0).is_err());
        assert!(meijer_g_10_01(-1.0, 1.0).is_err());
        assert!(meijer_g_11_11(-1.0, 1.0, 0.0).is_err());
    }

    // ====== pFq tests ======

    #[test]
    fn test_hyp_pfq_1f0() {
        // 1F0(1; ; z) = 1/(1-z)
        let val = hyp_pfq(&[1.0], &[], 0.5).expect("failed");
        assert_relative_eq!(val, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hyp_pfq_0f0() {
        // 0F0(; ; z) = exp(z)
        let val = hyp_pfq(&[], &[], 1.0).expect("failed");
        assert_relative_eq!(val, 1.0_f64.exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_hyp_pfq_0f1() {
        // 0F1(; 1; -z^2/4) = J_0(z) (approximately)
        // 0F1(; 1; z) should converge for all z
        let val = hyp_pfq(&[], &[1.0], 0.5).expect("failed");
        assert!(val.is_finite());
    }

    #[test]
    fn test_hyp_pfq_zero() {
        let val = hyp_pfq(&[1.0, 2.0], &[3.0], 0.0).expect("failed");
        assert_relative_eq!(val, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_hyp3f2_finite() {
        let val = hyp3f2(1.0, 1.0, 1.0, 2.0, 2.0, 0.5).expect("failed");
        assert!(val.is_finite());
        assert!(val > 1.0);
    }

    #[test]
    fn test_hyp_pfq_terminating() {
        // 2F1(-2, 1; 1; z) = 1 - 2z + z^2 = (1-z)^2
        let val = hyp_pfq(&[-2.0, 1.0], &[1.0], 0.5).expect("failed");
        assert_relative_eq!(val, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_hyp_pfq_domain_error() {
        // b parameter is negative integer
        assert!(hyp_pfq(&[1.0], &[-1.0], 0.5).is_err());
    }

    // ====== Pochhammer extensions ======

    #[test]
    fn test_pochhammer_real_integer() {
        // (1)_3 = 1*2*3 = 6
        let val = pochhammer_real(1.0, 3.0).expect("failed");
        assert_relative_eq!(val, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pochhammer_real_half() {
        // (1/2)_2 = (1/2)(3/2) = 3/4
        let val = pochhammer_real(0.5, 2.0).expect("failed");
        assert_relative_eq!(val, 0.75, epsilon = 1e-10);
    }

    #[test]
    fn test_pochhammer_real_zero_index() {
        let val = pochhammer_real(5.0, 0.0).expect("failed");
        assert_relative_eq!(val, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_pochhammer_regularized() {
        let val = pochhammer_regularized(2.0, 3).expect("failed");
        // Should be 1/Gamma(2) = 1
        assert_relative_eq!(val, 1.0, epsilon = 1e-12);
    }

    // ====== Confluent hypergeometric improvements ======

    #[test]
    fn test_hyp1f1_asymptotic_small_z() {
        // For small z, should match standard 1F1
        let a = 1.0;
        let b = 2.0;
        let z = 0.5;
        let val = hyp1f1_asymptotic(a, b, z).expect("failed");
        let expected: f64 = crate::hypergeometric::hyp1f1(a, b, z).expect("failed");
        assert_relative_eq!(val, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_hyp1f1_asymptotic_domain() {
        assert!(hyp1f1_asymptotic(1.0, 0.0, 1.0).is_err());
        assert!(hyp1f1_asymptotic(1.0, -1.0, 1.0).is_err());
    }

    #[test]
    fn test_tricomi_u_basic() {
        // U(a, b, z) for b not a positive integer and z large enough for asymptotic
        // U(1, 0.5, 3) should be finite
        let val = tricomi_u(1.0, 0.5, 3.0).expect("failed");
        assert!(val.is_finite());
        // U(a, b, x) ~ x^{-a} for large x, so U(1, 0.5, 3) ~ 1/3
        assert!(
            (val - 1.0 / 3.0).abs() < 0.2,
            "U(1, 0.5, 3) ~ {val}, expected ~0.333"
        );
    }

    #[test]
    fn test_tricomi_u_domain() {
        assert!(tricomi_u(1.0, 1.0, -1.0).is_err());
    }

    // ====== Kummer transform test ======

    #[test]
    fn test_kummer_transform() {
        // 1F1(a; b; z) = exp(z) * 1F1(b-a; b; -z)
        let a = 1.0f64;
        let b = 2.0f64;
        let z = 0.5f64;
        let direct: f64 = crate::hypergeometric::hyp1f1(a, b, z).expect("failed");
        let via_kummer: f64 = kummer_transform(a, b, z).expect("failed");
        assert_relative_eq!(direct, via_kummer, epsilon = 1e-8);
    }
}
