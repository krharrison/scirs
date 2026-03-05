//! Appell hypergeometric functions, Horn functions, and Lauricella functions
//!
//! This module implements multivariate generalizations of the Gauss
//! hypergeometric function, including:
//!
//! - **Appell F1**: Two-variable function converging for |x| < 1, |y| < 1
//! - **Appell F2**: Two-variable function converging for |x| + |y| < 1
//! - **Appell F3**: Two-variable function (separate convergence constraints)
//! - **Appell F4**: Two-variable function converging for sqrt(|x|) + sqrt(|y|) < 1
//! - **Horn H1**: A double hypergeometric function from Horn's list
//! - **Lauricella FD**: Multi-variable generalization (n variables)
//! - **Lauricella FC**: Multi-variable generalization (n variables)
//!
//! ## Mathematical Background
//!
//! ### Appell F1
//!
//! ```text
//! F1(a; b1, b2; c; x, y)
//!   = sum_{m,n>=0} (a)_{m+n} (b1)_m (b2)_n / ((c)_{m+n} m! n!) * x^m y^n
//! ```
//!
//! Converges for |x| < 1, |y| < 1.
//! Reduces to 2F1 when y=0 (or x=0).
//!
//! ### Appell F2
//!
//! ```text
//! F2(a; b1, b2; c1, c2; x, y)
//!   = sum_{m,n>=0} (a)_{m+n} (b1)_m (b2)_n / ((c1)_m (c2)_n m! n!) * x^m y^n
//! ```
//!
//! Converges for |x| + |y| < 1.
//!
//! ### Appell F3
//!
//! ```text
//! F3(a1, a2; b1, b2; c; x, y)
//!   = sum_{m,n>=0} (a1)_m (a2)_n (b1)_m (b2)_n / ((c)_{m+n} m! n!) * x^m y^n
//! ```
//!
//! Converges for |x| < 1, |y| < 1.
//!
//! ### Appell F4
//!
//! ```text
//! F4(a, b; c1, c2; x, y)
//!   = sum_{m,n>=0} (a)_{m+n} (b)_{m+n} / ((c1)_m (c2)_n m! n!) * x^m y^n
//! ```
//!
//! Converges for sqrt(|x|) + sqrt(|y|) < 1.
//!
//! ### Horn H1
//!
//! ```text
//! H1(a, b, b'; c; x, y)
//!   = sum_{m,n>=0} (a)_{m-n} (b)_m (b')_n / ((c)_m m! n!) * x^m y^n
//! ```
//!
//! This is convergent for |x| < 4/27 (related to generalized Airy-type functions).
//!
//! ### Lauricella FD
//!
//! ```text
//! FD(a; b1,...,bn; c; x1,...,xn)
//!   = sum_{m1,...,mn>=0} (a)_{|m|} (b1)_{m1}...(bn)_{mn}
//!                        / ((c)_{|m|} m1!...mn!) * x1^m1...xn^mn
//! ```
//!
//! where |m| = m1 + ... + mn. Converges for all |xi| < 1.
//! This is the n-variable generalization of Appell F1.

use crate::error::{SpecialError, SpecialResult};

/// Maximum number of terms per dimension in double series
const MAX_SINGLE_DIM_TERMS: usize = 80;

/// Maximum total terms across all dimensions (to prevent excessive computation)
const MAX_TOTAL_TERMS: usize = 10_000;

/// Convergence tolerance
const CONV_TOL: f64 = 1e-14;

// ============================================================================
// Helper: Pochhammer rising factorial (a)_n
// ============================================================================

/// Compute the Pochhammer rising factorial (a)_n = a*(a+1)*...*(a+n-1).
///
/// Handles the special cases:
/// - (a)_0 = 1 for all a
/// - Returns 0.0 if a is a non-positive integer and n exceeds |a| (terminating)
#[inline]
fn pochhammer(a: f64, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut result = 1.0_f64;
    for k in 0..n {
        result *= a + k as f64;
        if result == 0.0 {
            return 0.0;
        }
    }
    result
}

// ============================================================================
// Appell F1
// ============================================================================

/// Compute the Appell hypergeometric function F1(a; b1, b2; c; x, y).
///
/// Uses the double power series:
/// ```text
/// F1 = sum_{m,n>=0} (a)_{m+n} (b1)_m (b2)_n / ((c)_{m+n} m! n!) * x^m y^n
/// ```
///
/// This is computed column-by-column (fixed m, varying n) to improve convergence tracking.
///
/// # Arguments
/// * `a`       - Parameter a
/// * `b1`      - Parameter b1
/// * `b2`      - Parameter b2
/// * `c`       - Parameter c (must not be 0, -1, -2, ...)
/// * `x`       - First variable (|x| < 1 for convergence)
/// * `y`       - Second variable (|y| < 1 for convergence)
/// * `n_terms` - Maximum number of terms per dimension (controls precision vs. speed)
///
/// # Returns
/// Value of F1(a; b1, b2; c; x, y)
///
/// # Errors
/// * `DomainError` if c is a non-positive integer
/// * `DomainError` if |x| >= 1 or |y| >= 1
///
/// # Examples
/// ```
/// use scirs2_special::appell::appell_f1;
/// // F1(a; b1, b2; c; 0, 0) = 1
/// let val = appell_f1(1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 50).expect("ok");
/// assert!((val - 1.0).abs() < 1e-14);
/// // When y=0, reduces to 2F1(a, b1; c; x)
/// let val2 = appell_f1(0.5, 1.0, 2.0, 3.0, 0.5, 0.0, 50).expect("ok");
/// // Should match 2F1(0.5, 1.0; 3.0; 0.5)
/// assert!(val2.is_finite() && val2 > 0.0);
/// ```
pub fn appell_f1(
    a: f64,
    b1: f64,
    b2: f64,
    c: f64,
    x: f64,
    y: f64,
    n_terms: usize,
) -> SpecialResult<f64> {
    // Validate c
    if c <= 0.0 && (c.fract().abs() < 1e-10) && c.round() as i64 <= 0 {
        return Err(SpecialError::DomainError(format!(
            "Appell F1: c = {c} must not be a non-positive integer"
        )));
    }

    // Convergence domain
    if x.abs() >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "Appell F1: |x| = {} >= 1 is outside the convergence region",
            x.abs()
        )));
    }
    if y.abs() >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "Appell F1: |y| = {} >= 1 is outside the convergence region",
            y.abs()
        )));
    }

    // Quick special cases
    if x.abs() < f64::EPSILON && y.abs() < f64::EPSILON {
        return Ok(1.0);
    }

    let max_terms = n_terms.min(MAX_SINGLE_DIM_TERMS);
    let mut result = 0.0_f64;
    let mut total = 0usize;

    // Outer sum over m (x variable)
    let mut x_m = 1.0_f64;        // x^m
    let mut poch_b1_m = 1.0_f64;  // (b1)_m
    let mut fact_m = 1.0_f64;     // m!

    for m in 0..max_terms {
        if m > 0 {
            x_m *= x;
            poch_b1_m *= b1 + (m - 1) as f64;
            fact_m *= m as f64;
        }

        if x_m.abs() < f64::EPSILON * 1e-10 && m > 0 {
            break;
        }

        // Inner sum over n (y variable)
        let mut y_n = 1.0_f64;
        let mut poch_b2_n = 1.0_f64;
        let mut fact_n = 1.0_f64;
        // (a)_{m+n} starts at n=0: (a)_m
        let mut poch_a_mn = pochhammer(a, m);
        // (c)_{m+n} starts at n=0: (c)_m
        let mut poch_c_mn = pochhammer(c, m);

        let mut col_sum = 0.0_f64;

        for n in 0..max_terms {
            if n > 0 {
                y_n *= y;
                poch_b2_n *= b2 + (n - 1) as f64;
                fact_n *= n as f64;
                poch_a_mn *= a + (m + n - 1) as f64;
                poch_c_mn *= c + (m + n - 1) as f64;
            }

            total += 1;
            if total > MAX_TOTAL_TERMS {
                break;
            }

            if poch_c_mn.abs() < 1e-300 {
                return Err(SpecialError::DomainError(
                    "Appell F1: (c)_{m+n} = 0; denominator vanishes".to_string(),
                ));
            }

            let term = poch_a_mn * poch_b1_m * poch_b2_n
                / (poch_c_mn * fact_m * fact_n)
                * x_m
                * y_n;

            col_sum += term;

            // Inner convergence check
            if n > 2 && term.abs() < CONV_TOL * col_sum.abs().max(1e-300) {
                break;
            }
        }

        result += col_sum;

        // Outer convergence check
        if m > 2 && col_sum.abs() < CONV_TOL * result.abs().max(1e-300) {
            break;
        }

        if total > MAX_TOTAL_TERMS {
            break;
        }
    }

    Ok(result)
}

// ============================================================================
// Appell F2
// ============================================================================

/// Compute the Appell hypergeometric function F2(a; b1, b2; c1, c2; x, y).
///
/// Uses the double power series:
/// ```text
/// F2 = sum_{m,n>=0} (a)_{m+n} (b1)_m (b2)_n / ((c1)_m (c2)_n m! n!) * x^m y^n
/// ```
///
/// Convergence region: |x| + |y| < 1.
///
/// # Arguments
/// * `a`       - Parameter a
/// * `b1`      - Parameter b1
/// * `b2`      - Parameter b2
/// * `c1`      - Parameter c1 (must not be 0, -1, -2, ...)
/// * `c2`      - Parameter c2 (must not be 0, -1, -2, ...)
/// * `x`       - First variable
/// * `y`       - Second variable
/// * `n_terms` - Maximum terms per dimension
///
/// # Returns
/// Value of F2(a; b1, b2; c1, c2; x, y)
///
/// # Errors
/// * `DomainError` if c1 or c2 is a non-positive integer
/// * `DomainError` if |x| + |y| >= 1
///
/// # Examples
/// ```
/// use scirs2_special::appell::appell_f2;
/// let val = appell_f2(1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 50).expect("ok");
/// assert!((val - 1.0).abs() < 1e-14);
/// let val2 = appell_f2(1.0, 1.0, 1.0, 2.0, 2.0, 0.2, 0.3, 50).expect("ok");
/// assert!(val2.is_finite() && val2 > 0.0);
/// ```
pub fn appell_f2(
    a: f64,
    b1: f64,
    b2: f64,
    c1: f64,
    c2: f64,
    x: f64,
    y: f64,
    n_terms: usize,
) -> SpecialResult<f64> {
    // Validate c1, c2
    for (ci, name) in &[(c1, "c1"), (c2, "c2")] {
        if *ci <= 0.0 && (ci.fract().abs() < 1e-10) && ci.round() as i64 <= 0 {
            return Err(SpecialError::DomainError(format!(
                "Appell F2: {name} = {ci} must not be a non-positive integer"
            )));
        }
    }

    // Convergence domain: |x| + |y| < 1
    if x.abs() + y.abs() >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "Appell F2: |x|+|y| = {} >= 1 is outside the convergence region",
            x.abs() + y.abs()
        )));
    }

    if x.abs() < f64::EPSILON && y.abs() < f64::EPSILON {
        return Ok(1.0);
    }

    let max_terms = n_terms.min(MAX_SINGLE_DIM_TERMS);
    let mut result = 0.0_f64;
    let mut total = 0usize;

    let mut x_m = 1.0_f64;
    let mut poch_b1_m = 1.0_f64;
    let mut poch_c1_m = 1.0_f64;
    let mut fact_m = 1.0_f64;

    for m in 0..max_terms {
        if m > 0 {
            x_m *= x;
            poch_b1_m *= b1 + (m - 1) as f64;
            poch_c1_m *= c1 + (m - 1) as f64;
            fact_m *= m as f64;
        }

        if poch_c1_m.abs() < 1e-300 && m > 0 {
            return Err(SpecialError::DomainError("Appell F2: (c1)_m = 0".to_string()));
        }

        let mut y_n = 1.0_f64;
        let mut poch_b2_n = 1.0_f64;
        let mut poch_c2_n = 1.0_f64;
        let mut fact_n = 1.0_f64;
        let mut poch_a_mn = pochhammer(a, m);

        let mut col_sum = 0.0_f64;

        for n in 0..max_terms {
            if n > 0 {
                y_n *= y;
                poch_b2_n *= b2 + (n - 1) as f64;
                poch_c2_n *= c2 + (n - 1) as f64;
                fact_n *= n as f64;
                poch_a_mn *= a + (m + n - 1) as f64;
            }

            total += 1;
            if total > MAX_TOTAL_TERMS {
                break;
            }

            if poch_c2_n.abs() < 1e-300 && n > 0 {
                return Err(SpecialError::DomainError("Appell F2: (c2)_n = 0".to_string()));
            }

            let denom = poch_c1_m * poch_c2_n * fact_m * fact_n;
            if denom.abs() < 1e-300 {
                continue;
            }

            let term = poch_a_mn * poch_b1_m * poch_b2_n / denom * x_m * y_n;
            col_sum += term;

            if n > 2 && term.abs() < CONV_TOL * col_sum.abs().max(1e-300) {
                break;
            }
        }

        result += col_sum;

        if m > 2 && col_sum.abs() < CONV_TOL * result.abs().max(1e-300) {
            break;
        }
        if total > MAX_TOTAL_TERMS {
            break;
        }
    }

    Ok(result)
}

// ============================================================================
// Appell F3
// ============================================================================

/// Compute the Appell hypergeometric function F3(a1, a2; b1, b2; c; x, y).
///
/// Defined by:
/// ```text
/// F3(a1,a2;b1,b2;c;x,y) = sum_{m,n>=0} (a1)_m (a2)_n (b1)_m (b2)_n
///                          / ((c)_{m+n} m! n!) * x^m y^n
/// ```
///
/// Converges for |x| < 1, |y| < 1.
///
/// # Arguments
/// * `a1`, `a2` - Parameters
/// * `b1`, `b2` - Parameters
/// * `c`        - Denominator parameter (must not be a non-positive integer)
/// * `x`        - First variable (|x| < 1)
/// * `y`        - Second variable (|y| < 1)
/// * `n_terms`  - Maximum terms per dimension
///
/// # Returns
/// Value of F3(a1,a2;b1,b2;c;x,y)
///
/// # Examples
/// ```
/// use scirs2_special::appell::appell_f3;
/// let val = appell_f3(1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 50).expect("ok");
/// assert!((val - 1.0).abs() < 1e-14);
/// ```
pub fn appell_f3(
    a1: f64,
    a2: f64,
    b1: f64,
    b2: f64,
    c: f64,
    x: f64,
    y: f64,
    n_terms: usize,
) -> SpecialResult<f64> {
    if c <= 0.0 && (c.fract().abs() < 1e-10) && c.round() as i64 <= 0 {
        return Err(SpecialError::DomainError(format!(
            "Appell F3: c = {c} must not be a non-positive integer"
        )));
    }
    if x.abs() >= 1.0 || y.abs() >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "Appell F3: |x|={}, |y|={} must be < 1",
            x.abs(),
            y.abs()
        )));
    }

    if x.abs() < f64::EPSILON && y.abs() < f64::EPSILON {
        return Ok(1.0);
    }

    let max_terms = n_terms.min(MAX_SINGLE_DIM_TERMS);
    let mut result = 0.0_f64;
    let mut total = 0usize;

    let mut x_m = 1.0_f64;
    let mut poch_a1_m = 1.0_f64;
    let mut poch_b1_m = 1.0_f64;
    let mut fact_m = 1.0_f64;

    for m in 0..max_terms {
        if m > 0 {
            x_m *= x;
            poch_a1_m *= a1 + (m - 1) as f64;
            poch_b1_m *= b1 + (m - 1) as f64;
            fact_m *= m as f64;
        }

        let mut y_n = 1.0_f64;
        let mut poch_a2_n = 1.0_f64;
        let mut poch_b2_n = 1.0_f64;
        let mut fact_n = 1.0_f64;
        let mut poch_c_mn = pochhammer(c, m);

        let mut col_sum = 0.0_f64;

        for n in 0..max_terms {
            if n > 0 {
                y_n *= y;
                poch_a2_n *= a2 + (n - 1) as f64;
                poch_b2_n *= b2 + (n - 1) as f64;
                fact_n *= n as f64;
                poch_c_mn *= c + (m + n - 1) as f64;
            }

            total += 1;
            if total > MAX_TOTAL_TERMS {
                break;
            }

            if poch_c_mn.abs() < 1e-300 {
                return Err(SpecialError::DomainError("Appell F3: (c)_{m+n} = 0".to_string()));
            }

            let denom = poch_c_mn * fact_m * fact_n;
            let term = poch_a1_m * poch_a2_n * poch_b1_m * poch_b2_n / denom * x_m * y_n;
            col_sum += term;

            if n > 2 && term.abs() < CONV_TOL * col_sum.abs().max(1e-300) {
                break;
            }
        }

        result += col_sum;

        if m > 2 && col_sum.abs() < CONV_TOL * result.abs().max(1e-300) {
            break;
        }
        if total > MAX_TOTAL_TERMS {
            break;
        }
    }

    Ok(result)
}

// ============================================================================
// Appell F4
// ============================================================================

/// Compute the Appell hypergeometric function F4(a, b; c1, c2; x, y).
///
/// Defined by:
/// ```text
/// F4(a,b;c1,c2;x,y) = sum_{m,n>=0} (a)_{m+n} (b)_{m+n} / ((c1)_m (c2)_n m! n!) * x^m y^n
/// ```
///
/// Convergence region: sqrt(|x|) + sqrt(|y|) < 1.
///
/// # Arguments
/// * `a`       - Parameter a
/// * `b`       - Parameter b
/// * `c1`      - First denominator parameter (must not be a non-positive integer)
/// * `c2`      - Second denominator parameter (must not be a non-positive integer)
/// * `x`       - First variable
/// * `y`       - Second variable
/// * `n_terms` - Maximum terms per dimension
///
/// # Returns
/// Value of F4(a,b;c1,c2;x,y)
///
/// # Examples
/// ```
/// use scirs2_special::appell::appell_f4;
/// let val = appell_f4(1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 50).expect("ok");
/// assert!((val - 1.0).abs() < 1e-14);
/// ```
pub fn appell_f4(
    a: f64,
    b: f64,
    c1: f64,
    c2: f64,
    x: f64,
    y: f64,
    n_terms: usize,
) -> SpecialResult<f64> {
    for (ci, name) in &[(c1, "c1"), (c2, "c2")] {
        if *ci <= 0.0 && (ci.fract().abs() < 1e-10) && ci.round() as i64 <= 0 {
            return Err(SpecialError::DomainError(format!(
                "Appell F4: {name} = {ci} must not be a non-positive integer"
            )));
        }
    }

    // Convergence domain: sqrt(|x|) + sqrt(|y|) < 1
    if x.abs().sqrt() + y.abs().sqrt() >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "Appell F4: sqrt(|x|)+sqrt(|y|) = {} >= 1 is outside convergence region",
            x.abs().sqrt() + y.abs().sqrt()
        )));
    }

    if x.abs() < f64::EPSILON && y.abs() < f64::EPSILON {
        return Ok(1.0);
    }

    let max_terms = n_terms.min(MAX_SINGLE_DIM_TERMS);
    let mut result = 0.0_f64;
    let mut total = 0usize;

    let mut x_m = 1.0_f64;
    let mut poch_c1_m = 1.0_f64;
    let mut fact_m = 1.0_f64;

    for m in 0..max_terms {
        if m > 0 {
            x_m *= x;
            poch_c1_m *= c1 + (m - 1) as f64;
            fact_m *= m as f64;
        }

        if poch_c1_m.abs() < 1e-300 && m > 0 {
            return Err(SpecialError::DomainError("Appell F4: (c1)_m = 0".to_string()));
        }

        let mut y_n = 1.0_f64;
        let mut poch_c2_n = 1.0_f64;
        let mut fact_n = 1.0_f64;
        // (a)_{m+n} and (b)_{m+n} start at n=0 as (a)_m, (b)_m
        let mut poch_a_mn = pochhammer(a, m);
        let mut poch_b_mn = pochhammer(b, m);

        let mut col_sum = 0.0_f64;

        for n in 0..max_terms {
            if n > 0 {
                y_n *= y;
                poch_c2_n *= c2 + (n - 1) as f64;
                fact_n *= n as f64;
                poch_a_mn *= a + (m + n - 1) as f64;
                poch_b_mn *= b + (m + n - 1) as f64;
            }

            total += 1;
            if total > MAX_TOTAL_TERMS {
                break;
            }

            if poch_c2_n.abs() < 1e-300 && n > 0 {
                return Err(SpecialError::DomainError("Appell F4: (c2)_n = 0".to_string()));
            }

            let denom = poch_c1_m * poch_c2_n * fact_m * fact_n;
            if denom.abs() < 1e-300 {
                continue;
            }

            let term = poch_a_mn * poch_b_mn / denom * x_m * y_n;
            col_sum += term;

            if n > 2 && term.abs() < CONV_TOL * col_sum.abs().max(1e-300) {
                break;
            }
        }

        result += col_sum;

        if m > 2 && col_sum.abs() < CONV_TOL * result.abs().max(1e-300) {
            break;
        }
        if total > MAX_TOTAL_TERMS {
            break;
        }
    }

    Ok(result)
}

// ============================================================================
// Horn H1 Function
// ============================================================================

/// Compute the Horn H1 hypergeometric function H1(a, b, b'; c; x, y).
///
/// The Horn H1 function is defined by the double series:
/// ```text
/// H1(a, b, b'; c; x, y) = sum_{m,n>=0} (a)_{m-n} (b)_m (b')_n / ((c)_m m! n!) * x^m y^n
/// ```
///
/// where (a)_{m-n} = Gamma(a+m-n)/Gamma(a) is the Pochhammer symbol
/// extended to negative integers via (a)_{-k} = (-1)^k / (1-a)_k.
///
/// The convergence region for H1 is related to |x| < 4/27 (derived from
/// the Horn convergence theory for 2-variable hypergeometric series).
/// In practice, convergence is more subtle; we evaluate the series directly
/// and check for convergence.
///
/// Note: (a)_{m-n} for m < n requires the extension to negative arguments:
/// ```text
/// (a)_{-k} = (-1)^k / (1-a)_k,  k >= 0
/// ```
///
/// # Arguments
/// * `a`       - Parameter a
/// * `b`       - Parameter b
/// * `b_prime` - Parameter b'
/// * `c`       - Denominator parameter (must not be 0, -1, -2, ...)
/// * `x`       - First variable
/// * `y`       - Second variable
/// * `n_terms` - Maximum number of terms per dimension
///
/// # Returns
/// Value of H1(a, b, b'; c; x, y)
///
/// # Errors
/// * `DomainError` if c is a non-positive integer
///
/// # Examples
/// ```
/// use scirs2_special::appell::horn_h1;
/// // H1(a, b, b'; c; 0, 0) = 1
/// let val = horn_h1(1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 30).expect("ok");
/// assert!((val - 1.0).abs() < 1e-14);
/// ```
pub fn horn_h1(
    a: f64,
    b: f64,
    b_prime: f64,
    c: f64,
    x: f64,
    y: f64,
    n_terms: usize,
) -> SpecialResult<f64> {
    if c <= 0.0 && (c.fract().abs() < 1e-10) && c.round() as i64 <= 0 {
        return Err(SpecialError::DomainError(format!(
            "Horn H1: c = {c} must not be a non-positive integer"
        )));
    }

    if x.abs() < f64::EPSILON && y.abs() < f64::EPSILON {
        return Ok(1.0);
    }

    let max_terms = n_terms.min(MAX_SINGLE_DIM_TERMS);
    let mut result = 0.0_f64;
    let mut total = 0usize;

    // Outer sum over m
    let mut x_m = 1.0_f64;
    let mut poch_b_m = 1.0_f64;
    let mut poch_c_m = 1.0_f64;
    let mut fact_m = 1.0_f64;

    for m in 0..max_terms {
        if m > 0 {
            x_m *= x;
            poch_b_m *= b + (m - 1) as f64;
            poch_c_m *= c + (m - 1) as f64;
            fact_m *= m as f64;
        }

        if poch_c_m.abs() < 1e-300 && m > 0 {
            return Err(SpecialError::DomainError("Horn H1: (c)_m = 0".to_string()));
        }

        // Inner sum over n
        let mut y_n = 1.0_f64;
        let mut poch_bprime_n = 1.0_f64;
        let mut fact_n = 1.0_f64;

        let mut col_sum = 0.0_f64;

        for n in 0..max_terms {
            if n > 0 {
                y_n *= y;
                poch_bprime_n *= b_prime + (n - 1) as f64;
                fact_n *= n as f64;
            }

            total += 1;
            if total > MAX_TOTAL_TERMS {
                break;
            }

            // Compute (a)_{m-n}
            let shift = m as i64 - n as i64;
            let poch_a_mminusn = pochhammer_shifted(a, shift);

            let denom = poch_c_m * fact_m * fact_n;
            if denom.abs() < 1e-300 {
                continue;
            }

            let term = poch_a_mminusn * poch_b_m * poch_bprime_n / denom * x_m * y_n;
            col_sum += term;

            if n > 2 && term.abs() < CONV_TOL * col_sum.abs().max(1e-300) {
                break;
            }
        }

        result += col_sum;

        if m > 2 && col_sum.abs() < CONV_TOL * result.abs().max(1e-300) {
            break;
        }
        if total > MAX_TOTAL_TERMS {
            break;
        }
    }

    Ok(result)
}

// ============================================================================
// Lauricella FD Function
// ============================================================================

/// Compute the Lauricella FD hypergeometric function.
///
/// The Lauricella FD function is the n-variable generalization of Appell F1:
///
/// ```text
/// FD(a; b1,...,bn; c; x1,...,xn)
///   = sum_{m1,...,mn >= 0}
///       (a)_{|m|} (b1)_{m1} ... (bn)_{mn}
///       / ((c)_{|m|} m1! ... mn!)
///       * x1^{m1} ... xn^{mn}
/// ```
///
/// where |m| = m1 + ... + mn.
///
/// Converges for all |xi| < 1.
///
/// For n=1 this reduces to 2F1(a, b1; c; x1).
/// For n=2 this reduces to Appell F1(a; b1, b2; c; x1, x2).
///
/// # Arguments
/// * `a`       - Parameter a (scalar)
/// * `b_vec`   - Parameters [b1, ..., bn]
/// * `c`       - Denominator parameter (must not be a non-positive integer)
/// * `x_vec`   - Arguments [x1, ..., xn] (all |xi| < 1 required)
/// * `n_terms` - Maximum number of terms per dimension
///
/// # Returns
/// Value of FD(a; b1,...,bn; c; x1,...,xn)
///
/// # Errors
/// * `DomainError` if c is a non-positive integer
/// * `DomainError` if any |xi| >= 1
/// * `ValueError` if b_vec and x_vec have different lengths
///
/// # Examples
/// ```
/// use scirs2_special::appell::lauricella_fd;
/// // FD(a; b1; c; x) = 2F1(a, b1; c; x) (n=1 case)
/// let val = lauricella_fd(1.0, &[2.0], 3.0, &[0.0], 50).expect("ok");
/// assert!((val - 1.0).abs() < 1e-14);
/// // FD = 1 when all x=0
/// let val2 = lauricella_fd(1.0, &[2.0, 3.0, 4.0], 5.0, &[0.0, 0.0, 0.0], 30).expect("ok");
/// assert!((val2 - 1.0).abs() < 1e-14);
/// ```
pub fn lauricella_fd(
    a: f64,
    b_vec: &[f64],
    c: f64,
    x_vec: &[f64],
    n_terms: usize,
) -> SpecialResult<f64> {
    if b_vec.len() != x_vec.len() {
        return Err(SpecialError::ValueError(format!(
            "lauricella_fd: b_vec.len()={} != x_vec.len()={}",
            b_vec.len(),
            x_vec.len()
        )));
    }

    let n_vars = b_vec.len();

    if c <= 0.0 && (c.fract().abs() < 1e-10) && c.round() as i64 <= 0 {
        return Err(SpecialError::DomainError(format!(
            "Lauricella FD: c = {c} must not be a non-positive integer"
        )));
    }

    // Validate all |xi| < 1
    for (i, &xi) in x_vec.iter().enumerate() {
        if xi.abs() >= 1.0 {
            return Err(SpecialError::DomainError(format!(
                "Lauricella FD: |x[{i}]| = {} >= 1 (outside convergence region)",
                xi.abs()
            )));
        }
    }

    // Special case: n_vars = 0
    if n_vars == 0 {
        return Ok(1.0);
    }

    // Special case: all x = 0
    if x_vec.iter().all(|&xi| xi.abs() < f64::EPSILON) {
        return Ok(1.0);
    }

    // General n-variable case using recursive summation.
    // We enumerate all multi-index (m1,...,mn) with sum <= max_total
    // via a recursive approach.
    let max_per_dim = n_terms.min(MAX_SINGLE_DIM_TERMS);
    // Total order cutoff: to keep computation tractable
    let max_order = (max_per_dim as f64).powf(1.0 / n_vars as f64).ceil() as usize + 2;

    lauricella_fd_recursive(a, b_vec, c, x_vec, n_vars, max_order)
}

/// Recursive computation of Lauricella FD by enumerating multi-indices.
///
/// This uses a recursive approach where we fix m_k and sum over the remaining variables.
fn lauricella_fd_recursive(
    a: f64,
    b_vec: &[f64],
    c: f64,
    x_vec: &[f64],
    n_vars: usize,
    max_order: usize,
) -> SpecialResult<f64> {
    // We use a flat multi-index loop for n_vars <= 4, and recursive otherwise.
    // For larger n_vars, we use a priority-queue approach (sum only significant terms).

    match n_vars {
        1 => {
            // Reduce to 2F1(a, b; c; x)
            lauricella_fd_1d(a, b_vec[0], c, x_vec[0], max_order)
        }
        2 => {
            // Appell F1
            appell_f1(a, b_vec[0], b_vec[1], c, x_vec[0], x_vec[1], max_order)
        }
        _ => {
            // General n-variable case
            lauricella_fd_general(a, b_vec, c, x_vec, n_vars, max_order)
        }
    }
}

/// One-variable Lauricella FD (= 2F1).
fn lauricella_fd_1d(a: f64, b: f64, c: f64, x: f64, max_order: usize) -> SpecialResult<f64> {
    // 2F1(a, b; c; x)
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    let mut x_k = 1.0_f64;

    for k in 1..=max_order {
        let k_f = (k - 1) as f64;
        x_k *= x;
        let numer = (a + k_f) * (b + k_f);
        let denom = (c + k_f) * (k as f64);
        if denom.abs() < 1e-300 {
            break;
        }
        term *= numer / denom;
        sum += term * x_k / x_k * x; // avoid redundant multiplication

        // Recompute directly
        let _ = x_k;
        // Use iterative approach
        break; // fallthrough to clean loop below
    }

    // Clean implementation
    let mut term2 = 1.0_f64;
    let mut sum2 = 1.0_f64;
    for k in 1..=max_order {
        let kf = (k - 1) as f64;
        let numer = (a + kf) * (b + kf) * x;
        let denom = (c + kf) * (k as f64);
        if denom.abs() < 1e-300 {
            break;
        }
        term2 *= numer / denom;
        sum2 += term2;
        if term2.abs() < CONV_TOL * sum2.abs().max(1e-300) {
            break;
        }
    }

    let _ = (term, sum); // suppress unused
    Ok(sum2)
}

/// General n-variable Lauricella FD using recursive enumeration.
fn lauricella_fd_general(
    a: f64,
    b_vec: &[f64],
    c: f64,
    x_vec: &[f64],
    n_vars: usize,
    max_order: usize,
) -> SpecialResult<f64> {
    // Use a stack-based multi-index enumeration.
    // Each entry: (m_indices as Vec<usize>, current (a)_{|m|}, (c)_{|m|}, product of b_i^{m_i}/m_i!, x_i^{m_i})

    // For efficiency, we use depth-first recursion on the variable index.
    // fix_sum(k, depth) = contribution from vars[depth..n_vars] given vars[0..depth] = m[0..depth]

    let result = lauricella_fd_inner(a, b_vec, c, x_vec, 0, n_vars, 0, 1.0, 1.0, 1.0, max_order)?;
    Ok(result)
}

/// Inner recursive helper for Lauricella FD.
///
/// Fixes variables 0..depth and sums over variables depth..n_vars.
///
/// # Parameters
/// * `depth`        - Current variable index
/// * `n_vars`       - Total number of variables
/// * `m_sum`        - m_0 + ... + m_{depth-1} (accumulated index sum)
/// * `poch_a`       - (a)_{m_sum} accumulated
/// * `poch_c`       - (c)_{m_sum} accumulated
/// * `prefix_coeff` - product of (b_i)_{m_i} * x_i^{m_i} / m_i! for i < depth
/// * `max_order`    - maximum per-variable order
fn lauricella_fd_inner(
    a: f64,
    b_vec: &[f64],
    c: f64,
    x_vec: &[f64],
    depth: usize,
    n_vars: usize,
    m_sum: usize,
    poch_a: f64,
    poch_c: f64,
    prefix_coeff: f64,
    max_order: usize,
) -> SpecialResult<f64> {
    if depth == n_vars {
        // We've fixed all variable indices. Contribution = (a)_{|m|}/(c)_{|m|} * prefix_coeff
        if poch_c.abs() < 1e-300 {
            return Ok(0.0);
        }
        return Ok(poch_a / poch_c * prefix_coeff);
    }

    let b_k = b_vec[depth];
    let x_k = x_vec[depth];

    let mut sum = 0.0_f64;
    let mut x_k_mk = 1.0_f64; // x_k^{m_k}
    let mut poch_b_mk = 1.0_f64; // (b_k)_{m_k}
    let mut fact_mk = 1.0_f64; // m_k!
    let mut poch_a_cur = poch_a; // (a)_{m_sum + m_k}
    let mut poch_c_cur = poch_c; // (c)_{m_sum + m_k}

    for m_k in 0..=max_order {
        if m_k > 0 {
            x_k_mk *= x_k;
            poch_b_mk *= b_k + (m_k - 1) as f64;
            fact_mk *= m_k as f64;
            poch_a_cur *= a + (m_sum + m_k - 1) as f64;
            poch_c_cur *= c + (m_sum + m_k - 1) as f64;
        }

        let coeff_k = poch_b_mk * x_k_mk / fact_mk;
        let new_prefix = prefix_coeff * coeff_k;

        if new_prefix.abs() < 1e-300 * prefix_coeff.abs().max(1e-300) && m_k > 0 {
            break;
        }

        let sub_result = lauricella_fd_inner(
            a,
            b_vec,
            c,
            x_vec,
            depth + 1,
            n_vars,
            m_sum + m_k,
            poch_a_cur,
            poch_c_cur,
            new_prefix,
            max_order,
        )?;

        sum += sub_result;

        if m_k > 2 && sub_result.abs() < CONV_TOL * sum.abs().max(1e-300) {
            break;
        }
    }

    Ok(sum)
}

// ============================================================================
// Lauricella FC function (bonus)
// ============================================================================

/// Compute the Lauricella FC hypergeometric function (2-variable case).
///
/// The Lauricella FC function (n=2) equals Appell F4:
/// ```text
/// FC(a, b; c1, c2; x, y) = F4(a, b; c1, c2; x, y)
///   = sum_{m,n>=0} (a)_{m+n} (b)_{m+n} / ((c1)_m (c2)_n m! n!) * x^m y^n
/// ```
///
/// Convergence region: sqrt(|x|) + sqrt(|y|) < 1.
///
/// # Arguments
/// * `a`, `b`   - Parameters
/// * `c_vec`    - Denominator parameters [c1, c2]
/// * `x_vec`    - Arguments [x1, x2]
/// * `n_terms`  - Maximum terms per dimension
///
/// # Returns
/// Value of FC(a,b;c1,c2;x1,x2)
///
/// # Examples
/// ```
/// use scirs2_special::appell::lauricella_fc;
/// let val = lauricella_fc(1.0, 1.0, &[2.0, 3.0], &[0.0, 0.0], 30).expect("ok");
/// assert!((val - 1.0).abs() < 1e-14);
/// ```
pub fn lauricella_fc(
    a: f64,
    b: f64,
    c_vec: &[f64],
    x_vec: &[f64],
    n_terms: usize,
) -> SpecialResult<f64> {
    if c_vec.len() != 2 || x_vec.len() != 2 {
        return Err(SpecialError::ValueError(
            "lauricella_fc: currently only 2-variable case supported".to_string(),
        ));
    }
    appell_f4(a, b, c_vec[0], c_vec[1], x_vec[0], x_vec[1], n_terms)
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Pochhammer symbol extended to negative integer shifts: (a)_k for integer k.
///
/// For k >= 0: (a)_k = a*(a+1)*...*(a+k-1)
/// For k < 0:  (a)_k = 1 / ((a-1)*(a-2)*...*(a+k)) = (-1)^|k| / (1-a)_{|k|}
#[inline]
fn pochhammer_shifted(a: f64, k: i64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if k > 0 {
        let mut result = 1.0_f64;
        for j in 0..k {
            result *= a + j as f64;
        }
        result
    } else {
        // (a)_{-n} = (-1)^n / (1-a)_n
        let n = (-k) as u64;
        let mut denom = 1.0_f64;
        let sign = if n % 2 == 1 { -1.0_f64 } else { 1.0 };
        for j in 0..n {
            denom *= (1.0 - a) + j as f64;
        }
        if denom.abs() < 1e-300 {
            return 0.0;
        }
        sign / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    // ---- Appell F1 ----

    #[test]
    fn test_appell_f1_at_origin() {
        let val = appell_f1(1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 50).expect("ok");
        assert!((val - 1.0).abs() < 1e-14, "F1(0,0)=1: got {val}");
    }

    #[test]
    fn test_appell_f1_reduces_to_2f1_y_zero() {
        // F1(a; b1, b2; c; x, 0) = 2F1(a, b1; c; x)
        // 2F1(1, 1; 2; 0.5) = -2*ln(0.5) = 2*ln(2) ≈ 1.3862943...
        let val = appell_f1(1.0, 1.0, 999.0, 2.0, 0.5, 0.0, 100).expect("ok");
        let expected = 2.0 * 2.0_f64.ln(); // 2F1(1,1;2;0.5) = -ln(1-x)/x * ... actually = sum 0.5^n/(n+1) * n+1/(1) ... let me compute
        // 2F1(1,1;2;x) = -log(1-x)/x, at x=0.5: -log(0.5)/0.5 = log(2)/0.5 = 2*log(2)
        assert!(
            (val - expected).abs() < TOL,
            "F1 ~ 2F1 when y=0: got {val}, expected {expected}"
        );
    }

    #[test]
    fn test_appell_f1_reduces_to_2f1_x_zero() {
        // F1(a; b1, b2; c; 0, y) = 2F1(a, b2; c; y)
        // 2F1(1, 1; 2; 0.3) = -log(1-0.3)/0.3 = log(10/7)/0.3
        let val = appell_f1(1.0, 999.0, 1.0, 2.0, 0.0, 0.3, 100).expect("ok");
        let expected = -(1.0_f64 - 0.3).ln() / 0.3; // 2F1(1,1;2;0.3)
        assert!(
            (val - expected).abs() < TOL,
            "F1 ~ 2F1 when x=0: got {val}, expected {expected}"
        );
    }

    #[test]
    fn test_appell_f1_domain_error_c() {
        assert!(appell_f1(1.0, 2.0, 3.0, 0.0, 0.1, 0.1, 50).is_err());
        assert!(appell_f1(1.0, 2.0, 3.0, -1.0, 0.1, 0.1, 50).is_err());
    }

    #[test]
    fn test_appell_f1_domain_error_xy() {
        assert!(appell_f1(1.0, 2.0, 3.0, 4.0, 1.0, 0.1, 50).is_err());
        assert!(appell_f1(1.0, 2.0, 3.0, 4.0, 0.1, -1.5, 50).is_err());
    }

    #[test]
    fn test_appell_f1_symmetry() {
        // F1 is NOT symmetric in b1,b2 under x<->y swap in general,
        // but F1(a;b1,b2;c;x,y) = F1(a;b2,b1;c;y,x)
        let v1 = appell_f1(2.0, 1.0, 3.0, 5.0, 0.3, 0.2, 60).expect("ok");
        let v2 = appell_f1(2.0, 3.0, 1.0, 5.0, 0.2, 0.3, 60).expect("ok");
        assert!(
            (v1 - v2).abs() < 1e-8,
            "F1 symmetry: F1(a;b1,b2;c;x,y)=F1(a;b2,b1;c;y,x): {v1} vs {v2}"
        );
    }

    // ---- Appell F2 ----

    #[test]
    fn test_appell_f2_at_origin() {
        let val = appell_f2(1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 50).expect("ok");
        assert!((val - 1.0).abs() < 1e-14, "F2(0,0)=1: {val}");
    }

    #[test]
    fn test_appell_f2_domain_error() {
        assert!(appell_f2(1.0, 1.0, 1.0, 0.0, 2.0, 0.1, 0.1, 50).is_err());
        assert!(appell_f2(1.0, 1.0, 1.0, 2.0, 0.0, 0.1, 0.1, 50).is_err());
        // |x|+|y| >= 1
        assert!(appell_f2(1.0, 1.0, 1.0, 2.0, 2.0, 0.6, 0.5, 50).is_err());
    }

    #[test]
    fn test_appell_f2_reduces_to_2f1() {
        // F2(a; b1, b2; c1, c2; x, 0) = 2F1(a, b1; c1; x)
        let val = appell_f2(1.0, 1.0, 999.0, 2.0, 3.0, 0.3, 0.0, 100).expect("ok");
        let expected = -(1.0_f64 - 0.3).ln() / 0.3;
        assert!(
            (val - expected).abs() < TOL,
            "F2 ~ 2F1 when y=0: got {val}, expected {expected}"
        );
    }

    // ---- Appell F3 ----

    #[test]
    fn test_appell_f3_at_origin() {
        let val = appell_f3(1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 50).expect("ok");
        assert!((val - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_appell_f3_domain_error() {
        assert!(appell_f3(1.0, 1.0, 1.0, 1.0, -1.0, 0.1, 0.1, 50).is_err());
        assert!(appell_f3(1.0, 1.0, 1.0, 1.0, 2.0, 1.5, 0.1, 50).is_err());
    }

    // ---- Appell F4 ----

    #[test]
    fn test_appell_f4_at_origin() {
        let val = appell_f4(1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 50).expect("ok");
        assert!((val - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_appell_f4_domain_error() {
        // sqrt(|x|) + sqrt(|y|) >= 1
        assert!(appell_f4(1.0, 1.0, 2.0, 2.0, 0.5, 0.5, 50).is_err());
        assert!(appell_f4(1.0, 1.0, 0.0, 2.0, 0.1, 0.0, 50).is_err());
    }

    // ---- Horn H1 ----

    #[test]
    fn test_horn_h1_at_origin() {
        let val = horn_h1(1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 30).expect("ok");
        assert!((val - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_horn_h1_domain_error_c() {
        assert!(horn_h1(1.0, 2.0, 3.0, 0.0, 0.1, 0.1, 30).is_err());
    }

    #[test]
    fn test_horn_h1_small_values() {
        // For small x, y, should converge quickly
        let val = horn_h1(0.5, 1.0, 1.0, 2.0, 0.01, 0.01, 30).expect("ok");
        // Leading correction: (a)_{1} (b)_1 (b')_0 / (c)_1 * x/(1*1) = a*b/c * x = 0.5*1/2*0.01 = 0.0025
        //                   + (a)_{0} (b)_0 (b')_1 / (c)_0 * y/(1*1) = a_{-1}... no, for (m=0,n=1): shift=-1
        // First nonzero correction comes from m=1, n=0: term = (a)_1 (b)_1 / (c)_1 / 1! / 0! * x
        //   = 0.5 * 1 / 2 * 0.01 = 0.0025
        // So val ~ 1 + 0.0025 + ...
        assert!(
            (val - 1.0).abs() < 0.1,
            "H1 near origin should be close to 1: {val}"
        );
    }

    // ---- Lauricella FD ----

    #[test]
    fn test_lauricella_fd_at_origin() {
        let val = lauricella_fd(1.0, &[2.0, 3.0], 4.0, &[0.0, 0.0], 30).expect("ok");
        assert!((val - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_lauricella_fd_n1_is_2f1() {
        // FD(a; b; c; x) = 2F1(a, b; c; x) for n=1
        // 2F1(1, 1; 2; 0.4) = -log(1-0.4)/0.4 = log(5/3)/0.4
        let val = lauricella_fd(1.0, &[1.0], 2.0, &[0.4], 100).expect("ok");
        let expected = -(1.0_f64 - 0.4_f64).ln() / 0.4;
        assert!(
            (val - expected).abs() < TOL,
            "FD n=1 should equal 2F1: got {val}, expected {expected}"
        );
    }

    #[test]
    fn test_lauricella_fd_n2_is_appell_f1() {
        // FD(a; b1, b2; c; x, y) = F1(a; b1, b2; c; x, y)
        let v_fd = lauricella_fd(2.0, &[1.0, 1.0], 3.0, &[0.2, 0.3], 50).expect("ok");
        let v_f1 = appell_f1(2.0, 1.0, 1.0, 3.0, 0.2, 0.3, 50).expect("ok");
        assert!(
            (v_fd - v_f1).abs() < 1e-8,
            "FD n=2 should equal Appell F1: {v_fd} vs {v_f1}"
        );
    }

    #[test]
    fn test_lauricella_fd_n3() {
        // Should compute without error and return a finite value
        let val = lauricella_fd(1.0, &[1.0, 1.0, 1.0], 2.0, &[0.2, 0.1, 0.15], 20).expect("ok");
        assert!(val.is_finite(), "FD n=3 should be finite: {val}");
        assert!(val > 1.0, "FD n=3 with positive args should exceed 1: {val}");
    }

    #[test]
    fn test_lauricella_fd_domain_error() {
        assert!(lauricella_fd(1.0, &[2.0], 0.0, &[0.1], 30).is_err()); // c=0
        assert!(lauricella_fd(1.0, &[2.0], 3.0, &[1.5], 30).is_err()); // |x|>1
        assert!(lauricella_fd(1.0, &[2.0, 3.0], 4.0, &[0.1], 30).is_err()); // length mismatch
    }

    // ---- Lauricella FC ----

    #[test]
    fn test_lauricella_fc_at_origin() {
        let val = lauricella_fc(1.0, 2.0, &[3.0, 4.0], &[0.0, 0.0], 50).expect("ok");
        assert!((val - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_lauricella_fc_wrong_size() {
        assert!(lauricella_fc(1.0, 2.0, &[3.0], &[0.0, 0.0], 50).is_err());
        assert!(lauricella_fc(1.0, 2.0, &[3.0, 4.0, 5.0], &[0.0, 0.0], 50).is_err());
    }

    // ---- Pochhammer shifted ----

    #[test]
    fn test_pochhammer_shifted_zero() {
        assert_eq!(pochhammer_shifted(5.0, 0), 1.0);
        assert_eq!(pochhammer_shifted(-3.0, 0), 1.0);
    }

    #[test]
    fn test_pochhammer_shifted_positive() {
        // (2)_3 = 2*3*4 = 24
        assert!((pochhammer_shifted(2.0, 3) - 24.0).abs() < 1e-14);
    }

    #[test]
    fn test_pochhammer_shifted_negative() {
        // (a)_{-1} = 1 / (a-1) * (-1)^1 = -1/(a-1)
        let a = 3.0;
        let val = pochhammer_shifted(a, -1);
        let expected = -1.0 / (1.0 - a); // = -1/(1-3) = -1/(-2) = 0.5
        assert!(
            (val - expected).abs() < 1e-14,
            "(3)_{{-1}} should be {expected}, got {val}"
        );
    }
}
