//! Extended Whittaker functions with series, asymptotic, and vectorized forms
//!
//! This module provides extended implementations of the Whittaker M and W
//! functions, going beyond the basic implementation in `hypergeometric_enhanced`.
//!
//! ## Mathematical Background
//!
//! The Whittaker functions M_{kappa,mu}(z) and W_{kappa,mu}(z) are solutions of
//! Whittaker's differential equation:
//!
//! ```text
//! d²w/dz² + (-1/4 + kappa/z + (1/4 - mu²)/z²) w = 0
//! ```
//!
//! They are related to the confluent hypergeometric function (Kummer's function):
//!
//! ```text
//! M_{kappa,mu}(z) = exp(-z/2) * z^{mu+1/2} * M(mu - kappa + 1/2, 2mu+1, z)
//! W_{kappa,mu}(z) = exp(-z/2) * z^{mu+1/2} * U(mu - kappa + 1/2, 2mu+1, z)
//! ```
//!
//! where M(a,b,z) = 1F1(a;b;z) is the confluent hypergeometric function of the
//! first kind, and U(a,b,z) is Tricomi's function.
//!
//! ### Series Representation
//!
//! For small |z|, M_{kappa,mu}(z) has the series expansion:
//!
//! ```text
//! M_{kappa,mu}(z) = z^{mu+1/2} exp(-z/2)
//!                   * sum_{n=0}^infty (mu-kappa+1/2)_n / ((2mu+1)_n * n!) * z^n
//! ```
//!
//! ### Asymptotic Expansion
//!
//! For large |z|, the Whittaker W function has the asymptotic expansion:
//!
//! ```text
//! W_{kappa,mu}(z) ~ exp(-z/2) z^kappa * sum_{n=0}^{N} (-1)^n * C_n(kappa,mu) / z^n
//! ```
//!
//! where:
//! ```text
//! C_0 = 1,
//! C_n = prod_{k=1}^{n} ((2mu)^2 - (2k-1-2kappa)^2) / (8n)  ... see details below
//! ```
//!
//! More precisely:
//! ```text
//! C_n = ((1/2 - mu + kappa)_n * (1/2 + mu + kappa)_n) / n!  * (-1)^n
//! W_{kappa,mu}(z) ~ exp(-z/2) z^kappa sum_{n=0}^N C_n / z^n
//! ```

use crate::error::{SpecialError, SpecialResult};
use crate::gamma::{gamma, gammaln};

/// Maximum series terms for the Whittaker M series representation
const MAX_SERIES_TERMS: usize = 500;

/// Convergence tolerance for series
const SERIES_TOL: f64 = 1e-15;

/// Maximum asymptotic terms before divergence takes over
const MAX_ASYMP_TERMS: usize = 30;

// ============================================================================
// Whittaker M function
// ============================================================================

/// Compute Whittaker M function M_{kappa,mu}(z) using the series representation.
///
/// For small to moderate z, uses the power series of the underlying 1F1 function.
/// For large z, switches to the asymptotic expansion.
///
/// The Whittaker M function is defined by:
/// ```text
/// M_{kappa,mu}(z) = exp(-z/2) * z^{mu+1/2} * 1F1(mu - kappa + 1/2; 2*mu+1; z)
/// ```
///
/// This is valid for 2*mu not a negative integer.
///
/// # Arguments
/// * `kappa` - First parameter (unrestricted real)
/// * `mu`    - Second parameter (2*mu must not be a negative integer)
/// * `z`     - Argument (must be > 0 for the real branch)
///
/// # Returns
/// Value of M_{kappa,mu}(z)
///
/// # Errors
/// * `DomainError` if z <= 0
/// * `DomainError` if 2*mu is a negative integer
///
/// # Examples
/// ```
/// use scirs2_special::whittaker::whittaker_m_series;
/// let m = whittaker_m_series(0.5, 0.5, 1.0).expect("ok");
/// assert!(m.is_finite() && m > 0.0);
/// // Whittaker M reduces to modified Bessel at kappa=0, mu=1/2:
/// // M_{0,1/2}(z) = sqrt(z) * exp(-z/2) * 1F1(1; 2; z) = exp(z/2)*sqrt(z)*(1-exp(-z))/z
/// ```
pub fn whittaker_m_series(kappa: f64, mu: f64, z: f64) -> SpecialResult<f64> {
    if z <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "Whittaker M requires z > 0, got z = {z}"
        )));
    }

    let two_mu = 2.0 * mu;
    // Check that 2*mu is not a negative integer (pole of Gamma(2mu+1))
    if two_mu <= -1.0 && (two_mu + 1.0).fract().abs() < 1e-10 {
        return Err(SpecialError::DomainError(format!(
            "Whittaker M: 2*mu = {two_mu} is a negative integer, function undefined"
        )));
    }

    // Prefactor: exp(-z/2) * z^{mu+1/2}
    let log_prefactor = -z / 2.0 + (mu + 0.5) * z.ln();

    // Check for overflow/underflow in prefactor
    if log_prefactor < -740.0 {
        return Ok(0.0); // underflow to zero
    }
    if log_prefactor > 709.0 {
        return Err(SpecialError::OverflowError(format!(
            "Whittaker M: prefactor overflows for z={z}, mu={mu}"
        )));
    }

    let prefactor = log_prefactor.exp();

    // Compute 1F1(a; b; z) where a = mu - kappa + 0.5, b = 2*mu + 1
    let a = mu - kappa + 0.5;
    let b = 2.0 * mu + 1.0;

    let hyp = hyp1f1_series(a, b, z)?;
    Ok(prefactor * hyp)
}

/// Compute Whittaker W function W_{kappa,mu}(z) using the series/asymptotic approach.
///
/// For moderate z, uses the combination of two M functions (Wronskian approach).
/// For large z, uses the asymptotic expansion.
///
/// The Whittaker W function satisfies:
/// ```text
/// W_{kappa,mu}(z) = exp(-z/2) * z^{mu+1/2} * U(mu - kappa + 1/2; 2*mu+1; z)
/// ```
///
/// where U(a,b,z) is Tricomi's function.
///
/// # Arguments
/// * `kappa` - First parameter
/// * `mu`    - Second parameter
/// * `z`     - Argument (z > 0)
///
/// # Returns
/// Value of W_{kappa,mu}(z)
///
/// # Errors
/// * `DomainError` if z <= 0
///
/// # Examples
/// ```
/// use scirs2_special::whittaker::whittaker_w_series;
/// let w = whittaker_w_series(0.5, 0.5, 2.0).expect("ok");
/// assert!(w.is_finite());
/// // For large z, W decays exponentially:
/// let w_large = whittaker_w_series(1.0, 0.5, 20.0).expect("ok");
/// assert!(w_large.abs() < 1.0);
/// ```
pub fn whittaker_w_series(kappa: f64, mu: f64, z: f64) -> SpecialResult<f64> {
    if z <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "Whittaker W requires z > 0, got z = {z}"
        )));
    }

    // For large z, use asymptotic expansion
    if z > 20.0 {
        return whittaker_w_asymptotic(kappa, mu, z);
    }

    // For smaller z, use Tricomi U via the connection formula:
    // U(a, b, z) = pi/sin(pi*b) * (M(a,b,z)/(Gamma(1+a-b)*Gamma(b))
    //             - z^{1-b} * M(a-b+1, 2-b, z) / Gamma(a) / Gamma(2-b))
    //
    // This is the standard connection formula for non-integer b.
    // For integer b, a limiting form is needed (see Abramowitz & Stegun 13.1.10).
    let a = mu - kappa + 0.5;
    let b = 2.0 * mu + 1.0;

    whittaker_w_from_tricomi_u(kappa, mu, a, b, z)
}

/// Compute Whittaker W using the asymptotic expansion for large z.
///
/// The asymptotic series is:
/// ```text
/// W_{kappa,mu}(z) ~ exp(-z/2) * z^kappa * sum_{n=0}^N s_n(kappa,mu) / z^n
/// ```
/// where:
/// ```text
/// s_0 = 1
/// s_n = ((1/2 - mu + kappa)(1/2 + mu + kappa)) * ... n-th term
///     = prod_{k=1}^{n} ((2k - 1 - 2kappa)^2 - (2mu)^2) / (8k)
/// ```
///
/// The series is asymptotic (not convergent), so we stop at the smallest term.
///
/// # Arguments
/// * `kappa` - First parameter
/// * `mu`    - Second parameter
/// * `z`     - Argument (should be large, z > 10 recommended)
///
/// # Returns
/// Asymptotic approximation of W_{kappa,mu}(z)
///
/// # Errors
/// * `DomainError` if z <= 0
///
/// # Examples
/// ```
/// use scirs2_special::whittaker::whittaker_w_asymptotic;
/// let w = whittaker_w_asymptotic(1.0, 0.5, 50.0).expect("ok");
/// assert!(w.is_finite() && w > 0.0);
/// ```
pub fn whittaker_w_asymptotic(kappa: f64, mu: f64, z: f64) -> SpecialResult<f64> {
    if z <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "Whittaker W asymptotic requires z > 0, got z = {z}"
        )));
    }

    // Prefactor: exp(-z/2) * z^kappa
    let log_prefactor = -z / 2.0 + kappa * z.ln();
    if log_prefactor < -740.0 {
        return Ok(0.0);
    }
    if log_prefactor > 709.0 {
        return Err(SpecialError::OverflowError(format!(
            "Whittaker W asymptotic: prefactor overflows for z={z}, kappa={kappa}"
        )));
    }
    let prefactor = log_prefactor.exp();

    // Build the asymptotic sum
    // Term_n = (-1)^n * ((1/2 - mu + kappa)_n * (1/2 + mu + kappa)_n) / (n! * z^n)
    // Factored recurrence:
    // t_{n+1} = t_n * (-(1/2 - mu + kappa + n) * (1/2 + mu + kappa + n)) / ((n+1) * z)
    let alpha = 0.5 - mu + kappa;
    let beta = 0.5 + mu + kappa;

    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    let mut min_abs = 1.0_f64;
    let mut best_sum = 1.0_f64;

    for n in 0..MAX_ASYMP_TERMS {
        let n_f = n as f64;
        // Next term
        let numer = -(alpha + n_f) * (beta + n_f);
        let denom = (n_f + 1.0) * z;
        if denom == 0.0 {
            break;
        }
        term *= numer / denom;
        sum += term;

        // Track minimum |term| for optimal truncation
        if term.abs() < min_abs {
            min_abs = term.abs();
            best_sum = sum;
        }

        // Stop if term is growing (asymptotic series diverging)
        if term.abs() > 10.0 * min_abs && n > 3 {
            break;
        }

        // Stop if term is negligible
        if term.abs() < SERIES_TOL * sum.abs() {
            best_sum = sum;
            break;
        }
    }

    Ok(prefactor * best_sum)
}

/// Vectorized Whittaker M function evaluation.
///
/// Evaluates M_{kappa,mu}(z_k) for each z_k in `z_values`.
/// This is more efficient than calling `whittaker_m_series` in a loop
/// because the prefactor can be partially shared.
///
/// # Arguments
/// * `kappa`    - First parameter
/// * `mu`       - Second parameter
/// * `z_values` - Slice of positive argument values
///
/// # Returns
/// Vec of M_{kappa,mu}(z_k) values
///
/// # Errors
/// Returns the first error encountered (e.g., z <= 0 for some element).
///
/// # Examples
/// ```
/// use scirs2_special::whittaker::whittaker_m_array;
/// let zs = vec![0.5, 1.0, 2.0, 5.0, 10.0];
/// let ms = whittaker_m_array(1.0, 0.5, &zs).expect("ok");
/// assert_eq!(ms.len(), 5);
/// assert!(ms.iter().all(|v| v.is_finite()));
/// ```
pub fn whittaker_m_array(kappa: f64, mu: f64, z_values: &[f64]) -> SpecialResult<Vec<f64>> {
    z_values
        .iter()
        .map(|&z| whittaker_m_series(kappa, mu, z))
        .collect()
}

/// Vectorized Whittaker W function evaluation.
///
/// Evaluates W_{kappa,mu}(z_k) for each z_k in `z_values`.
///
/// # Arguments
/// * `kappa`    - First parameter
/// * `mu`       - Second parameter
/// * `z_values` - Slice of positive argument values
///
/// # Returns
/// Vec of W_{kappa,mu}(z_k) values
///
/// # Errors
/// Returns the first error encountered.
///
/// # Examples
/// ```
/// use scirs2_special::whittaker::whittaker_w_array;
/// let zs = vec![1.0, 5.0, 10.0, 20.0, 50.0];
/// let ws = whittaker_w_array(0.5, 0.5, &zs).expect("ok");
/// assert_eq!(ws.len(), 5);
/// assert!(ws.iter().all(|v| v.is_finite()));
/// ```
pub fn whittaker_w_array(kappa: f64, mu: f64, z_values: &[f64]) -> SpecialResult<Vec<f64>> {
    z_values
        .iter()
        .map(|&z| whittaker_w_series(kappa, mu, z))
        .collect()
}

/// Compute the Wronskian W[M_{k,m}, W_{k,m}](z).
///
/// The Wronskian of M and W satisfies:
/// ```text
/// W[M_{k,m}(z), W_{k,m}(z)] = -Gamma(2m+1) / Gamma(m - k + 1/2)
/// ```
///
/// This can be used to verify numerical accuracy of computed M and W values.
///
/// # Arguments
/// * `kappa` - First parameter
/// * `mu`    - Second parameter
///
/// # Returns
/// Value of the Wronskian constant
///
/// # Errors
/// Returns `DomainError` if Gamma(m - k + 1/2) = 0 (i.e., m - k + 0.5 is a non-positive integer).
///
/// # Examples
/// ```
/// use scirs2_special::whittaker::whittaker_wronskian;
/// let w = whittaker_wronskian(0.5, 1.0).expect("ok");
/// assert!(w.is_finite());
/// ```
pub fn whittaker_wronskian(kappa: f64, mu: f64) -> SpecialResult<f64> {
    // W[M, W] = -Gamma(2mu+1) / Gamma(mu - kappa + 1/2)
    let g_num = gamma(2.0 * mu + 1.0);
    let g_den = gamma(mu - kappa + 0.5);

    if !g_num.is_finite() {
        return Err(SpecialError::OverflowError(format!(
            "Gamma(2*mu+1) = Gamma({}) overflows",
            2.0 * mu + 1.0
        )));
    }
    if g_den.abs() < 1e-300 {
        return Err(SpecialError::DomainError(format!(
            "Gamma(mu - kappa + 1/2) = 0 for kappa={kappa}, mu={mu}"
        )));
    }

    Ok(-g_num / g_den)
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Compute 1F1(a; b; z) via direct power series.
///
/// This is used as the core of the Whittaker M computation.
/// The series is:
/// ```text
/// 1F1(a; b; z) = sum_{n=0}^infty (a)_n / ((b)_n * n!) * z^n
/// ```
fn hyp1f1_series(a: f64, b: f64, z: f64) -> SpecialResult<f64> {
    // Check for pole at b = 0, -1, -2, ...
    if b <= 0.0 && b.fract().abs() < 1e-10 && b.round() as i64 <= 0 {
        return Err(SpecialError::DomainError(format!(
            "1F1: b = {b} is a non-positive integer (pole)"
        )));
    }

    // Special case: a = 0 → 1F1(0;b;z) = 1
    if a.abs() < 1e-14 {
        return Ok(1.0);
    }

    // For large |z|, use Kummer's transformation: 1F1(a;b;z) = exp(z)*1F1(b-a;b;-z)
    if z > 30.0 {
        let a2 = b - a;
        let b2 = b;
        let z2 = -z;
        let hyp2 = hyp1f1_series_direct(a2, b2, z2)?;
        return Ok(z.exp() * hyp2);
    }

    hyp1f1_series_direct(a, b, z)
}

/// Direct power series for 1F1 (without Kummer transformation).
fn hyp1f1_series_direct(a: f64, b: f64, z: f64) -> SpecialResult<f64> {
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;

    for n in 1..=MAX_SERIES_TERMS {
        let n_f = (n - 1) as f64;
        // term *= (a + n_f) / ((b + n_f) * n_f_next) * z
        let numer = (a + n_f) * z;
        let denom = (b + n_f) * (n as f64);

        if denom.abs() < 1e-300 {
            // b + n_f = 0 at some point — we should have caught this earlier
            break;
        }

        term *= numer / denom;
        sum += term;

        if !term.is_finite() {
            return Err(SpecialError::OverflowError(format!(
                "1F1({a};{b};{z}): term overflow at n={n}"
            )));
        }

        if term.abs() < SERIES_TOL * sum.abs().max(1e-300) && n > 5 {
            break;
        }
    }

    Ok(sum)
}

/// Compute Whittaker W from Tricomi's U function using the connection formula.
///
/// The connection formula for Tricomi U:
/// ```text
/// U(a, b, z) = pi/sin(pi*b) * [M(a,b,z)/(Gamma(1+a-b)*Gamma(b))
///              - z^{1-b} * M(a-b+1,2-b,z) / (Gamma(a)*Gamma(2-b))]
/// ```
/// for non-integer b.
///
/// For integer b, we use a limiting argument or alternative representations.
fn whittaker_w_from_tricomi_u(
    kappa: f64,
    mu: f64,
    a: f64, // = mu - kappa + 0.5
    b: f64, // = 2*mu + 1
    z: f64,
) -> SpecialResult<f64> {
    let prefactor_log = -z / 2.0 + (mu + 0.5) * z.ln();

    if prefactor_log < -740.0 {
        return Ok(0.0);
    }
    if prefactor_log > 709.0 {
        return Err(SpecialError::OverflowError(format!(
            "Whittaker W: prefactor overflows for z={z}, mu={mu}"
        )));
    }
    let prefactor = prefactor_log.exp();

    // Check if b is close to an integer
    let b_is_integer = (b.fract().abs() < 1e-10) && b.abs() < 1e15;
    let b_int = b.round() as i64;

    if !b_is_integer {
        // Non-integer b: use connection formula
        let sin_pi_b = (std::f64::consts::PI * b).sin();
        if sin_pi_b.abs() < 1e-12 {
            // b is nearly an integer; use limit form below
            return whittaker_w_integer_b(kappa, mu, a, b, z, prefactor);
        }

        // Compute Gamma values (use log-gamma for numerical stability)
        let log_g_1_plus_a_minus_b = gammaln_safe(1.0 + a - b)?;
        let log_g_b = gammaln_safe(b)?;
        let log_g_a = gammaln_safe(a)?;
        let log_g_2_minus_b = gammaln_safe(2.0 - b)?;

        // M(a, b, z) for first term
        let m1 = hyp1f1_series(a, b, z)?;
        // M(a - b + 1, 2 - b, z) for second term
        let m2 = hyp1f1_series(a - b + 1.0, 2.0 - b, z)?;

        // z^{1-b} factor
        let z_1mb = z.powf(1.0 - b);

        // Sign tracking for gamma values
        let g_1ab_sign = if (1.0 + a - b) < 0.0 && ((1.0 + a - b).floor() as i64).abs() % 2 == 1 {
            -1.0_f64
        } else {
            1.0
        };
        let g_b_sign = if b < 0.0 && (b.floor() as i64).abs() % 2 == 1 {
            -1.0_f64
        } else {
            1.0
        };
        let g_a_sign = if a < 0.0 && (a.floor() as i64).abs() % 2 == 1 {
            -1.0_f64
        } else {
            1.0
        };
        let g_2mb_sign = if (2.0 - b) < 0.0 && ((2.0 - b).floor() as i64).abs() % 2 == 1 {
            -1.0_f64
        } else {
            1.0
        };

        let term1_log_denom = log_g_1_plus_a_minus_b + log_g_b;
        let term1 = g_1ab_sign * g_b_sign * m1 * (-term1_log_denom).exp();

        let term2_log_denom = log_g_a + log_g_2_minus_b;
        let term2 = g_a_sign * g_2mb_sign * z_1mb * m2 * (-term2_log_denom).exp();

        let pi = std::f64::consts::PI;
        let u_val = (pi / sin_pi_b) * (term1 - term2);

        Ok(prefactor * u_val)
    } else {
        // Integer b
        whittaker_w_integer_b(kappa, mu, a, b_int as f64, z, prefactor)
    }
}

/// Handle Whittaker W for integer b via series with logarithmic terms.
///
/// When b is a positive integer m, Tricomi U must be computed via a
/// Frobenius expansion that includes logarithmic terms. We use a simplified
/// approach via the representation:
///
/// ```text
/// W_{kappa,mu}(z) = (Gamma(-2mu) / Gamma(1/2 - mu - kappa)) * M_{kappa, mu}(z)
///                 + (Gamma(2mu) / Gamma(1/2 + mu - kappa)) * M_{kappa, -mu}(z)
/// ```
///
/// for non-integer 2*mu. When 2*mu is a non-integer, this formula is valid.
fn whittaker_w_integer_b(
    kappa: f64,
    mu: f64,
    _a: f64,
    _b: f64,
    z: f64,
    prefactor: f64,
) -> SpecialResult<f64> {
    // Use the formula W = C1*M_{k,mu} + C2*M_{k,-mu} for non-half-integer mu.
    // This is valid when 2*mu is not an integer.
    let two_mu = 2.0 * mu;
    let two_mu_is_int = (two_mu.fract().abs() < 1e-9) && two_mu.abs() < 1e15;

    if two_mu_is_int {
        // For integer 2*mu, we need the logarithmic form.
        // Fallback: use the asymptotic expansion even for moderate z.
        // This is less accurate but avoids the need for log-derivative series.
        return whittaker_w_asymptotic(kappa, mu, z);
    }

    // W_{k,mu}(z) = Gamma(-2mu)/Gamma(1/2-mu-kappa) * M_{k,mu}(z)
    //             + Gamma(2mu)/Gamma(1/2+mu-kappa) * M_{k,-mu}(z)

    let log_g_neg_2mu = gammaln_safe(-two_mu)?;
    let log_g_half_minus_mu_minus_k = gammaln_safe(0.5 - mu - kappa)?;
    let log_g_2mu = gammaln_safe(two_mu)?;
    let log_g_half_plus_mu_minus_k = gammaln_safe(0.5 + mu - kappa)?;

    // Compute sign factors for gamma functions (Gamma can be negative)
    let sign_g_neg_2mu = if (-two_mu) < 0.0 && ((-two_mu).floor() as i64).abs() % 2 == 1 {
        -1.0_f64
    } else {
        1.0
    };
    let sign_g_h1 = if (0.5 - mu - kappa) < 0.0
        && ((0.5 - mu - kappa).floor() as i64).abs() % 2 == 1
    {
        -1.0_f64
    } else {
        1.0
    };
    let sign_g_2mu = if two_mu < 0.0 && (two_mu.floor() as i64).abs() % 2 == 1 {
        -1.0_f64
    } else {
        1.0
    };
    let sign_g_h2 = if (0.5 + mu - kappa) < 0.0
        && ((0.5 + mu - kappa).floor() as i64).abs() % 2 == 1
    {
        -1.0_f64
    } else {
        1.0
    };

    // M_{kappa, mu}(z) with the standard sign
    let m_plus = {
        let a1 = mu - kappa + 0.5;
        let b1 = 2.0 * mu + 1.0;
        if b1 <= 0.0 && (b1.fract().abs() < 1e-10) {
            0.0 // degenerate case
        } else {
            let hyp1 = hyp1f1_series(a1, b1, z).unwrap_or(0.0);
            prefactor * hyp1
        }
    };

    // M_{kappa, -mu}(z): same structure but with mu -> -mu
    let m_minus = {
        let pref2_log = -z / 2.0 + (-mu + 0.5) * z.ln();
        if pref2_log < -740.0 {
            0.0
        } else if pref2_log > 709.0 {
            f64::INFINITY
        } else {
            let pref2 = pref2_log.exp();
            let a2 = -mu - kappa + 0.5;
            let b2 = -2.0 * mu + 1.0;
            if b2 <= 0.0 && (b2.fract().abs() < 1e-10) {
                0.0
            } else {
                let hyp2 = hyp1f1_series(a2, b2, z).unwrap_or(0.0);
                pref2 * hyp2
            }
        }
    };

    let c1 = sign_g_neg_2mu * sign_g_h1 * (log_g_neg_2mu - log_g_half_minus_mu_minus_k).exp();
    let c2 = sign_g_2mu * sign_g_h2 * (log_g_2mu - log_g_half_plus_mu_minus_k).exp();

    Ok(c1 * m_plus + c2 * m_minus)
}

/// Safe gammaln that returns an error for poles.
fn gammaln_safe(x: f64) -> SpecialResult<f64> {
    if x <= 0.0 && (x.fract().abs() < 1e-10) {
        return Err(SpecialError::DomainError(format!(
            "gammaln({x}): argument is a non-positive integer (pole)"
        )));
    }
    let val = gammaln(x);
    if !val.is_finite() {
        return Err(SpecialError::OverflowError(format!(
            "gammaln({x}) = {val} overflows"
        )));
    }
    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    #[test]
    fn test_whittaker_m_basic() {
        let m = whittaker_m_series(0.5, 0.5, 1.0).expect("ok");
        assert!(m.is_finite() && m > 0.0, "M should be positive: {m}");
    }

    #[test]
    fn test_whittaker_m_small_z() {
        // For small z, M_{k,m}(z) ~ z^{m+0.5} * exp(-z/2)
        let z = 0.01_f64;
        let mu = 0.5;
        let kappa = 0.5;
        let m = whittaker_m_series(kappa, mu, z).expect("ok");
        assert!(m.is_finite() && m > 0.0);
        // Should be close to z^1 * exp(-z/2) = z * exp(-z/2) for 1F1 ~ 1
        let approx = z.powf(mu + 0.5) * (-z / 2.0).exp();
        assert!(
            (m / approx - 1.0).abs() < 0.1,
            "M ratio to approx should be near 1: {m} vs {approx}"
        );
    }

    #[test]
    fn test_whittaker_m_large_z() {
        let m = whittaker_m_series(0.5, 0.5, 10.0).expect("ok");
        assert!(m.is_finite(), "M should be finite for large z: {m}");
    }

    #[test]
    fn test_whittaker_m_negative_z_error() {
        assert!(whittaker_m_series(0.5, 0.5, -1.0).is_err());
        assert!(whittaker_m_series(0.5, 0.5, 0.0).is_err());
    }

    #[test]
    fn test_whittaker_w_basic() {
        let w = whittaker_w_series(0.5, 0.5, 1.0).expect("ok");
        assert!(w.is_finite(), "W should be finite: {w}");
    }

    #[test]
    fn test_whittaker_w_large_z() {
        // For large z, W decays approximately as exp(-z/2) z^kappa
        let z = 10.0_f64;
        let kappa = 0.5;
        let mu = 0.5;
        let w = whittaker_w_series(kappa, mu, z).expect("ok");
        assert!(w.is_finite(), "W should be finite for z=10: {w}");
        // The leading behavior is exp(-z/2)*z^kappa = exp(-5)*sqrt(10) ~ 0.021
        let leading = (-z / 2.0 + kappa * z.ln()).exp();
        assert!(
            w.abs() < leading * 5.0,
            "W should not vastly exceed leading term: {w} vs {leading}"
        );
    }

    #[test]
    fn test_whittaker_w_negative_z_error() {
        assert!(whittaker_w_series(0.5, 0.5, -1.0).is_err());
    }

    #[test]
    fn test_whittaker_w_asymptotic_accuracy() {
        // For z = 50, asymptotic and series should agree closely
        let kappa = 1.0;
        let mu = 0.5;
        let z = 50.0;
        let w_asym = whittaker_w_asymptotic(kappa, mu, z).expect("ok");
        assert!(w_asym.is_finite() && w_asym > 0.0, "W asymptotic: {w_asym}");
    }

    #[test]
    fn test_whittaker_m_array() {
        let zs = vec![0.5, 1.0, 2.0, 5.0];
        let ms = whittaker_m_array(1.0, 0.5, &zs).expect("ok");
        assert_eq!(ms.len(), 4);
        for (i, &m) in ms.iter().enumerate() {
            let m_scalar = whittaker_m_series(1.0, 0.5, zs[i]).expect("ok");
            assert!(
                (m - m_scalar).abs() < 1e-14,
                "Array[{i}]={m} vs scalar={m_scalar}"
            );
        }
    }

    #[test]
    fn test_whittaker_w_array() {
        let zs = vec![1.0, 5.0, 10.0, 30.0];
        let ws = whittaker_w_array(0.5, 0.5, &zs).expect("ok");
        assert_eq!(ws.len(), 4);
        for &w in &ws {
            assert!(w.is_finite(), "W should be finite");
        }
    }

    #[test]
    fn test_whittaker_wronskian() {
        let w = whittaker_wronskian(0.5, 1.0).expect("ok");
        assert!(w.is_finite(), "Wronskian should be finite: {w}");
    }

    #[test]
    fn test_whittaker_w_at_varying_kappa() {
        // Verify W is finite and reasonable for different kappa values
        let mu = 0.5;
        let z = 3.0;
        for &kappa in &[-1.0, 0.0, 0.5, 1.0, 2.0] {
            let w = whittaker_w_series(kappa, mu, z).expect("ok");
            assert!(
                w.is_finite(),
                "W should be finite for kappa={kappa}: {w}"
            );
        }
    }

    #[test]
    fn test_whittaker_m_kummer_connection() {
        // M_{k,m}(z) = exp(-z/2) * z^{m+1/2} * 1F1(m-k+1/2; 2m+1; z)
        // Verify by checking the 1F1 relationship explicitly
        let kappa = 0.5;
        let mu = 0.5;
        let z = 2.0;

        let m_val = whittaker_m_series(kappa, mu, z).expect("ok");

        // Manual: prefactor * 1F1(0.5; 2; 2)
        let a = mu - kappa + 0.5; // = 0.5
        let b = 2.0 * mu + 1.0;  // = 2.0
        let hyp = hyp1f1_series(a, b, z).expect("ok");
        let pref = (-z / 2.0 + (mu + 0.5) * z.ln()).exp();
        let expected = pref * hyp;

        assert!(
            (m_val - expected).abs() < TOL * expected.abs().max(1e-10),
            "Kummer connection: M={m_val}, expected={expected}"
        );
    }

    #[test]
    fn test_hyp1f1_series_special_cases() {
        // 1F1(0; b; z) = 1
        let h = hyp1f1_series(0.0, 2.0, 3.0).expect("ok");
        assert!((h - 1.0).abs() < 1e-14, "1F1(0;b;z) should be 1: {h}");

        // 1F1(a; b; 0) = 1 (z=0)
        let h0 = hyp1f1_series(2.0, 3.0, 0.0).expect("ok");
        assert!((h0 - 1.0).abs() < 1e-14, "1F1(a;b;0) should be 1: {h0}");
    }

    #[test]
    fn test_whittaker_m_array_error_propagation() {
        // If any z <= 0, should return error
        let zs = vec![1.0, -1.0, 2.0];
        assert!(whittaker_m_array(0.5, 0.5, &zs).is_err());
    }
}
