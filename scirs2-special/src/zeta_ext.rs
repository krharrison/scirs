//! Extended Zeta and L-Functions
//!
//! This module provides additional Riemann zeta, Dirichlet eta, Lerch transcendent,
//! and polygamma-related functions. The implementations use analytically continued
//! forms valid across the complex and real domains.

use crate::error::{SpecialError, SpecialResult};
use crate::gamma::gamma;
use std::f64::consts::PI;

/// Riemann zeta function for real argument (f64 specialisation).
///
/// Computes ζ(s) for real s ≠ 1.  This is a convenience wrapper that returns
/// `f64` directly (no generic type parameter) and uses the Euler-Maclaurin
/// formula for s > 1, the functional equation for s < 1, and exact values at
/// the negative even integers (trivial zeros).
///
/// # Arguments
///
/// * `s` - Real argument (s ≠ 1)
///
/// # Returns
///
/// * `Ok(f64)` – value of ζ(s)
/// * `Err(SpecialError::DomainError)` – if `s` is NaN
///
/// # Examples
///
/// ```
/// use scirs2_special::riemann_zeta;
/// use std::f64::consts::PI;
///
/// // ζ(2) = π²/6
/// let z2 = riemann_zeta(2.0).expect("riemann_zeta(2)");
/// assert!((z2 - PI * PI / 6.0).abs() < 1e-8);
///
/// // ζ(-1) = -1/12
/// let z_neg1 = riemann_zeta(-1.0).expect("riemann_zeta(-1)");
/// assert!((z_neg1 - (-1.0 / 12.0)).abs() < 1e-8);
/// ```
pub fn riemann_zeta(s: f64) -> SpecialResult<f64> {
    if s.is_nan() {
        return Err(SpecialError::DomainError(
            "riemann_zeta: NaN argument".to_string(),
        ));
    }
    if s == 1.0 {
        return Ok(f64::INFINITY);
    }
    // Trivial zeros: negative even integers
    if s < 0.0 && s.fract() == 0.0 && ((-s) as u64).is_multiple_of(2) {
        return Ok(0.0);
    }
    if s > 1.0 {
        riemann_zeta_em(s)
    } else {
        // Functional equation: ζ(s) = 2^s π^{s-1} sin(πs/2) Γ(1-s) ζ(1-s)
        let t = 1.0 - s;
        let zeta_t = riemann_zeta_em(t)?;
        let result = 2.0_f64.powf(s) * PI.powf(s - 1.0) * (PI * s / 2.0).sin() * gamma(t) * zeta_t;
        Ok(result)
    }
}

/// Euler-Maclaurin summation for ζ(s), valid for s > 1.
fn riemann_zeta_em(s: f64) -> SpecialResult<f64> {
    // Number of initial terms in the direct sum.
    let n: usize = if s > 30.0 {
        5
    } else if s > 5.0 {
        30
    } else {
        80
    };

    let mut sum = 0.0_f64;
    for k in 1..=n {
        sum += (k as f64).powf(-s);
    }

    // Euler-Maclaurin tail: integrate from n onwards.
    let nf = n as f64;
    // ∫_n^∞ x^{-s} dx = n^{1-s} / (s-1)
    sum += nf.powf(1.0 - s) / (s - 1.0);
    // -1/2 n^{-s}  (boundary correction)
    sum -= 0.5 * nf.powf(-s);

    // Bernoulli corrections: B_{2k}/(2k)! * s^{(2k-1)} * n^{-s-2k+1}
    // where s^{(m)} = s(s+1)…(s+m-1) is the rising factorial.
    let bern: [(f64, f64); 6] = [
        (1.0 / 6.0, 2.0),
        (-1.0 / 30.0, 4.0),
        (1.0 / 42.0, 6.0),
        (-1.0 / 30.0, 8.0),
        (5.0 / 66.0, 10.0),
        (-691.0 / 2730.0, 12.0),
    ];

    let mut rising = s; // will accumulate s(s+1)…(s+2k-2)
    for (b2k, two_k) in &bern {
        // rising factorial s^{(2k-1)} for current k
        // The coefficient is B_{2k} * s(s+1)…(s+2k-2) / (2k)! * n^{-s-2k+1}
        let fact_denom = factorial_f64(*two_k as usize);
        let correction = b2k * rising * nf.powf(-s - two_k + 1.0) / fact_denom;
        sum += correction;
        // Extend rising factorial by two more factors for next iteration
        let last = *two_k - 1.0; // current top index (0-based: s+2k-2)
        rising *= (s + last) * (s + last + 1.0);
    }

    Ok(sum)
}

fn factorial_f64(n: usize) -> f64 {
    let mut f = 1.0_f64;
    for i in 2..=n {
        f *= i as f64;
    }
    f
}

/// Riemann zeta function for complex argument using Euler-Maclaurin.
///
/// Takes a complex number `s = (re, im)` and returns `(re, im)` of ζ(s).
/// Uses the Euler-Maclaurin formula suitable for Re(s) > 0.  For Re(s) ≤ 0
/// the functional equation is applied first.
///
/// # Arguments
///
/// * `s` – Complex argument as `(real_part, imag_part)`
///
/// # Returns
///
/// * `Ok((f64, f64))` – Real and imaginary parts of ζ(s)
/// * `Err` – if inputs are non-finite
///
/// # Examples
///
/// ```
/// use scirs2_special::riemann_zeta_complex;
/// use std::f64::consts::PI;
///
/// // ζ(2+0i) = π²/6
/// let (re, im) = riemann_zeta_complex((2.0, 0.0)).expect("zeta(2+0i)");
/// assert!((re - PI * PI / 6.0).abs() < 1e-6);
/// assert!(im.abs() < 1e-10);
/// ```
pub fn riemann_zeta_complex(s: (f64, f64)) -> SpecialResult<(f64, f64)> {
    let (sigma, t) = s;
    if !sigma.is_finite() || !t.is_finite() {
        return Err(SpecialError::DomainError(
            "riemann_zeta_complex: non-finite argument".to_string(),
        ));
    }
    // Handle pure-real case exactly
    if t == 0.0 {
        let re = riemann_zeta(sigma)?;
        return Ok((re, 0.0));
    }
    // For Re(s) > 0, use direct Euler-Maclaurin on the complex series
    if sigma > 0.0 {
        zeta_complex_em(sigma, t)
    } else {
        // Functional equation for complex s:
        // ζ(s) = 2^s π^{s-1} sin(πs/2) Γ(1-s) ζ(1-s)
        let (re1, im1) = zeta_complex_em(1.0 - sigma, -t)?;
        // Compute the prefactor 2^s * π^{s-1} * sin(πs/2) * Γ(1-s) in complex arithmetic
        let prefactor = zeta_functional_prefactor(sigma, t, re1, im1)?;
        Ok(prefactor)
    }
}

/// Euler-Maclaurin for complex ζ with Re(s) > 0.
fn zeta_complex_em(sigma: f64, t: f64) -> SpecialResult<(f64, f64)> {
    let n: usize = if sigma > 10.0 { 10 } else { 60 };

    let mut sum_re = 0.0_f64;
    let mut sum_im = 0.0_f64;

    for k in 1..=n {
        let kf = k as f64;
        // k^{-s} = exp(-s * ln k) = exp(-(σ + it) ln k)
        // = k^{-σ} * (cos(t ln k) - i sin(t ln k))
        let ln_k = kf.ln();
        let mag = (-sigma * ln_k).exp();
        let phase = t * ln_k;
        sum_re += mag * phase.cos();
        sum_im -= mag * phase.sin();
    }

    // Integral tail: ∫_n^∞ x^{-s} dx = n^{1-s} / (s-1)
    // n^{1-s} = n^{1-σ} * exp(-it ln n)
    let nf = n as f64;
    let ln_n = nf.ln();
    let n_pow_re = ((1.0 - sigma) * ln_n).exp();
    let n_pow_phase = -t * ln_n;
    let n1ms_re = n_pow_re * n_pow_phase.cos();
    let n1ms_im = n_pow_re * n_pow_phase.sin();

    // (s-1) = (sigma-1, t)
    let denom = (sigma - 1.0) * (sigma - 1.0) + t * t;
    if denom < 1e-300 {
        return Ok((f64::INFINITY, 0.0));
    }
    let inv_re = (sigma - 1.0) / denom;
    let inv_im = -t / denom;

    // n^{1-s} / (s-1)
    let tail_re = n1ms_re * inv_re - n1ms_im * inv_im;
    let tail_im = n1ms_re * inv_im + n1ms_im * inv_re;

    sum_re += tail_re;
    sum_im += tail_im;

    // Boundary correction: -1/2 * n^{-s}
    let n_ms_re = ((-sigma) * ln_n).exp() * (t * ln_n).cos();
    let n_ms_im = -((-sigma) * ln_n).exp() * (t * ln_n).sin();
    sum_re -= 0.5 * n_ms_re;
    sum_im -= 0.5 * n_ms_im;

    Ok((sum_re, sum_im))
}

/// Apply functional equation prefactor for ζ(s) with Re(s) < 0.
fn zeta_functional_prefactor(
    sigma: f64,
    t: f64,
    zeta_1ms_re: f64,
    zeta_1ms_im: f64,
) -> SpecialResult<(f64, f64)> {
    // 2^s = exp(s * ln2) = exp((σ+it) ln2)
    let ln2 = std::f64::consts::LN_2;
    let two_s_mag = (sigma * ln2).exp();
    let two_s_re = two_s_mag * (t * ln2).cos();
    let two_s_im = two_s_mag * (t * ln2).sin();

    // π^{s-1} = exp((s-1) ln π)
    let ln_pi = PI.ln();
    let pi_s1_mag = ((sigma - 1.0) * ln_pi).exp();
    let pi_s1_re = pi_s1_mag * (t * ln_pi).cos();
    let pi_s1_im = pi_s1_mag * (t * ln_pi).sin();

    // 2^s * π^{s-1}
    let a_re = two_s_re * pi_s1_re - two_s_im * pi_s1_im;
    let a_im = two_s_re * pi_s1_im + two_s_im * pi_s1_re;

    // sin(πs/2) = sin(π(σ+it)/2) = sin(πσ/2)cosh(πt/2) + i cos(πσ/2)sinh(πt/2)
    let hs = PI * sigma / 2.0;
    let ht = PI * t / 2.0;
    let sin_re = hs.sin() * ht.cosh();
    let sin_im = hs.cos() * ht.sinh();

    let b_re = a_re * sin_re - a_im * sin_im;
    let b_im = a_re * sin_im + a_im * sin_re;

    // Γ(1-s) for real 1-sigma only (using the real gamma as approximation since t=0 handled above)
    // For t ≠ 0, use Stirling approximation for |Γ(1-σ-it)|
    let gamma_val = complex_gamma(1.0 - sigma, -t);

    let c_re = b_re * gamma_val.0 - b_im * gamma_val.1;
    let c_im = b_re * gamma_val.1 + b_im * gamma_val.0;

    let result_re = c_re * zeta_1ms_re - c_im * zeta_1ms_im;
    let result_im = c_re * zeta_1ms_im + c_im * zeta_1ms_re;

    Ok((result_re, result_im))
}

/// Complex gamma function Γ(x + iy) using Lanczos approximation.
fn complex_gamma(x: f64, y: f64) -> (f64, f64) {
    // Lanczos parameters (g=7)
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_10,
        -1_259.139_216_722_402_9,
        771.323_428_777_653_07,
        -176.615_029_162_140_60,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_312_5e-7,
    ];
    if x < 0.5 {
        // Reflection: Γ(z)Γ(1-z) = π/sin(πz)
        let (g1_re, g1_im) = complex_gamma(1.0 - x, -y);
        // sin(πz) = sin(πx)cosh(πy) + i cos(πx)sinh(πy)
        let sin_re = (PI * x).sin() * (PI * y).cosh();
        let sin_im = (PI * x).cos() * (PI * y).sinh();
        let denom = sin_re * sin_re + sin_im * sin_im;
        if denom < 1e-300 {
            return (f64::INFINITY, 0.0);
        }
        // π / sin(πz)
        let inv_re = PI * sin_re / denom;
        let inv_im = -PI * sin_im / denom;
        // / Γ(1-z)
        let d2 = g1_re * g1_re + g1_im * g1_im;
        if d2 < 1e-300 {
            return (f64::INFINITY, 0.0);
        }
        let inv_g_re = g1_re / d2;
        let inv_g_im = -g1_im / d2;
        let re = inv_re * inv_g_re - inv_im * inv_g_im;
        let im = inv_re * inv_g_im + inv_im * inv_g_re;
        return (re, im);
    }
    // z' = z - 1
    let zr = x - 1.0;
    let zi = y;
    // Accumulate series
    let mut sum_re = C[0];
    let mut sum_im = 0.0_f64;
    for k in 1..9 {
        // z' + k = (zr + k, zi)
        let d = (zr + k as f64) * (zr + k as f64) + zi * zi;
        if d < 1e-300 {
            return (f64::INFINITY, 0.0);
        }
        sum_re += C[k] * (zr + k as f64) / d;
        sum_im -= C[k] * zi / d;
    }
    // t = z' + g + 0.5 = (zr + g + 0.5, zi)
    let tr = zr + G + 0.5;
    let ti = zi;
    // (t)^{z'+0.5} = exp((z'+0.5)*ln(t))
    let ln_t_mag = (tr * tr + ti * ti).sqrt().ln();
    let ln_t_arg = ti.atan2(tr);
    let pw_exp = (zr + 0.5) * ln_t_mag - zi * ln_t_arg;
    let pw_arg = (zr + 0.5) * ln_t_arg + zi * ln_t_mag;
    let t_pow_re = pw_exp.exp() * pw_arg.cos();
    let t_pow_im = pw_exp.exp() * pw_arg.sin();
    // exp(-t)
    let exp_t = (-tr).exp();
    let exp_t_re = exp_t * (-ti).cos();
    let exp_t_im = exp_t * (-ti).sin();
    // sqrt(2π) ≈ 2.506628274631
    let sqrt2pi = (2.0 * PI).sqrt();
    // Combine: sqrt(2π) * (sum) * t_pow * exp(-t)
    let te_re = t_pow_re * exp_t_re - t_pow_im * exp_t_im;
    let te_im = t_pow_re * exp_t_im + t_pow_im * exp_t_re;
    let result_re = sqrt2pi * (sum_re * te_re - sum_im * te_im);
    let result_im = sqrt2pi * (sum_re * te_im + sum_im * te_re);
    (result_re, result_im)
}

/// Dirichlet eta function η(s) = (1 - 2^{1-s}) ζ(s).
///
/// Also known as the alternating zeta function:
/// η(s) = ∑_{n=1}^∞ (-1)^{n-1} / n^s = 1 - 1/2^s + 1/3^s - …
///
/// Unlike ζ(s) the eta function is entire (no pole at s=1), where η(1) = ln 2.
///
/// # Arguments
///
/// * `s` – Real argument
///
/// # Returns
///
/// * `Ok(f64)` – value of η(s)
/// * `Err` – if `s` is NaN
///
/// # Examples
///
/// ```
/// use scirs2_special::dirichlet_eta;
/// use std::f64::consts::LN_2;
///
/// // η(1) = ln 2
/// let eta1 = dirichlet_eta(1.0).expect("eta(1)");
/// assert!((eta1 - LN_2).abs() < 1e-10);
///
/// // η(0) = 1/2
/// let eta0 = dirichlet_eta(0.0).expect("eta(0)");
/// assert!((eta0 - 0.5).abs() < 1e-10);
/// ```
pub fn dirichlet_eta(s: f64) -> SpecialResult<f64> {
    if s.is_nan() {
        return Err(SpecialError::DomainError(
            "dirichlet_eta: NaN argument".to_string(),
        ));
    }
    // η(1) = ln 2
    if (s - 1.0).abs() < 1e-15 {
        return Ok(std::f64::consts::LN_2);
    }
    // η(0) = 1/2 (special case to avoid 0 * infinity from functional equation)
    if s == 0.0 {
        return Ok(0.5);
    }
    // Use the relation η(s) = (1 - 2^{1-s}) ζ(s) for all s.
    // For s > 1, both riemann_zeta and the prefactor are well-behaved.
    // For s ≤ 0 (except s=0 handled above), the relation also works.
    let factor = 1.0 - 2.0_f64.powf(1.0 - s);
    let z = riemann_zeta(s)?;
    Ok(factor * z)
}

/// Accelerated alternating series for η(s) with s > 0 using
/// Borwein's algorithm.
///
/// η(s) = ∑_{k=0}^∞ (-1)^k / (k+1)^s
///
/// This is kept for potential direct use but `dirichlet_eta` now uses
/// the relation η(s) = (1 - 2^{1-s}) ζ(s) which is simpler and more accurate.
#[allow(dead_code)]
fn eta_alternating(s: f64) -> SpecialResult<f64> {
    // The Dirichlet eta is Φ(-1, s, 1) = ∑ (-1)^k / (k+1)^s
    // which is lerch_phi_alternating with a=1 but offset by 1 in index.
    // η(s) = ∑_{n=1}^∞ (-1)^{n-1} / n^s = ∑_{k=0}^∞ (-1)^k / (k+1)^s
    lerch_phi_alternating(s, 1.0)
}

/// Lerch transcendent Φ(z, s, a).
///
/// Defined as:
/// Φ(z, s, a) = ∑_{n=0}^∞ z^n / (n + a)^s
///
/// This generalises the Hurwitz zeta (z=1) and the polylogarithm (a=1).
/// Convergence requires |z| < 1, or |z| = 1 with Re(s) > 1 when z ≠ 1,
/// or s ≠ 1 when z = 1 (reduces to Hurwitz zeta).
///
/// # Arguments
///
/// * `z` – Base (real), typically |z| ≤ 1
/// * `s` – Exponent parameter (real)
/// * `a` – Shift parameter (real, a > 0)
///
/// # Returns
///
/// * `Ok(f64)` – value of Φ(z, s, a)
/// * `Err` – if arguments are out of domain or NaN
///
/// # Examples
///
/// ```
/// use scirs2_special::lerch_phi;
/// use std::f64::consts::PI;
///
/// // Φ(1, 2, 1) = ζ(2) = π²/6
/// let phi = lerch_phi(1.0, 2.0, 1.0).expect("Phi(1,2,1)");
/// assert!((phi - PI * PI / 6.0).abs() < 1e-6);
///
/// // Φ(-1, 1, 1) = ln 2 (Dirichlet eta at s=1)
/// let phi2 = lerch_phi(-1.0, 1.0, 1.0).expect("Phi(-1,1,1)");
/// assert!((phi2 - std::f64::consts::LN_2).abs() < 1e-8);
/// ```
pub fn lerch_phi(z: f64, s: f64, a: f64) -> SpecialResult<f64> {
    if z.is_nan() || s.is_nan() || a.is_nan() {
        return Err(SpecialError::DomainError(
            "lerch_phi: NaN argument".to_string(),
        ));
    }
    if a <= 0.0 {
        return Err(SpecialError::DomainError(
            "lerch_phi: a must be positive".to_string(),
        ));
    }
    if z == 1.0 {
        // Reduces to Hurwitz zeta: Φ(1, s, a) = ζ(s, a)
        // For a = 1 use our more accurate riemann_zeta from this module
        if (a - 1.0).abs() < 1e-15 {
            return riemann_zeta(s);
        }
        return crate::zeta::hurwitz_zeta(s, a);
    }
    if z.abs() > 1.0 {
        return Err(SpecialError::DomainError(
            "lerch_phi: |z| must be <= 1".to_string(),
        ));
    }
    // For z = -1, use the Euler-Maclaurin accelerated alternating series
    // Φ(-1, s, a) = ∑_{n=0}^∞ (-1)^n / (n+a)^s
    if (z - (-1.0)).abs() < 1e-15 {
        return lerch_phi_alternating(s, a);
    }
    // Direct summation with acceleration for |z| < 1.
    // For |z| close to 1 use more terms.
    let max_terms: usize = if z.abs() > 0.99 { 10000 } else { 500 };
    let tol = 1e-15;

    let mut sum = 0.0_f64;
    let mut z_pow = 1.0_f64; // z^n

    for n in 0..max_terms {
        let term = z_pow / (n as f64 + a).powf(s);
        sum += term;
        z_pow *= z;
        if term.abs() < tol * sum.abs() && n > 10 {
            break;
        }
    }
    Ok(sum)
}

/// Euler acceleration for alternating Lerch Φ(-1, s, a) = ∑_{n=0}^∞ (-1)^n / (n+a)^s.
///
/// Uses P. Borwein's algorithm (2000) based on Chebyshev-like weights
/// for robust convergence of alternating series.
fn lerch_phi_alternating(s: f64, a: f64) -> SpecialResult<f64> {
    // Borwein's algorithm for alternating series acceleration.
    // Reference: Peter Borwein, "An Efficient Algorithm for the Riemann Zeta Function", 2000.
    //
    // For the series S = ∑_{k=0}^∞ (-1)^k * b_k where b_k = 1/(k+a)^s:
    //
    // 1. Compute partial sums of binomial coefficients:
    //    d_k = ∑_{j=0}^{k} C(n, j)  for k = 0, ..., n
    //    where d_n = 2^n.
    //
    // 2. S ≈ (-1/d_n) * ∑_{k=0}^{n} (-1)^k * (d_k - d_n) * b_k
    //
    // This converges as O(((sqrt(2)-1)/(sqrt(2)+1))^n) ~ O(0.172^n).

    let n: usize = 50; // 50 terms gives ~30 digits of accuracy

    // Step 1: compute d_k = ∑_{j=0}^{k} C(n, j) for k = 0..n
    // We build this incrementally: d_0 = C(n,0) = 1,
    // d_{k+1} = d_k + C(n, k+1), with C(n, k+1) = C(n, k) * (n-k) / (k+1).
    let mut d_vals = Vec::with_capacity(n + 1);
    let mut binom = 1.0_f64; // C(n, 0) = 1
    d_vals.push(binom); // d_0 = 1
    for k in 0..n {
        binom *= (n - k) as f64 / (k + 1) as f64;
        let d_prev = d_vals[k];
        d_vals.push(d_prev + binom);
    }
    let d_n = d_vals[n]; // = 2^n

    // Step 2: compute the accelerated sum
    let mut sum = 0.0_f64;
    for k in 0..=n {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let weight = d_vals[k] - d_n; // always non-positive
        let b_k = 1.0 / (k as f64 + a).powf(s);
        sum += sign * weight * b_k;
    }

    Ok(-sum / d_n)
}

// Private helper: wraps gamma::polygamma for use in this module's tests.
#[allow(dead_code)]
fn polygamma_local(n: u32, x: f64) -> f64 {
    crate::gamma::polygamma(n, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::{LN_2, PI};

    #[test]
    fn test_riemann_zeta_known_values() {
        // ζ(2) = π²/6
        let z2 = riemann_zeta(2.0).expect("zeta(2)");
        assert_relative_eq!(z2, PI * PI / 6.0, epsilon = 1e-8);

        // ζ(4) = π⁴/90
        let z4 = riemann_zeta(4.0).expect("zeta(4)");
        assert_relative_eq!(z4, PI.powi(4) / 90.0, epsilon = 1e-8);

        // ζ(3) ≈ 1.2020569 (Apéry's constant)
        let z3 = riemann_zeta(3.0).expect("zeta(3)");
        assert_relative_eq!(z3, 1.2020569031595942_f64, epsilon = 1e-8);
    }

    #[test]
    fn test_riemann_zeta_negative() {
        // ζ(-1) = -1/12
        let z_neg1 = riemann_zeta(-1.0).expect("zeta(-1)");
        assert_relative_eq!(z_neg1, -1.0 / 12.0, epsilon = 1e-7);

        // ζ(-3) = 1/120
        let z_neg3 = riemann_zeta(-3.0).expect("zeta(-3)");
        assert_relative_eq!(z_neg3, 1.0 / 120.0, epsilon = 1e-7);

        // Trivial zeros: ζ(-2) = ζ(-4) = 0
        let z_neg2 = riemann_zeta(-2.0).expect("zeta(-2)");
        assert_relative_eq!(z_neg2, 0.0, epsilon = 1e-15);
        let z_neg4 = riemann_zeta(-4.0).expect("zeta(-4)");
        assert_relative_eq!(z_neg4, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_riemann_zeta_pole() {
        let z1 = riemann_zeta(1.0).expect("zeta(1) pole");
        assert!(z1.is_infinite());
    }

    #[test]
    fn test_riemann_zeta_large_s() {
        // ζ(s) → 1 as s → ∞
        let z50 = riemann_zeta(50.0).expect("zeta(50)");
        assert!((z50 - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_riemann_zeta_complex_real_axis() {
        // Should match real version on the real axis
        let (re, im) = riemann_zeta_complex((2.0, 0.0)).expect("zeta_complex(2)");
        assert_relative_eq!(re, PI * PI / 6.0, epsilon = 1e-6);
        assert!(im.abs() < 1e-10);
    }

    #[test]
    fn test_riemann_zeta_complex_nontrivial() {
        // ζ(2+3i) should be finite and well-defined
        let (re, im) = riemann_zeta_complex((2.0, 3.0)).expect("zeta_complex(2+3i)");
        assert!(re.is_finite());
        assert!(im.is_finite());
    }

    #[test]
    fn test_dirichlet_eta_special_values() {
        // η(1) = ln 2
        let eta1 = dirichlet_eta(1.0).expect("eta(1)");
        assert_relative_eq!(eta1, LN_2, epsilon = 1e-10);

        // η(0) = 1/2
        let eta0 = dirichlet_eta(0.0).expect("eta(0)");
        assert_relative_eq!(eta0, 0.5, epsilon = 1e-10);

        // η(2) = π²/12
        let eta2 = dirichlet_eta(2.0).expect("eta(2)");
        assert_relative_eq!(eta2, PI * PI / 12.0, epsilon = 1e-8);
    }

    #[test]
    fn test_dirichlet_eta_relation_to_zeta() {
        // η(s) = (1 - 2^{1-s}) ζ(s)
        for s in [2.0_f64, 3.0, 4.0, 5.0] {
            let eta_val = dirichlet_eta(s).expect("eta");
            let zeta_val = riemann_zeta(s).expect("zeta");
            let expected = (1.0 - 2.0_f64.powf(1.0 - s)) * zeta_val;
            let diff = (eta_val - expected).abs();
            assert!(
                diff < 1e-8,
                "eta({s}) = {eta_val}, expected {expected}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_lerch_phi_reduces_to_zeta() {
        // Φ(1, s, 1) = ζ(s)
        let phi = lerch_phi(1.0, 2.0, 1.0).expect("Phi(1,2,1)");
        assert_relative_eq!(phi, PI * PI / 6.0, epsilon = 1e-6);
    }

    #[test]
    fn test_lerch_phi_reduces_to_eta() {
        // Φ(-1, 1, 1) = ln 2
        let phi = lerch_phi(-1.0, 1.0, 1.0).expect("Phi(-1,1,1)");
        assert_relative_eq!(phi, LN_2, epsilon = 1e-8);
    }

    #[test]
    fn test_lerch_phi_geometric() {
        // Φ(z, 0, a) = ∑_{n=0}^∞ z^n = 1/(1-z) for |z| < 1
        let z = 0.5_f64;
        let phi = lerch_phi(z, 0.0, 1.0).expect("Phi(0.5,0,1)");
        assert_relative_eq!(phi, 1.0 / (1.0 - z), epsilon = 1e-6);
    }

    #[test]
    fn test_polygamma_wrapper() {
        // ψ^{(0)}(1) = -γ
        let euler_gamma = 0.577_215_664_901_532_9_f64;
        let psi0 = polygamma_local(0, 1.0_f64);
        assert_relative_eq!(psi0, -euler_gamma, epsilon = 1e-8);

        // ψ^{(1)}(1) = π²/6
        let psi1 = polygamma_local(1, 1.0_f64);
        assert_relative_eq!(psi1, PI * PI / 6.0, epsilon = 1e-8);
    }
}
