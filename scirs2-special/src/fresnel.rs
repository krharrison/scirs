//! Fresnel integrals
//!
//! This module provides implementations of the Fresnel integrals S(x) and C(x).
//! These integrals arise in optics and electromagnetics, particularly in the
//! study of diffraction patterns and are defined as:
//!
//! S(x) = ∫_0_^x sin(πt²/2) dt
//! C(x) = ∫_0_^x cos(πt²/2) dt
//!
//! There are also modified Fresnel integrals that are used in certain applications.

use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::Zero;
use std::f64::consts::PI;

use crate::error::{SpecialError, SpecialResult};

/// Compute the Fresnel sine and cosine integrals.
///
/// # Definition
///
/// The Fresnel integrals are defined as:
///
/// S(x) = ∫_0_^x sin(πt²/2) dt
/// C(x) = ∫_0_^x cos(πt²/2) dt
///
/// # Arguments
///
/// * `x` - Real or complex argument
///
/// # Returns
///
/// * A tuple (S(x), C(x)) containing the values of the Fresnel sine and cosine integrals
///
/// # Examples
///
/// ```
/// use scirs2_special::fresnel;
///
/// let (s, c) = fresnel(1.0).expect("fresnel failed");
/// println!("S(1.0) = {}, C(1.0) = {}", s, c);
/// ```
#[allow(dead_code)]
pub fn fresnel(x: f64) -> SpecialResult<(f64, f64)> {
    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to fresnel".to_string(),
        ));
    }

    if x == 0.0 {
        return Ok((0.0, 0.0));
    }

    // For x with large magnitude, use the asymptotic form
    if x.abs() > 6.0 {
        let (s, c) = fresnel_asymptotic(x)?;
        return Ok((s, c));
    }

    // For small to moderate x, use power series or auxiliary functions
    fresnel_power_series(x)
}

/// Compute the Fresnel sine and cosine integrals for complex argument.
///
/// # Arguments
///
/// * `z` - Complex argument
///
/// # Returns
///
/// * A tuple (S(z), C(z)) containing the complex values of the Fresnel sine and cosine integrals
///
/// # Examples
///
/// ```
/// use scirs2_special::fresnel_complex;
/// use scirs2_core::numeric::Complex64;
///
/// let z = Complex64::new(1.0, 0.5);
/// let (s, c) = fresnel_complex(z).expect("fresnel_complex failed");
/// println!("S({} + {}i) = {} + {}i", z.re, z.im, s.re, s.im);
/// println!("C({} + {}i) = {} + {}i", z.re, z.im, c.re, c.im);
/// ```
#[allow(dead_code)]
pub fn fresnel_complex(z: Complex64) -> SpecialResult<(Complex64, Complex64)> {
    if z.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to fresnel_complex".to_string(),
        ));
    }

    if z.is_zero() {
        return Ok((Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)));
    }

    // For z with large magnitude, use the asymptotic form
    if z.norm() > 6.0 {
        let (s, c) = fresnel_complex_asymptotic(z)?;
        return Ok((s, c));
    }

    // For small to moderate z, use power series
    fresnel_complex_power_series(z)
}

/// Implementation of Fresnel integrals using power series.
///
/// Uses the standard Taylor series:
///   S(x) = sum_{n=0}^inf (-1)^n * (pi/2)^(2n+1) * x^(4n+3) / ((2n+1)! * (4n+3))
///   C(x) = sum_{n=0}^inf (-1)^n * (pi/2)^(2n) * x^(4n+1) / ((2n)! * (4n+1))
#[allow(dead_code)]
fn fresnel_power_series(x: f64) -> SpecialResult<(f64, f64)> {
    let sign = x.signum();
    let ax = x.abs();

    // Special case for very small x to avoid underflow
    if ax < 1e-100 {
        return Ok((0.0, 0.0));
    }

    let pi_half = PI / 2.0;
    let t = pi_half * ax * ax;

    // For the series, compute in terms of t = pi*x^2/2
    // S(x) = x * sum_{n=0}^inf (-t)^n * t / ((2n+1)*(4n+3))  -- not quite, use direct approach
    //
    // Direct series:
    //   S(x) = sum_{n=0}^inf (-1)^n (pi/2)^{2n+1} x^{4n+3} / ((2n+1)! (4n+3))
    //   C(x) = sum_{n=0}^inf (-1)^n (pi/2)^{2n} x^{4n+1} / ((2n)! (4n+1))
    //
    // Rearranged for stable computation:
    //   S(x) = x^3 * pi/2 * sum_{n=0} (-1)^n (pi*x^2/2)^{2n} / ((2n+1)! (4n+3))
    //   C(x) = x * sum_{n=0} (-1)^n (pi*x^2/2)^{2n} / ((2n)! (4n+1))

    // Use the t-based formulation for stability
    let t2 = t * t; // (pi*x^2/2)^2

    // Compute C(x)
    let mut c_sum = 0.0;
    let mut c_term = 1.0; // n=0 term coefficient: 1 / (0! * 1) = 1
    for n in 0..50 {
        let denom = (4 * n + 1) as f64;
        c_sum += c_term / denom;

        // Prepare for next term: multiply by -t^2 / ((2n+1)*(2n+2))
        let next_factor = -t2 / (((2 * n + 1) * (2 * n + 2)) as f64);
        c_term *= next_factor;

        if c_term.abs() / denom < 1e-16 * c_sum.abs().max(1e-300) {
            break;
        }
    }
    let c = ax * c_sum;

    // Compute S(x)
    let mut s_sum = 0.0;
    let mut s_term = 1.0; // n=0 term coefficient: 1 / (1! * 3) but we factor out t*x below
    for n in 0..50 {
        let denom = (4 * n + 3) as f64;
        s_sum += s_term / denom;

        // Prepare for next term: multiply by -t^2 / ((2n+2)*(2n+3))
        let next_factor = -t2 / (((2 * n + 2) * (2 * n + 3)) as f64);
        s_term *= next_factor;

        if s_term.abs() / denom < 1e-16 * s_sum.abs().max(1e-300) {
            break;
        }
    }
    let s = ax * t * s_sum;

    Ok((sign * s, sign * c))
}

/// Implementation of Fresnel integrals using asymptotic expansions for large x.
#[allow(dead_code)]
fn fresnel_asymptotic(x: f64) -> SpecialResult<(f64, f64)> {
    let sign = x.signum();
    let x = x.abs();

    // Special case for extremely large x
    if x > 1e100 {
        // For extremely large x, the Fresnel integrals approach 1/2
        return Ok((sign * 0.5, sign * 0.5));
    }

    // For large x, the Fresnel integrals approach 1/2
    // S(x) → 1/2 - f(x)cos(πx²/2) - g(x)sin(πx²/2)
    // C(x) → 1/2 + f(x)sin(πx²/2) - g(x)cos(πx²/2)
    // where f(x) and g(x) are asymptotic series

    // Use a scaled approach for very large x to avoid overflow in x²
    let z = if x > 1e7 {
        // For very large x, compute z carefully to avoid overflow
        let scaled_x = x / 1e7;
        PI * scaled_x * scaled_x * 1e14 / 2.0
    } else {
        PI * x * x / 2.0
    };

    // The argument may be so large that z cannot be represented accurately
    // In that case, simplify the computation by modding out the periods of sine and cosine
    let reduced_z = if z > 1e10 {
        // For extremely large z, reduce to principal values
        let two_pi = 2.0 * PI;
        z % two_pi
    } else {
        z
    };

    // Compute sine and cosine of the reduced argument
    let sin_z = reduced_z.sin();
    let cos_z = reduced_z.cos();

    // Different strategies based on magnitude of x for stability
    if x > 20.0 {
        // For very large x, use a more accurate asymptotic form
        // that avoids potential cancellation errors

        // Compute just the first few terms of the asymptotic series
        // This avoids divergence issues with asymptotic series for large orders
        let f_first_term = 1.0 / (PI * x);
        let g_first_term = 1.0 / (PI * 3.0 * 2.0 * z); // First term of g series

        // For extremely large x, the higher-order terms are negligible
        let s = 0.5 - f_first_term * cos_z - g_first_term * sin_z;
        let c = 0.5 + f_first_term * sin_z - g_first_term * cos_z;

        return Ok((sign * s, sign * c));
    }

    // For moderately large x, compute more terms of the series
    let z2 = z * z;
    let z2_inv = 1.0 / z2;

    // Initialize with leading terms
    let mut f = 1.0 / (PI * x);
    let mut g = 0.0;

    // Keep track of previous sums for convergence monitoring
    let mut prev_f = f;
    let mut prev_g = g;
    let mut num_stable_terms = 0;

    // Asymptotic series for f(x) and g(x) with enhanced stability
    for k in 1..25 {
        // Extended series for better accuracy
        // Compute terms carefully to avoid overflow in large powers
        let k_f64 = k as f64;

        // Avoid direct power calculation which could overflow
        // Instead, build up the power by multiplication
        let mut z2_pow_k: f64 = z2_inv; // Start with (1/z2)
        for _ in 1..k {
            z2_pow_k *= z2_inv; // Multiply by (1/z2) k-1 more times

            // Check for underflow
            if z2_pow_k.abs() < 1e-300 {
                break; // Underflow, further terms are negligible
            }
        }

        // Calculate f and g terms with improved numerical stability
        let f_term =
            if k % 2 == 1 { -1.0 } else { 1.0 } * (4.0 * k_f64 - 1.0) * z2_pow_k / (PI * x);

        let g_term = if k % 2 == 1 { -1.0 } else { 1.0 } * (4.0 * k_f64 + 1.0) * z2_pow_k
            / ((2.0 * k_f64 + 1.0) * PI);

        // Add terms to the sums
        f += f_term;
        g += g_term;

        // Multiple convergence criteria for better stability

        // Absolute tolerance
        let abs_tol = 1e-15;

        // Relative tolerance
        let f_rel_tol = 1e-15 * f.abs().max(1e-300);
        let g_rel_tol = 1e-15 * g.abs().max(1e-300);

        // Check for convergence
        if f_term.abs() < abs_tol && g_term.abs() < abs_tol {
            break; // Terms are absolutely small
        }

        if f_term.abs() < f_rel_tol && g_term.abs() < g_rel_tol {
            break; // Terms are relatively small
        }

        // Check if sums are stabilizing (not changing significantly)
        if (f - prev_f).abs() < f_rel_tol && (g - prev_g).abs() < g_rel_tol {
            num_stable_terms += 1;
            if num_stable_terms > 2 {
                break; // Sums have stabilized
            }
        } else {
            num_stable_terms = 0;
        }

        // Check for divergence (which can happen with asymptotic series)
        if f_term.abs() > 100.0 * prev_f.abs() || g_term.abs() > 100.0 * prev_g.abs() {
            // Series is starting to diverge, so use the previous sum
            f = prev_f;
            g = prev_g;
            break;
        }

        prev_f = f;
        prev_g = g;
    }

    // Compute the Fresnel integrals
    let s = 0.5 - f * cos_z - g * sin_z;
    let c = 0.5 + f * sin_z - g * cos_z;

    // Apply the sign
    Ok((sign * s, sign * c))
}

/// Implementation of complex Fresnel integrals using power series.
#[allow(dead_code)]
fn fresnel_complex_power_series(z: Complex64) -> SpecialResult<(Complex64, Complex64)> {
    // Special case for very small |z|
    if z.norm() < 1e-100 {
        return Ok((Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)));
    }

    // Use the same approach as the real-valued fresnel_power_series but with complex z.
    // The Fresnel integrals are:
    //   C(z) = integral_0^z cos(pi*t^2/2) dt
    //   S(z) = integral_0^z sin(pi*t^2/2) dt
    //
    // Taylor series with t = pi*z^2/2:
    //   C(z) = z * sum_{n=0}^inf (-1)^n t^{2n} / ((2n)! * (4n+1))
    //   S(z) = z * t * sum_{n=0}^inf (-1)^n t^{2n} / ((2n+1)! * (4n+3))
    //
    // Using recurrence for the factorial-weighted terms.

    let pi_half = Complex64::new(PI / 2.0, 0.0);
    let t = pi_half * z * z;
    let t2 = t * t;

    // Compute C(z)
    let mut c_sum = Complex64::new(0.0, 0.0);
    let mut c_term = Complex64::new(1.0, 0.0);
    for n in 0..80 {
        let denom = (4 * n + 1) as f64;
        c_sum += c_term / denom;

        // Next term: multiply by -t^2 / ((2n+1)*(2n+2))
        let next_factor = -t2 / Complex64::new(((2 * n + 1) * (2 * n + 2)) as f64, 0.0);
        c_term *= next_factor;

        if !c_term.is_finite() {
            break;
        }
        if c_term.norm() / denom < 1e-16 * c_sum.norm().max(1e-300) && n > 3 {
            break;
        }
    }
    let c = z * c_sum;

    // Compute S(z)
    let mut s_sum = Complex64::new(0.0, 0.0);
    let mut s_term = Complex64::new(1.0, 0.0);
    for n in 0..80 {
        let denom = (4 * n + 3) as f64;
        s_sum += s_term / denom;

        // Next term: multiply by -t^2 / ((2n+2)*(2n+3))
        let next_factor = -t2 / Complex64::new(((2 * n + 2) * (2 * n + 3)) as f64, 0.0);
        s_term *= next_factor;

        if !s_term.is_finite() {
            break;
        }
        if s_term.norm() / denom < 1e-16 * s_sum.norm().max(1e-300) && n > 3 {
            break;
        }
    }
    let s = z * t * s_sum;

    Ok((s, c))
}

/// Implementation of complex Fresnel integrals using asymptotic expansions.
#[allow(dead_code)]
fn fresnel_complex_asymptotic(z: Complex64) -> SpecialResult<(Complex64, Complex64)> {
    // Special cases for extreme values
    if !z.is_finite() {
        return Err(SpecialError::DomainError(
            "Infinite or NaN input to fresnel_complex_asymptotic".to_string(),
        ));
    }

    // For extremely large |z|, directly return the limit
    if z.norm() > 1e100 {
        return Ok((Complex64::new(0.5, 0.0), Complex64::new(0.5, 0.0)));
    }

    // Calculate with appropriate scaling to avoid overflow
    let pi_z2_half = if z.norm() > 1e7 {
        // For very large z, compute carefully to avoid overflow in z²
        let scaled_z = z / 1e7;
        PI * scaled_z * scaled_z * 1e14 / 2.0
    } else {
        PI * z * z / 2.0
    };

    // For very large arguments, reduce trigonometric arguments to principal values
    let reduced_pi_z2_half = if pi_z2_half.norm() > 1e10 {
        let two_pi = Complex64::new(2.0 * PI, 0.0);
        // Complex modulo operation
        let n = (pi_z2_half / two_pi).re.floor();
        pi_z2_half - two_pi * Complex64::new(n, 0.0)
    } else {
        pi_z2_half
    };

    // Compute sine and cosine of the reduced argument
    let sin_pi_z2_half = reduced_pi_z2_half.sin();
    let cos_pi_z2_half = reduced_pi_z2_half.cos();

    // For very large |z|, use simplified asymptotic form
    if z.norm() > 20.0 {
        // Just use the first term of the asymptotic series
        let f_first_term = Complex64::new(1.0, 0.0) / (PI * z);

        // g is numerically smaller than f for large |z|
        let g_first_term = f_first_term / (3.0 * pi_z2_half);

        // Calculate the Fresnel integrals with just these first terms
        let half = Complex64::new(0.5, 0.0);
        let s = half - f_first_term * cos_pi_z2_half - g_first_term * sin_pi_z2_half;
        let c = half + f_first_term * sin_pi_z2_half - g_first_term * cos_pi_z2_half;

        return Ok((s, c));
    }

    // For moderately large |z|, compute more terms of the asymptotic series
    let pi_z2_half_sq = pi_z2_half * pi_z2_half;

    // Initialize with the first terms
    let mut f = Complex64::new(1.0, 0.0) / (PI * z);
    let mut g = Complex64::new(0.0, 0.0);

    // Track previous sums for convergence monitoring
    let mut prev_f = f;
    let mut prev_g = g;
    let mut num_stable_terms = 0;

    // Asymptotic series with enhanced stability
    for k in 1..20 {
        // Use safer term calculation
        let k_f64 = k as f64;

        // Compute powers more carefully using division instead of power
        let mut pi_z2_half_sq_pow_k = Complex64::new(1.0, 0.0);
        for _ in 0..k {
            pi_z2_half_sq_pow_k /= pi_z2_half_sq;

            // Check for underflow/overflow
            if !pi_z2_half_sq_pow_k.is_finite() || pi_z2_half_sq_pow_k.norm() < 1e-300 {
                break;
            }
        }

        // Alternating sign based on k
        let sign = if k % 2 == 1 { -1.0 } else { 1.0 };

        // Calculate f term
        let f_term = sign * (4.0 * k_f64 - 1.0) * pi_z2_half_sq_pow_k / (PI * z);

        // Calculate g term
        let g_term = sign * (4.0 * k_f64 + 1.0) * pi_z2_half_sq_pow_k / ((2.0 * k_f64 + 1.0) * PI);

        // Only add terms if they're finite (to handle potential overflow/underflow)
        if f_term.is_finite() {
            f += f_term;
        }

        if g_term.is_finite() {
            g += g_term;
        }

        // Multiple convergence checks
        let f_norm = f_term.norm();
        let g_norm = g_term.norm();
        let f_sum_norm = f.norm().max(1e-300);
        let g_sum_norm = g.norm().max(1e-300);

        // Check for absolute and relative convergence
        if f_norm < 1e-15 && g_norm < 1e-15 {
            break; // Both terms are absolutely small
        }

        if f_norm < 1e-15 * f_sum_norm && g_norm < 1e-15 * g_sum_norm {
            break; // Both terms are relatively small
        }

        // Check if sums are stabilizing
        if (f - prev_f).norm() < 1e-15 * f_sum_norm && (g - prev_g).norm() < 1e-15 * g_sum_norm {
            num_stable_terms += 1;
            if num_stable_terms > 2 {
                break; // Sums have stabilized
            }
        } else {
            num_stable_terms = 0;
        }

        // Check for potential divergence
        if f_norm > 100.0 * prev_f.norm() || g_norm > 100.0 * prev_g.norm() {
            // Series is starting to diverge, use previous sum
            f = prev_f;
            g = prev_g;
            break;
        }

        prev_f = f;
        prev_g = g;
    }

    // Compute the Fresnel integrals
    let half = Complex64::new(0.5, 0.0);
    let s = half - f * cos_pi_z2_half - g * sin_pi_z2_half;
    let c = half + f * sin_pi_z2_half - g * cos_pi_z2_half;

    // Final check for numerical issues
    if !s.is_finite() || !c.is_finite() {
        // Fallback to the simplest approximation for large |z|
        let s_approx = Complex64::new(0.5, 0.0);
        let c_approx = Complex64::new(0.5, 0.0);
        return Ok((s_approx, c_approx));
    }

    Ok((s, c))
}

/// Compute the Fresnel sine integral.
///
/// # Definition
///
/// The Fresnel sine integral is defined as:
///
/// S(x) = ∫_0_^x sin(πt²/2) dt
///
/// # Arguments
///
/// * `x` - Real argument
///
/// # Returns
///
/// * Value of the Fresnel sine integral S(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::fresnels;
///
/// let s = fresnels(1.0).expect("fresnels failed");
/// println!("S(1.0) = {}", s);
/// ```
#[allow(dead_code)]
pub fn fresnels(x: f64) -> SpecialResult<f64> {
    let (s, _) = fresnel(x)?;
    Ok(s)
}

/// Compute the Fresnel cosine integral.
///
/// # Definition
///
/// The Fresnel cosine integral is defined as:
///
/// C(x) = ∫_0_^x cos(πt²/2) dt
///
/// # Arguments
///
/// * `x` - Real argument
///
/// # Returns
///
/// * Value of the Fresnel cosine integral C(x)
///
/// # Examples
///
/// ```
/// use scirs2_special::fresnelc;
///
/// let c = fresnelc(1.0).expect("fresnelc failed");
/// println!("C(1.0) = {}", c);
/// ```
#[allow(dead_code)]
pub fn fresnelc(x: f64) -> SpecialResult<f64> {
    let (_, c) = fresnel(x)?;
    Ok(c)
}

/// Compute the modified Fresnel plus integrals.
///
/// # Definition
///
/// The modified Fresnel plus integrals are defined as:
///
/// F₊(x) = ∫_x^∞ exp(it²) dt
/// K₊(x) = 1/√π · exp(-i(x² + π/4)) · F₊(x)
///
/// # Arguments
///
/// * `x` - Real argument
///
/// # Returns
///
/// * A tuple (F₊(x), K₊(x)) containing the values of the modified Fresnel plus integrals
///
/// # Examples
///
/// ```
/// use scirs2_special::mod_fresnel_plus;
///
/// let (f_plus, k_plus) = mod_fresnel_plus(1.0).expect("mod_fresnel_plus failed");
/// println!("F₊(1.0) = {} + {}i", f_plus.re, f_plus.im);
/// println!("K₊(1.0) = {} + {}i", k_plus.re, k_plus.im);
/// ```
#[allow(dead_code)]
pub fn mod_fresnel_plus(x: f64) -> SpecialResult<(Complex64, Complex64)> {
    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to mod_fresnel_plus".to_string(),
        ));
    }

    // Special case for extremely small x
    if x.abs() < 1e-100 {
        // For x ≈ 0, F₊(0) approaches √π·e^(iπ/4)/2
        let sqrt_pi = PI.sqrt();
        let exp_i_pi_4 = Complex64::new(1.0, 0.0) * Complex64::new(0.5, 0.5).sqrt();
        let f_plus_0 = sqrt_pi * exp_i_pi_4 / 2.0;

        // K₊(0) ≈ 1/2
        let k_plus_0 = Complex64::new(0.5, 0.0);

        return Ok((f_plus_0, k_plus_0));
    }

    // Special case for extremely large x
    if x.abs() > 1e100 {
        // For |x| → ∞, F₊(x) → 0 and K₊(x) → 0
        let zero = Complex64::new(0.0, 0.0);
        return Ok((zero, zero));
    }

    // The modified Fresnel plus integrals can be expressed in terms of the standard Fresnel integrals
    let z = x.abs();

    // Compute auxiliary values (Fresnel integrals)
    let (s, c) = fresnel(z)?;
    let sqrt_pi = PI.sqrt();
    let sqrt_pi_inv = 1.0 / sqrt_pi;

    // For large z, compute the phase carefully to avoid overflow in z²
    // exp(±i(z² + π/4))
    let phase = if z > 1e7 {
        // Scale to avoid overflow
        let scaled_z = z / 1e7;
        let scaled_z_sq = scaled_z * scaled_z * 1e14;
        Complex64::new(0.0, scaled_z_sq + PI / 4.0)
    } else {
        Complex64::new(0.0, z * z + PI / 4.0)
    };

    // Check for potential overflow in the exponential
    // If the imaginary part of phase is very large, reduce it modulo 2π
    let reduced_phase = if phase.im.abs() > 100.0 {
        let two_pi = 2.0 * PI;
        let n = (phase.im / two_pi).floor();
        Complex64::new(phase.re, phase.im - n * two_pi)
    } else {
        phase
    };

    let exp_phase = reduced_phase.exp();
    let exp_i_pi_4 = Complex64::new(0.5, 0.5).sqrt(); // e^(iπ/4) = (1+i)/√2

    // Compute F₊(x) with improved numerical stability
    let f_plus = if x >= 0.0 {
        // For positive x: F₊(x) = (1/2 - C(x) - iS(x))·√π·exp(iπ/4)

        // Handle potential cancellation in 0.5 - c for large x
        // For large x, both c and s approach 0.5, so we compute the difference directly
        let halfminus_c = if z > 10.0 {
            // For large z, compute the difference using the asymptotic series directly
            let _z_sq = z * z;
            let pi_z = PI * z;
            1.0 / (2.0 * pi_z) * z.cos() // First term of asymptotic expansion
        } else {
            0.5 - c
        };

        // Similarly for s approaching 0.5
        let minus_s = if z > 10.0 {
            let _z_sq = z * z;
            let pi_z = PI * z;
            -0.5 + 1.0 / (2.0 * pi_z) * z.sin() // First term of asymptotic expansion
        } else {
            -s
        };

        let halfminus_cminus_is = Complex64::new(halfminus_c, minus_s);
        halfminus_cminus_is * sqrt_pi * exp_i_pi_4
    } else {
        // For negative x: F₊(-x) = (1/2 + C(x) + iS(x))·√π·exp(iπ/4)
        // Similar improved calculations for -x
        let half_plus_c = if z > 10.0 {
            let _z_sq = z * z;
            let pi_z = PI * z;
            0.5 + 1.0 / (2.0 * pi_z) * z.cos()
        } else {
            0.5 + c
        };

        let plus_s = if z > 10.0 {
            let _z_sq = z * z;
            let pi_z = PI * z;
            0.5 - 1.0 / (2.0 * pi_z) * z.sin()
        } else {
            s
        };

        let half_plus_c_plus_is = Complex64::new(half_plus_c, plus_s);
        half_plus_c_plus_is * sqrt_pi * exp_i_pi_4
    };

    // Compute K₊(x) = exp(-i(x² + π/4)) · F₊(x) / √π with careful multiplication
    // Use intermediate variable to avoid catastrophic cancellation
    let k_plus_unnormalized = exp_phase.conj() * f_plus;
    let k_plus = k_plus_unnormalized * sqrt_pi_inv;

    // Final check for numerical stability
    if !f_plus.is_finite() || !k_plus.is_finite() {
        // Fallback to asymptotic approximations for very large arguments
        if x.abs() > 10.0 {
            // For large |x|, the integrals decay like 1/x
            let decay_factor = 1.0 / x.abs();
            let f_plus_approx = Complex64::new(decay_factor, decay_factor);
            let k_plus_approx = Complex64::new(decay_factor, -decay_factor) * sqrt_pi_inv;
            return Ok((f_plus_approx, k_plus_approx));
        }
    }

    Ok((f_plus, k_plus))
}

/// Compute the modified Fresnel minus integrals.
///
/// # Definition
///
/// The modified Fresnel minus integrals are defined as:
///
/// F₋(x) = ∫_x^∞ exp(-it²) dt
/// K₋(x) = 1/√π · exp(i(x² + π/4)) · F₋(x)
///
/// # Arguments
///
/// * `x` - Real argument
///
/// # Returns
///
/// * A tuple (F₋(x), K₋(x)) containing the values of the modified Fresnel minus integrals
///
/// # Examples
///
/// ```
/// use scirs2_special::mod_fresnelminus;
///
/// let (fminus, kminus) = mod_fresnelminus(1.0).expect("mod_fresnelminus failed");
/// println!("F₋(1.0) = {} + {}i", fminus.re, fminus.im);
/// println!("K₋(1.0) = {} + {}i", kminus.re, kminus.im);
/// ```
#[allow(dead_code)]
pub fn mod_fresnelminus(x: f64) -> SpecialResult<(Complex64, Complex64)> {
    if x.is_nan() {
        return Err(SpecialError::DomainError(
            "NaN input to mod_fresnelminus".to_string(),
        ));
    }

    // Special case for extremely small x
    if x.abs() < 1e-100 {
        // For x ≈ 0, F₋(0) approaches √π·e^(-iπ/4)/2
        let sqrt_pi = PI.sqrt();
        let expminus_i_pi_4 = Complex64::new(1.0, 0.0) * Complex64::new(0.5, -0.5).sqrt();
        let fminus_0 = sqrt_pi * expminus_i_pi_4 / 2.0;

        // K₋(0) ≈ 1/2
        let kminus_0 = Complex64::new(0.5, 0.0);

        return Ok((fminus_0, kminus_0));
    }

    // Special case for extremely large x
    if x.abs() > 1e100 {
        // For |x| → ∞, F₋(x) → 0 and K₋(x) → 0
        let zero = Complex64::new(0.0, 0.0);
        return Ok((zero, zero));
    }

    // The modified Fresnel minus integrals can be expressed in terms of the standard Fresnel integrals
    let z = x.abs();

    // Compute auxiliary values (Fresnel integrals)
    let (s, c) = fresnel(z)?;
    let sqrt_pi = PI.sqrt();
    let sqrt_pi_inv = 1.0 / sqrt_pi;

    // For large z, compute the phase carefully to avoid overflow in z²
    // exp(±i(z² + π/4))
    let phase = if z > 1e7 {
        // Scale to avoid overflow
        let scaled_z = z / 1e7;
        let scaled_z_sq = scaled_z * scaled_z * 1e14;
        Complex64::new(0.0, scaled_z_sq + PI / 4.0)
    } else {
        Complex64::new(0.0, z * z + PI / 4.0)
    };

    // Check for potential overflow in the exponential
    // If the imaginary part of phase is very large, reduce it modulo 2π
    let reduced_phase = if phase.im.abs() > 100.0 {
        let two_pi = 2.0 * PI;
        let n = (phase.im / two_pi).floor();
        Complex64::new(phase.re, phase.im - n * two_pi)
    } else {
        phase
    };

    let exp_phase = reduced_phase.exp();
    let expminus_i_pi_4 = Complex64::new(0.5, -0.5).sqrt(); // e^(-iπ/4) = (1-i)/√2

    // Compute F₋(x) with improved numerical stability
    let fminus = if x >= 0.0 {
        // For positive x: F₋(x) = (1/2 - C(x) + iS(x))·√π·exp(-iπ/4)

        // Handle potential cancellation in 0.5 - c for large x
        // For large x, both c and s approach 0.5, so we compute the difference directly
        let halfminus_c = if z > 10.0 {
            // For large z, compute the difference using the asymptotic series directly
            let pi_z = PI * z;
            1.0 / (2.0 * pi_z) * z.cos() // First term of asymptotic expansion
        } else {
            0.5 - c
        };

        // Similarly for s approaching 0.5
        let plus_s = if z > 10.0 {
            let pi_z = PI * z;
            0.5 - 1.0 / (2.0 * pi_z) * z.sin() // First term of asymptotic expansion
        } else {
            s
        };

        let halfminus_c_plus_is = Complex64::new(halfminus_c, plus_s);
        halfminus_c_plus_is * sqrt_pi * expminus_i_pi_4
    } else {
        // For negative x: F₋(-x) = (1/2 + C(x) - iS(x))·√π·exp(-iπ/4)
        // Similar improved calculations for -x
        let half_plus_c = if z > 10.0 {
            let pi_z = PI * z;
            0.5 + 1.0 / (2.0 * pi_z) * z.cos()
        } else {
            0.5 + c
        };

        let minus_s = if z > 10.0 {
            let pi_z = PI * z;
            -0.5 + 1.0 / (2.0 * pi_z) * z.sin()
        } else {
            -s
        };

        let half_plus_cminus_is = Complex64::new(half_plus_c, minus_s);
        half_plus_cminus_is * sqrt_pi * expminus_i_pi_4
    };

    // Compute K₋(x) = exp(i(x² + π/4)) · F₋(x) / √π with careful multiplication
    // Use intermediate variable to avoid catastrophic cancellation
    let kminus_unnormalized = exp_phase * fminus;
    let kminus = kminus_unnormalized * sqrt_pi_inv;

    // Final check for numerical stability
    if !fminus.is_finite() || !kminus.is_finite() {
        // Fallback to asymptotic approximations for very large arguments
        if x.abs() > 10.0 {
            // For large |x|, the integrals decay like 1/x
            let decay_factor = 1.0 / x.abs();
            let fminus_approx = Complex64::new(decay_factor, -decay_factor);
            let kminus_approx = Complex64::new(decay_factor, decay_factor) * sqrt_pi_inv;
            return Ok((fminus_approx, kminus_approx));
        }
    }

    Ok((fminus, kminus))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ====== Fresnel S(x) and C(x) combined tests ======

    #[test]
    fn test_fresnel_at_zero() {
        let (s, c) = fresnel(0.0).expect("fresnel(0) failed");
        assert!((s - 0.0).abs() < 1e-14, "S(0) should be 0, got {s}");
        assert!((c - 0.0).abs() < 1e-14, "C(0) should be 0, got {c}");
    }

    #[test]
    fn test_fresnel_at_one() {
        // Reference: S(1) ~ 0.438259147, C(1) ~ 0.779893400
        let (s, c) = fresnel(1.0).expect("fresnel(1) failed");
        assert!((s - 0.438_259_147).abs() < 1e-6, "S(1) ~ 0.438259, got {s}");
        assert!((c - 0.779_893_400).abs() < 1e-6, "C(1) ~ 0.779893, got {c}");
    }

    #[test]
    fn test_fresnel_at_two() {
        // Reference: S(2) ~ 0.343415, C(2) ~ 0.488253
        let (s, c) = fresnel(2.0).expect("fresnel(2) failed");
        assert!((s - 0.343_415).abs() < 1e-4, "S(2) ~ 0.3434, got {s}");
        assert!((c - 0.488_253).abs() < 1e-4, "C(2) ~ 0.4883, got {c}");
    }

    #[test]
    fn test_fresnel_large_x_approaches_half() {
        // For large x, S(x) and C(x) should approach 0.5
        let (s, c) = fresnel(50.0).expect("fresnel(50) failed");
        assert!((s - 0.5).abs() < 0.05, "S(50) should be near 0.5, got {s}");
        assert!((c - 0.5).abs() < 0.05, "C(50) should be near 0.5, got {c}");
    }

    #[test]
    fn test_fresnel_odd_symmetry() {
        // S(-x) = -S(x), C(-x) = -C(x) (odd functions)
        let (s_pos, c_pos) = fresnel(1.5).expect("fresnel(1.5) failed");
        let (s_neg, c_neg) = fresnel(-1.5).expect("fresnel(-1.5) failed");
        assert!(
            (s_neg + s_pos).abs() < 1e-10,
            "S should be odd: S(1.5)={s_pos}, S(-1.5)={s_neg}"
        );
        assert!(
            (c_neg + c_pos).abs() < 1e-10,
            "C should be odd: C(1.5)={c_pos}, C(-1.5)={c_neg}"
        );
    }

    #[test]
    fn test_fresnel_nan_input() {
        let result = fresnel(f64::NAN);
        assert!(
            result.is_err(),
            "fresnel with NaN input should return error"
        );
    }

    // ====== fresnels tests ======

    #[test]
    fn test_fresnels_at_zero() {
        let s = fresnels(0.0).expect("fresnels(0) failed");
        assert!((s - 0.0).abs() < 1e-14, "S(0) should be 0, got {s}");
    }

    #[test]
    fn test_fresnels_at_one() {
        let s = fresnels(1.0).expect("fresnels(1) failed");
        assert!((s - 0.438_259_147).abs() < 1e-6, "S(1) ~ 0.438259, got {s}");
    }

    #[test]
    fn test_fresnels_matches_fresnel() {
        let s1 = fresnels(2.5).expect("fresnels(2.5) failed");
        let (s2, _) = fresnel(2.5).expect("fresnel(2.5) failed");
        assert!(
            (s1 - s2).abs() < 1e-10,
            "fresnels should match fresnel: {s1} vs {s2}"
        );
    }

    #[test]
    fn test_fresnels_nan() {
        let result = fresnels(f64::NAN);
        assert!(
            result.is_err(),
            "fresnels with NaN input should return error"
        );
    }

    #[test]
    fn test_fresnels_negative() {
        let s_pos = fresnels(1.0).expect("fresnels(1) failed");
        let s_neg = fresnels(-1.0).expect("fresnels(-1) failed");
        assert!((s_neg + s_pos).abs() < 1e-10, "S should be odd function");
    }

    // ====== fresnelc tests ======

    #[test]
    fn test_fresnelc_at_zero() {
        let c = fresnelc(0.0).expect("fresnelc(0) failed");
        assert!((c - 0.0).abs() < 1e-14, "C(0) should be 0, got {c}");
    }

    #[test]
    fn test_fresnelc_at_one() {
        let c = fresnelc(1.0).expect("fresnelc(1) failed");
        assert!((c - 0.779_893_400).abs() < 1e-6, "C(1) ~ 0.779893, got {c}");
    }

    #[test]
    fn test_fresnelc_matches_fresnel() {
        let c1 = fresnelc(2.5).expect("fresnelc(2.5) failed");
        let (_, c2) = fresnel(2.5).expect("fresnel(2.5) failed");
        assert!(
            (c1 - c2).abs() < 1e-10,
            "fresnelc should match fresnel: {c1} vs {c2}"
        );
    }

    #[test]
    fn test_fresnelc_nan() {
        let result = fresnelc(f64::NAN);
        assert!(
            result.is_err(),
            "fresnelc with NaN input should return error"
        );
    }

    #[test]
    fn test_fresnelc_negative() {
        let c_pos = fresnelc(1.0).expect("fresnelc(1) failed");
        let c_neg = fresnelc(-1.0).expect("fresnelc(-1) failed");
        assert!((c_neg + c_pos).abs() < 1e-10, "C should be odd function");
    }

    // ====== fresnel_complex tests ======

    #[test]
    fn test_fresnel_complex_real_axis() {
        // On the real axis, fresnel_complex should match fresnel
        let z = Complex64::new(1.0, 0.0);
        let (s_c, c_c) = fresnel_complex(z).expect("fresnel_complex failed");
        let (s_r, c_r) = fresnel(1.0).expect("fresnel failed");
        assert!(
            (s_c.re - s_r).abs() < 1e-8,
            "complex fresnel S on real axis should match: {s_c} vs {s_r}"
        );
        assert!(
            s_c.im.abs() < 1e-8,
            "imaginary part of S on real axis should be ~0"
        );
        assert!(
            (c_c.re - c_r).abs() < 1e-8,
            "complex fresnel C on real axis should match: {c_c} vs {c_r}"
        );
    }

    #[test]
    fn test_fresnel_complex_at_zero() {
        let z = Complex64::new(0.0, 0.0);
        let (s, c) = fresnel_complex(z).expect("fresnel_complex(0) failed");
        assert!(s.norm() < 1e-14, "S(0) should be 0");
        assert!(c.norm() < 1e-14, "C(0) should be 0");
    }

    #[test]
    fn test_fresnel_complex_purely_imaginary() {
        let z = Complex64::new(0.0, 1.0);
        let (s, c) = fresnel_complex(z).expect("fresnel_complex(i) failed");
        assert!(s.is_finite(), "S(i) should be finite");
        assert!(c.is_finite(), "C(i) should be finite");
    }

    #[test]
    fn test_fresnel_complex_nan() {
        let z = Complex64::new(f64::NAN, 0.0);
        let result = fresnel_complex(z);
        assert!(
            result.is_err(),
            "fresnel_complex with NaN should return error"
        );
    }

    #[test]
    fn test_fresnel_complex_moderate() {
        let z = Complex64::new(1.0, 0.5);
        let (s, c) = fresnel_complex(z).expect("fresnel_complex(1+0.5i) failed");
        assert!(s.is_finite(), "S(1+0.5i) should be finite");
        assert!(c.is_finite(), "C(1+0.5i) should be finite");
    }

    // ====== mod_fresnel_plus tests ======

    #[test]
    fn test_mod_fresnel_plus_at_zero() {
        let (f_plus, k_plus) = mod_fresnel_plus(0.0).expect("mod_fresnel_plus(0) failed");
        assert!(f_plus.is_finite(), "F+(0) should be finite");
        assert!(k_plus.is_finite(), "K+(0) should be finite");
    }

    #[test]
    fn test_mod_fresnel_plus_at_one() {
        let (f_plus, k_plus) = mod_fresnel_plus(1.0).expect("mod_fresnel_plus(1) failed");
        assert!(f_plus.is_finite(), "F+(1) should be finite");
        assert!(k_plus.is_finite(), "K+(1) should be finite");
    }

    #[test]
    fn test_mod_fresnel_plus_moderate_x() {
        let (f_plus, k_plus) = mod_fresnel_plus(5.0).expect("mod_fresnel_plus(5) failed");
        assert!(f_plus.is_finite(), "F+(5) should be finite");
        assert!(k_plus.is_finite(), "K+(5) should be finite");
    }

    #[test]
    fn test_mod_fresnel_plus_negative() {
        let (f_plus, k_plus) = mod_fresnel_plus(-1.0).expect("mod_fresnel_plus(-1) failed");
        assert!(f_plus.is_finite(), "F+(-1) should be finite");
        assert!(k_plus.is_finite(), "K+(-1) should be finite");
    }

    #[test]
    fn test_mod_fresnel_plus_large_x() {
        let (f_plus, k_plus) = mod_fresnel_plus(20.0).expect("mod_fresnel_plus(20) failed");
        assert!(f_plus.is_finite(), "F+(20) should be finite");
        assert!(k_plus.is_finite(), "K+(20) should be finite");
    }

    // ====== mod_fresnelminus tests ======

    #[test]
    fn test_mod_fresnelminus_at_zero() {
        let (fminus, kminus) = mod_fresnelminus(0.0).expect("mod_fresnelminus(0) failed");
        assert!(fminus.is_finite(), "F-(0) should be finite");
        assert!(kminus.is_finite(), "K-(0) should be finite");
    }

    #[test]
    fn test_mod_fresnelminus_at_one() {
        let (fminus, kminus) = mod_fresnelminus(1.0).expect("mod_fresnelminus(1) failed");
        assert!(fminus.is_finite(), "F-(1) should be finite");
        assert!(kminus.is_finite(), "K-(1) should be finite");
    }

    #[test]
    fn test_mod_fresnelminus_moderate_x() {
        let (fminus, kminus) = mod_fresnelminus(5.0).expect("mod_fresnelminus(5) failed");
        assert!(fminus.is_finite(), "F-(5) should be finite");
        assert!(kminus.is_finite(), "K-(5) should be finite");
    }

    #[test]
    fn test_mod_fresnelminus_negative() {
        let (fminus, kminus) = mod_fresnelminus(-1.0).expect("mod_fresnelminus(-1) failed");
        assert!(fminus.is_finite(), "F-(-1) should be finite");
        assert!(kminus.is_finite(), "K-(-1) should be finite");
    }

    #[test]
    fn test_mod_fresnelminus_large_x() {
        let (fminus, kminus) = mod_fresnelminus(20.0).expect("mod_fresnelminus(20) failed");
        assert!(fminus.is_finite(), "F-(20) should be finite");
        assert!(kminus.is_finite(), "K-(20) should be finite");
    }
}
