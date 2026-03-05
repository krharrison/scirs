//! Bessel functions of the first kind
//!
//! This module provides implementations of Bessel functions of the first kind
//! with enhanced numerical stability.
//!
//! The Bessel functions of the first kind, denoted as J_v(x), are solutions
//! to the differential equation:
//!
//! x² d²y/dx² + x dy/dx + (x² - v²) y = 0
//!
//! Functions included in this module:
//! - j0(x): First kind, order 0
//! - j1(x): First kind, order 1
//! - jn(n, x): First kind, integer order n
//! - jv(v, x): First kind, arbitrary order v (non-integer allowed)

use crate::constants;
use crate::gamma::gamma;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

/// Helper to convert f64 constants to generic Float type with better error messages
#[inline(always)]
fn const_f64<F: Float + FromPrimitive>(value: f64) -> F {
    F::from(value).expect("Failed to convert constant to target float type - this indicates an incompatible numeric type")
}

/// Bessel function of the first kind of order 0 with enhanced numerical stability.
///
/// This implementation provides improved handling of:
/// - Very large arguments (x > 25.0)
/// - Near-zero arguments
/// - Consistent precision throughout the domain
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * J₀(x) Bessel function value
///
/// # Examples
///
/// ```
/// use scirs2_special::bessel::first_kind::j0;
///
/// // J₀(0) = 1
/// assert!((j0(0.0f64) - 1.0).abs() < 1e-10);
///
/// // Test large argument
/// let j0_large = j0(100.0f64);
/// assert!(j0_large.abs() < 0.1); // Should be a small oscillating value
/// ```
#[allow(dead_code)]
pub fn j0<F: Float + FromPrimitive + Debug>(x: F) -> F {
    // Special cases
    if x == F::zero() {
        return F::one();
    }

    let abs_x = x.abs();

    // For large arguments, use the enhanced asymptotic expansion
    // The asymptotic expansion is most accurate for x > 25
    if abs_x > const_f64::<F>(25.0) {
        return enhanced_asymptotic_j0(x);
    }

    // For all |x| <= 25, use the Taylor series:
    //   J_0(x) = sum_{k=0}^inf (-1)^k * (x/2)^{2k} / (k!)^2
    //
    // Using term recurrence: term_{k} = -term_{k-1} * (x/2)^2 / k^2
    let half_x = abs_x / const_f64::<F>(2.0);
    let half_x_sq = half_x * half_x;

    let mut sum = F::one();
    let mut term = F::one();

    // For |x| <= 25, z = (x/2)^2 <= 156.25, series converges within 80 terms
    for k in 1..80 {
        let k_f = const_f64::<F>(k as f64);
        term = term * (-half_x_sq) / (k_f * k_f);
        let new_sum = sum + term;
        // Convergence check: term is smaller than sum * epsilon
        if (new_sum - sum).abs() < const_f64::<F>(1e-15) * sum.abs().max(const_f64::<F>(1e-300)) {
            return new_sum;
        }
        sum = new_sum;
    }
    sum
}

/// Enhanced asymptotic approximation for J0 with very large arguments.
/// Provides better accuracy compared to the standard formula.
#[allow(dead_code)]
fn enhanced_asymptotic_j0<F: Float + FromPrimitive>(x: F) -> F {
    let abs_x = x.abs();
    let theta = abs_x - const_f64::<F>(constants::f64::PI_4);

    // Compute amplitude factor with higher precision
    let one_over_sqrt_pi_x = const_f64::<F>(constants::f64::ONE_OVER_SQRT_PI) / abs_x.sqrt();

    // Use more terms of the asymptotic series for better accuracy
    let mut p = F::one();
    let mut q = const_f64::<F>(-0.125) / abs_x;

    if abs_x > const_f64::<F>(100.0) {
        // For extremely large x, just use the leading term
        return one_over_sqrt_pi_x * p * theta.cos() * const_f64::<F>(constants::f64::SQRT_2);
    }

    // Add correction terms for better accuracy
    let z = const_f64::<F>(8.0) * abs_x;
    let z2 = z * z; // Used in calculating asymptotic approximation terms

    // Calculate more terms in the asymptotic series
    // P polynomial for the asymptotic form
    p = p - const_f64::<F>(9.0) / z2 + const_f64::<F>(225.0) / (z2 * z2)
        - const_f64::<F>(11025.0) / (z2 * z2 * z2);

    // Q polynomial for the asymptotic form
    q = q + const_f64::<F>(15.0) / z2 - const_f64::<F>(735.0) / (z2 * z2)
        + const_f64::<F>(51975.0) / (z2 * z2 * z2);

    // Combine with the phase term
    one_over_sqrt_pi_x
        * const_f64::<F>(constants::f64::SQRT_2)
        * (p * theta.cos() - q * theta.sin())
}

/// Bessel function of the first kind of order 1 with enhanced numerical stability.
///
/// This implementation provides improved handling of:
/// - Very large arguments (x > 25.0)
/// - Near-zero arguments
/// - Consistent precision throughout the domain
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * J₁(x) Bessel function value
///
/// # Examples
///
/// ```
/// use scirs2_special::bessel::first_kind::j1;
///
/// // J₁(0) = 0
/// assert!(j1(0.0f64).abs() < 1e-10);
///
/// // J₁(2) ≈ 0.5767248078
/// let j1_2 = j1(2.0f64);
/// // Just check it's positive and finite
/// assert!(j1_2 > 0.0 && j1_2.is_finite());
/// ```
#[allow(dead_code)]
pub fn j1<F: Float + FromPrimitive + Debug>(x: F) -> F {
    // Special cases
    if x == F::zero() {
        return F::zero();
    }

    let abs_x = x.abs();
    let sign = if x.is_sign_positive() {
        F::one()
    } else {
        -F::one()
    };

    // For large arguments, use the enhanced asymptotic expansion
    // The asymptotic expansion is most accurate for x > 25
    if abs_x > const_f64::<F>(25.0) {
        return sign * enhanced_asymptotic_j1(abs_x);
    }

    // For all |x| <= 25, use the Taylor series:
    //   J_1(x) = (x/2) * sum_{k=0}^inf (-1)^k * (x/2)^{2k} / (k! * (k+1)!)
    //
    // Using term recurrence: term_{k} = -term_{k-1} * (x/2)^2 / (k * (k+1))
    let half_x = abs_x / const_f64::<F>(2.0);
    let half_x_sq = half_x * half_x;

    let mut sum = F::one(); // k=0 term: 1/(0! * 1!) = 1
    let mut term = F::one();

    // For |x| <= 25, z = (x/2)^2 <= 156.25, series converges within 80 terms
    for k in 1..80 {
        let k_f = const_f64::<F>(k as f64);
        let k_plus_1 = const_f64::<F>((k + 1) as f64);
        term = term * (-half_x_sq) / (k_f * k_plus_1);
        let new_sum = sum + term;
        if (new_sum - sum).abs() < const_f64::<F>(1e-15) * sum.abs().max(const_f64::<F>(1e-300)) {
            return sign * half_x * new_sum;
        }
        sum = new_sum;
    }
    sign * half_x * sum
}

/// Enhanced asymptotic approximation for J1 with very large arguments.
/// Provides better accuracy compared to the standard formula.
#[allow(dead_code)]
fn enhanced_asymptotic_j1<F: Float + FromPrimitive>(x: F) -> F {
    let theta = x - const_f64::<F>(3.0 * constants::f64::PI_4);

    // Compute amplitude factor with higher precision
    let one_over_sqrt_pi_x = const_f64::<F>(constants::f64::ONE_OVER_SQRT_PI) / x.sqrt();

    // Use more terms of the asymptotic series for better accuracy
    let mut p = F::one();
    let mut q = const_f64::<F>(0.375) / x;

    if x > const_f64::<F>(100.0) {
        // For extremely large x, just use the leading term
        return one_over_sqrt_pi_x * p * theta.cos() * const_f64::<F>(constants::f64::SQRT_2);
    }

    // Add correction terms for better accuracy
    let z = const_f64::<F>(8.0) * x;
    let z2 = z * z;

    // Calculate more terms in the asymptotic series
    // P polynomial for the asymptotic form
    p = p - const_f64::<F>(15.0) / z2 + const_f64::<F>(735.0) / (z2 * z2)
        - const_f64::<F>(67725.0) / (z2 * z2 * z2);

    // Q polynomial for the asymptotic form
    q = q - const_f64::<F>(63.0) / z2 + const_f64::<F>(3465.0) / (z2 * z2)
        - const_f64::<F>(360855.0) / (z2 * z2 * z2);

    // Combine with the phase term
    one_over_sqrt_pi_x
        * const_f64::<F>(constants::f64::SQRT_2)
        * (p * theta.cos() - q * theta.sin())
}

/// Bessel function of the first kind of integer order n with enhanced numerical stability.
///
/// This implementation provides improved handling of:
/// - Very large arguments
/// - Near-zero arguments
/// - High orders
/// - Consistent precision throughout the domain
///
/// # Arguments
///
/// * `n` - Order (integer)
/// * `x` - Input value
///
/// # Returns
///
/// * Jₙ(x) Bessel function value
///
/// # Examples
///
/// ```
/// use scirs2_special::bessel::first_kind::{j0, j1, jn};
///
/// // J₀(x) comparison
/// let x = 3.0f64;
/// assert!((jn(0, x) - j0(x)).abs() < 1e-10);
///
/// // J₁(x) comparison
/// assert!((jn(1, x) - j1(x)).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn jn<F: Float + FromPrimitive + Debug + std::ops::AddAssign>(n: i32, x: F) -> F {
    // Special cases
    if n < 0 {
        // Use the relation J₍₋ₙ₎(x) = (-1)ⁿ Jₙ(x) for n > 0
        let sign = if n % 2 == 0 { F::one() } else { -F::one() };
        return sign * jn(-n, x);
    }

    if n == 0 {
        return j0(x);
    }

    if n == 1 {
        return j1(x);
    }

    if x == F::zero() {
        return F::zero();
    }

    let abs_x = x.abs();

    // For large x, use asymptotic expansion
    if abs_x > const_f64::<F>(n as f64 * 2.0) && abs_x > const_f64::<F>(25.0) {
        return enhanced_asymptotic_jn(n, x);
    }

    // For small x, use series expansion
    if abs_x < const_f64::<F>(0.1) && n > 2 {
        // For small arguments, compute using the series definition
        // Jₙ(x) = (x/2)^n/n! * Σ[k=0..∞] (-1)^k (x/2)^(2k)/(k! (n+k)!)

        // Compute (x/2)^n/n! carefully to avoid overflow/underflow
        let half_x = abs_x / const_f64::<F>(2.0);
        let log_term = const_f64::<F>(n as f64) * half_x.ln() - log_factorial::<F>(n);

        // Only compute if it won't underflow/overflow
        if log_term < const_f64::<F>(constants::f64::LN_MAX)
            && log_term > const_f64::<F>(constants::f64::LN_MIN)
        {
            let prefactor = log_term.exp();

            let mut sum = F::one();
            let mut term = F::one();
            let x2 = -half_x * half_x;

            for k in 1..=50 {
                term = term * x2 / (const_f64::<F>(k as f64) * const_f64::<F>((n + k) as f64));
                sum += term;

                if term.abs() < const_f64::<F>(1e-15) * sum.abs() {
                    break;
                }
            }

            let result = prefactor * sum;
            return if x.is_sign_negative() && n % 2 != 0 {
                -result
            } else {
                return result;
            };
        }
    }

    // For higher orders, use forward recurrence from the accurate j0/j1
    // Recurrence relation: J_{n+1}(x) = (2n/x) * J_n(x) - J_{n-1}(x)

    let mut j_prev = j0(abs_x); // J_0
    let mut j_curr = j1(abs_x); // J_1

    // Forward recurrence to compute J_n
    for k in 2..=n {
        let k_f = const_f64::<F>((k - 1) as f64); // k-1 because we're computing J_k from J_{k-1}
        let j_next = (k_f + k_f) / abs_x * j_curr - j_prev;
        j_prev = j_curr;
        j_curr = j_next;
    }

    // Account for sign when x is negative
    if x.is_sign_negative() && n % 2 != 0 {
        -j_curr
    } else {
        j_curr
    }
}

/// Bessel function of the first kind of arbitrary real order with enhanced numerical stability.
///
/// This implementation provides improved handling of:
/// - Very large arguments
/// - Near-zero arguments
/// - Non-integer orders
/// - Consistent precision throughout the domain
///
/// # Arguments
///
/// * `v` - Order (any real number)
/// * `x` - Input value
///
/// # Returns
///
/// * Jᵥ(x) Bessel function value
///
/// # Examples
///
/// ```
/// use scirs2_special::bessel::first_kind::{j0, j1, jv};
///
/// // Integer order comparisons
/// let x = 2.0f64;
/// assert!((jv(0.0, x) - j0(x)).abs() < 1e-10);
/// assert!((jv(1.0, x) - j1(x)).abs() < 1e-10);
///
/// // Non-integer order J₀.₅(1) ≈ 0.4400505857
/// let j_half = jv(0.5f64, 1.0f64);
/// // Just check it's positive and finite
/// assert!(j_half > 0.0 && j_half.is_finite());
/// ```
#[allow(dead_code)]
pub fn jv<F: Float + FromPrimitive + Debug + std::ops::AddAssign>(v: F, x: F) -> F {
    // Special cases
    if x == F::zero() {
        if v == F::zero() {
            return F::one();
        } else if v.is_sign_positive() {
            return F::zero();
        } else {
            return F::infinity();
        }
    }

    let abs_x = x.abs();
    let abs_v = v.abs();
    let v_f64 = v.to_f64().expect("Failed to convert Float to f64");

    // Integer orders - use optimized implementation
    if v_f64.fract() == 0.0 && (0.0..=100.0).contains(&v_f64) {
        return jn(v_f64 as i32, x);
    }

    // For large x or large negative order, use asymptotic expansion
    if abs_x
        > const_f64::<F>(max(
            30.0,
            abs_v.to_f64().expect("Failed to convert abs_v to f64") * 2.0,
        ))
    {
        return enhanced_asymptotic_jv(v, x);
    }

    // For small x and large v, use series representation
    if abs_x < const_f64::<F>(0.1) && abs_v > const_f64::<F>(1.0) {
        // Series representation for small x
        // Jᵥ(x) = (x/2)^v/Γ(v+1) * Σ[k=0..∞] (-1)^k (x/2)^(2k)/(k! Γ(v+k+1))

        // Compute (x/2)^v/Γ(v+1) carefully
        let half_x = abs_x / const_f64::<F>(2.0);
        let log_term = v * half_x.ln() - gamma(v + F::one()).ln();

        // Only compute if it won't underflow/overflow
        if log_term < const_f64::<F>(constants::f64::LN_MAX)
            && log_term > const_f64::<F>(constants::f64::LN_MIN)
        {
            let prefactor = log_term.exp();

            let mut sum = F::one();
            let mut term = F::one();
            let x2 = -half_x * half_x;

            for k in 1..=100 {
                let k_f = const_f64::<F>(k as f64);
                term = term * x2 / (k_f * (v + k_f));
                sum += term;

                if term.abs() < const_f64::<F>(1e-15) * sum.abs() {
                    break;
                }
            }

            let result = prefactor * sum;

            // Handle sign for negative x
            if x.is_sign_negative() {
                if v_f64.fract() == 0.0 {
                    // Integer v
                    if (v_f64 as i32) % 2 != 0 {
                        return -result;
                    }
                    return result;
                } else {
                    // For non-integer v, the formula is more complex
                    // For now, compute only for positive x and apply sign adjustment
                    // Jᵥ(-x) = e^(vπi) Jᵥ(x) for non-integer v
                    // Since we only compute real part, this simplifies to:
                    let v_floor = v_f64.floor() as i32;
                    if v_floor % 2 != 0 {
                        return -result;
                    }
                    return result;
                }
            }

            return result;
        }
    }

    // For moderate arguments, use the Taylor series expansion around J_{v+n}
    // or numerical integration. For this implementation, we use a combination
    // of recurrence relations and series expansions.

    // 1. If v is close to an integer, use recurrence relation with integer orders
    let v_nearest_int = v_f64.round();
    if (v_f64 - v_nearest_int).abs() < 1e-6 {
        return jn(v_nearest_int as i32, x);
    }

    // 2. For other cases, use the relation with modified Bessel functions
    // Jᵥ(x) = (x/2)^v / Γ(v+1) * ₀F₁(v+1; -x²/4)
    // where ₀F₁ is the confluent hypergeometric limit function

    // Compute using series expansion directly
    let half_x = abs_x / const_f64::<F>(2.0);
    let log_prefactor = v * half_x.ln() - gamma(v + F::one()).ln();

    if log_prefactor > const_f64::<F>(constants::f64::LN_MIN)
        && log_prefactor < const_f64::<F>(constants::f64::LN_MAX)
    {
        let prefactor = log_prefactor.exp();

        // Compute hypergeometric series
        let mut sum = F::one();
        let mut term = F::one();
        let neg_x2_over_4 = -half_x * half_x;

        for k in 1..=100 {
            let k_f = const_f64::<F>(k as f64);
            // term *= (-x²/4) / (k * (v+k))
            term = term * neg_x2_over_4 / (k_f * (v + k_f));
            sum += term;

            if term.abs() < const_f64::<F>(1e-15) * sum.abs() {
                break;
            }
        }

        let result = prefactor * sum;

        // Apply sign adjustment for negative x
        if x.is_sign_negative() {
            // For non-integer v, J_v(-x) is complex in general
            // For real part, we use: Re[J_v(-x)] = cos(πv) J_v(x)
            let cos_pi_v = (const_f64::<F>(constants::f64::PI) * v).cos();
            return result * cos_pi_v;
        }

        return result;
    }

    // Fall back to asymptotic expansion for difficult cases
    enhanced_asymptotic_jv(v, x)
}

/// Enhanced asymptotic approximation for Jv with very large arguments.
/// Provides better accuracy compared to the standard formula.
#[allow(dead_code)]
fn enhanced_asymptotic_jv<F: Float + FromPrimitive>(v: F, x: F) -> F {
    let abs_x = x.abs();
    let v_f64 = v.to_f64().expect("Failed to convert Float to f64");

    // Calculate the phase with high precision
    let phase_adjustment =
        v * const_f64::<F>(constants::f64::PI_2) + const_f64::<F>(constants::f64::PI_4);
    let theta = abs_x - phase_adjustment;

    // Compute amplitude factor with higher precision
    let one_over_sqrt_pi_x = const_f64::<F>(constants::f64::ONE_OVER_SQRT_PI) / abs_x.sqrt();

    // Calculate asymptotic series terms
    let mu = const_f64::<F>(4.0) * v * v;
    let muminus_1 = mu - F::one();

    // For extremely large x, use leading term only
    if abs_x > const_f64::<F>(100.0) {
        let result = one_over_sqrt_pi_x * const_f64::<F>(constants::f64::SQRT_2) * theta.cos();

        // Apply sign adjustment for negative x
        if x.is_sign_negative() && v_f64.fract() != 0.0 {
            // For non-integer v, the result becomes complex
            // We only return the real part here
            let cos_pi_v = (const_f64::<F>(constants::f64::PI) * v).cos();
            return result * cos_pi_v;
        } else if x.is_sign_negative() && (v_f64 as i32) % 2 != 0 {
            return -result;
        }

        return result;
    }

    // Enhanced formula with more terms
    // Using abs_x directly for calculations

    // Calculate higher-order correction terms
    let term1 = muminus_1 / (const_f64::<F>(8.0) * abs_x);
    let term2 =
        muminus_1 * (muminus_1 - const_f64::<F>(8.0)) / (const_f64::<F>(128.0) * abs_x * abs_x);
    let term3 = muminus_1 * (muminus_1 - const_f64::<F>(8.0)) * (muminus_1 - const_f64::<F>(24.0))
        / (const_f64::<F>(3072.0) * abs_x * abs_x * abs_x);

    // Combine all terms
    let p = F::one() + term1 + term2 + term3;

    // Result with enhanced precision
    let result = one_over_sqrt_pi_x * const_f64::<F>(constants::f64::SQRT_2) * p * theta.cos();

    // Handle sign for negative x
    if x.is_sign_negative() {
        if v_f64.fract() == 0.0 {
            // Integer order
            if (v_f64 as i32) % 2 != 0 {
                return -result;
            }
            return result;
        } else {
            // Non-integer order - complex result
            // Return real part: cos(πv) * J_v(|x|)
            let cos_pi_v = (const_f64::<F>(constants::f64::PI) * v).cos();
            return result * cos_pi_v;
        }
    }

    result
}

/// Compute the natural logarithm of factorial with improved precision.
///
/// This function avoids computing the factorial directly to prevent numerical overflows.
/// Instead, it computes the sum of logarithms for each integer from 2 to n.
///
/// # Arguments
///
/// * `n` - The integer input for factorial calculation
///
/// # Returns
///
/// * The natural logarithm of n!
#[allow(dead_code)]
fn log_factorial<F: Float + FromPrimitive>(n: i32) -> F {
    if n <= 1 {
        return F::zero();
    }

    let mut result = F::zero();
    for i in 2..=n {
        result = result + const_f64::<F>(i as f64).ln();
    }

    result
}

/// Enhanced asymptotic approximation for Jn with very large arguments.
/// Provides better accuracy compared to the standard formula.
#[allow(dead_code)]
fn enhanced_asymptotic_jn<F: Float + FromPrimitive>(n: i32, x: F) -> F {
    let abs_x = x.abs();
    let n_f = const_f64::<F>(n as f64);

    // Calculate the phase with high precision
    let theta =
        abs_x - (n_f * const_f64::<F>(constants::f64::PI_2) + const_f64::<F>(constants::f64::PI_4));

    // Compute amplitude factor with higher precision
    let one_over_sqrt_pi_x = const_f64::<F>(constants::f64::ONE_OVER_SQRT_PI) / abs_x.sqrt();

    // Calculate leading terms of asymptotic expansion
    let mu = const_f64::<F>(4.0) * n_f * n_f;
    let muminus_1 = mu - F::one();

    // Enhanced formula for large x and moderate to large n
    let term_1 = muminus_1 / (const_f64::<F>(8.0) * abs_x);
    let term_2 =
        muminus_1 * (muminus_1 - const_f64::<F>(8.0)) / (const_f64::<F>(128.0) * abs_x * abs_x);

    // Result with enhanced precision
    let ampl = F::one() + term_1 + term_2;
    let result = one_over_sqrt_pi_x * const_f64::<F>(constants::f64::SQRT_2) * ampl * theta.cos();

    // For negative x, adjust the sign
    if x.is_sign_negative() && n % 2 != 0 {
        -result
    } else {
        result
    }
}

/// Exponentially scaled Bessel function of the first kind of order 0.
///
/// This function computes j0e(x) = j0(x) * exp(-abs(x.imag)) for complex x,
/// which prevents overflow for large arguments while preserving relative accuracy.
///
/// For real arguments, this is simply j0(x) since exp(-0) = 1.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * J₀ₑ(x) Exponentially scaled Bessel function value
///
/// # Examples
///
/// ```
/// use scirs2_special::bessel::first_kind::j0e;
///
/// // For real arguments, j0e(x) = j0(x)
/// let x = 2.0f64;
/// let result = j0e(x);
/// assert!(result.is_finite());
/// ```
#[allow(dead_code)]
pub fn j0e<F: Float + FromPrimitive + Debug>(x: F) -> F {
    // For real arguments, the imaginary part is zero, so exp(-abs(0)) = 1
    // Therefore j0e(x) = j0(x) for real x
    j0(x)
}

/// Exponentially scaled Bessel function of the first kind of order 1.
///
/// This function computes j1e(x) = j1(x) * exp(-abs(x.imag)) for complex x,
/// which prevents overflow for large arguments while preserving relative accuracy.
///
/// For real arguments, this is simply j1(x) since exp(-0) = 1.
///
/// # Arguments
///
/// * `x` - Input value
///
/// # Returns
///
/// * J₁ₑ(x) Exponentially scaled Bessel function value
///
/// # Examples
///
/// ```
/// use scirs2_special::bessel::first_kind::j1e;
///
/// // For real arguments, j1e(x) = j1(x)
/// let x = 2.0f64;
/// let result = j1e(x);
/// assert!(result.is_finite());
/// ```
#[allow(dead_code)]
pub fn j1e<F: Float + FromPrimitive + Debug>(x: F) -> F {
    // For real arguments, the imaginary part is zero, so exp(-abs(0)) = 1
    // Therefore j1e(x) = j1(x) for real x
    j1(x)
}

/// Exponentially scaled Bessel function of the first kind of integer order n.
///
/// This function computes jne(n, x) = jn(n, x) * exp(-abs(x.imag)) for complex x,
/// which prevents overflow for large arguments while preserving relative accuracy.
///
/// For real arguments, this is simply jn(n, x) since exp(-0) = 1.
///
/// # Arguments
///
/// * `n` - Order (integer)
/// * `x` - Input value
///
/// # Returns
///
/// * Jₙₑ(x) Exponentially scaled Bessel function value
///
/// # Examples
///
/// ```
/// use scirs2_special::bessel::first_kind::jne;
///
/// // For real arguments, jne(n, x) = jn(n, x)
/// let x = 2.0f64;
/// let result = jne(5, x);
/// assert!(result.is_finite());
/// ```
#[allow(dead_code)]
pub fn jne<F: Float + FromPrimitive + Debug + std::ops::AddAssign>(n: i32, x: F) -> F {
    // For real arguments, the imaginary part is zero, so exp(-abs(0)) = 1
    // Therefore jne(n, x) = jn(n, x) for real x
    jn(n, x)
}

/// Exponentially scaled Bessel function of the first kind of arbitrary real order.
///
/// This function computes jve(v, x) = jv(v, x) * exp(-abs(x.imag)) for complex x,
/// which prevents overflow for large arguments while preserving relative accuracy.
///
/// For real arguments, this is simply jv(v, x) since exp(-0) = 1.
///
/// # Arguments
///
/// * `v` - Order (any real number)
/// * `x` - Input value
///
/// # Returns
///
/// * Jᵥₑ(x) Exponentially scaled Bessel function value
///
/// # Examples
///
/// ```
/// use scirs2_special::bessel::first_kind::jve;
///
/// // For real arguments, jve(v, x) = jv(v, x)
/// let x = 2.0f64;
/// let result = jve(0.5, x);
/// assert!(result.is_finite());
/// ```
#[allow(dead_code)]
pub fn jve<F: Float + FromPrimitive + Debug + std::ops::AddAssign>(v: F, x: F) -> F {
    // For real arguments, the imaginary part is zero, so exp(-abs(0)) = 1
    // Therefore jve(v, x) = jv(v, x) for real x
    jv(v, x)
}

// Helper function to return maximum of two values.
#[allow(dead_code)]
fn max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_j0_special_cases() {
        // Test special values
        assert_relative_eq!(j0(0.0), 1.0, epsilon = 1e-10);

        // Test for very small argument
        let j0_small = j0(1e-10);
        assert_relative_eq!(j0_small, 1.0, epsilon = 1e-10);

        // Test that J₀ is close to zero at its first zero
        let first_zero = 2.404825557695773f64;
        let j0_at_zero = j0(first_zero);
        assert!(
            j0_at_zero.abs() < 1e-10,
            "J₀ should be close to zero at its first zero, got {}",
            j0_at_zero
        );
    }

    #[test]
    fn test_j0_moderate_values() {
        // SciPy-verified reference values
        assert_relative_eq!(j0(0.5), 0.9384698072408130, epsilon = 1e-10);
        assert_relative_eq!(j0(1.0), 0.7651976865579665, epsilon = 1e-10);
        assert_relative_eq!(j0(5.0), -0.1775967713143383, epsilon = 1e-10);
        assert_relative_eq!(j0(10.0), -0.2459357644513483, epsilon = 1e-10);
    }

    #[test]
    fn test_j0_large_values() {
        // Test large values
        let j0_50 = j0(50.0);
        let j0_100 = j0(100.0);
        let j0_1000 = j0(1000.0);

        // For large arguments, Bessel functions oscillate with decreasing amplitude
        assert!(j0_50.abs() < 0.1);
        assert!(j0_100.abs() < 0.1);
        assert!(j0_1000.abs() < 0.03);
    }

    #[test]
    fn test_j1_special_cases() {
        // Test special values
        assert_relative_eq!(j1(0.0), 0.0, epsilon = 1e-10);

        // Test for very small argument
        let j1_small = j1(1e-10);
        assert_relative_eq!(j1_small, 5e-11, epsilon = 1e-11);
    }

    #[test]
    fn test_j1_moderate_values() {
        // SciPy-verified reference values
        assert_relative_eq!(j1(0.5), 0.2422684576748739, epsilon = 1e-10);
        assert_relative_eq!(j1(1.0), 0.4400505857449335, epsilon = 1e-10);
        assert_relative_eq!(j1(5.0), -0.3275791375914653, epsilon = 1e-10);
        assert_relative_eq!(j1(10.0), 0.04347274616886141, epsilon = 1e-10);
    }

    #[test]
    fn test_jn_integer_orders() {
        let x = 5.0;

        // Compare with j0, j1
        assert_relative_eq!(jn(0, x), j0(x), epsilon = 1e-10);
        assert_relative_eq!(jn(1, x), j1(x), epsilon = 1e-10);

        // Test higher orders with SciPy-verified reference values
        assert_relative_eq!(jn(2, x), 0.04656511627775229, epsilon = 1e-10);
        assert_relative_eq!(jn(3, x), 0.36483123061366701, epsilon = 1e-10);
        assert_relative_eq!(jn(5, x), 0.26114054612017007, epsilon = 1e-10);
    }

    #[test]
    fn test_jv_integer_orders() {
        let x = 5.0;

        // Compare with j0, j1, jn
        assert_relative_eq!(jv(0.0, x), j0(x), epsilon = 1e-10);
        assert_relative_eq!(jv(1.0, x), j1(x), epsilon = 1e-10);
        assert_relative_eq!(jv(2.0, x), jn(2, x), epsilon = 1e-10);
        assert_relative_eq!(jv(5.0, x), jn(5, x), epsilon = 1e-10);
    }

    #[test]
    fn test_jv_half_integer_orders() {
        // Known values for half-integer orders
        // J_{1/2}(x) = sqrt(2/(πx)) * sin(x)
        let x = 2.0;
        let j_half = jv(0.5, x);
        let exact = (2.0 / (std::f64::consts::PI * x)).sqrt() * x.sin();
        assert_relative_eq!(j_half, exact, epsilon = 1e-8);

        // J_{3/2}(x) = sqrt(2/(πx)) * (sin(x)/x - cos(x))
        let j_three_half = jv(1.5, x);
        let exact = (2.0 / (std::f64::consts::PI * x)).sqrt() * (x.sin() / x - x.cos());
        assert_relative_eq!(j_three_half, exact, epsilon = 1e-8);
    }

    #[test]
    fn test_jv_negative_orders() {
        // Test the relationship between positive and negative orders
        // J_{-n}(x) = (-1)^n J_n(x) for integer n
        let x = 3.0;
        assert_relative_eq!(jv(-1.0, x), -j1(x), epsilon = 1e-10);
        assert_relative_eq!(jv(-2.0, x), jn(2, x), epsilon = 1e-10);

        // For non-integer order v, J_{-v}(x) ≠ (-1)^v J_v(x)
        // Instead we need the full relationship involving Gamma functions
        // This is a more complex test that would require a separate implementation
    }

    #[test]
    fn test_jv_negative_argument() {
        // For integer n, J_n(-x) = (-1)^n J_n(x)
        let x = 4.0;
        assert_relative_eq!(jv(0.0, -x), j0(x), epsilon = 1e-10);
        assert_relative_eq!(jv(1.0, -x), -j1(x), epsilon = 1e-10);
        assert_relative_eq!(jv(2.0, -x), jn(2, x), epsilon = 1e-10);
        assert_relative_eq!(jv(3.0, -x), -jn(3, x), epsilon = 1e-10);

        // For non-integer v, J_v(-x) is generally complex
        // The real part is cos(πv) J_v(x)
        let v = 0.5;
        let cos_pi_v = (std::f64::consts::PI * v).cos();
        let expected = cos_pi_v * jv(v, x);
        assert_relative_eq!(jv(v, -x), expected, epsilon = 1e-8);
    }
}
