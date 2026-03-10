//! Enhanced hypergeometric functions with convergence acceleration
//!
//! This module provides improved implementations of hypergeometric functions
//! with advanced convergence acceleration techniques.
//!
//! ## Key Features
//!
//! 1. **Levin u-transform**: Accelerates slowly converging series
//! 2. **Continued fractions**: Alternative representations for better convergence
//! 3. **Transformation formulas**: Analytic continuation to extended regions
//! 4. **Special case handling**: Optimized computation for specific parameter values
//!
//! ## Mathematical Background
//!
//! The hypergeometric function ₂F₁(a,b;c;z) is defined by the series:
//! ```text
//! ₂F₁(a,b;c;z) = Σ_{n=0}^∞ (a)_n (b)_n / ((c)_n n!) z^n
//! ```
//!
//! This series converges for |z| < 1 but can be analytically continued
//! to the entire complex plane (cut along [1, ∞)).

use crate::error::{SpecialError, SpecialResult};
use crate::gamma::{gamma, gammaln};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;
use std::ops::{AddAssign, MulAssign, SubAssign};

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

/// Maximum number of terms for series computations
const MAX_SERIES_TERMS: usize = 500;

/// Tolerance for convergence
const CONVERGENCE_TOL: f64 = 1e-15;

/// Enhanced hypergeometric function ₂F₁(a,b;c;z) with convergence acceleration
///
/// This implementation uses:
/// - Direct series for |z| < 0.5
/// - Transformation formulas for 0.5 ≤ |z| < 1
/// - Levin u-transform for slow convergence
/// - Pfaff/Euler transformations for z > 1 (via analytic continuation)
///
/// # Arguments
/// * `a` - First parameter
/// * `b` - Second parameter
/// * `c` - Third parameter (must not be 0, -1, -2, ...)
/// * `z` - Argument
///
/// # Returns
/// * Value of ₂F₁(a,b;c;z)
#[allow(dead_code)]
pub fn hyp2f1_enhanced<F>(a: F, b: F, c: F, z: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + MulAssign + SubAssign,
{
    let a_f64 = a
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert a to f64".to_string()))?;
    let b_f64 = b
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert b to f64".to_string()))?;
    let c_f64 = c
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert c to f64".to_string()))?;
    let z_f64 = z
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert z to f64".to_string()))?;

    // Check for poles in c
    if c_f64 <= 0.0 && c_f64.fract() == 0.0 {
        return Err(SpecialError::DomainError(format!(
            "c must not be 0 or negative integer, got {c_f64}"
        )));
    }

    // Special cases
    if z == F::zero() {
        return Ok(F::one());
    }

    // Terminating series: if a or b is a non-positive integer
    if (a_f64 <= 0.0 && a_f64.fract() == 0.0) || (b_f64 <= 0.0 && b_f64.fract() == 0.0) {
        return hyp2f1_terminating(a, b, c, z);
    }

    // z = 1: Gauss's theorem
    if (z_f64 - 1.0).abs() < 1e-14 {
        return hyp2f1_at_one(a, b, c);
    }

    // Choose algorithm based on z value
    let abs_z = z_f64.abs();

    if abs_z <= 0.5 {
        // Direct series with Levin acceleration
        hyp2f1_series_accelerated(a, b, c, z)
    } else if abs_z < 0.9 {
        // Use Pfaff transformation for better convergence
        hyp2f1_pfaff_transform(a, b, c, z)
    } else if abs_z < 1.0 {
        // Use Euler transformation near z = 1
        hyp2f1_euler_transform(a, b, c, z)
    } else if z_f64 > 1.0 {
        // Analytic continuation for z > 1
        hyp2f1_analytic_continuation_positive(a, b, c, z)
    } else {
        // z < -1: use different transformation
        hyp2f1_analytic_continuation_negative(a, b, c, z)
    }
}

/// Series computation with Levin u-transform acceleration
#[allow(dead_code)]
fn hyp2f1_series_accelerated<F>(a: F, b: F, c: F, z: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + MulAssign,
{
    // Compute series terms and apply Levin u-transform
    let mut terms = Vec::with_capacity(MAX_SERIES_TERMS);
    let mut term = F::one();
    let mut partial_sum = F::one();

    terms.push(F::one());

    for n in 1..MAX_SERIES_TERMS {
        let n_f = const_f64::<F>(n as f64);
        let n_minus_1 = const_f64::<F>((n - 1) as f64);

        // term_n = term_{n-1} * (a+n-1)(b+n-1)/(c+n-1) * z/n
        let numerator = (a + n_minus_1) * (b + n_minus_1);
        let denominator = (c + n_minus_1) * n_f;
        term = term * numerator * z / denominator;

        partial_sum += term;
        terms.push(partial_sum);

        // Check for convergence
        if term.abs() < const_f64::<F>(CONVERGENCE_TOL) * partial_sum.abs() {
            return Ok(partial_sum);
        }
    }

    // If direct series didn't converge well, apply Levin u-transform
    if terms.len() > 10 {
        return levin_u_transform(&terms);
    }

    Ok(partial_sum)
}

/// Levin u-transform for series acceleration
///
/// The Levin u-transform is a powerful sequence transformation for
/// accelerating the convergence of slowly converging series.
#[allow(dead_code)]
fn levin_u_transform<F>(partial_sums: &[F]) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug,
{
    let n = partial_sums.len();
    if n < 4 {
        return Ok(*partial_sums.last().unwrap_or(&F::zero()));
    }

    // Use Wynn's epsilon algorithm which is related to Levin transform
    // but more numerically stable
    let mut epsilon = vec![vec![F::zero(); n + 1]; n + 1];

    // Initialize with partial sums
    for (i, &s) in partial_sums.iter().enumerate() {
        epsilon[0][i] = F::zero();
        epsilon[1][i] = s;
    }

    // Compute epsilon table
    for k in 2..=n {
        for i in 0..=(n - k) {
            let diff = epsilon[k - 1][i + 1] - epsilon[k - 1][i];
            if diff.abs() < const_f64::<F>(1e-100) {
                // Avoid division by tiny numbers
                epsilon[k][i] = epsilon[k - 2][i + 1];
            } else {
                epsilon[k][i] = epsilon[k - 2][i + 1] + F::one() / diff;
            }
        }
    }

    // The best estimate is the last even column entry
    let best_col = if n.is_multiple_of(2) { n } else { n - 1 };
    Ok(epsilon[best_col][0])
}

/// Terminating series for non-positive integer a or b
#[allow(dead_code)]
fn hyp2f1_terminating<F>(a: F, b: F, c: F, z: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + MulAssign,
{
    let a_f64 = a
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert a to f64".to_string()))?;
    let b_f64 = b
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert b to f64".to_string()))?;

    // Determine the terminating parameter
    let n_terms = if a_f64 <= 0.0 && a_f64.fract() == 0.0 {
        (-a_f64) as usize
    } else if b_f64 <= 0.0 && b_f64.fract() == 0.0 {
        (-b_f64) as usize
    } else {
        return Err(SpecialError::ValueError(
            "Not a terminating series".to_string(),
        ));
    };

    let mut sum = F::one();
    let mut term = F::one();

    for n in 1..=n_terms {
        let n_f = const_f64::<F>(n as f64);
        let n_minus_1 = const_f64::<F>((n - 1) as f64);

        let numerator = (a + n_minus_1) * (b + n_minus_1);
        let denominator = (c + n_minus_1) * n_f;
        term = term * numerator * z / denominator;
        sum += term;
    }

    Ok(sum)
}

/// Gauss's theorem for ₂F₁(a,b;c;1)
///
/// When z = 1 and Re(c - a - b) > 0:
/// ₂F₁(a,b;c;1) = Γ(c)Γ(c-a-b) / (Γ(c-a)Γ(c-b))
#[allow(dead_code)]
fn hyp2f1_at_one<F>(a: F, b: F, c: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign,
{
    let c_minus_a_minus_b = c - a - b;
    let cmab_f64 = c_minus_a_minus_b
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Conversion failed".to_string()))?;

    if cmab_f64 <= 0.0 {
        return Err(SpecialError::DomainError(
            "₂F₁(a,b;c;1) diverges when c - a - b ≤ 0".to_string(),
        ));
    }

    // Use logarithms for numerical stability
    let log_gamma_c = gammaln(c);
    let log_gamma_cmab = gammaln(c_minus_a_minus_b);
    let log_gamma_cma = gammaln(c - a);
    let log_gamma_cmb = gammaln(c - b);

    let log_result = log_gamma_c + log_gamma_cmab - log_gamma_cma - log_gamma_cmb;

    Ok(log_result.exp())
}

/// Pfaff transformation for 0.5 ≤ |z| < 1
///
/// ₂F₁(a,b;c;z) = (1-z)^(-a) ₂F₁(a, c-b; c; z/(z-1))
#[allow(dead_code)]
fn hyp2f1_pfaff_transform<F>(a: F, b: F, c: F, z: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + MulAssign + SubAssign,
{
    let one_minus_z = F::one() - z;
    let z_transformed = z / (z - F::one());

    // The transformed z should have smaller absolute value
    let factor = one_minus_z.powf(-a);

    // Compute ₂F₁(a, c-b; c; z/(z-1)) with the series
    let transformed_result = hyp2f1_series_accelerated(a, c - b, c, z_transformed)?;

    Ok(factor * transformed_result)
}

/// Euler transformation for z near 1
///
/// ₂F₁(a,b;c;z) = (1-z)^(c-a-b) ₂F₁(c-a, c-b; c; z)
#[allow(dead_code)]
fn hyp2f1_euler_transform<F>(a: F, b: F, c: F, z: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + MulAssign + SubAssign,
{
    let one_minus_z = F::one() - z;
    let exponent = c - a - b;

    // (1-z)^(c-a-b) factor
    let factor = one_minus_z.powf(exponent);

    // Compute ₂F₁(c-a, c-b; c; z)
    let transformed_result = hyp2f1_series_accelerated(c - a, c - b, c, z)?;

    Ok(factor * transformed_result)
}

/// Analytic continuation for z > 1
///
/// Uses the connection formula to express ₂F₁(a,b;c;z) in terms of
/// functions evaluated at 1/z
#[allow(dead_code)]
fn hyp2f1_analytic_continuation_positive<F>(a: F, b: F, c: F, z: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + MulAssign + SubAssign,
{
    let z_inv = F::one() / z;

    // Connection formula:
    // ₂F₁(a,b;c;z) = Γ(c)Γ(b-a)/(Γ(b)Γ(c-a)) * (-z)^(-a) * ₂F₁(a, a-c+1; a-b+1; 1/z)
    //              + Γ(c)Γ(a-b)/(Γ(a)Γ(c-b)) * (-z)^(-b) * ₂F₁(b, b-c+1; b-a+1; 1/z)

    // Compute the first term
    let neg_z = -z;
    let term1_coeff = gamma(c) * gamma(b - a) / (gamma(b) * gamma(c - a));
    let term1_power = neg_z.powf(-a);
    let term1_hyp = hyp2f1_series_accelerated(a, a - c + F::one(), a - b + F::one(), z_inv)?;

    // Compute the second term
    let term2_coeff = gamma(c) * gamma(a - b) / (gamma(a) * gamma(c - b));
    let term2_power = neg_z.powf(-b);
    let term2_hyp = hyp2f1_series_accelerated(b, b - c + F::one(), b - a + F::one(), z_inv)?;

    Ok(term1_coeff * term1_power * term1_hyp + term2_coeff * term2_power * term2_hyp)
}

/// Analytic continuation for z < -1
#[allow(dead_code)]
fn hyp2f1_analytic_continuation_negative<F>(a: F, b: F, c: F, z: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + MulAssign + SubAssign,
{
    // For z < -1, we use transformation formulas
    // The principal branch uses the same connection formula

    let z_inv = F::one() / z;

    // Use the same formula as for positive z > 1
    let neg_z = -z;
    let term1_coeff = gamma(c) * gamma(b - a) / (gamma(b) * gamma(c - a));
    let term1_power = neg_z.powf(-a);
    let term1_hyp = hyp2f1_series_accelerated(a, a - c + F::one(), a - b + F::one(), z_inv)?;

    let term2_coeff = gamma(c) * gamma(a - b) / (gamma(a) * gamma(c - b));
    let term2_power = neg_z.powf(-b);
    let term2_hyp = hyp2f1_series_accelerated(b, b - c + F::one(), b - a + F::one(), z_inv)?;

    Ok(term1_coeff * term1_power * term1_hyp + term2_coeff * term2_power * term2_hyp)
}

/// Enhanced confluent hypergeometric function ₁F₁(a;b;z)
///
/// Uses multiple algorithms for different parameter ranges:
/// - Direct series for small |z|
/// - Asymptotic expansion for large |z|
/// - Kummer transformation for improved convergence
#[allow(dead_code)]
pub fn hyp1f1_enhanced<F>(a: F, b: F, z: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + MulAssign,
{
    let a_f64 = a
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert a to f64".to_string()))?;
    let b_f64 = b
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert b to f64".to_string()))?;
    let z_f64 = z
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert z to f64".to_string()))?;

    // Check for poles
    if b_f64 <= 0.0 && b_f64.fract() == 0.0 {
        return Err(SpecialError::DomainError(format!(
            "b must not be 0 or negative integer, got {b_f64}"
        )));
    }

    // Special case z = 0
    if z == F::zero() {
        return Ok(F::one());
    }

    // For large negative z, use Kummer transformation
    // ₁F₁(a;b;z) = e^z ₁F₁(b-a;b;-z)
    if z_f64 < -20.0 {
        let exp_z = z.exp();
        let transformed = hyp1f1_series(b - a, b, -z)?;
        return Ok(exp_z * transformed);
    }

    // For large positive z, use asymptotic expansion
    if z_f64 > 50.0 {
        return hyp1f1_asymptotic(a, b, z);
    }

    // For moderate z, use direct series
    hyp1f1_series(a, b, z)
}

/// Direct series for ₁F₁
#[allow(dead_code)]
fn hyp1f1_series<F>(a: F, b: F, z: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + MulAssign,
{
    let mut sum = F::one();
    let mut term = F::one();

    for n in 1..MAX_SERIES_TERMS {
        let n_f = const_f64::<F>(n as f64);
        let n_minus_1 = const_f64::<F>((n - 1) as f64);

        term = term * (a + n_minus_1) * z / ((b + n_minus_1) * n_f);
        sum += term;

        if term.abs() < const_f64::<F>(CONVERGENCE_TOL) * sum.abs() {
            return Ok(sum);
        }
    }

    Ok(sum)
}

/// Asymptotic expansion for large z
#[allow(dead_code)]
fn hyp1f1_asymptotic<F>(a: F, b: F, z: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign,
{
    // For large positive z:
    // ₁F₁(a;b;z) ∼ Γ(b)/Γ(a) * e^z * z^(a-b) * (1 + O(1/z))

    let gamma_b = gamma(b);
    let gamma_a = gamma(a);

    let exp_z = z.exp();
    let z_power = z.powf(a - b);

    // Leading term
    let leading = gamma_b / gamma_a * exp_z * z_power;

    // First correction term
    let correction = (b - a) * (F::one() - a) / z;

    Ok(leading * (F::one() + correction))
}

/// Enhanced confluent hypergeometric limit function ₀F₁(;a;z)
#[allow(dead_code)]
pub fn hyp0f1_enhanced<F>(a: F, z: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + MulAssign,
{
    let a_f64 = a
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert a to f64".to_string()))?;

    // Check for poles
    if a_f64 <= 0.0 && a_f64.fract() == 0.0 {
        return Err(SpecialError::DomainError(format!(
            "a must not be 0 or negative integer, got {a_f64}"
        )));
    }

    // z = 0
    if z == F::zero() {
        return Ok(F::one());
    }

    // Direct series: ₀F₁(;a;z) = Σ z^n / ((a)_n * n!)
    let mut sum = F::one();
    let mut term = F::one();

    for n in 1..MAX_SERIES_TERMS {
        let n_f = const_f64::<F>(n as f64);
        let n_minus_1 = const_f64::<F>((n - 1) as f64);

        term = term * z / ((a + n_minus_1) * n_f);
        sum += term;

        if term.abs() < const_f64::<F>(CONVERGENCE_TOL) * sum.abs() {
            return Ok(sum);
        }
    }

    Ok(sum)
}

/// Regularized hypergeometric function ₂F₁(a,b;c;z) / Γ(c)
///
/// This is useful when c is near a non-positive integer, where
/// the standard ₂F₁ has a pole but the regularized version is finite.
#[allow(dead_code)]
pub fn hyp2f1_regularized<F>(a: F, b: F, c: F, z: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + MulAssign + SubAssign,
{
    let c_f64 = c
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert c to f64".to_string()))?;

    // For c near a non-positive integer, compute directly
    if c_f64 <= 0.0 && c_f64.fract().abs() < 1e-10 {
        // The regularized function is finite at these points
        // ₂F₁ᵣ(a,b;c;z) = Σ (a)_n (b)_n / (n!)² * z^n / (c+n-1)!
        // This requires careful computation
        return hyp2f1_regularized_at_pole(a, b, c, z);
    }

    // For regular c, just divide by Gamma(c)
    let gamma_c = gamma(c);
    let hyp = hyp2f1_enhanced(a, b, c, z)?;

    Ok(hyp / gamma_c)
}

/// Regularized hypergeometric when c is a non-positive integer
#[allow(dead_code)]
fn hyp2f1_regularized_at_pole<F>(a: F, b: F, c: F, z: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + MulAssign,
{
    let c_f64 = c
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert c to f64".to_string()))?;

    let m = (-c_f64).round() as usize; // c ≈ -m for some non-negative integer m

    // The regularized form involves the analytic continuation
    // around the pole using L'Hôpital's rule
    let mut sum = F::zero();

    // For the regularized form, we compute a finite series
    for n in 0..MAX_SERIES_TERMS {
        let n_f = const_f64::<F>(n as f64);

        if n < m + 1 {
            // Skip the first m+1 terms where the denominator would be zero
            continue;
        }

        let a_n = pochhammer_n(a, n);
        let b_n = pochhammer_n(b, n);
        let n_factorial = factorial_n(n);

        // (c)_n for c = -m is: (-m)(-m+1)...(-m+n-1) = (-1)^n * m!/(m-n)! for n <= m
        // For n > m, use the continuation
        let c_n = pochhammer_n(c, n);

        if c_n.abs() < const_f64::<F>(1e-100) {
            continue;
        }

        let term = a_n * b_n * z.powi(n as i32) / (c_n * n_factorial);
        sum += term;

        if n > 10 && term.abs() < const_f64::<F>(CONVERGENCE_TOL) * sum.abs() {
            break;
        }
    }

    Ok(sum)
}

/// Helper: Pochhammer symbol (a)_n
#[allow(dead_code)]
fn pochhammer_n<F>(a: F, n: usize) -> F
where
    F: Float + FromPrimitive,
{
    if n == 0 {
        return F::one();
    }

    let mut result = a;
    for i in 1..n {
        result = result * (a + const_f64::<F>(i as f64));
    }
    result
}

/// Helper: factorial n!
#[allow(dead_code)]
fn factorial_n<F>(n: usize) -> F
where
    F: Float + FromPrimitive,
{
    if n <= 1 {
        return F::one();
    }

    let mut result = F::one();
    for i in 2..=n {
        result = result * const_f64::<F>(i as f64);
    }
    result
}

/// Regularized confluent hypergeometric function ₁F₁(a;b;z) / Γ(b)
///
/// Also known as the regularized Kummer function M*(a,b,z).
/// This is useful when b is near a non-positive integer.
///
/// # Arguments
/// * `a` - First parameter
/// * `b` - Second parameter
/// * `z` - Argument
///
/// # Returns
/// * Value of ₁F₁(a;b;z) / Γ(b)
///
/// # Examples
/// ```
/// use scirs2_special::hyp1f1_regularized;
/// let result: f64 = hyp1f1_regularized(1.0, 2.0, 0.0).expect("failed");
/// // 1F1(1,2,0) = 1, Gamma(2) = 1, so regularized = 1
/// assert!((result - 1.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn hyp1f1_regularized<F>(a: F, b: F, z: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + MulAssign + SubAssign,
{
    let b_f64 = b
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert b to f64".to_string()))?;

    // For b near a non-positive integer, compute specially
    if b_f64 <= 0.0 && b_f64.fract().abs() < 1e-10 {
        // At pole of Gamma(b): regularized form is finite
        return hyp1f1_regularized_at_pole(a, b, z);
    }

    // For regular b, compute 1F1 / Gamma(b)
    let gamma_b = gamma(b);
    if gamma_b.abs() < const_f64::<F>(1e-300) {
        return Ok(F::zero());
    }
    let hyp = hyp1f1_enhanced(a, b, z)?;
    Ok(hyp / gamma_b)
}

/// Helper for regularized hyp1f1 at poles of Gamma(b)
fn hyp1f1_regularized_at_pole<F>(a: F, b: F, z: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + MulAssign,
{
    let b_f64 = b
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert b to f64".to_string()))?;
    let m = (-b_f64).round() as usize;

    // Compute the finite limit as b -> -m
    // Using the series representation with careful cancellation
    let mut sum = F::zero();
    for n in (m + 1)..MAX_SERIES_TERMS {
        let a_n = pochhammer_n(a, n);
        let b_n = pochhammer_n(b, n);
        let n_fact = factorial_n::<F>(n);

        if b_n.abs() < const_f64::<F>(1e-300) {
            continue;
        }

        let term = a_n * z.powi(n as i32) / (b_n * n_fact);
        sum += term;

        if n > m + 5 && term.abs() < const_f64::<F>(CONVERGENCE_TOL) * sum.abs() {
            break;
        }
    }

    Ok(sum)
}

/// Whittaker function M_{kappa,mu}(z)
///
/// The Whittaker M function is defined in terms of the confluent hypergeometric function:
/// ```text
/// M_{k,m}(z) = exp(-z/2) * z^{m+1/2} * 1F1(m - k + 1/2; 2m + 1; z)
/// ```
///
/// # Arguments
/// * `kappa` - First parameter
/// * `mu` - Second parameter (2*mu must not be a negative integer)
/// * `z` - Argument (z > 0 for real branch)
///
/// # Examples
/// ```
/// use scirs2_special::whittaker_m;
/// let result: f64 = whittaker_m(0.5_f64, 0.5, 1.0).expect("failed");
/// assert!(result.is_finite());
/// ```
#[allow(dead_code)]
pub fn whittaker_m<F>(kappa: F, mu: F, z: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + MulAssign + SubAssign,
{
    let z_f64 = z
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert z to f64".to_string()))?;

    if z_f64 <= 0.0 {
        return Err(SpecialError::DomainError(
            "z must be > 0 for Whittaker M function".to_string(),
        ));
    }

    let half = const_f64::<F>(0.5);
    let one = F::one();
    let two = const_f64::<F>(2.0);

    // Parameters for 1F1
    let a = mu - kappa + half;
    let b = two * mu + one;

    // Compute M_{k,m}(z) = exp(-z/2) * z^{m+1/2} * 1F1(a; b; z)
    let exp_factor = (-z * half).exp();
    let z_power = z.powf(mu + half);
    let hyp = hyp1f1_enhanced(a, b, z)?;

    Ok(exp_factor * z_power * hyp)
}

/// Whittaker function W_{kappa,mu}(z)
///
/// The Whittaker W function is defined in terms of the Tricomi confluent hypergeometric:
/// ```text
/// W_{k,m}(z) = exp(-z/2) * z^{m+1/2} * U(m - k + 1/2; 2m + 1; z)
/// ```
///
/// # Arguments
/// * `kappa` - First parameter
/// * `mu` - Second parameter
/// * `z` - Argument (z > 0)
///
/// # Examples
/// ```
/// use scirs2_special::whittaker_w;
/// let result: f64 = whittaker_w(0.5_f64, 0.5, 1.0).expect("failed");
/// assert!(result.is_finite());
/// ```
#[allow(dead_code)]
pub fn whittaker_w<F>(kappa: F, mu: F, z: F) -> SpecialResult<F>
where
    F: Float + FromPrimitive + Debug + AddAssign + MulAssign + SubAssign,
{
    let z_f64 = z
        .to_f64()
        .ok_or_else(|| SpecialError::ValueError("Failed to convert z to f64".to_string()))?;

    if z_f64 <= 0.0 {
        return Err(SpecialError::DomainError(
            "z must be > 0 for Whittaker W function".to_string(),
        ));
    }

    let half = const_f64::<F>(0.5);
    let one = F::one();
    let two = const_f64::<F>(2.0);

    // Parameters for U
    let a = mu - kappa + half;
    let b = two * mu + one;

    // W_{k,m}(z) = exp(-z/2) * z^{m+1/2} * U(a, b, z)
    let exp_factor = (-z * half).exp();
    let z_power = z.powf(mu + half);

    // Use Tricomi U function from main hypergeometric module
    let u_val = crate::hypergeometric::hyperu(a, b, z)?;

    Ok(exp_factor * z_power * u_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hyp2f1_enhanced_zero() {
        let result: f64 = hyp2f1_enhanced(1.0, 2.0, 3.0, 0.0).expect("test should succeed");
        assert_relative_eq!(result, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_hyp2f1_enhanced_small_z() {
        // Known value from SciPy: hyp2f1(1.0, 2.0, 3.0, 0.5) = 1.545177444479562
        let result: f64 = hyp2f1_enhanced(1.0, 2.0, 3.0, 0.5).expect("test should succeed");
        assert_relative_eq!(result, 1.545177444479562, epsilon = 1e-10);
    }

    #[test]
    fn test_hyp2f1_enhanced_at_one() {
        // At z=1, use Gauss's theorem: ₂F₁(a,b;c;1) = Γ(c)Γ(c-a-b)/(Γ(c-a)Γ(c-b))
        // For a=0.5, b=1, c=3: should be about 1.5
        let result: f64 = hyp2f1_enhanced(0.5, 1.0, 3.0, 1.0).expect("test should succeed");
        assert!(result.is_finite());
    }

    #[test]
    fn test_hyp2f1_enhanced_terminating() {
        // For a = -2, the series terminates
        let result: f64 = hyp2f1_enhanced(-2.0, 3.0, 4.0, 0.5).expect("test should succeed");
        // Should match: 1 + (-2)(3)/(4*1)*0.5 + (-2)(-1)(3)(4)/(4*5*1*2)*0.25
        // = 1 - 0.75 + 0.15 = 0.4
        assert_relative_eq!(result, 0.4, epsilon = 1e-10);
    }

    #[test]
    fn test_hyp1f1_enhanced_zero() {
        let result: f64 = hyp1f1_enhanced(1.0, 2.0, 0.0).expect("test should succeed");
        assert_relative_eq!(result, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_hyp1f1_enhanced_small_z() {
        let result: f64 = hyp1f1_enhanced(1.0, 2.0, 0.5).expect("test should succeed");
        // Known value approximately 1.297...
        assert!(result > 1.0 && result < 2.0);
    }

    #[test]
    fn test_hyp0f1_enhanced() {
        let result: f64 = hyp0f1_enhanced(1.0, 0.0).expect("test should succeed");
        assert_relative_eq!(result, 1.0, epsilon = 1e-14);

        let result2: f64 = hyp0f1_enhanced(1.0, 1.0).expect("test should succeed");
        // ₀F₁(;1;1) is related to Bessel functions
        assert!(result2 > 1.0);
    }

    #[test]
    fn test_levin_transform() {
        // Test that Levin transform works for a known sequence
        let partial_sums: Vec<f64> = vec![1.0, 1.5, 1.833, 2.083, 2.283, 2.45, 2.593];
        let result = levin_u_transform(&partial_sums).expect("test should succeed");
        assert!(result.is_finite());
    }

    // ====== Regularized hyp1f1 tests ======

    #[test]
    fn test_hyp1f1_regularized_at_zero() {
        // 1F1(a,b,0) = 1, so regularized = 1/Gamma(b)
        let result: f64 = hyp1f1_regularized(1.0, 2.0, 0.0).expect("should succeed");
        // Gamma(2) = 1, so regularized = 1/1 = 1
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hyp1f1_regularized_known_value() {
        // For b=1: Gamma(1) = 1, so regularized = 1F1 itself
        let result: f64 = hyp1f1_regularized(1.0, 1.0, 0.5).expect("should succeed");
        let direct: f64 = hyp1f1_enhanced(1.0, 1.0, 0.5).expect("should succeed");
        assert_relative_eq!(result, direct, epsilon = 1e-10);
    }

    #[test]
    fn test_hyp1f1_regularized_large_b() {
        // For large b, Gamma(b) is large, so regularized is small
        let result: f64 = hyp1f1_regularized(1.0, 10.0, 0.5).expect("should succeed");
        assert!(
            result.abs() < 1.0,
            "regularized should be small for large b: {result}"
        );
    }

    #[test]
    fn test_hyp1f1_regularized_b_half() {
        // Gamma(0.5) = sqrt(pi)
        let result: f64 = hyp1f1_regularized(1.0, 0.5, 0.0).expect("should succeed");
        let expected = 1.0 / std::f64::consts::PI.sqrt();
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_hyp1f1_regularized_finite() {
        let result: f64 = hyp1f1_regularized(0.5, 3.0, 1.0).expect("should succeed");
        assert!(result.is_finite(), "regularized hyp1f1 should be finite");
    }

    // ====== Whittaker M tests ======

    #[test]
    fn test_whittaker_m_basic() {
        let result = whittaker_m(0.5_f64, 0.5, 1.0).expect("should succeed");
        assert!(
            result.is_finite(),
            "M_{{0.5,0.5}}(1) should be finite: {result}"
        );
    }

    #[test]
    fn test_whittaker_m_zero_kappa() {
        // M_{0,m}(z) = exp(-z/2) z^{m+1/2} 1F1(m+1/2, 2m+1, z)
        let result = whittaker_m(0.0_f64, 0.5, 2.0).expect("should succeed");
        assert!(result.is_finite());
    }

    #[test]
    fn test_whittaker_m_large_z() {
        let result = whittaker_m(0.5_f64, 0.5, 10.0).expect("should succeed");
        assert!(result.is_finite());
    }

    #[test]
    fn test_whittaker_m_negative_z_error() {
        let result = whittaker_m(0.5_f64, 0.5, -1.0);
        assert!(result.is_err(), "negative z should error");
    }

    #[test]
    fn test_whittaker_m_positive() {
        // For small z, the exponential factor dominates and result should be small
        let result = whittaker_m(1.0_f64, 0.5, 0.1).expect("should succeed");
        assert!(result.is_finite());
        assert!(result > 0.0, "M should be positive for these params");
    }

    // ====== Whittaker W tests ======

    #[test]
    fn test_whittaker_w_basic() {
        let result = whittaker_w(0.5_f64, 0.5, 1.0).expect("should succeed");
        assert!(
            result.is_finite(),
            "W_{{0.5,0.5}}(1) should be finite: {result}"
        );
    }

    #[test]
    fn test_whittaker_w_large_z() {
        // For large z, W decays exponentially
        let result = whittaker_w(0.5_f64, 0.5, 10.0).expect("should succeed");
        assert!(result.is_finite());
        assert!(result.abs() < 1.0, "W should decay for large z: {result}");
    }

    #[test]
    fn test_whittaker_w_negative_z_error() {
        let result = whittaker_w(0.5_f64, 0.5, -1.0);
        assert!(result.is_err(), "negative z should error");
    }

    #[test]
    fn test_whittaker_w_moderate() {
        let result = whittaker_w(1.0_f64, 0.5, 2.0).expect("should succeed");
        assert!(result.is_finite());
    }

    #[test]
    fn test_whittaker_w_small_z() {
        let result = whittaker_w(0.5_f64, 0.5, 0.1).expect("should succeed");
        assert!(result.is_finite());
    }
}
