//! Zolotarev rational approximation for spectral filtering
//!
//! This module provides optimal rational function approximations based on
//! Zolotarev's theory, primarily for approximating the sign function and step
//! function on the real line. These rational approximations can serve as spectral
//! filters in eigenvalue problems, providing an alternative to contour integration.
//!
//! ## Mathematical Background
//!
//! Zolotarev's optimal rational approximation gives the best L-infinity approximation
//! to the sign function `sign(x)` on `[-1, -delta] ∪ [delta, 1]`, where `delta > 0`
//! defines the gap around the origin.
//!
//! The rational approximation takes the form:
//!   `r(x) = x * prod_j (x^2 + c_j) / prod_j (x^2 + d_j)`
//!
//! where the `c_j` and `d_j` are determined by Jacobi elliptic functions.
//!
//! ## Applications
//!
//! - Spectral projectors for eigenvalue problems (rational filtering)
//! - Sign function computation for matrix functions
//! - Step function approximation for counting eigenvalues in intervals
//!
//! ## References
//!
//! - Zolotarev, E. I. (1877). "Application of elliptic functions to questions of
//!   functions deviating least and most from zero."
//! - Nakatsukasa, Y., & Freund, R. W. (2016). "Computing fundamental matrix
//!   decompositions accurately via the matrix sign function in two iterations."
//! - Guttel, S., Polizzi, E., Tang, P. T. P., & Vioreanu, G. (2015). "Zolotarev
//!   quadrature rules and load balancing for the FEAST eigensolver."

use scirs2_core::numeric::Float;
use std::f64::consts::PI;

use crate::error::{LinalgError, LinalgResult};

/// Result type for Zolotarev filter: (shifts as complex pairs, weights)
type ZolotarevFilterResult<F> = LinalgResult<(Vec<(F, F)>, Vec<F>)>;

/// Result of a Zolotarev rational approximation computation
#[derive(Clone, Debug)]
pub struct ZolotarevApproximation<F: Float> {
    /// Poles of the rational function (complex, stored as (real, imag) pairs).
    /// For sign function: poles are purely imaginary, at x = +/- `i*pole_im[j]`.
    pub poles: Vec<(F, F)>,
    /// Residues at the poles (complex, stored as (real, imag) pairs).
    /// For the partial fraction form: r(x) = sum_j alpha_j * x / (x^2 + c_j)
    pub residues: Vec<(F, F)>,
    /// Numerator squared values: a_j^2 where the numerator is prod_j(x^2 + a_j^2)
    pub numerator_sq: Vec<F>,
    /// Denominator squared values: b_j^2 where the denominator is prod_j(x^2 + b_j^2)
    pub denominator_sq: Vec<F>,
    /// Scaling constant for the product form
    pub scale: F,
    /// Degree of the approximation (number of pole-residue pairs)
    pub degree: usize,
    /// Gap parameter delta
    pub delta: F,
    /// Maximum approximation error on the domain
    pub max_error: F,
}

/// Compute the complete elliptic integral of the first kind K(k)
/// using the arithmetic-geometric mean (AGM) iteration.
///
/// K(k) = pi / (2 * AGM(1, sqrt(1 - k^2)))
fn complete_elliptic_k(k: f64) -> LinalgResult<f64> {
    if !(0.0..1.0).contains(&k) {
        return Err(LinalgError::DomainError(format!(
            "Elliptic integral modulus k must be in [0, 1), got {}",
            k
        )));
    }

    if k == 0.0 {
        return Ok(PI / 2.0);
    }

    let kp = (1.0 - k * k).sqrt(); // complementary modulus
    let mut a = 1.0;
    let mut b = kp;

    for _ in 0..100 {
        let a_new = (a + b) / 2.0;
        let b_new = (a * b).sqrt();
        if (a_new - b_new).abs() < 1e-15 * a_new {
            return Ok(PI / (2.0 * a_new));
        }
        a = a_new;
        b = b_new;
    }

    Ok(PI / (2.0 * ((a + b) / 2.0)))
}

/// Compute the Jacobi elliptic functions sn(u, k), cn(u, k), dn(u, k) using
/// the arithmetic-geometric mean (AGM) method (Abramowitz & Stegun algorithm).
///
/// This is the standard method: descend via AGM to find the amplitude, then ascend.
fn jacobi_elliptic(u: f64, k: f64) -> LinalgResult<(f64, f64, f64)> {
    if k.abs() < 1e-15 {
        return Ok((u.sin(), u.cos(), 1.0));
    }
    if (k - 1.0).abs() < 1e-15 {
        let sn = u.tanh();
        let cn = 1.0 / u.cosh();
        return Ok((sn, cn, cn));
    }

    // AGM descending sequence
    let mut a = vec![1.0];
    let mut b = vec![(1.0 - k * k).sqrt()]; // b_0 = k'
    let mut c = vec![k];

    for _ in 0..50 {
        let a_prev = *a.last().unwrap_or(&1.0);
        let b_prev = *b.last().unwrap_or(&1.0);
        let a_new = (a_prev + b_prev) / 2.0;
        let b_new = (a_prev * b_prev).sqrt();
        let c_new = (a_prev - b_prev) / 2.0;

        a.push(a_new);
        b.push(b_new);
        c.push(c_new);

        if c_new.abs() < 1e-15 {
            break;
        }
    }

    let n = a.len() - 1;

    // Compute phi_n = 2^n * a_n * u
    let mut phi = (1u64 << n.min(62)) as f64 * a[n] * u;

    // Ascending recurrence for the amplitude
    for j in (0..n).rev() {
        phi = (phi + (c[j + 1] / a[j + 1] * phi.sin()).asin()) / 2.0;
    }

    let sn = phi.sin();
    let cn = phi.cos();
    let dn = (1.0 - k * k * sn * sn).max(0.0).sqrt();

    Ok((sn, cn, dn))
}

/// Compute the Jacobi elliptic function sn(u, k).
fn jacobi_sn(u: f64, k: f64) -> LinalgResult<f64> {
    let (sn, _, _) = jacobi_elliptic(u, k)?;
    Ok(sn)
}

/// Compute the Jacobi elliptic function cn(u, k).
fn jacobi_cn(u: f64, k: f64) -> LinalgResult<f64> {
    let (_, cn, _) = jacobi_elliptic(u, k)?;
    Ok(cn)
}

/// Compute the Jacobi elliptic function dn(u, k).
fn jacobi_dn(u: f64, k: f64) -> LinalgResult<f64> {
    let (_, _, dn) = jacobi_elliptic(u, k)?;
    Ok(dn)
}

/// Compute the Zolotarev rational approximation to the sign function.
///
/// Computes an optimal degree-n rational approximation `r(x)` to `sign(x)`
/// on `[-1, -delta] ∪ [delta, 1]`, where the approximation satisfies:
///
///   `max_{x in domain} |r(x) - sign(x)| = minimal`
///
/// The approximation has the partial fraction form:
///   `r(x) = alpha * x * sum_j (residue_j / (x^2 - pole_j))`
///
/// # Arguments
///
/// * `degree` - Degree of the rational approximation (number of terms). Must be >= 1.
/// * `delta` - Gap parameter. The approximation is valid on `[-1, -delta] ∪ [delta, 1]`.
///   Must be in `(0, 1)`. Smaller delta requires higher degree for same accuracy.
///
/// # Returns
///
/// * `ZolotarevApproximation` containing poles, residues, and error estimate
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::eigen::zolotarev::zolotarev_sign;
///
/// // Degree-4 approximation with delta = 0.1
/// let approx = zolotarev_sign::<f64>(4, 0.1).expect("Zolotarev computation failed");
/// assert_eq!(approx.degree, 4);
/// // Error bound decreases exponentially with degree
/// assert!(approx.max_error < 1.0);
/// ```
pub fn zolotarev_sign<F>(degree: usize, delta: f64) -> LinalgResult<ZolotarevApproximation<F>>
where
    F: Float,
{
    if degree == 0 {
        return Err(LinalgError::DomainError(
            "Zolotarev degree must be >= 1".to_string(),
        ));
    }

    if delta <= 0.0 || delta >= 1.0 {
        return Err(LinalgError::DomainError(format!(
            "Delta must be in (0, 1), got {}",
            delta
        )));
    }

    let n = degree;

    // For the Zolotarev sign function approximation on [-1, -delta] ∪ [delta, 1]:
    //
    // Following the standard formulation (Nakatsukasa & Freund 2016):
    //   r_n(x) = x * prod_{j=1}^{n-1} (x^2 + c_{2j}^2) / prod_{j=0}^{n-1} (x^2 + c_{2j+1}^2)
    //
    // where c_m = cn(m*K'/(2n), k') * dn(m*K'/(2n), k') / sn(m*K'/(2n), k')
    //
    // with k' = delta and K' = K(k') is the complete elliptic integral of k'.
    //
    // The number of denominator factors is n, number of numerator factors is n-1,
    // plus the leading x, giving total degree (2n-1, 2n-2) in x.

    let kp = delta; // complementary modulus
    let k = (1.0 - kp * kp).sqrt();

    let kk = complete_elliptic_k(k)?;
    let kkp = complete_elliptic_k(kp)?;

    // Approximation error bound
    let ratio = kkp / kk;
    let max_error = 4.0 * (-(n as f64) * PI * ratio).exp();

    // Compute c_m for m = 1, 2, ..., 2n-1
    // c_m = cn(m*K'/(2n), k') * dn(m*K'/(2n), k') / sn(m*K'/(2n), k')
    let mut c_vals = Vec::with_capacity(2 * n);
    for m in 1..=(2 * n - 1) {
        let u = m as f64 * kkp / (2.0 * n as f64);
        let (sn_val, cn_val, dn_val) = jacobi_elliptic(u, kp)?;
        if sn_val.abs() < 1e-30 {
            // This shouldn't happen for m < 2n
            c_vals.push(1e30);
        } else {
            c_vals.push(cn_val * dn_val / sn_val);
        }
    }

    // Denominator factors: c_{2j+1} for j = 0, ..., n-1 => c_1, c_3, c_5, ..., c_{2n-1}
    // Numerator factors:   c_{2j}   for j = 1, ..., n-1 => c_2, c_4, c_6, ..., c_{2n-2}
    let mut den_sq = Vec::with_capacity(n); // c_{odd}^2
    let mut num_sq = Vec::with_capacity(n - 1); // c_{even}^2

    for j in 0..n {
        let m = 2 * j + 1; // m = 1, 3, 5, ..., 2n-1
        let c = c_vals[m - 1]; // 0-indexed
        den_sq.push(c * c);
    }
    for j in 1..n {
        let m = 2 * j; // m = 2, 4, 6, ..., 2n-2
        let c = c_vals[m - 1];
        num_sq.push(c * c);
    }

    // Compute scale so that r(1) = 1
    // r(1) = scale * 1 * prod_j(1 + c_{2j}^2) / prod_j(1 + c_{2j+1}^2) = 1
    let mut num_at_1 = 1.0;
    for &a2 in &num_sq {
        num_at_1 *= 1.0 + a2;
    }
    let mut den_at_1 = 1.0;
    for &b2 in &den_sq {
        den_at_1 *= 1.0 + b2;
    }
    let scale = if num_at_1.abs() > 1e-30 {
        den_at_1 / num_at_1
    } else {
        1.0
    };

    // Store poles: denominator has factors (x^2 + c_{2j+1}^2), so poles at x = +/- i*c_{2j+1}
    let mut poles = Vec::with_capacity(n);
    for bsq in den_sq.iter().take(n) {
        let c = bsq.sqrt();
        poles.push((
            F::from(0.0).unwrap_or(F::zero()),
            F::from(c).unwrap_or(F::zero()),
        ));
    }

    // Compute partial fraction residues numerically
    // r(x)/x = scale * prod_j(x^2 + num_sq[j]) / prod_j(x^2 + den_sq[j])
    //
    // Partial fraction of r(x)/x:
    // r(x)/x = A_0 + sum_j A_j / (x^2 + den_sq[j])
    //
    // A_j = lim_{x^2 -> -den_sq[j]} (x^2 + den_sq[j]) * scale * prod(x^2+num_sq[i]) / prod(x^2+den_sq[i])
    //     = scale * prod_i(-den_sq[j] + num_sq[i]) / prod_{i!=j}(-den_sq[j] + den_sq[i])

    let mut residues = Vec::with_capacity(n);
    for j in 0..n {
        let bj = den_sq[j];
        let mut num_val = scale;
        for &a2 in &num_sq {
            num_val *= a2 - bj; // (x^2 + a2) at x^2 = -bj => a2 - bj
        }
        let mut den_val = 1.0;
        for (i, &bi) in den_sq.iter().enumerate() {
            if i != j {
                den_val *= bi - bj; // (x^2 + bi) at x^2 = -bj => bi - bj, but we need
                                    // prod_{i!=j}(x^2+den_sq[i]) at x^2=-den_sq[j] = prod(den_sq[i]-den_sq[j])
            }
        }
        let alpha_j = if den_val.abs() > 1e-30 {
            num_val / den_val
        } else {
            0.0
        };
        residues.push((
            F::from(alpha_j).unwrap_or(F::zero()),
            F::from(0.0).unwrap_or(F::zero()),
        ));
    }

    Ok(ZolotarevApproximation {
        poles,
        residues,
        numerator_sq: num_sq
            .iter()
            .map(|&v| F::from(v).unwrap_or(F::zero()))
            .collect(),
        denominator_sq: den_sq
            .iter()
            .map(|&v| F::from(v).unwrap_or(F::zero()))
            .collect(),
        scale: F::from(scale).unwrap_or(F::one()),
        degree: n,
        delta: F::from(delta).unwrap_or(F::zero()),
        max_error: F::from(max_error).unwrap_or(F::zero()),
    })
}

/// Evaluate the Zolotarev rational approximation at a given point.
///
/// Computes `r(x) = sum_j alpha_j * x / (x^2 + c_j)` where `c_j = pole_imag_j^2`.
///
/// # Arguments
///
/// * `x` - Point at which to evaluate the approximation
/// * `approx` - The Zolotarev approximation parameters
///
/// # Returns
///
/// * The value `r(x)` of the rational approximation
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::eigen::zolotarev::{zolotarev_sign, evaluate_rational};
///
/// let approx = zolotarev_sign::<f64>(8, 0.1).expect("Zolotarev computation failed");
///
/// // Should be close to sign(0.5) = 1.0
/// let val = evaluate_rational(0.5, &approx);
/// assert!((val - 1.0).abs() < 0.01);
///
/// // Should be close to sign(-0.5) = -1.0
/// let val_neg = evaluate_rational(-0.5, &approx);
/// assert!((val_neg + 1.0).abs() < 0.01);
/// ```
pub fn evaluate_rational<F>(x: F, approx: &ZolotarevApproximation<F>) -> F
where
    F: Float,
{
    // Product form: r(x) = scale * x * prod_j(x^2 + a_j^2) / prod_j(x^2 + b_j^2)
    let x2 = x * x;

    let mut numerator = approx.scale * x;
    for &a2 in &approx.numerator_sq {
        numerator = numerator * (x2 + a2);
    }

    let mut denominator = F::one();
    for &b2 in &approx.denominator_sq {
        denominator = denominator * (x2 + b2);
    }

    if denominator.abs() > F::epsilon() {
        numerator / denominator
    } else {
        // Near a pole; use partial fraction form as fallback
        // r(x) = sum_j alpha_j * x / (x^2 + b_j^2)
        let mut result = F::zero();
        for j in 0..approx.residues.len() {
            let (alpha_re, _) = approx.residues[j];
            let b2 = approx.denominator_sq[j];
            let denom = x2 + b2;
            if denom.abs() > F::epsilon() {
                result = result + alpha_re * x / denom;
            }
        }
        result
    }
}

/// Evaluate the Zolotarev step function approximation at a given point.
///
/// The step function `h(x) = (1 + sign(x)) / 2` can be approximated using
/// the sign function approximation: `h(x) = (1 + r(x)) / 2`.
///
/// # Arguments
///
/// * `x` - Point at which to evaluate
/// * `approx` - The Zolotarev approximation parameters (for sign function)
///
/// # Returns
///
/// * The value of the step function approximation at `x`
pub fn evaluate_step<F>(x: F, approx: &ZolotarevApproximation<F>) -> F
where
    F: Float,
{
    let two = F::from(2.0).unwrap_or(F::one() + F::one());
    (F::one() + evaluate_rational(x, approx)) / two
}

/// Compute the Type I Zolotarev function Z_n(x, k).
///
/// The Type I Zolotarev function is defined as:
///   Z_n(x, k) = sn(n * K * arcsin(x) / arcsin(k), k_n)
///
/// where k_n is determined by the degree n and the modulus k.
/// This function is the optimal rational Chebyshev approximation
/// and arises naturally in filter design and spectral problems.
///
/// # Arguments
///
/// * `x` - Point at which to evaluate, should be in [-1, 1]
/// * `n` - Degree of the Zolotarev function
/// * `k` - Modulus parameter in (0, 1)
///
/// # Returns
///
/// * Value of Z_n(x, k)
pub fn zolotarev_type1(x: f64, n: usize, k: f64) -> LinalgResult<f64> {
    if n == 0 {
        return Ok(1.0);
    }
    if k <= 0.0 || k >= 1.0 {
        return Err(LinalgError::DomainError(format!(
            "Modulus k must be in (0, 1), got {}",
            k
        )));
    }

    let kk = complete_elliptic_k(k)?;
    let kp = (1.0 - k * k).sqrt();
    let kkp = complete_elliptic_k(kp)?;

    // Compute the transformed modulus k_n
    // For Type I: the nome q = exp(-pi * K'/K)
    // q_n = q^n, and k_n is derived from q_n
    let q = (-PI * kkp / kk).exp();
    let q_n = q.powi(n as i32);

    // Approximate k_n from q_n: k_n ≈ 4 * sqrt(q_n) for small q_n
    // For better accuracy, use the theta function relation
    let k_n = compute_modulus_from_nome(q_n)?;

    // u = n * K(k_n) * asin(x) / asin(k)  ... but we need K with respect to k
    // More precisely: u = n * K_n * F(arcsin(x), k) / K
    // where F is the incomplete elliptic integral
    // For simplicity, use the direct evaluation:
    let asin_x = x.asin();
    let asin_k = k.asin();

    if asin_k.abs() < 1e-15 {
        return Err(LinalgError::DomainError(
            "k too small for Type I evaluation".to_string(),
        ));
    }

    let kk_n = complete_elliptic_k(k_n)?;
    let u = n as f64 * kk_n * asin_x / asin_k;

    jacobi_sn(u, k_n)
}

/// Compute the Type III Zolotarev function.
///
/// The Type III function is the optimal rational approximation of the sign function
/// and is the basis of the `zolotarev_sign` function. It provides the best uniform
/// rational approximation to sign(x) on [-1, -delta] ∪ [delta, 1].
///
/// This function evaluates Z^{III}_n(x, delta) directly via Jacobi elliptic functions.
///
/// # Arguments
///
/// * `x` - Point at which to evaluate
/// * `n` - Degree
/// * `delta` - Gap parameter in (0, 1)
///
/// # Returns
///
/// * Value of the Type III Zolotarev function
pub fn zolotarev_type3(x: f64, n: usize, delta: f64) -> LinalgResult<f64> {
    if delta <= 0.0 || delta >= 1.0 {
        return Err(LinalgError::DomainError(format!(
            "Delta must be in (0, 1), got {}",
            delta
        )));
    }

    // Construct the rational approximation and evaluate
    let approx = zolotarev_sign::<f64>(n, delta)?;
    Ok(evaluate_rational(x, &approx))
}

/// Compute the elliptic modulus k from the nome q using theta functions.
///
/// k = (theta_2(q) / theta_3(q))^2
/// where theta_2(q) = 2 * sum_{n=0}^{inf} q^{(n+1/2)^2}
///       theta_3(q) = 1 + 2 * sum_{n=1}^{inf} q^{n^2}
fn compute_modulus_from_nome(q: f64) -> LinalgResult<f64> {
    if !(0.0..1.0).contains(&q) {
        return Err(LinalgError::DomainError(format!(
            "Nome q must be in [0, 1), got {}",
            q
        )));
    }

    if q < 1e-15 {
        return Ok(4.0 * q.sqrt());
    }

    // Compute theta_2 and theta_3
    let mut theta2 = 0.0;
    for nn in 0..100 {
        let exponent = (nn as f64 + 0.5) * (nn as f64 + 0.5);
        let term = q.powf(exponent);
        if term < 1e-16 {
            break;
        }
        theta2 += term;
    }
    theta2 *= 2.0;

    let mut theta3 = 1.0;
    for nn in 1..100 {
        let exponent = (nn * nn) as f64;
        let term = q.powf(exponent);
        if term < 1e-16 {
            break;
        }
        theta3 += 2.0 * term;
    }

    if theta3.abs() < 1e-15 {
        return Err(LinalgError::ComputationError(
            "theta3 is too small".to_string(),
        ));
    }

    let ratio = theta2 / theta3;
    Ok(ratio * ratio)
}

/// Compute Zolotarev rational approximation poles and residues
/// for use as a rational filter in eigenvalue problems.
///
/// Given an eigenvalue interval [a, b] within a larger spectral range [lambda_min, lambda_max],
/// this function computes the optimal rational filter that projects onto the eigenspace
/// corresponding to eigenvalues in [a, b].
///
/// # Arguments
///
/// * `degree` - Degree of the rational approximation
/// * `interval_lower` - Lower bound of target interval
/// * `interval_upper` - Upper bound of target interval
/// * `spectrum_lower` - Lower bound of the full spectrum
/// * `spectrum_upper` - Upper bound of the full spectrum
///
/// # Returns
///
/// * Tuple of (shifts, weights) for the rational filter:
///   At each shift z_j, solve (z_j*I - A) * X = Y, then Q = sum(w_j * X_j)
pub fn zolotarev_filter<F>(
    degree: usize,
    interval_lower: f64,
    interval_upper: f64,
    spectrum_lower: f64,
    spectrum_upper: f64,
) -> ZolotarevFilterResult<F>
where
    F: Float,
{
    if interval_lower >= interval_upper {
        return Err(LinalgError::DomainError(
            "Interval lower bound must be less than upper bound".to_string(),
        ));
    }

    if spectrum_lower >= spectrum_upper {
        return Err(LinalgError::DomainError(
            "Spectrum lower bound must be less than upper bound".to_string(),
        ));
    }

    // Map [a, b] within [lambda_min, lambda_max] to the sign function domain
    let center = (interval_lower + interval_upper) / 2.0;
    let half_width = (interval_upper - interval_lower) / 2.0;

    // delta = half_width / max(|lambda_max - center|, |center - lambda_min|)
    let range = (spectrum_upper - center)
        .abs()
        .max((center - spectrum_lower).abs());

    if range < 1e-15 {
        return Err(LinalgError::DomainError(
            "Spectrum range is too small".to_string(),
        ));
    }

    let delta = (half_width / range).min(0.999);
    let delta = delta.max(0.001);

    let approx = zolotarev_sign::<f64>(degree, delta)?;

    // Map poles back to the original spectral domain
    let mut shifts = Vec::with_capacity(degree);
    let mut weights = Vec::with_capacity(degree);

    for j in 0..approx.degree {
        let (_pole_re, pole_im) = approx.poles[j];

        // The shift in the original domain: z_j = center + range * i * pole_im
        let shift_re = F::from(center).unwrap_or(F::zero());
        let shift_im = F::from(range * pole_im).unwrap_or(F::zero());
        shifts.push((shift_re, shift_im));

        let (alpha_re, _) = approx.residues[j];
        weights.push(F::from(alpha_re / range).unwrap_or(F::zero()));
    }

    Ok((shifts, weights))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_zolotarev_sign_accuracy_vs_degree() {
        // Higher degree should give better approximation
        let delta = 0.1;

        let approx_4 = zolotarev_sign::<f64>(4, delta).expect("degree 4 failed");
        let approx_8 = zolotarev_sign::<f64>(8, delta).expect("degree 8 failed");
        let approx_16 = zolotarev_sign::<f64>(16, delta).expect("degree 16 failed");

        // Max error should decrease with increasing degree
        assert!(
            approx_8.max_error < approx_4.max_error,
            "degree 8 error ({}) should be less than degree 4 error ({})",
            approx_8.max_error,
            approx_4.max_error
        );
        assert!(
            approx_16.max_error < approx_8.max_error,
            "degree 16 error ({}) should be less than degree 8 error ({})",
            approx_16.max_error,
            approx_8.max_error
        );
    }

    #[test]
    fn test_zolotarev_poles_structure() {
        let approx = zolotarev_sign::<f64>(6, 0.2).expect("Zolotarev failed");

        assert_eq!(approx.poles.len(), 6);
        assert_eq!(approx.residues.len(), 6);

        // Poles should be purely imaginary (real part = 0)
        for (j, &(re, im)) in approx.poles.iter().enumerate() {
            assert!(
                re.abs() < 1e-14,
                "Pole {} real part should be 0, got {}",
                j,
                re
            );
            assert!(
                im > 0.0,
                "Pole {} imaginary part should be positive, got {}",
                j,
                im
            );
        }

        // Poles should be distinct
        for i in 0..approx.poles.len() {
            for j in (i + 1)..approx.poles.len() {
                let diff = (approx.poles[i].1 - approx.poles[j].1).abs();
                assert!(
                    diff > 1e-10,
                    "Poles {} and {} should be distinct, diff = {}",
                    i,
                    j,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_evaluate_rational_on_test_points() {
        let approx = zolotarev_sign::<f64>(8, 0.1).expect("Zolotarev failed");

        // Test on positive points: r(x) should be close to 1
        let test_points = [0.2, 0.3, 0.5, 0.7, 0.9, 1.0];
        for &x in &test_points {
            let val = evaluate_rational(x, &approx);
            assert!(
                (val - 1.0).abs() < 0.1, // generous bound for degree 8
                "r({}) = {} should be close to 1.0",
                x,
                val
            );
        }

        // Test on negative points: r(x) should be close to -1
        for &x in &test_points {
            let val = evaluate_rational(-x, &approx);
            assert!(
                (val + 1.0).abs() < 0.1,
                "r(-{}) = {} should be close to -1.0",
                x,
                val
            );
        }

        // Test antisymmetry: r(-x) = -r(x)
        for &x in &test_points {
            let val_pos = evaluate_rational(x, &approx);
            let val_neg = evaluate_rational(-x, &approx);
            assert_relative_eq!(val_neg, -val_pos, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_evaluate_step_function() {
        let approx = zolotarev_sign::<f64>(8, 0.1).expect("Zolotarev failed");

        // h(x) for positive x should be close to 1
        for &x in &[0.2, 0.5, 1.0] {
            let val = evaluate_step(x, &approx);
            assert!(
                (val - 1.0).abs() < 0.1,
                "step({}) = {} should be close to 1.0",
                x,
                val
            );
        }

        // h(x) for negative x should be close to 0
        for &x in &[0.2, 0.5, 1.0] {
            let val = evaluate_step(-x, &approx);
            assert!(
                val.abs() < 0.1,
                "step(-{}) = {} should be close to 0.0",
                x,
                val
            );
        }
    }

    #[test]
    fn test_zolotarev_sign_invalid_inputs() {
        // degree 0
        assert!(zolotarev_sign::<f64>(0, 0.5).is_err());

        // delta out of range
        assert!(zolotarev_sign::<f64>(4, 0.0).is_err());
        assert!(zolotarev_sign::<f64>(4, 1.0).is_err());
        assert!(zolotarev_sign::<f64>(4, -0.1).is_err());
        assert!(zolotarev_sign::<f64>(4, 1.5).is_err());
    }

    #[test]
    fn test_complete_elliptic_k() {
        // K(0) = pi/2
        let k0 = complete_elliptic_k(0.0).expect("K(0) failed");
        assert_relative_eq!(k0, PI / 2.0, epsilon = 1e-12);

        // K(1/sqrt(2)) = Gamma(1/4)^2 / (4*sqrt(pi)) ≈ 1.8540746773
        let k_half = complete_elliptic_k(1.0 / 2.0_f64.sqrt()).expect("K(1/sqrt(2)) failed");
        assert_relative_eq!(k_half, 1.8540746773013719, epsilon = 1e-8);
    }

    #[test]
    fn test_jacobi_sn() {
        // sn(0, k) = 0
        let val = jacobi_sn(0.0, 0.5).expect("sn(0, 0.5) failed");
        assert!(val.abs() < 1e-14, "sn(0, 0.5) = {} should be 0", val);

        // sn(K, k) = 1
        let kk = complete_elliptic_k(0.5).expect("K(0.5) failed");
        let val_k = jacobi_sn(kk, 0.5).expect("sn(K, 0.5) failed");
        assert_relative_eq!(val_k, 1.0, epsilon = 1e-8);

        // When k → 0, sn(u, k) → sin(u)
        let val_sin = jacobi_sn(1.0, 1e-10).expect("sn(1, ~0) failed");
        assert_relative_eq!(val_sin, 1.0_f64.sin(), epsilon = 1e-6);
    }

    #[test]
    fn test_zolotarev_type3() {
        let delta = 0.2;
        let n = 6;

        // Type III should approximate sign function
        let val_pos = zolotarev_type3(0.5, n, delta).expect("type3 failed at 0.5");
        assert!(
            (val_pos - 1.0).abs() < 0.1,
            "Z^III(0.5) = {} should be ~1.0",
            val_pos
        );

        let val_neg = zolotarev_type3(-0.5, n, delta).expect("type3 failed at -0.5");
        assert!(
            (val_neg + 1.0).abs() < 0.1,
            "Z^III(-0.5) = {} should be ~-1.0",
            val_neg
        );

        // Antisymmetry
        assert_relative_eq!(val_neg, -val_pos, epsilon = 1e-10);
    }

    #[test]
    fn test_zolotarev_filter() {
        let (shifts, weights) =
            zolotarev_filter::<f64>(4, 2.0, 4.0, 0.0, 10.0).expect("filter failed");

        assert_eq!(shifts.len(), 4);
        assert_eq!(weights.len(), 4);

        // Shifts should be centered around the interval midpoint
        let center = 3.0;
        for &(re, _im) in &shifts {
            assert_relative_eq!(re, center, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_zolotarev_residues_real() {
        // For the sign function approximation on a symmetric domain,
        // residues should be real (imaginary part = 0)
        let approx = zolotarev_sign::<f64>(6, 0.3).expect("Zolotarev failed");

        for (j, &(_re, im)) in approx.residues.iter().enumerate() {
            assert!(
                im.abs() < 1e-14,
                "Residue {} imaginary part should be 0, got {}",
                j,
                im
            );
        }
    }
}
