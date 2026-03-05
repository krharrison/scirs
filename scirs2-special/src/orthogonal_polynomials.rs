//! Extended orthogonal polynomial support
//!
//! This module provides advanced orthogonal polynomial functionality beyond the
//! basic evaluators in `orthogonal.rs` and the derivatives/quadrature in `orthogonal_ext/`.
//!
//! ## Features
//!
//! - **Polynomial roots**: Compute roots of orthogonal polynomials via the Golub-Welsch
//!   algorithm (eigenvalue decomposition of companion tridiagonal matrices)
//! - **Batch evaluation**: Evaluate polynomials at multiple points efficiently
//! - **Coefficient extraction**: Get explicit polynomial coefficients
//! - **Stable recurrence**: Numerically stable three-term recurrence evaluation
//! - **Higher-order derivatives**: Compute k-th derivatives of orthogonal polynomials
//!
//! ## Mathematical Background
//!
//! All classical orthogonal polynomials satisfy a three-term recurrence relation:
//! ```text
//! p_{n+1}(x) = (A_n x + B_n) p_n(x) - C_n p_{n-1}(x)
//! ```
//! where A_n, B_n, C_n depend on the polynomial family and degree.
//!
//! The roots of p_n(x) are the eigenvalues of the symmetric tridiagonal Jacobi
//! matrix constructed from the recurrence coefficients. This is the Golub-Welsch
//! algorithm, which provides both roots and quadrature weights simultaneously.

use crate::error::{SpecialError, SpecialResult};
use crate::orthogonal::{
    chebyshev, gegenbauer, hermite, hermite_prob, jacobi, laguerre, laguerre_generalized, legendre,
};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

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
// Polynomial Roots via Golub-Welsch Algorithm
// ========================================================================

/// Compute the roots of the Legendre polynomial P_n(x).
///
/// Uses the Golub-Welsch algorithm: the roots are eigenvalues of the
/// symmetric tridiagonal Jacobi matrix for the Legendre recurrence.
///
/// # Arguments
/// * `n` - Degree of the polynomial (must be >= 1)
///
/// # Returns
/// Vector of n roots sorted in ascending order, all in (-1, 1).
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_polynomials::legendre_roots;
/// let roots = legendre_roots(3).expect("failed");
/// assert_eq!(roots.len(), 3);
/// // P_3 roots: 0, +/- sqrt(3/5)
/// assert!(roots[1].abs() < 1e-14);
/// ```
pub fn legendre_roots(n: usize) -> SpecialResult<Vec<f64>> {
    if n == 0 {
        return Err(SpecialError::ValueError(
            "Degree must be >= 1 for root computation".to_string(),
        ));
    }

    // Jacobi matrix for Legendre: alpha_i = 0, beta_i = i / sqrt(4i^2 - 1)
    let diag = vec![0.0f64; n];
    let mut sub_diag = vec![0.0f64; n.saturating_sub(1)];

    for i in 0..n.saturating_sub(1) {
        let ip1 = (i + 1) as f64;
        sub_diag[i] = ip1 / (4.0 * ip1 * ip1 - 1.0).sqrt();
    }

    let eigenvalues = symmetric_tridiag_eigenvalues(&diag, &sub_diag)?;
    let mut roots = eigenvalues;
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(roots)
}

/// Compute the roots of the Chebyshev polynomial T_n(x) of the first kind.
///
/// The roots of T_n(x) have closed-form expressions:
/// ```text
/// x_k = cos((2k-1) pi / (2n)), k = 1, ..., n
/// ```
///
/// # Arguments
/// * `n` - Degree of the polynomial (must be >= 1)
///
/// # Returns
/// Vector of n roots sorted in ascending order.
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_polynomials::chebyshev_t_roots;
/// let roots = chebyshev_t_roots(4).expect("failed");
/// assert_eq!(roots.len(), 4);
/// for r in &roots {
///     assert!(r.abs() <= 1.0);
/// }
/// ```
pub fn chebyshev_t_roots(n: usize) -> SpecialResult<Vec<f64>> {
    if n == 0 {
        return Err(SpecialError::ValueError(
            "Degree must be >= 1 for root computation".to_string(),
        ));
    }

    let mut roots: Vec<f64> = (1..=n)
        .map(|k| {
            let angle = (2 * k - 1) as f64 * std::f64::consts::PI / (2.0 * n as f64);
            angle.cos()
        })
        .collect();

    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(roots)
}

/// Compute the roots of the Chebyshev polynomial U_n(x) of the second kind.
///
/// The roots of U_n(x) have closed-form expressions:
/// ```text
/// x_k = cos(k pi / (n+1)), k = 1, ..., n
/// ```
///
/// # Arguments
/// * `n` - Degree of the polynomial (must be >= 1)
///
/// # Returns
/// Vector of n roots sorted in ascending order.
pub fn chebyshev_u_roots(n: usize) -> SpecialResult<Vec<f64>> {
    if n == 0 {
        return Err(SpecialError::ValueError(
            "Degree must be >= 1 for root computation".to_string(),
        ));
    }

    let mut roots: Vec<f64> = (1..=n)
        .map(|k| {
            let angle = k as f64 * std::f64::consts::PI / (n as f64 + 1.0);
            angle.cos()
        })
        .collect();

    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(roots)
}

/// Compute the roots of the Hermite polynomial H_n(x) (physicist's).
///
/// Uses the Golub-Welsch algorithm with the Hermite Jacobi matrix:
/// alpha_i = 0, beta_i = sqrt(i/2)
///
/// # Arguments
/// * `n` - Degree of the polynomial (must be >= 1)
///
/// # Returns
/// Vector of n roots sorted in ascending order.
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_polynomials::hermite_roots;
/// let roots = hermite_roots(2).expect("failed");
/// // H_2(x) = 4x^2 - 2, roots at +/- 1/sqrt(2)
/// assert!((roots[0] + (0.5f64).sqrt()).abs() < 1e-12);
/// assert!((roots[1] - (0.5f64).sqrt()).abs() < 1e-12);
/// ```
pub fn hermite_roots(n: usize) -> SpecialResult<Vec<f64>> {
    if n == 0 {
        return Err(SpecialError::ValueError(
            "Degree must be >= 1 for root computation".to_string(),
        ));
    }

    let diag = vec![0.0f64; n];
    let mut sub_diag = vec![0.0f64; n.saturating_sub(1)];

    for i in 0..n.saturating_sub(1) {
        sub_diag[i] = ((i + 1) as f64 / 2.0).sqrt();
    }

    let eigenvalues = symmetric_tridiag_eigenvalues(&diag, &sub_diag)?;
    let mut roots = eigenvalues;
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(roots)
}

/// Compute the roots of the probabilist's Hermite polynomial He_n(x).
///
/// Uses the Golub-Welsch algorithm with the probabilist's Hermite Jacobi matrix:
/// alpha_i = 0, beta_i = sqrt(i)
///
/// # Arguments
/// * `n` - Degree of the polynomial (must be >= 1)
///
/// # Returns
/// Vector of n roots sorted in ascending order.
pub fn hermite_prob_roots(n: usize) -> SpecialResult<Vec<f64>> {
    if n == 0 {
        return Err(SpecialError::ValueError(
            "Degree must be >= 1 for root computation".to_string(),
        ));
    }

    let diag = vec![0.0f64; n];
    let mut sub_diag = vec![0.0f64; n.saturating_sub(1)];

    for i in 0..n.saturating_sub(1) {
        sub_diag[i] = ((i + 1) as f64).sqrt();
    }

    let eigenvalues = symmetric_tridiag_eigenvalues(&diag, &sub_diag)?;
    let mut roots = eigenvalues;
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(roots)
}

/// Compute the roots of the Laguerre polynomial L_n(x).
///
/// Uses the Golub-Welsch algorithm with the Laguerre Jacobi matrix:
/// alpha_i = 2i + 1, beta_i = i
///
/// # Arguments
/// * `n` - Degree of the polynomial (must be >= 1)
///
/// # Returns
/// Vector of n positive roots sorted in ascending order.
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_polynomials::laguerre_roots;
/// let roots = laguerre_roots(3).expect("failed");
/// assert_eq!(roots.len(), 3);
/// for r in &roots {
///     assert!(*r > 0.0);
/// }
/// ```
pub fn laguerre_roots(n: usize) -> SpecialResult<Vec<f64>> {
    laguerre_generalized_roots(n, 0.0)
}

/// Compute the roots of the generalized Laguerre polynomial L_n^(alpha)(x).
///
/// Uses the Golub-Welsch algorithm with the generalized Laguerre Jacobi matrix:
/// alpha_i = 2i + alpha + 1, beta_i = sqrt(i * (i + alpha))
///
/// # Arguments
/// * `n` - Degree of the polynomial (must be >= 1)
/// * `alpha` - Parameter (must be > -1)
///
/// # Returns
/// Vector of n positive roots sorted in ascending order.
pub fn laguerre_generalized_roots(n: usize, alpha: f64) -> SpecialResult<Vec<f64>> {
    if n == 0 {
        return Err(SpecialError::ValueError(
            "Degree must be >= 1 for root computation".to_string(),
        ));
    }

    if alpha <= -1.0 {
        return Err(SpecialError::DomainError(
            "alpha must be > -1 for generalized Laguerre roots".to_string(),
        ));
    }

    let mut diag = vec![0.0f64; n];
    let mut sub_diag = vec![0.0f64; n.saturating_sub(1)];

    for i in 0..n {
        diag[i] = 2.0 * (i as f64) + alpha + 1.0;
    }

    for i in 0..n.saturating_sub(1) {
        let ip1 = (i + 1) as f64;
        sub_diag[i] = (ip1 * (ip1 + alpha)).sqrt();
    }

    let eigenvalues = symmetric_tridiag_eigenvalues(&diag, &sub_diag)?;
    let mut roots = eigenvalues;
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(roots)
}

/// Compute the roots of the Jacobi polynomial P_n^(alpha,beta)(x).
///
/// Uses the Golub-Welsch algorithm with the Jacobi recurrence coefficients.
///
/// # Arguments
/// * `n` - Degree of the polynomial (must be >= 1)
/// * `alpha` - First parameter (must be > -1)
/// * `beta` - Second parameter (must be > -1)
///
/// # Returns
/// Vector of n roots in (-1, 1) sorted in ascending order.
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_polynomials::jacobi_roots;
/// // Jacobi(0,0) = Legendre
/// let roots = jacobi_roots(3, 0.0, 0.0).expect("failed");
/// assert_eq!(roots.len(), 3);
/// assert!(roots[1].abs() < 1e-12); // middle root at 0
/// ```
pub fn jacobi_roots(n: usize, alpha: f64, beta: f64) -> SpecialResult<Vec<f64>> {
    if n == 0 {
        return Err(SpecialError::ValueError(
            "Degree must be >= 1 for root computation".to_string(),
        ));
    }

    if alpha <= -1.0 || beta <= -1.0 {
        return Err(SpecialError::DomainError(
            "alpha and beta must be > -1 for Jacobi roots".to_string(),
        ));
    }

    let mut diag = vec![0.0f64; n];
    let mut sub_diag = vec![0.0f64; n.saturating_sub(1)];

    // Jacobi recurrence coefficients for the monic three-term recurrence
    for i in 0..n {
        let i_f = i as f64;
        let denom = (2.0 * i_f + alpha + beta) * (2.0 * i_f + alpha + beta + 2.0);
        if denom.abs() < 1e-300 {
            diag[i] = 0.0;
        } else {
            diag[i] = (beta * beta - alpha * alpha) / denom;
        }
    }

    for i in 0..n.saturating_sub(1) {
        let ip1 = (i + 1) as f64;
        let numer = 4.0 * ip1 * (ip1 + alpha) * (ip1 + beta) * (ip1 + alpha + beta);
        let denom_base = 2.0 * ip1 + alpha + beta;
        let denom = denom_base * denom_base * (denom_base + 1.0) * (denom_base - 1.0);
        if denom.abs() < 1e-300 {
            sub_diag[i] = 0.0;
        } else {
            sub_diag[i] = (numer / denom).sqrt();
        }
    }

    let eigenvalues = symmetric_tridiag_eigenvalues(&diag, &sub_diag)?;
    let mut roots = eigenvalues;
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(roots)
}

/// Compute the roots of the Gegenbauer polynomial C_n^(lambda)(x).
///
/// Uses the Jacobi root-finding since C_n^(lambda) = special case of Jacobi
/// with alpha = beta = lambda - 1/2.
///
/// # Arguments
/// * `n` - Degree of the polynomial (must be >= 1)
/// * `lambda` - Parameter (must be > -1/2)
///
/// # Returns
/// Vector of n roots in (-1, 1) sorted in ascending order.
pub fn gegenbauer_roots(n: usize, lambda: f64) -> SpecialResult<Vec<f64>> {
    if lambda <= -0.5 {
        return Err(SpecialError::DomainError(
            "lambda must be > -1/2 for Gegenbauer roots".to_string(),
        ));
    }

    let alpha = lambda - 0.5;
    let beta = alpha;
    jacobi_roots(n, alpha, beta)
}

// ========================================================================
// Batch Evaluation
// ========================================================================

/// Evaluate Legendre polynomial P_n at multiple points.
///
/// # Arguments
/// * `n` - Degree of the polynomial
/// * `points` - Slice of evaluation points
///
/// # Returns
/// Vector of P_n(x) values for each x in points.
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_polynomials::legendre_batch;
/// let vals = legendre_batch(2, &[0.0, 0.5, 1.0]);
/// assert!((vals[0] - (-0.5)).abs() < 1e-14); // P_2(0) = -0.5
/// assert!((vals[2] - 1.0).abs() < 1e-14);    // P_2(1) = 1
/// ```
pub fn legendre_batch<F: Float + FromPrimitive + Debug>(n: usize, points: &[F]) -> Vec<F> {
    points.iter().map(|&x| legendre(n, x)).collect()
}

/// Evaluate Chebyshev polynomial T_n at multiple points.
pub fn chebyshev_t_batch<F: Float + FromPrimitive + Debug>(n: usize, points: &[F]) -> Vec<F> {
    points.iter().map(|&x| chebyshev(n, x, true)).collect()
}

/// Evaluate Chebyshev polynomial U_n at multiple points.
pub fn chebyshev_u_batch<F: Float + FromPrimitive + Debug>(n: usize, points: &[F]) -> Vec<F> {
    points.iter().map(|&x| chebyshev(n, x, false)).collect()
}

/// Evaluate physicist's Hermite polynomial H_n at multiple points.
pub fn hermite_batch<F: Float + FromPrimitive + Debug>(n: usize, points: &[F]) -> Vec<F> {
    points.iter().map(|&x| hermite(n, x)).collect()
}

/// Evaluate probabilist's Hermite polynomial He_n at multiple points.
pub fn hermite_prob_batch<F: Float + FromPrimitive + Debug>(n: usize, points: &[F]) -> Vec<F> {
    points.iter().map(|&x| hermite_prob(n, x)).collect()
}

/// Evaluate Laguerre polynomial L_n at multiple points.
pub fn laguerre_batch<F: Float + FromPrimitive + Debug>(n: usize, points: &[F]) -> Vec<F> {
    points.iter().map(|&x| laguerre(n, x)).collect()
}

/// Evaluate generalized Laguerre polynomial L_n^(alpha) at multiple points.
pub fn laguerre_generalized_batch<F: Float + FromPrimitive + Debug>(
    n: usize,
    alpha: F,
    points: &[F],
) -> Vec<F> {
    points
        .iter()
        .map(|&x| laguerre_generalized(n, alpha, x))
        .collect()
}

/// Evaluate Jacobi polynomial P_n^(alpha,beta) at multiple points.
pub fn jacobi_batch<F: Float + FromPrimitive + Debug>(
    n: usize,
    alpha: F,
    beta: F,
    points: &[F],
) -> Vec<F> {
    points.iter().map(|&x| jacobi(n, alpha, beta, x)).collect()
}

/// Evaluate Gegenbauer polynomial C_n^(lambda) at multiple points.
pub fn gegenbauer_batch<F: Float + FromPrimitive + Debug>(
    n: usize,
    lambda: F,
    points: &[F],
) -> Vec<F> {
    points.iter().map(|&x| gegenbauer(n, lambda, x)).collect()
}

// ========================================================================
// Polynomial Coefficients
// ========================================================================

/// Get the coefficients of the Legendre polynomial P_n(x).
///
/// Returns coefficients [a_0, a_1, ..., a_n] such that
/// P_n(x) = a_0 + a_1 x + a_2 x^2 + ... + a_n x^n.
///
/// # Arguments
/// * `n` - Degree of the polynomial
///
/// # Returns
/// Vector of n+1 coefficients.
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_polynomials::legendre_coefficients;
/// // P_2(x) = (3x^2 - 1)/2
/// let c = legendre_coefficients(2);
/// assert!((c[0] - (-0.5)).abs() < 1e-14);
/// assert!(c[1].abs() < 1e-14);
/// assert!((c[2] - 1.5).abs() < 1e-14);
/// ```
pub fn legendre_coefficients(n: usize) -> Vec<f64> {
    polynomial_coefficients_from_recurrence(n, |k| {
        // Legendre: (k+1) P_{k+1} = (2k+1) x P_k - k P_{k-1}
        // => P_{k+1} = ((2k+1)/(k+1)) x P_k - (k/(k+1)) P_{k-1}
        let k_f = k as f64;
        let a = (2.0 * k_f + 1.0) / (k_f + 1.0); // coefficient of x * P_k
        let b = 0.0; // no constant term in recurrence
        let c = k_f / (k_f + 1.0); // coefficient of P_{k-1}
        (a, b, c)
    })
}

/// Get the coefficients of the Chebyshev polynomial T_n(x).
///
/// Returns coefficients [a_0, a_1, ..., a_n].
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_polynomials::chebyshev_t_coefficients;
/// // T_2(x) = 2x^2 - 1
/// let c = chebyshev_t_coefficients(2);
/// assert!((c[0] - (-1.0)).abs() < 1e-14);
/// assert!(c[1].abs() < 1e-14);
/// assert!((c[2] - 2.0).abs() < 1e-14);
/// ```
pub fn chebyshev_t_coefficients(n: usize) -> Vec<f64> {
    polynomial_coefficients_from_recurrence(n, |k| {
        // T_1 = x (from T_0 = 1), then T_{k+1} = 2x T_k - T_{k-1} for k >= 1
        if k == 0 {
            (1.0, 0.0, 0.0)
        } else {
            (2.0, 0.0, 1.0)
        }
    })
}

/// Get the coefficients of the Hermite polynomial H_n(x) (physicist's).
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_polynomials::hermite_coefficients;
/// // H_2(x) = 4x^2 - 2
/// let c = hermite_coefficients(2);
/// assert!((c[0] - (-2.0)).abs() < 1e-14);
/// assert!(c[1].abs() < 1e-14);
/// assert!((c[2] - 4.0).abs() < 1e-14);
/// ```
pub fn hermite_coefficients(n: usize) -> Vec<f64> {
    polynomial_coefficients_from_recurrence(n, |k| {
        // H_{k+1} = 2x H_k - 2k H_{k-1}
        let k_f = k as f64;
        (2.0, 0.0, 2.0 * k_f)
    })
}

/// Get the coefficients of the probabilist's Hermite polynomial He_n(x).
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_polynomials::hermite_prob_coefficients;
/// // He_2(x) = x^2 - 1
/// let c = hermite_prob_coefficients(2);
/// assert!((c[0] - (-1.0)).abs() < 1e-14);
/// assert!(c[1].abs() < 1e-14);
/// assert!((c[2] - 1.0).abs() < 1e-14);
/// ```
pub fn hermite_prob_coefficients(n: usize) -> Vec<f64> {
    polynomial_coefficients_from_recurrence(n, |k| {
        // He_{k+1} = x He_k - k He_{k-1}
        let k_f = k as f64;
        (1.0, 0.0, k_f)
    })
}

// ========================================================================
// Higher-Order Derivatives
// ========================================================================

/// Compute the k-th derivative of the Legendre polynomial P_n(x).
///
/// Uses the identity for k-th derivatives via the recurrence:
/// d^k/dx^k P_n(x) can be computed using iterated derivative formulas.
///
/// # Arguments
/// * `n` - Degree of the polynomial
/// * `k` - Order of derivative
/// * `x` - Point of evaluation
///
/// # Returns
/// Value of the k-th derivative at x.
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_polynomials::legendre_kth_derivative;
/// // P_2(x) = (3x^2 - 1)/2, P''_2(x) = 3
/// assert!((legendre_kth_derivative(2, 2, 0.5f64) - 3.0).abs() < 1e-10);
/// ```
pub fn legendre_kth_derivative<F: Float + FromPrimitive + Debug>(n: usize, k: usize, x: F) -> F {
    if k == 0 {
        return legendre(n, x);
    }
    if k > n {
        return F::zero();
    }

    // Use numerical differentiation for higher orders
    // by applying the derivative relation iteratively:
    // P'_n = n(x P_n - P_{n-1}) / (x^2 - 1) (for |x| != 1)
    //
    // For a more stable approach, we use the fact that
    // d^k/dx^k P_n(x) = sum of products involving lower Legendre functions
    //
    // The simplest stable approach: use the relation
    // d/dx P_n^m = ... iteratively, or use finite differences
    // with carefully chosen step size.

    // For k=1, use the standard formula
    if k == 1 {
        return crate::orthogonal_ext::legendre_derivative(n, x);
    }

    // For higher k, use Rodrigues formula coefficient:
    // d^k/dx^k P_n(x) = (2n-1)!! / (n-k)! * ... (complicated)
    //
    // Instead, use finite differences with Richardson extrapolation
    // for robustness:
    let h_base = const_f64::<F>(1e-4);
    richardson_kth_derivative(|t| legendre(n, t), x, k, h_base)
}

/// Compute the k-th derivative of the Chebyshev polynomial T_n(x).
pub fn chebyshev_t_kth_derivative<F: Float + FromPrimitive + Debug>(n: usize, k: usize, x: F) -> F {
    if k == 0 {
        return chebyshev(n, x, true);
    }
    if k > n {
        return F::zero();
    }
    if k == 1 {
        return crate::orthogonal_ext::chebyshev_t_derivative(n, x);
    }

    let h_base = const_f64::<F>(1e-4);
    richardson_kth_derivative(|t| chebyshev(n, t, true), x, k, h_base)
}

/// Compute the k-th derivative of the Hermite polynomial H_n(x).
///
/// Uses the identity: H^(k)_n(x) = 2^k n! / (n-k)! H_{n-k}(x)
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_polynomials::hermite_kth_derivative;
/// // H_3(x) = 8x^3 - 12x, H'_3(x) = 24x^2 - 12, H''_3(x) = 48x, H'''_3(x) = 48
/// assert!((hermite_kth_derivative(3, 3, 0.5f64) - 48.0).abs() < 1e-8);
/// ```
pub fn hermite_kth_derivative<F: Float + FromPrimitive + Debug>(n: usize, k: usize, x: F) -> F {
    if k == 0 {
        return hermite(n, x);
    }
    if k > n {
        return F::zero();
    }

    // H^(k)_n(x) = 2^k * n!/(n-k)! * H_{n-k}(x)
    let two_pow_k = const_f64::<F>(2.0_f64.powi(k as i32));
    let mut falling_factorial = F::one();
    for i in 0..k {
        falling_factorial = falling_factorial * F::from(n - i).unwrap_or(F::zero());
    }

    two_pow_k * falling_factorial * hermite(n - k, x)
}

/// Compute the k-th derivative of the probabilist's Hermite polynomial He_n(x).
///
/// Uses: He^(k)_n(x) = n!/(n-k)! * He_{n-k}(x)
pub fn hermite_prob_kth_derivative<F: Float + FromPrimitive + Debug>(
    n: usize,
    k: usize,
    x: F,
) -> F {
    if k == 0 {
        return hermite_prob(n, x);
    }
    if k > n {
        return F::zero();
    }

    let mut falling_factorial = F::one();
    for i in 0..k {
        falling_factorial = falling_factorial * F::from(n - i).unwrap_or(F::zero());
    }

    falling_factorial * hermite_prob(n - k, x)
}

// ========================================================================
// Recurrence Relation Evaluation (Enhanced Stability)
// ========================================================================

/// Evaluate a general three-term recurrence polynomial at x.
///
/// Given the recurrence:
/// p_{n+1}(x) = (a_n * x + b_n) * p_n(x) - c_n * p_{n-1}(x)
///
/// with p_0 = 1 and p_1 = a_0 * x + b_0, compute p_n(x).
///
/// # Arguments
/// * `n` - Degree of the polynomial
/// * `x` - Evaluation point
/// * `recurrence_fn` - Function returning (a_k, b_k, c_k) for each k
///
/// # Returns
/// Value of p_n(x).
pub fn eval_three_term_recurrence<F: Float + FromPrimitive + Debug>(
    n: usize,
    x: F,
    recurrence_fn: impl Fn(usize) -> (F, F, F),
) -> F {
    if n == 0 {
        return F::one();
    }

    let (a0, b0, _c0) = recurrence_fn(0);
    let mut p_prev = F::one();
    let mut p_curr = a0 * x + b0;

    for k in 1..n {
        let (a_k, b_k, c_k) = recurrence_fn(k);
        let p_next = (a_k * x + b_k) * p_curr - c_k * p_prev;
        p_prev = p_curr;
        p_curr = p_next;
    }

    p_curr
}

/// Evaluate all polynomial values p_0(x), p_1(x), ..., p_n(x) simultaneously.
///
/// This is useful when you need multiple degrees at the same point (e.g., for
/// spectral methods or expansions).
///
/// # Arguments
/// * `n_max` - Maximum degree
/// * `x` - Evaluation point
/// * `recurrence_fn` - Function returning (a_k, b_k, c_k) for each k
///
/// # Returns
/// Vector of length n_max + 1 with values [p_0(x), p_1(x), ..., p_{n_max}(x)].
pub fn eval_all_degrees<F: Float + FromPrimitive + Debug>(
    n_max: usize,
    x: F,
    recurrence_fn: impl Fn(usize) -> (F, F, F),
) -> Vec<F> {
    let mut result = Vec::with_capacity(n_max + 1);
    result.push(F::one());

    if n_max == 0 {
        return result;
    }

    let (a0, b0, _c0) = recurrence_fn(0);
    result.push(a0 * x + b0);

    for k in 1..n_max {
        let (a_k, b_k, c_k) = recurrence_fn(k);
        let p_next = (a_k * x + b_k) * result[k] - c_k * result[k - 1];
        result.push(p_next);
    }

    result
}

/// Evaluate all Legendre polynomials P_0(x), ..., P_n(x) simultaneously.
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_polynomials::legendre_all;
/// let vals = legendre_all(3, 0.5f64);
/// assert!((vals[0] - 1.0).abs() < 1e-14);    // P_0
/// assert!((vals[1] - 0.5).abs() < 1e-14);     // P_1
/// ```
pub fn legendre_all<F: Float + FromPrimitive + Debug>(n_max: usize, x: F) -> Vec<F> {
    eval_all_degrees(n_max, x, |k| {
        let k_f = F::from(k).unwrap_or(F::zero());
        let a = (k_f + k_f + F::one()) / (k_f + F::one());
        let b = F::zero();
        let c = k_f / (k_f + F::one());
        (a, b, c)
    })
}

/// Evaluate all Chebyshev polynomials T_0(x), ..., T_n(x) simultaneously.
pub fn chebyshev_t_all<F: Float + FromPrimitive + Debug>(n_max: usize, x: F) -> Vec<F> {
    eval_all_degrees(n_max, x, |k| {
        let two = F::one() + F::one();
        // T_1 = x (from T_0 = 1), then T_{k+1} = 2x T_k - T_{k-1} for k >= 1
        if k == 0 {
            (F::one(), F::zero(), F::zero())
        } else {
            (two, F::zero(), F::one())
        }
    })
}

/// Evaluate all Hermite polynomials H_0(x), ..., H_n(x) simultaneously.
pub fn hermite_all<F: Float + FromPrimitive + Debug>(n_max: usize, x: F) -> Vec<F> {
    eval_all_degrees(n_max, x, |k| {
        let two = F::one() + F::one();
        let k_f = F::from(k).unwrap_or(F::zero());
        (two, F::zero(), two * k_f)
    })
}

// ========================================================================
// Clenshaw Algorithm for Evaluating Polynomial Expansions
// ========================================================================

/// Evaluate a Chebyshev expansion using the Clenshaw algorithm.
///
/// Given coefficients c_0, c_1, ..., c_n, computes:
/// S(x) = sum_{k=0}^{n} c_k T_k(x)
///
/// The Clenshaw algorithm evaluates this in O(n) time with excellent
/// numerical stability.
///
/// # Arguments
/// * `coeffs` - Chebyshev expansion coefficients [c_0, c_1, ..., c_n]
/// * `x` - Evaluation point (typically in [-1, 1])
///
/// # Returns
/// Value of the expansion at x.
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_polynomials::clenshaw_chebyshev;
/// // f(x) = 1*T_0 + 0*T_1 + (-1)*T_2 = 1 + 0 - (2x^2 - 1) = 2 - 2x^2
/// let coeffs = [1.0, 0.0, -1.0];
/// let val = clenshaw_chebyshev(&coeffs, 0.5f64);
/// assert!((val - 1.5).abs() < 1e-14); // 2 - 2*0.25 = 1.5
/// ```
pub fn clenshaw_chebyshev<F: Float + FromPrimitive + Debug>(coeffs: &[F], x: F) -> F {
    let n = coeffs.len();
    if n == 0 {
        return F::zero();
    }
    if n == 1 {
        return coeffs[0];
    }

    let two = F::one() + F::one();
    let mut b_kplus2 = F::zero();
    let mut b_kplus1 = F::zero();

    for k in (1..n).rev() {
        let b_k = two * x * b_kplus1 - b_kplus2 + coeffs[k];
        b_kplus2 = b_kplus1;
        b_kplus1 = b_k;
    }

    // Final step: b_0 = x * b_1 - b_2 + c_0
    x * b_kplus1 - b_kplus2 + coeffs[0]
}

/// Evaluate a Legendre expansion using the Clenshaw algorithm.
///
/// Given coefficients c_0, c_1, ..., c_n, computes:
/// S(x) = sum_{k=0}^{n} c_k P_k(x)
///
/// # Arguments
/// * `coeffs` - Legendre expansion coefficients
/// * `x` - Evaluation point
pub fn clenshaw_legendre<F: Float + FromPrimitive + Debug>(coeffs: &[F], x: F) -> F {
    let n = coeffs.len();
    if n == 0 {
        return F::zero();
    }
    if n == 1 {
        return coeffs[0];
    }

    // Legendre Clenshaw:
    // b_{n+1} = b_{n+2} = 0
    // b_k = ((2k+1)/(k+1)) x b_{k+1} - ((k+1)/(k+2)) b_{k+2} + c_k
    // S(x) = b_0

    let mut b_kplus2 = F::zero();
    let mut b_kplus1 = F::zero();

    for k in (0..n).rev() {
        let k_f = F::from(k).unwrap_or(F::zero());
        let alpha = (k_f + k_f + F::one()) / (k_f + F::one()) * x;
        let beta = if k + 2 < n {
            (k_f + F::one()) / (k_f + F::one() + F::one())
        } else {
            F::zero()
        };
        let b_k = alpha * b_kplus1 - beta * b_kplus2 + coeffs[k];
        b_kplus2 = b_kplus1;
        b_kplus1 = b_k;
    }

    b_kplus1
}

// ========================================================================
// Internal Helper Functions
// ========================================================================

/// Build polynomial coefficients from a three-term recurrence relation.
///
/// Given p_{k+1}(x) = a_k * x * p_k(x) + b_k * p_k(x) - c_k * p_{k-1}(x)
/// with p_0 = 1 and custom first-degree polynomial.
fn polynomial_coefficients_from_recurrence(
    n: usize,
    recurrence_fn: impl Fn(usize) -> (f64, f64, f64),
) -> Vec<f64> {
    if n == 0 {
        return vec![1.0];
    }

    // Store coefficients as vectors: coeffs[k] = coefficients of p_k
    let mut coeffs: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    coeffs.push(vec![1.0]); // p_0 = 1

    let (a0, b0, _c0) = recurrence_fn(0);
    // p_1(x) = a_0 * x + b_0
    coeffs.push(vec![b0, a0]);

    for k in 1..n {
        let (a_k, b_k, c_k) = recurrence_fn(k);
        let prev = &coeffs[k];
        let prev_prev = &coeffs[k - 1];

        let mut new_coeffs = vec![0.0; k + 2];

        // a_k * x * p_k(x): shift coefficients up by 1
        for (i, &c) in prev.iter().enumerate() {
            new_coeffs[i + 1] += a_k * c;
        }

        // b_k * p_k(x)
        for (i, &c) in prev.iter().enumerate() {
            new_coeffs[i] += b_k * c;
        }

        // -c_k * p_{k-1}(x)
        for (i, &c) in prev_prev.iter().enumerate() {
            new_coeffs[i] -= c_k * c;
        }

        coeffs.push(new_coeffs);
    }

    coeffs.into_iter().last().unwrap_or_else(|| vec![1.0])
}

/// Richardson extrapolation for k-th derivative via finite differences.
fn richardson_kth_derivative<F: Float + FromPrimitive + Debug>(
    f: impl Fn(F) -> F,
    x: F,
    k: usize,
    h_base: F,
) -> F {
    // Use central differences with Richardson extrapolation
    // for numerical stability
    let two = F::one() + F::one();

    // For k-th derivative, use the k-th order central difference formula
    // with step h, then refine with Richardson extrapolation

    let compute_diff = |h: F| -> F {
        // k-th central difference: sum_{j=0}^{k} (-1)^j C(k,j) f(x + (k/2 - j) h)
        let half_k = F::from(k).unwrap_or(F::zero()) / two;
        let mut result = F::zero();
        let mut binom = 1i64;
        for j in 0..=k {
            let sign = if j % 2 == 0 { F::one() } else { -F::one() };
            let point = x + (half_k - F::from(j).unwrap_or(F::zero())) * h;
            result = result + sign * F::from(binom).unwrap_or(F::zero()) * f(point);
            if j < k {
                binom = binom * (k - j) as i64 / (j + 1) as i64;
            }
        }
        let h_pow_k = h.powi(k as i32);
        if h_pow_k.abs() < F::from(1e-300).unwrap_or(F::zero()) {
            return F::zero();
        }
        result / h_pow_k
    };

    // Two levels of Richardson extrapolation
    let d1 = compute_diff(h_base);
    let d2 = compute_diff(h_base / two);
    let four = two * two;

    // Richardson: D = (4 * D(h/2) - D(h)) / 3
    (four * d2 - d1) / (four - F::one())
}

/// Symmetric tridiagonal eigenvalue solver (eigenvalues only).
///
/// Uses the implicit QL algorithm with shifts for symmetric tridiagonal matrices.
/// This is a standard implementation based on LAPACK's DSTEQR algorithm.
fn symmetric_tridiag_eigenvalues(diag: &[f64], sub_diag: &[f64]) -> SpecialResult<Vec<f64>> {
    let n = diag.len();
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        return Ok(vec![diag[0]]);
    }

    let mut d = diag.to_vec();
    let mut e = vec![0.0f64; n];
    for i in 0..n - 1 {
        e[i] = sub_diag[i];
    }
    e[n - 1] = 0.0;

    // QL iteration with implicit shift (based on Numerical Recipes tqli)
    for l in 0..n {
        let mut iteration = 0u32;
        let max_iter = 300u32;

        loop {
            // Find small sub-diagonal element
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                // Use a relative+absolute tolerance
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }

            if m == l {
                break; // Eigenvalue found
            }

            iteration += 1;
            if iteration > max_iter {
                return Err(SpecialError::ConvergenceError(format!(
                    "Tridiagonal eigenvalue computation did not converge after {max_iter} iterations for l={l}"
                )));
            }

            // Form shift
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = (g * g + 1.0).sqrt();
            // g = d[m] - d[l] + e[l] / (g + sign(r, g))
            g = d[m] - d[l] + e[l] / (g + if g >= 0.0 { r } else { -r });

            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;

            // QL sweep from m-1 down to l
            let mut i = m;
            let mut early_break = false;
            while i > l {
                i -= 1;
                let f = s * e[i];
                let b = c * e[i];

                // Givens rotation
                r = (f * f + g * g).sqrt();
                e[i + 1] = r;

                if r.abs() < 1e-300 {
                    // Lucky breakdown: recover
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    early_break = true;
                    break;
                }

                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }

            if !early_break {
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        }
    }

    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ====== Root-finding tests ======

    #[test]
    fn test_legendre_roots_degree1() {
        let roots = legendre_roots(1).expect("failed");
        assert_eq!(roots.len(), 1);
        assert!(roots[0].abs() < 1e-14);
    }

    #[test]
    fn test_legendre_roots_degree2() {
        let roots = legendre_roots(2).expect("failed");
        assert_eq!(roots.len(), 2);
        // P_2 roots: +/- 1/sqrt(3)
        let expected = 1.0 / 3.0_f64.sqrt();
        assert!((roots[0] + expected).abs() < 1e-12);
        assert!((roots[1] - expected).abs() < 1e-12);
    }

    #[test]
    fn test_legendre_roots_degree3() {
        let roots = legendre_roots(3).expect("failed");
        assert_eq!(roots.len(), 3);
        assert!(
            roots[1].abs() < 1e-12,
            "middle root should be 0, got {}",
            roots[1]
        );
        let s = (3.0 / 5.0_f64).sqrt();
        assert!((roots[0] + s).abs() < 1e-12);
        assert!((roots[2] - s).abs() < 1e-12);
    }

    #[test]
    fn test_legendre_roots_are_roots() {
        let roots = legendre_roots(5).expect("failed");
        for &r in &roots {
            let val = legendre(5, r);
            assert!(val.abs() < 1e-10, "P_5({r}) = {val}, expected ~0");
        }
    }

    #[test]
    fn test_chebyshev_t_roots_count() {
        let roots = chebyshev_t_roots(5).expect("failed");
        assert_eq!(roots.len(), 5);
        for &r in &roots {
            assert!(r.abs() <= 1.0, "root out of range: {r}");
        }
    }

    #[test]
    fn test_chebyshev_t_roots_are_roots() {
        let roots = chebyshev_t_roots(4).expect("failed");
        for &r in &roots {
            let val = chebyshev(4, r, true);
            assert!(val.abs() < 1e-12, "T_4({r}) = {val}");
        }
    }

    #[test]
    fn test_chebyshev_u_roots_are_roots() {
        // U_n roots are cos(k*pi/(n+1)) for k=1..n
        // Verify analytically: U_n(cos(theta)) = sin((n+1)*theta)/sin(theta)
        let n = 4;
        let roots = chebyshev_u_roots(n).expect("failed");
        for &r in &roots {
            // Use the trig definition: U_n(cos(theta)) = sin((n+1)*theta)/sin(theta)
            let theta = r.acos();
            let sin_theta = theta.sin();
            let val = if sin_theta.abs() < 1e-15 {
                // L'Hopital for theta near 0 or pi
                ((n + 1) as f64) * ((n as f64 + 1.0) * theta).cos() / theta.cos()
            } else {
                ((n as f64 + 1.0) * theta).sin() / sin_theta
            };
            assert!(val.abs() < 1e-10, "U_{n}({r}) = {val}");
        }
    }

    #[test]
    fn test_hermite_roots_degree2() {
        let roots = hermite_roots(2).expect("failed");
        let expected = (0.5_f64).sqrt();
        assert!((roots[0] + expected).abs() < 1e-12);
        assert!((roots[1] - expected).abs() < 1e-12);
    }

    #[test]
    fn test_hermite_roots_are_roots() {
        let roots = hermite_roots(5).expect("failed");
        for &r in &roots {
            let val = hermite(5, r);
            assert!(val.abs() < 1e-6, "H_5({r}) = {val}");
        }
    }

    #[test]
    fn test_hermite_prob_roots_are_roots() {
        let roots = hermite_prob_roots(4).expect("failed");
        for &r in &roots {
            let val = hermite_prob(4, r);
            assert!(val.abs() < 1e-8, "He_4({r}) = {val}");
        }
    }

    #[test]
    fn test_laguerre_roots_positive() {
        let roots = laguerre_roots(4).expect("failed");
        for &r in &roots {
            assert!(r > 0.0, "Laguerre root should be positive: {r}");
        }
    }

    #[test]
    fn test_laguerre_roots_are_roots() {
        let roots = laguerre_roots(4).expect("failed");
        for &r in &roots {
            let val = laguerre(4, r);
            assert!(val.abs() < 1e-8, "L_4({r}) = {val}");
        }
    }

    #[test]
    fn test_jacobi_roots_in_range() {
        let roots = jacobi_roots(5, 1.0, 2.0).expect("failed");
        for &r in &roots {
            assert!(
                r > -1.0 - 1e-10 && r < 1.0 + 1e-10,
                "root out of range: {r}"
            );
        }
    }

    #[test]
    fn test_jacobi_roots_legendre_case() {
        let j_roots = jacobi_roots(3, 0.0, 0.0).expect("failed");
        let l_roots = legendre_roots(3).expect("failed");
        for (jr, lr) in j_roots.iter().zip(l_roots.iter()) {
            assert!(
                (jr - lr).abs() < 1e-10,
                "Jacobi(0,0) roots should match Legendre"
            );
        }
    }

    #[test]
    fn test_gegenbauer_roots_count() {
        let roots = gegenbauer_roots(4, 1.5).expect("failed");
        assert_eq!(roots.len(), 4);
    }

    #[test]
    fn test_root_error_degree0() {
        assert!(legendre_roots(0).is_err());
        assert!(hermite_roots(0).is_err());
        assert!(laguerre_roots(0).is_err());
    }

    // ====== Batch evaluation tests ======

    #[test]
    fn test_legendre_batch() {
        let points = vec![0.0f64, 0.5, 1.0];
        let vals = legendre_batch(2, &points);
        assert_relative_eq!(vals[0], -0.5, epsilon = 1e-14);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_chebyshev_t_batch() {
        let points = vec![0.0f64, 0.5, 1.0];
        let vals = chebyshev_t_batch(2, &points);
        assert_relative_eq!(vals[0], -1.0, epsilon = 1e-14);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_hermite_batch() {
        let points = vec![0.0f64, 0.5, 1.0];
        let vals = hermite_batch(2, &points);
        assert_relative_eq!(vals[0], -2.0, epsilon = 1e-14);
        assert_relative_eq!(vals[2], 2.0, epsilon = 1e-14);
    }

    // ====== Coefficient tests ======

    #[test]
    fn test_legendre_coefficients_p0() {
        let c = legendre_coefficients(0);
        assert_eq!(c.len(), 1);
        assert_relative_eq!(c[0], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_legendre_coefficients_p1() {
        let c = legendre_coefficients(1);
        assert_eq!(c.len(), 2);
        assert_relative_eq!(c[0], 0.0, epsilon = 1e-14);
        assert_relative_eq!(c[1], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_legendre_coefficients_p2() {
        let c = legendre_coefficients(2);
        // P_2 = (3x^2 - 1)/2
        assert_relative_eq!(c[0], -0.5, epsilon = 1e-14);
        assert!(c[1].abs() < 1e-14);
        assert_relative_eq!(c[2], 1.5, epsilon = 1e-14);
    }

    #[test]
    fn test_chebyshev_t_coefficients_t2() {
        let c = chebyshev_t_coefficients(2);
        // T_2 = 2x^2 - 1
        assert_relative_eq!(c[0], -1.0, epsilon = 1e-14);
        assert!(c[1].abs() < 1e-14);
        assert_relative_eq!(c[2], 2.0, epsilon = 1e-14);
    }

    #[test]
    fn test_hermite_coefficients_h2() {
        let c = hermite_coefficients(2);
        // H_2 = 4x^2 - 2
        assert_relative_eq!(c[0], -2.0, epsilon = 1e-14);
        assert!(c[1].abs() < 1e-14);
        assert_relative_eq!(c[2], 4.0, epsilon = 1e-14);
    }

    #[test]
    fn test_hermite_prob_coefficients_he2() {
        let c = hermite_prob_coefficients(2);
        // He_2 = x^2 - 1
        assert_relative_eq!(c[0], -1.0, epsilon = 1e-14);
        assert!(c[1].abs() < 1e-14);
        assert_relative_eq!(c[2], 1.0, epsilon = 1e-14);
    }

    // ====== Higher-order derivative tests ======

    #[test]
    fn test_legendre_kth_derivative_k0() {
        assert_relative_eq!(
            legendre_kth_derivative(3, 0, 0.5f64),
            legendre(3, 0.5f64),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_legendre_kth_derivative_k1() {
        let d1 = legendre_kth_derivative(2, 1, 0.5f64);
        // P_2' = 3x, so P_2'(0.5) = 1.5
        assert_relative_eq!(d1, 1.5, epsilon = 1e-8);
    }

    #[test]
    fn test_legendre_kth_derivative_k2() {
        let d2 = legendre_kth_derivative(2, 2, 0.5f64);
        // P_2'' = 3
        assert_relative_eq!(d2, 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hermite_kth_derivative_exact() {
        // H_3(x) = 8x^3 - 12x
        // H'_3 = 24x^2 - 12
        // H''_3 = 48x
        // H'''_3 = 48
        let d3 = hermite_kth_derivative(3, 3, 0.5f64);
        assert_relative_eq!(d3, 48.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hermite_kth_derivative_identity() {
        // H^(k)_n = 2^k * n!/(n-k)! * H_{n-k}
        let x = 0.7f64;
        let d2 = hermite_kth_derivative(4, 2, x);
        let expected = 4.0 * (4.0 * 3.0) * hermite(2, x);
        assert_relative_eq!(d2, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_hermite_prob_kth_derivative_identity() {
        let x = 0.7f64;
        let d2 = hermite_prob_kth_derivative(4, 2, x);
        let expected = (4.0 * 3.0) * hermite_prob(2, x);
        assert_relative_eq!(d2, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_kth_derivative_beyond_degree_is_zero() {
        assert_relative_eq!(legendre_kth_derivative(3, 4, 0.5f64), 0.0, epsilon = 1e-14);
        assert_relative_eq!(hermite_kth_derivative(2, 3, 0.5f64), 0.0, epsilon = 1e-14);
    }

    // ====== All-degrees evaluation tests ======

    #[test]
    fn test_legendre_all() {
        let vals = legendre_all(4, 0.5f64);
        assert_eq!(vals.len(), 5);
        for (i, &v) in vals.iter().enumerate() {
            let expected = legendre(i, 0.5f64);
            assert_relative_eq!(v, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_chebyshev_t_all() {
        let vals = chebyshev_t_all(4, 0.3f64);
        assert_eq!(vals.len(), 5);
        for (i, &v) in vals.iter().enumerate() {
            let expected = chebyshev(i, 0.3f64, true);
            assert_relative_eq!(v, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_hermite_all() {
        let vals = hermite_all(4, 0.5f64);
        assert_eq!(vals.len(), 5);
        for (i, &v) in vals.iter().enumerate() {
            let expected = hermite(i, 0.5f64);
            assert_relative_eq!(v, expected, epsilon = 1e-10);
        }
    }

    // ====== Clenshaw algorithm tests ======

    #[test]
    fn test_clenshaw_chebyshev_constant() {
        let coeffs = [3.0f64];
        assert_relative_eq!(clenshaw_chebyshev(&coeffs, 0.5), 3.0, epsilon = 1e-14);
    }

    #[test]
    fn test_clenshaw_chebyshev_linear() {
        // f(x) = 2*T_0 + 3*T_1 = 2 + 3x
        let coeffs = [2.0f64, 3.0];
        assert_relative_eq!(clenshaw_chebyshev(&coeffs, 0.5), 3.5, epsilon = 1e-14);
    }

    #[test]
    fn test_clenshaw_chebyshev_quadratic() {
        // f(x) = 1*T_0 + 0*T_1 + (-1)*T_2 = 1 - (2x^2 - 1) = 2 - 2x^2
        let coeffs = [1.0f64, 0.0, -1.0];
        assert_relative_eq!(clenshaw_chebyshev(&coeffs, 0.5), 1.5, epsilon = 1e-14);
    }

    #[test]
    fn test_clenshaw_legendre_constant() {
        let coeffs = [5.0f64];
        assert_relative_eq!(clenshaw_legendre(&coeffs, 0.3), 5.0, epsilon = 1e-14);
    }

    #[test]
    fn test_clenshaw_empty() {
        let coeffs: [f64; 0] = [];
        assert_relative_eq!(clenshaw_chebyshev(&coeffs, 0.5), 0.0, epsilon = 1e-14);
        assert_relative_eq!(clenshaw_legendre(&coeffs, 0.5), 0.0, epsilon = 1e-14);
    }

    // ====== Three-term recurrence tests ======

    #[test]
    fn test_eval_three_term_legendre() {
        // Use three-term recurrence to evaluate Legendre
        let val = eval_three_term_recurrence(3, 0.5f64, |k| {
            let k_f = k as f64;
            let a = (2.0 * k_f + 1.0) / (k_f + 1.0);
            let b = 0.0;
            let c = k_f / (k_f + 1.0);
            (a, b, c)
        });
        assert_relative_eq!(val, legendre(3, 0.5f64), epsilon = 1e-12);
    }

    #[test]
    fn test_eval_all_degrees_basic() {
        let vals = eval_all_degrees(3, 0.5f64, |k| {
            let k_f = k as f64;
            let a = (2.0 * k_f + 1.0) / (k_f + 1.0);
            let b = 0.0;
            let c = k_f / (k_f + 1.0);
            (a, b, c)
        });
        assert_eq!(vals.len(), 4);
        assert_relative_eq!(vals[0], 1.0, epsilon = 1e-14);
        assert_relative_eq!(vals[1], 0.5, epsilon = 1e-14);
    }
}
