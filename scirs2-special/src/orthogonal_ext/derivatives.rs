//! Derivative computation for orthogonal polynomials
//!
//! Provides analytic derivative formulas for Legendre, Chebyshev, Hermite,
//! Laguerre, Jacobi, and Gegenbauer polynomials. These are needed for:
//! - Newton iteration in root-finding
//! - Differential equation solvers
//! - Sensitivity analysis

use crate::orthogonal::{chebyshev, gegenbauer, hermite, hermite_prob, jacobi, laguerre, laguerre_generalized, legendre, legendre_assoc};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

/// Helper to convert f64 constants to generic Float type
#[inline(always)]
fn const_f64<F: Float + FromPrimitive>(value: f64) -> F {
    F::from(value).expect("Failed to convert constant to target float type")
}

/// Derivative of Legendre polynomial P_n(x).
///
/// Uses the identity:
/// ```text
/// P'_n(x) = n * [x P_n(x) - P_{n-1}(x)] / (x^2 - 1)
/// ```
/// For |x| = 1, uses the limit formula: P'_n(1) = n(n+1)/2, P'_n(-1) = (-1)^{n+1} n(n+1)/2
///
/// # Arguments
/// * `n` - Degree of the polynomial
/// * `x` - Point of evaluation
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_ext::legendre_derivative;
/// // P'_1(x) = 1 (derivative of x)
/// assert!((legendre_derivative(1, 0.5f64) - 1.0).abs() < 1e-10);
/// ```
pub fn legendre_derivative<F: Float + FromPrimitive + Debug>(n: usize, x: F) -> F {
    if n == 0 {
        return F::zero();
    }

    let one = F::one();
    let x2_minus_1 = x * x - one;

    // Handle endpoint singularities
    if x2_minus_1.abs() < F::from(1e-14).expect("Failed to convert constant") {
        // At x = 1: P'_n(1) = n(n+1)/2
        // At x = -1: P'_n(-1) = (-1)^{n+1} * n(n+1)/2
        let n_f = F::from(n).expect("Failed to convert n");
        let val = n_f * (n_f + one) / const_f64::<F>(2.0);
        if x > F::zero() {
            return val;
        } else {
            // (-1)^{n+1}
            return if (n + 1) % 2 == 0 { val } else { -val };
        }
    }

    let n_f = F::from(n).expect("Failed to convert n");
    let p_n = legendre(n, x);
    let p_nm1 = if n > 0 { legendre(n - 1, x) } else { F::zero() };

    n_f * (x * p_n - p_nm1) / x2_minus_1
}

/// Derivative of Chebyshev polynomial of the first kind T_n(x).
///
/// Uses the identity: T'_n(x) = n * U_{n-1}(x)
/// where U is the Chebyshev polynomial of the second kind.
///
/// # Arguments
/// * `n` - Degree
/// * `x` - Point of evaluation
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_ext::chebyshev_t_derivative;
/// // T'_2(x) = 4x, so T'_2(0.5) = 2.0
/// assert!((chebyshev_t_derivative(2, 0.5f64) - 2.0).abs() < 1e-10);
/// ```
pub fn chebyshev_t_derivative<F: Float + FromPrimitive + Debug>(n: usize, x: F) -> F {
    if n == 0 {
        return F::zero();
    }

    let n_f = F::from(n).expect("Failed to convert n");
    // T'_n(x) = n * U_{n-1}(x)
    let u_nm1 = chebyshev(n - 1, x, false); // false = second kind
    n_f * u_nm1
}

/// Derivative of Chebyshev polynomial of the second kind U_n(x).
///
/// Uses the recurrence:
/// ```text
/// (1-x^2) U'_n(x) = -n x U_n(x) + (n+1) U_{n-1}(x) = (n+1) T_{n+1}(x) - x U_n(x)
/// ```
///
/// For |x| near 1, uses an endpoint limit.
pub fn chebyshev_u_derivative<F: Float + FromPrimitive + Debug>(n: usize, x: F) -> F {
    if n == 0 {
        return F::zero();
    }

    let one = F::one();
    let x2_minus_1 = x * x - one;

    if x2_minus_1.abs() < F::from(1e-14).expect("Failed to convert constant") {
        // At endpoints, use limit formula
        // U'_n(1) = n(n+1)(n+2)/3
        let n_f = F::from(n).expect("Failed to convert n");
        let val = n_f * (n_f + one) * (n_f + const_f64::<F>(2.0)) / const_f64::<F>(3.0);
        if x > F::zero() {
            return val;
        } else {
            return if n % 2 == 0 { -val } else { val };
        }
    }

    let n_f = F::from(n).expect("Failed to convert n");
    let u_n = chebyshev(n, x, false);
    let t_np1 = chebyshev(n + 1, x, true);

    // (1 - x^2) U'_n = (n+1) T_{n+1} - x U_n
    // => U'_n = [(n+1) T_{n+1} - x U_n] / (1 - x^2)
    ((n_f + one) * t_np1 - x * u_n) / (one - x * x)
}

/// Derivative of physicist's Hermite polynomial H_n(x).
///
/// Uses the identity: H'_n(x) = 2n H_{n-1}(x)
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_ext::hermite_derivative;
/// // H'_1(x) = 2*1*H_0(x) = 2
/// assert!((hermite_derivative(1, 0.5f64) - 2.0).abs() < 1e-10);
/// ```
pub fn hermite_derivative<F: Float + FromPrimitive + Debug>(n: usize, x: F) -> F {
    if n == 0 {
        return F::zero();
    }
    let n_f = F::from(n).expect("Failed to convert n");
    const_f64::<F>(2.0) * n_f * hermite(n - 1, x)
}

/// Derivative of probabilist's Hermite polynomial He_n(x).
///
/// Uses the identity: He'_n(x) = n He_{n-1}(x)
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_ext::hermite_prob_derivative;
/// // He'_2(x) = 2*He_1(x) = 2x, so He'_2(0.5) = 1.0
/// assert!((hermite_prob_derivative(2, 0.5f64) - 1.0).abs() < 1e-10);
/// ```
pub fn hermite_prob_derivative<F: Float + FromPrimitive + Debug>(n: usize, x: F) -> F {
    if n == 0 {
        return F::zero();
    }
    let n_f = F::from(n).expect("Failed to convert n");
    n_f * hermite_prob(n - 1, x)
}

/// Derivative of Laguerre polynomial L_n(x).
///
/// Uses: L'_n(x) = -L_{n-1}^{(1)}(x) (generalized Laguerre with alpha=1)
///
/// Alternatively, by recurrence: L'_n(x) = (n L_n(x) - n L_{n-1}(x)) / x for x != 0
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_ext::laguerre_derivative;
/// // L'_1(x) = -1
/// assert!((laguerre_derivative(1, 0.5f64) + 1.0).abs() < 1e-10);
/// ```
pub fn laguerre_derivative<F: Float + FromPrimitive + Debug>(n: usize, x: F) -> F {
    if n == 0 {
        return F::zero();
    }
    // L'_n(x) = -L_{n-1}^{(1)}(x)
    -laguerre_generalized(n - 1, F::one(), x)
}

/// Derivative of generalized Laguerre polynomial L_n^(alpha)(x).
///
/// Uses: d/dx L_n^(alpha)(x) = -L_{n-1}^{(alpha+1)}(x)
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_ext::laguerre_generalized_derivative;
/// // L'_1^(0)(x) = -1
/// assert!((laguerre_generalized_derivative(1, 0.0f64, 0.5f64) + 1.0).abs() < 1e-10);
/// ```
pub fn laguerre_generalized_derivative<F: Float + FromPrimitive + Debug>(
    n: usize,
    alpha: F,
    x: F,
) -> F {
    if n == 0 {
        return F::zero();
    }
    -laguerre_generalized(n - 1, alpha + F::one(), x)
}

/// Derivative of Jacobi polynomial P_n^(alpha,beta)(x).
///
/// Uses: d/dx P_n^(a,b)(x) = (n + a + b + 1)/2 * P_{n-1}^{(a+1,b+1)}(x)
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_ext::jacobi_derivative;
/// // For a=b=0 (Legendre): P'_1(x) = 1
/// assert!((jacobi_derivative(1, 0.0f64, 0.0f64, 0.5f64) - 1.0).abs() < 1e-10);
/// ```
pub fn jacobi_derivative<F: Float + FromPrimitive + Debug>(
    n: usize,
    alpha: F,
    beta: F,
    x: F,
) -> F {
    if n == 0 {
        return F::zero();
    }

    let n_f = F::from(n).expect("Failed to convert n");
    let factor = (n_f + alpha + beta + F::one()) / const_f64::<F>(2.0);
    factor * jacobi(n - 1, alpha + F::one(), beta + F::one(), x)
}

/// Derivative of Gegenbauer polynomial C_n^(lambda)(x).
///
/// Uses: C'_n^(lambda)(x) = 2 lambda C_{n-1}^{(lambda+1)}(x)
///
/// # Examples
/// ```
/// use scirs2_special::orthogonal_ext::gegenbauer_derivative;
/// // C'_1^(1)(x) = 2*1*C_0^(2)(x) = 2
/// assert!((gegenbauer_derivative(1, 1.0f64, 0.5f64) - 2.0).abs() < 1e-10);
/// ```
pub fn gegenbauer_derivative<F: Float + FromPrimitive + Debug>(
    n: usize,
    lambda: F,
    x: F,
) -> F {
    if n == 0 {
        return F::zero();
    }
    const_f64::<F>(2.0) * lambda * gegenbauer(n - 1, lambda + F::one(), x)
}

/// Derivative of associated Legendre function P_n^m(x).
///
/// Uses the recurrence:
/// ```text
/// (1-x^2) dP_n^m/dx = (n+1) x P_n^m(x) - (n-m+1) P_{n+1}^m(x)
/// ```
///
/// For |x| near 1, special care is taken.
pub fn legendre_assoc_derivative<F: Float + FromPrimitive + Debug>(
    n: usize,
    m: i32,
    x: F,
) -> F {
    let one = F::one();
    let one_minus_x2 = one - x * x;

    if one_minus_x2.abs() < F::from(1e-14).expect("Failed to convert constant") {
        // At the poles, the derivative is typically singular for m != 0
        // For m = 0, it reduces to the Legendre derivative
        if m == 0 {
            return legendre_derivative(n, x);
        }
        // For m != 0 at endpoints, the derivative involves division by zero
        // Return a large finite value or the limit
        return if x > F::zero() { F::infinity() } else { F::neg_infinity() };
    }

    let n_f = F::from(n).expect("Failed to convert n");
    let m_abs = m.unsigned_abs() as usize;
    let p_n_m = legendre_assoc(n, m, x);
    let p_np1_m = legendre_assoc(n + 1, m, x);
    let n_minus_m_plus_1 = F::from(n - m_abs + 1 + if m < 0 { 2 * m_abs } else { 0 })
        .expect("Failed to convert");

    ((n_f + one) * x * p_n_m - n_minus_m_plus_1 * p_np1_m) / one_minus_x2
}

// ====== SciPy-compatible eval_* wrappers ======

/// Evaluate Legendre polynomial (SciPy compatible: `eval_legendre`).
///
/// This is a convenience wrapper matching scipy.special.eval_legendre.
pub fn eval_legendre<F: Float + FromPrimitive + Debug>(n: usize, x: F) -> F {
    legendre(n, x)
}

/// Evaluate associated Legendre function (SciPy compatible: `eval_legendre_assoc`).
pub fn eval_legendre_assoc<F: Float + FromPrimitive + Debug>(n: usize, m: i32, x: F) -> F {
    legendre_assoc(n, m, x)
}

/// Evaluate Chebyshev T polynomial (SciPy compatible: `eval_chebyt`).
pub fn eval_chebyt<F: Float + FromPrimitive + Debug>(n: usize, x: F) -> F {
    chebyshev(n, x, true)
}

/// Evaluate Chebyshev U polynomial (SciPy compatible: `eval_chebyu`).
pub fn eval_chebyu<F: Float + FromPrimitive + Debug>(n: usize, x: F) -> F {
    chebyshev(n, x, false)
}

/// Evaluate physicist's Hermite polynomial (SciPy compatible: `eval_hermite`).
pub fn eval_hermite<F: Float + FromPrimitive + Debug>(n: usize, x: F) -> F {
    hermite(n, x)
}

/// Evaluate probabilist's Hermite polynomial (SciPy compatible: `eval_hermitenorm`).
pub fn eval_hermitenorm<F: Float + FromPrimitive + Debug>(n: usize, x: F) -> F {
    hermite_prob(n, x)
}

/// Evaluate Laguerre polynomial (SciPy compatible: `eval_laguerre`).
pub fn eval_laguerre<F: Float + FromPrimitive + Debug>(n: usize, x: F) -> F {
    laguerre(n, x)
}

/// Evaluate generalized Laguerre polynomial (SciPy compatible: `eval_genlaguerre`).
pub fn eval_genlaguerre<F: Float + FromPrimitive + Debug>(n: usize, alpha: F, x: F) -> F {
    laguerre_generalized(n, alpha, x)
}

/// Evaluate Jacobi polynomial (SciPy compatible: `eval_jacobi`).
pub fn eval_jacobi<F: Float + FromPrimitive + Debug>(
    n: usize,
    alpha: F,
    beta: F,
    x: F,
) -> F {
    jacobi(n, alpha, beta, x)
}

/// Evaluate Gegenbauer polynomial (SciPy compatible: `eval_gegenbauer`).
pub fn eval_gegenbauer<F: Float + FromPrimitive + Debug>(n: usize, lambda: F, x: F) -> F {
    gegenbauer(n, lambda, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ====== Legendre derivative tests ======

    #[test]
    fn test_legendre_derivative_n0() {
        assert_relative_eq!(legendre_derivative(0, 0.5f64), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_legendre_derivative_n1() {
        // P'_1(x) = 1 for all x
        assert_relative_eq!(legendre_derivative(1, 0.5f64), 1.0, epsilon = 1e-10);
        assert_relative_eq!(legendre_derivative(1, 0.0f64), 1.0, epsilon = 1e-10);
        assert_relative_eq!(legendre_derivative(1, -0.3f64), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_legendre_derivative_n2() {
        // P_2(x) = (3x^2 - 1)/2, P'_2(x) = 3x
        assert_relative_eq!(legendre_derivative(2, 0.5f64), 1.5, epsilon = 1e-10);
        assert_relative_eq!(legendre_derivative(2, 0.0f64), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_legendre_derivative_n3() {
        // P_3(x) = (5x^3 - 3x)/2, P'_3(x) = (15x^2 - 3)/2
        let expected = (15.0 * 0.5 * 0.5 - 3.0) / 2.0;
        assert_relative_eq!(legendre_derivative(3, 0.5f64), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_legendre_derivative_at_one() {
        // P'_n(1) = n(n+1)/2
        assert_relative_eq!(legendre_derivative(3, 1.0f64), 6.0, epsilon = 1e-10);
        assert_relative_eq!(legendre_derivative(4, 1.0f64), 10.0, epsilon = 1e-10);
    }

    // ====== Chebyshev derivative tests ======

    #[test]
    fn test_chebyshev_t_derivative_n0() {
        assert_relative_eq!(chebyshev_t_derivative(0, 0.5f64), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_chebyshev_t_derivative_n1() {
        // T'_1(x) = 1 (T_1 = x)
        assert_relative_eq!(chebyshev_t_derivative(1, 0.5f64), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_chebyshev_t_derivative_n2() {
        // T_2(x) = 2x^2 - 1, T'_2(x) = 4x
        assert_relative_eq!(chebyshev_t_derivative(2, 0.5f64), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_chebyshev_t_derivative_n3() {
        // T_3(x) = 4x^3 - 3x, T'_3(x) = 12x^2 - 3
        let expected = 12.0 * 0.5 * 0.5 - 3.0;
        assert_relative_eq!(chebyshev_t_derivative(3, 0.5f64), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_chebyshev_t_derivative_identity() {
        // Verify T'_n = n * U_{n-1} for n=4 at x=0.3
        let x = 0.3f64;
        let deriv = chebyshev_t_derivative(4, x);
        let u3 = chebyshev(3, x, false);
        assert_relative_eq!(deriv, 4.0 * u3, epsilon = 1e-10);
    }

    // ====== Hermite derivative tests ======

    #[test]
    fn test_hermite_derivative_n0() {
        assert_relative_eq!(hermite_derivative(0, 0.5f64), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hermite_derivative_n1() {
        // H'_1(x) = 2*1*H_0(x) = 2
        assert_relative_eq!(hermite_derivative(1, 0.5f64), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hermite_derivative_n2() {
        // H'_2(x) = 2*2*H_1(x) = 4*2x = 8x
        assert_relative_eq!(hermite_derivative(2, 0.5f64), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hermite_derivative_n3() {
        // H'_3(x) = 2*3*H_2(x) = 6*(4x^2-2) = 24x^2 - 12
        let expected = 24.0 * 0.25 - 12.0;
        assert_relative_eq!(hermite_derivative(3, 0.5f64), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_hermite_prob_derivative_n2() {
        // He'_2(x) = 2*He_1(x) = 2x
        assert_relative_eq!(hermite_prob_derivative(2, 0.5f64), 1.0, epsilon = 1e-10);
    }

    // ====== Laguerre derivative tests ======

    #[test]
    fn test_laguerre_derivative_n0() {
        assert_relative_eq!(laguerre_derivative(0, 0.5f64), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_laguerre_derivative_n1() {
        // L_1(x) = 1 - x, L'_1(x) = -1
        assert_relative_eq!(laguerre_derivative(1, 0.5f64), -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_laguerre_derivative_n2() {
        // L_2(x) = 1 - 2x + x^2/2, L'_2(x) = -2 + x
        assert_relative_eq!(laguerre_derivative(2, 0.5f64), -1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_laguerre_generalized_derivative_n1() {
        // d/dx L_1^(0)(x) = -1
        assert_relative_eq!(
            laguerre_generalized_derivative(1, 0.0f64, 0.5f64),
            -1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_laguerre_generalized_derivative_identity() {
        // d/dx L_n^(a)(x) = -L_{n-1}^{a+1}(x)
        let x = 0.7f64;
        let deriv = laguerre_generalized_derivative(3, 1.0, x);
        let expected = -laguerre_generalized(2, 2.0, x);
        assert_relative_eq!(deriv, expected, epsilon = 1e-10);
    }

    // ====== Jacobi derivative tests ======

    #[test]
    fn test_jacobi_derivative_n0() {
        assert_relative_eq!(
            jacobi_derivative(0, 1.0f64, 1.0f64, 0.5f64),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_jacobi_derivative_legendre() {
        // For a=b=0, should match Legendre derivative
        let x = 0.3f64;
        let j_deriv = jacobi_derivative(3, 0.0, 0.0, x);
        let l_deriv = legendre_derivative(3, x);
        assert_relative_eq!(j_deriv, l_deriv, epsilon = 1e-8);
    }

    #[test]
    fn test_jacobi_derivative_n1() {
        // P_1^(a,b)(x) = (a+1) + (a+b+2)(x-1)/2
        // d/dx P_1^(a,b)(x) = (a+b+2)/2
        let a = 2.0f64;
        let b = 3.0f64;
        let expected = (a + b + 2.0) / 2.0;
        assert_relative_eq!(
            jacobi_derivative(1, a, b, 0.5),
            expected,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_jacobi_derivative_identity() {
        // d/dx P_n^(a,b)(x) = (n+a+b+1)/2 * P_{n-1}^{(a+1,b+1)}(x)
        let x = 0.4f64;
        let deriv = jacobi_derivative(3, 1.0, 2.0, x);
        let expected = (3.0 + 1.0 + 2.0 + 1.0) / 2.0 * jacobi(2, 2.0, 3.0, x);
        assert_relative_eq!(deriv, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_gegenbauer_derivative_n1() {
        // C'_1^(lambda)(x) = 2*lambda (constant)
        assert_relative_eq!(
            gegenbauer_derivative(1, 1.5f64, 0.5f64),
            3.0,
            epsilon = 1e-10
        );
    }

    // ====== eval_* wrapper tests ======

    #[test]
    fn test_eval_legendre_matches() {
        assert_relative_eq!(
            eval_legendre(3, 0.5f64),
            legendre(3, 0.5f64),
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_eval_chebyt_matches() {
        assert_relative_eq!(
            eval_chebyt(4, 0.3f64),
            chebyshev(4, 0.3f64, true),
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_eval_chebyu_matches() {
        assert_relative_eq!(
            eval_chebyu(3, 0.4f64),
            chebyshev(3, 0.4f64, false),
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_eval_hermite_matches() {
        assert_relative_eq!(
            eval_hermite(4, 1.0f64),
            hermite(4, 1.0f64),
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_eval_hermitenorm_matches() {
        assert_relative_eq!(
            eval_hermitenorm(3, 1.0f64),
            hermite_prob(3, 1.0f64),
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_eval_laguerre_matches() {
        assert_relative_eq!(
            eval_laguerre(3, 1.5f64),
            laguerre(3, 1.5f64),
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_eval_genlaguerre_matches() {
        assert_relative_eq!(
            eval_genlaguerre(2, 1.0f64, 1.5f64),
            laguerre_generalized(2, 1.0, 1.5),
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_eval_jacobi_matches() {
        assert_relative_eq!(
            eval_jacobi(2, 1.0f64, 2.0f64, 0.5f64),
            jacobi(2, 1.0, 2.0, 0.5),
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_eval_gegenbauer_matches() {
        assert_relative_eq!(
            eval_gegenbauer(3, 1.5f64, 0.5f64),
            gegenbauer(3, 1.5, 0.5),
            epsilon = 1e-14
        );
    }
}
