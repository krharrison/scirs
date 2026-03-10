//! Extended matrix functions: square root, logarithm, sign, p-th root, condition
//!
//! Provides advanced matrix function algorithms beyond the basic set in
//! `matrix_functions`:
//!
//! - **Matrix square root** (Denman-Beavers iteration, Schur method)
//! - **Matrix logarithm** (inverse scaling and squaring + Pade approximation)
//! - **Matrix sign function** (Roberts iteration)
//! - **Matrix p-th root** (Newton-Schulz iteration)
//! - **Condition number** for matrix functions

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Trait alias
// ---------------------------------------------------------------------------

/// Trait alias for floating-point bounds used across this module.
pub trait MFFloat: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}
impl<T> MFFloat for T where T: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static {}

// ===================================================================
// Helper: matrix Frobenius norm (avoids importing norm module)
// ===================================================================

fn frobenius_norm<F: MFFloat>(m: &Array2<F>) -> F {
    let mut acc = F::zero();
    for &v in m.iter() {
        acc += v * v;
    }
    acc.sqrt()
}

// ===================================================================
// Matrix square root -- Denman-Beavers iteration
// ===================================================================

/// Result of a matrix square root computation.
#[derive(Debug, Clone)]
pub struct MatrixSqrtResult<F> {
    /// The principal square root S such that S*S ~ A.
    pub sqrt: Array2<F>,
    /// Number of iterations used.
    pub iterations: usize,
    /// Final residual ||S*S - A||_F.
    pub residual: F,
}

/// Compute the matrix square root using the Denman-Beavers iteration.
///
/// Given a nonsingular matrix A, this finds S such that S*S = A.
/// The iteration simultaneously converges Y_k -> S and Z_k -> S^{-1}:
///
///   Y_{k+1} = (Y_k + Z_k^{-1}) / 2
///   Z_{k+1} = (Z_k + Y_k^{-1}) / 2
///
/// # Arguments
/// * `a`       - Input nonsingular square matrix
/// * `max_iter`- Maximum iterations (default 100)
/// * `tol`     - Convergence tolerance (default 1e-12)
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions_extended::sqrt_denman_beavers;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
/// let res = sqrt_denman_beavers(&a.view(), None, None).expect("sqrt failed");
/// // res.sqrt should be approximately [[2, 0], [0, 3]]
/// assert!((res.sqrt[[0, 0]] - 2.0_f64).abs() < 1e-8);
/// assert!((res.sqrt[[1, 1]] - 3.0_f64).abs() < 1e-8);
/// ```
pub fn sqrt_denman_beavers<F: MFFloat>(
    a: &ArrayView2<F>,
    max_iter: Option<usize>,
    tol: Option<F>,
) -> LinalgResult<MatrixSqrtResult<F>> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError("Matrix must be square".into()));
    }

    let max_it = max_iter.unwrap_or(100);
    let eps = tol.unwrap_or_else(|| F::from(1e-12).unwrap_or(F::epsilon()));
    let half =
        F::from(0.5).ok_or_else(|| LinalgError::ComputationError("Cannot convert 0.5".into()))?;

    let mut y = a.to_owned();
    let mut z = Array2::<F>::eye(n);

    for iter in 0..max_it {
        let z_inv = crate::inv(&z.view(), None)?;
        let y_inv = crate::inv(&y.view(), None)?;

        let y_new = (&y + &z_inv) * half;
        let z_new = (&z + &y_inv) * half;

        let diff = frobenius_norm(&(&y_new - &y));
        y = y_new;
        z = z_new;

        if diff < eps {
            let residual = frobenius_norm(&(y.dot(&y) - a));
            return Ok(MatrixSqrtResult {
                sqrt: y,
                iterations: iter + 1,
                residual,
            });
        }
    }

    let residual = frobenius_norm(&(y.dot(&y) - a));
    Ok(MatrixSqrtResult {
        sqrt: y,
        iterations: max_it,
        residual,
    })
}

/// Compute the matrix square root using the Schur method.
///
/// 1. Compute Schur decomposition A = Q T Q^T
/// 2. Compute square root of quasi-upper-triangular T element-by-element
/// 3. Back-transform: S = Q * sqrt(T) * Q^T
///
/// More robust than Denman-Beavers for ill-conditioned matrices.
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions_extended::sqrt_schur;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
/// let s = sqrt_schur(&a.view()).expect("sqrt failed");
/// assert!((s[[0, 0]] - 2.0_f64).abs() < 1e-6);
/// assert!((s[[1, 1]] - 3.0_f64).abs() < 1e-6);
/// ```
pub fn sqrt_schur<F: MFFloat>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError("Matrix must be square".into()));
    }
    if n == 0 {
        return Ok(Array2::<F>::zeros((0, 0)));
    }

    let (q, t) = crate::decomposition::schur(a)?;

    // Compute square root of quasi-upper-triangular T
    let mut s = Array2::<F>::zeros((n, n));

    // Diagonal elements first
    for i in 0..n {
        if t[[i, i]] < F::zero() {
            return Err(LinalgError::DomainError(
                "Matrix has negative eigenvalues; principal square root does not exist".into(),
            ));
        }
        s[[i, i]] = t[[i, i]].sqrt();
    }

    // Super-diagonal elements by recurrence:
    // s_{ij} = (t_{ij} - sum_{k=i+1}^{j-1} s_{ik} s_{kj}) / (s_{ii} + s_{jj})
    for j in 1..n {
        for i in (0..j).rev() {
            let mut off_sum = F::zero();
            for k in (i + 1)..j {
                off_sum += s[[i, k]] * s[[k, j]];
            }
            let denom = s[[i, i]] + s[[j, j]];
            if denom.abs() < F::epsilon() {
                // Nearly repeated eigenvalues; use a fallback
                s[[i, j]] = F::zero();
            } else {
                s[[i, j]] = (t[[i, j]] - off_sum) / denom;
            }
        }
    }

    // Back-transform: sqrt(A) = Q S Q^T
    Ok(q.dot(&s).dot(&q.t()))
}

// ===================================================================
// Matrix logarithm -- inverse scaling and squaring + Pade
// ===================================================================

/// Compute the matrix logarithm via inverse scaling and squaring with Pade
/// approximation.
///
/// Algorithm:
/// 1. Scale A by repeated square-rooting until ||A - I|| is small
/// 2. Apply diagonal Pade approximant to log(I + X) where X = A_scaled - I
/// 3. Undo the scaling: log(A) = 2^s * log(A_scaled)
///
/// # Arguments
/// * `a`       - Input square matrix with no eigenvalues on the closed negative real axis
/// * `max_sqrt`- Maximum scaling steps (default 50)
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions_extended::logm_iss;
///
/// let a = array![[2.718281828_f64, 0.0], [0.0, 7.389056099]];
/// let l = logm_iss(&a.view(), None).expect("logm failed");
/// // log of diag(e, e^2) ~ diag(1, 2)
/// assert!((l[[0, 0]] - 1.0_f64).abs() < 1e-4);
/// assert!((l[[1, 1]] - 2.0_f64).abs() < 1e-4);
/// ```
pub fn logm_iss<F: MFFloat>(a: &ArrayView2<F>, max_sqrt: Option<usize>) -> LinalgResult<Array2<F>> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError("Matrix must be square".into()));
    }
    if n == 0 {
        return Ok(Array2::<F>::zeros((0, 0)));
    }

    let max_s = max_sqrt.unwrap_or(50);
    let eye = Array2::<F>::eye(n);

    // Phase 1: repeated square-root scaling until ||A_k - I||_F < 0.5
    let mut a_k = a.to_owned();
    let mut s: usize = 0;
    let threshold = F::from(0.5).unwrap_or(F::one());

    for _ in 0..max_s {
        let diff = frobenius_norm(&(&a_k - &eye));
        if diff < threshold {
            break;
        }
        // a_k <- sqrt(a_k) via Denman-Beavers (quick, low-iter)
        let res = sqrt_denman_beavers(&a_k.view(), Some(40), None)?;
        a_k = res.sqrt;
        s += 1;
    }

    // Phase 2: Pade approximation for log(I + X), X = a_k - I
    let x = &a_k - &eye;

    // Use [m/m] diagonal Pade of degree m = 7 for log(1+x)
    // Coefficients derived from Pade table for log(1+x)
    // We use the recurrence-based approach for the matrix rational function.
    let log_x = pade_log_approximant(&x, 8)?;

    // Phase 3: undo scaling
    let two = F::one() + F::one();
    let scale = two.powi(s as i32);
    Ok(log_x * scale)
}

/// Pade approximant for log(I + X) using partial fractions.
/// Uses the [m/m] diagonal Pade approximant computed via the identity:
///   log(I + X) ~ X * (I + X/2)^{-1}  (order-1 Pade)
/// For higher order, we iterate: use the identity
///   log(I + X) = sum_{k=1}^{m} c_k * X * (I + d_k * X)^{-1}
/// with coefficients from Gauss-Legendre quadrature of integral representation.
fn pade_log_approximant<F: MFFloat>(x: &Array2<F>, order: usize) -> LinalgResult<Array2<F>> {
    let n = x.shape()[0];
    let eye = Array2::<F>::eye(n);

    // Use Gauss-Legendre quadrature nodes/weights for
    //   log(I + X) = integral_0^1 X (I + t X)^{-1} dt
    // For `order` quadrature points:
    let (nodes, weights) = gauss_legendre_nodes(order);

    let mut result = Array2::<F>::zeros((n, n));
    for k in 0..order {
        let t_k = F::from(nodes[k]).unwrap_or(F::zero());
        let w_k = F::from(weights[k]).unwrap_or(F::zero());
        // (I + t_k * X)^{-1}
        let mat = &eye + &(x * t_k);
        let mat_inv = crate::inv(&mat.view(), None)?;
        // accumulate w_k * X * mat_inv
        result = result + x.dot(&mat_inv) * w_k;
    }
    Ok(result)
}

/// Gauss-Legendre quadrature nodes and weights on [0, 1].
/// Returns (nodes, weights) for the given order.
fn gauss_legendre_nodes(order: usize) -> (Vec<f64>, Vec<f64>) {
    // Pre-computed for common orders, transformed from [-1,1] to [0,1]
    match order {
        1 => (vec![0.5], vec![1.0]),
        2 => (vec![0.2113248654, 0.7886751346], vec![0.5, 0.5]),
        4 => (
            vec![0.0694318442, 0.3300094782, 0.6699905218, 0.9305681558],
            vec![0.1739274226, 0.3260725774, 0.3260725774, 0.1739274226],
        ),
        8 => (
            vec![
                0.0198550718,
                0.1016667613,
                0.2372337950,
                0.4082826788,
                0.5917173212,
                0.7627662050,
                0.8983332387,
                0.9801449282,
            ],
            vec![
                0.0506142681,
                0.1111905172,
                0.1568533229,
                0.1813418917,
                0.1813418917,
                0.1568533229,
                0.1111905172,
                0.0506142681,
            ],
        ),
        _ => {
            // Fallback: uniform trapezoid rule
            let mut nodes = Vec::with_capacity(order);
            let mut weights = Vec::with_capacity(order);
            let h = 1.0 / (order as f64);
            for i in 0..order {
                nodes.push((i as f64 + 0.5) * h);
                weights.push(h);
            }
            (nodes, weights)
        }
    }
}

// ===================================================================
// Matrix sign function -- Roberts iteration
// ===================================================================

/// Result of matrix sign function computation.
#[derive(Debug, Clone)]
pub struct MatrixSignResult<F> {
    /// The matrix sign S such that S*S ~ I.
    pub sign: Array2<F>,
    /// Number of iterations used.
    pub iterations: usize,
    /// Final residual ||S*S - I||_F.
    pub residual: F,
}

/// Compute the matrix sign function using the Roberts iteration.
///
/// The Roberts iteration is:
///   S_{k+1} = (S_k + S_k^{-1}) / 2
///
/// starting from S_0 = A. Converges quadratically to sign(A).
///
/// The matrix sign function satisfies sign(A)^2 = I and can be used
/// to split the spectrum of A into stable/unstable parts.
///
/// # Arguments
/// * `a`       - Input nonsingular square matrix
/// * `max_iter`- Maximum iterations (default 100)
/// * `tol`     - Convergence tolerance (default 1e-12)
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions_extended::sign_roberts;
///
/// // For a positive definite matrix, sign(A) = I
/// let a = array![[2.0_f64, 0.0], [0.0, 3.0]];
/// let res = sign_roberts(&a.view(), None, None).expect("sign failed");
/// assert!((res.sign[[0, 0]] - 1.0_f64).abs() < 1e-8);
/// assert!((res.sign[[1, 1]] - 1.0_f64).abs() < 1e-8);
/// ```
pub fn sign_roberts<F: MFFloat>(
    a: &ArrayView2<F>,
    max_iter: Option<usize>,
    tol: Option<F>,
) -> LinalgResult<MatrixSignResult<F>> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError("Matrix must be square".into()));
    }

    let max_it = max_iter.unwrap_or(100);
    let eps = tol.unwrap_or_else(|| F::from(1e-12).unwrap_or(F::epsilon()));
    let half =
        F::from(0.5).ok_or_else(|| LinalgError::ComputationError("Cannot convert 0.5".into()))?;

    let eye = Array2::<F>::eye(n);
    let mut s = a.to_owned();

    for iter in 0..max_it {
        let s_inv = crate::inv(&s.view(), None)?;
        let s_new = (&s + &s_inv) * half;
        let diff = frobenius_norm(&(&s_new - &s));
        s = s_new;

        if diff < eps {
            let residual = frobenius_norm(&(s.dot(&s) - &eye));
            return Ok(MatrixSignResult {
                sign: s,
                iterations: iter + 1,
                residual,
            });
        }
    }

    let residual = frobenius_norm(&(s.dot(&s) - &eye));
    Ok(MatrixSignResult {
        sign: s,
        iterations: max_it,
        residual,
    })
}

// ===================================================================
// Matrix p-th root -- Newton-Schulz iteration
// ===================================================================

/// Result of the matrix p-th root computation.
#[derive(Debug, Clone)]
pub struct MatrixPthRootResult<F> {
    /// The p-th root R such that R^p ~ A.
    pub root: Array2<F>,
    /// Number of iterations used.
    pub iterations: usize,
    /// Final residual.
    pub residual: F,
}

/// Compute the matrix p-th root via a coupled Newton iteration.
///
/// Finds R such that R^p = A using the iteration:
///   R_{k+1} = ((p-1) R_k + R_k^{1-p} A) / p
///
/// with simplification via the inverse iterates:
///   R_{k+1} = R_k * ((p-1)I + M_k) / p
///   M_{k+1} = ((p-1)I + M_k)^{-p} * M_k   (where M_k approximates R_k^{-p} A)
///
/// A simpler formulation is used: coupled Newton for p-th root.
///
/// # Arguments
/// * `a`       - Input nonsingular square matrix
/// * `p`       - Root degree (>= 2)
/// * `max_iter`- Maximum iterations (default 200)
/// * `tol`     - Convergence tolerance (default 1e-10)
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions_extended::pth_root;
///
/// // Cube root of diag(8, 27) = diag(2, 3)
/// let a = array![[8.0_f64, 0.0], [0.0, 27.0]];
/// let res = pth_root(&a.view(), 3, None, None).expect("pth_root failed");
/// assert!((res.root[[0, 0]] - 2.0_f64).abs() < 1e-4);
/// assert!((res.root[[1, 1]] - 3.0_f64).abs() < 1e-4);
/// ```
pub fn pth_root<F: MFFloat>(
    a: &ArrayView2<F>,
    p: usize,
    max_iter: Option<usize>,
    tol: Option<F>,
) -> LinalgResult<MatrixPthRootResult<F>> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError("Matrix must be square".into()));
    }
    if p < 2 {
        return Err(LinalgError::ValueError("Root degree p must be >= 2".into()));
    }

    let max_it = max_iter.unwrap_or(200);
    let eps = tol.unwrap_or_else(|| F::from(1e-10).unwrap_or(F::epsilon()));

    let eye = Array2::<F>::eye(n);
    let p_f = F::from(p)
        .ok_or_else(|| LinalgError::ComputationError("Cannot convert p to float".into()))?;
    let p_minus_1 =
        F::from(p - 1).ok_or_else(|| LinalgError::ComputationError("Cannot convert p-1".into()))?;
    let inv_p = F::one() / p_f;

    // Coupled Newton iteration for p-th root
    // X_{k+1} = ((p-1) X_k + X_k^{1-p} A) / p
    // M_{k+1} = M_k ((p-1) I + M_k)^{-1}... simplified as:
    // Use M_k = X_k^{-1} A approach:
    //   X_{k+1} = X_k * (p_minus_1 * I + M_k) / p
    //   M_{k+1} = (p_minus_1 * I + M_k)^{-(p-1)} * M_k^{p-1} ... too complex.
    //
    // Instead use the DB-style coupled iteration for p-th root:
    //   X_{k+1} = X_k * ((p-1)I + M_k) * inv_p
    //   M_{k+1} = inv_p^p * ((p-1)I + M_k)^{-p} * M_k   -- but simplified:
    //
    // Actually, use the standard coupled iteration (Iannazzo 2006):
    //   X_{k+1} = X_k * (I + (p-1)(I - M_k)/p)
    //            = X_k * ((p-1+p)I - (p-1)M_k) / p   -- no.
    //
    // Simplest correct approach: Newton on f(X) = X^p - A.
    //   X_{k+1} = X_k - (X_k^p - A) * (p * X_k^{p-1})^{-1}
    //           = X_k - X_k * (I - X_k^{-p} * A) / p
    //           = X_k * ((p-1)I + X_k^{-p} * A) / p

    // Start with X_0 = I * ||A||^{1/p}
    let a_norm = frobenius_norm(&a.to_owned());
    let scale = a_norm.powf(F::one() / p_f);
    let mut x_k = &eye * scale;

    for iter in 0..max_it {
        let x_old = x_k.clone();

        // Compute X_k^{-1}
        let x_inv = crate::inv(&x_k.view(), None)?;

        // Compute X_k^{-p} * A iteratively: start with x_inv, multiply (p-1) more times
        let mut x_inv_p_a = x_inv.clone();
        for _ in 1..p {
            x_inv_p_a = x_inv_p_a.dot(&x_inv);
        }
        x_inv_p_a = x_inv_p_a.dot(a);

        // X_{k+1} = X_k * ((p-1)I + X_k^{-p} * A) / p
        let bracket = &eye * p_minus_1 + &x_inv_p_a;
        x_k = x_k.dot(&bracket) * inv_p;

        let diff = frobenius_norm(&(&x_k - &x_old));
        if diff < eps {
            // Compute residual: ||X^p - A||
            let mut x_p = x_k.clone();
            for _ in 1..p {
                x_p = x_p.dot(&x_k);
            }
            let residual = frobenius_norm(&(x_p - a));
            return Ok(MatrixPthRootResult {
                root: x_k,
                iterations: iter + 1,
                residual,
            });
        }
    }

    let mut x_p = x_k.clone();
    for _ in 1..p {
        x_p = x_p.dot(&x_k);
    }
    let residual = frobenius_norm(&(x_p - a));
    Ok(MatrixPthRootResult {
        root: x_k,
        iterations: max_it,
        residual,
    })
}

// ===================================================================
// Condition number for matrix functions
// ===================================================================

/// Estimate the condition number of a matrix function f(A) using
/// finite differences.
///
/// cond_f(A) ~ ||f(A + epsilon*E) - f(A)||_F / (epsilon * ||E||_F) * ||A||_F / ||f(A)||_F
///
/// where E is a random perturbation direction, averaged over several trials.
///
/// # Arguments
/// * `a`    - Input square matrix
/// * `f`    - Matrix function to evaluate
/// * `n_samples` - Number of random perturbation directions (default 5)
///
/// # Example
/// ```
/// use scirs2_core::ndarray::{Array2, ArrayView2};
/// use scirs2_linalg::matrix_functions_extended::matrix_function_condition;
/// use scirs2_linalg::error::LinalgResult;
///
/// fn my_exp(a: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
///     scirs2_linalg::matrix_functions::expm(a, None)
/// }
///
/// let a = scirs2_core::ndarray::array![[1.0, 0.1], [0.0, 2.0]];
/// let cond = matrix_function_condition(&a.view(), my_exp, None).expect("cond failed");
/// assert!(cond > 0.0);
/// ```
pub fn matrix_function_condition<F: MFFloat>(
    a: &ArrayView2<F>,
    f: fn(&ArrayView2<F>) -> LinalgResult<Array2<F>>,
    n_samples: Option<usize>,
) -> LinalgResult<F> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError("Matrix must be square".into()));
    }

    let samples = n_samples.unwrap_or(5);
    let eps_val = F::from(1e-7).unwrap_or(F::epsilon());

    let f_a = f(a)?;
    let norm_a = frobenius_norm(&a.to_owned());
    let norm_fa = frobenius_norm(&f_a);

    if norm_fa < F::epsilon() {
        return Ok(F::infinity());
    }

    let mut max_cond = F::zero();

    // Use deterministic perturbation directions
    for trial in 0..samples {
        let mut e_mat = Array2::<F>::zeros((n, n));
        // Create a structured perturbation
        for i in 0..n {
            for j in 0..n {
                // Simple deterministic pattern: use trial index to vary direction
                let val = F::from((i + j + trial) % 7).unwrap_or(F::one())
                    - F::from(3).unwrap_or(F::zero());
                e_mat[[i, j]] = val;
            }
        }
        let norm_e = frobenius_norm(&e_mat);
        if norm_e < F::epsilon() {
            continue;
        }
        // Normalize
        let e_norm = &e_mat * (F::one() / norm_e);

        let a_pert = a + &(&e_norm * eps_val);
        let f_pert = f(&a_pert.view())?;
        let diff = &f_pert - &f_a;
        let norm_diff = frobenius_norm(&diff);

        let cond_est = (norm_diff / eps_val) * (norm_a / norm_fa);
        if cond_est > max_cond {
            max_cond = cond_est;
        }
    }

    Ok(max_cond)
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    // --- Matrix square root (Denman-Beavers) ---

    #[test]
    fn test_sqrt_db_diagonal() {
        let a = array![[4.0, 0.0], [0.0, 9.0]];
        let res = sqrt_denman_beavers(&a.view(), None, None).expect("sqrt failed");
        assert_abs_diff_eq!(res.sqrt[[0, 0]], 2.0, epsilon = 1e-8);
        assert_abs_diff_eq!(res.sqrt[[1, 1]], 3.0, epsilon = 1e-8);
        assert!(res.residual < 1e-8);
    }

    #[test]
    fn test_sqrt_db_identity() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let res = sqrt_denman_beavers(&a.view(), None, None).expect("sqrt failed");
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(res.sqrt[[i, j]], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_sqrt_db_general() {
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let res = sqrt_denman_beavers(&a.view(), None, None).expect("sqrt failed");
        // Verify S*S ~ A
        let ss = res.sqrt.dot(&res.sqrt);
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(ss[[i, j]], a[[i, j]], epsilon = 1e-8);
            }
        }
    }

    // --- Matrix square root (Schur) ---

    #[test]
    fn test_sqrt_schur_diagonal() {
        let a = array![[4.0, 0.0], [0.0, 9.0]];
        let s = sqrt_schur(&a.view()).expect("sqrt failed");
        assert_abs_diff_eq!(s[[0, 0]], 2.0, epsilon = 1e-4);
        assert_abs_diff_eq!(s[[1, 1]], 3.0, epsilon = 1e-4);
    }

    #[test]
    fn test_sqrt_schur_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let s = sqrt_schur(&eye.view()).expect("sqrt failed");
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(s[[i, j]], expected, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_sqrt_schur_general_2x2() {
        let a = array![[5.0, 2.0], [2.0, 5.0]];
        let s = sqrt_schur(&a.view()).expect("sqrt failed");
        let ss = s.dot(&s);
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(ss[[i, j]], a[[i, j]], epsilon = 1e-4);
            }
        }
    }

    // --- Matrix logarithm (ISS) ---

    #[test]
    fn test_logm_iss_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let l = logm_iss(&eye.view(), None).expect("logm failed");
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(l[[i, j]], 0.0, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_logm_iss_diagonal() {
        let e_val = std::f64::consts::E;
        let a = array![[e_val, 0.0], [0.0, e_val * e_val]];
        let l = logm_iss(&a.view(), None).expect("logm failed");
        assert_abs_diff_eq!(l[[0, 0]], 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(l[[1, 1]], 2.0, epsilon = 1e-4);
        assert!(l[[0, 1]].abs() < 1e-4);
        assert!(l[[1, 0]].abs() < 1e-4);
    }

    // --- Matrix sign (Roberts) ---

    #[test]
    fn test_sign_positive_definite() {
        let a = array![[2.0, 0.0], [0.0, 3.0]];
        let res = sign_roberts(&a.view(), None, None).expect("sign failed");
        assert_abs_diff_eq!(res.sign[[0, 0]], 1.0, epsilon = 1e-8);
        assert_abs_diff_eq!(res.sign[[1, 1]], 1.0, epsilon = 1e-8);
        assert!(res.residual < 1e-8);
    }

    #[test]
    fn test_sign_mixed_spectrum() {
        // sign(diag(2, -3)) = diag(1, -1)
        let a = array![[2.0, 0.0], [0.0, -3.0]];
        let res = sign_roberts(&a.view(), None, None).expect("sign failed");
        assert_abs_diff_eq!(res.sign[[0, 0]], 1.0, epsilon = 1e-8);
        assert_abs_diff_eq!(res.sign[[1, 1]], -1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_sign_involutory() {
        // sign(A)^2 = I
        let a = array![[3.0, 1.0], [0.0, -2.0]];
        let res = sign_roberts(&a.view(), None, None).expect("sign failed");
        let ss = res.sign.dot(&res.sign);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(ss[[i, j]], expected, epsilon = 1e-6);
            }
        }
    }

    // --- p-th root ---

    #[test]
    fn test_pth_root_cube_root_diagonal() {
        let a = array![[8.0, 0.0], [0.0, 27.0]];
        let res = pth_root(&a.view(), 3, None, None).expect("pth_root failed");
        assert_abs_diff_eq!(res.root[[0, 0]], 2.0, epsilon = 1e-4);
        assert_abs_diff_eq!(res.root[[1, 1]], 3.0, epsilon = 1e-4);
    }

    #[test]
    fn test_pth_root_square_root() {
        // p=2 should give the same as matrix square root
        let a = array![[4.0, 0.0], [0.0, 16.0]];
        let res = pth_root(&a.view(), 2, None, None).expect("pth_root failed");
        assert_abs_diff_eq!(res.root[[0, 0]], 2.0, epsilon = 1e-4);
        assert_abs_diff_eq!(res.root[[1, 1]], 4.0, epsilon = 1e-4);
    }

    #[test]
    fn test_pth_root_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let res = pth_root(&eye.view(), 5, None, None).expect("pth_root failed");
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(res.root[[i, j]], expected, epsilon = 1e-6);
            }
        }
    }

    // --- Condition number ---

    #[test]
    fn test_condition_number_exp() {
        fn my_exp(a: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
            crate::matrix_functions::expm(a, None)
        }
        let a = array![[1.0, 0.0], [0.0, 2.0]];
        let cond = matrix_function_condition(&a.view(), my_exp, Some(3)).expect("cond failed");
        assert!(cond > 0.0);
        assert!(cond < 1e10);
    }

    #[test]
    fn test_sqrt_db_nonsquare_error() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let res = sqrt_denman_beavers(&a.view(), None, None);
        assert!(res.is_err());
    }

    #[test]
    fn test_pth_root_invalid_p() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let res = pth_root(&a.view(), 1, None, None);
        assert!(res.is_err());
    }
}
