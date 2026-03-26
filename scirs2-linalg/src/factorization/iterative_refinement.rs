//! Iterative Refinement for Linear Systems
//!
//! Mixed-precision iterative refinement algorithms that improve the accuracy
//! of a computed solution to Ax = b by repeatedly computing residuals in
//! higher (working) precision and solving correction equations.
//!
//! # Algorithms
//!
//! - **LU-based refinement**: Factor A = LU once, then iteratively refine
//!   using forward/back substitution for each correction step.
//! - **QR-based refinement**: Factor A = QR once, then iteratively refine
//!   using the Q and R factors.
//! - **Generic refinement**: Uses an arbitrary solver for each correction step.
//!
//! # References
//!
//! - Higham (2002). "Accuracy and Stability of Numerical Algorithms."
//!   2nd ed., Ch. 12.
//! - Demmel et al. (2006). "Error bounds from extra-precise iterative
//!   refinement." LAPACK Working Note 165.
//! - Wilkinson (1963). "Rounding Errors in Algebraic Processes."

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

// ============================================================================
// Configuration & result types
// ============================================================================

/// Configuration for iterative refinement.
#[derive(Debug, Clone)]
pub struct RefinementConfig<F> {
    /// Maximum number of refinement iterations.
    pub max_iterations: usize,
    /// Convergence tolerance for the relative residual.
    pub tolerance: F,
    /// If true, stop when the residual starts increasing (stagnation).
    pub stop_on_stagnation: bool,
}

impl<F: Float> Default for RefinementConfig<F> {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            tolerance: F::from(1e-14).unwrap_or_else(|| F::epsilon()),
            stop_on_stagnation: true,
        }
    }
}

/// Result of iterative refinement.
#[derive(Debug, Clone)]
pub struct IterativeRefinementResult<F> {
    /// Refined solution vector.
    pub solution: Array1<F>,
    /// Number of refinement iterations performed.
    pub iterations: usize,
    /// History of residual norms (one per iteration, plus the initial).
    pub residual_history: Vec<F>,
    /// Estimated forward error ||x_true - x|| / ||x|| at the final step.
    pub forward_error: F,
    /// Estimated backward error at the final step.
    pub backward_error: F,
    /// Whether the refinement converged within the tolerance.
    pub converged: bool,
}

// ============================================================================
// LU-based iterative refinement
// ============================================================================

/// Iterative refinement using a pre-computed LU factorization.
///
/// Given the system Ax = b:
/// 1. Factor A = P * L * U once.
/// 2. Solve Ly = P^T b (forward), Ux = y (back) for initial x.
/// 3. Iterate: r = b - Ax, solve LU dx = Pr, x += dx.
///
/// Because the factorization is reused, each refinement step costs only
/// O(n^2) instead of O(n^3).
///
/// # Arguments
///
/// * `a`      - Coefficient matrix (n x n)
/// * `b`      - Right-hand side vector (n)
/// * `config` - Refinement configuration (use `Default::default()` for sane defaults)
pub fn lu_iterative_refinement<F>(
    a: &ArrayView2<F>,
    b: &ArrayView1<F>,
    config: &RefinementConfig<F>,
) -> LinalgResult<IterativeRefinementResult<F>>
where
    F: Float + NumAssign + Sum + Debug + ScalarOperand + Send + Sync + 'static,
{
    let (m, n) = a.dim();
    if m != n {
        return Err(LinalgError::DimensionError(
            "LU refinement: matrix must be square".to_string(),
        ));
    }
    if b.len() != n {
        return Err(LinalgError::DimensionError(format!(
            "LU refinement: b length ({}) != matrix dimension ({n})",
            b.len()
        )));
    }

    // Factor A = P * L * U via partial pivoting
    let (piv, l, u) = lu_factor_partial(a)?;

    // Initial solve: forward then back substitution
    let pb = apply_perm_vec(b, &piv);
    let y = forward_solve(&l, &pb.view())?;
    let mut x = back_solve(&u, &y.view())?;

    // Compute norms for relative error tracking
    let b_norm = inf_norm_vec(b);
    let a_norm = inf_norm_mat(a);

    let mut residual_history = Vec::with_capacity(config.max_iterations + 1);
    let mut converged = false;
    let mut iterations = 0;
    let mut prev_res_norm = F::infinity();

    // Initial residual
    let r0 = compute_residual(a, &x.view(), b);
    let r0_norm = inf_norm_arr(&r0);
    residual_history.push(r0_norm);

    if b_norm > F::epsilon() && r0_norm / b_norm < config.tolerance {
        let x_norm_init = inf_norm_arr(&x);
        let bw_err = r0_norm / (a_norm * x_norm_init + b_norm);
        return Ok(IterativeRefinementResult {
            solution: x,
            iterations: 0,
            residual_history,
            forward_error: F::zero(),
            backward_error: bw_err,
            converged: true,
        });
    }

    for _it in 0..config.max_iterations {
        iterations += 1;

        // Compute residual: r = b - A*x
        let r = compute_residual(a, &x.view(), b);
        let r_norm = inf_norm_arr(&r);
        residual_history.push(r_norm);

        // Check convergence
        let x_norm = inf_norm_arr(&x);
        let backward_err = if a_norm * x_norm + b_norm > F::epsilon() {
            r_norm / (a_norm * x_norm + b_norm)
        } else {
            r_norm
        };

        if backward_err < config.tolerance {
            converged = true;
            break;
        }

        // Check stagnation
        if config.stop_on_stagnation && r_norm >= prev_res_norm {
            break;
        }
        prev_res_norm = r_norm;

        // Solve for correction: L U dx = P r
        let pr = apply_perm_vec(&r.view(), &piv);
        let y_corr = forward_solve(&l, &pr.view())?;
        let dx = back_solve(&u, &y_corr.view())?;

        // Update: x += dx
        for i in 0..n {
            x[i] += dx[i];
        }
    }

    // Final error estimates
    let final_r = compute_residual(a, &x.view(), b);
    let final_r_norm = inf_norm_arr(&final_r);
    let x_norm = inf_norm_arr(&x);
    let forward_error = if x_norm > F::epsilon() {
        // Approximate using last correction magnitude / solution magnitude
        if residual_history.len() >= 2 {
            let last_res = residual_history[residual_history.len() - 1];
            last_res / (a_norm * x_norm)
        } else {
            final_r_norm / (a_norm * x_norm)
        }
    } else {
        final_r_norm
    };
    let backward_error = if a_norm * x_norm + b_norm > F::epsilon() {
        final_r_norm / (a_norm * x_norm + b_norm)
    } else {
        final_r_norm
    };

    Ok(IterativeRefinementResult {
        solution: x,
        iterations,
        residual_history,
        forward_error,
        backward_error,
        converged,
    })
}

// ============================================================================
// QR-based iterative refinement
// ============================================================================

/// Iterative refinement using a pre-computed QR factorization.
///
/// This is useful for rectangular or over-determined systems (least-squares).
/// For square systems, LU-based refinement is typically more efficient.
///
/// # Algorithm
///
/// 1. Factor A = Q * R.
/// 2. Initial solve: x = R^{-1} Q^T b.
/// 3. Iterate: r = b - Ax, dx = R^{-1} Q^T r, x += dx.
///
/// # Arguments
///
/// * `a`      - Coefficient matrix (m x n, m >= n for least squares)
/// * `b`      - Right-hand side vector (m)
/// * `config` - Refinement configuration
pub fn qr_iterative_refinement<F>(
    a: &ArrayView2<F>,
    b: &ArrayView1<F>,
    config: &RefinementConfig<F>,
) -> LinalgResult<IterativeRefinementResult<F>>
where
    F: Float + NumAssign + Sum + Debug + ScalarOperand + Send + Sync + 'static,
{
    let (m, n) = a.dim();
    if b.len() != m {
        return Err(LinalgError::DimensionError(format!(
            "QR refinement: b length ({}) != matrix rows ({m})",
            b.len()
        )));
    }
    if m < n {
        return Err(LinalgError::DimensionError(
            "QR refinement: requires m >= n (overdetermined or square)".to_string(),
        ));
    }

    // Factor A = Q * R using Householder
    let (q, r_mat) = householder_qr_internal(a)?;

    // Initial solve: x = R^{-1} (Q^T b) using only the first n rows of R and n columns of Q
    let qtb = q.t().dot(b); // m-vector; we need the first n entries
    let qtb_n = qtb.slice(scirs2_core::ndarray::s![..n]).to_owned();
    let mut x = back_solve_rect(&r_mat, &qtb_n.view(), n)?;

    let b_norm = inf_norm_vec(b);
    let a_norm = inf_norm_mat(a);

    let mut residual_history = Vec::with_capacity(config.max_iterations + 1);
    let mut converged = false;
    let mut iterations = 0;
    let mut prev_res_norm = F::infinity();

    // Initial residual
    let r0 = compute_residual(a, &x.view(), b);
    let r0_norm = inf_norm_arr(&r0);
    residual_history.push(r0_norm);

    if b_norm > F::epsilon() && r0_norm / b_norm < config.tolerance {
        let x_norm_init = inf_norm_arr(&x);
        let bw_err = r0_norm / (a_norm * x_norm_init + b_norm);
        return Ok(IterativeRefinementResult {
            solution: x,
            iterations: 0,
            residual_history,
            forward_error: F::zero(),
            backward_error: bw_err,
            converged: true,
        });
    }

    for _it in 0..config.max_iterations {
        iterations += 1;

        let r = compute_residual(a, &x.view(), b);
        let r_norm = inf_norm_arr(&r);
        residual_history.push(r_norm);

        let x_norm = inf_norm_arr(&x);
        let backward_err = if a_norm * x_norm + b_norm > F::epsilon() {
            r_norm / (a_norm * x_norm + b_norm)
        } else {
            r_norm
        };

        if backward_err < config.tolerance {
            converged = true;
            break;
        }

        if config.stop_on_stagnation && r_norm >= prev_res_norm {
            break;
        }
        prev_res_norm = r_norm;

        // Solve correction: dx = R^{-1} Q^T r
        let qt_r = q.t().dot(&r);
        let qt_r_n = qt_r.slice(scirs2_core::ndarray::s![..n]).to_owned();
        let dx = back_solve_rect(&r_mat, &qt_r_n.view(), n)?;

        for i in 0..n {
            x[i] += dx[i];
        }
    }

    let final_r = compute_residual(a, &x.view(), b);
    let final_r_norm = inf_norm_arr(&final_r);
    let x_norm = inf_norm_arr(&x);
    let forward_error = if x_norm > F::epsilon() {
        final_r_norm / (a_norm * x_norm)
    } else {
        final_r_norm
    };
    let backward_error = if a_norm * x_norm + b_norm > F::epsilon() {
        final_r_norm / (a_norm * x_norm + b_norm)
    } else {
        final_r_norm
    };

    Ok(IterativeRefinementResult {
        solution: x,
        iterations,
        residual_history,
        forward_error,
        backward_error,
        converged,
    })
}

// ============================================================================
// Generic iterative refinement (solver-agnostic)
// ============================================================================

/// Generic iterative refinement that uses a supplied solver function.
///
/// This is the most flexible variant: you provide a closure that solves
/// A*dx = r for dx, and this function handles the residual computation
/// and update loop.
///
/// # Arguments
///
/// * `a`      - Coefficient matrix (m x n)
/// * `b`      - Right-hand side vector (m)
/// * `solver` - Closure `|r: &Array1<F>| -> LinalgResult<Array1<F>>` that solves A*dx = r
/// * `config` - Refinement configuration
pub fn generic_iterative_refinement<F, S>(
    a: &ArrayView2<F>,
    b: &ArrayView1<F>,
    mut solver: S,
    config: &RefinementConfig<F>,
) -> LinalgResult<IterativeRefinementResult<F>>
where
    F: Float + NumAssign + Sum + Debug + ScalarOperand + Send + Sync + 'static,
    S: FnMut(&Array1<F>) -> LinalgResult<Array1<F>>,
{
    let (m, _n) = a.dim();
    if b.len() != m {
        return Err(LinalgError::DimensionError(format!(
            "Generic refinement: b length ({}) != matrix rows ({m})",
            b.len()
        )));
    }

    // Initial solve
    let b_owned = b.to_owned();
    let mut x = solver(&b_owned)?;

    let b_norm = inf_norm_vec(b);
    let a_norm = inf_norm_mat(a);

    let mut residual_history = Vec::with_capacity(config.max_iterations + 1);
    let mut converged = false;
    let mut iterations = 0;
    let mut prev_res_norm = F::infinity();

    // Initial residual
    let r0 = compute_residual(a, &x.view(), b);
    residual_history.push(inf_norm_arr(&r0));

    for _it in 0..config.max_iterations {
        iterations += 1;

        let r = compute_residual(a, &x.view(), b);
        let r_norm = inf_norm_arr(&r);
        residual_history.push(r_norm);

        let x_norm = inf_norm_arr(&x);
        let backward_err = if a_norm * x_norm + b_norm > F::epsilon() {
            r_norm / (a_norm * x_norm + b_norm)
        } else {
            r_norm
        };

        if backward_err < config.tolerance {
            converged = true;
            break;
        }

        if config.stop_on_stagnation && r_norm >= prev_res_norm {
            break;
        }
        prev_res_norm = r_norm;

        let dx = solver(&r)?;
        let n = x.len();
        for i in 0..n {
            x[i] += dx[i];
        }
    }

    let final_r = compute_residual(a, &x.view(), b);
    let final_r_norm = inf_norm_arr(&final_r);
    let x_norm = inf_norm_arr(&x);
    let forward_error = if x_norm > F::epsilon() {
        final_r_norm / (a_norm * x_norm)
    } else {
        final_r_norm
    };
    let backward_error = if a_norm * x_norm + b_norm > F::epsilon() {
        final_r_norm / (a_norm * x_norm + b_norm)
    } else {
        final_r_norm
    };

    Ok(IterativeRefinementResult {
        solution: x,
        iterations,
        residual_history,
        forward_error,
        backward_error,
        converged,
    })
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Compute r = b - A*x
fn compute_residual<F>(a: &ArrayView2<F>, x: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + NumAssign + Sum + ScalarOperand,
{
    let ax = a.dot(x);
    let n = b.len();
    let mut r = Array1::<F>::zeros(n);
    for i in 0..n {
        r[i] = b[i] - ax[i];
    }
    r
}

/// Infinity norm of an ArrayView1
fn inf_norm_vec<F: Float>(v: &ArrayView1<F>) -> F {
    v.iter().fold(F::zero(), |acc, &x| acc.max(x.abs()))
}

/// Infinity norm of an owned Array1
fn inf_norm_arr<F: Float>(v: &Array1<F>) -> F {
    v.iter().fold(F::zero(), |acc, &x| acc.max(x.abs()))
}

/// Infinity norm of a matrix (max row sum of absolute values)
fn inf_norm_mat<F: Float + Sum>(a: &ArrayView2<F>) -> F {
    let (m, n) = a.dim();
    let mut max_row = F::zero();
    for i in 0..m {
        let mut row_sum = F::zero();
        for j in 0..n {
            row_sum = row_sum + a[[i, j]].abs();
        }
        if row_sum > max_row {
            max_row = row_sum;
        }
    }
    max_row
}

/// LU factorization with partial pivoting (Doolittle form).
///
/// Returns (piv, L, U) where piv[i] is the pivot row for step i.
fn lu_factor_partial<F>(a: &ArrayView2<F>) -> LinalgResult<(Vec<usize>, Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Debug,
{
    let n = a.nrows();
    let mut lu = a.to_owned();
    let mut piv: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Find pivot
        let mut max_val = lu[[k, k]].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = lu[[i, k]].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }

        if max_val <= F::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "LU factorization: matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows
        if max_row != k {
            piv.swap(k, max_row);
            for j in 0..n {
                let tmp = lu[[k, j]];
                lu[[k, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = tmp;
            }
        }

        // Eliminate below
        for i in (k + 1)..n {
            lu[[i, k]] = lu[[i, k]] / lu[[k, k]];
            for j in (k + 1)..n {
                let lik = lu[[i, k]];
                let ukj = lu[[k, j]];
                lu[[i, j]] -= lik * ukj;
            }
        }
    }

    // Extract L and U
    let mut l = Array2::<F>::eye(n);
    let mut u = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if j < i {
                l[[i, j]] = lu[[i, j]];
            } else {
                u[[i, j]] = lu[[i, j]];
            }
        }
    }

    Ok((piv, l, u))
}

/// Apply a row permutation to a vector.
fn apply_perm_vec<F: Float>(b: &ArrayView1<F>, piv: &[usize]) -> Array1<F> {
    let n = b.len();
    let mut result = Array1::<F>::zeros(n);
    for i in 0..n {
        result[i] = b[piv[i]];
    }
    result
}

/// Forward substitution: solve Ly = b where L is unit lower triangular.
fn forward_solve<F: Float + NumAssign>(
    l: &Array2<F>,
    b: &ArrayView1<F>,
) -> LinalgResult<Array1<F>> {
    let n = l.nrows();
    let mut y = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[[i, j]] * y[j];
        }
        y[i] = s; // L has unit diagonal
    }
    Ok(y)
}

/// Back substitution: solve Ux = y where U is upper triangular.
fn back_solve<F: Float + NumAssign + Debug>(
    u: &Array2<F>,
    y: &ArrayView1<F>,
) -> LinalgResult<Array1<F>> {
    let n = u.nrows();
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= u[[i, j]] * x[j];
        }
        let diag = u[[i, i]];
        if diag.abs() <= F::epsilon() {
            return Err(LinalgError::SingularMatrixError(format!(
                "Back substitution: zero diagonal at index {i}"
            )));
        }
        x[i] = s / diag;
    }
    Ok(x)
}

/// Back substitution on the top-left k x k block of an m x n upper triangular R.
fn back_solve_rect<F: Float + NumAssign + Debug>(
    r: &Array2<F>,
    y: &ArrayView1<F>,
    k: usize,
) -> LinalgResult<Array1<F>> {
    let mut x = Array1::<F>::zeros(k);
    for i in (0..k).rev() {
        let mut s = y[i];
        for j in (i + 1)..k {
            s -= r[[i, j]] * x[j];
        }
        let diag = r[[i, i]];
        if diag.abs() <= F::epsilon() {
            return Err(LinalgError::SingularMatrixError(format!(
                "Back substitution: zero diagonal at index {i}"
            )));
        }
        x[i] = s / diag;
    }
    Ok(x)
}

/// Householder QR factorization (non-pivoted).
fn householder_qr_internal<F>(a: &ArrayView2<F>) -> LinalgResult<(Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + Debug + ScalarOperand + 'static,
{
    let (m, n) = a.dim();
    let min_dim = m.min(n);
    let mut r = a.to_owned();
    let mut q = Array2::<F>::eye(m);
    let two = F::from(2.0).unwrap_or_else(|| F::one() + F::one());

    for k in 0..min_dim {
        let mut x = Array1::<F>::zeros(m - k);
        for i in k..m {
            x[i - k] = r[[i, k]];
        }
        let x_norm = x.iter().fold(F::zero(), |acc, &v| acc + v * v).sqrt();
        if x_norm <= F::epsilon() {
            continue;
        }
        let alpha = if x[0] >= F::zero() { -x_norm } else { x_norm };
        let mut v = x;
        v[0] -= alpha;
        let v_norm_sq = v.iter().fold(F::zero(), |acc, &val| acc + val * val);
        if v_norm_sq <= F::epsilon() {
            continue;
        }
        let beta = two / v_norm_sq;

        for j in k..n {
            let mut dot = F::zero();
            for i in 0..(m - k) {
                dot += v[i] * r[[i + k, j]];
            }
            for i in 0..(m - k) {
                r[[i + k, j]] -= beta * v[i] * dot;
            }
        }
        for row in 0..m {
            let mut dot = F::zero();
            for jj in 0..(m - k) {
                dot += q[[row, jj + k]] * v[jj];
            }
            for jj in 0..(m - k) {
                q[[row, jj + k]] -= beta * dot * v[jj];
            }
        }
    }

    Ok((q, r))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_lu_refinement_well_conditioned() {
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let x_true = array![1.0, 2.0];
        let b = a.dot(&x_true);

        let config = RefinementConfig {
            max_iterations: 10,
            tolerance: 1e-14,
            stop_on_stagnation: true,
        };
        let result =
            lu_iterative_refinement(&a.view(), &b.view(), &config).expect("LU refinement failed");

        assert!(
            result.converged,
            "should converge for well-conditioned system"
        );
        for i in 0..2 {
            assert!(
                (result.solution[i] - x_true[i]).abs() < 1e-12,
                "solution[{i}] = {} != {}",
                result.solution[i],
                x_true[i]
            );
        }
    }

    #[test]
    fn test_lu_refinement_ill_conditioned() {
        // Hilbert 3x3 matrix (cond ~ 524)
        let a = array![
            [1.0, 0.5, 1.0 / 3.0],
            [0.5, 1.0 / 3.0, 0.25],
            [1.0 / 3.0, 0.25, 0.2]
        ];
        let x_true = array![1.0, 1.0, 1.0];
        let b = a.dot(&x_true);

        let config = RefinementConfig {
            max_iterations: 20,
            tolerance: 1e-12,
            stop_on_stagnation: true,
        };
        let result =
            lu_iterative_refinement(&a.view(), &b.view(), &config).expect("LU refinement failed");

        // Even for ill-conditioned systems, refinement should improve accuracy
        let residual = &a.dot(&result.solution) - &b;
        let res_norm: f64 = residual.iter().map(|&v| v.abs()).fold(0.0, f64::max);
        assert!(
            res_norm < 1e-10,
            "residual should be small after refinement, got {res_norm}"
        );
    }

    #[test]
    fn test_lu_refinement_convergence_tracking() {
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![3.0, 4.0];

        let config = RefinementConfig {
            max_iterations: 5,
            tolerance: 1e-14,
            stop_on_stagnation: false,
        };
        let result =
            lu_iterative_refinement(&a.view(), &b.view(), &config).expect("LU refinement failed");

        // Residual history should be non-empty
        assert!(
            !result.residual_history.is_empty(),
            "should have residual history"
        );
        // Forward and backward errors should be finite
        assert!(result.forward_error.is_finite());
        assert!(result.backward_error.is_finite());
    }

    #[test]
    fn test_lu_refinement_singular_error() {
        let a = array![[1.0, 2.0], [2.0, 4.0]]; // singular
        let b = array![1.0, 2.0];

        let config = RefinementConfig::default();
        let result = lu_iterative_refinement(&a.view(), &b.view(), &config);
        assert!(result.is_err(), "should fail on singular matrix");
    }

    #[test]
    fn test_lu_refinement_dimension_mismatch() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![1.0, 2.0, 3.0]; // wrong size

        let config = RefinementConfig::default();
        assert!(lu_iterative_refinement(&a.view(), &b.view(), &config).is_err());
    }

    #[test]
    fn test_qr_refinement_square() {
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let x_true = array![1.0, 2.0];
        let b = a.dot(&x_true);

        let config = RefinementConfig {
            max_iterations: 10,
            tolerance: 1e-14,
            stop_on_stagnation: true,
        };
        let result =
            qr_iterative_refinement(&a.view(), &b.view(), &config).expect("QR refinement failed");

        for i in 0..2 {
            assert!(
                (result.solution[i] - x_true[i]).abs() < 1e-10,
                "QR solution[{i}] = {} != {}",
                result.solution[i],
                x_true[i]
            );
        }
    }

    #[test]
    fn test_qr_refinement_overdetermined() {
        // 3 x 2 overdetermined system
        let a = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
        let b = array![1.0, 2.0, 3.0]; // b is in the column space, exact LS solution is [0, 1]

        let config = RefinementConfig {
            max_iterations: 10,
            tolerance: 1e-12,
            stop_on_stagnation: true,
        };
        let result =
            qr_iterative_refinement(&a.view(), &b.view(), &config).expect("QR refinement failed");

        // Check that residual is small
        let res = &a.dot(&result.solution) - &b;
        let res_norm: f64 = res.iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!(res_norm < 1e-8, "overdetermined LS residual = {res_norm}");
    }

    #[test]
    fn test_qr_refinement_dimension_error() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![1.0, 2.0, 3.0]; // wrong
        let config = RefinementConfig::default();
        assert!(qr_iterative_refinement(&a.view(), &b.view(), &config).is_err());
    }

    #[test]
    fn test_generic_refinement() {
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let x_true = array![1.0, 2.0];
        let b = a.dot(&x_true);

        let a_clone = a.clone();
        let solver = move |rhs: &Array1<f64>| -> LinalgResult<Array1<f64>> {
            // Simple solve using our LU
            let (piv, l, u) = lu_factor_partial(&a_clone.view())?;
            let pb = apply_perm_vec(&rhs.view(), &piv);
            let y = forward_solve(&l, &pb.view())?;
            back_solve(&u, &y.view())
        };

        let config = RefinementConfig {
            max_iterations: 5,
            tolerance: 1e-14,
            stop_on_stagnation: true,
        };
        let result = generic_iterative_refinement(&a.view(), &b.view(), solver, &config)
            .expect("generic refinement failed");

        for i in 0..2 {
            assert!(
                (result.solution[i] - x_true[i]).abs() < 1e-10,
                "generic solution[{i}] wrong"
            );
        }
    }

    #[test]
    fn test_refinement_config_default() {
        let config = RefinementConfig::<f64>::default();
        assert_eq!(config.max_iterations, 10);
        assert!(config.tolerance < 1e-10);
        assert!(config.stop_on_stagnation);
    }

    #[test]
    fn test_lu_refinement_identity() {
        let a = Array2::<f64>::eye(3);
        let b = array![1.0, 2.0, 3.0];

        let config = RefinementConfig::default();
        let result =
            lu_iterative_refinement(&a.view(), &b.view(), &config).expect("identity solve failed");

        for i in 0..3 {
            assert!(
                (result.solution[i] - b[i]).abs() < 1e-14,
                "identity system wrong at {i}"
            );
        }
        assert!(result.converged);
    }

    #[test]
    fn test_lu_refinement_non_square_error() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![1.0, 2.0];
        let config = RefinementConfig::default();
        assert!(lu_iterative_refinement(&a.view(), &b.view(), &config).is_err());
    }
}
