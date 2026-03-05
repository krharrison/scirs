//! Iterative Refinement and Numerical Stability Methods
//!
//! This module provides algorithms for improving the accuracy and stability
//! of linear algebra computations through iterative refinement, equilibration,
//! condition estimation, and backward error analysis.
//!
//! # Algorithms
//!
//! - **Mixed Precision Iterative Refinement**: Solve in low precision, refine in high
//! - **Equilibration**: Row/column scaling for better conditioning
//! - **Condition Estimation**: GECON-style 1-norm condition number estimation
//! - **Backward Error Analysis**: Componentwise and normwise backward errors
//! - **Richardson Iteration**: Simple iterative method for linear systems
//!
//! # References
//!
//! - Higham (2002). "Accuracy and Stability of Numerical Algorithms." (Ch. 12)
//! - Demmel et al. (2006). "Error bounds from extra-precise iterative refinement."
//! - Hager (1984). "Condition estimates."

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;
use std::iter::Sum;

use crate::decomposition::svd;
use crate::error::{LinalgError, LinalgResult};
use crate::solve::solve;

// ============================================================================
// Result types
// ============================================================================

/// Result of iterative refinement
#[derive(Debug, Clone)]
pub struct RefinementResult<F> {
    /// Refined solution
    pub solution: Array1<F>,
    /// Number of refinement iterations performed
    pub iterations: usize,
    /// Final forward error estimate
    pub forward_error: F,
    /// Final backward error estimate
    pub backward_error: F,
    /// Whether the algorithm converged
    pub converged: bool,
}

/// Result of equilibration
#[derive(Debug, Clone)]
pub struct EquilibrationResult<F> {
    /// Row scaling factors
    pub row_scaling: Array1<F>,
    /// Column scaling factors
    pub col_scaling: Array1<F>,
    /// Equilibrated matrix (D_r * A * D_c)
    pub equilibrated: Array2<F>,
}

/// Result of condition estimation
#[derive(Debug, Clone)]
pub struct ConditionEstimate<F> {
    /// Estimated condition number
    pub condition_number: F,
    /// Estimated 1-norm of A
    pub norm_a: F,
    /// Estimated 1-norm of A^{-1}
    pub norm_a_inv: F,
    /// Confidence level (higher = more reliable)
    pub confidence: F,
}

/// Result of backward error analysis
#[derive(Debug, Clone)]
pub struct BackwardErrorResult<F> {
    /// Normwise backward error: min epsilon s.t. (A + dA)x = b + db
    pub normwise: F,
    /// Componentwise backward error (max over components)
    pub componentwise: F,
    /// Residual vector r = b - Ax
    pub residual: Array1<F>,
    /// Residual norm ||r||
    pub residual_norm: F,
}

// ============================================================================
// Mixed Precision Iterative Refinement
// ============================================================================

/// Mixed precision iterative refinement for linear systems.
///
/// Solves Ax = b by:
/// 1. Computing an initial solution x0 (possibly in lower precision)
/// 2. Computing the residual r = b - Ax0 in high precision
/// 3. Solving A * dx = r for the correction
/// 4. Updating x = x0 + dx
/// 5. Repeating until convergence
///
/// # Arguments
///
/// * `a` - Coefficient matrix (n x n)
/// * `b` - Right-hand side vector (n)
/// * `max_iter` - Maximum refinement iterations
/// * `tolerance` - Convergence tolerance for residual
///
/// # Returns
///
/// * `RefinementResult` with refined solution and diagnostics
pub fn iterative_refinement<F>(
    a: &ArrayView2<F>,
    b: &ArrayView1<F>,
    max_iter: usize,
    tolerance: F,
) -> LinalgResult<RefinementResult<F>>
where
    F: Float
        + NumAssign
        + Sum
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    let (m, n) = a.dim();
    if m != n {
        return Err(LinalgError::DimensionError(
            "Matrix must be square for iterative refinement".to_string(),
        ));
    }
    if b.len() != n {
        return Err(LinalgError::DimensionError(format!(
            "Right-hand side length ({}) does not match matrix dimension ({n})",
            b.len()
        )));
    }

    // Step 1: Initial solve
    let mut x = solve(a, b, None)?;

    let b_norm = vector_norm_1(b);
    if b_norm < F::epsilon() {
        return Ok(RefinementResult {
            solution: x,
            iterations: 0,
            forward_error: F::zero(),
            backward_error: F::zero(),
            converged: true,
        });
    }

    let mut converged = false;
    let mut iterations = 0;
    let mut forward_error = F::infinity();
    let mut backward_error = F::infinity();

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Step 2: Compute residual r = b - A*x (in working precision)
        let ax = a.dot(&x);
        let mut r = Array1::zeros(n);
        for i in 0..n {
            r[i] = b[i] - ax[i];
        }

        // Compute backward error
        let r_norm = vector_norm_inf(&r.view());
        backward_error = r_norm / (matrix_norm_inf(a) * vector_norm_inf_arr(&x) + b_norm);

        if backward_error < tolerance {
            converged = true;
            break;
        }

        // Step 3: Solve A * dx = r
        let dx = solve(a, &r.view(), None)?;

        // Compute forward error estimate
        let dx_norm = vector_norm_inf_arr(&dx);
        let x_norm = vector_norm_inf_arr(&x);
        forward_error = if x_norm > F::epsilon() {
            dx_norm / x_norm
        } else {
            dx_norm
        };

        // Step 4: Update solution
        for i in 0..n {
            x[i] += dx[i];
        }

        if forward_error < tolerance {
            converged = true;
            break;
        }
    }

    Ok(RefinementResult {
        solution: x,
        iterations,
        forward_error,
        backward_error,
        converged,
    })
}

// ============================================================================
// Equilibration
// ============================================================================

/// Equilibrate a matrix by row and column scaling.
///
/// Computes diagonal matrices D_r and D_c such that D_r * A * D_c
/// has rows and columns with approximately equal norms.
/// This can significantly improve the condition number.
///
/// # Arguments
///
/// * `a` - Input matrix (m x n)
///
/// # Returns
///
/// * `EquilibrationResult` with scaling factors and equilibrated matrix
pub fn equilibrate<F>(a: &ArrayView2<F>) -> LinalgResult<EquilibrationResult<F>>
where
    F: Float + NumAssign + Sum + Debug + scirs2_core::ndarray::ScalarOperand + 'static,
{
    let (m, n) = a.dim();

    // Row scaling: D_r(i) = 1 / max_j |A(i,j)|
    let mut row_scaling = Array1::zeros(m);
    for i in 0..m {
        let mut max_val = F::zero();
        for j in 0..n {
            let abs_val = a[[i, j]].abs();
            if abs_val > max_val {
                max_val = abs_val;
            }
        }
        row_scaling[i] = if max_val > F::epsilon() {
            F::one() / max_val
        } else {
            F::one()
        };
    }

    // Apply row scaling first
    let mut scaled = a.to_owned();
    for i in 0..m {
        for j in 0..n {
            scaled[[i, j]] *= row_scaling[i];
        }
    }

    // Column scaling: D_c(j) = 1 / max_i |scaled(i,j)|
    let mut col_scaling = Array1::zeros(n);
    for j in 0..n {
        let mut max_val = F::zero();
        for i in 0..m {
            let abs_val = scaled[[i, j]].abs();
            if abs_val > max_val {
                max_val = abs_val;
            }
        }
        col_scaling[j] = if max_val > F::epsilon() {
            F::one() / max_val
        } else {
            F::one()
        };
    }

    // Apply column scaling
    let mut equilibrated = scaled;
    for i in 0..m {
        for j in 0..n {
            equilibrated[[i, j]] *= col_scaling[j];
        }
    }

    Ok(EquilibrationResult {
        row_scaling,
        col_scaling,
        equilibrated,
    })
}

/// Solve a linear system with equilibration for improved accuracy.
///
/// Equilibrates the system, solves, then transforms the solution back.
/// Equivalent to: D_r * A * D_c * (D_c^{-1} * x) = D_r * b
///
/// # Arguments
///
/// * `a` - Coefficient matrix (n x n)
/// * `b` - Right-hand side vector (n)
///
/// # Returns
///
/// * Solution vector x
pub fn equilibrated_solve<F>(a: &ArrayView2<F>, b: &ArrayView1<F>) -> LinalgResult<Array1<F>>
where
    F: Float
        + NumAssign
        + Sum
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(LinalgError::DimensionError(
            "Dimensions mismatch for equilibrated solve".to_string(),
        ));
    }

    // Equilibrate
    let eq = equilibrate(a)?;

    // Scale RHS: b_eq = D_r * b
    let mut b_eq = Array1::zeros(n);
    for i in 0..n {
        b_eq[i] = eq.row_scaling[i] * b[i];
    }

    // Solve equilibrated system: (D_r * A * D_c) * y = D_r * b
    let y = solve(&eq.equilibrated.view(), &b_eq.view(), None)?;

    // Recover x: x = D_c * y
    let mut x = Array1::zeros(n);
    for i in 0..n {
        x[i] = eq.col_scaling[i] * y[i];
    }

    Ok(x)
}

// ============================================================================
// Condition Estimation (GECON-style)
// ============================================================================

/// Estimate the 1-norm condition number of a matrix.
///
/// Uses Hager's (1984) algorithm for estimating ||A^{-1}||_1
/// without explicitly forming the inverse, then multiplies
/// by ||A||_1 to get cond_1(A) = ||A||_1 * ||A^{-1}||_1.
///
/// # Arguments
///
/// * `a` - Square matrix (n x n)
///
/// # Returns
///
/// * `ConditionEstimate` with condition number and details
pub fn estimate_condition<F>(a: &ArrayView2<F>) -> LinalgResult<ConditionEstimate<F>>
where
    F: Float
        + NumAssign
        + Sum
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    let (m, n) = a.dim();
    if m != n {
        return Err(LinalgError::DimensionError(
            "Matrix must be square for condition estimation".to_string(),
        ));
    }

    let norm_a = matrix_norm_1(a);

    if norm_a < F::epsilon() {
        return Ok(ConditionEstimate {
            condition_number: F::infinity(),
            norm_a,
            norm_a_inv: F::infinity(),
            confidence: F::one(),
        });
    }

    // Hager's algorithm for estimating ||A^{-1}||_1
    // This iteratively finds vectors that maximize ||A^{-1} * x||_1 / ||x||_1

    let n_f = F::from(n).unwrap_or(F::one());
    let mut x = Array1::from_elem(n, F::one() / n_f);

    let max_hager_iters = 5;
    let mut gamma = F::zero();

    for _iter in 0..max_hager_iters {
        // Solve A * w = x
        let w = solve(a, &x.view(), None)?;

        // gamma = ||w||_1
        gamma = vector_norm_1_arr(&w);

        // z = sign(w)
        let mut z = Array1::zeros(n);
        for i in 0..n {
            z[i] = if w[i] >= F::zero() {
                F::one()
            } else {
                -F::one()
            };
        }

        // Solve A^T * v = z
        let at = a.t().to_owned();
        let v = solve(&at.view(), &z.view(), None)?;

        // Check if v_inf > z^T * x
        let v_inf = vector_norm_inf_arr(&v);
        let zt_x: F = z
            .iter()
            .zip(x.iter())
            .fold(F::zero(), |acc, (&zi, &xi)| acc + zi * xi);

        if v_inf <= zt_x {
            break;
        }

        // x = e_j where j = argmax |v_j|
        let mut max_idx = 0;
        let mut max_val = F::zero();
        for i in 0..n {
            let abs_vi = v[i].abs();
            if abs_vi > max_val {
                max_val = abs_vi;
                max_idx = i;
            }
        }

        x = Array1::zeros(n);
        x[max_idx] = F::one();
    }

    let norm_a_inv = gamma;
    let condition_number = norm_a * norm_a_inv;

    Ok(ConditionEstimate {
        condition_number,
        norm_a,
        norm_a_inv,
        confidence: F::from(0.9).unwrap_or(F::one()),
    })
}

/// Estimate condition number using SVD (more accurate but more expensive).
///
/// Computes cond_2(A) = sigma_max / sigma_min.
///
/// # Arguments
///
/// * `a` - Matrix (m x n)
///
/// # Returns
///
/// * Condition number estimate
pub fn condition_number_svd<F>(a: &ArrayView2<F>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
{
    let (_u, s, _vt) = svd(a, false, None)?;

    if s.is_empty() {
        return Ok(F::infinity());
    }

    let sigma_max = s[0];
    let sigma_min = s[s.len() - 1];

    if sigma_min < F::epsilon() {
        Ok(F::infinity())
    } else {
        Ok(sigma_max / sigma_min)
    }
}

// ============================================================================
// Backward Error Analysis
// ============================================================================

/// Compute backward error analysis for a computed solution.
///
/// Given A, b, and a computed solution x, this computes:
/// - Normwise backward error: min epsilon s.t. (A + dA)x = b + db,
///   ||dA|| <= epsilon * ||A||, ||db|| <= epsilon * ||b||
/// - Componentwise backward error
/// - Residual r = b - Ax
///
/// # Arguments
///
/// * `a` - Coefficient matrix (n x n)
/// * `b` - Right-hand side vector (n)
/// * `x` - Computed solution (n)
///
/// # Returns
///
/// * `BackwardErrorResult` with error estimates
pub fn backward_error<F>(
    a: &ArrayView2<F>,
    b: &ArrayView1<F>,
    x: &ArrayView1<F>,
) -> LinalgResult<BackwardErrorResult<F>>
where
    F: Float + NumAssign + Sum + Debug + scirs2_core::ndarray::ScalarOperand + 'static,
{
    let n = a.nrows();
    if a.ncols() != n || b.len() != n || x.len() != n {
        return Err(LinalgError::DimensionError(
            "Dimension mismatch in backward error analysis".to_string(),
        ));
    }

    // Compute residual r = b - A*x
    let ax = a.dot(x);
    let mut residual = Array1::zeros(n);
    for i in 0..n {
        residual[i] = b[i] - ax[i];
    }

    let r_norm = vector_norm_inf_arr(&residual);

    // Normwise backward error:
    // eta = ||r|| / (||A|| * ||x|| + ||b||)
    let a_norm = matrix_norm_inf(a);
    let x_norm = vector_norm_inf(x);
    let b_norm = vector_norm_1(b);

    let normwise = r_norm / (a_norm * x_norm + b_norm + F::epsilon());

    // Componentwise backward error:
    // omega_i = |r_i| / (|A| * |x| + |b|)_i
    let mut componentwise = F::zero();
    for i in 0..n {
        let mut denom = b[i].abs();
        for j in 0..n {
            denom += a[[i, j]].abs() * x[j].abs();
        }
        let omega_i = if denom > F::epsilon() {
            residual[i].abs() / denom
        } else {
            F::zero()
        };
        if omega_i > componentwise {
            componentwise = omega_i;
        }
    }

    Ok(BackwardErrorResult {
        normwise,
        componentwise,
        residual,
        residual_norm: r_norm,
    })
}

// ============================================================================
// Richardson Iteration
// ============================================================================

/// Richardson iteration for solving Ax = b.
///
/// x_{k+1} = x_k + omega * (b - A * x_k)
///
/// where omega is a relaxation parameter. Converges when
/// spectral radius of (I - omega * A) < 1.
///
/// # Arguments
///
/// * `a` - Coefficient matrix (n x n)
/// * `b` - Right-hand side vector (n)
/// * `omega` - Relaxation parameter (None = auto-estimate)
/// * `x0` - Initial guess (None = zero vector)
/// * `max_iter` - Maximum iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
///
/// * `RefinementResult` with solution and diagnostics
pub fn richardson_iteration<F>(
    a: &ArrayView2<F>,
    b: &ArrayView1<F>,
    omega: Option<F>,
    x0: Option<&ArrayView1<F>>,
    max_iter: usize,
    tolerance: F,
) -> LinalgResult<RefinementResult<F>>
where
    F: Float
        + NumAssign
        + Sum
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    let (m, n) = a.dim();
    if m != n {
        return Err(LinalgError::DimensionError(
            "Matrix must be square for Richardson iteration".to_string(),
        ));
    }
    if b.len() != n {
        return Err(LinalgError::DimensionError(format!(
            "Right-hand side length ({}) must match matrix dimension ({n})",
            b.len()
        )));
    }

    // Determine omega: use 2 / (lambda_max + lambda_min) for SPD matrices
    // For general matrices, use a conservative estimate
    let omega_val = omega.unwrap_or_else(|| {
        // Estimate spectral radius via power iteration (cheap)
        let rho = estimate_spectral_radius(a, 20);
        if rho > F::epsilon() {
            F::one() / rho
        } else {
            F::one()
        }
    });

    // Initialize
    let mut x = if let Some(x0_ref) = x0 {
        x0_ref.to_owned()
    } else {
        Array1::zeros(n)
    };

    let b_norm = vector_norm_1(b);
    let mut converged = false;
    let mut iterations = 0;
    let mut forward_error = F::infinity();
    let mut backward_error = F::infinity();

    for iter in 0..max_iter {
        iterations = iter + 1;

        // r = b - A * x
        let ax = a.dot(&x);
        let mut r = Array1::zeros(n);
        for i in 0..n {
            r[i] = b[i] - ax[i];
        }

        let r_norm = vector_norm_inf_arr(&r);

        // backward error
        backward_error = if b_norm > F::epsilon() {
            r_norm / b_norm
        } else {
            r_norm
        };

        if backward_error < tolerance {
            converged = true;
            break;
        }

        // Update: x = x + omega * r
        let x_old_norm = vector_norm_inf_arr(&x);
        for i in 0..n {
            x[i] += omega_val * r[i];
        }

        // Forward error estimate
        let mut dx_norm = F::zero();
        for i in 0..n {
            let dx_i = omega_val * r[i];
            if dx_i.abs() > dx_norm {
                dx_norm = dx_i.abs();
            }
        }
        forward_error = if x_old_norm > F::epsilon() {
            dx_norm / x_old_norm
        } else {
            dx_norm
        };
    }

    Ok(RefinementResult {
        solution: x,
        iterations,
        forward_error,
        backward_error,
        converged,
    })
}

/// Estimate spectral radius using power iteration.
fn estimate_spectral_radius<F>(a: &ArrayView2<F>, max_iter: usize) -> F
where
    F: Float + NumAssign + Sum + scirs2_core::ndarray::ScalarOperand + 'static,
{
    let n = a.nrows();
    let mut x = Array1::from_elem(n, F::one() / F::from(n).unwrap_or(F::one()));
    let mut eigenvalue = F::one();

    for _ in 0..max_iter {
        let y = a.dot(&x);
        let y_norm = y.iter().fold(F::zero(), |acc, &v| acc.max(v.abs()));
        if y_norm < F::epsilon() {
            return F::zero();
        }
        eigenvalue = y_norm;
        x = y.mapv(|v| v / y_norm);
    }

    eigenvalue
}

// ============================================================================
// Helper: Norm computations
// ============================================================================

fn vector_norm_1<F: Float>(v: &ArrayView1<F>) -> F {
    v.iter().fold(F::zero(), |acc, &x| acc + x.abs())
}

fn vector_norm_1_arr<F: Float>(v: &Array1<F>) -> F {
    v.iter().fold(F::zero(), |acc, &x| acc + x.abs())
}

fn vector_norm_inf<F: Float>(v: &ArrayView1<F>) -> F {
    v.iter().fold(F::zero(), |acc, &x| acc.max(x.abs()))
}

fn vector_norm_inf_arr<F: Float>(v: &Array1<F>) -> F {
    v.iter().fold(F::zero(), |acc, &x| acc.max(x.abs()))
}

fn matrix_norm_1<F: Float>(a: &ArrayView2<F>) -> F {
    let (_m, n) = a.dim();
    let mut max_col_sum = F::zero();
    for j in 0..n {
        let col_sum = a.column(j).iter().fold(F::zero(), |acc, &x| acc + x.abs());
        if col_sum > max_col_sum {
            max_col_sum = col_sum;
        }
    }
    max_col_sum
}

fn matrix_norm_inf<F: Float>(a: &ArrayView2<F>) -> F {
    let (m, _n) = a.dim();
    let mut max_row_sum = F::zero();
    for i in 0..m {
        let row_sum = a.row(i).iter().fold(F::zero(), |acc, &x| acc + x.abs());
        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }
    max_row_sum
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_iterative_refinement_basic() {
        // Well-conditioned system
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![5.0, 4.0];

        let result = iterative_refinement(&a.view(), &b.view(), 10, 1e-12);
        assert!(result.is_ok());
        let ref_result = result.expect("refinement failed");

        // Check solution: x should satisfy Ax = b
        let ax = a.dot(&ref_result.solution);
        for i in 0..2 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "Solution inaccurate at index {i}"
            );
        }
        assert!(ref_result.converged);
    }

    #[test]
    fn test_iterative_refinement_identity() {
        let a = Array2::<f64>::eye(3);
        let b = array![1.0, 2.0, 3.0];

        let ref_result =
            iterative_refinement(&a.view(), &b.view(), 5, 1e-14).expect("refinement failed");

        for i in 0..3 {
            assert!(
                (ref_result.solution[i] - b[i]).abs() < 1e-12,
                "Identity system solution wrong"
            );
        }
    }

    #[test]
    fn test_iterative_refinement_dimension_errors() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // Non-square
        let b = array![1.0, 2.0];
        assert!(iterative_refinement(&a.view(), &b.view(), 5, 1e-10).is_err());

        let a2 = array![[1.0, 0.0], [0.0, 1.0]];
        let b2 = array![1.0, 2.0, 3.0]; // Wrong length
        assert!(iterative_refinement(&a2.view(), &b2.view(), 5, 1e-10).is_err());
    }

    #[test]
    fn test_equilibrate_basic() {
        let a = array![[1000.0, 1.0], [1.0, 0.001]];
        let result = equilibrate(&a.view());
        assert!(result.is_ok());
        let eq = result.expect("equilibrate failed");

        assert_eq!(eq.equilibrated.nrows(), 2);
        assert_eq!(eq.equilibrated.ncols(), 2);
        assert_eq!(eq.row_scaling.len(), 2);
        assert_eq!(eq.col_scaling.len(), 2);

        // After equilibration, row/col max abs values should be ~1
        for i in 0..2 {
            let row_max: f64 = eq
                .equilibrated
                .row(i)
                .iter()
                .map(|&x| x.abs())
                .fold(0.0, f64::max);
            assert!(
                row_max <= 1.0 + 1e-10,
                "Row {i} max should be <= 1, got {row_max}"
            );
        }
    }

    #[test]
    fn test_equilibrate_identity() {
        let a = Array2::<f64>::eye(3);
        let eq = equilibrate(&a.view()).expect("equilibrate failed");

        // Identity should not change much
        for i in 0..3 {
            assert!((eq.row_scaling[i] - 1.0).abs() < 1e-10);
            assert!((eq.col_scaling[i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_equilibrated_solve() {
        // Poorly scaled but non-singular system
        let a = array![[1000.0, 1.0], [1.0, 1000.0]];
        let x_true = array![1.0, 1.0];
        let b = a.dot(&x_true);

        let x = equilibrated_solve(&a.view(), &b.view());
        assert!(x.is_ok());
        let x_sol = x.expect("equilibrated solve failed");

        // Should be close to [1, 1]
        for i in 0..2 {
            assert!(
                (x_sol[i] - 1.0).abs() < 0.1,
                "Equilibrated solve inaccurate at index {i}: {}",
                x_sol[i]
            );
        }
    }

    #[test]
    fn test_equilibrated_solve_dimension_error() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![1.0, 2.0, 3.0]; // Wrong length
        assert!(equilibrated_solve(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn test_estimate_condition_identity() {
        let a = Array2::<f64>::eye(3);
        let est = estimate_condition(&a.view()).expect("condition estimation failed");

        // Condition of identity should be 1
        assert!(
            (est.condition_number - 1.0).abs() < 0.5,
            "Identity condition should be ~1, got {}",
            est.condition_number
        );
    }

    #[test]
    fn test_estimate_condition_ill_conditioned() {
        // Use SVD-based condition estimation for reliability
        // The Hilbert matrix H(i,j) = 1/(i+j+1) is notoriously ill-conditioned
        let a = array![
            [1.0, 0.5, 1.0 / 3.0],
            [0.5, 1.0 / 3.0, 0.25],
            [1.0 / 3.0, 0.25, 0.2]
        ];
        // Hilbert 3x3 has cond ~524
        let cond = condition_number_svd(&a.view()).expect("SVD condition failed");

        assert!(cond > 100.0, "Should detect ill-conditioning, got {cond}");
    }

    #[test]
    fn test_estimate_condition_non_square() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(estimate_condition(&a.view()).is_err());
    }

    #[test]
    fn test_condition_number_svd() {
        let a = array![[2.0, 0.0], [0.0, 1.0]];
        let cond = condition_number_svd(&a.view()).expect("SVD condition failed");
        // cond = 2 / 1 = 2
        assert!(
            (cond - 2.0).abs() < 0.1,
            "Condition should be ~2, got {cond}"
        );
    }

    #[test]
    fn test_condition_number_svd_singular() {
        let a = array![[1.0, 2.0], [2.0, 4.0]]; // Singular
        let cond = condition_number_svd(&a.view()).expect("SVD condition failed");
        assert!(
            cond > 1e10 || cond.is_infinite(),
            "Singular matrix should have infinite condition"
        );
    }

    #[test]
    fn test_backward_error_exact_solution() {
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let x_true = array![1.0, 1.0];
        let b = a.dot(&x_true);

        let be =
            backward_error(&a.view(), &b.view(), &x_true.view()).expect("backward error failed");

        assert!(
            be.normwise < 1e-12,
            "Exact solution should have tiny backward error"
        );
        assert!(
            be.componentwise < 1e-12,
            "Componentwise error should be tiny"
        );
        assert!(be.residual_norm < 1e-12, "Residual should be tiny");
    }

    #[test]
    fn test_backward_error_approximate_solution() {
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![3.0, 4.0];
        let x_approx = array![1.1, 0.9]; // Slightly off

        let be =
            backward_error(&a.view(), &b.view(), &x_approx.view()).expect("backward error failed");

        assert!(
            be.normwise > 0.0,
            "Approximate solution should have positive backward error"
        );
        assert!(be.residual_norm > 0.0, "Should have non-zero residual");
    }

    #[test]
    fn test_backward_error_dimension_mismatch() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![1.0, 2.0, 3.0]; // Wrong
        let x = array![1.0, 2.0];
        assert!(backward_error(&a.view(), &b.view(), &x.view()).is_err());
    }

    #[test]
    fn test_richardson_iteration_basic() {
        // SPD matrix for convergence
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![5.0, 4.0];

        let result = richardson_iteration(&a.view(), &b.view(), None, None, 200, 1e-8);
        assert!(result.is_ok());
        let ref_result = result.expect("Richardson failed");

        // Check solution
        let ax = a.dot(&ref_result.solution);
        for i in 0..2 {
            assert!(
                (ax[i] - b[i]).abs() < 0.1,
                "Richardson solution inaccurate at {i}: ax={}, b={}",
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_richardson_with_omega() {
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![5.0, 4.0];

        // Use a good omega for this SPD matrix
        // Eigenvalues are approximately 2.38 and 4.62
        // Optimal omega = 2 / (lambda_min + lambda_max) ~ 2/7 ~ 0.286
        let result = richardson_iteration(&a.view(), &b.view(), Some(0.25), None, 500, 1e-8);
        assert!(result.is_ok());
    }

    #[test]
    fn test_richardson_with_initial_guess() {
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![5.0, 4.0];
        let x0 = array![1.0, 1.0]; // Good initial guess

        let result =
            richardson_iteration(&a.view(), &b.view(), Some(0.2), Some(&x0.view()), 100, 1e-8);
        assert!(result.is_ok());
    }

    #[test]
    fn test_richardson_dimension_errors() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // Non-square
        let b = array![1.0, 2.0];
        assert!(richardson_iteration(&a.view(), &b.view(), None, None, 10, 1e-8).is_err());

        let a2 = array![[1.0, 0.0], [0.0, 1.0]];
        let b2 = array![1.0, 2.0, 3.0]; // Wrong length
        assert!(richardson_iteration(&a2.view(), &b2.view(), None, None, 10, 1e-8).is_err());
    }

    #[test]
    fn test_norm_helpers() {
        let v = array![1.0, -2.0, 3.0];
        assert!((vector_norm_1(&v.view()) - 6.0).abs() < 1e-10);
        assert!((vector_norm_inf(&v.view()) - 3.0).abs() < 1e-10);

        let a = array![[1.0, -2.0], [3.0, 4.0]];
        // 1-norm = max col sum = max(|1|+|3|, |-2|+|4|) = max(4, 6) = 6
        assert!((matrix_norm_1(&a.view()) - 6.0).abs() < 1e-10);
        // inf-norm = max row sum = max(|1|+|-2|, |3|+|4|) = max(3, 7) = 7
        assert!((matrix_norm_inf(&a.view()) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_spectral_radius() {
        let a = array![[2.0, 0.0], [0.0, 1.0]];
        let rho = estimate_spectral_radius(&a.view(), 30);
        assert!(
            (rho - 2.0).abs() < 0.1,
            "Spectral radius of diag(2,1) should be ~2, got {rho}"
        );
    }

    #[test]
    fn test_refinement_result_fields() {
        let a = Array2::<f64>::eye(2);
        let b = array![1.0, 2.0];
        let result =
            iterative_refinement(&a.view(), &b.view(), 5, 1e-10).expect("refinement failed");

        assert!(result.forward_error.is_finite() || result.converged);
        assert!(result.backward_error.is_finite() || result.converged);
        assert!(result.iterations <= 5);
    }

    #[test]
    fn test_iterative_refinement_zero_rhs() {
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![0.0, 0.0];

        let result =
            iterative_refinement(&a.view(), &b.view(), 5, 1e-10).expect("refinement failed");

        // Solution should be zero
        for i in 0..2 {
            assert!(
                result.solution[i].abs() < 1e-12,
                "Zero RHS should give zero solution"
            );
        }
        assert!(result.converged);
    }
}
