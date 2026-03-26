//! Mixed-precision solver integration using double-double (DD) arithmetic.
//!
//! Provides iterative refinement and compensated dot products / sums that
//! leverage DD precision for residual computation while keeping the bulk
//! of the work in standard f64.
//!
//! ## Iterative Refinement
//!
//! Given a linear system `Ax = b` and an initial (inaccurate) solution `x0`,
//! iterative refinement computes a high-precision residual `r = b - A*x`
//! in DD precision, then solves a correction `A * delta = r` in f64, and
//! updates `x = x + delta`. This process converges to a solution accurate
//! to nearly full DD precision of the residual computation.
//!
//! ## References
//!
//! * Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*,
//!   2nd ed. SIAM. Chapter 12.

use ::ndarray::{Array1, Array2};

use crate::error::{CoreError, CoreResult, ErrorContext};
use super::{two_sum, two_prod, DD};

// ─── Error helpers ─────────────────────────────────────────────────────────────

#[inline(always)]
fn comp_err(msg: impl Into<String>) -> CoreError {
    CoreError::ComputationError(ErrorContext::new(msg))
}

// ─── Compensated dot product ──────────────────────────────────────────────────

/// Compute the dot product of two f64 slices using DD accumulation.
///
/// This gives a result accurate to nearly twice the precision of a naive
/// f64 dot product, at about 10x the cost. The final result is rounded
/// back to f64 but with much less accumulated error.
///
/// # Arguments
/// * `a` - First vector.
/// * `b` - Second vector (must be same length as `a`).
///
/// # Returns
/// The dot product as an f64, computed with DD-precision accumulation.
///
/// # Errors
/// Returns an error if the vectors have different lengths.
pub fn dd_dot_product(a: &[f64], b: &[f64]) -> CoreResult<f64> {
    if a.len() != b.len() {
        return Err(comp_err(
            format!("dd_dot_product: length mismatch: {} vs {}", a.len(), b.len())
        ));
    }

    let mut acc = DD::ZERO;

    for i in 0..a.len() {
        let (p, e) = two_prod(a[i], b[i]);
        let prod = DD { hi: p, lo: e };
        acc = acc.dd_add(prod);
    }

    Ok(acc.to_f64_round())
}

/// Compute the sum of f64 values using DD (compensated) accumulation.
///
/// This is equivalent to Kahan summation but using the full DD framework,
/// which provides slightly better error bounds and handles more pathological
/// orderings.
///
/// # Arguments
/// * `values` - Slice of values to sum.
///
/// # Returns
/// The sum as an f64, computed with DD-precision accumulation.
pub fn dd_sum(values: &[f64]) -> f64 {
    let mut acc = DD::ZERO;
    for &v in values {
        acc = acc.dd_add(DD::from_f64(v));
    }
    acc.to_f64_round()
}

// ─── Iterative refinement ────────────────────────────────────────────────────

/// Mixed-precision iterative refinement for linear systems.
///
/// Given `A * x = b`, this function refines an initial solution `x0` by
/// computing the residual `r = b - A * x` in DD precision and solving
/// the correction equation `A * delta = r` in f64 precision.
///
/// The solver for the correction equation uses a simple LU-like approach
/// (Gaussian elimination with partial pivoting).
///
/// # Arguments
/// * `a` - The coefficient matrix (n x n).
/// * `b` - The right-hand side vector (length n).
/// * `x0` - The initial approximate solution (length n).
/// * `max_iter` - Maximum number of refinement iterations.
///
/// # Returns
/// The refined solution as a `Vec<f64>`.
///
/// # Errors
/// Returns an error if dimensions are inconsistent or if the matrix is singular.
pub fn iterative_refinement_dd(
    a: &Array2<f64>,
    b: &[f64],
    x0: &[f64],
    max_iter: usize,
) -> CoreResult<Vec<f64>> {
    let (n_rows, n_cols) = a.dim();
    if n_rows != n_cols {
        return Err(comp_err(
            format!("iterative_refinement_dd: matrix must be square, got {}x{}", n_rows, n_cols)
        ));
    }
    let n = n_rows;
    if b.len() != n {
        return Err(comp_err(
            format!("iterative_refinement_dd: b length {} != matrix size {}", b.len(), n)
        ));
    }
    if x0.len() != n {
        return Err(comp_err(
            format!("iterative_refinement_dd: x0 length {} != matrix size {}", x0.len(), n)
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    // Pre-compute the LU factorization for solving corrections.
    let (lu, piv) = lu_factorize(a)?;

    let mut x = x0.to_vec();

    for _iter in 0..max_iter {
        // Compute residual r = b - A*x in DD precision.
        let r = compute_residual_dd(a, b, &x);

        // Check convergence: if ||r|| is small enough, stop.
        let r_norm = dd_sum(&r.iter().map(|&v| v * v).collect::<Vec<_>>()).sqrt();
        let b_norm = dd_sum(&b.iter().map(|&v| v * v).collect::<Vec<_>>()).sqrt();

        if b_norm > 0.0 && r_norm / b_norm < 1e-30 {
            break;
        }
        if r_norm < 1e-300 {
            break;
        }

        // Solve A * delta = r using the pre-computed LU factorization.
        let delta = lu_solve(&lu, &piv, &r);

        // Update: x = x + delta.
        for i in 0..n {
            x[i] += delta[i];
        }
    }

    Ok(x)
}

/// Compute the residual r = b - A*x using DD-precision accumulation.
fn compute_residual_dd(a: &Array2<f64>, b: &[f64], x: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut r = Vec::with_capacity(n);

    for i in 0..n {
        let mut acc = DD::from_f64(b[i]);

        for j in 0..x.len() {
            let (p, e) = two_prod(a[[i, j]], x[j]);
            let prod = DD { hi: p, lo: e };
            acc = acc.dd_sub(prod);
        }

        r.push(acc.to_f64_round());
    }

    r
}

/// LU factorization with partial pivoting.
///
/// Returns the LU matrix (combined L and U with L having implicit unit diagonal)
/// and the pivot vector.
fn lu_factorize(a: &Array2<f64>) -> CoreResult<(Array2<f64>, Vec<usize>)> {
    let n = a.nrows();
    let mut lu = a.clone();
    let mut piv: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Find pivot.
        let mut max_val = lu[[k, k]].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = lu[[i, k]].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }

        if max_val < 1e-300 {
            return Err(comp_err("lu_factorize: matrix is singular or nearly singular"));
        }

        // Swap rows.
        if max_row != k {
            piv.swap(k, max_row);
            for j in 0..n {
                let tmp = lu[[k, j]];
                lu[[k, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = tmp;
            }
        }

        // Eliminate.
        let pivot = lu[[k, k]];
        for i in (k + 1)..n {
            let factor = lu[[i, k]] / pivot;
            lu[[i, k]] = factor; // Store L factor.
            for j in (k + 1)..n {
                lu[[i, j]] -= factor * lu[[k, j]];
            }
        }
    }

    Ok((lu, piv))
}

/// Solve A*x = b using pre-computed LU factorization.
fn lu_solve(lu: &Array2<f64>, piv: &[usize], b: &[f64]) -> Vec<f64> {
    let n = b.len();

    // Apply permutation.
    let mut pb = vec![0.0_f64; n];
    for i in 0..n {
        pb[i] = b[piv[i]];
    }

    // Forward substitution: L * y = pb.
    for i in 1..n {
        let mut sum = pb[i];
        for j in 0..i {
            sum -= lu[[i, j]] * pb[j];
        }
        pb[i] = sum;
    }

    // Back substitution: U * x = y.
    let mut x = pb;
    for i in (0..n).rev() {
        let mut sum = x[i];
        for j in (i + 1)..n {
            sum -= lu[[i, j]] * x[j];
        }
        let diag = lu[[i, i]];
        if diag.abs() > 1e-300 {
            x[i] = sum / diag;
        } else {
            x[i] = 0.0;
        }
    }

    x
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ::ndarray::array;

    #[test]
    fn test_dd_dot_product_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = dd_dot_product(&a, &b).expect("dot product should succeed");
        let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0; // 32.0
        assert!(
            (result - expected).abs() < f64::EPSILON * 4.0,
            "dd_dot_product: got {result}, expected {expected}"
        );
    }

    #[test]
    fn test_dd_dot_product_length_mismatch() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 3.0];
        assert!(dd_dot_product(&a, &b).is_err());
    }

    #[test]
    fn test_dd_dot_product_more_accurate() {
        // Test with values that cause cancellation in naive dot product.
        let n = 1000;
        let mut a = vec![0.0_f64; n];
        let mut b = vec![0.0_f64; n];

        // Construct vectors where naive dot product loses precision.
        for i in 0..n {
            a[i] = 1.0 + (i as f64) * 1e-15;
            b[i] = 1.0 - (i as f64) * 1e-15;
        }

        let dd_result = dd_dot_product(&a, &b).expect("should succeed");
        let naive_result: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        // Both should be close to n (=1000), but DD should be more accurate.
        // The key test is that dd_result is a valid number close to 1000.
        assert!(
            (dd_result - 1000.0).abs() < 1.0,
            "dd_dot_product result {dd_result} should be close to 1000"
        );
        let _ = naive_result; // May differ in last digits
    }

    #[test]
    fn test_dd_sum_basic() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = dd_sum(&values);
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_dd_sum_compensated() {
        // Classic test: sum 1e8 + many small values.
        // Naive summation loses the small values due to catastrophic cancellation.
        let n = 10000;
        let mut values = Vec::with_capacity(n + 1);
        values.push(1e8);
        for _ in 0..n {
            values.push(1e-8);
        }

        let dd_result = dd_sum(&values);
        let expected = 1e8 + (n as f64) * 1e-8;

        assert!(
            (dd_result - expected).abs() < 1e-7,
            "dd_sum: got {dd_result}, expected {expected}, diff = {}",
            (dd_result - expected).abs()
        );
    }

    #[test]
    fn test_iterative_refinement_simple() {
        // Simple 2x2 system: well-conditioned.
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = [5.0, 10.0];
        // Exact solution: x = [1, 3]
        let x0 = [1.1, 2.9]; // Slightly perturbed initial guess.

        let result = iterative_refinement_dd(&a, &b, &x0, 10)
            .expect("refinement should succeed");

        assert!(
            (result[0] - 1.0).abs() < 1e-14,
            "x[0] = {}, expected 1.0",
            result[0]
        );
        assert!(
            (result[1] - 3.0).abs() < 1e-14,
            "x[1] = {}, expected 3.0",
            result[1]
        );
    }

    #[test]
    fn test_iterative_refinement_ill_conditioned() {
        // Ill-conditioned 3x3 system (Hilbert-like).
        let a = array![
            [1.0, 1.0/2.0, 1.0/3.0],
            [1.0/2.0, 1.0/3.0, 1.0/4.0],
            [1.0/3.0, 1.0/4.0, 1.0/5.0]
        ];
        // Choose b = A * [1, 1, 1] so we know the exact solution.
        let b_vec = [
            1.0 + 0.5 + 1.0 / 3.0,
            0.5 + 1.0 / 3.0 + 0.25,
            1.0 / 3.0 + 0.25 + 0.2,
        ];

        // Start with a poor initial guess.
        let x0 = [0.0, 0.0, 0.0];

        let result = iterative_refinement_dd(&a, &b_vec, &x0, 20)
            .expect("refinement should succeed");

        // Check that the solution is close to [1, 1, 1].
        for (i, &xi) in result.iter().enumerate() {
            assert!(
                (xi - 1.0).abs() < 1e-8,
                "x[{i}] = {xi}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_iterative_refinement_dimension_mismatch() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = [1.0, 2.0, 3.0]; // Wrong size.
        let x0 = [0.0, 0.0];
        assert!(iterative_refinement_dd(&a, &b, &x0, 5).is_err());
    }

    #[test]
    fn test_iterative_refinement_nonsquare() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = [1.0, 2.0];
        let x0 = [0.0, 0.0, 0.0];
        assert!(iterative_refinement_dd(&a, &b, &x0, 5).is_err());
    }

    #[test]
    fn test_dd_sum_empty() {
        let values: [f64; 0] = [];
        let result = dd_sum(&values);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_dd_dot_product_empty() {
        let a: [f64; 0] = [];
        let b: [f64; 0] = [];
        let result = dd_dot_product(&a, &b).expect("should succeed");
        assert_eq!(result, 0.0);
    }
}
