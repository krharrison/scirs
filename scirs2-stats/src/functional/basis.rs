//! Basis function evaluation and curve smoothing for functional data analysis.
//!
//! Provides B-spline, Fourier, and polynomial basis evaluations, along with
//! penalized least squares smoothing and GCV-based smoothing parameter selection.

use scirs2_core::ndarray::{Array1, Array2, Axis};

use super::types::BasisType;
use crate::error::{StatsError, StatsResult};

/// Evaluate basis functions on a grid.
///
/// Returns a matrix of shape (n_grid, n_basis) where each column is one basis
/// function evaluated at every grid point.
///
/// # Errors
/// Returns an error if the basis configuration is invalid.
pub fn evaluate_basis(grid: &[f64], basis_type: &BasisType) -> StatsResult<Array2<f64>> {
    match basis_type {
        BasisType::BSpline { n_basis, degree } => evaluate_bspline_basis(grid, *n_basis, *degree),
        BasisType::Fourier { n_basis } => evaluate_fourier_basis(grid, *n_basis),
        BasisType::Polynomial { degree } => evaluate_polynomial_basis(grid, *degree),
        _ => Err(StatsError::NotImplementedError(
            "Unknown basis type".to_string(),
        )),
    }
}

/// Evaluate B-spline basis functions using the Cox-de Boor recursion.
///
/// Constructs a uniform knot sequence with appropriate boundary knots
/// and evaluates each B-spline basis function at every grid point.
fn evaluate_bspline_basis(grid: &[f64], n_basis: usize, degree: usize) -> StatsResult<Array2<f64>> {
    if n_basis < degree + 1 {
        return Err(StatsError::InvalidArgument(format!(
            "n_basis ({}) must be >= degree + 1 ({})",
            n_basis,
            degree + 1
        )));
    }
    if grid.is_empty() {
        return Err(StatsError::InvalidInput(
            "Grid must not be empty".to_string(),
        ));
    }

    let n = grid.len();
    let t_min = grid.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = grid.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Number of internal knots
    let n_internal = n_basis - degree - 1;
    // Total knots = n_internal + 2*(degree+1)
    let n_knots = n_internal + 2 * (degree + 1);
    let mut knots = Vec::with_capacity(n_knots);

    // Clamped knot sequence: degree+1 copies of t_min, then internal, then degree+1 copies of t_max
    for _ in 0..=degree {
        knots.push(t_min);
    }
    if n_internal > 0 {
        let step = (t_max - t_min) / (n_internal as f64 + 1.0);
        for j in 1..=n_internal {
            knots.push(t_min + j as f64 * step);
        }
    }
    for _ in 0..=degree {
        knots.push(t_max);
    }

    let mut basis = Array2::<f64>::zeros((n, n_basis));

    for (row, &t) in grid.iter().enumerate() {
        // Evaluate all n_basis B-splines of given degree at point t
        // using the Cox-de Boor recursion
        // B_{i,0}(t) = 1 if knots[i] <= t < knots[i+1], else 0
        // Special case: for the last interval, include the right endpoint

        let order = degree + 1;
        // We need n_basis + degree + 1 knots total, which is n_knots
        // The basis functions are B_{0,degree}, B_{1,degree}, ..., B_{n_basis-1,degree}

        // Start with degree-0 basis functions
        let n_intervals = knots.len() - 1;
        let mut prev = vec![0.0f64; n_intervals];

        // For clamped B-splines at the right boundary:
        // When t == t_max, we need to find the last non-degenerate interval
        // and assign it value 1.0 (this ensures partition of unity at boundary).
        let at_right_boundary = (t - t_max).abs() < 1e-14;
        if at_right_boundary {
            // Find the last interval with non-zero width
            for i in (0..n_intervals).rev() {
                if knots[i] < knots[i + 1] - 1e-14 {
                    prev[i] = 1.0;
                    break;
                }
            }
        } else {
            for i in 0..n_intervals {
                if t >= knots[i] && t < knots[i + 1] {
                    prev[i] = 1.0;
                }
            }
        }

        // Recurse up to desired degree
        for p in 1..order {
            let n_funcs = n_intervals - p;
            let mut curr = vec![0.0f64; n_funcs];
            for i in 0..n_funcs {
                let denom1 = knots[i + p] - knots[i];
                let denom2 = knots[i + p + 1] - knots[i + 1];
                let left = if denom1.abs() > 1e-14 {
                    (t - knots[i]) / denom1 * prev[i]
                } else {
                    0.0
                };
                let right = if denom2.abs() > 1e-14 {
                    (knots[i + p + 1] - t) / denom2 * prev[i + 1]
                } else {
                    0.0
                };
                curr[i] = left + right;
            }
            prev = curr;
        }

        // prev now has exactly n_basis values
        for (j, &val) in prev.iter().enumerate().take(n_basis) {
            basis[[row, j]] = val;
        }
    }

    Ok(basis)
}

/// Evaluate Fourier basis functions on a grid.
///
/// The basis consists of: constant 1, then pairs sin(2*pi*k*t'), cos(2*pi*k*t')
/// where t' is normalized to [0,1].
fn evaluate_fourier_basis(grid: &[f64], n_basis: usize) -> StatsResult<Array2<f64>> {
    if n_basis == 0 {
        return Err(StatsError::InvalidArgument(
            "n_basis must be at least 1".to_string(),
        ));
    }
    if grid.is_empty() {
        return Err(StatsError::InvalidInput(
            "Grid must not be empty".to_string(),
        ));
    }

    let n = grid.len();
    let t_min = grid.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = grid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = t_max - t_min;

    let mut basis = Array2::<f64>::zeros((n, n_basis));

    for (row, &t) in grid.iter().enumerate() {
        let t_norm = if range.abs() > 1e-14 {
            (t - t_min) / range
        } else {
            0.0
        };

        // First basis function: constant
        basis[[row, 0]] = 1.0;

        // Remaining: sin/cos pairs
        let mut col = 1;
        let mut k = 1u32;
        while col < n_basis {
            let arg = 2.0 * std::f64::consts::PI * f64::from(k) * t_norm;
            basis[[row, col]] = arg.sin();
            col += 1;
            if col < n_basis {
                basis[[row, col]] = arg.cos();
                col += 1;
            }
            k += 1;
        }
    }

    Ok(basis)
}

/// Evaluate polynomial basis functions on a grid.
///
/// Basis: 1, t', t'^2, ..., t'^degree  where t' is normalized to [-1,1].
fn evaluate_polynomial_basis(grid: &[f64], degree: usize) -> StatsResult<Array2<f64>> {
    if grid.is_empty() {
        return Err(StatsError::InvalidInput(
            "Grid must not be empty".to_string(),
        ));
    }

    let n = grid.len();
    let n_basis = degree + 1;
    let t_min = grid.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = grid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = t_max - t_min;

    let mut basis = Array2::<f64>::zeros((n, n_basis));

    for (row, &t) in grid.iter().enumerate() {
        // Normalize to [-1, 1]
        let t_norm = if range.abs() > 1e-14 {
            2.0 * (t - t_min) / range - 1.0
        } else {
            0.0
        };
        let mut power = 1.0;
        for d in 0..n_basis {
            basis[[row, d]] = power;
            if d < degree {
                power *= t_norm;
            }
        }
    }

    Ok(basis)
}

/// Smooth a single curve using penalized least squares.
///
/// Minimizes ||y - B*c||^2 + lambda * c' P c
/// where B is the basis matrix, c are coefficients, and P is a second-derivative penalty.
///
/// Returns the smoothed curve evaluated on the grid.
///
/// # Errors
/// Returns an error if the linear system cannot be solved.
pub fn smooth_curve(
    y: &[f64],
    grid: &[f64],
    basis_type: &BasisType,
    lambda: f64,
) -> StatsResult<Vec<f64>> {
    let phi = evaluate_basis(grid, basis_type)?;
    let n_basis = phi.ncols();

    // B'B
    let btb = phi.t().dot(&phi);
    // B'y
    let y_arr = Array1::from_vec(y.to_vec());
    let bty = phi.t().dot(&y_arr);

    // Penalty matrix: approximate second derivative penalty
    let penalty = second_derivative_penalty(n_basis);

    // Solve (B'B + lambda * P) c = B'y
    let mut lhs = btb + &(&penalty * lambda);
    let coeffs = solve_symmetric_positive(&mut lhs, &bty)?;

    // Evaluate smoothed curve: y_hat = B * c
    let y_hat = phi.dot(&coeffs);
    Ok(y_hat.to_vec())
}

/// Compute basis coefficients for a curve using penalized least squares.
///
/// Returns the coefficient vector c that minimizes ||y - B*c||^2 + lambda * c' P c.
pub(crate) fn compute_basis_coefficients(
    y: &[f64],
    phi: &Array2<f64>,
    lambda: f64,
) -> StatsResult<Array1<f64>> {
    let n_basis = phi.ncols();
    let btb = phi.t().dot(phi);
    let y_arr = Array1::from_vec(y.to_vec());
    let bty = phi.t().dot(&y_arr);
    let penalty = second_derivative_penalty(n_basis);
    let mut lhs = btb + &(&penalty * lambda);
    solve_symmetric_positive(&mut lhs, &bty)
}

/// Construct a second-derivative penalty matrix for smoothing.
///
/// Uses finite differences of the coefficient vector to approximate the
/// integrated squared second derivative penalty.
fn second_derivative_penalty(n_basis: usize) -> Array2<f64> {
    if n_basis < 3 {
        return Array2::<f64>::zeros((n_basis, n_basis));
    }
    // D2 matrix: (n_basis - 2) x n_basis
    let m = n_basis - 2;
    let mut d2 = Array2::<f64>::zeros((m, n_basis));
    for i in 0..m {
        d2[[i, i]] = 1.0;
        d2[[i, i + 1]] = -2.0;
        d2[[i, i + 2]] = 1.0;
    }
    // Penalty = D2' * D2
    d2.t().dot(&d2)
}

/// Select smoothing parameter lambda by Generalized Cross-Validation.
///
/// GCV(lambda) = (1/n) * ||y - S_lambda * y||^2 / (1 - tr(S_lambda)/n)^2
/// where S_lambda = B (B'B + lambda*P)^{-1} B' is the smoother matrix.
///
/// # Errors
/// Returns an error if basis evaluation or linear algebra fails.
pub fn gcv_select_lambda(y: &[f64], grid: &[f64], basis_type: &BasisType) -> StatsResult<f64> {
    let phi = evaluate_basis(grid, basis_type)?;
    let n = grid.len() as f64;
    let n_basis = phi.ncols();
    let y_arr = Array1::from_vec(y.to_vec());
    let btb = phi.t().dot(&phi);
    let bty = phi.t().dot(&y_arr);
    let penalty = second_derivative_penalty(n_basis);

    // Search over a grid of log-spaced lambda values
    let mut best_lambda = 1e-4;
    let mut best_gcv = f64::INFINITY;

    let log_lambdas: Vec<f64> = (-8..=6).map(|k| 10.0f64.powi(k)).collect();

    for &lam in &log_lambdas {
        let mut lhs = &btb + &(&penalty * lam);
        let coeffs = match solve_symmetric_positive(&mut lhs, &bty) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Fitted values
        let y_hat = phi.dot(&coeffs);
        let residuals = &y_arr - &y_hat;
        let rss: f64 = residuals.iter().map(|r| r * r).sum();

        // Trace of smoother matrix: tr(S) = tr(B (B'B + lam*P)^{-1} B')
        // = tr((B'B + lam*P)^{-1} B'B)
        // We compute this by solving (B'B + lam*P) X = B'B and taking trace of X
        let trace = compute_smoother_trace(&btb, &penalty, lam, n_basis);

        let denom = 1.0 - trace / n;
        if denom.abs() < 1e-14 {
            continue;
        }
        let gcv = (rss / n) / (denom * denom);

        if gcv < best_gcv {
            best_gcv = gcv;
            best_lambda = lam;
        }
    }

    Ok(best_lambda)
}

/// Compute the trace of the smoother matrix for GCV.
///
/// trace(S) = trace((B'B + lambda*P)^{-1} B'B)
fn compute_smoother_trace(
    btb: &Array2<f64>,
    penalty: &Array2<f64>,
    lambda: f64,
    n_basis: usize,
) -> f64 {
    let mut lhs = btb + &(penalty * lambda);
    // Solve columns of B'B one at a time
    let mut trace = 0.0;
    for j in 0..n_basis {
        let rhs = btb.column(j).to_owned();
        if let Ok(x) = solve_symmetric_positive(&mut lhs, &rhs) {
            trace += x[j];
        }
    }
    trace
}

/// Solve a symmetric positive-definite linear system Ax = b via Cholesky decomposition.
///
/// Uses a simple in-place Cholesky factorization (pure Rust).
fn solve_symmetric_positive(a: &mut Array2<f64>, b: &Array1<f64>) -> StatsResult<Array1<f64>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "Matrix is {}x{}, vector length {}",
            a.nrows(),
            a.ncols(),
            b.len()
        )));
    }

    // Add small diagonal regularization for numerical stability
    for i in 0..n {
        a[[i, i]] += 1e-10;
    }

    // Cholesky: A = L L^T
    let mut l = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[[j, k]] * l[[j, k]];
        }
        let diag = a[[j, j]] - sum;
        if diag <= 0.0 {
            return Err(StatsError::ComputationError(
                "Matrix is not positive definite (Cholesky failed)".to_string(),
            ));
        }
        l[[j, j]] = diag.sqrt();

        for i in (j + 1)..n {
            let mut sum2 = 0.0;
            for k in 0..j {
                sum2 += l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = (a[[i, j]] - sum2) / l[[j, j]];
        }
    }

    // Forward substitution: L z = b
    let mut z = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[[i, k]] * z[k];
        }
        z[i] = s / l[[i, i]];
    }

    // Back substitution: L^T x = z
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in (i + 1)..n {
            s -= l[[k, i]] * x[k];
        }
        x[i] = s / l[[i, i]];
    }

    Ok(x)
}

/// Compute the hat matrix (smoother matrix) H = B (B'B + lambda*P)^{-1} B'
///
/// Used internally for computing effective degrees of freedom.
pub(crate) fn compute_hat_matrix(phi: &Array2<f64>, lambda: f64) -> StatsResult<Array2<f64>> {
    let n_basis = phi.ncols();
    let n = phi.nrows();
    let btb = phi.t().dot(phi);
    let penalty = second_derivative_penalty(n_basis);
    let mut lhs = btb + &(&penalty * lambda);

    // Solve (B'B + lambda*P) X = B' column by column
    let mut inv_bt = Array2::<f64>::zeros((n_basis, n));
    for j in 0..n {
        let rhs = phi.t().column(j).to_owned();
        let col = solve_symmetric_positive(&mut lhs, &rhs)?;
        for i in 0..n_basis {
            inv_bt[[i, j]] = col[i];
        }
    }

    // H = B * (B'B + lambda*P)^{-1} * B'
    Ok(phi.dot(&inv_bt))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bspline_partition_of_unity() {
        let grid: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
        let basis =
            evaluate_bspline_basis(&grid, 10, 3).expect("B-spline evaluation should succeed");

        // B-splines form a partition of unity: sum of basis functions = 1 at every point
        for row in 0..grid.len() {
            let row_sum: f64 = (0..basis.ncols()).map(|c| basis[[row, c]]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "Row {} sum = {}, expected 1.0",
                row,
                row_sum
            );
        }
    }

    #[test]
    fn test_fourier_basis_orthogonality() {
        // On a uniform grid, Fourier basis functions should be approximately orthogonal
        let n = 200;
        let grid: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let basis = evaluate_fourier_basis(&grid, 5).expect("Fourier evaluation should succeed");

        // Check approximate orthogonality: B'B should be approximately diagonal
        let btb = basis.t().dot(&basis);
        let scale = n as f64;
        for i in 0..5 {
            for j in 0..5 {
                if i == j {
                    assert!(
                        btb[[i, j]] > 0.1 * scale,
                        "Diagonal entry [{},{}] = {} too small",
                        i,
                        j,
                        btb[[i, j]]
                    );
                } else {
                    assert!(
                        btb[[i, j]].abs() < 0.1 * scale,
                        "Off-diagonal [{},{}] = {} too large",
                        i,
                        j,
                        btb[[i, j]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_polynomial_basis() {
        let grid = vec![0.0, 0.5, 1.0];
        let basis =
            evaluate_polynomial_basis(&grid, 2).expect("Polynomial evaluation should succeed");

        // At t=0 -> t_norm = -1: [1, -1, 1]
        assert!((basis[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((basis[[0, 1]] - (-1.0)).abs() < 1e-10);
        assert!((basis[[0, 2]] - 1.0).abs() < 1e-10);

        // At t=0.5 -> t_norm = 0: [1, 0, 0]
        assert!((basis[[1, 0]] - 1.0).abs() < 1e-10);
        assert!(basis[[1, 1]].abs() < 1e-10);
        assert!(basis[[1, 2]].abs() < 1e-10);

        // At t=1 -> t_norm = 1: [1, 1, 1]
        assert!((basis[[2, 0]] - 1.0).abs() < 1e-10);
        assert!((basis[[2, 1]] - 1.0).abs() < 1e-10);
        assert!((basis[[2, 2]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_smoothing_reduces_noise() {
        let n = 100;
        let grid: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        // True signal: sin(2*pi*t)
        let true_signal: Vec<f64> = grid
            .iter()
            .map(|&t| (2.0 * std::f64::consts::PI * t).sin())
            .collect();
        // Noisy signal: add some deterministic "noise"
        let noisy: Vec<f64> = true_signal
            .iter()
            .enumerate()
            .map(|(i, &s)| s + 0.3 * ((i as f64 * 7.3).sin()))
            .collect();

        let basis_type = BasisType::BSpline {
            n_basis: 15,
            degree: 3,
        };
        let smoothed =
            smooth_curve(&noisy, &grid, &basis_type, 0.01).expect("Smoothing should succeed");

        // Smoothed curve should be closer to true signal than noisy data
        let noise_mse: f64 = noisy
            .iter()
            .zip(true_signal.iter())
            .map(|(n, t)| (n - t).powi(2))
            .sum::<f64>()
            / n as f64;

        let smooth_mse: f64 = smoothed
            .iter()
            .zip(true_signal.iter())
            .map(|(s, t)| (s - t).powi(2))
            .sum::<f64>()
            / n as f64;

        assert!(
            smooth_mse < noise_mse,
            "Smoothed MSE ({}) should be less than noise MSE ({})",
            smooth_mse,
            noise_mse
        );
    }

    #[test]
    fn test_gcv_selects_reasonable_lambda() {
        let n = 100;
        let grid: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let noisy: Vec<f64> = grid
            .iter()
            .enumerate()
            .map(|(i, &t)| (2.0 * std::f64::consts::PI * t).sin() + 0.2 * ((i as f64 * 13.7).sin()))
            .collect();

        let basis_type = BasisType::BSpline {
            n_basis: 15,
            degree: 3,
        };
        let lambda = gcv_select_lambda(&noisy, &grid, &basis_type).expect("GCV should succeed");

        // Lambda should be positive and not absurdly large or small
        assert!(lambda > 0.0, "Lambda should be positive, got {}", lambda);
        assert!(
            lambda < 1e8,
            "Lambda should not be too large, got {}",
            lambda
        );
    }

    #[test]
    fn test_bspline_basis_nonnegative() {
        let grid: Vec<f64> = (0..=50).map(|i| i as f64 / 50.0).collect();
        let basis =
            evaluate_bspline_basis(&grid, 8, 3).expect("B-spline evaluation should succeed");

        for row in 0..grid.len() {
            for col in 0..basis.ncols() {
                assert!(
                    basis[[row, col]] >= -1e-14,
                    "B-spline values should be non-negative, got {} at [{},{}]",
                    basis[[row, col]],
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn test_bspline_invalid_n_basis() {
        let grid = vec![0.0, 0.5, 1.0];
        let result = evaluate_bspline_basis(&grid, 2, 3);
        assert!(result.is_err(), "n_basis < degree+1 should fail");
    }

    #[test]
    fn test_fourier_basis_constant_term() {
        let grid: Vec<f64> = (0..20).map(|i| i as f64 / 20.0).collect();
        let basis = evaluate_fourier_basis(&grid, 5).expect("Fourier evaluation should succeed");

        // First column should be all 1.0 (constant term)
        for row in 0..grid.len() {
            assert!(
                (basis[[row, 0]] - 1.0).abs() < 1e-10,
                "First Fourier basis should be constant 1.0"
            );
        }
    }

    #[test]
    fn test_smooth_curve_with_fourier_basis() {
        let n = 80;
        let grid: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let signal: Vec<f64> = grid
            .iter()
            .enumerate()
            .map(|(i, &t)| t * t + 0.5 * ((i as f64 * 11.1).sin()))
            .collect();

        let basis_type = BasisType::Fourier { n_basis: 9 };
        let smoothed = smooth_curve(&signal, &grid, &basis_type, 0.1)
            .expect("Smoothing with Fourier should succeed");

        assert_eq!(smoothed.len(), n);
    }
}
