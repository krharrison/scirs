//! Portfolio analytics: return, variance, and efficient frontier computation
//!
//! Implements core mean-variance portfolio theory (Markowitz 1952).
//! The efficient frontier is computed via a parametric sweep over target returns.

use crate::error::{CoreError, CoreResult};
use crate::ndarray::Array2;

// ============================================================
// Portfolio return
// ============================================================

/// Compute the portfolio return as the weighted sum of asset returns.
///
/// `Rp = Σ wᵢ * rᵢ`
///
/// # Arguments
/// * `weights` - Portfolio weights (must sum to approximately 1.0)
/// * `returns` - Per-asset expected (or realised) returns
///
/// # Errors
/// Returns [`CoreError::InvalidArgument`] if slices have different lengths or are empty.
pub fn portfolio_return(weights: &[f64], returns: &[f64]) -> CoreResult<f64> {
    validate_portfolio_inputs(weights, returns)?;
    let rp = weights.iter().zip(returns.iter()).map(|(w, r)| w * r).sum();
    Ok(rp)
}

// ============================================================
// Portfolio variance
// ============================================================

/// Compute portfolio variance using the covariance matrix.
///
/// `σ²p = wᵀ Σ w`
///
/// # Arguments
/// * `weights` - Portfolio weights vector of length N
/// * `cov_matrix` - N×N covariance matrix of asset returns
///
/// # Errors
/// Returns [`CoreError::InvalidArgument`] if `weights` and `cov_matrix` dimensions mismatch,
/// or if `cov_matrix` is not square.
pub fn portfolio_variance(weights: &[f64], cov_matrix: &Array2<f64>) -> CoreResult<f64> {
    let n = weights.len();
    let (rows, cols) = cov_matrix.dim();

    if rows != cols {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            format!("Covariance matrix must be square, got {rows}×{cols}"),
        )));
    }
    if n == 0 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "Weights vector must not be empty",
        )));
    }
    if n != rows {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            format!("Weights length ({n}) does not match covariance matrix dimension ({rows})",),
        )));
    }

    // σ²p = wᵀ Σ w  (computed as Σᵢ Σⱼ wᵢ σᵢⱼ wⱼ)
    let mut variance = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            variance += weights[i] * cov_matrix[[i, j]] * weights[j];
        }
    }

    Ok(variance)
}

// ============================================================
// Efficient frontier
// ============================================================

/// Compute the mean-variance efficient frontier via the critical-line sweep.
///
/// For each target return `μ*` between `min(expected_returns)` and
/// `max(expected_returns)`, the function solves the **minimum-variance portfolio**
/// for that target return using the analytic two-fund separation theorem:
///
/// ```text
/// min  wᵀ Σ w
/// s.t. wᵀ 1 = 1,  wᵀ μ = μ*
/// ```
///
/// The exact closed-form solution is derived from the bordered Hessian system.
/// No short-selling constraint is applied (long-short allowed).
///
/// # Arguments
/// * `expected_returns` - Vector of N asset expected returns
/// * `cov_matrix` - N×N covariance matrix (must be positive-definite)
/// * `n_points` - Number of frontier points to compute
///
/// # Returns
/// `Vec<(risk, return)>` where `risk = σp = sqrt(variance)` and `return = μ*`.
/// Sorted by increasing risk.
///
/// # Errors
/// * [`CoreError::InvalidArgument`] for dimension mismatches or fewer than 2 assets.
/// * [`CoreError::ComputationError`] if the covariance matrix is singular.
pub fn efficient_frontier(
    expected_returns: &[f64],
    cov_matrix: &Array2<f64>,
    n_points: usize,
) -> CoreResult<Vec<(f64, f64)>> {
    let n = expected_returns.len();
    let (rows, cols) = cov_matrix.dim();

    if rows != cols {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            format!("Covariance matrix must be square, got {rows}×{cols}"),
        )));
    }
    if n < 2 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "Efficient frontier requires at least 2 assets",
        )));
    }
    if n != rows {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            format!("expected_returns length ({n}) must match cov_matrix dimension ({rows})"),
        )));
    }
    if n_points < 2 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "n_points must be at least 2",
        )));
    }

    // Solve Σ⁻¹ using Cholesky / Gaussian elimination
    let cov_inv = invert_spd(cov_matrix, n)?;

    // Scalars from the analytic frontier formulas
    // A = μᵀ Σ⁻¹ 1,  B = μᵀ Σ⁻¹ μ,  C = 1ᵀ Σ⁻¹ 1,  D = BC - A²
    let (big_a, big_b, big_c) = compute_frontier_scalars(expected_returns, &cov_inv, n);
    let big_d = big_b * big_c - big_a * big_a;

    if big_d.abs() < 1e-14 {
        return Err(CoreError::ComputationError(
            crate::error::ErrorContext::new(
                "Covariance matrix is (near) singular or all assets have the same expected return",
            ),
        ));
    }

    // Range of target returns
    let mu_min = expected_returns
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let mu_max = expected_returns
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // Extend range slightly for a richer frontier shape
    let mu_range = mu_max - mu_min;
    let mu_lo = mu_min - 0.0 * mu_range;
    let mu_hi = mu_max + 0.0 * mu_range;
    let step = (mu_hi - mu_lo) / (n_points - 1) as f64;

    let mut frontier: Vec<(f64, f64)> = (0..n_points)
        .map(|i| {
            let target_mu = mu_lo + i as f64 * step;
            // Minimum variance for target return:
            // σ²p = (C*μ*² - 2*A*μ* + B) / D
            let variance =
                (big_c * target_mu * target_mu - 2.0 * big_a * target_mu + big_b) / big_d;
            let risk = variance.max(0.0).sqrt();
            (risk, target_mu)
        })
        .collect();

    frontier.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    Ok(frontier)
}

// ============================================================
// Minimum variance portfolio
// ============================================================

/// Compute the global minimum variance portfolio weights.
///
/// Solves:
/// ```text
/// min wᵀ Σ w   s.t. wᵀ 1 = 1
/// ```
///
/// The analytic solution is `w* = (Σ⁻¹ 1) / (1ᵀ Σ⁻¹ 1)`.
///
/// # Arguments
/// * `cov_matrix` - N×N covariance matrix
///
/// # Returns
/// Weight vector `w*` summing to 1.
///
/// # Errors
/// Returns [`CoreError::ComputationError`] if the covariance matrix is singular.
pub fn min_variance_weights(cov_matrix: &Array2<f64>) -> CoreResult<Vec<f64>> {
    let (rows, cols) = cov_matrix.dim();
    if rows != cols || rows == 0 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "Covariance matrix must be square and non-empty",
        )));
    }

    let n = rows;
    let cov_inv = invert_spd(cov_matrix, n)?;

    // w* = Σ⁻¹ * 1  (column sum of inverse)
    let mut w: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| cov_inv[[i, j]]).sum())
        .collect();
    let sum: f64 = w.iter().sum();

    if sum.abs() < 1e-14 {
        return Err(CoreError::ComputationError(
            crate::error::ErrorContext::new("Inverse of covariance matrix sums to near zero"),
        ));
    }

    w.iter_mut().for_each(|wi| *wi /= sum);
    Ok(w)
}

// ============================================================
// Internal helpers
// ============================================================

fn validate_portfolio_inputs(weights: &[f64], returns: &[f64]) -> CoreResult<()> {
    if weights.is_empty() {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "Weights vector must not be empty",
        )));
    }
    if weights.len() != returns.len() {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            format!(
                "Weights length ({}) must equal returns length ({})",
                weights.len(),
                returns.len()
            ),
        )));
    }
    Ok(())
}

/// Invert an N×N positive-semidefinite matrix via Gaussian elimination with partial pivoting.
fn invert_spd(mat: &Array2<f64>, n: usize) -> CoreResult<Array2<f64>> {
    // Build augmented matrix [A | I]
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = (0..n).map(|j| mat[[i, j]]).collect();
            let mut id_part = vec![0.0_f64; n];
            id_part[i] = 1.0;
            row.extend(id_part);
            row
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return Err(CoreError::ComputationError(
                crate::error::ErrorContext::new("Covariance matrix is singular or near-singular"),
            ));
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..(2 * n) {
                let sub = factor * aug[col][j];
                aug[row][j] -= sub;
            }
        }
    }

    // Extract right half as inverse
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[i][n + j];
        }
    }
    Ok(inv)
}

/// Compute scalars A, B, C used in the analytic efficient frontier formula.
fn compute_frontier_scalars(mu: &[f64], cov_inv: &Array2<f64>, n: usize) -> (f64, f64, f64) {
    let mut big_a = 0.0_f64; // μᵀ Σ⁻¹ 1
    let mut big_b = 0.0_f64; // μᵀ Σ⁻¹ μ
    let mut big_c = 0.0_f64; // 1ᵀ Σ⁻¹ 1

    for i in 0..n {
        for j in 0..n {
            let sigma_inv_ij = cov_inv[[i, j]];
            big_a += mu[i] * sigma_inv_ij; // μᵀ Σ⁻¹ 1 (j-th column summed)
            big_b += mu[i] * sigma_inv_ij * mu[j]; // μᵀ Σ⁻¹ μ
            big_c += sigma_inv_ij; // 1ᵀ Σ⁻¹ 1
        }
    }
    (big_a, big_b, big_c)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray::array;

    // --- portfolio_return ---
    #[test]
    fn test_portfolio_return_equal_weight() {
        let weights = [0.5, 0.5];
        let returns = [0.10, 0.20];
        let rp = portfolio_return(&weights, &returns).expect("should succeed");
        assert!((rp - 0.15).abs() < 1e-10, "Equal weight return: {rp}");
    }

    #[test]
    fn test_portfolio_return_concentrated() {
        let weights = [1.0, 0.0, 0.0];
        let returns = [0.08, 0.12, 0.06];
        let rp = portfolio_return(&weights, &returns).expect("should succeed");
        assert!((rp - 0.08).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_return_length_mismatch() {
        assert!(portfolio_return(&[0.5, 0.5], &[0.1]).is_err());
    }

    #[test]
    fn test_portfolio_return_empty() {
        assert!(portfolio_return(&[], &[]).is_err());
    }

    // --- portfolio_variance ---
    #[test]
    fn test_portfolio_variance_two_assets_uncorrelated() {
        // Two uncorrelated assets: σ²p = w1²σ1² + w2²σ2²
        let weights = [0.6, 0.4];
        let cov = array![[0.04, 0.0], [0.0, 0.09]]; // σ1=0.2, σ2=0.3
        let var = portfolio_variance(&weights, &cov).expect("should succeed");
        let expected = 0.6 * 0.6 * 0.04 + 0.4 * 0.4 * 0.09;
        assert!(
            (var - expected).abs() < 1e-10,
            "Uncorrelated variance: {var:.8} vs {expected:.8}"
        );
    }

    #[test]
    fn test_portfolio_variance_single_asset() {
        let weights = [1.0];
        let cov = array![[0.04]];
        let var = portfolio_variance(&weights, &cov).expect("should succeed");
        assert!((var - 0.04).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_variance_non_negative() {
        let weights = [0.3, 0.4, 0.3];
        let cov = array![
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.015],
            [0.005, 0.015, 0.0625]
        ];
        let var = portfolio_variance(&weights, &cov).expect("should succeed");
        assert!(var >= 0.0, "Variance must be non-negative: {var}");
    }

    #[test]
    fn test_portfolio_variance_shape_mismatch() {
        let weights = [0.5, 0.5];
        let cov = array![[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.01]];
        assert!(portfolio_variance(&weights, &cov).is_err());
    }

    #[test]
    fn test_portfolio_variance_non_square() {
        let weights = [0.5, 0.5];
        let cov = Array2::<f64>::zeros((2, 3));
        assert!(portfolio_variance(&weights, &cov).is_err());
    }

    // --- efficient frontier ---
    #[test]
    fn test_efficient_frontier_two_assets_shape() {
        let mu = [0.05, 0.10];
        let cov = array![[0.04, 0.01], [0.01, 0.09]];
        let frontier = efficient_frontier(&mu, &cov, 10).expect("should succeed");
        assert_eq!(frontier.len(), 10);
    }

    #[test]
    fn test_efficient_frontier_risks_nonnegative() {
        let mu = [0.06, 0.10, 0.14];
        let cov = array![
            [0.04, 0.006, 0.004],
            [0.006, 0.09, 0.012],
            [0.004, 0.012, 0.16]
        ];
        let frontier = efficient_frontier(&mu, &cov, 20).expect("should succeed");
        for (risk, _ret) in &frontier {
            assert!(*risk >= 0.0, "Risk must be non-negative: {risk}");
        }
    }

    #[test]
    fn test_efficient_frontier_sorted_by_risk() {
        let mu = [0.05, 0.10];
        let cov = array![[0.04, 0.01], [0.01, 0.09]];
        let frontier = efficient_frontier(&mu, &cov, 15).expect("should succeed");
        for i in 1..frontier.len() {
            assert!(
                frontier[i].0 >= frontier[i - 1].0 - 1e-10,
                "Frontier should be sorted by risk"
            );
        }
    }

    #[test]
    fn test_efficient_frontier_too_few_assets() {
        let mu = [0.05];
        let cov = array![[0.04]];
        assert!(efficient_frontier(&mu, &cov, 10).is_err());
    }

    #[test]
    fn test_efficient_frontier_dimension_mismatch() {
        let mu = [0.05, 0.10, 0.15];
        let cov = array![[0.04, 0.01], [0.01, 0.09]];
        assert!(efficient_frontier(&mu, &cov, 10).is_err());
    }

    // --- min_variance_weights ---
    #[test]
    fn test_min_variance_weights_sum_to_one() {
        let cov = array![[0.04, 0.01], [0.01, 0.09]];
        let w = min_variance_weights(&cov).expect("should succeed");
        let sum: f64 = w.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Weights must sum to 1: {sum:.10}"
        );
    }

    #[test]
    fn test_min_variance_single_asset() {
        let cov = array![[0.04]];
        let w = min_variance_weights(&cov).expect("should succeed");
        assert!((w[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_variance_equal_variance_uncorrelated() {
        // Two uncorrelated assets with equal variance -> equal weights
        let var = 0.04;
        let cov = array![[var, 0.0], [0.0, var]];
        let w = min_variance_weights(&cov).expect("should succeed");
        assert!(
            (w[0] - 0.5).abs() < 1e-8,
            "Equal weight expected: w[0]={:.8}",
            w[0]
        );
        assert!(
            (w[1] - 0.5).abs() < 1e-8,
            "Equal weight expected: w[1]={:.8}",
            w[1]
        );
    }

    #[test]
    fn test_min_variance_lower_weight_on_riskier_asset() {
        // Asset 0 has lower variance -> should receive higher weight
        let cov = array![[0.01, 0.0], [0.0, 0.09]];
        let w = min_variance_weights(&cov).expect("should succeed");
        assert!(
            w[0] > w[1],
            "Lower-variance asset should have more weight: w0={:.4} w1={:.4}",
            w[0],
            w[1]
        );
    }
}
