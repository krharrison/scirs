//! Hyperparameter exploration and integration for INLA
//!
//! This module handles:
//! - Evaluating the log-posterior of hyperparameters via Laplace approximation
//! - Grid-based exploration of the hyperparameter space
//! - Central Composite Design (CCD) integration points
//! - Numerical integration on the log scale

use scirs2_core::ndarray::{Array1, Array2};

use super::laplace;
use super::types::{
    HyperparameterPosterior, INLAConfig, IntegrationStrategy, LatentGaussianModel, LikelihoodFamily,
};
use crate::error::StatsError;

/// A single hyperparameter configuration with its log-posterior value
#[derive(Debug, Clone)]
pub struct HyperparameterPoint {
    /// Hyperparameter values (e.g., log-precision for Gaussian, etc.)
    pub theta: Vec<f64>,
    /// Log-posterior density p̃(θ|y) evaluated via Laplace approximation
    pub log_posterior: f64,
    /// The mode result at this hyperparameter configuration
    pub mode: Array1<f64>,
    /// Diagonal of the inverse negative Hessian (marginal variances at this θ)
    pub marginal_variances: Array1<f64>,
}

/// Evaluate the log-posterior of hyperparameters using Laplace approximation
///
/// log p̃(θ|y) ∝ log p(y|θ) + log p(θ)
///
/// where log p(y|θ) is approximated using the Laplace method.
///
/// # Arguments
/// * `theta` - Hyperparameter value (log-precision scale)
/// * `model` - The latent Gaussian model
/// * `config` - INLA configuration
///
/// # Returns
/// A `HyperparameterPoint` containing the log-posterior and associated quantities
pub fn evaluate_hyperparameter(
    theta: f64,
    model: &LatentGaussianModel,
    config: &INLAConfig,
) -> Result<HyperparameterPoint, StatsError> {
    // Scale the precision matrix by exp(theta) (theta is log-precision)
    let scale = theta.exp();
    let scaled_precision = &model.precision_matrix * scale;

    // Find the posterior mode at this hyperparameter value
    let mode_result = laplace::find_mode(
        &scaled_precision,
        &model.y,
        &model.design_matrix,
        model.likelihood,
        model.n_trials.as_ref(),
        model.observation_precision,
        config.max_newton_iter,
        config.newton_tol,
        config.newton_damping,
    )?;

    // Compute Laplace approximation to log p(y|θ)
    let log_marginal = laplace::laplace_log_marginal_likelihood(&mode_result, &scaled_precision)?;

    // Log prior on θ (flat/improper prior by default)
    let log_prior_theta = log_hyperprior(theta, config);

    // Compute marginal variances (diagonal of H^{-1})
    let marginal_vars = laplace::inverse_diagonal(&mode_result.neg_hessian)?;

    Ok(HyperparameterPoint {
        theta: vec![theta],
        log_posterior: log_marginal + log_prior_theta,
        mode: mode_result.mode,
        marginal_variances: marginal_vars,
    })
}

/// Log-prior for hyperparameters
///
/// Uses a flat prior by default, or a Gaussian prior if range is specified.
fn log_hyperprior(theta: f64, config: &INLAConfig) -> f64 {
    match config.hyperparameter_range {
        Some((lo, hi)) => {
            // Penalized complexity prior: log-Gaussian centered at midpoint
            let mid = (lo + hi) / 2.0;
            let scale = (hi - lo) / 4.0; // 95% within range
            if scale <= 0.0 {
                return 0.0;
            }
            -0.5 * ((theta - mid) / scale).powi(2)
        }
        None => 0.0, // flat (improper) prior
    }
}

/// Explore the hyperparameter space on a grid
///
/// Creates a 1D grid of hyperparameter values and evaluates the
/// Laplace-approximated log-posterior at each point.
///
/// # Arguments
/// * `model` - The latent Gaussian model
/// * `config` - INLA configuration
///
/// # Returns
/// Vector of `HyperparameterPoint` sorted by log-posterior (descending)
pub fn explore_hyperparameter_grid(
    model: &LatentGaussianModel,
    config: &INLAConfig,
) -> Result<Vec<HyperparameterPoint>, StatsError> {
    let n_grid = config.n_hyperparameter_grid;
    if n_grid == 0 {
        return Err(StatsError::InvalidArgument(
            "Number of hyperparameter grid points must be positive".to_string(),
        ));
    }

    // Determine grid range
    let (lo, hi) = config.hyperparameter_range.unwrap_or((-3.0, 3.0));

    let grid_points = create_grid(lo, hi, n_grid);

    let mut results = Vec::with_capacity(n_grid);
    for &theta in &grid_points {
        match evaluate_hyperparameter(theta, model, config) {
            Ok(point) => results.push(point),
            Err(_) => {
                // Skip points where mode finding fails (e.g., numerical issues)
                continue;
            }
        }
    }

    if results.is_empty() {
        return Err(StatsError::ConvergenceError(
            "INLA failed to evaluate any hyperparameter grid point".to_string(),
        ));
    }

    // Sort by log-posterior (descending)
    results.sort_by(|a, b| {
        b.log_posterior
            .partial_cmp(&a.log_posterior)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(results)
}

/// Create a uniform grid of points in [lo, hi]
fn create_grid(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![(lo + hi) / 2.0];
    }
    let step = (hi - lo) / (n - 1) as f64;
    (0..n).map(|i| lo + i as f64 * step).collect()
}

/// Generate Central Composite Design (CCD) integration points
///
/// CCD is more efficient than a full grid for multivariate hyperparameter
/// integration. For d hyperparameters, it uses:
/// - 1 center point
/// - 2*d axial points at distance ±α from center
/// - 2^d factorial points (for d ≤ 4, else a fraction)
///
/// Total points: 1 + 2*d + min(2^d, 2*d) for large d
///
/// # Arguments
/// * `n_hyperparams` - Number of hyperparameters
///
/// # Returns
/// Vector of point coordinates (each is a `Vec<f64>` of length n_hyperparams)
/// The points are on a standardized scale (centered at 0, scaled by 1).
pub fn ccd_integration_points(n_hyperparams: usize) -> Result<Vec<Vec<f64>>, StatsError> {
    if n_hyperparams == 0 {
        return Err(StatsError::InvalidArgument(
            "Number of hyperparameters must be positive".to_string(),
        ));
    }

    let mut points = Vec::new();

    // Center point
    points.push(vec![0.0; n_hyperparams]);

    // Axial distance: alpha = sqrt(n_hyperparams) for rotatability
    let alpha = (n_hyperparams as f64).sqrt();

    // Axial points: ±alpha along each axis
    for d in 0..n_hyperparams {
        let mut point_pos = vec![0.0; n_hyperparams];
        point_pos[d] = alpha;
        points.push(point_pos);

        let mut point_neg = vec![0.0; n_hyperparams];
        point_neg[d] = -alpha;
        points.push(point_neg);
    }

    // Factorial points: all combinations of ±1
    // For large d, use fractional factorial
    let max_factorial = if n_hyperparams <= 6 {
        1usize << n_hyperparams // 2^d
    } else {
        // Fractional factorial for high dimensions
        2 * n_hyperparams
    };

    let n_factorial = (1usize << n_hyperparams).min(max_factorial);
    for i in 0..n_factorial {
        let mut point = vec![0.0; n_hyperparams];
        for d in 0..n_hyperparams {
            point[d] = if (i >> d) & 1 == 0 { -1.0 } else { 1.0 };
        }
        points.push(point);
    }

    Ok(points)
}

/// Perform numerical integration on the log scale
///
/// Given log-densities at grid points, compute the normalized weights
/// and the log of the normalizing constant.
///
/// Uses the log-sum-exp trick for numerical stability.
///
/// # Arguments
/// * `log_densities` - Log-density values at grid points
/// * `grid_spacing` - Spacing between grid points (for trapezoidal rule)
///
/// # Returns
/// Tuple of (normalized_weights, log_normalizing_constant)
pub fn grid_integration(
    log_densities: &[f64],
    grid_spacing: f64,
) -> Result<(Vec<f64>, f64), StatsError> {
    if log_densities.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Log densities array is empty".to_string(),
        ));
    }

    // Find maximum for log-sum-exp trick
    let max_log = log_densities
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    if max_log.is_infinite() && max_log < 0.0 {
        return Err(StatsError::ComputationError(
            "All log densities are -infinity".to_string(),
        ));
    }

    // Compute weights using trapezoidal rule on log scale
    let n = log_densities.len();
    let mut weights = Vec::with_capacity(n);
    for i in 0..n {
        let trap_factor = if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
        weights.push((log_densities[i] - max_log).exp() * trap_factor * grid_spacing);
    }

    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        return Err(StatsError::ComputationError(
            "Total integration weight is non-positive".to_string(),
        ));
    }

    let log_normalizing = max_log + total_weight.ln();

    // Normalize weights
    let normalized: Vec<f64> = weights.iter().map(|w| w / total_weight).collect();

    Ok((normalized, log_normalizing))
}

/// Compute posterior summary statistics for a hyperparameter from grid evaluation
///
/// # Arguments
/// * `grid_points` - Grid point values
/// * `log_densities` - Log-density at each grid point
/// * `grid_spacing` - Spacing between grid points
///
/// # Returns
/// `HyperparameterPosterior` with mean, variance, and density information
pub fn summarize_hyperparameter_posterior(
    grid_points: &[f64],
    log_densities: &[f64],
    grid_spacing: f64,
) -> Result<HyperparameterPosterior, StatsError> {
    if grid_points.len() != log_densities.len() {
        return Err(StatsError::DimensionMismatch(
            "Grid points and log densities must have the same length".to_string(),
        ));
    }

    let (weights, _) = grid_integration(log_densities, grid_spacing)?;

    // Compute mean: E[θ] = Σ w_i * θ_i
    let mean: f64 = weights
        .iter()
        .zip(grid_points.iter())
        .map(|(w, t)| w * t)
        .sum();

    // Compute variance: Var[θ] = Σ w_i * (θ_i - mean)^2
    let variance: f64 = weights
        .iter()
        .zip(grid_points.iter())
        .map(|(w, t)| w * (t - mean).powi(2))
        .sum();

    Ok(HyperparameterPosterior {
        grid_points: grid_points.to_vec(),
        log_densities: log_densities.to_vec(),
        mean,
        variance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_create_grid() {
        let grid = create_grid(-1.0, 1.0, 5);
        assert_eq!(grid.len(), 5);
        assert!((grid[0] - (-1.0)).abs() < 1e-10);
        assert!((grid[4] - 1.0).abs() < 1e-10);
        assert!((grid[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_create_grid_single() {
        let grid = create_grid(-1.0, 1.0, 1);
        assert_eq!(grid.len(), 1);
        assert!((grid[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ccd_1d() {
        let points = ccd_integration_points(1).expect("CCD should succeed for 1D");
        // 1D: 1 center + 2 axial + 2 factorial = 5
        assert_eq!(points.len(), 5);
        // Center point
        assert!((points[0][0]).abs() < 1e-10);
        // Axial points at ±1
        assert!((points[1][0] - 1.0).abs() < 1e-10);
        assert!((points[2][0] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_ccd_2d() {
        let points = ccd_integration_points(2).expect("CCD should succeed for 2D");
        // 2D: 1 center + 4 axial + 4 factorial = 9
        assert_eq!(points.len(), 9);
        // Center
        assert!((points[0][0]).abs() < 1e-10);
        assert!((points[0][1]).abs() < 1e-10);
    }

    #[test]
    fn test_ccd_3d() {
        let points = ccd_integration_points(3).expect("CCD should succeed for 3D");
        // 3D: 1 center + 6 axial + 8 factorial = 15
        assert_eq!(points.len(), 15);
    }

    #[test]
    fn test_ccd_zero() {
        let result = ccd_integration_points(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_grid_integration_uniform() {
        // Uniform log-densities should give equal weights
        let log_densities = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let (weights, _) =
            grid_integration(&log_densities, 1.0).expect("Integration should succeed");
        // Middle points get weight 1, endpoints get weight 0.5, total = 4
        // So normalized: 0.125, 0.25, 0.25, 0.25, 0.125
        assert!((weights[0] - 0.125).abs() < 1e-10);
        assert!((weights[2] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_grid_integration_peaked() {
        // Strongly peaked distribution
        let log_densities = vec![-100.0, -10.0, 0.0, -10.0, -100.0];
        let (weights, _) =
            grid_integration(&log_densities, 1.0).expect("Integration should succeed");
        // Most weight should be on the center point
        assert!(weights[2] > 0.9);
    }

    #[test]
    fn test_grid_integration_empty() {
        let result = grid_integration(&[], 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_summarize_posterior() {
        // Symmetric around 0 should give mean ≈ 0
        let grid_points = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let log_densities = vec![-2.0, -0.5, 0.0, -0.5, -2.0];
        let result = summarize_hyperparameter_posterior(&grid_points, &log_densities, 1.0)
            .expect("Summary should succeed");
        assert!(
            result.mean.abs() < 0.1,
            "Mean should be near 0, got {}",
            result.mean
        );
        assert!(result.variance > 0.0, "Variance should be positive");
    }

    #[test]
    fn test_explore_grid_gaussian() {
        let n = 3;
        let y = array![1.0, 2.0, 3.0];
        let design = Array2::eye(n);
        let precision = Array2::eye(n);

        let model = LatentGaussianModel::new(y, design, precision, LikelihoodFamily::Gaussian)
            .with_observation_precision(1.0);

        let config = INLAConfig {
            n_hyperparameter_grid: 5,
            hyperparameter_range: Some((-1.0, 1.0)),
            max_newton_iter: 50,
            ..INLAConfig::default()
        };

        let results =
            explore_hyperparameter_grid(&model, &config).expect("Grid exploration should succeed");

        assert!(!results.is_empty(), "Should have some valid grid points");
        // Results should be sorted by log-posterior (descending)
        for i in 1..results.len() {
            assert!(
                results[i - 1].log_posterior >= results[i].log_posterior,
                "Results should be sorted descending"
            );
        }
    }

    #[test]
    fn test_dimension_mismatch_summary() {
        let grid = vec![1.0, 2.0];
        let densities = vec![0.0, 0.0, 0.0];
        let result = summarize_hyperparameter_posterior(&grid, &densities, 1.0);
        assert!(result.is_err());
    }
}
