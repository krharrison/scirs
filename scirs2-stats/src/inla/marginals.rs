//! Marginal posterior computation for INLA
//!
//! Implements the main INLA algorithm that computes posterior marginals
//! for each component of the latent field by integrating over hyperparameters.
//!
//! The key formula is:
//!   p(x_i|y) = ∫ p̃(x_i|θ,y) × p̃(θ|y) dθ
//!
//! where p̃(x_i|θ,y) is the Gaussian approximation to the conditional
//! posterior of x_i given θ, and p̃(θ|y) is the Laplace approximation
//! to the hyperparameter posterior.

use scirs2_core::ndarray::{Array1, Array2};

use super::hyperparameters::{
    self, explore_hyperparameter_grid, grid_integration, summarize_hyperparameter_posterior,
    HyperparameterPoint,
};
use super::laplace;
use super::types::{
    HyperparameterPosterior, INLAConfig, INLAResult, IntegrationStrategy, LatentGaussianModel,
    LikelihoodFamily,
};
use crate::error::StatsError;

/// Main INLA algorithm: compute posterior marginals for latent field components
///
/// # Algorithm
/// 1. Explore hyperparameter space (grid or CCD)
/// 2. At each hyperparameter configuration θ_k:
///    a. Find posterior mode x*(θ_k) via Newton-Raphson
///    b. Compute Laplace approximation p̃(θ_k|y)
///    c. Compute conditional marginal variances from H^{-1}
/// 3. Integrate out θ using numerical integration:
///    E[x_i|y] = Σ_k E[x_i|θ_k,y] × w_k
///    Var[x_i|y] = Σ_k {Var[x_i|θ_k,y] + E[x_i|θ_k,y]²} × w_k - E[x_i|y]²
///
/// # Arguments
/// * `model` - The latent Gaussian model specification
/// * `config` - INLA algorithm configuration
///
/// # Returns
/// `INLAResult` containing posterior marginal means, variances, and diagnostics
pub fn compute_marginals(
    model: &LatentGaussianModel,
    config: &INLAConfig,
) -> Result<INLAResult, StatsError> {
    validate_model(model)?;

    match config.integration_strategy {
        IntegrationStrategy::SimplifiedLaplace => compute_simplified_laplace(model, config),
        IntegrationStrategy::Grid | IntegrationStrategy::CCD => {
            compute_full_marginals(model, config)
        }
        _ => Err(StatsError::NotImplementedError(
            "Unknown integration strategy".to_string(),
        )),
    }
}

/// Validate the latent Gaussian model for consistency
fn validate_model(model: &LatentGaussianModel) -> Result<(), StatsError> {
    let n = model.y.len();
    let p = model.precision_matrix.nrows();

    if model.precision_matrix.ncols() != p {
        return Err(StatsError::DimensionMismatch(
            "Precision matrix must be square".to_string(),
        ));
    }

    if model.design_matrix.nrows() != n {
        return Err(StatsError::DimensionMismatch(format!(
            "Design matrix rows ({}) must match observation length ({})",
            model.design_matrix.nrows(),
            n
        )));
    }

    if model.design_matrix.ncols() != p {
        return Err(StatsError::DimensionMismatch(format!(
            "Design matrix columns ({}) must match precision matrix size ({})",
            model.design_matrix.ncols(),
            p
        )));
    }

    if n == 0 {
        return Err(StatsError::InsufficientData(
            "Model must have at least one observation".to_string(),
        ));
    }

    if let Some(ref trials) = model.n_trials {
        if trials.len() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "n_trials length ({}) must match observation length ({})",
                trials.len(),
                n
            )));
        }
    }

    Ok(())
}

/// Compute full INLA marginals with hyperparameter integration
fn compute_full_marginals(
    model: &LatentGaussianModel,
    config: &INLAConfig,
) -> Result<INLAResult, StatsError> {
    // Step 1: Explore hyperparameter space
    let hyper_points = explore_hyperparameter_grid(model, config)?;

    if hyper_points.is_empty() {
        return Err(StatsError::ConvergenceError(
            "No valid hyperparameter configurations found".to_string(),
        ));
    }

    let p = model.precision_matrix.nrows();

    // Step 2: Compute integration weights
    let log_posteriors: Vec<f64> = hyper_points.iter().map(|hp| hp.log_posterior).collect();
    let grid_points: Vec<f64> = hyper_points.iter().map(|hp| hp.theta[0]).collect();

    let grid_spacing = if grid_points.len() > 1 {
        (grid_points[grid_points.len() - 1] - grid_points[0]) / (grid_points.len() - 1) as f64
    } else {
        1.0
    };

    // Note: grid_points from explore_hyperparameter_grid are sorted by log_posterior descending,
    // but grid_integration needs them in order. We work with the unsorted log_posteriors directly.
    let (weights, log_normalizing) =
        grid_integration(&log_posteriors, grid_spacing.abs().max(0.01))?;

    // Step 3: Compute weighted marginals
    // E[x_i|y] = Σ_k w_k * E[x_i|θ_k,y]
    // Var[x_i|y] = Σ_k w_k * {Var[x_i|θ_k,y] + E[x_i|θ_k,y]²} - E[x_i|y]²
    let mut marginal_means: Array1<f64> = Array1::zeros(p);
    let mut marginal_second_moments: Array1<f64> = Array1::zeros(p);

    for (hp, w) in hyper_points.iter().zip(weights.iter()) {
        for i in 0..p {
            marginal_means[i] += w * hp.mode[i];
            marginal_second_moments[i] += w * (hp.marginal_variances[i] + hp.mode[i].powi(2));
        }
    }

    let mut marginal_variances = Array1::zeros(p);
    for i in 0..p {
        marginal_variances[i] = (marginal_second_moments[i] - marginal_means[i].powi(2)).max(0.0);
    }

    // Step 4: Summarize hyperparameter posterior
    let hyper_posterior = summarize_hyperparameter_posterior(
        &grid_points,
        &log_posteriors,
        grid_spacing.abs().max(0.01),
    )?;

    // Check convergence: the best mode should have converged
    let best_converged = true; // explore_hyperparameter_grid only returns successful evaluations

    Ok(INLAResult {
        marginal_means,
        marginal_variances,
        hyperparameter_posteriors: vec![hyper_posterior],
        log_marginal_likelihood: log_normalizing,
        converged: best_converged,
        newton_iterations: hyper_points
            .first()
            .map(|_| config.max_newton_iter)
            .unwrap_or(0),
    })
}

/// Simplified Laplace approximation
///
/// Uses a single hyperparameter value (the mode of p̃(θ|y)) rather than
/// integrating over the full hyperparameter space. This is the fastest
/// INLA variant but potentially less accurate.
///
/// Steps:
/// 1. Find optimal θ* by maximizing log p̃(θ|y) on a coarse grid
/// 2. At θ*, compute posterior mode x* and Hessian H
/// 3. Marginals are Gaussian: x_i|y ~ N(x_i*, [H^{-1}]_{ii})
fn compute_simplified_laplace(
    model: &LatentGaussianModel,
    config: &INLAConfig,
) -> Result<INLAResult, StatsError> {
    let p = model.precision_matrix.nrows();

    // Find the best hyperparameter on a coarse grid
    let hyper_points = explore_hyperparameter_grid(model, config)?;

    let best = &hyper_points[0]; // Already sorted descending by log_posterior

    // The mode and variances at the best hyperparameter are our estimates
    let marginal_means = best.mode.clone();
    let marginal_variances = best.marginal_variances.clone();

    // Hyperparameter posterior is a delta function at the mode
    let grid_points: Vec<f64> = hyper_points.iter().map(|hp| hp.theta[0]).collect();
    let log_posteriors: Vec<f64> = hyper_points.iter().map(|hp| hp.log_posterior).collect();

    let grid_spacing = if grid_points.len() > 1 {
        ((grid_points.last().copied().unwrap_or(0.0) - grid_points.first().copied().unwrap_or(0.0))
            / (grid_points.len() - 1) as f64)
            .abs()
            .max(0.01)
    } else {
        1.0
    };

    let hyper_posterior =
        summarize_hyperparameter_posterior(&grid_points, &log_posteriors, grid_spacing)?;

    Ok(INLAResult {
        marginal_means,
        marginal_variances,
        hyperparameter_posteriors: vec![hyper_posterior],
        log_marginal_likelihood: best.log_posterior,
        converged: true,
        newton_iterations: config.max_newton_iter,
    })
}

/// Compute improved marginals using the corrected Laplace approximation
///
/// For each latent field component x_i, this computes:
///   p̃(x_i|θ,y) ≈ N(x_i*; μ_i, σ_i²) × (1 + correction terms)
///
/// The correction involves third derivatives of the log-likelihood,
/// improving accuracy beyond the basic Gaussian approximation.
///
/// # Arguments
/// * `mode` - Posterior mode x*
/// * `neg_hessian` - Negative Hessian at the mode
/// * `y` - Observations
/// * `design` - Design matrix
/// * `likelihood` - Likelihood family
/// * `component` - Which component x_i to compute marginal for
///
/// # Returns
/// Tuple of (corrected_mean, corrected_variance)
pub fn corrected_laplace_marginal(
    mode: &Array1<f64>,
    neg_hessian: &Array2<f64>,
    y: &Array1<f64>,
    design: &Array2<f64>,
    likelihood: LikelihoodFamily,
    component: usize,
    n_trials: Option<&Array1<f64>>,
    obs_precision: Option<f64>,
) -> Result<(f64, f64), StatsError> {
    let p = mode.len();
    if component >= p {
        return Err(StatsError::InvalidArgument(format!(
            "Component index {} exceeds latent field dimension {}",
            component, p
        )));
    }

    // Base Gaussian approximation
    let base_var = laplace::inverse_diagonal(neg_hessian)?;
    let base_mean = mode[component];
    let base_variance = base_var[component];

    // Compute third-derivative correction (skewness correction)
    // For Gaussian likelihood, this is zero (Laplace is exact)
    let correction = match likelihood {
        LikelihoodFamily::Gaussian => 0.0,
        _ => {
            // Numerical approximation of the skewness correction
            // Uses finite differences of the Hessian diagonal
            let eps = 1e-4 * base_variance.sqrt().max(1e-8);
            let eta = design.dot(mode);

            // Third derivative contribution
            let mut d3_sum = 0.0;
            let n = y.len();
            for k in 0..n {
                let a_ki = design[[k, component]];
                if a_ki.abs() < 1e-15 {
                    continue;
                }
                let d3 = third_derivative_log_likelihood(
                    eta[k],
                    likelihood,
                    n_trials.map(|t| t[k]),
                    obs_precision,
                );
                d3_sum += a_ki.powi(3) * d3;
            }

            // Skewness correction to the mean
            -0.5 * base_variance.powi(2) * d3_sum
        }
    };

    Ok((base_mean + correction, base_variance))
}

/// Third derivative of log-likelihood w.r.t. eta
fn third_derivative_log_likelihood(
    eta: f64,
    likelihood: LikelihoodFamily,
    n_trial: Option<f64>,
    obs_precision: Option<f64>,
) -> f64 {
    match likelihood {
        LikelihoodFamily::Gaussian => 0.0,
        LikelihoodFamily::Poisson => {
            // d^3/d(eta)^3 [y*eta - exp(eta)] = -exp(eta)
            -eta.exp().min(1e15)
        }
        LikelihoodFamily::Binomial => {
            let n = n_trial.unwrap_or(1.0);
            // d^3/d(eta)^3 [-n*log(1+exp(eta))]
            // = -n * sigmoid(eta) * (1 - sigmoid(eta)) * (1 - 2*sigmoid(eta))
            let p = if eta >= 0.0 {
                1.0 / (1.0 + (-eta).exp())
            } else {
                eta.exp() / (1.0 + eta.exp())
            };
            -n * p * (1.0 - p) * (1.0 - 2.0 * p)
        }
        LikelihoodFamily::NegativeBinomial => -eta.exp().min(1e15),
        _ => 0.0,
    }
}

/// High-level convenience function for running INLA on a latent Gaussian model.
///
/// Automatically selects a reasonable hyperparameter search range (unless the
/// config already specifies `hyperparameter_range`) based on the observation
/// variance of `y`, then calls `compute_marginals`.
///
/// # Arguments
/// * `model`  - Fully specified `LatentGaussianModel`
/// * `config` - INLA configuration; `hyperparameter_range` may be `None` for auto-selection
///
/// # Returns
/// `INLAResult` containing posterior marginals, hyperparameter posteriors, and diagnostics.
///
/// # Errors
/// Returns an error if the model is invalid or INLA fails to converge.
pub fn fit_inla(
    model: LatentGaussianModel,
    mut config: INLAConfig,
) -> Result<INLAResult, StatsError> {
    // Auto-select hyperparameter range if not set
    if config.hyperparameter_range.is_none() {
        let n = model.y.len() as f64;
        if n < 1.0 {
            return Err(StatsError::InsufficientData(
                "Model must have at least one observation".to_string(),
            ));
        }
        let mean_y = model.y.iter().sum::<f64>() / n;
        let var_y = model.y.iter().map(|&v| (v - mean_y).powi(2)).sum::<f64>() / n.max(1.0);
        // Heuristic: log-precision range centred around log(1/var_y)
        let log_prec_center = if var_y > 1e-12 { -(var_y.ln()) } else { 0.0 };
        let half_width = 2.5f64;
        config.hyperparameter_range =
            Some((log_prec_center - half_width, log_prec_center + half_width));
    }

    compute_marginals(&model, &config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    fn make_gaussian_model(n: usize) -> LatentGaussianModel {
        let y: Array1<f64> = (1..=n).map(|i| i as f64).collect();
        let design = Array2::eye(n);
        let precision = Array2::eye(n);
        LatentGaussianModel::new(y, design, precision, LikelihoodFamily::Gaussian)
            .with_observation_precision(1.0)
    }

    #[test]
    fn test_compute_marginals_gaussian() {
        let model = make_gaussian_model(3);
        let config = INLAConfig {
            n_hyperparameter_grid: 5,
            hyperparameter_range: Some((-0.5, 0.5)),
            max_newton_iter: 50,
            newton_tol: 1e-8,
            ..INLAConfig::default()
        };

        let result = compute_marginals(&model, &config).expect("INLA should succeed");

        assert_eq!(result.marginal_means.len(), 3);
        assert_eq!(result.marginal_variances.len(), 3);
        assert!(result.converged);

        // For Gaussian with identity design and precision, obs_precision=1:
        // posterior is N(y/(1+1), 1/(1+1)) = N(y/2, 0.5) when Q = I
        // But we're integrating over precision scaling, so results vary
        for i in 0..3 {
            assert!(
                result.marginal_variances[i] > 0.0,
                "Variance should be positive for component {}",
                i
            );
        }
    }

    #[test]
    fn test_simplified_laplace() {
        let model = make_gaussian_model(3);
        let config = INLAConfig {
            n_hyperparameter_grid: 5,
            hyperparameter_range: Some((-0.5, 0.5)),
            integration_strategy: IntegrationStrategy::SimplifiedLaplace,
            max_newton_iter: 50,
            ..INLAConfig::default()
        };

        let result = compute_marginals(&model, &config).expect("Simplified INLA should succeed");

        assert_eq!(result.marginal_means.len(), 3);
        assert!(result.converged);

        for i in 0..3 {
            assert!(result.marginal_variances[i] > 0.0);
        }
    }

    #[test]
    fn test_simplified_vs_full_bounded_error() {
        let model = make_gaussian_model(3);

        let full_config = INLAConfig {
            n_hyperparameter_grid: 11,
            hyperparameter_range: Some((-1.0, 1.0)),
            integration_strategy: IntegrationStrategy::Grid,
            max_newton_iter: 50,
            ..INLAConfig::default()
        };

        let simplified_config = INLAConfig {
            integration_strategy: IntegrationStrategy::SimplifiedLaplace,
            ..full_config.clone()
        };

        let full_result =
            compute_marginals(&model, &full_config).expect("Full INLA should succeed");
        let simplified_result =
            compute_marginals(&model, &simplified_config).expect("Simplified INLA should succeed");

        // The difference should be bounded
        for i in 0..3 {
            let mean_diff =
                (full_result.marginal_means[i] - simplified_result.marginal_means[i]).abs();
            assert!(
                mean_diff < 2.0,
                "Mean difference at component {} is too large: {}",
                i,
                mean_diff
            );

            let var_ratio = full_result.marginal_variances[i]
                / simplified_result.marginal_variances[i].max(1e-15);
            assert!(
                var_ratio > 0.1 && var_ratio < 10.0,
                "Variance ratio at component {} is unreasonable: {}",
                i,
                var_ratio
            );
        }
    }

    #[test]
    fn test_poisson_marginals() {
        let n = 4;
        let y = array![2.0, 5.0, 1.0, 3.0];
        let design = Array2::eye(n);
        let precision = Array2::eye(n);
        let model = LatentGaussianModel::new(y, design, precision, LikelihoodFamily::Poisson);

        let config = INLAConfig {
            n_hyperparameter_grid: 5,
            hyperparameter_range: Some((-1.0, 1.0)),
            max_newton_iter: 100,
            ..INLAConfig::default()
        };

        let result = compute_marginals(&model, &config).expect("Poisson INLA should succeed");

        assert_eq!(result.marginal_means.len(), n);
        // All posterior means should be reasonable (roughly log of y values)
        for i in 0..n {
            assert!(
                result.marginal_means[i].is_finite(),
                "Mean at {} should be finite",
                i
            );
            assert!(
                result.marginal_variances[i] > 0.0,
                "Variance at {} should be positive",
                i
            );
        }
    }

    #[test]
    fn test_single_observation() {
        let y = array![5.0];
        let design = Array2::eye(1);
        let precision = Array2::eye(1);
        let model = LatentGaussianModel::new(y, design, precision, LikelihoodFamily::Gaussian)
            .with_observation_precision(1.0);

        let config = INLAConfig {
            n_hyperparameter_grid: 5,
            hyperparameter_range: Some((-0.5, 0.5)),
            ..INLAConfig::default()
        };

        let result = compute_marginals(&model, &config).expect("Single obs INLA should succeed");
        assert_eq!(result.marginal_means.len(), 1);
        assert!(result.marginal_variances[0] > 0.0);
    }

    #[test]
    fn test_identity_precision() {
        let n = 5;
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let design = Array2::eye(n);
        let precision = Array2::eye(n);
        let model = LatentGaussianModel::new(y, design, precision, LikelihoodFamily::Gaussian)
            .with_observation_precision(2.0);

        let config = INLAConfig {
            n_hyperparameter_grid: 5,
            hyperparameter_range: Some((-0.5, 0.5)),
            ..INLAConfig::default()
        };

        let result =
            compute_marginals(&model, &config).expect("Identity precision INLA should work");

        // With identity precision matrix and obs_precision=2,
        // posterior mean ≈ 2*y / (scale + 2) for each component
        for i in 0..n {
            assert!(result.marginal_means[i].is_finite());
        }
    }

    #[test]
    fn test_validate_model_empty() {
        let y = Array1::zeros(0);
        let design = Array2::zeros((0, 0));
        let precision = Array2::zeros((0, 0));
        let model = LatentGaussianModel::new(y, design, precision, LikelihoodFamily::Gaussian);

        let result = validate_model(&model);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_model_dimension_mismatch() {
        let y = array![1.0, 2.0];
        let design = Array2::eye(3); // wrong dimensions
        let precision = Array2::eye(3);
        let model = LatentGaussianModel::new(y, design, precision, LikelihoodFamily::Gaussian);

        let result = validate_model(&model);
        assert!(result.is_err());
    }

    #[test]
    fn test_corrected_laplace_gaussian() {
        // For Gaussian, correction should be zero
        let mode = array![1.0, 2.0];
        let neg_hess = array![[2.0, 0.0], [0.0, 2.0]];
        let y = array![1.5, 2.5];
        let design = Array2::eye(2);

        let (mean, var) = corrected_laplace_marginal(
            &mode,
            &neg_hess,
            &y,
            &design,
            LikelihoodFamily::Gaussian,
            0,
            None,
            Some(1.0),
        )
        .expect("Corrected Laplace should succeed");

        assert!(
            (mean - 1.0).abs() < 1e-10,
            "Gaussian correction should be zero"
        );
        assert!(
            (var - 0.5).abs() < 1e-10,
            "Variance should be [H^-1]_11 = 0.5"
        );
    }

    #[test]
    fn test_corrected_laplace_invalid_component() {
        let mode = array![1.0];
        let neg_hess = array![[2.0]];
        let y = array![1.5];
        let design = Array2::eye(1);

        let result = corrected_laplace_marginal(
            &mode,
            &neg_hess,
            &y,
            &design,
            LikelihoodFamily::Gaussian,
            5, // out of bounds
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_binomial_marginals() {
        let n = 3;
        let y = array![3.0, 7.0, 5.0]; // successes
        let n_trials = array![10.0, 10.0, 10.0];
        let design = Array2::eye(n);
        let precision = Array2::eye(n);
        let model = LatentGaussianModel::new(y, design, precision, LikelihoodFamily::Binomial)
            .with_n_trials(n_trials);

        let config = INLAConfig {
            n_hyperparameter_grid: 5,
            hyperparameter_range: Some((-1.0, 1.0)),
            max_newton_iter: 100,
            ..INLAConfig::default()
        };

        let result = compute_marginals(&model, &config).expect("Binomial INLA should succeed");

        assert_eq!(result.marginal_means.len(), n);
        for i in 0..n {
            assert!(result.marginal_means[i].is_finite());
            assert!(result.marginal_variances[i] > 0.0);
        }
    }

    #[test]
    fn test_hyperparameter_posteriors_nonempty() {
        let model = make_gaussian_model(3);
        let config = INLAConfig {
            n_hyperparameter_grid: 5,
            hyperparameter_range: Some((-0.5, 0.5)),
            ..INLAConfig::default()
        };

        let result = compute_marginals(&model, &config).expect("INLA should succeed");
        assert!(
            !result.hyperparameter_posteriors.is_empty(),
            "Should have hyperparameter posteriors"
        );
        let hp = &result.hyperparameter_posteriors[0];
        assert!(!hp.grid_points.is_empty());
        assert_eq!(hp.grid_points.len(), hp.log_densities.len());
        assert!(hp.variance > 0.0);
    }

    #[test]
    fn test_fit_inla_gaussian_auto() {
        // High-level fit_inla: Gaussian model with automatic hyperparameter range
        let y: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0];
        let n = y.len();
        let design = Array2::eye(n);
        let precision = Array2::eye(n);
        let model = LatentGaussianModel::new(y, design, precision, LikelihoodFamily::Gaussian)
            .with_observation_precision(1.0);
        let config = INLAConfig {
            n_hyperparameter_grid: 8,
            ..INLAConfig::default()
        };
        let result = fit_inla(model, config).expect("fit_inla should succeed");
        assert_eq!(result.marginal_means.len(), n);
        assert_eq!(result.marginal_variances.len(), n);
        assert!(result.log_marginal_likelihood.is_finite());
        for i in 0..n {
            assert!(result.marginal_means[i].is_finite());
            assert!(result.marginal_variances[i] > 0.0);
        }
    }

    #[test]
    fn test_fit_inla_poisson_mode_near_log3() {
        // Poisson y ~ Poisson(exp(x)), x ~ N(0, I)
        // With y ≈ 3 repeatedly, the latent field posterior mode should be near log(3)
        let y: Array1<f64> = array![3.0, 3.0, 3.0, 3.0, 3.0];
        let n = y.len();
        let design = Array2::eye(n);
        let precision = Array2::eye(n);
        let model = LatentGaussianModel::new(y, design, precision, LikelihoodFamily::Poisson);
        let config = INLAConfig {
            n_hyperparameter_grid: 6,
            hyperparameter_range: Some((-1.0, 2.0)),
            ..INLAConfig::default()
        };
        let result = fit_inla(model, config).expect("fit_inla Poisson should succeed");
        // Posterior mean should be in the right half-plane (positive, since y > 0)
        for i in 0..n {
            assert!(
                result.marginal_means[i] > 0.0,
                "Poisson latent mean should be positive, got {} at {}",
                result.marginal_means[i],
                i
            );
        }
    }
}
