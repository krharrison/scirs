//! Laplace approximation for Bayesian neural networks.
//!
//! The **Laplace approximation** (MacKay 1992; Ritter et al. 2018) fits a
//! Gaussian posterior centred at the MAP estimate θ*:
//!
//! ```text
//!   p(θ | D) ≈ N(θ | θ*, H⁻¹)
//! ```
//!
//! where `H = −∂² log p(θ|D) / ∂θ²` is the negative Hessian of the log-posterior.
//!
//! ## Diagonal GGN approximation (practical)
//!
//! For large networks, we use the **diagonal Generalized Gauss-Newton** (GGN)
//! approximation:
//!
//! ```text
//!   H_ii ≈ Σ_n (∂loss_n / ∂θ_i)²   (squared per-sample gradients, Fisher approx)
//!   H_ii += λ                        (prior precision)
//! ```
//!
//! This gives the diagonal posterior variance:
//!
//! ```text
//!   σ²_i = 1 / H_ii
//! ```

use crate::error::{StatsError, StatsResult};

use super::types::{BnnApproxResult, HessianMethod, LaplaceConfig};

// ============================================================================
// Core GGN diagonal computation
// ============================================================================

/// Compute the **diagonal GGN** (squared-gradient Fisher) approximation of the
/// Hessian given per-data-point gradients.
///
/// # Arguments
/// * `grad_matrix` — Matrix of shape `[n_data × n_params]` where `grad_matrix[i][j]`
///   is `∂loss_i / ∂θ_j`.
///
/// # Returns
/// Vector of length `n_params` with `H_diag[j] = Σᵢ (∂loss_i/∂θ_j)²`.
pub fn diagonal_ggn(grad_matrix: &[Vec<f64>]) -> StatsResult<Vec<f64>> {
    if grad_matrix.is_empty() {
        return Err(StatsError::invalid_argument(
            "diagonal_ggn: grad_matrix must not be empty",
        ));
    }
    let n_params = grad_matrix[0].len();
    if n_params == 0 {
        return Err(StatsError::invalid_argument(
            "diagonal_ggn: each gradient vector must have length > 0",
        ));
    }
    for (i, row) in grad_matrix.iter().enumerate() {
        if row.len() != n_params {
            return Err(StatsError::dimension_mismatch(format!(
                "grad_matrix row {} has length {} ≠ n_params {}",
                i,
                row.len(),
                n_params
            )));
        }
    }

    let mut ggn = vec![0.0f64; n_params];
    for row in grad_matrix {
        for (j, &g) in row.iter().enumerate() {
            ggn[j] += g * g;
        }
    }
    Ok(ggn)
}

// ============================================================================
// Finite-difference gradient helper
// ============================================================================

/// Compute the per-sample finite-difference gradient of a scalar loss with
/// respect to the weight vector, returning a `[n_data × n_params]` matrix.
///
/// The `loss_fn` receives a weight slice and returns the **per-sample** losses
/// `(loss_0, loss_1, ..., loss_{n-1})`. We differentiate each loss separately
/// via central differences.
///
/// This is the most straightforward (and most expensive) approach: it calls
/// `loss_fn` twice per parameter per sample. For practical networks use
/// automatic differentiation instead.
pub fn fd_per_sample_gradients(
    weights: &[f64],
    loss_fn: &dyn Fn(&[f64]) -> Vec<f64>,
    fd_step: f64,
) -> StatsResult<Vec<Vec<f64>>> {
    if weights.is_empty() {
        return Err(StatsError::invalid_argument(
            "fd_per_sample_gradients: weights must not be empty",
        ));
    }
    if fd_step <= 0.0 {
        return Err(StatsError::invalid_argument(
            "fd_per_sample_gradients: fd_step must be > 0",
        ));
    }

    // Determine n_data from a probe call
    let losses_at_w = loss_fn(weights);
    let n_data = losses_at_w.len();
    if n_data == 0 {
        return Err(StatsError::invalid_argument(
            "fd_per_sample_gradients: loss_fn returned empty vector",
        ));
    }

    let n_params = weights.len();
    // grad_matrix[i][j] = ∂loss_i / ∂θ_j
    let mut grad_matrix = vec![vec![0.0f64; n_params]; n_data];

    let mut w_fwd = weights.to_vec();
    let mut w_bwd = weights.to_vec();

    for j in 0..n_params {
        w_fwd[j] = weights[j] + fd_step;
        w_bwd[j] = weights[j] - fd_step;

        let l_fwd = loss_fn(&w_fwd);
        let l_bwd = loss_fn(&w_bwd);

        for i in 0..n_data {
            grad_matrix[i][j] = (l_fwd[i] - l_bwd[i]) / (2.0 * fd_step);
        }

        // Restore
        w_fwd[j] = weights[j];
        w_bwd[j] = weights[j];
    }

    Ok(grad_matrix)
}

// ============================================================================
// Laplace posterior variance
// ============================================================================

/// Compute the diagonal posterior variance from the GGN diagonal.
///
/// `σ²_i = 1 / (H_ii + λ)` where `λ = config.damping` is the prior precision.
///
/// # Errors
/// Returns an error if `ggn_diag` is empty or contains non-finite values.
pub fn posterior_variance_from_ggn(ggn_diag: &[f64], damping: f64) -> StatsResult<Vec<f64>> {
    if ggn_diag.is_empty() {
        return Err(StatsError::invalid_argument(
            "posterior_variance_from_ggn: ggn_diag must not be empty",
        ));
    }
    let var: Vec<f64> = ggn_diag
        .iter()
        .map(|&h| {
            let denom = h + damping;
            if denom <= 0.0 {
                1.0 / damping.max(1e-12)
            } else {
                1.0 / denom
            }
        })
        .collect();
    Ok(var)
}

// ============================================================================
// Prediction with linearized uncertainty
// ============================================================================

/// Compute the MAP (mean) prediction at a single test point `x`.
///
/// Uses a simple linear model `f(x; θ) = θ · x` (dot product).
/// Replace with your own model forward pass for more complex networks.
pub fn predict_mean_linear(x: &[f64], weights: &[f64]) -> StatsResult<f64> {
    if x.len() != weights.len() {
        return Err(StatsError::dimension_mismatch(format!(
            "predict_mean: x.len()={} ≠ weights.len()={}",
            x.len(),
            weights.len()
        )));
    }
    Ok(x.iter().zip(weights).map(|(&xi, &wi)| xi * wi).sum())
}

/// Compute the **linearized predictive variance** at a single test point `x`:
///
/// ```text
///   σ²_pred(x) = Σᵢ (∂f/∂θᵢ)² · σ²_posterior[i]
///              = Σᵢ x_i² · σ²[i]           (for the linear model f = θᵀx)
/// ```
///
/// For general models, supply the Jacobian of f w.r.t. θ at x instead of x itself.
pub fn predict_variance_linear(x: &[f64], posterior_var: &[f64]) -> StatsResult<f64> {
    if x.len() != posterior_var.len() {
        return Err(StatsError::dimension_mismatch(format!(
            "predict_variance: x.len()={} ≠ posterior_var.len()={}",
            x.len(),
            posterior_var.len()
        )));
    }
    Ok(x.iter()
        .zip(posterior_var)
        .map(|(&xi, &vi)| xi * xi * vi)
        .sum())
}

// ============================================================================
// High-level fit_laplace
// ============================================================================

/// Fit a diagonal Laplace approximation at `map_weights`.
///
/// # Arguments
/// * `map_weights` — MAP weight vector θ*
/// * `loss_fn` — Closure computing per-sample losses `loss_fn(θ) -> Vec<f64>`
///   of length `n_data`.
/// * `config` — Laplace configuration.
///
/// # Returns
/// A [`BnnApproxResult`] with `mean_weights = map_weights` and
/// `uncertainty = posterior_variance`.
///
/// # Errors
/// Returns an error if `map_weights` is empty or `loss_fn` returns empty losses.
pub fn fit_laplace(
    map_weights: &[f64],
    loss_fn: &dyn Fn(&[f64]) -> Vec<f64>,
    config: &LaplaceConfig,
) -> StatsResult<BnnApproxResult> {
    if map_weights.is_empty() {
        return Err(StatsError::invalid_argument(
            "fit_laplace: map_weights must not be empty",
        ));
    }

    let grad_matrix = match config.hessian_method {
        HessianMethod::GGN | HessianMethod::Diagonal | HessianMethod::KFAC => {
            fd_per_sample_gradients(map_weights, loss_fn, config.fd_step)?
        }
        _ => {
            // Fallback for future variants
            fd_per_sample_gradients(map_weights, loss_fn, config.fd_step)?
        }
    };

    let ggn = diagonal_ggn(&grad_matrix)?;
    let posterior_var = posterior_variance_from_ggn(&ggn, config.damping)?;

    Ok(BnnApproxResult {
        mean_weights: map_weights.to_vec(),
        uncertainty: posterior_var,
        method: format!("Laplace-{:?}", config.hessian_method),
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagonal_laplace_squared_grads() {
        // Simple case: 2 data points, 2 params
        // grads = [[1.0, 2.0], [3.0, 4.0]]
        // expected GGN = [1² + 3², 2² + 4²] = [10, 20]
        let grads = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let ggn = diagonal_ggn(&grads).expect("ok");
        assert!((ggn[0] - 10.0).abs() < 1e-12);
        assert!((ggn[1] - 20.0).abs() < 1e-12);
    }

    #[test]
    fn test_diagonal_ggn_single_sample() {
        let grads = vec![vec![3.0, -1.0, 2.0]];
        let ggn = diagonal_ggn(&grads).expect("ok");
        assert!((ggn[0] - 9.0).abs() < 1e-12);
        assert!((ggn[1] - 1.0).abs() < 1e-12);
        assert!((ggn[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_laplace_posterior_variance_positive() {
        let ggn = vec![1.0, 0.0, 5.0];
        let var = posterior_variance_from_ggn(&ggn, 1.0).expect("ok");
        for &v in &var {
            assert!(v > 0.0, "variance should be positive, got {v}");
        }
        // ggn[0]+λ = 2 → var = 0.5
        assert!((var[0] - 0.5).abs() < 1e-12);
        // ggn[1]+λ = 1 → var = 1.0
        assert!((var[1] - 1.0).abs() < 1e-12);
        // ggn[2]+λ = 6 → var = 1/6
        assert!((var[2] - 1.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_laplace_predict_variance_finite() {
        // linear model f = w·x; x = [1.0, 2.0], var = [0.5, 0.25]
        let x = vec![1.0, 2.0];
        let posterior_var = vec![0.5, 0.25];
        let var = predict_variance_linear(&x, &posterior_var).expect("ok");
        // σ² = 1²·0.5 + 2²·0.25 = 0.5 + 1.0 = 1.5
        assert!(var.is_finite(), "variance should be finite");
        assert!((var - 1.5).abs() < 1e-12, "expected 1.5, got {var}");
    }

    #[test]
    fn test_predict_mean_linear() {
        let x = vec![1.0, 2.0, 3.0];
        let w = vec![1.0, 1.0, 1.0];
        let pred = predict_mean_linear(&x, &w).expect("ok");
        assert!((pred - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_laplace_uncertainty_increases_far_from_data() {
        // For a linear model f = w·x, the predictive variance is x² · σ²_w.
        // At x=10 (far) vs x=1 (near), the uncertainty should be higher.
        // This test uses the analytical formula directly.
        let posterior_var = vec![0.1]; // σ²_w = 0.1
        let x_near = vec![1.0];
        let x_far = vec![10.0];
        let var_near = predict_variance_linear(&x_near, &posterior_var).expect("near");
        let var_far = predict_variance_linear(&x_far, &posterior_var).expect("far");
        assert!(
            var_far > var_near,
            "Uncertainty should be higher far from data: near={var_near}, far={var_far}"
        );
    }

    #[test]
    fn test_fit_laplace_basic() {
        // Single-weight linear model: f(x; w) = w * x
        // Data: x = [1.0, 2.0, 3.0], y = [1.0, 2.0, 3.0] (perfect fit at w=1)
        let x_data = vec![1.0f64, 2.0, 3.0];
        let y_data = vec![1.0f64, 2.0, 3.0];
        let loss_fn = move |w: &[f64]| -> Vec<f64> {
            x_data
                .iter()
                .zip(&y_data)
                .map(|(&xi, &yi)| (yi - w[0] * xi).powi(2))
                .collect()
        };

        let config = LaplaceConfig::default();
        let result = fit_laplace(&[1.0], &loss_fn, &config).expect("fit");
        assert_eq!(result.mean_weights.len(), 1);
        assert_eq!(result.uncertainty.len(), 1);
        assert!(result.uncertainty[0] > 0.0, "variance must be positive");
        assert!(result.uncertainty[0].is_finite(), "variance must be finite");
    }

    #[test]
    fn test_fit_laplace_empty_weights_error() {
        let loss_fn = |_: &[f64]| vec![1.0];
        let config = LaplaceConfig::default();
        assert!(fit_laplace(&[], &loss_fn, &config).is_err());
    }

    #[test]
    fn test_diagonal_ggn_empty_error() {
        assert!(diagonal_ggn(&[]).is_err());
    }
}
