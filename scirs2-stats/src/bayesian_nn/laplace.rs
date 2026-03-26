//! Laplace approximation for Bayesian neural networks.
//!
//! The Laplace approximation fits a Gaussian posterior centered at the MAP
//! estimate with covariance equal to the inverse of the Gauss-Newton Hessian:
//!
//!   H = J^T J + prior_precision * I
//!   Sigma = H^{-1}
//!
//! where J is the Jacobian of the network outputs with respect to parameters.

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{StatsError, StatsResult};

use super::types::{BNNConfig, BNNPosterior, CovarianceType, PredictiveDistribution};

/// Laplace approximation of the posterior over neural network weights.
#[derive(Debug, Clone)]
pub struct LaplaceApproximation {
    /// Fitted posterior
    posterior: BNNPosterior,
    /// Configuration used
    config: BNNConfig,
}

impl LaplaceApproximation {
    /// Fit a Laplace approximation at the MAP estimate.
    ///
    /// Computes H = J^T J + prior_precision * I and inverts it to obtain
    /// the posterior covariance Sigma = H^{-1}.
    ///
    /// # Arguments
    /// * `weights` - MAP weights theta* (length d)
    /// * `jacobian` - Jacobian matrix J\[i,j\] = df_i / dtheta_j, shape \[n x d\]
    /// * `residuals` - residuals y - f(x; theta*), length n
    /// * `config` - BNN configuration
    ///
    /// # Errors
    /// Returns an error if dimensions are inconsistent or the Hessian is singular.
    pub fn fit(
        weights: &Array1<f64>,
        jacobian: &Array2<f64>,
        residuals: &Array1<f64>,
        config: &BNNConfig,
    ) -> StatsResult<Self> {
        let n = jacobian.nrows();
        let d = jacobian.ncols();

        if weights.len() != d {
            return Err(StatsError::dimension_mismatch(format!(
                "weights length {} != jacobian columns {}",
                weights.len(),
                d
            )));
        }
        if residuals.len() != n {
            return Err(StatsError::dimension_mismatch(format!(
                "residuals length {} != jacobian rows {}",
                residuals.len(),
                n
            )));
        }
        if d == 0 {
            return Err(StatsError::invalid_argument(
                "Number of parameters must be > 0",
            ));
        }

        // H = J^T J + prior_precision * I
        let jtj = jacobian.t().dot(jacobian);
        let mut hessian = jtj;
        for i in 0..d {
            hessian[[i, i]] += config.prior_precision;
        }

        // Invert H via Cholesky decomposition
        let cov = cholesky_inverse(&hessian)?;

        // Log marginal likelihood:
        // log p(y|M) ~ log p(y|theta*) + log p(theta*) + (d/2) log(2 pi) - 0.5 log|H|
        let sse: f64 = residuals.iter().map(|r| r * r).sum();
        let log_likelihood = -0.5 * sse; // assuming unit noise variance
        let log_prior = -0.5 * config.prior_precision * weights.iter().map(|w| w * w).sum::<f64>();
        let log_det_h = cholesky_log_det(&hessian)?;
        let log_marginal =
            log_likelihood + log_prior + 0.5 * (d as f64) * (2.0 * std::f64::consts::PI).ln()
                - 0.5 * log_det_h;

        let posterior = BNNPosterior {
            mean: weights.clone(),
            covariance_type: CovarianceType::Full(cov),
            log_marginal_likelihood: log_marginal,
        };

        Ok(Self {
            posterior,
            config: config.clone(),
        })
    }

    /// Predict with uncertainty via linearized model.
    ///
    /// Under the linearization f(x; theta) ~ f(x; theta*) + J_x (theta - theta*),
    /// the predictive variance is Var\[f\] = J_x Sigma J_x^T.
    ///
    /// # Arguments
    /// * `jacobian_test` - Jacobian at test points, shape \[n_test x d\]
    /// * `mean_prediction` - f(x_test; theta*), length n_test
    ///
    /// # Errors
    /// Returns an error if dimensions are inconsistent.
    pub fn predict(
        &self,
        jacobian_test: &Array2<f64>,
        mean_prediction: &Array1<f64>,
    ) -> StatsResult<PredictiveDistribution> {
        let n_test = jacobian_test.nrows();
        let d = jacobian_test.ncols();

        if mean_prediction.len() != n_test {
            return Err(StatsError::dimension_mismatch(format!(
                "mean_prediction length {} != jacobian_test rows {}",
                mean_prediction.len(),
                n_test
            )));
        }

        let cov = match &self.posterior.covariance_type {
            CovarianceType::Full(c) => c,
            _ => {
                return Err(StatsError::computation(
                    "Laplace predict requires Full covariance",
                ))
            }
        };

        if cov.nrows() != d || cov.ncols() != d {
            return Err(StatsError::dimension_mismatch(format!(
                "Covariance shape [{}, {}] incompatible with Jacobian columns {}",
                cov.nrows(),
                cov.ncols(),
                d
            )));
        }

        // Var[f_i] = J_x[i,:] Sigma J_x[i,:]^T
        let j_sigma = jacobian_test.dot(cov); // [n_test x d]
        let mut variance = Array1::zeros(n_test);
        for i in 0..n_test {
            let mut v = 0.0;
            for j in 0..d {
                v += j_sigma[[i, j]] * jacobian_test[[i, j]];
            }
            variance[i] = v;
        }

        Ok(PredictiveDistribution {
            mean: mean_prediction.clone(),
            variance,
            samples: None,
        })
    }

    /// Return the estimated log marginal likelihood.
    pub fn log_marginal_likelihood(&self) -> f64 {
        self.posterior.log_marginal_likelihood
    }

    /// Return a reference to the fitted posterior.
    pub fn posterior(&self) -> &BNNPosterior {
        &self.posterior
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &BNNConfig {
        &self.config
    }
}

/// Compute Kronecker-factored approximate curvature (KFAC) factors.
///
/// For a single layer with input activations a and output gradients g,
/// the Fisher information is approximated as F ~ A kron B where:
///   A = (1/n) * activations^T * activations
///   B = (1/n) * gradients^T * gradients
///
/// # Arguments
/// * `activations` - Input activations, shape \[n x d_in\]
/// * `gradients` - Output gradients, shape \[n x d_out\]
///
/// # Returns
/// A tuple (A, B) of the Kronecker factors.
pub fn kfac_factors(
    activations: &Array2<f64>,
    gradients: &Array2<f64>,
) -> StatsResult<(Array2<f64>, Array2<f64>)> {
    let n_a = activations.nrows();
    let n_g = gradients.nrows();

    if n_a != n_g {
        return Err(StatsError::dimension_mismatch(format!(
            "activations rows {} != gradients rows {}",
            n_a, n_g
        )));
    }
    if n_a == 0 {
        return Err(StatsError::invalid_argument(
            "Need at least 1 sample for KFAC",
        ));
    }

    let n = n_a as f64;
    let a_factor = activations.t().dot(activations) / n;
    let b_factor = gradients.t().dot(gradients) / n;

    Ok((a_factor, b_factor))
}

/// Cholesky decomposition of a symmetric positive-definite matrix.
/// Returns the lower-triangular factor L such that A = L L^T.
fn cholesky_decompose(a: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(StatsError::dimension_mismatch("Matrix must be square"));
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut s = 0.0;
        for k in 0..j {
            s += l[[j, k]] * l[[j, k]];
        }
        let diag = a[[j, j]] - s;
        if diag <= 0.0 {
            return Err(StatsError::computation(format!(
                "Matrix is not positive definite (pivot {} at index {})",
                diag, j
            )));
        }
        l[[j, j]] = diag.sqrt();
        for i in (j + 1)..n {
            let mut s2 = 0.0;
            for k in 0..j {
                s2 += l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = (a[[i, j]] - s2) / l[[j, j]];
        }
    }
    Ok(l)
}

/// Invert a symmetric positive-definite matrix via Cholesky decomposition.
fn cholesky_inverse(a: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let l = cholesky_decompose(a)?;
    let n = l.nrows();

    // Invert L (lower triangular)
    let mut l_inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        l_inv[[i, i]] = 1.0 / l[[i, i]];
        for j in (0..i).rev() {
            let mut s = 0.0;
            for k in (j + 1)..=i {
                s += l[[i, k]] * l_inv[[k, j]];
            }
            l_inv[[i, j]] = -s / l[[i, i]]; // note: l[[i,i]] not l[[j,j]]
        }
    }

    // A^{-1} = L^{-T} L^{-1}
    let inv = l_inv.t().dot(&l_inv);
    Ok(inv)
}

/// Compute log determinant via Cholesky: log|A| = 2 * sum(log(L_ii))
fn cholesky_log_det(a: &Array2<f64>) -> StatsResult<f64> {
    let l = cholesky_decompose(a)?;
    let n = l.nrows();
    let mut log_det = 0.0;
    for i in 0..n {
        log_det += l[[i, i]].ln();
    }
    Ok(2.0 * log_det)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, Array2};

    #[test]
    fn test_cholesky_identity() {
        let eye = Array2::from_diag(&Array1::from_vec(vec![1.0, 1.0, 1.0]));
        let l = cholesky_decompose(&eye).expect("Cholesky of identity should succeed");
        for i in 0..3 {
            assert!((l[[i, i]] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cholesky_inverse_identity() {
        let eye = Array2::from_diag(&Array1::from_vec(vec![2.0, 3.0, 5.0]));
        let inv = cholesky_inverse(&eye).expect("inverse of diagonal should succeed");
        assert!((inv[[0, 0]] - 0.5).abs() < 1e-12);
        assert!((inv[[1, 1]] - 1.0 / 3.0).abs() < 1e-12);
        assert!((inv[[2, 2]] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_laplace_quadratic_loss() {
        // For a quadratic f(x;w) = w*x with data y = w*x + noise,
        // the exact posterior under Gaussian likelihood is Gaussian.
        // J = x (n x 1), H = x^T x + prior_precision
        let x_data = array![[1.0], [2.0], [3.0]];
        let w_map = array![2.0];
        let residuals = array![0.1, -0.05, 0.02]; // small residuals at MAP

        let config = BNNConfig {
            prior_precision: 1.0,
            ..BNNConfig::default()
        };

        let lap = LaplaceApproximation::fit(&w_map, &x_data, &residuals, &config)
            .expect("Laplace fit should succeed");

        // H = 1^2 + 2^2 + 3^2 + 1 = 15, Sigma = 1/15
        match &lap.posterior().covariance_type {
            CovarianceType::Full(cov) => {
                let expected_var = 1.0 / 15.0;
                assert!(
                    (cov[[0, 0]] - expected_var).abs() < 1e-10,
                    "Expected variance {}, got {}",
                    expected_var,
                    cov[[0, 0]]
                );
            }
            _ => panic!("Expected Full covariance"),
        }
    }

    #[test]
    fn test_laplace_predict_uncertainty_grows() {
        // Uncertainty should grow with distance from training data
        let x_data = array![[1.0], [2.0]];
        let w_map = array![1.0];
        let residuals = array![0.0, 0.0];
        let config = BNNConfig::default();

        let lap = LaplaceApproximation::fit(&w_map, &x_data, &residuals, &config).expect("fit");

        let j_near = array![[1.5]]; // close to training
        let j_far = array![[10.0]]; // far from training

        let pred_near = lap.predict(&j_near, &array![1.5]).expect("predict near");
        let pred_far = lap.predict(&j_far, &array![10.0]).expect("predict far");

        assert!(
            pred_far.variance[0] > pred_near.variance[0],
            "Uncertainty should grow: near={}, far={}",
            pred_near.variance[0],
            pred_far.variance[0]
        );
    }

    #[test]
    fn test_laplace_dimension_mismatch() {
        let w = array![1.0, 2.0];
        let j = array![[1.0]]; // 1 col, but w has 2 elements
        let r = array![0.1];
        let config = BNNConfig::default();
        assert!(LaplaceApproximation::fit(&w, &j, &r, &config).is_err());
    }

    #[test]
    fn test_kfac_factors_symmetric_psd() {
        let activations = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let gradients = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]];

        let (a, b) = kfac_factors(&activations, &gradients).expect("KFAC should succeed");

        // A should be d_in x d_in = 2x2
        assert_eq!(a.nrows(), 2);
        assert_eq!(a.ncols(), 2);
        // B should be d_out x d_out = 3x3
        assert_eq!(b.nrows(), 3);
        assert_eq!(b.ncols(), 3);

        // Symmetric: A[i,j] == A[j,i]
        assert!((a[[0, 1]] - a[[1, 0]]).abs() < 1e-12);

        // PSD: diagonal elements >= 0
        for i in 0..a.nrows() {
            assert!(a[[i, i]] >= 0.0);
        }
        for i in 0..b.nrows() {
            assert!(b[[i, i]] >= 0.0);
        }
    }

    #[test]
    fn test_kfac_row_mismatch() {
        let a = array![[1.0], [2.0]];
        let g = array![[1.0], [2.0], [3.0]];
        assert!(kfac_factors(&a, &g).is_err());
    }

    #[test]
    fn test_log_marginal_likelihood_finite() {
        let j = array![[1.0, 0.0], [0.0, 1.0]];
        let w = array![1.0, 1.0];
        let r = array![0.1, -0.1];
        let config = BNNConfig::default();
        let lap = LaplaceApproximation::fit(&w, &j, &r, &config).expect("fit");
        let lml = lap.log_marginal_likelihood();
        assert!(lml.is_finite(), "log marginal likelihood should be finite");
    }
}
