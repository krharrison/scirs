//! Recursive Least Squares (RLS) for online system identification
//!
//! Implements RLS with exponential forgetting for tracking time-varying
//! systems in real-time. The algorithm maintains a running estimate
//! of the parameter vector and its covariance matrix.
//!
//! Update equations:
//!   K(t) = P(t-1) phi(t) / (lambda + phi(t)^T P(t-1) phi(t))
//!   theta(t) = theta(t-1) + K(t) * (y(t) - phi(t)^T theta(t-1))
//!   P(t) = (P(t-1) - K(t) phi(t)^T P(t-1)) / lambda

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};

use super::types::RlsConfig;

/// Online RLS estimator for system identification
///
/// # Example
/// ```rust
/// use scirs2_signal::system_identification::{RlsEstimator, RlsConfig};
/// use scirs2_core::ndarray::Array1;
///
/// // Track parameters of y[k] = a*y[k-1] + b*u[k-1]
/// let config = RlsConfig {
///     n_params: 2,
///     forgetting_factor: 0.98,
///     initial_covariance: 1000.0,
/// };
/// let mut rls = RlsEstimator::new(&config).expect("RLS init failed");
///
/// // Feed data points
/// let phi = Array1::from_vec(vec![-0.5, 0.3]); // [-y[k-1], u[k-1]]
/// let y_k = 0.8 * 0.5 + 0.5 * 0.3; // true output
/// let error = rls.update(&phi, y_k).expect("RLS update failed");
/// ```
#[derive(Debug, Clone)]
pub struct RlsEstimator {
    /// Current parameter estimates
    pub theta: Array1<f64>,
    /// Covariance matrix (inverse of information matrix)
    pub covariance: Array2<f64>,
    /// Forgetting factor
    pub lambda: f64,
    /// Number of parameters
    pub n_params: usize,
    /// Number of updates performed
    pub n_updates: usize,
    /// Cumulative prediction error energy
    pub cumulative_error: f64,
}

impl RlsEstimator {
    /// Create a new RLS estimator
    ///
    /// # Arguments
    /// * `config` - RLS configuration
    pub fn new(config: &RlsConfig) -> SignalResult<Self> {
        if config.n_params == 0 {
            return Err(SignalError::ValueError(
                "Number of parameters must be positive".into(),
            ));
        }
        if config.forgetting_factor <= 0.0 || config.forgetting_factor > 1.0 {
            return Err(SignalError::ValueError(
                "Forgetting factor must be in (0, 1]".into(),
            ));
        }
        if config.initial_covariance <= 0.0 {
            return Err(SignalError::ValueError(
                "Initial covariance must be positive".into(),
            ));
        }

        let theta = Array1::<f64>::zeros(config.n_params);
        let covariance = Array2::<f64>::eye(config.n_params) * config.initial_covariance;

        Ok(Self {
            theta,
            covariance,
            lambda: config.forgetting_factor,
            n_params: config.n_params,
            n_updates: 0,
            cumulative_error: 0.0,
        })
    }

    /// Update parameter estimates with a new data point
    ///
    /// # Arguments
    /// * `phi` - Regression vector (length = n_params)
    /// * `y` - Measured output
    ///
    /// # Returns
    /// * A-priori prediction error (before update)
    pub fn update(&mut self, phi: &Array1<f64>, y: f64) -> SignalResult<f64> {
        if phi.len() != self.n_params {
            return Err(SignalError::DimensionMismatch(format!(
                "Regression vector length {} does not match n_params {}",
                phi.len(),
                self.n_params
            )));
        }

        // A-priori prediction error
        let y_hat = self.theta.dot(phi);
        let error = y - y_hat;

        // Gain vector: K = P * phi / (lambda + phi^T * P * phi)
        let p_phi = self.covariance.dot(phi);
        let denom = self.lambda + phi.dot(&p_phi);

        if denom.abs() < 1e-15 {
            return Err(SignalError::ComputationError(
                "RLS update: denominator near zero (numerical issue)".into(),
            ));
        }

        let gain = &p_phi / denom;

        // Parameter update: theta = theta + K * error
        self.theta = &self.theta + &(&gain * error);

        // Covariance update: P = (P - K * phi^T * P) / lambda
        // Using Joseph form for better numerical stability:
        // P = (I - K * phi^T) * P / lambda
        let k_phi_t = gain
            .clone()
            .insert_axis(Axis(1))
            .dot(&phi.clone().insert_axis(Axis(0)));
        let eye = Array2::<f64>::eye(self.n_params);
        let factor = &eye - &k_phi_t;
        self.covariance = factor.dot(&self.covariance) / self.lambda;

        // Symmetrize covariance to prevent drift
        let cov_t = self.covariance.t().to_owned();
        self.covariance = (&self.covariance + &cov_t) * 0.5;

        self.n_updates += 1;
        self.cumulative_error += error * error;

        Ok(error)
    }

    /// Get current parameter estimates
    pub fn parameters(&self) -> &Array1<f64> {
        &self.theta
    }

    /// Get current covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.covariance
    }

    /// Get parameter standard deviations (sqrt of diagonal of covariance)
    pub fn parameter_std(&self) -> Array1<f64> {
        let n = self.n_params;
        let mut std_devs = Array1::<f64>::zeros(n);
        for i in 0..n {
            std_devs[i] = self.covariance[[i, i]].max(0.0).sqrt();
        }
        std_devs
    }

    /// Get average prediction error energy
    pub fn average_error(&self) -> f64 {
        if self.n_updates > 0 {
            self.cumulative_error / self.n_updates as f64
        } else {
            0.0
        }
    }

    /// Reset the estimator to initial state
    pub fn reset(&mut self, initial_covariance: f64) {
        self.theta = Array1::<f64>::zeros(self.n_params);
        self.covariance = Array2::<f64>::eye(self.n_params) * initial_covariance;
        self.n_updates = 0;
        self.cumulative_error = 0.0;
    }
}

/// Run batch RLS estimation on input-output data
///
/// Convenience function that creates an RLS estimator, builds regression
/// vectors from the data, and returns the final parameter estimates.
///
/// # Arguments
/// * `y` - Output signal
/// * `u` - Input signal
/// * `na` - Number of AR (output) lags
/// * `nb` - Number of B (input) lags (including b_0)
/// * `nk` - Input delay
/// * `config` - RLS configuration
///
/// # Returns
/// * Final RLS estimator with converged parameters
pub fn rls_batch(
    y: &Array1<f64>,
    u: &Array1<f64>,
    na: usize,
    nb: usize,
    nk: usize,
    config: &RlsConfig,
) -> SignalResult<RlsEstimator> {
    let n = y.len();
    if n != u.len() {
        return Err(SignalError::DimensionMismatch(
            "Input and output must have the same length".into(),
        ));
    }

    let start = na.max(nb + nk);
    if n <= start {
        return Err(SignalError::ValueError(
            "Insufficient data for the specified model orders".into(),
        ));
    }

    let expected_params = na + nb + 1;
    let adjusted_config = RlsConfig {
        n_params: expected_params,
        forgetting_factor: config.forgetting_factor,
        initial_covariance: config.initial_covariance,
    };

    let mut rls = RlsEstimator::new(&adjusted_config)?;

    for t in start..n {
        let mut phi = Array1::<f64>::zeros(expected_params);

        // AR part: -y(t-1), ..., -y(t-na)
        for j in 0..na {
            phi[j] = -y[t - 1 - j];
        }

        // B part: u(t-nk), ..., u(t-nk-nb)
        for j in 0..=nb {
            let idx = t as isize - nk as isize - j as isize;
            if idx >= 0 && (idx as usize) < n {
                phi[na + j] = u[idx as usize];
            }
        }

        let _ = rls.update(&phi, y[t])?;
    }

    Ok(rls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rls_known_system() {
        // True system: y = 2*x1 + 3*x2
        let config = RlsConfig {
            n_params: 2,
            forgetting_factor: 0.99,
            initial_covariance: 1000.0,
        };
        let mut rls = RlsEstimator::new(&config).expect("RLS init failed");

        let data = vec![
            (vec![1.0, 2.0], 8.0),
            (vec![2.0, 1.0], 7.0),
            (vec![0.5, 1.5], 5.5),
            (vec![1.5, 0.5], 4.5),
            (vec![3.0, 1.0], 9.0),
            (vec![1.0, 3.0], 11.0),
        ];

        for _ in 0..50 {
            for (x, y_val) in &data {
                let phi = Array1::from_vec(x.clone());
                let _ = rls.update(&phi, *y_val).expect("RLS update failed");
            }
        }

        let params = rls.parameters();
        assert_relative_eq!(params[0], 2.0, epsilon = 0.1);
        assert_relative_eq!(params[1], 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_rls_batch_arx() {
        // True system: y[k] = 0.8*y[k-1] + 0.5*u[k-1]
        let n = 500;
        let mut u = Array1::<f64>::zeros(n);
        let mut y = Array1::<f64>::zeros(n);

        for i in 0..n {
            u[i] = ((i as f64) * 0.1).sin();
        }
        for i in 1..n {
            y[i] = 0.8 * y[i - 1] + 0.5 * u[i - 1];
        }

        let config = RlsConfig {
            n_params: 2,
            forgetting_factor: 0.99,
            initial_covariance: 1000.0,
        };

        let rls = rls_batch(&y, &u, 1, 0, 1, &config).expect("RLS batch failed");

        let params = rls.parameters();
        // a_1 coefficient (negated in regression): should be close to -0.8
        assert_relative_eq!(params[0], -0.8, epsilon = 0.1);
        // b_0 coefficient: should be close to 0.5
        assert_relative_eq!(params[1], 0.5, epsilon = 0.1);
    }

    #[test]
    fn test_rls_parameter_std() {
        let config = RlsConfig {
            n_params: 2,
            forgetting_factor: 0.99,
            initial_covariance: 100.0,
        };
        let rls = RlsEstimator::new(&config).expect("RLS init failed");

        let std_devs = rls.parameter_std();
        assert_eq!(std_devs.len(), 2);
        // Initial std should be sqrt(100) = 10
        assert_relative_eq!(std_devs[0], 10.0, epsilon = 0.01);
    }

    #[test]
    fn test_rls_invalid_config() {
        let config = RlsConfig {
            n_params: 0,
            forgetting_factor: 0.99,
            initial_covariance: 100.0,
        };
        assert!(RlsEstimator::new(&config).is_err());

        let config2 = RlsConfig {
            n_params: 2,
            forgetting_factor: 0.0,
            initial_covariance: 100.0,
        };
        assert!(RlsEstimator::new(&config2).is_err());
    }

    #[test]
    fn test_rls_dimension_mismatch() {
        let config = RlsConfig {
            n_params: 2,
            forgetting_factor: 0.99,
            initial_covariance: 100.0,
        };
        let mut rls = RlsEstimator::new(&config).expect("RLS init failed");

        let phi = Array1::from_vec(vec![1.0, 2.0, 3.0]); // wrong size
        assert!(rls.update(&phi, 1.0).is_err());
    }

    #[test]
    fn test_rls_reset() {
        let config = RlsConfig {
            n_params: 2,
            forgetting_factor: 0.99,
            initial_covariance: 100.0,
        };
        let mut rls = RlsEstimator::new(&config).expect("RLS init failed");

        let phi = Array1::from_vec(vec![1.0, 2.0]);
        let _ = rls.update(&phi, 5.0);
        assert!(rls.n_updates > 0);

        rls.reset(200.0);
        assert_eq!(rls.n_updates, 0);
        assert_relative_eq!(rls.covariance[[0, 0]], 200.0);
    }
}
