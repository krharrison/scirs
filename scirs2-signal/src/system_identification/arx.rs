//! ARX (AutoRegressive with eXogenous input) model estimation
//!
//! Model structure:
//!   A(q) y(t) = B(q) u(t-nk) + e(t)
//! where:
//!   A(q) = 1 + a_1 q^{-1} + ... + a_na q^{-na}
//!   B(q) = b_0 + b_1 q^{-1} + ... + b_nb q^{-nb}
//!
//! Estimation uses ordinary least squares (OLS) since the ARX model
//! is linear in its parameters.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};

use super::types::{compute_fit_percentage, compute_information_criteria, ArxConfig, SysIdResult};

/// Estimate an ARX model from input-output data using least squares
///
/// # Arguments
/// * `y` - Output signal
/// * `u` - Input signal
/// * `config` - ARX configuration (na, nb, nk)
///
/// # Returns
/// * `SysIdResult` with estimated parameters and fit metrics
///
/// # Example
/// ```rust
/// use scirs2_signal::system_identification::{arx_estimate, ArxConfig};
/// use scirs2_core::ndarray::Array1;
///
/// let n = 200;
/// let mut u = Array1::<f64>::zeros(n);
/// let mut y = Array1::<f64>::zeros(n);
/// for i in 0..n { u[i] = ((i as f64) * 0.1).sin(); }
/// for i in 1..n { y[i] = 0.8 * y[i - 1] + 0.5 * u[i - 1]; }
///
/// let config = ArxConfig { na: 1, nb: 1, nk: 1 };
/// let result = arx_estimate(&y, &u, &config).expect("ARX failed");
/// assert!(result.fit_percentage > 90.0);
/// ```
pub fn arx_estimate(
    y: &Array1<f64>,
    u: &Array1<f64>,
    config: &ArxConfig,
) -> SignalResult<SysIdResult> {
    let n = y.len();
    if n != u.len() {
        return Err(SignalError::DimensionMismatch(
            "Input and output must have the same length".into(),
        ));
    }

    let na = config.na;
    let nb = config.nb;
    let nk = config.nk;

    // Total number of parameters: na (AR) + nb + 1 (B polynomial order nb means nb+1 coeffs)
    let n_params = na + nb + 1;

    // We need at least max(na, nb + nk) initial samples
    let start = na.max(nb + nk);
    if n <= start + n_params {
        return Err(SignalError::ValueError(
            "Insufficient data for the specified model orders".into(),
        ));
    }

    let data_len = n - start;

    // Build regression matrix Phi and target vector Y
    let mut phi = Array2::<f64>::zeros((data_len, n_params));
    let mut y_target = Array1::<f64>::zeros(data_len);

    for i in 0..data_len {
        let t = i + start;
        y_target[i] = y[t];

        // AR part: -y(t-1), -y(t-2), ..., -y(t-na)
        for j in 0..na {
            phi[[i, j]] = -y[t - 1 - j];
        }

        // B part: u(t-nk), u(t-nk-1), ..., u(t-nk-nb)
        for j in 0..=nb {
            let idx = t as isize - nk as isize - j as isize;
            if idx >= 0 && (idx as usize) < n {
                phi[[i, na + j]] = u[idx as usize];
            }
        }
    }

    // Solve least squares: theta = (Phi^T Phi)^{-1} Phi^T y
    let phi_t = phi.t();
    let ata = phi_t.dot(&phi);
    let atb = phi_t.dot(&y_target);

    let theta = solve_linear_system_regularized(&ata, &atb, 1e-12)?;

    // Extract coefficients
    let a_coeffs = theta.slice(scirs2_core::ndarray::s![0..na]).to_owned();
    let b_coeffs = theta
        .slice(scirs2_core::ndarray::s![na..na + nb + 1])
        .to_owned();

    // Compute prediction and residuals
    let y_pred = phi.dot(&theta);
    let residuals_short = &y_target - &y_pred;

    // Full-length residuals
    let mut residuals = Array1::<f64>::zeros(n);
    for i in 0..data_len {
        residuals[i + start] = residuals_short[i];
    }

    // Noise variance
    let noise_var = if data_len > n_params {
        residuals_short.mapv(|r| r * r).sum() / (data_len - n_params) as f64
    } else {
        residuals_short.mapv(|r| r * r).sum() / data_len.max(1) as f64
    };

    // Reconstruct full-length y_hat for fit calculation
    let mut y_hat = Array1::<f64>::zeros(n);
    for i in 0..data_len {
        y_hat[i + start] = y_pred[i];
    }
    // Copy initial values as-is for the skipped part
    for i in 0..start {
        y_hat[i] = y[i];
    }

    let fit_percentage = compute_fit_percentage(&y_target, &y_pred);
    let (aic, bic, fpe) = compute_information_criteria(noise_var, data_len, n_params);

    Ok(SysIdResult {
        a_coeffs,
        b_coeffs,
        c_coeffs: None,
        noise_variance: noise_var,
        fit_percentage,
        residuals,
        aic,
        bic,
        fpe,
        n_params,
    })
}

/// Solve linear system (A^T A) x = A^T b with Tikhonov regularization
pub(crate) fn solve_linear_system_regularized(
    ata: &Array2<f64>,
    atb: &Array1<f64>,
    regularization: f64,
) -> SignalResult<Array1<f64>> {
    let n = ata.nrows();
    if n != ata.ncols() || n != atb.len() {
        return Err(SignalError::DimensionMismatch(
            "Dimension mismatch in linear system".into(),
        ));
    }

    // Add regularization: (A^T A + lambda * I) x = A^T b
    let mut ata_reg = ata.clone();
    for i in 0..n {
        ata_reg[[i, i]] += regularization;
    }

    match scirs2_linalg::solve(&ata_reg.view(), &atb.view(), None) {
        Ok(x) => Ok(x),
        Err(_) => Err(SignalError::ComputationError(
            "Failed to solve normal equations (matrix may be singular)".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_arx_first_order_system() {
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

        let config = ArxConfig {
            na: 1,
            nb: 1,
            nk: 1,
        };
        let result = arx_estimate(&y, &u, &config).expect("ARX estimation failed");

        // Check that estimated coefficients are close to true values
        assert_relative_eq!(result.a_coeffs[0], -0.8, epsilon = 0.05);
        assert_relative_eq!(result.b_coeffs[0], 0.5, epsilon = 0.05);
        assert!(
            result.fit_percentage > 95.0,
            "Fit = {}",
            result.fit_percentage
        );
        assert!(result.noise_variance < 0.01);
    }

    #[test]
    fn test_arx_second_order_system() {
        // True system: y[k] = 0.5*y[k-1] - 0.3*y[k-2] + 0.4*u[k-1] + 0.2*u[k-2]
        let n = 1000;
        let mut u = Array1::<f64>::zeros(n);
        let mut y = Array1::<f64>::zeros(n);

        for i in 0..n {
            u[i] = ((i as f64) * 0.05).sin() + 0.5 * ((i as f64) * 0.13).cos();
        }
        for i in 2..n {
            y[i] = 0.5 * y[i - 1] - 0.3 * y[i - 2] + 0.4 * u[i - 1] + 0.2 * u[i - 2];
        }

        let config = ArxConfig {
            na: 2,
            nb: 2,
            nk: 1,
        };
        let result = arx_estimate(&y, &u, &config).expect("ARX estimation failed");

        assert_relative_eq!(result.a_coeffs[0], -0.5, epsilon = 0.05);
        assert_relative_eq!(result.a_coeffs[1], 0.3, epsilon = 0.05);
        assert!(
            result.fit_percentage > 95.0,
            "Fit = {}",
            result.fit_percentage
        );
    }

    #[test]
    fn test_arx_dimension_mismatch() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let u = Array1::from_vec(vec![1.0, 2.0]);
        let config = ArxConfig {
            na: 1,
            nb: 1,
            nk: 1,
        };
        assert!(arx_estimate(&y, &u, &config).is_err());
    }

    #[test]
    fn test_arx_insufficient_data() {
        let y = Array1::from_vec(vec![1.0, 2.0]);
        let u = Array1::from_vec(vec![1.0, 2.0]);
        let config = ArxConfig {
            na: 2,
            nb: 2,
            nk: 1,
        };
        assert!(arx_estimate(&y, &u, &config).is_err());
    }

    #[test]
    fn test_arx_with_delay() {
        // True system: y[k] = 0.7*y[k-1] + 0.3*u[k-2] (delay nk=2)
        let n = 500;
        let mut u = Array1::<f64>::zeros(n);
        let mut y = Array1::<f64>::zeros(n);

        for i in 0..n {
            u[i] = ((i as f64) * 0.08).sin();
        }
        for i in 2..n {
            y[i] = 0.7 * y[i - 1] + 0.3 * u[i - 2];
        }

        let config = ArxConfig {
            na: 1,
            nb: 0,
            nk: 2,
        };
        let result = arx_estimate(&y, &u, &config).expect("ARX with delay failed");

        assert_relative_eq!(result.a_coeffs[0], -0.7, epsilon = 0.05);
        assert_relative_eq!(result.b_coeffs[0], 0.3, epsilon = 0.05);
        assert!(result.fit_percentage > 95.0);
    }

    #[test]
    fn test_arx_information_criteria() {
        let n = 300;
        let mut u = Array1::<f64>::zeros(n);
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            u[i] = ((i as f64) * 0.1).sin();
        }
        for i in 1..n {
            y[i] = 0.8 * y[i - 1] + 0.5 * u[i - 1];
        }

        let config = ArxConfig {
            na: 1,
            nb: 1,
            nk: 1,
        };
        let result = arx_estimate(&y, &u, &config).expect("ARX failed");

        assert!(result.aic.is_finite());
        assert!(result.bic.is_finite());
        assert!(result.fpe.is_finite());
        assert!(result.fpe >= 0.0);
    }
}
