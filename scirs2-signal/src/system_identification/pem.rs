//! Prediction Error Method (PEM) for system identification
//!
//! General-purpose nonlinear optimization for minimizing the prediction
//! error criterion. Handles ARMAX-like models using Gauss-Newton iterations
//! with gradient computation via sensitivity equations.
//!
//! Model structure:
//!   A(q) y(t) = B(q) u(t-nk) + C(q) e(t)
//!
//! The prediction error is:
//!   epsilon(t, theta) = [A(q)/C(q)] y(t) - [B(q)/C(q)] u(t-nk)
//!
//! The PEM minimizes V(theta) = (1/N) sum epsilon(t, theta)^2

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};

use super::arx::solve_linear_system_regularized;
use super::types::{compute_fit_percentage, compute_information_criteria, PemConfig, SysIdResult};

/// Estimate a model using the Prediction Error Method
///
/// Uses a damped Gauss-Newton algorithm to minimize the sum of squared
/// prediction errors. The gradient is computed numerically via finite
/// differences of the prediction error w.r.t. each parameter.
///
/// # Arguments
/// * `y` - Output signal
/// * `u` - Input signal
/// * `config` - PEM configuration
///
/// # Returns
/// * `SysIdResult` with estimated parameters
///
/// # Example
/// ```rust
/// use scirs2_signal::system_identification::{pem_estimate, PemConfig};
/// use scirs2_core::ndarray::Array1;
///
/// let n = 300;
/// let mut u = Array1::<f64>::zeros(n);
/// let mut y = Array1::<f64>::zeros(n);
/// for i in 0..n { u[i] = ((i as f64) * 0.1).sin(); }
/// for i in 1..n { y[i] = 0.8 * y[i - 1] + 0.5 * u[i - 1]; }
///
/// let config = PemConfig { na: 1, nb: 1, nc: 0, nk: 1, ..Default::default() };
/// let result = pem_estimate(&y, &u, &config).expect("PEM failed");
/// assert!(result.fit_percentage > 80.0);
/// ```
pub fn pem_estimate(
    y: &Array1<f64>,
    u: &Array1<f64>,
    config: &PemConfig,
) -> SignalResult<SysIdResult> {
    let n = y.len();
    if n != u.len() {
        return Err(SignalError::DimensionMismatch(
            "Input and output must have the same length".into(),
        ));
    }

    let na = config.na;
    let nb = config.nb;
    let nc = config.nc;
    let nk = config.nk;
    let n_params = na + nb + 1 + nc;

    let start = na.max(nb + nk).max(nc);
    if n <= start + n_params {
        return Err(SignalError::ValueError(
            "Insufficient data for the specified PEM model orders".into(),
        ));
    }

    let data_len = n - start;

    // Initialize parameters using ARX estimate (ignoring C part)
    let mut theta = initialize_from_arx(y, u, na, nb, nk, nc, start)?;

    let epsilon_fd = 1e-7; // Finite difference step

    for _iteration in 0..config.max_iter {
        // Compute prediction errors at current theta
        let errors = compute_prediction_errors(y, u, &theta, na, nb, nc, nk, start)?;

        // Compute Jacobian via finite differences
        let mut jacobian = Array2::<f64>::zeros((data_len, n_params));

        for p in 0..n_params {
            let mut theta_plus = theta.clone();
            theta_plus[p] += epsilon_fd;
            let errors_plus = compute_prediction_errors(y, u, &theta_plus, na, nb, nc, nk, start)?;

            for t in 0..data_len {
                jacobian[[t, p]] = (errors_plus[t] - errors[t]) / epsilon_fd;
            }
        }

        // Gauss-Newton step: delta = -(J^T J)^{-1} J^T e
        let jt = jacobian.t();
        let jtj = jt.dot(&jacobian);
        let jte = jt.dot(&errors);

        let delta = match solve_linear_system_regularized(&jtj, &jte, 1e-6) {
            Ok(d) => d,
            Err(_) => break, // Convergence issue
        };

        // Check convergence before update
        let delta_norm = delta.mapv(|x| x * x).sum().sqrt();
        let theta_norm = theta.mapv(|x| x * x).sum().sqrt().max(1.0);

        if delta_norm < config.tolerance * theta_norm {
            break;
        }

        // Damped update: theta = theta - damping * delta
        theta = &theta - &(&delta * config.damping);
    }

    // Extract final coefficients
    let a_coeffs = theta.slice(scirs2_core::ndarray::s![0..na]).to_owned();
    let b_coeffs = theta
        .slice(scirs2_core::ndarray::s![na..na + nb + 1])
        .to_owned();
    let c_coeffs = if nc > 0 {
        Some(
            theta
                .slice(scirs2_core::ndarray::s![na + nb + 1..n_params])
                .to_owned(),
        )
    } else {
        None
    };

    // Compute final residuals
    let final_errors = compute_prediction_errors(y, u, &theta, na, nb, nc, nk, start)?;

    let mut residuals = Array1::<f64>::zeros(n);
    for i in 0..data_len {
        residuals[i + start] = final_errors[i];
    }

    let noise_var = if data_len > n_params {
        final_errors.mapv(|e| e * e).sum() / (data_len - n_params) as f64
    } else {
        final_errors.mapv(|e| e * e).sum() / data_len.max(1) as f64
    };

    // Compute fit using one-step-ahead prediction
    let y_sub = y.slice(scirs2_core::ndarray::s![start..]).to_owned();
    let y_hat = &y_sub - &final_errors;
    let fit_percentage = compute_fit_percentage(&y_sub, &y_hat);
    let (aic, bic, fpe) = compute_information_criteria(noise_var, data_len, n_params);

    Ok(SysIdResult {
        a_coeffs,
        b_coeffs,
        c_coeffs,
        noise_variance: noise_var,
        fit_percentage,
        residuals,
        aic,
        bic,
        fpe,
        n_params,
    })
}

/// Initialize PEM parameters from an ARX estimate
fn initialize_from_arx(
    y: &Array1<f64>,
    u: &Array1<f64>,
    na: usize,
    nb: usize,
    nk: usize,
    nc: usize,
    start: usize,
) -> SignalResult<Array1<f64>> {
    let n = y.len();
    let n_arx_params = na + nb + 1;
    let data_len = n - start;

    // Build ARX regression matrix
    let mut phi = Array2::<f64>::zeros((data_len, n_arx_params));
    let mut y_target = Array1::<f64>::zeros(data_len);

    for i in 0..data_len {
        let t = i + start;
        y_target[i] = y[t];

        for j in 0..na {
            phi[[i, j]] = -y[t - 1 - j];
        }
        for j in 0..=nb {
            let idx = t as isize - nk as isize - j as isize;
            if idx >= 0 && (idx as usize) < n {
                phi[[i, na + j]] = u[idx as usize];
            }
        }
    }

    let phi_t = phi.t();
    let ata = phi_t.dot(&phi);
    let atb = phi_t.dot(&y_target);
    let arx_theta = solve_linear_system_regularized(&ata, &atb, 1e-10)?;

    // Concatenate ARX params with zero C params
    let n_total = na + nb + 1 + nc;
    let mut theta = Array1::<f64>::zeros(n_total);
    for i in 0..n_arx_params {
        theta[i] = arx_theta[i];
    }
    // C coefficients initialized to zero

    Ok(theta)
}

/// Compute prediction errors for given parameters
///
/// For the model A(q)y(t) = B(q)u(t-nk) + C(q)e(t):
///   e(t) = [A(q)/C(q)] y(t) - [B(q)/C(q)] u(t-nk)
///
/// Implemented via the recurrence:
///   e(t) = y(t) + a_1*y(t-1) + ... + a_na*y(t-na)
///          - b_0*u(t-nk) - ... - b_nb*u(t-nk-nb)
///          - c_1*e(t-1) - ... - c_nc*e(t-nc)
fn compute_prediction_errors(
    y: &Array1<f64>,
    u: &Array1<f64>,
    theta: &Array1<f64>,
    na: usize,
    nb: usize,
    nc: usize,
    nk: usize,
    start: usize,
) -> SignalResult<Array1<f64>> {
    let n = y.len();
    let data_len = n - start;

    let a = theta.slice(scirs2_core::ndarray::s![0..na]);
    let b = theta.slice(scirs2_core::ndarray::s![na..na + nb + 1]);
    let c = if nc > 0 {
        Some(theta.slice(scirs2_core::ndarray::s![na + nb + 1..na + nb + 1 + nc]))
    } else {
        None
    };

    // Compute errors using the recurrence
    let mut e_full = Array1::<f64>::zeros(n);

    for t in start..n {
        // A(q) y(t) part: y(t) + a_1*y(t-1) + ...
        let mut val = y[t];
        for j in 0..na {
            val += a[j] * y[t - 1 - j];
        }

        // -B(q) u(t-nk) part
        for j in 0..=nb {
            let idx = t as isize - nk as isize - j as isize;
            if idx >= 0 && (idx as usize) < n {
                val -= b[j] * u[idx as usize];
            }
        }

        // -C(q) e(t) part (MA part)
        if let Some(ref c_arr) = c {
            for j in 0..nc {
                let idx = t as isize - 1 - j as isize;
                if idx >= 0 {
                    val -= c_arr[j] * e_full[idx as usize];
                }
            }
        }

        e_full[t] = val;
    }

    Ok(e_full.slice(scirs2_core::ndarray::s![start..]).to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pem_arx_like() {
        // PEM with nc=0 should give results similar to ARX
        let n = 500;
        let mut u = Array1::<f64>::zeros(n);
        let mut y = Array1::<f64>::zeros(n);

        for i in 0..n {
            u[i] = ((i as f64) * 0.1).sin();
        }
        for i in 1..n {
            y[i] = 0.8 * y[i - 1] + 0.5 * u[i - 1];
        }

        let config = PemConfig {
            na: 1,
            nb: 1,
            nc: 0,
            nk: 1,
            max_iter: 30,
            tolerance: 1e-8,
            damping: 0.5,
        };
        let result = pem_estimate(&y, &u, &config).expect("PEM failed");

        // Should recover close to true parameters
        assert_relative_eq!(result.a_coeffs[0], -0.8, epsilon = 0.1);
        assert_relative_eq!(result.b_coeffs[0], 0.5, epsilon = 0.1);
        assert!(
            result.fit_percentage > 90.0,
            "Fit = {}",
            result.fit_percentage
        );
    }

    #[test]
    fn test_pem_with_c_polynomial() {
        // Test PEM with nonzero nc (ARMAX-like)
        let n = 500;
        let mut u = Array1::<f64>::zeros(n);
        let mut y = Array1::<f64>::zeros(n);

        for i in 0..n {
            u[i] = ((i as f64) * 0.07).sin();
        }
        for i in 1..n {
            y[i] = 0.7 * y[i - 1] + 0.4 * u[i - 1];
        }

        let config = PemConfig {
            na: 1,
            nb: 1,
            nc: 1,
            nk: 1,
            max_iter: 50,
            tolerance: 1e-8,
            damping: 0.3,
        };
        let result = pem_estimate(&y, &u, &config).expect("PEM failed");

        assert!(
            result.fit_percentage > 80.0,
            "Fit = {}",
            result.fit_percentage
        );
        assert!(result.noise_variance < 0.1);
    }

    #[test]
    fn test_pem_dimension_mismatch() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let u = Array1::from_vec(vec![1.0, 2.0]);
        let config = PemConfig::default();
        assert!(pem_estimate(&y, &u, &config).is_err());
    }

    #[test]
    fn test_pem_information_criteria() {
        let n = 300;
        let mut u = Array1::<f64>::zeros(n);
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            u[i] = ((i as f64) * 0.1).sin();
        }
        for i in 1..n {
            y[i] = 0.8 * y[i - 1] + 0.5 * u[i - 1];
        }

        let config = PemConfig {
            na: 1,
            nb: 1,
            nc: 0,
            nk: 1,
            ..Default::default()
        };
        let result = pem_estimate(&y, &u, &config).expect("PEM failed");

        assert!(result.aic.is_finite());
        assert!(result.bic.is_finite());
        assert!(result.n_params == 3); // 1 AR + 2 B
    }
}
