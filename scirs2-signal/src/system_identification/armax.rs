//! ARMAX (AutoRegressive Moving Average with eXogenous input) model estimation
//!
//! Model structure:
//!   A(q) y(t) = B(q) u(t-nk) + C(q) e(t)
//! where:
//!   A(q) = 1 + a_1 q^{-1} + ... + a_na q^{-na}
//!   B(q) = b_0 + b_1 q^{-1} + ... + b_nb q^{-nb}
//!   C(q) = 1 + c_1 q^{-1} + ... + c_nc q^{-nc}
//!
//! Estimation uses iterative (extended) least squares since the MA part
//! makes the problem nonlinear in the parameters.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};

use super::arx::solve_linear_system_regularized;
use super::types::{
    compute_fit_percentage, compute_information_criteria, ArmaxConfig, SysIdResult,
};

/// Estimate an ARMAX model from input-output data using iterative least squares
///
/// The algorithm:
/// 1. Start with ARX estimate (C(q) = 1) to get initial A, B
/// 2. Compute residuals e(t)
/// 3. Re-estimate with the residuals as additional regressors for C
/// 4. Repeat until convergence
///
/// # Arguments
/// * `y` - Output signal
/// * `u` - Input signal
/// * `config` - ARMAX configuration
///
/// # Returns
/// * `SysIdResult` with estimated A, B, C parameters
pub fn armax_estimate(
    y: &Array1<f64>,
    u: &Array1<f64>,
    config: &ArmaxConfig,
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
            "Insufficient data for the specified ARMAX model orders".into(),
        ));
    }

    let data_len = n - start;

    // Initialize residuals to zero (first iteration = ARX-like)
    let mut e_hat = Array1::<f64>::zeros(n);

    let mut theta = Array1::<f64>::zeros(n_params);
    let mut prev_theta = Array1::<f64>::zeros(n_params);

    for iteration in 0..config.max_iter {
        // Build regression matrix with AR, B, and MA columns
        let mut phi = Array2::<f64>::zeros((data_len, n_params));
        let mut y_target = Array1::<f64>::zeros(data_len);

        for i in 0..data_len {
            let t = i + start;
            y_target[i] = y[t];

            // AR part: -y(t-1), ..., -y(t-na)
            for j in 0..na {
                phi[[i, j]] = -y[t - 1 - j];
            }

            // B part: u(t-nk), ..., u(t-nk-nb)
            for j in 0..=nb {
                let idx = t as isize - nk as isize - j as isize;
                if idx >= 0 && (idx as usize) < n {
                    phi[[i, na + j]] = u[idx as usize];
                }
            }

            // MA (C) part: e(t-1), ..., e(t-nc)
            for j in 0..nc {
                let idx = t as isize - 1 - j as isize;
                if idx >= 0 {
                    phi[[i, na + nb + 1 + j]] = e_hat[idx as usize];
                }
            }
        }

        // Solve least squares
        let phi_t = phi.t();
        let ata = phi_t.dot(&phi);
        let atb = phi_t.dot(&y_target);
        theta = solve_linear_system_regularized(&ata, &atb, 1e-10)?;

        // Update residuals
        let y_pred = phi.dot(&theta);
        let residuals_short = &y_target - &y_pred;
        for i in 0..data_len {
            e_hat[i + start] = residuals_short[i];
        }

        // Check convergence
        if iteration > 0 {
            let param_change: f64 = (&theta - &prev_theta).mapv(|x| x * x).sum();
            let param_norm: f64 = theta.mapv(|x| x * x).sum();
            if param_change < config.tolerance * config.tolerance * param_norm.max(1.0) {
                break;
            }
        }

        prev_theta.assign(&theta);
    }

    // Extract coefficients
    let a_coeffs = theta.slice(scirs2_core::ndarray::s![0..na]).to_owned();
    let b_coeffs = theta
        .slice(scirs2_core::ndarray::s![na..na + nb + 1])
        .to_owned();
    let c_coeffs = theta
        .slice(scirs2_core::ndarray::s![na + nb + 1..n_params])
        .to_owned();

    // Compute final residuals
    let mut residuals = Array1::<f64>::zeros(n);
    residuals
        .slice_mut(scirs2_core::ndarray::s![start..])
        .assign(&e_hat.slice(scirs2_core::ndarray::s![start..]));

    let noise_var = if data_len > n_params {
        e_hat
            .slice(scirs2_core::ndarray::s![start..])
            .mapv(|r| r * r)
            .sum()
            / (data_len - n_params) as f64
    } else {
        e_hat
            .slice(scirs2_core::ndarray::s![start..])
            .mapv(|r| r * r)
            .sum()
            / data_len.max(1) as f64
    };

    // Compute y_hat for fit calculation
    let y_sub = y.slice(scirs2_core::ndarray::s![start..]).to_owned();
    let y_hat_sub = &y_sub - &e_hat.slice(scirs2_core::ndarray::s![start..]);
    let fit_percentage = compute_fit_percentage(&y_sub, &y_hat_sub);
    let (aic, bic, fpe) = compute_information_criteria(noise_var, data_len, n_params);

    Ok(SysIdResult {
        a_coeffs,
        b_coeffs,
        c_coeffs: Some(c_coeffs),
        noise_variance: noise_var,
        fit_percentage,
        residuals,
        aic,
        bic,
        fpe,
        n_params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_armax_no_ma_part() {
        // If nc=0, ARMAX should degenerate to ARX
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

        let config = ArmaxConfig {
            na: 1,
            nb: 1,
            nc: 0,
            nk: 1,
            max_iter: 20,
            tolerance: 1e-6,
        };
        let result = armax_estimate(&y, &u, &config).expect("ARMAX failed");

        assert!(
            result.fit_percentage > 90.0,
            "Fit = {}",
            result.fit_percentage
        );
        assert!(result.noise_variance < 0.01);
    }

    #[test]
    fn test_armax_with_ma() {
        // True system: y[k] = 0.7*y[k-1] + 0.3*u[k-1] + e[k] + 0.4*e[k-1]
        // We can't generate exact noise here but can test convergence
        let n = 500;
        let mut u = Array1::<f64>::zeros(n);
        let mut y = Array1::<f64>::zeros(n);

        for i in 0..n {
            u[i] = ((i as f64) * 0.07).sin();
        }
        for i in 1..n {
            y[i] = 0.7 * y[i - 1] + 0.3 * u[i - 1];
        }

        let config = ArmaxConfig {
            na: 1,
            nb: 1,
            nc: 1,
            nk: 1,
            max_iter: 30,
            tolerance: 1e-8,
        };
        let result = armax_estimate(&y, &u, &config).expect("ARMAX failed");

        // MA coefficient should be near 0 since there's no noise
        assert!(
            result.fit_percentage > 85.0,
            "Fit = {}",
            result.fit_percentage
        );
        if let Some(ref c) = result.c_coeffs {
            assert!(c[0].abs() < 0.2, "c_1 = {} should be small", c[0]);
        }
    }

    #[test]
    fn test_armax_dimension_mismatch() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let u = Array1::from_vec(vec![1.0, 2.0]);
        let config = ArmaxConfig::default();
        assert!(armax_estimate(&y, &u, &config).is_err());
    }

    #[test]
    fn test_armax_information_criteria() {
        let n = 300;
        let mut u = Array1::<f64>::zeros(n);
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            u[i] = ((i as f64) * 0.1).sin();
        }
        for i in 1..n {
            y[i] = 0.8 * y[i - 1] + 0.5 * u[i - 1];
        }

        let config = ArmaxConfig {
            na: 1,
            nb: 1,
            nc: 1,
            nk: 1,
            ..Default::default()
        };
        let result = armax_estimate(&y, &u, &config).expect("ARMAX failed");

        assert!(result.aic.is_finite());
        assert!(result.bic.is_finite());
    }
}
