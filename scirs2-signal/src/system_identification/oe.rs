//! Output-Error (OE) model estimation
//!
//! Model structure:
//!   y(t) = [B(q)/F(q)] u(t-nk) + e(t)
//! where:
//!   B(q) = b_0 + b_1 q^{-1} + ... + b_nb q^{-nb}
//!   F(q) = 1 + f_1 q^{-1} + ... + f_nf q^{-nf}
//!
//! The noise enters directly on the output (no noise model shaping).
//! Estimation uses iterative Gauss-Newton optimization since the
//! F polynomial makes the problem nonlinear.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};

use super::arx::solve_linear_system_regularized;
use super::types::{compute_fit_percentage, compute_information_criteria, OeConfig, SysIdResult};

/// Estimate an Output-Error model from input-output data
///
/// Uses a Gauss-Newton iterative approach:
/// 1. Initialize B from ARX, F from zeros
/// 2. Compute model output w(t) = B(q)/F(q) * u(t-nk)
/// 3. Linearize around current parameters and solve for updates
/// 4. Repeat until convergence
///
/// # Arguments
/// * `y` - Output signal
/// * `u` - Input signal
/// * `config` - OE configuration
///
/// # Returns
/// * `SysIdResult` where a_coeffs = F coefficients, b_coeffs = B coefficients
pub fn oe_estimate(
    y: &Array1<f64>,
    u: &Array1<f64>,
    config: &OeConfig,
) -> SignalResult<SysIdResult> {
    let n = y.len();
    if n != u.len() {
        return Err(SignalError::DimensionMismatch(
            "Input and output must have the same length".into(),
        ));
    }

    let nb = config.nb;
    let nf = config.nf;
    let nk = config.nk;
    let n_params = nb + 1 + nf;

    let start = (nf).max(nb + nk);
    if n <= start + n_params {
        return Err(SignalError::ValueError(
            "Insufficient data for the specified OE model orders".into(),
        ));
    }

    let data_len = n - start;

    // Initialize parameters: B coefficients from simple regression, F = 0
    let mut b_params = Array1::<f64>::zeros(nb + 1);
    let mut f_params = Array1::<f64>::zeros(nf);

    // Initial B estimate: simple input-output regression
    {
        let mut phi_init = Array2::<f64>::zeros((data_len, nb + 1));
        let mut y_init = Array1::<f64>::zeros(data_len);
        for i in 0..data_len {
            let t = i + start;
            y_init[i] = y[t];
            for j in 0..=nb {
                let idx = t as isize - nk as isize - j as isize;
                if idx >= 0 && (idx as usize) < n {
                    phi_init[[i, j]] = u[idx as usize];
                }
            }
        }
        let phi_t = phi_init.t();
        let ata = phi_t.dot(&phi_init);
        let atb = phi_t.dot(&y_init);
        if let Ok(b_init) = solve_linear_system_regularized(&ata, &atb, 1e-10) {
            b_params.assign(&b_init);
        }
    }

    // Gauss-Newton iterations
    for _iteration in 0..config.max_iter {
        // Compute model output w(t) = B(q)/F(q) * u(t-nk)
        // Equivalently: w(t) + f_1*w(t-1) + ... + f_nf*w(t-nf) = B(q)*u(t-nk)
        let mut w = Array1::<f64>::zeros(n);
        for t in start..n {
            // B(q)*u(t-nk)
            let mut bu = 0.0;
            for j in 0..=nb {
                let idx = t as isize - nk as isize - j as isize;
                if idx >= 0 && (idx as usize) < n {
                    bu += b_params[j] * u[idx as usize];
                }
            }
            // -F(q)*w(t) part (shifted: f_1*w(t-1) + ...)
            let mut fw = 0.0;
            for j in 0..nf {
                let idx = t as isize - 1 - j as isize;
                if idx >= 0 {
                    fw += f_params[j] * w[idx as usize];
                }
            }
            w[t] = bu - fw;
        }

        // Compute error
        let mut error = Array1::<f64>::zeros(data_len);
        for i in 0..data_len {
            error[i] = y[i + start] - w[i + start];
        }

        // Compute Jacobian: partial derivatives of w(t) w.r.t. each parameter
        let mut jacobian = Array2::<f64>::zeros((data_len, n_params));

        // Derivative w.r.t. b_j: dw/db_j satisfies the same recursion as w
        // but driven by u(t-nk-j) instead
        for j in 0..=nb {
            let mut dw_dbj = Array1::<f64>::zeros(n);
            for t in start..n {
                let idx = t as isize - nk as isize - j as isize;
                let driver = if idx >= 0 && (idx as usize) < n {
                    u[idx as usize]
                } else {
                    0.0
                };
                let mut fw_part = 0.0;
                for k in 0..nf {
                    let idx2 = t as isize - 1 - k as isize;
                    if idx2 >= 0 {
                        fw_part += f_params[k] * dw_dbj[idx2 as usize];
                    }
                }
                dw_dbj[t] = driver - fw_part;
            }
            for i in 0..data_len {
                jacobian[[i, j]] = dw_dbj[i + start];
            }
        }

        // Derivative w.r.t. f_j: dw/df_j = -w(t-1-j) - F(q)*dw/df_j(past)
        for j in 0..nf {
            let mut dw_dfj = Array1::<f64>::zeros(n);
            for t in start..n {
                let w_past_idx = t as isize - 1 - j as isize;
                let w_past = if w_past_idx >= 0 {
                    w[w_past_idx as usize]
                } else {
                    0.0
                };
                let mut fw_part = 0.0;
                for k in 0..nf {
                    let idx2 = t as isize - 1 - k as isize;
                    if idx2 >= 0 {
                        fw_part += f_params[k] * dw_dfj[idx2 as usize];
                    }
                }
                dw_dfj[t] = -w_past - fw_part;
            }
            for i in 0..data_len {
                jacobian[[i, nb + 1 + j]] = dw_dfj[i + start];
            }
        }

        // Gauss-Newton step: delta = (J^T J)^{-1} J^T error
        let jt = jacobian.t();
        let jtj = jt.dot(&jacobian);
        let jte = jt.dot(&error);

        let delta = match solve_linear_system_regularized(&jtj, &jte, 1e-8) {
            Ok(d) => d,
            Err(_) => break, // Convergence issue, stop iterating
        };

        // Check convergence
        let delta_norm = delta.mapv(|x| x * x).sum().sqrt();
        let param_norm = b_params.mapv(|x| x * x).sum() + f_params.mapv(|x| x * x).sum();
        let param_norm_sqrt = param_norm.sqrt().max(1.0);

        // Update parameters with damping
        for j in 0..=nb {
            b_params[j] += config.tolerance.max(0.1).min(1.0) * delta[j];
        }
        for j in 0..nf {
            f_params[j] += config.tolerance.max(0.1).min(1.0) * delta[nb + 1 + j];
        }

        if delta_norm < config.tolerance * param_norm_sqrt {
            break;
        }
    }

    // Final prediction
    let mut w_final = Array1::<f64>::zeros(n);
    for t in start..n {
        let mut bu = 0.0;
        for j in 0..=nb {
            let idx = t as isize - nk as isize - j as isize;
            if idx >= 0 && (idx as usize) < n {
                bu += b_params[j] * u[idx as usize];
            }
        }
        let mut fw = 0.0;
        for j in 0..nf {
            let idx = t as isize - 1 - j as isize;
            if idx >= 0 {
                fw += f_params[j] * w_final[idx as usize];
            }
        }
        w_final[t] = bu - fw;
    }

    let mut residuals = Array1::<f64>::zeros(n);
    for t in start..n {
        residuals[t] = y[t] - w_final[t];
    }

    let res_sub = residuals
        .slice(scirs2_core::ndarray::s![start..])
        .to_owned();
    let noise_var = if data_len > n_params {
        res_sub.mapv(|r| r * r).sum() / (data_len - n_params) as f64
    } else {
        res_sub.mapv(|r| r * r).sum() / data_len.max(1) as f64
    };

    let y_sub = y.slice(scirs2_core::ndarray::s![start..]).to_owned();
    let w_sub = w_final.slice(scirs2_core::ndarray::s![start..]).to_owned();
    let fit_percentage = compute_fit_percentage(&y_sub, &w_sub);
    let (aic, bic, fpe) = compute_information_criteria(noise_var, data_len, n_params);

    Ok(SysIdResult {
        a_coeffs: f_params, // F polynomial = denominator
        b_coeffs: b_params, // B polynomial = numerator
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oe_simple_fir() {
        // OE with nf=0 is just FIR: y[k] = b_0*u[k-1] + b_1*u[k-2]
        let n = 500;
        let mut u = Array1::<f64>::zeros(n);
        let mut y = Array1::<f64>::zeros(n);

        for i in 0..n {
            u[i] = ((i as f64) * 0.1).sin();
        }
        for i in 2..n {
            y[i] = 0.6 * u[i - 1] + 0.3 * u[i - 2];
        }

        let config = OeConfig {
            nb: 1,
            nf: 0,
            nk: 1,
            max_iter: 30,
            tolerance: 1e-6,
        };
        let result = oe_estimate(&y, &u, &config).expect("OE failed");
        assert!(
            result.fit_percentage > 80.0,
            "Fit = {}",
            result.fit_percentage
        );
    }

    #[test]
    fn test_oe_first_order() {
        // OE model: y[k] = B(q)/F(q) * u[k-1] where B=0.5, F=1+0.3*q^{-1}
        // This means: w[k] = -0.3*w[k-1] + 0.5*u[k-1], y[k] = w[k]
        let n = 500;
        let mut u = Array1::<f64>::zeros(n);
        let mut y = Array1::<f64>::zeros(n);

        for i in 0..n {
            u[i] = ((i as f64) * 0.08).sin();
        }
        for i in 1..n {
            y[i] = -0.3 * y[i - 1] + 0.5 * u[i - 1];
        }

        let config = OeConfig {
            nb: 0,
            nf: 1,
            nk: 1,
            max_iter: 50,
            tolerance: 1e-8,
        };
        let result = oe_estimate(&y, &u, &config).expect("OE failed");
        assert!(
            result.fit_percentage > 70.0,
            "Fit = {}",
            result.fit_percentage
        );
    }

    #[test]
    fn test_oe_dimension_mismatch() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let u = Array1::from_vec(vec![1.0, 2.0]);
        let config = OeConfig::default();
        assert!(oe_estimate(&y, &u, &config).is_err());
    }
}
