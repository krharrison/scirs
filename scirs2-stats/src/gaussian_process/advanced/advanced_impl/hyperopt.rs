//! Hyperparameter optimisation for GP models with ARD kernels.

use super::kernels::ARDKernel;
use super::linalg::{cholesky_jitter, log_det_from_cholesky, solve_lower, solve_upper};
use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Hyperparameter optimiser for GP models.
///
/// Maximises the log marginal likelihood using Adam gradient ascent in log-space.
///
/// # Features
/// - Automatic Relevance Determination (ARD): per-dimension length scales
/// - Noise variance optimisation
/// - Multiple random restarts
#[derive(Debug, Clone)]
pub struct GPHyperparamOpt {
    pub max_iter: usize,
    pub fd_step: f64,
    pub learning_rate: f64,
    pub log_lower: f64,
    pub log_upper: f64,
    pub lml_history: Vec<f64>,
}

impl Default for GPHyperparamOpt {
    fn default() -> Self {
        Self {
            max_iter: 100,
            fd_step: 1e-4,
            learning_rate: 0.05,
            log_lower: -6.0,
            log_upper: 6.0,
            lml_history: Vec::new(),
        }
    }
}

impl GPHyperparamOpt {
    pub fn new(max_iter: usize) -> Self {
        Self { max_iter, ..Default::default() }
    }

    /// Compute log marginal likelihood for ARD RBF kernel.
    ///
    /// `log_params = [log(ls_1)..log(ls_d), log(sigma^2), log(sigma_noise^2)]`
    pub fn log_marginal_likelihood(
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        log_params: &[f64],
    ) -> StatsResult<f64> {
        let d = x_train.ncols();
        let n = x_train.nrows();
        if log_params.len() < d + 2 {
            return Err(StatsError::InvalidArgument(format!(
                "log_params length {} < d+2={}",
                log_params.len(), d + 2
            )));
        }
        let length_scales: Vec<f64> = log_params[..d].iter().map(|&p| p.exp()).collect();
        let signal_var = log_params[d].exp();
        let noise_var = log_params[d + 1].exp();

        let mut k = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let sq: f64 = (0..d)
                    .map(|dd| {
                        let diff = x_train[[i, dd]] - x_train[[j, dd]];
                        diff * diff / (length_scales[dd] * length_scales[dd])
                    })
                    .sum();
                k[[i, j]] = signal_var * (-0.5 * sq).exp();
            }
            k[[i, i]] += noise_var;
        }

        let l = cholesky_jitter(&k)?;
        let alpha_half = solve_lower(&l, y_train)?;
        let alpha = solve_upper(&l.t().to_owned(), &alpha_half)?;
        let data_fit = -0.5 * y_train.iter().zip(alpha.iter()).map(|(&y, &a)| y * a).sum::<f64>();
        let log_det = log_det_from_cholesky(&l);
        let norm = -0.5 * n as f64 * (2.0 * PI).ln();
        Ok(data_fit - 0.5 * log_det + norm)
    }

    fn numerical_gradient(
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        log_params: &[f64],
        fd_step: f64,
    ) -> StatsResult<Vec<f64>> {
        let np = log_params.len();
        let lml_base = Self::log_marginal_likelihood(x_train, y_train, log_params)?;
        let mut grad = vec![0.0; np];
        for i in 0..np {
            let mut params_plus = log_params.to_vec();
            params_plus[i] += fd_step;
            let lml_plus = Self::log_marginal_likelihood(x_train, y_train, &params_plus)
                .unwrap_or(f64::NEG_INFINITY);
            grad[i] = (lml_plus - lml_base) / fd_step;
        }
        Ok(grad)
    }

    /// Run Adam optimisation from `init_log_params`.
    pub fn optimize(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        init_log_params: &[f64],
    ) -> StatsResult<(Vec<f64>, f64)> {
        let np = init_log_params.len();
        let mut params = init_log_params.to_vec();
        let mut best_params = params.clone();
        let mut best_lml = f64::NEG_INFINITY;

        let beta1 = 0.9_f64;
        let beta2 = 0.999_f64;
        let eps = 1e-8_f64;
        let mut m = vec![0.0_f64; np];
        let mut v = vec![0.0_f64; np];

        self.lml_history.clear();

        for t in 1..=self.max_iter {
            let lml = Self::log_marginal_likelihood(x_train, y_train, &params)
                .unwrap_or(f64::NEG_INFINITY);
            if lml.is_finite() && lml > best_lml {
                best_lml = lml;
                best_params = params.clone();
            }
            self.lml_history.push(lml);

            let grad = Self::numerical_gradient(x_train, y_train, &params, self.fd_step)?;

            let t_f = t as f64;
            for i in 0..np {
                m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
                let m_hat = m[i] / (1.0 - beta1.powi(t as i32));
                let v_hat = v[i] / (1.0 - beta2.powi(t as i32));
                let lr = self.learning_rate / (t_f.sqrt() + 1.0);
                params[i] += lr * m_hat / (v_hat.sqrt() + eps);
                params[i] = params[i].clamp(self.log_lower, self.log_upper);
            }
        }

        Ok((best_params, best_lml))
    }

    /// Fit ARD GP kernel using multiple restarts.
    pub fn fit_ard_gp(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        n_restarts: usize,
    ) -> StatsResult<ARDKernel> {
        let d = x_train.ncols();
        let mut best_params = vec![0.0_f64; d + 2];
        let mut best_lml = f64::NEG_INFINITY;

        let seeds: Vec<u64> = (0..n_restarts.max(1)).map(|i| i as u64 * 12345 + 1).collect();
        for seed in seeds {
            let mut state = seed;
            let mut rng = || -> f64 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 11) as f64 / (1u64 << 53) as f64) * 4.0 - 2.0
            };
            let init: Vec<f64> = (0..(d + 2)).map(|_| rng()).collect();
            let (params, lml) = self.optimize(x_train, y_train, &init)?;
            if lml > best_lml {
                best_lml = lml;
                best_params = params;
            }
        }

        let length_scales: Vec<f64> = best_params[..d].iter().map(|&p| p.exp()).collect();
        let variance = best_params[d].exp();
        Ok(ARDKernel::new(length_scales, variance))
    }
}
