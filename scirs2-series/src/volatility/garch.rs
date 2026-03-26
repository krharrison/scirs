//! GARCH (Generalised ARCH) volatility models
//!
//! This module implements the GARCH(p,q) model introduced by Bollerslev (1986).
//! GARCH extends ARCH by also including lagged conditional variances in the
//! variance equation, which provides a parsimonious representation of
//! long-memory volatility dynamics.
//!
//! # Model Specification
//!
//! The GARCH(p,q) conditional variance equation is:
//!
//! ```text
//! σ²ₜ = ω + Σᵢ₌₁ᵖ αᵢ ε²ₜ₋ᵢ + Σⱼ₌₁ᵍ βⱼ σ²ₜ₋ⱼ
//! ```
//!
//! where:
//! - `ω > 0`     — unconditional variance intercept
//! - `αᵢ ≥ 0`  — ARCH (shock) coefficients
//! - `βⱼ ≥ 0`  — GARCH (persistence) coefficients
//! - For weak stationarity: Σαᵢ + Σβⱼ < 1
//!
//! # Examples
//!
//! ```rust
//! use scirs2_series::volatility::garch::{
//!     GARCHModel, fit_garch, garch_variance_forecast, garch_log_likelihood
//! };
//! use scirs2_core::ndarray::Array1;
//!
//! let returns: Array1<f64> = Array1::from(vec![
//!     0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.018, 0.009,
//!     -0.003, 0.007, 0.025, -0.014, 0.008, -0.006, 0.011, -0.019,
//!     0.022, 0.003, -0.011, 0.017,
//! ]);
//! let model = fit_garch(&returns, 1, 1).expect("should succeed");
//! let forecasts = garch_variance_forecast(&model, 0.0001_f64, 5).expect("should succeed");
//! ```

use scirs2_core::ndarray::Array1;

use crate::error::{Result, TimeSeriesError};
use crate::volatility::arch::{chi2_survival, ln_gamma};

/// GARCH(p,q) model parameters
///
/// Holds the estimated parameters from a GARCH(p,q) fit.
#[derive(Debug, Clone)]
pub struct GARCHModel {
    /// Number of lagged squared innovations (ARCH order)
    pub p: usize,
    /// Number of lagged conditional variances (GARCH order)
    pub q: usize,
    /// Intercept ω > 0
    pub omega: f64,
    /// ARCH coefficients α₁, …, αₚ (non-negative)
    pub alpha: Vec<f64>,
    /// GARCH coefficients β₁, …, βᵧ (non-negative)
    pub beta: Vec<f64>,
    /// Gaussian log-likelihood at the MLE estimates
    pub log_likelihood: f64,
    /// Number of observations used in fitting
    pub n_obs: usize,
}

impl GARCHModel {
    /// Create a GARCH model, validating parameter constraints.
    pub fn new(p: usize, q: usize, omega: f64, alpha: Vec<f64>, beta: Vec<f64>) -> Result<Self> {
        if alpha.len() != p {
            return Err(TimeSeriesError::InvalidModel(format!(
                "alpha length {} does not match p={}",
                alpha.len(),
                p
            )));
        }
        if beta.len() != q {
            return Err(TimeSeriesError::InvalidModel(format!(
                "beta length {} does not match q={}",
                beta.len(),
                q
            )));
        }
        if omega <= 0.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "omega".to_string(),
                message: "omega must be strictly positive".to_string(),
            });
        }
        for (i, &a) in alpha.iter().enumerate() {
            if a < 0.0 {
                return Err(TimeSeriesError::InvalidParameter {
                    name: format!("alpha[{}]", i),
                    message: "ARCH coefficients must be non-negative".to_string(),
                });
            }
        }
        for (j, &b) in beta.iter().enumerate() {
            if b < 0.0 {
                return Err(TimeSeriesError::InvalidParameter {
                    name: format!("beta[{}]", j),
                    message: "GARCH coefficients must be non-negative".to_string(),
                });
            }
        }
        Ok(Self {
            p,
            q,
            omega,
            alpha,
            beta,
            log_likelihood: f64::NEG_INFINITY,
            n_obs: 0,
        })
    }

    /// Persistence: sum of α and β coefficients.
    ///
    /// Must be < 1 for weak stationarity.  Values close to 1 indicate
    /// long-memory (integrated GARCH).
    pub fn persistence(&self) -> f64 {
        let alpha_sum: f64 = self.alpha.iter().sum();
        let beta_sum: f64 = self.beta.iter().sum();
        alpha_sum + beta_sum
    }

    /// Unconditional (long-run) variance: ω / (1 − Σα − Σβ)
    pub fn unconditional_variance(&self) -> Result<f64> {
        let pers = self.persistence();
        if pers >= 1.0 {
            return Err(TimeSeriesError::InvalidModel(
                "Model is non-stationary: persistence >= 1".to_string(),
            ));
        }
        Ok(self.omega / (1.0 - pers))
    }

    /// Number of free parameters: 1 + p + q
    pub fn n_params(&self) -> usize {
        1 + self.p + self.q
    }

    /// AIC = −2·ℓ + 2·k
    pub fn aic(&self) -> f64 {
        -2.0 * self.log_likelihood + 2.0 * self.n_params() as f64
    }

    /// BIC = −2·ℓ + k·ln(n)
    pub fn bic(&self) -> f64 {
        -2.0 * self.log_likelihood + self.n_params() as f64 * (self.n_obs as f64).ln()
    }
}

// ---------------------------------------------------------------------------
// Core recursion
// ---------------------------------------------------------------------------

/// Compute the GARCH conditional variance series and Gaussian log-likelihood.
///
/// Returns `(log_likelihood, sigma2_vec)`.
pub fn garch_log_likelihood(
    residuals: &Array1<f64>,
    omega: f64,
    alpha: &[f64],
    beta: &[f64],
) -> Result<(f64, Vec<f64>)> {
    let n = residuals.len();
    let p = alpha.len();
    let q = beta.len();
    let max_lag = p.max(q);

    if n <= max_lag {
        return Err(TimeSeriesError::InsufficientData {
            message: "GARCH log-likelihood".to_string(),
            required: max_lag + 1,
            actual: n,
        });
    }

    let mut sigma2 = vec![0.0_f64; n];

    // Initialise with sample variance
    let mean = residuals.iter().sum::<f64>() / n as f64;
    let sample_var: f64 = residuals.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    let init_var = sample_var.max(1e-8);

    for i in 0..max_lag.min(n) {
        sigma2[i] = init_var;
    }

    for t in max_lag..n {
        let mut var_t = omega;
        for lag in 1..=p {
            let idx = t - lag;
            var_t += alpha[lag - 1] * residuals[idx] * residuals[idx];
        }
        for lag in 1..=q {
            let idx = t - lag;
            var_t += beta[lag - 1] * sigma2[idx];
        }
        sigma2[t] = var_t.max(1e-10);
    }

    // Gaussian log-likelihood (excluding initial observations)
    let log2pi = (2.0 * std::f64::consts::PI).ln();
    let mut ll = 0.0_f64;
    for t in max_lag..n {
        ll -= 0.5 * (log2pi + sigma2[t].ln() + residuals[t] * residuals[t] / sigma2[t]);
    }

    Ok((ll, sigma2))
}

// ---------------------------------------------------------------------------
// Parameter projection
// ---------------------------------------------------------------------------

/// Map unconstrained raw parameters to the feasible GARCH space.
fn project_params_garch(raw: &[f64], p: usize, q: usize) -> (f64, Vec<f64>, Vec<f64>) {
    let softplus = |x: f64| -> f64 { (1.0 + x.exp()).ln().max(1e-8) };

    let omega = softplus(raw[0]);
    let alpha: Vec<f64> = (0..p).map(|i| softplus(raw[1 + i])).collect();
    let beta: Vec<f64> = (0..q).map(|j| softplus(raw[1 + p + j])).collect();

    let total: f64 = alpha.iter().sum::<f64>() + beta.iter().sum::<f64>();
    if total >= 0.999 {
        let scale = 0.95 / total;
        let alpha_s: Vec<f64> = alpha.iter().map(|&a| a * scale).collect();
        let beta_s: Vec<f64> = beta.iter().map(|&b| b * scale).collect();
        return (omega, alpha_s, beta_s);
    }

    (omega, alpha, beta)
}

// ---------------------------------------------------------------------------
// Nelder-Mead (copied locally to avoid cross-module import complexity)
// ---------------------------------------------------------------------------

fn nelder_mead_garch<F>(f: F, x0: Vec<f64>, max_iter: usize, tol: f64) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.clone());
    for i in 0..n {
        let mut vertex = x0.clone();
        vertex[i] += if vertex[i].abs() > 1e-5 {
            0.05 * vertex[i].abs()
        } else {
            0.00025
        };
        simplex.push(vertex);
    }
    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for _iter in 0..max_iter {
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let best = order[0];
        let worst = order[n];
        let second_worst = order[n - 1];

        let spread: f64 = order
            .iter()
            .map(|&i| (fvals[i] - fvals[best]).abs())
            .fold(0.0_f64, f64::max);
        if spread < tol {
            break;
        }

        let mut centroid = vec![0.0_f64; n];
        for &i in order.iter().take(n) {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        for c in centroid.iter_mut() {
            *c /= n as f64;
        }

        let reflected: Vec<f64> = (0..n)
            .map(|j| centroid[j] + 1.0 * (centroid[j] - simplex[worst][j]))
            .collect();
        let f_reflected = f(&reflected);

        if f_reflected < fvals[best] {
            let expanded: Vec<f64> = (0..n)
                .map(|j| centroid[j] + 2.0 * (reflected[j] - centroid[j]))
                .collect();
            let f_expanded = f(&expanded);
            if f_expanded < f_reflected {
                simplex[worst] = expanded;
                fvals[worst] = f_expanded;
            } else {
                simplex[worst] = reflected;
                fvals[worst] = f_reflected;
            }
        } else if f_reflected < fvals[second_worst] {
            simplex[worst] = reflected;
            fvals[worst] = f_reflected;
        } else {
            let contracted: Vec<f64> = (0..n)
                .map(|j| centroid[j] + 0.5 * (simplex[worst][j] - centroid[j]))
                .collect();
            let f_contracted = f(&contracted);
            if f_contracted < fvals[worst] {
                simplex[worst] = contracted;
                fvals[worst] = f_contracted;
            } else {
                let best_vertex = simplex[best].clone();
                for i in 0..=n {
                    if i != best {
                        for j in 0..n {
                            simplex[i][j] = best_vertex[j] + 0.5 * (simplex[i][j] - best_vertex[j]);
                        }
                        fvals[i] = f(&simplex[i]);
                    }
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..=n).collect();
    order.sort_by(|&a, &b| {
        fvals[a]
            .partial_cmp(&fvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let best = order[0];
    (simplex[best].clone(), fvals[best])
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fit a GARCH(p,q) model via maximum likelihood.
///
/// Maximises the Gaussian log-likelihood using the Nelder-Mead simplex
/// algorithm in a reparameterised (unconstrained) space.
///
/// # Arguments
///
/// * `residuals` - Demeaned return series (zero-mean innovations)
/// * `p`         - ARCH order (typically 1)
/// * `q`         - GARCH order (typically 1)
///
/// # Examples
///
/// ```rust
/// use scirs2_series::volatility::garch::fit_garch;
/// use scirs2_core::ndarray::Array1;
///
/// let returns = Array1::from(vec![
///     0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.018, 0.009,
///     -0.003, 0.007, 0.025, -0.014, 0.008, -0.006, 0.011, -0.019,
///     0.022, 0.003, -0.011, 0.017,
/// ]);
/// let model = fit_garch(&returns, 1, 1).expect("should succeed");
/// assert!(model.omega > 0.0);
/// assert!(model.persistence() < 1.0);
/// ```
pub fn fit_garch(residuals: &Array1<f64>, p: usize, q: usize) -> Result<GARCHModel> {
    let n = residuals.len();
    let min_obs = p.max(q) + 10;
    if n < min_obs {
        return Err(TimeSeriesError::InsufficientData {
            message: "GARCH fitting".to_string(),
            required: min_obs,
            actual: n,
        });
    }
    if p == 0 && q == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "p,q".to_string(),
            message: "At least one of p or q must be positive".to_string(),
        });
    }

    let mean = residuals.iter().sum::<f64>() / n as f64;
    let sample_var = residuals.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    let sample_var = sample_var.max(1e-8);

    // Initial raw parameters
    let n_params = 1 + p + q;
    let mut x0 = vec![0.0_f64; n_params];
    // omega: target ~10% of sample_var
    x0[0] = (sample_var * 0.1).ln();
    // alpha: start at 0.1
    for i in 0..p {
        x0[1 + i] = 0.1_f64.ln();
    }
    // beta: start at 0.8
    for j in 0..q {
        x0[1 + p + j] = 0.8_f64.ln();
    }

    let resid_clone: Vec<f64> = residuals.iter().cloned().collect();

    let obj = move |raw: &[f64]| {
        let (omega, alpha, beta) = project_params_garch(raw, p, q);
        let arr = Array1::from(resid_clone.clone());
        match garch_log_likelihood(&arr, omega, &alpha, &beta) {
            Ok((ll, _)) => -ll,
            Err(_) => f64::INFINITY,
        }
    };

    let (best_raw, neg_ll) = nelder_mead_garch(obj, x0, 3000 * n_params, 1e-8);
    let (omega, alpha, beta) = project_params_garch(&best_raw, p, q);

    let mut model = GARCHModel::new(p, q, omega, alpha, beta)?;
    model.log_likelihood = -neg_ll;
    model.n_obs = n;
    Ok(model)
}

/// Generate multi-step-ahead variance forecasts from a fitted GARCH model.
///
/// For h-step-ahead forecasts the GARCH recursion is evaluated using:
/// - The last q conditional variances from the in-sample fit
/// - The last p squared residuals
/// - Futures sigma^2 replaced by sigma^2(t+h-1) in subsequent steps (`E[eps^2] = sigma^2`)
///
/// # Arguments
///
/// * `model`         - Fitted GARCH model
/// * `last_variance` - Most recent in-sample conditional variance σ²ₜ
/// * `h`             - Forecast horizon (number of steps ahead)
///
/// # Returns
///
/// Vector of length h containing σ²ₜ₊₁, …, σ²ₜ₊ₕ
///
/// # Examples
///
/// ```rust
/// use scirs2_series::volatility::garch::{fit_garch, garch_variance_forecast};
/// use scirs2_core::ndarray::Array1;
///
/// let returns = Array1::from(vec![
///     0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.018, 0.009,
///     -0.003, 0.007, 0.025, -0.014, 0.008, -0.006, 0.011, -0.019,
///     0.022, 0.003, -0.011, 0.017,
/// ]);
/// let model = fit_garch(&returns, 1, 1).expect("should succeed");
/// let last_var = 0.0001_f64;
/// let forecasts = garch_variance_forecast(&model, last_var, 5).expect("should succeed");
/// assert_eq!(forecasts.len(), 5);
/// ```
pub fn garch_variance_forecast(
    model: &GARCHModel,
    last_variance: f64,
    h: usize,
) -> Result<Vec<f64>> {
    if h == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "h".to_string(),
            message: "Forecast horizon must be at least 1".to_string(),
        });
    }
    if last_variance <= 0.0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "last_variance".to_string(),
            message: "last_variance must be positive".to_string(),
        });
    }

    let unc_var = model.unconditional_variance().unwrap_or(last_variance);
    let alpha_sum: f64 = model.alpha.iter().sum();
    let beta_sum: f64 = model.beta.iter().sum();
    let pers = alpha_sum + beta_sum;

    // For the multi-step recursion with only one lag (GARCH(1,1)):
    // σ²ₜ₊ₕ = ω + (α + β)·σ²ₜ₊ₕ₋₁  where future ε² is replaced by σ²
    // General: mean-reverting geometric series toward unconditional variance.

    let mut forecasts = vec![0.0_f64; h];

    // Step 1: use the last observed variance
    let step1 = model.omega
        + model.alpha.iter().sum::<f64>() * last_variance
        + model.beta.iter().sum::<f64>() * last_variance;
    forecasts[0] = step1.max(1e-10);

    // Steps 2..h: mean-reversion recursion
    for step in 1..h {
        forecasts[step] = model.omega + pers * forecasts[step - 1];
        // Clamp to avoid numeric overflow
        forecasts[step] = forecasts[step].max(1e-10);
    }

    // Verify convergence toward unconditional variance if stationary
    if pers < 1.0 {
        for f in &mut forecasts {
            // Cap at a large multiple of the unconditional variance
            if *f > unc_var * 1e6 {
                *f = unc_var * 1e6;
            }
        }
    }

    Ok(forecasts)
}

/// Ljung-Box test on squared standardised residuals.
///
/// Computes the Q statistic for `lags` autocorrelation lags of the squared
/// standardised residuals.  Used to assess GARCH model adequacy.
///
/// # Arguments
///
/// * `std_resid` - Standardised residuals εₜ/σₜ
/// * `lags`      - Number of autocorrelation lags
///
/// # Returns
///
/// `(Q_statistic, p_value)` where Q ~ χ²(lags) under H₀
pub fn ljung_box_squared(std_resid: &[f64], lags: usize) -> Result<(f64, f64)> {
    let n = std_resid.len();
    if n < lags + 5 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Ljung-Box test".to_string(),
            required: lags + 5,
            actual: n,
        });
    }

    let sq: Vec<f64> = std_resid.iter().map(|&r| r * r).collect();
    let sq_mean = sq.iter().sum::<f64>() / n as f64;

    // Sample autocovariance at lag 0
    let gamma0: f64 = sq.iter().map(|&s| (s - sq_mean).powi(2)).sum::<f64>() / n as f64;

    if gamma0 < 1e-15 {
        return Err(TimeSeriesError::NumericalInstability(
            "Zero variance in squared residuals".to_string(),
        ));
    }

    let mut q_stat = 0.0_f64;
    for lag in 1..=lags {
        // Sample autocorrelation at this lag
        let gamma_lag: f64 = (lag..n)
            .map(|t| (sq[t] - sq_mean) * (sq[t - lag] - sq_mean))
            .sum::<f64>()
            / n as f64;
        let rho = gamma_lag / gamma0;
        q_stat += rho * rho / (n - lag) as f64;
    }
    q_stat *= n as f64 * (n as f64 + 2.0);

    let p_value = chi2_survival(q_stat, lags as f64);
    Ok((q_stat, p_value))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn sample_returns() -> Array1<f64> {
        Array1::from(vec![
            0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.018, 0.009, -0.003, 0.007, 0.025, -0.014,
            0.008, -0.006, 0.011, -0.019, 0.022, 0.003, -0.011, 0.017, -0.005, 0.031, -0.013,
            0.009, 0.002, -0.027, 0.016, -0.007, 0.013, 0.004,
        ])
    }

    #[test]
    fn test_garch_model_new() {
        let m = GARCHModel::new(1, 1, 0.0001, vec![0.1], vec![0.8]).expect("Should create");
        assert_eq!(m.p, 1);
        assert_eq!(m.q, 1);
        assert!((m.persistence() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_garch_invalid_params() {
        assert!(GARCHModel::new(1, 1, -1e-4, vec![0.1], vec![0.8]).is_err());
        assert!(GARCHModel::new(1, 1, 1e-4, vec![-0.1], vec![0.8]).is_err());
        assert!(GARCHModel::new(1, 1, 1e-4, vec![0.1], vec![-0.8]).is_err());
        assert!(GARCHModel::new(2, 1, 1e-4, vec![0.1], vec![0.8]).is_err());
    }

    #[test]
    fn test_garch_unconditional_variance() {
        let m = GARCHModel::new(1, 1, 0.0001, vec![0.1], vec![0.8]).expect("Should create");
        let uv = m.unconditional_variance().expect("Should compute");
        let expected = 0.0001 / (1.0 - 0.9);
        assert!((uv - expected).abs() < 1e-10);
    }

    #[test]
    fn test_garch_log_likelihood_shape() {
        let returns = sample_returns();
        let (ll, sigma2) =
            garch_log_likelihood(&returns, 0.0001, &[0.1], &[0.8]).expect("Should compute");
        assert!(ll.is_finite());
        assert_eq!(sigma2.len(), returns.len());
        for &s in &sigma2 {
            assert!(s > 0.0);
        }
    }

    #[test]
    fn test_fit_garch_11() {
        let returns = sample_returns();
        let model = fit_garch(&returns, 1, 1).expect("Should fit");
        assert!(model.omega > 0.0);
        assert!(model.alpha[0] >= 0.0);
        assert!(model.beta[0] >= 0.0);
        assert!(model.persistence() < 1.0);
        assert!(model.log_likelihood.is_finite());
    }

    #[test]
    fn test_fit_garch_10() {
        // GARCH(1,0) is equivalent to ARCH(1)
        let returns = sample_returns();
        let model = fit_garch(&returns, 1, 0).expect("Should fit");
        assert_eq!(model.q, 0);
        assert!(model.beta.is_empty());
    }

    #[test]
    fn test_garch_variance_forecast() {
        let returns = sample_returns();
        let model = fit_garch(&returns, 1, 1).expect("Should fit");
        let forecasts = garch_variance_forecast(&model, 0.0001, 10).expect("Should forecast");
        assert_eq!(forecasts.len(), 10);
        for &f in &forecasts {
            assert!(f > 0.0);
        }
        // Forecasts should converge toward unconditional variance
        // (monotone not guaranteed, but final values should be finite)
        assert!(forecasts.iter().all(|&f| f.is_finite()));
    }

    #[test]
    fn test_garch_forecast_invalid() {
        let m = GARCHModel::new(1, 1, 0.0001, vec![0.1], vec![0.8]).expect("Should create");
        assert!(garch_variance_forecast(&m, 0.0001, 0).is_err());
        assert!(garch_variance_forecast(&m, -0.0001, 5).is_err());
    }

    #[test]
    fn test_garch_aic_bic() {
        let returns = sample_returns();
        let model = fit_garch(&returns, 1, 1).expect("Should fit");
        let aic = model.aic();
        let bic = model.bic();
        assert!(aic.is_finite());
        assert!(bic.is_finite());
        // BIC penalises more: BIC > AIC for n > e^2 ≈ 7.4
        assert!(bic >= aic);
    }

    #[test]
    fn test_ljung_box_squared() {
        let std_resid: Vec<f64> = vec![
            0.5, -1.2, 0.8, -0.3, 1.1, -0.7, 0.4, -0.9, 0.6, -1.0, 0.2, 1.3, -0.5, 0.7, -0.4,
        ];
        let (q, p) = ljung_box_squared(&std_resid, 2).expect("Should compute");
        assert!(q >= 0.0);
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn test_ln_gamma_known_values() {
        // Γ(1) = 1, ln(Γ(1)) = 0
        assert!((ln_gamma(1.0) - 0.0).abs() < 1e-6);
        // Γ(0.5) = √π, ln(Γ(0.5)) = 0.5 ln(π)
        let expected = 0.5 * std::f64::consts::PI.ln();
        assert!((ln_gamma(0.5) - expected).abs() < 1e-6);
    }
}
