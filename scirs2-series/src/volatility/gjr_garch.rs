//! GJR-GARCH and TGARCH asymmetric volatility models
//!
//! # GJR-GARCH — Glosten, Jagannathan & Runkle (1993)
//!
//! The GJR-GARCH model augments the standard GARCH variance equation with an
//! indicator term that captures the **leverage effect**: negative shocks (bad
//! news) tend to increase volatility more than positive shocks of equal
//! magnitude.
//!
//! ```text
//! σ²_t = ω + Σᵢ (αᵢ + γᵢ I[ε_{t-i}<0]) ε²_{t-i} + Σⱼ βⱼ σ²_{t-j}
//! ```
//!
//! For stationarity: `Σαᵢ + ½Σγᵢ + Σβⱼ < 1` (assuming symmetric innovations).
//!
//! # TGARCH — Zakoïan (1994)
//!
//! The Threshold GARCH (also Asymmetric Power ARCH in its linear form)
//! models the conditional standard deviation rather than variance:
//!
//! ```text
//! σ_t = ω + Σᵢ (αᵢ ε⁺_{t-i} + γᵢ |ε⁻_{t-i}|) + Σⱼ βⱼ σ_{t-j}
//! ```
//!
//! where `ε⁺ = max(ε,0)` and `|ε⁻| = max(-ε,0)`.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_series::volatility::gjr_garch::{GJRGARCHModel, fit_gjr_garch};
//! use scirs2_core::ndarray::Array1;
//!
//! let returns: Array1<f64> = Array1::from(vec![
//!     0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.018, 0.009,
//!     -0.003, 0.007, 0.025, -0.014, 0.008, -0.006, 0.011, -0.019,
//!     0.022, 0.003, -0.011, 0.017, -0.004, 0.008, -0.013, 0.019,
//!     0.001, -0.009, 0.016, -0.002, 0.011, 0.006,
//! ]);
//! let model = fit_gjr_garch(&returns, 1, 1).expect("GJR-GARCH fitting failed");
//! println!("leverage γ = {:.4}", model.gamma[0]);
//! ```

use scirs2_core::ndarray::Array1;

use crate::error::{Result, TimeSeriesError};
use crate::volatility::egarch::nelder_mead;

// ============================================================
// GJR-GARCH model struct
// ============================================================

/// GJR-GARCH(p,q) model parameters
///
/// Variance equation:
/// ```text
/// σ²_t = ω + Σᵢ (αᵢ + γᵢ I[ε_{t-i}<0]) ε²_{t-i} + Σⱼ βⱼ σ²_{t-j}
/// ```
#[derive(Debug, Clone)]
pub struct GJRGARCHModel {
    /// ARCH order p (number of lagged squared innovations)
    pub p: usize,
    /// GARCH order q (number of lagged conditional variances)
    pub q: usize,
    /// Constant term ω > 0
    pub omega: f64,
    /// ARCH (symmetric) coefficients αᵢ ≥ 0 (length p)
    pub alpha: Vec<f64>,
    /// Asymmetry (leverage) coefficients γᵢ ≥ 0 (length p)
    pub gamma: Vec<f64>,
    /// GARCH persistence coefficients βⱼ ≥ 0 (length q)
    pub beta: Vec<f64>,
    /// Log-likelihood at the fitted parameters
    pub log_likelihood: f64,
    /// Number of observations used in fitting
    pub n_obs: usize,
}

impl GJRGARCHModel {
    /// Construct a new `GJRGARCHModel` from raw parameter vectors.
    ///
    /// # Errors
    /// Returns an error when vector lengths do not match the declared orders
    /// or when `ω ≤ 0`, any `α < 0`, any `γ < 0`, any `β < 0`, or the
    /// stationarity condition is violated.
    pub fn new(
        p: usize,
        q: usize,
        omega: f64,
        alpha: Vec<f64>,
        gamma: Vec<f64>,
        beta: Vec<f64>,
    ) -> Result<Self> {
        if omega <= 0.0 {
            return Err(TimeSeriesError::InvalidModel(
                "GJR-GARCH: ω must be positive".into(),
            ));
        }
        if alpha.len() != p {
            return Err(TimeSeriesError::InvalidModel(format!(
                "alpha length {} != p={}",
                alpha.len(),
                p
            )));
        }
        if gamma.len() != p {
            return Err(TimeSeriesError::InvalidModel(format!(
                "gamma length {} != p={}",
                gamma.len(),
                p
            )));
        }
        if beta.len() != q {
            return Err(TimeSeriesError::InvalidModel(format!(
                "beta length {} != q={}",
                beta.len(),
                q
            )));
        }
        for (i, &a) in alpha.iter().enumerate() {
            if a < 0.0 {
                return Err(TimeSeriesError::InvalidModel(format!(
                    "GJR-GARCH: alpha[{}] = {a:.4} < 0",
                    i
                )));
            }
        }
        for (i, &g) in gamma.iter().enumerate() {
            if g < 0.0 {
                return Err(TimeSeriesError::InvalidModel(format!(
                    "GJR-GARCH: gamma[{}] = {g:.4} < 0",
                    i
                )));
            }
        }
        for (i, &b) in beta.iter().enumerate() {
            if b < 0.0 {
                return Err(TimeSeriesError::InvalidModel(format!(
                    "GJR-GARCH: beta[{}] = {b:.4} < 0",
                    i
                )));
            }
        }
        let persistence = alpha.iter().sum::<f64>()
            + 0.5 * gamma.iter().sum::<f64>()
            + beta.iter().sum::<f64>();
        if persistence >= 1.0 {
            return Err(TimeSeriesError::InvalidModel(format!(
                "GJR-GARCH: stationarity violated (Σα + ½Σγ + Σβ = {persistence:.4} >= 1)"
            )));
        }
        Ok(Self {
            p,
            q,
            omega,
            alpha,
            gamma,
            beta,
            log_likelihood: f64::NEG_INFINITY,
            n_obs: 0,
        })
    }

    /// Persistence: `Σαᵢ + ½Σγᵢ + Σβⱼ`
    pub fn persistence(&self) -> f64 {
        self.alpha.iter().sum::<f64>()
            + 0.5 * self.gamma.iter().sum::<f64>()
            + self.beta.iter().sum::<f64>()
    }

    /// Unconditional variance: `ω / (1 − Σα − ½Σγ − Σβ)`
    pub fn unconditional_variance(&self) -> Result<f64> {
        let denom = 1.0 - self.persistence();
        if denom <= 0.0 {
            return Err(TimeSeriesError::NumericalError(
                "GJR-GARCH: stationarity violated — cannot compute unconditional variance".into(),
            ));
        }
        Ok(self.omega / denom)
    }

    /// AIC = -2 * LL + 2 * n_params
    pub fn aic(&self) -> f64 {
        -2.0 * self.log_likelihood + 2.0 * self.n_params() as f64
    }

    /// BIC = -2 * LL + n_params * log(n_obs)
    pub fn bic(&self) -> f64 {
        -2.0 * self.log_likelihood + self.n_params() as f64 * (self.n_obs as f64).ln()
    }

    /// Number of free parameters: 1 + p + p + q
    pub fn n_params(&self) -> usize {
        1 + self.p + self.p + self.q
    }
}

// ============================================================
// Conditional variance recursion
// ============================================================

/// Compute the conditional variance series `σ²_t` for GJR-GARCH(p,q).
pub fn gjr_garch_variance(
    returns: &Array1<f64>,
    omega: f64,
    alpha: &[f64],
    gamma: &[f64],
    beta: &[f64],
) -> Result<Vec<f64>> {
    let n = returns.len();
    let p = alpha.len();
    let q = beta.len();

    if n < p.max(q) + 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "GJR-GARCH: too few observations".into(),
            required: p.max(q) + 2,
            actual: n,
        });
    }
    if omega <= 0.0 {
        return Err(TimeSeriesError::InvalidInput(
            "GJR-GARCH: omega must be positive".into(),
        ));
    }

    let mean = returns.mean().unwrap_or(0.0);
    let sample_var = returns
        .iter()
        .map(|&r| (r - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let init_var = sample_var.max(omega);

    let mut sigma2 = vec![init_var; n];

    for t in 1..n {
        let mut var_t = omega;

        // ARCH + asymmetry terms
        for (i, (&ai, &gi)) in alpha.iter().zip(gamma.iter()).enumerate() {
            let lag = i + 1;
            let eps_lag = if t >= lag {
                returns[t - lag] - mean
            } else {
                0.0
            };
            let indicator = if eps_lag < 0.0 { 1.0 } else { 0.0 };
            var_t += (ai + gi * indicator) * eps_lag * eps_lag;
        }

        // GARCH persistence terms
        for (j, &bj) in beta.iter().enumerate() {
            let lag = j + 1;
            let s2_lag = if t >= lag { sigma2[t - lag] } else { init_var };
            var_t += bj * s2_lag;
        }

        sigma2[t] = var_t.max(1e-15);
    }

    Ok(sigma2)
}

// ============================================================
// Log-likelihood
// ============================================================

/// Gaussian log-likelihood for GJR-GARCH(p,q).
///
/// Returns `(log_likelihood, sigma2_series)`.
pub fn gjr_garch_log_likelihood(
    returns: &Array1<f64>,
    omega: f64,
    alpha: &[f64],
    gamma: &[f64],
    beta: &[f64],
) -> Result<(f64, Vec<f64>)> {
    let sigma2 = gjr_garch_variance(returns, omega, alpha, gamma, beta)?;
    let n = returns.len();
    let burn_in = alpha.len().max(beta.len()).max(1);
    let mean = returns.mean().unwrap_or(0.0);

    let mut ll = 0.0_f64;
    for t in burn_in..n {
        let s2 = sigma2[t];
        if s2 <= 0.0 || !s2.is_finite() {
            return Err(TimeSeriesError::NumericalError(
                "GJR-GARCH: non-positive conditional variance".into(),
            ));
        }
        let eps = returns[t] - mean;
        ll += -0.5 * (std::f64::consts::TAU.ln() + s2.ln() + eps * eps / s2);
    }

    if !ll.is_finite() {
        return Err(TimeSeriesError::NumericalError(
            "GJR-GARCH: log-likelihood is not finite".into(),
        ));
    }

    Ok((ll, sigma2))
}

// ============================================================
// Fitting
// ============================================================

/// Fit a GJR-GARCH(p,q) model via Nelder-Mead maximisation of the Gaussian
/// log-likelihood.
///
/// # Arguments
/// * `returns` — observed return series
/// * `p` — ARCH order (lagged squared innovations, with leverage)
/// * `q` — GARCH order (lagged conditional variances)
pub fn fit_gjr_garch(returns: &Array1<f64>, p: usize, q: usize) -> Result<GJRGARCHModel> {
    let n = returns.len();
    let min_obs = (p.max(q) + 1) * 5;
    if n < min_obs {
        return Err(TimeSeriesError::InsufficientData {
            message: "GJR-GARCH: insufficient observations".into(),
            required: min_obs,
            actual: n,
        });
    }

    let mean = returns.mean().unwrap_or(0.0);
    let r: Array1<f64> = returns.mapv(|x| x - mean);
    let sample_var = r.iter().map(|&x| x * x).sum::<f64>() / n as f64;

    // Parameter layout: [omega, alpha_0..p-1, gamma_0..p-1, beta_0..q-1]
    let n_params = 1 + 2 * p + q;
    let mut theta = vec![0.0_f64; n_params];

    theta[0] = sample_var * 0.05; // omega
    for i in 0..p {
        theta[1 + i] = 0.05; // alpha
        theta[1 + p + i] = 0.08; // gamma (leverage)
    }
    for j in 0..q {
        theta[1 + 2 * p + j] = 0.85 / q as f64; // beta
    }

    let objective = |th: &[f64]| -> f64 {
        let omega = th[0];
        if omega <= 0.0 {
            return 1e15;
        }
        let alpha: Vec<f64> = th[1..1 + p].to_vec();
        let gamma: Vec<f64> = th[1 + p..1 + 2 * p].to_vec();
        let beta: Vec<f64> = th[1 + 2 * p..].to_vec();

        if alpha.iter().any(|&a| a < 0.0)
            || gamma.iter().any(|&g| g < 0.0)
            || beta.iter().any(|&b| b < 0.0)
        {
            return 1e15;
        }

        let persist = alpha.iter().sum::<f64>()
            + 0.5 * gamma.iter().sum::<f64>()
            + beta.iter().sum::<f64>();
        if persist >= 0.9999 {
            return 1e15;
        }

        // Use full returns for log-likelihood (not demeaned slice)
        match gjr_garch_log_likelihood(returns, omega, &alpha, &gamma, &beta) {
            Ok((ll, _)) => {
                if ll.is_finite() {
                    -ll
                } else {
                    1e15
                }
            }
            Err(_) => 1e15,
        }
    };

    let best = nelder_mead(&theta, &objective, 3000, 1e-9)?;

    let omega = best[0].max(1e-10);
    let alpha: Vec<f64> = best[1..1 + p].iter().map(|&v| v.max(0.0)).collect();
    let gamma: Vec<f64> = best[1 + p..1 + 2 * p].iter().map(|&v| v.max(0.0)).collect();
    let beta: Vec<f64> = best[1 + 2 * p..].iter().map(|&v| v.max(0.0)).collect();

    let persist = alpha.iter().sum::<f64>()
        + 0.5 * gamma.iter().sum::<f64>()
        + beta.iter().sum::<f64>();
    if persist >= 1.0 {
        return Err(TimeSeriesError::FittingError(
            "GJR-GARCH: fitted persistence >= 1 — model is non-stationary".into(),
        ));
    }

    let (ll, _) = gjr_garch_log_likelihood(returns, omega, &alpha, &gamma, &beta)?;

    let mut model = GJRGARCHModel::new(p, q, omega, alpha, gamma, beta)?;
    model.log_likelihood = ll;
    model.n_obs = n;
    Ok(model)
}

// ============================================================
// Forecasting
// ============================================================

/// Compute `h`-step-ahead conditional variance forecasts for a GJR-GARCH model.
///
/// Multi-step forecasts use the expectation: `E[I_{ε<0}] = 0.5` for symmetric
/// innovations, so effectively `ᾱ_i = α_i + γ_i/2` in the recursion.
pub fn gjr_garch_forecast(
    model: &GJRGARCHModel,
    returns: &Array1<f64>,
    h: usize,
) -> Result<Vec<f64>> {
    if h == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "forecast horizon must be >= 1".into(),
        ));
    }

    let sigma2 = gjr_garch_variance(returns, model.omega, &model.alpha, &model.gamma, &model.beta)?;
    let last_sigma2 = *sigma2.last().ok_or_else(|| TimeSeriesError::InsufficientData {
        message: "GJR-GARCH forecast: empty sigma2 series".into(),
        required: 1,
        actual: 0,
    })?;

    // Effective alpha under E[I_{eps<0}] = 0.5
    let alpha_eff: Vec<f64> = model
        .alpha
        .iter()
        .zip(model.gamma.iter())
        .map(|(&a, &g)| a + 0.5 * g)
        .collect();

    let persist: f64 = alpha_eff.iter().sum::<f64>() + model.beta.iter().sum::<f64>();
    let uncond = model.omega / (1.0 - persist).max(1e-12);

    let mut forecasts = Vec::with_capacity(h);
    let mut current = last_sigma2;

    for _ in 0..h {
        current = model.omega + persist * current;
        // Mean-revert toward unconditional
        let _ = uncond;
        forecasts.push(current.max(1e-15));
    }

    Ok(forecasts)
}

/// Compute conditional volatility series (σ_t) from a GJR-GARCH model.
pub fn gjr_conditional_volatility(model: &GJRGARCHModel, returns: &Array1<f64>) -> Result<Vec<f64>> {
    let sigma2 = gjr_garch_variance(returns, model.omega, &model.alpha, &model.gamma, &model.beta)?;
    Ok(sigma2.into_iter().map(|v| v.sqrt()).collect())
}

/// Compute standardised residuals from a GJR-GARCH model.
pub fn gjr_standardised_residuals(model: &GJRGARCHModel, returns: &Array1<f64>) -> Result<Vec<f64>> {
    let mean = returns.mean().unwrap_or(0.0);
    let sigma2 = gjr_garch_variance(returns, model.omega, &model.alpha, &model.gamma, &model.beta)?;
    let z: Vec<f64> = returns
        .iter()
        .zip(sigma2.iter())
        .map(|(&r, &s2)| (r - mean) / s2.sqrt().max(1e-12))
        .collect();
    Ok(z)
}

// ============================================================
// TGARCH (Threshold GARCH — Zakoïan 1994)
// ============================================================

/// TGARCH(p,q) model (models conditional standard deviation)
///
/// ```text
/// σ_t = ω + Σᵢ (αᵢ ε⁺_{t-i} + γᵢ |ε⁻_{t-i}|) + Σⱼ βⱼ σ_{t-j}
/// ```
///
/// where `ε⁺ = max(ε, 0)` and `|ε⁻| = max(-ε, 0)`.
#[derive(Debug, Clone)]
pub struct TGARCHModel {
    /// ARCH order p
    pub p: usize,
    /// GARCH order q
    pub q: usize,
    /// Constant term ω > 0
    pub omega: f64,
    /// Positive-shock coefficients αᵢ ≥ 0
    pub alpha: Vec<f64>,
    /// Negative-shock (leverage) coefficients γᵢ ≥ 0; γ > α ⇒ leverage effect
    pub gamma: Vec<f64>,
    /// Conditional-std persistence coefficients βⱼ ≥ 0
    pub beta: Vec<f64>,
    /// Log-likelihood at the fitted parameters
    pub log_likelihood: f64,
    /// Number of observations used in fitting
    pub n_obs: usize,
}

impl TGARCHModel {
    /// Construct a TGARCH model from raw parameters (with validation).
    pub fn new(
        p: usize,
        q: usize,
        omega: f64,
        alpha: Vec<f64>,
        gamma: Vec<f64>,
        beta: Vec<f64>,
    ) -> Result<Self> {
        if omega <= 0.0 {
            return Err(TimeSeriesError::InvalidModel("TGARCH: ω must be positive".into()));
        }
        if alpha.len() != p || gamma.len() != p || beta.len() != q {
            return Err(TimeSeriesError::InvalidModel(
                "TGARCH: parameter vector lengths do not match model orders".into(),
            ));
        }
        if alpha.iter().any(|&a| a < 0.0) || gamma.iter().any(|&g| g < 0.0) || beta.iter().any(|&b| b < 0.0) {
            return Err(TimeSeriesError::InvalidModel(
                "TGARCH: all coefficients must be non-negative".into(),
            ));
        }
        Ok(Self {
            p, q, omega, alpha, gamma, beta,
            log_likelihood: f64::NEG_INFINITY,
            n_obs: 0,
        })
    }

    /// Persistence: (Σα + Σγ)/2 + Σβ (for symmetric innovations)
    pub fn persistence(&self) -> f64 {
        0.5 * (self.alpha.iter().sum::<f64>() + self.gamma.iter().sum::<f64>())
            + self.beta.iter().sum::<f64>()
    }
}

/// Compute the conditional standard deviation series σ_t for a TGARCH model.
pub fn tgarch_sigma(
    returns: &Array1<f64>,
    omega: f64,
    alpha: &[f64],
    gamma: &[f64],
    beta: &[f64],
) -> Result<Vec<f64>> {
    let n = returns.len();
    let p = alpha.len();
    let q = beta.len();

    if n < p.max(q) + 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "TGARCH: too few observations".into(),
            required: p.max(q) + 2,
            actual: n,
        });
    }

    let mean = returns.mean().unwrap_or(0.0);
    let sample_std = (returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64)
        .sqrt()
        .max(1e-8);
    let init_sigma = sample_std;

    let mut sigma = vec![init_sigma; n];

    for t in 1..n {
        let mut s_t = omega;

        for (i, (&ai, &gi)) in alpha.iter().zip(gamma.iter()).enumerate() {
            let lag = i + 1;
            let eps_lag = if t >= lag { returns[t - lag] - mean } else { 0.0 };
            let eps_pos = eps_lag.max(0.0);
            let eps_neg = (-eps_lag).max(0.0);
            s_t += ai * eps_pos + gi * eps_neg;
        }

        for (j, &bj) in beta.iter().enumerate() {
            let lag = j + 1;
            let s_lag = if t >= lag { sigma[t - lag] } else { init_sigma };
            s_t += bj * s_lag;
        }

        sigma[t] = s_t.max(1e-12);
    }

    Ok(sigma)
}

/// Fit a TGARCH(p,q) model via Nelder-Mead.
pub fn fit_tgarch(returns: &Array1<f64>, p: usize, q: usize) -> Result<TGARCHModel> {
    let n = returns.len();
    let min_obs = (p.max(q) + 1) * 5;
    if n < min_obs {
        return Err(TimeSeriesError::InsufficientData {
            message: "TGARCH: insufficient observations".into(),
            required: min_obs,
            actual: n,
        });
    }

    let mean = returns.mean().unwrap_or(0.0);
    let sample_std = (returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64)
        .sqrt()
        .max(1e-8);

    let n_params = 1 + 2 * p + q;
    let mut theta = vec![0.0_f64; n_params];
    theta[0] = sample_std * 0.05;
    for i in 0..p {
        theta[1 + i] = 0.05;
        theta[1 + p + i] = 0.08;
    }
    for j in 0..q {
        theta[1 + 2 * p + j] = 0.80 / q as f64;
    }

    let objective = |th: &[f64]| -> f64 {
        let omega = th[0];
        if omega <= 0.0 { return 1e15; }
        let alpha: Vec<f64> = th[1..1 + p].to_vec();
        let gamma_v: Vec<f64> = th[1 + p..1 + 2 * p].to_vec();
        let beta: Vec<f64> = th[1 + 2 * p..].to_vec();

        if alpha.iter().any(|&a| a < 0.0)
            || gamma_v.iter().any(|&g| g < 0.0)
            || beta.iter().any(|&b| b < 0.0)
        {
            return 1e15;
        }

        let persist = 0.5 * (alpha.iter().sum::<f64>() + gamma_v.iter().sum::<f64>())
            + beta.iter().sum::<f64>();
        if persist >= 0.9999 { return 1e15; }

        match tgarch_log_likelihood(returns, omega, &alpha, &gamma_v, &beta) {
            Ok((ll, _)) => if ll.is_finite() { -ll } else { 1e15 },
            Err(_) => 1e15,
        }
    };

    let best = nelder_mead(&theta, &objective, 3000, 1e-9)?;

    let omega = best[0].max(1e-10);
    let alpha: Vec<f64> = best[1..1 + p].iter().map(|&v| v.max(0.0)).collect();
    let gamma: Vec<f64> = best[1 + p..1 + 2 * p].iter().map(|&v| v.max(0.0)).collect();
    let beta: Vec<f64> = best[1 + 2 * p..].iter().map(|&v| v.max(0.0)).collect();

    let (ll, _) = tgarch_log_likelihood(returns, omega, &alpha, &gamma, &beta)?;

    let mut model = TGARCHModel::new(p, q, omega, alpha, gamma, beta)?;
    model.log_likelihood = ll;
    model.n_obs = n;
    Ok(model)
}

/// Gaussian log-likelihood for TGARCH.
pub fn tgarch_log_likelihood(
    returns: &Array1<f64>,
    omega: f64,
    alpha: &[f64],
    gamma: &[f64],
    beta: &[f64],
) -> Result<(f64, Vec<f64>)> {
    let sigma = tgarch_sigma(returns, omega, alpha, gamma, beta)?;
    let n = returns.len();
    let burn_in = alpha.len().max(beta.len()).max(1);
    let mean = returns.mean().unwrap_or(0.0);

    let mut ll = 0.0_f64;
    for t in burn_in..n {
        let s = sigma[t];
        if s <= 0.0 || !s.is_finite() {
            return Err(TimeSeriesError::NumericalError(
                "TGARCH: non-positive conditional std".into(),
            ));
        }
        let eps = returns[t] - mean;
        ll += -0.5 * (std::f64::consts::TAU.ln() + 2.0 * s.ln() + eps * eps / (s * s));
    }

    if !ll.is_finite() {
        return Err(TimeSeriesError::NumericalError(
            "TGARCH: log-likelihood not finite".into(),
        ));
    }

    Ok((ll, sigma))
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn make_returns() -> Array1<f64> {
        Array1::from(vec![
            0.008, -0.015, 0.011, -0.007, 0.013, 0.005, -0.018, 0.009,
            -0.003, 0.007, 0.025, -0.014, 0.008, -0.006, 0.011, -0.019,
            0.022, 0.003, -0.011, 0.017, -0.004, 0.008, -0.013, 0.019,
            0.001, -0.009, 0.016, -0.002, 0.011, 0.006, 0.010, -0.020,
            0.014, -0.009, 0.015, 0.003, -0.016, 0.010, -0.004, 0.008,
            0.023, -0.012, 0.007, -0.007, 0.012, -0.021, 0.018, 0.002,
            -0.010, 0.016,
        ])
    }

    #[test]
    fn test_gjr_garch_model_new_valid() {
        let m = GJRGARCHModel::new(1, 1, 1e-5, vec![0.05], vec![0.08], vec![0.85])
            .expect("Should create");
        assert!(m.persistence() < 1.0);
        let uv = m.unconditional_variance().expect("Should compute");
        assert!(uv > 0.0);
    }

    #[test]
    fn test_gjr_garch_model_invalid_omega() {
        assert!(GJRGARCHModel::new(1, 1, -1e-5, vec![0.05], vec![0.08], vec![0.85]).is_err());
    }

    #[test]
    fn test_gjr_garch_model_invalid_stationarity() {
        assert!(GJRGARCHModel::new(1, 1, 1e-5, vec![0.5], vec![0.4], vec![0.8]).is_err());
    }

    #[test]
    fn test_gjr_garch_variance_basic() {
        let r = make_returns();
        let s2 = gjr_garch_variance(&r, 1e-5, &[0.05], &[0.08], &[0.85])
            .expect("Should compute variance");
        assert_eq!(s2.len(), r.len());
        for &v in &s2 {
            assert!(v > 0.0, "conditional variance must be positive: {v}");
        }
    }

    #[test]
    fn test_gjr_garch_log_likelihood() {
        let r = make_returns();
        let (ll, s2) = gjr_garch_log_likelihood(&r, 1e-5, &[0.05], &[0.08], &[0.85])
            .expect("Should compute LL");
        assert!(ll.is_finite());
        assert_eq!(s2.len(), r.len());
    }

    #[test]
    fn test_fit_gjr_garch_11() {
        let r = make_returns();
        let model = fit_gjr_garch(&r, 1, 1).expect("GJR-GARCH(1,1) should fit");
        assert!(model.omega > 0.0);
        assert!(model.alpha[0] >= 0.0);
        assert!(model.gamma[0] >= 0.0);
        assert!(model.beta[0] >= 0.0);
        assert!(model.persistence() < 1.0);
        assert!(model.log_likelihood.is_finite());
        assert!(model.aic().is_finite());
        assert!(model.bic().is_finite());
    }

    #[test]
    fn test_gjr_garch_forecast() {
        let r = make_returns();
        let model = fit_gjr_garch(&r, 1, 1).expect("Should fit");
        let f = gjr_garch_forecast(&model, &r, 5).expect("Should forecast");
        assert_eq!(f.len(), 5);
        for &v in &f {
            assert!(v > 0.0);
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_gjr_conditional_volatility() {
        let r = make_returns();
        let model = GJRGARCHModel::new(1, 1, 1e-5, vec![0.05], vec![0.08], vec![0.85])
            .expect("Should create");
        let vol = gjr_conditional_volatility(&model, &r).expect("Should compute");
        assert_eq!(vol.len(), r.len());
        for &v in &vol {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_tgarch_sigma_basic() {
        let r = make_returns();
        let sigma = tgarch_sigma(&r, 1e-4, &[0.04], &[0.08], &[0.85])
            .expect("Should compute sigma");
        assert_eq!(sigma.len(), r.len());
        for &s in &sigma {
            assert!(s > 0.0);
        }
    }

    #[test]
    fn test_fit_tgarch_11() {
        let r = make_returns();
        let model = fit_tgarch(&r, 1, 1).expect("TGARCH(1,1) should fit");
        assert!(model.omega > 0.0);
        assert!(model.log_likelihood.is_finite());
    }
}
