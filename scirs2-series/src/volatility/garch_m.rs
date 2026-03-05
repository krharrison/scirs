//! GARCH-M (GARCH-in-Mean) volatility model — Engle, Lilien & Robins (1987)
//!
//! The GARCH-M model extends GARCH by including the conditional standard deviation
//! (or variance) in the mean equation, capturing a risk premium: higher volatility
//! is rewarded with higher expected returns.
//!
//! # Model Specification
//!
//! ```text
//! r_t = μ + δ·f(σ²_t) + ε_t
//! σ²_t = ω + Σᵢ αᵢ ε²_{t-i} + Σⱼ βⱼ σ²_{t-j}
//! ```
//!
//! where `f(σ²_t)` can be:
//! - `σ²_t`       (variance-in-mean)
//! - `σ_t`        (standard deviation in mean)
//! - `ln(σ²_t)`   (log-variance in mean)
//!
//! The coefficient δ measures the **risk premium**: its significance tests whether
//! agents require compensation for bearing additional volatility risk.
//!
//! # References
//! - Engle, R. F., Lilien, D. M., & Robins, R. P. (1987). Estimating time varying
//!   risk premia in the term structure: The ARCH-M model.
//!   *Econometrica*, 55(2), 391–407.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_series::volatility::garch_m::{GarchMModel, RiskMeasure, fit_garch_m};
//! use scirs2_core::ndarray::Array1;
//!
//! let returns: Array1<f64> = Array1::from(vec![
//!     0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.018, 0.009,
//!     -0.003, 0.007, 0.025, -0.014, 0.008, -0.006, 0.011, -0.019,
//!     0.022, 0.003, -0.011, 0.017, -0.004, 0.008, -0.013, 0.019,
//!     0.001, -0.009, 0.016, -0.002, 0.011, 0.006,
//! ]);
//! let model = fit_garch_m(&returns, 1, 1, RiskMeasure::StdDev).expect("GARCH-M fitting failed");
//! println!("delta(risk premium)={:.4}, omega={:.6}", model.delta, model.omega);
//! ```

use scirs2_core::ndarray::Array1;

use crate::error::{Result, TimeSeriesError};
use crate::volatility::egarch::nelder_mead;

/// The risk measure included in the mean equation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskMeasure {
    /// Include conditional variance σ²_t in mean
    Variance,
    /// Include conditional standard deviation σ_t in mean (most common)
    StdDev,
    /// Include log conditional variance ln(σ²_t) in mean
    LogVariance,
}

impl RiskMeasure {
    /// Apply risk measure transformation to conditional variance
    pub fn apply(&self, var: f64) -> f64 {
        match self {
            Self::Variance => var,
            Self::StdDev => var.max(0.0).sqrt(),
            Self::LogVariance => {
                if var > 0.0 {
                    var.ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
    }
}

/// GARCH-M(p,q) model parameters
///
/// Mean equation: `r_t = μ + δ·f(σ²_t) + ε_t`
/// Variance equation: `σ²_t = ω + Σα_i ε²_{t-i} + Σβ_j σ²_{t-j}`
#[derive(Debug, Clone)]
pub struct GarchMModel {
    /// ARCH order (number of lagged squared innovations)
    pub p: usize,
    /// GARCH order (number of lagged conditional variances)
    pub q: usize,
    /// Mean intercept μ
    pub mu: f64,
    /// Risk premium coefficient δ (key parameter)
    pub delta: f64,
    /// Risk measure form in mean equation
    pub risk_measure: RiskMeasure,
    /// Variance intercept ω > 0
    pub omega: f64,
    /// ARCH coefficients α₁, …, αₚ (non-negative)
    pub alpha: Vec<f64>,
    /// GARCH coefficients β₁, …, βᵧ (non-negative)
    pub beta: Vec<f64>,
    /// Gaussian log-likelihood at MLE estimates
    pub log_likelihood: f64,
    /// Number of observations used in fitting
    pub n_obs: usize,
}

impl GarchMModel {
    /// Create a new GARCH-M model, validating all parameter constraints.
    pub fn new(
        p: usize,
        q: usize,
        mu: f64,
        delta: f64,
        risk_measure: RiskMeasure,
        omega: f64,
        alpha: Vec<f64>,
        beta: Vec<f64>,
    ) -> Result<Self> {
        if p == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "GARCH-M ARCH order p must be at least 1".to_string(),
            ));
        }
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
                    name: format!("alpha[{i}]"),
                    message: "ARCH coefficients must be non-negative".to_string(),
                });
            }
        }
        for (j, &b) in beta.iter().enumerate() {
            if b < 0.0 {
                return Err(TimeSeriesError::InvalidParameter {
                    name: format!("beta[{j}]"),
                    message: "GARCH coefficients must be non-negative".to_string(),
                });
            }
        }
        Ok(Self {
            p,
            q,
            mu,
            delta,
            risk_measure,
            omega,
            alpha,
            beta,
            log_likelihood: f64::NEG_INFINITY,
            n_obs: 0,
        })
    }

    /// GARCH persistence: Σα + Σβ. Must be < 1 for weak stationarity.
    pub fn persistence(&self) -> f64 {
        let a: f64 = self.alpha.iter().sum();
        let b: f64 = self.beta.iter().sum();
        a + b
    }

    /// Unconditional variance: ω / (1 − persistence)
    pub fn unconditional_variance(&self) -> Result<f64> {
        let pers = self.persistence();
        if pers >= 1.0 {
            return Err(TimeSeriesError::InvalidModel(
                "Model is non-stationary: persistence >= 1".to_string(),
            ));
        }
        Ok(self.omega / (1.0 - pers))
    }

    /// Number of free parameters: 2 (μ,δ) + 1 (ω) + p + q
    pub fn n_params(&self) -> usize {
        2 + 1 + self.p + self.q
    }

    /// AIC = −2·ℓ + 2·k
    pub fn aic(&self) -> f64 {
        -2.0 * self.log_likelihood + 2.0 * self.n_params() as f64
    }

    /// BIC = −2·ℓ + k·ln(n)
    pub fn bic(&self) -> f64 {
        let k = self.n_params() as f64;
        let n = self.n_obs as f64;
        -2.0 * self.log_likelihood + k * n.ln()
    }

    /// Compute conditional variances for the given return series.
    pub fn conditional_variances(&self, returns: &Array1<f64>) -> Result<Vec<f64>> {
        garch_m_variances(
            returns,
            self.mu,
            self.delta,
            self.risk_measure,
            self.omega,
            &self.alpha,
            &self.beta,
        )
    }

    /// Compute conditional volatilities (square roots of variances).
    pub fn conditional_volatilities(&self, returns: &Array1<f64>) -> Result<Vec<f64>> {
        let vars = self.conditional_variances(returns)?;
        Ok(vars.into_iter().map(|v| v.max(0.0).sqrt()).collect())
    }

    /// Multi-step variance forecast, holding ε_t = 0 beyond the sample.
    pub fn forecast(&self, returns: &Array1<f64>, steps: usize) -> Result<Vec<f64>> {
        if steps == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "steps must be at least 1".to_string(),
            ));
        }
        let n = returns.len();
        if n < self.p.max(self.q) {
            return Err(TimeSeriesError::InvalidInput(
                "Insufficient observations to initialise the forecast".to_string(),
            ));
        }

        // Compute in-sample conditional variances
        let h_in = self.conditional_variances(returns)?;

        // Residuals for in-sample period
        let eps_in: Vec<f64> = returns
            .iter()
            .zip(h_in.iter())
            .map(|(&r, &h)| r - self.mu - self.delta * self.risk_measure.apply(h))
            .collect();

        let uncond_var = self.unconditional_variance().unwrap_or(self.omega);
        let mut h_ext: Vec<f64> = h_in.clone();
        let mut eps_ext: Vec<f64> = eps_in.clone();

        for _ in 0..steps {
            let t = h_ext.len();
            // For forecasting periods, E[ε²_{t+s}] = σ²_{t+s} (zero-mean residuals)
            let mut h_new = self.omega;
            for i in 0..self.p {
                let idx = t.saturating_sub(i + 1);
                if idx < n {
                    h_new += self.alpha[i] * eps_ext[idx].powi(2);
                } else {
                    h_new += self.alpha[i] * uncond_var;
                }
            }
            for j in 0..self.q {
                let idx = t.saturating_sub(j + 1);
                h_new += self.beta[j] * h_ext[idx];
            }
            h_new = h_new.max(1e-12);
            eps_ext.push(0.0); // E[ε_{t+s}] = 0
            h_ext.push(h_new);
        }

        Ok(h_ext[n..].to_vec())
    }
}

/// Compute GARCH-M conditional variances given parameters.
///
/// Variance equation: `σ²_t = ω + Σα_i ε²_{t-i} + Σβ_j σ²_{t-j}`
/// where residuals are `ε_t = r_t − μ − δ·f(σ²_t)` (iterated jointly).
///
/// # Note on Joint Estimation
/// Because ε_t depends on σ²_t (through the mean), we iterate the variance
/// and residuals simultaneously using the previous σ²_{t-1} as the
/// right-hand side approximation (one-step lag). This is the standard
/// practice for GARCH-M quasi-MLE.
fn garch_m_variances(
    returns: &Array1<f64>,
    mu: f64,
    delta: f64,
    risk_measure: RiskMeasure,
    omega: f64,
    alpha: &[f64],
    beta: &[f64],
) -> Result<Vec<f64>> {
    let n = returns.len();
    let p = alpha.len();
    let q = beta.len();

    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Return series is empty".to_string(),
        ));
    }

    // Estimate initial unconditional variance from returns
    let mean_r: f64 = returns.iter().sum::<f64>() / n as f64;
    let init_var: f64 = {
        let v: f64 = returns.iter().map(|&r| (r - mean_r).powi(2)).sum::<f64>() / n as f64;
        if v > 0.0 { v } else { omega }
    };

    let mut h: Vec<f64> = vec![init_var; n];
    let mut eps: Vec<f64> = vec![0.0; n];

    // Initialise residuals using initial variance
    for t in 0..n {
        eps[t] = returns[t] - mu - delta * risk_measure.apply(h[t]);
    }

    // Iterate: update variances then residuals
    for _iter in 0..100 {
        let h_prev = h.clone();
        let eps_prev = eps.clone();

        for t in 0..n {
            let mut ht = omega;
            for i in 0..p {
                if t > i {
                    ht += alpha[i] * eps_prev[t - i - 1].powi(2);
                } else {
                    ht += alpha[i] * init_var;
                }
            }
            for j in 0..q {
                if t > j {
                    ht += beta[j] * h_prev[t - j - 1];
                } else {
                    ht += beta[j] * init_var;
                }
            }
            h[t] = ht.max(1e-12);
            eps[t] = returns[t] - mu - delta * risk_measure.apply(h[t]);
        }

        // Check convergence
        let max_diff: f64 = h
            .iter()
            .zip(h_prev.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        if max_diff < 1e-10 {
            break;
        }
    }

    Ok(h)
}

/// Compute GARCH-M Gaussian log-likelihood and conditional variances.
fn garch_m_log_likelihood(
    returns: &Array1<f64>,
    mu: f64,
    delta: f64,
    risk_measure: RiskMeasure,
    omega: f64,
    alpha: &[f64],
    beta: &[f64],
) -> Result<(f64, Vec<f64>)> {
    use std::f64::consts::PI;

    let h = garch_m_variances(returns, mu, delta, risk_measure, omega, alpha, beta)?;
    let n = returns.len();

    let mut ll = 0.0;
    for t in 0..n {
        let eps = returns[t] - mu - delta * risk_measure.apply(h[t]);
        if h[t] <= 0.0 || !h[t].is_finite() {
            return Err(TimeSeriesError::NumericalInstability(format!(
                "Non-positive conditional variance at t={t}"
            )));
        }
        ll -= 0.5 * ((2.0 * PI).ln() + h[t].ln() + eps.powi(2) / h[t]);
    }

    if !ll.is_finite() {
        return Err(TimeSeriesError::NumericalInstability(
            "Log-likelihood is not finite".to_string(),
        ));
    }
    Ok((ll, h))
}

/// Fit a GARCH-M(p,q) model by quasi-maximum likelihood (Gaussian innovations).
///
/// # Arguments
/// - `returns` — observed return series
/// - `p`       — ARCH order (≥ 1)
/// - `q`       — GARCH order (≥ 0)
/// - `risk_measure` — form of risk term in mean equation
///
/// # Returns
/// Fitted [`GarchMModel`] with estimated parameters.
pub fn fit_garch_m(
    returns: &Array1<f64>,
    p: usize,
    q: usize,
    risk_measure: RiskMeasure,
) -> Result<GarchMModel> {
    let n = returns.len();
    if p == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "ARCH order p must be at least 1".to_string(),
        ));
    }
    let min_obs = 2 * (p + q) + 10;
    if n < min_obs {
        return Err(TimeSeriesError::InvalidInput(format!(
            "Need at least {min_obs} observations for GARCH-M({p},{q}), got {n}"
        )));
    }

    // Initial parameter estimates
    let mean_r: f64 = returns.iter().sum::<f64>() / n as f64;
    let var_r: f64 = {
        let v: f64 = returns.iter().map(|&r| (r - mean_r).powi(2)).sum::<f64>() / n as f64;
        if v > 0.0 { v } else { 1e-4 }
    };

    // Parameter vector: [mu, delta, omega, alpha_1..p, beta_1..q]
    let n_params = 2 + 1 + p + q;
    let mut x0 = vec![0.0_f64; n_params];
    x0[0] = mean_r; // mu
    x0[1] = 0.1;    // delta (small positive initial)
    x0[2] = var_r * 0.05; // omega

    let pers_init = if p + q > 0 { 0.85 / (p + q) as f64 } else { 0.0 };
    for i in 0..p {
        x0[3 + i] = pers_init * 0.3;
    }
    for j in 0..q {
        x0[3 + p + j] = pers_init * 0.7;
    }

    let objective = {
        let ret = returns.clone();
        let rm = risk_measure;
        move |params: &[f64]| -> f64 {
            let mu = params[0];
            let delta = params[1];
            let omega = params[2].abs() + 1e-8;
            let alpha: Vec<f64> = (0..p).map(|i| params[3 + i].abs()).collect();
            let beta: Vec<f64> = (0..q).map(|j| params[3 + p + j].abs()).collect();

            // Penalty if non-stationary
            let pers: f64 = alpha.iter().sum::<f64>() + beta.iter().sum::<f64>();
            if pers >= 0.9999 {
                return 1e10 * (1.0 + pers);
            }

            match garch_m_log_likelihood(&ret, mu, delta, rm, omega, &alpha, &beta) {
                Ok((ll, _)) if ll.is_finite() => -ll,
                _ => 1e10,
            }
        }
    };

    let max_iter = 3000 + 200 * n_params;
    let best = nelder_mead(&x0, &objective, max_iter, 1e-8)?;

    // Extract and enforce constraints
    let mu = best[0];
    let delta = best[1];
    let omega = best[2].abs() + 1e-8;
    let alpha: Vec<f64> = (0..p).map(|i| best[3 + i].abs()).collect();
    let beta: Vec<f64> = (0..q).map(|j| best[3 + p + j].abs()).collect();

    let (ll, _) = garch_m_log_likelihood(returns, mu, delta, risk_measure, omega, &alpha, &beta)?;

    let mut model = GarchMModel::new(p, q, mu, delta, risk_measure, omega, alpha, beta)?;
    model.log_likelihood = ll;
    model.n_obs = n;
    Ok(model)
}

/// Compute standardised residuals z_t = ε_t / σ_t for diagnostics.
pub fn garch_m_standardised_residuals(model: &GarchMModel, returns: &Array1<f64>) -> Result<Vec<f64>> {
    let h = model.conditional_variances(returns)?;
    let z: Vec<f64> = returns
        .iter()
        .zip(h.iter())
        .map(|(&r, &hi)| {
            let sigma = hi.max(0.0).sqrt();
            let eps = r - model.mu - model.delta * model.risk_measure.apply(hi);
            if sigma > 0.0 { eps / sigma } else { 0.0 }
        })
        .collect();
    Ok(z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn make_returns() -> Array1<f64> {
        Array1::from(vec![
            0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.018, 0.009,
            -0.003, 0.007, 0.025, -0.014, 0.008, -0.006, 0.011, -0.019,
            0.022, 0.003, -0.011, 0.017, -0.004, 0.008, -0.013, 0.019,
            0.001, -0.009, 0.016, -0.002, 0.011, 0.006, 0.014, -0.007,
            0.020, -0.005, 0.009, 0.003, -0.015, 0.012, -0.001, 0.018,
        ])
    }

    #[test]
    fn test_garch_m_model_new_valid() {
        let m = GarchMModel::new(
            1, 1, 0.001, 0.2,
            RiskMeasure::StdDev,
            1e-5, vec![0.1], vec![0.85],
        )
        .expect("Should create model");
        assert!((m.persistence() - 0.95).abs() < 1e-10);
        let uv = m.unconditional_variance().expect("Should compute unconditional variance");
        assert!(uv > 0.0);
    }

    #[test]
    fn test_garch_m_model_invalid_omega() {
        let r = GarchMModel::new(1, 1, 0.0, 0.0, RiskMeasure::Variance, -1e-5, vec![0.1], vec![0.85]);
        assert!(r.is_err(), "Negative omega should fail");
    }

    #[test]
    fn test_garch_m_model_invalid_alpha_count() {
        let r = GarchMModel::new(2, 1, 0.0, 0.0, RiskMeasure::Variance, 1e-5, vec![0.1], vec![0.85]);
        assert!(r.is_err(), "Mismatched alpha length should fail");
    }

    #[test]
    fn test_garch_m_conditional_variances() {
        let r = make_returns();
        let model = GarchMModel::new(1, 1, 0.001, 0.1, RiskMeasure::StdDev, 1e-5, vec![0.1], vec![0.85])
            .expect("Should create");
        let h = model.conditional_variances(&r).expect("Should compute variances");
        assert_eq!(h.len(), r.len());
        for &hi in &h {
            assert!(hi > 0.0, "All conditional variances must be positive");
            assert!(hi.is_finite(), "All conditional variances must be finite");
        }
    }

    #[test]
    fn test_risk_measure_apply() {
        let var = 0.04_f64;
        assert_eq!(RiskMeasure::Variance.apply(var), 0.04);
        assert!((RiskMeasure::StdDev.apply(var) - 0.2).abs() < 1e-10);
        assert!((RiskMeasure::LogVariance.apply(var) - var.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_fit_garch_m_11() {
        let r = make_returns();
        let model = fit_garch_m(&r, 1, 1, RiskMeasure::StdDev).expect("GARCH-M(1,1) should fit");
        assert!(model.mu.is_finite());
        assert!(model.delta.is_finite());
        assert!(model.omega > 0.0);
        assert!(model.alpha[0] >= 0.0);
        assert!(model.beta[0] >= 0.0);
        assert!(model.persistence() < 1.0);
        assert!(model.log_likelihood.is_finite());
        assert!(model.aic().is_finite());
        assert!(model.bic().is_finite());
        assert!(model.bic() >= model.aic()); // BIC penalises more for n > e^2
    }

    #[test]
    fn test_fit_garch_m_variance_form() {
        let r = make_returns();
        let model = fit_garch_m(&r, 1, 1, RiskMeasure::Variance).expect("Variance form should fit");
        assert!(model.log_likelihood.is_finite());
    }

    #[test]
    fn test_fit_garch_m_logvar_form() {
        let r = make_returns();
        let model = fit_garch_m(&r, 1, 1, RiskMeasure::LogVariance).expect("Log-variance form should fit");
        assert!(model.log_likelihood.is_finite());
    }

    #[test]
    fn test_garch_m_forecast() {
        let r = make_returns();
        let model = GarchMModel::new(1, 1, 0.001, 0.1, RiskMeasure::StdDev, 1e-5, vec![0.1], vec![0.85])
            .expect("Should create");
        let fc = model.forecast(&r, 5).expect("Should forecast");
        assert_eq!(fc.len(), 5);
        for &f in &fc {
            assert!(f > 0.0, "Forecast variances must be positive");
            assert!(f.is_finite());
        }
    }

    #[test]
    fn test_garch_m_standardised_residuals() {
        let r = make_returns();
        let model = GarchMModel::new(1, 1, 0.001, 0.1, RiskMeasure::StdDev, 1e-5, vec![0.1], vec![0.85])
            .expect("Should create");
        let z = garch_m_standardised_residuals(&model, &r).expect("Should compute residuals");
        assert_eq!(z.len(), r.len());
        for &zi in &z {
            assert!(zi.is_finite());
        }
    }
}
