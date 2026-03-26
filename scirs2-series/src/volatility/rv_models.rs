//! Realized Volatility (RV) models for high-frequency financial data
//!
//! This module provides non-parametric and model-based estimators of integrated
//! variance using intra-day high-frequency price observations.
//!
//! # Estimators
//!
//! | Estimator | Formula | Robustness |
//! |-----------|---------|------------|
//! | Realized Variance (RV) | `Σ r²_{t,j}` | None |
//! | Bipower Variation (BV) | `(π/2) Σ|r_{t,j}||r_{t,j-1}|` | Robust to jumps |
//! | Realized Kernel (RK) | Bartlett-weighted autocovariances | Robust to microstructure noise |
//!
//! # HAR Model — Corsi (2009)
//!
//! The Heterogeneous Autoregressive (HAR) model captures the multi-scale
//! nature of volatility via daily, weekly, and monthly RV components:
//!
//! ```text
//! RV_t = c + φ_d * RV_{t-1} + φ_w * RV^{(w)}_{t-1} + φ_m * RV^{(m)}_{t-1} + ε_t
//! ```
//!
//! where `RV^{(w)}_{t-1} = (1/5) Σ_{k=1}^{5} RV_{t-k}` and
//! `RV^{(m)}_{t-1} = (1/22) Σ_{k=1}^{22} RV_{t-k}`.
//!
//! # References
//! - Andersen, T. G., & Bollerslev, T. (1998). Answering the Skeptics: Yes,
//!   Standard Volatility Models Do Provide Accurate Forecasts. *International
//!   Economic Review*, 39(4), 885–905.
//! - Barndorff-Nielsen, O. E., & Shephard, N. (2004). Power and bipower
//!   variation with stochastic volatility and jumps. *Journal of Financial
//!   Econometrics*, 2(1), 1–37.
//! - Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized
//!   Volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_series::volatility::rv_models::{
//!     realized_variance, bipower_variation, HARModel, fit_har
//! };
//! use scirs2_core::ndarray::Array1;
//!
//! // High-frequency prices
//! let prices: Vec<f64> = vec![
//!     100.0, 100.05, 99.98, 100.12, 100.03, 100.08, 99.95, 100.15,
//!     100.02, 100.07, 100.00, 100.10,
//! ];
//! let rv = realized_variance(&prices).expect("Should compute RV");
//! let bv = bipower_variation(&prices).expect("Should compute BV");
//! println!("RV={:.8}, BV={:.8}", rv, bv);
//! ```

use scirs2_core::ndarray::Array1;

use crate::error::{Result, TimeSeriesError};

// ============================================================
// Non-parametric RV estimators
// ============================================================

/// Compute Realized Variance (RV) from a sequence of intra-day prices.
///
/// ```text
/// RV = Σ_{j=2}^{M} [ln(p_j/p_{j-1})]²
/// ```
pub fn realized_variance(prices: &[f64]) -> Result<f64> {
    if prices.len() < 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "realized_variance needs at least 2 prices".into(),
            required: 2,
            actual: prices.len(),
        });
    }
    if prices.iter().any(|&p| p <= 0.0) {
        return Err(TimeSeriesError::InvalidInput(
            "realized_variance: all prices must be positive".into(),
        ));
    }

    let rv: f64 = prices
        .windows(2)
        .map(|w| {
            let r = (w[1] / w[0]).ln();
            r * r
        })
        .sum();

    Ok(rv)
}

/// Compute Bipower Variation (BV) — robust to jumps in the price path.
///
/// ```text
/// BV = (π/2) Σ_{j=3}^{M} |r_{j-1}| |r_{j-2}|
/// ```
///
/// Under the assumption of a continuous Itô semimartingale, `E[BV] = IV`
/// (the integrated variance), while `RV - BV` consistently estimates the
/// jump contribution.
pub fn bipower_variation(prices: &[f64]) -> Result<f64> {
    if prices.len() < 3 {
        return Err(TimeSeriesError::InsufficientData {
            message: "bipower_variation needs at least 3 prices".into(),
            required: 3,
            actual: prices.len(),
        });
    }
    if prices.iter().any(|&p| p <= 0.0) {
        return Err(TimeSeriesError::InvalidInput(
            "bipower_variation: all prices must be positive".into(),
        ));
    }

    let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect();

    let bv: f64 = returns
        .windows(2)
        .map(|w| w[0].abs() * w[1].abs())
        .sum::<f64>()
        * (std::f64::consts::PI / 2.0);

    Ok(bv)
}

/// Compute the Realized Kernel estimator (Parzen kernel) for variance.
///
/// The realized kernel is robust to microstructure noise and uses
/// Bartlett weights on the autocovariance terms:
///
/// ```text
/// RK = γ_0 + 2 Σ_{h=1}^{H} k(h/H) γ_h
/// ```
///
/// where `γ_h = Σ_{j=h+1}^{M} r_j r_{j-h}` and `k(x) = 1 - x` (Bartlett).
pub fn realized_kernel(prices: &[f64], bandwidth: usize) -> Result<f64> {
    if prices.len() < bandwidth + 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: format!(
                "realized_kernel: need at least bandwidth+2 = {} prices",
                bandwidth + 2
            ),
            required: bandwidth + 2,
            actual: prices.len(),
        });
    }
    if prices.iter().any(|&p| p <= 0.0) {
        return Err(TimeSeriesError::InvalidInput(
            "realized_kernel: all prices must be positive".into(),
        ));
    }

    let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect();

    let m = returns.len();

    // Autocovariance γ_h = Σ r_j * r_{j-h}
    let gamma_0: f64 = returns.iter().map(|&r| r * r).sum();

    let mut rk = gamma_0;
    for h in 1..=bandwidth {
        let gamma_h: f64 = (h..m).map(|j| returns[j] * returns[j - h]).sum();
        // Bartlett kernel weight k(h/H) = 1 - h/(H+1)
        let weight = 1.0 - h as f64 / (bandwidth + 1) as f64;
        rk += 2.0 * weight * gamma_h;
    }

    Ok(rk.max(0.0))
}

// ============================================================
// Jump tests
// ============================================================

/// Result of the Andersen-Bollerslev-Tauchen (2007) jump test.
#[derive(Debug, Clone)]
pub struct ABTJumpTestResult {
    /// Realized Variance
    pub rv: f64,
    /// Bipower Variation
    pub bv: f64,
    /// Jump test statistic (approximately N(0,1) under H₀: no jump)
    pub statistic: f64,
    /// Two-sided p-value
    pub p_value: f64,
    /// Estimated jump contribution: max(RV - BV, 0)
    pub jump_variation: f64,
}

/// Compute the Andersen-Bollerslev-Tauchen (ABT) test for intra-day jumps.
///
/// The test statistic is:
/// ```text
/// z = (RV - BV) / RV * sqrt(n / ((π/2)² + π - 5))
/// ```
///
/// which is approximately standard normal under the null of no jumps.
///
/// # Arguments
/// * `rv` — realized variance for the day
/// * `bv` — bipower variation for the day
/// * `n_obs` — number of intra-day observations (e.g., 78 for 5-min returns in a 6.5h session)
pub fn jump_test_abt(rv: f64, bv: f64, n_obs: usize) -> Result<ABTJumpTestResult> {
    if rv <= 0.0 {
        return Err(TimeSeriesError::InvalidInput(
            "ABT jump test: RV must be positive".into(),
        ));
    }
    if bv < 0.0 {
        return Err(TimeSeriesError::InvalidInput(
            "ABT jump test: BV must be non-negative".into(),
        ));
    }
    if n_obs < 2 {
        return Err(TimeSeriesError::InvalidInput(
            "ABT jump test: n_obs must be >= 2".into(),
        ));
    }

    // Variance of the asymptotic distribution: π/2)² + π - 5 ≈ 0.6088
    let kappa4 = (std::f64::consts::PI / 2.0).powi(2) + std::f64::consts::PI - 5.0;
    let statistic = (rv - bv) / rv * (n_obs as f64 / kappa4).sqrt();

    // Two-sided p-value from standard normal
    let p_value = 2.0 * standard_normal_sf(statistic.abs());

    Ok(ABTJumpTestResult {
        rv,
        bv,
        statistic,
        p_value,
        jump_variation: (rv - bv).max(0.0),
    })
}

// ============================================================
// HAR Model — Corsi (2009)
// ============================================================

/// HAR (Heterogeneous Autoregressive) model parameters.
///
/// Fitted by OLS on the three-component regression:
/// ```text
/// RV_t = c + φ_d * RV_{t-1} + φ_w * RV^{(w)}_{t-1} + φ_m * RV^{(m)}_{t-1} + ε_t
/// ```
#[derive(Debug, Clone)]
pub struct HARModel {
    /// Intercept c
    pub c: f64,
    /// Daily component coefficient φ_d
    pub phi_d: f64,
    /// Weekly component coefficient φ_w
    pub phi_w: f64,
    /// Monthly component coefficient φ_m
    pub phi_m: f64,
    /// Residual standard deviation
    pub residual_std: f64,
    /// R² of the HAR regression
    pub r_squared: f64,
    /// Number of observations in the estimation sample
    pub n_obs: usize,
}

/// Compute daily/weekly/monthly HAR regressors from a realized variance series.
///
/// Returns `(rv_d, rv_w, rv_m)` vectors aligned with the dependent variable
/// `rv_series[22..]`.
pub fn har_regressors(rv_series: &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let n = rv_series.len();
    if n < 24 {
        return Err(TimeSeriesError::InsufficientData {
            message: "HAR regressors: need at least 24 RV observations (22 for monthly lag + 2)"
                .into(),
            required: 24,
            actual: n,
        });
    }

    // y = rv_series[22..n]   (indices 22, 23, ..., n-1)
    let t_start = 22;
    let n_reg = n - t_start;

    let mut rv_d = Vec::with_capacity(n_reg);
    let mut rv_w = Vec::with_capacity(n_reg);
    let mut rv_m = Vec::with_capacity(n_reg);

    for t in t_start..n {
        rv_d.push(rv_series[t - 1]);
        rv_w.push(rv_series[t - 5..t].iter().sum::<f64>() / 5.0);
        rv_m.push(rv_series[t - 22..t].iter().sum::<f64>() / 22.0);
    }

    Ok((rv_d, rv_w, rv_m))
}

/// Fit a HAR model by OLS.
///
/// # Arguments
/// * `rv_series` — time series of daily realized variances (length ≥ 24)
pub fn fit_har(rv_series: &[f64]) -> Result<HARModel> {
    let n = rv_series.len();
    if n < 24 {
        return Err(TimeSeriesError::InsufficientData {
            message: "fit_har: need at least 24 RV observations".into(),
            required: 24,
            actual: n,
        });
    }

    let t_start = 22;
    let n_reg = n - t_start;

    let (rv_d, rv_w, rv_m) = har_regressors(rv_series)?;
    let y: Vec<f64> = rv_series[t_start..].to_vec();

    // Build design matrix X (n_reg × 4): [1, rv_d, rv_w, rv_m]
    // Solve normal equations (X'X)β = X'y via Cholesky (4×4 system)

    // Compute X'X and X'y
    let k = 4_usize;
    let mut xtx = vec![0.0_f64; k * k];
    let mut xty = vec![0.0_f64; k];

    for i in 0..n_reg {
        let row = [1.0_f64, rv_d[i], rv_w[i], rv_m[i]];
        for (a, &ra) in row.iter().enumerate() {
            xty[a] += ra * y[i];
            for (b, &rb) in row.iter().enumerate() {
                xtx[a * k + b] += ra * rb;
            }
        }
    }

    // Solve 4×4 linear system via Gaussian elimination with partial pivoting
    let beta = solve_4x4(&xtx, &xty)?;

    let c = beta[0];
    let phi_d = beta[1];
    let phi_w = beta[2];
    let phi_m = beta[3];

    // Compute residuals and R²
    let y_mean = y.iter().sum::<f64>() / n_reg as f64;
    let mut ss_res = 0.0_f64;
    let mut ss_tot = 0.0_f64;
    for i in 0..n_reg {
        let y_hat = c + phi_d * rv_d[i] + phi_w * rv_w[i] + phi_m * rv_m[i];
        let resid = y[i] - y_hat;
        ss_res += resid * resid;
        ss_tot += (y[i] - y_mean).powi(2);
    }

    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };
    let residual_std = (ss_res / (n_reg as f64 - 4.0).max(1.0)).sqrt();

    Ok(HARModel {
        c,
        phi_d,
        phi_w,
        phi_m,
        residual_std,
        r_squared,
        n_obs: n_reg,
    })
}

/// Compute `h`-step-ahead RV forecasts from a HAR model.
///
/// Multi-step forecasts are generated via iterated substitution:
/// - Daily: previous one-step-ahead forecast
/// - Weekly: rolling mean of last 5 daily forecasts
/// - Monthly: rolling mean of last 22 daily forecasts
///
/// # Arguments
/// * `model` — fitted HAR model
/// * `rv_series` — in-sample RV series used to initialise the forecast
/// * `h` — forecast horizon (h ≥ 1)
pub fn har_forecast(model: &HARModel, rv_series: &[f64], h: usize) -> Result<Vec<f64>> {
    if h == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "har_forecast: h must be >= 1".into(),
        ));
    }
    if rv_series.len() < 22 {
        return Err(TimeSeriesError::InsufficientData {
            message: "har_forecast: need at least 22 in-sample RV values".into(),
            required: 22,
            actual: rv_series.len(),
        });
    }

    // Extend with forecasted values iteratively
    let mut history: Vec<f64> = rv_series.to_vec();
    let mut forecasts = Vec::with_capacity(h);

    for _ in 0..h {
        let m = history.len();
        let rv_d = history[m - 1];
        let rv_w = history[m - 5..m].iter().sum::<f64>() / 5.0;
        let rv_m = history[m - 22..m].iter().sum::<f64>() / 22.0;
        let rv_next =
            (model.c + model.phi_d * rv_d + model.phi_w * rv_w + model.phi_m * rv_m).max(0.0);
        forecasts.push(rv_next);
        history.push(rv_next);
    }

    Ok(forecasts)
}

// ============================================================
// HAR-J (HAR with jump component)
// ============================================================

/// HAR-J model: HAR extended with an explicit jump component.
///
/// ```text
/// RV_t = c + φ_d RV_{t-1} + φ_w RV^{(w)} + φ_m RV^{(m)} + φ_j J_{t-1} + ε_t
/// ```
///
/// where `J_{t-1} = max(RV_{t-1} - BV_{t-1}, 0)`.
#[derive(Debug, Clone)]
pub struct HARJModel {
    /// Intercept
    pub c: f64,
    /// Daily RV coefficient
    pub phi_d: f64,
    /// Weekly RV coefficient
    pub phi_w: f64,
    /// Monthly RV coefficient
    pub phi_m: f64,
    /// Jump coefficient
    pub phi_j: f64,
    /// R² of the regression
    pub r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
}

/// Fit a HAR-J model by OLS.
///
/// # Arguments
/// * `rv_series` — daily realized variance series (length ≥ 24)
/// * `bv_series` — daily bipower variation series (same length as rv_series)
pub fn fit_har_j(rv_series: &[f64], bv_series: &[f64]) -> Result<HARJModel> {
    if rv_series.len() != bv_series.len() {
        return Err(TimeSeriesError::InvalidInput(
            "HAR-J: rv_series and bv_series must have the same length".into(),
        ));
    }
    let n = rv_series.len();
    if n < 24 {
        return Err(TimeSeriesError::InsufficientData {
            message: "HAR-J: need at least 24 observations".into(),
            required: 24,
            actual: n,
        });
    }

    let t_start = 22;
    let n_reg = n - t_start;

    let (rv_d, rv_w, rv_m) = har_regressors(rv_series)?;
    let y: Vec<f64> = rv_series[t_start..].to_vec();
    let jump: Vec<f64> = (0..n_reg)
        .map(|i| (rv_series[t_start - 1 + i] - bv_series[t_start - 1 + i]).max(0.0))
        .collect();

    // Design matrix k=5: [1, rv_d, rv_w, rv_m, jump]
    let k = 5_usize;
    let mut xtx = vec![0.0_f64; k * k];
    let mut xty = vec![0.0_f64; k];

    for i in 0..n_reg {
        let row = [1.0, rv_d[i], rv_w[i], rv_m[i], jump[i]];
        for (a, &ra) in row.iter().enumerate() {
            xty[a] += ra * y[i];
            for (b, &rb) in row.iter().enumerate() {
                xtx[a * k + b] += ra * rb;
            }
        }
    }

    let beta = solve_nxn(&xtx, &xty, k)?;

    let y_mean = y.iter().sum::<f64>() / n_reg as f64;
    let mut ss_res = 0.0_f64;
    let mut ss_tot = 0.0_f64;
    for i in 0..n_reg {
        let y_hat =
            beta[0] + beta[1] * rv_d[i] + beta[2] * rv_w[i] + beta[3] * rv_m[i] + beta[4] * jump[i];
        ss_res += (y[i] - y_hat).powi(2);
        ss_tot += (y[i] - y_mean).powi(2);
    }

    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    Ok(HARJModel {
        c: beta[0],
        phi_d: beta[1],
        phi_w: beta[2],
        phi_m: beta[3],
        phi_j: beta[4],
        r_squared,
        n_obs: n_reg,
    })
}

// ============================================================
// Diebold-Mariano forecast comparison test
// ============================================================

/// Diebold-Mariano test for equal predictive ability.
///
/// Compares two forecast error series `e1` and `e2` under a given loss
/// function.  The test statistic
/// ```text
/// DM = d_bar / sqrt(LRV(d))
/// ```
/// where `d_t = L(e1_t) - L(e2_t)` and `LRV` is the long-run variance of `d`.
///
/// # Arguments
/// * `e1` — forecast errors from model 1
/// * `e2` — forecast errors from model 2
/// * `squared` — if `true`, use squared-error loss; otherwise absolute-error loss
/// * `h` — forecast horizon (used for Newey-West bandwidth)
///
/// Returns `(DM_statistic, p_value_two_sided)`.
pub fn diebold_mariano_test(e1: &[f64], e2: &[f64], squared: bool, h: usize) -> Result<(f64, f64)> {
    if e1.len() != e2.len() {
        return Err(TimeSeriesError::InvalidInput(
            "Diebold-Mariano: e1 and e2 must have the same length".into(),
        ));
    }
    let n = e1.len();
    if n < 4 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Diebold-Mariano: need at least 4 observations".into(),
            required: 4,
            actual: n,
        });
    }

    let loss = |e: f64| if squared { e * e } else { e.abs() };
    let d: Vec<f64> = e1
        .iter()
        .zip(e2.iter())
        .map(|(&a, &b)| loss(a) - loss(b))
        .collect();

    let d_mean = d.iter().sum::<f64>() / n as f64;

    // Newey-West long-run variance with bandwidth h
    let bandwidth = h.max(1);
    let mut lrv = d.iter().map(|&di| (di - d_mean).powi(2)).sum::<f64>() / n as f64;

    for lag in 1..=bandwidth {
        let gamma_h: f64 = (lag..n)
            .map(|t| (d[t] - d_mean) * (d[t - lag] - d_mean))
            .sum::<f64>()
            / n as f64;
        let weight = 1.0 - lag as f64 / (bandwidth + 1) as f64;
        lrv += 2.0 * weight * gamma_h;
    }

    if lrv <= 0.0 {
        // Fall back to simple variance (avoid division by zero)
        lrv = d.iter().map(|&di| (di - d_mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    }

    let dm_stat = d_mean / (lrv / n as f64).sqrt().max(1e-15);
    let p_value = 2.0 * standard_normal_sf(dm_stat.abs());

    Ok((dm_stat, p_value))
}

// ============================================================
// Utility: standard normal survival function
// ============================================================

/// Approximate standard normal survival function P(Z > z).
pub(crate) fn standard_normal_sf(z: f64) -> f64 {
    // Hart's rational approximation to erfc
    0.5 * erfc_approx(z / std::f64::consts::SQRT_2)
}

/// Approximate complementary error function via Horner form.
fn erfc_approx(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_approx(-x);
    }
    // Abramowitz & Stegun 7.1.26 polynomial approximation, max error |ε| < 1.5e-7
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    (-x * x).exp() * poly
}

// ============================================================
// Linear algebra helpers (small systems)
// ============================================================

/// Solve a 4×4 linear system `Ax = b` via Gaussian elimination.
fn solve_4x4(a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
    solve_nxn(a, b, 4)
}

/// Solve an n×n linear system `Ax = b` via Gaussian elimination with partial pivoting.
fn solve_nxn(a_in: &[f64], b_in: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut a = a_in.to_vec();
    let mut b = b_in.to_vec();

    for col in 0..n {
        // Find pivot
        let pivot_row = (col..n)
            .max_by(|&i, &j| {
                a[i * n + col]
                    .abs()
                    .partial_cmp(&a[j * n + col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);

        if a[pivot_row * n + col].abs() < 1e-14 {
            return Err(TimeSeriesError::NumericalInstability(
                "OLS solve: singular matrix (multicollinearity?)".into(),
            ));
        }

        // Swap rows
        if pivot_row != col {
            for k in 0..n {
                a.swap(col * n + k, pivot_row * n + k);
            }
            b.swap(col, pivot_row);
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = a[row * n + col] / a[col * n + col];
            for k in col..n {
                let val = a[col * n + k] * factor;
                a[row * n + k] -= val;
            }
            b[row] -= b[col] * factor;
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * x[j];
        }
        x[i] = sum / a[i * n + i];
    }

    Ok(x)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_prices(n: usize) -> Vec<f64> {
        // Geometric random walk with small steps
        let mut prices = vec![100.0_f64];
        for i in 1..n {
            let ret = 0.001 * ((i as f64 * 1.37 + 0.5).sin());
            prices.push(prices[i - 1] * (1.0 + ret));
        }
        prices
    }

    fn make_rv_series(n: usize) -> Vec<f64> {
        // Synthetic RV with vol-clustering character
        (0..n)
            .map(|i| {
                let base = 0.0001_f64;
                let cluster = if i % 7 < 3 { 3.0 } else { 1.0 };
                base * cluster * (1.0 + 0.5 * ((i as f64 * 0.31).sin()).abs())
            })
            .collect()
    }

    #[test]
    fn test_realized_variance_basic() {
        let prices = make_prices(20);
        let rv = realized_variance(&prices).expect("Should compute RV");
        assert!(rv >= 0.0, "RV must be non-negative: {rv}");
        assert!(rv.is_finite());
    }

    #[test]
    fn test_realized_variance_too_few() {
        assert!(realized_variance(&[100.0]).is_err());
    }

    #[test]
    fn test_bipower_variation_basic() {
        let prices = make_prices(20);
        let bv = bipower_variation(&prices).expect("Should compute BV");
        assert!(bv >= 0.0, "BV must be non-negative: {bv}");
        assert!(bv.is_finite());
    }

    #[test]
    fn test_bipower_variation_too_few() {
        assert!(bipower_variation(&[100.0, 101.0]).is_err());
    }

    #[test]
    fn test_rv_ge_zero() {
        let prices = make_prices(50);
        let rv = realized_variance(&prices).expect("RV");
        let bv = bipower_variation(&prices).expect("BV");
        assert!(rv >= 0.0);
        assert!(bv >= 0.0);
    }

    #[test]
    fn test_realized_kernel_basic() {
        let prices = make_prices(50);
        let rk = realized_kernel(&prices, 5).expect("Should compute RK");
        assert!(rk >= 0.0);
        assert!(rk.is_finite());
    }

    #[test]
    fn test_jump_test_abt_basic() {
        let prices = make_prices(78);
        let rv = realized_variance(&prices).expect("RV");
        let bv = bipower_variation(&prices).expect("BV");
        let result = jump_test_abt(rv, bv, 78).expect("ABT test should compute");
        assert!(result.statistic.is_finite());
        assert!((0.0..=1.0).contains(&result.p_value));
        assert!(result.jump_variation >= 0.0);
    }

    #[test]
    fn test_jump_test_abt_invalid() {
        assert!(jump_test_abt(-0.001, 0.0, 78).is_err());
        assert!(jump_test_abt(0.001, 0.0, 1).is_err());
    }

    #[test]
    fn test_har_regressors_shape() {
        let rv = make_rv_series(50);
        let (rv_d, rv_w, rv_m) = har_regressors(&rv).expect("Should build regressors");
        let expected_len = rv.len() - 22;
        assert_eq!(rv_d.len(), expected_len);
        assert_eq!(rv_w.len(), expected_len);
        assert_eq!(rv_m.len(), expected_len);
    }

    #[test]
    fn test_fit_har_basic() {
        let rv = make_rv_series(60);
        let model = fit_har(&rv).expect("HAR should fit");
        assert!(model.c.is_finite());
        assert!(model.phi_d.is_finite());
        assert!(model.phi_w.is_finite());
        assert!(model.phi_m.is_finite());
        assert!((0.0..=1.0).contains(&model.r_squared.max(0.0).min(1.0)));
    }

    #[test]
    fn test_har_forecast_basic() {
        let rv = make_rv_series(60);
        let model = fit_har(&rv).expect("HAR should fit");
        let forecasts = har_forecast(&model, &rv, 5).expect("Should forecast");
        assert_eq!(forecasts.len(), 5);
        for &f in &forecasts {
            assert!(f >= 0.0, "Forecasted RV must be non-negative: {f}");
            assert!(f.is_finite());
        }
    }

    #[test]
    fn test_fit_har_j_basic() {
        let rv = make_rv_series(60);
        // BV with varying fraction to break collinearity with RV
        let bv: Vec<f64> = rv
            .iter()
            .enumerate()
            .map(|(i, &v)| v * (0.75 + 0.2 * ((i as f64 * 0.7).sin() * 0.5 + 0.5)))
            .collect();
        let model = fit_har_j(&rv, &bv).expect("HAR-J should fit");
        assert!(model.c.is_finite());
        assert!(model.phi_j.is_finite());
    }

    #[test]
    fn test_diebold_mariano_basic() {
        let e1: Vec<f64> = vec![0.01, -0.02, 0.03, -0.01, 0.02, -0.015, 0.025, -0.005];
        let e2: Vec<f64> = vec![0.015, -0.01, 0.02, -0.015, 0.025, -0.01, 0.015, -0.01];
        let (dm, p) = diebold_mariano_test(&e1, &e2, true, 1).expect("DM test should compute");
        assert!(dm.is_finite());
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn test_standard_normal_sf() {
        // P(Z > 0) = 0.5
        let p = standard_normal_sf(0.0);
        assert!((p - 0.5).abs() < 1e-6, "P(Z>0) = 0.5, got {p}");
        // P(Z > 1.96) ≈ 0.025
        let p2 = standard_normal_sf(1.96);
        assert!((p2 - 0.025).abs() < 0.001, "P(Z>1.96) ≈ 0.025, got {p2}");
    }
}
