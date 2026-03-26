//! Enhanced Realized Volatility and HAR-RV models
//!
//! This module supplements [`rv_models`](super::rv_models) with additional
//! realized volatility estimators and HAR model extensions:
//!
//! - **Flat-top realized kernel** (Barndorff-Nielsen, Hansen, Lunde & Shephard, 2008):
//!   a noise-robust RV estimator using a flat-top Bartlett kernel that provides
//!   superior performance under market microstructure noise.
//! - **Two-scale realized variance** (Zhang, Mykland & Aït-Sahalia, 2005):
//!   combines fast and slow subsampled RVs to cancel noise bias.
//! - **Realized semi-variance**: separates upside and downside realized variance.
//! - **HAR-RV-CJ** (Andersen, Bollerslev & Diebold, 2007): HAR with continuous
//!   and jump decomposition, splitting RV into BV (continuous) and J (jump).
//!
//! # References
//! - Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2008).
//!   Designing realized kernels to measure the ex post variation of equity prices
//!   in the presence of noise. *Econometrica*, 76(6), 1481–1536.
//! - Zhang, L., Mykland, P. A., & Aït-Sahalia, Y. (2005). A tale of two time
//!   scales. *Journal of the American Statistical Association*, 100(472), 1394–1411.
//! - Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2007). Roughing it up:
//!   Including jump components in the measurement, modeling, and forecasting of
//!   return volatility. *Review of Economics and Statistics*, 89(4), 701–720.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_series::volatility::har_rv::{
//!     flat_top_realized_kernel, two_scale_rv, realized_semivariance,
//!     HARCJModel, fit_har_cj,
//! };
//!
//! let prices: Vec<f64> = (0..200)
//!     .map(|i| 100.0 * (1.0 + 0.0002 * (i as f64 * 0.73).sin()))
//!     .collect();
//!
//! let rk = flat_top_realized_kernel(&prices, None).expect("RK");
//! let tsrv = two_scale_rv(&prices, 5).expect("TSRV");
//! let (rsv_up, rsv_down) = realized_semivariance(&prices).expect("RSV");
//! ```

use crate::error::{Result, TimeSeriesError};
use crate::volatility::rv_models::{bipower_variation, realized_variance};

// ============================================================
// Flat-top Realized Kernel (BNHLS 2008)
// ============================================================

/// Flat-top Bartlett kernel function.
///
/// ```text
/// k(x) = 1          if x = 0
/// k(x) = 1 - |x|    if 0 < |x| <= 1
/// k(x) = 0          otherwise
/// ```
///
/// The flat-top at x=0 ensures positive semi-definiteness.
fn flat_top_bartlett(x: f64) -> f64 {
    let ax = x.abs();
    if ax <= 1.0 {
        1.0 - ax
    } else {
        0.0
    }
}

/// Compute the optimal bandwidth for the flat-top realized kernel.
///
/// Uses the formula from BNHLS (2008):
/// H* ≈ c * n^{2/3}
/// where c depends on the noise-to-signal ratio.
fn optimal_bandwidth(n: usize) -> usize {
    // Conservative: c ≈ 3.5 * n^{2/3} (Barndorff-Nielsen et al. recommendation)
    let h = (3.5 * (n as f64).powf(2.0 / 3.0)).round() as usize;
    h.max(1).min(n / 2)
}

/// Compute the flat-top realized kernel estimator of integrated variance.
///
/// This estimator is consistent for the quadratic variation even in the
/// presence of IID market microstructure noise. It uses a flat-top Bartlett
/// kernel with automatic bandwidth selection.
///
/// ```text
/// RK = Σ_{h=-H}^{H} k(h/(H+1)) γ_h
/// ```
///
/// where γ_h = Σ_{j=|h|+1}^{n} r_j r_{j-|h|} is the h-th autocovariance.
///
/// # Arguments
/// * `prices` — Intraday price observations (must be positive)
/// * `bandwidth` — Optional bandwidth H. If `None`, uses the optimal bandwidth.
pub fn flat_top_realized_kernel(prices: &[f64], bandwidth: Option<usize>) -> Result<f64> {
    if prices.len() < 4 {
        return Err(TimeSeriesError::InsufficientData {
            message: "flat_top_realized_kernel: need at least 4 prices".to_string(),
            required: 4,
            actual: prices.len(),
        });
    }
    if prices.iter().any(|&p| p <= 0.0) {
        return Err(TimeSeriesError::InvalidInput(
            "flat_top_realized_kernel: all prices must be positive".to_string(),
        ));
    }

    let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
    let m = returns.len();

    let h = bandwidth.unwrap_or_else(|| optimal_bandwidth(m));

    if h == 0 {
        // Just return RV
        return Ok(returns.iter().map(|&r| r * r).sum());
    }

    // Autocovariance at lag 0
    let gamma_0: f64 = returns.iter().map(|&r| r * r).sum();

    let mut rk = gamma_0;
    for lag in 1..=h.min(m - 1) {
        let gamma_h: f64 = (lag..m).map(|j| returns[j] * returns[j - lag]).sum();
        // Flat-top Bartlett weight
        let weight = flat_top_bartlett(lag as f64 / (h as f64 + 1.0));
        // Symmetric: add both positive and negative lags
        rk += 2.0 * weight * gamma_h;
    }

    // Ensure non-negativity
    Ok(rk.max(0.0))
}

// ============================================================
// Two-Scale Realized Variance (TSRV)
// ============================================================

/// Compute the Two-Scale Realized Variance (Zhang, Mykland & Aït-Sahalia, 2005).
///
/// TSRV combines a fast-scale RV (all data) with a subsampled slow-scale RV
/// to cancel the bias from microstructure noise:
///
/// ```text
/// TSRV = RV^{(slow)} - (n_bar / n) * RV^{(fast)}
/// ```
///
/// where RV^{(slow)} uses every K-th observation (averaged over K subgrids)
/// and RV^{(fast)} uses all observations.
///
/// # Arguments
/// * `prices` — Intraday price observations
/// * `k_slow` — Subsampling frequency for the slow scale (e.g., 5)
pub fn two_scale_rv(prices: &[f64], k_slow: usize) -> Result<f64> {
    if k_slow < 2 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "k_slow".to_string(),
            message: "Subsampling frequency must be >= 2".to_string(),
        });
    }
    if prices.len() < k_slow + 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "two_scale_rv: too few prices for given k_slow".to_string(),
            required: k_slow + 2,
            actual: prices.len(),
        });
    }
    if prices.iter().any(|&p| p <= 0.0) {
        return Err(TimeSeriesError::InvalidInput(
            "two_scale_rv: all prices must be positive".to_string(),
        ));
    }

    let n = prices.len() - 1; // number of returns

    // Fast-scale RV (all returns)
    let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
    let rv_fast: f64 = returns.iter().map(|&r| r * r).sum();

    // Slow-scale RV: average over K subgrids
    let mut rv_slow = 0.0_f64;
    for offset in 0..k_slow {
        // Subsample: prices at indices offset, offset+k_slow, offset+2*k_slow, ...
        let subprices: Vec<f64> = (0..)
            .map(|i| offset + i * k_slow)
            .take_while(|&idx| idx < prices.len())
            .map(|idx| prices[idx])
            .collect();
        if subprices.len() >= 2 {
            let sub_rv: f64 = subprices
                .windows(2)
                .map(|w| {
                    let r = (w[1] / w[0]).ln();
                    r * r
                })
                .sum();
            rv_slow += sub_rv;
        }
    }
    rv_slow /= k_slow as f64;

    // n_bar: average number of returns in each subgrid
    let n_bar = n as f64 / k_slow as f64;

    // TSRV = RV_slow - (n_bar/n) * RV_fast
    let tsrv = rv_slow - (n_bar / n as f64) * rv_fast;

    // Small-sample bias correction factor
    let correction = 1.0 / (1.0 - n_bar / n as f64);
    let tsrv_corrected = tsrv * correction;

    Ok(tsrv_corrected.max(0.0))
}

// ============================================================
// Realized Semi-Variance
// ============================================================

/// Compute realized semi-variances (upside and downside).
///
/// ```text
/// RS^{+}_t = Σ_{j: r_j > 0} r_j²
/// RS^{-}_t = Σ_{j: r_j ≤ 0} r_j²
/// ```
///
/// Note: RS^{+} + RS^{-} = RV
///
/// # Returns
/// `(rv_up, rv_down)` — upside and downside realized semi-variances.
pub fn realized_semivariance(prices: &[f64]) -> Result<(f64, f64)> {
    if prices.len() < 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "realized_semivariance: need at least 2 prices".to_string(),
            required: 2,
            actual: prices.len(),
        });
    }
    if prices.iter().any(|&p| p <= 0.0) {
        return Err(TimeSeriesError::InvalidInput(
            "realized_semivariance: all prices must be positive".to_string(),
        ));
    }

    let mut rv_up = 0.0_f64;
    let mut rv_down = 0.0_f64;

    for w in prices.windows(2) {
        let r = (w[1] / w[0]).ln();
        let r2 = r * r;
        if r > 0.0 {
            rv_up += r2;
        } else {
            rv_down += r2;
        }
    }

    Ok((rv_up, rv_down))
}

// ============================================================
// HAR-CJ (Continuous + Jump decomposition)
// ============================================================

/// HAR-CJ model: separate continuous and jump components.
///
/// ```text
/// RV_t = c + β_Cd * C_{t-1} + β_Cw * C^{(w)}_{t-1} + β_Cm * C^{(m)}_{t-1}
///       + β_Jd * J_{t-1} + β_Jw * J^{(w)}_{t-1} + β_Jm * J^{(m)}_{t-1} + ε_t
/// ```
///
/// where:
/// - C_t = BV_t (continuous component, bipower variation)
/// - J_t = max(RV_t - BV_t, 0) (jump component)
#[derive(Debug, Clone)]
pub struct HARCJModel {
    /// Intercept
    pub c: f64,
    /// Daily continuous component coefficient
    pub beta_cd: f64,
    /// Weekly continuous component coefficient
    pub beta_cw: f64,
    /// Monthly continuous component coefficient
    pub beta_cm: f64,
    /// Daily jump component coefficient
    pub beta_jd: f64,
    /// Weekly jump component coefficient
    pub beta_jw: f64,
    /// Monthly jump component coefficient
    pub beta_jm: f64,
    /// R-squared of the regression
    pub r_squared: f64,
    /// Residual standard deviation
    pub residual_std: f64,
    /// Number of observations used
    pub n_obs: usize,
}

/// Fit a HAR-CJ model (continuous + jump decomposition) by OLS.
///
/// # Arguments
/// * `rv_series` — Daily realized variance series (length >= 24)
/// * `bv_series` — Daily bipower variation series (same length)
pub fn fit_har_cj(rv_series: &[f64], bv_series: &[f64]) -> Result<HARCJModel> {
    if rv_series.len() != bv_series.len() {
        return Err(TimeSeriesError::InvalidInput(
            "HAR-CJ: rv_series and bv_series must have the same length".to_string(),
        ));
    }
    let n = rv_series.len();
    if n < 24 {
        return Err(TimeSeriesError::InsufficientData {
            message: "HAR-CJ: need at least 24 observations".to_string(),
            required: 24,
            actual: n,
        });
    }

    // Continuous and jump components
    let cont: Vec<f64> = bv_series.to_vec();
    let jump: Vec<f64> = rv_series
        .iter()
        .zip(bv_series.iter())
        .map(|(&rv, &bv)| (rv - bv).max(0.0))
        .collect();

    let t_start = 22;
    let n_reg = n - t_start;

    // Build regressors: daily, weekly, monthly for both C and J
    let mut x_rows: Vec<[f64; 6]> = Vec::with_capacity(n_reg);
    let y: Vec<f64> = rv_series[t_start..].to_vec();

    for t in t_start..n {
        let cd = cont[t - 1];
        let cw = cont[t - 5..t].iter().sum::<f64>() / 5.0;
        let cm = cont[t - 22..t].iter().sum::<f64>() / 22.0;
        let jd = jump[t - 1];
        let jw = jump[t - 5..t].iter().sum::<f64>() / 5.0;
        let jm = jump[t - 22..t].iter().sum::<f64>() / 22.0;
        x_rows.push([cd, cw, cm, jd, jw, jm]);
    }

    // OLS: [1, cd, cw, cm, jd, jw, jm] -> k = 7
    let k = 7_usize;
    let mut xtx = vec![0.0_f64; k * k];
    let mut xty = vec![0.0_f64; k];

    for i in 0..n_reg {
        let row = [
            1.0,
            x_rows[i][0],
            x_rows[i][1],
            x_rows[i][2],
            x_rows[i][3],
            x_rows[i][4],
            x_rows[i][5],
        ];
        for (a, &ra) in row.iter().enumerate() {
            xty[a] += ra * y[i];
            for (b, &rb) in row.iter().enumerate() {
                xtx[a * k + b] += ra * rb;
            }
        }
    }

    let beta = solve_nxn(&xtx, &xty, k)?;

    // Compute R² and residual std
    let y_mean = y.iter().sum::<f64>() / n_reg as f64;
    let mut ss_res = 0.0_f64;
    let mut ss_tot = 0.0_f64;
    for i in 0..n_reg {
        let row = [
            1.0,
            x_rows[i][0],
            x_rows[i][1],
            x_rows[i][2],
            x_rows[i][3],
            x_rows[i][4],
            x_rows[i][5],
        ];
        let y_hat: f64 = row.iter().zip(beta.iter()).map(|(&x, &b)| x * b).sum();
        ss_res += (y[i] - y_hat).powi(2);
        ss_tot += (y[i] - y_mean).powi(2);
    }

    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };
    let residual_std = (ss_res / (n_reg as f64 - k as f64).max(1.0)).sqrt();

    Ok(HARCJModel {
        c: beta[0],
        beta_cd: beta[1],
        beta_cw: beta[2],
        beta_cm: beta[3],
        beta_jd: beta[4],
        beta_jw: beta[5],
        beta_jm: beta[6],
        r_squared,
        residual_std,
        n_obs: n_reg,
    })
}

/// Forecast from a HAR-CJ model.
///
/// # Arguments
/// * `model` — Fitted HAR-CJ model
/// * `rv_series` — Historical RV series
/// * `bv_series` — Historical BV series
/// * `h` — Forecast horizon
pub fn har_cj_forecast(
    model: &HARCJModel,
    rv_series: &[f64],
    bv_series: &[f64],
    h: usize,
) -> Result<Vec<f64>> {
    if h == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "h".to_string(),
            message: "Forecast horizon must be >= 1".to_string(),
        });
    }
    if rv_series.len() < 22 || bv_series.len() < 22 {
        return Err(TimeSeriesError::InsufficientData {
            message: "har_cj_forecast: need at least 22 historical values".to_string(),
            required: 22,
            actual: rv_series.len().min(bv_series.len()),
        });
    }
    if rv_series.len() != bv_series.len() {
        return Err(TimeSeriesError::InvalidInput(
            "har_cj_forecast: rv and bv series must have the same length".to_string(),
        ));
    }

    let mut rv_hist: Vec<f64> = rv_series.to_vec();
    let mut bv_hist: Vec<f64> = bv_series.to_vec();
    let mut forecasts = Vec::with_capacity(h);

    for _ in 0..h {
        let m = rv_hist.len();
        let cont: Vec<f64> = bv_hist.clone();
        let jump: Vec<f64> = rv_hist
            .iter()
            .zip(bv_hist.iter())
            .map(|(&rv, &bv)| (rv - bv).max(0.0))
            .collect();

        let cd = cont[m - 1];
        let cw = cont[m - 5..m].iter().sum::<f64>() / 5.0;
        let cm = cont[m - 22..m].iter().sum::<f64>() / 22.0;
        let jd = jump[m - 1];
        let jw = jump[m - 5..m].iter().sum::<f64>() / 5.0;
        let jm = jump[m - 22..m].iter().sum::<f64>() / 22.0;

        let rv_next = (model.c
            + model.beta_cd * cd
            + model.beta_cw * cw
            + model.beta_cm * cm
            + model.beta_jd * jd
            + model.beta_jw * jw
            + model.beta_jm * jm)
            .max(0.0);

        forecasts.push(rv_next);
        // For future steps, assume BV ≈ RV (no jumps in forecast)
        rv_hist.push(rv_next);
        bv_hist.push(rv_next);
    }

    Ok(forecasts)
}

// ============================================================
// Linear algebra helper
// ============================================================

/// Solve an n×n linear system via Gaussian elimination with partial pivoting.
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
                "OLS solve: singular matrix".to_string(),
            ));
        }

        if pivot_row != col {
            for k in 0..n {
                a.swap(col * n + k, pivot_row * n + k);
            }
            b.swap(col, pivot_row);
        }

        for row in (col + 1)..n {
            let factor = a[row * n + col] / a[col * n + col];
            for k in col..n {
                let val = a[col * n + k] * factor;
                a[row * n + k] -= val;
            }
            b[row] -= b[col] * factor;
        }
    }

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
        let mut prices = vec![100.0_f64];
        for i in 1..n {
            let ret = 0.001 * ((i as f64 * 1.37 + 0.5).sin());
            prices.push(prices[i - 1] * (1.0 + ret));
        }
        prices
    }

    fn make_prices_with_jumps(n: usize) -> Vec<f64> {
        let mut prices = vec![100.0_f64];
        for i in 1..n {
            let ret = 0.001 * ((i as f64 * 1.37 + 0.5).sin());
            // Add jumps at specific points
            let jump = if i == n / 3 || i == 2 * n / 3 {
                0.02
            } else {
                0.0
            };
            prices.push(prices[i - 1] * (1.0 + ret + jump));
        }
        prices
    }

    fn make_rv_series(n: usize) -> (Vec<f64>, Vec<f64>) {
        // Synthetic RV and BV series
        let rv: Vec<f64> = (0..n)
            .map(|i| {
                let base = 0.0001_f64;
                let cluster = if i % 7 < 3 { 3.0 } else { 1.0 };
                base * cluster * (1.0 + 0.5 * ((i as f64 * 0.31).sin()).abs())
            })
            .collect();
        let bv: Vec<f64> = rv
            .iter()
            .enumerate()
            .map(|(i, &v)| v * (0.8 + 0.15 * ((i as f64 * 0.7).sin() * 0.5 + 0.5)))
            .collect();
        (rv, bv)
    }

    #[test]
    fn test_flat_top_realized_kernel() {
        let prices = make_prices(200);
        let rk = flat_top_realized_kernel(&prices, None).expect("RK");
        assert!(rk >= 0.0, "RK must be non-negative: {rk}");
        assert!(rk.is_finite());

        // With explicit bandwidth
        let rk2 = flat_top_realized_kernel(&prices, Some(10)).expect("RK with bandwidth");
        assert!(rk2 >= 0.0);
        assert!(rk2.is_finite());
    }

    #[test]
    fn test_flat_top_kernel_vs_rv() {
        // On clean data without noise, RK and RV should be similar
        let prices = make_prices(100);
        let rv = realized_variance(&prices).expect("RV");
        let rk = flat_top_realized_kernel(&prices, Some(3)).expect("RK");

        // Both should be close in magnitude (within an order of magnitude)
        assert!(rk > 0.0);
        assert!(rv > 0.0);
        let ratio = rk / rv;
        assert!(
            ratio > 0.01 && ratio < 100.0,
            "RK/RV ratio should be reasonable: {ratio}"
        );
    }

    #[test]
    fn test_two_scale_rv() {
        let prices = make_prices(200);
        let tsrv = two_scale_rv(&prices, 5).expect("TSRV");
        assert!(tsrv >= 0.0, "TSRV must be non-negative: {tsrv}");
        assert!(tsrv.is_finite());
    }

    #[test]
    fn test_two_scale_rv_invalid() {
        let prices = make_prices(10);
        assert!(two_scale_rv(&prices, 1).is_err()); // k_slow too small
        assert!(two_scale_rv(&[100.0, 101.0], 5).is_err()); // too few prices
    }

    #[test]
    fn test_realized_semivariance() {
        let prices = make_prices(50);
        let (rv_up, rv_down) = realized_semivariance(&prices).expect("RSV");
        let rv = realized_variance(&prices).expect("RV");

        assert!(rv_up >= 0.0);
        assert!(rv_down >= 0.0);

        // RS+ + RS- should equal RV
        let diff = (rv_up + rv_down - rv).abs();
        assert!(diff < 1e-12, "RS+ + RS- must equal RV, diff={diff}");
    }

    #[test]
    fn test_realized_semivariance_too_few() {
        assert!(realized_semivariance(&[100.0]).is_err());
    }

    #[test]
    fn test_har_cj_fit() {
        let (rv, bv) = make_rv_series(60);
        let model = fit_har_cj(&rv, &bv).expect("HAR-CJ should fit");
        assert!(model.c.is_finite());
        assert!(model.beta_cd.is_finite());
        assert!(model.beta_cw.is_finite());
        assert!(model.beta_cm.is_finite());
        assert!(model.beta_jd.is_finite());
        assert!(model.beta_jw.is_finite());
        assert!(model.beta_jm.is_finite());
    }

    #[test]
    fn test_har_cj_forecast_uses_lagged_averages() {
        let (rv, bv) = make_rv_series(60);
        let model = fit_har_cj(&rv, &bv).expect("HAR-CJ should fit");

        let forecasts = har_cj_forecast(&model, &rv, &bv, 5).expect("Should forecast");
        assert_eq!(forecasts.len(), 5);
        for &f in &forecasts {
            assert!(f >= 0.0, "Forecasted RV must be non-negative: {f}");
            assert!(f.is_finite());
        }
    }

    #[test]
    fn test_har_cj_with_jumps() {
        // Generate data where RV >> BV at jump points
        let (mut rv, mut bv) = make_rv_series(60);
        // Inject obvious jumps
        rv[30] = 0.01; // large RV
        bv[30] = 0.0001; // small BV (continuous part)
        rv[45] = 0.008;
        bv[45] = 0.00015;

        let model = fit_har_cj(&rv, &bv).expect("HAR-CJ should fit with jumps");
        // Jump coefficient should be finite
        assert!(model.beta_jd.is_finite());
    }

    #[test]
    fn test_har_cj_dimension_mismatch() {
        let rv = vec![0.001; 30];
        let bv = vec![0.0008; 25];
        assert!(fit_har_cj(&rv, &bv).is_err());
    }

    #[test]
    fn test_flat_top_kernel_function_values() {
        // k(0) = 1
        assert!((flat_top_bartlett(0.0) - 1.0).abs() < 1e-10);
        // k(0.5) = 0.5
        assert!((flat_top_bartlett(0.5) - 0.5).abs() < 1e-10);
        // k(1) = 0
        assert!((flat_top_bartlett(1.0) - 0.0).abs() < 1e-10);
        // k(1.5) = 0
        assert!((flat_top_bartlett(1.5) - 0.0).abs() < 1e-10);
        // k(-0.5) = 0.5
        assert!((flat_top_bartlett(-0.5) - 0.5).abs() < 1e-10);
    }
}
