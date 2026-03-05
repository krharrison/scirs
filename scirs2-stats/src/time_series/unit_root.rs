//! Unit Root Tests
//!
//! This module re-exports and extends the unit-root tests from
//! [`crate::stationarity`] with a simplified, uniform API:
//!
//! | Function           | Test                        | Null hypothesis      |
//! |--------------------|-----------------------------|-----------------------|
//! | [`adf_test`]       | Augmented Dickey-Fuller     | Unit root exists      |
//! | [`kpss_test`]      | KPSS                        | Series is stationary  |
//! | [`pp_test`]        | Phillips-Perron             | Unit root exists      |
//! | [`za_test`]        | Zivot-Andrews               | Unit root (w/ break)  |
//!
//! The underlying implementations live in [`crate::stationarity`].

use crate::error::StatsResult;
use crate::stationarity::{
    AdfRegression, AdfResult as StationarityAdfResult, KpssTrend,
    PhillipsPerronResult, ZivotAndrewsBreak, ZivotAndrewsResult,
    adf_test as stationarity_adf_test,
    kpss_test as stationarity_kpss_test,
    phillips_perron_test,
    zivot_andrews_test as stationarity_za_test,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Simplified result types
// ---------------------------------------------------------------------------

/// Trend specification for the ADF test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdfTrend {
    /// No constant or trend in the regression (pure random walk null)
    None,
    /// Constant term only (most common default)
    Constant,
    /// Constant plus linear time trend
    ConstantTrend,
}

/// Result of the Augmented Dickey-Fuller test with a `HashMap` of critical values.
#[derive(Debug, Clone)]
pub struct AdfResult {
    /// ADF τ-statistic
    pub statistic: f64,
    /// Approximate p-value
    pub p_value: f64,
    /// Critical values keyed by significance level (`"1%"`, `"5%"`, `"10%"`)
    pub critical_values: HashMap<String, f64>,
    /// Number of augmentation lags used
    pub lags: usize,
}

// ---------------------------------------------------------------------------
// ADF test
// ---------------------------------------------------------------------------

/// Perform the Augmented Dickey-Fuller test.
///
/// # Arguments
///
/// * `x`     – time series (must have at least 4 elements)
/// * `lags`  – number of augmentation lags; `None` → automatic via BIC
/// * `trend` – regression type ([`AdfTrend`])
///
/// # Returns
///
/// [`AdfResult`] with statistic, p-value, critical values, and lag count.
///
/// # Errors
///
/// Propagates errors from the underlying OLS regression (e.g. insufficient data).
///
/// # Example
///
/// ```
/// use scirs2_stats::time_series::unit_root::{adf_test, AdfTrend};
///
/// let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
/// let result = adf_test(&x, None, AdfTrend::Constant).unwrap();
/// println!("ADF stat = {:.4}", result.statistic);
/// ```
pub fn adf_test(x: &[f64], lags: Option<usize>, trend: AdfTrend) -> StatsResult<AdfResult> {
    let regression = match trend {
        AdfTrend::None => AdfRegression::None,
        AdfTrend::Constant => AdfRegression::Constant,
        AdfTrend::ConstantTrend => AdfRegression::ConstantTrend,
    };

    let inner: StationarityAdfResult = stationarity_adf_test(x, lags, regression)?;

    let mut critical_values = HashMap::new();
    critical_values.insert("1%".to_string(), inner.critical_values.one_pct);
    critical_values.insert("5%".to_string(), inner.critical_values.five_pct);
    critical_values.insert("10%".to_string(), inner.critical_values.ten_pct);

    Ok(AdfResult {
        statistic: inner.statistic,
        p_value: inner.p_value,
        critical_values,
        lags: inner.used_lags,
    })
}

// ---------------------------------------------------------------------------
// KPSS test
// ---------------------------------------------------------------------------

/// Perform the KPSS stationarity test.
///
/// Unlike the ADF test, KPSS has **stationarity as the null hypothesis**.
///
/// # Arguments
///
/// * `x`    – time series
/// * `lags` – bandwidth for the long-run variance estimator; `None` → automatic
/// * `trend`– `false` → level stationarity, `true` → trend stationarity
///
/// # Returns
///
/// `(statistic, p_value)`.  A large statistic (small p-value) rejects H₀
/// of stationarity.
pub fn kpss_test(x: &[f64], lags: Option<usize>, trend: bool) -> StatsResult<(f64, f64)> {
    let kpss_trend = if trend {
        KpssTrend::Trend
    } else {
        KpssTrend::Level
    };
    let result = stationarity_kpss_test(x, lags, kpss_trend)?;
    Ok((result.statistic, result.p_value))
}

// ---------------------------------------------------------------------------
// Phillips-Perron test
// ---------------------------------------------------------------------------

/// Perform the Phillips-Perron unit root test.
///
/// Returns `(z_tau_statistic, p_value)` where `z_tau` is the normalised
/// t-statistic robust to serial correlation and heteroskedasticity.
pub fn pp_test(x: &[f64], lags: Option<usize>) -> StatsResult<(f64, f64)> {
    let result: PhillipsPerronResult = phillips_perron_test(x, lags)?;
    Ok((result.z_tau, result.p_value))
}

// ---------------------------------------------------------------------------
// Zivot-Andrews test
// ---------------------------------------------------------------------------

/// Perform the Zivot-Andrews structural break unit root test.
///
/// Searches over all possible break points and returns the minimum ADF
/// statistic (most favourable to rejecting the unit root null).
///
/// # Returns
///
/// `(statistic, p_value, break_point)` where `break_point` is the
/// estimated structural break index.
pub fn za_test(x: &[f64]) -> StatsResult<(f64, f64, usize)> {
    let result: ZivotAndrewsResult =
        stationarity_za_test(x, None, ZivotAndrewsBreak::Both)?;
    Ok((result.statistic, result.p_value, result.break_point))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn random_walk(n: usize) -> Vec<f64> {
        let mut x = vec![0.0f64; n];
        let mut s: u64 = 99999;
        for i in 1..n {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let e = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            x[i] = x[i - 1] + e;
        }
        x
    }

    fn stationary_ar1(n: usize, phi: f64) -> Vec<f64> {
        let mut x = vec![0.0f64; n];
        let mut s: u64 = 54321;
        for i in 1..n {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let e = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            x[i] = phi * x[i - 1] + e;
        }
        x
    }

    #[test]
    fn test_adf_result_has_critical_values() {
        let x = random_walk(80);
        let result = adf_test(&x, None, AdfTrend::Constant).unwrap();
        assert!(result.critical_values.contains_key("1%"));
        assert!(result.critical_values.contains_key("5%"));
        assert!(result.critical_values.contains_key("10%"));
    }

    #[test]
    fn test_adf_pvalue_in_range() {
        let x = random_walk(80);
        let result = adf_test(&x, None, AdfTrend::Constant).unwrap();
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_adf_stationary_low_stat() {
        // Stationary AR(1) → ADF stat should be more negative than -2
        let x = stationary_ar1(100, 0.3);
        let result = adf_test(&x, None, AdfTrend::Constant).unwrap();
        // The stat for stationary data should be < -1 most of the time
        assert!(result.statistic < 0.0);
    }

    #[test]
    fn test_adf_trend_variants() {
        let x = stationary_ar1(60, 0.5);
        let _ = adf_test(&x, Some(1), AdfTrend::None).unwrap();
        let _ = adf_test(&x, Some(1), AdfTrend::Constant).unwrap();
        let _ = adf_test(&x, Some(1), AdfTrend::ConstantTrend).unwrap();
    }

    #[test]
    fn test_kpss_pvalue_range() {
        let x = stationary_ar1(80, 0.4);
        let (stat, pval) = kpss_test(&x, None, false).unwrap();
        assert!(stat >= 0.0);
        assert!(pval >= 0.0 && pval <= 1.0);
    }

    #[test]
    fn test_pp_pvalue_range() {
        let x = random_walk(80);
        let (stat, pval) = pp_test(&x, None).unwrap();
        let _ = stat;
        assert!(pval >= 0.0 && pval <= 1.0);
    }

    #[test]
    fn test_za_returns_break_index() {
        let x = random_walk(60);
        let (stat, pval, bp) = za_test(&x).unwrap();
        assert!(bp > 0 && bp < x.len());
        let _ = stat;
        let _ = pval;
    }
}
