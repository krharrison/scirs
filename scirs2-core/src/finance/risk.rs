//! Risk metrics for financial portfolios
//!
//! Implements historical Value-at-Risk (VaR), Conditional VaR (CVaR / Expected Shortfall),
//! Sharpe ratio, Sortino ratio, maximum drawdown, and rolling Sharpe ratio.

use crate::error::{CoreError, CoreResult};

// ============================================================
// Value at Risk (VaR) — historical simulation
// ============================================================

/// Historical Value-at-Risk (VaR) at the specified confidence level.
///
/// Sorts the return series and returns the (1-confidence) quantile negated so
/// that VaR is reported as a **positive loss** (a loss of `VaR` means the portfolio
/// loses at least this much with probability `1-confidence`).
///
/// # Arguments
/// * `returns` - Slice of period returns (e.g. daily log-returns)
/// * `confidence` - Confidence level, e.g. 0.95 for 95% VaR
///
/// # Errors
/// Returns [`CoreError::InvalidArgument`] if `returns` is empty or
/// `confidence` is not in (0, 1).
pub fn value_at_risk(returns: &[f64], confidence: f64) -> CoreResult<f64> {
    validate_returns_and_confidence(returns, confidence)?;

    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let idx = ((alpha * sorted.len() as f64).floor() as usize).min(sorted.len() - 1);
    // VaR is a positive loss magnitude
    Ok(-sorted[idx])
}

/// Conditional Value-at-Risk (CVaR), also known as Expected Shortfall (ES).
///
/// The average loss in the worst `(1-confidence)` fraction of outcomes.
///
/// # Arguments
/// * `returns` - Slice of period returns
/// * `confidence` - Confidence level in (0, 1)
///
/// # Errors
/// Returns [`CoreError::InvalidArgument`] for empty returns or out-of-range confidence.
pub fn conditional_var(returns: &[f64], confidence: f64) -> CoreResult<f64> {
    validate_returns_and_confidence(returns, confidence)?;

    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let cutoff_idx = ((alpha * sorted.len() as f64).floor() as usize).min(sorted.len() - 1);

    // Average of returns at and below the VaR quantile
    let tail: &[f64] = &sorted[..=cutoff_idx];
    if tail.is_empty() {
        return Ok(-sorted[0]);
    }

    let mean_tail: f64 = tail.iter().sum::<f64>() / tail.len() as f64;
    Ok(-mean_tail)
}

// ============================================================
// Sharpe ratio
// ============================================================

/// Annualised Sharpe ratio of a return series.
///
/// `Sharpe = (mean(returns) - risk_free) / std_dev(returns) * sqrt(252)`
///
/// This function **does not** annualise the risk-free rate — pass the
/// per-period risk-free rate consistent with the return frequency
/// (e.g. daily risk-free = annual_rate / 252).
///
/// # Arguments
/// * `returns` - Slice of period excess-to-base returns
/// * `risk_free` - Per-period risk-free rate (same frequency as `returns`)
///
/// # Errors
/// Returns [`CoreError::InvalidArgument`] if `returns` has fewer than 2 elements.
/// Returns [`CoreError::ComputationError`] if return standard deviation is zero.
pub fn sharpe_ratio(returns: &[f64], risk_free: f64) -> CoreResult<f64> {
    if returns.len() < 2 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "At least 2 return observations are required to compute Sharpe ratio",
        )));
    }

    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let variance = returns.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / (n - 1.0);

    if variance < 1e-20 {
        return Err(CoreError::ComputationError(
            crate::error::ErrorContext::new(
                "Return standard deviation is (near) zero; Sharpe ratio undefined",
            ),
        ));
    }

    let std_dev = variance.sqrt();
    let excess_return = mean - risk_free;
    // Annualise assuming daily returns (252 trading days)
    Ok(excess_return / std_dev * (252.0_f64).sqrt())
}

// ============================================================
// Sortino ratio
// ============================================================

/// Annualised Sortino ratio, penalising only downside volatility.
///
/// `Sortino = (mean(returns) - risk_free) / downside_std * sqrt(252)`
///
/// Downside deviation uses a minimum acceptable return (MAR) equal to `risk_free`.
///
/// # Arguments
/// * `returns` - Slice of period returns
/// * `risk_free` - Per-period minimum acceptable return (MAR)
///
/// # Errors
/// Returns [`CoreError::InvalidArgument`] for fewer than 2 observations.
/// Returns [`CoreError::ComputationError`] if downside deviation is zero (no losses).
pub fn sortino_ratio(returns: &[f64], risk_free: f64) -> CoreResult<f64> {
    if returns.len() < 2 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "At least 2 return observations are required to compute Sortino ratio",
        )));
    }

    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;

    // Downside deviation: square root of mean squared negative deviations below MAR
    let downside_sq_sum: f64 = returns
        .iter()
        .map(|r| {
            let dev = r - risk_free;
            if dev < 0.0 {
                dev * dev
            } else {
                0.0
            }
        })
        .sum();

    let downside_var = downside_sq_sum / (n - 1.0);

    if downside_var < 1e-20 {
        return Err(CoreError::ComputationError(
            crate::error::ErrorContext::new(
                "Downside deviation is (near) zero; Sortino ratio undefined (no below-MAR returns)",
            ),
        ));
    }

    let downside_std = downside_var.sqrt();
    let excess_return = mean - risk_free;
    Ok(excess_return / downside_std * (252.0_f64).sqrt())
}

// ============================================================
// Maximum Drawdown
// ============================================================

/// Maximum drawdown of a price (or cumulative return) series.
///
/// Computes the largest peak-to-trough decline as a fraction of the peak value.
/// Returns a value in `[0, 1]` (0 = no drawdown; 1 = complete loss).
///
/// # Arguments
/// * `prices` - Time-ordered series of prices or portfolio values
///
/// # Errors
/// Returns [`CoreError::InvalidArgument`] if `prices` is empty or contains non-positive values.
pub fn max_drawdown(prices: &[f64]) -> CoreResult<f64> {
    if prices.is_empty() {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "Price series must not be empty",
        )));
    }
    for (i, &p) in prices.iter().enumerate() {
        if p <= 0.0 {
            return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
                format!("Price at index {i} is non-positive ({p}); all prices must be > 0"),
            )));
        }
    }

    let mut peak = prices[0];
    let mut max_dd = 0.0_f64;

    for &price in prices.iter().skip(1) {
        if price > peak {
            peak = price;
        }
        let dd = (peak - price) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    Ok(max_dd)
}

// ============================================================
// Rolling Sharpe ratio
// ============================================================

/// Rolling Sharpe ratio over a sliding window.
///
/// Each element `i` (starting from `window - 1`) is the annualised Sharpe ratio
/// computed over returns `[i-window+1 .. i]`.  Positions before the first full
/// window are `f64::NAN`.
///
/// # Arguments
/// * `returns` - Slice of period returns
/// * `window` - Rolling window size (must be ≥ 2)
/// * `risk_free` - Per-period risk-free rate
///
/// # Errors
/// Returns [`CoreError::InvalidArgument`] if `window < 2` or `returns.len() < window`.
pub fn rolling_sharpe(returns: &[f64], window: usize, risk_free: f64) -> CoreResult<Vec<f64>> {
    if window < 2 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "Rolling window must be at least 2",
        )));
    }
    if returns.len() < window {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            format!(
                "Return series length {} is less than window {}",
                returns.len(),
                window
            ),
        )));
    }

    let n = returns.len();
    let mut result = vec![f64::NAN; n];

    for i in (window - 1)..n {
        let slice = &returns[(i + 1 - window)..=i];
        // Compute mean and variance over window slice
        let wn = slice.len() as f64;
        let mean = slice.iter().sum::<f64>() / wn;
        let var = slice.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / (wn - 1.0);
        let sharpe = if var < 1e-20 {
            f64::NAN
        } else {
            (mean - risk_free) / var.sqrt() * (252.0_f64).sqrt()
        };
        result[i] = sharpe;
    }

    Ok(result)
}

// ============================================================
// Internal helpers
// ============================================================

fn validate_returns_and_confidence(returns: &[f64], confidence: f64) -> CoreResult<()> {
    if returns.is_empty() {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "Return series must not be empty",
        )));
    }
    if !(0.0 < confidence && confidence < 1.0) {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            format!("Confidence level must be in (0,1), got {confidence}"),
        )));
    }
    Ok(())
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_returns() -> Vec<f64> {
        vec![
            0.01, -0.02, 0.015, -0.005, 0.008, -0.012, 0.02, -0.018, 0.005, 0.003, -0.025, 0.011,
            0.007, -0.009, 0.016, -0.003, 0.012, -0.007, 0.004, -0.001,
        ]
    }

    // --- VaR ---
    #[test]
    fn test_var_95_basic() {
        let returns = make_returns();
        let var = value_at_risk(&returns, 0.95).expect("should succeed");
        // VaR should be positive and less than the maximum loss
        assert!(var > 0.0, "VaR should be positive: {var}");
        let max_loss = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(var <= max_loss + 1e-10, "VaR should not exceed max_loss");
    }

    #[test]
    fn test_var_empty_returns() {
        assert!(value_at_risk(&[], 0.95).is_err());
    }

    #[test]
    fn test_var_bad_confidence() {
        let r = make_returns();
        assert!(value_at_risk(&r, 0.0).is_err());
        assert!(value_at_risk(&r, 1.0).is_err());
        assert!(value_at_risk(&r, -0.1).is_err());
    }

    #[test]
    fn test_var_increases_with_confidence() {
        let returns = make_returns();
        let var_90 = value_at_risk(&returns, 0.90).expect("should succeed");
        let var_99 = value_at_risk(&returns, 0.99).expect("should succeed");
        // Higher confidence level should give larger (or equal) VaR
        assert!(
            var_99 >= var_90 - 1e-10,
            "99% VaR should >= 90% VaR: {var_99:.6} vs {var_90:.6}"
        );
    }

    // --- CVaR ---
    #[test]
    fn test_cvar_ge_var() {
        let returns = make_returns();
        let var = value_at_risk(&returns, 0.95).expect("should succeed");
        let cvar = conditional_var(&returns, 0.95).expect("should succeed");
        assert!(
            cvar >= var - 1e-10,
            "CVaR should be >= VaR: cvar={cvar:.6} var={var:.6}"
        );
    }

    #[test]
    fn test_cvar_empty_returns() {
        assert!(conditional_var(&[], 0.95).is_err());
    }

    // --- Sharpe ratio ---
    #[test]
    fn test_sharpe_positive_mean() {
        // Use alternating returns so std dev is nonzero but mean is positive
        let returns: Vec<f64> = (0..252)
            .map(|i| if i % 2 == 0 { 0.002 } else { 0.0005 })
            .collect();
        let s = sharpe_ratio(&returns, 0.0).expect("should succeed");
        assert!(
            s > 0.0,
            "Sharpe ratio with positive mean return should be positive: {s}"
        );
    }

    #[test]
    fn test_sharpe_zero_std_error() {
        // Constant returns -> zero std dev -> error
        let returns = vec![0.001; 10];
        assert!(sharpe_ratio(&returns, 0.0).is_err());
    }

    #[test]
    fn test_sharpe_too_few_obs() {
        assert!(sharpe_ratio(&[0.01], 0.0).is_err());
    }

    #[test]
    fn test_sharpe_known_value() {
        // Mean 0.001/day, std 0.01/day -> Sharpe = 0.001/0.01 * sqrt(252) ≈ 1.5874
        let n = 1000;
        let mean = 0.001_f64;
        let std = 0.01_f64;
        // Construct returns with exact mean and approximately correct std
        // alternate +std/-std around mean so std dev is exact
        let returns: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { mean + std } else { mean - std })
            .collect();
        let s = sharpe_ratio(&returns, 0.0).expect("should succeed");
        let expected = mean / std * (252.0_f64).sqrt();
        assert!(
            (s - expected).abs() < 0.05 * expected,
            "Sharpe expected≈{expected:.4} got={s:.4}"
        );
    }

    // --- Sortino ratio ---
    #[test]
    fn test_sortino_positive_skew() {
        // Large positive returns, small negatives -> high Sortino
        let returns: Vec<f64> = (0..100)
            .map(|i| if i % 5 == 0 { -0.001 } else { 0.005 })
            .collect();
        let s = sortino_ratio(&returns, 0.0).expect("should succeed");
        assert!(s > 0.0, "Sortino positive skew: {s}");
    }

    #[test]
    fn test_sortino_no_downside_error() {
        let returns: Vec<f64> = vec![0.001, 0.002, 0.003];
        // Risk-free = 0, no return < 0, downside std = 0
        assert!(sortino_ratio(&returns, 0.0).is_err());
    }

    #[test]
    fn test_sortino_ge_sharpe_positive_skew() {
        // When returns are mostly positive, Sortino >= Sharpe
        let returns: Vec<f64> = (0..100)
            .map(|i| if i % 10 == 0 { -0.002 } else { 0.003 })
            .collect();
        let sh = sharpe_ratio(&returns, 0.0).expect("should succeed");
        let so = sortino_ratio(&returns, 0.0).expect("should succeed");
        assert!(
            so >= sh - 1e-8,
            "Sortino should >= Sharpe for positive-skewed returns: so={so:.4} sh={sh:.4}"
        );
    }

    // --- Max drawdown ---
    #[test]
    fn test_max_drawdown_known() {
        // Prices: 100, 90, 80 -> drawdown = (100-80)/100 = 0.2
        let prices = vec![100.0, 90.0, 80.0];
        let dd = max_drawdown(&prices).expect("should succeed");
        assert!((dd - 0.2).abs() < 1e-10, "Max drawdown: {dd:.6}");
    }

    #[test]
    fn test_max_drawdown_recovery() {
        // 100 -> 50 -> 150: drawdown = 0.5 (the 100->50 drop)
        let prices = vec![100.0, 50.0, 150.0];
        let dd = max_drawdown(&prices).expect("should succeed");
        assert!(
            (dd - 0.5).abs() < 1e-10,
            "Max drawdown with recovery: {dd:.6}"
        );
    }

    #[test]
    fn test_max_drawdown_monotone_rising() {
        // Monotone rise -> drawdown = 0
        let prices: Vec<f64> = (1..=10).map(|i| i as f64 * 10.0).collect();
        let dd = max_drawdown(&prices).expect("should succeed");
        assert_eq!(dd, 0.0);
    }

    #[test]
    fn test_max_drawdown_empty() {
        assert!(max_drawdown(&[]).is_err());
    }

    #[test]
    fn test_max_drawdown_nonpositive_price() {
        assert!(max_drawdown(&[100.0, 0.0, 50.0]).is_err());
        assert!(max_drawdown(&[100.0, -10.0, 50.0]).is_err());
    }

    // --- Rolling Sharpe ---
    #[test]
    fn test_rolling_sharpe_length_matches() {
        let returns = make_returns();
        let rs = rolling_sharpe(&returns, 5, 0.0).expect("should succeed");
        assert_eq!(rs.len(), returns.len());
    }

    #[test]
    fn test_rolling_sharpe_first_window_minus_1_nan() {
        let returns = make_returns();
        let window = 5;
        let rs = rolling_sharpe(&returns, window, 0.0).expect("should succeed");
        for v in rs.iter().take(window - 1) {
            assert!(v.is_nan(), "Values before first full window should be NaN");
        }
    }

    #[test]
    fn test_rolling_sharpe_valid_after_window() {
        let returns = make_returns();
        let window = 5;
        let rs = rolling_sharpe(&returns, window, 0.0).expect("should succeed");
        let valid: Vec<f64> = rs.into_iter().skip(window - 1).collect();
        assert!(
            valid.iter().any(|v| !v.is_nan()),
            "Should have some valid Sharpe values"
        );
    }

    #[test]
    fn test_rolling_sharpe_bad_window() {
        let returns = make_returns();
        assert!(rolling_sharpe(&returns, 1, 0.0).is_err());
        assert!(rolling_sharpe(&returns, 0, 0.0).is_err());
    }

    #[test]
    fn test_rolling_sharpe_window_exceeds_length() {
        let returns = vec![0.01, 0.02, 0.03];
        assert!(rolling_sharpe(&returns, 10, 0.0).is_err());
    }
}
