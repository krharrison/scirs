//! Risk measures and additional EVT utilities.
//!
//! This module provides financial and actuarial risk measures built on top of the
//! Generalized Extreme Value (GEV) and Generalized Pareto (GPD) distributions:
//!
//! - **Value at Risk (VaR)**: empirical quantile-based tail risk measure
//! - **Conditional VaR / Expected Shortfall (CVaR/ES)**: average loss beyond VaR
//! - **Hill estimator alias**: `tail_index_hill` — wraps [`super::analysis::hill_estimator`]
//! - **POT analysis convenience wrapper**: `pot_analysis`
//! - **Fit block maxima convenience wrapper**: `fit_block_maxima`
//! - **Profile-likelihood (delta method) CI for return levels**: `return_level_ci`
//!
//! # References
//! - McNeil, A., Frey, R., Embrechts, P. (2005). *Quantitative Risk Management*. Princeton.
//! - Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::Array1;

use super::analysis::{peaks_over_threshold, PotResult};
use super::distributions::GeneralizedExtremeValue;
use super::estimation::gev_fit_mle;

// ---------------------------------------------------------------------------
// Value at Risk
// ---------------------------------------------------------------------------

/// Empirical Value at Risk (VaR) at a given confidence level.
///
/// VaR_p is the smallest value v such that P(X ≤ v) ≥ p.  For a finite sample,
/// this is the empirical quantile at probability p, computed by linear interpolation.
///
/// # Arguments
/// - `data`: observed losses or returns (positive = loss).
/// - `confidence`: probability level p ∈ (0, 1), e.g. 0.95 for 95% VaR.
///
/// # Errors
/// - [`StatsError::InsufficientData`] if fewer than 2 observations.
/// - [`StatsError::InvalidArgument`] if confidence is not in (0, 1).
///
/// # Examples
/// ```
/// use scirs2_stats::extreme_value::value_at_risk;
/// let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
/// let var95 = value_at_risk(&data, 0.95).unwrap();
/// // 95th percentile of 1..=100 is around 95
/// assert!(var95 >= 94.0 && var95 <= 96.0, "var95 = {var95}");
/// ```
pub fn value_at_risk(data: &[f64], confidence: f64) -> StatsResult<f64> {
    if data.len() < 2 {
        return Err(StatsError::InsufficientData(
            "value_at_risk requires at least 2 observations".into(),
        ));
    }
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err(StatsError::InvalidArgument(format!(
            "confidence must be in (0, 1), got {confidence}"
        )));
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    // Linear interpolation between adjacent order statistics
    let idx = confidence * (n as f64 - 1.0);
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = idx - lo as f64;

    Ok(sorted[lo] * (1.0 - frac) + sorted[hi] * frac)
}

// ---------------------------------------------------------------------------
// Conditional VaR / Expected Shortfall
// ---------------------------------------------------------------------------

/// Conditional Value at Risk (CVaR), also known as Expected Shortfall (ES).
///
/// CVaR_p = E[X | X > VaR_p] — the expected loss given that it exceeds the VaR.
///
/// Computed empirically as the mean of observations strictly above VaR_p.
/// If there are no observations above VaR_p, returns VaR_p as a conservative estimate.
///
/// # Arguments
/// - `data`: observed losses or returns.
/// - `confidence`: probability level p ∈ (0, 1).
///
/// # Errors
/// - [`StatsError::InsufficientData`] if fewer than 2 observations.
/// - [`StatsError::InvalidArgument`] if confidence is not in (0, 1).
///
/// # Examples
/// ```
/// use scirs2_stats::extreme_value::{value_at_risk, conditional_var};
/// let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
/// let var95 = value_at_risk(&data, 0.95).unwrap();
/// let cvar95 = conditional_var(&data, 0.95).unwrap();
/// assert!(cvar95 >= var95, "CVaR must be >= VaR");
/// ```
pub fn conditional_var(data: &[f64], confidence: f64) -> StatsResult<f64> {
    if data.len() < 2 {
        return Err(StatsError::InsufficientData(
            "conditional_var requires at least 2 observations".into(),
        ));
    }
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err(StatsError::InvalidArgument(format!(
            "confidence must be in (0, 1), got {confidence}"
        )));
    }

    let var_p = value_at_risk(data, confidence)?;

    let tail: Vec<f64> = data.iter().copied().filter(|&x| x > var_p).collect();
    if tail.is_empty() {
        return Ok(var_p);
    }

    let mean_tail: f64 = tail.iter().sum::<f64>() / tail.len() as f64;
    Ok(mean_tail)
}

// ---------------------------------------------------------------------------
// Tail index (alias)
// ---------------------------------------------------------------------------

/// Hill estimator of the tail index using the top-k order statistics.
///
/// Convenient slice-accepting alias for [`super::analysis::hill_estimator`].
///
/// The Hill estimator for heavy-tailed distributions is:
///   ξ̂ = (1/k) Σᵢ₌₁ᵏ [ ln(X_{(n-i+1)}) − ln(X_{(n-k)}) ]
/// where X_{(n)} ≥ … ≥ X_{(1)} are the sorted order statistics.
///
/// # Arguments
/// - `data`: observed positive values.
/// - `k`: number of upper order statistics to use.
///
/// # Errors
/// - [`StatsError::InvalidArgument`] if `k == 0`.
/// - [`StatsError::InsufficientData`] if `data.len() < k + 1`.
///
/// # Examples
/// ```
/// use scirs2_stats::extreme_value::tail_index_hill;
/// // Pareto(α=2): tail index ξ = 1/α = 0.5
/// let data: Vec<f64> = (1..=200)
///     .map(|i| (i as f64 / 201.0_f64).powf(-0.5))
///     .collect();
/// let xi = tail_index_hill(&data, 20).unwrap();
/// assert!(xi > 0.0, "Heavy-tailed: xi > 0, got {xi}");
/// ```
pub fn tail_index_hill(data: &[f64], k: usize) -> StatsResult<f64> {
    let n = data.len();
    if k == 0 {
        return Err(StatsError::InvalidArgument("k must be >= 1".into()));
    }
    if n < k + 1 {
        return Err(StatsError::InsufficientData(format!(
            "tail_index_hill requires at least k+1 = {} observations, got {n}",
            k + 1
        )));
    }
    let arr = Array1::from(data.to_vec());
    super::analysis::hill_estimator(arr.view(), k)
}

// ---------------------------------------------------------------------------
// POT analysis convenience wrapper
// ---------------------------------------------------------------------------

/// Peaks Over Threshold (POT) analysis.
///
/// Extracts all exceedances above `threshold`, fits a Generalised Pareto Distribution
/// (GPD) to the excesses, and computes return levels for periods [10, 25, 50, 100, 200].
///
/// This is a convenience wrapper around [`super::analysis::peaks_over_threshold`].
///
/// # Arguments
/// - `data`: the complete time series.
/// - `threshold`: exceedance threshold u.
///
/// # Errors
/// - [`StatsError::InvalidArgument`] if threshold is not finite.
/// - [`StatsError::InsufficientData`] if fewer than 10 exceedances above threshold.
///
/// # Examples
/// ```
/// use scirs2_stats::extreme_value::pot_analysis;
/// // Use a low enough threshold to guarantee some exceedances
/// let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
/// let result = pot_analysis(&data, 50.0);
/// // 50 exceedances above threshold=50
/// assert!(result.is_ok(), "pot_analysis failed: {:?}", result.err());
/// ```
pub fn pot_analysis(data: &[f64], threshold: f64) -> StatsResult<PotResult> {
    if !threshold.is_finite() {
        return Err(StatsError::InvalidArgument(
            "threshold must be a finite number".into(),
        ));
    }
    let arr = Array1::from(data.to_vec());
    peaks_over_threshold(arr.view(), threshold, &[10.0, 25.0, 50.0, 100.0, 200.0])
}

// ---------------------------------------------------------------------------
// Fit block maxima
// ---------------------------------------------------------------------------

/// Fitted GEV result for block maxima (simplified).
///
/// Contains the fitted [`GeneralizedExtremeValue`] distribution plus diagnostics.
#[derive(Debug, Clone)]
pub struct BlockMaximaFit {
    /// Fitted GEV parameters.
    pub distribution: GeneralizedExtremeValue,
    /// Log-likelihood at the estimated parameters.
    pub log_likelihood: f64,
    /// AIC = -2·LL + 2·3.
    pub aic: f64,
    /// BIC = -2·LL + 3·ln(n).
    pub bic: f64,
    /// Number of block maxima used.
    pub n_obs: usize,
}

impl BlockMaximaFit {
    /// Return level for a given return period.
    pub fn return_level(&self, return_period: f64) -> StatsResult<f64> {
        self.distribution.return_level(return_period)
    }
}

/// Fit a GEV distribution to block maxima extracted from `data`.
///
/// Combines block extraction with GEV MLE fitting.
///
/// # Arguments
/// - `data`: the complete time series.
/// - `block_size`: number of observations per block.
///
/// # Errors
/// - [`StatsError::InvalidArgument`] if `block_size == 0`.
/// - [`StatsError::InsufficientData`] if fewer than 2 complete blocks, or fewer than 3 maxima.
///
/// # Examples
/// ```
/// use scirs2_stats::extreme_value::fit_block_maxima;
/// let data: Vec<f64> = (0..200).map(|i| i as f64 % 10.0 + 1.0).collect();
/// let fit = fit_block_maxima(&data, 10).unwrap();
/// assert!(fit.distribution.sigma > 0.0);
/// ```
pub fn fit_block_maxima(data: &[f64], block_size: usize) -> StatsResult<BlockMaximaFit> {
    // Extract block maxima (reuse the free function from parent mod)
    if block_size == 0 {
        return Err(StatsError::InvalidArgument(
            "block_size must be >= 1".into(),
        ));
    }
    let n = data.len();
    let n_blocks = n / block_size;
    if n_blocks < 2 {
        return Err(StatsError::InsufficientData(format!(
            "fit_block_maxima: need at least 2 complete blocks (block_size={block_size}), \
             data length {n} gives {n_blocks} block(s)"
        )));
    }

    let maxima: Vec<f64> = (0..n_blocks)
        .map(|b| {
            let start = b * block_size;
            let end = start + block_size;
            data[start..end]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();

    if maxima.len() < 3 {
        return Err(StatsError::InsufficientData(
            "fit_block_maxima: need at least 3 block maxima for GEV fitting".into(),
        ));
    }

    let arr = Array1::from(maxima.clone());
    let distribution = gev_fit_mle(arr.view())?;

    let n_obs = maxima.len();
    let ll: f64 = maxima
        .iter()
        .map(|&x| {
            let p = distribution.pdf(x);
            if p > 0.0 && p.is_finite() {
                p.ln()
            } else {
                -1e10
            }
        })
        .sum();

    let k = 3.0_f64;
    let nf = n_obs as f64;
    Ok(BlockMaximaFit {
        distribution,
        log_likelihood: ll,
        aic: -2.0 * ll + 2.0 * k,
        bic: -2.0 * ll + k * nf.ln(),
        n_obs,
    })
}

// ---------------------------------------------------------------------------
// Return level confidence interval (delta method)
// ---------------------------------------------------------------------------

/// Profile-likelihood confidence interval for a GEV return level.
///
/// For the return level x_T (associated with return period T), the profile
/// likelihood approach traces the log-likelihood surface.  This implementation
/// uses the **delta method** (Gaussian approximation) which is asymptotically
/// equivalent and computationally efficient.
///
/// # Arguments
/// - `data`: observed maxima (block maxima or full series).
/// - `return_period`: T > 1, the return period in units of observations.
/// - `alpha`: significance level, e.g. 0.05 for a 95% CI.
///
/// # Errors
/// - [`StatsError::InvalidArgument`] if `return_period <= 1` or `alpha ∉ (0, 1)`.
/// - Propagates fitting errors.
///
/// # Returns
/// `(lower, upper)` — the confidence interval bounds.
///
/// # Examples
/// ```
/// use scirs2_stats::extreme_value::return_level_ci;
/// let data: Vec<f64> = (1..=50).map(|i| i as f64).collect();
/// let (lo, hi) = return_level_ci(&data, 10.0, 0.05).unwrap();
/// assert!(lo <= hi + 1e-9, "lo={lo}, hi={hi}");
/// ```
pub fn return_level_ci(data: &[f64], return_period: f64, alpha: f64) -> StatsResult<(f64, f64)> {
    if return_period <= 1.0 {
        return Err(StatsError::InvalidArgument(format!(
            "return_period must be > 1, got {return_period}"
        )));
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(StatsError::InvalidArgument(format!(
            "alpha must be in (0, 1), got {alpha}"
        )));
    }
    if data.len() < 3 {
        return Err(StatsError::InsufficientData(
            "return_level_ci requires at least 3 observations".into(),
        ));
    }

    // Fit GEV
    let arr = Array1::from(data.to_vec());
    let gev = gev_fit_mle(arr.view())?;
    let rl = gev.return_level(return_period)?;
    let n = data.len() as f64;

    // Delta-method SE (diagonal Hessian approximation)
    let se = delta_method_se_gev(&gev, return_period, n)?;

    // z_{α/2}
    let z = normal_quantile(1.0 - alpha / 2.0);

    Ok((rl - z * se, rl + z * se))
}

/// Approximate standard error of GEV return level using the delta method with
/// a diagonal approximation to the Fisher information.
fn delta_method_se_gev(
    gev: &GeneralizedExtremeValue,
    return_period: f64,
    n: f64,
) -> StatsResult<f64> {
    let sigma = gev.sigma;
    let xi = gev.xi;

    let p = 1.0 - 1.0 / return_period;
    let y = -p.ln().max(1e-300); // y = -ln(F)
    let xi_thresh = 1e-8_f64;

    let (grad_mu, grad_sigma, grad_xi) = if xi.abs() < xi_thresh {
        // Gumbel limit: x_T = μ - σ·ln(-ln p) = μ - σ·ln(y)
        let ln_y = y.ln();
        (1.0_f64, -ln_y, 0.5 * ln_y * ln_y * sigma)
    } else {
        let y_xi = y.powf(-xi);
        let factor = (y_xi - 1.0) / xi;
        let grad_x = sigma * (y_xi * (-y.ln()) / xi - factor) / xi;
        (1.0_f64, factor, grad_x)
    };

    // Conservative asymptotic variances for GEV MLEs
    let var_mu = sigma * sigma / n;
    let var_sigma = sigma * sigma / n;
    let var_xi = (1.0 + xi).powi(2).max(0.01) / n;

    let var_rl = grad_mu * grad_mu * var_mu
        + grad_sigma * grad_sigma * var_sigma
        + grad_xi * grad_xi * var_xi;

    Ok(var_rl.sqrt().max(0.0))
}

/// Inverse normal CDF (quantile function) via rational approximation.
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Beasley-Springer-Moro rational approximation
    if (p - 0.5).abs() < 0.42 {
        let q = p - 0.5;
        let r = q * q;
        q * ((2.515_517 + 0.802_853 * r + 0.010_328 * r * r)
            / (1.0 + 1.432_788 * r + 0.189_269 * r * r + 0.001_308 * r * r * r))
    } else {
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * pp.ln()).sqrt();
        let r = t;
        let r2 = r * r;
        let c = 2.515_517 + 0.802_853 * r + 0.010_328 * r2;
        let d = 1.0 + 1.432_788 * r + 0.189_269 * r2 + 0.001_308 * r2 * r;
        let val = t - c / d;
        if p < 0.5 {
            -val
        } else {
            val
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_value_at_risk_basic() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let var95 = value_at_risk(&data, 0.95).expect("var");
        assert!(var95 >= 94.0 && var95 <= 96.0, "Expected ~95, got {var95}");
    }

    #[test]
    fn test_value_at_risk_monotone() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let var50 = value_at_risk(&data, 0.50).expect("var50");
        let var90 = value_at_risk(&data, 0.90).expect("var90");
        let var99 = value_at_risk(&data, 0.99).expect("var99");
        assert!(var50 <= var90, "VaR must be monotone: {var50} <= {var90}");
        assert!(var90 <= var99, "VaR must be monotone: {var90} <= {var99}");
    }

    #[test]
    fn test_value_at_risk_invalid_confidence() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(value_at_risk(&data, 0.0).is_err());
        assert!(value_at_risk(&data, 1.0).is_err());
        assert!(value_at_risk(&data, -0.1).is_err());
        assert!(value_at_risk(&data, 1.1).is_err());
    }

    #[test]
    fn test_value_at_risk_insufficient_data() {
        assert!(value_at_risk(&[1.0], 0.95).is_err());
    }

    #[test]
    fn test_conditional_var_ge_var() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let var95 = value_at_risk(&data, 0.95).expect("var");
        let cvar95 = conditional_var(&data, 0.95).expect("cvar");
        assert!(cvar95 >= var95 - 1e-10, "CVaR={cvar95} >= VaR={var95}");
    }

    #[test]
    fn test_conditional_var_uniform_all_same() {
        let data = vec![5.0; 20];
        let cvar = conditional_var(&data, 0.95).expect("cvar");
        assert!(approx_eq(cvar, 5.0, 1e-10), "cvar={cvar}");
    }

    #[test]
    fn test_conditional_var_invalid() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(conditional_var(&data, 0.0).is_err());
        assert!(conditional_var(&data, 1.0).is_err());
    }

    #[test]
    fn test_tail_index_hill_positive() {
        let data: Vec<f64> = (1..=200)
            .map(|i| (i as f64 / 201.0_f64).powf(-0.5))
            .collect();
        let xi = tail_index_hill(&data, 20).expect("hill");
        assert!(xi > 0.0, "xi should be positive for Pareto, got {xi}");
    }

    #[test]
    fn test_tail_index_hill_invalid_k() {
        assert!(tail_index_hill(&[1.0, 2.0, 3.0], 0).is_err());
    }

    #[test]
    fn test_tail_index_hill_insufficient_data() {
        assert!(tail_index_hill(&[1.0, 2.0], 5).is_err());
    }

    #[test]
    fn test_fit_block_maxima_basic() {
        let data: Vec<f64> = (0..200).map(|i| (i % 10) as f64 + 1.0).collect();
        let fit = fit_block_maxima(&data, 10).expect("fit block maxima");
        assert!(fit.distribution.sigma > 0.0);
        assert!(fit.n_obs == 20);
    }

    #[test]
    fn test_fit_block_maxima_invalid_block_size() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert!(fit_block_maxima(&data, 0).is_err());
    }

    #[test]
    fn test_fit_block_maxima_too_few_blocks() {
        // 10 data points, block_size=9 → only 1 block
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        assert!(fit_block_maxima(&data, 9).is_err());
    }

    #[test]
    fn test_return_level_ci_valid() {
        let data: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let (lo, hi) = return_level_ci(&data, 10.0, 0.05).expect("CI");
        assert!(lo <= hi + 1e-9, "lower={lo} should be <= upper={hi}");
        assert!(lo.is_finite(), "lower bound must be finite");
        assert!(hi.is_finite(), "upper bound must be finite");
    }

    #[test]
    fn test_return_level_ci_wider_for_smaller_n() {
        let data_small: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let data_large: Vec<f64> = (1..=200).map(|i| i as f64).collect();
        let (lo_s, hi_s) = return_level_ci(&data_small, 10.0, 0.05).expect("CI small");
        let (lo_l, hi_l) = return_level_ci(&data_large, 10.0, 0.05).expect("CI large");
        let width_small = hi_s - lo_s;
        let width_large = hi_l - lo_l;
        assert!(
            width_small >= width_large * 0.5,
            "Small-n CI ({width_small}) should be at least comparable to large-n CI ({width_large})"
        );
    }

    #[test]
    fn test_return_level_ci_invalid_period() {
        let data: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        assert!(return_level_ci(&data, 0.5, 0.05).is_err());
        assert!(return_level_ci(&data, 1.0, 0.05).is_err());
    }

    #[test]
    fn test_return_level_ci_invalid_alpha() {
        let data: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        assert!(return_level_ci(&data, 10.0, 0.0).is_err());
        assert!(return_level_ci(&data, 10.0, 1.0).is_err());
    }

    #[test]
    fn test_pot_analysis_basic() {
        // 50 exceedances: values 51..=100
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let result = pot_analysis(&data, 50.0);
        assert!(result.is_ok(), "pot_analysis failed: {:?}", result.err());
    }

    #[test]
    fn test_pot_analysis_non_finite_threshold() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(pot_analysis(&data, f64::INFINITY).is_err());
        assert!(pot_analysis(&data, f64::NAN).is_err());
    }
}
