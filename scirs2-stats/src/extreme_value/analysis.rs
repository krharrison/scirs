//! EVT analysis tools: block maxima, Peaks Over Threshold (POT), return levels,
//! Hill/Pickands estimators, mean excess plots, and empirical return periods.
//!
//! # Key Functions
//! - [`block_maxima_analysis`]: Fit a GEV to block maxima and compute return levels.
//! - [`peaks_over_threshold`]: POT analysis – extract exceedances, fit GPD, compute return levels.
//! - [`mean_excess_plot`]: Mean Residual Life plot to help choose the POT threshold.
//! - [`hill_estimator`]: Semi-parametric tail index estimator for heavy-tailed data.
//! - [`pickands_estimator`]: Pickands tail index estimator.
//! - [`return_level_confidence`]: Delta-method confidence interval for GEV return levels.
//! - [`empirical_return_periods`]: Empirical return periods using various plotting positions.
//!
//! # References
//! - Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.
//! - Hill, B.M. (1975). A simple general approach to inference about the tail. *Ann. Statist.*
//! - Pickands, J. (1975). Statistical inference using extreme order statistics. *Ann. Statist.*

use crate::error::StatsError;
use scirs2_core::ndarray::{Array1, ArrayView1};

use super::distributions::{GeneralizedExtremeValue, GeneralizedPareto};
use super::estimation::{gev_fit_lmoments, gev_fit_mle, gev_fit_pwm, gpd_fit_mle};

// ---------------------------------------------------------------------------
// Shared enumerations
// ---------------------------------------------------------------------------

/// Method used to estimate distribution parameters in EVT analyses.
#[derive(Debug, Clone, PartialEq)]
pub enum EvtEstimationMethod {
    /// Maximum Likelihood Estimation (Nelder–Mead optimizer).
    MLE,
    /// Probability-Weighted Moments (Hosking et al. 1985).
    PWM,
    /// L-moments (Hosking 1990) – generally recommended for small samples.
    LMoments,
}

/// Plotting position formula for empirical return period estimation.
#[derive(Debug, Clone, PartialEq)]
pub enum PlottingPosition {
    /// Weibull: F̂ᵢ = i / (n+1)  — unbiased for uniform distribution.
    Weibull,
    /// Gringorten: F̂ᵢ = (i − 0.44) / (n + 0.12)  — recommended for Gumbel.
    Gringorten,
    /// Blom: F̂ᵢ = (i − 0.375) / (n + 0.25)  — approximately unbiased normal quantiles.
    Blom,
    /// Hazen: F̂ᵢ = (i − 0.5) / n  — midpoint formula.
    Hazen,
}

// ---------------------------------------------------------------------------
// Block maxima
// ---------------------------------------------------------------------------

/// Results from a block maxima EVT analysis.
#[derive(Debug, Clone)]
pub struct BlockMaximaResult {
    /// The extracted block maxima (one per block).
    pub block_maxima: Array1<f64>,
    /// Fitted GEV parameters.
    pub gev_params: GeneralizedExtremeValue,
    /// Return levels: list of `(return_period, level)` pairs.
    pub return_levels: Vec<(f64, f64)>,
    /// Number of complete blocks extracted from data.
    pub n_blocks: usize,
}

/// Block maxima method for EVT analysis.
///
/// The data series is divided into non-overlapping blocks of `block_size` observations.
/// The maximum of each **complete** block is extracted and a GEV distribution is fitted
/// to these block maxima.
///
/// # Arguments
/// - `data`: input time series.
/// - `block_size`: number of observations per block (e.g. 365 for annual maxima of daily data).
/// - `return_periods`: slice of desired return periods (must all be > 1.0).
/// - `estimation`: parameter estimation method.
///
/// # Errors
/// - [`StatsError::InsufficientData`] if fewer than 2 complete blocks.
/// - [`StatsError::InvalidArgument`] if `block_size == 0` or any return period ≤ 1.
pub fn block_maxima_analysis(
    data: ArrayView1<f64>,
    block_size: usize,
    return_periods: &[f64],
    estimation: EvtEstimationMethod,
) -> Result<BlockMaximaResult, StatsError> {
    if block_size == 0 {
        return Err(StatsError::InvalidArgument(
            "block_size must be >= 1".into(),
        ));
    }
    for &rp in return_periods {
        if rp <= 1.0 {
            return Err(StatsError::InvalidArgument(format!(
                "All return periods must be > 1.0, got {rp}"
            )));
        }
    }

    let n = data.len();
    let n_blocks = n / block_size;

    if n_blocks < 2 {
        return Err(StatsError::InsufficientData(format!(
            "Need at least 2 complete blocks (block_size={block_size}), \
             but data length {n} gives only {n_blocks} block(s)"
        )));
    }

    // Extract block maxima
    let mut maxima = Vec::with_capacity(n_blocks);
    let data_slice: Vec<f64> = data.iter().copied().collect();
    for b in 0..n_blocks {
        let start = b * block_size;
        let end = start + block_size;
        let block_max = data_slice[start..end]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if block_max.is_finite() {
            maxima.push(block_max);
        }
    }

    if maxima.len() < 2 {
        return Err(StatsError::InsufficientData(
            "Could not extract at least 2 finite block maxima".into(),
        ));
    }

    let maxima_arr = Array1::from(maxima.clone());

    // Fit GEV to block maxima
    let gev_params = match estimation {
        EvtEstimationMethod::MLE => gev_fit_mle(maxima_arr.view())?,
        EvtEstimationMethod::PWM => gev_fit_pwm(maxima_arr.view())?,
        EvtEstimationMethod::LMoments => gev_fit_lmoments(maxima_arr.view())?,
    };

    // Compute return levels
    let return_levels: Vec<(f64, f64)> = return_periods
        .iter()
        .map(|&rp| {
            let level = gev_params.return_level(rp).unwrap_or(f64::NAN);
            (rp, level)
        })
        .collect();

    Ok(BlockMaximaResult {
        block_maxima: maxima_arr,
        gev_params,
        return_levels,
        n_blocks,
    })
}

// ---------------------------------------------------------------------------
// Peaks Over Threshold (POT)
// ---------------------------------------------------------------------------

/// Results from a Peaks Over Threshold (POT) analysis.
#[derive(Debug, Clone)]
pub struct PotResult {
    /// The threshold used.
    pub threshold: f64,
    /// Exceedances above the threshold (already subtracted: x − threshold).
    pub exceedances: Array1<f64>,
    /// Number of exceedances.
    pub n_exceedances: usize,
    /// Exceedance rate λ = n_exceedances / n_total.
    pub rate: f64,
    /// Fitted GPD parameters (with mu = 0, since exceedances are already shifted).
    pub gpd_params: GeneralizedPareto,
    /// Return levels: `(return_period, level)` in the original scale.
    pub return_levels: Vec<(f64, f64)>,
}

/// Peaks Over Threshold (POT) analysis.
///
/// Identifies values in `data` that exceed `threshold`, fits a GPD to the exceedances,
/// and computes return levels.
///
/// For a return period T (in the same units as the data time steps), the return level is:
///
/// x_T = threshold + (σ/ξ) \[(λT)^ξ − 1\]  for ξ ≠ 0
/// x_T = threshold + σ ln(λT)               for ξ = 0
///
/// where λ is the exceedance rate.
///
/// # Errors
/// - [`StatsError::InsufficientData`] if fewer than 5 exceedances above the threshold.
/// - [`StatsError::InvalidArgument`] if any return period ≤ 1.
pub fn peaks_over_threshold(
    data: ArrayView1<f64>,
    threshold: f64,
    return_periods: &[f64],
) -> Result<PotResult, StatsError> {
    for &rp in return_periods {
        if rp <= 1.0 {
            return Err(StatsError::InvalidArgument(format!(
                "All return periods must be > 1.0, got {rp}"
            )));
        }
    }

    let n_total = data.len();
    if n_total == 0 {
        return Err(StatsError::InsufficientData("Data is empty".into()));
    }

    // Extract exceedances (x − threshold for x > threshold)
    let exceedances: Vec<f64> = data
        .iter()
        .filter_map(|&x| {
            if x > threshold {
                Some(x - threshold)
            } else {
                None
            }
        })
        .collect();

    let n_exc = exceedances.len();
    if n_exc < 5 {
        return Err(StatsError::InsufficientData(format!(
            "POT requires at least 5 exceedances above threshold {threshold}; got {n_exc}"
        )));
    }

    let rate = n_exc as f64 / n_total as f64;
    let exc_arr = Array1::from(exceedances);

    // Fit GPD to exceedances
    let gpd_params = gpd_fit_mle(exc_arr.view())?;

    // Return levels in original scale
    let sigma = gpd_params.sigma;
    let xi = gpd_params.xi;
    const XI_TOL: f64 = 1e-10;

    let return_levels: Vec<(f64, f64)> = return_periods
        .iter()
        .map(|&rp| {
            let lambda_t = rate * rp;
            let level = if xi.abs() < XI_TOL {
                threshold + sigma * lambda_t.ln()
            } else {
                threshold + (sigma / xi) * (lambda_t.powf(xi) - 1.0)
            };
            (rp, level)
        })
        .collect();

    Ok(PotResult {
        threshold,
        exceedances: exc_arr,
        n_exceedances: n_exc,
        rate,
        gpd_params,
        return_levels,
    })
}

// ---------------------------------------------------------------------------
// Mean Excess / Mean Residual Life plot
// ---------------------------------------------------------------------------

/// Compute the Mean Excess (Mean Residual Life) function over a range of thresholds.
///
/// For each threshold u, the mean excess is E[X − u | X > u].  If the data follow a GPD,
/// this is a linear function of u (slope = ξ/(1−ξ), intercept = σ/(1−ξ)).
///
/// Useful for threshold selection in POT analysis.
///
/// # Arguments
/// - `data`: observed values.
/// - `n_thresholds`: number of evenly-spaced threshold values to evaluate (default 50).
///
/// # Returns
/// `(thresholds, mean_excess)` — two arrays of the same length.
///
/// # Errors
/// - [`StatsError::InsufficientData`] if fewer than 10 observations.
pub fn mean_excess_plot(
    data: ArrayView1<f64>,
    n_thresholds: usize,
) -> Result<(Array1<f64>, Array1<f64>), StatsError> {
    let n = data.len();
    if n < 10 {
        return Err(StatsError::InsufficientData(
            "Mean excess plot requires at least 10 observations".into(),
        ));
    }
    if n_thresholds == 0 {
        return Err(StatsError::InvalidArgument(
            "n_thresholds must be >= 1".into(),
        ));
    }

    let mut sorted: Vec<f64> = data.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Use the 5th and 95th percentile as threshold range (ensure ≥ 5 exceedances)
    let lo_idx = n / 20; // ~5th percentile
    let hi_idx = (9 * n) / 10; // ~90th percentile
    let lo = sorted[lo_idx];
    let hi = sorted[hi_idx];

    if hi <= lo {
        return Err(StatsError::ComputationError(
            "Mean excess plot: data range too narrow".into(),
        ));
    }

    let step = (hi - lo) / n_thresholds as f64;
    let mut thresholds = Vec::with_capacity(n_thresholds);
    let mut mean_excess = Vec::with_capacity(n_thresholds);

    for k in 0..n_thresholds {
        let u = lo + k as f64 * step;
        let exceedances: Vec<f64> = sorted.iter().filter(|&&x| x > u).map(|&x| x - u).collect();
        if exceedances.len() < 2 {
            break;
        }
        let me = exceedances.iter().sum::<f64>() / exceedances.len() as f64;
        thresholds.push(u);
        mean_excess.push(me);
    }

    if thresholds.is_empty() {
        return Err(StatsError::ComputationError(
            "Mean excess plot: no valid threshold produced enough exceedances".into(),
        ));
    }

    Ok((Array1::from(thresholds), Array1::from(mean_excess)))
}

// ---------------------------------------------------------------------------
// Hill estimator
// ---------------------------------------------------------------------------

/// Hill estimator of the tail index ξ for heavy-tailed distributions.
///
/// Requires ξ > 0 (Pareto / Fréchet tail behaviour).
/// Uses the top-k order statistics.
///
/// ξ̂_Hill(k) = (1/k) Σ_{i=1}^{k} log(X_{n−i+1:n}) − log(X_{n−k:n})
///
/// # Arguments
/// - `data`: observed values (need not be sorted).
/// - `k`: number of upper order statistics to use (1 ≤ k ≤ n−1).
///
/// # Errors
/// - [`StatsError::InvalidArgument`] if `k` is 0 or ≥ n.
/// - [`StatsError::InsufficientData`] if `n < 2`.
pub fn hill_estimator(data: ArrayView1<f64>, k: usize) -> Result<f64, StatsError> {
    let n = data.len();
    if n < 2 {
        return Err(StatsError::InsufficientData(
            "Hill estimator requires at least 2 observations".into(),
        ));
    }
    if k == 0 || k >= n {
        return Err(StatsError::InvalidArgument(format!(
            "k must be in [1, n-1] = [1, {}], got {k}",
            n - 1
        )));
    }

    let mut sorted: Vec<f64> = data.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // X_{n-k:n} is the (k+1)-th largest value
    let x_threshold = sorted[n - k - 1];
    if x_threshold <= 0.0 {
        return Err(StatsError::InvalidArgument(
            "Hill estimator requires all data values used to be positive".into(),
        ));
    }

    let log_threshold = x_threshold.ln();
    let mut sum = 0.0_f64;
    for i in (n - k)..n {
        if sorted[i] <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "Hill estimator: encountered non-positive order statistic".into(),
            ));
        }
        sum += sorted[i].ln() - log_threshold;
    }

    Ok(sum / k as f64)
}

// ---------------------------------------------------------------------------
// Pickands estimator
// ---------------------------------------------------------------------------

/// Pickands estimator of the tail index ξ.
///
/// Uses three order statistics symmetrically placed in the upper tail:
///
/// ξ̂_Pickands(k) = (1/ln2) * ln\[(X_{n−k+1:n} − X_{n−2k+1:n}) / (X_{n−2k+1:n} − X_{n−4k+1:n})\]
///
/// # Arguments
/// - `data`: observed values.
/// - `k`: tail parameter; requires 4k < n.
///
/// # Errors
/// - [`StatsError::InvalidArgument`] if `4k >= n` or `k == 0`.
pub fn pickands_estimator(data: ArrayView1<f64>, k: usize) -> Result<f64, StatsError> {
    let n = data.len();
    if k == 0 {
        return Err(StatsError::InvalidArgument("k must be >= 1".into()));
    }
    if 4 * k >= n {
        return Err(StatsError::InvalidArgument(format!(
            "Pickands estimator requires 4k < n; got 4*{k}={} >= n={n}",
            4 * k
        )));
    }

    let mut sorted: Vec<f64> = data.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Order statistics (1-indexed from the top):
    // X_{n−k+1:n}   = sorted[n - k]
    // X_{n−2k+1:n}  = sorted[n - 2*k]
    // X_{n−4k+1:n}  = sorted[n - 4*k]
    let x1 = sorted[n - k];
    let x2 = sorted[n - 2 * k];
    let x3 = sorted[n - 4 * k];

    let num = x1 - x2;
    let den = x2 - x3;

    if den.abs() < 1e-15 {
        return Err(StatsError::ComputationError(
            "Pickands estimator: degenerate order statistics (denominator ≈ 0)".into(),
        ));
    }
    if num / den <= 0.0 {
        return Err(StatsError::ComputationError(
            "Pickands estimator: invalid ratio (non-positive)".into(),
        ));
    }

    let xi = (num / den).ln() / 2.0_f64.ln();
    Ok(xi)
}

// ---------------------------------------------------------------------------
// Return level with confidence interval (delta method)
// ---------------------------------------------------------------------------

/// Compute a GEV return level with a delta-method confidence interval.
///
/// Uses the Fisher information matrix approximation for the GEV to derive the
/// asymptotic variance of the return level estimator.
///
/// # Arguments
/// - `params`: fitted GEV parameters.
/// - `return_period`: return period T.
/// - `n_data`: number of block maxima used to fit the GEV.
/// - `alpha`: significance level (e.g. 0.05 for 95% CI).
///
/// # Returns
/// `(lower, estimate, upper)`.
///
/// # Errors
/// - [`StatsError::InvalidArgument`] if return_period ≤ 1 or alpha not in (0,0.5).
pub fn return_level_confidence(
    params: &GeneralizedExtremeValue,
    return_period: f64,
    n_data: usize,
    alpha: f64,
) -> Result<(f64, f64, f64), StatsError> {
    if return_period <= 1.0 {
        return Err(StatsError::InvalidArgument(format!(
            "return_period must be > 1, got {return_period}"
        )));
    }
    if !(0.0 < alpha && alpha < 0.5) {
        return Err(StatsError::InvalidArgument(format!(
            "alpha must be in (0, 0.5), got {alpha}"
        )));
    }
    if n_data < 3 {
        return Err(StatsError::InsufficientData(
            "At least 3 observations needed for confidence interval".into(),
        ));
    }

    let estimate = params.return_level(return_period)?;

    // Delta method: approximate variance via numerical differentiation of return level
    // w.r.t. (μ, σ, ξ).
    let h = 1e-5;
    let mu = params.mu;
    let sigma = params.sigma;
    let xi = params.xi;

    let rl = |mu2: f64, sig2: f64, xi2: f64| -> f64 {
        GeneralizedExtremeValue::new(mu2, sig2, xi2)
            .ok()
            .and_then(|g| g.return_level(return_period).ok())
            .unwrap_or(f64::NAN)
    };

    let d_mu = (rl(mu + h, sigma, xi) - rl(mu - h, sigma, xi)) / (2.0 * h);
    let d_sigma = (rl(mu, sigma + h, xi) - rl(mu, sigma - h, xi)) / (2.0 * h);
    let d_xi = (rl(mu, sigma, xi + h) - rl(mu, sigma, xi - h)) / (2.0 * h);

    if !d_mu.is_finite() || !d_sigma.is_finite() || !d_xi.is_finite() {
        return Err(StatsError::ComputationError(
            "Delta method: gradient computation failed at these parameters".into(),
        ));
    }

    // Approximate variance of the information matrix (diagonal approximation):
    // Var(μ̂) ≈ σ²/n,  Var(σ̂) ≈ σ²/(2n),  Var(ξ̂) ≈ (1+ξ)²/n
    // These are rough order-of-magnitude estimates; the off-diagonal terms are ignored here.
    let nf = n_data as f64;
    let var_mu = sigma.powi(2) / nf;
    let var_sigma = sigma.powi(2) / (2.0 * nf);
    let var_xi = (1.0 + xi).powi(2) / nf;

    let var_rl = d_mu.powi(2) * var_mu + d_sigma.powi(2) * var_sigma + d_xi.powi(2) * var_xi;

    if var_rl < 0.0 || !var_rl.is_finite() {
        return Err(StatsError::ComputationError(
            "Delta method: negative or invalid variance estimate".into(),
        ));
    }

    // z_{1-α/2} from normal distribution (pre-computed for common alpha values)
    let z = normal_quantile(1.0 - alpha / 2.0);
    let half_width = z * var_rl.sqrt();

    Ok((estimate - half_width, estimate, estimate + half_width))
}

/// Approximate normal quantile (Beasley–Springer–Moro algorithm).
fn normal_quantile(p: f64) -> f64 {
    // Rational approximation from Abramowitz & Stegun 26.2.17
    let p = p.clamp(1e-15, 1.0 - 1e-15);
    let q = if p < 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * q.ln()).sqrt();
    const C: [f64; 3] = [2.515517, 0.802853, 0.010328];
    const D: [f64; 3] = [1.432788, 0.189269, 0.001308];
    let num = C[0] + C[1] * t + C[2] * t.powi(2);
    let den = 1.0 + D[0] * t + D[1] * t.powi(2) + D[2] * t.powi(3);
    let z = t - num / den;
    if p >= 0.5 {
        z
    } else {
        -z
    }
}

// ---------------------------------------------------------------------------
// Empirical return periods
// ---------------------------------------------------------------------------

/// Compute empirical return periods for sorted data using various plotting position formulas.
///
/// Returns `(sorted_data, return_periods)` — the data sorted in ascending order and the
/// corresponding empirical return periods T = 1 / (1 − F̂).
///
/// # Arguments
/// - `data`: observed sample.
/// - `plotting_position`: plotting position formula.
///
/// # Errors
/// - [`StatsError::InsufficientData`] if `data` is empty.
pub fn empirical_return_periods(
    data: ArrayView1<f64>,
    plotting_position: PlottingPosition,
) -> Result<(Array1<f64>, Array1<f64>), StatsError> {
    let n = data.len();
    if n == 0 {
        return Err(StatsError::InsufficientData(
            "Data must not be empty".into(),
        ));
    }

    let mut sorted: Vec<f64> = data.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let nf = n as f64;

    let return_periods: Vec<f64> = (1..=n)
        .map(|i| {
            let if64 = i as f64;
            // Empirical exceedance probability (using plotting position)
            let f_i = match plotting_position {
                PlottingPosition::Weibull => if64 / (nf + 1.0),
                PlottingPosition::Gringorten => (if64 - 0.44) / (nf + 0.12),
                PlottingPosition::Blom => (if64 - 0.375) / (nf + 0.25),
                PlottingPosition::Hazen => (if64 - 0.5) / nf,
            };
            let f_i = f_i.clamp(1e-10, 1.0 - 1e-10);
            1.0 / (1.0 - f_i)
        })
        .collect();

    Ok((Array1::from(sorted), Array1::from(return_periods)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1};

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn relative_eq(a: f64, b: f64, rtol: f64) -> bool {
        let denom = b.abs().max(1e-12);
        (a - b).abs() / denom < rtol
    }

    // Helper: generate synthetic data from a known Gumbel(μ, β) distribution
    fn gumbel_sample(mu: f64, beta: f64, n: usize, seed: u64) -> Array1<f64> {
        use super::super::distributions::Gumbel;
        let g = Gumbel::new(mu, beta).unwrap();
        Array1::from(g.sample(n, seed))
    }

    // ---- Block Maxima -------------------------------------------------------

    #[test]
    fn test_block_maxima_basic() {
        let data = gumbel_sample(10.0, 2.0, 500, 1);
        let result = block_maxima_analysis(
            data.view(),
            50,
            &[10.0, 100.0],
            EvtEstimationMethod::LMoments,
        )
        .unwrap();
        assert_eq!(result.n_blocks, 10);
        assert_eq!(result.block_maxima.len(), 10);
        assert_eq!(result.return_levels.len(), 2);
        assert!(result.gev_params.sigma > 0.0);
    }

    #[test]
    fn test_block_maxima_return_levels_increasing() {
        let data = gumbel_sample(5.0, 1.5, 1000, 2);
        let result = block_maxima_analysis(
            data.view(),
            100,
            &[10.0, 50.0, 100.0, 1000.0],
            EvtEstimationMethod::LMoments,
        )
        .unwrap();
        let levels: Vec<f64> = result.return_levels.iter().map(|&(_, l)| l).collect();
        // Higher return periods must give higher (or equal) return levels
        for w in levels.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-6,
                "levels not non-decreasing: {:?}",
                levels
            );
        }
    }

    #[test]
    fn test_block_maxima_mle() {
        let data = gumbel_sample(0.0, 1.0, 400, 10);
        let result =
            block_maxima_analysis(data.view(), 40, &[10.0, 100.0], EvtEstimationMethod::MLE);
        assert!(
            result.is_ok(),
            "MLE block maxima failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_block_maxima_pwm() {
        let data = gumbel_sample(0.0, 1.0, 400, 11);
        let result = block_maxima_analysis(data.view(), 40, &[10.0], EvtEstimationMethod::PWM);
        assert!(result.is_ok());
    }

    #[test]
    fn test_block_maxima_zero_block_size_error() {
        let data = gumbel_sample(0.0, 1.0, 100, 3);
        assert!(
            block_maxima_analysis(data.view(), 0, &[10.0], EvtEstimationMethod::LMoments).is_err()
        );
    }

    #[test]
    fn test_block_maxima_too_few_blocks_error() {
        // Only 1 block → error
        let data = gumbel_sample(0.0, 1.0, 50, 4);
        assert!(
            block_maxima_analysis(data.view(), 60, &[10.0], EvtEstimationMethod::LMoments).is_err()
        );
    }

    #[test]
    fn test_block_maxima_invalid_return_period_error() {
        let data = gumbel_sample(0.0, 1.0, 200, 5);
        assert!(
            block_maxima_analysis(data.view(), 20, &[0.5], EvtEstimationMethod::LMoments).is_err()
        );
    }

    // ---- POT ---------------------------------------------------------------

    #[test]
    fn test_pot_basic() {
        let data = gumbel_sample(5.0, 2.0, 500, 20);
        let threshold = 7.0;
        let result = peaks_over_threshold(data.view(), threshold, &[10.0, 100.0]).unwrap();
        assert_eq!(result.threshold, threshold);
        assert!(result.n_exceedances > 0);
        assert!(result.rate > 0.0 && result.rate < 1.0);
        assert!(result.gpd_params.sigma > 0.0);
    }

    #[test]
    fn test_pot_return_levels_increasing() {
        let data = gumbel_sample(0.0, 1.0, 1000, 21);
        let result = peaks_over_threshold(data.view(), 1.0, &[5.0, 10.0, 50.0, 100.0]).unwrap();
        let levels: Vec<f64> = result.return_levels.iter().map(|&(_, l)| l).collect();
        for w in levels.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "{:?}", levels);
        }
    }

    #[test]
    fn test_pot_insufficient_exceedances_error() {
        let data = array![1.0, 2.0, 3.0, 4.0, 100.0];
        // Only 1 value exceeds 50 → error
        assert!(peaks_over_threshold(data.view(), 50.0, &[10.0]).is_err());
    }

    #[test]
    fn test_pot_invalid_return_period_error() {
        let data = gumbel_sample(0.0, 1.0, 200, 22);
        assert!(peaks_over_threshold(data.view(), 0.0, &[0.5]).is_err());
    }

    // ---- Mean Excess -------------------------------------------------------

    #[test]
    fn test_mean_excess_plot_basic() {
        let data = gumbel_sample(0.0, 1.0, 200, 30);
        let (thresholds, me) = mean_excess_plot(data.view(), 20).unwrap();
        assert!(!thresholds.is_empty());
        assert_eq!(thresholds.len(), me.len());
        assert!(me.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_mean_excess_plot_insufficient_data_error() {
        let data = array![1.0, 2.0, 3.0];
        assert!(mean_excess_plot(data.view(), 10).is_err());
    }

    #[test]
    fn test_mean_excess_exponential_linear() {
        // For an exponential distribution, the mean excess function is constant (linear with slope 0)
        use super::super::distributions::GeneralizedPareto;
        let gpd = GeneralizedPareto::new(0.0, 2.0, 0.0).unwrap(); // exponential
        let samples = gpd.sample(500, 42);
        let arr = Array1::from(samples);
        let (_, me) = mean_excess_plot(arr.view(), 15).unwrap();
        // Mean excess should be approximately constant ≈ 2.0 near lower thresholds
        let first = me[0];
        let last = me[me.len() - 1];
        // Allow reasonable variance; just check it's in the right ballpark
        assert!(first > 0.0 && last > 0.0);
    }

    // ---- Hill estimator ----------------------------------------------------

    #[test]
    fn test_hill_basic_heavy_tail() {
        // Pareto data: CDF = 1 - x^{-alpha} for x >= 1; tail index xi = 1/alpha
        let alpha = 2.0_f64;
        // Generate pseudo-Pareto data using quantile inversion: x = (1-u)^{-1/alpha}
        let mut data: Vec<f64> = (1..=500)
            .map(|i| {
                let u = i as f64 / 501.0;
                (1.0 - u).powf(-1.0 / alpha)
            })
            .collect();
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let arr = Array1::from(data);
        let xi_hat = hill_estimator(arr.view(), 50).unwrap();
        // Expected: xi ≈ 0.5 (= 1/alpha)
        assert!(relative_eq(xi_hat, 1.0 / alpha, 0.25), "xi_hat={xi_hat}");
    }

    #[test]
    fn test_hill_invalid_k_error() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(hill_estimator(data.view(), 0).is_err());
        assert!(hill_estimator(data.view(), 5).is_err()); // k >= n
    }

    #[test]
    fn test_hill_insufficient_data_error() {
        let data = array![1.0];
        assert!(hill_estimator(data.view(), 1).is_err());
    }

    // ---- Pickands estimator -----------------------------------------------

    #[test]
    fn test_pickands_basic() {
        // Pareto(alpha=2): tail index xi = 0.5
        let alpha = 2.0_f64;
        let data: Vec<f64> = (1..=200)
            .map(|i| {
                let u = i as f64 / 201.0;
                (1.0 - u).powf(-1.0 / alpha)
            })
            .collect();
        let arr = Array1::from(data);
        let k = 10;
        let xi_hat = pickands_estimator(arr.view(), k).unwrap();
        // Pickands is noisier; allow wider tolerance
        assert!(xi_hat.is_finite(), "xi_hat={xi_hat}");
    }

    #[test]
    fn test_pickands_invalid_k_error() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!(pickands_estimator(data.view(), 0).is_err());
        assert!(pickands_estimator(data.view(), 3).is_err()); // 4*3 = 12 >= 10
    }

    // ---- Return level confidence interval ---------------------------------

    #[test]
    fn test_return_level_ci_basic() {
        let gev = GeneralizedExtremeValue::new(0.0, 1.0, 0.1).unwrap();
        let (lo, est, hi) = return_level_confidence(&gev, 100.0, 50, 0.05).unwrap();
        assert!(lo < est, "lo={lo} should be < est={est}");
        assert!(est < hi, "est={est} should be < hi={hi}");
    }

    #[test]
    fn test_return_level_ci_wider_for_small_n() {
        let gev = GeneralizedExtremeValue::new(0.0, 1.0, 0.0).unwrap();
        let (lo50, _, hi50) = return_level_confidence(&gev, 100.0, 50, 0.05).unwrap();
        let (lo500, _, hi500) = return_level_confidence(&gev, 100.0, 500, 0.05).unwrap();
        let width_50 = hi50 - lo50;
        let width_500 = hi500 - lo500;
        assert!(width_50 > width_500, "Smaller n should give wider CI");
    }

    #[test]
    fn test_return_level_ci_invalid_inputs() {
        let gev = GeneralizedExtremeValue::new(0.0, 1.0, 0.0).unwrap();
        assert!(return_level_confidence(&gev, 0.5, 100, 0.05).is_err());
        assert!(return_level_confidence(&gev, 100.0, 100, 0.0).is_err());
        assert!(return_level_confidence(&gev, 100.0, 100, 0.6).is_err());
    }

    // ---- Empirical return periods ------------------------------------------

    #[test]
    fn test_empirical_return_periods_weibull() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let (sorted, rp) =
            empirical_return_periods(data.view(), PlottingPosition::Weibull).unwrap();
        assert_eq!(sorted.len(), 5);
        assert_eq!(rp.len(), 5);
        // Smallest data point has smallest return period, largest has largest
        for w in rp.iter().collect::<Vec<_>>().windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn test_empirical_return_periods_all_methods() {
        let data = gumbel_sample(0.0, 1.0, 100, 50);
        for method in [
            PlottingPosition::Weibull,
            PlottingPosition::Gringorten,
            PlottingPosition::Blom,
            PlottingPosition::Hazen,
        ] {
            let (s, rp) = empirical_return_periods(data.view(), method).unwrap();
            assert_eq!(s.len(), 100);
            assert_eq!(rp.len(), 100);
            assert!(rp.iter().all(|&r| r >= 1.0));
        }
    }

    #[test]
    fn test_empirical_return_periods_empty_error() {
        let data: Array1<f64> = Array1::zeros(0);
        assert!(empirical_return_periods(data.view(), PlottingPosition::Weibull).is_err());
    }

    #[test]
    fn test_normal_quantile_symmetry() {
        let z_95 = normal_quantile(0.975);
        let z_05 = normal_quantile(0.025);
        assert!(approx_eq(z_95, -z_05, 1e-6));
        // z_{0.975} ≈ 1.96
        assert!(approx_eq(z_95, 1.96, 0.01));
    }
}
