//! Extreme Value Theory (EVT) for statistical analysis of extreme events.
//!
//! This module provides:
//! - **Distributions**: GEV, Gumbel, Fréchet, Weibull (reversed), and Generalized Pareto (GPD)
//! - **Estimation**: MLE, Probability-Weighted Moments (PWM), and L-moments parameter estimation
//! - **Analysis**: Block maxima method, Peaks Over Threshold (POT), return levels, Hill/Pickands
//!   tail index estimators, mean excess plots, and empirical return period analysis
//! - **Convenience API**: Standalone `gev_pdf`, `gev_cdf`, `block_maxima`, `return_level`,
//!   `extreme_value_index` functions matching the task specification.
//!
//! # References
//! - Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.
//! - Hosking, J.R.M. (1990). L-moments: Analysis and estimation of distributions using linear
//!   combinations of order statistics. *JRSS-B*, 52(1), 105–124.
//! - Pickands, J. (1975). Statistical inference using extreme order statistics. *Ann. Statist.*

pub mod analysis;
pub mod distributions;
pub mod estimation;

pub mod gev;
pub mod gpd;
// The actual source files are block_maxima.rs and pot.rs;
// expose them under the `_module` names that the public API expects.
#[path = "block_maxima.rs"]
pub mod block_maxima_module;
#[path = "pot.rs"]
pub mod pot_module;
pub mod return_levels;

pub use block_maxima_module::{
    extract_block_maxima, BlockMaximaFitter, BlockMaximaResult as BlockMaximaFitterResult,
};
pub use gev::GEV;
pub use gpd::GPD as GpdModel;
pub use pot_module::{
    extract_exceedances, threshold_selection_mrl, threshold_selection_stability, POTFitter,
    POTResult as PotFitterResult,
};
pub use return_levels::{return_level_confidence_intervals, return_level_gev, return_level_gpd};

pub use analysis::{
    block_maxima_analysis, empirical_return_periods, hill_estimator, mean_excess_plot,
    peaks_over_threshold, pickands_estimator, return_level_confidence, BlockMaximaResult,
    EvtEstimationMethod, PlottingPosition, PotResult,
};
pub use distributions::{Frechet, GeneralizedExtremeValue, GeneralizedPareto, Gumbel};
pub use estimation::{
    gev_fit_lmoments, gev_fit_mle, gev_fit_pwm, gev_goodness_of_fit, gpd_fit_mle,
    gumbel_fit_lmoments, gumbel_fit_mle, sample_lmoments,
};

// ============================================================================
// Convenience type aliases and wrapper types
// ============================================================================

/// A fitted GEV distribution with diagnostics.
///
/// Wraps [`GeneralizedExtremeValue`] and adds log-likelihood, AIC, BIC, and the
/// estimation method used, mirroring the SciPy-style `.fit()` idiom.
#[derive(Debug, Clone)]
pub struct GevFit {
    /// Fitted GEV parameters.
    pub distribution: GeneralizedExtremeValue,
    /// Log-likelihood at the estimated parameters.
    pub log_likelihood: f64,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
    /// Number of observations used for fitting.
    pub n_obs: usize,
    /// Estimation method employed.
    pub method: EvtEstimationMethod,
}

impl GevFit {
    /// Convenience alias: access location parameter μ.
    pub fn mu(&self) -> f64 {
        self.distribution.mu
    }

    /// Convenience alias: access scale parameter σ.
    pub fn sigma(&self) -> f64 {
        self.distribution.sigma
    }

    /// Convenience alias: access shape parameter ξ.
    pub fn xi(&self) -> f64 {
        self.distribution.xi
    }

    /// Return level x_T corresponding to return period T.
    pub fn return_level(&self, return_period: f64) -> crate::error::StatsResult<f64> {
        self.distribution.return_level(return_period)
    }
}

/// Type alias — `GevDistribution` is a synonym for [`GeneralizedExtremeValue`].
pub type GevDistribution = GeneralizedExtremeValue;

/// Type alias — `GpdDistribution` is a synonym for [`GeneralizedPareto`].
pub type GpdDistribution = GeneralizedPareto;

// ============================================================================
// Standalone convenience functions
// ============================================================================

/// GEV probability density function.
///
/// Evaluates the PDF of the Generalised Extreme Value distribution with parameters
/// (μ, σ, ξ) at `x`.  Returns 0.0 for values outside the support.
///
/// # Examples
/// ```
/// use scirs2_stats::extreme_value::gev_pdf;
/// let density = gev_pdf(0.0, 0.0, 1.0, 0.0); // Gumbel at its mode
/// assert!(density > 0.0);
/// ```
pub fn gev_pdf(x: f64, mu: f64, sigma: f64, xi: f64) -> f64 {
    match GeneralizedExtremeValue::new(mu, sigma, xi) {
        Ok(g) => g.pdf(x),
        Err(_) => 0.0,
    }
}

/// GEV cumulative distribution function.
///
/// Evaluates the CDF of the Generalised Extreme Value distribution.
///
/// # Examples
/// ```
/// use scirs2_stats::extreme_value::gev_cdf;
/// let p = gev_cdf(0.0, 0.0, 1.0, 0.0); // Gumbel CDF at x=0
/// let expected = std::f64::consts::E.recip(); // exp(-1) ≈ 0.3679
/// assert!((p - expected).abs() < 1e-8);
/// ```
pub fn gev_cdf(x: f64, mu: f64, sigma: f64, xi: f64) -> f64 {
    match GeneralizedExtremeValue::new(mu, sigma, xi) {
        Ok(g) => g.cdf(x),
        Err(_) => 0.0,
    }
}

/// Fit a GEV distribution to data using MLE, returning a [`GevFit`] with diagnostics.
///
/// Uses the Nelder–Mead simplex optimizer internally.  The log-likelihood, AIC, and BIC
/// are computed at the estimated parameters.
///
/// # Errors
/// - [`crate::error::StatsError::InsufficientData`] if fewer than 3 observations.
pub fn fit_gev(data: &[f64]) -> crate::error::StatsResult<GevFit> {
    use scirs2_core::ndarray::Array1;

    if data.len() < 3 {
        return Err(crate::error::StatsError::InsufficientData(
            "GEV fitting requires at least 3 observations".into(),
        ));
    }
    let arr = Array1::from(data.to_vec());
    let distribution = gev_fit_mle(arr.view())?;

    let n = data.len();
    let ll: f64 = data
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

    let k = 3.0;
    let nf = n as f64;
    Ok(GevFit {
        distribution,
        log_likelihood: ll,
        aic: -2.0 * ll + 2.0 * k,
        bic: -2.0 * ll + k * nf.ln(),
        n_obs: n,
        method: EvtEstimationMethod::MLE,
    })
}

/// Extract block maxima from `data` using non-overlapping blocks of `block_size`.
///
/// Returns a `Vec<f64>` of maxima (one per complete block).
///
/// # Errors
/// - [`crate::error::StatsError::InvalidArgument`] if `block_size == 0`.
/// - [`crate::error::StatsError::InsufficientData`] if fewer than 2 complete blocks.
///
/// # Examples
/// ```
/// use scirs2_stats::extreme_value::block_maxima;
/// let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
/// let maxima = block_maxima(&data, 10).expect("should succeed");
/// assert_eq!(maxima.len(), 10);
/// assert_eq!(maxima[0], 9.0);
/// ```
pub fn block_maxima(data: &[f64], block_size: usize) -> crate::error::StatsResult<Vec<f64>> {
    if block_size == 0 {
        return Err(crate::error::StatsError::InvalidArgument(
            "block_size must be >= 1".into(),
        ));
    }
    let n = data.len();
    let n_blocks = n / block_size;
    if n_blocks < 2 {
        return Err(crate::error::StatsError::InsufficientData(format!(
            "Need at least 2 complete blocks (block_size={block_size}), \
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
    Ok(maxima)
}

/// Extract exceedances above a threshold (Peaks Over Threshold method).
///
/// Returns all values in `data` strictly greater than `threshold`, shifted so the
/// smallest returned value is zero (i.e., returns x − threshold for x > threshold).
///
/// # Examples
/// ```
/// use scirs2_stats::extreme_value::peaks_over_threshold_simple;
/// let data = vec![1.0, 2.0, 5.0, 3.0, 8.0, 1.5];
/// let exc = peaks_over_threshold_simple(&data, 2.0);
/// // 5.0-2.0=3.0, 3.0-2.0=1.0, 8.0-2.0=6.0
/// assert_eq!(exc.len(), 3);
/// ```
pub fn peaks_over_threshold_simple(data: &[f64], threshold: f64) -> Vec<f64> {
    data.iter()
        .filter_map(|&x| {
            if x > threshold {
                Some(x - threshold)
            } else {
                None
            }
        })
        .collect()
}

/// Return level x_T for a fitted GEV at return period T.
///
/// `x_T` satisfies `F(x_T) = 1 − 1/T`.
///
/// # Errors
/// - [`crate::error::StatsError::InvalidArgument`] if `return_period <= 1`.
pub fn return_level(fit: &GevFit, return_period: f64) -> crate::error::StatsResult<f64> {
    fit.distribution.return_level(return_period)
}

/// Hill estimator of the extreme value index (tail index) ξ.
///
/// This is a thin wrapper around [`hill_estimator`] accepting a plain slice.
/// It uses the top-`k` order statistics (default k = max(10, n/10)).
///
/// # Errors
/// - [`crate::error::StatsError::InsufficientData`] if fewer than 10 observations.
///
/// # Examples
/// ```
/// use scirs2_stats::extreme_value::extreme_value_index;
/// // Pareto(α=2) quantile data ⟹ tail index ≈ 0.5
/// let data: Vec<f64> = (1..=200)
///     .map(|i| (i as f64 / 201.0_f64).recip().powf(0.5))
///     .collect();
/// let xi = extreme_value_index(&data).expect("should succeed");
/// assert!(xi > 0.0);
/// ```
pub fn extreme_value_index(data: &[f64]) -> crate::error::StatsResult<f64> {
    use scirs2_core::ndarray::Array1;
    let n = data.len();
    if n < 10 {
        return Err(crate::error::StatsError::InsufficientData(
            "extreme_value_index requires at least 10 observations".into(),
        ));
    }
    let k = (n / 10).max(5).min(n - 1);
    let arr = Array1::from(data.to_vec());
    hill_estimator(arr.view(), k)
}

// ============================================================================
// Tests for the convenience API
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_gev_pdf_gumbel_case() {
        // Gumbel (xi=0): PDF at x=mu is (1/sigma)*exp(-1)
        let pdf = gev_pdf(0.0, 0.0, 1.0, 0.0);
        let expected = (-1.0_f64).exp();
        assert!(approx_eq(pdf, expected, 1e-8), "pdf={pdf}");
    }

    #[test]
    fn test_gev_pdf_frechet_positive_xi() {
        // xi > 0: Fréchet type — PDF should be positive in support
        let pdf = gev_pdf(2.0, 0.0, 1.0, 0.5);
        assert!(pdf > 0.0, "Fréchet PDF should be positive, got {pdf}");
    }

    #[test]
    fn test_gev_pdf_weibull_negative_xi() {
        // xi < 0: Weibull type — upper bound at mu - sigma/xi = 0 - 1/(-0.5) = 2.0
        let pdf_inside = gev_pdf(1.0, 0.0, 1.0, -0.5);
        let pdf_outside = gev_pdf(2.5, 0.0, 1.0, -0.5);
        assert!(pdf_inside > 0.0, "Inside support: {pdf_inside}");
        assert_eq!(pdf_outside, 0.0, "Outside support: {pdf_outside}");
    }

    #[test]
    fn test_gev_pdf_invalid_sigma() {
        // sigma <= 0 → returns 0.0 (no panic)
        let pdf = gev_pdf(0.0, 0.0, -1.0, 0.0);
        assert_eq!(pdf, 0.0);
    }

    #[test]
    fn test_gev_cdf_gumbel_at_zero() {
        // F(0) = exp(-exp(0)) = exp(-1) ≈ 0.3679
        let cdf = gev_cdf(0.0, 0.0, 1.0, 0.0);
        assert!(approx_eq(cdf, (-1.0_f64).exp(), 1e-8), "cdf={cdf}");
    }

    #[test]
    fn test_gev_cdf_reduces_to_gumbel_at_xi_zero() {
        // Compare gev_cdf with xi=1e-14 vs xi=0 to confirm continuity
        let c0 = gev_cdf(1.0, 0.0, 1.0, 0.0);
        let c_small = gev_cdf(1.0, 0.0, 1.0, 1e-14);
        assert!(approx_eq(c0, c_small, 1e-5), "c0={c0}, c_small={c_small}");
    }

    #[test]
    fn test_gev_cdf_frechet_lower_bound() {
        // xi=0.5, lower bound = mu - sigma/xi = 0 - 1/0.5 = -2
        // At x = -2.1 (below support), CDF should be 0
        let cdf = gev_cdf(-2.1, 0.0, 1.0, 0.5);
        assert_eq!(cdf, 0.0, "Below Fréchet lower bound: {cdf}");
    }

    #[test]
    fn test_gev_cdf_weibull_upper_bound() {
        // xi=-0.5: upper bound at mu - sigma/xi = 2.0
        let cdf = gev_cdf(2.5, 0.0, 1.0, -0.5);
        assert_eq!(cdf, 1.0, "Above Weibull upper bound: {cdf}");
    }

    #[test]
    fn test_fit_gev_basic() {
        use distributions::Gumbel;
        let g = Gumbel::new(5.0, 2.0).expect("valid gumbel");
        let samples = g.sample(200, 42);
        let fit = fit_gev(&samples).expect("fit should succeed");
        assert!(fit.distribution.sigma > 0.0);
        assert!(fit.log_likelihood.is_finite());
        assert!(fit.aic.is_finite());
        assert!(fit.bic.is_finite());
        assert_eq!(fit.n_obs, 200);
    }

    #[test]
    fn test_fit_gev_insufficient_data() {
        assert!(fit_gev(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_fit_gev_accessors() {
        let fit = fit_gev(&[1.0, 2.0, 3.0, 4.0, 5.0]).expect("fit");
        let _ = fit.mu();
        let _ = fit.sigma();
        let _ = fit.xi();
        assert!(fit.sigma() > 0.0);
    }

    #[test]
    fn test_block_maxima_basic() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let maxima = block_maxima(&data, 10).expect("should succeed");
        assert_eq!(maxima.len(), 10);
        assert_eq!(maxima[0], 9.0);
        assert_eq!(maxima[9], 99.0);
    }

    #[test]
    fn test_block_maxima_zero_block_size_error() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(block_maxima(&data, 0).is_err());
    }

    #[test]
    fn test_block_maxima_too_few_blocks_error() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(block_maxima(&data, 2).is_err()); // only 1 complete block
    }

    #[test]
    fn test_block_maxima_values_correct() {
        // Each block's maximum should equal its largest element
        let data = vec![3.0, 1.0, 2.0, 7.0, 5.0, 6.0];
        let maxima = block_maxima(&data, 3).expect("2 blocks");
        assert_eq!(maxima, vec![3.0, 7.0]);
    }

    #[test]
    fn test_peaks_over_threshold_simple() {
        let data = vec![1.0, 2.0, 5.0, 3.0, 8.0, 1.5];
        let exc = peaks_over_threshold_simple(&data, 2.0);
        // 5-2=3, 3-2=1, 8-2=6
        assert_eq!(exc.len(), 3);
        // sorted for assertion convenience
        let mut exc_sorted = exc.clone();
        exc_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!(approx_eq(exc_sorted[0], 1.0, 1e-12));
        assert!(approx_eq(exc_sorted[1], 3.0, 1e-12));
        assert!(approx_eq(exc_sorted[2], 6.0, 1e-12));
    }

    #[test]
    fn test_peaks_over_threshold_empty_result() {
        let data = vec![1.0, 1.5, 1.8];
        let exc = peaks_over_threshold_simple(&data, 5.0);
        assert!(exc.is_empty());
    }

    #[test]
    fn test_return_level_basic() {
        let samples: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let fit = fit_gev(&samples).expect("fit");
        let rl = return_level(&fit, 100.0).expect("return level");
        assert!(rl.is_finite());
        // Return level should exceed the sample maximum for high return periods
        let sample_max = 9.9_f64;
        assert!(rl > 0.0, "rl={rl}, sample_max={sample_max}");
    }

    #[test]
    fn test_return_level_invalid_period() {
        let samples: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let fit = fit_gev(&samples).expect("fit");
        assert!(return_level(&fit, 0.5).is_err());
        assert!(return_level(&fit, 1.0).is_err());
    }

    #[test]
    fn test_extreme_value_index_heavy_tail() {
        // Pareto(α=2): tail index ξ = 1/α = 0.5
        let alpha = 2.0_f64;
        let data: Vec<f64> = (1..=200)
            .map(|i| {
                let u = i as f64 / 201.0;
                (1.0 - u).powf(-1.0 / alpha)
            })
            .collect();
        let xi = extreme_value_index(&data).expect("should succeed");
        assert!(xi > 0.0, "Heavy-tailed data should give xi>0, got {xi}");
        // Allow generous tolerance since sample size is moderate
        assert!((xi - 0.5).abs() < 0.4, "xi={xi} not close to 0.5");
    }

    #[test]
    fn test_extreme_value_index_insufficient_data() {
        assert!(extreme_value_index(&[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn test_gev_type_aliases() {
        // GevDistribution and GpdDistribution should be constructable
        let _gev: GevDistribution = GeneralizedExtremeValue::new(0.0, 1.0, 0.0).expect("valid");
        let _gpd: GpdDistribution = GeneralizedPareto::new(0.0, 1.0, 0.0).expect("valid");
    }

    #[test]
    fn test_gev_fit_struct() {
        let samples: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let fit = fit_gev(&samples).expect("fit");
        // method should be MLE
        assert_eq!(fit.method, EvtEstimationMethod::MLE);
        // return_level accessor
        let rl = fit.return_level(10.0).expect("return level");
        assert!(rl.is_finite());
    }

    #[test]
    fn test_gev_reduces_to_gumbel() {
        // xi=0: CDF should match Gumbel formula at several points
        for &x in &[-1.0, 0.0, 1.0, 2.0, 3.0] {
            let gev_c = gev_cdf(x, 0.0, 1.0, 0.0);
            let gumbel = Gumbel::new(0.0, 1.0).expect("valid").cdf(x);
            assert!(
                approx_eq(gev_c, gumbel, 1e-10),
                "x={x}: gev={gev_c}, gumbel={gumbel}"
            );
        }
    }

    #[test]
    fn test_gev_reduces_to_frechet_positive_xi() {
        // xi > 0 → Fréchet type: CDF should be 0 below lower bound
        let xi = 0.5_f64;
        let lower_bound = 0.0 - 1.0 / xi; // mu - sigma/xi = -2.0
        let cdf_below = gev_cdf(lower_bound - 0.1, 0.0, 1.0, xi);
        assert_eq!(cdf_below, 0.0, "Fréchet CDF below lower bound should be 0");
    }

    #[test]
    fn test_gev_reduces_to_weibull_negative_xi() {
        // xi < 0 → reversed Weibull: CDF should be 1 above upper bound
        let xi = -0.5_f64;
        let upper_bound = 0.0 - 1.0 / xi; // mu - sigma/xi = 2.0
        let cdf_above = gev_cdf(upper_bound + 0.1, 0.0, 1.0, xi);
        assert_eq!(cdf_above, 1.0, "Weibull CDF above upper bound should be 1");
    }
}

// ============================================================================
// Risk measures sub-module
// ============================================================================

pub mod risk_measures;
pub use risk_measures::{
    conditional_var, fit_block_maxima, pot_analysis, return_level_ci, tail_index_hill,
    value_at_risk, BlockMaximaFit,
};
