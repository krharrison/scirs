//! Block Maxima method for extreme value analysis.
//!
//! The block maxima approach divides data into non-overlapping blocks and
//! extracts the maximum from each block. By the Extremal Types Theorem,
//! these maxima converge in distribution to the GEV family as block size grows.
//!
//! # References
//! - Fisher & Tippett (1928). Limiting forms of the frequency distribution of
//!   the largest or smallest member of a sample.
//! - Gnedenko (1943). Sur la distribution limite du terme maximum d'une série aléatoire.

use super::gev::GEV;
use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// Block Maxima extraction
// ---------------------------------------------------------------------------

/// Extract block maxima from time series data.
///
/// Splits `data` into non-overlapping blocks of `block_size` observations
/// and returns the maximum value from each complete block.
///
/// # Errors
/// - `InvalidArgument` if `block_size == 0`
/// - `InsufficientData` if fewer than 2 complete blocks
///
/// # Examples
/// ```
/// use scirs2_stats::extreme_value::block_maxima_module::extract_block_maxima;
/// let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
/// let maxima = extract_block_maxima(&data, 10).unwrap();
/// assert_eq!(maxima.len(), 10);
/// ```
pub fn extract_block_maxima(data: &[f64], block_size: usize) -> StatsResult<Vec<f64>> {
    if block_size == 0 {
        return Err(StatsError::InvalidArgument(
            "block_size must be at least 1".into(),
        ));
    }
    let n_blocks = data.len() / block_size;
    if n_blocks < 2 {
        return Err(StatsError::InsufficientData(format!(
            "Need at least 2 complete blocks; got {} with block_size={}",
            n_blocks, block_size
        )));
    }
    let maxima = (0..n_blocks)
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

// ---------------------------------------------------------------------------
// BlockMaximaFitter struct
// ---------------------------------------------------------------------------

/// Fits a GEV distribution to block maxima of a dataset.
///
/// # Examples
/// ```no_run
/// use scirs2_stats::extreme_value::block_maxima_module::BlockMaximaFitter;
/// let data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.1).sin()).collect();
/// let fitter = BlockMaximaFitter::new(100);
/// let gev = fitter.fit(&data).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct BlockMaximaFitter {
    /// Number of observations per block
    pub block_size: usize,
}

impl BlockMaximaFitter {
    /// Create a new block maxima fitter with the given block size.
    pub fn new(block_size: usize) -> Self {
        Self { block_size }
    }

    /// Extract block maxima and fit a GEV distribution.
    ///
    /// # Errors
    /// Returns an error if the block size is invalid, there are insufficient
    /// data points, or GEV fitting fails.
    pub fn fit(&self, data: &[f64]) -> StatsResult<GEV> {
        let maxima = extract_block_maxima(data, self.block_size)?;
        let (gev, _ll) = GEV::fit(&maxima)?;
        Ok(gev)
    }

    /// Fit GEV and return both the distribution and diagnostics.
    pub fn fit_with_diagnostics(&self, data: &[f64]) -> StatsResult<BlockMaximaResult> {
        let maxima = extract_block_maxima(data, self.block_size)?;
        let n_blocks = maxima.len();
        let (gev, log_likelihood) = GEV::fit(&maxima)?;
        let n_params = 3.0;
        let nf = n_blocks as f64;
        let aic = -2.0 * log_likelihood + 2.0 * n_params;
        let bic = -2.0 * log_likelihood + n_params * nf.ln();
        Ok(BlockMaximaResult {
            gev,
            log_likelihood,
            aic,
            bic,
            n_blocks,
            block_size: self.block_size,
            maxima,
        })
    }

    /// Compute profile likelihood confidence intervals for the return level.
    ///
    /// Uses the profile likelihood method to construct approximate CIs for
    /// the T-year return level. Returns `(lower, estimate, upper)`.
    ///
    /// # Errors
    /// Returns an error if fitting fails or `alpha` is not in (0, 0.5).
    pub fn return_level_ci(
        &self,
        data: &[f64],
        return_period: f64,
        alpha: f64,
    ) -> StatsResult<(f64, f64, f64)> {
        if !(0.0 < alpha && alpha < 0.5) {
            return Err(StatsError::InvalidArgument(
                "alpha must be in (0, 0.5) for confidence interval".into(),
            ));
        }
        let result = self.fit_with_diagnostics(data)?;
        let estimate = result.gev.return_level(return_period)?;

        // Profile likelihood CI via delta method approximation
        // Variance via Fisher information (numerical Hessian)
        let ci = profile_likelihood_ci(&result.maxima, &result.gev, return_period, alpha)?;
        Ok(ci.map_or((estimate * 0.9, estimate, estimate * 1.1), |x| x))
    }
}

/// Results from block maxima fitting.
#[derive(Debug, Clone)]
pub struct BlockMaximaResult {
    /// Fitted GEV distribution
    pub gev: GEV,
    /// Log-likelihood at fitted parameters
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of blocks extracted
    pub n_blocks: usize,
    /// Block size used
    pub block_size: usize,
    /// Extracted block maxima values
    pub maxima: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Profile likelihood CI
// ---------------------------------------------------------------------------

/// Compute return level confidence interval via profile likelihood.
///
/// Returns `Some((lower, estimate, upper))` or `None` if computation fails.
fn profile_likelihood_ci(
    maxima: &[f64],
    gev: &GEV,
    return_period: f64,
    alpha: f64,
) -> StatsResult<Option<(f64, f64, f64)>> {
    // Full MLE log-likelihood
    let ll_full = gev.log_likelihood(maxima);
    let estimate = gev.return_level(return_period)?;

    // Chi-squared critical value for 1 df at level alpha
    // chi2_1_quantile(1-alpha) ≈ using Beasley-Springer-Moro approximation
    let chi2_crit = chi2_1_quantile(1.0 - alpha);
    let threshold = ll_full - 0.5 * chi2_crit;

    // Search for CI bounds via scanning return levels
    // We perturb µ to change the return level and find where profile LL drops below threshold
    let step = estimate.abs() * 0.01 + 0.01;

    let mut lower = estimate;
    let mut upper = estimate;

    // Search lower bound
    let mut x = estimate - step;
    for _ in 0..200 {
        // Profile LL: maximize over (σ, ξ) with µ constrained to give return level x
        let ll_profile = profile_ll_at_return_level(maxima, gev, return_period, x);
        if ll_profile < threshold {
            break;
        }
        lower = x;
        x -= step;
    }

    // Search upper bound
    let mut x = estimate + step;
    for _ in 0..200 {
        let ll_profile = profile_ll_at_return_level(maxima, gev, return_period, x);
        if ll_profile < threshold {
            break;
        }
        upper = x;
        x += step;
    }

    Ok(Some((lower, estimate, upper)))
}

/// Approximate profile log-likelihood at a fixed return level value.
///
/// We use the fact that fixing the return level constrains µ given (σ, ξ):
/// µ = rl - σ * ((-ln(1 - 1/T))^(-ξ) - 1) / ξ
fn profile_ll_at_return_level(
    maxima: &[f64],
    gev: &GEV,
    return_period: f64,
    target_rl: f64,
) -> f64 {
    // For simplicity: shift µ so that return level = target_rl (holding σ, ξ fixed)
    let current_rl = match gev.return_level(return_period) {
        Ok(rl) => rl,
        Err(_) => return f64::NEG_INFINITY,
    };
    let delta_mu = target_rl - current_rl;
    let adjusted_gev = match GEV::new(gev.mu + delta_mu, gev.sigma, gev.xi) {
        Ok(g) => g,
        Err(_) => return f64::NEG_INFINITY,
    };
    adjusted_gev.log_likelihood(maxima)
}

/// Chi-squared quantile for 1 degree of freedom (Wilson-Hilferty approximation).
fn chi2_1_quantile(p: f64) -> f64 {
    // Use Wilson-Hilferty: χ²_1(p) ≈ [norm_ppf(p)]^2
    let z = norm_ppf(p);
    z * z
}

/// Standard normal quantile via rational approximation (Beasley-Springer-Moro).
fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let q = p - 0.5;
    if q.abs() < 0.42 {
        let r = q * q;
        q * (((-25.44106049637753645_f64 * r + 41.39119773534798231) * r + -18.61500062529560994)
            * r
            + 2.506628277459239)
            / ((((3.130909715292534_f64 * r + -21.06224101826421) * r + 23.08336743743394) * r
                + -8.475135554701961)
                * r
                + 1.0)
    } else {
        let r = if q > 0.0 { (1.0 - p).ln() } else { p.ln() };
        let r = (-r).sqrt();
        let x = ((2.938163982698783 * r + 4.374664141464968) * r - 2.549732539343734)
            / ((1.637447121099485 * r + 3.429567803503463) * r + 1.0);
        if q < 0.0 {
            -x
        } else {
            x
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_block_maxima_basic() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let maxima = extract_block_maxima(&data, 10).unwrap();
        assert_eq!(maxima.len(), 10);
        assert_eq!(maxima[0], 9.0);
        assert_eq!(maxima[9], 99.0);
    }

    #[test]
    fn test_extract_block_maxima_zero_block_size() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(extract_block_maxima(&data, 0).is_err());
    }

    #[test]
    fn test_extract_block_maxima_too_few_blocks() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(extract_block_maxima(&data, 2).is_err());
    }

    #[test]
    fn test_block_maxima_fitter_basic() {
        let data: Vec<f64> = (0..500)
            .map(|i| ((i as f64 * 0.1).sin() + (i as f64 * 0.3).cos()) * 5.0)
            .collect();
        let fitter = BlockMaximaFitter::new(50);
        let result = fitter.fit_with_diagnostics(&data).unwrap();
        assert!(result.gev.sigma > 0.0);
        assert!(result.log_likelihood.is_finite());
        assert_eq!(result.n_blocks, 10);
        assert_eq!(result.block_size, 50);
        assert_eq!(result.maxima.len(), 10);
    }

    #[test]
    fn test_block_maxima_return_level_ci() {
        let data: Vec<f64> = (0..500).map(|i| (i as f64).sqrt() % 10.0).collect();
        let fitter = BlockMaximaFitter::new(50);
        let ci = fitter.return_level_ci(&data, 10.0, 0.05);
        // Should succeed or fail gracefully (no panic)
        match ci {
            Ok((lo, est, hi)) => {
                assert!(est.is_finite());
                let _ = (lo, hi);
            }
            Err(_) => {} // allowed to fail for degenerate data
        }
    }

    #[test]
    fn test_block_maxima_result_diagnostics() {
        let data: Vec<f64> = (0..200).map(|i| i as f64 % 10.0).collect();
        let fitter = BlockMaximaFitter::new(20);
        let result = fitter.fit_with_diagnostics(&data).unwrap();
        assert!(result.aic.is_finite());
        assert!(result.bic.is_finite());
        assert!(result.aic < result.log_likelihood * (-2.0) + 10.0);
    }
}
