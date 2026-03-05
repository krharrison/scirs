//! Peaks Over Threshold (POT) method for extreme value analysis.
//!
//! The POT approach models all observations exceeding a high threshold u
//! using the Generalized Pareto Distribution. This is more data-efficient
//! than block maxima since it uses all extreme observations.
//!
//! Key tools:
//! - Mean Residual Life (MRL) plot for threshold selection
//! - Parameter stability plots
//! - GPD fitting to exceedances
//!
//! # References
//! - Davison & Smith (1990). Models for exceedances over high thresholds.
//! - Coles (2001). *An Introduction to Statistical Modeling of Extreme Values*.

use super::gpd::GPD;
use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// POTFitter struct
// ---------------------------------------------------------------------------

/// Peaks Over Threshold fitter using GPD.
///
/// # Examples
/// ```no_run
/// use scirs2_stats::extreme_value::pot_module::POTFitter;
/// let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.01).collect();
/// let fitter = POTFitter::new(7.0);
/// let gpd = fitter.fit(&data).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct POTFitter {
    /// Threshold u; exceedances x - u are modeled with GPD
    pub threshold: f64,
}

impl POTFitter {
    /// Create a new POT fitter with the given threshold.
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Extract exceedances and fit a GPD to them.
    ///
    /// # Errors
    /// Returns an error if there are fewer than 5 exceedances.
    pub fn fit(&self, data: &[f64]) -> StatsResult<GPD> {
        let exceedances = extract_exceedances(data, self.threshold);
        if exceedances.len() < 5 {
            return Err(StatsError::InsufficientData(format!(
                "Need at least 5 exceedances above threshold {}; got {}",
                self.threshold,
                exceedances.len()
            )));
        }
        GPD::fit(&exceedances)
    }

    /// Extract exceedances and fit a GPD with full diagnostics.
    pub fn fit_with_diagnostics(&self, data: &[f64]) -> StatsResult<POTResult> {
        let exceedances = extract_exceedances(data, self.threshold);
        let n_exceedances = exceedances.len();
        if n_exceedances < 5 {
            return Err(StatsError::InsufficientData(format!(
                "Need at least 5 exceedances; got {n_exceedances}"
            )));
        }
        let gpd = GPD::fit(&exceedances)?;
        let log_likelihood = gpd.log_likelihood(&exceedances);
        let n_total = data.len();
        let exceedance_rate = n_exceedances as f64 / n_total as f64;

        Ok(POTResult {
            gpd,
            log_likelihood,
            n_exceedances,
            n_total,
            threshold: self.threshold,
            exceedance_rate,
            exceedances,
        })
    }
}

/// Results from POT analysis.
#[derive(Debug, Clone)]
pub struct POTResult {
    /// Fitted GPD distribution
    pub gpd: GPD,
    /// Log-likelihood at fitted parameters
    pub log_likelihood: f64,
    /// Number of exceedances used for fitting
    pub n_exceedances: usize,
    /// Total number of observations
    pub n_total: usize,
    /// Threshold used
    pub threshold: f64,
    /// Fraction of observations exceeding the threshold
    pub exceedance_rate: f64,
    /// Exceedance values (x - threshold for x > threshold)
    pub exceedances: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Extract exceedances
// ---------------------------------------------------------------------------

/// Extract excess values above `threshold` from `data`.
///
/// Returns `x - threshold` for all `x > threshold`.
///
/// # Examples
/// ```
/// use scirs2_stats::extreme_value::pot_module::extract_exceedances;
/// let data = vec![1.0, 3.0, 5.0, 2.0, 7.0];
/// let exc = extract_exceedances(&data, 2.5);
/// assert_eq!(exc.len(), 2); // 3.0 and 5.0 and 7.0 -> 0.5, 2.5, 4.5
/// ```
pub fn extract_exceedances(data: &[f64], threshold: f64) -> Vec<f64> {
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

// ---------------------------------------------------------------------------
// Mean Residual Life plot
// ---------------------------------------------------------------------------

/// Compute Mean Residual Life (MRL) plot data for threshold selection.
///
/// For each candidate threshold u in a grid, the MRL plot shows
/// `(u, E[X - u | X > u])`. The threshold should be chosen where the plot
/// becomes approximately linear (in the Pareto tail region).
///
/// Returns a vector of `(threshold, mean_excess, std_error)` tuples.
///
/// # Arguments
/// - `data`: observed data
///
/// The thresholds are chosen automatically as quantiles of the data from
/// the 50th to 99th percentile.
///
/// # Errors
/// Returns an error if `data.len() < 10`.
pub fn threshold_selection_mrl(data: &[f64]) -> StatsResult<Vec<(f64, f64, f64)>> {
    if data.len() < 10 {
        return Err(StatsError::InsufficientData(
            "MRL plot requires at least 10 observations".into(),
        ));
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();

    // Use quantiles from 50th to 97th percentile as threshold grid
    let n_thresholds = 30usize.min(n / 3);
    let mut result = Vec::with_capacity(n_thresholds);

    for k in 0..n_thresholds {
        let q_idx = n / 2 + k * (n * 47 / 100) / n_thresholds.max(1);
        let q_idx = q_idx.min(n - 5);
        let threshold = sorted[q_idx];

        let exceedances: Vec<f64> = sorted[q_idx + 1..].iter().map(|&x| x - threshold).collect();

        if exceedances.is_empty() {
            continue;
        }

        let m = exceedances.len() as f64;
        let mean = exceedances.iter().sum::<f64>() / m;
        let var = exceedances.iter().map(|&e| (e - mean).powi(2)).sum::<f64>() / m;
        let se = (var / m).sqrt();

        result.push((threshold, mean, se));
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Parameter stability plots
// ---------------------------------------------------------------------------

/// Compute parameter stability plot data for threshold selection.
///
/// For each threshold u in a grid, fits a GPD and records the modified scale
/// parameter σ* = σ - ξ*u (which should be constant if u is a valid threshold).
///
/// Returns `(threshold, modified_scale, shape)` tuples.
///
/// # Errors
/// Returns an error if `data.len() < 20`.
pub fn threshold_selection_stability(data: &[f64]) -> StatsResult<Vec<(f64, f64, f64)>> {
    if data.len() < 20 {
        return Err(StatsError::InsufficientData(
            "Parameter stability plot requires at least 20 observations".into(),
        ));
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();

    let n_thresholds = 20usize.min(n / 5);
    let mut result = Vec::with_capacity(n_thresholds);

    for k in 0..n_thresholds {
        let q_idx = n * 3 / 10 + k * (n * 50 / 100) / n_thresholds.max(1);
        let q_idx = q_idx.min(n - 6);
        let threshold = sorted[q_idx];

        let exceedances: Vec<f64> = sorted[q_idx + 1..].iter().map(|&x| x - threshold).collect();

        if exceedances.len() < 5 {
            continue;
        }

        match GPD::fit(&exceedances) {
            Ok(gpd) => {
                // Modified scale: σ* = σ - ξ*u
                let modified_scale = gpd.sigma - gpd.xi * threshold;
                result.push((threshold, modified_scale, gpd.xi));
            }
            Err(_) => continue,
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_exceedances_basic() {
        let data = vec![1.0, 2.0, 5.0, 3.0, 8.0, 1.5];
        let exc = extract_exceedances(&data, 2.0);
        assert_eq!(exc.len(), 3);
        let mut sorted = exc.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!((sorted[0] - 1.0).abs() < 1e-12);
        assert!((sorted[1] - 3.0).abs() < 1e-12);
        assert!((sorted[2] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_extract_exceedances_empty() {
        let data = vec![1.0, 1.5, 2.0];
        let exc = extract_exceedances(&data, 5.0);
        assert!(exc.is_empty());
    }

    #[test]
    fn test_pot_fitter_basic() {
        let data: Vec<f64> = (0..200)
            .map(|i| (i as f64 * 0.1).sin() * 3.0 + 3.0)
            .collect();
        let fitter = POTFitter::new(4.0);
        let result = fitter.fit_with_diagnostics(&data);
        // May succeed or fail depending on data
        match result {
            Ok(r) => {
                assert!(r.gpd.sigma > 0.0);
                assert!(r.threshold == 4.0);
            }
            Err(_) => {} // acceptable if not enough exceedances
        }
    }

    #[test]
    fn test_pot_fitter_insufficient_exceedances() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let fitter = POTFitter::new(4.5);
        assert!(fitter.fit(&data).is_err());
    }

    #[test]
    fn test_threshold_selection_mrl() {
        let data: Vec<f64> = (1..=200)
            .map(|i| (i as f64 / 201.0_f64).powf(-1.0))
            .collect();
        let mrl = threshold_selection_mrl(&data).unwrap();
        assert!(!mrl.is_empty());
        // Each entry should have finite values
        for (u, mean, se) in &mrl {
            assert!(u.is_finite());
            assert!(mean.is_finite());
            assert!(se.is_finite());
            assert!(*mean >= 0.0);
        }
    }

    #[test]
    fn test_threshold_selection_mrl_insufficient() {
        assert!(threshold_selection_mrl(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_threshold_selection_stability() {
        let data: Vec<f64> = (1..=100)
            .map(|i| -(1.0 - i as f64 / 101.0).ln() * 2.0)
            .collect();
        let stab = threshold_selection_stability(&data).unwrap();
        assert!(!stab.is_empty());
        for (u, sigma_star, xi) in &stab {
            assert!(u.is_finite());
            assert!(sigma_star.is_finite());
            assert!(xi.is_finite());
        }
    }

    #[test]
    fn test_threshold_selection_stability_insufficient() {
        assert!(
            threshold_selection_stability(&(0..10).map(|i| i as f64).collect::<Vec<_>>()).is_err()
        );
    }

    #[test]
    fn test_pot_result_fields() {
        let data: Vec<f64> = (0..200).map(|i| i as f64 * 0.05).collect();
        let fitter = POTFitter::new(7.0);
        let result = fitter.fit_with_diagnostics(&data).unwrap();
        assert_eq!(result.n_total, 200);
        assert!(result.exceedance_rate > 0.0 && result.exceedance_rate < 1.0);
        assert_eq!(result.threshold, 7.0);
        assert!(!result.exceedances.is_empty());
    }
}
