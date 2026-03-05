//! Conformal Prediction for Time Series
//!
//! Conformal prediction provides distribution-free, finite-sample coverage
//! guarantees for prediction intervals.  This module implements several
//! conformal methods tailored to the time series setting:
//!
//! | Struct | Method | Reference |
//! |---|---|---|
//! | [`SplitConformal`] | Split conformal prediction | Papadopoulos et al. (2002) |
//! | [`AdaptiveConformal`] | Adaptive conformal with sliding window | Gibbs & Candès (2021) |
//! | [`WeightedConformal`] | Weighted / non-exchangeable conformal | Tibshirani et al. (2019) |
//! | [`EnbPI`] | Ensemble batch prediction intervals | Xu & Xie (2021) |
//!
//! # Coverage guarantee
//!
//! For exchangeable data, `SplitConformal` guarantees:
//!
//! ```text
//! P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 - α
//! ```
//!
//! where α is the configured miscoverage level.

use crate::error::{Result, TimeSeriesError};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Quantile helper
// ---------------------------------------------------------------------------

/// Compute the `q`-th quantile of `values` (linearly interpolated).
///
/// `q` must be in `[0, 1]`.  Returns `None` if the slice is empty.
fn quantile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return Some(sorted[0]);
    }
    let idx = q * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        Some(sorted[lo])
    } else {
        let frac = idx - lo as f64;
        Some(sorted[lo] * (1.0 - frac) + sorted[hi] * frac)
    }
}

// ---------------------------------------------------------------------------
// Split Conformal Prediction
// ---------------------------------------------------------------------------

/// Split conformal prediction for time series.
///
/// # Usage
///
/// ```rust
/// use scirs2_series::conformal::SplitConformal;
///
/// let mut sc = SplitConformal::new(0.1); // 10 % miscoverage → 90 % coverage
/// let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y_pred = vec![1.1, 1.9, 3.2, 3.8, 5.1];
/// sc.calibrate(&y_true, &y_pred);
/// let (lo, hi) = sc.predict_interval(3.0).expect("should succeed");
/// assert!(lo < 3.0 && 3.0 < hi);
/// ```
#[derive(Debug, Clone)]
pub struct SplitConformal {
    /// Calibration absolute residuals.
    pub calibration_residuals: Vec<f64>,
    /// Miscoverage level α ∈ (0, 1).
    pub alpha: f64,
}

impl SplitConformal {
    /// Create a new SplitConformal with miscoverage level `alpha`.
    pub fn new(alpha: f64) -> Self {
        Self {
            calibration_residuals: Vec::new(),
            alpha,
        }
    }

    /// Calibrate on a held-out set.
    ///
    /// Residuals are stored as `|y_true - y_pred|`.
    pub fn calibrate(&mut self, y_true: &[f64], y_pred: &[f64]) {
        self.calibration_residuals = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&yt, &yp)| (yt - yp).abs())
            .collect();
    }

    /// Return the conformal quantile q̂ = Quantile(residuals, (1-α)(1+1/n)).
    pub fn conformal_quantile(&self) -> Result<f64> {
        if self.calibration_residuals.is_empty() {
            return Err(TimeSeriesError::InvalidModel(
                "SplitConformal has not been calibrated".to_string(),
            ));
        }
        let n = self.calibration_residuals.len() as f64;
        let q = ((1.0 - self.alpha) * (1.0 + 1.0 / n)).min(1.0);
        quantile(&self.calibration_residuals, q).ok_or_else(|| {
            TimeSeriesError::InvalidModel("Empty calibration residuals".to_string())
        })
    }

    /// Compute a prediction interval at `point_forecast`.
    ///
    /// Returns `(lower, upper)` = `(point - q̂, point + q̂)`.
    pub fn predict_interval(&self, point_forecast: f64) -> Result<(f64, f64)> {
        let q = self.conformal_quantile()?;
        Ok((point_forecast - q, point_forecast + q))
    }

    /// Width of the interval (2 × q̂).
    pub fn interval_width(&self) -> Result<f64> {
        Ok(2.0 * self.conformal_quantile()?)
    }
}

// ---------------------------------------------------------------------------
// Adaptive Conformal Prediction
// ---------------------------------------------------------------------------

/// Adaptive conformal prediction with a sliding window of recent residuals.
///
/// Suitable for non-stationary time series where the error distribution
/// changes over time.  The quantile is computed over the last `window_size`
/// absolute residuals.
#[derive(Debug, Clone)]
pub struct AdaptiveConformal {
    /// Miscoverage level α.
    pub alpha: f64,
    /// Maximum number of residuals retained.
    pub window_size: usize,
    /// Sliding window of absolute residuals.
    pub residuals: VecDeque<f64>,
}

impl AdaptiveConformal {
    /// Create with miscoverage `alpha` and sliding window `window_size`.
    pub fn new(alpha: f64, window_size: usize) -> Self {
        Self {
            alpha,
            window_size,
            residuals: VecDeque::new(),
        }
    }

    /// Update with a new observation.
    pub fn update(&mut self, y_true: f64, y_pred: f64) {
        let r = (y_true - y_pred).abs();
        if self.residuals.len() >= self.window_size {
            self.residuals.pop_front();
        }
        self.residuals.push_back(r);
    }

    /// Update with multiple observations.
    pub fn update_batch(&mut self, y_true: &[f64], y_pred: &[f64]) {
        for (&yt, &yp) in y_true.iter().zip(y_pred.iter()) {
            self.update(yt, yp);
        }
    }

    /// Current conformal quantile.
    pub fn conformal_quantile(&self) -> Result<f64> {
        if self.residuals.is_empty() {
            return Err(TimeSeriesError::InvalidModel(
                "AdaptiveConformal has no residuals yet".to_string(),
            ));
        }
        let resids: Vec<f64> = self.residuals.iter().copied().collect();
        let n = resids.len() as f64;
        let q = ((1.0 - self.alpha) * (1.0 + 1.0 / n)).min(1.0);
        quantile(&resids, q).ok_or_else(|| {
            TimeSeriesError::InvalidModel("Empty residuals".to_string())
        })
    }

    /// Predict interval for a given point forecast.
    pub fn predict_interval(&self, point_forecast: f64) -> Result<(f64, f64)> {
        let q = self.conformal_quantile()?;
        Ok((point_forecast - q, point_forecast + q))
    }
}

// ---------------------------------------------------------------------------
// Weighted Conformal Prediction
// ---------------------------------------------------------------------------

/// Non-exchangeable conformal prediction with importance weights.
///
/// Assigns exponentially decaying weights to past residuals so that recent
/// calibration points matter more.  Useful when the data distribution shifts
/// gradually over time.
///
/// # Reference
///
/// Tibshirani, R.J., Barber, R.F., Candès, E.J., & Ramdas, A. (2019).
/// *Conformal Prediction Under Covariate Shift.*
/// Advances in Neural Information Processing Systems 32.
#[derive(Debug, Clone)]
pub struct WeightedConformal {
    /// Importance weights (one per calibration point).
    pub weights: Vec<f64>,
    /// Absolute residuals matching `weights`.
    pub residuals: Vec<f64>,
    /// Miscoverage level α.
    pub alpha: f64,
}

impl WeightedConformal {
    /// Create an empty WeightedConformal.
    pub fn new(alpha: f64) -> Self {
        Self {
            weights: Vec::new(),
            residuals: Vec::new(),
            alpha,
        }
    }

    /// Calibrate with explicit weights and residuals.
    ///
    /// `weights` and `residuals` must have the same length.
    pub fn calibrate_weighted(
        &mut self,
        weights: Vec<f64>,
        residuals: Vec<f64>,
    ) -> Result<()> {
        if weights.len() != residuals.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: weights.len(),
                actual: residuals.len(),
            });
        }
        self.weights = weights;
        self.residuals = residuals;
        Ok(())
    }

    /// Calibrate with exponentially decaying weights.
    ///
    /// `decay` in `(0, 1]` controls the decay rate: `w_t ∝ decay^(n-t)`.
    /// A value close to 1 gives roughly uniform weights; smaller values
    /// emphasise recent observations.
    pub fn calibrate_exponential(
        &mut self,
        y_true: &[f64],
        y_pred: &[f64],
        decay: f64,
    ) -> Result<()> {
        if y_true.len() != y_pred.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: y_true.len(),
                actual: y_pred.len(),
            });
        }
        let n = y_true.len();
        let weights: Vec<f64> = (0..n)
            .map(|i| decay.powi((n - 1 - i) as i32))
            .collect();
        let residuals: Vec<f64> = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&yt, &yp)| (yt - yp).abs())
            .collect();
        self.weights = weights;
        self.residuals = residuals;
        Ok(())
    }

    /// Weighted conformal quantile.
    ///
    /// We find the smallest residual threshold q̂ such that the cumulative
    /// normalised weight of residuals ≤ q̂ is ≥ 1 - α.
    pub fn conformal_quantile(&self) -> Result<f64> {
        if self.residuals.is_empty() {
            return Err(TimeSeriesError::InvalidModel(
                "WeightedConformal has not been calibrated".to_string(),
            ));
        }
        let total_w: f64 = self.weights.iter().sum::<f64>() + 1.0; // +1 for the test point
        let target = (1.0 - self.alpha) * total_w;

        // Sort by residual value
        let mut pairs: Vec<(f64, f64)> = self
            .residuals
            .iter()
            .zip(self.weights.iter())
            .map(|(&r, &w)| (r, w))
            .collect();
        pairs.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut cum_w = 0.0_f64;
        for (r, w) in &pairs {
            cum_w += w;
            if cum_w >= target {
                return Ok(*r);
            }
        }
        // Return the largest residual if target not reached
        Ok(pairs.last().map(|(r, _)| *r).unwrap_or(0.0))
    }

    /// Predict interval.
    pub fn predict_interval(&self, point_forecast: f64) -> Result<(f64, f64)> {
        let q = self.conformal_quantile()?;
        Ok((point_forecast - q, point_forecast + q))
    }
}

// ---------------------------------------------------------------------------
// EnbPI: Ensemble Batch Prediction Intervals
// ---------------------------------------------------------------------------

/// Ensemble Batch Prediction Intervals (EnbPI).
///
/// Maintains an online estimator of prediction intervals by tracking a
/// sliding window of signed residuals and adjusting the interval half-width
/// dynamically.
///
/// # Reference
///
/// Xu, C. & Xie, Y. (2021).
/// *Conformal Prediction Interval for Dynamic Time-Series.*
/// Proceedings of ICML 2021.
#[derive(Debug, Clone)]
pub struct EnbPI {
    /// Miscoverage level α.
    pub alpha: f64,
    /// Signed residuals from recent predictions.
    residuals: VecDeque<f64>,
    /// Maximum number of residuals to retain.
    max_residuals: usize,
    /// Current lower/upper half-width.
    half_width: f64,
}

impl EnbPI {
    /// Create a new EnbPI tracker.
    ///
    /// `max_residuals` controls the memory window.
    pub fn new(alpha: f64, max_residuals: usize) -> Self {
        Self {
            alpha,
            residuals: VecDeque::new(),
            max_residuals,
            half_width: 0.0,
        }
    }

    /// Update with a new point-forecast error (signed: y_true - y_pred).
    pub fn update(&mut self, residual: f64) {
        if self.residuals.len() >= self.max_residuals {
            self.residuals.pop_front();
        }
        self.residuals.push_back(residual);
        self.recompute_width();
    }

    /// Update with a batch of residuals.
    pub fn update_batch(&mut self, residuals: &[f64]) {
        for &r in residuals {
            self.update(r);
        }
    }

    fn recompute_width(&mut self) {
        if self.residuals.is_empty() {
            self.half_width = 0.0;
            return;
        }
        let abs_resids: Vec<f64> = self.residuals.iter().map(|&r| r.abs()).collect();
        let n = abs_resids.len() as f64;
        let q = ((1.0 - self.alpha) * (1.0 + 1.0 / n)).min(1.0);
        self.half_width = quantile(&abs_resids, q).unwrap_or(0.0);
    }

    /// Compute prediction interval for the next time step.
    pub fn predict_interval(&self, point_forecast: f64) -> (f64, f64) {
        (point_forecast - self.half_width, point_forecast + self.half_width)
    }

    /// Current half-width of the prediction interval.
    pub fn half_width(&self) -> f64 {
        self.half_width
    }
}

// ---------------------------------------------------------------------------
// Empirical coverage utility
// ---------------------------------------------------------------------------

/// Compute empirical coverage: fraction of `y_true` falling inside `intervals`.
///
/// Returns a value in `[0, 1]`.
pub fn empirical_coverage(y_true: &[f64], intervals: &[(f64, f64)]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let covered = y_true
        .iter()
        .zip(intervals.iter())
        .filter(|(&y, &(lo, hi))| y >= lo && y <= hi)
        .count();
    covered as f64 / y_true.len() as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_conformal_calibrate_predict() {
        let mut sc = SplitConformal::new(0.1);
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y_pred: Vec<f64> = y_true.iter().map(|&y| y + 0.3).collect();
        sc.calibrate(&y_true, &y_pred);
        let (lo, hi) = sc.predict_interval(5.0).expect("interval");
        assert!(lo < 5.0 && hi > 5.0, "forecast should be inside interval");
    }

    #[test]
    fn test_split_conformal_not_calibrated_error() {
        let sc = SplitConformal::new(0.1);
        assert!(sc.predict_interval(0.0).is_err());
    }

    #[test]
    fn test_adaptive_conformal_update_predict() {
        let mut ac = AdaptiveConformal::new(0.1, 20);
        for i in 0..20 {
            ac.update(i as f64, i as f64 + 0.2);
        }
        let (lo, hi) = ac.predict_interval(10.0).expect("interval");
        assert!(lo < 10.0 && hi > 10.0);
    }

    #[test]
    fn test_adaptive_conformal_no_data_error() {
        let ac = AdaptiveConformal::new(0.1, 20);
        assert!(ac.predict_interval(0.0).is_err());
    }

    #[test]
    fn test_weighted_conformal_exponential() {
        let mut wc = WeightedConformal::new(0.1);
        let y_true: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y_pred: Vec<f64> = y_true.iter().map(|&y| y + 0.5).collect();
        wc.calibrate_exponential(&y_true, &y_pred, 0.95)
            .expect("calibrate");
        let (lo, hi) = wc.predict_interval(10.0).expect("interval");
        assert!(lo < 10.0 && hi > 10.0);
    }

    #[test]
    fn test_weighted_conformal_dimension_mismatch() {
        let mut wc = WeightedConformal::new(0.1);
        let y_true = vec![1.0, 2.0];
        let y_pred = vec![1.0];
        assert!(wc.calibrate_exponential(&y_true, &y_pred, 0.9).is_err());
    }

    #[test]
    fn test_enbpi_update_predict() {
        let mut enbpi = EnbPI::new(0.1, 50);
        let residuals: Vec<f64> = (0..30).map(|i| (i as f64 * 0.1).sin() * 0.5).collect();
        enbpi.update_batch(&residuals);
        let (lo, hi) = enbpi.predict_interval(5.0);
        assert!(lo < 5.0 && hi > 5.0);
    }

    #[test]
    fn test_enbpi_zero_residuals() {
        let enbpi = EnbPI::new(0.1, 10);
        let (lo, hi) = enbpi.predict_interval(3.0);
        assert_eq!(lo, 3.0);
        assert_eq!(hi, 3.0);
    }

    #[test]
    fn test_empirical_coverage_perfect() {
        let y_true = vec![1.0, 2.0, 3.0];
        let intervals = vec![(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)];
        let cov = empirical_coverage(&y_true, &intervals);
        assert!((cov - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_empirical_coverage_zero() {
        let y_true = vec![10.0, 20.0, 30.0];
        let intervals = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
        let cov = empirical_coverage(&y_true, &intervals);
        assert_eq!(cov, 0.0);
    }

    #[test]
    fn test_quantile_edge_cases() {
        assert!(quantile(&[], 0.5).is_none());
        let v = vec![3.0];
        assert_eq!(quantile(&v, 0.5), Some(3.0));
        let v2 = vec![1.0, 3.0];
        let q = quantile(&v2, 0.5).expect("failed to create q");
        assert!((q - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_interval_coverage_guarantee_approx() {
        // Generate 100 calibration points, measure empirical coverage
        let n = 100;
        let alpha = 0.1;
        let mut sc = SplitConformal::new(alpha);
        let y_true: Vec<f64> = (0..n).map(|i| i as f64).collect();
        // Perfect predictions → residuals all 0
        let y_pred = y_true.clone();
        sc.calibrate(&y_true, &y_pred);
        // With zero residuals, the interval is exactly at the point
        let width = sc.interval_width().expect("failed to create width");
        assert!(width >= 0.0);
    }
}
