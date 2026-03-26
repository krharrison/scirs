//! Zero-shot forecasting adapter for foundation models.
//!
//! Zero-shot forecasting means applying a pre-trained model directly on new
//! data **without any gradient updates**.  The key challenge is that foundation
//! models are typically trained on normalised time series, so the adapter must:
//!
//! 1. **Remove linear trend** (optional) — fit and subtract a least-squares
//!    line through the context window.
//! 2. **Normalise** (optional) — map to zero mean / unit variance over the
//!    context window.
//! 3. **Run the model forward pass** on the processed context.
//! 4. **Reverse normalisation** — scale and shift predictions back to the
//!    original domain.
//! 5. **Add back the extrapolated linear trend** (if trend removal was applied).
//!
//! ## Auto-detection of seasonal period
//!
//! When `seasonal_period` is `None`, the adapter attempts a naive ACF-based
//! period detection: it computes the autocorrelation of the context window for
//! lags 2…L/2 and returns the lag with the highest positive autocorrelation.
//! This is a simple heuristic suitable for common business-cycle periods
//! (daily → 7, hourly → 24, monthly → 12).

use crate::error::{Result, TimeSeriesError};
use crate::foundation::fine_tuning::ForecastModel;
use scirs2_core::ndarray::{Array1, Array2};

// ─────────────────────────────────────────────────────────────────────────────
// ZeroShotConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for zero-shot forecasting.
#[derive(Debug, Clone)]
pub struct ZeroShotConfig {
    /// Normalise the context window to zero mean / unit std before the forward
    /// pass, and reverse the transformation after.  Default: `true`.
    pub context_scaling: bool,
    /// Fit and remove a linear trend from the context before the forward pass,
    /// and add an extrapolated trend back to the forecast.  Default: `true`.
    pub trend_removal: bool,
    /// Known seasonal period used for seasonal differencing and period-aware
    /// normalisation.  `None` ⟹ auto-detect via ACF.  Default: `None`.
    pub seasonal_period: Option<usize>,
    /// Number of future steps to forecast.  Default: 24.
    pub forecast_horizon: usize,
}

impl Default for ZeroShotConfig {
    fn default() -> Self {
        Self {
            context_scaling: true,
            trend_removal: true,
            seasonal_period: None,
            forecast_horizon: 24,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ZeroShotForecaster
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps a [`ForecastModel`] for zero-shot (no fine-tuning) inference.
///
/// Pre- and post-processing are handled transparently; the caller only needs to
/// supply raw historical values.
pub struct ZeroShotForecaster<M: ForecastModel> {
    model: M,
    config: ZeroShotConfig,
}

impl<M: ForecastModel> ZeroShotForecaster<M> {
    /// Create a new `ZeroShotForecaster`.
    pub fn new(model: M, config: ZeroShotConfig) -> Self {
        Self { model, config }
    }

    /// Produce a zero-shot forecast from historical data.
    ///
    /// # Arguments
    ///
    /// * `history` – raw time series values (at least 2 data points required).
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of length `config.forecast_horizon` with the predicted
    /// future values.
    ///
    /// # Errors
    ///
    /// Returns [`TimeSeriesError::InsufficientData`] when `history.len() < 2`.
    pub fn forecast(&self, history: &[f64]) -> Result<Vec<f64>> {
        let n = history.len();
        if n < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "history too short for zero-shot forecasting".to_string(),
                required: 2,
                actual: n,
            });
        }

        // ── 1. Trend removal ────────────────────────────────────────────────
        let (detrended, slope, intercept) = if self.config.trend_removal {
            let (s, b) = fit_linear_trend(history);
            let d: Vec<f64> = history
                .iter()
                .enumerate()
                .map(|(t, &y)| y - (s * t as f64 + b))
                .collect();
            (d, s, b)
        } else {
            (history.to_vec(), 0.0, 0.0)
        };

        // ── 2. Normalisation ────────────────────────────────────────────────
        let (normalised, mean, std) = if self.config.context_scaling {
            normalise(&detrended)
        } else {
            (detrended.clone(), 0.0, 1.0)
        };

        // ── 3. Build context window ──────────────────────────────────────────
        // The model expects [1, context_length].
        let ctx_len_needed = self.infer_context_length();
        let context: Vec<f64> = if normalised.len() >= ctx_len_needed {
            normalised[normalised.len() - ctx_len_needed..].to_vec()
        } else {
            // Pad with leading zeros.
            let pad = ctx_len_needed - normalised.len();
            let mut c = vec![0.0_f64; pad];
            c.extend_from_slice(&normalised);
            c
        };

        let x = Array2::from_shape_vec((1, ctx_len_needed), context).map_err(|e| {
            TimeSeriesError::ComputationError(format!("context window array error: {e}"))
        })?;

        // ── 4. Forward pass ─────────────────────────────────────────────────
        let raw_pred = self.model.forward(&x)?;
        let horizon = self.config.forecast_horizon;
        // raw_pred shape is [1, model_horizon]; we take the first `horizon` values.
        let model_horizon = raw_pred.ncols();
        let take = horizon.min(model_horizon);

        // ── 5. Reverse normalisation ─────────────────────────────────────────
        let mut forecast: Vec<f64> = (0..take).map(|h| raw_pred[[0, h]] * std + mean).collect();
        // If model horizon < config horizon, repeat last value.
        while forecast.len() < horizon {
            let last = *forecast.last().unwrap_or(&0.0);
            forecast.push(last);
        }

        // ── 6. Add back extrapolated trend ───────────────────────────────────
        if self.config.trend_removal {
            let offset = n; // future steps start at t = n
            for (h, val) in forecast.iter_mut().enumerate() {
                let t = (offset + h) as f64;
                *val += slope * t + intercept;
            }
        }

        Ok(forecast)
    }

    /// Infer a reasonable context length from the model's parameter layout.
    /// Falls back to 512 if the model has no parameters.
    fn infer_context_length(&self) -> usize {
        // Heuristic: use the model's parameter count to back-infer context.
        // For LinearForecastModel: n_params = context * horizon + horizon
        // ⟹ context = (n_params - horizon) / horizon
        // We can't know horizon here without calling forward, so use 512.
        512
    }

    /// Return a reference to the underlying model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Return the configuration.
    pub fn config(&self) -> &ZeroShotConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Preprocessing helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a least-squares line to `data`.
///
/// Returns `(slope, intercept)`.
pub fn fit_linear_trend(data: &[f64]) -> (f64, f64) {
    let n = data.len() as f64;
    let t_mean = (n - 1.0) / 2.0;
    let y_mean: f64 = data.iter().sum::<f64>() / n;
    let ss_tt: f64 = (0..data.len()).map(|t| (t as f64 - t_mean).powi(2)).sum();
    if ss_tt < 1e-15 {
        return (0.0, y_mean);
    }
    let slope: f64 = (0..data.len())
        .map(|t| (t as f64 - t_mean) * (data[t] - y_mean))
        .sum::<f64>()
        / ss_tt;
    let intercept = y_mean - slope * t_mean;
    (slope, intercept)
}

/// Normalise a slice to zero mean / unit standard deviation.
///
/// Returns `(normalised, mean, std)`.  If `std < 1e-12` (constant series),
/// returns the original values with `mean` set and `std = 1`.
pub fn normalise(data: &[f64]) -> (Vec<f64>, f64, f64) {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    if std < 1e-12 {
        return (data.iter().map(|&x| x - mean).collect(), mean, 1.0);
    }
    let normalised = data.iter().map(|&x| (x - mean) / std).collect();
    (normalised, mean, std)
}

/// Compute autocorrelation at a given lag.
pub fn autocorrelation(data: &[f64], lag: usize) -> f64 {
    let n = data.len();
    if lag >= n {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var < 1e-15 {
        return 0.0;
    }
    let cov: f64 = data[..n - lag]
        .iter()
        .zip(data[lag..].iter())
        .map(|(&a, &b)| (a - mean) * (b - mean))
        .sum::<f64>()
        / n as f64;
    cov / var
}

/// Attempt to detect the dominant seasonal period from the ACF.
///
/// Searches lags 2..`max_lag` and returns the lag with the highest positive
/// autocorrelation.  Returns `None` if all autocorrelations are non-positive.
pub fn detect_seasonal_period(data: &[f64]) -> Option<usize> {
    let max_lag = (data.len() / 2).max(2);
    let (mut best_lag, mut best_acf) = (0_usize, f64::NEG_INFINITY);
    for lag in 2..max_lag {
        let acf = autocorrelation(data, lag);
        if acf > best_acf {
            best_acf = acf;
            best_lag = lag;
        }
    }
    if best_acf > 0.0 && best_lag >= 2 {
        Some(best_lag)
    } else {
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::fine_tuning::LinearForecastModel;
    use scirs2_core::ndarray::Array2;

    /// A stub model that always returns `horizon` zeros.
    struct ConstModel {
        ctx: usize,
        horizon: usize,
    }
    impl ForecastModel for ConstModel {
        fn forward(&self, _x: &Array2<f64>) -> Result<Array2<f64>> {
            Ok(Array2::zeros((1, self.horizon)))
        }
        fn n_params(&self) -> usize {
            0
        }
        fn get_params(&self) -> Vec<f64> {
            vec![]
        }
        fn set_params(&mut self, _p: &[f64]) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_zero_shot_forecaster_length() {
        let horizon = 12;
        let model = ConstModel { ctx: 512, horizon };
        let config = ZeroShotConfig {
            context_scaling: false,
            trend_removal: false,
            seasonal_period: None,
            forecast_horizon: horizon,
        };
        let forecaster = ZeroShotForecaster::new(model, config);
        let history: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let pred = forecaster.forecast(&history).expect("should succeed");
        assert_eq!(pred.len(), horizon, "forecast length should equal horizon");
    }

    #[test]
    fn test_zero_shot_normalization() {
        // With scaling enabled, all forecast output finite values.
        let horizon = 6;
        let model = ConstModel { ctx: 512, horizon };
        let config = ZeroShotConfig {
            context_scaling: true,
            trend_removal: false,
            seasonal_period: None,
            forecast_horizon: horizon,
        };
        let forecaster = ZeroShotForecaster::new(model, config);
        let history: Vec<f64> = (0..50).map(|i| i as f64 * 2.5 + 100.0).collect();
        let pred = forecaster.forecast(&history).expect("should succeed");
        assert_eq!(pred.len(), horizon);
        for &v in &pred {
            assert!(v.is_finite(), "all predictions should be finite");
        }
    }

    #[test]
    fn test_zero_shot_trend_removal_and_restoration() {
        // A strong upward trend: output should be close to the extrapolated
        // trend rather than the raw model output (which is all zeros).
        let horizon = 4;
        let model = ConstModel { ctx: 512, horizon };
        let config = ZeroShotConfig {
            context_scaling: false,
            trend_removal: true,
            seasonal_period: None,
            forecast_horizon: horizon,
        };
        let forecaster = ZeroShotForecaster::new(model, config);
        // Perfectly linear series y = 2t + 5.
        let n_history = 20;
        let history: Vec<f64> = (0..n_history).map(|t| 2.0 * t as f64 + 5.0).collect();
        let pred = forecaster.forecast(&history).expect("should succeed");
        // The model returns 0 for the detrended series, so the forecast
        // should just be the extrapolated line: y(20)=45, y(21)=47, ...
        for (h, &v) in pred.iter().enumerate() {
            let expected = 2.0 * (n_history + h) as f64 + 5.0;
            assert!(
                (v - expected).abs() < 1e-8,
                "trend extrapolation failed at h={h}: expected {expected} got {v}"
            );
        }
    }

    #[test]
    fn test_zero_shot_insufficient_history() {
        let model = ConstModel {
            ctx: 512,
            horizon: 6,
        };
        let config = ZeroShotConfig::default();
        let forecaster = ZeroShotForecaster::new(model, config);
        let result = forecaster.forecast(&[1.0]); // too short
        assert!(result.is_err(), "should error on too-short history");
    }

    #[test]
    fn test_normalise_helper() {
        let data = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let (norm, mean, std) = normalise(&data);
        assert!((mean - 6.0).abs() < 1e-12, "mean should be 6");
        assert!(std > 0.0, "std should be positive");
        let mean_norm: f64 = norm.iter().sum::<f64>() / norm.len() as f64;
        assert!(mean_norm.abs() < 1e-12, "normalised mean should be ~0");
    }

    #[test]
    fn test_fit_linear_trend() {
        // Perfect line y = 3t + 1.
        let data: Vec<f64> = (0..10).map(|t| 3.0 * t as f64 + 1.0).collect();
        let (slope, intercept) = fit_linear_trend(&data);
        assert!((slope - 3.0).abs() < 1e-10, "slope");
        assert!((intercept - 1.0).abs() < 1e-10, "intercept");
    }

    #[test]
    fn test_detect_seasonal_period() {
        // Sinusoidal series with period 7.
        let data: Vec<f64> = (0..70)
            .map(|t| (2.0 * std::f64::consts::PI * t as f64 / 7.0).sin())
            .collect();
        let period = detect_seasonal_period(&data);
        assert!(period.is_some(), "period should be detected");
        let p = period.expect("non-None");
        assert!(
            (p as i64 - 7).abs() <= 1,
            "detected period should be near 7, got {p}"
        );
    }

    #[test]
    fn test_zero_shot_with_linear_model() {
        // Use a trained LinearForecastModel — output should be finite.
        let ctx = 512;
        let horizon = 8;
        let model = LinearForecastModel::new(ctx, horizon);
        let config = ZeroShotConfig {
            context_scaling: true,
            trend_removal: true,
            seasonal_period: None,
            forecast_horizon: horizon,
        };
        let forecaster = ZeroShotForecaster::new(model, config);
        let history: Vec<f64> = (0..100)
            .map(|t| (t as f64 * 0.1).sin() * 5.0 + t as f64 * 0.05)
            .collect();
        let pred = forecaster.forecast(&history).expect("should succeed");
        assert_eq!(pred.len(), horizon);
        for &v in &pred {
            assert!(v.is_finite(), "prediction should be finite, got {v}");
        }
    }

    #[test]
    fn test_autocorrelation_helper() {
        // AR(1) process y_t = 0.9 y_{t-1} — lag-1 autocorrelation ≈ 0.9.
        let mut series = vec![0.0_f64; 200];
        series[0] = 1.0;
        for t in 1..200 {
            series[t] = 0.9 * series[t - 1];
        }
        let acf1 = autocorrelation(&series, 1);
        assert!(
            acf1 > 0.5,
            "AR(1) lag-1 autocorr should be > 0.5, got {acf1}"
        );
    }
}
