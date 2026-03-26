//! Prompt-based time series conditioning.
//!
//! This module provides an API for conditioning time series forecasts on textual
//! or structured prompts, enabling domain knowledge injection into model outputs.
//!
//! ## Overview
//!
//! - [`TimeSeriesPrompt`] — declarative description of expected behaviour
//! - [`PromptConditioner`] — blend a base forecast with prompt-derived signals
//! - [`ConditionedForecast`] — output with prediction intervals and decomposition
//!
//! ## Example
//!
//! ```rust,no_run
//! use scirs2_series::prompt_conditioning::{
//!     PromptConditioner, PromptConfig, TimeSeriesPrompt, TrendDirection,
//! };
//!
//! let config = PromptConfig { horizon: 12, blend_weight: 0.3, ..Default::default() };
//! let conditioner = PromptConditioner::new(config);
//! let base = vec![0.0f64; 12];
//! let prompts = vec![TimeSeriesPrompt::Trend { direction: TrendDirection::Up, magnitude: 5.0 }];
//! let result = conditioner.condition_forecast(&base, &prompts);
//! assert_eq!(result.values.len(), 12);
//! ```

use crate::error::TimeSeriesError;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Direction enum
// ─────────────────────────────────────────────────────────────────────────────

/// Direction of a trend component in a [`TimeSeriesPrompt::Trend`] variant.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    /// Monotonically increasing trend.
    Up,
    /// Monotonically decreasing trend.
    Down,
    /// No appreciable trend.
    Flat,
    /// Alternating / sinusoidal trend.
    Oscillating,
}

// ─────────────────────────────────────────────────────────────────────────────
// TimeSeriesPrompt
// ─────────────────────────────────────────────────────────────────────────────

/// A declarative description of expected time series behaviour over the forecast horizon.
///
/// Multiple prompts can be combined; their generated signals are averaged before
/// blending with the base forecast.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum TimeSeriesPrompt {
    /// A monotonic or oscillating trend over the horizon.
    Trend {
        /// Direction of the trend.
        direction: TrendDirection,
        /// Magnitude of the trend effect (e.g. 5.0 = 5 units end-to-end change).
        magnitude: f64,
    },
    /// Periodic seasonal pattern.
    Seasonal {
        /// Length of one complete cycle (in timesteps).
        period: usize,
        /// Peak-to-trough amplitude.
        amplitude: f64,
    },
    /// A single anomalous spike or dip at a specific forecast step.
    Anomaly {
        /// Zero-indexed step within the forecast horizon at which the anomaly occurs.
        at_step: usize,
        /// Magnitude of the spike (positive = up, negative = down).
        magnitude: f64,
    },
    /// An abrupt level shift starting at a specific forecast step.
    Level {
        /// Magnitude of the shift.
        shift: f64,
        /// Zero-indexed step at which the shift occurs.
        at_step: usize,
    },
    /// Additive Gaussian noise with the given standard deviation.
    Noise {
        /// Standard deviation of the noise signal.
        std: f64,
    },
    /// Arbitrary user-supplied prior signal.
    Custom(Vec<f64>),
}

// ─────────────────────────────────────────────────────────────────────────────
// PromptConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`PromptConditioner`].
#[derive(Debug, Clone)]
pub struct PromptConfig {
    /// Forecast horizon (number of steps).
    pub horizon: usize,
    /// Weight given to the prompt signal when blending with the base forecast.
    /// `final = (1 - blend_weight) * base + blend_weight * prompt_signal`.
    pub blend_weight: f64,
    /// Multiplicative inflation applied to the prediction interval half-width.
    pub uncertainty_inflate: f64,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            horizon: 24,
            blend_weight: 0.3,
            uncertainty_inflate: 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConditionedForecast
// ─────────────────────────────────────────────────────────────────────────────

/// A forecast conditioned on one or more [`TimeSeriesPrompt`]s.
#[derive(Debug, Clone)]
pub struct ConditionedForecast {
    /// Blended point forecast values.
    pub values: Vec<f64>,
    /// Lower bound of the prediction interval.
    pub lower_bound: Vec<f64>,
    /// Upper bound of the prediction interval.
    pub upper_bound: Vec<f64>,
    /// The aggregate prompt signal that was blended into the base forecast.
    pub prompt_contribution: Vec<f64>,
}

impl ConditionedForecast {
    /// Return the (`lower`, `upper`) prediction interval slices.
    ///
    /// The `alpha` parameter is accepted for API compatibility but the stored
    /// bounds correspond to the interval specified at conditioning time.
    pub fn prediction_interval(&self, _alpha: f64) -> (&[f64], &[f64]) {
        (&self.lower_bound, &self.upper_bound)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PromptConditioner
// ─────────────────────────────────────────────────────────────────────────────

/// Conditions a base time series forecast on declarative prompts.
pub struct PromptConditioner {
    config: PromptConfig,
}

impl PromptConditioner {
    /// Create a new conditioner with the given configuration.
    pub fn new(config: PromptConfig) -> Self {
        Self { config }
    }

    /// Generate the prior signal for a single prompt over `horizon` steps.
    ///
    /// Returns a `Vec<f64>` of length `horizon`.
    pub fn generate_prompt_signal(&self, prompt: &TimeSeriesPrompt, horizon: usize) -> Vec<f64> {
        match prompt {
            TimeSeriesPrompt::Trend {
                direction,
                magnitude,
            } => (0..horizon)
                .map(|t| {
                    let frac = if horizon <= 1 {
                        0.0
                    } else {
                        t as f64 / (horizon - 1) as f64
                    };
                    match direction {
                        TrendDirection::Up => magnitude * frac,
                        TrendDirection::Down => -magnitude * frac,
                        TrendDirection::Flat => 0.0,
                        TrendDirection::Oscillating => magnitude * (PI * frac * 2.0).sin(),
                    }
                })
                .collect(),
            TimeSeriesPrompt::Seasonal { period, amplitude } => {
                let period_f = (*period).max(1) as f64;
                (0..horizon)
                    .map(|t| amplitude * (2.0 * PI * t as f64 / period_f).sin())
                    .collect()
            }
            TimeSeriesPrompt::Anomaly { at_step, magnitude } => {
                let mut signal = vec![0.0f64; horizon];
                if *at_step < horizon {
                    signal[*at_step] = *magnitude;
                }
                signal
            }
            TimeSeriesPrompt::Level { shift, at_step } => (0..horizon)
                .map(|t| if t >= *at_step { *shift } else { 0.0 })
                .collect(),
            TimeSeriesPrompt::Noise { std } => {
                // Deterministic pseudo-noise using a simple LCG so tests are reproducible.
                let mut state: u64 = 12345;
                (0..horizon)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        // Map to [-1, 1]
                        let u = (state >> 33) as f64 / (u32::MAX as f64);
                        (u - 0.5) * 2.0 * std
                    })
                    .collect()
            }
            TimeSeriesPrompt::Custom(prior) => {
                // Truncate or repeat to fill horizon
                if prior.is_empty() {
                    vec![0.0; horizon]
                } else {
                    (0..horizon).map(|t| prior[t % prior.len()]).collect()
                }
            }
        }
    }

    /// Blend the base forecast with signals derived from the given prompts.
    ///
    /// If multiple prompts are provided, their signals are averaged before blending.
    pub fn condition_forecast(
        &self,
        base_forecast: &[f64],
        prompts: &[TimeSeriesPrompt],
    ) -> ConditionedForecast {
        let horizon = base_forecast.len();
        if prompts.is_empty() || horizon == 0 {
            let std_dev = Self::estimate_std(base_forecast) * self.config.uncertainty_inflate;
            let lower: Vec<f64> = base_forecast.iter().map(|v| v - 1.96 * std_dev).collect();
            let upper: Vec<f64> = base_forecast.iter().map(|v| v + 1.96 * std_dev).collect();
            return ConditionedForecast {
                values: base_forecast.to_vec(),
                lower_bound: lower,
                upper_bound: upper,
                prompt_contribution: vec![0.0; horizon],
            };
        }

        // Aggregate prompt signals (simple average)
        let mut agg_signal = vec![0.0f64; horizon];
        for prompt in prompts {
            let sig = self.generate_prompt_signal(prompt, horizon);
            for (a, s) in agg_signal.iter_mut().zip(sig.iter()) {
                *a += s;
            }
        }
        let n_prompts = prompts.len() as f64;
        for v in &mut agg_signal {
            *v /= n_prompts;
        }

        let w = self.config.blend_weight.clamp(0.0, 1.0);
        let values: Vec<f64> = base_forecast
            .iter()
            .zip(agg_signal.iter())
            .map(|(b, p)| (1.0 - w) * b + w * p)
            .collect();

        // Build prediction interval
        let std_dev = Self::estimate_std(&values) * self.config.uncertainty_inflate;
        let lower: Vec<f64> = values.iter().map(|v| v - 1.96 * std_dev).collect();
        let upper: Vec<f64> = values.iter().map(|v| v + 1.96 * std_dev).collect();

        ConditionedForecast {
            values,
            lower_bound: lower,
            upper_bound: upper,
            prompt_contribution: agg_signal,
        }
    }

    /// Naive keyword-based parser: converts a natural-language string into a prompt.
    ///
    /// Returns `None` if no recognisable pattern is found.
    pub fn from_text_description(text: &str) -> Option<TimeSeriesPrompt> {
        let lower = text.to_lowercase();

        if lower.contains("level shift") || lower.contains("level change") {
            return Some(TimeSeriesPrompt::Level {
                shift: 10.0,
                at_step: 0,
            });
        }
        if lower.contains("spike") || lower.contains("anomaly") || lower.contains("outlier") {
            return Some(TimeSeriesPrompt::Anomaly {
                at_step: 0,
                magnitude: 5.0,
            });
        }
        if lower.contains("seasonal") || lower.contains("cycle") || lower.contains("periodic") {
            return Some(TimeSeriesPrompt::Seasonal {
                period: 12,
                amplitude: 1.0,
            });
        }
        if lower.contains("upward")
            || lower.contains(" up ")
            || lower.starts_with("up ")
            || lower.contains("uptrend")
            || lower.contains("increasing")
        {
            return Some(TimeSeriesPrompt::Trend {
                direction: TrendDirection::Up,
                magnitude: 1.0,
            });
        }
        if lower.contains("downward")
            || lower.contains(" down ")
            || lower.contains("downtrend")
            || lower.contains("decreasing")
        {
            return Some(TimeSeriesPrompt::Trend {
                direction: TrendDirection::Down,
                magnitude: 1.0,
            });
        }
        if lower.contains("noise") || lower.contains("random") {
            return Some(TimeSeriesPrompt::Noise { std: 0.5 });
        }
        None
    }

    /// Estimate the standard deviation of a slice (population std).
    fn estimate_std(values: &[f64]) -> f64 {
        let n = values.len();
        if n < 2 {
            return 0.0;
        }
        let mean: f64 = values.iter().sum::<f64>() / n as f64;
        let var: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        var.sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Suppress unused warning for TimeSeriesError import used in docstring
// ─────────────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
fn _use_error(_e: TimeSeriesError) {}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trend_up_monotonic() {
        let config = PromptConfig {
            horizon: 10,
            ..Default::default()
        };
        let conditioner = PromptConditioner::new(config);
        let prompt = TimeSeriesPrompt::Trend {
            direction: TrendDirection::Up,
            magnitude: 5.0,
        };
        let signal = conditioner.generate_prompt_signal(&prompt, 10);
        assert_eq!(signal.len(), 10);
        // Should be monotonically non-decreasing
        for i in 1..signal.len() {
            assert!(
                signal[i] >= signal[i - 1],
                "Up trend signal should be non-decreasing at step {i}: {} < {}",
                signal[i],
                signal[i - 1]
            );
        }
        assert!(
            signal.last().copied().unwrap_or(0.0) > 0.0,
            "final value should be positive"
        );
    }

    #[test]
    fn test_seasonal_signal_periodic() {
        let conditioner = PromptConditioner::new(PromptConfig::default());
        let prompt = TimeSeriesPrompt::Seasonal {
            period: 4,
            amplitude: 2.0,
        };
        let signal = conditioner.generate_prompt_signal(&prompt, 8);
        assert_eq!(signal.len(), 8);
        // Period-4: signal[0] should equal signal[4] (both at sin(0) = 0)
        assert!(
            (signal[0] - signal[4]).abs() < 1e-10,
            "seasonal signal should be periodic: {} != {}",
            signal[0],
            signal[4]
        );
        assert!(
            (signal[1] - signal[5]).abs() < 1e-10,
            "seasonal signal should be periodic at offset 1"
        );
    }

    #[test]
    fn test_from_text_description_upward() {
        let result = PromptConditioner::from_text_description("upward trend in the data");
        assert!(result.is_some(), "should parse 'upward trend'");
        if let Some(TimeSeriesPrompt::Trend { direction, .. }) = result {
            assert_eq!(direction, TrendDirection::Up);
        } else {
            panic!("expected Trend(Up) prompt");
        }
    }

    #[test]
    fn test_from_text_description_seasonal() {
        let result = PromptConditioner::from_text_description("seasonal pattern detected");
        assert!(result.is_some());
        assert!(matches!(result, Some(TimeSeriesPrompt::Seasonal { .. })));
    }

    #[test]
    fn test_conditioned_forecast_blend() {
        let config = PromptConfig {
            horizon: 5,
            blend_weight: 0.5,
            uncertainty_inflate: 1.0,
        };
        let conditioner = PromptConditioner::new(config);
        let base = vec![0.0f64; 5];
        let prompts = vec![TimeSeriesPrompt::Trend {
            direction: TrendDirection::Up,
            magnitude: 4.0,
        }];
        let result = conditioner.condition_forecast(&base, &prompts);
        assert_eq!(result.values.len(), 5);
        // With blend=0.5 and base=0 and prompt=trend(up,4), values should be positive overall
        let total: f64 = result.values.iter().sum();
        assert!(
            total > 0.0,
            "blended forecast should be positive (total={total})"
        );
        // Prompt contribution should be non-negative (up trend)
        for v in &result.prompt_contribution {
            assert!(*v >= -1e-12, "contribution should be non-negative");
        }
    }

    #[test]
    fn test_anomaly_signal_spike() {
        let conditioner = PromptConditioner::new(PromptConfig::default());
        let prompt = TimeSeriesPrompt::Anomaly {
            at_step: 3,
            magnitude: 10.0,
        };
        let signal = conditioner.generate_prompt_signal(&prompt, 8);
        assert_eq!(signal.len(), 8);
        // Only step 3 should be non-zero
        for (i, v) in signal.iter().enumerate() {
            if i == 3 {
                assert!((v - 10.0).abs() < 1e-10, "spike should be at step 3");
            } else {
                assert!(v.abs() < 1e-10, "non-spike steps should be zero (step {i})");
            }
        }
    }

    #[test]
    fn test_level_shift_signal() {
        let conditioner = PromptConditioner::new(PromptConfig::default());
        let prompt = TimeSeriesPrompt::Level {
            shift: 5.0,
            at_step: 2,
        };
        let signal = conditioner.generate_prompt_signal(&prompt, 6);
        assert_eq!(signal[0], 0.0);
        assert_eq!(signal[1], 0.0);
        assert_eq!(signal[2], 5.0);
        assert_eq!(signal[5], 5.0);
    }
}
