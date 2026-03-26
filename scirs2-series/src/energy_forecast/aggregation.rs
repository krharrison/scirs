//! Temporal aggregation and event detection for energy forecasts
//!
//! Provides hourly-to-daily aggregation with variance propagation,
//! peak demand detection, and ramp event detection.

use crate::error::{Result, TimeSeriesError};

use super::types::{EnergyForecastResult, QuantileForecast, RampDirection, RampEvent};

/// Aggregate hourly forecast results to daily resolution.
///
/// Sums point forecasts over 24-hour blocks. For quantile forecasts,
/// sums the quantile values (appropriate for approximately independent
/// hourly errors; exact for Gaussian).
pub fn aggregate_hourly_to_daily(hourly: &EnergyForecastResult) -> Result<EnergyForecastResult> {
    let n = hourly.point_forecasts.len();
    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "No hourly forecasts to aggregate".to_string(),
        ));
    }

    let n_days = n / 24;
    if n_days == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Need at least 24 hourly values for daily aggregation".to_string(),
        ));
    }

    // Aggregate point forecasts
    let mut daily_points = Vec::with_capacity(n_days);
    for d in 0..n_days {
        let start = d * 24;
        let end = start + 24;
        let sum: f64 = hourly.point_forecasts[start..end].iter().sum();
        daily_points.push(sum);
    }

    // Aggregate quantile forecasts
    let daily_quantiles: Vec<QuantileForecast> = hourly
        .quantile_forecasts
        .iter()
        .map(|qf| {
            let mut daily_vals = Vec::with_capacity(n_days);
            for d in 0..n_days {
                let start = d * 24;
                let end = (start + 24).min(qf.values.len());
                if end > start {
                    let sum: f64 = qf.values[start..end].iter().sum();
                    daily_vals.push(sum);
                }
            }
            QuantileForecast {
                quantile: qf.quantile,
                values: daily_vals,
            }
        })
        .collect();

    Ok(EnergyForecastResult {
        point_forecasts: daily_points,
        quantile_forecasts: daily_quantiles,
        metrics: None,
    })
}

/// Detect indices where the forecast exceeds a threshold quantile value.
///
/// Returns indices into `forecasts` where the value exceeds the given
/// quantile of the entire forecast distribution.
pub fn detect_peak_demand(forecasts: &[f64], threshold_quantile: f64) -> Vec<usize> {
    if forecasts.is_empty() {
        return Vec::new();
    }

    // Compute the threshold value from the empirical distribution
    let mut sorted = forecasts.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx =
        ((threshold_quantile * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
    let threshold = sorted[idx];

    forecasts
        .iter()
        .enumerate()
        .filter(|(_, &v)| v >= threshold)
        .map(|(i, _)| i)
        .collect()
}

/// Detect ramp events in the forecast series.
///
/// A ramp event is a sequence of consecutive time steps where the
/// cumulative change exceeds `ramp_threshold`.
pub fn detect_ramp_events(forecasts: &[f64], ramp_threshold: f64) -> Vec<RampEvent> {
    if forecasts.len() < 2 {
        return Vec::new();
    }

    let mut events = Vec::new();
    let mut i = 0;

    while i < forecasts.len() - 1 {
        let start = i;
        let mut cumulative: f64 = 0.0;
        let initial_dir = if forecasts[i + 1] >= forecasts[i] {
            RampDirection::Up
        } else {
            RampDirection::Down
        };

        // Extend ramp while direction is consistent
        while i < forecasts.len() - 1 {
            let diff = forecasts[i + 1] - forecasts[i];
            let current_dir = if diff >= 0.0 {
                RampDirection::Up
            } else {
                RampDirection::Down
            };

            if current_dir != initial_dir && cumulative.abs() > 0.0 {
                break;
            }
            cumulative += diff;
            i += 1;
        }

        if cumulative.abs() >= ramp_threshold {
            events.push(RampEvent {
                start_idx: start,
                end_idx: i,
                magnitude: cumulative.abs(),
                direction: if cumulative >= 0.0 {
                    RampDirection::Up
                } else {
                    RampDirection::Down
                },
            });
        }

        if i == start {
            i += 1;
        }
    }

    events
}

/// Propagate variance when aggregating n independent intervals.
///
/// If hourly variances are σ²_h, the daily variance for the sum of 24
/// independent hours is Σ σ²_h. The aggregated interval half-width scales
/// as sqrt(Σ w²_h) where w_h is the hourly half-width.
pub fn propagate_variance_sum(half_widths: &[f64]) -> f64 {
    let sum_sq: f64 = half_widths.iter().map(|w| w * w).sum();
    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hourly_to_daily_aggregation() {
        // 48 hourly values (2 days), each = 10.0
        let hourly = EnergyForecastResult {
            point_forecasts: vec![10.0; 48],
            quantile_forecasts: vec![QuantileForecast {
                quantile: 0.5,
                values: vec![10.0; 48],
            }],
            metrics: None,
        };
        let daily = aggregate_hourly_to_daily(&hourly).expect("should aggregate");
        assert_eq!(daily.point_forecasts.len(), 2);
        assert!((daily.point_forecasts[0] - 240.0).abs() < 1e-10);
        assert!((daily.point_forecasts[1] - 240.0).abs() < 1e-10);
    }

    #[test]
    fn test_detect_peak_demand() {
        let forecasts = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let peaks = detect_peak_demand(&forecasts, 0.8);
        // 0.8 quantile of [10,20,30,40,50] => idx=3.2 => 40
        // values >= 40: indices 3, 4
        assert!(peaks.contains(&3));
        assert!(peaks.contains(&4));
        assert!(!peaks.contains(&0));
    }

    #[test]
    fn test_detect_ramp_events_step() {
        // Step function: flat 10, then jump to 50, then flat
        let forecasts = vec![10.0, 10.0, 10.0, 50.0, 50.0, 50.0];
        let events = detect_ramp_events(&forecasts, 30.0);
        assert!(!events.is_empty(), "should detect the step as a ramp");
        let up_events: Vec<_> = events
            .iter()
            .filter(|e| e.direction == RampDirection::Up)
            .collect();
        assert!(!up_events.is_empty());
    }

    #[test]
    fn test_detect_ramp_events_no_ramp() {
        let forecasts = vec![10.0, 10.1, 10.0, 9.9, 10.0];
        let events = detect_ramp_events(&forecasts, 5.0);
        assert!(
            events.is_empty(),
            "small fluctuations should not trigger ramp"
        );
    }

    #[test]
    fn test_variance_propagation() {
        let widths = vec![1.0, 1.0, 1.0, 1.0];
        let agg = propagate_variance_sum(&widths);
        // sqrt(4 * 1) = 2
        assert!((agg - 2.0).abs() < 1e-10);
    }
}
