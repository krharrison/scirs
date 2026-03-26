//! Types for probabilistic energy forecasting

/// Temporal aggregation level for energy data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AggregationLevel {
    /// Hourly resolution
    Hourly,
    /// Daily resolution (24 hours)
    Daily,
    /// Weekly resolution (168 hours)
    Weekly,
    /// Monthly resolution (~720 hours)
    Monthly,
}

/// Configuration for energy forecast models
#[derive(Debug, Clone)]
pub struct EnergyForecastConfig {
    /// Forecast horizon (number of steps)
    pub horizon: usize,
    /// Quantile levels for probabilistic forecasts
    pub quantile_levels: Vec<f64>,
    /// Temporal aggregation level
    pub aggregation_level: AggregationLevel,
    /// Number of boosting estimators
    pub n_estimators: usize,
    /// Learning rate for gradient boosting
    pub learning_rate: f64,
}

impl Default for EnergyForecastConfig {
    fn default() -> Self {
        Self {
            horizon: 24,
            quantile_levels: vec![0.1, 0.25, 0.5, 0.75, 0.9],
            aggregation_level: AggregationLevel::Hourly,
            n_estimators: 100,
            learning_rate: 0.1,
        }
    }
}

/// Load profile input data
#[derive(Debug, Clone)]
pub struct LoadProfile {
    /// Timestamps (hours from epoch or similar monotonic sequence)
    pub timestamps: Vec<f64>,
    /// Observed load values (e.g., MW)
    pub load_values: Vec<f64>,
    /// Optional temperature observations
    pub temperatures: Option<Vec<f64>>,
    /// Optional holiday mask (true = holiday)
    pub holiday_mask: Option<Vec<bool>>,
}

/// A single quantile forecast
#[derive(Debug, Clone)]
pub struct QuantileForecast {
    /// Quantile level (e.g. 0.1, 0.5, 0.9)
    pub quantile: f64,
    /// Forecast values at this quantile
    pub values: Vec<f64>,
}

/// Evaluation metrics for probabilistic forecasts
#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    /// Mean pinball loss across all quantiles
    pub pinball: f64,
    /// Continuous Ranked Probability Score
    pub crps: f64,
    /// Empirical coverage of the widest interval
    pub coverage: f64,
    /// Mean interval width (sharpness)
    pub sharpness: f64,
}

/// Energy forecast result containing point and quantile forecasts
#[derive(Debug, Clone)]
pub struct EnergyForecastResult {
    /// Point forecasts (median or mean)
    pub point_forecasts: Vec<f64>,
    /// Quantile forecasts at each requested level
    pub quantile_forecasts: Vec<QuantileForecast>,
    /// Optional evaluation metrics (set when actuals are available)
    pub metrics: Option<EvaluationMetrics>,
}

/// Direction of a ramp event
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RampDirection {
    /// Increasing load
    Up,
    /// Decreasing load
    Down,
}

/// A detected ramp event in the load forecast
#[derive(Debug, Clone)]
pub struct RampEvent {
    /// Start index of the ramp
    pub start_idx: usize,
    /// End index of the ramp
    pub end_idx: usize,
    /// Magnitude of the ramp (absolute change)
    pub magnitude: f64,
    /// Direction of the ramp
    pub direction: RampDirection,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = EnergyForecastConfig::default();
        assert_eq!(cfg.horizon, 24);
        assert_eq!(cfg.quantile_levels.len(), 5);
        assert_eq!(cfg.n_estimators, 100);
        assert!((cfg.learning_rate - 0.1).abs() < 1e-12);
        assert_eq!(cfg.aggregation_level, AggregationLevel::Hourly);
    }

    #[test]
    fn test_aggregation_level_variants() {
        let levels = [
            AggregationLevel::Hourly,
            AggregationLevel::Daily,
            AggregationLevel::Weekly,
            AggregationLevel::Monthly,
        ];
        assert_eq!(levels.len(), 4);
    }

    #[test]
    fn test_ramp_direction() {
        let up = RampDirection::Up;
        let down = RampDirection::Down;
        assert_ne!(up, down);
    }
}
