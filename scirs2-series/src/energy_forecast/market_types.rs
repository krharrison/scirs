//! Additional types for probabilistic electricity market forecasting.

/// Probabilistic electricity price / load forecast.
///
/// Contains quantile forecasts, Monte-Carlo scenarios, and the mean trajectory.
#[derive(Debug, Clone)]
pub struct ProbabilisticForecast {
    /// Quantile forecasts: outer index = quantile level, inner = horizon step.
    pub quantile_forecasts: Vec<Vec<f64>>,
    /// Simulated price / load scenarios: outer index = scenario, inner = horizon step.
    pub scenarios: Vec<Vec<f64>>,
    /// Expected value (mean across scenarios) at each horizon step.
    pub mean: Vec<f64>,
}

impl ProbabilisticForecast {
    /// Return the number of quantile levels.
    pub fn n_quantiles(&self) -> usize {
        self.quantile_forecasts.len()
    }

    /// Return the forecast horizon (number of steps).
    pub fn horizon(&self) -> usize {
        self.mean.len()
    }

    /// Return the number of simulated scenarios.
    pub fn n_scenarios(&self) -> usize {
        self.scenarios.len()
    }
}

/// Market-clearing price estimate with confidence interval.
#[derive(Debug, Clone)]
pub struct MarketClearingPrice {
    /// Expected market-clearing price.
    pub price: f64,
    /// Symmetric confidence interval (lower, upper).
    pub confidence_interval: (f64, f64),
}

impl MarketClearingPrice {
    /// Return the half-width of the confidence interval.
    pub fn uncertainty(&self) -> f64 {
        (self.confidence_interval.1 - self.confidence_interval.0) / 2.0
    }
}

/// Raw energy market data suitable for model input.
#[derive(Debug, Clone)]
pub struct EnergyMarketData {
    /// Electricity spot prices (e.g., $/MWh).
    pub prices: Vec<f64>,
    /// System load observations (e.g., MW).
    pub loads: Vec<f64>,
    /// Total generation dispatch (e.g., MW).
    pub generation: Vec<f64>,
    /// Timestamps (e.g., hours since epoch).
    pub timestamps: Vec<f64>,
}

impl EnergyMarketData {
    /// Construct energy market data, validating that all series have the same length.
    pub fn new(
        prices: Vec<f64>,
        loads: Vec<f64>,
        generation: Vec<f64>,
        timestamps: Vec<f64>,
    ) -> Result<Self, crate::error::TimeSeriesError> {
        let n = prices.len();
        if loads.len() != n || generation.len() != n || timestamps.len() != n {
            return Err(crate::error::TimeSeriesError::InvalidInput(
                "all EnergyMarketData series must have the same length".to_string(),
            ));
        }
        Ok(Self {
            prices,
            loads,
            generation,
            timestamps,
        })
    }

    /// Return the number of observations.
    pub fn len(&self) -> usize {
        self.prices.len()
    }

    /// Return true if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_market_data_construction() {
        let data = EnergyMarketData::new(
            vec![50.0, 55.0],
            vec![1000.0, 1100.0],
            vec![900.0, 950.0],
            vec![0.0, 1.0],
        )
        .expect("valid data");
        assert_eq!(data.len(), 2);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_probabilistic_forecast_shape() {
        let f = ProbabilisticForecast {
            quantile_forecasts: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            scenarios: vec![vec![1.5, 2.5]; 10],
            mean: vec![2.0, 3.0],
        };
        assert_eq!(f.n_quantiles(), 2);
        assert_eq!(f.horizon(), 2);
        assert_eq!(f.n_scenarios(), 10);
    }

    #[test]
    fn test_market_clearing_price_uncertainty() {
        let mcp = MarketClearingPrice {
            price: 50.0,
            confidence_interval: (45.0, 55.0),
        };
        assert!((mcp.uncertainty() - 5.0).abs() < 1e-10);
    }
}
