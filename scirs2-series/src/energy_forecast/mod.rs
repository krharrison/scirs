//! Probabilistic Energy Forecasting
//!
//! Provides quantile regression, gradient-boosted stumps, and evaluation
//! metrics for probabilistic load forecasting with calendar, temperature,
//! lag, and Fourier feature engineering.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | `models` | [`QuantileRegressor`], [`GradientBoostedStumps`], [`EnergyForecaster`] |
//! | `price_model` | [`MeanReversionModel`] (OU), [`SpikePriceModel`] |
//! | `load_forecast` | [`LoadForecaster`], [`seasonal_decompose`], [`levinson_durbin`] |
//! | `market_types` | [`ProbabilisticForecast`], [`MarketClearingPrice`], [`EnergyMarketData`] |

mod aggregation;
mod evaluation;
mod features;
pub mod load_forecast;
pub mod market_types;
mod models;
pub mod price_model;
mod types;

pub use aggregation::{aggregate_hourly_to_daily, detect_peak_demand, detect_ramp_events};
pub use evaluation::{
    coverage, crps, pinball_loss, reliability_diagram, sharpness, skill_score, winkler_score,
};
pub use features::{
    CalendarFeatures, FeatureConfig, FeatureMatrix, FourierFeatures, LagFeatures,
    TemperatureFeatures,
};
pub use load_forecast::{
    auto_regression_residual, fit_ar, levinson_durbin, seasonal_decompose, LoadForecaster,
};
pub use market_types::{EnergyMarketData, MarketClearingPrice, ProbabilisticForecast};
pub use models::{EnergyForecaster, GradientBoostedStumps, QuantileRegressor};
pub use price_model::{forecast_prices, MeanReversionModel, SpikePriceModel};
pub use types::{
    AggregationLevel, EnergyForecastConfig, EnergyForecastResult, EvaluationMetrics, LoadProfile,
    QuantileForecast, RampDirection, RampEvent,
};
