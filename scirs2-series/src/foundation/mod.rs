//! Foundation model fine-tuning and zero-shot forecasting interfaces.
//!
//! This module provides adapters for working with pre-trained time series
//! foundation models (e.g. TimeGPT, PatchTST, TimesNet).  The interfaces are
//! model-agnostic: any type implementing [`fine_tuning::ForecastModel`] can be
//! wrapped for fine-tuning or zero-shot prediction.

pub mod fine_tuning;
pub mod zero_shot;

pub use fine_tuning::{
    FineTuner, FineTuningConfig, FineTuningResult, ForecastModel, FoundationModelType,
    LinearForecastModel, LoraForecastModel,
};
pub use zero_shot::{ZeroShotConfig, ZeroShotForecaster};
