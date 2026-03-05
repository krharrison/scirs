//! Online and incremental learning methods for streaming data
//!
//! This module provides algorithms that process data one batch (or sample) at a
//! time, making them suitable for large-scale or continuously-arriving datasets
//! that do not fit in memory.
//!
//! # Modules
//!
//! - [`incremental_pca`]: Incremental PCA using the Arora et al. algorithm
//! - [`online_scaler`]: Online normalization (min-max, z-score, robust)
//! - [`sketching`]: Sketching algorithms (Count-Min, Bloom filter, HyperLogLog, …)
//! - [`online_regression`]: Online regression (LASSO, ridge, PA, FTRL)
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_transform::online::incremental_pca::IncrementalPCA;
//! use scirs2_core::ndarray::Array2;
//!
//! let mut ipca = IncrementalPCA::new(2, Some(10));
//! let batch: Array2<f64> = Array2::zeros((10, 5));
//! ipca.partial_fit(&batch).expect("should succeed");
//! let projected = ipca.transform(&batch).expect("should succeed");
//! ```

pub mod incremental_pca;
pub mod online_scaler;
pub mod online_regression;
pub mod sketching;

pub use incremental_pca::IncrementalPCA;
pub use online_scaler::{OnlineMinMaxScaler, OnlineRobustScaler, OnlineStandardScaler};
pub use online_regression::{
    PassiveAggressiveRegressor, FtrlRegressor, OnlineLasso, OnlineRidgeRegression,
};
pub use sketching::{
    BloomFilter, CountMinSketch, CountSketch, HyperLogLog, ReservoirSampler,
};
