//! Conformal Prediction with Adaptive Coverage Guarantees
//!
//! This module provides a comprehensive implementation of conformal prediction
//! methods for both regression and classification tasks.  All methods provide
//! rigorous finite-sample marginal coverage guarantees of at least `1 − α`
//! under the exchangeability assumption.
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`types`] | Core types: `ConformalConfig`, `PredictionSet`, `ConformalResult`, etc. |
//! | [`split_conformal`] | Split (inductive) conformal prediction for regression and classification |
//! | [`adaptive_conformal`] | Normalized, CQR, RAPS, and Mondrian conformal prediction |
//! | [`online_conformal`] | ACI (Adaptive Conformal Inference) for online / streaming data |
//!
//! ## Quick Example
//!
//! ```rust
//! use scirs2_stats::conformal::split_conformal::SplitConformal;
//!
//! // Calibrate on a held-out calibration set
//! let cal_preds  = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let cal_actual = vec![1.1, 1.9, 3.2, 3.8, 5.1];
//! let mut sc = SplitConformal::new();
//! sc.calibrate(&cal_preds, &cal_actual);
//!
//! // Predict a 90% interval for a new point
//! let interval = sc.predict_interval(3.0, 0.1).expect("calibrated");
//! assert!(interval.lower < 3.0 && interval.upper > 3.0);
//! ```

pub mod adaptive_conformal;
pub mod online_conformal;
pub mod split_conformal;
pub mod types;

// Convenience re-exports
pub use adaptive_conformal::{CqrConformal, MondrianConformal, NormalizedConformal, RapsConformal};
pub use online_conformal::{coverage_drift, running_coverage, AciConformal};
pub use split_conformal::{
    empirical_coverage_classification, empirical_coverage_regression, SplitConformal,
    SplitConformalClassifier,
};
pub use types::{
    conformal_quantile, ConformalConfig, ConformalResult, CpConfig, PredictionSet, RapsConfig,
    ScoreType,
};
