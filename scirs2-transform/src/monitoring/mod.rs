//! Production monitoring with drift detection and concept drift algorithms
//!
//! This module provides:
//!
//! - **Core monitoring** (requires `monitoring` feature): `TransformationMonitor` with
//!   KS, PSI, MMD, Wasserstein drift detection, performance metrics, and alerting.
//! - **Drift detection**: `DriftDetector` trait with KS, PSI, Wasserstein, and MMD
//!   implementations for comparing reference vs. test distributions.
//! - **ADWIN**: ADaptive WINdowing algorithm for online concept drift detection in
//!   streaming data.

pub mod adwin;
#[cfg(feature = "monitoring")]
mod core;
pub mod drift_detection;

// Re-export everything from the original monitoring module (feature-gated)
#[cfg(feature = "monitoring")]
pub use self::core::*;

// Re-export drift detection types (always available)
pub use drift_detection::{
    DriftDetector, DriftResult, KolmogorovSmirnovDetector, MaximumMeanDiscrepancyDetector,
    PopulationStabilityIndexDetector, WassersteinDetector,
};

// Re-export ADWIN (always available)
pub use adwin::Adwin;
