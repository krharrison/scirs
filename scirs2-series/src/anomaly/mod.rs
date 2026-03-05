//! Anomaly detection for time series
//!
//! This module provides both legacy anomaly detection and advanced statistical methods:
//! - `legacy`: Original anomaly detection methods (SPC, IsolationForest, ZScore, etc.)
//! - `statistical`: Advanced statistical methods (STL, GESD, Shewhart, EWMA, CUSUM, IsolationForestTS)

pub mod legacy;
pub mod statistical;

// Re-export legacy types for backward compatibility
pub use legacy::{
    detect_anomalies, AnomalyMethod, AnomalyOptions, AnomalyResult, DistanceMetric, MethodInfo,
    SPCMethod,
};

// Re-export new statistical anomaly detection types
pub use statistical::{
    AnomalyDetectionResult, CUSUMChart, CUSUMConfig, CUSUMState, EWMAChart, EWMAConfig, GESDResult,
    IsolationForestTS, IsolationForestTSConfig, SHEWHARTChart, STLAnomalyConfig,
    STLAnomalyDetector, ShewhartConfig, GESD,
};
