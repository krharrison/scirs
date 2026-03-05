//! Types for tensor core operations.
//!
//! This module is split into logical sub-modules:
//! - `enums`: All enumerations, simple data structures, and `StabilityMetrics`
//! - `stability`: `PerformanceAccuracyAnalyzer` and `NumericalStabilityMonitor`
//! - `distance`: Distance matrix computation, `TensorCoreClustering`, and `ErrorRecoverySystem`

pub mod distance;
pub mod enums;
pub mod stability;

// Re-export all public types so that existing code using `types::*` continues to work.
pub use distance::{
    AdvancedTensorCoreDistanceMatrix, ErrorRecoverySystem, TensorCoreClustering,
    TensorCoreDistanceMatrix,
};
pub use enums::{
    DynamicPrecisionConfig, GpuArchitecture, NumericalErrorType, OptimizationObjective,
    PrecisionMode, RecoveryAction, RecoveryAttempt, ScalingStrategy, StabilityLevel,
    StabilityMetrics, TensorCoreCapabilities, TensorCoreType, TensorLayout, TradeOffParams,
};
pub use stability::{NumericalStabilityMonitor, PerformanceAccuracyAnalyzer};
