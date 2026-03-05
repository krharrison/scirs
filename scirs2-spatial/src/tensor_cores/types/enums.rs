//! Enumerations and simple data types for tensor core operations.

use std::time::{Duration, Instant};

use crate::error::SpatialError;
use scirs2_core::ndarray::Array2;

/// GPU architecture types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuArchitecture {
    /// NVIDIA Volta (V100)
    Volta,
    /// NVIDIA Ampere (A100, RTX 30 series)
    Ampere,
    /// NVIDIA Hopper (H100)
    Hopper,
    /// AMD CDNA2 (MI250X)
    CDNA2,
    /// AMD CDNA3 (MI300)
    CDNA3,
    /// Intel Xe HPC (Ponte Vecchio)
    XeHPC,
    /// Intel Xe Graphics (Arc)
    XeGraphics,
    /// Unknown or fallback
    Unknown,
}

/// Numerical error types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumericalErrorType {
    /// Overflow in computation
    Overflow,
    /// Underflow in computation
    Underflow,
    /// Loss of precision
    PrecisionLoss,
    /// Convergence failure
    ConvergenceFailure,
    /// Ill-conditioned matrix
    IllConditioned,
    /// NaN or Inf values
    InvalidValues,
}

/// Precision modes for tensor core operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecisionMode {
    /// Full precision (FP32)
    Full32,
    /// Mixed precision (FP16 compute, FP32 accumulate)
    Mixed16,
    /// Brain floating point (BF16)
    BrainFloat16,
    /// 8-bit integer with dynamic scaling
    Int8Dynamic,
    /// 4-bit integer with advanced quantization
    Int4Advanced,
    /// Automatic precision selection
    Adaptive,
    /// Advanced-adaptive with stability monitoring
    AdvancedAdaptive,
}

/// Numerical stability level
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StabilityLevel {
    /// Excellent numerical stability
    Excellent,
    /// Good numerical stability
    Good,
    /// Moderate numerical stability
    Moderate,
    /// Poor numerical stability - increase precision
    Poor,
    /// Critical numerical instability - recovery needed
    Critical,
}

/// Tensor layout optimization strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorLayout {
    /// Row-major layout (C-style)
    RowMajor,
    /// Column-major layout (Fortran-style)
    ColMajor,
    /// Blocked layout for cache efficiency
    Blocked,
    /// Hierarchical Z-order layout
    ZOrder,
    /// Hardware-optimized layout
    HardwareOptimized,
}

/// Optimization objectives
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationObjective {
    /// Maximize performance (minimize time)
    MaxPerformance,
    /// Maximize accuracy
    MaxAccuracy,
    /// Balance performance and accuracy
    Balanced,
    /// Minimize energy consumption
    MinEnergy,
    /// Custom weighted objective
    Custom,
}

/// Dynamic precision scaling strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalingStrategy {
    /// Conservative - always use higher precision when uncertain
    Conservative,
    /// Balanced - balance performance and accuracy
    Balanced,
    /// Aggressive - favor performance over precision
    Aggressive,
    /// Custom - user-defined thresholds
    Custom,
}

/// Recovery action types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecoveryAction {
    /// Increase precision mode
    IncreasePrecision,
    /// Reduce tile size
    ReduceTileSize,
    /// Switch to fallback algorithm
    FallbackAlgorithm,
    /// Apply numerical stabilization
    NumericalStabilization,
    /// Retry with different parameters
    RetryWithNewParams,
    /// Switch to CPU computation
    SwitchToCPU,
}

/// Tensor core types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorCoreType {
    /// NVIDIA Tensor Cores (WMMA)
    NvidiaTensorCore,
    /// AMD Matrix Cores
    AmdMatrixCore,
    /// Intel XMX units
    IntelXMX,
    /// Standard CUDA/OpenCL cores (fallback)
    StandardCores,
}

/// Tensor core capabilities
#[derive(Debug, Clone)]
pub struct TensorCoreCapabilities {
    /// Available tensor core types
    pub tensor_core_types: Vec<TensorCoreType>,
    /// Supported precision modes
    pub supported_precisions: Vec<PrecisionMode>,
    /// Maximum tensor dimensions
    pub max_tensor_size: (usize, usize, usize),
    /// Peak throughput (TOPS)
    pub peak_throughput_tops: f64,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth_gbps: f64,
    /// L2 cache size (MB)
    pub l2_cache_mb: f64,
    /// Number of streaming multiprocessors
    pub num_sms: usize,
    /// Architecture
    pub architecture: GpuArchitecture,
}

/// Trade-off optimization parameters
#[derive(Debug, Clone)]
pub struct TradeOffParams {
    /// Weight for performance (speed)
    pub performance_weight: f64,
    /// Weight for accuracy
    pub accuracy_weight: f64,
    /// Weight for energy efficiency
    pub energy_weight: f64,
    /// Minimum acceptable accuracy
    pub min_accuracy: f64,
    /// Maximum acceptable time
    pub max_time: Duration,
    /// Optimization objective
    pub objective: OptimizationObjective,
}

/// Dynamic precision scaling configuration
#[derive(Debug, Clone)]
pub struct DynamicPrecisionConfig {
    /// Scaling strategy
    pub strategy: ScalingStrategy,
    /// Minimum precision level
    pub min_precision: PrecisionMode,
    /// Maximum precision level
    pub max_precision: PrecisionMode,
    /// Stability threshold for precision increase
    pub stability_threshold_up: f64,
    /// Stability threshold for precision decrease
    pub stability_threshold_down: f64,
    /// Performance weight in decision making
    pub performance_weight: f64,
    /// Accuracy weight in decision making
    pub accuracy_weight: f64,
    /// Maximum precision changes per operation
    pub max_changes_per_operation: usize,
    /// Cooldown period between precision changes
    pub change_cooldown: Duration,
}

/// Recovery attempt record
#[derive(Debug, Clone)]
pub struct RecoveryAttempt {
    /// Error type that triggered recovery
    pub error_type: NumericalErrorType,
    /// Recovery action taken
    pub action: RecoveryAction,
    /// Success/failure of recovery
    pub success: bool,
    /// Time taken for recovery
    pub duration: Duration,
    /// Stability metrics after recovery
    pub post_recovery_metrics: Option<StabilityMetrics>,
    /// Timestamp
    pub timestamp: Instant,
}

/// Numerical stability metrics
#[derive(Debug, Clone)]
pub struct StabilityMetrics {
    /// Condition number of the computation
    pub condition_number: f64,
    /// Relative error estimate
    pub relative_error: f64,
    /// Forward error bound
    pub forward_error: f64,
    /// Backward error bound
    pub backward_error: f64,
    /// Loss of significant digits
    pub digit_loss: f64,
    /// Current stability level
    pub stability_level: StabilityLevel,
    /// Detected error types
    pub error_types: Vec<NumericalErrorType>,
    /// Timestamp of measurement
    pub timestamp: Instant,
}

impl StabilityMetrics {
    /// Create new stability metrics
    pub fn new() -> Self {
        Self {
            condition_number: 1.0,
            relative_error: 0.0,
            forward_error: 0.0,
            backward_error: 0.0,
            digit_loss: 0.0,
            stability_level: StabilityLevel::Excellent,
            error_types: Vec::new(),
            timestamp: Instant::now(),
        }
    }

    /// Update stability level based on metrics
    pub fn update_stability_level(&mut self) {
        self.stability_level = if self.condition_number > 1e12 || self.relative_error > 1e-3 {
            StabilityLevel::Critical
        } else if self.condition_number > 1e8 || self.relative_error > 1e-6 {
            StabilityLevel::Poor
        } else if self.condition_number > 1e4 || self.relative_error > 1e-9 {
            StabilityLevel::Moderate
        } else if self.condition_number > 1e2 || self.relative_error > 1e-12 {
            StabilityLevel::Good
        } else {
            StabilityLevel::Excellent
        };
    }

    /// Check for numerical errors
    pub fn detect_errors(&mut self, data: &Array2<f64>) {
        self.error_types.clear();
        for &value in data.iter() {
            if !value.is_finite() {
                self.error_types.push(NumericalErrorType::InvalidValues);
                break;
            }
        }
        let max_val = data.fold(0.0f64, |acc, &x| acc.max(x.abs()));
        if max_val > 1e100 {
            self.error_types.push(NumericalErrorType::Overflow);
        } else if max_val < 1e-100 && max_val > 0.0 {
            self.error_types.push(NumericalErrorType::Underflow);
        }
        if self.digit_loss > 6.0 {
            self.error_types.push(NumericalErrorType::PrecisionLoss);
        }
        if self.condition_number > 1e12 {
            self.error_types.push(NumericalErrorType::IllConditioned);
        }
    }
}
