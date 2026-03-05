//! Training utilities and infrastructure
//!
//! This module provides comprehensive utilities for training neural networks,
//! including advanced features like gradient accumulation, mixed precision training,
//! distributed training, and sophisticated training loop management.
//!
//! ## Key Features
//!
//! - **Enhanced Training Loop**: Comprehensive training with validation, early stopping,
//!   and learning rate scheduling
//! - **Profiled Training**: Automatic timing analysis and optimization recommendations
//! - **Optimized Data Loading**: Prefetching, parallel loading, and batch caching
//! - **Gradient Accumulation**: Memory-efficient training with large effective batch sizes
//! - **Mixed Precision**: FP16/BF16 training for faster computation (with GPU feature)
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_neural::training::{EnhancedTrainingConfig, EarlyStoppingConfig};
//!
//! let config = EnhancedTrainingConfig {
//!     epochs: 10,
//!     batch_size: 32,
//!     learning_rate: 0.001,
//!     early_stopping: Some(EarlyStoppingConfig::default()),
//!     ..Default::default()
//! };
//!
//! assert_eq!(config.epochs, 10);
//! assert_eq!(config.batch_size, 32);
//! ```

use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::{Float, NumAssign};
use std::collections::HashMap;
use std::fmt::Debug;

// Core training modules
pub mod augmentation;
pub mod backprop_efficient;
pub mod checkpoint;
pub mod early_stopping;
pub mod gradient_accumulation;
pub mod gradient_checkpointing;
pub mod gradient_clipping;
pub mod metrics_tracker;
pub mod mixed_precision;
pub mod progress_monitor;
pub mod quantization_aware;
pub mod schedulers;
pub mod sparse_training;

// Enhanced training loop (v0.2.0)
pub mod enhanced_trainer;
pub mod optimized_dataloader;

// Advanced training capabilities (v0.3.0+)
pub mod curriculum;
pub mod federated;
pub mod hparam_tuner;
pub mod lr_finder;
pub mod profiler;

// Re-export core modules
pub use backprop_efficient::*;
pub use checkpoint::{
    best_checkpoint, checkpoint_dir_name, latest_checkpoint, list_checkpoints, load_checkpoint,
    save_checkpoint, CheckpointConfig, CheckpointError, CheckpointManager, CheckpointMetadata,
    LrSchedulerState, OptimizerCheckpointState, OptimizerStateMetadata, ParamGroupState,
    TrainingCheckpoint,
};
pub use gradient_accumulation::*;
pub use gradient_checkpointing::*;
pub use mixed_precision::*;
pub use progress_monitor::*;
pub use quantization_aware::*;
pub use schedulers::{
    ChainedScheduler, CosineAnnealingLR as CosineAnnealingScheduler, CosineAnnealingWarmRestarts,
    CyclicLR, CyclicMode, ExponentialLR, LRScheduler, LinearWarmup, MultiStepLR, OneCycleLR,
    PolynomialLR, ReduceOnPlateau as ReduceOnPlateauScheduler, StepLR, WarmupCosine,
};
pub use sparse_training::*;

// Re-export gradient clipping (v0.3.0)
pub use gradient_clipping::{
    clip_grad_adaptive, clip_grad_agc, clip_grad_norm, clip_grad_value, grad_norm,
    AdaptiveGradClipConfig, ClipNormType, GradientClipResult,
};

// Re-export standalone early stopping (v0.3.0)
pub use early_stopping::{
    EarlyStopping, EarlyStoppingWithState, StepResult as EarlyStoppingStepResult, StopReason,
    StoppingMode,
};

// Re-export metrics tracker (v0.3.0)
pub use metrics_tracker::{
    BestMetric, MetricEntry, MetricGoal, MetricHistory, MetricStats, MetricsTracker,
    TrainingHistory,
};

// Re-export augmentation (v0.3.0)
pub use augmentation::{apply_cutmix, apply_mixup, AugmentationPipeline, AugmentationType};

// Re-export enhanced training (v0.2.0)
pub use enhanced_trainer::{
    EarlyStoppingConfig, EnhancedTrainer, EnhancedTrainingConfig, GradientAccumulationSettings,
    LRWarmupConfig, OperationTiming, OptimizationAnalyzer, OptimizationRecommendation,
    ProfilingConfig, ProfilingResults, ProgressConfig, RecommendationType, TrainingState,
    ValidationConfig, WarmupSchedule,
};
pub use optimized_dataloader::{
    BatchSizeOptimizationResult, BatchSizeOptimizer, LoadingStats, MemoryAwareConfig,
    MemoryAwareDataLoader, MemoryAwarePrefetchIter, OptimizedDataLoader, OptimizedLoaderConfig,
    PrefetchingIterator,
};

// Re-export LR finder (v0.3.0+)
pub use lr_finder::{
    find_optimal_lr, LRFinder, LRFinderConfig, LRFinderConfigBuilder, LRFinderPoint,
    LRFinderResult, LRFinderStatus, LRScheduleType, TypedLRFinder,
};

// Re-export curriculum learning (v0.3.0+)
pub use curriculum::{
    CompetenceSchedule, CurriculumConfig, CurriculumConfigBuilder, CurriculumLearner,
    CurriculumSchedule, CurriculumStrategy, DifficultyScorer, LossBasedScorer, StaticScorer,
};

// Re-export federated learning (v0.3.0+)
pub use federated::{
    clip_l2_norm, AggregationMethod, ClientSelectionStrategy, ClientUpdate,
    DifferentialPrivacyConfig, FederatedConfig, FederatedConfigBuilder, FederatedServer,
    GradientCompressionConfig, RoundStats,
};

// Re-export training profiler (v0.3.0+)
pub use profiler::{
    estimate_conv2d_memory, estimate_dense_flops, estimate_dense_memory, BatchStats, Bottleneck,
    EpochStats, LayerProfile, ProfilePhase, ProfileSummary, TrainingProfiler,
};

// Re-export hyperparameter tuner (v0.3.0+)
pub use hparam_tuner::{
    HParamSpace, HParamTuner, HParamValue, SearchStrategy, SpaceType, TrialResult,
};

/// Configuration structure for training neural networks
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of samples in each training batch
    pub batch_size: usize,
    /// Whether to shuffle the training data between epochs
    pub shuffle: bool,
    /// Number of parallel workers for data loading
    pub num_workers: usize,
    /// Base learning rate for the optimizer
    pub learning_rate: f64,
    /// Number of complete passes through the training dataset
    pub epochs: usize,
    /// Verbosity level for training output
    pub verbose: usize,
    /// Validation configuration
    pub validation: Option<ValidationSettings>,
    /// Gradient accumulation configuration
    pub gradient_accumulation: Option<GradientAccumulationConfig>,
    /// Mixed precision training configuration
    pub mixed_precision: Option<MixedPrecisionConfig>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
            num_workers: 0,
            learning_rate: 0.001,
            epochs: 10,
            verbose: 1,
            validation: None,
            gradient_accumulation: None,
            mixed_precision: None,
        }
    }
}

/// Configuration for validation during training
#[derive(Debug, Clone)]
pub struct ValidationSettings {
    /// Whether to enable validation during training
    pub enabled: bool,
    /// Fraction of training data to use for validation (0.0 to 1.0)
    pub validation_split: f64,
    /// Batch size for validation
    pub batch_size: usize,
    /// Number of parallel workers for validation data loading
    pub num_workers: usize,
}

impl Default for ValidationSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            validation_split: 0.2,
            batch_size: 32,
            num_workers: 0,
        }
    }
}

/// Training session for tracking training history
#[derive(Debug, Clone)]
pub struct TrainingSession<F: Float + Debug + ScalarOperand> {
    /// Training metrics history
    pub history: HashMap<String, Vec<F>>,
    /// Initial learning rate
    pub initial_learning_rate: F,
    /// Number of epochs trained
    pub epochs_trained: usize,
    /// Current epoch number
    pub current_epoch: usize,
    /// Best validation score achieved
    pub best_validation_score: Option<F>,
    /// Whether training has been stopped early
    pub early_stopped: bool,
}

impl<F: Float + Debug + ScalarOperand> TrainingSession<F> {
    /// Create a new training session
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            history: HashMap::new(),
            initial_learning_rate: F::from(config.learning_rate)
                .expect("Failed to convert to float"),
            epochs_trained: 0,
            current_epoch: 0,
            best_validation_score: None,
            early_stopped: false,
        }
    }

    /// Add a metric value to the history
    pub fn add_metric(&mut self, metricname: &str, value: F) {
        self.history
            .entry(metricname.to_string())
            .or_default()
            .push(value);
    }

    /// Get the history for a specific metric
    pub fn get_metric_history(&self, metricname: &str) -> Option<&Vec<F>> {
        self.history.get(metricname)
    }

    /// Get all metric names
    pub fn get_metric_names(&self) -> Vec<&String> {
        self.history.keys().collect()
    }

    /// Update the current epoch
    pub fn next_epoch(&mut self) {
        self.current_epoch += 1;
        self.epochs_trained += 1;
    }

    /// Mark training as completed
    pub fn finish_training(&mut self) {
        // Training completed normally
    }

    /// Mark training as early stopped
    pub fn early_stop(&mut self) {
        self.early_stopped = true;
    }
}

impl<F: Float + Debug + ScalarOperand> Default for TrainingSession<F> {
    fn default() -> Self {
        Self {
            history: HashMap::new(),
            initial_learning_rate: F::from(0.001).expect("Failed to convert constant to float"),
            epochs_trained: 0,
            current_epoch: 0,
            best_validation_score: None,
            early_stopped: false,
        }
    }
}
