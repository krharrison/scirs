//! Enhanced training loop with comprehensive features
//!
//! This module provides a production-ready training loop with:
//! - Train/validation split support
//! - Early stopping with patience
//! - Learning rate warmup and scheduling
//! - Gradient accumulation for memory efficiency
//! - Training progress tracking with metrics
//! - Profiled training with timing analysis
//! - Automatic optimization recommendations

use crate::callbacks::{Callback, CallbackContext, CallbackManager, CallbackTiming};
use crate::data::{DataLoader, Dataset};
use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use crate::losses::Loss;
use crate::models::History;
use crate::optimizers::Optimizer;
use crate::training::checkpoint::{
    CheckpointConfig, CheckpointManager, OptimizerCheckpointState, TrainingCheckpoint,
};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, ToPrimitive};
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::NumAssign;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Type alias for a boxed dataset with Send + Sync bounds
type BoxedDataset<F> = Box<dyn Dataset<F> + Send + Sync>;

// =============================================================================
// Enhanced Training Configuration
// =============================================================================

/// Comprehensive configuration for enhanced training
#[derive(Debug, Clone)]
pub struct EnhancedTrainingConfig {
    /// Number of epochs to train
    pub epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Initial learning rate
    pub learning_rate: f64,
    /// Whether to shuffle training data
    pub shuffle: bool,
    /// Validation configuration
    pub validation: Option<ValidationConfig>,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Learning rate warmup configuration
    pub lr_warmup: Option<LRWarmupConfig>,
    /// Gradient accumulation configuration
    pub gradient_accumulation: Option<GradientAccumulationSettings>,
    /// Progress tracking configuration
    pub progress: ProgressConfig,
    /// Profiling configuration
    pub profiling: ProfilingConfig,
    /// Verbosity level (0: silent, 1: progress, 2: detailed)
    pub verbose: usize,
    /// Checkpoint configuration (None = no checkpointing)
    pub checkpoint_config: Option<CheckpointConfig>,
    /// Path to a checkpoint directory to resume training from.
    /// If set, training will start at the epoch recorded in the checkpoint.
    pub resume_from: Option<PathBuf>,
}

impl Default for EnhancedTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 32,
            learning_rate: 0.001,
            shuffle: true,
            validation: Some(ValidationConfig::default()),
            early_stopping: Some(EarlyStoppingConfig::default()),
            lr_warmup: None,
            gradient_accumulation: None,
            progress: ProgressConfig::default(),
            profiling: ProfilingConfig::default(),
            verbose: 1,
            checkpoint_config: None,
            resume_from: None,
        }
    }
}

/// Configuration for validation during training
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Fraction of data to use for validation (0.0 to 1.0)
    pub validation_split: f64,
    /// Batch size for validation (uses training batch size if None)
    pub batch_size: Option<usize>,
    /// Frequency of validation (every N epochs)
    pub validation_frequency: usize,
    /// Metric to monitor for validation
    pub monitor_metric: String,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            validation_split: 0.2,
            batch_size: None,
            validation_frequency: 1,
            monitor_metric: "val_loss".to_string(),
        }
    }
}

/// Configuration for early stopping
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Number of epochs with no improvement before stopping
    pub patience: usize,
    /// Minimum change to qualify as improvement
    pub min_delta: f64,
    /// Whether to restore best weights
    pub restore_best_weights: bool,
    /// Metric to monitor
    pub monitor: String,
    /// Whether lower is better for the monitored metric
    pub mode_min: bool,
    /// Baseline value for the monitored metric
    pub baseline: Option<f64>,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            patience: 5,
            min_delta: 0.0001,
            restore_best_weights: true,
            monitor: "val_loss".to_string(),
            mode_min: true,
            baseline: None,
        }
    }
}

/// Configuration for learning rate warmup
#[derive(Debug, Clone)]
pub struct LRWarmupConfig {
    /// Number of warmup steps/epochs
    pub warmup_steps: usize,
    /// Whether warmup_steps is in epochs (true) or batches (false)
    pub warmup_by_epoch: bool,
    /// Initial learning rate multiplier (start from lr * initial_lr_scale)
    pub initial_lr_scale: f64,
    /// Warmup schedule type
    pub schedule: WarmupSchedule,
}

impl Default for LRWarmupConfig {
    fn default() -> Self {
        Self {
            warmup_steps: 5,
            warmup_by_epoch: true,
            initial_lr_scale: 0.01,
            schedule: WarmupSchedule::Linear,
        }
    }
}

/// Warmup schedule type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmupSchedule {
    /// Linear warmup
    Linear,
    /// Exponential warmup
    Exponential,
    /// Cosine warmup
    Cosine,
}

/// Configuration for gradient accumulation
#[derive(Debug, Clone)]
pub struct GradientAccumulationSettings {
    /// Number of steps to accumulate gradients over
    pub accumulation_steps: usize,
    /// Whether to normalize gradients by accumulation steps
    pub normalize_gradients: bool,
    /// Maximum gradient norm for clipping (None for no clipping)
    pub max_grad_norm: Option<f64>,
}

impl Default for GradientAccumulationSettings {
    fn default() -> Self {
        Self {
            accumulation_steps: 1,
            normalize_gradients: true,
            max_grad_norm: Some(1.0),
        }
    }
}

/// Configuration for progress tracking
#[derive(Debug, Clone)]
pub struct ProgressConfig {
    /// Whether to show progress bar
    pub show_progress: bool,
    /// Update frequency (batches)
    pub update_frequency: usize,
    /// Metrics to display
    pub display_metrics: Vec<String>,
    /// Whether to log to file
    pub log_to_file: bool,
    /// Log file path
    pub log_path: Option<String>,
}

impl Default for ProgressConfig {
    fn default() -> Self {
        Self {
            show_progress: true,
            update_frequency: 10,
            display_metrics: vec!["loss".to_string(), "val_loss".to_string()],
            log_to_file: false,
            log_path: None,
        }
    }
}

/// Configuration for profiling
#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    /// Whether to enable profiling
    pub enabled: bool,
    /// Profile forward pass
    pub profile_forward: bool,
    /// Profile backward pass
    pub profile_backward: bool,
    /// Profile optimizer step
    pub profile_optimizer: bool,
    /// Profile data loading
    pub profile_data_loading: bool,
    /// Generate optimization recommendations
    pub generate_recommendations: bool,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            profile_forward: true,
            profile_backward: true,
            profile_optimizer: true,
            profile_data_loading: true,
            generate_recommendations: true,
        }
    }
}

// =============================================================================
// Training State
// =============================================================================

/// Current state of training for callbacks and tracking
#[derive(Debug, Clone)]
pub struct TrainingState<F: Float + Debug + ScalarOperand + NumAssign> {
    /// Current epoch (0-indexed)
    pub current_epoch: usize,
    /// Total number of epochs
    pub total_epochs: usize,
    /// Current batch within epoch (0-indexed)
    pub current_batch: usize,
    /// Total batches per epoch
    pub total_batches: usize,
    /// Global step counter
    pub global_step: usize,
    /// Current learning rate
    pub current_lr: F,
    /// Best validation metric value
    pub best_metric_value: Option<F>,
    /// Epoch of best metric
    pub best_metric_epoch: Option<usize>,
    /// Whether training should stop
    pub should_stop: bool,
    /// Early stopping counter
    pub early_stopping_counter: usize,
    /// Training history
    pub history: History<F>,
    /// Current epoch metrics
    pub epoch_metrics: HashMap<String, F>,
    /// Batch metrics (running average)
    pub batch_metrics: HashMap<String, F>,
}

impl<F: Float + Debug + ScalarOperand + NumAssign> Default for TrainingState<F> {
    fn default() -> Self {
        Self {
            current_epoch: 0,
            total_epochs: 0,
            current_batch: 0,
            total_batches: 0,
            global_step: 0,
            current_lr: F::from(0.001).unwrap_or_else(|| F::zero()),
            best_metric_value: None,
            best_metric_epoch: None,
            should_stop: false,
            early_stopping_counter: 0,
            history: History::default(),
            epoch_metrics: HashMap::new(),
            batch_metrics: HashMap::new(),
        }
    }
}

// =============================================================================
// Profiling Results
// =============================================================================

/// Timing information for a single operation
#[derive(Debug, Clone)]
pub struct OperationTiming {
    /// Operation name
    pub name: String,
    /// Total time spent
    pub total_time: Duration,
    /// Number of calls
    pub call_count: usize,
    /// Average time per call
    pub avg_time: Duration,
    /// Minimum time
    pub min_time: Duration,
    /// Maximum time
    pub max_time: Duration,
}

impl Default for OperationTiming {
    fn default() -> Self {
        Self {
            name: String::new(),
            total_time: Duration::ZERO,
            call_count: 0,
            avg_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
        }
    }
}

impl OperationTiming {
    /// Create a new timing tracker
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Record a new timing
    pub fn record(&mut self, duration: Duration) {
        self.total_time += duration;
        self.call_count += 1;
        self.avg_time = self.total_time / self.call_count as u32;
        self.min_time = self.min_time.min(duration);
        self.max_time = self.max_time.max(duration);
    }
}

/// Profiling results for the entire training run
#[derive(Debug, Clone, Default)]
pub struct ProfilingResults {
    /// Forward pass timings
    pub forward_timing: OperationTiming,
    /// Backward pass timings
    pub backward_timing: OperationTiming,
    /// Optimizer step timings
    pub optimizer_timing: OperationTiming,
    /// Data loading timings
    pub data_loading_timing: OperationTiming,
    /// Validation timings
    pub validation_timing: OperationTiming,
    /// Total epoch timings
    pub epoch_timing: OperationTiming,
    /// Custom operation timings
    pub custom_timings: HashMap<String, OperationTiming>,
}

impl ProfilingResults {
    /// Create new profiling results
    pub fn new() -> Self {
        Self {
            forward_timing: OperationTiming::new("forward"),
            backward_timing: OperationTiming::new("backward"),
            optimizer_timing: OperationTiming::new("optimizer"),
            data_loading_timing: OperationTiming::new("data_loading"),
            validation_timing: OperationTiming::new("validation"),
            epoch_timing: OperationTiming::new("epoch"),
            custom_timings: HashMap::new(),
        }
    }

    /// Get a summary of profiling results
    pub fn summary(&self) -> String {
        let mut result = String::from("=== Profiling Results ===\n");

        let format_timing = |t: &OperationTiming| -> String {
            if t.call_count > 0 {
                format!(
                    "{}: total={:?}, avg={:?}, min={:?}, max={:?}, calls={}",
                    t.name, t.total_time, t.avg_time, t.min_time, t.max_time, t.call_count
                )
            } else {
                format!("{}: no data", t.name)
            }
        };

        result.push_str(&format_timing(&self.epoch_timing));
        result.push('\n');
        result.push_str(&format_timing(&self.data_loading_timing));
        result.push('\n');
        result.push_str(&format_timing(&self.forward_timing));
        result.push('\n');
        result.push_str(&format_timing(&self.backward_timing));
        result.push('\n');
        result.push_str(&format_timing(&self.optimizer_timing));
        result.push('\n');
        result.push_str(&format_timing(&self.validation_timing));
        result.push('\n');

        for timing in self.custom_timings.values() {
            result.push_str(&format_timing(timing));
            result.push('\n');
        }

        result
    }
}

// =============================================================================
// Optimization Recommendations
// =============================================================================

/// Type of optimization recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendationType {
    /// Increase batch size
    IncreaseBatchSize,
    /// Use gradient accumulation
    UseGradientAccumulation,
    /// Use mixed precision
    UseMixedPrecision,
    /// Optimize data loading
    OptimizeDataLoading,
    /// Use learning rate warmup
    UseLRWarmup,
    /// Adjust learning rate
    AdjustLearningRate,
    /// Use early stopping
    UseEarlyStopping,
    /// Other recommendation
    Other,
}

/// An optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Type of recommendation
    pub recommendation_type: RecommendationType,
    /// Priority (1-10, higher is more important)
    pub priority: u8,
    /// Description of the recommendation
    pub description: String,
    /// Expected improvement
    pub expected_improvement: String,
    /// How to implement the recommendation
    pub implementation_hint: String,
}

/// Analyzer for generating optimization recommendations
pub struct OptimizationAnalyzer {
    /// Profiling results to analyze
    profiling_results: ProfilingResults,
    /// Training configuration used
    config: EnhancedTrainingConfig,
    /// Training history
    train_loss_history: Vec<f64>,
    /// Validation loss history
    val_loss_history: Vec<f64>,
}

impl OptimizationAnalyzer {
    /// Create a new analyzer
    pub fn new(
        profiling_results: ProfilingResults,
        config: EnhancedTrainingConfig,
        train_loss_history: Vec<f64>,
        val_loss_history: Vec<f64>,
    ) -> Self {
        Self {
            profiling_results,
            config,
            train_loss_history,
            val_loss_history,
        }
    }

    /// Generate optimization recommendations
    pub fn generate_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Check data loading bottleneck
        self.check_data_loading_bottleneck(&mut recommendations);

        // Check if gradient accumulation would help
        self.check_gradient_accumulation(&mut recommendations);

        // Check learning rate issues
        self.check_learning_rate_issues(&mut recommendations);

        // Check for overfitting
        self.check_overfitting(&mut recommendations);

        // Check warmup benefit
        self.check_warmup_benefit(&mut recommendations);

        // Sort by priority
        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));

        recommendations
    }

    fn check_data_loading_bottleneck(&self, recommendations: &mut Vec<OptimizationRecommendation>) {
        let data_time = self
            .profiling_results
            .data_loading_timing
            .total_time
            .as_secs_f64();
        let forward_time = self
            .profiling_results
            .forward_timing
            .total_time
            .as_secs_f64();
        let backward_time = self
            .profiling_results
            .backward_timing
            .total_time
            .as_secs_f64();

        let compute_time = forward_time + backward_time;

        if data_time > 0.0 && compute_time > 0.0 {
            let data_ratio = data_time / (data_time + compute_time);

            if data_ratio > 0.3 {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: RecommendationType::OptimizeDataLoading,
                    priority: 8,
                    description: format!(
                        "Data loading takes {:.1}% of training time",
                        data_ratio * 100.0
                    ),
                    expected_improvement: "Up to 30% faster training".to_string(),
                    implementation_hint:
                        "Consider using prefetching, more workers, or caching data in memory"
                            .to_string(),
                });
            }
        }
    }

    fn check_gradient_accumulation(&self, recommendations: &mut Vec<OptimizationRecommendation>) {
        if self.config.gradient_accumulation.is_none() && self.config.batch_size < 64 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::UseGradientAccumulation,
                priority: 5,
                description: "Small batch size may lead to noisy gradients".to_string(),
                expected_improvement: "More stable training with larger effective batch size"
                    .to_string(),
                implementation_hint: format!(
                    "Try gradient_accumulation with {} steps for effective batch size {}",
                    4,
                    self.config.batch_size * 4
                ),
            });
        }
    }

    fn check_learning_rate_issues(&self, recommendations: &mut Vec<OptimizationRecommendation>) {
        if self.train_loss_history.len() >= 3 {
            // Check for loss not decreasing
            let recent_losses: Vec<f64> = self
                .train_loss_history
                .iter()
                .rev()
                .take(5)
                .copied()
                .collect();

            if recent_losses.len() >= 3 {
                let avg_change: f64 = recent_losses
                    .windows(2)
                    .map(|w| (w[0] - w[1]).abs())
                    .sum::<f64>()
                    / (recent_losses.len() - 1) as f64;

                let first_loss = self.train_loss_history.first().copied().unwrap_or(0.0);
                let relative_change = if first_loss > 0.0 {
                    avg_change / first_loss
                } else {
                    0.0
                };

                if relative_change < 0.001 {
                    recommendations.push(OptimizationRecommendation {
                        recommendation_type: RecommendationType::AdjustLearningRate,
                        priority: 7,
                        description: "Training loss has plateaued".to_string(),
                        expected_improvement: "Continued loss decrease".to_string(),
                        implementation_hint:
                            "Try increasing learning rate or using learning rate scheduling"
                                .to_string(),
                    });
                }
            }
        }
    }

    fn check_overfitting(&self, recommendations: &mut Vec<OptimizationRecommendation>) {
        if self.train_loss_history.len() >= 5 && self.val_loss_history.len() >= 5 {
            let recent_train: Vec<f64> = self
                .train_loss_history
                .iter()
                .rev()
                .take(5)
                .copied()
                .collect();
            let recent_val: Vec<f64> = self
                .val_loss_history
                .iter()
                .rev()
                .take(5)
                .copied()
                .collect();

            // Check if training loss decreasing but validation increasing
            let train_decreasing = recent_train.windows(2).all(|w| w[0] <= w[1] * 1.05);
            let val_increasing = recent_val.windows(2).filter(|w| w[0] > w[1]).count() >= 3;

            if train_decreasing && val_increasing && self.config.early_stopping.is_none() {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: RecommendationType::UseEarlyStopping,
                    priority: 9,
                    description: "Signs of overfitting detected".to_string(),
                    expected_improvement: "Prevent overfitting and save training time".to_string(),
                    implementation_hint: "Enable early stopping with patience=5".to_string(),
                });
            }
        }
    }

    fn check_warmup_benefit(&self, recommendations: &mut Vec<OptimizationRecommendation>) {
        if self.config.lr_warmup.is_none() && self.train_loss_history.len() >= 3 {
            // Check for unstable initial training
            let initial_losses: Vec<f64> =
                self.train_loss_history.iter().take(3).copied().collect();

            if initial_losses.len() >= 2 {
                let volatility: f64 = initial_losses
                    .windows(2)
                    .map(|w| (w[1] - w[0]).abs() / w[0].max(1e-8))
                    .sum::<f64>()
                    / (initial_losses.len() - 1) as f64;

                if volatility > 0.5 {
                    recommendations.push(OptimizationRecommendation {
                        recommendation_type: RecommendationType::UseLRWarmup,
                        priority: 6,
                        description: "High loss volatility in early training".to_string(),
                        expected_improvement: "More stable training start".to_string(),
                        implementation_hint: "Try warmup for first 5 epochs with linear schedule"
                            .to_string(),
                    });
                }
            }
        }
    }
}

// =============================================================================
// Enhanced Trainer
// =============================================================================

/// Enhanced trainer with comprehensive training features
pub struct EnhancedTrainer<
    F: Float + Debug + ScalarOperand + FromPrimitive + ToPrimitive + NumAssign + Send + Sync + 'static,
> {
    /// Training configuration
    config: EnhancedTrainingConfig,
    /// Training state
    state: TrainingState<F>,
    /// Callback manager
    callback_manager: CallbackManager<F>,
    /// Profiling results
    profiling_results: ProfilingResults,
    /// Best model weights (for early stopping)
    best_weights: Option<Vec<Array<F, IxDyn>>>,
    /// Checkpoint manager (instantiated if `config.checkpoint_config` is Some)
    checkpoint_manager: Option<CheckpointManager<F>>,
}

impl<
        F: Float
            + Debug
            + ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + NumAssign
            + Send
            + Sync
            + 'static,
    > EnhancedTrainer<F>
{
    /// Create a new enhanced trainer
    pub fn new(config: EnhancedTrainingConfig) -> Self {
        let checkpoint_manager = config.checkpoint_config.clone().map(CheckpointManager::new);
        Self {
            config,
            state: TrainingState::default(),
            callback_manager: CallbackManager::new(),
            profiling_results: ProfilingResults::new(),
            best_weights: None,
            checkpoint_manager,
        }
    }

    /// Add a callback to the trainer
    pub fn add_callback<C: Callback<F> + Send + Sync + 'static>(&mut self, callback: C) {
        self.callback_manager.add_callback(Box::new(callback));
    }

    /// Get the training state
    pub fn state(&self) -> &TrainingState<F> {
        &self.state
    }

    /// Get the profiling results
    pub fn profiling_results(&self) -> &ProfilingResults {
        &self.profiling_results
    }

    /// Get optimization recommendations
    pub fn get_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let train_loss: Vec<f64> = self
            .state
            .history
            .train_loss
            .iter()
            .map(|f| f.to_f64().unwrap_or(0.0))
            .collect();
        let val_loss: Vec<f64> = self
            .state
            .history
            .val_loss
            .iter()
            .map(|f| f.to_f64().unwrap_or(0.0))
            .collect();

        let analyzer = OptimizationAnalyzer::new(
            self.profiling_results.clone(),
            self.config.clone(),
            train_loss,
            val_loss,
        );

        analyzer.generate_recommendations()
    }

    /// Calculate warmup learning rate
    fn calculate_warmup_lr(&self, step: usize, base_lr: F) -> F {
        let warmup_config = match &self.config.lr_warmup {
            Some(cfg) => cfg,
            None => return base_lr,
        };

        let warmup_steps = warmup_config.warmup_steps;
        if step >= warmup_steps {
            return base_lr;
        }

        let initial_scale = F::from(warmup_config.initial_lr_scale)
            .unwrap_or_else(|| F::from(0.01).unwrap_or_else(|| F::zero()));
        let progress = F::from(step).unwrap_or_else(|| F::zero())
            / F::from(warmup_steps).unwrap_or_else(|| F::one());

        match warmup_config.schedule {
            WarmupSchedule::Linear => {
                let scale = initial_scale + (F::one() - initial_scale) * progress;
                base_lr * scale
            }
            WarmupSchedule::Exponential => {
                let log_initial = initial_scale.ln();
                let log_scale = log_initial * (F::one() - progress);
                base_lr * log_scale.exp()
            }
            WarmupSchedule::Cosine => {
                let pi = F::from(std::f64::consts::PI).unwrap_or_else(|| F::zero());
                let cosine_factor =
                    (F::one() - (progress * pi).cos()) / F::from(2.0).unwrap_or_else(|| F::one());
                let scale = initial_scale + (F::one() - initial_scale) * cosine_factor;
                base_lr * scale
            }
        }
    }

    /// Check early stopping condition
    fn check_early_stopping(&mut self, metric_value: F) -> bool {
        let early_stopping = match &self.config.early_stopping {
            Some(cfg) => cfg,
            None => return false,
        };

        let min_delta = F::from(early_stopping.min_delta).unwrap_or_else(|| F::zero());

        let improved = match self.state.best_metric_value {
            None => true,
            Some(best) => {
                if early_stopping.mode_min {
                    metric_value < best - min_delta
                } else {
                    metric_value > best + min_delta
                }
            }
        };

        if improved {
            self.state.best_metric_value = Some(metric_value);
            self.state.best_metric_epoch = Some(self.state.current_epoch);
            self.state.early_stopping_counter = 0;
            true
        } else {
            self.state.early_stopping_counter += 1;
            if self.state.early_stopping_counter >= early_stopping.patience {
                self.state.should_stop = true;
            }
            false
        }
    }

    /// Train the model
    pub fn fit<D, L, O>(
        &mut self,
        model: &mut dyn ParamLayer<F>,
        train_dataset: D,
        loss_fn: &L,
        optimizer: &mut O,
    ) -> Result<&History<F>>
    where
        D: Dataset<F> + Clone + 'static,
        L: Loss<F>,
        O: Optimizer<F>,
    {
        let base_lr = F::from(self.config.learning_rate).ok_or_else(|| {
            NeuralError::TrainingError("Failed to convert learning rate".to_string())
        })?;
        self.state.current_lr = base_lr;
        self.state.total_epochs = self.config.epochs;

        // Create train/validation split if configured
        let (train_data, val_data) = self.create_train_val_split(train_dataset)?;

        // Create data loaders
        let train_loader = DataLoader::new(
            train_data,
            self.config.batch_size,
            self.config.shuffle,
            false,
        );

        let val_loader = val_data.map(|vd| {
            let val_batch_size = self
                .config
                .validation
                .as_ref()
                .and_then(|v| v.batch_size)
                .unwrap_or(self.config.batch_size);
            DataLoader::new(vd, val_batch_size, false, false)
        });

        self.state.total_batches = train_loader.num_batches();

        // ── Resume from checkpoint ────────────────────────────────────────────
        let resume_epoch = if let Some(ref resume_path) = self.config.resume_from.clone() {
            match CheckpointManager::<F>::load(resume_path) {
                Ok((ckpt_state, model_params)) => {
                    // Restore model parameters from the checkpoint
                    let params_vec: Vec<Array<F, IxDyn>> = model_params
                        .parameters
                        .iter()
                        .filter_map(|(_, values, shape)| {
                            let ix_shape = scirs2_core::ndarray::IxDyn(shape.as_slice());
                            let f_vec: Vec<F> = values.iter().filter_map(|&v| F::from(v)).collect();
                            if f_vec.len() != values.len() {
                                return None;
                            }
                            Array::from_shape_vec(ix_shape, f_vec).ok()
                        })
                        .collect();
                    if !params_vec.is_empty() {
                        if let Err(e) = model.set_parameters(params_vec) {
                            if self.config.verbose >= 1 {
                                eprintln!("Warning: could not restore model parameters: {e}");
                            }
                        }
                    }
                    // Restore training state
                    self.state.global_step = ckpt_state.step;
                    if let Some(best) = ckpt_state.best_metric {
                        if let Some(f_best) = F::from(best) {
                            self.state.best_metric_value = Some(f_best);
                        }
                    }
                    let resume_epoch = ckpt_state.epoch + 1; // start from next epoch
                    if self.config.verbose >= 1 {
                        println!(
                            "Resumed from checkpoint at epoch {}, step {}",
                            ckpt_state.epoch, ckpt_state.step
                        );
                    }
                    resume_epoch
                }
                Err(e) => {
                    if self.config.verbose >= 1 {
                        eprintln!(
                            "Warning: failed to load checkpoint from {:?}: {e}",
                            resume_path
                        );
                    }
                    0
                }
            }
        } else {
            0
        };

        // Notify callbacks of training start
        self.callback_manager.on_train_begin()?;

        // Training loop — start from resume_epoch if resuming
        for epoch in resume_epoch..self.config.epochs {
            if self.state.should_stop {
                break;
            }

            self.state.current_epoch = epoch;

            let epoch_start = Instant::now();

            // Notify callbacks of epoch start
            self.callback_manager.on_epoch_begin(epoch)?;

            // Train epoch
            let train_loss = self.train_epoch(model, &train_loader, loss_fn, optimizer, epoch)?;

            // Validation
            let val_loss = if let Some(ref loader) = val_loader {
                if epoch
                    % self
                        .config
                        .validation
                        .as_ref()
                        .map_or(1, |v| v.validation_frequency)
                    == 0
                {
                    let val_start = Instant::now();
                    let loss = self.validate(model, loader, loss_fn)?;
                    if self.config.profiling.enabled {
                        self.profiling_results
                            .validation_timing
                            .record(val_start.elapsed());
                    }
                    Some(loss)
                } else {
                    None
                }
            } else {
                None
            };

            // Record epoch timing
            if self.config.profiling.enabled {
                self.profiling_results
                    .epoch_timing
                    .record(epoch_start.elapsed());
            }

            // Update history
            self.state.history.train_loss.push(train_loss);
            if let Some(vl) = val_loss {
                self.state.history.val_loss.push(vl);
            }

            // Update epoch metrics
            self.state
                .epoch_metrics
                .insert("loss".to_string(), train_loss);
            if let Some(vl) = val_loss {
                self.state.epoch_metrics.insert("val_loss".to_string(), vl);
            }

            // Check early stopping
            if self.config.early_stopping.is_some() {
                let metric = val_loss.unwrap_or(train_loss);
                let improved = self.check_early_stopping(metric);

                if improved
                    && self
                        .config
                        .early_stopping
                        .as_ref()
                        .is_some_and(|e| e.restore_best_weights)
                {
                    self.best_weights = Some(model.get_parameters());
                }
            }

            // Notify callbacks of epoch end
            let stop = self
                .callback_manager
                .on_epoch_end(epoch, &self.state.epoch_metrics)?;
            if stop {
                self.state.should_stop = true;
            }

            // Print progress
            if self.config.verbose >= 1 {
                self.print_epoch_summary(epoch, train_loss, val_loss);
            }

            // ── Checkpoint saving ─────────────────────────────────────────────
            if let Some(ref mut ckpt_mgr) = self.checkpoint_manager {
                // Build model parameters snapshot via get_parameters()
                let raw_params = model.get_parameters();
                let mut named_params = crate::serialization::traits::NamedParameters::new();
                for (idx, param) in raw_params.iter().enumerate() {
                    let name = format!("param_{idx:04}");
                    let flat: Vec<f64> = param.iter().filter_map(|v| v.to_f64()).collect();
                    let shape: Vec<usize> = param.shape().to_vec();
                    named_params.add(&name, flat, shape);
                }

                // Build epoch metrics as HashMap<String, F>
                let epoch_metrics_f: HashMap<String, F> = self.state.epoch_metrics.clone();

                // Build TrainingCheckpoint
                let mut ckpt = TrainingCheckpoint::new(epoch, self.state.global_step, "model");
                ckpt.best_metric = self.state.best_metric_value.and_then(|v| v.to_f64());
                ckpt.total_epochs = Some(self.config.epochs);
                ckpt.optimizer_state = OptimizerCheckpointState {
                    optimizer_type: "Unknown".to_string(),
                    learning_rate: self.state.current_lr.to_f64().unwrap_or(0.001),
                    step: self.state.global_step,
                    ..Default::default()
                };
                // Record epoch metrics in history
                let metrics_f64: HashMap<String, f64> = epoch_metrics_f
                    .iter()
                    .filter_map(|(k, v)| v.to_f64().map(|fv| (k.clone(), fv)))
                    .collect();
                ckpt.metrics_history.push(metrics_f64);

                match ckpt_mgr.save(&ckpt, &named_params, epoch, &epoch_metrics_f) {
                    Ok(path) => {
                        if self.config.verbose >= 2 {
                            println!("Checkpoint saved to: {}", path.display());
                        }
                    }
                    Err(e) => {
                        if self.config.verbose >= 1 {
                            eprintln!("Warning: checkpoint save failed at epoch {epoch}: {e}");
                        }
                    }
                }
            }
        }

        // Restore best weights if early stopping triggered
        if self.state.should_stop
            && self
                .config
                .early_stopping
                .as_ref()
                .is_some_and(|e| e.restore_best_weights)
        {
            if let Some(ref weights) = self.best_weights {
                model.set_parameters(weights.clone())?;
                if self.config.verbose >= 1 {
                    println!(
                        "Restored best weights from epoch {}",
                        self.state.best_metric_epoch.unwrap_or(0)
                    );
                }
            }
        }

        // Notify callbacks of training end
        self.callback_manager.on_train_end()?;

        // Print profiling results if enabled
        if self.config.profiling.enabled && self.config.verbose >= 1 {
            println!("\n{}", self.profiling_results.summary());
        }

        // Print recommendations if enabled
        if self.config.profiling.generate_recommendations && self.config.verbose >= 1 {
            let recommendations = self.get_recommendations();
            if !recommendations.is_empty() {
                println!("\n=== Optimization Recommendations ===");
                for rec in recommendations.iter().take(5) {
                    println!(
                        "[Priority {}] {}: {}",
                        rec.priority, rec.description, rec.implementation_hint
                    );
                }
            }
        }

        Ok(&self.state.history)
    }

    /// Create train/validation split
    fn create_train_val_split<D: Dataset<F> + Clone + 'static>(
        &self,
        dataset: D,
    ) -> Result<(BoxedDataset<F>, Option<BoxedDataset<F>>)> {
        let validation_config = match &self.config.validation {
            Some(cfg) if cfg.validation_split > 0.0 => cfg,
            _ => return Ok((Box::new(dataset), None)),
        };

        let total_samples = dataset.len();
        let val_samples = (total_samples as f64 * validation_config.validation_split) as usize;
        let train_samples = total_samples - val_samples;

        if train_samples == 0 || val_samples == 0 {
            return Err(NeuralError::TrainingError(
                "Invalid validation split resulting in empty dataset".to_string(),
            ));
        }

        // Create indices and shuffle
        let mut indices: Vec<usize> = (0..total_samples).collect();
        let mut rng = scirs2_core::random::rng();
        indices.shuffle(&mut rng);

        let train_indices: Vec<usize> = indices[..train_samples].to_vec();
        let val_indices: Vec<usize> = indices[train_samples..].to_vec();

        // Create subset datasets
        let train_dataset = crate::data::SubsetDataset::new(dataset.clone(), train_indices)?;
        let val_dataset = crate::data::SubsetDataset::new(dataset, val_indices)?;

        Ok((Box::new(train_dataset), Some(Box::new(val_dataset))))
    }

    /// Train a single epoch
    fn train_epoch<D, L, O>(
        &mut self,
        model: &mut dyn ParamLayer<F>,
        loader: &DataLoader<F, D>,
        loss_fn: &L,
        optimizer: &mut O,
        epoch: usize,
    ) -> Result<F>
    where
        D: Dataset<F> + Send + Sync,
        L: Loss<F>,
        O: Optimizer<F>,
    {
        let base_lr = F::from(self.config.learning_rate).ok_or_else(|| {
            NeuralError::TrainingError("Failed to convert learning rate".to_string())
        })?;

        let mut total_loss = F::zero();
        let mut batch_count = 0usize;

        let accumulation_steps = self
            .config
            .gradient_accumulation
            .as_ref()
            .map_or(1, |g| g.accumulation_steps);

        // Reset loader for new epoch
        let mut train_loader = DataLoader::new(
            loader.dataset.box_clone(),
            loader.batch_size,
            self.config.shuffle,
            loader.drop_last,
        );
        train_loader.reset();

        // Collect batches
        let batches: Vec<_> = (&mut train_loader).collect();

        for (batch_idx, batch_result) in batches.into_iter().enumerate() {
            self.state.current_batch = batch_idx;
            self.state.global_step = epoch * self.state.total_batches + batch_idx;

            // Calculate warmup learning rate
            let warmup_step = if self
                .config
                .lr_warmup
                .as_ref()
                .is_some_and(|w| w.warmup_by_epoch)
            {
                epoch
            } else {
                self.state.global_step
            };
            self.state.current_lr = self.calculate_warmup_lr(warmup_step, base_lr);
            optimizer.set_learning_rate(self.state.current_lr);

            // Notify callbacks
            self.callback_manager.on_batch_begin(batch_idx)?;

            // Load batch
            let data_start = Instant::now();
            let (inputs, targets) = batch_result?;
            if self.config.profiling.enabled && self.config.profiling.profile_data_loading {
                self.profiling_results
                    .data_loading_timing
                    .record(data_start.elapsed());
            }

            // Forward pass
            let forward_start = Instant::now();
            let outputs = model.forward(&inputs)?;
            if self.config.profiling.enabled && self.config.profiling.profile_forward {
                self.profiling_results
                    .forward_timing
                    .record(forward_start.elapsed());
            }

            // Compute loss
            let loss = loss_fn.forward(&outputs, &targets)?;
            total_loss += loss;
            batch_count += 1;

            // Backward pass
            let backward_start = Instant::now();
            let grad_output = loss_fn.backward(&outputs, &targets)?;
            let _grad_input = model.backward(&inputs, &grad_output)?;
            if self.config.profiling.enabled && self.config.profiling.profile_backward {
                self.profiling_results
                    .backward_timing
                    .record(backward_start.elapsed());
            }

            // Gradient accumulation step
            if (batch_idx + 1) % accumulation_steps == 0
                || batch_idx == self.state.total_batches - 1
            {
                let opt_start = Instant::now();

                // Apply gradient normalization if configured
                if self
                    .config
                    .gradient_accumulation
                    .as_ref()
                    .is_some_and(|g| g.normalize_gradients)
                {
                    let scale = F::from(accumulation_steps).ok_or_else(|| {
                        NeuralError::TrainingError(
                            "Failed to convert accumulation steps".to_string(),
                        )
                    })?;
                    let grads = model.get_gradients();
                    let scaled_grads: Vec<_> =
                        grads.iter().map(|g| g.mapv(|v| v / scale)).collect();
                    model.set_gradients(&scaled_grads)?;
                }

                // Optimizer step
                optimizer.step_model(model)?;

                if self.config.profiling.enabled && self.config.profiling.profile_optimizer {
                    self.profiling_results
                        .optimizer_timing
                        .record(opt_start.elapsed());
                }
            }

            // Update batch metrics
            self.state.batch_metrics.insert("loss".to_string(), loss);

            // Notify callbacks
            self.callback_manager
                .on_batch_end(batch_idx, &self.state.batch_metrics)?;
        }

        // Calculate average loss
        let avg_loss = if batch_count > 0 {
            total_loss / F::from(batch_count).unwrap_or_else(|| F::one())
        } else {
            F::zero()
        };

        Ok(avg_loss)
    }

    /// Validate the model
    fn validate<D, L>(
        &self,
        model: &dyn ParamLayer<F>,
        loader: &DataLoader<F, D>,
        loss_fn: &L,
    ) -> Result<F>
    where
        D: Dataset<F> + Send + Sync,
        L: Loss<F>,
    {
        let mut total_loss = F::zero();
        let mut batch_count = 0usize;

        // Create a new loader for validation
        let mut val_loader =
            DataLoader::new(loader.dataset.box_clone(), loader.batch_size, false, false);
        val_loader.reset();

        for batch_result in &mut val_loader {
            let (inputs, targets) = batch_result?;
            let outputs = model.forward(&inputs)?;
            let loss = loss_fn.forward(&outputs, &targets)?;
            total_loss += loss;
            batch_count += 1;
        }

        let avg_loss = if batch_count > 0 {
            total_loss / F::from(batch_count).unwrap_or_else(|| F::one())
        } else {
            F::zero()
        };

        Ok(avg_loss)
    }

    /// Print epoch summary
    fn print_epoch_summary(&self, epoch: usize, train_loss: F, val_loss: Option<F>) {
        let mut summary = format!(
            "Epoch {}/{} - loss: {:.6}",
            epoch + 1,
            self.config.epochs,
            train_loss.to_f64().unwrap_or(0.0)
        );

        if let Some(vl) = val_loss {
            summary.push_str(&format!(" - val_loss: {:.6}", vl.to_f64().unwrap_or(0.0)));
        }

        if self.config.verbose >= 2 {
            summary.push_str(&format!(
                " - lr: {:.6}",
                self.state.current_lr.to_f64().unwrap_or(0.0)
            ));

            if let Some(ref es) = self.config.early_stopping {
                summary.push_str(&format!(
                    " - patience: {}/{}",
                    self.state.early_stopping_counter, es.patience
                ));
            }
        }

        println!("{summary}");
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_training_config_default() {
        let config = EnhancedTrainingConfig::default();
        assert_eq!(config.epochs, 10);
        assert_eq!(config.batch_size, 32);
        assert!((config.learning_rate - 0.001).abs() < 1e-10);
        assert!(config.shuffle);
        assert!(config.validation.is_some());
        assert!(config.early_stopping.is_some());
    }

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert!((config.validation_split - 0.2).abs() < 1e-10);
        assert_eq!(config.validation_frequency, 1);
        assert_eq!(config.monitor_metric, "val_loss");
    }

    #[test]
    fn test_early_stopping_config_default() {
        let config = EarlyStoppingConfig::default();
        assert_eq!(config.patience, 5);
        assert!((config.min_delta - 0.0001).abs() < 1e-10);
        assert!(config.restore_best_weights);
        assert!(config.mode_min);
    }

    #[test]
    fn test_lr_warmup_config_default() {
        let config = LRWarmupConfig::default();
        assert_eq!(config.warmup_steps, 5);
        assert!(config.warmup_by_epoch);
        assert!((config.initial_lr_scale - 0.01).abs() < 1e-10);
        assert_eq!(config.schedule, WarmupSchedule::Linear);
    }

    #[test]
    fn test_gradient_accumulation_settings_default() {
        let config = GradientAccumulationSettings::default();
        assert_eq!(config.accumulation_steps, 1);
        assert!(config.normalize_gradients);
        assert!(config.max_grad_norm.is_some());
    }

    #[test]
    fn test_operation_timing() {
        let mut timing = OperationTiming::new("test");
        timing.record(Duration::from_millis(100));
        timing.record(Duration::from_millis(200));

        assert_eq!(timing.call_count, 2);
        assert_eq!(timing.total_time, Duration::from_millis(300));
        assert_eq!(timing.avg_time, Duration::from_millis(150));
        assert_eq!(timing.min_time, Duration::from_millis(100));
        assert_eq!(timing.max_time, Duration::from_millis(200));
    }

    #[test]
    fn test_profiling_results_new() {
        let results = ProfilingResults::new();
        assert_eq!(results.forward_timing.name, "forward");
        assert_eq!(results.backward_timing.name, "backward");
        assert_eq!(results.optimizer_timing.name, "optimizer");
    }

    #[test]
    fn test_profiling_summary() {
        let mut results = ProfilingResults::new();
        results.forward_timing.record(Duration::from_millis(100));
        results.backward_timing.record(Duration::from_millis(150));

        let summary = results.summary();
        assert!(summary.contains("forward"));
        assert!(summary.contains("backward"));
    }

    #[test]
    fn test_training_state_default() {
        let state: TrainingState<f64> = TrainingState::default();
        assert_eq!(state.current_epoch, 0);
        assert_eq!(state.global_step, 0);
        assert!(!state.should_stop);
        assert_eq!(state.early_stopping_counter, 0);
    }

    #[test]
    fn test_optimization_recommendation() {
        let rec = OptimizationRecommendation {
            recommendation_type: RecommendationType::IncreaseBatchSize,
            priority: 8,
            description: "Test description".to_string(),
            expected_improvement: "50% faster".to_string(),
            implementation_hint: "Do this".to_string(),
        };

        assert_eq!(rec.priority, 8);
        assert_eq!(
            rec.recommendation_type,
            RecommendationType::IncreaseBatchSize
        );
    }

    #[test]
    fn test_enhanced_trainer_creation() {
        let config = EnhancedTrainingConfig::default();
        let trainer: EnhancedTrainer<f64> = EnhancedTrainer::new(config);

        assert_eq!(trainer.state().current_epoch, 0);
        assert!(!trainer.state().should_stop);
    }

    #[test]
    fn test_warmup_schedule_linear() {
        let config = EnhancedTrainingConfig {
            lr_warmup: Some(LRWarmupConfig {
                warmup_steps: 10,
                warmup_by_epoch: true,
                initial_lr_scale: 0.1,
                schedule: WarmupSchedule::Linear,
            }),
            ..Default::default()
        };

        let trainer: EnhancedTrainer<f64> = EnhancedTrainer::new(config);

        // At step 0, should be at initial scale
        let lr_0 = trainer.calculate_warmup_lr(0, 0.01);
        assert!((lr_0 - 0.001).abs() < 1e-10); // 0.01 * 0.1 = 0.001

        // At step 5, should be at 55% of base (linear interpolation)
        let lr_5 = trainer.calculate_warmup_lr(5, 0.01);
        let expected = 0.01 * (0.1 + (1.0 - 0.1) * 0.5);
        assert!((lr_5 - expected).abs() < 1e-10);

        // At step 10+, should be at full learning rate
        let lr_10 = trainer.calculate_warmup_lr(10, 0.01);
        assert!((lr_10 - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_early_stopping_check() {
        let config = EnhancedTrainingConfig {
            early_stopping: Some(EarlyStoppingConfig {
                patience: 3,
                min_delta: 0.01,
                mode_min: true,
                ..Default::default()
            }),
            ..Default::default()
        };

        let mut trainer: EnhancedTrainer<f64> = EnhancedTrainer::new(config);

        // First call should set best value
        assert!(trainer.check_early_stopping(1.0));
        assert_eq!(trainer.state.best_metric_value, Some(1.0));
        assert_eq!(trainer.state.early_stopping_counter, 0);

        // Improvement should reset counter
        assert!(trainer.check_early_stopping(0.5));
        assert_eq!(trainer.state.best_metric_value, Some(0.5));
        assert_eq!(trainer.state.early_stopping_counter, 0);

        // No improvement should increment counter
        assert!(!trainer.check_early_stopping(0.5));
        assert_eq!(trainer.state.early_stopping_counter, 1);

        assert!(!trainer.check_early_stopping(0.6));
        assert_eq!(trainer.state.early_stopping_counter, 2);

        assert!(!trainer.check_early_stopping(0.7));
        assert_eq!(trainer.state.early_stopping_counter, 3);

        // Should trigger stop
        assert!(trainer.state.should_stop);
    }

    #[test]
    fn test_optimization_analyzer_recommendations() {
        let profiling = ProfilingResults::new();
        let config = EnhancedTrainingConfig {
            batch_size: 16,
            early_stopping: None,
            lr_warmup: None,
            ..Default::default()
        };

        let analyzer = OptimizationAnalyzer::new(
            profiling,
            config,
            vec![1.0, 0.9, 0.85, 0.8, 0.79, 0.79, 0.79],
            vec![1.0, 0.95, 0.92, 0.93, 0.95, 0.98, 1.0],
        );

        let recommendations = analyzer.generate_recommendations();

        // Should recommend early stopping due to overfitting pattern
        let has_early_stopping_rec = recommendations
            .iter()
            .any(|r| r.recommendation_type == RecommendationType::UseEarlyStopping);
        assert!(has_early_stopping_rec);

        // Should recommend gradient accumulation due to small batch size
        let has_grad_accum_rec = recommendations
            .iter()
            .any(|r| r.recommendation_type == RecommendationType::UseGradientAccumulation);
        assert!(has_grad_accum_rec);
    }
}
