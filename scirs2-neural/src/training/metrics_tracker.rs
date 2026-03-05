//! Training metrics tracker for neural network training
//!
//! Provides comprehensive metric recording at both epoch and batch levels,
//! with moving average smoothing, best metric tracking, and history export.
//!
//! # Features
//!
//! - **Epoch-level and batch-level recording**: Track metrics at different granularities
//! - **Moving average smoothing**: Exponential moving average (EMA) for noisy metrics
//! - **Best metric tracking**: Automatically track when the best value was achieved
//! - **History export**: Export to Rust structs suitable for serialization
//! - **Aggregation**: Mean, min, max, std-dev over windows
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::metrics_tracker::{MetricsTracker, MetricGoal};
//!
//! let mut tracker = MetricsTracker::new();
//! tracker.register_metric("loss", MetricGoal::Minimize);
//! tracker.register_metric("accuracy", MetricGoal::Maximize);
//!
//! for epoch in 0..10 {
//!     for batch in 0..100 {
//!         tracker.record_batch("loss", epoch, batch, 0.5);
//!     }
//!     tracker.end_epoch("loss", epoch);
//! }
//!
//! let history = tracker.export_history("loss");
//! ```

use std::collections::HashMap;
use std::fmt::Debug;

// ============================================================================
// Types
// ============================================================================

/// Whether the goal is to minimize or maximize a metric
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricGoal {
    /// Lower is better (e.g., loss, error rate)
    Minimize,
    /// Higher is better (e.g., accuracy, F1 score)
    Maximize,
}

/// A single metric entry recorded at a specific time
#[derive(Debug, Clone)]
pub struct MetricEntry {
    /// The raw metric value
    pub value: f64,
    /// The exponentially smoothed value (if smoothing is enabled)
    pub smoothed_value: f64,
    /// Epoch index
    pub epoch: usize,
    /// Batch index within the epoch (None for epoch-level entries)
    pub batch: Option<usize>,
}

/// Summary of the best metric achieved
#[derive(Debug, Clone)]
pub struct BestMetric {
    /// The best raw metric value
    pub value: f64,
    /// The epoch at which the best was achieved
    pub epoch: usize,
    /// The batch at which the best was achieved (if batch-level)
    pub batch: Option<usize>,
}

/// Exported history for a single metric
#[derive(Debug, Clone)]
pub struct MetricHistory {
    /// Name of the metric
    pub name: String,
    /// Goal direction
    pub goal: MetricGoal,
    /// Epoch-level values (one per epoch)
    pub epoch_values: Vec<f64>,
    /// Epoch-level smoothed values
    pub epoch_smoothed: Vec<f64>,
    /// Batch-level values (all batches across all epochs)
    pub batch_values: Vec<f64>,
    /// Batch-level smoothed values
    pub batch_smoothed: Vec<f64>,
    /// Best metric info
    pub best: Option<BestMetric>,
    /// Summary statistics
    pub stats: MetricStats,
}

/// Summary statistics for a metric
#[derive(Debug, Clone)]
pub struct MetricStats {
    /// Number of epoch-level entries
    pub num_epochs: usize,
    /// Total number of batch-level entries
    pub num_batches: usize,
    /// Mean of epoch-level values
    pub mean: f64,
    /// Standard deviation of epoch-level values
    pub std_dev: f64,
    /// Minimum epoch-level value
    pub min: f64,
    /// Maximum epoch-level value
    pub max: f64,
    /// Value of the last epoch
    pub last: f64,
}

/// Full training history export (all metrics)
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    /// All metric histories, keyed by name
    pub metrics: HashMap<String, MetricHistory>,
    /// Total number of epochs recorded
    pub total_epochs: usize,
}

// ============================================================================
// Internal per-metric state
// ============================================================================

#[derive(Debug, Clone)]
struct MetricState {
    goal: MetricGoal,
    smoothing_factor: f64,

    // Epoch-level data
    epoch_values: Vec<f64>,
    epoch_smoothed: Vec<f64>,
    epoch_ema: f64,
    epoch_ema_initialized: bool,

    // Batch-level data (within current epoch)
    batch_values_current_epoch: Vec<f64>,
    // All batch values across all epochs (for export)
    batch_values_all: Vec<f64>,
    batch_smoothed_all: Vec<f64>,
    batch_ema: f64,
    batch_ema_initialized: bool,

    // Best tracking
    best: Option<BestMetric>,
}

impl MetricState {
    fn new(goal: MetricGoal, smoothing_factor: f64) -> Self {
        Self {
            goal,
            smoothing_factor,
            epoch_values: Vec::new(),
            epoch_smoothed: Vec::new(),
            epoch_ema: 0.0,
            epoch_ema_initialized: false,
            batch_values_current_epoch: Vec::new(),
            batch_values_all: Vec::new(),
            batch_smoothed_all: Vec::new(),
            batch_ema: 0.0,
            batch_ema_initialized: false,
            best: None,
        }
    }

    fn update_ema(current_ema: &mut f64, initialized: &mut bool, value: f64, alpha: f64) -> f64 {
        if !*initialized {
            *current_ema = value;
            *initialized = true;
        } else {
            *current_ema = alpha * value + (1.0 - alpha) * *current_ema;
        }
        *current_ema
    }

    fn is_better(&self, new_val: f64, old_val: f64) -> bool {
        match self.goal {
            MetricGoal::Minimize => new_val < old_val,
            MetricGoal::Maximize => new_val > old_val,
        }
    }
}

// ============================================================================
// MetricsTracker
// ============================================================================

/// Comprehensive metrics tracker for neural network training.
///
/// Tracks metrics at both epoch and batch granularity with EMA smoothing
/// and automatic best-value tracking.
#[derive(Debug, Clone)]
pub struct MetricsTracker {
    /// Per-metric state, keyed by name
    metrics: HashMap<String, MetricState>,
    /// Default EMA smoothing factor (0.0 = no smoothing, 1.0 = only latest value)
    default_smoothing: f64,
    /// Total epochs completed
    total_epochs: usize,
}

impl MetricsTracker {
    /// Create a new metrics tracker with default EMA smoothing factor of 0.1.
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            default_smoothing: 0.1,
            total_epochs: 0,
        }
    }

    /// Create a new metrics tracker with a custom default smoothing factor.
    ///
    /// # Arguments
    ///
    /// * `smoothing` - EMA smoothing factor in `[0.0, 1.0]`.
    ///   A value of 0.0 means no smoothing (smoothed = first value always).
    ///   A value close to 1.0 means the smoothed value tracks the raw value closely.
    pub fn with_smoothing(smoothing: f64) -> Self {
        Self {
            metrics: HashMap::new(),
            default_smoothing: smoothing.clamp(0.0, 1.0),
            total_epochs: 0,
        }
    }

    /// Register a metric to be tracked.
    ///
    /// This must be called before recording values for a given metric name.
    /// If the metric is already registered, this is a no-op.
    pub fn register_metric(&mut self, name: &str, goal: MetricGoal) {
        self.metrics
            .entry(name.to_string())
            .or_insert_with(|| MetricState::new(goal, self.default_smoothing));
    }

    /// Register a metric with a custom smoothing factor.
    pub fn register_metric_with_smoothing(&mut self, name: &str, goal: MetricGoal, smoothing: f64) {
        self.metrics
            .entry(name.to_string())
            .or_insert_with(|| MetricState::new(goal, smoothing.clamp(0.0, 1.0)));
    }

    /// Record a batch-level metric value.
    ///
    /// # Arguments
    ///
    /// * `name` - Metric name (must be registered first)
    /// * `epoch` - Current epoch index
    /// * `batch` - Current batch index within the epoch
    /// * `value` - The metric value
    ///
    /// # Returns
    ///
    /// The smoothed value, or `None` if the metric is not registered.
    pub fn record_batch(
        &mut self,
        name: &str,
        epoch: usize,
        batch: usize,
        value: f64,
    ) -> Option<f64> {
        let state = self.metrics.get_mut(name)?;

        state.batch_values_current_epoch.push(value);
        state.batch_values_all.push(value);

        let smoothed = MetricState::update_ema(
            &mut state.batch_ema,
            &mut state.batch_ema_initialized,
            value,
            state.smoothing_factor,
        );
        state.batch_smoothed_all.push(smoothed);

        // Update best (batch-level)
        let is_new_best = match &state.best {
            None => true,
            Some(best) => state.is_better(value, best.value),
        };
        if is_new_best {
            state.best = Some(BestMetric {
                value,
                epoch,
                batch: Some(batch),
            });
        }

        Some(smoothed)
    }

    /// Record an epoch-level metric value.
    ///
    /// # Arguments
    ///
    /// * `name` - Metric name (must be registered first)
    /// * `epoch` - Current epoch index
    /// * `value` - The epoch-level metric value
    ///
    /// # Returns
    ///
    /// The smoothed value, or `None` if the metric is not registered.
    pub fn record_epoch(&mut self, name: &str, epoch: usize, value: f64) -> Option<f64> {
        let state = self.metrics.get_mut(name)?;

        state.epoch_values.push(value);

        let smoothed = MetricState::update_ema(
            &mut state.epoch_ema,
            &mut state.epoch_ema_initialized,
            value,
            state.smoothing_factor,
        );
        state.epoch_smoothed.push(smoothed);

        // Update best (epoch-level, only if no batch-level best or this is better)
        let is_new_best = match &state.best {
            None => true,
            Some(best) => state.is_better(value, best.value),
        };
        if is_new_best {
            state.best = Some(BestMetric {
                value,
                epoch,
                batch: None,
            });
        }

        Some(smoothed)
    }

    /// End an epoch: computes the mean of batch-level values for this epoch and
    /// records it as the epoch-level value. Resets the per-epoch batch accumulator.
    ///
    /// # Arguments
    ///
    /// * `name` - Metric name
    /// * `epoch` - The epoch that just finished
    ///
    /// # Returns
    ///
    /// The mean batch value for this epoch, or `None` if no batches were recorded
    /// or the metric is not registered.
    pub fn end_epoch(&mut self, name: &str, epoch: usize) -> Option<f64> {
        let state = self.metrics.get_mut(name)?;

        if state.batch_values_current_epoch.is_empty() {
            return None;
        }

        let sum: f64 = state.batch_values_current_epoch.iter().sum();
        let count = state.batch_values_current_epoch.len() as f64;
        let mean = sum / count;

        state.batch_values_current_epoch.clear();

        // Record as epoch-level value
        state.epoch_values.push(mean);
        let smoothed = MetricState::update_ema(
            &mut state.epoch_ema,
            &mut state.epoch_ema_initialized,
            mean,
            state.smoothing_factor,
        );
        state.epoch_smoothed.push(smoothed);

        self.total_epochs = self.total_epochs.max(epoch + 1);

        Some(mean)
    }

    /// Get the latest (most recent) raw value for a metric.
    pub fn latest_value(&self, name: &str) -> Option<f64> {
        let state = self.metrics.get(name)?;
        state
            .epoch_values
            .last()
            .or(state.batch_values_all.last())
            .copied()
    }

    /// Get the latest smoothed (EMA) value for a metric.
    pub fn latest_smoothed(&self, name: &str) -> Option<f64> {
        let state = self.metrics.get(name)?;
        state
            .epoch_smoothed
            .last()
            .or(state.batch_smoothed_all.last())
            .copied()
    }

    /// Get the best metric info for a named metric.
    pub fn best(&self, name: &str) -> Option<&BestMetric> {
        self.metrics.get(name)?.best.as_ref()
    }

    /// Get all epoch-level raw values for a metric.
    pub fn epoch_values(&self, name: &str) -> Option<&[f64]> {
        Some(self.metrics.get(name)?.epoch_values.as_slice())
    }

    /// Get all batch-level raw values for a metric.
    pub fn batch_values(&self, name: &str) -> Option<&[f64]> {
        Some(self.metrics.get(name)?.batch_values_all.as_slice())
    }

    /// Compute a rolling mean over a window of epoch-level values.
    ///
    /// # Arguments
    ///
    /// * `name` - Metric name
    /// * `window` - Number of recent epochs to average over
    ///
    /// # Returns
    ///
    /// The rolling mean, or `None` if not enough data.
    pub fn rolling_mean(&self, name: &str, window: usize) -> Option<f64> {
        let state = self.metrics.get(name)?;
        let vals = &state.epoch_values;
        if vals.len() < window || window == 0 {
            return None;
        }
        let sum: f64 = vals[vals.len() - window..].iter().sum();
        Some(sum / window as f64)
    }

    /// Get all registered metric names.
    pub fn metric_names(&self) -> Vec<&str> {
        self.metrics.keys().map(|s| s.as_str()).collect()
    }

    /// Export the full history for a single metric.
    pub fn export_history(&self, name: &str) -> Option<MetricHistory> {
        let state = self.metrics.get(name)?;
        let stats = compute_stats(&state.epoch_values);

        Some(MetricHistory {
            name: name.to_string(),
            goal: state.goal,
            epoch_values: state.epoch_values.clone(),
            epoch_smoothed: state.epoch_smoothed.clone(),
            batch_values: state.batch_values_all.clone(),
            batch_smoothed: state.batch_smoothed_all.clone(),
            best: state.best.clone(),
            stats,
        })
    }

    /// Export the full training history for all metrics.
    pub fn export_all(&self) -> TrainingHistory {
        let mut metrics = HashMap::new();
        for name in self.metrics.keys() {
            if let Some(history) = self.export_history(name) {
                metrics.insert(name.clone(), history);
            }
        }
        TrainingHistory {
            metrics,
            total_epochs: self.total_epochs,
        }
    }

    /// Clear all recorded data for all metrics while preserving registrations.
    pub fn clear(&mut self) {
        for state in self.metrics.values_mut() {
            state.epoch_values.clear();
            state.epoch_smoothed.clear();
            state.epoch_ema_initialized = false;
            state.batch_values_current_epoch.clear();
            state.batch_values_all.clear();
            state.batch_smoothed_all.clear();
            state.batch_ema_initialized = false;
            state.best = None;
        }
        self.total_epochs = 0;
    }

    /// Returns the total number of epochs recorded.
    pub fn total_epochs(&self) -> usize {
        self.total_epochs
    }
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper
// ============================================================================

fn compute_stats(values: &[f64]) -> MetricStats {
    if values.is_empty() {
        return MetricStats {
            num_epochs: 0,
            num_batches: 0,
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            last: 0.0,
        };
    }

    let n = values.len() as f64;
    let sum: f64 = values.iter().sum();
    let mean = sum / n;

    let var = if values.len() > 1 {
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    let std_dev = var.sqrt();

    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let last = values.last().copied().unwrap_or(0.0);

    MetricStats {
        num_epochs: values.len(),
        num_batches: 0, // filled by export_history caller
        mean,
        std_dev,
        min,
        max,
        last,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_record_epoch() {
        let mut tracker = MetricsTracker::new();
        tracker.register_metric("loss", MetricGoal::Minimize);

        let s0 = tracker.record_epoch("loss", 0, 1.0);
        assert!(s0.is_some());
        let s1 = tracker.record_epoch("loss", 1, 0.8);
        assert!(s1.is_some());

        assert_eq!(tracker.epoch_values("loss"), Some(&[1.0, 0.8][..]));
    }

    #[test]
    fn test_batch_recording_and_end_epoch() {
        let mut tracker = MetricsTracker::new();
        tracker.register_metric("loss", MetricGoal::Minimize);

        // Record 4 batches in epoch 0
        tracker.record_batch("loss", 0, 0, 1.0);
        tracker.record_batch("loss", 0, 1, 0.9);
        tracker.record_batch("loss", 0, 2, 0.8);
        tracker.record_batch("loss", 0, 3, 0.7);

        let mean = tracker.end_epoch("loss", 0);
        assert!(mean.is_some());
        let mean_val = mean.expect("should have mean");
        assert!((mean_val - 0.85).abs() < 1e-10);

        // Epoch-level should now have one entry
        assert_eq!(tracker.epoch_values("loss").map(|v| v.len()), Some(1));
    }

    #[test]
    fn test_best_tracking_minimize() {
        let mut tracker = MetricsTracker::new();
        tracker.register_metric("loss", MetricGoal::Minimize);

        tracker.record_epoch("loss", 0, 1.0);
        tracker.record_epoch("loss", 1, 0.5);
        tracker.record_epoch("loss", 2, 0.7);

        let best = tracker.best("loss").expect("should have best");
        assert!((best.value - 0.5).abs() < 1e-10);
        assert_eq!(best.epoch, 1);
    }

    #[test]
    fn test_best_tracking_maximize() {
        let mut tracker = MetricsTracker::new();
        tracker.register_metric("accuracy", MetricGoal::Maximize);

        tracker.record_epoch("accuracy", 0, 0.7);
        tracker.record_epoch("accuracy", 1, 0.9);
        tracker.record_epoch("accuracy", 2, 0.85);

        let best = tracker.best("accuracy").expect("should have best");
        assert!((best.value - 0.9).abs() < 1e-10);
        assert_eq!(best.epoch, 1);
    }

    #[test]
    fn test_smoothed_values() {
        let mut tracker = MetricsTracker::with_smoothing(0.5);
        tracker.register_metric("loss", MetricGoal::Minimize);

        // EMA with alpha=0.5:
        // step 0: EMA = 1.0 (initialization)
        // step 1: EMA = 0.5 * 0.5 + 0.5 * 1.0 = 0.75
        // step 2: EMA = 0.5 * 0.3 + 0.5 * 0.75 = 0.525
        tracker.record_epoch("loss", 0, 1.0);
        tracker.record_epoch("loss", 1, 0.5);
        tracker.record_epoch("loss", 2, 0.3);

        let smoothed = tracker
            .latest_smoothed("loss")
            .expect("should have smoothed");
        assert!((smoothed - 0.525).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_mean() {
        let mut tracker = MetricsTracker::new();
        tracker.register_metric("loss", MetricGoal::Minimize);

        for i in 0..10 {
            tracker.record_epoch("loss", i, i as f64);
        }

        // Last 3: 7, 8, 9 => mean = 8.0
        let rm = tracker.rolling_mean("loss", 3);
        assert!(rm.is_some());
        assert!((rm.expect("should have rolling mean") - 8.0).abs() < 1e-10);

        // Window larger than data
        let rm_big = tracker.rolling_mean("loss", 20);
        assert!(rm_big.is_none());
    }

    #[test]
    fn test_export_history() {
        let mut tracker = MetricsTracker::new();
        tracker.register_metric("loss", MetricGoal::Minimize);

        tracker.record_epoch("loss", 0, 1.0);
        tracker.record_epoch("loss", 1, 0.5);
        tracker.record_epoch("loss", 2, 0.8);

        let history = tracker.export_history("loss").expect("should export");
        assert_eq!(history.name, "loss");
        assert_eq!(history.goal, MetricGoal::Minimize);
        assert_eq!(history.epoch_values.len(), 3);
        assert_eq!(history.epoch_smoothed.len(), 3);
        assert!((history.stats.mean - (1.0 + 0.5 + 0.8) / 3.0).abs() < 1e-10);
        assert!((history.stats.min - 0.5).abs() < 1e-10);
        assert!((history.stats.max - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_export_all() {
        let mut tracker = MetricsTracker::new();
        tracker.register_metric("loss", MetricGoal::Minimize);
        tracker.register_metric("accuracy", MetricGoal::Maximize);

        tracker.record_epoch("loss", 0, 1.0);
        tracker.record_epoch("accuracy", 0, 0.5);

        let all = tracker.export_all();
        assert_eq!(all.metrics.len(), 2);
        assert!(all.metrics.contains_key("loss"));
        assert!(all.metrics.contains_key("accuracy"));
    }

    #[test]
    fn test_unregistered_metric_returns_none() {
        let mut tracker = MetricsTracker::new();
        assert!(tracker.record_epoch("bogus", 0, 1.0).is_none());
        assert!(tracker.record_batch("bogus", 0, 0, 1.0).is_none());
        assert!(tracker.latest_value("bogus").is_none());
        assert!(tracker.best("bogus").is_none());
        assert!(tracker.export_history("bogus").is_none());
    }

    #[test]
    fn test_clear_preserves_registrations() {
        let mut tracker = MetricsTracker::new();
        tracker.register_metric("loss", MetricGoal::Minimize);
        tracker.record_epoch("loss", 0, 1.0);

        tracker.clear();

        // Registration still exists
        assert!(tracker.record_epoch("loss", 0, 2.0).is_some());
        assert_eq!(tracker.epoch_values("loss"), Some(&[2.0][..]));
        assert!(tracker.best("loss").is_some());
    }

    #[test]
    fn test_multiple_epochs_batch_recording() {
        let mut tracker = MetricsTracker::new();
        tracker.register_metric("loss", MetricGoal::Minimize);

        // Epoch 0
        tracker.record_batch("loss", 0, 0, 1.0);
        tracker.record_batch("loss", 0, 1, 0.9);
        tracker.end_epoch("loss", 0);

        // Epoch 1
        tracker.record_batch("loss", 1, 0, 0.7);
        tracker.record_batch("loss", 1, 1, 0.6);
        tracker.end_epoch("loss", 1);

        // Should have 2 epoch values
        let epochs = tracker.epoch_values("loss").expect("should exist");
        assert_eq!(epochs.len(), 2);
        assert!((epochs[0] - 0.95).abs() < 1e-10);
        assert!((epochs[1] - 0.65).abs() < 1e-10);

        // 4 batch values total
        let batches = tracker.batch_values("loss").expect("should exist");
        assert_eq!(batches.len(), 4);
    }

    #[test]
    fn test_metric_stats_std_dev() {
        let mut tracker = MetricsTracker::new();
        tracker.register_metric("loss", MetricGoal::Minimize);

        // Values: 2, 4, 4, 4, 5, 5, 7, 9 => mean=5, variance=4.571..., std_dev=2.138
        for (i, &v) in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0].iter().enumerate() {
            tracker.record_epoch("loss", i, v);
        }

        let history = tracker.export_history("loss").expect("should export");
        let mean = (2.0 + 4.0 + 4.0 + 4.0 + 5.0 + 5.0 + 7.0 + 9.0) / 8.0;
        assert!((history.stats.mean - mean).abs() < 1e-10);
        // Sample std dev
        let var = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / 7.0;
        assert!((history.stats.std_dev - var.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_custom_smoothing_per_metric() {
        let mut tracker = MetricsTracker::new();
        tracker.register_metric_with_smoothing("loss", MetricGoal::Minimize, 1.0);
        // smoothing=1.0 means EMA tracks latest value exactly

        tracker.record_epoch("loss", 0, 5.0);
        tracker.record_epoch("loss", 1, 3.0);

        let smoothed = tracker.latest_smoothed("loss").expect("should exist");
        assert!((smoothed - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_latest_value_from_batch_when_no_epoch() {
        let mut tracker = MetricsTracker::new();
        tracker.register_metric("loss", MetricGoal::Minimize);

        tracker.record_batch("loss", 0, 0, 0.5);
        tracker.record_batch("loss", 0, 1, 0.4);

        // No epoch recorded, but batch values exist
        let latest = tracker.latest_value("loss").expect("should exist");
        assert!((latest - 0.4).abs() < 1e-10);
    }
}
