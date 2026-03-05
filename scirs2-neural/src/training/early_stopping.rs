//! Standalone early stopping for neural network training
//!
//! Provides a reusable `EarlyStopping` component that monitors a metric and
//! signals when training should stop because the metric has stopped improving.
//!
//! # Features
//!
//! - Configurable patience (number of epochs with no improvement before stopping)
//! - Configurable minimum delta (smallest change that counts as improvement)
//! - Mode selection: minimize or maximize the monitored metric
//! - Best metric value and epoch tracking
//! - Best model state snapshot (via generic parameter state storage)
//! - Warm-up period: ignore early epochs before enforcing stopping criteria
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::early_stopping::{EarlyStopping, StoppingMode};
//!
//! let mut stopper = EarlyStopping::new(
//!     5,          // patience
//!     0.001,      // min_delta
//!     StoppingMode::Min,
//! );
//!
//! // Simulate training epochs with decreasing then stagnating loss
//! let losses = [1.0, 0.8, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
//! for (epoch, &val_loss) in losses.iter().enumerate() {
//!     if stopper.check(val_loss) {
//!         println!("Early stopping at epoch {epoch}");
//!         break;
//!     }
//! }
//! println!("Best loss: {} at epoch {}", stopper.best_value(), stopper.best_epoch());
//! ```

use scirs2_core::ndarray::{ArrayD, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, ToPrimitive};
use std::fmt::Debug;

// ============================================================================
// Types
// ============================================================================

/// Mode for determining "improvement" direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoppingMode {
    /// Lower values are better (e.g., loss)
    Min,
    /// Higher values are better (e.g., accuracy)
    Max,
}

/// Reason why training was stopped
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// Patience exhausted (no improvement for `patience` consecutive checks)
    PatienceExhausted {
        /// Number of checks since last improvement
        checks_without_improvement: usize,
    },
    /// The metric diverged (became NaN or infinite)
    MetricDiverged,
    /// The metric exceeded a user-defined absolute threshold
    ThresholdExceeded {
        /// The threshold that was exceeded
        threshold: String,
    },
    /// Training has not been stopped
    NotStopped,
}

/// Status returned by `EarlyStopping::step`
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Whether training should stop now
    pub should_stop: bool,
    /// Whether this step was an improvement
    pub improved: bool,
    /// Current patience counter (checks since last improvement)
    pub patience_counter: usize,
    /// The reason for stopping (if should_stop is true)
    pub reason: StopReason,
}

// ============================================================================
// EarlyStopping
// ============================================================================

/// Standalone early stopping monitor.
///
/// Tracks a metric over training and decides when to stop based on configurable
/// patience, improvement thresholds, and mode (minimize vs maximize).
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    /// Number of checks with no improvement before stopping
    patience: usize,
    /// Minimum change to qualify as an improvement
    min_delta: f64,
    /// Whether we are minimizing or maximizing
    mode: StoppingMode,
    /// Number of initial checks to skip before enforcing stopping
    warmup_checks: usize,
    /// Optional absolute divergence threshold (stop if metric exceeds this)
    divergence_threshold: Option<f64>,

    // --- internal state ---
    /// Best metric value seen so far
    best_value: f64,
    /// Epoch/check index at which the best value was observed
    best_epoch: usize,
    /// Current number of checks since last improvement
    counter: usize,
    /// Total number of checks performed
    total_checks: usize,
    /// Whether stopping has been triggered
    stopped: bool,
    /// The reason for stopping
    stop_reason: StopReason,
    /// History of all metric values received
    metric_history: Vec<f64>,
}

impl EarlyStopping {
    /// Create a new `EarlyStopping` instance.
    ///
    /// # Arguments
    ///
    /// * `patience` - Number of checks with no improvement before triggering stop
    /// * `min_delta` - Minimum absolute change in the metric to count as improvement
    /// * `mode` - Whether to minimize or maximize the metric
    pub fn new(patience: usize, min_delta: f64, mode: StoppingMode) -> Self {
        let initial_best = match mode {
            StoppingMode::Min => f64::INFINITY,
            StoppingMode::Max => f64::NEG_INFINITY,
        };

        Self {
            patience,
            min_delta: min_delta.abs(), // always positive
            mode,
            warmup_checks: 0,
            divergence_threshold: None,
            best_value: initial_best,
            best_epoch: 0,
            counter: 0,
            total_checks: 0,
            stopped: false,
            stop_reason: StopReason::NotStopped,
            metric_history: Vec::new(),
        }
    }

    /// Set a warmup period during which early stopping is not enforced.
    ///
    /// Useful for models that need a few epochs to stabilize before meaningful
    /// metric comparisons can be made.
    pub fn with_warmup(mut self, warmup_checks: usize) -> Self {
        self.warmup_checks = warmup_checks;
        self
    }

    /// Set a divergence threshold.
    ///
    /// In `Min` mode, if the metric exceeds this threshold, training stops immediately.
    /// In `Max` mode, if the metric falls below the negative of this threshold,
    /// training stops immediately.
    pub fn with_divergence_threshold(mut self, threshold: f64) -> Self {
        self.divergence_threshold = Some(threshold);
        self
    }

    /// Check whether training should stop, given the latest metric value.
    ///
    /// This is the simple interface that returns a boolean.
    ///
    /// # Arguments
    ///
    /// * `metric` - The current metric value (e.g., validation loss)
    ///
    /// # Returns
    ///
    /// `true` if training should stop, `false` if training should continue.
    pub fn check(&mut self, metric: f64) -> bool {
        self.step(metric).should_stop
    }

    /// Perform a step with the latest metric value, returning detailed results.
    ///
    /// # Arguments
    ///
    /// * `metric` - The current metric value
    ///
    /// # Returns
    ///
    /// A `StepResult` with detailed information about the decision.
    pub fn step(&mut self, metric: f64) -> StepResult {
        self.total_checks += 1;
        self.metric_history.push(metric);

        // Already stopped -- keep returning stopped
        if self.stopped {
            return StepResult {
                should_stop: true,
                improved: false,
                patience_counter: self.counter,
                reason: self.stop_reason.clone(),
            };
        }

        // Check for NaN / infinity
        if !metric.is_finite() {
            self.stopped = true;
            self.stop_reason = StopReason::MetricDiverged;
            return StepResult {
                should_stop: true,
                improved: false,
                patience_counter: self.counter,
                reason: StopReason::MetricDiverged,
            };
        }

        // Check divergence threshold
        if let Some(threshold) = self.divergence_threshold {
            let diverged = match self.mode {
                StoppingMode::Min => metric > threshold,
                StoppingMode::Max => metric < -threshold,
            };
            if diverged {
                self.stopped = true;
                self.stop_reason = StopReason::ThresholdExceeded {
                    threshold: format!("{threshold}"),
                };
                return StepResult {
                    should_stop: true,
                    improved: false,
                    patience_counter: self.counter,
                    reason: self.stop_reason.clone(),
                };
            }
        }

        // Check if this is an improvement
        let improved = self.is_improvement(metric);

        if improved {
            self.best_value = metric;
            self.best_epoch = self.total_checks - 1; // 0-indexed
            self.counter = 0;
        } else {
            self.counter += 1;
        }

        // During warmup, never trigger stopping
        if self.total_checks <= self.warmup_checks {
            return StepResult {
                should_stop: false,
                improved,
                patience_counter: self.counter,
                reason: StopReason::NotStopped,
            };
        }

        // Check patience
        if self.counter >= self.patience {
            self.stopped = true;
            self.stop_reason = StopReason::PatienceExhausted {
                checks_without_improvement: self.counter,
            };
            return StepResult {
                should_stop: true,
                improved: false,
                patience_counter: self.counter,
                reason: self.stop_reason.clone(),
            };
        }

        StepResult {
            should_stop: false,
            improved,
            patience_counter: self.counter,
            reason: StopReason::NotStopped,
        }
    }

    /// Returns whether the given metric value represents an improvement.
    fn is_improvement(&self, metric: f64) -> bool {
        match self.mode {
            StoppingMode::Min => metric < self.best_value - self.min_delta,
            StoppingMode::Max => metric > self.best_value + self.min_delta,
        }
    }

    /// Get the best metric value observed so far.
    pub fn best_value(&self) -> f64 {
        self.best_value
    }

    /// Get the epoch/check index at which the best value was observed (0-indexed).
    pub fn best_epoch(&self) -> usize {
        self.best_epoch
    }

    /// Get the current patience counter (checks since last improvement).
    pub fn patience_counter(&self) -> usize {
        self.counter
    }

    /// Get the total number of checks performed.
    pub fn total_checks(&self) -> usize {
        self.total_checks
    }

    /// Get the complete history of metric values.
    pub fn metric_history(&self) -> &[f64] {
        &self.metric_history
    }

    /// Check whether stopping has been triggered.
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Get the stop reason.
    pub fn stop_reason(&self) -> &StopReason {
        &self.stop_reason
    }

    /// Reset the early stopping state, allowing training to continue.
    pub fn reset(&mut self) {
        let initial_best = match self.mode {
            StoppingMode::Min => f64::INFINITY,
            StoppingMode::Max => f64::NEG_INFINITY,
        };
        self.best_value = initial_best;
        self.best_epoch = 0;
        self.counter = 0;
        self.total_checks = 0;
        self.stopped = false;
        self.stop_reason = StopReason::NotStopped;
        self.metric_history.clear();
    }
}

// ============================================================================
// EarlyStoppingWithState -- stores best model parameters
// ============================================================================

/// Early stopping that also stores a snapshot of the best model state.
///
/// This is useful when you want to restore the model to the best-performing
/// state after training completes.
#[derive(Debug)]
pub struct EarlyStoppingWithState<F>
where
    F: Float + Debug + ScalarOperand + FromPrimitive + ToPrimitive + Clone,
{
    /// The core early stopping logic
    inner: EarlyStopping,
    /// Snapshot of the best parameter state
    best_params: Option<Vec<ArrayD<F>>>,
}

impl<F> EarlyStoppingWithState<F>
where
    F: Float + Debug + ScalarOperand + FromPrimitive + ToPrimitive + Clone,
{
    /// Create a new instance with the given early stopping configuration.
    pub fn new(patience: usize, min_delta: f64, mode: StoppingMode) -> Self {
        Self {
            inner: EarlyStopping::new(patience, min_delta, mode),
            best_params: None,
        }
    }

    /// Set a warmup period.
    pub fn with_warmup(mut self, warmup_checks: usize) -> Self {
        self.inner = self.inner.with_warmup(warmup_checks);
        self
    }

    /// Set a divergence threshold.
    pub fn with_divergence_threshold(mut self, threshold: f64) -> Self {
        self.inner = self.inner.with_divergence_threshold(threshold);
        self
    }

    /// Check the metric and, if improved, snapshot the current parameters.
    ///
    /// # Arguments
    ///
    /// * `metric` - The current metric value
    /// * `params` - The current model parameters to snapshot if improved
    ///
    /// # Returns
    ///
    /// A `StepResult` indicating whether training should stop.
    pub fn step(&mut self, metric: f64, params: &[ArrayD<F>]) -> StepResult {
        let result = self.inner.step(metric);
        if result.improved {
            self.best_params = Some(params.to_vec());
        }
        result
    }

    /// Simple check interface (does NOT snapshot parameters -- use `step` for that).
    pub fn check(&mut self, metric: f64) -> bool {
        self.inner.check(metric)
    }

    /// Get the stored best parameters, if any.
    pub fn best_params(&self) -> Option<&[ArrayD<F>]> {
        self.best_params.as_deref()
    }

    /// Get the best metric value.
    pub fn best_value(&self) -> f64 {
        self.inner.best_value()
    }

    /// Get the epoch of the best metric.
    pub fn best_epoch(&self) -> usize {
        self.inner.best_epoch()
    }

    /// Check whether stopping has been triggered.
    pub fn is_stopped(&self) -> bool {
        self.inner.is_stopped()
    }

    /// Get the stop reason.
    pub fn stop_reason(&self) -> &StopReason {
        self.inner.stop_reason()
    }

    /// Get the complete metric history.
    pub fn metric_history(&self) -> &[f64] {
        self.inner.metric_history()
    }

    /// Reset the state (clears best params too).
    pub fn reset(&mut self) {
        self.inner.reset();
        self.best_params = None;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    // ------- EarlyStopping (no state) -------

    #[test]
    fn test_min_mode_basic_improvement() {
        let mut es = EarlyStopping::new(3, 0.0, StoppingMode::Min);
        // Monotonically decreasing loss -- should never trigger stop
        for &loss in &[1.0, 0.9, 0.8, 0.7, 0.6, 0.5] {
            assert!(!es.check(loss));
        }
        assert_eq!(es.best_epoch(), 5);
        assert!((es.best_value() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_min_mode_patience_exhausted() {
        let mut es = EarlyStopping::new(3, 0.0, StoppingMode::Min);
        // Best = 0.5, then no improvement for 3 checks
        assert!(!es.check(0.5)); // improvement (from +inf)
        assert!(!es.check(0.6)); // counter=1
        assert!(!es.check(0.7)); // counter=2
        assert!(es.check(0.8)); // counter=3 => stop
        assert_eq!(es.patience_counter(), 3);
    }

    #[test]
    fn test_max_mode_basic() {
        let mut es = EarlyStopping::new(2, 0.0, StoppingMode::Max);
        assert!(!es.check(0.5)); // improve from -inf
        assert!(!es.check(0.7)); // improve
        assert!(!es.check(0.6)); // no improve, counter=1
        assert!(es.check(0.6)); // no improve, counter=2 => stop

        assert_eq!(es.best_epoch(), 1);
        assert!((es.best_value() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_min_delta_effect() {
        let mut es = EarlyStopping::new(3, 0.01, StoppingMode::Min);
        assert!(!es.check(1.0));
        // 0.995 is an improvement by 0.005, less than min_delta=0.01, so NOT counted
        assert!(!es.check(0.995)); // counter=1
        assert!(!es.check(0.996)); // counter=2
        assert!(es.check(0.997)); // counter=3 => stop

        // But best value is still 1.0 because 0.995 didn't meet the delta threshold
        assert!((es.best_value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nan_causes_divergence_stop() {
        let mut es = EarlyStopping::new(10, 0.0, StoppingMode::Min);
        assert!(!es.check(0.5));
        let result = es.step(f64::NAN);
        assert!(result.should_stop);
        assert_eq!(es.stop_reason(), &StopReason::MetricDiverged);
    }

    #[test]
    fn test_infinity_causes_divergence_stop() {
        let mut es = EarlyStopping::new(10, 0.0, StoppingMode::Min);
        assert!(!es.check(0.5));
        assert!(es.check(f64::INFINITY));
        assert_eq!(es.stop_reason(), &StopReason::MetricDiverged);
    }

    #[test]
    fn test_divergence_threshold() {
        let mut es = EarlyStopping::new(10, 0.0, StoppingMode::Min).with_divergence_threshold(5.0);

        assert!(!es.check(1.0));
        assert!(!es.check(4.0));
        let result = es.step(6.0); // exceeds threshold
        assert!(result.should_stop);
        match es.stop_reason() {
            StopReason::ThresholdExceeded { .. } => {} // expected
            other => panic!("Expected ThresholdExceeded, got {:?}", other),
        }
    }

    #[test]
    fn test_warmup_period() {
        let mut es = EarlyStopping::new(2, 0.0, StoppingMode::Min).with_warmup(3);

        // During warmup (checks 1..3), no stopping even with no improvement
        assert!(!es.check(0.5)); // check 1 (warmup)
        assert!(!es.check(0.6)); // check 2 (warmup), counter=1
        assert!(!es.check(0.7)); // check 3 (warmup), counter=2 -- but still in warmup

        // After warmup, patience still active from before
        // counter is already at 2, and patience is 2, so this should trigger
        // Actually, counter was 2, now check 4 => 0.8, counter=3 => patience=2 exceeded
        assert!(es.check(0.8)); // check 4 (post-warmup), counter=3 >= patience=2
    }

    #[test]
    fn test_reset() {
        let mut es = EarlyStopping::new(2, 0.0, StoppingMode::Min);
        assert!(!es.check(0.5));
        assert!(!es.check(0.6));
        assert!(es.check(0.7));
        assert!(es.is_stopped());

        es.reset();
        assert!(!es.is_stopped());
        assert_eq!(es.total_checks(), 0);
        assert!(es.metric_history().is_empty());
    }

    #[test]
    fn test_step_result_details() {
        let mut es = EarlyStopping::new(3, 0.0, StoppingMode::Min);
        let r1 = es.step(1.0);
        assert!(r1.improved);
        assert!(!r1.should_stop);
        assert_eq!(r1.patience_counter, 0);

        let r2 = es.step(1.5);
        assert!(!r2.improved);
        assert_eq!(r2.patience_counter, 1);
    }

    #[test]
    fn test_metric_history_tracking() {
        let mut es = EarlyStopping::new(5, 0.0, StoppingMode::Min);
        for &v in &[0.9, 0.8, 0.85, 0.7, 0.75] {
            es.check(v);
        }
        assert_eq!(es.metric_history(), &[0.9, 0.8, 0.85, 0.7, 0.75]);
        assert_eq!(es.total_checks(), 5);
    }

    // ------- EarlyStoppingWithState -------

    #[test]
    fn test_with_state_stores_best_params() {
        let mut es = EarlyStoppingWithState::<f64>::new(3, 0.0, StoppingMode::Min);

        let params1 = vec![Array::from_vec(vec![1.0, 2.0]).into_dyn()];
        let params2 = vec![Array::from_vec(vec![3.0, 4.0]).into_dyn()];
        let params3 = vec![Array::from_vec(vec![5.0, 6.0]).into_dyn()];

        let r1 = es.step(1.0, &params1); // improvement
        assert!(r1.improved);
        assert!(es.best_params().is_some());

        let r2 = es.step(0.5, &params2); // improvement -- updates stored params
        assert!(r2.improved);
        let best = es.best_params().expect("should have best params");
        assert_eq!(best[0].as_slice().expect("contiguous"), &[3.0, 4.0]);

        let r3 = es.step(0.8, &params3); // no improvement -- keeps params2
        assert!(!r3.improved);
        let best = es.best_params().expect("should still have best params");
        assert_eq!(best[0].as_slice().expect("contiguous"), &[3.0, 4.0]);
    }

    #[test]
    fn test_with_state_reset_clears_params() {
        let mut es = EarlyStoppingWithState::<f64>::new(3, 0.0, StoppingMode::Min);
        let params = vec![Array::from_vec(vec![1.0]).into_dyn()];
        es.step(0.5, &params);
        assert!(es.best_params().is_some());

        es.reset();
        assert!(es.best_params().is_none());
        assert!(!es.is_stopped());
    }

    #[test]
    fn test_with_state_stopping_behavior() {
        let mut es = EarlyStoppingWithState::<f64>::new(2, 0.0, StoppingMode::Max);
        let params = vec![Array::from_vec(vec![0.0]).into_dyn()];

        let r1 = es.step(0.8, &params); // improve
        assert!(!r1.should_stop);
        let r2 = es.step(0.7, &params); // no improve, counter=1
        assert!(!r2.should_stop);
        let r3 = es.step(0.6, &params); // no improve, counter=2 => stop
        assert!(r3.should_stop);
        assert!(es.is_stopped());
    }
}
