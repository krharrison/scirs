//! Learning Rate Finder — Leslie Smith's LR Range Test
//!
//! Implements the learning rate range test described in:
//! "Cyclical Learning Rates for Training Neural Networks" (Smith, 2017).
//!
//! The idea is to train the network for a short period while increasing the
//! learning rate from a very small value to a very large one, recording the
//! loss at each step. The optimal learning rate is found at the point of
//! steepest loss decrease.
//!
//! # Features
//!
//! - **Exponential and linear LR schedules** during the range test
//! - **Loss smoothing** via exponential moving average (EMA)
//! - **Automatic suggestion** of optimal LR (steepest loss decrease)
//! - **Divergence detection** to stop the test early
//! - **Configurable**: min/max LR, number of iterations, smoothing factor,
//!   divergence threshold
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::lr_finder::{
//!     LRFinderConfig, LRFinderResult, LRScheduleType, find_optimal_lr,
//! };
//!
//! // Configure the LR range test
//! let config = LRFinderConfig::builder()
//!     .min_lr(1e-7)
//!     .max_lr(10.0)
//!     .num_iterations(100)
//!     .schedule(LRScheduleType::Exponential)
//!     .smoothing_factor(0.98)
//!     .divergence_threshold(5.0)
//!     .build()
//!     .expect("invalid config");
//!
//! // Simulate a loss curve for testing
//! let losses: Vec<f64> = (0..100).map(|i| {
//!     let lr = config.lr_at_step(i);
//!     // Simulate: loss decreases then explodes
//!     if lr < 0.01 { 1.0 - lr * 10.0 } else { 1.0 + (lr - 0.01).powi(2) * 1000.0 }
//! }).collect();
//!
//! let result = LRFinderResult::from_losses(&losses, &config);
//! let suggested = result.suggested_lr();
//! assert!(suggested.is_some());
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use std::fmt::{self, Debug, Display};

// ============================================================================
// Schedule type
// ============================================================================

/// Learning rate schedule used during the range test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LRScheduleType {
    /// Exponential increase: `lr = min_lr * (max_lr / min_lr)^(step / total_steps)`.
    /// This gives more resolution at lower LRs, which is usually where the
    /// interesting behaviour lives.
    Exponential,
    /// Linear increase: `lr = min_lr + (max_lr - min_lr) * (step / total_steps)`.
    Linear,
}

impl Display for LRScheduleType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exponential => write!(f, "Exponential"),
            Self::Linear => write!(f, "Linear"),
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the learning rate range test.
#[derive(Debug, Clone)]
pub struct LRFinderConfig {
    /// Minimum (starting) learning rate.
    pub min_lr: f64,
    /// Maximum (ending) learning rate.
    pub max_lr: f64,
    /// Number of iterations (mini-batch steps) to run.
    pub num_iterations: usize,
    /// Type of LR schedule during the test.
    pub schedule: LRScheduleType,
    /// Exponential moving average smoothing factor (beta/momentum).
    /// A value of 0.0 means no smoothing (raw loss);
    /// 0.98 means heavy smoothing (only 2% weight on new values).
    /// Recommended: 0.98.
    pub smoothing_factor: f64,
    /// Stop the test if the smoothed loss exceeds `divergence_threshold *
    /// best_loss`. Set to `f64::INFINITY` to disable.
    pub divergence_threshold: f64,
    /// Whether to use gradient accumulation steps (for very large models).
    pub accumulation_steps: usize,
}

impl Default for LRFinderConfig {
    fn default() -> Self {
        Self {
            min_lr: 1e-7,
            max_lr: 10.0,
            num_iterations: 100,
            schedule: LRScheduleType::Exponential,
            smoothing_factor: 0.98,
            divergence_threshold: 5.0,
            accumulation_steps: 1,
        }
    }
}

impl LRFinderConfig {
    /// Create a builder for `LRFinderConfig`.
    pub fn builder() -> LRFinderConfigBuilder {
        LRFinderConfigBuilder::default()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.min_lr <= 0.0 {
            return Err(NeuralError::InvalidArgument(
                "min_lr must be positive".to_string(),
            ));
        }
        if self.max_lr <= self.min_lr {
            return Err(NeuralError::InvalidArgument(
                "max_lr must be greater than min_lr".to_string(),
            ));
        }
        if self.num_iterations == 0 {
            return Err(NeuralError::InvalidArgument(
                "num_iterations must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.smoothing_factor) {
            return Err(NeuralError::InvalidArgument(
                "smoothing_factor must be in [0.0, 1.0]".to_string(),
            ));
        }
        if self.divergence_threshold <= 1.0 {
            return Err(NeuralError::InvalidArgument(
                "divergence_threshold must be > 1.0".to_string(),
            ));
        }
        if self.accumulation_steps == 0 {
            return Err(NeuralError::InvalidArgument(
                "accumulation_steps must be positive".to_string(),
            ));
        }
        Ok(())
    }

    /// Compute the learning rate at a given step index.
    pub fn lr_at_step(&self, step: usize) -> f64 {
        let total = self.num_iterations.max(1) as f64;
        let t = (step as f64) / total;
        match self.schedule {
            LRScheduleType::Exponential => {
                // lr = min_lr * (max_lr / min_lr)^t
                self.min_lr * (self.max_lr / self.min_lr).powf(t)
            }
            LRScheduleType::Linear => self.min_lr + (self.max_lr - self.min_lr) * t,
        }
    }
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for [`LRFinderConfig`].
#[derive(Debug, Clone, Default)]
pub struct LRFinderConfigBuilder {
    config: LRFinderConfig,
}

impl LRFinderConfigBuilder {
    /// Set the minimum (starting) learning rate.
    pub fn min_lr(mut self, lr: f64) -> Self {
        self.config.min_lr = lr;
        self
    }

    /// Set the maximum (ending) learning rate.
    pub fn max_lr(mut self, lr: f64) -> Self {
        self.config.max_lr = lr;
        self
    }

    /// Set the number of iterations.
    pub fn num_iterations(mut self, n: usize) -> Self {
        self.config.num_iterations = n;
        self
    }

    /// Set the LR schedule type.
    pub fn schedule(mut self, s: LRScheduleType) -> Self {
        self.config.schedule = s;
        self
    }

    /// Set the EMA smoothing factor.
    pub fn smoothing_factor(mut self, f: f64) -> Self {
        self.config.smoothing_factor = f;
        self
    }

    /// Set the divergence threshold.
    pub fn divergence_threshold(mut self, t: f64) -> Self {
        self.config.divergence_threshold = t;
        self
    }

    /// Set the gradient accumulation steps.
    pub fn accumulation_steps(mut self, n: usize) -> Self {
        self.config.accumulation_steps = n;
        self
    }

    /// Build the configuration, validating all parameters.
    pub fn build(self) -> Result<LRFinderConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

// ============================================================================
// LR Finder Result
// ============================================================================

/// A single recorded point during the LR range test.
#[derive(Debug, Clone)]
pub struct LRFinderPoint {
    /// The step index.
    pub step: usize,
    /// The learning rate used at this step.
    pub lr: f64,
    /// The raw (unsmoothed) loss.
    pub raw_loss: f64,
    /// The exponentially smoothed loss.
    pub smoothed_loss: f64,
    /// The loss gradient (finite difference of smoothed loss w.r.t. log(lr)).
    pub loss_gradient: Option<f64>,
}

/// Result of a learning rate range test.
#[derive(Debug, Clone)]
pub struct LRFinderResult {
    /// All recorded points during the test.
    pub points: Vec<LRFinderPoint>,
    /// Whether the test was stopped early due to divergence.
    pub diverged: bool,
    /// The step at which divergence was detected (if any).
    pub divergence_step: Option<usize>,
    /// The best (lowest) smoothed loss observed.
    pub best_loss: f64,
    /// The learning rate at which the best loss was observed.
    pub best_loss_lr: f64,
    /// The configuration used for the test.
    pub config: LRFinderConfig,
}

impl LRFinderResult {
    /// Construct a result from a vector of raw losses and a config.
    ///
    /// This applies EMA smoothing, computes loss gradients w.r.t. log(lr),
    /// and detects divergence.
    pub fn from_losses(losses: &[f64], config: &LRFinderConfig) -> Self {
        let beta = config.smoothing_factor; // beta = momentum (0 = no smoothing)
        let mut smoothed = 0.0_f64;
        let mut best_loss = f64::MAX;
        let mut best_loss_lr = config.min_lr;
        let mut points = Vec::with_capacity(losses.len());
        let mut diverged = false;
        let mut divergence_step = None;

        for (i, &raw_loss) in losses.iter().enumerate() {
            let lr = config.lr_at_step(i);

            // Exponential moving average: smoothed = beta * smoothed + (1 - beta) * raw
            // With bias correction for initial steps.
            // When beta = 0.0, smoothed = raw_loss (no smoothing).
            if i == 0 {
                smoothed = raw_loss;
            } else {
                smoothed = beta * smoothed + (1.0 - beta) * raw_loss;
            }
            // Bias correction (Adam-style)
            let correction = 1.0 - beta.powi((i + 1) as i32);
            let corrected = if correction.abs() > 1e-15 {
                smoothed / correction
            } else {
                raw_loss // beta ~= 1.0 edge case, just use raw
            };

            if corrected < best_loss {
                best_loss = corrected;
                best_loss_lr = lr;
            }

            // Divergence check
            if !diverged && best_loss > 0.0 && corrected > best_loss * config.divergence_threshold {
                diverged = true;
                divergence_step = Some(i);
            }

            points.push(LRFinderPoint {
                step: i,
                lr,
                raw_loss,
                smoothed_loss: corrected,
                loss_gradient: None,
            });

            // Stop recording after divergence (we still have the point)
            if diverged {
                break;
            }
        }

        // Compute loss gradients: d(smoothed_loss) / d(log(lr))
        // Using finite differences on consecutive points
        if points.len() >= 2 {
            for i in 1..points.len() {
                let log_lr_curr = points[i].lr.ln();
                let log_lr_prev = points[i - 1].lr.ln();
                let d_log_lr = log_lr_curr - log_lr_prev;
                if d_log_lr.abs() > f64::EPSILON {
                    let d_loss = points[i].smoothed_loss - points[i - 1].smoothed_loss;
                    points[i].loss_gradient = Some(d_loss / d_log_lr);
                }
            }
        }

        Self {
            points,
            diverged,
            divergence_step,
            best_loss,
            best_loss_lr,
            config: config.clone(),
        }
    }

    /// Suggest the optimal learning rate.
    ///
    /// The optimal LR is found at the point of steepest loss decrease, i.e.,
    /// the point where the loss gradient is most negative. We look at the
    /// smoothed gradient curve and pick the minimum.
    ///
    /// Returns `None` if no gradient data is available.
    pub fn suggested_lr(&self) -> Option<f64> {
        let mut min_gradient = f64::MAX;
        let mut best_lr = None;

        for point in &self.points {
            if let Some(grad) = point.loss_gradient {
                if grad < min_gradient {
                    min_gradient = grad;
                    best_lr = Some(point.lr);
                }
            }
        }

        best_lr
    }

    /// Suggest the optimal LR using a conservative strategy.
    ///
    /// Returns the LR one order of magnitude smaller than the one at
    /// minimum loss, which is a common heuristic for the initial LR.
    pub fn suggested_lr_conservative(&self) -> f64 {
        self.best_loss_lr / 10.0
    }

    /// Get the learning rates as a vector.
    pub fn learning_rates(&self) -> Vec<f64> {
        self.points.iter().map(|p| p.lr).collect()
    }

    /// Get the raw losses as a vector.
    pub fn raw_losses(&self) -> Vec<f64> {
        self.points.iter().map(|p| p.raw_loss).collect()
    }

    /// Get the smoothed losses as a vector.
    pub fn smoothed_losses(&self) -> Vec<f64> {
        self.points.iter().map(|p| p.smoothed_loss).collect()
    }

    /// Get the loss gradients as a vector (None entries become NaN).
    pub fn loss_gradients(&self) -> Vec<f64> {
        self.points
            .iter()
            .map(|p| p.loss_gradient.unwrap_or(f64::NAN))
            .collect()
    }

    /// Generate a text summary of the LR range test.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Learning Rate Range Test Summary ===\n");
        out.push_str(&format!("Schedule: {}\n", self.config.schedule));
        out.push_str(&format!(
            "LR range: [{:.2e}, {:.2e}]\n",
            self.config.min_lr, self.config.max_lr
        ));
        out.push_str(&format!("Iterations: {}\n", self.points.len()));
        out.push_str(&format!(
            "Best loss: {:.6} at lr={:.2e}\n",
            self.best_loss, self.best_loss_lr
        ));
        if self.diverged {
            out.push_str(&format!(
                "Diverged at step {} (lr={:.2e})\n",
                self.divergence_step.unwrap_or(0),
                self.points
                    .last()
                    .map(|p| p.lr)
                    .unwrap_or(self.config.max_lr)
            ));
        }
        if let Some(lr) = self.suggested_lr() {
            out.push_str(&format!("Suggested LR (steepest decrease): {lr:.2e}\n"));
        }
        out.push_str(&format!(
            "Suggested LR (conservative): {:.2e}\n",
            self.suggested_lr_conservative()
        ));
        out
    }
}

// ============================================================================
// LR Finder (stateful, for use in a training loop)
// ============================================================================

/// Stateful learning rate finder that can be used in a training loop.
///
/// Usage:
/// 1. Create with `LRFinder::new(config)`
/// 2. For each training step, call `next_lr()` to get the LR, train, then
///    call `record_loss(loss)`
/// 3. When `is_finished()` returns true (or `record_loss` returns
///    `LRFinderStatus::Diverged`), call `result()` to get the analysis.
#[derive(Debug, Clone)]
pub struct LRFinder {
    config: LRFinderConfig,
    step: usize,
    raw_losses: Vec<f64>,
    finished: bool,
    diverged: bool,
    ema_loss: f64,
    best_loss: f64,
}

/// Status returned by `LRFinder::record_loss`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LRFinderStatus {
    /// Continue training — more steps needed.
    Continue,
    /// The test is complete (all iterations done).
    Complete,
    /// The loss diverged; stop training.
    Diverged,
}

impl LRFinder {
    /// Create a new LR finder with the given configuration.
    pub fn new(config: LRFinderConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            step: 0,
            raw_losses: Vec::new(),
            finished: false,
            diverged: false,
            ema_loss: 0.0,
            best_loss: f64::MAX,
        })
    }

    /// Get the learning rate for the current step.
    ///
    /// Returns `None` if the test is already finished.
    pub fn next_lr(&self) -> Option<f64> {
        if self.finished {
            return None;
        }
        Some(self.config.lr_at_step(self.step))
    }

    /// Record the loss for the current step and advance.
    ///
    /// Returns the status of the finder after recording.
    pub fn record_loss(&mut self, loss: f64) -> LRFinderStatus {
        if self.finished {
            return if self.diverged {
                LRFinderStatus::Diverged
            } else {
                LRFinderStatus::Complete
            };
        }

        self.raw_losses.push(loss);

        // EMA smoothing: beta = smoothing_factor (momentum)
        let beta = self.config.smoothing_factor;
        if self.step == 0 {
            self.ema_loss = loss;
        } else {
            self.ema_loss = beta * self.ema_loss + (1.0 - beta) * loss;
        }

        // Bias correction
        let correction = 1.0 - beta.powi((self.step + 1) as i32);
        let corrected = if correction.abs() > 1e-15 {
            self.ema_loss / correction
        } else {
            loss // beta ~= 1.0, use raw
        };

        if corrected < self.best_loss {
            self.best_loss = corrected;
        }

        // Divergence check
        if self.best_loss > 0.0 && corrected > self.best_loss * self.config.divergence_threshold {
            self.finished = true;
            self.diverged = true;
            return LRFinderStatus::Diverged;
        }

        self.step += 1;

        if self.step >= self.config.num_iterations {
            self.finished = true;
            return LRFinderStatus::Complete;
        }

        LRFinderStatus::Continue
    }

    /// Whether the test has finished.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Whether the test diverged.
    pub fn is_diverged(&self) -> bool {
        self.diverged
    }

    /// Get the current step index.
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Get the total number of iterations configured.
    pub fn total_iterations(&self) -> usize {
        self.config.num_iterations
    }

    /// Produce the final result. Can be called at any time, but is most
    /// useful after the test has finished.
    pub fn result(&self) -> LRFinderResult {
        LRFinderResult::from_losses(&self.raw_losses, &self.config)
    }

    /// Reset the finder to start a new test (with the same config).
    pub fn reset(&mut self) {
        self.step = 0;
        self.raw_losses.clear();
        self.finished = false;
        self.diverged = false;
        self.ema_loss = 0.0;
        self.best_loss = f64::MAX;
    }
}

// ============================================================================
// Convenience function
// ============================================================================

/// Convenience function: given a pre-recorded vector of losses (one per
/// iteration), build the result and suggest an optimal learning rate.
///
/// Returns `(suggested_lr, result)`.
pub fn find_optimal_lr(
    losses: &[f64],
    config: &LRFinderConfig,
) -> Result<(Option<f64>, LRFinderResult)> {
    config.validate()?;
    let result = LRFinderResult::from_losses(losses, config);
    let lr = result.suggested_lr();
    Ok((lr, result))
}

// ============================================================================
// Generic LR finder for typed models
// ============================================================================

/// A generic learning rate finder that works with typed parameters.
///
/// This is the generic version that accepts `Array<F, IxDyn>` parameters
/// and can be integrated into a typed training loop.
pub struct TypedLRFinder<F: Float + Debug + ScalarOperand + NumAssign> {
    /// The inner LR finder.
    inner: LRFinder,
    /// Original parameters snapshot (for restoring after the test).
    original_params: Option<Vec<Array<F, IxDyn>>>,
}

impl<F: Float + Debug + ScalarOperand + NumAssign> TypedLRFinder<F> {
    /// Create a new typed LR finder.
    pub fn new(config: LRFinderConfig) -> Result<Self> {
        Ok(Self {
            inner: LRFinder::new(config)?,
            original_params: None,
        })
    }

    /// Snapshot the current model parameters so they can be restored after the test.
    pub fn save_params(&mut self, params: &[Array<F, IxDyn>]) {
        self.original_params = Some(params.to_vec());
    }

    /// Get the saved original parameters (for restoring after the test).
    pub fn original_params(&self) -> Option<&[Array<F, IxDyn>]> {
        self.original_params.as_deref()
    }

    /// Get the next learning rate as the model's float type.
    pub fn next_lr(&self) -> Option<F> {
        self.inner.next_lr().and_then(|lr| F::from(lr))
    }

    /// Record the loss (as the model's float type) and advance.
    pub fn record_loss(&mut self, loss: F) -> LRFinderStatus {
        let loss_f64 = loss.to_f64().unwrap_or(f64::NAN);
        self.inner.record_loss(loss_f64)
    }

    /// Whether the test has finished.
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    /// Get the result.
    pub fn result(&self) -> LRFinderResult {
        self.inner.result()
    }

    /// Reset the finder.
    pub fn reset(&mut self) {
        self.inner.reset();
        self.original_params = None;
    }
}

impl<F: Float + Debug + ScalarOperand + NumAssign> Debug for TypedLRFinder<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TypedLRFinder")
            .field("inner", &self.inner)
            .field("has_original_params", &self.original_params.is_some())
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = LRFinderConfig::default();
        assert!((config.min_lr - 1e-7).abs() < 1e-15);
        assert!((config.max_lr - 10.0).abs() < 1e-10);
        assert_eq!(config.num_iterations, 100);
        assert_eq!(config.schedule, LRScheduleType::Exponential);
    }

    #[test]
    fn test_config_builder() {
        let config = LRFinderConfig::builder()
            .min_lr(1e-5)
            .max_lr(1.0)
            .num_iterations(50)
            .schedule(LRScheduleType::Linear)
            .smoothing_factor(0.1)
            .divergence_threshold(4.0)
            .build()
            .expect("build should succeed");

        assert!((config.min_lr - 1e-5).abs() < 1e-15);
        assert!((config.max_lr - 1.0).abs() < 1e-10);
        assert_eq!(config.num_iterations, 50);
        assert_eq!(config.schedule, LRScheduleType::Linear);
    }

    #[test]
    fn test_config_validation_errors() {
        // min_lr <= 0
        assert!(LRFinderConfig::builder().min_lr(-1.0).build().is_err());
        assert!(LRFinderConfig::builder().min_lr(0.0).build().is_err());

        // max_lr <= min_lr
        assert!(LRFinderConfig::builder()
            .min_lr(1.0)
            .max_lr(0.5)
            .build()
            .is_err());

        // num_iterations == 0
        assert!(LRFinderConfig::builder().num_iterations(0).build().is_err());

        // smoothing_factor out of range
        assert!(LRFinderConfig::builder()
            .smoothing_factor(1.5)
            .build()
            .is_err());
        assert!(LRFinderConfig::builder()
            .smoothing_factor(-0.1)
            .build()
            .is_err());

        // divergence_threshold <= 1.0
        assert!(LRFinderConfig::builder()
            .divergence_threshold(1.0)
            .build()
            .is_err());
        assert!(LRFinderConfig::builder()
            .divergence_threshold(0.5)
            .build()
            .is_err());

        // accumulation_steps == 0
        assert!(LRFinderConfig::builder()
            .accumulation_steps(0)
            .build()
            .is_err());
    }

    #[test]
    fn test_exponential_lr_schedule() {
        let config = LRFinderConfig::builder()
            .min_lr(1e-4)
            .max_lr(1.0)
            .num_iterations(100)
            .schedule(LRScheduleType::Exponential)
            .build()
            .expect("build should succeed");

        let lr_start = config.lr_at_step(0);
        let lr_end = config.lr_at_step(100);
        let lr_mid = config.lr_at_step(50);

        assert!((lr_start - 1e-4).abs() < 1e-10);
        assert!((lr_end - 1.0).abs() < 1e-6);
        // For exponential, lr_mid = min_lr * (max_lr / min_lr)^0.5 = sqrt(min_lr * max_lr)
        let expected_mid = (1e-4_f64 * 1.0).sqrt();
        assert!((lr_mid - expected_mid).abs() < 1e-6);
    }

    #[test]
    fn test_linear_lr_schedule() {
        let config = LRFinderConfig::builder()
            .min_lr(0.0001)
            .max_lr(1.0)
            .num_iterations(100)
            .schedule(LRScheduleType::Linear)
            .build()
            .expect("build should succeed");

        let lr_start = config.lr_at_step(0);
        let lr_end = config.lr_at_step(100);
        let lr_mid = config.lr_at_step(50);

        assert!((lr_start - 0.0001).abs() < 1e-10);
        assert!((lr_end - 1.0).abs() < 1e-6);
        let expected_mid = 0.0001 + (1.0 - 0.0001) * 0.5;
        assert!((lr_mid - expected_mid).abs() < 1e-6);
    }

    #[test]
    fn test_lr_finder_decreasing_then_diverging_loss() {
        let config = LRFinderConfig::builder()
            .min_lr(1e-5)
            .max_lr(1.0)
            .num_iterations(100)
            .smoothing_factor(0.0) // no smoothing for predictable behavior
            .divergence_threshold(5.0)
            .build()
            .expect("build should succeed");

        // Simulate a typical loss curve:
        // Loss decreases then explodes
        let losses: Vec<f64> = (0..100)
            .map(|i| {
                let t = i as f64 / 100.0;
                if t < 0.4 {
                    1.0 - t * 2.0 // decreasing from 1.0 to 0.2
                } else {
                    0.2 + (t - 0.4).powi(2) * 50.0 // exploding
                }
            })
            .collect();

        let result = LRFinderResult::from_losses(&losses, &config);

        // Should find a suggested LR (steepest decrease)
        let suggested = result.suggested_lr();
        assert!(suggested.is_some());
        let lr = suggested.expect("should have suggested lr");
        // The suggested LR should be in the early region (where loss decreases)
        assert!(lr > 1e-5, "lr={lr} should be > 1e-5");
        assert!(lr < 1.0, "lr={lr} should be < 1.0");
    }

    #[test]
    fn test_lr_finder_divergence_detection() {
        let config = LRFinderConfig::builder()
            .min_lr(1e-5)
            .max_lr(100.0)
            .num_iterations(100)
            .smoothing_factor(0.0)  // no smoothing for simple test
            .divergence_threshold(3.0)
            .build()
            .expect("build should succeed");

        // Loss that starts at 1.0 and quickly explodes
        let losses: Vec<f64> = (0..100)
            .map(|i| {
                if i < 30 {
                    1.0 - i as f64 * 0.01 // decreasing
                } else {
                    0.7 + (i as f64 - 30.0).powi(2) * 0.1 // exploding
                }
            })
            .collect();

        let result = LRFinderResult::from_losses(&losses, &config);
        assert!(result.diverged);
        assert!(result.divergence_step.is_some());
        // Should have stopped before all 100 iterations
        assert!(result.points.len() < 100);
    }

    #[test]
    fn test_lr_finder_no_divergence() {
        let config = LRFinderConfig::builder()
            .min_lr(1e-5)
            .max_lr(0.1)
            .num_iterations(50)
            .smoothing_factor(0.0)
            .divergence_threshold(5.0)
            .build()
            .expect("build should succeed");

        // Monotonically decreasing loss — no divergence
        let losses: Vec<f64> = (0..50).map(|i| 1.0 - i as f64 * 0.015).collect();

        let result = LRFinderResult::from_losses(&losses, &config);
        assert!(!result.diverged);
        assert_eq!(result.points.len(), 50);
    }

    #[test]
    fn test_stateful_lr_finder() {
        let config = LRFinderConfig::builder()
            .min_lr(1e-5)
            .max_lr(1.0)
            .num_iterations(10)
            .smoothing_factor(0.0)
            .divergence_threshold(5.0)
            .build()
            .expect("build should succeed");

        let mut finder = LRFinder::new(config).expect("should create finder");

        for i in 0..10 {
            assert!(!finder.is_finished());
            let lr = finder.next_lr().expect("should have lr");
            assert!(lr > 0.0);

            let loss = 1.0 - (i as f64) * 0.05;
            let status = finder.record_loss(loss);

            if i < 9 {
                assert_eq!(status, LRFinderStatus::Continue);
            } else {
                assert_eq!(status, LRFinderStatus::Complete);
            }
        }

        assert!(finder.is_finished());
        assert!(!finder.is_diverged());

        let result = finder.result();
        assert_eq!(result.points.len(), 10);
    }

    #[test]
    fn test_stateful_lr_finder_divergence() {
        let config = LRFinderConfig::builder()
            .min_lr(1e-5)
            .max_lr(1.0)
            .num_iterations(100)
            .smoothing_factor(0.0)
            .divergence_threshold(3.0)
            .build()
            .expect("build should succeed");

        let mut finder = LRFinder::new(config).expect("should create finder");

        // Feed a loss that decreases then explodes
        let mut step = 0;
        loop {
            if finder.is_finished() {
                break;
            }
            let _lr = finder.next_lr().expect("should have lr");
            let loss = if step < 5 {
                1.0 - step as f64 * 0.05
            } else {
                0.75 + (step as f64 - 5.0).powi(2) * 0.5
            };
            let status = finder.record_loss(loss);
            if status != LRFinderStatus::Continue {
                break;
            }
            step += 1;
        }

        assert!(finder.is_finished());
        assert!(finder.is_diverged());
    }

    #[test]
    fn test_stateful_lr_finder_reset() {
        let config = LRFinderConfig::builder()
            .min_lr(1e-5)
            .max_lr(1.0)
            .num_iterations(5)
            .smoothing_factor(0.0)
            .divergence_threshold(5.0)
            .build()
            .expect("build should succeed");

        let mut finder = LRFinder::new(config).expect("should create finder");

        // Run to completion
        for i in 0..5 {
            finder.record_loss(1.0 - i as f64 * 0.1);
        }
        assert!(finder.is_finished());

        // Reset and run again
        finder.reset();
        assert!(!finder.is_finished());
        assert_eq!(finder.current_step(), 0);

        for i in 0..5 {
            finder.record_loss(0.5 - i as f64 * 0.05);
        }
        assert!(finder.is_finished());
        let result = finder.result();
        assert_eq!(result.points.len(), 5);
    }

    #[test]
    fn test_find_optimal_lr_convenience() {
        let config = LRFinderConfig::builder()
            .min_lr(1e-5)
            .max_lr(1.0)
            .num_iterations(50)
            .smoothing_factor(0.05)
            .divergence_threshold(5.0)
            .build()
            .expect("build should succeed");

        let losses: Vec<f64> = (0..50)
            .map(|i| {
                let t = i as f64 / 50.0;
                // U-shaped loss in log-space
                0.5 + (t - 0.4).powi(2) * 2.0
            })
            .collect();

        let (suggested, result) = find_optimal_lr(&losses, &config).expect("should succeed");
        assert!(suggested.is_some());
        assert!(!result.diverged);
    }

    #[test]
    fn test_result_accessors() {
        let config = LRFinderConfig::default();
        let losses: Vec<f64> = (0..100).map(|i| 1.0 - i as f64 * 0.005).collect();
        let result = LRFinderResult::from_losses(&losses, &config);

        let lrs = result.learning_rates();
        assert_eq!(lrs.len(), result.points.len());

        let raw = result.raw_losses();
        assert_eq!(raw.len(), result.points.len());

        let smoothed = result.smoothed_losses();
        assert_eq!(smoothed.len(), result.points.len());

        let grads = result.loss_gradients();
        assert_eq!(grads.len(), result.points.len());
        // First gradient should be NaN (no predecessor)
        assert!(grads[0].is_nan());
    }

    #[test]
    fn test_summary_generation() {
        let config = LRFinderConfig::default();
        let losses: Vec<f64> = (0..20).map(|i| 1.0 - i as f64 * 0.01).collect();
        let result = LRFinderResult::from_losses(&losses, &config);

        let summary = result.summary();
        assert!(summary.contains("Learning Rate Range Test Summary"));
        assert!(summary.contains("Exponential"));
        assert!(summary.contains("Best loss"));
    }

    #[test]
    fn test_conservative_suggestion() {
        let config = LRFinderConfig::builder()
            .min_lr(1e-5)
            .max_lr(1.0)
            .num_iterations(20)
            .smoothing_factor(0.0)
            .divergence_threshold(5.0)
            .build()
            .expect("build should succeed");

        let losses: Vec<f64> = (0..20)
            .map(|i| {
                let t = i as f64 / 20.0;
                (t - 0.5).powi(2) + 0.1
            })
            .collect();

        let result = LRFinderResult::from_losses(&losses, &config);
        let conservative = result.suggested_lr_conservative();
        // Conservative should be 10x smaller than best_loss_lr
        assert!((conservative - result.best_loss_lr / 10.0).abs() < 1e-15);
    }

    #[test]
    fn test_typed_lr_finder() {
        let config = LRFinderConfig::builder()
            .min_lr(1e-5)
            .max_lr(1.0)
            .num_iterations(10)
            .smoothing_factor(0.0)
            .divergence_threshold(5.0)
            .build()
            .expect("build should succeed");

        let mut finder = TypedLRFinder::<f64>::new(config).expect("should create finder");

        // Save some "params"
        let params = vec![Array::<f64, IxDyn>::zeros(IxDyn(&[3, 3]))];
        finder.save_params(&params);
        assert!(finder.original_params().is_some());

        for i in 0..10 {
            let lr: f64 = finder.next_lr().expect("should have lr");
            assert!(lr > 0.0);
            let loss = 1.0 - i as f64 * 0.05;
            finder.record_loss(loss);
        }

        assert!(finder.is_finished());
        let result = finder.result();
        assert_eq!(result.points.len(), 10);
    }

    #[test]
    fn test_typed_lr_finder_f32() {
        let config = LRFinderConfig::builder()
            .min_lr(1e-5)
            .max_lr(1.0)
            .num_iterations(5)
            .smoothing_factor(0.0)
            .divergence_threshold(5.0)
            .build()
            .expect("build should succeed");

        let mut finder = TypedLRFinder::<f32>::new(config).expect("should create finder");

        for i in 0..5 {
            let lr: f32 = finder.next_lr().expect("should have lr");
            assert!(lr > 0.0);
            let loss: f32 = 1.0 - i as f32 * 0.1;
            finder.record_loss(loss);
        }

        assert!(finder.is_finished());
    }

    #[test]
    fn test_ema_smoothing() {
        let config = LRFinderConfig::builder()
            .min_lr(1e-5)
            .max_lr(1.0)
            .num_iterations(100)
            .smoothing_factor(0.1)  // heavy smoothing
            .divergence_threshold(100.0)  // disable divergence
            .build()
            .expect("build should succeed");

        // Noisy loss with an underlying downward trend
        let losses: Vec<f64> = (0..100)
            .map(|i| {
                let trend = 1.0 - i as f64 * 0.005;
                let noise = if i % 2 == 0 { 0.1 } else { -0.1 };
                trend + noise
            })
            .collect();

        let result = LRFinderResult::from_losses(&losses, &config);

        // Smoothed losses should be less noisy than raw losses
        let raw = result.raw_losses();
        let smoothed = result.smoothed_losses();

        // Compute variance of differences
        fn variance(data: &[f64]) -> f64 {
            if data.len() < 2 {
                return 0.0;
            }
            let diffs: Vec<f64> = data.windows(2).map(|w| (w[1] - w[0]).powi(2)).collect();
            diffs.iter().sum::<f64>() / diffs.len() as f64
        }

        let raw_var = variance(&raw);
        let smoothed_var = variance(&smoothed);
        // Smoothed should have lower variance of differences
        assert!(
            smoothed_var < raw_var,
            "smoothed_var={smoothed_var} should be < raw_var={raw_var}"
        );
    }

    #[test]
    fn test_lr_schedule_display() {
        assert_eq!(format!("{}", LRScheduleType::Exponential), "Exponential");
        assert_eq!(format!("{}", LRScheduleType::Linear), "Linear");
    }

    #[test]
    fn test_lr_finder_after_finished() {
        let config = LRFinderConfig::builder()
            .min_lr(1e-5)
            .max_lr(1.0)
            .num_iterations(3)
            .smoothing_factor(0.0)
            .divergence_threshold(5.0)
            .build()
            .expect("build should succeed");

        let mut finder = LRFinder::new(config).expect("should create finder");
        for _ in 0..3 {
            finder.record_loss(0.5);
        }
        assert!(finder.is_finished());

        // Calling next_lr after finished should return None
        assert!(finder.next_lr().is_none());

        // Recording loss after finished should return Complete
        assert_eq!(finder.record_loss(0.5), LRFinderStatus::Complete);
    }

    #[test]
    fn test_typed_lr_finder_reset() {
        let config = LRFinderConfig::builder()
            .min_lr(1e-5)
            .max_lr(1.0)
            .num_iterations(3)
            .smoothing_factor(0.0)
            .divergence_threshold(5.0)
            .build()
            .expect("build should succeed");

        let mut finder = TypedLRFinder::<f64>::new(config).expect("should create finder");
        let params = vec![Array::<f64, IxDyn>::zeros(IxDyn(&[2]))];
        finder.save_params(&params);

        for _ in 0..3 {
            finder.record_loss(0.5);
        }
        assert!(finder.is_finished());

        finder.reset();
        assert!(!finder.is_finished());
        assert!(finder.original_params().is_none());
    }
}
