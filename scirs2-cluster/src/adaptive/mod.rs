//! Self-adaptive mini-batch size controller.
//!
//! `BatchSizeController` monitors a streaming loss signal and automatically
//! grows or shrinks the mini-batch size using simple heuristics:
//!
//! - If the loss is **decreasing fast** (relative standard deviation of recent
//!   losses is small): the current gradient direction is reliable → **grow** the
//!   batch to exploit it.
//! - If the loss **plateaus or increases** (recent mean is above the previous
//!   window mean): gradient estimates are noisy or the optimum is near →
//!   **shrink** the batch for finer stochastic exploration.

use crate::error::ClusteringError;

/// Configuration for [`BatchSizeController`].
#[derive(Debug, Clone)]
pub struct AdaptiveBatchConfig {
    /// Starting batch size.
    pub initial_batch_size: usize,
    /// Hard lower bound on batch size.
    pub min_batch: usize,
    /// Hard upper bound on batch size.
    pub max_batch: usize,
    /// Multiplicative factor when growing the batch (must be > 1).
    pub growth_factor: f64,
    /// Multiplicative factor when shrinking the batch (must be in (0, 1)).
    pub decay_factor: f64,
    /// Number of recent losses used to compute statistics.
    pub window: usize,
}

impl Default for AdaptiveBatchConfig {
    fn default() -> Self {
        Self {
            initial_batch_size: 32,
            min_batch: 16,
            max_batch: 2048,
            growth_factor: 1.5,
            decay_factor: 0.8,
            window: 6, // 3 "recent" + 3 "previous"
        }
    }
}

/// Online controller that tracks a loss history and recommends a batch size.
pub struct BatchSizeController {
    /// Current recommended batch size.
    pub current_size: usize,
    /// Full history of recorded losses (oldest first).
    pub loss_history: Vec<f64>,
    config: AdaptiveBatchConfig,
}

impl BatchSizeController {
    /// Create a new controller with the given configuration.
    pub fn new(config: AdaptiveBatchConfig) -> Self {
        let initial = config
            .initial_batch_size
            .clamp(config.min_batch, config.max_batch);
        Self {
            current_size: initial,
            loss_history: Vec::new(),
            config,
        }
    }

    /// Record a new loss value **without** updating the batch size recommendation.
    pub fn record_loss(&mut self, loss: f64) {
        self.loss_history.push(loss);
    }

    /// Recommend a batch size based on the recorded loss history.
    ///
    /// Decision rules (applied to the most recent `window` observations):
    ///
    /// 1. Not enough history → return current size unchanged.
    /// 2. Split history into `last_half` and `prev_half` (each `window/2` long).
    /// 3. If `std(last_half) / mean(last_half) < 0.01` → **grow** (stable descent).
    /// 4. If `mean(last_half) > mean(prev_half)` → **shrink** (loss increased).
    /// 5. Otherwise → no change.
    pub fn recommend_size(&self) -> usize {
        let w = self.config.window.max(2);
        let half = w / 2;

        if self.loss_history.len() < w {
            return self.current_size;
        }

        let recent: &[f64] = &self.loss_history[self.loss_history.len() - half..];
        let prev: &[f64] =
            &self.loss_history[self.loss_history.len() - w..self.loss_history.len() - half];

        let mean_recent = mean(recent);
        let mean_prev = mean(prev);
        let std_recent = std_dev(recent);

        // Rule 1: loss is decreasing reliably → grow
        let relative_std = if mean_recent.abs() > 1e-12 {
            std_recent / mean_recent.abs()
        } else {
            std_recent
        };

        if relative_std < 0.01 {
            let new_size =
                ((self.current_size as f64) * self.config.growth_factor).round() as usize;
            return new_size.clamp(self.config.min_batch, self.config.max_batch);
        }

        // Rule 2: loss has increased → shrink
        if mean_recent > mean_prev {
            let new_size = ((self.current_size as f64) * self.config.decay_factor).round() as usize;
            return new_size.clamp(self.config.min_batch, self.config.max_batch);
        }

        self.current_size
    }

    /// Record `loss`, update `current_size` and return the new recommended size.
    pub fn adapt(&mut self, loss: f64) -> usize {
        self.record_loss(loss);
        let new_size = self.recommend_size();
        self.current_size = new_size;
        new_size
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.current_size = self
            .config
            .initial_batch_size
            .clamp(self.config.min_batch, self.config.max_batch);
        self.loss_history.clear();
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), ClusteringError> {
        if self.config.growth_factor <= 1.0 {
            return Err(ClusteringError::InvalidInput(
                "growth_factor must be > 1".into(),
            ));
        }
        if self.config.decay_factor <= 0.0 || self.config.decay_factor >= 1.0 {
            return Err(ClusteringError::InvalidInput(
                "decay_factor must be in (0, 1)".into(),
            ));
        }
        if self.config.min_batch > self.config.max_batch {
            return Err(ClusteringError::InvalidInput(
                "min_batch must be ≤ max_batch".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_size_clamped() {
        let config = AdaptiveBatchConfig {
            initial_batch_size: 4,
            min_batch: 16,
            max_batch: 2048,
            ..Default::default()
        };
        let ctrl = BatchSizeController::new(config);
        assert_eq!(ctrl.current_size, 16);
    }

    #[test]
    fn test_not_enough_history_returns_current() {
        let mut ctrl = BatchSizeController::new(AdaptiveBatchConfig::default());
        ctrl.record_loss(1.0);
        ctrl.record_loss(0.9);
        // window=6 not reached yet
        assert_eq!(ctrl.recommend_size(), ctrl.current_size);
    }

    #[test]
    fn test_decreasing_loss_grows_batch() {
        let config = AdaptiveBatchConfig {
            initial_batch_size: 64,
            min_batch: 16,
            max_batch: 2048,
            growth_factor: 2.0,
            decay_factor: 0.5,
            window: 6,
        };
        let mut ctrl = BatchSizeController::new(config);

        // Very stable decreasing losses → small relative std
        for i in 0..6 {
            ctrl.record_loss(1.0 - 0.001 * i as f64);
        }
        let size = ctrl.recommend_size();
        assert!(
            size > 64,
            "Batch size should grow on stable decreasing loss, got {}",
            size
        );
    }

    #[test]
    fn test_increasing_loss_shrinks_batch() {
        let config = AdaptiveBatchConfig {
            initial_batch_size: 256,
            min_batch: 16,
            max_batch: 2048,
            growth_factor: 1.5,
            decay_factor: 0.5,
            window: 6,
        };
        let mut ctrl = BatchSizeController::new(config);

        // prev_half: low losses; last_half: high losses → mean_recent > mean_prev
        ctrl.record_loss(0.1);
        ctrl.record_loss(0.11);
        ctrl.record_loss(0.12);
        ctrl.record_loss(1.5);
        ctrl.record_loss(1.6);
        ctrl.record_loss(1.7);

        let size = ctrl.recommend_size();
        assert!(
            size < 256,
            "Batch size should shrink on increasing loss, got {}",
            size
        );
    }

    #[test]
    fn test_adapt_updates_current_size() {
        let mut ctrl = BatchSizeController::new(AdaptiveBatchConfig {
            initial_batch_size: 256,
            window: 6,
            ..Default::default()
        });

        // Force shrink: inject increasing losses
        ctrl.adapt(0.1);
        ctrl.adapt(0.11);
        ctrl.adapt(0.12);
        ctrl.adapt(1.5);
        ctrl.adapt(1.6);
        let final_size = ctrl.adapt(1.7);
        assert_eq!(
            final_size, ctrl.current_size,
            "adapt() should update current_size"
        );
    }

    #[test]
    fn test_bounds_respected() {
        let config = AdaptiveBatchConfig {
            initial_batch_size: 17,
            min_batch: 16,
            max_batch: 18,
            growth_factor: 1000.0, // extreme
            decay_factor: 0.001,   // extreme
            window: 6,
        };
        let mut ctrl = BatchSizeController::new(config);

        // Trigger grow
        for i in 0..6 {
            ctrl.record_loss(1.0 - 0.0001 * i as f64);
        }
        let grown = ctrl.recommend_size();
        assert!(grown <= 18, "Must not exceed max_batch");

        // Trigger shrink
        ctrl.reset();
        ctrl.record_loss(0.01);
        ctrl.record_loss(0.01);
        ctrl.record_loss(0.01);
        ctrl.record_loss(10.0);
        ctrl.record_loss(10.0);
        ctrl.record_loss(10.0);
        let shrunk = ctrl.recommend_size();
        assert!(shrunk >= 16, "Must not go below min_batch");
    }

    #[test]
    fn test_validate_config() {
        let ctrl = BatchSizeController::new(AdaptiveBatchConfig::default());
        assert!(ctrl.validate().is_ok());

        let bad = BatchSizeController::new(AdaptiveBatchConfig {
            growth_factor: 0.5, // invalid
            ..Default::default()
        });
        assert!(bad.validate().is_err());
    }
}
