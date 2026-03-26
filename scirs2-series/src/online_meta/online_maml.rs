//! Online MAML — streaming adaptation for time series.
//!
//! [`OnlineMetaLearner`] accumulates observations into a sliding window of tasks.
//! When the buffer reaches `task_buffer_size`, it triggers a FOMAML meta-update
//! and resets the current task buffer.  Between meta-updates it uses the
//! current adapted model for predictions.

use super::maml::MamlOptimizer;
use super::types::{linear_predict, MetaLearnerConfig, Task};
use crate::error::TimeSeriesError;

// ─────────────────────────────────────────────────────────────────────────────

/// Internal builder accumulating `(x, y)` pairs for the current open task.
struct TaskBuilder {
    xs: Vec<Vec<f64>>,
    ys: Vec<f64>,
    split_ratio: f64, // fraction used as support vs query
}

impl TaskBuilder {
    fn new(split_ratio: f64) -> Self {
        Self {
            xs: Vec::new(),
            ys: Vec::new(),
            split_ratio,
        }
    }

    fn add(&mut self, x: Vec<f64>, y: f64) {
        self.xs.push(x);
        self.ys.push(y);
    }

    fn len(&self) -> usize {
        self.ys.len()
    }

    /// Convert the accumulated observations into a [`Task`], splitting into
    /// support (first `split_ratio` fraction) and query (remainder).
    fn build(self) -> Option<Task> {
        let n = self.xs.len();
        if n < 2 {
            return None;
        }
        let n_support = (n as f64 * self.split_ratio).ceil() as usize;
        let n_support = n_support.min(n - 1).max(1);
        let n_query = n - n_support;
        if n_query == 0 {
            return None;
        }
        let support_x = self.xs[..n_support].to_vec();
        let support_y = self.ys[..n_support].to_vec();
        let query_x = self.xs[n_support..].to_vec();
        let query_y = self.ys[n_support..].to_vec();
        Some(Task {
            support_x,
            support_y,
            query_x,
            query_y,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Online meta-learner that adapts to streaming time series in real time.
///
/// The learner maintains:
/// - A rolling buffer of completed [`Task`]s.
/// - The current open task being built from incoming observations.
/// - A MAML outer model and a task-adapted inner model for prediction.
pub struct OnlineMetaLearner {
    /// MAML optimiser holding meta-parameters.
    optimizer: MamlOptimizer,
    /// Buffer of recently completed tasks.
    task_buffer: Vec<Task>,
    /// Current open task being constructed from streaming observations.
    current_task_builder: TaskBuilder,
    /// Number of observations per task window before finalising.
    task_window_size: usize,
    /// Adapted model used for predictions (cloned from meta-model + inner update).
    adapted_weights: Vec<f64>,
    adapted_bias: f64,
    /// Total number of meta-update steps performed.
    meta_update_count: usize,
}

impl OnlineMetaLearner {
    /// Create a new online meta-learner from a configuration.
    pub fn new(config: MetaLearnerConfig, task_window_size: usize) -> Self {
        let adapted_weights = vec![0.0; config.feature_dim];
        let optimizer = MamlOptimizer::new(config);
        Self {
            optimizer,
            task_buffer: Vec::new(),
            current_task_builder: TaskBuilder::new(0.7),
            task_window_size,
            adapted_weights,
            adapted_bias: 0.0,
            meta_update_count: 0,
        }
    }

    /// Add a new observation `(x, y)` to the current open task.
    pub fn update(&mut self, x: &[f64], y: f64) {
        self.current_task_builder.add(x.to_vec(), y);
    }

    /// Check how many observations are in the current open task.
    pub fn current_task_len(&self) -> usize {
        self.current_task_builder.len()
    }

    /// How many tasks are in the buffer.
    pub fn task_buffer_len(&self) -> usize {
        self.task_buffer.len()
    }

    /// Finalise the current task and push it to the buffer.
    ///
    /// If the current task has fewer than 2 observations the call is a no-op.
    pub fn finalize_task(&mut self) -> Option<Task> {
        // Swap in a fresh builder
        let old_builder = std::mem::replace(&mut self.current_task_builder, TaskBuilder::new(0.7));
        let task = old_builder.build()?;
        // Update adapted model on this task
        let adapted = self.optimizer.model().inner_update(
            &task,
            self.optimizer.config.inner_lr,
            self.optimizer.config.n_inner_steps,
        );
        self.adapted_weights = adapted.weights.clone();
        self.adapted_bias = adapted.bias;
        // Push to rolling buffer; evict oldest if full
        let max_buf = self.optimizer.config.task_buffer_size;
        if self.task_buffer.len() >= max_buf {
            self.task_buffer.remove(0);
        }
        self.task_buffer.push(task.clone());
        Some(task)
    }

    /// Finalise the current open window if it has reached `task_window_size`.
    pub fn maybe_finalize(&mut self) -> Option<Task> {
        if self.current_task_builder.len() >= self.task_window_size {
            self.finalize_task()
        } else {
            None
        }
    }

    /// Run a MAML meta-update if the buffer has enough tasks.
    ///
    /// Returns `Some(meta_loss)` if a meta-update was performed, otherwise `None`.
    pub fn meta_update_if_ready(&mut self) -> Option<f64> {
        let min_tasks = self.optimizer.config.min_tasks_for_update;
        if self.task_buffer.len() < min_tasks {
            return None;
        }
        let loss = self.optimizer.meta_train_step(&self.task_buffer);
        self.meta_update_count += 1;
        Some(loss)
    }

    /// Predict `y` for feature vector `x` using the current adapted model.
    pub fn predict(&self, x: &[f64]) -> f64 {
        linear_predict(&self.adapted_weights, self.adapted_bias, x)
    }

    /// Number of completed meta-updates.
    pub fn meta_update_count(&self) -> usize {
        self.meta_update_count
    }

    /// Force-finalize the current open task and then run a meta-update if ready.
    pub fn flush(&mut self) -> Result<Option<f64>, TimeSeriesError> {
        self.finalize_task();
        Ok(self.meta_update_if_ready())
    }

    /// Return the current meta-model weights (for inspection / transfer).
    pub fn meta_weights(&self) -> (&[f64], f64) {
        let m = self.optimizer.model();
        (&m.weights, m.bias)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_maml_buffer_fills() {
        let config = MetaLearnerConfig {
            feature_dim: 2,
            task_buffer_size: 4,
            min_tasks_for_update: 4,
            ..Default::default()
        };
        let mut learner = OnlineMetaLearner::new(config, 10);

        // Feed 4 task windows
        for task_idx in 0..4_usize {
            for i in 0..10 {
                let x = vec![i as f64 / 10.0, task_idx as f64];
                let y = i as f64 / 10.0 + task_idx as f64;
                learner.update(&x, y);
            }
            learner.finalize_task();
        }
        assert_eq!(
            learner.task_buffer_len(),
            4,
            "buffer should contain 4 tasks"
        );
    }

    #[test]
    fn test_online_maml_predict_shape() {
        let config = MetaLearnerConfig {
            feature_dim: 3,
            ..Default::default()
        };
        let learner = OnlineMetaLearner::new(config, 20);
        // Even before any training, predict should return a finite f64
        let pred = learner.predict(&[1.0, 2.0, 3.0]);
        assert!(pred.is_finite(), "prediction should be finite");
    }

    #[test]
    fn test_online_maml_meta_update_triggered() {
        let config = MetaLearnerConfig {
            feature_dim: 1,
            task_buffer_size: 4,
            min_tasks_for_update: 4,
            inner_lr: 0.05,
            outer_lr: 0.01,
            n_inner_steps: 5,
        };
        let mut learner = OnlineMetaLearner::new(config, 8);

        // Feed enough tasks to trigger meta-update
        for slope in [1.0_f64, 2.0, 3.0, 4.0] {
            for i in 0..8 {
                let x = vec![i as f64 / 8.0];
                let y = slope * i as f64 / 8.0;
                learner.update(&x, y);
            }
            learner.finalize_task();
        }
        let result = learner.meta_update_if_ready();
        assert!(result.is_some(), "meta-update should have been triggered");
        let loss = result.expect("meta loss should exist");
        assert!(loss.is_finite(), "meta loss should be finite: {loss}");
    }
}
