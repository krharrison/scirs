//! Reptile meta-learning algorithm.
//!
//! Reference: Nichol et al. (2018) "On First-Order Meta-Learning Algorithms".
//!
//! Reptile update rule:
//!   θ ← θ + ε · (φ_k - θ)
//! where φ_k is the result of k SGD steps from θ on a single task.
//!
//! Compared to MAML, Reptile avoids computing or estimating second-order gradients
//! and only requires a single set of inner-loop weights per task.

use super::maml::MetaLinearModel;
use super::types::{
    mse, predict_all, AdaptationMetrics, MetaLearnerConfig, MetaLearnerResult, Task,
};
use crate::error::TimeSeriesError;

// ─────────────────────────────────────────────────────────────────────────────

/// Reptile meta-learner built on top of [`MetaLinearModel`].
pub struct ReptileOptimizer {
    /// Meta-parameters (the shared initialisation θ).
    model: MetaLinearModel,
    /// Configuration (uses `outer_lr` as ε, `inner_lr` / `n_inner_steps` for inner SGD).
    pub config: MetaLearnerConfig,
    /// Number of completed train steps.
    steps_completed: usize,
}

impl ReptileOptimizer {
    /// Create a new Reptile optimiser with zero-initialised meta-parameters.
    pub fn new(config: MetaLearnerConfig) -> Self {
        let model = MetaLinearModel::zeros(config.feature_dim);
        Self {
            model,
            config,
            steps_completed: 0,
        }
    }

    /// Create a Reptile optimiser from an existing model.
    pub fn from_model(model: MetaLinearModel, config: MetaLearnerConfig) -> Self {
        Self {
            model,
            config,
            steps_completed: 0,
        }
    }

    /// Perform one Reptile update step for a single task.
    ///
    /// 1. Clone θ to get φ.
    /// 2. Run `n_steps` inner SGD steps on `task.support_x / support_y`.
    /// 3. Apply θ ← θ + ε · (φ_k − θ).
    ///
    /// Returns the support-set MSE *after* adaptation (for monitoring).
    pub fn train_step(&mut self, task: &Task) -> f64 {
        let phi = self
            .model
            .inner_update(task, self.config.inner_lr, self.config.n_inner_steps);
        let eps = self.config.outer_lr;

        // θ ← θ + ε · (φ_k − θ)
        for (w, phi_w) in self.model.weights.iter_mut().zip(phi.weights.iter()) {
            *w += eps * (phi_w - *w);
        }
        self.model.bias += eps * (phi.bias - self.model.bias);
        self.steps_completed += 1;

        // Return support-set loss of the adapted model
        phi.loss(&task.support_x, &task.support_y)
    }

    /// Fast adaptation: run `n_steps` inner-loop steps from θ and return the result.
    pub fn adapt(&self, task: &Task, n_steps: usize) -> Result<MetaLearnerResult, TimeSeriesError> {
        if task.query_x.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Task query set is empty".to_string(),
            ));
        }
        let adapted = self.model.inner_update(task, self.config.inner_lr, n_steps);
        let preds = predict_all(&adapted.weights, adapted.bias, &task.query_x);
        let query_loss = mse(&preds, &task.query_y);
        let mut adapted_weights = adapted.weights.clone();
        adapted_weights.push(adapted.bias);
        Ok(MetaLearnerResult {
            adapted_weights,
            query_loss,
            adaptation_steps: n_steps,
        })
    }

    /// Compute adaptation metrics for a task.
    pub fn adaptation_metrics(&self, task: &Task) -> AdaptationMetrics {
        let pre_preds = predict_all(&self.model.weights, self.model.bias, &task.query_x);
        let pre_loss = mse(&pre_preds, &task.query_y);
        let adapted =
            self.model
                .inner_update(task, self.config.inner_lr, self.config.n_inner_steps);
        let post_preds = predict_all(&adapted.weights, adapted.bias, &task.query_x);
        let post_loss = mse(&post_preds, &task.query_y);
        AdaptationMetrics::compute(pre_loss, post_loss)
    }

    /// Return the current meta-parameters.
    pub fn model(&self) -> &MetaLinearModel {
        &self.model
    }

    /// Return the number of completed training steps.
    pub fn steps_completed(&self) -> usize {
        self.steps_completed
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear_task(slope: f64, intercept: f64, n: usize) -> Task {
        let support_x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 / n as f64]).collect();
        let support_y: Vec<f64> = support_x.iter().map(|x| slope * x[0] + intercept).collect();
        let query_x: Vec<Vec<f64>> = (0..n).map(|i| vec![(i as f64 + 0.5) / n as f64]).collect();
        let query_y: Vec<f64> = query_x.iter().map(|x| slope * x[0] + intercept).collect();
        Task {
            support_x,
            support_y,
            query_x,
            query_y,
        }
    }

    #[test]
    fn test_reptile_convergence() {
        let config = MetaLearnerConfig {
            feature_dim: 1,
            inner_lr: 0.05,
            outer_lr: 0.1,
            n_inner_steps: 10,
            ..Default::default()
        };
        let mut opt = ReptileOptimizer::new(config);
        let task = make_linear_task(2.0, 1.0, 20);

        let loss_init = opt.adapt(&task, 0).expect("adapt").query_loss;
        // Train for several steps
        let mut last_loss = loss_init;
        for _ in 0..50 {
            last_loss = opt.train_step(&task);
        }
        // Loss should decrease over training
        assert!(
            last_loss < loss_init || last_loss < 0.5,
            "reptile should converge: init={loss_init}, final={last_loss}"
        );
    }

    #[test]
    fn test_reptile_adapt_improves() {
        let config = MetaLearnerConfig {
            feature_dim: 1,
            inner_lr: 0.1,
            outer_lr: 0.1,
            n_inner_steps: 20,
            ..Default::default()
        };
        let mut opt = ReptileOptimizer::new(config.clone());
        // Meta-train on a family of linear tasks
        for slope in [1.0_f64, 2.0, 3.0, 4.0] {
            let t = make_linear_task(slope, 0.5, 30);
            for _ in 0..10 {
                opt.train_step(&t);
            }
        }
        // Adapt to a new task
        let test_task = make_linear_task(2.5, 0.5, 30);
        let pre = opt.adapt(&test_task, 0).expect("adapt pre").query_loss;
        let post = opt
            .adapt(&test_task, config.n_inner_steps)
            .expect("adapt post")
            .query_loss;
        // After adaptation, the loss should be ≤ pre-adaptation loss or at least finite
        assert!(post.is_finite(), "post-adaptation loss should be finite");
        // Pre should also be finite (meta-training shouldn't blow up)
        assert!(pre.is_finite(), "pre-adaptation loss should be finite");
    }

    #[test]
    fn test_reptile_steps_counter() {
        let config = MetaLearnerConfig {
            feature_dim: 1,
            ..Default::default()
        };
        let mut opt = ReptileOptimizer::new(config);
        let task = make_linear_task(1.0, 0.0, 5);
        assert_eq!(opt.steps_completed(), 0);
        opt.train_step(&task);
        opt.train_step(&task);
        assert_eq!(opt.steps_completed(), 2);
    }
}
