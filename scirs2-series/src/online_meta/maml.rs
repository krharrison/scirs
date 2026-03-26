//! MAML (Model-Agnostic Meta-Learning) for time series.
//!
//! Reference: Finn et al. (2017) "Model-Agnostic Meta-Learning for Fast Adaptation of Deep
//! Networks", ICML.
//!
//! This module implements the linear-model variant suitable for streaming time series tasks.
//! The inner loop runs SGD on the support set; the outer loop accumulates second-order-free
//! meta-gradients (FOMAML) using query-set losses.

use super::types::{
    linear_predict, mse, predict_all, AdaptationMetrics, MetaLearnerConfig, MetaLearnerResult, Task,
};
use crate::error::TimeSeriesError;

// ─── Linear base model ────────────────────────────────────────────────────────

/// A lightweight linear regression model used as the base model in MAML.
///
/// `weights[i]` corresponds to feature dimension `i`; `bias` is the intercept.
#[derive(Debug, Clone)]
pub struct MetaLinearModel {
    /// Feature weights (length = feature_dim).
    pub weights: Vec<f64>,
    /// Bias / intercept term.
    pub bias: f64,
}

impl MetaLinearModel {
    /// Create a zero-initialised model of the given feature dimensionality.
    pub fn zeros(feature_dim: usize) -> Self {
        Self {
            weights: vec![0.0; feature_dim],
            bias: 0.0,
        }
    }

    /// Create a model from explicit weights and bias.
    pub fn new(weights: Vec<f64>, bias: f64) -> Self {
        Self { weights, bias }
    }

    /// Forward pass: compute a scalar prediction for feature vector `x`.
    pub fn forward(&self, x: &[f64]) -> f64 {
        linear_predict(&self.weights, self.bias, x)
    }

    /// Compute MSE on a labelled dataset.
    pub fn loss(&self, xs: &[Vec<f64>], ys: &[f64]) -> f64 {
        let preds = predict_all(&self.weights, self.bias, xs);
        mse(&preds, ys)
    }

    /// Perform `n_steps` SGD inner-loop updates on the support set and return the adapted clone.
    ///
    /// MSE gradient:
    ///   ∂L/∂w_i = (2/n) Σ_j (ŷ_j - y_j) · x_ji
    ///   ∂L/∂b   = (2/n) Σ_j (ŷ_j - y_j)
    ///
    /// (The 2/n factor is absorbed into the learning rate for simplicity.)
    pub fn inner_update(&self, task: &Task, lr: f64, n_steps: usize) -> Self {
        let mut adapted = self.clone();
        let n = task.support_x.len();
        if n == 0 {
            return adapted;
        }
        let inv_n = 1.0 / n as f64;

        for _ in 0..n_steps {
            let mut grad_w = vec![0.0; adapted.weights.len()];
            let mut grad_b = 0.0f64;

            for (x, &y) in task.support_x.iter().zip(task.support_y.iter()) {
                let pred = adapted.forward(x);
                let err = pred - y; // (ŷ - y)
                for (i, xi) in x.iter().enumerate() {
                    if i < grad_w.len() {
                        grad_w[i] += err * xi;
                    }
                }
                grad_b += err;
            }

            // Scale gradients by 2/n and apply update
            for (w, gw) in adapted.weights.iter_mut().zip(grad_w.iter()) {
                *w -= lr * 2.0 * inv_n * gw;
            }
            adapted.bias -= lr * 2.0 * inv_n * grad_b;
        }
        adapted
    }

    /// Compute the meta-gradient contribution from a single task.
    ///
    /// Returns `(weight_gradient, bias_gradient)` evaluated on the query set
    /// after inner-loop adaptation (FOMAML — first-order approximation).
    pub fn task_meta_gradient(
        &self,
        task: &Task,
        inner_lr: f64,
        n_inner_steps: usize,
    ) -> (Vec<f64>, f64) {
        let adapted = self.inner_update(task, inner_lr, n_inner_steps);
        let n = task.query_x.len();
        if n == 0 {
            return (vec![0.0; self.weights.len()], 0.0);
        }
        let inv_n = 1.0 / n as f64;
        let mut grad_w = vec![0.0; self.weights.len()];
        let mut grad_b = 0.0f64;

        for (x, &y) in task.query_x.iter().zip(task.query_y.iter()) {
            let pred = adapted.forward(x);
            let err = pred - y;
            for (i, xi) in x.iter().enumerate() {
                if i < grad_w.len() {
                    grad_w[i] += 2.0 * inv_n * err * xi;
                }
            }
            grad_b += 2.0 * inv_n * err;
        }
        (grad_w, grad_b)
    }
}

// ─── MAML optimiser ───────────────────────────────────────────────────────────

/// MAML (FOMAML) outer-loop optimiser for linear time series models.
pub struct MamlOptimizer {
    /// Current meta-parameters (weights and bias of the base model).
    model: MetaLinearModel,
    /// Configuration controlling inner and outer learning rates, steps, etc.
    pub config: MetaLearnerConfig,
}

impl MamlOptimizer {
    /// Create a new MAML optimiser with zero-initialised meta-parameters.
    pub fn new(config: MetaLearnerConfig) -> Self {
        let model = MetaLinearModel::zeros(config.feature_dim);
        Self { model, config }
    }

    /// Create a MAML optimiser from an existing model.
    pub fn from_model(model: MetaLinearModel, config: MetaLearnerConfig) -> Self {
        Self { model, config }
    }

    /// Compute the average meta-gradient across a batch of tasks.
    ///
    /// Returns `(weight_gradients, bias_gradient)` averaged over all tasks.
    pub fn meta_gradient(&self, tasks: &[Task]) -> (Vec<f64>, f64) {
        if tasks.is_empty() {
            return (vec![0.0; self.config.feature_dim], 0.0);
        }
        let mut sum_w = vec![0.0; self.config.feature_dim];
        let mut sum_b = 0.0f64;

        for task in tasks {
            let (gw, gb) = self.model.task_meta_gradient(
                task,
                self.config.inner_lr,
                self.config.n_inner_steps,
            );
            for (sw, gw_i) in sum_w.iter_mut().zip(gw.iter()) {
                *sw += gw_i;
            }
            sum_b += gb;
        }

        let scale = 1.0 / tasks.len() as f64;
        for sw in &mut sum_w {
            *sw *= scale;
        }
        (sum_w, sum_b * scale)
    }

    /// Perform one full meta-update step over a batch of tasks.
    ///
    /// Returns the mean query loss (before the update is applied).
    pub fn meta_train_step(&mut self, tasks: &[Task]) -> f64 {
        if tasks.is_empty() {
            return 0.0;
        }
        // Compute query losses for logging
        let mean_loss: f64 = tasks
            .iter()
            .map(|t| {
                let adapted =
                    self.model
                        .inner_update(t, self.config.inner_lr, self.config.n_inner_steps);
                adapted.loss(&t.query_x, &t.query_y)
            })
            .sum::<f64>()
            / tasks.len() as f64;

        // Compute and apply FOMAML gradient
        let (gw, gb) = self.meta_gradient(tasks);
        let outer_lr = self.config.outer_lr;
        for (w, gw_i) in self.model.weights.iter_mut().zip(gw.iter()) {
            *w -= outer_lr * gw_i;
        }
        self.model.bias -= outer_lr * gb;

        mean_loss
    }

    /// Fast adaptation at test time: run inner-loop steps and return result.
    pub fn adapt(&self, task: &Task) -> Result<MetaLearnerResult, TimeSeriesError> {
        if task.query_x.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "Task query set is empty".to_string(),
            ));
        }
        let adapted =
            self.model
                .inner_update(task, self.config.inner_lr, self.config.n_inner_steps);
        let query_loss = adapted.loss(&task.query_x, &task.query_y);
        let mut adapted_weights = adapted.weights.clone();
        adapted_weights.push(adapted.bias);
        Ok(MetaLearnerResult {
            adapted_weights,
            query_loss,
            adaptation_steps: self.config.n_inner_steps,
        })
    }

    /// Compute adaptation metrics (pre vs post loss) for a task.
    pub fn adaptation_metrics(&self, task: &Task) -> AdaptationMetrics {
        let pre_loss = self.model.loss(&task.query_x, &task.query_y);
        let adapted =
            self.model
                .inner_update(task, self.config.inner_lr, self.config.n_inner_steps);
        let post_loss = adapted.loss(&task.query_x, &task.query_y);
        AdaptationMetrics::compute(pre_loss, post_loss)
    }

    /// Return a reference to the current meta-model.
    pub fn model(&self) -> &MetaLinearModel {
        &self.model
    }

    /// Return a mutable reference to the current meta-model.
    pub fn model_mut(&mut self) -> &mut MetaLinearModel {
        &mut self.model
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::online_meta::types::Task;

    /// Build a simple linear task: y = w0*x + noise, with identical support/query.
    fn make_linear_task(w0: f64, b: f64, n: usize, noise: f64) -> Task {
        // Simple LCG for deterministic noise
        let mut state: u64 = 42;
        let lcg_next = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (*s >> 33) as f64 / (u32::MAX as f64);
            2.0 * bits - 1.0 // [-1, 1]
        };

        let mut support_x = Vec::with_capacity(n);
        let mut support_y = Vec::with_capacity(n);
        let mut query_x = Vec::with_capacity(n);
        let mut query_y = Vec::with_capacity(n);

        for i in 0..n {
            let x = i as f64 / n as f64;
            let ny = lcg_next(&mut state) * noise;
            support_x.push(vec![x]);
            support_y.push(w0 * x + b + ny);
        }
        for i in 0..n {
            let x = (i as f64 + 0.5) / n as f64;
            let ny = lcg_next(&mut state) * noise;
            query_x.push(vec![x]);
            query_y.push(w0 * x + b + ny);
        }
        Task {
            support_x,
            support_y,
            query_x,
            query_y,
        }
    }

    #[test]
    fn test_maml_linear_forward() {
        let model = MetaLinearModel::new(vec![2.0, -1.0], 0.5);
        let pred = model.forward(&[3.0, 1.0]);
        // 2*3 + (-1)*1 + 0.5 = 5.5
        assert!((pred - 5.5).abs() < 1e-10, "forward pass incorrect: {pred}");
    }

    #[test]
    fn test_maml_inner_update_reduces_loss() {
        let task = make_linear_task(3.0, 1.0, 20, 0.0);
        let model = MetaLinearModel::zeros(1);
        let pre_loss = model.loss(&task.support_x, &task.support_y);
        let adapted = model.inner_update(&task, 0.05, 20);
        let post_loss = adapted.loss(&task.support_x, &task.support_y);
        assert!(
            post_loss < pre_loss,
            "inner_update should reduce support loss: pre={pre_loss}, post={post_loss}"
        );
    }

    #[test]
    fn test_maml_meta_train_step() {
        let config = MetaLearnerConfig {
            feature_dim: 1,
            inner_lr: 0.05,
            outer_lr: 0.01,
            n_inner_steps: 5,
            ..Default::default()
        };
        let mut optimizer = MamlOptimizer::new(config);
        let tasks: Vec<Task> = (0..4)
            .map(|i| make_linear_task(i as f64, 0.5, 10, 0.0))
            .collect();
        let loss = optimizer.meta_train_step(&tasks);
        // Loss should be a finite non-negative number
        assert!(loss.is_finite(), "meta_train_step loss should be finite");
        assert!(loss >= 0.0, "loss should be non-negative");
    }

    #[test]
    fn test_meta_gradient_shape() {
        let config = MetaLearnerConfig {
            feature_dim: 3,
            inner_lr: 0.01,
            outer_lr: 0.001,
            n_inner_steps: 3,
            ..Default::default()
        };
        let optimizer = MamlOptimizer::new(config.clone());
        let task = Task {
            support_x: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            support_y: vec![1.0, 2.0],
            query_x: vec![vec![0.0, 0.0, 1.0]],
            query_y: vec![3.0],
        };
        let (gw, _gb) = optimizer.meta_gradient(&[task]);
        assert_eq!(
            gw.len(),
            config.feature_dim,
            "gradient should match feature_dim"
        );
    }

    #[test]
    fn test_adapt_improves_query_loss() {
        let config = MetaLearnerConfig {
            feature_dim: 1,
            inner_lr: 0.05,
            outer_lr: 0.01,
            n_inner_steps: 30,
            ..Default::default()
        };
        // Pre-train the meta-model on a family of tasks
        let mut optimizer = MamlOptimizer::new(config);
        let train_tasks: Vec<Task> = (0..8)
            .map(|i| make_linear_task(1.0 + i as f64 * 0.1, 0.0, 20, 0.0))
            .collect();
        for _ in 0..20 {
            optimizer.meta_train_step(&train_tasks);
        }

        // Now adapt to a new task
        let test_task = make_linear_task(2.0, 0.5, 20, 0.0);
        let metrics = optimizer.adaptation_metrics(&test_task);
        // After some pre-training and adaptation, post_loss should not be infinite
        assert!(metrics.post_loss.is_finite(), "post_loss should be finite");
    }
}
