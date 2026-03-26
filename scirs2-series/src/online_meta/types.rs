//! Types for online meta-learning.

use crate::error::TimeSeriesError;

/// Configuration for MAML / Reptile meta-learners.
#[derive(Debug, Clone)]
pub struct MetaLearnerConfig {
    /// Inner (task-level) learning rate.
    pub inner_lr: f64,
    /// Outer (meta-level) learning rate.
    pub outer_lr: f64,
    /// Number of gradient steps on the support set during inner-loop adaptation.
    pub n_inner_steps: usize,
    /// Maximum number of tasks kept in the rolling buffer.
    pub task_buffer_size: usize,
    /// Minimum tasks required to trigger a meta-update.
    pub min_tasks_for_update: usize,
    /// Feature dimensionality (number of input features per sample).
    pub feature_dim: usize,
}

impl Default for MetaLearnerConfig {
    fn default() -> Self {
        Self {
            inner_lr: 0.01,
            outer_lr: 0.001,
            n_inner_steps: 5,
            task_buffer_size: 16,
            min_tasks_for_update: 4,
            feature_dim: 4,
        }
    }
}

/// A single meta-learning task consisting of support and query sets.
///
/// The support set is used for inner-loop adaptation; the query set evaluates
/// generalisation after adaptation.
#[derive(Debug, Clone)]
pub struct Task {
    /// Feature matrix for the support set (each inner Vec is one sample).
    pub support_x: Vec<Vec<f64>>,
    /// Labels for the support set.
    pub support_y: Vec<f64>,
    /// Feature matrix for the query set.
    pub query_x: Vec<Vec<f64>>,
    /// Labels for the query set.
    pub query_y: Vec<f64>,
}

impl Task {
    /// Construct a task; panics if length mismatches.
    pub fn new(
        support_x: Vec<Vec<f64>>,
        support_y: Vec<f64>,
        query_x: Vec<Vec<f64>>,
        query_y: Vec<f64>,
    ) -> Result<Self, TimeSeriesError> {
        if support_x.len() != support_y.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: support_x.len(),
                actual: support_y.len(),
            });
        }
        if query_x.len() != query_y.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: query_x.len(),
                actual: query_y.len(),
            });
        }
        Ok(Self {
            support_x,
            support_y,
            query_x,
            query_y,
        })
    }

    /// Number of support samples.
    pub fn n_support(&self) -> usize {
        self.support_y.len()
    }

    /// Number of query samples.
    pub fn n_query(&self) -> usize {
        self.query_y.len()
    }
}

/// Output of a fast-adaptation step.
#[derive(Debug, Clone)]
pub struct MetaLearnerResult {
    /// Weights of the adapted linear model (bias appended as last element).
    pub adapted_weights: Vec<f64>,
    /// MSE on the query set after adaptation.
    pub query_loss: f64,
    /// Number of inner gradient steps performed.
    pub adaptation_steps: usize,
}

/// Metrics summarising adaptation quality.
#[derive(Debug, Clone)]
pub struct AdaptationMetrics {
    /// MSE on the query set *before* adaptation.
    pub pre_loss: f64,
    /// MSE on the query set *after* adaptation.
    pub post_loss: f64,
    /// Ratio pre_loss / post_loss (>1 means adaptation helped).
    pub speedup_ratio: f64,
}

impl AdaptationMetrics {
    /// Compute metrics from pre- and post-adaptation losses.
    pub fn compute(pre_loss: f64, post_loss: f64) -> Self {
        let speedup_ratio = if post_loss.abs() < 1e-12 {
            1.0
        } else {
            pre_loss / post_loss
        };
        Self {
            pre_loss,
            post_loss,
            speedup_ratio,
        }
    }
}

// ─── shared helpers ───────────────────────────────────────────────────────────

/// Compute MSE between predictions and targets.
pub fn mse(predictions: &[f64], targets: &[f64]) -> f64 {
    if predictions.is_empty() || predictions.len() != targets.len() {
        return 0.0;
    }
    let n = predictions.len() as f64;
    predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, y)| (p - y).powi(2))
        .sum::<f64>()
        / n
}

/// Linear dot product for a weight vector (bias excluded) plus bias term.
///
/// `weights` has length `d`, `bias` is a scalar, `x` has length `d`.
pub fn linear_predict(weights: &[f64], bias: f64, x: &[f64]) -> f64 {
    weights
        .iter()
        .zip(x.iter())
        .map(|(w, xi)| w * xi)
        .sum::<f64>()
        + bias
}

/// Compute predictions for a feature matrix.
pub fn predict_all(weights: &[f64], bias: f64, xs: &[Vec<f64>]) -> Vec<f64> {
    xs.iter()
        .map(|x| linear_predict(weights, bias, x))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_construction() {
        let task = Task::new(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![1.0, 2.0],
            vec![vec![5.0, 6.0]],
            vec![3.0],
        )
        .expect("Task construction should succeed");
        assert_eq!(task.n_support(), 2);
        assert_eq!(task.n_query(), 1);
        // Support and query fields accessible
        assert_eq!(task.support_x[0][0], 1.0);
        assert_eq!(task.query_y[0], 3.0);
    }

    #[test]
    fn test_mse_zero() {
        let preds = vec![1.0, 2.0, 3.0];
        assert!((mse(&preds, &preds) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_linear_predict() {
        let w = vec![2.0, -1.0];
        let x = vec![3.0, 1.0];
        // 2*3 + (-1)*1 + 0.5 = 5.5
        let pred = linear_predict(&w, 0.5, &x);
        assert!((pred - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_adaptation_metrics() {
        let m = AdaptationMetrics::compute(4.0, 2.0);
        assert!((m.speedup_ratio - 2.0).abs() < 1e-10);
        assert!(m.post_loss < m.pre_loss);
    }
}
