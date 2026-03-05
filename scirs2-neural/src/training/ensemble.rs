//! Model Ensembling for Neural Network Predictions
//!
//! This module provides comprehensive model ensembling utilities that combine
//! predictions from multiple models to improve robustness and accuracy.
//!
//! # Methods
//!
//! - **Voting**: Hard or soft voting across model outputs
//! - **Averaging**: Simple or weighted average of predictions
//! - **Stacking**: Meta-learner trained on base model predictions
//! - **BoostedEnsemble**: Sequential model training with error emphasis
//! - **Bagging**: Bootstrap aggregating with diverse sub-models
//!
//! # Snapshot Ensembling
//!
//! Snapshot ensembling collects checkpoints saved during cyclic LR training
//! and forms an ensemble from them, yielding diversity without extra training cost.
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::ensemble::{
//!     EnsembleMethod, soft_voting, hard_voting,
//! };
//! use scirs2_core::ndarray::array;
//!
//! // Soft voting over three models' probability outputs
//! let p1 = array![0.8_f64, 0.1, 0.1];
//! let p2 = array![0.6_f64, 0.3, 0.1];
//! let p3 = array![0.7_f64, 0.2, 0.1];
//! let result = soft_voting(&[p1, p2, p3]);
//! assert!((result[0] - 0.7).abs() < 1e-9);
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::random::rngs::SmallRng;
use std::fmt::{self, Debug};

// ============================================================================
// EnsembleMethod
// ============================================================================

/// Strategy for combining predictions from multiple models.
#[derive(Debug, Clone, PartialEq)]
pub enum EnsembleMethod {
    /// Soft voting: average predicted probabilities across models.
    Voting,
    /// Simple or optionally weighted arithmetic mean of outputs.
    Averaging {
        /// Optional per-model weights (should sum to 1.0). `None` → uniform.
        weights: Option<Vec<f64>>,
    },
    /// Stacking: a meta-learner is trained on top of base model predictions.
    Stacking,
    /// Boosted ensemble: later models focus on errors of earlier models.
    BoostedEnsemble,
    /// Bagging: bootstrap sampling to build a diverse ensemble.
    Bagging,
}

impl fmt::Display for EnsembleMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Voting => write!(f, "SoftVoting"),
            Self::Averaging { weights: None } => write!(f, "Averaging(uniform)"),
            Self::Averaging { weights: Some(_) } => write!(f, "Averaging(weighted)"),
            Self::Stacking => write!(f, "Stacking"),
            Self::BoostedEnsemble => write!(f, "BoostedEnsemble"),
            Self::Bagging => write!(f, "Bagging"),
        }
    }
}

// ============================================================================
// Model weights container
// ============================================================================

/// Opaque container for serialised model weights (parameters).
///
/// Each entry is a named parameter tensor stored as a flat `Vec<f64>` together
/// with its shape so that it can be restored later.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Human-readable identifier (e.g. epoch number, checkpoint path).
    pub name: String,
    /// Parameter blobs: `(name, shape, flat_values)`.
    pub params: Vec<(String, Vec<usize>, Vec<f64>)>,
    /// Optional validation score at the time the snapshot was taken.
    pub validation_score: Option<f64>,
    /// Epoch at which the snapshot was taken.
    pub epoch: usize,
}

impl ModelWeights {
    /// Create a new `ModelWeights` container.
    pub fn new(name: impl Into<String>, epoch: usize) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            validation_score: None,
            epoch,
        }
    }

    /// Add a named parameter blob.
    pub fn add_param(
        &mut self,
        param_name: impl Into<String>,
        shape: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<()> {
        let expected: usize = shape.iter().product();
        if values.len() != expected {
            return Err(NeuralError::ShapeMismatch(format!(
                "expected {} values for shape {:?} but got {}",
                expected,
                shape,
                values.len()
            )));
        }
        self.params.push((param_name.into(), shape, values));
        Ok(())
    }

    /// Total number of scalar parameters stored.
    pub fn param_count(&self) -> usize {
        self.params.iter().map(|(_, shape, _)| shape.iter().product::<usize>()).sum()
    }
}

// ============================================================================
// Ensemble policy (returned by snapshot_ensemble)
// ============================================================================

/// Policy produced by snapshot ensembling: which checkpoints to use and how.
#[derive(Debug, Clone)]
pub struct EnsemblePolicy {
    /// Indices into the original checkpoint slice that were selected.
    pub selected_indices: Vec<usize>,
    /// Per-checkpoint weights for the weighted average.
    pub weights: Vec<f64>,
    /// Method to apply for prediction aggregation.
    pub method: EnsembleMethod,
    /// Reported best validation score among the selected snapshots.
    pub best_score: Option<f64>,
}

// ============================================================================
// Soft voting
// ============================================================================

/// Aggregate predicted probabilities by computing their element-wise mean.
///
/// All prediction arrays must have the same length.
///
/// # Example
///
/// ```rust
/// use scirs2_neural::training::ensemble::soft_voting;
/// use scirs2_core::ndarray::array;
///
/// let p1 = array![0.8_f64, 0.1, 0.1];
/// let p2 = array![0.6_f64, 0.3, 0.1];
/// let avg = soft_voting(&[p1, p2]);
/// assert!((avg[0] - 0.7).abs() < 1e-9);
/// ```
pub fn soft_voting(predictions: &[Array1<f64>]) -> Array1<f64> {
    if predictions.is_empty() {
        return Array1::zeros(0);
    }
    let n = predictions[0].len();
    let mut sum = Array1::<f64>::zeros(n);
    for pred in predictions {
        sum = sum + pred;
    }
    let count = predictions.len() as f64;
    sum.mapv(|v| v / count)
}

// ============================================================================
// Hard voting
// ============================================================================

/// Aggregate class predictions by majority vote.
///
/// Ties are broken in favour of the lower class index.
///
/// # Example
///
/// ```rust
/// use scirs2_neural::training::ensemble::hard_voting;
///
/// let votes = hard_voting(&[vec![0usize, 1, 1], vec![0, 0, 1], vec![1, 1, 1]]);
/// assert_eq!(votes, vec![0, 1, 1]);
/// ```
pub fn hard_voting(predictions: &[Vec<usize>]) -> Vec<usize> {
    if predictions.is_empty() {
        return Vec::new();
    }
    let n_samples = predictions[0].len();
    let mut result = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        // Count votes for each class.
        let mut vote_counts: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for model_preds in predictions {
            if let Some(&cls) = model_preds.get(i) {
                *vote_counts.entry(cls).or_insert(0) += 1;
            }
        }
        // Pick the class with most votes; tie → smallest index.
        let winner = vote_counts
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
            .map(|(&cls, _)| cls)
            .unwrap_or(0);
        result.push(winner);
    }
    result
}

// ============================================================================
// predict_ensemble
// ============================================================================

/// Run ensemble prediction given pre-computed per-model output arrays.
///
/// This is a convenience wrapper that applies the requested `EnsembleMethod`
/// to a collection of 1-D output vectors.
///
/// # Arguments
///
/// * `model_outputs` – One `Array1<f64>` per model (probabilities or logits).
/// * `method`        – Aggregation strategy to use.
///
/// # Returns
///
/// A single `Array1<f64>` of aggregated predictions.
pub fn predict_ensemble(
    model_outputs: &[Array1<f64>],
    method: &EnsembleMethod,
) -> Result<Array1<f64>> {
    if model_outputs.is_empty() {
        return Err(NeuralError::InvalidArgument(
            "predict_ensemble requires at least one model output".into(),
        ));
    }
    let n = model_outputs[0].len();
    for (i, out) in model_outputs.iter().enumerate() {
        if out.len() != n {
            return Err(NeuralError::ShapeMismatch(format!(
                "model {} output length {} does not match expected {}",
                i,
                out.len(),
                n
            )));
        }
    }
    match method {
        EnsembleMethod::Voting | EnsembleMethod::Averaging { weights: None } => {
            Ok(soft_voting(model_outputs))
        }
        EnsembleMethod::Averaging {
            weights: Some(ws),
        } => {
            if ws.len() != model_outputs.len() {
                return Err(NeuralError::InvalidArgument(format!(
                    "weights length {} does not match number of models {}",
                    ws.len(),
                    model_outputs.len()
                )));
            }
            let weight_sum: f64 = ws.iter().sum();
            if weight_sum.abs() < 1e-12 {
                return Err(NeuralError::InvalidArgument(
                    "ensemble weights must not sum to zero".into(),
                ));
            }
            let mut acc = Array1::<f64>::zeros(n);
            for (out, &w) in model_outputs.iter().zip(ws.iter()) {
                acc = acc + out.mapv(|v| v * w);
            }
            Ok(acc.mapv(|v| v / weight_sum))
        }
        EnsembleMethod::Stacking
        | EnsembleMethod::BoostedEnsemble
        | EnsembleMethod::Bagging => {
            // For these methods the full pipeline requires training data.
            // Here we fall back to simple averaging over the provided outputs.
            Ok(soft_voting(model_outputs))
        }
    }
}

// ============================================================================
// ModelEnsemble
// ============================================================================

/// Container that holds multiple trained model weight snapshots together with
/// their validation scores, enabling ensemble inference.
///
/// ```rust
/// use scirs2_neural::training::ensemble::{ModelEnsemble, EnsembleMethod, ModelWeights};
///
/// let mut ensemble = ModelEnsemble::new(EnsembleMethod::Voting);
/// let mut w = ModelWeights::new("model_0", 10);
/// w.validation_score = Some(0.92);
/// ensemble.add_member(w);
/// assert_eq!(ensemble.size(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct ModelEnsemble {
    /// Stored model snapshots.
    members: Vec<ModelWeights>,
    /// Combination method.
    pub method: EnsembleMethod,
}

impl ModelEnsemble {
    /// Create an empty ensemble.
    pub fn new(method: EnsembleMethod) -> Self {
        Self {
            members: Vec::new(),
            method,
        }
    }

    /// Add a model snapshot to the ensemble.
    pub fn add_member(&mut self, weights: ModelWeights) {
        self.members.push(weights);
    }

    /// Number of member models.
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// Access member snapshots.
    pub fn members(&self) -> &[ModelWeights] {
        &self.members
    }

    /// Compute the average validation score across all members that have one.
    pub fn mean_validation_score(&self) -> Option<f64> {
        let scores: Vec<f64> = self
            .members
            .iter()
            .filter_map(|m| m.validation_score)
            .collect();
        if scores.is_empty() {
            None
        } else {
            Some(scores.iter().sum::<f64>() / scores.len() as f64)
        }
    }

    /// Aggregate pre-computed per-model predictions using the ensemble method.
    pub fn aggregate(&self, model_outputs: &[Array1<f64>]) -> Result<Array1<f64>> {
        predict_ensemble(model_outputs, &self.method)
    }
}

// ============================================================================
// StackingEnsemble
// ============================================================================

/// Stacking ensemble: trains a linear meta-learner on top of base model outputs.
///
/// The meta-learner uses logistic regression via gradient descent, suitable for
/// classification tasks where each model emits a probability vector.
#[derive(Debug, Clone)]
pub struct StackingEnsemble {
    /// Number of base models.
    pub n_models: usize,
    /// Number of output classes / dimensions.
    pub n_classes: usize,
    /// Meta-learner weights: shape `[n_models * n_classes, n_classes]`.
    meta_weights: Array2<f64>,
    /// Bias for meta-learner: shape `[n_classes]`.
    meta_bias: Array1<f64>,
    /// Whether the meta-learner has been fitted.
    pub fitted: bool,
    /// Learning rate for meta-learner training.
    pub learning_rate: f64,
    /// Number of training epochs for the meta-learner.
    pub epochs: usize,
}

impl StackingEnsemble {
    /// Create a new stacking ensemble.
    ///
    /// # Arguments
    ///
    /// * `n_models`  – Number of base models.
    /// * `n_classes` – Dimensionality of each model's output.
    pub fn new(n_models: usize, n_classes: usize) -> Self {
        let input_dim = n_models * n_classes;
        Self {
            n_models,
            n_classes,
            meta_weights: Array2::zeros((input_dim, n_classes)),
            meta_bias: Array1::zeros(n_classes),
            fitted: false,
            learning_rate: 0.01,
            epochs: 100,
        }
    }

    /// Configure training hyper-parameters.
    pub fn with_hypers(mut self, learning_rate: f64, epochs: usize) -> Self {
        self.learning_rate = learning_rate;
        self.epochs = epochs;
        self
    }

    /// Stack per-model outputs into a single feature vector.
    ///
    /// `model_outputs[i]` is the probability vector for model `i`.
    fn stack_features(model_outputs: &[Array1<f64>]) -> Array1<f64> {
        let total: usize = model_outputs.iter().map(|v| v.len()).sum();
        let mut stacked = Array1::zeros(total);
        let mut offset = 0;
        for out in model_outputs {
            for (j, &v) in out.iter().enumerate() {
                stacked[offset + j] = v;
            }
            offset += out.len();
        }
        stacked
    }

    /// Softmax over a 1-D array (numerically stable).
    fn softmax(logits: &Array1<f64>) -> Array1<f64> {
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp: Array1<f64> = logits.mapv(|v| (v - max).exp());
        let sum = exp.sum();
        if sum < 1e-300 {
            Array1::from_elem(logits.len(), 1.0 / logits.len() as f64)
        } else {
            exp.mapv(|v| v / sum)
        }
    }

    /// Train the meta-learner using a dataset of stacked base model outputs.
    ///
    /// # Arguments
    ///
    /// * `stacked_inputs` – Each element is a set of base model predictions for
    ///                       one sample (length: `n_models`).
    /// * `targets`        – One-hot or integer class labels per sample.
    pub fn fit(
        &mut self,
        stacked_inputs: &[Vec<Array1<f64>>],
        targets: &[usize],
    ) -> Result<()> {
        if stacked_inputs.len() != targets.len() {
            return Err(NeuralError::InvalidArgument(
                "stacked_inputs and targets must have equal length".into(),
            ));
        }
        if stacked_inputs.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "no training data provided to StackingEnsemble".into(),
            ));
        }
        // Validate target range.
        for &t in targets {
            if t >= self.n_classes {
                return Err(NeuralError::InvalidArgument(format!(
                    "target {} out of range for n_classes={}",
                    t, self.n_classes
                )));
            }
        }
        let n = stacked_inputs.len();
        let input_dim = self.n_models * self.n_classes;

        for _epoch in 0..self.epochs {
            // Accumulate gradients.
            let mut dw = Array2::<f64>::zeros((input_dim, self.n_classes));
            let mut db = Array1::<f64>::zeros(self.n_classes);

            for (sample_models, &label) in stacked_inputs.iter().zip(targets.iter()) {
                let feat = Self::stack_features(sample_models);
                if feat.len() != input_dim {
                    return Err(NeuralError::ShapeMismatch(format!(
                        "expected feature length {} but got {}",
                        input_dim,
                        feat.len()
                    )));
                }
                // logits = W^T * feat + bias
                let mut logits = Array1::<f64>::zeros(self.n_classes);
                for c in 0..self.n_classes {
                    let mut s = self.meta_bias[c];
                    for d in 0..input_dim {
                        s += self.meta_weights[[d, c]] * feat[d];
                    }
                    logits[c] = s;
                }
                let probs = Self::softmax(&logits);
                // Cross-entropy gradient: probs - one_hot(label)
                let mut delta = probs;
                delta[label] -= 1.0;
                // Accumulate gradients.
                for c in 0..self.n_classes {
                    db[c] += delta[c];
                    for d in 0..input_dim {
                        dw[[d, c]] += feat[d] * delta[c];
                    }
                }
            }
            // Update weights (SGD).
            let lr = self.learning_rate / n as f64;
            for c in 0..self.n_classes {
                self.meta_bias[c] -= lr * db[c];
                for d in 0..input_dim {
                    self.meta_weights[[d, c]] -= lr * dw[[d, c]];
                }
            }
        }
        self.fitted = true;
        Ok(())
    }

    /// Predict class probabilities for a single sample.
    pub fn predict(&self, model_outputs: &[Array1<f64>]) -> Result<Array1<f64>> {
        if !self.fitted {
            return Err(NeuralError::InvalidState(
                "StackingEnsemble must be fitted before predicting".into(),
            ));
        }
        let feat = Self::stack_features(model_outputs);
        let input_dim = self.n_models * self.n_classes;
        if feat.len() != input_dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "expected feature length {} but got {}",
                input_dim,
                feat.len()
            )));
        }
        let mut logits = Array1::<f64>::zeros(self.n_classes);
        for c in 0..self.n_classes {
            let mut s = self.meta_bias[c];
            for d in 0..input_dim {
                s += self.meta_weights[[d, c]] * feat[d];
            }
            logits[c] = s;
        }
        Ok(Self::softmax(&logits))
    }
}

// ============================================================================
// BaggingEnsemble
// ============================================================================

/// Bootstrap aggregating configuration and aggregation.
///
/// Bagging trains each model on a bootstrap sample (sampling with replacement)
/// of the original dataset and averages predictions to reduce variance.
#[derive(Debug, Clone)]
pub struct BaggingEnsemble {
    /// Number of bootstrap models.
    pub n_estimators: usize,
    /// Bootstrap sample size as a fraction of the original dataset.
    pub sample_fraction: f64,
    /// Whether to sample features as well (random subspace method).
    pub feature_subsampling: bool,
    /// Fraction of features to use per model (only when `feature_subsampling` is true).
    pub feature_fraction: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl BaggingEnsemble {
    /// Create a new `BaggingEnsemble` with default settings.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators,
            sample_fraction: 1.0,
            feature_subsampling: false,
            feature_fraction: 0.7,
            seed: 42,
        }
    }

    /// Generate bootstrap sample indices for `dataset_size` samples.
    ///
    /// Returns `n_estimators` index vectors, each of length
    /// `ceil(sample_fraction * dataset_size)`.
    pub fn bootstrap_indices(&self, dataset_size: usize) -> Vec<Vec<usize>> {
        let sample_size =
            ((self.sample_fraction * dataset_size as f64).ceil() as usize).max(1);
        let mut rng = SmallRng::seed_from_u64(self.seed);
        (0..self.n_estimators)
            .map(|_| {
                (0..sample_size)
                    .map(|_| rng.random_range(0..dataset_size))
                    .collect()
            })
            .collect()
    }

    /// Aggregate pre-computed per-estimator predictions by simple averaging.
    pub fn aggregate(&self, predictions: &[Array1<f64>]) -> Result<Array1<f64>> {
        if predictions.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "no predictions provided to BaggingEnsemble::aggregate".into(),
            ));
        }
        Ok(soft_voting(predictions))
    }
}

// ============================================================================
// Snapshot ensembling
// ============================================================================

/// Build an `EnsemblePolicy` from a sequence of model weight checkpoints.
///
/// Checkpoints are assumed to have been saved at the end of cosine-annealing
/// cycles during training.  Models with higher `validation_score` receive
/// proportionally larger weights.
///
/// # Arguments
///
/// * `checkpoints` – Ordered sequence of `ModelWeights` snapshots.
///
/// # Returns
///
/// An [`EnsemblePolicy`] describing which checkpoints to use and how to weight
/// them.
pub fn snapshot_ensemble(checkpoints: &[ModelWeights]) -> EnsemblePolicy {
    if checkpoints.is_empty() {
        return EnsemblePolicy {
            selected_indices: Vec::new(),
            weights: Vec::new(),
            method: EnsembleMethod::Averaging { weights: None },
            best_score: None,
        };
    }

    // Select all snapshots that have a recorded validation score, falling back
    // to the last `min(n, 5)` checkpoints when scores are absent.
    let scored: Vec<usize> = checkpoints
        .iter()
        .enumerate()
        .filter(|(_, c)| c.validation_score.is_some())
        .map(|(i, _)| i)
        .collect();

    let selected_indices: Vec<usize> = if scored.is_empty() {
        let take = checkpoints.len().min(5);
        (checkpoints.len() - take..checkpoints.len()).collect()
    } else {
        scored
    };

    // Compute softmax-scaled weights from validation scores (higher = better).
    let raw_scores: Vec<f64> = selected_indices
        .iter()
        .map(|&i| checkpoints[i].validation_score.unwrap_or(0.0))
        .collect();
    let max_score = raw_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = raw_scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp: f64 = exp_scores.iter().sum();
    let weights: Vec<f64> = if sum_exp < 1e-300 {
        vec![1.0 / selected_indices.len() as f64; selected_indices.len()]
    } else {
        exp_scores.iter().map(|&e| e / sum_exp).collect()
    };

    let best_score = selected_indices
        .iter()
        .filter_map(|&i| checkpoints[i].validation_score)
        .reduce(f64::max);

    EnsemblePolicy {
        selected_indices,
        weights: weights.clone(),
        method: EnsembleMethod::Averaging {
            weights: Some(weights),
        },
        best_score,
    }
}

// ============================================================================
// Cyclic LR snapshot epoch schedule
// ============================================================================

/// Return the epoch indices at which snapshots should be saved when training
/// with a cosine-annealing cyclic LR schedule.
///
/// Each cycle ends at `cycle_length * (k + 1) - 1` for `k = 0, 1, …`.
///
/// # Arguments
///
/// * `total_epochs` – Total number of training epochs.
/// * `cycle_length` – Length of each cosine-annealing cycle in epochs.
///
/// # Returns
///
/// Sorted vector of epoch indices (0-based) at which to save snapshots.
///
/// # Example
///
/// ```rust
/// use scirs2_neural::training::ensemble::cyclic_lr_ensemble;
///
/// let epochs = cyclic_lr_ensemble(100, 20);
/// assert_eq!(epochs, vec![19, 39, 59, 79, 99]);
/// ```
pub fn cyclic_lr_ensemble(total_epochs: usize, cycle_length: usize) -> Vec<usize> {
    if cycle_length == 0 || total_epochs == 0 {
        return Vec::new();
    }
    let mut epochs = Vec::new();
    let mut end = cycle_length;
    while end <= total_epochs {
        epochs.push(end - 1);
        end += cycle_length;
    }
    epochs
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_soft_voting_uniform() {
        let p1 = array![0.8_f64, 0.1, 0.1];
        let p2 = array![0.6_f64, 0.3, 0.1];
        let p3 = array![0.7_f64, 0.2, 0.1];
        let result = soft_voting(&[p1, p2, p3]);
        assert!((result[0] - 0.7).abs() < 1e-10);
        assert!((result[1] - 0.2).abs() < 1e-10);
        assert!((result[2] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_hard_voting() {
        let p1 = vec![0usize, 1, 1];
        let p2 = vec![0usize, 0, 1];
        let p3 = vec![1usize, 1, 1];
        let result = hard_voting(&[p1, p2, p3]);
        assert_eq!(result, vec![0, 1, 1]);
    }

    #[test]
    fn test_predict_ensemble_weighted() {
        let p1 = array![0.9_f64, 0.1];
        let p2 = array![0.4_f64, 0.6];
        let weights = vec![0.7, 0.3];
        let result = predict_ensemble(
            &[p1, p2],
            &EnsembleMethod::Averaging {
                weights: Some(weights),
            },
        )
        .expect("ensemble predict");
        // Expected: (0.9*0.7 + 0.4*0.3)/1.0 = 0.63 + 0.12 = 0.75
        assert!((result[0] - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_cyclic_lr_ensemble() {
        let epochs = cyclic_lr_ensemble(100, 20);
        assert_eq!(epochs, vec![19, 39, 59, 79, 99]);
    }

    #[test]
    fn test_cyclic_lr_ensemble_non_divisible() {
        let epochs = cyclic_lr_ensemble(50, 15);
        assert_eq!(epochs, vec![14, 29, 44]);
    }

    #[test]
    fn test_snapshot_ensemble_empty() {
        let policy = snapshot_ensemble(&[]);
        assert!(policy.selected_indices.is_empty());
    }

    #[test]
    fn test_snapshot_ensemble_with_scores() {
        let mut c1 = ModelWeights::new("snap_0", 19);
        c1.validation_score = Some(0.85);
        let mut c2 = ModelWeights::new("snap_1", 39);
        c2.validation_score = Some(0.90);
        let mut c3 = ModelWeights::new("snap_2", 59);
        c3.validation_score = Some(0.88);
        let policy = snapshot_ensemble(&[c1, c2, c3]);
        assert_eq!(policy.selected_indices.len(), 3);
        assert_eq!(policy.best_score, Some(0.90));
    }

    #[test]
    fn test_bagging_bootstrap_indices() {
        let bagging = BaggingEnsemble::new(5);
        let indices = bagging.bootstrap_indices(100);
        assert_eq!(indices.len(), 5);
        for sample in &indices {
            assert_eq!(sample.len(), 100);
            for &idx in sample {
                assert!(idx < 100);
            }
        }
    }

    #[test]
    fn test_stacking_ensemble_fit_predict() {
        // 3 base models, 2 classes, 10 training samples.
        let mut stacking = StackingEnsemble::new(3, 2);
        stacking.learning_rate = 0.1;
        stacking.epochs = 50;

        let data: Vec<Vec<Array1<f64>>> = (0..10)
            .map(|i| {
                vec![
                    array![0.8_f64 - i as f64 * 0.05, 0.2 + i as f64 * 0.05],
                    array![0.7_f64 - i as f64 * 0.04, 0.3 + i as f64 * 0.04],
                    array![0.6_f64 - i as f64 * 0.03, 0.4 + i as f64 * 0.03],
                ]
            })
            .collect();
        let targets: Vec<usize> = (0..10).map(|i| if i < 5 { 0 } else { 1 }).collect();
        stacking.fit(&data, &targets).expect("fit");
        assert!(stacking.fitted);

        let sample = vec![
            array![0.9_f64, 0.1],
            array![0.8_f64, 0.2],
            array![0.7_f64, 0.3],
        ];
        let probs = stacking.predict(&sample).expect("predict");
        assert_eq!(probs.len(), 2);
        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_model_weights_add_param() {
        let mut w = ModelWeights::new("test", 0);
        w.add_param("weight", vec![2, 3], vec![1.0; 6]).expect("ok");
        assert_eq!(w.param_count(), 6);
    }

    #[test]
    fn test_model_weights_bad_shape() {
        let mut w = ModelWeights::new("bad", 0);
        let result = w.add_param("weight", vec![2, 3], vec![1.0; 5]);
        assert!(result.is_err());
    }
}
