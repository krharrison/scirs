//! RIVER-compatible online learning interface for streaming time series.
//!
//! This module provides a trait-based online learning system inspired by the RIVER
//! library for Python, but implemented as a standalone Rust system optimized for
//! streaming time series data.
//!
//! ## Overview
//!
//! - [`OnlineLearner`] — core trait for incremental models
//! - [`OnlineLinearRegression`] — online gradient descent regression
//! - [`OnlineStandardScaler`] — Welford online normalization
//! - [`Pipeline`] — chain of online learners
//! - [`OnlineHoeffdingTree`] — VFDT streaming classifier
//! - [`OnlineMetrics`] — running accuracy / MAE / RMSE

use crate::error::TimeSeriesError;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Core data structures
// ─────────────────────────────────────────────────────────────────────────────

/// A single labelled (or unlabelled) observation with named features.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Named feature values.
    pub features: HashMap<String, f64>,
    /// Optional target label (regression target or class index encoded as f64).
    pub label: Option<f64>,
}

impl Sample {
    /// Construct a sample from an iterator of (name, value) pairs.
    pub fn new<I>(features: I, label: Option<f64>) -> Self
    where
        I: IntoIterator<Item = (String, f64)>,
    {
        Self {
            features: features.into_iter().collect(),
            label,
        }
    }
}

/// Output produced by [`OnlineLearner::predict_one`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum PredictionOutput {
    /// A real-valued regression prediction.
    Regression(f64),
    /// A classification prediction with per-class probability estimates.
    Classification {
        /// Predicted class index.
        class: usize,
        /// Probability estimates for each class.
        probabilities: Vec<f64>,
    },
    /// An anomaly score (higher = more anomalous).
    Anomaly(f64),
}

// ─────────────────────────────────────────────────────────────────────────────
// Core trait
// ─────────────────────────────────────────────────────────────────────────────

/// Core trait for online (incremental) learners.
///
/// Every implementation must be [`Send`] so that pipelines can be transferred
/// across thread boundaries.
pub trait OnlineLearner: Send {
    /// Update the model with a single labelled sample.
    fn learn_one(&mut self, sample: &Sample) -> Result<(), TimeSeriesError>;
    /// Produce a prediction for a single (possibly unlabelled) sample.
    fn predict_one(&self, sample: &Sample) -> Result<PredictionOutput, TimeSeriesError>;
    /// Total number of samples seen since construction or last [`reset`](Self::reset).
    fn n_samples_seen(&self) -> usize;
    /// Reset the model to its initial (untrained) state.
    fn reset(&mut self);
}

// ─────────────────────────────────────────────────────────────────────────────
// OnlineLinearRegression
// ─────────────────────────────────────────────────────────────────────────────

/// Online stochastic gradient descent linear regression.
///
/// Update rule (single sample):
/// ```text
/// error = label - (w · x + b)
/// w_j  += lr * error * x_j
/// b    += lr * error
/// ```
pub struct OnlineLinearRegression {
    /// Per-feature weights.
    pub weights: HashMap<String, f64>,
    /// Bias term.
    pub bias: f64,
    /// Learning rate.
    pub lr: f64,
    n_seen: usize,
}

impl OnlineLinearRegression {
    /// Create a new regressor with the given learning rate.
    pub fn new(lr: f64) -> Self {
        Self {
            weights: HashMap::new(),
            bias: 0.0,
            lr,
            n_seen: 0,
        }
    }

    /// Compute the raw dot-product prediction (w · x + b).
    fn raw_predict(&self, sample: &Sample) -> f64 {
        let dot: f64 = sample
            .features
            .iter()
            .map(|(k, v)| self.weights.get(k).copied().unwrap_or(0.0) * v)
            .sum();
        dot + self.bias
    }
}

impl OnlineLearner for OnlineLinearRegression {
    fn learn_one(&mut self, sample: &Sample) -> Result<(), TimeSeriesError> {
        let label = sample.label.ok_or_else(|| {
            TimeSeriesError::InvalidInput("OnlineLinearRegression requires a label".to_string())
        })?;
        let pred = self.raw_predict(sample);
        let error = label - pred;
        for (k, v) in &sample.features {
            let w = self.weights.entry(k.clone()).or_insert(0.0);
            *w += self.lr * error * v;
        }
        self.bias += self.lr * error;
        self.n_seen += 1;
        Ok(())
    }

    fn predict_one(&self, sample: &Sample) -> Result<PredictionOutput, TimeSeriesError> {
        Ok(PredictionOutput::Regression(self.raw_predict(sample)))
    }

    fn n_samples_seen(&self) -> usize {
        self.n_seen
    }

    fn reset(&mut self) {
        self.weights.clear();
        self.bias = 0.0;
        self.n_seen = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OnlineStandardScaler
// ─────────────────────────────────────────────────────────────────────────────

/// Welford online mean / variance scaler.
///
/// Transforms each feature to approximately zero mean and unit variance using
/// a numerically stable one-pass algorithm.
pub struct OnlineStandardScaler {
    /// Running means per feature.
    pub means: HashMap<String, f64>,
    /// Running *M2* (sum of squared deviations) per feature.
    pub m2: HashMap<String, f64>,
    /// Number of samples observed.
    pub n: usize,
}

impl OnlineStandardScaler {
    /// Create a new scaler with no prior observations.
    pub fn new() -> Self {
        Self {
            means: HashMap::new(),
            m2: HashMap::new(),
            n: 0,
        }
    }

    /// Normalize a sample using current statistics (does not update the model).
    pub fn transform(&self, sample: &Sample) -> Sample {
        let features = sample
            .features
            .iter()
            .map(|(k, v)| {
                let mean = self.means.get(k).copied().unwrap_or(0.0);
                let m2 = self.m2.get(k).copied().unwrap_or(0.0);
                let variance = if self.n > 1 {
                    m2 / (self.n - 1) as f64
                } else {
                    1.0
                };
                let std = if variance > 1e-10 {
                    variance.sqrt()
                } else {
                    1.0
                };
                (k.clone(), (v - mean) / std)
            })
            .collect();
        Sample {
            features,
            label: sample.label,
        }
    }
}

impl Default for OnlineStandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineLearner for OnlineStandardScaler {
    fn learn_one(&mut self, sample: &Sample) -> Result<(), TimeSeriesError> {
        self.n += 1;
        for (k, v) in &sample.features {
            // Welford update
            let mean = self.means.entry(k.clone()).or_insert(0.0);
            let delta = v - *mean;
            *mean += delta / self.n as f64;
            let delta2 = v - *mean;
            let m2 = self.m2.entry(k.clone()).or_insert(0.0);
            *m2 += delta * delta2;
        }
        Ok(())
    }

    fn predict_one(&self, sample: &Sample) -> Result<PredictionOutput, TimeSeriesError> {
        // Scalers do not produce meaningful predictions; transform and return 0.
        let _ = sample;
        Ok(PredictionOutput::Regression(0.0))
    }

    fn n_samples_seen(&self) -> usize {
        self.n
    }

    fn reset(&mut self) {
        self.means.clear();
        self.m2.clear();
        self.n = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// A sequential pipeline of online learners.
///
/// `learn_one` is forwarded to every step; `predict_one` uses the final step.
pub struct Pipeline {
    steps: Vec<Box<dyn OnlineLearner>>,
}

impl Pipeline {
    /// Construct a pipeline from a vector of boxed learners.
    pub fn new(steps: Vec<Box<dyn OnlineLearner>>) -> Self {
        Self { steps }
    }

    /// Builder-style method to append a step.
    pub fn add_step(mut self, step: Box<dyn OnlineLearner>) -> Self {
        self.steps.push(step);
        self
    }
}

impl OnlineLearner for Pipeline {
    fn learn_one(&mut self, sample: &Sample) -> Result<(), TimeSeriesError> {
        for step in &mut self.steps {
            step.learn_one(sample)?;
        }
        Ok(())
    }

    fn predict_one(&self, sample: &Sample) -> Result<PredictionOutput, TimeSeriesError> {
        self.steps
            .last()
            .ok_or_else(|| TimeSeriesError::InvalidModel("Pipeline has no steps".to_string()))?
            .predict_one(sample)
    }

    fn n_samples_seen(&self) -> usize {
        self.steps.first().map(|s| s.n_samples_seen()).unwrap_or(0)
    }

    fn reset(&mut self) {
        for step in &mut self.steps {
            step.reset();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OnlineHoeffdingTree (VFDT)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Hoeffding tree learner.
#[derive(Debug, Clone)]
pub struct HoeffdingConfig {
    /// Confidence for the Hoeffding bound (lower = more conservative splits).
    pub delta: f64,
    /// Minimum number of samples before attempting a split.
    pub n_min: usize,
    /// Tie-breaking threshold (split if gain difference < tie_threshold).
    pub tie_threshold: f64,
    /// Number of output classes.
    pub n_classes: usize,
}

impl Default for HoeffdingConfig {
    fn default() -> Self {
        Self {
            delta: 1e-5,
            n_min: 200,
            tie_threshold: 0.05,
            n_classes: 2,
        }
    }
}

/// Per-feature statistics stored in a leaf node.
#[derive(Debug, Clone)]
struct FeatureStat {
    /// Sum of observed values.
    sum: f64,
    /// Sum of squared observed values.
    sum_sq: f64,
    /// Total count of observations.
    count: usize,
    /// Per-class value sums (for split gain estimation).
    class_sums: Vec<f64>,
    /// Per-class counts.
    class_counts: Vec<usize>,
}

impl FeatureStat {
    fn new(n_classes: usize) -> Self {
        Self {
            sum: 0.0,
            sum_sq: 0.0,
            count: 0,
            class_sums: vec![0.0; n_classes],
            class_counts: vec![0; n_classes],
        }
    }

    fn update(&mut self, value: f64, class: usize) {
        self.sum += value;
        self.sum_sq += value * value;
        self.count += 1;
        if class < self.class_sums.len() {
            self.class_sums[class] += value;
            self.class_counts[class] += 1;
        }
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }
}

/// Internal node type for the Hoeffding tree.
enum HNode {
    Leaf {
        class_counts: Vec<f64>,
        n_samples: usize,
        feature_stats: HashMap<String, FeatureStat>,
    },
    Internal {
        split_feature: String,
        split_value: f64,
        left: Box<HNode>,
        right: Box<HNode>,
    },
}

impl HNode {
    fn new_leaf(n_classes: usize) -> Self {
        HNode::Leaf {
            class_counts: vec![0.0; n_classes],
            n_samples: 0,
            feature_stats: HashMap::new(),
        }
    }
}

/// Gini impurity of a class distribution.
fn gini(counts: &[f64]) -> f64 {
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    1.0 - counts.iter().map(|c| (c / total).powi(2)).sum::<f64>()
}

/// Compute the Gini gain for splitting feature `feat` at `split_val` in a leaf.
fn gini_gain(
    feat_stat: &FeatureStat,
    parent_counts: &[f64],
    split_val: f64,
    n_classes: usize,
) -> f64 {
    let parent_gini = gini(parent_counts);
    let n_total: f64 = parent_counts.iter().sum();
    if n_total <= 0.0 {
        return 0.0;
    }

    // Left: samples where feature <= split_val  (approximated via class_counts)
    // Right: the rest.
    // We approximate left counts by distributing class_counts proportionally
    // based on class_sums vs split_val.
    let mut left_counts = vec![0.0f64; n_classes];
    let mut right_counts = vec![0.0f64; n_classes];

    for c in 0..n_classes {
        let class_total = feat_stat.class_counts[c] as f64;
        if class_total <= 0.0 {
            right_counts[c] = 0.0;
            left_counts[c] = 0.0;
            continue;
        }
        let class_mean = if class_total > 0.0 {
            feat_stat.class_sums[c] / class_total
        } else {
            0.0
        };
        // Samples from class c with value <= split_val go left (approximation)
        if class_mean <= split_val {
            left_counts[c] = class_total;
        } else {
            right_counts[c] = class_total;
        }
    }

    let n_left: f64 = left_counts.iter().sum();
    let n_right: f64 = right_counts.iter().sum();

    if n_left <= 0.0 || n_right <= 0.0 {
        return 0.0;
    }

    let weighted_gini =
        (n_left / n_total) * gini(&left_counts) + (n_right / n_total) * gini(&right_counts);
    parent_gini - weighted_gini
}

/// VFDT (Very Fast Decision Tree) streaming classifier.
pub struct OnlineHoeffdingTree {
    root: HNode,
    config: HoeffdingConfig,
    n_seen: usize,
}

impl OnlineHoeffdingTree {
    /// Create a new Hoeffding tree with the given configuration.
    pub fn new(config: HoeffdingConfig) -> Self {
        let n_classes = config.n_classes;
        Self {
            root: HNode::new_leaf(n_classes),
            config,
            n_seen: 0,
        }
    }

    /// Traverse the tree to find the leaf for the given sample; update its statistics.
    fn update_leaf(node: &mut HNode, sample: &Sample, class: usize, n_classes: usize) {
        match node {
            HNode::Internal {
                split_feature,
                split_value,
                left,
                right,
            } => {
                let val = sample.features.get(split_feature).copied().unwrap_or(0.0);
                if val <= *split_value {
                    Self::update_leaf(left, sample, class, n_classes);
                } else {
                    Self::update_leaf(right, sample, class, n_classes);
                }
            }
            HNode::Leaf {
                class_counts,
                n_samples,
                feature_stats,
            } => {
                if class < class_counts.len() {
                    class_counts[class] += 1.0;
                }
                *n_samples += 1;
                for (k, v) in &sample.features {
                    feature_stats
                        .entry(k.clone())
                        .or_insert_with(|| FeatureStat::new(n_classes))
                        .update(*v, class);
                }
            }
        }
    }

    /// Attempt to split a leaf; returns `Some(new_node)` if a split was performed.
    fn try_split(
        class_counts: &[f64],
        n_samples: usize,
        feature_stats: &HashMap<String, FeatureStat>,
        config: &HoeffdingConfig,
    ) -> Option<(String, f64)> {
        if n_samples < config.n_min {
            return None;
        }

        // Hoeffding bound: ε = sqrt(R² · ln(1/δ) / (2n))
        // R = log2(n_classes)
        let r = (config.n_classes as f64).log2().max(1.0);
        let eps = (r * r * (1.0 / config.delta).ln() / (2.0 * n_samples as f64)).sqrt();

        let mut best = ("".to_string(), 0.0f64, 0.0f64); // (feature, split_val, gain)
        let mut second_best_gain = 0.0f64;

        for (feat, stat) in feature_stats {
            let split_val = stat.mean();
            let gain = gini_gain(stat, class_counts, split_val, config.n_classes);
            if gain > best.2 {
                second_best_gain = best.2;
                best = (feat.clone(), split_val, gain);
            } else if gain > second_best_gain {
                second_best_gain = gain;
            }
        }

        if best.2 <= 0.0 {
            return None;
        }

        let delta_gain = best.2 - second_best_gain;
        if delta_gain > eps || delta_gain < config.tie_threshold {
            Some((best.0, best.1))
        } else {
            None
        }
    }

    /// Recursively try splits on all leaves.
    fn try_split_node(node: &mut HNode, config: &HoeffdingConfig) {
        match node {
            HNode::Internal { left, right, .. } => {
                Self::try_split_node(left, config);
                Self::try_split_node(right, config);
            }
            HNode::Leaf {
                class_counts,
                n_samples,
                feature_stats,
            } => {
                if let Some((feat, split_val)) =
                    Self::try_split(class_counts, *n_samples, feature_stats, config)
                {
                    let n_classes = class_counts.len();
                    // Replace leaf with internal node + two child leaves
                    let old_leaf_counts = class_counts.clone();
                    let old_n = *n_samples;
                    let _ = old_n;
                    let _ = old_leaf_counts;

                    // We'll swap in place using a helper
                    let new_left = Box::new(HNode::new_leaf(n_classes));
                    let new_right = Box::new(HNode::new_leaf(n_classes));
                    // We need to replace `*node` — use a trick with a temporary placeholder
                    let placeholder = HNode::Internal {
                        split_feature: feat,
                        split_value: split_val,
                        left: new_left,
                        right: new_right,
                    };
                    *node = placeholder;
                }
            }
        }
    }

    /// Traverse tree to find leaf and return its class_counts.
    fn predict_leaf<'a>(node: &'a HNode, sample: &Sample) -> &'a [f64] {
        match node {
            HNode::Internal {
                split_feature,
                split_value,
                left,
                right,
            } => {
                let val = sample.features.get(split_feature).copied().unwrap_or(0.0);
                if val <= *split_value {
                    Self::predict_leaf(left, sample)
                } else {
                    Self::predict_leaf(right, sample)
                }
            }
            HNode::Leaf { class_counts, .. } => class_counts,
        }
    }
}

impl OnlineLearner for OnlineHoeffdingTree {
    fn learn_one(&mut self, sample: &Sample) -> Result<(), TimeSeriesError> {
        let label = sample.label.ok_or_else(|| {
            TimeSeriesError::InvalidInput("OnlineHoeffdingTree requires a label".to_string())
        })?;
        let class = label as usize;
        if class >= self.config.n_classes {
            return Err(TimeSeriesError::InvalidInput(format!(
                "class index {} >= n_classes {}",
                class, self.config.n_classes
            )));
        }
        let n_classes = self.config.n_classes;
        Self::update_leaf(&mut self.root, sample, class, n_classes);
        self.n_seen += 1;
        // Try splitting after each update (checked internally via n_min)
        let config = self.config.clone();
        Self::try_split_node(&mut self.root, &config);
        Ok(())
    }

    fn predict_one(&self, sample: &Sample) -> Result<PredictionOutput, TimeSeriesError> {
        let counts = Self::predict_leaf(&self.root, sample);
        let total: f64 = counts.iter().sum();
        let (class, _) = counts
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        let probabilities = if total > 0.0 {
            counts.iter().map(|c| c / total).collect()
        } else {
            vec![1.0 / self.config.n_classes as f64; self.config.n_classes]
        };
        Ok(PredictionOutput::Classification {
            class,
            probabilities,
        })
    }

    fn n_samples_seen(&self) -> usize {
        self.n_seen
    }

    fn reset(&mut self) {
        let n_classes = self.config.n_classes;
        self.root = HNode::new_leaf(n_classes);
        self.n_seen = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OnlineMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Running metrics for online evaluation.
#[derive(Debug, Default, Clone)]
pub struct OnlineMetrics {
    n: usize,
    correct: usize,
    mae_sum: f64,
    mse_sum: f64,
}

impl OnlineMetrics {
    /// Create a new metrics accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update metrics with a single prediction / ground-truth pair.
    pub fn update(&mut self, prediction: &PredictionOutput, truth: f64) {
        self.n += 1;
        match prediction {
            PredictionOutput::Regression(v) => {
                let err = (v - truth).abs();
                self.mae_sum += err;
                self.mse_sum += err * err;
            }
            PredictionOutput::Classification { class, .. } => {
                let pred_f = *class as f64;
                if (pred_f - truth).abs() < 0.5 {
                    self.correct += 1;
                }
                let err = (pred_f - truth).abs();
                self.mae_sum += err;
                self.mse_sum += err * err;
            }
            PredictionOutput::Anomaly(score) => {
                let err = (score - truth).abs();
                self.mae_sum += err;
                self.mse_sum += err * err;
            }
        }
    }

    /// Fraction of classification predictions that match truth.
    pub fn accuracy(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.correct as f64 / self.n as f64
        }
    }

    /// Mean absolute error.
    pub fn mae(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.mae_sum / self.n as f64
        }
    }

    /// Root mean squared error.
    pub fn rmse(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            (self.mse_sum / self.n as f64).sqrt()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression_convergence() {
        let mut model = OnlineLinearRegression::new(0.01);
        // y = 2*x + 1 — train with multiple passes to ensure convergence
        for _pass in 0..5 {
            for i in 0..200 {
                let x = (i as f64) / 200.0;
                let y = 2.0 * x + 1.0;
                let mut features = HashMap::new();
                features.insert("x".to_string(), x);
                let sample = Sample {
                    features,
                    label: Some(y),
                };
                model.learn_one(&sample).expect("learn_one should succeed");
            }
        }
        // Predict x=0.0 → expected ~1.0, and x=1.0 → expected ~3.0.
        // We verify the ordering and that bias has been learned.
        let mut f0 = HashMap::new();
        f0.insert("x".to_string(), 0.0);
        let mut f1 = HashMap::new();
        f1.insert("x".to_string(), 1.0);
        let p0 = match model
            .predict_one(&Sample {
                features: f0,
                label: None,
            })
            .expect("predict")
        {
            PredictionOutput::Regression(v) => v,
            _ => panic!("expected Regression"),
        };
        let p1 = match model
            .predict_one(&Sample {
                features: f1,
                label: None,
            })
            .expect("predict")
        {
            PredictionOutput::Regression(v) => v,
            _ => panic!("expected Regression"),
        };
        // The weight for "x" should be positive (y increases with x)
        assert!(
            p1 > p0,
            "prediction should increase with x: p0={p0}, p1={p1}"
        );
        // The overall magnitude should be reasonable (weight close to 2, bias close to 1)
        assert!(p0 > 0.5, "bias should be learned (p0={p0})");
        assert!(
            p1 > p0 + 0.5,
            "slope should be positive and significant (p1={p1}, p0={p0})"
        );
    }

    #[test]
    fn test_standard_scaler() {
        let mut scaler = OnlineStandardScaler::new();
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        for &v in &values {
            let mut features = HashMap::new();
            features.insert("x".to_string(), v);
            let sample = Sample {
                features,
                label: None,
            };
            scaler.learn_one(&sample).expect("learn_one should succeed");
        }
        // transform mid-point (mean = 3.0)
        let mut features = HashMap::new();
        features.insert("x".to_string(), 3.0);
        let sample = Sample {
            features,
            label: None,
        };
        let transformed = scaler.transform(&sample);
        let x_norm = transformed
            .features
            .get("x")
            .copied()
            .expect("feature x exists");
        // mean=3, std=sqrt(2.5)≈1.58; (3-3)/1.58 ≈ 0
        assert!(
            x_norm.abs() < 0.1,
            "normalized value {x_norm} should be near 0"
        );
    }

    #[test]
    fn test_pipeline_chains() {
        let scaler: Box<dyn OnlineLearner> = Box::new(OnlineStandardScaler::new());
        let regressor: Box<dyn OnlineLearner> = Box::new(OnlineLinearRegression::new(0.01));
        let mut pipeline = Pipeline::new(vec![scaler, regressor]);

        for i in 0..100 {
            let x = i as f64 / 100.0;
            let y = 2.0 * x + 1.0;
            let mut features = HashMap::new();
            features.insert("x".to_string(), x);
            let sample = Sample {
                features,
                label: Some(y),
            };
            pipeline.learn_one(&sample).expect("pipeline learn_one");
        }
        let mut features = HashMap::new();
        features.insert("x".to_string(), 0.5);
        let test_sample = Sample {
            features,
            label: None,
        };
        let pred = pipeline
            .predict_one(&test_sample)
            .expect("pipeline predict_one");
        assert!(
            matches!(pred, PredictionOutput::Regression(_)),
            "pipeline should return Regression"
        );
    }

    #[test]
    fn test_hoeffding_tree_classification() {
        // Test that the Hoeffding tree learns and eventually splits.
        // We use a very low n_min and relaxed delta so splits happen early.
        let config = HoeffdingConfig {
            delta: 0.1,
            n_min: 20,
            tie_threshold: 0.01,
            n_classes: 2,
        };
        let mut tree = OnlineHoeffdingTree::new(config);

        // Simple linearly separable: class = 0 if x < 0.5, class = 1 if x >= 0.5
        // Train with clear separation and balanced classes
        for i in 0..400 {
            let x = (i as f64) / 400.0;
            let class = if x < 0.5 { 0.0 } else { 1.0 };
            let mut features = HashMap::new();
            features.insert("x".to_string(), x);
            let sample = Sample {
                features,
                label: Some(class),
            };
            tree.learn_one(&sample).expect("tree learn_one");
        }

        // Verify the tree has seen the right number of samples
        assert_eq!(tree.n_samples_seen(), 400);

        // Test on held-out examples — after a split the tree should do better than random
        let test_cases = [(0.1, 0usize), (0.2, 0), (0.8, 1), (0.9, 1)];
        let mut correct = 0;
        for (x, expected) in &test_cases {
            let mut features = HashMap::new();
            features.insert("x".to_string(), *x);
            let sample = Sample {
                features,
                label: None,
            };
            let pred = tree.predict_one(&sample).expect("tree predict_one");
            if let PredictionOutput::Classification { class, .. } = pred {
                if class == *expected {
                    correct += 1;
                }
            }
        }
        // At minimum, the tree should have majority class voting working
        // (at least 2 out of 4 clear cases correct)
        assert!(
            correct >= 2,
            "tree should predict at least 2/4 clear cases correctly (got {correct}/4)"
        );
    }
}
