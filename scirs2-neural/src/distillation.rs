//! Knowledge distillation utilities for neural networks
//!
//! This module provides tools for knowledge distillation including:
//! - Teacher-student training frameworks
//! - Various distillation loss functions
//! - Feature-based distillation
//! - Self-distillation techniques
//! - High-level `distill()` convenience function

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::ArrayStatCompat;
use scirs2_core::ndarray::{Array, ArrayD, Axis};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Knowledge distillation method
#[derive(Debug, Clone, PartialEq)]
pub enum DistillationMethod {
    /// Response-based distillation (output matching)
    ResponseBased {
        /// Temperature for softmax scaling
        temperature: f64,
        /// Weight for distillation loss
        alpha: f64,
        /// Weight for ground truth loss
        beta: f64,
    },
    /// Feature-based distillation (intermediate layer matching)
    FeatureBased {
        /// Names of feature layers to match
        feature_layers: Vec<String>,
        /// Method for adapting feature dimensions
        adaptation_method: FeatureAdaptation,
    },
    /// Attention-based distillation
    AttentionBased {
        /// Names of attention layers to match
        attention_layers: Vec<String>,
        /// Type of attention mechanism
        attention_type: AttentionType,
    },
    /// Relation-based distillation
    RelationBased {
        /// Type of relational information to distill
        relation_type: RelationType,
        /// Distance metric for comparing relations
        distance_metric: DistanceMetric,
    },
    /// Self-distillation
    SelfDistillation {
        /// Number of models in the ensemble
        ensemble_size: usize,
        /// Method for aggregating ensemble outputs
        aggregation: EnsembleAggregation,
    },
}

/// Feature adaptation methods for different sized features
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureAdaptation {
    /// Linear transformation
    Linear,
    /// Convolutional adaptation
    Convolutional {
        /// Convolution kernel size (height, width)
        kernel_size: (usize, usize),
        /// Convolution stride (height, width)
        stride: (usize, usize),
    },
    /// Attention-based adaptation
    Attention,
    /// Average pooling adaptation
    AvgPool {
        /// Average pooling size (height, width)
        pool_size: (usize, usize),
    },
}

/// Attention types for distillation
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionType {
    /// Spatial attention maps
    Spatial,
    /// Channel attention
    Channel,
    /// Self-attention matrices
    SelfAttention,
}

/// Relation types for relation-based distillation
#[derive(Debug, Clone, PartialEq)]
pub enum RelationType {
    /// Pairwise relationships between samples
    SampleWise,
    /// Channel-wise relationships
    ChannelWise,
    /// Spatial relationships
    SpatialWise,
}

/// Distance metrics for relation computation
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Cosine similarity
    Cosine,
    /// Manhattan distance
    Manhattan,
    /// KL divergence
    KLDivergence,
}

/// Ensemble aggregation methods
#[derive(Debug, Clone, PartialEq)]
pub enum EnsembleAggregation {
    /// Simple averaging
    Average,
    /// Weighted averaging
    Weighted {
        /// Weights for each ensemble member
        weights: Vec<f64>,
    },
    /// Soft voting
    SoftVoting,
}

/// Knowledge distillation trainer
pub struct DistillationTrainer<F: Float + Debug> {
    /// Distillation method
    method: DistillationMethod,
    /// Feature extractors for intermediate layers
    #[allow(dead_code)]
    feature_extractors: HashMap<String, Box<dyn Layer<F> + Send + Sync>>,
    /// Adaptation layers for feature matching
    #[allow(dead_code)]
    adaptation_layers: HashMap<String, Box<dyn Layer<F> + Send + Sync>>,
    /// Training statistics
    training_stats: DistillationStatistics<F>,
}

/// Statistics tracking for distillation training
#[derive(Debug, Clone)]
pub struct DistillationStatistics<F: Float + Debug> {
    /// Total distillation loss over time
    pub distillation_loss_history: Vec<F>,
    /// Ground truth loss over time
    pub ground_truth_loss_history: Vec<F>,
    /// Feature matching losses per layer
    pub feature_losses: HashMap<String, Vec<F>>,
    /// Teacher-student similarity metrics
    pub similarity_metrics: HashMap<String, F>,
    /// Training step
    pub current_step: usize,
}

/// Configuration for the high-level `distill` function
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// Temperature for softening probabilities
    pub temperature: f64,
    /// Weight for distillation loss (alpha)
    pub alpha: f64,
    /// Weight for hard-target loss (beta = 1 - alpha typically)
    pub beta: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate for the student model
    pub learning_rate: f64,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 3.0,
            alpha: 0.7,
            beta: 0.3,
            epochs: 10,
            learning_rate: 0.001,
        }
    }
}

/// Result of a distillation training run
#[derive(Debug, Clone)]
pub struct DistillationResult<F: Float + Debug> {
    /// Loss history per epoch (distillation + hard target combined)
    pub loss_history: Vec<F>,
    /// Final combined loss
    pub final_loss: F,
    /// Number of epochs trained
    pub epochs_trained: usize,
}

/// High-level knowledge distillation function
///
/// Performs knowledge distillation from a teacher model to a student model
/// using temperature-scaled soft targets and optional hard-target loss.
///
/// # Arguments
/// * `teacher` - The teacher model (frozen, used for inference only)
/// * `student` - The student model (will be trained)
/// * `data` - Training data as a slice of input arrays
/// * `targets` - Optional hard targets for supervised loss
/// * `config` - Distillation configuration (temperature, alpha, beta, epochs, lr)
///
/// # Returns
/// * `DistillationResult` with loss history and final loss
pub fn distill<F>(
    teacher: &dyn Layer<F>,
    student: &mut dyn Layer<F>,
    data: &[ArrayD<F>],
    targets: Option<&[ArrayD<F>]>,
    config: &DistillationConfig,
) -> Result<DistillationResult<F>>
where
    F: Float
        + Debug
        + scirs2_core::numeric::FromPrimitive
        + scirs2_core::ndarray::ScalarOperand
        + scirs2_core::numeric::NumAssign,
{
    if data.is_empty() {
        return Err(NeuralError::InvalidArchitecture(
            "Training data cannot be empty".to_string(),
        ));
    }

    let temp = F::from(config.temperature)
        .ok_or_else(|| NeuralError::InvalidArchitecture("Invalid temperature".to_string()))?;
    let alpha = F::from(config.alpha)
        .ok_or_else(|| NeuralError::InvalidArchitecture("Invalid alpha".to_string()))?;
    let beta = F::from(config.beta)
        .ok_or_else(|| NeuralError::InvalidArchitecture("Invalid beta".to_string()))?;
    let lr = F::from(config.learning_rate)
        .ok_or_else(|| NeuralError::InvalidArchitecture("Invalid learning rate".to_string()))?;

    let mut loss_history = Vec::with_capacity(config.epochs);

    for _epoch in 0..config.epochs {
        let mut epoch_loss = F::zero();
        let mut sample_count = 0usize;

        for (i, input) in data.iter().enumerate() {
            // Get teacher predictions (soft targets)
            let teacher_logits = teacher.forward(input)?;

            // Get student predictions
            let student_logits = student.forward(input)?;

            // Compute soft targets with temperature scaling
            let teacher_soft = softmax_with_temperature(&teacher_logits, temp)?;
            let student_soft = softmax_with_temperature(&student_logits, temp)?;

            // KL divergence loss (distillation loss)
            let kl_loss = kl_divergence_loss(&teacher_soft, &student_soft)?;
            let distillation_loss = kl_loss * temp * temp;

            let mut total_loss = alpha * distillation_loss;

            // Add hard-target cross-entropy loss if targets are available
            if let Some(tgts) = targets {
                if i < tgts.len() {
                    let ce_loss = cross_entropy_loss(&student_logits, &tgts[i])?;
                    total_loss += beta * ce_loss;
                }
            }

            epoch_loss += total_loss;
            sample_count += 1;
        }

        // Average loss over samples
        if sample_count > 0 {
            let count_f = F::from(sample_count).ok_or_else(|| {
                NeuralError::ComputationError("Failed to convert count".to_string())
            })?;
            epoch_loss /= count_f;
        }

        loss_history.push(epoch_loss);

        // Update student model
        student.update(lr)?;
    }

    let final_loss = loss_history.last().copied().unwrap_or(F::zero());

    Ok(DistillationResult {
        loss_history,
        final_loss,
        epochs_trained: config.epochs,
    })
}

/// Compute softmax with temperature scaling
fn softmax_with_temperature<F: Float + Debug + scirs2_core::ndarray::ScalarOperand>(
    logits: &ArrayD<F>,
    temperature: F,
) -> Result<ArrayD<F>> {
    let scaled_logits = logits / temperature;
    softmax_array(&scaled_logits)
}

/// Compute softmax along the last axis
fn softmax_array<F: Float + Debug + scirs2_core::ndarray::ScalarOperand>(
    x: &ArrayD<F>,
) -> Result<ArrayD<F>> {
    if x.ndim() == 0 {
        return Ok(x.clone());
    }
    let last_axis = x.ndim() - 1;
    let axis = Axis(last_axis);
    // Subtract max for numerical stability
    let max_vals = x.map_axis(axis, |view| {
        view.iter().cloned().fold(F::neg_infinity(), F::max)
    });
    let shifted = x - &max_vals.insert_axis(axis);
    let exp_vals = shifted.mapv(|v| v.exp());
    let sum_exp = exp_vals.sum_axis(axis);
    let result = exp_vals / &sum_exp.insert_axis(axis);
    Ok(result)
}

/// Compute KL divergence loss between target and prediction distributions
fn kl_divergence_loss<F: Float + Debug + scirs2_core::ndarray::ScalarOperand>(
    target: &ArrayD<F>,
    prediction: &ArrayD<F>,
) -> Result<F> {
    let eps = F::from(1e-8)
        .ok_or_else(|| NeuralError::ComputationError("Failed to convert epsilon".to_string()))?;
    let log_target = target.mapv(|x| (x + eps).ln());
    let log_pred = prediction.mapv(|x| (x + eps).ln());
    let kl = target * &(log_target - log_pred);
    let n = F::from(target.len())
        .ok_or_else(|| NeuralError::ComputationError("Failed to convert length".to_string()))?;
    let loss = kl.sum() / n;
    Ok(loss)
}

/// Compute cross-entropy loss between logits and targets
fn cross_entropy_loss<F: Float + Debug + scirs2_core::ndarray::ScalarOperand>(
    logits: &ArrayD<F>,
    targets: &ArrayD<F>,
) -> Result<F> {
    let eps = F::from(1e-8)
        .ok_or_else(|| NeuralError::ComputationError("Failed to convert epsilon".to_string()))?;
    let probs = softmax_array(logits)?;
    let log_probs = probs.mapv(|x| (x + eps).ln());
    let batch_size = F::from(targets.shape().first().copied().unwrap_or(1))
        .ok_or_else(|| NeuralError::ComputationError("Failed to convert batch size".to_string()))?;
    let ce = -(targets * &log_probs).sum() / batch_size;
    Ok(ce)
}

impl<
        F: Float
            + Debug
            + 'static
            + scirs2_core::numeric::FromPrimitive
            + scirs2_core::ndarray::ScalarOperand,
    > DistillationTrainer<F>
{
    /// Create a new distillation trainer
    pub fn new(method: DistillationMethod) -> Self {
        Self {
            method,
            feature_extractors: HashMap::new(),
            adaptation_layers: HashMap::new(),
            training_stats: DistillationStatistics {
                distillation_loss_history: Vec::new(),
                ground_truth_loss_history: Vec::new(),
                feature_losses: HashMap::new(),
                similarity_metrics: HashMap::new(),
                current_step: 0,
            },
        }
    }

    /// Add feature extractor for a specific layer
    pub fn add_feature_extractor(
        &mut self,
        layer_name: String,
        extractor: Box<dyn Layer<F> + Send + Sync>,
    ) {
        self.feature_extractors.insert(layer_name, extractor);
    }

    /// Add adaptation layer for feature matching
    pub fn add_adaptation_layer(
        &mut self,
        layer_name: String,
        adapter: Box<dyn Layer<F> + Send + Sync>,
    ) {
        self.adaptation_layers.insert(layer_name, adapter);
    }

    /// Compute distillation loss between teacher and student outputs
    pub fn compute_distillation_loss(
        &mut self,
        teacher_outputs: &HashMap<String, ArrayD<F>>,
        student_outputs: &HashMap<String, ArrayD<F>>,
        ground_truth: Option<&ArrayD<F>>,
    ) -> Result<F> {
        match &self.method {
            DistillationMethod::ResponseBased {
                temperature,
                alpha,
                beta,
            } => self.compute_response_loss(
                teacher_outputs,
                student_outputs,
                ground_truth,
                *temperature,
                *alpha,
                *beta,
            ),
            DistillationMethod::FeatureBased {
                feature_layers,
                adaptation_method,
            } => self.compute_feature_loss(
                teacher_outputs,
                student_outputs,
                feature_layers,
                adaptation_method,
            ),
            DistillationMethod::AttentionBased {
                attention_layers,
                attention_type,
            } => self.compute_attention_loss(
                teacher_outputs,
                student_outputs,
                attention_layers,
                attention_type,
            ),
            DistillationMethod::RelationBased {
                relation_type,
                distance_metric,
            } => self.compute_relation_loss(
                teacher_outputs,
                student_outputs,
                relation_type,
                distance_metric,
            ),
            DistillationMethod::SelfDistillation {
                ensemble_size,
                aggregation,
            } => self.compute_self_distillation_loss(student_outputs, *ensemble_size, aggregation),
        }
    }

    fn compute_response_loss(
        &mut self,
        teacher_outputs: &HashMap<String, ArrayD<F>>,
        student_outputs: &HashMap<String, ArrayD<F>>,
        ground_truth: Option<&ArrayD<F>>,
        temperature: f64,
        alpha: f64,
        beta: f64,
    ) -> Result<F> {
        let temp = F::from(temperature)
            .ok_or_else(|| NeuralError::InvalidArchitecture("Invalid temperature".to_string()))?;
        let alpha_f = F::from(alpha)
            .ok_or_else(|| NeuralError::InvalidArchitecture("Invalid alpha".to_string()))?;
        let beta_f = F::from(beta)
            .ok_or_else(|| NeuralError::InvalidArchitecture("Invalid beta".to_string()))?;

        // Get final outputs
        let teacher_logits = teacher_outputs
            .get("output")
            .ok_or_else(|| NeuralError::ComputationError("Teacher output not found".to_string()))?;
        let student_logits = student_outputs
            .get("output")
            .ok_or_else(|| NeuralError::ComputationError("Student output not found".to_string()))?;

        if teacher_logits.shape() != student_logits.shape() {
            return Err(NeuralError::DimensionMismatch(
                "Teacher and student output shapes don't match".to_string(),
            ));
        }

        // Compute soft targets using temperature scaling
        let teacher_soft = softmax_with_temperature(teacher_logits, temp)?;
        let student_soft = softmax_with_temperature(student_logits, temp)?;

        // KL divergence loss between soft targets
        let kl_loss = kl_divergence_loss(&teacher_soft, &student_soft)?;
        let distillation_loss = kl_loss * temp * temp;
        let mut total_loss = alpha_f * distillation_loss;

        // Add ground truth loss if available
        if let Some(gt) = ground_truth {
            let ce_loss = cross_entropy_loss(student_logits, gt)?;
            total_loss = total_loss + beta_f * ce_loss;
            self.training_stats.ground_truth_loss_history.push(ce_loss);
        }

        self.training_stats
            .distillation_loss_history
            .push(distillation_loss);
        self.training_stats.current_step += 1;
        Ok(total_loss)
    }

    fn compute_feature_loss(
        &self,
        teacher_outputs: &HashMap<String, ArrayD<F>>,
        student_outputs: &HashMap<String, ArrayD<F>>,
        feature_layers: &[String],
        adaptation_method: &FeatureAdaptation,
    ) -> Result<F> {
        let mut total_loss = F::zero();
        let mut layer_count = 0;
        for layer_name in feature_layers {
            if let (Some(teacher_feat), Some(student_feat)) = (
                teacher_outputs.get(layer_name),
                student_outputs.get(layer_name),
            ) {
                let adapted_student =
                    self.adapt_features(student_feat, teacher_feat, adaptation_method)?;
                let diff = teacher_feat - &adapted_student;
                let layer_loss = diff.mapv(|x| x * x).mean_or(F::zero());
                total_loss = total_loss + layer_loss;
                layer_count += 1;
            }
        }
        if layer_count > 0 {
            let count_f = F::from(layer_count).ok_or_else(|| {
                NeuralError::ComputationError("Failed to convert layer_count".to_string())
            })?;
            total_loss = total_loss / count_f;
        }
        Ok(total_loss)
    }

    fn compute_attention_loss(
        &self,
        teacher_outputs: &HashMap<String, ArrayD<F>>,
        student_outputs: &HashMap<String, ArrayD<F>>,
        attention_layers: &[String],
        attention_type: &AttentionType,
    ) -> Result<F> {
        let mut total_loss = F::zero();
        let mut layer_count = 0;
        for layer_name in attention_layers {
            if let (Some(teacher_feat), Some(student_feat)) = (
                teacher_outputs.get(layer_name),
                student_outputs.get(layer_name),
            ) {
                let teacher_attention =
                    self.compute_attention_maps(teacher_feat, attention_type)?;
                let student_attention =
                    self.compute_attention_maps(student_feat, attention_type)?;
                let teacher_norm = self.normalize_attention(&teacher_attention)?;
                let student_norm = self.normalize_attention(&student_attention)?;
                let diff = teacher_norm - student_norm;
                let layer_loss = diff.mapv(|x| x * x).mean_or(F::zero());
                total_loss = total_loss + layer_loss;
                layer_count += 1;
            }
        }
        if layer_count > 0 {
            let count_f = F::from(layer_count).ok_or_else(|| {
                NeuralError::ComputationError("Failed to convert layer_count".to_string())
            })?;
            total_loss = total_loss / count_f;
        }
        Ok(total_loss)
    }

    fn compute_relation_loss(
        &self,
        teacher_outputs: &HashMap<String, ArrayD<F>>,
        student_outputs: &HashMap<String, ArrayD<F>>,
        relation_type: &RelationType,
        distance_metric: &DistanceMetric,
    ) -> Result<F> {
        let teacher_feat = teacher_outputs
            .get("features")
            .or_else(|| teacher_outputs.get("output"))
            .ok_or_else(|| {
                NeuralError::ComputationError("Teacher features not found".to_string())
            })?;
        let student_feat = student_outputs
            .get("features")
            .or_else(|| student_outputs.get("output"))
            .ok_or_else(|| {
                NeuralError::ComputationError("Student features not found".to_string())
            })?;

        let teacher_relations =
            self.compute_relations(teacher_feat, relation_type, distance_metric)?;
        let student_relations =
            self.compute_relations(student_feat, relation_type, distance_metric)?;

        let diff = teacher_relations - student_relations;
        let loss = diff.mapv(|x| x * x).mean_or(F::zero());
        Ok(loss)
    }

    fn compute_self_distillation_loss(
        &self,
        student_outputs: &HashMap<String, ArrayD<F>>,
        ensemble_size: usize,
        aggregation: &EnsembleAggregation,
    ) -> Result<F> {
        if ensemble_size < 2 {
            return Ok(F::zero());
        }
        let mut ensemble_outputs = Vec::new();
        for i in 0..ensemble_size {
            let key = format!("output_{}", i);
            if let Some(output) = student_outputs.get(&key) {
                ensemble_outputs.push(output);
            }
        }
        if ensemble_outputs.len() < 2 {
            return Ok(F::zero());
        }

        let ensemble_pred = self.aggregate_ensemble(&ensemble_outputs, aggregation)?;
        let mut total_loss = F::zero();
        for output in &ensemble_outputs {
            let kl_loss = kl_divergence_loss(&ensemble_pred, output)?;
            total_loss = total_loss + kl_loss;
        }
        let count_f = F::from(ensemble_outputs.len()).ok_or_else(|| {
            NeuralError::ComputationError("Failed to convert ensemble size".to_string())
        })?;
        total_loss = total_loss / count_f;
        Ok(total_loss)
    }

    fn adapt_features(
        &self,
        student_feat: &ArrayD<F>,
        teacher_feat: &ArrayD<F>,
        method: &FeatureAdaptation,
    ) -> Result<ArrayD<F>> {
        if student_feat.shape() == teacher_feat.shape() {
            return Ok(student_feat.clone());
        }
        match method {
            FeatureAdaptation::Linear => {
                if student_feat.len() == teacher_feat.len() {
                    Ok(student_feat
                        .clone()
                        .into_shape_with_order(teacher_feat.raw_dim())
                        .map_err(|e| {
                            NeuralError::ComputationError(format!("Reshape failed: {}", e))
                        })?
                        .to_owned())
                } else {
                    let target_shape = teacher_feat.raw_dim();
                    let mut adapted = Array::zeros(target_shape);
                    let min_size = student_feat.len().min(teacher_feat.len());
                    for (i, &val) in student_feat.iter().take(min_size).enumerate() {
                        if i < adapted.len() {
                            adapted[i] = val;
                        }
                    }
                    Ok(adapted)
                }
            }
            _ => {
                // For other adaptation methods, return student features as-is
                Ok(student_feat.clone())
            }
        }
    }

    fn compute_attention_maps(
        &self,
        features: &ArrayD<F>,
        attention_type: &AttentionType,
    ) -> Result<ArrayD<F>> {
        match attention_type {
            AttentionType::Spatial => {
                if features.ndim() >= 3 {
                    let spatial_map = features.sum_axis(Axis(1));
                    Ok(spatial_map)
                } else {
                    Ok(features.clone())
                }
            }
            AttentionType::Channel => {
                if features.ndim() >= 3 {
                    let mut channel_map = features.clone();
                    for _ in 2..features.ndim() {
                        channel_map = channel_map
                            .mean_axis(Axis(channel_map.ndim() - 1))
                            .ok_or_else(|| {
                                NeuralError::ComputationError("Failed to compute mean".to_string())
                            })?;
                    }
                    Ok(channel_map)
                } else {
                    Ok(features.clone())
                }
            }
            AttentionType::SelfAttention => Ok(features.clone()),
        }
    }

    fn normalize_attention(&self, attention: &ArrayD<F>) -> Result<ArrayD<F>> {
        let sum = attention.sum();
        if sum > F::zero() {
            Ok(attention / sum)
        } else {
            Ok(attention.clone())
        }
    }

    fn compute_relations(
        &self,
        features: &ArrayD<F>,
        relation_type: &RelationType,
        distance_metric: &DistanceMetric,
    ) -> Result<ArrayD<F>> {
        match relation_type {
            RelationType::SampleWise => self.compute_sample_relations(features, distance_metric),
            RelationType::ChannelWise => self.compute_channel_relations(features, distance_metric),
            RelationType::SpatialWise => self.compute_spatial_relations(features, distance_metric),
        }
    }

    fn compute_sample_relations(
        &self,
        features: &ArrayD<F>,
        metric: &DistanceMetric,
    ) -> Result<ArrayD<F>> {
        let batch_size = features.shape()[0];
        let mut relations = Array::zeros((batch_size, batch_size));
        for i in 0..batch_size {
            for j in 0..batch_size {
                let feat_i = features.slice(scirs2_core::ndarray::s![i, ..]);
                let feat_j = features.slice(scirs2_core::ndarray::s![j, ..]);
                let distance = match metric {
                    DistanceMetric::Euclidean => {
                        let diff = &feat_i - &feat_j;
                        diff.mapv(|x| x * x).sum().sqrt()
                    }
                    DistanceMetric::Cosine => {
                        let dot = (&feat_i * &feat_j).sum();
                        let norm_i = feat_i.mapv(|x| x * x).sum().sqrt();
                        let norm_j = feat_j.mapv(|x| x * x).sum().sqrt();
                        let denom = norm_i * norm_j;
                        if denom > F::zero() {
                            dot / denom
                        } else {
                            F::zero()
                        }
                    }
                    DistanceMetric::Manhattan => {
                        let diff = &feat_i - &feat_j;
                        diff.mapv(|x| x.abs()).sum()
                    }
                    DistanceMetric::KLDivergence => {
                        let eps = F::from(1e-8).unwrap_or(F::zero());
                        let p = feat_i.mapv(|x| x + eps);
                        let q = feat_j.mapv(|x| x + eps);
                        let p_sum = p.sum();
                        let q_sum = q.sum();
                        if p_sum > F::zero() && q_sum > F::zero() {
                            let p_norm = &p / p_sum;
                            let q_norm = &q / q_sum;
                            (p_norm.clone() * (p_norm.mapv(|x| x.ln()) - q_norm.mapv(|x| x.ln())))
                                .sum()
                        } else {
                            F::zero()
                        }
                    }
                };
                relations[[i, j]] = distance;
            }
        }
        Ok(relations.into_dyn())
    }

    fn compute_channel_relations(
        &self,
        features: &ArrayD<F>,
        _metric: &DistanceMetric,
    ) -> Result<ArrayD<F>> {
        if features.ndim() < 2 {
            return Ok(features.clone());
        }
        let channels = features.shape()[1];
        let relations = Array::<F, _>::eye(channels);
        Ok(relations.into_dyn())
    }

    fn compute_spatial_relations(
        &self,
        features: &ArrayD<F>,
        _metric: &DistanceMetric,
    ) -> Result<ArrayD<F>> {
        Ok(features.clone())
    }

    fn aggregate_ensemble(
        &self,
        outputs: &[&ArrayD<F>],
        method: &EnsembleAggregation,
    ) -> Result<ArrayD<F>> {
        if outputs.is_empty() {
            return Err(NeuralError::ComputationError("Empty ensemble".to_string()));
        }
        match method {
            EnsembleAggregation::Average => {
                let mut sum = outputs[0].clone();
                for output in outputs.iter().skip(1) {
                    sum = sum + *output;
                }
                let n = F::from(outputs.len()).ok_or_else(|| {
                    NeuralError::ComputationError("Failed to convert length".to_string())
                })?;
                Ok(sum / n)
            }
            EnsembleAggregation::Weighted { weights } => {
                if weights.len() != outputs.len() {
                    return Err(NeuralError::InvalidArchitecture(
                        "Weight count doesn't match ensemble size".to_string(),
                    ));
                }
                let w0 = F::from(weights[0]).ok_or_else(|| {
                    NeuralError::ComputationError("Failed to convert weight".to_string())
                })?;
                let mut result = outputs[0] * w0;
                for (output, &weight) in outputs.iter().zip(weights.iter()).skip(1) {
                    let w = F::from(weight).ok_or_else(|| {
                        NeuralError::ComputationError("Failed to convert weight".to_string())
                    })?;
                    result = result + (*output * w);
                }
                Ok(result)
            }
            EnsembleAggregation::SoftVoting => {
                let mut result = outputs[0].clone().to_owned();
                result.fill(F::zero());
                for output in outputs {
                    result = result + *output;
                }
                let n = F::from(outputs.len()).ok_or_else(|| {
                    NeuralError::ComputationError("Failed to convert length".to_string())
                })?;
                Ok(result / n)
            }
        }
    }

    /// Get training statistics
    pub fn get_statistics(&self) -> &DistillationStatistics<F> {
        &self.training_stats
    }

    /// Reset training statistics
    pub fn reset_statistics(&mut self) {
        self.training_stats = DistillationStatistics {
            distillation_loss_history: Vec::new(),
            ground_truth_loss_history: Vec::new(),
            feature_losses: HashMap::new(),
            similarity_metrics: HashMap::new(),
            current_step: 0,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_response_based_distillation() {
        let method = DistillationMethod::ResponseBased {
            temperature: 3.0,
            alpha: 0.7,
            beta: 0.3,
        };
        let mut trainer = DistillationTrainer::<f64>::new(method);
        let mut teacher_outputs = HashMap::new();
        let mut student_outputs = HashMap::new();
        teacher_outputs.insert(
            "output".to_string(),
            Array2::from_shape_vec((2, 3), vec![2.0, 1.0, 0.5, 1.5, 2.5, 1.0])
                .expect("Test: array creation")
                .into_dyn(),
        );
        student_outputs.insert(
            "output".to_string(),
            Array2::from_shape_vec((2, 3), vec![1.8, 1.2, 0.6, 1.4, 2.3, 1.1])
                .expect("Test: array creation")
                .into_dyn(),
        );
        let loss = trainer.compute_distillation_loss(&teacher_outputs, &student_outputs, None);
        assert!(loss.is_ok());
        assert!(loss.expect("Test: loss computation") > 0.0);
    }

    #[test]
    fn test_feature_based_distillation() {
        let method = DistillationMethod::FeatureBased {
            feature_layers: vec!["layer1".to_string(), "layer2".to_string()],
            adaptation_method: FeatureAdaptation::Linear,
        };
        let mut trainer = DistillationTrainer::<f64>::new(method);
        let mut teacher_outputs = HashMap::new();
        let mut student_outputs = HashMap::new();
        teacher_outputs.insert(
            "layer1".to_string(),
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                .expect("Test: array creation")
                .into_dyn(),
        );
        student_outputs.insert(
            "layer1".to_string(),
            Array2::from_shape_vec((2, 4), vec![1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9])
                .expect("Test: array creation")
                .into_dyn(),
        );
        let loss = trainer.compute_distillation_loss(&teacher_outputs, &student_outputs, None);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_attention_based_distillation() {
        let method = DistillationMethod::AttentionBased {
            attention_layers: vec!["conv1".to_string()],
            attention_type: AttentionType::Spatial,
        };
        let mut trainer = DistillationTrainer::<f64>::new(method);
        let mut teacher_outputs = HashMap::new();
        let mut student_outputs = HashMap::new();
        teacher_outputs.insert(
            "conv1".to_string(),
            Array::from_shape_vec(
                scirs2_core::ndarray::IxDyn(&[1, 2, 4]),
                (0..8).map(|x| x as f64).collect(),
            )
            .expect("Test: array creation"),
        );
        student_outputs.insert(
            "conv1".to_string(),
            Array::from_shape_vec(
                scirs2_core::ndarray::IxDyn(&[1, 2, 4]),
                (0..8).map(|x| x as f64 + 0.1).collect(),
            )
            .expect("Test: array creation"),
        );
        let loss = trainer.compute_distillation_loss(&teacher_outputs, &student_outputs, None);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_softmax_with_temperature_fn() {
        let logits = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5])
            .expect("Test: array creation")
            .into_dyn();
        let temperature = 2.0;
        let result = softmax_with_temperature(&logits, temperature);
        assert!(result.is_ok());
        let softmax_output = result.expect("Test: softmax");
        // Check that probabilities sum to 1 for each sample
        for i in 0..2 {
            let row_sum: f64 = (0..3).map(|j| softmax_output[[i, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_kl_divergence_loss_fn() {
        let target = Array2::from_shape_vec((2, 3), vec![0.7, 0.2, 0.1, 0.3, 0.4, 0.3])
            .expect("Test: array creation")
            .into_dyn();
        let prediction = Array2::from_shape_vec((2, 3), vec![0.6, 0.3, 0.1, 0.2, 0.5, 0.3])
            .expect("Test: array creation")
            .into_dyn();
        let loss = kl_divergence_loss(&target, &prediction);
        assert!(loss.is_ok());
        assert!(loss.expect("Test: KL loss") >= 0.0);
    }

    #[test]
    fn test_ensemble_aggregation() {
        let trainer = DistillationTrainer::<f64>::new(DistillationMethod::SelfDistillation {
            ensemble_size: 3,
            aggregation: EnsembleAggregation::Average,
        });
        let output1 = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Test: array creation")
            .into_dyn();
        let output2 = Array2::from_shape_vec((2, 3), vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1])
            .expect("Test: array creation")
            .into_dyn();
        let output3 = Array2::from_shape_vec((2, 3), vec![0.9, 1.9, 2.9, 3.9, 4.9, 5.9])
            .expect("Test: array creation")
            .into_dyn();
        let outputs = vec![&output1, &output2, &output3];
        let result = trainer.aggregate_ensemble(&outputs, &EnsembleAggregation::Average);
        assert!(result.is_ok());
        let avg_output = result.expect("Test: ensemble aggregation");
        assert_eq!(avg_output.shape(), output1.shape());
        assert!((avg_output[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((avg_output[[1, 2]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_distillation_statistics() {
        let method = DistillationMethod::ResponseBased {
            temperature: 3.0,
            alpha: 0.7,
            beta: 0.3,
        };
        let mut trainer = DistillationTrainer::<f64>::new(method);
        let mut teacher_outputs = HashMap::new();
        let mut student_outputs = HashMap::new();
        teacher_outputs.insert(
            "output".to_string(),
            Array2::from_shape_vec((2, 3), vec![2.0, 1.0, 0.5, 1.5, 2.5, 1.0])
                .expect("Test: array creation")
                .into_dyn(),
        );
        student_outputs.insert(
            "output".to_string(),
            Array2::from_shape_vec((2, 3), vec![1.8, 1.2, 0.6, 1.4, 2.3, 1.1])
                .expect("Test: array creation")
                .into_dyn(),
        );
        for _ in 0..3 {
            let _ = trainer.compute_distillation_loss(&teacher_outputs, &student_outputs, None);
        }
        let stats = trainer.get_statistics();
        assert_eq!(stats.distillation_loss_history.len(), 3);
        assert_eq!(stats.current_step, 3);
    }

    #[test]
    fn test_distillation_config_default() {
        let config = DistillationConfig::default();
        assert!((config.temperature - 3.0).abs() < 1e-10);
        assert!((config.alpha - 0.7).abs() < 1e-10);
        assert!((config.beta - 0.3).abs() < 1e-10);
        assert_eq!(config.epochs, 10);
    }

    #[test]
    fn test_weighted_ensemble_aggregation() {
        let trainer = DistillationTrainer::<f64>::new(DistillationMethod::SelfDistillation {
            ensemble_size: 2,
            aggregation: EnsembleAggregation::Average,
        });
        let output1 = Array2::from_shape_vec((1, 2), vec![1.0, 0.0])
            .expect("Test: array creation")
            .into_dyn();
        let output2 = Array2::from_shape_vec((1, 2), vec![0.0, 1.0])
            .expect("Test: array creation")
            .into_dyn();
        let outputs = vec![&output1, &output2];
        let weights = EnsembleAggregation::Weighted {
            weights: vec![0.8, 0.2],
        };
        let result = trainer.aggregate_ensemble(&outputs, &weights);
        assert!(result.is_ok());
        let weighted = result.expect("Test: weighted ensemble");
        assert!((weighted[[0, 0]] - 0.8).abs() < 1e-10);
        assert!((weighted[[0, 1]] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_cross_entropy_loss_fn() {
        let logits = Array2::from_shape_vec((2, 3), vec![2.0, 1.0, 0.1, 0.1, 3.0, 0.1])
            .expect("Test: array creation")
            .into_dyn();
        let targets = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            .expect("Test: array creation")
            .into_dyn();
        let loss = cross_entropy_loss(&logits, &targets);
        assert!(loss.is_ok());
        let loss_val = loss.expect("Test: CE loss");
        assert!(loss_val > 0.0);
        assert!(loss_val.is_finite());
    }

    #[test]
    fn test_reset_statistics() {
        let method = DistillationMethod::ResponseBased {
            temperature: 3.0,
            alpha: 0.7,
            beta: 0.3,
        };
        let mut trainer = DistillationTrainer::<f64>::new(method);
        let mut teacher_outputs = HashMap::new();
        let mut student_outputs = HashMap::new();
        teacher_outputs.insert(
            "output".to_string(),
            Array2::from_shape_vec((1, 2), vec![1.0, 0.5])
                .expect("Test: array creation")
                .into_dyn(),
        );
        student_outputs.insert(
            "output".to_string(),
            Array2::from_shape_vec((1, 2), vec![0.8, 0.6])
                .expect("Test: array creation")
                .into_dyn(),
        );
        let _ = trainer.compute_distillation_loss(&teacher_outputs, &student_outputs, None);
        assert_eq!(trainer.get_statistics().current_step, 1);

        trainer.reset_statistics();
        assert_eq!(trainer.get_statistics().current_step, 0);
        assert!(trainer
            .get_statistics()
            .distillation_loss_history
            .is_empty());
    }

    #[test]
    fn test_relation_based_distillation() {
        let method = DistillationMethod::RelationBased {
            relation_type: RelationType::SampleWise,
            distance_metric: DistanceMetric::Euclidean,
        };
        let mut trainer = DistillationTrainer::<f64>::new(method);
        let mut teacher_outputs = HashMap::new();
        let mut student_outputs = HashMap::new();
        teacher_outputs.insert(
            "output".to_string(),
            Array2::from_shape_vec(
                (3, 4),
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
            )
            .expect("Test: array creation")
            .into_dyn(),
        );
        student_outputs.insert(
            "output".to_string(),
            Array2::from_shape_vec(
                (3, 4),
                vec![
                    1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1,
                ],
            )
            .expect("Test: array creation")
            .into_dyn(),
        );
        let loss = trainer.compute_distillation_loss(&teacher_outputs, &student_outputs, None);
        assert!(loss.is_ok());
    }
}
