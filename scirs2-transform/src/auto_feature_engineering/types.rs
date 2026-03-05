//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, TransformError};
use scirs2_core::ndarray::{Array1, ArrayStatCompat, ArrayView1, ArrayView2};
use scirs2_core::validation::check_not_empty;
use std::collections::HashMap;
#[cfg(feature = "auto-feature-engineering")]
use std::collections::VecDeque;

/// Performance record for historical analysis
#[cfg(feature = "auto-feature-engineering")]
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    /// Dataset meta-features
    meta_features: DatasetMetaFeatures,
    /// Applied transformations
    transformations: Vec<TransformationConfig>,
    /// Performance metrics
    metrics: PerformanceMetrics,
    /// Computational cost
    computational_cost: f64,
    /// Timestamp
    timestamp: u64,
}
/// Experience tuple for reinforcement learning
#[cfg(feature = "auto-feature-engineering")]
#[derive(Debug, Clone)]
pub struct Experience {
    /// State representation (meta-features)
    state: Vec<f64>,
    /// Action taken (transformation choice)
    action: usize,
    /// Reward received (performance improvement)
    reward: f64,
    /// Next state
    next_state: Vec<f64>,
    /// Whether episode terminated
    done: bool,
}
/// Available transformation types for automated selection
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TransformationType {
    /// Standardization (Z-score normalization)
    StandardScaler,
    /// Min-max scaling
    MinMaxScaler,
    /// Robust scaling using median and IQR
    RobustScaler,
    /// Power transformation (Box-Cox/Yeo-Johnson)
    PowerTransformer,
    /// Polynomial feature generation
    PolynomialFeatures,
    /// Principal Component Analysis
    PCA,
    /// Feature selection based on variance
    VarianceThreshold,
    /// Quantile transformation
    QuantileTransformer,
    /// Binary encoding for categorical features
    BinaryEncoder,
    /// Target encoding
    TargetEncoder,
}
/// Multi-objective optimization weights
#[cfg(feature = "auto-feature-engineering")]
#[derive(Debug, Clone)]
pub struct FeatureOptimizationWeights {
    /// Weight for prediction performance
    pub performance_weight: f64,
    /// Weight for computational efficiency
    pub efficiency_weight: f64,
    /// Weight for model interpretability
    pub interpretability_weight: f64,
    /// Weight for robustness
    pub robustness_weight: f64,
}
/// Performance metrics for multi-objective optimization
#[cfg(feature = "auto-feature-engineering")]
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Prediction accuracy/score
    accuracy: f64,
    /// Training time in seconds
    training_time: f64,
    /// Memory usage in MB
    memory_usage: f64,
    /// Model complexity score
    complexity_score: f64,
    /// Cross-validation score
    cv_score: f64,
}
/// Multi-objective recommendation system (placeholder)
pub struct MultiObjectiveRecommendation;
/// Advanced meta-learning system for feature engineering (placeholder)
pub struct AdvancedMetaLearningSystem;
/// Enhanced meta-features for advanced analysis (placeholder)
pub struct EnhancedMetaFeatures;
/// Configuration for a transformation with its parameters
#[derive(Debug, Clone)]
pub struct TransformationConfig {
    /// Type of transformation to apply
    pub transformation_type: TransformationType,
    /// Parameters for the transformation
    pub parameters: HashMap<String, f64>,
    /// Expected performance score for this transformation
    pub expected_performance: f64,
}
/// A single dense (fully-connected) layer in a pure Rust neural network.
///
/// Stores weights as a flat `Vec<f64>` in row-major order (output_dim x input_dim)
/// and bias as `Vec<f64>` of length output_dim.
#[cfg(feature = "auto-feature-engineering")]
#[derive(Debug, Clone)]
pub struct DenseLayer {
    /// Weight matrix stored in row-major order: weights[row * input_dim + col]
    /// Shape: (output_dim, input_dim)
    weights: Vec<f64>,
    /// Bias vector of length output_dim
    biases: Vec<f64>,
    /// Number of input features
    input_dim: usize,
    /// Number of output features
    output_dim: usize,
}

#[cfg(feature = "auto-feature-engineering")]
impl DenseLayer {
    /// Create a new dense layer with He initialization.
    ///
    /// He initialization uses N(0, sqrt(2/fan_in)) which works well with ReLU activations.
    /// Uses a simple deterministic LCG seeded per layer to ensure reproducibility.
    fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let n_weights = input_dim * output_dim;
        let mut weights = Vec::with_capacity(n_weights);
        let std_dev = (2.0 / input_dim as f64).sqrt();

        // Simple deterministic PRNG (xoshiro-style splitmix64) for reproducible init
        let mut state = seed;
        for _ in 0..n_weights {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Box-Muller transform for normal distribution
            let u1 = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            let u1_clamped = u1.max(1e-15); // avoid ln(0)
            let z = (-2.0 * u1_clamped.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            weights.push(z * std_dev);
        }

        let biases = vec![0.0; output_dim];

        DenseLayer {
            weights,
            biases,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass: y = Wx + b
    ///
    /// input: slice of length input_dim
    /// output: Vec of length output_dim
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut output = Vec::with_capacity(self.output_dim);
        for row in 0..self.output_dim {
            let offset = row * self.input_dim;
            let mut sum = self.biases[row];
            for col in 0..self.input_dim {
                sum += self.weights[offset + col] * input[col];
            }
            output.push(sum);
        }
        output
    }

    /// Backward pass for a dense layer.
    ///
    /// Given the gradient of the loss w.r.t. this layer's output (d_output),
    /// and the input that was fed during forward pass, compute:
    /// - gradient w.r.t. weights (accumulated into grad_weights)
    /// - gradient w.r.t. biases (accumulated into grad_biases)
    /// - gradient w.r.t. input (returned for chain rule to previous layer)
    fn backward(
        &self,
        d_output: &[f64],
        input: &[f64],
        grad_weights: &mut [f64],
        grad_biases: &mut [f64],
    ) -> Vec<f64> {
        // grad_biases[j] += d_output[j]
        for j in 0..self.output_dim {
            grad_biases[j] += d_output[j];
        }

        // grad_weights[j * input_dim + i] += d_output[j] * input[i]
        for j in 0..self.output_dim {
            let offset = j * self.input_dim;
            for i in 0..self.input_dim {
                grad_weights[offset + i] += d_output[j] * input[i];
            }
        }

        // d_input[i] = sum_j(d_output[j] * weights[j * input_dim + i])
        let mut d_input = vec![0.0; self.input_dim];
        for j in 0..self.output_dim {
            let offset = j * self.input_dim;
            for i in 0..self.input_dim {
                d_input[i] += d_output[j] * self.weights[offset + i];
            }
        }
        d_input
    }

    /// Apply SGD update: w -= lr * grad
    fn sgd_update(&mut self, grad_weights: &[f64], grad_biases: &[f64], learning_rate: f64) {
        for (w, &gw) in self.weights.iter_mut().zip(grad_weights.iter()) {
            *w -= learning_rate * gw;
        }
        for (b, &gb) in self.biases.iter_mut().zip(grad_biases.iter()) {
            *b -= learning_rate * gb;
        }
    }
}

/// Activation function types supported by the pure Rust neural network.
#[cfg(feature = "auto-feature-engineering")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Activation {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Softmax across all outputs (for the final layer)
    Softmax,
}

/// A pure Rust feed-forward neural network.
///
/// Implements a multi-layer perceptron with ReLU hidden activations and
/// a softmax output layer. Supports forward pass, analytical backpropagation,
/// and SGD training -- all in pure Rust with no external C/Fortran dependencies.
///
/// Architecture: Linear(10,64) -> ReLU -> Linear(64,32) -> ReLU ->
///               Linear(32,16) -> ReLU -> Linear(16,10) -> Softmax
#[cfg(feature = "auto-feature-engineering")]
#[derive(Debug, Clone)]
pub struct PureRustNN {
    /// Dense layers in order
    layers: Vec<DenseLayer>,
    /// Activation function after each layer
    activations: Vec<Activation>,
}

#[cfg(feature = "auto-feature-engineering")]
impl PureRustNN {
    /// Create a new neural network with the specified layer sizes and activations.
    ///
    /// `layer_sizes` is a slice like `[10, 64, 32, 16, 10]` meaning:
    ///   layer 0: 10 -> 64
    ///   layer 1: 64 -> 32
    ///   layer 2: 32 -> 16
    ///   layer 3: 16 -> 10
    ///
    /// The last activation is always Softmax; all others are ReLU.
    fn new(layer_sizes: &[usize]) -> Result<Self> {
        if layer_sizes.len() < 2 {
            return Err(TransformError::InvalidInput(
                "Neural network needs at least 2 layer sizes (input and output)".to_string(),
            ));
        }

        let n_layers = layer_sizes.len() - 1;
        let mut layers = Vec::with_capacity(n_layers);
        let mut activations = Vec::with_capacity(n_layers);

        for i in 0..n_layers {
            let seed = 42u64.wrapping_add(i as u64 * 1000003);
            layers.push(DenseLayer::new(layer_sizes[i], layer_sizes[i + 1], seed));
            if i < n_layers - 1 {
                activations.push(Activation::ReLU);
            } else {
                activations.push(Activation::Softmax);
            }
        }

        Ok(PureRustNN {
            layers,
            activations,
        })
    }

    /// Forward pass through the entire network.
    ///
    /// Returns the final output and all intermediate activations (needed for backprop).
    /// `intermediates[i]` is the output of layer i BEFORE activation.
    /// `activated[i]` is the output AFTER activation.
    fn forward_with_intermediates(
        &self,
        input: &[f64],
    ) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut pre_activations: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len());
        let mut post_activations: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len());
        let mut current = input.to_vec();

        for (i, layer) in self.layers.iter().enumerate() {
            let pre = layer.forward(&current);
            let post = apply_activation(&pre, self.activations[i]);
            pre_activations.push(pre);
            current = post.clone();
            post_activations.push(post);
        }

        (current, pre_activations, post_activations)
    }

    /// Simple forward pass (inference only).
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            let pre = layer.forward(&current);
            current = apply_activation(&pre, self.activations[i]);
        }
        current
    }

    /// Train the network using mini-batch SGD with analytical backpropagation.
    ///
    /// `inputs`: batch of input vectors (each of length input_dim)
    /// `targets`: batch of target vectors (each of length output_dim)
    /// `learning_rate`: SGD step size
    /// `epochs`: number of full passes over the data
    ///
    /// Uses MSE loss with softmax output. The softmax-MSE gradient combination
    /// is computed analytically for each sample, then averaged over the batch.
    fn train(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        learning_rate: f64,
        epochs: usize,
    ) -> Result<()> {
        if inputs.is_empty() || targets.is_empty() {
            return Err(TransformError::InvalidInput(
                "Training data cannot be empty".to_string(),
            ));
        }
        if inputs.len() != targets.len() {
            return Err(TransformError::InvalidInput(
                "Number of inputs must match number of targets".to_string(),
            ));
        }

        let batch_size = inputs.len() as f64;
        let n_layers = self.layers.len();

        for _epoch in 0..epochs {
            // Allocate gradient accumulators
            let mut all_grad_weights: Vec<Vec<f64>> = self
                .layers
                .iter()
                .map(|l| vec![0.0; l.weights.len()])
                .collect();
            let mut all_grad_biases: Vec<Vec<f64>> = self
                .layers
                .iter()
                .map(|l| vec![0.0; l.biases.len()])
                .collect();

            // Accumulate gradients over the batch
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let (_output, pre_acts, post_acts) = self.forward_with_intermediates(input);

                // Compute d_loss/d_output for MSE loss: 2 * (output - target) / output_dim
                // Combined with softmax derivative using the Jacobian
                let output = &post_acts[n_layers - 1];
                let output_dim = output.len();

                // MSE gradient w.r.t. softmax output: d_mse/d_softmax = 2*(softmax - target)/n
                let d_mse_d_softmax: Vec<f64> = output
                    .iter()
                    .zip(target.iter())
                    .map(|(&o, &t)| 2.0 * (o - t) / output_dim as f64)
                    .collect();

                // Softmax Jacobian: d_softmax_i/d_z_j = softmax_i * (delta_ij - softmax_j)
                // So d_loss/d_z = sum_i (d_mse/d_softmax_i * d_softmax_i/d_z_j)
                //               = sum_i (d_mse_i * softmax_i * (delta_ij - softmax_j))
                //               = softmax_j * (d_mse_j - sum_i(d_mse_i * softmax_i))
                let dot_product: f64 = d_mse_d_softmax
                    .iter()
                    .zip(output.iter())
                    .map(|(&dm, &s)| dm * s)
                    .sum();
                let mut d_loss_d_z: Vec<f64> = Vec::with_capacity(output_dim);
                for j in 0..output_dim {
                    d_loss_d_z.push(output[j] * (d_mse_d_softmax[j] - dot_product));
                }

                // Backpropagate through layers in reverse
                let mut d_current = d_loss_d_z;
                for layer_idx in (0..n_layers).rev() {
                    let layer_input = if layer_idx == 0 {
                        input.as_slice()
                    } else {
                        post_acts[layer_idx - 1].as_slice()
                    };

                    // For hidden layers with ReLU, apply ReLU derivative before the linear backward
                    if layer_idx < n_layers - 1 {
                        // ReLU derivative: 1 if pre_activation > 0, else 0
                        let pre = &pre_acts[layer_idx];
                        for (d, &p) in d_current.iter_mut().zip(pre.iter()) {
                            if p <= 0.0 {
                                *d = 0.0;
                            }
                        }
                    }

                    d_current = self.layers[layer_idx].backward(
                        &d_current,
                        layer_input,
                        &mut all_grad_weights[layer_idx],
                        &mut all_grad_biases[layer_idx],
                    );
                }
            }

            // Average gradients over the batch and apply SGD update
            for layer_idx in 0..n_layers {
                for gw in all_grad_weights[layer_idx].iter_mut() {
                    *gw /= batch_size;
                }
                for gb in all_grad_biases[layer_idx].iter_mut() {
                    *gb /= batch_size;
                }
                self.layers[layer_idx].sgd_update(
                    &all_grad_weights[layer_idx],
                    &all_grad_biases[layer_idx],
                    learning_rate,
                );
            }
        }

        Ok(())
    }
}

/// Apply an activation function element-wise (or across the vector for Softmax).
#[cfg(feature = "auto-feature-engineering")]
fn apply_activation(input: &[f64], activation: Activation) -> Vec<f64> {
    match activation {
        Activation::ReLU => input.iter().map(|&x| x.max(0.0)).collect(),
        Activation::Softmax => {
            // Numerically stable softmax: subtract max before exp
            let max_val = input
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| if b > a { b } else { a });
            let exps: Vec<f64> = input.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f64 = exps.iter().sum();
            if sum < f64::EPSILON {
                // Fallback: uniform distribution if sum is too small
                vec![1.0 / input.len() as f64; input.len()]
            } else {
                exps.iter().map(|&e| e / sum).collect()
            }
        }
    }
}

/// Meta-learning model for transformation selection.
///
/// Uses a pure Rust feed-forward neural network with analytical backpropagation
/// and SGD training for predicting optimal data transformations based on
/// dataset meta-features.
#[cfg(feature = "auto-feature-engineering")]
pub struct MetaLearningModel {
    /// Pure Rust neural network: Linear(10,64)->ReLU->Linear(64,32)->ReLU->
    /// Linear(32,16)->ReLU->Linear(16,10)->Softmax
    network: PureRustNN,
    /// Training data cache for incremental learning
    training_cache: Vec<(DatasetMetaFeatures, Vec<TransformationConfig>)>,
}

#[cfg(feature = "auto-feature-engineering")]
impl MetaLearningModel {
    /// Create a new meta-learning model with He-initialized weights.
    pub fn new() -> Result<Self> {
        let network = PureRustNN::new(&[10, 64, 32, 16, 10])?;
        Ok(MetaLearningModel {
            network,
            training_cache: Vec::new(),
        })
    }

    /// Train the meta-learning model on historical transformation performance data.
    ///
    /// Uses mini-batch SGD with analytical backpropagation through the network.
    /// The network learns to map dataset meta-features (10 dimensions) to
    /// transformation performance scores (10 transformation types).
    pub fn train(
        &mut self,
        training_data: Vec<(DatasetMetaFeatures, Vec<TransformationConfig>)>,
    ) -> Result<()> {
        self.training_cache.extend(training_data.clone());

        let (inputs, targets) = self.prepare_training_data(&training_data)?;

        // Train with SGD: learning rate 0.01, 100 epochs
        self.network.train(&inputs, &targets, 0.01, 100)?;

        Ok(())
    }

    /// Predict optimal transformations for a given dataset based on its meta-features.
    pub fn predict_transformations(
        &self,
        meta_features: &DatasetMetaFeatures,
    ) -> Result<Vec<TransformationConfig>> {
        let input = self.meta_features_to_vec(meta_features)?;
        let scores = self.network.forward(&input);
        self.scores_to_transformations(&scores)
    }

    /// Prepare training data by extracting feature vectors and target score vectors.
    fn prepare_training_data(
        &self,
        training_data: &[(DatasetMetaFeatures, Vec<TransformationConfig>)],
    ) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        if training_data.is_empty() {
            return Err(TransformError::InvalidInput(
                "Training data cannot be empty".to_string(),
            ));
        }

        let mut inputs = Vec::with_capacity(training_data.len());
        let mut targets = Vec::with_capacity(training_data.len());

        for (meta_features, transformations) in training_data {
            let features = vec![
                (meta_features.n_samples as f64).ln().max(0.0),
                (meta_features.n_features as f64).ln().max(0.0),
                meta_features.sparsity.clamp(0.0, 1.0),
                meta_features.mean_correlation.clamp(-1.0, 1.0),
                meta_features.std_correlation.max(0.0),
                meta_features.mean_skewness.clamp(-10.0, 10.0),
                meta_features.mean_kurtosis.clamp(-10.0, 10.0),
                meta_features.missing_ratio.clamp(0.0, 1.0),
                meta_features.variance_ratio.max(0.0),
                meta_features.outlier_ratio.clamp(0.0, 1.0),
            ];
            if features.iter().any(|&f| !f.is_finite()) {
                return Err(TransformError::ComputationError(
                    "Non-finite values detected in meta-features".to_string(),
                ));
            }
            inputs.push(features);

            let mut scores = vec![0.0f64; 10];
            for config in transformations {
                let idx = self.transformation_type_to_index(&config.transformation_type);
                let performance = config.expected_performance.clamp(0.0, 1.0);
                scores[idx] = scores[idx].max(performance);
            }
            targets.push(scores);
        }

        Ok((inputs, targets))
    }

    /// Convert DatasetMetaFeatures to a normalized feature vector.
    fn meta_features_to_vec(&self, meta_features: &DatasetMetaFeatures) -> Result<Vec<f64>> {
        let features = vec![
            (meta_features.n_samples as f64).ln().max(0.0),
            (meta_features.n_features as f64).ln().max(0.0),
            meta_features.sparsity.clamp(0.0, 1.0),
            meta_features.mean_correlation.clamp(-1.0, 1.0),
            meta_features.std_correlation.max(0.0),
            meta_features.mean_skewness.clamp(-10.0, 10.0),
            meta_features.mean_kurtosis.clamp(-10.0, 10.0),
            meta_features.missing_ratio.clamp(0.0, 1.0),
            meta_features.variance_ratio.max(0.0),
            meta_features.outlier_ratio.clamp(0.0, 1.0),
        ];
        if features.iter().any(|&f| !f.is_finite()) {
            return Err(TransformError::ComputationError(
                "Non-finite values detected in meta-features".to_string(),
            ));
        }
        Ok(features)
    }

    /// Convert network output scores to transformation configurations.
    ///
    /// Selects transformations whose predicted scores exceed a dynamic threshold
    /// based on the maximum and mean scores. Falls back to top-3 if none exceed
    /// the threshold.
    fn scores_to_transformations(&self, scores: &[f64]) -> Result<Vec<TransformationConfig>> {
        if scores.len() != 10 {
            return Err(TransformError::ComputationError(format!(
                "Expected 10 prediction scores, got {}",
                scores.len()
            )));
        }

        let mut transformations = Vec::new();
        let max_score = scores.iter().fold(0.0f64, |a, &b| a.max(b));
        let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
        let threshold = (max_score * 0.7 + mean_score * 0.3).max(0.3);

        for (i, &score) in scores.iter().enumerate() {
            if score > threshold && score.is_finite() {
                let transformation_type = self.index_to_transformation_type(i);
                let config = TransformationConfig {
                    transformation_type: transformation_type.clone(),
                    parameters: self.get_default_parameters_for_type(&transformation_type),
                    expected_performance: score.clamp(0.0, 1.0),
                };
                transformations.push(config);
            }
        }

        // Fallback: if no transformation exceeds threshold, pick top 3
        if transformations.is_empty() {
            let mut score_indices: Vec<(usize, f64)> = scores
                .iter()
                .enumerate()
                .filter(|(_, &score)| score.is_finite())
                .map(|(i, &score)| (i, score))
                .collect();
            score_indices
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (i, score) in score_indices.into_iter().take(3) {
                let transformation_type = self.index_to_transformation_type(i);
                let config = TransformationConfig {
                    transformation_type: transformation_type.clone(),
                    parameters: self.get_default_parameters_for_type(&transformation_type),
                    expected_performance: score.clamp(0.0, 1.0),
                };
                transformations.push(config);
            }
        }

        transformations.sort_by(|a, b| {
            b.expected_performance
                .partial_cmp(&a.expected_performance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(transformations)
    }

    fn transformation_type_to_index(&self, t_type: &TransformationType) -> usize {
        match t_type {
            TransformationType::StandardScaler => 0,
            TransformationType::MinMaxScaler => 1,
            TransformationType::RobustScaler => 2,
            TransformationType::PowerTransformer => 3,
            TransformationType::PolynomialFeatures => 4,
            TransformationType::PCA => 5,
            TransformationType::VarianceThreshold => 6,
            TransformationType::QuantileTransformer => 7,
            TransformationType::BinaryEncoder => 8,
            TransformationType::TargetEncoder => 9,
        }
    }

    fn index_to_transformation_type(&self, index: usize) -> TransformationType {
        match index {
            0 => TransformationType::StandardScaler,
            1 => TransformationType::MinMaxScaler,
            2 => TransformationType::RobustScaler,
            3 => TransformationType::PowerTransformer,
            4 => TransformationType::PolynomialFeatures,
            5 => TransformationType::PCA,
            6 => TransformationType::VarianceThreshold,
            7 => TransformationType::QuantileTransformer,
            8 => TransformationType::BinaryEncoder,
            _ => TransformationType::StandardScaler,
        }
    }

    fn get_default_parameters_for_type(&self, t_type: &TransformationType) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        match t_type {
            TransformationType::PCA => {
                params.insert("n_components".to_string(), 0.95);
            }
            TransformationType::PolynomialFeatures => {
                params.insert("degree".to_string(), 2.0);
                params.insert("include_bias".to_string(), 0.0);
            }
            TransformationType::VarianceThreshold => {
                params.insert("threshold".to_string(), 0.01);
            }
            _ => {}
        }
        params
    }
}

/// Reinforcement learning agent for transformation selection.
///
/// Uses pure Rust neural networks for Q-value estimation with experience replay.
/// The agent learns to select optimal transformations through trial and error.
#[cfg(feature = "auto-feature-engineering")]
pub struct RLAgent {
    /// Q-network for value estimation (pure Rust NN)
    q_network: PureRustNN,
    /// Target network for stable training (pure Rust NN)
    target_network: PureRustNN,
    /// Experience replay buffer
    replay_buffer: VecDeque<Experience>,
    /// Epsilon for exploration
    epsilon: f64,
    /// Learning rate
    learning_rate: f64,
    /// Discount factor
    gamma: f64,
}
/// Meta-features extracted from datasets for transformation selection
#[derive(Debug, Clone)]
pub struct DatasetMetaFeatures {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Sparsity ratio (fraction of zero values)
    pub sparsity: f64,
    /// Mean of feature correlations
    pub mean_correlation: f64,
    /// Standard deviation of feature correlations
    pub std_correlation: f64,
    /// Skewness statistics
    pub mean_skewness: f64,
    /// Kurtosis statistics
    pub mean_kurtosis: f64,
    /// Number of missing values
    pub missing_ratio: f64,
    /// Feature variance statistics
    pub variance_ratio: f64,
    /// Outlier ratio
    pub outlier_ratio: f64,
    /// Whether the dataset has missing values
    pub has_missing: bool,
}
/// Automated feature engineering pipeline
pub struct AutoFeatureEngineer {
    #[cfg(feature = "auto-feature-engineering")]
    meta_model: MetaLearningModel,
    /// Historical transformation performance data
    #[cfg(feature = "auto-feature-engineering")]
    transformation_history: Vec<(DatasetMetaFeatures, Vec<TransformationConfig>, f64)>,
}
impl AutoFeatureEngineer {
    /// Expose pearson_correlation as a public method for external use
    #[allow(dead_code)]
    pub fn pearson_correlation(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64> {
        self.pearson_correlation_internal(x, y)
    }
    /// Create a new automated feature engineer
    pub fn new() -> Result<Self> {
        #[cfg(feature = "auto-feature-engineering")]
        let meta_model = MetaLearningModel::new()?;
        Ok(AutoFeatureEngineer {
            #[cfg(feature = "auto-feature-engineering")]
            meta_model,
            #[cfg(feature = "auto-feature-engineering")]
            transformation_history: Vec::new(),
        })
    }
    /// Extract meta-features from a dataset
    pub fn extract_meta_features(&self, x: &ArrayView2<f64>) -> Result<DatasetMetaFeatures> {
        check_not_empty(x, "x")?;
        for &val in x.iter() {
            if !val.is_finite() {
                return Err(crate::error::TransformError::DataValidationError(
                    "Data contains non-finite values".to_string(),
                ));
            }
        }
        let (n_samples, n_features) = x.dim();
        if n_samples < 2 || n_features < 1 {
            return Err(TransformError::InvalidInput(
                "Dataset must have at least 2 samples and 1 feature".to_string(),
            ));
        }
        let zeros = x.iter().filter(|&&val| val == 0.0).count();
        let sparsity = zeros as f64 / (n_samples * n_features) as f64;
        let correlations = self.compute_feature_correlations(x)?;
        let mean_correlation = correlations.mean().unwrap_or(0.0);
        let std_correlation = 0.0;
        let (mean_skewness, mean_kurtosis) = self.compute_distribution_stats(x)?;
        let missing_count = x.iter().filter(|val| val.is_nan()).count();
        let missing_ratio = missing_count as f64 / (n_samples * n_features) as f64;
        let has_missing = missing_count > 0;
        let variances: Array1<f64> = x.var_axis(scirs2_core::ndarray::Axis(0), 0.0);
        let finite_variances: Vec<f64> = variances
            .iter()
            .filter(|&&v| v.is_finite() && v >= 0.0)
            .copied()
            .collect();
        let variance_ratio = if finite_variances.is_empty() {
            0.0
        } else {
            let mean_var = finite_variances.iter().sum::<f64>() / finite_variances.len() as f64;
            if mean_var < f64::EPSILON {
                0.0
            } else {
                let var_of_vars = finite_variances
                    .iter()
                    .map(|&v| (v - mean_var).powi(2))
                    .sum::<f64>()
                    / finite_variances.len() as f64;
                (var_of_vars.sqrt() / mean_var).min(100.0)
            }
        };
        let outlier_ratio = self.compute_outlier_ratio(x)?;
        Ok(DatasetMetaFeatures {
            n_samples,
            n_features,
            sparsity,
            mean_correlation,
            std_correlation,
            mean_skewness,
            mean_kurtosis,
            missing_ratio,
            variance_ratio,
            outlier_ratio,
            has_missing,
        })
    }
    /// Recommend optimal transformations for a dataset
    #[cfg(feature = "auto-feature-engineering")]
    pub fn recommend_transformations(
        &self,
        x: &ArrayView2<f64>,
    ) -> Result<Vec<TransformationConfig>> {
        let meta_features = self.extract_meta_features(x)?;
        self.meta_model.predict_transformations(&meta_features)
    }
    /// Recommend optimal transformations for a dataset (fallback implementation)
    #[cfg(not(feature = "auto-feature-engineering"))]
    pub fn recommend_transformations(
        &self,
        x: &ArrayView2<f64>,
    ) -> Result<Vec<TransformationConfig>> {
        self.rule_based_recommendations(x)
    }
    /// Rule-based transformation recommendations (fallback)
    fn rule_based_recommendations(&self, x: &ArrayView2<f64>) -> Result<Vec<TransformationConfig>> {
        let meta_features = self.extract_meta_features(x)?;
        let mut recommendations = Vec::new();
        if meta_features.mean_skewness.abs() > 1.0 {
            recommendations.push(TransformationConfig {
                transformation_type: TransformationType::PowerTransformer,
                parameters: HashMap::new(),
                expected_performance: 0.8,
            });
        }
        if meta_features.n_features > 100 {
            let mut params = HashMap::new();
            params.insert("n_components".to_string(), 0.95);
            recommendations.push(TransformationConfig {
                transformation_type: TransformationType::PCA,
                parameters: params,
                expected_performance: 0.75,
            });
        }
        if meta_features.variance_ratio > 1.0 {
            recommendations.push(TransformationConfig {
                transformation_type: TransformationType::StandardScaler,
                parameters: HashMap::new(),
                expected_performance: 0.9,
            });
        }
        if meta_features.outlier_ratio > 0.1 {
            recommendations.push(TransformationConfig {
                transformation_type: TransformationType::RobustScaler,
                parameters: HashMap::new(),
                expected_performance: 0.85,
            });
        }
        recommendations.sort_by(|a, b| {
            b.expected_performance
                .partial_cmp(&a.expected_performance)
                .expect("Operation failed")
        });
        Ok(recommendations)
    }
    /// Train the meta-learning model with new data
    #[cfg(feature = "auto-feature-engineering")]
    pub fn update_model(
        &mut self,
        meta_features: DatasetMetaFeatures,
        transformations: Vec<TransformationConfig>,
        performance: f64,
    ) -> Result<()> {
        self.transformation_history.push((
            meta_features.clone(),
            transformations.clone(),
            performance,
        ));
        if self.transformation_history.len() % 10 == 0 {
            let training_data: Vec<_> = self
                .transformation_history
                .iter()
                .map(|(meta, trans, _perf)| (meta.clone(), trans.clone()))
                .collect();
            self.meta_model.train(training_data)?;
        }
        Ok(())
    }
    fn compute_feature_correlations(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>> {
        let n_features = x.ncols();
        if n_features < 2 {
            return Ok(Array1::zeros(0));
        }
        let mut correlations = Vec::with_capacity((n_features * (n_features - 1)) / 2);
        for i in 0..n_features {
            for j in i + 1..n_features {
                let col_i = x.column(i);
                let col_j = x.column(j);
                let correlation = self.pearson_correlation_internal(&col_i, &col_j)?;
                correlations.push(correlation);
            }
        }
        Ok(Array1::from_vec(correlations))
    }
    fn pearson_correlation_internal(
        &self,
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
    ) -> Result<f64> {
        if x.len() != y.len() {
            return Err(TransformError::InvalidInput(
                "Arrays must have the same length for correlation calculation".to_string(),
            ));
        }
        if x.len() < 2 {
            return Ok(0.0);
        }
        let _n = x.len() as f64;
        let mean_x = x.mean_or(0.0);
        let mean_y = y.mean_or(0.0);
        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();
        let sum_sq_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator < f64::EPSILON {
            Ok(0.0)
        } else {
            let correlation = numerator / denominator;
            Ok(correlation.clamp(-1.0, 1.0))
        }
    }
    fn compute_distribution_stats(&self, x: &ArrayView2<f64>) -> Result<(f64, f64)> {
        let mut skewness_values = Vec::new();
        let mut kurtosis_values = Vec::new();
        for col in x.columns() {
            let finite_values: Vec<f64> = col
                .iter()
                .filter(|&&val| val.is_finite())
                .copied()
                .collect();
            if finite_values.len() < 3 {
                continue;
            }
            let n = finite_values.len() as f64;
            let mean = finite_values.iter().sum::<f64>() / n;
            let variance = finite_values
                .iter()
                .map(|&val| (val - mean).powi(2))
                .sum::<f64>()
                / (n - 1.0);
            let std = variance.sqrt();
            if std > f64::EPSILON * 1000.0 {
                let m3: f64 = finite_values
                    .iter()
                    .map(|&val| ((val - mean) / std).powi(3))
                    .sum::<f64>()
                    / n;
                let skew = if n > 2.0 {
                    m3 * (n * (n - 1.0)).sqrt() / (n - 2.0)
                } else {
                    m3
                };
                let m4: f64 = finite_values
                    .iter()
                    .map(|&val| ((val - mean) / std).powi(4))
                    .sum::<f64>()
                    / n;
                let kurt = if n > 3.0 {
                    let numerator = (n - 1.0) * ((n + 1.0) * m4 - 3.0 * (n - 1.0));
                    let denominator = (n - 2.0) * (n - 3.0);
                    numerator / denominator
                } else {
                    m4 - 3.0
                };
                skewness_values.push(skew.clamp(-20.0, 20.0));
                kurtosis_values.push(kurt.clamp(-20.0, 20.0));
            }
        }
        let mean_skewness = if skewness_values.is_empty() {
            0.0
        } else {
            skewness_values.iter().sum::<f64>() / skewness_values.len() as f64
        };
        let mean_kurtosis = if kurtosis_values.is_empty() {
            0.0
        } else {
            kurtosis_values.iter().sum::<f64>() / kurtosis_values.len() as f64
        };
        Ok((mean_skewness, mean_kurtosis))
    }
    fn compute_outlier_ratio(&self, x: &ArrayView2<f64>) -> Result<f64> {
        let mut total_outliers = 0;
        let mut total_values = 0;
        for col in x.columns() {
            let mut sorted_col: Vec<f64> = col
                .iter()
                .filter(|&&val| val.is_finite())
                .copied()
                .collect();
            if sorted_col.is_empty() {
                continue;
            }
            sorted_col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted_col.len();
            if n < 4 {
                continue;
            }
            let q1_idx = (n as f64 * 0.25) as usize;
            let q3_idx = (n as f64 * 0.75) as usize;
            let q1 = sorted_col[q1_idx.min(n - 1)];
            let q3 = sorted_col[q3_idx.min(n - 1)];
            let iqr = q3 - q1;
            if iqr < f64::EPSILON {
                continue;
            }
            let lower_bound = q1 - 1.5 * iqr;
            let upper_bound = q3 + 1.5 * iqr;
            let outliers = col
                .iter()
                .filter(|&&val| val.is_finite() && (val < lower_bound || val > upper_bound))
                .count();
            total_outliers += outliers;
            total_values += col.len();
        }
        if total_values == 0 {
            Ok(0.0)
        } else {
            Ok(total_outliers as f64 / total_values as f64)
        }
    }
}
