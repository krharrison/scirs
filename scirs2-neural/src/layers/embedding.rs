//! Embedding layer implementations
//!
//! This module provides implementations of various embedding layers
//! such as word embeddings, positional embeddings, and patch embeddings for vision.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

/// Configuration for the Embedding layer
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Number of embeddings in the embedding table
    pub num_embeddings: usize,
    /// Dimension of each embedding vector
    pub embedding_dim: usize,
    /// Optional padding index that will have its embedding vector filled with zeros
    pub padding_idx: Option<usize>,
    /// Maximum norm for embedding vectors
    pub max_norm: Option<f64>,
    /// Type of norm to use with max_norm
    pub norm_type: f64,
    /// Whether to scale gradients by the inverse of frequency of the indices
    pub scale_grad_by_freq: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            num_embeddings: 1,
            embedding_dim: 1,
            padding_idx: None,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
        }
    }
}

/// Embedding layer that stores embeddings for discrete inputs
///
/// This layer is often used to store word embeddings and retrieve them using indices.
/// The input to the module is a list of indices, and the output is the corresponding
/// embedding vectors.
///
/// # Examples
///
/// ```rust
/// use scirs2_neural::layers::{Embedding, EmbeddingConfig, Layer};
/// use scirs2_core::ndarray::Array2;
///
/// let config = EmbeddingConfig {
///     num_embeddings: 100,
///     embedding_dim: 64,
///     ..Default::default()
/// };
/// let embedding = Embedding::<f64>::new(config).expect("Operation failed");
///
/// // Input: indices as floats (will be converted to usize)
/// let indices = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("Operation failed");
/// let output = embedding.forward(&indices.into_dyn()).expect("Operation failed");
/// assert_eq!(output.shape(), &[2, 3, 64]);
/// ```
#[derive(Debug, Clone)]
pub struct Embedding<F: Float + Debug + Send + Sync + NumAssign> {
    /// Configuration for the embedding layer
    config: EmbeddingConfig,
    /// Weight matrix containing the embeddings (num_embeddings x embedding_dim)
    weights: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of the weight matrix
    weight_grad: Arc<RwLock<Array<F, IxDyn>>>,
    /// Frequency counter for indices (for scale_grad_by_freq)
    freq_counter: Option<Vec<usize>>,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + NumAssign> Embedding<F> {
    /// Create a new Embedding layer with the given configuration
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        if config.num_embeddings == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "num_embeddings must be greater than 0".to_string(),
            ));
        }

        if config.embedding_dim == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "embedding_dim must be greater than 0".to_string(),
            ));
        }

        // Validate padding_idx
        if let Some(idx) = config.padding_idx {
            if idx >= config.num_embeddings {
                return Err(NeuralError::InvalidArchitecture(format!(
                    "padding_idx ({}) must be less than num_embeddings ({})",
                    idx, config.num_embeddings
                )));
            }
        }

        // Initialize weights with uniform distribution scaled to [-1, 1]
        let weights_shape = IxDyn(&[config.num_embeddings, config.embedding_dim]);
        let mut weights = Array::from_shape_fn(weights_shape.clone(), |idx| {
            // Simple deterministic initialization
            let (i, j) = (idx[0], idx[1]);
            let value = ((i * 31 + j * 17) % 1000) as f64 / 1000.0 - 0.5;
            F::from(value).expect("Failed to convert to float")
        });

        // Set padding_idx embeddings to zero if specified
        if let Some(idx) = config.padding_idx {
            for j in 0..config.embedding_dim {
                weights[[idx, j]] = F::zero();
            }
        }

        // Initialize gradients with zeros
        let weight_grad = Array::zeros(weights_shape);

        // Initialize frequency counter if needed
        let freq_counter = if config.scale_grad_by_freq {
            Some(vec![0; config.num_embeddings])
        } else {
            None
        };

        Ok(Self {
            config,
            weights: Arc::new(RwLock::new(weights)),
            weight_grad: Arc::new(RwLock::new(weight_grad)),
            freq_counter,
            input_cache: Arc::new(RwLock::new(None)),
            _phantom: PhantomData,
        })
    }

    /// Create an Embedding layer from pretrained embeddings
    pub fn from_pretrained(
        embeddings: Array<F, IxDyn>,
        padding_idx: Option<usize>,
        max_norm: Option<f64>,
        norm_type: f64,
        scale_grad_by_freq: bool,
    ) -> Result<Self> {
        if embeddings.ndim() != 2 {
            return Err(NeuralError::InvalidArchitecture(
                "Embeddings parameter is expected to be 2-dimensional".to_string(),
            ));
        }

        let shape = embeddings.shape();
        let num_embeddings = shape[0];
        let embedding_dim = shape[1];

        if let Some(idx) = padding_idx {
            if idx >= num_embeddings {
                return Err(NeuralError::InvalidArchitecture(format!(
                    "padding_idx ({}) must be less than num_embeddings ({})",
                    idx, num_embeddings
                )));
            }
        }

        let config = EmbeddingConfig {
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
        };

        let weight_grad = Array::zeros(IxDyn(&[num_embeddings, embedding_dim]));
        let freq_counter = if scale_grad_by_freq {
            Some(vec![0; num_embeddings])
        } else {
            None
        };

        Ok(Self {
            config,
            weights: Arc::new(RwLock::new(embeddings)),
            weight_grad: Arc::new(RwLock::new(weight_grad)),
            freq_counter,
            input_cache: Arc::new(RwLock::new(None)),
            _phantom: PhantomData,
        })
    }

    /// Get the number of embeddings
    pub fn num_embeddings(&self) -> usize {
        self.config.num_embeddings
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Apply max_norm to the embeddings if specified
    fn apply_max_norm(&self) -> Result<()> {
        if let Some(max_norm) = self.config.max_norm {
            let norm_type = self.config.norm_type;
            let p = F::from(norm_type).ok_or_else(|| {
                NeuralError::InvalidArchitecture(format!("Invalid norm_type: {}", norm_type))
            })?;
            let max_norm_f = F::from(max_norm).ok_or_else(|| {
                NeuralError::InvalidArchitecture(format!("Invalid max_norm: {}", max_norm))
            })?;

            let mut weights = self.weights.write().map_err(|_| {
                NeuralError::InferenceError("Failed to acquire write lock on weights".to_string())
            })?;

            // Calculate norms for each embedding vector
            for i in 0..self.config.num_embeddings {
                let mut norm = F::zero();

                // Calculate p-norm
                for j in 0..self.config.embedding_dim {
                    let val = weights[[i, j]];
                    if p == F::from(2.0).expect("Failed to convert constant to float") {
                        norm += val * val;
                    } else {
                        norm += val.abs().powf(p);
                    }
                }

                if p == F::from(2.0).expect("Failed to convert constant to float") {
                    norm = norm.sqrt();
                } else {
                    norm = norm.powf(F::one() / p);
                }

                // Apply max_norm if needed
                if norm > max_norm_f {
                    let scale = max_norm_f / norm;
                    for j in 0..self.config.embedding_dim {
                        weights[[i, j]] *= scale;
                    }
                }
            }
        }
        Ok(())
    }

    /// Internal forward pass implementation
    fn forward_impl(&self, indices: &[usize]) -> Result<Array<F, IxDyn>> {
        // Validate indices
        for &idx in indices.iter() {
            if idx >= self.config.num_embeddings {
                return Err(NeuralError::InferenceError(format!(
                    "Index {} out of bounds for embedding with {} entries",
                    idx, self.config.num_embeddings
                )));
            }
        }

        // Apply max_norm if specified
        self.apply_max_norm()?;

        let weights = self.weights.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on weights".to_string())
        })?;

        // Create output array
        let output_shape = IxDyn(&[indices.len(), self.config.embedding_dim]);
        let mut output = Array::zeros(output_shape);

        // Lookup embeddings
        for (i, &idx) in indices.iter().enumerate() {
            for j in 0..self.config.embedding_dim {
                output[[i, j]] = weights[[idx, j]];
            }
        }

        Ok(output)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + NumAssign> Layer<F>
    for Embedding<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(input.clone());
        }

        // Convert input to indices
        let input_shape = input.shape().to_vec();
        let total_elements: usize = input_shape.iter().product();

        let indices: Vec<usize> = input.iter().map(|&x| x.to_usize().unwrap_or(0)).collect();

        // Get embeddings
        let flat_output = self.forward_impl(&indices)?;

        // Reshape output to match input shape + embedding_dim
        let mut output_shape = input_shape;
        output_shape.push(self.config.embedding_dim);

        let output = flat_output
            .into_shape_with_order(IxDyn(&output_shape))
            .map_err(|e| NeuralError::InferenceError(format!("Shape mismatch: {}", e)))?;

        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Get cached input
        let input_guard = self.input_cache.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on input cache".to_string())
        })?;

        let input = input_guard.as_ref().ok_or_else(|| {
            NeuralError::InferenceError(
                "No cached input for backward pass. Call forward() first.".to_string(),
            )
        })?;

        // Convert input to indices
        let indices: Vec<usize> = input.iter().map(|&x| x.to_usize().unwrap_or(0)).collect();

        // Accumulate gradients
        let mut weight_grad = self.weight_grad.write().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire write lock on weight_grad".to_string())
        })?;

        // Flatten grad_output for processing
        let grad_flat = grad_output.view();
        let grad_shape = grad_flat.shape();
        let embedding_dim = grad_shape[grad_shape.len() - 1];

        // Accumulate gradients for each embedding
        for (i, &idx) in indices.iter().enumerate() {
            // Skip padding indices
            if let Some(padding_idx) = self.config.padding_idx {
                if idx == padding_idx {
                    continue;
                }
            }

            for j in 0..embedding_dim {
                // Handle multi-dimensional input
                if grad_shape.len() > 2 {
                    // For now, use flat index
                    let flat_grad_idx = i * embedding_dim + j;
                    if flat_grad_idx < grad_output.len() {
                        weight_grad[[idx, j]] +=
                            grad_output.as_slice().expect("Operation failed")[flat_grad_idx];
                    }
                } else {
                    weight_grad[[idx, j]] += grad_output[[i, j]];
                }
            }
        }

        // Return zeros for input gradient (indices don't have meaningful gradients)
        Ok(Array::zeros(input.raw_dim()))
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        let lr = learning_rate;

        let weight_grad = self.weight_grad.read().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire read lock on weight_grad".to_string())
        })?;

        let mut weights = self.weights.write().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire write lock on weights".to_string())
        })?;

        // Handle frequency-based scaling
        if let Some(ref counter) = self.freq_counter {
            for (i, &count) in counter.iter().enumerate().take(self.config.num_embeddings) {
                // Skip padding indices
                if let Some(padding_idx) = self.config.padding_idx {
                    if i == padding_idx {
                        continue;
                    }
                }

                let scale = if count > 0 {
                    F::from(1.0 / count as f64).expect("Failed to convert to float")
                } else {
                    F::one()
                };

                for j in 0..self.config.embedding_dim {
                    weights[[i, j]] -= lr * scale * weight_grad[[i, j]];
                }
            }
        } else {
            // Standard gradient update
            for i in 0..self.config.num_embeddings {
                // Skip padding indices
                if let Some(padding_idx) = self.config.padding_idx {
                    if i == padding_idx {
                        continue;
                    }
                }

                for j in 0..self.config.embedding_dim {
                    weights[[i, j]] -= lr * weight_grad[[i, j]];
                }
            }
        }

        // Reset gradients
        drop(weights);
        drop(weight_grad);

        let mut weight_grad = self.weight_grad.write().map_err(|_| {
            NeuralError::InferenceError("Failed to acquire write lock on weight_grad".to_string())
        })?;
        weight_grad.fill(F::zero());

        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "Embedding"
    }

    fn parameter_count(&self) -> usize {
        self.config.num_embeddings * self.config.embedding_dim
    }

    fn layer_description(&self) -> String {
        format!(
            "type:Embedding, num_embeddings:{}, embedding_dim:{}, params:{}",
            self.config.num_embeddings,
            self.config.embedding_dim,
            self.parameter_count()
        )
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        if let Ok(w) = self.weights.read() {
            vec![w.clone()]
        } else {
            vec![]
        }
    }

    fn set_params(&mut self, params: &[Array<F, IxDyn>]) -> Result<()> {
        if let Some(new_weights) = params.first() {
            if let Ok(mut w) = self.weights.write() {
                *w = new_weights.clone();
            }
        }
        Ok(())
    }
}

/// Positional Embedding layer for transformers and sequence models
///
/// This layer adds positional information to embeddings to help models
/// understand the position of elements in a sequence.
#[derive(Debug)]
pub struct PositionalEmbedding<F: Float + Debug + Send + Sync + NumAssign> {
    /// Maximum sequence length supported
    max_seq_length: usize,
    /// Embedding dimension
    embedding_dim: usize,
    /// Whether to use learned positional embeddings (true) or fixed sinusoidal (false)
    learned: bool,
    /// Weight matrix for learned positional embeddings
    weights: Option<Arc<RwLock<Array<F, IxDyn>>>>,
    /// Weight gradients
    weight_grad: Option<Arc<RwLock<Array<F, IxDyn>>>>,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + NumAssign> PositionalEmbedding<F> {
    /// Create a new PositionalEmbedding layer
    pub fn new(max_seq_length: usize, embedding_dim: usize, learned: bool) -> Result<Self> {
        if max_seq_length == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "max_seq_length must be greater than 0".to_string(),
            ));
        }

        if embedding_dim == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "embedding_dim must be greater than 0".to_string(),
            ));
        }

        let (weights, weight_grad) = if learned {
            // Initialize learned positional embeddings
            let weights_shape = IxDyn(&[max_seq_length, embedding_dim]);

            // Xavier initialization
            let scale = F::from((2.0 / (max_seq_length + embedding_dim) as f64).sqrt())
                .expect("Operation failed");
            let weights = Array::from_shape_fn(weights_shape.clone(), |_| {
                scale
                    * (F::from(0.5).expect("Failed to convert constant to float")
                        - F::from(0.25).expect("Failed to convert constant to float"))
            });

            let weight_grad = Array::zeros(weights_shape);

            (
                Some(Arc::new(RwLock::new(weights))),
                Some(Arc::new(RwLock::new(weight_grad))),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            max_seq_length,
            embedding_dim,
            learned,
            weights,
            weight_grad,
            _phantom: PhantomData,
        })
    }

    /// Generate sinusoidal positional embeddings
    fn generate_sinusoidal_embeddings(&self, seq_length: usize) -> Result<Array<F, IxDyn>> {
        if seq_length > self.max_seq_length {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Sequence length {} exceeds maximum supported length {}",
                seq_length, self.max_seq_length
            )));
        }

        let mut pos_embeddings = Array::zeros(IxDyn(&[seq_length, self.embedding_dim]));

        for pos in 0..seq_length {
            for i in 0..self.embedding_dim {
                let div_term = (10000.0_f64).powf(2.0 * (i / 2) as f64 / self.embedding_dim as f64);
                let angle = pos as f64 / div_term;

                if i % 2 == 0 {
                    pos_embeddings[[pos, i]] = F::from(angle.sin()).expect("Operation failed");
                } else {
                    pos_embeddings[[pos, i]] = F::from(angle.cos()).expect("Operation failed");
                }
            }
        }

        Ok(pos_embeddings)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + NumAssign> Layer<F>
    for PositionalEmbedding<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Validate input shape - at least 2D with last dimension being embedding_dim
        if input.ndim() < 2 {
            return Err(NeuralError::InferenceError(
                "Input to PositionalEmbedding must be at least 2D".to_string(),
            ));
        }

        let last_dim = *input.shape().last().expect("Operation failed");
        if last_dim != self.embedding_dim {
            return Err(NeuralError::InferenceError(format!(
                "Input embedding dimension {} doesn't match layer embedding dimension {}",
                last_dim, self.embedding_dim
            )));
        }

        // Get sequence length from the input shape (second to last dimension)
        let seq_dim = input.ndim() - 2;
        let seq_length = input.shape()[seq_dim];

        if seq_length > self.max_seq_length {
            return Err(NeuralError::InferenceError(format!(
                "Input sequence length {} exceeds maximum supported length {}",
                seq_length, self.max_seq_length
            )));
        }

        // Get positional embeddings
        let pos_embeddings = if self.learned {
            let weights = self
                .weights
                .as_ref()
                .expect("Operation failed")
                .read()
                .map_err(|_| {
                    NeuralError::InferenceError(
                        "Failed to acquire read lock on weights".to_string(),
                    )
                })?;

            // Extract the first seq_length positions
            let mut pos_emb = Array::zeros(IxDyn(&[seq_length, self.embedding_dim]));
            for i in 0..seq_length {
                for j in 0..self.embedding_dim {
                    pos_emb[[i, j]] = weights[[i, j]];
                }
            }
            pos_emb
        } else {
            self.generate_sinusoidal_embeddings(seq_length)?
        };

        // Add positional embeddings to input
        let mut output = input.clone();
        let input_shape = input.shape();

        // Simple broadcast: iterate over all positions and embedding dims
        // Handle batch dimensions by iterating over the full array
        let batch_dims: usize = input_shape[..seq_dim].iter().product();

        for batch_idx in 0..batch_dims {
            for pos in 0..seq_length {
                for emb_idx in 0..self.embedding_dim {
                    // Calculate flat index for this element
                    // For a simple 3D case (batch, seq, emb), this is straightforward
                    if input_shape.len() == 3 {
                        let b = batch_idx;
                        output[[b, pos, emb_idx]] += pos_embeddings[[pos, emb_idx]];
                    } else if input_shape.len() == 2 {
                        // 2D case (seq, emb)
                        output[[pos, emb_idx]] += pos_embeddings[[pos, emb_idx]];
                    }
                }
            }
        }

        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // For PositionalEmbedding, gradients flow through directly
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Only update weights if using learned positional embeddings
        if self.learned {
            if let (Some(ref weights_lock), Some(ref weight_grad_lock)) =
                (&self.weights, &self.weight_grad)
            {
                let lr = learning_rate;

                let weight_grad = weight_grad_lock.read().map_err(|_| {
                    NeuralError::InferenceError(
                        "Failed to acquire read lock on weight_grad".to_string(),
                    )
                })?;

                let mut weights = weights_lock.write().map_err(|_| {
                    NeuralError::InferenceError(
                        "Failed to acquire write lock on weights".to_string(),
                    )
                })?;

                for i in 0..self.max_seq_length {
                    for j in 0..self.embedding_dim {
                        weights[[i, j]] -= lr * weight_grad[[i, j]];
                    }
                }

                // Reset gradients
                drop(weights);
                drop(weight_grad);

                let mut weight_grad = weight_grad_lock.write().map_err(|_| {
                    NeuralError::InferenceError(
                        "Failed to acquire write lock on weight_grad".to_string(),
                    )
                })?;
                weight_grad.fill(F::zero());
            }
        }

        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        if self.learned {
            "LearnedPositionalEmbedding"
        } else {
            "SinusoidalPositionalEmbedding"
        }
    }

    fn parameter_count(&self) -> usize {
        if self.learned {
            self.max_seq_length * self.embedding_dim
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.num_embeddings, 1);
        assert_eq!(config.embedding_dim, 1);
        assert!(config.padding_idx.is_none());
    }

    #[test]
    fn test_embedding_creation() {
        let config = EmbeddingConfig {
            num_embeddings: 100,
            embedding_dim: 64,
            padding_idx: Some(0),
            ..Default::default()
        };

        let embedding = Embedding::<f64>::new(config).expect("Operation failed");
        assert_eq!(embedding.num_embeddings(), 100);
        assert_eq!(embedding.embedding_dim(), 64);
    }

    #[test]
    fn test_embedding_creation_invalid() {
        let config = EmbeddingConfig {
            num_embeddings: 0,
            embedding_dim: 64,
            ..Default::default()
        };
        assert!(Embedding::<f64>::new(config).is_err());
    }

    #[test]
    fn test_embedding_forward() {
        let config = EmbeddingConfig {
            num_embeddings: 10,
            embedding_dim: 4,
            ..Default::default()
        };

        let embedding = Embedding::<f64>::new(config).expect("Operation failed");

        // Input: indices as floats
        let indices = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Operation failed");
        let output = embedding
            .forward(&indices.into_dyn())
            .expect("Operation failed");

        // Output shape: (2, 3, 4)
        assert_eq!(output.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_embedding_parameter_count() {
        let config = EmbeddingConfig {
            num_embeddings: 100,
            embedding_dim: 64,
            ..Default::default()
        };

        let embedding = Embedding::<f64>::new(config).expect("Operation failed");
        assert_eq!(embedding.parameter_count(), 100 * 64);
    }

    #[test]
    fn test_positional_embedding_learned() {
        let pos_emb = PositionalEmbedding::<f64>::new(100, 64, true).expect("Operation failed");
        assert!(pos_emb.weights.is_some());
        assert_eq!(pos_emb.parameter_count(), 100 * 64);
    }

    #[test]
    fn test_positional_embedding_sinusoidal() {
        let pos_emb = PositionalEmbedding::<f64>::new(100, 64, false).expect("Operation failed");
        assert!(pos_emb.weights.is_none());
        assert_eq!(pos_emb.parameter_count(), 0);
    }

    #[test]
    fn test_positional_embedding_forward() {
        let pos_emb = PositionalEmbedding::<f64>::new(100, 64, false).expect("Operation failed");

        // Input: (batch=2, seq=10, emb=64)
        let input = Array::from_elem(IxDyn(&[2, 10, 64]), 1.0);
        let output = pos_emb.forward(&input).expect("Operation failed");

        assert_eq!(output.shape(), &[2, 10, 64]);
    }
}
