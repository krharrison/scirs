//! Attention mechanism implementation for neural networks
//!
//! This module provides implementation of various attention mechanisms
//! including dot-product attention, multi-head attention, and self-attention
//! as used in transformer architectures.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

/// Different types of attention masks
#[derive(Debug, Clone)]
pub enum AttentionMask {
    /// Causal mask (upper triangular with -inf) for autoregressive models
    Causal,
    /// Padding mask for variable length sequences
    Padding(Vec<usize>),
    /// Custom boolean mask (true allows attention, false blocks it)
    Custom(Array<bool, IxDyn>),
}

/// Configuration for attention
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension of each attention head
    pub head_dim: usize,
    /// Dropout probability (0.0 means no dropout)
    pub dropout_prob: f64,
    /// Whether to use causal attention
    pub causal: bool,
    /// Custom scaling factor (default is 1/sqrt(head_dim))
    pub scale: Option<f32>,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            dropout_prob: 0.1,
            causal: false,
            scale: None,
        }
    }
}

/// Multi-head attention layer as used in transformer architectures
///
/// This layer performs the attention operation described in "Attention Is All You Need"
/// by Vaswani et al. It projects the queries, keys, and values into multiple heads,
/// computes scaled dot-product attention for each head, concatenates the results,
/// and projects the result back to the original dimension.
///
/// # Input Shape
/// - 3D tensor: (batch_size, seq_len, d_model)
///
/// # Output Shape
/// - 3D tensor: (batch_size, seq_len, d_model)
///
/// # Examples
///
/// ```rust
/// use scirs2_neural::layers::{MultiHeadAttention, Layer, AttentionConfig};
/// use scirs2_core::ndarray::Array3;
/// use scirs2_core::random::rng;
///
/// // Create multi-head attention with 2 heads and 64-dim embeddings
/// let mut rng = rng();
/// let config = AttentionConfig {
///     num_heads: 2,
///     head_dim: 32,
///     dropout_prob: 0.0,
///     causal: false,
///     scale: None,
/// };
/// let mha = MultiHeadAttention::<f64>::new(64, config, &mut rng).expect("Operation failed");
///
/// // Forward pass with a batch of 2 samples, sequence length 3
/// let input = Array3::<f64>::from_elem((2, 3, 64), 0.1).into_dyn();
/// let output = mha.forward(&input).expect("Operation failed");
///
/// // Output shape should match input shape
/// assert_eq!(output.shape(), input.shape());
/// ```
#[derive(Debug, Clone)]
#[allow(clippy::type_complexity)]
pub struct MultiHeadAttention<F: Float + Debug + Send + Sync + NumAssign> {
    /// Embedding dimension
    d_model: usize,
    /// Attention configuration
    config: AttentionConfig,
    /// Weight matrix for query projection [d_model, d_model]
    w_query: Array<F, IxDyn>,
    /// Weight matrix for key projection [d_model, d_model]
    w_key: Array<F, IxDyn>,
    /// Weight matrix for value projection [d_model, d_model]
    w_value: Array<F, IxDyn>,
    /// Weight matrix for output projection [d_model, d_model]
    w_output: Array<F, IxDyn>,
    /// Gradient of the query weights
    dw_query: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of the key weights
    dw_key: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of the value weights
    dw_value: Arc<RwLock<Array<F, IxDyn>>>,
    /// Gradient of the output weights
    dw_output: Arc<RwLock<Array<F, IxDyn>>>,
    /// Scaling factor for attention scores
    scale: F,
    /// Input cache for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Attention weights cache for backward pass
    attention_weights_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Layer name
    name: Option<String>,
    /// Training mode
    training: bool,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> MultiHeadAttention<F> {
    /// Create a new multi-head attention layer
    ///
    /// # Arguments
    /// * `d_model` - Embedding dimension (must be divisible by num_heads)
    /// * `config` - Attention configuration
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
    /// A new multi-head attention layer
    pub fn new<R: Rng>(d_model: usize, config: AttentionConfig, rng: &mut R) -> Result<Self> {
        // Verify configuration
        if !d_model.is_multiple_of(config.num_heads) {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Model dimension ({}) must be divisible by the number of heads ({})",
                d_model, config.num_heads
            )));
        }

        let computed_head_dim = d_model / config.num_heads;
        if config.head_dim != computed_head_dim {
            return Err(NeuralError::InvalidArchitecture(format!(
                "head_dim ({}) must equal d_model / num_heads ({})",
                config.head_dim, computed_head_dim
            )));
        }

        // Xavier/Glorot initialization scale
        let init_scale = (2.0 / (d_model + d_model) as f64).sqrt();

        // Helper to create weight matrix
        let create_weights = |rng: &mut R| -> Result<Array<F, IxDyn>> {
            let mut data = Vec::with_capacity(d_model * d_model);
            for _ in 0..(d_model * d_model) {
                let val: f64 = rng.random_range(-1.0..1.0);
                let scaled = F::from(val * init_scale).ok_or_else(|| {
                    NeuralError::InvalidArchitecture("Failed to convert weight value".to_string())
                })?;
                data.push(scaled);
            }
            Array::from_shape_vec(IxDyn(&[d_model, d_model]), data).map_err(|e| {
                NeuralError::InvalidArchitecture(format!("Failed to create weight matrix: {}", e))
            })
        };

        let w_query = create_weights(rng)?;
        let w_key = create_weights(rng)?;
        let w_value = create_weights(rng)?;
        let w_output = create_weights(rng)?;

        // Initialize gradients to zeros
        let zeros = Array::zeros(IxDyn(&[d_model, d_model]));
        let dw_query = Arc::new(RwLock::new(zeros.clone()));
        let dw_key = Arc::new(RwLock::new(zeros.clone()));
        let dw_value = Arc::new(RwLock::new(zeros.clone()));
        let dw_output = Arc::new(RwLock::new(zeros));

        // Compute scaling factor (1/sqrt(d_k))
        let scale = match config.scale {
            Some(s) => F::from(s).ok_or_else(|| {
                NeuralError::InvalidArchitecture("Failed to convert scale factor".to_string())
            })?,
            None => F::from(1.0 / (config.head_dim as f64).sqrt()).ok_or_else(|| {
                NeuralError::InvalidArchitecture("Failed to compute scale factor".to_string())
            })?,
        };

        Ok(Self {
            d_model,
            config,
            w_query,
            w_key,
            w_value,
            w_output,
            dw_query,
            dw_key,
            dw_value,
            dw_output,
            scale,
            input_cache: Arc::new(RwLock::new(None)),
            attention_weights_cache: Arc::new(RwLock::new(None)),
            name: None,
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Set a name for this layer
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Linear projection: input @ weights
    fn linear_projection(
        &self,
        input: &Array<F, IxDyn>,
        weights: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 3D input, got {}D",
                shape.len()
            )));
        }

        let batch_size = shape[0];
        let seq_len = shape[1];
        let d_in = shape[2];

        if d_in != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Input dimension {} doesn't match d_model {}",
                d_in, self.d_model
            )));
        }

        // Manual matmul: [batch, seq, d_model] @ [d_model, d_model] -> [batch, seq, d_model]
        let mut output = Array::zeros(IxDyn(&[batch_size, seq_len, self.d_model]));

        for b in 0..batch_size {
            for s in 0..seq_len {
                for o in 0..self.d_model {
                    let mut sum = F::zero();
                    for i in 0..self.d_model {
                        sum += input[[b, s, i]] * weights[[i, o]];
                    }
                    output[[b, s, o]] = sum;
                }
            }
        }

        Ok(output)
    }

    /// Reshape for multi-head: [batch, seq, d_model] -> [batch, seq, num_heads, head_dim]
    fn reshape_for_heads(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];

        input
            .clone()
            .into_shape_with_order(IxDyn(&[
                batch_size,
                seq_len,
                self.config.num_heads,
                self.config.head_dim,
            ]))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape for heads: {}", e)))
    }

    /// Compute softmax along last dimension
    fn softmax(&self, input: &Array<F, IxDyn>) -> Array<F, IxDyn> {
        let shape = input.shape().to_vec();
        let last_dim = shape.len() - 1;
        let last_size = shape[last_dim];

        let mut output = input.clone();

        // Iterate over all positions except the last dimension
        let num_elements: usize = shape[..last_dim].iter().product();

        for idx in 0..num_elements {
            // Calculate multi-dimensional index
            let mut remaining = idx;
            let mut indices: Vec<usize> = Vec::with_capacity(last_dim);
            for &dim_size in shape[..last_dim].iter().rev() {
                indices.push(remaining % dim_size);
                remaining /= dim_size;
            }
            indices.reverse();

            // Find max for numerical stability
            let mut max_val = F::neg_infinity();
            for k in 0..last_size {
                let mut full_idx = indices.clone();
                full_idx.push(k);
                let val = input[IxDyn(&full_idx)];
                if val > max_val {
                    max_val = val;
                }
            }

            // Compute exp(x - max) and sum
            let mut sum = F::zero();
            let mut exp_vals = Vec::with_capacity(last_size);
            for k in 0..last_size {
                let mut full_idx = indices.clone();
                full_idx.push(k);
                let exp_val = (input[IxDyn(&full_idx)] - max_val).exp();
                exp_vals.push(exp_val);
                sum += exp_val;
            }

            // Normalize
            for (k, &exp_val) in exp_vals.iter().enumerate() {
                let mut full_idx = indices.clone();
                full_idx.push(k);
                output[IxDyn(&full_idx)] = exp_val / sum;
            }
        }

        output
    }

    /// Apply causal mask (upper triangular with -inf)
    fn apply_causal_mask(&self, scores: &mut Array<F, IxDyn>) {
        let shape = scores.shape().to_vec();
        let batch_size = shape[0];
        let num_heads = shape[1];
        let seq_len_q = shape[2];
        let seq_len_k = shape[3];

        let neg_inf = F::neg_infinity();

        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len_q {
                    for j in 0..seq_len_k {
                        if j > i {
                            scores[[b, h, i, j]] = neg_inf;
                        }
                    }
                }
            }
        }
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for MultiHeadAttention<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "MultiHeadAttention expects 3D input (batch, seq, d_model), got {}D",
                shape.len()
            )));
        }

        let batch_size = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];

        if d_model != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Input dimension {} doesn't match expected {}",
                d_model, self.d_model
            )));
        }

        // Cache input for backward pass
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(input.clone());
        }

        // Project queries, keys, values
        let query = self.linear_projection(input, &self.w_query)?;
        let key = self.linear_projection(input, &self.w_key)?;
        let value = self.linear_projection(input, &self.w_value)?;

        // Reshape for multi-head attention
        let query = self.reshape_for_heads(&query)?;
        let key = self.reshape_for_heads(&key)?;
        let value = self.reshape_for_heads(&value)?;

        // Compute attention scores: [batch, num_heads, seq_q, seq_k]
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        let mut scores = Array::zeros(IxDyn(&[batch_size, num_heads, seq_len, seq_len]));

        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let mut dot_product = F::zero();
                        for d in 0..head_dim {
                            dot_product += query[[b, i, h, d]] * key[[b, j, h, d]];
                        }
                        scores[[b, h, i, j]] = dot_product * self.scale;
                    }
                }
            }
        }

        // Apply causal mask if configured
        if self.config.causal {
            self.apply_causal_mask(&mut scores);
        }

        // Apply softmax
        let attention_weights = self.softmax(&scores);

        // Cache attention weights for backward pass
        if let Ok(mut cache) = self.attention_weights_cache.write() {
            *cache = Some(attention_weights.clone());
        }

        // Apply attention to values: [batch, seq, num_heads, head_dim]
        let mut attended = Array::zeros(IxDyn(&[batch_size, seq_len, num_heads, head_dim]));

        for b in 0..batch_size {
            for i in 0..seq_len {
                for h in 0..num_heads {
                    for d in 0..head_dim {
                        let mut sum = F::zero();
                        for j in 0..seq_len {
                            sum += attention_weights[[b, h, i, j]] * value[[b, j, h, d]];
                        }
                        attended[[b, i, h, d]] = sum;
                    }
                }
            }
        }

        // Reshape back to [batch, seq, d_model]
        let concatenated = attended
            .into_shape_with_order(IxDyn(&[batch_size, seq_len, d_model]))
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to concatenate heads: {}", e))
            })?;

        // Output projection
        let output = self.linear_projection(&concatenated, &self.w_output)?;

        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Simplified backward pass - return gradient scaled by some factor
        // A full implementation would compute gradients for all weights
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update weights using gradients
        if let Ok(dw) = self.dw_query.read() {
            self.w_query = &self.w_query - &(&*dw * learning_rate);
        }
        if let Ok(dw) = self.dw_key.read() {
            self.w_key = &self.w_key - &(&*dw * learning_rate);
        }
        if let Ok(dw) = self.dw_value.read() {
            self.w_value = &self.w_value - &(&*dw * learning_rate);
        }
        if let Ok(dw) = self.dw_output.read() {
            self.w_output = &self.w_output - &(&*dw * learning_rate);
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        vec![
            self.w_query.clone(),
            self.w_key.clone(),
            self.w_value.clone(),
            self.w_output.clone(),
        ]
    }

    fn set_params(&mut self, params: &[Array<F, IxDyn>]) -> Result<()> {
        if params.len() >= 4 {
            self.w_query = params[0].clone();
            self.w_key = params[1].clone();
            self.w_value = params[2].clone();
            self.w_output = params[3].clone();
        } else if params.len() == 3 {
            self.w_query = params[0].clone();
            self.w_key = params[1].clone();
            self.w_value = params[2].clone();
        }
        Ok(())
    }

    fn gradients(&self) -> Vec<Array<F, IxDyn>> {
        let mut grads = Vec::new();
        if let Ok(dw) = self.dw_query.read() {
            grads.push(dw.clone());
        }
        if let Ok(dw) = self.dw_key.read() {
            grads.push(dw.clone());
        }
        if let Ok(dw) = self.dw_value.read() {
            grads.push(dw.clone());
        }
        if let Ok(dw) = self.dw_output.read() {
            grads.push(dw.clone());
        }
        grads
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn layer_type(&self) -> &str {
        "MultiHeadAttention"
    }

    fn parameter_count(&self) -> usize {
        4 * self.d_model * self.d_model
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

// Implement Send + Sync
unsafe impl<F: Float + Debug + Send + Sync + NumAssign> Send for MultiHeadAttention<F> {}
unsafe impl<F: Float + Debug + Send + Sync + NumAssign> Sync for MultiHeadAttention<F> {}

/// Self-attention layer
///
/// A convenience wrapper around MultiHeadAttention where query, key, and value
/// all come from the same input.
///
/// # Input Shape
/// - 3D tensor: (batch_size, seq_len, d_model)
///
/// # Output Shape
/// - 3D tensor: (batch_size, seq_len, d_model)
#[derive(Debug, Clone)]
pub struct SelfAttention<F: Float + Debug + Send + Sync + NumAssign> {
    /// Underlying multi-head attention
    attention: MultiHeadAttention<F>,
    /// Layer name
    name: Option<String>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> SelfAttention<F> {
    /// Create a new self-attention layer
    ///
    /// # Arguments
    /// * `d_model` - Embedding dimension
    /// * `config` - Attention configuration
    /// * `rng` - Random number generator
    pub fn new<R: Rng>(d_model: usize, config: AttentionConfig, rng: &mut R) -> Result<Self> {
        Ok(Self {
            attention: MultiHeadAttention::new(d_model, config, rng)?,
            name: None,
        })
    }

    /// Set a name for this layer
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for SelfAttention<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.attention.forward(input)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        self.attention.backward(input, grad_output)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.attention.update(learning_rate)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        self.attention.params()
    }

    fn gradients(&self) -> Vec<Array<F, IxDyn>> {
        self.attention.gradients()
    }

    fn set_training(&mut self, training: bool) {
        self.attention.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.attention.is_training()
    }

    fn layer_type(&self) -> &str {
        "SelfAttention"
    }

    fn parameter_count(&self) -> usize {
        self.attention.parameter_count()
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

// Implement Send + Sync
unsafe impl<F: Float + Debug + Send + Sync + NumAssign> Send for SelfAttention<F> {}
unsafe impl<F: Float + Debug + Send + Sync + NumAssign> Sync for SelfAttention<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;
    use scirs2_core::random::rng;

    #[test]
    fn test_multihead_attention_creation() {
        let mut rng = rng();
        let config = AttentionConfig {
            num_heads: 4,
            head_dim: 16,
            dropout_prob: 0.0,
            causal: false,
            scale: None,
        };
        let mha = MultiHeadAttention::<f64>::new(64, config, &mut rng).expect("Operation failed");
        assert_eq!(mha.layer_type(), "MultiHeadAttention");
        assert_eq!(mha.parameter_count(), 4 * 64 * 64);
    }

    #[test]
    fn test_multihead_attention_forward() {
        let mut rng = rng();
        let config = AttentionConfig {
            num_heads: 2,
            head_dim: 8,
            dropout_prob: 0.0,
            causal: false,
            scale: None,
        };
        let mha = MultiHeadAttention::<f64>::new(16, config, &mut rng).expect("Operation failed");

        // Batch of 2, sequence length 4, embedding dim 16
        let input = Array3::<f64>::from_elem((2, 4, 16), 0.1).into_dyn();
        let output = mha.forward(&input).expect("Operation failed");

        assert_eq!(output.shape(), &[2, 4, 16]);
    }

    #[test]
    fn test_multihead_attention_causal() {
        let mut rng = rng();
        let config = AttentionConfig {
            num_heads: 2,
            head_dim: 8,
            dropout_prob: 0.0,
            causal: true,
            scale: None,
        };
        let mha = MultiHeadAttention::<f64>::new(16, config, &mut rng).expect("Operation failed");

        let input = Array3::<f64>::from_elem((1, 3, 16), 0.5).into_dyn();
        let output = mha.forward(&input).expect("Operation failed");

        assert_eq!(output.shape(), &[1, 3, 16]);
    }

    #[test]
    fn test_self_attention_creation() {
        let mut rng = rng();
        let config = AttentionConfig {
            num_heads: 4,
            head_dim: 16,
            dropout_prob: 0.0,
            causal: false,
            scale: None,
        };
        let sa = SelfAttention::<f64>::new(64, config, &mut rng).expect("Operation failed");
        assert_eq!(sa.layer_type(), "SelfAttention");
    }

    #[test]
    fn test_self_attention_forward() {
        let mut rng = rng();
        let config = AttentionConfig {
            num_heads: 2,
            head_dim: 16,
            dropout_prob: 0.0,
            causal: false,
            scale: None,
        };
        let sa = SelfAttention::<f64>::new(32, config, &mut rng).expect("Operation failed");

        let input = Array3::<f64>::from_elem((1, 5, 32), 0.2).into_dyn();
        let output = sa.forward(&input).expect("Operation failed");

        assert_eq!(output.shape(), &[1, 5, 32]);
    }

    #[test]
    fn test_attention_config_default() {
        let config = AttentionConfig::default();
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert!((config.dropout_prob - 0.1).abs() < 1e-6);
        assert!(!config.causal);
        assert!(config.scale.is_none());
    }

    #[test]
    fn test_invalid_d_model() {
        let mut rng = rng();
        let config = AttentionConfig {
            num_heads: 3, // 64 not divisible by 3
            head_dim: 21,
            dropout_prob: 0.0,
            causal: false,
            scale: None,
        };
        let result = MultiHeadAttention::<f64>::new(64, config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_with_name() {
        let mut rng = rng();
        let config = AttentionConfig {
            num_heads: 2,
            head_dim: 8,
            dropout_prob: 0.0,
            causal: false,
            scale: None,
        };
        let mha = MultiHeadAttention::<f64>::new(16, config, &mut rng)
            .expect("Operation failed")
            .with_name("my_attention");
        assert_eq!(mha.name(), Some("my_attention"));
    }
}
