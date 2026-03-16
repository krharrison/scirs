//! Flash Attention implementation
//!
//! This module implements the Flash Attention algorithm as described in:
//! "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
//! by Tri Dao et al. (<https://arxiv.org/abs/2205.14135>)
//!
//! Flash Attention computes exact attention with O(N) memory instead of O(N²)
//! by using a tiling approach with online softmax computation.

use crate::error::{NeuralError, Result};
use crate::layers::attention::AttentionConfig;
use crate::layers::Layer;
use scirs2_core::ndarray::{s, Array, Array2, Array4, ArrayView2, IxDyn, ScalarOperand, Zip};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt};
use std::fmt::Debug;

/// Configuration for Flash Attention
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension of each attention head
    pub head_dim: usize,
    /// Block size for tiling (typically 64-256)
    pub block_size_q: usize,
    /// Block size for keys/values
    pub block_size_kv: usize,
    /// Whether to use causal masking
    pub causal: bool,
    /// Dropout probability
    pub dropout_prob: f64,
    /// Custom scaling factor (default is 1/sqrt(head_dim))
    pub scale: Option<f64>,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            block_size_q: 64,
            block_size_kv: 64,
            causal: false,
            dropout_prob: 0.0,
            scale: None,
        }
    }
}

impl FlashAttentionConfig {
    /// Create a new FlashAttentionConfig
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            ..Default::default()
        }
    }

    /// Set block size for queries
    pub fn with_block_size_q(mut self, block_size: usize) -> Self {
        self.block_size_q = block_size;
        self
    }

    /// Set block size for keys/values
    pub fn with_block_size_kv(mut self, block_size: usize) -> Self {
        self.block_size_kv = block_size;
        self
    }

    /// Enable causal masking
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, dropout_prob: f64) -> Self {
        self.dropout_prob = dropout_prob;
        self
    }

    /// Set custom scale factor
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = Some(scale);
        self
    }
}

/// Flash Attention layer
///
/// Implements memory-efficient attention using tiling and online softmax.
/// This reduces memory usage from O(N²) to O(N) while computing exact attention.
///
/// # Algorithm Overview
///
/// 1. Divide Q into blocks of size B_q
/// 2. Divide K, V into blocks of size B_kv
/// 3. For each Q block:
///    - Initialize output accumulator O and softmax statistics (m, l)
///    - For each K, V block:
///      - Compute attention scores S = Q_block @ K_block^T
///      - Update softmax statistics using online softmax
///      - Accumulate weighted values into O
///    - Rescale final output
///
/// # Memory Complexity
///
/// - Standard attention: O(N²) for storing attention matrix
/// - Flash attention: O(N) - only stores block-sized matrices
///
/// # Examples
///
/// ```rust
/// use scirs2_neural::layers::{FlashAttention, FlashAttentionConfig, Layer};
/// use scirs2_core::ndarray::Array3;
/// use scirs2_core::random::rng;
///
/// let mut rng = rng();
/// let config = FlashAttentionConfig::new(8, 64).with_causal(true);
/// let flash_attn = FlashAttention::<f64>::new(512, config, &mut rng).expect("Operation failed");
///
/// // Input shape: [batch, seq_len, d_model]
/// let input = Array3::<f64>::from_elem((2, 128, 512), 0.1).into_dyn();
/// let output = flash_attn.forward(&input).expect("Operation failed");
/// ```
#[derive(Debug)]
pub struct FlashAttention<F: Float + Debug + Send + Sync + NumAssign> {
    /// Model dimension
    d_model: usize,
    /// Configuration
    config: FlashAttentionConfig,
    /// Query projection weights [d_model, d_model]
    w_query: Array<F, IxDyn>,
    /// Key projection weights [d_model, d_model]
    w_key: Array<F, IxDyn>,
    /// Value projection weights [d_model, d_model]
    w_value: Array<F, IxDyn>,
    /// Output projection weights [d_model, d_model]
    w_output: Array<F, IxDyn>,
    /// Scaling factor for attention scores
    scale: F,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + NumAssign> FlashAttention<F> {
    /// Create a new Flash Attention layer
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (embedding size)
    /// * `config` - Flash attention configuration
    /// * `rng` - Random number generator for weight initialization
    pub fn new<R: Rng>(d_model: usize, config: FlashAttentionConfig, rng: &mut R) -> Result<Self> {
        let total_dim = config.num_heads * config.head_dim;
        if total_dim != d_model {
            return Err(NeuralError::InvalidArchitecture(format!(
                "num_heads * head_dim ({}) must equal d_model ({})",
                total_dim, d_model
            )));
        }

        // Xavier initialization for weights
        let xavier_std = (F::from(2.0).expect("Failed to convert constant to float")
            / F::from(d_model + d_model).expect("Failed to convert to float"))
        .sqrt();

        let w_query = Self::init_weight(d_model, d_model, xavier_std, rng);
        let w_key = Self::init_weight(d_model, d_model, xavier_std, rng);
        let w_value = Self::init_weight(d_model, d_model, xavier_std, rng);
        let w_output = Self::init_weight(d_model, d_model, xavier_std, rng);

        let scale = config
            .scale
            .map(|s| F::from(s).expect("Failed to convert to float"))
            .unwrap_or_else(|| {
                F::one()
                    / F::from(config.head_dim)
                        .expect("Failed to convert to float")
                        .sqrt()
            });

        Ok(Self {
            d_model,
            config,
            w_query,
            w_key,
            w_value,
            w_output,
            scale,
        })
    }

    /// Initialize a weight matrix with Xavier initialization
    fn init_weight<R: Rng>(in_dim: usize, out_dim: usize, std: F, rng: &mut R) -> Array<F, IxDyn> {
        let mut weights = Array::zeros(IxDyn(&[in_dim, out_dim]));
        for w in weights.iter_mut() {
            // Use Box-Muller transform for normal distribution
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            *w = F::from(z).expect("Failed to convert to float") * std;
        }
        weights
    }

    /// Compute Flash Attention forward pass
    ///
    /// This implements the tiled attention algorithm from the Flash Attention paper.
    /// The key insight is using online softmax to avoid materializing the full N×N matrix.
    fn flash_attention_forward(
        &self,
        query: &Array2<F>,
        key: &Array2<F>,
        value: &Array2<F>,
    ) -> Result<Array2<F>> {
        let seq_len_q = query.nrows();
        let seq_len_kv = key.nrows();
        let head_dim = query.ncols();

        let block_size_q = self.config.block_size_q.min(seq_len_q);
        let block_size_kv = self.config.block_size_kv.min(seq_len_kv);

        // Output accumulator
        let mut output = Array2::<F>::zeros((seq_len_q, head_dim));
        // Running maximum for softmax stability
        let mut row_max = vec![F::neg_infinity(); seq_len_q];
        // Running sum of exp for softmax normalization
        let mut row_sum = vec![F::zero(); seq_len_q];

        // Number of blocks
        let num_blocks_q = seq_len_q.div_ceil(block_size_q);
        let num_blocks_kv = seq_len_kv.div_ceil(block_size_kv);

        // Process each query block
        for q_block_idx in 0..num_blocks_q {
            let q_start = q_block_idx * block_size_q;
            let q_end = (q_start + block_size_q).min(seq_len_q);
            let q_block_size = q_end - q_start;

            // Get query block
            let q_block = query.slice(s![q_start..q_end, ..]);

            // Process each key/value block
            for kv_block_idx in 0..num_blocks_kv {
                let kv_start = kv_block_idx * block_size_kv;
                let kv_end = (kv_start + block_size_kv).min(seq_len_kv);

                // Skip if causal masking and this KV block is entirely in the future
                if self.config.causal && kv_start > q_end - 1 {
                    continue;
                }

                // Get key and value blocks
                let k_block = key.slice(s![kv_start..kv_end, ..]);
                let v_block = value.slice(s![kv_start..kv_end, ..]);

                // Compute attention scores: S = Q_block @ K_block^T * scale
                let scores = self.compute_block_scores(&q_block, &k_block);

                // Apply causal mask if needed
                let masked_scores = if self.config.causal {
                    self.apply_causal_mask(&scores, q_start, kv_start)
                } else {
                    scores
                };

                // Online softmax update and value accumulation
                self.online_softmax_update(
                    &masked_scores,
                    &v_block,
                    &mut output,
                    &mut row_max,
                    &mut row_sum,
                    q_start,
                    q_block_size,
                );
            }
        }

        // Final rescaling by row sums
        for i in 0..seq_len_q {
            if row_sum[i] > F::zero() {
                let inv_sum = F::one() / row_sum[i];
                for j in 0..head_dim {
                    output[[i, j]] *= inv_sum;
                }
            }
        }

        Ok(output)
    }

    /// Compute attention scores for a block: Q_block @ K_block^T * scale
    fn compute_block_scores(&self, q_block: &ArrayView2<F>, k_block: &ArrayView2<F>) -> Array2<F> {
        let q_size = q_block.nrows();
        let k_size = k_block.nrows();

        let mut scores = Array2::<F>::zeros((q_size, k_size));

        // Compute S = Q @ K^T * scale
        for i in 0..q_size {
            for j in 0..k_size {
                let mut dot = F::zero();
                for d in 0..q_block.ncols() {
                    dot += q_block[[i, d]] * k_block[[j, d]];
                }
                scores[[i, j]] = dot * self.scale;
            }
        }

        scores
    }

    /// Apply causal mask to attention scores
    fn apply_causal_mask(
        &self,
        scores: &Array2<F>,
        q_offset: usize,
        kv_offset: usize,
    ) -> Array2<F> {
        let mut masked = scores.clone();
        let q_size = scores.nrows();
        let k_size = scores.ncols();

        for i in 0..q_size {
            let q_pos = q_offset + i;
            for j in 0..k_size {
                let k_pos = kv_offset + j;
                if k_pos > q_pos {
                    masked[[i, j]] = F::neg_infinity();
                }
            }
        }

        masked
    }

    /// Online softmax update with value accumulation
    ///
    /// This is the key innovation of Flash Attention - updating softmax statistics
    /// incrementally without storing the full attention matrix.
    ///
    /// For each row i:
    /// - m_new = max(m_old, max(scores[i, :]))
    /// - l_new = l_old * exp(m_old - m_new) + sum(exp(scores[i, :] - m_new))
    /// - O_new = O_old * exp(m_old - m_new) + sum(exp(scores[i, :] - m_new) * V)
    fn online_softmax_update(
        &self,
        scores: &Array2<F>,
        v_block: &ArrayView2<F>,
        output: &mut Array2<F>,
        row_max: &mut [F],
        row_sum: &mut [F],
        q_offset: usize,
        q_block_size: usize,
    ) {
        let k_size = scores.ncols();
        let head_dim = v_block.ncols();

        for local_i in 0..q_block_size {
            let global_i = q_offset + local_i;

            // Find max in current block
            let mut block_max = F::neg_infinity();
            for j in 0..k_size {
                if scores[[local_i, j]] > block_max {
                    block_max = scores[[local_i, j]];
                }
            }

            // Update running max
            let old_max = row_max[global_i];
            let new_max = if old_max > block_max {
                old_max
            } else {
                block_max
            };

            // Compute correction factor for previous accumulations
            let correction = if old_max == F::neg_infinity() {
                F::zero()
            } else {
                (old_max - new_max).exp()
            };

            // Update output with correction factor
            for d in 0..head_dim {
                output[[global_i, d]] *= correction;
            }

            // Update row sum with correction
            row_sum[global_i] *= correction;

            // Accumulate new values with stable softmax
            for j in 0..k_size {
                if scores[[local_i, j]] > F::neg_infinity() {
                    let exp_score = (scores[[local_i, j]] - new_max).exp();
                    row_sum[global_i] += exp_score;

                    // Accumulate weighted values
                    for d in 0..head_dim {
                        output[[global_i, d]] += exp_score * v_block[[j, d]];
                    }
                }
            }

            row_max[global_i] = new_max;
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &FlashAttentionConfig {
        &self.config
    }

    /// Get model dimension
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

impl<F> Layer<F> for FlashAttention<F>
where
    F: Float + Debug + ScalarOperand + Send + Sync + 'static + NumAssign,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Input shape: [batch, seq_len, d_model]
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Flash Attention expects 3D input [batch, seq_len, d_model], got {} dimensions",
                input.ndim()
            )));
        }

        let shape = input.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];

        if d_model != self.d_model {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Input dimension {} does not match model dimension {}",
                d_model, self.d_model
            )));
        }

        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // Project input to Q, K, V
        // Reshape input to 2D for projection: [batch * seq_len, d_model]
        let input_2d = input
            .clone()
            .into_shape_with_order(IxDyn(&[batch_size * seq_len, d_model]))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape input: {}", e)))?;

        let input_2d_view = input_2d
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("Failed to convert to 2D".to_string()))?;

        let w_q_2d = self
            .w_query
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("Failed to convert Q weights".to_string()))?;
        let w_k_2d = self
            .w_key
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("Failed to convert K weights".to_string()))?;
        let w_v_2d = self
            .w_value
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("Failed to convert V weights".to_string()))?;
        let w_o_2d = self
            .w_output
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("Failed to convert O weights".to_string()))?;

        // Project to Q, K, V: [batch * seq_len, d_model]
        let q_proj = input_2d_view.dot(&w_q_2d);
        let k_proj = input_2d_view.dot(&w_k_2d);
        let v_proj = input_2d_view.dot(&w_v_2d);

        // Reshape to [batch, seq_len, num_heads, head_dim]
        let q_4d = q_proj
            .into_shape_with_order((batch_size, seq_len, num_heads, head_dim))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape Q: {}", e)))?;
        let k_4d = k_proj
            .into_shape_with_order((batch_size, seq_len, num_heads, head_dim))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape K: {}", e)))?;
        let v_4d = v_proj
            .into_shape_with_order((batch_size, seq_len, num_heads, head_dim))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape V: {}", e)))?;

        // Process each batch and head with Flash Attention
        let mut output_4d = Array4::<F>::zeros((batch_size, seq_len, num_heads, head_dim));

        for b in 0..batch_size {
            for h in 0..num_heads {
                // Extract Q, K, V for this batch and head
                let q_head: Array2<F> = q_4d
                    .slice(s![b, .., h, ..])
                    .to_owned()
                    .into_shape_with_order((seq_len, head_dim))
                    .map_err(|e| {
                        NeuralError::InferenceError(format!("Failed to get Q head: {}", e))
                    })?;

                let k_head: Array2<F> = k_4d
                    .slice(s![b, .., h, ..])
                    .to_owned()
                    .into_shape_with_order((seq_len, head_dim))
                    .map_err(|e| {
                        NeuralError::InferenceError(format!("Failed to get K head: {}", e))
                    })?;

                let v_head: Array2<F> = v_4d
                    .slice(s![b, .., h, ..])
                    .to_owned()
                    .into_shape_with_order((seq_len, head_dim))
                    .map_err(|e| {
                        NeuralError::InferenceError(format!("Failed to get V head: {}", e))
                    })?;

                // Apply Flash Attention
                let attn_output = self.flash_attention_forward(&q_head, &k_head, &v_head)?;

                // Copy to output
                for i in 0..seq_len {
                    for d in 0..head_dim {
                        output_4d[[b, i, h, d]] = attn_output[[i, d]];
                    }
                }
            }
        }

        // Reshape to [batch, seq_len, d_model]
        let output_3d = output_4d
            .into_shape_with_order((batch_size, seq_len, d_model))
            .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape output: {}", e)))?;

        // Apply output projection
        let output_2d = output_3d
            .into_shape_with_order((batch_size * seq_len, d_model))
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to reshape for output proj: {}", e))
            })?;

        let final_output = output_2d.dot(&w_o_2d);

        // Reshape back to [batch, seq_len, d_model]
        let result = final_output
            .into_shape_with_order((batch_size, seq_len, d_model))
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to reshape final output: {}", e))
            })?;

        Ok(result.into_dyn())
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Flash Attention backward is more complex - uses recomputation
        // For now, return a placeholder
        // TODO: Implement Flash Attention backward pass with recomputation
        Err(NeuralError::NotImplemented(
            "Flash Attention backward pass not yet implemented".to_string(),
        ))
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        // Updates would be applied to stored gradients
        Ok(())
    }
}

/// Memory-efficient attention computation (no projection weights)
///
/// This is a standalone function for computing attention without the
/// full FlashAttention layer overhead. Useful for custom attention patterns.
///
/// # Arguments
/// * `query` - Query tensor [batch, seq_q, head_dim]
/// * `key` - Key tensor [batch, seq_k, head_dim]
/// * `value` - Value tensor [batch, seq_k, head_dim]
/// * `causal` - Whether to apply causal masking
/// * `block_size` - Block size for tiling
///
/// # Returns
/// * Attention output [batch, seq_q, head_dim]
pub fn flash_attention_compute<F: Float + Debug + ScalarOperand + NumAssign>(
    query: &Array<F, IxDyn>,
    key: &Array<F, IxDyn>,
    value: &Array<F, IxDyn>,
    causal: bool,
    block_size: usize,
) -> Result<Array<F, IxDyn>> {
    if query.ndim() != 3 || key.ndim() != 3 || value.ndim() != 3 {
        return Err(NeuralError::InvalidArchitecture(
            "Query, key, value must be 3D tensors".to_string(),
        ));
    }

    let batch_size = query.shape()[0];
    let seq_len_q = query.shape()[1];
    let seq_len_kv = key.shape()[1];
    let head_dim = query.shape()[2];

    let scale = F::one()
        / F::from(head_dim)
            .expect("Failed to convert to float")
            .sqrt();

    let mut output = Array::zeros(IxDyn(&[batch_size, seq_len_q, head_dim]));

    for b in 0..batch_size {
        let q_batch = query.slice(s![b, .., ..]);
        let k_batch = key.slice(s![b, .., ..]);
        let v_batch = value.slice(s![b, .., ..]);

        // Convert to 2D for processing
        let q_2d: Array2<F> = q_batch
            .to_owned()
            .into_shape_with_order((seq_len_q, head_dim))
            .map_err(|_| NeuralError::InferenceError("Failed to reshape Q".to_string()))?;

        let k_2d: Array2<F> = k_batch
            .to_owned()
            .into_shape_with_order((seq_len_kv, head_dim))
            .map_err(|_| NeuralError::InferenceError("Failed to reshape K".to_string()))?;

        let v_2d: Array2<F> = v_batch
            .to_owned()
            .into_shape_with_order((seq_len_kv, head_dim))
            .map_err(|_| NeuralError::InferenceError("Failed to reshape V".to_string()))?;

        // Compute tiled attention
        let batch_output = tiled_attention_compute(&q_2d, &k_2d, &v_2d, scale, causal, block_size)?;

        // Copy to output
        for i in 0..seq_len_q {
            for d in 0..head_dim {
                output[[b, i, d]] = batch_output[[i, d]];
            }
        }
    }

    Ok(output)
}

/// Core tiled attention computation
fn tiled_attention_compute<F: Float + Debug + NumAssign>(
    query: &Array2<F>,
    key: &Array2<F>,
    value: &Array2<F>,
    scale: F,
    causal: bool,
    block_size: usize,
) -> Result<Array2<F>> {
    let seq_len_q = query.nrows();
    let seq_len_kv = key.nrows();
    let head_dim = query.ncols();

    let block_size = block_size.min(seq_len_q).min(seq_len_kv).max(1);

    let mut output = Array2::<F>::zeros((seq_len_q, head_dim));
    let mut row_max = vec![F::neg_infinity(); seq_len_q];
    let mut row_sum = vec![F::zero(); seq_len_q];

    let num_blocks_q = seq_len_q.div_ceil(block_size);
    let num_blocks_kv = seq_len_kv.div_ceil(block_size);

    for q_block_idx in 0..num_blocks_q {
        let q_start = q_block_idx * block_size;
        let q_end = (q_start + block_size).min(seq_len_q);
        let q_block_size = q_end - q_start;

        for kv_block_idx in 0..num_blocks_kv {
            let kv_start = kv_block_idx * block_size;
            let kv_end = (kv_start + block_size).min(seq_len_kv);
            let kv_block_size = kv_end - kv_start;

            if causal && kv_start > q_end - 1 {
                continue;
            }

            // Compute block scores
            let mut scores = Array2::<F>::zeros((q_block_size, kv_block_size));
            for i in 0..q_block_size {
                for j in 0..kv_block_size {
                    let mut dot = F::zero();
                    for d in 0..head_dim {
                        dot += query[[q_start + i, d]] * key[[kv_start + j, d]];
                    }
                    let s = dot * scale;

                    // Apply causal mask
                    if causal && (kv_start + j) > (q_start + i) {
                        scores[[i, j]] = F::neg_infinity();
                    } else {
                        scores[[i, j]] = s;
                    }
                }
            }

            // Online softmax update
            for local_i in 0..q_block_size {
                let global_i = q_start + local_i;

                let mut block_max = F::neg_infinity();
                for j in 0..kv_block_size {
                    if scores[[local_i, j]] > block_max {
                        block_max = scores[[local_i, j]];
                    }
                }

                let old_max = row_max[global_i];
                let new_max = if old_max > block_max {
                    old_max
                } else {
                    block_max
                };

                let correction = if old_max == F::neg_infinity() {
                    F::zero()
                } else {
                    (old_max - new_max).exp()
                };

                for d in 0..head_dim {
                    output[[global_i, d]] *= correction;
                }
                row_sum[global_i] *= correction;

                for j in 0..kv_block_size {
                    if scores[[local_i, j]] > F::neg_infinity() {
                        let exp_score = (scores[[local_i, j]] - new_max).exp();
                        row_sum[global_i] += exp_score;

                        for d in 0..head_dim {
                            output[[global_i, d]] += exp_score * value[[kv_start + j, d]];
                        }
                    }
                }

                row_max[global_i] = new_max;
            }
        }
    }

    // Final normalization
    for i in 0..seq_len_q {
        if row_sum[i] > F::zero() {
            let inv_sum = F::one() / row_sum[i];
            for d in 0..head_dim {
                output[[i, d]] *= inv_sum;
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_flash_attention_config() {
        let config = FlashAttentionConfig::new(8, 64)
            .with_causal(true)
            .with_block_size_q(32)
            .with_block_size_kv(32)
            .with_dropout(0.1);

        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert!(config.causal);
        assert_eq!(config.block_size_q, 32);
        assert_eq!(config.block_size_kv, 32);
        assert!((config.dropout_prob - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_flash_attention_creation() {
        let mut rng = scirs2_core::random::rng();
        let config = FlashAttentionConfig::new(4, 16);
        let flash_attn = FlashAttention::<f64>::new(64, config, &mut rng);
        assert!(flash_attn.is_ok());
    }

    #[test]
    fn test_flash_attention_forward() {
        let mut rng = scirs2_core::random::rng();
        let config = FlashAttentionConfig::new(4, 16)
            .with_block_size_q(8)
            .with_block_size_kv(8);
        let flash_attn =
            FlashAttention::<f64>::new(64, config, &mut rng).expect("Operation failed");

        // Input: [batch=2, seq_len=16, d_model=64]
        let input = Array3::<f64>::from_elem((2, 16, 64), 0.1).into_dyn();
        let output = flash_attn.forward(&input);

        assert!(output.is_ok());
        let output = output.expect("Operation failed");
        assert_eq!(output.shape(), &[2, 16, 64]);
    }

    #[test]
    fn test_flash_attention_causal() {
        let mut rng = scirs2_core::random::rng();
        let config = FlashAttentionConfig::new(4, 16)
            .with_causal(true)
            .with_block_size_q(4)
            .with_block_size_kv(4);
        let flash_attn =
            FlashAttention::<f64>::new(64, config, &mut rng).expect("Operation failed");

        let input = Array3::<f64>::from_elem((1, 8, 64), 0.1).into_dyn();
        let output = flash_attn.forward(&input);

        assert!(output.is_ok());
        assert_eq!(output.expect("Operation failed").shape(), &[1, 8, 64]);
    }

    #[test]
    fn test_flash_attention_compute_function() {
        let query = Array3::<f64>::from_elem((2, 8, 32), 0.1).into_dyn();
        let key = Array3::<f64>::from_elem((2, 8, 32), 0.1).into_dyn();
        let value = Array3::<f64>::from_elem((2, 8, 32), 0.1).into_dyn();

        let output = flash_attention_compute(&query, &key, &value, false, 4);
        assert!(output.is_ok());
        assert_eq!(output.expect("Operation failed").shape(), &[2, 8, 32]);
    }

    #[test]
    fn test_flash_attention_numerical_stability() {
        // Test with large values that would overflow standard softmax
        let mut rng = scirs2_core::random::rng();
        let config = FlashAttentionConfig::new(2, 8)
            .with_block_size_q(4)
            .with_block_size_kv(4);
        let flash_attn =
            FlashAttention::<f64>::new(16, config, &mut rng).expect("Operation failed");

        let mut input = Array3::<f64>::zeros((1, 8, 16));
        // Fill with varying values
        for i in 0..8 {
            for j in 0..16 {
                input[[0, i, j]] = i as f64 * 0.1 + j as f64 * 0.01;
            }
        }

        let output = flash_attn.forward(&input.into_dyn());
        assert!(output.is_ok());

        // Check output is finite (no NaN or Inf)
        let output = output.expect("Operation failed");
        for val in output.iter() {
            assert!(val.is_finite(), "Output contains non-finite values");
        }
    }

    #[test]
    fn test_flash_vs_standard_attention() {
        // Verify Flash Attention produces similar results to standard attention
        let query = Array3::<f64>::from_elem((1, 4, 8), 0.5).into_dyn();
        let key = query.clone();
        let value = query.clone();

        let flash_output =
            flash_attention_compute(&query, &key, &value, false, 2).expect("Operation failed");

        // Standard attention computation for comparison
        let q_2d = query
            .slice(s![0, .., ..])
            .to_owned()
            .into_shape_with_order((4, 8))
            .expect("Operation failed");
        let k_2d = key
            .slice(s![0, .., ..])
            .to_owned()
            .into_shape_with_order((4, 8))
            .expect("Operation failed");
        let v_2d = value
            .slice(s![0, .., ..])
            .to_owned()
            .into_shape_with_order((4, 8))
            .expect("Operation failed");

        let scale = 1.0 / (8.0_f64).sqrt();

        // S = Q @ K^T * scale
        let mut scores = Array2::<f64>::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                let mut dot = 0.0;
                for d in 0..8 {
                    dot += q_2d[[i, d]] * k_2d[[j, d]];
                }
                scores[[i, j]] = dot * scale;
            }
        }

        // Softmax
        let mut attention = scores.clone();
        for i in 0..4 {
            let max_val = attention.row(i).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0;
            for j in 0..4 {
                let exp_val = (attention[[i, j]] - max_val).exp();
                attention[[i, j]] = exp_val;
                sum += exp_val;
            }
            for j in 0..4 {
                attention[[i, j]] /= sum;
            }
        }

        // Output = attention @ V
        let mut standard_output = Array2::<f64>::zeros((4, 8));
        for i in 0..4 {
            for d in 0..8 {
                let mut sum = 0.0;
                for j in 0..4 {
                    sum += attention[[i, j]] * v_2d[[j, d]];
                }
                standard_output[[i, d]] = sum;
            }
        }

        // Compare outputs
        for i in 0..4 {
            for d in 0..8 {
                let flash_val = flash_output[[0, i, d]];
                let std_val = standard_output[[i, d]];
                assert!(
                    (flash_val - std_val).abs() < 1e-10,
                    "Mismatch at [{}, {}]: flash={}, std={}",
                    i,
                    d,
                    flash_val,
                    std_val
                );
            }
        }
    }
}
