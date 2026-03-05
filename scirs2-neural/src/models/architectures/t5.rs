//! T5 (Text-to-Text Transfer Transformer) implementation
//!
//! T5 is a unified framework that converts all text-based NLP problems into a
//! text-to-text format. It uses a standard encoder-decoder transformer with
//! relative positional embeddings (T5-style bucket-based biases) and no
//! absolute positional encodings.
//!
//! Reference: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
//! Raffel et al. (2020) <https://arxiv.org/abs/1910.10683>

use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Dropout, Layer, LayerNorm};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::simd_ops::SimdUnifiedOps;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a T5 model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T5Config {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Model (embedding) dimension
    pub d_model: usize,
    /// Inner dimension of each feed-forward layer
    pub d_ff: usize,
    /// Key/Value projection dimension (per head)
    pub d_kv: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of encoder layers
    pub num_encoder_layers: usize,
    /// Number of decoder layers
    pub num_decoder_layers: usize,
    /// Dropout probability
    pub dropout_rate: f64,
    /// Number of relative attention buckets
    pub relative_attention_num_buckets: usize,
    /// Maximum distance for relative position encoding
    pub relative_attention_max_distance: usize,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Decoder start token id
    pub decoder_start_token_id: usize,
    /// EOS token id
    pub eos_token_id: usize,
    /// PAD token id
    pub pad_token_id: usize,
}

impl Default for T5Config {
    fn default() -> Self {
        // T5-Small configuration
        Self {
            vocab_size: 32128,
            d_model: 512,
            d_ff: 2048,
            d_kv: 64,
            num_heads: 8,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
            dropout_rate: 0.1,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            layer_norm_eps: 1e-6,
            decoder_start_token_id: 0,
            eos_token_id: 1,
            pad_token_id: 0,
        }
    }
}

impl T5Config {
    /// T5-Small (60M parameters)
    pub fn t5_small() -> Self {
        Self::default()
    }

    /// T5-Base (220M parameters)
    pub fn t5_base() -> Self {
        Self {
            d_model: 768,
            d_ff: 3072,
            d_kv: 64,
            num_heads: 12,
            num_encoder_layers: 12,
            num_decoder_layers: 12,
            ..Self::default()
        }
    }

    /// T5-Large (770M parameters)
    pub fn t5_large() -> Self {
        Self {
            d_model: 1024,
            d_ff: 4096,
            d_kv: 64,
            num_heads: 16,
            num_encoder_layers: 24,
            num_decoder_layers: 24,
            ..Self::default()
        }
    }

    /// T5-3B parameters
    pub fn t5_3b() -> Self {
        Self {
            d_model: 1024,
            d_ff: 16384,
            d_kv: 128,
            num_heads: 32,
            num_encoder_layers: 24,
            num_decoder_layers: 24,
            ..Self::default()
        }
    }

    /// T5-11B parameters
    pub fn t5_11b() -> Self {
        Self {
            d_model: 1024,
            d_ff: 65536,
            d_kv: 128,
            num_heads: 128,
            num_encoder_layers: 24,
            num_decoder_layers: 24,
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// T5 Relative Position Bias
// ---------------------------------------------------------------------------

/// Relative position bias used in T5's attention mechanism
///
/// Instead of absolute positional embeddings, T5 uses learned relative bias
/// values that are added to attention logits based on the relative distance
/// between query and key positions, bucketed into `num_buckets` categories.
#[derive(Debug, Clone)]
pub struct T5RelativeAttentionBias<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand> {
    /// Learnable bias table: [num_buckets, num_heads]
    bias_table: Array<F, IxDyn>,
    num_buckets: usize,
    max_distance: usize,
    num_heads: usize,
    is_bidirectional: bool,
}

impl<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand + FromPrimitive + ToPrimitive>
    T5RelativeAttentionBias<F>
{
    pub fn new(
        num_buckets: usize,
        max_distance: usize,
        num_heads: usize,
        is_bidirectional: bool,
    ) -> Self {
        let bias_table = Array::zeros(IxDyn(&[num_buckets, num_heads]));
        Self {
            bias_table,
            num_buckets,
            max_distance,
            num_heads,
            is_bidirectional,
        }
    }

    /// Compute the bucket index for a relative position
    fn relative_position_bucket(&self, relative_position: isize) -> usize {
        let mut n = relative_position;
        let mut num_buckets = self.num_buckets;
        let mut ret = 0usize;

        if self.is_bidirectional {
            num_buckets /= 2;
            if n < 0 {
                ret += num_buckets;
                n = -n;
            }
        } else {
            n = n.min(0).abs();
        }

        // Half of the buckets are for exact increments in positions
        let max_exact = num_buckets / 2;
        let is_small = n < max_exact as isize;

        if is_small {
            ret += n as usize;
        } else {
            let n_f = n as f64;
            let max_exact_f = max_exact as f64;
            let max_dist = self.max_distance as f64;
            let log_val = (n_f / max_exact_f).ln() / (max_dist / max_exact_f).ln();
            let scaled = log_val * (num_buckets - max_exact) as f64;
            let bucket = max_exact + scaled as usize;
            ret += bucket.min(num_buckets - 1);
        }
        ret.min(self.num_buckets - 1)
    }

    /// Compute relative position biases for a given query/key sequence length pair
    ///
    /// Returns shape: [1, num_heads, query_len, key_len]
    pub fn compute_bias(&self, query_len: usize, key_len: usize) -> Array<F, IxDyn> {
        let mut biases = Array::zeros(IxDyn(&[1, self.num_heads, query_len, key_len]));
        for qi in 0..query_len {
            for ki in 0..key_len {
                let rel = (ki as isize) - (qi as isize);
                let bucket = self.relative_position_bucket(rel);
                for h in 0..self.num_heads {
                    biases[[0, h, qi, ki]] = self.bias_table[[bucket, h]];
                }
            }
        }
        biases
    }

    pub fn parameter_count(&self) -> usize {
        self.bias_table.len()
    }
}

// ---------------------------------------------------------------------------
// T5 Layer Normalization (RMS Norm without centering)
// ---------------------------------------------------------------------------

/// T5-style RMS Layer Normalization (no bias, no mean subtraction)
#[derive(Debug, Clone)]
pub struct T5LayerNorm<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand> {
    weight: Array<F, IxDyn>,
    eps: f64,
    dim: usize,
}

impl<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand + FromPrimitive + ToPrimitive>
    T5LayerNorm<F>
{
    pub fn new(dim: usize, eps: f64) -> Self {
        let weight = Array::ones(IxDyn(&[dim]));
        Self { weight, eps, dim }
    }

    /// Forward: apply RMS norm along last dimension
    pub fn forward(&self, x: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = x.shape().to_vec();
        let last_dim = *shape.last().ok_or_else(|| {
            NeuralError::InvalidArgument("empty input".to_string())
        })?;
        if last_dim != self.dim {
            return Err(NeuralError::InvalidArgument(format!(
                "T5LayerNorm: expected last dim {}, got {last_dim}",
                self.dim
            )));
        }
        let outer: usize = shape[..shape.len() - 1].iter().product();
        let mut out = x.clone();
        let eps = F::from(self.eps).unwrap_or(F::from(1e-6).unwrap_or(F::zero()));

        for i in 0..outer {
            // Compute mean square
            let mut ms = F::zero();
            for j in 0..last_dim {
                let flat_idx = i * last_dim + j;
                let v = x.as_slice().map(|s| s[flat_idx]).unwrap_or(F::zero());
                ms = ms + v * v;
            }
            ms = ms / F::from(last_dim as f64).unwrap_or(F::one());
            let rms = (ms + eps).sqrt();

            for j in 0..last_dim {
                let flat_idx = i * last_dim + j;
                let v = x.as_slice().map(|s| s[flat_idx]).unwrap_or(F::zero());
                let normed = v / rms * self.weight[[j]];
                if let Some(s) = out.as_slice_mut() {
                    s[flat_idx] = normed;
                }
            }
        }
        Ok(out)
    }

    pub fn update(&mut self, _lr: F) -> Result<()> {
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// T5 Feed-Forward Network (dense-ReLU-dense or GEGLU variant)
// ---------------------------------------------------------------------------

/// T5 Feed-Forward layer
///
/// T5 uses a variant with two independent linear projections (gated GEGLU):
///   FFN(x) = w2 * (RELU(w1 * x) * w3 * x)
/// But for simplicity and compatibility this implements the standard variant:
///   FFN(x) = w2 * RELU(w1 * x)
/// with pre-norm applied by the calling block.
#[derive(Debug, Clone)]
pub struct T5FeedForward<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand> {
    wi: Dense<F>,
    wo: Dense<F>,
    dropout: Dropout<F>,
    norm: T5LayerNorm<F>,
}

impl<
        F: Float
            + Debug
            + Send
            + Sync
            + NumAssign
            + ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + 'static,
    > T5FeedForward<F>
{
    pub fn new<R: Rng + Clone + Send + Sync + 'static>(d_model: usize, d_ff: usize, dropout: f64, eps: f64, rng: &mut R) -> Result<Self> {
        let wi = Dense::<F>::new(d_model, d_ff, Some("relu"), rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("wi failed: {e}")))?;
        let wo = Dense::<F>::new(d_ff, d_model, None, rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("wo failed: {e}")))?;
        let dropout_layer = Dropout::<F>::new(dropout, rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("dropout failed: {e}")))?;
        let norm = T5LayerNorm::<F>::new(d_model, eps);
        Ok(Self {
            wi,
            wo,
            dropout: dropout_layer,
            norm,
        })
    }

    pub fn forward(&self, x: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Pre-norm
        let xn = self.norm.forward(x)?;
        // FFN
        let h = self.wi.forward(&xn)?;
        let h = self.dropout.forward(&h)?;
        let h = self.wo.forward(&h)?;
        let h = self.dropout.forward(&h)?;
        // Residual
        Ok(x + &h)
    }

    pub fn update(&mut self, lr: F) -> Result<()> {
        self.wi.update(lr)?;
        self.wo.update(lr)?;
        self.norm.update(lr)?;
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        self.wi.parameter_count() + self.wo.parameter_count() + self.norm.parameter_count()
    }
}

// ---------------------------------------------------------------------------
// T5 Attention
// ---------------------------------------------------------------------------

/// T5 multi-head attention with optional cross-attention
///
/// Supports both self-attention (encoder) and cross-attention (decoder).
/// Uses T5's relative position bias when `is_self_attention` is true.
#[derive(Debug, Clone)]
pub struct T5Attention<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand> {
    q: Dense<F>,
    k: Dense<F>,
    v: Dense<F>,
    o: Dense<F>,
    norm: T5LayerNorm<F>,
    dropout: Dropout<F>,
    rel_bias: Option<T5RelativeAttentionBias<F>>,
    num_heads: usize,
    d_kv: usize,
    d_model: usize,
}

impl<
        F: Float
            + Debug
            + Send
            + Sync
            + NumAssign
            + ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + 'static,
    > T5Attention<F>
{
    /// Create a new T5 attention layer
    ///
    /// `is_self_attention`: if true, includes relative position bias
    /// `is_bidirectional`: if true, relative bias uses bidirectional buckets (for encoder)
    pub fn new<R: Rng + Clone + Send + Sync + 'static>(
        d_model: usize,
        d_kv: usize,
        num_heads: usize,
        dropout: f64,
        eps: f64,
        is_self_attention: bool,
        is_bidirectional: bool,
        num_buckets: usize,
        max_distance: usize,
        rng: &mut R,
    ) -> Result<Self> {
        let inner_dim = num_heads * d_kv;
        let q = Dense::<F>::new(d_model, inner_dim, None, rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("q failed: {e}")))?;
        let k = Dense::<F>::new(d_model, inner_dim, None, rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("k failed: {e}")))?;
        let v = Dense::<F>::new(d_model, inner_dim, None, rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("v failed: {e}")))?;
        let o = Dense::<F>::new(inner_dim, d_model, None, rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("o failed: {e}")))?;
        let norm = T5LayerNorm::<F>::new(d_model, eps);
        let dropout_layer = Dropout::<F>::new(dropout, rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("dropout failed: {e}")))?;

        let rel_bias = if is_self_attention {
            Some(T5RelativeAttentionBias::<F>::new(
                num_buckets,
                max_distance,
                num_heads,
                is_bidirectional,
            ))
        } else {
            None
        };

        Ok(Self {
            q,
            k,
            v,
            o,
            norm,
            dropout: dropout_layer,
            rel_bias,
            num_heads,
            d_kv,
            d_model,
        })
    }

    /// Forward attention
    ///
    /// `hidden`: [batch, seq_q, d_model] (query source)
    /// `key_value`: [batch, seq_kv, d_model] (for cross-attention, else same as hidden)
    /// `mask`: optional [batch, 1, seq_q, seq_kv] additive mask
    pub fn forward(
        &self,
        hidden: &Array<F, IxDyn>,
        key_value: &Array<F, IxDyn>,
        mask: Option<&Array<F, IxDyn>>,
    ) -> Result<Array<F, IxDyn>> {
        let h_shape = hidden.shape();
        let kv_shape = key_value.shape();
        if h_shape.len() != 3 || kv_shape.len() != 3 {
            return Err(NeuralError::InvalidArgument(
                "T5Attention expects 3D [B, L, D] tensors".to_string(),
            ));
        }
        let batch = h_shape[0];
        let seq_q = h_shape[1];
        let seq_kv = kv_shape[1];
        let inner_dim = self.num_heads * self.d_kv;

        // Pre-norm on query source
        let hidden_norm = self.norm.forward(hidden)?;

        // Project Q, K, V
        let q_out = self.q.forward(&hidden_norm)?; // [B, seq_q, inner_dim]
        let k_out = self.k.forward(key_value)?;    // [B, seq_kv, inner_dim]
        let v_out = self.v.forward(key_value)?;    // [B, seq_kv, inner_dim]

        // Split into heads: [B, num_heads, seq, d_kv]
        let q4 = reshape_to_heads(&q_out, batch, seq_q, self.num_heads, self.d_kv)?;
        let k4 = reshape_to_heads(&k_out, batch, seq_kv, self.num_heads, self.d_kv)?;
        let v4 = reshape_to_heads(&v_out, batch, seq_kv, self.num_heads, self.d_kv)?;

        // Compute attention scores: [B, num_heads, seq_q, seq_kv]
        let scale = F::from(1.0 / (self.d_kv as f64).sqrt()).unwrap_or(F::one());
        let mut scores = Array::zeros(IxDyn(&[batch, self.num_heads, seq_q, seq_kv]));
        for b in 0..batch {
            for h in 0..self.num_heads {
                for qi in 0..seq_q {
                    for ki in 0..seq_kv {
                        let mut dot = F::zero();
                        for d in 0..self.d_kv {
                            dot = dot + q4[[b, h, qi, d]] * k4[[b, h, ki, d]];
                        }
                        scores[[b, h, qi, ki]] = dot * scale;
                    }
                }
            }
        }

        // Add relative position bias if available
        if let Some(ref rb) = self.rel_bias {
            let bias = rb.compute_bias(seq_q, seq_kv);
            for b in 0..batch {
                for h in 0..self.num_heads {
                    for qi in 0..seq_q {
                        for ki in 0..seq_kv {
                            scores[[b, h, qi, ki]] =
                                scores[[b, h, qi, ki]] + bias[[0, h, qi, ki]];
                        }
                    }
                }
            }
        }

        // Apply mask (additive)
        if let Some(m) = mask {
            let m_shape = m.shape();
            let m_batch = m_shape[0].min(batch);
            let m_seq_q = if m_shape.len() >= 3 { m_shape[2].min(seq_q) } else { seq_q };
            let m_seq_kv = if m_shape.len() >= 4 { m_shape[3].min(seq_kv) } else { seq_kv };
            for b in 0..m_batch {
                for h in 0..self.num_heads {
                    for qi in 0..m_seq_q {
                        for ki in 0..m_seq_kv {
                            let mask_val = if m_shape.len() == 4 {
                                m[[b, 0, qi, ki]]
                            } else if m_shape.len() == 3 {
                                m[[b, qi, ki]]
                            } else {
                                F::zero()
                            };
                            scores[[b, h, qi, ki]] = scores[[b, h, qi, ki]] + mask_val;
                        }
                    }
                }
            }
        }

        // Softmax
        softmax_4d_last(&mut scores)?;

        // Dropout on attention weights
        let scores_3d = scores
            .clone()
            .into_shape_with_order(IxDyn(&[batch * self.num_heads, seq_q, seq_kv]))
            .map_err(|e| NeuralError::InvalidArgument(format!("reshape: {e}")))?
            .to_owned();
        let scores_drop = self.dropout.forward(&scores_3d)?;
        let scores_4d = scores_drop
            .into_shape_with_order(IxDyn(&[batch, self.num_heads, seq_q, seq_kv]))
            .map_err(|e| NeuralError::InvalidArgument(format!("reshape: {e}")))?
            .to_owned();

        // Context: [B, num_heads, seq_q, d_kv]
        let mut ctx = Array::zeros(IxDyn(&[batch, self.num_heads, seq_q, self.d_kv]));
        for b in 0..batch {
            for h in 0..self.num_heads {
                for qi in 0..seq_q {
                    for d in 0..self.d_kv {
                        let mut acc = F::zero();
                        for ki in 0..seq_kv {
                            acc = acc + scores_4d[[b, h, qi, ki]] * v4[[b, h, ki, d]];
                        }
                        ctx[[b, h, qi, d]] = acc;
                    }
                }
            }
        }

        // Merge heads: [B, seq_q, inner_dim]
        let mut merged = Array::zeros(IxDyn(&[batch, seq_q, inner_dim]));
        for b in 0..batch {
            for qi in 0..seq_q {
                for h in 0..self.num_heads {
                    for d in 0..self.d_kv {
                        merged[[b, qi, h * self.d_kv + d]] = ctx[[b, h, qi, d]];
                    }
                }
            }
        }

        // Output projection
        let out = self.o.forward(&merged)?;
        let out = self.dropout.forward(&out)?;

        // Residual connection
        Ok(hidden + &out)
    }

    pub fn update(&mut self, lr: F) -> Result<()> {
        self.q.update(lr)?;
        self.k.update(lr)?;
        self.v.update(lr)?;
        self.o.update(lr)?;
        self.norm.update(lr)?;
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        let inner_dim = self.num_heads * self.d_kv;
        let qkvo = 4 * self.d_model * inner_dim;
        let norm = self.norm.parameter_count();
        let rb = self.rel_bias.as_ref().map(|r| r.parameter_count()).unwrap_or(0);
        qkvo + norm + rb
    }
}

// ---------------------------------------------------------------------------
// T5 Encoder Layer
// ---------------------------------------------------------------------------

/// A single T5 encoder layer: self-attention + FFN
#[derive(Debug, Clone)]
pub struct T5EncoderLayer<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand> {
    self_attn: T5Attention<F>,
    ffn: T5FeedForward<F>,
}

impl<
        F: Float
            + Debug
            + Send
            + Sync
            + NumAssign
            + ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + 'static,
    > T5EncoderLayer<F>
{
    pub fn new<R: Rng + Clone + Send + Sync + 'static>(config: &T5Config, is_first_layer: bool, rng: &mut R) -> Result<Self> {
        let self_attn = T5Attention::<F>::new(
            config.d_model,
            config.d_kv,
            config.num_heads,
            config.dropout_rate,
            config.layer_norm_eps,
            true,
            true, // bidirectional for encoder
            if is_first_layer { config.relative_attention_num_buckets } else { 0 },
            config.relative_attention_max_distance,
            rng,
        )?;
        let ffn = T5FeedForward::<F>::new(
            config.d_model,
            config.d_ff,
            config.dropout_rate,
            config.layer_norm_eps,
            rng,
        )?;
        Ok(Self { self_attn, ffn })
    }

    pub fn forward(
        &self,
        hidden: &Array<F, IxDyn>,
        mask: Option<&Array<F, IxDyn>>,
    ) -> Result<Array<F, IxDyn>> {
        let h = self.self_attn.forward(hidden, hidden, mask)?;
        self.ffn.forward(&h)
    }

    pub fn update(&mut self, lr: F) -> Result<()> {
        self.self_attn.update(lr)?;
        self.ffn.update(lr)?;
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        self.self_attn.parameter_count() + self.ffn.parameter_count()
    }
}

// ---------------------------------------------------------------------------
// T5 Decoder Layer
// ---------------------------------------------------------------------------

/// A single T5 decoder layer: self-attention + cross-attention + FFN
#[derive(Debug, Clone)]
pub struct T5DecoderLayer<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand> {
    self_attn: T5Attention<F>,
    cross_attn: T5Attention<F>,
    ffn: T5FeedForward<F>,
}

impl<
        F: Float
            + Debug
            + Send
            + Sync
            + NumAssign
            + ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + 'static,
    > T5DecoderLayer<F>
{
    pub fn new<R: Rng + Clone + Send + Sync + 'static>(config: &T5Config, is_first_layer: bool, rng: &mut R) -> Result<Self> {
        let self_attn = T5Attention::<F>::new(
            config.d_model,
            config.d_kv,
            config.num_heads,
            config.dropout_rate,
            config.layer_norm_eps,
            true,
            false, // unidirectional for decoder
            if is_first_layer { config.relative_attention_num_buckets } else { 0 },
            config.relative_attention_max_distance,
            rng,
        )?;
        let cross_attn = T5Attention::<F>::new(
            config.d_model,
            config.d_kv,
            config.num_heads,
            config.dropout_rate,
            config.layer_norm_eps,
            false, // cross-attention: no relative bias
            false,
            0,
            0,
            rng,
        )?;
        let ffn = T5FeedForward::<F>::new(
            config.d_model,
            config.d_ff,
            config.dropout_rate,
            config.layer_norm_eps,
            rng,
        )?;
        Ok(Self {
            self_attn,
            cross_attn,
            ffn,
        })
    }

    /// Forward decoder layer
    ///
    /// `hidden`: [batch, tgt_len, d_model]
    /// `encoder_out`: [batch, src_len, d_model]
    /// `self_mask`: causal mask [batch, 1, tgt_len, tgt_len]
    /// `cross_mask`: encoder padding mask [batch, 1, 1, src_len]
    pub fn forward(
        &self,
        hidden: &Array<F, IxDyn>,
        encoder_out: &Array<F, IxDyn>,
        self_mask: Option<&Array<F, IxDyn>>,
        cross_mask: Option<&Array<F, IxDyn>>,
    ) -> Result<Array<F, IxDyn>> {
        // Causal self-attention
        let h = self.self_attn.forward(hidden, hidden, self_mask)?;
        // Cross-attention to encoder output
        let h = self.cross_attn.forward(&h, encoder_out, cross_mask)?;
        // FFN
        self.ffn.forward(&h)
    }

    pub fn update(&mut self, lr: F) -> Result<()> {
        self.self_attn.update(lr)?;
        self.cross_attn.update(lr)?;
        self.ffn.update(lr)?;
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        self.self_attn.parameter_count()
            + self.cross_attn.parameter_count()
            + self.ffn.parameter_count()
    }
}

// ---------------------------------------------------------------------------
// T5 Encoder
// ---------------------------------------------------------------------------

/// T5 Encoder stack
#[derive(Debug, Clone)]
pub struct T5Encoder<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand> {
    embed: Array<F, IxDyn>,
    layers: Vec<T5EncoderLayer<F>>,
    final_norm: T5LayerNorm<F>,
    dropout: Dropout<F>,
    d_model: usize,
    vocab_size: usize,
}

impl<
        F: Float
            + Debug
            + Send
            + Sync
            + NumAssign
            + ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + 'static,
    > T5Encoder<F>
{
    pub fn new<R: Rng + Clone + Send + Sync + 'static>(config: &T5Config, rng: &mut R) -> Result<Self> {
        let scale = F::from(1.0 / (config.d_model as f64).sqrt()).unwrap_or(F::one());
        let embed_data: Vec<F> = (0..config.vocab_size * config.d_model)
            .map(|_| {
                let v = rng.random::<f64>() * 2.0 - 1.0;
                F::from(v).unwrap_or(F::zero()) * scale
            })
            .collect();
        let embed = Array::from_shape_vec(
            IxDyn(&[config.vocab_size, config.d_model]),
            embed_data,
        )
        .map_err(|e| NeuralError::InvalidArchitecture(format!("embed failed: {e}")))?;

        let mut layers = Vec::with_capacity(config.num_encoder_layers);
        for i in 0..config.num_encoder_layers {
            layers.push(T5EncoderLayer::<F>::new(config, i == 0, rng)?);
        }

        let final_norm = T5LayerNorm::<F>::new(config.d_model, config.layer_norm_eps);
        let dropout = Dropout::<F>::new(config.dropout_rate, rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("dropout failed: {e}")))?;

        Ok(Self {
            embed,
            layers,
            final_norm,
            dropout,
            d_model: config.d_model,
            vocab_size: config.vocab_size,
        })
    }

    /// Forward encoder
    ///
    /// `input_ids`: [batch, src_len] (integer token indices as f32/f64)
    /// Returns: [batch, src_len, d_model]
    pub fn forward(
        &self,
        input_ids: &Array<F, IxDyn>,
        mask: Option<&Array<F, IxDyn>>,
    ) -> Result<Array<F, IxDyn>> {
        let shape = input_ids.shape();
        if shape.len() != 2 {
            return Err(NeuralError::InvalidArgument(
                "T5Encoder expects 2D input_ids [B, L]".to_string(),
            ));
        }
        let batch = shape[0];
        let seq_len = shape[1];

        // Token embedding lookup
        let mut hidden = Array::zeros(IxDyn(&[batch, seq_len, self.d_model]));
        for b in 0..batch {
            for t in 0..seq_len {
                let token_id = input_ids[[b, t]]
                    .to_usize()
                    .unwrap_or(0)
                    .min(self.vocab_size - 1);
                for d in 0..self.d_model {
                    hidden[[b, t, d]] = self.embed[[token_id, d]];
                }
            }
        }

        let mut hidden = self.dropout.forward(&hidden)?;

        for layer in &self.layers {
            hidden = layer.forward(&hidden, mask)?;
        }

        self.final_norm.forward(&hidden)
    }

    pub fn update(&mut self, lr: F) -> Result<()> {
        for layer in &mut self.layers {
            layer.update(lr)?;
        }
        self.final_norm.update(lr)?;
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        let embed_params = self.vocab_size * self.d_model;
        let layer_params: usize = self.layers.iter().map(|l| l.parameter_count()).sum();
        let norm_params = self.final_norm.parameter_count();
        embed_params + layer_params + norm_params
    }
}

// ---------------------------------------------------------------------------
// T5 Decoder
// ---------------------------------------------------------------------------

/// T5 Decoder stack
#[derive(Debug, Clone)]
pub struct T5Decoder<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand> {
    embed: Array<F, IxDyn>,
    layers: Vec<T5DecoderLayer<F>>,
    final_norm: T5LayerNorm<F>,
    dropout: Dropout<F>,
    d_model: usize,
    vocab_size: usize,
}

impl<
        F: Float
            + Debug
            + Send
            + Sync
            + NumAssign
            + ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + 'static,
    > T5Decoder<F>
{
    pub fn new<R: Rng + Clone + Send + Sync + 'static>(config: &T5Config, rng: &mut R) -> Result<Self> {
        let scale = F::from(1.0 / (config.d_model as f64).sqrt()).unwrap_or(F::one());
        let embed_data: Vec<F> = (0..config.vocab_size * config.d_model)
            .map(|_| {
                let v = rng.random::<f64>() * 2.0 - 1.0;
                F::from(v).unwrap_or(F::zero()) * scale
            })
            .collect();
        let embed = Array::from_shape_vec(
            IxDyn(&[config.vocab_size, config.d_model]),
            embed_data,
        )
        .map_err(|e| NeuralError::InvalidArchitecture(format!("embed failed: {e}")))?;

        let mut layers = Vec::with_capacity(config.num_decoder_layers);
        for i in 0..config.num_decoder_layers {
            layers.push(T5DecoderLayer::<F>::new(config, i == 0, rng)?);
        }

        let final_norm = T5LayerNorm::<F>::new(config.d_model, config.layer_norm_eps);
        let dropout = Dropout::<F>::new(config.dropout_rate, rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("dropout failed: {e}")))?;

        Ok(Self {
            embed,
            layers,
            final_norm,
            dropout,
            d_model: config.d_model,
            vocab_size: config.vocab_size,
        })
    }

    /// Forward decoder
    ///
    /// `decoder_input_ids`: [batch, tgt_len]
    /// `encoder_output`: [batch, src_len, d_model]
    /// Returns: [batch, tgt_len, d_model]
    pub fn forward(
        &self,
        decoder_input_ids: &Array<F, IxDyn>,
        encoder_output: &Array<F, IxDyn>,
        self_mask: Option<&Array<F, IxDyn>>,
        cross_mask: Option<&Array<F, IxDyn>>,
    ) -> Result<Array<F, IxDyn>> {
        let shape = decoder_input_ids.shape();
        if shape.len() != 2 {
            return Err(NeuralError::InvalidArgument(
                "T5Decoder expects 2D decoder_input_ids [B, L]".to_string(),
            ));
        }
        let batch = shape[0];
        let seq_len = shape[1];

        let mut hidden = Array::zeros(IxDyn(&[batch, seq_len, self.d_model]));
        for b in 0..batch {
            for t in 0..seq_len {
                let token_id = decoder_input_ids[[b, t]]
                    .to_usize()
                    .unwrap_or(0)
                    .min(self.vocab_size - 1);
                for d in 0..self.d_model {
                    hidden[[b, t, d]] = self.embed[[token_id, d]];
                }
            }
        }

        let mut hidden = self.dropout.forward(&hidden)?;

        for layer in &self.layers {
            hidden = layer.forward(&hidden, encoder_output, self_mask, cross_mask)?;
        }

        self.final_norm.forward(&hidden)
    }

    pub fn update(&mut self, lr: F) -> Result<()> {
        for layer in &mut self.layers {
            layer.update(lr)?;
        }
        self.final_norm.update(lr)?;
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        let embed_params = self.vocab_size * self.d_model;
        let layer_params: usize = self.layers.iter().map(|l| l.parameter_count()).sum();
        let norm_params = self.final_norm.parameter_count();
        embed_params + layer_params + norm_params
    }
}

// ---------------------------------------------------------------------------
// T5 Model (Encoder-Decoder)
// ---------------------------------------------------------------------------

/// T5 model output
#[derive(Debug, Clone)]
pub struct T5Output<F: Float> {
    /// Logits over vocabulary: [batch, tgt_len, vocab_size]
    pub logits: Array<F, IxDyn>,
    /// Encoder output hidden states: [batch, src_len, d_model]
    pub encoder_output: Array<F, IxDyn>,
    /// Decoder output hidden states: [batch, tgt_len, d_model]
    pub decoder_output: Array<F, IxDyn>,
}

/// Complete T5 Encoder-Decoder model
///
/// Supports text-to-text tasks where both input and output are sequences
/// of token ids.
#[derive(Debug, Clone)]
pub struct T5Model<F: Float + Debug + Send + Sync + NumAssign + ScalarOperand + SimdUnifiedOps> {
    config: T5Config,
    encoder: T5Encoder<F>,
    decoder: T5Decoder<F>,
    /// Language model head: projects d_model → vocab_size
    lm_head: Dense<F>,
    /// Final layer norm for the LM head input
    lm_norm: LayerNorm<F>,
}

impl<
        F: Float
            + Debug
            + Send
            + Sync
            + NumAssign
            + ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + SimdUnifiedOps
            + 'static,
    > T5Model<F>
{
    /// Create a new T5 model
    pub fn new(config: T5Config) -> Result<Self> {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42u8; 32]);
        let encoder = T5Encoder::<F>::new(&config, &mut rng)?;
        let decoder = T5Decoder::<F>::new(&config, &mut rng)?;
        let lm_head = Dense::<F>::new(config.d_model, config.vocab_size, None, &mut rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("lm_head failed: {e}")))?;
        let lm_norm = LayerNorm::<F>::new(config.d_model, config.layer_norm_eps, &mut rng)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("lm_norm failed: {e}")))?;
        Ok(Self {
            config,
            encoder,
            decoder,
            lm_head,
            lm_norm,
        })
    }

    /// Forward pass of the full T5 model
    ///
    /// `input_ids`: [batch, src_len] token ids (as floats)
    /// `decoder_input_ids`: [batch, tgt_len] token ids (as floats)
    ///
    /// Returns `T5Output` with logits [batch, tgt_len, vocab_size]
    pub fn forward(
        &self,
        input_ids: &Array<F, IxDyn>,
        decoder_input_ids: &Array<F, IxDyn>,
    ) -> Result<T5Output<F>> {
        // Generate causal mask for decoder self-attention
        let tgt_len = decoder_input_ids.shape()[1];
        let causal_mask = make_causal_mask::<F>(tgt_len);

        let encoder_output = self.encoder.forward(input_ids, None)?;
        let decoder_output = self.decoder.forward(
            decoder_input_ids,
            &encoder_output,
            Some(&causal_mask),
            None,
        )?;

        // LM head
        let normed = self.lm_norm.forward(&decoder_output)?;
        let logits = self.lm_head.forward(&normed)?;

        Ok(T5Output {
            logits,
            encoder_output,
            decoder_output,
        })
    }

    /// Encode input sequence only (useful for embedding extraction)
    pub fn encode(&self, input_ids: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.encoder.forward(input_ids, None)
    }

    /// Greedy decode given encoder output
    ///
    /// Starts from `decoder_start_token_id` and generates up to `max_length` tokens.
    pub fn greedy_decode(
        &self,
        input_ids: &Array<F, IxDyn>,
        max_length: usize,
    ) -> Result<Vec<usize>> {
        let batch = input_ids.shape()[0];
        if batch != 1 {
            return Err(NeuralError::InvalidArgument(
                "greedy_decode only supports batch_size=1".to_string(),
            ));
        }

        let encoder_out = self.encoder.forward(input_ids, None)?;

        let mut generated = vec![self.config.decoder_start_token_id];
        let eos = self.config.eos_token_id;

        for _ in 0..max_length {
            let tgt_len = generated.len();
            let mut dec_ids: Vec<F> = generated
                .iter()
                .map(|&id| F::from(id as f64).unwrap_or(F::zero()))
                .collect();
            // Pad to same shape
            dec_ids.push(F::zero()); // dummy
            let dec_ids = dec_ids[..tgt_len].to_vec();
            let dec_arr = Array::from_shape_vec(IxDyn(&[1, tgt_len]), dec_ids)
                .map_err(|e| NeuralError::InvalidArgument(format!("array error: {e}")))?;

            let causal_mask = make_causal_mask::<F>(tgt_len);
            let dec_out = self.decoder.forward(
                &dec_arr,
                &encoder_out,
                Some(&causal_mask),
                None,
            )?;

            // Get logits for last position
            let normed = self.lm_norm.forward(&dec_out)?;
            let logits = self.lm_head.forward(&normed)?;

            // Argmax at last time step
            let vocab_size = self.config.vocab_size;
            let mut best_id = 0usize;
            let mut best_val = F::neg_infinity();
            for v in 0..vocab_size {
                let val = logits[[0, tgt_len - 1, v]];
                if val > best_val {
                    best_val = val;
                    best_id = v;
                }
            }

            generated.push(best_id);
            if best_id == eos {
                break;
            }
        }

        Ok(generated)
    }

    /// Return configuration
    pub fn config(&self) -> &T5Config {
        &self.config
    }

    /// Count trainable parameters
    pub fn parameter_count(&self) -> usize {
        self.encoder.parameter_count()
            + self.decoder.parameter_count()
            + self.lm_head.parameter_count()
            + self.lm_norm.parameter_count()
    }
}

impl<
        F: Float
            + Debug
            + Send
            + Sync
            + NumAssign
            + ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + SimdUnifiedOps
            + 'static,
    > Layer<F> for T5Model<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // When used as a Layer, treat first half as input_ids and second half as decoder_ids
        // For simple usage, use forward() directly.
        let shape = input.shape();
        if shape.len() != 3 || shape[2] != 2 {
            return Err(NeuralError::InvalidArgument(
                "T5Model Layer::forward expects [B, L, 2] where [:,:,0]=encoder_ids, [:,:,1]=decoder_ids".to_string()
            ));
        }
        let batch = shape[0];
        let seq = shape[1];
        let mut enc_ids = Array::zeros(IxDyn(&[batch, seq]));
        let mut dec_ids = Array::zeros(IxDyn(&[batch, seq]));
        for b in 0..batch {
            for t in 0..seq {
                enc_ids[[b, t]] = input[[b, t, 0]];
                dec_ids[[b, t]] = input[[b, t, 1]];
            }
        }
        let out = self.forward(&enc_ids, &dec_ids)?;
        Ok(out.logits)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, lr: F) -> Result<()> {
        self.encoder.update(lr)?;
        self.decoder.update(lr)?;
        self.lm_head.update(lr)?;
        self.lm_norm.update(lr)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "T5Model"
    }

    fn parameter_count(&self) -> usize {
        self.parameter_count()
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Build a causal attention mask for decoder self-attention
///
/// Returns [1, 1, tgt_len, tgt_len] where upper triangle is -1e9
fn make_causal_mask<F: Float + Debug + NumAssign + ScalarOperand + FromPrimitive>(
    tgt_len: usize,
) -> Array<F, IxDyn> {
    let large_neg = F::from(-1e9_f64).unwrap_or(F::from(-1e6_f64).unwrap_or(F::zero()));
    let mut mask = Array::zeros(IxDyn(&[1, 1, tgt_len, tgt_len]));
    for i in 0..tgt_len {
        for j in (i + 1)..tgt_len {
            mask[[0, 0, i, j]] = large_neg;
        }
    }
    mask
}

/// Reshape [B, L, H*D] → [B, H, L, D]
fn reshape_to_heads<F: Float + Debug + NumAssign + ScalarOperand>(
    x: &Array<F, IxDyn>,
    batch: usize,
    seq: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Array<F, IxDyn>> {
    let mut out = Array::zeros(IxDyn(&[batch, num_heads, seq, head_dim]));
    for b in 0..batch {
        for t in 0..seq {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    out[[b, h, t, d]] = x[[b, t, h * head_dim + d]];
                }
            }
        }
    }
    Ok(out)
}

/// Softmax in-place along the last dim of a 4D array
fn softmax_4d_last<F: Float + Debug + NumAssign + ScalarOperand>(
    x: &mut Array<F, IxDyn>,
) -> Result<()> {
    let shape = x.shape().to_vec();
    if shape.len() != 4 {
        return Err(NeuralError::InvalidArgument(
            "softmax_4d_last requires 4D array".to_string(),
        ));
    }
    let (b, h, q, k) = (shape[0], shape[1], shape[2], shape[3]);
    let eps = F::from(1e-9).unwrap_or(F::zero());

    for bi in 0..b {
        for hi in 0..h {
            for qi in 0..q {
                // Find max
                let mut max_v = F::neg_infinity();
                for ki in 0..k {
                    if x[[bi, hi, qi, ki]] > max_v {
                        max_v = x[[bi, hi, qi, ki]];
                    }
                }
                // Exp and sum
                let mut sum = F::zero();
                for ki in 0..k {
                    let v = (x[[bi, hi, qi, ki]] - max_v).exp();
                    x[[bi, hi, qi, ki]] = v;
                    sum = sum + v;
                }
                // Normalize
                let inv = F::one() / (sum + eps);
                for ki in 0..k {
                    x[[bi, hi, qi, ki]] = x[[bi, hi, qi, ki]] * inv;
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> T5Config {
        T5Config {
            vocab_size: 64,
            d_model: 32,
            d_ff: 64,
            d_kv: 8,
            num_heads: 4,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            dropout_rate: 0.0,
            relative_attention_num_buckets: 16,
            relative_attention_max_distance: 32,
            layer_norm_eps: 1e-6,
            decoder_start_token_id: 0,
            eos_token_id: 1,
            pad_token_id: 0,
        }
    }

    #[test]
    fn test_t5_config_defaults() {
        let cfg = T5Config::default();
        assert_eq!(cfg.d_model, 512);
        assert_eq!(cfg.num_encoder_layers, 6);
        assert_eq!(cfg.num_decoder_layers, 6);
    }

    #[test]
    fn test_t5_config_base() {
        let cfg = T5Config::t5_base();
        assert_eq!(cfg.d_model, 768);
        assert_eq!(cfg.num_heads, 12);
    }

    #[test]
    fn test_t5_layer_norm() {
        let norm = T5LayerNorm::<f32>::new(16, 1e-6);
        let x = Array::from_elem(IxDyn(&[2, 4, 16]), 1.0_f32);
        let out = norm.forward(&x).expect("RMS norm failed");
        assert_eq!(out.shape(), x.shape());
    }

    #[test]
    fn test_relative_attention_bias() {
        let rb = T5RelativeAttentionBias::<f32>::new(16, 32, 4, true);
        let bias = rb.compute_bias(5, 7);
        assert_eq!(bias.shape(), &[1, 4, 5, 7]);
    }

    #[test]
    fn test_t5_attention_self() {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([0u8; 32]);
        let attn = T5Attention::<f32>::new(32, 8, 4, 0.0, 1e-6, true, true, 16, 32, &mut rng)
            .expect("Failed to create attention");
        let x = Array::zeros(IxDyn(&[1, 5, 32]));
        let out = attn.forward(&x, &x, None).expect("Forward failed");
        assert_eq!(out.shape(), &[1, 5, 32]);
    }

    #[test]
    fn test_t5_attention_cross() {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([0u8; 32]);
        let attn = T5Attention::<f32>::new(32, 8, 4, 0.0, 1e-6, false, false, 0, 0, &mut rng)
            .expect("Failed to create cross-attention");
        let query = Array::zeros(IxDyn(&[1, 3, 32]));
        let kv = Array::zeros(IxDyn(&[1, 5, 32]));
        let out = attn.forward(&query, &kv, None).expect("Cross-attn forward failed");
        assert_eq!(out.shape(), &[1, 3, 32]);
    }

    #[test]
    fn test_t5_ffn() {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([0u8; 32]);
        let ffn = T5FeedForward::<f32>::new(32, 64, 0.0, 1e-6, &mut rng)
            .expect("FFN creation failed");
        let x = Array::zeros(IxDyn(&[2, 5, 32]));
        let out = ffn.forward(&x).expect("FFN forward failed");
        assert_eq!(out.shape(), &[2, 5, 32]);
    }

    #[test]
    fn test_t5_encoder_layer() {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([0u8; 32]);
        let cfg = tiny_config();
        let layer = T5EncoderLayer::<f32>::new(&cfg, true, &mut rng)
            .expect("Encoder layer failed");
        let x = Array::zeros(IxDyn(&[1, 6, 32]));
        let out = layer.forward(&x, None).expect("Forward failed");
        assert_eq!(out.shape(), &[1, 6, 32]);
    }

    #[test]
    fn test_t5_decoder_layer() {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([0u8; 32]);
        let cfg = tiny_config();
        let layer = T5DecoderLayer::<f32>::new(&cfg, true, &mut rng)
            .expect("Decoder layer failed");
        let dec_hidden = Array::zeros(IxDyn(&[1, 4, 32]));
        let enc_out = Array::zeros(IxDyn(&[1, 6, 32]));
        let out = layer.forward(&dec_hidden, &enc_out, None, None).expect("Forward failed");
        assert_eq!(out.shape(), &[1, 4, 32]);
    }

    #[test]
    fn test_t5_encoder_forward() {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([0u8; 32]);
        let cfg = tiny_config();
        let encoder = T5Encoder::<f32>::new(&cfg, &mut rng).expect("Encoder creation failed");
        let input_ids = Array::from_elem(IxDyn(&[1, 5]), 3.0_f32);
        let out = encoder.forward(&input_ids, None).expect("Encoder forward failed");
        assert_eq!(out.shape(), &[1, 5, 32]);
    }

    #[test]
    fn test_t5_decoder_forward() {
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([0u8; 32]);
        let cfg = tiny_config();
        let decoder = T5Decoder::<f32>::new(&cfg, &mut rng).expect("Decoder creation failed");
        let dec_ids = Array::from_elem(IxDyn(&[1, 3]), 2.0_f32);
        let enc_out = Array::zeros(IxDyn(&[1, 5, 32]));
        let causal_mask = make_causal_mask::<f32>(3);
        let out = decoder
            .forward(&dec_ids, &enc_out, Some(&causal_mask), None)
            .expect("Decoder forward failed");
        assert_eq!(out.shape(), &[1, 3, 32]);
    }

    #[test]
    fn test_t5_model_forward() {
        let cfg = tiny_config();
        let model = T5Model::<f32>::new(cfg.clone()).expect("Model creation failed");
        let input_ids = Array::from_elem(IxDyn(&[1, 5]), 3.0_f32);
        let dec_ids = Array::from_elem(IxDyn(&[1, 3]), 2.0_f32);
        let out = model.forward(&input_ids, &dec_ids).expect("Model forward failed");
        assert_eq!(out.logits.shape(), &[1, 3, cfg.vocab_size]);
        assert_eq!(out.encoder_output.shape(), &[1, 5, cfg.d_model]);
        assert_eq!(out.decoder_output.shape(), &[1, 3, cfg.d_model]);
    }

    #[test]
    fn test_t5_model_encode() {
        let cfg = tiny_config();
        let model = T5Model::<f32>::new(cfg.clone()).expect("Model creation failed");
        let input_ids = Array::from_elem(IxDyn(&[2, 4]), 1.0_f32);
        let enc = model.encode(&input_ids).expect("Encode failed");
        assert_eq!(enc.shape(), &[2, 4, cfg.d_model]);
    }

    #[test]
    fn test_t5_greedy_decode() {
        let cfg = tiny_config();
        let model = T5Model::<f32>::new(cfg).expect("Model creation failed");
        let input_ids = Array::from_elem(IxDyn(&[1, 5]), 3.0_f32);
        let generated = model.greedy_decode(&input_ids, 5).expect("Greedy decode failed");
        // Should start with decoder_start_token_id and generate up to 5 more tokens
        assert!(!generated.is_empty());
        assert_eq!(generated[0], 0); // decoder_start_token_id
    }

    #[test]
    fn test_t5_model_parameter_count() {
        let cfg = tiny_config();
        let model = T5Model::<f32>::new(cfg).expect("Model creation failed");
        let count = model.parameter_count();
        assert!(count > 0, "Model should have parameters");
    }

    #[test]
    fn test_t5_model_layer_type() {
        use crate::layers::Layer;
        let cfg = tiny_config();
        let model = T5Model::<f32>::new(cfg).expect("Model creation failed");
        assert_eq!(model.layer_type(), "T5Model");
    }

    #[test]
    fn test_causal_mask_shape() {
        let mask = make_causal_mask::<f32>(4);
        assert_eq!(mask.shape(), &[1, 1, 4, 4]);
        // Upper triangle should be large negative
        assert!(mask[[0, 0, 0, 1]] < -1e6);
        // Diagonal and lower should be zero
        assert_eq!(mask[[0, 0, 0, 0]], 0.0);
        assert_eq!(mask[[0, 0, 2, 1]], 0.0);
    }
}
