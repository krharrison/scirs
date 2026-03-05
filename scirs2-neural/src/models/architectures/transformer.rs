//! Transformer architecture for sequence-to-sequence tasks
//!
//! This module provides a high-level Transformer architecture that builds on
//! the core transformer primitives. It follows the original "Attention Is All
//! You Need" design with configurable encoder/decoder stacks, multi-head attention,
//! feed-forward networks, and sinusoidal positional encoding.
//!
//! # References
//! - Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
//!   <https://arxiv.org/abs/1706.03762>

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use crate::transformer::{TransformerDecoder, TransformerEncoder};
use scirs2_core::ndarray::{Array, Array2, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::Rng;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the high-level [`SequenceTransformer`] model.
///
/// Field names mirror the convention used by PyTorch's `nn.Transformer` so
/// that the two implementations are easy to compare side-by-side.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Embedding / model dimension (d_model)
    pub d_model: usize,
    /// Number of attention heads (nhead)
    pub nhead: usize,
    /// Number of encoder layers
    pub num_encoder_layers: usize,
    /// Number of decoder layers
    pub num_decoder_layers: usize,
    /// Dimension of each feed-forward sub-layer (dim_feedforward)
    pub dim_feedforward: usize,
    /// Dropout probability applied inside each sub-layer
    pub dropout: f64,
    /// Maximum sequence length used to pre-compute positional encodings
    pub max_seq_len: usize,
    /// Small epsilon used in layer normalisation
    pub epsilon: f64,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            nhead: 8,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
            dim_feedforward: 2048,
            dropout: 0.1,
            max_seq_len: 512,
            epsilon: 1e-5,
        }
    }
}

// ---------------------------------------------------------------------------
// Positional encoding (sinusoidal helper)
// ---------------------------------------------------------------------------

/// Generate a sinusoidal positional encoding matrix.
///
/// The returned tensor has shape `[seq_len, d_model]` and is computed once
/// then used to offset token embeddings.
///
/// ```text
/// PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
/// ```
pub fn sinusoidal_encoding<F: Float>(seq_len: usize, d_model: usize) -> Result<Array2<F>> {
    let mut pe = Array2::<F>::zeros((seq_len, d_model));
    let ten_thousand = F::from(10_000.0_f64).ok_or_else(|| {
        NeuralError::InvalidArchitecture("Cannot convert 10000 to Float".to_string())
    })?;
    let d_model_f = F::from(d_model as f64).ok_or_else(|| {
        NeuralError::InvalidArchitecture("Cannot convert d_model to Float".to_string())
    })?;
    let two = F::from(2.0_f64).ok_or_else(|| {
        NeuralError::InvalidArchitecture("Cannot convert 2 to Float".to_string())
    })?;

    for pos in 0..seq_len {
        let pos_f = F::from(pos as f64).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Cannot convert pos to Float".to_string())
        })?;
        for i in 0..(d_model / 2) {
            let i_f = F::from(i as f64).ok_or_else(|| {
                NeuralError::InvalidArchitecture("Cannot convert i to Float".to_string())
            })?;
            let exp = (two * i_f) / d_model_f;
            let denom = ten_thousand.powf(exp);
            let angle = pos_f / denom;
            pe[[pos, 2 * i]] = angle.sin();
            if 2 * i + 1 < d_model {
                pe[[pos, 2 * i + 1]] = angle.cos();
            }
        }
    }
    Ok(pe)
}

// ---------------------------------------------------------------------------
// TransformerEncoderStack
// ---------------------------------------------------------------------------

/// A stack of N identical encoder layers, each containing:
///  1. Multi-head self-attention
///  2. Position-wise feed-forward network
///  3. Residual connections + layer normalisation
///
/// Input/output shape: `[batch, seq_len, d_model]`.
pub struct TransformerEncoderStack<F: Float + Debug + Send + Sync + SimdUnifiedOps + NumAssign> {
    inner: TransformerEncoder<F>,
    d_model: usize,
    num_layers: usize,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign>
    TransformerEncoderStack<F>
{
    /// Create a new encoder stack.
    pub fn new<R: Rng>(config: &TransformerConfig, rng: &mut R) -> Result<Self> {
        let inner = TransformerEncoder::new(
            config.d_model,
            config.num_encoder_layers,
            config.nhead,
            config.dim_feedforward,
            config.dropout,
            config.epsilon,
            rng,
        )?;
        Ok(Self {
            inner,
            d_model: config.d_model,
            num_layers: config.num_encoder_layers,
        })
    }

    /// Forward pass: `[batch, src_len, d_model] → [batch, src_len, d_model]`.
    pub fn forward(&self, src: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = src.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "TransformerEncoderStack: expected 3-D input [batch, seq, d_model], got {:?}",
                shape
            )));
        }
        if shape[2] != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "TransformerEncoderStack: d_model mismatch – expected {}, got {}",
                self.d_model, shape[2]
            )));
        }
        self.inner.forward(src)
    }

    /// Number of encoder layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign> Clone
    for TransformerEncoderStack<F>
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            d_model: self.d_model,
            num_layers: self.num_layers,
        }
    }
}

// ---------------------------------------------------------------------------
// TransformerDecoderStack
// ---------------------------------------------------------------------------

/// A stack of N identical decoder layers, each containing:
///  1. Masked multi-head self-attention
///  2. Multi-head cross-attention over the encoder output
///  3. Position-wise feed-forward network
///  4. Residual connections + layer normalisation
///
/// Input/output shape: `[batch, tgt_len, d_model]`.
pub struct TransformerDecoderStack<F: Float + Debug + Send + Sync + SimdUnifiedOps + NumAssign> {
    inner: TransformerDecoder<F>,
    d_model: usize,
    num_layers: usize,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign>
    TransformerDecoderStack<F>
{
    /// Create a new decoder stack.
    pub fn new<R: Rng>(config: &TransformerConfig, rng: &mut R) -> Result<Self> {
        let inner = TransformerDecoder::new(
            config.d_model,
            config.num_decoder_layers,
            config.nhead,
            config.dim_feedforward,
            config.dropout,
            config.epsilon,
            rng,
        )?;
        Ok(Self {
            inner,
            d_model: config.d_model,
            num_layers: config.num_decoder_layers,
        })
    }

    /// Forward pass using memory (encoder output) and target input.
    /// Returns `[batch, tgt_len, d_model]`.
    pub fn forward(
        &self,
        tgt: &Array<F, IxDyn>,
        memory: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let tshape = tgt.shape();
        let mshape = memory.shape();
        if tshape.len() != 3 || mshape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "TransformerDecoderStack: expected 3-D inputs, tgt {:?} memory {:?}",
                tshape, mshape
            )));
        }
        if tshape[2] != self.d_model || mshape[2] != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "TransformerDecoderStack: d_model mismatch – expected {}, tgt[2]={}, memory[2]={}",
                self.d_model, tshape[2], mshape[2]
            )));
        }
        self.inner.forward_with_encoder(tgt, memory)
    }

    /// Number of decoder layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign> Clone
    for TransformerDecoderStack<F>
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            d_model: self.d_model,
            num_layers: self.num_layers,
        }
    }
}

// ---------------------------------------------------------------------------
// SequenceTransformer
// ---------------------------------------------------------------------------

/// Complete encoder-decoder transformer architecture.
///
/// Combines:
/// - Sinusoidal positional encoding for both source and target
/// - A stack of `num_encoder_layers` encoder blocks
/// - A stack of `num_decoder_layers` decoder blocks
///
/// # Examples
/// ```no_run
/// use scirs2_neural::models::architectures::transformer::{SequenceTransformer, TransformerConfig};
/// use scirs2_core::ndarray::Array3;
/// use scirs2_core::random::rngs::SmallRng;
/// use scirs2_core::random::SeedableRng;
///
/// let cfg = TransformerConfig {
///     d_model: 64,
///     nhead: 4,
///     num_encoder_layers: 2,
///     num_decoder_layers: 2,
///     dim_feedforward: 128,
///     dropout: 0.1,
///     max_seq_len: 50,
///     epsilon: 1e-5,
/// };
/// let mut rng = SmallRng::seed_from_u64(0);
/// let model = SequenceTransformer::<f64>::new(cfg, &mut rng).expect("build failed");
///
/// let src = Array3::<f64>::zeros((2, 8, 64)).into_dyn();
/// let tgt = Array3::<f64>::zeros((2, 6, 64)).into_dyn();
/// let out = model.forward(&src, &tgt, None, None).expect("forward failed");
/// assert_eq!(out.shape(), tgt.shape());
/// ```
pub struct SequenceTransformer<F: Float + Debug + Send + Sync + SimdUnifiedOps + NumAssign> {
    encoder: TransformerEncoderStack<F>,
    decoder: TransformerDecoderStack<F>,
    config: TransformerConfig,
    /// Cache of the last encoder output (for autoregressive decoding)
    encoder_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign>
    SequenceTransformer<F>
{
    /// Build a new [`SequenceTransformer`] from a [`TransformerConfig`].
    pub fn new<R: Rng>(config: TransformerConfig, rng: &mut R) -> Result<Self> {
        let encoder = TransformerEncoderStack::new(&config, rng)?;
        let decoder = TransformerDecoderStack::new(&config, rng)?;
        Ok(Self {
            encoder,
            decoder,
            config,
            encoder_cache: Arc::new(RwLock::new(None)),
        })
    }

    /// Full encoder-decoder forward pass.
    ///
    /// # Arguments
    /// * `src`      – Source token embeddings `[batch, src_len, d_model]`
    /// * `tgt`      – Target token embeddings `[batch, tgt_len, d_model]`
    /// * `src_mask` – Optional source attention mask (reserved for future use)
    /// * `tgt_mask` – Optional target causal mask  (reserved for future use)
    ///
    /// # Returns
    /// Decoder output `[batch, tgt_len, d_model]`.
    pub fn forward(
        &self,
        src: &Array<F, IxDyn>,
        tgt: &Array<F, IxDyn>,
        _src_mask: Option<&Array<F, IxDyn>>,
        _tgt_mask: Option<&Array<F, IxDyn>>,
    ) -> Result<Array<F, IxDyn>> {
        self.forward_full(src, tgt)
    }

    /// Full encoder-decoder pass returning the 3-D decoder output tensor.
    pub fn forward_full(
        &self,
        src: &Array<F, IxDyn>,
        tgt: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Validate input shapes
        let src_shape = src.shape();
        let tgt_shape = tgt.shape();
        if src_shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "SequenceTransformer: src must be 3-D [batch, src_len, d_model], got {:?}",
                src_shape
            )));
        }
        if tgt_shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "SequenceTransformer: tgt must be 3-D [batch, tgt_len, d_model], got {:?}",
                tgt_shape
            )));
        }
        if src_shape[0] != tgt_shape[0] {
            return Err(NeuralError::InferenceError(format!(
                "SequenceTransformer: batch size mismatch – src {} vs tgt {}",
                src_shape[0], tgt_shape[0]
            )));
        }
        if src_shape[2] != self.config.d_model {
            return Err(NeuralError::InferenceError(format!(
                "SequenceTransformer: src d_model mismatch – expected {}, got {}",
                self.config.d_model, src_shape[2]
            )));
        }
        if tgt_shape[2] != self.config.d_model {
            return Err(NeuralError::InferenceError(format!(
                "SequenceTransformer: tgt d_model mismatch – expected {}, got {}",
                self.config.d_model, tgt_shape[2]
            )));
        }

        // Add positional encoding to source
        let src_pe = self.add_positional_encoding(src)?;
        // Encode
        let memory = self.encoder.forward(&src_pe)?;
        // Cache for later autoregressive use
        *self
            .encoder_cache
            .write()
            .map_err(|e| NeuralError::InferenceError(format!("RwLock poison: {e}")))? =
            Some(memory.clone());

        // Add positional encoding to target
        let tgt_pe = self.add_positional_encoding(tgt)?;
        // Decode
        let output = self.decoder.forward(&tgt_pe, &memory)?;
        Ok(output)
    }

    /// Encoder-only forward pass (useful for BERT-style models).
    pub fn encode(&self, src: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = src.shape();
        if shape.len() != 3 || shape[2] != self.config.d_model {
            return Err(NeuralError::InferenceError(format!(
                "SequenceTransformer::encode: expected [batch, seq, {}], got {:?}",
                self.config.d_model, shape
            )));
        }
        let src_pe = self.add_positional_encoding(src)?;
        self.encoder.forward(&src_pe)
    }

    /// Decoder-only forward pass using a pre-computed memory tensor.
    pub fn decode(
        &self,
        tgt: &Array<F, IxDyn>,
        memory: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let tgt_pe = self.add_positional_encoding(tgt)?;
        self.decoder.forward(&tgt_pe, memory)
    }

    /// Apply sinusoidal positional encoding to a `[batch, seq_len, d_model]` tensor.
    fn add_positional_encoding(&self, x: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let xshape = x.shape();
        let batch_size = xshape[0];
        let seq_len = xshape[1];
        let d_model = xshape[2];
        // Build sinusoidal PE matrix [seq_len, d_model]
        let pe = sinusoidal_encoding::<F>(seq_len, d_model)?;
        // Broadcast + add: output[b, t, i] = x[b, t, i] + pe[t, i]
        let mut output = x.to_owned();
        for b in 0..batch_size {
            for t in 0..seq_len {
                for i in 0..d_model {
                    output[[b, t, i]] = output[[b, t, i]] + pe[[t, i]];
                }
            }
        }
        Ok(output)
    }

    /// Access the model configuration.
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Access the encoder stack.
    pub fn encoder(&self) -> &TransformerEncoderStack<F> {
        &self.encoder
    }

    /// Access the decoder stack.
    pub fn decoder(&self) -> &TransformerDecoderStack<F> {
        &self.decoder
    }

    /// Approximate parameter count.
    pub fn parameter_count(&self) -> usize {
        let d = self.config.d_model;
        let ff = self.config.dim_feedforward;
        // Each encoder layer: 4 projection matrices (d×d each) + FFN (2 × d × ff) + LayerNorm params (4d)
        let enc_per_layer = 4 * d * d + 2 * d * ff + 4 * d;
        // Each decoder layer: 8 projection matrices (2 attn) + FFN + LayerNorm params (6d)
        let dec_per_layer = 8 * d * d + 2 * d * ff + 6 * d;
        enc_per_layer * self.config.num_encoder_layers
            + dec_per_layer * self.config.num_decoder_layers
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + NumAssign> Clone
    for SequenceTransformer<F>
{
    fn clone(&self) -> Self {
        Self {
            encoder: self.encoder.clone(),
            decoder: self.decoder.clone(),
            config: self.config.clone(),
            encoder_cache: Arc::new(RwLock::new(None)),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;
    use scirs2_core::random::rngs::SmallRng;
    use scirs2_core::random::SeedableRng;

    fn small_config() -> TransformerConfig {
        TransformerConfig {
            d_model: 32,
            nhead: 4,
            num_encoder_layers: 1,
            num_decoder_layers: 1,
            dim_feedforward: 64,
            dropout: 0.0,
            max_seq_len: 32,
            epsilon: 1e-5,
        }
    }

    #[test]
    fn test_transformer_construction() {
        let mut rng = SmallRng::seed_from_u64(42);
        let model = SequenceTransformer::<f64>::new(small_config(), &mut rng);
        assert!(model.is_ok(), "Should build without error");
        let model = model.expect("build failed");
        assert_eq!(model.config().d_model, 32);
        assert_eq!(model.config().nhead, 4);
        assert_eq!(model.encoder().num_layers(), 1);
        assert_eq!(model.decoder().num_layers(), 1);
    }

    #[test]
    fn test_transformer_forward_shape() {
        let mut rng = SmallRng::seed_from_u64(1);
        let model =
            SequenceTransformer::<f64>::new(small_config(), &mut rng).expect("build failed");
        let src = Array3::<f64>::zeros((2, 6, 32)).into_dyn();
        let tgt = Array3::<f64>::zeros((2, 4, 32)).into_dyn();
        let out = model
            .forward(&src, &tgt, None, None)
            .expect("forward failed");
        assert_eq!(
            out.shape(),
            &[2, 4, 32],
            "Output shape should match [batch, tgt_len, d_model]"
        );
    }

    #[test]
    fn test_transformer_encode_shape() {
        let mut rng = SmallRng::seed_from_u64(2);
        let model =
            SequenceTransformer::<f64>::new(small_config(), &mut rng).expect("build failed");
        let src = Array3::<f64>::from_elem((3, 5, 32), 0.1).into_dyn();
        let enc = model.encode(&src).expect("encode failed");
        assert_eq!(enc.shape(), &[3, 5, 32]);
    }

    #[test]
    fn test_transformer_decode_shape() {
        let mut rng = SmallRng::seed_from_u64(3);
        let model =
            SequenceTransformer::<f64>::new(small_config(), &mut rng).expect("build failed");
        let src = Array3::<f64>::from_elem((2, 8, 32), 0.05).into_dyn();
        let memory = model.encode(&src).expect("encode failed");
        let tgt = Array3::<f64>::zeros((2, 3, 32)).into_dyn();
        let dec = model.decode(&tgt, &memory).expect("decode failed");
        assert_eq!(dec.shape(), &[2, 3, 32]);
    }

    #[test]
    fn test_sinusoidal_encoding_shape() {
        let pe = sinusoidal_encoding::<f64>(10, 32).expect("pe failed");
        assert_eq!(pe.shape(), &[10, 32]);
    }

    #[test]
    fn test_sinusoidal_encoding_values() {
        // PE(0, 0) = sin(0 / 10000^0) = sin(0) = 0.0
        let pe = sinusoidal_encoding::<f64>(4, 8).expect("pe failed");
        let val = pe[[0, 0]];
        assert!(val.abs() < 1e-10, "PE(0,0) should be 0.0, got {}", val);
        // PE(0, 1) = cos(0 / ...) = cos(0) = 1.0
        let val1 = pe[[0, 1]];
        assert!(
            (val1 - 1.0).abs() < 1e-10,
            "PE(0,1) should be 1.0, got {}",
            val1
        );
    }

    #[test]
    fn test_transformer_invalid_input_ndim() {
        let mut rng = SmallRng::seed_from_u64(5);
        let model =
            SequenceTransformer::<f64>::new(small_config(), &mut rng).expect("build failed");
        // 2-D input should fail
        let bad_src = scirs2_core::ndarray::Array2::<f64>::zeros((4, 32)).into_dyn();
        let bad_tgt = Array3::<f64>::zeros((2, 4, 32)).into_dyn();
        assert!(model.forward(&bad_src, &bad_tgt, None, None).is_err());
    }

    #[test]
    fn test_transformer_invalid_dmodel() {
        let mut rng = SmallRng::seed_from_u64(6);
        let model =
            SequenceTransformer::<f64>::new(small_config(), &mut rng).expect("build failed");
        // Wrong d_model (64 != 32)
        let bad_src = Array3::<f64>::zeros((2, 5, 64)).into_dyn();
        let tgt = Array3::<f64>::zeros((2, 4, 32)).into_dyn();
        assert!(model.forward(&bad_src, &tgt, None, None).is_err());
    }

    #[test]
    fn test_transformer_batch_size_mismatch() {
        let mut rng = SmallRng::seed_from_u64(7);
        let model =
            SequenceTransformer::<f64>::new(small_config(), &mut rng).expect("build failed");
        let src = Array3::<f64>::zeros((2, 6, 32)).into_dyn();
        let tgt = Array3::<f64>::zeros((3, 4, 32)).into_dyn(); // different batch
        assert!(model.forward(&src, &tgt, None, None).is_err());
    }

    #[test]
    fn test_transformer_clone() {
        let mut rng = SmallRng::seed_from_u64(8);
        let model =
            SequenceTransformer::<f64>::new(small_config(), &mut rng).expect("build failed");
        let cloned = model.clone();
        assert_eq!(cloned.config().d_model, model.config().d_model);
    }

    #[test]
    fn test_transformer_parameter_count() {
        let mut rng = SmallRng::seed_from_u64(9);
        let model =
            SequenceTransformer::<f64>::new(small_config(), &mut rng).expect("build failed");
        let count = model.parameter_count();
        assert!(count > 0, "Parameter count should be positive");
    }
}
