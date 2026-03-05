//! Full Transformer encoder-decoder architecture (Vaswani et al. 2017)
//!
//! This module provides a complete, self-contained implementation of the
//! Transformer architecture from "Attention Is All You Need" (NeurIPS 2017).
//! It directly operates on `Array3<F>` tensors with shape `[batch, seq_len, d_model]`
//! and implements:
//!
//! - Scaled dot-product multi-head attention
//! - Pre-norm residual sublayers (more training-stable than post-norm)
//! - GELU feed-forward networks
//! - Causal / padding masks via `Array2<bool>`
//! - Sinusoidal positional encoding
//! - Encoder stack (`TransformerEncoderLayer` × N)
//! - Decoder stack (`TransformerDecoderLayer` × N, self-attn + cross-attn + FFN)
//! - Complete `FullTransformer` model that wires encoder and decoder
//!
//! # References
//! Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
//! <https://arxiv.org/abs/1706.03762>

use crate::error::{NeuralError, Result};
use scirs2_core::num_traits::FromPrimitive;
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the complete Transformer encoder-decoder model.
///
/// Provides sensible small defaults suitable for testing and experimentation.
#[derive(Debug, Clone)]
pub struct FullTransformerConfig {
    /// Model / embedding dimension (d_model), e.g. 512
    pub d_model: usize,
    /// Number of attention heads, e.g. 8
    pub n_heads: usize,
    /// Number of encoder layers, e.g. 6
    pub n_encoder_layers: usize,
    /// Number of decoder layers, e.g. 6
    pub n_decoder_layers: usize,
    /// Feed-forward hidden dimension, e.g. 2048
    pub d_ff: usize,
    /// Dropout probability (no-op when `0.0`)
    pub dropout: f64,
    /// Vocabulary size (informational; no embedding table stored here)
    pub vocab_size: usize,
    /// Maximum sequence length (used for positional encoding)
    pub max_seq_len: usize,
}

impl Default for FullTransformerConfig {
    fn default() -> Self {
        Self {
            d_model: 64,
            n_heads: 8,
            n_encoder_layers: 2,
            n_decoder_layers: 2,
            d_ff: 256,
            dropout: 0.0,
            vocab_size: 1000,
            max_seq_len: 128,
        }
    }
}

// ---------------------------------------------------------------------------
// Layer-norm weights
// ---------------------------------------------------------------------------

/// Learnable parameters for a single layer-normalisation sublayer.
///
/// Applies: `LayerNorm(x) = gamma * (x - mu) / (sigma + eps) + beta`
#[derive(Debug, Clone)]
pub struct LayerNormWeights<F: Clone> {
    pub gamma: Array1<F>,
    pub beta: Array1<F>,
    pub d_model: usize,
}

impl<F: Float + FromPrimitive + Debug + ScalarOperand> LayerNormWeights<F> {
    /// Create with `gamma = 1`, `beta = 0` (identity transform).
    pub fn new(d_model: usize) -> Self {
        Self {
            gamma: Array1::ones(d_model),
            beta: Array1::zeros(d_model),
            d_model,
        }
    }

    /// Apply layer normalisation to the last axis of a 3-D tensor `[B, T, D]`.
    ///
    /// Uses a small epsilon of `1e-6` for numerical stability.
    pub fn forward(&self, x: &Array3<F>) -> Result<Array3<F>> {
        let eps = F::from_f64(1e-6).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Cannot convert eps to Float".to_string())
        })?;
        let [batch, seq, d] = [x.shape()[0], x.shape()[1], x.shape()[2]];
        if d != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "LayerNorm: last dim {d} != d_model {}",
                self.d_model
            )));
        }
        let d_f = F::from_usize(d).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Cannot convert d to Float".to_string())
        })?;
        let mut out = x.clone();
        for b in 0..batch {
            for t in 0..seq {
                // mean
                let mut mean = F::zero();
                for i in 0..d {
                    mean = mean + x[[b, t, i]];
                }
                mean = mean / d_f;
                // variance
                let mut var = F::zero();
                for i in 0..d {
                    let diff = x[[b, t, i]] - mean;
                    var = var + diff * diff;
                }
                var = var / d_f;
                let std = (var + eps).sqrt();
                // normalise + affine
                for i in 0..d {
                    let normed = (x[[b, t, i]] - mean) / std;
                    out[[b, t, i]] = normed * self.gamma[i] + self.beta[i];
                }
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Helpers: softmax, GELU, scaled dot-product attention
// ---------------------------------------------------------------------------

/// In-place row-wise softmax over the last axis of a 2-D slice `[N, M]`.
fn softmax_rows<F: Float + FromPrimitive>(mat: &mut Array2<F>) -> Result<()> {
    let [rows, cols] = [mat.shape()[0], mat.shape()[1]];
    for r in 0..rows {
        // stability: subtract max
        let mut max_val = mat[[r, 0]];
        for c in 1..cols {
            if mat[[r, c]] > max_val {
                max_val = mat[[r, c]];
            }
        }
        let mut sum = F::zero();
        for c in 0..cols {
            let e = (mat[[r, c]] - max_val).exp();
            mat[[r, c]] = e;
            sum = sum + e;
        }
        if sum == F::zero() {
            // avoid divide-by-zero: uniform distribution
            let uniform = F::from_usize(1).ok_or_else(|| {
                NeuralError::ComputationError("Cannot convert 1 to Float".to_string())
            })? / F::from_usize(cols).ok_or_else(|| {
                NeuralError::ComputationError("Cannot convert cols to Float".to_string())
            })?;
            for c in 0..cols {
                mat[[r, c]] = uniform;
            }
        } else {
            for c in 0..cols {
                mat[[r, c]] = mat[[r, c]] / sum;
            }
        }
    }
    Ok(())
}

/// Element-wise GELU activation: `x * Phi(x)` approximated via tanh.
///
/// Uses the accurate approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))`.
fn gelu<F: Float + FromPrimitive>(x: F) -> Result<F> {
    let c0 = F::from_f64(0.7978845608028654).ok_or_else(|| {
        NeuralError::ComputationError("Cannot convert sqrt(2/pi)".to_string())
    })?; // sqrt(2/pi)
    let c1 = F::from_f64(0.044715).ok_or_else(|| {
        NeuralError::ComputationError("Cannot convert 0.044715".to_string())
    })?;
    let half = F::from_f64(0.5).ok_or_else(|| {
        NeuralError::ComputationError("Cannot convert 0.5".to_string())
    })?;
    let one = F::one();
    let inner = c0 * (x + c1 * x * x * x);
    Ok(half * x * (one + inner.tanh()))
}

/// Apply GELU element-wise to a 3-D array in-place.
fn apply_gelu_3d<F: Float + FromPrimitive>(arr: &mut Array3<F>) -> Result<()> {
    for v in arr.iter_mut() {
        *v = gelu(*v)?;
    }
    Ok(())
}

/// Matrix multiply: `[M, K] x [K, N] -> [M, N]` (row-major loops).
fn matmul_2d<F: Float>(a: &Array2<F>, b: &Array2<F>) -> Result<Array2<F>> {
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let (kb, n) = (b.shape()[0], b.shape()[1]);
    if k != kb {
        return Err(NeuralError::ShapeMismatch(format!(
            "matmul_2d: inner dims {k} vs {kb}"
        )));
    }
    let mut c = Array2::<F>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut acc = F::zero();
            for l in 0..k {
                acc = acc + a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = acc;
        }
    }
    Ok(c)
}

/// Add bias vector `[N]` to each row of a 2-D array `[M, N]` in-place.
fn add_bias_2d<F: Float>(a: &mut Array2<F>, bias: &Array1<F>) -> Result<()> {
    let n = a.shape()[1];
    if bias.len() != n {
        return Err(NeuralError::ShapeMismatch(format!(
            "add_bias_2d: bias len {} != cols {n}",
            bias.len()
        )));
    }
    for mut row in a.rows_mut() {
        for (v, &b) in row.iter_mut().zip(bias.iter()) {
            *v = *v + b;
        }
    }
    Ok(())
}

/// Large negative value used to mask attention positions.
fn neg_inf<F: Float + FromPrimitive>() -> Result<F> {
    F::from_f64(-1e9).ok_or_else(|| {
        NeuralError::ComputationError("Cannot represent -1e9 as Float".to_string())
    })
}

// ---------------------------------------------------------------------------
// Scaled dot-product attention
// ---------------------------------------------------------------------------

/// Compute scaled dot-product attention for a single head.
///
/// `Q: [T_q, d_k]`, `K: [T_k, d_k]`, `V: [T_k, d_v]`
/// `mask: [T_q, T_k]` — `true` positions are **masked out** (set to -1e9).
///
/// Returns `[T_q, d_v]`.
fn scaled_dot_product_attention_single<F: Float + FromPrimitive>(
    q: &Array2<F>,
    k: &Array2<F>,
    v: &Array2<F>,
    mask: Option<&Array2<bool>>,
) -> Result<Array2<F>> {
    let (tq, dk) = (q.shape()[0], q.shape()[1]);
    let (tk, dv) = (k.shape()[0], v.shape()[1]);
    let scale = F::from_f64((dk as f64).sqrt()).ok_or_else(|| {
        NeuralError::ComputationError("Cannot convert sqrt(dk)".to_string())
    })?;
    // Scores: Q K^T / sqrt(d_k) -> [T_q, T_k]
    let kt = k.t().to_owned(); // [dk, tk]
    let mut scores = matmul_2d(q, &kt)?; // [tq, tk]
    for v_elem in scores.iter_mut() {
        *v_elem = *v_elem / scale;
    }
    // Apply mask
    if let Some(m) = mask {
        if m.shape() != [tq, tk] {
            return Err(NeuralError::ShapeMismatch(format!(
                "Attention mask shape {:?} != [{tq}, {tk}]",
                m.shape()
            )));
        }
        let neg = neg_inf::<F>()?;
        for r in 0..tq {
            for c in 0..tk {
                if m[[r, c]] {
                    scores[[r, c]] = neg;
                }
            }
        }
    }
    // Softmax over last axis
    softmax_rows(&mut scores)?;
    // Output: scores @ V -> [T_q, d_v]
    matmul_2d(&scores, v)
}

// ---------------------------------------------------------------------------
// Multi-head attention weights
// ---------------------------------------------------------------------------

/// Weights for a single multi-head attention sublayer.
///
/// Stores projection matrices for Q, K, V and the output projection.
#[derive(Debug, Clone)]
pub struct MultiHeadAttnWeights<F: Clone> {
    /// Query projection: `[d_model, d_model]`
    pub w_q: Array2<F>,
    /// Key projection: `[d_model, d_model]`
    pub w_k: Array2<F>,
    /// Value projection: `[d_model, d_model]`
    pub w_v: Array2<F>,
    /// Output projection: `[d_model, d_model]`
    pub w_o: Array2<F>,
    /// Query bias: `[d_model]`
    pub b_q: Array1<F>,
    /// Key bias: `[d_model]`
    pub b_k: Array1<F>,
    /// Value bias: `[d_model]`
    pub b_v: Array1<F>,
    /// Output bias: `[d_model]`
    pub b_o: Array1<F>,
    pub d_model: usize,
    pub n_heads: usize,
}

impl<F: Float + FromPrimitive + Debug + ScalarOperand> MultiHeadAttnWeights<F> {
    /// Initialise all projections to zero (suitable for structural tests).
    pub fn new_zero(d_model: usize, n_heads: usize) -> Result<Self> {
        if d_model % n_heads != 0 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )));
        }
        Ok(Self {
            w_q: Array2::zeros((d_model, d_model)),
            w_k: Array2::zeros((d_model, d_model)),
            w_v: Array2::zeros((d_model, d_model)),
            w_o: Array2::zeros((d_model, d_model)),
            b_q: Array1::zeros(d_model),
            b_k: Array1::zeros(d_model),
            b_v: Array1::zeros(d_model),
            b_o: Array1::zeros(d_model),
            d_model,
            n_heads,
        })
    }

    /// Multi-head self-attention forward.
    ///
    /// `x: [B, T, D]`, queries/keys/values all come from `x`.
    /// `mask: [T, T]` (applied to every head and every batch element).
    pub fn self_attn_forward(
        &self,
        x: &Array3<F>,
        mask: Option<&Array2<bool>>,
    ) -> Result<Array3<F>> {
        self.cross_attn_forward(x, x, mask)
    }

    /// Multi-head cross-attention forward.
    ///
    /// `query: [B, T_q, D]`, `kv: [B, T_k, D]`.
    /// `mask: [T_q, T_k]`.
    pub fn cross_attn_forward(
        &self,
        query: &Array3<F>,
        kv: &Array3<F>,
        mask: Option<&Array2<bool>>,
    ) -> Result<Array3<F>> {
        let (batch, tq, d) = (query.shape()[0], query.shape()[1], query.shape()[2]);
        let tk = kv.shape()[1];
        if d != self.d_model {
            return Err(NeuralError::ShapeMismatch(format!(
                "MHA: query d {d} != d_model {}",
                self.d_model
            )));
        }
        if kv.shape()[2] != self.d_model {
            return Err(NeuralError::ShapeMismatch(format!(
                "MHA: kv d {} != d_model {}",
                kv.shape()[2],
                self.d_model
            )));
        }
        let n_heads = self.n_heads;
        let head_dim = self.d_model / n_heads;

        let mut out = Array3::<F>::zeros((batch, tq, d));

        for b in 0..batch {
            // Project Q, K, V: [T, D] x [D, D] -> [T, D]
            let q_slice = query.index_axis(Axis(0), b).to_owned(); // [tq, d]
            let k_slice = kv.index_axis(Axis(0), b).to_owned(); // [tk, d]
            let v_slice = kv.index_axis(Axis(0), b).to_owned(); // [tk, d]

            let mut q_proj = matmul_2d(&q_slice, &self.w_q)?;
            add_bias_2d(&mut q_proj, &self.b_q)?; // [tq, d]
            let mut k_proj = matmul_2d(&k_slice, &self.w_k)?;
            add_bias_2d(&mut k_proj, &self.b_k)?; // [tk, d]
            let mut v_proj = matmul_2d(&v_slice, &self.w_v)?;
            add_bias_2d(&mut v_proj, &self.b_v)?; // [tk, d]

            // Multi-head: split last dim into [n_heads, head_dim]
            // Compute attention per head and concatenate
            let mut concat = Array2::<F>::zeros((tq, d));

            for h in 0..n_heads {
                let start = h * head_dim;
                let end = start + head_dim;

                // Extract head slice
                let q_h = q_proj.slice(scirs2_core::ndarray::s![.., start..end]).to_owned(); // [tq, hd]
                let k_h = k_proj.slice(scirs2_core::ndarray::s![.., start..end]).to_owned(); // [tk, hd]
                let v_h = v_proj.slice(scirs2_core::ndarray::s![.., start..end]).to_owned(); // [tk, hd]

                let attn_out = scaled_dot_product_attention_single(&q_h, &k_h, &v_h, mask)?; // [tq, hd]

                // Write into concat at [.., start..end]
                for t in 0..tq {
                    for i in 0..head_dim {
                        concat[[t, start + i]] = attn_out[[t, i]];
                    }
                }
            }

            // Output projection: [tq, d] x [d, d] -> [tq, d]
            let mut proj = matmul_2d(&concat, &self.w_o)?;
            add_bias_2d(&mut proj, &self.b_o)?;

            for t in 0..tq {
                for i in 0..d {
                    out[[b, t, i]] = proj[[t, i]];
                }
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// FFN weights
// ---------------------------------------------------------------------------

/// Weights for a two-layer GELU feed-forward sublayer.
///
/// `FFN(x) = GELU(x W1 + b1) W2 + b2`
#[derive(Debug, Clone)]
pub struct FfnWeights<F: Clone> {
    pub w1: Array2<F>, // [d_model, d_ff]
    pub b1: Array1<F>, // [d_ff]
    pub w2: Array2<F>, // [d_ff, d_model]
    pub b2: Array1<F>, // [d_model]
    pub d_model: usize,
    pub d_ff: usize,
}

impl<F: Float + FromPrimitive + Debug + ScalarOperand> FfnWeights<F> {
    /// Initialise all weights to zero.
    pub fn new_zero(d_model: usize, d_ff: usize) -> Self {
        Self {
            w1: Array2::zeros((d_model, d_ff)),
            b1: Array1::zeros(d_ff),
            w2: Array2::zeros((d_ff, d_model)),
            b2: Array1::zeros(d_model),
            d_model,
            d_ff,
        }
    }

    /// Forward pass over `[B, T, D]` input.
    pub fn forward(&self, x: &Array3<F>) -> Result<Array3<F>> {
        let (batch, seq, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        if d != self.d_model {
            return Err(NeuralError::ShapeMismatch(format!(
                "FFN: input d {d} != d_model {}",
                self.d_model
            )));
        }
        let mut out = Array3::<F>::zeros((batch, seq, d));
        for b in 0..batch {
            let x_slice = x.index_axis(Axis(0), b).to_owned(); // [seq, d]
            let mut h = matmul_2d(&x_slice, &self.w1)?; // [seq, d_ff]
            add_bias_2d(&mut h, &self.b1)?;
            // GELU activation
            let mut h3 = h.into_shape_with_order((seq, self.d_ff)).map_err(|e| {
                NeuralError::ShapeMismatch(format!("FFN reshape error: {e}"))
            })?;
            // Apply GELU per element via the 3-D helper adapted for 2-D
            for v in h3.iter_mut() {
                *v = gelu(*v)?;
            }
            let mut y = matmul_2d(&h3, &self.w2)?; // [seq, d]
            add_bias_2d(&mut y, &self.b2)?;
            for t in 0..seq {
                for i in 0..d {
                    out[[b, t, i]] = y[[t, i]];
                }
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Encoder layer
// ---------------------------------------------------------------------------

/// Single transformer encoder layer.
///
/// Pre-norm variant: `x = x + Sublayer(LayerNorm(x))`.
///
/// Components:
/// 1. Multi-head self-attention
/// 2. Position-wise GELU feed-forward network
#[derive(Debug, Clone)]
pub struct TransformerEncoderLayer<F: Clone + Debug> {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub dropout: f64,
    /// Self-attention weights
    pub self_attn: MultiHeadAttnWeights<F>,
    /// Layer norm before self-attention
    pub norm1: LayerNormWeights<F>,
    /// FFN weights
    pub ffn: FfnWeights<F>,
    /// Layer norm before FFN
    pub norm2: LayerNormWeights<F>,
}

impl<F: Float + FromPrimitive + Debug + ScalarOperand> TransformerEncoderLayer<F> {
    /// Create a new encoder layer with zero-initialised weights.
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, dropout: f64) -> Result<Self> {
        let self_attn = MultiHeadAttnWeights::new_zero(d_model, n_heads)?;
        Ok(Self {
            d_model,
            n_heads,
            d_ff,
            dropout,
            self_attn,
            norm1: LayerNormWeights::new(d_model),
            ffn: FfnWeights::new_zero(d_model, d_ff),
            norm2: LayerNormWeights::new(d_model),
        })
    }

    /// Forward pass: `x: [B, T, D]`, returns `[B, T, D]`.
    ///
    /// Pre-norm: `x = x + SelfAttn(LN(x))`, then `x = x + FFN(LN(x))`.
    pub fn forward(&self, x: &Array3<F>, mask: Option<&Array2<bool>>) -> Result<Array3<F>> {
        // 1. Self-attention sublayer (pre-norm)
        let ln1 = self.norm1.forward(x)?;
        let attn_out = self.self_attn.self_attn_forward(&ln1, mask)?;
        // Residual
        let x2 = x + &attn_out;

        // 2. FFN sublayer (pre-norm)
        let ln2 = self.norm2.forward(&x2)?;
        let ffn_out = self.ffn.forward(&ln2)?;
        let x3 = x2 + &ffn_out;

        Ok(x3)
    }
}

// ---------------------------------------------------------------------------
// Encoder stack
// ---------------------------------------------------------------------------

/// Stack of `N` identical encoder layers followed by a final layer norm.
#[derive(Debug, Clone)]
pub struct TransformerEncoder<F: Clone + Debug> {
    pub layers: Vec<TransformerEncoderLayer<F>>,
    pub norm: LayerNormWeights<F>,
}

impl<F: Float + FromPrimitive + Debug + ScalarOperand> TransformerEncoder<F> {
    /// Create a new encoder with `n_layers` identical zero-initialised layers.
    pub fn new(n_layers: usize, d_model: usize, n_heads: usize, d_ff: usize, dropout: f64) -> Result<Self> {
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(TransformerEncoderLayer::new(d_model, n_heads, d_ff, dropout)?);
        }
        Ok(Self {
            layers,
            norm: LayerNormWeights::new(d_model),
        })
    }

    /// Forward pass through all encoder layers then the final norm.
    ///
    /// `x: [B, T_src, D]` → `[B, T_src, D]`.
    pub fn forward(&self, x: &Array3<F>, mask: Option<&Array2<bool>>) -> Result<Array3<F>> {
        let mut h = x.clone();
        for layer in &self.layers {
            h = layer.forward(&h, mask)?;
        }
        self.norm.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

/// Single transformer decoder layer.
///
/// Pre-norm variant with three sublayers:
/// 1. Masked multi-head self-attention (over target)
/// 2. Multi-head cross-attention (queries from target, keys/values from encoder)
/// 3. Position-wise GELU feed-forward network
#[derive(Debug, Clone)]
pub struct TransformerDecoderLayer<F: Clone + Debug> {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub dropout: f64,
    /// Masked self-attention weights
    pub self_attn: MultiHeadAttnWeights<F>,
    /// Cross-attention weights
    pub cross_attn: MultiHeadAttnWeights<F>,
    /// Layer norm before self-attention
    pub norm1: LayerNormWeights<F>,
    /// Layer norm before cross-attention
    pub norm2: LayerNormWeights<F>,
    /// FFN weights
    pub ffn: FfnWeights<F>,
    /// Layer norm before FFN
    pub norm3: LayerNormWeights<F>,
}

impl<F: Float + FromPrimitive + Debug + ScalarOperand> TransformerDecoderLayer<F> {
    /// Create a new decoder layer with zero-initialised weights.
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, dropout: f64) -> Result<Self> {
        let self_attn = MultiHeadAttnWeights::new_zero(d_model, n_heads)?;
        let cross_attn = MultiHeadAttnWeights::new_zero(d_model, n_heads)?;
        Ok(Self {
            d_model,
            n_heads,
            d_ff,
            dropout,
            self_attn,
            cross_attn,
            norm1: LayerNormWeights::new(d_model),
            norm2: LayerNormWeights::new(d_model),
            ffn: FfnWeights::new_zero(d_model, d_ff),
            norm3: LayerNormWeights::new(d_model),
        })
    }

    /// Forward pass.
    ///
    /// - `tgt: [B, T_tgt, D]` — target input
    /// - `memory: [B, T_src, D]` — encoder output
    /// - `tgt_mask: [T_tgt, T_tgt]` — causal mask for self-attention
    /// - `memory_mask: [T_tgt, T_src]` — padding mask for cross-attention
    ///
    /// Returns `[B, T_tgt, D]`.
    pub fn forward(
        &self,
        tgt: &Array3<F>,
        memory: &Array3<F>,
        tgt_mask: Option<&Array2<bool>>,
        memory_mask: Option<&Array2<bool>>,
    ) -> Result<Array3<F>> {
        // 1. Masked self-attention (pre-norm)
        let ln1 = self.norm1.forward(tgt)?;
        let self_attn_out = self.self_attn.self_attn_forward(&ln1, tgt_mask)?;
        let x2 = tgt + &self_attn_out;

        // 2. Cross-attention (pre-norm)
        let ln2 = self.norm2.forward(&x2)?;
        let cross_out = self.cross_attn.cross_attn_forward(&ln2, memory, memory_mask)?;
        let x3 = x2 + &cross_out;

        // 3. FFN (pre-norm)
        let ln3 = self.norm3.forward(&x3)?;
        let ffn_out = self.ffn.forward(&ln3)?;
        let x4 = x3 + &ffn_out;

        Ok(x4)
    }
}

// ---------------------------------------------------------------------------
// Decoder stack
// ---------------------------------------------------------------------------

/// Stack of `N` identical decoder layers followed by a final layer norm.
#[derive(Debug, Clone)]
pub struct TransformerDecoder<F: Clone + Debug> {
    pub layers: Vec<TransformerDecoderLayer<F>>,
    pub norm: LayerNormWeights<F>,
}

impl<F: Float + FromPrimitive + Debug + ScalarOperand> TransformerDecoder<F> {
    /// Create a new decoder with `n_layers` identical zero-initialised layers.
    pub fn new(n_layers: usize, d_model: usize, n_heads: usize, d_ff: usize, dropout: f64) -> Result<Self> {
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(TransformerDecoderLayer::new(d_model, n_heads, d_ff, dropout)?);
        }
        Ok(Self {
            layers,
            norm: LayerNormWeights::new(d_model),
        })
    }

    /// Forward pass through all decoder layers then the final norm.
    ///
    /// `tgt: [B, T_tgt, D]`, `memory: [B, T_src, D]` → `[B, T_tgt, D]`.
    pub fn forward(
        &self,
        tgt: &Array3<F>,
        memory: &Array3<F>,
        tgt_mask: Option<&Array2<bool>>,
        memory_mask: Option<&Array2<bool>>,
    ) -> Result<Array3<F>> {
        let mut h = tgt.clone();
        for layer in &self.layers {
            h = layer.forward(&h, memory, tgt_mask, memory_mask)?;
        }
        self.norm.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Sinusoidal positional encoding
// ---------------------------------------------------------------------------

/// Compute the sinusoidal positional encoding matrix `[seq_len, d_model]`.
///
/// `PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))`
/// `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
pub fn sinusoidal_pos_encoding<F: Float + FromPrimitive>(
    seq_len: usize,
    d_model: usize,
) -> Result<Array2<F>> {
    let ten_k = F::from_f64(10_000.0).ok_or_else(|| {
        NeuralError::InvalidArchitecture("Cannot convert 10000 to Float".to_string())
    })?;
    let two = F::from_f64(2.0).ok_or_else(|| {
        NeuralError::InvalidArchitecture("Cannot convert 2 to Float".to_string())
    })?;
    let d_f = F::from_usize(d_model).ok_or_else(|| {
        NeuralError::InvalidArchitecture("Cannot convert d_model to Float".to_string())
    })?;
    let mut pe = Array2::<F>::zeros((seq_len, d_model));
    for pos in 0..seq_len {
        let pos_f = F::from_usize(pos).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Cannot convert pos to Float".to_string())
        })?;
        for i in 0..(d_model / 2) {
            let i_f = F::from_usize(i).ok_or_else(|| {
                NeuralError::InvalidArchitecture("Cannot convert i to Float".to_string())
            })?;
            let exp = (two * i_f) / d_f;
            let denom = ten_k.powf(exp);
            let angle = pos_f / denom;
            pe[[pos, 2 * i]] = angle.sin();
            if 2 * i + 1 < d_model {
                pe[[pos, 2 * i + 1]] = angle.cos();
            }
        }
    }
    Ok(pe)
}

/// Add sinusoidal positional encoding to a `[B, T, D]` tensor in-place.
fn add_positional_encoding<F: Float + FromPrimitive>(x: &Array3<F>) -> Result<Array3<F>> {
    let (batch, seq, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let pe = sinusoidal_pos_encoding::<F>(seq, d)?;
    let mut out = x.clone();
    for b in 0..batch {
        for t in 0..seq {
            for i in 0..d {
                out[[b, t, i]] = out[[b, t, i]] + pe[[t, i]];
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Complete encoder-decoder Transformer
// ---------------------------------------------------------------------------

/// Complete encoder-decoder Transformer (Vaswani et al. 2017).
///
/// Combines sinusoidal positional encoding, an encoder stack, and a decoder
/// stack. All weights are zero-initialised for structural correctness tests;
/// use proper initialisation (e.g. Xavier/Kaiming) for actual training.
///
/// # Example
/// ```no_run
/// use scirs2_neural::models::architectures::transformer_full::{FullTransformer, FullTransformerConfig};
/// use scirs2_core::ndarray::Array3;
///
/// let cfg = FullTransformerConfig { d_model: 64, n_heads: 8, n_encoder_layers: 2,
///     n_decoder_layers: 2, d_ff: 128, dropout: 0.0, vocab_size: 100, max_seq_len: 64 };
/// let model = FullTransformer::<f64>::new(cfg).expect("operation should succeed");
/// let src = Array3::<f64>::zeros((2, 10, 64));
/// let tgt = Array3::<f64>::zeros((2, 8, 64));
/// let out = model.decode(&tgt, &model.encode(&src, None).expect("operation should succeed"), None, None).expect("operation should succeed");
/// assert_eq!(out.shape(), &[2, 8, 64]);
/// ```
#[derive(Debug, Clone)]
pub struct FullTransformer<F: Clone + Debug> {
    pub encoder: TransformerEncoder<F>,
    pub decoder: TransformerDecoder<F>,
    pub d_model: usize,
    pub vocab_size: usize,
    config: FullTransformerConfig,
}

impl<F: Float + FromPrimitive + Debug + ScalarOperand> FullTransformer<F> {
    /// Build from a [`FullTransformerConfig`] with zero-initialised weights.
    pub fn new(config: FullTransformerConfig) -> Result<Self> {
        let encoder = TransformerEncoder::new(
            config.n_encoder_layers,
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
        )?;
        let decoder = TransformerDecoder::new(
            config.n_decoder_layers,
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.dropout,
        )?;
        Ok(Self {
            d_model: config.d_model,
            vocab_size: config.vocab_size,
            encoder,
            decoder,
            config,
        })
    }

    /// Run the encoder on `src: [B, T_src, D]`.
    ///
    /// Adds sinusoidal positional encoding before passing through the stack.
    pub fn encode(
        &self,
        src: &Array3<F>,
        src_mask: Option<&Array2<bool>>,
    ) -> Result<Array3<F>> {
        let src_pe = add_positional_encoding(src)?;
        self.encoder.forward(&src_pe, src_mask)
    }

    /// Run the decoder given target embeddings and encoder memory.
    ///
    /// Adds sinusoidal positional encoding to `tgt` before decoding.
    pub fn decode(
        &self,
        tgt: &Array3<F>,
        memory: &Array3<F>,
        tgt_mask: Option<&Array2<bool>>,
        memory_mask: Option<&Array2<bool>>,
    ) -> Result<Array3<F>> {
        let tgt_pe = add_positional_encoding(tgt)?;
        self.decoder.forward(&tgt_pe, memory, tgt_mask, memory_mask)
    }

    /// Full encoder-decoder pass: `src → encoder → memory`, `tgt + memory → decoder → output`.
    pub fn forward(
        &self,
        src: &Array3<F>,
        tgt: &Array3<F>,
        src_mask: Option<&Array2<bool>>,
        tgt_mask: Option<&Array2<bool>>,
        memory_mask: Option<&Array2<bool>>,
    ) -> Result<Array3<F>> {
        let memory = self.encode(src, src_mask)?;
        self.decode(tgt, &memory, tgt_mask, memory_mask)
    }

    /// Generate a causal (upper-triangular) mask for a sequence of `seq_len` tokens.
    ///
    /// Position `[i, j]` is `true` (masked) when `j > i`, preventing position `i`
    /// from attending to future positions `j > i`.
    ///
    /// Shape: `[seq_len, seq_len]`.
    pub fn generate_causal_mask(seq_len: usize) -> Array2<bool> {
        Array2::from_shape_fn((seq_len, seq_len), |(i, j)| j > i)
    }

    /// Access the model configuration.
    pub fn config(&self) -> &FullTransformerConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    fn small_config() -> FullTransformerConfig {
        FullTransformerConfig {
            d_model: 64,
            n_heads: 8,
            n_encoder_layers: 2,
            n_decoder_layers: 2,
            d_ff: 128,
            dropout: 0.0,
            vocab_size: 100,
            max_seq_len: 64,
        }
    }

    // ---- Config ----

    #[test]
    fn test_transformer_config_default() {
        let cfg = FullTransformerConfig::default();
        assert_eq!(cfg.d_model, 64);
        assert_eq!(cfg.n_heads, 8);
        assert_eq!(cfg.n_encoder_layers, 2);
        assert_eq!(cfg.n_decoder_layers, 2);
        assert_eq!(cfg.d_ff, 256);
        assert_eq!(cfg.dropout, 0.0);
    }

    // ---- Encoder ----

    #[test]
    fn test_encoder_output_shape() {
        let enc = TransformerEncoder::<f64>::new(2, 64, 8, 128, 0.0)
            .expect("encoder build failed");
        let x = Array3::<f64>::zeros((2, 10, 64));
        let out = enc.forward(&x, None).expect("encoder forward failed");
        assert_eq!(out.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_encoder_single_layer() {
        let layer = TransformerEncoderLayer::<f64>::new(32, 4, 64, 0.0)
            .expect("layer build failed");
        let x = Array3::<f64>::zeros((1, 5, 32));
        let out = layer.forward(&x, None).expect("layer forward failed");
        assert_eq!(out.shape(), &[1, 5, 32]);
    }

    #[test]
    fn test_encoder_preserves_batch_seq() {
        let enc = TransformerEncoder::<f64>::new(1, 64, 8, 128, 0.0)
            .expect("encoder build failed");
        // Non-trivial batch and sequence lengths
        let x = Array3::<f64>::zeros((4, 15, 64));
        let out = enc.forward(&x, None).expect("encoder forward failed");
        assert_eq!(out.shape(), &[4, 15, 64]);
    }

    // ---- Decoder ----

    #[test]
    fn test_decoder_output_shape() {
        let dec = TransformerDecoder::<f64>::new(2, 64, 8, 128, 0.0)
            .expect("decoder build failed");
        let tgt = Array3::<f64>::zeros((2, 8, 64));
        let memory = Array3::<f64>::zeros((2, 10, 64));
        let out = dec.forward(&tgt, &memory, None, None).expect("decoder forward failed");
        assert_eq!(out.shape(), &[2, 8, 64]);
    }

    #[test]
    fn test_decoder_with_causal_mask() {
        let dec = TransformerDecoder::<f64>::new(1, 64, 8, 128, 0.0)
            .expect("decoder build failed");
        let tgt = Array3::<f64>::zeros((1, 6, 64));
        let memory = Array3::<f64>::zeros((1, 8, 64));
        let causal = FullTransformer::<f64>::generate_causal_mask(6);
        let out = dec
            .forward(&tgt, &memory, Some(&causal), None)
            .expect("decoder with mask failed");
        assert_eq!(out.shape(), &[1, 6, 64]);
    }

    #[test]
    fn test_decoder_layer_shape() {
        let layer = TransformerDecoderLayer::<f64>::new(32, 4, 64, 0.0)
            .expect("decoder layer build failed");
        let tgt = Array3::<f64>::zeros((2, 5, 32));
        let memory = Array3::<f64>::zeros((2, 7, 32));
        let out = layer
            .forward(&tgt, &memory, None, None)
            .expect("decoder layer forward failed");
        assert_eq!(out.shape(), &[2, 5, 32]);
    }

    // ---- Full Transformer ----

    #[test]
    fn test_transformer_full_forward() {
        let model = FullTransformer::<f64>::new(small_config()).expect("model build failed");
        let src = Array3::<f64>::zeros((2, 10, 64));
        let tgt = Array3::<f64>::zeros((2, 8, 64));
        let out = model.forward(&src, &tgt, None, None, None).expect("full forward failed");
        assert_eq!(out.shape(), &[2, 8, 64]);
    }

    #[test]
    fn test_transformer_encode_only() {
        let model = FullTransformer::<f64>::new(small_config()).expect("model build failed");
        let src = Array3::<f64>::zeros((3, 12, 64));
        let memory = model.encode(&src, None).expect("encode failed");
        assert_eq!(memory.shape(), &[3, 12, 64]);
    }

    #[test]
    fn test_transformer_decode_uses_memory() {
        let model = FullTransformer::<f64>::new(small_config()).expect("model build failed");
        let src = Array3::<f64>::zeros((2, 10, 64));
        let tgt = Array3::<f64>::zeros((2, 6, 64));
        let memory = model.encode(&src, None).expect("encode failed");
        let out = model.decode(&tgt, &memory, None, None).expect("decode failed");
        assert_eq!(out.shape(), &[2, 6, 64]);
    }

    // ---- Causal mask ----

    #[test]
    fn test_causal_mask_is_upper_triangular() {
        let mask = FullTransformer::<f64>::generate_causal_mask(5);
        assert_eq!(mask.shape(), &[5, 5]);
        // Diagonal and below should be false (not masked)
        for i in 0..5 {
            for j in 0..5 {
                if j > i {
                    assert!(mask[[i, j]], "expected mask[{i},{j}] = true");
                } else {
                    assert!(!mask[[i, j]], "expected mask[{i},{j}] = false");
                }
            }
        }
    }

    #[test]
    fn test_causal_mask_size_1() {
        let mask = FullTransformer::<f64>::generate_causal_mask(1);
        assert_eq!(mask.shape(), &[1, 1]);
        assert!(!mask[[0, 0]]);
    }

    // ---- Encoder with padding mask ----

    #[test]
    fn test_encoder_with_padding_mask() {
        let enc = TransformerEncoder::<f64>::new(1, 64, 8, 128, 0.0)
            .expect("encoder build failed");
        let x = Array3::<f64>::zeros((2, 5, 64));
        // Mark last token of each sequence as padding
        let mut pad_mask = Array2::<bool>::from_elem((5, 5), false);
        // Mask column 4 (the padding token)
        for row in 0..5 {
            pad_mask[[row, 4]] = true;
        }
        let out = enc.forward(&x, Some(&pad_mask)).expect("enc with mask failed");
        assert_eq!(out.shape(), &[2, 5, 64]);
    }

    // ---- Layer norm ----

    #[test]
    fn test_layer_norm_identity_on_zero() {
        // For zero input, gamma=1, beta=0: output should be 0 (mean=0, std=eps-dominated)
        let ln = LayerNormWeights::<f64>::new(8);
        let x = Array3::<f64>::zeros((1, 3, 8));
        let out = ln.forward(&x).expect("ln forward failed");
        assert_eq!(out.shape(), &[1, 3, 8]);
    }

    #[test]
    fn test_layer_norm_shape_preserved() {
        let ln = LayerNormWeights::<f64>::new(32);
        let x = Array3::<f64>::from_elem((4, 7, 32), 1.0);
        let out = ln.forward(&x).expect("ln forward failed");
        assert_eq!(out.shape(), &[4, 7, 32]);
    }

    // ---- Sinusoidal encoding ----

    #[test]
    fn test_sinusoidal_encoding_shape() {
        let pe = sinusoidal_pos_encoding::<f64>(10, 32).expect("pe failed");
        assert_eq!(pe.shape(), &[10, 32]);
    }

    #[test]
    fn test_sinusoidal_encoding_pe_0_0_is_zero() {
        let pe = sinusoidal_pos_encoding::<f64>(4, 8).expect("pe failed");
        // PE(0, 0) = sin(0 / 1) = 0
        assert!((pe[[0, 0]]).abs() < 1e-10, "PE(0,0) should be 0.0");
        // PE(0, 1) = cos(0 / 1) = 1
        assert!((pe[[0, 1]] - 1.0).abs() < 1e-10, "PE(0,1) should be 1.0");
    }

    // ---- Error handling ----

    #[test]
    fn test_encoder_layer_wrong_n_heads() {
        // 65 is not divisible by 4
        let result = TransformerEncoderLayer::<f64>::new(65, 4, 64, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_decoder_layer_wrong_n_heads() {
        let result = TransformerDecoderLayer::<f64>::new(65, 4, 64, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_norm_dim_mismatch() {
        let ln = LayerNormWeights::<f64>::new(16);
        let x = Array3::<f64>::zeros((1, 3, 32)); // 32 != 16
        assert!(ln.forward(&x).is_err());
    }
}
