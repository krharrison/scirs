//! Advanced attention mechanism implementations
//!
//! This module provides state-of-the-art attention variants used in modern
//! transformer architectures:
//!
//! - **Relative Positional Attention** (Shaw et al., 2018): Adds learnable
//!   relative position embeddings to attention scores.
//! - **Rotary Position Embedding (RoPE)** (Su et al., 2021): Applies rotation
//!   matrices to encode positional information in queries and keys.
//! - **Grouped Query Attention (GQA)** (Ainslie et al., 2023): Shares key/value
//!   heads across multiple query heads for memory efficiency.
//! - **Sliding Window Attention** (Beltagy et al., 2020): Restricts attention
//!   to a local window around each token for linear complexity.
//! - **Linear Attention** (Katharopoulos et al., 2020): Replaces softmax with
//!   kernel feature maps for O(n) complexity.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// Helpers shared across attention variants
// ---------------------------------------------------------------------------

/// Compute softmax along the last dimension of a multi-dimensional array.
fn softmax_last_dim<F: Float + NumAssign>(input: &Array<F, IxDyn>) -> Array<F, IxDyn> {
    let shape = input.shape().to_vec();
    let last_dim = shape.len() - 1;
    let last_size = shape[last_dim];
    let mut output = input.clone();
    let num_outer: usize = shape[..last_dim].iter().product();

    for idx in 0..num_outer {
        let mut remaining = idx;
        let mut indices: Vec<usize> = Vec::with_capacity(last_dim);
        for &dim_size in shape[..last_dim].iter().rev() {
            indices.push(remaining % dim_size);
            remaining /= dim_size;
        }
        indices.reverse();

        // max for numerical stability
        let mut max_val = F::neg_infinity();
        for k in 0..last_size {
            let mut full = indices.clone();
            full.push(k);
            let val = input[IxDyn(&full)];
            if val > max_val {
                max_val = val;
            }
        }

        let mut sum = F::zero();
        let mut exp_vals = Vec::with_capacity(last_size);
        for k in 0..last_size {
            let mut full = indices.clone();
            full.push(k);
            let e = (input[IxDyn(&full)] - max_val).exp();
            exp_vals.push(e);
            sum += e;
        }

        for (k, &e) in exp_vals.iter().enumerate() {
            let mut full = indices.clone();
            full.push(k);
            output[IxDyn(&full)] = e / sum;
        }
    }
    output
}

/// Xavier weight initializer returning a flat Vec.
fn xavier_init<F: Float, R: Rng>(
    fan_in: usize,
    fan_out: usize,
    count: usize,
    rng: &mut R,
) -> Result<Vec<F>> {
    let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();
    let mut data = Vec::with_capacity(count);
    for _ in 0..count {
        let val: f64 = rng.random_range(-1.0..1.0);
        let scaled = F::from(val * scale).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert init value".into())
        })?;
        data.push(scaled);
    }
    Ok(data)
}

/// Linear projection: [batch, seq, d_in] @ [d_in, d_out] -> [batch, seq, d_out]
fn linear_project<F: Float + NumAssign>(
    input: &Array<F, IxDyn>,
    weights: &Array<F, IxDyn>,
    d_in: usize,
    d_out: usize,
) -> Result<Array<F, IxDyn>> {
    let shape = input.shape();
    let batch = shape[0];
    let seq = shape[1];
    let mut out = Array::zeros(IxDyn(&[batch, seq, d_out]));
    for b in 0..batch {
        for s in 0..seq {
            for o in 0..d_out {
                let mut acc = F::zero();
                for i in 0..d_in {
                    acc += input[[b, s, i]] * weights[[i, o]];
                }
                out[[b, s, o]] = acc;
            }
        }
    }
    Ok(out)
}

// ===========================================================================
// 1. Relative Positional Attention (Shaw et al.)
// ===========================================================================

/// Configuration for relative positional attention.
#[derive(Debug, Clone)]
pub struct RelativeAttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Maximum relative position (positions are clipped to [-max_rel, max_rel]).
    pub max_relative_position: usize,
    /// Whether to use causal masking.
    pub causal: bool,
}

/// Relative Positional Attention (Shaw et al., 2018).
///
/// Augments scaled dot-product attention with learnable relative position
/// embeddings that are added to the attention logits. The position vocabulary
/// is `[-max_rel, ..., 0, ..., max_rel]`, i.e. size `2 * max_rel + 1`.
#[derive(Debug)]
pub struct RelativePositionalAttention<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: RelativeAttentionConfig,
    w_query: Array<F, IxDyn>,
    w_key: Array<F, IxDyn>,
    w_value: Array<F, IxDyn>,
    w_output: Array<F, IxDyn>,
    /// Relative position key embeddings [vocab, head_dim]
    rel_key_emb: Array<F, IxDyn>,
    /// Relative position value embeddings [vocab, head_dim]
    rel_val_emb: Array<F, IxDyn>,
    scale: F,
    dw_query: Arc<RwLock<Array<F, IxDyn>>>,
    dw_key: Arc<RwLock<Array<F, IxDyn>>>,
    dw_value: Arc<RwLock<Array<F, IxDyn>>>,
    dw_output: Arc<RwLock<Array<F, IxDyn>>>,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static>
    RelativePositionalAttention<F>
{
    /// Create a new relative positional attention layer.
    pub fn new<R: Rng>(
        d_model: usize,
        config: RelativeAttentionConfig,
        rng: &mut R,
    ) -> Result<Self> {
        let nh = config.num_heads;
        let hd = config.head_dim;
        if d_model != nh * hd {
            return Err(NeuralError::InvalidArchitecture(format!(
                "d_model ({d_model}) must equal num_heads * head_dim ({} * {hd})",
                nh
            )));
        }

        let mk_weight = |rng: &mut R| -> Result<Array<F, IxDyn>> {
            let data = xavier_init(d_model, d_model, d_model * d_model, rng)?;
            Array::from_shape_vec(IxDyn(&[d_model, d_model]), data)
                .map_err(|e| NeuralError::InvalidArchitecture(format!("weight shape: {e}")))
        };

        let vocab = 2 * config.max_relative_position + 1;
        let rel_data_k = xavier_init(vocab, hd, vocab * hd, rng)?;
        let rel_data_v = xavier_init(vocab, hd, vocab * hd, rng)?;
        let rel_key_emb = Array::from_shape_vec(IxDyn(&[vocab, hd]), rel_data_k)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("rel_key shape: {e}")))?;
        let rel_val_emb = Array::from_shape_vec(IxDyn(&[vocab, hd]), rel_data_v)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("rel_val shape: {e}")))?;

        let scale = F::from(1.0 / (hd as f64).sqrt())
            .ok_or_else(|| NeuralError::InvalidArchitecture("scale convert".into()))?;

        let zeros = Array::zeros(IxDyn(&[d_model, d_model]));

        Ok(Self {
            d_model,
            config,
            w_query: mk_weight(rng)?,
            w_key: mk_weight(rng)?,
            w_value: mk_weight(rng)?,
            w_output: mk_weight(rng)?,
            rel_key_emb,
            rel_val_emb,
            scale,
            dw_query: Arc::new(RwLock::new(zeros.clone())),
            dw_key: Arc::new(RwLock::new(zeros.clone())),
            dw_value: Arc::new(RwLock::new(zeros.clone())),
            dw_output: Arc::new(RwLock::new(zeros)),
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Clip relative position to [-max_rel, max_rel] and shift to index.
    fn rel_idx(&self, qi: usize, kj: usize) -> usize {
        let diff = kj as isize - qi as isize;
        let max_r = self.config.max_relative_position as isize;
        let clipped = diff.clamp(-max_r, max_r);
        (clipped + max_r) as usize
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for RelativePositionalAttention<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 3D input, got {}D",
                shape.len()
            )));
        }
        let (batch, seq, dm) = (shape[0], shape[1], shape[2]);
        if dm != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "d_model mismatch: expected {}, got {dm}",
                self.d_model
            )));
        }

        let nh = self.config.num_heads;
        let hd = self.config.head_dim;

        let q = linear_project(input, &self.w_query, dm, dm)?;
        let k = linear_project(input, &self.w_key, dm, dm)?;
        let v = linear_project(input, &self.w_value, dm, dm)?;

        // Reshape to [batch, seq, nh, hd]
        let q = q
            .into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape q: {e}")))?;
        let k = k
            .into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape k: {e}")))?;
        let v = v
            .into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape v: {e}")))?;

        // Attention scores [batch, nh, seq_q, seq_k]
        let mut scores = Array::zeros(IxDyn(&[batch, nh, seq, seq]));
        for b in 0..batch {
            for h in 0..nh {
                for i in 0..seq {
                    for j in 0..seq {
                        let mut dot = F::zero();
                        let ridx = self.rel_idx(i, j);
                        for d in 0..hd {
                            dot +=
                                q[[b, i, h, d]] * (k[[b, j, h, d]] + self.rel_key_emb[[ridx, d]]);
                        }
                        scores[[b, h, i, j]] = dot * self.scale;
                    }
                }
            }
        }

        // Causal mask
        if self.config.causal {
            let neg_inf = F::neg_infinity();
            for b in 0..batch {
                for h in 0..nh {
                    for i in 0..seq {
                        for j in (i + 1)..seq {
                            scores[[b, h, i, j]] = neg_inf;
                        }
                    }
                }
            }
        }

        let attn = softmax_last_dim(&scores);

        // Weighted sum with relative value embeddings
        let mut attended = Array::zeros(IxDyn(&[batch, seq, nh, hd]));
        for b in 0..batch {
            for i in 0..seq {
                for h in 0..nh {
                    for d in 0..hd {
                        let mut sum = F::zero();
                        for j in 0..seq {
                            let ridx = self.rel_idx(i, j);
                            sum += attn[[b, h, i, j]]
                                * (v[[b, j, h, d]] + self.rel_val_emb[[ridx, d]]);
                        }
                        attended[[b, i, h, d]] = sum;
                    }
                }
            }
        }

        let concat = attended
            .into_shape_with_order(IxDyn(&[batch, seq, dm]))
            .map_err(|e| NeuralError::InferenceError(format!("concat: {e}")))?;
        linear_project(&concat, &self.w_output, dm, dm)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, lr: F) -> Result<()> {
        macro_rules! upd {
            ($w:expr, $dw:expr) => {
                if let Ok(dw) = $dw.read() {
                    $w = &$w - &(&*dw * lr);
                }
            };
        }
        upd!(self.w_query, self.dw_query);
        upd!(self.w_key, self.dw_key);
        upd!(self.w_value, self.dw_value);
        upd!(self.w_output, self.dw_output);
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
            self.rel_key_emb.clone(),
            self.rel_val_emb.clone(),
        ]
    }

    fn set_params(&mut self, params: &[Array<F, IxDyn>]) -> Result<()> {
        if params.len() >= 6 {
            self.w_query = params[0].clone();
            self.w_key = params[1].clone();
            self.w_value = params[2].clone();
            self.w_output = params[3].clone();
            self.rel_key_emb = params[4].clone();
            self.rel_val_emb = params[5].clone();
        }
        Ok(())
    }

    fn set_training(&mut self, t: bool) {
        self.training = t;
    }
    fn is_training(&self) -> bool {
        self.training
    }
    fn layer_type(&self) -> &str {
        "RelativePositionalAttention"
    }
    fn parameter_count(&self) -> usize {
        4 * self.d_model * self.d_model
            + 2 * (2 * self.config.max_relative_position + 1) * self.config.head_dim
    }
}

// ===========================================================================
// 2. Rotary Position Embedding (RoPE) Attention
// ===========================================================================

/// Configuration for RoPE attention.
#[derive(Debug, Clone)]
pub struct RoPEAttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head (must be even for rotation pairs).
    pub head_dim: usize,
    /// Base for the rotation frequency (default 10000).
    pub rope_base: f64,
    /// Whether to apply causal masking.
    pub causal: bool,
}

/// Multi-head attention with Rotary Position Embeddings (Su et al., 2021).
///
/// RoPE encodes absolute position through rotation matrices applied to
/// query and key vectors, naturally yielding relative position information
/// in the dot product.
#[derive(Debug)]
pub struct RoPEAttention<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: RoPEAttentionConfig,
    w_query: Array<F, IxDyn>,
    w_key: Array<F, IxDyn>,
    w_value: Array<F, IxDyn>,
    w_output: Array<F, IxDyn>,
    scale: F,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> RoPEAttention<F> {
    /// Create a new RoPE attention layer.
    pub fn new<R: Rng>(d_model: usize, config: RoPEAttentionConfig, rng: &mut R) -> Result<Self> {
        let nh = config.num_heads;
        let hd = config.head_dim;
        if d_model != nh * hd {
            return Err(NeuralError::InvalidArchitecture(format!(
                "d_model ({d_model}) != num_heads * head_dim ({nh} * {hd})"
            )));
        }
        if hd % 2 != 0 {
            return Err(NeuralError::InvalidArchitecture(
                "head_dim must be even for RoPE rotations".into(),
            ));
        }

        let mk = |rng: &mut R| -> Result<Array<F, IxDyn>> {
            let data = xavier_init(d_model, d_model, d_model * d_model, rng)?;
            Array::from_shape_vec(IxDyn(&[d_model, d_model]), data)
                .map_err(|e| NeuralError::InvalidArchitecture(format!("{e}")))
        };

        let scale = F::from(1.0 / (hd as f64).sqrt())
            .ok_or_else(|| NeuralError::InvalidArchitecture("scale".into()))?;

        Ok(Self {
            d_model,
            config,
            w_query: mk(rng)?,
            w_key: mk(rng)?,
            w_value: mk(rng)?,
            w_output: mk(rng)?,
            scale,
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Apply RoPE rotation to a vector in-place.
    /// For each pair (x_{2i}, x_{2i+1}), apply 2D rotation by angle
    /// `pos / base^{2i / head_dim}`.
    fn apply_rope(vec: &mut [F], pos: usize, head_dim: usize, base: f64) {
        let half = head_dim / 2;
        for i in 0..half {
            let freq = 1.0 / base.powf(2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * freq;
            let cos_a = F::from(angle.cos()).unwrap_or(F::one());
            let sin_a = F::from(angle.sin()).unwrap_or(F::zero());
            let x0 = vec[2 * i];
            let x1 = vec[2 * i + 1];
            vec[2 * i] = x0 * cos_a - x1 * sin_a;
            vec[2 * i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for RoPEAttention<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 3D, got {}D",
                shape.len()
            )));
        }
        let (batch, seq, dm) = (shape[0], shape[1], shape[2]);
        if dm != self.d_model {
            return Err(NeuralError::InferenceError("d_model mismatch".into()));
        }

        let nh = self.config.num_heads;
        let hd = self.config.head_dim;

        let q = linear_project(input, &self.w_query, dm, dm)?;
        let k = linear_project(input, &self.w_key, dm, dm)?;
        let v = linear_project(input, &self.w_value, dm, dm)?;

        let mut q = q
            .into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("{e}")))?;
        let mut k = k
            .into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("{e}")))?;
        let v = v
            .into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("{e}")))?;

        // Apply RoPE to queries and keys
        let base = self.config.rope_base;
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..nh {
                    let mut qvec: Vec<F> = (0..hd).map(|d| q[[b, s, h, d]]).collect();
                    let mut kvec: Vec<F> = (0..hd).map(|d| k[[b, s, h, d]]).collect();
                    Self::apply_rope(&mut qvec, s, hd, base);
                    Self::apply_rope(&mut kvec, s, hd, base);
                    for d in 0..hd {
                        q[[b, s, h, d]] = qvec[d];
                        k[[b, s, h, d]] = kvec[d];
                    }
                }
            }
        }

        // Scaled dot-product attention
        let mut scores = Array::zeros(IxDyn(&[batch, nh, seq, seq]));
        for b in 0..batch {
            for h in 0..nh {
                for i in 0..seq {
                    for j in 0..seq {
                        let mut dot = F::zero();
                        for d in 0..hd {
                            dot += q[[b, i, h, d]] * k[[b, j, h, d]];
                        }
                        scores[[b, h, i, j]] = dot * self.scale;
                    }
                }
            }
        }

        if self.config.causal {
            let neg_inf = F::neg_infinity();
            for b in 0..batch {
                for h in 0..nh {
                    for i in 0..seq {
                        for j in (i + 1)..seq {
                            scores[[b, h, i, j]] = neg_inf;
                        }
                    }
                }
            }
        }

        let attn = softmax_last_dim(&scores);

        let mut attended = Array::zeros(IxDyn(&[batch, seq, nh, hd]));
        for b in 0..batch {
            for i in 0..seq {
                for h in 0..nh {
                    for d in 0..hd {
                        let mut s = F::zero();
                        for j in 0..seq {
                            s += attn[[b, h, i, j]] * v[[b, j, h, d]];
                        }
                        attended[[b, i, h, d]] = s;
                    }
                }
            }
        }

        let concat = attended
            .into_shape_with_order(IxDyn(&[batch, seq, dm]))
            .map_err(|e| NeuralError::InferenceError(format!("{e}")))?;
        linear_project(&concat, &self.w_output, dm, dm)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad.clone())
    }
    fn update(&mut self, _lr: F) -> Result<()> {
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
    fn set_params(&mut self, p: &[Array<F, IxDyn>]) -> Result<()> {
        if p.len() >= 4 {
            self.w_query = p[0].clone();
            self.w_key = p[1].clone();
            self.w_value = p[2].clone();
            self.w_output = p[3].clone();
        }
        Ok(())
    }
    fn set_training(&mut self, t: bool) {
        self.training = t;
    }
    fn is_training(&self) -> bool {
        self.training
    }
    fn layer_type(&self) -> &str {
        "RoPEAttention"
    }
    fn parameter_count(&self) -> usize {
        4 * self.d_model * self.d_model
    }
}

// ===========================================================================
// 3. Grouped Query Attention (GQA)
// ===========================================================================

/// Configuration for Grouped Query Attention.
#[derive(Debug, Clone)]
pub struct GQAConfig {
    /// Total number of query heads.
    pub num_query_heads: usize,
    /// Number of key-value head groups (must divide num_query_heads).
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Whether to apply causal masking.
    pub causal: bool,
}

/// Grouped Query Attention (Ainslie et al., 2023).
///
/// GQA reduces memory and compute by sharing each key-value head across
/// multiple query heads. When `num_kv_heads == 1` this is Multi-Query
/// Attention; when `num_kv_heads == num_query_heads` this is standard MHA.
#[derive(Debug)]
pub struct GroupedQueryAttention<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: GQAConfig,
    w_query: Array<F, IxDyn>,
    /// Key projection [d_model, num_kv_heads * head_dim]
    w_key: Array<F, IxDyn>,
    /// Value projection [d_model, num_kv_heads * head_dim]
    w_value: Array<F, IxDyn>,
    w_output: Array<F, IxDyn>,
    scale: F,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static>
    GroupedQueryAttention<F>
{
    /// Create a new GQA layer.
    pub fn new<R: Rng>(d_model: usize, config: GQAConfig, rng: &mut R) -> Result<Self> {
        let nq = config.num_query_heads;
        let nkv = config.num_kv_heads;
        let hd = config.head_dim;

        if d_model != nq * hd {
            return Err(NeuralError::InvalidArchitecture(format!(
                "d_model ({d_model}) != num_query_heads * head_dim ({nq} * {hd})"
            )));
        }
        if nq % nkv != 0 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "num_query_heads ({nq}) must be divisible by num_kv_heads ({nkv})"
            )));
        }

        let kv_dim = nkv * hd;
        let mk = |din: usize, dout: usize, rng: &mut R| -> Result<Array<F, IxDyn>> {
            let data = xavier_init(din, dout, din * dout, rng)?;
            Array::from_shape_vec(IxDyn(&[din, dout]), data)
                .map_err(|e| NeuralError::InvalidArchitecture(format!("{e}")))
        };

        let scale = F::from(1.0 / (hd as f64).sqrt())
            .ok_or_else(|| NeuralError::InvalidArchitecture("scale".into()))?;

        Ok(Self {
            d_model,
            config,
            w_query: mk(d_model, d_model, rng)?,
            w_key: mk(d_model, kv_dim, rng)?,
            w_value: mk(d_model, kv_dim, rng)?,
            w_output: mk(d_model, d_model, rng)?,
            scale,
            training: true,
            _phantom: PhantomData,
        })
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for GroupedQueryAttention<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 3D, got {}D",
                shape.len()
            )));
        }
        let (batch, seq, dm) = (shape[0], shape[1], shape[2]);
        if dm != self.d_model {
            return Err(NeuralError::InferenceError("d_model mismatch".into()));
        }

        let nq = self.config.num_query_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let kv_dim = nkv * hd;
        let groups = nq / nkv; // query heads per kv group

        let q = linear_project(input, &self.w_query, dm, dm)?;
        let k = linear_project(input, &self.w_key, dm, kv_dim)?;
        let v = linear_project(input, &self.w_value, dm, kv_dim)?;

        let q = q
            .into_shape_with_order(IxDyn(&[batch, seq, nq, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("{e}")))?;
        let k = k
            .into_shape_with_order(IxDyn(&[batch, seq, nkv, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("{e}")))?;
        let v = v
            .into_shape_with_order(IxDyn(&[batch, seq, nkv, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("{e}")))?;

        // scores [batch, nq, seq_q, seq_k]
        let mut scores = Array::zeros(IxDyn(&[batch, nq, seq, seq]));
        for b in 0..batch {
            for qh in 0..nq {
                let kv_group = qh / groups;
                for i in 0..seq {
                    for j in 0..seq {
                        let mut dot = F::zero();
                        for d in 0..hd {
                            dot += q[[b, i, qh, d]] * k[[b, j, kv_group, d]];
                        }
                        scores[[b, qh, i, j]] = dot * self.scale;
                    }
                }
            }
        }

        if self.config.causal {
            let neg_inf = F::neg_infinity();
            for b in 0..batch {
                for h in 0..nq {
                    for i in 0..seq {
                        for j in (i + 1)..seq {
                            scores[[b, h, i, j]] = neg_inf;
                        }
                    }
                }
            }
        }

        let attn = softmax_last_dim(&scores);

        let mut attended = Array::zeros(IxDyn(&[batch, seq, nq, hd]));
        for b in 0..batch {
            for i in 0..seq {
                for qh in 0..nq {
                    let kv_group = qh / groups;
                    for d in 0..hd {
                        let mut s = F::zero();
                        for j in 0..seq {
                            s += attn[[b, qh, i, j]] * v[[b, j, kv_group, d]];
                        }
                        attended[[b, i, qh, d]] = s;
                    }
                }
            }
        }

        let concat = attended
            .into_shape_with_order(IxDyn(&[batch, seq, dm]))
            .map_err(|e| NeuralError::InferenceError(format!("{e}")))?;
        linear_project(&concat, &self.w_output, dm, dm)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad.clone())
    }
    fn update(&mut self, _lr: F) -> Result<()> {
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
    fn set_params(&mut self, p: &[Array<F, IxDyn>]) -> Result<()> {
        if p.len() >= 4 {
            self.w_query = p[0].clone();
            self.w_key = p[1].clone();
            self.w_value = p[2].clone();
            self.w_output = p[3].clone();
        }
        Ok(())
    }
    fn set_training(&mut self, t: bool) {
        self.training = t;
    }
    fn is_training(&self) -> bool {
        self.training
    }
    fn layer_type(&self) -> &str {
        "GroupedQueryAttention"
    }
    fn parameter_count(&self) -> usize {
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        self.d_model * self.d_model   // w_query
            + self.d_model * kv_dim   // w_key
            + self.d_model * kv_dim   // w_value
            + self.d_model * self.d_model // w_output
    }
}

// ===========================================================================
// 4. Sliding Window Attention
// ===========================================================================

/// Configuration for sliding window attention.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Window size (each token attends to `window_size` tokens on each side).
    pub window_size: usize,
    /// Whether to also apply causal masking within the window.
    pub causal: bool,
}

/// Sliding Window Attention (Beltagy et al., 2020).
///
/// Each query token attends only to keys within a fixed-size window,
/// giving O(n * w) complexity instead of O(n^2).
#[derive(Debug)]
pub struct SlidingWindowAttention<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: SlidingWindowConfig,
    w_query: Array<F, IxDyn>,
    w_key: Array<F, IxDyn>,
    w_value: Array<F, IxDyn>,
    w_output: Array<F, IxDyn>,
    scale: F,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static>
    SlidingWindowAttention<F>
{
    /// Create a new sliding window attention layer.
    pub fn new<R: Rng>(d_model: usize, config: SlidingWindowConfig, rng: &mut R) -> Result<Self> {
        let nh = config.num_heads;
        let hd = config.head_dim;
        if d_model != nh * hd {
            return Err(NeuralError::InvalidArchitecture(format!(
                "d_model ({d_model}) != num_heads * head_dim ({nh} * {hd})"
            )));
        }
        if config.window_size == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "window_size must be > 0".into(),
            ));
        }

        let mk = |rng: &mut R| -> Result<Array<F, IxDyn>> {
            let data = xavier_init(d_model, d_model, d_model * d_model, rng)?;
            Array::from_shape_vec(IxDyn(&[d_model, d_model]), data)
                .map_err(|e| NeuralError::InvalidArchitecture(format!("{e}")))
        };

        let scale = F::from(1.0 / (hd as f64).sqrt())
            .ok_or_else(|| NeuralError::InvalidArchitecture("scale".into()))?;

        Ok(Self {
            d_model,
            config,
            w_query: mk(rng)?,
            w_key: mk(rng)?,
            w_value: mk(rng)?,
            w_output: mk(rng)?,
            scale,
            training: true,
            _phantom: PhantomData,
        })
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for SlidingWindowAttention<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 3D, got {}D",
                shape.len()
            )));
        }
        let (batch, seq, dm) = (shape[0], shape[1], shape[2]);
        if dm != self.d_model {
            return Err(NeuralError::InferenceError("d_model mismatch".into()));
        }

        let nh = self.config.num_heads;
        let hd = self.config.head_dim;
        let w = self.config.window_size;

        let q = linear_project(input, &self.w_query, dm, dm)?;
        let k = linear_project(input, &self.w_key, dm, dm)?;
        let v = linear_project(input, &self.w_value, dm, dm)?;

        let q = q
            .into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("{e}")))?;
        let k = k
            .into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("{e}")))?;
        let v = v
            .into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("{e}")))?;

        let neg_inf = F::neg_infinity();

        // For each query position, only attend within the window
        let mut attended = Array::zeros(IxDyn(&[batch, seq, nh, hd]));
        for b in 0..batch {
            for h in 0..nh {
                for i in 0..seq {
                    let j_start = if i >= w { i - w } else { 0 };
                    let j_end = if self.config.causal {
                        i + 1
                    } else {
                        (i + w + 1).min(seq)
                    };

                    // Compute scores for the window
                    let window_len = j_end - j_start;
                    let mut window_scores = vec![neg_inf; window_len];
                    for (wi, j) in (j_start..j_end).enumerate() {
                        let mut dot = F::zero();
                        for d in 0..hd {
                            dot += q[[b, i, h, d]] * k[[b, j, h, d]];
                        }
                        window_scores[wi] = dot * self.scale;
                    }

                    // Softmax over window
                    let mut max_val = F::neg_infinity();
                    for &s in &window_scores {
                        if s > max_val {
                            max_val = s;
                        }
                    }
                    let mut sum = F::zero();
                    let mut exp_vals = Vec::with_capacity(window_len);
                    for &s in &window_scores {
                        let e = (s - max_val).exp();
                        exp_vals.push(e);
                        sum += e;
                    }

                    // Weighted sum
                    for (wi, j) in (j_start..j_end).enumerate() {
                        let w_attn = exp_vals[wi] / sum;
                        for d in 0..hd {
                            attended[[b, i, h, d]] += w_attn * v[[b, j, h, d]];
                        }
                    }
                }
            }
        }

        let concat = attended
            .into_shape_with_order(IxDyn(&[batch, seq, dm]))
            .map_err(|e| NeuralError::InferenceError(format!("{e}")))?;
        linear_project(&concat, &self.w_output, dm, dm)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad.clone())
    }
    fn update(&mut self, _lr: F) -> Result<()> {
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
    fn set_params(&mut self, p: &[Array<F, IxDyn>]) -> Result<()> {
        if p.len() >= 4 {
            self.w_query = p[0].clone();
            self.w_key = p[1].clone();
            self.w_value = p[2].clone();
            self.w_output = p[3].clone();
        }
        Ok(())
    }
    fn set_training(&mut self, t: bool) {
        self.training = t;
    }
    fn is_training(&self) -> bool {
        self.training
    }
    fn layer_type(&self) -> &str {
        "SlidingWindowAttention"
    }
    fn parameter_count(&self) -> usize {
        4 * self.d_model * self.d_model
    }
}

// ===========================================================================
// 5. Linear Attention
// ===========================================================================

/// Type of kernel feature map for linear attention.
#[derive(Debug, Clone, Copy)]
pub enum KernelFeatureMap {
    /// ELU(x) + 1 (Katharopoulos et al.)
    Elu,
    /// ReLU feature map
    Relu,
    /// Identity (dot-product, no non-linearity)
    Identity,
}

/// Configuration for linear attention.
#[derive(Debug, Clone)]
pub struct LinearAttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Kernel feature map type.
    pub feature_map: KernelFeatureMap,
    /// Small epsilon for numerical stability.
    pub eps: f64,
}

/// Linear Attention (Katharopoulos et al., 2020).
///
/// Replaces `softmax(Q K^T) V` with `phi(Q) (phi(K)^T V)` where `phi` is
/// a kernel feature map, achieving O(n * d^2) complexity.
#[derive(Debug)]
pub struct LinearAttention<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: LinearAttentionConfig,
    w_query: Array<F, IxDyn>,
    w_key: Array<F, IxDyn>,
    w_value: Array<F, IxDyn>,
    w_output: Array<F, IxDyn>,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> LinearAttention<F> {
    /// Create a new linear attention layer.
    pub fn new<R: Rng>(d_model: usize, config: LinearAttentionConfig, rng: &mut R) -> Result<Self> {
        let nh = config.num_heads;
        let hd = config.head_dim;
        if d_model != nh * hd {
            return Err(NeuralError::InvalidArchitecture(format!(
                "d_model ({d_model}) != num_heads * head_dim ({nh} * {hd})"
            )));
        }

        let mk = |rng: &mut R| -> Result<Array<F, IxDyn>> {
            let data = xavier_init(d_model, d_model, d_model * d_model, rng)?;
            Array::from_shape_vec(IxDyn(&[d_model, d_model]), data)
                .map_err(|e| NeuralError::InvalidArchitecture(format!("{e}")))
        };

        Ok(Self {
            d_model,
            config,
            w_query: mk(rng)?,
            w_key: mk(rng)?,
            w_value: mk(rng)?,
            w_output: mk(rng)?,
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Apply the kernel feature map element-wise.
    fn apply_feature_map(&self, val: F) -> F {
        match self.config.feature_map {
            KernelFeatureMap::Elu => {
                // elu(x) + 1
                if val > F::zero() {
                    val + F::one()
                } else {
                    val.exp() // exp(x) for x <= 0 gives elu(x)+1
                }
            }
            KernelFeatureMap::Relu => {
                if val > F::zero() {
                    val
                } else {
                    F::zero()
                }
            }
            KernelFeatureMap::Identity => val,
        }
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for LinearAttention<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 3D, got {}D",
                shape.len()
            )));
        }
        let (batch, seq, dm) = (shape[0], shape[1], shape[2]);
        if dm != self.d_model {
            return Err(NeuralError::InferenceError("d_model mismatch".into()));
        }

        let nh = self.config.num_heads;
        let hd = self.config.head_dim;
        let eps = F::from(self.config.eps).unwrap_or(F::from(1e-6).unwrap_or(F::zero()));

        let q = linear_project(input, &self.w_query, dm, dm)?;
        let k = linear_project(input, &self.w_key, dm, dm)?;
        let v = linear_project(input, &self.w_value, dm, dm)?;

        let q = q
            .into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("{e}")))?;
        let k = k
            .into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("{e}")))?;
        let v = v
            .into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("{e}")))?;

        // Apply feature maps
        let mut phi_q = Array::zeros(q.raw_dim());
        let mut phi_k = Array::zeros(k.raw_dim());
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..nh {
                    for d in 0..hd {
                        phi_q[[b, s, h, d]] = self.apply_feature_map(q[[b, s, h, d]]);
                        phi_k[[b, s, h, d]] = self.apply_feature_map(k[[b, s, h, d]]);
                    }
                }
            }
        }

        // Linear attention: phi(Q) @ (phi(K)^T @ V)
        // KV: [batch, nh, hd_k, hd_v]  (accumulated across seq)
        // Then output_i = (phi_q_i @ KV) / (phi_q_i @ K_sum)
        let mut attended = Array::zeros(IxDyn(&[batch, seq, nh, hd]));

        for b in 0..batch {
            for h in 0..nh {
                // Accumulate K^T V  (hd x hd)
                let mut kv = vec![F::zero(); hd * hd];
                // Accumulate K_sum  (hd)
                let mut k_sum = vec![F::zero(); hd];

                for j in 0..seq {
                    for dk in 0..hd {
                        k_sum[dk] += phi_k[[b, j, h, dk]];
                        for dv in 0..hd {
                            kv[dk * hd + dv] += phi_k[[b, j, h, dk]] * v[[b, j, h, dv]];
                        }
                    }
                }

                for i in 0..seq {
                    // numerator: phi_q_i @ KV -> [hd]
                    // denominator: phi_q_i . K_sum -> scalar
                    let mut denom = F::zero();
                    for dk in 0..hd {
                        denom += phi_q[[b, i, h, dk]] * k_sum[dk];
                    }
                    let norm = if denom.abs() < eps { eps } else { denom };

                    for dv in 0..hd {
                        let mut num = F::zero();
                        for dk in 0..hd {
                            num += phi_q[[b, i, h, dk]] * kv[dk * hd + dv];
                        }
                        attended[[b, i, h, dv]] = num / norm;
                    }
                }
            }
        }

        let concat = attended
            .into_shape_with_order(IxDyn(&[batch, seq, dm]))
            .map_err(|e| NeuralError::InferenceError(format!("{e}")))?;
        linear_project(&concat, &self.w_output, dm, dm)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad.clone())
    }
    fn update(&mut self, _lr: F) -> Result<()> {
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
    fn set_params(&mut self, p: &[Array<F, IxDyn>]) -> Result<()> {
        if p.len() >= 4 {
            self.w_query = p[0].clone();
            self.w_key = p[1].clone();
            self.w_value = p[2].clone();
            self.w_output = p[3].clone();
        }
        Ok(())
    }
    fn set_training(&mut self, t: bool) {
        self.training = t;
    }
    fn is_training(&self) -> bool {
        self.training
    }
    fn layer_type(&self) -> &str {
        "LinearAttention"
    }
    fn parameter_count(&self) -> usize {
        4 * self.d_model * self.d_model
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;
    use scirs2_core::random::rng;

    // ------- RelativePositionalAttention -------

    #[test]
    fn test_relative_attention_creation() {
        let mut r = rng();
        let config = RelativeAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            max_relative_position: 4,
            causal: false,
        };
        let layer =
            RelativePositionalAttention::<f64>::new(16, config, &mut r).expect("creation failed");
        assert_eq!(layer.layer_type(), "RelativePositionalAttention");
    }

    #[test]
    fn test_relative_attention_forward() {
        let mut r = rng();
        let config = RelativeAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            max_relative_position: 4,
            causal: false,
        };
        let layer =
            RelativePositionalAttention::<f64>::new(16, config, &mut r).expect("creation failed");
        let input = Array3::<f64>::from_elem((2, 5, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 5, 16]);
    }

    #[test]
    fn test_relative_attention_causal() {
        let mut r = rng();
        let config = RelativeAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            max_relative_position: 3,
            causal: true,
        };
        let layer =
            RelativePositionalAttention::<f64>::new(16, config, &mut r).expect("creation failed");
        let input = Array3::<f64>::from_elem((1, 4, 16), 0.2).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 4, 16]);
    }

    #[test]
    fn test_relative_attention_params() {
        let mut r = rng();
        let config = RelativeAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            max_relative_position: 4,
            causal: false,
        };
        let layer =
            RelativePositionalAttention::<f64>::new(16, config, &mut r).expect("creation failed");
        assert_eq!(layer.params().len(), 6); // 4 projections + 2 rel embeddings
    }

    // ------- RoPE Attention -------

    #[test]
    fn test_rope_attention_creation() {
        let mut r = rng();
        let config = RoPEAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            rope_base: 10000.0,
            causal: false,
        };
        let layer = RoPEAttention::<f64>::new(16, config, &mut r).expect("creation failed");
        assert_eq!(layer.layer_type(), "RoPEAttention");
    }

    #[test]
    fn test_rope_attention_forward() {
        let mut r = rng();
        let config = RoPEAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            rope_base: 10000.0,
            causal: false,
        };
        let layer = RoPEAttention::<f64>::new(16, config, &mut r).expect("creation failed");
        let input = Array3::<f64>::from_elem((2, 6, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 6, 16]);
    }

    #[test]
    fn test_rope_even_head_dim_required() {
        let mut r = rng();
        let config = RoPEAttentionConfig {
            num_heads: 3,
            head_dim: 5,
            rope_base: 10000.0,
            causal: false,
        };
        let result = RoPEAttention::<f64>::new(15, config, &mut r);
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_causal() {
        let mut r = rng();
        let config = RoPEAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            rope_base: 10000.0,
            causal: true,
        };
        let layer = RoPEAttention::<f64>::new(16, config, &mut r).expect("creation failed");
        let input = Array3::<f64>::from_elem((1, 4, 16), 0.3).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 4, 16]);
    }

    // ------- Grouped Query Attention -------

    #[test]
    fn test_gqa_creation() {
        let mut r = rng();
        let config = GQAConfig {
            num_query_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            causal: false,
        };
        let layer = GroupedQueryAttention::<f64>::new(32, config, &mut r).expect("creation failed");
        assert_eq!(layer.layer_type(), "GroupedQueryAttention");
    }

    #[test]
    fn test_gqa_forward() {
        let mut r = rng();
        let config = GQAConfig {
            num_query_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            causal: false,
        };
        let layer = GroupedQueryAttention::<f64>::new(32, config, &mut r).expect("creation failed");
        let input = Array3::<f64>::from_elem((2, 5, 32), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 5, 32]);
    }

    #[test]
    fn test_gqa_single_kv_head() {
        let mut r = rng();
        let config = GQAConfig {
            num_query_heads: 4,
            num_kv_heads: 1,
            head_dim: 8,
            causal: true,
        };
        let layer = GroupedQueryAttention::<f64>::new(32, config, &mut r).expect("creation failed");
        let input = Array3::<f64>::from_elem((1, 3, 32), 0.2).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 3, 32]);
    }

    #[test]
    fn test_gqa_invalid_divisibility() {
        let mut r = rng();
        let config = GQAConfig {
            num_query_heads: 5,
            num_kv_heads: 3,
            head_dim: 8,
            causal: false,
        };
        let result = GroupedQueryAttention::<f64>::new(40, config, &mut r);
        assert!(result.is_err());
    }

    // ------- Sliding Window Attention -------

    #[test]
    fn test_sliding_window_creation() {
        let mut r = rng();
        let config = SlidingWindowConfig {
            num_heads: 2,
            head_dim: 8,
            window_size: 3,
            causal: false,
        };
        let layer =
            SlidingWindowAttention::<f64>::new(16, config, &mut r).expect("creation failed");
        assert_eq!(layer.layer_type(), "SlidingWindowAttention");
    }

    #[test]
    fn test_sliding_window_forward() {
        let mut r = rng();
        let config = SlidingWindowConfig {
            num_heads: 2,
            head_dim: 8,
            window_size: 2,
            causal: false,
        };
        let layer =
            SlidingWindowAttention::<f64>::new(16, config, &mut r).expect("creation failed");
        let input = Array3::<f64>::from_elem((2, 6, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 6, 16]);
    }

    #[test]
    fn test_sliding_window_causal() {
        let mut r = rng();
        let config = SlidingWindowConfig {
            num_heads: 2,
            head_dim: 8,
            window_size: 3,
            causal: true,
        };
        let layer =
            SlidingWindowAttention::<f64>::new(16, config, &mut r).expect("creation failed");
        let input = Array3::<f64>::from_elem((1, 8, 16), 0.2).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 8, 16]);
    }

    #[test]
    fn test_sliding_window_zero_size_error() {
        let mut r = rng();
        let config = SlidingWindowConfig {
            num_heads: 2,
            head_dim: 8,
            window_size: 0,
            causal: false,
        };
        let result = SlidingWindowAttention::<f64>::new(16, config, &mut r);
        assert!(result.is_err());
    }

    // ------- Linear Attention -------

    #[test]
    fn test_linear_attention_creation() {
        let mut r = rng();
        let config = LinearAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            feature_map: KernelFeatureMap::Elu,
            eps: 1e-6,
        };
        let layer = LinearAttention::<f64>::new(16, config, &mut r).expect("creation failed");
        assert_eq!(layer.layer_type(), "LinearAttention");
    }

    #[test]
    fn test_linear_attention_elu() {
        let mut r = rng();
        let config = LinearAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            feature_map: KernelFeatureMap::Elu,
            eps: 1e-6,
        };
        let layer = LinearAttention::<f64>::new(16, config, &mut r).expect("creation failed");
        let input = Array3::<f64>::from_elem((2, 5, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 5, 16]);
    }

    #[test]
    fn test_linear_attention_relu() {
        let mut r = rng();
        let config = LinearAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            feature_map: KernelFeatureMap::Relu,
            eps: 1e-6,
        };
        let layer = LinearAttention::<f64>::new(16, config, &mut r).expect("creation failed");
        let input = Array3::<f64>::from_elem((1, 4, 16), 0.5).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 4, 16]);
    }

    #[test]
    fn test_linear_attention_identity() {
        let mut r = rng();
        let config = LinearAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            feature_map: KernelFeatureMap::Identity,
            eps: 1e-6,
        };
        let layer = LinearAttention::<f64>::new(16, config, &mut r).expect("creation failed");
        let input = Array3::<f64>::from_elem((1, 3, 16), 0.2).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 3, 16]);
    }

    #[test]
    fn test_linear_attention_params() {
        let mut r = rng();
        let config = LinearAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            feature_map: KernelFeatureMap::Elu,
            eps: 1e-6,
        };
        let layer = LinearAttention::<f64>::new(16, config, &mut r).expect("creation failed");
        assert_eq!(layer.params().len(), 4);
        assert_eq!(layer.parameter_count(), 4 * 16 * 16);
    }
}
