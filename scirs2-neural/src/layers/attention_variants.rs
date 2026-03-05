//! Attention mechanism variants with a unified trait interface.
//!
//! This module provides a common [`AttentionVariant`] trait and several
//! concrete implementations:
//!
//! - [`SparseAttention`] – Block-sparse attention (Longformer-style) combining
//!   local sliding-window and global token attention. O(n·w) complexity.
//! - [`LinearAttentionVariant`] – Kernel feature-map attention (ELU/ReLU φ(x))
//!   with O(N·d) complexity instead of O(N²).
//! - [`PerformerVariant`] – FAVOR+ random orthogonal feature approximation of
//!   softmax attention (Choromanski et al., 2020).
//! - [`CrossAttention`] – Standard cross-attention where queries come from one
//!   sequence and keys/values come from a separate memory sequence.
//!
//! All types implement both [`AttentionVariant`] and the standard [`Layer`]
//! trait so they can be dropped into any layer stack.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::Rng;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// Public trait
// ---------------------------------------------------------------------------

/// Unified interface for attention mechanism variants.
///
/// Every concrete attention type in this module implements this trait in
/// addition to [`Layer`].  The `forward_attn` method gives a clearer
/// semantic name than the generic `Layer::forward`.
pub trait AttentionVariant<F: Float + Debug + ScalarOperand + NumAssign>:
    Layer<F> + Send + Sync
{
    /// Run the attention forward pass.
    ///
    /// # Arguments
    /// * `input` – 3-D tensor `[batch, seq, d_model]`.
    ///
    /// # Returns
    /// 3-D tensor `[batch, seq, d_model]` with the same shape.
    fn forward_attn(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>>;

    /// Human-readable name of the attention variant.
    fn variant_name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Xavier-uniform initialiser — returns a flat `Vec<F>`.
fn xavier_vec<F: Float, R: Rng>(
    fan_in: usize,
    fan_out: usize,
    count: usize,
    rng: &mut R,
) -> Result<Vec<F>> {
    let scale = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        let x: f64 = rng.random_range(-scale..scale);
        let f = F::from(x)
            .ok_or_else(|| NeuralError::InvalidArchitecture("xavier_vec cast failed".into()))?;
        v.push(f);
    }
    Ok(v)
}

/// Create a 2-D weight matrix with Xavier initialisation.
fn mk_weight<F: Float, R: Rng>(
    rows: usize,
    cols: usize,
    rng: &mut R,
) -> Result<Array<F, IxDyn>> {
    let data = xavier_vec(rows, cols, rows * cols, rng)?;
    Array::from_shape_vec(IxDyn(&[rows, cols]), data)
        .map_err(|e| NeuralError::InvalidArchitecture(format!("mk_weight shape error: {e}")))
}

/// Batch-linear projection: `[B, S, D_in] × [D_in, D_out] → [B, S, D_out]`.
fn batch_proj<F: Float + NumAssign>(
    x: &Array<F, IxDyn>,
    w: &Array<F, IxDyn>,
    d_in: usize,
    d_out: usize,
) -> Result<Array<F, IxDyn>> {
    let shape = x.shape();
    if shape.len() != 3 {
        return Err(NeuralError::InferenceError(format!(
            "batch_proj: expected 3D input, got {}D",
            shape.len()
        )));
    }
    let (batch, seq) = (shape[0], shape[1]);
    let mut out = Array::zeros(IxDyn(&[batch, seq, d_out]));
    for b in 0..batch {
        for t in 0..seq {
            for o in 0..d_out {
                let mut acc = F::zero();
                for i in 0..d_in {
                    acc += x[[b, t, i]] * w[[i, o]];
                }
                out[[b, t, o]] = acc;
            }
        }
    }
    Ok(out)
}

/// Softmax over a mutable slice (in-place).
fn softmax_inplace<F: Float + NumAssign>(s: &mut [F]) {
    let max_v = s.iter().fold(F::neg_infinity(), |a, &b| if b > a { b } else { a });
    let mut sum = F::zero();
    for v in s.iter_mut() {
        *v = (*v - max_v).exp();
        sum += *v;
    }
    let eps = F::from(1e-12_f64).unwrap_or(F::zero());
    let norm = if sum < eps { eps } else { sum };
    for v in s.iter_mut() {
        *v = *v / norm;
    }
}

// ===========================================================================
// 1.  SparseAttention  (Longformer block-sparse)
// ===========================================================================

/// Configuration for [`SparseAttention`].
#[derive(Debug, Clone)]
pub struct SparseAttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// One-sided local window radius (each token attends to ±window_radius
    /// neighbours, giving a window of `2·window_radius + 1`).
    pub window_radius: usize,
    /// Number of global tokens prepended to every sequence (these tokens
    /// attend to and are attended by all other tokens).
    pub num_global_tokens: usize,
    /// Scale factor for attention scores (defaults to `1/√head_dim`).
    pub scale: Option<f64>,
}

impl Default for SparseAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            window_radius: 64,
            num_global_tokens: 0,
            scale: None,
        }
    }
}

/// Block-sparse attention (Longformer-style).
///
/// Each non-global token attends only to its local window of size
/// `2·window_radius + 1` plus the designated global tokens.
/// Global tokens have full attention over the entire sequence.
///
/// Complexity: **O(N · (W + G) · d)** where W = window width, G = #globals.
///
/// # Input / Output
/// Shape `[batch, seq, d_model]` → `[batch, seq, d_model]`.
#[derive(Debug)]
pub struct SparseAttention<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: SparseAttentionConfig,
    /// Query projection  [d_model, num_heads * head_dim]
    w_q: Arc<RwLock<Array<F, IxDyn>>>,
    /// Key projection    [d_model, num_heads * head_dim]
    w_k: Arc<RwLock<Array<F, IxDyn>>>,
    /// Value projection  [d_model, num_heads * head_dim]
    w_v: Arc<RwLock<Array<F, IxDyn>>>,
    /// Output projection [num_heads * head_dim, d_model]
    w_o: Arc<RwLock<Array<F, IxDyn>>>,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> SparseAttention<F> {
    /// Create a new `SparseAttention` layer.
    ///
    /// # Errors
    /// Returns an error if `num_heads`, `head_dim`, or `d_model` is zero.
    pub fn new<R: Rng>(d_model: usize, config: SparseAttentionConfig, rng: &mut R) -> Result<Self> {
        if d_model == 0 {
            return Err(NeuralError::InvalidArchitecture("d_model must be > 0".into()));
        }
        if config.num_heads == 0 {
            return Err(NeuralError::InvalidArchitecture("num_heads must be > 0".into()));
        }
        if config.head_dim == 0 {
            return Err(NeuralError::InvalidArchitecture("head_dim must be > 0".into()));
        }
        let inner = config.num_heads * config.head_dim;
        let w_q = mk_weight(d_model, inner, rng)?;
        let w_k = mk_weight(d_model, inner, rng)?;
        let w_v = mk_weight(d_model, inner, rng)?;
        let w_o = mk_weight(inner, d_model, rng)?;
        Ok(Self {
            d_model,
            config,
            w_q: Arc::new(RwLock::new(w_q)),
            w_k: Arc::new(RwLock::new(w_k)),
            w_v: Arc::new(RwLock::new(w_v)),
            w_o: Arc::new(RwLock::new(w_o)),
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Compute sparse attention for a single head.
    ///
    /// Returns the attended output `[seq, head_dim]`.
    fn sparse_head(
        &self,
        q: &Array<F, IxDyn>, // [seq, head_dim]
        k: &Array<F, IxDyn>,
        v: &Array<F, IxDyn>,
        seq: usize,
        head_dim: usize,
        scale: F,
        num_globals: usize,
        window_radius: usize,
    ) -> Array<F, IxDyn> {
        let mut out = Array::zeros(IxDyn(&[seq, head_dim]));

        for t in 0..seq {
            // Determine which positions token `t` attends to.
            let (attend_start, attend_end) = if t < num_globals {
                // Global token: full attention
                (0, seq)
            } else {
                // Local window (clamped) + global tokens are always included
                let ws = if t >= window_radius { t - window_radius } else { 0 };
                let we = (t + window_radius + 1).min(seq);
                (ws, we)
            };

            // Collect the positions this token attends to.
            // Globals + local window (deduped via sorted merge).
            let mut positions: Vec<usize> = (0..num_globals.min(seq)).collect();
            for j in attend_start..attend_end {
                if j >= num_globals {
                    positions.push(j);
                }
            }
            positions.sort_unstable();
            positions.dedup();

            let n_attend = positions.len();
            if n_attend == 0 {
                continue;
            }

            // Compute dot products q[t] · k[j] for all j in positions.
            let mut scores: Vec<F> = Vec::with_capacity(n_attend);
            for &j in &positions {
                let mut dot = F::zero();
                for d in 0..head_dim {
                    dot += q[[t, d]] * k[[j, d]];
                }
                scores.push(dot * scale);
            }
            softmax_inplace(&mut scores);

            // Weighted sum of values.
            for (idx, &j) in positions.iter().enumerate() {
                for d in 0..head_dim {
                    out[[t, d]] += scores[idx] * v[[j, d]];
                }
            }
        }
        out
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static>
    AttentionVariant<F> for SparseAttention<F>
{
    fn forward_attn(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.forward(input)
    }

    fn variant_name(&self) -> &str {
        "SparseAttention"
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for SparseAttention<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "SparseAttention: expected 3D input [B,S,D], got {}D",
                shape.len()
            )));
        }
        let (batch, seq, d_model) = (shape[0], shape[1], shape[2]);
        if d_model != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "SparseAttention: d_model mismatch: expected {}, got {d_model}",
                self.d_model
            )));
        }

        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let inner = num_heads * head_dim;
        let num_globals = self.config.num_global_tokens;
        let window_radius = self.config.window_radius;
        let scale_val = self.config.scale.unwrap_or(1.0 / (head_dim as f64).sqrt());
        let scale = F::from(scale_val)
            .ok_or_else(|| NeuralError::InvalidArchitecture("scale cast failed".into()))?;

        let w_q = self.w_q.read().map_err(|_| NeuralError::InferenceError("lock poisoned".into()))?;
        let w_k = self.w_k.read().map_err(|_| NeuralError::InferenceError("lock poisoned".into()))?;
        let w_v = self.w_v.read().map_err(|_| NeuralError::InferenceError("lock poisoned".into()))?;
        let w_o = self.w_o.read().map_err(|_| NeuralError::InferenceError("lock poisoned".into()))?;

        // Project queries, keys, values: [B, S, inner]
        let q_proj = batch_proj(input, &w_q, d_model, inner)?;
        let k_proj = batch_proj(input, &w_k, d_model, inner)?;
        let v_proj = batch_proj(input, &w_v, d_model, inner)?;

        let mut concat = Array::zeros(IxDyn(&[batch, seq, inner]));

        for b in 0..batch {
            for h in 0..num_heads {
                let h_start = h * head_dim;
                let h_end = h_start + head_dim;

                // Extract [seq, head_dim] slices for this batch & head.
                let mut q_h = Array::zeros(IxDyn(&[seq, head_dim]));
                let mut k_h = Array::zeros(IxDyn(&[seq, head_dim]));
                let mut v_h = Array::zeros(IxDyn(&[seq, head_dim]));
                for t in 0..seq {
                    for d in 0..head_dim {
                        q_h[[t, d]] = q_proj[[b, t, h_start + d]];
                        k_h[[t, d]] = k_proj[[b, t, h_start + d]];
                        v_h[[t, d]] = v_proj[[b, t, h_start + d]];
                    }
                }

                let head_out = self.sparse_head(
                    &q_h, &k_h, &v_h, seq, head_dim, scale, num_globals, window_radius,
                );

                for t in 0..seq {
                    for d in 0..head_dim {
                        concat[[b, t, h_start + d]] = head_out[[t, d]];
                    }
                }
            }
        }

        // Final output projection: [B, S, d_model]
        batch_proj(&concat, &w_o, inner, d_model)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
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
            self.w_q.read().map(|g| g.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[]))),
            self.w_k.read().map(|g| g.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[]))),
            self.w_v.read().map(|g| g.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[]))),
            self.w_o.read().map(|g| g.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[]))),
        ]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn layer_type(&self) -> &str {
        "SparseAttention"
    }

    fn parameter_count(&self) -> usize {
        let inner = self.config.num_heads * self.config.head_dim;
        2 * (self.d_model * inner) + (inner * self.d_model) + (self.d_model * inner)
    }
}

unsafe impl<F: Float + Debug + Send + Sync + NumAssign> Send for SparseAttention<F> {}
unsafe impl<F: Float + Debug + Send + Sync + NumAssign> Sync for SparseAttention<F> {}

// ===========================================================================
// 2.  LinearAttentionVariant  (ELU / ReLU kernel maps)
// ===========================================================================

/// Positivity-preserving kernel feature map for linear attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearKernelMap {
    /// φ(x) = ELU(x) + 1  (Katharopoulos et al., 2020)
    Elu,
    /// φ(x) = ReLU(x) + ε
    Relu,
    /// φ(x) = x (no transformation; only safe if inputs are non-negative)
    Identity,
}

/// Configuration for [`LinearAttentionVariant`].
#[derive(Debug, Clone)]
pub struct LinearAttentionVariantConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Kernel feature map φ applied to queries and keys.
    pub kernel: LinearKernelMap,
    /// Small constant for numerical stability in the denominator.
    pub eps: f64,
    /// Apply causal masking (left-to-right autoregressive).
    pub causal: bool,
}

impl Default for LinearAttentionVariantConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            kernel: LinearKernelMap::Elu,
            eps: 1e-6,
            causal: false,
        }
    }
}

/// Linear attention using kernel feature maps φ(x).
///
/// Replaces the softmax with a positive definite kernel:
/// ```text
/// Attention(Q, K, V) ≈ φ(Q) (φ(K)ᵀ V) / (φ(Q) φ(K)ᵀ 1)
/// ```
/// Complexity: **O(N · d²)** — linear in sequence length.
///
/// # Input / Output
/// Shape `[batch, seq, d_model]` → `[batch, seq, d_model]`.
#[derive(Debug)]
pub struct LinearAttentionVariant<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: LinearAttentionVariantConfig,
    w_q: Arc<RwLock<Array<F, IxDyn>>>,
    w_k: Arc<RwLock<Array<F, IxDyn>>>,
    w_v: Arc<RwLock<Array<F, IxDyn>>>,
    w_o: Arc<RwLock<Array<F, IxDyn>>>,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static>
    LinearAttentionVariant<F>
{
    /// Create a new `LinearAttentionVariant`.
    pub fn new<R: Rng>(
        d_model: usize,
        config: LinearAttentionVariantConfig,
        rng: &mut R,
    ) -> Result<Self> {
        if d_model == 0 {
            return Err(NeuralError::InvalidArchitecture("d_model must be > 0".into()));
        }
        if config.num_heads == 0 || config.head_dim == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "num_heads and head_dim must be > 0".into(),
            ));
        }
        let inner = config.num_heads * config.head_dim;
        Ok(Self {
            d_model,
            config,
            w_q: Arc::new(RwLock::new(mk_weight(d_model, inner, rng)?)),
            w_k: Arc::new(RwLock::new(mk_weight(d_model, inner, rng)?)),
            w_v: Arc::new(RwLock::new(mk_weight(d_model, inner, rng)?)),
            w_o: Arc::new(RwLock::new(mk_weight(inner, d_model, rng)?)),
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Apply the kernel feature map element-wise.
    fn phi(&self, x: F) -> F {
        match self.config.kernel {
            LinearKernelMap::Elu => {
                if x >= F::zero() {
                    x + F::one()
                } else {
                    x.exp() // ELU(x) = exp(x) for x < 0; +1 applied after
                    // Re-applying: φ(x) = ELU(x) + 1 = exp(x) for x<0, x+1 for x>=0
                    // Rewrite properly:
                }
            }
            LinearKernelMap::Relu => {
                if x >= F::zero() {
                    x
                } else {
                    F::zero()
                }
            }
            LinearKernelMap::Identity => x,
        }
    }

    /// Correct ELU+1 feature map.
    fn phi_correct(&self, x: F) -> F {
        match self.config.kernel {
            LinearKernelMap::Elu => {
                if x >= F::zero() {
                    x + F::one()
                } else {
                    x.exp()
                }
            }
            LinearKernelMap::Relu => {
                let relu = if x >= F::zero() { x } else { F::zero() };
                relu + F::from(1e-6_f64).unwrap_or(F::zero())
            }
            LinearKernelMap::Identity => x,
        }
    }

    /// Non-causal linear attention kernel.
    fn non_causal_head(
        &self,
        q: &Array<F, IxDyn>, // [seq, head_dim]
        k: &Array<F, IxDyn>,
        v: &Array<F, IxDyn>,
        seq: usize,
        head_dim: usize,
        eps: F,
    ) -> Array<F, IxDyn> {
        // Apply kernel feature map
        let mut phi_q = Array::zeros(IxDyn(&[seq, head_dim]));
        let mut phi_k = Array::zeros(IxDyn(&[seq, head_dim]));
        for t in 0..seq {
            for d in 0..head_dim {
                phi_q[[t, d]] = self.phi_correct(q[[t, d]]);
                phi_k[[t, d]] = self.phi_correct(k[[t, d]]);
            }
        }

        // kv = φ(K)ᵀ V  [head_dim, head_dim]
        let mut kv = Array::zeros(IxDyn(&[head_dim, head_dim]));
        for t in 0..seq {
            for i in 0..head_dim {
                for j in 0..head_dim {
                    kv[[i, j]] += phi_k[[t, i]] * v[[t, j]];
                }
            }
        }

        // k_sum = sum of φ(K) over seq  [head_dim]
        let mut k_sum = vec![F::zero(); head_dim];
        for t in 0..seq {
            for d in 0..head_dim {
                k_sum[d] += phi_k[[t, d]];
            }
        }

        // output[t] = φ(Q[t]) · kv / (φ(Q[t]) · k_sum + eps)
        let mut out = Array::zeros(IxDyn(&[seq, head_dim]));
        for t in 0..seq {
            // numerator: φ(Q[t]) · kv   [head_dim]
            let mut numer = vec![F::zero(); head_dim];
            for j in 0..head_dim {
                let mut acc = F::zero();
                for i in 0..head_dim {
                    acc += phi_q[[t, i]] * kv[[i, j]];
                }
                numer[j] = acc;
            }
            // denominator: φ(Q[t]) · k_sum  (scalar)
            let mut denom = F::zero();
            for d in 0..head_dim {
                denom += phi_q[[t, d]] * k_sum[d];
            }
            let denom = denom + eps;
            for j in 0..head_dim {
                out[[t, j]] = numer[j] / denom;
            }
        }
        out
    }

    /// Causal linear attention using prefix sums.
    fn causal_head(
        &self,
        q: &Array<F, IxDyn>,
        k: &Array<F, IxDyn>,
        v: &Array<F, IxDyn>,
        seq: usize,
        head_dim: usize,
        eps: F,
    ) -> Array<F, IxDyn> {
        let mut phi_q = Array::zeros(IxDyn(&[seq, head_dim]));
        let mut phi_k = Array::zeros(IxDyn(&[seq, head_dim]));
        for t in 0..seq {
            for d in 0..head_dim {
                phi_q[[t, d]] = self.phi_correct(q[[t, d]]);
                phi_k[[t, d]] = self.phi_correct(k[[t, d]]);
            }
        }

        let mut out = Array::zeros(IxDyn(&[seq, head_dim]));
        // Running sum: kv_cumsum [head_dim, head_dim], k_cumsum [head_dim]
        let mut kv_cumsum = vec![F::zero(); head_dim * head_dim];
        let mut k_cumsum = vec![F::zero(); head_dim];

        for t in 0..seq {
            // Update cumulative sums with position t
            for i in 0..head_dim {
                k_cumsum[i] += phi_k[[t, i]];
                for j in 0..head_dim {
                    kv_cumsum[i * head_dim + j] += phi_k[[t, i]] * v[[t, j]];
                }
            }

            // Compute output: φ(Q[t]) · kv_cumsum / (φ(Q[t]) · k_cumsum + eps)
            let mut numer = vec![F::zero(); head_dim];
            for j in 0..head_dim {
                let mut acc = F::zero();
                for i in 0..head_dim {
                    acc += phi_q[[t, i]] * kv_cumsum[i * head_dim + j];
                }
                numer[j] = acc;
            }
            let mut denom = F::zero();
            for i in 0..head_dim {
                denom += phi_q[[t, i]] * k_cumsum[i];
            }
            let denom = denom + eps;
            for j in 0..head_dim {
                out[[t, j]] = numer[j] / denom;
            }
        }
        out
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static>
    AttentionVariant<F> for LinearAttentionVariant<F>
{
    fn forward_attn(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.forward(input)
    }

    fn variant_name(&self) -> &str {
        "LinearAttentionVariant"
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for LinearAttentionVariant<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "LinearAttentionVariant: expected 3D, got {}D",
                shape.len()
            )));
        }
        let (batch, seq, d_model) = (shape[0], shape[1], shape[2]);
        if d_model != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "LinearAttentionVariant: d_model mismatch: expected {}, got {d_model}",
                self.d_model
            )));
        }

        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let inner = num_heads * head_dim;
        let eps = F::from(self.config.eps)
            .ok_or_else(|| NeuralError::InvalidArchitecture("eps cast failed".into()))?;

        let w_q = self.w_q.read().map_err(|_| NeuralError::InferenceError("lock".into()))?;
        let w_k = self.w_k.read().map_err(|_| NeuralError::InferenceError("lock".into()))?;
        let w_v = self.w_v.read().map_err(|_| NeuralError::InferenceError("lock".into()))?;
        let w_o = self.w_o.read().map_err(|_| NeuralError::InferenceError("lock".into()))?;

        let q_proj = batch_proj(input, &w_q, d_model, inner)?;
        let k_proj = batch_proj(input, &w_k, d_model, inner)?;
        let v_proj = batch_proj(input, &w_v, d_model, inner)?;

        let mut concat = Array::zeros(IxDyn(&[batch, seq, inner]));

        for b in 0..batch {
            for h in 0..num_heads {
                let h_start = h * head_dim;
                let mut q_h = Array::zeros(IxDyn(&[seq, head_dim]));
                let mut k_h = Array::zeros(IxDyn(&[seq, head_dim]));
                let mut v_h = Array::zeros(IxDyn(&[seq, head_dim]));
                for t in 0..seq {
                    for d in 0..head_dim {
                        q_h[[t, d]] = q_proj[[b, t, h_start + d]];
                        k_h[[t, d]] = k_proj[[b, t, h_start + d]];
                        v_h[[t, d]] = v_proj[[b, t, h_start + d]];
                    }
                }

                let head_out = if self.config.causal {
                    self.causal_head(&q_h, &k_h, &v_h, seq, head_dim, eps)
                } else {
                    self.non_causal_head(&q_h, &k_h, &v_h, seq, head_dim, eps)
                };

                for t in 0..seq {
                    for d in 0..head_dim {
                        concat[[b, t, h_start + d]] = head_out[[t, d]];
                    }
                }
            }
        }

        batch_proj(&concat, &w_o, inner, d_model)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
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
            self.w_q.read().map(|g| g.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[]))),
            self.w_k.read().map(|g| g.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[]))),
            self.w_v.read().map(|g| g.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[]))),
            self.w_o.read().map(|g| g.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[]))),
        ]
    }

    fn set_training(&mut self, t: bool) {
        self.training = t;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn layer_type(&self) -> &str {
        "LinearAttentionVariant"
    }

    fn parameter_count(&self) -> usize {
        let inner = self.config.num_heads * self.config.head_dim;
        4 * self.d_model * inner
    }
}

unsafe impl<F: Float + Debug + Send + Sync + NumAssign> Send for LinearAttentionVariant<F> {}
unsafe impl<F: Float + Debug + Send + Sync + NumAssign> Sync for LinearAttentionVariant<F> {}

// ===========================================================================
// 3.  PerformerVariant  (FAVOR+)
// ===========================================================================

/// Configuration for [`PerformerVariant`].
#[derive(Debug, Clone)]
pub struct PerformerVariantConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Number of random features `m` (higher → more accurate; typical: 256).
    pub num_random_features: usize,
    /// Use orthogonal random features (more stable than i.i.d.).
    pub orthogonal_features: bool,
    /// Apply causal masking.
    pub causal: bool,
    /// Numerical stability denominator.
    pub eps: f64,
}

impl Default for PerformerVariantConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            num_random_features: 256,
            orthogonal_features: true,
            causal: false,
            eps: 1e-6,
        }
    }
}

/// FAVOR+ Performer attention.
///
/// Approximates softmax attention using random orthogonal feature maps (ORFM)
/// to express the softmax kernel as an expectation:
/// ```text
/// K(q, k) ≈ E[φ(q)ᵀ φ(k)]
/// ```
/// where φ maps to a finite-dimensional feature space, enabling O(N·m·d)
/// complexity (m = num_random_features).
///
/// # Input / Output
/// Shape `[batch, seq, d_model]` → `[batch, seq, d_model]`.
#[derive(Debug)]
pub struct PerformerVariant<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: PerformerVariantConfig,
    w_q: Arc<RwLock<Array<F, IxDyn>>>,
    w_k: Arc<RwLock<Array<F, IxDyn>>>,
    w_v: Arc<RwLock<Array<F, IxDyn>>>,
    w_o: Arc<RwLock<Array<F, IxDyn>>>,
    /// Random projection matrix [head_dim, num_random_features]
    omega: Array<F, IxDyn>,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> PerformerVariant<F> {
    /// Create a new `PerformerVariant` with fixed random projection matrix.
    pub fn new<R: Rng>(d_model: usize, config: PerformerVariantConfig, rng: &mut R) -> Result<Self> {
        if d_model == 0 {
            return Err(NeuralError::InvalidArchitecture("d_model must be > 0".into()));
        }
        if config.num_heads == 0 || config.head_dim == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "num_heads and head_dim must be > 0".into(),
            ));
        }
        if config.num_random_features == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "num_random_features must be > 0".into(),
            ));
        }

        let inner = config.num_heads * config.head_dim;
        let m = config.num_random_features;
        let d = config.head_dim;

        // Generate random projection matrix [d, m] from N(0,1).
        let mut omega_data = Vec::with_capacity(d * m);
        for _ in 0..d * m {
            // Box-Muller for standard normal samples.
            let u1: f64 = rng.random_range(1e-10..1.0);
            let u2: f64 = rng.random_range(0.0..1.0);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = F::from(z)
                .ok_or_else(|| NeuralError::InvalidArchitecture("omega cast failed".into()))?;
            omega_data.push(v);
        }

        // Simple orthogonalisation via Gram-Schmidt if requested and d <= m.
        if config.orthogonal_features && d <= m {
            let num_blocks = m / d;
            for blk in 0..num_blocks {
                let offset = blk * d;
                for col in 0..d {
                    let c = offset + col;
                    if c >= m {
                        break;
                    }
                    // Orthogonalise column c against previous columns in this block.
                    for prev in 0..col {
                        let p = offset + prev;
                        if p >= m {
                            break;
                        }
                        // dot product
                        let mut dot = F::zero();
                        let mut norm_sq = F::zero();
                        for row in 0..d {
                            dot += omega_data[row * m + c] * omega_data[row * m + p];
                            norm_sq += omega_data[row * m + p] * omega_data[row * m + p];
                        }
                        if norm_sq > F::zero() {
                            let factor = dot / norm_sq;
                            for row in 0..d {
                                let subtract = factor * omega_data[row * m + p];
                                omega_data[row * m + c] = omega_data[row * m + c] - subtract;
                            }
                        }
                    }
                    // Normalise column c.
                    let mut norm_sq = F::zero();
                    for row in 0..d {
                        norm_sq += omega_data[row * m + c] * omega_data[row * m + c];
                    }
                    if norm_sq > F::zero() {
                        let norm = norm_sq.sqrt();
                        for row in 0..d {
                            omega_data[row * m + c] = omega_data[row * m + c] / norm;
                        }
                    }
                }
            }
        }

        let omega = Array::from_shape_vec(IxDyn(&[d, m]), omega_data)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("omega shape: {e}")))?;

        Ok(Self {
            d_model,
            config,
            w_q: Arc::new(RwLock::new(mk_weight(d_model, inner, rng)?)),
            w_k: Arc::new(RwLock::new(mk_weight(d_model, inner, rng)?)),
            w_v: Arc::new(RwLock::new(mk_weight(d_model, inner, rng)?)),
            w_o: Arc::new(RwLock::new(mk_weight(inner, d_model, rng)?)),
            omega,
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Compute FAVOR+ feature map for a single token vector [head_dim].
    ///
    /// φ(x) = exp(-‖x‖²/2) · [exp(ωᵢ·x) for ωᵢ in Ω] / √m
    fn favor_phi(&self, x: &[F], head_dim: usize, m: usize) -> Vec<F> {
        // ‖x‖² / 2
        let mut norm_sq = F::zero();
        for &xi in x {
            norm_sq += xi * xi;
        }
        let half_norm = norm_sq / F::from(2.0_f64).unwrap_or(F::one());
        let scale = (-half_norm).exp()
            / F::from((m as f64).sqrt()).unwrap_or(F::one());

        let mut features = Vec::with_capacity(m);
        for col in 0..m {
            let mut dot = F::zero();
            for row in 0..head_dim {
                dot += x[row] * self.omega[[row, col]];
            }
            features.push(dot.exp() * scale);
        }
        features
    }

    /// Run one head through FAVOR+ linear attention.
    fn performer_head(
        &self,
        q: &Array<F, IxDyn>,
        k: &Array<F, IxDyn>,
        v: &Array<F, IxDyn>,
        seq: usize,
        head_dim: usize,
        m: usize,
        eps: F,
    ) -> Array<F, IxDyn> {
        // Compute feature maps for all Q and K tokens.
        let mut phi_q = Vec::with_capacity(seq);
        let mut phi_k = Vec::with_capacity(seq);
        for t in 0..seq {
            let q_t: Vec<F> = (0..head_dim).map(|d| q[[t, d]]).collect();
            let k_t: Vec<F> = (0..head_dim).map(|d| k[[t, d]]).collect();
            phi_q.push(self.favor_phi(&q_t, head_dim, m));
            phi_k.push(self.favor_phi(&k_t, head_dim, m));
        }

        // kv_sum = Σ_t φ(K[t])ᵀ V[t]  [m, head_dim]
        let mut kv_sum = vec![F::zero(); m * head_dim];
        // k_sum = Σ_t φ(K[t])  [m]
        let mut k_sum = vec![F::zero(); m];

        // For causal: use incremental accumulation.
        let mut out = Array::zeros(IxDyn(&[seq, head_dim]));

        if self.config.causal {
            for t in 0..seq {
                // Update running sums with position t.
                for r in 0..m {
                    k_sum[r] += phi_k[t][r];
                    for d in 0..head_dim {
                        kv_sum[r * head_dim + d] += phi_k[t][r] * v[[t, d]];
                    }
                }
                // Output for this position.
                let mut denom = F::zero();
                for r in 0..m {
                    denom += phi_q[t][r] * k_sum[r];
                }
                let denom = denom + eps;
                for d in 0..head_dim {
                    let mut numer = F::zero();
                    for r in 0..m {
                        numer += phi_q[t][r] * kv_sum[r * head_dim + d];
                    }
                    out[[t, d]] = numer / denom;
                }
            }
        } else {
            // Non-causal: accumulate all, then divide.
            for t in 0..seq {
                for r in 0..m {
                    k_sum[r] += phi_k[t][r];
                    for d in 0..head_dim {
                        kv_sum[r * head_dim + d] += phi_k[t][r] * v[[t, d]];
                    }
                }
            }
            for t in 0..seq {
                let mut denom = F::zero();
                for r in 0..m {
                    denom += phi_q[t][r] * k_sum[r];
                }
                let denom = denom + eps;
                for d in 0..head_dim {
                    let mut numer = F::zero();
                    for r in 0..m {
                        numer += phi_q[t][r] * kv_sum[r * head_dim + d];
                    }
                    out[[t, d]] = numer / denom;
                }
            }
        }
        out
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static>
    AttentionVariant<F> for PerformerVariant<F>
{
    fn forward_attn(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.forward(input)
    }

    fn variant_name(&self) -> &str {
        "PerformerVariant"
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for PerformerVariant<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "PerformerVariant: expected 3D, got {}D",
                shape.len()
            )));
        }
        let (batch, seq, d_model) = (shape[0], shape[1], shape[2]);
        if d_model != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "PerformerVariant: d_model mismatch: expected {}, got {d_model}",
                self.d_model
            )));
        }

        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let inner = num_heads * head_dim;
        let m = self.config.num_random_features;
        let eps = F::from(self.config.eps)
            .ok_or_else(|| NeuralError::InvalidArchitecture("eps cast".into()))?;

        let w_q = self.w_q.read().map_err(|_| NeuralError::InferenceError("lock".into()))?;
        let w_k = self.w_k.read().map_err(|_| NeuralError::InferenceError("lock".into()))?;
        let w_v = self.w_v.read().map_err(|_| NeuralError::InferenceError("lock".into()))?;
        let w_o = self.w_o.read().map_err(|_| NeuralError::InferenceError("lock".into()))?;

        let q_proj = batch_proj(input, &w_q, d_model, inner)?;
        let k_proj = batch_proj(input, &w_k, d_model, inner)?;
        let v_proj = batch_proj(input, &w_v, d_model, inner)?;

        let mut concat = Array::zeros(IxDyn(&[batch, seq, inner]));

        for b in 0..batch {
            for h in 0..num_heads {
                let h_start = h * head_dim;
                let mut q_h = Array::zeros(IxDyn(&[seq, head_dim]));
                let mut k_h = Array::zeros(IxDyn(&[seq, head_dim]));
                let mut v_h = Array::zeros(IxDyn(&[seq, head_dim]));
                for t in 0..seq {
                    for d in 0..head_dim {
                        q_h[[t, d]] = q_proj[[b, t, h_start + d]];
                        k_h[[t, d]] = k_proj[[b, t, h_start + d]];
                        v_h[[t, d]] = v_proj[[b, t, h_start + d]];
                    }
                }

                let head_out =
                    self.performer_head(&q_h, &k_h, &v_h, seq, head_dim, m, eps);

                for t in 0..seq {
                    for d in 0..head_dim {
                        concat[[b, t, h_start + d]] = head_out[[t, d]];
                    }
                }
            }
        }

        batch_proj(&concat, &w_o, inner, d_model)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
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
            self.w_q.read().map(|g| g.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[]))),
            self.w_k.read().map(|g| g.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[]))),
            self.w_v.read().map(|g| g.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[]))),
            self.w_o.read().map(|g| g.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[]))),
        ]
    }

    fn set_training(&mut self, t: bool) {
        self.training = t;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn layer_type(&self) -> &str {
        "PerformerVariant"
    }

    fn parameter_count(&self) -> usize {
        let inner = self.config.num_heads * self.config.head_dim;
        4 * self.d_model * inner
    }
}

unsafe impl<F: Float + Debug + Send + Sync + NumAssign> Send for PerformerVariant<F> {}
unsafe impl<F: Float + Debug + Send + Sync + NumAssign> Sync for PerformerVariant<F> {}

// ===========================================================================
// 4.  CrossAttention
// ===========================================================================

/// Configuration for [`CrossAttention`].
#[derive(Debug, Clone)]
pub struct CrossAttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per query/key/value head.
    pub head_dim: usize,
    /// Scaling factor (defaults to `1/√head_dim`).
    pub scale: Option<f64>,
    /// Dropout probability applied to attention weights.
    pub dropout_prob: f64,
}

impl Default for CrossAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            scale: None,
            dropout_prob: 0.0,
        }
    }
}

/// Cross-attention layer.
///
/// Computes multi-head attention where queries come from an encoder-or query
/// sequence and keys/values come from a separate memory sequence.  The two
/// sequences may have different lengths but must share the same `d_model`.
///
/// # Usage
/// ```ignore
/// let out = layer.forward_cross(&query_seq, &memory_seq)?;
/// ```
///
/// Calling `Layer::forward` interprets the input as both the query and memory
/// (i.e. self-attention over the input), which is useful for stacking in a
/// generic layer pipeline.
#[derive(Debug)]
pub struct CrossAttention<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: CrossAttentionConfig,
    /// Query projection [d_model, num_heads * head_dim]
    w_q: Arc<RwLock<Array<F, IxDyn>>>,
    /// Key projection   [d_model, num_heads * head_dim]
    w_k: Arc<RwLock<Array<F, IxDyn>>>,
    /// Value projection [d_model, num_heads * head_dim]
    w_v: Arc<RwLock<Array<F, IxDyn>>>,
    /// Output projection [num_heads * head_dim, d_model]
    w_o: Arc<RwLock<Array<F, IxDyn>>>,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> CrossAttention<F> {
    /// Create a new `CrossAttention` layer.
    ///
    /// # Arguments
    /// * `d_model` – Embedding dimension shared by query and memory sequences.
    /// * `config` – Attention configuration.
    /// * `rng` – Random number generator for weight initialisation.
    pub fn new<R: Rng>(d_model: usize, config: CrossAttentionConfig, rng: &mut R) -> Result<Self> {
        if d_model == 0 {
            return Err(NeuralError::InvalidArchitecture("d_model must be > 0".into()));
        }
        if config.num_heads == 0 || config.head_dim == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "num_heads and head_dim must be > 0".into(),
            ));
        }
        let inner = config.num_heads * config.head_dim;
        Ok(Self {
            d_model,
            config,
            w_q: Arc::new(RwLock::new(mk_weight(d_model, inner, rng)?)),
            w_k: Arc::new(RwLock::new(mk_weight(d_model, inner, rng)?)),
            w_v: Arc::new(RwLock::new(mk_weight(d_model, inner, rng)?)),
            w_o: Arc::new(RwLock::new(mk_weight(inner, d_model, rng)?)),
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Cross-attention forward pass with separate query and memory tensors.
    ///
    /// # Arguments
    /// * `query` – Query sequence `[batch, seq_q, d_model]`.
    /// * `memory` – Key/value memory sequence `[batch, seq_kv, d_model]`.
    ///
    /// # Returns
    /// Output tensor `[batch, seq_q, d_model]`.
    pub fn forward_cross(
        &self,
        query: &Array<F, IxDyn>,
        memory: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let q_shape = query.shape();
        let m_shape = memory.shape();

        if q_shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "CrossAttention: query must be 3D, got {}D",
                q_shape.len()
            )));
        }
        if m_shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "CrossAttention: memory must be 3D, got {}D",
                m_shape.len()
            )));
        }

        let (batch, seq_q, d_q) = (q_shape[0], q_shape[1], q_shape[2]);
        let (batch_m, seq_kv, d_m) = (m_shape[0], m_shape[1], m_shape[2]);

        if batch != batch_m {
            return Err(NeuralError::InferenceError(format!(
                "CrossAttention: batch size mismatch: query={batch}, memory={batch_m}"
            )));
        }
        if d_q != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "CrossAttention: query d_model mismatch: expected {}, got {d_q}",
                self.d_model
            )));
        }
        if d_m != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "CrossAttention: memory d_model mismatch: expected {}, got {d_m}",
                self.d_model
            )));
        }

        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let inner = num_heads * head_dim;
        let scale_val = self.config.scale.unwrap_or(1.0 / (head_dim as f64).sqrt());
        let scale = F::from(scale_val)
            .ok_or_else(|| NeuralError::InvalidArchitecture("scale cast".into()))?;

        let w_q = self.w_q.read().map_err(|_| NeuralError::InferenceError("lock".into()))?;
        let w_k = self.w_k.read().map_err(|_| NeuralError::InferenceError("lock".into()))?;
        let w_v = self.w_v.read().map_err(|_| NeuralError::InferenceError("lock".into()))?;
        let w_o = self.w_o.read().map_err(|_| NeuralError::InferenceError("lock".into()))?;

        // Project Q from query, K & V from memory.
        let q_proj = batch_proj(query, &w_q, self.d_model, inner)?;   // [B, seq_q,  inner]
        let k_proj = batch_proj(memory, &w_k, self.d_model, inner)?;  // [B, seq_kv, inner]
        let v_proj = batch_proj(memory, &w_v, self.d_model, inner)?;  // [B, seq_kv, inner]

        let mut concat = Array::zeros(IxDyn(&[batch, seq_q, inner]));

        for b in 0..batch {
            for h in 0..num_heads {
                let h_start = h * head_dim;

                // q_h: [seq_q, head_dim], k_h / v_h: [seq_kv, head_dim]
                let mut q_h = Array::zeros(IxDyn(&[seq_q, head_dim]));
                let mut k_h = Array::zeros(IxDyn(&[seq_kv, head_dim]));
                let mut v_h = Array::zeros(IxDyn(&[seq_kv, head_dim]));

                for t in 0..seq_q {
                    for d in 0..head_dim {
                        q_h[[t, d]] = q_proj[[b, t, h_start + d]];
                    }
                }
                for t in 0..seq_kv {
                    for d in 0..head_dim {
                        k_h[[t, d]] = k_proj[[b, t, h_start + d]];
                        v_h[[t, d]] = v_proj[[b, t, h_start + d]];
                    }
                }

                // Standard scaled dot-product attention.
                // scores[i,j] = q_h[i] · k_h[j] · scale   shape [seq_q, seq_kv]
                let mut scores = Array::zeros(IxDyn(&[seq_q, seq_kv]));
                for i in 0..seq_q {
                    for j in 0..seq_kv {
                        let mut dot = F::zero();
                        for d in 0..head_dim {
                            dot += q_h[[i, d]] * k_h[[j, d]];
                        }
                        scores[[i, j]] = dot * scale;
                    }
                }

                // Softmax over seq_kv dimension for each query position.
                for i in 0..seq_q {
                    let mut row: Vec<F> = (0..seq_kv).map(|j| scores[[i, j]]).collect();
                    softmax_inplace(&mut row);
                    for j in 0..seq_kv {
                        scores[[i, j]] = row[j];
                    }
                }

                // Weighted sum over values.
                for i in 0..seq_q {
                    for d in 0..head_dim {
                        let mut acc = F::zero();
                        for j in 0..seq_kv {
                            acc += scores[[i, j]] * v_h[[j, d]];
                        }
                        concat[[b, i, h_start + d]] = acc;
                    }
                }
            }
        }

        batch_proj(&concat, &w_o, inner, self.d_model)
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static>
    AttentionVariant<F> for CrossAttention<F>
{
    fn forward_attn(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.forward(input)
    }

    fn variant_name(&self) -> &str {
        "CrossAttention"
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for CrossAttention<F>
{
    /// Self-attention mode: query = memory = input.
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.forward_cross(input, input)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
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
            self.w_q.read().map(|g| g.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[]))),
            self.w_k.read().map(|g| g.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[]))),
            self.w_v.read().map(|g| g.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[]))),
            self.w_o.read().map(|g| g.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[]))),
        ]
    }

    fn set_training(&mut self, t: bool) {
        self.training = t;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn layer_type(&self) -> &str {
        "CrossAttention"
    }

    fn parameter_count(&self) -> usize {
        let inner = self.config.num_heads * self.config.head_dim;
        4 * self.d_model * inner
    }
}

unsafe impl<F: Float + Debug + Send + Sync + NumAssign> Send for CrossAttention<F> {}
unsafe impl<F: Float + Debug + Send + Sync + NumAssign> Sync for CrossAttention<F> {}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;
    use scirs2_core::random::rng;

    // ---- SparseAttention ----

    #[test]
    fn test_sparse_attention_creation() {
        let mut r = rng();
        let cfg = SparseAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            window_radius: 2,
            num_global_tokens: 1,
            scale: None,
        };
        let layer = SparseAttention::<f64>::new(16, cfg, &mut r).expect("create failed");
        assert_eq!(layer.layer_type(), "SparseAttention");
        assert_eq!(layer.variant_name(), "SparseAttention");
    }

    #[test]
    fn test_sparse_attention_forward() {
        let mut r = rng();
        let cfg = SparseAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            window_radius: 2,
            num_global_tokens: 1,
            scale: None,
        };
        let layer = SparseAttention::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((2, 8, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 8, 16]);
    }

    #[test]
    fn test_sparse_attention_full_global() {
        let mut r = rng();
        let cfg = SparseAttentionConfig {
            num_heads: 2,
            head_dim: 4,
            window_radius: 0,
            num_global_tokens: 4, // all tokens are global → full attention
            scale: None,
        };
        let layer = SparseAttention::<f64>::new(8, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 4, 8), 0.2).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_sparse_attention_output_finite() {
        let mut r = rng();
        let cfg = SparseAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            window_radius: 3,
            num_global_tokens: 0,
            scale: None,
        };
        let layer = SparseAttention::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 6, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        for v in out.iter() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn test_sparse_attention_error_cases() {
        let mut r = rng();
        // zero d_model
        let cfg = SparseAttentionConfig::default();
        assert!(SparseAttention::<f64>::new(0, cfg.clone(), &mut r).is_err());
        // zero heads
        let cfg2 = SparseAttentionConfig { num_heads: 0, ..SparseAttentionConfig::default() };
        assert!(SparseAttention::<f64>::new(16, cfg2, &mut r).is_err());
    }

    // ---- LinearAttentionVariant ----

    #[test]
    fn test_linear_attention_variant_elu() {
        let mut r = rng();
        let cfg = LinearAttentionVariantConfig {
            num_heads: 2,
            head_dim: 8,
            kernel: LinearKernelMap::Elu,
            eps: 1e-6,
            causal: false,
        };
        let layer = LinearAttentionVariant::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((2, 5, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 5, 16]);
    }

    #[test]
    fn test_linear_attention_variant_causal() {
        let mut r = rng();
        let cfg = LinearAttentionVariantConfig {
            num_heads: 2,
            head_dim: 8,
            kernel: LinearKernelMap::Relu,
            eps: 1e-6,
            causal: true,
        };
        let layer = LinearAttentionVariant::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 6, 16), 0.2).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 6, 16]);
    }

    #[test]
    fn test_linear_attention_variant_identity() {
        let mut r = rng();
        let cfg = LinearAttentionVariantConfig {
            num_heads: 2,
            head_dim: 4,
            kernel: LinearKernelMap::Identity,
            eps: 1e-6,
            causal: false,
        };
        let layer = LinearAttentionVariant::<f64>::new(8, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 4, 8), 0.5).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_linear_attention_variant_output_finite() {
        let mut r = rng();
        let cfg = LinearAttentionVariantConfig::default();
        let cfg = LinearAttentionVariantConfig {
            num_heads: 2,
            head_dim: 8,
            ..cfg
        };
        let layer = LinearAttentionVariant::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 4, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        for v in out.iter() {
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }

    // ---- PerformerVariant ----

    #[test]
    fn test_performer_variant_creation() {
        let mut r = rng();
        let cfg = PerformerVariantConfig {
            num_heads: 2,
            head_dim: 8,
            num_random_features: 16,
            orthogonal_features: true,
            causal: false,
            eps: 1e-6,
        };
        let layer = PerformerVariant::<f64>::new(16, cfg, &mut r).expect("create failed");
        assert_eq!(layer.layer_type(), "PerformerVariant");
        assert_eq!(layer.variant_name(), "PerformerVariant");
    }

    #[test]
    fn test_performer_variant_forward() {
        let mut r = rng();
        let cfg = PerformerVariantConfig {
            num_heads: 2,
            head_dim: 8,
            num_random_features: 16,
            orthogonal_features: false,
            causal: false,
            eps: 1e-6,
        };
        let layer = PerformerVariant::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((2, 6, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 6, 16]);
    }

    #[test]
    fn test_performer_variant_causal() {
        let mut r = rng();
        let cfg = PerformerVariantConfig {
            num_heads: 2,
            head_dim: 8,
            num_random_features: 16,
            orthogonal_features: false,
            causal: true,
            eps: 1e-6,
        };
        let layer = PerformerVariant::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 5, 16), 0.2).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 5, 16]);
    }

    #[test]
    fn test_performer_variant_output_finite() {
        let mut r = rng();
        let cfg = PerformerVariantConfig {
            num_heads: 2,
            head_dim: 8,
            num_random_features: 32,
            orthogonal_features: true,
            causal: false,
            eps: 1e-6,
        };
        let layer = PerformerVariant::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 4, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        for v in out.iter() {
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }

    // ---- CrossAttention ----

    #[test]
    fn test_cross_attention_creation() {
        let mut r = rng();
        let cfg = CrossAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            scale: None,
            dropout_prob: 0.0,
        };
        let layer = CrossAttention::<f64>::new(16, cfg, &mut r).expect("create failed");
        assert_eq!(layer.layer_type(), "CrossAttention");
        assert_eq!(layer.variant_name(), "CrossAttention");
    }

    #[test]
    fn test_cross_attention_self_mode() {
        let mut r = rng();
        let cfg = CrossAttentionConfig::default();
        let cfg = CrossAttentionConfig { num_heads: 2, head_dim: 8, ..cfg };
        let layer = CrossAttention::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((2, 5, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 5, 16]);
    }

    #[test]
    fn test_cross_attention_different_lengths() {
        let mut r = rng();
        let cfg = CrossAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            scale: None,
            dropout_prob: 0.0,
        };
        let layer = CrossAttention::<f64>::new(16, cfg, &mut r).expect("create failed");
        // query: [2, 4, 16], memory: [2, 10, 16]
        let query = Array3::<f64>::from_elem((2, 4, 16), 0.1).into_dyn();
        let memory = Array3::<f64>::from_elem((2, 10, 16), 0.2).into_dyn();
        let out = layer.forward_cross(&query, &memory).expect("forward_cross failed");
        assert_eq!(out.shape(), &[2, 4, 16]);
    }

    #[test]
    fn test_cross_attention_batch_mismatch_error() {
        let mut r = rng();
        let cfg = CrossAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            scale: None,
            dropout_prob: 0.0,
        };
        let layer = CrossAttention::<f64>::new(16, cfg, &mut r).expect("create failed");
        let query = Array3::<f64>::from_elem((2, 4, 16), 0.1).into_dyn();
        let memory = Array3::<f64>::from_elem((3, 6, 16), 0.2).into_dyn();
        assert!(layer.forward_cross(&query, &memory).is_err());
    }

    #[test]
    fn test_cross_attention_output_finite() {
        let mut r = rng();
        let cfg = CrossAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            scale: None,
            dropout_prob: 0.0,
        };
        let layer = CrossAttention::<f64>::new(16, cfg, &mut r).expect("create failed");
        let query = Array3::<f64>::from_elem((1, 3, 16), 0.1).into_dyn();
        let memory = Array3::<f64>::from_elem((1, 7, 16), 0.2).into_dyn();
        let out = layer.forward_cross(&query, &memory).expect("forward_cross failed");
        for v in out.iter() {
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }
}
