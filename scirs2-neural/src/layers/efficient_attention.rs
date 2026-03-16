//! Memory-efficient attention functions and structures.
//!
//! This module provides efficient attention primitives that reduce memory or
//! compute complexity while maintaining correctness:
//!
//! - [`sliding_window_attention`] – Token-level sliding window: each query
//!   attends only to a local window of keys/values. O(N·W·d) where W is the
//!   window width.
//! - [`causal_linear_attention`] – Autoregressive (causal) linear attention
//!   using ELU+1 feature maps and running prefix sums. O(N·d²).
//! - [`grouped_query_attention`] – GQA: share key-value heads across groups of
//!   query heads, reducing KV cache size while keeping per-query expressiveness.
//! - [`FlashAttentionSimple`] is provided in `sparse_attention` (re-exported
//!   in `mod.rs`); this module adds a standalone function variant for use
//!   without constructing a full layer struct.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Xavier initialiser — flat Vec.
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
            .ok_or_else(|| NeuralError::InvalidArchitecture("xavier cast".into()))?;
        v.push(f);
    }
    Ok(v)
}

/// Build a 2-D weight array [rows, cols].
fn mk_weight<F: Float, R: Rng>(rows: usize, cols: usize, rng: &mut R) -> Result<Array<F, IxDyn>> {
    let data = xavier_vec(rows, cols, rows * cols, rng)?;
    Array::from_shape_vec(IxDyn(&[rows, cols]), data)
        .map_err(|e| NeuralError::InvalidArchitecture(format!("mk_weight: {e}")))
}

/// Batch linear projection `[B, S, D_in] × [D_in, D_out] → [B, S, D_out]`.
fn batch_proj<F: Float + NumAssign>(
    x: &Array<F, IxDyn>,
    w: &Array<F, IxDyn>,
    d_in: usize,
    d_out: usize,
) -> Result<Array<F, IxDyn>> {
    let s = x.shape();
    if s.len() != 3 {
        return Err(NeuralError::InferenceError(format!(
            "batch_proj: expected 3D, got {}D",
            s.len()
        )));
    }
    let (batch, seq) = (s[0], s[1]);
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
// 1.  sliding_window_attention  (free function)
// ===========================================================================

/// Sliding-window attention as a pure functional operation.
///
/// Each query token at position `t` attends to keys/values in the window
/// `[t - left_radius, t + right_radius]` (clamped to sequence bounds).
///
/// # Arguments
/// * `q` – Query tensor `[batch, seq, num_heads, head_dim]`.
/// * `k` – Key tensor  `[batch, seq, num_heads, head_dim]`.
/// * `v` – Value tensor `[batch, seq, num_heads, head_dim]`.
/// * `left_radius` – Tokens to the left each query can attend to.
/// * `right_radius` – Tokens to the right (set to 0 for causal masking).
///
/// # Returns
/// Output tensor `[batch, seq, num_heads, head_dim]`.
///
/// # Errors
/// Returns an error when input shapes are inconsistent.
pub fn sliding_window_attention<F: Float + NumAssign>(
    q: &Array<F, IxDyn>,
    k: &Array<F, IxDyn>,
    v: &Array<F, IxDyn>,
    left_radius: usize,
    right_radius: usize,
) -> Result<Array<F, IxDyn>> {
    let qs = q.shape();
    if qs.len() != 4 {
        return Err(NeuralError::InferenceError(format!(
            "sliding_window_attention: expected 4D q [B,S,H,D], got {}D",
            qs.len()
        )));
    }
    let (batch, seq, num_heads, head_dim) = (qs[0], qs[1], qs[2], qs[3]);

    let ks = k.shape();
    let vs = v.shape();
    if ks != qs || vs != qs {
        return Err(NeuralError::InferenceError(
            "sliding_window_attention: q, k, v must have identical shapes".into(),
        ));
    }

    let scale = F::from(1.0 / (head_dim as f64).sqrt())
        .ok_or_else(|| NeuralError::InvalidArchitecture("scale cast".into()))?;

    let mut out = Array::zeros(IxDyn(&[batch, seq, num_heads, head_dim]));

    for b in 0..batch {
        for h in 0..num_heads {
            for t in 0..seq {
                let win_start = if t >= left_radius { t - left_radius } else { 0 };
                let win_end = (t + right_radius + 1).min(seq);
                let win_len = win_end - win_start;

                // Compute attention scores over the window.
                let mut scores = Vec::with_capacity(win_len);
                for j in win_start..win_end {
                    let mut dot = F::zero();
                    for d in 0..head_dim {
                        dot += q[[b, t, h, d]] * k[[b, j, h, d]];
                    }
                    scores.push(dot * scale);
                }
                softmax_inplace(&mut scores);

                // Weighted sum of values.
                for d in 0..head_dim {
                    let mut acc = F::zero();
                    for (wi, j) in (win_start..win_end).enumerate() {
                        acc += scores[wi] * v[[b, j, h, d]];
                    }
                    out[[b, t, h, d]] = acc;
                }
            }
        }
    }
    Ok(out)
}

// ===========================================================================
// 2.  causal_linear_attention  (free function)
// ===========================================================================

/// Causal (autoregressive) linear attention as a pure functional operation.
///
/// Implements the recurrent formulation of linear attention with ELU+1 feature
/// maps, accumulating running sums from left to right:
/// ```text
/// S_t = S_{t-1} + φ(K[t])ᵀ V[t]    (outer product accumulated)
/// z_t = z_{t-1} + φ(K[t])            (denominator accumulated)
/// out[t] = φ(Q[t]) S_t / (φ(Q[t])·z_t + ε)
/// ```
///
/// # Arguments
/// * `q` – Query tensor  `[batch, seq, num_heads, head_dim]`.
/// * `k` – Key tensor    `[batch, seq, num_heads, head_dim]`.
/// * `v` – Value tensor  `[batch, seq, num_heads, head_dim]`.
/// * `eps` – Numerical stability constant (typical: 1e-6).
///
/// # Returns
/// Output tensor `[batch, seq, num_heads, head_dim]`.
pub fn causal_linear_attention<F: Float + NumAssign>(
    q: &Array<F, IxDyn>,
    k: &Array<F, IxDyn>,
    v: &Array<F, IxDyn>,
    eps: F,
) -> Result<Array<F, IxDyn>> {
    let qs = q.shape();
    if qs.len() != 4 {
        return Err(NeuralError::InferenceError(format!(
            "causal_linear_attention: expected 4D [B,S,H,D], got {}D",
            qs.len()
        )));
    }
    let (batch, seq, num_heads, head_dim) = (qs[0], qs[1], qs[2], qs[3]);

    let ks = k.shape();
    let vs = v.shape();
    if ks[0] != batch || ks[1] != seq || ks[2] != num_heads || ks[3] != head_dim {
        return Err(NeuralError::InferenceError(
            "causal_linear_attention: k shape mismatch".into(),
        ));
    }
    if vs[0] != batch || vs[1] != seq || vs[2] != num_heads || vs[3] != head_dim {
        return Err(NeuralError::InferenceError(
            "causal_linear_attention: v shape mismatch".into(),
        ));
    }

    /// ELU + 1 feature map (ensures positivity).
    fn phi<F: Float>(x: F) -> F {
        if x >= F::zero() {
            x + F::one()
        } else {
            x.exp()
        }
    }

    let mut out = Array::zeros(IxDyn(&[batch, seq, num_heads, head_dim]));

    for b in 0..batch {
        for h in 0..num_heads {
            // Running outer product S: [head_dim × head_dim], stored row-major.
            let mut s_running = vec![F::zero(); head_dim * head_dim];
            // Running denominator z: [head_dim]
            let mut z_running = vec![F::zero(); head_dim];

            for t in 0..seq {
                // φ(K[t]) and φ(Q[t])
                let phi_k: Vec<F> = (0..head_dim).map(|d| phi(k[[b, t, h, d]])).collect();
                let phi_q: Vec<F> = (0..head_dim).map(|d| phi(q[[b, t, h, d]])).collect();

                // Update S += φ(K[t]) ⊗ V[t]
                for i in 0..head_dim {
                    z_running[i] += phi_k[i];
                    for j in 0..head_dim {
                        s_running[i * head_dim + j] += phi_k[i] * v[[b, t, h, j]];
                    }
                }

                // Denominator: φ(Q[t]) · z
                let mut denom = F::zero();
                for i in 0..head_dim {
                    denom += phi_q[i] * z_running[i];
                }
                let denom = denom + eps;

                // Numerator: φ(Q[t]) · S  [head_dim]
                for j in 0..head_dim {
                    let mut numer = F::zero();
                    for i in 0..head_dim {
                        numer += phi_q[i] * s_running[i * head_dim + j];
                    }
                    out[[b, t, h, j]] = numer / denom;
                }
            }
        }
    }
    Ok(out)
}

// ===========================================================================
// 3.  grouped_query_attention  (free function)
// ===========================================================================

/// Grouped query attention as a pure functional operation (GQA).
///
/// GQA uses `num_kv_heads` key-value heads that are shared across groups of
/// query heads.  Each group of `num_q_heads / num_kv_heads` query heads attends
/// to the same key-value pair, reducing the KV cache by a factor equal to the
/// group size.
///
/// # Arguments
/// * `q` – Query tensor `[batch, seq, num_q_heads, head_dim]`.
/// * `k` – Key tensor   `[batch, seq, num_kv_heads, head_dim]`.
/// * `v` – Value tensor `[batch, seq, num_kv_heads, head_dim]`.
/// * `causal` – If true, applies a causal (lower triangular) mask.
///
/// # Returns
/// Output tensor `[batch, seq, num_q_heads, head_dim]`.
///
/// # Errors
/// Returns an error if `num_q_heads` is not divisible by `num_kv_heads`, or
/// if other shape constraints are violated.
pub fn grouped_query_attention<F: Float + NumAssign>(
    q: &Array<F, IxDyn>,
    k: &Array<F, IxDyn>,
    v: &Array<F, IxDyn>,
    causal: bool,
) -> Result<Array<F, IxDyn>> {
    let qs = q.shape();
    if qs.len() != 4 {
        return Err(NeuralError::InferenceError(format!(
            "grouped_query_attention: q must be 4D [B,S,H_q,D], got {}D",
            qs.len()
        )));
    }
    let (batch, seq, num_q_heads, head_dim) = (qs[0], qs[1], qs[2], qs[3]);

    let ks = k.shape();
    if ks.len() != 4 {
        return Err(NeuralError::InferenceError(format!(
            "grouped_query_attention: k must be 4D, got {}D",
            ks.len()
        )));
    }
    let num_kv_heads = ks[2];

    if ks[0] != batch || ks[1] != seq || ks[3] != head_dim {
        return Err(NeuralError::InferenceError(
            "grouped_query_attention: k shape mismatch (batch/seq/head_dim)".into(),
        ));
    }

    let vs = v.shape();
    if vs[0] != batch || vs[1] != seq || vs[2] != num_kv_heads || vs[3] != head_dim {
        return Err(NeuralError::InferenceError(
            "grouped_query_attention: v must have same shape as k".into(),
        ));
    }

    if num_q_heads == 0 || num_kv_heads == 0 {
        return Err(NeuralError::InferenceError(
            "grouped_query_attention: head counts must be > 0".into(),
        ));
    }

    if num_q_heads % num_kv_heads != 0 {
        return Err(NeuralError::InferenceError(format!(
            "grouped_query_attention: num_q_heads ({num_q_heads}) must be divisible by \
             num_kv_heads ({num_kv_heads})"
        )));
    }

    let group_size = num_q_heads / num_kv_heads;
    let scale = F::from(1.0 / (head_dim as f64).sqrt())
        .ok_or_else(|| NeuralError::InvalidArchitecture("scale cast".into()))?;

    let mut out = Array::zeros(IxDyn(&[batch, seq, num_q_heads, head_dim]));

    for b in 0..batch {
        for kv_h in 0..num_kv_heads {
            // The range of query heads that share this KV head.
            let q_h_start = kv_h * group_size;
            let q_h_end = q_h_start + group_size;

            for q_h in q_h_start..q_h_end {
                for t in 0..seq {
                    // Compute attention scores q[t] · k[s] for all s.
                    let mut scores = Vec::with_capacity(seq);
                    for s in 0..seq {
                        // Causal mask: future positions are excluded.
                        if causal && s > t {
                            scores.push(F::neg_infinity());
                        } else {
                            let mut dot = F::zero();
                            for d in 0..head_dim {
                                dot += q[[b, t, q_h, d]] * k[[b, s, kv_h, d]];
                            }
                            scores.push(dot * scale);
                        }
                    }
                    softmax_inplace(&mut scores);

                    // Weighted sum of values.
                    for d in 0..head_dim {
                        let mut acc = F::zero();
                        for s in 0..seq {
                            acc += scores[s] * v[[b, s, kv_h, d]];
                        }
                        out[[b, t, q_h, d]] = acc;
                    }
                }
            }
        }
    }
    Ok(out)
}

// ===========================================================================
// 4.  EfficientAttentionLayer  (layer-struct wrapper over the above functions)
// ===========================================================================

/// Which efficient attention algorithm to use inside `EfficientAttentionLayer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfficientAttentionKind {
    /// Sliding-window local attention.
    SlidingWindow {
        /// One-sided left radius.
        left_radius: usize,
        /// One-sided right radius (0 = causal).
        right_radius: usize,
    },
    /// Causal linear attention (ELU+1 feature map, O(N·d²)).
    CausalLinear,
    /// Grouped query attention with `kv_heads` KV heads.
    GroupedQuery {
        /// Number of KV heads (must divide num_q_heads evenly).
        kv_heads: usize,
        /// Apply causal masking.
        causal: bool,
    },
}

/// Configuration for [`EfficientAttentionLayer`].
#[derive(Debug, Clone)]
pub struct EfficientAttentionConfig {
    /// Number of query heads.
    pub num_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Which efficient algorithm to run.
    pub kind: EfficientAttentionKind,
    /// Numerical stability eps (used by causal linear).
    pub eps: f64,
}

impl Default for EfficientAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            kind: EfficientAttentionKind::SlidingWindow {
                left_radius: 128,
                right_radius: 0,
            },
            eps: 1e-6,
        }
    }
}

/// Layer wrapper that provides a unified [`Layer`] interface around the
/// standalone efficient-attention functions in this module.
///
/// # Input / Output
/// Shape `[batch, seq, d_model]` → `[batch, seq, d_model]`.
#[derive(Debug)]
pub struct EfficientAttentionLayer<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: EfficientAttentionConfig,
    /// Query projection  [d_model, num_heads * head_dim]
    w_q: Arc<RwLock<Array<F, IxDyn>>>,
    /// Key projection    [d_model, num_kv_heads * head_dim]
    w_k: Arc<RwLock<Array<F, IxDyn>>>,
    /// Value projection  [d_model, num_kv_heads * head_dim]
    w_v: Arc<RwLock<Array<F, IxDyn>>>,
    /// Output projection [num_heads * head_dim, d_model]
    w_o: Arc<RwLock<Array<F, IxDyn>>>,
    /// Actual KV head count (may differ from query heads for GQA).
    num_kv_heads: usize,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static>
    EfficientAttentionLayer<F>
{
    /// Create a new `EfficientAttentionLayer`.
    ///
    /// # Errors
    /// Returns an error if `d_model`, `num_heads`, or `head_dim` is zero, or
    /// if the GQA `kv_heads` does not evenly divide `num_heads`.
    pub fn new<R: Rng>(
        d_model: usize,
        config: EfficientAttentionConfig,
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

        let num_kv_heads = match config.kind {
            EfficientAttentionKind::GroupedQuery { kv_heads, .. } => {
                if kv_heads == 0 || kv_heads > config.num_heads {
                    return Err(NeuralError::InvalidArchitecture(
                        "GQA kv_heads must be in [1, num_heads]".into(),
                    ));
                }
                if config.num_heads % kv_heads != 0 {
                    return Err(NeuralError::InvalidArchitecture(format!(
                        "GQA: num_heads ({}) must be divisible by kv_heads ({})",
                        config.num_heads, kv_heads
                    )));
                }
                kv_heads
            }
            _ => config.num_heads,
        };

        let q_inner = config.num_heads * config.head_dim;
        let kv_inner = num_kv_heads * config.head_dim;

        Ok(Self {
            d_model,
            config,
            w_q: Arc::new(RwLock::new(mk_weight(d_model, q_inner, rng)?)),
            w_k: Arc::new(RwLock::new(mk_weight(d_model, kv_inner, rng)?)),
            w_v: Arc::new(RwLock::new(mk_weight(d_model, kv_inner, rng)?)),
            w_o: Arc::new(RwLock::new(mk_weight(q_inner, d_model, rng)?)),
            num_kv_heads,
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Number of KV heads (equals `num_heads` for non-GQA configurations).
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for EfficientAttentionLayer<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "EfficientAttentionLayer: expected 3D [B,S,D], got {}D",
                shape.len()
            )));
        }
        let (batch, seq, d_model) = (shape[0], shape[1], shape[2]);
        if d_model != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "EfficientAttentionLayer: d_model mismatch: expected {}, got {d_model}",
                self.d_model
            )));
        }

        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let q_inner = num_heads * head_dim;
        let kv_inner = self.num_kv_heads * head_dim;
        let eps = F::from(self.config.eps)
            .ok_or_else(|| NeuralError::InvalidArchitecture("eps cast".into()))?;

        let w_q = self.w_q.read().map_err(|_| NeuralError::InferenceError("lock".into()))?;
        let w_k = self.w_k.read().map_err(|_| NeuralError::InferenceError("lock".into()))?;
        let w_v = self.w_v.read().map_err(|_| NeuralError::InferenceError("lock".into()))?;
        let w_o = self.w_o.read().map_err(|_| NeuralError::InferenceError("lock".into()))?;

        // Project to Q, K, V in flat head-concat form.
        let q_flat = batch_proj(input, &w_q, d_model, q_inner)?;  // [B, S, H_q * D]
        let k_flat = batch_proj(input, &w_k, d_model, kv_inner)?; // [B, S, H_kv * D]
        let v_flat = batch_proj(input, &w_v, d_model, kv_inner)?; // [B, S, H_kv * D]

        // Reshape to [B, S, H, D] (4D).
        let mut q_4d = Array::zeros(IxDyn(&[batch, seq, num_heads, head_dim]));
        let mut k_4d = Array::zeros(IxDyn(&[batch, seq, self.num_kv_heads, head_dim]));
        let mut v_4d = Array::zeros(IxDyn(&[batch, seq, self.num_kv_heads, head_dim]));

        for b in 0..batch {
            for t in 0..seq {
                for h in 0..num_heads {
                    for d in 0..head_dim {
                        q_4d[[b, t, h, d]] = q_flat[[b, t, h * head_dim + d]];
                    }
                }
                for h in 0..self.num_kv_heads {
                    for d in 0..head_dim {
                        k_4d[[b, t, h, d]] = k_flat[[b, t, h * head_dim + d]];
                        v_4d[[b, t, h, d]] = v_flat[[b, t, h * head_dim + d]];
                    }
                }
            }
        }

        // Call the appropriate efficient attention function.
        let attn_out = match self.config.kind {
            EfficientAttentionKind::SlidingWindow { left_radius, right_radius } => {
                sliding_window_attention(&q_4d, &k_4d, &v_4d, left_radius, right_radius)?
            }
            EfficientAttentionKind::CausalLinear => {
                causal_linear_attention(&q_4d, &k_4d, &v_4d, eps)?
            }
            EfficientAttentionKind::GroupedQuery { causal, .. } => {
                grouped_query_attention(&q_4d, &k_4d, &v_4d, causal)?
            }
        };

        // Flatten back to [B, S, H_q * D].
        let mut flat = Array::zeros(IxDyn(&[batch, seq, q_inner]));
        for b in 0..batch {
            for t in 0..seq {
                for h in 0..num_heads {
                    for d in 0..head_dim {
                        flat[[b, t, h * head_dim + d]] = attn_out[[b, t, h, d]];
                    }
                }
            }
        }

        // Final output projection.
        batch_proj(&flat, &w_o, q_inner, d_model)
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
        "EfficientAttentionLayer"
    }

    fn parameter_count(&self) -> usize {
        let q_inner = self.config.num_heads * self.config.head_dim;
        let kv_inner = self.num_kv_heads * self.config.head_dim;
        self.d_model * q_inner + 2 * self.d_model * kv_inner + q_inner * self.d_model
    }
}

unsafe impl<F: Float + Debug + Send + Sync + NumAssign> Send for EfficientAttentionLayer<F> {}
unsafe impl<F: Float + Debug + Send + Sync + NumAssign> Sync for EfficientAttentionLayer<F> {}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array4;
    use scirs2_core::random::rng;

    // ---- sliding_window_attention ----

    #[test]
    fn test_sliding_window_fn_shape() {
        // [B=2, S=6, H=2, D=8]
        let q = Array4::<f64>::from_elem((2, 6, 2, 8), 0.1).into_dyn();
        let k = q.clone();
        let v = q.clone();
        let out = sliding_window_attention(&q, &k, &v, 2, 2).expect("failed");
        assert_eq!(out.shape(), &[2, 6, 2, 8]);
    }

    #[test]
    fn test_sliding_window_fn_causal() {
        let q = Array4::<f64>::from_elem((1, 8, 2, 8), 0.2).into_dyn();
        let k = q.clone();
        let v = q.clone();
        // right_radius = 0 → causal
        let out = sliding_window_attention(&q, &k, &v, 3, 0).expect("failed");
        assert_eq!(out.shape(), &[1, 8, 2, 8]);
    }

    #[test]
    fn test_sliding_window_fn_finite() {
        let q = Array4::<f64>::from_elem((1, 4, 2, 4), 0.3).into_dyn();
        let k = q.clone();
        let v = q.clone();
        let out = sliding_window_attention(&q, &k, &v, 1, 1).expect("failed");
        for v in out.iter() {
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }

    #[test]
    fn test_sliding_window_fn_wrong_rank() {
        let q = scirs2_core::ndarray::Array3::<f64>::from_elem((1, 4, 8), 0.1).into_dyn();
        let k = q.clone();
        let v = q.clone();
        assert!(sliding_window_attention(&q, &k, &v, 1, 1).is_err());
    }

    // ---- causal_linear_attention ----

    #[test]
    fn test_causal_linear_fn_shape() {
        let q = Array4::<f64>::from_elem((2, 6, 2, 8), 0.1).into_dyn();
        let k = q.clone();
        let v = q.clone();
        let out = causal_linear_attention(&q, &k, &v, 1e-6).expect("failed");
        assert_eq!(out.shape(), &[2, 6, 2, 8]);
    }

    #[test]
    fn test_causal_linear_fn_finite() {
        let q = Array4::<f64>::from_elem((1, 5, 2, 4), 0.1).into_dyn();
        let k = q.clone();
        let v = q.clone();
        let out = causal_linear_attention(&q, &k, &v, 1e-6).expect("failed");
        for v in out.iter() {
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }

    #[test]
    fn test_causal_linear_fn_wrong_rank() {
        let q = scirs2_core::ndarray::Array3::<f64>::from_elem((1, 4, 8), 0.1).into_dyn();
        let k = q.clone();
        let v = q.clone();
        assert!(causal_linear_attention(&q, &k, &v, 1e-6).is_err());
    }

    // ---- grouped_query_attention ----

    #[test]
    fn test_gqa_fn_basic() {
        // 4 Q heads, 2 KV heads (group_size = 2)
        let q = Array4::<f64>::from_elem((2, 5, 4, 8), 0.1).into_dyn();
        let k = Array4::<f64>::from_elem((2, 5, 2, 8), 0.1).into_dyn();
        let v = Array4::<f64>::from_elem((2, 5, 2, 8), 0.1).into_dyn();
        let out = grouped_query_attention(&q, &k, &v, false).expect("failed");
        assert_eq!(out.shape(), &[2, 5, 4, 8]);
    }

    #[test]
    fn test_gqa_fn_causal() {
        let q = Array4::<f64>::from_elem((1, 6, 4, 8), 0.2).into_dyn();
        let k = Array4::<f64>::from_elem((1, 6, 2, 8), 0.2).into_dyn();
        let v = Array4::<f64>::from_elem((1, 6, 2, 8), 0.2).into_dyn();
        let out = grouped_query_attention(&q, &k, &v, true).expect("failed");
        assert_eq!(out.shape(), &[1, 6, 4, 8]);
    }

    #[test]
    fn test_gqa_fn_mha_mode() {
        // 1 Q head per KV head = standard MHA
        let q = Array4::<f64>::from_elem((1, 4, 2, 8), 0.1).into_dyn();
        let k = Array4::<f64>::from_elem((1, 4, 2, 8), 0.1).into_dyn();
        let v = Array4::<f64>::from_elem((1, 4, 2, 8), 0.1).into_dyn();
        let out = grouped_query_attention(&q, &k, &v, false).expect("failed");
        assert_eq!(out.shape(), &[1, 4, 2, 8]);
    }

    #[test]
    fn test_gqa_fn_head_mismatch_error() {
        // num_q_heads=3 not divisible by num_kv_heads=2
        let q = Array4::<f64>::from_elem((1, 4, 3, 8), 0.1).into_dyn();
        let k = Array4::<f64>::from_elem((1, 4, 2, 8), 0.1).into_dyn();
        let v = Array4::<f64>::from_elem((1, 4, 2, 8), 0.1).into_dyn();
        assert!(grouped_query_attention(&q, &k, &v, false).is_err());
    }

    #[test]
    fn test_gqa_fn_finite() {
        let q = Array4::<f64>::from_elem((1, 3, 4, 4), 0.1).into_dyn();
        let k = Array4::<f64>::from_elem((1, 3, 2, 4), 0.1).into_dyn();
        let v = Array4::<f64>::from_elem((1, 3, 2, 4), 0.1).into_dyn();
        let out = grouped_query_attention(&q, &k, &v, false).expect("failed");
        for v in out.iter() {
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }

    // ---- EfficientAttentionLayer ----

    #[test]
    fn test_efficient_layer_sliding_window() {
        let mut r = rng();
        let cfg = EfficientAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            kind: EfficientAttentionKind::SlidingWindow { left_radius: 2, right_radius: 2 },
            eps: 1e-6,
        };
        let layer = EfficientAttentionLayer::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = scirs2_core::ndarray::Array3::<f64>::from_elem((2, 6, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 6, 16]);
        assert_eq!(layer.layer_type(), "EfficientAttentionLayer");
    }

    #[test]
    fn test_efficient_layer_causal_linear() {
        let mut r = rng();
        let cfg = EfficientAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            kind: EfficientAttentionKind::CausalLinear,
            eps: 1e-6,
        };
        let layer = EfficientAttentionLayer::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = scirs2_core::ndarray::Array3::<f64>::from_elem((1, 5, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 5, 16]);
    }

    #[test]
    fn test_efficient_layer_gqa() {
        let mut r = rng();
        let cfg = EfficientAttentionConfig {
            num_heads: 4,
            head_dim: 8,
            kind: EfficientAttentionKind::GroupedQuery { kv_heads: 2, causal: false },
            eps: 1e-6,
        };
        let layer = EfficientAttentionLayer::<f64>::new(32, cfg, &mut r).expect("create failed");
        assert_eq!(layer.num_kv_heads(), 2);
        let input = scirs2_core::ndarray::Array3::<f64>::from_elem((2, 4, 32), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_efficient_layer_gqa_bad_divisor() {
        let mut r = rng();
        // num_heads=3 not divisible by kv_heads=2
        let cfg = EfficientAttentionConfig {
            num_heads: 3,
            head_dim: 8,
            kind: EfficientAttentionKind::GroupedQuery { kv_heads: 2, causal: false },
            eps: 1e-6,
        };
        assert!(EfficientAttentionLayer::<f64>::new(24, cfg, &mut r).is_err());
    }

    #[test]
    fn test_efficient_layer_output_finite() {
        let mut r = rng();
        let cfg = EfficientAttentionConfig {
            num_heads: 2,
            head_dim: 4,
            kind: EfficientAttentionKind::SlidingWindow { left_radius: 1, right_radius: 1 },
            eps: 1e-6,
        };
        let layer = EfficientAttentionLayer::<f64>::new(8, cfg, &mut r).expect("create failed");
        let input = scirs2_core::ndarray::Array3::<f64>::from_elem((1, 4, 8), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        for v in out.iter() {
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }
}
