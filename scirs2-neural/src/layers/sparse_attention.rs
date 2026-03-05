//! Sparse and structured attention mechanisms
//!
//! This module provides attention variants that avoid the full O(n²) attention
//! matrix through sparsity or structural decompositions:
//!
//! - **`LocalAttention`** – Sliding window: each token attends only to its
//!   immediate `window_size` neighbourhood (Luong et al., 2015; Beltagy et al.,
//!   2020).
//! - **`BigBirdAttention`** – Combines random + local + global attention
//!   (Zaheer et al., 2020) for O(n) complexity while preserving theoretical
//!   expressiveness of full attention.
//! - **`AxialAttention`** – Factored 2-D attention: process row- and
//!   column-slices of a spatial grid independently (Ho et al., 2019).
//! - **`FlashAttentionSimple`** – Sequential, tile-based computation that
//!   avoids materialising the full N×N attention matrix using the online-
//!   softmax trick (Dao et al., 2022).

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::Rng;
use std::fmt::Debug;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn xavier_vec<F: Float, R: Rng>(
    fan_in: usize,
    fan_out: usize,
    count: usize,
    rng: &mut R,
) -> Result<Vec<F>> {
    let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        let x: f64 = rng.random_range(-1.0..1.0);
        let f = F::from(x * scale)
            .ok_or_else(|| NeuralError::InvalidArchitecture("xavier_vec cast".into()))?;
        v.push(f);
    }
    Ok(v)
}

fn mk_weight<F: Float, R: Rng>(rows: usize, cols: usize, rng: &mut R) -> Result<Array<F, IxDyn>> {
    let data = xavier_vec(rows, cols, rows * cols, rng)?;
    Array::from_shape_vec(IxDyn(&[rows, cols]), data)
        .map_err(|e| NeuralError::InvalidArchitecture(format!("mk_weight: {e}")))
}

/// Dense batch projection `[B, S, D_in] @ [D_in, D_out] -> [B, S, D_out]`.
fn batch_linear<F: Float + NumAssign>(
    x: &Array<F, IxDyn>,
    w: &Array<F, IxDyn>,
    d_in: usize,
    d_out: usize,
) -> Result<Array<F, IxDyn>> {
    let s = x.shape();
    let batch = s[0];
    let seq = s[1];
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

/// Softmax over a mutable slice in place.
fn softmax_slice_inplace<F: Float + NumAssign>(slice: &mut [F]) {
    let max_v = slice
        .iter()
        .fold(F::neg_infinity(), |a, &b| if b > a { b } else { a });
    let mut sum = F::zero();
    for s in slice.iter_mut() {
        *s = (*s - max_v).exp();
        sum += *s;
    }
    let eps = F::from(1e-12_f64).unwrap_or(F::zero());
    let norm = if sum < eps { eps } else { sum };
    for s in slice.iter_mut() {
        *s = *s / norm;
    }
}

// ===========================================================================
// 1.  LocalAttention
// ===========================================================================

/// Configuration for `LocalAttention`.
#[derive(Debug, Clone)]
pub struct LocalAttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Half-window: each token attends to positions `[i-window, i+window]`
    /// (inclusive).  Total window size is `2 * window_size + 1`.
    pub window_size: usize,
    /// If `true`, restrict attention to `[i-window, i]` (causal).
    pub causal: bool,
}

/// Sliding-window (local) attention.
///
/// Each query token `i` computes scaled dot-product attention only over
/// keys `j ∈ [max(0, i-w), min(n-1, i+w)]`, giving O(n · w) complexity
/// instead of O(n²).  `w = window_size`.
///
/// Tokens at the boundary simply have a smaller effective window; no padding
/// is needed because scores outside the sequence are never computed.
#[derive(Debug)]
pub struct LocalAttention<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: LocalAttentionConfig,
    w_query: Array<F, IxDyn>,
    w_key: Array<F, IxDyn>,
    w_value: Array<F, IxDyn>,
    w_output: Array<F, IxDyn>,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> LocalAttention<F> {
    /// Create a new `LocalAttention` layer.
    pub fn new<R: Rng>(d_model: usize, config: LocalAttentionConfig, rng: &mut R) -> Result<Self> {
        let nh = config.num_heads;
        let hd = config.head_dim;
        if d_model != nh * hd {
            return Err(NeuralError::InvalidArchitecture(format!(
                "d_model ({d_model}) must equal num_heads * head_dim ({nh} * {hd})"
            )));
        }
        if config.window_size == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "window_size must be > 0".into(),
            ));
        }
        Ok(Self {
            d_model,
            config,
            w_query: mk_weight(d_model, d_model, rng)?,
            w_key: mk_weight(d_model, d_model, rng)?,
            w_value: mk_weight(d_model, d_model, rng)?,
            w_output: mk_weight(d_model, d_model, rng)?,
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Compute local attention.
    ///
    /// `q`, `k`, `v`: `[batch, seq, nh, hd]`
    ///
    /// Returns `[batch, seq, nh, hd]`.
    pub fn local_attention(
        q: &Array<F, IxDyn>,
        k: &Array<F, IxDyn>,
        v: &Array<F, IxDyn>,
        window_size: usize,
        causal: bool,
    ) -> Result<Array<F, IxDyn>> {
        let s = q.shape();
        let (batch, seq, nh, hd) = (s[0], s[1], s[2], s[3]);
        let scale = F::from(1.0 / (hd as f64).sqrt())
            .ok_or_else(|| NeuralError::InferenceError("scale cast".into()))?;

        let mut out = Array::zeros(IxDyn(&[batch, seq, nh, hd]));

        for b in 0..batch {
            for h in 0..nh {
                for i in 0..seq {
                    let j_start = if i >= window_size { i - window_size } else { 0 };
                    let j_end = if causal {
                        i + 1
                    } else {
                        (i + window_size + 1).min(seq)
                    };
                    let win_len = j_end - j_start;

                    let mut scores = vec![F::zero(); win_len];
                    for (wi, j) in (j_start..j_end).enumerate() {
                        let mut dot = F::zero();
                        for d in 0..hd {
                            dot += q[[b, i, h, d]] * k[[b, j, h, d]];
                        }
                        scores[wi] = dot * scale;
                    }

                    softmax_slice_inplace(&mut scores);

                    for (wi, j) in (j_start..j_end).enumerate() {
                        for d in 0..hd {
                            out[[b, i, h, d]] += scores[wi] * v[[b, j, h, d]];
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for LocalAttention<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "LocalAttention expects 3D input, got {}D",
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

        let q = batch_linear(input, &self.w_query, dm, dm)?;
        let k = batch_linear(input, &self.w_key, dm, dm)?;
        let v = batch_linear(input, &self.w_value, dm, dm)?;

        let q = q.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape q: {e}")))?;
        let k = k.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape k: {e}")))?;
        let v = v.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape v: {e}")))?;

        let attended =
            Self::local_attention(&q, &k, &v, self.config.window_size, self.config.causal)?;

        let concat = attended
            .into_shape_with_order(IxDyn(&[batch, seq, dm]))
            .map_err(|e| NeuralError::InferenceError(format!("concat: {e}")))?;
        batch_linear(&concat, &self.w_output, dm, dm)
    }

    fn backward(&self, _input: &Array<F, IxDyn>, grad: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        Ok(grad.clone())
    }

    fn update(&mut self, _lr: F) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

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

    fn set_training(&mut self, t: bool) { self.training = t; }
    fn is_training(&self) -> bool { self.training }
    fn layer_type(&self) -> &str { "LocalAttention" }
    fn parameter_count(&self) -> usize { 4 * self.d_model * self.d_model }
}

// ===========================================================================
// 2.  BigBirdAttention
// ===========================================================================

/// Configuration for `BigBirdAttention`.
#[derive(Debug, Clone)]
pub struct BigBirdConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Local window half-size: each token attends to `[i-w, i+w]`.
    pub window: usize,
    /// Number of *random* keys each token attends to (sampled once at
    /// construction time; fixed for a given layer instance).
    pub n_random: usize,
    /// Number of *global* tokens prepended (indices `0..n_global`).
    /// Every other token attends to all global tokens and global tokens
    /// attend to all positions.
    pub n_global: usize,
}

/// BigBird attention (Zaheer et al., 2020).
///
/// Three attention types are combined:
/// 1. **Local** – window attention as in `LocalAttention`.
/// 2. **Random** – each token additionally attends to `n_random` randomly
///    sampled key positions (fixed at layer construction time, reproducing
///    the static variant of BigBird).
/// 3. **Global** – the first `n_global` tokens attend to all positions; all
///    tokens attend to the first `n_global` positions.
///
/// This gives O(n) complexity (for fixed `window`, `n_random`, `n_global`)
/// while empirically approaching the modelling power of full attention.
///
/// ### Static random indices
///
/// For reproducibility and inference efficiency the random attention pattern is
/// sampled once during `new()` and stored as a fixed list of `n_random` token
/// indices that every non-global query attends to.  In the original paper a
/// *different* random pattern is sampled per training step; that variant can be
/// approximated by re-creating the layer each step, which is inexpensive.
#[derive(Debug)]
pub struct BigBirdAttention<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: BigBirdConfig,
    w_query: Array<F, IxDyn>,
    w_key: Array<F, IxDyn>,
    w_value: Array<F, IxDyn>,
    w_output: Array<F, IxDyn>,
    /// Fixed random indices attended to by every non-global token.
    /// Shape: `[n_random]` (indices stored as `usize`; capped at max_seq-1).
    random_indices: Vec<usize>,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> BigBirdAttention<F> {
    /// Create a new `BigBirdAttention` layer.
    ///
    /// `max_seq` is used to bound the sampled random indices; it should be at
    /// least as large as the longest sequence this layer will encounter.
    pub fn new<R: Rng>(
        d_model: usize,
        max_seq: usize,
        config: BigBirdConfig,
        rng: &mut R,
    ) -> Result<Self> {
        let nh = config.num_heads;
        let hd = config.head_dim;
        if d_model != nh * hd {
            return Err(NeuralError::InvalidArchitecture(format!(
                "d_model ({d_model}) must equal num_heads * head_dim ({nh} * {hd})"
            )));
        }
        if config.n_global > max_seq {
            return Err(NeuralError::InvalidArchitecture(format!(
                "n_global ({}) > max_seq ({max_seq})",
                config.n_global
            )));
        }

        // Sample fixed random indices in [0, max_seq).
        let n_rand = config.n_random.min(max_seq);
        let mut rand_idx = Vec::with_capacity(n_rand);
        let mut pool: Vec<usize> = (0..max_seq).collect();
        for i in 0..n_rand {
            // Fisher-Yates partial shuffle.
            let remaining = max_seq - i;
            let j = i + (rng.random::<f64>() * remaining as f64) as usize % remaining;
            pool.swap(i, j);
            rand_idx.push(pool[i]);
        }
        rand_idx.sort_unstable();

        Ok(Self {
            d_model,
            config,
            w_query: mk_weight(d_model, d_model, rng)?,
            w_key: mk_weight(d_model, d_model, rng)?,
            w_value: mk_weight(d_model, d_model, rng)?,
            w_output: mk_weight(d_model, d_model, rng)?,
            random_indices: rand_idx,
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Compute BigBird forward pass on projected Q, K, V tensors.
    ///
    /// All tensors have shape `[batch, seq, nh, hd]`.
    pub fn bigbird_forward(
        q: &Array<F, IxDyn>,
        k: &Array<F, IxDyn>,
        v: &Array<F, IxDyn>,
        window: usize,
        global_indices: &[usize],
        random_indices: &[usize],
    ) -> Result<Array<F, IxDyn>> {
        let s = q.shape();
        let (batch, seq, nh, hd) = (s[0], s[1], s[2], s[3]);
        let scale = F::from(1.0 / (hd as f64).sqrt())
            .ok_or_else(|| NeuralError::InferenceError("scale cast".into()))?;

        let mut out = Array::zeros(IxDyn(&[batch, seq, nh, hd]));

        for b in 0..batch {
            for h in 0..nh {
                for i in 0..seq {
                    // Build the set of attended-to positions for token i.
                    let mut attend: Vec<usize> = Vec::new();

                    // 1. Global tokens: every token attends to all global positions.
                    for &g in global_indices {
                        if g < seq {
                            attend.push(g);
                        }
                    }

                    // 2. Local window (skip positions already included via global).
                    let j_start = if i >= window { i - window } else { 0 };
                    let j_end = (i + window + 1).min(seq);
                    for j in j_start..j_end {
                        attend.push(j);
                    }

                    // 3. Random positions (skip out-of-range).
                    for &r in random_indices {
                        if r < seq {
                            attend.push(r);
                        }
                    }

                    // If this token is itself a global token, attend to all.
                    let is_global = global_indices.contains(&i);
                    if is_global {
                        attend = (0..seq).collect();
                    }

                    // Deduplicate while preserving order.
                    attend.sort_unstable();
                    attend.dedup();

                    let win_len = attend.len();
                    if win_len == 0 {
                        continue;
                    }

                    // Compute scores over the attended set.
                    let mut scores = vec![F::zero(); win_len];
                    for (wi, &j) in attend.iter().enumerate() {
                        let mut dot = F::zero();
                        for d in 0..hd {
                            dot += q[[b, i, h, d]] * k[[b, j, h, d]];
                        }
                        scores[wi] = dot * scale;
                    }

                    softmax_slice_inplace(&mut scores);

                    for (wi, &j) in attend.iter().enumerate() {
                        for d in 0..hd {
                            out[[b, i, h, d]] += scores[wi] * v[[b, j, h, d]];
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for BigBirdAttention<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "BigBirdAttention expects 3D input, got {}D",
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
        let ng = self.config.n_global;

        let q = batch_linear(input, &self.w_query, dm, dm)?;
        let k = batch_linear(input, &self.w_key, dm, dm)?;
        let v = batch_linear(input, &self.w_value, dm, dm)?;

        let q = q.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape q: {e}")))?;
        let k = k.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape k: {e}")))?;
        let v = v.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape v: {e}")))?;

        // Global indices: first ng positions.
        let actual_ng = ng.min(seq);
        let global_idx: Vec<usize> = (0..actual_ng).collect();

        // Clip random indices to actual sequence length.
        let rand_idx: Vec<usize> = self
            .random_indices
            .iter()
            .filter(|&&r| r < seq)
            .copied()
            .collect();

        let attended = Self::bigbird_forward(
            &q,
            &k,
            &v,
            self.config.window,
            &global_idx,
            &rand_idx,
        )?;

        let concat = attended
            .into_shape_with_order(IxDyn(&[batch, seq, dm]))
            .map_err(|e| NeuralError::InferenceError(format!("concat: {e}")))?;
        batch_linear(&concat, &self.w_output, dm, dm)
    }

    fn backward(&self, _input: &Array<F, IxDyn>, grad: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        Ok(grad.clone())
    }

    fn update(&mut self, _lr: F) -> Result<()> { Ok(()) }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

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

    fn set_training(&mut self, t: bool) { self.training = t; }
    fn is_training(&self) -> bool { self.training }
    fn layer_type(&self) -> &str { "BigBirdAttention" }
    fn parameter_count(&self) -> usize { 4 * self.d_model * self.d_model }
}

// ===========================================================================
// 3.  AxialAttention
// ===========================================================================

/// Configuration for `AxialAttention`.
#[derive(Debug, Clone)]
pub struct AxialAttentionConfig {
    /// Number of attention heads for both row and column attention.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Whether to apply causal masking within each axis.
    pub causal: bool,
}

/// Axial (factored 2-D) attention (Ho et al., 2019; Huang et al., 2019).
///
/// A 2-D spatial sequence of shape `(H, W)` is processed by applying
/// standard scaled dot-product attention along each axis independently:
///
/// 1. **Row attention** – for each row `h`, tokens in that row attend to each
///    other.  The sequence of length `W` is attended over per row.
/// 2. **Column attention** – for each column `w`, tokens in that column attend
///    to each other.  The sequence of length `H` is attended over per column.
///
/// The outputs of both passes are averaged to combine both axes.
///
/// Because each pass is O(H · W²) or O(W · H²), and we alternate, total
/// cost is O(H · W · (H + W)) ≈ O(n · √n) for square inputs, a significant
/// saving over O(n²) full attention.
///
/// ### Input format
///
/// The `forward()` method accepts 3-D inputs `[batch, height * width, d_model]`
/// (i.e. the spatial grid has been flattened into a sequence).  `height` and
/// `width` must be provided to `axial_attention_2d()` to reconstruct the grid.
#[derive(Debug)]
pub struct AxialAttention<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: AxialAttentionConfig,
    /// Separate projection weights for row and column sub-attentions.
    w_q_row: Array<F, IxDyn>,
    w_k_row: Array<F, IxDyn>,
    w_v_row: Array<F, IxDyn>,
    w_q_col: Array<F, IxDyn>,
    w_k_col: Array<F, IxDyn>,
    w_v_col: Array<F, IxDyn>,
    w_output: Array<F, IxDyn>,
    /// Stored `height` for the `Layer::forward` path (set by the user or
    /// during `new`; can be updated via `set_spatial_dims`).
    height: usize,
    /// Stored `width`.
    width: usize,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> AxialAttention<F> {
    /// Create a new `AxialAttention` layer.
    ///
    /// `height` and `width` define the spatial grid; their product must equal
    /// the sequence length presented at forward time.
    pub fn new<R: Rng>(
        d_model: usize,
        height: usize,
        width: usize,
        config: AxialAttentionConfig,
        rng: &mut R,
    ) -> Result<Self> {
        let nh = config.num_heads;
        let hd = config.head_dim;
        if d_model != nh * hd {
            return Err(NeuralError::InvalidArchitecture(format!(
                "d_model ({d_model}) must equal num_heads * head_dim ({nh} * {hd})"
            )));
        }
        if height == 0 || width == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "height and width must both be > 0".into(),
            ));
        }

        Ok(Self {
            d_model,
            config,
            w_q_row: mk_weight(d_model, d_model, rng)?,
            w_k_row: mk_weight(d_model, d_model, rng)?,
            w_v_row: mk_weight(d_model, d_model, rng)?,
            w_q_col: mk_weight(d_model, d_model, rng)?,
            w_k_col: mk_weight(d_model, d_model, rng)?,
            w_v_col: mk_weight(d_model, d_model, rng)?,
            w_output: mk_weight(d_model, d_model, rng)?,
            height,
            width,
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Update the spatial dimensions used by `Layer::forward`.
    pub fn set_spatial_dims(&mut self, height: usize, width: usize) {
        self.height = height;
        self.width = width;
    }

    /// Scaled dot-product attention on a `[batch, seq, nh, hd]` tensor.
    ///
    /// Returns `[batch, seq, nh, hd]`.
    fn sdpa(
        q: &Array<F, IxDyn>,
        k: &Array<F, IxDyn>,
        v: &Array<F, IxDyn>,
        causal: bool,
    ) -> Result<Array<F, IxDyn>> {
        let qs = q.shape();
        let (batch, seq, nh, hd) = (qs[0], qs[1], qs[2], qs[3]);
        let scale = F::from(1.0 / (hd as f64).sqrt())
            .ok_or_else(|| NeuralError::InferenceError("scale cast".into()))?;
        let neg_inf = F::neg_infinity();

        let mut out = Array::zeros(IxDyn(&[batch, seq, nh, hd]));
        for b in 0..batch {
            for h in 0..nh {
                // scores: [seq x seq]
                let mut scores = vec![F::zero(); seq * seq];
                for i in 0..seq {
                    for j in 0..seq {
                        let mut dot = F::zero();
                        for d in 0..hd {
                            dot += q[[b, i, h, d]] * k[[b, j, h, d]];
                        }
                        let s = dot * scale;
                        scores[i * seq + j] = if causal && j > i { neg_inf } else { s };
                    }
                }
                // softmax per row
                for i in 0..seq {
                    softmax_slice_inplace(&mut scores[i * seq..(i + 1) * seq]);
                }
                // weighted sum
                for i in 0..seq {
                    for d in 0..hd {
                        let mut acc = F::zero();
                        for j in 0..seq {
                            acc += scores[i * seq + j] * v[[b, j, h, d]];
                        }
                        out[[b, i, h, d]] = acc;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Factored 2-D axial attention.
    ///
    /// `x`:  `[batch, height * width, d_model]`
    ///
    /// Returns `[batch, height * width, d_model]`.
    ///
    /// Internally:
    /// - Row attention: reshape to `[batch * height, width, d_model]`, attend.
    /// - Column attention: transpose to `[batch * width, height, d_model]`, attend.
    /// - Average and project.
    pub fn axial_attention_2d(
        &self,
        x: &Array<F, IxDyn>,
        height: usize,
        width: usize,
    ) -> Result<Array<F, IxDyn>> {
        let xs = x.shape();
        let (batch, seq, dm) = (xs[0], xs[1], xs[2]);
        if seq != height * width {
            return Err(NeuralError::InferenceError(format!(
                "axial_attention_2d: seq ({seq}) != height * width ({height} * {width})"
            )));
        }
        if dm != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "d_model mismatch: expected {}, got {dm}",
                self.d_model
            )));
        }

        let nh = self.config.num_heads;
        let hd = self.config.head_dim;
        let causal = self.config.causal;

        // --- Row attention ---
        // Reshape: [batch, height, width, dm] -> [batch * height, width, dm]
        let x_rows = x.clone()
            .into_shape_with_order(IxDyn(&[batch * height, width, dm]))
            .map_err(|e| NeuralError::InferenceError(format!("row reshape: {e}")))?;

        let q_r = batch_linear(&x_rows, &self.w_q_row, dm, dm)?;
        let k_r = batch_linear(&x_rows, &self.w_k_row, dm, dm)?;
        let v_r = batch_linear(&x_rows, &self.w_v_row, dm, dm)?;

        let q_r = q_r.into_shape_with_order(IxDyn(&[batch * height, width, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("q_r reshape: {e}")))?;
        let k_r = k_r.into_shape_with_order(IxDyn(&[batch * height, width, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("k_r reshape: {e}")))?;
        let v_r = v_r.into_shape_with_order(IxDyn(&[batch * height, width, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("v_r reshape: {e}")))?;

        let row_out = Self::sdpa(&q_r, &k_r, &v_r, causal)?
            .into_shape_with_order(IxDyn(&[batch, height, width, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("row_out reshape1: {e}")))?
            .into_shape_with_order(IxDyn(&[batch, height * width, dm]))
            .map_err(|e| NeuralError::InferenceError(format!("row_out reshape2: {e}")))?;

        // --- Column attention ---
        // We need to permute to [batch, width, height, dm] so that each column
        // (fixed width index, varying height) becomes a length-height sequence.
        // We do this by manual reindexing into a new array.
        let mut x_cols = Array::zeros(IxDyn(&[batch * width, height, dm]));
        for b in 0..batch {
            for w_idx in 0..width {
                for h_idx in 0..height {
                    let flat = h_idx * width + w_idx; // original flat index
                    for d in 0..dm {
                        x_cols[[b * width + w_idx, h_idx, d]] = x[[b, flat, d]];
                    }
                }
            }
        }

        let q_c = batch_linear(&x_cols, &self.w_q_col, dm, dm)?;
        let k_c = batch_linear(&x_cols, &self.w_k_col, dm, dm)?;
        let v_c = batch_linear(&x_cols, &self.w_v_col, dm, dm)?;

        let q_c = q_c.into_shape_with_order(IxDyn(&[batch * width, height, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("q_c reshape: {e}")))?;
        let k_c = k_c.into_shape_with_order(IxDyn(&[batch * width, height, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("k_c reshape: {e}")))?;
        let v_c = v_c.into_shape_with_order(IxDyn(&[batch * width, height, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("v_c reshape: {e}")))?;

        let col_attended = Self::sdpa(&q_c, &k_c, &v_c, causal)?;
        // col_attended: [batch * width, height, nh, hd]
        let col_attended = col_attended
            .into_shape_with_order(IxDyn(&[batch * width, height, dm]))
            .map_err(|e| NeuralError::InferenceError(format!("col_attended flat: {e}")))?;

        // Undo the column permutation -> back to [batch, height * width, dm]
        let mut col_out = Array::zeros(IxDyn(&[batch, height * width, dm]));
        for b in 0..batch {
            for w_idx in 0..width {
                for h_idx in 0..height {
                    let flat = h_idx * width + w_idx;
                    for d in 0..dm {
                        col_out[[b, flat, d]] =
                            col_attended[[b * width + w_idx, h_idx, d]];
                    }
                }
            }
        }

        // Average row and column outputs, then project.
        let half = F::from(0.5_f64).unwrap_or_else(|| F::one());
        let mut combined = Array::zeros(IxDyn(&[batch, seq, dm]));
        for b in 0..batch {
            for t in 0..seq {
                for d in 0..dm {
                    combined[[b, t, d]] =
                        (row_out[[b, t, d]] + col_out[[b, t, d]]) * half;
                }
            }
        }

        batch_linear(&combined, &self.w_output, dm, dm)
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for AxialAttention<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "AxialAttention expects 3D input [batch, H*W, d_model], got {}D",
                shape.len()
            )));
        }
        let seq = shape[1];
        let expected = self.height * self.width;
        if seq != expected {
            return Err(NeuralError::InferenceError(format!(
                "seq ({seq}) != height * width ({} * {}) = {expected}",
                self.height, self.width
            )));
        }
        self.axial_attention_2d(input, self.height, self.width)
    }

    fn backward(&self, _input: &Array<F, IxDyn>, grad: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        Ok(grad.clone())
    }

    fn update(&mut self, _lr: F) -> Result<()> { Ok(()) }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        vec![
            self.w_q_row.clone(),
            self.w_k_row.clone(),
            self.w_v_row.clone(),
            self.w_q_col.clone(),
            self.w_k_col.clone(),
            self.w_v_col.clone(),
            self.w_output.clone(),
        ]
    }

    fn set_params(&mut self, p: &[Array<F, IxDyn>]) -> Result<()> {
        if p.len() >= 7 {
            self.w_q_row = p[0].clone();
            self.w_k_row = p[1].clone();
            self.w_v_row = p[2].clone();
            self.w_q_col = p[3].clone();
            self.w_k_col = p[4].clone();
            self.w_v_col = p[5].clone();
            self.w_output = p[6].clone();
        }
        Ok(())
    }

    fn set_training(&mut self, t: bool) { self.training = t; }
    fn is_training(&self) -> bool { self.training }
    fn layer_type(&self) -> &str { "AxialAttention" }
    fn parameter_count(&self) -> usize { 7 * self.d_model * self.d_model }
}

// ===========================================================================
// 4.  FlashAttentionSimple
// ===========================================================================

/// Configuration for `FlashAttentionSimple`.
#[derive(Debug, Clone)]
pub struct FlashAttentionSimpleConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Tile (block) size along the sequence dimension.  Larger tiles use
    /// more temporary memory but may be more cache-efficient.
    pub tile_size: usize,
    /// Whether to apply causal masking.
    pub causal: bool,
}

impl Default for FlashAttentionSimpleConfig {
    fn default() -> Self {
        Self {
            num_heads: 4,
            head_dim: 16,
            tile_size: 32,
            causal: false,
        }
    }
}

/// Memory-efficient tiled attention (sequential Flash Attention).
///
/// This layer computes *exact* scaled dot-product attention without ever
/// materialising the full N×N score matrix.  Instead, an *online softmax*
/// algorithm (Milakov & Gimelshein, 2018) is used: for each query tile, we
/// iterate over all key/value tiles and accumulate:
///
/// ```text
/// For each query row i:
///   m_i = running maximum of scores  (for numerical stability)
///   ℓ_i = running sum of exp(scores − m_i)
///   O_i = running weighted sum
///
/// On each new KV tile:
///   m_new = max(m_old, block_max)
///   ℓ_new = ℓ_old * exp(m_old - m_new) + Σ exp(s_ij - m_new)
///   O_new = O_old * exp(m_old - m_new) + Σ exp(s_ij - m_new) * V_j
///
/// Final: O_i /= ℓ_i
/// ```
///
/// **Memory**: O(tile² + n · d) instead of O(n²).
/// **Compute**: Identical to standard attention for the same sequence.
///
/// This is the *sequential* (CPU-friendly) variant; the GPU variant in the
/// original Flash Attention paper parallelises over tiles.
#[derive(Debug)]
pub struct FlashAttentionSimple<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: FlashAttentionSimpleConfig,
    w_query: Array<F, IxDyn>,
    w_key: Array<F, IxDyn>,
    w_value: Array<F, IxDyn>,
    w_output: Array<F, IxDyn>,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static>
    FlashAttentionSimple<F>
{
    /// Create a new `FlashAttentionSimple` layer.
    pub fn new<R: Rng>(
        d_model: usize,
        config: FlashAttentionSimpleConfig,
        rng: &mut R,
    ) -> Result<Self> {
        let nh = config.num_heads;
        let hd = config.head_dim;
        if d_model != nh * hd {
            return Err(NeuralError::InvalidArchitecture(format!(
                "d_model ({d_model}) must equal num_heads * head_dim ({nh} * {hd})"
            )));
        }
        if config.tile_size == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "tile_size must be > 0".into(),
            ));
        }

        Ok(Self {
            d_model,
            config,
            w_query: mk_weight(d_model, d_model, rng)?,
            w_key: mk_weight(d_model, d_model, rng)?,
            w_value: mk_weight(d_model, d_model, rng)?,
            w_output: mk_weight(d_model, d_model, rng)?,
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Tiled online-softmax attention for a single head.
    ///
    /// `q`, `k`, `v`: `[seq, hd]` (2-D slices for one head).
    ///
    /// Returns `[seq, hd]`.
    fn tiled_attention_head(
        q: &[Vec<F>],
        k: &[Vec<F>],
        v: &[Vec<F>],
        seq_q: usize,
        seq_kv: usize,
        hd: usize,
        tile: usize,
        causal: bool,
        scale: F,
    ) -> Vec<F> {
        let mut output = vec![F::zero(); seq_q * hd];
        let mut row_max = vec![F::neg_infinity(); seq_q];
        let mut row_sum = vec![F::zero(); seq_q];

        let neg_inf = F::neg_infinity();

        let n_q_tiles = seq_q.div_ceil(tile);
        let n_kv_tiles = seq_kv.div_ceil(tile);

        for qi in 0..n_q_tiles {
            let q_start = qi * tile;
            let q_end = (q_start + tile).min(seq_q);

            for kvi in 0..n_kv_tiles {
                let kv_start = kvi * tile;
                let kv_end = (kv_start + tile).min(seq_kv);

                // Skip entirely-masked KV tiles for causal attention.
                if causal && kv_start > q_end.saturating_sub(1) {
                    continue;
                }

                // Compute scores for this tile: [q_tile x kv_tile]
                let q_tile_size = q_end - q_start;
                let kv_tile_size = kv_end - kv_start;
                let mut scores = vec![F::zero(); q_tile_size * kv_tile_size];

                for local_i in 0..q_tile_size {
                    let global_i = q_start + local_i;
                    for local_j in 0..kv_tile_size {
                        let global_j = kv_start + local_j;
                        let s = if causal && global_j > global_i {
                            neg_inf
                        } else {
                            let dot: F = (0..hd)
                                .map(|d| q[global_i][d] * k[global_j][d])
                                .fold(F::zero(), |a, b| a + b);
                            dot * scale
                        };
                        scores[local_i * kv_tile_size + local_j] = s;
                    }
                }

                // Online softmax update per query row.
                for local_i in 0..q_tile_size {
                    let global_i = q_start + local_i;

                    // Block max.
                    let block_max = scores[local_i * kv_tile_size..(local_i + 1) * kv_tile_size]
                        .iter()
                        .fold(neg_inf, |a, &b| if b > a { b } else { a });

                    let old_max = row_max[global_i];
                    let new_max = if old_max > block_max { old_max } else { block_max };

                    // Correction factor for previous accumulations.
                    let correction = if old_max == neg_inf {
                        F::zero()
                    } else {
                        (old_max - new_max).exp()
                    };

                    // Rescale previous output and sum.
                    for d in 0..hd {
                        output[global_i * hd + d] = output[global_i * hd + d] * correction;
                    }
                    row_sum[global_i] = row_sum[global_i] * correction;

                    // Accumulate new contributions.
                    for local_j in 0..kv_tile_size {
                        let raw_score = scores[local_i * kv_tile_size + local_j];
                        if raw_score > neg_inf {
                            let global_j = kv_start + local_j;
                            let exp_s = (raw_score - new_max).exp();
                            row_sum[global_i] = row_sum[global_i] + exp_s;
                            for d in 0..hd {
                                output[global_i * hd + d] =
                                    output[global_i * hd + d] + exp_s * v[global_j][d];
                            }
                        }
                    }

                    row_max[global_i] = new_max;
                }
            }
        }

        // Final normalisation.
        for i in 0..seq_q {
            let s = row_sum[i];
            if s > F::zero() {
                let inv = F::one() / s;
                for d in 0..hd {
                    output[i * hd + d] = output[i * hd + d] * inv;
                }
            }
        }

        output
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for FlashAttentionSimple<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "FlashAttentionSimple expects 3D input, got {}D",
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
        let tile = self.config.tile_size;
        let causal = self.config.causal;
        let scale = F::from(1.0 / (hd as f64).sqrt())
            .ok_or_else(|| NeuralError::InferenceError("scale cast".into()))?;

        let q = batch_linear(input, &self.w_query, dm, dm)?;
        let k = batch_linear(input, &self.w_key, dm, dm)?;
        let v = batch_linear(input, &self.w_value, dm, dm)?;

        let q = q.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape q: {e}")))?;
        let k = k.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape k: {e}")))?;
        let v = v.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape v: {e}")))?;

        let mut attended = Array::zeros(IxDyn(&[batch, seq, nh, hd]));

        for b in 0..batch {
            for h in 0..nh {
                // Extract per-head slices as Vec<Vec<F>> for the tiled kernel.
                let q_head: Vec<Vec<F>> = (0..seq)
                    .map(|t| (0..hd).map(|d| q[[b, t, h, d]]).collect())
                    .collect();
                let k_head: Vec<Vec<F>> = (0..seq)
                    .map(|t| (0..hd).map(|d| k[[b, t, h, d]]).collect())
                    .collect();
                let v_head: Vec<Vec<F>> = (0..seq)
                    .map(|t| (0..hd).map(|d| v[[b, t, h, d]]).collect())
                    .collect();

                let result = Self::tiled_attention_head(
                    &q_head, &k_head, &v_head,
                    seq, seq, hd,
                    tile, causal, scale,
                );

                for i in 0..seq {
                    for d in 0..hd {
                        attended[[b, i, h, d]] = result[i * hd + d];
                    }
                }
            }
        }

        let concat = attended
            .into_shape_with_order(IxDyn(&[batch, seq, dm]))
            .map_err(|e| NeuralError::InferenceError(format!("concat: {e}")))?;
        batch_linear(&concat, &self.w_output, dm, dm)
    }

    fn backward(&self, _input: &Array<F, IxDyn>, grad: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        Ok(grad.clone())
    }

    fn update(&mut self, _lr: F) -> Result<()> { Ok(()) }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

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

    fn set_training(&mut self, t: bool) { self.training = t; }
    fn is_training(&self) -> bool { self.training }
    fn layer_type(&self) -> &str { "FlashAttentionSimple" }
    fn parameter_count(&self) -> usize { 4 * self.d_model * self.d_model }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;
    use scirs2_core::random::rng;

    // ---- LocalAttention ----

    #[test]
    fn test_local_attention_creation() {
        let mut r = rng();
        let cfg = LocalAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            window_size: 3,
            causal: false,
        };
        let layer = LocalAttention::<f64>::new(16, cfg, &mut r).expect("create failed");
        assert_eq!(layer.layer_type(), "LocalAttention");
    }

    #[test]
    fn test_local_attention_forward() {
        let mut r = rng();
        let cfg = LocalAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            window_size: 2,
            causal: false,
        };
        let layer = LocalAttention::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((2, 6, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 6, 16]);
    }

    #[test]
    fn test_local_attention_causal() {
        let mut r = rng();
        let cfg = LocalAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            window_size: 3,
            causal: true,
        };
        let layer = LocalAttention::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 8, 16), 0.2).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 8, 16]);
    }

    #[test]
    fn test_local_attention_zero_window_error() {
        let mut r = rng();
        let cfg = LocalAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            window_size: 0,
            causal: false,
        };
        assert!(LocalAttention::<f64>::new(16, cfg, &mut r).is_err());
    }

    #[test]
    fn test_local_attention_output_finite() {
        let mut r = rng();
        let cfg = LocalAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            window_size: 2,
            causal: false,
        };
        let layer = LocalAttention::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 5, 16), 0.3).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        for v in out.iter() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    // ---- BigBirdAttention ----

    #[test]
    fn test_bigbird_creation() {
        let mut r = rng();
        let cfg = BigBirdConfig {
            num_heads: 2,
            head_dim: 8,
            window: 2,
            n_random: 3,
            n_global: 1,
        };
        let layer = BigBirdAttention::<f64>::new(16, 32, cfg, &mut r).expect("create failed");
        assert_eq!(layer.layer_type(), "BigBirdAttention");
    }

    #[test]
    fn test_bigbird_forward() {
        let mut r = rng();
        let cfg = BigBirdConfig {
            num_heads: 2,
            head_dim: 8,
            window: 2,
            n_random: 3,
            n_global: 2,
        };
        let layer = BigBirdAttention::<f64>::new(16, 16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((2, 8, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 8, 16]);
    }

    #[test]
    fn test_bigbird_output_finite() {
        let mut r = rng();
        let cfg = BigBirdConfig {
            num_heads: 2,
            head_dim: 8,
            window: 2,
            n_random: 2,
            n_global: 1,
        };
        let layer = BigBirdAttention::<f64>::new(16, 16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 8, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        for v in out.iter() {
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }

    // ---- AxialAttention ----

    #[test]
    fn test_axial_attention_creation() {
        let mut r = rng();
        let cfg = AxialAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            causal: false,
        };
        let layer = AxialAttention::<f64>::new(16, 4, 4, cfg, &mut r).expect("create failed");
        assert_eq!(layer.layer_type(), "AxialAttention");
        assert_eq!(layer.parameter_count(), 7 * 16 * 16);
    }

    #[test]
    fn test_axial_attention_forward() {
        let mut r = rng();
        let cfg = AxialAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            causal: false,
        };
        let layer = AxialAttention::<f64>::new(16, 3, 4, cfg, &mut r).expect("create failed");
        // [batch=2, 3*4=12, d_model=16]
        let input = Array3::<f64>::from_elem((2, 12, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 12, 16]);
    }

    #[test]
    fn test_axial_attention_2d_direct() {
        let mut r = rng();
        let cfg = AxialAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            causal: false,
        };
        let layer = AxialAttention::<f64>::new(16, 2, 3, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 6, 16), 0.2).into_dyn();
        let out = layer.axial_attention_2d(&input, 2, 3).expect("2d failed");
        assert_eq!(out.shape(), &[1, 6, 16]);
    }

    #[test]
    fn test_axial_attention_seq_mismatch_error() {
        let mut r = rng();
        let cfg = AxialAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            causal: false,
        };
        let layer = AxialAttention::<f64>::new(16, 4, 4, cfg, &mut r).expect("create failed");
        // seq=6 != 4*4=16
        let input = Array3::<f64>::from_elem((1, 6, 16), 0.1).into_dyn();
        assert!(layer.forward(&input).is_err());
    }

    #[test]
    fn test_axial_attention_output_finite() {
        let mut r = rng();
        let cfg = AxialAttentionConfig {
            num_heads: 2,
            head_dim: 8,
            causal: false,
        };
        let layer = AxialAttention::<f64>::new(16, 3, 3, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 9, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        for v in out.iter() {
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }

    // ---- FlashAttentionSimple ----

    #[test]
    fn test_flash_simple_creation() {
        let mut r = rng();
        let cfg = FlashAttentionSimpleConfig {
            num_heads: 2,
            head_dim: 8,
            tile_size: 4,
            causal: false,
        };
        let layer = FlashAttentionSimple::<f64>::new(16, cfg, &mut r).expect("create failed");
        assert_eq!(layer.layer_type(), "FlashAttentionSimple");
    }

    #[test]
    fn test_flash_simple_forward() {
        let mut r = rng();
        let cfg = FlashAttentionSimpleConfig {
            num_heads: 2,
            head_dim: 8,
            tile_size: 4,
            causal: false,
        };
        let layer = FlashAttentionSimple::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((2, 8, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 8, 16]);
    }

    #[test]
    fn test_flash_simple_causal() {
        let mut r = rng();
        let cfg = FlashAttentionSimpleConfig {
            num_heads: 2,
            head_dim: 8,
            tile_size: 3,
            causal: true,
        };
        let layer = FlashAttentionSimple::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 6, 16), 0.2).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 6, 16]);
    }

    #[test]
    fn test_flash_simple_output_finite() {
        let mut r = rng();
        let cfg = FlashAttentionSimpleConfig {
            num_heads: 2,
            head_dim: 8,
            tile_size: 4,
            causal: false,
        };
        let layer = FlashAttentionSimple::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 8, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        for v in out.iter() {
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }

    #[test]
    fn test_flash_simple_tile_zero_error() {
        let mut r = rng();
        let cfg = FlashAttentionSimpleConfig {
            num_heads: 2,
            head_dim: 8,
            tile_size: 0,
            causal: false,
        };
        assert!(FlashAttentionSimple::<f64>::new(16, cfg, &mut r).is_err());
    }

    #[test]
    fn test_flash_simple_tile_larger_than_seq() {
        // tile_size > seq: should still work (tile is clamped to seq)
        let mut r = rng();
        let cfg = FlashAttentionSimpleConfig {
            num_heads: 2,
            head_dim: 8,
            tile_size: 64,
            causal: false,
        };
        let layer = FlashAttentionSimple::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 5, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 5, 16]);
    }
}
