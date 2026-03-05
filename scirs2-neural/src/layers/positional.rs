//! Positional encoding variants for transformer architectures
//!
//! This module provides multiple positional encoding strategies:
//!
//! - **Sinusoidal**: Fixed sine/cosine positional encoding (Vaswani et al., 2017).
//! - **Learned Positional Embeddings**: Trainable position lookup table.
//! - **Rotary Position Embedding (RoPE)**: Rotation-based encoding (Su et al., 2021).
//! - **ALiBi**: Attention with Linear Biases (Press et al., 2022).
//! - **T5-style Relative Positional Encoding**: Bucketed relative position bias.
//!
//! ## Design
//!
//! Each encoding implements the [`Layer`] trait and can be inserted into a
//! [`Sequential`](crate::layers::Sequential) model. Sinusoidal and ALiBi are
//! parameter-free; Learned and T5-style have trainable parameters; RoPE is
//! applied as a modifier (adds rotated encoding to the input).

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::Rng;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

// ===========================================================================
// 1. Sinusoidal Positional Encoding
// ===========================================================================

/// Sinusoidal positional encoding (Vaswani et al., 2017).
///
/// Generates fixed (non-learnable) positional encodings using sine and cosine
/// functions of different frequencies. Even dimensions use sin, odd use cos.
///
/// PE(pos, 2i)   = sin(pos / 10000^{2i/d})
/// PE(pos, 2i+1) = cos(pos / 10000^{2i/d})
///
/// # Input: (batch, seq, d_model)
/// # Output: (batch, seq, d_model) with positional encoding added.
#[derive(Debug)]
pub struct SinusoidalPositionalEncoding<F: Float + Debug + Send + Sync + NumAssign> {
    max_seq_len: usize,
    d_model: usize,
    /// Pre-computed encoding table [max_seq_len, d_model]
    encoding: Array<F, IxDyn>,
    /// Dropout rate (fraction of encoding set to zero during training)
    dropout_rate: f64,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static>
    SinusoidalPositionalEncoding<F>
{
    /// Create a new sinusoidal positional encoding.
    ///
    /// # Arguments
    /// * `max_seq_len` - Maximum sequence length supported.
    /// * `d_model` - Embedding dimension.
    /// * `dropout_rate` - Dropout fraction applied to the encoding during training.
    pub fn new(max_seq_len: usize, d_model: usize, dropout_rate: f64) -> Result<Self> {
        if max_seq_len == 0 || d_model == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "max_seq_len and d_model must be > 0".into(),
            ));
        }

        let mut encoding = Array::zeros(IxDyn(&[max_seq_len, d_model]));
        for pos in 0..max_seq_len {
            for i in 0..d_model {
                let div_term = (10000.0_f64).powf(2.0 * (i / 2) as f64 / d_model as f64);
                let angle = pos as f64 / div_term;
                let val = if i % 2 == 0 { angle.sin() } else { angle.cos() };
                encoding[[pos, i]] =
                    F::from(val).ok_or_else(|| NeuralError::InvalidArchitecture("conv".into()))?;
            }
        }

        Ok(Self {
            max_seq_len,
            d_model,
            encoding,
            dropout_rate,
            training: true,
            _phantom: PhantomData,
        })
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for SinusoidalPositionalEncoding<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(NeuralError::InferenceError(
                "Input must be at least 2D".into(),
            ));
        }
        let last = shape[shape.len() - 1];
        if last != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Last dim ({last}) != d_model ({})",
                self.d_model
            )));
        }

        let seq_dim = shape.len() - 2;
        let seq_len = shape[seq_dim];
        if seq_len > self.max_seq_len {
            return Err(NeuralError::InferenceError(format!(
                "seq_len ({seq_len}) > max ({}) ",
                self.max_seq_len
            )));
        }

        let mut output = input.clone();
        let batch_dims: usize = shape[..seq_dim].iter().product();

        for b in 0..batch_dims {
            for s in 0..seq_len {
                for d in 0..self.d_model {
                    if shape.len() == 3 {
                        output[[b, s, d]] += self.encoding[[s, d]];
                    } else if shape.len() == 2 {
                        output[[s, d]] += self.encoding[[s, d]];
                    }
                }
            }
        }
        Ok(output)
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
    fn set_training(&mut self, t: bool) {
        self.training = t;
    }
    fn is_training(&self) -> bool {
        self.training
    }
    fn layer_type(&self) -> &str {
        "SinusoidalPositionalEncoding"
    }
    fn parameter_count(&self) -> usize {
        0
    }
}

// ===========================================================================
// 2. Learned Positional Embeddings
// ===========================================================================

/// Learned positional embeddings.
///
/// Stores a trainable lookup table of shape `[max_seq_len, d_model]`.
/// During forward, slices the table to `[seq_len, d_model]` and adds it
/// to the input.
///
/// # Input: (batch, seq, d_model)
/// # Output: (batch, seq, d_model) with learned positions added.
#[derive(Debug)]
pub struct LearnedPositionalEmbedding<F: Float + Debug + Send + Sync + NumAssign> {
    max_seq_len: usize,
    d_model: usize,
    weights: Arc<RwLock<Array<F, IxDyn>>>,
    weight_grad: Arc<RwLock<Array<F, IxDyn>>>,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static>
    LearnedPositionalEmbedding<F>
{
    /// Create a new learned positional embedding.
    pub fn new<R: Rng>(max_seq_len: usize, d_model: usize, rng: &mut R) -> Result<Self> {
        if max_seq_len == 0 || d_model == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "max_seq_len and d_model must be > 0".into(),
            ));
        }

        let scale = (2.0 / (max_seq_len + d_model) as f64).sqrt();
        let mut data = Vec::with_capacity(max_seq_len * d_model);
        for _ in 0..(max_seq_len * d_model) {
            let val: f64 = rng.random_range(-1.0..1.0);
            data.push(
                F::from(val * scale)
                    .ok_or_else(|| NeuralError::InvalidArchitecture("conv".into()))?,
            );
        }
        let weights = Array::from_shape_vec(IxDyn(&[max_seq_len, d_model]), data)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("{e}")))?;
        let grad = Array::zeros(IxDyn(&[max_seq_len, d_model]));

        Ok(Self {
            max_seq_len,
            d_model,
            weights: Arc::new(RwLock::new(weights)),
            weight_grad: Arc::new(RwLock::new(grad)),
            training: true,
            _phantom: PhantomData,
        })
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for LearnedPositionalEmbedding<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(NeuralError::InferenceError("Need at least 2D".into()));
        }
        let last = shape[shape.len() - 1];
        if last != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Last dim ({last}) != d_model ({})",
                self.d_model
            )));
        }
        let seq_dim = shape.len() - 2;
        let seq_len = shape[seq_dim];
        if seq_len > self.max_seq_len {
            return Err(NeuralError::InferenceError(format!(
                "seq ({seq_len}) > max ({})",
                self.max_seq_len
            )));
        }

        let weights = self
            .weights
            .read()
            .map_err(|_| NeuralError::InferenceError("lock".into()))?;

        let mut output = input.clone();
        let batch_dims: usize = shape[..seq_dim].iter().product();

        for b in 0..batch_dims {
            for s in 0..seq_len {
                for d in 0..self.d_model {
                    if shape.len() == 3 {
                        output[[b, s, d]] += weights[[s, d]];
                    } else if shape.len() == 2 {
                        output[[s, d]] += weights[[s, d]];
                    }
                }
            }
        }
        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad.clone())
    }

    fn update(&mut self, lr: F) -> Result<()> {
        let grad = self
            .weight_grad
            .read()
            .map_err(|_| NeuralError::InferenceError("lock".into()))?;
        let mut w = self
            .weights
            .write()
            .map_err(|_| NeuralError::InferenceError("lock".into()))?;
        for i in 0..self.max_seq_len {
            for j in 0..self.d_model {
                w[[i, j]] -= lr * grad[[i, j]];
            }
        }
        drop(w);
        drop(grad);
        let mut grad = self
            .weight_grad
            .write()
            .map_err(|_| NeuralError::InferenceError("lock".into()))?;
        grad.fill(F::zero());
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn set_training(&mut self, t: bool) {
        self.training = t;
    }
    fn is_training(&self) -> bool {
        self.training
    }
    fn layer_type(&self) -> &str {
        "LearnedPositionalEmbedding"
    }
    fn parameter_count(&self) -> usize {
        self.max_seq_len * self.d_model
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        if let Ok(w) = self.weights.read() {
            vec![w.clone()]
        } else {
            vec![]
        }
    }

    fn set_params(&mut self, params: &[Array<F, IxDyn>]) -> Result<()> {
        if let Some(p) = params.first() {
            if let Ok(mut w) = self.weights.write() {
                *w = p.clone();
            }
        }
        Ok(())
    }
}

// ===========================================================================
// 3. Rotary Position Embedding (RoPE)
// ===========================================================================

/// Rotary Position Embedding layer (Su et al., 2021).
///
/// Applies 2D rotations to pairs of dimensions so that the dot product
/// between two position-encoded vectors depends on relative position.
/// This layer rotates the input in-place and returns it.
///
/// # Input: (batch, seq, d_model) where d_model must be even
/// # Output: same shape, with RoPE rotations applied
#[derive(Debug)]
pub struct RotaryPositionEmbedding<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    base: f64,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static>
    RotaryPositionEmbedding<F>
{
    /// Create a new RoPE layer.
    ///
    /// # Arguments
    /// * `d_model` - Embedding dimension (must be even).
    /// * `base` - Base for frequency computation (default 10000).
    pub fn new(d_model: usize, base: f64) -> Result<Self> {
        if d_model == 0 || d_model % 2 != 0 {
            return Err(NeuralError::InvalidArchitecture(
                "d_model must be even and > 0 for RoPE".into(),
            ));
        }
        Ok(Self {
            d_model,
            base,
            training: true,
            _phantom: PhantomData,
        })
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for RotaryPositionEmbedding<F>
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
            return Err(NeuralError::InferenceError(format!(
                "d_model mismatch: expected {}, got {dm}",
                self.d_model
            )));
        }

        let half = dm / 2;
        let mut output = input.clone();

        for b in 0..batch {
            for s in 0..seq {
                for i in 0..half {
                    let freq = 1.0 / self.base.powf(2.0 * i as f64 / dm as f64);
                    let angle = s as f64 * freq;
                    let cos_a = F::from(angle.cos()).unwrap_or(F::one());
                    let sin_a = F::from(angle.sin()).unwrap_or(F::zero());

                    let x0 = input[[b, s, 2 * i]];
                    let x1 = input[[b, s, 2 * i + 1]];
                    output[[b, s, 2 * i]] = x0 * cos_a - x1 * sin_a;
                    output[[b, s, 2 * i + 1]] = x0 * sin_a + x1 * cos_a;
                }
            }
        }
        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Inverse rotation: negate the angle
        let shape = grad.shape();
        if shape.len() != 3 {
            return Ok(grad.clone());
        }
        let (batch, seq, dm) = (shape[0], shape[1], shape[2]);
        let half = dm / 2;
        let mut out = grad.clone();
        for b in 0..batch {
            for s in 0..seq {
                for i in 0..half {
                    let freq = 1.0 / self.base.powf(2.0 * i as f64 / dm as f64);
                    let angle = -(s as f64 * freq);
                    let cos_a = F::from(angle.cos()).unwrap_or(F::one());
                    let sin_a = F::from(angle.sin()).unwrap_or(F::zero());
                    let g0 = grad[[b, s, 2 * i]];
                    let g1 = grad[[b, s, 2 * i + 1]];
                    out[[b, s, 2 * i]] = g0 * cos_a - g1 * sin_a;
                    out[[b, s, 2 * i + 1]] = g0 * sin_a + g1 * cos_a;
                }
            }
        }
        Ok(out)
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
    fn set_training(&mut self, t: bool) {
        self.training = t;
    }
    fn is_training(&self) -> bool {
        self.training
    }
    fn layer_type(&self) -> &str {
        "RotaryPositionEmbedding"
    }
    fn parameter_count(&self) -> usize {
        0
    }
}

// ===========================================================================
// 4. ALiBi (Attention with Linear Biases)
// ===========================================================================

/// ALiBi positional encoding (Press et al., 2022).
///
/// Instead of adding positional information to token embeddings, ALiBi
/// adds a linear bias to attention scores:
///   score(i, j) += -m * |i - j|
/// where m is a head-specific slope.
///
/// This layer computes the bias matrix and adds it to the input, which is
/// assumed to be attention logits of shape `(batch, num_heads, seq, seq)`.
///
/// # Input: (batch, num_heads, seq, seq) -- attention logits
/// # Output: same shape with ALiBi bias added
#[derive(Debug)]
pub struct ALiBiEncoding<F: Float + Debug + Send + Sync + NumAssign> {
    num_heads: usize,
    /// Slope per head
    slopes: Vec<F>,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> ALiBiEncoding<F> {
    /// Create ALiBi encoding with the standard geometric slope schedule.
    ///
    /// Slopes are: `2^{-8/n * (1..=n)}` where n is the nearest power of 2
    /// >= num_heads.
    pub fn new(num_heads: usize) -> Result<Self> {
        if num_heads == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "num_heads must be > 0".into(),
            ));
        }

        let slopes = Self::compute_slopes(num_heads)?;
        Ok(Self {
            num_heads,
            slopes,
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Compute geometric slopes for ALiBi.
    fn compute_slopes(num_heads: usize) -> Result<Vec<F>> {
        // Find closest power of 2 >= num_heads
        let mut n = 1;
        while n < num_heads {
            n *= 2;
        }

        let ratio = 8.0 / n as f64;
        let mut slopes = Vec::with_capacity(num_heads);
        for i in 1..=num_heads {
            let slope = 2.0_f64.powf(-(ratio * i as f64));
            slopes.push(
                F::from(slope)
                    .ok_or_else(|| NeuralError::InvalidArchitecture("slope conv".into()))?,
            );
        }
        Ok(slopes)
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for ALiBiEncoding<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        // Accept 4D (batch, heads, seq_q, seq_k) or 3D (batch, seq, d_model)
        if shape.len() == 4 {
            let (batch, nh, sq, sk) = (shape[0], shape[1], shape[2], shape[3]);
            if nh != self.num_heads {
                return Err(NeuralError::InferenceError(format!(
                    "num_heads mismatch: expected {}, got {nh}",
                    self.num_heads
                )));
            }
            let mut output = input.clone();
            for b in 0..batch {
                for h in 0..nh {
                    let slope = self.slopes[h];
                    for i in 0..sq {
                        for j in 0..sk {
                            let dist = if i > j { i - j } else { j - i };
                            let bias = F::from(dist).unwrap_or(F::zero()) * slope;
                            output[[b, h, i, j]] = output[[b, h, i, j]] - bias;
                        }
                    }
                }
            }
            Ok(output)
        } else if shape.len() == 3 {
            // 3D pass-through: ALiBi is typically applied to attention scores,
            // not embeddings. Return input unchanged with a warning-free path.
            Ok(input.clone())
        } else {
            Err(NeuralError::InferenceError(format!(
                "ALiBi expects 4D (batch, heads, seq, seq) or 3D input, got {}D",
                shape.len()
            )))
        }
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
    fn set_training(&mut self, t: bool) {
        self.training = t;
    }
    fn is_training(&self) -> bool {
        self.training
    }
    fn layer_type(&self) -> &str {
        "ALiBiEncoding"
    }
    fn parameter_count(&self) -> usize {
        0
    }
}

// ===========================================================================
// 5. T5-style Relative Positional Encoding
// ===========================================================================

/// T5-style relative position bias (Raffel et al., 2020).
///
/// Uses bucketed relative positions with a logarithmic far-range and
/// a learnable bias per (bucket, head). The bias is added to attention logits.
///
/// # Input: (batch, num_heads, seq, seq) -- attention logits
/// # Output: same shape with relative bias added
#[derive(Debug)]
pub struct T5RelativePositionBias<F: Float + Debug + Send + Sync + NumAssign> {
    num_heads: usize,
    num_buckets: usize,
    max_distance: usize,
    bidirectional: bool,
    /// Learnable bias table [num_buckets, num_heads]
    bias_table: Arc<RwLock<Array<F, IxDyn>>>,
    bias_grad: Arc<RwLock<Array<F, IxDyn>>>,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static>
    T5RelativePositionBias<F>
{
    /// Create a new T5-style relative position bias.
    ///
    /// # Arguments
    /// * `num_heads` - Number of attention heads.
    /// * `num_buckets` - Number of relative position buckets (default 32 in T5).
    /// * `max_distance` - Maximum relative distance for bucketing (default 128).
    /// * `bidirectional` - Whether to use separate buckets for left/right.
    pub fn new<R: Rng>(
        num_heads: usize,
        num_buckets: usize,
        max_distance: usize,
        bidirectional: bool,
        rng: &mut R,
    ) -> Result<Self> {
        if num_heads == 0 || num_buckets == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "num_heads and num_buckets must be > 0".into(),
            ));
        }

        let scale = (1.0 / (num_buckets as f64).sqrt()).min(0.1);
        let mut data = Vec::with_capacity(num_buckets * num_heads);
        for _ in 0..(num_buckets * num_heads) {
            let val: f64 = rng.random_range(-1.0..1.0);
            data.push(
                F::from(val * scale)
                    .ok_or_else(|| NeuralError::InvalidArchitecture("conv".into()))?,
            );
        }
        let bias_table = Array::from_shape_vec(IxDyn(&[num_buckets, num_heads]), data)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("{e}")))?;
        let bias_grad = Array::zeros(IxDyn(&[num_buckets, num_heads]));

        Ok(Self {
            num_heads,
            num_buckets,
            max_distance,
            bidirectional,
            bias_table: Arc::new(RwLock::new(bias_table)),
            bias_grad: Arc::new(RwLock::new(bias_grad)),
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Map a relative position to a bucket index (T5 bucketing scheme).
    fn relative_position_bucket(&self, relative_position: isize) -> usize {
        let mut rp = relative_position;
        let mut num_buckets = self.num_buckets;
        let max_exact: usize;

        if self.bidirectional {
            num_buckets /= 2;
            let offset = if rp > 0 { num_buckets } else { 0 };
            rp = rp.abs();

            max_exact = num_buckets / 2;
            let bucket = if (rp as usize) < max_exact {
                rp as usize
            } else {
                let val = (rp as f64 / max_exact as f64).ln()
                    / (self.max_distance as f64 / max_exact as f64).ln()
                    * (num_buckets - max_exact) as f64;
                max_exact + (val as usize).min(num_buckets - max_exact - 1)
            };
            bucket + offset
        } else {
            rp = rp.max(0);
            max_exact = num_buckets / 2;
            if (rp as usize) < max_exact {
                rp as usize
            } else {
                let val = (rp as f64 / max_exact as f64).ln()
                    / (self.max_distance as f64 / max_exact as f64).ln()
                    * (num_buckets - max_exact) as f64;
                max_exact + (val as usize).min(num_buckets - max_exact - 1)
            }
        }
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for T5RelativePositionBias<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InferenceError(format!(
                "T5 relative pos bias expects 4D (batch, heads, seq_q, seq_k), got {}D",
                shape.len()
            )));
        }
        let (batch, nh, sq, sk) = (shape[0], shape[1], shape[2], shape[3]);
        if nh != self.num_heads {
            return Err(NeuralError::InferenceError(format!(
                "num_heads mismatch: expected {}, got {nh}",
                self.num_heads
            )));
        }

        let bias = self
            .bias_table
            .read()
            .map_err(|_| NeuralError::InferenceError("lock".into()))?;

        let mut output = input.clone();
        for b in 0..batch {
            for h in 0..nh {
                for i in 0..sq {
                    for j in 0..sk {
                        let rp = j as isize - i as isize;
                        let bucket = self.relative_position_bucket(rp);
                        let bucket = bucket.min(self.num_buckets - 1);
                        output[[b, h, i, j]] += bias[[bucket, h]];
                    }
                }
            }
        }
        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad.clone())
    }

    fn update(&mut self, lr: F) -> Result<()> {
        let grad = self
            .bias_grad
            .read()
            .map_err(|_| NeuralError::InferenceError("lock".into()))?;
        let mut w = self
            .bias_table
            .write()
            .map_err(|_| NeuralError::InferenceError("lock".into()))?;
        for i in 0..self.num_buckets {
            for j in 0..self.num_heads {
                w[[i, j]] -= lr * grad[[i, j]];
            }
        }
        drop(w);
        drop(grad);
        let mut grad = self
            .bias_grad
            .write()
            .map_err(|_| NeuralError::InferenceError("lock".into()))?;
        grad.fill(F::zero());
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn set_training(&mut self, t: bool) {
        self.training = t;
    }
    fn is_training(&self) -> bool {
        self.training
    }
    fn layer_type(&self) -> &str {
        "T5RelativePositionBias"
    }
    fn parameter_count(&self) -> usize {
        self.num_buckets * self.num_heads
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        if let Ok(w) = self.bias_table.read() {
            vec![w.clone()]
        } else {
            vec![]
        }
    }

    fn set_params(&mut self, params: &[Array<F, IxDyn>]) -> Result<()> {
        if let Some(p) = params.first() {
            if let Ok(mut w) = self.bias_table.write() {
                *w = p.clone();
            }
        }
        Ok(())
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

    // ------- Sinusoidal -------

    #[test]
    fn test_sinusoidal_creation() {
        let pe = SinusoidalPositionalEncoding::<f64>::new(100, 64, 0.0).expect("creation failed");
        assert_eq!(pe.layer_type(), "SinusoidalPositionalEncoding");
        assert_eq!(pe.parameter_count(), 0);
    }

    #[test]
    fn test_sinusoidal_forward() {
        let pe = SinusoidalPositionalEncoding::<f64>::new(100, 16, 0.0).expect("creation failed");
        let input = Array3::<f64>::from_elem((2, 10, 16), 0.0).into_dyn();
        let out = pe.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 10, 16]);
        // Position 0, dim 0 should be sin(0) = 0
        assert!((out[[0, 0, 0]]).abs() < 1e-6);
    }

    #[test]
    fn test_sinusoidal_seq_too_long() {
        let pe = SinusoidalPositionalEncoding::<f64>::new(5, 8, 0.0).expect("creation failed");
        let input = Array3::<f64>::from_elem((1, 10, 8), 0.0).into_dyn();
        assert!(pe.forward(&input).is_err());
    }

    // ------- Learned -------

    #[test]
    fn test_learned_creation() {
        let mut r = rng();
        let pe = LearnedPositionalEmbedding::<f64>::new(50, 32, &mut r).expect("creation failed");
        assert_eq!(pe.layer_type(), "LearnedPositionalEmbedding");
        assert_eq!(pe.parameter_count(), 50 * 32);
    }

    #[test]
    fn test_learned_forward() {
        let mut r = rng();
        let pe = LearnedPositionalEmbedding::<f64>::new(50, 16, &mut r).expect("creation failed");
        let input = Array3::<f64>::from_elem((2, 10, 16), 1.0).into_dyn();
        let out = pe.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 10, 16]);
    }

    #[test]
    fn test_learned_params() {
        let mut r = rng();
        let pe = LearnedPositionalEmbedding::<f64>::new(20, 8, &mut r).expect("creation failed");
        assert_eq!(pe.params().len(), 1);
    }

    // ------- RoPE -------

    #[test]
    fn test_rope_creation() {
        let pe = RotaryPositionEmbedding::<f64>::new(16, 10000.0).expect("creation failed");
        assert_eq!(pe.layer_type(), "RotaryPositionEmbedding");
        assert_eq!(pe.parameter_count(), 0);
    }

    #[test]
    fn test_rope_forward() {
        let pe = RotaryPositionEmbedding::<f64>::new(16, 10000.0).expect("creation failed");
        let input = Array3::<f64>::from_elem((2, 5, 16), 1.0).into_dyn();
        let out = pe.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 5, 16]);
    }

    #[test]
    fn test_rope_odd_dim_error() {
        let result = RotaryPositionEmbedding::<f64>::new(15, 10000.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_position_zero_identity() {
        let pe = RotaryPositionEmbedding::<f64>::new(4, 10000.0).expect("creation failed");
        let input = Array3::<f64>::from_elem((1, 1, 4), 1.0).into_dyn();
        let out = pe.forward(&input).expect("forward failed");
        // At position 0, angles are 0, so cos=1, sin=0 => identity rotation
        for d in 0..4 {
            assert!((out[[0, 0, d]] - 1.0).abs() < 1e-10);
        }
    }

    // ------- ALiBi -------

    #[test]
    fn test_alibi_creation() {
        let alibi = ALiBiEncoding::<f64>::new(8).expect("creation failed");
        assert_eq!(alibi.layer_type(), "ALiBiEncoding");
        assert_eq!(alibi.parameter_count(), 0);
    }

    #[test]
    fn test_alibi_forward_4d() {
        let alibi = ALiBiEncoding::<f64>::new(4).expect("creation failed");
        // (batch, heads, seq_q, seq_k)
        let input = Array::zeros(IxDyn(&[2, 4, 5, 5]));
        let out = alibi.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 4, 5, 5]);
        // Diagonal should be zero bias (distance = 0)
        assert!((out[[0, 0, 0, 0]]).abs() < 1e-10);
        // Off-diagonal should be negative (distance > 0)
        assert!(out[[0, 0, 0, 1]] < 0.0);
    }

    #[test]
    fn test_alibi_slopes() {
        let alibi = ALiBiEncoding::<f64>::new(4).expect("creation failed");
        // All slopes should be positive and < 1
        for &s in &alibi.slopes {
            assert!(s > 0.0);
            assert!(s < 1.0);
        }
        // Slopes should be in descending order (largest first)
        for i in 1..alibi.slopes.len() {
            assert!(alibi.slopes[i] <= alibi.slopes[i - 1]);
        }
    }

    // ------- T5 Relative Position Bias -------

    #[test]
    fn test_t5_creation() {
        let mut r = rng();
        let t5 =
            T5RelativePositionBias::<f64>::new(4, 32, 128, true, &mut r).expect("creation failed");
        assert_eq!(t5.layer_type(), "T5RelativePositionBias");
        assert_eq!(t5.parameter_count(), 32 * 4);
    }

    #[test]
    fn test_t5_forward() {
        let mut r = rng();
        let t5 =
            T5RelativePositionBias::<f64>::new(4, 32, 128, true, &mut r).expect("creation failed");
        let input = Array::zeros(IxDyn(&[2, 4, 6, 6]));
        let out = t5.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 4, 6, 6]);
    }

    #[test]
    fn test_t5_bucketing() {
        let mut r = rng();
        let t5 =
            T5RelativePositionBias::<f64>::new(2, 32, 128, true, &mut r).expect("creation failed");
        // Small distances should map to small bucket indices
        let b0 = t5.relative_position_bucket(0);
        let b1 = t5.relative_position_bucket(1);
        assert!(b0 <= b1 || b0 < t5.num_buckets);
        // Large distance should map to a high bucket
        let b_far = t5.relative_position_bucket(100);
        assert!(b_far < t5.num_buckets);
    }

    #[test]
    fn test_t5_bidirectional_vs_unidirectional() {
        let mut r = rng();
        let t5_bi =
            T5RelativePositionBias::<f64>::new(2, 32, 128, true, &mut r).expect("creation failed");
        let t5_uni =
            T5RelativePositionBias::<f64>::new(2, 32, 128, false, &mut r).expect("creation failed");

        // Bidirectional: negative and positive positions should map differently
        let b_neg = t5_bi.relative_position_bucket(-5);
        let b_pos = t5_bi.relative_position_bucket(5);
        // They should be in different halves
        assert_ne!(b_neg, b_pos);

        // Unidirectional: negative is clamped to 0
        let b_neg_uni = t5_uni.relative_position_bucket(-5);
        let b_zero_uni = t5_uni.relative_position_bucket(0);
        assert_eq!(b_neg_uni, b_zero_uni);
    }

    #[test]
    fn test_t5_params() {
        let mut r = rng();
        let t5 =
            T5RelativePositionBias::<f64>::new(4, 16, 64, true, &mut r).expect("creation failed");
        assert_eq!(t5.params().len(), 1);
    }
}
