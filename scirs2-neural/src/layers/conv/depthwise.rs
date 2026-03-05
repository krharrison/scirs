//! Depthwise and mobile-style convolutional layers
//!
//! Provides lightweight convolutional building blocks following the MobileNet
//! family of architectures:
//!
//! - [`DepthwiseConv2d`] – applies one filter **per input channel**
//!   (channel-wise convolution, groups == in_channels)
//! - [`PointwiseConv2d`] – 1×1 convolution that mixes channels without
//!   changing spatial dimensions
//! - [`MobileBlock`] – MobileNetV1 depthwise-separable block
//!   (DepthwiseConv2d → BN → ReLU6 → PointwiseConv2d → BN → ReLU6)
//! - [`InvertedResidual`] – MobileNetV2 inverted residual block with expansion
//!
//! # References
//! - Howard et al., "MobileNets", 2017. <https://arxiv.org/abs/1704.04861>
//! - Sandler et al., "MobileNetV2", 2018. <https://arxiv.org/abs/1801.04381>

use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Distribution, Rng, Uniform};
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Apply ReLU6 element-wise: `min(max(x, 0), 6)`.
#[inline]
fn relu6<F: Float>(x: F) -> F {
    let zero = F::zero();
    let six = F::from(6.0_f64).unwrap_or(F::one() + F::one());
    x.max(zero).min(six)
}

/// Simplified channel-wise batch normalisation for inference.
fn batch_norm_forward<F: Float + NumAssign + Debug>(
    x: &Array<F, IxDyn>,
    gamma: &Array<F, IxDyn>,
    beta: &Array<F, IxDyn>,
    epsilon: F,
) -> Result<Array<F, IxDyn>> {
    let shape = x.shape();
    if shape.len() != 4 {
        return Err(NeuralError::InferenceError(
            "batch_norm_forward: expected 4-D input".to_string(),
        ));
    }
    let channels = shape[1];
    let mut out = x.to_owned();
    for c in 0..channels {
        let chan = x.slice(scirs2_core::ndarray::s![.., c, .., ..]);
        let n = F::from(chan.len())
            .ok_or_else(|| NeuralError::InferenceError("Cannot convert len".to_string()))?;
        let mean = chan.iter().fold(F::zero(), |a, &v| a + v) / n;
        let var = chan
            .iter()
            .fold(F::zero(), |a, &v| a + (v - mean) * (v - mean))
            / n;
        let std_inv = F::one() / (var + epsilon).sqrt();
        let g = gamma[c];
        let b = beta[c];
        for batch in 0..shape[0] {
            for h in 0..shape[2] {
                for w in 0..shape[3] {
                    out[[batch, c, h, w]] = g * (x[[batch, c, h, w]] - mean) * std_inv + b;
                }
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// DepthwiseConv2d
// ---------------------------------------------------------------------------

/// Depthwise (channel-wise) 2-D convolution.
///
/// Each input channel is convolved with its own dedicated filter. No
/// cross-channel mixing occurs. Output channels == input channels.
///
/// # Input / Output shape
/// `[batch, C_in, H, W] → [batch, C_in, H_out, W_out]`
///
/// # Examples
/// ```
/// use scirs2_neural::layers::conv::depthwise::DepthwiseConv2d;
/// use scirs2_neural::layers::Layer;
/// use scirs2_core::ndarray::Array4;
/// use scirs2_core::random::rngs::SmallRng;
/// use scirs2_core::random::SeedableRng;
///
/// let mut rng = SmallRng::seed_from_u64(0);
/// let dw = DepthwiseConv2d::<f64>::new(8, (3, 3), (1, 1), (1, 1), &mut rng)
///     .expect("build failed");
/// let x = Array4::<f64>::zeros((2, 8, 16, 16)).into_dyn();
/// let y = dw.forward(&x).expect("forward failed");
/// assert_eq!(y.shape(), &[2, 8, 16, 16]);
/// ```
pub struct DepthwiseConv2d<F: Float + Debug + Send + Sync + NumAssign> {
    channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    /// Weights: [channels, 1, kH, kW]
    weights: Arc<RwLock<Array<F, IxDyn>>>,
    /// Bias: [channels]
    bias: Arc<RwLock<Array<F, IxDyn>>>,
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> DepthwiseConv2d<F> {
    /// Create a new `DepthwiseConv2d` layer.
    pub fn new<R: Rng>(
        channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        rng: &mut R,
    ) -> Result<Self> {
        if channels == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "DepthwiseConv2d: channels must be > 0".to_string(),
            ));
        }
        let fan_in = kernel_size.0 * kernel_size.1;
        let scale = (2.0_f64 / fan_in as f64).sqrt();
        let dist = Uniform::new(-scale, scale)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("Uniform dist: {e}")))?;
        let w_size = channels * kernel_size.0 * kernel_size.1;
        let w_flat: Vec<F> = (0..w_size)
            .map(|_| F::from(dist.sample(rng)).unwrap_or(F::zero()))
            .collect();
        let weights = Array::from_shape_vec(
            IxDyn(&[channels, 1, kernel_size.0, kernel_size.1]),
            w_flat,
        )
        .map_err(|e| NeuralError::InvalidArchitecture(format!("weight init: {e}")))?;
        let bias = Array::zeros(IxDyn(&[channels]));
        Ok(Self {
            channels,
            kernel_size,
            stride,
            padding,
            weights: Arc::new(RwLock::new(weights)),
            bias: Arc::new(RwLock::new(bias)),
            input_cache: Arc::new(RwLock::new(None)),
        })
    }

    fn run_conv(
        &self,
        input: &Array<F, IxDyn>,
        weights: &Array<F, IxDyn>,
        bias: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InferenceError(
                "DepthwiseConv2d: expected 4-D input [batch, C, H, W]".to_string(),
            ));
        }
        let (batch, c_in, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        if c_in != self.channels {
            return Err(NeuralError::InferenceError(format!(
                "DepthwiseConv2d: channel mismatch – expected {}, got {}",
                self.channels, c_in
            )));
        }
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let h_out = (h_in + 2 * ph).saturating_sub(kh) / sh + 1;
        let w_out = (w_in + 2 * pw).saturating_sub(kw) / sw + 1;
        let mut out = Array::<F, IxDyn>::zeros(IxDyn(&[batch, c_in, h_out, w_out]));
        for b in 0..batch {
            for c in 0..c_in {
                let b_c = bias[c];
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = F::zero();
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = (oh * sh + ki).wrapping_sub(ph);
                                let iw = (ow * sw + kj).wrapping_sub(pw);
                                if ih < h_in && iw < w_in {
                                    acc = acc + input[[b, c, ih, iw]] * weights[[c, 0, ki, kj]];
                                }
                            }
                        }
                        out[[b, c, oh, ow]] = acc + b_c;
                    }
                }
            }
        }
        Ok(out)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for DepthwiseConv2d<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let w = self
            .weights
            .read()
            .map_err(|e| NeuralError::InferenceError(format!("RwLock: {e}")))?;
        let b = self
            .bias
            .read()
            .map_err(|e| NeuralError::InferenceError(format!("RwLock: {e}")))?;
        *self
            .input_cache
            .write()
            .map_err(|e| NeuralError::InferenceError(format!("RwLock: {e}")))? =
            Some(input.clone());
        self.run_conv(input, &w, &b)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "DepthwiseConv2d"
    }

    fn parameter_count(&self) -> usize {
        let w = self.weights.read().map(|w| w.len()).unwrap_or(0);
        let b = self.bias.read().map(|b| b.len()).unwrap_or(0);
        w + b
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> ParamLayer<F>
    for DepthwiseConv2d<F>
{

    fn get_parameters(&self) -> Vec<Array<F, IxDyn>> {
        let w = self.weights.read().map(|w| w.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[])));
        let b = self.bias.read().map(|b| b.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[])));
        vec![w, b]
    }

    fn get_gradients(&self) -> Vec<Array<F, IxDyn>> {
        Vec::new()
    }

    fn set_parameters(&mut self, params: Vec<Array<F, IxDyn>>) -> Result<()> {
        if params.len() != 2 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "DepthwiseConv2d::set_parameters: expected 2, got {}",
                params.len()
            )));
        }
        *self
            .weights
            .write()
            .map_err(|e| NeuralError::InferenceError(format!("RwLock: {e}")))? =
            params[0].clone();
        *self
            .bias
            .write()
            .map_err(|e| NeuralError::InferenceError(format!("RwLock: {e}")))? =
            params[1].clone();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PointwiseConv2d
// ---------------------------------------------------------------------------

/// Pointwise (1×1) 2-D convolution.
///
/// Maps `in_channels → out_channels` with a 1×1 kernel.
///
/// # Input / Output shape
/// `[batch, C_in, H, W] → [batch, C_out, H, W]`
///
/// # Examples
/// ```
/// use scirs2_neural::layers::conv::depthwise::PointwiseConv2d;
/// use scirs2_neural::layers::Layer;
/// use scirs2_core::ndarray::Array4;
/// use scirs2_core::random::rngs::SmallRng;
/// use scirs2_core::random::SeedableRng;
///
/// let mut rng = SmallRng::seed_from_u64(0);
/// let pw = PointwiseConv2d::<f64>::new(8, 16, &mut rng).expect("build failed");
/// let x = Array4::<f64>::zeros((2, 8, 16, 16)).into_dyn();
/// let y = pw.forward(&x).expect("forward failed");
/// assert_eq!(y.shape(), &[2, 16, 16, 16]);
/// ```
pub struct PointwiseConv2d<F: Float + Debug + Send + Sync + NumAssign> {
    in_channels: usize,
    out_channels: usize,
    /// Weights: [out_channels, in_channels]
    weights: Arc<RwLock<Array<F, IxDyn>>>,
    /// Bias: [out_channels]
    bias: Arc<RwLock<Array<F, IxDyn>>>,
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> PointwiseConv2d<F> {
    /// Create a new `PointwiseConv2d` layer.
    pub fn new<R: Rng>(in_channels: usize, out_channels: usize, rng: &mut R) -> Result<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "PointwiseConv2d: channels must be > 0".to_string(),
            ));
        }
        let scale = (2.0_f64 / in_channels as f64).sqrt();
        let dist = Uniform::new(-scale, scale)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("Uniform dist: {e}")))?;
        let w_flat: Vec<F> = (0..out_channels * in_channels)
            .map(|_| F::from(dist.sample(rng)).unwrap_or(F::zero()))
            .collect();
        let weights = Array::from_shape_vec(IxDyn(&[out_channels, in_channels]), w_flat)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("weight init: {e}")))?;
        let bias = Array::zeros(IxDyn(&[out_channels]));
        Ok(Self {
            in_channels,
            out_channels,
            weights: Arc::new(RwLock::new(weights)),
            bias: Arc::new(RwLock::new(bias)),
            input_cache: Arc::new(RwLock::new(None)),
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for PointwiseConv2d<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(NeuralError::InferenceError(
                "PointwiseConv2d: expected 4-D input [batch, C, H, W]".to_string(),
            ));
        }
        let (batch, c_in, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        if c_in != self.in_channels {
            return Err(NeuralError::InferenceError(format!(
                "PointwiseConv2d: channel mismatch – expected {}, got {}",
                self.in_channels, c_in
            )));
        }
        *self
            .input_cache
            .write()
            .map_err(|e| NeuralError::InferenceError(format!("RwLock: {e}")))? =
            Some(input.clone());
        let weights = self
            .weights
            .read()
            .map_err(|e| NeuralError::InferenceError(format!("RwLock: {e}")))?;
        let bias = self
            .bias
            .read()
            .map_err(|e| NeuralError::InferenceError(format!("RwLock: {e}")))?;
        let mut out = Array::<F, IxDyn>::zeros(IxDyn(&[batch, self.out_channels, h, w]));
        for b in 0..batch {
            for oc in 0..self.out_channels {
                let b_oc = bias[oc];
                for ih in 0..h {
                    for iw in 0..w {
                        let mut acc = F::zero();
                        for ic in 0..c_in {
                            acc = acc + input[[b, ic, ih, iw]] * weights[[oc, ic]];
                        }
                        out[[b, oc, ih, iw]] = acc + b_oc;
                    }
                }
            }
        }
        Ok(out)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "PointwiseConv2d"
    }

    fn parameter_count(&self) -> usize {
        let w = self.weights.read().map(|w| w.len()).unwrap_or(0);
        let b = self.bias.read().map(|b| b.len()).unwrap_or(0);
        w + b
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> ParamLayer<F>
    for PointwiseConv2d<F>
{

    fn get_parameters(&self) -> Vec<Array<F, IxDyn>> {
        let w = self.weights.read().map(|w| w.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[])));
        let b = self.bias.read().map(|b| b.clone()).unwrap_or_else(|_| Array::zeros(IxDyn(&[])));
        vec![w, b]
    }

    fn get_gradients(&self) -> Vec<Array<F, IxDyn>> {
        Vec::new()
    }

    fn set_parameters(&mut self, params: Vec<Array<F, IxDyn>>) -> Result<()> {
        if params.len() != 2 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "PointwiseConv2d::set_parameters: expected 2, got {}",
                params.len()
            )));
        }
        *self
            .weights
            .write()
            .map_err(|e| NeuralError::InferenceError(format!("RwLock: {e}")))? =
            params[0].clone();
        *self
            .bias
            .write()
            .map_err(|e| NeuralError::InferenceError(format!("RwLock: {e}")))? =
            params[1].clone();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MobileBlock
// ---------------------------------------------------------------------------

/// MobileNetV1-style depthwise-separable convolutional block.
///
/// Sequence: DepthwiseConv2d → BN+ReLU6 → PointwiseConv2d → BN+ReLU6.
pub struct MobileBlock<F: Float + Debug + Send + Sync + NumAssign> {
    depthwise: DepthwiseConv2d<F>,
    pointwise: PointwiseConv2d<F>,
    dw_gamma: Array<F, IxDyn>,
    dw_beta: Array<F, IxDyn>,
    pw_gamma: Array<F, IxDyn>,
    pw_beta: Array<F, IxDyn>,
    bn_eps: F,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> MobileBlock<F> {
    /// Create a new `MobileBlock`.
    pub fn new<R: Rng>(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        rng: &mut R,
    ) -> Result<Self> {
        let depthwise = DepthwiseConv2d::new(in_channels, kernel_size, stride, padding, rng)?;
        let pointwise = PointwiseConv2d::new(in_channels, out_channels, rng)?;
        let dw_gamma = Array::from_elem(IxDyn(&[in_channels]), F::one());
        let dw_beta = Array::zeros(IxDyn(&[in_channels]));
        let pw_gamma = Array::from_elem(IxDyn(&[out_channels]), F::one());
        let pw_beta = Array::zeros(IxDyn(&[out_channels]));
        let bn_eps = F::from(1e-5_f64).unwrap_or_else(F::zero);
        Ok(Self {
            depthwise,
            pointwise,
            dw_gamma,
            dw_beta,
            pw_gamma,
            pw_beta,
            bn_eps,
        })
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for MobileBlock<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let dw_out = self.depthwise.forward(input)?;
        let bn1 = batch_norm_forward(&dw_out, &self.dw_gamma, &self.dw_beta, self.bn_eps)?;
        let relu1 = bn1.mapv(relu6);
        let pw_out = self.pointwise.forward(&relu1)?;
        let bn2 = batch_norm_forward(&pw_out, &self.pw_gamma, &self.pw_beta, self.bn_eps)?;
        Ok(bn2.mapv(relu6))
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "MobileBlock"
    }

    fn parameter_count(&self) -> usize {
        self.depthwise.parameter_count() + self.pointwise.parameter_count()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> ParamLayer<F>
    for MobileBlock<F>
{

    fn get_parameters(&self) -> Vec<Array<F, IxDyn>> {
        let mut p = self.depthwise.get_parameters();
        p.extend(self.pointwise.get_parameters());
        p
    }

    fn get_gradients(&self) -> Vec<Array<F, IxDyn>> {
        Vec::new()
    }

    fn set_parameters(&mut self, params: Vec<Array<F, IxDyn>>) -> Result<()> {
        let dw_count = self.depthwise.parameter_count();
        // Split into depthwise and pointwise parameter sets
        // Depthwise has 2 params (weights + bias), pointwise has 2 params
        if params.len() != 4 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "MobileBlock::set_parameters: expected 4 (dw_w, dw_b, pw_w, pw_b), got {}",
                params.len()
            )));
        }
        let _ = dw_count;
        self.depthwise.set_parameters(params[..2].to_vec())?;
        self.pointwise.set_parameters(params[2..].to_vec())?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// InvertedResidual
// ---------------------------------------------------------------------------

/// MobileNetV2 inverted residual block.
///
/// Structure (with expansion factor `t`):
/// 1. **Expand** (if t > 1): 1×1 conv, `in_ch → in_ch * t`, BN + ReLU6
/// 2. **Depthwise**: 3×3 depthwise conv, BN + ReLU6
/// 3. **Project**: 1×1 linear conv, `in_ch * t → out_ch` (no activation)
/// 4. **Residual**: added when `stride == 1` AND `in_ch == out_ch`
pub struct InvertedResidual<F: Float + Debug + Send + Sync + NumAssign> {
    /// Optional expansion pointwise conv (None when expansion == 1)
    expand_conv: Option<PointwiseConv2d<F>>,
    depthwise: DepthwiseConv2d<F>,
    project_conv: PointwiseConv2d<F>,
    in_channels: usize,
    out_channels: usize,
    use_residual: bool,
    /// BN params for the expand stage (if present)
    exp_gamma: Option<Array<F, IxDyn>>,
    exp_beta: Option<Array<F, IxDyn>>,
    /// BN params for the depthwise stage
    dw_gamma: Array<F, IxDyn>,
    dw_beta: Array<F, IxDyn>,
    bn_eps: F,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> InvertedResidual<F> {
    /// Create a new `InvertedResidual` block.
    ///
    /// # Arguments
    /// * `in_channels`  – Input channels
    /// * `out_channels` – Output channels
    /// * `stride`       – Depthwise conv stride (1 or 2)
    /// * `expansion`    – Channel expansion factor (1 or 6)
    /// * `rng`          – Random number generator
    pub fn new<R: Rng>(
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize),
        expansion: usize,
        rng: &mut R,
    ) -> Result<Self> {
        if expansion == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "InvertedResidual: expansion must be >= 1".to_string(),
            ));
        }
        let mid_channels = in_channels * expansion;
        let (expand_conv, exp_gamma, exp_beta) = if expansion == 1 {
            (None, None, None)
        } else {
            let ec = PointwiseConv2d::new(in_channels, mid_channels, rng)?;
            let g = Array::from_elem(IxDyn(&[mid_channels]), F::one());
            let b = Array::zeros(IxDyn(&[mid_channels]));
            (Some(ec), Some(g), Some(b))
        };
        let dw = DepthwiseConv2d::new(mid_channels, (3, 3), stride, (1, 1), rng)?;
        let pw = PointwiseConv2d::new(mid_channels, out_channels, rng)?;
        let dw_g = Array::from_elem(IxDyn(&[mid_channels]), F::one());
        let dw_b = Array::zeros(IxDyn(&[mid_channels]));
        let bn_eps = F::from(1e-5_f64).unwrap_or_else(F::zero);
        let use_residual = stride == (1, 1) && in_channels == out_channels;
        Ok(Self {
            expand_conv,
            depthwise: dw,
            project_conv: pw,
            in_channels,
            out_channels,
            use_residual,
            exp_gamma,
            exp_beta,
            dw_gamma: dw_g,
            dw_beta: dw_b,
            bn_eps,
        })
    }

    /// Whether this block uses a skip connection.
    pub fn use_residual(&self) -> bool {
        self.use_residual
    }

    /// Input channel count.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Output channel count.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for InvertedResidual<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut x = input.clone();
        // 1. Expand (optional)
        if let Some(ref ec) = self.expand_conv {
            let exp_out = ec.forward(&x)?;
            let g = self
                .exp_gamma
                .as_ref()
                .ok_or_else(|| NeuralError::InferenceError("Missing exp_gamma".to_string()))?;
            let b = self
                .exp_beta
                .as_ref()
                .ok_or_else(|| NeuralError::InferenceError("Missing exp_beta".to_string()))?;
            let bn = batch_norm_forward(&exp_out, g, b, self.bn_eps)?;
            x = bn.mapv(relu6);
        }
        // 2. Depthwise
        let dw_out = self.depthwise.forward(&x)?;
        let dw_bn = batch_norm_forward(&dw_out, &self.dw_gamma, &self.dw_beta, self.bn_eps)?;
        x = dw_bn.mapv(relu6);
        // 3. Project (no activation)
        let proj = self.project_conv.forward(&x)?;
        // 4. Residual
        if self.use_residual {
            Ok(proj + input)
        } else {
            Ok(proj)
        }
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "InvertedResidual"
    }

    fn parameter_count(&self) -> usize {
        let exp = self
            .expand_conv
            .as_ref()
            .map(|e| e.parameter_count())
            .unwrap_or(0);
        exp + self.depthwise.parameter_count() + self.project_conv.parameter_count()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> ParamLayer<F>
    for InvertedResidual<F>
{

    fn get_parameters(&self) -> Vec<Array<F, IxDyn>> {
        let mut p = Vec::new();
        if let Some(ref ec) = self.expand_conv {
            p.extend(ec.get_parameters());
        }
        p.extend(self.depthwise.get_parameters());
        p.extend(self.project_conv.get_parameters());
        p
    }

    fn get_gradients(&self) -> Vec<Array<F, IxDyn>> {
        Vec::new()
    }

    fn set_parameters(&mut self, _params: Vec<Array<F, IxDyn>>) -> Result<()> {
        // Not implemented for composite layers in this simplified version
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array4;
    use scirs2_core::random::rngs::SmallRng;
    use scirs2_core::random::SeedableRng;

    fn make_rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    // --- DepthwiseConv2d ---

    #[test]
    fn test_depthwise_output_shape_same_padding() {
        let mut rng = make_rng();
        let dw = DepthwiseConv2d::<f64>::new(4, (3, 3), (1, 1), (1, 1), &mut rng)
            .expect("build failed");
        let x = Array4::<f64>::zeros((2, 4, 8, 8)).into_dyn();
        let y = dw.forward(&x).expect("forward failed");
        assert_eq!(y.shape(), &[2, 4, 8, 8]);
    }

    #[test]
    fn test_depthwise_output_shape_valid_padding() {
        let mut rng = make_rng();
        let dw = DepthwiseConv2d::<f64>::new(4, (3, 3), (1, 1), (0, 0), &mut rng)
            .expect("build failed");
        let x = Array4::<f64>::zeros((2, 4, 8, 8)).into_dyn();
        let y = dw.forward(&x).expect("forward failed");
        assert_eq!(y.shape(), &[2, 4, 6, 6]);
    }

    #[test]
    fn test_depthwise_wrong_channels_error() {
        let mut rng = make_rng();
        let dw = DepthwiseConv2d::<f64>::new(4, (3, 3), (1, 1), (1, 1), &mut rng)
            .expect("build failed");
        let x = Array4::<f64>::zeros((2, 8, 8, 8)).into_dyn(); // 8 != 4
        assert!(dw.forward(&x).is_err());
    }

    #[test]
    fn test_depthwise_parameter_count() {
        let mut rng = make_rng();
        let dw = DepthwiseConv2d::<f64>::new(8, (3, 3), (1, 1), (1, 1), &mut rng)
            .expect("build failed");
        // 8 * 1 * 3 * 3 = 72 weights + 8 biases = 80
        assert_eq!(dw.parameter_count(), 72 + 8);
    }

    #[test]
    fn test_depthwise_layer_type() {
        let mut rng = make_rng();
        let dw = DepthwiseConv2d::<f64>::new(4, (3, 3), (1, 1), (1, 1), &mut rng)
            .expect("build failed");
        assert_eq!(dw.layer_type(), "DepthwiseConv2d");
    }

    // --- PointwiseConv2d ---

    #[test]
    fn test_pointwise_output_shape() {
        let mut rng = make_rng();
        let pw = PointwiseConv2d::<f64>::new(4, 16, &mut rng).expect("build failed");
        let x = Array4::<f64>::zeros((2, 4, 8, 8)).into_dyn();
        let y = pw.forward(&x).expect("forward failed");
        assert_eq!(y.shape(), &[2, 16, 8, 8]);
    }

    #[test]
    fn test_pointwise_parameter_count() {
        let mut rng = make_rng();
        let pw = PointwiseConv2d::<f64>::new(4, 16, &mut rng).expect("build failed");
        // 16 * 4 = 64 weights + 16 biases = 80
        assert_eq!(pw.parameter_count(), 64 + 16);
    }

    // --- MobileBlock ---

    #[test]
    fn test_mobile_block_output_shape() {
        let mut rng = make_rng();
        let block =
            MobileBlock::<f64>::new(4, 8, (3, 3), (1, 1), (1, 1), &mut rng).expect("build failed");
        let x = Array4::<f64>::from_elem((2, 4, 8, 8), 0.5).into_dyn();
        let y = block.forward(&x).expect("forward failed");
        assert_eq!(y.shape(), &[2, 8, 8, 8]);
    }

    #[test]
    fn test_mobile_block_layer_type() {
        let mut rng = make_rng();
        let block =
            MobileBlock::<f64>::new(4, 8, (3, 3), (1, 1), (1, 1), &mut rng).expect("build failed");
        assert_eq!(block.layer_type(), "MobileBlock");
    }

    // --- InvertedResidual ---

    #[test]
    fn test_inverted_residual_with_residual() {
        let mut rng = make_rng();
        let ir =
            InvertedResidual::<f64>::new(8, 8, (1, 1), 6, &mut rng).expect("build failed");
        assert!(ir.use_residual(), "stride=1, in==out => residual");
        let x = Array4::<f64>::from_elem((2, 8, 8, 8), 0.1).into_dyn();
        let y = ir.forward(&x).expect("forward failed");
        assert_eq!(y.shape(), &[2, 8, 8, 8]);
    }

    #[test]
    fn test_inverted_residual_stride2_no_residual() {
        let mut rng = make_rng();
        let ir =
            InvertedResidual::<f64>::new(8, 16, (2, 2), 6, &mut rng).expect("build failed");
        assert!(!ir.use_residual(), "stride=2 => no residual");
        let x = Array4::<f64>::from_elem((2, 8, 8, 8), 0.1).into_dyn();
        let y = ir.forward(&x).expect("forward failed");
        assert_eq!(y.shape()[0], 2);
        assert_eq!(y.shape()[1], 16);
    }

    #[test]
    fn test_inverted_residual_expansion_1() {
        let mut rng = make_rng();
        let ir =
            InvertedResidual::<f64>::new(4, 4, (1, 1), 1, &mut rng).expect("build failed");
        assert!(ir.use_residual());
        let x = Array4::<f64>::from_elem((1, 4, 4, 4), 0.1).into_dyn();
        let y = ir.forward(&x).expect("forward failed");
        assert_eq!(y.shape(), &[1, 4, 4, 4]);
    }
}
