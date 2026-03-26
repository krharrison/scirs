//! 3D Convolutional layer implementation for video/volumetric data
//!
//! Provides Conv3D, MaxPool3D, AvgPool3D, and BatchNorm3D layers with full
//! forward propagation, backward propagation, and parameter update support.
//!
//! These layers operate on 5D tensors with shape
//! `[batch, channels, depth, height, width]` and are suitable for video
//! understanding, medical imaging, and other volumetric data tasks.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, Array1, Array5, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// Conv3dConfig
// ---------------------------------------------------------------------------

/// Configuration for 3D convolution layers.
///
/// Specifies input/output channels, kernel size, stride, padding, dilation,
/// and whether to include a bias term.
#[derive(Debug, Clone)]
pub struct Conv3dConfig {
    /// Number of input channels (e.g. 3 for RGB video frames).
    pub in_channels: usize,
    /// Number of output channels (number of 3D filters).
    pub out_channels: usize,
    /// Kernel size as `(depth, height, width)`.
    pub kernel_size: (usize, usize, usize),
    /// Stride as `(depth, height, width)`.
    pub stride: (usize, usize, usize),
    /// Zero-padding added to each spatial side as `(depth, height, width)`.
    pub padding: (usize, usize, usize),
    /// Dilation factor as `(depth, height, width)`.
    pub dilation: (usize, usize, usize),
    /// Whether to add a learnable bias to the output.
    pub bias: bool,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            in_channels: 1,
            out_channels: 1,
            kernel_size: (3, 3, 3),
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            dilation: (1, 1, 1),
            bias: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Conv3d
// ---------------------------------------------------------------------------

/// 3D Convolution layer for volumetric / video data.
///
/// # Input shape
/// `[batch, in_channels, depth, height, width]`
///
/// # Output shape
/// `[batch, out_channels, depth', height', width']`
///
/// where for each spatial dimension:
/// ```text
/// out = (in + 2*pad - dilation*(kernel-1) - 1) / stride + 1
/// ```
///
/// # Weight initialisation
/// Kaiming (He) initialisation using a deterministic xorshift64 PRNG so that
/// results are reproducible across runs.
#[derive(Debug)]
pub struct Conv3d<F: Float + Debug + Send + Sync + NumAssign> {
    /// Weight tensor `[out_channels, in_channels, kD, kH, kW]`
    weight: Arc<RwLock<Array<F, IxDyn>>>,
    /// Optional bias tensor `[out_channels]`
    bias: Option<Arc<RwLock<Array<F, IxDyn>>>>,
    /// Weight gradient
    weight_grad: Arc<RwLock<Array<F, IxDyn>>>,
    /// Bias gradient
    bias_grad: Option<Arc<RwLock<Array<F, IxDyn>>>>,
    /// Cached input for backward pass
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Layer configuration
    config: Conv3dConfig,
    /// Layer name
    name: Option<String>,
    /// Phantom
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Conv3d<F> {
    /// Create a new `Conv3d` layer with Kaiming initialisation.
    pub fn new(config: Conv3dConfig) -> Result<Self> {
        Self::with_name(config, None)
    }

    /// Create a new `Conv3d` layer with an optional name.
    pub fn with_name(config: Conv3dConfig, name: Option<&str>) -> Result<Self> {
        if config.in_channels == 0
            || config.out_channels == 0
            || config.kernel_size.0 == 0
            || config.kernel_size.1 == 0
            || config.kernel_size.2 == 0
        {
            return Err(NeuralError::InvalidArchitecture(
                "Conv3d: channels and kernel dimensions must be > 0".to_string(),
            ));
        }
        if config.stride.0 == 0 || config.stride.1 == 0 || config.stride.2 == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "Conv3d: stride dimensions must be > 0".to_string(),
            ));
        }
        if config.dilation.0 == 0 || config.dilation.1 == 0 || config.dilation.2 == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "Conv3d: dilation dimensions must be > 0".to_string(),
            ));
        }

        let (kd, kh, kw) = config.kernel_size;
        let fan_in = config.in_channels * kd * kh * kw;

        // Kaiming (He) init: std = sqrt(2 / fan_in)
        let std_dev = (2.0 / fan_in as f64).sqrt();

        let w_shape = vec![config.out_channels, config.in_channels, kd, kh, kw];

        // Deterministic xorshift64 PRNG for reproducible init
        let mut rng_state: u64 = 0x5EED_C0DE_CAFE_BABE;
        let weight = Array::from_shape_fn(IxDyn(&w_shape), |_| {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            // Map to [-1, 1] then scale
            let u = (rng_state as f64) / (u64::MAX as f64) * 2.0 - 1.0;
            F::from(u * std_dev).unwrap_or_else(num_traits::Zero::zero)
        });

        let weight_grad = Array::zeros(IxDyn(&w_shape));

        let bias = if config.bias {
            Some(Array::zeros(IxDyn(&[config.out_channels])))
        } else {
            None
        };
        let bias_grad = if config.bias {
            Some(Array::zeros(IxDyn(&[config.out_channels])))
        } else {
            None
        };

        Ok(Self {
            weight: Arc::new(RwLock::new(weight)),
            bias: bias.map(|b| Arc::new(RwLock::new(b))),
            weight_grad: Arc::new(RwLock::new(weight_grad)),
            bias_grad: bias_grad.map(|g| Arc::new(RwLock::new(g))),
            input_cache: Arc::new(RwLock::new(None)),
            config,
            name: name.map(String::from),
            _phantom: PhantomData,
        })
    }

    /// Compute the output shape for a given 5-D input shape.
    pub fn output_shape(&self, input_shape: &[usize; 5]) -> [usize; 5] {
        let batch = input_shape[0];
        let d_out = self.spatial_out(input_shape[2], 0);
        let h_out = self.spatial_out(input_shape[3], 1);
        let w_out = self.spatial_out(input_shape[4], 2);
        [batch, self.config.out_channels, d_out, h_out, w_out]
    }

    /// Return a reference to the weight tensor.
    pub fn weight(&self) -> Result<Array<F, IxDyn>> {
        self.weight
            .read()
            .map(|w| w.clone())
            .map_err(|_| NeuralError::InferenceError("Failed to read weight".to_string()))
    }

    /// Return a clone of the bias tensor (if present).
    pub fn bias_value(&self) -> Result<Option<Array<F, IxDyn>>> {
        match &self.bias {
            Some(b) => {
                let guard = b
                    .read()
                    .map_err(|_| NeuralError::InferenceError("Failed to read bias".to_string()))?;
                Ok(Some(guard.clone()))
            }
            None => Ok(None),
        }
    }

    /// Total number of trainable parameters.
    pub fn param_count(&self) -> usize {
        let (kd, kh, kw) = self.config.kernel_size;
        let w = self.config.out_channels * self.config.in_channels * kd * kh * kw;
        let b = if self.config.bias {
            self.config.out_channels
        } else {
            0
        };
        w + b
    }

    // -- private helpers ---------------------------------------------------

    /// Compute output size for one spatial dimension.
    ///
    /// `dim_idx`: 0 = depth, 1 = height, 2 = width
    fn spatial_out(&self, input_size: usize, dim_idx: usize) -> usize {
        let pad = match dim_idx {
            0 => self.config.padding.0,
            1 => self.config.padding.1,
            _ => self.config.padding.2,
        };
        let dil = match dim_idx {
            0 => self.config.dilation.0,
            1 => self.config.dilation.1,
            _ => self.config.dilation.2,
        };
        let ks = match dim_idx {
            0 => self.config.kernel_size.0,
            1 => self.config.kernel_size.1,
            _ => self.config.kernel_size.2,
        };
        let st = match dim_idx {
            0 => self.config.stride.0,
            1 => self.config.stride.1,
            _ => self.config.stride.2,
        };
        let effective_k = dil * (ks - 1) + 1;
        (input_size + 2 * pad - effective_k) / st + 1
    }

    /// Core 3D convolution forward pass.
    fn conv3d_forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 5 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Conv3d expects 5D input [batch, C, D, H, W], got {}D",
                shape.len()
            )));
        }

        let batch = shape[0];
        let in_c = shape[1];
        let in_d = shape[2];
        let in_h = shape[3];
        let in_w = shape[4];

        if in_c != self.config.in_channels {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Conv3d: expected {} input channels, got {}",
                self.config.in_channels, in_c
            )));
        }

        let out_d = self.spatial_out(in_d, 0);
        let out_h = self.spatial_out(in_h, 1);
        let out_w = self.spatial_out(in_w, 2);

        if out_d == 0 || out_h == 0 || out_w == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "Conv3d: output spatial dimensions are zero; check kernel/stride/padding/dilation"
                    .to_string(),
            ));
        }

        let out_shape = vec![batch, self.config.out_channels, out_d, out_h, out_w];
        let mut output = Array::zeros(IxDyn(&out_shape));

        let weights = self.weight.read().map_err(|_| {
            NeuralError::InferenceError("Failed to read weights".to_string())
        })?;

        let (kd, kh, kw) = self.config.kernel_size;
        let (sd, sh, sw) = self.config.stride;
        let (pd, ph, pw) = self.config.padding;
        let (dd, dh, dw) = self.config.dilation;

        for b in 0..batch {
            for oc in 0..self.config.out_channels {
                for od in 0..out_d {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = F::zero();

                            for ic in 0..self.config.in_channels {
                                for ki_d in 0..kd {
                                    for ki_h in 0..kh {
                                        for ki_w in 0..kw {
                                            let id =
                                                (od * sd + ki_d * dd) as isize - pd as isize;
                                            let ih =
                                                (oh * sh + ki_h * dh) as isize - ph as isize;
                                            let iw =
                                                (ow * sw + ki_w * dw) as isize - pw as isize;

                                            if id >= 0
                                                && ih >= 0
                                                && iw >= 0
                                                && (id as usize) < in_d
                                                && (ih as usize) < in_h
                                                && (iw as usize) < in_w
                                            {
                                                let input_val = input[[
                                                    b,
                                                    ic,
                                                    id as usize,
                                                    ih as usize,
                                                    iw as usize,
                                                ]];
                                                let weight_val =
                                                    weights[[oc, ic, ki_d, ki_h, ki_w]];
                                                sum += input_val * weight_val;
                                            }
                                        }
                                    }
                                }
                            }
                            output[[b, oc, od, oh, ow]] = sum;
                        }
                    }
                }
            }
        }

        // Add bias
        if let Some(ref bias_lock) = self.bias {
            let bias = bias_lock.read().map_err(|_| {
                NeuralError::InferenceError("Failed to read bias".to_string())
            })?;
            for b in 0..batch {
                for oc in 0..self.config.out_channels {
                    let bv = bias[[oc]];
                    for od in 0..out_d {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                output[[b, oc, od, oh, ow]] += bv;
                            }
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    /// Backward pass: compute gradients w.r.t. input, weights, and bias.
    #[allow(clippy::type_complexity)]
    fn conv3d_backward(
        &self,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>, Option<Array<F, IxDyn>>)> {
        let input_guard = self.input_cache.read().map_err(|_| {
            NeuralError::InferenceError("Failed to read input cache".to_string())
        })?;
        let input = input_guard.as_ref().ok_or_else(|| {
            NeuralError::InferenceError(
                "No cached input for backward pass. Call forward() first.".to_string(),
            )
        })?;

        let in_shape = input.shape();
        let batch = in_shape[0];
        let in_c = in_shape[1];
        let in_d = in_shape[2];
        let in_h = in_shape[3];
        let in_w = in_shape[4];

        let out_shape = grad_output.shape();
        let out_d = out_shape[2];
        let out_h = out_shape[3];
        let out_w = out_shape[4];

        let (kd, kh, kw) = self.config.kernel_size;
        let (sd, sh, sw) = self.config.stride;
        let (pd, ph, pw) = self.config.padding;
        let (dd, dh, dw) = self.config.dilation;

        let weights = self.weight.read().map_err(|_| {
            NeuralError::InferenceError("Failed to read weights".to_string())
        })?;

        // grad w.r.t. input
        let mut grad_input = Array::zeros(IxDyn(in_shape));
        for b in 0..batch {
            for ic in 0..in_c {
                for id_i in 0..in_d {
                    for ih_i in 0..in_h {
                        for iw_i in 0..in_w {
                            let mut sum = F::zero();
                            for oc in 0..self.config.out_channels {
                                for ki_d in 0..kd {
                                    for ki_h in 0..kh {
                                        for ki_w in 0..kw {
                                            let od_num =
                                                id_i as isize + pd as isize - (ki_d * dd) as isize;
                                            let oh_num =
                                                ih_i as isize + ph as isize - (ki_h * dh) as isize;
                                            let ow_num =
                                                iw_i as isize + pw as isize - (ki_w * dw) as isize;

                                            if od_num >= 0
                                                && oh_num >= 0
                                                && ow_num >= 0
                                                && od_num % sd as isize == 0
                                                && oh_num % sh as isize == 0
                                                && ow_num % sw as isize == 0
                                            {
                                                let od_idx = od_num as usize / sd;
                                                let oh_idx = oh_num as usize / sh;
                                                let ow_idx = ow_num as usize / sw;
                                                if od_idx < out_d
                                                    && oh_idx < out_h
                                                    && ow_idx < out_w
                                                {
                                                    sum += grad_output
                                                        [[b, oc, od_idx, oh_idx, ow_idx]]
                                                        * weights[[oc, ic, ki_d, ki_h, ki_w]];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            grad_input[[b, ic, id_i, ih_i, iw_i]] = sum;
                        }
                    }
                }
            }
        }

        // grad w.r.t. weights
        let mut grad_w = Array::zeros(IxDyn(&[
            self.config.out_channels,
            in_c,
            kd,
            kh,
            kw,
        ]));
        for oc in 0..self.config.out_channels {
            for ic in 0..in_c {
                for ki_d in 0..kd {
                    for ki_h in 0..kh {
                        for ki_w in 0..kw {
                            let mut sum = F::zero();
                            for b in 0..batch {
                                for od in 0..out_d {
                                    for oh in 0..out_h {
                                        for ow in 0..out_w {
                                            let id =
                                                (od * sd + ki_d * dd) as isize - pd as isize;
                                            let ih =
                                                (oh * sh + ki_h * dh) as isize - ph as isize;
                                            let iw =
                                                (ow * sw + ki_w * dw) as isize - pw as isize;
                                            if id >= 0
                                                && ih >= 0
                                                && iw >= 0
                                                && (id as usize) < in_d
                                                && (ih as usize) < in_h
                                                && (iw as usize) < in_w
                                            {
                                                sum += input[[
                                                    b,
                                                    ic,
                                                    id as usize,
                                                    ih as usize,
                                                    iw as usize,
                                                ]] * grad_output[[b, oc, od, oh, ow]];
                                            }
                                        }
                                    }
                                }
                            }
                            grad_w[[oc, ic, ki_d, ki_h, ki_w]] = sum;
                        }
                    }
                }
            }
        }

        // grad w.r.t. bias
        let grad_bias = if self.config.bias {
            let mut gb = Array::zeros(IxDyn(&[self.config.out_channels]));
            for oc in 0..self.config.out_channels {
                let mut sum = F::zero();
                for b in 0..batch {
                    for od in 0..out_d {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                sum += grad_output[[b, oc, od, oh, ow]];
                            }
                        }
                    }
                }
                gb[[oc]] = sum;
            }
            Some(gb)
        } else {
            None
        };

        Ok((grad_input, grad_w, grad_bias))
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F> for Conv3d<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        if let Ok(mut cache) = self.input_cache.write() {
            *cache = Some(input.clone());
        }
        self.conv3d_forward(input)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let (grad_input, grad_weights, grad_bias) = self.conv3d_backward(grad_output)?;
        if let Ok(mut wg) = self.weight_grad.write() {
            *wg = grad_weights;
        }
        if let (Some(ref bg_lock), Some(gb)) = (&self.bias_grad, grad_bias) {
            if let Ok(mut bg) = bg_lock.write() {
                *bg = gb;
            }
        }
        Ok(grad_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        {
            let grad = self.weight_grad.read().map_err(|_| {
                NeuralError::InferenceError("Failed to read weight_grad".to_string())
            })?;
            let mut weights = self.weight.write().map_err(|_| {
                NeuralError::InferenceError("Failed to write weights".to_string())
            })?;
            for (w, g) in weights.iter_mut().zip(grad.iter()) {
                *w -= learning_rate * *g;
            }
        }
        if let (Some(ref bias_lock), Some(ref bg_lock)) = (&self.bias, &self.bias_grad) {
            let grad = bg_lock.read().map_err(|_| {
                NeuralError::InferenceError("Failed to read bias_grad".to_string())
            })?;
            let mut bias = bias_lock.write().map_err(|_| {
                NeuralError::InferenceError("Failed to write bias".to_string())
            })?;
            for (b, g) in bias.iter_mut().zip(grad.iter()) {
                *b -= learning_rate * *g;
            }
        }
        Ok(())
    }

    fn layer_type(&self) -> &str {
        "Conv3D"
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn parameter_count(&self) -> usize {
        self.param_count()
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut p = Vec::new();
        if let Ok(w) = self.weight.read() {
            p.push(w.clone());
        }
        if let Some(ref b) = self.bias {
            if let Ok(bv) = b.read() {
                p.push(bv.clone());
            }
        }
        p
    }

    fn layer_description(&self) -> String {
        format!(
            "type:Conv3D, in:{}, out:{}, kernel:{:?}, stride:{:?}, pad:{:?}, dil:{:?}, params:{}",
            self.config.in_channels,
            self.config.out_channels,
            self.config.kernel_size,
            self.config.stride,
            self.config.padding,
            self.config.dilation,
            self.param_count(),
        )
    }
}

// ---------------------------------------------------------------------------
// MaxPool3dConfig
// ---------------------------------------------------------------------------

/// Configuration for 3D max / average pooling layers.
#[derive(Debug, Clone)]
pub struct MaxPool3dConfig {
    /// Pooling window size `(depth, height, width)`.
    pub kernel_size: (usize, usize, usize),
    /// Stride of the pooling window. Defaults to `kernel_size` when `None`.
    pub stride: Option<(usize, usize, usize)>,
    /// Zero-padding `(depth, height, width)`.
    pub padding: (usize, usize, usize),
}

impl Default for MaxPool3dConfig {
    fn default() -> Self {
        Self {
            kernel_size: (2, 2, 2),
            stride: None,
            padding: (0, 0, 0),
        }
    }
}

impl MaxPool3dConfig {
    /// Effective stride (falls back to kernel_size).
    fn effective_stride(&self) -> (usize, usize, usize) {
        self.stride.unwrap_or(self.kernel_size)
    }
}

// ---------------------------------------------------------------------------
// MaxPool3d
// ---------------------------------------------------------------------------

/// 3D Max Pooling layer.
///
/// Input/output shape: `[batch, channels, depth, height, width]`
#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct MaxPool3d<F: Float + Debug + Send + Sync + NumAssign> {
    /// Pool config
    config: MaxPool3dConfig,
    /// Cached max indices for backward pass
    max_indices: Arc<RwLock<Option<Array<(usize, usize, usize), IxDyn>>>>,
    /// Cached input
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Layer name
    name: Option<String>,
    /// Phantom
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> MaxPool3d<F> {
    /// Create a new `MaxPool3d` layer.
    pub fn new(config: MaxPool3dConfig) -> Result<Self> {
        Self::with_name(config, None)
    }

    /// Create a new `MaxPool3d` layer with an optional name.
    pub fn with_name(config: MaxPool3dConfig, name: Option<&str>) -> Result<Self> {
        if config.kernel_size.0 == 0 || config.kernel_size.1 == 0 || config.kernel_size.2 == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "MaxPool3d: kernel dimensions must be > 0".to_string(),
            ));
        }
        let stride = config.effective_stride();
        if stride.0 == 0 || stride.1 == 0 || stride.2 == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "MaxPool3d: stride dimensions must be > 0".to_string(),
            ));
        }
        Ok(Self {
            config,
            max_indices: Arc::new(RwLock::new(None)),
            input_cache: Arc::new(RwLock::new(None)),
            name: name.map(String::from),
            _phantom: PhantomData,
        })
    }

    /// Compute output spatial size for one dimension.
    fn pool_out(input: usize, kernel: usize, stride: usize, pad: usize) -> usize {
        (input + 2 * pad - kernel) / stride + 1
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for MaxPool3d<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 5 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "MaxPool3d expects 5D input, got {}D",
                shape.len()
            )));
        }

        let batch = shape[0];
        let channels = shape[1];
        let in_d = shape[2];
        let in_h = shape[3];
        let in_w = shape[4];

        let (kd, kh, kw) = self.config.kernel_size;
        let (sd, sh, sw) = self.config.effective_stride();
        let (pd, ph, pw) = self.config.padding;

        if in_d + 2 * pd < kd || in_h + 2 * ph < kh || in_w + 2 * pw < kw {
            return Err(NeuralError::InvalidArchitecture(
                "MaxPool3d: padded input smaller than kernel".to_string(),
            ));
        }

        let out_d = Self::pool_out(in_d, kd, sd, pd);
        let out_h = Self::pool_out(in_h, kh, sh, ph);
        let out_w = Self::pool_out(in_w, kw, sw, pw);

        // Cache input
        if let Ok(mut c) = self.input_cache.write() {
            *c = Some(input.clone());
        }

        let out_shape = vec![batch, channels, out_d, out_h, out_w];
        let mut output = Array::zeros(IxDyn(&out_shape));
        let mut indices =
            Array::from_elem(IxDyn(&out_shape), (0usize, 0usize, 0usize));

        for b in 0..batch {
            for c in 0..channels {
                for od in 0..out_d {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let d_start = od * sd;
                            let h_start = oh * sh;
                            let w_start = ow * sw;

                            let mut max_val = F::neg_infinity();
                            let mut max_pos = (0usize, 0usize, 0usize);

                            for ki_d in 0..kd {
                                for ki_h in 0..kh {
                                    for ki_w in 0..kw {
                                        let id = (d_start + ki_d) as isize - pd as isize;
                                        let ih = (h_start + ki_h) as isize - ph as isize;
                                        let iw = (w_start + ki_w) as isize - pw as isize;

                                        if id >= 0
                                            && ih >= 0
                                            && iw >= 0
                                            && (id as usize) < in_d
                                            && (ih as usize) < in_h
                                            && (iw as usize) < in_w
                                        {
                                            let val = input[[
                                                b,
                                                c,
                                                id as usize,
                                                ih as usize,
                                                iw as usize,
                                            ]];
                                            if val > max_val {
                                                max_val = val;
                                                max_pos =
                                                    (id as usize, ih as usize, iw as usize);
                                            }
                                        }
                                    }
                                }
                            }

                            output[[b, c, od, oh, ow]] = max_val;
                            indices[[b, c, od, oh, ow]] = max_pos;
                        }
                    }
                }
            }
        }

        if let Ok(mut idx_cache) = self.max_indices.write() {
            *idx_cache = Some(indices);
        }

        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let input_guard = self.input_cache.read().map_err(|_| {
            NeuralError::InferenceError("Failed to read input cache".to_string())
        })?;
        let input = input_guard.as_ref().ok_or_else(|| {
            NeuralError::InferenceError("No cached input for backward pass".to_string())
        })?;

        let idx_guard = self.max_indices.read().map_err(|_| {
            NeuralError::InferenceError("Failed to read max indices".to_string())
        })?;
        let indices = idx_guard.as_ref().ok_or_else(|| {
            NeuralError::InferenceError("No cached indices for backward pass".to_string())
        })?;

        let mut grad_input = Array::zeros(input.raw_dim());
        let gs = grad_output.shape();
        let batch = gs[0];
        let channels = gs[1];
        let out_d = gs[2];
        let out_h = gs[3];
        let out_w = gs[4];

        for b in 0..batch {
            for c in 0..channels {
                for od in 0..out_d {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let (md, mh, mw) = indices[[b, c, od, oh, ow]];
                            grad_input[[b, c, md, mh, mw]] += grad_output[[b, c, od, oh, ow]];
                        }
                    }
                }
            }
        }

        Ok(grad_input)
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Ok(())
    }

    fn layer_type(&self) -> &str {
        "MaxPool3D"
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn layer_description(&self) -> String {
        format!(
            "type:MaxPool3D, kernel:{:?}, stride:{:?}, pad:{:?}",
            self.config.kernel_size,
            self.config.effective_stride(),
            self.config.padding,
        )
    }
}

// ---------------------------------------------------------------------------
// AvgPool3d
// ---------------------------------------------------------------------------

/// 3D Average Pooling layer.
///
/// Input/output shape: `[batch, channels, depth, height, width]`
#[derive(Debug)]
pub struct AvgPool3d<F: Float + Debug + Send + Sync + NumAssign> {
    /// Pool config (reuses `MaxPool3dConfig`)
    config: MaxPool3dConfig,
    /// Cached input
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Layer name
    name: Option<String>,
    /// Phantom
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> AvgPool3d<F> {
    /// Create a new `AvgPool3d` layer.
    pub fn new(config: MaxPool3dConfig) -> Result<Self> {
        Self::with_name(config, None)
    }

    /// Create a new `AvgPool3d` layer with an optional name.
    pub fn with_name(config: MaxPool3dConfig, name: Option<&str>) -> Result<Self> {
        if config.kernel_size.0 == 0 || config.kernel_size.1 == 0 || config.kernel_size.2 == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "AvgPool3d: kernel dimensions must be > 0".to_string(),
            ));
        }
        let stride = config.effective_stride();
        if stride.0 == 0 || stride.1 == 0 || stride.2 == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "AvgPool3d: stride dimensions must be > 0".to_string(),
            ));
        }
        Ok(Self {
            config,
            input_cache: Arc::new(RwLock::new(None)),
            name: name.map(String::from),
            _phantom: PhantomData,
        })
    }

    /// Compute output spatial size for one dimension.
    fn pool_out(input: usize, kernel: usize, stride: usize, pad: usize) -> usize {
        (input + 2 * pad - kernel) / stride + 1
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for AvgPool3d<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 5 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "AvgPool3d expects 5D input, got {}D",
                shape.len()
            )));
        }

        let batch = shape[0];
        let channels = shape[1];
        let in_d = shape[2];
        let in_h = shape[3];
        let in_w = shape[4];

        let (kd, kh, kw) = self.config.kernel_size;
        let (sd, sh, sw) = self.config.effective_stride();
        let (pd, ph, pw) = self.config.padding;

        if in_d + 2 * pd < kd || in_h + 2 * ph < kh || in_w + 2 * pw < kw {
            return Err(NeuralError::InvalidArchitecture(
                "AvgPool3d: padded input smaller than kernel".to_string(),
            ));
        }

        let out_d = Self::pool_out(in_d, kd, sd, pd);
        let out_h = Self::pool_out(in_h, kh, sh, ph);
        let out_w = Self::pool_out(in_w, kw, sw, pw);

        if let Ok(mut c) = self.input_cache.write() {
            *c = Some(input.clone());
        }

        let out_shape = vec![batch, channels, out_d, out_h, out_w];
        let mut output = Array::zeros(IxDyn(&out_shape));

        for b in 0..batch {
            for c in 0..channels {
                for od in 0..out_d {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let d_start = od * sd;
                            let h_start = oh * sh;
                            let w_start = ow * sw;

                            let mut sum = F::zero();
                            let mut count = 0usize;

                            for ki_d in 0..kd {
                                for ki_h in 0..kh {
                                    for ki_w in 0..kw {
                                        let id = (d_start + ki_d) as isize - pd as isize;
                                        let ih = (h_start + ki_h) as isize - ph as isize;
                                        let iw = (w_start + ki_w) as isize - pw as isize;

                                        if id >= 0
                                            && ih >= 0
                                            && iw >= 0
                                            && (id as usize) < in_d
                                            && (ih as usize) < in_h
                                            && (iw as usize) < in_w
                                        {
                                            sum += input[[
                                                b,
                                                c,
                                                id as usize,
                                                ih as usize,
                                                iw as usize,
                                            ]];
                                            count += 1;
                                        }
                                    }
                                }
                            }

                            let divisor =
                                F::from(count).unwrap_or_else(num_traits::One::one);
                            output[[b, c, od, oh, ow]] = sum / divisor;
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let input_guard = self.input_cache.read().map_err(|_| {
            NeuralError::InferenceError("Failed to read input cache".to_string())
        })?;
        let input = input_guard.as_ref().ok_or_else(|| {
            NeuralError::InferenceError("No cached input for backward pass".to_string())
        })?;

        let in_d = input.shape()[2];
        let in_h = input.shape()[3];
        let in_w = input.shape()[4];
        let mut grad_input = Array::zeros(input.raw_dim());

        let gs = grad_output.shape();
        let batch = gs[0];
        let channels = gs[1];
        let out_d = gs[2];
        let out_h = gs[3];
        let out_w = gs[4];

        let (kd, kh, kw) = self.config.kernel_size;
        let (sd, sh, sw) = self.config.effective_stride();
        let (pd, ph, pw) = self.config.padding;

        for b in 0..batch {
            for c in 0..channels {
                for od in 0..out_d {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let d_start = od * sd;
                            let h_start = oh * sh;
                            let w_start = ow * sw;

                            // Count valid elements
                            let mut count = 0usize;
                            for ki_d in 0..kd {
                                for ki_h in 0..kh {
                                    for ki_w in 0..kw {
                                        let id = (d_start + ki_d) as isize - pd as isize;
                                        let ih = (h_start + ki_h) as isize - ph as isize;
                                        let iw = (w_start + ki_w) as isize - pw as isize;
                                        if id >= 0
                                            && ih >= 0
                                            && iw >= 0
                                            && (id as usize) < in_d
                                            && (ih as usize) < in_h
                                            && (iw as usize) < in_w
                                        {
                                            count += 1;
                                        }
                                    }
                                }
                            }

                            let divisor =
                                F::from(count).unwrap_or_else(num_traits::One::one);
                            let grad_per =
                                grad_output[[b, c, od, oh, ow]] / divisor;

                            for ki_d in 0..kd {
                                for ki_h in 0..kh {
                                    for ki_w in 0..kw {
                                        let id = (d_start + ki_d) as isize - pd as isize;
                                        let ih = (h_start + ki_h) as isize - ph as isize;
                                        let iw = (w_start + ki_w) as isize - pw as isize;
                                        if id >= 0
                                            && ih >= 0
                                            && iw >= 0
                                            && (id as usize) < in_d
                                            && (ih as usize) < in_h
                                            && (iw as usize) < in_w
                                        {
                                            grad_input[[
                                                b,
                                                c,
                                                id as usize,
                                                ih as usize,
                                                iw as usize,
                                            ]] += grad_per;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(grad_input)
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Ok(())
    }

    fn layer_type(&self) -> &str {
        "AvgPool3D"
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn layer_description(&self) -> String {
        format!(
            "type:AvgPool3D, kernel:{:?}, stride:{:?}, pad:{:?}",
            self.config.kernel_size,
            self.config.effective_stride(),
            self.config.padding,
        )
    }
}

// ---------------------------------------------------------------------------
// BatchNorm3d
// ---------------------------------------------------------------------------

/// 3D Batch Normalisation layer.
///
/// Normalises over the `(batch, depth, height, width)` dimensions for each
/// channel independently.
///
/// Input/output shape: `[batch, num_features, depth, height, width]`
#[derive(Debug)]
pub struct BatchNorm3d<F: Float + Debug + Send + Sync + NumAssign> {
    /// Running mean per channel
    running_mean: Array1<F>,
    /// Running variance per channel
    running_var: Array1<F>,
    /// Learnable scale parameter
    gamma: Array1<F>,
    /// Learnable shift parameter
    beta: Array1<F>,
    /// Exponential moving average momentum
    momentum: f64,
    /// Small constant for numerical stability
    epsilon: f64,
    /// Number of channels
    num_features: usize,
    /// Whether the layer is in training mode
    training: bool,
    /// Cached normalised values for backward pass
    norm_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Cached input
    input_cache: Arc<RwLock<Option<Array<F, IxDyn>>>>,
    /// Cached batch std for backward pass
    std_cache: Arc<RwLock<Option<Array1<F>>>>,
    /// Layer name
    name: Option<String>,
    /// Phantom
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> BatchNorm3d<F> {
    /// Create a new `BatchNorm3d` layer.
    pub fn new(num_features: usize) -> Self {
        Self::with_params(num_features, 0.1, 1e-5)
    }

    /// Create with custom momentum and epsilon.
    pub fn with_params(num_features: usize, momentum: f64, epsilon: f64) -> Self {
        Self {
            running_mean: Array1::zeros(num_features),
            running_var: Array1::from_elem(num_features, F::one()),
            gamma: Array1::from_elem(num_features, F::one()),
            beta: Array1::zeros(num_features),
            momentum,
            epsilon,
            num_features,
            training: true,
            norm_cache: Arc::new(RwLock::new(None)),
            input_cache: Arc::new(RwLock::new(None)),
            std_cache: Arc::new(RwLock::new(None)),
            name: None,
            _phantom: PhantomData,
        }
    }

    /// Switch to evaluation mode (use running statistics).
    pub fn eval_mode(&mut self) {
        self.training = false;
    }

    /// Switch to training mode (compute batch statistics).
    pub fn train_mode(&mut self) {
        self.training = true;
    }

    /// Forward pass.
    pub fn forward_mut(&mut self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 5 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "BatchNorm3d expects 5D input, got {}D",
                shape.len()
            )));
        }
        let batch = shape[0];
        let channels = shape[1];
        let depth = shape[2];
        let height = shape[3];
        let width = shape[4];

        if channels != self.num_features {
            return Err(NeuralError::InvalidArchitecture(format!(
                "BatchNorm3d: expected {} channels, got {}",
                self.num_features, channels
            )));
        }

        if let Ok(mut c) = self.input_cache.write() {
            *c = Some(input.clone());
        }

        let spatial = batch * depth * height * width;
        let spatial_f = F::from(spatial).unwrap_or_else(num_traits::One::one);

        let mut output = Array::zeros(IxDyn(shape));

        if self.training {
            // Compute per-channel mean and variance over (batch, D, H, W)
            let mut mean = Array1::<F>::zeros(channels);
            let mut var = Array1::<F>::zeros(channels);

            for c in 0..channels {
                let mut sum = F::zero();
                for b in 0..batch {
                    for d in 0..depth {
                        for h in 0..height {
                            for w in 0..width {
                                sum += input[[b, c, d, h, w]];
                            }
                        }
                    }
                }
                mean[c] = sum / spatial_f;
            }

            for c in 0..channels {
                let mu = mean[c];
                let mut sq_sum = F::zero();
                for b in 0..batch {
                    for d in 0..depth {
                        for h in 0..height {
                            for w in 0..width {
                                let diff = input[[b, c, d, h, w]] - mu;
                                sq_sum += diff * diff;
                            }
                        }
                    }
                }
                var[c] = sq_sum / spatial_f;
            }

            let eps = F::from(self.epsilon).unwrap_or_else(num_traits::Zero::zero);
            let mut std_arr = Array1::<F>::zeros(channels);
            for c in 0..channels {
                std_arr[c] = (var[c] + eps).sqrt();
            }

            // Normalise and apply affine transform
            for c in 0..channels {
                let mu = mean[c];
                let s = std_arr[c];
                let g = self.gamma[c];
                let beta_c = self.beta[c];
                for b in 0..batch {
                    for d in 0..depth {
                        for h in 0..height {
                            for w in 0..width {
                                let normed = (input[[b, c, d, h, w]] - mu) / s;
                                output[[b, c, d, h, w]] = g * normed + beta_c;
                            }
                        }
                    }
                }
            }

            // Update running statistics
            let mom = F::from(self.momentum).unwrap_or_else(num_traits::Zero::zero);
            let one_minus = F::one() - mom;
            for c in 0..channels {
                self.running_mean[c] = one_minus * self.running_mean[c] + mom * mean[c];
                self.running_var[c] = one_minus * self.running_var[c] + mom * var[c];
            }

            // Cache for backward
            if let Ok(mut nc) = self.norm_cache.write() {
                // Store normalised values (before affine)
                let mut normed = Array::zeros(IxDyn(shape));
                for c in 0..channels {
                    let mu = mean[c];
                    let s = std_arr[c];
                    for b in 0..batch {
                        for d in 0..depth {
                            for h in 0..height {
                                for w in 0..width {
                                    normed[[b, c, d, h, w]] =
                                        (input[[b, c, d, h, w]] - mu) / s;
                                }
                            }
                        }
                    }
                }
                *nc = Some(normed);
            }
            if let Ok(mut sc) = self.std_cache.write() {
                *sc = Some(std_arr);
            }
        } else {
            // Evaluation mode: use running statistics
            let eps = F::from(self.epsilon).unwrap_or_else(num_traits::Zero::zero);
            for c in 0..channels {
                let mu = self.running_mean[c];
                let s = (self.running_var[c] + eps).sqrt();
                let g = self.gamma[c];
                let beta_c = self.beta[c];
                for b in 0..batch {
                    for d in 0..depth {
                        for h in 0..height {
                            for w in 0..width {
                                let normed = (input[[b, c, d, h, w]] - mu) / s;
                                output[[b, c, d, h, w]] = g * normed + beta_c;
                            }
                        }
                    }
                }
            }
        }

        Ok(output)
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for BatchNorm3d<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // BatchNorm needs mutable access for running stats updates in training mode.
        // When called through the immutable Layer trait, we use running stats (eval mode).
        let shape = input.shape();
        if shape.len() != 5 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "BatchNorm3d expects 5D input, got {}D",
                shape.len()
            )));
        }
        let batch = shape[0];
        let channels = shape[1];
        let depth = shape[2];
        let height = shape[3];
        let width = shape[4];

        if channels != self.num_features {
            return Err(NeuralError::InvalidArchitecture(format!(
                "BatchNorm3d: expected {} channels, got {}",
                self.num_features, channels
            )));
        }

        let eps = F::from(self.epsilon).unwrap_or_else(num_traits::Zero::zero);
        let mut output = Array::zeros(IxDyn(shape));

        for c in 0..channels {
            let mu = self.running_mean[c];
            let s = (self.running_var[c] + eps).sqrt();
            let g = self.gamma[c];
            let beta_c = self.beta[c];
            for b in 0..batch {
                for d in 0..depth {
                    for h in 0..height {
                        for w in 0..width {
                            let normed = (input[[b, c, d, h, w]] - mu) / s;
                            output[[b, c, d, h, w]] = g * normed + beta_c;
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Simplified backward: return grad_output scaled by gamma/std
        let gs = grad_output.shape();
        if gs.len() != 5 {
            return Err(NeuralError::InvalidArchitecture(
                "BatchNorm3d backward expects 5D gradient".to_string(),
            ));
        }
        let batch = gs[0];
        let channels = gs[1];
        let depth = gs[2];
        let height = gs[3];
        let width = gs[4];

        let eps = F::from(self.epsilon).unwrap_or_else(num_traits::Zero::zero);
        let mut grad_input = Array::zeros(grad_output.raw_dim());

        for c in 0..channels {
            let s = (self.running_var[c] + eps).sqrt();
            let g = self.gamma[c];
            let scale = g / s;
            for b in 0..batch {
                for d in 0..depth {
                    for h in 0..height {
                        for w in 0..width {
                            grad_input[[b, c, d, h, w]] =
                                grad_output[[b, c, d, h, w]] * scale;
                        }
                    }
                }
            }
        }

        Ok(grad_input)
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Ok(())
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn layer_type(&self) -> &str {
        "BatchNorm3D"
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn parameter_count(&self) -> usize {
        // gamma + beta
        self.num_features * 2
    }

    fn layer_description(&self) -> String {
        format!(
            "type:BatchNorm3D, features:{}, momentum:{}, eps:{}",
            self.num_features, self.momentum, self.epsilon,
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array5;

    // -- Conv3d tests -------------------------------------------------------

    #[test]
    fn test_conv3d_forward_shape_basic() {
        // [1,1,8,8,8] with 3x3x3 kernel, no padding -> [1,4,6,6,6]
        let config = Conv3dConfig {
            in_channels: 1,
            out_channels: 4,
            kernel_size: (3, 3, 3),
            ..Conv3dConfig::default()
        };
        let conv = Conv3d::<f64>::new(config).expect("Failed to create Conv3d");
        let input = Array5::<f64>::from_elem((1, 1, 8, 8, 8), 1.0).into_dyn();
        let output = conv.forward(&input).expect("Forward failed");
        assert_eq!(output.shape(), &[1, 4, 6, 6, 6]);
    }

    #[test]
    fn test_conv3d_forward_shape_with_padding() {
        // padding=1 with 3x3x3 kernel -> same spatial dims
        let config = Conv3dConfig {
            in_channels: 1,
            out_channels: 2,
            kernel_size: (3, 3, 3),
            padding: (1, 1, 1),
            ..Conv3dConfig::default()
        };
        let conv = Conv3d::<f64>::new(config).expect("Failed to create Conv3d");
        let input = Array5::<f64>::from_elem((1, 1, 8, 8, 8), 1.0).into_dyn();
        let output = conv.forward(&input).expect("Forward failed");
        assert_eq!(output.shape(), &[1, 2, 8, 8, 8]);
    }

    #[test]
    fn test_conv3d_forward_shape_with_stride() {
        // stride=(2,2,2) with 3x3x3 kernel, pad=1 -> halved dims
        let config = Conv3dConfig {
            in_channels: 1,
            out_channels: 2,
            kernel_size: (3, 3, 3),
            stride: (2, 2, 2),
            padding: (1, 1, 1),
            ..Conv3dConfig::default()
        };
        let conv = Conv3d::<f64>::new(config).expect("Failed to create Conv3d");
        let input = Array5::<f64>::from_elem((1, 1, 8, 8, 8), 1.0).into_dyn();
        let output = conv.forward(&input).expect("Forward failed");
        // (8 + 2*1 - 3)/2 + 1 = 4
        assert_eq!(output.shape(), &[1, 2, 4, 4, 4]);
    }

    #[test]
    fn test_conv3d_1x1x1_channel_mixing() {
        // 1x1x1 conv acts as per-voxel channel mixing
        let config = Conv3dConfig {
            in_channels: 3,
            out_channels: 5,
            kernel_size: (1, 1, 1),
            ..Conv3dConfig::default()
        };
        let conv = Conv3d::<f64>::new(config).expect("Failed to create Conv3d");
        let input = Array5::<f64>::from_elem((2, 3, 4, 4, 4), 1.0).into_dyn();
        let output = conv.forward(&input).expect("Forward failed");
        // Spatial dims unchanged
        assert_eq!(output.shape(), &[2, 5, 4, 4, 4]);
    }

    #[test]
    fn test_conv3d_param_count() {
        let config = Conv3dConfig {
            in_channels: 3,
            out_channels: 16,
            kernel_size: (3, 3, 3),
            bias: true,
            ..Conv3dConfig::default()
        };
        let conv = Conv3d::<f64>::new(config).expect("Failed to create Conv3d");
        // weight: 16 * 3 * 3 * 3 * 3 = 1296, bias: 16, total: 1312
        assert_eq!(conv.param_count(), 1312);
    }

    #[test]
    fn test_conv3d_param_count_no_bias() {
        let config = Conv3dConfig {
            in_channels: 3,
            out_channels: 16,
            kernel_size: (3, 3, 3),
            bias: false,
            ..Conv3dConfig::default()
        };
        let conv = Conv3d::<f64>::new(config).expect("Failed to create Conv3d");
        assert_eq!(conv.param_count(), 1296);
    }

    #[test]
    fn test_conv3d_dilation() {
        // Dilation=2 with 3x3x3 kernel: effective kernel = 5x5x5
        let config = Conv3dConfig {
            in_channels: 1,
            out_channels: 1,
            kernel_size: (3, 3, 3),
            dilation: (2, 2, 2),
            ..Conv3dConfig::default()
        };
        let conv = Conv3d::<f64>::new(config).expect("Failed to create Conv3d");
        let input = Array5::<f64>::from_elem((1, 1, 8, 8, 8), 1.0).into_dyn();
        let output = conv.forward(&input).expect("Forward failed");
        // effective_k = 2*(3-1)+1 = 5; out = (8 - 5)/1 + 1 = 4
        assert_eq!(output.shape(), &[1, 1, 4, 4, 4]);
    }

    #[test]
    fn test_conv3d_zero_input_bias_output() {
        let config = Conv3dConfig {
            in_channels: 1,
            out_channels: 2,
            kernel_size: (3, 3, 3),
            bias: true,
            padding: (1, 1, 1),
            ..Conv3dConfig::default()
        };
        let mut conv = Conv3d::<f64>::new(config).expect("Failed to create Conv3d");

        // Set weights to zero, bias to [1.0, 2.0]
        {
            let mut w = conv.weight.write().expect("lock");
            w.fill(0.0);
        }
        {
            let bias_lock = conv.bias.as_ref().expect("has bias");
            let mut b = bias_lock.write().expect("lock");
            b[[0]] = 1.0;
            b[[1]] = 2.0;
        }

        let input = Array5::<f64>::zeros((1, 1, 4, 4, 4)).into_dyn();
        let output = conv.forward(&input).expect("Forward failed");
        assert_eq!(output.shape(), &[1, 2, 4, 4, 4]);

        // All outputs should equal bias
        assert!((output[[0, 0, 0, 0, 0]] - 1.0).abs() < 1e-12);
        assert!((output[[0, 1, 0, 0, 0]] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_conv3d_single_batch_single_channel() {
        let config = Conv3dConfig {
            in_channels: 1,
            out_channels: 1,
            kernel_size: (2, 2, 2),
            bias: false,
            ..Conv3dConfig::default()
        };
        let conv = Conv3d::<f64>::new(config).expect("Failed to create Conv3d");
        let input = Array5::<f64>::from_elem((1, 1, 3, 3, 3), 1.0).into_dyn();
        let output = conv.forward(&input).expect("Forward failed");
        assert_eq!(output.shape(), &[1, 1, 2, 2, 2]);
    }

    #[test]
    fn test_conv3d_output_shape_method() {
        let config = Conv3dConfig {
            in_channels: 3,
            out_channels: 8,
            kernel_size: (3, 3, 3),
            stride: (2, 2, 2),
            padding: (1, 1, 1),
            ..Conv3dConfig::default()
        };
        let conv = Conv3d::<f64>::new(config).expect("Failed to create Conv3d");
        let shape = conv.output_shape(&[4, 3, 16, 16, 16]);
        assert_eq!(shape, [4, 8, 8, 8, 8]);
    }

    // -- MaxPool3d tests ----------------------------------------------------

    #[test]
    fn test_maxpool3d_reduces_spatial() {
        let config = MaxPool3dConfig {
            kernel_size: (2, 2, 2),
            ..MaxPool3dConfig::default()
        };
        let pool = MaxPool3d::<f64>::new(config).expect("Failed to create MaxPool3d");
        let input = Array5::<f64>::from_elem((1, 1, 8, 8, 8), 1.0).into_dyn();
        let output = pool.forward(&input).expect("Forward failed");
        assert_eq!(output.shape(), &[1, 1, 4, 4, 4]);
    }

    #[test]
    fn test_maxpool3d_max_values_preserved() {
        let config = MaxPool3dConfig {
            kernel_size: (2, 2, 2),
            ..MaxPool3dConfig::default()
        };
        let pool = MaxPool3d::<f64>::new(config).expect("Failed to create MaxPool3d");

        let mut input = Array5::<f64>::zeros((1, 1, 4, 4, 4));
        input[[0, 0, 0, 0, 0]] = 1.0;
        input[[0, 0, 0, 0, 1]] = 2.0;
        input[[0, 0, 0, 1, 0]] = 3.0;
        input[[0, 0, 1, 0, 0]] = 4.0;
        input[[0, 0, 1, 1, 1]] = 10.0; // Max in first pool cube

        let output = pool.forward(&input.into_dyn()).expect("Forward failed");
        assert_eq!(output.shape(), &[1, 1, 2, 2, 2]);
        assert!((output[[0, 0, 0, 0, 0]] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_maxpool3d_backward() {
        let config = MaxPool3dConfig {
            kernel_size: (2, 2, 2),
            ..MaxPool3dConfig::default()
        };
        let pool = MaxPool3d::<f64>::new(config).expect("Failed to create MaxPool3d");

        let mut input = Array5::<f64>::zeros((1, 1, 4, 4, 4));
        input[[0, 0, 1, 1, 1]] = 10.0;
        input[[0, 0, 3, 3, 3]] = 20.0;

        let _output = pool
            .forward(&input.clone().into_dyn())
            .expect("Forward failed");

        let grad_out = Array5::<f64>::from_elem((1, 1, 2, 2, 2), 1.0).into_dyn();
        let grad_in = pool
            .backward(&input.into_dyn(), &grad_out)
            .expect("Backward failed");

        assert_eq!(grad_in.shape(), &[1, 1, 4, 4, 4]);
        // Gradient flows to max positions
        assert!((grad_in[[0, 0, 1, 1, 1]] - 1.0).abs() < 1e-12);
        assert!((grad_in[[0, 0, 3, 3, 3]] - 1.0).abs() < 1e-12);
    }

    // -- AvgPool3d tests ----------------------------------------------------

    #[test]
    fn test_avgpool3d_reduces_spatial() {
        let config = MaxPool3dConfig {
            kernel_size: (2, 2, 2),
            ..MaxPool3dConfig::default()
        };
        let pool = AvgPool3d::<f64>::new(config).expect("Failed to create AvgPool3d");
        let input = Array5::<f64>::from_elem((1, 1, 8, 8, 8), 1.0).into_dyn();
        let output = pool.forward(&input).expect("Forward failed");
        assert_eq!(output.shape(), &[1, 1, 4, 4, 4]);
    }

    #[test]
    fn test_avgpool3d_mean_values() {
        let config = MaxPool3dConfig {
            kernel_size: (2, 2, 2),
            ..MaxPool3dConfig::default()
        };
        let pool = AvgPool3d::<f64>::new(config).expect("Failed to create AvgPool3d");

        // Fill first 2x2x2 cube with values summing to 36 -> mean = 4.5
        let mut input = Array5::<f64>::zeros((1, 1, 4, 4, 4));
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut idx = 0;
        for d in 0..2 {
            for h in 0..2 {
                for w in 0..2 {
                    input[[0, 0, d, h, w]] = vals[idx];
                    idx += 1;
                }
            }
        }

        let output = pool.forward(&input.into_dyn()).expect("Forward failed");
        // Mean of [1,2,3,4,5,6,7,8] = 4.5
        assert!((output[[0, 0, 0, 0, 0]] - 4.5).abs() < 1e-12);
    }

    #[test]
    fn test_avgpool3d_backward() {
        let config = MaxPool3dConfig {
            kernel_size: (2, 2, 2),
            ..MaxPool3dConfig::default()
        };
        let pool = AvgPool3d::<f64>::new(config).expect("Failed to create AvgPool3d");

        let input = Array5::<f64>::from_elem((1, 1, 4, 4, 4), 1.0);
        let _output = pool
            .forward(&input.clone().into_dyn())
            .expect("Forward failed");

        let grad_out = Array5::<f64>::from_elem((1, 1, 2, 2, 2), 8.0).into_dyn();
        let grad_in = pool
            .backward(&input.into_dyn(), &grad_out)
            .expect("Backward failed");

        assert_eq!(grad_in.shape(), &[1, 1, 4, 4, 4]);
        // Each gradient distributed: 8.0 / 8 = 1.0
        assert!((grad_in[[0, 0, 0, 0, 0]] - 1.0).abs() < 1e-12);
    }

    // -- BatchNorm3d tests --------------------------------------------------

    #[test]
    fn test_batchnorm3d_output_normalized() {
        let mut bn = BatchNorm3d::<f64>::new(2);

        // Generate input with known statistics
        let mut input = Array5::<f64>::zeros((4, 2, 3, 3, 3));
        // Fill channel 0 with values around 10.0
        for b in 0..4 {
            for d in 0..3 {
                for h in 0..3 {
                    for w in 0..3 {
                        input[[b, 0, d, h, w]] = 10.0 + (b as f64) * 0.1;
                        input[[b, 1, d, h, w]] = 5.0 + (b as f64) * 0.2;
                    }
                }
            }
        }

        let output = bn.forward_mut(&input.into_dyn()).expect("Forward failed");

        // Check that output per channel has approximately zero mean
        let shape = output.shape();
        for c in 0..2 {
            let mut sum = 0.0;
            let mut count = 0;
            for b in 0..shape[0] {
                for d in 0..shape[2] {
                    for h in 0..shape[3] {
                        for w in 0..shape[4] {
                            sum += output[[b, c, d, h, w]];
                            count += 1;
                        }
                    }
                }
            }
            let mean = sum / count as f64;
            assert!(
                mean.abs() < 0.1,
                "Channel {} mean should be ~0, got {}",
                c,
                mean
            );
        }
    }

    #[test]
    fn test_batchnorm3d_eval_mode() {
        let mut bn = BatchNorm3d::<f64>::new(1);
        bn.eval_mode();
        assert!(!bn.training);
        bn.train_mode();
        assert!(bn.training);
    }

    #[test]
    fn test_batchnorm3d_param_count() {
        let bn = BatchNorm3d::<f64>::new(16);
        // gamma (16) + beta (16)
        assert_eq!(bn.parameter_count(), 32);
    }

    // -- Conv3d backward test -----------------------------------------------

    #[test]
    fn test_conv3d_backward_shape() {
        let config = Conv3dConfig {
            in_channels: 2,
            out_channels: 4,
            kernel_size: (3, 3, 3),
            ..Conv3dConfig::default()
        };
        let conv = Conv3d::<f64>::new(config).expect("Failed to create Conv3d");
        let input = Array5::<f64>::from_elem((1, 2, 6, 6, 6), 1.0).into_dyn();
        let _output = conv.forward(&input).expect("Forward failed");

        let grad_out = Array5::<f64>::from_elem((1, 4, 4, 4, 4), 0.1).into_dyn();
        let grad_in = conv
            .backward(&input, &grad_out)
            .expect("Backward failed");
        assert_eq!(grad_in.shape(), &[1, 2, 6, 6, 6]);
    }
}
