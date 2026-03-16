//! Mamba architecture implementation
//!
//! This module implements the Mamba architecture as described in:
//! "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
//! by Albert Gu and Tri Dao (<https://arxiv.org/abs/2312.00752>)
//!
//! Mamba is a selective state space model (SSM) that achieves linear-time
//! sequence modeling with competitive performance to Transformers while
//! being significantly more efficient for long sequences.

use crate::activations::Activation;
use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Dropout, Layer, LayerNorm};
use scirs2_core::ndarray::{s, Array, Array1, Array2, Array3, IxDyn, ScalarOperand, Zip};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt, SeedableRng};
use scirs2_core::simd_ops::SimdUnifiedOps;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Configuration for the Mamba model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MambaConfig {
    /// Model dimension (d_model)
    pub d_model: usize,
    /// State dimension (n)
    pub d_state: usize,
    /// Convolution kernel size
    pub d_conv: usize,
    /// Expansion factor for the intermediate dimension
    pub expand: usize,
    /// Number of Mamba blocks
    pub n_layers: usize,
    /// Dropout probability
    pub dropout_prob: f64,
    /// Vocabulary size (for embedding)
    pub vocab_size: Option<usize>,
    /// Number of output classes (for classification)
    pub num_classes: Option<usize>,
    /// Rank for delta projection (dt_rank)
    pub dt_rank: Option<usize>,
    /// Whether to use bias in projections
    pub bias: bool,
    /// Initialization range for delta
    pub dt_min: f64,
    pub dt_max: f64,
}

impl Default for MambaConfig {
    fn default() -> Self {
        Self {
            d_model: 256,
            d_state: 16,
            d_conv: 4,
            expand: 2,
            n_layers: 4,
            dropout_prob: 0.1,
            vocab_size: None,
            num_classes: None,
            dt_rank: None, // Auto-computed as ceil(d_model / 16)
            bias: false,
            dt_min: 0.001,
            dt_max: 0.1,
        }
    }
}

impl MambaConfig {
    /// Create a new MambaConfig with the specified model dimension
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            ..Default::default()
        }
    }

    /// Set the state dimension
    pub fn with_d_state(mut self, d_state: usize) -> Self {
        self.d_state = d_state;
        self
    }

    /// Set the number of layers
    pub fn with_n_layers(mut self, n_layers: usize) -> Self {
        self.n_layers = n_layers;
        self
    }

    /// Set the expansion factor
    pub fn with_expand(mut self, expand: usize) -> Self {
        self.expand = expand;
        self
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, dropout_prob: f64) -> Self {
        self.dropout_prob = dropout_prob;
        self
    }

    /// Set vocabulary size
    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = Some(vocab_size);
        self
    }

    /// Set number of classes for classification
    pub fn with_num_classes(mut self, num_classes: usize) -> Self {
        self.num_classes = Some(num_classes);
        self
    }

    /// Get the inner dimension
    pub fn d_inner(&self) -> usize {
        self.d_model * self.expand
    }

    /// Get the dt_rank (auto-computed if not set)
    pub fn get_dt_rank(&self) -> usize {
        self.dt_rank.unwrap_or_else(|| self.d_model.div_ceil(16)) // ceil division
    }
}

/// Selective State Space Model (S6) - the core component of Mamba
///
/// Implements the discretized SSM:
/// h_t = A_bar * h_{t-1} + B_bar * x_t
/// y_t = C * h_t
///
/// Where A_bar and B_bar are computed from continuous-time parameters
/// using the zero-order hold (ZOH) discretization.
#[derive(Debug)]
pub struct SelectiveSSM<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> {
    /// State dimension
    d_state: usize,
    /// Inner dimension
    d_inner: usize,
    /// Continuous-time A matrix (diagonal, initialized to special values)
    a_log: Array2<F>,
    /// D parameter (skip connection)
    d: Array1<F>,
    /// Delta projection weights
    dt_proj: Dense<F>,
    /// Projection for B
    x_proj_b: Dense<F>,
    /// Projection for C
    x_proj_c: Dense<F>,
    /// Projection for delta
    x_proj_dt: Dense<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> SelectiveSSM<F> {
    /// Create a new SelectiveSSM
    pub fn new<R: Rng>(
        d_inner: usize,
        d_state: usize,
        dt_rank: usize,
        rng: &mut R,
    ) -> Result<Self> {
        // Initialize A with the S4D-Real initialization
        // A = -exp(uniformly spaced values from ln(1) to ln(state_dim))
        let mut a_log = Array2::<F>::zeros((d_inner, d_state));
        for i in 0..d_inner {
            for j in 0..d_state {
                // Log-spaced from 1 to d_state
                let val = (j as f64 + 1.0).ln();
                a_log[[i, j]] = F::from(val).expect("Failed to convert to float");
            }
        }

        // D initialization (skip connection, initialized to 1)
        let d = Array1::<F>::from_elem(d_inner, F::one());

        // Delta projection from dt_rank to d_inner
        let dt_proj = Dense::<F>::new(dt_rank, d_inner, Some("dt_proj"), rng)?;

        // Projections for B, C, and dt from input
        let x_proj_b = Dense::<F>::new(d_inner, d_state, Some("x_proj_b"), rng)?;
        let x_proj_c = Dense::<F>::new(d_inner, d_state, Some("x_proj_c"), rng)?;
        let x_proj_dt = Dense::<F>::new(d_inner, dt_rank, Some("x_proj_dt"), rng)?;

        Ok(Self {
            d_state,
            d_inner,
            a_log,
            d,
            dt_proj,
            x_proj_b,
            x_proj_c,
            x_proj_dt,
        })
    }

    /// Compute the SSM output using selective scan
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, d_inner]
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, d_inner]
    pub fn forward(&self, x: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        if x.ndim() != 3 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "SelectiveSSM expects 3D input, got {}D",
                x.ndim()
            )));
        }

        let shape = x.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let d_inner = shape[2];

        if d_inner != self.d_inner {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Input dimension {} doesn't match d_inner {}",
                d_inner, self.d_inner
            )));
        }

        // Compute A from a_log: A = -exp(a_log)
        let a_neg = self.a_log.mapv(|v| -v.exp());

        // Project input to get B, C, delta
        // Reshape for dense layer
        let x_2d = x
            .clone()
            .into_shape_with_order(IxDyn(&[batch_size * seq_len, d_inner]))
            .map_err(|e| NeuralError::InferenceError(format!("Reshape error: {}", e)))?;

        // Get B, C, dt projections
        let b_proj = self.x_proj_b.forward(&x_2d)?;
        let c_proj = self.x_proj_c.forward(&x_2d)?;
        let dt_proj_input = self.x_proj_dt.forward(&x_2d)?;
        let delta_proj = self.dt_proj.forward(&dt_proj_input)?;

        // Apply softplus to delta: delta = softplus(delta_proj)
        let delta = delta_proj.mapv(|v: F| {
            if v > F::from(20.0).expect("Failed to convert constant to float") {
                v
            } else {
                (F::one() + v.exp()).ln()
            }
        });

        // Reshape B, C, delta back to [batch, seq, ...]
        let b = b_proj
            .into_shape_with_order(IxDyn(&[batch_size, seq_len, self.d_state]))
            .map_err(|e| NeuralError::InferenceError(format!("B reshape error: {}", e)))?;

        let c = c_proj
            .into_shape_with_order(IxDyn(&[batch_size, seq_len, self.d_state]))
            .map_err(|e| NeuralError::InferenceError(format!("C reshape error: {}", e)))?;

        let delta_3d = delta
            .into_shape_with_order(IxDyn(&[batch_size, seq_len, self.d_inner]))
            .map_err(|e| NeuralError::InferenceError(format!("Delta reshape error: {}", e)))?;

        // Perform selective scan
        let mut output = Array::zeros(IxDyn(&[batch_size, seq_len, d_inner]));

        for batch_idx in 0..batch_size {
            // Initialize state: [d_inner, d_state]
            let mut h = Array2::<F>::zeros((d_inner, self.d_state));

            for t in 0..seq_len {
                // Get delta for this timestep: [d_inner]
                let dt = delta_3d.slice(s![batch_idx, t, ..]);

                // Get B and C for this timestep: [d_state]
                let b_t = b.slice(s![batch_idx, t, ..]);
                let c_t = c.slice(s![batch_idx, t, ..]);

                // Get input for this timestep: [d_inner]
                let x_t = x.slice(s![batch_idx, t, ..]);

                // Discretize A and B using zero-order hold
                // A_bar = exp(delta * A)
                // B_bar = (A_bar - I) * A^(-1) * B ≈ delta * B (simplified)

                // Update state for each dimension
                for i in 0..d_inner {
                    let dt_i = dt[i];

                    for j in 0..self.d_state {
                        // A_bar[i,j] = exp(dt[i] * A[i,j])
                        let a_bar = (dt_i * a_neg[[i, j]]).exp();
                        // B_bar[i,j] ≈ dt[i] * B[j] (simplified discretization)
                        let b_bar = dt_i * b_t[j];

                        // State update: h = A_bar * h + B_bar * x
                        h[[i, j]] = a_bar * h[[i, j]] + b_bar * x_t[i];
                    }
                }

                // Compute output: y = C * h + D * x
                for i in 0..d_inner {
                    let mut y_i = F::zero();
                    for j in 0..self.d_state {
                        y_i += c_t[j] * h[[i, j]];
                    }
                    // Add skip connection
                    output[[batch_idx, t, i]] = y_i + self.d[[i]] * x_t[i];
                }
            }
        }

        Ok(output)
    }
}

/// 1D Convolution layer for Mamba
#[derive(Debug)]
struct Conv1D<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> {
    /// Convolution weights [out_channels, kernel_size]
    weights: Array2<F>,
    /// Bias [out_channels]
    bias: Array1<F>,
    /// Kernel size
    kernel_size: usize,
    /// Number of channels
    channels: usize,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Conv1D<F> {
    fn new<R: Rng>(channels: usize, kernel_size: usize, rng: &mut R) -> Result<Self> {
        let std = (F::from(2.0).expect("Failed to convert constant to float")
            / F::from(channels * kernel_size).expect("Failed to convert to float"))
        .sqrt();

        let mut weights = Array2::<F>::zeros((channels, kernel_size));
        for w in weights.iter_mut() {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            *w = F::from(z).expect("Failed to convert to float") * std;
        }

        let bias = Array1::<F>::zeros(channels);

        Ok(Self {
            weights,
            bias,
            kernel_size,
            channels,
        })
    }

    fn forward(&self, x: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // x: [batch, seq_len, channels]
        let shape = x.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let channels = shape[2];

        if channels != self.channels {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Channel mismatch: {} vs {}",
                channels, self.channels
            )));
        }

        // Causal 1D convolution with padding
        let pad = self.kernel_size - 1;
        let mut output = Array::zeros(IxDyn(&[batch_size, seq_len, channels]));

        for b in 0..batch_size {
            for t in 0..seq_len {
                for c in 0..channels {
                    let mut sum = self.bias[c];
                    for k in 0..self.kernel_size {
                        let input_idx = t as isize + k as isize - pad as isize;
                        if input_idx >= 0 && (input_idx as usize) < seq_len {
                            sum += self.weights[[c, k]] * x[[b, input_idx as usize, c]];
                        }
                    }
                    output[[b, t, c]] = sum;
                }
            }
        }

        Ok(output)
    }
}

/// SiLU (Swish) activation function
#[derive(Debug, Clone, Copy)]
struct SiLU;

impl SiLU {
    fn forward<F: Float>(&self, x: F) -> F {
        x * (F::one() / (F::one() + (-x).exp()))
    }
}

/// A single Mamba block
#[derive(Debug)]
pub struct MambaBlock<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign>
where
    F: SimdUnifiedOps,
{
    /// Configuration
    d_model: usize,
    d_inner: usize,
    /// Input projection [d_model -> d_inner * 2]
    in_proj: Dense<F>,
    /// 1D convolution
    conv1d: Conv1D<F>,
    /// Selective SSM
    ssm: SelectiveSSM<F>,
    /// Output projection [d_inner -> d_model]
    out_proj: Dense<F>,
    /// Layer normalization
    norm: LayerNorm<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static>
    MambaBlock<F>
{
    /// Create a new MambaBlock
    pub fn new<R: Rng>(config: &MambaConfig, rng: &mut R) -> Result<Self> {
        let d_inner = config.d_inner();
        let dt_rank = config.get_dt_rank();

        // Input projection: d_model -> d_inner * 2 (for x and z branches)
        let in_proj = Dense::<F>::new(config.d_model, d_inner * 2, Some("in_proj"), rng)?;

        // 1D causal convolution
        let conv1d = Conv1D::new(d_inner, config.d_conv, rng)?;

        // Selective SSM
        let ssm = SelectiveSSM::new(d_inner, config.d_state, dt_rank, rng)?;

        // Output projection
        let out_proj = Dense::<F>::new(d_inner, config.d_model, Some("out_proj"), rng)?;

        // Layer norm
        let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
        let norm = LayerNorm::<F>::new(config.d_model, 1e-5, &mut rng)?;

        Ok(Self {
            d_model: config.d_model,
            d_inner,
            in_proj,
            conv1d,
            ssm,
            out_proj,
            norm,
        })
    }

    /// Forward pass
    pub fn forward(&self, x: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // x: [batch, seq_len, d_model]
        let residual = x.clone();

        // Layer norm
        let normed = self.norm.forward(x)?;

        let shape = normed.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];

        // Input projection: [batch, seq, d_model] -> [batch, seq, d_inner * 2]
        let x_2d = normed
            .clone()
            .into_shape_with_order(IxDyn(&[batch_size * seq_len, self.d_model]))
            .map_err(|e| NeuralError::InferenceError(format!("Reshape error: {}", e)))?;

        let proj = self.in_proj.forward(&x_2d)?;

        let proj_3d = proj
            .into_shape_with_order(IxDyn(&[batch_size, seq_len, self.d_inner * 2]))
            .map_err(|e| NeuralError::InferenceError(format!("Reshape error: {}", e)))?;

        // Split into x and z branches
        let x_branch = proj_3d
            .slice(s![.., .., ..self.d_inner])
            .to_owned()
            .into_dyn();
        let z_branch = proj_3d
            .slice(s![.., .., self.d_inner..])
            .to_owned()
            .into_dyn();

        // Apply convolution and SiLU to x branch
        let x_conv = self.conv1d.forward(&x_branch)?;

        let silu = SiLU;
        let x_silu = x_conv.mapv(|v| silu.forward(v));

        // Apply SSM
        let x_ssm = self.ssm.forward(&x_silu)?;

        // Gate with z branch (SiLU activation)
        let z_silu = z_branch.mapv(|v| silu.forward(v));

        // Element-wise multiplication
        let gated = &x_ssm * &z_silu;

        // Output projection
        let gated_2d = gated
            .into_shape_with_order(IxDyn(&[batch_size * seq_len, self.d_inner]))
            .map_err(|e| NeuralError::InferenceError(format!("Reshape error: {}", e)))?;

        let output = self.out_proj.forward(&gated_2d)?;

        let output_3d = output
            .into_shape_with_order(IxDyn(&[batch_size, seq_len, self.d_model]))
            .map_err(|e| NeuralError::InferenceError(format!("Reshape error: {}", e)))?;

        // Residual connection
        Ok(&residual + &output_3d)
    }
}

/// The Mamba model
///
/// A state-space model that achieves linear-time sequence modeling
/// with selective state spaces.
///
/// # Architecture
///
/// - Optional embedding layer (for language modeling)
/// - Stack of Mamba blocks
/// - Final layer norm
/// - Optional classification head
///
/// # Examples
///
/// ```rust
/// use scirs2_neural::models::architectures::{Mamba, MambaConfig};
/// use scirs2_neural::layers::Layer;
/// use scirs2_core::ndarray::Array3;
/// use scirs2_core::random::rng;
///
/// let mut rng = rng();
/// let config = MambaConfig::new(256)
///     .with_n_layers(4)
///     .with_d_state(16);
///
/// let mamba = Mamba::<f64>::new(config, &mut rng).expect("Operation failed");
///
/// // Input: [batch, seq_len, d_model]
/// let input = Array3::<f64>::from_elem((2, 32, 256), 0.1).into_dyn();
/// let output = mamba.forward(&input).expect("Operation failed");
/// ```
#[derive(Debug)]
pub struct Mamba<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign>
where
    F: SimdUnifiedOps,
{
    /// Configuration
    config: MambaConfig,
    /// Stack of Mamba blocks
    blocks: Vec<MambaBlock<F>>,
    /// Final layer normalization
    final_norm: LayerNorm<F>,
    /// Optional classification head
    classifier: Option<Dense<F>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + SimdUnifiedOps + NumAssign + 'static>
    Mamba<F>
{
    /// Create a new Mamba model
    pub fn new<R: Rng>(config: MambaConfig, rng: &mut R) -> Result<Self> {
        // Create Mamba blocks
        let mut blocks = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            blocks.push(MambaBlock::new(&config, rng)?);
        }

        // Final layer norm
        let mut rng_final = scirs2_core::random::rngs::SmallRng::from_seed([43; 32]);
        let final_norm = LayerNorm::<F>::new(config.d_model, 1e-5, &mut rng_final)?;

        // Optional classifier
        let classifier = if let Some(num_classes) = config.num_classes {
            Some(Dense::<F>::new(
                config.d_model,
                num_classes,
                Some("classifier"),
                rng,
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            blocks,
            final_norm,
            classifier,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &MambaConfig {
        &self.config
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }
}

impl<F> Layer<F> for Mamba<F>
where
    F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static + SimdUnifiedOps,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Input: [batch, seq_len, d_model]
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Mamba expects 3D input [batch, seq_len, d_model], got {}D",
                input.ndim()
            )));
        }

        let shape = input.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];

        if d_model != self.config.d_model {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Input dimension {} doesn't match config d_model {}",
                d_model, self.config.d_model
            )));
        }

        // Pass through Mamba blocks
        let mut hidden = input.clone();
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }

        // Final layer norm
        let normed = self.final_norm.forward(&hidden)?;

        // If classifier, apply it (use last token or mean pooling)
        if let Some(ref classifier) = self.classifier {
            // Mean pooling over sequence
            let mut pooled = Array::zeros(IxDyn(&[batch_size, self.config.d_model]));
            let seq_len_f = F::from(seq_len).expect("Failed to convert to float");

            for b in 0..batch_size {
                for d in 0..self.config.d_model {
                    let mut sum = F::zero();
                    for t in 0..seq_len {
                        sum += normed[[b, t, d]];
                    }
                    pooled[[b, d]] = sum / seq_len_f;
                }
            }

            classifier.forward(&pooled)
        } else {
            Ok(normed)
        }
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Err(NeuralError::NotImplemented(
            "Mamba backward pass not yet implemented".to_string(),
        ))
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Ok(())
    }
}

/// Simplified State Space Model (S4) layer
///
/// This implements a basic structured state space model without
/// the selective mechanism. Useful for comparison and simpler use cases.
#[derive(Debug)]
pub struct S4Layer<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign> {
    /// Dimension
    d_model: usize,
    /// State dimension
    d_state: usize,
    /// A matrix (HiPPO-initialized)
    a: Array2<F>,
    /// B matrix
    b: Array2<F>,
    /// C matrix
    c: Array2<F>,
    /// D matrix (skip connection)
    d: Array1<F>,
    /// Delta (step size)
    delta: F,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> S4Layer<F> {
    /// Create a new S4 layer with HiPPO initialization
    pub fn new<R: Rng>(d_model: usize, d_state: usize, rng: &mut R) -> Result<Self> {
        // HiPPO matrix initialization for A
        // A[i,j] = -sqrt((2i+1)(2j+1)) if i > j else -(i+1) if i == j else 0
        let mut a = Array2::<F>::zeros((d_state, d_state));
        for i in 0..d_state {
            for j in 0..d_state {
                let val = if i > j {
                    -((2.0 * i as f64 + 1.0) * (2.0 * j as f64 + 1.0)).sqrt()
                } else if i == j {
                    -(i as f64 + 1.0)
                } else {
                    0.0
                };
                a[[i, j]] = F::from(val).expect("Failed to convert to float");
            }
        }

        // B initialization
        let mut b = Array2::<F>::zeros((d_state, d_model));
        for i in 0..d_state {
            let val = (2.0 * i as f64 + 1.0).sqrt();
            for j in 0..d_model {
                let u: f64 = rng.random();
                b[[i, j]] = F::from(val * (u - 0.5) * 0.1).expect("Operation failed");
            }
        }

        // C initialization (learnable output projection)
        let mut c = Array2::<F>::zeros((d_model, d_state));
        let std = (2.0 / (d_model + d_state) as f64).sqrt();
        for i in 0..d_model {
            for j in 0..d_state {
                let u1: f64 = rng.random();
                let u2: f64 = rng.random();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                c[[i, j]] = F::from(z * std).expect("Failed to convert to float");
            }
        }

        // D (skip connection)
        let d = Array1::<F>::from_elem(d_model, F::one());

        // Default step size
        let delta = F::from(0.001).expect("Failed to convert constant to float");

        Ok(Self {
            d_model,
            d_state,
            a,
            b,
            c,
            d,
            delta,
        })
    }

    /// Forward pass using convolution mode (parallel over sequence)
    pub fn forward(&self, x: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // x: [batch, seq_len, d_model]
        if x.ndim() != 3 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "S4Layer expects 3D input, got {}D",
                x.ndim()
            )));
        }

        let shape = x.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let d_model = shape[2];

        if d_model != self.d_model {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Input dimension {} doesn't match d_model {}",
                d_model, self.d_model
            )));
        }

        // Discretize using ZOH
        // A_bar = exp(delta * A)
        // For simplicity, use first-order approximation: A_bar ≈ I + delta * A
        let mut a_bar = Array2::<F>::eye(self.d_state);
        for i in 0..self.d_state {
            for j in 0..self.d_state {
                a_bar[[i, j]] += self.delta * self.a[[i, j]];
            }
        }

        // B_bar ≈ delta * B
        let b_bar = &self.b * self.delta;

        // Run SSM
        let mut output = Array::zeros(IxDyn(&[batch_size, seq_len, d_model]));

        for b in 0..batch_size {
            // State: [d_state]
            let mut state = Array1::<F>::zeros(self.d_state);

            for t in 0..seq_len {
                // Get input: [d_model]
                let x_t: Array1<F> = x
                    .slice(s![b, t, ..])
                    .to_owned()
                    .into_shape_with_order(d_model)
                    .map_err(|_| {
                        NeuralError::InferenceError("Failed to reshape input".to_string())
                    })?;

                // State update: state = A_bar @ state + B_bar @ x_t
                let new_state = a_bar.dot(&state) + b_bar.dot(&x_t);
                state = new_state;

                // Output: y_t = C @ state + D * x_t
                let y_t = self.c.dot(&state) + &self.d * &x_t;

                for d in 0..d_model {
                    output[[b, t, d]] = y_t[d];
                }
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_mamba_config() {
        let config = MambaConfig::new(256)
            .with_n_layers(4)
            .with_d_state(16)
            .with_expand(2);

        assert_eq!(config.d_model, 256);
        assert_eq!(config.n_layers, 4);
        assert_eq!(config.d_state, 16);
        assert_eq!(config.d_inner(), 512);
    }

    #[test]
    fn test_mamba_creation() {
        let mut rng = scirs2_core::random::rng();
        let config = MambaConfig::new(64).with_n_layers(2).with_d_state(8);

        let mamba = Mamba::<f64>::new(config, &mut rng);
        assert!(mamba.is_ok());
    }

    #[test]
    fn test_mamba_forward() {
        let mut rng = scirs2_core::random::rng();
        let config = MambaConfig::new(32)
            .with_n_layers(2)
            .with_d_state(8)
            .with_expand(2);

        let mamba = Mamba::<f64>::new(config, &mut rng).expect("Operation failed");

        // Input: [batch=2, seq_len=8, d_model=32]
        let input = Array3::<f64>::from_elem((2, 8, 32), 0.1).into_dyn();
        let output = mamba.forward(&input);

        assert!(output.is_ok());
        let output = output.expect("Operation failed");
        assert_eq!(output.shape(), &[2, 8, 32]);
    }

    #[test]
    fn test_mamba_with_classifier() {
        let mut rng = scirs2_core::random::rng();
        let config = MambaConfig::new(32)
            .with_n_layers(2)
            .with_d_state(8)
            .with_num_classes(10);

        let mamba = Mamba::<f64>::new(config, &mut rng).expect("Operation failed");

        let input = Array3::<f64>::from_elem((2, 8, 32), 0.1).into_dyn();
        let output = mamba.forward(&input);

        assert!(output.is_ok());
        let output = output.expect("Operation failed");
        // With classifier, output should be [batch, num_classes]
        assert_eq!(output.shape(), &[2, 10]);
    }

    #[test]
    fn test_selective_ssm() {
        let mut rng = scirs2_core::random::rng();
        let d_inner = 16;
        let d_state = 4;
        let dt_rank = 2;

        let ssm = SelectiveSSM::<f64>::new(d_inner, d_state, dt_rank, &mut rng)
            .expect("Operation failed");

        let input = Array3::<f64>::from_elem((2, 4, d_inner), 0.1).into_dyn();
        let output = ssm.forward(&input);

        assert!(output.is_ok());
        assert_eq!(output.expect("Operation failed").shape(), &[2, 4, d_inner]);
    }

    #[test]
    fn test_s4_layer() {
        let mut rng = scirs2_core::random::rng();
        let d_model = 16;
        let d_state = 8;

        let s4 = S4Layer::<f64>::new(d_model, d_state, &mut rng).expect("Operation failed");

        let input = Array3::<f64>::from_elem((2, 8, d_model), 0.1).into_dyn();
        let output = s4.forward(&input);

        assert!(output.is_ok());
        assert_eq!(output.expect("Operation failed").shape(), &[2, 8, d_model]);
    }

    #[test]
    fn test_mamba_block() {
        let mut rng = scirs2_core::random::rng();
        let config = MambaConfig::new(32).with_d_state(8);

        let block = MambaBlock::<f64>::new(&config, &mut rng).expect("Operation failed");

        let input = Array3::<f64>::from_elem((2, 4, 32), 0.1).into_dyn();
        let output = block.forward(&input);

        assert!(output.is_ok());
        assert_eq!(output.expect("Operation failed").shape(), &[2, 4, 32]);
    }

    #[test]
    fn test_mamba_numerical_stability() {
        let mut rng = scirs2_core::random::rng();
        let config = MambaConfig::new(16).with_n_layers(1).with_d_state(4);

        let mamba = Mamba::<f64>::new(config, &mut rng).expect("Operation failed");

        // Test with varying input values
        let mut input = Array3::<f64>::zeros((1, 8, 16));
        for i in 0..8 {
            for j in 0..16 {
                input[[0, i, j]] = (i as f64 - 4.0) * 0.1 + j as f64 * 0.01;
            }
        }

        let output = mamba.forward(&input.into_dyn());
        assert!(output.is_ok());

        // Check all values are finite
        for val in output.expect("Operation failed").iter() {
            assert!(val.is_finite(), "Output contains non-finite values");
        }
    }

    #[test]
    fn test_conv1d() {
        let mut rng = scirs2_core::random::rng();
        let conv = Conv1D::<f64>::new(8, 3, &mut rng).expect("Operation failed");

        let input = Array3::<f64>::from_elem((2, 4, 8), 0.1).into_dyn();
        let output = conv.forward(&input);

        assert!(output.is_ok());
        assert_eq!(output.expect("Operation failed").shape(), &[2, 4, 8]);
    }
}
