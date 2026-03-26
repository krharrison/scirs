//! LoRA (Low-Rank Adaptation) layer implementation
//!
//! LoRA adds low-rank decomposition matrices alongside frozen pre-trained weights.
//! During training, only the low-rank matrices A and B are updated, dramatically
//! reducing the number of trainable parameters.
//!
//! Forward pass: y = Wx + (alpha/rank) * B * A * x
//!
//! Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Distribution, Normal, Uniform};
use std::fmt::Debug;
use std::sync::RwLock;

/// Configuration for a LoRA layer
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Input dimension of the base linear layer
    pub input_dim: usize,
    /// Output dimension of the base linear layer
    pub output_dim: usize,
    /// Rank of the low-rank decomposition (typically 1-64)
    pub rank: usize,
    /// Scaling factor alpha (effective scaling = alpha / rank)
    pub alpha: f64,
    /// Dropout rate applied to the LoRA path (0.0 = no dropout)
    pub dropout: f64,
    /// Whether to include bias in the base layer
    pub use_bias: bool,
    /// Whether the base weights are frozen (not updated during training)
    pub freeze_base: bool,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            output_dim: 64,
            rank: 4,
            alpha: 1.0,
            dropout: 0.0,
            use_bias: true,
            freeze_base: true,
        }
    }
}

/// LoRA (Low-Rank Adaptation) layer
///
/// Wraps a linear/dense layer with trainable low-rank A and B matrices.
/// The base weight matrix W is frozen; only A and B are trained.
///
/// Output: y = Wx + b + (alpha/rank) * B * A * x
pub struct LoraLayer<F: Float + Debug + Send + Sync + NumAssign> {
    config: LoraConfig,
    /// Base weight matrix W: [input_dim, output_dim]
    base_weights: Array<F, IxDyn>,
    /// Bias vector: [output_dim]
    bias: Option<Array<F, IxDyn>>,
    /// LoRA down-projection A: [input_dim, rank]
    lora_a: Array<F, IxDyn>,
    /// LoRA up-projection B: [rank, output_dim]
    lora_b: Array<F, IxDyn>,
    /// Whether LoRA weights have been merged into base weights
    merged: bool,
    /// Scaling factor: alpha / rank
    scaling: F,
    /// Training mode flag
    training: bool,
    /// Cached input for backward pass
    cached_input: RwLock<Option<Array<F, IxDyn>>>,
    /// Gradients for lora_a
    grad_lora_a: RwLock<Array<F, IxDyn>>,
    /// Gradients for lora_b
    grad_lora_b: RwLock<Array<F, IxDyn>>,
    /// Gradients for base weights (only used if not frozen)
    grad_base_weights: RwLock<Array<F, IxDyn>>,
    /// Gradients for bias
    grad_bias: RwLock<Option<Array<F, IxDyn>>>,
    /// Simple dropout mask seed counter
    dropout_counter: RwLock<u64>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> std::fmt::Debug
    for LoraLayer<F>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoraLayer")
            .field("input_dim", &self.config.input_dim)
            .field("output_dim", &self.config.output_dim)
            .field("rank", &self.config.rank)
            .field("alpha", &self.config.alpha)
            .field("merged", &self.merged)
            .field("freeze_base", &self.config.freeze_base)
            .finish()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> LoraLayer<F> {
    /// Create a new LoRA layer with the given configuration
    ///
    /// # Arguments
    /// * `config` - LoRA configuration
    /// * `rng` - Random number generator for weight initialization
    pub fn new<R: scirs2_core::random::Rng>(config: LoraConfig, rng: &mut R) -> Result<Self> {
        if config.rank == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "LoRA rank must be > 0".to_string(),
            ));
        }
        if config.input_dim == 0 || config.output_dim == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "Input and output dimensions must be > 0".to_string(),
            ));
        }
        if config.dropout < 0.0 || config.dropout >= 1.0 {
            return Err(NeuralError::InvalidArchitecture(
                "Dropout rate must be in [0.0, 1.0)".to_string(),
            ));
        }

        let input_dim = config.input_dim;
        let output_dim = config.output_dim;
        let rank = config.rank;

        // Initialize base weights with Xavier/Glorot
        let scale = F::from(1.0 / f64::sqrt(input_dim as f64)).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert scale factor".to_string())
        })?;
        let uniform = Uniform::new(-1.0, 1.0).map_err(|e| {
            NeuralError::InvalidArchitecture(format!("Failed to create uniform distribution: {e}"))
        })?;

        let base_weights_vec: Vec<F> = (0..(input_dim * output_dim))
            .map(|_| F::from(uniform.sample(rng)).unwrap_or(F::zero()) * scale)
            .collect();
        let base_weights = Array::from_shape_vec(IxDyn(&[input_dim, output_dim]), base_weights_vec)
            .map_err(|e| {
                NeuralError::InvalidArchitecture(format!(
                    "Failed to create base weights array: {e}"
                ))
            })?;

        // Initialize bias
        let bias = if config.use_bias {
            Some(Array::zeros(IxDyn(&[output_dim])))
        } else {
            None
        };

        // Initialize LoRA A with Kaiming/He (normal, std = 1/sqrt(input_dim))
        let normal = Normal::new(0.0, 1.0 / f64::sqrt(input_dim as f64)).map_err(|e| {
            NeuralError::InvalidArchitecture(format!("Failed to create normal distribution: {e}"))
        })?;
        let lora_a_vec: Vec<F> = (0..(input_dim * rank))
            .map(|_| F::from(normal.sample(rng)).unwrap_or(F::zero()))
            .collect();
        let lora_a = Array::from_shape_vec(IxDyn(&[input_dim, rank]), lora_a_vec).map_err(|e| {
            NeuralError::InvalidArchitecture(format!("Failed to create lora_a array: {e}"))
        })?;

        // Initialize LoRA B with zeros (so initial LoRA contribution is zero)
        let lora_b = Array::zeros(IxDyn(&[rank, output_dim]));

        let scaling = F::from(config.alpha / config.rank as f64).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert scaling factor".to_string())
        })?;

        let grad_lora_a = RwLock::new(Array::zeros(IxDyn(&[input_dim, rank])));
        let grad_lora_b = RwLock::new(Array::zeros(IxDyn(&[rank, output_dim])));
        let grad_base_weights = RwLock::new(Array::zeros(IxDyn(&[input_dim, output_dim])));
        let grad_bias = RwLock::new(if config.use_bias {
            Some(Array::zeros(IxDyn(&[output_dim])))
        } else {
            None
        });

        Ok(Self {
            config,
            base_weights,
            bias,
            lora_a,
            lora_b,
            merged: false,
            scaling,
            training: true,
            cached_input: RwLock::new(None),
            grad_lora_a,
            grad_lora_b,
            grad_base_weights,
            grad_bias,
            dropout_counter: RwLock::new(0),
        })
    }

    /// Create a LoRA layer from an existing base weight matrix
    ///
    /// Useful when adapting a pre-trained model: pass in the pre-trained weights,
    /// and LoRA will add low-rank adaptation on top.
    pub fn from_pretrained<R: scirs2_core::random::Rng>(
        base_weights: Array<F, IxDyn>,
        bias: Option<Array<F, IxDyn>>,
        rank: usize,
        alpha: f64,
        rng: &mut R,
    ) -> Result<Self> {
        if base_weights.ndim() != 2 {
            return Err(NeuralError::InvalidArchitecture(
                "Base weights must be 2-dimensional".to_string(),
            ));
        }
        let input_dim = base_weights.shape()[0];
        let output_dim = base_weights.shape()[1];

        let config = LoraConfig {
            input_dim,
            output_dim,
            rank,
            alpha,
            dropout: 0.0,
            use_bias: bias.is_some(),
            freeze_base: true,
        };

        let normal = Normal::new(0.0, 1.0 / f64::sqrt(input_dim as f64)).map_err(|e| {
            NeuralError::InvalidArchitecture(format!("Failed to create normal distribution: {e}"))
        })?;
        let lora_a_vec: Vec<F> = (0..(input_dim * rank))
            .map(|_| F::from(normal.sample(rng)).unwrap_or(F::zero()))
            .collect();
        let lora_a = Array::from_shape_vec(IxDyn(&[input_dim, rank]), lora_a_vec).map_err(|e| {
            NeuralError::InvalidArchitecture(format!("Failed to create lora_a array: {e}"))
        })?;

        let lora_b = Array::zeros(IxDyn(&[rank, output_dim]));

        let scaling = F::from(alpha / rank as f64).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert scaling factor".to_string())
        })?;

        let grad_lora_a = RwLock::new(Array::zeros(IxDyn(&[input_dim, rank])));
        let grad_lora_b = RwLock::new(Array::zeros(IxDyn(&[rank, output_dim])));
        let grad_base_weights = RwLock::new(Array::zeros(IxDyn(&[input_dim, output_dim])));
        let grad_bias = RwLock::new(if config.use_bias {
            Some(Array::zeros(IxDyn(&[output_dim])))
        } else {
            None
        });

        Ok(Self {
            config,
            base_weights,
            bias,
            lora_a,
            lora_b,
            merged: false,
            scaling,
            training: true,
            cached_input: RwLock::new(None),
            grad_lora_a,
            grad_lora_b,
            grad_base_weights,
            grad_bias,
            dropout_counter: RwLock::new(0),
        })
    }

    /// Merge LoRA weights into the base weights
    ///
    /// After merging, the forward pass uses only the combined weight matrix,
    /// which is faster for inference. Call `unmerge()` to separate them again.
    pub fn merge(&mut self) -> Result<()> {
        if self.merged {
            return Err(NeuralError::InvalidState(
                "LoRA weights are already merged".to_string(),
            ));
        }

        // Compute B * A contribution: lora_a [input_dim, rank] @ lora_b [rank, output_dim]
        // = [input_dim, output_dim]
        let lora_weight = self.matmul_2d(&self.lora_a, &self.lora_b)?;

        // base_weights += scaling * lora_weight
        let input_dim = self.config.input_dim;
        let output_dim = self.config.output_dim;
        for i in 0..input_dim {
            for j in 0..output_dim {
                self.base_weights[[i, j]] += self.scaling * lora_weight[[i, j]];
            }
        }

        self.merged = true;
        Ok(())
    }

    /// Unmerge LoRA weights from the base weights
    ///
    /// Separates the LoRA contribution from the base weights, restoring
    /// the original base weights. This is useful if you need to switch
    /// between different LoRA adaptations.
    pub fn unmerge(&mut self) -> Result<()> {
        if !self.merged {
            return Err(NeuralError::InvalidState(
                "LoRA weights are not merged".to_string(),
            ));
        }

        let lora_weight = self.matmul_2d(&self.lora_a, &self.lora_b)?;

        let input_dim = self.config.input_dim;
        let output_dim = self.config.output_dim;
        for i in 0..input_dim {
            for j in 0..output_dim {
                self.base_weights[[i, j]] -= self.scaling * lora_weight[[i, j]];
            }
        }

        self.merged = false;
        Ok(())
    }

    /// Check if LoRA weights are merged into base weights
    pub fn is_merged(&self) -> bool {
        self.merged
    }

    /// Get the effective scaling factor (alpha / rank)
    pub fn effective_scaling(&self) -> F {
        self.scaling
    }

    /// Get the LoRA rank
    pub fn rank(&self) -> usize {
        self.config.rank
    }

    /// Get the number of trainable parameters (only LoRA A + B if base is frozen)
    pub fn trainable_parameter_count(&self) -> usize {
        let lora_params =
            self.config.input_dim * self.config.rank + self.config.rank * self.config.output_dim;
        if self.config.freeze_base {
            lora_params
        } else {
            lora_params
                + self.config.input_dim * self.config.output_dim
                + if self.config.use_bias {
                    self.config.output_dim
                } else {
                    0
                }
        }
    }

    /// Get the compression ratio (trainable params / total params)
    pub fn compression_ratio(&self) -> f64 {
        let total = self.config.input_dim * self.config.output_dim
            + if self.config.use_bias {
                self.config.output_dim
            } else {
                0
            };
        if total == 0 {
            return 0.0;
        }
        self.trainable_parameter_count() as f64 / total as f64
    }

    /// Get a reference to the LoRA A matrix
    pub fn lora_a(&self) -> &Array<F, IxDyn> {
        &self.lora_a
    }

    /// Get a reference to the LoRA B matrix
    pub fn lora_b(&self) -> &Array<F, IxDyn> {
        &self.lora_b
    }

    /// Get a reference to the base weight matrix
    pub fn base_weights(&self) -> &Array<F, IxDyn> {
        &self.base_weights
    }

    /// Set the LoRA A matrix
    pub fn set_lora_a(&mut self, a: Array<F, IxDyn>) -> Result<()> {
        if a.shape() != [self.config.input_dim, self.config.rank] {
            return Err(NeuralError::ShapeMismatch(format!(
                "Expected lora_a shape [{}, {}], got {:?}",
                self.config.input_dim,
                self.config.rank,
                a.shape()
            )));
        }
        self.lora_a = a;
        Ok(())
    }

    /// Set the LoRA B matrix
    pub fn set_lora_b(&mut self, b: Array<F, IxDyn>) -> Result<()> {
        if b.shape() != [self.config.rank, self.config.output_dim] {
            return Err(NeuralError::ShapeMismatch(format!(
                "Expected lora_b shape [{}, {}], got {:?}",
                self.config.rank,
                self.config.output_dim,
                b.shape()
            )));
        }
        self.lora_b = b;
        Ok(())
    }

    /// Simple 2D matrix multiplication: C = A @ B
    /// A: [m, k], B: [k, n] -> C: [m, n]
    fn matmul_2d(&self, a: &Array<F, IxDyn>, b: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];
        if b.shape()[0] != k {
            return Err(NeuralError::ShapeMismatch(format!(
                "Matrix multiply shape mismatch: [{}, {}] @ [{}, {}]",
                m,
                k,
                b.shape()[0],
                n
            )));
        }
        let mut c = Array::zeros(IxDyn(&[m, n]));
        for i in 0..m {
            for j in 0..n {
                let mut sum = F::zero();
                for p in 0..k {
                    sum += a[[i, p]] * b[[p, j]];
                }
                c[[i, j]] = sum;
            }
        }
        Ok(c)
    }

    /// Batched forward computation: handles [batch, input_dim] inputs
    fn compute_forward_batched(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let batch_size = input.shape()[0];
        let input_dim = self.config.input_dim;
        let output_dim = self.config.output_dim;

        // y = W * x
        let mut output = Array::zeros(IxDyn(&[batch_size, output_dim]));
        for b in 0..batch_size {
            for j in 0..output_dim {
                let mut sum = F::zero();
                for i in 0..input_dim {
                    sum += input[[b, i]] * self.base_weights[[i, j]];
                }
                output[[b, j]] = sum;
            }
        }

        // Add bias
        if let Some(ref bias) = self.bias {
            for b in 0..batch_size {
                for j in 0..output_dim {
                    output[[b, j]] += bias[j];
                }
            }
        }

        // Add LoRA contribution if not merged
        if !self.merged {
            let rank = self.config.rank;

            // Compute A * x: [batch, input_dim] @ [input_dim, rank] = [batch, rank]
            let mut ax = Array::zeros(IxDyn(&[batch_size, rank]));
            for b in 0..batch_size {
                for r in 0..rank {
                    let mut sum = F::zero();
                    for i in 0..input_dim {
                        sum += input[[b, i]] * self.lora_a[[i, r]];
                    }
                    ax[[b, r]] = sum;
                }
            }

            // Apply dropout during training
            if self.training && self.config.dropout > 0.0 {
                let dropout_keep = F::from(1.0 - self.config.dropout).unwrap_or(F::one());
                let dropout_scale = F::one() / dropout_keep;
                let counter = {
                    let mut c = self
                        .dropout_counter
                        .write()
                        .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
                    *c = c.wrapping_add(1);
                    *c
                };
                // Simple deterministic dropout based on position and counter
                for b in 0..batch_size {
                    for r in 0..rank {
                        let hash = ((b as u64).wrapping_mul(2654435761))
                            ^ ((r as u64).wrapping_mul(40503))
                            ^ counter;
                        let val = (hash % 1000) as f64 / 1000.0;
                        if val < self.config.dropout {
                            ax[[b, r]] = F::zero();
                        } else {
                            ax[[b, r]] *= dropout_scale;
                        }
                    }
                }
            }

            // Compute B * (A * x): [batch, rank] @ [rank, output_dim] = [batch, output_dim]
            for b in 0..batch_size {
                for j in 0..output_dim {
                    let mut sum = F::zero();
                    for r in 0..rank {
                        sum += ax[[b, r]] * self.lora_b[[r, j]];
                    }
                    output[[b, j]] += self.scaling * sum;
                }
            }
        }

        Ok(output)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for LoraLayer<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward
        {
            let mut cache = self
                .cached_input
                .write()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            *cache = Some(input.clone());
        }

        // Ensure input is 2D
        let input_2d = if input.ndim() == 1 {
            input
                .clone()
                .into_shape_with_order(IxDyn(&[1, self.config.input_dim]))
                .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape input: {e}")))?
        } else {
            input.clone()
        };

        if input_2d.shape()[1] != self.config.input_dim {
            return Err(NeuralError::InvalidArgument(format!(
                "Input dimension mismatch: expected {}, got {}",
                self.config.input_dim,
                input_2d.shape()[1]
            )));
        }

        self.compute_forward_batched(&input_2d)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let cached_input = {
            let cache = self
                .cached_input
                .read()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            cache.clone().ok_or_else(|| {
                NeuralError::InferenceError("No cached input for backward pass".to_string())
            })?
        };

        let input_2d = if cached_input.ndim() == 1 {
            cached_input
                .into_shape_with_order(IxDyn(&[1, self.config.input_dim]))
                .map_err(|e| {
                    NeuralError::InferenceError(format!("Failed to reshape cached input: {e}"))
                })?
        } else {
            cached_input
        };

        let batch_size = input_2d.shape()[0];
        let input_dim = self.config.input_dim;
        let output_dim = self.config.output_dim;
        let rank = self.config.rank;

        // Gradient w.r.t. base weights: dW = input^T @ grad_output
        if !self.config.freeze_base {
            let mut dw = self
                .grad_base_weights
                .write()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            for i in 0..input_dim {
                for j in 0..output_dim {
                    let mut sum = F::zero();
                    for b in 0..batch_size {
                        sum += input_2d[[b, i]] * grad_output[[b, j]];
                    }
                    dw[[i, j]] = sum;
                }
            }
        }

        // Gradient w.r.t. bias
        if self.config.use_bias {
            let mut db = self
                .grad_bias
                .write()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            if let Some(ref mut db_arr) = *db {
                for j in 0..output_dim {
                    let mut sum = F::zero();
                    for b in 0..batch_size {
                        sum += grad_output[[b, j]];
                    }
                    db_arr[j] = sum;
                }
            }
        }

        // LoRA gradients
        // Forward LoRA path: output_lora = scaling * (input @ A) @ B
        // d(loss)/dB = scaling * (input @ A)^T @ grad_output
        //            = scaling * A^T @ input^T @ grad_output
        // d(loss)/dA = scaling * input^T @ grad_output @ B^T

        // Compute input @ A: [batch, rank]
        let mut input_a = Array::zeros(IxDyn(&[batch_size, rank]));
        for b in 0..batch_size {
            for r in 0..rank {
                let mut sum = F::zero();
                for i in 0..input_dim {
                    sum += input_2d[[b, i]] * self.lora_a[[i, r]];
                }
                input_a[[b, r]] = sum;
            }
        }

        // dB = scaling * (input @ A)^T @ grad_output: [rank, output_dim]
        {
            let mut dlb = self
                .grad_lora_b
                .write()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            for r in 0..rank {
                for j in 0..output_dim {
                    let mut sum = F::zero();
                    for b in 0..batch_size {
                        sum += input_a[[b, r]] * grad_output[[b, j]];
                    }
                    dlb[[r, j]] = self.scaling * sum;
                }
            }
        }

        // Compute grad_output @ B^T: [batch, rank]
        let mut grad_b_t = Array::zeros(IxDyn(&[batch_size, rank]));
        for b in 0..batch_size {
            for r in 0..rank {
                let mut sum = F::zero();
                for j in 0..output_dim {
                    sum += grad_output[[b, j]] * self.lora_b[[r, j]];
                }
                grad_b_t[[b, r]] = sum;
            }
        }

        // dA = scaling * input^T @ (grad_output @ B^T): [input_dim, rank]
        {
            let mut dla = self
                .grad_lora_a
                .write()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            for i in 0..input_dim {
                for r in 0..rank {
                    let mut sum = F::zero();
                    for b in 0..batch_size {
                        sum += input_2d[[b, i]] * grad_b_t[[b, r]];
                    }
                    dla[[i, r]] = self.scaling * sum;
                }
            }
        }

        // Gradient w.r.t. input: grad_input = grad_output @ W^T + scaling * grad_output @ B^T @ A^T
        let mut grad_input = Array::zeros(IxDyn(&[batch_size, input_dim]));

        // grad_output @ W^T
        for b in 0..batch_size {
            for i in 0..input_dim {
                let mut sum = F::zero();
                for j in 0..output_dim {
                    sum += grad_output[[b, j]] * self.base_weights[[i, j]];
                }
                grad_input[[b, i]] = sum;
            }
        }

        // + scaling * grad_b_t @ A^T
        if !self.merged {
            for b in 0..batch_size {
                for i in 0..input_dim {
                    let mut sum = F::zero();
                    for r in 0..rank {
                        sum += grad_b_t[[b, r]] * self.lora_a[[i, r]];
                    }
                    grad_input[[b, i]] += self.scaling * sum;
                }
            }
        }

        Ok(grad_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update LoRA A
        {
            let dla = self
                .grad_lora_a
                .read()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            let input_dim = self.config.input_dim;
            let rank = self.config.rank;
            for i in 0..input_dim {
                for r in 0..rank {
                    self.lora_a[[i, r]] -= learning_rate * dla[[i, r]];
                }
            }
        }

        // Update LoRA B
        {
            let dlb = self
                .grad_lora_b
                .read()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            let rank = self.config.rank;
            let output_dim = self.config.output_dim;
            for r in 0..rank {
                for j in 0..output_dim {
                    self.lora_b[[r, j]] -= learning_rate * dlb[[r, j]];
                }
            }
        }

        // Update base weights if not frozen
        if !self.config.freeze_base {
            let dw = self
                .grad_base_weights
                .read()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            let input_dim = self.config.input_dim;
            let output_dim = self.config.output_dim;
            for i in 0..input_dim {
                for j in 0..output_dim {
                    self.base_weights[[i, j]] -= learning_rate * dw[[i, j]];
                }
            }

            // Update bias
            if let Some(ref mut bias) = self.bias {
                let db = self
                    .grad_bias
                    .read()
                    .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
                if let Some(ref db_arr) = *db {
                    for j in 0..self.config.output_dim {
                        bias[j] -= learning_rate * db_arr[j];
                    }
                }
            }
        }

        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut p = vec![self.lora_a.clone(), self.lora_b.clone()];
        if !self.config.freeze_base {
            p.push(self.base_weights.clone());
            if let Some(ref bias) = self.bias {
                p.push(bias.clone());
            }
        }
        p
    }

    fn gradients(&self) -> Vec<Array<F, IxDyn>> {
        let dla = self
            .grad_lora_a
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| Array::zeros(IxDyn(&[self.config.input_dim, self.config.rank])));
        let dlb = self
            .grad_lora_b
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| Array::zeros(IxDyn(&[self.config.rank, self.config.output_dim])));
        let mut grads = vec![dla, dlb];
        if !self.config.freeze_base {
            let dw = self
                .grad_base_weights
                .read()
                .map(|g| g.clone())
                .unwrap_or_else(|_| {
                    Array::zeros(IxDyn(&[self.config.input_dim, self.config.output_dim]))
                });
            grads.push(dw);
        }
        grads
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "LoRA"
    }

    fn parameter_count(&self) -> usize {
        let base = self.config.input_dim * self.config.output_dim;
        let lora =
            self.config.input_dim * self.config.rank + self.config.rank * self.config.output_dim;
        let bias = if self.config.use_bias {
            self.config.output_dim
        } else {
            0
        };
        base + lora + bias
    }

    fn layer_description(&self) -> String {
        format!(
            "LoRA(in={}, out={}, rank={}, alpha={}, frozen={}, merged={})",
            self.config.input_dim,
            self.config.output_dim,
            self.config.rank,
            self.config.alpha,
            self.config.freeze_base,
            self.merged
        )
    }

    fn inputshape(&self) -> Option<Vec<usize>> {
        Some(vec![self.config.input_dim])
    }

    fn outputshape(&self) -> Option<Vec<usize>> {
        Some(vec![self.config.output_dim])
    }

    fn name(&self) -> Option<&str> {
        Some("LoRA")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::rng;

    #[test]
    fn test_lora_construction_default() {
        let mut r = rng();
        let config = LoraConfig::default();
        let layer = LoraLayer::<f64>::new(config, &mut r).unwrap();
        assert_eq!(layer.rank(), 4);
        assert!(!layer.is_merged());
        assert_eq!(layer.lora_a().shape(), &[64, 4]);
        assert_eq!(layer.lora_b().shape(), &[4, 64]);
    }

    #[test]
    fn test_lora_construction_custom() {
        let mut r = rng();
        let config = LoraConfig {
            input_dim: 128,
            output_dim: 64,
            rank: 8,
            alpha: 16.0,
            dropout: 0.1,
            use_bias: false,
            freeze_base: true,
        };
        let layer = LoraLayer::<f64>::new(config, &mut r).unwrap();
        assert_eq!(layer.rank(), 8);
        assert_eq!(layer.lora_a().shape(), &[128, 8]);
        assert_eq!(layer.lora_b().shape(), &[8, 64]);
        assert_eq!(layer.trainable_parameter_count(), 128 * 8 + 8 * 64);
    }

    #[test]
    fn test_lora_construction_invalid() {
        let mut r = rng();
        let config = LoraConfig {
            rank: 0,
            ..Default::default()
        };
        assert!(LoraLayer::<f64>::new(config, &mut r).is_err());

        let config2 = LoraConfig {
            input_dim: 0,
            ..Default::default()
        };
        assert!(LoraLayer::<f64>::new(config2, &mut r).is_err());

        let config3 = LoraConfig {
            dropout: 1.0,
            ..Default::default()
        };
        assert!(LoraLayer::<f64>::new(config3, &mut r).is_err());
    }

    #[test]
    fn test_lora_forward_shape() {
        let mut r = rng();
        let config = LoraConfig {
            input_dim: 16,
            output_dim: 8,
            rank: 4,
            alpha: 1.0,
            dropout: 0.0,
            use_bias: true,
            freeze_base: true,
        };
        let layer = LoraLayer::<f64>::new(config, &mut r).unwrap();

        // Single sample
        let input = Array::zeros(IxDyn(&[1, 16]));
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 8]);

        // Batch
        let input_batch = Array::zeros(IxDyn(&[4, 16]));
        let output_batch = layer.forward(&input_batch).unwrap();
        assert_eq!(output_batch.shape(), &[4, 8]);
    }

    #[test]
    fn test_lora_initial_output_matches_base() {
        // Since lora_b is initialized to zeros, the LoRA contribution should be zero
        // at initialization. Output should equal base linear transform.
        let mut r = rng();
        let config = LoraConfig {
            input_dim: 8,
            output_dim: 4,
            rank: 2,
            alpha: 1.0,
            dropout: 0.0,
            use_bias: true,
            freeze_base: true,
        };
        let layer = LoraLayer::<f64>::new(config, &mut r).unwrap();

        let input = Array::from_shape_vec(IxDyn(&[1, 8]), vec![1.0; 8]).unwrap();
        let output = layer.forward(&input).unwrap();

        // Manually compute W * x + b
        let w = layer.base_weights();
        let mut expected = Array::zeros(IxDyn(&[1, 4]));
        for j in 0..4 {
            let mut sum = 0.0;
            for i in 0..8 {
                sum += input[[0, i]] * w[[i, j]];
            }
            expected[[0, j]] = sum; // bias is zero initially
        }
        for j in 0..4 {
            assert!(
                (output[[0, j]] - expected[[0, j]]).abs() < 1e-10,
                "Output mismatch at index {j}"
            );
        }
    }

    #[test]
    fn test_lora_merge_unmerge_roundtrip() {
        let mut r = rng();
        let config = LoraConfig {
            input_dim: 8,
            output_dim: 4,
            rank: 2,
            alpha: 2.0,
            dropout: 0.0,
            use_bias: true,
            freeze_base: true,
        };
        let mut layer = LoraLayer::<f64>::new(config, &mut r).unwrap();

        // Set some non-zero LoRA B weights to make the test meaningful
        let mut lora_b = Array::zeros(IxDyn(&[2, 4]));
        lora_b[[0, 0]] = 0.5;
        lora_b[[1, 1]] = 0.3;
        layer.set_lora_b(lora_b).unwrap();

        let input = Array::from_shape_vec(IxDyn(&[2, 8]), vec![0.5; 16]).unwrap();

        // Get output before merge
        let output_before = layer.forward(&input).unwrap();

        // Merge
        layer.merge().unwrap();
        assert!(layer.is_merged());
        let output_merged = layer.forward(&input).unwrap();

        // Outputs should be the same (within floating point tolerance)
        for i in 0..output_before.len() {
            assert!(
                (output_before.as_slice().unwrap()[i] - output_merged.as_slice().unwrap()[i]).abs()
                    < 1e-10,
                "Merged output differs at index {i}"
            );
        }

        // Unmerge
        layer.unmerge().unwrap();
        assert!(!layer.is_merged());
        let output_unmerged = layer.forward(&input).unwrap();

        // Should still match
        for i in 0..output_before.len() {
            assert!(
                (output_before.as_slice().unwrap()[i] - output_unmerged.as_slice().unwrap()[i])
                    .abs()
                    < 1e-10,
                "Unmerged output differs at index {i}"
            );
        }
    }

    #[test]
    fn test_lora_merge_already_merged_error() {
        let mut r = rng();
        let config = LoraConfig::default();
        let mut layer = LoraLayer::<f64>::new(config, &mut r).unwrap();
        layer.merge().unwrap();
        assert!(layer.merge().is_err());
    }

    #[test]
    fn test_lora_unmerge_not_merged_error() {
        let mut r = rng();
        let config = LoraConfig::default();
        let mut layer = LoraLayer::<f64>::new(config, &mut r).unwrap();
        assert!(layer.unmerge().is_err());
    }

    #[test]
    fn test_lora_from_pretrained() {
        let mut r = rng();
        let pretrained = Array::from_shape_vec(IxDyn(&[8, 4]), vec![0.1; 32]).unwrap();
        let bias = Some(Array::from_shape_vec(IxDyn(&[4]), vec![0.01; 4]).unwrap());
        let layer = LoraLayer::<f64>::from_pretrained(pretrained, bias, 2, 4.0, &mut r).unwrap();
        assert_eq!(layer.rank(), 2);
        assert_eq!(layer.base_weights().shape(), &[8, 4]);
    }

    #[test]
    fn test_lora_backward_shape() {
        let mut r = rng();
        let config = LoraConfig {
            input_dim: 8,
            output_dim: 4,
            rank: 2,
            alpha: 1.0,
            dropout: 0.0,
            use_bias: true,
            freeze_base: false,
        };
        let layer = LoraLayer::<f64>::new(config, &mut r).unwrap();

        let input = Array::from_shape_vec(IxDyn(&[2, 8]), vec![0.5; 16]).unwrap();
        let _output = layer.forward(&input).unwrap();

        let grad_output = Array::from_shape_vec(IxDyn(&[2, 4]), vec![1.0; 8]).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), &[2, 8]);
    }

    #[test]
    fn test_lora_compression_ratio() {
        let mut r = rng();
        let config = LoraConfig {
            input_dim: 1024,
            output_dim: 1024,
            rank: 4,
            alpha: 1.0,
            dropout: 0.0,
            use_bias: false,
            freeze_base: true,
        };
        let layer = LoraLayer::<f64>::new(config, &mut r).unwrap();
        // LoRA params: 1024*4 + 4*1024 = 8192
        // Total base params: 1024*1024 = 1048576
        let ratio = layer.compression_ratio();
        assert!(ratio < 0.01, "Compression ratio should be < 1%: {ratio}");
    }

    #[test]
    fn test_lora_layer_trait() {
        let mut r = rng();
        let config = LoraConfig {
            input_dim: 16,
            output_dim: 8,
            rank: 4,
            ..Default::default()
        };
        let mut layer = LoraLayer::<f64>::new(config, &mut r).unwrap();
        assert_eq!(layer.layer_type(), "LoRA");
        assert!(layer.is_training());
        layer.set_training(false);
        assert!(!layer.is_training());
        assert!(layer.inputshape().is_some());
        assert!(layer.outputshape().is_some());
    }
}
