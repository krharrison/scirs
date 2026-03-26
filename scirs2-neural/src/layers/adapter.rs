//! Adapter layer implementation for parameter-efficient fine-tuning
//!
//! Adapter layers are small bottleneck modules inserted between transformer layers.
//! They consist of a down-projection, a nonlinearity, and an up-projection with a
//! residual connection: output = x + up_project(activation(down_project(x)))
//!
//! Reference: Houlsby et al., "Parameter-Efficient Transfer Learning for NLP" (2019)

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Distribution, Normal};
use std::fmt::Debug;
use std::sync::RwLock;

/// Activation function choices for the adapter bottleneck
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdapterActivation {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Gaussian Error Linear Unit
    GELU,
    /// Hyperbolic tangent
    Tanh,
    /// Sigmoid
    Sigmoid,
    /// Swish / SiLU: x * sigmoid(x)
    Swish,
}

/// Configuration for an adapter layer
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    /// Input/output dimension (adapter preserves dimension via residual)
    pub hidden_dim: usize,
    /// Bottleneck dimension (smaller than hidden_dim for compression)
    pub bottleneck_dim: usize,
    /// Activation function in the bottleneck
    pub activation: AdapterActivation,
    /// Whether to use layer normalization before the adapter
    pub use_layer_norm: bool,
    /// Scaling factor applied to the adapter output before adding residual
    pub residual_scale: f64,
    /// Whether to initialize near-identity (output close to zero initially)
    pub init_near_identity: bool,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 64,
            bottleneck_dim: 16,
            activation: AdapterActivation::ReLU,
            use_layer_norm: false,
            residual_scale: 1.0,
            init_near_identity: true,
        }
    }
}

/// Bottleneck adapter layer
///
/// Implements: output = x + scale * up_project(activation(down_project(norm(x))))
///
/// The bottleneck compresses the representation to `bottleneck_dim`, applies a
/// nonlinearity, then projects back to the original dimension. The residual
/// connection ensures the adapter can be initialized to approximate identity.
pub struct AdapterLayer<F: Float + Debug + Send + Sync + NumAssign> {
    config: AdapterConfig,
    /// Down-projection: [hidden_dim, bottleneck_dim]
    down_weights: Array<F, IxDyn>,
    /// Down-projection bias: [bottleneck_dim]
    down_bias: Array<F, IxDyn>,
    /// Up-projection: [bottleneck_dim, hidden_dim]
    up_weights: Array<F, IxDyn>,
    /// Up-projection bias: [hidden_dim]
    up_bias: Array<F, IxDyn>,
    /// Layer norm gamma (scale): [hidden_dim]
    ln_gamma: Option<Array<F, IxDyn>>,
    /// Layer norm beta (shift): [hidden_dim]
    ln_beta: Option<Array<F, IxDyn>>,
    /// Residual scaling factor
    residual_scale: F,
    /// Training mode flag
    training: bool,
    /// Cached input for backward pass
    cached_input: RwLock<Option<Array<F, IxDyn>>>,
    /// Cached post-down-projection (pre-activation) for backward pass
    cached_down_output: RwLock<Option<Array<F, IxDyn>>>,
    /// Cached post-activation for backward pass
    cached_activated: RwLock<Option<Array<F, IxDyn>>>,
    /// Cached normalized input for backward pass
    cached_normed: RwLock<Option<Array<F, IxDyn>>>,
    // Gradients
    grad_down_weights: RwLock<Array<F, IxDyn>>,
    grad_down_bias: RwLock<Array<F, IxDyn>>,
    grad_up_weights: RwLock<Array<F, IxDyn>>,
    grad_up_bias: RwLock<Array<F, IxDyn>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> std::fmt::Debug
    for AdapterLayer<F>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdapterLayer")
            .field("hidden_dim", &self.config.hidden_dim)
            .field("bottleneck_dim", &self.config.bottleneck_dim)
            .field("activation", &self.config.activation)
            .field("use_layer_norm", &self.config.use_layer_norm)
            .finish()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> AdapterLayer<F> {
    /// Create a new adapter layer
    ///
    /// # Arguments
    /// * `config` - Adapter configuration
    /// * `rng` - Random number generator for weight initialization
    pub fn new<R: scirs2_core::random::Rng>(config: AdapterConfig, rng: &mut R) -> Result<Self> {
        if config.hidden_dim == 0 || config.bottleneck_dim == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "Hidden and bottleneck dimensions must be > 0".to_string(),
            ));
        }
        if config.bottleneck_dim > config.hidden_dim {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Bottleneck dim ({}) should not exceed hidden dim ({})",
                config.bottleneck_dim, config.hidden_dim
            )));
        }

        let hidden_dim = config.hidden_dim;
        let bottleneck_dim = config.bottleneck_dim;

        // Initialize down-projection with small random values
        let std_down = if config.init_near_identity {
            1e-3
        } else {
            1.0 / f64::sqrt(hidden_dim as f64)
        };
        let normal_down = Normal::new(0.0, std_down).map_err(|e| {
            NeuralError::InvalidArchitecture(format!("Failed to create normal distribution: {e}"))
        })?;

        let down_weights_vec: Vec<F> = (0..(hidden_dim * bottleneck_dim))
            .map(|_| F::from(normal_down.sample(rng)).unwrap_or(F::zero()))
            .collect();
        let down_weights = Array::from_shape_vec(
            IxDyn(&[hidden_dim, bottleneck_dim]),
            down_weights_vec,
        )
        .map_err(|e| {
            NeuralError::InvalidArchitecture(format!("Failed to create down_weights array: {e}"))
        })?;
        let down_bias = Array::zeros(IxDyn(&[bottleneck_dim]));

        // Initialize up-projection
        let std_up = if config.init_near_identity {
            1e-3
        } else {
            1.0 / f64::sqrt(bottleneck_dim as f64)
        };
        let normal_up = Normal::new(0.0, std_up).map_err(|e| {
            NeuralError::InvalidArchitecture(format!("Failed to create normal distribution: {e}"))
        })?;

        let up_weights_vec: Vec<F> = (0..(bottleneck_dim * hidden_dim))
            .map(|_| F::from(normal_up.sample(rng)).unwrap_or(F::zero()))
            .collect();
        let up_weights = Array::from_shape_vec(
            IxDyn(&[bottleneck_dim, hidden_dim]),
            up_weights_vec,
        )
        .map_err(|e| {
            NeuralError::InvalidArchitecture(format!("Failed to create up_weights array: {e}"))
        })?;
        let up_bias = Array::zeros(IxDyn(&[hidden_dim]));

        // Initialize layer norm parameters
        let (ln_gamma, ln_beta) = if config.use_layer_norm {
            (
                Some(Array::ones(IxDyn(&[hidden_dim]))),
                Some(Array::zeros(IxDyn(&[hidden_dim]))),
            )
        } else {
            (None, None)
        };

        let residual_scale = F::from(config.residual_scale).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert residual_scale".to_string())
        })?;

        let grad_down_weights = RwLock::new(Array::zeros(IxDyn(&[hidden_dim, bottleneck_dim])));
        let grad_down_bias = RwLock::new(Array::zeros(IxDyn(&[bottleneck_dim])));
        let grad_up_weights = RwLock::new(Array::zeros(IxDyn(&[bottleneck_dim, hidden_dim])));
        let grad_up_bias = RwLock::new(Array::zeros(IxDyn(&[hidden_dim])));

        Ok(Self {
            config,
            down_weights,
            down_bias,
            up_weights,
            up_bias,
            ln_gamma,
            ln_beta,
            residual_scale,
            training: true,
            cached_input: RwLock::new(None),
            cached_down_output: RwLock::new(None),
            cached_activated: RwLock::new(None),
            cached_normed: RwLock::new(None),
            grad_down_weights,
            grad_down_bias,
            grad_up_weights,
            grad_up_bias,
        })
    }

    /// Get the bottleneck dimension
    pub fn bottleneck_dim(&self) -> usize {
        self.config.bottleneck_dim
    }

    /// Get the hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    /// Get the number of trainable parameters
    pub fn trainable_parameter_count(&self) -> usize {
        let down_params =
            self.config.hidden_dim * self.config.bottleneck_dim + self.config.bottleneck_dim;
        let up_params =
            self.config.bottleneck_dim * self.config.hidden_dim + self.config.hidden_dim;
        let ln_params = if self.config.use_layer_norm {
            2 * self.config.hidden_dim
        } else {
            0
        };
        down_params + up_params + ln_params
    }

    /// Get the compression ratio (bottleneck_dim / hidden_dim)
    pub fn compression_ratio(&self) -> f64 {
        self.config.bottleneck_dim as f64 / self.config.hidden_dim as f64
    }

    /// Apply the chosen activation function element-wise
    fn apply_activation(&self, x: &Array<F, IxDyn>) -> Array<F, IxDyn> {
        x.mapv(|v| self.activate_scalar(v))
    }

    /// Compute derivative of the activation function
    fn activation_derivative(&self, x: &Array<F, IxDyn>) -> Array<F, IxDyn> {
        x.mapv(|v| self.activate_derivative_scalar(v))
    }

    /// Scalar activation
    fn activate_scalar(&self, x: F) -> F {
        match self.config.activation {
            AdapterActivation::ReLU => {
                if x > F::zero() {
                    x
                } else {
                    F::zero()
                }
            }
            AdapterActivation::GELU => {
                // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let sqrt_2_pi = F::from(0.7978845608).unwrap_or(F::one());
                let coeff = F::from(0.044715).unwrap_or(F::zero());
                let half = F::from(0.5).unwrap_or(F::zero());
                let inner = sqrt_2_pi * (x + coeff * x * x * x);
                half * x * (F::one() + inner.tanh())
            }
            AdapterActivation::Tanh => x.tanh(),
            AdapterActivation::Sigmoid => F::one() / (F::one() + (-x).exp()),
            AdapterActivation::Swish => {
                let sigmoid = F::one() / (F::one() + (-x).exp());
                x * sigmoid
            }
        }
    }

    /// Scalar activation derivative
    fn activate_derivative_scalar(&self, x: F) -> F {
        match self.config.activation {
            AdapterActivation::ReLU => {
                if x > F::zero() {
                    F::one()
                } else {
                    F::zero()
                }
            }
            AdapterActivation::GELU => {
                // Numerical approximation of GELU derivative
                let eps = F::from(1e-5).unwrap_or(F::zero());
                let f_plus = self.activate_scalar(x + eps);
                let f_minus = self.activate_scalar(x - eps);
                let two_eps = eps + eps;
                if two_eps > F::zero() {
                    (f_plus - f_minus) / two_eps
                } else {
                    F::one()
                }
            }
            AdapterActivation::Tanh => {
                let t = x.tanh();
                F::one() - t * t
            }
            AdapterActivation::Sigmoid => {
                let s = F::one() / (F::one() + (-x).exp());
                s * (F::one() - s)
            }
            AdapterActivation::Swish => {
                let s = F::one() / (F::one() + (-x).exp());
                s + x * s * (F::one() - s)
            }
        }
    }

    /// Apply layer normalization to input
    fn apply_layer_norm(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let gamma = self.ln_gamma.as_ref().ok_or_else(|| {
            NeuralError::InvalidState("Layer norm gamma not initialized".to_string())
        })?;
        let beta = self.ln_beta.as_ref().ok_or_else(|| {
            NeuralError::InvalidState("Layer norm beta not initialized".to_string())
        })?;

        let batch_size = input.shape()[0];
        let hidden_dim = self.config.hidden_dim;
        let eps = F::from(1e-5).unwrap_or(F::zero());

        let mut output = Array::zeros(input.raw_dim());
        for b in 0..batch_size {
            // Compute mean
            let mut mean = F::zero();
            for i in 0..hidden_dim {
                mean += input[[b, i]];
            }
            mean = mean / F::from(hidden_dim).unwrap_or(F::one());

            // Compute variance
            let mut var = F::zero();
            for i in 0..hidden_dim {
                let diff = input[[b, i]] - mean;
                var += diff * diff;
            }
            var = var / F::from(hidden_dim).unwrap_or(F::one());

            // Normalize and apply affine
            let inv_std = F::one() / (var + eps).sqrt();
            for i in 0..hidden_dim {
                output[[b, i]] = gamma[i] * (input[[b, i]] - mean) * inv_std + beta[i];
            }
        }

        Ok(output)
    }

    /// Compute the adapter forward pass
    fn compute_forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let batch_size = input.shape()[0];
        let hidden_dim = self.config.hidden_dim;
        let bottleneck_dim = self.config.bottleneck_dim;

        // Optional layer normalization
        let normed = if self.config.use_layer_norm {
            let n = self.apply_layer_norm(input)?;
            {
                let mut cache = self
                    .cached_normed
                    .write()
                    .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
                *cache = Some(n.clone());
            }
            n
        } else {
            input.clone()
        };

        // Down-projection: [batch, hidden] @ [hidden, bottleneck] = [batch, bottleneck]
        let mut down_output = Array::zeros(IxDyn(&[batch_size, bottleneck_dim]));
        for b in 0..batch_size {
            for j in 0..bottleneck_dim {
                let mut sum = F::zero();
                for i in 0..hidden_dim {
                    sum += normed[[b, i]] * self.down_weights[[i, j]];
                }
                down_output[[b, j]] = sum + self.down_bias[j];
            }
        }

        // Cache for backward
        {
            let mut cache = self
                .cached_down_output
                .write()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            *cache = Some(down_output.clone());
        }

        // Apply activation
        let activated = self.apply_activation(&down_output);
        {
            let mut cache = self
                .cached_activated
                .write()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            *cache = Some(activated.clone());
        }

        // Up-projection: [batch, bottleneck] @ [bottleneck, hidden] = [batch, hidden]
        let mut up_output = Array::zeros(IxDyn(&[batch_size, hidden_dim]));
        for b in 0..batch_size {
            for j in 0..hidden_dim {
                let mut sum = F::zero();
                for i in 0..bottleneck_dim {
                    sum += activated[[b, i]] * self.up_weights[[i, j]];
                }
                up_output[[b, j]] = sum + self.up_bias[j];
            }
        }

        // Residual connection: output = input + scale * adapter_output
        let mut output = input.clone();
        for b in 0..batch_size {
            for j in 0..hidden_dim {
                output[[b, j]] += self.residual_scale * up_output[[b, j]];
            }
        }

        Ok(output)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for AdapterLayer<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input
        {
            let mut cache = self
                .cached_input
                .write()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            *cache = Some(input.clone());
        }

        // Ensure 2D
        let input_2d = if input.ndim() == 1 {
            input
                .clone()
                .into_shape_with_order(IxDyn(&[1, self.config.hidden_dim]))
                .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape input: {e}")))?
        } else {
            input.clone()
        };

        if input_2d.shape()[1] != self.config.hidden_dim {
            return Err(NeuralError::InvalidArgument(format!(
                "Input dimension mismatch: expected {}, got {}",
                self.config.hidden_dim,
                input_2d.shape()[1]
            )));
        }

        self.compute_forward(&input_2d)
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
                NeuralError::InferenceError("No cached input for backward".to_string())
            })?
        };

        let cached_down = {
            let cache = self
                .cached_down_output
                .read()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            cache.clone().ok_or_else(|| {
                NeuralError::InferenceError("No cached down output for backward".to_string())
            })?
        };

        let cached_act = {
            let cache = self
                .cached_activated
                .read()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            cache.clone().ok_or_else(|| {
                NeuralError::InferenceError("No cached activation for backward".to_string())
            })?
        };

        let batch_size = grad_output.shape()[0];
        let hidden_dim = self.config.hidden_dim;
        let bottleneck_dim = self.config.bottleneck_dim;

        // The output was: output = input + scale * up_project(act(down_project(input)))
        // grad_output flows through both the residual and the adapter path

        // Scale the adapter path gradient
        // d(adapter_output)/d(up_output) = scale
        // grad_up_output = scale * grad_output
        // But residual also passes grad_output through to input directly

        // Gradient through up-projection
        // grad_up_bias = sum over batch of (scale * grad_output)
        {
            let mut gub = self
                .grad_up_bias
                .write()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            for j in 0..hidden_dim {
                let mut sum = F::zero();
                for b in 0..batch_size {
                    sum += self.residual_scale * grad_output[[b, j]];
                }
                gub[j] = sum;
            }
        }

        // grad_up_weights = cached_act^T @ (scale * grad_output)
        {
            let mut guw = self
                .grad_up_weights
                .write()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            for i in 0..bottleneck_dim {
                for j in 0..hidden_dim {
                    let mut sum = F::zero();
                    for b in 0..batch_size {
                        sum += cached_act[[b, i]] * self.residual_scale * grad_output[[b, j]];
                    }
                    guw[[i, j]] = sum;
                }
            }
        }

        // Gradient flowing to activated: grad_act = (scale * grad_output) @ up_weights^T
        let mut grad_act = Array::zeros(IxDyn(&[batch_size, bottleneck_dim]));
        for b in 0..batch_size {
            for i in 0..bottleneck_dim {
                let mut sum = F::zero();
                for j in 0..hidden_dim {
                    sum += self.residual_scale * grad_output[[b, j]] * self.up_weights[[i, j]];
                }
                grad_act[[b, i]] = sum;
            }
        }

        // Gradient through activation
        let act_deriv = self.activation_derivative(&cached_down);
        let grad_down_out = &grad_act * &act_deriv;

        // grad_down_bias
        {
            let mut gdb = self
                .grad_down_bias
                .write()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            for j in 0..bottleneck_dim {
                let mut sum = F::zero();
                for b in 0..batch_size {
                    sum += grad_down_out[[b, j]];
                }
                gdb[j] = sum;
            }
        }

        // Determine the input to the down-projection for gradient computation
        let normed_input = if self.config.use_layer_norm {
            let cache = self
                .cached_normed
                .read()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            cache.clone().unwrap_or(cached_input.clone())
        } else {
            let input_2d = if cached_input.ndim() == 1 {
                cached_input
                    .clone()
                    .into_shape_with_order(IxDyn(&[1, hidden_dim]))
                    .map_err(|e| NeuralError::InferenceError(format!("Reshape failed: {e}")))?
            } else {
                cached_input
            };
            input_2d
        };

        // grad_down_weights = normed_input^T @ grad_down_out
        {
            let mut gdw = self
                .grad_down_weights
                .write()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            for i in 0..hidden_dim {
                for j in 0..bottleneck_dim {
                    let mut sum = F::zero();
                    for b in 0..batch_size {
                        sum += normed_input[[b, i]] * grad_down_out[[b, j]];
                    }
                    gdw[[i, j]] = sum;
                }
            }
        }

        // Gradient w.r.t. input = grad_output (from residual)
        //   + grad_down_out @ down_weights^T (from adapter path)
        let mut grad_input = grad_output.clone();
        for b in 0..batch_size {
            for i in 0..hidden_dim {
                let mut sum = F::zero();
                for j in 0..bottleneck_dim {
                    sum += grad_down_out[[b, j]] * self.down_weights[[i, j]];
                }
                grad_input[[b, i]] += sum;
            }
        }

        Ok(grad_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update down-projection
        {
            let gdw = self
                .grad_down_weights
                .read()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            let hidden_dim = self.config.hidden_dim;
            let bottleneck_dim = self.config.bottleneck_dim;
            for i in 0..hidden_dim {
                for j in 0..bottleneck_dim {
                    self.down_weights[[i, j]] -= learning_rate * gdw[[i, j]];
                }
            }
        }
        {
            let gdb = self
                .grad_down_bias
                .read()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            for j in 0..self.config.bottleneck_dim {
                self.down_bias[j] -= learning_rate * gdb[j];
            }
        }

        // Update up-projection
        {
            let guw = self
                .grad_up_weights
                .read()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            let bottleneck_dim = self.config.bottleneck_dim;
            let hidden_dim = self.config.hidden_dim;
            for i in 0..bottleneck_dim {
                for j in 0..hidden_dim {
                    self.up_weights[[i, j]] -= learning_rate * guw[[i, j]];
                }
            }
        }
        {
            let gub = self
                .grad_up_bias
                .read()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            for j in 0..self.config.hidden_dim {
                self.up_bias[j] -= learning_rate * gub[j];
            }
        }

        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut p = vec![
            self.down_weights.clone(),
            self.down_bias.clone(),
            self.up_weights.clone(),
            self.up_bias.clone(),
        ];
        if let Some(ref gamma) = self.ln_gamma {
            p.push(gamma.clone());
        }
        if let Some(ref beta) = self.ln_beta {
            p.push(beta.clone());
        }
        p
    }

    fn gradients(&self) -> Vec<Array<F, IxDyn>> {
        let gdw = self
            .grad_down_weights
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| {
                Array::zeros(IxDyn(&[self.config.hidden_dim, self.config.bottleneck_dim]))
            });
        let gdb = self
            .grad_down_bias
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| Array::zeros(IxDyn(&[self.config.bottleneck_dim])));
        let guw = self
            .grad_up_weights
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| {
                Array::zeros(IxDyn(&[self.config.bottleneck_dim, self.config.hidden_dim]))
            });
        let gub = self
            .grad_up_bias
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| Array::zeros(IxDyn(&[self.config.hidden_dim])));
        vec![gdw, gdb, guw, gub]
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
        "Adapter"
    }

    fn parameter_count(&self) -> usize {
        self.trainable_parameter_count()
    }

    fn layer_description(&self) -> String {
        format!(
            "Adapter(hidden={}, bottleneck={}, activation={:?}, ln={}, scale={})",
            self.config.hidden_dim,
            self.config.bottleneck_dim,
            self.config.activation,
            self.config.use_layer_norm,
            self.config.residual_scale
        )
    }

    fn inputshape(&self) -> Option<Vec<usize>> {
        Some(vec![self.config.hidden_dim])
    }

    fn outputshape(&self) -> Option<Vec<usize>> {
        Some(vec![self.config.hidden_dim])
    }

    fn name(&self) -> Option<&str> {
        Some("Adapter")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::rng;

    #[test]
    fn test_adapter_construction_default() {
        let mut r = rng();
        let config = AdapterConfig::default();
        let layer = AdapterLayer::<f64>::new(config, &mut r).unwrap();
        assert_eq!(layer.hidden_dim(), 64);
        assert_eq!(layer.bottleneck_dim(), 16);
    }

    #[test]
    fn test_adapter_construction_custom() {
        let mut r = rng();
        let config = AdapterConfig {
            hidden_dim: 256,
            bottleneck_dim: 32,
            activation: AdapterActivation::GELU,
            use_layer_norm: true,
            residual_scale: 0.5,
            init_near_identity: true,
        };
        let layer = AdapterLayer::<f64>::new(config, &mut r).unwrap();
        assert_eq!(layer.hidden_dim(), 256);
        assert_eq!(layer.bottleneck_dim(), 32);
        assert!(layer.compression_ratio() < 0.15);
    }

    #[test]
    fn test_adapter_construction_invalid() {
        let mut r = rng();
        let config = AdapterConfig {
            hidden_dim: 0,
            ..Default::default()
        };
        assert!(AdapterLayer::<f64>::new(config, &mut r).is_err());

        let config2 = AdapterConfig {
            hidden_dim: 16,
            bottleneck_dim: 32, // larger than hidden
            ..Default::default()
        };
        assert!(AdapterLayer::<f64>::new(config2, &mut r).is_err());
    }

    #[test]
    fn test_adapter_forward_shape() {
        let mut r = rng();
        let config = AdapterConfig {
            hidden_dim: 32,
            bottleneck_dim: 8,
            ..Default::default()
        };
        let layer = AdapterLayer::<f64>::new(config, &mut r).unwrap();

        // Single sample
        let input = Array::zeros(IxDyn(&[1, 32]));
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 32]); // Same shape as input (residual)

        // Batch
        let input_batch = Array::zeros(IxDyn(&[4, 32]));
        let output_batch = layer.forward(&input_batch).unwrap();
        assert_eq!(output_batch.shape(), &[4, 32]);
    }

    #[test]
    fn test_adapter_residual_connection() {
        // With near-identity initialization, the adapter output should be close to input
        let mut r = rng();
        let config = AdapterConfig {
            hidden_dim: 8,
            bottleneck_dim: 2,
            init_near_identity: true,
            residual_scale: 1.0,
            ..Default::default()
        };
        let layer = AdapterLayer::<f64>::new(config, &mut r).unwrap();

        let input =
            Array::from_shape_vec(IxDyn(&[1, 8]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                .unwrap();
        let output = layer.forward(&input).unwrap();

        // Output should be approximately the input since adapter is initialized near zero
        for i in 0..8 {
            let diff = (output[[0, i]] - input[[0, i]]).abs();
            assert!(
                diff < 0.5,
                "Residual connection: output should be close to input, diff={diff} at idx={i}"
            );
        }
    }

    #[test]
    fn test_adapter_with_layer_norm() {
        let mut r = rng();
        let config = AdapterConfig {
            hidden_dim: 16,
            bottleneck_dim: 4,
            use_layer_norm: true,
            ..Default::default()
        };
        let layer = AdapterLayer::<f64>::new(config, &mut r).unwrap();

        let input =
            Array::from_shape_vec(IxDyn(&[2, 16]), (0..32).map(|i| i as f64 * 0.1).collect())
                .unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 16]);
    }

    #[test]
    fn test_adapter_different_activations() {
        let mut r = rng();
        let activations = [
            AdapterActivation::ReLU,
            AdapterActivation::GELU,
            AdapterActivation::Tanh,
            AdapterActivation::Sigmoid,
            AdapterActivation::Swish,
        ];

        let input = Array::from_shape_vec(IxDyn(&[1, 8]), vec![0.5; 8]).unwrap();

        for act in &activations {
            let config = AdapterConfig {
                hidden_dim: 8,
                bottleneck_dim: 4,
                activation: *act,
                ..Default::default()
            };
            let layer = AdapterLayer::<f64>::new(config, &mut r).unwrap();
            let output = layer.forward(&input).unwrap();
            assert_eq!(
                output.shape(),
                &[1, 8],
                "Activation {:?} failed shape check",
                act
            );
        }
    }

    #[test]
    fn test_adapter_backward_shape() {
        let mut r = rng();
        let config = AdapterConfig {
            hidden_dim: 16,
            bottleneck_dim: 4,
            ..Default::default()
        };
        let layer = AdapterLayer::<f64>::new(config, &mut r).unwrap();

        let input = Array::from_shape_vec(IxDyn(&[2, 16]), vec![0.5; 32]).unwrap();
        let _output = layer.forward(&input).unwrap();

        let grad_output = Array::from_shape_vec(IxDyn(&[2, 16]), vec![1.0; 32]).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), &[2, 16]);
    }

    #[test]
    fn test_adapter_parameter_count() {
        let mut r = rng();
        let config = AdapterConfig {
            hidden_dim: 64,
            bottleneck_dim: 16,
            use_layer_norm: false,
            ..Default::default()
        };
        let layer = AdapterLayer::<f64>::new(config, &mut r).unwrap();
        // down: 64*16 + 16 = 1040, up: 16*64 + 64 = 1088, total = 2128
        assert_eq!(layer.trainable_parameter_count(), 2128);
    }

    #[test]
    fn test_adapter_layer_trait() {
        let mut r = rng();
        let config = AdapterConfig::default();
        let mut layer = AdapterLayer::<f64>::new(config, &mut r).unwrap();
        assert_eq!(layer.layer_type(), "Adapter");
        assert!(layer.is_training());
        layer.set_training(false);
        assert!(!layer.is_training());
        assert_eq!(layer.name(), Some("Adapter"));
    }
}
