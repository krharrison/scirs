//! Prefix Tuning layer implementation
//!
//! Prefix tuning prepends learnable "soft prompt" vectors to the key and value
//! matrices in attention layers. Only the prefix parameters are trained while the
//! rest of the model remains frozen.
//!
//! The prefix can optionally be reparameterized through a small MLP to stabilize
//! training and improve expressiveness.
//!
//! Reference: Li & Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation" (2021)

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Distribution, Normal};
use std::fmt::Debug;
use std::sync::RwLock;

/// Configuration for prefix tuning
#[derive(Debug, Clone)]
pub struct PrefixTuningConfig {
    /// Number of prefix tokens to prepend
    pub prefix_length: usize,
    /// Hidden dimension of the model (dimension of key/value vectors)
    pub hidden_dim: usize,
    /// Number of attention heads (for head-specific prefixes)
    pub num_heads: usize,
    /// Number of layers that share or have independent prefixes
    pub num_layers: usize,
    /// Whether to use an MLP reparameterization for the prefix embeddings
    pub use_reparameterization: bool,
    /// Hidden dimension of the reparameterization MLP (if used)
    pub reparam_hidden_dim: usize,
    /// Whether each layer gets independent prefix parameters
    pub independent_layers: bool,
    /// Initialization standard deviation
    pub init_std: f64,
}

impl Default for PrefixTuningConfig {
    fn default() -> Self {
        Self {
            prefix_length: 10,
            hidden_dim: 64,
            num_heads: 4,
            num_layers: 1,
            use_reparameterization: true,
            reparam_hidden_dim: 128,
            independent_layers: false,
            init_std: 0.02,
        }
    }
}

/// Prefix Tuning layer
///
/// Generates learnable key and value prefix vectors that are prepended to the
/// attention key/value matrices. The forward pass takes an input and returns
/// it concatenated with (or augmented by) the prefix embeddings.
///
/// When `use_reparameterization` is true, the prefix embeddings are generated
/// by passing learned embedding vectors through a 2-layer MLP:
///   prefix = W2 * tanh(W1 * embedding + b1) + b2
pub struct PrefixTuningLayer<F: Float + Debug + Send + Sync + NumAssign> {
    config: PrefixTuningConfig,
    /// Raw prefix embeddings: [num_effective_layers, prefix_length, hidden_dim]
    /// These are either used directly or passed through the reparam MLP
    prefix_embeddings: Array<F, IxDyn>,
    /// Reparameterization MLP weight 1: [hidden_dim, reparam_hidden_dim]
    reparam_w1: Option<Array<F, IxDyn>>,
    /// Reparameterization MLP bias 1: [reparam_hidden_dim]
    reparam_b1: Option<Array<F, IxDyn>>,
    /// Reparameterization MLP weight 2: [reparam_hidden_dim, hidden_dim]
    reparam_w2: Option<Array<F, IxDyn>>,
    /// Reparameterization MLP bias 2: [hidden_dim]
    reparam_b2: Option<Array<F, IxDyn>>,
    /// Training mode
    training: bool,
    /// Cached prefix output for backward
    cached_prefix_output: RwLock<Option<Array<F, IxDyn>>>,
    /// Cached input for backward
    cached_input: RwLock<Option<Array<F, IxDyn>>>,
    /// Cached intermediate MLP activation for backward
    cached_mlp_hidden: RwLock<Option<Array<F, IxDyn>>>,
    // Gradients
    grad_prefix_embeddings: RwLock<Array<F, IxDyn>>,
    grad_reparam_w1: RwLock<Option<Array<F, IxDyn>>>,
    grad_reparam_b1: RwLock<Option<Array<F, IxDyn>>>,
    grad_reparam_w2: RwLock<Option<Array<F, IxDyn>>>,
    grad_reparam_b2: RwLock<Option<Array<F, IxDyn>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> std::fmt::Debug
    for PrefixTuningLayer<F>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixTuningLayer")
            .field("prefix_length", &self.config.prefix_length)
            .field("hidden_dim", &self.config.hidden_dim)
            .field("num_heads", &self.config.num_heads)
            .field("num_layers", &self.config.num_layers)
            .field(
                "use_reparameterization",
                &self.config.use_reparameterization,
            )
            .field("independent_layers", &self.config.independent_layers)
            .finish()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> PrefixTuningLayer<F> {
    /// Create a new prefix tuning layer
    ///
    /// # Arguments
    /// * `config` - Prefix tuning configuration
    /// * `rng` - Random number generator for initialization
    pub fn new<R: scirs2_core::random::Rng>(
        config: PrefixTuningConfig,
        rng: &mut R,
    ) -> Result<Self> {
        if config.prefix_length == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "Prefix length must be > 0".to_string(),
            ));
        }
        if config.hidden_dim == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "Hidden dimension must be > 0".to_string(),
            ));
        }
        if config.num_heads == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "Number of heads must be > 0".to_string(),
            ));
        }
        if config.hidden_dim % config.num_heads != 0 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Hidden dim ({}) must be divisible by num_heads ({})",
                config.hidden_dim, config.num_heads
            )));
        }

        let num_effective_layers = if config.independent_layers {
            config.num_layers
        } else {
            1
        };

        let prefix_length = config.prefix_length;
        let hidden_dim = config.hidden_dim;

        // Initialize prefix embeddings
        let normal = Normal::new(0.0, config.init_std).map_err(|e| {
            NeuralError::InvalidArchitecture(format!("Failed to create normal distribution: {e}"))
        })?;

        let embed_size = num_effective_layers * prefix_length * hidden_dim;
        let embed_vec: Vec<F> = (0..embed_size)
            .map(|_| F::from(normal.sample(rng)).unwrap_or(F::zero()))
            .collect();
        let prefix_embeddings = Array::from_shape_vec(
            IxDyn(&[num_effective_layers, prefix_length, hidden_dim]),
            embed_vec,
        )
        .map_err(|e| {
            NeuralError::InvalidArchitecture(format!("Failed to create prefix embeddings: {e}"))
        })?;

        // Initialize reparameterization MLP if requested
        let (reparam_w1, reparam_b1, reparam_w2, reparam_b2) = if config.use_reparameterization {
            let reparam_hidden = config.reparam_hidden_dim;

            let normal_reparam =
                Normal::new(0.0, 1.0 / f64::sqrt(hidden_dim as f64)).map_err(|e| {
                    NeuralError::InvalidArchitecture(format!(
                        "Failed to create normal distribution: {e}"
                    ))
                })?;

            // W1: [hidden_dim, reparam_hidden]
            let w1_vec: Vec<F> = (0..(hidden_dim * reparam_hidden))
                .map(|_| F::from(normal_reparam.sample(rng)).unwrap_or(F::zero()))
                .collect();
            let w1 = Array::from_shape_vec(IxDyn(&[hidden_dim, reparam_hidden]), w1_vec).map_err(
                |e| NeuralError::InvalidArchitecture(format!("Failed to create reparam W1: {e}")),
            )?;
            let b1 = Array::zeros(IxDyn(&[reparam_hidden]));

            let normal_reparam2 = Normal::new(0.0, 1.0 / f64::sqrt(reparam_hidden as f64))
                .map_err(|e| {
                    NeuralError::InvalidArchitecture(format!(
                        "Failed to create normal distribution: {e}"
                    ))
                })?;

            // W2: [reparam_hidden, hidden_dim]
            let w2_vec: Vec<F> = (0..(reparam_hidden * hidden_dim))
                .map(|_| F::from(normal_reparam2.sample(rng)).unwrap_or(F::zero()))
                .collect();
            let w2 = Array::from_shape_vec(IxDyn(&[reparam_hidden, hidden_dim]), w2_vec).map_err(
                |e| NeuralError::InvalidArchitecture(format!("Failed to create reparam W2: {e}")),
            )?;
            let b2 = Array::zeros(IxDyn(&[hidden_dim]));

            (Some(w1), Some(b1), Some(w2), Some(b2))
        } else {
            (None, None, None, None)
        };

        let grad_prefix_embeddings = RwLock::new(Array::zeros(prefix_embeddings.raw_dim()));

        let grad_reparam_w1 = RwLock::new(reparam_w1.as_ref().map(|w| Array::zeros(w.raw_dim())));
        let grad_reparam_b1 = RwLock::new(reparam_b1.as_ref().map(|b| Array::zeros(b.raw_dim())));
        let grad_reparam_w2 = RwLock::new(reparam_w2.as_ref().map(|w| Array::zeros(w.raw_dim())));
        let grad_reparam_b2 = RwLock::new(reparam_b2.as_ref().map(|b| Array::zeros(b.raw_dim())));

        Ok(Self {
            config,
            prefix_embeddings,
            reparam_w1,
            reparam_b1,
            reparam_w2,
            reparam_b2,
            training: true,
            cached_prefix_output: RwLock::new(None),
            cached_input: RwLock::new(None),
            cached_mlp_hidden: RwLock::new(None),
            grad_prefix_embeddings,
            grad_reparam_w1,
            grad_reparam_b1,
            grad_reparam_w2,
            grad_reparam_b2,
        })
    }

    /// Get the prefix length
    pub fn prefix_length(&self) -> usize {
        self.config.prefix_length
    }

    /// Get the hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    /// Get the number of attention heads
    pub fn num_heads(&self) -> usize {
        self.config.num_heads
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// Get the number of trainable parameters
    pub fn trainable_parameter_count(&self) -> usize {
        let num_effective_layers = if self.config.independent_layers {
            self.config.num_layers
        } else {
            1
        };
        let embed_params =
            num_effective_layers * self.config.prefix_length * self.config.hidden_dim;

        if self.config.use_reparameterization {
            let reparam_params = self.config.hidden_dim * self.config.reparam_hidden_dim
                + self.config.reparam_hidden_dim
                + self.config.reparam_hidden_dim * self.config.hidden_dim
                + self.config.hidden_dim;
            embed_params + reparam_params
        } else {
            embed_params
        }
    }

    /// Get the prefix embeddings for a specific layer
    ///
    /// Returns the prefix key/value vectors for the given layer index.
    /// If layers share prefixes (independent_layers=false), layer_idx is ignored.
    pub fn get_prefix_for_layer(&self, layer_idx: usize) -> Result<Array<F, IxDyn>> {
        let effective_idx = if self.config.independent_layers {
            if layer_idx >= self.config.num_layers {
                return Err(NeuralError::InvalidArgument(format!(
                    "Layer index {} out of range (num_layers={})",
                    layer_idx, self.config.num_layers
                )));
            }
            layer_idx
        } else {
            0
        };

        let prefix_length = self.config.prefix_length;
        let hidden_dim = self.config.hidden_dim;

        // Extract the embedding for this layer: [prefix_length, hidden_dim]
        let mut embedding = Array::zeros(IxDyn(&[prefix_length, hidden_dim]));
        for p in 0..prefix_length {
            for d in 0..hidden_dim {
                embedding[[p, d]] = self.prefix_embeddings[[effective_idx, p, d]];
            }
        }

        // Apply reparameterization MLP if configured
        if self.config.use_reparameterization {
            self.apply_reparameterization(&embedding)
        } else {
            Ok(embedding)
        }
    }

    /// Apply the reparameterization MLP: prefix = W2 * tanh(W1 * embedding + b1) + b2
    fn apply_reparameterization(&self, embedding: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let w1 = self.reparam_w1.as_ref().ok_or_else(|| {
            NeuralError::InvalidState("Reparameterization W1 not initialized".to_string())
        })?;
        let b1 = self.reparam_b1.as_ref().ok_or_else(|| {
            NeuralError::InvalidState("Reparameterization B1 not initialized".to_string())
        })?;
        let w2 = self.reparam_w2.as_ref().ok_or_else(|| {
            NeuralError::InvalidState("Reparameterization W2 not initialized".to_string())
        })?;
        let b2 = self.reparam_b2.as_ref().ok_or_else(|| {
            NeuralError::InvalidState("Reparameterization B2 not initialized".to_string())
        })?;

        let prefix_length = self.config.prefix_length;
        let hidden_dim = self.config.hidden_dim;
        let reparam_hidden = self.config.reparam_hidden_dim;

        // hidden = tanh(embedding @ W1 + b1): [prefix_length, reparam_hidden]
        let mut hidden = Array::zeros(IxDyn(&[prefix_length, reparam_hidden]));
        for p in 0..prefix_length {
            for h in 0..reparam_hidden {
                let mut sum = F::zero();
                for d in 0..hidden_dim {
                    sum += embedding[[p, d]] * w1[[d, h]];
                }
                hidden[[p, h]] = (sum + b1[h]).tanh();
            }
        }

        // Cache MLP hidden for backward
        {
            let mut cache = self
                .cached_mlp_hidden
                .write()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            *cache = Some(hidden.clone());
        }

        // output = hidden @ W2 + b2: [prefix_length, hidden_dim]
        let mut output = Array::zeros(IxDyn(&[prefix_length, hidden_dim]));
        for p in 0..prefix_length {
            for d in 0..hidden_dim {
                let mut sum = F::zero();
                for h in 0..reparam_hidden {
                    sum += hidden[[p, h]] * w2[[h, d]];
                }
                output[[p, d]] = sum + b2[d];
            }
        }

        Ok(output)
    }

    /// Compute the forward pass
    ///
    /// The forward pass prepends prefix vectors to the input sequence.
    /// Input shape: [batch, seq_len, hidden_dim]
    /// Output shape: [batch, prefix_length + seq_len, hidden_dim]
    fn compute_forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let hidden_dim = self.config.hidden_dim;
        let prefix_length = self.config.prefix_length;

        // Get prefix vectors (using layer 0 by default in forward pass)
        let prefix = self.get_prefix_for_layer(0)?;

        // Cache prefix output
        {
            let mut cache = self
                .cached_prefix_output
                .write()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            *cache = Some(prefix.clone());
        }

        // Concatenate: [batch, prefix_length + seq_len, hidden_dim]
        let total_len = prefix_length + seq_len;
        let mut output = Array::zeros(IxDyn(&[batch_size, total_len, hidden_dim]));

        // Fill prefix (same for all batch elements)
        for b in 0..batch_size {
            for p in 0..prefix_length {
                for d in 0..hidden_dim {
                    output[[b, p, d]] = prefix[[p, d]];
                }
            }
        }

        // Fill original sequence
        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..hidden_dim {
                    output[[b, prefix_length + s, d]] = input[[b, s, d]];
                }
            }
        }

        Ok(output)
    }

    /// Get reference to prefix embeddings (raw, before reparameterization)
    pub fn prefix_embeddings(&self) -> &Array<F, IxDyn> {
        &self.prefix_embeddings
    }

    /// Check if reparameterization is enabled
    pub fn uses_reparameterization(&self) -> bool {
        self.config.use_reparameterization
    }

    /// Check if layers have independent prefixes
    pub fn has_independent_layers(&self) -> bool {
        self.config.independent_layers
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for PrefixTuningLayer<F>
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

        // Handle different input dimensionalities:
        // 2D: [batch, hidden_dim] -> treat as [batch, 1, hidden_dim]
        // 3D: [batch, seq_len, hidden_dim] -> use directly
        let input_3d = match input.ndim() {
            2 => {
                let batch = input.shape()[0];
                let dim = input.shape()[1];
                if dim != self.config.hidden_dim {
                    return Err(NeuralError::InvalidArgument(format!(
                        "Input dim mismatch: expected {}, got {}",
                        self.config.hidden_dim, dim
                    )));
                }
                input
                    .clone()
                    .into_shape_with_order(IxDyn(&[batch, 1, dim]))
                    .map_err(|e| {
                        NeuralError::InferenceError(format!("Failed to reshape input: {e}"))
                    })?
            }
            3 => {
                if input.shape()[2] != self.config.hidden_dim {
                    return Err(NeuralError::InvalidArgument(format!(
                        "Input hidden_dim mismatch: expected {}, got {}",
                        self.config.hidden_dim,
                        input.shape()[2]
                    )));
                }
                input.clone()
            }
            _ => {
                return Err(NeuralError::InvalidArgument(format!(
                    "Input must be 2D or 3D, got {}D",
                    input.ndim()
                )));
            }
        };

        self.compute_forward(&input_3d)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // grad_output shape: [batch, prefix_length + seq_len, hidden_dim]
        // The prefix portion's gradient updates the prefix embeddings
        // The sequence portion's gradient is passed through to the input

        let batch_size = grad_output.shape()[0];
        let total_len = grad_output.shape()[1];
        let hidden_dim = self.config.hidden_dim;
        let prefix_length = self.config.prefix_length;

        if total_len < prefix_length {
            return Err(NeuralError::ShapeMismatch(format!(
                "grad_output seq_len ({total_len}) < prefix_length ({prefix_length})"
            )));
        }

        let seq_len = total_len - prefix_length;

        // Accumulate gradient for prefix embeddings (sum over batch)
        {
            let mut grad_embed = self
                .grad_prefix_embeddings
                .write()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;

            // Zero out first
            grad_embed.fill(F::zero());

            for b in 0..batch_size {
                for p in 0..prefix_length {
                    for d in 0..hidden_dim {
                        grad_embed[[0, p, d]] += grad_output[[b, p, d]];
                    }
                }
            }
        }

        // Extract gradient for the input sequence: [batch, seq_len, hidden_dim]
        let mut grad_input = Array::zeros(IxDyn(&[batch_size, seq_len, hidden_dim]));
        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..hidden_dim {
                    grad_input[[b, s, d]] = grad_output[[b, prefix_length + s, d]];
                }
            }
        }

        Ok(grad_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update prefix embeddings
        {
            let grad = self
                .grad_prefix_embeddings
                .read()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            let shape = self.prefix_embeddings.shape().to_vec();
            for idx in 0..self.prefix_embeddings.len() {
                // Compute multi-dimensional index from flat index
                let mut remaining = idx;
                let mut indices = vec![0usize; shape.len()];
                for (dim_idx, &dim_size) in shape.iter().enumerate().rev() {
                    indices[dim_idx] = remaining % dim_size;
                    remaining /= dim_size;
                }
                let ix: Vec<usize> = indices;
                self.prefix_embeddings[ix.as_slice()] -= learning_rate * grad[ix.as_slice()];
            }
        }

        // Update reparameterization MLP weights if present
        if let Some(ref mut w1) = self.reparam_w1 {
            let grad = self
                .grad_reparam_w1
                .read()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            if let Some(ref g) = *grad {
                let len = w1.len();
                for idx in 0..len {
                    let flat_slice = w1.as_slice_mut().ok_or_else(|| {
                        NeuralError::InferenceError("W1 not contiguous".to_string())
                    })?;
                    let grad_slice = g.as_slice().ok_or_else(|| {
                        NeuralError::InferenceError("grad W1 not contiguous".to_string())
                    })?;
                    flat_slice[idx] -= learning_rate * grad_slice[idx];
                }
            }
        }

        if let Some(ref mut b1) = self.reparam_b1 {
            let grad = self
                .grad_reparam_b1
                .read()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            if let Some(ref g) = *grad {
                for i in 0..b1.len() {
                    b1[i] -= learning_rate * g[i];
                }
            }
        }

        if let Some(ref mut w2) = self.reparam_w2 {
            let grad = self
                .grad_reparam_w2
                .read()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            if let Some(ref g) = *grad {
                let len = w2.len();
                for idx in 0..len {
                    let flat_slice = w2.as_slice_mut().ok_or_else(|| {
                        NeuralError::InferenceError("W2 not contiguous".to_string())
                    })?;
                    let grad_slice = g.as_slice().ok_or_else(|| {
                        NeuralError::InferenceError("grad W2 not contiguous".to_string())
                    })?;
                    flat_slice[idx] -= learning_rate * grad_slice[idx];
                }
            }
        }

        if let Some(ref mut b2) = self.reparam_b2 {
            let grad = self
                .grad_reparam_b2
                .read()
                .map_err(|e| NeuralError::InferenceError(format!("Lock poisoned: {e}")))?;
            if let Some(ref g) = *grad {
                for i in 0..b2.len() {
                    b2[i] -= learning_rate * g[i];
                }
            }
        }

        Ok(())
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut p = vec![self.prefix_embeddings.clone()];
        if let Some(ref w1) = self.reparam_w1 {
            p.push(w1.clone());
        }
        if let Some(ref b1) = self.reparam_b1 {
            p.push(b1.clone());
        }
        if let Some(ref w2) = self.reparam_w2 {
            p.push(w2.clone());
        }
        if let Some(ref b2) = self.reparam_b2 {
            p.push(b2.clone());
        }
        p
    }

    fn gradients(&self) -> Vec<Array<F, IxDyn>> {
        let ge = self
            .grad_prefix_embeddings
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| Array::zeros(self.prefix_embeddings.raw_dim()));
        vec![ge]
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
        "PrefixTuning"
    }

    fn parameter_count(&self) -> usize {
        self.trainable_parameter_count()
    }

    fn layer_description(&self) -> String {
        format!(
            "PrefixTuning(prefix_len={}, hidden={}, heads={}, layers={}, reparam={}, independent={})",
            self.config.prefix_length,
            self.config.hidden_dim,
            self.config.num_heads,
            self.config.num_layers,
            self.config.use_reparameterization,
            self.config.independent_layers
        )
    }

    fn inputshape(&self) -> Option<Vec<usize>> {
        Some(vec![self.config.hidden_dim])
    }

    fn outputshape(&self) -> Option<Vec<usize>> {
        Some(vec![self.config.hidden_dim])
    }

    fn name(&self) -> Option<&str> {
        Some("PrefixTuning")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::rng;

    #[test]
    fn test_prefix_construction_default() {
        let mut r = rng();
        let config = PrefixTuningConfig::default();
        let layer = PrefixTuningLayer::<f64>::new(config, &mut r).unwrap();
        assert_eq!(layer.prefix_length(), 10);
        assert_eq!(layer.hidden_dim(), 64);
        assert_eq!(layer.num_heads(), 4);
        assert!(layer.uses_reparameterization());
    }

    #[test]
    fn test_prefix_construction_custom() {
        let mut r = rng();
        let config = PrefixTuningConfig {
            prefix_length: 20,
            hidden_dim: 128,
            num_heads: 8,
            num_layers: 6,
            use_reparameterization: false,
            reparam_hidden_dim: 256,
            independent_layers: true,
            init_std: 0.01,
        };
        let layer = PrefixTuningLayer::<f64>::new(config, &mut r).unwrap();
        assert_eq!(layer.prefix_length(), 20);
        assert_eq!(layer.hidden_dim(), 128);
        assert_eq!(layer.num_layers(), 6);
        assert!(layer.has_independent_layers());
        assert!(!layer.uses_reparameterization());
    }

    #[test]
    fn test_prefix_construction_invalid() {
        let mut r = rng();
        let config = PrefixTuningConfig {
            prefix_length: 0,
            ..Default::default()
        };
        assert!(PrefixTuningLayer::<f64>::new(config, &mut r).is_err());

        let config2 = PrefixTuningConfig {
            hidden_dim: 0,
            ..Default::default()
        };
        assert!(PrefixTuningLayer::<f64>::new(config2, &mut r).is_err());

        let config3 = PrefixTuningConfig {
            hidden_dim: 65,
            num_heads: 4, // 65 % 4 != 0
            ..Default::default()
        };
        assert!(PrefixTuningLayer::<f64>::new(config3, &mut r).is_err());
    }

    #[test]
    fn test_prefix_forward_shape_3d() {
        let mut r = rng();
        let config = PrefixTuningConfig {
            prefix_length: 5,
            hidden_dim: 16,
            num_heads: 4,
            use_reparameterization: false,
            ..Default::default()
        };
        let layer = PrefixTuningLayer::<f64>::new(config, &mut r).unwrap();

        // 3D input: [batch=2, seq_len=10, hidden=16]
        let input = Array::zeros(IxDyn(&[2, 10, 16]));
        let output = layer.forward(&input).unwrap();
        // Output should be [2, 15, 16] (prefix_length=5 + seq_len=10)
        assert_eq!(output.shape(), &[2, 15, 16]);
    }

    #[test]
    fn test_prefix_forward_shape_2d() {
        let mut r = rng();
        let config = PrefixTuningConfig {
            prefix_length: 3,
            hidden_dim: 8,
            num_heads: 2,
            use_reparameterization: false,
            ..Default::default()
        };
        let layer = PrefixTuningLayer::<f64>::new(config, &mut r).unwrap();

        // 2D input: [batch=4, hidden=8] -> treated as [4, 1, 8]
        let input = Array::zeros(IxDyn(&[4, 8]));
        let output = layer.forward(&input).unwrap();
        // Output should be [4, 4, 8] (prefix_length=3 + seq_len=1)
        assert_eq!(output.shape(), &[4, 4, 8]);
    }

    #[test]
    fn test_prefix_forward_preserves_input() {
        let mut r = rng();
        let config = PrefixTuningConfig {
            prefix_length: 2,
            hidden_dim: 4,
            num_heads: 2,
            use_reparameterization: false,
            ..Default::default()
        };
        let layer = PrefixTuningLayer::<f64>::new(config, &mut r).unwrap();

        let input = Array::from_shape_vec(
            IxDyn(&[1, 3, 4]),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let output = layer.forward(&input).unwrap();

        // The last 3 positions (seq_len=3) should match the input exactly
        for s in 0..3 {
            for d in 0..4 {
                assert!(
                    (output[[0, 2 + s, d]] - input[[0, s, d]]).abs() < 1e-12,
                    "Input not preserved at seq={s}, dim={d}"
                );
            }
        }
    }

    #[test]
    fn test_prefix_with_reparameterization() {
        let mut r = rng();
        let config = PrefixTuningConfig {
            prefix_length: 4,
            hidden_dim: 8,
            num_heads: 2,
            use_reparameterization: true,
            reparam_hidden_dim: 16,
            ..Default::default()
        };
        let layer = PrefixTuningLayer::<f64>::new(config, &mut r).unwrap();

        let input = Array::zeros(IxDyn(&[2, 6, 8]));
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 10, 8]); // 4 + 6 = 10
    }

    #[test]
    fn test_prefix_independent_layers() {
        let mut r = rng();
        let config = PrefixTuningConfig {
            prefix_length: 3,
            hidden_dim: 8,
            num_heads: 2,
            num_layers: 4,
            independent_layers: true,
            use_reparameterization: false,
            ..Default::default()
        };
        let layer = PrefixTuningLayer::<f64>::new(config, &mut r).unwrap();

        // Each layer should get different prefix vectors
        let prefix_0 = layer.get_prefix_for_layer(0).unwrap();
        let prefix_1 = layer.get_prefix_for_layer(1).unwrap();
        let prefix_3 = layer.get_prefix_for_layer(3).unwrap();

        assert_eq!(prefix_0.shape(), &[3, 8]);
        assert_eq!(prefix_1.shape(), &[3, 8]);
        assert_eq!(prefix_3.shape(), &[3, 8]);

        // They should generally be different (random init)
        let mut all_same = true;
        for p in 0..3 {
            for d in 0..8 {
                if (prefix_0[[p, d]] - prefix_1[[p, d]]).abs() > 1e-12 {
                    all_same = false;
                    break;
                }
            }
        }
        assert!(!all_same, "Independent layer prefixes should differ");

        // Out-of-range should error
        assert!(layer.get_prefix_for_layer(4).is_err());
    }

    #[test]
    fn test_prefix_shared_layers() {
        let mut r = rng();
        let config = PrefixTuningConfig {
            prefix_length: 3,
            hidden_dim: 8,
            num_heads: 2,
            num_layers: 4,
            independent_layers: false,
            use_reparameterization: false,
            ..Default::default()
        };
        let layer = PrefixTuningLayer::<f64>::new(config, &mut r).unwrap();

        // All layers should get the same prefix (shared)
        let prefix_0 = layer.get_prefix_for_layer(0).unwrap();
        let prefix_3 = layer.get_prefix_for_layer(3).unwrap();

        for p in 0..3 {
            for d in 0..8 {
                assert!(
                    (prefix_0[[p, d]] - prefix_3[[p, d]]).abs() < 1e-12,
                    "Shared prefixes should be identical"
                );
            }
        }
    }

    #[test]
    fn test_prefix_backward_shape() {
        let mut r = rng();
        let config = PrefixTuningConfig {
            prefix_length: 3,
            hidden_dim: 8,
            num_heads: 2,
            use_reparameterization: false,
            ..Default::default()
        };
        let layer = PrefixTuningLayer::<f64>::new(config, &mut r).unwrap();

        let input = Array::zeros(IxDyn(&[2, 5, 8]));
        let _output = layer.forward(&input).unwrap();

        // grad_output: [2, 8, 8] (prefix_length=3 + seq_len=5)
        let grad_output = Array::from_elem(IxDyn(&[2, 8, 8]), 1.0);
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        // Should match original input shape (without prefix)
        assert_eq!(grad_input.shape(), &[2, 5, 8]);
    }

    #[test]
    fn test_prefix_parameter_count() {
        let mut r = rng();

        // Without reparameterization
        let config1 = PrefixTuningConfig {
            prefix_length: 10,
            hidden_dim: 64,
            num_heads: 4,
            use_reparameterization: false,
            ..Default::default()
        };
        let layer1 = PrefixTuningLayer::<f64>::new(config1, &mut r).unwrap();
        assert_eq!(layer1.trainable_parameter_count(), 10 * 64); // 640

        // With reparameterization
        let config2 = PrefixTuningConfig {
            prefix_length: 10,
            hidden_dim: 64,
            num_heads: 4,
            use_reparameterization: true,
            reparam_hidden_dim: 128,
            ..Default::default()
        };
        let layer2 = PrefixTuningLayer::<f64>::new(config2, &mut r).unwrap();
        // embed: 640, w1: 64*128=8192, b1: 128, w2: 128*64=8192, b2: 64
        assert_eq!(
            layer2.trainable_parameter_count(),
            640 + 8192 + 128 + 8192 + 64
        );
    }

    #[test]
    fn test_prefix_layer_trait() {
        let mut r = rng();
        let config = PrefixTuningConfig::default();
        let mut layer = PrefixTuningLayer::<f64>::new(config, &mut r).unwrap();
        assert_eq!(layer.layer_type(), "PrefixTuning");
        assert!(layer.is_training());
        layer.set_training(false);
        assert!(!layer.is_training());
        assert_eq!(layer.name(), Some("PrefixTuning"));
    }
}
