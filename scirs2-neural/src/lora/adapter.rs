//! Bottleneck adapter layer implementation.
//!
//! Adapter layers insert small trainable bottleneck modules into a frozen model:
//! output = residual + up_project(activation(down_project(input)))
//!
//! This provides an alternative to LoRA for parameter-efficient fine-tuning.

use scirs2_core::ndarray::{Array1, Array2, Axis};

use super::types::{AdapterActivation, AdapterConfig, LoRAStats};
use crate::{NeuralError, Result};

/// A simple xorshift64 PRNG for reproducible initialization.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_1234
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_uniform(&mut self, bound: f64) -> f64 {
        (self.next_f64() * 2.0 - 1.0) * bound
    }
}

/// Compute Xavier uniform bound: sqrt(6 / (fan_in + fan_out)).
fn xavier_uniform_bound(fan_in: usize, fan_out: usize) -> f64 {
    (6.0 / (fan_in + fan_out) as f64).sqrt()
}

/// Bottleneck adapter layer for parameter-efficient fine-tuning.
///
/// Inserts a small trainable module: input -> down_project -> activation -> up_project,
/// with an optional residual connection that adds the adapter output to the original input.
///
/// # Architecture
///
/// ```text
///                           ┌─────────────────────────────┐
///   input ─────────────────>│        residual (optional)    │───> output
///     │                     └─────────────────────────────┘     │
///     │   ┌────────────┐   ┌────────────┐   ┌──────────┐      │
///     └──>│down_project│──>│ activation │──>│up_project│──────>┘
///         │(bottleneck) │   │ (ReLU etc) │   │(input_dim)│
///         └────────────┘   └────────────┘   └──────────┘
/// ```
///
/// # Example
///
/// ```
/// use scirs2_neural::lora::{BottleneckAdapter, AdapterConfig};
/// use scirs2_core::ndarray::Array2;
///
/// let config = AdapterConfig { bottleneck_dim: 16, ..Default::default() };
/// let adapter = BottleneckAdapter::new(64, &config).expect("creation failed");
///
/// let input = Array2::<f64>::ones((2, 64));
/// let output = adapter.forward(&input).expect("forward failed");
/// assert_eq!(output.shape(), &[2, 64]);
/// ```
pub struct BottleneckAdapter {
    /// Down-projection weight [bottleneck_dim x input_dim].
    down_project: Array2<f64>,
    /// Up-projection weight [input_dim x bottleneck_dim].
    up_project: Array2<f64>,
    /// Bias for down-projection [bottleneck_dim].
    bias_down: Array1<f64>,
    /// Bias for up-projection [input_dim].
    bias_up: Array1<f64>,
    /// Activation function.
    activation: AdapterActivation,
    /// Whether to add a residual connection.
    residual: bool,
    /// Input dimension (for stats).
    input_dim: usize,
    /// Bottleneck dimension (for stats).
    bottleneck_dim: usize,
}

impl BottleneckAdapter {
    /// Create a new bottleneck adapter.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input/output feature dimension
    /// * `config` - Adapter configuration
    ///
    /// # Errors
    ///
    /// Returns an error if bottleneck_dim exceeds input_dim or config is invalid.
    pub fn new(input_dim: usize, config: &AdapterConfig) -> Result<Self> {
        config.validate()?;

        if input_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "Adapter input_dim must be > 0".to_string(),
            ));
        }

        let bottleneck_dim = config.bottleneck_dim;
        let mut rng = Xorshift64::new(config.seed);

        // Xavier initialization for down and up projections
        let down_bound = xavier_uniform_bound(input_dim, bottleneck_dim);
        let down_project = Array2::from_shape_fn((bottleneck_dim, input_dim), |_| {
            rng.next_uniform(down_bound)
        });

        let up_bound = xavier_uniform_bound(bottleneck_dim, input_dim);
        let up_project =
            Array2::from_shape_fn((input_dim, bottleneck_dim), |_| rng.next_uniform(up_bound));

        // Zero-initialize biases (standard practice for adapters)
        let bias_down = Array1::zeros(bottleneck_dim);
        let bias_up = Array1::zeros(input_dim);

        Ok(Self {
            down_project,
            up_project,
            bias_down,
            bias_up,
            activation: config.activation,
            residual: config.residual,
            input_dim,
            bottleneck_dim,
        })
    }

    /// Forward pass through the adapter.
    ///
    /// Computes: output = input + up_project(activation(down_project(input) + bias_down)) + bias_up
    /// (or without the residual if `residual=false`)
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [batch_size x input_dim]
    ///
    /// # Errors
    ///
    /// Returns an error on dimension mismatch.
    pub fn forward(&self, input: &Array2<f64>) -> Result<Array2<f64>> {
        if input.ncols() != self.input_dim {
            return Err(NeuralError::DimensionMismatch(format!(
                "Input has {} features but adapter expects {}",
                input.ncols(),
                self.input_dim
            )));
        }

        // Down-project: [batch x input_dim] @ [input_dim x bottleneck_dim]
        let hidden = input.dot(&self.down_project.t());

        // Add bias_down: broadcast [bottleneck_dim] across batch
        let hidden = &hidden + &self.bias_down;

        // Apply activation element-wise
        let activation = self.activation;
        let hidden = hidden.mapv(|x| activation.apply(x));

        // Up-project: [batch x bottleneck_dim] @ [bottleneck_dim x input_dim]
        let adapter_output = hidden.dot(&self.up_project.t());

        // Add bias_up
        let adapter_output = &adapter_output + &self.bias_up;

        // Residual connection
        if self.residual {
            Ok(input + &adapter_output)
        } else {
            Ok(adapter_output)
        }
    }

    /// Compute parameter statistics.
    pub fn stats(&self) -> LoRAStats {
        let trainable_params = self.bottleneck_dim * self.input_dim  // down_project
            + self.input_dim * self.bottleneck_dim                    // up_project
            + self.bottleneck_dim                                     // bias_down
            + self.input_dim; // bias_up
        let frozen_params = 0; // Adapter has no frozen params
        let total_params = trainable_params;
        let compression_ratio = 1.0; // All params are trainable

        LoRAStats {
            total_params,
            trainable_params,
            frozen_params,
            compression_ratio,
        }
    }

    /// Get a reference to the down-projection weight.
    pub fn down_project(&self) -> &Array2<f64> {
        &self.down_project
    }

    /// Get a reference to the up-projection weight.
    pub fn up_project(&self) -> &Array2<f64> {
        &self.up_project
    }

    /// Get the activation function.
    pub fn activation(&self) -> AdapterActivation {
        self.activation
    }

    /// Whether residual connection is enabled.
    pub fn has_residual(&self) -> bool {
        self.residual
    }

    /// Get input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get bottleneck dimension.
    pub fn bottleneck_dim(&self) -> usize {
        self.bottleneck_dim
    }

    /// Set the down-projection weight.
    ///
    /// # Errors
    ///
    /// Returns an error if shape doesn't match.
    pub fn set_down_project(&mut self, w: Array2<f64>) -> Result<()> {
        if w.shape() != self.down_project.shape() {
            return Err(NeuralError::ShapeMismatch(format!(
                "Expected down_project shape {:?}, got {:?}",
                self.down_project.shape(),
                w.shape()
            )));
        }
        self.down_project = w;
        Ok(())
    }

    /// Set the up-projection weight.
    ///
    /// # Errors
    ///
    /// Returns an error if shape doesn't match.
    pub fn set_up_project(&mut self, w: Array2<f64>) -> Result<()> {
        if w.shape() != self.up_project.shape() {
            return Err(NeuralError::ShapeMismatch(format!(
                "Expected up_project shape {:?}, got {:?}",
                self.up_project.shape(),
                w.shape()
            )));
        }
        self.up_project = w;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_adapter_creation() {
        let config = AdapterConfig {
            bottleneck_dim: 16,
            ..Default::default()
        };
        let adapter = BottleneckAdapter::new(64, &config);
        assert!(adapter.is_ok());
        let adapter = adapter.expect("creation should succeed");
        assert_eq!(adapter.input_dim(), 64);
        assert_eq!(adapter.bottleneck_dim(), 16);
    }

    #[test]
    fn test_adapter_forward_shape() {
        let config = AdapterConfig {
            bottleneck_dim: 8,
            ..Default::default()
        };
        let adapter = BottleneckAdapter::new(32, &config).expect("creation failed");

        let input = Array2::<f64>::ones((4, 32));
        let output = adapter.forward(&input).expect("forward failed");
        assert_eq!(output.shape(), &[4, 32]);
    }

    #[test]
    fn test_adapter_residual_preserves_input_with_zero_weights() {
        let config = AdapterConfig {
            bottleneck_dim: 8,
            residual: true,
            ..Default::default()
        };
        let mut adapter = BottleneckAdapter::new(16, &config).expect("creation failed");

        // Zero out all weights so adapter output is zero
        adapter
            .set_down_project(Array2::zeros((8, 16)))
            .expect("set failed");
        adapter
            .set_up_project(Array2::zeros((16, 8)))
            .expect("set failed");

        let input = Array2::from_shape_fn((2, 16), |(i, j)| (i * 16 + j) as f64 * 0.1);
        let output = adapter.forward(&input).expect("forward failed");

        // With zero adapter and residual=true, output should equal input
        for (a, b) in output.iter().zip(input.iter()) {
            assert!((a - b).abs() < 1e-10, "residual not preserved: {a} vs {b}");
        }
    }

    #[test]
    fn test_adapter_no_residual() {
        let config = AdapterConfig {
            bottleneck_dim: 8,
            residual: false,
            ..Default::default()
        };
        let mut adapter = BottleneckAdapter::new(16, &config).expect("creation failed");

        // Zero out all weights
        adapter
            .set_down_project(Array2::zeros((8, 16)))
            .expect("set failed");
        adapter
            .set_up_project(Array2::zeros((16, 8)))
            .expect("set failed");

        let input = Array2::from_shape_fn((2, 16), |(i, j)| (i * 16 + j) as f64 * 0.1);
        let output = adapter.forward(&input).expect("forward failed");

        // Without residual and zero weights, output should be zero
        for val in output.iter() {
            assert!(val.abs() < 1e-10, "expected zero output, got {val}");
        }
    }

    #[test]
    fn test_adapter_bottleneck_reduces_params() {
        let config_large = AdapterConfig {
            bottleneck_dim: 64,
            ..Default::default()
        };
        let config_small = AdapterConfig {
            bottleneck_dim: 8,
            ..Default::default()
        };
        let adapter_large = BottleneckAdapter::new(256, &config_large).expect("creation failed");
        let adapter_small = BottleneckAdapter::new(256, &config_small).expect("creation failed");

        assert!(adapter_small.stats().trainable_params < adapter_large.stats().trainable_params);
    }

    #[test]
    fn test_adapter_different_activations() {
        let activations = [
            AdapterActivation::ReLU,
            AdapterActivation::GELU,
            AdapterActivation::SiLU,
            AdapterActivation::Tanh,
        ];

        for act in &activations {
            let config = AdapterConfig {
                bottleneck_dim: 4,
                activation: *act,
                ..Default::default()
            };
            let adapter = BottleneckAdapter::new(8, &config).expect("creation failed");
            let input = Array2::<f64>::ones((2, 8));
            let output = adapter.forward(&input).expect("forward failed");
            assert_eq!(output.shape(), &[2, 8], "shape wrong for {act}");
        }
    }

    #[test]
    fn test_adapter_dimension_mismatch() {
        let config = AdapterConfig {
            bottleneck_dim: 4,
            ..Default::default()
        };
        let adapter = BottleneckAdapter::new(8, &config).expect("creation failed");

        let bad_input = Array2::<f64>::ones((2, 10)); // wrong features
        assert!(adapter.forward(&bad_input).is_err());
    }

    #[test]
    fn test_adapter_stats() {
        let config = AdapterConfig {
            bottleneck_dim: 8,
            ..Default::default()
        };
        let adapter = BottleneckAdapter::new(32, &config).expect("creation failed");
        let stats = adapter.stats();

        // down: 8*32=256, up: 32*8=256, bias_down: 8, bias_up: 32 => 552
        assert_eq!(stats.trainable_params, 256 + 256 + 8 + 32);
        assert_eq!(stats.frozen_params, 0);
    }
}
