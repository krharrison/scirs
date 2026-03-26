//! Types and configuration for LoRA (Low-Rank Adaptation) and Adapter layers.
//!
//! This module provides the configuration types for parameter-efficient fine-tuning
//! techniques including LoRA and bottleneck adapters.

use std::fmt;

/// Specifies which attention/FFN components to apply LoRA to.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LoRATarget {
    /// Apply LoRA to query projections
    Query,
    /// Apply LoRA to key projections
    Key,
    /// Apply LoRA to value projections
    Value,
    /// Apply LoRA to output projections
    Output,
    /// Apply LoRA to feed-forward layers
    FeedForward,
    /// Apply LoRA to all supported targets
    All,
}

impl fmt::Display for LoRATarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[allow(unreachable_patterns)]
        match self {
            LoRATarget::Query => write!(f, "Query"),
            LoRATarget::Key => write!(f, "Key"),
            LoRATarget::Value => write!(f, "Value"),
            LoRATarget::Output => write!(f, "Output"),
            LoRATarget::FeedForward => write!(f, "FeedForward"),
            LoRATarget::All => write!(f, "All"),
            _ => write!(f, "Unknown"),
        }
    }
}

/// Configuration for LoRA (Low-Rank Adaptation).
///
/// LoRA decomposes weight updates into low-rank matrices A and B,
/// so that W_new = W_frozen + (alpha/rank) * B @ A.
/// This dramatically reduces the number of trainable parameters.
///
/// # Example
///
/// ```
/// use scirs2_neural::lora::LoRAConfig;
///
/// let config = LoRAConfig {
///     rank: 16,
///     alpha: 32.0,
///     ..Default::default()
/// };
/// assert_eq!(config.rank, 16);
/// ```
#[derive(Debug, Clone)]
pub struct LoRAConfig {
    /// Low-rank dimension r (typically 4-64).
    /// Smaller rank = fewer parameters but less expressive.
    pub rank: usize,
    /// Scaling factor alpha (typically rank or 2*rank).
    /// The effective scaling is alpha/rank.
    pub alpha: f64,
    /// Dropout probability for LoRA layers (0.0 = no dropout).
    pub dropout: f64,
    /// Which model components to apply LoRA to.
    pub targets: Vec<LoRATarget>,
    /// Random seed for reproducible initialization.
    pub seed: u64,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            targets: vec![LoRATarget::Query, LoRATarget::Value],
            seed: 42,
        }
    }
}

impl LoRAConfig {
    /// Compute the scaling factor alpha/rank.
    pub fn scaling(&self) -> f64 {
        if self.rank == 0 {
            0.0
        } else {
            self.alpha / self.rank as f64
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> crate::Result<()> {
        if self.rank == 0 {
            return Err(crate::NeuralError::InvalidArgument(
                "LoRA rank must be > 0".to_string(),
            ));
        }
        if self.alpha <= 0.0 {
            return Err(crate::NeuralError::InvalidArgument(
                "LoRA alpha must be > 0".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(crate::NeuralError::InvalidArgument(
                "LoRA dropout must be in [0.0, 1.0)".to_string(),
            ));
        }
        if self.targets.is_empty() {
            return Err(crate::NeuralError::InvalidArgument(
                "LoRA targets must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

/// Activation function for bottleneck adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdapterActivation {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Gaussian Error Linear Unit: x * Phi(x)
    GELU,
    /// Sigmoid Linear Unit (Swish): x * sigmoid(x)
    SiLU,
    /// Hyperbolic tangent
    Tanh,
}

impl AdapterActivation {
    /// Apply the activation function element-wise to a value.
    pub fn apply(&self, x: f64) -> f64 {
        #[allow(unreachable_patterns)]
        match self {
            AdapterActivation::ReLU => x.max(0.0),
            AdapterActivation::GELU => {
                // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let inner = (2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            }
            AdapterActivation::SiLU => x * (1.0 / (1.0 + (-x).exp())),
            AdapterActivation::Tanh => x.tanh(),
            _ => x, // fallback for future variants
        }
    }
}

impl fmt::Display for AdapterActivation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[allow(unreachable_patterns)]
        match self {
            AdapterActivation::ReLU => write!(f, "ReLU"),
            AdapterActivation::GELU => write!(f, "GELU"),
            AdapterActivation::SiLU => write!(f, "SiLU"),
            AdapterActivation::Tanh => write!(f, "Tanh"),
            _ => write!(f, "Unknown"),
        }
    }
}

/// Configuration for bottleneck adapter layers.
///
/// Adapters insert small bottleneck modules: down_project -> activation -> up_project,
/// with an optional residual connection.
///
/// # Example
///
/// ```
/// use scirs2_neural::lora::{AdapterConfig, AdapterActivation};
///
/// let config = AdapterConfig {
///     bottleneck_dim: 64,
///     activation: AdapterActivation::GELU,
///     ..Default::default()
/// };
/// assert_eq!(config.bottleneck_dim, 64);
/// ```
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    /// Down-projection dimension (bottleneck size).
    /// Smaller = fewer parameters.
    pub bottleneck_dim: usize,
    /// Activation function between down and up projections.
    pub activation: AdapterActivation,
    /// Whether to add a residual connection (input + adapter_output).
    pub residual: bool,
    /// Random seed for reproducible initialization.
    pub seed: u64,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            bottleneck_dim: 64,
            activation: AdapterActivation::ReLU,
            residual: true,
            seed: 42,
        }
    }
}

impl AdapterConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> crate::Result<()> {
        if self.bottleneck_dim == 0 {
            return Err(crate::NeuralError::InvalidArgument(
                "Adapter bottleneck_dim must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Statistics about LoRA/adapter parameter counts.
#[derive(Debug, Clone)]
pub struct LoRAStats {
    /// Total number of parameters (frozen + trainable).
    pub total_params: usize,
    /// Number of trainable (LoRA/adapter) parameters.
    pub trainable_params: usize,
    /// Number of frozen (original) parameters.
    pub frozen_params: usize,
    /// Compression ratio: trainable / total.
    pub compression_ratio: f64,
}

impl fmt::Display for LoRAStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LoRAStats {{ total: {}, trainable: {}, frozen: {}, ratio: {:.4} }}",
            self.total_params, self.trainable_params, self.frozen_params, self.compression_ratio
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_default() {
        let config = LoRAConfig::default();
        assert_eq!(config.rank, 8);
        assert!((config.alpha - 16.0).abs() < f64::EPSILON);
        assert!((config.dropout).abs() < f64::EPSILON);
        assert_eq!(config.targets.len(), 2);
    }

    #[test]
    fn test_lora_config_scaling() {
        let config = LoRAConfig {
            rank: 4,
            alpha: 8.0,
            ..Default::default()
        };
        assert!((config.scaling() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lora_config_validate() {
        let mut config = LoRAConfig::default();
        assert!(config.validate().is_ok());

        config.rank = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_adapter_config_default() {
        let config = AdapterConfig::default();
        assert_eq!(config.bottleneck_dim, 64);
        assert_eq!(config.activation, AdapterActivation::ReLU);
        assert!(config.residual);
    }

    #[test]
    fn test_adapter_activation_apply() {
        let relu = AdapterActivation::ReLU;
        assert!((relu.apply(1.0) - 1.0).abs() < f64::EPSILON);
        assert!((relu.apply(-1.0)).abs() < f64::EPSILON);

        let tanh = AdapterActivation::Tanh;
        assert!((tanh.apply(0.0)).abs() < f64::EPSILON);

        let silu = AdapterActivation::SiLU;
        assert!((silu.apply(0.0)).abs() < f64::EPSILON);

        let gelu = AdapterActivation::GELU;
        assert!((gelu.apply(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_lora_stats_display() {
        let stats = LoRAStats {
            total_params: 1000,
            trainable_params: 100,
            frozen_params: 900,
            compression_ratio: 0.1,
        };
        let s = format!("{stats}");
        assert!(s.contains("1000"));
        assert!(s.contains("100"));
    }

    #[test]
    fn test_lora_target_display() {
        assert_eq!(format!("{}", LoRATarget::Query), "Query");
        assert_eq!(format!("{}", LoRATarget::All), "All");
    }
}
