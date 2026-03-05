//! Module for model serialization and deserialization
//!
//! This module provides comprehensive serialization support for all neural network
//! architectures in scirs2-neural, including:
//!
//! - **Generic traits**: `ModelSerialize`, `ModelDeserialize`, `ExtractParameters`
//! - **SafeTensors format**: HuggingFace-compatible binary format (safe, no pickle)
//! - **Architecture-specific serialization**: ResNet, BERT, GPT, Mamba, EfficientNet, MobileNet
//! - **Legacy support**: JSON for Sequential models (via `legacy_serialization` feature)
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_neural::serialization::{ModelSerialize, ModelDeserialize, ModelFormat};
//!
//! // ModelFormat enumerates the supported serialization formats
//! let format = ModelFormat::SafeTensors;
//! assert_eq!(format, ModelFormat::SafeTensors);
//! ```

// Sub-modules
pub mod architecture;
pub mod model_serializer;
pub mod safetensors;
pub mod traits;

// Re-export key types from sub-modules
pub use architecture::{
    detect_architecture, detect_architecture_from_bytes, ArchitectureConfig,
    SerializableBertConfig, SerializableGPTConfig, SerializableMambaConfig,
    SerializableMobileNetConfig, SerializableResNetConfig,
};
pub use model_serializer::{
    load_bert, load_resnet, named_parameters_to_map, save_bert, save_resnet, ModelSerializer,
};
pub use safetensors::{
    read_named_parameters, validate_safetensors_file, write_named_parameters, SafeTensorsDtype,
    SafeTensorsHeaderEntry, SafeTensorsReader, SafeTensorsWriter,
};
pub use traits::{
    ExtractParameters, ModelDeserialize, ModelFormat, ModelMetadata, ModelSerialize,
    NamedParameters, TensorInfo,
};

// Legacy imports for existing serialization code
use crate::activations::*;
use crate::error::{NeuralError, Result};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

// Imports needed by legacy feature
#[cfg(feature = "legacy_serialization")]
use crate::layers::conv::PaddingMode;
#[cfg(feature = "legacy_serialization")]
use crate::layers::*;
#[cfg(feature = "legacy_serialization")]
use crate::models::sequential::Sequential;
#[cfg(feature = "legacy_serialization")]
use scirs2_core::ndarray::{Array, ScalarOperand};
#[cfg(feature = "legacy_serialization")]
use scirs2_core::numeric::{FromPrimitive, NumAssign, ToPrimitive};
#[cfg(feature = "legacy_serialization")]
use scirs2_core::random::SeedableRng;
#[cfg(feature = "legacy_serialization")]
use std::fmt::Display;
#[cfg(feature = "legacy_serialization")]
use std::fs;
#[cfg(feature = "legacy_serialization")]
use std::path::Path;

/// Model serialization format (legacy)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    /// JSON serialization format
    JSON,
    /// CBOR serialization format (serialized as JSON in legacy mode)
    CBOR,
    /// MessagePack serialization format (serialized as JSON in legacy mode)
    MessagePack,
}

/// Layer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    /// Dense (fully connected) layer
    Dense,
    /// Convolutional 2D layer
    Conv2D,
    /// Layer normalization
    LayerNorm,
    /// Batch normalization
    BatchNorm,
    /// Dropout layer
    Dropout,
    /// Max pooling 2D layer
    MaxPool2D,
}

/// Layer configuration for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LayerConfig {
    /// Dense layer configuration
    #[serde(rename = "Dense")]
    Dense(DenseConfig),
    /// Conv2D layer configuration
    #[serde(rename = "Conv2D")]
    Conv2D(Conv2DConfig),
    /// LayerNorm layer configuration
    #[serde(rename = "LayerNorm")]
    LayerNorm(LayerNormConfig),
    /// BatchNorm layer configuration
    #[serde(rename = "BatchNorm")]
    BatchNorm(BatchNormConfig),
    /// Dropout layer configuration
    #[serde(rename = "Dropout")]
    Dropout(DropoutConfig),
    /// MaxPool2D layer configuration
    #[serde(rename = "MaxPool2D")]
    MaxPool2D(MaxPool2DConfig),
}

/// Dense layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Activation function name
    pub activation: Option<String>,
}

/// Conv2D layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv2DConfig {
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output channels
    pub out_channels: usize,
    /// Kernel size (square)
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding mode
    pub padding_mode: String,
}

/// LayerNorm layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNormConfig {
    /// Normalized shape
    pub normalizedshape: usize,
    /// Epsilon for numerical stability
    pub eps: f64,
}

/// BatchNorm layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNormConfig {
    /// Number of features
    pub num_features: usize,
    /// Momentum
    pub momentum: f64,
    /// Epsilon for numerical stability
    pub eps: f64,
}

/// Dropout layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropoutConfig {
    /// Dropout probability
    pub p: f64,
}

/// MaxPool2D layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxPool2DConfig {
    /// Kernel size
    pub kernel_size: (usize, usize),
    /// Stride
    pub stride: (usize, usize),
    /// Padding
    pub padding: Option<(usize, usize)>,
}

/// Serialized model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedModel {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model layers configuration
    pub layers: Vec<LayerConfig>,
    /// Model parameters (weights and biases)
    pub parameters: Vec<Vec<Vec<f64>>>,
}

// =============================================================================
// Legacy serialization functions (available under legacy_serialization feature)
// =============================================================================

/// Save model to file
///
/// Legacy function for saving `Sequential` models to JSON.
/// For new code, prefer the `SafeTensors`-based API via `ModelSerialize::save()`.
#[cfg(feature = "legacy_serialization")]
#[allow(dead_code)]
pub fn save_model<
    F: Float + Debug + Display + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
    P: AsRef<Path>,
>(
    model: &Sequential<F>,
    path: P,
    _format: SerializationFormat,
) -> Result<()> {
    let serialized = serialize_model(model)?;
    let bytes = serde_json::to_vec_pretty(&serialized)
        .map_err(|e| NeuralError::SerializationError(e.to_string()))?;
    fs::write(path, bytes).map_err(|e| NeuralError::IOError(e.to_string()))?;
    Ok(())
}

/// Load model from file
///
/// Legacy function for loading `Sequential` models from JSON.
/// For new code, prefer the `SafeTensors`-based API via `ModelDeserialize::load()`.
#[cfg(feature = "legacy_serialization")]
#[allow(dead_code)]
pub fn load_model<
    F: Float + Debug + Display + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
    P: AsRef<Path>,
>(
    path: P,
    _format: SerializationFormat,
) -> Result<Sequential<F>> {
    let bytes = fs::read(path).map_err(|e| NeuralError::IOError(e.to_string()))?;
    let serialized: SerializedModel = serde_json::from_slice(&bytes)
        .map_err(|e| NeuralError::DeserializationError(e.to_string()))?;
    deserialize_model(&serialized)
}

/// Serialize model to SerializedModel
#[cfg(feature = "legacy_serialization")]
#[allow(dead_code)]
fn serialize_model<
    F: Float + Debug + Display + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
>(
    model: &Sequential<F>,
) -> Result<SerializedModel> {
    let mut layers = Vec::new();
    let mut parameters = Vec::new();

    for layer in model.layers() {
        if let Some(dense) = layer.as_any().downcast_ref::<Dense<F>>() {
            let config = LayerConfig::Dense(DenseConfig {
                input_dim: dense.input_dim(),
                output_dim: dense.output_dim(),
                activation: None, // Dense::activation_name() not available
            });
            layers.push(config);
            let layer_params_owned = dense.get_parameters();
            let layer_params: Vec<&Array<F, scirs2_core::ndarray::IxDyn>> =
                layer_params_owned.iter().collect();
            let params = extract_parameters(layer_params)?;
            parameters.push(params);
        } else if let Some(dropout) = layer.as_any().downcast_ref::<Dropout<F>>() {
            let _ = dropout; // p() not available on Dropout
            let config = LayerConfig::Dropout(DropoutConfig { p: 0.5 });
            layers.push(config);
            parameters.push(Vec::new());
        } else {
            return Err(NeuralError::SerializationError(
                "Unsupported layer type for legacy serialization. Use SafeTensors API instead."
                    .to_string(),
            ));
        }
    }

    Ok(SerializedModel {
        name: "SciRS2 Sequential Model".to_string(),
        version: "0.1.0".to_string(),
        layers,
        parameters,
    })
}

/// Extract parameters from layer
#[cfg(feature = "legacy_serialization")]
#[allow(dead_code)]
fn extract_parameters<F: Float + Debug + ScalarOperand + Send + Sync>(
    params: Vec<&Array<F, scirs2_core::ndarray::IxDyn>>,
) -> Result<Vec<Vec<f64>>> {
    let mut result = Vec::new();
    for param in params.iter() {
        let f64_vec: Vec<f64> = param
            .iter()
            .map(|&x| {
                x.to_f64().ok_or_else(|| {
                    NeuralError::SerializationError("Cannot convert parameter to f64".to_string())
                })
            })
            .collect::<Result<Vec<f64>>>()?;
        result.push(f64_vec);
    }
    Ok(result)
}

/// Deserialize model from SerializedModel
#[cfg(feature = "legacy_serialization")]
#[allow(dead_code)]
fn deserialize_model<
    F: Float + Debug + Display + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
>(
    serialized: &SerializedModel,
) -> Result<Sequential<F>> {
    let empty_params: Vec<Vec<f64>> = Vec::new();
    let mut bound_layers: Vec<Box<dyn Layer<F> + Send + Sync>> = Vec::new();

    for (i, layer_config) in serialized.layers.iter().enumerate() {
        let params = if i < serialized.parameters.len() {
            &serialized.parameters[i]
        } else {
            &empty_params
        };

        match layer_config {
            LayerConfig::Dense(config) => {
                let layer = create_dense_layer::<F>(config, params)?;
                bound_layers.push(Box::new(layer));
            }
            LayerConfig::Dropout(config) => {
                let layer = create_dropout::<F>(config)?;
                bound_layers.push(Box::new(layer));
            }
            _ => {
                return Err(NeuralError::DeserializationError(
                    "Layer type not supported in legacy deserialization. Use SafeTensors API."
                        .to_string(),
                ));
            }
        }
    }

    Ok(Sequential::from_layers(bound_layers))
}

/// Create a Dense layer from configuration and parameters
#[cfg(feature = "legacy_serialization")]
#[allow(dead_code)]
fn create_dense_layer<
    F: Float + Debug + Display + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
>(
    config: &DenseConfig,
    params: &[Vec<f64>],
) -> Result<Dense<F>> {
    let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
    let mut layer = Dense::new(
        config.input_dim,
        config.output_dim,
        config.activation.as_deref(),
        &mut rng,
    )?;

    if params.len() >= 2 {
        let weightsshape = [config.input_dim, config.output_dim];
        let biasshape = [config.output_dim];

        if params[0].len() == config.output_dim * config.input_dim {
            let weights_array = match array_from_vec::<F>(&params[0], &weightsshape) {
                Ok(arr) => arr,
                Err(_) => {
                    let transposedshape = [config.output_dim, config.input_dim];
                    let transposed_arr = array_from_vec::<F>(&params[0], &transposedshape)?;
                    transposed_arr.t().to_owned().into_dyn()
                }
            };
            let bias_array = array_from_vec::<F>(&params[1], &biasshape)?;
            layer.set_parameters(vec![weights_array, bias_array])?;
        } else {
            return Err(NeuralError::SerializationError(format!(
                "Weight vector length ({}) doesn't match expected shape size ({})",
                params[0].len(),
                config.input_dim * config.output_dim
            )));
        }
    }
    Ok(layer)
}

/// Create a Dropout layer from configuration
#[cfg(feature = "legacy_serialization")]
#[allow(dead_code)]
fn create_dropout<
    F: Float + Debug + Display + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
>(
    config: &DropoutConfig,
) -> Result<Dropout<F>> {
    let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
    Dropout::new(config.p, &mut rng)
}

/// Convert a vector of f64 values to an ndarray with the given shape
#[cfg(feature = "legacy_serialization")]
#[allow(dead_code)]
fn array_from_vec<
    F: Float + Debug + Display + ScalarOperand + FromPrimitive + NumAssign + Send + Sync + 'static,
>(
    vec: &[f64],
    shape: &[usize],
) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
    let shape_size: usize = shape.iter().product();
    if vec.len() != shape_size {
        return Err(NeuralError::SerializationError(format!(
            "Parameter vector length ({}) doesn't match expected shape size ({})",
            vec.len(),
            shape_size
        )));
    }
    let f_vec: Vec<F> = vec
        .iter()
        .map(|&x| {
            F::from(x).ok_or_else(|| {
                NeuralError::SerializationError(format!("Cannot convert {} to target type", x))
            })
        })
        .collect::<Result<Vec<F>>>()?;
    let shape_ix = scirs2_core::ndarray::IxDyn(shape);
    Array::from_shape_vec(shape_ix, f_vec)
        .map_err(|e| NeuralError::SerializationError(e.to_string()))
}

// =============================================================================
// Activation function utilities (always available)
// =============================================================================

/// Serializable activation function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    /// ReLU activation
    ReLU,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Softmax activation
    Softmax,
    /// LeakyReLU activation
    LeakyReLU(f64),
    /// ELU activation (serialized; implemented as LeakyReLU for forward compat)
    ELU(f64),
    /// GELU activation
    GELU,
    /// Swish activation
    Swish,
    /// Mish activation
    Mish,
}

impl ActivationFunction {
    /// Convert activation function name to ActivationFunction enum
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "relu" | "ReLU" => Some(ActivationFunction::ReLU),
            "sigmoid" | "Sigmoid" => Some(ActivationFunction::Sigmoid),
            "tanh" | "Tanh" => Some(ActivationFunction::Tanh),
            "softmax" | "Softmax" => Some(ActivationFunction::Softmax),
            "gelu" | "GELU" => Some(ActivationFunction::GELU),
            "swish" | "Swish" => Some(ActivationFunction::Swish),
            "mish" | "Mish" => Some(ActivationFunction::Mish),
            _ => {
                if name.starts_with("leaky_relu") || name.starts_with("LeakyReLU") {
                    let parts: Vec<&str> = name.split('(').collect();
                    if parts.len() == 2 {
                        let alpha_str = parts[1].trim_end_matches(')');
                        if let Ok(alpha) = alpha_str.parse::<f64>() {
                            return Some(ActivationFunction::LeakyReLU(alpha));
                        }
                    }
                    Some(ActivationFunction::LeakyReLU(0.01))
                } else if name.starts_with("elu") || name.starts_with("ELU") {
                    let parts: Vec<&str> = name.split('(').collect();
                    if parts.len() == 2 {
                        let alpha_str = parts[1].trim_end_matches(')');
                        if let Ok(alpha) = alpha_str.parse::<f64>() {
                            return Some(ActivationFunction::ELU(alpha));
                        }
                    }
                    Some(ActivationFunction::ELU(1.0))
                } else {
                    None
                }
            }
        }
    }

    /// Convert ActivationFunction enum to activation function name
    pub fn to_name(&self) -> String {
        match self {
            ActivationFunction::ReLU => "relu".to_string(),
            ActivationFunction::Sigmoid => "sigmoid".to_string(),
            ActivationFunction::Tanh => "tanh".to_string(),
            ActivationFunction::Softmax => "softmax".to_string(),
            ActivationFunction::LeakyReLU(alpha) => format!("leaky_relu({})", alpha),
            ActivationFunction::ELU(alpha) => format!("elu({})", alpha),
            ActivationFunction::GELU => "gelu".to_string(),
            ActivationFunction::Swish => "swish".to_string(),
            ActivationFunction::Mish => "mish".to_string(),
        }
    }

    /// Create activation function from enum
    ///
    /// Note: ELU is not currently implemented; it falls back to LeakyReLU.
    pub fn create<
        F: Float + Debug + scirs2_core::NumAssign + scirs2_core::ndarray::ScalarOperand + Send + Sync,
    >(
        &self,
    ) -> Box<dyn Activation<F>> {
        match self {
            ActivationFunction::ReLU => Box::new(ReLU::new()),
            ActivationFunction::Sigmoid => Box::new(Sigmoid::new()),
            ActivationFunction::Tanh => Box::new(Tanh::new()),
            ActivationFunction::Softmax => Box::new(Softmax::new(1)),
            ActivationFunction::LeakyReLU(alpha) => Box::new(LeakyReLU::new(*alpha)),
            ActivationFunction::ELU(alpha) => Box::new(LeakyReLU::new(*alpha)),
            ActivationFunction::GELU => Box::new(GELU::new()),
            ActivationFunction::Swish => Box::new(Swish::new(1.0)),
            ActivationFunction::Mish => Box::new(Mish::new()),
        }
    }
}

/// Activation function factory
pub struct ActivationFactory;

impl ActivationFactory {
    /// Create activation function from name
    pub fn create<
        F: Float + Debug + scirs2_core::NumAssign + scirs2_core::ndarray::ScalarOperand + Send + Sync,
    >(
        name: &str,
    ) -> Option<Box<dyn Activation<F>>> {
        ActivationFunction::from_name(name).map(|af| af.create::<F>())
    }

    /// Get activation function names
    pub fn get_activation_names() -> HashMap<&'static str, &'static str> {
        let mut names = HashMap::new();
        names.insert("relu", "ReLU activation function");
        names.insert("sigmoid", "Sigmoid activation function");
        names.insert("tanh", "Tanh activation function");
        names.insert("softmax", "Softmax activation function");
        names.insert("leaky_relu", "Leaky ReLU activation function");
        names.insert("elu", "ELU activation function");
        names.insert("gelu", "GELU activation function");
        names.insert("swish", "Swish activation function");
        names.insert("mish", "Mish activation function");
        names
    }
}

#[cfg(test)]
mod tests;
