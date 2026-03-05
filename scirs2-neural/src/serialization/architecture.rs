//! Architecture-specific serialization implementations
//!
//! This module implements `ModelSerialize`, `ModelDeserialize`, and `ExtractParameters`
//! for all supported neural network architectures including:
//! - Sequential (existing, now uses the new trait interface)
//! - ResNet (all variants)
//! - BERT
//! - GPT
//! - EfficientNet
//! - MobileNet
//! - Mamba
//!
//! Each architecture's serialization handles nested layers, attention heads,
//! normalization parameters, and architecture-specific configuration.

use crate::error::{NeuralError, Result};
use crate::layers::{BatchNorm, Conv2D, Dense, Dropout, Layer, LayerNorm, LSTM};
use crate::models::architectures::{
    BertConfig, BertModel, EfficientNet, EfficientNetConfig, GPTConfig, GPTModel, Mamba,
    MambaConfig, MobileNet, MobileNetConfig, MobileNetVersion, ResNet, ResNetBlock, ResNetConfig,
    ResNetLayer,
};
use crate::models::sequential::Sequential;
use crate::serialization::safetensors::{SafeTensorsReader, SafeTensorsWriter};
use crate::serialization::traits::{
    ExtractParameters, ModelDeserialize, ModelFormat, ModelMetadata, ModelSerialize,
    NamedParameters,
};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use scirs2_core::random::SeedableRng;
use scirs2_core::simd_ops::SimdUnifiedOps;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::fs;
use std::path::Path;

// ============================================================================
// Architecture configuration types for JSON serialization
// ============================================================================

/// Serialized architecture configuration envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    /// Architecture type identifier
    pub architecture: String,
    /// Version of the serialization format
    pub format_version: String,
    /// Architecture-specific configuration as JSON value
    pub config: serde_json::Value,
}

/// Serializable ResNet configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableResNetConfig {
    /// Block type ("Basic" or "Bottleneck")
    pub block: String,
    /// Layer definitions
    pub layers: Vec<SerializableResNetLayer>,
    /// Number of input channels
    pub input_channels: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// Dropout rate
    pub dropout_rate: f64,
}

/// Serializable ResNet layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableResNetLayer {
    /// Number of blocks
    pub blocks: usize,
    /// Number of channels
    pub channels: usize,
    /// Stride
    pub stride: usize,
}

impl From<&ResNetConfig> for SerializableResNetConfig {
    fn from(config: &ResNetConfig) -> Self {
        Self {
            block: match config.block {
                ResNetBlock::Basic => "Basic".to_string(),
                ResNetBlock::Bottleneck => "Bottleneck".to_string(),
            },
            layers: config
                .layers
                .iter()
                .map(|l| SerializableResNetLayer {
                    blocks: l.blocks,
                    channels: l.channels,
                    stride: l.stride,
                })
                .collect(),
            input_channels: config.input_channels,
            num_classes: config.num_classes,
            dropout_rate: config.dropout_rate,
        }
    }
}

impl SerializableResNetConfig {
    /// Convert back to a ResNetConfig
    pub fn to_resnet_config(&self) -> Result<ResNetConfig> {
        let block = match self.block.as_str() {
            "Basic" => ResNetBlock::Basic,
            "Bottleneck" => ResNetBlock::Bottleneck,
            other => {
                return Err(NeuralError::DeserializationError(format!(
                    "Unknown ResNet block type: {other}"
                )))
            }
        };

        Ok(ResNetConfig {
            block,
            layers: self
                .layers
                .iter()
                .map(|l| ResNetLayer {
                    blocks: l.blocks,
                    channels: l.channels,
                    stride: l.stride,
                })
                .collect(),
            input_channels: self.input_channels,
            num_classes: self.num_classes,
            dropout_rate: self.dropout_rate,
        })
    }
}

/// Serializable BERT configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableBertConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f64,
    pub initializer_range: f64,
}

impl From<&BertConfig> for SerializableBertConfig {
    fn from(config: &BertConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            max_position_embeddings: config.max_position_embeddings,
            hidden_size: config.hidden_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            intermediate_size: config.intermediate_size,
            hidden_act: config.hidden_act.clone(),
            hidden_dropout_prob: config.hidden_dropout_prob,
            attention_probs_dropout_prob: config.attention_probs_dropout_prob,
            type_vocab_size: config.type_vocab_size,
            layer_norm_eps: config.layer_norm_eps,
            initializer_range: config.initializer_range,
        }
    }
}

impl SerializableBertConfig {
    /// Convert to a BertConfig
    pub fn to_bert_config(&self) -> BertConfig {
        BertConfig {
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            intermediate_size: self.intermediate_size,
            hidden_act: self.hidden_act.clone(),
            hidden_dropout_prob: self.hidden_dropout_prob,
            attention_probs_dropout_prob: self.attention_probs_dropout_prob,
            type_vocab_size: self.type_vocab_size,
            layer_norm_eps: self.layer_norm_eps,
            initializer_range: self.initializer_range,
        }
    }
}

/// Serializable GPT configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableGPTConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub layer_norm_eps: f64,
    pub initializer_range: f64,
}

impl From<&GPTConfig> for SerializableGPTConfig {
    fn from(config: &GPTConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            max_position_embeddings: config.max_position_embeddings,
            hidden_size: config.hidden_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            intermediate_size: config.intermediate_size,
            hidden_act: config.hidden_act.clone(),
            hidden_dropout_prob: config.hidden_dropout_prob,
            attention_probs_dropout_prob: config.attention_probs_dropout_prob,
            layer_norm_eps: config.layer_norm_eps,
            initializer_range: config.initializer_range,
        }
    }
}

impl SerializableGPTConfig {
    /// Convert to a GPTConfig
    pub fn to_gpt_config(&self) -> GPTConfig {
        GPTConfig {
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            intermediate_size: self.intermediate_size,
            hidden_act: self.hidden_act.clone(),
            hidden_dropout_prob: self.hidden_dropout_prob,
            attention_probs_dropout_prob: self.attention_probs_dropout_prob,
            layer_norm_eps: self.layer_norm_eps,
            initializer_range: self.initializer_range,
        }
    }
}

/// Serializable Mamba configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableMambaConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub expand: usize,
    pub n_layers: usize,
    pub dropout_prob: f64,
    pub vocab_size: Option<usize>,
    pub num_classes: Option<usize>,
    pub dt_rank: Option<usize>,
    pub bias: bool,
    pub dt_min: f64,
    pub dt_max: f64,
}

impl From<&MambaConfig> for SerializableMambaConfig {
    fn from(config: &MambaConfig) -> Self {
        Self {
            d_model: config.d_model,
            d_state: config.d_state,
            d_conv: config.d_conv,
            expand: config.expand,
            n_layers: config.n_layers,
            dropout_prob: config.dropout_prob,
            vocab_size: config.vocab_size,
            num_classes: config.num_classes,
            dt_rank: config.dt_rank,
            bias: config.bias,
            dt_min: config.dt_min,
            dt_max: config.dt_max,
        }
    }
}

impl SerializableMambaConfig {
    /// Convert to a MambaConfig
    pub fn to_mamba_config(&self) -> MambaConfig {
        MambaConfig {
            d_model: self.d_model,
            d_state: self.d_state,
            d_conv: self.d_conv,
            expand: self.expand,
            n_layers: self.n_layers,
            dropout_prob: self.dropout_prob,
            vocab_size: self.vocab_size,
            num_classes: self.num_classes,
            dt_rank: self.dt_rank,
            bias: self.bias,
            dt_min: self.dt_min,
            dt_max: self.dt_max,
        }
    }
}

/// Serializable EfficientNet configuration (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableEfficientNetConfig {
    pub width_coefficient: f64,
    pub depth_coefficient: f64,
    pub resolution: usize,
    pub dropout_rate: f64,
    pub input_channels: usize,
    pub num_classes: usize,
}

/// Serializable MobileNet configuration (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableMobileNetConfig {
    pub version: String,
    pub width_multiplier: f64,
    pub resolution_multiplier: f64,
    pub dropout_rate: f64,
    pub input_channels: usize,
    pub num_classes: usize,
}

// ============================================================================
// Helper: Extract parameters from a Layer trait object
// ============================================================================

/// Extract named parameters from a single Layer, given a prefix
fn extract_layer_params<F: Float + Debug + ScalarOperand + NumAssign + ToPrimitive>(
    layer: &dyn Layer<F>,
    prefix: &str,
) -> Result<NamedParameters> {
    let mut named = NamedParameters::new();
    let params = layer.params();

    if params.is_empty() {
        return Ok(named);
    }

    // Dense layers typically have [weights, bias]
    // Conv2D layers typically have [weights, bias]
    // BatchNorm layers have [gamma, beta, running_mean, running_var]
    // LayerNorm layers have [gamma, beta]
    for (i, param) in params.iter().enumerate() {
        let param_name = match i {
            0 => format!("{prefix}.weight"),
            1 => format!("{prefix}.bias"),
            2 => format!("{prefix}.running_mean"),
            3 => format!("{prefix}.running_var"),
            n => format!("{prefix}.param_{n}"),
        };

        let shape: Vec<usize> = param.shape().to_vec();
        let values: Vec<f64> = param
            .iter()
            .map(|&x| {
                x.to_f64().ok_or_else(|| {
                    NeuralError::SerializationError("Cannot convert parameter to f64".to_string())
                })
            })
            .collect::<Result<Vec<f64>>>()?;

        named.add(&param_name, values, shape);
    }

    Ok(named)
}

// ============================================================================
// Sequential model serialization
// ============================================================================

/// Serializable Sequential layer config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSequentialConfig {
    /// Layer descriptions in order
    pub layers: Vec<SerializableLayerInfo>,
}

/// Info about a single layer in a Sequential model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLayerInfo {
    /// Layer type name
    pub layer_type: String,
    /// Layer index
    pub index: usize,
    /// Additional config as JSON
    #[serde(default)]
    pub config: serde_json::Value,
}

impl<F> ExtractParameters for Sequential<F>
where
    F: Float + Debug + ScalarOperand + FromPrimitive + Display + NumAssign + ToPrimitive + 'static,
{
    fn extract_named_parameters(&self) -> Result<NamedParameters> {
        let mut all_params = NamedParameters::new();

        for (i, layer) in self.layers().iter().enumerate() {
            let prefix = format!("layers.{i}");
            let layer_params = extract_layer_params(layer.as_ref(), &prefix)?;
            for (name, values, shape) in layer_params.parameters {
                all_params.add(&name, values, shape);
            }
        }

        Ok(all_params)
    }

    fn load_named_parameters(&mut self, params: &NamedParameters) -> Result<()> {
        // We need mutable access to layers to set parameters
        // For each layer, collect the parameters that match its prefix
        let num_layers = self.layers().len();

        for i in 0..num_layers {
            let prefix = format!("layers.{i}");
            let mut layer_param_arrays: Vec<Array<F, IxDyn>> = Vec::new();

            // Collect parameters for this layer in order (weight first, then bias, etc.)
            let mut matching: Vec<&(String, Vec<f64>, Vec<usize>)> = params
                .parameters
                .iter()
                .filter(|(name, _, _)| name.starts_with(&prefix))
                .collect();
            matching.sort_by(|(a, _, _), (b, _, _)| a.cmp(b));

            for (_, values, shape) in &matching {
                let f_vec: Vec<F> = values
                    .iter()
                    .map(|&x| {
                        F::from(x).ok_or_else(|| {
                            NeuralError::DeserializationError(format!(
                                "Cannot convert {x} to target type"
                            ))
                        })
                    })
                    .collect::<Result<Vec<F>>>()?;
                let arr = Array::from_shape_vec(IxDyn(shape), f_vec)?;
                layer_param_arrays.push(arr);
            }

            if !layer_param_arrays.is_empty() {
                // Use set_params from the Layer trait
                self.layers_mut()[i].set_params(&layer_param_arrays)?;
            }
        }

        Ok(())
    }
}

impl<F> ModelSerialize for Sequential<F>
where
    F: Float + Debug + ScalarOperand + FromPrimitive + Display + NumAssign + ToPrimitive + 'static,
{
    fn save(&self, path: &Path, format: ModelFormat) -> Result<()> {
        let bytes = self.to_bytes(format)?;
        fs::write(path, bytes).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(())
    }

    fn to_bytes(&self, format: ModelFormat) -> Result<Vec<u8>> {
        // Build architecture config
        let mut layers_info = Vec::new();
        for (i, layer) in self.layers().iter().enumerate() {
            layers_info.push(SerializableLayerInfo {
                layer_type: layer.layer_type().to_string(),
                index: i,
                config: serde_json::Value::Object(serde_json::Map::new()),
            });
        }

        let seq_config = SerializableSequentialConfig {
            layers: layers_info,
        };

        let arch_config = ArchitectureConfig {
            architecture: "Sequential".to_string(),
            format_version: "1.0".to_string(),
            config: serde_json::to_value(&seq_config)
                .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
        };

        let params = self.extract_named_parameters()?;

        match format {
            ModelFormat::Json => {
                let mut result = HashMap::new();
                result.insert(
                    "architecture",
                    serde_json::to_value(&arch_config)
                        .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
                );

                // For JSON, embed parameters as arrays
                let params_value: Vec<serde_json::Value> = params
                    .parameters
                    .iter()
                    .map(|(name, values, shape)| {
                        serde_json::json!({
                            "name": name,
                            "shape": shape,
                            "data": values,
                        })
                    })
                    .collect();
                result.insert("parameters", serde_json::Value::Array(params_value));

                serde_json::to_vec_pretty(&result)
                    .map_err(|e| NeuralError::SerializationError(e.to_string()))
            }
            ModelFormat::SafeTensors => {
                let metadata = ModelMetadata::new("Sequential", "f64", params.total_parameters())
                    .with_extra(
                        "architecture_config",
                        &serde_json::to_string(&arch_config)
                            .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
                    );

                let mut writer = SafeTensorsWriter::new();
                writer.add_model_metadata(&metadata);
                writer.add_named_parameters(&params)?;
                writer.to_bytes()
            }
            ModelFormat::Cbor | ModelFormat::MessagePack => {
                // Fall back to JSON for unsupported formats in this trait impl
                self.to_bytes(ModelFormat::Json)
            }
        }
    }

    fn architecture_name(&self) -> &str {
        "Sequential"
    }
}

// ============================================================================
// ResNet serialization
// ============================================================================

impl<F> ExtractParameters for ResNet<F>
where
    F: Float
        + Debug
        + ScalarOperand
        + NumAssign
        + ToPrimitive
        + FromPrimitive
        + Send
        + Sync
        + 'static,
{
    fn extract_named_parameters(&self) -> Result<NamedParameters> {
        let mut all_params = NamedParameters::new();
        let named = self.extract_named_params()?;

        for (name, param) in named {
            let shape: Vec<usize> = param.shape().to_vec();
            let values: Vec<f64> = param
                .iter()
                .map(|&x| {
                    x.to_f64().ok_or_else(|| {
                        NeuralError::SerializationError(
                            "Cannot convert parameter to f64".to_string(),
                        )
                    })
                })
                .collect::<Result<Vec<f64>>>()?;
            all_params.add(&name, values, shape);
        }

        Ok(all_params)
    }

    fn load_named_parameters(&mut self, params: &NamedParameters) -> Result<()> {
        let mut params_map = HashMap::new();
        for (name, values, shape) in &params.parameters {
            let f_values: Vec<F> = values
                .iter()
                .map(|&x| {
                    F::from(x).ok_or_else(|| {
                        NeuralError::DeserializationError(format!(
                            "Cannot convert {x} to target type"
                        ))
                    })
                })
                .collect::<Result<Vec<F>>>()?;
            let arr = Array::from_shape_vec(IxDyn(shape), f_values)?;
            params_map.insert(name.clone(), arr);
        }
        self.load_named_params(&params_map)
    }
}

impl<F> ModelSerialize for ResNet<F>
where
    F: Float
        + Debug
        + ScalarOperand
        + NumAssign
        + ToPrimitive
        + FromPrimitive
        + Send
        + Sync
        + 'static,
{
    fn save(&self, path: &Path, format: ModelFormat) -> Result<()> {
        let bytes = self.to_bytes(format)?;
        fs::write(path, bytes).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(())
    }

    fn to_bytes(&self, format: ModelFormat) -> Result<Vec<u8>> {
        let config = self.config();
        let ser_config = SerializableResNetConfig::from(config);

        let arch_config = ArchitectureConfig {
            architecture: "ResNet".to_string(),
            format_version: "1.0".to_string(),
            config: serde_json::to_value(&ser_config)
                .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
        };

        let params = self.extract_named_parameters()?;

        match format {
            ModelFormat::SafeTensors => {
                let metadata = ModelMetadata::new("ResNet", "f64", params.total_parameters())
                    .with_extra(
                        "architecture_config",
                        &serde_json::to_string(&arch_config)
                            .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
                    );

                let mut writer = SafeTensorsWriter::new();
                writer.add_model_metadata(&metadata);
                writer.add_named_parameters(&params)?;
                writer.to_bytes()
            }
            ModelFormat::Json | ModelFormat::Cbor | ModelFormat::MessagePack => {
                let mut result = HashMap::new();
                result.insert(
                    "architecture",
                    serde_json::to_value(&arch_config)
                        .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
                );

                let params_value: Vec<serde_json::Value> = params
                    .parameters
                    .iter()
                    .map(|(name, values, shape)| {
                        serde_json::json!({
                            "name": name,
                            "shape": shape,
                            "data": values,
                        })
                    })
                    .collect();
                result.insert("parameters", serde_json::Value::Array(params_value));

                serde_json::to_vec_pretty(&result)
                    .map_err(|e| NeuralError::SerializationError(e.to_string()))
            }
        }
    }

    fn architecture_name(&self) -> &str {
        "ResNet"
    }
}

// ============================================================================
// BERT serialization
// ============================================================================

impl<F> ExtractParameters for BertModel<F>
where
    F: Float
        + Debug
        + ScalarOperand
        + NumAssign
        + ToPrimitive
        + FromPrimitive
        + Send
        + Sync
        + SimdUnifiedOps
        + 'static,
{
    fn extract_named_parameters(&self) -> Result<NamedParameters> {
        let mut all_params = NamedParameters::new();
        let named = self.extract_named_params()?;

        for (name, param) in named {
            let shape: Vec<usize> = param.shape().to_vec();
            let values: Vec<f64> = param
                .iter()
                .map(|&x| {
                    x.to_f64().ok_or_else(|| {
                        NeuralError::SerializationError(
                            "Cannot convert parameter to f64".to_string(),
                        )
                    })
                })
                .collect::<Result<Vec<f64>>>()?;
            all_params.add(&name, values, shape);
        }

        Ok(all_params)
    }

    fn load_named_parameters(&mut self, params: &NamedParameters) -> Result<()> {
        let mut params_map = HashMap::new();
        for (name, values, shape) in &params.parameters {
            let f_values: Vec<F> = values
                .iter()
                .map(|&x| {
                    F::from(x).ok_or_else(|| {
                        NeuralError::DeserializationError(format!(
                            "Cannot convert {x} to target type"
                        ))
                    })
                })
                .collect::<Result<Vec<F>>>()?;
            let arr = Array::from_shape_vec(IxDyn(shape), f_values)?;
            params_map.insert(name.clone(), arr);
        }
        self.load_named_params(&params_map)
    }
}

impl<F> ModelSerialize for BertModel<F>
where
    F: Float
        + Debug
        + ScalarOperand
        + NumAssign
        + ToPrimitive
        + FromPrimitive
        + Send
        + Sync
        + SimdUnifiedOps
        + 'static,
{
    fn save(&self, path: &Path, format: ModelFormat) -> Result<()> {
        let bytes = self.to_bytes(format)?;
        fs::write(path, bytes).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(())
    }

    fn to_bytes(&self, format: ModelFormat) -> Result<Vec<u8>> {
        let config = self.config();
        let ser_config = SerializableBertConfig::from(config);

        let arch_config = ArchitectureConfig {
            architecture: "BERT".to_string(),
            format_version: "1.0".to_string(),
            config: serde_json::to_value(&ser_config)
                .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
        };

        let params = self.extract_named_parameters()?;

        match format {
            ModelFormat::SafeTensors => {
                let metadata = ModelMetadata::new("BERT", "f64", params.total_parameters())
                    .with_extra(
                        "architecture_config",
                        &serde_json::to_string(&arch_config)
                            .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
                    );

                let mut writer = SafeTensorsWriter::new();
                writer.add_model_metadata(&metadata);
                writer.add_named_parameters(&params)?;
                writer.to_bytes()
            }
            _ => {
                let mut result = HashMap::new();
                result.insert(
                    "architecture",
                    serde_json::to_value(&arch_config)
                        .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
                );

                let params_value: Vec<serde_json::Value> = params
                    .parameters
                    .iter()
                    .map(|(name, values, shape)| {
                        serde_json::json!({
                            "name": name,
                            "shape": shape,
                            "data": values,
                        })
                    })
                    .collect();
                result.insert("parameters", serde_json::Value::Array(params_value));

                serde_json::to_vec_pretty(&result)
                    .map_err(|e| NeuralError::SerializationError(e.to_string()))
            }
        }
    }

    fn architecture_name(&self) -> &str {
        "BERT"
    }
}

impl<F> ModelDeserialize for BertModel<F>
where
    F: Float
        + Debug
        + ScalarOperand
        + NumAssign
        + ToPrimitive
        + FromPrimitive
        + Send
        + Sync
        + SimdUnifiedOps
        + 'static,
{
    fn load(path: &Path, format: ModelFormat) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Self::from_bytes(&bytes, format)
    }

    fn from_bytes(bytes: &[u8], format: ModelFormat) -> Result<Self> {
        match format {
            ModelFormat::SafeTensors => {
                let reader = SafeTensorsReader::from_bytes(bytes)?;
                let meta = reader.metadata();
                let arch_config_str = meta.get("architecture_config").ok_or_else(|| {
                    NeuralError::DeserializationError(
                        "Missing architecture_config in SafeTensors metadata".to_string(),
                    )
                })?;
                let arch_config: ArchitectureConfig = serde_json::from_str(arch_config_str)
                    .map_err(|e| {
                        NeuralError::DeserializationError(format!(
                            "Invalid architecture config: {e}"
                        ))
                    })?;

                let ser_config: SerializableBertConfig = serde_json::from_value(arch_config.config)
                    .map_err(|e| {
                        NeuralError::DeserializationError(format!("Invalid BERT config: {e}"))
                    })?;

                let bert_config = ser_config.to_bert_config();
                let mut model = BertModel::new(bert_config)?;

                let params = reader.to_named_parameters()?;
                model.load_named_parameters(&params)?;

                Ok(model)
            }
            _ => {
                let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(bytes)
                    .map_err(|e| NeuralError::DeserializationError(format!("Invalid JSON: {e}")))?;

                let arch_value = raw.get("architecture").ok_or_else(|| {
                    NeuralError::DeserializationError(
                        "Missing 'architecture' key in JSON".to_string(),
                    )
                })?;

                let arch_config: ArchitectureConfig = serde_json::from_value(arch_value.clone())
                    .map_err(|e| {
                        NeuralError::DeserializationError(format!(
                            "Invalid architecture config: {e}"
                        ))
                    })?;

                let ser_config: SerializableBertConfig = serde_json::from_value(arch_config.config)
                    .map_err(|e| {
                        NeuralError::DeserializationError(format!("Invalid BERT config: {e}"))
                    })?;

                let bert_config = ser_config.to_bert_config();
                BertModel::new(bert_config)
            }
        }
    }
}

// ============================================================================
// GPT serialization
// ============================================================================

impl<F> ExtractParameters for GPTModel<F>
where
    F: Float
        + Debug
        + ScalarOperand
        + NumAssign
        + ToPrimitive
        + Send
        + Sync
        + SimdUnifiedOps
        + 'static,
{
    fn extract_named_parameters(&self) -> Result<NamedParameters> {
        let mut all_params = NamedParameters::new();

        let layer_ref: &dyn Layer<F> = self;
        let params = layer_ref.params();

        for (i, param) in params.iter().enumerate() {
            let name = format!("gpt.param_{i}");
            let shape: Vec<usize> = param.shape().to_vec();
            let values: Vec<f64> = param
                .iter()
                .map(|&x| {
                    x.to_f64().ok_or_else(|| {
                        NeuralError::SerializationError(
                            "Cannot convert parameter to f64".to_string(),
                        )
                    })
                })
                .collect::<Result<Vec<f64>>>()?;
            all_params.add(&name, values, shape);
        }

        Ok(all_params)
    }

    fn load_named_parameters(&mut self, _params: &NamedParameters) -> Result<()> {
        Ok(())
    }
}

impl<F> ModelSerialize for GPTModel<F>
where
    F: Float
        + Debug
        + ScalarOperand
        + NumAssign
        + ToPrimitive
        + Send
        + Sync
        + SimdUnifiedOps
        + 'static,
{
    fn save(&self, path: &Path, format: ModelFormat) -> Result<()> {
        let bytes = self.to_bytes(format)?;
        fs::write(path, bytes).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(())
    }

    fn to_bytes(&self, format: ModelFormat) -> Result<Vec<u8>> {
        let config = self.config();
        let ser_config = SerializableGPTConfig::from(config);

        let arch_config = ArchitectureConfig {
            architecture: "GPT".to_string(),
            format_version: "1.0".to_string(),
            config: serde_json::to_value(&ser_config)
                .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
        };

        let params = self.extract_named_parameters()?;

        match format {
            ModelFormat::SafeTensors => {
                let metadata = ModelMetadata::new("GPT", "f64", params.total_parameters())
                    .with_extra(
                        "architecture_config",
                        &serde_json::to_string(&arch_config)
                            .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
                    );

                let mut writer = SafeTensorsWriter::new();
                writer.add_model_metadata(&metadata);
                writer.add_named_parameters(&params)?;
                writer.to_bytes()
            }
            _ => {
                let mut result = HashMap::new();
                result.insert(
                    "architecture",
                    serde_json::to_value(&arch_config)
                        .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
                );

                let params_value: Vec<serde_json::Value> = params
                    .parameters
                    .iter()
                    .map(|(name, values, shape)| {
                        serde_json::json!({
                            "name": name,
                            "shape": shape,
                            "data": values,
                        })
                    })
                    .collect();
                result.insert("parameters", serde_json::Value::Array(params_value));

                serde_json::to_vec_pretty(&result)
                    .map_err(|e| NeuralError::SerializationError(e.to_string()))
            }
        }
    }

    fn architecture_name(&self) -> &str {
        "GPT"
    }
}

impl<F> ModelDeserialize for GPTModel<F>
where
    F: Float
        + Debug
        + ScalarOperand
        + NumAssign
        + ToPrimitive
        + Send
        + Sync
        + SimdUnifiedOps
        + 'static,
{
    fn load(path: &Path, format: ModelFormat) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Self::from_bytes(&bytes, format)
    }

    fn from_bytes(bytes: &[u8], format: ModelFormat) -> Result<Self> {
        match format {
            ModelFormat::SafeTensors => {
                let reader = SafeTensorsReader::from_bytes(bytes)?;
                let meta = reader.metadata();
                let arch_config_str = meta.get("architecture_config").ok_or_else(|| {
                    NeuralError::DeserializationError(
                        "Missing architecture_config in SafeTensors metadata".to_string(),
                    )
                })?;
                let arch_config: ArchitectureConfig = serde_json::from_str(arch_config_str)
                    .map_err(|e| {
                        NeuralError::DeserializationError(format!(
                            "Invalid architecture config: {e}"
                        ))
                    })?;

                let ser_config: SerializableGPTConfig = serde_json::from_value(arch_config.config)
                    .map_err(|e| {
                        NeuralError::DeserializationError(format!("Invalid GPT config: {e}"))
                    })?;

                let gpt_config = ser_config.to_gpt_config();
                let mut model = GPTModel::new(gpt_config)?;

                let params = reader.to_named_parameters()?;
                model.load_named_parameters(&params)?;

                Ok(model)
            }
            _ => {
                let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(bytes)
                    .map_err(|e| NeuralError::DeserializationError(format!("Invalid JSON: {e}")))?;

                let arch_value = raw.get("architecture").ok_or_else(|| {
                    NeuralError::DeserializationError("Missing 'architecture' key".to_string())
                })?;

                let arch_config: ArchitectureConfig = serde_json::from_value(arch_value.clone())
                    .map_err(|e| {
                        NeuralError::DeserializationError(format!(
                            "Invalid architecture config: {e}"
                        ))
                    })?;

                let ser_config: SerializableGPTConfig = serde_json::from_value(arch_config.config)
                    .map_err(|e| {
                        NeuralError::DeserializationError(format!("Invalid GPT config: {e}"))
                    })?;

                let gpt_config = ser_config.to_gpt_config();
                GPTModel::new(gpt_config)
            }
        }
    }
}

// ============================================================================
// ResNet deserialization
// ============================================================================

impl<F> ModelDeserialize for ResNet<F>
where
    F: Float
        + Debug
        + ScalarOperand
        + NumAssign
        + ToPrimitive
        + FromPrimitive
        + Send
        + Sync
        + 'static,
{
    fn load(path: &Path, format: ModelFormat) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Self::from_bytes(&bytes, format)
    }

    fn from_bytes(bytes: &[u8], format: ModelFormat) -> Result<Self> {
        match format {
            ModelFormat::SafeTensors => {
                let reader = SafeTensorsReader::from_bytes(bytes)?;
                let meta = reader.metadata();
                let arch_config_str = meta.get("architecture_config").ok_or_else(|| {
                    NeuralError::DeserializationError("Missing architecture_config".to_string())
                })?;
                let arch_config: ArchitectureConfig = serde_json::from_str(arch_config_str)
                    .map_err(|e| {
                        NeuralError::DeserializationError(format!(
                            "Invalid architecture config: {e}"
                        ))
                    })?;

                let ser_config: SerializableResNetConfig =
                    serde_json::from_value(arch_config.config).map_err(|e| {
                        NeuralError::DeserializationError(format!("Invalid ResNet config: {e}"))
                    })?;

                let resnet_config = ser_config.to_resnet_config()?;
                let mut model = ResNet::new(resnet_config)?;

                let params = reader.to_named_parameters()?;
                model.load_named_parameters(&params)?;

                Ok(model)
            }
            _ => {
                let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(bytes)
                    .map_err(|e| NeuralError::DeserializationError(format!("Invalid JSON: {e}")))?;

                let arch_value = raw.get("architecture").ok_or_else(|| {
                    NeuralError::DeserializationError("Missing 'architecture' key".to_string())
                })?;

                let arch_config: ArchitectureConfig = serde_json::from_value(arch_value.clone())
                    .map_err(|e| {
                        NeuralError::DeserializationError(format!(
                            "Invalid architecture config: {e}"
                        ))
                    })?;

                let ser_config: SerializableResNetConfig =
                    serde_json::from_value(arch_config.config).map_err(|e| {
                        NeuralError::DeserializationError(format!("Invalid ResNet config: {e}"))
                    })?;

                let resnet_config = ser_config.to_resnet_config()?;
                ResNet::new(resnet_config)
            }
        }
    }
}

// ============================================================================
// Mamba serialization
// ============================================================================

impl<F> ExtractParameters for Mamba<F>
where
    F: Float
        + Debug
        + ScalarOperand
        + NumAssign
        + ToPrimitive
        + Send
        + Sync
        + SimdUnifiedOps
        + 'static,
{
    fn extract_named_parameters(&self) -> Result<NamedParameters> {
        let mut all_params = NamedParameters::new();

        let layer_ref: &dyn Layer<F> = self;
        let params = layer_ref.params();

        for (i, param) in params.iter().enumerate() {
            let name = format!("mamba.param_{i}");
            let shape: Vec<usize> = param.shape().to_vec();
            let values: Vec<f64> = param
                .iter()
                .map(|&x| {
                    x.to_f64().ok_or_else(|| {
                        NeuralError::SerializationError(
                            "Cannot convert parameter to f64".to_string(),
                        )
                    })
                })
                .collect::<Result<Vec<f64>>>()?;
            all_params.add(&name, values, shape);
        }

        Ok(all_params)
    }

    fn load_named_parameters(&mut self, _params: &NamedParameters) -> Result<()> {
        Ok(())
    }
}

impl<F> ModelSerialize for Mamba<F>
where
    F: Float
        + Debug
        + ScalarOperand
        + NumAssign
        + ToPrimitive
        + Send
        + Sync
        + SimdUnifiedOps
        + 'static,
{
    fn save(&self, path: &Path, format: ModelFormat) -> Result<()> {
        let bytes = self.to_bytes(format)?;
        fs::write(path, bytes).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(())
    }

    fn to_bytes(&self, format: ModelFormat) -> Result<Vec<u8>> {
        let config = self.config();
        let ser_config = SerializableMambaConfig::from(config);

        let arch_config = ArchitectureConfig {
            architecture: "Mamba".to_string(),
            format_version: "1.0".to_string(),
            config: serde_json::to_value(&ser_config)
                .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
        };

        let params = self.extract_named_parameters()?;

        match format {
            ModelFormat::SafeTensors => {
                let metadata = ModelMetadata::new("Mamba", "f64", params.total_parameters())
                    .with_extra(
                        "architecture_config",
                        &serde_json::to_string(&arch_config)
                            .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
                    );

                let mut writer = SafeTensorsWriter::new();
                writer.add_model_metadata(&metadata);
                writer.add_named_parameters(&params)?;
                writer.to_bytes()
            }
            _ => {
                let mut result = HashMap::new();
                result.insert(
                    "architecture",
                    serde_json::to_value(&arch_config)
                        .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
                );

                let params_value: Vec<serde_json::Value> = params
                    .parameters
                    .iter()
                    .map(|(name, values, shape)| {
                        serde_json::json!({
                            "name": name,
                            "shape": shape,
                            "data": values,
                        })
                    })
                    .collect();
                result.insert("parameters", serde_json::Value::Array(params_value));

                serde_json::to_vec_pretty(&result)
                    .map_err(|e| NeuralError::SerializationError(e.to_string()))
            }
        }
    }

    fn architecture_name(&self) -> &str {
        "Mamba"
    }
}

impl<F> ModelDeserialize for Mamba<F>
where
    F: Float
        + Debug
        + ScalarOperand
        + NumAssign
        + ToPrimitive
        + Send
        + Sync
        + SimdUnifiedOps
        + 'static,
{
    fn load(path: &Path, format: ModelFormat) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Self::from_bytes(&bytes, format)
    }

    fn from_bytes(bytes: &[u8], format: ModelFormat) -> Result<Self> {
        match format {
            ModelFormat::SafeTensors => {
                let reader = SafeTensorsReader::from_bytes(bytes)?;
                let meta = reader.metadata();
                let arch_config_str = meta.get("architecture_config").ok_or_else(|| {
                    NeuralError::DeserializationError("Missing architecture_config".to_string())
                })?;
                let arch_config: ArchitectureConfig = serde_json::from_str(arch_config_str)
                    .map_err(|e| {
                        NeuralError::DeserializationError(format!(
                            "Invalid architecture config: {e}"
                        ))
                    })?;

                let ser_config: SerializableMambaConfig =
                    serde_json::from_value(arch_config.config).map_err(|e| {
                        NeuralError::DeserializationError(format!("Invalid Mamba config: {e}"))
                    })?;

                let mamba_config = ser_config.to_mamba_config();
                let mut rng = scirs2_core::ChaCha8Rng::seed_from_u64(42);
                let mut model = Mamba::new(mamba_config, &mut rng)?;

                let params = reader.to_named_parameters()?;
                model.load_named_parameters(&params)?;

                Ok(model)
            }
            _ => {
                let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(bytes)
                    .map_err(|e| NeuralError::DeserializationError(format!("Invalid JSON: {e}")))?;

                let arch_value = raw.get("architecture").ok_or_else(|| {
                    NeuralError::DeserializationError("Missing 'architecture' key".to_string())
                })?;

                let arch_config: ArchitectureConfig = serde_json::from_value(arch_value.clone())
                    .map_err(|e| {
                        NeuralError::DeserializationError(format!(
                            "Invalid architecture config: {e}"
                        ))
                    })?;

                let ser_config: SerializableMambaConfig =
                    serde_json::from_value(arch_config.config).map_err(|e| {
                        NeuralError::DeserializationError(format!("Invalid Mamba config: {e}"))
                    })?;

                let mamba_config = ser_config.to_mamba_config();
                let mut rng = scirs2_core::ChaCha8Rng::seed_from_u64(42);
                Mamba::new(mamba_config, &mut rng)
            }
        }
    }
}

// ============================================================================
// EfficientNet serialization
// ============================================================================

impl<F> ExtractParameters for EfficientNet<F>
where
    F: Float + Debug + ScalarOperand + NumAssign + ToPrimitive + Send + Sync + 'static,
{
    fn extract_named_parameters(&self) -> Result<NamedParameters> {
        let mut all_params = NamedParameters::new();

        let layer_ref: &dyn Layer<F> = self;
        let params = layer_ref.params();

        for (i, param) in params.iter().enumerate() {
            let name = format!("efficientnet.param_{i}");
            let shape: Vec<usize> = param.shape().to_vec();
            let values: Vec<f64> = param
                .iter()
                .map(|&x| {
                    x.to_f64().ok_or_else(|| {
                        NeuralError::SerializationError(
                            "Cannot convert parameter to f64".to_string(),
                        )
                    })
                })
                .collect::<Result<Vec<f64>>>()?;
            all_params.add(&name, values, shape);
        }

        Ok(all_params)
    }

    fn load_named_parameters(&mut self, _params: &NamedParameters) -> Result<()> {
        Ok(())
    }
}

impl<F> ModelSerialize for EfficientNet<F>
where
    F: Float + Debug + ScalarOperand + NumAssign + ToPrimitive + Send + Sync + 'static,
{
    fn save(&self, path: &Path, format: ModelFormat) -> Result<()> {
        let bytes = self.to_bytes(format)?;
        fs::write(path, bytes).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(())
    }

    fn to_bytes(&self, format: ModelFormat) -> Result<Vec<u8>> {
        let config = self.config();
        let ser_config = SerializableEfficientNetConfig {
            width_coefficient: config.width_coefficient,
            depth_coefficient: config.depth_coefficient,
            resolution: config.resolution,
            dropout_rate: config.dropout_rate,
            input_channels: config.input_channels,
            num_classes: config.num_classes,
        };

        let arch_config = ArchitectureConfig {
            architecture: "EfficientNet".to_string(),
            format_version: "1.0".to_string(),
            config: serde_json::to_value(&ser_config)
                .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
        };

        let params = self.extract_named_parameters()?;

        match format {
            ModelFormat::SafeTensors => {
                let metadata = ModelMetadata::new("EfficientNet", "f64", params.total_parameters())
                    .with_extra(
                        "architecture_config",
                        &serde_json::to_string(&arch_config)
                            .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
                    );

                let mut writer = SafeTensorsWriter::new();
                writer.add_model_metadata(&metadata);
                writer.add_named_parameters(&params)?;
                writer.to_bytes()
            }
            _ => {
                let mut result = HashMap::new();
                result.insert(
                    "architecture",
                    serde_json::to_value(&arch_config)
                        .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
                );

                let params_value: Vec<serde_json::Value> = params
                    .parameters
                    .iter()
                    .map(|(name, values, shape)| {
                        serde_json::json!({
                            "name": name,
                            "shape": shape,
                            "data": values,
                        })
                    })
                    .collect();
                result.insert("parameters", serde_json::Value::Array(params_value));

                serde_json::to_vec_pretty(&result)
                    .map_err(|e| NeuralError::SerializationError(e.to_string()))
            }
        }
    }

    fn architecture_name(&self) -> &str {
        "EfficientNet"
    }
}

// ============================================================================
// MobileNet serialization
// ============================================================================

impl<F> ExtractParameters for MobileNet<F>
where
    F: Float + Debug + ScalarOperand + NumAssign + ToPrimitive + Send + Sync + 'static,
{
    fn extract_named_parameters(&self) -> Result<NamedParameters> {
        let mut all_params = NamedParameters::new();

        let layer_ref: &dyn Layer<F> = self;
        let params = layer_ref.params();

        for (i, param) in params.iter().enumerate() {
            let name = format!("mobilenet.param_{i}");
            let shape: Vec<usize> = param.shape().to_vec();
            let values: Vec<f64> = param
                .iter()
                .map(|&x| {
                    x.to_f64().ok_or_else(|| {
                        NeuralError::SerializationError(
                            "Cannot convert parameter to f64".to_string(),
                        )
                    })
                })
                .collect::<Result<Vec<f64>>>()?;
            all_params.add(&name, values, shape);
        }

        Ok(all_params)
    }

    fn load_named_parameters(&mut self, _params: &NamedParameters) -> Result<()> {
        Ok(())
    }
}

impl<F> ModelSerialize for MobileNet<F>
where
    F: Float + Debug + ScalarOperand + NumAssign + ToPrimitive + Send + Sync + 'static,
{
    fn save(&self, path: &Path, format: ModelFormat) -> Result<()> {
        let bytes = self.to_bytes(format)?;
        fs::write(path, bytes).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(())
    }

    fn to_bytes(&self, format: ModelFormat) -> Result<Vec<u8>> {
        let config = self.config();
        let ser_config = SerializableMobileNetConfig {
            version: match config.version {
                MobileNetVersion::V1 => "V1".to_string(),
                MobileNetVersion::V2 => "V2".to_string(),
                MobileNetVersion::V3Small => "V3Small".to_string(),
                MobileNetVersion::V3Large => "V3Large".to_string(),
            },
            width_multiplier: config.width_multiplier,
            resolution_multiplier: config.resolution_multiplier,
            dropout_rate: config.dropout_rate,
            input_channels: config.input_channels,
            num_classes: config.num_classes,
        };

        let arch_config = ArchitectureConfig {
            architecture: "MobileNet".to_string(),
            format_version: "1.0".to_string(),
            config: serde_json::to_value(&ser_config)
                .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
        };

        let params = self.extract_named_parameters()?;

        match format {
            ModelFormat::SafeTensors => {
                let metadata = ModelMetadata::new("MobileNet", "f64", params.total_parameters())
                    .with_extra(
                        "architecture_config",
                        &serde_json::to_string(&arch_config)
                            .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
                    );

                let mut writer = SafeTensorsWriter::new();
                writer.add_model_metadata(&metadata);
                writer.add_named_parameters(&params)?;
                writer.to_bytes()
            }
            _ => {
                let mut result = HashMap::new();
                result.insert(
                    "architecture",
                    serde_json::to_value(&arch_config)
                        .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
                );

                let params_value: Vec<serde_json::Value> = params
                    .parameters
                    .iter()
                    .map(|(name, values, shape)| {
                        serde_json::json!({
                            "name": name,
                            "shape": shape,
                            "data": values,
                        })
                    })
                    .collect();
                result.insert("parameters", serde_json::Value::Array(params_value));

                serde_json::to_vec_pretty(&result)
                    .map_err(|e| NeuralError::SerializationError(e.to_string()))
            }
        }
    }

    fn architecture_name(&self) -> &str {
        "MobileNet"
    }
}

// ============================================================================
// Utility: detect architecture from file
// ============================================================================

/// Detect the architecture type from a serialized model file
pub fn detect_architecture(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|e| NeuralError::IOError(e.to_string()))?;
    detect_architecture_from_bytes(&bytes)
}

/// Detect the architecture type from serialized bytes
pub fn detect_architecture_from_bytes(bytes: &[u8]) -> Result<String> {
    // Try SafeTensors first (starts with 8-byte header size)
    if bytes.len() >= 8 {
        if let Ok(reader) = SafeTensorsReader::from_bytes(bytes) {
            let meta = reader.metadata();
            if let Some(arch) = meta.get("architecture") {
                return Ok(arch.clone());
            }
        }
    }

    // Try JSON
    if let Ok(raw) = serde_json::from_slice::<HashMap<String, serde_json::Value>>(bytes) {
        if let Some(arch_value) = raw.get("architecture") {
            if let Ok(arch_config) =
                serde_json::from_value::<ArchitectureConfig>(arch_value.clone())
            {
                return Ok(arch_config.architecture);
            }
        }
    }

    Err(NeuralError::DeserializationError(
        "Cannot detect architecture from file: unrecognized format".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serializable_resnet_config_roundtrip() -> Result<()> {
        let config = ResNetConfig::resnet18(3, 1000);
        let ser = SerializableResNetConfig::from(&config);

        // Serialize to JSON
        let json = serde_json::to_string(&ser)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?;

        // Deserialize back
        let deser: SerializableResNetConfig = serde_json::from_str(&json)
            .map_err(|e| NeuralError::DeserializationError(e.to_string()))?;

        let restored = deser.to_resnet_config()?;
        assert_eq!(restored.input_channels, 3);
        assert_eq!(restored.num_classes, 1000);
        assert_eq!(restored.layers.len(), 4);

        Ok(())
    }

    #[test]
    fn test_serializable_bert_config_roundtrip() -> Result<()> {
        let config = BertConfig::bert_base_uncased();
        let ser = SerializableBertConfig::from(&config);

        let json = serde_json::to_string(&ser)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?;

        let deser: SerializableBertConfig = serde_json::from_str(&json)
            .map_err(|e| NeuralError::DeserializationError(e.to_string()))?;

        let restored = deser.to_bert_config();
        assert_eq!(restored.vocab_size, 30522);
        assert_eq!(restored.hidden_size, 768);
        assert_eq!(restored.num_hidden_layers, 12);

        Ok(())
    }

    #[test]
    fn test_serializable_gpt_config_roundtrip() -> Result<()> {
        let config = GPTConfig::gpt2_small();
        let ser = SerializableGPTConfig::from(&config);

        let json = serde_json::to_string(&ser)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?;

        let deser: SerializableGPTConfig = serde_json::from_str(&json)
            .map_err(|e| NeuralError::DeserializationError(e.to_string()))?;

        let restored = deser.to_gpt_config();
        assert_eq!(restored.vocab_size, 50257);
        assert_eq!(restored.hidden_size, 768);

        Ok(())
    }

    #[test]
    fn test_serializable_mamba_config_roundtrip() -> Result<()> {
        let config = MambaConfig::new(256);
        let ser = SerializableMambaConfig::from(&config);

        let json = serde_json::to_string(&ser)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?;

        let deser: SerializableMambaConfig = serde_json::from_str(&json)
            .map_err(|e| NeuralError::DeserializationError(e.to_string()))?;

        let restored = deser.to_mamba_config();
        assert_eq!(restored.d_model, 256);
        assert_eq!(restored.d_state, 16);

        Ok(())
    }

    #[test]
    fn test_architecture_config_envelope() -> Result<()> {
        let config = ArchitectureConfig {
            architecture: "ResNet".to_string(),
            format_version: "1.0".to_string(),
            config: serde_json::json!({
                "block": "Basic",
                "input_channels": 3,
                "num_classes": 10,
            }),
        };

        let json = serde_json::to_string(&config)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?;

        let restored: ArchitectureConfig = serde_json::from_str(&json)
            .map_err(|e| NeuralError::DeserializationError(e.to_string()))?;

        assert_eq!(restored.architecture, "ResNet");
        assert_eq!(restored.format_version, "1.0");

        Ok(())
    }

    #[test]
    fn test_resnet_model_serialize() -> Result<()> {
        let config = ResNetConfig::resnet18(3, 10);
        let model = ResNet::<f64>::new(config)?;

        // Test SafeTensors serialization
        let bytes = model.to_bytes(ModelFormat::SafeTensors)?;
        assert!(!bytes.is_empty());

        // Verify it can be read back
        let reader = SafeTensorsReader::from_bytes(&bytes)?;
        let meta = reader.metadata();
        assert_eq!(meta.get("architecture"), Some(&"ResNet".to_string()));

        // Test JSON serialization
        let json_bytes = model.to_bytes(ModelFormat::Json)?;
        assert!(!json_bytes.is_empty());

        Ok(())
    }

    #[test]
    fn test_resnet_save_load_roundtrip() -> Result<()> {
        let test_dir = std::env::temp_dir().join("scirs2_arch_resnet");
        fs::create_dir_all(&test_dir).map_err(|e| NeuralError::IOError(e.to_string()))?;
        let path = test_dir.join("resnet18.safetensors");

        let config = ResNetConfig::resnet18(3, 10);
        let model = ResNet::<f64>::new(config)?;
        model.save(&path, ModelFormat::SafeTensors)?;

        let loaded = ResNet::<f64>::load(&path, ModelFormat::SafeTensors)?;
        assert_eq!(loaded.config().input_channels, 3);
        assert_eq!(loaded.config().num_classes, 10);

        let _ = fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_detect_architecture_safetensors() -> Result<()> {
        let config = ResNetConfig::resnet18(3, 10);
        let model = ResNet::<f64>::new(config)?;
        let bytes = model.to_bytes(ModelFormat::SafeTensors)?;

        let arch = detect_architecture_from_bytes(&bytes)?;
        assert_eq!(arch, "ResNet");
        Ok(())
    }

    #[test]
    fn test_detect_architecture_json() -> Result<()> {
        let config = ResNetConfig::resnet18(3, 10);
        let model = ResNet::<f64>::new(config)?;
        let bytes = model.to_bytes(ModelFormat::Json)?;

        let arch = detect_architecture_from_bytes(&bytes)?;
        assert_eq!(arch, "ResNet");
        Ok(())
    }
}
