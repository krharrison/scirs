//! ModelSerializer trait and implementations for non-Sequential neural architectures.
//!
//! This module provides a unified, HuggingFace-compatible serialization interface
//! for complex neural network architectures including ResNet and BERT. The serialization
//! format uses SafeTensors for weights and JSON for configuration, stored together in
//! a single directory (config.json + weights.safetensors) or as a single .safetensors
//! file with embedded configuration metadata.
//!
//! ## Naming conventions
//!
//! ### ResNet
//! - `conv1.weight`, `bn1.weight`, `bn1.bias`
//! - `layer1.0.conv1.weight`, `layer1.0.bn1.weight`, `layer1.0.bn1.bias`
//! - `layer1.0.downsample.0.weight` (conv in skip connection)
//! - `layer1.0.downsample.1.weight` (bn in skip connection)
//! - `fc.weight`, `fc.bias`
//!
//! ### BERT
//! - `embeddings.word_embeddings.weight`
//! - `embeddings.LayerNorm.weight`, `embeddings.LayerNorm.bias`
//! - `encoder.layer.0.attention.self.query.weight`
//! - `encoder.layer.0.attention.output.dense.weight`
//! - `encoder.layer.0.intermediate.dense.weight`
//! - `encoder.layer.0.output.dense.weight`
//! - `pooler.dense.weight`, `pooler.dense.bias`

use crate::error::{NeuralError, Result};
use crate::models::architectures::{
    BertConfig, BertModel, ResNet, ResNetBlock, ResNetConfig, ResNetLayer,
};
use crate::serialization::architecture::{
    ArchitectureConfig, SerializableBertConfig, SerializableResNetConfig,
};
use crate::serialization::safetensors::{SafeTensorsReader, SafeTensorsWriter};
use crate::serialization::traits::{ModelMetadata, NamedParameters};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use scirs2_core::simd_ops::SimdUnifiedOps;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::path::Path;

// ============================================================================
// ModelSerializer trait
// ============================================================================

/// A comprehensive serialization trait for neural network models.
///
/// This trait provides HuggingFace-compatible save/load functionality for any
/// neural network architecture. It operates on named parameter maps, enabling:
/// - Partial loading (graceful handling of added/removed layers)
/// - Cross-format compatibility (SafeTensors, JSON)
/// - Directory-based serialization (config.json + weights.safetensors)
///
/// # Example (conceptual)
///
/// ```rust,no_run
/// use scirs2_neural::serialization::model_serializer::ModelSerializer;
/// use scirs2_neural::models::architectures::ResNet;
/// use std::path::Path;
///
/// let model = ResNet::<f64>::resnet18(3, 100).expect("failed to create model");
/// model.save_model(Path::new("/tmp/myresnet")).expect("failed to save model");
///
/// let mut loaded = ResNet::<f64>::resnet18(3, 100).expect("failed to create model");
/// loaded.load_model(Path::new("/tmp/myresnet")).expect("failed to load model");
/// ```
pub trait ModelSerializer<
    F: Float + Debug + ScalarOperand + NumAssign + ToPrimitive + FromPrimitive + 'static,
>
{
    /// Get the model configuration as a JSON value for persistence.
    fn get_config(&self) -> Result<serde_json::Value>;

    /// Get all named parameters as a flat ordered list.
    ///
    /// Names follow HuggingFace/PyTorch conventions for interoperability.
    fn named_params(&self) -> Result<Vec<(String, Array<F, IxDyn>)>>;

    /// Load parameters from a named map.
    ///
    /// Parameter names not present in the model are silently ignored, enabling
    /// graceful forward/backward compatibility when adding or removing layers.
    fn load_params(&mut self, params: &HashMap<String, Array<F, IxDyn>>) -> Result<()>;

    /// Save the complete model to a directory.
    ///
    /// Creates two files:
    /// - `config.json` — architecture configuration
    /// - `weights.safetensors` — all named parameter tensors
    ///
    /// The directory is created if it does not exist.
    fn save_model(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir).map_err(|e| NeuralError::IOError(e.to_string()))?;

        // Write config.json
        let config_value = self.get_config()?;
        let config_json = serde_json::to_string_pretty(&config_value)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?;
        fs::write(dir.join("config.json"), config_json)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;

        // Write weights.safetensors
        let named = self.named_params()?;
        let total_params: usize = named.iter().map(|(_, a)| a.len()).sum();

        // Build NamedParameters for the writer
        let mut np = NamedParameters::new();
        for (name, arr) in &named {
            let shape: Vec<usize> = arr.shape().to_vec();
            let values: Vec<f64> = arr
                .iter()
                .map(|&x| {
                    x.to_f64().ok_or_else(|| {
                        NeuralError::SerializationError(
                            "Cannot convert parameter to f64".to_string(),
                        )
                    })
                })
                .collect::<Result<Vec<f64>>>()?;
            np.add(name, values, shape);
        }

        let metadata = ModelMetadata::new("model", "f64", total_params).with_extra(
            "config",
            &serde_json::to_string(&config_value)
                .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
        );

        let mut writer = SafeTensorsWriter::new();
        writer.add_model_metadata(&metadata);
        writer.add_named_parameters(&np)?;
        writer.write_to_file(&dir.join("weights.safetensors"))?;

        Ok(())
    }

    /// Load the complete model from a directory.
    ///
    /// Reads `weights.safetensors` and restores all parameters by name.
    /// The `config.json` is not used during loading (call `get_config` separately
    /// to reconstruct the model from scratch before calling this method).
    fn load_model(&mut self, dir: &Path) -> Result<()> {
        let weights_path = dir.join("weights.safetensors");
        let reader = SafeTensorsReader::from_file(&weights_path)?;

        // Read all tensors and build a name -> array map
        let mut params_map: HashMap<String, Array<F, IxDyn>> = HashMap::new();
        for name in reader.tensor_names() {
            let (values_f64, shape) = reader.read_f64_tensor(name)?;
            let f_values: Vec<F> = values_f64
                .iter()
                .map(|&x| {
                    F::from(x).ok_or_else(|| {
                        NeuralError::DeserializationError(format!(
                            "Cannot convert {x} to target float type"
                        ))
                    })
                })
                .collect::<Result<Vec<F>>>()?;
            let arr = Array::from_shape_vec(IxDyn(&shape), f_values)?;
            params_map.insert(name.to_string(), arr);
        }

        self.load_params(&params_map)
    }
}

// ============================================================================
// ResNet implementation
// ============================================================================

impl<F> ModelSerializer<F> for ResNet<F>
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
    fn get_config(&self) -> Result<serde_json::Value> {
        let ser_config = SerializableResNetConfig::from(self.config());
        let arch = ArchitectureConfig {
            architecture: "ResNet".to_string(),
            format_version: "1.0".to_string(),
            config: serde_json::to_value(&ser_config)
                .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
        };
        serde_json::to_value(&arch).map_err(|e| NeuralError::SerializationError(e.to_string()))
    }

    fn named_params(&self) -> Result<Vec<(String, Array<F, IxDyn>)>> {
        self.extract_named_params()
    }

    fn load_params(&mut self, params: &HashMap<String, Array<F, IxDyn>>) -> Result<()> {
        self.load_named_params(params)
    }
}

// ============================================================================
// BertModel implementation
// ============================================================================

impl<F> ModelSerializer<F> for BertModel<F>
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
    fn get_config(&self) -> Result<serde_json::Value> {
        let ser_config = SerializableBertConfig::from(self.config());
        let arch = ArchitectureConfig {
            architecture: "BERT".to_string(),
            format_version: "1.0".to_string(),
            config: serde_json::to_value(&ser_config)
                .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
        };
        serde_json::to_value(&arch).map_err(|e| NeuralError::SerializationError(e.to_string()))
    }

    fn named_params(&self) -> Result<Vec<(String, Array<F, IxDyn>)>> {
        self.extract_named_params()
    }

    fn load_params(&mut self, params: &HashMap<String, Array<F, IxDyn>>) -> Result<()> {
        self.load_named_params(params)
    }
}

// ============================================================================
// Helper: convert NamedParameters → HashMap<String, Array<F, IxDyn>>
// ============================================================================

/// Convert a `NamedParameters` collection (f64 values) to a typed HashMap.
pub fn named_parameters_to_map<F>(
    params: &NamedParameters,
) -> Result<HashMap<String, Array<F, IxDyn>>>
where
    F: Float + FromPrimitive + 'static,
{
    let mut map = HashMap::new();
    for (name, values, shape) in &params.parameters {
        let f_values: Vec<F> = values
            .iter()
            .map(|&x| {
                F::from(x).ok_or_else(|| {
                    NeuralError::DeserializationError(format!(
                        "Cannot convert {x} to target float type"
                    ))
                })
            })
            .collect::<Result<Vec<F>>>()?;
        let arr = Array::from_shape_vec(IxDyn(shape), f_values)?;
        map.insert(name.clone(), arr);
    }
    Ok(map)
}

// ============================================================================
// Standalone save/load helpers
// ============================================================================

/// Save a ResNet model to a directory using the `ModelSerializer` interface.
pub fn save_resnet<F>(model: &ResNet<F>, dir: &Path) -> Result<()>
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
    model.save_model(dir)
}

/// Load ResNet weights from a directory using the `ModelSerializer` interface.
pub fn load_resnet<F>(model: &mut ResNet<F>, dir: &Path) -> Result<()>
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
    model.load_model(dir)
}

/// Save a BertModel to a directory using the `ModelSerializer` interface.
pub fn save_bert<F>(model: &BertModel<F>, dir: &Path) -> Result<()>
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
    model.save_model(dir)
}

/// Load BertModel weights from a directory using the `ModelSerializer` interface.
pub fn load_bert<F>(model: &mut BertModel<F>, dir: &Path) -> Result<()>
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
    model.load_model(dir)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::ResNetConfig;

    #[test]
    fn test_resnet_extract_named_params() -> Result<()> {
        let model = ResNet::<f64>::resnet18(3, 10)?;
        let params = model.extract_named_params()?;

        // Should have at minimum conv1 and fc params
        assert!(!params.is_empty(), "ResNet should have parameters");

        // Check that names are present
        let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            names.contains(&"conv1.weight"),
            "Should have conv1.weight, got: {:?}",
            &names[..names.len().min(5)]
        );
        assert!(names.contains(&"fc.weight"), "Should have fc.weight");
        assert!(names.contains(&"fc.bias"), "Should have fc.bias");
        assert!(names.contains(&"bn1.weight"), "Should have bn1.weight");
        assert!(names.contains(&"bn1.bias"), "Should have bn1.bias");

        Ok(())
    }

    #[test]
    fn test_resnet_save_load_roundtrip() -> Result<()> {
        let test_dir = std::env::temp_dir().join("scirs2_resnet_serialization_test");
        fs::create_dir_all(&test_dir).map_err(|e| NeuralError::IOError(e.to_string()))?;

        // Create and save model
        let model = ResNet::<f64>::resnet18(3, 10)?;
        model.save_model(&test_dir)?;

        // Verify files exist
        assert!(
            test_dir.join("config.json").exists(),
            "config.json should exist"
        );
        assert!(
            test_dir.join("weights.safetensors").exists(),
            "weights.safetensors should exist"
        );

        // Load into a fresh model
        let mut loaded_model = ResNet::<f64>::resnet18(3, 10)?;
        loaded_model.load_model(&test_dir)?;

        // Extract params from both and compare
        let original_params = model.extract_named_params()?;
        let loaded_params = loaded_model.extract_named_params()?;

        assert_eq!(
            original_params.len(),
            loaded_params.len(),
            "Parameter count mismatch after roundtrip"
        );

        for ((orig_name, orig_arr), (load_name, load_arr)) in
            original_params.iter().zip(loaded_params.iter())
        {
            assert_eq!(orig_name, load_name, "Parameter name mismatch");
            assert_eq!(
                orig_arr.shape(),
                load_arr.shape(),
                "Shape mismatch for {orig_name}"
            );

            // Check values are close (they should be exact for f64 roundtrip)
            let max_diff = orig_arr
                .iter()
                .zip(load_arr.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_diff < 1e-10,
                "Value mismatch for {orig_name}: max_diff = {max_diff}"
            );
        }

        // Cleanup
        let _ = fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_resnet_get_config() -> Result<()> {
        let model = ResNet::<f64>::resnet18(3, 100)?;
        let config = model.get_config()?;

        assert!(config.get("architecture").is_some());
        let arch = config["architecture"]
            .as_str()
            .expect("architecture should be a string");
        assert_eq!(arch, "ResNet");

        Ok(())
    }

    #[test]
    fn test_resnet_partial_load_graceful() -> Result<()> {
        // Test that loading an empty map doesn't error (graceful compatibility)
        let mut model = ResNet::<f64>::resnet18(3, 10)?;
        let empty_map: HashMap<String, Array<f64, IxDyn>> = HashMap::new();
        let result = model.load_params(&empty_map);
        assert!(
            result.is_ok(),
            "Loading empty param map should succeed gracefully"
        );
        Ok(())
    }

    #[test]
    fn test_resnet_named_params_no_duplicates() -> Result<()> {
        let model = ResNet::<f64>::resnet18(3, 10)?;
        let params = model.extract_named_params()?;

        let mut seen = std::collections::HashSet::new();
        for (name, _) in &params {
            assert!(
                seen.insert(name.clone()),
                "Duplicate parameter name: {name}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_bert_extract_named_params() -> Result<()> {
        // Use a tiny custom BERT for speed
        let config = BertConfig::custom(100, 32, 2, 4);
        let model = BertModel::<f64>::new(config)?;
        let params = model.extract_named_params()?;

        assert!(!params.is_empty(), "BERT should have parameters");

        let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();

        // Embedding parameters
        assert!(
            names.contains(&"embeddings.word_embeddings.weight"),
            "Should have word embeddings, got names: {:?}",
            &names[..names.len().min(10)]
        );
        assert!(names.contains(&"embeddings.position_embeddings.weight"));
        assert!(names.contains(&"embeddings.token_type_embeddings.weight"));
        assert!(names.contains(&"embeddings.LayerNorm.weight"));
        assert!(names.contains(&"embeddings.LayerNorm.bias"));

        // Encoder layer 0 attention
        assert!(names.contains(&"encoder.layer.0.attention.self.query.weight"));
        assert!(names.contains(&"encoder.layer.0.attention.self.key.weight"));
        assert!(names.contains(&"encoder.layer.0.attention.self.value.weight"));
        assert!(names.contains(&"encoder.layer.0.attention.output.dense.weight"));
        assert!(names.contains(&"encoder.layer.0.attention.output.LayerNorm.weight"));

        // Feed-forward
        assert!(names.contains(&"encoder.layer.0.intermediate.dense.weight"));
        assert!(names.contains(&"encoder.layer.0.output.dense.weight"));
        assert!(names.contains(&"encoder.layer.0.output.LayerNorm.weight"));

        // Pooler
        assert!(names.contains(&"pooler.dense.weight"));
        assert!(names.contains(&"pooler.dense.bias"));

        Ok(())
    }

    #[test]
    fn test_bert_save_load_roundtrip() -> Result<()> {
        let test_dir = std::env::temp_dir().join("scirs2_bert_serialization_test");
        fs::create_dir_all(&test_dir).map_err(|e| NeuralError::IOError(e.to_string()))?;

        // Tiny BERT for speed
        let config = BertConfig::custom(100, 32, 2, 4);
        let model = BertModel::<f64>::new(config.clone())?;
        model.save_model(&test_dir)?;

        // Verify files
        assert!(test_dir.join("config.json").exists());
        assert!(test_dir.join("weights.safetensors").exists());

        // Load into fresh model
        let mut loaded_model = BertModel::<f64>::new(config)?;
        loaded_model.load_model(&test_dir)?;

        // Compare parameters
        let original_params = model.extract_named_params()?;
        let loaded_params = loaded_model.extract_named_params()?;

        assert_eq!(
            original_params.len(),
            loaded_params.len(),
            "BERT parameter count mismatch after roundtrip"
        );

        // Build map for name-based lookup
        let loaded_map: HashMap<String, &Array<f64, IxDyn>> =
            loaded_params.iter().map(|(n, a)| (n.clone(), a)).collect();

        for (name, orig_arr) in &original_params {
            let load_arr = loaded_map.get(name).ok_or_else(|| {
                NeuralError::DeserializationError(format!("Missing parameter: {name}"))
            })?;

            let max_diff = orig_arr
                .iter()
                .zip(load_arr.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_diff < 1e-10,
                "Value mismatch for {name}: max_diff = {max_diff}"
            );
        }

        // Cleanup
        let _ = fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_bert_no_duplicate_param_names() -> Result<()> {
        let config = BertConfig::custom(100, 32, 2, 4);
        let model = BertModel::<f64>::new(config)?;
        let params = model.extract_named_params()?;

        let mut seen = std::collections::HashSet::new();
        for (name, _) in &params {
            assert!(
                seen.insert(name.clone()),
                "Duplicate BERT parameter name: {name}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_bert_partial_load_graceful() -> Result<()> {
        let config = BertConfig::custom(100, 32, 1, 4);
        let mut model = BertModel::<f64>::new(config)?;
        let empty_map: HashMap<String, Array<f64, IxDyn>> = HashMap::new();
        let result = model.load_params(&empty_map);
        assert!(
            result.is_ok(),
            "Loading empty param map should succeed gracefully"
        );
        Ok(())
    }

    #[test]
    fn test_bert_cross_version_compatibility() -> Result<()> {
        // Save a 2-layer BERT, load into a 1-layer BERT
        // Extra parameters from the 2-layer model should be ignored
        let test_dir = std::env::temp_dir().join("scirs2_bert_cross_version_test");
        fs::create_dir_all(&test_dir).map_err(|e| NeuralError::IOError(e.to_string()))?;

        let config_2_layers = BertConfig::custom(100, 32, 2, 4);
        let model_2l = BertModel::<f64>::new(config_2_layers)?;
        model_2l.save_model(&test_dir)?;

        // Load into 1-layer model — layer 1 params won't match any key and are ignored
        let config_1_layer = BertConfig::custom(100, 32, 1, 4);
        let mut model_1l = BertModel::<f64>::new(config_1_layer)?;
        let result = model_1l.load_model(&test_dir);
        assert!(
            result.is_ok(),
            "Cross-version load (2-layer into 1-layer) should succeed gracefully: {:?}",
            result
        );

        // Cleanup
        let _ = fs::remove_dir_all(&test_dir);
        Ok(())
    }
}
