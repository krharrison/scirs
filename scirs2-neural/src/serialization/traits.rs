//! Generic model serialization traits
//!
//! This module provides the `ModelSerialize` and `ModelDeserialize` traits
//! that allow any neural network architecture to be saved to and loaded from disk.
//! These traits work with multiple formats (JSON, SafeTensors, etc.) and handle
//! nested layers, attention heads, and normalization parameters.

use crate::error::Result;
use std::path::Path;

/// Supported serialization formats for model persistence
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// JSON format - human-readable, larger files
    Json,
    /// SafeTensors format - binary, HuggingFace-compatible
    SafeTensors,
    /// CBOR format - binary, compact
    Cbor,
    /// MessagePack format - binary, compact
    MessagePack,
}

/// Trait for serializing a model to disk
///
/// Any neural network architecture that implements this trait can be saved
/// to a file in one of the supported formats. The serialization captures
/// both the model configuration (architecture) and the learned parameters (weights).
///
/// # Example
///
/// ```rust
/// use scirs2_neural::serialization::traits::{ModelSerialize, ModelFormat};
///
/// // ModelSerialize is a trait implemented by model architectures.
/// // Example usage (with a model that implements ModelSerialize):
/// let format = ModelFormat::SafeTensors;
/// assert_eq!(format, ModelFormat::SafeTensors);
/// ```
pub trait ModelSerialize {
    /// Save the model to the specified path in the given format
    ///
    /// This method serializes both the model architecture (configuration)
    /// and all learned parameters (weights, biases, normalization stats, etc.)
    fn save(&self, path: &Path, format: ModelFormat) -> Result<()>;

    /// Serialize the model to bytes in the given format
    ///
    /// This is useful when you want to store the serialized model in memory
    /// or send it over a network rather than writing to disk.
    fn to_bytes(&self, format: ModelFormat) -> Result<Vec<u8>>;

    /// Get the architecture name for this model (e.g., "ResNet", "BERT", "GPT")
    fn architecture_name(&self) -> &str;

    /// Get the model version string
    fn model_version(&self) -> String {
        "0.1.0".to_string()
    }
}

/// Trait for deserializing a model from disk
///
/// Any neural network architecture that implements this trait can be loaded
/// from a file that was previously saved with `ModelSerialize`.
///
/// # Example
///
/// ```rust
/// use scirs2_neural::serialization::traits::{ModelDeserialize, ModelFormat};
///
/// // ModelDeserialize is a trait implemented by model architectures.
/// // Example usage (with a model that implements ModelDeserialize):
/// let format = ModelFormat::Json;
/// assert_eq!(format, ModelFormat::Json);
/// ```
pub trait ModelDeserialize: Sized {
    /// Load the model from the specified path in the given format
    ///
    /// This method deserializes both the model architecture and all
    /// learned parameters, reconstructing a fully functional model.
    fn load(path: &Path, format: ModelFormat) -> Result<Self>;

    /// Deserialize the model from bytes in the given format
    ///
    /// This is useful when loading from a network stream or in-memory buffer.
    fn from_bytes(bytes: &[u8], format: ModelFormat) -> Result<Self>;
}

/// Metadata about a serialized model, stored alongside the weights
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelMetadata {
    /// Architecture name (e.g., "ResNet", "BERT", "GPT")
    pub architecture: String,
    /// Model version
    pub version: String,
    /// Framework version that produced this file
    pub framework_version: String,
    /// Number of parameters in the model
    pub num_parameters: usize,
    /// Data type used for parameters (e.g., "f32", "f64")
    pub dtype: String,
    /// Additional key-value metadata
    pub extra: std::collections::HashMap<String, String>,
}

impl ModelMetadata {
    /// Create new metadata for a model
    pub fn new(architecture: &str, dtype: &str, num_parameters: usize) -> Self {
        Self {
            architecture: architecture.to_string(),
            version: "0.1.0".to_string(),
            framework_version: env!("CARGO_PKG_VERSION").to_string(),
            num_parameters,
            dtype: dtype.to_string(),
            extra: std::collections::HashMap::new(),
        }
    }

    /// Add an extra metadata key-value pair
    pub fn with_extra(mut self, key: &str, value: &str) -> Self {
        self.extra.insert(key.to_string(), value.to_string());
        self
    }
}

/// Information about a single tensor in a serialized model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorInfo {
    /// Name of the tensor (e.g., "layer1.weight", "encoder.attention.query")
    pub name: String,
    /// Data type (e.g., "F32", "F64")
    pub dtype: String,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Byte offset in the data section
    pub data_offset: usize,
    /// Number of bytes for this tensor
    pub byte_length: usize,
}

impl TensorInfo {
    /// Create a new TensorInfo
    pub fn new(
        name: &str,
        dtype: &str,
        shape: Vec<usize>,
        data_offset: usize,
        byte_length: usize,
    ) -> Self {
        Self {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape,
            data_offset,
            byte_length,
        }
    }

    /// Get the total number of elements in this tensor
    pub fn num_elements(&self) -> usize {
        if self.shape.is_empty() {
            0
        } else {
            self.shape.iter().product()
        }
    }
}

/// A named parameter collection that can be extracted from any model
///
/// This provides a uniform interface for accessing model parameters
/// regardless of the underlying architecture.
#[derive(Debug, Clone)]
pub struct NamedParameters {
    /// Ordered list of (name, flattened_f64_values, shape) tuples
    pub parameters: Vec<(String, Vec<f64>, Vec<usize>)>,
}

impl NamedParameters {
    /// Create a new empty NamedParameters collection
    pub fn new() -> Self {
        Self {
            parameters: Vec::new(),
        }
    }

    /// Add a parameter tensor
    pub fn add(&mut self, name: &str, values: Vec<f64>, shape: Vec<usize>) {
        self.parameters.push((name.to_string(), values, shape));
    }

    /// Get the total number of scalar parameters
    pub fn total_parameters(&self) -> usize {
        self.parameters.iter().map(|(_, v, _)| v.len()).sum()
    }

    /// Find a parameter by name
    pub fn get(&self, name: &str) -> Option<&(String, Vec<f64>, Vec<usize>)> {
        self.parameters.iter().find(|(n, _, _)| n == name)
    }

    /// Get the number of named parameter groups
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }
}

impl Default for NamedParameters {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for extracting named parameters from a model
///
/// This trait provides a standardized way to extract all named parameters
/// from any model architecture, enabling format-agnostic serialization.
pub trait ExtractParameters {
    /// Extract all named parameters from the model
    ///
    /// Parameters are returned as named `(String, Vec<f64>, Vec<usize>)` tuples
    /// where the first element is the name (e.g., "encoder.layer.0.attention.query.weight"),
    /// the second is the flattened parameter values, and the third is the shape.
    fn extract_named_parameters(&self) -> Result<NamedParameters>;

    /// Load named parameters into the model
    ///
    /// This method takes a NamedParameters collection and sets the model's
    /// parameters accordingly. Parameter names must match those returned
    /// by `extract_named_parameters`.
    fn load_named_parameters(&mut self, params: &NamedParameters) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_metadata_creation() {
        let metadata = ModelMetadata::new("ResNet", "f32", 11_000_000);
        assert_eq!(metadata.architecture, "ResNet");
        assert_eq!(metadata.dtype, "f32");
        assert_eq!(metadata.num_parameters, 11_000_000);
    }

    #[test]
    fn test_model_metadata_with_extra() {
        let metadata = ModelMetadata::new("BERT", "f32", 110_000_000)
            .with_extra("variant", "base-uncased")
            .with_extra("vocab_size", "30522");
        assert_eq!(
            metadata.extra.get("variant"),
            Some(&"base-uncased".to_string())
        );
        assert_eq!(metadata.extra.get("vocab_size"), Some(&"30522".to_string()));
    }

    #[test]
    fn test_tensor_info() {
        let info = TensorInfo::new("layer1.weight", "F32", vec![768, 3072], 0, 768 * 3072 * 4);
        assert_eq!(info.num_elements(), 768 * 3072);
        assert_eq!(info.byte_length, 768 * 3072 * 4);
    }

    #[test]
    fn test_tensor_info_empty_shape() {
        let info = TensorInfo::new("empty", "F32", vec![], 0, 0);
        assert_eq!(info.num_elements(), 0);
    }

    #[test]
    fn test_named_parameters() {
        let mut params = NamedParameters::new();
        assert!(params.is_empty());
        assert_eq!(params.len(), 0);

        params.add("layer1.weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        params.add("layer1.bias", vec![0.1, 0.2], vec![2]);

        assert_eq!(params.len(), 2);
        assert!(!params.is_empty());
        assert_eq!(params.total_parameters(), 6);

        let found = params.get("layer1.weight");
        assert!(found.is_some());
        let (name, values, shape) = found.expect("parameter should exist");
        assert_eq!(name, "layer1.weight");
        assert_eq!(values, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape, &[2, 2]);

        assert!(params.get("nonexistent").is_none());
    }

    #[test]
    fn test_model_format_enum() {
        let fmt = ModelFormat::SafeTensors;
        assert_eq!(fmt, ModelFormat::SafeTensors);
        assert_ne!(fmt, ModelFormat::Json);

        // Test all variants exist
        let _json = ModelFormat::Json;
        let _st = ModelFormat::SafeTensors;
        let _cbor = ModelFormat::Cbor;
        let _mp = ModelFormat::MessagePack;
    }
}
