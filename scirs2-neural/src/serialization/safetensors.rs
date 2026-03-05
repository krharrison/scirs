//! SafeTensors format implementation
//!
//! This module implements the SafeTensors binary format for model serialization,
//! compatible with HuggingFace's safetensors format. The format is designed for
//! safe and efficient tensor storage without arbitrary code execution risks.
//!
//! ## Format Structure
//!
//! ```text
//! [8 bytes: header_size (u64 LE)]
//! [header_size bytes: JSON header with tensor metadata]
//! [remaining bytes: raw tensor data]
//! ```
//!
//! The JSON header contains a map of tensor names to their metadata:
//! - `dtype`: Data type string (e.g., "F32", "F64")
//! - `shape`: Array of dimension sizes
//! - `data_offsets`: [start, end] byte offsets in the data section
//!
//! An optional `__metadata__` key can hold arbitrary string key-value pairs.

use crate::error::{NeuralError, Result};
use crate::serialization::traits::{ModelMetadata, NamedParameters, TensorInfo};
use scirs2_core::numeric::{Float, ToPrimitive};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::io::{Read as IoRead, Write as IoWrite};
use std::path::Path;

/// Data type identifiers for SafeTensors format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafeTensorsDtype {
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// 16-bit floating point (half precision)
    F16,
    /// 16-bit brain floating point
    BF16,
    /// 32-bit integer
    I32,
    /// 64-bit integer
    I64,
}

impl SafeTensorsDtype {
    /// Get the number of bytes per element
    pub fn element_size(&self) -> usize {
        match self {
            SafeTensorsDtype::F32 => 4,
            SafeTensorsDtype::F64 => 8,
            SafeTensorsDtype::F16 => 2,
            SafeTensorsDtype::BF16 => 2,
            SafeTensorsDtype::I32 => 4,
            SafeTensorsDtype::I64 => 8,
        }
    }

    /// Get the string representation
    pub fn as_str(&self) -> &str {
        match self {
            SafeTensorsDtype::F32 => "F32",
            SafeTensorsDtype::F64 => "F64",
            SafeTensorsDtype::F16 => "F16",
            SafeTensorsDtype::BF16 => "BF16",
            SafeTensorsDtype::I32 => "I32",
            SafeTensorsDtype::I64 => "I64",
        }
    }

    /// Parse from string representation
    pub fn from_str_repr(s: &str) -> Result<Self> {
        match s {
            "F32" => Ok(SafeTensorsDtype::F32),
            "F64" => Ok(SafeTensorsDtype::F64),
            "F16" => Ok(SafeTensorsDtype::F16),
            "BF16" => Ok(SafeTensorsDtype::BF16),
            "I32" => Ok(SafeTensorsDtype::I32),
            "I64" => Ok(SafeTensorsDtype::I64),
            other => Err(NeuralError::DeserializationError(format!(
                "Unknown SafeTensors dtype: {other}"
            ))),
        }
    }
}

/// Entry in the SafeTensors JSON header for a single tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeTensorsHeaderEntry {
    /// Data type string (e.g., "F32", "F64")
    pub dtype: String,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// [start, end] byte offsets into the data section
    pub data_offsets: [usize; 2],
}

/// Complete SafeTensors header including optional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeTensorsHeader {
    /// Tensor entries keyed by tensor name
    #[serde(flatten)]
    pub tensors: HashMap<String, serde_json::Value>,
}

/// Writer for the SafeTensors format
///
/// Collects tensor data and writes it all at once to produce a valid SafeTensors file.
pub struct SafeTensorsWriter {
    /// Collected tensor entries: (name, dtype, shape, raw_bytes)
    tensors: Vec<(String, SafeTensorsDtype, Vec<usize>, Vec<u8>)>,
    /// Optional metadata key-value pairs
    metadata: HashMap<String, String>,
}

impl SafeTensorsWriter {
    /// Create a new SafeTensors writer
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the file
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Add model metadata to the file
    pub fn add_model_metadata(&mut self, metadata: &ModelMetadata) {
        self.metadata
            .insert("architecture".to_string(), metadata.architecture.clone());
        self.metadata
            .insert("version".to_string(), metadata.version.clone());
        self.metadata.insert(
            "framework_version".to_string(),
            metadata.framework_version.clone(),
        );
        self.metadata.insert(
            "num_parameters".to_string(),
            metadata.num_parameters.to_string(),
        );
        self.metadata
            .insert("dtype".to_string(), metadata.dtype.clone());
        for (k, v) in &metadata.extra {
            self.metadata.insert(k.clone(), v.clone());
        }
    }

    /// Add an f32 tensor
    pub fn add_f32_tensor(&mut self, name: &str, data: &[f32], shape: &[usize]) -> Result<()> {
        let expected_elements: usize = shape.iter().product();
        if data.len() != expected_elements {
            return Err(NeuralError::SerializationError(format!(
                "Tensor '{name}': data length {} does not match shape {:?} (expected {expected_elements} elements)",
                data.len(),
                shape
            )));
        }

        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &val in data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        self.tensors.push((
            name.to_string(),
            SafeTensorsDtype::F32,
            shape.to_vec(),
            bytes,
        ));
        Ok(())
    }

    /// Add an f64 tensor
    pub fn add_f64_tensor(&mut self, name: &str, data: &[f64], shape: &[usize]) -> Result<()> {
        let expected_elements: usize = shape.iter().product();
        if data.len() != expected_elements {
            return Err(NeuralError::SerializationError(format!(
                "Tensor '{name}': data length {} does not match shape {:?} (expected {expected_elements} elements)",
                data.len(),
                shape
            )));
        }

        let mut bytes = Vec::with_capacity(data.len() * 8);
        for &val in data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        self.tensors.push((
            name.to_string(),
            SafeTensorsDtype::F64,
            shape.to_vec(),
            bytes,
        ));
        Ok(())
    }

    /// Add a tensor from generic Float values (converts to f64 for storage)
    pub fn add_float_tensor<F: Float + Debug + ToPrimitive>(
        &mut self,
        name: &str,
        data: &[F],
        shape: &[usize],
        use_f32: bool,
    ) -> Result<()> {
        if use_f32 {
            let f32_data: Vec<f32> = data
                .iter()
                .map(|&x| {
                    x.to_f32().ok_or_else(|| {
                        NeuralError::SerializationError(format!(
                            "Cannot convert parameter to f32 in tensor '{name}'"
                        ))
                    })
                })
                .collect::<Result<Vec<f32>>>()?;
            self.add_f32_tensor(name, &f32_data, shape)
        } else {
            let f64_data: Vec<f64> = data
                .iter()
                .map(|&x| {
                    x.to_f64().ok_or_else(|| {
                        NeuralError::SerializationError(format!(
                            "Cannot convert parameter to f64 in tensor '{name}'"
                        ))
                    })
                })
                .collect::<Result<Vec<f64>>>()?;
            self.add_f64_tensor(name, &f64_data, shape)
        }
    }

    /// Add tensors from NamedParameters
    pub fn add_named_parameters(&mut self, params: &NamedParameters) -> Result<()> {
        for (name, values, shape) in &params.parameters {
            self.add_f64_tensor(name, values, shape)?;
        }
        Ok(())
    }

    /// Write the SafeTensors file to disk
    pub fn write_to_file(&self, path: &Path) -> Result<()> {
        let bytes = self.to_bytes()?;
        fs::write(path, bytes).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(())
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        // Build the header JSON
        let mut header_map: HashMap<String, serde_json::Value> = HashMap::new();

        // Add metadata if present
        if !self.metadata.is_empty() {
            let metadata_value = serde_json::to_value(&self.metadata)
                .map_err(|e| NeuralError::SerializationError(e.to_string()))?;
            header_map.insert("__metadata__".to_string(), metadata_value);
        }

        // Calculate offsets for each tensor
        let mut current_offset: usize = 0;
        let mut tensor_offsets: Vec<(usize, usize)> = Vec::with_capacity(self.tensors.len());

        for (_, _, _, bytes) in &self.tensors {
            let start = current_offset;
            let end = start + bytes.len();
            tensor_offsets.push((start, end));
            current_offset = end;
        }

        // Add tensor entries to header
        for (i, (name, dtype, shape, _)) in self.tensors.iter().enumerate() {
            let (start, end) = tensor_offsets[i];
            let entry = SafeTensorsHeaderEntry {
                dtype: dtype.as_str().to_string(),
                shape: shape.clone(),
                data_offsets: [start, end],
            };
            let entry_value = serde_json::to_value(&entry)
                .map_err(|e| NeuralError::SerializationError(e.to_string()))?;
            header_map.insert(name.clone(), entry_value);
        }

        // Serialize header to JSON
        let header_json = serde_json::to_string(&header_map)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?;
        let header_bytes = header_json.as_bytes();
        let header_size = header_bytes.len() as u64;

        // Build the complete buffer
        let total_size = 8 + header_bytes.len() + current_offset;
        let mut buffer = Vec::with_capacity(total_size);

        // Write header size (8 bytes, little-endian u64)
        buffer.extend_from_slice(&header_size.to_le_bytes());

        // Write header JSON
        buffer.extend_from_slice(header_bytes);

        // Write tensor data in order
        for (_, _, _, bytes) in &self.tensors {
            buffer.extend_from_slice(bytes);
        }

        Ok(buffer)
    }
}

impl Default for SafeTensorsWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Reader for the SafeTensors format
///
/// Parses a SafeTensors file and provides access to individual tensors
/// and metadata.
pub struct SafeTensorsReader {
    /// Parsed header entries
    header_entries: HashMap<String, SafeTensorsHeaderEntry>,
    /// Metadata from the file
    metadata: HashMap<String, String>,
    /// Raw data section (everything after the header)
    data: Vec<u8>,
}

impl SafeTensorsReader {
    /// Read a SafeTensors file from disk
    pub fn from_file(path: &Path) -> Result<Self> {
        let file_bytes = fs::read(path).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Self::from_bytes(&file_bytes)
    }

    /// Parse a SafeTensors file from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(NeuralError::DeserializationError(
                "SafeTensors file too small: missing header size".to_string(),
            ));
        }

        // Read header size
        let mut header_size_bytes = [0u8; 8];
        header_size_bytes.copy_from_slice(&bytes[0..8]);
        let header_size = u64::from_le_bytes(header_size_bytes) as usize;

        if bytes.len() < 8 + header_size {
            return Err(NeuralError::DeserializationError(format!(
                "SafeTensors file truncated: expected {} header bytes, got {}",
                header_size,
                bytes.len() - 8
            )));
        }

        // Parse header JSON
        let header_json = std::str::from_utf8(&bytes[8..8 + header_size]).map_err(|e| {
            NeuralError::DeserializationError(format!("Invalid UTF-8 in SafeTensors header: {e}"))
        })?;

        let raw_header: HashMap<String, serde_json::Value> = serde_json::from_str(header_json)
            .map_err(|e| {
                NeuralError::DeserializationError(format!(
                    "Invalid JSON in SafeTensors header: {e}"
                ))
            })?;

        // Separate metadata from tensor entries
        let mut metadata = HashMap::new();
        let mut header_entries = HashMap::new();

        for (key, value) in &raw_header {
            if key == "__metadata__" {
                // Parse metadata
                if let Some(map) = value.as_object() {
                    for (mk, mv) in map {
                        if let Some(s) = mv.as_str() {
                            metadata.insert(mk.clone(), s.to_string());
                        }
                    }
                }
            } else {
                // Parse tensor entry
                let entry: SafeTensorsHeaderEntry =
                    serde_json::from_value(value.clone()).map_err(|e| {
                        NeuralError::DeserializationError(format!(
                            "Invalid tensor entry for '{key}': {e}"
                        ))
                    })?;
                header_entries.insert(key.clone(), entry);
            }
        }

        // Extract data section
        let data = bytes[8 + header_size..].to_vec();

        Ok(Self {
            header_entries,
            metadata,
            data,
        })
    }

    /// Get the metadata from the file
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.header_entries.keys().map(|s| s.as_str()).collect()
    }

    /// Get tensor info for a specific tensor
    pub fn tensor_info(&self, name: &str) -> Result<TensorInfo> {
        let entry = self.header_entries.get(name).ok_or_else(|| {
            NeuralError::DeserializationError(format!(
                "Tensor '{name}' not found in SafeTensors file"
            ))
        })?;

        Ok(TensorInfo::new(
            name,
            &entry.dtype,
            entry.shape.clone(),
            entry.data_offsets[0],
            entry.data_offsets[1] - entry.data_offsets[0],
        ))
    }

    /// Get all tensor infos
    pub fn all_tensor_infos(&self) -> Result<Vec<TensorInfo>> {
        let mut infos = Vec::with_capacity(self.header_entries.len());
        for (name, entry) in &self.header_entries {
            infos.push(TensorInfo::new(
                name,
                &entry.dtype,
                entry.shape.clone(),
                entry.data_offsets[0],
                entry.data_offsets[1] - entry.data_offsets[0],
            ));
        }
        Ok(infos)
    }

    /// Read a tensor as f32 values
    pub fn read_f32_tensor(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let entry = self.header_entries.get(name).ok_or_else(|| {
            NeuralError::DeserializationError(format!("Tensor '{name}' not found"))
        })?;

        let start = entry.data_offsets[0];
        let end = entry.data_offsets[1];

        if end > self.data.len() {
            return Err(NeuralError::DeserializationError(format!(
                "Tensor '{name}' data offsets [{start}, {end}] exceed data section size {}",
                self.data.len()
            )));
        }

        let raw_bytes = &self.data[start..end];

        let values = match entry.dtype.as_str() {
            "F32" => {
                if !raw_bytes.len().is_multiple_of(4) {
                    return Err(NeuralError::DeserializationError(format!(
                        "Tensor '{name}': F32 data length {} is not a multiple of 4",
                        raw_bytes.len()
                    )));
                }
                raw_bytes
                    .chunks_exact(4)
                    .map(|chunk| {
                        let mut bytes = [0u8; 4];
                        bytes.copy_from_slice(chunk);
                        f32::from_le_bytes(bytes)
                    })
                    .collect()
            }
            "F64" => {
                if !raw_bytes.len().is_multiple_of(8) {
                    return Err(NeuralError::DeserializationError(format!(
                        "Tensor '{name}': F64 data length {} is not a multiple of 8",
                        raw_bytes.len()
                    )));
                }
                raw_bytes
                    .chunks_exact(8)
                    .map(|chunk| {
                        let mut bytes = [0u8; 8];
                        bytes.copy_from_slice(chunk);
                        f64::from_le_bytes(bytes) as f32
                    })
                    .collect()
            }
            dtype => {
                return Err(NeuralError::DeserializationError(format!(
                    "Cannot read tensor '{name}' as f32: unsupported dtype '{dtype}'"
                )));
            }
        };

        Ok((values, entry.shape.clone()))
    }

    /// Read a tensor as f64 values
    pub fn read_f64_tensor(&self, name: &str) -> Result<(Vec<f64>, Vec<usize>)> {
        let entry = self.header_entries.get(name).ok_or_else(|| {
            NeuralError::DeserializationError(format!("Tensor '{name}' not found"))
        })?;

        let start = entry.data_offsets[0];
        let end = entry.data_offsets[1];

        if end > self.data.len() {
            return Err(NeuralError::DeserializationError(format!(
                "Tensor '{name}' data offsets [{start}, {end}] exceed data section size {}",
                self.data.len()
            )));
        }

        let raw_bytes = &self.data[start..end];

        let values = match entry.dtype.as_str() {
            "F64" => {
                if !raw_bytes.len().is_multiple_of(8) {
                    return Err(NeuralError::DeserializationError(format!(
                        "Tensor '{name}': F64 data length {} is not a multiple of 8",
                        raw_bytes.len()
                    )));
                }
                raw_bytes
                    .chunks_exact(8)
                    .map(|chunk| {
                        let mut bytes = [0u8; 8];
                        bytes.copy_from_slice(chunk);
                        f64::from_le_bytes(bytes)
                    })
                    .collect()
            }
            "F32" => {
                if !raw_bytes.len().is_multiple_of(4) {
                    return Err(NeuralError::DeserializationError(format!(
                        "Tensor '{name}': F32 data length {} is not a multiple of 4",
                        raw_bytes.len()
                    )));
                }
                raw_bytes
                    .chunks_exact(4)
                    .map(|chunk| {
                        let mut bytes = [0u8; 4];
                        bytes.copy_from_slice(chunk);
                        f32::from_le_bytes(bytes) as f64
                    })
                    .collect()
            }
            dtype => {
                return Err(NeuralError::DeserializationError(format!(
                    "Cannot read tensor '{name}' as f64: unsupported dtype '{dtype}'"
                )));
            }
        };

        Ok((values, entry.shape.clone()))
    }

    /// Read all tensors as NamedParameters (f64)
    pub fn to_named_parameters(&self) -> Result<NamedParameters> {
        let mut params = NamedParameters::new();
        // Sort tensor names for deterministic order
        let mut names: Vec<&str> = self.tensor_names();
        names.sort();

        for name in names {
            let (values, shape) = self.read_f64_tensor(name)?;
            params.add(name, values, shape);
        }
        Ok(params)
    }

    /// Get the number of tensors in the file
    pub fn num_tensors(&self) -> usize {
        self.header_entries.len()
    }

    /// Check if a tensor exists by name
    pub fn has_tensor(&self, name: &str) -> bool {
        self.header_entries.contains_key(name)
    }
}

/// Validate that a file at the given path is a valid SafeTensors file
///
/// This performs a lightweight check: reads and validates the header
/// without loading all tensor data into memory.
pub fn validate_safetensors_file(path: &Path) -> Result<Vec<TensorInfo>> {
    let mut file = fs::File::open(path).map_err(|e| NeuralError::IOError(e.to_string()))?;

    // Read header size
    let mut header_size_bytes = [0u8; 8];
    file.read_exact(&mut header_size_bytes)
        .map_err(|e| NeuralError::IOError(format!("Failed to read header size: {e}")))?;
    let header_size = u64::from_le_bytes(header_size_bytes) as usize;

    // Read header
    let mut header_buf = vec![0u8; header_size];
    file.read_exact(&mut header_buf)
        .map_err(|e| NeuralError::IOError(format!("Failed to read header: {e}")))?;

    let header_json = std::str::from_utf8(&header_buf)
        .map_err(|e| NeuralError::DeserializationError(format!("Invalid UTF-8 in header: {e}")))?;

    let raw_header: HashMap<String, serde_json::Value> = serde_json::from_str(header_json)
        .map_err(|e| NeuralError::DeserializationError(format!("Invalid JSON in header: {e}")))?;

    let mut infos = Vec::new();
    for (key, value) in &raw_header {
        if key == "__metadata__" {
            continue;
        }
        let entry: SafeTensorsHeaderEntry = serde_json::from_value(value.clone()).map_err(|e| {
            NeuralError::DeserializationError(format!("Invalid tensor entry '{key}': {e}"))
        })?;

        infos.push(TensorInfo::new(
            key,
            &entry.dtype,
            entry.shape.clone(),
            entry.data_offsets[0],
            entry.data_offsets[1] - entry.data_offsets[0],
        ));
    }

    Ok(infos)
}

/// Write NamedParameters to a SafeTensors file with optional metadata
pub fn write_named_parameters(
    path: &Path,
    params: &NamedParameters,
    metadata: Option<&ModelMetadata>,
) -> Result<()> {
    let mut writer = SafeTensorsWriter::new();
    if let Some(meta) = metadata {
        writer.add_model_metadata(meta);
    }
    writer.add_named_parameters(params)?;
    writer.write_to_file(path)
}

/// Read NamedParameters from a SafeTensors file
pub fn read_named_parameters(path: &Path) -> Result<NamedParameters> {
    let reader = SafeTensorsReader::from_file(path)?;
    reader.to_named_parameters()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_dtype() {
        assert_eq!(SafeTensorsDtype::F32.element_size(), 4);
        assert_eq!(SafeTensorsDtype::F64.element_size(), 8);
        assert_eq!(SafeTensorsDtype::F16.element_size(), 2);
        assert_eq!(SafeTensorsDtype::BF16.element_size(), 2);
        assert_eq!(SafeTensorsDtype::I32.element_size(), 4);
        assert_eq!(SafeTensorsDtype::I64.element_size(), 8);

        assert_eq!(SafeTensorsDtype::F32.as_str(), "F32");
        assert_eq!(SafeTensorsDtype::F64.as_str(), "F64");

        let parsed = SafeTensorsDtype::from_str_repr("F32");
        assert!(parsed.is_ok());
        assert_eq!(parsed.expect("should parse"), SafeTensorsDtype::F32);

        let bad = SafeTensorsDtype::from_str_repr("INVALID");
        assert!(bad.is_err());
    }

    #[test]
    fn test_safetensors_roundtrip_f32() -> Result<()> {
        let test_dir = std::env::temp_dir().join("scirs2_safetensors_test_f32");
        fs::create_dir_all(&test_dir).map_err(|e| NeuralError::IOError(e.to_string()))?;
        let file_path = test_dir.join("test_f32.safetensors");

        // Write
        let mut writer = SafeTensorsWriter::new();
        writer.add_metadata("test_key", "test_value");
        writer.add_f32_tensor("weight", &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
        writer.add_f32_tensor("bias", &[0.1f32, 0.2, 0.3], &[3])?;
        writer.write_to_file(&file_path)?;

        // Read
        let reader = SafeTensorsReader::from_file(&file_path)?;
        assert_eq!(reader.num_tensors(), 2);
        assert!(reader.has_tensor("weight"));
        assert!(reader.has_tensor("bias"));
        assert!(!reader.has_tensor("nonexistent"));

        let meta = reader.metadata();
        assert_eq!(meta.get("test_key"), Some(&"test_value".to_string()));

        let (w_data, w_shape) = reader.read_f32_tensor("weight")?;
        assert_eq!(w_shape, vec![2, 3]);
        assert_eq!(w_data.len(), 6);
        assert!((w_data[0] - 1.0f32).abs() < 1e-6);
        assert!((w_data[5] - 6.0f32).abs() < 1e-6);

        let (b_data, b_shape) = reader.read_f32_tensor("bias")?;
        assert_eq!(b_shape, vec![3]);
        assert_eq!(b_data.len(), 3);
        assert!((b_data[0] - 0.1f32).abs() < 1e-6);

        // Cleanup
        let _ = fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_safetensors_roundtrip_f64() -> Result<()> {
        let test_dir = std::env::temp_dir().join("scirs2_safetensors_test_f64");
        fs::create_dir_all(&test_dir).map_err(|e| NeuralError::IOError(e.to_string()))?;
        let file_path = test_dir.join("test_f64.safetensors");

        let mut writer = SafeTensorsWriter::new();
        writer.add_f64_tensor("layer1.weight", &[1.0f64, 2.0, 3.0, 4.0], &[2, 2])?;
        writer.add_f64_tensor("layer1.bias", &[0.5f64, -0.5], &[2])?;
        writer.write_to_file(&file_path)?;

        let reader = SafeTensorsReader::from_file(&file_path)?;
        let (data, shape) = reader.read_f64_tensor("layer1.weight")?;
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);

        let (bias, bias_shape) = reader.read_f64_tensor("layer1.bias")?;
        assert_eq!(bias_shape, vec![2]);
        assert!((bias[0] - 0.5).abs() < 1e-12);
        assert!((bias[1] - (-0.5)).abs() < 1e-12);

        let _ = fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_safetensors_named_parameters_roundtrip() -> Result<()> {
        let test_dir = std::env::temp_dir().join("scirs2_safetensors_named");
        fs::create_dir_all(&test_dir).map_err(|e| NeuralError::IOError(e.to_string()))?;
        let file_path = test_dir.join("named_params.safetensors");

        let mut params = NamedParameters::new();
        params.add(
            "encoder.layer.0.weight",
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        );
        params.add("encoder.layer.0.bias", vec![0.1, 0.2], vec![2]);
        params.add(
            "decoder.weight",
            vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![2, 3],
        );

        let metadata = ModelMetadata::new("TestModel", "f64", params.total_parameters());

        write_named_parameters(&file_path, &params, Some(&metadata))?;

        let loaded = read_named_parameters(&file_path)?;
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.total_parameters(), params.total_parameters());

        // Check a specific parameter
        let (_, values, shape) = loaded
            .get("encoder.layer.0.weight")
            .ok_or_else(|| NeuralError::DeserializationError("parameter not found".to_string()))?;
        assert_eq!(values, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape, &[2, 2]);

        let _ = fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_safetensors_validation() -> Result<()> {
        let test_dir = std::env::temp_dir().join("scirs2_safetensors_validate");
        fs::create_dir_all(&test_dir).map_err(|e| NeuralError::IOError(e.to_string()))?;
        let file_path = test_dir.join("validate.safetensors");

        let mut writer = SafeTensorsWriter::new();
        writer.add_f32_tensor("a", &[1.0f32, 2.0], &[2])?;
        writer.add_f32_tensor("b", &[3.0f32, 4.0, 5.0], &[3])?;
        writer.write_to_file(&file_path)?;

        let infos = validate_safetensors_file(&file_path)?;
        assert_eq!(infos.len(), 2);

        let _ = fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_safetensors_cross_dtype_read() -> Result<()> {
        let test_dir = std::env::temp_dir().join("scirs2_safetensors_cross");
        fs::create_dir_all(&test_dir).map_err(|e| NeuralError::IOError(e.to_string()))?;
        let file_path = test_dir.join("cross_dtype.safetensors");

        // Write as f32
        let mut writer = SafeTensorsWriter::new();
        writer.add_f32_tensor("x", &[1.5f32, 2.5, 3.5], &[3])?;
        writer.write_to_file(&file_path)?;

        // Read as f64
        let reader = SafeTensorsReader::from_file(&file_path)?;
        let (data, shape) = reader.read_f64_tensor("x")?;
        assert_eq!(shape, vec![3]);
        assert!((data[0] - 1.5).abs() < 1e-6);
        assert!((data[1] - 2.5).abs() < 1e-6);
        assert!((data[2] - 3.5).abs() < 1e-6);

        let _ = fs::remove_dir_all(&test_dir);
        Ok(())
    }

    #[test]
    fn test_safetensors_to_bytes_and_back() -> Result<()> {
        let mut writer = SafeTensorsWriter::new();
        writer.add_metadata("format", "safetensors");
        writer.add_f32_tensor("tensor1", &[1.0f32, 2.0, 3.0], &[3])?;

        let bytes = writer.to_bytes()?;

        let reader = SafeTensorsReader::from_bytes(&bytes)?;
        assert_eq!(reader.num_tensors(), 1);
        assert!(reader.has_tensor("tensor1"));

        let meta = reader.metadata();
        assert_eq!(meta.get("format"), Some(&"safetensors".to_string()));

        let (data, shape) = reader.read_f32_tensor("tensor1")?;
        assert_eq!(shape, vec![3]);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);

        Ok(())
    }

    #[test]
    fn test_safetensors_error_on_truncated() {
        // Too small
        let result = SafeTensorsReader::from_bytes(&[0u8; 4]);
        assert!(result.is_err());

        // Header size says more than available
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&100u64.to_le_bytes()); // says 100 bytes header
        bytes.extend_from_slice(&[0u8; 10]); // but only 10 bytes
        let result = SafeTensorsReader::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_safetensors_shape_mismatch_error() {
        let mut writer = SafeTensorsWriter::new();
        // Shape says 3 elements but we provide 4
        let result = writer.add_f32_tensor("bad", &[1.0f32, 2.0, 3.0, 4.0], &[3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_safetensors_model_metadata() -> Result<()> {
        let metadata = ModelMetadata::new("BERT", "f32", 110_000_000).with_extra("variant", "base");

        let mut writer = SafeTensorsWriter::new();
        writer.add_model_metadata(&metadata);
        writer.add_f32_tensor("dummy", &[0.0f32], &[1])?;

        let bytes = writer.to_bytes()?;
        let reader = SafeTensorsReader::from_bytes(&bytes)?;

        let meta = reader.metadata();
        assert_eq!(meta.get("architecture"), Some(&"BERT".to_string()));
        assert_eq!(meta.get("dtype"), Some(&"f32".to_string()));
        assert_eq!(meta.get("num_parameters"), Some(&"110000000".to_string()));
        assert_eq!(meta.get("variant"), Some(&"base".to_string()));

        Ok(())
    }
}
