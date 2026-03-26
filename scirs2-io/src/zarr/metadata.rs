//! Zarr v2 and v3 metadata structures with JSON serialization.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::DataType;
use crate::error::{IoError, Result};

// ────────────────────────────── Zarr v2 ──────────────────────────────────────

/// Zarr v2 `.zarray` metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayMetadataV2 {
    /// Zarr format version (always 2).
    pub zarr_format: u32,
    /// Array shape.
    pub shape: Vec<u64>,
    /// Chunk shape.
    pub chunks: Vec<u64>,
    /// NumPy-style dtype string (e.g. `"<f8"`).
    pub dtype: String,
    /// Compressor specification (null for uncompressed).
    pub compressor: Option<CompressorV2>,
    /// Fill value for uninitialised chunks.
    pub fill_value: serde_json::Value,
    /// Memory layout order: `"C"` (row-major) or `"F"` (column-major).
    pub order: String,
    /// Optional user-defined filters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filters: Option<Vec<serde_json::Value>>,
    /// Dimension separator (default `"."`).
    #[serde(default = "default_dimension_separator")]
    pub dimension_separator: String,
}

fn default_dimension_separator() -> String {
    ".".to_string()
}

/// Compressor specification in Zarr v2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressorV2 {
    /// Compressor identifier (e.g. `"zstd"`, `"zlib"`, `"blosc"`).
    pub id: String,
    /// Compression level (compressor-specific).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<i32>,
    /// Additional configuration.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Zarr v2 `.zgroup` metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupMetadataV2 {
    /// Zarr format version (always 2).
    pub zarr_format: u32,
}

impl GroupMetadataV2 {
    /// Create default v2 group metadata.
    pub fn new() -> Self {
        Self { zarr_format: 2 }
    }
}

impl Default for GroupMetadataV2 {
    fn default() -> Self {
        Self::new()
    }
}

/// Zarr v2 consolidated metadata (`.zmetadata`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedMetadata {
    /// Top-level zarr_format.
    pub zarr_format: u32,
    /// Map of keys (relative paths) to their JSON metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ConsolidatedMetadata {
    /// Create an empty consolidated metadata container.
    pub fn new() -> Self {
        Self {
            zarr_format: 2,
            metadata: HashMap::new(),
        }
    }

    /// Insert array metadata under the given key.
    pub fn insert_array(&mut self, path: &str, meta: &ArrayMetadataV2) -> Result<()> {
        let key = format!("{path}/.zarray");
        let value = serde_json::to_value(meta).map_err(|e| {
            IoError::SerializationError(format!("Failed to serialize array metadata: {e}"))
        })?;
        self.metadata.insert(key, value);
        Ok(())
    }

    /// Insert group metadata under the given key.
    pub fn insert_group(&mut self, path: &str, meta: &GroupMetadataV2) -> Result<()> {
        let key = format!("{path}/.zgroup");
        let value = serde_json::to_value(meta).map_err(|e| {
            IoError::SerializationError(format!("Failed to serialize group metadata: {e}"))
        })?;
        self.metadata.insert(key, value);
        Ok(())
    }

    /// Serialize to JSON bytes.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        serde_json::to_vec_pretty(self).map_err(|e| {
            IoError::SerializationError(format!("Failed to serialize consolidated metadata: {e}"))
        })
    }

    /// Deserialize from JSON bytes.
    pub fn from_json(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(|e| {
            IoError::DeserializationError(format!(
                "Failed to deserialize consolidated metadata: {e}"
            ))
        })
    }
}

impl Default for ConsolidatedMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayMetadataV2 {
    /// Create metadata for a new array.
    pub fn new(
        shape: Vec<u64>,
        chunks: Vec<u64>,
        dtype: DataType,
        compressor: Option<CompressorV2>,
        fill_value: serde_json::Value,
    ) -> Self {
        Self {
            zarr_format: 2,
            shape,
            chunks,
            dtype: dtype.to_v2_dtype().to_string(),
            compressor,
            fill_value,
            order: "C".to_string(),
            filters: None,
            dimension_separator: ".".to_string(),
        }
    }

    /// Serialize to JSON bytes.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        serde_json::to_vec_pretty(self)
            .map_err(|e| IoError::SerializationError(format!("Failed to serialize .zarray: {e}")))
    }

    /// Deserialize from JSON bytes.
    pub fn from_json(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(|e| {
            IoError::DeserializationError(format!("Failed to deserialize .zarray: {e}"))
        })
    }

    /// Parsed data type.
    pub fn data_type(&self) -> Result<DataType> {
        DataType::from_v2_dtype(&self.dtype)
    }
}

// ────────────────────────────── Zarr v3 ──────────────────────────────────────

/// Zarr v3 array or group metadata stored in `zarr.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayMetadataV3 {
    /// Always 3.
    pub zarr_format: u32,
    /// `"array"` or `"group"`.
    pub node_type: String,
    /// Array shape (only for arrays).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shape: Vec<u64>,
    /// Data type name (e.g. `"float64"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_type: Option<String>,
    /// Chunk grid configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_grid: Option<ChunkGrid>,
    /// Chunk key encoding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_key_encoding: Option<ChunkKeyEncoding>,
    /// Fill value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill_value: Option<serde_json::Value>,
    /// Ordered list of codecs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub codecs: Vec<CodecMetadata>,
    /// User attributes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attributes: Option<HashMap<String, serde_json::Value>>,
}

/// Zarr v3 group metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupMetadataV3 {
    /// Always 3.
    pub zarr_format: u32,
    /// Always `"group"`.
    pub node_type: String,
    /// User attributes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attributes: Option<HashMap<String, serde_json::Value>>,
}

impl GroupMetadataV3 {
    /// Create default v3 group metadata.
    pub fn new() -> Self {
        Self {
            zarr_format: 3,
            node_type: "group".to_string(),
            attributes: None,
        }
    }
}

impl Default for GroupMetadataV3 {
    fn default() -> Self {
        Self::new()
    }
}

/// Chunk grid specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkGrid {
    /// Grid name, e.g. `"regular"`.
    pub name: String,
    /// Grid configuration.
    pub configuration: ChunkGridConfig,
}

/// Chunk grid configuration for regular grids.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkGridConfig {
    /// Chunk shape.
    pub chunk_shape: Vec<u64>,
}

/// Chunk key encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkKeyEncoding {
    /// Encoding name, e.g. `"default"` or `"v2"`.
    pub name: String,
    /// Optional separator configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configuration: Option<ChunkKeyEncodingConfig>,
}

/// Chunk key encoding config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkKeyEncodingConfig {
    /// Separator character (default `/`).
    pub separator: String,
}

/// A single codec in the v3 codec pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecMetadata {
    /// Codec name (e.g. `"bytes"`, `"transpose"`, `"zstd"`).
    pub name: String,
    /// Codec-specific configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configuration: Option<serde_json::Value>,
}

impl ArrayMetadataV3 {
    /// Create metadata for a new v3 array.
    pub fn new_array(
        shape: Vec<u64>,
        chunk_shape: Vec<u64>,
        dtype: DataType,
        fill_value: serde_json::Value,
        codecs: Vec<CodecMetadata>,
    ) -> Self {
        Self {
            zarr_format: 3,
            node_type: "array".to_string(),
            shape,
            data_type: Some(dtype.to_v3_name().to_string()),
            chunk_grid: Some(ChunkGrid {
                name: "regular".to_string(),
                configuration: ChunkGridConfig { chunk_shape },
            }),
            chunk_key_encoding: Some(ChunkKeyEncoding {
                name: "default".to_string(),
                configuration: Some(ChunkKeyEncodingConfig {
                    separator: "/".to_string(),
                }),
            }),
            fill_value: Some(fill_value),
            codecs,
            attributes: None,
        }
    }

    /// Serialize to JSON bytes.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        serde_json::to_vec_pretty(self)
            .map_err(|e| IoError::SerializationError(format!("Failed to serialize zarr.json: {e}")))
    }

    /// Deserialize from JSON bytes.
    pub fn from_json(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(|e| {
            IoError::DeserializationError(format!("Failed to deserialize zarr.json: {e}"))
        })
    }

    /// Parsed data type.
    pub fn data_type_parsed(&self) -> Result<DataType> {
        let name = self
            .data_type
            .as_deref()
            .ok_or_else(|| IoError::FormatError("Missing data_type in zarr.json".to_string()))?;
        DataType::from_v3_name(name)
    }

    /// Chunk shape from the chunk grid.
    pub fn chunk_shape(&self) -> Result<&[u64]> {
        self.chunk_grid
            .as_ref()
            .map(|g| g.configuration.chunk_shape.as_slice())
            .ok_or_else(|| IoError::FormatError("Missing chunk_grid in zarr.json".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v2_array_metadata_roundtrip() {
        let meta = ArrayMetadataV2::new(
            vec![100, 200],
            vec![10, 20],
            DataType::Float64,
            None,
            serde_json::json!(0.0),
        );
        let json = meta.to_json().expect("serialize");
        let parsed = ArrayMetadataV2::from_json(&json).expect("deserialize");
        assert_eq!(parsed.shape, vec![100, 200]);
        assert_eq!(parsed.chunks, vec![10, 20]);
        assert_eq!(parsed.dtype, "<f8");
        assert_eq!(parsed.zarr_format, 2);
        assert_eq!(parsed.order, "C");
    }

    #[test]
    fn test_v2_group_metadata() {
        let meta = GroupMetadataV2::new();
        let json = serde_json::to_vec(&meta).expect("serialize");
        let parsed: GroupMetadataV2 = serde_json::from_slice(&json).expect("deserialize");
        assert_eq!(parsed.zarr_format, 2);
    }

    #[test]
    fn test_v3_array_metadata_roundtrip() {
        let meta = ArrayMetadataV3::new_array(
            vec![1000, 2000],
            vec![100, 200],
            DataType::Int32,
            serde_json::json!(0),
            vec![CodecMetadata {
                name: "bytes".to_string(),
                configuration: Some(serde_json::json!({"endian": "little"})),
            }],
        );
        let json = meta.to_json().expect("serialize");
        let parsed = ArrayMetadataV3::from_json(&json).expect("deserialize");
        assert_eq!(parsed.zarr_format, 3);
        assert_eq!(parsed.node_type, "array");
        assert_eq!(parsed.shape, vec![1000, 2000]);
        assert_eq!(parsed.data_type_parsed().expect("dtype"), DataType::Int32);
        assert_eq!(parsed.chunk_shape().expect("chunks"), &[100, 200]);
        assert_eq!(parsed.codecs.len(), 1);
    }

    #[test]
    fn test_v3_group_metadata() {
        let meta = GroupMetadataV3::new();
        let json = serde_json::to_vec(&meta).expect("serialize");
        let parsed: GroupMetadataV3 = serde_json::from_slice(&json).expect("deserialize");
        assert_eq!(parsed.zarr_format, 3);
        assert_eq!(parsed.node_type, "group");
    }

    #[test]
    fn test_v2_metadata_with_compressor() {
        let comp = CompressorV2 {
            id: "zstd".to_string(),
            level: Some(3),
            extra: HashMap::new(),
        };
        let meta = ArrayMetadataV2::new(
            vec![50],
            vec![10],
            DataType::UInt8,
            Some(comp),
            serde_json::json!(255),
        );
        let json = meta.to_json().expect("serialize");
        let parsed = ArrayMetadataV2::from_json(&json).expect("deserialize");
        let c = parsed.compressor.expect("compressor present");
        assert_eq!(c.id, "zstd");
        assert_eq!(c.level, Some(3));
    }

    #[test]
    fn test_consolidated_metadata() {
        let mut cm = ConsolidatedMetadata::new();
        let arr_meta = ArrayMetadataV2::new(
            vec![10],
            vec![5],
            DataType::Float32,
            None,
            serde_json::json!(0.0),
        );
        cm.insert_array("data", &arr_meta).expect("insert array");
        let grp_meta = GroupMetadataV2::new();
        cm.insert_group("", &grp_meta).expect("insert group");

        let json = cm.to_json().expect("serialize");
        let parsed = ConsolidatedMetadata::from_json(&json).expect("deserialize");
        assert!(parsed.metadata.contains_key("data/.zarray"));
        assert!(parsed.metadata.contains_key("/.zgroup"));
    }

    #[test]
    fn test_v2_data_type_parsing() {
        let meta = ArrayMetadataV2::new(
            vec![10],
            vec![5],
            DataType::Int16,
            None,
            serde_json::json!(0),
        );
        assert_eq!(meta.data_type().expect("parse"), DataType::Int16);
    }
}
