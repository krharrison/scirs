//! Zarr v2/v3 chunked array format support
//!
//! Provides reading and writing of Zarr arrays stored in directory stores:
//!
//! - **Zarr v2**: `.zarray` / `.zgroup` metadata, `"0.1.2"` chunk keys
//! - **Zarr v3**: `zarr.json` metadata, `"c/0/1/2"` chunk keys
//! - Codec pipeline (bytes, transpose, compression via oxiarc-*)
//! - Arbitrary slice reads/writes across chunk boundaries
//!
//! All compression uses pure Rust oxiarc-* crates (COOLJAPAN Policy).

mod array;
mod codecs;
pub mod group;
mod metadata;
mod store;
pub mod types;

pub use array::ZarrArray;
pub use codecs::{
    BytesCodec, Codec, CodecPipeline, Endian, ShuffleCodec, TransposeCodec, ZstdCodec,
};
pub use metadata::{
    ArrayMetadataV2, ArrayMetadataV3, ChunkGrid, CodecMetadata, ConsolidatedMetadata,
    GroupMetadataV2, GroupMetadataV3,
};
pub use store::DirectoryStore;

use std::fmt;

/// Zarr specification version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZarrVersion {
    /// Zarr v2 (.zarray / .zgroup)
    V2,
    /// Zarr v3 (zarr.json)
    V3,
}

impl fmt::Display for ZarrVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZarrVersion::V2 => write!(f, "2"),
            ZarrVersion::V3 => write!(f, "3"),
        }
    }
}

/// Scalar data types supported by Zarr arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DataType {
    /// Boolean
    Bool,
    /// Signed 8-bit integer
    Int8,
    /// Signed 16-bit integer
    Int16,
    /// Signed 32-bit integer
    Int32,
    /// Signed 64-bit integer
    Int64,
    /// Unsigned 8-bit integer
    UInt8,
    /// Unsigned 16-bit integer
    UInt16,
    /// Unsigned 32-bit integer
    UInt32,
    /// Unsigned 64-bit integer
    UInt64,
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
}

impl DataType {
    /// Size of a single element in bytes.
    pub fn byte_size(&self) -> usize {
        match self {
            DataType::Bool | DataType::Int8 | DataType::UInt8 => 1,
            DataType::Int16 | DataType::UInt16 => 2,
            DataType::Int32 | DataType::UInt32 | DataType::Float32 => 4,
            DataType::Int64 | DataType::UInt64 | DataType::Float64 => 8,
        }
    }

    /// Convert to a Zarr v2-style dtype string (e.g. `"<f8"`).
    pub fn to_v2_dtype(&self) -> &'static str {
        match self {
            DataType::Bool => "|b1",
            DataType::Int8 => "|i1",
            DataType::Int16 => "<i2",
            DataType::Int32 => "<i4",
            DataType::Int64 => "<i8",
            DataType::UInt8 => "|u1",
            DataType::UInt16 => "<u2",
            DataType::UInt32 => "<u4",
            DataType::UInt64 => "<u8",
            DataType::Float32 => "<f4",
            DataType::Float64 => "<f8",
        }
    }

    /// Parse a Zarr v2-style dtype string.
    pub fn from_v2_dtype(s: &str) -> crate::error::Result<Self> {
        match s {
            "|b1" => Ok(DataType::Bool),
            "|i1" => Ok(DataType::Int8),
            "<i2" | ">i2" => Ok(DataType::Int16),
            "<i4" | ">i4" => Ok(DataType::Int32),
            "<i8" | ">i8" => Ok(DataType::Int64),
            "|u1" => Ok(DataType::UInt8),
            "<u2" | ">u2" => Ok(DataType::UInt16),
            "<u4" | ">u4" => Ok(DataType::UInt32),
            "<u8" | ">u8" => Ok(DataType::UInt64),
            "<f4" | ">f4" => Ok(DataType::Float32),
            "<f8" | ">f8" => Ok(DataType::Float64),
            _ => Err(crate::error::IoError::FormatError(format!(
                "Unknown Zarr v2 dtype: {s}"
            ))),
        }
    }

    /// Convert to Zarr v3 data_type string.
    pub fn to_v3_name(&self) -> &'static str {
        match self {
            DataType::Bool => "bool",
            DataType::Int8 => "int8",
            DataType::Int16 => "int16",
            DataType::Int32 => "int32",
            DataType::Int64 => "int64",
            DataType::UInt8 => "uint8",
            DataType::UInt16 => "uint16",
            DataType::UInt32 => "uint32",
            DataType::UInt64 => "uint64",
            DataType::Float32 => "float32",
            DataType::Float64 => "float64",
        }
    }

    /// Parse a Zarr v3 data_type name.
    pub fn from_v3_name(s: &str) -> crate::error::Result<Self> {
        match s {
            "bool" => Ok(DataType::Bool),
            "int8" => Ok(DataType::Int8),
            "int16" => Ok(DataType::Int16),
            "int32" => Ok(DataType::Int32),
            "int64" => Ok(DataType::Int64),
            "uint8" => Ok(DataType::UInt8),
            "uint16" => Ok(DataType::UInt16),
            "uint32" => Ok(DataType::UInt32),
            "uint64" => Ok(DataType::UInt64),
            "float32" => Ok(DataType::Float32),
            "float64" => Ok(DataType::Float64),
            _ => Err(crate::error::IoError::FormatError(format!(
                "Unknown Zarr v3 data type: {s}"
            ))),
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_v3_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_byte_size() {
        assert_eq!(DataType::Bool.byte_size(), 1);
        assert_eq!(DataType::Int32.byte_size(), 4);
        assert_eq!(DataType::Float64.byte_size(), 8);
        assert_eq!(DataType::UInt16.byte_size(), 2);
    }

    #[test]
    fn test_data_type_v2_roundtrip() {
        let types = [
            DataType::Bool,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Float32,
            DataType::Float64,
        ];
        for dt in &types {
            let s = dt.to_v2_dtype();
            let parsed = DataType::from_v2_dtype(s).expect("roundtrip should succeed");
            assert_eq!(*dt, parsed);
        }
    }

    #[test]
    fn test_data_type_v3_roundtrip() {
        let types = [
            DataType::Bool,
            DataType::Float32,
            DataType::Float64,
            DataType::Int64,
            DataType::UInt8,
        ];
        for dt in &types {
            let s = dt.to_v3_name();
            let parsed = DataType::from_v3_name(s).expect("roundtrip should succeed");
            assert_eq!(*dt, parsed);
        }
    }

    #[test]
    fn test_data_type_v2_big_endian() {
        assert_eq!(
            DataType::from_v2_dtype(">f8").expect("should parse"),
            DataType::Float64
        );
        assert_eq!(
            DataType::from_v2_dtype(">i4").expect("should parse"),
            DataType::Int32
        );
    }

    #[test]
    fn test_zarr_version_display() {
        assert_eq!(ZarrVersion::V2.to_string(), "2");
        assert_eq!(ZarrVersion::V3.to_string(), "3");
    }
}
