//! ONNX error types for the autograd ONNX interoperability module.

use crate::error::AutogradError;
use std::fmt;
use thiserror::Error;

/// Error type for ONNX operations
#[derive(Debug, Error)]
pub enum OnnxError {
    /// JSON serialization/deserialization error
    #[error("ONNX JSON error: {0}")]
    JsonError(String),

    /// Shape mismatch between tensors
    #[error("ONNX shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<i64>,
        actual: Vec<i64>,
    },

    /// Unsupported ONNX operation type
    #[error("Unsupported ONNX op type: {op_type}")]
    UnsupportedOp { op_type: String },

    /// Invalid attribute on an ONNX node
    #[error("Invalid ONNX attribute '{name}' on node '{node}': {reason}")]
    InvalidAttribute {
        node: String,
        name: String,
        reason: String,
    },

    /// Missing required input or output
    #[error("Missing {kind} '{name}' in ONNX graph")]
    MissingTensor { kind: String, name: String },

    /// Data type conversion error
    #[error("ONNX data type error: {0}")]
    DataTypeError(String),

    /// Graph validation error
    #[error("ONNX graph validation error: {0}")]
    ValidationError(String),

    /// I/O error
    #[error("ONNX I/O error: {0}")]
    IoError(String),

    /// Generic error
    #[error("ONNX error: {0}")]
    Other(String),
}

impl From<serde_json::Error> for OnnxError {
    fn from(e: serde_json::Error) -> Self {
        OnnxError::JsonError(e.to_string())
    }
}

impl From<std::io::Error> for OnnxError {
    fn from(e: std::io::Error) -> Self {
        OnnxError::IoError(e.to_string())
    }
}

impl From<OnnxError> for AutogradError {
    fn from(e: OnnxError) -> Self {
        AutogradError::SerializationError(format!("ONNX: {}", e))
    }
}

/// Result type alias for ONNX operations
pub type OnnxResult<T> = std::result::Result<T, OnnxError>;

/// Display helper for ONNX data type codes
pub(crate) fn data_type_name(code: u32) -> &'static str {
    match code {
        1 => "FLOAT32",
        2 => "UINT8",
        3 => "INT8",
        5 => "INT16",
        6 => "INT32",
        7 => "INT64",
        9 => "BOOL",
        10 => "FLOAT16",
        11 => "FLOAT64",
        12 => "UINT32",
        13 => "UINT64",
        _ => "UNKNOWN",
    }
}
