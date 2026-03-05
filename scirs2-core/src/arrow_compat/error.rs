//! Error types for Arrow interoperability
//!
//! Provides specialized error types for Arrow ↔ ndarray conversions,
//! IPC serialization, and schema operations.

use crate::error::{CoreError, ErrorContext};
use std::fmt;

/// Errors that can occur during Arrow interoperability operations
#[derive(Debug)]
pub enum ArrowCompatError {
    /// Type mismatch between expected and actual Arrow data types
    TypeMismatch { expected: String, actual: String },
    /// Shape mismatch when converting between Arrow and ndarray
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// Column index out of bounds in a RecordBatch
    ColumnOutOfBounds { index: usize, num_columns: usize },
    /// Column name not found in a RecordBatch
    ColumnNotFound { name: String },
    /// Null values encountered where not expected
    NullValuesPresent {
        null_count: usize,
        total_count: usize,
    },
    /// Error from the underlying Arrow library
    ArrowError(arrow::error::ArrowError),
    /// I/O error during IPC operations
    IoError(std::io::Error),
    /// Schema validation error
    SchemaError(String),
    /// Zero-copy operation not possible
    ZeroCopyNotPossible(String),
    /// Inconsistent column lengths in a RecordBatch
    InconsistentColumnLengths {
        expected_len: usize,
        column_index: usize,
        column_len: usize,
    },
}

impl fmt::Display for ArrowCompatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeMismatch { expected, actual } => {
                write!(
                    f,
                    "Arrow type mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "Shape mismatch: expected {:?}, got {:?}",
                    expected, actual
                )
            }
            Self::ColumnOutOfBounds { index, num_columns } => {
                write!(
                    f,
                    "Column index {} out of bounds (num_columns={})",
                    index, num_columns
                )
            }
            Self::ColumnNotFound { name } => {
                write!(f, "Column '{}' not found in RecordBatch", name)
            }
            Self::NullValuesPresent {
                null_count,
                total_count,
            } => {
                write!(
                    f,
                    "Found {} null values out of {} total (use nullable conversion instead)",
                    null_count, total_count
                )
            }
            Self::ArrowError(e) => write!(f, "Arrow error: {}", e),
            Self::IoError(e) => write!(f, "I/O error: {}", e),
            Self::SchemaError(msg) => write!(f, "Schema error: {}", msg),
            Self::ZeroCopyNotPossible(reason) => {
                write!(f, "Zero-copy not possible: {}", reason)
            }
            Self::InconsistentColumnLengths {
                expected_len,
                column_index,
                column_len,
            } => {
                write!(
                    f,
                    "Inconsistent column lengths: expected {} rows, column {} has {} rows",
                    expected_len, column_index, column_len
                )
            }
        }
    }
}

impl std::error::Error for ArrowCompatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ArrowError(e) => Some(e),
            Self::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<arrow::error::ArrowError> for ArrowCompatError {
    fn from(e: arrow::error::ArrowError) -> Self {
        Self::ArrowError(e)
    }
}

impl From<std::io::Error> for ArrowCompatError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

impl From<ArrowCompatError> for CoreError {
    fn from(e: ArrowCompatError) -> Self {
        CoreError::ComputationError(ErrorContext::new(format!("Arrow interop: {}", e)))
    }
}

/// Result type for Arrow interoperability operations
pub type ArrowResult<T> = Result<T, ArrowCompatError>;
