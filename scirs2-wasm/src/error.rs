//! Error types for WASM bindings

use thiserror::Error;
use wasm_bindgen::prelude::*;

/// Error types for SciRS2-WASM operations
#[derive(Error, Debug)]
pub enum WasmError {
    /// Array shape mismatch
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape
        expected: Vec<usize>,
        /// Actual shape
        actual: Vec<usize>,
    },

    /// Invalid dimensions
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// Index out of bounds
    #[error("Index out of bounds: {0}")]
    IndexOutOfBounds(String),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Computation error
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Core error from scirs2-core
    #[error("Core error: {0}")]
    CoreError(String),
}

impl From<WasmError> for JsValue {
    fn from(err: WasmError) -> Self {
        // On wasm32 targets, create a proper JS string error.
        // On native targets (used for testing), JsValue::from_str() panics
        // because the JS runtime is unavailable. Use JsValue::NULL as a
        // safe placeholder so error-path tests can verify Result::is_err()
        // without triggering SIGABRT.
        #[cfg(target_arch = "wasm32")]
        {
            JsValue::from_str(&err.to_string())
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = err;
            JsValue::NULL
        }
    }
}

impl From<scirs2_core::error::CoreError> for WasmError {
    fn from(err: scirs2_core::error::CoreError) -> Self {
        WasmError::CoreError(err.to_string())
    }
}

impl From<serde_json::Error> for WasmError {
    fn from(err: serde_json::Error) -> Self {
        WasmError::SerializationError(err.to_string())
    }
}

impl From<serde_wasm_bindgen::Error> for WasmError {
    fn from(err: serde_wasm_bindgen::Error) -> Self {
        WasmError::SerializationError(err.to_string())
    }
}

/// Result type for WASM operations
pub type WasmResult<T> = Result<T, WasmError>;
