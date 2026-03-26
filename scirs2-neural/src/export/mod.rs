//! ONNX-compatible model export and weight conversion utilities.
//!
//! This module provides facilities to export trained neural network models to
//! an ONNX-like interchange format (serialized via `oxicode` or JSON) and to
//! convert weight tensors between SciRS2 and external framework conventions.
//!
//! ## Overview
//!
//! | Sub-module | Purpose |
//! |-----------|---------|
//! [`onnx`] | ONNX graph/model structures and layer exporters |
//! [`weights`] | [`WeightStore`] for framework-agnostic weight I/O |
//!
//! ## Quick-start
//!
//! ```rust
//! use scirs2_neural::export::onnx::{export_linear, export_activation, export_sequential, OnnxModel};
//! use scirs2_neural::export::weights::{WeightStore, WeightFormat};
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! // Build a tiny linear export
//! let w = Array2::<f64>::zeros((4, 8));
//! let b = Array1::<f64>::zeros(4);
//! let (nodes, inits) = export_linear(&w, Some(&b), "x", "y", "fc0");
//! assert_eq!(nodes.len(), 1);
//!
//! // Use WeightStore for weight persistence
//! let mut store = WeightStore::new();
//! store.insert("fc.weight", w.into_dyn());
//! store.insert("fc.bias", b.into_dyn());
//! assert_eq!(store.names().len(), 2);
//! ```

pub mod onnx;
pub mod weights;

pub use onnx::{
    export_activation, export_batchnorm, export_conv2d, export_linear, export_sequential,
    OnnxAttribute, OnnxDataType, OnnxExportable, OnnxGraph, OnnxModel, OnnxNode, OnnxTensor,
    OnnxValueInfo,
};
pub use weights::{pytorch_to_scirs2_names, scirs2_to_pytorch_names, WeightFormat, WeightStore};
