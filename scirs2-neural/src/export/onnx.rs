//! ONNX-like model representation and export utilities.
//!
//! This module provides pure-Rust data structures that mirror the ONNX protobuf
//! schema (nodes, graphs, tensors, value-info) without requiring any C library.
//! Serialisation uses `oxicode` for compact binary interchange and `serde_json`
//! for human-readable JSON.
//!
//! # ONNX compatibility notes
//!
//! - Default opset version: **17** (matches ONNX 1.13 / ONNX Runtime 1.15).
//! - Only float32 weights are stored in [`OnnxTensor`].  Float64 sources are
//!   downcast transparently during export.
//! - Dynamic batch dimensions are represented as `None` in [`OnnxValueInfo::shape`].

use crate::error::{NeuralError, Result};
use oxicode::{config as oxicode_config, serde as oxicode_serde};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ndarray imports via scirs2-core re-exports
use scirs2_core::ndarray::{Array1, Array2, Array4};

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// Supported ONNX element data types (subset used by this exporter).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i32)]
#[derive(Default)]
pub enum OnnxDataType {
    /// 32-bit IEEE-754 float (ONNX type 1)
    #[default]
    Float32 = 1,
    /// 32-bit signed integer (ONNX type 6)
    Int32 = 6,
    /// 64-bit signed integer (ONNX type 7)
    Int64 = 7,
    /// 64-bit IEEE-754 float (ONNX type 11)
    Float64 = 11,
}

/// Attribute value attached to an [`OnnxNode`].
///
/// Follows ONNX `AttributeProto` semantics.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OnnxAttribute {
    /// Single float scalar.
    Float(f32),
    /// Single int scalar.
    Int(i64),
    /// String attribute.
    String(String),
    /// Repeated floats.
    Floats(Vec<f32>),
    /// Repeated ints.
    Ints(Vec<i64>),
}

/// A single compute node in the ONNX graph (corresponds to `NodeProto`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxNode {
    /// ONNX operator type string (e.g. `"Gemm"`, `"Conv"`, `"Relu"`).
    pub op_type: String,
    /// Human-readable name (unique within the graph).
    pub name: String,
    /// Names of input tensors consumed by this node.
    pub inputs: Vec<String>,
    /// Names of output tensors produced by this node.
    pub outputs: Vec<String>,
    /// Operator attributes (kernel size, dilations, activations, …).
    pub attributes: HashMap<String, OnnxAttribute>,
}

impl OnnxNode {
    /// Construct a node with no attributes.
    pub fn new(
        op_type: impl Into<String>,
        name: impl Into<String>,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) -> Self {
        Self {
            op_type: op_type.into(),
            name: name.into(),
            inputs,
            outputs,
            attributes: HashMap::new(),
        }
    }

    /// Add an attribute and return `self` for builder-style chaining.
    pub fn with_attr(mut self, key: impl Into<String>, value: OnnxAttribute) -> Self {
        self.attributes.insert(key.into(), value);
        self
    }
}

/// A named constant tensor stored in the graph (corresponds to `TensorProto`).
///
/// Only float32 data is kept; callers should downcast f64 weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxTensor {
    /// Tensor name (must match an [`OnnxNode`] input name).
    pub name: String,
    /// Element type tag.
    pub data_type: OnnxDataType,
    /// Dimension sizes in row-major order.
    pub dims: Vec<i64>,
    /// Flat float32 payload (populated when `data_type == Float32`).
    pub float_data: Vec<f32>,
    /// Flat int64 payload (populated when `data_type == Int64`).
    pub int64_data: Vec<i64>,
}

impl OnnxTensor {
    /// Build an `OnnxTensor` from a flat slice of f64 values.
    ///
    /// Values are cast to f32 for wire-format compatibility.
    pub fn from_f64_slice(name: impl Into<String>, dims: Vec<i64>, data: &[f64]) -> Self {
        Self {
            name: name.into(),
            data_type: OnnxDataType::Float32,
            dims,
            float_data: data.iter().map(|&v| v as f32).collect(),
            int64_data: Vec::new(),
        }
    }

    /// Build an `OnnxTensor` directly from f32 values (no cast required).
    pub fn from_f32_slice(name: impl Into<String>, dims: Vec<i64>, data: &[f32]) -> Self {
        Self {
            name: name.into(),
            data_type: OnnxDataType::Float32,
            dims,
            float_data: data.to_vec(),
            int64_data: Vec::new(),
        }
    }

    /// Return the total number of elements (product of dims).
    pub fn numel(&self) -> usize {
        self.dims
            .iter()
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1)
    }
}

/// Typed tensor description used for graph inputs/outputs (corresponds to
/// `ValueInfoProto`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxValueInfo {
    /// Tensor name.
    pub name: String,
    /// Element type.
    pub data_type: OnnxDataType,
    /// Shape; `None` entries indicate dynamic (symbolic) dimensions.
    pub shape: Vec<Option<i64>>,
}

impl OnnxValueInfo {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, data_type: OnnxDataType, shape: Vec<Option<i64>>) -> Self {
        Self {
            name: name.into(),
            data_type,
            shape,
        }
    }
}

/// A complete ONNX compute graph (`GraphProto`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxGraph {
    /// Ordered list of compute nodes.
    pub nodes: Vec<OnnxNode>,
    /// Graph input descriptors.
    pub inputs: Vec<OnnxValueInfo>,
    /// Graph output descriptors.
    pub outputs: Vec<OnnxValueInfo>,
    /// Constant tensors (model weights and biases).
    pub initializers: Vec<OnnxTensor>,
}

impl OnnxGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            initializers: Vec::new(),
        }
    }
}

impl Default for OnnxGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Top-level ONNX model wrapper (`ModelProto`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxModel {
    /// Compute graph.
    pub graph: OnnxGraph,
    /// ONNX opset version (default: 17).
    pub opset_version: i64,
    /// ONNX IR version (default: 8).
    pub ir_version: i64,
    /// Framework that produced this model.
    pub producer_name: String,
    /// Caller-assigned model version integer.
    pub model_version: i64,
}

impl Default for OnnxModel {
    fn default() -> Self {
        Self {
            graph: OnnxGraph::new(),
            opset_version: 17,
            ir_version: 8,
            producer_name: "scirs2-neural".to_string(),
            model_version: 1,
        }
    }
}

impl OnnxModel {
    /// Create a new model with default metadata and the given graph.
    pub fn new(graph: OnnxGraph) -> Self {
        Self {
            graph,
            ..Default::default()
        }
    }

    // ------------------------------------------------------------------
    // Serialization helpers
    // ------------------------------------------------------------------

    /// Serialise the model to compact oxicode bytes.
    ///
    /// The binary layout uses `oxicode`'s `serde` API which produces a
    /// self-describing, SIMD-optimised payload.  Use [`from_bytes`] to
    /// deserialise.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let cfg = oxicode_config::standard();
        oxicode_serde::encode_to_vec(self, cfg)
            .map_err(|e| NeuralError::SerializationError(format!("oxicode encode error: {e}")))
    }

    /// Deserialise a model that was produced by [`to_bytes`].
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let cfg = oxicode_config::standard();
        oxicode_serde::decode_owned_from_slice(data, cfg)
            .map(|(model, _)| model)
            .map_err(|e| NeuralError::DeserializationError(format!("oxicode decode error: {e}")))
    }

    /// Serialise to pretty-printed JSON (useful for inspection / debugging).
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| NeuralError::SerializationError(format!("JSON encode error: {e}")))
    }

    /// Deserialise from JSON produced by [`to_json`].
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| NeuralError::DeserializationError(format!("JSON decode error: {e}")))
    }
}

// ---------------------------------------------------------------------------
// OnnxExportable trait
// ---------------------------------------------------------------------------

/// Implemented by neural-network layer types that can produce ONNX nodes.
pub trait OnnxExportable {
    /// Emit the ONNX compute node(s) for this layer.
    ///
    /// * `input_name`  — name of the tensor flowing *into* this layer.
    /// * `output_name` — name of the tensor produced by this layer.
    /// * `prefix`      — namespace prefix for generated weight tensor names.
    fn to_onnx_nodes(&self, input_name: &str, output_name: &str, prefix: &str) -> Vec<OnnxNode>;

    /// Emit the weight [`OnnxTensor`]s (initializers) for this layer.
    fn to_onnx_initializers(&self, prefix: &str) -> Vec<OnnxTensor>;
}

// ---------------------------------------------------------------------------
// Free-standing layer exporters
// ---------------------------------------------------------------------------

/// Export a fully-connected (`Gemm`) layer.
///
/// `weights` has shape `[out_features, in_features]` following PyTorch/ONNX
/// convention.  The produced node uses `transB=1` so that the weight matrix
/// stored as `[out, in]` is transposed during the matrix-multiply.
///
/// Returns `(nodes, initializers)` ready for insertion into an [`OnnxGraph`].
pub fn export_linear(
    weights: &Array2<f64>,
    bias: Option<&Array1<f64>>,
    input_name: &str,
    output_name: &str,
    prefix: &str,
) -> (Vec<OnnxNode>, Vec<OnnxTensor>) {
    let w_name = format!("{prefix}.weight");
    let b_name = format!("{prefix}.bias");

    let out_features = weights.nrows() as i64;
    let in_features = weights.ncols() as i64;

    // Build weight initializer
    let w_flat: Vec<f64> = weights.iter().copied().collect();
    let w_tensor = OnnxTensor::from_f64_slice(&w_name, vec![out_features, in_features], &w_flat);

    let mut node_inputs = vec![input_name.to_string(), w_name.clone()];
    let mut initializers = vec![w_tensor];

    // Optionally include bias
    if let Some(b) = bias {
        let b_flat: Vec<f64> = b.iter().copied().collect();
        let b_tensor = OnnxTensor::from_f64_slice(&b_name, vec![out_features], &b_flat);
        initializers.push(b_tensor);
        node_inputs.push(b_name.clone());
    }

    let node = OnnxNode::new(
        "Gemm",
        format!("{prefix}/Gemm"),
        node_inputs,
        vec![output_name.to_string()],
    )
    .with_attr("transB", OnnxAttribute::Int(1))
    .with_attr("alpha", OnnxAttribute::Float(1.0))
    .with_attr("beta", OnnxAttribute::Float(1.0));

    (vec![node], initializers)
}

/// Export a 2-D convolution (`Conv`) layer.
///
/// `weights` has ONNX layout `[out_channels, in_channels, kH, kW]`.
/// `stride` and `padding` are `[H, W]` pairs.
///
/// Returns `(nodes, initializers)`.
pub fn export_conv2d(
    weights: &Array4<f64>,
    bias: Option<&Array1<f64>>,
    stride: &[usize],
    padding: &[usize],
    input_name: &str,
    output_name: &str,
    prefix: &str,
) -> (Vec<OnnxNode>, Vec<OnnxTensor>) {
    let w_name = format!("{prefix}.weight");
    let b_name = format!("{prefix}.bias");

    let shape = weights.shape();
    let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let w_flat: Vec<f64> = weights.iter().copied().collect();
    let w_tensor = OnnxTensor::from_f64_slice(&w_name, dims, &w_flat);

    let strides_attr: Vec<i64> = stride.iter().map(|&s| s as i64).collect();
    let pads_onnx: Vec<i64> = padding
        .iter()
        .chain(padding.iter())
        .map(|&p| p as i64)
        .collect(); // [top, left, bottom, right]

    let mut node_inputs = vec![input_name.to_string(), w_name.clone()];
    let mut initializers = vec![w_tensor];

    if let Some(b) = bias {
        let out_channels = shape[0] as i64;
        let b_flat: Vec<f64> = b.iter().copied().collect();
        let b_tensor = OnnxTensor::from_f64_slice(&b_name, vec![out_channels], &b_flat);
        initializers.push(b_tensor);
        node_inputs.push(b_name.clone());
    }

    let node = OnnxNode::new(
        "Conv",
        format!("{prefix}/Conv"),
        node_inputs,
        vec![output_name.to_string()],
    )
    .with_attr("strides", OnnxAttribute::Ints(strides_attr))
    .with_attr("pads", OnnxAttribute::Ints(pads_onnx));

    (vec![node], initializers)
}

/// Export an elementwise activation function.
///
/// Supported `kind` values: `"relu"`, `"sigmoid"`, `"tanh"`, `"gelu"`,
/// `"leaky_relu"`, `"elu"`, `"selu"`, `"softmax"`, `"log_softmax"`.
///
/// Unknown kinds fall back to `"Relu"` with a warning attribute.
pub fn export_activation(kind: &str, input_name: &str, output_name: &str) -> OnnxNode {
    let (op_type, extra): (&str, Option<(&str, OnnxAttribute)>) = match kind.to_lowercase().as_str()
    {
        "relu" => ("Relu", None),
        "sigmoid" => ("Sigmoid", None),
        "tanh" => ("Tanh", None),
        "gelu" => ("Gelu", None),
        "leaky_relu" => ("LeakyRelu", Some(("alpha", OnnxAttribute::Float(0.01)))),
        "elu" => ("Elu", Some(("alpha", OnnxAttribute::Float(1.0)))),
        "selu" => ("Selu", None),
        "softmax" => ("Softmax", Some(("axis", OnnxAttribute::Int(-1)))),
        "log_softmax" => ("LogSoftmax", Some(("axis", OnnxAttribute::Int(-1)))),
        unknown => {
            let mut node = OnnxNode::new(
                "Relu",
                format!("{unknown}/fallback_Relu"),
                vec![input_name.to_string()],
                vec![output_name.to_string()],
            );
            node.attributes.insert(
                "_scirs2_unsupported_activation".to_string(),
                OnnxAttribute::String(unknown.to_string()),
            );
            return node;
        }
    };

    let mut node = OnnxNode::new(
        op_type,
        format!("{input_name}/{op_type}"),
        vec![input_name.to_string()],
        vec![output_name.to_string()],
    );

    if let Some((key, val)) = extra {
        node.attributes.insert(key.to_string(), val);
    }

    node
}

/// Export a batch-normalisation layer.
///
/// Produces a single `BatchNormalization` node with four weight initializers:
/// `scale` (γ), `bias` (β), `mean` (running mean), `var` (running variance).
///
/// `epsilon` defaults to `1e-5` if `None`.
///
/// Returns `(nodes, initializers)`.
pub fn export_batchnorm(
    scale: &[f64],
    bias: &[f64],
    mean: &[f64],
    var: &[f64],
    epsilon: Option<f32>,
    input_name: &str,
    output_name: &str,
    prefix: &str,
) -> (Vec<OnnxNode>, Vec<OnnxTensor>) {
    let num_features = scale.len() as i64;
    let eps = epsilon.unwrap_or(1e-5_f32);

    let scale_name = format!("{prefix}.scale");
    let bias_name = format!("{prefix}.bias");
    let mean_name = format!("{prefix}.mean");
    let var_name = format!("{prefix}.var");

    let initializers = vec![
        OnnxTensor::from_f64_slice(&scale_name, vec![num_features], scale),
        OnnxTensor::from_f64_slice(&bias_name, vec![num_features], bias),
        OnnxTensor::from_f64_slice(&mean_name, vec![num_features], mean),
        OnnxTensor::from_f64_slice(&var_name, vec![num_features], var),
    ];

    let node = OnnxNode::new(
        "BatchNormalization",
        format!("{prefix}/BatchNormalization"),
        vec![
            input_name.to_string(),
            scale_name,
            bias_name,
            mean_name,
            var_name,
        ],
        vec![output_name.to_string()],
    )
    .with_attr("epsilon", OnnxAttribute::Float(eps));

    (vec![node], initializers)
}

// ---------------------------------------------------------------------------
// Sequential model exporter
// ---------------------------------------------------------------------------

/// Assemble an [`OnnxModel`] from a list of pre-exported layer segments.
///
/// Each entry in `layers` is `(layer_name, nodes, initializers)`.  Tensor
/// names are assigned automatically: the graph input is `"input_0"` and
/// intermediate activations follow the convention `"{layer_name}_out"`.
///
/// `input_shape` describes the graph input (use `None` for dynamic / batch
/// dimensions).
///
/// ```rust
/// use scirs2_neural::export::onnx::{export_linear, export_activation, export_sequential};
/// use scirs2_core::ndarray::Array2;
///
/// let w1 = Array2::<f64>::zeros((64, 784));
/// let (n1, i1) = export_linear(&w1, None, "input_0", "fc0_out", "fc0");
/// let act1 = export_activation("relu", "fc0_out", "act0_out");
///
/// let w2 = Array2::<f64>::zeros((10, 64));
/// let (n2, i2) = export_linear(&w2, None, "act0_out", "output_0", "fc1");
///
/// let layers = vec![
///     ("fc0".to_string(), n1, i1),
///     ("act0".to_string(), vec![act1], vec![]),
///     ("fc1".to_string(), n2, i2),
/// ];
///
/// let model = export_sequential(&layers, &[None, Some(784)]);
/// assert_eq!(model.graph.nodes.len(), 3);
/// assert_eq!(model.opset_version, 17);
/// ```
pub fn export_sequential(
    layers: &[(String, Vec<OnnxNode>, Vec<OnnxTensor>)],
    input_shape: &[Option<i64>],
) -> OnnxModel {
    let mut graph = OnnxGraph::new();

    // Graph input
    graph.inputs.push(OnnxValueInfo::new(
        "input_0",
        OnnxDataType::Float32,
        input_shape.to_vec(),
    ));

    // Collect all nodes and initializers
    let mut last_output = "input_0".to_string();
    for (layer_name, nodes, inits) in layers {
        graph.initializers.extend(inits.iter().cloned());
        for node in nodes {
            graph.nodes.push(node.clone());
        }
        // Track last output produced by this layer's final node
        if let Some(last_node) = nodes.last() {
            if let Some(out) = last_node.outputs.first() {
                last_output = out.clone();
            } else {
                last_output = format!("{layer_name}_out");
            }
        }
    }

    // Graph output
    graph.outputs.push(OnnxValueInfo::new(
        last_output,
        OnnxDataType::Float32,
        vec![None],
    ));

    OnnxModel::new(graph)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2, Array4};

    // -----------------------------------------------------------------------
    // Node construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_onnx_activation_node_relu() {
        let node = export_activation("relu", "x", "y");
        assert_eq!(node.op_type, "Relu");
        assert_eq!(node.inputs, vec!["x".to_string()]);
        assert_eq!(node.outputs, vec!["y".to_string()]);
    }

    #[test]
    fn test_onnx_activation_node_sigmoid() {
        let node = export_activation("sigmoid", "x", "y");
        assert_eq!(node.op_type, "Sigmoid");
    }

    #[test]
    fn test_onnx_activation_node_tanh() {
        let node = export_activation("tanh", "x", "y");
        assert_eq!(node.op_type, "Tanh");
    }

    #[test]
    fn test_onnx_activation_node_softmax() {
        let node = export_activation("softmax", "x", "y");
        assert_eq!(node.op_type, "Softmax");
        assert!(node.attributes.contains_key("axis"));
    }

    #[test]
    fn test_onnx_activation_node_unknown_fallback() {
        let node = export_activation("crelu_custom", "x", "y");
        // Falls back to Relu but records the unknown name
        assert_eq!(node.op_type, "Relu");
        assert!(node
            .attributes
            .contains_key("_scirs2_unsupported_activation"));
    }

    // -----------------------------------------------------------------------
    // Linear (Gemm) exporter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_onnx_linear_node_no_bias() {
        let w = Array2::<f64>::zeros((4, 8));
        let (nodes, inits) = export_linear(&w, None, "x", "y", "fc");
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].op_type, "Gemm");
        // Only weight initializer (no bias)
        assert_eq!(inits.len(), 1);
        assert_eq!(inits[0].dims, vec![4_i64, 8_i64]);
        assert_eq!(inits[0].float_data.len(), 32);
    }

    #[test]
    fn test_onnx_linear_node_with_bias() {
        let w = Array2::<f64>::zeros((4, 8));
        let b = Array1::<f64>::zeros(4);
        let (nodes, inits) = export_linear(&w, Some(&b), "x", "y", "fc");
        assert_eq!(nodes.len(), 1);
        assert_eq!(inits.len(), 2); // weight + bias
                                    // Bias is the second initializer
        assert_eq!(inits[1].dims, vec![4_i64]);
        assert_eq!(inits[1].float_data.len(), 4);
    }

    #[test]
    fn test_onnx_linear_trans_b_attribute() {
        let w = Array2::<f64>::zeros((3, 5));
        let (nodes, _) = export_linear(&w, None, "x", "y", "fc");
        let trans_b = nodes[0].attributes.get("transB").expect("transB attribute");
        assert_eq!(trans_b, &OnnxAttribute::Int(1));
    }

    // -----------------------------------------------------------------------
    // Conv2d exporter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_onnx_conv2d_node() {
        // [out_channels, in_channels, kH, kW]
        let w = Array4::<f64>::zeros((16, 3, 3, 3));
        let (nodes, inits) = export_conv2d(&w, None, &[1, 1], &[1, 1], "x", "y", "conv1");
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].op_type, "Conv");
        assert_eq!(inits.len(), 1);
        assert_eq!(inits[0].dims, vec![16, 3, 3, 3]);
        assert_eq!(inits[0].float_data.len(), 16 * 3 * 3 * 3);
    }

    #[test]
    fn test_onnx_conv2d_with_bias() {
        let w = Array4::<f64>::zeros((8, 1, 5, 5));
        let b = Array1::<f64>::zeros(8);
        let (nodes, inits) = export_conv2d(&w, Some(&b), &[2, 2], &[0, 0], "x", "y", "conv0");
        assert_eq!(nodes.len(), 1);
        assert_eq!(inits.len(), 2);
        // Check stride attribute
        let strides = nodes[0].attributes.get("strides").expect("strides");
        assert_eq!(strides, &OnnxAttribute::Ints(vec![2, 2]));
    }

    // -----------------------------------------------------------------------
    // BatchNorm exporter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_onnx_batchnorm_export() {
        let scale = vec![1.0_f64; 32];
        let bias = vec![0.0_f64; 32];
        let mean = vec![0.0_f64; 32];
        let var = vec![1.0_f64; 32];
        let (nodes, inits) = export_batchnorm(&scale, &bias, &mean, &var, None, "x", "y", "bn1");
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].op_type, "BatchNormalization");
        // Must have exactly 4 initializers: scale, bias, mean, var
        assert_eq!(inits.len(), 4);
        for init in &inits {
            assert_eq!(init.dims, vec![32_i64]);
            assert_eq!(init.float_data.len(), 32);
        }
    }

    #[test]
    fn test_onnx_batchnorm_epsilon_attribute() {
        let v = vec![1.0_f64; 4];
        let (nodes, _) = export_batchnorm(&v, &v, &v, &v, Some(1e-3), "x", "y", "bn");
        let eps = nodes[0].attributes.get("epsilon").expect("epsilon attr");
        assert_eq!(eps, &OnnxAttribute::Float(1e-3_f32));
    }

    // -----------------------------------------------------------------------
    // OnnxModel metadata tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_onnx_opset_default() {
        let model = OnnxModel::default();
        assert_eq!(model.opset_version, 17);
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.producer_name, "scirs2-neural");
    }

    // -----------------------------------------------------------------------
    // Serialisation round-trip tests
    // -----------------------------------------------------------------------

    fn build_small_model() -> OnnxModel {
        let w = Array2::<f64>::zeros((4, 8));
        let b = Array1::<f64>::zeros(4);
        let (nodes, inits) = export_linear(&w, Some(&b), "input_0", "output_0", "fc0");
        let mut graph = OnnxGraph::new();
        graph.inputs.push(OnnxValueInfo::new(
            "input_0",
            OnnxDataType::Float32,
            vec![None, Some(8)],
        ));
        graph.outputs.push(OnnxValueInfo::new(
            "output_0",
            OnnxDataType::Float32,
            vec![None, Some(4)],
        ));
        graph.nodes.extend(nodes);
        graph.initializers.extend(inits);
        OnnxModel::new(graph)
    }

    #[test]
    fn test_onnx_model_roundtrip_bytes() {
        let original = build_small_model();
        let bytes = original.to_bytes().expect("to_bytes failed");
        let restored = OnnxModel::from_bytes(&bytes).expect("from_bytes failed");
        assert_eq!(restored.opset_version, original.opset_version);
        assert_eq!(restored.graph.nodes.len(), original.graph.nodes.len());
        assert_eq!(
            restored.graph.initializers.len(),
            original.graph.initializers.len()
        );
        assert_eq!(
            restored.graph.initializers[0].float_data.len(),
            original.graph.initializers[0].float_data.len()
        );
    }

    #[test]
    fn test_onnx_json_roundtrip() {
        let original = build_small_model();
        let json = original.to_json().expect("to_json failed");
        assert!(json.contains("Gemm"));
        let restored = OnnxModel::from_json(&json).expect("from_json failed");
        assert_eq!(restored.graph.nodes[0].op_type, "Gemm");
        assert_eq!(restored.graph.inputs[0].name, "input_0");
    }

    #[test]
    fn test_onnx_json_contains_producer_name() {
        let model = OnnxModel::default();
        let json = model.to_json().expect("to_json");
        assert!(json.contains("scirs2-neural"));
    }

    // -----------------------------------------------------------------------
    // Sequential exporter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_onnx_sequential_graph() {
        let w1 = Array2::<f64>::zeros((64, 784));
        let (n1, i1) = export_linear(&w1, None, "input_0", "fc0_out", "fc0");
        let act1 = export_activation("relu", "fc0_out", "act0_out");

        let w2 = Array2::<f64>::zeros((10, 64));
        let (n2, i2) = export_linear(&w2, None, "act0_out", "output_0", "fc1");

        let layers = vec![
            ("fc0".to_string(), n1, i1),
            ("act0".to_string(), vec![act1], vec![]),
            ("fc1".to_string(), n2, i2),
        ];

        let model = export_sequential(&layers, &[None, Some(784)]);
        // 3 nodes total: Gemm, Relu, Gemm
        assert_eq!(model.graph.nodes.len(), 3);
        assert_eq!(model.graph.nodes[0].op_type, "Gemm");
        assert_eq!(model.graph.nodes[1].op_type, "Relu");
        assert_eq!(model.graph.nodes[2].op_type, "Gemm");
        // 2 weight initializers (one per linear layer)
        assert_eq!(model.graph.initializers.len(), 2);
        assert_eq!(model.opset_version, 17);
    }

    #[test]
    fn test_onnx_sequential_single_layer() {
        let w = Array2::<f64>::zeros((2, 3));
        let (nodes, inits) = export_linear(&w, None, "input_0", "output_0", "fc");
        let layers = vec![("fc".to_string(), nodes, inits)];
        let model = export_sequential(&layers, &[None, Some(3)]);
        assert_eq!(model.graph.nodes.len(), 1);
        assert_eq!(model.graph.inputs[0].name, "input_0");
    }

    #[test]
    fn test_onnx_tensor_numel() {
        let t = OnnxTensor::from_f64_slice("t", vec![2, 3, 4], &[0.0_f64; 24]);
        assert_eq!(t.numel(), 24);
    }

    #[test]
    fn test_onnx_node_builder_with_attr() {
        let node = OnnxNode::new("Relu", "r", vec!["x".to_string()], vec!["y".to_string()])
            .with_attr("alpha", OnnxAttribute::Float(0.1));
        assert!(node.attributes.contains_key("alpha"));
    }
}
