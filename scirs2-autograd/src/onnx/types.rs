//! Core ONNX data structures for pure Rust ONNX interoperability.
//!
//! These types represent the ONNX intermediate representation without
//! requiring protobuf. They are serializable to/from JSON for interchange.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ONNX IR version constants
pub const ONNX_IR_VERSION: i64 = 9;
/// Default ONNX opset version
pub const ONNX_OPSET_VERSION: i64 = 20;
/// Producer name for SciRS2
pub const ONNX_PRODUCER_NAME: &str = "scirs2-autograd";
/// Producer version
pub const ONNX_PRODUCER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// ONNX tensor data types (matches ONNX spec TensorProto.DataType enum values)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum OnnxDataType {
    /// 32-bit floating point
    Float32 = 1,
    /// 8-bit unsigned integer
    Uint8 = 2,
    /// 8-bit signed integer
    Int8 = 3,
    /// 16-bit signed integer
    Int16 = 5,
    /// 32-bit signed integer
    Int32 = 6,
    /// 64-bit signed integer
    Int64 = 7,
    /// Boolean
    Bool = 9,
    /// 16-bit floating point
    Float16 = 10,
    /// 64-bit floating point
    Float64 = 11,
    /// 32-bit unsigned integer
    Uint32 = 12,
    /// 64-bit unsigned integer
    Uint64 = 13,
}

impl OnnxDataType {
    /// Create from ONNX spec integer code
    pub fn from_code(code: u32) -> Option<Self> {
        match code {
            1 => Some(OnnxDataType::Float32),
            2 => Some(OnnxDataType::Uint8),
            3 => Some(OnnxDataType::Int8),
            5 => Some(OnnxDataType::Int16),
            6 => Some(OnnxDataType::Int32),
            7 => Some(OnnxDataType::Int64),
            9 => Some(OnnxDataType::Bool),
            10 => Some(OnnxDataType::Float16),
            11 => Some(OnnxDataType::Float64),
            12 => Some(OnnxDataType::Uint32),
            13 => Some(OnnxDataType::Uint64),
            _ => None,
        }
    }

    /// Get the ONNX spec integer code
    pub fn code(&self) -> u32 {
        *self as u32
    }

    /// Get the byte size of a single element of this type
    pub fn element_size(&self) -> usize {
        match self {
            OnnxDataType::Bool | OnnxDataType::Uint8 | OnnxDataType::Int8 => 1,
            OnnxDataType::Float16 | OnnxDataType::Int16 => 2,
            OnnxDataType::Float32 | OnnxDataType::Int32 | OnnxDataType::Uint32 => 4,
            OnnxDataType::Float64 | OnnxDataType::Int64 | OnnxDataType::Uint64 => 8,
        }
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            OnnxDataType::Float32 => "float32",
            OnnxDataType::Uint8 => "uint8",
            OnnxDataType::Int8 => "int8",
            OnnxDataType::Int16 => "int16",
            OnnxDataType::Int32 => "int32",
            OnnxDataType::Int64 => "int64",
            OnnxDataType::Bool => "bool",
            OnnxDataType::Float16 => "float16",
            OnnxDataType::Float64 => "float64",
            OnnxDataType::Uint32 => "uint32",
            OnnxDataType::Uint64 => "uint64",
        }
    }
}

impl std::fmt::Display for OnnxDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// ONNX attribute value (used on nodes to configure operations)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum OnnxAttribute {
    /// 64-bit integer
    Int(i64),
    /// 64-bit float
    Float(f64),
    /// UTF-8 string
    String(String),
    /// Tensor value
    Tensor(OnnxTensor),
    /// List of integers
    Ints(Vec<i64>),
    /// List of floats
    Floats(Vec<f64>),
    /// List of strings
    Strings(Vec<String>),
}

impl OnnxAttribute {
    /// Try to get this attribute as an integer
    pub fn as_int(&self) -> Option<i64> {
        match self {
            OnnxAttribute::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get this attribute as a float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            OnnxAttribute::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get this attribute as a string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            OnnxAttribute::String(v) => Some(v),
            _ => None,
        }
    }

    /// Try to get this attribute as a list of integers
    pub fn as_ints(&self) -> Option<&[i64]> {
        match self {
            OnnxAttribute::Ints(v) => Some(v),
            _ => None,
        }
    }

    /// Try to get this attribute as a list of floats
    pub fn as_floats(&self) -> Option<&[f64]> {
        match self {
            OnnxAttribute::Floats(v) => Some(v),
            _ => None,
        }
    }

    /// Try to get this attribute as a list of strings
    pub fn as_strings(&self) -> Option<&[String]> {
        match self {
            OnnxAttribute::Strings(v) => Some(v),
            _ => None,
        }
    }
}

/// ONNX tensor value (used for inputs, outputs, initializers)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxTensor {
    /// Tensor name (unique within graph)
    pub name: String,
    /// Data type
    pub data_type: OnnxDataType,
    /// Dimensions (shape). -1 indicates dynamic dimension.
    pub dims: Vec<i64>,
    /// Float32 data (populated when data_type is Float32)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub float_data: Vec<f32>,
    /// Float64 data (populated when data_type is Float64)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub double_data: Vec<f64>,
    /// Int32 data (populated when data_type is Int32)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub int32_data: Vec<i32>,
    /// Int64 data (populated when data_type is Int64)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub int64_data: Vec<i64>,
    /// Raw byte data (alternative storage, populated for Bool/Uint8/Int8 or when compact encoding is used)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub raw_data: Vec<u8>,
    /// Optional documentation string
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_string: Option<String>,
}

impl OnnxTensor {
    /// Create a new tensor specification (no data, just shape info)
    pub fn spec(name: &str, dims: &[i64], data_type: OnnxDataType) -> Self {
        OnnxTensor {
            name: name.to_string(),
            data_type,
            dims: dims.to_vec(),
            float_data: Vec::new(),
            double_data: Vec::new(),
            int32_data: Vec::new(),
            int64_data: Vec::new(),
            raw_data: Vec::new(),
            doc_string: None,
        }
    }

    /// Create a tensor with f32 data
    pub fn from_f32(name: &str, dims: &[i64], data: Vec<f32>) -> Self {
        OnnxTensor {
            name: name.to_string(),
            data_type: OnnxDataType::Float32,
            dims: dims.to_vec(),
            float_data: data,
            double_data: Vec::new(),
            int32_data: Vec::new(),
            int64_data: Vec::new(),
            raw_data: Vec::new(),
            doc_string: None,
        }
    }

    /// Create a tensor with f64 data
    pub fn from_f64(name: &str, dims: &[i64], data: Vec<f64>) -> Self {
        OnnxTensor {
            name: name.to_string(),
            data_type: OnnxDataType::Float64,
            dims: dims.to_vec(),
            float_data: Vec::new(),
            double_data: data,
            int32_data: Vec::new(),
            int64_data: Vec::new(),
            raw_data: Vec::new(),
            doc_string: None,
        }
    }

    /// Create a tensor with i32 data
    pub fn from_i32(name: &str, dims: &[i64], data: Vec<i32>) -> Self {
        OnnxTensor {
            name: name.to_string(),
            data_type: OnnxDataType::Int32,
            dims: dims.to_vec(),
            float_data: Vec::new(),
            double_data: Vec::new(),
            int32_data: data,
            int64_data: Vec::new(),
            raw_data: Vec::new(),
            doc_string: None,
        }
    }

    /// Create a tensor with i64 data
    pub fn from_i64(name: &str, dims: &[i64], data: Vec<i64>) -> Self {
        OnnxTensor {
            name: name.to_string(),
            data_type: OnnxDataType::Int64,
            dims: dims.to_vec(),
            float_data: Vec::new(),
            double_data: Vec::new(),
            int32_data: Vec::new(),
            int64_data: data,
            raw_data: Vec::new(),
            doc_string: None,
        }
    }

    /// Return the total number of elements based on dims (ignoring dynamic dimensions)
    pub fn num_elements(&self) -> Option<usize> {
        let mut count: usize = 1;
        for &d in &self.dims {
            if d < 0 {
                return None; // dynamic dimension
            }
            count = count.checked_mul(d as usize)?;
        }
        Some(count)
    }

    /// Check if this tensor has actual data (not just a shape specification)
    pub fn has_data(&self) -> bool {
        !self.float_data.is_empty()
            || !self.double_data.is_empty()
            || !self.int32_data.is_empty()
            || !self.int64_data.is_empty()
            || !self.raw_data.is_empty()
    }
}

/// ONNX graph node (represents a single operation)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxNode {
    /// ONNX op type (e.g., "MatMul", "Relu", "Conv", "Add")
    pub op_type: String,
    /// Node name (unique within graph)
    pub name: String,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor names
    pub outputs: Vec<String>,
    /// Node attributes
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub attributes: HashMap<String, OnnxAttribute>,
    /// Optional documentation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_string: Option<String>,
    /// ONNX domain (empty string = default ONNX domain)
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub domain: String,
}

impl OnnxNode {
    /// Create a new node
    pub fn new(op_type: &str, name: &str, inputs: Vec<String>, outputs: Vec<String>) -> Self {
        OnnxNode {
            op_type: op_type.to_string(),
            name: name.to_string(),
            inputs,
            outputs,
            attributes: HashMap::new(),
            doc_string: None,
            domain: String::new(),
        }
    }

    /// Add an attribute to this node
    pub fn with_attribute(mut self, key: &str, value: OnnxAttribute) -> Self {
        self.attributes.insert(key.to_string(), value);
        self
    }

    /// Get an attribute by key
    pub fn get_attribute(&self, key: &str) -> Option<&OnnxAttribute> {
        self.attributes.get(key)
    }
}

/// ONNX opset import information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxOpsetImport {
    /// Domain (empty string = default ONNX domain)
    #[serde(default)]
    pub domain: String,
    /// Opset version
    pub version: i64,
}

impl Default for OnnxOpsetImport {
    fn default() -> Self {
        OnnxOpsetImport {
            domain: String::new(),
            version: ONNX_OPSET_VERSION,
        }
    }
}

/// ONNX computation graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxGraph {
    /// Graph name
    pub name: String,
    /// Nodes (operations) in topological order
    pub nodes: Vec<OnnxNode>,
    /// Input tensor specifications
    pub inputs: Vec<OnnxTensor>,
    /// Output tensor specifications
    pub outputs: Vec<OnnxTensor>,
    /// Initializer tensors (model weights/constants)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub initializers: Vec<OnnxTensor>,
    /// Optional documentation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_string: Option<String>,
}

impl OnnxGraph {
    /// Create a new empty graph
    pub fn new(name: &str) -> Self {
        OnnxGraph {
            name: name.to_string(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            initializers: Vec::new(),
            doc_string: None,
        }
    }

    /// Get a node by name
    pub fn get_node(&self, name: &str) -> Option<&OnnxNode> {
        self.nodes.iter().find(|n| n.name == name)
    }

    /// Get an initializer by name
    pub fn get_initializer(&self, name: &str) -> Option<&OnnxTensor> {
        self.initializers.iter().find(|t| t.name == name)
    }

    /// Get total number of parameters (elements in all initializers)
    pub fn total_parameters(&self) -> usize {
        self.initializers
            .iter()
            .filter_map(|t| t.num_elements())
            .sum()
    }
}

/// ONNX model (top-level container wrapping a graph)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxModel {
    /// IR version
    pub ir_version: i64,
    /// Opset imports
    pub opset_imports: Vec<OnnxOpsetImport>,
    /// Producer name
    pub producer_name: String,
    /// Producer version
    pub producer_version: String,
    /// Model domain
    #[serde(default)]
    pub domain: String,
    /// Model version
    #[serde(default)]
    pub model_version: i64,
    /// Model documentation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_string: Option<String>,
    /// The computation graph
    pub graph: OnnxGraph,
    /// Metadata properties
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

impl OnnxModel {
    /// Create a new model wrapping a graph
    pub fn new(graph: OnnxGraph) -> Self {
        OnnxModel {
            ir_version: ONNX_IR_VERSION,
            opset_imports: vec![OnnxOpsetImport::default()],
            producer_name: ONNX_PRODUCER_NAME.to_string(),
            producer_version: ONNX_PRODUCER_VERSION.to_string(),
            domain: String::new(),
            model_version: 1,
            doc_string: None,
            graph,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata key-value pair
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Set documentation string
    pub fn with_doc_string(mut self, doc: &str) -> Self {
        self.doc_string = Some(doc.to_string());
        self
    }
}

/// Convolution attributes for ONNX Conv node
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvAttributes {
    /// Kernel shape (height, width)
    pub kernel_shape: Vec<i64>,
    /// Strides
    #[serde(default = "ConvAttributes::default_strides")]
    pub strides: Vec<i64>,
    /// Pads (begin_h, begin_w, end_h, end_w)
    #[serde(default)]
    pub pads: Vec<i64>,
    /// Dilations
    #[serde(default = "ConvAttributes::default_dilations")]
    pub dilations: Vec<i64>,
    /// Number of groups
    #[serde(default = "ConvAttributes::default_group")]
    pub group: i64,
    /// Auto-pad mode: "NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"
    #[serde(default = "ConvAttributes::default_auto_pad")]
    pub auto_pad: String,
}

impl ConvAttributes {
    fn default_strides() -> Vec<i64> {
        vec![1, 1]
    }
    fn default_dilations() -> Vec<i64> {
        vec![1, 1]
    }
    fn default_group() -> i64 {
        1
    }
    fn default_auto_pad() -> String {
        "NOTSET".to_string()
    }

    /// Create new ConvAttributes with given kernel shape
    pub fn new(kernel_shape: Vec<i64>) -> Self {
        ConvAttributes {
            kernel_shape,
            strides: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            dilations: vec![1, 1],
            group: 1,
            auto_pad: "NOTSET".to_string(),
        }
    }

    /// Set strides
    pub fn with_strides(mut self, strides: Vec<i64>) -> Self {
        self.strides = strides;
        self
    }

    /// Set pads
    pub fn with_pads(mut self, pads: Vec<i64>) -> Self {
        self.pads = pads;
        self
    }

    /// Set dilations
    pub fn with_dilations(mut self, dilations: Vec<i64>) -> Self {
        self.dilations = dilations;
        self
    }

    /// Set group
    pub fn with_group(mut self, group: i64) -> Self {
        self.group = group;
        self
    }

    /// Convert to ONNX attribute map
    pub fn to_attributes(&self) -> HashMap<String, OnnxAttribute> {
        let mut attrs = HashMap::new();
        attrs.insert(
            "kernel_shape".to_string(),
            OnnxAttribute::Ints(self.kernel_shape.clone()),
        );
        attrs.insert(
            "strides".to_string(),
            OnnxAttribute::Ints(self.strides.clone()),
        );
        attrs.insert("pads".to_string(), OnnxAttribute::Ints(self.pads.clone()));
        attrs.insert(
            "dilations".to_string(),
            OnnxAttribute::Ints(self.dilations.clone()),
        );
        attrs.insert("group".to_string(), OnnxAttribute::Int(self.group));
        if self.auto_pad != "NOTSET" {
            attrs.insert(
                "auto_pad".to_string(),
                OnnxAttribute::String(self.auto_pad.clone()),
            );
        }
        attrs
    }

    /// Parse from ONNX attribute map
    pub fn from_attributes(attrs: &HashMap<String, OnnxAttribute>) -> Option<Self> {
        let kernel_shape = attrs.get("kernel_shape")?.as_ints()?.to_vec();
        let strides = attrs
            .get("strides")
            .and_then(|a| a.as_ints())
            .map(|s| s.to_vec())
            .unwrap_or_else(Self::default_strides);
        let pads = attrs
            .get("pads")
            .and_then(|a| a.as_ints())
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let dilations = attrs
            .get("dilations")
            .and_then(|a| a.as_ints())
            .map(|s| s.to_vec())
            .unwrap_or_else(Self::default_dilations);
        let group = attrs.get("group").and_then(|a| a.as_int()).unwrap_or(1);
        let auto_pad = attrs
            .get("auto_pad")
            .and_then(|a| a.as_string())
            .map(|s| s.to_string())
            .unwrap_or_else(Self::default_auto_pad);

        Some(ConvAttributes {
            kernel_shape,
            strides,
            pads,
            dilations,
            group,
            auto_pad,
        })
    }
}

/// Pooling attributes for ONNX MaxPool/AveragePool nodes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoolAttributes {
    /// Kernel shape
    pub kernel_shape: Vec<i64>,
    /// Strides
    #[serde(default = "PoolAttributes::default_strides")]
    pub strides: Vec<i64>,
    /// Pads
    #[serde(default)]
    pub pads: Vec<i64>,
    /// Ceil mode (0=floor, 1=ceil)
    #[serde(default)]
    pub ceil_mode: i64,
}

impl PoolAttributes {
    fn default_strides() -> Vec<i64> {
        vec![1, 1]
    }

    /// Create new PoolAttributes
    pub fn new(kernel_shape: Vec<i64>) -> Self {
        PoolAttributes {
            kernel_shape,
            strides: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            ceil_mode: 0,
        }
    }

    /// Set strides
    pub fn with_strides(mut self, strides: Vec<i64>) -> Self {
        self.strides = strides;
        self
    }

    /// Set pads
    pub fn with_pads(mut self, pads: Vec<i64>) -> Self {
        self.pads = pads;
        self
    }

    /// Convert to ONNX attribute map
    pub fn to_attributes(&self) -> HashMap<String, OnnxAttribute> {
        let mut attrs = HashMap::new();
        attrs.insert(
            "kernel_shape".to_string(),
            OnnxAttribute::Ints(self.kernel_shape.clone()),
        );
        attrs.insert(
            "strides".to_string(),
            OnnxAttribute::Ints(self.strides.clone()),
        );
        attrs.insert("pads".to_string(), OnnxAttribute::Ints(self.pads.clone()));
        if self.ceil_mode != 0 {
            attrs.insert("ceil_mode".to_string(), OnnxAttribute::Int(self.ceil_mode));
        }
        attrs
    }
}

/// Gemm (General Matrix Multiply) attributes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GemmAttributes {
    /// Scalar alpha (default 1.0)
    #[serde(default = "GemmAttributes::default_alpha")]
    pub alpha: f64,
    /// Scalar beta (default 1.0)
    #[serde(default = "GemmAttributes::default_beta")]
    pub beta: f64,
    /// Transpose A
    #[serde(default)]
    pub trans_a: i64,
    /// Transpose B
    #[serde(default)]
    pub trans_b: i64,
}

impl GemmAttributes {
    fn default_alpha() -> f64 {
        1.0
    }
    fn default_beta() -> f64 {
        1.0
    }

    /// Create default Gemm attributes
    pub fn new() -> Self {
        GemmAttributes {
            alpha: 1.0,
            beta: 1.0,
            trans_a: 0,
            trans_b: 0,
        }
    }

    /// Convert to ONNX attribute map
    pub fn to_attributes(&self) -> HashMap<String, OnnxAttribute> {
        let mut attrs = HashMap::new();
        if (self.alpha - 1.0).abs() > f64::EPSILON {
            attrs.insert("alpha".to_string(), OnnxAttribute::Float(self.alpha));
        }
        if (self.beta - 1.0).abs() > f64::EPSILON {
            attrs.insert("beta".to_string(), OnnxAttribute::Float(self.beta));
        }
        if self.trans_a != 0 {
            attrs.insert("transA".to_string(), OnnxAttribute::Int(self.trans_a));
        }
        if self.trans_b != 0 {
            attrs.insert("transB".to_string(), OnnxAttribute::Int(self.trans_b));
        }
        attrs
    }
}

impl Default for GemmAttributes {
    fn default() -> Self {
        Self::new()
    }
}
