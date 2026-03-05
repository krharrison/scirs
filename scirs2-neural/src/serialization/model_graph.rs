//! Computation graph export and import for neural network models.
//!
//! This module provides an intermediate representation (IR) for neural network
//! architectures, enabling:
//! - JSON export/import of the full model graph (nodes + edges + tensor shapes)
//! - A simplified ONNX-like format that captures operator types and tensor metadata
//!   using serde_json — no C protobuf runtime required.
//!
//! ## Design
//!
//! A [`ModelGraph`] is a directed acyclic graph (DAG) where:
//! - [`GraphNode`]s represent operators (Conv2D, Dense, BatchNorm, etc.) and I/O.
//! - [`GraphEdge`]s connect a producer node's output slot to a consumer node's input slot.
//! - [`TensorShape`] describes each tensor's rank, dimensions, and dtype.
//!
//! The graph can be serialized to/from JSON with [`ModelGraph::export_to_json`] and
//! [`ModelGraph::import_from_json`], and converted to a simplified ONNX-like JSON
//! envelope with [`ModelGraph::export_onnx_like`].

use crate::error::{NeuralError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

// ============================================================================
// Tensor shape / dtype
// ============================================================================

/// Element type of a tensor in the graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TensorDtype {
    /// 32-bit float
    F32,
    /// 64-bit float
    F64,
    /// 16-bit float (half precision)
    F16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// Boolean
    Bool,
}

impl TensorDtype {
    /// Return the ONNX-compatible string for this dtype.
    pub fn onnx_str(&self) -> &str {
        match self {
            TensorDtype::F32 => "float",
            TensorDtype::F64 => "double",
            TensorDtype::F16 => "float16",
            TensorDtype::I32 => "int32",
            TensorDtype::I64 => "int64",
            TensorDtype::Bool => "bool",
        }
    }

    /// Parse from an ONNX-compatible string.
    pub fn from_onnx_str(s: &str) -> Result<Self> {
        match s {
            "float" | "F32" | "f32" => Ok(TensorDtype::F32),
            "double" | "F64" | "f64" => Ok(TensorDtype::F64),
            "float16" | "F16" | "f16" => Ok(TensorDtype::F16),
            "int32" | "I32" | "i32" => Ok(TensorDtype::I32),
            "int64" | "I64" | "i64" => Ok(TensorDtype::I64),
            "bool" | "Bool" => Ok(TensorDtype::Bool),
            other => Err(NeuralError::DeserializationError(format!(
                "Unknown tensor dtype: {other}"
            ))),
        }
    }

    /// Byte size of a single element.
    pub fn element_size(&self) -> usize {
        match self {
            TensorDtype::F32 => 4,
            TensorDtype::F64 => 8,
            TensorDtype::F16 => 2,
            TensorDtype::I32 => 4,
            TensorDtype::I64 => 8,
            TensorDtype::Bool => 1,
        }
    }
}

impl Default for TensorDtype {
    fn default() -> Self {
        TensorDtype::F32
    }
}

impl std::fmt::Display for TensorDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.onnx_str())
    }
}

/// Shape of a tensor in the model graph.
///
/// Dimensions may be `None` (symbolic / dynamic), e.g. the batch dimension.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorShape {
    /// Ordered list of dimensions. `None` means dynamic / unknown at export time.
    pub dims: Vec<Option<i64>>,
    /// Element type.
    pub dtype: TensorDtype,
}

impl TensorShape {
    /// Create a fully-static shape.
    pub fn new(dims: Vec<i64>, dtype: TensorDtype) -> Self {
        Self {
            dims: dims.into_iter().map(Some).collect(),
            dtype,
        }
    }

    /// Create a shape with the first dimension dynamic (batch dimension).
    pub fn with_batch_dim(spatial_dims: Vec<i64>, dtype: TensorDtype) -> Self {
        let mut dims = vec![None];
        dims.extend(spatial_dims.into_iter().map(Some));
        Self { dims, dtype }
    }

    /// Total number of elements; returns `None` if any dimension is dynamic.
    pub fn num_elements(&self) -> Option<i64> {
        let mut product = 1i64;
        for d in &self.dims {
            match d {
                Some(v) => product = product.checked_mul(*v)?,
                None => return None,
            }
        }
        Some(product)
    }

    /// Return the rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

// ============================================================================
// Node attributes (operator hyper-parameters)
// ============================================================================

/// Padding specification for 2-D operators.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PaddingSpec {
    /// Top padding in pixels
    pub top: usize,
    /// Bottom padding in pixels
    pub bottom: usize,
    /// Left padding in pixels
    pub left: usize,
    /// Right padding in pixels
    pub right: usize,
}

impl PaddingSpec {
    /// Create a symmetric padding spec (same value on all sides).
    pub fn same(value: usize) -> Self {
        Self {
            top: value,
            bottom: value,
            left: value,
            right: value,
        }
    }

    /// Create a zero-padding spec.
    pub fn zero() -> Self {
        Self::same(0)
    }
}

/// Attributes for a Dense (fully-connected) layer node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseAttrs {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Whether the layer includes a bias vector.
    pub use_bias: bool,
}

/// Attributes for a 2-D convolution node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conv2DAttrs {
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Vertical stride.
    pub stride_h: usize,
    /// Horizontal stride.
    pub stride_w: usize,
    /// Padding specification.
    pub padding: PaddingSpec,
    /// Dilation factor.
    pub dilation: usize,
    /// Number of groups (for depth-wise / grouped convolutions).
    pub groups: usize,
    /// Whether the convolution includes a bias.
    pub use_bias: bool,
}

/// Attributes for a Batch Normalization node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchNormAttrs {
    /// Number of features / channels being normalized.
    pub num_features: usize,
    /// Numerical stability epsilon.
    pub eps: f64,
    /// Exponential moving average momentum.
    pub momentum: f64,
    /// Whether learnable affine parameters (gamma, beta) are used.
    pub affine: bool,
}

/// Attributes for a Layer Normalization node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerNormAttrs {
    /// Normalized shape (last N dimensions).
    pub normalized_shape: Vec<usize>,
    /// Numerical stability epsilon.
    pub eps: f64,
    /// Whether learnable elementwise affine parameters are used.
    pub elementwise_affine: bool,
}

/// Attributes for a 2-D max / average pooling node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pool2DAttrs {
    /// Pooling kernel height.
    pub kernel_h: usize,
    /// Pooling kernel width.
    pub kernel_w: usize,
    /// Vertical stride. Defaults to kernel_h if `None`.
    pub stride_h: Option<usize>,
    /// Horizontal stride. Defaults to kernel_w if `None`.
    pub stride_w: Option<usize>,
    /// Padding.
    pub padding: PaddingSpec,
}

/// Attributes for a Dropout node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DropoutAttrs {
    /// Drop probability in `[0, 1)`.
    pub p: f64,
}

/// Attributes for a Reshape node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReshapeAttrs {
    /// Target shape; `-1` denotes an inferred dimension.
    pub shape: Vec<i64>,
}

/// Attributes for a Multi-Head Attention node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionAttrs {
    /// Model embedding dimension.
    pub embed_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dropout applied to attention weights.
    pub attn_dropout: f64,
    /// Whether to include a projection bias.
    pub use_bias: bool,
}

/// Attributes for an Embedding node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingAttrs {
    /// Vocabulary size.
    pub num_embeddings: usize,
    /// Embedding dimensionality.
    pub embedding_dim: usize,
    /// Optional index treated as padding (output is zeroed for this index).
    pub padding_idx: Option<i64>,
}

/// Attributes for an activation node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivationAttrs {
    /// Activation function name (e.g., "relu", "gelu", "tanh").
    pub function: String,
    /// Optional scalar parameter (e.g., negative slope for LeakyReLU).
    pub alpha: Option<f64>,
}

/// Attributes for a residual (skip) connection add node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AddAttrs {
    /// Whether to multiply inputs by a learnable scalar before adding.
    pub learnable_scale: bool,
}

// ============================================================================
// GraphNode enum
// ============================================================================

/// A single operator or I/O node in the model computation graph.
///
/// Each variant carries the hyper-parameters (attributes) needed to reconstruct
/// that operator. Weight tensors are stored separately in a [`WeightStore`](super::weight_format::WeightStore)
/// and referenced by node name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "node_type", rename_all = "snake_case")]
pub enum GraphNode {
    /// Model input placeholder.
    Input {
        /// Name of this input tensor.
        name: String,
        /// Expected tensor shape (batch dimension may be `None`).
        shape: TensorShape,
    },
    /// Model output placeholder.
    Output {
        /// Name of this output tensor.
        name: String,
        /// Expected tensor shape.
        shape: TensorShape,
    },
    /// Fully-connected (dense) layer.
    Dense {
        /// Unique node name within the graph.
        name: String,
        /// Layer hyper-parameters.
        attrs: DenseAttrs,
        /// Output tensor shape.
        output_shape: TensorShape,
    },
    /// 2-D convolution.
    Conv2D {
        /// Unique node name within the graph.
        name: String,
        /// Convolution hyper-parameters.
        attrs: Conv2DAttrs,
        /// Output tensor shape.
        output_shape: TensorShape,
    },
    /// Depth-wise separable convolution (Conv2D with groups=in_channels).
    DepthwiseConv2D {
        /// Unique node name within the graph.
        name: String,
        /// Convolution hyper-parameters.
        attrs: Conv2DAttrs,
        /// Output tensor shape.
        output_shape: TensorShape,
    },
    /// Batch normalization.
    BatchNorm {
        /// Unique node name within the graph.
        name: String,
        /// BatchNorm hyper-parameters.
        attrs: BatchNormAttrs,
        /// Output tensor shape (same as input).
        output_shape: TensorShape,
    },
    /// Layer normalization.
    LayerNorm {
        /// Unique node name within the graph.
        name: String,
        /// LayerNorm hyper-parameters.
        attrs: LayerNormAttrs,
        /// Output tensor shape (same as input).
        output_shape: TensorShape,
    },
    /// Element-wise activation function.
    Activation {
        /// Unique node name within the graph.
        name: String,
        /// Activation attributes.
        attrs: ActivationAttrs,
        /// Output tensor shape (same as input).
        output_shape: TensorShape,
    },
    /// 2-D max pooling.
    MaxPool2D {
        /// Unique node name within the graph.
        name: String,
        /// Pooling hyper-parameters.
        attrs: Pool2DAttrs,
        /// Output tensor shape.
        output_shape: TensorShape,
    },
    /// 2-D average pooling.
    AvgPool2D {
        /// Unique node name within the graph.
        name: String,
        /// Pooling hyper-parameters.
        attrs: Pool2DAttrs,
        /// Output tensor shape.
        output_shape: TensorShape,
    },
    /// Global average pooling (spatial → scalar per channel).
    GlobalAvgPool {
        /// Unique node name within the graph.
        name: String,
        /// Output tensor shape (batch, channels).
        output_shape: TensorShape,
    },
    /// Dropout regularization.
    Dropout {
        /// Unique node name within the graph.
        name: String,
        /// Dropout attributes.
        attrs: DropoutAttrs,
        /// Output tensor shape (same as input).
        output_shape: TensorShape,
    },
    /// Reshape / view.
    Reshape {
        /// Unique node name within the graph.
        name: String,
        /// Reshape attributes.
        attrs: ReshapeAttrs,
        /// Output tensor shape.
        output_shape: TensorShape,
    },
    /// Flatten (reshape to 2-D: batch × features).
    Flatten {
        /// Unique node name within the graph.
        name: String,
        /// Output tensor shape.
        output_shape: TensorShape,
    },
    /// Element-wise add (residual / skip connection).
    Add {
        /// Unique node name within the graph.
        name: String,
        /// Add attributes.
        attrs: AddAttrs,
        /// Output tensor shape.
        output_shape: TensorShape,
    },
    /// Concatenation along a specified axis.
    Concat {
        /// Unique node name within the graph.
        name: String,
        /// Axis along which to concatenate.
        axis: i64,
        /// Output tensor shape.
        output_shape: TensorShape,
    },
    /// Multi-head self-/cross-attention.
    Attention {
        /// Unique node name within the graph.
        name: String,
        /// Attention attributes.
        attrs: AttentionAttrs,
        /// Output tensor shape.
        output_shape: TensorShape,
    },
    /// Token / positional embedding.
    Embedding {
        /// Unique node name within the graph.
        name: String,
        /// Embedding attributes.
        attrs: EmbeddingAttrs,
        /// Output tensor shape (batch, seq_len, embed_dim).
        output_shape: TensorShape,
    },
    /// Softmax normalization.
    Softmax {
        /// Unique node name within the graph.
        name: String,
        /// Axis to apply softmax over.
        axis: i64,
        /// Output tensor shape (same as input).
        output_shape: TensorShape,
    },
    /// Generic / custom operator not covered by the above variants.
    Custom {
        /// Unique node name within the graph.
        name: String,
        /// Operator type string (free-form identifier).
        op_type: String,
        /// Key-value attribute map.
        attributes: HashMap<String, serde_json::Value>,
        /// Output tensor shape.
        output_shape: TensorShape,
    },
}

impl GraphNode {
    /// Return the unique name of this node.
    pub fn name(&self) -> &str {
        match self {
            GraphNode::Input { name, .. }
            | GraphNode::Output { name, .. }
            | GraphNode::Dense { name, .. }
            | GraphNode::Conv2D { name, .. }
            | GraphNode::DepthwiseConv2D { name, .. }
            | GraphNode::BatchNorm { name, .. }
            | GraphNode::LayerNorm { name, .. }
            | GraphNode::Activation { name, .. }
            | GraphNode::MaxPool2D { name, .. }
            | GraphNode::AvgPool2D { name, .. }
            | GraphNode::GlobalAvgPool { name, .. }
            | GraphNode::Dropout { name, .. }
            | GraphNode::Reshape { name, .. }
            | GraphNode::Flatten { name, .. }
            | GraphNode::Add { name, .. }
            | GraphNode::Concat { name, .. }
            | GraphNode::Attention { name, .. }
            | GraphNode::Embedding { name, .. }
            | GraphNode::Softmax { name, .. }
            | GraphNode::Custom { name, .. } => name,
        }
    }

    /// Return the output tensor shape of this node.
    pub fn output_shape(&self) -> &TensorShape {
        match self {
            GraphNode::Input { shape, .. } | GraphNode::Output { shape, .. } => shape,
            GraphNode::Dense { output_shape, .. }
            | GraphNode::Conv2D { output_shape, .. }
            | GraphNode::DepthwiseConv2D { output_shape, .. }
            | GraphNode::BatchNorm { output_shape, .. }
            | GraphNode::LayerNorm { output_shape, .. }
            | GraphNode::Activation { output_shape, .. }
            | GraphNode::MaxPool2D { output_shape, .. }
            | GraphNode::AvgPool2D { output_shape, .. }
            | GraphNode::GlobalAvgPool { output_shape, .. }
            | GraphNode::Dropout { output_shape, .. }
            | GraphNode::Reshape { output_shape, .. }
            | GraphNode::Flatten { output_shape, .. }
            | GraphNode::Add { output_shape, .. }
            | GraphNode::Concat { output_shape, .. }
            | GraphNode::Attention { output_shape, .. }
            | GraphNode::Embedding { output_shape, .. }
            | GraphNode::Softmax { output_shape, .. }
            | GraphNode::Custom { output_shape, .. } => output_shape,
        }
    }

    /// Return the ONNX op_type string for this node.
    pub fn onnx_op_type(&self) -> &str {
        match self {
            GraphNode::Input { .. } => "Input",
            GraphNode::Output { .. } => "Output",
            GraphNode::Dense { .. } => "Gemm",
            GraphNode::Conv2D { .. } => "Conv",
            GraphNode::DepthwiseConv2D { .. } => "Conv",
            GraphNode::BatchNorm { .. } => "BatchNormalization",
            GraphNode::LayerNorm { .. } => "LayerNormalization",
            GraphNode::Activation { attrs, .. } => match attrs.function.as_str() {
                "relu" => "Relu",
                "sigmoid" => "Sigmoid",
                "tanh" => "Tanh",
                "gelu" => "Gelu",
                "leaky_relu" => "LeakyRelu",
                "elu" => "Elu",
                "swish" | "silu" => "Swish",
                _ => "Activation",
            },
            GraphNode::MaxPool2D { .. } => "MaxPool",
            GraphNode::AvgPool2D { .. } => "AveragePool",
            GraphNode::GlobalAvgPool { .. } => "GlobalAveragePool",
            GraphNode::Dropout { .. } => "Dropout",
            GraphNode::Reshape { .. } => "Reshape",
            GraphNode::Flatten { .. } => "Flatten",
            GraphNode::Add { .. } => "Add",
            GraphNode::Concat { .. } => "Concat",
            GraphNode::Attention { .. } => "MultiHeadAttention",
            GraphNode::Embedding { .. } => "Gather",
            GraphNode::Softmax { .. } => "Softmax",
            GraphNode::Custom { op_type, .. } => op_type,
        }
    }

    /// Return the number of learnable parameter tensors expected for this node.
    ///
    /// Useful for weight-count estimations. Returns `0` for stateless ops.
    pub fn num_weight_tensors(&self) -> usize {
        match self {
            GraphNode::Dense { attrs, .. } => {
                if attrs.use_bias {
                    2
                } else {
                    1
                }
            }
            GraphNode::Conv2D { attrs, .. } | GraphNode::DepthwiseConv2D { attrs, .. } => {
                if attrs.use_bias {
                    2
                } else {
                    1
                }
            }
            GraphNode::BatchNorm { attrs, .. } => {
                if attrs.affine {
                    4
                } else {
                    2
                }
            }
            GraphNode::LayerNorm { attrs, .. } => {
                if attrs.elementwise_affine {
                    2
                } else {
                    0
                }
            }
            GraphNode::Embedding { .. } | GraphNode::Attention { .. } => 1,
            _ => 0,
        }
    }
}

// ============================================================================
// GraphEdge
// ============================================================================

/// A directed data-flow edge in the model computation graph.
///
/// An edge connects one output slot of a source node to one input slot of a
/// destination node. Slot indices are zero-based.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Name of the producer node.
    pub from_node: String,
    /// Output slot index of the producer (0 for single-output nodes).
    pub from_slot: usize,
    /// Name of the consumer node.
    pub to_node: String,
    /// Input slot index of the consumer (0 for single-input nodes).
    pub to_slot: usize,
}

impl GraphEdge {
    /// Create a simple single-input single-output edge.
    pub fn simple(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from_node: from.into(),
            from_slot: 0,
            to_node: to.into(),
            to_slot: 0,
        }
    }

    /// Create an edge with explicit slot indices.
    pub fn with_slots(
        from: impl Into<String>,
        from_slot: usize,
        to: impl Into<String>,
        to_slot: usize,
    ) -> Self {
        Self {
            from_node: from.into(),
            from_slot,
            to_node: to.into(),
            to_slot,
        }
    }
}

// ============================================================================
// ModelGraph
// ============================================================================

/// Graph-level metadata stored alongside nodes and edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    /// Model architecture name (e.g., "ResNet", "BERT").
    pub architecture: String,
    /// SciRS2 framework version that produced this graph.
    pub framework_version: String,
    /// Graph format version.
    pub graph_format_version: String,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// Arbitrary extra key-value annotations.
    pub extra: HashMap<String, String>,
}

impl GraphMetadata {
    /// Create graph metadata with required fields.
    pub fn new(architecture: impl Into<String>) -> Self {
        Self {
            architecture: architecture.into(),
            framework_version: env!("CARGO_PKG_VERSION").to_string(),
            graph_format_version: "1.0".to_string(),
            description: None,
            extra: HashMap::new(),
        }
    }

    /// Add a human-readable description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add an extra annotation key-value pair.
    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

/// The complete computation graph of a neural network model.
///
/// A [`ModelGraph`] is a typed DAG where:
/// - **Nodes** represent operators (e.g., Dense, Conv2D, BatchNorm).
/// - **Edges** represent tensor data-flow between operators.
///
/// # Serialization
///
/// ```rust
/// use scirs2_neural::serialization::model_graph::{
///     ModelGraph, GraphNode, GraphEdge, GraphMetadata, TensorShape, TensorDtype,
///     DenseAttrs, ActivationAttrs,
/// };
///
/// // Build a tiny two-layer MLP graph
/// let metadata = GraphMetadata::new("MLP");
/// let mut graph = ModelGraph::new(metadata);
///
/// graph.add_node(GraphNode::Input {
///     name: "input".to_string(),
///     shape: TensorShape::with_batch_dim(vec![784], TensorDtype::F32),
/// });
/// graph.add_node(GraphNode::Dense {
///     name: "fc1".to_string(),
///     attrs: DenseAttrs { in_features: 784, out_features: 256, use_bias: true },
///     output_shape: TensorShape::with_batch_dim(vec![256], TensorDtype::F32),
/// });
/// graph.add_node(GraphNode::Activation {
///     name: "relu1".to_string(),
///     attrs: ActivationAttrs { function: "relu".to_string(), alpha: None },
///     output_shape: TensorShape::with_batch_dim(vec![256], TensorDtype::F32),
/// });
/// graph.add_node(GraphNode::Dense {
///     name: "fc2".to_string(),
///     attrs: DenseAttrs { in_features: 256, out_features: 10, use_bias: true },
///     output_shape: TensorShape::with_batch_dim(vec![10], TensorDtype::F32),
/// });
/// graph.add_node(GraphNode::Output {
///     name: "output".to_string(),
///     shape: TensorShape::with_batch_dim(vec![10], TensorDtype::F32),
/// });
///
/// graph.add_edge(GraphEdge::simple("input", "fc1"));
/// graph.add_edge(GraphEdge::simple("fc1", "relu1"));
/// graph.add_edge(GraphEdge::simple("relu1", "fc2"));
/// graph.add_edge(GraphEdge::simple("fc2", "output"));
///
/// // Serialize
/// let json = graph.export_to_json().expect("export failed");
/// assert!(!json.is_empty());
///
/// // Round-trip
/// let restored = ModelGraph::import_from_json(&json).expect("import failed");
/// assert_eq!(restored.nodes().len(), graph.nodes().len());
/// assert_eq!(restored.edges().len(), graph.edges().len());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGraph {
    /// Graph-level metadata.
    pub metadata: GraphMetadata,
    /// Ordered list of nodes; the insertion order defines topological sort hints.
    nodes: Vec<GraphNode>,
    /// Directed edges (data-flow connections).
    edges: Vec<GraphEdge>,
}

impl ModelGraph {
    /// Create an empty graph with the given metadata.
    pub fn new(metadata: GraphMetadata) -> Self {
        Self {
            metadata,
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Append a node to the graph.
    ///
    /// Returns an error if a node with the same name already exists.
    pub fn add_node(&mut self, node: GraphNode) -> Result<()> {
        let name = node.name().to_string();
        if self.nodes.iter().any(|n| n.name() == name) {
            return Err(NeuralError::InvalidArgument(format!(
                "Node '{name}' already exists in the graph"
            )));
        }
        self.nodes.push(node);
        Ok(())
    }

    /// Append a directed edge to the graph.
    ///
    /// Returns an error if either the source or destination node does not exist.
    pub fn add_edge(&mut self, edge: GraphEdge) -> Result<()> {
        let from_exists = self.nodes.iter().any(|n| n.name() == edge.from_node);
        let to_exists = self.nodes.iter().any(|n| n.name() == edge.to_node);
        if !from_exists {
            return Err(NeuralError::InvalidArgument(format!(
                "Source node '{}' not found in graph",
                edge.from_node
            )));
        }
        if !to_exists {
            return Err(NeuralError::InvalidArgument(format!(
                "Destination node '{}' not found in graph",
                edge.to_node
            )));
        }
        self.edges.push(edge);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Read-only accessors
    // -----------------------------------------------------------------------

    /// Return a reference to the ordered node list.
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Return a reference to the edge list.
    pub fn edges(&self) -> &[GraphEdge] {
        &self.edges
    }

    /// Look up a node by name.
    pub fn find_node(&self, name: &str) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.name() == name)
    }

    /// Return all edges whose source is `node_name`.
    pub fn outgoing_edges(&self, node_name: &str) -> Vec<&GraphEdge> {
        self.edges
            .iter()
            .filter(|e| e.from_node == node_name)
            .collect()
    }

    /// Return all edges whose destination is `node_name`.
    pub fn incoming_edges(&self, node_name: &str) -> Vec<&GraphEdge> {
        self.edges
            .iter()
            .filter(|e| e.to_node == node_name)
            .collect()
    }

    /// Return the total number of scalar parameters (weights) in the graph.
    ///
    /// Returns `None` if any output shape contains a dynamic dimension.
    pub fn total_parameters(&self) -> Option<i64> {
        let mut total = 0i64;
        for node in &self.nodes {
            let w = node.num_weight_tensors();
            if w == 0 {
                continue;
            }
            let elems = node.output_shape().num_elements()?;
            total = total.checked_add(elems * w as i64)?;
        }
        Some(total)
    }

    /// Validate the graph for structural consistency.
    ///
    /// Checks performed:
    /// - All edge endpoints reference existing nodes.
    /// - No duplicate node names.
    /// - At least one `Input` and one `Output` node exist.
    pub fn validate(&self) -> Result<()> {
        // Check for duplicate names
        let mut seen_names: HashSet<&str> = HashSet::new();
        for node in &self.nodes {
            if !seen_names.insert(node.name()) {
                return Err(NeuralError::ValidationError(format!(
                    "Duplicate node name: '{}'",
                    node.name()
                )));
            }
        }

        // Check edge endpoints
        for edge in &self.edges {
            if !seen_names.contains(edge.from_node.as_str()) {
                return Err(NeuralError::ValidationError(format!(
                    "Edge references unknown source node: '{}'",
                    edge.from_node
                )));
            }
            if !seen_names.contains(edge.to_node.as_str()) {
                return Err(NeuralError::ValidationError(format!(
                    "Edge references unknown destination node: '{}'",
                    edge.to_node
                )));
            }
        }

        // Check for at least one input and one output
        let has_input = self
            .nodes
            .iter()
            .any(|n| matches!(n, GraphNode::Input { .. }));
        let has_output = self
            .nodes
            .iter()
            .any(|n| matches!(n, GraphNode::Output { .. }));
        if !has_input {
            return Err(NeuralError::ValidationError(
                "Graph must have at least one Input node".to_string(),
            ));
        }
        if !has_output {
            return Err(NeuralError::ValidationError(
                "Graph must have at least one Output node".to_string(),
            ));
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // JSON serialization
    // -----------------------------------------------------------------------

    /// Export the graph to a JSON string.
    ///
    /// The output is a pretty-printed JSON object containing:
    /// - `"metadata"` — graph-level metadata
    /// - `"nodes"` — ordered list of serialized [`GraphNode`]s
    /// - `"edges"` — list of serialized [`GraphEdge`]s
    pub fn export_to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))
    }

    /// Import a graph from a JSON string produced by [`ModelGraph::export_to_json`].
    pub fn import_from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| NeuralError::DeserializationError(e.to_string()))
    }

    /// Save the graph to a JSON file at the given path.
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let json = self.export_to_json()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| NeuralError::IOError(e.to_string()))?;
        }
        fs::write(path, json.as_bytes()).map_err(|e| NeuralError::IOError(e.to_string()))
    }

    /// Load the graph from a JSON file.
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| NeuralError::IOError(e.to_string()))?;
        let json =
            std::str::from_utf8(&bytes).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Self::import_from_json(json)
    }

    // -----------------------------------------------------------------------
    // ONNX-like export
    // -----------------------------------------------------------------------

    /// Export the graph to a simplified ONNX-like JSON envelope.
    ///
    /// The produced JSON follows a structure inspired by the ONNX model format
    /// (but expressed in JSON rather than protobuf):
    ///
    /// ```json
    /// {
    ///   "ir_version": 8,
    ///   "opset_import": [{"domain": "", "version": 17}],
    ///   "model_version": 1,
    ///   "doc_string": "...",
    ///   "graph": {
    ///     "name": "...",
    ///     "input": [...],
    ///     "output": [...],
    ///     "node": [...],
    ///     "value_info": [...]
    ///   }
    /// }
    /// ```
    ///
    /// This format is not binary-compatible with the ONNX protobuf standard, but
    /// it is easily convertible by tools that accept JSON ONNX representations.
    pub fn export_onnx_like(&self) -> Result<String> {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut onnx_nodes: Vec<serde_json::Value> = Vec::new();
        let mut value_info: Vec<serde_json::Value> = Vec::new();

        for node in &self.nodes {
            let shape_dims: Vec<serde_json::Value> = node
                .output_shape()
                .dims
                .iter()
                .map(|d| match d {
                    Some(v) => serde_json::json!({ "dim_value": v }),
                    None => serde_json::json!({ "dim_param": "batch_size" }),
                })
                .collect();

            let type_proto = serde_json::json!({
                "tensor_type": {
                    "elem_type": onnx_elem_type(&node.output_shape().dtype),
                    "shape": { "dim": shape_dims }
                }
            });

            match node {
                GraphNode::Input { name, .. } => {
                    inputs.push(serde_json::json!({
                        "name": name,
                        "type": type_proto
                    }));
                }
                GraphNode::Output { name, .. } => {
                    outputs.push(serde_json::json!({
                        "name": name,
                        "type": type_proto
                    }));
                }
                other => {
                    // Build the input/output name lists from edges
                    let input_names: Vec<String> = self
                        .incoming_edges(other.name())
                        .into_iter()
                        .map(|e| e.from_node.clone())
                        .collect();
                    let output_names = vec![other.name().to_string()];

                    // Collect attributes as ONNX attribute list
                    let attrs = onnx_node_attributes(other);

                    onnx_nodes.push(serde_json::json!({
                        "op_type": other.onnx_op_type(),
                        "name": other.name(),
                        "input": input_names,
                        "output": output_names,
                        "attribute": attrs,
                    }));

                    // Add to value_info (intermediate tensors)
                    value_info.push(serde_json::json!({
                        "name": other.name(),
                        "type": type_proto
                    }));
                }
            }
        }

        let doc_string = self
            .metadata
            .description
            .clone()
            .unwrap_or_else(|| format!("{} model graph", self.metadata.architecture));

        let onnx_model = serde_json::json!({
            "ir_version": 8,
            "opset_import": [{ "domain": "", "version": 17 }],
            "model_version": 1,
            "producer_name": "scirs2-neural",
            "producer_version": env!("CARGO_PKG_VERSION"),
            "domain": "ai.onnx",
            "doc_string": doc_string,
            "graph": {
                "name": self.metadata.architecture,
                "doc_string": doc_string,
                "input": inputs,
                "output": outputs,
                "node": onnx_nodes,
                "value_info": value_info,
            }
        });

        serde_json::to_string_pretty(&onnx_model)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))
    }
}

// ============================================================================
// ONNX helpers (private)
// ============================================================================

/// Map a [`TensorDtype`] to an ONNX element type integer
/// (as defined by `onnx.TensorProto.DataType`).
fn onnx_elem_type(dtype: &TensorDtype) -> u32 {
    match dtype {
        TensorDtype::F32 => 1,  // FLOAT
        TensorDtype::F64 => 11, // DOUBLE
        TensorDtype::F16 => 10, // FLOAT16
        TensorDtype::I32 => 6,  // INT32
        TensorDtype::I64 => 7,  // INT64
        TensorDtype::Bool => 9, // BOOL
    }
}

/// Collect ONNX-style attribute objects from a [`GraphNode`].
fn onnx_node_attributes(node: &GraphNode) -> Vec<serde_json::Value> {
    let mut attrs = Vec::new();
    match node {
        GraphNode::Conv2D { attrs: a, .. } | GraphNode::DepthwiseConv2D { attrs: a, .. } => {
            attrs.push(serde_json::json!({
                "name": "kernel_shape",
                "type": "INTS",
                "ints": [a.kernel_h, a.kernel_w]
            }));
            attrs.push(serde_json::json!({
                "name": "strides",
                "type": "INTS",
                "ints": [a.stride_h, a.stride_w]
            }));
            attrs.push(serde_json::json!({
                "name": "pads",
                "type": "INTS",
                "ints": [a.padding.top, a.padding.left, a.padding.bottom, a.padding.right]
            }));
            attrs.push(serde_json::json!({
                "name": "dilations",
                "type": "INTS",
                "ints": [a.dilation, a.dilation]
            }));
            attrs.push(serde_json::json!({
                "name": "group",
                "type": "INT",
                "i": a.groups
            }));
        }
        GraphNode::MaxPool2D { attrs: a, .. } | GraphNode::AvgPool2D { attrs: a, .. } => {
            attrs.push(serde_json::json!({
                "name": "kernel_shape",
                "type": "INTS",
                "ints": [a.kernel_h, a.kernel_w]
            }));
            let sh = a.stride_h.unwrap_or(a.kernel_h);
            let sw = a.stride_w.unwrap_or(a.kernel_w);
            attrs.push(serde_json::json!({
                "name": "strides",
                "type": "INTS",
                "ints": [sh, sw]
            }));
            attrs.push(serde_json::json!({
                "name": "pads",
                "type": "INTS",
                "ints": [a.padding.top, a.padding.left, a.padding.bottom, a.padding.right]
            }));
        }
        GraphNode::BatchNorm { attrs: a, .. } => {
            attrs.push(serde_json::json!({
                "name": "epsilon",
                "type": "FLOAT",
                "f": a.eps
            }));
            attrs.push(serde_json::json!({
                "name": "momentum",
                "type": "FLOAT",
                "f": 1.0 - a.momentum
            }));
        }
        GraphNode::Dropout { attrs: a, .. } => {
            attrs.push(serde_json::json!({
                "name": "seed",
                "type": "INT",
                "i": 0
            }));
            attrs.push(serde_json::json!({
                "name": "ratio",
                "type": "FLOAT",
                "f": a.p
            }));
        }
        GraphNode::Activation { attrs: a, .. } => {
            if let Some(alpha) = a.alpha {
                attrs.push(serde_json::json!({
                    "name": "alpha",
                    "type": "FLOAT",
                    "f": alpha
                }));
            }
        }
        GraphNode::Concat { axis, .. } => {
            attrs.push(serde_json::json!({
                "name": "axis",
                "type": "INT",
                "i": axis
            }));
        }
        GraphNode::Softmax { axis, .. } => {
            attrs.push(serde_json::json!({
                "name": "axis",
                "type": "INT",
                "i": axis
            }));
        }
        GraphNode::Reshape { attrs: a, .. } => {
            attrs.push(serde_json::json!({
                "name": "shape",
                "type": "INTS",
                "ints": a.shape
            }));
        }
        _ => {}
    }
    attrs
}

// ============================================================================
// Builder helpers
// ============================================================================

/// A builder for constructing [`ModelGraph`]s incrementally with a fluent API.
///
/// # Example
///
/// ```rust
/// use scirs2_neural::serialization::model_graph::{
///     ModelGraphBuilder, TensorShape, TensorDtype, DenseAttrs, ActivationAttrs,
/// };
///
/// let graph = ModelGraphBuilder::new("MLP")
///     .input("x", TensorShape::with_batch_dim(vec![784], TensorDtype::F32))
///     .dense("fc1", 784, 128, true, TensorShape::with_batch_dim(vec![128], TensorDtype::F32))
///     .activation("act1", "relu", None, TensorShape::with_batch_dim(vec![128], TensorDtype::F32))
///     .dense("fc2", 128, 10, true, TensorShape::with_batch_dim(vec![10], TensorDtype::F32))
///     .output("y", TensorShape::with_batch_dim(vec![10], TensorDtype::F32))
///     .chain("x", "fc1")
///     .chain("fc1", "act1")
///     .chain("act1", "fc2")
///     .chain("fc2", "y")
///     .build()
///     .expect("build failed");
///
/// assert_eq!(graph.nodes().len(), 5);
/// assert_eq!(graph.edges().len(), 4);
/// ```
pub struct ModelGraphBuilder {
    graph: ModelGraph,
    /// Pending simple chain pairs (from, to) accumulated before `build()`.
    pending_chains: Vec<(String, String)>,
}

impl ModelGraphBuilder {
    /// Start building a new graph with the given architecture name.
    pub fn new(architecture: impl Into<String>) -> Self {
        Self {
            graph: ModelGraph::new(GraphMetadata::new(architecture)),
            pending_chains: Vec::new(),
        }
    }

    /// Add an input node.
    pub fn input(mut self, name: impl Into<String>, shape: TensorShape) -> Self {
        let n = name.into();
        let _ = self.graph.add_node(GraphNode::Input {
            name: n,
            shape,
        });
        self
    }

    /// Add an output node.
    pub fn output(mut self, name: impl Into<String>, shape: TensorShape) -> Self {
        let n = name.into();
        let _ = self.graph.add_node(GraphNode::Output {
            name: n,
            shape,
        });
        self
    }

    /// Add a Dense node.
    pub fn dense(
        mut self,
        name: impl Into<String>,
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        output_shape: TensorShape,
    ) -> Self {
        let n = name.into();
        let _ = self.graph.add_node(GraphNode::Dense {
            name: n,
            attrs: DenseAttrs {
                in_features,
                out_features,
                use_bias,
            },
            output_shape,
        });
        self
    }

    /// Add an Activation node.
    pub fn activation(
        mut self,
        name: impl Into<String>,
        function: impl Into<String>,
        alpha: Option<f64>,
        output_shape: TensorShape,
    ) -> Self {
        let n = name.into();
        let _ = self.graph.add_node(GraphNode::Activation {
            name: n,
            attrs: ActivationAttrs {
                function: function.into(),
                alpha,
            },
            output_shape,
        });
        self
    }

    /// Add a Conv2D node.
    pub fn conv2d(
        mut self,
        name: impl Into<String>,
        attrs: Conv2DAttrs,
        output_shape: TensorShape,
    ) -> Self {
        let n = name.into();
        let _ = self.graph.add_node(GraphNode::Conv2D {
            name: n,
            attrs,
            output_shape,
        });
        self
    }

    /// Add a BatchNorm node.
    pub fn batch_norm(
        mut self,
        name: impl Into<String>,
        attrs: BatchNormAttrs,
        output_shape: TensorShape,
    ) -> Self {
        let n = name.into();
        let _ = self.graph.add_node(GraphNode::BatchNorm {
            name: n,
            attrs,
            output_shape,
        });
        self
    }

    /// Add a simple linear chain edge: `from → to`.
    pub fn chain(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.pending_chains.push((from.into(), to.into()));
        self
    }

    /// Add an explicit edge between two slots.
    pub fn edge(mut self, edge: GraphEdge) -> Self {
        let _ = self.graph.add_edge(edge);
        self
    }

    /// Consume the builder and produce a validated [`ModelGraph`].
    pub fn build(mut self) -> Result<ModelGraph> {
        // Flush pending simple-chain edges
        let chains = std::mem::take(&mut self.pending_chains);
        for (from, to) in chains {
            self.graph.add_edge(GraphEdge::simple(from, to))?;
        }
        self.graph.validate()?;
        Ok(self.graph)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mlp_graph() -> ModelGraph {
        ModelGraphBuilder::new("MLP")
            .input(
                "x",
                TensorShape::with_batch_dim(vec![784], TensorDtype::F32),
            )
            .dense(
                "fc1",
                784,
                256,
                true,
                TensorShape::with_batch_dim(vec![256], TensorDtype::F32),
            )
            .activation(
                "relu1",
                "relu",
                None,
                TensorShape::with_batch_dim(vec![256], TensorDtype::F32),
            )
            .dense(
                "fc2",
                256,
                10,
                true,
                TensorShape::with_batch_dim(vec![10], TensorDtype::F32),
            )
            .output(
                "y",
                TensorShape::with_batch_dim(vec![10], TensorDtype::F32),
            )
            .chain("x", "fc1")
            .chain("fc1", "relu1")
            .chain("relu1", "fc2")
            .chain("fc2", "y")
            .build()
            .expect("build failed")
    }

    #[test]
    fn test_model_graph_builder_basic() {
        let graph = make_mlp_graph();
        assert_eq!(graph.nodes().len(), 5);
        assert_eq!(graph.edges().len(), 4);
    }

    #[test]
    fn test_graph_json_roundtrip() {
        let graph = make_mlp_graph();
        let json = graph.export_to_json().expect("export failed");
        assert!(!json.is_empty());
        let restored = ModelGraph::import_from_json(&json).expect("import failed");
        assert_eq!(restored.nodes().len(), graph.nodes().len());
        assert_eq!(restored.edges().len(), graph.edges().len());
        assert_eq!(restored.metadata.architecture, "MLP");
    }

    #[test]
    fn test_graph_file_roundtrip() {
        let graph = make_mlp_graph();
        let dir = std::env::temp_dir().join("scirs2_model_graph_test");
        let path = dir.join("mlp_graph.json");
        graph.save_to_file(&path).expect("save failed");
        let loaded = ModelGraph::load_from_file(&path).expect("load failed");
        assert_eq!(loaded.nodes().len(), graph.nodes().len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_graph_onnx_like_export() {
        let graph = make_mlp_graph();
        let onnx_json = graph.export_onnx_like().expect("onnx export failed");
        let parsed: serde_json::Value =
            serde_json::from_str(&onnx_json).expect("onnx json should be valid json");
        assert_eq!(parsed["ir_version"], 8);
        assert!(parsed["graph"]["node"].is_array());
        assert!(parsed["graph"]["input"].is_array());
        assert!(parsed["graph"]["output"].is_array());
    }

    #[test]
    fn test_graph_validate_missing_input() {
        let mut graph = ModelGraph::new(GraphMetadata::new("Test"));
        graph
            .add_node(GraphNode::Output {
                name: "out".to_string(),
                shape: TensorShape::new(vec![10], TensorDtype::F32),
            })
            .expect("add node");
        assert!(graph.validate().is_err());
    }

    #[test]
    fn test_graph_validate_missing_output() {
        let mut graph = ModelGraph::new(GraphMetadata::new("Test"));
        graph
            .add_node(GraphNode::Input {
                name: "in".to_string(),
                shape: TensorShape::new(vec![10], TensorDtype::F32),
            })
            .expect("add node");
        assert!(graph.validate().is_err());
    }

    #[test]
    fn test_graph_validate_bad_edge() {
        let mut graph = ModelGraph::new(GraphMetadata::new("Test"));
        graph
            .add_node(GraphNode::Input {
                name: "in".to_string(),
                shape: TensorShape::new(vec![10], TensorDtype::F32),
            })
            .expect("add node");
        graph
            .add_node(GraphNode::Output {
                name: "out".to_string(),
                shape: TensorShape::new(vec![10], TensorDtype::F32),
            })
            .expect("add node");
        let result = graph.add_edge(GraphEdge::simple("nonexistent", "out"));
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_node_rejected() {
        let mut graph = ModelGraph::new(GraphMetadata::new("Test"));
        graph
            .add_node(GraphNode::Input {
                name: "x".to_string(),
                shape: TensorShape::new(vec![10], TensorDtype::F32),
            })
            .expect("first add");
        let result = graph.add_node(GraphNode::Input {
            name: "x".to_string(),
            shape: TensorShape::new(vec![10], TensorDtype::F32),
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_shape_static_num_elements() {
        let shape = TensorShape::new(vec![3, 4, 5], TensorDtype::F32);
        assert_eq!(shape.num_elements(), Some(60));
    }

    #[test]
    fn test_tensor_shape_dynamic_num_elements() {
        let shape = TensorShape::with_batch_dim(vec![3, 4], TensorDtype::F32);
        assert_eq!(shape.num_elements(), None);
    }

    #[test]
    fn test_conv2d_onnx_attributes() {
        let graph = ModelGraphBuilder::new("ConvNet")
            .input(
                "x",
                TensorShape::with_batch_dim(vec![3, 224, 224], TensorDtype::F32),
            )
            .conv2d(
                "conv1",
                Conv2DAttrs {
                    in_channels: 3,
                    out_channels: 64,
                    kernel_h: 3,
                    kernel_w: 3,
                    stride_h: 1,
                    stride_w: 1,
                    padding: PaddingSpec::same(1),
                    dilation: 1,
                    groups: 1,
                    use_bias: false,
                },
                TensorShape::with_batch_dim(vec![64, 224, 224], TensorDtype::F32),
            )
            .output(
                "y",
                TensorShape::with_batch_dim(vec![64, 224, 224], TensorDtype::F32),
            )
            .chain("x", "conv1")
            .chain("conv1", "y")
            .build()
            .expect("build");

        let onnx = graph.export_onnx_like().expect("onnx export");
        let val: serde_json::Value = serde_json::from_str(&onnx).expect("parse");
        let node = &val["graph"]["node"][0];
        assert_eq!(node["op_type"], "Conv");
        let attrs: Vec<serde_json::Value> =
            serde_json::from_value(node["attribute"].clone()).expect("parse attrs");
        let kernel_attr = attrs
            .iter()
            .find(|a| a["name"] == "kernel_shape")
            .expect("kernel_shape attr");
        assert_eq!(kernel_attr["ints"], serde_json::json!([3, 3]));
    }

    #[test]
    fn test_dtype_roundtrip() {
        let dtypes = [
            TensorDtype::F32,
            TensorDtype::F64,
            TensorDtype::F16,
            TensorDtype::I32,
            TensorDtype::I64,
            TensorDtype::Bool,
        ];
        for dt in &dtypes {
            let s = dt.onnx_str();
            let restored = TensorDtype::from_onnx_str(s).expect("roundtrip");
            assert_eq!(dt, &restored);
        }
    }

    #[test]
    fn test_graph_node_name_accessor() {
        let node = GraphNode::Dense {
            name: "my_layer".to_string(),
            attrs: DenseAttrs {
                in_features: 10,
                out_features: 5,
                use_bias: true,
            },
            output_shape: TensorShape::new(vec![5], TensorDtype::F32),
        };
        assert_eq!(node.name(), "my_layer");
        assert_eq!(node.onnx_op_type(), "Gemm");
        assert_eq!(node.num_weight_tensors(), 2); // weight + bias
    }
}
