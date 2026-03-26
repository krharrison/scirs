//! Core types for the static computation graph.
//!
//! Defines the node and graph structures used to represent a traced neural
//! network as a directed acyclic graph (DAG) of operations.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Data type descriptor
// ---------------------------------------------------------------------------

/// Element type of a tensor.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    #[default]
    F64,
    /// 8-bit signed integer
    I8,
    /// 32-bit signed integer
    I32,
}

// ---------------------------------------------------------------------------
// TensorSpec — shape + dtype descriptor
// ---------------------------------------------------------------------------

/// Describes the shape and element type of a tensor, without holding data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// Dimensions (e.g. `[batch, features]`)
    pub shape: Vec<usize>,
    /// Element type
    pub dtype: DType,
}

impl TensorSpec {
    /// Create a new TensorSpec.
    pub fn new(shape: Vec<usize>, dtype: DType) -> Self {
        Self { shape, dtype }
    }

    /// Compute the total number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

// ---------------------------------------------------------------------------
// OpAttr — operation attribute value
// ---------------------------------------------------------------------------

/// A typed attribute value attached to an operation node.
#[derive(Debug, Clone, PartialEq)]
pub enum OpAttr {
    /// Integer scalar attribute
    Int(i64),
    /// Floating-point scalar attribute
    Float(f64),
    /// List of integer values
    IntList(Vec<i64>),
    /// List of floating-point values
    FloatList(Vec<f64>),
    /// String attribute
    String(String),
}

// ---------------------------------------------------------------------------
// OpType — kinds of operations
// ---------------------------------------------------------------------------

/// The kind of computation performed by an `OpNode`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpType {
    /// Fully-connected linear layer: y = x W^T + b
    Linear,
    /// 1-D convolution
    Conv1d,
    /// Rectified linear unit: max(0, x)
    ReLU,
    /// Sigmoid activation: 1 / (1 + e^{-x})
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Element-wise addition
    Add,
    /// Element-wise multiplication
    Mul,
    /// Reshape (change dimensions without copying data)
    Reshape,
    /// Transpose axes
    Transpose,
    /// Softmax along a given dimension
    Softmax,
    /// Layer normalization
    LayerNorm,
    /// Batch normalization
    BatchNorm,
    /// Fused Linear + ReLU (result of operator fusion)
    FusedLinearReLU,
    /// Constant tensor node (no inputs)
    Constant,
}

// ---------------------------------------------------------------------------
// OpNode — a single operation in the graph
// ---------------------------------------------------------------------------

/// One node in the static computation graph.
#[derive(Debug, Clone)]
pub struct OpNode {
    /// Unique node identifier within the graph
    pub id: usize,
    /// Type of operation
    pub op_type: OpType,
    /// Indices of input nodes (into `StaticGraph::nodes`)
    pub inputs: Vec<usize>,
    /// Indices of output consumers (into `StaticGraph::nodes`)
    pub outputs: Vec<usize>,
    /// Operation-specific attributes (e.g. `"out_features"`, `"eps"`)
    pub attrs: HashMap<String, OpAttr>,
    /// Output tensor specification for this node
    pub output_spec: TensorSpec,
    /// Human-readable name (optional, used for debugging)
    pub name: Option<String>,
}

impl OpNode {
    /// Create a new operation node.
    pub fn new(
        id: usize,
        op_type: OpType,
        inputs: Vec<usize>,
        attrs: HashMap<String, OpAttr>,
        output_spec: TensorSpec,
    ) -> Self {
        Self {
            id,
            op_type,
            inputs,
            outputs: Vec::new(),
            attrs,
            output_spec,
            name: None,
        }
    }

    /// Attach a human-readable name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

// ---------------------------------------------------------------------------
// StaticGraph — the complete traced computation graph
// ---------------------------------------------------------------------------

/// A fully-traced static computation graph.
///
/// Nodes are stored in topological order (earlier nodes first). The graph
/// can be executed by `GraphExecutor` or optimized by `optimize`.
#[derive(Debug, Clone)]
pub struct StaticGraph {
    /// All operation nodes (in topological order after `build()`)
    pub nodes: Vec<OpNode>,
    /// Specifications for the graph-level inputs
    pub inputs: Vec<TensorSpec>,
    /// Specifications for the graph-level outputs
    pub outputs: Vec<TensorSpec>,
    /// Map from node id → index in `nodes` (for O(1) lookup)
    pub(crate) id_to_idx: HashMap<usize, usize>,
    /// Node IDs of graph inputs (placeholder nodes)
    pub input_node_ids: Vec<usize>,
    /// Node IDs of graph outputs
    pub output_node_ids: Vec<usize>,
}

impl StaticGraph {
    /// Create an empty graph.
    pub fn new(inputs: Vec<TensorSpec>, outputs: Vec<TensorSpec>) -> Self {
        Self {
            nodes: Vec::new(),
            inputs,
            outputs,
            id_to_idx: HashMap::new(),
            input_node_ids: Vec::new(),
            output_node_ids: Vec::new(),
        }
    }

    /// Look up a node by its ID.
    pub fn get_node(&self, id: usize) -> Option<&OpNode> {
        self.id_to_idx.get(&id).map(|&idx| &self.nodes[idx])
    }

    /// Number of nodes in the graph.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

// ---------------------------------------------------------------------------
// TraceConfig
// ---------------------------------------------------------------------------

/// Configuration for the tracing and graph-building process.
#[derive(Debug, Clone)]
pub struct TraceConfig {
    /// Whether to run optimization passes after tracing
    pub optimize: bool,
    /// Whether to fold constant expressions during optimization
    pub fold_constants: bool,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            optimize: true,
            fold_constants: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_spec_num_elements() {
        let spec = TensorSpec::new(vec![2, 3, 4], DType::F64);
        assert_eq!(spec.num_elements(), 24);
    }

    #[test]
    fn test_op_node_creation() {
        let spec = TensorSpec::new(vec![4], DType::F64);
        let node = OpNode::new(0, OpType::ReLU, vec![], HashMap::new(), spec.clone());
        assert_eq!(node.id, 0);
        assert_eq!(node.op_type, OpType::ReLU);
        assert!(node.outputs.is_empty());
    }

    #[test]
    fn test_static_graph_lookup() {
        let input_spec = TensorSpec::new(vec![4], DType::F64);
        let out_spec = TensorSpec::new(vec![2], DType::F64);
        let mut graph = StaticGraph::new(vec![input_spec.clone()], vec![out_spec.clone()]);
        let node = OpNode::new(0, OpType::Linear, vec![], HashMap::new(), out_spec);
        graph.nodes.push(node);
        graph.id_to_idx.insert(0, 0);

        assert!(graph.get_node(0).is_some());
        assert!(graph.get_node(99).is_none());
        assert_eq!(graph.num_nodes(), 1);
    }

    #[test]
    fn test_trace_config_default() {
        let cfg = TraceConfig::default();
        assert!(cfg.optimize);
        assert!(cfg.fold_constants);
    }
}
