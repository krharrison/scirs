//! ONNX graph builder with fluent API for constructing ONNX computation graphs.
//!
//! The builder validates the graph structure on `build()` and provides
//! convenience methods for common ONNX operations.

use std::collections::{HashMap, HashSet};

use super::error::{OnnxError, OnnxResult};
use super::types::{
    ConvAttributes, GemmAttributes, OnnxAttribute, OnnxDataType, OnnxGraph, OnnxNode, OnnxTensor,
    PoolAttributes,
};

/// Builder for constructing ONNX computation graphs with validation.
///
/// # Example
/// ```
/// use scirs2_autograd::onnx::{OnnxGraphBuilder, OnnxDataType};
///
/// let graph = OnnxGraphBuilder::new("mlp")
///     .add_input("x", &[-1, 784], OnnxDataType::Float32)
///     .add_input("w1", &[784, 256], OnnxDataType::Float32)
///     .add_input("b1", &[256], OnnxDataType::Float32)
///     .add_matmul("matmul_1", "x", "w1", "hidden_raw")
///     .add_add("add_bias_1", "hidden_raw", "b1", "hidden_biased")
///     .add_relu("relu_1", "hidden_biased", "hidden")
///     .add_output("hidden", &[-1, 256], OnnxDataType::Float32)
///     .build()
///     .expect("Failed to build graph");
///
/// assert_eq!(graph.nodes.len(), 3);
/// assert_eq!(graph.inputs.len(), 3);
/// ```
pub struct OnnxGraphBuilder {
    name: String,
    nodes: Vec<OnnxNode>,
    inputs: Vec<OnnxTensor>,
    outputs: Vec<OnnxTensor>,
    initializers: Vec<OnnxTensor>,
    doc_string: Option<String>,
    // Track defined tensor names for validation
    defined_names: HashSet<String>,
}

impl OnnxGraphBuilder {
    /// Create a new graph builder
    pub fn new(name: &str) -> Self {
        OnnxGraphBuilder {
            name: name.to_string(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            initializers: Vec::new(),
            doc_string: None,
            defined_names: HashSet::new(),
        }
    }

    /// Set documentation string
    pub fn doc_string(mut self, doc: &str) -> Self {
        self.doc_string = Some(doc.to_string());
        self
    }

    /// Add a graph input specification
    pub fn add_input(mut self, name: &str, dims: &[i64], dtype: OnnxDataType) -> Self {
        self.defined_names.insert(name.to_string());
        self.inputs.push(OnnxTensor::spec(name, dims, dtype));
        self
    }

    /// Add a graph output specification
    pub fn add_output(mut self, name: &str, dims: &[i64], dtype: OnnxDataType) -> Self {
        self.outputs.push(OnnxTensor::spec(name, dims, dtype));
        self
    }

    /// Add an initializer tensor (model weight/constant)
    pub fn add_initializer(mut self, tensor: OnnxTensor) -> Self {
        self.defined_names.insert(tensor.name.clone());
        self.initializers.push(tensor);
        self
    }

    /// Add a generic node
    pub fn add_node(mut self, node: OnnxNode) -> Self {
        for output in &node.outputs {
            self.defined_names.insert(output.clone());
        }
        self.nodes.push(node);
        self
    }

    // ---------------------------------------------------------------
    // Convenience methods for common ONNX operations
    // ---------------------------------------------------------------

    /// Add a MatMul node: output = input_a @ input_b
    pub fn add_matmul(self, name: &str, input_a: &str, input_b: &str, output: &str) -> Self {
        let node = OnnxNode::new(
            "MatMul",
            name,
            vec![input_a.to_string(), input_b.to_string()],
            vec![output.to_string()],
        );
        self.add_node(node)
    }

    /// Add a Gemm (General Matrix Multiply) node: output = alpha * A @ B + beta * C
    pub fn add_gemm(
        self,
        name: &str,
        a: &str,
        b: &str,
        c: &str,
        output: &str,
        attrs: GemmAttributes,
    ) -> Self {
        let mut node = OnnxNode::new(
            "Gemm",
            name,
            vec![a.to_string(), b.to_string(), c.to_string()],
            vec![output.to_string()],
        );
        node.attributes = attrs.to_attributes();
        self.add_node(node)
    }

    /// Add a Relu activation: output = max(0, input)
    pub fn add_relu(self, name: &str, input: &str, output: &str) -> Self {
        let node = OnnxNode::new(
            "Relu",
            name,
            vec![input.to_string()],
            vec![output.to_string()],
        );
        self.add_node(node)
    }

    /// Add a Sigmoid activation: output = 1 / (1 + exp(-input))
    pub fn add_sigmoid(self, name: &str, input: &str, output: &str) -> Self {
        let node = OnnxNode::new(
            "Sigmoid",
            name,
            vec![input.to_string()],
            vec![output.to_string()],
        );
        self.add_node(node)
    }

    /// Add a Tanh activation: output = tanh(input)
    pub fn add_tanh(self, name: &str, input: &str, output: &str) -> Self {
        let node = OnnxNode::new(
            "Tanh",
            name,
            vec![input.to_string()],
            vec![output.to_string()],
        );
        self.add_node(node)
    }

    /// Add a Softmax node: output = softmax(input, axis)
    pub fn add_softmax(self, name: &str, input: &str, output: &str, axis: i64) -> Self {
        let node = OnnxNode::new(
            "Softmax",
            name,
            vec![input.to_string()],
            vec![output.to_string()],
        )
        .with_attribute("axis", OnnxAttribute::Int(axis));
        self.add_node(node)
    }

    /// Add an Add node: output = a + b
    pub fn add_add(self, name: &str, a: &str, b: &str, output: &str) -> Self {
        let node = OnnxNode::new(
            "Add",
            name,
            vec![a.to_string(), b.to_string()],
            vec![output.to_string()],
        );
        self.add_node(node)
    }

    /// Add a Sub node: output = a - b
    pub fn add_sub(self, name: &str, a: &str, b: &str, output: &str) -> Self {
        let node = OnnxNode::new(
            "Sub",
            name,
            vec![a.to_string(), b.to_string()],
            vec![output.to_string()],
        );
        self.add_node(node)
    }

    /// Add a Mul node: output = a * b
    pub fn add_mul(self, name: &str, a: &str, b: &str, output: &str) -> Self {
        let node = OnnxNode::new(
            "Mul",
            name,
            vec![a.to_string(), b.to_string()],
            vec![output.to_string()],
        );
        self.add_node(node)
    }

    /// Add a Div node: output = a / b
    pub fn add_div(self, name: &str, a: &str, b: &str, output: &str) -> Self {
        let node = OnnxNode::new(
            "Div",
            name,
            vec![a.to_string(), b.to_string()],
            vec![output.to_string()],
        );
        self.add_node(node)
    }

    /// Add a Conv node with attributes
    pub fn add_conv(
        self,
        name: &str,
        input: &str,
        weight: &str,
        output: &str,
        attrs: ConvAttributes,
    ) -> Self {
        let mut node = OnnxNode::new(
            "Conv",
            name,
            vec![input.to_string(), weight.to_string()],
            vec![output.to_string()],
        );
        node.attributes = attrs.to_attributes();
        self.add_node(node)
    }

    /// Add a Conv node with bias
    pub fn add_conv_with_bias(
        self,
        name: &str,
        input: &str,
        weight: &str,
        bias: &str,
        output: &str,
        attrs: ConvAttributes,
    ) -> Self {
        let mut node = OnnxNode::new(
            "Conv",
            name,
            vec![input.to_string(), weight.to_string(), bias.to_string()],
            vec![output.to_string()],
        );
        node.attributes = attrs.to_attributes();
        self.add_node(node)
    }

    /// Add a MaxPool node
    pub fn add_max_pool(
        self,
        name: &str,
        input: &str,
        output: &str,
        attrs: PoolAttributes,
    ) -> Self {
        let mut node = OnnxNode::new(
            "MaxPool",
            name,
            vec![input.to_string()],
            vec![output.to_string()],
        );
        node.attributes = attrs.to_attributes();
        self.add_node(node)
    }

    /// Add an AveragePool node
    pub fn add_average_pool(
        self,
        name: &str,
        input: &str,
        output: &str,
        attrs: PoolAttributes,
    ) -> Self {
        let mut node = OnnxNode::new(
            "AveragePool",
            name,
            vec![input.to_string()],
            vec![output.to_string()],
        );
        node.attributes = attrs.to_attributes();
        self.add_node(node)
    }

    /// Add a GlobalAveragePool node
    pub fn add_global_average_pool(self, name: &str, input: &str, output: &str) -> Self {
        let node = OnnxNode::new(
            "GlobalAveragePool",
            name,
            vec![input.to_string()],
            vec![output.to_string()],
        );
        self.add_node(node)
    }

    /// Add a BatchNormalization node
    pub fn add_batch_norm(
        self,
        name: &str,
        input: &str,
        scale: &str,
        bias: &str,
        mean: &str,
        var: &str,
        output: &str,
        epsilon: f64,
    ) -> Self {
        let node = OnnxNode::new(
            "BatchNormalization",
            name,
            vec![
                input.to_string(),
                scale.to_string(),
                bias.to_string(),
                mean.to_string(),
                var.to_string(),
            ],
            vec![output.to_string()],
        )
        .with_attribute("epsilon", OnnxAttribute::Float(epsilon));
        self.add_node(node)
    }

    /// Add a Reshape node: output = reshape(data, shape)
    pub fn add_reshape(self, name: &str, data: &str, shape: &str, output: &str) -> Self {
        let node = OnnxNode::new(
            "Reshape",
            name,
            vec![data.to_string(), shape.to_string()],
            vec![output.to_string()],
        );
        self.add_node(node)
    }

    /// Add a Flatten node
    pub fn add_flatten(self, name: &str, input: &str, output: &str, axis: i64) -> Self {
        let node = OnnxNode::new(
            "Flatten",
            name,
            vec![input.to_string()],
            vec![output.to_string()],
        )
        .with_attribute("axis", OnnxAttribute::Int(axis));
        self.add_node(node)
    }

    /// Add a Transpose node
    pub fn add_transpose(self, name: &str, input: &str, output: &str, perm: &[i64]) -> Self {
        let node = OnnxNode::new(
            "Transpose",
            name,
            vec![input.to_string()],
            vec![output.to_string()],
        )
        .with_attribute("perm", OnnxAttribute::Ints(perm.to_vec()));
        self.add_node(node)
    }

    /// Add a Concat node
    pub fn add_concat(self, name: &str, inputs: &[&str], output: &str, axis: i64) -> Self {
        let node = OnnxNode::new(
            "Concat",
            name,
            inputs.iter().map(|s| s.to_string()).collect(),
            vec![output.to_string()],
        )
        .with_attribute("axis", OnnxAttribute::Int(axis));
        self.add_node(node)
    }

    /// Add a ReduceMean node
    pub fn add_reduce_mean(
        self,
        name: &str,
        input: &str,
        output: &str,
        axes: &[i64],
        keepdims: bool,
    ) -> Self {
        let node = OnnxNode::new(
            "ReduceMean",
            name,
            vec![input.to_string()],
            vec![output.to_string()],
        )
        .with_attribute("axes", OnnxAttribute::Ints(axes.to_vec()))
        .with_attribute("keepdims", OnnxAttribute::Int(if keepdims { 1 } else { 0 }));
        self.add_node(node)
    }

    /// Add a ReduceSum node
    pub fn add_reduce_sum(
        self,
        name: &str,
        input: &str,
        output: &str,
        axes: &[i64],
        keepdims: bool,
    ) -> Self {
        let node = OnnxNode::new(
            "ReduceSum",
            name,
            vec![input.to_string()],
            vec![output.to_string()],
        )
        .with_attribute("axes", OnnxAttribute::Ints(axes.to_vec()))
        .with_attribute("keepdims", OnnxAttribute::Int(if keepdims { 1 } else { 0 }));
        self.add_node(node)
    }

    /// Add a Dropout node
    pub fn add_dropout(self, name: &str, input: &str, output: &str, ratio: f64) -> Self {
        let node = OnnxNode::new(
            "Dropout",
            name,
            vec![input.to_string()],
            vec![output.to_string()],
        )
        .with_attribute("ratio", OnnxAttribute::Float(ratio));
        self.add_node(node)
    }

    /// Add a LeakyRelu node
    pub fn add_leaky_relu(self, name: &str, input: &str, output: &str, alpha: f64) -> Self {
        let node = OnnxNode::new(
            "LeakyRelu",
            name,
            vec![input.to_string()],
            vec![output.to_string()],
        )
        .with_attribute("alpha", OnnxAttribute::Float(alpha));
        self.add_node(node)
    }

    /// Add a Clip node (clamp values)
    pub fn add_clip(
        self,
        name: &str,
        input: &str,
        min_name: &str,
        max_name: &str,
        output: &str,
    ) -> Self {
        let node = OnnxNode::new(
            "Clip",
            name,
            vec![
                input.to_string(),
                min_name.to_string(),
                max_name.to_string(),
            ],
            vec![output.to_string()],
        );
        self.add_node(node)
    }

    /// Add a Constant node (creates a constant tensor)
    pub fn add_constant(self, name: &str, output: &str, tensor: OnnxTensor) -> Self {
        let node = OnnxNode::new("Constant", name, vec![], vec![output.to_string()])
            .with_attribute("value", OnnxAttribute::Tensor(tensor));
        self.add_node(node)
    }

    /// Add an Identity node (pass-through)
    pub fn add_identity(self, name: &str, input: &str, output: &str) -> Self {
        let node = OnnxNode::new(
            "Identity",
            name,
            vec![input.to_string()],
            vec![output.to_string()],
        );
        self.add_node(node)
    }

    /// Build the graph, performing validation
    pub fn build(self) -> OnnxResult<OnnxGraph> {
        self.validate()?;

        Ok(OnnxGraph {
            name: self.name,
            nodes: self.nodes,
            inputs: self.inputs,
            outputs: self.outputs,
            initializers: self.initializers,
            doc_string: self.doc_string,
        })
    }

    /// Validate the graph structure
    fn validate(&self) -> OnnxResult<()> {
        // Check that we have at least one input and one output
        if self.inputs.is_empty() {
            return Err(OnnxError::ValidationError(
                "Graph must have at least one input".to_string(),
            ));
        }
        if self.outputs.is_empty() {
            return Err(OnnxError::ValidationError(
                "Graph must have at least one output".to_string(),
            ));
        }

        // Check for duplicate node names
        let mut node_names = HashSet::new();
        for node in &self.nodes {
            if !node.name.is_empty() && !node_names.insert(node.name.clone()) {
                return Err(OnnxError::ValidationError(format!(
                    "Duplicate node name: '{}'",
                    node.name
                )));
            }
        }

        // Check that all output tensor names are produced by some node, input, or initializer
        let mut produced_tensors: HashSet<&str> = HashSet::new();
        for inp in &self.inputs {
            produced_tensors.insert(&inp.name);
        }
        for init in &self.initializers {
            produced_tensors.insert(&init.name);
        }
        for node in &self.nodes {
            // Constant nodes produce outputs from attributes
            if node.op_type == "Constant" {
                for out in &node.outputs {
                    produced_tensors.insert(out);
                }
                continue;
            }
            for out in &node.outputs {
                produced_tensors.insert(out);
            }
        }

        for out_spec in &self.outputs {
            if !produced_tensors.contains(out_spec.name.as_str()) {
                return Err(OnnxError::ValidationError(format!(
                    "Output tensor '{}' is not produced by any node, input, or initializer",
                    out_spec.name
                )));
            }
        }

        // Check for empty op_type on nodes
        for node in &self.nodes {
            if node.op_type.is_empty() {
                return Err(OnnxError::ValidationError(format!(
                    "Node '{}' has empty op_type",
                    node.name
                )));
            }
        }

        // Validate initializer data consistency
        for init in &self.initializers {
            if init.has_data() {
                if let Some(expected_count) = init.num_elements() {
                    let actual_count = match init.data_type {
                        OnnxDataType::Float32 => init.float_data.len(),
                        OnnxDataType::Float64 => init.double_data.len(),
                        OnnxDataType::Int32 => init.int32_data.len(),
                        OnnxDataType::Int64 => init.int64_data.len(),
                        _ => {
                            // raw_data: check byte count
                            init.raw_data.len() / init.data_type.element_size()
                        }
                    };
                    if actual_count != expected_count && actual_count != 0 {
                        return Err(OnnxError::ValidationError(format!(
                            "Initializer '{}': shape {:?} expects {} elements but got {}",
                            init.name, init.dims, expected_count, actual_count
                        )));
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let graph = OnnxGraphBuilder::new("test")
            .add_input("x", &[1, 10], OnnxDataType::Float32)
            .add_relu("relu_1", "x", "y")
            .add_output("y", &[1, 10], OnnxDataType::Float32)
            .build();
        assert!(graph.is_ok());
        let g = graph.expect("build failed");
        assert_eq!(g.name, "test");
        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.nodes[0].op_type, "Relu");
    }

    #[test]
    fn test_builder_no_inputs() {
        let result = OnnxGraphBuilder::new("bad")
            .add_output("y", &[1], OnnxDataType::Float32)
            .build();
        assert!(result.is_err());
        let err = result.expect_err("should fail");
        assert!(format!("{}", err).contains("at least one input"));
    }

    #[test]
    fn test_builder_no_outputs() {
        let result = OnnxGraphBuilder::new("bad")
            .add_input("x", &[1], OnnxDataType::Float32)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_mlp() {
        let graph = OnnxGraphBuilder::new("mlp")
            .add_input("x", &[-1, 784], OnnxDataType::Float32)
            .add_input("w1", &[784, 256], OnnxDataType::Float32)
            .add_input("b1", &[256], OnnxDataType::Float32)
            .add_input("w2", &[256, 10], OnnxDataType::Float32)
            .add_input("b2", &[10], OnnxDataType::Float32)
            .add_matmul("mm1", "x", "w1", "h1_raw")
            .add_add("add1", "h1_raw", "b1", "h1_bias")
            .add_relu("relu1", "h1_bias", "h1")
            .add_matmul("mm2", "h1", "w2", "h2_raw")
            .add_add("add2", "h2_raw", "b2", "logits")
            .add_softmax("softmax", "logits", "probs", -1)
            .add_output("probs", &[-1, 10], OnnxDataType::Float32)
            .build();

        assert!(graph.is_ok());
        let g = graph.expect("build failed");
        assert_eq!(g.nodes.len(), 6);
    }

    #[test]
    fn test_builder_duplicate_node_names() {
        let result = OnnxGraphBuilder::new("dup")
            .add_input("x", &[1], OnnxDataType::Float32)
            .add_relu("same_name", "x", "y1")
            .add_relu("same_name", "y1", "y2")
            .add_output("y2", &[1], OnnxDataType::Float32)
            .build();
        assert!(result.is_err());
    }
}
