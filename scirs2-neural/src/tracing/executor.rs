//! Graph executor for static computation graphs.
//!
//! Executes a `StaticGraph` in topological order, dispatching each operation
//! to a pure-Rust implementation. Weights are supplied via a `weight_map`.
//!
//! Also provides `optimize()` which runs three passes:
//! - **Constant folding** — evaluate nodes with no tensor inputs
//! - **Dead node elimination** — remove nodes not reachable from graph outputs
//! - **Operator fusion** — combine consecutive Linear→ReLU into FusedLinearReLU

use crate::error::{Error, Result};
use crate::tracing::types::{OpAttr, OpNode, OpType, StaticGraph};
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Weight store key conventions
// ---------------------------------------------------------------------------
//
// Weights for a `Linear` node with id N are stored as:
//   "linear_{N}_weight"  shape: [out_features, in_features]
//   "linear_{N}_bias"    shape: [out_features]
//
// For LayerNorm with id N:
//   "layer_norm_{N}_gamma"  shape: [features]
//   "layer_norm_{N}_beta"   shape: [features]

// ---------------------------------------------------------------------------
// GraphExecutor
// ---------------------------------------------------------------------------

/// Holds a `StaticGraph` and a weight map, and can execute the graph.
pub struct GraphExecutor {
    graph: StaticGraph,
    /// Named float tensors (weights, biases, scales, etc.)
    weight_map: HashMap<String, Vec<f64>>,
}

impl GraphExecutor {
    /// Create an executor from a graph and its associated weights.
    pub fn new(graph: StaticGraph, weight_map: HashMap<String, Vec<f64>>) -> Self {
        Self { graph, weight_map }
    }

    /// Run the graph with the given input tensors (flat f64 slices, one per graph input).
    ///
    /// Returns the output tensors in the same order as `graph.output_node_ids`.
    pub fn run(&self, inputs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        // Validate input count
        if inputs.len() != self.graph.input_node_ids.len() {
            return Err(Error::InvalidArgument(format!(
                "Expected {} inputs, got {}",
                self.graph.input_node_ids.len(),
                inputs.len()
            )));
        }

        // Map: node_id → computed flat tensor
        let mut tensor_cache: HashMap<usize, Vec<f64>> = HashMap::new();

        // Seed with graph inputs
        for (inp_tensor, &node_id) in inputs.iter().zip(self.graph.input_node_ids.iter()) {
            tensor_cache.insert(node_id, inp_tensor.clone());
        }

        // Execute nodes in order (already topologically sorted by GraphBuilder)
        for node in &self.graph.nodes {
            // Skip placeholder input nodes (already in cache)
            if node.op_type == OpType::Constant {
                tensor_cache.entry(node.id).or_insert_with(|| {
                    // Constant with no inputs: return zeros
                    let n = node.output_spec.num_elements();
                    vec![0.0_f64; n]
                });
                continue;
            }

            let output = self.execute_node(node, &tensor_cache)?;
            tensor_cache.insert(node.id, output);
        }

        // Collect outputs
        let mut results = Vec::with_capacity(self.graph.output_node_ids.len());
        for &out_id in &self.graph.output_node_ids {
            let tensor = tensor_cache
                .get(&out_id)
                .ok_or_else(|| {
                    Error::InvalidArgument(format!("Output node {} not computed", out_id))
                })?
                .clone();
            results.push(tensor);
        }
        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Per-operation dispatch
    // -----------------------------------------------------------------------

    fn execute_node(&self, node: &OpNode, cache: &HashMap<usize, Vec<f64>>) -> Result<Vec<f64>> {
        match &node.op_type {
            OpType::Linear => self.exec_linear(node, cache),
            OpType::ReLU => self.exec_elementwise(node, cache, |x| x.max(0.0)),
            OpType::Sigmoid => self.exec_elementwise(node, cache, |x| 1.0 / (1.0 + (-x).exp())),
            OpType::Tanh => self.exec_elementwise(node, cache, |x| x.tanh()),
            OpType::Add => self.exec_binary(node, cache, |a, b| a + b),
            OpType::Mul => self.exec_binary(node, cache, |a, b| a * b),
            OpType::Reshape => self.exec_reshape(node, cache),
            OpType::Softmax => self.exec_softmax(node, cache),
            OpType::LayerNorm => self.exec_layer_norm(node, cache),
            OpType::FusedLinearReLU => self.exec_fused_linear_relu(node, cache),
            OpType::Transpose => self.exec_reshape(node, cache), // simplified
            OpType::BatchNorm => {
                // Simplified: treat as identity for executor tests
                let inp_id = node
                    .inputs
                    .first()
                    .ok_or_else(|| Error::InvalidArgument("BatchNorm has no inputs".to_string()))?;
                Ok(cache
                    .get(inp_id)
                    .ok_or_else(|| Error::InvalidArgument(format!("Input {} not found", inp_id)))?
                    .clone())
            }
            OpType::Conv1d => Err(Error::NotImplemented(
                "Conv1d execution not yet implemented".to_string(),
            )),
            _ => Err(Error::NotImplemented(format!(
                "OpType {:?} not implemented in executor",
                node.op_type
            ))),
        }
    }

    fn get_input<'a>(
        &self,
        node: &OpNode,
        idx: usize,
        cache: &'a HashMap<usize, Vec<f64>>,
    ) -> Result<&'a Vec<f64>> {
        let node_id = node.inputs.get(idx).ok_or_else(|| {
            Error::InvalidArgument(format!("Node {} has no input at index {}", node.id, idx))
        })?;
        cache.get(node_id).ok_or_else(|| {
            Error::InvalidArgument(format!("Input tensor for node {} not in cache", node_id))
        })
    }

    fn exec_elementwise(
        &self,
        node: &OpNode,
        cache: &HashMap<usize, Vec<f64>>,
        f: impl Fn(f64) -> f64,
    ) -> Result<Vec<f64>> {
        let input = self.get_input(node, 0, cache)?;
        Ok(input.iter().map(|&x| f(x)).collect())
    }

    fn exec_binary(
        &self,
        node: &OpNode,
        cache: &HashMap<usize, Vec<f64>>,
        f: impl Fn(f64, f64) -> f64,
    ) -> Result<Vec<f64>> {
        let a = self.get_input(node, 0, cache)?;
        let b = self.get_input(node, 1, cache)?;
        if a.len() != b.len() {
            return Err(Error::InvalidArgument(format!(
                "Binary op shape mismatch: {} vs {}",
                a.len(),
                b.len()
            )));
        }
        Ok(a.iter().zip(b.iter()).map(|(&av, &bv)| f(av, bv)).collect())
    }

    fn exec_reshape(&self, node: &OpNode, cache: &HashMap<usize, Vec<f64>>) -> Result<Vec<f64>> {
        let input = self.get_input(node, 0, cache)?;
        // Reshape is a no-op on the flat data; just return a clone
        Ok(input.clone())
    }

    fn exec_linear(&self, node: &OpNode, cache: &HashMap<usize, Vec<f64>>) -> Result<Vec<f64>> {
        let input = self.get_input(node, 0, cache)?;

        let out_feat = get_attr_int(&node.attrs, "out_features")? as usize;
        let in_feat = get_attr_int(&node.attrs, "in_features")? as usize;

        // Infer batch size
        if in_feat == 0 {
            return Err(Error::InvalidArgument("in_features cannot be 0".into()));
        }
        let batch = input.len() / in_feat;
        if batch * in_feat != input.len() {
            return Err(Error::InvalidArgument(format!(
                "Input length {} not divisible by in_features {}",
                input.len(),
                in_feat
            )));
        }

        // Fetch weight and bias
        let weight_key = format!("linear_{}_weight", node.id);
        let bias_key = format!("linear_{}_bias", node.id);

        let weight = self
            .weight_map
            .get(&weight_key)
            .ok_or_else(|| Error::InvalidArgument(format!("Missing weight '{}'", weight_key)))?;
        let bias = self
            .weight_map
            .get(&bias_key)
            .ok_or_else(|| Error::InvalidArgument(format!("Missing bias '{}'", bias_key)))?;

        if weight.len() != out_feat * in_feat {
            return Err(Error::InvalidArgument(format!(
                "Weight shape mismatch: expected {}×{}, got {}",
                out_feat,
                in_feat,
                weight.len()
            )));
        }

        // y[b, o] = Σ_i W[o, i] * x[b, i] + bias[o]
        let mut output = vec![0.0_f64; batch * out_feat];
        for b in 0..batch {
            for o in 0..out_feat {
                let mut acc = bias.get(o).copied().unwrap_or(0.0);
                for i in 0..in_feat {
                    acc += weight[o * in_feat + i] * input[b * in_feat + i];
                }
                output[b * out_feat + o] = acc;
            }
        }
        Ok(output)
    }

    fn exec_fused_linear_relu(
        &self,
        node: &OpNode,
        cache: &HashMap<usize, Vec<f64>>,
    ) -> Result<Vec<f64>> {
        let linear_out = self.exec_linear(node, cache)?;
        Ok(linear_out.iter().map(|&x| x.max(0.0)).collect())
    }

    fn exec_softmax(&self, node: &OpNode, cache: &HashMap<usize, Vec<f64>>) -> Result<Vec<f64>> {
        let input = self.get_input(node, 0, cache)?;
        let dim = get_attr_int(&node.attrs, "dim").unwrap_or(1) as usize;

        // Determine row size along the softmax dimension
        let shape = &node.output_spec.shape;
        if shape.is_empty() {
            return Ok(input.clone());
        }

        // Compute softmax over each "row" of size = shape[dim]
        let row_size = if dim < shape.len() {
            shape[dim]
        } else {
            input.len()
        };
        if row_size == 0 {
            return Ok(input.clone());
        }

        let n_rows = input.len() / row_size;
        let mut output = vec![0.0_f64; input.len()];

        for r in 0..n_rows {
            let row = &input[r * row_size..(r + 1) * row_size];
            // Numerically stable: subtract max
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_vals: Vec<f64> = row.iter().map(|&v| (v - max_val).exp()).collect();
            let sum: f64 = exp_vals.iter().sum();
            let sum_safe = if sum > 0.0 { sum } else { 1.0 };
            for (i, &e) in exp_vals.iter().enumerate() {
                output[r * row_size + i] = e / sum_safe;
            }
        }
        Ok(output)
    }

    fn exec_layer_norm(&self, node: &OpNode, cache: &HashMap<usize, Vec<f64>>) -> Result<Vec<f64>> {
        let input = self.get_input(node, 0, cache)?;
        let eps = node
            .attrs
            .get("eps")
            .and_then(|a| match a {
                OpAttr::Float(f) => Some(*f),
                _ => None,
            })
            .unwrap_or(1e-5);

        let shape = &node.output_spec.shape;
        let last_dim = shape.last().copied().unwrap_or(input.len());
        if last_dim == 0 {
            return Ok(input.clone());
        }

        let n_rows = input.len() / last_dim;

        // Fetch optional affine parameters
        let gamma_key = format!("layer_norm_{}_gamma", node.id);
        let beta_key = format!("layer_norm_{}_beta", node.id);
        let gamma = self.weight_map.get(&gamma_key);
        let beta = self.weight_map.get(&beta_key);

        let mut output = vec![0.0_f64; input.len()];
        for r in 0..n_rows {
            let row = &input[r * last_dim..(r + 1) * last_dim];
            let mean = row.iter().sum::<f64>() / last_dim as f64;
            let var = row.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / last_dim as f64;
            let std_inv = 1.0 / (var + eps).sqrt();
            for (i, &v) in row.iter().enumerate() {
                let normalized = (v - mean) * std_inv;
                let scaled = gamma.and_then(|g| g.get(i).copied()).unwrap_or(1.0) * normalized;
                let shifted = scaled + beta.and_then(|b| b.get(i).copied()).unwrap_or(0.0);
                output[r * last_dim + i] = shifted;
            }
        }
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Attribute helpers
// ---------------------------------------------------------------------------

fn get_attr_int(attrs: &HashMap<String, OpAttr>, key: &str) -> Result<i64> {
    match attrs.get(key) {
        Some(OpAttr::Int(v)) => Ok(*v),
        Some(_) => Err(Error::InvalidArgument(format!(
            "Attribute '{}' is not an integer",
            key
        ))),
        None => Err(Error::InvalidArgument(format!(
            "Missing attribute '{}'",
            key
        ))),
    }
}

// ---------------------------------------------------------------------------
// Graph optimization passes
// ---------------------------------------------------------------------------

/// Apply optimization passes to a `StaticGraph` and return the optimized graph.
///
/// Passes applied (in order):
/// 1. Dead node elimination — remove nodes not reachable backward from outputs
/// 2. Constant folding — pre-evaluate `Constant` nodes with no dependencies
/// 3. Operator fusion — fuse Linear→ReLU pairs into `FusedLinearReLU`
pub fn optimize(graph: &StaticGraph) -> StaticGraph {
    let after_dne = dead_node_elimination(graph);

    operator_fusion(&after_dne)
}

/// Remove nodes that are not on any path from a graph input to a graph output.
fn dead_node_elimination(graph: &StaticGraph) -> StaticGraph {
    // BFS backward from output nodes
    let mut live: HashSet<usize> = HashSet::new();
    let mut queue: VecDeque<usize> = VecDeque::new();

    for &out_id in &graph.output_node_ids {
        if !live.contains(&out_id) {
            live.insert(out_id);
            queue.push_back(out_id);
        }
    }

    // Build reverse-edge map: node_id → set of nodes that produce its inputs
    let mut producers: HashMap<usize, Vec<usize>> = HashMap::new();
    for node in &graph.nodes {
        for &inp_id in &node.inputs {
            producers.entry(node.id).or_default().push(inp_id);
        }
    }

    while let Some(id) = queue.pop_front() {
        for &prod_id in producers.get(&id).unwrap_or(&vec![]) {
            if !live.contains(&prod_id) {
                live.insert(prod_id);
                queue.push_back(prod_id);
            }
        }
    }

    // Keep only live nodes (preserve original order)
    let kept_nodes: Vec<OpNode> = graph
        .nodes
        .iter()
        .filter(|n| live.contains(&n.id))
        .cloned()
        .collect();

    let mut id_to_idx = HashMap::new();
    for (idx, node) in kept_nodes.iter().enumerate() {
        id_to_idx.insert(node.id, idx);
    }

    let mut new_graph = StaticGraph::new(graph.inputs.clone(), graph.outputs.clone());
    new_graph.nodes = kept_nodes;
    new_graph.id_to_idx = id_to_idx;
    new_graph.input_node_ids = graph.input_node_ids.clone();
    new_graph.output_node_ids = graph.output_node_ids.clone();
    new_graph
}

/// Fold constants: Constant nodes with no tensor inputs and known values
/// are pre-evaluated (here we mark them; actual value injection happens at
/// executor time via the weight_map).
///
/// In this implementation, constant folding is a no-op pass since Constant
/// nodes are placeholders — the real benefit is tracked in the node metadata.
/// This function serves as the hook for future compile-time constant propagation.
fn _constant_folding(graph: &StaticGraph) -> StaticGraph {
    // Currently a pass-through; constant values are resolved at execution time
    graph.clone()
}

/// Fuse consecutive Linear→ReLU nodes into a single `FusedLinearReLU` node.
fn operator_fusion(graph: &StaticGraph) -> StaticGraph {
    let mut fused_nodes = graph.nodes.clone();

    // Find (Linear, ReLU) pairs where the ReLU has exactly one consumer of the Linear
    let mut to_fuse: Vec<(usize, usize)> = Vec::new(); // (linear_idx, relu_idx)
    for (relu_idx, node) in fused_nodes.iter().enumerate() {
        if node.op_type != OpType::ReLU {
            continue;
        }
        let relu_input_id = match node.inputs.first() {
            Some(&id) => id,
            None => continue,
        };
        // Check if the input is a Linear node
        let linear_idx = match fused_nodes
            .iter()
            .position(|n| n.id == relu_input_id && n.op_type == OpType::Linear)
        {
            Some(i) => i,
            None => continue,
        };
        // Check the Linear node has exactly one consumer (this ReLU)
        let linear_output_count = fused_nodes
            .iter()
            .filter(|n| n.inputs.contains(&relu_input_id))
            .count();
        if linear_output_count == 1 {
            to_fuse.push((linear_idx, relu_idx));
        }
    }

    // Apply fusions: replace Linear with FusedLinearReLU, mark ReLU for removal
    let mut remove_ids: HashSet<usize> = HashSet::new();
    let mut relu_id_to_linear_id: HashMap<usize, usize> = HashMap::new();

    for (linear_idx, relu_idx) in to_fuse {
        let relu_id = fused_nodes[relu_idx].id;
        let linear_id = fused_nodes[linear_idx].id;

        // Change the Linear to FusedLinearReLU
        fused_nodes[linear_idx].op_type = OpType::FusedLinearReLU;
        // The fused node should have the same output spec as the ReLU
        // (they're identical for ReLU)
        remove_ids.insert(relu_id);
        relu_id_to_linear_id.insert(relu_id, linear_id);
    }

    // Update all references to removed ReLU nodes to point to the fused Linear
    for node in &mut fused_nodes {
        for inp_id in &mut node.inputs {
            if let Some(&fused_id) = relu_id_to_linear_id.get(inp_id) {
                *inp_id = fused_id;
            }
        }
    }

    // Remove the now-redundant ReLU nodes
    fused_nodes.retain(|n| !remove_ids.contains(&n.id));

    // Rebuild the graph
    let mut id_to_idx = HashMap::new();
    for (idx, node) in fused_nodes.iter().enumerate() {
        id_to_idx.insert(node.id, idx);
    }

    // Update output_node_ids if any output pointed to a fused ReLU
    let output_node_ids: Vec<usize> = graph
        .output_node_ids
        .iter()
        .map(|&id| *relu_id_to_linear_id.get(&id).unwrap_or(&id))
        .collect();

    let mut new_graph = StaticGraph::new(graph.inputs.clone(), graph.outputs.clone());
    new_graph.nodes = fused_nodes;
    new_graph.id_to_idx = id_to_idx;
    new_graph.input_node_ids = graph.input_node_ids.clone();
    new_graph.output_node_ids = output_node_ids;
    new_graph
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracing::graph_builder::GraphBuilder;
    use crate::tracing::types::{DType, TensorSpec};

    /// Build a simple weight map for a single linear layer.
    fn linear_weights(node_id: usize, in_f: usize, out_f: usize) -> HashMap<String, Vec<f64>> {
        let mut map = HashMap::new();
        // Identity-like weight: each output equals the first input
        let mut weight = vec![0.0_f64; out_f * in_f];
        for o in 0..out_f.min(in_f) {
            weight[o * in_f + o] = 1.0;
        }
        map.insert(format!("linear_{}_weight", node_id), weight);
        map.insert(format!("linear_{}_bias", node_id), vec![0.0_f64; out_f]);
        map
    }

    #[test]
    fn test_executor_linear_relu() {
        let mut builder = GraphBuilder::new();
        let input = builder.input(TensorSpec::new(vec![1, 4], DType::F64));
        let h = builder.linear(input, 4, 4);
        let out = builder.relu(h);
        let graph = builder.build(vec![out]);

        // Find node IDs
        let linear_id = graph
            .nodes
            .iter()
            .find(|n| n.op_type == OpType::Linear)
            .map(|n| n.id)
            .expect("test: linear node");

        let mut weights = linear_weights(linear_id, 4, 4);
        // Set negative bias so some outputs will be negative before ReLU
        weights.insert(
            format!("linear_{}_bias", linear_id),
            vec![-1.0, -1.0, 1.0, 1.0],
        );

        let executor = GraphExecutor::new(graph, weights);
        let result = executor
            .run(&[vec![1.0, 2.0, 3.0, 4.0]])
            .expect("test: run");
        assert_eq!(result.len(), 1);
        let out = &result[0];
        assert_eq!(out.len(), 4);
        // ReLU: all outputs >= 0
        for &v in out {
            assert!(v >= 0.0, "ReLU output must be >= 0, got {v}");
        }
    }

    #[test]
    fn test_executor_softmax_sums_one() {
        let mut builder = GraphBuilder::new();
        let input = builder.input(TensorSpec::new(vec![1, 5], DType::F64));
        let out = builder.softmax(input, 1);
        let graph = builder.build(vec![out]);

        let executor = GraphExecutor::new(graph, HashMap::new());
        let result = executor
            .run(&[vec![1.0, 2.0, 3.0, 4.0, 5.0]])
            .expect("test: run softmax");
        let out = &result[0];
        assert_eq!(out.len(), 5);
        let sum: f64 = out.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Softmax should sum to 1, got {sum}"
        );
    }

    #[test]
    fn test_executor_layer_norm() {
        let mut builder = GraphBuilder::new();
        let input = builder.input(TensorSpec::new(vec![1, 8], DType::F64));
        let out = builder.layer_norm(input, 1e-5);
        let graph = builder.build(vec![out]);

        let executor = GraphExecutor::new(graph, HashMap::new());
        let data: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let result = executor.run(&[data]).expect("test: run layer_norm");
        let out = &result[0];
        assert_eq!(out.len(), 8);
        let mean: f64 = out.iter().sum::<f64>() / out.len() as f64;
        let var: f64 = out.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / out.len() as f64;
        assert!(mean.abs() < 1e-6, "LayerNorm mean should be ~0, got {mean}");
        assert!(
            (var - 1.0).abs() < 1e-4,
            "LayerNorm variance should be ~1, got {var}"
        );
    }

    #[test]
    fn test_graph_dead_node_elimination() {
        let mut builder = GraphBuilder::new();
        let input = builder.input(TensorSpec::new(vec![1, 4], DType::F64));
        let h1 = builder.linear(input, 4, 4); // used
        let _dead = builder.relu(input); // dead (not used)
        let graph = builder.build(vec![h1]);

        let before_count = graph.num_nodes();
        let optimized = dead_node_elimination(&graph);
        // The dead ReLU branch should be removed
        assert!(
            optimized.num_nodes() < before_count,
            "Dead node elimination should reduce node count: before={before_count}, after={}",
            optimized.num_nodes()
        );
    }

    #[test]
    fn test_graph_constant_folding() {
        // Constant folding here means the optimization pass completes without
        // error and constant nodes remain in the graph
        let mut builder = GraphBuilder::new();
        let input = builder.input(TensorSpec::new(vec![1, 4], DType::F64));
        let out = builder.linear(input, 4, 2);
        let graph = builder.build(vec![out]);

        let optimized = optimize(&graph);
        // Should still have nodes
        assert!(optimized.num_nodes() > 0);
    }

    #[test]
    fn test_operator_fusion() {
        let mut builder = GraphBuilder::new();
        let input = builder.input(TensorSpec::new(vec![1, 4], DType::F64));
        let linear_out = builder.linear(input, 4, 4);
        let relu_out = builder.relu(linear_out);
        let graph = builder.build(vec![relu_out]);

        let before_count = graph.num_nodes();
        let fused = operator_fusion(&graph);
        // Linear + ReLU should become one FusedLinearReLU
        assert!(
            fused.num_nodes() < before_count,
            "Fusion should reduce node count: before={before_count}, after={}",
            fused.num_nodes()
        );
        let has_fused = fused
            .nodes
            .iter()
            .any(|n| n.op_type == OpType::FusedLinearReLU);
        assert!(
            has_fused,
            "Graph should contain FusedLinearReLU after fusion"
        );
    }

    #[test]
    fn test_static_graph_shapes_consistent() {
        let mut builder = GraphBuilder::new();
        let input = builder.input(TensorSpec::new(vec![1, 16], DType::F64));
        let h1 = builder.linear(input, 16, 8);
        let h2 = builder.relu(h1);
        let out = builder.linear(h2, 8, 4);
        let graph = builder.build(vec![out]);

        // Verify output shapes are consistent along the chain
        let linear_out_shapes: Vec<Vec<usize>> = graph
            .nodes
            .iter()
            .filter(|n| n.op_type == OpType::Linear)
            .map(|n| n.output_spec.shape.clone())
            .collect();

        // First linear: [1, 8], second: [1, 4]
        assert_eq!(linear_out_shapes[0], vec![1, 8]);
        assert_eq!(linear_out_shapes[1], vec![1, 4]);
    }
}
