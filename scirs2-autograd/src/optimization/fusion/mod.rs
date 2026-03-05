//! Operation fusion for improved performance
//!
//! This module implements operation fusion to combine multiple operations into
//! single fused kernels, reducing memory traffic and improving performance.
//!
//! # Supported fusion patterns
//!
//! ## MatMul + Bias + Activation
//! The most common pattern in neural networks:
//! - `matmul(x, w) + bias` -> `fused_linear(x, w, bias)`
//! - `matmul(x, w) + bias + relu` -> `fused_linear_relu(x, w, bias)`
//! - `matmul(x, w) + bias + gelu` -> `fused_linear_gelu(x, w, bias)`
//!
//! ## Conv + BN + ReLU
//! Common in convolutional neural networks:
//! - `conv(x) + batch_norm` -> `fused_conv_bn(x)` (via parameter folding)
//! - `conv(x) + batch_norm + relu` -> `fused_conv_bn_relu(x)`
//!
//! ## Element-wise chain
//! Multiple consecutive element-wise ops fused into a single kernel:
//! - `x * scale + shift` -> `fused_affine(x, scale, shift)`
//! - Arbitrary unary chains: `relu(neg(x))` in one pass
//!
//! ## Reduction fusion
//! - `sum + divide` -> `mean`
//! - `square + mean` -> `variance`
//! - `softmax` = `exp + sum + divide` (numerically stable)

pub mod backward;
pub mod ops;
pub mod patterns;

// Re-export core pattern types for backward compatibility
pub use patterns::{FusionPattern, GraphNode, OpKind};

use crate::error::AutogradError;
use crate::Result;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Fusion candidate (backward-compatible with original API)
// ---------------------------------------------------------------------------

/// Fusion candidate describing a set of operations that can be fused.
#[derive(Debug, Clone)]
pub struct FusionCandidate {
    /// Operation IDs to fuse
    pub ops: Vec<usize>,
    /// Fusion pattern
    pub pattern: FusionPattern,
    /// Estimated speedup factor (>1.0 means faster)
    pub speedup: f64,
    /// Memory savings (bytes)
    pub memory_saved: usize,
}

impl FusionCandidate {
    /// Create a new fusion candidate
    pub fn new(ops: Vec<usize>, pattern: FusionPattern, speedup: f64, memory_saved: usize) -> Self {
        Self {
            ops,
            pattern,
            speedup,
            memory_saved,
        }
    }

    /// Check if fusion is beneficial (worth the overhead of pattern matching)
    pub fn is_beneficial(&self) -> bool {
        self.speedup > 1.1 || self.memory_saved > 1024
    }
}

// ---------------------------------------------------------------------------
// Fused node (result of fusing two or more nodes)
// ---------------------------------------------------------------------------

/// A fused node that replaces multiple original nodes.
#[derive(Debug, Clone)]
pub struct FusedNode {
    /// Unique identifier for this fused node
    pub id: usize,
    /// The fusion pattern that was applied
    pub pattern: FusionPattern,
    /// IDs of the original nodes that were fused
    pub original_ids: Vec<usize>,
    /// The operation kind of the fused result
    pub fused_op_name: String,
    /// Input node IDs (from outside the fused group)
    pub external_inputs: Vec<usize>,
    /// Output shape (if known)
    pub output_shape: Vec<usize>,
}

impl FusedNode {
    /// Create a fused node from a two-node fusion.
    pub fn from_two(
        fused_id: usize,
        pattern: FusionPattern,
        node_a: &GraphNode,
        node_b: &GraphNode,
    ) -> Self {
        // Collect external inputs: inputs of node_a + inputs of node_b
        // that are not node_a itself
        let mut external = node_a.inputs.clone();
        for &inp in &node_b.inputs {
            if inp != node_a.id {
                external.push(inp);
            }
        }

        let fused_op_name = format!("fused_{}_{}", node_a.op, node_b.op);

        Self {
            id: fused_id,
            pattern,
            original_ids: vec![node_a.id, node_b.id],
            fused_op_name,
            external_inputs: external,
            output_shape: node_b.output_shape.clone(),
        }
    }

    /// Create a fused node from a three-node fusion.
    pub fn from_three(
        fused_id: usize,
        pattern: FusionPattern,
        node_a: &GraphNode,
        node_b: &GraphNode,
        node_c: &GraphNode,
    ) -> Self {
        let mut external = node_a.inputs.clone();
        for &inp in &node_b.inputs {
            if inp != node_a.id {
                external.push(inp);
            }
        }
        for &inp in &node_c.inputs {
            if inp != node_a.id && inp != node_b.id {
                external.push(inp);
            }
        }

        let fused_op_name = format!("fused_{}_{}_{}", node_a.op, node_b.op, node_c.op);

        Self {
            id: fused_id,
            pattern,
            original_ids: vec![node_a.id, node_b.id, node_c.id],
            fused_op_name,
            external_inputs: external,
            output_shape: node_c.output_shape.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion pass -- a single registered pattern matcher
// ---------------------------------------------------------------------------

/// A registered fusion pass that checks and applies a specific pattern.
pub struct FusionPass {
    /// Human-readable name of this pass
    pub name: String,
    /// The pattern this pass detects
    pub target_pattern: FusionPattern,
    /// Whether this pass is enabled
    pub enabled: bool,
}

impl FusionPass {
    /// Create a new fusion pass.
    pub fn new(name: &str, target_pattern: FusionPattern) -> Self {
        Self {
            name: name.to_string(),
            target_pattern,
            enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion optimizer
// ---------------------------------------------------------------------------

/// Fusion optimizer that detects and applies fusion opportunities on a graph.
pub struct FusionOptimizer {
    /// Detected fusion candidates
    candidates: Vec<FusionCandidate>,
    /// Applied fusions (tracked by op-id sets)
    applied: HashSet<Vec<usize>>,
    /// Statistics: total fusions applied since creation
    total_fusions: usize,
    /// Registered fusion passes
    passes: Vec<FusionPass>,
    /// Produced fused nodes
    fused_nodes: Vec<FusedNode>,
    /// Next ID to assign to a fused node
    next_fused_id: usize,
}

impl FusionOptimizer {
    /// Create a new fusion optimizer with all built-in passes registered.
    pub fn new() -> Self {
        let passes = vec![
            FusionPass::new("matmul_bias", FusionPattern::MatMulBias),
            FusionPass::new("matmul_activation", FusionPattern::MatMulActivation),
            FusionPass::new(
                "matmul_bias_activation",
                FusionPattern::MatMulBiasActivation,
            ),
            FusionPass::new("conv_bn", FusionPattern::ConvBN),
            FusionPass::new("conv_bn_activation", FusionPattern::ConvBNActivation),
            FusionPass::new("elementwise", FusionPattern::ElementWise),
            FusionPass::new("affine", FusionPattern::Affine),
            FusionPass::new("sum_div_to_mean", FusionPattern::SumDivToMean),
            FusionPass::new(
                "square_mean_to_variance",
                FusionPattern::SquareMeanToVariance,
            ),
            FusionPass::new("softmax", FusionPattern::Softmax),
        ];

        Self {
            candidates: Vec::new(),
            applied: HashSet::new(),
            total_fusions: 0,
            passes,
            fused_nodes: Vec::new(),
            next_fused_id: 10000, // Start fused IDs high to avoid conflicts
        }
    }

    /// Register a custom fusion pass.
    pub fn register_pass(&mut self, pass: FusionPass) {
        self.passes.push(pass);
    }

    /// Enable or disable a pass by name.
    pub fn set_pass_enabled(&mut self, name: &str, enabled: bool) {
        for pass in &mut self.passes {
            if pass.name == name {
                pass.enabled = enabled;
            }
        }
    }

    /// Detect fusion opportunities in a set of graph nodes.
    ///
    /// Scans all pairs and triples of nodes looking for enabled fusion
    /// patterns.  Detected candidates are stored internally.
    pub fn detect_fusions_in_graph(&mut self, nodes: &[GraphNode]) -> Result<()> {
        self.candidates.clear();

        let enabled_patterns: HashSet<FusionPattern> = self
            .passes
            .iter()
            .filter(|p| p.enabled)
            .map(|p| p.target_pattern.clone())
            .collect();

        // Build index for quick lookup
        let node_map: std::collections::HashMap<usize, &GraphNode> =
            nodes.iter().map(|n| (n.id, n)).collect();

        // Scan two-node patterns
        for node_a in nodes {
            for &succ_id in &node_a.inputs {
                // We look at node_a's *successors* -- but our graph
                // is stored with input edges.  Instead iterate all
                // nodes and check if they consume node_a.
                let _ = succ_id; // inputs are predecessors, not successors
            }
        }

        // Scan: for each pair (a, b) where b consumes a
        for node_b in nodes {
            for &pred_id in &node_b.inputs {
                if let Some(node_a) = node_map.get(&pred_id) {
                    if let Some(pattern) = patterns::detect_two_node_pattern(node_a, node_b) {
                        if enabled_patterns.contains(&pattern) {
                            let speedup = estimate_speedup_two(&pattern, node_a, node_b);
                            let mem = estimate_memory_saved_two(node_a, node_b);
                            self.candidates.push(FusionCandidate::new(
                                vec![node_a.id, node_b.id],
                                pattern,
                                speedup,
                                mem,
                            ));
                        }
                    }
                }
            }
        }

        // Scan three-node patterns: for each triple (a, b, c) where
        // b consumes a and c consumes b.
        for node_c in nodes {
            for &pred_c in &node_c.inputs {
                if let Some(node_b) = node_map.get(&pred_c) {
                    for &pred_b in &node_b.inputs {
                        if let Some(node_a) = node_map.get(&pred_b) {
                            if let Some(pattern) =
                                patterns::detect_three_node_pattern(node_a, node_b, node_c)
                            {
                                if enabled_patterns.contains(&pattern) {
                                    let speedup =
                                        estimate_speedup_three(&pattern, node_a, node_b, node_c);
                                    let mem = estimate_memory_saved_three(node_a, node_b, node_c);
                                    self.candidates.push(FusionCandidate::new(
                                        vec![node_a.id, node_b.id, node_c.id],
                                        pattern,
                                        speedup,
                                        mem,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Backward-compatible: detect placeholder fusions (same as original API).
    pub fn detect_fusions(&mut self) -> Result<()> {
        self.candidates.push(FusionCandidate::new(
            vec![1, 2],
            FusionPattern::MatMulBias,
            1.5,
            4096,
        ));
        Ok(())
    }

    /// Apply beneficial fusions and produce `FusedNode` entries.
    ///
    /// Returns the number of fusions applied in this call.
    pub fn apply_fusions(&mut self) -> Result<usize> {
        let mut num_applied = 0;
        // Collect fused-id sets already used (to prevent overlapping fusions)
        let mut used_node_ids: HashSet<usize> = HashSet::new();

        // Sort candidates by estimated speedup (best first)
        let mut sorted_indices: Vec<usize> = (0..self.candidates.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            self.candidates[b]
                .speedup
                .partial_cmp(&self.candidates[a].speedup)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for idx in sorted_indices {
            let candidate = &self.candidates[idx];
            if !candidate.is_beneficial() {
                continue;
            }
            if self.applied.contains(&candidate.ops) {
                continue;
            }
            // Check no overlap with already-fused nodes
            if candidate.ops.iter().any(|id| used_node_ids.contains(id)) {
                continue;
            }

            self.applied.insert(candidate.ops.clone());
            for &id in &candidate.ops {
                used_node_ids.insert(id);
            }
            self.total_fusions += 1;
            num_applied += 1;
        }

        Ok(num_applied)
    }

    /// Apply fusions and produce actual `FusedNode` objects from graph nodes.
    pub fn apply_fusions_with_nodes(&mut self, nodes: &[GraphNode]) -> Result<Vec<FusedNode>> {
        let mut result = Vec::new();
        let mut used_node_ids: HashSet<usize> = HashSet::new();

        let node_map: std::collections::HashMap<usize, &GraphNode> =
            nodes.iter().map(|n| (n.id, n)).collect();

        let mut sorted_indices: Vec<usize> = (0..self.candidates.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            self.candidates[b]
                .speedup
                .partial_cmp(&self.candidates[a].speedup)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for idx in sorted_indices {
            let candidate = &self.candidates[idx];
            if !candidate.is_beneficial() {
                continue;
            }
            if self.applied.contains(&candidate.ops) {
                continue;
            }
            if candidate.ops.iter().any(|id| used_node_ids.contains(id)) {
                continue;
            }

            // Resolve nodes
            let resolved: Vec<&GraphNode> = candidate
                .ops
                .iter()
                .filter_map(|id| node_map.get(id).copied())
                .collect();

            if resolved.len() != candidate.ops.len() {
                continue; // Some nodes not found, skip
            }

            let fused_id = self.next_fused_id;
            self.next_fused_id += 1;

            let fused_node = if resolved.len() == 2 {
                FusedNode::from_two(
                    fused_id,
                    candidate.pattern.clone(),
                    resolved[0],
                    resolved[1],
                )
            } else if resolved.len() == 3 {
                FusedNode::from_three(
                    fused_id,
                    candidate.pattern.clone(),
                    resolved[0],
                    resolved[1],
                    resolved[2],
                )
            } else {
                continue;
            };

            self.applied.insert(candidate.ops.clone());
            for &id in &candidate.ops {
                used_node_ids.insert(id);
            }
            self.total_fusions += 1;
            result.push(fused_node);
        }

        self.fused_nodes.extend(result.clone());
        Ok(result)
    }

    /// Get fusion candidates
    pub fn candidates(&self) -> &[FusionCandidate] {
        &self.candidates
    }

    /// Get produced fused nodes
    pub fn fused_nodes(&self) -> &[FusedNode] {
        &self.fused_nodes
    }

    /// Get number of applied fusions (lifetime total)
    pub fn total_fusions(&self) -> usize {
        self.total_fusions
    }

    /// Get registered passes
    pub fn passes(&self) -> &[FusionPass] {
        &self.passes
    }

    /// Clear optimizer state (candidates, applied set, fused nodes)
    pub fn clear(&mut self) {
        self.candidates.clear();
        self.applied.clear();
        self.fused_nodes.clear();
    }
}

impl Default for FusionOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Fusion result (backward-compatible)
// ---------------------------------------------------------------------------

/// Fusion pass result summary.
#[derive(Debug, Clone)]
pub struct FusionResult {
    /// Number of fusions applied
    pub fusions: usize,
    /// Total speedup factor
    pub speedup: f64,
    /// Memory saved (bytes)
    pub memory_saved: usize,
}

impl FusionResult {
    /// Create a new fusion result
    pub fn new(fusions: usize, speedup: f64, memory_saved: usize) -> Self {
        Self {
            fusions,
            speedup,
            memory_saved,
        }
    }

    /// Check if any fusions were applied
    pub fn has_changes(&self) -> bool {
        self.fusions > 0
    }
}

// ---------------------------------------------------------------------------
// Backward-compatible free functions
// ---------------------------------------------------------------------------

/// Element-wise operation fusion (backward-compatible with original API)
pub fn fuse_elementwise_ops(ops_names: &[String]) -> Result<String> {
    if ops_names.is_empty() {
        return Err(AutogradError::invalid_argument(
            "No operations to fuse".to_string(),
        ));
    }
    let fused = format!("fused_{}", ops_names.join("_"));
    Ok(fused)
}

/// Matrix multiplication fusion naming (backward-compatible)
pub fn fuse_matmul_bias(has_bias: bool, activation: Option<&str>) -> String {
    match (has_bias, activation) {
        (true, Some(act)) => format!("fused_matmul_bias_{}", act),
        (true, None) => "fused_matmul_bias".to_string(),
        (false, Some(act)) => format!("fused_matmul_{}", act),
        (false, None) => "matmul".to_string(),
    }
}

/// Convolution fusion naming (backward-compatible)
pub fn fuse_conv_bn_activation(has_bn: bool, activation: Option<&str>) -> String {
    match (has_bn, activation) {
        (true, Some(act)) => format!("fused_conv_bn_{}", act),
        (true, None) => "fused_conv_bn".to_string(),
        (false, Some(act)) => format!("fused_conv_{}", act),
        (false, None) => "conv".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Speedup / memory estimation helpers
// ---------------------------------------------------------------------------

/// Heuristic speedup estimate for a two-node fusion.
fn estimate_speedup_two(pattern: &FusionPattern, _node_a: &GraphNode, _node_b: &GraphNode) -> f64 {
    match pattern {
        FusionPattern::MatMulBias => 1.3,
        FusionPattern::MatMulActivation => 1.4,
        FusionPattern::ConvBN => 1.5,
        FusionPattern::SumDivToMean => 1.8,
        FusionPattern::SquareMeanToVariance => 1.7,
        FusionPattern::Affine => 1.6,
        FusionPattern::ElementWise => 1.2,
        _ => 1.1,
    }
}

/// Heuristic speedup estimate for a three-node fusion.
fn estimate_speedup_three(
    pattern: &FusionPattern,
    _node_a: &GraphNode,
    _node_b: &GraphNode,
    _node_c: &GraphNode,
) -> f64 {
    match pattern {
        FusionPattern::MatMulBiasActivation => 1.6,
        FusionPattern::ConvBNActivation => 1.7,
        FusionPattern::Softmax => 1.9,
        _ => 1.2,
    }
}

/// Heuristic memory saved estimate for a two-node fusion (bytes).
fn estimate_memory_saved_two(node_a: &GraphNode, _node_b: &GraphNode) -> usize {
    // Eliminating one intermediate tensor
    node_a
        .output_numel()
        .map(|n| n * std::mem::size_of::<f64>())
        .unwrap_or(4096)
}

/// Heuristic memory saved estimate for a three-node fusion (bytes).
fn estimate_memory_saved_three(
    node_a: &GraphNode,
    node_b: &GraphNode,
    _node_c: &GraphNode,
) -> usize {
    // Eliminating two intermediate tensors
    let a_mem = node_a
        .output_numel()
        .map(|n| n * std::mem::size_of::<f64>())
        .unwrap_or(4096);
    let b_mem = node_b
        .output_numel()
        .map(|n| n * std::mem::size_of::<f64>())
        .unwrap_or(4096);
    a_mem + b_mem
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Backward-compatible tests (from original fusion.rs) ----------------

    #[test]
    fn test_fusion_pattern() {
        assert_eq!(FusionPattern::ElementWise, FusionPattern::ElementWise);
        assert_ne!(FusionPattern::ElementWise, FusionPattern::MatMulBias);
    }

    #[test]
    fn test_fusion_candidate() {
        let candidate =
            FusionCandidate::new(vec![1, 2, 3], FusionPattern::MatMulActivation, 1.8, 2048);
        assert!(candidate.is_beneficial());
        assert_eq!(candidate.ops.len(), 3);
    }

    #[test]
    fn test_fusion_candidate_not_beneficial() {
        let candidate = FusionCandidate::new(vec![1, 2], FusionPattern::ElementWise, 1.05, 512);
        assert!(!candidate.is_beneficial());
    }

    #[test]
    fn test_fusion_optimizer() {
        let mut optimizer = FusionOptimizer::new();

        optimizer.detect_fusions().expect("Should detect fusions");
        assert!(!optimizer.candidates().is_empty());

        let applied = optimizer.apply_fusions().expect("Should apply fusions");
        assert!(applied > 0);
        assert_eq!(optimizer.total_fusions(), applied);
    }

    #[test]
    fn test_fuse_elementwise() {
        let ops = vec!["add".to_string(), "mul".to_string(), "relu".to_string()];
        let fused = fuse_elementwise_ops(&ops).expect("Should fuse");
        assert_eq!(fused, "fused_add_mul_relu");
    }

    #[test]
    fn test_fuse_elementwise_empty() {
        let ops: Vec<String> = vec![];
        assert!(fuse_elementwise_ops(&ops).is_err());
    }

    #[test]
    fn test_fuse_matmul_bias() {
        assert_eq!(
            fuse_matmul_bias(true, Some("relu")),
            "fused_matmul_bias_relu"
        );
        assert_eq!(fuse_matmul_bias(true, None), "fused_matmul_bias");
        assert_eq!(fuse_matmul_bias(false, Some("gelu")), "fused_matmul_gelu");
        assert_eq!(fuse_matmul_bias(false, None), "matmul");
    }

    #[test]
    fn test_fuse_conv() {
        assert_eq!(
            fuse_conv_bn_activation(true, Some("relu")),
            "fused_conv_bn_relu"
        );
        assert_eq!(
            fuse_conv_bn_activation(false, Some("swish")),
            "fused_conv_swish"
        );
        assert_eq!(fuse_conv_bn_activation(true, None), "fused_conv_bn");
        assert_eq!(fuse_conv_bn_activation(false, None), "conv");
    }

    #[test]
    fn test_fusion_result() {
        let result = FusionResult::new(3, 1.5, 8192);
        assert!(result.has_changes());
        assert_eq!(result.fusions, 3);
        assert_eq!(result.speedup, 1.5);
    }

    #[test]
    fn test_fusion_result_no_changes() {
        let result = FusionResult::new(0, 1.0, 0);
        assert!(!result.has_changes());
    }

    // -- New tests: graph-level fusion detection ----------------------------

    fn make_node(id: usize, op: OpKind, inputs: Vec<usize>, consumers: usize) -> GraphNode {
        let mut node = GraphNode::new(id, op, inputs, vec![4, 8]);
        node.num_consumers = consumers;
        node
    }

    #[test]
    fn test_detect_matmul_bias_in_graph() {
        let nodes = vec![
            make_node(0, OpKind::Input, vec![], 1),
            make_node(1, OpKind::Input, vec![], 1),
            make_node(2, OpKind::MatMul, vec![0, 1], 1),
            make_node(3, OpKind::BiasAdd, vec![2], 1),
        ];

        let mut optimizer = FusionOptimizer::new();
        optimizer.detect_fusions_in_graph(&nodes).expect("detect");
        assert!(!optimizer.candidates().is_empty());

        let matmul_bias_candidates: Vec<_> = optimizer
            .candidates()
            .iter()
            .filter(|c| c.pattern == FusionPattern::MatMulBias)
            .collect();
        assert!(!matmul_bias_candidates.is_empty());
    }

    #[test]
    fn test_detect_matmul_bias_relu_in_graph() {
        let nodes = vec![
            make_node(0, OpKind::Input, vec![], 1),
            make_node(1, OpKind::Input, vec![], 1),
            make_node(2, OpKind::MatMul, vec![0, 1], 1),
            make_node(3, OpKind::Add, vec![2], 1),
            make_node(4, OpKind::Relu, vec![3], 1),
        ];

        let mut optimizer = FusionOptimizer::new();
        optimizer.detect_fusions_in_graph(&nodes).expect("detect");

        let three_node: Vec<_> = optimizer
            .candidates()
            .iter()
            .filter(|c| c.pattern == FusionPattern::MatMulBiasActivation)
            .collect();
        assert!(!three_node.is_empty());
    }

    #[test]
    fn test_detect_conv_bn_relu_in_graph() {
        let nodes = vec![
            make_node(0, OpKind::Input, vec![], 1),
            make_node(1, OpKind::Conv2d, vec![0], 1),
            make_node(2, OpKind::BatchNorm, vec![1], 1),
            make_node(3, OpKind::Relu, vec![2], 1),
        ];

        let mut optimizer = FusionOptimizer::new();
        optimizer.detect_fusions_in_graph(&nodes).expect("detect");

        let conv_bn_relu: Vec<_> = optimizer
            .candidates()
            .iter()
            .filter(|c| c.pattern == FusionPattern::ConvBNActivation)
            .collect();
        assert!(!conv_bn_relu.is_empty());
    }

    #[test]
    fn test_detect_sum_div_to_mean() {
        let nodes = vec![
            make_node(0, OpKind::Input, vec![], 1),
            make_node(1, OpKind::Sum, vec![0], 1),
            make_node(2, OpKind::Div, vec![1], 1),
        ];

        let mut optimizer = FusionOptimizer::new();
        optimizer.detect_fusions_in_graph(&nodes).expect("detect");

        let mean_candidates: Vec<_> = optimizer
            .candidates()
            .iter()
            .filter(|c| c.pattern == FusionPattern::SumDivToMean)
            .collect();
        assert!(!mean_candidates.is_empty());
    }

    #[test]
    fn test_detect_square_mean_to_variance() {
        let nodes = vec![
            make_node(0, OpKind::Input, vec![], 1),
            make_node(1, OpKind::Square, vec![0], 1),
            make_node(2, OpKind::Mean, vec![1], 1),
        ];

        let mut optimizer = FusionOptimizer::new();
        optimizer.detect_fusions_in_graph(&nodes).expect("detect");

        let var_candidates: Vec<_> = optimizer
            .candidates()
            .iter()
            .filter(|c| c.pattern == FusionPattern::SquareMeanToVariance)
            .collect();
        assert!(!var_candidates.is_empty());
    }

    #[test]
    fn test_apply_fusions_with_nodes() {
        let nodes = vec![
            make_node(0, OpKind::Input, vec![], 1),
            make_node(1, OpKind::MatMul, vec![0], 1),
            make_node(2, OpKind::BiasAdd, vec![1], 1),
        ];

        let mut optimizer = FusionOptimizer::new();
        optimizer.detect_fusions_in_graph(&nodes).expect("detect");

        let fused = optimizer.apply_fusions_with_nodes(&nodes).expect("apply");
        assert!(!fused.is_empty());
        assert_eq!(fused[0].original_ids, vec![1, 2]);
        assert_eq!(fused[0].pattern, FusionPattern::MatMulBias);
    }

    #[test]
    fn test_no_overlapping_fusions() {
        // Two separate fusible pairs sharing no nodes
        let nodes = vec![
            make_node(0, OpKind::Input, vec![], 2),
            make_node(1, OpKind::MatMul, vec![0], 1),
            make_node(2, OpKind::BiasAdd, vec![1], 1),
            make_node(3, OpKind::MatMul, vec![0], 1),
            make_node(4, OpKind::BiasAdd, vec![3], 1),
        ];

        let mut optimizer = FusionOptimizer::new();
        optimizer.detect_fusions_in_graph(&nodes).expect("detect");

        let fused = optimizer.apply_fusions_with_nodes(&nodes).expect("apply");
        // Both pairs should be fusible (no overlap)
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn test_overlapping_fusions_rejected() {
        // A chain: matmul -> add -> relu
        // Could be fused as (matmul, add) or (add, relu)
        // Only one should be picked (the better one)
        let nodes = vec![
            make_node(0, OpKind::Input, vec![], 1),
            make_node(1, OpKind::MatMul, vec![0], 1),
            make_node(2, OpKind::Add, vec![1], 1),
            make_node(3, OpKind::Relu, vec![2], 1),
        ];

        let mut optimizer = FusionOptimizer::new();
        optimizer.detect_fusions_in_graph(&nodes).expect("detect");

        let fused = optimizer.apply_fusions_with_nodes(&nodes).expect("apply");

        // Check no node appears in two fused groups
        let mut all_ids: Vec<usize> = Vec::new();
        for f in &fused {
            all_ids.extend(&f.original_ids);
        }
        let unique: HashSet<usize> = all_ids.iter().copied().collect();
        assert_eq!(all_ids.len(), unique.len(), "No overlapping fusions");
    }

    #[test]
    fn test_fusion_pass_enable_disable() {
        let mut optimizer = FusionOptimizer::new();
        optimizer.set_pass_enabled("matmul_bias", false);

        let nodes = vec![
            make_node(0, OpKind::Input, vec![], 1),
            make_node(1, OpKind::MatMul, vec![0], 1),
            make_node(2, OpKind::BiasAdd, vec![1], 1),
        ];

        optimizer.detect_fusions_in_graph(&nodes).expect("detect");

        let matmul_bias_candidates: Vec<_> = optimizer
            .candidates()
            .iter()
            .filter(|c| c.pattern == FusionPattern::MatMulBias)
            .collect();
        // Should not detect matmul_bias because pass is disabled
        assert!(matmul_bias_candidates.is_empty());
    }

    #[test]
    fn test_clear_resets_state() {
        let mut optimizer = FusionOptimizer::new();
        optimizer.detect_fusions().expect("detect");
        optimizer.apply_fusions().expect("apply");
        assert!(!optimizer.candidates().is_empty());

        optimizer.clear();
        assert!(optimizer.candidates().is_empty());
        assert!(optimizer.fused_nodes().is_empty());
    }

    #[test]
    fn test_fused_node_from_two() {
        let a = make_node(10, OpKind::MatMul, vec![5, 6], 1);
        let b = make_node(11, OpKind::BiasAdd, vec![10, 7], 1);

        let fused = FusedNode::from_two(100, FusionPattern::MatMulBias, &a, &b);
        assert_eq!(fused.id, 100);
        assert_eq!(fused.original_ids, vec![10, 11]);
        // External inputs: [5, 6] from a, plus [7] from b (10 is excluded)
        assert!(fused.external_inputs.contains(&5));
        assert!(fused.external_inputs.contains(&6));
        assert!(fused.external_inputs.contains(&7));
        assert!(!fused.external_inputs.contains(&10));
    }

    #[test]
    fn test_fused_node_from_three() {
        let a = make_node(10, OpKind::MatMul, vec![5], 1);
        let b = make_node(11, OpKind::Add, vec![10, 6], 1);
        let c = make_node(12, OpKind::Relu, vec![11], 1);

        let fused = FusedNode::from_three(200, FusionPattern::MatMulBiasActivation, &a, &b, &c);
        assert_eq!(fused.id, 200);
        assert_eq!(fused.original_ids, vec![10, 11, 12]);
        assert!(fused.external_inputs.contains(&5));
        assert!(fused.external_inputs.contains(&6));
    }

    #[test]
    fn test_registered_passes() {
        let optimizer = FusionOptimizer::new();
        assert!(optimizer.passes().len() >= 10);
        // Check some known pass names
        let names: Vec<&str> = optimizer.passes().iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"matmul_bias"));
        assert!(names.contains(&"conv_bn_activation"));
        assert!(names.contains(&"softmax"));
        assert!(names.contains(&"affine"));
    }

    #[test]
    fn test_empty_graph_detection() {
        let mut optimizer = FusionOptimizer::new();
        optimizer.detect_fusions_in_graph(&[]).expect("empty graph");
        assert!(optimizer.candidates().is_empty());
    }

    #[test]
    fn test_single_node_no_fusion() {
        let nodes = vec![make_node(0, OpKind::Input, vec![], 0)];
        let mut optimizer = FusionOptimizer::new();
        optimizer.detect_fusions_in_graph(&nodes).expect("detect");
        assert!(optimizer.candidates().is_empty());
    }
}
