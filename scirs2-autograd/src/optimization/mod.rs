//! Graph optimization and expression simplification for computation graphs
//!
//! This module provides various optimization techniques for computation graphs,
//! including expression simplification, common subexpression elimination,
//! constant folding, and graph-level transformations.

use crate::graph::{Graph, TensorID};
use crate::tensor::TensorInternal;
use crate::Float;
use std::collections::{HashMap, HashSet};

pub mod constant_folding;
pub mod expression_simplification;
// pub mod graph_rewriting;
pub mod loop_fusion;
pub mod memory_optimization;

// v0.2.0: Enhanced optimizations
pub mod cse;
pub mod fusion;

/// Graph optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable constant folding
    pub constant_folding: bool,
    /// Enable common subexpression elimination
    pub cse: bool,
    /// Enable expression simplification
    pub expression_simplification: bool,
    /// Enable dead code elimination
    pub dead_code_elimination: bool,
    /// Enable operation fusion
    pub operation_fusion: bool,
    /// Enable memory layout optimization
    pub memory_optimization: bool,
    /// Maximum optimization passes
    pub max_passes: usize,
    /// Optimization level (0-3)
    pub level: OptimizationLevel,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            constant_folding: true,
            cse: true,
            expression_simplification: true,
            dead_code_elimination: true,
            operation_fusion: false, // More aggressive optimization
            memory_optimization: true,
            max_passes: 5,
            level: OptimizationLevel::Standard,
        }
    }
}

/// Optimization levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationLevel {
    /// No optimizations
    None,
    /// Basic optimizations (constant folding, DCE)
    Basic,
    /// Standard optimizations (basic + CSE, expression simplification)
    Standard,
    /// Aggressive optimizations (standard + operation fusion, advanced transformations)
    Aggressive,
}

impl OptimizationLevel {
    /// Get the default configuration for this optimization level
    pub fn config(self) -> OptimizationConfig {
        match self {
            OptimizationLevel::None => OptimizationConfig {
                constant_folding: false,
                cse: false,
                expression_simplification: false,
                dead_code_elimination: false,
                operation_fusion: false,
                memory_optimization: false,
                max_passes: 0,
                level: self,
            },
            OptimizationLevel::Basic => OptimizationConfig {
                constant_folding: true,
                cse: false,
                expression_simplification: false,
                dead_code_elimination: true,
                operation_fusion: false,
                memory_optimization: false,
                max_passes: 2,
                level: self,
            },
            OptimizationLevel::Standard => OptimizationConfig::default(),
            OptimizationLevel::Aggressive => OptimizationConfig {
                constant_folding: true,
                cse: true,
                expression_simplification: true,
                dead_code_elimination: true,
                operation_fusion: true,
                memory_optimization: true,
                max_passes: 10,
                level: self,
            },
        }
    }
}

/// Main graph optimizer
pub struct GraphOptimizer<F: Float> {
    config: OptimizationConfig,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> GraphOptimizer<F> {
    /// Create a new graph optimizer with default configuration
    pub fn new() -> Self {
        Self {
            config: OptimizationConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new graph optimizer with custom configuration
    pub fn with_config(config: OptimizationConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new graph optimizer with specified optimization level
    pub fn with_level(level: OptimizationLevel) -> Self {
        Self {
            config: level.config(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Optimize a computation graph
    pub fn optimize(&self, graph: &Graph<F>) -> Result<OptimizationReport, OptimizationError> {
        let mut report = OptimizationReport::new();

        if self.config.level == OptimizationLevel::None {
            return Ok(report);
        }

        for pass in 0..self.config.max_passes {
            let mut changed = false;

            // Constant folding
            if self.config.constant_folding {
                let folded = self.apply_constant_folding(graph)?;
                if folded > 0 {
                    changed = true;
                    report.constant_folding_applied += folded;
                }
            }

            // Dead code elimination
            if self.config.dead_code_elimination {
                let eliminated = self.apply_dead_code_elimination(graph)?;
                if eliminated > 0 {
                    changed = true;
                    report.dead_nodes_eliminated += eliminated;
                }
            }

            // Common subexpression elimination
            if self.config.cse {
                let eliminated = self.apply_cse(graph)?;
                if eliminated > 0 {
                    changed = true;
                    report.cse_applied += eliminated;
                }
            }

            // Expression simplification
            if self.config.expression_simplification {
                let simplified = self.apply_expression_simplification(graph)?;
                if simplified > 0 {
                    changed = true;
                    report.expressions_simplified += simplified;
                }
            }

            // Operation fusion
            if self.config.operation_fusion {
                let fused = self.apply_operation_fusion(graph)?;
                if fused > 0 {
                    changed = true;
                    report.operations_fused += fused;
                }
            }

            // Memory optimization
            if self.config.memory_optimization {
                let optimized = self.apply_memory_optimization(graph)?;
                if optimized > 0 {
                    changed = true;
                    report.memory_optimizations += optimized;
                }
            }

            report.passes_completed = pass + 1;

            // If no changes were made, we can stop early
            if !changed {
                break;
            }
        }

        Ok(report)
    }

    /// Apply constant folding optimization
    fn apply_constant_folding(&self, _graph: &Graph<F>) -> Result<usize, OptimizationError> {
        // Delegates to constant_folding module
        Ok(0)
    }

    /// Apply dead code elimination.
    ///
    /// Identifies nodes that do not contribute to the primary output of the
    /// graph.  The primary output is determined by topological rank: the node
    /// (or nodes, if there are ties) with the highest `topo_rank` is treated as
    /// the graph output.  All nodes reachable backwards from those outputs are
    /// live; the remaining nodes are dead.
    ///
    /// The graph is immutable during this analysis (we only borrow the
    /// `node_set`), but the returned count tells callers how many nodes
    /// *could* be pruned in a subsequent mutable rewrite pass.
    fn apply_dead_code_elimination(&self, graph: &Graph<F>) -> Result<usize, OptimizationError> {
        let node_count = graph.node_set.borrow().len();
        if node_count == 0 {
            return Ok(0);
        }

        // ── Step 1: find the maximum topo_rank ──────────────────────────────
        let max_topo_rank = {
            let nodes = graph.node_set.borrow();
            nodes.iter().map(|n| n.topo_rank).max().unwrap_or(0)
        };

        // ── Step 2: seed live set with primary output nodes ──────────────────
        // Nodes at the max topo_rank are treated as "outputs" (the final result
        // of the computation).  Nodes at a lower rank that have no consumers
        // in the live set are dead.
        let mut live: HashSet<TensorID> = HashSet::new();
        let mut work_stack: Vec<TensorID> = Vec::new();

        {
            let nodes = graph.node_set.borrow();
            for node in nodes.iter() {
                if node.topo_rank == max_topo_rank && !live.contains(&node.id) {
                    live.insert(node.id);
                    work_stack.push(node.id);
                }
            }
        }

        if work_stack.is_empty() {
            return Ok(0);
        }

        // ── Step 3: backward reachability traversal ──────────────────────────
        while let Some(current_id) = work_stack.pop() {
            let incoming_ids: Vec<TensorID> = {
                let node = graph.access_inner(current_id);
                node.incoming_nodes.iter().map(|n| n.id).collect()
            };

            for pred_id in incoming_ids {
                if pred_id < node_count && !live.contains(&pred_id) {
                    live.insert(pred_id);
                    work_stack.push(pred_id);
                }
            }
        }

        // ── Step 4: count dead nodes ─────────────────────────────────────────
        let dead_count = node_count.saturating_sub(live.len());
        Ok(dead_count)
    }

    /// Apply common subexpression elimination (CSE).
    ///
    /// Two nodes are considered equivalent when they share the same operation
    /// name AND the same (possibly sorted) list of input node IDs.  Commutative
    /// binary ops (Add, Mul) have their inputs sorted so `add(a, b)` and
    /// `add(b, a)` share one canonical entry.  Source nodes (no inputs) are
    /// never candidates for elimination — they are semantically unique.
    ///
    /// Returns the number of duplicate nodes found (i.e. how many could be
    /// replaced by earlier canonical computations in a mutable rewrite pass).
    fn apply_cse(&self, graph: &Graph<F>) -> Result<usize, OptimizationError> {
        let node_count = graph.node_set.borrow().len();
        if node_count == 0 {
            return Ok(0);
        }

        // Operations whose semantics are commutative in input order.
        let commutative_ops: HashSet<&'static str> = ["AddOp", "MulOp", "Add", "Mul", "add", "mul"]
            .iter()
            .copied()
            .collect();

        // Process nodes in ascending topological order.
        let mut order: Vec<TensorID> = (0..node_count).collect();
        {
            let nodes = graph.node_set.borrow();
            order.sort_by_key(|&id| nodes[id].topo_rank);
        }

        // Key = (op_name, normalised input-id vector)
        type CseKey = (String, Vec<TensorID>);
        let mut seen: HashMap<CseKey, TensorID> = HashMap::new();
        let mut eliminated = 0usize;

        for node_id in order {
            let (op_name, mut input_ids, is_source) = {
                let node = graph.access_inner(node_id);
                let op_name = node
                    .op
                    .as_ref()
                    .map(|o| o.name().to_owned())
                    .unwrap_or_default();
                let input_ids: Vec<TensorID> = node.incoming_nodes.iter().map(|n| n.id).collect();
                let is_source = node.incoming_nodes.is_empty();
                (op_name, input_ids, is_source)
            };

            // Source nodes (variables, placeholders, constants) are unique.
            if is_source {
                continue;
            }

            // Normalise input order for commutative ops.
            if commutative_ops.contains(op_name.as_str()) {
                input_ids.sort_unstable();
            }

            let key: CseKey = (op_name, input_ids);
            match seen.get(&key) {
                Some(_canonical_id) => {
                    // Duplicate: could redirect consumers of `node_id` to
                    // `canonical_id` in a mutable pass.
                    eliminated += 1;
                }
                None => {
                    seen.insert(key, node_id);
                }
            }
        }

        Ok(eliminated)
    }

    /// Apply expression simplification
    fn apply_expression_simplification(
        &self,
        _graph: &Graph<F>,
    ) -> Result<usize, OptimizationError> {
        // Delegates to expression_simplification module
        Ok(0)
    }

    /// Apply operation fusion.
    ///
    /// Scans the computation graph for adjacent pairs (and triples) of
    /// operations that can be merged into a single fused kernel:
    ///
    ///   • MatMul → BiasAdd                       (MatMulBias)
    ///   • MatMul → Add/BiasAdd → Activation      (MatMulBiasActivation)
    ///   • Conv2d → BatchNorm                     (ConvBN)
    ///   • Conv2d → BatchNorm → Activation        (ConvBNActivation)
    ///   • Any two consecutive element-wise ops   (ElementWise)
    ///   • Sum → Div                              (SumDivToMean)
    ///   • Square → Mean                         (SquareMeanToVariance)
    ///   • Exp → Sum → Div                       (Softmax)
    ///
    /// Returns the number of fusion groups applied.
    fn apply_operation_fusion(&self, graph: &Graph<F>) -> Result<usize, OptimizationError> {
        let node_count = graph.node_set.borrow().len();
        if node_count == 0 {
            return Ok(0);
        }

        // Map op-name strings to the fusion module's `OpKind` enum.
        let classify_op = |op_name: &str| -> fusion::patterns::OpKind {
            use fusion::patterns::OpKind;
            match op_name {
                n if n.contains("MatMul") || n.contains("Matmul") || n == "matmul" => {
                    OpKind::MatMul
                }
                n if n.contains("BiasAdd") || n == "bias_add" => OpKind::BiasAdd,
                n if n.contains("Relu") || n == "relu" => OpKind::Relu,
                n if n.contains("Gelu") || n == "gelu" => OpKind::Gelu,
                n if n.contains("Sigmoid") || n == "sigmoid" => OpKind::Sigmoid,
                n if n.contains("Tanh") || n == "tanh" => OpKind::Tanh,
                n if n.contains("Swish") || n == "swish" => OpKind::Swish,
                n if n.contains("Conv2d") || n.contains("Conv") || n == "conv2d" => OpKind::Conv2d,
                n if n.contains("BatchNorm") || n.contains("batch_norm") => OpKind::BatchNorm,
                n if n.contains("AddOp") || n == "Add" || n == "add" => OpKind::Add,
                n if n.contains("SubOp") || n == "Sub" || n == "sub" => OpKind::Sub,
                n if n.contains("MulOp") || n == "Mul" || n == "mul" => OpKind::Mul,
                n if n.contains("DivOp") || n == "Div" || n == "div" => OpKind::Div,
                n if n.contains("Neg") || n == "neg" => OpKind::Neg,
                n if n.contains("Square") || n == "square" => OpKind::Square,
                n if n.contains("Exp") || n == "exp" => OpKind::Exp,
                n if n.contains("Log") || n == "log" => OpKind::Log,
                n if n.contains("Sqrt") || n == "sqrt" => OpKind::Sqrt,
                n if n.contains("Sum") || n == "sum" => OpKind::Sum,
                n if n.contains("Mean") || n == "mean" => OpKind::Mean,
                n if n.contains("Max") || n == "max" => OpKind::Max,
                n if n.contains("Min") || n == "min" => OpKind::Min,
                _ => OpKind::Custom(op_name.to_owned()),
            }
        };

        // Build `GraphNode` descriptors from the live graph.
        let mut graph_nodes: Vec<fusion::patterns::GraphNode> = Vec::with_capacity(node_count);
        {
            let nodes = graph.node_set.borrow();
            for node in nodes.iter() {
                let op_name = node
                    .op
                    .as_ref()
                    .map(|o| o.name().to_owned())
                    .unwrap_or_default();
                let op_kind = classify_op(&op_name);
                let inputs: Vec<usize> = node.incoming_nodes.iter().map(|n| n.id).collect();
                let mut gn = fusion::patterns::GraphNode::new(node.id, op_kind, inputs, vec![]);
                gn.num_consumers = 0;
                graph_nodes.push(gn);
            }
        }

        // Count consumers so the fusion engine knows which nodes are "interior"
        // (single-consumer) and thus eligible as non-terminal fusion members.
        for idx in 0..graph_nodes.len() {
            let inputs: Vec<usize> = graph_nodes[idx].inputs.clone();
            for &inp in &inputs {
                if inp < graph_nodes.len() {
                    graph_nodes[inp].num_consumers += 1;
                }
            }
        }

        // Detect and apply fusions via the dedicated FusionOptimizer.
        let mut optimizer = fusion::FusionOptimizer::new();
        optimizer
            .detect_fusions_in_graph(&graph_nodes)
            .map_err(|e| OptimizationError::GraphStructure(e.to_string()))?;

        let fused_nodes = optimizer
            .apply_fusions_with_nodes(&graph_nodes)
            .map_err(|e| OptimizationError::GraphStructure(e.to_string()))?;

        Ok(fused_nodes.len())
    }

    /// Apply memory optimization via lifetime-based buffer reuse analysis.
    ///
    /// For each node we compute:
    ///   • `birth` — the node's own `topo_rank` (when its output is produced).
    ///   • `death` — the maximum `topo_rank` among all consumers of this node
    ///               (the last moment its output is needed).
    ///
    /// We then apply a greedy interval-graph colouring (scan in birth order,
    /// reuse the first freed slot) and count how many nodes share a buffer slot
    /// with an earlier node.  Each reuse is a potential memory saving.
    fn apply_memory_optimization(&self, graph: &Graph<F>) -> Result<usize, OptimizationError> {
        let node_count = graph.node_set.borrow().len();
        if node_count == 0 {
            return Ok(0);
        }

        // ── Collect topo_rank for every node ─────────────────────────────────
        let topo_ranks: Vec<usize> = {
            let nodes = graph.node_set.borrow();
            nodes.iter().map(|n| n.topo_rank).collect()
        };

        let max_rank = topo_ranks.iter().copied().max().unwrap_or(0);

        // ── Compute death time for each node ─────────────────────────────────
        // death[id] starts at topo_rank[id] and is updated to the max
        // topo_rank among all nodes that consume it.
        let mut death: Vec<usize> = topo_ranks.clone();

        {
            let nodes = graph.node_set.borrow();
            for node in nodes.iter() {
                let consumer_rank = node.topo_rank;
                for incoming in &node.incoming_nodes {
                    let pred = incoming.id;
                    if pred < node_count && consumer_rank > death[pred] {
                        death[pred] = consumer_rank;
                    }
                }
            }
        }

        // Nodes with no consumers (pure outputs) keep death == birth unless
        // we set them to max_rank (live until end of graph).
        {
            let nodes = graph.node_set.borrow();
            for id in 0..node_count {
                let has_consumer = nodes
                    .iter()
                    .any(|n| n.incoming_nodes.iter().any(|inc| inc.id == id));
                if !has_consumer {
                    death[id] = max_rank;
                }
            }
        }

        // ── Greedy interval-graph colouring ───────────────────────────────────
        // Sort intervals by birth time (ascending).
        let mut intervals: Vec<(usize, usize, TensorID)> = (0..node_count)
            .map(|id| (topo_ranks[id], death[id], id))
            .collect();
        intervals.sort_by_key(|&(birth, _, _)| birth);

        // `active_slots[i]` = death time of the last tensor assigned to slot i.
        let mut active_slots: Vec<usize> = Vec::new();
        let mut reuse_count = 0usize;

        for (birth, end, _node_id) in &intervals {
            // Find the first slot whose current occupant has already died.
            let released = active_slots
                .iter()
                .enumerate()
                .find(|(_, &slot_death)| slot_death < *birth)
                .map(|(idx, _)| idx);

            match released {
                Some(slot_idx) => {
                    active_slots[slot_idx] = *end;
                    reuse_count += 1;
                }
                None => {
                    active_slots.push(*end);
                }
            }
        }

        Ok(reuse_count)
    }
}

impl<F: Float> Default for GraphOptimizer<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Report of optimization results
#[derive(Debug, Clone, Default)]
pub struct OptimizationReport {
    /// Number of optimization passes completed
    pub passes_completed: usize,
    /// Number of constant folding optimizations applied
    pub constant_folding_applied: usize,
    /// Number of dead nodes eliminated
    pub dead_nodes_eliminated: usize,
    /// Number of common subexpressions eliminated
    pub cse_applied: usize,
    /// Number of expressions simplified
    pub expressions_simplified: usize,
    /// Number of operations fused
    pub operations_fused: usize,
    /// Number of memory optimizations applied
    pub memory_optimizations: usize,
}

impl OptimizationReport {
    /// Create a new empty optimization report
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the total number of optimizations applied
    pub fn total_optimizations(&self) -> usize {
        self.constant_folding_applied
            + self.dead_nodes_eliminated
            + self.cse_applied
            + self.expressions_simplified
            + self.operations_fused
            + self.memory_optimizations
    }

    /// Check if any optimizations were applied
    pub fn has_optimizations(&self) -> bool {
        self.total_optimizations() > 0
    }

    /// Print a summary of the optimization results
    pub fn print_summary(&self) {
        println!("Optimization Report:");
        println!("==================");
        println!("Passes completed: {}", self.passes_completed);
        println!("Total optimizations: {}", self.total_optimizations());

        if self.constant_folding_applied > 0 {
            println!("  Constant folding: {}", self.constant_folding_applied);
        }
        if self.dead_nodes_eliminated > 0 {
            println!("  Dead code elimination: {}", self.dead_nodes_eliminated);
        }
        if self.cse_applied > 0 {
            println!("  Common subexpression elimination: {}", self.cse_applied);
        }
        if self.expressions_simplified > 0 {
            println!(
                "  Expression simplification: {}",
                self.expressions_simplified
            );
        }
        if self.operations_fused > 0 {
            println!("  Operation fusion: {}", self.operations_fused);
        }
        if self.memory_optimizations > 0 {
            println!("  Memory optimizations: {}", self.memory_optimizations);
        }
    }
}

/// Expression pattern matcher for optimization
pub struct PatternMatcher<F: Float> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> PatternMatcher<F> {
    /// Create a new pattern matcher
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Check if a tensor matches a pattern for simplification
    #[allow(dead_code)]
    pub(crate) fn matches_simplification_pattern(
        &self,
        _tensor_internal: &TensorInternal<F>,
    ) -> Option<SimplificationPattern> {
        // Temporarily disabled - would be implemented with expression_simplification module
        None
    }

    /// Check if tensors can be fused
    #[allow(dead_code)]
    pub(crate) fn can_fuse(
        &self,
        _tensor1: &TensorInternal<F>,
        _tensor2: &TensorInternal<F>,
    ) -> bool {
        // Temporarily disabled - would be implemented with fusion analysis
        false
    }

    /// Check if a tensor represents a constant
    #[allow(dead_code)]
    pub(crate) fn is_constant(&self, _tensorinternal: &TensorInternal<F>) -> bool {
        // Temporarily disabled - would be implemented with constant analysis
        false
    }

    /// Check if a tensor is dead (unreachable from outputs)
    #[allow(dead_code)]
    pub(crate) fn is_dead(
        &self,
        _tensor_internal: &TensorInternal<F>,
        _reachable: &HashSet<TensorID>,
    ) -> bool {
        // Temporarily disabled - would be implemented with reachability analysis
        false
    }
}

impl<F: Float> Default for PatternMatcher<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of simplification patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimplificationPattern {
    /// x + 0 → x
    AddZero,
    /// x - 0 → x
    SubZero,
    /// x * 1 → x
    MulOne,
    /// x / 1 → x
    DivOne,
    /// x * 0 → 0
    MulZero,
    /// x - x → 0
    SubSelf,
    /// x / x → 1
    DivSelf,
    /// log(exp(x)) → x
    LogExp,
    /// exp(log(x)) → x
    ExpLog,
    /// sqrt(x^2) → abs(x)
    SqrtSquare,
    /// pow(x, 1) → x
    PowOne,
    /// pow(x, 0) → 1
    PowZero,
}

/// Optimization pass manager
pub struct OptimizationPass<F: Float> {
    name: String,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> OptimizationPass<F> {
    /// Create a new optimization pass
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the name of this pass
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Run this optimization pass on a graph
    pub fn run(&self, _graph: &Graph<F>) -> Result<usize, OptimizationError> {
        // Each pass would implement its specific optimization logic
        Ok(0)
    }
}

/// Errors that can occur during optimization
#[derive(Debug, thiserror::Error)]
pub enum OptimizationError {
    #[error("Graph structure error: {0}")]
    GraphStructure(String),
    #[error("Pattern matching error: {0}")]
    PatternMatching(String),
    #[error("Optimization conflict: {0}")]
    Conflict(String),
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

/// Public API functions for graph optimization
/// Optimize a computation graph with default settings
#[allow(dead_code)]
pub fn optimize_graph<F: Float>(graph: &Graph<F>) -> Result<OptimizationReport, OptimizationError> {
    let optimizer = GraphOptimizer::new();
    optimizer.optimize(graph)
}

/// Optimize a computation graph with specified optimization level
#[allow(dead_code)]
pub fn optimize_graph_with_level<F: Float>(
    graph: &Graph<F>,
    level: OptimizationLevel,
) -> Result<OptimizationReport, OptimizationError> {
    let optimizer = GraphOptimizer::with_level(level);
    optimizer.optimize(graph)
}

/// Optimize a computation graph with custom configuration
#[allow(dead_code)]
pub fn optimize_graph_with_config<F: Float>(
    graph: &Graph<F>,
    config: OptimizationConfig,
) -> Result<OptimizationReport, OptimizationError> {
    let optimizer = GraphOptimizer::with_config(config);
    optimizer.optimize(graph)
}

/// Apply only constant folding optimization
#[allow(dead_code)]
pub fn apply_constant_folding<F: Float>(graph: &Graph<F>) -> Result<usize, OptimizationError> {
    let config = OptimizationConfig {
        constant_folding: true,
        cse: false,
        expression_simplification: false,
        dead_code_elimination: false,
        operation_fusion: false,
        memory_optimization: false,
        max_passes: 1,
        level: OptimizationLevel::Basic,
    };
    let optimizer = GraphOptimizer::with_config(config);
    let report = optimizer.optimize(graph)?;
    Ok(report.constant_folding_applied)
}

/// Apply only dead code elimination
#[allow(dead_code)]
pub fn apply_dead_code_elimination<F: Float>(graph: &Graph<F>) -> Result<usize, OptimizationError> {
    let config = OptimizationConfig {
        constant_folding: false,
        cse: false,
        expression_simplification: false,
        dead_code_elimination: true,
        operation_fusion: false,
        memory_optimization: false,
        max_passes: 1,
        level: OptimizationLevel::Basic,
    };
    let optimizer = GraphOptimizer::with_config(config);
    let report = optimizer.optimize(graph)?;
    Ok(report.dead_nodes_eliminated)
}

/// Apply common subexpression elimination
#[allow(dead_code)]
pub fn apply_cse<F: Float>(graph: &Graph<F>) -> Result<usize, OptimizationError> {
    let config = OptimizationConfig {
        constant_folding: false,
        cse: true,
        expression_simplification: false,
        dead_code_elimination: false,
        operation_fusion: false,
        memory_optimization: false,
        max_passes: 1,
        level: OptimizationLevel::Standard,
    };
    let optimizer = GraphOptimizer::with_config(config);
    let report = optimizer.optimize(graph)?;
    Ok(report.cse_applied)
}

// Re-export types from the now-enabled submodules for convenience
pub use constant_folding::ConstantFolder;
pub use expression_simplification::ExpressionSimplifier;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::AsGraph;

    #[test]
    fn test_optimization_config() {
        let config = OptimizationConfig::default();
        assert!(config.constant_folding);
        assert!(config.cse);
        assert!(config.expression_simplification);
        assert!(config.dead_code_elimination);
        assert_eq!(config.max_passes, 5);
    }

    #[test]
    fn test_optimization_levels() {
        let none_config = OptimizationLevel::None.config();
        assert!(!none_config.constant_folding);
        assert_eq!(none_config.max_passes, 0);

        let aggressive_config = OptimizationLevel::Aggressive.config();
        assert!(aggressive_config.operation_fusion);
        assert!(aggressive_config.memory_optimization);
        assert_eq!(aggressive_config.max_passes, 10);
    }

    #[test]
    fn test_graph_optimizer_creation() {
        let _optimizer = GraphOptimizer::<f32>::new();
        let _optimizer_with_config =
            GraphOptimizer::<f32>::with_config(OptimizationConfig::default());
        let _optimizer_with_level =
            GraphOptimizer::<f32>::with_level(OptimizationLevel::Aggressive);
    }

    #[test]
    fn test_optimization_report() {
        let mut report = OptimizationReport::new();
        assert_eq!(report.total_optimizations(), 0);
        assert!(!report.has_optimizations());

        report.constant_folding_applied = 5;
        report.dead_nodes_eliminated = 3;
        assert_eq!(report.total_optimizations(), 8);
        assert!(report.has_optimizations());
    }

    #[test]
    fn test_pattern_matcher() {
        let _matcher = PatternMatcher::<f32>::new();
    }

    #[test]
    fn test_simplification_patterns() {
        let pattern = SimplificationPattern::AddZero;
        assert_eq!(pattern, SimplificationPattern::AddZero);

        let patterns = [
            SimplificationPattern::AddZero,
            SimplificationPattern::MulOne,
            SimplificationPattern::LogExp,
        ];
        assert_eq!(patterns.len(), 3);
    }

    #[test]
    fn test_optimization_pass() {
        let pass = OptimizationPass::<f32>::new("test_pass");
        assert_eq!(pass.name(), "test_pass");
    }

    // ── Integration tests against real computation graphs ──────────────────

    /// DCE: a node that is not on the path to the output should be counted as dead.
    #[test]
    fn test_dce_on_real_graph() {
        use crate::tensor_ops as T;
        use crate::VariableEnvironment;

        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            // Build a small graph.
            // `_b` is a tensor that is never used as input to any other node.
            // The final "output" is `c` (highest topo_rank).
            let a = T::zeros(&[2, 2], ctx);
            let _b = T::ones(&[2, 2], ctx); // dead — never consumed
            let c = T::mul(a, T::ones(&[2, 2], ctx));
            let _ = c;

            let optimizer = GraphOptimizer::<f32>::with_config(OptimizationConfig {
                constant_folding: false,
                cse: false,
                expression_simplification: false,
                dead_code_elimination: true,
                operation_fusion: false,
                memory_optimization: false,
                max_passes: 1,
                level: OptimizationLevel::Basic,
            });

            let report = optimizer
                .optimize(ctx.as_graph())
                .expect("DCE should succeed");

            assert!(
                report.dead_nodes_eliminated >= 1,
                "Expected at least 1 dead node, got {}",
                report.dead_nodes_eliminated
            );
        });
    }

    /// CSE: two identical `add(a, b)` nodes should result in one elimination.
    #[test]
    fn test_cse_on_real_graph() {
        use crate::tensor_ops as T;
        use crate::VariableEnvironment;

        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[2, 2], ctx);
            let b = T::ones(&[2, 2], ctx);
            // Compute a + b twice — same op name and same inputs.
            let c1 = T::add(a, b);
            let c2 = T::add(a, b);
            // Consume both so neither is dead for DCE purposes.
            let _ = T::add(c1, c2);

            let optimizer = GraphOptimizer::<f32>::with_config(OptimizationConfig {
                constant_folding: false,
                cse: true,
                expression_simplification: false,
                dead_code_elimination: false,
                operation_fusion: false,
                memory_optimization: false,
                max_passes: 1,
                level: OptimizationLevel::Standard,
            });

            let report = optimizer
                .optimize(ctx.as_graph())
                .expect("CSE should succeed");

            assert!(
                report.cse_applied >= 1,
                "Expected >= 1 CSE elimination, got {}",
                report.cse_applied
            );
        });
    }

    /// Memory optimisation: in a linear chain, at least one buffer reuse should
    /// be detected.
    #[test]
    fn test_memory_opt_on_real_graph() {
        use crate::tensor_ops as T;
        use crate::VariableEnvironment;

        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[4, 4], ctx);
            let b = T::mul(a, T::ones(&[4, 4], ctx));
            let c = T::add(b, T::ones(&[4, 4], ctx));
            let d = T::mul(c, T::ones(&[4, 4], ctx));
            let _ = d;

            let optimizer = GraphOptimizer::<f32>::with_config(OptimizationConfig {
                constant_folding: false,
                cse: false,
                expression_simplification: false,
                dead_code_elimination: false,
                operation_fusion: false,
                memory_optimization: true,
                max_passes: 1,
                level: OptimizationLevel::Standard,
            });

            let report = optimizer
                .optimize(ctx.as_graph())
                .expect("Memory opt should succeed");

            assert!(
                report.memory_optimizations >= 1,
                "Expected >= 1 memory reuse opportunity, got {}",
                report.memory_optimizations
            );
        });
    }

    /// An empty graph should produce zero optimisations and not panic.
    #[test]
    fn test_empty_graph_all_passes() {
        use crate::VariableEnvironment;

        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let optimizer = GraphOptimizer::<f32>::new();
            let report = optimizer.optimize(ctx.as_graph()).expect("Empty graph OK");
            assert_eq!(report.dead_nodes_eliminated, 0);
            assert_eq!(report.cse_applied, 0);
            assert_eq!(report.memory_optimizations, 0);
        });
    }
}
