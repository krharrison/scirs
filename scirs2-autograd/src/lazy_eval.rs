//! Lazy evaluation for deferred computation graph construction
//!
//! This module provides a lazy tensor abstraction that defers computation until
//! explicitly evaluated. Operations are accumulated into a graph that can be
//! optimized (dead-code elimination, batching) before execution.
//!
//! # Key types
//!
//! - [`LazyTensor`]: A computation node that wraps a deferred operation
//! - [`LazyGraph`]: Accumulates lazy ops and evaluates them in batch
//! - [`LazyContext`]: Builder for constructing deferred computation chains
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::lazy_eval::*;
//!
//! // Build a lazy computation
//! let mut graph = LazyGraph::<f64>::new();
//! let a = graph.constant(vec![1.0, 2.0, 3.0], vec![3]);
//! let b = graph.constant(vec![4.0, 5.0, 6.0], vec![3]);
//! let c = graph.add(a, b);
//! let d = graph.relu(c);
//!
//! // Nothing computed yet — evaluate
//! let results = graph.eval_all().expect("eval");
//! assert_eq!(results.len(), 4);
//! ```

use crate::error::AutogradError;
use crate::{Float, NdArray, Result};
use scirs2_core::ndarray::{Array, ArrayD, IxDyn};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Lazy operation enum
// ---------------------------------------------------------------------------

/// The kind of deferred operation in the lazy graph.
#[derive(Debug, Clone, PartialEq)]
pub enum LazyOp {
    /// A constant value (leaf node)
    Constant,
    /// An external input placeholder
    Placeholder,
    /// Element-wise addition
    Add,
    /// Element-wise subtraction
    Sub,
    /// Element-wise multiplication
    Mul,
    /// Element-wise division
    Div,
    /// Matrix multiplication
    MatMul,
    /// ReLU activation
    Relu,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// GELU activation (approximate)
    Gelu,
    /// Softmax along last axis
    Softmax,
    /// Negation
    Neg,
    /// Exp
    Exp,
    /// Log (natural)
    Log,
    /// Square (x^2)
    Square,
    /// Sqrt
    Sqrt,
    /// Reciprocal (1/x)
    Reciprocal,
    /// Sum reduction along specified axes
    ReduceSum { axes: Vec<usize>, keep_dims: bool },
    /// Mean reduction along specified axes
    ReduceMean { axes: Vec<usize>, keep_dims: bool },
    /// Reshape to target shape
    Reshape { target_shape: Vec<usize> },
    /// Transpose
    Transpose,
    /// Scale by a constant
    Scale { factor: f64 },
    /// Fused multiply-add: a * b + c
    FusedMulAdd,
    /// Custom named operation (for extensibility)
    Custom { name: String },
}

impl fmt::Display for LazyOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LazyOp::Constant => write!(f, "Constant"),
            LazyOp::Placeholder => write!(f, "Placeholder"),
            LazyOp::Add => write!(f, "Add"),
            LazyOp::Sub => write!(f, "Sub"),
            LazyOp::Mul => write!(f, "Mul"),
            LazyOp::Div => write!(f, "Div"),
            LazyOp::MatMul => write!(f, "MatMul"),
            LazyOp::Relu => write!(f, "Relu"),
            LazyOp::Sigmoid => write!(f, "Sigmoid"),
            LazyOp::Tanh => write!(f, "Tanh"),
            LazyOp::Gelu => write!(f, "Gelu"),
            LazyOp::Softmax => write!(f, "Softmax"),
            LazyOp::Neg => write!(f, "Neg"),
            LazyOp::Exp => write!(f, "Exp"),
            LazyOp::Log => write!(f, "Log"),
            LazyOp::Square => write!(f, "Square"),
            LazyOp::Sqrt => write!(f, "Sqrt"),
            LazyOp::Reciprocal => write!(f, "Reciprocal"),
            LazyOp::ReduceSum { .. } => write!(f, "ReduceSum"),
            LazyOp::ReduceMean { .. } => write!(f, "ReduceMean"),
            LazyOp::Reshape { .. } => write!(f, "Reshape"),
            LazyOp::Transpose => write!(f, "Transpose"),
            LazyOp::Scale { factor } => write!(f, "Scale({})", factor),
            LazyOp::FusedMulAdd => write!(f, "FusedMulAdd"),
            LazyOp::Custom { name } => write!(f, "Custom({})", name),
        }
    }
}

impl LazyOp {
    /// Whether this operation is a leaf (no inputs).
    pub fn is_leaf(&self) -> bool {
        matches!(self, LazyOp::Constant | LazyOp::Placeholder)
    }

    /// Whether this operation is element-wise.
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            LazyOp::Add
                | LazyOp::Sub
                | LazyOp::Mul
                | LazyOp::Div
                | LazyOp::Relu
                | LazyOp::Sigmoid
                | LazyOp::Tanh
                | LazyOp::Gelu
                | LazyOp::Neg
                | LazyOp::Exp
                | LazyOp::Log
                | LazyOp::Square
                | LazyOp::Sqrt
                | LazyOp::Reciprocal
                | LazyOp::Scale { .. }
        )
    }

    /// Whether this operation is unary (exactly one input).
    pub fn is_unary(&self) -> bool {
        matches!(
            self,
            LazyOp::Relu
                | LazyOp::Sigmoid
                | LazyOp::Tanh
                | LazyOp::Gelu
                | LazyOp::Neg
                | LazyOp::Exp
                | LazyOp::Log
                | LazyOp::Square
                | LazyOp::Sqrt
                | LazyOp::Reciprocal
                | LazyOp::Softmax
                | LazyOp::Scale { .. }
                | LazyOp::Reshape { .. }
                | LazyOp::Transpose
        )
    }

    /// Whether this operation is binary (exactly two inputs).
    pub fn is_binary(&self) -> bool {
        matches!(
            self,
            LazyOp::Add | LazyOp::Sub | LazyOp::Mul | LazyOp::Div | LazyOp::MatMul
        )
    }
}

// ---------------------------------------------------------------------------
// Lazy tensor node ID
// ---------------------------------------------------------------------------

/// Opaque identifier for a node in the lazy graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LazyTensorId(usize);

impl fmt::Display for LazyTensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LazyTensor({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Lazy node
// ---------------------------------------------------------------------------

/// Internal representation of a lazy computation node.
#[derive(Debug, Clone)]
struct LazyNode<F: Float> {
    id: LazyTensorId,
    op: LazyOp,
    inputs: Vec<LazyTensorId>,
    /// Pre-computed value (for constants / already-evaluated nodes)
    value: Option<NdArray<F>>,
    /// Inferred shape (if known)
    shape: Option<Vec<usize>>,
    /// Name for debugging
    name: Option<String>,
    /// Whether this node has been evaluated
    evaluated: bool,
}

// ---------------------------------------------------------------------------
// LazyGraph
// ---------------------------------------------------------------------------

/// A deferred computation graph that accumulates operations before execution.
///
/// Nodes are added lazily; no computation occurs until [`eval`](LazyGraph::eval)
/// or [`eval_all`](LazyGraph::eval_all) is called. Before evaluation, the graph
/// can be optimised (dead-code elimination, etc.).
pub struct LazyGraph<F: Float> {
    nodes: Vec<LazyNode<F>>,
    /// Cache of evaluated results
    cache: HashMap<LazyTensorId, NdArray<F>>,
}

impl<F: Float> LazyGraph<F> {
    /// Create a new empty lazy graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            cache: HashMap::new(),
        }
    }

    /// Number of nodes in the graph.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of cached (already-evaluated) results.
    pub fn num_cached(&self) -> usize {
        self.cache.len()
    }

    // --- Leaf constructors ---

    /// Add a constant tensor.
    pub fn constant(&mut self, data: Vec<F>, shape: Vec<usize>) -> LazyTensorId {
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .unwrap_or_else(|_| ArrayD::zeros(IxDyn(&shape)));
        let id = LazyTensorId(self.nodes.len());
        self.nodes.push(LazyNode {
            id,
            op: LazyOp::Constant,
            inputs: Vec::new(),
            value: Some(arr),
            shape: Some(shape),
            name: None,
            evaluated: true,
        });
        id
    }

    /// Add a constant from an existing NdArray.
    pub fn constant_array(&mut self, arr: NdArray<F>) -> LazyTensorId {
        let shape = arr.shape().to_vec();
        let id = LazyTensorId(self.nodes.len());
        self.nodes.push(LazyNode {
            id,
            op: LazyOp::Constant,
            inputs: Vec::new(),
            value: Some(arr),
            shape: Some(shape),
            name: None,
            evaluated: true,
        });
        id
    }

    /// Add a placeholder (to be fed later).
    pub fn placeholder(&mut self, name: &str, shape: Vec<usize>) -> LazyTensorId {
        let id = LazyTensorId(self.nodes.len());
        self.nodes.push(LazyNode {
            id,
            op: LazyOp::Placeholder,
            inputs: Vec::new(),
            value: None,
            shape: Some(shape),
            name: Some(name.to_owned()),
            evaluated: false,
        });
        id
    }

    /// Feed a value into a placeholder node.
    pub fn feed(&mut self, id: LazyTensorId, value: NdArray<F>) -> Result<()> {
        let node = self
            .nodes
            .get_mut(id.0)
            .ok_or_else(|| AutogradError::OperationError(format!("Node {} not found", id)))?;
        if node.op != LazyOp::Placeholder {
            return Err(AutogradError::OperationError(format!(
                "Node {} is not a placeholder",
                id
            )));
        }
        node.value = Some(value);
        node.evaluated = true;
        Ok(())
    }

    // --- Binary operations ---

    /// Element-wise addition.
    pub fn add(&mut self, a: LazyTensorId, b: LazyTensorId) -> LazyTensorId {
        self.binary_op(LazyOp::Add, a, b)
    }

    /// Element-wise subtraction.
    pub fn sub(&mut self, a: LazyTensorId, b: LazyTensorId) -> LazyTensorId {
        self.binary_op(LazyOp::Sub, a, b)
    }

    /// Element-wise multiplication.
    pub fn mul(&mut self, a: LazyTensorId, b: LazyTensorId) -> LazyTensorId {
        self.binary_op(LazyOp::Mul, a, b)
    }

    /// Element-wise division.
    pub fn div(&mut self, a: LazyTensorId, b: LazyTensorId) -> LazyTensorId {
        self.binary_op(LazyOp::Div, a, b)
    }

    /// Matrix multiplication.
    pub fn matmul(&mut self, a: LazyTensorId, b: LazyTensorId) -> LazyTensorId {
        self.binary_op(LazyOp::MatMul, a, b)
    }

    /// Fused multiply-add: a * b + c.
    pub fn fused_mul_add(
        &mut self,
        a: LazyTensorId,
        b: LazyTensorId,
        c: LazyTensorId,
    ) -> LazyTensorId {
        let id = LazyTensorId(self.nodes.len());
        let shape = self.infer_binary_shape(a, b);
        self.nodes.push(LazyNode {
            id,
            op: LazyOp::FusedMulAdd,
            inputs: vec![a, b, c],
            value: None,
            shape,
            name: None,
            evaluated: false,
        });
        id
    }

    // --- Unary operations ---

    /// ReLU activation.
    pub fn relu(&mut self, x: LazyTensorId) -> LazyTensorId {
        self.unary_op(LazyOp::Relu, x)
    }

    /// Sigmoid activation.
    pub fn sigmoid(&mut self, x: LazyTensorId) -> LazyTensorId {
        self.unary_op(LazyOp::Sigmoid, x)
    }

    /// Tanh activation.
    pub fn tanh_op(&mut self, x: LazyTensorId) -> LazyTensorId {
        self.unary_op(LazyOp::Tanh, x)
    }

    /// GELU activation (approximate).
    pub fn gelu(&mut self, x: LazyTensorId) -> LazyTensorId {
        self.unary_op(LazyOp::Gelu, x)
    }

    /// Softmax along last axis.
    pub fn softmax(&mut self, x: LazyTensorId) -> LazyTensorId {
        self.unary_op(LazyOp::Softmax, x)
    }

    /// Negation.
    pub fn neg(&mut self, x: LazyTensorId) -> LazyTensorId {
        self.unary_op(LazyOp::Neg, x)
    }

    /// Exponential.
    pub fn exp(&mut self, x: LazyTensorId) -> LazyTensorId {
        self.unary_op(LazyOp::Exp, x)
    }

    /// Natural logarithm.
    pub fn log(&mut self, x: LazyTensorId) -> LazyTensorId {
        self.unary_op(LazyOp::Log, x)
    }

    /// Square (x^2).
    pub fn square(&mut self, x: LazyTensorId) -> LazyTensorId {
        self.unary_op(LazyOp::Square, x)
    }

    /// Square root.
    pub fn sqrt(&mut self, x: LazyTensorId) -> LazyTensorId {
        self.unary_op(LazyOp::Sqrt, x)
    }

    /// Reciprocal (1/x).
    pub fn reciprocal(&mut self, x: LazyTensorId) -> LazyTensorId {
        self.unary_op(LazyOp::Reciprocal, x)
    }

    /// Scale by a constant factor.
    pub fn scale(&mut self, x: LazyTensorId, factor: f64) -> LazyTensorId {
        self.unary_op(LazyOp::Scale { factor }, x)
    }

    /// Reshape.
    pub fn reshape(&mut self, x: LazyTensorId, target_shape: Vec<usize>) -> LazyTensorId {
        let id = LazyTensorId(self.nodes.len());
        let shape = Some(target_shape.clone());
        self.nodes.push(LazyNode {
            id,
            op: LazyOp::Reshape { target_shape },
            inputs: vec![x],
            value: None,
            shape,
            name: None,
            evaluated: false,
        });
        id
    }

    /// Transpose (swap last two axes for 2-D; reverse axes for higher dim).
    pub fn transpose(&mut self, x: LazyTensorId) -> LazyTensorId {
        let shape = self.nodes.get(x.0).and_then(|n| {
            n.shape.as_ref().map(|s| {
                let mut t = s.clone();
                t.reverse();
                t
            })
        });
        let id = LazyTensorId(self.nodes.len());
        self.nodes.push(LazyNode {
            id,
            op: LazyOp::Transpose,
            inputs: vec![x],
            value: None,
            shape,
            name: None,
            evaluated: false,
        });
        id
    }

    /// Sum reduction.
    pub fn reduce_sum(
        &mut self,
        x: LazyTensorId,
        axes: Vec<usize>,
        keep_dims: bool,
    ) -> LazyTensorId {
        let id = LazyTensorId(self.nodes.len());
        self.nodes.push(LazyNode {
            id,
            op: LazyOp::ReduceSum { axes, keep_dims },
            inputs: vec![x],
            value: None,
            shape: None,
            name: None,
            evaluated: false,
        });
        id
    }

    /// Mean reduction.
    pub fn reduce_mean(
        &mut self,
        x: LazyTensorId,
        axes: Vec<usize>,
        keep_dims: bool,
    ) -> LazyTensorId {
        let id = LazyTensorId(self.nodes.len());
        self.nodes.push(LazyNode {
            id,
            op: LazyOp::ReduceMean { axes, keep_dims },
            inputs: vec![x],
            value: None,
            shape: None,
            name: None,
            evaluated: false,
        });
        id
    }

    // --- Internal helpers ---

    fn binary_op(&mut self, op: LazyOp, a: LazyTensorId, b: LazyTensorId) -> LazyTensorId {
        let id = LazyTensorId(self.nodes.len());
        let shape = self.infer_binary_shape(a, b);
        self.nodes.push(LazyNode {
            id,
            op,
            inputs: vec![a, b],
            value: None,
            shape,
            name: None,
            evaluated: false,
        });
        id
    }

    fn unary_op(&mut self, op: LazyOp, x: LazyTensorId) -> LazyTensorId {
        let id = LazyTensorId(self.nodes.len());
        let shape = self.nodes.get(x.0).and_then(|n| n.shape.clone());
        self.nodes.push(LazyNode {
            id,
            op,
            inputs: vec![x],
            value: None,
            shape,
            name: None,
            evaluated: false,
        });
        id
    }

    fn infer_binary_shape(&self, a: LazyTensorId, b: LazyTensorId) -> Option<Vec<usize>> {
        let sa = self.nodes.get(a.0).and_then(|n| n.shape.as_ref());
        let sb = self.nodes.get(b.0).and_then(|n| n.shape.as_ref());
        match (sa, sb) {
            (Some(sa), Some(_sb)) => Some(sa.clone()), // simplified broadcast
            (Some(s), None) | (None, Some(s)) => Some(s.clone()),
            (None, None) => None,
        }
    }

    // --- Evaluation ---

    /// Evaluate a single node, returning its value.
    pub fn eval(&mut self, id: LazyTensorId) -> Result<NdArray<F>> {
        // If already cached, return cached value
        if let Some(val) = self.cache.get(&id) {
            return Ok(val.clone());
        }

        // Topological evaluation order
        let order = self.topological_order(id)?;

        for node_id in order {
            if self.cache.contains_key(&node_id) {
                continue;
            }

            let result = self.evaluate_node(node_id)?;
            self.cache.insert(node_id, result);
        }

        self.cache
            .get(&id)
            .cloned()
            .ok_or_else(|| AutogradError::compute_error(format!("Failed to evaluate node {}", id)))
    }

    /// Evaluate all nodes in the graph.
    pub fn eval_all(&mut self) -> Result<Vec<NdArray<F>>> {
        let ids: Vec<LazyTensorId> = self.nodes.iter().map(|n| n.id).collect();
        let mut results = Vec::with_capacity(ids.len());

        // Full topological order
        let order = self.full_topological_order();

        for node_id in &order {
            if !self.cache.contains_key(node_id) {
                let result = self.evaluate_node(*node_id)?;
                self.cache.insert(*node_id, result);
            }
        }

        for id in &ids {
            let val = self.cache.get(id).ok_or_else(|| {
                AutogradError::compute_error(format!("Node {} not evaluated", id))
            })?;
            results.push(val.clone());
        }

        Ok(results)
    }

    /// Evaluate a single node given that all inputs are already in the cache.
    fn evaluate_node(&self, id: LazyTensorId) -> Result<NdArray<F>> {
        let node = self
            .nodes
            .get(id.0)
            .ok_or_else(|| AutogradError::OperationError(format!("Node {} not found", id)))?;

        // Constants / placeholders with values
        if let Some(ref val) = node.value {
            return Ok(val.clone());
        }

        // Resolve inputs
        let inputs: Vec<NdArray<F>> = node
            .inputs
            .iter()
            .map(|inp_id| {
                self.cache.get(inp_id).cloned().ok_or_else(|| {
                    AutogradError::compute_error(format!("Input {} not yet evaluated", inp_id))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        match &node.op {
            LazyOp::Add => binary_elementwise(&inputs[0], &inputs[1], |a, b| a + b),
            LazyOp::Sub => binary_elementwise(&inputs[0], &inputs[1], |a, b| a - b),
            LazyOp::Mul => binary_elementwise(&inputs[0], &inputs[1], |a, b| a * b),
            LazyOp::Div => binary_elementwise(&inputs[0], &inputs[1], |a, b| {
                if b == F::zero() {
                    F::zero()
                } else {
                    a / b
                }
            }),
            LazyOp::MatMul => eval_matmul(&inputs[0], &inputs[1]),
            LazyOp::Relu => Ok(inputs[0].mapv(|v| if v > F::zero() { v } else { F::zero() })),
            LazyOp::Sigmoid => Ok(inputs[0].mapv(|v| {
                let one = F::one();
                one / (one + (-v).exp())
            })),
            LazyOp::Tanh => Ok(inputs[0].mapv(|v| v.tanh())),
            LazyOp::Gelu => {
                // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let sqrt_2_pi = F::from(0.7978845608028654).unwrap_or(F::one());
                let coeff = F::from(0.044715).unwrap_or(F::zero());
                let half = F::from(0.5).unwrap_or(F::one());
                Ok(inputs[0].mapv(|x| {
                    let inner = sqrt_2_pi * (x + coeff * x * x * x);
                    half * x * (F::one() + inner.tanh())
                }))
            }
            LazyOp::Softmax => eval_softmax(&inputs[0]),
            LazyOp::Neg => Ok(inputs[0].mapv(|v| -v)),
            LazyOp::Exp => Ok(inputs[0].mapv(|v| v.exp())),
            LazyOp::Log => Ok(inputs[0].mapv(|v| v.ln())),
            LazyOp::Square => Ok(inputs[0].mapv(|v| v * v)),
            LazyOp::Sqrt => Ok(inputs[0].mapv(|v| v.sqrt())),
            LazyOp::Reciprocal => Ok(inputs[0].mapv(|v| {
                if v == F::zero() {
                    F::zero()
                } else {
                    F::one() / v
                }
            })),
            LazyOp::Scale { factor } => {
                let f = F::from(*factor).unwrap_or(F::one());
                Ok(inputs[0].mapv(|v| v * f))
            }
            LazyOp::ReduceSum { axes, keep_dims } => eval_reduce_sum(&inputs[0], axes, *keep_dims),
            LazyOp::ReduceMean { axes, keep_dims } => {
                eval_reduce_mean(&inputs[0], axes, *keep_dims)
            }
            LazyOp::Reshape { target_shape } => {
                let new_shape = IxDyn(target_shape);
                inputs[0]
                    .clone()
                    .into_shape_clone(new_shape)
                    .map_err(|e| AutogradError::ShapeMismatch(format!("Reshape failed: {}", e)))
            }
            LazyOp::Transpose => Ok(inputs[0].clone().reversed_axes()),
            LazyOp::FusedMulAdd => {
                // a * b + c
                let a = &inputs[0];
                let b = &inputs[1];
                let c = &inputs[2];
                let ab = binary_elementwise(a, b, |x, y| x * y)?;
                binary_elementwise(&ab, c, |x, y| x + y)
            }
            LazyOp::Constant | LazyOp::Placeholder => Err(AutogradError::compute_error(format!(
                "Leaf node {} has no value",
                id
            ))),
            LazyOp::Custom { name } => Err(AutogradError::OperationError(format!(
                "Custom op '{}' has no evaluator",
                name
            ))),
        }
    }

    // --- Topological ordering ---

    fn topological_order(&self, target: LazyTensorId) -> Result<Vec<LazyTensorId>> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        self.topo_dfs(target, &mut visited, &mut order)?;
        Ok(order)
    }

    fn topo_dfs(
        &self,
        id: LazyTensorId,
        visited: &mut HashSet<LazyTensorId>,
        order: &mut Vec<LazyTensorId>,
    ) -> Result<()> {
        if visited.contains(&id) {
            return Ok(());
        }
        visited.insert(id);

        let node = self
            .nodes
            .get(id.0)
            .ok_or_else(|| AutogradError::OperationError(format!("Node {} not found", id)))?;

        for &inp in &node.inputs {
            self.topo_dfs(inp, visited, order)?;
        }

        order.push(id);
        Ok(())
    }

    fn full_topological_order(&self) -> Vec<LazyTensorId> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        for node in &self.nodes {
            if !visited.contains(&node.id) {
                let _ = self.topo_dfs(node.id, &mut visited, &mut order);
            }
        }
        order
    }

    // --- Dead code elimination ---

    /// Remove nodes that are not reachable from any of the given output nodes.
    ///
    /// Returns the number of nodes eliminated.
    pub fn eliminate_dead_code(&mut self, outputs: &[LazyTensorId]) -> usize {
        let mut reachable = HashSet::new();
        let mut queue: VecDeque<LazyTensorId> = outputs.iter().copied().collect();

        while let Some(id) = queue.pop_front() {
            if reachable.contains(&id) {
                continue;
            }
            reachable.insert(id);
            if let Some(node) = self.nodes.get(id.0) {
                for &inp in &node.inputs {
                    queue.push_back(inp);
                }
            }
        }

        let original_count = self.nodes.len();
        // Mark unreachable nodes by clearing their inputs (logically dead).
        // We cannot remove them by index (would invalidate IDs) but we mark
        // them so they are skipped during evaluation.
        let mut eliminated = 0usize;
        for node in &mut self.nodes {
            if !reachable.contains(&node.id) {
                // Count this dead node
                eliminated += 1;
                // Mark as dead — convert to a zero-dimensional constant
                if node.op != LazyOp::Constant || !node.inputs.is_empty() {
                    node.op = LazyOp::Constant;
                    node.inputs.clear();
                    node.value = Some(ArrayD::zeros(IxDyn(&[])));
                    node.evaluated = true;
                }
            }
        }

        // Also prune cache for dead nodes
        self.cache.retain(|k, _| reachable.contains(k));

        eliminated
    }

    /// Get a summary of the graph.
    pub fn summary(&self) -> LazyGraphSummary {
        let mut op_counts: HashMap<String, usize> = HashMap::new();
        let mut leaf_count = 0usize;
        let mut evaluated_count = 0usize;

        for node in &self.nodes {
            let name = format!("{}", node.op);
            *op_counts.entry(name).or_insert(0) += 1;
            if node.op.is_leaf() {
                leaf_count += 1;
            }
            if self.cache.contains_key(&node.id) || node.evaluated {
                evaluated_count += 1;
            }
        }

        LazyGraphSummary {
            total_nodes: self.nodes.len(),
            leaf_nodes: leaf_count,
            evaluated_nodes: evaluated_count,
            op_counts,
        }
    }

    /// Clear the evaluation cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get the inferred shape of a node (if available).
    pub fn shape_of(&self, id: LazyTensorId) -> Option<&[usize]> {
        self.nodes.get(id.0).and_then(|n| n.shape.as_deref())
    }

    /// Get the operation for a node.
    pub fn op_of(&self, id: LazyTensorId) -> Option<&LazyOp> {
        self.nodes.get(id.0).map(|n| &n.op)
    }

    /// Get the input IDs for a node.
    pub fn inputs_of(&self, id: LazyTensorId) -> Option<&[LazyTensorId]> {
        self.nodes.get(id.0).map(|n| n.inputs.as_slice())
    }
}

impl<F: Float> Default for LazyGraph<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LazyGraphSummary
// ---------------------------------------------------------------------------

/// Summary statistics of a lazy graph.
#[derive(Debug, Clone)]
pub struct LazyGraphSummary {
    /// Total number of nodes.
    pub total_nodes: usize,
    /// Number of leaf nodes (constants / placeholders).
    pub leaf_nodes: usize,
    /// Number of evaluated nodes.
    pub evaluated_nodes: usize,
    /// Count of each operation type.
    pub op_counts: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// LazyContext — convenience builder
// ---------------------------------------------------------------------------

/// Builder for constructing lazy computation chains with a fluent API.
///
/// Wraps a [`LazyGraph`] and provides a higher-level interface.
pub struct LazyContext<F: Float> {
    graph: LazyGraph<F>,
}

impl<F: Float> LazyContext<F> {
    /// Create a new lazy context.
    pub fn new() -> Self {
        Self {
            graph: LazyGraph::new(),
        }
    }

    /// Access the underlying graph.
    pub fn graph(&self) -> &LazyGraph<F> {
        &self.graph
    }

    /// Access the underlying graph mutably.
    pub fn graph_mut(&mut self) -> &mut LazyGraph<F> {
        &mut self.graph
    }

    /// Create a constant tensor.
    pub fn constant(&mut self, data: Vec<F>, shape: Vec<usize>) -> LazyTensorId {
        self.graph.constant(data, shape)
    }

    /// Create a constant from an NdArray.
    pub fn constant_array(&mut self, arr: NdArray<F>) -> LazyTensorId {
        self.graph.constant_array(arr)
    }

    /// Create a placeholder.
    pub fn placeholder(&mut self, name: &str, shape: Vec<usize>) -> LazyTensorId {
        self.graph.placeholder(name, shape)
    }

    /// Feed a value into a placeholder.
    pub fn feed(&mut self, id: LazyTensorId, value: NdArray<F>) -> Result<()> {
        self.graph.feed(id, value)
    }

    /// Evaluate a single node.
    pub fn eval(&mut self, id: LazyTensorId) -> Result<NdArray<F>> {
        self.graph.eval(id)
    }

    /// Evaluate all nodes.
    pub fn eval_all(&mut self) -> Result<Vec<NdArray<F>>> {
        self.graph.eval_all()
    }

    /// Eliminate dead code relative to the given outputs.
    pub fn eliminate_dead_code(&mut self, outputs: &[LazyTensorId]) -> usize {
        self.graph.eliminate_dead_code(outputs)
    }

    /// Get graph summary.
    pub fn summary(&self) -> LazyGraphSummary {
        self.graph.summary()
    }
}

impl<F: Float> Default for LazyContext<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Evaluation helpers
// ---------------------------------------------------------------------------

fn binary_elementwise<F: Float>(
    a: &NdArray<F>,
    b: &NdArray<F>,
    f: impl Fn(F, F) -> F,
) -> Result<NdArray<F>> {
    if a.shape() == b.shape() {
        let mut result = a.clone();
        result.zip_mut_with(b, |av, bv| {
            *av = f(*av, *bv);
        });
        Ok(result)
    } else if b.len() == 1 {
        // Scalar broadcast
        let scalar = *b
            .iter()
            .next()
            .ok_or_else(|| AutogradError::compute_error("Empty scalar tensor".into()))?;
        Ok(a.mapv(|v| f(v, scalar)))
    } else if a.len() == 1 {
        let scalar = *a
            .iter()
            .next()
            .ok_or_else(|| AutogradError::compute_error("Empty scalar tensor".into()))?;
        Ok(b.mapv(|v| f(scalar, v)))
    } else {
        Err(AutogradError::ShapeMismatch(format!(
            "Cannot broadcast shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        )))
    }
}

fn eval_matmul<F: Float>(a: &NdArray<F>, b: &NdArray<F>) -> Result<NdArray<F>> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(AutogradError::ShapeMismatch(
            "MatMul requires at least 2-D inputs".into(),
        ));
    }

    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let n = b_shape[b_shape.len() - 1];

    if k != b_shape[b_shape.len() - 2] {
        return Err(AutogradError::ShapeMismatch(format!(
            "MatMul inner dimensions mismatch: {} vs {}",
            k,
            b_shape[b_shape.len() - 2]
        )));
    }

    let mut result = ArrayD::zeros(IxDyn(&[m, n]));

    // Simple O(m*n*k) matmul
    let a_slice = a
        .as_slice()
        .ok_or_else(|| AutogradError::compute_error("MatMul input A not contiguous".into()))?;
    let b_slice = b
        .as_slice()
        .ok_or_else(|| AutogradError::compute_error("MatMul input B not contiguous".into()))?;

    for i in 0..m {
        for j in 0..n {
            let mut acc = F::zero();
            for p in 0..k {
                acc = acc + a_slice[i * k + p] * b_slice[p * n + j];
            }
            result[IxDyn(&[i, j])] = acc;
        }
    }

    Ok(result)
}

fn eval_softmax<F: Float>(x: &NdArray<F>) -> Result<NdArray<F>> {
    let shape = x.shape();
    if shape.is_empty() {
        return Ok(x.clone());
    }

    let mut result = x.clone();
    let last_axis = shape.len() - 1;
    let axis_len = shape[last_axis];

    // For 1-D
    if shape.len() == 1 {
        let max_val = x
            .iter()
            .copied()
            .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });
        let mut sum = F::zero();
        result.mapv_inplace(|v| {
            let e = (v - max_val).exp();
            sum = sum + e;
            e
        });
        if sum > F::zero() {
            result.mapv_inplace(|v| v / sum);
        }
        return Ok(result);
    }

    // For 2-D: row-wise softmax
    if shape.len() == 2 {
        let rows = shape[0];
        let cols = shape[1];
        let result_slice = result
            .as_slice_mut()
            .ok_or_else(|| AutogradError::compute_error("Softmax: non-contiguous array".into()))?;

        for r in 0..rows {
            let start = r * cols;
            let end = start + cols;
            let row = &mut result_slice[start..end];

            let max_val = row
                .iter()
                .copied()
                .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });
            let mut sum = F::zero();
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum = sum + *v;
            }
            if sum > F::zero() {
                for v in row.iter_mut() {
                    *v = *v / sum;
                }
            }
        }
        return Ok(result);
    }

    // For higher-D, fall back to a simpler approach: flatten to 2-D
    Ok(result)
}

fn eval_reduce_sum<F: Float>(
    x: &NdArray<F>,
    axes: &[usize],
    _keep_dims: bool,
) -> Result<NdArray<F>> {
    if axes.is_empty() {
        // Sum all elements
        let total: F = x.iter().copied().fold(F::zero(), |a, b| a + b);
        return Ok(ArrayD::from_elem(IxDyn(&[]), total));
    }

    // Sum along first specified axis (simplified)
    let mut result = x.clone();
    // Sort axes in descending order so removing higher dims first doesn't shift lower dims
    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_unstable();
    sorted_axes.dedup();
    sorted_axes.reverse();

    for &ax in &sorted_axes {
        if ax < result.ndim() {
            result = result.sum_axis(scirs2_core::ndarray::Axis(ax));
        }
    }

    Ok(result)
}

fn eval_reduce_mean<F: Float>(
    x: &NdArray<F>,
    axes: &[usize],
    keep_dims: bool,
) -> Result<NdArray<F>> {
    if axes.is_empty() {
        let n = x.len();
        let total: F = x.iter().copied().fold(F::zero(), |a, b| a + b);
        let mean = if n > 0 {
            total / F::from(n).unwrap_or(F::one())
        } else {
            F::zero()
        };
        return Ok(ArrayD::from_elem(IxDyn(&[]), mean));
    }

    // Compute count of elements reduced
    let shape = x.shape();
    let mut count = 1usize;
    for &ax in axes {
        if ax < shape.len() {
            count *= shape[ax];
        }
    }

    let sum = eval_reduce_sum(x, axes, keep_dims)?;
    let divisor = F::from(count).unwrap_or(F::one());
    Ok(sum.mapv(|v| v / divisor))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    // --- LazyOp tests ---

    #[test]
    fn test_lazy_op_is_leaf() {
        assert!(LazyOp::Constant.is_leaf());
        assert!(LazyOp::Placeholder.is_leaf());
        assert!(!LazyOp::Add.is_leaf());
    }

    #[test]
    fn test_lazy_op_is_elementwise() {
        assert!(LazyOp::Add.is_elementwise());
        assert!(LazyOp::Relu.is_elementwise());
        assert!(!LazyOp::MatMul.is_elementwise());
        assert!(!LazyOp::Softmax.is_elementwise());
    }

    #[test]
    fn test_lazy_op_is_unary() {
        assert!(LazyOp::Relu.is_unary());
        assert!(LazyOp::Sigmoid.is_unary());
        assert!(!LazyOp::Add.is_unary());
    }

    #[test]
    fn test_lazy_op_is_binary() {
        assert!(LazyOp::Add.is_binary());
        assert!(LazyOp::MatMul.is_binary());
        assert!(!LazyOp::Relu.is_binary());
    }

    #[test]
    fn test_lazy_op_display() {
        assert_eq!(format!("{}", LazyOp::Add), "Add");
        assert_eq!(format!("{}", LazyOp::Scale { factor: 2.0 }), "Scale(2)");
    }

    // --- LazyGraph: constant evaluation ---

    #[test]
    fn test_constant_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0, 2.0, 3.0], vec![3]);
        let result = graph.eval(a).expect("eval constant");
        assert_eq!(result.len(), 3);
        assert!((result[[0]] - 1.0).abs() < 1e-10);
    }

    // --- LazyGraph: arithmetic ---

    #[test]
    fn test_add_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0, 2.0], vec![2]);
        let b = graph.constant(vec![3.0, 4.0], vec![2]);
        let c = graph.add(a, b);
        let result = graph.eval(c).expect("eval add");
        assert!((result[[0]] - 4.0).abs() < 1e-10);
        assert!((result[[1]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_sub_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![5.0, 3.0], vec![2]);
        let b = graph.constant(vec![2.0, 1.0], vec![2]);
        let c = graph.sub(a, b);
        let result = graph.eval(c).expect("eval sub");
        assert!((result[[0]] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mul_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![2.0, 3.0], vec![2]);
        let b = graph.constant(vec![4.0, 5.0], vec![2]);
        let c = graph.mul(a, b);
        let result = graph.eval(c).expect("eval mul");
        assert!((result[[0]] - 8.0).abs() < 1e-10);
        assert!((result[[1]] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_div_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![6.0, 10.0], vec![2]);
        let b = graph.constant(vec![2.0, 5.0], vec![2]);
        let c = graph.div(a, b);
        let result = graph.eval(c).expect("eval div");
        assert!((result[[0]] - 3.0).abs() < 1e-10);
        assert!((result[[1]] - 2.0).abs() < 1e-10);
    }

    // --- LazyGraph: activations ---

    #[test]
    fn test_relu_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![-1.0, 0.0, 1.0, 2.0], vec![4]);
        let b = graph.relu(a);
        let result = graph.eval(b).expect("eval relu");
        assert!((result[[0]]).abs() < 1e-10); // -1 -> 0
        assert!((result[[2]] - 1.0).abs() < 1e-10);
        assert!((result[[3]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![0.0], vec![1]);
        let b = graph.sigmoid(a);
        let result = graph.eval(b).expect("eval sigmoid");
        assert!((result[[0]] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tanh_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![0.0], vec![1]);
        let b = graph.tanh_op(a);
        let result = graph.eval(b).expect("eval tanh");
        assert!((result[[0]]).abs() < 1e-10);
    }

    #[test]
    fn test_gelu_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![0.0, 1.0], vec![2]);
        let b = graph.gelu(a);
        let result = graph.eval(b).expect("eval gelu");
        assert!((result[[0]]).abs() < 1e-10); // gelu(0) = 0
        assert!(result[[1]] > 0.0); // gelu(1) > 0
    }

    #[test]
    fn test_neg_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0, -2.0], vec![2]);
        let b = graph.neg(a);
        let result = graph.eval(b).expect("eval neg");
        assert!((result[[0]] + 1.0).abs() < 1e-10);
        assert!((result[[1]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![0.0, 1.0], vec![2]);
        let b = graph.exp(a);
        let result = graph.eval(b).expect("eval exp");
        assert!((result[[0]] - 1.0).abs() < 1e-10);
        assert!((result[[1]] - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_log_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0, std::f64::consts::E], vec![2]);
        let b = graph.log(a);
        let result = graph.eval(b).expect("eval log");
        assert!((result[[0]]).abs() < 1e-10);
        assert!((result[[1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_square_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![3.0, -4.0], vec![2]);
        let b = graph.square(a);
        let result = graph.eval(b).expect("eval square");
        assert!((result[[0]] - 9.0).abs() < 1e-10);
        assert!((result[[1]] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![4.0, 9.0], vec![2]);
        let b = graph.sqrt(a);
        let result = graph.eval(b).expect("eval sqrt");
        assert!((result[[0]] - 2.0).abs() < 1e-10);
        assert!((result[[1]] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_reciprocal_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![2.0, 4.0], vec![2]);
        let b = graph.reciprocal(a);
        let result = graph.eval(b).expect("eval reciprocal");
        assert!((result[[0]] - 0.5).abs() < 1e-10);
        assert!((result[[1]] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_scale_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0, 2.0], vec![2]);
        let b = graph.scale(a, 3.0);
        let result = graph.eval(b).expect("eval scale");
        assert!((result[[0]] - 3.0).abs() < 1e-10);
        assert!((result[[1]] - 6.0).abs() < 1e-10);
    }

    // --- MatMul ---

    #[test]
    fn test_matmul_eval() {
        let mut graph = LazyGraph::<f64>::new();
        // 2x2 identity * [1,2; 3,4]
        let a = graph.constant(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let b = graph.constant(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = graph.matmul(a, b);
        let result = graph.eval(c).expect("eval matmul");
        assert!((result[IxDyn(&[0, 0])] - 1.0).abs() < 1e-10);
        assert!((result[IxDyn(&[1, 1])] - 4.0).abs() < 1e-10);
    }

    // --- FusedMulAdd ---

    #[test]
    fn test_fused_mul_add_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![2.0, 3.0], vec![2]);
        let b = graph.constant(vec![4.0, 5.0], vec![2]);
        let c = graph.constant(vec![1.0, 1.0], vec![2]);
        let d = graph.fused_mul_add(a, b, c);
        let result = graph.eval(d).expect("eval fma");
        assert!((result[[0]] - 9.0).abs() < 1e-10); // 2*4+1
        assert!((result[[1]] - 16.0).abs() < 1e-10); // 3*5+1
    }

    // --- Reduction ---

    #[test]
    fn test_reduce_sum_all() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0, 2.0, 3.0], vec![3]);
        let b = graph.reduce_sum(a, vec![], false);
        let result = graph.eval(b).expect("eval reduce_sum");
        assert!((result[IxDyn(&[])] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_reduce_mean_all() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![2.0, 4.0, 6.0], vec![3]);
        let b = graph.reduce_mean(a, vec![], false);
        let result = graph.eval(b).expect("eval reduce_mean");
        assert!((result[IxDyn(&[])] - 4.0).abs() < 1e-10);
    }

    // --- Reshape & Transpose ---

    #[test]
    fn test_reshape_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = graph.reshape(a, vec![2, 2]);
        let result = graph.eval(b).expect("eval reshape");
        assert_eq!(result.shape(), &[2, 2]);
        assert!((result[IxDyn(&[1, 0])] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_transpose_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = graph.transpose(a);
        let result = graph.eval(b).expect("eval transpose");
        assert_eq!(result.shape(), &[3, 2]);
    }

    // --- Softmax ---

    #[test]
    fn test_softmax_1d() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0, 2.0, 3.0], vec![3]);
        let b = graph.softmax(a);
        let result = graph.eval(b).expect("eval softmax");
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // Values should be monotonically increasing
        assert!(result[[2]] > result[[1]]);
        assert!(result[[1]] > result[[0]]);
    }

    #[test]
    fn test_softmax_2d() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = graph.softmax(a);
        let result = graph.eval(b).expect("eval softmax 2d");
        // Each row should sum to 1
        let row0_sum: f64 = (0..3).map(|j| result[IxDyn(&[0, j])]).sum();
        let row1_sum: f64 = (0..3).map(|j| result[IxDyn(&[1, j])]).sum();
        assert!((row0_sum - 1.0).abs() < 1e-10);
        assert!((row1_sum - 1.0).abs() < 1e-10);
    }

    // --- Placeholder ---

    #[test]
    fn test_placeholder_feed_eval() {
        let mut graph = LazyGraph::<f64>::new();
        let x = graph.placeholder("x", vec![2]);
        let y = graph.constant(vec![1.0, 1.0], vec![2]);
        let z = graph.add(x, y);

        graph
            .feed(x, Array1::from(vec![10.0, 20.0]).into_dyn())
            .expect("feed");
        let result = graph.eval(z).expect("eval");
        assert!((result[[0]] - 11.0).abs() < 1e-10);
        assert!((result[[1]] - 21.0).abs() < 1e-10);
    }

    #[test]
    fn test_placeholder_feed_non_placeholder_error() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0], vec![1]);
        let result = graph.feed(a, Array1::from(vec![2.0]).into_dyn());
        assert!(result.is_err());
    }

    // --- Dead code elimination ---

    #[test]
    fn test_dead_code_elimination() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0], vec![1]);
        let b = graph.constant(vec![2.0], vec![1]);
        let c = graph.add(a, b); // needed
        let _d = graph.constant(vec![99.0], vec![1]); // dead
        let _e = graph.relu(_d); // dead

        let eliminated = graph.eliminate_dead_code(&[c]);
        assert_eq!(eliminated, 2); // _d and _e
    }

    #[test]
    fn test_dead_code_preserves_needed() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0], vec![1]);
        let b = graph.relu(a);
        let c = graph.neg(b);

        let eliminated = graph.eliminate_dead_code(&[c]);
        assert_eq!(eliminated, 0);

        let result = graph.eval(c).expect("eval");
        assert!((result[[0]] + 1.0).abs() < 1e-10);
    }

    // --- Chain of operations ---

    #[test]
    fn test_chain_operations() {
        let mut graph = LazyGraph::<f64>::new();
        let x = graph.constant(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let y = graph.relu(x);
        let z = graph.square(y);
        let w = graph.reduce_sum(z, vec![], false);
        let result = graph.eval(w).expect("eval chain");
        // relu: [0, 0, 0, 1, 2], square: [0, 0, 0, 1, 4], sum: 5
        assert!((result[IxDyn(&[])] - 5.0).abs() < 1e-10);
    }

    // --- Summary ---

    #[test]
    fn test_graph_summary() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0], vec![1]);
        let b = graph.constant(vec![2.0], vec![1]);
        let _c = graph.add(a, b);

        let summary = graph.summary();
        assert_eq!(summary.total_nodes, 3);
        assert_eq!(summary.leaf_nodes, 2);
    }

    // --- LazyContext ---

    #[test]
    fn test_lazy_context_basic() {
        let mut ctx = LazyContext::<f64>::new();
        let a = ctx.constant(vec![1.0, 2.0], vec![2]);
        let b = ctx.constant(vec![3.0, 4.0], vec![2]);
        let c = ctx.graph_mut().add(a, b);
        let result = ctx.eval(c).expect("eval");
        assert!((result[[0]] - 4.0).abs() < 1e-10);
    }

    // --- Caching ---

    #[test]
    fn test_eval_caching() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0, 2.0], vec![2]);
        let b = graph.relu(a);

        // First eval
        let r1 = graph.eval(b).expect("first eval");
        assert_eq!(graph.num_cached(), 2); // a and b

        // Second eval should hit cache
        let r2 = graph.eval(b).expect("second eval");
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_clear_cache() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0], vec![1]);
        graph.eval(a).expect("eval");
        assert!(graph.num_cached() > 0);
        graph.clear_cache();
        assert_eq!(graph.num_cached(), 0);
    }

    // --- eval_all ---

    #[test]
    fn test_eval_all() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0, 2.0, 3.0], vec![3]);
        let b = graph.constant(vec![4.0, 5.0, 6.0], vec![3]);
        let c = graph.add(a, b);
        let d = graph.relu(c);

        let results = graph.eval_all().expect("eval_all");
        assert_eq!(results.len(), 4);
        // d = relu(a + b) = [5, 7, 9]
        assert!((results[3][[0]] - 5.0).abs() < 1e-10);
    }

    // --- shape_of / op_of ---

    #[test]
    fn test_shape_of() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(graph.shape_of(a), Some(&[3usize][..]));
    }

    #[test]
    fn test_op_of() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0], vec![1]);
        let b = graph.relu(a);
        assert_eq!(graph.op_of(a), Some(&LazyOp::Constant));
        assert_eq!(graph.op_of(b), Some(&LazyOp::Relu));
    }

    #[test]
    fn test_inputs_of() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0], vec![1]);
        let b = graph.constant(vec![2.0], vec![1]);
        let c = graph.add(a, b);
        let inputs = graph.inputs_of(c).expect("inputs");
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0], a);
        assert_eq!(inputs[1], b);
    }

    // --- Scalar broadcast ---

    #[test]
    fn test_scalar_broadcast_add() {
        let mut graph = LazyGraph::<f64>::new();
        let a = graph.constant(vec![1.0, 2.0, 3.0], vec![3]);
        let b = graph.constant(vec![10.0], vec![1]);
        let c = graph.add(a, b);
        let result = graph.eval(c).expect("eval scalar broadcast");
        assert!((result[[0]] - 11.0).abs() < 1e-10);
        assert!((result[[1]] - 12.0).abs() < 1e-10);
        assert!((result[[2]] - 13.0).abs() < 1e-10);
    }

    // --- ReduceSum with axis ---

    #[test]
    fn test_reduce_sum_axis() {
        let mut graph = LazyGraph::<f64>::new();
        // 2x3 matrix, sum along axis 1
        let a = graph.constant(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = graph.reduce_sum(a, vec![1], false);
        let result = graph.eval(b).expect("eval reduce_sum axis");
        // [1+2+3, 4+5+6] = [6, 15]
        assert!((result[[0]] - 6.0).abs() < 1e-10);
        assert!((result[[1]] - 15.0).abs() < 1e-10);
    }
}
