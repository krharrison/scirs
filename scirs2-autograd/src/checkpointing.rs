//! Gradient checkpointing — selective recomputation of activations
//!
//! This module implements gradient checkpointing, a technique that trades
//! additional computation for reduced peak memory during backpropagation.
//! Instead of storing all intermediate activations, only a subset (the
//! *checkpoints*) is retained; non-checkpointed activations are recomputed
//! on demand during the backward pass.
//!
//! # Background
//!
//! For a feed-forward network with `L` layers, naïve backpropagation stores
//! all `L` activation tensors simultaneously, leading to O(L) peak memory.
//! Gradient checkpointing reduces this to O(√L) memory at the cost of O(√L)
//! extra forward passes (Chen et al., 2016 — "Training Deep Nets with
//! Sublinear Memory Cost").
//!
//! The DTR (Dynamic Tensor Rematerialisation) approach goes further by
//! choosing which tensors to evict and recompute at runtime based on cost
//! and benefit heuristics.
//!
//! # Structures
//!
//! | Type | Description |
//! |------|-------------|
//! | [`CheckpointedGraph`] | Computation graph with explicit checkpoint sets |
//! | [`MemoryBudget`] | Memory budget specification |
//! | [`CheckpointSchedule`] | Result of [`optimal_checkpointing_schedule`] |
//! | [`MemoryComputeTradeoff`] | Analysis of the memory–compute tradeoff |
//!
//! # Functions
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`checkpoint`] | Mark a segment of computation as recompute-on-backward |
//! | [`optimal_checkpointing_schedule`] | DTR-style optimal checkpoint selection |
//! | [`analyse_tradeoff`] | Compute the memory–compute Pareto frontier |
//! | [`sqrt_schedule`] | √n checkpointing schedule |
//! | [`uniform_schedule`] | Uniformly-spaced checkpointing schedule |
//! | [`binomial_schedule`] | Binomial (Revolve) schedule for given memory budget |

use crate::error::AutogradError;
use std::collections::BTreeMap;

// ─────────────────────────────────────────────────────────────────────────────
// Node metadata
// ─────────────────────────────────────────────────────────────────────────────

/// Metadata for a single node (layer / operation) in a sequential computation.
#[derive(Debug, Clone)]
pub struct NodeMeta {
    /// Unique node index (0-based, topological order)
    pub index: usize,
    /// Estimated size of the activation tensor in abstract units
    pub activation_size: usize,
    /// Estimated FLOP cost of the forward pass of this node
    pub forward_cost: f64,
    /// Human-readable name (optional)
    pub name: Option<String>,
}

impl NodeMeta {
    /// Construct a new node with the given index and costs.
    pub fn new(index: usize, activation_size: usize, forward_cost: f64) -> Self {
        Self {
            index,
            activation_size,
            forward_cost,
            name: None,
        }
    }

    /// Attach a human-readable name to the node.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CheckpointedGraph
// ─────────────────────────────────────────────────────────────────────────────

/// A sequential computation graph annotated with checkpoint information.
///
/// The graph is represented as an ordered sequence of [`NodeMeta`] entries.
/// The `checkpointed` set records which node indices have their activations
/// stored; all others must be recomputed during the backward pass.
///
/// # Example
/// ```rust
/// use scirs2_autograd::checkpointing::{CheckpointedGraph, NodeMeta};
///
/// let mut g = CheckpointedGraph::new();
/// g.add_node(NodeMeta::new(0, 100, 1.0).with_name("embed"));
/// g.add_node(NodeMeta::new(1, 200, 2.0).with_name("layer1"));
/// g.add_node(NodeMeta::new(2, 200, 2.0).with_name("layer2"));
/// g.checkpoint_node(0);
/// g.checkpoint_node(2);
///
/// assert_eq!(g.num_nodes(), 3);
/// assert_eq!(g.checkpoint_count(), 2);
/// ```
#[derive(Debug, Clone, Default)]
pub struct CheckpointedGraph {
    nodes: Vec<NodeMeta>,
    checkpointed: std::collections::BTreeSet<usize>,
}

impl CheckpointedGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node (must be added in topological order, index must equal current length).
    pub fn add_node(&mut self, meta: NodeMeta) {
        self.nodes.push(meta);
    }

    /// Mark node `index` as checkpointed (activation is stored).
    pub fn checkpoint_node(&mut self, index: usize) {
        self.checkpointed.insert(index);
    }

    /// Remove a checkpoint from node `index`.
    pub fn remove_checkpoint(&mut self, index: usize) {
        self.checkpointed.remove(&index);
    }

    /// Return the number of nodes in the graph.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of checkpointed nodes.
    pub fn checkpoint_count(&self) -> usize {
        self.checkpointed.len()
    }

    /// Return `true` if node `index` is checkpointed.
    pub fn is_checkpointed(&self, index: usize) -> bool {
        self.checkpointed.contains(&index)
    }

    /// Iterator over all nodes in topological order.
    pub fn nodes(&self) -> impl Iterator<Item = &NodeMeta> {
        self.nodes.iter()
    }

    /// Return the set of checkpointed node indices.
    pub fn checkpointed_indices(&self) -> &std::collections::BTreeSet<usize> {
        &self.checkpointed
    }

    /// Compute the peak memory cost of the current checkpoint configuration.
    ///
    /// During the backward pass, the maximum simultaneously live memory is the
    /// sum of activations of all checkpointed nodes plus the activations of nodes
    /// that must be recomputed for the current backward segment.
    ///
    /// For a simple sequential model the peak is: max over each segment
    /// `[c_i, c_{i+1})` of `activation(c_i) + sum_{j in segment} activation(j)`.
    pub fn peak_memory(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }

        // Build list of checkpoint positions in order
        let cp: Vec<usize> = self.checkpointed.iter().copied().collect();
        if cp.is_empty() {
            // No checkpoints: must store everything
            return self.nodes.iter().map(|n| n.activation_size).sum();
        }

        // Sum of checkpointed activations (all stored simultaneously)
        let checkpoint_mem: usize = cp
            .iter()
            .filter_map(|&i| self.nodes.get(i))
            .map(|n| n.activation_size)
            .sum();

        // For each segment between consecutive checkpoints, we must recompute
        // the segment.  During recomputation of segment [c_i, c_{i+1}), the
        // nodes c_i through the current recomputed node are alive.
        let n = self.nodes.len();
        let mut max_segment_mem = 0usize;

        // Segments: before first cp, between cps, after last cp
        let mut boundaries: Vec<(usize, usize)> = Vec::new();
        if cp[0] > 0 {
            boundaries.push((0, cp[0]));
        }
        for w in cp.windows(2) {
            if w[1] > w[0] + 1 {
                boundaries.push((w[0] + 1, w[1]));
            }
        }
        if let Some(&last_cp) = cp.last() {
            if last_cp + 1 < n {
                boundaries.push((last_cp + 1, n));
            }
        }

        for (start, end) in boundaries {
            let seg_mem: usize = self.nodes[start..end].iter().map(|nd| nd.activation_size).sum();
            if seg_mem > max_segment_mem {
                max_segment_mem = seg_mem;
            }
        }

        checkpoint_mem + max_segment_mem
    }

    /// Compute the total recomputation cost (in forward-pass FLOPs).
    ///
    /// Each non-checkpointed segment must be recomputed once during the backward
    /// pass.
    pub fn recomputation_cost(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        let mut total = 0.0f64;
        for node in &self.nodes {
            if !self.checkpointed.contains(&node.index) {
                total += node.forward_cost;
            }
        }
        total
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// checkpoint() — functional API (operates on abstract segments)
// ─────────────────────────────────────────────────────────────────────────────

/// Mark a computation segment for recomputation during the backward pass.
///
/// This is the functional-style entry point analogous to PyTorch's
/// `torch.utils.checkpoint.checkpoint`.  It accepts:
/// - A list of "input tensor" identifiers (node indices) that should be
///   stored as checkpoints at the segment boundary.
/// - A computation closure `f` that performs the forward pass.
///
/// The function records which inputs should be kept and returns a
/// [`CheckpointRecord`] that can be used to trigger recomputation.
///
/// # Arguments
/// * `checkpoint_inputs` – Indices of tensors to store at the boundary
/// * `f`                 – Forward computation (returns an output tensor index)
///
/// # Returns
/// A [`CheckpointRecord`] describing the segment.
///
/// # Example
/// ```rust
/// use scirs2_autograd::checkpointing::checkpoint;
///
/// let record = checkpoint(vec![0, 5, 10], || 15);
/// assert_eq!(record.stored_inputs, vec![0, 5, 10]);
/// assert_eq!(record.output_index, 15);
/// ```
pub fn checkpoint(checkpoint_inputs: Vec<usize>, f: impl FnOnce() -> usize) -> CheckpointRecord {
    let output_index = f();
    CheckpointRecord {
        stored_inputs: checkpoint_inputs,
        output_index,
    }
}

/// Record produced by [`checkpoint`].
#[derive(Debug, Clone)]
pub struct CheckpointRecord {
    /// Tensor indices that are stored at the segment boundary.
    pub stored_inputs: Vec<usize>,
    /// Index of the output tensor produced by the segment.
    pub output_index: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory budget
// ─────────────────────────────────────────────────────────────────────────────

/// Memory budget specification for checkpoint scheduling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryBudget {
    /// Absolute budget in the same units as `NodeMeta::activation_size`.
    Absolute(usize),
    /// Fraction of the total activation memory (0.0 – 1.0).
    Fraction(f64),
    /// Maximum number of checkpointed tensors (slots).
    Slots(usize),
}

// ─────────────────────────────────────────────────────────────────────────────
// CheckpointSchedule
// ─────────────────────────────────────────────────────────────────────────────

/// A checkpoint schedule: which node indices to checkpoint.
#[derive(Debug, Clone)]
pub struct CheckpointSchedule {
    /// Total number of nodes
    pub num_nodes: usize,
    /// Set of node indices to checkpoint
    pub checkpoints: Vec<usize>,
    /// Estimated peak memory under this schedule
    pub estimated_peak_memory: usize,
    /// Estimated total recomputation cost
    pub estimated_recomputation_cost: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory–compute tradeoff
// ─────────────────────────────────────────────────────────────────────────────

/// A point on the memory–compute Pareto frontier.
#[derive(Debug, Clone)]
pub struct TradeoffPoint {
    /// Number of checkpoint slots used
    pub slots: usize,
    /// Estimated peak memory
    pub peak_memory: usize,
    /// Estimated recomputation overhead (multiple of baseline forward pass)
    pub recomputation_overhead: f64,
}

/// Result of [`analyse_tradeoff`]: the Pareto frontier of memory vs. compute.
#[derive(Debug, Clone)]
pub struct MemoryComputeTradeoff {
    /// Points on the Pareto frontier, sorted by ascending memory.
    pub pareto_points: Vec<TradeoffPoint>,
    /// Total activation memory if no checkpointing is used.
    pub baseline_memory: usize,
    /// Total forward-pass cost (used to normalise recomputation overhead).
    pub baseline_forward_cost: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Scheduling algorithms
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a √n checkpointing schedule for a uniform-cost graph.
///
/// Places checkpoints every `ceil(√n)` layers.  For a graph of `n` layers this
/// gives O(√n) checkpoints, O(√n) memory, and O(√n) recomputation per
/// backward pass.
///
/// # Arguments
/// * `graph` – The computation graph
///
/// # Returns
/// [`CheckpointSchedule`] with √n checkpoint spacing.
///
/// # Example
/// ```rust
/// use scirs2_autograd::checkpointing::{CheckpointedGraph, NodeMeta, sqrt_schedule};
///
/// let mut g = CheckpointedGraph::new();
/// for i in 0..9 {
///     g.add_node(NodeMeta::new(i, 10, 1.0));
/// }
/// let sched = sqrt_schedule(&g);
/// // ceil(sqrt(9)) = 3; checkpoints at 0, 3, 6, (8)
/// assert!(sched.checkpoints.contains(&0));
/// assert!(sched.checkpoints.contains(&3));
/// ```
pub fn sqrt_schedule(graph: &CheckpointedGraph) -> CheckpointSchedule {
    let n = graph.num_nodes();
    if n == 0 {
        return CheckpointSchedule {
            num_nodes: 0,
            checkpoints: Vec::new(),
            estimated_peak_memory: 0,
            estimated_recomputation_cost: 0.0,
        };
    }

    let k = ((n as f64).sqrt().ceil() as usize).max(1);
    let mut checkpoints = Vec::new();
    let mut i = 0;
    while i < n {
        checkpoints.push(i);
        i += k;
    }
    // Always include last node
    if checkpoints.last().copied() != Some(n - 1) {
        checkpoints.push(n - 1);
    }

    build_schedule(graph, checkpoints, n)
}

/// Compute a uniformly-spaced checkpointing schedule.
///
/// Places checkpoints at positions `0, interval, 2*interval, …`.
///
/// # Arguments
/// * `graph`    – The computation graph
/// * `interval` – Spacing between checkpoints
///
/// # Errors
/// Returns `AutogradError` if `interval == 0`.
pub fn uniform_schedule(
    graph: &CheckpointedGraph,
    interval: usize,
) -> Result<CheckpointSchedule, AutogradError> {
    if interval == 0 {
        return Err(AutogradError::OperationError(
            "uniform_schedule: interval must be > 0".to_string(),
        ));
    }
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(CheckpointSchedule {
            num_nodes: 0,
            checkpoints: Vec::new(),
            estimated_peak_memory: 0,
            estimated_recomputation_cost: 0.0,
        });
    }

    let mut checkpoints = Vec::new();
    let mut i = 0;
    while i < n {
        checkpoints.push(i);
        i += interval;
    }
    if checkpoints.last().copied() != Some(n - 1) {
        checkpoints.push(n - 1);
    }

    Ok(build_schedule(graph, checkpoints, n))
}

/// Compute a binomial (Revolve-style) checkpointing schedule.
///
/// The binomial strategy distributes checkpoints so that the maximum
/// recomputation for any backward segment is minimised given a slot budget.
/// This is the algorithm underlying Chen et al.'s "Training Deep Nets with
/// Sublinear Memory Cost" and Griewank's Revolve algorithm.
///
/// For a chain of `n` nodes and `b` checkpoint slots, the optimal
/// placement minimises the maximum segment length, giving segments of
/// size approximately `n / b`.
///
/// # Arguments
/// * `graph`  – The computation graph
/// * `budget` – Memory budget specification
///
/// # Errors
/// Returns `AutogradError` if the budget is invalid (e.g. fraction > 1.0).
///
/// # Example
/// ```rust
/// use scirs2_autograd::checkpointing::{
///     CheckpointedGraph, NodeMeta, binomial_schedule, MemoryBudget,
/// };
///
/// let mut g = CheckpointedGraph::new();
/// for i in 0..16 {
///     g.add_node(NodeMeta::new(i, 1, 1.0));
/// }
/// // Allow 4 checkpoint slots → segments of size 4
/// let sched = binomial_schedule(&g, MemoryBudget::Slots(4)).expect("sched");
/// assert!(sched.checkpoints.len() <= 4 + 1); // at most 4 + last
/// ```
pub fn binomial_schedule(
    graph: &CheckpointedGraph,
    budget: MemoryBudget,
) -> Result<CheckpointSchedule, AutogradError> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(CheckpointSchedule {
            num_nodes: 0,
            checkpoints: Vec::new(),
            estimated_peak_memory: 0,
            estimated_recomputation_cost: 0.0,
        });
    }

    let total_mem: usize = graph.nodes().map(|nd| nd.activation_size).sum();
    let total_cost: f64 = graph.nodes().map(|nd| nd.forward_cost).sum();

    let slots = match budget {
        MemoryBudget::Slots(s) => {
            if s == 0 {
                return Err(AutogradError::OperationError(
                    "binomial_schedule: slot budget must be > 0".to_string(),
                ));
            }
            s
        }
        MemoryBudget::Absolute(mem) => {
            if mem == 0 {
                return Err(AutogradError::OperationError(
                    "binomial_schedule: memory budget must be > 0".to_string(),
                ));
            }
            // Estimate how many slots fit: at minimum 1
            let avg_activation = total_mem / n;
            if avg_activation == 0 {
                n
            } else {
                (mem / avg_activation).max(1)
            }
        }
        MemoryBudget::Fraction(f) => {
            if !(0.0..=1.0).contains(&f) {
                return Err(AutogradError::OperationError(format!(
                    "binomial_schedule: fraction {} must be in [0.0, 1.0]",
                    f
                )));
            }
            let target = (total_mem as f64 * f) as usize;
            let avg_activation = total_mem / n;
            if avg_activation == 0 {
                n
            } else {
                (target / avg_activation).max(1)
            }
        }
    };

    // Place `slots` checkpoints at positions that minimise the maximum segment.
    // Optimal strategy: evenly spaced.  For non-uniform costs, we use a
    // greedy balanced approach: pick positions that equalise cumulative cost.
    let checkpoints = if slots >= n {
        // All nodes are checkpointed
        (0..n).collect::<Vec<_>>()
    } else {
        // Place checkpoints at evenly-spaced cost intervals
        let costs: Vec<f64> = graph.nodes().map(|nd| nd.forward_cost).collect();
        let total_cost_pos: f64 = costs.iter().sum();
        let target_interval = total_cost_pos / (slots as f64 + 1.0);

        let mut cps = vec![0usize]; // Always include first
        let mut cumulative = 0.0f64;
        let mut next_target = target_interval;

        for (i, &c) in costs.iter().enumerate() {
            cumulative += c;
            if cumulative >= next_target && cps.last().copied() != Some(i) {
                cps.push(i);
                next_target += target_interval;
                if cps.len() >= slots {
                    break;
                }
            }
        }
        // Always include last
        if cps.last().copied() != Some(n - 1) {
            cps.push(n - 1);
        }
        cps
    };

    let _ = total_cost; // used above, suppress unused warning
    Ok(build_schedule(graph, checkpoints, n))
}

/// Compute the optimal checkpointing schedule for a given memory budget.
///
/// This uses a dynamic programming approach (DTR-style) to find the set of
/// checkpoint positions that minimises total recomputation cost subject to
/// the peak memory constraint.
///
/// # Arguments
/// * `graph`         – The computation graph
/// * `memory_budget` – Memory budget (see [`MemoryBudget`])
///
/// # Returns
/// Optimal [`CheckpointSchedule`] for the given budget.
///
/// # Errors
/// Returns `AutogradError` if the budget specification is invalid.
///
/// # Complexity
/// O(n² · b) where `n` is the number of nodes and `b` is the slot budget.
///
/// # Example
/// ```rust
/// use scirs2_autograd::checkpointing::{
///     CheckpointedGraph, NodeMeta, optimal_checkpointing_schedule, MemoryBudget,
/// };
///
/// let mut g = CheckpointedGraph::new();
/// for i in 0..8 {
///     g.add_node(NodeMeta::new(i, 10, 1.0));
/// }
/// let sched = optimal_checkpointing_schedule(&g, MemoryBudget::Slots(3))
///     .expect("optimal schedule");
/// assert!(sched.checkpoints.len() <= 4);
/// ```
pub fn optimal_checkpointing_schedule(
    graph: &CheckpointedGraph,
    memory_budget: MemoryBudget,
) -> Result<CheckpointSchedule, AutogradError> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(CheckpointSchedule {
            num_nodes: 0,
            checkpoints: Vec::new(),
            estimated_peak_memory: 0,
            estimated_recomputation_cost: 0.0,
        });
    }

    let total_mem: usize = graph.nodes().map(|nd| nd.activation_size).sum();

    // Convert budget to a maximum number of checkpoint slots
    let max_slots = match memory_budget {
        MemoryBudget::Slots(s) => {
            if s == 0 {
                return Err(AutogradError::OperationError(
                    "optimal_checkpointing_schedule: slot budget must be > 0".to_string(),
                ));
            }
            s.min(n)
        }
        MemoryBudget::Absolute(mem) => {
            if mem == 0 {
                return Err(AutogradError::OperationError(
                    "optimal_checkpointing_schedule: memory budget must be > 0".to_string(),
                ));
            }
            let avg = if n > 0 { total_mem / n } else { 1 };
            let avg = avg.max(1);
            (mem / avg).max(1).min(n)
        }
        MemoryBudget::Fraction(f) => {
            if !(0.0..=1.0).contains(&f) {
                return Err(AutogradError::OperationError(format!(
                    "optimal_checkpointing_schedule: fraction {} must be in [0.0, 1.0]",
                    f
                )));
            }
            let target = (total_mem as f64 * f) as usize;
            let avg = if n > 0 { total_mem / n } else { 1 };
            let avg = avg.max(1);
            (target / avg).max(1).min(n)
        }
    };

    // DP: dp[i][k] = minimum recomputation cost to handle nodes 0..i with k checkpoint slots.
    // Transition: for each possible last checkpoint position j < i,
    //   dp[i][k] = min over j of (dp[j][k-1] + recompute_cost(j+1..i))
    //
    // We track the path for reconstruction.

    let costs: Vec<f64> = graph.nodes().map(|nd| nd.forward_cost).collect();

    // Precompute prefix sums of costs
    let mut prefix = vec![0.0f64; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + costs[i];
    }
    // recompute_cost(a..b) = prefix[b] - prefix[a]
    let seg_cost = |a: usize, b: usize| -> f64 { prefix[b] - prefix[a] };

    // dp[i][k]: min recomputation cost handling first i+1 nodes with k slots
    let inf = f64::INFINITY;
    // We use BTreeMap for sparse storage to avoid huge allocations
    let mut dp: Vec<Vec<f64>> = vec![vec![inf; max_slots + 1]; n];
    // prev[i][k]: last checkpoint index chosen
    let mut prev: Vec<Vec<Option<usize>>> = vec![vec![None; max_slots + 1]; n];

    // Base: node 0 with 1 slot (checkpoint node 0 itself)
    for k in 1..=max_slots {
        dp[0][k] = 0.0; // node 0 is the first checkpoint, no recomputation needed
    }

    // Fill DP
    for i in 1..n {
        for k in 1..=max_slots {
            // Option A: node i is NOT a checkpoint — recompute from last checkpoint j=i-1
            // (only valid if k slots were used for nodes 0..i-1)
            // This is captured by the segment recomputation cost below.

            // Option B: node i IS a checkpoint — find best j for nodes 0..j with k-1 slots
            // Then recompute_cost(j+1..i) added.
            for j in 0..i {
                if dp[j][k - 1] < inf {
                    let cost = dp[j][k - 1] + seg_cost(j + 1, i);
                    if cost < dp[i][k] {
                        dp[i][k] = cost;
                        prev[i][k] = Some(j);
                    }
                }
            }
            // Also allow k slots from j with no intermediate checkpoints
            // (covered by the loop above starting at j=0)
        }
    }

    // Find the optimal k for the last node
    let last = n - 1;
    let (best_k, best_cost) = (1..=max_slots)
        .map(|k| (k, dp[last][k]))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((1, inf));

    if best_cost == inf {
        // Fallback: return sqrt schedule
        return Ok(sqrt_schedule(graph));
    }

    // Reconstruct checkpoint positions
    let mut checkpoints = Vec::new();
    let mut pos = last;
    let mut k_rem = best_k;
    loop {
        checkpoints.push(pos);
        if k_rem <= 1 || pos == 0 {
            break;
        }
        match prev[pos][k_rem] {
            Some(j) => {
                pos = j;
                k_rem -= 1;
            }
            None => break,
        }
    }
    // Always include node 0
    if !checkpoints.contains(&0) {
        checkpoints.push(0);
    }
    checkpoints.sort_unstable();
    checkpoints.dedup();

    Ok(build_schedule(graph, checkpoints, n))
}

/// Analyse the memory–compute tradeoff curve.
///
/// Computes a series of schedules with increasing numbers of checkpoint slots
/// (1 through `n`) and returns the Pareto-optimal frontier.
///
/// # Arguments
/// * `graph` – The computation graph
///
/// # Returns
/// [`MemoryComputeTradeoff`] containing the Pareto frontier.
///
/// # Example
/// ```rust
/// use scirs2_autograd::checkpointing::{CheckpointedGraph, NodeMeta, analyse_tradeoff};
///
/// let mut g = CheckpointedGraph::new();
/// for i in 0..8 {
///     g.add_node(NodeMeta::new(i, 10, 1.0));
/// }
/// let tradeoff = analyse_tradeoff(&g);
/// // At least 2 points: minimum memory and maximum compute (1 checkpoint) vs
/// // maximum memory and zero recomputation (all checkpoints)
/// assert!(!tradeoff.pareto_points.is_empty());
/// ```
pub fn analyse_tradeoff(graph: &CheckpointedGraph) -> MemoryComputeTradeoff {
    let n = graph.num_nodes();
    let baseline_memory: usize = graph.nodes().map(|nd| nd.activation_size).sum();
    let baseline_forward_cost: f64 = graph.nodes().map(|nd| nd.forward_cost).sum();

    if n == 0 {
        return MemoryComputeTradeoff {
            pareto_points: Vec::new(),
            baseline_memory,
            baseline_forward_cost,
        };
    }

    // Collect (memory, recompute_cost) for each slot count
    let mut raw_points: Vec<(usize, f64, usize)> = Vec::new(); // (mem, cost, slots)

    for slots in 1..=n {
        let sched = optimal_checkpointing_schedule(graph, MemoryBudget::Slots(slots));
        if let Ok(s) = sched {
            raw_points.push((s.estimated_peak_memory, s.estimated_recomputation_cost, slots));
        }
    }

    // Build Pareto frontier: keep only points that are not dominated
    // (neither memory nor recomputation cost is worse than another point)
    raw_points.sort_by_key(|&(mem, _, _)| mem);

    let mut pareto: Vec<TradeoffPoint> = Vec::new();
    let mut min_cost = f64::INFINITY;
    for (mem, cost, slots) in &raw_points {
        if *cost < min_cost {
            min_cost = *cost;
            pareto.push(TradeoffPoint {
                slots: *slots,
                peak_memory: *mem,
                recomputation_overhead: if baseline_forward_cost > 0.0 {
                    *cost / baseline_forward_cost
                } else {
                    0.0
                },
            });
        }
    }

    MemoryComputeTradeoff {
        pareto_points: pareto,
        baseline_memory,
        baseline_forward_cost,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helper
// ─────────────────────────────────────────────────────────────────────────────

/// Build a `CheckpointSchedule` from a list of checkpoint indices and a graph.
fn build_schedule(
    graph: &CheckpointedGraph,
    checkpoints: Vec<usize>,
    n: usize,
) -> CheckpointSchedule {
    // Compute peak memory and recomputation cost using the graph machinery
    let mut g = graph.clone();
    // Clear existing checkpoints and set new ones
    for i in 0..n {
        g.remove_checkpoint(i);
    }
    for &cp in &checkpoints {
        g.checkpoint_node(cp);
    }

    CheckpointSchedule {
        num_nodes: n,
        checkpoints,
        estimated_peak_memory: g.peak_memory(),
        estimated_recomputation_cost: g.recomputation_cost(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform_graph(n: usize, act_size: usize, fwd_cost: f64) -> CheckpointedGraph {
        let mut g = CheckpointedGraph::new();
        for i in 0..n {
            g.add_node(NodeMeta::new(i, act_size, fwd_cost));
        }
        g
    }

    // ── CheckpointedGraph ─────────────────────────────────────────────────

    #[test]
    fn test_graph_checkpoint_tracking() {
        let mut g = make_uniform_graph(5, 10, 1.0);
        g.checkpoint_node(0);
        g.checkpoint_node(2);
        g.checkpoint_node(4);
        assert_eq!(g.checkpoint_count(), 3);
        assert!(g.is_checkpointed(0));
        assert!(!g.is_checkpointed(1));
        g.remove_checkpoint(2);
        assert_eq!(g.checkpoint_count(), 2);
    }

    #[test]
    fn test_peak_memory_all_checkpointed() {
        // If all nodes are checkpointed, peak = sum of all activations
        let mut g = make_uniform_graph(4, 10, 1.0);
        for i in 0..4 {
            g.checkpoint_node(i);
        }
        assert_eq!(g.peak_memory(), 40);
    }

    #[test]
    fn test_peak_memory_no_checkpoints() {
        let g = make_uniform_graph(4, 10, 1.0);
        assert_eq!(g.peak_memory(), 40);
    }

    #[test]
    fn test_recomputation_cost() {
        let mut g = make_uniform_graph(4, 10, 2.0);
        g.checkpoint_node(0);
        g.checkpoint_node(3);
        // Nodes 1, 2 are not checkpointed: cost = 2 * 2.0 = 4.0
        assert!((g.recomputation_cost() - 4.0).abs() < 1e-9);
    }

    // ── checkpoint() functional API ───────────────────────────────────────

    #[test]
    fn test_checkpoint_record() {
        let record = checkpoint(vec![0, 5, 10], || 15);
        assert_eq!(record.stored_inputs, vec![0, 5, 10]);
        assert_eq!(record.output_index, 15);
    }

    // ── sqrt_schedule ─────────────────────────────────────────────────────

    #[test]
    fn test_sqrt_schedule_9_nodes() {
        let g = make_uniform_graph(9, 10, 1.0);
        let sched = sqrt_schedule(&g);
        // k = ceil(sqrt(9)) = 3; checkpoints at 0, 3, 6, 8
        assert!(sched.checkpoints.contains(&0), "should have 0");
        assert!(sched.checkpoints.contains(&3), "should have 3");
        assert!(sched.checkpoints.contains(&8), "should have last");
    }

    #[test]
    fn test_sqrt_schedule_empty() {
        let g = CheckpointedGraph::new();
        let sched = sqrt_schedule(&g);
        assert_eq!(sched.num_nodes, 0);
        assert!(sched.checkpoints.is_empty());
    }

    // ── uniform_schedule ──────────────────────────────────────────────────

    #[test]
    fn test_uniform_schedule_interval_2() {
        let g = make_uniform_graph(6, 10, 1.0);
        let sched = uniform_schedule(&g, 2).expect("uniform sched");
        // 0, 2, 4, 5 (last)
        assert!(sched.checkpoints.contains(&0));
        assert!(sched.checkpoints.contains(&2));
        assert!(sched.checkpoints.contains(&4));
        assert!(sched.checkpoints.contains(&5));
    }

    #[test]
    fn test_uniform_schedule_zero_interval_error() {
        let g = make_uniform_graph(4, 10, 1.0);
        let r = uniform_schedule(&g, 0);
        assert!(r.is_err());
    }

    // ── binomial_schedule ─────────────────────────────────────────────────

    #[test]
    fn test_binomial_schedule_slots() {
        let g = make_uniform_graph(16, 1, 1.0);
        let sched = binomial_schedule(&g, MemoryBudget::Slots(4)).expect("binomial sched");
        // At most 4 + 1 (last) checkpoints
        assert!(sched.checkpoints.len() <= 5, "too many cps: {:?}", sched.checkpoints);
    }

    #[test]
    fn test_binomial_schedule_fraction() {
        let g = make_uniform_graph(10, 10, 1.0);
        let sched = binomial_schedule(&g, MemoryBudget::Fraction(0.5)).expect("binomial frac");
        assert!(!sched.checkpoints.is_empty());
    }

    #[test]
    fn test_binomial_invalid_fraction_error() {
        let g = make_uniform_graph(4, 10, 1.0);
        let r = binomial_schedule(&g, MemoryBudget::Fraction(1.5));
        assert!(r.is_err());
    }

    // ── optimal_checkpointing_schedule ───────────────────────────────────

    #[test]
    fn test_optimal_schedule_small() {
        let g = make_uniform_graph(8, 10, 1.0);
        let sched = optimal_checkpointing_schedule(&g, MemoryBudget::Slots(3))
            .expect("optimal sched");
        assert!(!sched.checkpoints.is_empty());
        assert!(sched.checkpoints.len() <= 4);
    }

    #[test]
    fn test_optimal_schedule_all_slots() {
        // With enough slots, all nodes should be checkpointed
        let g = make_uniform_graph(5, 10, 1.0);
        let sched = optimal_checkpointing_schedule(&g, MemoryBudget::Slots(5))
            .expect("optimal all slots");
        assert_eq!(sched.checkpoints.len(), 5);
        assert!((sched.estimated_recomputation_cost - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_optimal_schedule_zero_slots_error() {
        let g = make_uniform_graph(4, 10, 1.0);
        let r = optimal_checkpointing_schedule(&g, MemoryBudget::Slots(0));
        assert!(r.is_err());
    }

    // ── analyse_tradeoff ──────────────────────────────────────────────────

    #[test]
    fn test_analyse_tradeoff_non_empty() {
        let g = make_uniform_graph(8, 10, 1.0);
        let tradeoff = analyse_tradeoff(&g);
        assert!(!tradeoff.pareto_points.is_empty());
        assert_eq!(tradeoff.baseline_memory, 80);
        assert!((tradeoff.baseline_forward_cost - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_analyse_tradeoff_monotone_memory() {
        // Pareto frontier should have non-decreasing memory
        let g = make_uniform_graph(8, 10, 1.0);
        let tradeoff = analyse_tradeoff(&g);
        let mems: Vec<usize> = tradeoff.pareto_points.iter().map(|p| p.peak_memory).collect();
        for w in mems.windows(2) {
            assert!(w[0] <= w[1], "memory not monotone: {:?}", mems);
        }
    }

    #[test]
    fn test_analyse_tradeoff_empty_graph() {
        let g = CheckpointedGraph::new();
        let tradeoff = analyse_tradeoff(&g);
        assert!(tradeoff.pareto_points.is_empty());
    }
}
