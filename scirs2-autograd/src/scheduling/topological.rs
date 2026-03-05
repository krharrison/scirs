//! Topological sort-based scheduling for computation graphs
//!
//! Provides forward-mode, reverse-mode, and memory-optimal scheduling
//! strategies for executing operations in a computation graph.

use crate::graph::{Graph, TensorID};
use crate::Float;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

/// Scheduling direction for graph traversal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleDirection {
    /// Forward: inputs to outputs (evaluation order)
    Forward,
    /// Reverse: outputs to inputs (gradient computation order)
    Reverse,
}

/// A single scheduled operation
#[derive(Debug, Clone)]
pub struct ScheduledOp {
    /// The tensor/node ID to execute
    pub node_id: TensorID,
    /// Topological rank (lower = earlier in forward order)
    pub topo_rank: usize,
    /// Operation name (for debugging/profiling)
    pub op_name: String,
    /// Input node IDs
    pub inputs: Vec<TensorID>,
    /// Estimated memory usage in bytes (0 if unknown)
    pub estimated_memory: usize,
}

/// Result of scheduling: an ordered list of operations
#[derive(Debug, Clone)]
pub struct Schedule {
    /// Ordered operations to execute
    pub ops: Vec<ScheduledOp>,
    /// Direction used for this schedule
    pub direction: ScheduleDirection,
    /// Peak estimated memory usage (bytes)
    pub peak_memory_estimate: usize,
    /// Total number of operations
    pub total_ops: usize,
}

impl Schedule {
    /// Create an empty schedule
    pub fn empty(direction: ScheduleDirection) -> Self {
        Self {
            ops: Vec::new(),
            direction,
            peak_memory_estimate: 0,
            total_ops: 0,
        }
    }
}

/// Build adjacency information from a computation graph.
///
/// Returns `(children, parents)` where:
/// - `children[i]` = set of nodes that consume the output of node `i`
/// - `parents[i]` = ordered list of inputs to node `i`
fn build_adjacency<F: Float>(graph: &Graph<F>) -> (Vec<HashSet<TensorID>>, Vec<Vec<TensorID>>) {
    let nodes = graph.node_set.borrow();
    let n = nodes.len();
    let mut children: Vec<HashSet<TensorID>> = vec![HashSet::new(); n];
    let mut parents: Vec<Vec<TensorID>> = vec![Vec::new(); n];

    for node in nodes.iter() {
        let nid = node.id;
        for inc in &node.incoming_nodes {
            let pid = inc.id;
            if pid < n {
                children[pid].insert(nid);
                parents[nid].push(pid);
            }
        }
    }

    (children, parents)
}

/// Compute in-degree for every node (number of unsatisfied predecessors).
fn compute_in_degree<F: Float>(graph: &Graph<F>) -> Vec<usize> {
    let nodes = graph.node_set.borrow();
    let n = nodes.len();
    let mut in_deg = vec![0usize; n];
    for node in nodes.iter() {
        in_deg[node.id] = node.incoming_nodes.len();
    }
    in_deg
}

/// Forward-mode scheduling: topological sort from inputs to outputs.
///
/// Uses Kahn's algorithm. Nodes with zero in-degree are emitted first.
/// Ties are broken by ascending `topo_rank` for determinism.
pub fn forward_schedule<F: Float>(graph: &Graph<F>) -> Schedule {
    let nodes = graph.node_set.borrow();
    let n = nodes.len();
    if n == 0 {
        return Schedule::empty(ScheduleDirection::Forward);
    }

    let (children, _parents) = {
        drop(nodes);
        build_adjacency(graph)
    };

    let mut in_deg = compute_in_degree(graph);

    // Min-heap keyed on topo_rank for deterministic ordering
    let mut ready: BinaryHeap<std::cmp::Reverse<(usize, TensorID)>> = BinaryHeap::new();

    let nodes = graph.node_set.borrow();
    for id in 0..n {
        if in_deg[id] == 0 {
            ready.push(std::cmp::Reverse((nodes[id].topo_rank, id)));
        }
    }

    let mut ops = Vec::with_capacity(n);

    while let Some(std::cmp::Reverse((rank, nid))) = ready.pop() {
        let op_name = nodes[nid]
            .op
            .as_ref()
            .map(|o| o.name().to_owned())
            .unwrap_or_else(|| "source".to_owned());

        let inputs: Vec<TensorID> = nodes[nid].incoming_nodes.iter().map(|inc| inc.id).collect();

        ops.push(ScheduledOp {
            node_id: nid,
            topo_rank: rank,
            op_name,
            inputs,
            estimated_memory: 0,
        });

        for &child in &children[nid] {
            in_deg[child] = in_deg[child].saturating_sub(1);
            if in_deg[child] == 0 {
                ready.push(std::cmp::Reverse((nodes[child].topo_rank, child)));
            }
        }
    }

    let total = ops.len();
    Schedule {
        ops,
        direction: ScheduleDirection::Forward,
        peak_memory_estimate: 0,
        total_ops: total,
    }
}

/// Reverse-mode scheduling: topological sort from outputs to inputs.
///
/// Visits nodes in reverse topological order — output nodes first,
/// then their predecessors. Useful for gradient (backpropagation) ordering.
pub fn reverse_schedule<F: Float>(graph: &Graph<F>) -> Schedule {
    let mut fwd = forward_schedule(graph);
    fwd.ops.reverse();
    fwd.direction = ScheduleDirection::Reverse;
    fwd
}

/// Memory-optimal scheduling: minimises peak memory usage.
///
/// Strategy: at every step pick the ready node whose execution frees the most
/// memory (i.e. the node that is the *last* consumer of the most predecessors).
/// This greedy heuristic keeps the live-set small.
pub fn memory_optimal_schedule<F: Float>(graph: &Graph<F>) -> Schedule {
    let nodes_ref = graph.node_set.borrow();
    let n = nodes_ref.len();
    if n == 0 {
        return Schedule::empty(ScheduleDirection::Forward);
    }

    // Pre-collect node data so we can drop the borrow
    let node_data: Vec<(usize, String, Vec<TensorID>)> = nodes_ref
        .iter()
        .map(|nd| {
            let op_name = nd
                .op
                .as_ref()
                .map(|o| o.name().to_owned())
                .unwrap_or_else(|| "source".to_owned());
            let inputs: Vec<TensorID> = nd.incoming_nodes.iter().map(|inc| inc.id).collect();
            (nd.topo_rank, op_name, inputs)
        })
        .collect();

    drop(nodes_ref);

    let (children, _parents) = build_adjacency(graph);
    let mut in_deg = compute_in_degree(graph);

    // ref_count[i] = how many unscheduled consumers of node i remain
    let mut ref_count: Vec<usize> = children.iter().map(|c| c.len()).collect();

    // Ready set: nodes whose all predecessors have been scheduled
    let mut ready: Vec<TensorID> = (0..n).filter(|&id| in_deg[id] == 0).collect();

    let mut ops = Vec::with_capacity(n);
    let mut live_tensors: HashSet<TensorID> = HashSet::new();
    let mut peak_memory: usize = 0;
    let mut current_memory: usize = 0;

    // Assumed tensor size (we don't know shapes at graph-build time).
    // Use 1 unit per tensor as a proxy for "number of live tensors".
    let tensor_unit = 1usize;

    while !ready.is_empty() {
        // Pick the ready node that frees the most predecessors.
        // Freeing score = number of inputs whose ref_count drops to 0.
        let best_idx = {
            let mut best = 0usize;
            let mut best_score = 0usize;

            for (idx, &nid) in ready.iter().enumerate() {
                let score = node_data[nid]
                    .2
                    .iter()
                    .filter(|&&pid| pid < n && ref_count[pid] == 1)
                    .count();
                // Prefer higher score (more memory freed), then lower topo_rank for ties.
                if score > best_score
                    || (score == best_score && node_data[nid].0 < node_data[ready[best]].0)
                {
                    best = idx;
                    best_score = score;
                }
            }

            best
        };

        let nid = ready.swap_remove(best_idx);
        let (rank, ref op_name, ref inputs) = node_data[nid];

        // Execute: produce output tensor → add to live set
        live_tensors.insert(nid);
        current_memory += tensor_unit;

        // Consume inputs: decrement ref_count, free if zero
        for &pid in inputs {
            if pid < n {
                ref_count[pid] = ref_count[pid].saturating_sub(1);
                if ref_count[pid] == 0 && live_tensors.contains(&pid) {
                    live_tensors.remove(&pid);
                    current_memory = current_memory.saturating_sub(tensor_unit);
                }
            }
        }

        if current_memory > peak_memory {
            peak_memory = current_memory;
        }

        ops.push(ScheduledOp {
            node_id: nid,
            topo_rank: rank,
            op_name: op_name.clone(),
            inputs: inputs.clone(),
            estimated_memory: 0,
        });

        // Unlock children whose in-degree is now zero
        for &child in &children[nid] {
            in_deg[child] = in_deg[child].saturating_sub(1);
            if in_deg[child] == 0 {
                ready.push(child);
            }
        }
    }

    let total = ops.len();
    Schedule {
        ops,
        direction: ScheduleDirection::Forward,
        peak_memory_estimate: peak_memory,
        total_ops: total,
    }
}

/// Compute the topological depth of every node (longest path from any source).
///
/// Returns a vector indexed by `TensorID` giving the depth.
pub fn compute_depth<F: Float>(graph: &Graph<F>) -> Vec<usize> {
    let nodes = graph.node_set.borrow();
    let n = nodes.len();
    let mut depth = vec![0usize; n];

    // Process in topo_rank order so parents are visited before children.
    let mut order: Vec<TensorID> = (0..n).collect();
    order.sort_by_key(|&id| nodes[id].topo_rank);

    for &id in &order {
        for inc in &nodes[id].incoming_nodes {
            let pid = inc.id;
            if pid < n && depth[pid] + 1 > depth[id] {
                depth[id] = depth[pid] + 1;
            }
        }
    }

    depth
}

/// Validate that a schedule contains all graph nodes exactly once and respects
/// dependency ordering.
///
/// Returns `Ok(())` on success or a descriptive error string.
pub fn validate_schedule<F: Float>(graph: &Graph<F>, schedule: &Schedule) -> Result<(), String> {
    let nodes = graph.node_set.borrow();
    let n = nodes.len();

    // Check that all nodes are present exactly once
    let mut seen: HashSet<TensorID> = HashSet::with_capacity(n);
    for op in &schedule.ops {
        if !seen.insert(op.node_id) {
            return Err(format!("Duplicate node {} in schedule", op.node_id));
        }
    }

    if seen.len() != n {
        return Err(format!(
            "Schedule contains {} nodes but graph has {}",
            seen.len(),
            n
        ));
    }

    // Check dependency ordering: every input of a node must appear before it
    // (for forward schedules).
    if schedule.direction == ScheduleDirection::Forward {
        let mut position: HashMap<TensorID, usize> = HashMap::with_capacity(n);
        for (pos, op) in schedule.ops.iter().enumerate() {
            position.insert(op.node_id, pos);
        }

        for op in &schedule.ops {
            let my_pos = position.get(&op.node_id).copied().unwrap_or(usize::MAX);
            for &inp in &op.inputs {
                let inp_pos = position.get(&inp).copied().unwrap_or(usize::MAX);
                if inp_pos >= my_pos {
                    return Err(format!(
                        "Dependency violation: node {} at position {} depends on node {} at position {}",
                        op.node_id, my_pos, inp, inp_pos
                    ));
                }
            }
        }
    }

    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::AsGraph;
    use crate::tensor_ops as T;
    use crate::VariableEnvironment;

    #[test]
    fn test_forward_schedule_linear_chain() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[2, 2], ctx);
            let b = T::ones(&[2, 2], ctx);
            let c = a + b;
            let _ = c * T::ones(&[2, 2], ctx);

            let sched = forward_schedule(ctx.as_graph());
            assert!(sched.total_ops > 0);
            assert_eq!(sched.direction, ScheduleDirection::Forward);
            assert!(validate_schedule(ctx.as_graph(), &sched).is_ok());
        });
    }

    #[test]
    fn test_reverse_schedule() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[3], ctx);
            let b = T::ones(&[3], ctx);
            let _ = a + b;

            let sched = reverse_schedule(ctx.as_graph());
            assert_eq!(sched.direction, ScheduleDirection::Reverse);
            assert!(sched.total_ops > 0);
        });
    }

    #[test]
    fn test_memory_optimal_schedule() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[4, 4], ctx);
            let b = T::ones(&[4, 4], ctx);
            let c = a + b;
            let d = a * b;
            let _ = c + d;

            let sched = memory_optimal_schedule(ctx.as_graph());
            assert!(sched.total_ops > 0);
            assert!(validate_schedule(ctx.as_graph(), &sched).is_ok());
        });
    }

    #[test]
    fn test_empty_graph_schedule() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let sched = forward_schedule(ctx.as_graph());
            assert_eq!(sched.total_ops, 0);
        });
    }

    #[test]
    fn test_compute_depth() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[2], ctx);
            let b = T::ones(&[2], ctx);
            let c = a + b;
            let d = c * T::ones(&[2], ctx);
            let _ = d;

            let depths = compute_depth(ctx.as_graph());
            // T::zeros/ones take a shape tensor as input, so depth(a) >= 1.
            // c = a + b has depth > depth(a).
            assert!(depths[c.id] > depths[a.id], "c should be deeper than a");
            // d depends on c, so depth(d) > depth(c).
            assert!(depths[d.id] > depths[c.id], "d should be deeper than c");
        });
    }

    #[test]
    fn test_validate_schedule_catches_missing_nodes() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[2], ctx);
            let _ = T::ones(&[2], ctx);
            let _ = a;

            let mut sched = forward_schedule(ctx.as_graph());
            // Remove one op to break validity
            if !sched.ops.is_empty() {
                sched.ops.pop();
                sched.total_ops = sched.ops.len();
                assert!(validate_schedule(ctx.as_graph(), &sched).is_err());
            }
        });
    }

    #[test]
    fn test_diamond_graph_schedule() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            // Diamond: a -> b, a -> c, b+c -> d
            let a = T::zeros(&[3], ctx);
            let b = a + T::ones(&[3], ctx);
            let c = a * T::ones(&[3], ctx);
            let _ = b + c;

            let sched = forward_schedule(ctx.as_graph());
            assert!(validate_schedule(ctx.as_graph(), &sched).is_ok());
        });
    }
}
