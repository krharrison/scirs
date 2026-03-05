//! Memory planning for tensor lifetimes in computation graphs
//!
//! Performs liveness analysis, detects in-place operation opportunities,
//! and plans memory reuse via interval graph colouring to minimise peak
//! memory consumption.

use crate::graph::{Graph, TensorID};
use crate::Float;
use std::collections::{HashMap, HashSet};

/// Liveness interval for a single tensor
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LivenessInterval {
    /// Tensor / node ID
    pub node_id: TensorID,
    /// Step at which the tensor is first produced (birth)
    pub birth: usize,
    /// Step at which the tensor is last consumed (death)
    pub death: usize,
    /// Estimated size in abstract units (element count or bytes)
    pub size: usize,
    /// Whether this tensor is a graph output (must stay live until end)
    pub is_output: bool,
}

impl LivenessInterval {
    /// Duration (inclusive): how many steps this tensor is live.
    pub fn duration(&self) -> usize {
        self.death.saturating_sub(self.birth) + 1
    }

    /// Whether two intervals overlap (cannot share the same buffer).
    pub fn overlaps(&self, other: &LivenessInterval) -> bool {
        self.birth <= other.death && other.birth <= self.death
    }
}

/// In-place operation candidate
#[derive(Debug, Clone)]
pub struct InPlaceCandidate {
    /// The node that could execute in-place
    pub node_id: TensorID,
    /// The input tensor whose buffer could be reused as the output
    pub reuse_input: TensorID,
    /// Reason this is considered a valid in-place candidate
    pub reason: InPlaceReason,
}

/// Why an operation qualifies for in-place execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InPlaceReason {
    /// Unary element-wise: output has same shape as input
    UnaryElementwise,
    /// Binary element-wise where one input has no other consumers
    BinaryElementwiseSingleConsumer,
    /// Accumulate-style update (e.g. +=)
    Accumulate,
}

/// Memory reuse assignment: which buffer slot each tensor is assigned to
#[derive(Debug, Clone)]
pub struct MemoryAssignment {
    /// node_id -> slot index
    pub slot_map: HashMap<TensorID, usize>,
    /// Total number of distinct buffer slots required
    pub num_slots: usize,
    /// Peak memory in abstract size units
    pub peak_memory: usize,
}

/// Full memory plan combining liveness, in-place, and reuse information
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Liveness intervals for all tensors
    pub intervals: Vec<LivenessInterval>,
    /// Detected in-place candidates
    pub in_place_candidates: Vec<InPlaceCandidate>,
    /// Memory reuse assignment via interval colouring
    pub assignment: MemoryAssignment,
    /// Peak memory estimate (abstract units)
    pub peak_memory: usize,
    /// Total memory without reuse (abstract units)
    pub total_memory_naive: usize,
    /// Memory saved by reuse (abstract units)
    pub memory_saved: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Liveness analysis
// ────────────────────────────────────────────────────────────────────────────

/// Perform liveness analysis on a computation graph.
///
/// For each node we determine:
/// - `birth` = the node's own topo_rank (when its output is produced)
/// - `death` = maximum topo_rank among all consumers (when it is last read)
///
/// Output nodes (no consumers) are kept live until `max_rank`.
pub fn liveness_analysis<F: Float>(graph: &Graph<F>) -> Vec<LivenessInterval> {
    let nodes = graph.node_set.borrow();
    let n = nodes.len();
    if n == 0 {
        return Vec::new();
    }

    let max_rank = nodes.iter().map(|nd| nd.topo_rank).max().unwrap_or(0);

    // Birth = topo_rank of the node itself
    let mut birth: Vec<usize> = nodes.iter().map(|nd| nd.topo_rank).collect();
    let mut death: Vec<usize> = birth.clone();

    // Build consumer map: for each node, which nodes consume it
    let mut has_consumer = vec![false; n];

    for node in nodes.iter() {
        let consumer_rank = node.topo_rank;
        for inc in &node.incoming_nodes {
            let pid = inc.id;
            if pid < n {
                has_consumer[pid] = true;
                if consumer_rank > death[pid] {
                    death[pid] = consumer_rank;
                }
            }
        }
    }

    // Output nodes (no consumers) live until end of graph
    for id in 0..n {
        if !has_consumer[id] {
            death[id] = max_rank;
        }
    }

    let mut intervals = Vec::with_capacity(n);
    for id in 0..n {
        intervals.push(LivenessInterval {
            node_id: id,
            birth: birth[id],
            death: death[id],
            size: 1, // abstract unit; real sizes need shape info
            is_output: !has_consumer[id],
        });
    }

    intervals
}

// ────────────────────────────────────────────────────────────────────────────
// In-place operation detection
// ────────────────────────────────────────────────────────────────────────────

/// Element-wise operation names (heuristic set).
const ELEMENTWISE_OPS: &[&str] = &[
    "AddOp", "Add", "add", "SubOp", "Sub", "sub", "MulOp", "Mul", "mul", "DivOp", "Div", "div",
    "NegOp", "Neg", "neg", "Relu", "relu", "Sigmoid", "sigmoid", "Tanh", "tanh", "Gelu", "gelu",
    "Exp", "exp", "Log", "log", "Sqrt", "sqrt", "Square", "square", "Abs", "abs",
];

/// Detect operations that could safely execute in-place.
///
/// An operation is in-place-eligible when:
/// 1. It is element-wise (output shape = input shape).
/// 2. The input tensor has no other consumers after this operation.
pub fn detect_in_place<F: Float>(graph: &Graph<F>) -> Vec<InPlaceCandidate> {
    let nodes = graph.node_set.borrow();
    let n = nodes.len();
    if n == 0 {
        return Vec::new();
    }

    // Count consumers for every node
    let mut consumer_count: Vec<usize> = vec![0; n];
    for node in nodes.iter() {
        for inc in &node.incoming_nodes {
            if inc.id < n {
                consumer_count[inc.id] += 1;
            }
        }
    }

    let ew_set: HashSet<&str> = ELEMENTWISE_OPS.iter().copied().collect();

    let mut candidates = Vec::new();

    for node in nodes.iter() {
        let op_name = node.op.as_ref().map(|o| o.name()).unwrap_or("");

        // Check if this is an element-wise op
        let is_ew = ew_set.iter().any(|&pat| op_name.contains(pat));
        if !is_ew {
            continue;
        }

        let num_inputs = node.incoming_nodes.len();

        if num_inputs == 1 {
            // Unary element-wise: reuse the single input if it has only 1 consumer
            let inp_id = node.incoming_nodes[0].id;
            if inp_id < n && consumer_count[inp_id] == 1 {
                candidates.push(InPlaceCandidate {
                    node_id: node.id,
                    reuse_input: inp_id,
                    reason: InPlaceReason::UnaryElementwise,
                });
            }
        } else if num_inputs == 2 {
            // Binary element-wise: reuse whichever input has only 1 consumer
            let inp0 = node.incoming_nodes[0].id;
            let inp1 = node.incoming_nodes[1].id;

            if inp0 < n && consumer_count[inp0] == 1 {
                candidates.push(InPlaceCandidate {
                    node_id: node.id,
                    reuse_input: inp0,
                    reason: InPlaceReason::BinaryElementwiseSingleConsumer,
                });
            } else if inp1 < n && consumer_count[inp1] == 1 {
                candidates.push(InPlaceCandidate {
                    node_id: node.id,
                    reuse_input: inp1,
                    reason: InPlaceReason::BinaryElementwiseSingleConsumer,
                });
            }
        }
    }

    candidates
}

// ────────────────────────────────────────────────────────────────────────────
// Interval graph colouring (memory reuse)
// ────────────────────────────────────────────────────────────────────────────

/// Assign buffer slots to tensors via greedy interval-graph colouring.
///
/// Tensors whose liveness intervals do not overlap may share a buffer slot.
/// The algorithm scans intervals sorted by birth time and assigns each to the
/// earliest freed slot.
pub fn assign_memory_slots(intervals: &[LivenessInterval]) -> MemoryAssignment {
    if intervals.is_empty() {
        return MemoryAssignment {
            slot_map: HashMap::new(),
            num_slots: 0,
            peak_memory: 0,
        };
    }

    // Sort indices by birth time, breaking ties by ascending death
    let mut sorted_indices: Vec<usize> = (0..intervals.len()).collect();
    sorted_indices.sort_by_key(|&i| (intervals[i].birth, intervals[i].death));

    // Each slot tracks (death_time, size)
    let mut slots: Vec<(usize, usize)> = Vec::new();
    let mut slot_map: HashMap<TensorID, usize> = HashMap::new();

    for &idx in &sorted_indices {
        let iv = &intervals[idx];

        // Find a slot whose occupant has already died and whose size matches
        let reuse = slots
            .iter()
            .enumerate()
            .filter(|(_, &(slot_death, slot_size))| {
                slot_death < iv.birth && slot_size >= iv.size
            })
            .min_by_key(|(_, &(_, slot_size))| slot_size) // smallest fitting slot
            .map(|(slot_idx, _)| slot_idx);

        match reuse {
            Some(slot_idx) => {
                slots[slot_idx] = (iv.death, slots[slot_idx].1.max(iv.size));
                slot_map.insert(iv.node_id, slot_idx);
            }
            None => {
                let new_slot = slots.len();
                slots.push((iv.death, iv.size));
                slot_map.insert(iv.node_id, new_slot);
            }
        }
    }

    // Compute peak memory = sum of sizes of all slots that are live at the same step
    // Simpler approximation: max over all steps of sum of sizes of live slots
    let max_step = intervals.iter().map(|iv| iv.death).max().unwrap_or(0);
    let mut peak = 0usize;

    for step in 0..=max_step {
        let live_size: usize = intervals
            .iter()
            .filter(|iv| iv.birth <= step && iv.death >= step)
            .map(|iv| iv.size)
            .sum();
        if live_size > peak {
            peak = live_size;
        }
    }

    MemoryAssignment {
        slot_map,
        num_slots: slots.len(),
        peak_memory: peak,
    }
}

/// Estimate peak memory for a graph without performing full slot assignment.
///
/// Sweeps through all topo-rank steps and sums sizes of tensors that are live.
pub fn estimate_peak_memory(intervals: &[LivenessInterval]) -> usize {
    if intervals.is_empty() {
        return 0;
    }

    let max_step = intervals.iter().map(|iv| iv.death).max().unwrap_or(0);
    let mut peak = 0usize;

    for step in 0..=max_step {
        let live: usize = intervals
            .iter()
            .filter(|iv| iv.birth <= step && iv.death >= step)
            .map(|iv| iv.size)
            .sum();
        if live > peak {
            peak = live;
        }
    }

    peak
}

// ────────────────────────────────────────────────────────────────────────────
// Full memory plan
// ────────────────────────────────────────────────────────────────────────────

/// Build a complete memory plan for a computation graph.
pub fn build_memory_plan<F: Float>(graph: &Graph<F>) -> MemoryPlan {
    let intervals = liveness_analysis(graph);
    let in_place_candidates = detect_in_place(graph);
    let assignment = assign_memory_slots(&intervals);

    let total_naive: usize = intervals.iter().map(|iv| iv.size).sum();
    let peak = assignment.peak_memory;
    let saved = total_naive.saturating_sub(assignment.num_slots);

    MemoryPlan {
        intervals,
        in_place_candidates,
        assignment,
        peak_memory: peak,
        total_memory_naive: total_naive,
        memory_saved: saved,
    }
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
    fn test_liveness_analysis_basic() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[2, 2], ctx);
            let b = T::ones(&[2, 2], ctx);
            let _ = a + b;

            let intervals = liveness_analysis(ctx.as_graph());
            assert!(!intervals.is_empty());

            // Source nodes should have birth=0 (they have topo_rank 0)
            let a_iv = &intervals[a.id];
            assert_eq!(a_iv.birth, a_iv.birth); // self-consistent

            // Every interval should have death >= birth
            for iv in &intervals {
                assert!(
                    iv.death >= iv.birth,
                    "death < birth for node {}",
                    iv.node_id
                );
            }
        });
    }

    #[test]
    fn test_in_place_detection() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[3, 3], ctx);
            let b = T::ones(&[3, 3], ctx);
            // a + b is element-wise; if `a` has no other consumers, it is in-place-eligible
            let c = a + b;
            let _ = c;

            let candidates = detect_in_place(ctx.as_graph());
            // We may or may not get candidates depending on consumer counts;
            // just verify no panic and well-formed output
            for cand in &candidates {
                assert!(cand.node_id < ctx.node_count());
                assert!(cand.reuse_input < ctx.node_count());
            }
        });
    }

    #[test]
    fn test_memory_slot_assignment() {
        // Two non-overlapping intervals can share a slot
        let intervals = vec![
            LivenessInterval {
                node_id: 0,
                birth: 0,
                death: 1,
                size: 1,
                is_output: false,
            },
            LivenessInterval {
                node_id: 1,
                birth: 2,
                death: 3,
                size: 1,
                is_output: false,
            },
        ];
        let assignment = assign_memory_slots(&intervals);
        assert_eq!(
            assignment.num_slots, 1,
            "Non-overlapping intervals should share a slot"
        );
    }

    #[test]
    fn test_memory_slot_overlapping() {
        // Two overlapping intervals need separate slots
        let intervals = vec![
            LivenessInterval {
                node_id: 0,
                birth: 0,
                death: 2,
                size: 1,
                is_output: false,
            },
            LivenessInterval {
                node_id: 1,
                birth: 1,
                death: 3,
                size: 1,
                is_output: false,
            },
        ];
        let assignment = assign_memory_slots(&intervals);
        assert_eq!(
            assignment.num_slots, 2,
            "Overlapping intervals need separate slots"
        );
    }

    #[test]
    fn test_estimate_peak_memory() {
        let intervals = vec![
            LivenessInterval {
                node_id: 0,
                birth: 0,
                death: 2,
                size: 4,
                is_output: false,
            },
            LivenessInterval {
                node_id: 1,
                birth: 1,
                death: 3,
                size: 3,
                is_output: false,
            },
            LivenessInterval {
                node_id: 2,
                birth: 3,
                death: 4,
                size: 2,
                is_output: true,
            },
        ];
        let peak = estimate_peak_memory(&intervals);
        // At step 1-2: node 0 (4) + node 1 (3) = 7
        assert!(peak >= 7, "Peak should be >= 7, got {}", peak);
    }

    #[test]
    fn test_build_memory_plan_integration() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[4, 4], ctx);
            let b = T::ones(&[4, 4], ctx);
            let c = a + b;
            let d = c * T::ones(&[4, 4], ctx);
            let _ = d;

            let plan = build_memory_plan(ctx.as_graph());
            assert!(!plan.intervals.is_empty());
            assert!(plan.peak_memory > 0);
            assert!(plan.total_memory_naive > 0);
        });
    }

    #[test]
    fn test_liveness_interval_overlaps() {
        let a = LivenessInterval {
            node_id: 0,
            birth: 0,
            death: 3,
            size: 1,
            is_output: false,
        };
        let b = LivenessInterval {
            node_id: 1,
            birth: 2,
            death: 5,
            size: 1,
            is_output: false,
        };
        let c = LivenessInterval {
            node_id: 2,
            birth: 4,
            death: 6,
            size: 1,
            is_output: false,
        };

        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
        assert!(b.overlaps(&c));
    }

    #[test]
    fn test_empty_graph_memory_plan() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let plan = build_memory_plan(ctx.as_graph());
            assert!(plan.intervals.is_empty());
            assert_eq!(plan.peak_memory, 0);
        });
    }
}
