//! Profiling and debugging tools for computation graphs
//!
//! Provides:
//! - Operation count per type
//! - FLOP estimation per operation
//! - Memory bandwidth estimation
//! - Graph complexity metrics
//! - Gradient flow analysis (detect vanishing/exploding gradients)
//! - Operation timing with optional instrumentation

use crate::graph::{Graph, TensorID};
use crate::Float;
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ────────────────────────────────────────────────────────────────────────────
// 1. Operation Counts
// ────────────────────────────────────────────────────────────────────────────

/// Summary of operation counts by type.
#[derive(Debug, Clone, Default)]
pub struct OpCounts {
    /// Map from operation name to count
    pub counts: HashMap<String, usize>,
    /// Total number of operations
    pub total: usize,
    /// Number of source (leaf) nodes
    pub sources: usize,
    /// Number of non-source (compute) nodes
    pub compute_nodes: usize,
}

impl OpCounts {
    /// Get the most common operation types, sorted by count descending.
    pub fn top_ops(&self, n: usize) -> Vec<(String, usize)> {
        let mut items: Vec<(String, usize)> = self.counts.clone().into_iter().collect();
        items.sort_by_key(|item| std::cmp::Reverse(item.1));
        items.truncate(n);
        items
    }
}

/// Count operations by type in a computation graph.
pub fn count_ops<F: Float>(graph: &Graph<F>) -> OpCounts {
    let nodes = graph.node_set.borrow();
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut sources = 0usize;
    let mut compute = 0usize;

    for node in nodes.iter() {
        let op_name = node
            .op
            .as_ref()
            .map(|o| o.name().to_owned())
            .unwrap_or_else(|| "unknown".to_owned());

        *counts.entry(op_name).or_insert(0) += 1;

        if node.incoming_nodes.is_empty() {
            sources += 1;
        } else {
            compute += 1;
        }
    }

    let total = nodes.len();
    OpCounts {
        counts,
        total,
        sources,
        compute_nodes: compute,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 2. FLOP Estimation
// ────────────────────────────────────────────────────────────────────────────

/// FLOP estimate for a single operation.
#[derive(Debug, Clone)]
pub struct FlopEstimate {
    /// Node ID
    pub node_id: TensorID,
    /// Operation name
    pub op_name: String,
    /// Estimated FLOPs (floating-point operations)
    pub flops: u64,
    /// Category of estimate (exact, heuristic, unknown)
    pub confidence: EstimateConfidence,
}

/// How confident we are in the FLOP estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimateConfidence {
    /// Exact count based on known shapes
    Exact,
    /// Heuristic based on operation type
    Heuristic,
    /// Unknown; using a conservative default
    Unknown,
}

/// Known FLOP costs per element for common operations.
fn flops_per_element(op_name: &str) -> (u64, EstimateConfidence) {
    let lower = op_name.to_lowercase();

    if lower.contains("add")
        || lower.contains("sub")
        || lower.contains("neg")
        || lower.contains("mul")
        || lower.contains("div")
        || lower.contains("relu")
    {
        (1, EstimateConfidence::Exact)
    } else if lower.contains("sigmoid") {
        (4, EstimateConfidence::Heuristic) // exp + add + div + neg
    } else if lower.contains("tanh") {
        (5, EstimateConfidence::Heuristic) // exp, sub, add, div
    } else if lower.contains("gelu") {
        (8, EstimateConfidence::Heuristic) // erf approximation
    } else if lower.contains("exp") || lower.contains("log") || lower.contains("sqrt") {
        (3, EstimateConfidence::Heuristic) // transcendental
    } else if lower.contains("softmax") {
        (5, EstimateConfidence::Heuristic) // exp + sum + div
    } else if lower.contains("matmul") {
        // For matmul, FLOPs = 2*m*n*k. We use a placeholder per-element cost.
        (2, EstimateConfidence::Heuristic) // needs shape info for accuracy
    } else if lower.contains("conv") {
        (2, EstimateConfidence::Heuristic)
    } else if lower.contains("batchnorm") || lower.contains("batch_norm") {
        (4, EstimateConfidence::Heuristic)
    } else if lower.contains("layernorm") || lower.contains("layer_norm") {
        (5, EstimateConfidence::Heuristic)
    } else {
        (1, EstimateConfidence::Unknown)
    }
}

/// Estimate FLOPs for every operation in the graph.
///
/// Without shape information, uses heuristic per-element costs multiplied by a
/// default element count (1024). When shapes are known, they should be passed
/// separately for accurate estimates.
pub fn estimate_flops<F: Float>(graph: &Graph<F>) -> Vec<FlopEstimate> {
    let nodes = graph.node_set.borrow();
    let default_elements: u64 = 1024; // conservative default

    nodes
        .iter()
        .map(|node| {
            let op_name = node
                .op
                .as_ref()
                .map(|o| o.name().to_owned())
                .unwrap_or_else(|| "source".to_owned());

            let (per_elem, confidence) = if node.incoming_nodes.is_empty() {
                (0, EstimateConfidence::Exact) // source nodes have 0 FLOPs
            } else {
                flops_per_element(&op_name)
            };

            FlopEstimate {
                node_id: node.id,
                op_name,
                flops: per_elem * default_elements,
                confidence,
            }
        })
        .collect()
}

/// Total estimated FLOPs for the entire graph.
pub fn total_flops<F: Float>(graph: &Graph<F>) -> u64 {
    estimate_flops(graph).iter().map(|e| e.flops).sum()
}

// ────────────────────────────────────────────────────────────────────────────
// 3. Memory Bandwidth Estimation
// ────────────────────────────────────────────────────────────────────────────

/// Memory bandwidth estimate for an operation.
#[derive(Debug, Clone)]
pub struct BandwidthEstimate {
    /// Node ID
    pub node_id: TensorID,
    /// Bytes read (inputs)
    pub bytes_read: u64,
    /// Bytes written (outputs)
    pub bytes_written: u64,
    /// Total bytes transferred
    pub total_bytes: u64,
    /// Arithmetic intensity (FLOPs / bytes)
    pub arithmetic_intensity: f64,
}

/// Estimate memory bandwidth for each operation.
///
/// Assumes each tensor element is `element_size` bytes (default 4 for f32, 8 for f64).
pub fn estimate_bandwidth<F: Float>(
    graph: &Graph<F>,
    element_size: u64,
    default_elements: u64,
) -> Vec<BandwidthEstimate> {
    let nodes = graph.node_set.borrow();
    let flops = estimate_flops_internal(&nodes, default_elements);

    nodes
        .iter()
        .enumerate()
        .map(|(idx, node)| {
            let num_inputs = node.incoming_nodes.len() as u64;
            let bytes_read = num_inputs * default_elements * element_size;
            let bytes_written = default_elements * element_size;
            let total = bytes_read + bytes_written;
            let ai = if total > 0 {
                flops[idx] as f64 / total as f64
            } else {
                0.0
            };

            BandwidthEstimate {
                node_id: node.id,
                bytes_read,
                bytes_written,
                total_bytes: total,
                arithmetic_intensity: ai,
            }
        })
        .collect()
}

fn estimate_flops_internal<F: Float>(
    nodes: &[crate::tensor::TensorInternal<F>],
    default_elements: u64,
) -> Vec<u64> {
    nodes
        .iter()
        .map(|node| {
            let op_name = node.op.as_ref().map(|o| o.name()).unwrap_or("source");
            let (per_elem, _) = if node.incoming_nodes.is_empty() {
                (0, EstimateConfidence::Exact)
            } else {
                flops_per_element(op_name)
            };
            per_elem * default_elements
        })
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// 4. Graph Complexity Metrics
// ────────────────────────────────────────────────────────────────────────────

/// Comprehensive graph complexity metrics.
#[derive(Debug, Clone)]
pub struct GraphComplexity {
    /// Total number of nodes
    pub num_nodes: usize,
    /// Number of edges (input references)
    pub num_edges: usize,
    /// Maximum depth (longest path from source to output)
    pub max_depth: usize,
    /// Maximum width (most nodes at any single depth level)
    pub max_width: usize,
    /// Average fan-in (inputs per node)
    pub avg_fan_in: f64,
    /// Average fan-out (consumers per node)
    pub avg_fan_out: f64,
    /// Maximum fan-in
    pub max_fan_in: usize,
    /// Maximum fan-out
    pub max_fan_out: usize,
    /// Number of distinct operation types
    pub num_op_types: usize,
    /// Graph density (edges / (nodes * (nodes-1)))
    pub density: f64,
}

/// Compute graph complexity metrics.
pub fn graph_complexity<F: Float>(graph: &Graph<F>) -> GraphComplexity {
    let nodes = graph.node_set.borrow();
    let n = nodes.len();

    if n == 0 {
        return GraphComplexity {
            num_nodes: 0,
            num_edges: 0,
            max_depth: 0,
            max_width: 0,
            avg_fan_in: 0.0,
            avg_fan_out: 0.0,
            max_fan_in: 0,
            max_fan_out: 0,
            num_op_types: 0,
            density: 0.0,
        };
    }

    // Edge count and fan-in
    let mut num_edges = 0usize;
    let mut max_fan_in = 0usize;
    let mut fan_out = vec![0usize; n];

    for node in nodes.iter() {
        let fan_in = node.incoming_nodes.len();
        num_edges += fan_in;
        if fan_in > max_fan_in {
            max_fan_in = fan_in;
        }
        for inc in &node.incoming_nodes {
            if inc.id < n {
                fan_out[inc.id] += 1;
            }
        }
    }

    let max_fan_out = fan_out.iter().copied().max().unwrap_or(0);
    let avg_fan_in = if n > 0 {
        num_edges as f64 / n as f64
    } else {
        0.0
    };
    let avg_fan_out = avg_fan_in; // same total edges

    // Depth computation
    let mut depth = vec![0usize; n];
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&id| nodes[id].topo_rank);
    for &id in &order {
        for inc in &nodes[id].incoming_nodes {
            let pid = inc.id;
            if pid < n {
                let candidate = depth[pid] + 1;
                if candidate > depth[id] {
                    depth[id] = candidate;
                }
            }
        }
    }
    let max_depth = depth.iter().copied().max().unwrap_or(0);

    // Width per depth level
    let mut level_counts: HashMap<usize, usize> = HashMap::new();
    for &d in &depth {
        *level_counts.entry(d).or_insert(0) += 1;
    }
    let max_width = level_counts.values().copied().max().unwrap_or(0);

    // Op types
    let mut op_types: std::collections::HashSet<String> = std::collections::HashSet::new();
    for node in nodes.iter() {
        let name = node
            .op
            .as_ref()
            .map(|o| o.name().to_owned())
            .unwrap_or_default();
        op_types.insert(name);
    }

    let density = if n > 1 {
        num_edges as f64 / (n as f64 * (n as f64 - 1.0))
    } else {
        0.0
    };

    GraphComplexity {
        num_nodes: n,
        num_edges,
        max_depth,
        max_width,
        avg_fan_in,
        avg_fan_out,
        max_fan_in,
        max_fan_out,
        num_op_types: op_types.len(),
        density,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 5. Gradient Flow Analysis
// ────────────────────────────────────────────────────────────────────────────

/// Gradient health status for a layer or node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientHealth {
    /// Gradient magnitude is in a healthy range
    Healthy,
    /// Gradient magnitude is dangerously small (vanishing)
    Vanishing,
    /// Gradient magnitude is dangerously large (exploding)
    Exploding,
    /// No gradient information available
    Unknown,
}

/// Per-node gradient flow statistics.
#[derive(Debug, Clone)]
pub struct GradientFlowStats {
    /// Node ID
    pub node_id: TensorID,
    /// Operation name
    pub op_name: String,
    /// Mean absolute gradient (if available)
    pub mean_abs_grad: Option<f64>,
    /// Max absolute gradient
    pub max_abs_grad: Option<f64>,
    /// Min absolute gradient
    pub min_abs_grad: Option<f64>,
    /// Health assessment
    pub health: GradientHealth,
}

/// Thresholds for gradient health classification.
#[derive(Debug, Clone)]
pub struct GradientThresholds {
    /// Below this mean magnitude: vanishing
    pub vanishing_threshold: f64,
    /// Above this mean magnitude: exploding
    pub exploding_threshold: f64,
}

impl Default for GradientThresholds {
    fn default() -> Self {
        Self {
            vanishing_threshold: 1e-7,
            exploding_threshold: 1e3,
        }
    }
}

/// Classify a gradient magnitude into a health status.
pub fn classify_gradient(mean_abs: f64, thresholds: &GradientThresholds) -> GradientHealth {
    if mean_abs < thresholds.vanishing_threshold {
        GradientHealth::Vanishing
    } else if mean_abs > thresholds.exploding_threshold {
        GradientHealth::Exploding
    } else {
        GradientHealth::Healthy
    }
}

/// Analyse gradient flow given per-node gradient magnitudes.
///
/// `gradient_magnitudes` maps node IDs to (mean_abs, max_abs, min_abs).
pub fn analyse_gradient_flow<F: Float>(
    graph: &Graph<F>,
    gradient_magnitudes: &HashMap<TensorID, (f64, f64, f64)>,
    thresholds: &GradientThresholds,
) -> Vec<GradientFlowStats> {
    let nodes = graph.node_set.borrow();

    nodes
        .iter()
        .map(|node| {
            let op_name = node
                .op
                .as_ref()
                .map(|o| o.name().to_owned())
                .unwrap_or_else(|| "unknown".to_owned());

            match gradient_magnitudes.get(&node.id) {
                Some(&(mean_abs, max_abs, min_abs)) => {
                    let health = classify_gradient(mean_abs, thresholds);
                    GradientFlowStats {
                        node_id: node.id,
                        op_name,
                        mean_abs_grad: Some(mean_abs),
                        max_abs_grad: Some(max_abs),
                        min_abs_grad: Some(min_abs),
                        health,
                    }
                }
                None => GradientFlowStats {
                    node_id: node.id,
                    op_name,
                    mean_abs_grad: None,
                    max_abs_grad: None,
                    min_abs_grad: None,
                    health: GradientHealth::Unknown,
                },
            }
        })
        .collect()
}

/// Quick check: are there any vanishing or exploding gradients?
pub fn has_gradient_issues(stats: &[GradientFlowStats]) -> bool {
    stats
        .iter()
        .any(|s| s.health == GradientHealth::Vanishing || s.health == GradientHealth::Exploding)
}

// ────────────────────────────────────────────────────────────────────────────
// 6. Operation Timing / Instrumentation
// ────────────────────────────────────────────────────────────────────────────

/// Timing record for a single operation execution.
#[derive(Debug, Clone)]
pub struct OpTiming {
    /// Node ID
    pub node_id: TensorID,
    /// Operation name
    pub op_name: String,
    /// Wall-clock duration
    pub duration: Duration,
}

/// Profiler that records operation timings.
#[derive(Debug)]
pub struct OperationProfiler {
    timings: Vec<OpTiming>,
    active_start: Option<(TensorID, String, Instant)>,
}

impl Default for OperationProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl OperationProfiler {
    /// Create a new profiler.
    pub fn new() -> Self {
        Self {
            timings: Vec::new(),
            active_start: None,
        }
    }

    /// Begin timing an operation.
    pub fn start_op(&mut self, node_id: TensorID, op_name: &str) {
        self.active_start = Some((node_id, op_name.to_owned(), Instant::now()));
    }

    /// End timing the current operation.
    pub fn end_op(&mut self) {
        if let Some((node_id, op_name, start)) = self.active_start.take() {
            self.timings.push(OpTiming {
                node_id,
                op_name,
                duration: start.elapsed(),
            });
        }
    }

    /// Record a timing directly (for external measurements).
    pub fn record(&mut self, node_id: TensorID, op_name: &str, duration: Duration) {
        self.timings.push(OpTiming {
            node_id,
            op_name: op_name.to_owned(),
            duration,
        });
    }

    /// Get all recorded timings.
    pub fn timings(&self) -> &[OpTiming] {
        &self.timings
    }

    /// Total time across all recorded operations.
    pub fn total_time(&self) -> Duration {
        self.timings.iter().map(|t| t.duration).sum()
    }

    /// Average time per operation.
    pub fn average_time(&self) -> Duration {
        if self.timings.is_empty() {
            return Duration::ZERO;
        }
        self.total_time() / self.timings.len() as u32
    }

    /// Top N slowest operations.
    pub fn slowest_ops(&self, n: usize) -> Vec<&OpTiming> {
        let mut sorted: Vec<&OpTiming> = self.timings.iter().collect();
        sorted.sort_by_key(|item| std::cmp::Reverse(item.duration));
        sorted.truncate(n);
        sorted
    }

    /// Aggregate time by operation type.
    pub fn time_by_op_type(&self) -> HashMap<String, Duration> {
        let mut agg: HashMap<String, Duration> = HashMap::new();
        for timing in &self.timings {
            *agg.entry(timing.op_name.clone()).or_insert(Duration::ZERO) += timing.duration;
        }
        agg
    }

    /// Clear all recorded timings.
    pub fn clear(&mut self) {
        self.timings.clear();
        self.active_start = None;
    }

    /// Number of recorded operations.
    pub fn num_records(&self) -> usize {
        self.timings.len()
    }
}

/// Full profiling report combining all analysis.
#[derive(Debug, Clone)]
pub struct ProfilingReport {
    /// Operation counts
    pub op_counts: OpCounts,
    /// Total estimated FLOPs
    pub total_flops: u64,
    /// Graph complexity metrics
    pub complexity: GraphComplexity,
    /// Number of gradient health issues (vanishing + exploding)
    pub gradient_issues: usize,
}

/// Generate a full profiling report for a graph.
pub fn profile_graph<F: Float>(graph: &Graph<F>) -> ProfilingReport {
    let op_counts = count_ops(graph);
    let flops = total_flops(graph);
    let complexity = graph_complexity(graph);

    ProfilingReport {
        op_counts,
        total_flops: flops,
        complexity,
        gradient_issues: 0, // requires runtime gradient data
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
    fn test_count_ops() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[2, 2], ctx);
            let b = T::ones(&[2, 2], ctx);
            let c = a + b;
            let _ = c * T::ones(&[2, 2], ctx);

            let counts = count_ops(ctx.as_graph());
            assert!(counts.total > 0);
            assert!(counts.sources >= 2);
            assert!(counts.compute_nodes >= 2);
        });
    }

    #[test]
    fn test_count_ops_empty() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let counts = count_ops(ctx.as_graph());
            assert_eq!(counts.total, 0);
        });
    }

    #[test]
    fn test_top_ops() {
        let mut counts = OpCounts::default();
        counts.counts.insert("AddOp".to_owned(), 10);
        counts.counts.insert("MulOp".to_owned(), 5);
        counts.counts.insert("Relu".to_owned(), 3);

        let top = counts.top_ops(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "AddOp");
        assert_eq!(top[1].0, "MulOp");
    }

    #[test]
    fn test_estimate_flops() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[4], ctx);
            let b = T::ones(&[4], ctx);
            let _ = a + b;

            let flop_estimates = estimate_flops(ctx.as_graph());
            assert!(!flop_estimates.is_empty());
            // Compute nodes (add) should have non-zero FLOPs
            let compute_flops: u64 = flop_estimates
                .iter()
                .filter(|e| e.op_name.contains("Add"))
                .map(|e| e.flops)
                .sum();
            assert!(compute_flops > 0, "AddOp should have non-zero FLOPs");
            // Total FLOPs should be non-zero
            let total: u64 = flop_estimates.iter().map(|e| e.flops).sum();
            assert!(total > 0);
        });
    }

    #[test]
    fn test_total_flops() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[4], ctx);
            let b = T::ones(&[4], ctx);
            let _ = a + b;

            let flops = total_flops(ctx.as_graph());
            assert!(flops > 0, "Non-trivial graph should have > 0 FLOPs");
        });
    }

    #[test]
    fn test_graph_complexity() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[2], ctx);
            let b = T::ones(&[2], ctx);
            let c = a + b;
            let d = a * b;
            let _ = c + d;

            let cx = graph_complexity(ctx.as_graph());
            assert!(cx.num_nodes > 0);
            assert!(cx.num_edges > 0);
            assert!(cx.max_depth >= 1);
            assert!(cx.max_width >= 1);
            assert!(cx.num_op_types >= 2);
        });
    }

    #[test]
    fn test_graph_complexity_empty() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let cx = graph_complexity(ctx.as_graph());
            assert_eq!(cx.num_nodes, 0);
            assert_eq!(cx.num_edges, 0);
        });
    }

    #[test]
    fn test_gradient_classification() {
        let thresholds = GradientThresholds::default();
        assert_eq!(
            classify_gradient(1e-10, &thresholds),
            GradientHealth::Vanishing
        );
        assert_eq!(
            classify_gradient(0.01, &thresholds),
            GradientHealth::Healthy
        );
        assert_eq!(
            classify_gradient(1e5, &thresholds),
            GradientHealth::Exploding
        );
    }

    #[test]
    fn test_gradient_flow_analysis() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[2], ctx);
            let b = T::ones(&[2], ctx);
            let _ = a + b;

            let mut grad_mags: HashMap<TensorID, (f64, f64, f64)> = HashMap::new();
            grad_mags.insert(0, (0.01, 0.02, 0.005));
            grad_mags.insert(1, (1e-10, 1e-10, 1e-10)); // vanishing

            let thresholds = GradientThresholds::default();
            let stats = analyse_gradient_flow(ctx.as_graph(), &grad_mags, &thresholds);

            assert!(!stats.is_empty());
            assert!(has_gradient_issues(&stats));
        });
    }

    #[test]
    fn test_no_gradient_issues() {
        let stats = vec![GradientFlowStats {
            node_id: 0,
            op_name: "add".to_owned(),
            mean_abs_grad: Some(0.1),
            max_abs_grad: Some(0.5),
            min_abs_grad: Some(0.01),
            health: GradientHealth::Healthy,
        }];
        assert!(!has_gradient_issues(&stats));
    }

    #[test]
    fn test_operation_profiler() {
        let mut profiler = OperationProfiler::new();
        assert_eq!(profiler.num_records(), 0);

        profiler.record(0, "add", Duration::from_micros(100));
        profiler.record(1, "mul", Duration::from_micros(200));
        profiler.record(2, "add", Duration::from_micros(50));

        assert_eq!(profiler.num_records(), 3);
        assert_eq!(profiler.total_time(), Duration::from_micros(350));

        let slowest = profiler.slowest_ops(1);
        assert_eq!(slowest[0].op_name, "mul");

        let by_type = profiler.time_by_op_type();
        assert_eq!(by_type.get("add"), Some(&Duration::from_micros(150)));
        assert_eq!(by_type.get("mul"), Some(&Duration::from_micros(200)));
    }

    #[test]
    fn test_profiler_start_end() {
        let mut profiler = OperationProfiler::new();
        profiler.start_op(0, "matmul");
        // Simulate some work
        std::thread::sleep(Duration::from_millis(1));
        profiler.end_op();

        assert_eq!(profiler.num_records(), 1);
        assert!(profiler.timings()[0].duration >= Duration::from_millis(1));
    }

    #[test]
    fn test_profiler_clear() {
        let mut profiler = OperationProfiler::new();
        profiler.record(0, "add", Duration::from_micros(10));
        assert_eq!(profiler.num_records(), 1);
        profiler.clear();
        assert_eq!(profiler.num_records(), 0);
    }

    #[test]
    fn test_estimate_bandwidth() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[4], ctx);
            let b = T::ones(&[4], ctx);
            let _ = a + b;

            let bw = estimate_bandwidth(ctx.as_graph(), 4, 1024);
            assert!(!bw.is_empty());
            // Compute nodes should have non-zero bytes
            let compute_bw: u64 = bw
                .iter()
                .filter(|b| b.bytes_read > 0)
                .map(|b| b.total_bytes)
                .sum();
            assert!(compute_bw > 0);
        });
    }

    #[test]
    fn test_profile_graph_integration() {
        let env = VariableEnvironment::<f32>::new();
        env.run(|ctx| {
            let a = T::zeros(&[4, 4], ctx);
            let b = T::ones(&[4, 4], ctx);
            let c = a + b;
            let _ = c * T::ones(&[4, 4], ctx);

            let report = profile_graph(ctx.as_graph());
            assert!(report.op_counts.total > 0);
            assert!(report.total_flops > 0);
            assert!(report.complexity.num_nodes > 0);
        });
    }

    #[test]
    fn test_flops_per_element_known() {
        let (f, c) = flops_per_element("AddOp");
        assert_eq!(f, 1);
        assert_eq!(c, EstimateConfidence::Exact);

        let (f, c) = flops_per_element("Sigmoid");
        assert_eq!(f, 4);
        assert_eq!(c, EstimateConfidence::Heuristic);
    }
}
