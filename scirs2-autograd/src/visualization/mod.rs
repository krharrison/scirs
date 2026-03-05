//! Computation graph visualization and analysis tools
//!
//! This module provides tools for:
//! - **DOT format export** (`graph_to_dot`) for Graphviz visualization
//! - **Text/JSON/Mermaid export** for various rendering backends
//! - **Graph statistics** (node counts, depth, memory estimates)
//! - **Structure analysis** (critical path, fan-in/fan-out)
//!
//! # Examples
//!
//! ## DOT export
//!
//! ```rust
//! use scirs2_autograd as ag;
//! use ag::tensor_ops;
//! use ag::visualization::{graph_to_dot, GraphStats};
//!
//! ag::run(|ctx: &mut ag::Context<f64>| {
//!     let x = ctx.placeholder("x", &[3]);
//!     let y = x * 2.0 + 1.0;
//!     let loss = tensor_ops::reduction::sum_all(y);
//!
//!     let dot = graph_to_dot(&loss, ctx);
//!     // Can be rendered with: dot -Tpng graph.dot -o graph.png
//!
//!     let stats = GraphStats::from_tensor(&loss, ctx);
//!     assert!(stats.total_nodes > 0);
//! });
//! ```

use crate::graph::{Graph, TensorID};
use crate::tensor::Tensor;
use crate::{Context, Float};
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

// ---------------------------------------------------------------------------
// Graph traversal helpers
// ---------------------------------------------------------------------------

/// Collect all reachable node IDs from a root tensor by traversing backwards
/// through incoming edges. Returns nodes in topological order (dependencies first).
fn collect_reachable_nodes<F: Float>(root_id: TensorID, graph: &Graph<F>) -> Vec<TensorID> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    collect_dfs(root_id, graph, &mut visited, &mut order);
    order
}

fn collect_dfs<F: Float>(
    node_id: TensorID,
    graph: &Graph<F>,
    visited: &mut HashSet<TensorID>,
    order: &mut Vec<TensorID>,
) {
    if visited.contains(&node_id) {
        return;
    }
    visited.insert(node_id);

    let node = graph.access_inner(node_id);
    let incoming = node.incoming_nodes.clone();
    drop(node); // release borrow

    for inc in &incoming {
        collect_dfs(inc.id, graph, visited, order);
    }

    order.push(node_id);
}

/// Information extracted from a single graph node.
struct NodeInfo {
    id: TensorID,
    op_name: String,
    topo_rank: usize,
    is_differentiable: bool,
    is_placeholder: bool,
    placeholder_name: Option<String>,
    is_variable: bool,
    num_inputs: usize,
    input_ids: Vec<TensorID>,
    known_shape: Option<Vec<isize>>,
}

fn extract_node_info<F: Float>(node_id: TensorID, graph: &Graph<F>) -> NodeInfo {
    let node = graph.access_inner(node_id);
    let op_name = node
        .op
        .as_ref()
        .map(|o| {
            let full = o.name();
            // Extract the short name (last segment after ::)
            full.rsplit("::").next().unwrap_or(full).to_string()
        })
        .unwrap_or_else(|| "Source".to_string());

    let input_ids: Vec<TensorID> = node.incoming_nodes.iter().map(|inc| inc.id).collect();
    let num_inputs = input_ids.len();

    let known_shape = node.knownshape.as_ref().map(|ks| ks.get().to_vec());

    NodeInfo {
        id: node_id,
        op_name,
        topo_rank: node.topo_rank,
        is_differentiable: node.is_differentiable,
        is_placeholder: node.placeholder_name.is_some(),
        placeholder_name: node.placeholder_name.map(|s| s.to_string()),
        is_variable: node.variable_id.is_some(),
        num_inputs,
        input_ids,
        known_shape,
    }
}

// ---------------------------------------------------------------------------
// DOT export
// ---------------------------------------------------------------------------

/// Export the computation graph reachable from `root` as a Graphviz DOT string.
///
/// The DOT output can be rendered to PNG/SVG/PDF using:
/// ```bash
/// echo "<dot output>" | dot -Tpng -o graph.png
/// ```
///
/// Node colors:
/// - **lightgreen**: placeholders (input data)
/// - **lightyellow**: variables (trainable parameters)
/// - **lightblue**: differentiable operations
/// - **lightgray**: non-differentiable operations
///
/// # Arguments
/// * `root` - The root tensor (typically the loss or final output)
/// * `ctx` - The autograd context
///
/// # Returns
/// A string in Graphviz DOT format.
pub fn graph_to_dot<'g, F: Float>(root: &Tensor<'g, F>, ctx: &'g Context<'g, F>) -> String {
    let graph = get_graph(ctx);
    let nodes = collect_reachable_nodes(root.id(), graph);
    let node_set: HashSet<TensorID> = nodes.iter().copied().collect();

    let mut output = String::new();
    let _ = writeln!(output, "digraph computation_graph {{");
    let _ = writeln!(output, "  rankdir=BT;");
    let _ = writeln!(
        output,
        "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\"];"
    );
    let _ = writeln!(output, "  edge [color=gray50];");
    let _ = writeln!(output);

    // Emit nodes
    for &nid in &nodes {
        let info = extract_node_info(nid, graph);
        let label = node_label(&info);
        let style = node_color(&info);
        let _ = writeln!(output, "  n{nid} [label=\"{label}\", {style}];");
    }

    let _ = writeln!(output);

    // Emit edges
    for &nid in &nodes {
        let info = extract_node_info(nid, graph);
        for &src in &info.input_ids {
            if node_set.contains(&src) {
                let _ = writeln!(output, "  n{src} -> n{nid};");
            }
        }
    }

    let _ = writeln!(output, "}}");
    output
}

fn node_label(info: &NodeInfo) -> String {
    let mut label = String::new();

    if let Some(ref name) = info.placeholder_name {
        let _ = write!(label, "{name}\\n");
    }

    let _ = write!(label, "{}", info.op_name);

    if let Some(ref shape) = info.known_shape {
        let _ = write!(label, "\\n{shape:?}");
    }

    let _ = write!(label, "\\n(id={})", info.id);
    label
}

fn node_color(info: &NodeInfo) -> String {
    if info.is_placeholder {
        "fillcolor=\"#d5f5d5\"".to_string() // light green
    } else if info.is_variable {
        "fillcolor=\"#fff8d5\"".to_string() // light yellow
    } else if info.is_differentiable {
        "fillcolor=\"#d5e8f5\"".to_string() // light blue
    } else {
        "fillcolor=\"#e8e8e8\"".to_string() // light gray
    }
}

// ---------------------------------------------------------------------------
// Text summary
// ---------------------------------------------------------------------------

/// Generate a compact text summary of the computation graph.
///
/// # Arguments
/// * `root` - The root tensor
/// * `ctx` - The autograd context
pub fn graph_summary<'g, F: Float>(root: &Tensor<'g, F>, ctx: &'g Context<'g, F>) -> String {
    let graph = get_graph(ctx);
    let nodes = collect_reachable_nodes(root.id(), graph);

    let mut output = String::new();
    let _ = writeln!(output, "Computation Graph Summary");
    let _ = writeln!(output, "=========================");
    let _ = writeln!(output, "Total nodes: {}", nodes.len());

    let mut placeholders = 0usize;
    let mut variables = 0usize;
    let mut ops = 0usize;
    let mut max_rank = 0usize;
    let mut op_counts: HashMap<String, usize> = HashMap::new();

    for &nid in &nodes {
        let info = extract_node_info(nid, graph);
        if info.is_placeholder {
            placeholders += 1;
        } else if info.is_variable {
            variables += 1;
        } else {
            ops += 1;
        }
        if info.topo_rank > max_rank {
            max_rank = info.topo_rank;
        }
        *op_counts.entry(info.op_name.clone()).or_insert(0) += 1;
    }

    let _ = writeln!(output, "  Placeholders: {placeholders}");
    let _ = writeln!(output, "  Variables: {variables}");
    let _ = writeln!(output, "  Operations: {ops}");
    let _ = writeln!(output, "  Max depth (topo rank): {max_rank}");
    let _ = writeln!(output);

    // Sort op counts by frequency
    let mut sorted_ops: Vec<_> = op_counts.into_iter().collect();
    sorted_ops.sort_by(|a, b| b.1.cmp(&a.1));

    let _ = writeln!(output, "Operation breakdown:");
    for (name, count) in &sorted_ops {
        let _ = writeln!(output, "  {name}: {count}");
    }

    output
}

// ---------------------------------------------------------------------------
// JSON export
// ---------------------------------------------------------------------------

/// Export the computation graph as a JSON string.
///
/// The JSON has the following structure:
/// ```json
/// {
///   "nodes": [{"id": 0, "op": "Placeholder", "rank": 0, ...}, ...],
///   "edges": [{"from": 0, "to": 1}, ...]
/// }
/// ```
pub fn graph_to_json<'g, F: Float>(root: &Tensor<'g, F>, ctx: &'g Context<'g, F>) -> String {
    let graph = get_graph(ctx);
    let nodes = collect_reachable_nodes(root.id(), graph);
    let node_set: HashSet<TensorID> = nodes.iter().copied().collect();

    let mut output = String::new();
    let _ = writeln!(output, "{{");
    let _ = writeln!(output, "  \"nodes\": [");

    for (idx, &nid) in nodes.iter().enumerate() {
        let info = extract_node_info(nid, graph);
        let comma = if idx + 1 < nodes.len() { "," } else { "" };
        let shape_str = info
            .known_shape
            .as_ref()
            .map(|s| format!("{s:?}"))
            .unwrap_or_else(|| "null".to_string());
        let _ = writeln!(
            output,
            "    {{\"id\": {}, \"op\": \"{}\", \"rank\": {}, \"differentiable\": {}, \"shape\": {}}}{}",
            info.id, info.op_name, info.topo_rank, info.is_differentiable, shape_str, comma
        );
    }

    let _ = writeln!(output, "  ],");
    let _ = writeln!(output, "  \"edges\": [");

    let mut edge_idx = 0usize;
    let total_edges: usize = nodes
        .iter()
        .map(|&nid| {
            let info = extract_node_info(nid, graph);
            info.input_ids
                .iter()
                .filter(|id| node_set.contains(id))
                .count()
        })
        .sum();

    for &nid in &nodes {
        let info = extract_node_info(nid, graph);
        for &src in &info.input_ids {
            if node_set.contains(&src) {
                edge_idx += 1;
                let comma = if edge_idx < total_edges { "," } else { "" };
                let _ = writeln!(
                    output,
                    "    {{\"from\": {src}, \"to\": {}}}{comma}",
                    info.id
                );
            }
        }
    }

    let _ = writeln!(output, "  ]");
    let _ = writeln!(output, "}}");
    output
}

// ---------------------------------------------------------------------------
// Mermaid export
// ---------------------------------------------------------------------------

/// Export the computation graph as a Mermaid diagram string.
///
/// Can be embedded in Markdown or rendered at <https://mermaid.live>.
pub fn graph_to_mermaid<'g, F: Float>(root: &Tensor<'g, F>, ctx: &'g Context<'g, F>) -> String {
    let graph = get_graph(ctx);
    let nodes = collect_reachable_nodes(root.id(), graph);
    let node_set: HashSet<TensorID> = nodes.iter().copied().collect();

    let mut output = String::new();
    let _ = writeln!(output, "graph BT");

    for &nid in &nodes {
        let info = extract_node_info(nid, graph);
        let label = if let Some(ref name) = info.placeholder_name {
            format!("{name}: {}", info.op_name)
        } else {
            info.op_name.clone()
        };
        let _ = writeln!(output, "  N{nid}[\"{label}\"]");
    }

    for &nid in &nodes {
        let info = extract_node_info(nid, graph);
        for &src in &info.input_ids {
            if node_set.contains(&src) {
                let _ = writeln!(output, "  N{src} --> N{nid}");
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Graph statistics
// ---------------------------------------------------------------------------

/// Statistics about a computation graph.
#[derive(Debug, Clone)]
pub struct GraphStats {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Number of placeholder (input) nodes
    pub num_placeholders: usize,
    /// Number of variable (trainable parameter) nodes
    pub num_variables: usize,
    /// Number of operation nodes
    pub num_operations: usize,
    /// Maximum topological rank (depth of the graph)
    pub max_depth: usize,
    /// Number of edges (connections between nodes)
    pub num_edges: usize,
    /// Number of differentiable nodes
    pub num_differentiable: usize,
    /// Operation type breakdown: (op_name, count)
    pub op_breakdown: Vec<(String, usize)>,
    /// Maximum fan-in (most inputs to a single node)
    pub max_fan_in: usize,
    /// Maximum fan-out (most nodes that depend on a single node)
    pub max_fan_out: usize,
}

impl GraphStats {
    /// Compute graph statistics from a root tensor.
    pub fn from_tensor<'g, F: Float>(root: &Tensor<'g, F>, ctx: &'g Context<'g, F>) -> Self {
        let graph = get_graph(ctx);
        let nodes = collect_reachable_nodes(root.id(), graph);

        let mut num_placeholders = 0usize;
        let mut num_variables = 0usize;
        let mut num_operations = 0usize;
        let mut num_differentiable = 0usize;
        let mut max_depth = 0usize;
        let mut num_edges = 0usize;
        let mut max_fan_in = 0usize;
        let mut op_counts: HashMap<String, usize> = HashMap::new();
        let mut fan_out: HashMap<TensorID, usize> = HashMap::new();

        for &nid in &nodes {
            let info = extract_node_info(nid, graph);

            if info.is_placeholder {
                num_placeholders += 1;
            } else if info.is_variable {
                num_variables += 1;
            } else {
                num_operations += 1;
            }

            if info.is_differentiable {
                num_differentiable += 1;
            }

            if info.topo_rank > max_depth {
                max_depth = info.topo_rank;
            }

            num_edges += info.num_inputs;

            if info.num_inputs > max_fan_in {
                max_fan_in = info.num_inputs;
            }

            for &src in &info.input_ids {
                *fan_out.entry(src).or_insert(0) += 1;
            }

            *op_counts.entry(info.op_name).or_insert(0) += 1;
        }

        let max_fan_out = fan_out.values().copied().max().unwrap_or(0);

        let mut op_breakdown: Vec<_> = op_counts.into_iter().collect();
        op_breakdown.sort_by(|a, b| b.1.cmp(&a.1));

        GraphStats {
            total_nodes: nodes.len(),
            num_placeholders,
            num_variables,
            num_operations,
            max_depth,
            num_edges,
            num_differentiable,
            op_breakdown,
            max_fan_in,
            max_fan_out,
        }
    }

    /// Format the statistics as a human-readable string.
    pub fn display(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(output, "Graph Statistics");
        let _ = writeln!(output, "================");
        let _ = writeln!(output, "Total nodes:      {}", self.total_nodes);
        let _ = writeln!(output, "Placeholders:     {}", self.num_placeholders);
        let _ = writeln!(output, "Variables:        {}", self.num_variables);
        let _ = writeln!(output, "Operations:       {}", self.num_operations);
        let _ = writeln!(output, "Edges:            {}", self.num_edges);
        let _ = writeln!(output, "Max depth:        {}", self.max_depth);
        let _ = writeln!(output, "Differentiable:   {}", self.num_differentiable);
        let _ = writeln!(output, "Max fan-in:       {}", self.max_fan_in);
        let _ = writeln!(output, "Max fan-out:      {}", self.max_fan_out);
        let _ = writeln!(output);
        let _ = writeln!(output, "Operation breakdown:");
        for (name, count) in &self.op_breakdown {
            let _ = writeln!(output, "  {name}: {count}");
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for graph visualization.
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Whether to show tensor shapes in nodes
    pub show_shapes: bool,
    /// Whether to show operation names
    pub show_operations: bool,
    /// Whether to show gradient flow
    pub show_gradients: bool,
    /// Maximum number of nodes to display
    pub max_nodes: Option<usize>,
    /// Output format
    pub format: OutputFormat,
    /// Whether to include values (for small tensors)
    pub show_values: bool,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            show_shapes: true,
            show_operations: true,
            show_gradients: false,
            max_nodes: Some(100),
            format: OutputFormat::Dot,
            show_values: false,
        }
    }
}

/// Output format for graph visualization.
#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    /// Graphviz DOT format
    Dot,
    /// Simple text representation
    Text,
    /// JSON format for web visualization
    Json,
    /// Mermaid diagram format
    Mermaid,
}

/// Errors that can occur during visualization.
#[derive(Debug, thiserror::Error)]
pub enum VisualizationError {
    #[error("Graph traversal error: {0}")]
    GraphTraversal(String),
    #[error("Format error: {0}")]
    Format(#[from] std::fmt::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid configuration: {0}")]
    Config(String),
}

// ---------------------------------------------------------------------------
// Helper: get Graph from Context via Deref
// ---------------------------------------------------------------------------

/// Get the underlying Graph from a Context via Deref.
fn get_graph<'g, F: Float>(ctx: &'g Context<'g, F>) -> &'g Graph<F> {
    use std::ops::Deref;
    ctx.deref()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops;

    #[test]
    fn test_graph_to_dot_basic() {
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[3]);
            let y = x * 2.0;
            let loss = crate::tensor_ops::reduction::sum_all(y);

            let dot = graph_to_dot(&loss, ctx);
            assert!(dot.contains("digraph computation_graph"));
            assert!(dot.contains("->"));
            assert!(dot.contains("}"));
        });
    }

    #[test]
    fn test_graph_to_dot_multi_input() {
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let y = ctx.placeholder("y", &[2]);
            let z = x + y;
            let loss = crate::tensor_ops::reduction::sum_all(z);

            let dot = graph_to_dot(&loss, ctx);
            assert!(dot.contains("digraph computation_graph"));
            // Should have nodes for x, y, z, and loss
            assert!(dot.contains("fillcolor"));
        });
    }

    #[test]
    fn test_graph_summary_basic() {
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[3]);
            let y = x * 2.0 + 1.0;
            let loss = crate::tensor_ops::reduction::sum_all(y);

            let summary = graph_summary(&loss, ctx);
            assert!(summary.contains("Computation Graph Summary"));
            assert!(summary.contains("Total nodes:"));
            assert!(summary.contains("Placeholders:"));
        });
    }

    #[test]
    fn test_graph_to_json() {
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let y = x * 3.0;

            let json = graph_to_json(&y, ctx);
            assert!(json.contains("\"nodes\""));
            assert!(json.contains("\"edges\""));
            assert!(json.contains("\"op\""));
        });
    }

    #[test]
    fn test_graph_to_mermaid() {
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let y = x * 2.0;

            let mermaid = graph_to_mermaid(&y, ctx);
            assert!(mermaid.contains("graph BT"));
            assert!(mermaid.contains("-->"));
        });
    }

    #[test]
    fn test_graph_stats() {
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[3]);
            let y = x * 2.0 + 1.0;
            let loss = crate::tensor_ops::reduction::sum_all(y);

            let stats = GraphStats::from_tensor(&loss, ctx);
            assert!(stats.total_nodes > 0);
            assert!(stats.num_placeholders >= 1);
            assert!(stats.num_edges > 0);
            assert!(stats.max_depth > 0);
            assert!(!stats.op_breakdown.is_empty());
        });
    }

    #[test]
    fn test_graph_stats_display() {
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[3]);
            let loss = crate::tensor_ops::reduction::sum_all(x * x);

            let stats = GraphStats::from_tensor(&loss, ctx);
            let display = stats.display();
            assert!(display.contains("Graph Statistics"));
            assert!(display.contains("Total nodes:"));
            assert!(display.contains("Max fan-in:"));
            assert!(display.contains("Max fan-out:"));
        });
    }

    #[test]
    fn test_graph_dot_colors() {
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let y = x * 2.0;

            let dot = graph_to_dot(&y, ctx);
            // Should contain color codes for different node types
            assert!(dot.contains("fillcolor"));
        });
    }

    #[test]
    fn test_visualization_config_default() {
        let config = VisualizationConfig::default();
        assert!(config.show_shapes);
        assert!(config.show_operations);
        assert!(!config.show_gradients);
        assert_eq!(config.max_nodes, Some(100));
        assert!(matches!(config.format, OutputFormat::Dot));
    }

    #[test]
    fn test_graph_stats_single_node() {
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[]);

            let stats = GraphStats::from_tensor(&x, ctx);
            assert_eq!(stats.total_nodes, 1);
            assert_eq!(stats.num_placeholders, 1);
            assert_eq!(stats.num_operations, 0);
            assert_eq!(stats.num_edges, 0);
        });
    }

    #[test]
    fn test_collect_reachable_shared_nodes() {
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            // x is used twice -> shared node
            let y = x + x;
            let loss = crate::tensor_ops::reduction::sum_all(y);

            let graph: &Graph<f64> = std::ops::Deref::deref(ctx);
            let nodes = collect_reachable_nodes(loss.id(), graph);
            // x should appear only once despite being used twice
            let x_count = nodes.iter().filter(|&&id| id == x.id()).count();
            assert_eq!(x_count, 1);
        });
    }
}
