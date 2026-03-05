//! Computation graph visualization and debugging utilities
//!
//! This module provides:
//! - [`ComputationGraphViz`]: DOT language / Mermaid graph generator
//! - [`export_dot`]: write Graphviz DOT file to disk
//! - [`export_mermaid`]: write Mermaid diagram file to disk
//! - [`print_graph_summary`]: Keras-style model summary string
//! - [`count_parameters`]: total trainable parameter count
//! - [`count_flops`]: rough FLOPs estimate given an input shape
//!
//! # Examples
//!
//! ```rust
//! use scirs2_autograd as ag;
//! use ag::tensor_ops;
//! use ag::graph_viz::{ComputationGraphViz, print_graph_summary, count_parameters};
//!
//! ag::run(|ctx: &mut ag::Context<f64>| {
//!     let x = ctx.placeholder("x", &[3]);
//!     let y = x * 2.0 + 1.0;
//!     let loss = tensor_ops::reduction::sum_all(y);
//!
//!     let viz = ComputationGraphViz::from_tensor(&loss, ctx);
//!     let dot_src = viz.to_dot();
//!     assert!(dot_src.contains("digraph"));
//!
//!     let summary = print_graph_summary(&loss, ctx);
//!     assert!(summary.contains("Total parameters"));
//!
//!     let params = count_parameters(&loss, ctx);
//!     let _ = params; // 0 for this graph (no variables)
//! });
//! ```

use crate::graph::{Graph, TensorID};
use crate::tensor::Tensor;
use crate::{Context, Float};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as FmtWrite;

// ─────────────────────────────────────────────────────────────────────────────
// Internal graph traversal helpers (mirror visualization::mod.rs approach)
// ─────────────────────────────────────────────────────────────────────────────

fn collect_reachable<F: Float>(root: TensorID, graph: &Graph<F>) -> Vec<TensorID> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    dfs_collect(root, graph, &mut visited, &mut order);
    order
}

fn dfs_collect<F: Float>(
    id: TensorID,
    graph: &Graph<F>,
    visited: &mut HashSet<TensorID>,
    order: &mut Vec<TensorID>,
) {
    if !visited.insert(id) {
        return;
    }
    let node = graph.access_inner(id);
    let incoming: Vec<TensorID> = node.incoming_nodes.iter().map(|inc| inc.id).collect();
    drop(node);
    for dep in incoming {
        dfs_collect(dep, graph, visited, order);
    }
    order.push(id);
}

/// Lightweight description of a single graph node.
#[derive(Debug, Clone)]
pub struct NodeDesc {
    id: TensorID,
    op_name: String,
    topo_rank: usize,
    is_differentiable: bool,
    is_placeholder: bool,
    placeholder_name: Option<String>,
    is_variable: bool,
    known_shape: Option<Vec<isize>>,
    input_ids: Vec<TensorID>,
}

fn describe_node<F: Float>(id: TensorID, graph: &Graph<F>) -> NodeDesc {
    let node = graph.access_inner(id);
    let op_name = node
        .op
        .as_ref()
        .map(|o| {
            let full = o.name();
            full.rsplit("::").next().unwrap_or(full).to_string()
        })
        .unwrap_or_else(|| "Source".to_string());

    let input_ids: Vec<TensorID> = node.incoming_nodes.iter().map(|inc| inc.id).collect();
    let known_shape = node.knownshape.as_ref().map(|ks| ks.get().to_vec());
    let placeholder_name = node.placeholder_name.map(|s| s.to_string());
    let is_placeholder = node.placeholder_name.is_some();
    let is_variable = node.variable_id.is_some();
    let is_differentiable = node.is_differentiable;
    let topo_rank = node.topo_rank;
    drop(node);

    NodeDesc {
        id,
        op_name,
        topo_rank,
        is_differentiable,
        is_placeholder,
        placeholder_name,
        is_variable,
        known_shape,
        input_ids,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ComputationGraphViz
// ─────────────────────────────────────────────────────────────────────────────

/// Computation graph visualization builder.
///
/// Collects all nodes reachable from a root tensor and can render them in
/// multiple formats (DOT, Mermaid) or produce structured metadata.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd as ag;
/// use ag::graph_viz::ComputationGraphViz;
///
/// ag::run(|ctx: &mut ag::Context<f64>| {
///     let x = ctx.placeholder("x", &[4]);
///     let y = x * 3.0;
///     let viz = ComputationGraphViz::from_tensor(&y, ctx);
///     let dot = viz.to_dot();
///     assert!(dot.contains("digraph"));
/// });
/// ```
#[derive(Debug, Clone)]
pub struct ComputationGraphViz {
    nodes: Vec<NodeDesc>,
}

impl ComputationGraphViz {
    /// Build the visualization state from a root tensor.
    pub fn from_tensor<'a, 'g, F: Float>(root: &Tensor<'g, F>, ctx: &'a Context<'g, F>) -> Self {
        let graph: &Graph<F> = std::ops::Deref::deref(ctx);
        let ids = collect_reachable(root.id(), graph);
        let nodes = ids.iter().map(|&id| describe_node(id, graph)).collect();
        Self { nodes }
    }

    // ── DOT ──────────────────────────────────────────────────────────────────

    /// Render the graph as a Graphviz DOT string.
    ///
    /// Node shape and color conventions:
    /// - light green  → placeholder (input data)
    /// - light yellow → variable (trainable parameter)
    /// - light blue   → differentiable operation
    /// - light gray   → non-differentiable operation
    pub fn to_dot(&self) -> String {
        let node_set: HashSet<TensorID> = self.nodes.iter().map(|n| n.id).collect();
        let mut out = String::new();

        let _ = writeln!(out, "digraph computation_graph {{");
        let _ = writeln!(out, "  rankdir=BT;");
        let _ = writeln!(
            out,
            "  node [shape=record, style=\"rounded,filled\", fontname=\"Helvetica\", fontsize=10];"
        );
        let _ = writeln!(out, "  edge [color=gray50, fontsize=8];");
        let _ = writeln!(out);

        // Emit node records
        for nd in &self.nodes {
            let label = self.node_to_dot(nd);
            let color = dot_color(nd);
            let _ = writeln!(out, "  n{} [label=\"{}\", {}];", nd.id, label, color);
        }

        let _ = writeln!(out);

        // Emit edges
        for nd in &self.nodes {
            for &src in &nd.input_ids {
                if node_set.contains(&src) {
                    let src_desc = self.nodes.iter().find(|n| n.id == src);
                    let edge_label = if let Some(s) = src_desc {
                        s.known_shape
                            .as_ref()
                            .map(|sh| format!("{sh:?}"))
                            .unwrap_or_default()
                    } else {
                        String::new()
                    };
                    if edge_label.is_empty() {
                        let _ = writeln!(out, "  n{src} -> n{};", nd.id);
                    } else {
                        let _ = writeln!(
                            out,
                            "  n{src} -> n{} [label=\"{}\"];",
                            nd.id, edge_label
                        );
                    }
                }
            }
        }

        let _ = writeln!(out, "}}");
        out
    }

    /// Format a single node as a DOT record label.
    ///
    /// The label includes:
    /// - op name
    /// - output shape (if known)
    /// - placeholder or variable annotation
    pub fn node_to_dot(&self, nd: &NodeDesc) -> String {
        let mut parts = Vec::new();

        // Header: op name with annotation
        let mut header = nd.op_name.clone();
        if let Some(ref pname) = nd.placeholder_name {
            header = format!("{pname} | {header}");
        } else if nd.is_variable {
            header = format!("var | {header}");
        }
        parts.push(header);

        // Shape line
        if let Some(ref sh) = nd.known_shape {
            parts.push(format!("shape: {sh:?}"));
        }

        // Grad indicator
        if nd.is_differentiable {
            parts.push("grad".to_string());
        }

        // Node id
        parts.push(format!("id={}", nd.id));

        parts.join(" | ")
    }

    // ── Mermaid ───────────────────────────────────────────────────────────────

    /// Render the graph as a Mermaid diagram string.
    ///
    /// The output can be embedded in Markdown or pasted into <https://mermaid.live>.
    pub fn to_mermaid(&self) -> String {
        let node_set: HashSet<TensorID> = self.nodes.iter().map(|n| n.id).collect();
        let mut out = String::new();
        let _ = writeln!(out, "graph BT");

        for nd in &self.nodes {
            let label = if let Some(ref pname) = nd.placeholder_name {
                format!("{pname}: {}", nd.op_name)
            } else if nd.is_variable {
                format!("var: {}", nd.op_name)
            } else {
                nd.op_name.clone()
            };
            let shape_str = nd
                .known_shape
                .as_ref()
                .map(|sh| format!("<br/>{sh:?}"))
                .unwrap_or_default();
            let _ = writeln!(out, "  N{}[\"{}{}\"]", nd.id, label, shape_str);
        }

        for nd in &self.nodes {
            for &src in &nd.input_ids {
                if node_set.contains(&src) {
                    let _ = writeln!(out, "  N{src} --> N{}", nd.id);
                }
            }
        }

        out
    }

    /// Total number of nodes in this view.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Nodes in topological order (dependencies first).
    pub fn nodes(&self) -> &[NodeDesc] {
        &self.nodes
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// File export helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Export the computation graph reachable from `root` as a Graphviz DOT file.
///
/// # Arguments
/// * `root`     - The root tensor (e.g. the loss)
/// * `ctx`      - The autograd context
/// * `filename` - Path to write the `.dot` file
///
/// # Errors
/// Returns an `std::io::Error` if the file cannot be written.
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_autograd as ag;
/// use ag::tensor_ops;
/// use ag::graph_viz::export_dot;
///
/// ag::run(|ctx: &mut ag::Context<f64>| {
///     let x = ctx.placeholder("x", &[3]);
///     let loss = tensor_ops::reduction::sum_all(x * 2.0);
///     export_dot(&loss, ctx, "/tmp/graph.dot").expect("write failed");
/// });
/// ```
pub fn export_dot<'g, F: Float>(
    root: &Tensor<'g, F>,
    ctx: &'g Context<'g, F>,
    filename: &str,
) -> std::io::Result<()> {
    let viz = ComputationGraphViz::from_tensor(root, ctx);
    let content = viz.to_dot();
    std::fs::write(filename, content)
}

/// Export the computation graph reachable from `root` as a Mermaid diagram file.
///
/// # Errors
/// Returns an `std::io::Error` if the file cannot be written.
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_autograd as ag;
/// use ag::tensor_ops;
/// use ag::graph_viz::export_mermaid;
///
/// ag::run(|ctx: &mut ag::Context<f64>| {
///     let x = ctx.placeholder("x", &[3]);
///     let loss = tensor_ops::reduction::sum_all(x * 2.0);
///     export_mermaid(&loss, ctx, "/tmp/graph.md").expect("write failed");
/// });
/// ```
pub fn export_mermaid<'g, F: Float>(
    root: &Tensor<'g, F>,
    ctx: &'g Context<'g, F>,
    filename: &str,
) -> std::io::Result<()> {
    let viz = ComputationGraphViz::from_tensor(root, ctx);
    let mermaid = viz.to_mermaid();
    let content = format!("```mermaid\n{mermaid}```\n");
    std::fs::write(filename, content)
}

// ─────────────────────────────────────────────────────────────────────────────
// Text summary (Keras model.summary() style)
// ─────────────────────────────────────────────────────────────────────────────

/// Print a Keras-style model summary for the computation graph.
///
/// The summary includes per-layer information (op name, output shape,
/// approximate parameter count) and totals.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd as ag;
/// use ag::tensor_ops;
/// use ag::graph_viz::print_graph_summary;
///
/// ag::run(|ctx: &mut ag::Context<f64>| {
///     let x = ctx.placeholder("x", &[3]);
///     let y = x * 2.0 + 1.0;
///     let loss = tensor_ops::reduction::sum_all(y);
///     let s = print_graph_summary(&loss, ctx);
///     assert!(s.contains("Total parameters"));
/// });
/// ```
pub fn print_graph_summary<'a, 'g, F: Float>(
    root: &Tensor<'g, F>,
    ctx: &'a Context<'g, F>,
) -> String {
    let graph: &Graph<F> = std::ops::Deref::deref(ctx);
    let ids = collect_reachable(root.id(), graph);

    let mut out = String::new();
    let sep = "─".repeat(70);

    let _ = writeln!(out, "{sep}");
    let _ = writeln!(
        out,
        " {:<30} {:<20} {:<10} {}",
        "Layer (op)", "Output Shape", "Params", "Grad"
    );
    let _ = writeln!(out, "{sep}");

    let mut total_params: usize = 0;
    let mut trainable_params: usize = 0;

    for &id in &ids {
        let nd = describe_node(id, graph);

        let op_col = if let Some(ref pname) = nd.placeholder_name {
            format!("{pname} ({})", nd.op_name)
        } else {
            nd.op_name.clone()
        };

        let shape_col = nd
            .known_shape
            .as_ref()
            .map(|sh| format!("{sh:?}"))
            .unwrap_or_else(|| "?".to_string());

        // Estimate parameter count for variable nodes using known shape.
        // For non-variable nodes we have no reliable count, so report 0.
        let params: usize = if nd.is_variable {
            nd.known_shape
                .as_ref()
                .map(|sh| sh.iter().map(|&d| d.max(1) as usize).product())
                .unwrap_or(0)
        } else {
            0
        };

        total_params += params;
        if nd.is_differentiable {
            trainable_params += params;
        }

        let grad_col = if nd.is_differentiable { "yes" } else { "no" };

        let _ = writeln!(
            out,
            " {:<30} {:<20} {:<10} {}",
            truncate(&op_col, 29),
            truncate(&shape_col, 19),
            params,
            grad_col
        );
    }

    let _ = writeln!(out, "{sep}");
    let _ = writeln!(out, " Total parameters:     {total_params}");
    let _ = writeln!(out, " Trainable parameters: {trainable_params}");
    let _ = writeln!(
        out,
        " Non-trainable params: {}",
        total_params.saturating_sub(trainable_params)
    );
    let _ = writeln!(out, " Total nodes:          {}", ids.len());
    let _ = writeln!(out, "{sep}");

    out
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len.saturating_sub(1)])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameter / FLOPs counting
// ─────────────────────────────────────────────────────────────────────────────

/// Count the total number of trainable parameters in the graph.
///
/// A node is counted as a trainable parameter if it is a *variable* node
/// (created via `ctx.variable()`).  The count is the product of all known
/// shape dimensions; nodes with unknown shape contribute 0.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd as ag;
/// use ag::graph_viz::count_parameters;
///
/// ag::run(|ctx: &mut ag::Context<f64>| {
///     let x = ctx.placeholder("x", &[3]);
///     let y = x * 2.0;
///     let params = count_parameters(&y, ctx);
///     assert_eq!(params, 0); // no variable nodes
/// });
/// ```
pub fn count_parameters<'a, 'g, F: Float>(root: &Tensor<'g, F>, ctx: &'a Context<'g, F>) -> usize {
    let graph: &Graph<F> = std::ops::Deref::deref(ctx);
    let ids = collect_reachable(root.id(), graph);
    ids.iter()
        .map(|&id| {
            let nd = describe_node(id, graph);
            if nd.is_variable {
                nd.known_shape
                    .as_ref()
                    .map(|sh| sh.iter().map(|&d| d.max(1) as usize).product())
                    .unwrap_or(0)
            } else {
                0
            }
        })
        .sum()
}

/// Estimate the number of floating-point operations (FLOPs) for one forward
/// pass through the graph given an optional batch `input_shape`.
///
/// The estimate is *rough* — it uses op-name pattern matching to classify
/// operations and applies standard FLOPs formulas:
///
/// | Op category         | FLOPs formula                         |
/// |---------------------|---------------------------------------|
/// | Matmul / Linear     | 2 * M * N * K (from known shapes)     |
/// | Element-wise (±×÷)  | product of output shape                |
/// | Reduction           | product of input shape                 |
/// | Activation          | product of input shape                 |
/// | Other               | product of input shape (fallback)      |
///
/// # Arguments
/// * `root`        - The root tensor
/// * `ctx`         - The autograd context
/// * `input_shape` - Optional hint for the batch input shape (unused in this
///                   implementation; FLOPs are derived from node shapes)
///
/// # Example
///
/// ```rust
/// use scirs2_autograd as ag;
/// use ag::tensor_ops;
/// use ag::graph_viz::count_flops;
///
/// ag::run(|ctx: &mut ag::Context<f64>| {
///     let x = ctx.placeholder("x", &[4, 8]);
///     let y = x * 2.0;
///     let loss = tensor_ops::reduction::sum_all(y);
///     let flops = count_flops(&loss, ctx, None);
///     assert!(flops > 0);
/// });
/// ```
pub fn count_flops<'a, 'g, F: Float>(
    root: &Tensor<'g, F>,
    ctx: &'a Context<'g, F>,
    _input_shape: Option<&[usize]>,
) -> u64 {
    let graph: &Graph<F> = std::ops::Deref::deref(ctx);
    let ids = collect_reachable(root.id(), graph);
    let mut total: u64 = 0;

    for &id in &ids {
        let nd = describe_node(id, graph);

        // Element count of the output tensor (best effort)
        let elem_count: u64 = nd
            .known_shape
            .as_ref()
            .map(|sh| sh.iter().map(|&d| d.max(1) as u64).product())
            .unwrap_or(1);

        let op_lower = nd.op_name.to_lowercase();

        let flops = if op_lower.contains("matmul")
            || op_lower.contains("dot")
            || op_lower.contains("gemm")
        {
            // 2 * output_elements (rough: assumes M*K + K*N ~ 2*M*N*K / N)
            2 * elem_count
        } else if op_lower.contains("conv") {
            // Conv2d: 2 * C_in * k_h * k_w * C_out * H_out * W_out
            // Without precise kernel info we multiply by a heuristic factor
            9 * elem_count
        } else if op_lower.contains("sum")
            || op_lower.contains("mean")
            || op_lower.contains("max")
            || op_lower.contains("min")
            || op_lower.contains("reduce")
        {
            elem_count
        } else if op_lower.contains("relu")
            || op_lower.contains("sigmoid")
            || op_lower.contains("tanh")
            || op_lower.contains("gelu")
            || op_lower.contains("softmax")
        {
            // Activations: ~4–8 FLOPs per element; use 4 as conservative estimate
            4 * elem_count
        } else if op_lower.contains("norm") || op_lower.contains("batch") {
            // Normalization: mean + variance + scale + shift ≈ 8 ops/element
            8 * elem_count
        } else if op_lower.contains("add")
            || op_lower.contains("sub")
            || op_lower.contains("mul")
            || op_lower.contains("div")
            || op_lower.contains("pow")
        {
            elem_count
        } else if nd.is_placeholder || nd.is_variable {
            0
        } else {
            // Unknown — use element count as fallback
            elem_count
        };

        total = total.saturating_add(flops);
    }

    total
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-node statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Per-node record in a `GraphNodeTable`.
#[derive(Debug, Clone)]
pub struct NodeRecord {
    /// Graph-internal node ID
    pub id: TensorID,
    /// Operation name (short)
    pub op_name: String,
    /// Known output shape, if available
    pub shape: Option<Vec<isize>>,
    /// Whether this node has a gradient
    pub differentiable: bool,
    /// Whether this node is a trainable variable
    pub is_variable: bool,
    /// Whether this node is an input placeholder
    pub is_placeholder: bool,
    /// Topological rank
    pub topo_rank: usize,
    /// IDs of direct input nodes
    pub input_ids: Vec<TensorID>,
}

/// Tabular representation of all nodes in a computation graph.
#[derive(Debug, Clone)]
pub struct GraphNodeTable {
    pub records: Vec<NodeRecord>,
}

impl GraphNodeTable {
    /// Build a table from a root tensor.
    pub fn from_tensor<'a, 'g, F: Float>(root: &Tensor<'g, F>, ctx: &'a Context<'g, F>) -> Self {
        let graph: &Graph<F> = std::ops::Deref::deref(ctx);
        let ids = collect_reachable(root.id(), graph);
        let records = ids
            .iter()
            .map(|&id| {
                let nd = describe_node(id, graph);
                NodeRecord {
                    id: nd.id,
                    op_name: nd.op_name,
                    shape: nd.known_shape,
                    differentiable: nd.is_differentiable,
                    is_variable: nd.is_variable,
                    is_placeholder: nd.is_placeholder,
                    topo_rank: nd.topo_rank,
                    input_ids: nd.input_ids,
                }
            })
            .collect();
        Self { records }
    }

    /// Total nodes
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True if the table is empty
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Count variable nodes
    pub fn variable_count(&self) -> usize {
        self.records.iter().filter(|r| r.is_variable).count()
    }

    /// Count placeholder nodes
    pub fn placeholder_count(&self) -> usize {
        self.records.iter().filter(|r| r.is_placeholder).count()
    }

    /// Total parameter count from variable nodes with known shapes
    pub fn total_parameters(&self) -> usize {
        self.records
            .iter()
            .filter(|r| r.is_variable)
            .map(|r| {
                r.shape
                    .as_ref()
                    .map(|sh| sh.iter().map(|&d| d.max(1) as usize).product())
                    .unwrap_or(0)
            })
            .sum()
    }

    /// Operation-name frequency map
    pub fn op_frequencies(&self) -> HashMap<String, usize> {
        let mut map: HashMap<String, usize> = HashMap::new();
        for r in &self.records {
            *map.entry(r.op_name.clone()).or_insert(0) += 1;
        }
        map
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: DOT node color
// ─────────────────────────────────────────────────────────────────────────────

fn dot_color(nd: &NodeDesc) -> String {
    if nd.is_placeholder {
        "fillcolor=\"#d5f5d5\", color=\"#6aaf6a\"".to_string()
    } else if nd.is_variable {
        "fillcolor=\"#fff8d5\", color=\"#c8a830\"".to_string()
    } else if nd.is_differentiable {
        "fillcolor=\"#d5e8f5\", color=\"#4a8fc0\"".to_string()
    } else {
        "fillcolor=\"#e8e8e8\", color=\"#999999\"".to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_ops;

    // Helper used in tests that only need a dot/mermaid string without re-using ctx
    fn build_simple_graph_dot() -> String {
        let mut out = String::new();
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[3]);
            let y = x * 2.0 + 1.0;
            let loss = tensor_ops::reduction::sum_all(y);
            let viz = ComputationGraphViz::from_tensor(&loss, ctx);
            out = viz.to_dot();
        });
        out
    }

    #[test]
    fn test_to_dot_contains_digraph() {
        let dot = build_simple_graph_dot();
        assert!(dot.contains("digraph computation_graph"));
        assert!(dot.contains("->"));
        assert!(dot.contains('}'));
    }

    #[test]
    fn test_to_mermaid_format() {
        let mut mermaid = String::new();
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let y = x + x;
            let viz = ComputationGraphViz::from_tensor(&y, ctx);
            mermaid = viz.to_mermaid();
        });
        assert!(mermaid.contains("graph BT"));
        assert!(mermaid.contains("-->"));
    }

    #[test]
    fn test_print_graph_summary_contains_totals() {
        let mut summary = String::new();
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[3]);
            let y = x * 2.0 + 1.0;
            let loss = tensor_ops::reduction::sum_all(y);
            summary = print_graph_summary(&loss, ctx);
        });
        assert!(summary.contains("Total parameters"));
        assert!(summary.contains("Total nodes"));
        assert!(summary.contains("Trainable parameters"));
    }

    #[test]
    fn test_count_parameters_no_variables() {
        let mut params = 99usize;
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[3]);
            let y = x * 2.0;
            params = count_parameters(&y, ctx);
        });
        assert_eq!(params, 0);
    }

    #[test]
    fn test_count_flops_positive() {
        let mut flops = 0u64;
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[4, 8]);
            let y = x * 2.0;
            let loss = tensor_ops::reduction::sum_all(y);
            flops = count_flops(&loss, ctx, None);
        });
        assert!(flops > 0);
    }

    #[test]
    fn test_count_flops_placeholder_zero() {
        let mut flops = 999u64;
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[4]);
            flops = count_flops(&x, ctx, None);
        });
        assert_eq!(flops, 0);
    }

    #[test]
    fn test_graph_node_table_basics() {
        let mut node_count = 0usize;
        let mut placeholder_count = 0usize;
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[3]);
            let y = x * 2.0 + 1.0;
            let loss = tensor_ops::reduction::sum_all(y);
            let table = GraphNodeTable::from_tensor(&loss, ctx);
            node_count = table.len();
            placeholder_count = table.placeholder_count();
        });
        assert!(node_count > 0);
        assert_eq!(placeholder_count, 1);
    }

    #[test]
    fn test_export_dot_writes_file() {
        let path = std::env::temp_dir().join("test_export.dot");
        let path_str = path.to_string_lossy().to_string();
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let y = x * 3.0;
            export_dot(&y, ctx, &path_str).expect("export_dot failed");
        });
        let content = std::fs::read_to_string(&path).expect("read failed");
        assert!(content.contains("digraph"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_mermaid_writes_file() {
        let path = std::env::temp_dir().join("test_export_mermaid.md");
        let path_str = path.to_string_lossy().to_string();
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[2]);
            let y = x * 3.0;
            export_mermaid(&y, ctx, &path_str).expect("export_mermaid failed");
        });
        let content = std::fs::read_to_string(&path).expect("read failed");
        assert!(content.contains("mermaid"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_node_to_dot_placeholder() {
        let mut label = String::new();
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[3]);
            let viz = ComputationGraphViz::from_tensor(&x, ctx);
            // The first (and only) node is the placeholder
            let nd = &viz.nodes()[0];
            label = viz.node_to_dot(nd);
        });
        assert!(label.contains("x"));
    }

    #[test]
    fn test_op_frequencies() {
        let mut has_ops = false;
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = ctx.placeholder("x", &[3]);
            let y = x * 2.0 + 1.0;
            let loss = tensor_ops::reduction::sum_all(y);
            let table = GraphNodeTable::from_tensor(&loss, ctx);
            let freqs = table.op_frequencies();
            has_ops = !freqs.is_empty();
        });
        assert!(has_ops);
    }
}
