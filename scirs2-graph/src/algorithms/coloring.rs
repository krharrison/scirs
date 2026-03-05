//! Graph coloring algorithms
//!
//! This module contains algorithms for graph coloring problems, including
//! vertex coloring, edge coloring, and chromatic number estimation.
//!
//! # Algorithms
//! - **Greedy coloring**: With ordering strategies (natural, largest-first, smallest-last, DSatur)
//! - **Welsh-Powell**: Degree-ordered greedy coloring
//! - **Chromatic number bounds**: Lower via clique, upper via greedy
//! - **Edge coloring**: With Vizing's theorem bound
//! - **List coloring**: Each vertex has a list of allowed colors

use crate::base::{EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};

/// Result of graph coloring
#[derive(Debug, Clone)]
pub struct GraphColoring<N: Node> {
    /// The coloring as a map from node to color (0-based)
    pub coloring: HashMap<N, usize>,
    /// The number of colors used
    pub num_colors: usize,
}

/// Ordering strategy for greedy coloring
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColoringOrder {
    /// Natural order (as nodes appear in the graph)
    Natural,
    /// Largest-first: order by degree descending
    LargestFirst,
    /// Smallest-last: iteratively remove smallest-degree vertex
    SmallestLast,
    /// DSatur: choose vertex with largest saturation degree
    DSatur,
    /// Random order (shuffled)
    Random,
}

/// Result of edge coloring
#[derive(Debug, Clone)]
pub struct EdgeColoring<N: Node> {
    /// The coloring: (source, target) -> color
    pub coloring: HashMap<(N, N), usize>,
    /// Number of colors used
    pub num_colors: usize,
    /// Maximum degree (Vizing lower bound)
    pub max_degree: usize,
    /// Whether the graph is class 1 (chi' = Delta) or class 2 (chi' = Delta + 1)
    pub is_class_one: bool,
}

/// Result of chromatic number bounds
#[derive(Debug, Clone)]
pub struct ChromaticBounds {
    /// Lower bound (clique number or other)
    pub lower: usize,
    /// Upper bound (greedy coloring)
    pub upper: usize,
    /// The best known coloring achieving the upper bound
    pub best_num_colors: usize,
}

/// Result of list coloring
#[derive(Debug, Clone)]
pub struct ListColoring<N: Node> {
    /// Whether a valid list coloring was found
    pub feasible: bool,
    /// The coloring (if feasible)
    pub coloring: HashMap<N, usize>,
    /// Number of colors used
    pub num_colors: usize,
}

// ============================================================================
// Greedy coloring with ordering strategies
// ============================================================================

/// Greedy graph coloring with configurable vertex ordering.
///
/// Colors vertices one at a time, always choosing the smallest available color.
/// The vertex ordering dramatically affects the number of colors used.
///
/// # Arguments
/// * `graph` - The graph to color
///
/// # Returns
/// * A graph coloring using natural vertex ordering
pub fn greedy_coloring<N, E, Ix>(graph: &Graph<N, E, Ix>) -> GraphColoring<N>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    greedy_coloring_with_order(graph, ColoringOrder::Natural)
}

/// Greedy coloring with a specified vertex ordering strategy.
pub fn greedy_coloring_with_order<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    order: ColoringOrder,
) -> GraphColoring<N>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let node_count = graph.inner().node_count();
    if node_count == 0 {
        return GraphColoring {
            coloring: HashMap::new(),
            num_colors: 0,
        };
    }

    let ordered_nodes = compute_ordering(graph, order);
    color_in_order(graph, &ordered_nodes)
}

/// Compute vertex ordering based on the strategy.
fn compute_ordering<N, E, Ix>(graph: &Graph<N, E, Ix>, order: ColoringOrder) -> Vec<N>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph
        .inner()
        .node_indices()
        .map(|idx| graph.inner()[idx].clone())
        .collect();

    match order {
        ColoringOrder::Natural => nodes,
        ColoringOrder::LargestFirst => largest_first_order(graph, &nodes),
        ColoringOrder::SmallestLast => smallest_last_order(graph, &nodes),
        ColoringOrder::DSatur => {
            // DSatur is handled differently - return natural order
            // and use dsatur_coloring directly
            nodes
        }
        ColoringOrder::Random => {
            let mut shuffled = nodes;
            // Simple deterministic shuffle using indices
            let n = shuffled.len();
            for i in 0..n {
                let j = (i * 7 + 3) % n;
                shuffled.swap(i, j);
            }
            shuffled
        }
    }
}

/// Largest-first ordering: sort by degree descending.
fn largest_first_order<N, E, Ix>(graph: &Graph<N, E, Ix>, nodes: &[N]) -> Vec<N>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let mut node_degrees: Vec<(N, usize)> =
        nodes.iter().map(|n| (n.clone(), graph.degree(n))).collect();
    node_degrees.sort_by(|a, b| b.1.cmp(&a.1));
    node_degrees.into_iter().map(|(n, _)| n).collect()
}

/// Smallest-last ordering: iteratively remove smallest-degree vertex.
///
/// This produces orderings that tend to use fewer colors, especially
/// for sparse graphs.
fn smallest_last_order<N, E, Ix>(graph: &Graph<N, E, Ix>, nodes: &[N]) -> Vec<N>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let n = nodes.len();
    let mut remaining: HashSet<N> = nodes.iter().cloned().collect();
    let mut order = Vec::with_capacity(n);

    // Build adjacency map
    let mut adj: HashMap<N, HashSet<N>> = HashMap::new();
    for node in nodes {
        if let Ok(neighbors) = graph.neighbors(node) {
            let neighbor_set: HashSet<N> = neighbors.into_iter().collect();
            adj.insert(node.clone(), neighbor_set);
        } else {
            adj.insert(node.clone(), HashSet::new());
        }
    }

    for _ in 0..n {
        // Find node with smallest degree among remaining nodes
        let mut min_deg = usize::MAX;
        let mut min_node = None;

        for node in &remaining {
            let deg = adj
                .get(node)
                .map(|neighbors| neighbors.iter().filter(|n| remaining.contains(n)).count())
                .unwrap_or(0);
            if deg < min_deg {
                min_deg = deg;
                min_node = Some(node.clone());
            }
        }

        if let Some(node) = min_node {
            order.push(node.clone());
            remaining.remove(&node);
        }
    }

    // Reverse: smallest-last means we color in reverse removal order
    order.reverse();
    order
}

/// Color nodes in the given order using greedy assignment.
fn color_in_order<N, E, Ix>(graph: &Graph<N, E, Ix>, ordered_nodes: &[N]) -> GraphColoring<N>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let mut coloring: HashMap<N, usize> = HashMap::new();
    let mut max_color = 0;

    for node in ordered_nodes {
        // Find colors used by already-colored neighbors
        let mut used_colors = HashSet::new();
        if let Ok(neighbors) = graph.neighbors(node) {
            for neighbor in &neighbors {
                if let Some(&color) = coloring.get(neighbor) {
                    used_colors.insert(color);
                }
            }
        }

        // Find smallest available color
        let mut color = 0;
        while used_colors.contains(&color) {
            color += 1;
        }

        coloring.insert(node.clone(), color);
        if color >= max_color {
            max_color = color + 1;
        }
    }

    GraphColoring {
        coloring,
        num_colors: max_color,
    }
}

// ============================================================================
// DSatur coloring
// ============================================================================

/// DSatur (Degree of Saturation) coloring algorithm.
///
/// At each step, colors the vertex with the highest saturation degree
/// (number of distinct colors among its colored neighbors). Ties are
/// broken by choosing the vertex with highest uncolored degree.
///
/// This generally produces better colorings than basic greedy.
pub fn dsatur_coloring<N, E, Ix>(graph: &Graph<N, E, Ix>) -> GraphColoring<N>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph
        .inner()
        .node_indices()
        .map(|idx| graph.inner()[idx].clone())
        .collect();

    let n = nodes.len();
    if n == 0 {
        return GraphColoring {
            coloring: HashMap::new(),
            num_colors: 0,
        };
    }

    let mut coloring: HashMap<N, usize> = HashMap::new();
    let mut saturation: HashMap<N, HashSet<usize>> = HashMap::new();
    let mut colored = HashSet::new();
    let mut max_color = 0;

    for node in &nodes {
        saturation.insert(node.clone(), HashSet::new());
    }

    for _ in 0..n {
        // Find uncolored vertex with highest saturation degree
        let mut best_node = None;
        let mut best_sat = 0;
        let mut best_deg = 0;

        for node in &nodes {
            if colored.contains(node) {
                continue;
            }

            let sat = saturation.get(node).map(|s| s.len()).unwrap_or(0);
            let deg = graph.degree(node);

            if best_node.is_none() || sat > best_sat || (sat == best_sat && deg > best_deg) {
                best_node = Some(node.clone());
                best_sat = sat;
                best_deg = deg;
            }
        }

        if let Some(node) = best_node {
            // Find used colors among neighbors
            let mut used_colors = HashSet::new();
            if let Ok(neighbors) = graph.neighbors(&node) {
                for neighbor in &neighbors {
                    if let Some(&color) = coloring.get(neighbor) {
                        used_colors.insert(color);
                    }
                }
            }

            // Pick smallest available color
            let mut color = 0;
            while used_colors.contains(&color) {
                color += 1;
            }

            coloring.insert(node.clone(), color);
            colored.insert(node.clone());
            if color + 1 > max_color {
                max_color = color + 1;
            }

            // Update saturation of uncolored neighbors
            if let Ok(neighbors) = graph.neighbors(&node) {
                for neighbor in &neighbors {
                    if !colored.contains(neighbor) {
                        if let Some(sat) = saturation.get_mut(neighbor) {
                            sat.insert(color);
                        }
                    }
                }
            }
        }
    }

    GraphColoring {
        coloring,
        num_colors: max_color,
    }
}

// ============================================================================
// Welsh-Powell Algorithm
// ============================================================================

/// Welsh-Powell coloring algorithm.
///
/// Orders vertices by decreasing degree, then applies greedy coloring.
/// This is equivalent to `greedy_coloring_with_order(graph, ColoringOrder::LargestFirst)`.
pub fn welsh_powell<N, E, Ix>(graph: &Graph<N, E, Ix>) -> GraphColoring<N>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    greedy_coloring_with_order(graph, ColoringOrder::LargestFirst)
}

// ============================================================================
// Chromatic Number Bounds
// ============================================================================

/// Finds the chromatic number of a graph (minimum number of colors needed).
///
/// Uses an exhaustive search up to `max_colors` to find the minimum coloring.
/// This is an NP-complete problem, so it may be slow for large graphs.
pub fn chromatic_number<N, E, Ix>(graph: &Graph<N, E, Ix>, max_colors: usize) -> Option<usize>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    if graph.inner().node_count() == 0 {
        return Some(0);
    }

    (1..=max_colors).find(|&num_colors| can_color_with_k_colors(graph, num_colors))
}

/// Check if a graph can be colored with k colors (backtracking).
fn can_color_with_k_colors<N, E, Ix>(graph: &Graph<N, E, Ix>, k: usize) -> bool
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<_> = graph.inner().node_indices().collect();
    let mut coloring = vec![0usize; nodes.len()];

    fn backtrack<N, E, Ix>(
        graph: &Graph<N, E, Ix>,
        nodes: &[petgraph::graph::NodeIndex<Ix>],
        coloring: &mut [usize],
        node_idx: usize,
        k: usize,
    ) -> bool
    where
        N: Node + std::fmt::Debug,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        if node_idx == nodes.len() {
            return true;
        }

        let node = nodes[node_idx];

        for color in 0..k {
            let mut valid = true;
            for (i, &other_node) in nodes.iter().enumerate().take(node_idx) {
                if (graph.inner().contains_edge(node, other_node)
                    || graph.inner().contains_edge(other_node, node))
                    && coloring[i] == color
                {
                    valid = false;
                    break;
                }
            }

            if valid {
                coloring[node_idx] = color;
                if backtrack(graph, nodes, coloring, node_idx + 1, k) {
                    return true;
                }
            }
        }

        false
    }

    backtrack(graph, &nodes, &mut coloring, 0, k)
}

/// Compute chromatic number bounds using clique (lower) and greedy (upper).
///
/// The lower bound is the clique number (found via greedy clique search).
/// The upper bound is from the best greedy coloring (DSatur).
pub fn chromatic_bounds<N, E, Ix>(graph: &Graph<N, E, Ix>) -> ChromaticBounds
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.inner().node_count();
    if n == 0 {
        return ChromaticBounds {
            lower: 0,
            upper: 0,
            best_num_colors: 0,
        };
    }

    // Lower bound: greedy clique finding
    let lower = greedy_clique_number(graph);

    // Upper bound: best of multiple orderings
    let dsatur_result = dsatur_coloring(graph);
    let welsh_result = welsh_powell(graph);
    let sl_result = greedy_coloring_with_order(graph, ColoringOrder::SmallestLast);

    let upper = dsatur_result
        .num_colors
        .min(welsh_result.num_colors)
        .min(sl_result.num_colors);

    // Also: upper <= max_degree + 1 (Brook's theorem, except complete graphs and odd cycles)
    let max_degree = graph
        .inner()
        .node_indices()
        .map(|idx| graph.inner().neighbors(idx).count())
        .max()
        .unwrap_or(0);
    let brooks_upper = max_degree + 1;

    let final_upper = upper.min(brooks_upper);

    ChromaticBounds {
        lower,
        upper: final_upper,
        best_num_colors: final_upper,
    }
}

/// Find a large clique greedily to bound chromatic number from below.
fn greedy_clique_number<N, E, Ix>(graph: &Graph<N, E, Ix>) -> usize
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph
        .inner()
        .node_indices()
        .map(|idx| graph.inner()[idx].clone())
        .collect();

    if nodes.is_empty() {
        return 0;
    }

    let mut max_clique_size = 1;

    // Try starting from each node (greedy clique extension)
    for start_node in &nodes {
        let mut clique = vec![start_node.clone()];

        // Try to extend the clique
        // Sort candidates by degree descending for better cliques
        let mut candidates: Vec<N> = Vec::new();
        if let Ok(neighbors) = graph.neighbors(start_node) {
            candidates = neighbors;
        }
        candidates.sort_by_key(|b| std::cmp::Reverse(graph.degree(b)));

        for candidate in &candidates {
            // Check if candidate is adjacent to all nodes in the clique
            let all_adjacent = clique.iter().all(|c| graph.has_edge(candidate, c));
            if all_adjacent {
                clique.push(candidate.clone());
            }
        }

        max_clique_size = max_clique_size.max(clique.len());
    }

    max_clique_size
}

// ============================================================================
// Edge Coloring
// ============================================================================

/// Edge coloring using greedy approach with Vizing's theorem bound.
///
/// Vizing's theorem states that every simple graph can be edge-colored
/// with at most Delta + 1 colors, where Delta is the maximum degree.
/// A graph is class 1 if Delta colors suffice, class 2 if Delta + 1 are needed.
pub fn edge_coloring<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<EdgeColoring<N>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Clone,
    Ix: petgraph::graph::IndexType,
{
    let edges = graph.edges();
    let n = graph.inner().node_count();

    if n == 0 || edges.is_empty() {
        return Ok(EdgeColoring {
            coloring: HashMap::new(),
            num_colors: 0,
            max_degree: 0,
            is_class_one: true,
        });
    }

    let max_degree = graph
        .inner()
        .node_indices()
        .map(|idx| graph.inner().neighbors(idx).count())
        .max()
        .unwrap_or(0);

    // Greedy edge coloring: color each edge with the smallest color
    // not used by any adjacent edge (edge sharing a vertex)
    let mut edge_colors: HashMap<(N, N), usize> = HashMap::new();
    let mut node_used_colors: HashMap<N, HashSet<usize>> = HashMap::new();
    let mut num_colors = 0;

    for edge in &edges {
        let src = edge.source.clone();
        let tgt = edge.target.clone();

        // Colors used at source
        let src_colors = node_used_colors.get(&src).cloned().unwrap_or_default();
        // Colors used at target
        let tgt_colors = node_used_colors.get(&tgt).cloned().unwrap_or_default();

        // Find smallest unused color at both endpoints
        let mut color = 0;
        while src_colors.contains(&color) || tgt_colors.contains(&color) {
            color += 1;
        }

        edge_colors.insert((src.clone(), tgt.clone()), color);

        node_used_colors.entry(src).or_default().insert(color);
        node_used_colors.entry(tgt).or_default().insert(color);

        if color + 1 > num_colors {
            num_colors = color + 1;
        }
    }

    let is_class_one = num_colors <= max_degree;

    Ok(EdgeColoring {
        coloring: edge_colors,
        num_colors,
        max_degree,
        is_class_one,
    })
}

// ============================================================================
// List Coloring
// ============================================================================

/// List coloring: each vertex has a set of allowed colors.
///
/// Attempts to find a proper coloring where each vertex is assigned
/// a color from its allowed list. Uses backtracking.
///
/// # Arguments
/// * `graph` - The graph to color
/// * `color_lists` - Map from node to set of allowed colors
pub fn list_coloring<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    color_lists: &HashMap<N, Vec<usize>>,
) -> Result<ListColoring<N>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph
        .inner()
        .node_indices()
        .map(|idx| graph.inner()[idx].clone())
        .collect();

    if nodes.is_empty() {
        return Ok(ListColoring {
            feasible: true,
            coloring: HashMap::new(),
            num_colors: 0,
        });
    }

    // Verify all nodes have color lists
    for node in &nodes {
        if !color_lists.contains_key(node) {
            return Err(GraphError::InvalidGraph(format!(
                "Node {node:?} has no color list"
            )));
        }
    }

    let mut coloring: HashMap<N, usize> = HashMap::new();
    let feasible = list_coloring_backtrack(graph, &nodes, color_lists, &mut coloring, 0);

    if feasible {
        let num_colors = coloring.values().copied().collect::<HashSet<_>>().len();
        Ok(ListColoring {
            feasible: true,
            coloring,
            num_colors,
        })
    } else {
        Ok(ListColoring {
            feasible: false,
            coloring: HashMap::new(),
            num_colors: 0,
        })
    }
}

/// Backtracking for list coloring.
fn list_coloring_backtrack<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    nodes: &[N],
    color_lists: &HashMap<N, Vec<usize>>,
    coloring: &mut HashMap<N, usize>,
    idx: usize,
) -> bool
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    if idx == nodes.len() {
        return true;
    }

    let node = &nodes[idx];
    let allowed = color_lists.get(node).cloned().unwrap_or_default();

    // Get colors used by already-colored neighbors
    let mut forbidden = HashSet::new();
    if let Ok(neighbors) = graph.neighbors(node) {
        for neighbor in &neighbors {
            if let Some(&c) = coloring.get(neighbor) {
                forbidden.insert(c);
            }
        }
    }

    for &color in &allowed {
        if !forbidden.contains(&color) {
            coloring.insert(node.clone(), color);
            if list_coloring_backtrack(graph, nodes, color_lists, coloring, idx + 1) {
                return true;
            }
            coloring.remove(node);
        }
    }

    false
}

// ============================================================================
// Utility: verify a coloring is valid
// ============================================================================

/// Verify that a graph coloring is valid (no two adjacent nodes share a color).
pub fn verify_coloring<N, E, Ix>(graph: &Graph<N, E, Ix>, coloring: &HashMap<N, usize>) -> bool
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    for edge in graph.inner().edge_references() {
        let src = &graph.inner()[edge.source()];
        let tgt = &graph.inner()[edge.target()];

        if let (Some(&c1), Some(&c2)) = (coloring.get(src), coloring.get(tgt)) {
            if c1 == c2 {
                return false;
            }
        }
    }
    true
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::create_graph;
    use petgraph::visit::EdgeRef;

    #[test]
    fn test_greedy_coloring() {
        let mut graph = create_graph::<char, ()>();
        graph.add_edge('A', 'B', ()).expect("add edge");
        graph.add_edge('B', 'C', ()).expect("add edge");
        graph.add_edge('C', 'A', ()).expect("add edge");

        let coloring = greedy_coloring(&graph);
        assert_eq!(coloring.num_colors, 3);
        assert_ne!(coloring.coloring[&'A'], coloring.coloring[&'B']);
        assert_ne!(coloring.coloring[&'B'], coloring.coloring[&'C']);
        assert_ne!(coloring.coloring[&'C'], coloring.coloring[&'A']);
    }

    #[test]
    fn test_bipartite_graph_coloring() {
        let mut graph = create_graph::<i32, ()>();
        graph.add_edge(0, 1, ()).expect("add edge");
        graph.add_edge(0, 3, ()).expect("add edge");
        graph.add_edge(2, 1, ()).expect("add edge");
        graph.add_edge(2, 3, ()).expect("add edge");

        let coloring = greedy_coloring(&graph);
        assert!(coloring.num_colors <= 2);
    }

    #[test]
    fn test_chromatic_number() {
        let mut triangle = create_graph::<i32, ()>();
        triangle.add_edge(0, 1, ()).expect("add edge");
        triangle.add_edge(1, 2, ()).expect("add edge");
        triangle.add_edge(2, 0, ()).expect("add edge");

        assert_eq!(chromatic_number(&triangle, 5), Some(3));

        let mut bipartite = create_graph::<i32, ()>();
        bipartite.add_edge(0, 1, ()).expect("add edge");
        bipartite.add_edge(1, 2, ()).expect("add edge");
        bipartite.add_edge(2, 3, ()).expect("add edge");
        bipartite.add_edge(3, 0, ()).expect("add edge");

        assert_eq!(chromatic_number(&bipartite, 5), Some(2));

        let empty = create_graph::<i32, ()>();
        assert_eq!(chromatic_number(&empty, 5), Some(0));
    }

    #[test]
    fn test_largest_first_coloring() {
        let mut graph = create_graph::<i32, ()>();
        graph.add_edge(0, 1, ()).expect("add edge");
        graph.add_edge(0, 2, ()).expect("add edge");
        graph.add_edge(0, 3, ()).expect("add edge");
        graph.add_edge(1, 2, ()).expect("add edge");

        let coloring = greedy_coloring_with_order(&graph, ColoringOrder::LargestFirst);
        assert!(verify_coloring(&graph, &coloring.coloring));
    }

    #[test]
    fn test_smallest_last_coloring() {
        let mut graph = create_graph::<i32, ()>();
        for i in 0..5 {
            for j in (i + 1)..5 {
                graph.add_edge(i, j, ()).expect("add edge");
            }
        }

        let coloring = greedy_coloring_with_order(&graph, ColoringOrder::SmallestLast);
        assert!(verify_coloring(&graph, &coloring.coloring));
        assert_eq!(coloring.num_colors, 5); // Complete graph K5 needs 5 colors
    }

    #[test]
    fn test_dsatur_coloring() {
        let mut graph = create_graph::<i32, ()>();
        graph.add_edge(0, 1, ()).expect("add edge");
        graph.add_edge(1, 2, ()).expect("add edge");
        graph.add_edge(2, 0, ()).expect("add edge");
        graph.add_edge(2, 3, ()).expect("add edge");
        graph.add_edge(3, 4, ()).expect("add edge");

        let coloring = dsatur_coloring(&graph);
        assert!(verify_coloring(&graph, &coloring.coloring));
        // Triangle needs 3, rest might need fewer
        assert!(coloring.num_colors >= 3);
    }

    #[test]
    fn test_welsh_powell() {
        let mut graph = create_graph::<i32, ()>();
        // Crown graph (bipartite)
        graph.add_edge(0, 3, ()).expect("add edge");
        graph.add_edge(0, 4, ()).expect("add edge");
        graph.add_edge(1, 3, ()).expect("add edge");
        graph.add_edge(1, 5, ()).expect("add edge");
        graph.add_edge(2, 4, ()).expect("add edge");
        graph.add_edge(2, 5, ()).expect("add edge");

        let coloring = welsh_powell(&graph);
        assert!(verify_coloring(&graph, &coloring.coloring));
        assert!(coloring.num_colors <= 2);
    }

    #[test]
    fn test_chromatic_bounds() {
        let mut graph = create_graph::<i32, ()>();
        // Petersen graph is 3-chromatic
        // But let's use a simpler example: triangle
        graph.add_edge(0, 1, ()).expect("add edge");
        graph.add_edge(1, 2, ()).expect("add edge");
        graph.add_edge(2, 0, ()).expect("add edge");

        let bounds = chromatic_bounds(&graph);
        assert!(bounds.lower >= 3); // Triangle is a 3-clique
        assert!(bounds.upper >= 3);
        assert!(bounds.lower <= bounds.upper);
    }

    #[test]
    fn test_edge_coloring() -> Result<()> {
        let mut graph = create_graph::<i32, ()>();
        graph.add_edge(0, 1, ())?;
        graph.add_edge(1, 2, ())?;
        graph.add_edge(2, 0, ())?;

        let result = edge_coloring(&graph)?;
        assert_eq!(result.max_degree, 2);
        // Triangle needs 3 edge colors (class 2)
        assert!(result.num_colors <= result.max_degree + 1);

        // Verify edge coloring: no two edges sharing a vertex have same color
        for edge1 in graph.inner().edge_references() {
            for edge2 in graph.inner().edge_references() {
                if edge1.id() == edge2.id() {
                    continue;
                }
                if edge1.source() == edge2.source()
                    || edge1.source() == edge2.target()
                    || edge1.target() == edge2.source()
                    || edge1.target() == edge2.target()
                {
                    let s1 = graph.inner()[edge1.source()].clone();
                    let t1 = graph.inner()[edge1.target()].clone();
                    let s2 = graph.inner()[edge2.source()].clone();
                    let t2 = graph.inner()[edge2.target()].clone();

                    let c1 = result
                        .coloring
                        .get(&(s1.clone(), t1.clone()))
                        .or_else(|| result.coloring.get(&(t1, s1)));
                    let c2 = result
                        .coloring
                        .get(&(s2.clone(), t2.clone()))
                        .or_else(|| result.coloring.get(&(t2, s2)));

                    if let (Some(c1), Some(c2)) = (c1, c2) {
                        assert_ne!(c1, c2, "Adjacent edges should have different colors");
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_edge_coloring_bipartite() -> Result<()> {
        // Bipartite graph: class 1 (edge chromatic number = max degree)
        let mut graph = create_graph::<i32, ()>();
        graph.add_edge(0, 2, ())?;
        graph.add_edge(0, 3, ())?;
        graph.add_edge(1, 2, ())?;
        graph.add_edge(1, 3, ())?;

        let result = edge_coloring(&graph)?;
        assert_eq!(result.max_degree, 2);
        assert!(result.num_colors <= 3); // At most Delta + 1
        Ok(())
    }

    #[test]
    fn test_list_coloring_feasible() -> Result<()> {
        let mut graph = create_graph::<i32, ()>();
        graph.add_edge(0, 1, ())?;
        graph.add_edge(1, 2, ())?;

        let mut lists = HashMap::new();
        lists.insert(0, vec![0, 1]);
        lists.insert(1, vec![0, 1]);
        lists.insert(2, vec![0, 1]);

        let result = list_coloring(&graph, &lists)?;
        assert!(result.feasible);
        assert!(result.num_colors <= 2);
        Ok(())
    }

    #[test]
    fn test_list_coloring_infeasible() -> Result<()> {
        // Triangle with too-restrictive lists
        let mut graph = create_graph::<i32, ()>();
        graph.add_edge(0, 1, ())?;
        graph.add_edge(1, 2, ())?;
        graph.add_edge(2, 0, ())?;

        let mut lists = HashMap::new();
        lists.insert(0, vec![0, 1]);
        lists.insert(1, vec![0, 1]);
        lists.insert(2, vec![0, 1]); // Triangle can't be 2-colored

        let result = list_coloring(&graph, &lists)?;
        assert!(!result.feasible);
        Ok(())
    }

    #[test]
    fn test_list_coloring_with_enough_colors() -> Result<()> {
        let mut graph = create_graph::<i32, ()>();
        graph.add_edge(0, 1, ())?;
        graph.add_edge(1, 2, ())?;
        graph.add_edge(2, 0, ())?;

        let mut lists = HashMap::new();
        lists.insert(0, vec![0, 1, 2]);
        lists.insert(1, vec![0, 1, 2]);
        lists.insert(2, vec![0, 1, 2]);

        let result = list_coloring(&graph, &lists)?;
        assert!(result.feasible);
        assert_eq!(result.num_colors, 3);
        Ok(())
    }

    #[test]
    fn test_verify_coloring_valid() {
        let mut graph = create_graph::<i32, ()>();
        graph.add_edge(0, 1, ()).expect("add edge");
        graph.add_edge(1, 2, ()).expect("add edge");

        let mut coloring = HashMap::new();
        coloring.insert(0, 0);
        coloring.insert(1, 1);
        coloring.insert(2, 0);

        assert!(verify_coloring(&graph, &coloring));
    }

    #[test]
    fn test_verify_coloring_invalid() {
        let mut graph = create_graph::<i32, ()>();
        graph.add_edge(0, 1, ()).expect("add edge");

        let mut coloring = HashMap::new();
        coloring.insert(0, 0);
        coloring.insert(1, 0); // Same color as neighbor

        assert!(!verify_coloring(&graph, &coloring));
    }

    #[test]
    fn test_empty_graph_coloring() {
        let graph = create_graph::<i32, ()>();
        let coloring = greedy_coloring(&graph);
        assert_eq!(coloring.num_colors, 0);
        assert!(coloring.coloring.is_empty());
    }

    #[test]
    fn test_single_node() {
        let mut graph = create_graph::<i32, ()>();
        let _ = graph.add_node(42);
        let coloring = greedy_coloring(&graph);
        assert_eq!(coloring.num_colors, 1);
        assert_eq!(coloring.coloring[&42], 0);
    }
}
