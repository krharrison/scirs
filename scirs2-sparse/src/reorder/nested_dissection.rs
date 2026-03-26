//! Nested Dissection ordering for fill-reducing reordering
//!
//! Nested dissection is a divide-and-conquer algorithm that recursively
//! bisects a graph using vertex separators. Separator nodes are ordered
//! last, and the algorithm recurses on the two subgraphs. This produces
//! orderings with near-optimal fill-in for many classes of sparse matrices,
//! particularly those arising from 2D and 3D finite element meshes.
//!
//! # Algorithm
//!
//! 1. Find a vertex separator S that splits graph G into subgraphs A and B.
//! 2. Recursively order A and B.
//! 3. Order separator S last.
//!
//! When the subgraph is small enough, switch to AMD for the base case.
//!
//! # Multi-level variant
//!
//! The multi-level variant coarsens the graph by heavy-edge matching,
//! partitions the coarsened graph, then uncoarsens and refines.
//!
//! # References
//!
//! - A. George, "Nested Dissection of a Regular Finite Element Mesh",
//!   SIAM J. Numer. Anal., 10(2), 1973.
//! - G. Karypis, V. Kumar, "A Fast and High Quality Multilevel Scheme for
//!   Partitioning Irregular Graphs", SIAM J. Sci. Comput., 20(1), 1998.

use std::collections::VecDeque;

use super::adjacency::AdjacencyGraph;
use super::amd;
use crate::error::SparseResult;

/// Configuration for nested dissection.
#[derive(Debug, Clone)]
pub struct NestedDissectionConfig {
    /// Minimum subgraph size before switching to AMD.
    /// Default: 64.
    pub min_subgraph_size: usize,
    /// Whether to use multi-level coarsening for separator finding.
    /// Default: true.
    pub multi_level: bool,
    /// Maximum number of coarsening levels in multi-level mode.
    /// Default: 10.
    pub max_coarsening_levels: usize,
    /// Number of Kernighan-Lin refinement passes per level.
    /// Default: 5.
    pub kl_passes: usize,
}

impl Default for NestedDissectionConfig {
    fn default() -> Self {
        Self {
            min_subgraph_size: 64,
            multi_level: true,
            max_coarsening_levels: 10,
            kl_passes: 5,
        }
    }
}

/// Result of nested dissection ordering.
#[derive(Debug, Clone)]
pub struct NestedDissectionResult {
    /// Permutation vector: `perm[new_index] = old_index`.
    pub perm: Vec<usize>,
    /// Inverse permutation.
    pub inv_perm: Vec<usize>,
    /// Number of separator nodes at the top level.
    pub top_separator_size: usize,
}

/// Compute a nested dissection ordering for a symmetric adjacency graph.
///
/// Uses default configuration (min subgraph size 64, multi-level enabled).
pub fn nested_dissection(graph: &AdjacencyGraph) -> SparseResult<Vec<usize>> {
    let config = NestedDissectionConfig::default();
    nested_dissection_with_config(graph, &config)
}

/// Compute a nested dissection ordering with custom configuration.
pub fn nested_dissection_with_config(
    graph: &AdjacencyGraph,
    config: &NestedDissectionConfig,
) -> SparseResult<Vec<usize>> {
    let result = nested_dissection_full(graph, config)?;
    Ok(result.perm)
}

/// Compute a nested dissection ordering with full result.
pub fn nested_dissection_full(
    graph: &AdjacencyGraph,
    config: &NestedDissectionConfig,
) -> SparseResult<NestedDissectionResult> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(NestedDissectionResult {
            perm: Vec::new(),
            inv_perm: Vec::new(),
            top_separator_size: 0,
        });
    }

    let nodes: Vec<usize> = (0..n).collect();
    let mut perm = Vec::with_capacity(n);
    let mut top_sep_size = 0;

    nd_recurse(graph, &nodes, config, &mut perm, 0, &mut top_sep_size)?;

    // Build inverse permutation
    let mut inv_perm = vec![0usize; n];
    for (new_i, &old_i) in perm.iter().enumerate() {
        inv_perm[old_i] = new_i;
    }

    Ok(NestedDissectionResult {
        perm,
        inv_perm,
        top_separator_size: top_sep_size,
    })
}

/// Recursive nested dissection on a subset of nodes.
fn nd_recurse(
    graph: &AdjacencyGraph,
    nodes: &[usize],
    config: &NestedDissectionConfig,
    perm: &mut Vec<usize>,
    depth: usize,
    top_sep_size: &mut usize,
) -> SparseResult<()> {
    let n = nodes.len();

    if n == 0 {
        return Ok(());
    }

    // Base case: switch to AMD for small subgraphs
    if n <= config.min_subgraph_size {
        let (sub_graph, mapping) = graph.subgraph(nodes);
        let amd_result = amd::amd(&sub_graph)?;
        for &p in &amd_result.perm {
            perm.push(mapping[p]);
        }
        return Ok(());
    }

    // Find a vertex separator
    let (part_a, separator, part_b) = if config.multi_level {
        find_separator_multilevel(graph, nodes, config)
    } else {
        find_separator_bfs(graph, nodes)
    };

    // Recurse on part A
    nd_recurse(graph, &part_a, config, perm, depth + 1, top_sep_size)?;

    // Recurse on part B
    nd_recurse(graph, &part_b, config, perm, depth + 1, top_sep_size)?;

    // Order separator last
    for &s in &separator {
        perm.push(s);
    }

    if depth == 0 {
        *top_sep_size = separator.len();
    }

    Ok(())
}

/// Find a vertex separator using BFS-based level-set approach.
///
/// 1. BFS from a peripheral node to get level sets.
/// 2. Pick the middle level as separator.
/// 3. Nodes in levels before the middle form part A, after form part B.
fn find_separator_bfs(
    graph: &AdjacencyGraph,
    nodes: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    if nodes.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    if nodes.len() <= 2 {
        // Too small to split meaningfully
        return (nodes.to_vec(), Vec::new(), Vec::new());
    }

    let node_set: std::collections::HashSet<usize> = nodes.iter().copied().collect();

    // Find peripheral node in subgraph (minimum degree)
    let start = nodes
        .iter()
        .copied()
        .min_by_key(|&u| {
            graph
                .neighbors(u)
                .iter()
                .filter(|&&v| node_set.contains(&v))
                .count()
        })
        .unwrap_or(nodes[0]);

    // BFS in subgraph
    let mut level = std::collections::HashMap::new();
    let mut queue = VecDeque::new();
    let mut order = Vec::new();
    level.insert(start, 0usize);
    queue.push_back(start);

    while let Some(u) = queue.pop_front() {
        order.push(u);
        let l = level[&u];
        for &v in graph.neighbors(u) {
            if node_set.contains(&v) && !level.contains_key(&v) {
                level.insert(v, l + 1);
                queue.push_back(v);
            }
        }
    }

    // Add any unvisited nodes (disconnected)
    for &u in nodes {
        level.entry(u).or_insert_with(|| {
            order.push(u);
            0
        });
    }

    let max_level = level.values().copied().max().unwrap_or(0);
    if max_level == 0 {
        // All nodes at same level (complete graph or single node)
        let mid = nodes.len() / 2;
        return (
            nodes[..mid].to_vec(),
            vec![nodes[mid]],
            nodes[mid + 1..].to_vec(),
        );
    }

    // Pick middle level as separator
    let sep_level = max_level / 2;

    let mut part_a = Vec::new();
    let mut separator = Vec::new();
    let mut part_b = Vec::new();

    for &u in nodes {
        let l = level.get(&u).copied().unwrap_or(0);
        if l < sep_level {
            part_a.push(u);
        } else if l == sep_level {
            separator.push(u);
        } else {
            part_b.push(u);
        }
    }

    // Ensure separator actually separates: if part_a or part_b is empty,
    // fall back to simple bisection
    if part_a.is_empty() || part_b.is_empty() {
        let mid = order.len() / 2;
        let sep_idx = mid.min(order.len().saturating_sub(1));
        part_a = order[..sep_idx].to_vec();
        separator = vec![order[sep_idx]];
        part_b = if sep_idx + 1 < order.len() {
            order[sep_idx + 1..].to_vec()
        } else {
            Vec::new()
        };
    }

    (part_a, separator, part_b)
}

/// Find a vertex separator using multi-level coarsening.
///
/// 1. Coarsen the subgraph by heavy-edge matching.
/// 2. Bisect the coarsest graph.
/// 3. Uncoarsen and refine with Kernighan-Lin.
/// 4. Extract separator from the partition boundary.
fn find_separator_multilevel(
    graph: &AdjacencyGraph,
    nodes: &[usize],
    config: &NestedDissectionConfig,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    if nodes.len() <= 3 {
        return find_separator_bfs(graph, nodes);
    }

    // Build subgraph
    let (sub_graph, mapping) = graph.subgraph(nodes);
    let sub_n = sub_graph.num_nodes();

    // Coarsen
    let coarsening_levels = coarsen_graph(&sub_graph, config.max_coarsening_levels);

    // Bisect the coarsest graph
    let coarsest = if coarsening_levels.is_empty() {
        &sub_graph
    } else {
        coarsening_levels
            .last()
            .map(|l| &l.coarse_graph)
            .unwrap_or(&sub_graph)
    };

    let mut partition = initial_bisection(coarsest);

    // Uncoarsen and refine
    for level in coarsening_levels.iter().rev() {
        partition = uncoarsen_partition(&partition, &level.fine_to_coarse, level.fine_n);
        kl_refine_partition(&level.coarse_graph, &mut partition, config.kl_passes);
    }

    // If we didn't coarsen at all, refine on the subgraph directly
    if coarsening_levels.is_empty() {
        kl_refine_partition(&sub_graph, &mut partition, config.kl_passes);
    }

    // Extract separator: boundary nodes between the two partitions
    let mut part_a = Vec::new();
    let mut separator = Vec::new();
    let mut part_b = Vec::new();

    // Find boundary nodes (nodes adjacent to the other partition)
    let mut is_boundary = vec![false; sub_n];
    for u in 0..sub_n {
        for &v in sub_graph.neighbors(u) {
            if partition[u] != partition[v] {
                is_boundary[u] = true;
                break;
            }
        }
    }

    // Use boundary nodes from the smaller partition as separator
    let count_0: usize = partition.iter().filter(|&&p| p == 0).count();
    let count_1 = sub_n - count_0;
    let sep_side = if count_0 <= count_1 { 0 } else { 1 };

    for u in 0..sub_n {
        if is_boundary[u] && partition[u] == sep_side {
            separator.push(mapping[u]);
        } else if partition[u] == 0 && (sep_side != 0 || !is_boundary[u]) {
            part_a.push(mapping[u]);
        } else if partition[u] == 1 && (sep_side != 1 || !is_boundary[u]) {
            part_b.push(mapping[u]);
        }
    }

    // Ensure we have a valid split
    if separator.is_empty() {
        // Fallback: use BFS-based separator
        return find_separator_bfs(graph, nodes);
    }

    (part_a, separator, part_b)
}

/// A coarsening level in the multi-level hierarchy.
struct CoarseningLevel {
    /// The coarsened graph.
    coarse_graph: AdjacencyGraph,
    /// Mapping from fine node to coarse node.
    fine_to_coarse: Vec<usize>,
    /// Number of fine nodes.
    fine_n: usize,
}

/// Coarsen a graph by heavy-edge matching.
fn coarsen_graph(graph: &AdjacencyGraph, max_levels: usize) -> Vec<CoarseningLevel> {
    let mut levels = Vec::new();
    let mut current = graph.clone();

    for _ in 0..max_levels {
        let n = current.num_nodes();
        if n <= 16 {
            break; // Small enough
        }

        // Heavy-edge matching: match each unmatched node with its
        // unmatched neighbor of highest degree
        let mut matched = vec![false; n];
        let mut coarse_id = vec![usize::MAX; n];
        let mut next_id = 0usize;

        // Process nodes in order of increasing degree
        let mut node_order: Vec<usize> = (0..n).collect();
        node_order.sort_unstable_by_key(|&u| current.degree(u));

        for &u in &node_order {
            if matched[u] {
                continue;
            }

            // Find best unmatched neighbor (highest degree = heaviest edge heuristic)
            let best_neighbor = current
                .neighbors(u)
                .iter()
                .copied()
                .filter(|&v| !matched[v])
                .max_by_key(|&v| current.degree(v));

            matched[u] = true;
            coarse_id[u] = next_id;

            if let Some(v) = best_neighbor {
                matched[v] = true;
                coarse_id[v] = next_id;
            }

            next_id += 1;
        }

        // Handle any unmatched nodes (shouldn't happen, but be safe)
        for u in 0..n {
            if coarse_id[u] == usize::MAX {
                coarse_id[u] = next_id;
                next_id += 1;
            }
        }

        let coarse_n = next_id;
        if coarse_n >= n * 3 / 4 {
            break; // Not coarsening enough, stop
        }

        // Build coarse graph
        let mut coarse_adj: Vec<Vec<usize>> = vec![Vec::new(); coarse_n];
        for u in 0..n {
            let cu = coarse_id[u];
            for &v in current.neighbors(u) {
                let cv = coarse_id[v];
                if cu != cv {
                    coarse_adj[cu].push(cv);
                }
            }
        }

        let coarse_graph = AdjacencyGraph::from_adjacency_list(coarse_adj);

        levels.push(CoarseningLevel {
            coarse_graph: coarse_graph.clone(),
            fine_to_coarse: coarse_id,
            fine_n: n,
        });

        current = coarse_graph;
    }

    levels
}

/// Initial bisection of a small graph using BFS.
fn initial_bisection(graph: &AdjacencyGraph) -> Vec<usize> {
    let n = graph.num_nodes();
    if n == 0 {
        return Vec::new();
    }

    // BFS from minimum degree node
    let start = (0..n).min_by_key(|&u| graph.degree(u)).unwrap_or(0);

    let mut visited = vec![false; n];
    let mut order = Vec::new();
    let mut queue = VecDeque::new();
    visited[start] = true;
    queue.push_back(start);

    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &v in graph.neighbors(u) {
            if !visited[v] {
                visited[v] = true;
                queue.push_back(v);
            }
        }
    }

    // Add unvisited nodes
    for u in 0..n {
        if !visited[u] {
            order.push(u);
        }
    }

    // First half -> partition 0, second half -> partition 1
    let mid = order.len() / 2;
    let mut partition = vec![0usize; n];
    for &u in &order[mid..] {
        partition[u] = 1;
    }

    partition
}

/// Project a coarse partition back to the fine level.
fn uncoarsen_partition(
    coarse_partition: &[usize],
    fine_to_coarse: &[usize],
    fine_n: usize,
) -> Vec<usize> {
    let mut fine_partition = vec![0usize; fine_n];
    for u in 0..fine_n {
        let cu = fine_to_coarse[u];
        if cu < coarse_partition.len() {
            fine_partition[u] = coarse_partition[cu];
        }
    }
    fine_partition
}

/// Kernighan-Lin refinement of a bisection.
fn kl_refine_partition(graph: &AdjacencyGraph, partition: &mut [usize], max_passes: usize) {
    let n = graph.num_nodes();
    if n < 2 {
        return;
    }

    for _pass in 0..max_passes {
        let mut improved = false;

        // Compute gain for moving each node
        let mut best_gain = 0i64;
        let mut best_node = None;

        for u in 0..n {
            let ext: i64 = graph
                .neighbors(u)
                .iter()
                .filter(|&&v| partition[v] != partition[u])
                .count() as i64;
            let int: i64 = graph
                .neighbors(u)
                .iter()
                .filter(|&&v| partition[v] == partition[u])
                .count() as i64;
            let gain = ext - int;

            if gain > best_gain {
                // Check balance: don't move if it makes partition too unbalanced
                let count_same: usize = partition.iter().filter(|&&p| p == partition[u]).count();
                if count_same > n / 4 {
                    best_gain = gain;
                    best_node = Some(u);
                }
            }
        }

        if let Some(u) = best_node {
            partition[u] = 1 - partition[u];
            improved = true;
        }

        if !improved {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_graph(n: usize) -> AdjacencyGraph {
        let mut adj = vec![Vec::new(); n];
        for i in 0..n.saturating_sub(1) {
            adj[i].push(i + 1);
            adj[i + 1].push(i);
        }
        AdjacencyGraph::from_adjacency_list(adj)
    }

    fn grid_graph(rows: usize, cols: usize) -> AdjacencyGraph {
        let n = rows * cols;
        let mut adj = vec![Vec::new(); n];
        for r in 0..rows {
            for c in 0..cols {
                let u = r * cols + c;
                if c + 1 < cols {
                    let v = r * cols + c + 1;
                    adj[u].push(v);
                    adj[v].push(u);
                }
                if r + 1 < rows {
                    let v = (r + 1) * cols + c;
                    adj[u].push(v);
                    adj[v].push(u);
                }
            }
        }
        AdjacencyGraph::from_adjacency_list(adj)
    }

    #[test]
    fn test_nd_valid_permutation() {
        let graph = path_graph(20);
        let perm = nested_dissection(&graph).expect("ND");
        assert_eq!(perm.len(), 20);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn test_nd_grid_valid_permutation() {
        let graph = grid_graph(4, 4);
        let perm = nested_dissection(&graph).expect("ND grid");
        assert_eq!(perm.len(), 16);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn test_nd_small_graph_uses_amd() {
        // With min_subgraph_size = 64, a 10-node graph goes straight to AMD
        let graph = path_graph(10);
        let config = NestedDissectionConfig {
            min_subgraph_size: 64,
            ..Default::default()
        };
        let perm = nested_dissection_with_config(&graph, &config).expect("ND small");
        assert_eq!(perm.len(), 10);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_nd_large_graph() {
        // Large enough to trigger actual nested dissection
        let graph = grid_graph(10, 10);
        let config = NestedDissectionConfig {
            min_subgraph_size: 16,
            multi_level: true,
            ..Default::default()
        };
        let result = nested_dissection_full(&graph, &config).expect("ND large");
        assert_eq!(result.perm.len(), 100);
        let mut sorted = result.perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn test_nd_empty_graph() {
        let graph = AdjacencyGraph::from_adjacency_list(Vec::new());
        let perm = nested_dissection(&graph).expect("ND empty");
        assert!(perm.is_empty());
    }

    #[test]
    fn test_nd_single_node() {
        let graph = AdjacencyGraph::from_adjacency_list(vec![Vec::new()]);
        let config = NestedDissectionConfig {
            min_subgraph_size: 1,
            multi_level: false,
            ..Default::default()
        };
        let perm = nested_dissection_with_config(&graph, &config).expect("ND single");
        assert_eq!(perm, vec![0]);
    }

    #[test]
    fn test_nd_bfs_separator() {
        // Test the BFS separator directly
        let graph = path_graph(10);
        let nodes: Vec<usize> = (0..10).collect();
        let (a, sep, b) = find_separator_bfs(&graph, &nodes);
        // All nodes should be accounted for
        let total = a.len() + sep.len() + b.len();
        assert_eq!(total, 10, "all nodes must be in some partition");
        // Separator should be non-empty for a path graph of size 10
        assert!(!sep.is_empty(), "separator should be non-empty");
    }

    #[test]
    fn test_nd_disconnected() {
        // Two disconnected components
        let mut adj = vec![Vec::new(); 8];
        for i in 0..3 {
            adj[i].push(i + 1);
            adj[i + 1].push(i);
        }
        for i in 4..7 {
            adj[i].push(i + 1);
            adj[i + 1].push(i);
        }
        let graph = AdjacencyGraph::from_adjacency_list(adj);
        let config = NestedDissectionConfig {
            min_subgraph_size: 2,
            multi_level: false,
            ..Default::default()
        };
        let perm = nested_dissection_with_config(&graph, &config).expect("ND disconnected");
        assert_eq!(perm.len(), 8);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..8).collect::<Vec<_>>());
    }
}
