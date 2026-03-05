//! Parallel graph algorithms for large-scale graph analytics
//!
//! This module provides parallel implementations of fundamental graph algorithms,
//! designed for performance on multi-core systems. All algorithms operate on the
//! [`CsrGraph`] format for cache-friendly, contiguous memory access.
//!
//! # Algorithms
//!
//! - **Parallel BFS**: Level-synchronous breadth-first search
//! - **Parallel Connected Components**: Union-find with path compression
//! - **Parallel PageRank**: Power iteration with parallel SpMV
//! - **Parallel Triangle Counting**: Intersection-based counting
//!
//! All algorithms are feature-gated behind `#[cfg(feature = "parallel")]`.

#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;

use crate::compressed::CsrGraph;
use crate::error::{GraphError, Result};

// ────────────────────────────────────────────────────────────────────────────
// Parallel BFS (Level-Synchronous)
// ────────────────────────────────────────────────────────────────────────────

/// Result of a BFS traversal.
#[derive(Debug, Clone)]
pub struct BfsResult {
    /// Distance from source to each node. `usize::MAX` means unreachable.
    pub distances: Vec<usize>,
    /// Parent of each node in the BFS tree. `usize::MAX` means no parent.
    pub parents: Vec<usize>,
    /// Number of nodes visited.
    pub num_visited: usize,
    /// Number of BFS levels (depth of BFS tree).
    pub num_levels: usize,
}

/// Perform a level-synchronous parallel BFS from a source node.
///
/// At each level, the current frontier is processed in parallel: each node's
/// neighbors are checked, and unvisited neighbors are added to the next frontier.
///
/// # Arguments
/// * `graph` - The CSR graph to traverse
/// * `source` - The source node for BFS
///
/// # Returns
/// A [`BfsResult`] containing distances, parents, and traversal statistics.
///
/// # Complexity
/// - Time: O((V + E) / P) with P threads, plus O(L) synchronization barriers
///   where L is the BFS depth
/// - Space: O(V) for distances, parents, and frontier arrays
#[cfg(feature = "parallel")]
pub fn parallel_bfs(graph: &CsrGraph, source: usize) -> Result<BfsResult> {
    let n = graph.num_nodes();
    if source >= n {
        return Err(GraphError::node_not_found_with_context(
            source,
            n,
            "parallel_bfs source",
        ));
    }

    let not_visited = usize::MAX;
    let mut distances = vec![not_visited; n];
    let mut parents = vec![not_visited; n];

    distances[source] = 0;

    let mut frontier = vec![source];
    let mut num_visited = 1;
    let mut max_level = 0;

    while !frontier.is_empty() {
        let next_level = max_level + 1;

        // Process current frontier in parallel
        // Each thread produces a local list of newly discovered nodes
        let next_frontiers: Vec<Vec<(usize, usize)>> = frontier
            .par_iter()
            .map(|&node| {
                let mut local_discovered = Vec::new();
                for (neighbor, _weight) in graph.neighbors(node) {
                    // We use a relaxed check here; final dedup happens below
                    if distances[neighbor] == not_visited {
                        local_discovered.push((neighbor, node));
                    }
                }
                local_discovered
            })
            .collect();

        // Merge and deduplicate: only the first discovery wins
        let mut next_frontier = Vec::new();
        for discovered in next_frontiers {
            for (neighbor, parent) in discovered {
                if distances[neighbor] == not_visited {
                    distances[neighbor] = next_level;
                    parents[neighbor] = parent;
                    next_frontier.push(neighbor);
                    num_visited += 1;
                }
            }
        }

        if !next_frontier.is_empty() {
            max_level = next_level;
        }
        frontier = next_frontier;
    }

    Ok(BfsResult {
        distances,
        parents,
        num_visited,
        num_levels: max_level,
    })
}

/// Sequential BFS for comparison and non-parallel builds.
pub fn bfs(graph: &CsrGraph, source: usize) -> Result<BfsResult> {
    let n = graph.num_nodes();
    if source >= n {
        return Err(GraphError::node_not_found_with_context(
            source,
            n,
            "bfs source",
        ));
    }

    let not_visited = usize::MAX;
    let mut distances = vec![not_visited; n];
    let mut parents = vec![not_visited; n];

    distances[source] = 0;

    let mut frontier = vec![source];
    let mut num_visited = 1;
    let mut max_level = 0;

    while !frontier.is_empty() {
        let next_level = max_level + 1;
        let mut next_frontier = Vec::new();
        for &node in &frontier {
            for (neighbor, _weight) in graph.neighbors(node) {
                if distances[neighbor] == not_visited {
                    distances[neighbor] = next_level;
                    parents[neighbor] = node;
                    next_frontier.push(neighbor);
                    num_visited += 1;
                }
            }
        }
        if !next_frontier.is_empty() {
            max_level = next_level;
        }
        frontier = next_frontier;
    }

    Ok(BfsResult {
        distances,
        parents,
        num_visited,
        num_levels: max_level,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Parallel Connected Components (Union-Find)
// ────────────────────────────────────────────────────────────────────────────

/// Result of connected components computation.
#[derive(Debug, Clone)]
pub struct ComponentsResult {
    /// Component label for each node. Nodes in the same component share a label.
    pub labels: Vec<usize>,
    /// Number of connected components.
    pub num_components: usize,
    /// Size of each component (indexed by component ID after relabeling).
    pub component_sizes: Vec<usize>,
}

/// Thread-safe union-find structure for parallel connected components.
struct UnionFind {
    parent: Vec<std::sync::atomic::AtomicUsize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        let parent: Vec<_> = (0..n).map(std::sync::atomic::AtomicUsize::new).collect();
        Self { parent }
    }

    fn find(&self, mut x: usize) -> usize {
        loop {
            let p = self.parent[x].load(std::sync::atomic::Ordering::Relaxed);
            if p == x {
                return x;
            }
            let gp = self.parent[p].load(std::sync::atomic::Ordering::Relaxed);
            // Path compression: point x to grandparent
            let _ = self.parent[x].compare_exchange_weak(
                p,
                gp,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            );
            x = gp;
        }
    }

    fn union(&self, x: usize, y: usize) {
        loop {
            let rx = self.find(x);
            let ry = self.find(y);
            if rx == ry {
                return;
            }
            // Always make the smaller root the parent (deterministic)
            let (small, large) = if rx < ry { (rx, ry) } else { (ry, rx) };
            match self.parent[large].compare_exchange_weak(
                large,
                small,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(_) => continue,
            }
        }
    }
}

/// Find connected components using parallel label propagation with union-find.
///
/// Uses a lock-free union-find data structure where each edge `(u, v)` triggers
/// a union operation. Edges are processed in parallel.
///
/// # Arguments
/// * `graph` - An undirected CSR graph
///
/// # Returns
/// A [`ComponentsResult`] with component labels, count, and sizes.
///
/// # Note
/// For directed graphs, this finds weakly connected components (treating edges
/// as undirected).
#[cfg(feature = "parallel")]
pub fn parallel_connected_components(graph: &CsrGraph) -> ComponentsResult {
    let n = graph.num_nodes();
    if n == 0 {
        return ComponentsResult {
            labels: vec![],
            num_components: 0,
            component_sizes: vec![],
        };
    }

    let uf = UnionFind::new(n);

    // Process all edges in parallel
    (0..n).into_par_iter().for_each(|node| {
        for (neighbor, _) in graph.neighbors(node) {
            uf.union(node, neighbor);
        }
    });

    // Finalize: find root for each node
    let labels: Vec<usize> = (0..n).into_par_iter().map(|i| uf.find(i)).collect();

    // Relabel to contiguous IDs and compute sizes
    relabel_components(&labels)
}

/// Sequential connected components (for non-parallel builds).
pub fn connected_components(graph: &CsrGraph) -> ComponentsResult {
    let n = graph.num_nodes();
    if n == 0 {
        return ComponentsResult {
            labels: vec![],
            num_components: 0,
            component_sizes: vec![],
        };
    }

    let not_visited = usize::MAX;
    let mut labels = vec![not_visited; n];
    let mut component_id = 0;

    for start in 0..n {
        if labels[start] != not_visited {
            continue;
        }
        // BFS from start
        let mut queue = vec![start];
        labels[start] = component_id;
        let mut head = 0;
        while head < queue.len() {
            let node = queue[head];
            head += 1;
            for (neighbor, _) in graph.neighbors(node) {
                if labels[neighbor] == not_visited {
                    labels[neighbor] = component_id;
                    queue.push(neighbor);
                }
            }
        }
        component_id += 1;
    }

    let num_components = component_id;
    let mut component_sizes = vec![0usize; num_components];
    for &label in &labels {
        if label < num_components {
            component_sizes[label] += 1;
        }
    }

    ComponentsResult {
        labels,
        num_components,
        component_sizes,
    }
}

/// Relabel component roots to contiguous IDs (0, 1, 2, ...) and compute sizes.
fn relabel_components(raw_labels: &[usize]) -> ComponentsResult {
    let n = raw_labels.len();
    let mut root_to_id = std::collections::HashMap::new();
    let mut next_id = 0usize;
    let mut labels = vec![0usize; n];

    for (i, &root) in raw_labels.iter().enumerate() {
        let id = root_to_id.entry(root).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        labels[i] = *id;
    }

    let num_components = next_id;
    let mut component_sizes = vec![0usize; num_components];
    for &label in &labels {
        component_sizes[label] += 1;
    }

    ComponentsResult {
        labels,
        num_components,
        component_sizes,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Parallel PageRank
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for PageRank computation.
#[derive(Debug, Clone)]
pub struct PageRankConfig {
    /// Damping factor (default: 0.85)
    pub damping: f64,
    /// Maximum iterations (default: 100)
    pub max_iterations: usize,
    /// Convergence tolerance on L1 norm (default: 1e-6)
    pub tolerance: f64,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

/// Result of PageRank computation.
#[derive(Debug, Clone)]
pub struct PageRankResult {
    /// PageRank score for each node.
    pub scores: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final L1 residual.
    pub residual: f64,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// Compute PageRank using parallel power iteration.
///
/// At each iteration:
/// 1. Compute the stochastic matrix column contributions in parallel
/// 2. Apply the PageRank formula: `r = (1-d)/n + d * A_norm * r_old`
/// 3. Check L1 convergence
///
/// Dangling nodes (out-degree 0) distribute their rank equally to all nodes.
///
/// # Arguments
/// * `graph` - A directed CSR graph (or undirected treated as bidirectional)
/// * `config` - PageRank parameters
///
/// # Returns
/// A [`PageRankResult`] with scores and convergence information.
#[cfg(feature = "parallel")]
pub fn parallel_pagerank(graph: &CsrGraph, config: &PageRankConfig) -> Result<PageRankResult> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(PageRankResult {
            scores: vec![],
            iterations: 0,
            residual: 0.0,
            converged: true,
        });
    }

    let d = config.damping;
    let inv_n = 1.0 / n as f64;

    // Compute out-degrees
    let out_degree: Vec<usize> = (0..n).into_par_iter().map(|i| graph.degree(i)).collect();

    // Initialize uniform scores
    let mut scores = vec![inv_n; n];
    let mut new_scores = vec![0.0f64; n];
    let mut iterations = 0;
    let mut residual = f64::MAX;

    while iterations < config.max_iterations && residual > config.tolerance {
        iterations += 1;

        // Compute dangling node contribution (nodes with out-degree 0)
        let dangling_sum: f64 = (0..n)
            .into_par_iter()
            .filter(|&i| out_degree[i] == 0)
            .map(|i| scores[i])
            .sum();

        let teleport = (1.0 - d) * inv_n + d * dangling_sum * inv_n;

        // Parallel SpMV-style computation
        // For each node v, sum scores[u] / out_degree[u] for all in-neighbors u
        // We compute this by iterating outgoing edges: each node u
        // contributes scores[u] / out_degree[u] to each of its neighbors

        // First zero out new_scores
        new_scores.iter_mut().for_each(|x| *x = 0.0);

        // Sequential accumulation (parallel reading, sequential writing)
        // For large graphs, we use a transpose approach:
        // Build contributions per-source, then scatter
        for src in 0..n {
            if out_degree[src] == 0 {
                continue;
            }
            let contrib = scores[src] / out_degree[src] as f64;
            for (dst, _) in graph.neighbors(src) {
                new_scores[dst] += contrib;
            }
        }

        // Apply damping and teleport
        let final_scores: Vec<f64> = new_scores
            .par_iter()
            .map(|&incoming| teleport + d * incoming)
            .collect();

        // Compute L1 residual in parallel
        residual = scores
            .par_iter()
            .zip(final_scores.par_iter())
            .map(|(&old, &new)| (old - new).abs())
            .sum();

        scores = final_scores;
    }

    Ok(PageRankResult {
        scores,
        iterations,
        residual,
        converged: residual <= config.tolerance,
    })
}

/// Sequential PageRank for non-parallel builds.
pub fn pagerank(graph: &CsrGraph, config: &PageRankConfig) -> Result<PageRankResult> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(PageRankResult {
            scores: vec![],
            iterations: 0,
            residual: 0.0,
            converged: true,
        });
    }

    let d = config.damping;
    let inv_n = 1.0 / n as f64;

    let out_degree: Vec<usize> = (0..n).map(|i| graph.degree(i)).collect();

    let mut scores = vec![inv_n; n];
    let mut new_scores = vec![0.0f64; n];
    let mut iterations = 0;
    let mut residual = f64::MAX;

    while iterations < config.max_iterations && residual > config.tolerance {
        iterations += 1;

        let dangling_sum: f64 = (0..n)
            .filter(|&i| out_degree[i] == 0)
            .map(|i| scores[i])
            .sum();

        let teleport = (1.0 - d) * inv_n + d * dangling_sum * inv_n;

        new_scores.iter_mut().for_each(|x| *x = 0.0);

        for src in 0..n {
            if out_degree[src] == 0 {
                continue;
            }
            let contrib = scores[src] / out_degree[src] as f64;
            for (dst, _) in graph.neighbors(src) {
                new_scores[dst] += contrib;
            }
        }

        residual = 0.0;
        for i in 0..n {
            let new_val = teleport + d * new_scores[i];
            residual += (scores[i] - new_val).abs();
            scores[i] = new_val;
        }
    }

    Ok(PageRankResult {
        scores,
        iterations,
        residual,
        converged: residual <= config.tolerance,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Parallel Triangle Counting
// ────────────────────────────────────────────────────────────────────────────

/// Result of triangle counting.
#[derive(Debug, Clone)]
pub struct TriangleCountResult {
    /// Total number of triangles in the graph.
    pub total_triangles: usize,
    /// Number of triangles each node participates in.
    pub per_node_triangles: Vec<usize>,
}

/// Count triangles in an undirected graph using parallel intersection.
///
/// For each edge (u, v) where u < v, count the number of common neighbors w
/// where w > v. This avoids triple-counting.
///
/// For directed graphs, edges are treated as undirected.
///
/// # Complexity
/// - Time: O(m * d_max / P) where m is edges, d_max is max degree, P is threads
/// - Space: O(V) for per-node triangle counts
#[cfg(feature = "parallel")]
pub fn parallel_triangle_count(graph: &CsrGraph) -> TriangleCountResult {
    let n = graph.num_nodes();
    if n < 3 {
        return TriangleCountResult {
            total_triangles: 0,
            per_node_triangles: vec![0; n],
        };
    }

    // For each node, count triangles it participates in
    // We use the approach: for each node u, for each neighbor v > u,
    // count |N(u) intersect N(v)| where neighbors > v
    let per_node: Vec<usize> = (0..n)
        .into_par_iter()
        .map(|u| {
            let mut count = 0;
            let neighbors_u: Vec<usize> = graph.neighbors(u).map(|(v, _)| v).collect();

            for &v in &neighbors_u {
                if v <= u {
                    continue;
                }
                // Count common neighbors w > v
                let neighbors_v: Vec<usize> = graph.neighbors(v).map(|(w, _)| w).collect();

                // Sorted intersection counting
                let mut iu = 0;
                let mut iv = 0;
                while iu < neighbors_u.len() && iv < neighbors_v.len() {
                    let nu = neighbors_u[iu];
                    let nv = neighbors_v[iv];
                    if nu <= v {
                        iu += 1;
                    } else if nv <= v {
                        iv += 1;
                    } else if nu == nv {
                        count += 1;
                        iu += 1;
                        iv += 1;
                    } else if nu < nv {
                        iu += 1;
                    } else {
                        iv += 1;
                    }
                }
            }
            count
        })
        .collect();

    let total: usize = per_node.iter().sum();

    // Each triangle is counted once per node that "owns" it (the smallest node u)
    // But per_node counts for each u the triangles where u is the smallest vertex
    // Total triangles = sum of per_node counts
    // For per_node_triangles, each node participates in triangles discovered by itself
    // AND by smaller nodes. We need a separate pass for per-node participation.

    // Build proper per-node participation counts
    let mut per_node_triangles = vec![0usize; n];

    // Recount with participation tracking
    for u in 0..n {
        let neighbors_u: Vec<usize> = graph.neighbors(u).map(|(v, _)| v).collect();
        for &v in &neighbors_u {
            if v <= u {
                continue;
            }
            let neighbors_v: Vec<usize> = graph.neighbors(v).map(|(w, _)| w).collect();
            let mut iu = 0;
            let mut iv = 0;
            while iu < neighbors_u.len() && iv < neighbors_v.len() {
                let nu = neighbors_u[iu];
                let nv = neighbors_v[iv];
                if nu <= v {
                    iu += 1;
                } else if nv <= v {
                    iv += 1;
                } else if nu == nv {
                    // Triangle: u, v, nu
                    per_node_triangles[u] += 1;
                    per_node_triangles[v] += 1;
                    per_node_triangles[nu] += 1;
                    iu += 1;
                    iv += 1;
                } else if nu < nv {
                    iu += 1;
                } else {
                    iv += 1;
                }
            }
        }
    }

    TriangleCountResult {
        total_triangles: total,
        per_node_triangles,
    }
}

/// Sequential triangle counting.
pub fn triangle_count(graph: &CsrGraph) -> TriangleCountResult {
    let n = graph.num_nodes();
    if n < 3 {
        return TriangleCountResult {
            total_triangles: 0,
            per_node_triangles: vec![0; n],
        };
    }

    let mut total = 0usize;
    let mut per_node_triangles = vec![0usize; n];

    for u in 0..n {
        let neighbors_u: Vec<usize> = graph.neighbors(u).map(|(v, _)| v).collect();
        for &v in &neighbors_u {
            if v <= u {
                continue;
            }
            let neighbors_v: Vec<usize> = graph.neighbors(v).map(|(w, _)| w).collect();

            // Sorted intersection for w > v
            let mut iu = 0;
            let mut iv = 0;
            while iu < neighbors_u.len() && iv < neighbors_v.len() {
                let nu = neighbors_u[iu];
                let nv = neighbors_v[iv];
                if nu <= v {
                    iu += 1;
                } else if nv <= v {
                    iv += 1;
                } else if nu == nv {
                    total += 1;
                    per_node_triangles[u] += 1;
                    per_node_triangles[v] += 1;
                    per_node_triangles[nu] += 1;
                    iu += 1;
                    iv += 1;
                } else if nu < nv {
                    iu += 1;
                } else {
                    iv += 1;
                }
            }
        }
    }

    TriangleCountResult {
        total_triangles: total,
        per_node_triangles,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_path_graph(n: usize) -> CsrGraph {
        let edges: Vec<(usize, usize, f64)> = (0..n - 1).map(|i| (i, i + 1, 1.0)).collect();
        CsrGraph::from_edges(n, edges, false).expect("build path")
    }

    fn make_cycle_graph(n: usize) -> CsrGraph {
        let mut edges: Vec<(usize, usize, f64)> = (0..n - 1).map(|i| (i, i + 1, 1.0)).collect();
        edges.push((n - 1, 0, 1.0));
        CsrGraph::from_edges(n, edges, false).expect("build cycle")
    }

    fn make_complete_graph(n: usize) -> CsrGraph {
        let mut edges = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                edges.push((i, j, 1.0));
            }
        }
        CsrGraph::from_edges(n, edges, false).expect("build complete")
    }

    fn make_disconnected_graph() -> CsrGraph {
        // Two components: {0,1,2} and {3,4}
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0), (3, 4, 1.0)];
        CsrGraph::from_edges(5, edges, false).expect("build disconnected")
    }

    // ── BFS Tests ──

    #[test]
    fn test_bfs_path() {
        let g = make_path_graph(5);
        let result = bfs(&g, 0).expect("bfs");

        assert_eq!(result.distances[0], 0);
        assert_eq!(result.distances[1], 1);
        assert_eq!(result.distances[4], 4);
        assert_eq!(result.num_visited, 5);
        assert_eq!(result.num_levels, 4);
    }

    #[test]
    fn test_bfs_cycle() {
        let g = make_cycle_graph(6);
        let result = bfs(&g, 0).expect("bfs");

        assert_eq!(result.distances[0], 0);
        assert_eq!(result.distances[1], 1);
        assert_eq!(result.distances[3], 3); // 0-1-2-3 or 0-5-4-3
        assert_eq!(result.num_visited, 6);
    }

    #[test]
    fn test_bfs_disconnected() {
        let g = make_disconnected_graph();
        let result = bfs(&g, 0).expect("bfs");

        assert_eq!(result.distances[0], 0);
        assert_eq!(result.distances[1], 1);
        assert_eq!(result.distances[2], 1);
        assert_eq!(result.distances[3], usize::MAX); // unreachable
        assert_eq!(result.distances[4], usize::MAX);
        assert_eq!(result.num_visited, 3);
    }

    #[test]
    fn test_bfs_invalid_source() {
        let g = make_path_graph(3);
        assert!(bfs(&g, 10).is_err());
    }

    #[test]
    fn test_bfs_single_node() {
        let g = CsrGraph::from_edges(1, vec![], false).expect("build");
        let result = bfs(&g, 0).expect("bfs");
        assert_eq!(result.distances[0], 0);
        assert_eq!(result.num_visited, 1);
        assert_eq!(result.num_levels, 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_bfs_path() {
        let g = make_path_graph(10);
        let result = parallel_bfs(&g, 0).expect("parallel bfs");

        assert_eq!(result.distances[0], 0);
        assert_eq!(result.distances[9], 9);
        assert_eq!(result.num_visited, 10);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_bfs_complete() {
        let g = make_complete_graph(5);
        let result = parallel_bfs(&g, 0).expect("parallel bfs");

        // In a complete graph, all nodes are at distance 1 from source
        for i in 1..5 {
            assert_eq!(result.distances[i], 1);
        }
        assert_eq!(result.num_visited, 5);
    }

    // ── Connected Components Tests ──

    #[test]
    fn test_cc_connected() {
        let g = make_path_graph(5);
        let result = connected_components(&g);

        assert_eq!(result.num_components, 1);
        // All nodes should have the same label
        let label = result.labels[0];
        for &l in &result.labels {
            assert_eq!(l, label);
        }
        assert_eq!(result.component_sizes[0], 5);
    }

    #[test]
    fn test_cc_disconnected() {
        let g = make_disconnected_graph();
        let result = connected_components(&g);

        assert_eq!(result.num_components, 2);
        // Nodes 0,1,2 in one component, 3,4 in another
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[0], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_cc_isolated_nodes() {
        let g = CsrGraph::from_edges(5, vec![], false).expect("build");
        let result = connected_components(&g);
        assert_eq!(result.num_components, 5);
        for &size in &result.component_sizes {
            assert_eq!(size, 1);
        }
    }

    #[test]
    fn test_cc_empty() {
        let g = CsrGraph::from_edges(0, vec![], false).expect("build");
        let result = connected_components(&g);
        assert_eq!(result.num_components, 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_cc_disconnected() {
        let g = make_disconnected_graph();
        let result = parallel_connected_components(&g);

        assert_eq!(result.num_components, 2);
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[0], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_cc_connected() {
        let g = make_complete_graph(10);
        let result = parallel_connected_components(&g);
        assert_eq!(result.num_components, 1);
        assert_eq!(result.component_sizes[0], 10);
    }

    // ── PageRank Tests ──

    #[test]
    fn test_pagerank_simple() {
        // Simple directed graph: 0->1, 1->2, 2->0
        let g = CsrGraph::from_edges(3, vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)], true)
            .expect("build");

        let config = PageRankConfig {
            damping: 0.85,
            max_iterations: 100,
            tolerance: 1e-8,
        };
        let result = pagerank(&g, &config).expect("pagerank");

        // Symmetric cycle: all scores should be equal (~1/3)
        let expected = 1.0 / 3.0;
        for &score in &result.scores {
            assert!(
                (score - expected).abs() < 1e-4,
                "score {score} != expected {expected}"
            );
        }
        assert!(result.converged);
    }

    #[test]
    fn test_pagerank_star() {
        // Star graph: center node 0 should have highest rank
        let edges: Vec<(usize, usize, f64)> = (1..5).map(|i| (i, 0, 1.0)).collect();
        let g = CsrGraph::from_edges(5, edges, true).expect("build");

        let result = pagerank(&g, &PageRankConfig::default()).expect("pagerank");
        // Node 0 should have the highest score (all point to it)
        let max_node = result
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(max_node, 0);
    }

    #[test]
    fn test_pagerank_empty() {
        let g = CsrGraph::from_edges(0, vec![], true).expect("build");
        let result = pagerank(&g, &PageRankConfig::default()).expect("pagerank");
        assert!(result.scores.is_empty());
        assert!(result.converged);
    }

    #[test]
    fn test_pagerank_dangling() {
        // Node 2 has no outgoing edges (dangling)
        let g = CsrGraph::from_edges(3, vec![(0, 1, 1.0), (1, 2, 1.0)], true).expect("build");
        let result = pagerank(&g, &PageRankConfig::default()).expect("pagerank");
        // Should still produce valid probabilities summing to ~1
        let total: f64 = result.scores.iter().sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "scores should sum to ~1.0, got {total}"
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_pagerank_cycle() {
        let g = CsrGraph::from_edges(
            4,
            vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)],
            true,
        )
        .expect("build");

        let result = parallel_pagerank(&g, &PageRankConfig::default()).expect("pagerank");

        let expected = 0.25;
        for &score in &result.scores {
            assert!((score - expected).abs() < 1e-4);
        }
        assert!(result.converged);
    }

    // ── Triangle Counting Tests ──

    #[test]
    fn test_triangle_count_k3() {
        let g = make_complete_graph(3);
        let result = triangle_count(&g);
        assert_eq!(result.total_triangles, 1);
        for &count in &result.per_node_triangles {
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn test_triangle_count_k4() {
        let g = make_complete_graph(4);
        let result = triangle_count(&g);
        assert_eq!(result.total_triangles, 4); // C(4,3) = 4
        for &count in &result.per_node_triangles {
            assert_eq!(count, 3); // each node in 3 triangles
        }
    }

    #[test]
    fn test_triangle_count_k5() {
        let g = make_complete_graph(5);
        let result = triangle_count(&g);
        assert_eq!(result.total_triangles, 10); // C(5,3) = 10
    }

    #[test]
    fn test_triangle_count_path() {
        let g = make_path_graph(5);
        let result = triangle_count(&g);
        assert_eq!(result.total_triangles, 0); // No triangles in a path
    }

    #[test]
    fn test_triangle_count_single_triangle() {
        let g = CsrGraph::from_edges(
            4,
            vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0), (2, 3, 1.0)],
            false,
        )
        .expect("build");
        let result = triangle_count(&g);
        assert_eq!(result.total_triangles, 1);
        assert_eq!(result.per_node_triangles[0], 1);
        assert_eq!(result.per_node_triangles[1], 1);
        assert_eq!(result.per_node_triangles[2], 1);
        assert_eq!(result.per_node_triangles[3], 0); // not in any triangle
    }

    #[test]
    fn test_triangle_count_empty() {
        let g = CsrGraph::from_edges(2, vec![], false).expect("build");
        let result = triangle_count(&g);
        assert_eq!(result.total_triangles, 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_triangle_count_k4() {
        let g = make_complete_graph(4);
        let result = parallel_triangle_count(&g);
        assert_eq!(result.total_triangles, 4);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_triangle_count_k5() {
        let g = make_complete_graph(5);
        let result = parallel_triangle_count(&g);
        assert_eq!(result.total_triangles, 10);
    }

    // ── Consistency Tests ──

    #[test]
    fn test_bfs_parent_tree() {
        let g = make_path_graph(5);
        let result = bfs(&g, 0).expect("bfs");

        // Verify parent tree structure
        assert_eq!(result.parents[0], usize::MAX); // source has no parent
        for i in 1..5 {
            let parent = result.parents[i];
            assert!(parent < 5);
            assert_eq!(result.distances[i], result.distances[parent] + 1);
        }
    }

    #[test]
    fn test_cc_label_consistency() {
        let g = make_disconnected_graph();
        let result = connected_components(&g);

        // Verify: nodes connected by an edge share a label
        for node in 0..g.num_nodes() {
            for (neighbor, _) in g.neighbors(node) {
                assert_eq!(
                    result.labels[node], result.labels[neighbor],
                    "nodes {node} and {neighbor} should be in same component"
                );
            }
        }
    }

    #[test]
    fn test_pagerank_scores_sum_to_one() {
        let g = make_complete_graph(5);
        // Complete graph treated as directed (both directions stored)
        let config = PageRankConfig {
            max_iterations: 200,
            tolerance: 1e-10,
            ..Default::default()
        };
        let result = pagerank(&g, &config).expect("pagerank");
        let total: f64 = result.scores.iter().sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "PageRank scores should sum to ~1.0, got {total}"
        );
    }
}
