//! Graph coloring algorithms for parallel computation
//!
//! Graph coloring assigns colors to vertices such that no two adjacent vertices
//! share the same color. This is useful for parallel Gauss-Seidel iterations
//! (nodes of the same color can be processed in parallel) and for computing
//! sparse Jacobians via column grouping (distance-2 coloring).
//!
//! # Algorithms
//!
//! - **Greedy coloring**: Simple greedy with configurable ordering strategies.
//! - **DSatur (Degree of Saturation)**: Colors the vertex with the most
//!   already-colored neighbors first, breaking ties by degree.
//! - **Distance-2 coloring**: Ensures vertices within distance 2 have
//!   distinct colors (useful for Jacobian computation).
//!
//! # References
//!
//! - D. Brelaz, "New methods to color the vertices of a graph",
//!   Communications of the ACM, 22(4), 1979.
//! - A.H. Gebremedhin et al., "What color is your Jacobian?",
//!   SIAM Review, 47(4), 2005.

use super::adjacency::AdjacencyGraph;
use crate::error::{SparseError, SparseResult};

/// Result of a graph coloring algorithm.
#[derive(Debug, Clone)]
pub struct ColoringResult {
    /// Color assigned to each vertex: `colors[v]` is the color of vertex `v`.
    /// Colors are numbered starting from 0.
    pub colors: Vec<usize>,
    /// Total number of distinct colors used.
    pub num_colors: usize,
}

/// Ordering strategy for greedy graph coloring.
#[derive(Debug, Clone, Copy)]
pub enum GreedyOrdering {
    /// Process vertices in natural order (0, 1, 2, ...).
    Natural,
    /// Process vertices in order of decreasing degree (largest-first).
    LargestFirst,
    /// Process vertices using smallest-last ordering:
    /// repeatedly remove the vertex of smallest degree, reverse the sequence.
    SmallestLast,
}

/// Compute a greedy graph coloring with the specified ordering strategy.
///
/// The greedy algorithm assigns each vertex (in the given order) the smallest
/// color not used by any of its already-colored neighbors.
///
/// # Guarantees
///
/// - Uses at most Delta + 1 colors, where Delta is the maximum degree.
/// - The actual number of colors depends heavily on the vertex ordering.
///
/// # Arguments
///
/// * `graph` - The adjacency graph to color.
/// * `ordering` - The ordering strategy to use.
///
/// # Returns
///
/// A `ColoringResult` with the color assignment and number of colors used.
pub fn greedy_color(
    graph: &AdjacencyGraph,
    ordering: GreedyOrdering,
) -> SparseResult<ColoringResult> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(ColoringResult {
            colors: Vec::new(),
            num_colors: 0,
        });
    }

    let order = match ordering {
        GreedyOrdering::Natural => (0..n).collect::<Vec<_>>(),
        GreedyOrdering::LargestFirst => {
            let mut ord: Vec<usize> = (0..n).collect();
            ord.sort_unstable_by_key(|&v| std::cmp::Reverse(graph.degree(v)));
            ord
        }
        GreedyOrdering::SmallestLast => smallest_last_ordering(graph),
    };

    let mut colors = vec![usize::MAX; n];
    let mut num_colors = 0usize;

    // Temporary buffer for tracking which colors are used by neighbors
    let mut used = vec![false; n + 1];

    for &v in &order {
        // Mark colors of already-colored neighbors
        for &nbr in graph.neighbors(v) {
            if colors[nbr] != usize::MAX && colors[nbr] < used.len() {
                used[colors[nbr]] = true;
            }
        }

        // Find the smallest available color
        let mut c = 0;
        while c < used.len() && used[c] {
            c += 1;
        }
        colors[v] = c;
        if c + 1 > num_colors {
            num_colors = c + 1;
        }

        // Reset the used buffer
        for &nbr in graph.neighbors(v) {
            if colors[nbr] != usize::MAX && colors[nbr] < used.len() {
                used[colors[nbr]] = false;
            }
        }
    }

    Ok(ColoringResult { colors, num_colors })
}

/// Compute a DSatur (Degree of Saturation) graph coloring.
///
/// DSatur colors the vertex with the highest saturation degree first
/// (the number of distinct colors in its neighborhood). Ties are broken
/// by the vertex degree (higher degree first).
///
/// DSatur typically produces better colorings than simple greedy with
/// a fixed ordering, especially on irregular graphs.
///
/// # Arguments
///
/// * `graph` - The adjacency graph to color.
///
/// # Returns
///
/// A `ColoringResult` with the color assignment and number of colors used.
pub fn dsatur_color(graph: &AdjacencyGraph) -> SparseResult<ColoringResult> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(ColoringResult {
            colors: Vec::new(),
            num_colors: 0,
        });
    }

    let mut colors = vec![usize::MAX; n];
    let mut num_colors = 0usize;
    let mut colored_count = 0usize;

    // Saturation degree: number of distinct colors in the neighborhood
    let mut saturation = vec![0usize; n];
    // Set of colors used by neighbors (using bitvector approximation)
    let mut neighbor_colors: Vec<Vec<bool>> = vec![Vec::new(); n];

    // Temporary buffer for greedy coloring
    let mut used = vec![false; n + 1];

    while colored_count < n {
        // Pick the uncolored vertex with highest saturation, break ties by degree
        let v = (0..n)
            .filter(|&u| colors[u] == usize::MAX)
            .max_by(|&a, &b| {
                saturation[a]
                    .cmp(&saturation[b])
                    .then_with(|| graph.degree(a).cmp(&graph.degree(b)))
            });

        let v = match v {
            Some(v) => v,
            None => break, // All colored (shouldn't reach here)
        };

        // Find smallest available color
        for &nbr in graph.neighbors(v) {
            if colors[nbr] != usize::MAX && colors[nbr] < used.len() {
                used[colors[nbr]] = true;
            }
        }

        let mut c = 0;
        while c < used.len() && used[c] {
            c += 1;
        }
        colors[v] = c;
        if c + 1 > num_colors {
            num_colors = c + 1;
        }
        colored_count += 1;

        // Reset used buffer
        for &nbr in graph.neighbors(v) {
            if colors[nbr] != usize::MAX && colors[nbr] < used.len() {
                used[colors[nbr]] = false;
            }
        }

        // Update saturation of uncolored neighbors
        for &nbr in graph.neighbors(v) {
            if colors[nbr] == usize::MAX {
                // Add color c to nbr's neighbor color set
                if neighbor_colors[nbr].len() <= c {
                    neighbor_colors[nbr].resize(c + 1, false);
                }
                if !neighbor_colors[nbr][c] {
                    neighbor_colors[nbr][c] = true;
                    saturation[nbr] += 1;
                }
            }
        }
    }

    Ok(ColoringResult { colors, num_colors })
}

/// Compute a distance-2 graph coloring.
///
/// In a distance-2 coloring, any two vertices within distance 2 in the graph
/// must have different colors. This is equivalent to coloring the graph G^2
/// (where G^2 has edges between any vertices at distance <= 2 in G).
///
/// Distance-2 coloring is used for computing sparse Jacobians: columns
/// with the same color can be evaluated simultaneously.
///
/// # Arguments
///
/// * `graph` - The adjacency graph.
///
/// # Returns
///
/// A `ColoringResult` for the distance-2 coloring.
pub fn distance2_color(graph: &AdjacencyGraph) -> SparseResult<ColoringResult> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(ColoringResult {
            colors: Vec::new(),
            num_colors: 0,
        });
    }

    // Build distance-2 neighbor sets
    let mut dist2_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for u in 0..n {
        let mut seen = vec![false; n];
        seen[u] = true;
        // Distance-1 neighbors
        for &v in graph.neighbors(u) {
            if !seen[v] {
                seen[v] = true;
                dist2_neighbors[u].push(v);
            }
            // Distance-2 neighbors (neighbors of neighbors)
            for &w in graph.neighbors(v) {
                if !seen[w] {
                    seen[w] = true;
                    dist2_neighbors[u].push(w);
                }
            }
        }
    }

    // Apply greedy coloring on the distance-2 adjacency
    let mut colors = vec![usize::MAX; n];
    let mut num_colors = 0usize;
    let max_d2_degree = dist2_neighbors.iter().map(|v| v.len()).max().unwrap_or(0);
    let mut used = vec![false; max_d2_degree + 2];

    // Use largest-first ordering on d2 graph
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| dist2_neighbors[b].len().cmp(&dist2_neighbors[a].len()));

    for &v in &order {
        for &nbr in &dist2_neighbors[v] {
            if colors[nbr] != usize::MAX && colors[nbr] < used.len() {
                used[colors[nbr]] = true;
            }
        }

        let mut c = 0;
        while c < used.len() && used[c] {
            c += 1;
        }
        colors[v] = c;
        if c + 1 > num_colors {
            num_colors = c + 1;
        }

        for &nbr in &dist2_neighbors[v] {
            if colors[nbr] != usize::MAX && colors[nbr] < used.len() {
                used[colors[nbr]] = false;
            }
        }
    }

    Ok(ColoringResult { colors, num_colors })
}

/// Verify that a coloring is valid (no two adjacent vertices share a color).
pub fn verify_coloring(graph: &AdjacencyGraph, colors: &[usize]) -> SparseResult<bool> {
    let n = graph.num_nodes();
    if colors.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: colors.len(),
        });
    }

    for u in 0..n {
        for &v in graph.neighbors(u) {
            if colors[u] == colors[v] {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Verify that a distance-2 coloring is valid.
pub fn verify_distance2_coloring(graph: &AdjacencyGraph, colors: &[usize]) -> SparseResult<bool> {
    let n = graph.num_nodes();
    if colors.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: colors.len(),
        });
    }

    for u in 0..n {
        for &v in graph.neighbors(u) {
            if colors[u] == colors[v] {
                return Ok(false);
            }
            // Check distance-2: neighbors of v
            for &w in graph.neighbors(v) {
                if w != u && colors[u] == colors[w] {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

/// Compute the smallest-last vertex ordering.
///
/// Repeatedly remove the vertex with the smallest degree from the remaining
/// graph, then reverse the removal sequence.
fn smallest_last_ordering(graph: &AdjacencyGraph) -> Vec<usize> {
    let n = graph.num_nodes();
    let mut removed = vec![false; n];
    let mut order = Vec::with_capacity(n);

    // Current degrees (will decrease as nodes are removed)
    let mut deg: Vec<usize> = (0..n).map(|u| graph.degree(u)).collect();

    for _ in 0..n {
        // Find unremoved vertex with minimum current degree
        let v = (0..n).filter(|&u| !removed[u]).min_by_key(|&u| deg[u]);

        let v = match v {
            Some(v) => v,
            None => break,
        };

        order.push(v);
        removed[v] = true;

        // Update degrees of neighbors
        for &nbr in graph.neighbors(v) {
            if !removed[nbr] {
                deg[nbr] = deg[nbr].saturating_sub(1);
            }
        }
    }

    // Reverse to get smallest-last ordering
    order.reverse();
    order
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

    fn cycle_graph(n: usize) -> AdjacencyGraph {
        let mut adj = vec![Vec::new(); n];
        for i in 0..n {
            adj[i].push((i + 1) % n);
            adj[(i + 1) % n].push(i);
        }
        AdjacencyGraph::from_adjacency_list(adj)
    }

    fn complete_graph(n: usize) -> AdjacencyGraph {
        let mut adj = vec![Vec::new(); n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    adj[i].push(j);
                }
            }
        }
        AdjacencyGraph::from_adjacency_list(adj)
    }

    fn petersen_graph() -> AdjacencyGraph {
        // The Petersen graph has 10 vertices and chromatic number 3
        let edges = vec![
            (0, 1),
            (0, 4),
            (0, 5),
            (1, 2),
            (1, 6),
            (2, 3),
            (2, 7),
            (3, 4),
            (3, 8),
            (4, 9),
            (5, 7),
            (5, 8),
            (6, 8),
            (6, 9),
            (7, 9),
        ];
        let mut adj = vec![Vec::new(); 10];
        for (u, v) in edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        AdjacencyGraph::from_adjacency_list(adj)
    }

    #[test]
    fn test_greedy_natural_valid() {
        let graph = path_graph(10);
        let result = greedy_color(&graph, GreedyOrdering::Natural).expect("greedy");
        assert!(verify_coloring(&graph, &result.colors).expect("verify"));
        // Path graph is 2-colorable
        assert!(result.num_colors <= 2);
    }

    #[test]
    fn test_greedy_largest_first_valid() {
        let graph = cycle_graph(7); // Odd cycle needs 3 colors
        let result = greedy_color(&graph, GreedyOrdering::LargestFirst).expect("greedy LF");
        assert!(verify_coloring(&graph, &result.colors).expect("verify"));
        assert!(result.num_colors <= 4); // At most Delta+1 = 3
    }

    #[test]
    fn test_greedy_smallest_last_valid() {
        let graph = petersen_graph();
        let result = greedy_color(&graph, GreedyOrdering::SmallestLast).expect("greedy SL");
        assert!(verify_coloring(&graph, &result.colors).expect("verify"));
        // Petersen has max degree 3, so at most 4 colors
        assert!(result.num_colors <= 4);
    }

    #[test]
    fn test_greedy_complete_graph() {
        let graph = complete_graph(5);
        let result = greedy_color(&graph, GreedyOrdering::Natural).expect("greedy K5");
        assert!(verify_coloring(&graph, &result.colors).expect("verify"));
        // Complete graph K5 needs exactly 5 colors
        assert_eq!(result.num_colors, 5);
    }

    #[test]
    fn test_greedy_delta_plus_one_bound() {
        let graph = petersen_graph(); // max degree = 3
        let result = greedy_color(&graph, GreedyOrdering::Natural).expect("greedy");
        assert!(verify_coloring(&graph, &result.colors).expect("verify"));
        let max_degree = (0..graph.num_nodes())
            .map(|u| graph.degree(u))
            .max()
            .unwrap_or(0);
        assert!(
            result.num_colors <= max_degree + 1,
            "greedy should use at most Delta+1 = {} colors, used {}",
            max_degree + 1,
            result.num_colors
        );
    }

    #[test]
    fn test_dsatur_valid() {
        let graph = petersen_graph();
        let result = dsatur_color(&graph).expect("dsatur");
        assert!(verify_coloring(&graph, &result.colors).expect("verify"));
    }

    #[test]
    fn test_dsatur_bipartite() {
        // Even cycle is bipartite -> 2 colors
        let graph = cycle_graph(8);
        let result = dsatur_color(&graph).expect("dsatur bipartite");
        assert!(verify_coloring(&graph, &result.colors).expect("verify"));
        assert_eq!(result.num_colors, 2);
    }

    #[test]
    fn test_dsatur_better_than_naive_on_crown() {
        // Crown graph: two sets of n vertices, with perfect matching
        // between them minus the identity matching.
        // DSatur should do well here.
        let n = 6; // 2*6 = 12 nodes total
        let total = 2 * n;
        let mut adj = vec![Vec::new(); total];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    adj[i].push(n + j);
                    adj[n + j].push(i);
                }
            }
        }
        let graph = AdjacencyGraph::from_adjacency_list(adj);

        let greedy_result = greedy_color(&graph, GreedyOrdering::Natural).expect("greedy crown");
        let dsatur_result = dsatur_color(&graph).expect("dsatur crown");

        assert!(verify_coloring(&graph, &greedy_result.colors).expect("verify greedy"));
        assert!(verify_coloring(&graph, &dsatur_result.colors).expect("verify dsatur"));

        // DSatur should use <= the number of colors greedy uses
        assert!(
            dsatur_result.num_colors <= greedy_result.num_colors,
            "DSatur ({}) should be no worse than greedy ({})",
            dsatur_result.num_colors,
            greedy_result.num_colors
        );
    }

    #[test]
    fn test_distance2_coloring() {
        let graph = path_graph(5);
        let result = distance2_color(&graph).expect("d2 color");
        assert!(verify_distance2_coloring(&graph, &result.colors).expect("verify d2"));
        // Path graph needs 3 colors for distance-2 coloring
        assert!(result.num_colors <= 3);
    }

    #[test]
    fn test_distance2_cycle() {
        let graph = cycle_graph(6);
        let result = distance2_color(&graph).expect("d2 cycle");
        assert!(verify_distance2_coloring(&graph, &result.colors).expect("verify d2"));
    }

    #[test]
    fn test_empty_graph_coloring() {
        let graph = AdjacencyGraph::from_adjacency_list(Vec::new());
        let result = greedy_color(&graph, GreedyOrdering::Natural).expect("empty");
        assert_eq!(result.num_colors, 0);
        assert!(result.colors.is_empty());
    }

    #[test]
    fn test_single_node_coloring() {
        let graph = AdjacencyGraph::from_adjacency_list(vec![Vec::new()]);
        let result = dsatur_color(&graph).expect("single");
        assert_eq!(result.num_colors, 1);
        assert_eq!(result.colors, vec![0]);
    }
}
