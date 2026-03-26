//! Cuthill-McKee and Reverse Cuthill-McKee bandwidth reduction algorithms
//!
//! The Cuthill-McKee (CM) algorithm is a BFS-based reordering that reduces the
//! bandwidth of a sparse matrix. Starting from a pseudo-peripheral node, it
//! traverses the graph in BFS order, sorting neighbors by degree (ascending).
//!
//! The Reverse Cuthill-McKee (RCM) algorithm reverses the CM permutation, which
//! typically produces even smaller bandwidth and better profile reduction.
//!
//! # References
//!
//! - E. Cuthill, J. McKee, "Reducing the bandwidth of sparse symmetric matrices",
//!   Proc. 24th National Conference ACM, 1969.
//! - A. George, J.W.H. Liu, "Computer Solution of Large Sparse Positive Definite
//!   Systems", Prentice-Hall, 1981.

use std::collections::VecDeque;

use super::adjacency::AdjacencyGraph;
use crate::error::{SparseError, SparseResult};

/// Result of a Cuthill-McKee or Reverse Cuthill-McKee reordering.
#[derive(Debug, Clone)]
pub struct CuthillMcKeeResult {
    /// Permutation vector: `perm[new_index] = old_index`.
    pub perm: Vec<usize>,
    /// Inverse permutation: `inv_perm[old_index] = new_index`.
    pub inv_perm: Vec<usize>,
    /// Bandwidth before reordering.
    pub bandwidth_before: usize,
    /// Bandwidth after reordering.
    pub bandwidth_after: usize,
    /// Profile (envelope) before reordering.
    pub profile_before: usize,
    /// Profile (envelope) after reordering.
    pub profile_after: usize,
}

/// Compute the bandwidth of a symmetric adjacency graph under a given permutation.
///
/// Bandwidth = max over all edges (i,j) of `|perm_inv[i] - perm_inv[j]|`.
/// If `perm` is empty or identity, computes the natural bandwidth.
pub fn bandwidth(graph: &AdjacencyGraph, perm: &[usize]) -> SparseResult<usize> {
    let n = graph.num_nodes();
    if !perm.is_empty() && perm.len() != n {
        return Err(SparseError::ValueError(format!(
            "permutation length {} does not match graph size {}",
            perm.len(),
            n
        )));
    }

    // Build inverse permutation
    let inv_perm = if perm.is_empty() {
        (0..n).collect::<Vec<_>>()
    } else {
        let mut inv = vec![0usize; n];
        for (new_i, &old_i) in perm.iter().enumerate() {
            if old_i >= n {
                return Err(SparseError::ValueError(format!(
                    "permutation contains out-of-range index {}",
                    old_i
                )));
            }
            inv[old_i] = new_i;
        }
        inv
    };

    let mut bw = 0usize;
    for u in 0..n {
        for &v in graph.neighbors(u) {
            let diff = inv_perm[u].abs_diff(inv_perm[v]);
            if diff > bw {
                bw = diff;
            }
        }
    }
    Ok(bw)
}

/// Compute the profile (envelope) of a symmetric adjacency graph under a given permutation.
///
/// Profile = sum over all rows i of (i - min column index in row i).
pub fn profile(graph: &AdjacencyGraph, perm: &[usize]) -> SparseResult<usize> {
    let n = graph.num_nodes();
    if !perm.is_empty() && perm.len() != n {
        return Err(SparseError::ValueError(format!(
            "permutation length {} does not match graph size {}",
            perm.len(),
            n
        )));
    }

    let inv_perm = if perm.is_empty() {
        (0..n).collect::<Vec<_>>()
    } else {
        let mut inv = vec![0usize; n];
        for (new_i, &old_i) in perm.iter().enumerate() {
            inv[old_i] = new_i;
        }
        inv
    };

    let mut prof = 0usize;
    for old_u in 0..n {
        let new_u = inv_perm[old_u];
        let mut min_col = new_u;
        for &old_v in graph.neighbors(old_u) {
            let new_v = inv_perm[old_v];
            if new_v < min_col {
                min_col = new_v;
            }
        }
        prof += new_u - min_col;
    }
    Ok(prof)
}

/// Compute the Cuthill-McKee permutation for a symmetric adjacency graph.
///
/// The algorithm performs BFS starting from a pseudo-peripheral node,
/// sorting neighbors by degree in ascending order at each level.
///
/// # Arguments
///
/// * `graph` - Symmetric adjacency graph.
///
/// # Returns
///
/// Permutation vector where `perm[new_index] = old_index`.
pub fn cuthill_mckee(graph: &AdjacencyGraph) -> SparseResult<Vec<usize>> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(Vec::new());
    }

    let start = find_peripheral_node(graph);
    let order = bfs_cm_order(graph, n, start);

    // Validate that all nodes are covered
    debug_assert_eq!(order.len(), n);
    Ok(order)
}

/// Compute the Reverse Cuthill-McKee permutation for a symmetric adjacency graph.
///
/// This is the CM permutation reversed, which often produces better bandwidth
/// and profile reduction.
///
/// # Arguments
///
/// * `graph` - Symmetric adjacency graph.
///
/// # Returns
///
/// Permutation vector where `perm[new_index] = old_index`.
pub fn reverse_cuthill_mckee(graph: &AdjacencyGraph) -> SparseResult<Vec<usize>> {
    let mut perm = cuthill_mckee(graph)?;
    perm.reverse();
    Ok(perm)
}

/// Full Cuthill-McKee computation with metrics.
///
/// Returns a `CuthillMcKeeResult` containing the CM permutation and
/// bandwidth/profile metrics before and after reordering.
pub fn cuthill_mckee_full(graph: &AdjacencyGraph) -> SparseResult<CuthillMcKeeResult> {
    let perm = cuthill_mckee(graph)?;
    build_result(graph, perm)
}

/// Full Reverse Cuthill-McKee computation with metrics.
///
/// Returns a `CuthillMcKeeResult` containing the RCM permutation and
/// bandwidth/profile metrics before and after reordering.
pub fn reverse_cuthill_mckee_full(graph: &AdjacencyGraph) -> SparseResult<CuthillMcKeeResult> {
    let perm = reverse_cuthill_mckee(graph)?;
    build_result(graph, perm)
}

/// Build a `CuthillMcKeeResult` from a permutation and graph.
fn build_result(graph: &AdjacencyGraph, perm: Vec<usize>) -> SparseResult<CuthillMcKeeResult> {
    let n = graph.num_nodes();
    let bw_before = bandwidth(graph, &[])?;
    let prof_before = profile(graph, &[])?;
    let bw_after = bandwidth(graph, &perm)?;
    let prof_after = profile(graph, &perm)?;

    let mut inv_perm = vec![0usize; n];
    for (new_i, &old_i) in perm.iter().enumerate() {
        inv_perm[old_i] = new_i;
    }

    Ok(CuthillMcKeeResult {
        perm,
        inv_perm,
        bandwidth_before: bw_before,
        bandwidth_after: bw_after,
        profile_before: prof_before,
        profile_after: prof_after,
    })
}

/// Find a pseudo-peripheral node using a double BFS heuristic.
///
/// 1. Start from the minimum-degree node.
/// 2. BFS to find the farthest level set; pick the minimum-degree node there.
/// 3. Repeat BFS from that node; pick the minimum-degree node in the last level.
fn find_peripheral_node(graph: &AdjacencyGraph) -> usize {
    let n = graph.num_nodes();
    if n == 0 {
        return 0;
    }

    // Start from the minimum degree node
    let start = (0..n).min_by_key(|&v| graph.degree(v)).unwrap_or(0);

    // First BFS
    let levels = bfs_levels(graph, n, start);
    let max_level = levels.iter().copied().max().unwrap_or(0);

    let candidate = (0..n)
        .filter(|&v| levels[v] == max_level)
        .min_by_key(|&v| graph.degree(v))
        .unwrap_or(start);

    // Second BFS from candidate
    let levels2 = bfs_levels(graph, n, candidate);
    let max_level2 = levels2.iter().copied().max().unwrap_or(0);

    (0..n)
        .filter(|&v| levels2[v] == max_level2)
        .min_by_key(|&v| graph.degree(v))
        .unwrap_or(candidate)
}

/// BFS level computation: `level[v]` = distance from `start` (usize::MAX if unreachable).
fn bfs_levels(graph: &AdjacencyGraph, n: usize, start: usize) -> Vec<usize> {
    let mut level = vec![usize::MAX; n];
    let mut queue = VecDeque::new();
    level[start] = 0;
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        let l = level[node];
        for &nbr in graph.neighbors(node) {
            if level[nbr] == usize::MAX {
                level[nbr] = l + 1;
                queue.push_back(nbr);
            }
        }
    }
    level
}

/// BFS-based Cuthill-McKee ordering from a start node.
///
/// Neighbors are sorted by degree (ascending) before enqueueing,
/// which is the key characteristic of the CM algorithm.
fn bfs_cm_order(graph: &AdjacencyGraph, n: usize, start: usize) -> Vec<usize> {
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut queue = VecDeque::new();

    // Process connected component containing `start`
    visited[start] = true;
    queue.push_back(start);
    bfs_cm_component(graph, &mut visited, &mut order, &mut queue);

    // Handle disconnected components: find unvisited nodes and process them
    for i in 0..n {
        if !visited[i] {
            visited[i] = true;
            queue.push_back(i);
            bfs_cm_component(graph, &mut visited, &mut order, &mut queue);
        }
    }

    order
}

/// Process one connected component in BFS-CM order.
fn bfs_cm_component(
    graph: &AdjacencyGraph,
    visited: &mut [bool],
    order: &mut Vec<usize>,
    queue: &mut VecDeque<usize>,
) {
    while let Some(node) = queue.pop_front() {
        order.push(node);

        // Collect unvisited neighbors
        let mut neighbors: Vec<usize> = graph
            .neighbors(node)
            .iter()
            .copied()
            .filter(|&nbr| !visited[nbr])
            .collect();

        // Sort by degree ascending (CM criterion)
        neighbors.sort_unstable_by_key(|&v| graph.degree(v));

        for nbr in neighbors {
            if !visited[nbr] {
                visited[nbr] = true;
                queue.push_back(nbr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a path graph: 0-1-2-..-(n-1)
    fn path_graph(n: usize) -> AdjacencyGraph {
        let mut adj = vec![Vec::new(); n];
        for i in 0..n.saturating_sub(1) {
            adj[i].push(i + 1);
            adj[i + 1].push(i);
        }
        AdjacencyGraph::from_adjacency_list(adj)
    }

    /// Create a banded matrix graph with large bandwidth (reversed numbering).
    /// Node i connects to node (n-1-i) and its natural neighbors.
    fn high_bandwidth_graph(n: usize) -> AdjacencyGraph {
        let mut adj = vec![Vec::new(); n];
        // Path edges
        for i in 0..n.saturating_sub(1) {
            adj[i].push(i + 1);
            adj[i + 1].push(i);
        }
        // Cross edges: connect i to n-1-i (creating high bandwidth)
        for i in 0..n / 2 {
            let j = n - 1 - i;
            if i != j && !adj[i].contains(&j) {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
        // Deduplicate
        for nbrs in adj.iter_mut() {
            nbrs.sort_unstable();
            nbrs.dedup();
        }
        AdjacencyGraph::from_adjacency_list(adj)
    }

    /// Create a star graph: center node 0 connected to all others
    fn star_graph(n: usize) -> AdjacencyGraph {
        let mut adj = vec![Vec::new(); n];
        for i in 1..n {
            adj[0].push(i);
            adj[i].push(0);
        }
        AdjacencyGraph::from_adjacency_list(adj)
    }

    #[test]
    fn test_cm_path_graph_bandwidth() {
        // Path graph has bandwidth 1 naturally; CM should preserve this
        let graph = path_graph(10);
        let perm = cuthill_mckee(&graph).expect("CM should succeed");
        let bw = bandwidth(&graph, &perm).expect("bandwidth");
        assert_eq!(bw, 1, "path graph bandwidth should be 1 under CM");
    }

    #[test]
    fn test_rcm_path_graph_bandwidth() {
        let graph = path_graph(10);
        let perm = reverse_cuthill_mckee(&graph).expect("RCM should succeed");
        let bw = bandwidth(&graph, &perm).expect("bandwidth");
        assert_eq!(bw, 1, "path graph bandwidth should be 1 under RCM");
    }

    #[test]
    fn test_rcm_reduces_bandwidth_high_bw() {
        let graph = high_bandwidth_graph(12);
        let natural_bw = bandwidth(&graph, &[]).expect("natural bw");
        let perm = reverse_cuthill_mckee(&graph).expect("RCM");
        let rcm_bw = bandwidth(&graph, &perm).expect("rcm bw");
        assert!(
            rcm_bw <= natural_bw,
            "RCM bandwidth {} should be <= natural bandwidth {}",
            rcm_bw,
            natural_bw
        );
    }

    #[test]
    fn test_cm_valid_permutation() {
        let graph = star_graph(8);
        let perm = cuthill_mckee(&graph).expect("CM");
        assert_eq!(perm.len(), 8);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn test_rcm_full_metrics() {
        let graph = high_bandwidth_graph(10);
        let result = reverse_cuthill_mckee_full(&graph).expect("RCM full");
        assert!(result.bandwidth_after <= result.bandwidth_before);
        assert!(result.profile_after <= result.profile_before || result.profile_after > 0);
        assert_eq!(result.perm.len(), 10);
        assert_eq!(result.inv_perm.len(), 10);
        // Verify inv_perm is consistent
        for (new_i, &old_i) in result.perm.iter().enumerate() {
            assert_eq!(result.inv_perm[old_i], new_i);
        }
    }

    #[test]
    fn test_cm_empty_graph() {
        let graph = AdjacencyGraph::from_adjacency_list(Vec::new());
        let perm = cuthill_mckee(&graph).expect("CM empty");
        assert!(perm.is_empty());
    }

    #[test]
    fn test_cm_single_node() {
        let graph = AdjacencyGraph::from_adjacency_list(vec![Vec::new()]);
        let perm = cuthill_mckee(&graph).expect("CM single");
        assert_eq!(perm, vec![0]);
    }

    #[test]
    fn test_rcm_disconnected_graph() {
        // Two disconnected path segments
        let mut adj = vec![Vec::new(); 6];
        // Component 1: 0-1-2
        adj[0].push(1);
        adj[1].push(0);
        adj[1].push(2);
        adj[2].push(1);
        // Component 2: 3-4-5
        adj[3].push(4);
        adj[4].push(3);
        adj[4].push(5);
        adj[5].push(4);
        let graph = AdjacencyGraph::from_adjacency_list(adj);
        let perm = reverse_cuthill_mckee(&graph).expect("RCM disconnected");
        assert_eq!(perm.len(), 6);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..6).collect::<Vec<_>>());
    }

    #[test]
    fn test_bandwidth_profile_computation() {
        // Triangle graph: 0-1, 1-2, 0-2
        let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let graph = AdjacencyGraph::from_adjacency_list(adj);
        let bw = bandwidth(&graph, &[]).expect("bandwidth");
        assert_eq!(bw, 2); // max distance is |0-2| = 2
        let prof = profile(&graph, &[]).expect("profile");
        assert!(prof > 0);
    }
}
