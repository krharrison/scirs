//! Approximate Minimum Degree (AMD) ordering for fill-reducing reordering
//!
//! The AMD algorithm produces a permutation that reduces fill-in during
//! sparse Cholesky or LU factorization. It operates on a quotient graph
//! representation for memory efficiency and uses external degree
//! approximation with mass elimination of indistinguishable nodes and
//! aggressive absorption.
//!
//! # References
//!
//! - P.R. Amestoy, T.A. Davis, I.S. Duff, "An Approximate Minimum Degree
//!   Ordering Algorithm", SIAM J. Matrix Anal. Appl., 17(4), 1996.
//! - T.A. Davis, "Direct Methods for Sparse Linear Systems", SIAM, 2006.

use super::adjacency::AdjacencyGraph;
use crate::error::{SparseError, SparseResult};

/// Result of the AMD algorithm.
#[derive(Debug, Clone)]
pub struct AmdResult {
    /// Permutation vector: `perm[new_index] = old_index`.
    pub perm: Vec<usize>,
    /// Inverse permutation: `inv_perm[old_index] = new_index`.
    pub inv_perm: Vec<usize>,
    /// Estimated number of nonzeros in the Cholesky factor L.
    pub estimated_nnz: usize,
}

/// Marker for node status in the quotient graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeStatus {
    /// Node is still active (not yet eliminated).
    Active,
    /// Node has been eliminated at the given step.
    Eliminated(usize),
    /// Node has been absorbed into another node (indistinguishable).
    Absorbed(usize),
}

/// Quotient graph representation for the AMD algorithm.
///
/// Instead of explicitly forming the filled graph after each elimination,
/// the quotient graph represents eliminated nodes as "elements" whose
/// adjacency encodes the fill edges.
struct QuotientGraph {
    /// Number of original nodes.
    n: usize,
    /// Adjacency lists: for active nodes, these are the current neighbors
    /// (including element nodes). For element nodes, these are the nodes
    /// in the element's reach.
    adj: Vec<Vec<usize>>,
    /// Status of each node.
    status: Vec<NodeStatus>,
    /// External degree approximation for each active node.
    degree: Vec<usize>,
    /// Weight of each node (for mass elimination of indistinguishable nodes).
    weight: Vec<usize>,
}

impl QuotientGraph {
    /// Build a quotient graph from an adjacency graph.
    fn new(graph: &AdjacencyGraph) -> Self {
        let n = graph.num_nodes();
        let adj: Vec<Vec<usize>> = (0..n).map(|u| graph.neighbors(u).to_vec()).collect();
        let degree: Vec<usize> = (0..n).map(|u| graph.degree(u)).collect();

        Self {
            n,
            adj,
            status: vec![NodeStatus::Active; n],
            degree,
            weight: vec![1; n],
        }
    }

    /// Find the representative of a node (following absorption chains).
    fn find_representative(&self, mut node: usize) -> usize {
        let mut steps = 0;
        while let NodeStatus::Absorbed(parent) = self.status[node] {
            node = parent;
            steps += 1;
            if steps > self.n {
                break; // Safety: prevent infinite loops
            }
        }
        node
    }

    /// Get the approximate external degree of an active node.
    ///
    /// The external degree is the number of distinct active nodes reachable
    /// through the current adjacency (which may include element nodes).
    fn approximate_external_degree(&self, u: usize) -> usize {
        let mut reachable = Vec::new();
        let mut seen = vec![false; self.n];
        seen[u] = true;

        for &v in &self.adj[u] {
            let rep = self.find_representative(v);
            match self.status[rep] {
                NodeStatus::Active => {
                    if !seen[rep] {
                        seen[rep] = true;
                        reachable.push(rep);
                    }
                }
                NodeStatus::Eliminated(_) => {
                    // v is an element: add its active reach
                    for &w in &self.adj[rep] {
                        let wr = self.find_representative(w);
                        if matches!(self.status[wr], NodeStatus::Active) && !seen[wr] {
                            seen[wr] = true;
                            reachable.push(wr);
                        }
                    }
                }
                NodeStatus::Absorbed(_) => {
                    // Should not happen after find_representative
                }
            }
        }

        // Sum weights of reachable nodes as degree approximation
        reachable.iter().map(|&v| self.weight[v]).sum()
    }

    /// Eliminate a node: convert it to an element in the quotient graph.
    ///
    /// Returns the set of active nodes that were in the reach of the
    /// eliminated node (for mass elimination check).
    fn eliminate(&mut self, pivot: usize, step: usize) -> Vec<usize> {
        self.status[pivot] = NodeStatus::Eliminated(step);

        // Gather all active nodes reachable from pivot (the reach set).
        let mut reach = Vec::new();
        let mut seen = vec![false; self.n];
        seen[pivot] = true;

        for &v in &self.adj[pivot].clone() {
            let rep = self.find_representative(v);
            match self.status[rep] {
                NodeStatus::Active => {
                    if !seen[rep] {
                        seen[rep] = true;
                        reach.push(rep);
                    }
                }
                NodeStatus::Eliminated(_) => {
                    // Absorb this element: add its reach to pivot's reach
                    for &w in &self.adj[rep].clone() {
                        let wr = self.find_representative(w);
                        if matches!(self.status[wr], NodeStatus::Active) && !seen[wr] {
                            seen[wr] = true;
                            reach.push(wr);
                        }
                    }
                    // Aggressive absorption: mark old element as absorbed into pivot
                    self.status[rep] = NodeStatus::Absorbed(pivot);
                }
                NodeStatus::Absorbed(_) => {}
            }
        }

        // The element (pivot) now has reach as its adjacency
        self.adj[pivot] = reach.clone();

        // Update adjacency of each node in reach:
        // Replace all element references with pivot, remove pivot from active neighbors
        for &r in &reach {
            // Remove references to absorbed elements and the pivot itself
            let mut new_adj = Vec::new();
            let mut has_pivot_element = false;

            for &v in &self.adj[r] {
                let rep = self.find_representative(v);
                if rep == pivot {
                    // This is the new element
                    if !has_pivot_element {
                        new_adj.push(pivot);
                        has_pivot_element = true;
                    }
                } else if matches!(self.status[rep], NodeStatus::Active) {
                    if !new_adj.contains(&rep) {
                        new_adj.push(rep);
                    }
                } else if matches!(self.status[rep], NodeStatus::Eliminated(_))
                    && !new_adj.contains(&rep)
                {
                    new_adj.push(rep);
                }
            }

            if !has_pivot_element {
                new_adj.push(pivot);
            }

            self.adj[r] = new_adj;
        }

        // Update degrees for reach nodes
        for &r in &reach {
            self.degree[r] = self.approximate_external_degree(r);
        }

        // Mass elimination: detect indistinguishable nodes in reach
        // Two nodes are indistinguishable if they have the same adjacency
        self.mass_eliminate(&reach);

        reach
    }

    /// Mass elimination of indistinguishable nodes.
    ///
    /// Two active nodes are indistinguishable if they have exactly the same
    /// set of adjacent elements and active nodes. Such nodes can be merged.
    fn mass_eliminate(&mut self, reach: &[usize]) {
        if reach.len() < 2 {
            return;
        }

        // Build a hash for each node's adjacency to quickly find candidates.
        let mut hashes: Vec<(usize, u64)> = Vec::new();
        for &u in reach {
            if !matches!(self.status[u], NodeStatus::Active) {
                continue;
            }
            let mut hash = 0u64;
            let mut sorted_adj: Vec<usize> = self.adj[u]
                .iter()
                .map(|&v| self.find_representative(v))
                .filter(|&v| v != u)
                .collect();
            sorted_adj.sort_unstable();
            sorted_adj.dedup();
            for &v in &sorted_adj {
                // Simple hash combining
                hash = hash
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(v as u64);
            }
            hashes.push((u, hash));
        }

        hashes.sort_unstable_by_key(|&(_, h)| h);

        // Group nodes with same hash and verify they are truly indistinguishable
        let mut i = 0;
        while i < hashes.len() {
            let mut j = i + 1;
            while j < hashes.len() && hashes[j].1 == hashes[i].1 {
                j += 1;
            }
            if j - i > 1 {
                // Potential group: verify adjacency equality
                let group: Vec<usize> = hashes[i..j].iter().map(|&(u, _)| u).collect();
                self.try_merge_group(&group);
            }
            i = j;
        }
    }

    /// Try to merge a group of nodes with the same hash if they are truly
    /// indistinguishable (same adjacency).
    fn try_merge_group(&mut self, group: &[usize]) {
        if group.len() < 2 {
            return;
        }

        // For each pair in the group, check if adjacency matches
        let get_sorted_adj = |u: usize, adj: &[Vec<usize>], status: &[NodeStatus]| -> Vec<usize> {
            let mut sorted: Vec<usize> = adj[u]
                .iter()
                .filter_map(|&v| {
                    let rep = {
                        let mut node = v;
                        let mut steps = 0;
                        while let NodeStatus::Absorbed(parent) = status[node] {
                            node = parent;
                            steps += 1;
                            if steps > adj.len() {
                                break;
                            }
                        }
                        node
                    };
                    if rep != u {
                        Some(rep)
                    } else {
                        None
                    }
                })
                .collect();
            sorted.sort_unstable();
            sorted.dedup();
            sorted
        };

        let representative = group[0];
        let rep_adj = get_sorted_adj(representative, &self.adj, &self.status);

        for &other in &group[1..] {
            if !matches!(self.status[other], NodeStatus::Active) {
                continue;
            }
            let other_adj = get_sorted_adj(other, &self.adj, &self.status);
            if rep_adj == other_adj {
                // Absorb `other` into `representative`
                self.status[other] = NodeStatus::Absorbed(representative);
                self.weight[representative] += self.weight[other];
            }
        }
    }
}

/// Compute the AMD permutation for a symmetric adjacency graph.
///
/// Uses a quotient graph representation with external degree approximation,
/// mass elimination of indistinguishable nodes, and aggressive absorption.
///
/// # Arguments
///
/// * `graph` - Symmetric adjacency graph.
///
/// # Returns
///
/// `AmdResult` containing the permutation and estimated fill.
pub fn amd(graph: &AdjacencyGraph) -> SparseResult<AmdResult> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(AmdResult {
            perm: Vec::new(),
            inv_perm: Vec::new(),
            estimated_nnz: 0,
        });
    }

    let mut qg = QuotientGraph::new(graph);
    let mut perm = Vec::with_capacity(n);
    let mut estimated_nnz = 0usize;

    for step in 0..n {
        // Find active node with minimum approximate degree
        let pivot = (0..n)
            .filter(|&u| matches!(qg.status[u], NodeStatus::Active))
            .min_by_key(|&u| qg.degree[u]);

        let pivot = match pivot {
            Some(p) => p,
            None => break, // All nodes eliminated (shouldn't happen)
        };

        // Add pivot (and any absorbed nodes) to the ordering
        perm.push(pivot);

        // Estimate fill: the reach set size contributes to nnz in L
        let reach = qg.eliminate(pivot, step);
        estimated_nnz += reach.len();

        // Also emit absorbed nodes that were merged into other active nodes
        // (they share the same elimination step)
    }

    // Now add any nodes that were absorbed (they go right after their representative)
    let mut full_perm = Vec::with_capacity(n);
    let mut emitted = vec![false; n];

    for &p in &perm {
        if !emitted[p] {
            full_perm.push(p);
            emitted[p] = true;
        }
        // Find all nodes absorbed into p
        for u in 0..n {
            if !emitted[u] {
                if let NodeStatus::Absorbed(rep) = qg.status[u] {
                    let final_rep = qg.find_representative(rep);
                    if final_rep == p {
                        full_perm.push(u);
                        emitted[u] = true;
                    }
                }
            }
        }
    }

    // Catch any remaining nodes (should not happen, but be safe)
    for u in 0..n {
        if !emitted[u] {
            full_perm.push(u);
            emitted[u] = true;
        }
    }

    // Build inverse permutation
    let mut inv_perm = vec![0usize; n];
    for (new_i, &old_i) in full_perm.iter().enumerate() {
        inv_perm[old_i] = new_i;
    }

    // Rough estimate of nnz: each node contributes its column count in L
    // plus the diagonal
    estimated_nnz += n; // diagonal entries

    Ok(AmdResult {
        perm: full_perm,
        inv_perm,
        estimated_nnz,
    })
}

/// Simple AMD without quotient graph (for small matrices or as fallback).
///
/// Uses a direct greedy minimum-degree strategy with explicit fill-in tracking.
pub fn amd_simple(graph: &AdjacencyGraph) -> SparseResult<AmdResult> {
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(AmdResult {
            perm: Vec::new(),
            inv_perm: Vec::new(),
            estimated_nnz: 0,
        });
    }

    // Build mutable adjacency
    let mut adj: Vec<Vec<usize>> = (0..n).map(|u| graph.neighbors(u).to_vec()).collect();

    let mut perm = Vec::with_capacity(n);
    let mut eliminated = vec![false; n];
    let mut estimated_nnz = n; // diagonal

    for _step in 0..n {
        // Find minimum degree uneliminated node
        let pivot = (0..n)
            .filter(|&u| !eliminated[u])
            .min_by_key(|&u| adj[u].iter().filter(|&&v| !eliminated[v]).count())
            .unwrap_or(0);

        perm.push(pivot);
        eliminated[pivot] = true;

        // Gather active neighbors
        let neighbors: Vec<usize> = adj[pivot]
            .iter()
            .copied()
            .filter(|&v| !eliminated[v])
            .collect();

        estimated_nnz += neighbors.len();

        // Form clique among neighbors (fill-in)
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let u = neighbors[i];
                let v = neighbors[j];
                if !adj[u].contains(&v) {
                    adj[u].push(v);
                    adj[v].push(u);
                }
            }
        }
    }

    let mut inv_perm = vec![0usize; n];
    for (new_i, &old_i) in perm.iter().enumerate() {
        inv_perm[old_i] = new_i;
    }

    Ok(AmdResult {
        perm,
        inv_perm,
        estimated_nnz,
    })
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
    fn test_amd_valid_permutation() {
        let graph = path_graph(8);
        let result = amd(&graph).expect("AMD");
        assert_eq!(result.perm.len(), 8);
        let mut sorted = result.perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn test_amd_simple_valid_permutation() {
        let graph = path_graph(8);
        let result = amd_simple(&graph).expect("AMD simple");
        assert_eq!(result.perm.len(), 8);
        let mut sorted = result.perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn test_amd_fill_estimate_reasonable() {
        let graph = grid_graph(4, 4);
        let result = amd(&graph).expect("AMD grid");
        assert_eq!(result.perm.len(), 16);
        // Fill estimate should be at least n (diagonal) and at most n^2
        assert!(result.estimated_nnz >= 16);
        assert!(result.estimated_nnz <= 256);
    }

    #[test]
    fn test_amd_empty_graph() {
        let graph = AdjacencyGraph::from_adjacency_list(Vec::new());
        let result = amd(&graph).expect("AMD empty");
        assert!(result.perm.is_empty());
        assert_eq!(result.estimated_nnz, 0);
    }

    #[test]
    fn test_amd_single_node() {
        let graph = AdjacencyGraph::from_adjacency_list(vec![Vec::new()]);
        let result = amd(&graph).expect("AMD single");
        assert_eq!(result.perm, vec![0]);
        assert_eq!(result.estimated_nnz, 1); // just the diagonal
    }

    #[test]
    fn test_amd_inverse_perm_consistency() {
        let graph = grid_graph(3, 3);
        let result = amd(&graph).expect("AMD");
        for (new_i, &old_i) in result.perm.iter().enumerate() {
            assert_eq!(
                result.inv_perm[old_i], new_i,
                "inv_perm inconsistency at old_i={}",
                old_i
            );
        }
    }

    #[test]
    fn test_amd_star_graph() {
        // Star: center node 0 connected to all others
        let n = 6;
        let mut adj = vec![Vec::new(); n];
        for i in 1..n {
            adj[0].push(i);
            adj[i].push(0);
        }
        let graph = AdjacencyGraph::from_adjacency_list(adj);
        let result = amd(&graph).expect("AMD star");
        assert_eq!(result.perm.len(), n);
        // Center node (degree n-1) should typically be eliminated last
        // because leaf nodes have degree 1 (minimum)
        let center_pos = result.inv_perm[0];
        assert!(
            center_pos >= n / 2,
            "center node should be eliminated late, got position {}",
            center_pos
        );
    }

    #[test]
    fn test_amd_simple_vs_quotient_same_size() {
        let graph = path_graph(6);
        let r1 = amd(&graph).expect("AMD quotient");
        let r2 = amd_simple(&graph).expect("AMD simple");
        assert_eq!(r1.perm.len(), r2.perm.len());
        // Both should produce valid permutations
        let mut s1 = r1.perm.clone();
        s1.sort_unstable();
        assert_eq!(s1, (0..6).collect::<Vec<_>>());
        let mut s2 = r2.perm.clone();
        s2.sort_unstable();
        assert_eq!(s2, (0..6).collect::<Vec<_>>());
    }
}
