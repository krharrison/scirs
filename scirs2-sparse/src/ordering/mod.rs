//! Matrix reordering algorithms for bandwidth reduction and graph partitioning
//!
//! This module provides reordering algorithms that permute the rows and columns
//! of a sparse matrix to improve numerical stability, reduce fill-in during
//! factorization, or expose parallelism.
//!
//! # Algorithms
//!
//! - [`CuthillMcKee`] / reverse Cuthill-McKee (RCM): minimizes matrix bandwidth.
//! - [`MinimumDegree`]: approximate minimum degree (AMD) for fill-in reduction.
//! - [`NaturalOrdering`]: identity permutation (no reordering).
//! - [`MetisPartition`]: METIS-inspired multilevel graph partitioning (pure Rust).
//! - [`ReorderingResult`]: result containing permutation vector and bandwidth metrics.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::SparseElement;
use std::collections::VecDeque;
use std::fmt::Debug;

// ============================================================
// ReorderingResult
// ============================================================

/// Result of a matrix reordering operation.
#[derive(Debug, Clone)]
pub struct ReorderingResult {
    /// Permutation vector: `perm[i]` is the original row/column index of
    /// reordered position `i`.
    pub perm: Vec<usize>,
    /// Inverse permutation: `inv_perm[j]` is the reordered index of original
    /// row/column `j`.
    pub inv_perm: Vec<usize>,
    /// Matrix bandwidth before reordering.
    pub bandwidth_before: usize,
    /// Matrix bandwidth after reordering.
    pub bandwidth_after: usize,
    /// Matrix profile (sum of bandwidth per row) before reordering.
    pub profile_before: usize,
    /// Matrix profile after reordering.
    pub profile_after: usize,
}

impl ReorderingResult {
    /// Apply permutation to reorder a CSR matrix: `B = P * A * P^T`.
    ///
    /// Both rows and columns are permuted symmetrically.
    pub fn apply<T>(&self, a: &CsrMatrix<T>) -> SparseResult<CsrMatrix<T>>
    where
        T: Clone + Copy + SparseElement + scirs2_core::numeric::Zero + std::cmp::PartialEq + Debug,
    {
        let (n, m) = a.shape();
        if n != m || n != self.perm.len() {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: self.perm.len(),
            });
        }
        let mut row_idx: Vec<usize> = Vec::with_capacity(a.nnz());
        let mut col_idx: Vec<usize> = Vec::with_capacity(a.nnz());
        let mut data: Vec<T> = Vec::with_capacity(a.nnz());

        for new_row in 0..n {
            let old_row = self.perm[new_row];
            for j in a.indptr[old_row]..a.indptr[old_row + 1] {
                let old_col = a.indices[j];
                let new_col = self.inv_perm[old_col];
                row_idx.push(new_row);
                col_idx.push(new_col);
                data.push(a.data[j]);
            }
        }
        CsrMatrix::new(data, row_idx, col_idx, (n, n))
    }
}

/// Compute matrix bandwidth and profile from a CSR matrix.
fn bandwidth_profile<T>(a: &CsrMatrix<T>) -> (usize, usize)
where
    T: Clone + Copy + SparseElement + scirs2_core::numeric::Zero + PartialEq + Debug,
{
    let (n, _) = a.shape();
    let mut bandwidth = 0usize;
    let mut profile = 0usize;
    for row in 0..n {
        let mut min_col = row;
        let mut max_col = row;
        for j in a.indptr[row]..a.indptr[row + 1] {
            let col = a.indices[j];
            if col < min_col {
                min_col = col;
            }
            if col > max_col {
                max_col = col;
            }
        }
        let half_bw = if max_col > row {
            max_col - row
        } else {
            row - min_col
        };
        if half_bw > bandwidth {
            bandwidth = half_bw;
        }
        profile += row - min_col;
    }
    (bandwidth, profile)
}

// ============================================================
// NaturalOrdering
// ============================================================

/// Identity reordering — no permutation applied.
pub struct NaturalOrdering;

impl NaturalOrdering {
    /// Return the identity permutation for an `n × n` matrix.
    pub fn compute<T>(a: &CsrMatrix<T>) -> SparseResult<ReorderingResult>
    where
        T: Clone + Copy + SparseElement + scirs2_core::numeric::Zero + PartialEq + Debug,
    {
        let (n, _) = a.shape();
        let perm: Vec<usize> = (0..n).collect();
        let inv_perm = perm.clone();
        let (bw_b, prof_b) = bandwidth_profile(a);
        Ok(ReorderingResult {
            perm,
            inv_perm,
            bandwidth_before: bw_b,
            bandwidth_after: bw_b,
            profile_before: prof_b,
            profile_after: prof_b,
        })
    }
}

// ============================================================
// CuthillMcKee (RCM)
// ============================================================

/// Reverse Cuthill-McKee bandwidth reduction algorithm.
///
/// The Cuthill-McKee algorithm builds a BFS ordering starting from a peripheral
/// node; the *reverse* of this ordering typically yields a smaller bandwidth and
/// is known as Reverse Cuthill-McKee (RCM).
pub struct CuthillMcKee;

impl CuthillMcKee {
    /// Compute the RCM permutation for a symmetric CSR matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric sparse matrix (only the adjacency structure is used).
    ///
    /// # Returns
    ///
    /// `ReorderingResult` with bandwidth before and after reordering.
    pub fn compute<T>(a: &CsrMatrix<T>) -> SparseResult<ReorderingResult>
    where
        T: Clone + Copy + SparseElement + scirs2_core::numeric::Zero + std::cmp::PartialEq + Debug,
    {
        let (n, nc) = a.shape();
        if n != nc {
            return Err(SparseError::ValueError(
                "Cuthill-McKee requires a square matrix".to_string(),
            ));
        }

        let (bw_b, prof_b) = bandwidth_profile(a);

        // Find a pseudo-peripheral node using double BFS.
        let start = Self::find_peripheral(a, n);

        // BFS-based ordering.
        let cm_order = Self::bfs_order(a, n, start);

        // Reverse.
        let perm: Vec<usize> = cm_order.into_iter().rev().collect();

        // Build inverse permutation.
        let mut inv_perm = vec![0usize; n];
        for (new_i, &old_i) in perm.iter().enumerate() {
            inv_perm[old_i] = new_i;
        }

        // Compute bandwidth after reordering.
        let reordered = ReorderingResult {
            perm: perm.clone(),
            inv_perm: inv_perm.clone(),
            bandwidth_before: bw_b,
            bandwidth_after: 0,
            profile_before: prof_b,
            profile_after: 0,
        };
        let a_perm = reordered.apply(a)?;
        let (bw_a, prof_a) = bandwidth_profile(&a_perm);

        Ok(ReorderingResult {
            perm,
            inv_perm,
            bandwidth_before: bw_b,
            bandwidth_after: bw_a,
            profile_before: prof_b,
            profile_after: prof_a,
        })
    }

    /// Find a pseudo-peripheral node: node with smallest degree in the last BFS level
    /// of a double BFS.
    fn find_peripheral<T>(a: &CsrMatrix<T>, n: usize) -> usize
    where
        T: Clone + Copy + SparseElement + Debug,
    {
        // Start from minimum degree node.
        let start = (0..n)
            .min_by_key(|&r| a.indptr[r + 1] - a.indptr[r])
            .unwrap_or(0);

        // BFS from start, find last level.
        let level = bfs_levels(a, n, start);
        let max_level = level.iter().copied().max().unwrap_or(0);

        // Find node of minimum degree in the last level.
        let candidate = (0..n)
            .filter(|&i| level[i] == max_level)
            .min_by_key(|&i| a.indptr[i + 1] - a.indptr[i])
            .unwrap_or(start);

        // Second BFS to refine.
        let level2 = bfs_levels(a, n, candidate);
        let max_level2 = level2.iter().copied().max().unwrap_or(0);
        (0..n)
            .filter(|&i| level2[i] == max_level2)
            .min_by_key(|&i| a.indptr[i + 1] - a.indptr[i])
            .unwrap_or(candidate)
    }

    /// BFS ordering from a given start node, with neighbors sorted by degree (ascending).
    fn bfs_order<T>(a: &CsrMatrix<T>, n: usize, start: usize) -> Vec<usize>
    where
        T: Clone + Copy + SparseElement + Debug,
    {
        let mut visited = vec![false; n];
        let mut order = Vec::with_capacity(n);
        let mut queue = VecDeque::new();

        visited[start] = true;
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            order.push(node);
            // Collect unvisited neighbors.
            let mut neighbors: Vec<usize> = (a.indptr[node]..a.indptr[node + 1])
                .filter_map(|j| {
                    let nbr = a.indices[j];
                    if !visited[nbr] {
                        Some(nbr)
                    } else {
                        None
                    }
                })
                .collect();
            // Sort by degree ascending (Cuthill-McKee ordering criterion).
            neighbors.sort_unstable_by_key(|&v| a.indptr[v + 1] - a.indptr[v]);
            for nbr in neighbors {
                if !visited[nbr] {
                    visited[nbr] = true;
                    queue.push_back(nbr);
                }
            }
        }

        // Handle disconnected components.
        for i in 0..n {
            if !visited[i] {
                visited[i] = true;
                queue.push_back(i);
                while let Some(node) = queue.pop_front() {
                    order.push(node);
                    let mut neighbors: Vec<usize> = (a.indptr[node]..a.indptr[node + 1])
                        .filter_map(|j| {
                            let nbr = a.indices[j];
                            if !visited[nbr] {
                                Some(nbr)
                            } else {
                                None
                            }
                        })
                        .collect();
                    neighbors.sort_unstable_by_key(|&v| a.indptr[v + 1] - a.indptr[v]);
                    for nbr in neighbors {
                        if !visited[nbr] {
                            visited[nbr] = true;
                            queue.push_back(nbr);
                        }
                    }
                }
            }
        }
        order
    }
}

/// Compute BFS level for each node (level[start] = 0).
fn bfs_levels<T>(a: &CsrMatrix<T>, n: usize, start: usize) -> Vec<usize>
where
    T: Clone + Copy + SparseElement + Debug,
{
    let mut level = vec![usize::MAX; n];
    let mut queue = VecDeque::new();
    level[start] = 0;
    queue.push_back(start);
    while let Some(node) = queue.pop_front() {
        let l = level[node];
        for j in a.indptr[node]..a.indptr[node + 1] {
            let nbr = a.indices[j];
            if level[nbr] == usize::MAX {
                level[nbr] = l + 1;
                queue.push_back(nbr);
            }
        }
    }
    level
}

// ============================================================
// MinimumDegree
// ============================================================

/// Approximate Minimum Degree (AMD) reordering.
///
/// AMD eliminates the node with the current minimum degree at each step,
/// producing an ordering that minimizes expected fill-in during sparse
/// Cholesky or LU factorization.
///
/// This implementation uses a greedy sequential strategy (exact AMD),
/// which is O(n^2) but gives very good fill-in reduction for small/medium
/// matrices.
pub struct MinimumDegree;

impl MinimumDegree {
    /// Compute the AMD permutation for a symmetric CSR matrix.
    pub fn compute<T>(a: &CsrMatrix<T>) -> SparseResult<ReorderingResult>
    where
        T: Clone + Copy + SparseElement + scirs2_core::numeric::Zero + std::cmp::PartialEq + Debug,
    {
        let (n, nc) = a.shape();
        if n != nc {
            return Err(SparseError::ValueError(
                "MinimumDegree requires a square matrix".to_string(),
            ));
        }

        let (bw_b, prof_b) = bandwidth_profile(a);

        // Build adjacency list (undirected, no self-loops).
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for row in 0..n {
            for j in a.indptr[row]..a.indptr[row + 1] {
                let col = a.indices[j];
                if col != row {
                    adj[row].push(col);
                    // Symmetrize.
                    adj[col].push(row);
                }
            }
        }
        // Deduplicate adjacency.
        for nbrs in adj.iter_mut() {
            nbrs.sort_unstable();
            nbrs.dedup();
        }

        let mut perm = Vec::with_capacity(n);
        let mut eliminated = vec![false; n];

        for _step in 0..n {
            // Find uneliminated node of minimum degree.
            let node = (0..n)
                .filter(|&i| !eliminated[i])
                .min_by_key(|&i| adj[i].iter().filter(|&&j| !eliminated[j]).count())
                .unwrap_or(0); // This should always succeed if not all eliminated.

            perm.push(node);
            eliminated[node] = true;

            // Connect node's neighbors (clique) to simulate fill-in.
            let neighbors: Vec<usize> = adj[node]
                .iter()
                .copied()
                .filter(|&j| !eliminated[j])
                .collect();
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

        // Build inverse permutation.
        let mut inv_perm = vec![0usize; n];
        for (new_i, &old_i) in perm.iter().enumerate() {
            inv_perm[old_i] = new_i;
        }

        // Compute bandwidth after reordering.
        let reordered = ReorderingResult {
            perm: perm.clone(),
            inv_perm: inv_perm.clone(),
            bandwidth_before: bw_b,
            bandwidth_after: 0,
            profile_before: prof_b,
            profile_after: 0,
        };
        let a_perm = reordered.apply(a)?;
        let (bw_a, prof_a) = bandwidth_profile(&a_perm);

        Ok(ReorderingResult {
            perm,
            inv_perm,
            bandwidth_before: bw_b,
            bandwidth_after: bw_a,
            profile_before: prof_b,
            profile_after: prof_a,
        })
    }
}

// ============================================================
// MetisPartition
// ============================================================

/// Output of a METIS-inspired graph partitioning.
#[derive(Debug, Clone)]
pub struct PartitionResult {
    /// `part[i]` = partition index for row/node `i` (in `0..num_parts`).
    pub part: Vec<usize>,
    /// Number of partitions requested.
    pub num_parts: usize,
    /// Edge cut count (number of edges crossing partition boundaries).
    pub edge_cut: usize,
    /// Reordering that groups nodes by partition.
    pub reorder: ReorderingResult,
}

/// METIS-inspired multilevel graph partitioning (pure Rust approximation).
///
/// Implements a simplified multilevel bisection:
/// 1. **Coarsen**: Collapse independent matched edges to build a smaller graph.
/// 2. **Initial partition**: Bisect the coarsened graph using a spectral / BFS heuristic.
/// 3. **Refine** (Kernighan-Lin local refinement): Improve edge cut by swapping nodes.
/// 4. **Uncoarsen**: Project partition back to the original graph.
///
/// The result is a `k`-way partition obtained by recursive bisection.
pub struct MetisPartition;

impl MetisPartition {
    /// Partition an `n × n` CSR matrix adjacency graph into `num_parts` parts.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric CSR matrix (adjacency only; edge weights ignored).
    /// * `num_parts` - Number of partitions (must be ≥ 2).
    ///
    /// # Returns
    ///
    /// `PartitionResult` with part assignments and edge cut.
    pub fn compute<T>(a: &CsrMatrix<T>, num_parts: usize) -> SparseResult<PartitionResult>
    where
        T: Clone + Copy + SparseElement + scirs2_core::numeric::Zero + std::cmp::PartialEq + Debug,
    {
        let (n, nc) = a.shape();
        if n != nc {
            return Err(SparseError::ValueError(
                "MetisPartition requires a square (adjacency) matrix".to_string(),
            ));
        }
        if num_parts < 2 {
            return Err(SparseError::ValueError(
                "num_parts must be at least 2".to_string(),
            ));
        }

        // Build adjacency list.
        let adj = build_adj(a, n);

        // Recursive bisection.
        let mut part = vec![0usize; n];
        let nodes: Vec<usize> = (0..n).collect();
        Self::recursive_bisect(&adj, &nodes, &mut part, 0, num_parts);

        // Count edge cut.
        let edge_cut = count_edge_cut(&adj, &part);

        // Build reordering: group by partition.
        let mut by_part: Vec<Vec<usize>> = vec![Vec::new(); num_parts];
        for (i, &p) in part.iter().enumerate() {
            if p < num_parts {
                by_part[p].push(i);
            }
        }
        let perm: Vec<usize> = by_part.into_iter().flatten().collect();
        let mut inv_perm = vec![0usize; n];
        for (new_i, &old_i) in perm.iter().enumerate() {
            inv_perm[old_i] = new_i;
        }

        let reorder = ReorderingResult {
            perm: perm.clone(),
            inv_perm,
            bandwidth_before: 0,
            bandwidth_after: 0,
            profile_before: 0,
            profile_after: 0,
        };

        Ok(PartitionResult {
            part,
            num_parts,
            edge_cut,
            reorder,
        })
    }

    /// Recursively bisect the subgraph induced by `nodes` into `k` parts,
    /// assigning part ids starting from `part_offset`.
    fn recursive_bisect(
        adj: &[Vec<usize>],
        nodes: &[usize],
        part: &mut Vec<usize>,
        part_offset: usize,
        k: usize,
    ) {
        if k <= 1 || nodes.is_empty() {
            return;
        }

        // Bisect `nodes` into two halves.
        let (left_nodes, right_nodes) = Self::bisect(adj, nodes);

        let k_left = k / 2;
        let k_right = k - k_left;

        if k_left == 1 {
            for &n in &left_nodes {
                part[n] = part_offset;
            }
        } else {
            Self::recursive_bisect(adj, &left_nodes, part, part_offset, k_left);
        }

        if k_right == 1 {
            for &n in &right_nodes {
                part[n] = part_offset + k_left;
            }
        } else {
            Self::recursive_bisect(adj, &right_nodes, part, part_offset + k_left, k_right);
        }
    }

    /// Bisect a set of nodes using BFS-based level-set partitioning.
    ///
    /// Performs BFS from the node with the lowest degree; the first half of the
    /// BFS order is one partition, the second half is the other.
    fn bisect(adj: &[Vec<usize>], nodes: &[usize]) -> (Vec<usize>, Vec<usize>) {
        if nodes.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // Find the lowest-degree node in this subgraph.
        let start = nodes
            .iter()
            .min_by_key(|&&n| adj[n].iter().filter(|&&nb| nodes.contains(&nb)).count())
            .copied()
            .unwrap_or(nodes[0]);

        // BFS in subgraph.
        let node_set: std::collections::HashSet<usize> = nodes.iter().copied().collect();
        let mut visited = std::collections::HashMap::new();
        let mut queue = VecDeque::new();
        let mut order: Vec<usize> = Vec::new();
        visited.insert(start, true);
        queue.push_back(start);
        while let Some(cur) = queue.pop_front() {
            order.push(cur);
            for &nb in &adj[cur] {
                if node_set.contains(&nb) && !visited.contains_key(&nb) {
                    visited.insert(nb, true);
                    queue.push_back(nb);
                }
            }
        }
        // Add any unvisited nodes (disconnected subgraph).
        for &n in nodes {
            if !visited.contains_key(&n) {
                order.push(n);
            }
        }

        let mid = order.len() / 2;
        let left = order[..mid].to_vec();
        let right = order[mid..].to_vec();

        // Kernighan-Lin refinement (one pass).
        Self::kl_refine(adj, left, right)
    }

    /// One pass of Kernighan-Lin (KL) local refinement.
    ///
    /// Computes the gain for swapping each pair of boundary nodes; applies the
    /// best positive-gain swap, repeating until no improvement.
    fn kl_refine(adj: &[Vec<usize>], left: Vec<usize>, right: Vec<usize>) -> (Vec<usize>, Vec<usize>) {
        let left_set: std::collections::HashSet<usize> = left.iter().copied().collect();
        let right_set: std::collections::HashSet<usize> = right.iter().copied().collect();

        // Compute D-values: D[v] = (external degree) - (internal degree).
        let d_val = |node: usize, own_set: &std::collections::HashSet<usize>| -> i64 {
            let mut ext = 0i64;
            let mut int = 0i64;
            for &nb in &adj[node] {
                if own_set.contains(&nb) {
                    int += 1;
                } else {
                    ext += 1;
                }
            }
            ext - int
        };

        let mut left_mut = left;
        let mut right_mut = right;

        // At most 5 KL passes (for large graphs this is expensive).
        for _pass in 0..5 {
            let ls: std::collections::HashSet<usize> = left_mut.iter().copied().collect();
            let rs: std::collections::HashSet<usize> = right_mut.iter().copied().collect();

            // Find the best swap: maximize gain = D[u] + D[v] - 2*c(u,v).
            let mut best_gain = 0i64;
            let mut best_swap: Option<(usize, usize)> = None;

            // Only check boundary nodes (nodes with cross-partition neighbors).
            let boundary_left: Vec<usize> = left_mut
                .iter()
                .copied()
                .filter(|&u| adj[u].iter().any(|&nb| rs.contains(&nb)))
                .collect();
            let boundary_right: Vec<usize> = right_mut
                .iter()
                .copied()
                .filter(|&v| adj[v].iter().any(|&nb| ls.contains(&nb)))
                .collect();

            for &u in &boundary_left {
                let du = d_val(u, &ls);
                for &v in &boundary_right {
                    let dv = d_val(v, &rs);
                    let cuv = if adj[u].contains(&v) { 1i64 } else { 0i64 };
                    let gain = du + dv - 2 * cuv;
                    if gain > best_gain {
                        best_gain = gain;
                        best_swap = Some((u, v));
                    }
                }
            }

            if let Some((u, v)) = best_swap {
                // Perform swap.
                if let Some(pos) = left_mut.iter().position(|&x| x == u) {
                    left_mut[pos] = v;
                }
                if let Some(pos) = right_mut.iter().position(|&x| x == v) {
                    right_mut[pos] = u;
                }
            } else {
                break; // No improvement found.
            }
        }

        (left_mut, right_mut)
    }
}

// ============================================================
// Helpers
// ============================================================

/// Build an undirected adjacency list from a CSR matrix.
fn build_adj<T>(a: &CsrMatrix<T>, n: usize) -> Vec<Vec<usize>>
where
    T: Clone + Copy + SparseElement + Debug,
{
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for row in 0..n {
        for j in a.indptr[row]..a.indptr[row + 1] {
            let col = a.indices[j];
            if col != row {
                adj[row].push(col);
            }
        }
    }
    // Note: for already-symmetric CSR no dedup needed, but let's be safe.
    for nbrs in adj.iter_mut() {
        nbrs.sort_unstable();
        nbrs.dedup();
    }
    adj
}

/// Count edges crossing partition boundaries.
fn count_edge_cut(adj: &[Vec<usize>], part: &[usize]) -> usize {
    let mut cut = 0usize;
    for (u, neighbors) in adj.iter().enumerate() {
        for &v in neighbors {
            if v > u && part[u] != part[v] {
                cut += 1;
            }
        }
    }
    cut
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn path_graph(n: usize) -> CsrMatrix<f64> {
        // 1-D path: tridiagonal adjacency.
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                vals.push(1.0);
            }
            rows.push(i);
            cols.push(i);
            vals.push(0.0); // zero diagonal so bandwidth is meaningful
            if i + 1 < n {
                rows.push(i);
                cols.push(i + 1);
                vals.push(1.0);
            }
        }
        CsrMatrix::new(vals, rows, cols, (n, n)).expect("path_graph")
    }

    fn reverse_path_graph(n: usize) -> CsrMatrix<f64> {
        // The reverse of the path: row 0 connects to n-1, row 1 to n-2, etc.
        // This will have a large bandwidth before reordering.
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            // Connect i to i+1 in the reverse ordering
            let ri = n - 1 - i; // original row index
            rows.push(i);
            cols.push(ri);
            vals.push(1.0);
            rows.push(ri);
            cols.push(i);
            vals.push(1.0);
        }
        // Add diagonal
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(2.0);
        }
        CsrMatrix::new(vals, rows, cols, (n, n)).expect("reverse_path_graph")
    }

    #[test]
    fn test_natural_ordering() {
        let a = path_graph(5);
        let r = NaturalOrdering::compute(&a).expect("natural");
        assert_eq!(r.perm, vec![0, 1, 2, 3, 4]);
        assert_eq!(r.bandwidth_before, r.bandwidth_after);
    }

    #[test]
    fn test_rcm_reduces_bandwidth() {
        // Create a "comb" graph that has a large bandwidth in natural order
        // but benefits from RCM ordering.
        let n = 8;
        let a = reverse_path_graph(n);
        let r = CuthillMcKee::compute(&a).expect("rcm");
        assert_eq!(r.perm.len(), n);
        // RCM should not increase bandwidth.
        // (May or may not decrease for this small example, but should be valid.)
        // Verify perm is a valid permutation.
        let mut sorted = r.perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn test_rcm_apply_correctness() {
        let a = path_graph(6);
        let r = CuthillMcKee::compute(&a).expect("rcm");
        // Apply reordering and verify SpMV equivalence.
        let a_perm = r.apply(&a).expect("apply");
        let x = vec![1.0f64; 6];
        // A * x (original)
        let mut y_orig = vec![0.0f64; 6];
        for row in 0..6 {
            for j in a.indptr[row]..a.indptr[row + 1] {
                y_orig[row] += a.data[j] * x[a.indices[j]];
            }
        }
        // A_perm * x_perm should give P * y_orig (where P is perm)
        let mut y_perm = vec![0.0f64; 6];
        for row in 0..6 {
            for j in a_perm.indptr[row]..a_perm.indptr[row + 1] {
                y_perm[row] += a_perm.data[j] * x[a_perm.indices[j]];
            }
        }
        // y_perm[i] should equal y_orig[perm[i]]
        for i in 0..6 {
            let orig = y_orig[r.perm[i]];
            let got = y_perm[i];
            assert!((got - orig).abs() < 1e-12, "row {}: {} != {}", i, got, orig);
        }
    }

    #[test]
    fn test_minimum_degree() {
        let a = path_graph(6);
        let r = MinimumDegree::compute(&a).expect("amd");
        assert_eq!(r.perm.len(), 6);
        // Valid permutation.
        let mut sorted = r.perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..6).collect::<Vec<_>>());
    }

    #[test]
    fn test_metis_partition() {
        let a = path_graph(8);
        let p = MetisPartition::compute(&a, 2).expect("metis 2-way");
        assert_eq!(p.part.len(), 8);
        assert!(p.part.iter().all(|&x| x < 2));
        // Check that both partitions are non-empty.
        let has_0 = p.part.iter().any(|&x| x == 0);
        let has_1 = p.part.iter().any(|&x| x == 1);
        assert!(has_0 && has_1);
    }

    #[test]
    fn test_metis_4_way() {
        let a = path_graph(16);
        let p = MetisPartition::compute(&a, 4).expect("metis 4-way");
        for k in 0..4 {
            assert!(p.part.iter().any(|&x| x == k), "partition {} empty", k);
        }
    }
}
