//! Adjacency graph representation for reordering algorithms
//!
//! Provides a lightweight undirected graph abstraction that can be constructed
//! from CSR/CSC sparse matrices or raw adjacency lists. All reordering
//! algorithms in this module operate on `AdjacencyGraph`.

use crate::csc_array::CscArray;
use crate::csr::CsrMatrix;
use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::numeric::{SparseElement, Zero};
use std::fmt::Debug;
use std::ops::Div;

/// An undirected adjacency graph without self-loops.
///
/// Nodes are numbered `0..n`. Each node stores a sorted, deduplicated
/// list of its neighbors.
#[derive(Debug, Clone)]
pub struct AdjacencyGraph {
    /// `adj[u]` contains the sorted neighbor list of node `u`.
    adj: Vec<Vec<usize>>,
}

impl AdjacencyGraph {
    /// Create an `AdjacencyGraph` from a pre-built adjacency list.
    ///
    /// Self-loops are removed. Neighbor lists are sorted and deduplicated.
    pub fn from_adjacency_list(mut adj: Vec<Vec<usize>>) -> Self {
        let n = adj.len();
        for (u, nbrs) in adj.iter_mut().enumerate() {
            nbrs.retain(|&v| v != u && v < n);
            nbrs.sort_unstable();
            nbrs.dedup();
        }
        Self { adj }
    }

    /// Build an adjacency graph from a CSR matrix.
    ///
    /// The matrix is treated as an adjacency matrix. Both upper and lower
    /// triangles are considered (symmetrized). Self-loops are removed.
    pub fn from_csr_matrix<T>(mat: &CsrMatrix<T>) -> SparseResult<Self>
    where
        T: Clone + Copy + SparseElement + Zero + PartialEq + Debug,
    {
        let (n, nc) = mat.shape();
        if n != nc {
            return Err(SparseError::ValueError(
                "adjacency graph requires a square matrix".to_string(),
            ));
        }

        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for row in 0..n {
            for j in mat.indptr[row]..mat.indptr[row + 1] {
                let col = mat.indices[j];
                if col != row {
                    adj[row].push(col);
                    adj[col].push(row);
                }
            }
        }
        for nbrs in adj.iter_mut() {
            nbrs.sort_unstable();
            nbrs.dedup();
        }
        Ok(Self { adj })
    }

    /// Build an adjacency graph from a `CsrArray`.
    pub fn from_csr_array<T>(arr: &CsrArray<T>) -> SparseResult<Self>
    where
        T: SparseElement + Div<Output = T> + Zero + PartialOrd + 'static,
    {
        let (n, nc) = arr.shape();
        if n != nc {
            return Err(SparseError::ValueError(
                "adjacency graph requires a square matrix".to_string(),
            ));
        }

        // Use the dense representation to extract adjacency
        let dense = <CsrArray<T> as SparseArray<T>>::to_array(arr);
        let zero = <T as Zero>::zero();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for row in 0..n {
            for col in 0..n {
                if row != col && dense[[row, col]] != zero {
                    adj[row].push(col);
                }
            }
        }
        for nbrs in adj.iter_mut() {
            nbrs.sort_unstable();
            nbrs.dedup();
        }
        Ok(Self { adj })
    }

    /// Build an adjacency graph from a `CscArray`.
    pub fn from_csc_array<T>(arr: &CscArray<T>) -> SparseResult<Self>
    where
        T: SparseElement
            + Div<Output = T>
            + Zero
            + PartialOrd
            + scirs2_core::numeric::Float
            + 'static,
    {
        let (n, nc) = <CscArray<T> as SparseArray<T>>::shape(arr);
        if n != nc {
            return Err(SparseError::ValueError(
                "adjacency graph requires a square matrix".to_string(),
            ));
        }

        let dense = <CscArray<T> as SparseArray<T>>::to_array(arr);
        let zero = <T as Zero>::zero();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for row in 0..n {
            for col in 0..n {
                if row != col && dense[[row, col]] != zero {
                    adj[row].push(col);
                }
            }
        }
        for nbrs in adj.iter_mut() {
            nbrs.sort_unstable();
            nbrs.dedup();
        }
        Ok(Self { adj })
    }

    /// Number of nodes in the graph.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.adj.len()
    }

    /// Degree of node `u` (number of neighbors, excluding self-loops).
    #[inline]
    pub fn degree(&self, u: usize) -> usize {
        self.adj.get(u).map_or(0, |v| v.len())
    }

    /// Sorted neighbor list of node `u`.
    #[inline]
    pub fn neighbors(&self, u: usize) -> &[usize] {
        self.adj.get(u).map_or(&[], |v| v.as_slice())
    }

    /// Total number of edges (each undirected edge counted once).
    pub fn num_edges(&self) -> usize {
        let total: usize = self.adj.iter().map(|v| v.len()).sum();
        total / 2
    }

    /// Check whether `u` and `v` are adjacent.
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        if u >= self.adj.len() || v >= self.adj.len() {
            return false;
        }
        self.adj[u].binary_search(&v).is_ok()
    }

    /// Return a subgraph induced by the given set of nodes.
    ///
    /// The returned graph has nodes numbered `0..nodes.len()`,
    /// with `mapping[i]` being the original node index.
    pub fn subgraph(&self, nodes: &[usize]) -> (AdjacencyGraph, Vec<usize>) {
        let n = nodes.len();
        // Build reverse map: original -> new index
        let mut rev = vec![usize::MAX; self.adj.len()];
        for (new_i, &old_i) in nodes.iter().enumerate() {
            if old_i < rev.len() {
                rev[old_i] = new_i;
            }
        }

        let mut adj = vec![Vec::new(); n];
        for (new_u, &old_u) in nodes.iter().enumerate() {
            if old_u >= self.adj.len() {
                continue;
            }
            for &old_v in &self.adj[old_u] {
                if old_v < rev.len() && rev[old_v] != usize::MAX {
                    adj[new_u].push(rev[old_v]);
                }
            }
            adj[new_u].sort_unstable();
            adj[new_u].dedup();
        }

        (AdjacencyGraph { adj }, nodes.to_vec())
    }

    /// Raw adjacency list (for internal use by other reorder algorithms).
    pub(crate) fn raw_adj(&self) -> &[Vec<usize>] {
        &self.adj
    }
}

/// Apply a symmetric permutation to a CSR matrix: `B = P * A * P^T`.
///
/// Given a permutation vector `perm` where `perm[new_i] = old_i`,
/// constructs the reordered matrix.
pub fn apply_permutation<T>(mat: &CsrMatrix<T>, perm: &[usize]) -> SparseResult<CsrMatrix<T>>
where
    T: Clone + Copy + SparseElement + Zero + PartialEq + Debug,
{
    let (n, nc) = mat.shape();
    if n != nc {
        return Err(SparseError::ValueError(
            "apply_permutation requires a square matrix".to_string(),
        ));
    }
    if perm.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: perm.len(),
        });
    }

    // Build inverse permutation
    let mut inv_perm = vec![0usize; n];
    for (new_i, &old_i) in perm.iter().enumerate() {
        if old_i >= n {
            return Err(SparseError::ValueError(format!(
                "permutation index {} out of range (n={})",
                old_i, n
            )));
        }
        inv_perm[old_i] = new_i;
    }

    let mut rows = Vec::with_capacity(mat.nnz());
    let mut cols = Vec::with_capacity(mat.nnz());
    let mut data = Vec::with_capacity(mat.nnz());

    for new_row in 0..n {
        let old_row = perm[new_row];
        for j in mat.indptr[old_row]..mat.indptr[old_row + 1] {
            let old_col = mat.indices[j];
            let new_col = inv_perm[old_col];
            rows.push(new_row);
            cols.push(new_col);
            data.push(mat.data[j]);
        }
    }

    CsrMatrix::new(data, rows, cols, (n, n))
}

/// Apply a symmetric permutation to a `CsrArray`: `B = P * A * P^T`.
pub fn apply_permutation_csr_array<T>(
    arr: &CsrArray<T>,
    perm: &[usize],
) -> SparseResult<CsrArray<T>>
where
    T: SparseElement + Div<Output = T> + Zero + PartialOrd + 'static,
{
    let (n, nc) = arr.shape();
    if n != nc {
        return Err(SparseError::ValueError(
            "apply_permutation requires a square matrix".to_string(),
        ));
    }
    if perm.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: perm.len(),
        });
    }

    let mut inv_perm = vec![0usize; n];
    for (new_i, &old_i) in perm.iter().enumerate() {
        if old_i >= n {
            return Err(SparseError::ValueError(format!(
                "permutation index {} out of range (n={})",
                old_i, n
            )));
        }
        inv_perm[old_i] = new_i;
    }

    // Extract dense, permute, rebuild
    let dense = <CsrArray<T> as SparseArray<T>>::to_array(arr);
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    let zero = <T as Zero>::zero();

    for new_row in 0..n {
        let old_row = perm[new_row];
        for old_col in 0..n {
            let val = dense[[old_row, old_col]];
            if val != zero {
                let new_col = inv_perm[old_col];
                rows.push(new_row);
                cols.push(new_col);
                data.push(val);
            }
        }
    }

    CsrArray::from_triplets(&rows, &cols, &data, (n, n), false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjacency_from_list() {
        let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let graph = AdjacencyGraph::from_adjacency_list(adj);
        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(graph.degree(0), 2);
        assert_eq!(graph.num_edges(), 3);
        assert!(graph.has_edge(0, 1));
        assert!(!graph.has_edge(0, 0)); // no self-loops
    }

    #[test]
    fn test_adjacency_from_csr_matrix() {
        // 3x3 symmetric matrix
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![1, 2, 0, 2, 0, 1];
        let data = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mat = CsrMatrix::new(data, rows, cols, (3, 3)).expect("csr");
        let graph = AdjacencyGraph::from_csr_matrix(&mat).expect("adj");
        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(graph.degree(0), 2);
    }

    #[test]
    fn test_subgraph() {
        let adj = vec![vec![1, 2, 3], vec![0, 2], vec![0, 1], vec![0]];
        let graph = AdjacencyGraph::from_adjacency_list(adj);
        let (sub, mapping) = graph.subgraph(&[0, 1, 2]);
        assert_eq!(sub.num_nodes(), 3);
        assert_eq!(mapping, vec![0, 1, 2]);
        // Node 3 is excluded, so node 0's degree drops from 3 to 2
        assert_eq!(sub.degree(0), 2);
    }

    #[test]
    fn test_apply_permutation() {
        // 3x3 tridiagonal
        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 1, 2];
        let data = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
        let mat = CsrMatrix::new(data, rows, cols, (3, 3)).expect("csr");

        // Reverse permutation
        let perm = vec![2, 1, 0];
        let permuted = apply_permutation(&mat, &perm).expect("apply");
        assert_eq!(permuted.shape(), (3, 3));
        assert_eq!(permuted.nnz(), 7);
    }
}
