//! Sparse adjacency matrix in CSR format for GNN operations.
//!
//! [`SparseAdjacency`] stores a graph in Compressed Sparse Row (CSR) format,
//! which gives O(deg(v)) neighbour access and O(E) storage for E edges.
//!
//! # Example
//! ```
//! use scirs2_neural::layers::gnn::SparseAdjacency;
//! let edges = vec![(0usize, 1usize), (1, 2), (2, 0)];
//! let adj = SparseAdjacency::from_edge_list(3, &edges).expect("operation should succeed");
//! assert_eq!(adj.num_nodes(), 3);
//! assert_eq!(adj.num_edges(), 3);
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::Array2;

// ─────────────────────────────────────────────────────────────────────────────
// SparseAdjacency
// ─────────────────────────────────────────────────────────────────────────────

/// Compressed Sparse Row adjacency matrix for graph neural network layers.
///
/// Stores an unweighted directed graph.  For undirected graphs the caller
/// should supply both (u, v) and (v, u) in the edge list.
///
/// Fields use standard CSR terminology:
/// - `row_ptr[i]` – index in `col_idx` where the neighbours of node `i` start.
/// - `col_idx`    – concatenated neighbour lists.
#[derive(Debug, Clone)]
pub struct SparseAdjacency {
    /// Number of nodes.
    num_nodes: usize,
    /// CSR row-pointer array of length `num_nodes + 1`.
    row_ptr: Vec<usize>,
    /// CSR column-index array of length `num_edges`.
    col_idx: Vec<usize>,
    /// Optional edge weights (same length as `col_idx`, or empty = all-ones).
    weights: Vec<f64>,
}

impl SparseAdjacency {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Build a `SparseAdjacency` from an unweighted edge list.
    ///
    /// Duplicate edges are kept as-is; self-loops are allowed.
    ///
    /// # Errors
    /// Returns [`NeuralError::InvalidArgument`] if any node index ≥ `num_nodes`.
    pub fn from_edge_list(num_nodes: usize, edges: &[(usize, usize)]) -> Result<Self> {
        Self::from_weighted_edge_list(
            num_nodes,
            &edges.iter().map(|&(u, v)| (u, v, 1.0_f64)).collect::<Vec<_>>(),
        )
    }

    /// Build a `SparseAdjacency` from a weighted edge list `(src, dst, weight)`.
    ///
    /// # Errors
    /// Returns [`NeuralError::InvalidArgument`] if any node index ≥ `num_nodes`.
    pub fn from_weighted_edge_list(
        num_nodes: usize,
        edges: &[(usize, usize, f64)],
    ) -> Result<Self> {
        // Validate indices
        for &(src, dst, _) in edges {
            if src >= num_nodes {
                return Err(NeuralError::InvalidArgument(format!(
                    "Source node index {src} out of bounds for graph with {num_nodes} nodes"
                )));
            }
            if dst >= num_nodes {
                return Err(NeuralError::InvalidArgument(format!(
                    "Destination node index {dst} out of bounds for graph with {num_nodes} nodes"
                )));
            }
        }

        // Count out-degrees
        let mut degree = vec![0usize; num_nodes];
        for &(src, _, _) in edges {
            degree[src] += 1;
        }

        // Build row_ptr via prefix sum
        let mut row_ptr = vec![0usize; num_nodes + 1];
        for i in 0..num_nodes {
            row_ptr[i + 1] = row_ptr[i] + degree[i];
        }

        // Fill col_idx and weights (insertion sort into each row)
        let num_edges = edges.len();
        let mut col_idx = vec![0usize; num_edges];
        let mut weights = vec![0.0_f64; num_edges];
        let mut cursor = row_ptr[..num_nodes].to_vec(); // per-row write cursor

        for &(src, dst, w) in edges {
            let pos = cursor[src];
            col_idx[pos] = dst;
            weights[pos] = w;
            cursor[src] += 1;
        }

        // Sort each row by column index (stable within row)
        for i in 0..num_nodes {
            let start = row_ptr[i];
            let end = row_ptr[i + 1];
            if end > start + 1 {
                let mut pairs: Vec<(usize, f64)> = col_idx[start..end]
                    .iter()
                    .zip(weights[start..end].iter())
                    .map(|(&c, &w)| (c, w))
                    .collect();
                pairs.sort_unstable_by_key(|&(c, _)| c);
                for (k, (c, w)) in pairs.into_iter().enumerate() {
                    col_idx[start + k] = c;
                    weights[start + k] = w;
                }
            }
        }

        Ok(Self {
            num_nodes,
            row_ptr,
            col_idx,
            weights,
        })
    }

    /// Build a `SparseAdjacency` from a dense adjacency matrix.
    ///
    /// Non-zero entries in `dense` become edges; their values become weights.
    ///
    /// # Errors
    /// Returns [`NeuralError::InvalidArgument`] if `dense` is not square.
    pub fn from_dense(dense: &Array2<f64>) -> Result<Self> {
        let shape = dense.shape();
        if shape[0] != shape[1] {
            return Err(NeuralError::InvalidArgument(format!(
                "Adjacency matrix must be square, got {}×{}",
                shape[0], shape[1]
            )));
        }
        let n = shape[0];
        let mut edges = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let v = dense[[i, j]];
                if v != 0.0 {
                    edges.push((i, j, v));
                }
            }
        }
        Self::from_weighted_edge_list(n, &edges)
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    /// Number of nodes in the graph.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Number of directed edges (= length of col_idx).
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.col_idx.len()
    }

    /// Return the slice of neighbour indices for node `v`.
    ///
    /// # Errors
    /// Returns [`NeuralError::InvalidArgument`] if `v >= num_nodes`.
    pub fn neighbors(&self, v: usize) -> Result<&[usize]> {
        if v >= self.num_nodes {
            return Err(NeuralError::InvalidArgument(format!(
                "Node index {v} out of bounds (num_nodes={})",
                self.num_nodes
            )));
        }
        let start = self.row_ptr[v];
        let end = self.row_ptr[v + 1];
        Ok(&self.col_idx[start..end])
    }

    /// Return the slice of edge weights leaving node `v`.
    ///
    /// # Errors
    /// Returns [`NeuralError::InvalidArgument`] if `v >= num_nodes`.
    pub fn neighbor_weights(&self, v: usize) -> Result<&[f64]> {
        if v >= self.num_nodes {
            return Err(NeuralError::InvalidArgument(format!(
                "Node index {v} out of bounds (num_nodes={})",
                self.num_nodes
            )));
        }
        let start = self.row_ptr[v];
        let end = self.row_ptr[v + 1];
        Ok(&self.weights[start..end])
    }

    /// Degree (out-degree) of node `v`.
    ///
    /// # Errors
    /// Returns [`NeuralError::InvalidArgument`] if `v >= num_nodes`.
    pub fn degree(&self, v: usize) -> Result<usize> {
        if v >= self.num_nodes {
            return Err(NeuralError::InvalidArgument(format!(
                "Node index {v} out of bounds (num_nodes={})",
                self.num_nodes
            )));
        }
        Ok(self.row_ptr[v + 1] - self.row_ptr[v])
    }

    // ── Transformations ──────────────────────────────────────────────────────

    /// Convert to a dense `num_nodes × num_nodes` adjacency matrix.
    ///
    /// The returned matrix contains edge weights (or 0/1 for unweighted graphs).
    pub fn to_dense(&self) -> Array2<f64> {
        let n = self.num_nodes;
        let mut dense = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for k in start..end {
                dense[[i, self.col_idx[k]]] = self.weights[k];
            }
        }
        dense
    }

    /// Return a new `SparseAdjacency` with self-loops added (i → i for all i).
    ///
    /// If a self-loop already exists its weight is overwritten with `weight`.
    pub fn add_self_loops(&self, weight: f64) -> Result<Self> {
        let n = self.num_nodes;
        // Collect existing edges
        let mut edge_set: Vec<(usize, usize, f64)> = Vec::with_capacity(self.num_edges() + n);
        for i in 0..n {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for k in start..end {
                // Skip old self-loops; we'll re-add them with the new weight
                if self.col_idx[k] != i {
                    edge_set.push((i, self.col_idx[k], self.weights[k]));
                }
            }
        }
        for i in 0..n {
            edge_set.push((i, i, weight));
        }
        Self::from_weighted_edge_list(n, &edge_set)
    }

    /// Compute the (out-)degree vector `d[i]` as a `Vec<f64>` summing edge weights.
    pub fn degree_vector(&self) -> Vec<f64> {
        (0..self.num_nodes)
            .map(|i| {
                let start = self.row_ptr[i];
                let end = self.row_ptr[i + 1];
                self.weights[start..end].iter().sum()
            })
            .collect()
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Expose `row_ptr` for use by GNN layer implementations.
    #[inline]
    pub(crate) fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Expose `col_idx` for use by GNN layer implementations.
    #[inline]
    pub(crate) fn col_idx(&self) -> &[usize] {
        &self.col_idx
    }

    /// Expose edge `weights` for use by GNN layer implementations.
    #[inline]
    pub(crate) fn edge_weights(&self) -> &[f64] {
        &self.weights
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_edge_list_basic() {
        let edges = vec![(0usize, 1usize), (1, 2), (2, 0)];
        let adj = SparseAdjacency::from_edge_list(3, &edges).expect("build adj");
        assert_eq!(adj.num_nodes(), 3);
        assert_eq!(adj.num_edges(), 3);
    }

    #[test]
    fn test_neighbors() {
        let edges = vec![(0usize, 1usize), (0, 2), (1, 2)];
        let adj = SparseAdjacency::from_edge_list(3, &edges).expect("build adj");
        let nbrs = adj.neighbors(0).expect("neighbors");
        assert_eq!(nbrs, &[1, 2]);
        let nbrs1 = adj.neighbors(1).expect("neighbors");
        assert_eq!(nbrs1, &[2]);
        let nbrs2 = adj.neighbors(2).expect("neighbors");
        assert!(nbrs2.is_empty());
    }

    #[test]
    fn test_to_dense_roundtrip() {
        let edges = vec![(0usize, 1usize), (1, 2), (2, 0)];
        let adj = SparseAdjacency::from_edge_list(3, &edges).expect("build adj");
        let dense = adj.to_dense();
        assert_eq!(dense[[0, 1]], 1.0);
        assert_eq!(dense[[1, 2]], 1.0);
        assert_eq!(dense[[2, 0]], 1.0);
        assert_eq!(dense[[0, 0]], 0.0);
    }

    #[test]
    fn test_add_self_loops() {
        let edges = vec![(0usize, 1usize), (1, 2)];
        let adj = SparseAdjacency::from_edge_list(3, &edges)
            .expect("build adj")
            .add_self_loops(1.0)
            .expect("self loops");
        // 2 original + 3 self-loops = 5
        assert_eq!(adj.num_edges(), 5);
        let dense = adj.to_dense();
        assert_eq!(dense[[0, 0]], 1.0);
        assert_eq!(dense[[1, 1]], 1.0);
        assert_eq!(dense[[2, 2]], 1.0);
    }

    #[test]
    fn test_from_dense() {
        use scirs2_core::ndarray::Array2;
        let mut m = Array2::<f64>::zeros((3, 3));
        m[[0, 1]] = 1.0;
        m[[1, 2]] = 0.5;
        let adj = SparseAdjacency::from_dense(&m).expect("from dense");
        assert_eq!(adj.num_edges(), 2);
        let ws = adj.neighbor_weights(1).expect("weights");
        assert!((ws[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_invalid_node_index() {
        let result = SparseAdjacency::from_edge_list(3, &[(0usize, 5usize)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_degree_vector() {
        let edges = vec![(0usize, 1usize), (0, 2), (1, 2)];
        let adj = SparseAdjacency::from_edge_list(3, &edges).expect("build adj");
        let dv = adj.degree_vector();
        assert_eq!(dv, vec![2.0, 1.0, 0.0]);
    }
}
