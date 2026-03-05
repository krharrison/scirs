//! Compressed Sparse Row (CSR) graph representation for large-scale graphs
//!
//! This module provides a memory-efficient CSR graph format optimized for
//! large graph analytics workloads. It offers O(1) neighbor access start,
//! O(degree) neighbor iteration, and efficient parallel construction.
//!
//! # Key Features
//!
//! - **Memory efficient**: Contiguous arrays minimize cache misses and allocator overhead
//! - **Fast neighbor access**: O(1) to locate neighbor list start, O(degree) iteration
//! - **Weighted/unweighted**: Optional edge weights with zero-cost abstraction
//! - **Directed/undirected**: Support for both graph types
//! - **Parallel construction**: Feature-gated parallel edge list sorting and prefix sum
//! - **Conversions**: Convert to/from adjacency list and `Graph<usize, f64>` types

use crate::error::{GraphError, Result};

#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;

/// A compressed sparse row (CSR) graph representation.
///
/// Stores edges in three contiguous arrays:
/// - `row_ptr[i]..row_ptr[i+1]` gives the range of neighbors for node `i`
/// - `col_indices[row_ptr[i]..row_ptr[i+1]]` are the neighbor node IDs
/// - `values[row_ptr[i]..row_ptr[i+1]]` are the corresponding edge weights
///
/// This format is identical to the CSR sparse matrix format from numerical
/// linear algebra, enabling direct use in sparse matrix-vector products
/// (e.g., for PageRank).
#[derive(Debug, Clone)]
pub struct CsrGraph {
    /// Number of nodes in the graph
    num_nodes: usize,
    /// Number of (directed) edges stored
    num_edges: usize,
    /// Row pointers: length = num_nodes + 1
    row_ptr: Vec<usize>,
    /// Column indices for each edge: length = num_edges
    col_indices: Vec<usize>,
    /// Edge weights: length = num_edges
    values: Vec<f64>,
    /// Whether this graph is directed
    directed: bool,
}

/// Builder for constructing CSR graphs from edge lists.
///
/// Accumulates edges and then performs a single sort + prefix-sum
/// to build the CSR arrays efficiently.
#[derive(Debug, Clone)]
pub struct CsrGraphBuilder {
    num_nodes: usize,
    edges: Vec<(usize, usize, f64)>,
    directed: bool,
}

/// An adjacency list representation for graph interchange.
#[derive(Debug, Clone)]
pub struct AdjacencyList {
    /// Number of nodes
    pub num_nodes: usize,
    /// For each node, a list of (neighbor, weight) pairs
    pub adjacency: Vec<Vec<(usize, f64)>>,
    /// Whether the graph is directed
    pub directed: bool,
}

// ────────────────────────────────────────────────────────────────────────────
// CsrGraphBuilder
// ────────────────────────────────────────────────────────────────────────────

impl CsrGraphBuilder {
    /// Create a new CSR graph builder.
    ///
    /// # Arguments
    /// * `num_nodes` - Number of nodes in the graph
    /// * `directed` - Whether the graph is directed
    pub fn new(num_nodes: usize, directed: bool) -> Self {
        Self {
            num_nodes,
            edges: Vec::new(),
            directed,
        }
    }

    /// Create a builder with pre-allocated edge capacity.
    pub fn with_capacity(num_nodes: usize, edge_capacity: usize, directed: bool) -> Self {
        Self {
            num_nodes,
            edges: Vec::with_capacity(edge_capacity),
            directed,
        }
    }

    /// Add a single edge.
    ///
    /// For undirected graphs, the reverse edge is added automatically during build.
    pub fn add_edge(&mut self, src: usize, dst: usize, weight: f64) -> Result<()> {
        if src >= self.num_nodes {
            return Err(GraphError::node_not_found_with_context(
                src,
                self.num_nodes,
                "CsrGraphBuilder::add_edge (source)",
            ));
        }
        if dst >= self.num_nodes {
            return Err(GraphError::node_not_found_with_context(
                dst,
                self.num_nodes,
                "CsrGraphBuilder::add_edge (destination)",
            ));
        }
        self.edges.push((src, dst, weight));
        Ok(())
    }

    /// Add an unweighted edge (weight = 1.0).
    pub fn add_unweighted_edge(&mut self, src: usize, dst: usize) -> Result<()> {
        self.add_edge(src, dst, 1.0)
    }

    /// Add edges from an iterator of `(src, dst, weight)` triples.
    pub fn add_edges<I>(&mut self, edges: I) -> Result<()>
    where
        I: IntoIterator<Item = (usize, usize, f64)>,
    {
        for (src, dst, weight) in edges {
            self.add_edge(src, dst, weight)?;
        }
        Ok(())
    }

    /// Build the CSR graph (sequential).
    pub fn build(self) -> Result<CsrGraph> {
        build_csr_sequential(self.num_nodes, self.edges, self.directed)
    }

    /// Build the CSR graph using parallel sorting and construction.
    #[cfg(feature = "parallel")]
    pub fn build_parallel(self) -> Result<CsrGraph> {
        build_csr_parallel(self.num_nodes, self.edges, self.directed)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Sequential CSR construction
// ────────────────────────────────────────────────────────────────────────────

fn build_csr_sequential(
    num_nodes: usize,
    mut edges: Vec<(usize, usize, f64)>,
    directed: bool,
) -> Result<CsrGraph> {
    // For undirected graphs, add reverse edges
    if !directed {
        let reverse: Vec<(usize, usize, f64)> = edges.iter().map(|&(s, d, w)| (d, s, w)).collect();
        edges.extend(reverse);
    }

    let num_edges = edges.len();

    // Validate all node indices
    for &(src, dst, _) in &edges {
        if src >= num_nodes {
            return Err(GraphError::node_not_found_with_context(
                src,
                num_nodes,
                "CSR construction (source)",
            ));
        }
        if dst >= num_nodes {
            return Err(GraphError::node_not_found_with_context(
                dst,
                num_nodes,
                "CSR construction (destination)",
            ));
        }
    }

    // Count degrees (first pass)
    let mut degree = vec![0usize; num_nodes];
    for &(src, _, _) in &edges {
        degree[src] += 1;
    }

    // Build row_ptr via prefix sum
    let mut row_ptr = Vec::with_capacity(num_nodes + 1);
    row_ptr.push(0);
    for &deg in &degree {
        let last = row_ptr.last().copied().unwrap_or(0);
        row_ptr.push(last + deg);
    }

    // Place edges into CSR arrays using counting sort
    let mut col_indices = vec![0usize; num_edges];
    let mut values = vec![0.0f64; num_edges];
    let mut current_pos: Vec<usize> = row_ptr[..num_nodes].to_vec();

    for (src, dst, weight) in &edges {
        let pos = current_pos[*src];
        col_indices[pos] = *dst;
        values[pos] = *weight;
        current_pos[*src] += 1;
    }

    // Sort neighbors within each row for cache-friendly access and binary search
    for node in 0..num_nodes {
        let start = row_ptr[node];
        let end = row_ptr[node + 1];
        if end > start + 1 {
            // Build (col, value) pairs, sort by col, write back
            let mut pairs: Vec<(usize, f64)> = col_indices[start..end]
                .iter()
                .zip(&values[start..end])
                .map(|(&c, &v)| (c, v))
                .collect();
            pairs.sort_unstable_by_key(|&(c, _)| c);
            for (i, (c, v)) in pairs.into_iter().enumerate() {
                col_indices[start + i] = c;
                values[start + i] = v;
            }
        }
    }

    Ok(CsrGraph {
        num_nodes,
        num_edges,
        row_ptr,
        col_indices,
        values,
        directed,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Parallel CSR construction
// ────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "parallel")]
fn build_csr_parallel(
    num_nodes: usize,
    mut edges: Vec<(usize, usize, f64)>,
    directed: bool,
) -> Result<CsrGraph> {
    // For undirected graphs, add reverse edges
    if !directed {
        let reverse: Vec<(usize, usize, f64)> = edges.iter().map(|&(s, d, w)| (d, s, w)).collect();
        edges.extend(reverse);
    }

    let num_edges = edges.len();

    // Validate all node indices
    for &(src, dst, _) in &edges {
        if src >= num_nodes {
            return Err(GraphError::node_not_found_with_context(
                src,
                num_nodes,
                "CSR parallel construction (source)",
            ));
        }
        if dst >= num_nodes {
            return Err(GraphError::node_not_found_with_context(
                dst,
                num_nodes,
                "CSR parallel construction (destination)",
            ));
        }
    }

    // Parallel sort edges by source node
    edges.par_sort_unstable_by_key(|&(src, _, _)| src);

    // Count degrees using parallel fold
    let degree: Vec<usize> = {
        let mut deg = vec![0usize; num_nodes];
        for &(src, _, _) in &edges {
            deg[src] += 1;
        }
        deg
    };

    // Build row_ptr via prefix sum
    let mut row_ptr = Vec::with_capacity(num_nodes + 1);
    row_ptr.push(0);
    for &deg in &degree {
        let last = row_ptr.last().copied().unwrap_or(0);
        row_ptr.push(last + deg);
    }

    // Since edges are sorted by source, we can directly split
    let mut col_indices = Vec::with_capacity(num_edges);
    let mut values = Vec::with_capacity(num_edges);
    for &(_, dst, weight) in &edges {
        col_indices.push(dst);
        values.push(weight);
    }

    // Sort neighbors within each row sequentially
    // (row-level parallelism is already handled by the parallel sort above)
    for node in 0..num_nodes {
        let start = row_ptr[node];
        let end = row_ptr[node + 1];
        if end > start + 1 {
            let mut pairs: Vec<(usize, f64)> = col_indices[start..end]
                .iter()
                .zip(&values[start..end])
                .map(|(&c, &v)| (c, v))
                .collect();
            pairs.sort_unstable_by_key(|&(c, _)| c);
            for (i, (c, v)) in pairs.into_iter().enumerate() {
                col_indices[start + i] = c;
                values[start + i] = v;
            }
        }
    }

    Ok(CsrGraph {
        num_nodes,
        num_edges,
        row_ptr,
        col_indices,
        values,
        directed,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// CsrGraph core API
// ────────────────────────────────────────────────────────────────────────────

impl CsrGraph {
    /// Construct a CSR graph directly from raw arrays.
    ///
    /// # Safety contract (logical, not `unsafe`)
    /// The caller must ensure:
    /// - `row_ptr.len() == num_nodes + 1`
    /// - All values in `col_indices` are `< num_nodes`
    /// - `col_indices.len() == values.len() == row_ptr[num_nodes]`
    pub fn from_raw(
        num_nodes: usize,
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
        directed: bool,
    ) -> Result<Self> {
        if row_ptr.len() != num_nodes + 1 {
            return Err(GraphError::InvalidGraph(format!(
                "row_ptr length {} does not match num_nodes + 1 = {}",
                row_ptr.len(),
                num_nodes + 1
            )));
        }
        if col_indices.len() != values.len() {
            return Err(GraphError::InvalidGraph(format!(
                "col_indices length {} != values length {}",
                col_indices.len(),
                values.len()
            )));
        }
        let num_edges = col_indices.len();
        let last_ptr = row_ptr.last().copied().unwrap_or(0);
        if last_ptr != num_edges {
            return Err(GraphError::InvalidGraph(format!(
                "row_ptr last element {} != num_edges {}",
                last_ptr, num_edges
            )));
        }
        // Validate column indices
        for (i, &col) in col_indices.iter().enumerate() {
            if col >= num_nodes {
                return Err(GraphError::node_not_found_with_context(
                    col,
                    num_nodes,
                    &format!("from_raw validation at edge index {i}"),
                ));
            }
        }
        Ok(Self {
            num_nodes,
            num_edges,
            row_ptr,
            col_indices,
            values,
            directed,
        })
    }

    /// Construct from an edge list (convenience wrapper).
    ///
    /// For undirected graphs, reverse edges are added automatically.
    pub fn from_edges(
        num_nodes: usize,
        edges: Vec<(usize, usize, f64)>,
        directed: bool,
    ) -> Result<Self> {
        build_csr_sequential(num_nodes, edges, directed)
    }

    /// Construct from an edge list using parallel construction.
    #[cfg(feature = "parallel")]
    pub fn from_edges_parallel(
        num_nodes: usize,
        edges: Vec<(usize, usize, f64)>,
        directed: bool,
    ) -> Result<Self> {
        build_csr_parallel(num_nodes, edges, directed)
    }

    /// Construct an unweighted CSR graph from an edge list.
    pub fn from_unweighted_edges(
        num_nodes: usize,
        edges: &[(usize, usize)],
        directed: bool,
    ) -> Result<Self> {
        let weighted: Vec<(usize, usize, f64)> = edges.iter().map(|&(s, d)| (s, d, 1.0)).collect();
        build_csr_sequential(num_nodes, weighted, directed)
    }

    /// Number of nodes.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Number of stored (directed) edges.
    ///
    /// For undirected graphs built from `n` input edges, this returns `2n`
    /// because both directions are stored.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Number of logical edges.
    ///
    /// For undirected graphs, returns `num_edges / 2` (the original count).
    /// For directed graphs, returns `num_edges`.
    #[inline]
    pub fn num_logical_edges(&self) -> usize {
        if self.directed {
            self.num_edges
        } else {
            self.num_edges / 2
        }
    }

    /// Whether the graph is directed.
    #[inline]
    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// Out-degree of a node (number of outgoing edges).
    ///
    /// Returns 0 if the node index is out of range.
    #[inline]
    pub fn degree(&self, node: usize) -> usize {
        if node >= self.num_nodes {
            return 0;
        }
        self.row_ptr[node + 1] - self.row_ptr[node]
    }

    /// Iterator over neighbors of `node` as `(neighbor_id, weight)` pairs.
    ///
    /// Returns an empty iterator if `node` is out of range.
    #[inline]
    pub fn neighbors(&self, node: usize) -> NeighborIter<'_> {
        if node >= self.num_nodes {
            return NeighborIter {
                col_iter: [].iter(),
                val_iter: [].iter(),
            };
        }
        let start = self.row_ptr[node];
        let end = self.row_ptr[node + 1];
        NeighborIter {
            col_iter: self.col_indices[start..end].iter(),
            val_iter: self.values[start..end].iter(),
        }
    }

    /// Check if an edge exists from `src` to `dst`.
    ///
    /// Uses binary search on the sorted neighbor list. O(log(degree)).
    pub fn has_edge(&self, src: usize, dst: usize) -> bool {
        if src >= self.num_nodes || dst >= self.num_nodes {
            return false;
        }
        let start = self.row_ptr[src];
        let end = self.row_ptr[src + 1];
        self.col_indices[start..end].binary_search(&dst).is_ok()
    }

    /// Get the weight of an edge from `src` to `dst`.
    ///
    /// Returns `None` if the edge does not exist.
    pub fn edge_weight(&self, src: usize, dst: usize) -> Option<f64> {
        if src >= self.num_nodes || dst >= self.num_nodes {
            return None;
        }
        let start = self.row_ptr[src];
        let end = self.row_ptr[src + 1];
        match self.col_indices[start..end].binary_search(&dst) {
            Ok(idx) => Some(self.values[start + idx]),
            Err(_) => None,
        }
    }

    /// Get the raw row pointer array (read-only).
    #[inline]
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Get the raw column index array (read-only).
    #[inline]
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Get the raw values array (read-only).
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Memory usage in bytes (approximate).
    pub fn memory_bytes(&self) -> usize {
        use std::mem::size_of;
        // Struct overhead
        size_of::<Self>()
            // row_ptr
            + self.row_ptr.capacity() * size_of::<usize>()
            // col_indices
            + self.col_indices.capacity() * size_of::<usize>()
            // values
            + self.values.capacity() * size_of::<f64>()
    }

    /// Sparse matrix-vector product: y = A * x.
    ///
    /// This is the core operation for iterative algorithms like PageRank.
    /// `x` and the returned vector both have length `num_nodes`.
    pub fn spmv(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.num_nodes {
            return Err(GraphError::InvalidGraph(format!(
                "spmv: vector length {} != num_nodes {}",
                x.len(),
                self.num_nodes
            )));
        }
        let mut y = vec![0.0f64; self.num_nodes];
        for row in 0..self.num_nodes {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            let mut sum = 0.0;
            for idx in start..end {
                sum += self.values[idx] * x[self.col_indices[idx]];
            }
            y[row] = sum;
        }
        Ok(y)
    }

    /// Parallel sparse matrix-vector product: y = A * x.
    #[cfg(feature = "parallel")]
    pub fn spmv_parallel(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.num_nodes {
            return Err(GraphError::InvalidGraph(format!(
                "spmv_parallel: vector length {} != num_nodes {}",
                x.len(),
                self.num_nodes
            )));
        }
        let y: Vec<f64> = (0..self.num_nodes)
            .into_par_iter()
            .map(|row| {
                let start = self.row_ptr[row];
                let end = self.row_ptr[row + 1];
                let mut sum = 0.0;
                for idx in start..end {
                    sum += self.values[idx] * x[self.col_indices[idx]];
                }
                sum
            })
            .collect();
        Ok(y)
    }

    /// Transpose the graph (reverse all edge directions).
    ///
    /// For undirected graphs, the transpose is the same graph.
    pub fn transpose(&self) -> Result<Self> {
        if !self.directed {
            return Ok(self.clone());
        }
        // Collect all edges in reversed form
        let mut edges = Vec::with_capacity(self.num_edges);
        for src in 0..self.num_nodes {
            for (dst, weight) in self.neighbors(src) {
                edges.push((dst, src, weight));
            }
        }
        build_csr_sequential(self.num_nodes, edges, true)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// NeighborIter
// ────────────────────────────────────────────────────────────────────────────

/// Iterator over (neighbor_id, weight) pairs for a node.
pub struct NeighborIter<'a> {
    col_iter: std::slice::Iter<'a, usize>,
    val_iter: std::slice::Iter<'a, f64>,
}

impl<'a> Iterator for NeighborIter<'a> {
    type Item = (usize, f64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match (self.col_iter.next(), self.val_iter.next()) {
            (Some(&col), Some(&val)) => Some((col, val)),
            _ => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.col_iter.size_hint()
    }
}

impl<'a> ExactSizeIterator for NeighborIter<'a> {}

// ────────────────────────────────────────────────────────────────────────────
// Conversions: CsrGraph <-> AdjacencyList
// ────────────────────────────────────────────────────────────────────────────

impl CsrGraph {
    /// Convert to an adjacency list representation.
    pub fn to_adjacency_list(&self) -> AdjacencyList {
        let mut adjacency = Vec::with_capacity(self.num_nodes);
        for node in 0..self.num_nodes {
            let neighbors: Vec<(usize, f64)> = self.neighbors(node).collect();
            adjacency.push(neighbors);
        }
        AdjacencyList {
            num_nodes: self.num_nodes,
            adjacency,
            directed: self.directed,
        }
    }

    /// Construct a CSR graph from an adjacency list.
    ///
    /// The adjacency list is consumed and edges are extracted directly.
    /// For undirected graphs, the adjacency list should already contain
    /// both directions (i.e., if `(u,v)` is present in `adj[u]`, then `(v,u)`
    /// should be in `adj[v]`).
    pub fn from_adjacency_list(adj: &AdjacencyList) -> Result<Self> {
        let num_nodes = adj.num_nodes;
        let mut edges = Vec::new();
        for (src, neighbors) in adj.adjacency.iter().enumerate() {
            for &(dst, weight) in neighbors {
                edges.push((src, dst, weight));
            }
        }
        // Since adjacency list already has both directions for undirected,
        // we build as directed to avoid doubling
        let num_edges = edges.len();

        // Validate
        for &(src, dst, _) in &edges {
            if src >= num_nodes {
                return Err(GraphError::node_not_found_with_context(
                    src,
                    num_nodes,
                    "from_adjacency_list (source)",
                ));
            }
            if dst >= num_nodes {
                return Err(GraphError::node_not_found_with_context(
                    dst,
                    num_nodes,
                    "from_adjacency_list (destination)",
                ));
            }
        }

        // Count degrees
        let mut degree = vec![0usize; num_nodes];
        for &(src, _, _) in &edges {
            degree[src] += 1;
        }

        // Build row_ptr
        let mut row_ptr = Vec::with_capacity(num_nodes + 1);
        row_ptr.push(0);
        for &deg in &degree {
            let last = row_ptr.last().copied().unwrap_or(0);
            row_ptr.push(last + deg);
        }

        // Fill arrays
        let mut col_indices = vec![0usize; num_edges];
        let mut values = vec![0.0f64; num_edges];
        let mut current_pos: Vec<usize> = row_ptr[..num_nodes].to_vec();

        for &(src, dst, weight) in &edges {
            let pos = current_pos[src];
            col_indices[pos] = dst;
            values[pos] = weight;
            current_pos[src] += 1;
        }

        // Sort neighbors within each row
        for node in 0..num_nodes {
            let start = row_ptr[node];
            let end = row_ptr[node + 1];
            if end > start + 1 {
                let mut pairs: Vec<(usize, f64)> = col_indices[start..end]
                    .iter()
                    .zip(&values[start..end])
                    .map(|(&c, &v)| (c, v))
                    .collect();
                pairs.sort_unstable_by_key(|&(c, _)| c);
                for (i, (c, v)) in pairs.into_iter().enumerate() {
                    col_indices[start + i] = c;
                    values[start + i] = v;
                }
            }
        }

        Ok(Self {
            num_nodes,
            num_edges,
            row_ptr,
            col_indices,
            values,
            directed: adj.directed,
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Conversions: CsrGraph <-> Graph<usize, f64>
// ────────────────────────────────────────────────────────────────────────────

impl CsrGraph {
    /// Convert to a `Graph<usize, f64>` (undirected adjacency-list graph).
    ///
    /// For directed CSR graphs, the resulting `Graph` will be undirected
    /// (edges are treated as undirected).
    pub fn to_graph(&self) -> crate::Graph<usize, f64> {
        let mut graph = crate::Graph::new();
        for i in 0..self.num_nodes {
            graph.add_node(i);
        }
        // For undirected CSR, each edge is stored twice. Only add once.
        for src in 0..self.num_nodes {
            for (dst, weight) in self.neighbors(src) {
                if self.directed || src <= dst {
                    // Ignore errors from duplicate edges
                    let _ = graph.add_edge(src, dst, weight);
                }
            }
        }
        graph
    }

    /// Construct a CSR graph from a `Graph<usize, f64>`.
    pub fn from_graph(graph: &crate::Graph<usize, f64>) -> Result<Self> {
        let num_nodes = graph.node_count();
        let edges: Vec<(usize, usize, f64)> = graph
            .edges()
            .into_iter()
            .map(|e| (e.source, e.target, e.weight))
            .collect();
        // Graph<usize, f64> is undirected
        build_csr_sequential(num_nodes, edges, false)
    }

    /// Construct a CSR graph from a `DiGraph<usize, f64>`.
    pub fn from_digraph(graph: &crate::DiGraph<usize, f64>) -> Result<Self> {
        let num_nodes = graph.node_count();
        let edges: Vec<(usize, usize, f64)> = graph
            .edges()
            .into_iter()
            .map(|e| (e.source, e.target, e.weight))
            .collect();
        build_csr_sequential(num_nodes, edges, true)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Display
// ────────────────────────────────────────────────────────────────────────────

impl std::fmt::Display for CsrGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CsrGraph(nodes={}, edges={}, directed={}, mem={}KB)",
            self.num_nodes,
            self.num_logical_edges(),
            self.directed,
            self.memory_bytes() / 1024
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_from_edges_directed() {
        let edges = vec![(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0), (2, 0, 0.5)];
        let g = CsrGraph::from_edges(4, edges, true).expect("build failed");

        assert_eq!(g.num_nodes(), 4);
        assert_eq!(g.num_edges(), 4);
        assert_eq!(g.num_logical_edges(), 4);
        assert!(g.is_directed());

        // Node 0 neighbors: 1, 2
        assert_eq!(g.degree(0), 2);
        let n0: Vec<(usize, f64)> = g.neighbors(0).collect();
        assert_eq!(n0, vec![(1, 1.0), (2, 2.0)]);

        // Node 2 neighbors: 0
        assert_eq!(g.degree(2), 1);

        // Node 3 has no outgoing edges
        assert_eq!(g.degree(3), 0);

        // Edge checks
        assert!(g.has_edge(0, 1));
        assert!(g.has_edge(2, 0));
        assert!(!g.has_edge(1, 0)); // directed
        assert!(!g.has_edge(3, 0));

        // Weight lookup
        assert_eq!(g.edge_weight(0, 2), Some(2.0));
        assert_eq!(g.edge_weight(1, 0), None);
    }

    #[test]
    fn test_csr_from_edges_undirected() {
        let edges = vec![(0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0)];
        let g = CsrGraph::from_edges(4, edges, false).expect("build failed");

        assert_eq!(g.num_nodes(), 4);
        assert_eq!(g.num_edges(), 6); // 3 edges * 2 directions
        assert_eq!(g.num_logical_edges(), 3);
        assert!(!g.is_directed());

        // Both directions present
        assert!(g.has_edge(0, 1));
        assert!(g.has_edge(1, 0));
        assert!(g.has_edge(2, 3));
        assert!(g.has_edge(3, 2));
        assert!(!g.has_edge(0, 3));
    }

    #[test]
    fn test_csr_builder() {
        let mut builder = CsrGraphBuilder::with_capacity(5, 4, true);
        builder.add_edge(0, 1, 1.0).expect("add edge failed");
        builder.add_edge(0, 2, 2.0).expect("add edge failed");
        builder.add_unweighted_edge(3, 4).expect("add edge failed");
        builder.add_unweighted_edge(4, 0).expect("add edge failed");

        let g = builder.build().expect("build failed");
        assert_eq!(g.num_nodes(), 5);
        assert_eq!(g.degree(0), 2);
        assert_eq!(g.degree(3), 1);
        assert!(g.has_edge(4, 0));
    }

    #[test]
    fn test_csr_builder_validation() {
        let mut builder = CsrGraphBuilder::new(3, true);
        assert!(builder.add_edge(0, 1, 1.0).is_ok());
        assert!(builder.add_edge(5, 1, 1.0).is_err()); // src out of range
        assert!(builder.add_edge(0, 5, 1.0).is_err()); // dst out of range
    }

    #[test]
    fn test_csr_from_raw() {
        let row_ptr = vec![0, 2, 3, 4];
        let col_indices = vec![1, 2, 0, 1];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let g = CsrGraph::from_raw(3, row_ptr, col_indices, values, true).expect("from_raw");
        assert_eq!(g.num_nodes(), 3);
        assert_eq!(g.num_edges(), 4);
        assert_eq!(g.degree(0), 2);
    }

    #[test]
    fn test_csr_from_raw_validation() {
        // Wrong row_ptr length
        let r = CsrGraph::from_raw(3, vec![0, 1], vec![0], vec![1.0], true);
        assert!(r.is_err());

        // Mismatched col/val lengths
        let r = CsrGraph::from_raw(2, vec![0, 1, 2], vec![1, 0], vec![1.0], true);
        assert!(r.is_err());

        // Column index out of range
        let r = CsrGraph::from_raw(2, vec![0, 1, 2], vec![1, 5], vec![1.0, 2.0], true);
        assert!(r.is_err());
    }

    #[test]
    fn test_csr_spmv() {
        // Simple directed: 0->1 (w=2), 1->0 (w=3)
        let g =
            CsrGraph::from_edges(2, vec![(0, 1, 2.0), (1, 0, 3.0)], true).expect("build failed");
        let x = vec![1.0, 2.0];
        let y = g.spmv(&x).expect("spmv failed");
        // y[0] = 2.0 * 2.0 = 4.0
        // y[1] = 3.0 * 1.0 = 3.0
        assert!((y[0] - 4.0).abs() < 1e-10);
        assert!((y[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_spmv_wrong_length() {
        let g = CsrGraph::from_edges(3, vec![(0, 1, 1.0)], true).expect("build failed");
        let r = g.spmv(&[1.0, 2.0]);
        assert!(r.is_err());
    }

    #[test]
    fn test_csr_transpose() {
        let g = CsrGraph::from_edges(3, vec![(0, 1, 1.0), (0, 2, 2.0), (2, 1, 3.0)], true)
            .expect("build");
        let gt = g.transpose().expect("transpose");

        assert_eq!(gt.num_nodes(), 3);
        assert!(gt.has_edge(1, 0));
        assert!(gt.has_edge(2, 0));
        assert!(gt.has_edge(1, 2));
        assert!(!gt.has_edge(0, 1)); // reversed
    }

    #[test]
    fn test_csr_transpose_undirected() {
        let g = CsrGraph::from_edges(3, vec![(0, 1, 1.0)], false).expect("build");
        let gt = g.transpose().expect("transpose");
        // For undirected, transpose is identity
        assert_eq!(gt.num_edges(), g.num_edges());
        assert!(gt.has_edge(0, 1));
        assert!(gt.has_edge(1, 0));
    }

    #[test]
    fn test_csr_adjacency_list_roundtrip() {
        let edges = vec![(0, 1, 1.5), (1, 2, 2.5), (2, 0, 3.5)];
        let g = CsrGraph::from_edges(3, edges, true).expect("build");

        let adj = g.to_adjacency_list();
        assert_eq!(adj.num_nodes, 3);
        assert_eq!(adj.adjacency[0].len(), 1); // 0->1
        assert_eq!(adj.adjacency[1].len(), 1); // 1->2
        assert_eq!(adj.adjacency[2].len(), 1); // 2->0

        let g2 = CsrGraph::from_adjacency_list(&adj).expect("from adj");
        assert_eq!(g2.num_nodes(), 3);
        assert_eq!(g2.num_edges(), 3);
        assert!(g2.has_edge(0, 1));
        assert!(g2.has_edge(1, 2));
        assert!(g2.has_edge(2, 0));
        assert_eq!(g2.edge_weight(0, 1), Some(1.5));
    }

    #[test]
    fn test_csr_graph_conversion() {
        let mut graph: crate::Graph<usize, f64> = crate::Graph::new();
        for i in 0..5 {
            graph.add_node(i);
        }
        graph.add_edge(0, 1, 1.0).expect("add edge");
        graph.add_edge(1, 2, 2.0).expect("add edge");
        graph.add_edge(2, 3, 3.0).expect("add edge");
        graph.add_edge(3, 4, 4.0).expect("add edge");

        let csr = CsrGraph::from_graph(&graph).expect("from_graph");
        assert_eq!(csr.num_nodes(), 5);
        assert_eq!(csr.num_logical_edges(), 4);
        assert!(!csr.is_directed());
        assert!(csr.has_edge(0, 1));
        assert!(csr.has_edge(1, 0)); // undirected

        // Convert back
        let graph2 = csr.to_graph();
        assert_eq!(graph2.node_count(), 5);
        assert_eq!(graph2.edge_count(), 4);
    }

    #[test]
    fn test_csr_empty_graph() {
        let g = CsrGraph::from_edges(5, vec![], true).expect("build");
        assert_eq!(g.num_nodes(), 5);
        assert_eq!(g.num_edges(), 0);
        assert_eq!(g.degree(0), 0);
        assert!(!g.has_edge(0, 1));
        let neighbors: Vec<_> = g.neighbors(0).collect();
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_csr_single_node() {
        let g = CsrGraph::from_edges(1, vec![], true).expect("build");
        assert_eq!(g.num_nodes(), 1);
        assert_eq!(g.degree(0), 0);
    }

    #[test]
    fn test_csr_self_loop() {
        let g = CsrGraph::from_edges(2, vec![(0, 0, 1.0), (0, 1, 2.0)], true).expect("build");
        assert_eq!(g.degree(0), 2);
        assert!(g.has_edge(0, 0));
        assert!(g.has_edge(0, 1));
    }

    #[test]
    fn test_csr_memory_bytes() {
        let g = CsrGraph::from_edges(100, vec![(0, 1, 1.0)], true).expect("build");
        let mem = g.memory_bytes();
        assert!(mem > 0);
        // Should be at least the row_ptr size
        assert!(mem >= 101 * std::mem::size_of::<usize>());
    }

    #[test]
    fn test_csr_display() {
        let g = CsrGraph::from_edges(10, vec![(0, 1, 1.0), (2, 3, 1.0)], false).expect("build");
        let s = format!("{g}");
        assert!(s.contains("CsrGraph"));
        assert!(s.contains("nodes=10"));
    }

    #[test]
    fn test_csr_neighbor_iter_exact_size() {
        let g = CsrGraph::from_edges(4, vec![(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)], true)
            .expect("build");
        let iter = g.neighbors(0);
        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_csr_out_of_range_node() {
        let g = CsrGraph::from_edges(3, vec![(0, 1, 1.0)], true).expect("build");
        // Out-of-range should return empty / 0
        assert_eq!(g.degree(100), 0);
        let n: Vec<_> = g.neighbors(100).collect();
        assert!(n.is_empty());
        assert!(!g.has_edge(100, 0));
        assert_eq!(g.edge_weight(100, 0), None);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_csr_parallel_build() {
        let edges: Vec<(usize, usize, f64)> = (0..100).map(|i| (i, (i + 1) % 100, 1.0)).collect();
        let g = CsrGraph::from_edges_parallel(100, edges, false).expect("parallel build");
        assert_eq!(g.num_nodes(), 100);
        assert_eq!(g.num_logical_edges(), 100);
        for i in 0..100 {
            assert!(g.has_edge(i, (i + 1) % 100));
            assert!(g.has_edge((i + 1) % 100, i));
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_csr_builder_parallel() {
        let mut builder = CsrGraphBuilder::with_capacity(10, 20, true);
        for i in 0..9 {
            builder.add_edge(i, i + 1, (i + 1) as f64).expect("add");
        }
        let g = builder.build_parallel().expect("build parallel");
        assert_eq!(g.num_nodes(), 10);
        assert_eq!(g.num_edges(), 9);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_csr_spmv_parallel() {
        let g = CsrGraph::from_edges(2, vec![(0, 1, 2.0), (1, 0, 3.0)], true).expect("build");
        let x = vec![1.0, 2.0];
        let y = g.spmv_parallel(&x).expect("spmv");
        assert!((y[0] - 4.0).abs() < 1e-10);
        assert!((y[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_unweighted_edges() {
        let edges = [(0, 1), (1, 2), (2, 3)];
        let g = CsrGraph::from_unweighted_edges(4, &edges, false).expect("build");
        assert_eq!(g.num_nodes(), 4);
        assert_eq!(g.num_logical_edges(), 3);
        assert_eq!(g.edge_weight(0, 1), Some(1.0));
    }

    #[test]
    fn test_csr_dense_graph() {
        // Complete graph K5
        let mut edges = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    edges.push((i, j, 1.0));
                }
            }
        }
        let g = CsrGraph::from_edges(5, edges, true).expect("build");
        assert_eq!(g.num_nodes(), 5);
        assert_eq!(g.num_edges(), 20); // 5*4 directed edges
        for i in 0..5 {
            assert_eq!(g.degree(i), 4);
        }
    }
}
