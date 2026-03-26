//! Community detection algorithms for graph/network data.
//!
//! This module provides advanced community detection methods beyond the basic
//! implementations in the `graph` module. It includes:
//!
//! - **Leiden Algorithm**: Refinement of Louvain guaranteeing well-connected communities
//! - **Label Propagation**: Near-linear time community detection with seed label support
//! - **Stochastic Block Model (SBM)**: Generative model-based community inference

pub mod label_propagation;
pub mod leiden;
pub mod sbm;

use serde::{Deserialize, Serialize};

/// Common result type for community detection algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityResult {
    /// Community label assignment for each node (0-indexed, consecutive).
    pub labels: Vec<usize>,
    /// Number of distinct communities found.
    pub num_communities: usize,
    /// Quality score of the partition (e.g. modularity). May be `None` for
    /// algorithms that do not compute a quality function.
    pub quality_score: Option<f64>,
}

/// Adjacency representation used by the community module.
///
/// Uses `f64` edge weights directly, avoiding the `Eq + Hash` constraint
/// problem present in the generic `Graph<F>` type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjacencyGraph {
    /// Number of nodes.
    pub n_nodes: usize,
    /// Adjacency list: `adjacency[u]` contains `(v, weight)` pairs.
    /// The graph is assumed undirected; every edge appears in both directions.
    pub adjacency: Vec<Vec<(usize, f64)>>,
}

impl AdjacencyGraph {
    /// Create an empty graph with `n` nodes.
    pub fn new(n: usize) -> Self {
        Self {
            n_nodes: n,
            adjacency: vec![Vec::new(); n],
        }
    }

    /// Build from a dense adjacency matrix (row-major, n x n).
    /// Only positive off-diagonal entries are treated as edges.
    pub fn from_dense(matrix: &[f64], n: usize) -> crate::error::Result<Self> {
        if matrix.len() != n * n {
            return Err(crate::error::ClusteringError::InvalidInput(
                "Matrix length must equal n*n".to_string(),
            ));
        }
        let mut g = Self::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                let w = matrix[i * n + j];
                if w > 0.0 {
                    g.adjacency[i].push((j, w));
                    g.adjacency[j].push((i, w));
                }
            }
        }
        Ok(g)
    }

    /// Build from a symmetric adjacency matrix stored as nested `Vec`.
    pub fn from_nested(adj: &[Vec<f64>]) -> crate::error::Result<Self> {
        let n = adj.len();
        for row in adj {
            if row.len() != n {
                return Err(crate::error::ClusteringError::InvalidInput(
                    "Adjacency matrix must be square".to_string(),
                ));
            }
        }
        let mut g = Self::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                let w = adj[i][j];
                if w > 0.0 {
                    g.adjacency[i].push((j, w));
                    g.adjacency[j].push((i, w));
                }
            }
        }
        Ok(g)
    }

    /// Add an undirected edge. Does NOT check for duplicates.
    pub fn add_edge(&mut self, u: usize, v: usize, weight: f64) -> crate::error::Result<()> {
        if u >= self.n_nodes || v >= self.n_nodes {
            return Err(crate::error::ClusteringError::InvalidInput(
                "Node index out of bounds".to_string(),
            ));
        }
        if u != v {
            self.adjacency[u].push((v, weight));
            self.adjacency[v].push((u, weight));
        }
        Ok(())
    }

    /// Weighted degree of a node.
    pub fn weighted_degree(&self, node: usize) -> f64 {
        self.adjacency
            .get(node)
            .map(|nbrs| nbrs.iter().map(|(_, w)| *w).sum())
            .unwrap_or(0.0)
    }

    /// Total edge weight (each undirected edge counted once).
    pub fn total_edge_weight(&self) -> f64 {
        let sum: f64 = self
            .adjacency
            .iter()
            .flat_map(|nbrs| nbrs.iter().map(|(_, w)| *w))
            .sum();
        sum / 2.0
    }

    /// Get edge weight between two nodes (0 if no edge).
    pub fn edge_weight(&self, u: usize, v: usize) -> f64 {
        if let Some(nbrs) = self.adjacency.get(u) {
            for &(nb, w) in nbrs {
                if nb == v {
                    return w;
                }
            }
        }
        0.0
    }

    /// Compute modularity for a given partition.
    ///
    /// Q = (1/2m) sum_{ij} [ A_{ij} - k_i k_j / (2m) ] delta(c_i, c_j)
    pub fn modularity(&self, labels: &[usize]) -> f64 {
        let m2 = self.total_edge_weight() * 2.0; // 2m
        if m2 == 0.0 {
            return 0.0;
        }
        let mut q = 0.0;
        for i in 0..self.n_nodes {
            let ki = self.weighted_degree(i);
            for &(j, w) in &self.adjacency[i] {
                if labels[i] == labels[j] {
                    q += w - ki * self.weighted_degree(j) / m2;
                }
            }
        }
        q / m2
    }
}

// Re-exports
pub use label_propagation::{
    label_propagation_community, LabelPropagationConfig, LabelPropagationResult,
};
pub use leiden::{leiden, LeidenConfig, QualityFunction};
pub use sbm::{StochasticBlockModel, StochasticBlockModelConfig};
