//! Core types for signed and directed graph learning.
//!
//! Provides data structures representing signed graphs (with +/-1 edge signs),
//! directed weighted graphs, and configuration/result types for their embeddings.

/// A signed edge: carries a sign of +1 (positive) or -1 (negative).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignedEdge {
    /// Source node index.
    pub src: usize,
    /// Destination node index.
    pub dst: usize,
    /// Edge sign: +1 for positive, -1 for negative.
    pub sign: i8,
}

impl SignedEdge {
    /// Create a new signed edge.
    pub fn new(src: usize, dst: usize, sign: i8) -> Self {
        Self { src, dst, sign }
    }

    /// Return `true` if this edge is positive.
    pub fn is_positive(&self) -> bool {
        self.sign > 0
    }
}

/// A directed weighted edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectedEdge {
    /// Source node index.
    pub src: usize,
    /// Destination node index.
    pub dst: usize,
    /// Edge weight.
    pub weight: f64,
}

impl DirectedEdge {
    /// Create a new directed edge.
    pub fn new(src: usize, dst: usize, weight: f64) -> Self {
        Self { src, dst, weight }
    }
}

/// A signed (undirected) graph with separate positive and negative adjacency lists.
#[derive(Debug, Clone)]
pub struct SignedGraph {
    /// Number of nodes.
    pub n_nodes: usize,
    /// All edges.
    pub edges: Vec<SignedEdge>,
    /// Positive adjacency list: `pos_adj[i]` = list of (neighbor, weight=+1.0).
    pub pos_adj: Vec<Vec<usize>>,
    /// Negative adjacency list: `neg_adj[i]` = list of (neighbor, weight=+1.0).
    pub neg_adj: Vec<Vec<usize>>,
}

impl SignedGraph {
    /// Create an empty signed graph with `n_nodes` nodes.
    pub fn new(n_nodes: usize) -> Self {
        Self {
            n_nodes,
            edges: Vec::new(),
            pos_adj: vec![Vec::new(); n_nodes],
            neg_adj: vec![Vec::new(); n_nodes],
        }
    }

    /// Add a signed edge (undirected: both directions are recorded).
    pub fn add_edge(&mut self, src: usize, dst: usize, sign: i8) {
        assert!(
            src < self.n_nodes && dst < self.n_nodes,
            "node index out of range"
        );
        self.edges.push(SignedEdge { src, dst, sign });
        if sign > 0 {
            self.pos_adj[src].push(dst);
            self.pos_adj[dst].push(src);
        } else {
            self.neg_adj[src].push(dst);
            self.neg_adj[dst].push(src);
        }
    }

    /// Return the number of positive edges.
    pub fn positive_edge_count(&self) -> usize {
        self.edges.iter().filter(|e| e.sign > 0).count()
    }

    /// Return the number of negative edges.
    pub fn negative_edge_count(&self) -> usize {
        self.edges.iter().filter(|e| e.sign < 0).count()
    }

    /// Degree of node `v` in the absolute-value adjacency (all edges regardless of sign).
    pub fn abs_degree(&self, v: usize) -> usize {
        self.pos_adj[v].len() + self.neg_adj[v].len()
    }
}

/// A directed weighted graph with per-node in-edge and out-edge lists.
#[derive(Debug, Clone)]
pub struct DirectedGraph {
    /// Number of nodes.
    pub n_nodes: usize,
    /// All edges.
    pub edges: Vec<DirectedEdge>,
    /// Out-adjacency list: `out_adj[i]` = list of (dst, weight).
    pub out_adj: Vec<Vec<(usize, f64)>>,
    /// In-adjacency list: `in_adj[i]` = list of (src, weight).
    pub in_adj: Vec<Vec<(usize, f64)>>,
}

impl DirectedGraph {
    /// Create an empty directed graph with `n_nodes` nodes.
    pub fn new(n_nodes: usize) -> Self {
        Self {
            n_nodes,
            edges: Vec::new(),
            out_adj: vec![Vec::new(); n_nodes],
            in_adj: vec![Vec::new(); n_nodes],
        }
    }

    /// Add a directed weighted edge from `src` to `dst`.
    pub fn add_edge(&mut self, src: usize, dst: usize, weight: f64) {
        assert!(
            src < self.n_nodes && dst < self.n_nodes,
            "node index out of range"
        );
        self.edges.push(DirectedEdge { src, dst, weight });
        self.out_adj[src].push((dst, weight));
        self.in_adj[dst].push((src, weight));
    }

    /// Out-degree of node `v`.
    pub fn out_degree(&self, v: usize) -> usize {
        self.out_adj[v].len()
    }

    /// In-degree of node `v`.
    pub fn in_degree(&self, v: usize) -> usize {
        self.in_adj[v].len()
    }
}

/// Configuration for signed graph spectral embedding (SPONGE / ratio-cut).
#[derive(Debug, Clone)]
pub struct SignedEmbedConfig {
    /// Embedding dimension.
    pub dim: usize,
    /// Number of power-iteration steps.
    pub n_iter: usize,
    /// Learning rate (used in gradient-based optimisation stages).
    pub lr: f64,
    /// Number of negative samples (used in sampling-based objectives).
    pub neg_sample: usize,
}

impl Default for SignedEmbedConfig {
    fn default() -> Self {
        Self {
            dim: 16,
            n_iter: 100,
            lr: 0.01,
            neg_sample: 5,
        }
    }
}

/// Configuration for directed graph embedding (HOPE / APP).
#[derive(Debug, Clone)]
pub struct DirectedEmbedConfig {
    /// Embedding dimension.
    pub dim: usize,
    /// Number of power-iteration / random-walk steps.
    pub n_iter: usize,
}

impl Default for DirectedEmbedConfig {
    fn default() -> Self {
        Self {
            dim: 16,
            n_iter: 10,
        }
    }
}

/// Result of an embedding computation: n_nodes × dim matrix stored row-major.
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// Row-major embedding matrix; `embeddings[i]` is the d-dimensional vector for node i.
    pub embeddings: Vec<Vec<f64>>,
    /// Embedding dimension.
    pub dim: usize,
    /// Number of nodes.
    pub n_nodes: usize,
}

impl EmbeddingResult {
    /// Construct a new zero-initialised result.
    pub fn zeros(n_nodes: usize, dim: usize) -> Self {
        Self {
            embeddings: vec![vec![0.0_f64; dim]; n_nodes],
            dim,
            n_nodes,
        }
    }

    /// Return the embedding vector for node `v`.
    pub fn get(&self, v: usize) -> &[f64] {
        &self.embeddings[v]
    }
}
