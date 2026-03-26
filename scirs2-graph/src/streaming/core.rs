//! Streaming graph processing for dynamic and large-scale graphs
//!
//! This module provides data structures and algorithms for processing graph
//! streams where edges arrive (and optionally depart) over time. It supports:
//!
//! - **Edge stream processing**: Incremental edge additions and deletions
//! - **Approximate degree distribution**: Maintained incrementally
//! - **Streaming triangle counting**: Doulion-style sampling and MASCOT-style
//!   edge sampling for approximate triangle counts
//! - **Sliding window model**: Maintain a graph over the most recent W edges
//! - **Memory-bounded processing**: Configurable memory limits with eviction
//!
//! # Design
//!
//! The streaming model assumes edges arrive one at a time. Algorithms maintain
//! approximate statistics without storing the entire graph, making them suitable
//! for graphs that do not fit in memory.

use crate::compressed::CsrGraph;
use crate::error::{GraphError, Result};
use scirs2_core::random::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

// ────────────────────────────────────────────────────────────────────────────
// StreamEdge
// ────────────────────────────────────────────────────────────────────────────

/// A single edge in a graph stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamEdge {
    /// Source node
    pub src: usize,
    /// Destination node
    pub dst: usize,
    /// Edge weight
    pub weight: f64,
    /// Timestamp (monotonically increasing)
    pub timestamp: u64,
}

/// Type of stream operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamOp {
    /// Add an edge
    Insert,
    /// Remove an edge
    Delete,
}

/// A stream event: an edge with an operation type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamEvent {
    /// The edge
    pub edge: StreamEdge,
    /// Whether this is an insertion or deletion
    pub op: StreamOp,
}

// ────────────────────────────────────────────────────────────────────────────
// StreamingGraph
// ────────────────────────────────────────────────────────────────────────────

/// A streaming graph that supports incremental edge additions and deletions.
///
/// Maintains an adjacency set representation that is updated incrementally.
/// Tracks basic statistics like degree distribution, edge count, and node count.
#[derive(Debug)]
pub struct StreamingGraph {
    /// Adjacency sets: node -> set of (neighbor, weight)
    adjacency: HashMap<usize, HashMap<usize, f64>>,
    /// Total number of edges currently in the graph
    num_edges: usize,
    /// Number of stream events processed
    events_processed: u64,
    /// Whether the graph is directed
    directed: bool,
    /// Maximum node ID seen
    max_node_id: usize,
}

impl StreamingGraph {
    /// Create a new empty streaming graph.
    pub fn new(directed: bool) -> Self {
        Self {
            adjacency: HashMap::new(),
            num_edges: 0,
            events_processed: 0,
            directed,
            max_node_id: 0,
        }
    }

    /// Process a stream event (insert or delete an edge).
    pub fn process_event(&mut self, event: &StreamEvent) {
        self.events_processed += 1;
        match event.op {
            StreamOp::Insert => self.insert_edge(event.edge.src, event.edge.dst, event.edge.weight),
            StreamOp::Delete => self.delete_edge(event.edge.src, event.edge.dst),
        }
    }

    /// Insert an edge into the graph.
    pub fn insert_edge(&mut self, src: usize, dst: usize, weight: f64) {
        self.max_node_id = self.max_node_id.max(src).max(dst);

        self.adjacency.entry(src).or_default().insert(dst, weight);

        if !self.directed {
            self.adjacency.entry(dst).or_default().insert(src, weight);
        }
        self.num_edges += 1;
    }

    /// Delete an edge from the graph.
    pub fn delete_edge(&mut self, src: usize, dst: usize) {
        if let Some(neighbors) = self.adjacency.get_mut(&src) {
            if neighbors.remove(&dst).is_some() {
                self.num_edges = self.num_edges.saturating_sub(1);
            }
        }
        if !self.directed {
            if let Some(neighbors) = self.adjacency.get_mut(&dst) {
                neighbors.remove(&src);
            }
        }
    }

    /// Check if an edge exists.
    pub fn has_edge(&self, src: usize, dst: usize) -> bool {
        self.adjacency
            .get(&src)
            .is_some_and(|neighbors| neighbors.contains_key(&dst))
    }

    /// Get the degree of a node.
    pub fn degree(&self, node: usize) -> usize {
        self.adjacency
            .get(&node)
            .map_or(0, |neighbors| neighbors.len())
    }

    /// Get the number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.adjacency.len()
    }

    /// Get the number of edges.
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Get the number of events processed.
    pub fn events_processed(&self) -> u64 {
        self.events_processed
    }

    /// Get neighbors of a node.
    pub fn neighbors(&self, node: usize) -> Vec<(usize, f64)> {
        self.adjacency
            .get(&node)
            .map_or_else(Vec::new, |neighbors| {
                neighbors.iter().map(|(&n, &w)| (n, w)).collect()
            })
    }

    /// Get the degree distribution as a histogram.
    pub fn degree_distribution(&self) -> DegreeDistribution {
        let mut dist = HashMap::new();
        let mut max_degree = 0;
        let mut total_degree = 0;

        for neighbors in self.adjacency.values() {
            let deg = neighbors.len();
            *dist.entry(deg).or_insert(0usize) += 1;
            max_degree = max_degree.max(deg);
            total_degree += deg;
        }

        let n = self.adjacency.len();
        let avg_degree = if n > 0 {
            total_degree as f64 / n as f64
        } else {
            0.0
        };

        DegreeDistribution {
            histogram: dist,
            max_degree,
            avg_degree,
            num_nodes: n,
        }
    }

    /// Snapshot the current graph state as a CSR graph.
    pub fn to_csr(&self) -> Result<CsrGraph> {
        let num_nodes = if self.adjacency.is_empty() {
            0
        } else {
            self.max_node_id + 1
        };

        let mut edges = Vec::with_capacity(self.num_edges);
        for (&src, neighbors) in &self.adjacency {
            for (&dst, &weight) in neighbors {
                if self.directed || src <= dst {
                    edges.push((src, dst, weight));
                }
            }
        }

        CsrGraph::from_edges(num_nodes, edges, self.directed)
    }
}

/// Degree distribution statistics.
#[derive(Debug, Clone)]
pub struct DegreeDistribution {
    /// Histogram: degree -> count
    pub histogram: HashMap<usize, usize>,
    /// Maximum degree observed
    pub max_degree: usize,
    /// Average degree
    pub avg_degree: f64,
    /// Number of nodes
    pub num_nodes: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// Streaming Triangle Counter (Doulion-style)
// ────────────────────────────────────────────────────────────────────────────

/// Approximate streaming triangle counter using edge sampling (Doulion algorithm).
///
/// The Doulion algorithm samples each edge with probability `p` and counts
/// triangles in the sampled subgraph. The triangle count is then scaled by `1/p^3`
/// to estimate the total.
///
/// # Reference
/// Tsourakakis et al., "Doulion: Counting Triangles in Massive Graphs with
/// a Coin", KDD 2009.
#[derive(Debug)]
pub struct DoulionTriangleCounter {
    /// Sampling probability
    sample_prob: f64,
    /// Sampled edges as adjacency sets
    sampled_adj: HashMap<usize, HashSet<usize>>,
    /// Number of triangles found in sampled subgraph
    sampled_triangles: usize,
    /// Number of edges processed
    edges_processed: u64,
    /// Number of edges sampled
    edges_sampled: u64,
    /// RNG for sampling
    rng: StdRng,
}

impl DoulionTriangleCounter {
    /// Create a new Doulion triangle counter with given sampling probability.
    ///
    /// Lower `sample_prob` uses less memory but gives less accurate estimates.
    /// A value of 0.1 is reasonable for graphs with millions of edges.
    pub fn new(sample_prob: f64, seed: u64) -> Result<Self> {
        if !(0.0..=1.0).contains(&sample_prob) {
            return Err(GraphError::InvalidGraph(
                "sample_prob must be in [0, 1]".to_string(),
            ));
        }
        Ok(Self {
            sample_prob,
            sampled_adj: HashMap::new(),
            sampled_triangles: 0,
            edges_processed: 0,
            edges_sampled: 0,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Process a new edge from the stream.
    ///
    /// With probability `p`, the edge is sampled. If both endpoints are already
    /// in the sample, we check for triangles formed.
    pub fn process_edge(&mut self, src: usize, dst: usize) {
        self.edges_processed += 1;

        // Sample this edge with probability p
        if self.rng.random::<f64>() >= self.sample_prob {
            return;
        }

        self.edges_sampled += 1;

        // Before adding the edge, count new triangles formed
        // A triangle is formed if there exists a node w such that
        // w is a neighbor of both src and dst in the sampled graph
        let neighbors_src: HashSet<usize> = self.sampled_adj.get(&src).cloned().unwrap_or_default();
        let neighbors_dst: HashSet<usize> = self.sampled_adj.get(&dst).cloned().unwrap_or_default();

        // Count common neighbors
        let common = neighbors_src.intersection(&neighbors_dst).count();
        self.sampled_triangles += common;

        // Add the edge to the sample
        self.sampled_adj.entry(src).or_default().insert(dst);
        self.sampled_adj.entry(dst).or_default().insert(src);
    }

    /// Get the estimated total number of triangles.
    pub fn estimated_triangles(&self) -> f64 {
        if self.sample_prob <= 0.0 {
            return 0.0;
        }
        // Scale by 1/p^3
        let p3 = self.sample_prob * self.sample_prob * self.sample_prob;
        self.sampled_triangles as f64 / p3
    }

    /// Get the number of sampled triangles (unscaled).
    pub fn sampled_triangles(&self) -> usize {
        self.sampled_triangles
    }

    /// Get statistics about the counter.
    pub fn stats(&self) -> TriangleCounterStats {
        TriangleCounterStats {
            edges_processed: self.edges_processed,
            edges_sampled: self.edges_sampled,
            sampled_triangles: self.sampled_triangles,
            estimated_triangles: self.estimated_triangles(),
            sample_prob: self.sample_prob,
            memory_nodes: self.sampled_adj.len(),
        }
    }
}

/// Statistics from a streaming triangle counter.
#[derive(Debug, Clone)]
pub struct TriangleCounterStats {
    /// Total edges processed from the stream
    pub edges_processed: u64,
    /// Number of edges retained in the sample
    pub edges_sampled: u64,
    /// Triangles found in the sample
    pub sampled_triangles: usize,
    /// Estimated total triangles (scaled)
    pub estimated_triangles: f64,
    /// Sampling probability used
    pub sample_prob: f64,
    /// Number of nodes stored in memory
    pub memory_nodes: usize,
}

// ────────────────────────────────────────────────────────────────────────────
// MASCOT-style Triangle Counter
// ────────────────────────────────────────────────────────────────────────────

/// MASCOT (Memory-efficient Accurate Sampling for Counting Local Triangles)
/// streaming triangle counter.
///
/// Maintains a fixed-size edge reservoir sample and updates triangle counts
/// as new edges arrive. More memory-efficient than Doulion for fixed memory budgets.
///
/// # Reference
/// Lim & Kang, "MASCOT: Memory-efficient and Accurate Sampling for Counting
/// Local Triangles in Graph Streams", KDD 2015.
#[derive(Debug)]
pub struct MascotTriangleCounter {
    /// Maximum number of edges to store
    max_edges: usize,
    /// Current edge reservoir
    edges: Vec<(usize, usize)>,
    /// Adjacency sets for quick triangle checks
    adj: HashMap<usize, HashSet<usize>>,
    /// Global triangle count estimate (with scaling)
    triangle_estimate: f64,
    /// Number of edges processed
    edges_processed: u64,
    /// RNG for reservoir sampling
    rng: StdRng,
}

impl MascotTriangleCounter {
    /// Create a new MASCOT counter with a fixed edge budget.
    pub fn new(max_edges: usize, seed: u64) -> Self {
        Self {
            max_edges,
            edges: Vec::with_capacity(max_edges),
            adj: HashMap::new(),
            triangle_estimate: 0.0,
            edges_processed: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Process a new edge from the stream.
    pub fn process_edge(&mut self, src: usize, dst: usize) {
        self.edges_processed += 1;

        // Count triangles formed with current sample
        let neighbors_src: Vec<usize> = self
            .adj
            .get(&src)
            .map_or_else(Vec::new, |s| s.iter().copied().collect());
        let neighbors_dst: HashSet<usize> = self.adj.get(&dst).cloned().unwrap_or_default();

        let common_count = neighbors_src
            .iter()
            .filter(|w| neighbors_dst.contains(w))
            .count();

        // Scale factor: probability that both other edges are in the sample
        let t = self.edges_processed;
        let m = self.max_edges;
        if t <= m as u64 {
            // All edges are stored, no scaling needed
            self.triangle_estimate += common_count as f64;
        } else {
            // Reservoir sampling probability for an edge to be in sample
            let p = m as f64 / t as f64;
            // Both other edges of triangle must be in sample: scale by 1/p^2
            if p > 0.0 {
                self.triangle_estimate += common_count as f64 / (p * p);
            }
        }

        // Reservoir sampling: decide whether to include this edge
        if self.edges.len() < self.max_edges {
            // Sample not full yet, always include
            self.edges.push((src, dst));
            self.adj.entry(src).or_default().insert(dst);
            self.adj.entry(dst).or_default().insert(src);
        } else {
            // Replace a random edge with probability max_edges / edges_processed
            let j = self.rng.random_range(0..self.edges_processed as usize);
            if j < self.max_edges {
                // Remove old edge
                let (old_src, old_dst) = self.edges[j];
                if let Some(set) = self.adj.get_mut(&old_src) {
                    set.remove(&old_dst);
                }
                if let Some(set) = self.adj.get_mut(&old_dst) {
                    set.remove(&old_src);
                }

                // Insert new edge
                self.edges[j] = (src, dst);
                self.adj.entry(src).or_default().insert(dst);
                self.adj.entry(dst).or_default().insert(src);
            }
        }
    }

    /// Get the estimated triangle count.
    pub fn estimated_triangles(&self) -> f64 {
        self.triangle_estimate
    }

    /// Get counter statistics.
    pub fn stats(&self) -> TriangleCounterStats {
        TriangleCounterStats {
            edges_processed: self.edges_processed,
            edges_sampled: self.edges.len() as u64,
            sampled_triangles: 0, // MASCOT tracks scaled estimate directly
            estimated_triangles: self.triangle_estimate,
            sample_prob: if self.edges_processed > 0 {
                self.edges.len() as f64 / self.edges_processed as f64
            } else {
                1.0
            },
            memory_nodes: self.adj.len(),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Sliding Window Graph
// ────────────────────────────────────────────────────────────────────────────

/// A sliding window graph that maintains only the most recent `W` edges.
///
/// As new edges arrive, old edges beyond the window are automatically evicted.
/// This is useful for maintaining a graph over a time-bounded stream.
#[derive(Debug)]
pub struct SlidingWindowGraph {
    /// Window size (maximum number of edges to retain)
    window_size: usize,
    /// Ordered queue of edges (front = oldest)
    edge_queue: VecDeque<(usize, usize, f64)>,
    /// Current adjacency representation
    adjacency: HashMap<usize, HashMap<usize, f64>>,
    /// Edge count for each (src, dst) pair to handle multi-edges
    edge_counts: HashMap<(usize, usize), usize>,
    /// Whether the graph is directed
    directed: bool,
    /// Total events processed
    events_processed: u64,
}

impl SlidingWindowGraph {
    /// Create a new sliding window graph.
    pub fn new(window_size: usize, directed: bool) -> Self {
        Self {
            window_size,
            edge_queue: VecDeque::with_capacity(window_size),
            adjacency: HashMap::new(),
            edge_counts: HashMap::new(),
            directed,
            events_processed: 0,
        }
    }

    /// Process a new edge. If the window is full, the oldest edge is evicted.
    pub fn process_edge(&mut self, src: usize, dst: usize, weight: f64) {
        self.events_processed += 1;

        // Evict oldest edge if window is full
        if self.edge_queue.len() >= self.window_size {
            if let Some((old_src, old_dst, _old_weight)) = self.edge_queue.pop_front() {
                self.remove_edge_internal(old_src, old_dst);
            }
        }

        // Add new edge
        self.edge_queue.push_back((src, dst, weight));
        self.add_edge_internal(src, dst, weight);
    }

    fn add_edge_internal(&mut self, src: usize, dst: usize, weight: f64) {
        self.adjacency.entry(src).or_default().insert(dst, weight);
        *self.edge_counts.entry((src, dst)).or_insert(0) += 1;

        if !self.directed {
            self.adjacency.entry(dst).or_default().insert(src, weight);
            *self.edge_counts.entry((dst, src)).or_insert(0) += 1;
        }
    }

    fn remove_edge_internal(&mut self, src: usize, dst: usize) {
        let key = (src, dst);
        if let Some(count) = self.edge_counts.get_mut(&key) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.edge_counts.remove(&key);
                if let Some(neighbors) = self.adjacency.get_mut(&src) {
                    neighbors.remove(&dst);
                    if neighbors.is_empty() {
                        self.adjacency.remove(&src);
                    }
                }
            }
        }

        if !self.directed {
            let rev_key = (dst, src);
            if let Some(count) = self.edge_counts.get_mut(&rev_key) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.edge_counts.remove(&rev_key);
                    if let Some(neighbors) = self.adjacency.get_mut(&dst) {
                        neighbors.remove(&src);
                        if neighbors.is_empty() {
                            self.adjacency.remove(&dst);
                        }
                    }
                }
            }
        }
    }

    /// Get the current number of edges in the window.
    pub fn num_edges(&self) -> usize {
        self.edge_queue.len()
    }

    /// Get the current number of active nodes.
    pub fn num_nodes(&self) -> usize {
        self.adjacency.len()
    }

    /// Get the window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get neighbors of a node in the current window.
    pub fn neighbors(&self, node: usize) -> Vec<(usize, f64)> {
        self.adjacency
            .get(&node)
            .map_or_else(Vec::new, |neighbors| {
                neighbors.iter().map(|(&n, &w)| (n, w)).collect()
            })
    }

    /// Get the degree of a node.
    pub fn degree(&self, node: usize) -> usize {
        self.adjacency
            .get(&node)
            .map_or(0, |neighbors| neighbors.len())
    }

    /// Check if an edge exists in the current window.
    pub fn has_edge(&self, src: usize, dst: usize) -> bool {
        self.adjacency
            .get(&src)
            .is_some_and(|n| n.contains_key(&dst))
    }

    /// Get total events processed.
    pub fn events_processed(&self) -> u64 {
        self.events_processed
    }

    /// Take a snapshot of the current window as a CSR graph.
    pub fn to_csr(&self) -> Result<CsrGraph> {
        let max_node = self
            .adjacency
            .keys()
            .chain(self.adjacency.values().flat_map(|n| n.keys()))
            .copied()
            .max()
            .map_or(0, |m| m + 1);

        let mut edges = Vec::new();
        for (&src, neighbors) in &self.adjacency {
            for (&dst, &weight) in neighbors {
                if self.directed || src <= dst {
                    edges.push((src, dst, weight));
                }
            }
        }

        CsrGraph::from_edges(max_node, edges, self.directed)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Memory-Bounded Stream Processor
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for memory-bounded stream processing.
#[derive(Debug, Clone)]
pub struct MemoryBoundedConfig {
    /// Maximum memory budget in bytes
    pub max_memory_bytes: usize,
    /// Eviction strategy when memory is exceeded
    pub eviction_strategy: EvictionStrategy,
    /// Whether to track degree distribution
    pub track_degrees: bool,
    /// Whether to count triangles
    pub count_triangles: bool,
    /// Triangle counting sample probability
    pub triangle_sample_prob: f64,
}

impl Default for MemoryBoundedConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 100 * 1024 * 1024, // 100 MB
            eviction_strategy: EvictionStrategy::LeastRecentEdge,
            track_degrees: true,
            count_triangles: false,
            triangle_sample_prob: 0.1,
        }
    }
}

/// Strategy for evicting data when memory budget is exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionStrategy {
    /// Remove the oldest edges first (FIFO)
    LeastRecentEdge,
    /// Remove edges from the lowest-degree nodes first
    LowestDegreeNode,
    /// Random edge removal
    RandomEdge,
}

/// A memory-bounded stream processor that enforces a memory budget.
///
/// Processes edge events and maintains approximate graph statistics
/// while staying within a configurable memory limit.
#[derive(Debug)]
pub struct MemoryBoundedProcessor {
    /// Configuration
    config: MemoryBoundedConfig,
    /// The underlying streaming graph
    graph: StreamingGraph,
    /// Edge insertion order (for LeastRecentEdge eviction)
    insertion_order: VecDeque<(usize, usize)>,
    /// Approximate memory usage in bytes
    estimated_memory: usize,
    /// Edges evicted
    edges_evicted: u64,
    /// RNG for random eviction
    rng: StdRng,
}

impl MemoryBoundedProcessor {
    /// Create a new memory-bounded processor.
    pub fn new(config: MemoryBoundedConfig) -> Self {
        Self {
            graph: StreamingGraph::new(false),
            insertion_order: VecDeque::new(),
            estimated_memory: 0,
            edges_evicted: 0,
            rng: StdRng::seed_from_u64(42),
            config,
        }
    }

    /// Process an edge event, evicting old edges if memory budget is exceeded.
    pub fn process_edge(&mut self, src: usize, dst: usize, weight: f64) {
        // Estimate memory for this edge (~80 bytes for HashMap entries + overhead)
        let edge_memory_estimate = 80;

        // Evict edges if over budget
        while self.estimated_memory + edge_memory_estimate > self.config.max_memory_bytes
            && !self.insertion_order.is_empty()
        {
            self.evict_one();
        }

        // Insert the new edge
        self.graph.insert_edge(src, dst, weight);
        self.insertion_order.push_back((src, dst));
        self.estimated_memory += edge_memory_estimate;
    }

    fn evict_one(&mut self) {
        match self.config.eviction_strategy {
            EvictionStrategy::LeastRecentEdge => {
                if let Some((src, dst)) = self.insertion_order.pop_front() {
                    self.graph.delete_edge(src, dst);
                    self.estimated_memory = self.estimated_memory.saturating_sub(80);
                    self.edges_evicted += 1;
                }
            }
            EvictionStrategy::LowestDegreeNode => {
                // Find the node with the lowest degree
                if let Some((&node, _)) = self
                    .graph
                    .adjacency
                    .iter()
                    .min_by_key(|(_, neighbors)| neighbors.len())
                {
                    let neighbors: Vec<usize> = self
                        .graph
                        .adjacency
                        .get(&node)
                        .map_or_else(Vec::new, |n| n.keys().copied().collect());
                    for neighbor in neighbors {
                        self.graph.delete_edge(node, neighbor);
                        self.estimated_memory = self.estimated_memory.saturating_sub(80);
                        self.edges_evicted += 1;
                    }
                }
            }
            EvictionStrategy::RandomEdge => {
                if !self.insertion_order.is_empty() {
                    let idx = self.rng.random_range(0..self.insertion_order.len());
                    if let Some((src, dst)) = self.insertion_order.remove(idx) {
                        self.graph.delete_edge(src, dst);
                        self.estimated_memory = self.estimated_memory.saturating_sub(80);
                        self.edges_evicted += 1;
                    }
                }
            }
        }
    }

    /// Get the current streaming graph.
    pub fn graph(&self) -> &StreamingGraph {
        &self.graph
    }

    /// Get the number of edges evicted.
    pub fn edges_evicted(&self) -> u64 {
        self.edges_evicted
    }

    /// Get estimated memory usage in bytes.
    pub fn estimated_memory(&self) -> usize {
        self.estimated_memory
    }

    /// Get the degree distribution of the current graph.
    pub fn degree_distribution(&self) -> DegreeDistribution {
        self.graph.degree_distribution()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── StreamingGraph Tests ──

    #[test]
    fn test_streaming_graph_insert() {
        let mut g = StreamingGraph::new(false);
        g.insert_edge(0, 1, 1.0);
        g.insert_edge(1, 2, 2.0);

        assert_eq!(g.num_edges(), 2);
        assert_eq!(g.num_nodes(), 3);
        assert!(g.has_edge(0, 1));
        assert!(g.has_edge(1, 0)); // undirected
        assert!(g.has_edge(1, 2));
    }

    #[test]
    fn test_streaming_graph_delete() {
        let mut g = StreamingGraph::new(false);
        g.insert_edge(0, 1, 1.0);
        g.insert_edge(1, 2, 2.0);
        g.delete_edge(0, 1);

        assert_eq!(g.num_edges(), 1);
        assert!(!g.has_edge(0, 1));
        assert!(!g.has_edge(1, 0)); // undirected deletion
        assert!(g.has_edge(1, 2));
    }

    #[test]
    fn test_streaming_graph_directed() {
        let mut g = StreamingGraph::new(true);
        g.insert_edge(0, 1, 1.0);

        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(1, 0)); // directed
    }

    #[test]
    fn test_streaming_graph_process_event() {
        let mut g = StreamingGraph::new(false);
        let event = StreamEvent {
            edge: StreamEdge {
                src: 0,
                dst: 1,
                weight: 1.0,
                timestamp: 0,
            },
            op: StreamOp::Insert,
        };
        g.process_event(&event);
        assert_eq!(g.num_edges(), 1);
        assert_eq!(g.events_processed(), 1);

        let del_event = StreamEvent {
            edge: StreamEdge {
                src: 0,
                dst: 1,
                weight: 1.0,
                timestamp: 1,
            },
            op: StreamOp::Delete,
        };
        g.process_event(&del_event);
        assert_eq!(g.num_edges(), 0);
        assert_eq!(g.events_processed(), 2);
    }

    #[test]
    fn test_streaming_graph_degree() {
        let mut g = StreamingGraph::new(false);
        g.insert_edge(0, 1, 1.0);
        g.insert_edge(0, 2, 1.0);
        g.insert_edge(0, 3, 1.0);

        assert_eq!(g.degree(0), 3);
        assert_eq!(g.degree(1), 1);
        assert_eq!(g.degree(4), 0); // non-existent node
    }

    #[test]
    fn test_streaming_graph_neighbors() {
        let mut g = StreamingGraph::new(false);
        g.insert_edge(0, 1, 1.0);
        g.insert_edge(0, 2, 2.0);

        let mut neighbors = g.neighbors(0);
        neighbors.sort_by_key(|&(n, _)| n);
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].0, 1);
        assert_eq!(neighbors[1].0, 2);
    }

    #[test]
    fn test_streaming_graph_degree_distribution() {
        let mut g = StreamingGraph::new(false);
        // Star graph: center=0, spokes=1,2,3,4
        g.insert_edge(0, 1, 1.0);
        g.insert_edge(0, 2, 1.0);
        g.insert_edge(0, 3, 1.0);
        g.insert_edge(0, 4, 1.0);

        let dist = g.degree_distribution();
        assert_eq!(dist.max_degree, 4);
        assert_eq!(dist.num_nodes, 5);
        // Node 0 has degree 4, nodes 1-4 have degree 1
        assert_eq!(dist.histogram.get(&4), Some(&1));
        assert_eq!(dist.histogram.get(&1), Some(&4));
    }

    #[test]
    fn test_streaming_graph_to_csr() {
        let mut g = StreamingGraph::new(false);
        g.insert_edge(0, 1, 1.0);
        g.insert_edge(1, 2, 2.0);

        let csr = g.to_csr().expect("to_csr");
        assert_eq!(csr.num_nodes(), 3);
        assert!(csr.has_edge(0, 1));
        assert!(csr.has_edge(1, 0));
        assert!(csr.has_edge(1, 2));
    }

    // ── Doulion Triangle Counter Tests ──

    #[test]
    fn test_doulion_basic() {
        // Complete graph K4 has 4 triangles
        let mut counter = DoulionTriangleCounter::new(1.0, 42).expect("new");

        // Add all edges of K4
        counter.process_edge(0, 1);
        counter.process_edge(0, 2);
        counter.process_edge(0, 3);
        counter.process_edge(1, 2);
        counter.process_edge(1, 3);
        counter.process_edge(2, 3);

        // With p=1.0, the estimate should be exact
        let est = counter.estimated_triangles();
        assert!((est - 4.0).abs() < 1e-6, "expected 4 triangles, got {est}");

        let stats = counter.stats();
        assert_eq!(stats.edges_processed, 6);
        assert_eq!(stats.edges_sampled, 6);
    }

    #[test]
    fn test_doulion_no_triangles() {
        let mut counter = DoulionTriangleCounter::new(1.0, 42).expect("new");

        // Path graph: no triangles
        counter.process_edge(0, 1);
        counter.process_edge(1, 2);
        counter.process_edge(2, 3);

        assert!(counter.estimated_triangles().abs() < 1e-6);
    }

    #[test]
    fn test_doulion_sampling() {
        // With p=0.5, the estimate may differ from exact
        let mut counter = DoulionTriangleCounter::new(0.5, 42).expect("new");

        // K4 edges
        counter.process_edge(0, 1);
        counter.process_edge(0, 2);
        counter.process_edge(0, 3);
        counter.process_edge(1, 2);
        counter.process_edge(1, 3);
        counter.process_edge(2, 3);

        // The estimate should be non-negative
        assert!(counter.estimated_triangles() >= 0.0);
    }

    #[test]
    fn test_doulion_invalid_prob() {
        assert!(DoulionTriangleCounter::new(1.5, 42).is_err());
        assert!(DoulionTriangleCounter::new(-0.1, 42).is_err());
    }

    // ── MASCOT Triangle Counter Tests ──

    #[test]
    fn test_mascot_basic() {
        let mut counter = MascotTriangleCounter::new(100, 42);

        // K4 edges
        counter.process_edge(0, 1);
        counter.process_edge(0, 2);
        counter.process_edge(0, 3);
        counter.process_edge(1, 2);
        counter.process_edge(1, 3);
        counter.process_edge(2, 3);

        // With budget=100 and only 6 edges, all are stored -> exact count
        let est = counter.estimated_triangles();
        assert!((est - 4.0).abs() < 1e-6, "expected 4 triangles, got {est}");
    }

    #[test]
    fn test_mascot_stats() {
        let mut counter = MascotTriangleCounter::new(100, 42);
        counter.process_edge(0, 1);
        counter.process_edge(1, 2);

        let stats = counter.stats();
        assert_eq!(stats.edges_processed, 2);
        assert_eq!(stats.edges_sampled, 2);
    }

    // ── Sliding Window Tests ──

    #[test]
    fn test_sliding_window_basic() {
        let mut sw = SlidingWindowGraph::new(3, false);

        sw.process_edge(0, 1, 1.0);
        sw.process_edge(1, 2, 2.0);
        sw.process_edge(2, 3, 3.0);

        assert_eq!(sw.num_edges(), 3);
        assert!(sw.has_edge(0, 1));
        assert!(sw.has_edge(1, 2));
        assert!(sw.has_edge(2, 3));

        // Add one more: oldest (0-1) should be evicted
        sw.process_edge(3, 4, 4.0);
        assert_eq!(sw.num_edges(), 3);
        assert!(!sw.has_edge(0, 1)); // evicted
        assert!(sw.has_edge(1, 2));
        assert!(sw.has_edge(2, 3));
        assert!(sw.has_edge(3, 4));
    }

    #[test]
    fn test_sliding_window_directed() {
        let mut sw = SlidingWindowGraph::new(5, true);

        sw.process_edge(0, 1, 1.0);
        assert!(sw.has_edge(0, 1));
        assert!(!sw.has_edge(1, 0)); // directed
    }

    #[test]
    fn test_sliding_window_degree() {
        let mut sw = SlidingWindowGraph::new(10, false);
        sw.process_edge(0, 1, 1.0);
        sw.process_edge(0, 2, 1.0);
        sw.process_edge(0, 3, 1.0);

        assert_eq!(sw.degree(0), 3);
        assert_eq!(sw.degree(1), 1);
    }

    #[test]
    fn test_sliding_window_events_processed() {
        let mut sw = SlidingWindowGraph::new(2, false);
        sw.process_edge(0, 1, 1.0);
        sw.process_edge(1, 2, 1.0);
        sw.process_edge(2, 3, 1.0); // evicts 0-1

        assert_eq!(sw.events_processed(), 3);
        assert_eq!(sw.num_edges(), 2);
    }

    #[test]
    fn test_sliding_window_to_csr() {
        let mut sw = SlidingWindowGraph::new(10, false);
        sw.process_edge(0, 1, 1.0);
        sw.process_edge(1, 2, 2.0);

        let csr = sw.to_csr().expect("to_csr");
        assert_eq!(csr.num_nodes(), 3);
        assert!(csr.has_edge(0, 1));
    }

    // ── Memory-Bounded Processor Tests ──

    #[test]
    fn test_memory_bounded_basic() {
        let config = MemoryBoundedConfig {
            max_memory_bytes: 400, // Very small: ~5 edges
            eviction_strategy: EvictionStrategy::LeastRecentEdge,
            track_degrees: true,
            count_triangles: false,
            triangle_sample_prob: 0.1,
        };
        let mut proc = MemoryBoundedProcessor::new(config);

        for i in 0..20 {
            proc.process_edge(i, i + 1, 1.0);
        }

        // Some edges should have been evicted
        assert!(proc.edges_evicted() > 0);
        assert!(proc.estimated_memory() <= 400);
    }

    #[test]
    fn test_memory_bounded_degree_dist() {
        let config = MemoryBoundedConfig {
            max_memory_bytes: 10_000,
            ..Default::default()
        };
        let mut proc = MemoryBoundedProcessor::new(config);

        proc.process_edge(0, 1, 1.0);
        proc.process_edge(0, 2, 1.0);
        proc.process_edge(0, 3, 1.0);

        let dist = proc.degree_distribution();
        assert!(dist.num_nodes > 0);
    }

    #[test]
    fn test_memory_bounded_eviction_strategies() {
        for strategy in &[
            EvictionStrategy::LeastRecentEdge,
            EvictionStrategy::RandomEdge,
        ] {
            let config = MemoryBoundedConfig {
                max_memory_bytes: 200,
                eviction_strategy: *strategy,
                ..Default::default()
            };
            let mut proc = MemoryBoundedProcessor::new(config);

            for i in 0..10 {
                proc.process_edge(i, i + 1, 1.0);
            }

            // Should not crash and should evict
            assert!(proc.edges_evicted() > 0 || proc.estimated_memory() <= 200);
        }
    }

    #[test]
    fn test_streaming_graph_empty() {
        let g = StreamingGraph::new(false);
        assert_eq!(g.num_nodes(), 0);
        assert_eq!(g.num_edges(), 0);
        assert!(g.neighbors(0).is_empty());
        assert_eq!(g.degree(0), 0);
    }

    #[test]
    fn test_streaming_graph_delete_nonexistent() {
        let mut g = StreamingGraph::new(false);
        g.insert_edge(0, 1, 1.0);
        g.delete_edge(5, 6); // should not crash
        assert_eq!(g.num_edges(), 1);
    }

    #[test]
    fn test_sliding_window_single_capacity() {
        let mut sw = SlidingWindowGraph::new(1, false);
        sw.process_edge(0, 1, 1.0);
        assert_eq!(sw.num_edges(), 1);
        assert!(sw.has_edge(0, 1));

        sw.process_edge(2, 3, 2.0);
        assert_eq!(sw.num_edges(), 1);
        assert!(!sw.has_edge(0, 1)); // evicted
        assert!(sw.has_edge(2, 3));
    }
}
