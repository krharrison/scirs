//! Temporal and Dynamic Graphs
//!
//! This module provides data structures and algorithms for temporal (dynamic) graphs,
//! where edges carry continuous-time timestamps rather than discrete time intervals.
//! It implements the stream-of-interactions model commonly used in the analysis of
//! real-world contact networks, communication networks, and social interaction data.
//!
//! # Key Concepts
//!
//! - **Temporal edge**: a directed or undirected contact `(u, v, t, w)` at time `t`
//! - **Time-respecting path**: a sequence of edges whose timestamps are non-decreasing
//! - **Temporal betweenness**: how often a node lies on optimal time-respecting paths
//! - **Burstiness**: statistical irregularity of inter-event times (Goh–Barabási 2008)
//! - **Activity-driven model**: synthetic generative model (Perra et al. 2012)
//!
//! # References
//!
//! - Holme & Saramäki, "Temporal networks", Physics Reports 519(3), 2012.
//! - Goh & Barabási, "Burstiness and memory in complex systems", EPL 81(4), 2008.
//! - Perra et al., "Activity driven modeling of time-varying networks", Sci. Rep. 2012.

use crate::base::{Graph, IndexType};
use crate::error::{GraphError, Result};
use scirs2_core::random::{Rng, RngExt, SeedableRng};
use std::collections::{BinaryHeap, HashMap, HashSet};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single temporal edge with a continuous-time timestamp.
///
/// Edges are directed by convention (source → target); callers who need undirected
/// semantics should add both directions.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalEdge {
    /// Source node index (0-based)
    pub source: usize,
    /// Target node index (0-based)
    pub target: usize,
    /// Time of the interaction (any real-valued unit: seconds, days, …)
    pub timestamp: f64,
    /// Optional edge weight (e.g. call duration, message volume)
    pub weight: Option<f64>,
}

impl TemporalEdge {
    /// Create an unweighted temporal edge.
    pub fn new(source: usize, target: usize, timestamp: f64) -> Self {
        TemporalEdge {
            source,
            target,
            timestamp,
            weight: None,
        }
    }

    /// Create a weighted temporal edge.
    pub fn with_weight(source: usize, target: usize, timestamp: f64, weight: f64) -> Self {
        TemporalEdge {
            source,
            target,
            timestamp,
            weight: Some(weight),
        }
    }
}

// ---------------------------------------------------------------------------
// TemporalGraph struct
// ---------------------------------------------------------------------------

/// A temporal (dynamic) graph stored as a sorted stream of timed edge contacts.
///
/// Nodes are identified by consecutive `usize` indices `0..n_nodes`.
/// Edges are kept sorted by timestamp to enable efficient windowed queries.
///
/// ## Example
/// ```
/// use scirs2_graph::temporal_graph::{TemporalGraph, TemporalEdge};
///
/// let mut tg = TemporalGraph::new(4);
/// tg.add_edge(TemporalEdge::new(0, 1, 1.0));
/// tg.add_edge(TemporalEdge::new(1, 2, 2.0));
/// tg.add_edge(TemporalEdge::new(2, 3, 3.0));
///
/// let snap = tg.snapshot(0.5, 2.5);
/// assert_eq!(snap.node_count(), 3); // nodes 0,1,2 appear in edges
/// ```
#[derive(Debug, Clone)]
pub struct TemporalGraph {
    /// Total number of nodes (fixed at construction time)
    n_nodes: usize,
    /// Edge stream sorted by timestamp (maintained automatically)
    edges: Vec<TemporalEdge>,
    /// Whether the edges vector is currently sorted
    sorted: bool,
}

impl TemporalGraph {
    /// Create an empty temporal graph with a fixed node count.
    pub fn new(n_nodes: usize) -> Self {
        TemporalGraph {
            n_nodes,
            edges: Vec::new(),
            sorted: true,
        }
    }

    /// Number of nodes (constant after construction).
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Number of temporal edge contacts stored.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Alias for [] — returns the number of nodes in this temporal graph.
    ///
    /// Provided for compatibility with code expecting a  method.
    pub fn node_count(&self) -> usize {
        self.n_nodes
    }

    /// Return a sorted clone of all temporal edges.
    ///
    /// Unlike [](Self::edges) this method does not require a mutable
    /// reference; instead it returns an owned  sorted by
    /// timestamp.  This is slightly less efficient than the mutable version
    /// (one extra allocation + possible sort), but works with `&self` borrows.
    pub fn sorted_edges_cloned(&self) -> Vec<TemporalEdge> {
        let mut v = self.edges.clone();
        if !self.sorted {
            v.sort_by(|a, b| {
                a.timestamp
                    .partial_cmp(&b.timestamp)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        v
    }

    /// Add a temporal edge.  Marks the edge list as unsorted when the new
    /// timestamp is earlier than the last stored timestamp.
    pub fn add_edge(&mut self, edge: TemporalEdge) {
        if let Some(last) = self.edges.last() {
            if edge.timestamp < last.timestamp {
                self.sorted = false;
            }
        }
        self.edges.push(edge);
    }

    /// Ensure the internal edge list is sorted by timestamp (stable sort).
    fn ensure_sorted(&mut self) {
        if !self.sorted {
            self.edges.sort_by(|a, b| {
                a.timestamp
                    .partial_cmp(&b.timestamp)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.sorted = true;
        }
    }

    /// Return a read-only slice of all temporal edges (in sorted order).
    pub fn edges(&mut self) -> &[TemporalEdge] {
        self.ensure_sorted();
        &self.edges
    }

    /// Return an iterator over edges in the time window `[t_start, t_end)`.
    pub fn edges_in_window(&mut self, t_start: f64, t_end: f64) -> &[TemporalEdge] {
        self.ensure_sorted();
        // Binary search bounds
        let lo = self.edges.partition_point(|e| e.timestamp < t_start);
        let hi = self.edges.partition_point(|e| e.timestamp < t_end);
        &self.edges[lo..hi]
    }

    // -----------------------------------------------------------------------
    // snapshot
    // -----------------------------------------------------------------------

    /// Build a static (undirected, weighted) snapshot of the temporal graph for
    /// all contacts that fall in the half-open window `[t_start, t_end)`.
    ///
    /// Repeated contacts between the same pair of nodes accumulate their weights
    /// (or count as 1.0 each when no weight is present).
    pub fn snapshot(&mut self, t_start: f64, t_end: f64) -> Graph<usize, f64> {
        let window = self.edges_in_window(t_start, t_end);

        // Accumulate edge weights
        let mut edge_weights: HashMap<(usize, usize), f64> = HashMap::new();
        let mut active_nodes: HashSet<usize> = HashSet::new();

        for e in window {
            active_nodes.insert(e.source);
            active_nodes.insert(e.target);
            let key = (e.source.min(e.target), e.source.max(e.target));
            let w = e.weight.unwrap_or(1.0);
            *edge_weights.entry(key).or_insert(0.0) += w;
        }

        let mut graph: Graph<usize, f64> = Graph::new();
        for &n in &active_nodes {
            graph.add_node(n);
        }
        for ((u, v), w) in edge_weights {
            // ignore errors (node not found would be a logic bug; nodes added above)
            let _ = graph.add_edge(u, v, w);
        }
        graph
    }

    // -----------------------------------------------------------------------
    // temporal_neighbors
    // -----------------------------------------------------------------------

    /// Return all neighbours of `node` reachable in the time window
    /// `[t_start, t_end)`, together with the earliest contact timestamp.
    ///
    /// Both directions of each edge are considered (undirected semantics).
    pub fn temporal_neighbors(
        &mut self,
        node: usize,
        t_start: f64,
        t_end: f64,
    ) -> Vec<(usize, f64)> {
        let window = self.edges_in_window(t_start, t_end);
        let mut first_contact: HashMap<usize, f64> = HashMap::new();

        for e in window {
            if e.source == node {
                first_contact
                    .entry(e.target)
                    .and_modify(|t| *t = t.min(e.timestamp))
                    .or_insert(e.timestamp);
            } else if e.target == node {
                first_contact
                    .entry(e.source)
                    .and_modify(|t| *t = t.min(e.timestamp))
                    .or_insert(e.timestamp);
            }
        }

        first_contact.into_iter().collect()
    }

    // -----------------------------------------------------------------------
    // temporal_path
    // -----------------------------------------------------------------------

    /// Find a time-respecting path from `source` to `target` using only edges
    /// with timestamps in `[t_start, t_end)`.
    ///
    /// Uses a Dijkstra-like priority queue keyed on the earliest-arrival time
    /// at each node, guaranteeing the fastest-arrival (foremost) path.
    ///
    /// Returns `None` when no such path exists.
    pub fn temporal_path(
        &mut self,
        source: usize,
        target: usize,
        t_start: f64,
        t_end: f64,
    ) -> Option<Vec<usize>> {
        if source >= self.n_nodes || target >= self.n_nodes {
            return None;
        }
        if source == target {
            return Some(vec![source]);
        }

        self.ensure_sorted();
        let edges = &self.edges;

        // arrival_time[v] = earliest time we can arrive at v
        let mut arrival: Vec<f64> = vec![f64::INFINITY; self.n_nodes];
        arrival[source] = t_start;

        // predecessor for path reconstruction
        let mut pred: Vec<Option<usize>> = vec![None; self.n_nodes];

        // Min-heap: (arrival_time, node) — we use ordered floats via bit conversion
        // BinaryHeap is a max-heap, so we negate
        #[derive(PartialEq)]
        struct State {
            neg_arrival: ordered_float::OrderedFloat<f64>,
            node: usize,
        }
        impl Eq for State {}
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for State {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.neg_arrival
                    .cmp(&other.neg_arrival)
                    .then(self.node.cmp(&other.node))
            }
        }

        let mut heap = BinaryHeap::new();
        heap.push(State {
            neg_arrival: ordered_float::OrderedFloat(-t_start),
            node: source,
        });

        while let Some(State { neg_arrival, node }) = heap.pop() {
            let arr = -neg_arrival.0;
            if arr > arrival[node] {
                // stale entry
                continue;
            }
            if node == target {
                // Reconstruct path
                let mut path = Vec::new();
                let mut cur = target;
                loop {
                    path.push(cur);
                    match pred[cur] {
                        None => break,
                        Some(p) => cur = p,
                    }
                }
                path.reverse();
                return Some(path);
            }

            // Scan only edges at or after our current arrival time
            let lo = edges.partition_point(|e| e.timestamp < arr);
            for e in &edges[lo..] {
                if e.timestamp >= t_end {
                    break;
                }
                let neighbour = if e.source == node {
                    e.target
                } else if e.target == node {
                    e.source
                } else {
                    continue;
                };
                if e.timestamp < arrival[neighbour] {
                    arrival[neighbour] = e.timestamp;
                    pred[neighbour] = Some(node);
                    heap.push(State {
                        neg_arrival: ordered_float::OrderedFloat(-e.timestamp),
                        node: neighbour,
                    });
                }
            }
        }

        None
    }

    // -----------------------------------------------------------------------
    // temporal_betweenness
    // -----------------------------------------------------------------------

    /// Compute temporal betweenness centrality for all nodes.
    ///
    /// For each ordered pair `(s, t)` we find the set of foremost
    /// (earliest-arrival) paths using the method above and give each
    /// intermediate node `1 / |paths|` credit.
    ///
    /// Returns a vector of length `n_nodes`.
    pub fn temporal_betweenness(&mut self, t_start: f64, t_end: f64) -> Vec<f64> {
        let n = self.n_nodes;
        let mut bet = vec![0.0f64; n];

        for s in 0..n {
            for t in 0..n {
                if s == t {
                    continue;
                }
                if let Some(path) = self.temporal_path(s, t, t_start, t_end) {
                    // intermediate nodes only
                    let len = path.len();
                    if len > 2 {
                        let credit = 1.0 / (len - 2) as f64;
                        for &v in &path[1..len - 1] {
                            bet[v] += credit;
                        }
                    }
                }
            }
        }

        // Normalise by max possible (n-1)(n-2)
        let norm = (n.saturating_sub(1) * n.saturating_sub(2)) as f64;
        if norm > 0.0 {
            for b in &mut bet {
                *b /= norm;
            }
        }
        bet
    }

    // -----------------------------------------------------------------------
    // aggregate_graph
    // -----------------------------------------------------------------------

    /// Collapse the temporal graph to a static undirected weighted graph by
    /// summing contact weights over all time.
    pub fn aggregate_graph(&mut self) -> Graph<usize, f64> {
        let t_start = self
            .edges
            .iter()
            .map(|e| e.timestamp)
            .fold(f64::INFINITY, f64::min);
        let t_end = self
            .edges
            .iter()
            .map(|e| e.timestamp)
            .fold(f64::NEG_INFINITY, f64::max);

        if t_start.is_infinite() {
            // empty graph
            let mut g: Graph<usize, f64> = Graph::new();
            for i in 0..self.n_nodes {
                g.add_node(i);
            }
            return g;
        }

        // Include the last timestamp — use open upper bound slightly above it
        self.snapshot(t_start, t_end + 1.0)
    }
}

// ---------------------------------------------------------------------------
// burstiness
// ---------------------------------------------------------------------------

/// Compute the Goh–Barabási **burstiness** coefficient for a sequence of
/// event times belonging to a single node.
///
/// Given inter-event times `τ_i = t_{i+1} − t_i`, the burstiness is:
///
/// ```text
/// B = (σ − μ) / (σ + μ)
/// ```
///
/// where `μ` and `σ` are the mean and standard deviation of the inter-event
/// times.  `B ∈ (-1, 1]`:
/// - `B = 0` → Poisson (regular random)
/// - `B > 0` → bursty
/// - `B < 0` → regular / periodic
///
/// Returns `0.0` if fewer than 2 events are provided.
///
/// # Reference
/// Goh & Barabási, "Burstiness and memory in complex systems", EPL 81(4) 48002, 2008.
pub fn burstiness(node_events: &[f64]) -> f64 {
    if node_events.len() < 2 {
        return 0.0;
    }

    // Compute inter-event times
    let mut sorted = node_events.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let iet: Vec<f64> = sorted.windows(2).map(|w| w[1] - w[0]).collect();
    let n = iet.len() as f64;

    let mu = iet.iter().sum::<f64>() / n;
    if mu == 0.0 {
        return 0.0;
    }

    let variance = iet.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / n;
    let sigma = variance.sqrt();

    (sigma - mu) / (sigma + mu)
}

// ---------------------------------------------------------------------------
// activity_driven_model
// ---------------------------------------------------------------------------

/// Generate a synthetic temporal graph using the **Activity-Driven Model**
/// (Perra et al. 2012).
///
/// At each discrete step `t = 0, 1, …, n_steps - 1`:
/// 1. Each node `i` becomes *active* with probability `activity_rates[i]`.
/// 2. Each active node creates one undirected contact with a uniformly
///    chosen partner (self-loops excluded).
///
/// The resulting graph is stored as a sorted stream of `TemporalEdge` contacts
/// with `timestamp = t as f64`.
///
/// # Arguments
/// * `n_nodes`         – number of nodes
/// * `n_steps`         – number of discrete time steps
/// * `activity_rates`  – activity probability for each node (values in `[0,1]`)
/// * `rng`             – seeded random number generator
///
/// # Errors
/// Returns `GraphError::InvalidGraph` if `activity_rates.len() != n_nodes` or
/// any activity rate is outside `[0, 1]`.
pub fn activity_driven_model<R: Rng>(
    n_nodes: usize,
    n_steps: usize,
    activity_rates: &[f64],
    rng: &mut R,
) -> Result<TemporalGraph> {
    if activity_rates.len() != n_nodes {
        return Err(GraphError::InvalidGraph(format!(
            "activity_rates length {} != n_nodes {}",
            activity_rates.len(),
            n_nodes
        )));
    }
    for (i, &a) in activity_rates.iter().enumerate() {
        if !(0.0..=1.0).contains(&a) {
            return Err(GraphError::InvalidGraph(format!(
                "activity_rates[{i}] = {a} is outside [0, 1]"
            )));
        }
    }
    if n_nodes < 2 {
        return Err(GraphError::InvalidGraph(
            "need at least 2 nodes for activity-driven model".to_string(),
        ));
    }

    let mut tg = TemporalGraph::new(n_nodes);

    for step in 0..n_steps {
        let t = step as f64;
        for i in 0..n_nodes {
            if rng.random::<f64>() < activity_rates[i] {
                // Choose a partner uniformly at random (excluding self)
                let offset: usize = rng.random_range(0..(n_nodes - 1));
                let j = if offset < i { offset } else { offset + 1 };
                tg.add_edge(TemporalEdge::new(i, j, t));
            }
        }
    }

    tg.ensure_sorted();
    Ok(tg)
}

// ---------------------------------------------------------------------------
// Convenience constructor — seeded activity-driven model
// ---------------------------------------------------------------------------

/// Convenience wrapper: run [`activity_driven_model`] with a seeded
/// `ChaCha20` RNG.
pub fn activity_driven_model_seeded(
    n_nodes: usize,
    n_steps: usize,
    activity_rates: &[f64],
    seed: u64,
) -> Result<TemporalGraph> {
    use scirs2_core::random::ChaCha20Rng;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    activity_driven_model(n_nodes, n_steps, activity_rates, &mut rng)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chain() -> TemporalGraph {
        // 0 -> 1 at t=1, 1 -> 2 at t=2, 2 -> 3 at t=3
        let mut tg = TemporalGraph::new(4);
        tg.add_edge(TemporalEdge::new(0, 1, 1.0));
        tg.add_edge(TemporalEdge::new(1, 2, 2.0));
        tg.add_edge(TemporalEdge::new(2, 3, 3.0));
        tg
    }

    #[test]
    fn test_add_and_sort() {
        let mut tg = TemporalGraph::new(3);
        tg.add_edge(TemporalEdge::new(0, 1, 5.0));
        tg.add_edge(TemporalEdge::new(1, 2, 2.0)); // out-of-order
        tg.add_edge(TemporalEdge::new(0, 2, 8.0));

        let edges = tg.edges();
        assert_eq!(edges.len(), 3);
        // must be sorted
        let timestamps: Vec<f64> = edges.iter().map(|e| e.timestamp).collect();
        assert!(timestamps.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn test_snapshot() {
        let mut tg = make_chain();
        let snap = tg.snapshot(0.0, 2.5);
        // edges at t=1 (0-1) and t=2 (1-2) are included
        assert_eq!(snap.edge_count(), 2);
    }

    #[test]
    fn test_temporal_neighbors() {
        let mut tg = make_chain();
        let nbrs = tg.temporal_neighbors(1, 0.0, 10.0);
        let nbr_ids: Vec<usize> = nbrs.iter().map(|(n, _)| *n).collect();
        assert!(nbr_ids.contains(&0));
        assert!(nbr_ids.contains(&2));
    }

    #[test]
    fn test_temporal_path_found() {
        let mut tg = make_chain();
        let path = tg.temporal_path(0, 3, 0.0, 10.0);
        assert!(path.is_some());
        let p = path.expect("path should exist");
        assert_eq!(p.first(), Some(&0));
        assert_eq!(p.last(), Some(&3));
    }

    #[test]
    fn test_temporal_path_no_backwards() {
        // Only allow a narrow window that cannot see t=3
        let mut tg = make_chain();
        let path = tg.temporal_path(0, 3, 0.0, 2.5);
        // Can't reach node 3 — its edge appears at t=3
        assert!(path.is_none());
    }

    #[test]
    fn test_temporal_path_same_source_target() {
        let mut tg = make_chain();
        let path = tg.temporal_path(2, 2, 0.0, 10.0);
        assert_eq!(path, Some(vec![2]));
    }

    #[test]
    fn test_temporal_betweenness_chain() {
        let mut tg = make_chain();
        let bet = tg.temporal_betweenness(0.0, 10.0);
        // On a chain 0-1-2-3, nodes 1 and 2 should have non-zero betweenness
        assert_eq!(bet.len(), 4);
        // node 0 and node 3 are endpoints, should have 0 betweenness
        assert_eq!(bet[0], 0.0);
        assert_eq!(bet[3], 0.0);
    }

    #[test]
    fn test_burstiness_regular() {
        // Regular intervals → sigma=0, B = (0-mu)/(0+mu) = -1 (perfectly periodic)
        let events: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let b = burstiness(&events);
        assert!(
            (b - (-1.0)).abs() < 1e-9,
            "perfectly regular events should have B=-1 (periodic), got {b}"
        );
    }

    #[test]
    fn test_burstiness_few_events() {
        assert_eq!(burstiness(&[]), 0.0);
        assert_eq!(burstiness(&[1.0]), 0.0);
    }

    #[test]
    fn test_burstiness_bursty() {
        // A truly bursty sequence: very small intervals followed by a large gap
        // IETs: [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 1000.0]
        // sigma >> mu => B positive (bursty)
        let mut events: Vec<f64> = (0..10).map(|i| i as f64 * 0.001).collect();
        events.push(1000.0);
        let b = burstiness(&events);
        // Should be positive (bursty)
        assert!(b > 0.0, "bursty sequence should have B>0, got {b}");
    }

    #[test]
    fn test_activity_driven_model() {
        let rates = vec![0.5; 5];
        let tg = activity_driven_model_seeded(5, 20, &rates, 42).expect("model generation failed");
        assert_eq!(tg.n_nodes(), 5);
        // With 5 nodes and 20 steps at activity 0.5, expect some edges
        assert!(tg.n_edges() > 0);
    }

    #[test]
    fn test_activity_driven_model_validation() {
        // Wrong length
        let err = activity_driven_model_seeded(5, 10, &[0.5; 3], 0);
        assert!(err.is_err());

        // Rate out of range
        let err = activity_driven_model_seeded(3, 10, &[0.5, 1.5, 0.3], 0);
        assert!(err.is_err());

        // Too few nodes
        let err = activity_driven_model_seeded(1, 10, &[0.5], 0);
        assert!(err.is_err());
    }

    #[test]
    fn test_aggregate_graph() {
        let mut tg = make_chain();
        let agg = tg.aggregate_graph();
        // All 4 nodes should be present (0,1,2,3)
        assert_eq!(agg.node_count(), 4);
        assert_eq!(agg.edge_count(), 3);
    }

    #[test]
    fn test_weighted_edge() {
        let e = TemporalEdge::with_weight(0, 1, 5.0, 2.5);
        assert_eq!(e.weight, Some(2.5));
        assert_eq!(e.timestamp, 5.0);
    }

    #[test]
    fn test_edges_in_window() {
        let mut tg = TemporalGraph::new(3);
        for t in [1.0, 2.0, 3.0, 4.0, 5.0] {
            tg.add_edge(TemporalEdge::new(0, 1, t));
        }
        let window = tg.edges_in_window(2.0, 4.0);
        assert_eq!(window.len(), 2); // t=2 and t=3
    }
}
