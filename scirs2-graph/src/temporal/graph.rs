//! Temporal graph data structures (stream model, f64 timestamps)
//!
//! This module provides `TemporalEdge` and `TemporalGraph` types for the
//! stream-of-interactions model where each edge carries a real-valued timestamp.
//! These types are designed for temporal network analysis algorithms such as
//! centrality, motif counting, and dynamic community detection.
//!
//! The `TemporalGraph` here is distinct from `crate::temporal::TemporalGraph`
//! (which uses generic typed nodes with time intervals) and from
//! `crate::temporal_graph::TemporalGraph` (the original stream model).
//! This module provides an ergonomic API focused on algorithm composability.

use crate::base::Graph;
use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// TemporalEdge
// ─────────────────────────────────────────────────────────────────────────────

/// A single directed temporal edge `(source, target, timestamp, weight)`.
///
/// Edges are directed by convention; callers wanting undirected semantics should
/// insert both `(u, v)` and `(v, u)` versions.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalEdge {
    /// Source node index (0-based, must be `< TemporalGraph::nodes`)
    pub source: usize,
    /// Target node index (0-based, must be `< TemporalGraph::nodes`)
    pub target: usize,
    /// Continuous-time timestamp of the interaction
    pub timestamp: f64,
    /// Edge weight (defaults to `1.0`)
    pub weight: f64,
}

impl TemporalEdge {
    /// Create a unit-weight temporal edge.
    pub fn new(source: usize, target: usize, timestamp: f64) -> Self {
        TemporalEdge {
            source,
            target,
            timestamp,
            weight: 1.0,
        }
    }

    /// Create a weighted temporal edge.
    pub fn with_weight(source: usize, target: usize, timestamp: f64, weight: f64) -> Self {
        TemporalEdge {
            source,
            target,
            timestamp,
            weight,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TemporalGraph
// ─────────────────────────────────────────────────────────────────────────────

/// Temporal graph in the stream-of-interactions model.
///
/// Stores a sorted sequence of `TemporalEdge` contacts; nodes are identified
/// by consecutive `usize` indices `0..nodes`.
///
/// ## Example
/// ```
/// use scirs2_graph::temporal::TemporalGraph;
/// use scirs2_graph::temporal::TemporalEdge;
///
/// let mut tg = TemporalGraph::new(4);
/// tg.add_edge(TemporalEdge::new(0, 1, 1.0));
/// tg.add_edge(TemporalEdge::new(1, 2, 2.0));
/// tg.add_edge(TemporalEdge::new(2, 3, 3.0));
///
/// // Static snapshot of interactions in [0.5, 2.5)
/// let snap = tg.snapshot(0.5, 2.5);
/// assert!(snap.edge_count() >= 1);
/// ```
#[derive(Debug, Clone)]
pub struct TemporalGraph {
    /// Total number of nodes (fixed at construction; node indices are `0..nodes`)
    pub nodes: usize,
    /// Edge stream, sorted by timestamp
    pub edges: Vec<TemporalEdge>,
    /// Internal: whether `edges` is currently sorted
    sorted: bool,
}

impl TemporalGraph {
    /// Create an empty temporal graph with `n_nodes` nodes.
    pub fn new(n_nodes: usize) -> Self {
        TemporalGraph {
            nodes: n_nodes,
            edges: Vec::new(),
            sorted: true,
        }
    }

    /// Add a temporal edge.  The internal list is lazily re-sorted when queried.
    pub fn add_edge(&mut self, edge: TemporalEdge) {
        if let Some(last) = self.edges.last() {
            if edge.timestamp < last.timestamp {
                self.sorted = false;
            }
        }
        self.edges.push(edge);
    }

    /// Ensure `edges` is sorted by timestamp (stable sort, O(n log n) worst case).
    pub(crate) fn ensure_sorted(&mut self) {
        if !self.sorted {
            self.edges.sort_by(|a, b| {
                a.timestamp
                    .partial_cmp(&b.timestamp)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.sorted = true;
        }
    }

    /// Return all edges in sorted order.
    pub fn time_ordered_edges(&mut self) -> &[TemporalEdge] {
        self.ensure_sorted();
        &self.edges
    }

    /// Borrow edges without sorting (useful if you know they are already sorted).
    pub fn edges_slice(&self) -> &[TemporalEdge] {
        &self.edges
    }

    /// Return edges in the half-open window `[start, end)`.
    pub fn edges_in_window(&mut self, start: f64, end: f64) -> &[TemporalEdge] {
        self.ensure_sorted();
        let lo = self.edges.partition_point(|e| e.timestamp < start);
        let hi = self.edges.partition_point(|e| e.timestamp < end);
        &self.edges[lo..hi]
    }

    // ────────────────────────────────────────────────────────────────────────
    // snapshot
    // ────────────────────────────────────────────────────────────────────────

    /// Build a static undirected snapshot for the window `[start, end)`.
    ///
    /// Repeated contacts between the same pair of nodes accumulate weights.
    /// The returned `Graph<usize, f64>` contains only nodes that appear in
    /// at least one edge within the window.
    pub fn snapshot(&mut self, start: f64, end: f64) -> Graph<usize, f64> {
        let window: Vec<TemporalEdge> = self.edges_in_window(start, end).to_vec();

        let mut edge_weights: HashMap<(usize, usize), f64> = HashMap::new();
        let mut active: HashSet<usize> = HashSet::new();

        for e in &window {
            active.insert(e.source);
            active.insert(e.target);
            let key = (e.source.min(e.target), e.source.max(e.target));
            *edge_weights.entry(key).or_insert(0.0) += e.weight;
        }

        let mut graph: Graph<usize, f64> = Graph::new();
        for &n in &active {
            graph.add_node(n);
        }
        for ((u, v), w) in edge_weights {
            // Ignore errors — nodes are already added above
            let _ = graph.add_edge(u, v, w);
        }
        graph
    }

    // ────────────────────────────────────────────────────────────────────────
    // aggregate_graph
    // ────────────────────────────────────────────────────────────────────────

    /// Collapse all temporal contacts into a static undirected weighted graph
    /// by summing contact weights over all time.
    pub fn aggregate_graph(&mut self) -> Graph<usize, f64> {
        if self.edges.is_empty() {
            let mut g: Graph<usize, f64> = Graph::new();
            for i in 0..self.nodes {
                g.add_node(i);
            }
            return g;
        }

        self.ensure_sorted();
        let t_start = self.edges.first().map(|e| e.timestamp).unwrap_or(0.0);
        let t_end = self.edges.last().map(|e| e.timestamp + 1.0).unwrap_or(1.0);
        self.snapshot(t_start, t_end)
    }

    // ────────────────────────────────────────────────────────────────────────
    // temporal_neighbors
    // ────────────────────────────────────────────────────────────────────────

    /// Return all neighbors of `node` contacted in `[t_start, t_end)`.
    ///
    /// Returns `(neighbor_id, earliest_contact_time)` pairs (undirected semantics).
    pub fn temporal_neighbors(
        &mut self,
        node: usize,
        t_start: f64,
        t_end: f64,
    ) -> Vec<(usize, f64)> {
        let window: Vec<TemporalEdge> = self.edges_in_window(t_start, t_end).to_vec();

        let mut first_contact: HashMap<usize, f64> = HashMap::new();
        for e in &window {
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

    // ────────────────────────────────────────────────────────────────────────
    // foremost_path  (Dijkstra on event times)
    // ────────────────────────────────────────────────────────────────────────

    /// Find a time-respecting foremost (earliest-arrival) path from `source`
    /// to `target` using only contacts in `[t_start, t_end)`.
    ///
    /// Returns `None` when no such path exists.
    pub fn foremost_path(
        &mut self,
        source: usize,
        target: usize,
        t_start: f64,
        t_end: f64,
    ) -> Option<Vec<usize>> {
        if source >= self.nodes || target >= self.nodes {
            return None;
        }
        if source == target {
            return Some(vec![source]);
        }

        self.ensure_sorted();

        let mut arrival = vec![f64::INFINITY; self.nodes];
        arrival[source] = t_start;
        let mut pred: Vec<Option<usize>> = vec![None; self.nodes];

        // Min-heap keyed on (arrival_time, node)
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        #[derive(PartialEq)]
        struct State(ordered_float::OrderedFloat<f64>, usize);
        impl Eq for State {}
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for State {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                // We want a min-heap, so we compare (reversed) arrival, then node
                Reverse(self.0)
                    .cmp(&Reverse(other.0))
                    .then(self.1.cmp(&other.1))
            }
        }

        let mut heap = BinaryHeap::new();
        heap.push(State(ordered_float::OrderedFloat(t_start), source));

        while let Some(State(arr_of, node)) = heap.pop() {
            let arr = arr_of.0;
            if arr > arrival[node] {
                continue; // stale
            }
            if node == target {
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

            // Scan edges at or after current arrival
            let lo = self.edges.partition_point(|e| e.timestamp < arr);
            for e in &self.edges[lo..] {
                if e.timestamp >= t_end {
                    break;
                }
                let nbr = if e.source == node {
                    e.target
                } else if e.target == node {
                    e.source
                } else {
                    continue;
                };
                if e.timestamp < arrival[nbr] {
                    arrival[nbr] = e.timestamp;
                    pred[nbr] = Some(node);
                    heap.push(State(ordered_float::OrderedFloat(e.timestamp), nbr));
                }
            }
        }

        None
    }

    /// Number of temporal contacts stored.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chain() -> TemporalGraph {
        let mut tg = TemporalGraph::new(4);
        tg.add_edge(TemporalEdge::new(0, 1, 1.0));
        tg.add_edge(TemporalEdge::new(1, 2, 2.0));
        tg.add_edge(TemporalEdge::new(2, 3, 3.0));
        tg
    }

    #[test]
    fn test_snapshot_basic() {
        let mut tg = make_chain();
        let snap = tg.snapshot(0.5, 2.5);
        // edges at t=1 (0-1) and t=2 (1-2) are in the window
        assert_eq!(snap.edge_count(), 2);
    }

    #[test]
    fn test_time_ordered_edges_sorted() {
        let mut tg = TemporalGraph::new(3);
        tg.add_edge(TemporalEdge::new(0, 1, 5.0));
        tg.add_edge(TemporalEdge::new(1, 2, 2.0));
        tg.add_edge(TemporalEdge::new(0, 2, 8.0));

        let edges = tg.time_ordered_edges();
        let ts: Vec<f64> = edges.iter().map(|e| e.timestamp).collect();
        assert!(ts.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn test_foremost_path() {
        let mut tg = make_chain();
        let path = tg.foremost_path(0, 3, 0.0, 10.0);
        assert!(path.is_some());
        let p = path.expect("path should exist");
        assert_eq!(p.first(), Some(&0));
        assert_eq!(p.last(), Some(&3));
    }

    #[test]
    fn test_foremost_path_no_time_travel() {
        let mut tg = make_chain();
        // Only window [0, 2.5) — edge to node 3 at t=3 not visible
        let path = tg.foremost_path(0, 3, 0.0, 2.5);
        assert!(path.is_none());
    }

    #[test]
    fn test_aggregate_graph() {
        let mut tg = make_chain();
        let agg = tg.aggregate_graph();
        assert_eq!(agg.node_count(), 4);
        assert_eq!(agg.edge_count(), 3);
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
    fn test_edge_with_weight() {
        let e = TemporalEdge::with_weight(0, 1, 5.0, 2.5);
        assert_eq!(e.weight, 2.5);
    }
}
