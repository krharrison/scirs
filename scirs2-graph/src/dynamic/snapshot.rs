//! Snapshot-based temporal graph representation.

use std::collections::HashMap;

/// A single temporal snapshot of a graph.
#[derive(Debug, Clone)]
pub struct GraphSnapshot {
    /// Timestamp for this snapshot.
    pub time: f64,
    /// Nodes present at this time.
    pub nodes: Vec<usize>,
    /// Edges at this time: (src, dst, weight).
    pub edges: Vec<(usize, usize, f64)>,
    /// Per-node attribute maps.
    pub node_attributes: HashMap<usize, HashMap<String, f64>>,
}

impl GraphSnapshot {
    /// Create a new empty snapshot at the given time.
    pub fn new(time: f64) -> Self {
        Self {
            time,
            nodes: Vec::new(),
            edges: Vec::new(),
            node_attributes: HashMap::new(),
        }
    }

    /// Add a node, ignoring duplicates.
    pub fn add_node(&mut self, node: usize) {
        if !self.nodes.contains(&node) {
            self.nodes.push(node);
        }
    }

    /// Add an edge, adding both endpoints if absent.
    pub fn add_edge(&mut self, src: usize, dst: usize, weight: f64) {
        self.add_node(src);
        self.add_node(dst);
        self.edges.push((src, dst, weight));
    }

    /// Degree of a node (undirected count).
    pub fn degree(&self, node: usize) -> usize {
        self.edges
            .iter()
            .filter(|(s, d, _)| *s == node || *d == node)
            .count()
    }

    /// Density of the snapshot (directed normalisation: n*(n-1)).
    pub fn density(&self) -> f64 {
        let n = self.nodes.len();
        if n < 2 {
            return 0.0;
        }
        self.edges.len() as f64 / (n * (n - 1)) as f64
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }
}

/// Sequence of graph snapshots forming a temporal graph.
#[derive(Debug, Clone)]
pub struct SnapshotGraph {
    /// Ordered list of snapshots.
    pub snapshots: Vec<GraphSnapshot>,
    /// Optional time-window width used when building snapshots.
    pub time_window: Option<f64>,
}

impl SnapshotGraph {
    /// Create an empty snapshot graph.
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            time_window: None,
        }
    }

    /// Attach a time-window parameter.
    pub fn with_time_window(mut self, window: f64) -> Self {
        self.time_window = Some(window);
        self
    }

    /// Insert a snapshot, maintaining chronological order.
    pub fn add_snapshot(&mut self, snapshot: GraphSnapshot) {
        self.snapshots.push(snapshot);
        self.snapshots.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Number of snapshots.
    pub fn n_snapshots(&self) -> usize {
        self.snapshots.len()
    }

    /// Get the snapshot whose timestamp is closest to `t`.
    pub fn snapshot_at(&self, t: f64) -> Option<&GraphSnapshot> {
        self.snapshots.iter().min_by(|a, b| {
            (a.time - t)
                .abs()
                .partial_cmp(&(b.time - t).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// All unique node identifiers across all snapshots.
    pub fn all_nodes(&self) -> Vec<usize> {
        let mut nodes: Vec<usize> = self
            .snapshots
            .iter()
            .flat_map(|s| s.nodes.iter().cloned())
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        nodes
    }

    /// Temporal degree sequence: (time, degree) for every snapshot.
    pub fn temporal_degree(&self, node: usize) -> Vec<(f64, usize)> {
        self.snapshots
            .iter()
            .map(|s| (s.time, s.degree(node)))
            .collect()
    }

    /// Burstiness coefficient B = (σ-μ)/(σ+μ) for inter-event times of edge (src,dst).
    ///
    /// Returns 0.0 when there are fewer than two activations.
    pub fn burstiness(&self, src: usize, dst: usize) -> f64 {
        let event_times: Vec<f64> = self
            .snapshots
            .iter()
            .filter(|s| {
                s.edges
                    .iter()
                    .any(|(s_, d_, _)| (*s_ == src && *d_ == dst) || (*s_ == dst && *d_ == src))
            })
            .map(|s| s.time)
            .collect();

        if event_times.len() < 2 {
            return 0.0;
        }

        let inter_events: Vec<f64> = event_times.windows(2).map(|w| w[1] - w[0]).collect();
        let n = inter_events.len() as f64;
        let mu = inter_events.iter().sum::<f64>() / n;
        let sigma = (inter_events.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / n).sqrt();

        if sigma + mu < 1e-10 {
            0.0
        } else {
            (sigma - mu) / (sigma + mu)
        }
    }

    /// Graph-level temporal autocorrelation (Jaccard similarity of edge sets at lag `lag`).
    pub fn temporal_correlation(&self, lag: usize) -> f64 {
        if self.snapshots.len() <= lag {
            return 0.0;
        }

        let n = self.snapshots.len() - lag;
        let mut similarity_sum = 0.0;

        for i in 0..n {
            let s1 = &self.snapshots[i];
            let s2 = &self.snapshots[i + lag];

            let e1: std::collections::HashSet<(usize, usize)> = s1
                .edges
                .iter()
                .map(|(a, b, _)| (*a.min(b), *a.max(b)))
                .collect();
            let e2: std::collections::HashSet<(usize, usize)> = s2
                .edges
                .iter()
                .map(|(a, b, _)| (*a.min(b), *a.max(b)))
                .collect();

            let intersection = e1.intersection(&e2).count();
            let union_size = e1.union(&e2).count();

            similarity_sum += if union_size > 0 {
                intersection as f64 / union_size as f64
            } else {
                0.0
            };
        }

        similarity_sum / n as f64
    }
}

impl Default for SnapshotGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_density() {
        let mut s = GraphSnapshot::new(1.0);
        s.add_edge(0, 1, 1.0);
        s.add_edge(0, 2, 1.0);
        // 3 nodes, 2 edges => density = 2 / (3*2) ≈ 0.333
        let d = s.density();
        assert!((d - 1.0 / 3.0).abs() < 1e-9, "density={d}");
    }

    #[test]
    fn test_temporal_correlation_identical() {
        let mut sg = SnapshotGraph::new();
        for t in 0..4 {
            let mut snap = GraphSnapshot::new(t as f64);
            snap.add_edge(0, 1, 1.0);
            snap.add_edge(1, 2, 1.0);
            sg.add_snapshot(snap);
        }
        // All snapshots have the same edges => correlation at any lag should be 1.0
        let corr = sg.temporal_correlation(1);
        assert!((corr - 1.0).abs() < 1e-9, "corr={corr}");
    }

    #[test]
    fn test_burstiness_regular() {
        // Regular inter-event times → σ = 0 → B = -1
        let mut sg = SnapshotGraph::new();
        for t in 0..5 {
            let mut snap = GraphSnapshot::new(t as f64);
            snap.add_edge(0, 1, 1.0);
            sg.add_snapshot(snap);
        }
        let b = sg.burstiness(0, 1);
        assert!(b < -0.5, "b={b}");
    }

    #[test]
    fn test_all_nodes() {
        let mut sg = SnapshotGraph::new();
        let mut s1 = GraphSnapshot::new(0.0);
        s1.add_edge(0, 1, 1.0);
        let mut s2 = GraphSnapshot::new(1.0);
        s2.add_edge(1, 2, 1.0);
        sg.add_snapshot(s1);
        sg.add_snapshot(s2);
        let nodes = sg.all_nodes();
        assert_eq!(nodes, vec![0, 1, 2]);
    }
}
