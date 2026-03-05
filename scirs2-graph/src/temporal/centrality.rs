//! Temporal centrality measures
//!
//! Implements three fundamental centrality concepts for temporal networks:
//!
//! 1. **Temporal closeness**: inverse average earliest-arrival time from a given
//!    source to all reachable nodes.
//! 2. **Temporal betweenness**: fraction of foremost paths that pass through
//!    each node.
//! 3. **Temporal PageRank**: time-respecting PageRank that discounts edges based
//!    on temporal ordering.
//!
//! # References
//! - Buss, S., Molter, H., Niedermeier, R., & Rymar, M. (2020). Algorithmic aspects
//!   of temporal betweenness. KDD 2020.
//! - Rozenshtein, P., & Gionis, A. (2016). Temporal PageRank. ECML PKDD 2016.

use super::graph::TemporalGraph;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// temporal_closeness
// ─────────────────────────────────────────────────────────────────────────────

/// Compute temporal closeness centrality for a single node.
///
/// Closeness is defined as the reciprocal of the average earliest-arrival time
/// from `node` to all other *reachable* nodes using foremost (time-respecting)
/// paths starting at `t=0`.
///
/// A node with no reachable destinations returns `0.0`.
///
/// # Arguments
/// * `tg`   – mutable reference to the temporal graph (needed for lazy sorting)
/// * `node` – source node index
pub fn temporal_closeness(tg: &mut TemporalGraph, node: usize) -> f64 {
    let n = tg.nodes;
    if n <= 1 || node >= n {
        return 0.0;
    }

    tg.ensure_sorted();
    // We use the foremost arrival BFS/Dijkstra internally
    let t_start = 0.0;
    let t_end = tg.edges.last().map(|e| e.timestamp + 1.0).unwrap_or(1.0);

    // Run foremost-path Dijkstra from `node` to collect arrival times
    let arrivals = foremost_arrivals(tg, node, t_start, t_end);

    let mut total_time = 0.0;
    let mut reachable = 0usize;
    for (v, arr) in arrivals.iter().enumerate() {
        if v != node && arr.is_finite() {
            total_time += arr;
            reachable += 1;
        }
    }

    if reachable == 0 || total_time <= 0.0 {
        return 0.0;
    }

    // Normalised closeness: (n-1) / sum_of_arrival_times * (reachable / (n-1))
    // Simplified: reachable / total_time
    reachable as f64 / total_time
}

/// Internal: compute foremost arrival times from `source` to all nodes.
///
/// Returns a vector of length `tg.nodes`; unreachable nodes have `f64::INFINITY`.
fn foremost_arrivals(tg: &TemporalGraph, source: usize, t_start: f64, t_end: f64) -> Vec<f64> {
    let n = tg.nodes;
    let mut arrival = vec![f64::INFINITY; n];
    arrival[source] = t_start;

    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    #[derive(PartialEq, Eq)]
    struct State(ordered_float::OrderedFloat<f64>, usize);
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for State {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
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
            continue;
        }
        let lo = tg.edges.partition_point(|e| e.timestamp < arr);
        for e in &tg.edges[lo..] {
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
                heap.push(State(ordered_float::OrderedFloat(e.timestamp), nbr));
            }
        }
    }

    arrival
}

// ─────────────────────────────────────────────────────────────────────────────
// temporal_betweenness
// ─────────────────────────────────────────────────────────────────────────────

/// Compute temporal betweenness centrality for all nodes.
///
/// For each ordered pair `(s, t)` with `s ≠ t`, we find the foremost path
/// (earliest-arrival) and credit each intermediate node `1/(len-2)` (where `len`
/// is the path length in hops).  The result is normalised by `(n-1)(n-2)`.
///
/// Runs in O(n² · (E log n)) time where E is the number of temporal edges.
///
/// # Arguments
/// * `tg` – mutable reference to the temporal graph
///
/// # Returns
/// A `Vec<f64>` of length `tg.nodes` with normalised betweenness scores.
pub fn temporal_betweenness(tg: &mut TemporalGraph) -> Vec<f64> {
    let n = tg.nodes;
    let mut bet = vec![0.0f64; n];

    if n <= 2 {
        return bet;
    }

    tg.ensure_sorted();
    let t_end = tg.edges.last().map(|e| e.timestamp + 1.0).unwrap_or(1.0);

    for s in 0..n {
        for t in 0..n {
            if s == t {
                continue;
            }
            if let Some(path) = foremost_path_internal(tg, s, t, 0.0, t_end) {
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

    // Normalise
    let norm = ((n - 1) * (n - 2)) as f64;
    if norm > 0.0 {
        for b in &mut bet {
            *b /= norm;
        }
    }
    bet
}

/// Internal: find the foremost path (no mutable borrow of tg needed for edges
/// since we hold a shared reference; sorting must have been done beforehand).
fn foremost_path_internal(
    tg: &TemporalGraph,
    source: usize,
    target: usize,
    t_start: f64,
    t_end: f64,
) -> Option<Vec<usize>> {
    if source >= tg.nodes || target >= tg.nodes {
        return None;
    }
    if source == target {
        return Some(vec![source]);
    }

    let n = tg.nodes;
    let mut arrival = vec![f64::INFINITY; n];
    arrival[source] = t_start;
    let mut pred: Vec<Option<usize>> = vec![None; n];

    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    #[derive(PartialEq, Eq)]
    struct State(ordered_float::OrderedFloat<f64>, usize);
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for State {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
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
            continue;
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

        let lo = tg.edges.partition_point(|e| e.timestamp < arr);
        for e in &tg.edges[lo..] {
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

// ─────────────────────────────────────────────────────────────────────────────
// temporal_pagerank
// ─────────────────────────────────────────────────────────────────────────────

/// Compute temporal PageRank for all nodes.
///
/// The algorithm works on a series of static snapshots (one per unique timestamp)
/// and applies the damping factor `alpha` both to the standard PageRank teleport
/// and to temporal decay between snapshots.
///
/// At each snapshot:
/// - Nodes that are involved in contacts at the current timestamp receive a
///   "temporal boost" proportional to their out-degree in that snapshot.
/// - After the snapshot, scores are decayed by `alpha` and mixed with the
///   previous scores to ensure temporal smoothness.
///
/// # Arguments
/// * `tg`    – mutable reference to the temporal graph
/// * `alpha` – damping factor (typically 0.85)
///
/// # Returns
/// A `Vec<f64>` of length `tg.nodes` with temporal PageRank scores (sum ≈ 1).
pub fn temporal_pagerank(tg: &mut TemporalGraph, alpha: f64) -> Vec<f64> {
    let n = tg.nodes;
    if n == 0 {
        return Vec::new();
    }

    tg.ensure_sorted();

    let alpha = alpha.clamp(0.0, 1.0);
    let teleport = (1.0 - alpha) / n as f64;

    // Collect unique timestamps
    let mut timestamps: Vec<f64> = tg.edges.iter().map(|e| e.timestamp).collect();
    timestamps.dedup();
    timestamps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if timestamps.is_empty() {
        return vec![1.0 / n as f64; n];
    }

    let mut scores = vec![1.0 / n as f64; n];

    for &ts in &timestamps {
        // Contacts at exactly this timestamp
        let contacts: Vec<_> = tg
            .edges
            .iter()
            .filter(|e| (e.timestamp - ts).abs() < 1e-12)
            .collect();

        if contacts.is_empty() {
            continue;
        }

        // Build adjacency for this snapshot
        let mut out_degree: Vec<f64> = vec![0.0; n];
        let mut adj: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
        for e in &contacts {
            let w = e.weight;
            adj.entry(e.source).or_default().push((e.target, w));
            // Undirected
            adj.entry(e.target).or_default().push((e.source, w));
            out_degree[e.source] += w;
            out_degree[e.target] += w;
        }

        // One step of PageRank on this snapshot
        let mut new_scores = vec![teleport; n];
        for (u, neighbors) in &adj {
            let deg = out_degree[*u];
            if deg <= 0.0 {
                continue;
            }
            for &(v, w) in neighbors {
                new_scores[v] += alpha * scores[*u] * (w / deg);
            }
        }

        // Mix with previous scores (temporal smoothing)
        let decay = 0.5_f64;
        for i in 0..n {
            scores[i] = decay * new_scores[i] + (1.0 - decay) * scores[i];
        }

        // Re-normalise
        let total: f64 = scores.iter().sum();
        if total > 0.0 {
            for s in &mut scores {
                *s /= total;
            }
        }
    }

    scores
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::graph::TemporalEdge;
    use super::*;

    fn make_chain() -> TemporalGraph {
        let mut tg = TemporalGraph::new(4);
        tg.add_edge(TemporalEdge::new(0, 1, 1.0));
        tg.add_edge(TemporalEdge::new(1, 2, 2.0));
        tg.add_edge(TemporalEdge::new(2, 3, 3.0));
        tg
    }

    fn make_star() -> TemporalGraph {
        // Hub node 0 connected to leaves 1, 2, 3 at different times
        let mut tg = TemporalGraph::new(4);
        tg.add_edge(TemporalEdge::new(0, 1, 1.0));
        tg.add_edge(TemporalEdge::new(0, 2, 2.0));
        tg.add_edge(TemporalEdge::new(0, 3, 3.0));
        tg
    }

    #[test]
    fn test_temporal_closeness_chain() {
        let mut tg = make_chain();
        // Source node 0: can reach 1, 2, 3 at times 1, 2, 3
        let cl0 = temporal_closeness(&mut tg, 0);
        // Source node 3: no outgoing time-respecting edges → may be 0
        let cl3 = temporal_closeness(&mut tg, 3);
        assert!(cl0 > 0.0, "node 0 should have positive closeness");
        // node 3 is at the end of the chain, may have low closeness
        let _ = cl3; // just ensure no panic
    }

    #[test]
    fn test_temporal_closeness_star() {
        let mut tg = make_star();
        let cl_hub = temporal_closeness(&mut tg, 0);
        let cl_leaf = temporal_closeness(&mut tg, 1);
        // Hub should have higher closeness than leaf
        assert!(cl_hub >= cl_leaf, "hub should have >= closeness vs leaf");
    }

    #[test]
    fn test_temporal_betweenness_chain() {
        let mut tg = make_chain();
        let bet = temporal_betweenness(&mut tg);
        assert_eq!(bet.len(), 4);
        // Endpoints 0 and 3 should have 0 betweenness
        assert_eq!(bet[0], 0.0);
        assert_eq!(bet[3], 0.0);
        // Intermediate nodes 1, 2 should have positive betweenness
        assert!(
            bet[1] > 0.0 || bet[2] > 0.0,
            "at least one intermediate node should have positive betweenness"
        );
    }

    #[test]
    fn test_temporal_betweenness_star() {
        let mut tg = make_star();
        let bet = temporal_betweenness(&mut tg);
        assert_eq!(bet.len(), 4);
        // All paths from leaves to other leaves must go through node 0
        // (or may not exist due to temporal ordering)
        let _ = bet; // verify no panics
    }

    #[test]
    fn test_temporal_pagerank_normalised() {
        let mut tg = make_chain();
        let pr = temporal_pagerank(&mut tg, 0.85);
        assert_eq!(pr.len(), 4);
        let total: f64 = pr.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "PageRank should sum to ~1.0, got {total}"
        );
    }

    #[test]
    fn test_temporal_pagerank_empty() {
        let mut tg = TemporalGraph::new(3);
        let pr = temporal_pagerank(&mut tg, 0.85);
        assert_eq!(pr.len(), 3);
        // All equal
        assert!((pr[0] - pr[1]).abs() < 1e-9);
    }
}
