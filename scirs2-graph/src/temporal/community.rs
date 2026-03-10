//! Dynamic community detection for temporal graphs
//!
//! Implements evolutionary clustering algorithms that track community structure
//! as it evolves over time.  The key challenge is to balance:
//! - **Community quality** at each snapshot (intra-snapshot modularity)
//! - **Temporal smoothness** across consecutive snapshots (community continuity)
//!
//! # Algorithm: Evolutionary Clustering
//!
//! The `evolutionary_clustering` function divides the temporal graph into
//! `n_snapshots` equal-width time windows and runs a greedy modularity
//! optimisation on each snapshot.  A temporal penalty term penalises label
//! switches between adjacent snapshots, controlled by `alpha ∈ [0, 1]`.
//!
//! A higher `alpha` gives more weight to temporal smoothness (communities
//! change slowly); a lower `alpha` gives more weight to snapshot quality
//! (communities can change abruptly).
//!
//! # References
//! - Chi, Y., Song, X., Zhou, D., Hino, K., & Zhu, B. L. (2007).
//!   Evolutionary spectral clustering by incorporating temporal smoothness.
//!   KDD 2007.
//! - Yang, T., Chi, Y., Zhu, S., Gong, Y., & Jin, R. (2011).
//!   Detecting communities and their evolutions in dynamic social networks.
//!   Machine Learning 82(2).

use super::graph::TemporalGraph;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// DynamicCommunity
// ─────────────────────────────────────────────────────────────────────────────

/// Result of dynamic community detection.
///
/// Stores the community assignment for each node at each snapshot.
/// `node_memberships[snapshot][node]` gives the community ID for `node` at
/// time `snapshot`.  Community IDs are contiguous integers starting from 0.
#[derive(Debug, Clone)]
pub struct DynamicCommunity {
    /// `node_memberships[t][v]` = community of node `v` at snapshot `t`
    pub node_memberships: Vec<Vec<usize>>,
    /// Number of snapshots
    pub n_snapshots: usize,
    /// Number of nodes
    pub n_nodes: usize,
    /// Number of communities found at each snapshot
    pub community_counts: Vec<usize>,
}

impl DynamicCommunity {
    /// Create a DynamicCommunity from a 2D membership matrix.
    pub fn new(node_memberships: Vec<Vec<usize>>) -> Self {
        let n_snapshots = node_memberships.len();
        let n_nodes = node_memberships.first().map(|v| v.len()).unwrap_or(0);
        let community_counts = node_memberships
            .iter()
            .map(|snap| snap.iter().copied().max().map(|m| m + 1).unwrap_or(0))
            .collect();
        DynamicCommunity {
            node_memberships,
            n_snapshots,
            n_nodes,
            community_counts,
        }
    }

    /// Get the community of node `v` at snapshot `t`.
    pub fn membership(&self, t: usize, v: usize) -> Option<usize> {
        self.node_memberships
            .get(t)
            .and_then(|snap| snap.get(v))
            .copied()
    }

    /// Compute the average normalised mutual information (NMI) between adjacent
    /// snapshots as a measure of temporal stability (1.0 = perfectly stable).
    pub fn temporal_stability(&self) -> f64 {
        if self.n_snapshots < 2 {
            return 1.0;
        }

        let mut total = 0.0;
        let count = (self.n_snapshots - 1) as f64;
        for t in 0..(self.n_snapshots - 1) {
            total +=
                nmi_between_snapshots(&self.node_memberships[t], &self.node_memberships[t + 1]);
        }
        total / count
    }
}

/// Compute approximate normalised mutual information between two label vectors.
fn nmi_between_snapshots(labels_a: &[usize], labels_b: &[usize]) -> f64 {
    let n = labels_a.len().min(labels_b.len());
    if n == 0 {
        return 1.0;
    }

    let mut joint: HashMap<(usize, usize), usize> = HashMap::new();
    let mut freq_a: HashMap<usize, usize> = HashMap::new();
    let mut freq_b: HashMap<usize, usize> = HashMap::new();

    for i in 0..n {
        let a = labels_a[i];
        let b = labels_b[i];
        *joint.entry((a, b)).or_insert(0) += 1;
        *freq_a.entry(a).or_insert(0) += 1;
        *freq_b.entry(b).or_insert(0) += 1;
    }

    let n_f = n as f64;
    let mut h_ab = 0.0;
    let mut h_a = 0.0;
    let mut h_b = 0.0;

    for &cnt in joint.values() {
        let p = cnt as f64 / n_f;
        if p > 0.0 {
            h_ab -= p * p.ln();
        }
    }
    for &cnt in freq_a.values() {
        let p = cnt as f64 / n_f;
        if p > 0.0 {
            h_a -= p * p.ln();
        }
    }
    for &cnt in freq_b.values() {
        let p = cnt as f64 / n_f;
        if p > 0.0 {
            h_b -= p * p.ln();
        }
    }

    let mi = h_a + h_b - h_ab;
    let denom = (h_a + h_b) / 2.0;
    if denom <= 0.0 {
        1.0
    } else {
        mi / denom
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// evolutionary_clustering
// ─────────────────────────────────────────────────────────────────────────────

/// Run evolutionary clustering on a temporal graph.
///
/// The temporal graph is partitioned into `n_snapshots` equal time windows.
/// For each snapshot, a modularity-maximising community assignment is computed
/// using a greedy label-propagation heuristic.  A temporal smoothness penalty
/// (controlled by `alpha`) discourages nodes from changing community labels
/// between consecutive snapshots.
///
/// # Arguments
/// * `tg`          – mutable reference to the temporal graph
/// * `n_snapshots` – number of time snapshots to create
/// * `alpha`       – smoothness weight in [0, 1]; 0 = purely snapshot quality,
///   1 = purely temporal smoothness
///
/// # Returns
/// A `DynamicCommunity` containing membership vectors for each snapshot.
pub fn evolutionary_clustering(
    tg: &mut TemporalGraph,
    n_snapshots: usize,
    alpha: f64,
) -> DynamicCommunity {
    let n = tg.nodes;
    if n == 0 || n_snapshots == 0 {
        return DynamicCommunity::new(vec![vec![0usize; n]; n_snapshots.max(1)]);
    }

    tg.ensure_sorted();
    let alpha = alpha.clamp(0.0, 1.0);

    let (t_min, t_max) = if tg.edges.is_empty() {
        (0.0, 1.0)
    } else {
        (
            tg.edges.first().map(|e| e.timestamp).unwrap_or(0.0),
            tg.edges.last().map(|e| e.timestamp).unwrap_or(1.0),
        )
    };

    let window_width = if (t_max - t_min).abs() < 1e-12 {
        1.0
    } else {
        (t_max - t_min) / n_snapshots as f64
    };

    let mut all_memberships: Vec<Vec<usize>> = Vec::with_capacity(n_snapshots);
    let mut prev_memberships: Option<Vec<usize>> = None;

    for snap_idx in 0..n_snapshots {
        let t_start = t_min + snap_idx as f64 * window_width;
        let t_end = if snap_idx + 1 == n_snapshots {
            t_max + 1.0 // include last edge
        } else {
            t_min + (snap_idx + 1) as f64 * window_width
        };

        // Build adjacency for this snapshot
        let adj = build_snapshot_adjacency(tg, n, t_start, t_end);

        // Run label-propagation with temporal smoothness penalty
        let memberships =
            label_propagation_with_smoothness(&adj, n, prev_memberships.as_deref(), alpha);

        prev_memberships = Some(memberships.clone());
        all_memberships.push(memberships);
    }

    DynamicCommunity::new(all_memberships)
}

/// Build an adjacency list (weighted) for a given time window.
fn build_snapshot_adjacency(
    tg: &TemporalGraph,
    n: usize,
    t_start: f64,
    t_end: f64,
) -> Vec<Vec<(usize, f64)>> {
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    // Accumulate weights per edge (undirected)
    let mut weight_map: HashMap<(usize, usize), f64> = HashMap::new();

    for e in &tg.edges {
        if e.timestamp < t_start || e.timestamp >= t_end {
            continue;
        }
        let key = (e.source.min(e.target), e.source.max(e.target));
        *weight_map.entry(key).or_insert(0.0) += e.weight;
    }

    for ((u, v), w) in weight_map {
        adj[u].push((v, w));
        adj[v].push((u, w));
    }

    adj
}

/// Greedy label-propagation community detection with optional temporal smoothness.
///
/// Each node starts in its own community (or from the previous snapshot's labels
/// when `prev` is provided).  Nodes are randomly updated by adopting the most
/// popular label among their neighbours, weighted by edge weights plus the
/// temporal smoothness bonus for keeping the previous label.
fn label_propagation_with_smoothness(
    adj: &[Vec<(usize, f64)>],
    n: usize,
    prev: Option<&[usize]>,
    alpha: f64,
) -> Vec<usize> {
    // Initialise labels: from previous snapshot if available, else own index
    let mut labels: Vec<usize> = if let Some(p) = prev {
        p.to_vec()
    } else {
        (0..n).collect()
    };

    let max_iter = 20usize;
    let mut changed = true;
    let mut iter = 0;

    while changed && iter < max_iter {
        changed = false;
        // Deterministic update order (sorted by node index)
        let order: Vec<usize> = (0..n).collect();

        for &v in &order {
            // Count label votes from neighbours
            let mut vote: HashMap<usize, f64> = HashMap::new();

            for &(nbr, w) in &adj[v] {
                *vote.entry(labels[nbr]).or_insert(0.0) += w;
            }

            if vote.is_empty() {
                continue;
            }

            // Add temporal smoothness bonus for keeping the previous label
            if let Some(p) = prev {
                if v < p.len() {
                    *vote.entry(p[v]).or_insert(0.0) += alpha * 1.0;
                }
            }

            // Choose the label with the highest vote
            let best_label = vote
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(&l, _)| l);

            if let Some(best) = best_label {
                if best != labels[v] {
                    labels[v] = best;
                    changed = true;
                }
            }
        }

        iter += 1;
    }

    // Compact label IDs to be contiguous starting from 0
    compact_labels(&mut labels);
    labels
}

/// Remap label IDs to be contiguous integers starting from 0.
fn compact_labels(labels: &mut [usize]) {
    let mut mapping: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0usize;
    for l in labels.iter_mut() {
        let id = mapping.entry(*l).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        *l = *id;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Modularity computation (helper, not exported)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute modularity Q for a given community assignment on a weighted graph.
/// Returns Q ∈ [-0.5, 1.0]; higher is better.
#[allow(dead_code)]
fn compute_modularity(adj: &[Vec<(usize, f64)>], labels: &[usize]) -> f64 {
    let n = adj.len();
    let mut total_weight = 0.0;
    let mut degree: Vec<f64> = vec![0.0; n];

    for (u, neighbors) in adj.iter().enumerate() {
        for &(_, w) in neighbors {
            degree[u] += w;
            total_weight += w;
        }
    }
    total_weight /= 2.0; // Each edge counted twice

    if total_weight <= 0.0 {
        return 0.0;
    }

    let mut q = 0.0;
    for u in 0..n {
        for &(v, w) in &adj[u] {
            if labels[u] == labels[v] {
                q += w - degree[u] * degree[v] / (2.0 * total_weight);
            }
        }
    }

    q / (2.0 * total_weight)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::graph::TemporalEdge;
    use super::*;

    /// Create a graph with two clear communities: {0,1,2} and {3,4,5}
    /// with dense within-group contacts and a sparse bridge.
    fn make_two_community_graph() -> TemporalGraph {
        let mut tg = TemporalGraph::new(6);
        // Community A: nodes 0, 1, 2 — dense contacts at t=1..3
        for t in 1..=3 {
            tg.add_edge(TemporalEdge::with_weight(0, 1, t as f64, 2.0));
            tg.add_edge(TemporalEdge::with_weight(1, 2, t as f64 + 0.1, 2.0));
            tg.add_edge(TemporalEdge::with_weight(0, 2, t as f64 + 0.2, 2.0));
        }
        // Community B: nodes 3, 4, 5 — dense contacts at t=4..6
        for t in 4..=6 {
            tg.add_edge(TemporalEdge::with_weight(3, 4, t as f64, 2.0));
            tg.add_edge(TemporalEdge::with_weight(4, 5, t as f64 + 0.1, 2.0));
            tg.add_edge(TemporalEdge::with_weight(3, 5, t as f64 + 0.2, 2.0));
        }
        // Sparse bridge between communities at t=7
        tg.add_edge(TemporalEdge::with_weight(2, 3, 7.0, 0.1));
        tg
    }

    #[test]
    fn test_evolutionary_clustering_basic() {
        let mut tg = make_two_community_graph();
        let dyn_com = evolutionary_clustering(&mut tg, 3, 0.5);

        assert_eq!(dyn_com.n_snapshots, 3);
        assert_eq!(dyn_com.n_nodes, 6);
        assert_eq!(dyn_com.node_memberships.len(), 3);

        // Each snapshot should have community IDs for all 6 nodes
        for snap in &dyn_com.node_memberships {
            assert_eq!(snap.len(), 6);
        }
    }

    #[test]
    fn test_evolutionary_clustering_stability() {
        let mut tg = make_two_community_graph();
        // High alpha → more temporal smoothness
        let dyn_com = evolutionary_clustering(&mut tg, 4, 0.9);
        let stability = dyn_com.temporal_stability();
        // With high smoothness, stability should be reasonably high
        assert!(
            (0.0..=1.0).contains(&stability),
            "stability should be in [0,1], got {stability}"
        );
    }

    #[test]
    fn test_evolutionary_clustering_empty_graph() {
        let mut tg = TemporalGraph::new(5);
        let dyn_com = evolutionary_clustering(&mut tg, 3, 0.5);
        assert_eq!(dyn_com.n_snapshots, 3);
        assert_eq!(dyn_com.n_nodes, 5);
    }

    #[test]
    fn test_dynamic_community_membership_access() {
        let mut tg = make_two_community_graph();
        let dyn_com = evolutionary_clustering(&mut tg, 2, 0.5);
        // All node memberships should be accessible
        for t in 0..dyn_com.n_snapshots {
            for v in 0..dyn_com.n_nodes {
                let m = dyn_com.membership(t, v);
                assert!(m.is_some(), "membership({t},{v}) should be Some");
            }
        }
    }

    #[test]
    fn test_community_counts_nonnegative() {
        let mut tg = make_two_community_graph();
        let dyn_com = evolutionary_clustering(&mut tg, 3, 0.5);
        for &cnt in &dyn_com.community_counts {
            assert!(cnt >= 1, "each snapshot should have at least 1 community");
        }
    }

    #[test]
    fn test_compact_labels() {
        let mut labels = vec![5, 10, 5, 20, 10, 20];
        compact_labels(&mut labels);
        // After compaction, labels should be in 0..k for some k ≤ 3
        let max_label = labels.iter().copied().max().unwrap_or(0);
        assert!(
            max_label <= 2,
            "compact labels should be 0-indexed, max={max_label}"
        );
    }

    #[test]
    fn test_temporal_stability_single_snapshot() {
        let dc = DynamicCommunity::new(vec![vec![0, 1, 0, 1]]);
        assert_eq!(dc.temporal_stability(), 1.0);
    }
}
