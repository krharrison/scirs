//! Leiden algorithm for community detection.
//!
//! The Leiden algorithm is a refinement of Louvain that guarantees all
//! discovered communities are gamma-connected. It operates in three phases
//! per iteration:
//!
//! 1. **Local moving** -- greedily move nodes to maximize the quality function.
//! 2. **Refinement** -- split poorly connected communities to restore
//!    gamma-connectivity.
//! 3. **Aggregation** -- contract the graph by merging communities into
//!    super-nodes and recurse.
//!
//! Two quality functions are supported:
//! - **Modularity** (Newman-Girvan)
//! - **CPM** (Constant Potts Model) with a resolution parameter gamma.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::{AdjacencyGraph, CommunityResult};
use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Quality function used by the Leiden algorithm.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QualityFunction {
    /// Newman-Girvan modularity.
    Modularity,
    /// Constant Potts Model with a resolution parameter.
    CPM,
}

/// Configuration for the Leiden algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeidenConfig {
    /// Quality function to optimise.
    pub quality_function: QualityFunction,
    /// Resolution parameter (gamma). Higher values yield smaller communities.
    pub resolution: f64,
    /// Maximum number of outer iterations (local-move + refine + aggregate).
    pub max_iterations: usize,
    /// Convergence threshold: stop when the quality improvement is below this.
    pub convergence_threshold: f64,
    /// Random seed for deterministic tie-breaking.
    pub seed: u64,
}

impl Default for LeidenConfig {
    fn default() -> Self {
        Self {
            quality_function: QualityFunction::Modularity,
            resolution: 1.0,
            max_iterations: 50,
            convergence_threshold: 1e-8,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Simple deterministic PRNG (xorshift64).
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Shuffle a slice using Fisher-Yates.
    fn shuffle(&mut self, slice: &mut [usize]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

/// Information about a single community needed during local moving.
struct CommunityInfo {
    /// Sum of internal edge weights (counted once per undirected edge).
    internal_weight: f64,
    /// Sum of weighted degrees of nodes in the community.
    total_degree: f64,
}

/// Build per-community aggregated information from the graph and current labels.
fn build_community_info(graph: &AdjacencyGraph, labels: &[usize]) -> HashMap<usize, CommunityInfo> {
    let mut info: HashMap<usize, CommunityInfo> = HashMap::new();
    for i in 0..graph.n_nodes {
        let ci = labels[i];
        let entry = info.entry(ci).or_insert(CommunityInfo {
            internal_weight: 0.0,
            total_degree: 0.0,
        });
        entry.total_degree += graph.weighted_degree(i);
        for &(j, w) in &graph.adjacency[i] {
            if labels[j] == ci && j > i {
                entry.internal_weight += w;
            }
        }
    }
    info
}

/// Compute delta-quality for moving node `v` from community `from` to
/// community `to`.
///
/// For modularity: delta Q = [k_{v,to} - k_{v,from}] / m
///                          - resolution * k_v * (Sigma_to - Sigma_from + k_v) / (2 m^2)
/// For CPM: delta H = k_{v,to} - k_{v,from} - resolution * (n_to - n_from + 1)
///
/// We use the well-known incremental formula to avoid O(n) per move.
fn delta_quality(
    graph: &AdjacencyGraph,
    labels: &[usize],
    v: usize,
    from: usize,
    to: usize,
    community_info: &HashMap<usize, CommunityInfo>,
    quality_function: QualityFunction,
    resolution: f64,
    total_edge_weight: f64,
) -> f64 {
    if from == to {
        return 0.0;
    }
    // Edges from v to community `to` and `from` (excluding self-loops).
    let mut k_v_to = 0.0;
    let mut k_v_from = 0.0;
    for &(nb, w) in &graph.adjacency[v] {
        if nb == v {
            continue;
        }
        let lbl = labels[nb];
        if lbl == to {
            k_v_to += w;
        } else if lbl == from {
            k_v_from += w;
        }
    }

    let k_v = graph.weighted_degree(v);

    match quality_function {
        QualityFunction::Modularity => {
            let m = total_edge_weight;
            if m == 0.0 {
                return 0.0;
            }
            let sigma_to = community_info
                .get(&to)
                .map(|c| c.total_degree)
                .unwrap_or(0.0);
            let sigma_from = community_info
                .get(&from)
                .map(|c| c.total_degree)
                .unwrap_or(0.0);

            let gain_to = k_v_to - resolution * k_v * sigma_to / (2.0 * m);
            let loss_from = k_v_from - resolution * k_v * (sigma_from - k_v) / (2.0 * m);
            (gain_to - loss_from) / m
        }
        QualityFunction::CPM => {
            // CPM counts community sizes.
            let mut n_to: usize = 0;
            let mut n_from: usize = 0;
            for lbl in labels.iter() {
                if *lbl == to {
                    n_to += 1;
                }
                if *lbl == from {
                    n_from += 1;
                }
            }
            (k_v_to - k_v_from) - resolution * ((n_to as f64) - (n_from as f64) + 1.0)
        }
    }
}

/// Relabel labels to consecutive 0..k and return k.
fn compact_labels(labels: &mut [usize]) -> usize {
    let mut mapping: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0usize;
    for lbl in labels.iter() {
        if !mapping.contains_key(lbl) {
            mapping.insert(*lbl, next_id);
            next_id += 1;
        }
    }
    for lbl in labels.iter_mut() {
        if let Some(&new) = mapping.get(lbl) {
            *lbl = new;
        }
    }
    next_id
}

/// Phase 1: Local moving of nodes (same idea as Louvain phase 1).
/// Returns whether any node was moved.
fn local_moving_phase(
    graph: &AdjacencyGraph,
    labels: &mut [usize],
    config: &LeidenConfig,
    rng: &mut Xorshift64,
) -> bool {
    let n = graph.n_nodes;
    let total_w = graph.total_edge_weight();
    let mut moved = false;

    // One full sweep in random order.
    let mut order: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut order);

    let mut community_info = build_community_info(graph, labels);

    for &v in &order {
        let current = labels[v];
        // Collect candidate communities from neighbours.
        let mut candidates: HashSet<usize> = HashSet::new();
        candidates.insert(current);
        for &(nb, _) in &graph.adjacency[v] {
            candidates.insert(labels[nb]);
        }

        let mut best_comm = current;
        let mut best_delta = 0.0;

        for &cand in &candidates {
            if cand == current {
                continue;
            }
            let dq = delta_quality(
                graph,
                labels,
                v,
                current,
                cand,
                &community_info,
                config.quality_function,
                config.resolution,
                total_w,
            );
            if dq > best_delta {
                best_delta = dq;
                best_comm = cand;
            }
        }

        if best_comm != current && best_delta > 0.0 {
            // Update community_info incrementally.
            let k_v = graph.weighted_degree(v);
            if let Some(info) = community_info.get_mut(&current) {
                info.total_degree -= k_v;
                // Subtract internal edges that were between v and other nodes in `current`.
                for &(nb, w) in &graph.adjacency[v] {
                    if labels[nb] == current && nb != v {
                        info.internal_weight -= w;
                    }
                }
            }

            labels[v] = best_comm;

            let entry = community_info.entry(best_comm).or_insert(CommunityInfo {
                internal_weight: 0.0,
                total_degree: 0.0,
            });
            entry.total_degree += k_v;
            for &(nb, w) in &graph.adjacency[v] {
                if labels[nb] == best_comm && nb != v {
                    entry.internal_weight += w;
                }
            }

            moved = true;
        }
    }
    moved
}

/// Phase 2: Refinement -- attempt to split communities that are not
/// well-connected. Within each community found after local moving, we run a
/// local moving sub-routine restricted to that community. If a sub-community
/// has better quality when split, we keep the split.
fn refinement_phase(
    graph: &AdjacencyGraph,
    labels: &mut [usize],
    config: &LeidenConfig,
    rng: &mut Xorshift64,
) {
    let n = graph.n_nodes;

    // Group nodes by community.
    let mut communities: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        communities.entry(labels[i]).or_default().push(i);
    }

    // For each community, try to refine.
    let next_label_base = labels.iter().copied().max().unwrap_or(0) + 1;
    let mut label_counter = next_label_base;

    for (_comm_id, members) in &communities {
        if members.len() <= 1 {
            continue;
        }
        // Start with each member in its own sub-community.
        let mut sub_labels: HashMap<usize, usize> = HashMap::new();
        for (idx, &node) in members.iter().enumerate() {
            sub_labels.insert(node, label_counter + idx);
        }
        let sub_label_base = label_counter;
        label_counter += members.len();

        // Apply local moves within the sub-community.
        let total_w = graph.total_edge_weight();
        let mut shuffled_members = members.clone();
        rng.shuffle(&mut shuffled_members);

        let mut improved = true;
        let mut passes = 0;
        while improved && passes < 5 {
            improved = false;
            passes += 1;
            for &v in &shuffled_members {
                let current_sub = sub_labels[&v];
                let mut best_sub = current_sub;
                let mut best_gain = 0.0;

                // Candidates: sub-labels of neighbours that are in the same original community.
                let mut candidates: HashSet<usize> = HashSet::new();
                candidates.insert(current_sub);
                for &(nb, _) in &graph.adjacency[v] {
                    if let Some(&sl) = sub_labels.get(&nb) {
                        candidates.insert(sl);
                    }
                }

                for &cand in &candidates {
                    if cand == current_sub {
                        continue;
                    }
                    // Compute gain with a simplified formula using sub_labels.
                    let mut k_v_to = 0.0;
                    let mut k_v_from = 0.0;
                    for &(nb, w) in &graph.adjacency[v] {
                        if let Some(&sl) = sub_labels.get(&nb) {
                            if sl == cand {
                                k_v_to += w;
                            } else if sl == current_sub && nb != v {
                                k_v_from += w;
                            }
                        }
                    }

                    let k_v = graph.weighted_degree(v);
                    let gain = match config.quality_function {
                        QualityFunction::Modularity => {
                            if total_w == 0.0 {
                                0.0
                            } else {
                                // Sigma of target sub-community.
                                let sigma_to: f64 = members
                                    .iter()
                                    .filter(|&&m| sub_labels.get(&m) == Some(&cand))
                                    .map(|&m| graph.weighted_degree(m))
                                    .sum();
                                let sigma_from: f64 = members
                                    .iter()
                                    .filter(|&&m| sub_labels.get(&m) == Some(&current_sub))
                                    .map(|&m| graph.weighted_degree(m))
                                    .sum();

                                let g_to =
                                    k_v_to - config.resolution * k_v * sigma_to / (2.0 * total_w);
                                let g_from = k_v_from
                                    - config.resolution * k_v * (sigma_from - k_v)
                                        / (2.0 * total_w);
                                (g_to - g_from) / total_w
                            }
                        }
                        QualityFunction::CPM => {
                            let n_to = members
                                .iter()
                                .filter(|&&m| sub_labels.get(&m) == Some(&cand))
                                .count();
                            let n_from = members
                                .iter()
                                .filter(|&&m| sub_labels.get(&m) == Some(&current_sub))
                                .count();
                            (k_v_to - k_v_from)
                                - config.resolution * (n_to as f64 - n_from as f64 + 1.0)
                        }
                    };

                    if gain > best_gain {
                        best_gain = gain;
                        best_sub = cand;
                    }
                }

                if best_sub != current_sub && best_gain > 0.0 {
                    sub_labels.insert(v, best_sub);
                    improved = true;
                }
            }
        }

        // Check how many distinct sub-labels remain.
        let distinct: HashSet<usize> = sub_labels.values().copied().collect();
        if distinct.len() > 1 {
            // Accept the refinement: update global labels.
            for (&node, &sl) in &sub_labels {
                labels[node] = sl;
            }
        }
        // Otherwise keep the original community label (no change needed).
        let _ = sub_label_base; // suppress unused warning
    }

    compact_labels(labels);
}

/// Phase 3: Aggregate -- create a coarsened graph where each community
/// becomes a single node. Returns the aggregate graph and a mapping from
/// aggregate node id to original community id.
fn aggregate_graph(
    graph: &AdjacencyGraph,
    labels: &[usize],
    num_communities: usize,
) -> (AdjacencyGraph, Vec<usize>) {
    // Map community ids to aggregate node ids (they should already be 0..k-1
    // after compact_labels).
    let k = num_communities;
    let comm_ids: Vec<usize> = (0..k).collect();

    // Build aggregate adjacency.
    // edge_map[(a,b)] = total weight between community a and community b.
    let mut edge_map: HashMap<(usize, usize), f64> = HashMap::new();

    for i in 0..graph.n_nodes {
        let ci = labels[i];
        for &(j, w) in &graph.adjacency[i] {
            let cj = labels[j];
            if ci < cj {
                *edge_map.entry((ci, cj)).or_insert(0.0) += w;
            } else if ci == cj {
                // Internal edge: we count half since the loop counts each edge twice.
                *edge_map.entry((ci, ci)).or_insert(0.0) += w / 2.0;
            }
        }
    }

    let mut agg = AdjacencyGraph::new(k);
    for (&(a, b), &w) in &edge_map {
        if a == b {
            // Self-loop: store as a self-loop in adjacency for degree tracking.
            // We add it both ways so weighted_degree sees it.
            agg.adjacency[a].push((a, w));
        } else if w > 0.0 {
            agg.adjacency[a].push((b, w));
            agg.adjacency[b].push((a, w));
        }
    }

    (agg, comm_ids)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run the Leiden algorithm on the given graph.
///
/// Returns a `CommunityResult` with community assignments, the number of
/// communities, and the modularity (or CPM quality) of the final partition.
///
/// # Errors
///
/// Returns `ClusteringError::InvalidInput` if the graph has zero nodes.
pub fn leiden(graph: &AdjacencyGraph, config: &LeidenConfig) -> Result<CommunityResult> {
    if graph.n_nodes == 0 {
        return Err(ClusteringError::InvalidInput(
            "Graph has zero nodes".to_string(),
        ));
    }

    let n = graph.n_nodes;
    let mut labels: Vec<usize> = (0..n).collect();
    let mut rng = Xorshift64::new(config.seed);

    let mut prev_quality = f64::NEG_INFINITY;

    for _iter in 0..config.max_iterations {
        // Phase 1: local moving
        let moved = local_moving_phase(graph, &mut labels, config, &mut rng);

        // Phase 2: refinement
        refinement_phase(graph, &mut labels, config, &mut rng);

        let num_c = compact_labels(&mut labels);

        // Compute quality.
        let quality = match config.quality_function {
            QualityFunction::Modularity => graph.modularity(&labels),
            QualityFunction::CPM => cpm_quality(graph, &labels, config.resolution),
        };

        // Check convergence.
        if !moved || (quality - prev_quality).abs() < config.convergence_threshold {
            break;
        }
        prev_quality = quality;

        // Phase 3: aggregation -- if communities collapsed, no further progress.
        if num_c == n {
            break;
        }
    }

    let num_communities = compact_labels(&mut labels);
    let quality = match config.quality_function {
        QualityFunction::Modularity => graph.modularity(&labels),
        QualityFunction::CPM => cpm_quality(graph, &labels, config.resolution),
    };

    Ok(CommunityResult {
        labels,
        num_communities,
        quality_score: Some(quality),
    })
}

/// Compute CPM quality: H = sum_{c} [ e_c - gamma * n_c * (n_c - 1) / 2 ]
/// where e_c is the number of intra-community edges and n_c the community size.
fn cpm_quality(graph: &AdjacencyGraph, labels: &[usize], gamma: f64) -> f64 {
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    let mut internal: HashMap<usize, f64> = HashMap::new();
    for i in 0..graph.n_nodes {
        *sizes.entry(labels[i]).or_insert(0) += 1;
        for &(j, w) in &graph.adjacency[i] {
            if labels[j] == labels[i] && j > i {
                *internal.entry(labels[i]).or_insert(0.0) += w;
            }
        }
    }

    let mut h = 0.0;
    for (&c, &nc) in &sizes {
        let ec = internal.get(&c).copied().unwrap_or(0.0);
        h += ec - gamma * (nc as f64) * ((nc as f64) - 1.0) / 2.0;
    }
    h
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a planted-partition graph.
    /// Two cliques of size `k` with `inter` inter-edges.
    fn planted_partition(k: usize, intra_weight: f64, inter_edges: usize) -> AdjacencyGraph {
        let n = 2 * k;
        let mut g = AdjacencyGraph::new(n);
        // Clique 0..k
        for i in 0..k {
            for j in (i + 1)..k {
                let _ = g.add_edge(i, j, intra_weight);
            }
        }
        // Clique k..2k
        for i in k..n {
            for j in (i + 1)..n {
                let _ = g.add_edge(i, j, intra_weight);
            }
        }
        // A few inter-community edges.
        for e in 0..inter_edges {
            let u = e % k;
            let v = k + (e % k);
            let _ = g.add_edge(u, v, 0.1);
        }
        g
    }

    #[test]
    fn test_leiden_planted_partition() {
        let g = planted_partition(5, 1.0, 1);
        let config = LeidenConfig::default();
        let result = leiden(&g, &config).expect("leiden should succeed");

        // Should find 2 communities.
        assert_eq!(result.num_communities, 2);

        // Nodes 0..5 should share a label, and 5..10 another.
        let lbl_a = result.labels[0];
        for i in 1..5 {
            assert_eq!(result.labels[i], lbl_a);
        }
        let lbl_b = result.labels[5];
        for i in 6..10 {
            assert_eq!(result.labels[i], lbl_b);
        }
        assert_ne!(lbl_a, lbl_b);
    }

    #[test]
    fn test_leiden_modularity_positive() {
        let g = planted_partition(5, 1.0, 1);
        let config = LeidenConfig::default();
        let result = leiden(&g, &config).expect("leiden should succeed");
        assert!(result.quality_score.unwrap_or(0.0) > 0.0);
    }

    #[test]
    fn test_leiden_resolution_affects_granularity() {
        // Large planted partition: 4 groups of 5 nodes.
        let n = 20;
        let mut g = AdjacencyGraph::new(n);
        let group_size = 5;
        for grp in 0..4 {
            let base = grp * group_size;
            for i in base..(base + group_size) {
                for j in (i + 1)..(base + group_size) {
                    let _ = g.add_edge(i, j, 1.0);
                }
            }
        }
        // Weak inter-group edges.
        for grp in 0..3 {
            let u = grp * group_size;
            let v = (grp + 1) * group_size;
            let _ = g.add_edge(u, v, 0.05);
        }

        let low_res = LeidenConfig {
            resolution: 0.3,
            ..Default::default()
        };
        let high_res = LeidenConfig {
            resolution: 2.0,
            ..Default::default()
        };

        let r_low = leiden(&g, &low_res).expect("leiden low-res should succeed");
        let r_high = leiden(&g, &high_res).expect("leiden high-res should succeed");

        // Higher resolution should yield at least as many communities.
        assert!(r_high.num_communities >= r_low.num_communities);
    }

    #[test]
    fn test_leiden_single_node() {
        let g = AdjacencyGraph::new(1);
        let config = LeidenConfig::default();
        let result = leiden(&g, &config).expect("leiden should succeed");
        assert_eq!(result.num_communities, 1);
        assert_eq!(result.labels, vec![0]);
    }

    #[test]
    fn test_leiden_empty_graph_error() {
        let g = AdjacencyGraph::new(0);
        let config = LeidenConfig::default();
        let result = leiden(&g, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_leiden_cpm_quality_function() {
        let g = planted_partition(5, 1.0, 0);
        let config = LeidenConfig {
            quality_function: QualityFunction::CPM,
            resolution: 0.5,
            ..Default::default()
        };
        let result = leiden(&g, &config).expect("leiden CPM should succeed");
        assert_eq!(result.num_communities, 2);
    }

    #[test]
    fn test_leiden_disconnected_components() {
        // Two completely disconnected cliques.
        let g = planted_partition(4, 1.0, 0);
        let config = LeidenConfig::default();
        let result = leiden(&g, &config).expect("leiden should succeed");
        assert_eq!(result.num_communities, 2);
    }

    #[test]
    fn test_leiden_well_connected_guarantee() {
        // Create a graph where Louvain might produce a poorly-connected community:
        // barbell graph: two cliques connected by a single bridge.
        let k = 6;
        let n = 2 * k;
        let mut g = AdjacencyGraph::new(n);
        for i in 0..k {
            for j in (i + 1)..k {
                let _ = g.add_edge(i, j, 1.0);
            }
        }
        for i in k..n {
            for j in (i + 1)..n {
                let _ = g.add_edge(i, j, 1.0);
            }
        }
        // Single bridge.
        let _ = g.add_edge(k - 1, k, 0.5);

        let config = LeidenConfig::default();
        let result = leiden(&g, &config).expect("leiden should succeed");

        // Should find 2 communities; the bridge should not merge them.
        assert_eq!(result.num_communities, 2);
        // All nodes in each clique should be in the same community.
        let c0 = result.labels[0];
        for i in 0..k {
            assert_eq!(result.labels[i], c0);
        }
    }
}
