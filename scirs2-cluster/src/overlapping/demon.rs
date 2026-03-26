//! DEMON: Democratic Estimate of the Modular Organization of a Network.
//!
//! A local, parameter-light overlapping community detection algorithm.
//! For each node v it extracts the ego-network, runs label propagation inside it,
//! and then merges the resulting local communities across all egos using a
//! Jaccard-similarity threshold.
//!
//! Reference: Coscia et al. (2012), "DEMON: A Local-First Discovery Method
//! for Overlapping Communities".

use crate::error::{ClusteringError, Result as ClusterResult};
use std::collections::{HashMap, HashSet};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the DEMON algorithm.
#[derive(Debug, Clone)]
pub struct DemonConfig {
    /// Jaccard similarity threshold used when merging overlapping local communities.
    /// Two communities are merged when |A∩B| / |A∪B| ≥ `merge_threshold`.  Default: 0.35.
    pub merge_threshold: f64,
    /// Communities smaller than this are discarded after merging.  Default: 3.
    pub min_community_size: usize,
    /// Convergence threshold for the ego-network label propagation.  Default: 1e-3.
    pub epsilon: f64,
    /// Maximum number of label-propagation iterations per ego.  Default: 20.
    pub max_lp_iter: usize,
}

impl Default for DemonConfig {
    fn default() -> Self {
        Self {
            merge_threshold: 0.35,
            min_community_size: 3,
            epsilon: 1e-3,
            max_lp_iter: 20,
        }
    }
}

// ─── DEMON struct ─────────────────────────────────────────────────────────────

/// DEMON overlapping community detector.
pub struct Demon {
    config: DemonConfig,
}

impl Demon {
    /// Create a new Demon instance.
    pub fn new(config: DemonConfig) -> Self {
        Self { config }
    }

    /// Detect overlapping communities in the graph represented by `adj`.
    ///
    /// `adj[v]` contains the (undirected) neighbours of node `v`.  Every edge must
    /// appear in both directions.  Returns a list of communities where each community
    /// is a sorted `Vec<usize>` of node indices.
    pub fn detect(&self, adj: &[Vec<usize>]) -> ClusterResult<Vec<Vec<usize>>> {
        let n = adj.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Validate: no neighbour index out of bounds.
        for (u, neighbours) in adj.iter().enumerate() {
            for &v in neighbours {
                if v >= n {
                    return Err(ClusteringError::InvalidInput(format!(
                        "Node {u} has out-of-bounds neighbour {v} (n={n})"
                    )));
                }
            }
        }

        let mut communities: Vec<Vec<usize>> = Vec::new();

        for ego in 0..n {
            let local_comms = self.ego_communities(ego, adj);
            communities.extend(local_comms);
        }

        // Iteratively merge communities whose Jaccard similarity exceeds threshold.
        let merged = self.merge_communities(communities);

        // Filter by minimum size.
        let filtered: Vec<Vec<usize>> = merged
            .into_iter()
            .filter(|c| c.len() >= self.config.min_community_size)
            .collect();

        Ok(filtered)
    }

    // ── Private: ego-network processing ──────────────────────────────────────

    /// Extract local communities from the ego-network of `ego`.
    fn ego_communities(&self, ego: usize, adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
        let neighbours: Vec<usize> = adj[ego].iter().copied().collect();
        if neighbours.is_empty() {
            // Isolated node – single community containing only itself.
            return vec![vec![ego]];
        }

        // Ego-network nodes = neighbours of ego (ego itself excluded from LP).
        // Build a local renaming: ego_node_index → global index.
        let local_nodes: Vec<usize> = neighbours.clone();
        let n_local = local_nodes.len();

        // Reverse mapping: global → local.
        let global_to_local: HashMap<usize, usize> = local_nodes
            .iter()
            .enumerate()
            .map(|(i, &g)| (g, i))
            .collect();

        // Build ego-network adjacency (only edges among neighbours, not to ego).
        let mut ego_adj: Vec<Vec<usize>> = vec![Vec::new(); n_local];
        for (li, &u) in local_nodes.iter().enumerate() {
            for &v in &adj[u] {
                if v == ego {
                    continue; // Exclude ego from LP edges
                }
                if let Some(&lv) = global_to_local.get(&v) {
                    ego_adj[li].push(lv);
                }
            }
        }

        // Label propagation inside the ego-network.
        let labels = label_propagation_ego(&ego_adj, self.config.max_lp_iter, self.config.epsilon);

        // Group local nodes by label; add ego to every group.
        let mut label_groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (li, &label) in labels.iter().enumerate() {
            label_groups.entry(label).or_default().push(local_nodes[li]);
        }

        // Build final communities: each group + ego.
        label_groups
            .into_values()
            .map(|mut members| {
                members.push(ego);
                members.sort_unstable();
                members.dedup();
                members
            })
            .collect()
    }

    // ── Private: community merging ────────────────────────────────────────────

    /// Merge communities greedily using the Jaccard threshold.
    ///
    /// Uses a union-find structure: whenever two communities have Jaccard ≥ threshold
    /// we merge them.  Iterates until no further merges occur.
    fn merge_communities(&self, mut communities: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        let threshold = self.config.merge_threshold;

        loop {
            let m = communities.len();
            if m < 2 {
                break;
            }
            let mut merged_flag = false;
            let mut merged_into: Vec<Option<usize>> = vec![None; m]; // i → index it was merged into

            'outer: for i in 0..m {
                if merged_into[i].is_some() {
                    continue;
                }
                for j in (i + 1)..m {
                    if merged_into[j].is_some() {
                        continue;
                    }
                    let j_sim = Self::jaccard(&communities[i], &communities[j]);
                    if j_sim >= threshold {
                        // Merge j into i.
                        let cj = communities[j].clone();
                        communities[i].extend_from_slice(&cj);
                        communities[i].sort_unstable();
                        communities[i].dedup();
                        merged_into[j] = Some(i);
                        merged_flag = true;
                        // Restart outer loop after a merge to keep indices valid.
                        break 'outer;
                    }
                }
            }

            if !merged_flag {
                break;
            }

            // Remove merged-away communities.
            communities = communities
                .into_iter()
                .enumerate()
                .filter(|(i, _)| merged_into[*i].is_none())
                .map(|(_, c)| c)
                .collect();
        }

        communities
    }

    // ── Public utility ────────────────────────────────────────────────────────

    /// Compute Jaccard similarity between two sorted slices of node indices.
    pub fn jaccard(a: &[usize], b: &[usize]) -> f64 {
        let set_a: HashSet<usize> = a.iter().copied().collect();
        let set_b: HashSet<usize> = b.iter().copied().collect();
        let inter = set_a.intersection(&set_b).count() as f64;
        let union = set_a.union(&set_b).count() as f64;
        if union == 0.0 {
            return 0.0;
        }
        inter / union
    }
}

// ─── Label propagation ────────────────────────────────────────────────────────

/// Run synchronous label propagation on the ego-network adjacency.
///
/// Each node starts with its own label (equal to its local index).
/// At each step every node adopts the plurality label among its neighbours.
/// Ties are broken by selecting the smallest label.
///
/// Returns the final label assignment (local indices).
pub fn label_propagation_ego(ego_adj: &[Vec<usize>], max_iter: usize, tol: f64) -> Vec<usize> {
    let n = ego_adj.len();
    if n == 0 {
        return Vec::new();
    }

    // Initialise: each node has its own label.
    let mut labels: Vec<usize> = (0..n).collect();

    for _iter in 0..max_iter {
        let old_labels = labels.clone();
        let mut changed = 0usize;

        for u in 0..n {
            if ego_adj[u].is_empty() {
                // Isolated node keeps own label.
                continue;
            }

            // Count label frequencies among neighbours.
            let mut freq: HashMap<usize, usize> = HashMap::new();
            for &v in &ego_adj[u] {
                *freq.entry(old_labels[v]).or_insert(0) += 1;
            }

            // Best label = highest frequency, ties → smallest label.
            let best = freq
                .into_iter()
                .max_by(|(la, fa), (lb, fb)| fa.cmp(fb).then(lb.cmp(la)))
                .map(|(l, _)| l)
                .unwrap_or(old_labels[u]);

            if best != old_labels[u] {
                labels[u] = best;
                changed += 1;
            }
        }

        // Convergence: fraction of nodes that changed < tol.
        if (changed as f64) / (n as f64) < tol {
            break;
        }
    }

    labels
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cliques_with_bridge() -> Vec<Vec<usize>> {
        // Clique A: 0,1,2  Clique B: 2,3,4  (node 2 is bridge)
        let mut adj = vec![vec![]; 5];
        for &(u, v) in &[(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)] {
            adj[u].push(v);
            adj[v].push(u);
        }
        adj
    }

    #[test]
    fn test_demon_two_cliques() {
        let adj = two_cliques_with_bridge();
        let config = DemonConfig {
            merge_threshold: 0.2,
            min_community_size: 2,
            ..Default::default()
        };
        let comms = Demon::new(config)
            .detect(&adj)
            .expect("detect should succeed");
        // Should find at least one non-trivial community.
        assert!(!comms.is_empty(), "should find at least one community");
        for c in &comms {
            assert!(c.len() >= 2, "all communities should have ≥ 2 nodes");
        }
    }

    #[test]
    fn test_demon_min_community_size_filter() {
        let adj = two_cliques_with_bridge();
        // Very large min size should filter everything out.
        let config = DemonConfig {
            min_community_size: 100,
            ..Default::default()
        };
        let comms = Demon::new(config)
            .detect(&adj)
            .expect("detect should succeed");
        assert!(comms.is_empty(), "all communities should be filtered out");
    }

    #[test]
    fn test_demon_jaccard_computation() {
        assert!((Demon::jaccard(&[0, 1, 2], &[1, 2, 3]) - 0.5).abs() < 1e-9);
        assert!((Demon::jaccard(&[0, 1], &[0, 1]) - 1.0).abs() < 1e-9);
        assert!((Demon::jaccard(&[0], &[1]) - 0.0).abs() < 1e-9);
        assert!((Demon::jaccard(&[], &[]) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_demon_empty_graph() {
        let adj: Vec<Vec<usize>> = vec![];
        let comms = Demon::new(DemonConfig::default())
            .detect(&adj)
            .expect("detect empty graph");
        assert!(comms.is_empty());
    }

    #[test]
    fn test_demon_single_node() {
        let adj = vec![vec![]];
        let config = DemonConfig {
            min_community_size: 1,
            ..Default::default()
        };
        let comms = Demon::new(config).detect(&adj).expect("single node detect");
        // Should produce one community containing just node 0.
        assert!(!comms.is_empty());
    }

    #[test]
    fn test_demon_out_of_bounds_error() {
        let adj = vec![vec![5]]; // neighbour 5 is out of bounds for n=1
        let result = Demon::new(DemonConfig::default()).detect(&adj);
        assert!(result.is_err());
    }

    #[test]
    fn test_label_propagation_ego_isolated() {
        // Isolated nodes should keep their own labels.
        let ego_adj = vec![vec![], vec![], vec![]];
        let labels = label_propagation_ego(&ego_adj, 10, 1e-3);
        assert_eq!(labels, vec![0, 1, 2]);
    }

    #[test]
    fn test_label_propagation_ego_triangle() {
        // Connected triangle: all nodes should converge to the same label.
        let ego_adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let labels = label_propagation_ego(&ego_adj, 20, 1e-3);
        // All should have the same label after convergence.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
    }

    #[test]
    fn test_demon_all_communities_valid_nodes() {
        let adj = two_cliques_with_bridge();
        let n = adj.len();
        let comms = Demon::new(DemonConfig {
            min_community_size: 1,
            ..Default::default()
        })
        .detect(&adj)
        .expect("detect should succeed");

        for c in &comms {
            for &node in c {
                assert!(node < n, "community contains invalid node index {node}");
            }
        }
    }
}
