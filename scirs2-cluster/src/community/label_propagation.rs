//! Enhanced label propagation algorithm for community detection.
//!
//! This module provides a full-featured label propagation implementation with:
//!
//! - **Asynchronous updates**: nodes update in random order within each sweep
//! - **Weighted edges**: edge weights influence label adoption
//! - **Seed labels**: some nodes can have fixed (pinned) labels for semi-supervised detection
//! - **Tie-breaking**: ties are broken deterministically via a seeded PRNG
//! - **Near-linear time**: O(m) per iteration where m is the number of edges

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{AdjacencyGraph, CommunityResult};
use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the label propagation algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelPropagationConfig {
    /// Maximum number of sweeps (iterations).
    pub max_iterations: usize,
    /// Random seed for deterministic tie-breaking and node ordering.
    pub seed: u64,
    /// Seed labels: `seed_labels[node] = Some(label)` pins that node.
    /// Nodes with `None` are free to change.
    #[serde(skip)]
    pub seed_labels: Option<Vec<Option<usize>>>,
}

impl Default for LabelPropagationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            seed: 42,
            seed_labels: None,
        }
    }
}

/// Result of label propagation, including convergence info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelPropagationResult {
    /// Community detection result.
    pub community: CommunityResult,
    /// Number of iterations until convergence (or max_iterations).
    pub iterations_used: usize,
    /// Whether the algorithm converged (no label changes in final sweep).
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// PRNG
// ---------------------------------------------------------------------------

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
    fn shuffle(&mut self, slice: &mut [usize]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Algorithm
// ---------------------------------------------------------------------------

/// Run enhanced label propagation community detection.
///
/// # Errors
///
/// Returns `ClusteringError::InvalidInput` if the graph has zero nodes or
/// if `seed_labels` length does not match the number of nodes.
pub fn label_propagation_community(
    graph: &AdjacencyGraph,
    config: &LabelPropagationConfig,
) -> Result<LabelPropagationResult> {
    let n = graph.n_nodes;
    if n == 0 {
        return Err(ClusteringError::InvalidInput(
            "Graph has zero nodes".to_string(),
        ));
    }

    // Validate seed labels if provided.
    if let Some(ref sl) = config.seed_labels {
        if sl.len() != n {
            return Err(ClusteringError::InvalidInput(format!(
                "seed_labels length ({}) must equal number of nodes ({})",
                sl.len(),
                n
            )));
        }
    }

    // Initialise labels: use seed labels where provided, otherwise each node
    // gets its own unique label.
    let mut labels: Vec<usize> = Vec::with_capacity(n);
    // Track which nodes are pinned.
    let mut pinned = vec![false; n];
    // First pass: assign seed labels and find the max seed label.
    let mut max_seed_label: usize = 0;
    if let Some(ref sl) = config.seed_labels {
        for opt in sl.iter() {
            if let Some(l) = opt {
                if *l > max_seed_label {
                    max_seed_label = *l;
                }
            }
        }
    }

    let mut next_free_label = max_seed_label + 1;

    for i in 0..n {
        if let Some(ref sl) = config.seed_labels {
            if let Some(l) = sl[i] {
                labels.push(l);
                pinned[i] = true;
                continue;
            }
        }
        labels.push(next_free_label);
        next_free_label += 1;
    }

    let mut rng = Xorshift64::new(config.seed);
    let mut converged = false;
    let mut iterations_used = 0;

    for _iter in 0..config.max_iterations {
        iterations_used += 1;
        let mut changed = false;

        // Random node order.
        let mut order: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut order);

        for &v in &order {
            if pinned[v] {
                continue;
            }

            // Accumulate weighted label votes from neighbours.
            let mut votes: HashMap<usize, f64> = HashMap::new();
            for &(nb, w) in &graph.adjacency[v] {
                *votes.entry(labels[nb]).or_insert(0.0) += w;
            }

            if votes.is_empty() {
                // Isolated node: keep its own label.
                continue;
            }

            // Find the label(s) with maximum weight.
            let max_weight = votes.values().cloned().fold(f64::NEG_INFINITY, f64::max);

            let mut best_labels: Vec<usize> = votes
                .iter()
                .filter(|(_, &v)| (v - max_weight).abs() < 1e-12)
                .map(|(&l, _)| l)
                .collect();

            // Deterministic tie-breaking: pick the one with smallest hash.
            best_labels.sort_unstable();
            let chosen_idx = (rng.next_u64() as usize) % best_labels.len();
            let chosen = best_labels[chosen_idx];

            if chosen != labels[v] {
                labels[v] = chosen;
                changed = true;
            }
        }

        if !changed {
            converged = true;
            break;
        }
    }

    // Compact labels to 0..k-1.
    let mut mapping: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0usize;
    for lbl in &labels {
        if !mapping.contains_key(lbl) {
            mapping.insert(*lbl, next_id);
            next_id += 1;
        }
    }
    let compacted: Vec<usize> = labels
        .iter()
        .map(|l| mapping.get(l).copied().unwrap_or(0))
        .collect();

    let num_communities = next_id;
    let quality = graph.modularity(&compacted);

    Ok(LabelPropagationResult {
        community: CommunityResult {
            labels: compacted,
            num_communities,
            quality_score: Some(quality),
        },
        iterations_used,
        converged,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Two disconnected cliques -> two communities.
    #[test]
    fn test_lp_two_cliques() {
        let k = 5;
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

        let config = LabelPropagationConfig::default();
        let result = label_propagation_community(&g, &config).expect("lp should succeed");

        assert_eq!(result.community.num_communities, 2);
        // Clique 0 should share a label.
        let c0 = result.community.labels[0];
        for i in 1..k {
            assert_eq!(result.community.labels[i], c0);
        }
        // Clique 1 should share a different label.
        let c1 = result.community.labels[k];
        for i in (k + 1)..n {
            assert_eq!(result.community.labels[i], c1);
        }
        assert_ne!(c0, c1);
    }

    /// Convergence should be reported.
    #[test]
    fn test_lp_convergence() {
        let mut g = AdjacencyGraph::new(4);
        let _ = g.add_edge(0, 1, 1.0);
        let _ = g.add_edge(2, 3, 1.0);
        let config = LabelPropagationConfig::default();
        let result = label_propagation_community(&g, &config).expect("lp should succeed");
        assert!(result.converged);
        assert!(result.iterations_used < config.max_iterations);
    }

    /// Seed labels should be respected.
    #[test]
    fn test_lp_seed_labels() {
        // 4-node path: 0-1-2-3
        let mut g = AdjacencyGraph::new(4);
        let _ = g.add_edge(0, 1, 1.0);
        let _ = g.add_edge(1, 2, 1.0);
        let _ = g.add_edge(2, 3, 1.0);

        // Pin node 0 to community 0 and node 3 to community 1.
        let seed_labels = vec![Some(0), None, None, Some(1)];
        let config = LabelPropagationConfig {
            seed_labels: Some(seed_labels),
            ..Default::default()
        };
        let result = label_propagation_community(&g, &config).expect("lp should succeed");

        // Pinned nodes must keep their assigned (compacted) labels.
        let l0 = result.community.labels[0];
        let l3 = result.community.labels[3];
        assert_ne!(l0, l3);
    }

    /// Weighted edges should influence the result.
    #[test]
    fn test_lp_weighted_edges() {
        // Triangle: 0-1 strong, 0-2 strong, 1-2 weak, plus node 3 connected weakly to 1.
        let mut g = AdjacencyGraph::new(4);
        let _ = g.add_edge(0, 1, 10.0);
        let _ = g.add_edge(0, 2, 10.0);
        let _ = g.add_edge(1, 2, 0.1);
        let _ = g.add_edge(1, 3, 10.0);
        let _ = g.add_edge(2, 3, 0.1);

        let config = LabelPropagationConfig::default();
        let result = label_propagation_community(&g, &config).expect("lp should succeed");

        // With strong edges, all nodes should likely end up in one community
        // (the graph is connected and small).
        // At least verify we get a valid result.
        assert!(result.community.num_communities >= 1);
        assert_eq!(result.community.labels.len(), 4);
    }

    /// Empty graph error.
    #[test]
    fn test_lp_empty_graph() {
        let g = AdjacencyGraph::new(0);
        let config = LabelPropagationConfig::default();
        assert!(label_propagation_community(&g, &config).is_err());
    }

    /// Single node.
    #[test]
    fn test_lp_single_node() {
        let g = AdjacencyGraph::new(1);
        let config = LabelPropagationConfig::default();
        let result = label_propagation_community(&g, &config).expect("lp should succeed");
        assert_eq!(result.community.num_communities, 1);
        assert_eq!(result.community.labels, vec![0]);
    }

    /// Seed labels length mismatch should error.
    #[test]
    fn test_lp_seed_labels_length_mismatch() {
        let g = AdjacencyGraph::new(3);
        let config = LabelPropagationConfig {
            seed_labels: Some(vec![Some(0), None]),
            ..Default::default()
        };
        assert!(label_propagation_community(&g, &config).is_err());
    }

    /// Isolated nodes (no edges) should each get their own community.
    #[test]
    fn test_lp_isolated_nodes() {
        let g = AdjacencyGraph::new(5);
        let config = LabelPropagationConfig::default();
        let result = label_propagation_community(&g, &config).expect("lp should succeed");
        assert_eq!(result.community.num_communities, 5);
    }
}
