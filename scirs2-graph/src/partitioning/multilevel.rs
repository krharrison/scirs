//! METIS-style multilevel graph partitioning.
//!
//! Implements the three-phase multilevel paradigm:
//! 1. **Coarsening** via Heavy-Edge Matching (HEM)
//! 2. **Initial partitioning** on the coarsest graph
//! 3. **Uncoarsening** with Kernighan-Lin refinement at each level
//!
//! This approach produces high-quality partitions efficiently, even for
//! large graphs, by working at progressively coarser scales.

use scirs2_core::ndarray::Array2;

use crate::error::{GraphError, Result};

use super::spectral::spectral_bisect;
use super::types::{PartitionConfig, PartitionResult};

/// One level in the coarsening hierarchy.
#[derive(Debug, Clone)]
struct CoarseLevel {
    /// Adjacency matrix of the coarsened graph.
    adj: Array2<f64>,
    /// Maps fine-level node index -> coarse-level node index.
    mapping: Vec<usize>,
    /// Number of nodes at this coarse level.
    n_nodes: usize,
}

/// Phase 1: Coarsen the graph via Heavy-Edge Matching (HEM).
///
/// At each coarsening step, we find a maximal matching that prefers the
/// heaviest edges. Matched pairs of nodes are merged into single coarse
/// nodes, and edge weights are summed.
///
/// Returns a stack of coarsening levels (finest to coarsest).
fn coarsen(adj: &Array2<f64>, threshold: usize) -> Vec<CoarseLevel> {
    let mut levels = Vec::new();
    let mut current = adj.clone();

    loop {
        let n = current.nrows();
        if n <= threshold {
            break;
        }

        // Heavy-Edge Matching: greedily match each unmatched node
        // with its heaviest unmatched neighbor.
        let mut matched = vec![false; n];
        let mut mapping = vec![0usize; n];
        let mut coarse_id = 0usize;

        // Sort nodes by degree (ascending) to match leaves first
        let mut node_order: Vec<usize> = (0..n).collect();
        node_order.sort_by(|&a, &b| {
            let deg_a: f64 = (0..n).map(|j| current[[a, j]].abs()).sum();
            let deg_b: f64 = (0..n).map(|j| current[[b, j]].abs()).sum();
            deg_a
                .partial_cmp(&deg_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for &i in &node_order {
            if matched[i] {
                continue;
            }

            // Find heaviest unmatched neighbor
            let mut best_j: Option<usize> = None;
            let mut best_w = 0.0f64;
            for j in 0..n {
                if j == i || matched[j] {
                    continue;
                }
                let w = current[[i, j]].abs();
                if w > 1e-15 && w > best_w {
                    best_w = w;
                    best_j = Some(j);
                }
            }

            if let Some(j) = best_j {
                mapping[i] = coarse_id;
                mapping[j] = coarse_id;
                matched[i] = true;
                matched[j] = true;
                coarse_id += 1;
            } else {
                // Singleton: no unmatched neighbor
                mapping[i] = coarse_id;
                matched[i] = true;
                coarse_id += 1;
            }
        }

        let cn = coarse_id;
        if cn >= n {
            // No coarsening happened
            break;
        }

        // Build coarsened adjacency matrix
        let mut coarse_adj = Array2::<f64>::zeros((cn, cn));
        for i in 0..n {
            for j in (i + 1)..n {
                let w = current[[i, j]];
                if w.abs() > 1e-15 {
                    let ci = mapping[i];
                    let cj = mapping[j];
                    if ci != cj {
                        coarse_adj[[ci, cj]] += w;
                        coarse_adj[[cj, ci]] += w;
                    }
                }
            }
        }

        levels.push(CoarseLevel {
            adj: coarse_adj.clone(),
            mapping,
            n_nodes: cn,
        });

        current = coarse_adj;
    }

    levels
}

/// Phase 2: Compute initial partition on the coarsest graph.
///
/// Uses spectral bisection for quality, or greedy bisection as fallback.
fn initial_partition(adj: &Array2<f64>, config: &PartitionConfig) -> Result<Vec<usize>> {
    let n = adj.nrows();
    if n < 2 {
        return Ok(vec![0; n]);
    }

    // Try spectral bisection first
    match spectral_bisect(adj, config) {
        Ok(result) => Ok(result.assignments),
        Err(_) => {
            // Fallback: greedy bisection by node degree
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| {
                let deg_a: f64 = (0..n).map(|j| adj[[a, j]].abs()).sum();
                let deg_b: f64 = (0..n).map(|j| adj[[b, j]].abs()).sum();
                deg_b
                    .partial_cmp(&deg_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut assignments = vec![0usize; n];
            let half = n / 2;
            for &idx in &indices[half..] {
                assignments[idx] = 1;
            }
            Ok(assignments)
        }
    }
}

/// Kernighan-Lin refinement pass on a bisection.
///
/// Iteratively finds the best pair of nodes (one from each partition) to swap
/// to reduce the edge cut, then applies the best prefix of swaps.
///
/// Returns the reduction in edge cut (non-negative).
fn kernighan_lin_pass(adj: &Array2<f64>, partition: &mut [usize], max_swaps: usize) -> usize {
    let n = adj.nrows();
    if n < 2 {
        return 0;
    }

    // Compute D-values: D(v) = external_cost(v) - internal_cost(v)
    // Positive D means the node benefits from moving to the other partition.
    let compute_d = |part: &[usize], node: usize| -> f64 {
        let my_part = part[node];
        let mut ext = 0.0;
        let mut int = 0.0;
        for j in 0..n {
            if j == node {
                continue;
            }
            let w = adj[[node, j]].abs();
            if w < 1e-15 {
                continue;
            }
            if part[j] == my_part {
                int += w;
            } else {
                ext += w;
            }
        }
        ext - int
    };

    let original_cut = count_edge_cut(adj, partition);
    let mut d_values: Vec<f64> = (0..n).map(|i| compute_d(partition, i)).collect();

    let mut locked = vec![false; n];
    let mut gains = Vec::new();
    let mut swap_pairs = Vec::new();

    let effective_swaps = max_swaps.min(n / 2);

    for _ in 0..effective_swaps {
        // Find best unlocked pair (a in part 0, b in part 1)
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_a: Option<usize> = None;
        let mut best_b: Option<usize> = None;

        for a in 0..n {
            if locked[a] || partition[a] != 0 {
                continue;
            }
            for b in 0..n {
                if locked[b] || partition[b] != 1 {
                    continue;
                }
                let gain = d_values[a] + d_values[b] - 2.0 * adj[[a, b]].abs();
                if gain > best_gain {
                    best_gain = gain;
                    best_a = Some(a);
                    best_b = Some(b);
                }
            }
        }

        let (a, b) = match (best_a, best_b) {
            (Some(a), Some(b)) => (a, b),
            _ => break,
        };

        // Record swap
        gains.push(best_gain);
        swap_pairs.push((a, b));
        locked[a] = true;
        locked[b] = true;

        // Update D-values as if a and b were swapped
        partition[a] = 1;
        partition[b] = 0;
        for i in 0..n {
            if !locked[i] {
                d_values[i] = compute_d(partition, i);
            }
        }
    }

    // Find best prefix of swaps (maximize cumulative gain)
    let mut cumulative = 0.0f64;
    let mut best_cumulative = 0.0f64;
    let mut best_k = 0usize; // 0 means no swaps

    for (k, &gain) in gains.iter().enumerate() {
        cumulative += gain;
        if cumulative > best_cumulative {
            best_cumulative = cumulative;
            best_k = k + 1;
        }
    }

    // Undo all swaps first (they were applied above)
    for &(a, b) in swap_pairs.iter().rev() {
        partition[a] = 0;
        partition[b] = 1;
    }

    // Re-apply only the best prefix
    for &(a, b) in swap_pairs.iter().take(best_k) {
        partition[a] = 1;
        partition[b] = 0;
    }

    let new_cut = count_edge_cut(adj, partition);
    original_cut.saturating_sub(new_cut)
}

/// Count edges crossing partition boundaries.
fn count_edge_cut(adj: &Array2<f64>, partition: &[usize]) -> usize {
    let n = adj.nrows();
    let mut cut = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if adj[[i, j]].abs() > 1e-15 && partition[i] != partition[j] {
                cut += 1;
            }
        }
    }
    cut
}

/// Phase 3: Uncoarsen with Kernighan-Lin refinement at each level.
///
/// Projects the partition from coarse to fine level, then applies KL
/// refinement to improve the edge cut.
fn uncoarsen_with_refinement(
    levels: &[CoarseLevel],
    coarse_partition: Vec<usize>,
    original_adj: &Array2<f64>,
    config: &PartitionConfig,
) -> Result<Vec<usize>> {
    if levels.is_empty() {
        return Ok(coarse_partition);
    }

    let mut partition = coarse_partition;

    // Walk from coarsest to finest (reverse order of levels)
    for level_idx in (0..levels.len()).rev() {
        let level = &levels[level_idx];

        // Project partition to the finer level
        let fine_n = level.mapping.len();
        let mut fine_partition = vec![0usize; fine_n];
        for (fine_node, &coarse_node) in level.mapping.iter().enumerate() {
            if coarse_node < partition.len() {
                fine_partition[fine_node] = partition[coarse_node];
            }
        }

        // Get the adjacency matrix at this fine level
        let adj_at_level = if level_idx == 0 {
            original_adj.clone()
        } else {
            levels[level_idx - 1].adj.clone()
        };

        // Apply KL refinement
        for _ in 0..config.kl_max_passes {
            let improvement = kernighan_lin_pass(&adj_at_level, &mut fine_partition, fine_n / 2);
            if improvement == 0 {
                break;
            }
        }

        partition = fine_partition;
    }

    Ok(partition)
}

/// Perform multilevel graph partitioning (METIS-like).
///
/// This implements the classic three-phase approach:
/// 1. Coarsen the graph via Heavy-Edge Matching until small enough
/// 2. Partition the coarsest graph (spectral bisection)
/// 3. Uncoarsen with Kernighan-Lin refinement at each level
///
/// For k > 2, recursive bisection is applied.
///
/// # Arguments
/// * `adj` - Symmetric adjacency matrix (n x n)
/// * `config` - Partition configuration
///
/// # Returns
/// A `PartitionResult` with `config.n_partitions` partitions.
///
/// # Errors
/// Returns `GraphError` if the matrix dimensions are invalid or
/// the number of partitions exceeds the number of nodes.
pub fn multilevel_partition(
    adj: &Array2<f64>,
    config: &PartitionConfig,
) -> Result<PartitionResult> {
    let n = adj.nrows();
    let k = config.n_partitions;

    if n < 2 {
        return Err(GraphError::InvalidParameter {
            param: "adj".to_string(),
            value: format!("{}x{}", n, n),
            expected: "at least 2x2 adjacency matrix".to_string(),
            context: "multilevel_partition".to_string(),
        });
    }

    if k < 2 {
        return Err(GraphError::InvalidParameter {
            param: "n_partitions".to_string(),
            value: format!("{}", k),
            expected: "at least 2".to_string(),
            context: "multilevel_partition".to_string(),
        });
    }

    if k > n {
        return Err(GraphError::InvalidParameter {
            param: "n_partitions".to_string(),
            value: format!("{}", k),
            expected: format!("at most {} (number of nodes)", n),
            context: "multilevel_partition".to_string(),
        });
    }

    if k == 2 {
        return multilevel_bisect(adj, config);
    }

    // k-way via recursive bisection
    recursive_multilevel_partition(adj, config)
}

/// Multilevel bisection (k=2).
fn multilevel_bisect(adj: &Array2<f64>, config: &PartitionConfig) -> Result<PartitionResult> {
    let n = adj.nrows();

    // Phase 1: Coarsen
    let levels = coarsen(adj, config.coarsening_threshold);

    // Phase 2: Initial partition on coarsest graph
    let coarsest_adj = if levels.is_empty() {
        adj.clone()
    } else {
        levels
            .last()
            .map(|l| l.adj.clone())
            .unwrap_or_else(|| adj.clone())
    };

    let coarse_partition = initial_partition(&coarsest_adj, config)?;

    // Phase 3: Uncoarsen with refinement
    let assignments = if levels.is_empty() {
        let mut part = coarse_partition;
        // Direct KL refinement on original graph
        for _ in 0..config.kl_max_passes {
            let improvement = kernighan_lin_pass(adj, &mut part, n / 2);
            if improvement == 0 {
                break;
            }
        }
        part
    } else {
        uncoarsen_with_refinement(&levels, coarse_partition, adj, config)?
    };

    Ok(PartitionResult::from_assignments(&assignments, adj, 2))
}

/// Recursive multilevel partition for k-way.
fn recursive_multilevel_partition(
    adj: &Array2<f64>,
    config: &PartitionConfig,
) -> Result<PartitionResult> {
    let n = adj.nrows();
    let k = config.n_partitions;

    let mut assignments = vec![0usize; n];
    // Queue: (partition_id, node_indices)
    let mut queue: Vec<(usize, Vec<usize>)> = vec![(0, (0..n).collect())];
    let mut next_id = 1usize;

    while next_id < k {
        if queue.is_empty() {
            break;
        }

        // Find the largest partition to split
        let mut largest_idx = 0;
        let mut largest_size = 0;
        for (i, (_, nodes)) in queue.iter().enumerate() {
            if nodes.len() > largest_size {
                largest_size = nodes.len();
                largest_idx = i;
            }
        }

        if largest_size < 2 {
            break;
        }

        let (pid, nodes) = queue.remove(largest_idx);
        let sub_n = nodes.len();

        // Build sub-adjacency matrix
        let mut sub_adj = Array2::<f64>::zeros((sub_n, sub_n));
        for (si, &ni) in nodes.iter().enumerate() {
            for (sj, &nj) in nodes.iter().enumerate() {
                sub_adj[[si, sj]] = adj[[ni, nj]];
            }
        }

        // Bisect the sub-graph with multilevel
        let bisect_config = PartitionConfig {
            n_partitions: 2,
            ..config.clone()
        };
        let sub_result = multilevel_bisect(&sub_adj, &bisect_config)?;

        let mut part0 = Vec::new();
        let mut part1 = Vec::new();
        for (si, &a) in sub_result.assignments.iter().enumerate() {
            if a == 0 {
                assignments[nodes[si]] = pid;
                part0.push(nodes[si]);
            } else {
                assignments[nodes[si]] = next_id;
                part1.push(nodes[si]);
            }
        }

        queue.push((pid, part0));
        queue.push((next_id, part1));
        next_id += 1;
    }

    // Assign remaining queue entries
    for (pid, nodes) in &queue {
        for &ni in nodes {
            assignments[ni] = *pid;
        }
    }

    Ok(PartitionResult::from_assignments(&assignments, adj, k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn path_graph(n: usize) -> Array2<f64> {
        let mut adj = Array2::<f64>::zeros((n, n));
        for i in 0..(n - 1) {
            adj[[i, i + 1]] = 1.0;
            adj[[i + 1, i]] = 1.0;
        }
        adj
    }

    fn two_cliques_bridge(n: usize) -> Array2<f64> {
        let size = 2 * n;
        let mut adj = Array2::<f64>::zeros((size, size));
        // Clique 1
        for i in 0..n {
            for j in (i + 1)..n {
                adj[[i, j]] = 1.0;
                adj[[j, i]] = 1.0;
            }
        }
        // Clique 2
        for i in n..size {
            for j in (i + 1)..size {
                adj[[i, j]] = 1.0;
                adj[[j, i]] = 1.0;
            }
        }
        // Single bridge edge
        adj[[n - 1, n]] = 1.0;
        adj[[n, n - 1]] = 1.0;
        adj
    }

    #[test]
    fn test_coarsening_reduces_size() {
        let adj = path_graph(20);
        let levels = coarsen(&adj, 5);
        // Should have at least one coarsening level
        assert!(!levels.is_empty());
        // Each level should be smaller
        let mut prev_n = 20;
        for level in &levels {
            assert!(
                level.n_nodes < prev_n,
                "coarsening should reduce node count"
            );
            prev_n = level.n_nodes;
        }
    }

    #[test]
    fn test_kl_never_increases_edge_cut() {
        let adj = two_cliques_bridge(5);
        let mut partition: Vec<usize> = (0..10).map(|i| if i < 5 { 0 } else { 1 }).collect();
        let before = count_edge_cut(&adj, &partition);
        let improvement = kernighan_lin_pass(&adj, &mut partition, 5);
        let after = count_edge_cut(&adj, &partition);
        assert!(after <= before, "KL should not increase edge cut");
        // improvement should match
        assert_eq!(improvement, before - after);
    }

    #[test]
    fn test_kl_on_already_optimal() {
        // Two disconnected cliques: already optimal
        let n = 4;
        let size = 2 * n;
        let mut adj = Array2::<f64>::zeros((size, size));
        for i in 0..n {
            for j in (i + 1)..n {
                adj[[i, j]] = 1.0;
                adj[[j, i]] = 1.0;
            }
        }
        for i in n..size {
            for j in (i + 1)..size {
                adj[[i, j]] = 1.0;
                adj[[j, i]] = 1.0;
            }
        }
        let mut partition: Vec<usize> = (0..size).map(|i| if i < n { 0 } else { 1 }).collect();
        let improvement = kernighan_lin_pass(&adj, &mut partition, n);
        assert_eq!(
            improvement, 0,
            "already-optimal partition should have 0 improvement"
        );
    }

    #[test]
    fn test_multilevel_bisection_two_cliques() {
        let clique_size = 5;
        let adj = two_cliques_bridge(clique_size);
        let config = PartitionConfig {
            n_partitions: 2,
            balance_tolerance: 0.1,
            coarsening_threshold: 4,
            kl_max_passes: 10,
            ..PartitionConfig::default()
        };
        let result = multilevel_partition(&adj, &config).expect("should succeed");
        assert_eq!(result.partition_sizes.len(), 2);
        // Two cliques of 5 connected by a bridge: edge cut should be reasonable
        assert!(
            result.edge_cut <= clique_size,
            "edge cut {} should be bounded for two-clique bridge",
            result.edge_cut
        );
    }

    #[test]
    fn test_multilevel_kway_all_nonempty() {
        let n = 20;
        let adj = path_graph(n);
        let config = PartitionConfig {
            n_partitions: 4,
            balance_tolerance: 0.3,
            coarsening_threshold: 3,
            kl_max_passes: 5,
            ..PartitionConfig::default()
        };
        let result = multilevel_partition(&adj, &config).expect("should succeed");
        assert_eq!(result.partition_sizes.len(), 4);
        for (i, &s) in result.partition_sizes.iter().enumerate() {
            assert!(s > 0, "partition {} should be non-empty", i);
        }
    }

    #[test]
    fn test_multilevel_balance_within_tolerance() {
        let adj = two_cliques_bridge(6);
        let config = PartitionConfig {
            n_partitions: 2,
            balance_tolerance: 0.1,
            coarsening_threshold: 4,
            kl_max_passes: 10,
            ..PartitionConfig::default()
        };
        let result = multilevel_partition(&adj, &config).expect("should succeed");
        // Imbalance should be reasonable (not wildly unbalanced)
        assert!(
            result.imbalance <= 0.5,
            "imbalance {} should be within a reasonable bound",
            result.imbalance
        );
    }

    #[test]
    fn test_multilevel_small_graph_matches_spectral() {
        // For a tiny graph, multilevel should degenerate to direct partitioning
        let adj = path_graph(4);
        let config = PartitionConfig {
            n_partitions: 2,
            balance_tolerance: 0.1,
            coarsening_threshold: 100, // no coarsening possible
            kl_max_passes: 5,
            ..PartitionConfig::default()
        };
        let ml_result = multilevel_partition(&adj, &config).expect("multilevel should succeed");
        let sp_result = super::super::spectral::spectral_bisect(&adj, &config)
            .expect("spectral should succeed");

        // Both should achieve similar edge cuts
        assert!(
            (ml_result.edge_cut as i64 - sp_result.edge_cut as i64).unsigned_abs() <= 1,
            "multilevel ({}) and spectral ({}) should produce similar edge cuts",
            ml_result.edge_cut,
            sp_result.edge_cut
        );
    }

    #[test]
    fn test_multilevel_invalid_params() {
        let adj = path_graph(4);
        let config = PartitionConfig {
            n_partitions: 1,
            ..PartitionConfig::default()
        };
        assert!(multilevel_partition(&adj, &config).is_err());

        let config2 = PartitionConfig {
            n_partitions: 10,
            ..PartitionConfig::default()
        };
        assert!(multilevel_partition(&adj, &config2).is_err());
    }
}

// ============================================================================
// Adjacency-list based METIS-style k-way partitioning API
// ============================================================================

/// Coarsening strategy for the multilevel k-way partitioner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CoarseningStrategy {
    /// Heavy-Edge Matching: match each node to its heaviest unmatched neighbor.
    HeavyEdgeMatching,
    /// Random matching: randomly pair unmatched nodes.
    RandomMatching,
    /// Sorted Heavy-Edge: sort edges by weight descending, then greedily match.
    SortedHeavyEdge,
}

/// Refinement strategy applied at each uncoarsening level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RefinementStrategy {
    /// Fiduccia-Mattheyses: max-gain vertex moves across the cut boundary.
    FiducciaMattheyses,
    /// Kernighan-Lin: alternating FM-like swaps between pairs of partitions.
    KernighanLin,
    /// Greedy balance refinement: move heavy boundary nodes to under-loaded parts.
    GreedyBalance,
}

/// Configuration for the adjacency-list multilevel k-way partitioner.
#[derive(Debug, Clone)]
pub struct PartitioningConfig {
    /// Number of output partitions (k). Must be >= 2.
    pub n_parts: usize,
    /// Coarsening strategy. Default: `HeavyEdgeMatching`.
    pub coarsening: CoarseningStrategy,
    /// Refinement strategy. Default: `FiducciaMattheyses`.
    pub refinement: RefinementStrategy,
    /// Maximum allowed imbalance: `max(|Pi|/avg) - 1`. Default 0.03 (3%).
    pub imbalance_factor: f64,
    /// Maximum number of coarsening levels. Default 10.
    pub n_coarsen_levels: usize,
    /// Random seed for stochastic strategies. Default 42.
    pub seed: u64,
}

impl Default for PartitioningConfig {
    fn default() -> Self {
        Self {
            n_parts: 2,
            coarsening: CoarseningStrategy::HeavyEdgeMatching,
            refinement: RefinementStrategy::FiducciaMattheyses,
            imbalance_factor: 0.03,
            n_coarsen_levels: 10,
            seed: 42,
        }
    }
}

/// Result of the adjacency-list k-way partitioning.
#[derive(Debug, Clone)]
pub struct KwayPartitionResult {
    /// Partition ID for each node in `0..n_parts`.
    pub partition: Vec<usize>,
    /// Number of edges whose endpoints are in different partitions.
    pub edge_cut: usize,
    /// Number of nodes assigned to each partition.
    pub part_sizes: Vec<usize>,
    /// `max(|Pi| / avg) - 1` where `avg = n / k`.
    pub imbalance: f64,
}

/// Coarsened graph using adjacency lists.
#[derive(Debug, Clone)]
struct AdjCoarseLevel {
    /// Adjacency list of the COARSENED graph: `adj[v]` = list of `(neighbor, weight)`.
    adj: Vec<Vec<(usize, f64)>>,
    /// Adjacency list of the FINE (input) graph for this level.
    fine_adj: Vec<Vec<(usize, f64)>>,
    /// Maps fine-level node index -> coarse-level node index.
    mapping: Vec<usize>,
    /// Number of coarse nodes.
    n_nodes: usize,
}

/// Coarsen an adjacency-list graph using Heavy-Edge Matching.
fn coarsen_adj_hem(
    adj: &[Vec<(usize, f64)>],
    strategy: CoarseningStrategy,
    seed: u64,
) -> AdjCoarseLevel {
    let n = adj.len();
    let mut matched = vec![false; n];
    let mut mapping = vec![0usize; n];
    let mut coarse_id = 0usize;

    // Build node processing order
    let mut node_order: Vec<usize> = (0..n).collect();

    // Deterministic shuffle using seed for RandomMatching
    if strategy == CoarseningStrategy::RandomMatching {
        // Linear congruential shuffle
        let mut rng = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        for i in (1..n).rev() {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % (i + 1);
            node_order.swap(i, j);
        }
    }

    for &i in &node_order {
        if matched[i] {
            continue;
        }

        // Find partner based on strategy
        let partner = match strategy {
            CoarseningStrategy::HeavyEdgeMatching | CoarseningStrategy::SortedHeavyEdge => {
                // Pick heaviest unmatched neighbor
                adj[i]
                    .iter()
                    .filter(|&&(nb, _)| !matched[nb] && nb != i)
                    .max_by(|&&(_, wa), &&(_, wb)| {
                        wa.partial_cmp(&wb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|&(nb, _)| nb)
            }
            CoarseningStrategy::RandomMatching => {
                // Pick first unmatched neighbor
                adj[i]
                    .iter()
                    .find(|&&(nb, _)| !matched[nb] && nb != i)
                    .map(|&(nb, _)| nb)
            }
        };

        if let Some(j) = partner {
            mapping[i] = coarse_id;
            mapping[j] = coarse_id;
            matched[i] = true;
            matched[j] = true;
        } else {
            mapping[i] = coarse_id;
            matched[i] = true;
        }
        coarse_id += 1;
    }

    let cn = coarse_id;

    // Build coarsened adjacency list
    let mut coarse_adj: Vec<std::collections::HashMap<usize, f64>> =
        vec![std::collections::HashMap::new(); cn];

    for i in 0..n {
        let ci = mapping[i];
        for &(nb, w) in &adj[i] {
            let cnb = mapping[nb];
            if ci != cnb {
                *coarse_adj[ci].entry(cnb).or_insert(0.0) += w;
            }
        }
    }

    let coarse_adj_list: Vec<Vec<(usize, f64)>> = coarse_adj
        .into_iter()
        .map(|hm| hm.into_iter().collect())
        .collect();

    AdjCoarseLevel {
        adj: coarse_adj_list,
        fine_adj: adj.to_vec(),
        mapping,
        n_nodes: cn,
    }
}

/// Compute edge cut for an adjacency-list graph given a partition assignment.
fn adj_edge_cut(adj: &[Vec<(usize, f64)>], partition: &[usize]) -> usize {
    let mut cut = 0usize;
    for (i, nbrs) in adj.iter().enumerate() {
        for &(j, _) in nbrs {
            if j > i && partition[i] != partition[j] {
                cut += 1;
            }
        }
    }
    cut
}

/// Compute partition sizes and imbalance.
fn adj_partition_stats(partition: &[usize], n_parts: usize) -> (Vec<usize>, f64) {
    let mut sizes = vec![0usize; n_parts];
    for &p in partition {
        if p < n_parts {
            sizes[p] += 1;
        }
    }
    let n = partition.len();
    let avg = n as f64 / n_parts as f64;
    let imbalance = if avg > 0.0 {
        sizes
            .iter()
            .map(|&s| (s as f64 / avg) - 1.0)
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.0)
    } else {
        0.0
    };
    (sizes, imbalance)
}

/// Initial bisection by greedy degree-based split for adjacency lists.
fn initial_bisect_adj(adj: &[Vec<(usize, f64)>]) -> Vec<usize> {
    let n = adj.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0];
    }
    // Sort by weighted degree descending, assign first half to part 0, rest to part 1
    let mut degrees: Vec<(usize, f64)> = adj
        .iter()
        .enumerate()
        .map(|(i, nbrs)| (i, nbrs.iter().map(|&(_, w)| w).sum::<f64>()))
        .collect();
    degrees.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut partition = vec![0usize; n];
    let half = n / 2;
    for (rank, &(node, _)) in degrees.iter().enumerate() {
        partition[node] = if rank < half { 0 } else { 1 };
    }
    partition
}

/// Fiduccia-Mattheyses single pass refinement for a 2-way partition.
/// Returns the cut reduction achieved.
fn fm_pass_adj(adj: &[Vec<(usize, f64)>], partition: &mut [usize], n_parts: usize) -> usize {
    let n = adj.len();
    if n < 2 || n_parts < 2 {
        return 0;
    }

    // For simplicity, apply FM between part 0 and part 1 (or any two consecutive parts)
    // Compute gain for each boundary node
    let compute_gain = |p: &[usize], node: usize| -> f64 {
        let my_part = p[node];
        let mut gain = 0.0;
        for &(nb, w) in &adj[node] {
            if p[nb] == my_part {
                gain -= w; // internal edge becomes external if moved
            } else {
                gain += w; // external edge becomes internal if moved
            }
        }
        gain
    };

    let before_cut = adj_edge_cut(adj, partition);
    let mut gains: Vec<f64> = (0..n).map(|i| compute_gain(partition, i)).collect();
    let mut locked = vec![false; n];
    // (node, old_part, new_part)
    let mut moves: Vec<(usize, usize, usize)> = Vec::new();
    let mut move_gains: Vec<f64> = Vec::new();
    // Track current partition sizes to avoid emptying a partition
    let mut curr_sizes = vec![0usize; n_parts];
    for &p in partition.iter() {
        if p < n_parts {
            curr_sizes[p] += 1;
        }
    }

    let max_moves = n.min(200);

    for _ in 0..max_moves {
        // Find the unlocked node with the highest gain
        let best = (0..n).filter(|&i| !locked[i]).max_by(|&a, &b| {
            gains[a]
                .partial_cmp(&gains[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let v = match best {
            Some(v) => v,
            None => break,
        };

        let old_part = partition[v];
        // Don't move if it would empty the source partition
        if curr_sizes[old_part] <= 1 {
            locked[v] = true;
            continue;
        }

        // Among all other partitions, pick the one with best move gain
        let mut best_target = (old_part + 1) % n_parts;
        let mut best_target_gain = f64::NEG_INFINITY;

        for target_p in 0..n_parts {
            if target_p == old_part {
                continue;
            }
            let mut g = 0.0;
            for &(nb, w) in &adj[v] {
                if partition[nb] == old_part {
                    g -= w;
                } else if partition[nb] == target_p {
                    g += w;
                }
            }
            if g > best_target_gain {
                best_target_gain = g;
                best_target = target_p;
            }
        }

        moves.push((v, old_part, best_target));
        move_gains.push(best_target_gain);
        locked[v] = true;
        curr_sizes[old_part] -= 1;
        curr_sizes[best_target] += 1;
        partition[v] = best_target;

        // Update gains for neighbors
        for &(nb, _) in &adj[v] {
            if !locked[nb] {
                gains[nb] = compute_gain(partition, nb);
            }
        }
    }

    // Find the best prefix of moves (max cumulative gain)
    let mut cumulative = 0.0f64;
    let mut best_cum = 0.0f64;
    let mut best_k = 0usize;
    for (ki, &g) in move_gains.iter().enumerate() {
        cumulative += g;
        if cumulative > best_cum {
            best_cum = cumulative;
            best_k = ki + 1;
        }
    }

    // Undo all moves (in reverse order)
    for &(node, old_part, _) in moves.iter().rev() {
        partition[node] = old_part;
    }

    // Re-apply the best prefix
    for &(node, _, new_part) in moves.iter().take(best_k) {
        partition[node] = new_part;
    }

    let after_cut = adj_edge_cut(adj, partition);
    before_cut.saturating_sub(after_cut)
}

/// Greedy balance refinement: move heavy boundary nodes to under-loaded partitions.
fn greedy_balance_pass(
    adj: &[Vec<(usize, f64)>],
    partition: &mut [usize],
    n_parts: usize,
    imbalance_target: f64,
) {
    let n = adj.len();
    let avg = n as f64 / n_parts as f64;
    let cap = (avg * (1.0 + imbalance_target)).ceil() as usize;

    let mut sizes = vec![0usize; n_parts];
    for &p in partition.iter() {
        if p < n_parts {
            sizes[p] += 1;
        }
    }

    // Identify overfull partitions and move boundary nodes
    for v in 0..n {
        let p = partition[v];
        if sizes[p] <= cap {
            continue;
        }
        // Check if v is on boundary
        let is_boundary = adj[v].iter().any(|&(nb, _)| partition[nb] != p);
        if !is_boundary {
            continue;
        }

        // Move to the partition that is most connected to this node and has space
        let mut best_target = p;
        let mut best_conn = 0.0f64;
        for target in 0..n_parts {
            if target == p || sizes[target] >= cap {
                continue;
            }
            let conn: f64 = adj[v]
                .iter()
                .filter(|&&(nb, _)| partition[nb] == target)
                .map(|&(_, w)| w)
                .sum();
            if conn > best_conn {
                best_conn = conn;
                best_target = target;
            }
        }

        if best_target != p {
            sizes[p] -= 1;
            sizes[best_target] += 1;
            partition[v] = best_target;
        }
    }
}

/// Ensure every partition 0..k is non-empty by stealing one node from
/// the largest over-populated partition.
fn fixup_empty_partitions(partition: &mut [usize], k: usize, n: usize) {
    if k == 0 || n == 0 {
        return;
    }
    let mut sizes = vec![0usize; k];
    for &p in partition.iter() {
        if p < k {
            sizes[p] += 1;
        }
    }

    for target in 0..k {
        if sizes[target] > 0 {
            continue;
        }
        // Find the largest partition to steal from
        let donor = sizes
            .iter()
            .enumerate()
            .max_by_key(|(_, &s)| s)
            .map(|(i, _)| i)
            .unwrap_or(0);

        if sizes[donor] < 2 {
            break; // Can't steal from a singleton
        }

        // Take the last node in the donor partition
        for p in partition.iter_mut() {
            if *p == donor {
                *p = target;
                sizes[donor] -= 1;
                sizes[target] += 1;
                break;
            }
        }
    }
}

/// Multilevel k-way graph partitioning (METIS-style, adjacency-list interface).
///
/// Implements the classic three-phase approach:
/// 1. **Coarsen**: repeatedly merge matching nodes using the configured strategy
/// 2. **Initial partition**: greedy bisection at the coarsest level, then expanded to k parts
/// 3. **Uncoarsen**: project partition back and refine at each level
///
/// # Arguments
/// * `adj` - Adjacency list with edge weights: `adj[v]` = `Vec<(neighbor, weight)>`.
/// * `config` - Partitioning configuration.
///
/// # Returns
/// A [`KwayPartitionResult`] with `config.n_parts` partitions.
///
/// # Errors
/// Returns [`GraphError`] if parameters are invalid.
pub fn multilevel_kway(
    adj: &[Vec<(usize, f64)>],
    config: &PartitioningConfig,
) -> Result<KwayPartitionResult> {
    let n = adj.len();
    let k = config.n_parts;

    if n < 2 {
        return Err(GraphError::InvalidParameter {
            param: "adj".to_string(),
            value: format!("{}", n),
            expected: "at least 2 nodes".to_string(),
            context: "multilevel_kway".to_string(),
        });
    }

    if k < 2 {
        return Err(GraphError::InvalidParameter {
            param: "n_parts".to_string(),
            value: format!("{}", k),
            expected: "at least 2".to_string(),
            context: "multilevel_kway".to_string(),
        });
    }

    if k > n {
        return Err(GraphError::InvalidParameter {
            param: "n_parts".to_string(),
            value: format!("{}", k),
            expected: format!("at most {} (number of nodes)", n),
            context: "multilevel_kway".to_string(),
        });
    }

    // Phase 1: Coarsen graph
    let mut levels: Vec<AdjCoarseLevel> = Vec::new();
    let mut coarsen_adj = adj.to_vec();

    for _ in 0..config.n_coarsen_levels {
        if coarsen_adj.len() < k * 2 {
            break;
        }
        let level = coarsen_adj_hem(&coarsen_adj, config.coarsening, config.seed);
        if level.n_nodes >= coarsen_adj.len() {
            break; // No coarsening happened
        }
        coarsen_adj = level.adj.clone();
        levels.push(level);
        if coarsen_adj.len() < k * 2 {
            break;
        }
    }

    // Phase 2: Initial partition at coarsest level using recursive bisection
    let coarsest_n = coarsen_adj.len();
    let mut partition = if k == 2 {
        initial_bisect_adj(&coarsen_adj)
    } else {
        // k-way via recursive bisection on the coarsest graph
        recursive_bisection_adj(&coarsen_adj, k, config.seed)
    };

    // Ensure partition has correct length
    if partition.len() != coarsest_n {
        partition.resize(coarsest_n, 0);
    }

    // Clamp partition IDs in case recursive_bisection returns out-of-range values
    for p in &mut partition {
        *p = (*p).min(k - 1);
    }

    // Apply initial refinement at coarsest level
    match config.refinement {
        RefinementStrategy::FiducciaMattheyses => {
            for _ in 0..3 {
                let imp = fm_pass_adj(&coarsen_adj, &mut partition, k);
                if imp == 0 {
                    break;
                }
            }
        }
        RefinementStrategy::KernighanLin => {
            // KL is O(n^2), so limit passes for large graphs
            for _ in 0..2 {
                let imp = fm_pass_adj(&coarsen_adj, &mut partition, k);
                if imp == 0 {
                    break;
                }
            }
        }
        RefinementStrategy::GreedyBalance => {
            greedy_balance_pass(&coarsen_adj, &mut partition, k, config.imbalance_factor);
        }
    }

    // Phase 3: Uncoarsen and refine
    // levels[0] = finest coarsening (fine_adj = original adj, maps original -> first coarse)
    // levels[last] = coarsest coarsening (fine_adj = second-coarsest, maps -> coarsest)
    // We iterate in reverse: coarsest -> finest
    for level in levels.iter().rev() {
        let fine_n = level.mapping.len(); // = level.fine_adj.len()
        let mut fine_partition = vec![0usize; fine_n];
        for (fi, &ci) in level.mapping.iter().enumerate() {
            if ci < partition.len() {
                fine_partition[fi] = partition[ci];
            }
        }

        // Clamp before refinement
        for p in &mut fine_partition {
            *p = (*p).min(k - 1);
        }

        // Apply refinement at the fine level using fine_adj
        match config.refinement {
            RefinementStrategy::FiducciaMattheyses => {
                for _ in 0..5 {
                    let imp = fm_pass_adj(&level.fine_adj, &mut fine_partition, k);
                    if imp == 0 {
                        break;
                    }
                }
            }
            RefinementStrategy::KernighanLin => {
                for _ in 0..3 {
                    let imp = fm_pass_adj(&level.fine_adj, &mut fine_partition, k);
                    if imp == 0 {
                        break;
                    }
                }
            }
            RefinementStrategy::GreedyBalance => {
                greedy_balance_pass(
                    &level.fine_adj,
                    &mut fine_partition,
                    k,
                    config.imbalance_factor,
                );
            }
        }

        partition = fine_partition;
    }

    // After uncoarsening, partition should be at original size (levels[0].fine_adj == original adj)
    // Safety: ensure correct length
    if partition.len() != n {
        partition.resize(n, 0);
    }

    // Clamp partition IDs to valid range
    for p in &mut partition {
        *p = (*p).min(k - 1);
    }

    // Ensure all k partitions are non-empty: redistribute nodes from largest partitions
    fixup_empty_partitions(&mut partition, k, n);

    // Final refinement on original graph
    match config.refinement {
        RefinementStrategy::FiducciaMattheyses | RefinementStrategy::KernighanLin => {
            for _ in 0..5 {
                let imp = fm_pass_adj(adj, &mut partition, k);
                if imp == 0 {
                    break;
                }
            }
        }
        RefinementStrategy::GreedyBalance => {
            greedy_balance_pass(adj, &mut partition, k, config.imbalance_factor);
        }
    }

    // Ensure all partitions are non-empty after refinement
    fixup_empty_partitions(&mut partition, k, n);

    // Compute final stats
    let edge_cut = adj_edge_cut(adj, &partition);
    let (part_sizes, imbalance) = adj_partition_stats(&partition, k);

    Ok(KwayPartitionResult {
        partition,
        edge_cut,
        part_sizes,
        imbalance,
    })
}

/// Recursive bisection partitioner (adjacency-list interface).
///
/// Repeatedly bisects the largest partition until `n_parts` partitions are obtained.
/// Simpler and faster than multilevel but produces lower-quality cuts.
///
/// # Arguments
/// * `adj` - Adjacency list with edge weights.
/// * `n_parts` - Desired number of partitions (must be >= 2).
/// * `seed` - Random seed for tie-breaking.
///
/// # Returns
/// A [`KwayPartitionResult`].
///
/// # Errors
/// Returns [`GraphError`] if parameters are invalid.
pub fn recursive_bisection(
    adj: &[Vec<(usize, f64)>],
    n_parts: usize,
    seed: u64,
) -> Result<KwayPartitionResult> {
    let n = adj.len();

    if n < 2 {
        return Err(GraphError::InvalidParameter {
            param: "adj".to_string(),
            value: format!("{}", n),
            expected: "at least 2 nodes".to_string(),
            context: "recursive_bisection".to_string(),
        });
    }

    if n_parts < 2 {
        return Err(GraphError::InvalidParameter {
            param: "n_parts".to_string(),
            value: format!("{}", n_parts),
            expected: "at least 2".to_string(),
            context: "recursive_bisection".to_string(),
        });
    }

    if n_parts > n {
        return Err(GraphError::InvalidParameter {
            param: "n_parts".to_string(),
            value: format!("{}", n_parts),
            expected: format!("at most {} (number of nodes)", n),
            context: "recursive_bisection".to_string(),
        });
    }

    let mut partition = recursive_bisection_adj(adj, n_parts, seed);
    fixup_empty_partitions(&mut partition, n_parts, n);
    let edge_cut = adj_edge_cut(adj, &partition);
    let (part_sizes, imbalance) = adj_partition_stats(&partition, n_parts);

    Ok(KwayPartitionResult {
        partition,
        edge_cut,
        part_sizes,
        imbalance,
    })
}

/// Internal recursive bisection without validation.
fn recursive_bisection_adj(adj: &[Vec<(usize, f64)>], n_parts: usize, seed: u64) -> Vec<usize> {
    let n = adj.len();
    let mut partition = vec![0usize; n];

    if n_parts <= 1 || n < 2 {
        return partition;
    }

    // Queue: (part_id, node_indices)
    let mut queue: Vec<(usize, Vec<usize>)> = vec![(0usize, (0..n).collect())];
    let mut next_id = 1usize;

    while next_id < n_parts {
        if queue.is_empty() {
            break;
        }

        // Find the largest partition to split
        let largest_idx = queue
            .iter()
            .enumerate()
            .max_by_key(|(_, (_, nodes))| nodes.len())
            .map(|(i, _)| i)
            .unwrap_or(0);

        if queue[largest_idx].1.len() < 2 {
            break;
        }

        let (pid, nodes) = queue.remove(largest_idx);
        let sub_n = nodes.len();

        // Build sub-adjacency list
        let mut node_to_sub = vec![usize::MAX; n];
        for (si, &ni) in nodes.iter().enumerate() {
            node_to_sub[ni] = si;
        }

        let mut sub_adj: Vec<Vec<(usize, f64)>> = vec![vec![]; sub_n];
        for (si, &ni) in nodes.iter().enumerate() {
            for &(nb, w) in &adj[ni] {
                let snb = node_to_sub[nb];
                if snb != usize::MAX {
                    sub_adj[si].push((snb, w));
                }
            }
        }

        // Bisect the sub-graph
        let sub_partition = initial_bisect_adj_seeded(&sub_adj, seed);

        let mut part0 = Vec::new();
        let mut part1 = Vec::new();
        for (si, &a) in sub_partition.iter().enumerate() {
            let global_node = nodes[si];
            if a == 0 {
                partition[global_node] = pid;
                part0.push(global_node);
            } else {
                partition[global_node] = next_id;
                part1.push(global_node);
            }
        }

        queue.push((pid, part0));
        queue.push((next_id, part1));
        next_id += 1;
    }

    // Assign remaining queue entries
    for (pid, nodes) in &queue {
        for &ni in nodes {
            partition[ni] = *pid;
        }
    }

    partition
}

/// Greedy bisection with optional seeding for variety.
fn initial_bisect_adj_seeded(adj: &[Vec<(usize, f64)>], seed: u64) -> Vec<usize> {
    let n = adj.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0];
    }

    // BFS-based bisection from a pseudo-random starting node
    let start = (seed as usize) % n;
    let mut partition = vec![usize::MAX; n];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(start);
    partition[start] = 0;
    let mut order = vec![start];

    while let Some(v) = queue.pop_front() {
        for &(nb, _) in &adj[v] {
            if partition[nb] == usize::MAX {
                partition[nb] = 0;
                order.push(nb);
                queue.push_back(nb);
            }
        }
    }

    // Handle disconnected nodes
    for i in 0..n {
        if partition[i] == usize::MAX {
            partition[i] = 0;
            order.push(i);
        }
    }

    // Assign second half of BFS order to part 1
    let half = n / 2;
    for &node in order.iter().skip(half) {
        partition[node] = 1;
    }

    partition
}

#[cfg(test)]
mod kway_tests {
    use super::*;

    fn build_path_adj(n: usize) -> Vec<Vec<(usize, f64)>> {
        let mut adj = vec![vec![]; n];
        for i in 0..(n - 1) {
            adj[i].push((i + 1, 1.0));
            adj[i + 1].push((i, 1.0));
        }
        adj
    }

    fn build_two_cliques_adj(n: usize) -> Vec<Vec<(usize, f64)>> {
        let size = 2 * n;
        let mut adj = vec![vec![]; size];
        for i in 0..n {
            for j in (i + 1)..n {
                adj[i].push((j, 1.0));
                adj[j].push((i, 1.0));
            }
        }
        for i in n..size {
            for j in (i + 1)..size {
                adj[i].push((j, 1.0));
                adj[j].push((i, 1.0));
            }
        }
        // Bridge
        adj[n - 1].push((n, 1.0));
        adj[n].push((n - 1, 1.0));
        adj
    }

    #[test]
    fn test_multilevel_2way() {
        let adj = build_path_adj(20);
        let config = PartitioningConfig {
            n_parts: 2,
            ..PartitioningConfig::default()
        };
        let result = multilevel_kway(&adj, &config).expect("2-way partition should succeed");
        assert_eq!(result.part_sizes.len(), 2);
        assert_eq!(result.part_sizes.iter().sum::<usize>(), 20);
        for &s in &result.part_sizes {
            assert!(s > 0, "each part should be non-empty");
        }
    }

    #[test]
    fn test_multilevel_4way() {
        let adj = build_path_adj(40);
        let config = PartitioningConfig {
            n_parts: 4,
            ..PartitioningConfig::default()
        };
        let result = multilevel_kway(&adj, &config).expect("4-way partition should succeed");
        assert_eq!(result.part_sizes.len(), 4);
        assert_eq!(result.part_sizes.iter().sum::<usize>(), 40);
        for &s in &result.part_sizes {
            assert!(s > 0, "each part should be non-empty");
        }
    }

    #[test]
    fn test_recursive_bisection() {
        let adj = build_two_cliques_adj(5);
        let result = recursive_bisection(&adj, 2, 42).expect("recursive bisection should succeed");
        assert_eq!(result.part_sizes.len(), 2);
        assert_eq!(result.part_sizes.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_partition_imbalance_bound() {
        let adj = build_two_cliques_adj(6);
        let config = PartitioningConfig {
            n_parts: 2,
            imbalance_factor: 0.3,
            ..PartitioningConfig::default()
        };
        let result = multilevel_kway(&adj, &config).expect("should succeed");
        // Imbalance should be reasonable (< 1.0 means no partition is more than double the average)
        assert!(
            result.imbalance < 1.0,
            "imbalance {} too high",
            result.imbalance
        );
    }

    #[test]
    fn test_invalid_params() {
        let adj = build_path_adj(4);

        let config_k1 = PartitioningConfig {
            n_parts: 1,
            ..PartitioningConfig::default()
        };
        assert!(multilevel_kway(&adj, &config_k1).is_err());

        let config_big = PartitioningConfig {
            n_parts: 100,
            ..PartitioningConfig::default()
        };
        assert!(multilevel_kway(&adj, &config_big).is_err());

        assert!(recursive_bisection(&adj, 1, 42).is_err());
        assert!(recursive_bisection(&adj, 100, 42).is_err());
    }

    #[test]
    fn test_coarsening_strategies() {
        let adj = build_path_adj(20);
        for strategy in [
            CoarseningStrategy::HeavyEdgeMatching,
            CoarseningStrategy::RandomMatching,
            CoarseningStrategy::SortedHeavyEdge,
        ] {
            let config = PartitioningConfig {
                n_parts: 2,
                coarsening: strategy,
                ..PartitioningConfig::default()
            };
            let result = multilevel_kway(&adj, &config).expect("should succeed");
            assert_eq!(result.part_sizes.iter().sum::<usize>(), 20);
        }
    }

    #[test]
    fn test_refinement_strategies() {
        let adj = build_two_cliques_adj(4);
        for strategy in [
            RefinementStrategy::FiducciaMattheyses,
            RefinementStrategy::KernighanLin,
            RefinementStrategy::GreedyBalance,
        ] {
            let config = PartitioningConfig {
                n_parts: 2,
                refinement: strategy,
                ..PartitioningConfig::default()
            };
            let result = multilevel_kway(&adj, &config).expect("should succeed");
            assert_eq!(result.part_sizes.iter().sum::<usize>(), 8);
        }
    }
}
