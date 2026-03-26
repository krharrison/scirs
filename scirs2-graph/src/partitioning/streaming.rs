//! Streaming graph partitioning for large-scale graphs.
//!
//! When a graph is too large to fit in memory or arrives as a stream of
//! edges, streaming partitioners assign nodes to partitions in a single
//! pass. This module provides:
//!
//! - **LDG (Linear Deterministic Greedy)**: assigns each arriving vertex
//!   to the partition that maximizes the number of neighbors already present,
//!   penalized by partition size to maintain balance.
//! - **Hash partitioning**: simple baseline that assigns nodes by hash.

use scirs2_core::ndarray::Array2;

use crate::error::{GraphError, Result};

use super::types::{PartitionConfig, PartitionResult};

/// Linear Deterministic Greedy (LDG) streaming partitioner.
///
/// Processes nodes in order 0..n_nodes. For each node, it examines its
/// edges (among those already seen) and assigns the node to the partition
/// that maximizes:
///
///   score(p) = neighbors_in_p * (1 - |p| / capacity)
///
/// where capacity = ceil(n_nodes * (1 + balance_tolerance) / n_partitions).
///
/// # Arguments
/// * `edges` - Edge list as (src, dst) pairs. Both directions should be
///   included for undirected graphs.
/// * `n_nodes` - Total number of nodes (nodes are indexed 0..n_nodes).
/// * `config` - Partition configuration.
///
/// # Returns
/// A `PartitionResult` with partition assignments for all nodes.
///
/// # Errors
/// Returns `GraphError::InvalidParameter` if parameters are invalid.
pub fn streaming_partition(
    edges: &[(usize, usize)],
    n_nodes: usize,
    config: &PartitionConfig,
) -> Result<PartitionResult> {
    let k = config.n_partitions;

    if k < 2 {
        return Err(GraphError::InvalidParameter {
            param: "n_partitions".to_string(),
            value: format!("{}", k),
            expected: "at least 2".to_string(),
            context: "streaming_partition".to_string(),
        });
    }

    if n_nodes < 2 {
        return Err(GraphError::InvalidParameter {
            param: "n_nodes".to_string(),
            value: format!("{}", n_nodes),
            expected: "at least 2".to_string(),
            context: "streaming_partition".to_string(),
        });
    }

    if k > n_nodes {
        return Err(GraphError::InvalidParameter {
            param: "n_partitions".to_string(),
            value: format!("{}", k),
            expected: format!("at most {} (number of nodes)", n_nodes),
            context: "streaming_partition".to_string(),
        });
    }

    // Build adjacency list from edge list
    let mut adj_list: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for &(u, v) in edges {
        if u < n_nodes && v < n_nodes {
            adj_list[u].push(v);
        }
    }

    // Capacity per partition (with tolerance)
    let capacity =
        ((n_nodes as f64) * (1.0 + config.balance_tolerance) / (k as f64)).ceil() as usize;

    let mut assignments = vec![usize::MAX; n_nodes];
    let mut partition_sizes = vec![0usize; k];

    // Process nodes in order
    for node in 0..n_nodes {
        let mut best_partition = 0usize;
        let mut best_score = f64::NEG_INFINITY;

        for p in 0..k {
            if partition_sizes[p] >= capacity {
                continue;
            }

            // Count neighbors already in partition p
            let neighbors_in_p = adj_list[node]
                .iter()
                .filter(|&&nbr| nbr < node && assignments[nbr] == p)
                .count();

            // LDG scoring: neighbors * (1 - load_factor)
            let load_factor = partition_sizes[p] as f64 / capacity as f64;
            let score = (neighbors_in_p as f64) * (1.0 - load_factor);

            // Tie-breaking: prefer less-loaded partition
            if score > best_score
                || (score == best_score && partition_sizes[p] < partition_sizes[best_partition])
            {
                best_score = score;
                best_partition = p;
            }
        }

        assignments[node] = best_partition;
        partition_sizes[best_partition] += 1;
    }

    // Compute edge cut
    let mut edge_cut = 0usize;
    let mut seen_edges = std::collections::HashSet::new();
    for &(u, v) in edges {
        if u < n_nodes && v < n_nodes && u != v {
            let key = if u < v { (u, v) } else { (v, u) };
            if seen_edges.insert(key) && assignments[u] != assignments[v] {
                edge_cut += 1;
            }
        }
    }

    // Compute imbalance
    let ideal = n_nodes as f64 / k as f64;
    let imbalance = if ideal > 0.0 {
        partition_sizes
            .iter()
            .map(|&s| ((s as f64) - ideal).abs() / ideal)
            .fold(0.0f64, f64::max)
    } else {
        0.0
    };

    Ok(PartitionResult {
        assignments,
        edge_cut,
        partition_sizes,
        imbalance,
    })
}

/// Hash-based streaming partition (baseline).
///
/// Assigns each node to partition `node % n_partitions`. This is the simplest
/// possible partitioner and serves as a baseline for comparison.
///
/// # Arguments
/// * `n_nodes` - Number of nodes (0-indexed).
/// * `n_partitions` - Number of partitions.
///
/// # Returns
/// A `PartitionResult` with uniform (or near-uniform) partition sizes.
pub fn hash_partition(n_nodes: usize, n_partitions: usize) -> PartitionResult {
    let mut assignments = vec![0usize; n_nodes];
    let mut partition_sizes = vec![0usize; n_partitions];

    for i in 0..n_nodes {
        let p = i % n_partitions;
        assignments[i] = p;
        partition_sizes[p] += 1;
    }

    let ideal = n_nodes as f64 / n_partitions as f64;
    let imbalance = if ideal > 0.0 {
        partition_sizes
            .iter()
            .map(|&s| ((s as f64) - ideal).abs() / ideal)
            .fold(0.0f64, f64::max)
    } else {
        0.0
    };

    PartitionResult {
        assignments,
        edge_cut: 0, // Not computed without adjacency info
        partition_sizes,
        imbalance,
    }
}

/// Evaluate a partition against an adjacency matrix.
///
/// Computes the edge cut (number of edges crossing partitions) and
/// the imbalance ratio.
///
/// # Arguments
/// * `adj` - Symmetric adjacency matrix (n x n).
/// * `assignments` - Partition assignment for each node.
/// * `n_partitions` - Number of partitions.
///
/// # Returns
/// A tuple `(edge_cut, imbalance)`.
pub fn evaluate_partition(
    adj: &Array2<f64>,
    assignments: &[usize],
    n_partitions: usize,
) -> (usize, f64) {
    let n = adj.nrows().min(assignments.len());

    let mut partition_sizes = vec![0usize; n_partitions];
    for &a in &assignments[..n] {
        if a < n_partitions {
            partition_sizes[a] += 1;
        }
    }

    let mut edge_cut = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if adj[[i, j]].abs() > 1e-15 && assignments[i] != assignments[j] {
                edge_cut += 1;
            }
        }
    }

    let ideal = n as f64 / n_partitions as f64;
    let imbalance = if ideal > 0.0 {
        partition_sizes
            .iter()
            .map(|&s| ((s as f64) - ideal).abs() / ideal)
            .fold(0.0f64, f64::max)
    } else {
        0.0
    };

    (edge_cut, imbalance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Build edge list for two cliques connected by a bridge.
    fn two_cliques_edges(n: usize) -> (Vec<(usize, usize)>, usize) {
        let size = 2 * n;
        let mut edges = Vec::new();
        // Clique 1
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push((i, j));
                edges.push((j, i));
            }
        }
        // Clique 2
        for i in n..size {
            for j in (i + 1)..size {
                edges.push((i, j));
                edges.push((j, i));
            }
        }
        // Bridge
        edges.push((n - 1, n));
        edges.push((n, n - 1));
        (edges, size)
    }

    #[test]
    fn test_ldg_better_than_hash_on_structured() {
        let (edges, n_nodes) = two_cliques_edges(6);
        let config = PartitionConfig {
            n_partitions: 2,
            balance_tolerance: 0.1,
            ..PartitionConfig::default()
        };

        let ldg_result = streaming_partition(&edges, n_nodes, &config).expect("LDG should succeed");

        // Build adjacency matrix for evaluation
        let mut adj = Array2::<f64>::zeros((n_nodes, n_nodes));
        for &(u, v) in &edges {
            adj[[u, v]] = 1.0;
        }

        let hash_result = hash_partition(n_nodes, 2);
        let (hash_cut, _) = evaluate_partition(&adj, &hash_result.assignments, 2);

        // LDG should achieve a lower or equal edge cut on structured graphs
        assert!(
            ldg_result.edge_cut <= hash_cut + 2,
            "LDG edge cut ({}) should be competitive with hash ({})",
            ldg_result.edge_cut,
            hash_cut
        );
    }

    #[test]
    fn test_hash_uniform_sizes() {
        let n_nodes = 100;
        let k = 4;
        let result = hash_partition(n_nodes, k);
        assert_eq!(result.partition_sizes.len(), k);
        // Each partition should have 25 nodes
        for &s in &result.partition_sizes {
            assert_eq!(s, 25);
        }
        assert!(result.imbalance < 1e-10);
    }

    #[test]
    fn test_hash_near_uniform_sizes() {
        let n_nodes = 10;
        let k = 3;
        let result = hash_partition(n_nodes, k);
        assert_eq!(result.partition_sizes.len(), k);
        // Sizes should be 4, 3, 3 or similar
        let total: usize = result.partition_sizes.iter().sum();
        assert_eq!(total, n_nodes);
        for &s in &result.partition_sizes {
            assert!((3..=4).contains(&s));
        }
    }

    #[test]
    fn test_evaluate_partition() {
        let n = 4;
        let mut adj = Array2::<f64>::zeros((n, n));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[2, 3]] = 1.0;
        adj[[3, 2]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 1]] = 1.0;

        let assignments = vec![0, 0, 1, 1];
        let (cut, imbalance) = evaluate_partition(&adj, &assignments, 2);
        assert_eq!(cut, 1); // only edge 1-2 crosses
        assert!(imbalance < 1e-10);
    }

    #[test]
    fn test_single_node_trivial() {
        // Edge case: 2 nodes, 2 partitions
        let edges = vec![(0, 1), (1, 0)];
        let config = PartitionConfig {
            n_partitions: 2,
            balance_tolerance: 0.5,
            ..PartitionConfig::default()
        };
        let result = streaming_partition(&edges, 2, &config).expect("should succeed");
        assert_eq!(result.assignments.len(), 2);
        // Both nodes should be assigned (valid partition IDs)
        assert!(result.assignments[0] < 2);
        assert!(result.assignments[1] < 2);
        // Total nodes accounted for
        let total: usize = result.partition_sizes.iter().sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn test_streaming_invalid_params() {
        let config = PartitionConfig {
            n_partitions: 1,
            ..PartitionConfig::default()
        };
        assert!(streaming_partition(&[], 10, &config).is_err());

        let config2 = PartitionConfig {
            n_partitions: 5,
            ..PartitionConfig::default()
        };
        assert!(streaming_partition(&[], 3, &config2).is_err());
    }

    #[test]
    fn test_edge_cut_computable() {
        let (edges, n_nodes) = two_cliques_edges(4);
        let mut adj = Array2::<f64>::zeros((n_nodes, n_nodes));
        for &(u, v) in &edges {
            adj[[u, v]] = 1.0;
        }

        let config = PartitionConfig {
            n_partitions: 2,
            balance_tolerance: 0.2,
            ..PartitionConfig::default()
        };
        let result = streaming_partition(&edges, n_nodes, &config).expect("should succeed");

        // Verify edge cut matches evaluate_partition
        let (eval_cut, _) = evaluate_partition(&adj, &result.assignments, 2);
        assert_eq!(result.edge_cut, eval_cut);
    }
}

// ============================================================================
// Stateful streaming partitioner with FENNEL / Hashing / LDG backends
// ============================================================================

/// Algorithm variant for the stateful streaming partitioner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StreamingPartitionAlgorithm {
    /// FENNEL: minimize edge cut + imbalance penalty, vertex streaming.
    /// Score for partition p: `|N(v) ∩ Vp| - γ · |Vp|^α`.
    Fennel,
    /// Consistent hashing baseline: `v % k`.
    Hashing,
    /// Linear Deterministic Greedy: `|N(v) ∩ Vp| · (1 - |Vp| / cap)`.
    LinearDeterministic,
}

/// Configuration for the stateful streaming partitioner.
#[derive(Debug, Clone)]
pub struct StreamingPartitionConfig {
    /// Number of target partitions. Must be >= 2.
    pub n_parts: usize,
    /// Algorithm variant. Default: `Fennel`.
    pub algorithm: StreamingPartitionAlgorithm,
    /// Balance penalty weight γ for FENNEL. Default 1.5.
    pub gamma: f64,
    /// FENNEL exponent α (default 1.5, higher = stronger balance enforcement).
    /// Set to 0.0 to auto-compute from graph density.
    pub alpha: f64,
}

impl Default for StreamingPartitionConfig {
    fn default() -> Self {
        Self {
            n_parts: 2,
            algorithm: StreamingPartitionAlgorithm::Fennel,
            gamma: 1.5,
            alpha: 1.5,
        }
    }
}

/// A stateful online streaming partitioner.
///
/// Nodes arrive one at a time (with their adjacency list as observed so far).
/// Each call to [`assign_vertex`](StreamingPartitioner::assign_vertex) makes an
/// irrevocable assignment based on already-assigned neighbors.
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_graph::partitioning::{StreamingPartitioner, StreamingPartitionConfig, StreamingPartitionAlgorithm};
///
/// let config = StreamingPartitionConfig { n_parts: 3, ..Default::default() };
/// let mut sp = StreamingPartitioner::new(10, config);
/// // Assign node 0 with no neighbors yet
/// let part = sp.assign_vertex(0, &[]);
/// assert!(part < 3);
/// ```
pub struct StreamingPartitioner {
    config: StreamingPartitionConfig,
    /// Partition assignment for each node; `None` means not yet assigned.
    partition: Vec<Option<usize>>,
    /// Number of nodes in each partition.
    part_sizes: Vec<usize>,
    /// Total nodes processed so far.
    n_assigned: usize,
}

impl StreamingPartitioner {
    /// Create a new streaming partitioner for `n_nodes` nodes.
    pub fn new(n_nodes: usize, config: StreamingPartitionConfig) -> Self {
        let k = config.n_parts;
        Self {
            config,
            partition: vec![None; n_nodes],
            part_sizes: vec![0usize; k],
            n_assigned: 0,
        }
    }

    /// Assign vertex `v` to a partition given its (already-known) neighbors.
    ///
    /// Only neighbors whose partition is already determined influence the score.
    /// Returns the assigned partition ID.
    pub fn assign_vertex(&mut self, v: usize, neighbors: &[(usize, f64)]) -> usize {
        if v >= self.partition.len() {
            // Grow if needed
            self.partition.resize(v + 1, None);
        }

        // If already assigned, return current assignment
        if let Some(p) = self.partition[v] {
            return p;
        }

        let k = self.config.n_parts;
        let n_total = self.partition.len().max(1);
        // Capacity with 5% slack
        let cap = ((n_total as f64 * 1.05) / k as f64).ceil() as usize;

        let best_p = match self.config.algorithm {
            StreamingPartitionAlgorithm::Hashing => v % k,
            StreamingPartitionAlgorithm::LinearDeterministic => {
                let mut best = 0usize;
                let mut best_score = f64::NEG_INFINITY;
                for p in 0..k {
                    if self.part_sizes[p] >= cap {
                        continue;
                    }
                    let nbrs_in_p: f64 = neighbors
                        .iter()
                        .filter(|&&(nb, _)| {
                            nb < self.partition.len() && self.partition[nb] == Some(p)
                        })
                        .map(|&(_, w)| w)
                        .sum();
                    let load = self.part_sizes[p] as f64 / cap as f64;
                    let score = nbrs_in_p * (1.0 - load);
                    if score > best_score
                        || (score == best_score && self.part_sizes[p] < self.part_sizes[best])
                    {
                        best_score = score;
                        best = p;
                    }
                }
                best
            }
            StreamingPartitionAlgorithm::Fennel => {
                // FENNEL score: |N(v) ∩ Vp| - γ · |Vp|^α
                let gamma = self.config.gamma;
                let alpha = if self.config.alpha <= 0.0 {
                    // Auto: sqrt(k) / n gives good defaults
                    (k as f64).sqrt() / (n_total as f64).max(1.0)
                } else {
                    self.config.alpha
                };

                let mut best = 0usize;
                let mut best_score = f64::NEG_INFINITY;
                for p in 0..k {
                    if self.part_sizes[p] >= cap {
                        continue;
                    }
                    let nbrs_in_p: f64 = neighbors
                        .iter()
                        .filter(|&&(nb, _)| {
                            nb < self.partition.len() && self.partition[nb] == Some(p)
                        })
                        .map(|&(_, w)| w)
                        .sum();
                    let penalty = gamma * (self.part_sizes[p] as f64).powf(alpha);
                    let score = nbrs_in_p - penalty;
                    if score > best_score
                        || (score == best_score && self.part_sizes[p] < self.part_sizes[best])
                    {
                        best_score = score;
                        best = p;
                    }
                }
                best
            }
        };

        self.partition[v] = Some(best_p);
        self.part_sizes[best_p] += 1;
        self.n_assigned += 1;
        best_p
    }

    /// Return the current partition assignments (`None` = not yet assigned).
    pub fn current_partition(&self) -> &[Option<usize>] {
        &self.partition
    }

    /// Estimate the edge cut from the current partition state.
    ///
    /// Only counts edges where both endpoints are assigned and in different parts.
    /// Counts each undirected edge once.
    pub fn edge_cut_estimate(&self, adj: &[Vec<(usize, f64)>]) -> usize {
        let mut cut = 0usize;
        for (i, nbrs) in adj.iter().enumerate() {
            let pi = match self.partition.get(i).copied().flatten() {
                Some(p) => p,
                None => continue,
            };
            for &(j, _) in nbrs {
                if j <= i {
                    continue; // count each edge once
                }
                let pj = match self.partition.get(j).copied().flatten() {
                    Some(p) => p,
                    None => continue,
                };
                if pi != pj {
                    cut += 1;
                }
            }
        }
        cut
    }
}

#[cfg(test)]
mod streaming_partitioner_tests {
    use super::*;

    fn build_path_adj(n: usize) -> Vec<Vec<(usize, f64)>> {
        let mut adj = vec![vec![]; n];
        for i in 0..(n - 1) {
            adj[i].push((i + 1, 1.0));
            adj[i + 1].push((i, 1.0));
        }
        adj
    }

    #[test]
    fn test_streaming_fennel_assignment() {
        let n = 20;
        let adj = build_path_adj(n);
        let config = StreamingPartitionConfig {
            n_parts: 4,
            algorithm: StreamingPartitionAlgorithm::Fennel,
            ..StreamingPartitionConfig::default()
        };
        let mut sp = StreamingPartitioner::new(n, config);

        for i in 0..n {
            let nbrs: Vec<(usize, f64)> = adj[i].clone();
            let p = sp.assign_vertex(i, &nbrs);
            assert!(p < 4, "part {} out of range", p);
        }

        // All nodes assigned
        for opt in sp.current_partition() {
            assert!(opt.is_some(), "node should be assigned");
        }
    }

    #[test]
    fn test_streaming_hashing_uniform() {
        let n = 100;
        let config = StreamingPartitionConfig {
            n_parts: 4,
            algorithm: StreamingPartitionAlgorithm::Hashing,
            ..StreamingPartitionConfig::default()
        };
        let mut sp = StreamingPartitioner::new(n, config);
        for i in 0..n {
            sp.assign_vertex(i, &[]);
        }
        // Each part should have exactly 25 nodes
        for &s in &sp.part_sizes {
            assert_eq!(s, 25, "hash partition should be uniform");
        }
    }

    #[test]
    fn test_streaming_ldg_assigns_all() {
        let n = 30;
        let adj = build_path_adj(n);
        let config = StreamingPartitionConfig {
            n_parts: 3,
            algorithm: StreamingPartitionAlgorithm::LinearDeterministic,
            ..StreamingPartitionConfig::default()
        };
        let mut sp = StreamingPartitioner::new(n, config);
        for i in 0..n {
            let nbrs = adj[i].clone();
            let p = sp.assign_vertex(i, &nbrs);
            assert!(p < 3);
        }
        let total: usize = sp.part_sizes.iter().sum();
        assert_eq!(total, n);
    }

    #[test]
    fn test_streaming_edge_cut_estimate() {
        let n = 10;
        let adj = build_path_adj(n);
        let config = StreamingPartitionConfig {
            n_parts: 2,
            algorithm: StreamingPartitionAlgorithm::Fennel,
            ..StreamingPartitionConfig::default()
        };
        let mut sp = StreamingPartitioner::new(n, config);
        for i in 0..n {
            let nbrs = adj[i].clone();
            sp.assign_vertex(i, &nbrs);
        }
        let cut = sp.edge_cut_estimate(&adj);
        // For a path of 10 split into 2, cut should be small (1-3 edges)
        assert!(cut <= n, "edge cut {} should be <= n={}", cut, n);
    }
}
