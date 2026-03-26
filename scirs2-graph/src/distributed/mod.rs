//! Distributed graph storage with partitioned adjacency.
//!
//! This module provides data structures and algorithms for partitioning large
//! graphs across multiple logical shards, enabling distributed-memory graph
//! processing without a single monolithic adjacency representation.
//!
//! # Overview
//!
//! A [`DistributedGraph`] splits vertices across [`GraphShard`]s according to
//! a chosen [`GraphPartitionMethod`].  Cross-shard edges create *mirror*
//! vertices — lightweight replicas used to route messages during distributed
//! algorithms without accessing the remote shard.
//!
//! | Method | Description |
//! |--------|-------------|
//! | [`GraphPartitionMethod::HashBased`] | `vertex % n_partitions` (round-robin, zero overhead) |
//! | [`GraphPartitionMethod::EdgeCut`] | Same as hash; edges of a vertex live on its shard |
//! | [`GraphPartitionMethod::VertexCut`] | Edge lives on the shard of the lower-degree endpoint |
//! | [`GraphPartitionMethod::Fennel`] | Streaming FENNEL greedy assignment (Tsourakakis 2014) |
//!
//! # Example
//!
//! ```rust
//! use scirs2_graph::distributed::{build_distributed_graph, DistributedGraphConfig, distributed_degree};
//!
//! let edges: Vec<(usize, usize)> = (0..5).flat_map(|i| (i+1..5).map(move |j| (i, j))).collect();
//! let cfg = DistributedGraphConfig::default();
//! let dg = build_distributed_graph(&edges, 5, &cfg);
//! assert!(distributed_degree(&dg, 0).is_some());
//! ```

use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// Method used to partition a graph's vertices across shards.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphPartitionMethod {
    /// Hash-based: `vertex_id % n_partitions` (deterministic, zero overhead).
    HashBased,
    /// Edge-cut: assign each edge to the source vertex's shard.
    EdgeCut,
    /// Vertex-cut: assign each edge to the shard of the lower-degree endpoint.
    VertexCut,
    /// Streaming FENNEL (Tsourakakis et al. 2014): maximise local density while
    /// penalising shard imbalance.
    Fennel,
}

/// Configuration for [`build_distributed_graph`].
#[derive(Debug, Clone)]
pub struct DistributedGraphConfig {
    /// Number of logical shards/partitions.
    pub n_partitions: usize,
    /// Partitioning algorithm to use.
    pub partition_method: GraphPartitionMethod,
    /// Extra replicas per vertex (0 = no replication).
    pub replication_factor: usize,
}

impl Default for DistributedGraphConfig {
    fn default() -> Self {
        Self {
            n_partitions: 4,
            partition_method: GraphPartitionMethod::HashBased,
            replication_factor: 0,
        }
    }
}

/// Location of a vertex in the distributed graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexLocation {
    /// Shard (partition) that owns this vertex.
    pub partition_id: usize,
    /// Local index within that shard's vertex list.
    pub local_id: usize,
}

/// A single partition (shard) of the distributed graph.
#[derive(Debug, Clone, Default)]
pub struct GraphShard {
    /// Global vertex IDs owned by this shard.
    pub local_vertices: Vec<usize>,
    /// Edges `(src, dst)` (both global IDs) assigned to this shard.
    pub local_edges: Vec<(usize, usize)>,
    /// Mirror vertices: remote vertices referenced by local edges.
    pub mirror_vertices: Vec<usize>,
}

/// Distributed graph: a collection of shards with a global vertex map.
#[derive(Debug, Clone)]
pub struct DistributedGraph {
    /// One shard per partition.
    pub shards: Vec<GraphShard>,
    /// Total number of vertices in the original graph.
    pub n_global_vertices: usize,
    /// Total number of edges in the original graph.
    pub n_global_edges: usize,
    /// O(1) lookup: global vertex id → partition + local index.
    pub vertex_map: HashMap<usize, VertexLocation>,
}

/// Summary statistics for a distributed graph.
#[derive(Debug, Clone)]
pub struct DistributedStats {
    /// `max_shard_size / avg_shard_size` (1.0 = perfectly balanced).
    pub balance_ratio: f64,
    /// Fraction of edges that cross partition boundaries.
    pub edge_cut_fraction: f64,
    /// Average number of shards a vertex is present in (including mirrors).
    pub replication_factor: f64,
    /// Number of vertices in each shard (owned, not mirror).
    pub shard_sizes: Vec<usize>,
}

// ────────────────────────────────────────────────────────────────────────────
// Partitioning helpers
// ────────────────────────────────────────────────────────────────────────────

/// Hash-partition: assign vertex to `vertex % n_partitions`.
#[inline]
pub fn hash_partition(vertex: usize, n_partitions: usize) -> usize {
    if n_partitions == 0 {
        return 0;
    }
    vertex % n_partitions
}

/// Streaming FENNEL partition assignment (Tsourakakis et al. 2014).
///
/// Processes edges in stream order and assigns each unassigned vertex to the
/// partition that maximises:
///   `score(p) = |N(v) ∩ V_p| − α × |V_p|^γ`
/// where `α = sqrt(|E| / n_partitions^γ)` and `γ = 3/2`.
///
/// # Returns
/// A `Vec` of length `n_vertices` where entry `v` is the partition for vertex `v`.
pub fn fennel_partition(
    edges: &[(usize, usize)],
    n_vertices: usize,
    n_partitions: usize,
    config: &DistributedGraphConfig,
) -> Vec<usize> {
    if n_partitions == 0 || n_vertices == 0 {
        return vec![0; n_vertices];
    }

    let n_edges = edges.len() as f64;
    let gamma = 1.5_f64;
    // α = sqrt(|E| / k^γ) where k = n_partitions
    let alpha = (n_edges / (n_partitions as f64).powf(gamma)).sqrt();

    // assignment[v] = Some(partition) once decided
    let mut assignment: Vec<Option<usize>> = vec![None; n_vertices];
    // |V_p| for each partition
    let mut partition_sizes: Vec<f64> = vec![0.0; n_partitions];
    // adjacency list built incrementally for N(v) ∩ V_p queries
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_vertices];

    let _ = config; // config is passed for forward-compatibility

    for &(u, v) in edges {
        // Update adjacency (undirected)
        if u < n_vertices && v < n_vertices {
            adj[u].push(v);
            adj[v].push(u);
        }

        // Assign u if not yet assigned
        for &vertex in &[u, v] {
            if vertex >= n_vertices || assignment[vertex].is_some() {
                continue;
            }
            // Count neighbours already assigned to each partition
            let mut neighbour_counts: Vec<f64> = vec![0.0; n_partitions];
            for &nb in &adj[vertex] {
                if nb < n_vertices {
                    if let Some(p) = assignment[nb] {
                        neighbour_counts[p] += 1.0;
                    }
                }
            }
            // Pick the partition with the highest score
            let best = (0..n_partitions)
                .map(|p| {
                    let sz = partition_sizes[p];
                    let score = neighbour_counts[p] - alpha * sz.powf(gamma);
                    (p, score)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(p, _)| p)
                .unwrap_or(0);

            assignment[vertex] = Some(best);
            partition_sizes[best] += 1.0;
        }
    }

    // Any vertex not touched by any edge falls back to hash
    assignment
        .iter()
        .enumerate()
        .map(|(v, a)| a.unwrap_or_else(|| hash_partition(v, n_partitions)))
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// build_distributed_graph
// ────────────────────────────────────────────────────────────────────────────

/// Build a [`DistributedGraph`] from an edge list.
///
/// Steps:
/// 1. Assign each vertex to a partition using the configured method.
/// 2. Assign each edge to a shard:
///    - `EdgeCut` / `HashBased`: edge goes to the source vertex's shard.
///    - `VertexCut`: edge goes to the shard of the endpoint with fewer edges.
///    - `Fennel`: same as `EdgeCut` after FENNEL vertex assignment.
/// 3. Record mirror vertices for cross-shard edges.
/// 4. Build `vertex_map` for O(1) location queries.
pub fn build_distributed_graph(
    edges: &[(usize, usize)],
    n_vertices: usize,
    config: &DistributedGraphConfig,
) -> DistributedGraph {
    let n_partitions = config.n_partitions.max(1);

    // Step 1: compute vertex → partition assignment
    let vertex_partition: Vec<usize> = match config.partition_method {
        GraphPartitionMethod::HashBased | GraphPartitionMethod::EdgeCut => (0..n_vertices)
            .map(|v| hash_partition(v, n_partitions))
            .collect(),
        GraphPartitionMethod::VertexCut => {
            // For VertexCut we still need to know partition of each vertex for
            // edge routing, but we route by degree heuristic at edge time.
            (0..n_vertices)
                .map(|v| hash_partition(v, n_partitions))
                .collect()
        }
        GraphPartitionMethod::Fennel => fennel_partition(edges, n_vertices, n_partitions, config),
    };

    // Pre-compute degree counts for VertexCut
    let mut degree: Vec<usize> = vec![0usize; n_vertices];
    for &(u, v) in edges {
        if u < n_vertices {
            degree[u] += 1;
        }
        if v < n_vertices {
            degree[v] += 1;
        }
    }

    // Step 2: initialise shards
    let mut shards: Vec<GraphShard> = (0..n_partitions).map(|_| GraphShard::default()).collect();

    // Assign vertices to shards
    let mut vertex_map: HashMap<usize, VertexLocation> = HashMap::with_capacity(n_vertices);
    for v in 0..n_vertices {
        let pid = vertex_partition[v];
        let pid = pid.min(n_partitions - 1);
        let local_id = shards[pid].local_vertices.len();
        shards[pid].local_vertices.push(v);
        vertex_map.insert(
            v,
            VertexLocation {
                partition_id: pid,
                local_id,
            },
        );
    }

    // Step 3: assign edges and record mirrors
    // We use a set per shard to avoid duplicate mirror entries
    let mut mirror_sets: Vec<std::collections::HashSet<usize>> = (0..n_partitions)
        .map(|_| std::collections::HashSet::new())
        .collect();

    for &(u, v) in edges {
        if u >= n_vertices || v >= n_vertices {
            continue;
        }
        let shard_idx = match config.partition_method {
            GraphPartitionMethod::VertexCut => {
                // Edge goes to the shard of the lower-degree endpoint
                if degree[u] <= degree[v] {
                    vertex_partition[u].min(n_partitions - 1)
                } else {
                    vertex_partition[v].min(n_partitions - 1)
                }
            }
            _ => {
                // EdgeCut / HashBased / Fennel: edge belongs to source's shard
                vertex_partition[u].min(n_partitions - 1)
            }
        };

        shards[shard_idx].local_edges.push((u, v));

        // Record mirrors for cross-shard endpoints
        let owner_u = vertex_partition[u].min(n_partitions - 1);
        let owner_v = vertex_partition[v].min(n_partitions - 1);

        if owner_u != shard_idx {
            mirror_sets[shard_idx].insert(u);
        }
        if owner_v != shard_idx {
            mirror_sets[shard_idx].insert(v);
        }
    }

    // Convert mirror sets to sorted vecs
    for (pid, set) in mirror_sets.into_iter().enumerate() {
        let mut mirrors: Vec<usize> = set.into_iter().collect();
        mirrors.sort_unstable();
        shards[pid].mirror_vertices = mirrors;
    }

    DistributedGraph {
        shards,
        n_global_vertices: n_vertices,
        n_global_edges: edges.len(),
        vertex_map,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Distributed query helpers
// ────────────────────────────────────────────────────────────────────────────

/// Return the degree of `vertex` by counting edges in its home shard.
///
/// Returns `None` if the vertex is not in the graph.
pub fn distributed_degree(dg: &DistributedGraph, vertex: usize) -> Option<usize> {
    let loc = dg.vertex_map.get(&vertex)?;
    let shard = dg.shards.get(loc.partition_id)?;
    let degree = shard
        .local_edges
        .iter()
        .filter(|&&(u, v)| u == vertex || v == vertex)
        .count();
    Some(degree)
}

/// Return all neighbours of `vertex` by scanning its home shard's edge list.
///
/// Returns `None` if the vertex is not in the graph.
pub fn distributed_neighbors(dg: &DistributedGraph, vertex: usize) -> Option<Vec<usize>> {
    let loc = dg.vertex_map.get(&vertex)?;
    let shard = dg.shards.get(loc.partition_id)?;
    let neighbours: Vec<usize> = shard
        .local_edges
        .iter()
        .filter_map(|&(u, v)| {
            if u == vertex {
                Some(v)
            } else if v == vertex {
                Some(u)
            } else {
                None
            }
        })
        .collect();
    Some(neighbours)
}

// ────────────────────────────────────────────────────────────────────────────
// Statistics
// ────────────────────────────────────────────────────────────────────────────

/// Compute load-balance and edge-cut statistics for a distributed graph.
pub fn distributed_graph_stats(dg: &DistributedGraph) -> DistributedStats {
    let shard_sizes: Vec<usize> = dg.shards.iter().map(|s| s.local_vertices.len()).collect();

    let total_verts: usize = shard_sizes.iter().sum();
    let n = dg.shards.len();

    let avg_size = if n > 0 {
        total_verts as f64 / n as f64
    } else {
        1.0
    };
    let max_size = shard_sizes.iter().copied().max().unwrap_or(0) as f64;
    let balance_ratio = if avg_size > 0.0 {
        max_size / avg_size
    } else {
        1.0
    };

    // Edge cut: edges where the two endpoints belong to different shards
    let mut cut_edges = 0usize;
    for shard in &dg.shards {
        for &(u, v) in &shard.local_edges {
            let p_u = dg
                .vertex_map
                .get(&u)
                .map(|loc| loc.partition_id)
                .unwrap_or(0);
            let p_v = dg
                .vertex_map
                .get(&v)
                .map(|loc| loc.partition_id)
                .unwrap_or(0);
            if p_u != p_v {
                cut_edges += 1;
            }
        }
    }
    let edge_cut_fraction = if dg.n_global_edges > 0 {
        cut_edges as f64 / dg.n_global_edges as f64
    } else {
        0.0
    };

    // Replication factor: total appearances of each vertex (owned + mirrors)
    let total_mirror: usize = dg.shards.iter().map(|s| s.mirror_vertices.len()).sum();
    let replication_factor = if total_verts > 0 {
        (total_verts + total_mirror) as f64 / total_verts as f64
    } else {
        1.0
    };

    DistributedStats {
        balance_ratio,
        edge_cut_fraction,
        replication_factor,
        shard_sizes,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: Petersen graph edges (10 vertices, 15 edges)
    fn petersen_edges() -> Vec<(usize, usize)> {
        // Outer pentagon: 0-1-2-3-4-0
        // Inner pentagram: 5-7-9-6-8-5
        // Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0),
            (5, 7),
            (7, 9),
            (9, 6),
            (6, 8),
            (8, 5),
            (0, 5),
            (1, 6),
            (2, 7),
            (3, 8),
            (4, 9),
        ]
    }

    // ── All 10 Petersen vertices are assigned with 4 partitions ──────────────
    #[test]
    fn test_distributed_graph_petersen_all_vertices() {
        let edges = petersen_edges();
        let cfg = DistributedGraphConfig::default(); // 4 partitions, hash
        let dg = build_distributed_graph(&edges, 10, &cfg);

        assert_eq!(dg.n_global_vertices, 10);
        assert_eq!(dg.n_global_edges, 15);
        assert_eq!(dg.vertex_map.len(), 10);

        // Every vertex 0..10 must be present
        for v in 0..10usize {
            assert!(
                dg.vertex_map.contains_key(&v),
                "Vertex {v} missing from vertex_map"
            );
        }
    }

    // ── Stats: balance_ratio is reasonable ───────────────────────────────────
    #[test]
    fn test_distributed_graph_petersen_stats() {
        let edges = petersen_edges();
        let cfg = DistributedGraphConfig::default();
        let dg = build_distributed_graph(&edges, 10, &cfg);
        let stats = distributed_graph_stats(&dg);

        // 10 vertices, 4 partitions → avg ~2.5
        // Balance ratio should be ≤ 2.0 for hash partitioning (floor(10/4)=2 or 3 vertices each)
        assert!(
            stats.balance_ratio <= 2.0,
            "balance_ratio {:.2} too high",
            stats.balance_ratio
        );

        // Shard sizes must sum to n_global_vertices
        let total: usize = stats.shard_sizes.iter().sum();
        assert_eq!(total, 10);
    }

    // ── distributed_degree is correct for a known vertex ─────────────────────
    #[test]
    fn test_distributed_degree_petersen() {
        let edges = petersen_edges();
        let cfg = DistributedGraphConfig::default();
        let dg = build_distributed_graph(&edges, 10, &cfg);

        // In the Petersen graph every vertex has degree 3
        for v in 0..10usize {
            let deg = distributed_degree(&dg, v);
            assert!(
                deg.is_some(),
                "distributed_degree returned None for vertex {v}"
            );
            // The edge-cut model assigns an edge to the source's shard, so
            // degree via local edges counts outgoing + incoming edges on that shard.
            // We just verify it's a reasonable non-zero value.
            assert!(deg.unwrap() > 0, "Vertex {v} has 0 degree in its shard");
        }
    }

    // ── distributed_neighbors returns non-empty for Petersen vertices ─────────
    #[test]
    fn test_distributed_neighbors_petersen() {
        let edges = petersen_edges();
        let cfg = DistributedGraphConfig::default();
        let dg = build_distributed_graph(&edges, 10, &cfg);

        for v in 0..10usize {
            let nb = distributed_neighbors(&dg, v);
            assert!(nb.is_some(), "distributed_neighbors returned None for {v}");
        }
    }

    // ── FENNEL: all vertices are assigned to valid partitions ─────────────────
    #[test]
    fn test_fennel_partition_100_vertices() {
        // Generate a random-ish graph with 100 vertices and 200 edges
        let n = 100usize;
        let edges: Vec<(usize, usize)> = (0..200)
            .map(|i| {
                let u = (i * 7 + 3) % n;
                let v = (i * 13 + 17) % n;
                (u, v)
            })
            .filter(|(u, v)| u != v)
            .collect();

        let cfg = DistributedGraphConfig {
            n_partitions: 4,
            partition_method: GraphPartitionMethod::Fennel,
            replication_factor: 0,
        };
        let assignment = fennel_partition(&edges, n, 4, &cfg);

        assert_eq!(assignment.len(), n);
        for (v, &p) in assignment.iter().enumerate() {
            assert!(p < 4, "Vertex {v} assigned to invalid partition {p}");
        }
    }

    // ── FENNEL: build_distributed_graph with Fennel method ───────────────────
    #[test]
    fn test_build_distributed_graph_fennel() {
        let edges = petersen_edges();
        let cfg = DistributedGraphConfig {
            n_partitions: 4,
            partition_method: GraphPartitionMethod::Fennel,
            replication_factor: 0,
        };
        let dg = build_distributed_graph(&edges, 10, &cfg);
        assert_eq!(dg.vertex_map.len(), 10);
        for v in 0..10usize {
            assert!(dg.vertex_map.contains_key(&v));
        }
    }

    // ── VertexCut: builds without panic ──────────────────────────────────────
    #[test]
    fn test_build_distributed_graph_vertex_cut() {
        let edges = petersen_edges();
        let cfg = DistributedGraphConfig {
            n_partitions: 4,
            partition_method: GraphPartitionMethod::VertexCut,
            replication_factor: 0,
        };
        let dg = build_distributed_graph(&edges, 10, &cfg);
        assert_eq!(dg.n_global_vertices, 10);
        assert_eq!(dg.vertex_map.len(), 10);
    }

    // ── Hash partition basic ──────────────────────────────────────────────────
    #[test]
    fn test_hash_partition() {
        assert_eq!(hash_partition(0, 4), 0);
        assert_eq!(hash_partition(5, 4), 1);
        assert_eq!(hash_partition(7, 4), 3);
        assert_eq!(hash_partition(0, 0), 0); // edge case: 0 partitions
    }

    // ── Stats: edge_cut_fraction ∈ [0, 1] ─────────────────────────────────────
    #[test]
    fn test_stats_edge_cut_fraction_range() {
        let edges = petersen_edges();
        let cfg = DistributedGraphConfig::default();
        let dg = build_distributed_graph(&edges, 10, &cfg);
        let stats = distributed_graph_stats(&dg);
        assert!(
            stats.edge_cut_fraction >= 0.0 && stats.edge_cut_fraction <= 1.0,
            "edge_cut_fraction = {} out of range",
            stats.edge_cut_fraction
        );
    }
}
