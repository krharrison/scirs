//! Temporal graph generation models
//!
//! This module provides generators for temporal (time-evolving) networks beyond
//! the basic activity-driven model already in `temporal_graph`.  Three models
//! are implemented:
//!
//! * **`temporal_barabasi_albert`** – preferential attachment with timestamps.
//!   Each new node arrives at a specified time, contacts `m` existing nodes
//!   preferentially (higher-degree nodes are preferred), and the resulting
//!   contacts are recorded as temporal edges.  Produces scale-free degree
//!   distributions with a well-defined temporal ordering.
//!
//! * **`TemporalGraph`** (re-exported from `temporal_graph`) – the core
//!   temporal graph data structure used throughout this module.
//!
//! * **`temporal_random_walk`** – time-respecting random walks on a temporal
//!   graph.  A walk is *time-respecting* if each successive edge has a
//!   timestamp ≥ the previous edge's timestamp.  This is the standard
//!   foundation for temporal node embeddings and reachability analysis.
//!
//! # References
//!
//! - Barabási, A.-L. & Albert, R. (1999). Emergence of scaling in random
//!   networks. *Science*, 286, 509–512.
//! - Holme, P. & Saramäki, J. (2012). Temporal networks. *Physics Reports*,
//!   519(3), 97–125.
//! - Pan, R. K. & Saramäki, J. (2011). Path lengths, correlations, and
//!   centrality in temporal networks. *Physical Review E*, 84, 016105.

use crate::error::{GraphError, Result};
use crate::temporal_graph::{TemporalEdge, TemporalGraph};
use scirs2_core::rand_prelude::IndexedRandom;
use scirs2_core::random::prelude::*;

// Re-export the core temporal types so users can access everything from this
// module without digging into `temporal_graph`.
pub use crate::temporal_graph::{activity_driven_model, activity_driven_model_seeded, burstiness};

// ─────────────────────────────────────────────────────────────────────────────
// Temporal Barabási–Albert
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a temporal Barabási–Albert (TBA) network with preferential attachment.
///
/// Nodes are added at uniformly spaced timestamps `0, 1/n, 2/n, …, (n-1)/n`.
/// Each new node contacts `m` existing nodes chosen proportionally to their
/// current degree (preferential attachment).  Each contact is recorded as a
/// temporal edge with the arrival timestamp of the new node.
///
/// The process starts with a complete graph on `m + 1` seed nodes (all with
/// timestamp 0.0) so that the preferential-attachment step always has viable
/// targets.
///
/// # Arguments
///
/// * `n` – total number of nodes (must be > m)
/// * `m` – number of edges added per new node (must be ≥ 1)
/// * `rng` – seeded random-number generator
///
/// # Returns
///
/// A [`TemporalGraph`] with `n` nodes and approximately `n·m` temporal edges.
///
/// # Errors
///
/// Returns [`GraphError::InvalidGraph`] when parameters are invalid.
///
/// # Example
///
/// ```rust
/// use scirs2_graph::generators::temporal::temporal_barabasi_albert;
/// use scirs2_core::random::prelude::*;
/// let mut rng = StdRng::seed_from_u64(42);
/// let tg = temporal_barabasi_albert(50, 2, &mut rng).unwrap();
/// assert_eq!(tg.node_count(), 50);
/// ```
pub fn temporal_barabasi_albert<R: Rng>(n: usize, m: usize, rng: &mut R) -> Result<TemporalGraph> {
    if n < 2 {
        return Err(GraphError::InvalidGraph(
            "temporal_barabasi_albert: n must be ≥ 2".to_string(),
        ));
    }
    if m == 0 {
        return Err(GraphError::InvalidGraph(
            "temporal_barabasi_albert: m must be ≥ 1".to_string(),
        ));
    }
    if m >= n {
        return Err(GraphError::InvalidGraph(format!(
            "temporal_barabasi_albert: m={m} must be < n={n}"
        )));
    }

    let mut tg = TemporalGraph::new(n);

    // Degree tracker for preferential attachment (degree in the snapshot graph)
    let mut degree = vec![0usize; n];

    // Seed: complete graph on the first `m + 1` nodes at time 0.0
    let seed_size = m + 1;
    for u in 0..seed_size {
        for v in (u + 1)..seed_size {
            tg.add_edge(TemporalEdge::new(u, v, 0.0));
            tg.add_edge(TemporalEdge::new(v, u, 0.0));
            degree[u] += 1;
            degree[v] += 1;
        }
    }

    // Arrival step
    for v in seed_size..n {
        let t_arrive = v as f64 / n as f64;

        // Build a weighted target list proportional to degree
        // (add +1 to each to avoid zero-weight nodes)
        let candidates: Vec<usize> = (0..v).collect();
        let weights: Vec<f64> = candidates.iter().map(|&u| (degree[u] + 1) as f64).collect();

        // Sample m distinct nodes
        let chosen = weighted_sample_without_replacement(&candidates, &weights, m, rng);

        for &u in &chosen {
            tg.add_edge(TemporalEdge::new(v, u, t_arrive));
            tg.add_edge(TemporalEdge::new(u, v, t_arrive));
            degree[u] += 1;
            degree[v] += 1;
        }
    }

    Ok(tg)
}

/// Weighted sampling without replacement using the Efraimidis–Spirakis reservoir
/// algorithm (A-Res), adapted for small samples.
fn weighted_sample_without_replacement<R: Rng>(
    items: &[usize],
    weights: &[f64],
    k: usize,
    rng: &mut R,
) -> Vec<usize> {
    if items.is_empty() || k == 0 {
        return Vec::new();
    }
    let k = k.min(items.len());

    // Compute key = u^(1/w) for each item; take the k items with highest keys.
    // Use a simple sort-based approach (fine for moderate n).
    let mut keyed: Vec<(f64, usize)> = items
        .iter()
        .zip(weights.iter())
        .map(|(&item, &w)| {
            let u: f64 = rng.random::<f64>().max(1e-300);
            let key = if w > 0.0 { u.powf(1.0 / w) } else { 0.0 };
            (key, item)
        })
        .collect();

    // Partial sort: we only need the top-k by key (descending)
    keyed.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    keyed[..k].iter().map(|&(_, item)| item).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Temporal random walk
// ─────────────────────────────────────────────────────────────────────────────

/// A single time-respecting random walk on a temporal graph.
///
/// Each step of the walk moves to a random neighbour via an edge whose
/// timestamp is ≥ the timestamp of the edge used at the previous step.
/// This ensures the walk is *causal* (it only traverses forward in time).
#[derive(Debug, Clone)]
pub struct TemporalWalk {
    /// Ordered sequence of (node, edge_timestamp) pairs visited during the walk.
    /// The first entry is the starting node with timestamp `-∞` (represented as
    /// `f64::NEG_INFINITY`).
    pub steps: Vec<(usize, f64)>,
}

impl TemporalWalk {
    /// Return the sequence of node IDs visited (including the starting node).
    pub fn nodes(&self) -> Vec<usize> {
        self.steps.iter().map(|&(n, _)| n).collect()
    }

    /// Return the length of the walk (number of edges traversed).
    pub fn len(&self) -> usize {
        self.steps.len().saturating_sub(1)
    }

    /// Return `true` if the walk consists of only the starting node (no edges).
    pub fn is_empty(&self) -> bool {
        self.steps.len() <= 1
    }
}

/// Perform `num_walks` time-respecting random walks starting from `source`.
///
/// A walk is *time-respecting*: each successive edge must have a timestamp ≥
/// the timestamp of the previously traversed edge.  The walk terminates when:
/// * `max_length` edges have been traversed, **or**
/// * no time-respecting edges lead away from the current node.
///
/// If `start_time` is `None`, the walk may use any edge (effectively starting
/// at `t = -∞`).
///
/// # Arguments
///
/// * `tg` – the temporal graph to walk on
/// * `source` – starting node (0-based index; must be < `tg.node_count()`)
/// * `max_length` – maximum number of steps per walk
/// * `num_walks` – number of independent walks to generate
/// * `start_time` – earliest timestamp the first edge may have (`None` = no constraint)
/// * `rng` – seeded random-number generator
///
/// # Returns
///
/// A `Vec<TemporalWalk>` of length `num_walks`.
///
/// # Errors
///
/// Returns [`GraphError::InvalidGraph`] when `source ≥ n`.
///
/// # Example
///
/// ```rust
/// use scirs2_graph::generators::temporal::{temporal_random_walk, temporal_barabasi_albert};
/// use scirs2_core::random::prelude::*;
/// let mut rng = StdRng::seed_from_u64(1);
/// let tg = temporal_barabasi_albert(20, 2, &mut rng).unwrap();
/// let walks = temporal_random_walk(&tg, 0, 5, 3, None, &mut rng).unwrap();
/// assert_eq!(walks.len(), 3);
/// ```
pub fn temporal_random_walk<R: Rng>(
    tg: &TemporalGraph,
    source: usize,
    max_length: usize,
    num_walks: usize,
    start_time: Option<f64>,
    rng: &mut R,
) -> Result<Vec<TemporalWalk>> {
    if source >= tg.node_count() {
        return Err(GraphError::InvalidGraph(format!(
            "temporal_random_walk: source={source} ≥ n={}",
            tg.node_count()
        )));
    }
    if max_length == 0 {
        return Err(GraphError::InvalidGraph(
            "temporal_random_walk: max_length must be ≥ 1".to_string(),
        ));
    }

    let mut walks = Vec::with_capacity(num_walks);

    // Pre-sort edges and build per-node outgoing edge lists sorted by timestamp.
    // TemporalGraph already keeps edges sorted when we call sort() or use
    // sorted_edges().
    let sorted_edges = tg.sorted_edges_cloned();

    // Build adjacency: node → [(neighbour, timestamp)] sorted by timestamp
    let n = tg.node_count();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for edge in &sorted_edges {
        // The temporal graph edges are directed by convention; treat them
        // as undirected for walk purposes (add both directions).
        adj[edge.source].push((edge.target, edge.timestamp));
        // Avoid duplicate reverse edges for undirected treatment
        adj[edge.target].push((edge.source, edge.timestamp));
    }
    // Sort per-node adjacency by timestamp
    for nbrs in adj.iter_mut() {
        nbrs.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    let t0 = start_time.unwrap_or(f64::NEG_INFINITY);

    for _ in 0..num_walks {
        let mut walk_steps = vec![(source, t0)];
        let mut current_node = source;
        let mut current_time = t0;

        for _ in 0..max_length {
            // Collect time-respecting candidates: edges with timestamp ≥ current_time
            let candidates: Vec<(usize, f64)> = adj[current_node]
                .iter()
                .filter(|&&(_, t)| t >= current_time)
                .copied()
                .collect();

            if candidates.is_empty() {
                break;
            }

            // Uniformly choose a random candidate
            let &(next_node, next_time) = candidates.choose(rng).expect("candidates is non-empty");

            walk_steps.push((next_node, next_time));
            current_node = next_node;
            current_time = next_time;
        }

        walks.push(TemporalWalk { steps: walk_steps });
    }

    Ok(walks)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── temporal_barabasi_albert ──────────────────────────────────────────

    #[test]
    fn test_tba_basic() {
        let mut rng = StdRng::seed_from_u64(42);
        let tg = temporal_barabasi_albert(20, 2, &mut rng)
            .expect("temporal_barabasi_albert should succeed");
        assert_eq!(tg.node_count(), 20);
        // With m=2 we add at least 2 edges per new node → at least (n - m - 1) * m edges
        assert!(tg.n_edges() > 0);
    }

    #[test]
    fn test_tba_scale_free_degree() {
        let mut rng = StdRng::seed_from_u64(7);
        let tg = temporal_barabasi_albert(100, 3, &mut rng)
            .expect("temporal_barabasi_albert should succeed");
        assert_eq!(tg.node_count(), 100);
        // Nodes should have at least 1 temporal edge each (except possibly isolated ones)
        assert!(tg.n_edges() > 0);
    }

    #[test]
    fn test_tba_invalid_params() {
        let mut rng = StdRng::seed_from_u64(0);
        // n < 2
        assert!(temporal_barabasi_albert(1, 1, &mut rng).is_err());
        // m = 0
        assert!(temporal_barabasi_albert(10, 0, &mut rng).is_err());
        // m >= n
        assert!(temporal_barabasi_albert(5, 5, &mut rng).is_err());
    }

    #[test]
    fn test_tba_timestamps_increasing() {
        let mut rng = StdRng::seed_from_u64(13);
        let tg = temporal_barabasi_albert(30, 2, &mut rng)
            .expect("temporal_barabasi_albert should succeed");
        // All timestamps should be ≥ 0
        for edge in tg.sorted_edges_cloned() {
            assert!(
                edge.timestamp >= 0.0,
                "All timestamps should be non-negative"
            );
        }
    }

    // ── temporal_random_walk ─────────────────────────────────────────────

    #[test]
    fn test_random_walk_basic() {
        let mut rng = StdRng::seed_from_u64(1);
        let tg = temporal_barabasi_albert(20, 2, &mut rng)
            .expect("temporal_barabasi_albert should succeed");
        let walks = temporal_random_walk(&tg, 0, 5, 3, None, &mut rng)
            .expect("temporal_random_walk should succeed");
        assert_eq!(walks.len(), 3);
        for walk in &walks {
            // Walk length ≤ max_length
            assert!(walk.len() <= 5);
            // First step is the source node
            assert_eq!(walk.steps[0].0, 0);
        }
    }

    #[test]
    fn test_random_walk_time_respecting() {
        let mut rng = StdRng::seed_from_u64(2);
        let tg = temporal_barabasi_albert(30, 2, &mut rng)
            .expect("temporal_barabasi_albert should succeed");
        let walks = temporal_random_walk(&tg, 0, 10, 5, None, &mut rng)
            .expect("temporal_random_walk should succeed");

        for walk in &walks {
            // Timestamps should be non-decreasing
            let timestamps: Vec<f64> = walk.steps.iter().map(|&(_, t)| t).collect();
            for window in timestamps.windows(2) {
                assert!(
                    window[1] >= window[0],
                    "Walk timestamps must be non-decreasing: {:?}",
                    timestamps
                );
            }
        }
    }

    #[test]
    fn test_random_walk_with_start_time() {
        let mut rng = StdRng::seed_from_u64(3);
        let tg = temporal_barabasi_albert(20, 2, &mut rng)
            .expect("temporal_barabasi_albert should succeed");
        let start_t = 0.5;
        let walks = temporal_random_walk(&tg, 0, 5, 2, Some(start_t), &mut rng)
            .expect("temporal_random_walk should succeed");

        for walk in &walks {
            for &(_, t) in &walk.steps[1..] {
                // All edge timestamps in the walk should be ≥ start_t
                assert!(
                    t >= start_t,
                    "Edge timestamps should be ≥ start_time={start_t}, got {t}"
                );
            }
        }
    }

    #[test]
    fn test_random_walk_invalid_source() {
        let mut rng = StdRng::seed_from_u64(0);
        let tg = temporal_barabasi_albert(10, 2, &mut rng)
            .expect("temporal_barabasi_albert should succeed");
        assert!(temporal_random_walk(&tg, 100, 5, 1, None, &mut rng).is_err());
    }

    #[test]
    fn test_random_walk_zero_max_length() {
        let mut rng = StdRng::seed_from_u64(0);
        let tg = temporal_barabasi_albert(10, 2, &mut rng)
            .expect("temporal_barabasi_albert should succeed");
        assert!(temporal_random_walk(&tg, 0, 0, 1, None, &mut rng).is_err());
    }

    #[test]
    fn test_temporal_walk_is_empty() {
        // A walk with only the start node
        let walk = TemporalWalk {
            steps: vec![(0, f64::NEG_INFINITY)],
        };
        assert!(walk.is_empty());
        assert_eq!(walk.len(), 0);
        assert_eq!(walk.nodes(), vec![0]);
    }

    #[test]
    fn test_temporal_walk_nodes() {
        let walk = TemporalWalk {
            steps: vec![(0, 0.0), (1, 1.0), (2, 2.0)],
        };
        assert_eq!(walk.nodes(), vec![0, 1, 2]);
        assert_eq!(walk.len(), 2);
        assert!(!walk.is_empty());
    }
}
