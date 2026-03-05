//! Graph dataset generators.
//!
//! This module provides well-known benchmark graphs and synthetic graph
//! generators for testing graph algorithms.
//!
//! # Generators
//!
//! - [`make_karate_club`]      – Zachary's karate club social network.
//! - [`make_sbm`]              – Stochastic block model.
//! - [`make_barabasi_albert`]  – Barabási–Albert preferential attachment.
//! - [`make_watts_strogatz`]   – Watts–Strogatz small-world model.

use crate::error::{DatasetsError, Result};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

// ─────────────────────────────────────────────────────────────────────────────
// make_karate_club
// ─────────────────────────────────────────────────────────────────────────────

/// Return Zachary's karate club graph as an edge list with community labels.
///
/// The network has 34 nodes and 78 undirected edges.  After a conflict, the
/// club split into two factions led by node 0 (instructor) and node 33
/// (administrator/president).
///
/// # Returns
///
/// `(edges, labels)` where
/// - `edges`  is a `Vec<(usize, usize)>` of the 78 undirected edges.
/// - `labels` is a `Vec<usize>` of length 34: `0` for the instructor faction
///   and `1` for the administrator faction.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::graph_datasets::make_karate_club;
///
/// let (edges, labels) = make_karate_club();
/// assert_eq!(edges.len(), 78);
/// assert_eq!(labels.len(), 34);
/// ```
pub fn make_karate_club() -> (Vec<(usize, usize)>, Vec<usize>) {
    // Zachary 1977 – 78 undirected edges (0-indexed)
    let edges: Vec<(usize, usize)> = vec![
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
        (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
        (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
        (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
        (3, 7), (3, 12), (3, 13),
        (4, 6), (4, 10),
        (5, 6), (5, 10), (5, 16),
        (6, 16),
        (8, 30), (8, 32), (8, 33),
        (9, 33),
        (13, 33),
        (14, 32), (14, 33),
        (15, 32), (15, 33),
        (18, 32), (18, 33),
        (19, 33),
        (20, 32), (20, 33),
        (22, 32), (22, 33),
        (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
        (24, 25), (24, 27), (24, 31),
        (25, 31),
        (26, 29), (26, 33),
        (27, 33),
        (28, 31), (28, 33),
        (29, 32), (29, 33),
        (30, 32), (30, 33),
        (31, 32), (31, 33),
        (32, 33),
    ];

    // Community labels: 0 = instructor's faction, 1 = administrator's faction
    // Based on the original Zachary partition
    let labels: Vec<usize> = vec![
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 1, 1, 0, 0, 1, 0,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1,
    ];

    (edges, labels)
}

// ─────────────────────────────────────────────────────────────────────────────
// make_sbm
// ─────────────────────────────────────────────────────────────────────────────

/// Generate an undirected Stochastic Block Model (SBM) graph.
///
/// Within-block edges exist with probability `p_within`; between-block edges
/// exist with probability `p_between`.
///
/// # Arguments
///
/// * `block_sizes` – Number of nodes in each block (must all be > 0).
/// * `p_within`    – Intra-block edge probability (0.0 – 1.0).
/// * `p_between`   – Inter-block edge probability (0.0 – 1.0).
/// * `seed`        – Random seed.
///
/// # Returns
///
/// `Vec<(usize, usize)>` of undirected edges (u < v guaranteed).
///
/// # Errors
///
/// Returns an error if any block size is 0, or probabilities are out of range.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::graph_datasets::make_sbm;
///
/// let edges = make_sbm(&[10, 10, 10], 0.5, 0.05, 42).expect("sbm failed");
/// // Expect roughly 3*(10*9/2)*0.5 + 3*(10*10)*0.05 ≈ 67 + 15 = 82 edges
/// assert!(!edges.is_empty());
/// ```
pub fn make_sbm(
    block_sizes: &[usize],
    p_within: f64,
    p_between: f64,
    seed: u64,
) -> Result<Vec<(usize, usize)>> {
    if block_sizes.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "make_sbm: block_sizes must not be empty".to_string(),
        ));
    }
    for (i, &sz) in block_sizes.iter().enumerate() {
        if sz == 0 {
            return Err(DatasetsError::InvalidFormat(format!(
                "make_sbm: block_sizes[{i}] must be > 0"
            )));
        }
    }
    if !(0.0..=1.0).contains(&p_within) {
        return Err(DatasetsError::InvalidFormat(
            "make_sbm: p_within must be in [0, 1]".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&p_between) {
        return Err(DatasetsError::InvalidFormat(
            "make_sbm: p_between must be in [0, 1]".to_string(),
        ));
    }

    let mut rng = make_rng(seed);
    let uniform = scirs2_core::random::Uniform::new(0.0_f64, 1.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform distribution creation failed: {e}"))
    })?;

    // Compute global node offset for each block
    let mut offsets: Vec<usize> = Vec::with_capacity(block_sizes.len() + 1);
    offsets.push(0);
    for &sz in block_sizes {
        offsets.push(offsets.last().copied().unwrap_or(0) + sz);
    }
    let n_total = *offsets.last().unwrap_or(&0);

    // Assign block membership for each node
    let mut block_of: Vec<usize> = vec![0; n_total];
    for (b, &sz) in block_sizes.iter().enumerate() {
        let start = offsets[b];
        let end = offsets[b + 1];
        for node in start..end {
            block_of[node] = b;
        }
    }

    let mut edges: Vec<(usize, usize)> = Vec::new();
    for u in 0..n_total {
        for v in (u + 1)..n_total {
            let p = if block_of[u] == block_of[v] {
                p_within
            } else {
                p_between
            };
            if p >= 1.0 || (p > 0.0 && uniform.sample(&mut rng) < p) {
                edges.push((u, v));
            }
        }
    }

    Ok(edges)
}

// ─────────────────────────────────────────────────────────────────────────────
// make_barabasi_albert
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a Barabási–Albert preferential attachment graph.
///
/// Starting from a complete seed graph of `m` nodes, each new node connects
/// to `m` existing nodes with probability proportional to their current degree.
///
/// # Arguments
///
/// * `n`    – Total number of nodes (must be > m).
/// * `m`    – Number of edges each new node attaches (must be ≥ 1).
/// * `seed` – Random seed.
///
/// # Returns
///
/// `Vec<(usize, usize)>` of undirected edges (u < v guaranteed).
///
/// # Errors
///
/// Returns an error if `m == 0` or `n <= m`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::graph_datasets::make_barabasi_albert;
///
/// let edges = make_barabasi_albert(100, 2, 42).expect("BA failed");
/// // Seed: 2 nodes + 1 edge. Then 98 nodes × 2 edges = 196 additional edges → 197 total.
/// assert_eq!(edges.len(), 197);
/// ```
pub fn make_barabasi_albert(n: usize, m: usize, seed: u64) -> Result<Vec<(usize, usize)>> {
    if m == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_barabasi_albert: m must be >= 1".to_string(),
        ));
    }
    if n <= m {
        return Err(DatasetsError::InvalidFormat(format!(
            "make_barabasi_albert: n ({n}) must be > m ({m})"
        )));
    }

    let mut rng = make_rng(seed);

    let mut edges: Vec<(usize, usize)> = Vec::new();
    // degree[i] = degree of node i (for preferential attachment)
    let mut degree: Vec<usize> = vec![0usize; n];

    // Seed: fully connect first (m+1) nodes (or m nodes with m-1 edges)
    // We use the standard NetworkX approach: start with m nodes in a star/path,
    // then add nodes one at a time.
    // Seed graph: nodes 0..m form a star around node 0
    for v in 1..m {
        edges.push((0, v));
        degree[0] += 1;
        degree[v] += 1;
    }

    // "Repeated nodes" list used for alias-method preferential attachment
    // Each node appears proportional to its degree
    let mut repeated: Vec<usize> = Vec::with_capacity(2 * (n * m));
    for v in 0..m {
        for _ in 0..degree[v].max(1) {
            repeated.push(v);
        }
    }

    // Add nodes m..n
    for new_node in m..n {
        let mut targets: Vec<usize> = Vec::with_capacity(m);
        let mut chosen: std::collections::HashSet<usize> = std::collections::HashSet::new();

        while targets.len() < m {
            if repeated.is_empty() {
                break;
            }
            let idx = {
                let uniform_idx = scirs2_core::random::Uniform::new(0usize, repeated.len())
                    .map_err(|e| DatasetsError::ComputationError(format!("{e}")))?;
                uniform_idx.sample(&mut rng)
            };
            let candidate = repeated[idx];
            if !chosen.contains(&candidate) {
                chosen.insert(candidate);
                targets.push(candidate);
            }
        }

        for &t in &targets {
            let u = new_node.min(t);
            let v = new_node.max(t);
            edges.push((u, v));
            degree[new_node] += 1;
            degree[t] += 1;
            repeated.push(new_node);
            repeated.push(t);
        }
    }

    Ok(edges)
}

// ─────────────────────────────────────────────────────────────────────────────
// make_watts_strogatz
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a Watts–Strogatz small-world graph.
///
/// Starting from a ring lattice where each node connects to its `k` nearest
/// neighbors (k/2 each side), each edge is rewired with probability `p`.
///
/// # Arguments
///
/// * `n`    – Number of nodes (must be > k).
/// * `k`    – Number of nearest neighbors in the ring (must be even and ≥ 2).
/// * `p`    – Rewiring probability (0.0 = regular lattice, 1.0 = random graph).
/// * `seed` – Random seed.
///
/// # Returns
///
/// `Vec<(usize, usize)>` of undirected edges (u < v guaranteed).
///
/// # Errors
///
/// Returns an error if `k` is odd, `k < 2`, `n <= k`, or `p` is out of range.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::graph_datasets::make_watts_strogatz;
///
/// let edges = make_watts_strogatz(20, 4, 0.1, 42).expect("WS failed");
/// // Ring lattice: 20 * 2 = 40 edges before rewiring
/// assert_eq!(edges.len(), 40);
/// ```
pub fn make_watts_strogatz(n: usize, k: usize, p: f64, seed: u64) -> Result<Vec<(usize, usize)>> {
    if k < 2 {
        return Err(DatasetsError::InvalidFormat(
            "make_watts_strogatz: k must be >= 2".to_string(),
        ));
    }
    if k % 2 != 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_watts_strogatz: k must be even".to_string(),
        ));
    }
    if n <= k {
        return Err(DatasetsError::InvalidFormat(format!(
            "make_watts_strogatz: n ({n}) must be > k ({k})"
        )));
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(DatasetsError::InvalidFormat(
            "make_watts_strogatz: p must be in [0, 1]".to_string(),
        ));
    }

    let mut rng = make_rng(seed);
    let uniform = scirs2_core::random::Uniform::new(0.0_f64, 1.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform creation failed: {e}"))
    })?;

    // Build ring lattice: node i connects to i+1, i+2, ..., i+k/2 (mod n)
    // Represent as adjacency set for efficient rewiring
    let half = k / 2;
    // adj[u] = set of neighbors (we maintain u < v canonical form at the end)
    let mut adj: Vec<std::collections::HashSet<usize>> =
        (0..n).map(|_| std::collections::HashSet::new()).collect();

    for u in 0..n {
        for j in 1..=half {
            let v = (u + j) % n;
            adj[u].insert(v);
            adj[v].insert(u);
        }
    }

    // Rewiring pass (Watts–Strogatz algorithm)
    for u in 0..n {
        for j in 1..=half {
            let v = (u + j) % n;
            // Only rewire edges where u < v to avoid double processing
            if u >= v {
                continue;
            }
            if uniform.sample(&mut rng) < p {
                // Remove edge (u, v)
                adj[u].remove(&v);
                adj[v].remove(&u);

                // Pick a new target w ≠ u, w not already a neighbor of u
                let max_attempts = n * 10;
                let mut attempts = 0;
                let uniform_n = scirs2_core::random::Uniform::new(0usize, n).map_err(|e| {
                    DatasetsError::ComputationError(format!("{e}"))
                })?;
                loop {
                    let w = uniform_n.sample(&mut rng);
                    if w != u && !adj[u].contains(&w) {
                        adj[u].insert(w);
                        adj[w].insert(u);
                        break;
                    }
                    attempts += 1;
                    if attempts >= max_attempts {
                        // Could not rewire – restore original edge
                        adj[u].insert(v);
                        adj[v].insert(u);
                        break;
                    }
                }
            }
        }
    }

    // Collect canonical edges (u < v)
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for u in 0..n {
        for &v in &adj[u] {
            if u < v {
                edges.push((u, v));
            }
        }
    }
    edges.sort_unstable();

    Ok(edges)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── make_karate_club ─────────────────────────────────────────────────────

    #[test]
    fn test_karate_club_edges() {
        let (edges, labels) = make_karate_club();
        assert_eq!(edges.len(), 78, "karate club should have 78 edges");
        assert_eq!(labels.len(), 34, "karate club should have 34 nodes");
    }

    #[test]
    fn test_karate_club_labels_binary() {
        let (_, labels) = make_karate_club();
        for &l in &labels {
            assert!(l == 0 || l == 1, "labels must be 0 or 1, got {l}");
        }
    }

    #[test]
    fn test_karate_club_canonical_edges() {
        let (edges, _) = make_karate_club();
        for &(u, v) in &edges {
            assert!(u < v, "edges must be canonical (u < v); got ({u}, {v})");
            assert!(u < 34 && v < 34, "node indices must be < 34");
        }
    }

    // ── make_sbm ─────────────────────────────────────────────────────────────

    #[test]
    fn test_sbm_basic() {
        let edges = make_sbm(&[10, 10, 10], 0.5, 0.05, 42).expect("sbm basic");
        // Should produce edges; exact count is stochastic
        // With p_within=0.5 and 3 blocks of 10, at least some edges expected
        assert!(!edges.is_empty());
    }

    #[test]
    fn test_sbm_no_between_edges() {
        // With p_between=0.0 all edges are within blocks
        let edges = make_sbm(&[5, 5], 1.0, 0.0, 1).expect("sbm no between");
        // Both blocks fully connected → 2*(5*4/2) = 20 edges
        assert_eq!(edges.len(), 20, "expected 20 within-block edges with p_within=1.0");
    }

    #[test]
    fn test_sbm_all_edges() {
        // Both p_within=1.0 and p_between=1.0 → complete graph K10
        let edges = make_sbm(&[5, 5], 1.0, 1.0, 1).expect("sbm all edges");
        // K10 → 10*9/2 = 45 edges
        assert_eq!(edges.len(), 45);
    }

    #[test]
    fn test_sbm_error_empty_blocks() {
        assert!(make_sbm(&[], 0.5, 0.05, 1).is_err());
    }

    #[test]
    fn test_sbm_error_zero_block_size() {
        assert!(make_sbm(&[5, 0, 5], 0.5, 0.05, 1).is_err());
    }

    // ── make_barabasi_albert ─────────────────────────────────────────────────

    #[test]
    fn test_ba_edge_count() {
        // n=100, m=2: seed edge (0,1) = 1; then 98 new nodes × 2 = 196 → total 197
        let edges = make_barabasi_albert(100, 2, 42).expect("ba n=100 m=2");
        assert_eq!(edges.len(), 197, "BA n=100 m=2 should have 197 edges");
    }

    #[test]
    fn test_ba_canonical() {
        let edges = make_barabasi_albert(50, 2, 7).expect("ba canonical");
        for &(u, v) in &edges {
            assert!(u < v, "edges must be canonical; got ({u}, {v})");
            assert!(u < 50 && v < 50);
        }
    }

    #[test]
    fn test_ba_error_m_zero() {
        assert!(make_barabasi_albert(10, 0, 1).is_err());
    }

    #[test]
    fn test_ba_error_n_le_m() {
        assert!(make_barabasi_albert(3, 3, 1).is_err());
        assert!(make_barabasi_albert(2, 5, 1).is_err());
    }

    // ── make_watts_strogatz ──────────────────────────────────────────────────

    #[test]
    fn test_ws_edge_count_no_rewiring() {
        // p=0.0 → pure ring lattice, n*k/2 edges
        let edges = make_watts_strogatz(20, 4, 0.0, 1).expect("ws no rewiring");
        assert_eq!(edges.len(), 40, "ring lattice 20 nodes k=4 → 40 edges");
    }

    #[test]
    fn test_ws_edge_count_preserved() {
        // Rewiring preserves edge count (each rewired edge is replaced 1-for-1)
        let edges = make_watts_strogatz(30, 4, 0.3, 42).expect("ws rewired");
        assert_eq!(edges.len(), 60, "ws must preserve edge count after rewiring");
    }

    #[test]
    fn test_ws_canonical() {
        let edges = make_watts_strogatz(20, 4, 0.1, 5).expect("ws canonical");
        for &(u, v) in &edges {
            assert!(u < v, "edges must be canonical; got ({u}, {v})");
            assert!(u < 20 && v < 20);
        }
    }

    #[test]
    fn test_ws_error_k_odd() {
        assert!(make_watts_strogatz(20, 3, 0.1, 1).is_err());
    }

    #[test]
    fn test_ws_error_n_le_k() {
        assert!(make_watts_strogatz(4, 4, 0.1, 1).is_err());
    }

    #[test]
    fn test_ws_error_p_out_of_range() {
        assert!(make_watts_strogatz(20, 4, -0.1, 1).is_err());
        assert!(make_watts_strogatz(20, 4, 1.1, 1).is_err());
    }
}
