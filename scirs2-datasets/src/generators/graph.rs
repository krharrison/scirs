//! Graph dataset generators
//!
//! This module provides synthetic graph generators for benchmarking and
//! testing graph algorithms. Includes:
//!
//! - **Karate club**: Zachary's karate club social network (34 nodes)
//! - **Random graph (Erdos-Renyi)**: each edge exists independently with probability p
//! - **Barabasi-Albert**: preferential attachment model (scale-free networks)
//! - **Watts-Strogatz**: small-world network model
//!
//! All generators return a `Dataset` whose `data` field is the adjacency matrix
//! (symmetric, n_nodes x n_nodes), and optionally a `target` with node labels.

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;

/// Helper to create an RNG from an optional seed
fn create_rng(randomseed: Option<u64>) -> StdRng {
    match randomseed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = thread_rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    }
}

/// Generate Zachary's karate club graph
///
/// This is a well-known social network of 34 members of a karate club at a
/// US university, documented by Wayne Zachary in 1977. The network has 78 edges.
/// After a dispute, the club split into two groups led by node 0 (instructor)
/// and node 33 (president).
///
/// # Returns
///
/// A `Dataset` where:
/// - `data` has shape (34, 34), the symmetric adjacency matrix
/// - `target` has 34 values: 0.0 for instructor faction, 1.0 for president faction
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::graph::make_karate_club;
///
/// let ds = make_karate_club().expect("should succeed");
/// assert_eq!(ds.n_samples(), 34);
/// assert_eq!(ds.n_features(), 34);
/// ```
pub fn make_karate_club() -> Result<Dataset> {
    let n = 34;
    let mut adj = Array2::zeros((n, n));

    // Zachary's karate club edge list (0-indexed)
    // Source: Zachary, W. W. (1977). "An Information Flow Model for Conflict and
    //         Fission in Small Groups". Journal of Anthropological Research 33(4): 452-473.
    let edges: &[(usize, usize)] = &[
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 10),
        (0, 11),
        (0, 12),
        (0, 13),
        (0, 17),
        (0, 19),
        (0, 21),
        (0, 31),
        (1, 2),
        (1, 3),
        (1, 7),
        (1, 13),
        (1, 17),
        (1, 19),
        (1, 21),
        (1, 30),
        (2, 3),
        (2, 7),
        (2, 8),
        (2, 9),
        (2, 13),
        (2, 27),
        (2, 28),
        (2, 32),
        (3, 7),
        (3, 12),
        (3, 13),
        (4, 6),
        (4, 10),
        (5, 6),
        (5, 10),
        (5, 16),
        (6, 16),
        (8, 30),
        (8, 32),
        (8, 33),
        (9, 33),
        (13, 33),
        (14, 32),
        (14, 33),
        (15, 32),
        (15, 33),
        (18, 32),
        (18, 33),
        (19, 33),
        (20, 32),
        (20, 33),
        (22, 32),
        (22, 33),
        (23, 25),
        (23, 27),
        (23, 29),
        (23, 32),
        (23, 33),
        (24, 25),
        (24, 27),
        (24, 31),
        (25, 31),
        (26, 29),
        (26, 33),
        (27, 33),
        (28, 31),
        (28, 33),
        (29, 32),
        (29, 33),
        (30, 32),
        (30, 33),
        (31, 32),
        (31, 33),
        (32, 33),
    ];

    for &(u, v) in edges {
        adj[[u, v]] = 1.0;
        adj[[v, u]] = 1.0;
    }

    // Ground-truth community labels (Zachary's observed split)
    // 0 = instructor's group, 1 = president's group
    let labels_arr: [f64; 34] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, // 0-9
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, // 10-19
        1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // 20-29
        1.0, 1.0, 1.0, 1.0, // 30-33
    ];
    let target = Array1::from_vec(labels_arr.to_vec());

    let feature_names: Vec<String> = (0..n).map(|i| format!("node_{i}")).collect();

    let dataset = Dataset::new(adj, Some(target))
        .with_featurenames(feature_names)
        .with_targetnames(vec![
            "instructor_group".to_string(),
            "president_group".to_string(),
        ])
        .with_description("Zachary's karate club social network (34 nodes, 78 edges)".to_string())
        .with_metadata("generator", "make_karate_club")
        .with_metadata("n_nodes", "34")
        .with_metadata("n_edges", "78")
        .with_metadata("reference", "Zachary (1977)");

    Ok(dataset)
}

/// Generate an Erdos-Renyi random graph G(n, p)
///
/// Each possible edge exists independently with probability `edge_prob`.
/// The adjacency matrix is symmetric with zeros on the diagonal.
///
/// # Arguments
///
/// * `n_nodes` - Number of nodes in the graph (must be > 0)
/// * `edge_prob` - Probability of each edge existing (must be in [0, 1])
/// * `randomseed` - Optional random seed for reproducibility
///
/// # Returns
///
/// A `Dataset` where:
/// - `data` has shape (n_nodes, n_nodes), the symmetric adjacency matrix
/// - `target` is None (no ground-truth labels for random graphs)
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::graph::make_random_graph;
///
/// let ds = make_random_graph(50, 0.3, Some(42)).expect("should succeed");
/// assert_eq!(ds.n_samples(), 50);
/// assert_eq!(ds.n_features(), 50);
/// ```
pub fn make_random_graph(
    n_nodes: usize,
    edge_prob: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    if n_nodes == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_nodes must be > 0".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&edge_prob) {
        return Err(DatasetsError::InvalidFormat(
            "edge_prob must be in [0, 1]".to_string(),
        ));
    }

    let mut rng = create_rng(randomseed);
    let uniform = scirs2_core::random::Uniform::new(0.0, 1.0).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create uniform dist: {e}"))
    })?;

    let mut adj = Array2::zeros((n_nodes, n_nodes));

    for i in 0..n_nodes {
        for j in (i + 1)..n_nodes {
            if uniform.sample(&mut rng) < edge_prob {
                adj[[i, j]] = 1.0;
                adj[[j, i]] = 1.0;
            }
        }
    }

    let n_edges: usize = {
        let mut count = 0usize;
        for i in 0..n_nodes {
            for j in (i + 1)..n_nodes {
                if adj[[i, j]] > 0.5 {
                    count += 1;
                }
            }
        }
        count
    };

    let feature_names: Vec<String> = (0..n_nodes).map(|i| format!("node_{i}")).collect();

    let dataset = Dataset::new(adj, None)
        .with_featurenames(feature_names)
        .with_description(format!(
            "Erdos-Renyi random graph G({n_nodes}, {edge_prob})"
        ))
        .with_metadata("generator", "make_random_graph")
        .with_metadata("n_nodes", &n_nodes.to_string())
        .with_metadata("n_edges", &n_edges.to_string())
        .with_metadata("edge_prob", &edge_prob.to_string());

    Ok(dataset)
}

/// Generate a Barabasi-Albert preferential attachment graph
///
/// Starting from a complete graph of `m` nodes, new nodes are added one at a time,
/// each connecting to `m` existing nodes with probability proportional to their degree.
/// This produces a scale-free network with a power-law degree distribution.
///
/// # Arguments
///
/// * `n_nodes` - Total number of nodes in the final graph (must be > m)
/// * `m` - Number of edges to attach from a new node to existing nodes (must be >= 1)
/// * `randomseed` - Optional random seed for reproducibility
///
/// # Returns
///
/// A `Dataset` where:
/// - `data` has shape (n_nodes, n_nodes), the symmetric adjacency matrix
/// - `target` contains node degrees as f64
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::graph::make_barabasi_albert;
///
/// let ds = make_barabasi_albert(100, 3, Some(42)).expect("should succeed");
/// assert_eq!(ds.n_samples(), 100);
/// ```
pub fn make_barabasi_albert(n_nodes: usize, m: usize, randomseed: Option<u64>) -> Result<Dataset> {
    if m == 0 {
        return Err(DatasetsError::InvalidFormat("m must be >= 1".to_string()));
    }
    if n_nodes <= m {
        return Err(DatasetsError::InvalidFormat(format!(
            "n_nodes ({n_nodes}) must be > m ({m})"
        )));
    }

    let mut rng = create_rng(randomseed);

    let mut adj = Array2::zeros((n_nodes, n_nodes));
    let mut degrees = vec![0usize; n_nodes];

    // Start with a complete graph on m+1 nodes (nodes 0..=m)
    let initial_nodes = m + 1;
    for i in 0..initial_nodes {
        for j in (i + 1)..initial_nodes {
            adj[[i, j]] = 1.0;
            adj[[j, i]] = 1.0;
            degrees[i] += 1;
            degrees[j] += 1;
        }
    }

    // Repeated-stubs list for preferential attachment
    // Each node appears in the list once per edge it has
    let mut stubs: Vec<usize> = Vec::new();
    for (node, &deg) in degrees.iter().enumerate().take(initial_nodes) {
        for _ in 0..deg {
            stubs.push(node);
        }
    }

    // Add remaining nodes one at a time
    for new_node in initial_nodes..n_nodes {
        // Select m targets by sampling from stubs without replacement (unique targets)
        let mut targets: Vec<usize> = Vec::with_capacity(m);
        let mut attempts = 0;
        let max_attempts = m * 100;

        while targets.len() < m && attempts < max_attempts {
            attempts += 1;
            if stubs.is_empty() {
                break;
            }
            let idx = rng.random_range(0..stubs.len());
            let candidate = stubs[idx];
            if candidate != new_node && !targets.contains(&candidate) {
                targets.push(candidate);
            }
        }

        // Connect new_node to each target
        for &target in &targets {
            adj[[new_node, target]] = 1.0;
            adj[[target, new_node]] = 1.0;
            degrees[new_node] += 1;
            degrees[target] += 1;
            stubs.push(new_node);
            stubs.push(target);
        }
    }

    let target = Array1::from_vec(degrees.iter().map(|&d| d as f64).collect());

    let n_edges: usize = degrees.iter().sum::<usize>() / 2;
    let feature_names: Vec<String> = (0..n_nodes).map(|i| format!("node_{i}")).collect();

    let dataset = Dataset::new(adj, Some(target))
        .with_featurenames(feature_names)
        .with_description(format!(
            "Barabasi-Albert preferential attachment graph ({n_nodes} nodes, m={m})"
        ))
        .with_metadata("generator", "make_barabasi_albert")
        .with_metadata("n_nodes", &n_nodes.to_string())
        .with_metadata("n_edges", &n_edges.to_string())
        .with_metadata("m", &m.to_string());

    Ok(dataset)
}

/// Generate a Watts-Strogatz small-world network
///
/// Starts with a ring lattice where each node is connected to its `k` nearest
/// neighbours, then randomly rewires each edge with probability `p`.
///
/// # Arguments
///
/// * `n_nodes` - Number of nodes (must be > k)
/// * `k` - Each node is initially connected to k nearest neighbours (k/2 on each side).
///   Must be even and >= 2.
/// * `p` - Rewiring probability in [0, 1]. p=0 gives a regular lattice, p=1 gives random.
/// * `randomseed` - Optional random seed for reproducibility
///
/// # Returns
///
/// A `Dataset` where:
/// - `data` has shape (n_nodes, n_nodes), the symmetric adjacency matrix
/// - `target` contains node degrees as f64
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::graph::make_watts_strogatz;
///
/// let ds = make_watts_strogatz(100, 4, 0.3, Some(42)).expect("should succeed");
/// assert_eq!(ds.n_samples(), 100);
/// ```
pub fn make_watts_strogatz(
    n_nodes: usize,
    k: usize,
    p: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    if n_nodes == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_nodes must be > 0".to_string(),
        ));
    }
    if k == 0 || !k.is_multiple_of(2) {
        return Err(DatasetsError::InvalidFormat(
            "k must be even and >= 2".to_string(),
        ));
    }
    if k >= n_nodes {
        return Err(DatasetsError::InvalidFormat(format!(
            "k ({k}) must be < n_nodes ({n_nodes})"
        )));
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(DatasetsError::InvalidFormat(
            "p must be in [0, 1]".to_string(),
        ));
    }

    let mut rng = create_rng(randomseed);
    let uniform = scirs2_core::random::Uniform::new(0.0, 1.0).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create uniform dist: {e}"))
    })?;

    let mut adj = Array2::zeros((n_nodes, n_nodes));

    // Step 1: Create ring lattice (each node connected to k/2 neighbours on each side)
    let half_k = k / 2;
    for i in 0..n_nodes {
        for offset in 1..=half_k {
            let j = (i + offset) % n_nodes;
            adj[[i, j]] = 1.0;
            adj[[j, i]] = 1.0;
        }
    }

    // Step 2: Rewire edges with probability p
    // For each node i, consider each edge (i, j) where j = (i+offset) mod n for offset in 1..=half_k
    // With probability p, replace (i, j) with (i, k') where k' is chosen uniformly at random
    // from nodes that are not i and not already a neighbour of i.
    for i in 0..n_nodes {
        for offset in 1..=half_k {
            if uniform.sample(&mut rng) < p {
                let j = (i + offset) % n_nodes;

                // Remove the old edge
                adj[[i, j]] = 0.0;
                adj[[j, i]] = 0.0;

                // Find a new target that is not i and not already connected to i
                let mut new_target = j;
                let mut attempts = 0;
                let max_attempts = n_nodes * 10;
                while attempts < max_attempts {
                    attempts += 1;
                    new_target = rng.random_range(0..n_nodes);
                    if new_target != i && adj[[i, new_target]] < 0.5 {
                        break;
                    }
                }

                // Add the new edge (if we found a valid target)
                if new_target != i && adj[[i, new_target]] < 0.5 {
                    adj[[i, new_target]] = 1.0;
                    adj[[new_target, i]] = 1.0;
                } else {
                    // Restore original edge if no valid rewiring found
                    adj[[i, j]] = 1.0;
                    adj[[j, i]] = 1.0;
                }
            }
        }
    }

    // Compute degrees
    let mut degrees = vec![0usize; n_nodes];
    for i in 0..n_nodes {
        for j in 0..n_nodes {
            if adj[[i, j]] > 0.5 {
                degrees[i] += 1;
            }
        }
    }

    let target = Array1::from_vec(degrees.iter().map(|&d| d as f64).collect());

    let n_edges: usize = degrees.iter().sum::<usize>() / 2;
    let feature_names: Vec<String> = (0..n_nodes).map(|i| format!("node_{i}")).collect();

    let dataset = Dataset::new(adj, Some(target))
        .with_featurenames(feature_names)
        .with_description(format!(
            "Watts-Strogatz small-world graph ({n_nodes} nodes, k={k}, p={p})"
        ))
        .with_metadata("generator", "make_watts_strogatz")
        .with_metadata("n_nodes", &n_nodes.to_string())
        .with_metadata("n_edges", &n_edges.to_string())
        .with_metadata("k", &k.to_string())
        .with_metadata("p", &p.to_string());

    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // make_karate_club tests
    // =========================================================================

    #[test]
    fn test_karate_club_shape() {
        let ds = make_karate_club().expect("should succeed");
        assert_eq!(ds.n_samples(), 34);
        assert_eq!(ds.n_features(), 34);
        assert!(ds.target.is_some());
        let target = ds.target.as_ref().expect("target exists");
        assert_eq!(target.len(), 34);
    }

    #[test]
    fn test_karate_club_properties() {
        let ds = make_karate_club().expect("should succeed");

        // Adjacency matrix should be symmetric
        for i in 0..34 {
            for j in 0..34 {
                assert!(
                    (ds.data[[i, j]] - ds.data[[j, i]]).abs() < 1e-15,
                    "Adjacency matrix not symmetric at ({i},{j})"
                );
            }
        }

        // Diagonal should be zero (no self-loops)
        for i in 0..34 {
            assert!(ds.data[[i, i]].abs() < 1e-15, "Self-loop at node {i}");
        }

        // Count edges (upper triangle)
        let mut edge_count = 0usize;
        for i in 0..34 {
            for j in (i + 1)..34 {
                if ds.data[[i, j]] > 0.5 {
                    edge_count += 1;
                }
            }
        }
        assert_eq!(edge_count, 78, "Karate club should have 78 edges");
    }

    #[test]
    fn test_karate_club_reproducibility() {
        // Karate club is deterministic (no randomness)
        let ds1 = make_karate_club().expect("should succeed");
        let ds2 = make_karate_club().expect("should succeed");
        for i in 0..34 {
            for j in 0..34 {
                assert!(
                    (ds1.data[[i, j]] - ds2.data[[i, j]]).abs() < 1e-15,
                    "Adjacency differs at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_karate_club_node_degrees() {
        let ds = make_karate_club().expect("should succeed");

        // Node 0 (instructor) should have degree 16
        let degree_0: f64 = (0..34).map(|j| ds.data[[0, j]]).sum();
        assert!(
            (degree_0 - 16.0).abs() < 1e-10,
            "Node 0 degree should be 16, got {degree_0}"
        );

        // Node 33 (president) should have degree 17
        let degree_33: f64 = (0..34).map(|j| ds.data[[33, j]]).sum();
        assert!(
            (degree_33 - 17.0).abs() < 1e-10,
            "Node 33 degree should be 17, got {degree_33}"
        );
    }

    // =========================================================================
    // make_random_graph tests
    // =========================================================================

    #[test]
    fn test_random_graph_shape() {
        let ds = make_random_graph(50, 0.3, Some(42)).expect("should succeed");
        assert_eq!(ds.n_samples(), 50);
        assert_eq!(ds.n_features(), 50);
    }

    #[test]
    fn test_random_graph_symmetric() {
        let ds = make_random_graph(30, 0.5, Some(42)).expect("should succeed");
        for i in 0..30 {
            for j in 0..30 {
                assert!(
                    (ds.data[[i, j]] - ds.data[[j, i]]).abs() < 1e-15,
                    "Not symmetric at ({i},{j})"
                );
            }
            assert!(ds.data[[i, i]].abs() < 1e-15, "Self-loop at node {i}");
        }
    }

    #[test]
    fn test_random_graph_edge_density() {
        let n = 100;
        let p = 0.3;
        let ds = make_random_graph(n, p, Some(42)).expect("should succeed");

        let mut edge_count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if ds.data[[i, j]] > 0.5 {
                    edge_count += 1;
                }
            }
        }

        let max_edges = n * (n - 1) / 2;
        let actual_density = edge_count as f64 / max_edges as f64;

        // The density should be roughly p, with some variance for n=100
        assert!(
            (actual_density - p).abs() < 0.1,
            "Edge density {actual_density} should be close to p={p}"
        );
    }

    #[test]
    fn test_random_graph_reproducibility() {
        let ds1 = make_random_graph(40, 0.4, Some(123)).expect("should succeed");
        let ds2 = make_random_graph(40, 0.4, Some(123)).expect("should succeed");
        for i in 0..40 {
            for j in 0..40 {
                assert!(
                    (ds1.data[[i, j]] - ds2.data[[i, j]]).abs() < 1e-15,
                    "Adjacency differs at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_random_graph_extreme_probs() {
        // p=0: no edges
        let ds = make_random_graph(20, 0.0, Some(42)).expect("should succeed");
        let total: f64 = ds.data.iter().sum();
        assert!(total.abs() < 1e-10, "p=0 should produce no edges");

        // p=1: complete graph
        let ds = make_random_graph(10, 1.0, Some(42)).expect("should succeed");
        let mut edge_count = 0usize;
        for i in 0..10 {
            for j in (i + 1)..10 {
                if ds.data[[i, j]] > 0.5 {
                    edge_count += 1;
                }
            }
        }
        assert_eq!(
            edge_count, 45,
            "p=1 should produce complete graph (45 edges for n=10)"
        );
    }

    #[test]
    fn test_random_graph_validation() {
        assert!(make_random_graph(0, 0.5, None).is_err());
        assert!(make_random_graph(10, -0.1, None).is_err());
        assert!(make_random_graph(10, 1.1, None).is_err());
    }

    // =========================================================================
    // make_barabasi_albert tests
    // =========================================================================

    #[test]
    fn test_barabasi_albert_shape() {
        let ds = make_barabasi_albert(100, 3, Some(42)).expect("should succeed");
        assert_eq!(ds.n_samples(), 100);
        assert_eq!(ds.n_features(), 100);
        assert!(ds.target.is_some());
    }

    #[test]
    fn test_barabasi_albert_symmetric() {
        let ds = make_barabasi_albert(50, 2, Some(42)).expect("should succeed");
        for i in 0..50 {
            for j in 0..50 {
                assert!(
                    (ds.data[[i, j]] - ds.data[[j, i]]).abs() < 1e-15,
                    "Not symmetric at ({i},{j})"
                );
            }
            assert!(ds.data[[i, i]].abs() < 1e-15, "Self-loop at node {i}");
        }
    }

    #[test]
    fn test_barabasi_albert_degree_distribution() {
        // BA graphs have power-law degree distribution; early nodes should have higher degree
        let n = 200;
        let m = 2;
        let ds = make_barabasi_albert(n, m, Some(42)).expect("should succeed");
        let target = ds.target.as_ref().expect("target exists");

        // Average degree of the initial nodes (0..=m) should be higher than later nodes
        let initial_avg: f64 = (0..=m).map(|i| target[i]).sum::<f64>() / (m + 1) as f64;
        let late_avg: f64 = (n - 20..n).map(|i| target[i]).sum::<f64>() / 20.0;

        assert!(
            initial_avg > late_avg,
            "Initial nodes should have higher avg degree: {initial_avg} vs {late_avg}"
        );
    }

    #[test]
    fn test_barabasi_albert_edge_count() {
        let n = 50;
        let m = 3;
        let ds = make_barabasi_albert(n, m, Some(42)).expect("should succeed");

        let mut edge_count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if ds.data[[i, j]] > 0.5 {
                    edge_count += 1;
                }
            }
        }

        // Expected edges: C(m+1, 2) for initial complete graph + m*(n-m-1) for subsequent nodes
        let initial_edges = (m + 1) * m / 2;
        let subsequent_edges = m * (n - m - 1);
        let expected_edges = initial_edges + subsequent_edges;

        assert_eq!(
            edge_count, expected_edges,
            "BA graph should have {expected_edges} edges, got {edge_count}"
        );
    }

    #[test]
    fn test_barabasi_albert_reproducibility() {
        let ds1 = make_barabasi_albert(60, 2, Some(77)).expect("should succeed");
        let ds2 = make_barabasi_albert(60, 2, Some(77)).expect("should succeed");
        for i in 0..60 {
            for j in 0..60 {
                assert!(
                    (ds1.data[[i, j]] - ds2.data[[i, j]]).abs() < 1e-15,
                    "Adjacency differs at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_barabasi_albert_validation() {
        assert!(make_barabasi_albert(5, 0, None).is_err()); // m=0
        assert!(make_barabasi_albert(3, 3, None).is_err()); // n_nodes <= m
        assert!(make_barabasi_albert(3, 5, None).is_err()); // n_nodes <= m
    }

    // =========================================================================
    // make_watts_strogatz tests
    // =========================================================================

    #[test]
    fn test_watts_strogatz_shape() {
        let ds = make_watts_strogatz(100, 4, 0.3, Some(42)).expect("should succeed");
        assert_eq!(ds.n_samples(), 100);
        assert_eq!(ds.n_features(), 100);
        assert!(ds.target.is_some());
    }

    #[test]
    fn test_watts_strogatz_symmetric() {
        let ds = make_watts_strogatz(50, 4, 0.3, Some(42)).expect("should succeed");
        for i in 0..50 {
            for j in 0..50 {
                assert!(
                    (ds.data[[i, j]] - ds.data[[j, i]]).abs() < 1e-15,
                    "Not symmetric at ({i},{j})"
                );
            }
            assert!(ds.data[[i, i]].abs() < 1e-15, "Self-loop at node {i}");
        }
    }

    #[test]
    fn test_watts_strogatz_regular_lattice() {
        // With p=0, should be a regular ring lattice
        let n = 20;
        let k = 4;
        let ds = make_watts_strogatz(n, k, 0.0, Some(42)).expect("should succeed");
        let target = ds.target.as_ref().expect("target exists");

        // All degrees should be exactly k
        for i in 0..n {
            assert!(
                (target[i] - k as f64).abs() < 1e-10,
                "Node {i} degree should be {k}, got {}",
                target[i]
            );
        }
    }

    #[test]
    fn test_watts_strogatz_edge_count() {
        let n = 30;
        let k = 6;
        let ds = make_watts_strogatz(n, k, 0.3, Some(42)).expect("should succeed");

        let mut edge_count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if ds.data[[i, j]] > 0.5 {
                    edge_count += 1;
                }
            }
        }

        // Watts-Strogatz preserves the number of edges (n*k/2)
        let expected_edges = n * k / 2;
        assert_eq!(
            edge_count, expected_edges,
            "WS graph should have {expected_edges} edges, got {edge_count}"
        );
    }

    #[test]
    fn test_watts_strogatz_reproducibility() {
        let ds1 = make_watts_strogatz(40, 4, 0.5, Some(88)).expect("should succeed");
        let ds2 = make_watts_strogatz(40, 4, 0.5, Some(88)).expect("should succeed");
        for i in 0..40 {
            for j in 0..40 {
                assert!(
                    (ds1.data[[i, j]] - ds2.data[[i, j]]).abs() < 1e-15,
                    "Adjacency differs at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_watts_strogatz_validation() {
        assert!(make_watts_strogatz(0, 4, 0.3, None).is_err()); // n_nodes = 0
        assert!(make_watts_strogatz(10, 0, 0.3, None).is_err()); // k = 0
        assert!(make_watts_strogatz(10, 3, 0.3, None).is_err()); // k odd
        assert!(make_watts_strogatz(5, 6, 0.3, None).is_err()); // k >= n_nodes
        assert!(make_watts_strogatz(10, 4, -0.1, None).is_err()); // p < 0
        assert!(make_watts_strogatz(10, 4, 1.1, None).is_err()); // p > 1
    }
}
