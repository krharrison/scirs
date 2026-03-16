//! Functional random walk API for graph embeddings
//!
//! Provides simple function-based wrappers around the random walk machinery
//! in `random_walk.rs`.  These functions operate on `Graph<usize, f64>` which
//! is the natural representation when nodes are identified by their numeric
//! index (as assumed by skip-gram and spectral embedding).
//!
//! # Node2Vec / biased walks
//! - `biased_random_walk` — single walk from a given start node
//! - `generate_walks`     — all walks for all nodes
//!
//! # DeepWalk / uniform walks
//! - `deepwalk_walks`     — uniform random walks for all nodes

use crate::base::Graph;
use crate::error::Result;
use scirs2_core::random::{Rng, RngExt};

// ─────────────────────────────────────────────────────────────────────────────
// biased_random_walk
// ─────────────────────────────────────────────────────────────────────────────

/// Perform a single node2vec-style biased random walk on a `Graph<usize, f64>`.
///
/// The walk uses second-order Markov transitions controlled by parameters `p`
/// (return parameter) and `q` (in-out parameter):
/// - `p > 1` — less likely to return to the previous node (exploration)
/// - `p < 1` — more likely to return to the previous node
/// - `q > 1` — BFS-like, stays close to the source
/// - `q < 1` — DFS-like, explores further
///
/// # Arguments
/// * `graph`  – the undirected weighted graph
/// * `start`  – index of the starting node (must be `< graph.node_count()`)
/// * `length` – desired walk length (number of nodes, including `start`)
/// * `p`      – return parameter (> 0)
/// * `q`      – in-out parameter (> 0)
///
/// # Returns
/// A `Vec<usize>` of node indices visited.  May be shorter than `length` if the
/// walk reaches a dead-end (isolated node).
pub fn biased_random_walk(
    graph: &Graph<usize, f64>,
    start: usize,
    length: usize,
    p: f64,
    q: f64,
) -> Result<Vec<usize>> {
    let mut rng = scirs2_core::random::rng();
    biased_random_walk_with_rng(graph, start, length, p, q, &mut rng)
}

/// Same as [`biased_random_walk`] but accepts a caller-supplied RNG for
/// reproducibility.
pub fn biased_random_walk_with_rng<R: Rng>(
    graph: &Graph<usize, f64>,
    start: usize,
    length: usize,
    p: f64,
    q: f64,
    rng: &mut R,
) -> Result<Vec<usize>> {
    use crate::error::GraphError;

    if !graph.has_node(&start) {
        return Err(GraphError::node_not_found(format!("{start}")));
    }
    if p <= 0.0 || q <= 0.0 {
        return Err(GraphError::InvalidParameter {
            param: "p/q".to_string(),
            value: format!("p={p}, q={q}"),
            expected: "p > 0 and q > 0".to_string(),
            context: "biased_random_walk".to_string(),
        });
    }

    let mut walk = vec![start];
    if length <= 1 {
        return Ok(walk);
    }

    // First unbiased step
    let first_neighbors = graph.neighbors(&start)?;
    if first_neighbors.is_empty() {
        return Ok(walk);
    }
    let first_idx = rng.random_range(0..first_neighbors.len());
    let mut current = first_neighbors[first_idx];
    walk.push(current);

    // Subsequent biased steps
    for _ in 2..length {
        let neighbors = graph.neighbors(&current)?;
        if neighbors.is_empty() {
            break;
        }

        let prev = walk[walk.len() - 2];
        let weights: Vec<f64> = neighbors
            .iter()
            .map(|&nbr| {
                if nbr == prev {
                    1.0 / p
                } else if graph.has_edge(&prev, &nbr) {
                    1.0
                } else {
                    1.0 / q
                }
            })
            .collect();

        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            break;
        }

        let mut r = rng.random::<f64>() * total;
        let mut chosen = neighbors.len() - 1;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                chosen = i;
                break;
            }
        }

        current = neighbors[chosen];
        walk.push(current);
    }

    Ok(walk)
}

// ─────────────────────────────────────────────────────────────────────────────
// generate_walks (node2vec / biased)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate multiple biased random walks from every node in a `Graph<usize, f64>`.
///
/// This is the corpus-generation step for node2vec.
///
/// # Arguments
/// * `graph`       – the undirected weighted graph
/// * `num_walks`   – number of walks to start from each node
/// * `walk_length` – desired length of each walk
/// * `p`           – node2vec return parameter
/// * `q`           – node2vec in-out parameter
///
/// # Returns
/// A `Vec<Vec<usize>>` containing `num_walks * node_count` walks in total.
pub fn generate_walks(
    graph: &Graph<usize, f64>,
    num_walks: usize,
    walk_length: usize,
    p: f64,
    q: f64,
) -> Result<Vec<Vec<usize>>> {
    let mut rng = scirs2_core::random::rng();
    let mut all_walks = Vec::new();

    for &node in graph.nodes() {
        for _ in 0..num_walks {
            let walk = biased_random_walk_with_rng(graph, node, walk_length, p, q, &mut rng)?;
            all_walks.push(walk);
        }
    }

    Ok(all_walks)
}

// ─────────────────────────────────────────────────────────────────────────────
// deepwalk_walks (uniform / unbiased)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate multiple uniform random walks from every node (DeepWalk style).
///
/// This is equivalent to [`generate_walks`] with `p = 1.0` and `q = 1.0`.
///
/// # Arguments
/// * `graph`       – the undirected weighted graph
/// * `num_walks`   – number of walks per node
/// * `walk_length` – desired length of each walk
pub fn deepwalk_walks(
    graph: &Graph<usize, f64>,
    num_walks: usize,
    walk_length: usize,
) -> Result<Vec<Vec<usize>>> {
    generate_walks(graph, num_walks, walk_length, 1.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_square_graph() -> Graph<usize, f64> {
        let mut g: Graph<usize, f64> = Graph::new();
        for i in 0..4 {
            g.add_node(i);
        }
        let _ = g.add_edge(0, 1, 1.0);
        let _ = g.add_edge(1, 2, 1.0);
        let _ = g.add_edge(2, 3, 1.0);
        let _ = g.add_edge(3, 0, 1.0);
        g
    }

    #[test]
    fn test_biased_random_walk_length() {
        let g = make_square_graph();
        let walk = biased_random_walk(&g, 0, 10, 1.0, 1.0).expect("walk should succeed");
        // Walk is at most length 10 (may be shorter at dead-ends, but square has none)
        assert!(walk.len() <= 10);
        assert!(!walk.is_empty());
        assert_eq!(walk[0], 0);
    }

    #[test]
    fn test_biased_random_walk_nodes_valid() {
        let g = make_square_graph();
        let walk = biased_random_walk(&g, 0, 20, 1.0, 1.0).expect("walk should succeed");
        for &node in &walk {
            assert!(node < 4, "walk should only visit nodes 0..3, got {node}");
        }
    }

    #[test]
    fn test_biased_random_walk_invalid_node() {
        let g = make_square_graph();
        let result = biased_random_walk(&g, 99, 10, 1.0, 1.0);
        assert!(result.is_err(), "should fail for non-existent node");
    }

    #[test]
    fn test_biased_random_walk_invalid_params() {
        let g = make_square_graph();
        let result = biased_random_walk(&g, 0, 10, -1.0, 1.0);
        assert!(result.is_err(), "p <= 0 should return error");
    }

    #[test]
    fn test_generate_walks_count() {
        let g = make_square_graph();
        let walks = generate_walks(&g, 3, 5, 1.0, 1.0).expect("should succeed");
        // 4 nodes × 3 walks = 12 walks
        assert_eq!(walks.len(), 12);
    }

    #[test]
    fn test_generate_walks_all_start_valid() {
        let g = make_square_graph();
        let walks = generate_walks(&g, 2, 8, 2.0, 0.5).expect("should succeed");
        for walk in &walks {
            for &node in walk {
                assert!(node < 4, "walk node {node} out of range");
            }
        }
    }

    #[test]
    fn test_deepwalk_walks_equivalence() {
        // DeepWalk walks (p=1, q=1) should behave the same as generate_walks(p=1,q=1)
        let g = make_square_graph();
        let dw = deepwalk_walks(&g, 3, 6).expect("should succeed");
        assert_eq!(dw.len(), 12, "4 nodes × 3 walks = 12");
    }

    #[test]
    fn test_biased_walk_length_one() {
        let g = make_square_graph();
        let walk = biased_random_walk(&g, 0, 1, 1.0, 1.0).expect("walk should succeed");
        assert_eq!(walk, vec![0]);
    }

    #[test]
    fn test_biased_walk_isolated_node() {
        let mut g: Graph<usize, f64> = Graph::new();
        g.add_node(0);
        g.add_node(1);
        let _ = g.add_edge(0, 1, 1.0);
        // Node 2 is isolated
        g.add_node(2);
        // Walk from isolated node should return just [2]
        let walk = biased_random_walk(&g, 2, 10, 1.0, 1.0).expect("walk should succeed");
        assert_eq!(walk.len(), 1);
        assert_eq!(walk[0], 2);
    }
}
