//! GRASP (Greedy Randomized Adaptive Search Procedure) for network alignment.
//!
//! Implements a meta-heuristic that combines randomized greedy construction
//! with local search refinement to find high-quality network alignments.
//!
//! # Algorithm Overview
//!
//! GRASP operates in two phases per restart:
//! 1. **Construction**: Build an initial alignment greedily with randomization,
//!    selecting from a restricted candidate list (RCL) of top-k scoring pairs.
//! 2. **Local Search**: Improve the alignment by exploring swap neighborhoods,
//!    accepting moves that increase edge conservation.
//!
//! Multiple restarts are performed and the best alignment is kept.

use scirs2_core::ndarray::Array2;

use crate::error::{GraphError, Result};

use super::evaluation::edge_conservation;
use super::types::{AlignmentConfig, AlignmentResult};

/// Run the GRASP meta-heuristic for network alignment.
///
/// Performs multiple randomized construction + local search iterations,
/// returning the best alignment found across all restarts.
///
/// # Arguments
///
/// * `adj1` - Adjacency matrix of graph 1 `[n1 x n1]`
/// * `adj2` - Adjacency matrix of graph 2 `[n2 x n2]`
/// * `prior` - Optional prior similarity matrix `[n1 x n2]`
/// * `config` - Algorithm configuration
/// * `n_restarts` - Number of GRASP restarts (more = better quality but slower)
///
/// # Errors
///
/// Returns an error if adjacency matrices are not square, dimensions are zero,
/// or `n_restarts` is zero.
pub fn grasp_alignment(
    adj1: &Array2<f64>,
    adj2: &Array2<f64>,
    prior: Option<&Array2<f64>>,
    config: &AlignmentConfig,
    n_restarts: usize,
) -> Result<AlignmentResult> {
    let n1 = adj1.nrows();
    let n2 = adj2.nrows();

    // Validate inputs
    if adj1.nrows() != adj1.ncols() {
        return Err(GraphError::InvalidParameter {
            param: "adj1".to_string(),
            value: format!("{}x{}", adj1.nrows(), adj1.ncols()),
            expected: "square matrix".to_string(),
            context: "grasp_alignment".to_string(),
        });
    }
    if adj2.nrows() != adj2.ncols() {
        return Err(GraphError::InvalidParameter {
            param: "adj2".to_string(),
            value: format!("{}x{}", adj2.nrows(), adj2.ncols()),
            expected: "square matrix".to_string(),
            context: "grasp_alignment".to_string(),
        });
    }
    if n_restarts == 0 {
        return Err(GraphError::InvalidParameter {
            param: "n_restarts".to_string(),
            value: "0".to_string(),
            expected: "at least 1".to_string(),
            context: "grasp_alignment".to_string(),
        });
    }

    // Handle empty graphs
    if n1 == 0 || n2 == 0 {
        return Ok(AlignmentResult {
            mapping: Vec::new(),
            score: 0.0,
            edge_conservation: 0.0,
            converged: true,
            iterations: 0,
        });
    }

    // Handle single-node graphs
    if n1 == 1 && n2 == 1 {
        return Ok(AlignmentResult {
            mapping: vec![(0, 0)],
            score: 1.0,
            edge_conservation: 1.0,
            converged: true,
            iterations: 1,
        });
    }

    // Validate prior dimensions
    if let Some(p) = prior {
        if p.nrows() != n1 || p.ncols() != n2 {
            return Err(GraphError::InvalidParameter {
                param: "prior".to_string(),
                value: format!("{}x{}", p.nrows(), p.ncols()),
                expected: format!("{}x{}", n1, n2),
                context: "grasp_alignment: prior dimensions must match graph sizes".to_string(),
            });
        }
    }

    let mut best_mapping: Vec<(usize, usize)> = Vec::new();
    let mut best_score = f64::NEG_INFINITY;
    let mut total_iterations = 0;

    // Use a simple deterministic seed based on graph properties
    let mut rng_state: u64 = (n1 as u64)
        .wrapping_mul(2654435761)
        .wrapping_add(n2 as u64)
        .wrapping_mul(40503)
        .wrapping_add(17);

    for _restart in 0..n_restarts {
        // Construction phase
        let mut mapping = construct_alignment(adj1, adj2, prior, config, &mut rng_state);

        // Local search phase
        let score = local_search(
            &mut mapping,
            adj1,
            adj2,
            config.local_search_depth,
            &mut rng_state,
        );

        total_iterations += 1;

        if score > best_score {
            best_score = score;
            best_mapping = mapping;
        }
    }

    let ec = edge_conservation(&best_mapping, adj1, adj2);

    Ok(AlignmentResult {
        mapping: best_mapping,
        score: best_score,
        edge_conservation: ec,
        converged: true,
        iterations: total_iterations,
    })
}

/// Xorshift64 pseudo-random number generator.
///
/// Simple, fast PRNG suitable for randomized algorithm construction.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a random index in `[0, bound)` using xorshift64.
fn rand_index(state: &mut u64, bound: usize) -> usize {
    if bound == 0 {
        return 0;
    }
    (xorshift64(state) % bound as u64) as usize
}

/// Construction phase: build an alignment greedily with randomization.
///
/// At each step, computes the "edge conservation gain" for each unmatched pair,
/// then randomly selects from the top-k candidates (restricted candidate list).
fn construct_alignment(
    adj1: &Array2<f64>,
    adj2: &Array2<f64>,
    prior: Option<&Array2<f64>>,
    config: &AlignmentConfig,
    rng_state: &mut u64,
) -> Vec<(usize, usize)> {
    let n1 = adj1.nrows();
    let n2 = adj2.nrows();
    let n_pairs = n1.min(n2);

    let mut used_rows = vec![false; n1];
    let mut used_cols = vec![false; n2];
    let mut mapping: Vec<(usize, usize)> = Vec::with_capacity(n_pairs);

    for _ in 0..n_pairs {
        // Compute gain for each unmatched pair
        let mut candidates: Vec<(f64, usize, usize)> = Vec::new();

        for i in 0..n1 {
            if used_rows[i] {
                continue;
            }
            for j in 0..n2 {
                if used_cols[j] {
                    continue;
                }
                let gain = compute_pair_gain(i, j, &mapping, adj1, adj2, prior);
                candidates.push((gain, i, j));
            }
        }

        if candidates.is_empty() {
            break;
        }

        // Sort descending by gain
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Select from restricted candidate list (top-k)
        let rcl_size = config.greedy_candidates.min(candidates.len());
        let chosen_idx = rand_index(rng_state, rcl_size);
        let (_, ci, cj) = candidates[chosen_idx];

        mapping.push((ci, cj));
        used_rows[ci] = true;
        used_cols[cj] = true;
    }

    mapping.sort_by_key(|&(i, _)| i);
    mapping
}

/// Compute the gain of adding pair `(i, j)` to the current mapping.
///
/// The gain considers:
/// - Edge conservation: how many existing mapped neighbors of `i` in G1
///   have their counterparts as neighbors of `j` in G2
/// - Prior similarity (if available)
fn compute_pair_gain(
    i: usize,
    j: usize,
    current_mapping: &[(usize, usize)],
    adj1: &Array2<f64>,
    adj2: &Array2<f64>,
    prior: Option<&Array2<f64>>,
) -> f64 {
    let mut gain = 0.0;

    // Edge conservation gain: count edges preserved by this assignment
    for &(mi, mj) in current_mapping {
        let edge_in_g1 = adj1[[i, mi]].abs() > f64::EPSILON;
        let edge_in_g2 = adj2[[j, mj]].abs() > f64::EPSILON;

        if edge_in_g1 && edge_in_g2 {
            gain += 1.0;
        }

        // Also check reverse direction for undirected graphs
        let edge_in_g1_rev = adj1[[mi, i]].abs() > f64::EPSILON;
        let edge_in_g2_rev = adj2[[mj, j]].abs() > f64::EPSILON;

        if edge_in_g1_rev && edge_in_g2_rev {
            gain += 1.0;
        }
    }

    // Add prior similarity bonus
    if let Some(p) = prior {
        gain += p[[i, j]];
    }

    gain
}

/// Local search: improve alignment via swap moves.
///
/// For each pair of mappings `(i->j, k->l)`, tries swapping to `(i->l, k->j)`.
/// Accepts the swap if edge conservation improves. Continues until no improving
/// swap is found or the iteration limit is reached.
///
/// Returns the final edge conservation score.
fn local_search(
    mapping: &mut [(usize, usize)],
    adj1: &Array2<f64>,
    adj2: &Array2<f64>,
    max_iterations: usize,
    rng_state: &mut u64,
) -> f64 {
    if mapping.len() <= 1 {
        return edge_conservation(mapping, adj1, adj2);
    }

    let mut current_score = edge_conservation(mapping, adj1, adj2);

    for _ in 0..max_iterations {
        let mut improved = false;

        // Exhaustive swap-based local search
        let n = mapping.len();
        for p in 0..n {
            for q in (p + 1)..n {
                let (pi, pj) = mapping[p];
                let (qi, qj) = mapping[q];

                // Try swap: (pi->qj, qi->pj)
                mapping[p] = (pi, qj);
                mapping[q] = (qi, pj);

                let new_score = edge_conservation(mapping, adj1, adj2);

                if new_score > current_score + 1e-12 {
                    current_score = new_score;
                    improved = true;
                } else {
                    // Revert
                    mapping[p] = (pi, pj);
                    mapping[q] = (qi, qj);
                }
            }
        }

        if !improved {
            break;
        }
    }

    current_score
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn path_graph(n: usize) -> Array2<f64> {
        let mut adj = Array2::zeros((n, n));
        for i in 0..n.saturating_sub(1) {
            adj[[i, i + 1]] = 1.0;
            adj[[i + 1, i]] = 1.0;
        }
        adj
    }

    fn complete_graph(n: usize) -> Array2<f64> {
        let mut adj = Array2::ones((n, n));
        for i in 0..n {
            adj[[i, i]] = 0.0;
        }
        adj
    }

    #[test]
    fn test_grasp_improves_over_random() {
        let adj = path_graph(6);
        let config = AlignmentConfig {
            greedy_candidates: 3,
            local_search_depth: 20,
            ..AlignmentConfig::default()
        };

        // With 1 restart
        let result1 = grasp_alignment(&adj, &adj, None, &config, 1).expect("grasp should succeed");

        // With 10 restarts should be at least as good
        let result10 =
            grasp_alignment(&adj, &adj, None, &config, 10).expect("grasp should succeed");

        assert!(
            result10.score >= result1.score - 1e-10,
            "10 restarts ({}) should be >= 1 restart ({})",
            result10.score,
            result1.score
        );
    }

    #[test]
    fn test_local_search_never_decreases() {
        let adj = path_graph(5);
        let config = AlignmentConfig::default();
        let mut rng_state: u64 = 42;

        // Build initial alignment
        let mut mapping = construct_alignment(&adj, &adj, None, &config, &mut rng_state);
        let initial_ec = edge_conservation(&mapping, &adj, &adj);

        // Run local search
        let final_score = local_search(&mut mapping, &adj, &adj, 100, &mut rng_state);

        assert!(
            final_score >= initial_ec - 1e-10,
            "Local search should not decrease score: {} < {}",
            final_score,
            initial_ec
        );
    }

    #[test]
    fn test_multiple_restarts_best_score() {
        let adj = path_graph(8);
        let config = AlignmentConfig {
            greedy_candidates: 2,
            local_search_depth: 10,
            ..AlignmentConfig::default()
        };

        let result_1 = grasp_alignment(&adj, &adj, None, &config, 1).expect("should succeed");
        let result_5 = grasp_alignment(&adj, &adj, None, &config, 5).expect("should succeed");

        assert!(
            result_5.score >= result_1.score - 1e-10,
            "More restarts should give >= score"
        );
    }

    #[test]
    fn test_complete_graph_perfect_alignment() {
        let adj = complete_graph(4);
        let config = AlignmentConfig {
            local_search_depth: 50,
            ..AlignmentConfig::default()
        };

        let result = grasp_alignment(&adj, &adj, None, &config, 5).expect("should succeed");

        // For complete graph, any mapping preserves all edges
        assert!(
            (result.edge_conservation - 1.0).abs() < 1e-10,
            "Complete graph alignment should have EC=1.0, got {}",
            result.edge_conservation
        );
    }

    #[test]
    fn test_grasp_empty_graphs() {
        let adj = Array2::<f64>::zeros((0, 0));
        let config = AlignmentConfig::default();
        let result = grasp_alignment(&adj, &adj, None, &config, 3).expect("should handle empty");
        assert!(result.mapping.is_empty());
    }

    #[test]
    fn test_grasp_single_node() {
        let adj = Array2::<f64>::zeros((1, 1));
        let config = AlignmentConfig::default();
        let result =
            grasp_alignment(&adj, &adj, None, &config, 3).expect("should handle single node");
        assert_eq!(result.mapping, vec![(0, 0)]);
    }

    #[test]
    fn test_grasp_zero_restarts_error() {
        let adj = path_graph(3);
        let config = AlignmentConfig::default();
        assert!(grasp_alignment(&adj, &adj, None, &config, 0).is_err());
    }

    #[test]
    fn test_grasp_disconnected_components() {
        // Two isolated edges
        let mut adj = Array2::zeros((4, 4));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[2, 3]] = 1.0;
        adj[[3, 2]] = 1.0;

        let config = AlignmentConfig::default();
        let result = grasp_alignment(&adj, &adj, None, &config, 5).expect("should succeed");
        assert_eq!(result.mapping.len(), 4);
        assert!(result.edge_conservation > 0.0);
    }
}
