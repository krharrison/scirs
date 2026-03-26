//! IsoRank network alignment algorithm.
//!
//! Implements the IsoRank algorithm (Singh et al., 2008) for global network alignment
//! using spectral methods. The algorithm computes a similarity matrix via power iteration
//! on the Kronecker product of two adjacency matrices, then extracts a one-to-one mapping
//! using greedy or Hungarian matching.
//!
//! # Algorithm Overview
//!
//! IsoRank iteratively refines a similarity matrix `R` according to:
//!
//! ```text
//! R_new = alpha * (A1 * R * A2^T) + (1 - alpha) * E
//! ```
//!
//! where `A1`, `A2` are adjacency matrices and `E` is a prior similarity matrix.
//! The key insight is that `R[i,j]` captures both topological similarity (via `alpha`)
//! and sequence/attribute similarity (via `E`).

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{GraphError, Result};

use super::evaluation::edge_conservation;
use super::types::{AlignmentConfig, AlignmentResult, SimilarityMatrix};

/// Run the IsoRank algorithm to align two networks.
///
/// Given adjacency matrices of two graphs and an optional prior similarity matrix,
/// computes a functional similarity matrix via power iteration and extracts a
/// one-to-one node mapping.
///
/// # Arguments
///
/// * `adj1` - Adjacency matrix of graph 1 `[n1 x n1]`
/// * `adj2` - Adjacency matrix of graph 2 `[n2 x n2]`
/// * `prior` - Optional prior similarity matrix `[n1 x n2]`. If `None`, uniform prior is used.
/// * `config` - Algorithm configuration parameters
///
/// # Returns
///
/// An `AlignmentResult` containing the node mapping, quality score, and convergence info.
///
/// # Errors
///
/// Returns an error if adjacency matrices are not square or dimensions are zero.
pub fn isorank(
    adj1: &Array2<f64>,
    adj2: &Array2<f64>,
    prior: Option<&Array2<f64>>,
    config: &AlignmentConfig,
) -> Result<AlignmentResult> {
    let n1 = adj1.nrows();
    let n2 = adj2.nrows();

    // Validate inputs
    if adj1.nrows() != adj1.ncols() {
        return Err(GraphError::InvalidParameter {
            param: "adj1".to_string(),
            value: format!("{}x{}", adj1.nrows(), adj1.ncols()),
            expected: "square matrix".to_string(),
            context: "isorank".to_string(),
        });
    }
    if adj2.nrows() != adj2.ncols() {
        return Err(GraphError::InvalidParameter {
            param: "adj2".to_string(),
            value: format!("{}x{}", adj2.nrows(), adj2.ncols()),
            expected: "square matrix".to_string(),
            context: "isorank".to_string(),
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
            iterations: 0,
        });
    }

    // Validate prior dimensions if provided
    if let Some(p) = prior {
        if p.nrows() != n1 || p.ncols() != n2 {
            return Err(GraphError::InvalidParameter {
                param: "prior".to_string(),
                value: format!("{}x{}", p.nrows(), p.ncols()),
                expected: format!("{}x{}", n1, n2),
                context: "isorank: prior dimensions must match graph sizes".to_string(),
            });
        }
    }

    // Validate alpha
    if !(0.0..=1.0).contains(&config.alpha) {
        return Err(GraphError::InvalidParameter {
            param: "alpha".to_string(),
            value: config.alpha.to_string(),
            expected: "value in [0.0, 1.0]".to_string(),
            context: "isorank".to_string(),
        });
    }

    // Normalize adjacency matrices (row-stochastic)
    let norm_adj1 = row_normalize(adj1);
    let norm_adj2 = row_normalize(adj2);

    // Initialize prior similarity matrix E
    let e_matrix = match prior {
        Some(p) => {
            let mut sm = SimilarityMatrix::from_prior(p.clone())?;
            sm.normalize();
            sm.as_array().clone()
        }
        None => {
            let sm = SimilarityMatrix::new(n1, n2)?;
            sm.as_array().clone()
        }
    };

    // Initialize R to the prior
    let mut r = e_matrix.clone();

    // Power iteration
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..config.max_iter {
        let delta =
            isorank_power_iteration(&norm_adj1, &norm_adj2, &mut r, &e_matrix, config.alpha);
        iterations = iter + 1;

        if delta < config.tolerance {
            converged = true;
            break;
        }
    }

    // Extract mapping using greedy matching
    let n_min = n1.min(n2);
    let mapping = if n_min <= 100 {
        hungarian_matching(&r).unwrap_or_else(|_| greedy_matching(&r))
    } else {
        greedy_matching(&r)
    };

    // Compute quality metrics
    let ec = edge_conservation(&mapping, adj1, adj2);

    // Score = sum of R values at the chosen mapping
    let score: f64 = mapping.iter().map(|&(i, j)| r[[i, j]]).sum();

    Ok(AlignmentResult {
        mapping,
        score,
        edge_conservation: ec,
        converged,
        iterations,
    })
}

/// Perform one power iteration step for IsoRank.
///
/// Computes: `R_new = alpha * (A1 * R * A2^T) + (1 - alpha) * E`
///
/// This avoids explicitly forming the Kronecker product by using the identity:
/// `(A1 otimes A2) vec(R) = vec(A1 * R * A2^T)` (for column-stochastic form)
/// which allows O(n1^2 * n2 + n1 * n2^2) computation instead of O(n1^2 * n2^2).
///
/// Returns the Frobenius norm of the change (convergence delta).
fn isorank_power_iteration(
    adj1: &Array2<f64>,
    adj2: &Array2<f64>,
    r: &mut Array2<f64>,
    prior: &Array2<f64>,
    alpha: f64,
) -> f64 {
    // R_new = alpha * (A1 * R * A2^T) + (1 - alpha) * E
    // Step 1: temp = R * A2^T
    let adj2_t = adj2.t();
    let temp = r.dot(&adj2_t);
    // Step 2: topo = A1 * temp
    let topo = adj1.dot(&temp);

    // R_new = alpha * topo + (1 - alpha) * E
    let r_new = &topo * alpha + prior * (1.0 - alpha);

    // Normalize R_new
    let sum: f64 = r_new.iter().sum();
    let r_normalized = if sum.abs() < f64::EPSILON {
        r_new
    } else {
        &r_new / sum
    };

    // Compute delta (Frobenius norm of change)
    let diff = &r_normalized - &*r;
    let delta = diff.iter().map(|x| x * x).sum::<f64>().sqrt();

    // Update R in place
    r.assign(&r_normalized);

    delta
}

/// Extract alignment from similarity matrix using greedy maximum weight matching.
///
/// Iteratively selects the highest-similarity unmatched pair until all nodes
/// in the smaller graph are matched.
fn greedy_matching(similarity: &Array2<f64>) -> Vec<(usize, usize)> {
    let n1 = similarity.nrows();
    let n2 = similarity.ncols();
    let n_pairs = n1.min(n2);

    // Build a sorted list of (value, i, j) pairs
    let mut candidates: Vec<(f64, usize, usize)> = Vec::with_capacity(n1 * n2);
    for i in 0..n1 {
        for j in 0..n2 {
            candidates.push((similarity[[i, j]], i, j));
        }
    }
    // Sort descending by similarity value
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut used_rows = vec![false; n1];
    let mut used_cols = vec![false; n2];
    let mut mapping = Vec::with_capacity(n_pairs);

    for (_, i, j) in &candidates {
        if mapping.len() >= n_pairs {
            break;
        }
        if !used_rows[*i] && !used_cols[*j] {
            mapping.push((*i, *j));
            used_rows[*i] = true;
            used_cols[*j] = true;
        }
    }

    mapping.sort_by_key(|&(i, _)| i);
    mapping
}

/// Hungarian algorithm for optimal maximum weight matching.
///
/// Solves the assignment problem optimally for small to medium sized matrices.
/// For large matrices, consider using `greedy_matching` instead.
///
/// # Errors
///
/// Returns an error if the similarity matrix has zero dimensions.
fn hungarian_matching(similarity: &Array2<f64>) -> Result<Vec<(usize, usize)>> {
    let n1 = similarity.nrows();
    let n2 = similarity.ncols();

    if n1 == 0 || n2 == 0 {
        return Ok(Vec::new());
    }

    // Pad to square matrix if needed (use max weight assignment formulation)
    let n = n1.max(n2);

    // Convert to cost matrix (negate for minimization, shift to non-negative)
    let max_val = similarity.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut cost = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i < n1 && j < n2 {
                cost[[i, j]] = max_val - similarity[[i, j]];
            } else {
                cost[[i, j]] = max_val; // Dummy entries get max cost
            }
        }
    }

    // Hungarian algorithm (Kuhn-Munkres)
    let assignment = kuhn_munkres(&cost, n)?;

    // Filter to valid assignments
    let mapping: Vec<(usize, usize)> = assignment
        .iter()
        .enumerate()
        .filter(|&(i, &j)| i < n1 && j < n2)
        .map(|(i, &j)| (i, j))
        .collect();

    Ok(mapping)
}

/// Kuhn-Munkres (Hungarian) algorithm for the assignment problem.
///
/// Finds the minimum-cost perfect matching in a bipartite graph represented
/// as an `n x n` cost matrix.
///
/// Returns a vector where `result[i]` is the column assigned to row `i`.
fn kuhn_munkres(cost: &Array2<f64>, n: usize) -> Result<Vec<usize>> {
    if n == 0 {
        return Ok(Vec::new());
    }

    // u[i] = potential for row i, v[j] = potential for column j
    let mut u = vec![0.0_f64; n + 1];
    let mut v = vec![0.0_f64; n + 1];
    // p[j] = row matched to column j (0 means unmatched)
    let mut p = vec![0usize; n + 1];
    // way[j] = column that leads to column j in the alternating tree
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize; // "virtual" column
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;

            for j in 1..=n {
                if !used[j] {
                    let cur = cost[[i0 - 1, j - 1]] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            // Update potentials
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;

            if p[j0] == 0 {
                break;
            }
        }

        // Recover the augmenting path
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    // Build result: assignment[row] = col (0-indexed)
    let mut assignment = vec![0usize; n];
    for j in 1..=n {
        if p[j] > 0 {
            assignment[p[j] - 1] = j - 1;
        }
    }

    Ok(assignment)
}

/// Row-normalize an adjacency matrix to make it row-stochastic.
///
/// Each row is divided by its sum. Rows with zero sum are left as zeros.
fn row_normalize(adj: &Array2<f64>) -> Array2<f64> {
    let n = adj.nrows();
    let mut result = adj.clone();
    for i in 0..n {
        let row_sum: f64 = result.row(i).sum();
        if row_sum.abs() > f64::EPSILON {
            for j in 0..adj.ncols() {
                result[[i, j]] /= row_sum;
            }
        }
    }
    result
}

/// Compute the degree vector from an adjacency matrix.
fn _degree_vector(adj: &Array2<f64>) -> Array1<f64> {
    let n = adj.nrows();
    let mut deg = Array1::zeros(n);
    for i in 0..n {
        deg[i] = adj.row(i).sum();
    }
    deg
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

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

    /// Permute rows and columns of adjacency matrix by given permutation.
    fn permute_adj(adj: &Array2<f64>, perm: &[usize]) -> Array2<f64> {
        let n = adj.nrows();
        let mut result = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                result[[perm[i], perm[j]]] = adj[[i, j]];
            }
        }
        result
    }

    #[test]
    fn test_self_alignment_identity() {
        // Aligning a graph to itself should produce identity-like mapping
        let adj = path_graph(5);
        let config = AlignmentConfig::default();
        let result = isorank(&adj, &adj, None, &config).expect("isorank should succeed");
        assert!(!result.mapping.is_empty());
        // EC should be 1.0 for self-alignment (identity mapping)
        assert!(
            result.edge_conservation > 0.9,
            "EC should be high for self-alignment, got {}",
            result.edge_conservation
        );
    }

    #[test]
    fn test_permuted_graph_recovery() {
        let adj = path_graph(6);
        let perm = vec![3, 0, 5, 2, 4, 1]; // a permutation
        let adj_perm = permute_adj(&adj, &perm);

        // Build a prior that hints at the correct mapping
        // Prior[i, perm[i]] is high, others are low
        let mut prior = Array2::from_elem((6, 6), 0.01);
        for (i, &pi) in perm.iter().enumerate() {
            prior[[i, pi]] = 1.0;
        }

        let config = AlignmentConfig {
            alpha: 0.5,
            max_iter: 200,
            tolerance: 1e-10,
            ..AlignmentConfig::default()
        };

        let result =
            isorank(&adj, &adj_perm, Some(&prior), &config).expect("isorank should succeed");
        // With prior hint, should find a good alignment
        assert!(
            result.edge_conservation > 0.7,
            "EC should be reasonable for permuted graph with prior, got {}",
            result.edge_conservation
        );
    }

    #[test]
    fn test_power_iteration_convergence() {
        let adj = path_graph(4);
        let norm_a = row_normalize(&adj);
        let e = Array2::from_elem((4, 4), 1.0 / 16.0);
        let mut r = e.clone();

        let mut prev_delta = f64::INFINITY;
        for _ in 0..20 {
            let delta = isorank_power_iteration(&norm_a, &norm_a, &mut r, &e, 0.6);
            // Delta should generally decrease (may have minor fluctuations)
            if prev_delta < f64::INFINITY {
                // Allow small increases but overall trend should be downward
                assert!(
                    delta < prev_delta * 1.5 + 1e-14,
                    "delta {} unexpectedly much larger than prev {}",
                    delta,
                    prev_delta
                );
            }
            prev_delta = delta;
        }
    }

    #[test]
    fn test_uniform_prior_topology_only() {
        let adj1 = path_graph(4);
        let adj2 = path_graph(4);
        let config = AlignmentConfig {
            alpha: 0.95,
            ..AlignmentConfig::default()
        };
        let result = isorank(&adj1, &adj2, None, &config).expect("should succeed");
        assert!(!result.mapping.is_empty());
        assert!(result.edge_conservation > 0.5);
    }

    #[test]
    fn test_small_path_alignment() {
        let adj = path_graph(4);
        let config = AlignmentConfig::default();
        let result = isorank(&adj, &adj, None, &config).expect("should succeed");
        assert_eq!(result.mapping.len(), 4);
    }

    #[test]
    fn test_empty_graphs() {
        let adj1 = Array2::<f64>::zeros((0, 0));
        let adj2 = Array2::<f64>::zeros((0, 0));
        let config = AlignmentConfig::default();
        let result = isorank(&adj1, &adj2, None, &config).expect("should handle empty");
        assert!(result.mapping.is_empty());
        assert!(result.converged);
    }

    #[test]
    fn test_single_node_graphs() {
        let adj1 = Array2::<f64>::zeros((1, 1));
        let adj2 = Array2::<f64>::zeros((1, 1));
        let config = AlignmentConfig::default();
        let result = isorank(&adj1, &adj2, None, &config).expect("should handle single node");
        assert_eq!(result.mapping, vec![(0, 0)]);
    }

    #[test]
    fn test_invalid_non_square() {
        let adj1 = Array2::<f64>::zeros((3, 4));
        let adj2 = Array2::<f64>::zeros((3, 3));
        let config = AlignmentConfig::default();
        assert!(isorank(&adj1, &adj2, None, &config).is_err());
    }

    #[test]
    fn test_invalid_alpha() {
        let adj = path_graph(3);
        let config = AlignmentConfig {
            alpha: 1.5,
            ..AlignmentConfig::default()
        };
        assert!(isorank(&adj, &adj, None, &config).is_err());
    }

    #[test]
    fn test_greedy_matching_basic() {
        let sim = array![[0.1, 0.9], [0.8, 0.2]];
        let mapping = greedy_matching(&sim);
        assert_eq!(mapping.len(), 2);
        // Best pair is (0,1) with 0.9, then (1,0) with 0.8
        assert!(mapping.contains(&(0, 1)));
        assert!(mapping.contains(&(1, 0)));
    }

    #[test]
    fn test_hungarian_matching_basic() {
        let sim = array![[0.1, 0.9], [0.8, 0.2]];
        let mapping = hungarian_matching(&sim).expect("should succeed");
        assert_eq!(mapping.len(), 2);
        assert!(mapping.contains(&(0, 1)));
        assert!(mapping.contains(&(1, 0)));
    }

    #[test]
    fn test_hungarian_matching_rectangular() {
        let sim = array![[0.1, 0.9, 0.5], [0.8, 0.2, 0.3]];
        let mapping = hungarian_matching(&sim).expect("should succeed");
        // n1=2, n2=3 -> 2 pairs
        assert_eq!(mapping.len(), 2);
    }

    #[test]
    fn test_complete_graph_self_alignment() {
        let adj = complete_graph(4);
        let config = AlignmentConfig::default();
        let result = isorank(&adj, &adj, None, &config).expect("should succeed");
        // For complete graph, any permutation is valid, so EC should be 1.0
        assert!(
            (result.edge_conservation - 1.0).abs() < 1e-10,
            "Complete graph self-alignment should have EC=1.0, got {}",
            result.edge_conservation
        );
    }

    #[test]
    fn test_disconnected_components() {
        // Two disconnected edges: 0-1 and 2-3
        let mut adj = Array2::zeros((4, 4));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[2, 3]] = 1.0;
        adj[[3, 2]] = 1.0;

        let config = AlignmentConfig::default();
        let result = isorank(&adj, &adj, None, &config).expect("should succeed");
        assert_eq!(result.mapping.len(), 4);
        assert!(result.edge_conservation > 0.5);
    }
}
