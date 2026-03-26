//! CPU-parallel PageRank via power iteration.
//!
//! Supports standard and personalized PageRank. The implementation builds a
//! Compressed-Sparse-Row (CSR) adjacency structure in-place so no external
//! sparse-matrix crate is required.

use crate::error::{GraphError, Result};

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the PageRank algorithm.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct PageRankConfig {
    /// Damping factor α (probability of following a link). Default 0.85.
    pub damping_factor: f64,
    /// Maximum number of power iterations. Default 100.
    pub max_iterations: usize,
    /// Convergence threshold (L1 norm). Default 1e-6.
    pub tolerance: f64,
    /// Optional personalisation vector (length = n_nodes, must sum to 1).
    /// `None` means uniform teleport: each node gets weight 1/n.
    pub personalization: Option<Vec<f64>>,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping_factor: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
            personalization: None,
        }
    }
}

// ── Result ────────────────────────────────────────────────────────────────────

/// Result produced by [`pagerank`].
#[derive(Debug, Clone)]
pub struct PageRankResult {
    /// PageRank score for every node (length = n_nodes, sums to ≈ 1).
    pub scores: Vec<f64>,
    /// Number of power iterations executed.
    pub iterations: usize,
    /// Whether the algorithm converged within the tolerance.
    pub converged: bool,
    /// L1 residual at the final iteration.
    pub residual: f64,
}

// ── CSR helpers ───────────────────────────────────────────────────────────────

/// Lightweight CSR structure built from a weighted edge list.
struct Csr {
    /// row_ptr[v] .. row_ptr[v+1] indexes into `col` for node v's out-edges.
    row_ptr: Vec<usize>,
    /// Destination nodes.
    col: Vec<usize>,
    /// Edge weights (normalised so out-weights of every node sum to 1).
    weight: Vec<f64>,
    /// Out-degree of every node (used to detect dangling nodes).
    out_degree: Vec<f64>,
}

impl Csr {
    /// Build CSR from a (src, dst, weight) edge list.  Duplicate edges are
    /// accumulated.  Self-loops are included.
    fn build(edges: &[(usize, usize, f64)], n: usize) -> Result<Self> {
        // Count out-degree
        let mut raw_out: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for &(src, dst, w) in edges {
            if src >= n || dst >= n {
                return Err(GraphError::InvalidParameter {
                    param: "edge".to_string(),
                    value: format!("({src},{dst})"),
                    expected: format!("node indices in 0..{n}"),
                    context: "PageRank CSR build".to_string(),
                });
            }
            if w < 0.0 {
                return Err(GraphError::InvalidParameter {
                    param: "weight".to_string(),
                    value: format!("{w}"),
                    expected: "non-negative weight".to_string(),
                    context: "PageRank CSR build".to_string(),
                });
            }
            raw_out[src].push((dst, w));
        }

        let mut out_degree = vec![0.0_f64; n];
        for v in 0..n {
            for &(_, w) in &raw_out[v] {
                out_degree[v] += w;
            }
        }

        // Build CSR with weight-normalised entries
        let mut row_ptr = vec![0usize; n + 1];
        for v in 0..n {
            row_ptr[v + 1] = row_ptr[v] + raw_out[v].len();
        }
        let nnz = *row_ptr.last().unwrap_or(&0);
        let mut col = vec![0usize; nnz];
        let mut weight = vec![0.0_f64; nnz];

        for v in 0..n {
            let base = row_ptr[v];
            let od = out_degree[v];
            for (i, &(dst, w)) in raw_out[v].iter().enumerate() {
                col[base + i] = dst;
                // normalise so the column corresponds to the stochastic matrix
                weight[base + i] = if od > 0.0 { w / od } else { 0.0 };
            }
        }

        Ok(Self {
            row_ptr,
            col,
            weight,
            out_degree,
        })
    }
}

// ── Main function ─────────────────────────────────────────────────────────────

/// Compute PageRank scores for every node.
///
/// # Arguments
/// * `adj`     – edge list as `(src, dst, weight)` triples.
/// * `n_nodes` – total number of nodes (must be ≥ all node indices + 1).
/// * `config`  – algorithm parameters.
///
/// # Returns
/// A [`PageRankResult`] with normalised scores summing to ≈ 1.
///
/// # Errors
/// * [`GraphError::InvalidParameter`] – if `n_nodes` is 0, damping factor is
///   outside (0, 1], or an edge references an out-of-range node.
/// * [`GraphError::InvalidParameter`] – if a personalization vector has wrong
///   length or contains negative values.
pub fn pagerank(
    adj: &[(usize, usize, f64)],
    n_nodes: usize,
    config: &PageRankConfig,
) -> Result<PageRankResult> {
    if n_nodes == 0 {
        return Ok(PageRankResult {
            scores: vec![],
            iterations: 0,
            converged: true,
            residual: 0.0,
        });
    }
    let d = config.damping_factor;
    if !(0.0..=1.0).contains(&d) {
        return Err(GraphError::InvalidParameter {
            param: "damping_factor".to_string(),
            value: format!("{d}"),
            expected: "value in [0, 1]".to_string(),
            context: "PageRank".to_string(),
        });
    }
    let n = n_nodes;
    let n_f = n as f64;

    // Validate / build personalisation vector
    let teleport: Vec<f64> = match &config.personalization {
        None => vec![(1.0 - d) / n_f; n],
        Some(p) => {
            if p.len() != n {
                return Err(GraphError::InvalidParameter {
                    param: "personalization".to_string(),
                    value: format!("length {}", p.len()),
                    expected: format!("length {n}"),
                    context: "PageRank".to_string(),
                });
            }
            if p.iter().any(|&x| x < 0.0) {
                return Err(GraphError::InvalidParameter {
                    param: "personalization".to_string(),
                    value: "contains negative values".to_string(),
                    expected: "non-negative values".to_string(),
                    context: "PageRank".to_string(),
                });
            }
            // Normalise then scale by (1-d)
            let total: f64 = p.iter().sum();
            let norm = if total > 0.0 { total } else { 1.0 };
            p.iter().map(|&x| (1.0 - d) * x / norm).collect()
        }
    };

    let csr = Csr::build(adj, n)?;

    // r[v] = 1/n initially
    let mut r: Vec<f64> = vec![1.0 / n_f; n];
    let mut r_new: Vec<f64> = vec![0.0; n];

    let mut residual = f64::MAX;
    let mut iters = 0usize;
    let mut converged = false;

    // Identify dangling nodes once (out_degree == 0)
    let dangling: Vec<usize> = (0..n)
        .filter(|&v| csr.out_degree[v] == 0.0)
        .collect();

    for _iter in 0..config.max_iterations {
        iters += 1;

        // Dangling-node mass: distribute uniformly
        let dangling_mass: f64 = dangling.iter().map(|&v| r[v]).sum::<f64>() / n_f;

        // r_new[v] = teleport[v] + d * (dangling mass + Σ_{u→v} r[u] * w_norm[u,v])
        for v in 0..n {
            r_new[v] = teleport[v] + d * dangling_mass;
        }

        // Scatter: for each edge u→v add d * r[u] * w_norm
        for u in 0..n {
            let ru = r[u];
            if ru == 0.0 {
                continue;
            }
            for idx in csr.row_ptr[u]..csr.row_ptr[u + 1] {
                let v = csr.col[idx];
                let w = csr.weight[idx];
                r_new[v] += d * ru * w;
            }
        }

        // Compute L1 residual
        residual = r_new
            .iter()
            .zip(r.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        // Swap buffers
        std::mem::swap(&mut r, &mut r_new);

        if residual < config.tolerance {
            converged = true;
            break;
        }
    }

    // Normalise so scores sum exactly to 1
    let total: f64 = r.iter().sum();
    if total > 0.0 {
        for x in r.iter_mut() {
            *x /= total;
        }
    }

    Ok(PageRankResult {
        scores: r,
        iterations: iters,
        converged,
        residual,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a complete directed graph (all n*(n-1) directed edges, weight 1).
    fn complete_graph_edges(n: usize) -> Vec<(usize, usize, f64)> {
        let mut edges = Vec::new();
        for u in 0..n {
            for v in 0..n {
                if u != v {
                    edges.push((u, v, 1.0));
                }
            }
        }
        edges
    }

    #[test]
    fn test_pr_complete_graph_equal_scores() {
        let n = 6;
        let edges = complete_graph_edges(n);
        let cfg = PageRankConfig::default();
        let result = pagerank(&edges, n, &cfg).unwrap();
        assert_eq!(result.scores.len(), n);
        let expected = 1.0 / n as f64;
        for &s in &result.scores {
            assert!(
                (s - expected).abs() < 1e-5,
                "score {s} ≠ expected {expected}"
            );
        }
    }

    #[test]
    fn test_pr_star_hub_higher_score() {
        // Hub node 0, leaves 1..=4 all point to hub
        let n = 5;
        let mut edges = Vec::new();
        for leaf in 1..n {
            edges.push((leaf, 0, 1.0)); // leaf → hub
            edges.push((0, leaf, 1.0)); // hub → leaf
        }
        let cfg = PageRankConfig::default();
        let result = pagerank(&edges, n, &cfg).unwrap();
        let hub = result.scores[0];
        for leaf in 1..n {
            assert!(
                hub > result.scores[leaf],
                "hub score {hub} should be larger than leaf score {}",
                result.scores[leaf]
            );
        }
    }

    #[test]
    fn test_pr_damping_zero_uniform() {
        // With d=0 every node gets teleport probability = 1/n regardless of links.
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let cfg = PageRankConfig {
            damping_factor: 0.0,
            max_iterations: 50,
            tolerance: 1e-9,
            personalization: None,
        };
        let result = pagerank(&edges, 3, &cfg).unwrap();
        let expected = 1.0 / 3.0;
        for &s in &result.scores {
            assert!((s - expected).abs() < 1e-6, "score {s} ≠ {expected}");
        }
    }

    #[test]
    fn test_pr_damping_one_pure_link() {
        // With d=1 (no teleportation) a simple 3-cycle gives equal scores.
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let cfg = PageRankConfig {
            damping_factor: 1.0,
            max_iterations: 200,
            tolerance: 1e-9,
            personalization: None,
        };
        let result = pagerank(&edges, 3, &cfg).unwrap();
        // All three nodes in a cycle are equivalent
        let max_diff = result
            .scores
            .iter()
            .map(|s| (s - 1.0 / 3.0).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff < 1e-5, "max deviation = {max_diff}");
    }

    #[test]
    fn test_pr_convergence_flag() {
        let edges = complete_graph_edges(4);
        let cfg = PageRankConfig {
            max_iterations: 200,
            tolerance: 1e-8,
            ..Default::default()
        };
        let result = pagerank(&edges, 4, &cfg).unwrap();
        assert!(result.converged, "should converge on a complete graph");
    }

    #[test]
    fn test_pr_personalized_concentrates() {
        // Personalization heavily on node 0; node 0 should rank highest.
        let n = 5;
        let edges = complete_graph_edges(n);
        let mut p = vec![0.0f64; n];
        p[0] = 1.0;
        let cfg = PageRankConfig {
            damping_factor: 0.85,
            max_iterations: 200,
            tolerance: 1e-9,
            personalization: Some(p),
        };
        let result = pagerank(&edges, n, &cfg).unwrap();
        let best = result
            .scores
            .iter()
            .cloned()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(best, 0, "personalised node 0 should have highest rank");
    }

    #[test]
    fn test_pr_empty_graph() {
        let result = pagerank(&[], 0, &PageRankConfig::default()).unwrap();
        assert!(result.scores.is_empty());
        assert!(result.converged);
    }

    #[test]
    fn test_pr_single_node_dangling() {
        // Single node with no edges is dangling — score should be 1.0.
        let result = pagerank(&[], 1, &PageRankConfig::default()).unwrap();
        assert_eq!(result.scores.len(), 1);
        assert!((result.scores[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pr_scores_sum_to_one() {
        let edges = complete_graph_edges(8);
        let result = pagerank(&edges, 8, &PageRankConfig::default()).unwrap();
        let total: f64 = result.scores.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "sum = {total}");
    }

    #[test]
    fn test_pr_invalid_damping() {
        let err = pagerank(&[], 3, &PageRankConfig {
            damping_factor: 1.5,
            ..Default::default()
        });
        assert!(err.is_err());
    }
}
