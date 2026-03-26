//! Coreset-based graph condensation algorithms.
//!
//! Provides node selection strategies that choose a representative subset
//! of nodes from the original graph:
//!
//! - **K-center greedy**: Farthest-point sampling for geometric diversity.
//! - **Importance sampling**: Degree-weighted + feature-diversity sampling.
//! - **Herding**: Kernel herding that minimises MMD to the full dataset.
//! - **Subgraph extraction**: Builds the induced subgraph from selected nodes.

use scirs2_core::ndarray::{Array1, Array2, Axis};

use crate::error::{GraphError, Result};

use super::types::CondensedGraph;

// ---------------------------------------------------------------------------
// K-center greedy
// ---------------------------------------------------------------------------

/// Greedy farthest-point sampling (k-center).
///
/// Starting from a random seed, iteratively selects the node that maximises
/// its minimum shortest-path distance (approximated via adjacency weights)
/// to the already-selected set. Runs in O(n * k) time.
///
/// # Arguments
/// * `adj` - Symmetric adjacency matrix (n x n). Non-zero entries are edge weights.
/// * `k`   - Number of nodes to select.
///
/// # Errors
/// Returns an error if `k` is zero, exceeds the number of nodes, or the
/// adjacency matrix is not square.
pub fn k_center_greedy(adj: &Array2<f64>, k: usize) -> Result<Vec<usize>> {
    let n = adj.nrows();
    validate_inputs(n, adj.ncols(), k, "k_center_greedy")?;

    // Compute pairwise shortest-path distances via BFS-like on weighted adjacency.
    // For efficiency we use the adjacency directly: distance = 1/weight for
    // connected nodes, infinity otherwise. This gives a distance proxy.
    let distances = adjacency_to_distance(adj, n);

    let mut selected: Vec<usize> = Vec::with_capacity(k);
    // min_dist[i] = min distance from node i to any selected node
    let mut min_dist = vec![f64::INFINITY; n];

    // Seed: pick node with highest degree (most connections)
    let seed = (0..n)
        .max_by(|&a, &b| {
            let deg_a: f64 = adj.row(a).sum();
            let deg_b: f64 = adj.row(b).sum();
            deg_a
                .partial_cmp(&deg_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    selected.push(seed);

    // Update min_dist from seed
    for i in 0..n {
        min_dist[i] = distances[[seed, i]];
    }

    // Greedy loop
    for _ in 1..k {
        // Pick the node with the largest min_dist
        let next = (0..n).filter(|i| !selected.contains(i)).max_by(|&a, &b| {
            min_dist[a]
                .partial_cmp(&min_dist[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let next = match next {
            Some(idx) => idx,
            None => break, // no more nodes available
        };

        selected.push(next);

        // Update min_dist
        for i in 0..n {
            let d = distances[[next, i]];
            if d < min_dist[i] {
                min_dist[i] = d;
            }
        }
    }

    Ok(selected)
}

// ---------------------------------------------------------------------------
// Importance sampling
// ---------------------------------------------------------------------------

/// Degree-weighted + feature-diversity importance sampling.
///
/// Assigns each node a score combining its normalised degree with its
/// feature norm (diversity proxy). Nodes are sampled proportionally to
/// these scores without replacement.
///
/// # Arguments
/// * `adj`      - Adjacency matrix (n x n).
/// * `k`        - Number of nodes to select.
/// * `features` - Feature matrix (n x d).
///
/// # Errors
/// Returns an error if dimensions are inconsistent or `k` is invalid.
pub fn importance_sampling(
    adj: &Array2<f64>,
    k: usize,
    features: &Array2<f64>,
) -> Result<Vec<usize>> {
    let n = adj.nrows();
    validate_inputs(n, adj.ncols(), k, "importance_sampling")?;

    if features.nrows() != n {
        return Err(GraphError::InvalidParameter {
            param: "features".to_string(),
            value: format!("{} rows", features.nrows()),
            expected: format!("{n} rows (matching adjacency)"),
            context: "importance_sampling: feature matrix row count must match adjacency size"
                .to_string(),
        });
    }

    // Degree score: normalised row sums
    let degrees: Vec<f64> = (0..n).map(|i| adj.row(i).sum()).collect();
    let max_deg = degrees
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1e-12);

    // Feature diversity: L2 norm of each feature vector
    let feat_norms: Vec<f64> = (0..n)
        .map(|i| {
            let row = features.row(i);
            row.dot(&row).sqrt()
        })
        .collect();
    let max_norm = feat_norms
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1e-12);

    // Combined score: 0.5 * degree_norm + 0.5 * feature_norm
    let scores: Vec<f64> = (0..n)
        .map(|i| 0.5 * (degrees[i] / max_deg) + 0.5 * (feat_norms[i] / max_norm))
        .collect();

    // Deterministic weighted selection without replacement (greedy top-k by score,
    // with diversity penalty to avoid clustering)
    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut adjusted_scores = scores.clone();

    for _ in 0..k {
        let best = (0..n).filter(|i| !selected.contains(i)).max_by(|&a, &b| {
            adjusted_scores[a]
                .partial_cmp(&adjusted_scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best = match best {
            Some(idx) => idx,
            None => break,
        };

        selected.push(best);

        // Penalise neighbours of the selected node to encourage diversity
        for j in 0..n {
            if adj[[best, j]] > 0.0 && !selected.contains(&j) {
                adjusted_scores[j] *= 0.7; // decay factor
            }
        }
    }

    Ok(selected)
}

// ---------------------------------------------------------------------------
// Herding
// ---------------------------------------------------------------------------

/// Kernel herding selection.
///
/// Greedily selects points that minimise the Maximum Mean Discrepancy (MMD)
/// between the selected subset and the full dataset in the feature space.
/// Uses a linear kernel k(x, y) = x . y for efficiency.
///
/// # Arguments
/// * `features` - Feature matrix (n x d).
/// * `k`        - Number of points to select.
///
/// # Errors
/// Returns an error if `k` is zero or exceeds the number of data points.
pub fn herding_selection(features: &Array2<f64>, k: usize) -> Result<Vec<usize>> {
    let n = features.nrows();
    if k == 0 {
        return Err(GraphError::InvalidParameter {
            param: "k".to_string(),
            value: "0".to_string(),
            expected: "k > 0".to_string(),
            context: "herding_selection: must select at least one node".to_string(),
        });
    }
    if k > n {
        return Err(GraphError::InvalidParameter {
            param: "k".to_string(),
            value: k.to_string(),
            expected: format!("k <= {n}"),
            context: "herding_selection: cannot select more nodes than available".to_string(),
        });
    }

    let d = features.ncols();

    // Mean feature vector of the full dataset
    let mean_features: Array1<f64> = features
        .mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(d));

    let mut selected: Vec<usize> = Vec::with_capacity(k);
    // Running sum of selected feature vectors
    let mut running_sum = Array1::<f64>::zeros(d);

    for t in 0..k {
        let t_f64 = (t + 1) as f64;
        // Target: (t+1) * mean_features
        // We want to pick the point that makes running_sum + x_i closest to target
        // Equivalently, pick i that maximises the inner product with
        // (target - running_sum) = (t+1)*mean - running_sum

        let target_residual = &mean_features * t_f64 - &running_sum;

        let best = (0..n).filter(|i| !selected.contains(i)).max_by(|&a, &b| {
            let score_a: f64 = features.row(a).dot(&target_residual);
            let score_b: f64 = features.row(b).dot(&target_residual);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best = match best {
            Some(idx) => idx,
            None => break,
        };

        selected.push(best);
        running_sum = &running_sum + &features.row(best).to_owned();
    }

    Ok(selected)
}

// ---------------------------------------------------------------------------
// Subgraph extraction
// ---------------------------------------------------------------------------

/// Extract the induced subgraph from the selected node indices.
///
/// Builds a new adjacency matrix, feature matrix, and label vector
/// containing only the selected nodes and edges between them.
///
/// # Arguments
/// * `adj`      - Original adjacency matrix (n x n).
/// * `features` - Original feature matrix (n x d).
/// * `labels`   - Original node labels (length n).
/// * `selected` - Indices of nodes to keep.
///
/// # Errors
/// Returns an error if any selected index is out of bounds or dimensions
/// are inconsistent.
pub fn extract_subgraph(
    adj: &Array2<f64>,
    features: &Array2<f64>,
    labels: &[usize],
    selected: &[usize],
) -> Result<CondensedGraph> {
    let n = adj.nrows();

    if adj.nrows() != adj.ncols() {
        return Err(GraphError::InvalidParameter {
            param: "adj".to_string(),
            value: format!("{}x{}", adj.nrows(), adj.ncols()),
            expected: "square matrix".to_string(),
            context: "extract_subgraph".to_string(),
        });
    }
    if features.nrows() != n {
        return Err(GraphError::InvalidParameter {
            param: "features".to_string(),
            value: format!("{} rows", features.nrows()),
            expected: format!("{n} rows"),
            context: "extract_subgraph: feature row count must match adjacency".to_string(),
        });
    }
    if labels.len() != n {
        return Err(GraphError::InvalidParameter {
            param: "labels".to_string(),
            value: format!("length {}", labels.len()),
            expected: format!("length {n}"),
            context: "extract_subgraph: label count must match adjacency".to_string(),
        });
    }

    for &idx in selected {
        if idx >= n {
            return Err(GraphError::InvalidParameter {
                param: "selected".to_string(),
                value: format!("index {idx}"),
                expected: format!("index < {n}"),
                context: "extract_subgraph: selected index out of bounds".to_string(),
            });
        }
    }

    let k = selected.len();
    let d = features.ncols();

    let mut sub_adj = Array2::<f64>::zeros((k, k));
    let mut sub_features = Array2::<f64>::zeros((k, d));
    let mut sub_labels = Vec::with_capacity(k);

    for (new_i, &orig_i) in selected.iter().enumerate() {
        for (new_j, &orig_j) in selected.iter().enumerate() {
            sub_adj[[new_i, new_j]] = adj[[orig_i, orig_j]];
        }
        for f in 0..d {
            sub_features[[new_i, f]] = features[[orig_i, f]];
        }
        sub_labels.push(labels[orig_i]);
    }

    Ok(CondensedGraph {
        adjacency: sub_adj,
        features: sub_features,
        labels: sub_labels,
        source_mapping: selected.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute MMD^2 (linear kernel) between two sets of feature vectors.
///
/// MMD^2 = ||mean(X) - mean(Y)||^2  (for the linear kernel).
pub fn compute_mmd_squared(x: &Array2<f64>, y: &Array2<f64>) -> f64 {
    let mean_x = x.mean_axis(Axis(0));
    let mean_y = y.mean_axis(Axis(0));
    match (mean_x, mean_y) {
        (Some(mx), Some(my)) => {
            let diff = &mx - &my;
            diff.dot(&diff)
        }
        _ => 0.0,
    }
}

/// Validate common input parameters for coreset algorithms.
fn validate_inputs(n: usize, ncols: usize, k: usize, context: &str) -> Result<()> {
    if n != ncols {
        return Err(GraphError::InvalidParameter {
            param: "adj".to_string(),
            value: format!("{n}x{ncols}"),
            expected: "square matrix".to_string(),
            context: context.to_string(),
        });
    }
    if k == 0 {
        return Err(GraphError::InvalidParameter {
            param: "k".to_string(),
            value: "0".to_string(),
            expected: "k > 0".to_string(),
            context: format!("{context}: must select at least one node"),
        });
    }
    if k > n {
        return Err(GraphError::InvalidParameter {
            param: "k".to_string(),
            value: k.to_string(),
            expected: format!("k <= {n}"),
            context: format!("{context}: cannot select more nodes than available"),
        });
    }
    Ok(())
}

/// Convert adjacency matrix to distance matrix.
///
/// For connected pairs: distance = 1 / weight.
/// For disconnected pairs: distance = a large finite value (n * 10).
fn adjacency_to_distance(adj: &Array2<f64>, n: usize) -> Array2<f64> {
    let large = (n as f64) * 10.0;
    let mut dist = Array2::<f64>::from_elem((n, n), large);

    for i in 0..n {
        dist[[i, i]] = 0.0;
        for j in 0..n {
            if i != j && adj[[i, j]] > 0.0 {
                dist[[i, j]] = 1.0 / adj[[i, j]];
            }
        }
    }

    // Floyd-Warshall for all-pairs shortest paths
    for via in 0..n {
        for i in 0..n {
            for j in 0..n {
                let through_via = dist[[i, via]] + dist[[via, j]];
                if through_via < dist[[i, j]] {
                    dist[[i, j]] = through_via;
                }
            }
        }
    }

    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a two-cluster graph: nodes 0..half are connected to each other,
    /// nodes half..n are connected to each other, with a single bridge edge.
    fn two_cluster_graph(half: usize) -> (Array2<f64>, Array2<f64>, Vec<usize>) {
        let n = half * 2;
        let d = 2; // feature dimension
        let mut adj = Array2::<f64>::zeros((n, n));
        let mut features = Array2::<f64>::zeros((n, d));
        let mut labels = vec![0usize; n];

        // Cluster 0: fully connected, features near (0, 0)
        for i in 0..half {
            for j in 0..half {
                if i != j {
                    adj[[i, j]] = 1.0;
                }
            }
            features[[i, 0]] = i as f64 * 0.1;
            features[[i, 1]] = 0.0;
            labels[i] = 0;
        }

        // Cluster 1: fully connected, features near (10, 10)
        for i in half..n {
            for j in half..n {
                if i != j {
                    adj[[i, j]] = 1.0;
                }
            }
            features[[i, 0]] = 10.0 + (i - half) as f64 * 0.1;
            features[[i, 1]] = 10.0;
            labels[i] = 1;
        }

        // Bridge edge between the two clusters
        adj[[half - 1, half]] = 1.0;
        adj[[half, half - 1]] = 1.0;

        (adj, features, labels)
    }

    // -----------------------------------------------------------------------
    // k_center_greedy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_k_center_greedy_selects_diverse_nodes() {
        let (adj, _features, _labels) = two_cluster_graph(4);
        let selected =
            k_center_greedy(&adj, 4).expect("k_center_greedy should succeed on valid input");

        assert_eq!(selected.len(), 4);

        // Should pick nodes from both clusters
        let from_cluster0 = selected.iter().filter(|&&s| s < 4).count();
        let from_cluster1 = selected.iter().filter(|&&s| s >= 4).count();
        assert!(
            from_cluster0 >= 1,
            "should pick at least 1 node from cluster 0"
        );
        assert!(
            from_cluster1 >= 1,
            "should pick at least 1 node from cluster 1"
        );
    }

    #[test]
    fn test_k_center_greedy_single_node() {
        let (adj, _features, _labels) = two_cluster_graph(3);
        let selected = k_center_greedy(&adj, 1).expect("k_center_greedy should succeed for k=1");
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn test_k_center_greedy_all_nodes() {
        let (adj, _features, _labels) = two_cluster_graph(3);
        let n = adj.nrows();
        let selected = k_center_greedy(&adj, n).expect("k_center_greedy should succeed for k=n");
        assert_eq!(selected.len(), n);
        // All nodes should be selected (each exactly once)
        let mut sorted = selected.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), n);
    }

    #[test]
    fn test_k_center_greedy_error_k_zero() {
        let adj = Array2::<f64>::zeros((4, 4));
        let result = k_center_greedy(&adj, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_k_center_greedy_error_k_too_large() {
        let adj = Array2::<f64>::zeros((4, 4));
        let result = k_center_greedy(&adj, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_k_center_greedy_error_non_square() {
        let adj = Array2::<f64>::zeros((3, 4));
        let result = k_center_greedy(&adj, 2);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // importance_sampling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_importance_sampling_weighted_by_degree() {
        let (adj, features, _labels) = two_cluster_graph(4);
        let selected =
            importance_sampling(&adj, 4, &features).expect("importance_sampling should succeed");

        assert_eq!(selected.len(), 4);

        // Higher-degree nodes should be preferred. In a fully-connected
        // cluster of 4, all interior nodes have degree 3 (plus bridge node has degree 4).
        // The bridge nodes (3 and 4) have the highest degree.
        // Just check we get nodes from both clusters (diversity penalty).
        let from_cluster0 = selected.iter().filter(|&&s| s < 4).count();
        let from_cluster1 = selected.iter().filter(|&&s| s >= 4).count();
        assert!(from_cluster0 >= 1, "should select from cluster 0");
        assert!(from_cluster1 >= 1, "should select from cluster 1");
    }

    #[test]
    fn test_importance_sampling_single_selection() {
        let (adj, features, _labels) = two_cluster_graph(3);
        let selected = importance_sampling(&adj, 1, &features)
            .expect("importance_sampling should succeed for k=1");
        assert_eq!(selected.len(), 1);
    }

    #[test]
    fn test_importance_sampling_error_mismatched_features() {
        let adj = Array2::<f64>::zeros((4, 4));
        let features = Array2::<f64>::zeros((3, 2)); // wrong row count
        let result = importance_sampling(&adj, 2, &features);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // herding_selection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_herding_selection_reduces_mmd() {
        let (_adj, features, _labels) = two_cluster_graph(5);

        let selected = herding_selection(&features, 4).expect("herding_selection should succeed");
        assert_eq!(selected.len(), 4);

        // Build subset feature matrix
        let d = features.ncols();
        let mut subset_features = Array2::<f64>::zeros((selected.len(), d));
        for (new_i, &orig_i) in selected.iter().enumerate() {
            for f in 0..d {
                subset_features[[new_i, f]] = features[[orig_i, f]];
            }
        }

        // MMD between herding subset and full set should be small
        let mmd = compute_mmd_squared(&features, &subset_features);
        // Compare against a naive selection of the first 4 nodes (all from cluster 0)
        let naive_features = features
            .slice(scirs2_core::ndarray::s![0..4, ..])
            .to_owned();
        let mmd_naive = compute_mmd_squared(&features, &naive_features);

        assert!(
            mmd <= mmd_naive + 1e-6,
            "herding MMD ({mmd}) should be <= naive MMD ({mmd_naive})"
        );
    }

    #[test]
    fn test_herding_selection_error_k_zero() {
        let features = Array2::<f64>::zeros((5, 2));
        let result = herding_selection(&features, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_herding_selection_error_k_too_large() {
        let features = Array2::<f64>::zeros((5, 2));
        let result = herding_selection(&features, 6);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // extract_subgraph tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_subgraph_preserves_edges() {
        let (adj, features, labels) = two_cluster_graph(3);
        // Select nodes 0,1,2 (cluster 0) — they are fully connected
        let selected = vec![0, 1, 2];
        let sub = extract_subgraph(&adj, &features, &labels, &selected)
            .expect("extract_subgraph should succeed");

        assert_eq!(sub.adjacency.nrows(), 3);
        assert_eq!(sub.adjacency.ncols(), 3);
        // All off-diagonal edges should be 1.0 (fully connected cluster)
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(
                        (sub.adjacency[[i, j]] - 1.0).abs() < 1e-12,
                        "edge ({i},{j}) should be 1.0"
                    );
                }
            }
        }
        // Labels should all be 0
        assert!(sub.labels.iter().all(|&l| l == 0));
        assert_eq!(sub.source_mapping, vec![0, 1, 2]);
    }

    #[test]
    fn test_extract_subgraph_cross_cluster() {
        let (adj, features, labels) = two_cluster_graph(3);
        // Select one node from each cluster that are NOT bridge-connected
        let selected = vec![0, 5];
        let sub = extract_subgraph(&adj, &features, &labels, &selected)
            .expect("extract_subgraph should succeed");

        assert_eq!(sub.adjacency.nrows(), 2);
        // Nodes 0 and 5 are not directly connected
        assert!(sub.adjacency[[0, 1]].abs() < 1e-12);
        assert!(sub.adjacency[[1, 0]].abs() < 1e-12);
        // Labels: node 0 => label 0, node 5 => label 1
        assert_eq!(sub.labels, vec![0, 1]);
    }

    #[test]
    fn test_extract_subgraph_features_preserved() {
        let (adj, features, labels) = two_cluster_graph(3);
        let selected = vec![1, 4];
        let sub = extract_subgraph(&adj, &features, &labels, &selected)
            .expect("extract_subgraph should succeed");

        assert_eq!(sub.features.nrows(), 2);
        // Features of node 1 from original
        for f in 0..features.ncols() {
            assert!(
                (sub.features[[0, f]] - features[[1, f]]).abs() < 1e-12,
                "feature mismatch at dim {f}"
            );
        }
        // Features of node 4 from original
        for f in 0..features.ncols() {
            assert!(
                (sub.features[[1, f]] - features[[4, f]]).abs() < 1e-12,
                "feature mismatch at dim {f}"
            );
        }
    }

    #[test]
    fn test_extract_subgraph_error_index_out_of_bounds() {
        let (adj, features, labels) = two_cluster_graph(3);
        let result = extract_subgraph(&adj, &features, &labels, &[0, 100]);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_subgraph_error_non_square_adj() {
        let adj = Array2::<f64>::zeros((3, 4));
        let features = Array2::<f64>::zeros((3, 2));
        let labels = vec![0, 1, 0];
        let result = extract_subgraph(&adj, &features, &labels, &[0]);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // compute_mmd_squared tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mmd_squared_identical() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid shape");
        let mmd = compute_mmd_squared(&x, &x);
        assert!(
            mmd.abs() < 1e-12,
            "MMD of identical sets should be 0, got {mmd}"
        );
    }

    #[test]
    fn test_mmd_squared_different() {
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.0, 0.0]).expect("valid shape");
        let y = Array2::from_shape_vec((2, 2), vec![10.0, 10.0, 10.0, 10.0]).expect("valid shape");
        let mmd = compute_mmd_squared(&x, &y);
        assert!(
            mmd > 100.0,
            "MMD of distant sets should be large, got {mmd}"
        );
    }
}
