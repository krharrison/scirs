//! Tests for graph condensation (dataset distillation).

use scirs2_core::ndarray::Array2;
use scirs2_graph::condensation::{
    coreset::{
        compute_mmd_squared, extract_subgraph, herding_selection, importance_sampling,
        k_center_greedy,
    },
    distillation::{feature_alignment_loss, gradient_matching_condense, structure_matching_loss},
    evaluation::{
        degree_distribution_distance, evaluate_condensation, label_coverage, spectral_distance,
    },
    types::{CondensationConfig, CondensationMethod},
};

/// Helper: build a small two-cluster graph (nodes 0..4 in cluster A, 5..9 in cluster B).
fn two_cluster_graph() -> (Array2<f64>, Array2<f64>, Vec<usize>) {
    let n = 10;
    let d = 3;
    let mut adj = Array2::<f64>::zeros((n, n));
    let mut features = Array2::<f64>::zeros((n, d));
    let mut labels = vec![0usize; n];

    // Cluster A: nodes 0..5, fully connected
    for i in 0..5 {
        for j in 0..5 {
            if i != j {
                adj[[i, j]] = 1.0;
            }
        }
        features[[i, 0]] = 1.0;
        features[[i, 1]] = (i as f64) * 0.1;
        features[[i, 2]] = 0.0;
        labels[i] = 0;
    }

    // Cluster B: nodes 5..10, fully connected
    for i in 5..10 {
        for j in 5..10 {
            if i != j {
                adj[[i, j]] = 1.0;
            }
        }
        features[[i, 0]] = 0.0;
        features[[i, 1]] = (i as f64) * 0.1;
        features[[i, 2]] = 1.0;
        labels[i] = 1;
    }

    // One bridge edge between clusters
    adj[[2, 7]] = 1.0;
    adj[[7, 2]] = 1.0;

    (adj, features, labels)
}

/// Helper: build a simple path graph 0-1-2-3-4.
fn path_graph() -> (Array2<f64>, Array2<f64>, Vec<usize>) {
    let n = 5;
    let d = 2;
    let mut adj = Array2::<f64>::zeros((n, n));
    let mut features = Array2::<f64>::zeros((n, d));

    for i in 0..n - 1 {
        adj[[i, i + 1]] = 1.0;
        adj[[i + 1, i]] = 1.0;
    }

    for i in 0..n {
        features[[i, 0]] = i as f64;
        features[[i, 1]] = (n - i) as f64;
    }

    let labels = vec![0, 0, 1, 1, 2];
    (adj, features, labels)
}

// ===========================================================================
// K-center tests
// ===========================================================================

#[test]
fn test_k_center_selects_k_nodes() {
    let (adj, _, _) = two_cluster_graph();
    let selected = k_center_greedy(&adj, 4).expect("k_center should succeed");
    assert_eq!(selected.len(), 4);
    // All indices should be unique
    let mut dedup = selected.clone();
    dedup.sort();
    dedup.dedup();
    assert_eq!(dedup.len(), 4);
}

#[test]
fn test_k_center_selects_from_both_clusters() {
    let (adj, _, _) = two_cluster_graph();
    let selected = k_center_greedy(&adj, 4).expect("k_center should succeed");

    let from_a = selected.iter().filter(|&&i| i < 5).count();
    let from_b = selected.iter().filter(|&&i| i >= 5).count();

    // With farthest-point sampling on a 2-cluster graph, both clusters
    // should be represented.
    assert!(from_a > 0, "should select nodes from cluster A");
    assert!(from_b > 0, "should select nodes from cluster B");
}

#[test]
fn test_k_center_selects_diverse_nodes() {
    let (adj, _, _) = path_graph();
    let selected = k_center_greedy(&adj, 3).expect("k_center should succeed");

    // On a path graph, diverse selection should spread across the path.
    // The selected nodes should have a reasonable span.
    let min_node = selected.iter().copied().min().unwrap_or(0);
    let max_node = selected.iter().copied().max().unwrap_or(0);
    assert!(max_node - min_node >= 2, "should span the path graph");
}

#[test]
fn test_k_center_error_k_zero() {
    let adj = Array2::<f64>::zeros((5, 5));
    let result = k_center_greedy(&adj, 0);
    assert!(result.is_err());
}

#[test]
fn test_k_center_error_k_exceeds_n() {
    let adj = Array2::<f64>::zeros((3, 3));
    let result = k_center_greedy(&adj, 5);
    assert!(result.is_err());
}

// ===========================================================================
// Importance sampling tests
// ===========================================================================

#[test]
fn test_importance_sampling_selects_k_nodes() {
    let (adj, features, _) = two_cluster_graph();
    let selected =
        importance_sampling(&adj, 4, &features).expect("importance_sampling should succeed");
    assert_eq!(selected.len(), 4);
}

#[test]
fn test_importance_sampling_prefers_high_degree() {
    // Build a star graph: node 0 connects to all others
    let n = 6;
    let d = 2;
    let mut adj = Array2::<f64>::zeros((n, n));
    let mut features = Array2::<f64>::zeros((n, d));

    for i in 1..n {
        adj[[0, i]] = 1.0;
        adj[[i, 0]] = 1.0;
        features[[i, 0]] = i as f64;
        features[[i, 1]] = 1.0;
    }
    features[[0, 0]] = 0.0;
    features[[0, 1]] = 5.0;

    let selected = importance_sampling(&adj, 2, &features).expect("should succeed");
    // The hub node (0) should be selected first due to highest degree
    assert!(selected.contains(&0), "hub node should be selected");
}

// ===========================================================================
// Herding tests
// ===========================================================================

#[test]
fn test_herding_selects_k_nodes() {
    let (_, features, _) = two_cluster_graph();
    let selected = herding_selection(&features, 4).expect("herding should succeed");
    assert_eq!(selected.len(), 4);
}

#[test]
fn test_herding_reduces_mmd() {
    let (_, features, _) = two_cluster_graph();

    // Select 2 nodes vs 6 nodes -- 6 should have lower MMD to the full set
    let sel_2 = herding_selection(&features, 2).expect("should succeed");
    let sel_6 = herding_selection(&features, 6).expect("should succeed");

    let sub_2 = extract_feature_subset(&features, &sel_2);
    let sub_6 = extract_feature_subset(&features, &sel_6);

    let mmd_2 = compute_mmd_squared(&features, &sub_2);
    let mmd_6 = compute_mmd_squared(&features, &sub_6);

    assert!(
        mmd_6 <= mmd_2 + 1e-6,
        "selecting more nodes via herding should reduce MMD: mmd_6={mmd_6}, mmd_2={mmd_2}"
    );
}

#[test]
fn test_herding_error_k_zero() {
    let features = Array2::<f64>::zeros((5, 3));
    let result = herding_selection(&features, 0);
    assert!(result.is_err());
}

// ===========================================================================
// Subgraph extraction tests
// ===========================================================================

#[test]
fn test_extract_subgraph_preserves_edges() {
    let (adj, features, labels) = two_cluster_graph();
    // Select nodes 0, 1, 2 (all in cluster A, fully connected)
    let selected = vec![0, 1, 2];
    let sub = extract_subgraph(&adj, &features, &labels, &selected).expect("should succeed");

    assert_eq!(sub.adjacency.nrows(), 3);
    assert_eq!(sub.adjacency.ncols(), 3);

    // All pairs within cluster A should be connected
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                assert!(
                    sub.adjacency[[i, j]] > 0.0,
                    "edge ({i}, {j}) should be preserved"
                );
            }
        }
    }
}

#[test]
fn test_extract_subgraph_labels_mapping() {
    let (adj, features, labels) = two_cluster_graph();
    let selected = vec![0, 5, 9];
    let sub = extract_subgraph(&adj, &features, &labels, &selected).expect("should succeed");

    assert_eq!(sub.labels, vec![0, 1, 1]);
    assert_eq!(sub.source_mapping, vec![0, 5, 9]);
}

#[test]
fn test_extract_subgraph_out_of_bounds() {
    let (adj, features, labels) = path_graph();
    let result = extract_subgraph(&adj, &features, &labels, &[0, 1, 99]);
    assert!(result.is_err());
}

// ===========================================================================
// Gradient matching tests
// ===========================================================================

#[test]
fn test_gradient_matching_produces_output() {
    let (adj, features, labels) = two_cluster_graph();
    let config = CondensationConfig {
        target_nodes: 4,
        method: CondensationMethod::GradientMatching,
        max_iterations: 10,
        learning_rate: 0.01,
    };

    let result = gradient_matching_condense(&adj, &features, &labels, &config)
        .expect("gradient matching should succeed");

    assert_eq!(result.adjacency.nrows(), 4);
    assert_eq!(result.features.nrows(), 4);
    assert_eq!(result.labels.len(), 4);
}

#[test]
fn test_gradient_matching_loss_decreases() {
    let (adj, features, labels) = two_cluster_graph();

    // Run with few iterations
    let config_few = CondensationConfig {
        target_nodes: 4,
        method: CondensationMethod::GradientMatching,
        max_iterations: 5,
        learning_rate: 0.01,
    };
    let result_few =
        gradient_matching_condense(&adj, &features, &labels, &config_few).expect("should succeed");

    // Run with more iterations
    let config_more = CondensationConfig {
        target_nodes: 4,
        method: CondensationMethod::GradientMatching,
        max_iterations: 50,
        learning_rate: 0.01,
    };
    let result_more =
        gradient_matching_condense(&adj, &features, &labels, &config_more).expect("should succeed");

    let loss_few = feature_alignment_loss(&features, &result_few.features);
    let loss_more = feature_alignment_loss(&features, &result_more.features);

    // More iterations should generally produce equal or better alignment
    // (allow some tolerance since the optimisation is not convex)
    assert!(
        loss_more <= loss_few + 0.5,
        "more iterations should not dramatically worsen loss: few={loss_few}, more={loss_more}"
    );
}

#[test]
fn test_feature_alignment_loss_zero_for_identical() {
    let features = Array2::<f64>::ones((5, 3));
    let loss = feature_alignment_loss(&features, &features);
    assert!(
        loss < 1e-10,
        "identical features should have zero loss: {loss}"
    );
}

#[test]
fn test_structure_matching_loss_zero_for_identical() {
    let (adj, _, _) = path_graph();
    let loss = structure_matching_loss(&adj, &adj);
    assert!(
        loss < 1e-10,
        "identical graphs should have zero loss: {loss}"
    );
}

// ===========================================================================
// Evaluation tests
// ===========================================================================

#[test]
fn test_degree_distribution_distance_zero_for_identical() {
    let (adj, _, _) = path_graph();
    let dist = degree_distribution_distance(&adj, &adj);
    assert!(dist < 1e-6, "same graph should have ~0 distance: {dist}");
}

#[test]
fn test_degree_distribution_distance_positive_for_different() {
    let (adj1, _, _) = path_graph();
    let (adj2, _, _) = two_cluster_graph();
    let dist = degree_distribution_distance(&adj1, &adj2);
    assert!(dist > 0.0, "different graphs should have positive distance");
}

#[test]
fn test_spectral_distance_zero_for_identity() {
    let n = 4;
    let mut adj = Array2::<f64>::zeros((n, n));
    // Simple cycle
    for i in 0..n {
        adj[[i, (i + 1) % n]] = 1.0;
        adj[[(i + 1) % n, i]] = 1.0;
    }
    let dist = spectral_distance(&adj, &adj);
    assert!(
        dist < 1e-6,
        "same graph should have ~0 spectral distance: {dist}"
    );
}

#[test]
fn test_label_coverage_all_present() {
    let orig = vec![0, 1, 2, 0, 1, 2];
    let condensed = vec![0, 1, 2];
    let coverage = label_coverage(&orig, &condensed);
    assert!(
        (coverage - 1.0).abs() < 1e-10,
        "all labels present should give coverage=1.0: {coverage}"
    );
}

#[test]
fn test_label_coverage_partial() {
    let orig = vec![0, 1, 2, 3];
    let condensed = vec![0, 2];
    let coverage = label_coverage(&orig, &condensed);
    assert!(
        (coverage - 0.5).abs() < 1e-10,
        "half labels covered should give 0.5: {coverage}"
    );
}

#[test]
fn test_compression_ratio_correct() {
    let (adj, features, labels) = two_cluster_graph();
    let selected = k_center_greedy(&adj, 4).expect("should succeed");
    let condensed = extract_subgraph(&adj, &features, &labels, &selected).expect("should succeed");

    let orig_nodes = adj.nrows();
    let cond_nodes = condensed.adjacency.nrows();
    let ratio = orig_nodes as f64 / cond_nodes as f64;

    assert!(
        (ratio - 2.5).abs() < 1e-10,
        "10 nodes condensed to 4 should give ratio 2.5: {ratio}"
    );
}

#[test]
fn test_evaluate_condensation_complete() {
    let (adj, features, labels) = two_cluster_graph();
    let selected = k_center_greedy(&adj, 5).expect("should succeed");
    let condensed = extract_subgraph(&adj, &features, &labels, &selected).expect("should succeed");

    let metrics = evaluate_condensation(&adj, &labels, &condensed.adjacency, &condensed.labels);

    // All metrics should be finite
    assert!(metrics.degree_distribution_distance.is_finite());
    assert!(metrics.spectral_distance.is_finite());
    assert!(metrics.label_coverage.is_finite());
    assert!(metrics.label_coverage > 0.0);
    assert!(metrics.label_coverage <= 1.0);
}

#[test]
fn test_good_condensation_low_degree_distance() {
    // For a graph with uniform degree, selecting half the nodes should
    // preserve the degree distribution reasonably well.
    let n = 8;
    let mut adj = Array2::<f64>::zeros((n, n));
    // Ring graph: each node has degree 2
    for i in 0..n {
        adj[[i, (i + 1) % n]] = 1.0;
        adj[[(i + 1) % n, i]] = 1.0;
    }

    // Select every other node
    let selected = vec![0, 2, 4, 6];
    let features = Array2::<f64>::zeros((n, 1));
    let labels = vec![0; n];
    let condensed = extract_subgraph(&adj, &features, &labels, &selected).expect("should succeed");

    let dist = degree_distribution_distance(&adj, &condensed.adjacency);
    // The condensed ring subgraph may have different degree distribution,
    // but the distance should be bounded (not enormous).
    assert!(
        dist < 30.0,
        "degree distribution distance should be bounded: {dist}"
    );
}

// ===========================================================================
// Helper
// ===========================================================================

fn extract_feature_subset(features: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let d = features.ncols();
    let k = indices.len();
    let mut subset = Array2::<f64>::zeros((k, d));
    for (new_i, &orig_i) in indices.iter().enumerate() {
        for j in 0..d {
            subset[[new_i, j]] = features[[orig_i, j]];
        }
    }
    subset
}
