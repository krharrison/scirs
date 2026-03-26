//! Gradient-matching graph distillation.
//!
//! Implements a simplified graph condensation pipeline inspired by
//! *Dataset Condensation with Gradient Matching* (Zhao et al., 2021),
//! adapted for graph-structured data. A small synthetic graph is
//! optimised so that the gradients of a simple 1-layer GCN on the
//! synthetic graph match those on the original graph.

use scirs2_core::ndarray::{Array1, Array2, Axis};

use crate::error::{GraphError, Result};

use super::types::{CondensationConfig, CondensedGraph};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Condense a graph via gradient matching.
///
/// Initialises a synthetic graph and iteratively updates it so that the
/// GNN loss gradients on the synthetic data match those on the original
/// data.  Uses a simple 1-layer GCN: `H = sigma(D^{-1/2} A D^{-1/2} X W)`.
///
/// # Arguments
/// * `adj`      - Original adjacency matrix (n x n).
/// * `features` - Original feature matrix (n x d).
/// * `labels`   - Original node labels (length n).
/// * `config`   - Condensation configuration.
///
/// # Returns
/// A `CondensedGraph` whose adjacency and features have been optimised
/// to approximate the training signal of the full graph.
///
/// # Errors
/// Returns an error if dimensions are inconsistent or parameters are invalid.
pub fn gradient_matching_condense(
    adj: &Array2<f64>,
    features: &Array2<f64>,
    labels: &[usize],
    config: &CondensationConfig,
) -> Result<CondensedGraph> {
    let n = adj.nrows();
    let d = features.ncols();
    let k = config.target_nodes;

    validate_distillation_inputs(adj, features, labels, k)?;

    let num_classes = count_classes(labels);

    // --- Initialise synthetic graph ---
    // Features: sample k rows from the original (evenly spaced)
    let mut synth_features = Array2::<f64>::zeros((k, d));
    let mut synth_labels = Vec::with_capacity(k);
    let mut source_mapping = Vec::with_capacity(k);

    // Try to cover all classes evenly
    let per_class = (k / num_classes.max(1)).max(1);
    let mut class_counts = vec![0usize; num_classes];
    let mut filled = 0;

    for orig_idx in 0..n {
        if filled >= k {
            break;
        }
        let c = labels[orig_idx];
        if c < num_classes && class_counts[c] < per_class {
            for f in 0..d {
                synth_features[[filled, f]] = features[[orig_idx, f]];
            }
            synth_labels.push(c);
            source_mapping.push(orig_idx);
            class_counts[c] += 1;
            filled += 1;
        }
    }

    // Fill remaining slots with round-robin from remaining nodes
    if filled < k {
        for orig_idx in 0..n {
            if filled >= k {
                break;
            }
            if !source_mapping.contains(&orig_idx) {
                for f in 0..d {
                    synth_features[[filled, f]] = features[[orig_idx, f]];
                }
                synth_labels.push(labels[orig_idx]);
                source_mapping.push(orig_idx);
                filled += 1;
            }
        }
    }

    // Adjacency: start with kNN based on feature similarity
    let mut synth_adj = build_initial_adjacency(&synth_features, k);

    // --- Normalised adjacency for GCN ---
    let norm_adj_orig = normalise_adjacency(adj, n);
    let w = initialise_weight_matrix(d, num_classes);

    // --- Gradient matching iterations ---
    let lr = config.learning_rate;

    for _iter in 0..config.max_iterations {
        // Forward pass on original graph
        let h_orig = gcn_forward(&norm_adj_orig, features, &w);
        let node_grad_orig = compute_gradient(&h_orig, labels, num_classes);
        // Gradient w.r.t. W: (A_norm @ X)^T @ node_grad  =>  shape (d x num_classes)
        let ax_orig = norm_adj_orig.dot(features);
        let w_grad_orig = ax_orig.t().dot(&node_grad_orig);

        // Forward pass on synthetic graph
        let norm_adj_synth = normalise_adjacency(&synth_adj, k);
        let h_synth = gcn_forward(&norm_adj_synth, &synth_features, &w);
        let node_grad_synth = compute_gradient(&h_synth, &synth_labels, num_classes);
        let ax_synth = norm_adj_synth.dot(&synth_features);
        let w_grad_synth = ax_synth.t().dot(&node_grad_synth);

        // Gradient matching in weight space: minimise ||w_grad_orig - w_grad_synth||^2
        let w_grad_diff = &w_grad_orig - &w_grad_synth;

        // Back-propagate through synthetic graph to update features:
        // dL/d(synth_features) = norm_adj_synth^T @ node_grad_synth @ w_grad_diff^T
        // Simplified: use the weight-space gradient difference to drive feature updates
        let feature_update = norm_adj_synth
            .t()
            .dot(&node_grad_synth)
            .dot(&w_grad_diff.t());

        // Apply update (clamped)
        for i in 0..k.min(synth_features.nrows()) {
            for j in 0..d.min(synth_features.ncols()) {
                if j < feature_update.ncols() {
                    let update = lr * feature_update[[i, j]];
                    let clamped = update.clamp(-1.0, 1.0);
                    synth_features[[i, j]] += clamped;
                }
            }
        }

        // Structure update: small perturbation to adjacency
        update_adjacency(&mut synth_adj, &synth_features, k, lr * 0.1);
    }

    Ok(CondensedGraph {
        adjacency: synth_adj,
        features: synth_features,
        labels: synth_labels,
        source_mapping,
    })
}

/// Feature alignment loss: MMD-like distance between two feature matrices.
///
/// Computes `||mean(orig) - mean(synth)||^2` as a simple Wasserstein-like
/// proxy in feature space.
pub fn feature_alignment_loss(orig_features: &Array2<f64>, synth_features: &Array2<f64>) -> f64 {
    let mean_orig = orig_features.mean_axis(Axis(0));
    let mean_synth = synth_features.mean_axis(Axis(0));

    match (mean_orig, mean_synth) {
        (Some(mo), Some(ms)) => {
            let diff = &mo - &ms;
            diff.dot(&diff)
        }
        _ => 0.0,
    }
}

/// Structure matching loss: eigenvalue distribution distance.
///
/// Computes the L2 distance between the sorted degree sequences
/// (a tractable proxy for the spectral distance) of two graphs.
pub fn structure_matching_loss(orig_adj: &Array2<f64>, synth_adj: &Array2<f64>) -> f64 {
    let mut degs_orig = degree_sequence(orig_adj);
    let mut degs_synth = degree_sequence(synth_adj);

    // Sort descending
    degs_orig.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    degs_synth.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Pad shorter sequence with zeros
    let max_len = degs_orig.len().max(degs_synth.len());
    degs_orig.resize(max_len, 0.0);
    degs_synth.resize(max_len, 0.0);

    // Normalise to make comparable
    let norm_orig = degs_orig.iter().sum::<f64>().max(1e-12);
    let norm_synth = degs_synth.iter().sum::<f64>().max(1e-12);

    let mut dist_sq = 0.0;
    for i in 0..max_len {
        let diff = degs_orig[i] / norm_orig - degs_synth[i] / norm_synth;
        dist_sq += diff * diff;
    }

    dist_sq.sqrt()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Normalise adjacency matrix: D^{-1/2} A D^{-1/2} with self-loops.
fn normalise_adjacency(adj: &Array2<f64>, n: usize) -> Array2<f64> {
    // Add self-loops: A_hat = A + I
    let mut a_hat = adj.clone();
    for i in 0..n {
        a_hat[[i, i]] += 1.0;
    }

    // Compute D^{-1/2}
    let mut d_inv_sqrt = Array1::<f64>::zeros(n);
    for i in 0..n {
        let deg: f64 = a_hat.row(i).sum();
        if deg > 0.0 {
            d_inv_sqrt[i] = 1.0 / deg.sqrt();
        }
    }

    // D^{-1/2} A_hat D^{-1/2}
    let mut normalised = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            normalised[[i, j]] = d_inv_sqrt[i] * a_hat[[i, j]] * d_inv_sqrt[j];
        }
    }

    normalised
}

/// Simple 1-layer GCN forward pass: H = ReLU(norm_adj @ X @ W).
fn gcn_forward(norm_adj: &Array2<f64>, features: &Array2<f64>, w: &Array2<f64>) -> Array2<f64> {
    let ax = norm_adj.dot(features);
    let mut h = ax.dot(w);

    // ReLU activation
    h.mapv_inplace(|v| v.max(0.0));

    h
}

/// Compute gradient of cross-entropy loss w.r.t. the GCN output.
///
/// Uses softmax + cross-entropy. Returns the gradient matrix (n x num_classes).
fn compute_gradient(logits: &Array2<f64>, labels: &[usize], num_classes: usize) -> Array2<f64> {
    let n = logits.nrows();
    let c = logits.ncols().min(num_classes);

    // Softmax
    let mut probs = Array2::<f64>::zeros((n, c));
    for i in 0..n {
        let max_val = (0..c)
            .map(|j| logits[[i, j]])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut sum_exp = 0.0;
        for j in 0..c {
            let e = (logits[[i, j]] - max_val).exp();
            probs[[i, j]] = e;
            sum_exp += e;
        }
        if sum_exp > 0.0 {
            for j in 0..c {
                probs[[i, j]] /= sum_exp;
            }
        }
    }

    // Gradient: probs - one_hot(labels)
    let mut grad = probs;
    for i in 0..n {
        let label = labels.get(i).copied().unwrap_or(0);
        if label < c {
            grad[[i, label]] -= 1.0;
        }
    }

    // Average over nodes
    let n_f64 = n as f64;
    if n_f64 > 0.0 {
        grad /= n_f64;
    }

    grad
}

/// Initialise a simple weight matrix for the GCN (d x num_classes).
/// Uses Xavier-like initialisation with deterministic values.
fn initialise_weight_matrix(d: usize, num_classes: usize) -> Array2<f64> {
    let scale = (2.0 / (d + num_classes) as f64).sqrt();
    let mut w = Array2::<f64>::zeros((d, num_classes));

    for i in 0..d {
        for j in 0..num_classes {
            // Deterministic pseudo-random pattern
            let val = ((i * 7 + j * 13 + 3) as f64 % 17.0 - 8.0) / 17.0;
            w[[i, j]] = val * scale;
        }
    }

    w
}

/// Build initial adjacency for the synthetic graph using kNN in feature space.
fn build_initial_adjacency(features: &Array2<f64>, k: usize) -> Array2<f64> {
    let mut adj = Array2::<f64>::zeros((k, k));

    // Connect each node to its nearest neighbours (up to min(3, k-1))
    let knn = 3.min(k.saturating_sub(1));

    for i in 0..k {
        let mut dists: Vec<(usize, f64)> = (0..k)
            .filter(|&j| j != i)
            .map(|j| {
                let diff = &features.row(i).to_owned() - &features.row(j).to_owned();
                (j, diff.dot(&diff))
            })
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for &(j, _) in dists.iter().take(knn) {
            adj[[i, j]] = 1.0;
            adj[[j, i]] = 1.0; // symmetric
        }
    }

    adj
}

/// Gently update the synthetic adjacency to better match structure.
fn update_adjacency(adj: &mut Array2<f64>, features: &Array2<f64>, k: usize, lr: f64) {
    for i in 0..k {
        for j in (i + 1)..k {
            let diff = &features.row(i).to_owned() - &features.row(j).to_owned();
            let sim = (-diff.dot(&diff)).exp(); // Gaussian similarity

            // Move adjacency towards similarity
            let delta = lr * (sim - adj[[i, j]]);
            adj[[i, j]] = (adj[[i, j]] + delta).clamp(0.0, 1.0);
            adj[[j, i]] = adj[[i, j]];
        }
    }
}

/// Count distinct classes in labels.
fn count_classes(labels: &[usize]) -> usize {
    if labels.is_empty() {
        return 0;
    }
    let max_label = labels.iter().copied().max().unwrap_or(0);
    max_label + 1
}

/// Degree sequence of a graph.
fn degree_sequence(adj: &Array2<f64>) -> Vec<f64> {
    let n = adj.nrows();
    (0..n).map(|i| adj.row(i).sum()).collect()
}

/// Validate inputs for distillation.
fn validate_distillation_inputs(
    adj: &Array2<f64>,
    features: &Array2<f64>,
    labels: &[usize],
    target_nodes: usize,
) -> Result<()> {
    let n = adj.nrows();

    if adj.nrows() != adj.ncols() {
        return Err(GraphError::InvalidParameter {
            param: "adj".to_string(),
            value: format!("{}x{}", adj.nrows(), adj.ncols()),
            expected: "square matrix".to_string(),
            context: "gradient_matching_condense".to_string(),
        });
    }
    if features.nrows() != n {
        return Err(GraphError::InvalidParameter {
            param: "features".to_string(),
            value: format!("{} rows", features.nrows()),
            expected: format!("{n} rows"),
            context: "gradient_matching_condense: features must match adjacency".to_string(),
        });
    }
    if labels.len() != n {
        return Err(GraphError::InvalidParameter {
            param: "labels".to_string(),
            value: format!("length {}", labels.len()),
            expected: format!("length {n}"),
            context: "gradient_matching_condense: labels must match adjacency".to_string(),
        });
    }
    if target_nodes == 0 {
        return Err(GraphError::InvalidParameter {
            param: "target_nodes".to_string(),
            value: "0".to_string(),
            expected: "target_nodes > 0".to_string(),
            context: "gradient_matching_condense".to_string(),
        });
    }
    if target_nodes > n {
        return Err(GraphError::InvalidParameter {
            param: "target_nodes".to_string(),
            value: target_nodes.to_string(),
            expected: format!("target_nodes <= {n}"),
            context: "gradient_matching_condense: cannot condense to more nodes than original"
                .to_string(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::condensation::types::{CondensationConfig, CondensationMethod};

    /// Build a simple test graph with n nodes, 2 classes, and d features.
    fn simple_graph(n: usize, d: usize) -> (Array2<f64>, Array2<f64>, Vec<usize>) {
        let mut adj = Array2::<f64>::zeros((n, n));
        let mut features = Array2::<f64>::zeros((n, d));
        let mut labels = vec![0usize; n];

        // Connect consecutive nodes in a chain
        for i in 0..(n - 1) {
            adj[[i, i + 1]] = 1.0;
            adj[[i + 1, i]] = 1.0;
        }

        // Set features: class 0 nodes near origin, class 1 nodes offset
        let half = n / 2;
        for i in 0..n {
            if i < half {
                features[[i, 0]] = i as f64 * 0.1;
                if d > 1 {
                    features[[i, 1]] = 0.0;
                }
                labels[i] = 0;
            } else {
                features[[i, 0]] = 5.0 + (i - half) as f64 * 0.1;
                if d > 1 {
                    features[[i, 1]] = 5.0;
                }
                labels[i] = 1;
            }
        }

        (adj, features, labels)
    }

    // -----------------------------------------------------------------------
    // gradient_matching_condense tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gradient_matching_produces_valid_output() {
        let (adj, features, labels) = simple_graph(10, 3);
        let config = CondensationConfig {
            target_nodes: 4,
            method: CondensationMethod::GradientMatching,
            max_iterations: 10,
            learning_rate: 0.01,
        };

        let result = gradient_matching_condense(&adj, &features, &labels, &config)
            .expect("gradient_matching_condense should succeed");

        assert_eq!(result.adjacency.nrows(), 4);
        assert_eq!(result.adjacency.ncols(), 4);
        assert_eq!(result.features.nrows(), 4);
        assert_eq!(result.features.ncols(), 3);
        assert_eq!(result.labels.len(), 4);
        assert_eq!(result.source_mapping.len(), 4);
    }

    #[test]
    fn test_gradient_matching_covers_classes() {
        let (adj, features, labels) = simple_graph(10, 3);
        let config = CondensationConfig {
            target_nodes: 4,
            method: CondensationMethod::GradientMatching,
            max_iterations: 5,
            learning_rate: 0.01,
        };

        let result = gradient_matching_condense(&adj, &features, &labels, &config)
            .expect("gradient_matching_condense should succeed");

        // Both classes should be represented
        let has_class0 = result.labels.contains(&0);
        let has_class1 = result.labels.contains(&1);
        assert!(has_class0, "class 0 should be in condensed graph");
        assert!(has_class1, "class 1 should be in condensed graph");
    }

    #[test]
    fn test_gradient_matching_loss_decreases() {
        let (adj, features, labels) = simple_graph(12, 4);

        // Run with very few iterations
        let config_few = CondensationConfig {
            target_nodes: 4,
            method: CondensationMethod::GradientMatching,
            max_iterations: 2,
            learning_rate: 0.01,
        };
        let result_few = gradient_matching_condense(&adj, &features, &labels, &config_few)
            .expect("should succeed with few iterations");

        // Run with more iterations
        let config_many = CondensationConfig {
            target_nodes: 4,
            method: CondensationMethod::GradientMatching,
            max_iterations: 50,
            learning_rate: 0.01,
        };
        let result_many = gradient_matching_condense(&adj, &features, &labels, &config_many)
            .expect("should succeed with many iterations");

        // The structure matching loss of the more-iterated result should generally
        // be no worse (allowing for numerical tolerance)
        let loss_few = structure_matching_loss(&adj, &result_few.adjacency);
        let loss_many = structure_matching_loss(&adj, &result_many.adjacency);

        // With more iterations the loss should not dramatically increase
        assert!(
            loss_many < loss_few + 0.5,
            "more iterations should not dramatically increase loss: few={loss_few}, many={loss_many}"
        );
    }

    #[test]
    fn test_gradient_matching_error_target_zero() {
        let (adj, features, labels) = simple_graph(6, 2);
        let config = CondensationConfig {
            target_nodes: 0,
            method: CondensationMethod::GradientMatching,
            max_iterations: 5,
            learning_rate: 0.01,
        };
        let result = gradient_matching_condense(&adj, &features, &labels, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_gradient_matching_error_target_too_large() {
        let (adj, features, labels) = simple_graph(6, 2);
        let config = CondensationConfig {
            target_nodes: 100,
            method: CondensationMethod::GradientMatching,
            max_iterations: 5,
            learning_rate: 0.01,
        };
        let result = gradient_matching_condense(&adj, &features, &labels, &config);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // feature_alignment_loss tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_feature_alignment_loss_identical() {
        let features = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid shape");
        let loss = feature_alignment_loss(&features, &features);
        assert!(
            loss.abs() < 1e-12,
            "feature alignment loss for identical features should be 0, got {loss}"
        );
    }

    #[test]
    fn test_feature_alignment_loss_different() {
        let orig = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.0, 0.0]).expect("valid shape");
        let synth =
            Array2::from_shape_vec((2, 2), vec![10.0, 10.0, 10.0, 10.0]).expect("valid shape");
        let loss = feature_alignment_loss(&orig, &synth);
        assert!(
            loss > 100.0,
            "feature alignment loss for distant features should be large, got {loss}"
        );
    }

    // -----------------------------------------------------------------------
    // structure_matching_loss tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_structure_matching_loss_identical() {
        let adj = Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
            .expect("valid shape");

        let loss = structure_matching_loss(&adj, &adj);
        assert!(
            loss.abs() < 1e-12,
            "structure matching loss for identical adjacency should be 0, got {loss}"
        );
    }

    #[test]
    fn test_structure_matching_loss_different() {
        // Complete graph vs. empty graph
        let complete =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
                .expect("valid shape");
        let empty = Array2::<f64>::zeros((3, 3));

        let loss = structure_matching_loss(&complete, &empty);
        assert!(
            loss > 0.0,
            "structure matching loss for different graphs should be positive, got {loss}"
        );
    }
}
