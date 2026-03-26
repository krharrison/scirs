//! Evaluation metrics for graph condensation quality.
//!
//! Provides quantitative measures of how well a condensed graph
//! preserves properties of the original graph:
//!
//! - **Degree distribution distance**: KL divergence between degree histograms.
//! - **Spectral distance**: L2 distance between top-k eigenvalues.
//! - **Label coverage**: Fraction of original label classes represented.

use scirs2_core::ndarray::Array2;

use super::types::QualityMetrics;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// KL divergence between the degree distributions of two graphs.
///
/// The degree distribution is computed as a normalised histogram.
/// A small epsilon is added to avoid log(0). Returns 0.0 if both
/// distributions are identical.
///
/// # Arguments
/// * `orig_adj`      - Adjacency matrix of the original graph.
/// * `condensed_adj` - Adjacency matrix of the condensed graph.
pub fn degree_distribution_distance(orig_adj: &Array2<f64>, condensed_adj: &Array2<f64>) -> f64 {
    let orig_degs = degree_histogram(orig_adj);
    let cond_degs = degree_histogram(condensed_adj);

    kl_divergence(&orig_degs, &cond_degs)
}

/// L2 distance between the top-k eigenvalues of the graph Laplacians.
///
/// Eigenvalues are approximated using the power-iteration-based approach
/// on the Laplacian `L = D - A`. Only the top `k = min(n1, n2, 10)`
/// eigenvalues are compared.
///
/// Returns 0.0 if both graphs have identical spectral properties.
///
/// # Arguments
/// * `orig_adj`      - Adjacency matrix of the original graph.
/// * `condensed_adj` - Adjacency matrix of the condensed graph.
pub fn spectral_distance(orig_adj: &Array2<f64>, condensed_adj: &Array2<f64>) -> f64 {
    let orig_eigs = approximate_eigenvalues(orig_adj);
    let cond_eigs = approximate_eigenvalues(condensed_adj);

    let k = orig_eigs.len().min(cond_eigs.len());
    if k == 0 {
        return 0.0;
    }

    let mut dist_sq = 0.0;
    for i in 0..k {
        let diff = orig_eigs[i] - cond_eigs[i];
        dist_sq += diff * diff;
    }

    dist_sq.sqrt()
}

/// Fraction of original label classes that are present in the condensed graph.
///
/// Returns 1.0 if every label class from the original graph appears in
/// the condensed graph, 0.0 if none do.
///
/// # Arguments
/// * `orig_labels`      - Labels of the original graph nodes.
/// * `condensed_labels` - Labels of the condensed graph nodes.
pub fn label_coverage(orig_labels: &[usize], condensed_labels: &[usize]) -> f64 {
    if orig_labels.is_empty() {
        return 1.0;
    }

    let orig_classes: std::collections::HashSet<usize> = orig_labels.iter().copied().collect();
    let cond_classes: std::collections::HashSet<usize> = condensed_labels.iter().copied().collect();

    if orig_classes.is_empty() {
        return 1.0;
    }

    let covered = orig_classes.intersection(&cond_classes).count();
    covered as f64 / orig_classes.len() as f64
}

/// Evaluate the quality of a condensation by computing all metrics.
///
/// # Arguments
/// * `orig_adj`         - Original adjacency matrix.
/// * `orig_labels`      - Original node labels.
/// * `condensed_adj`    - Condensed adjacency matrix.
/// * `condensed_labels` - Condensed node labels.
pub fn evaluate_condensation(
    orig_adj: &Array2<f64>,
    orig_labels: &[usize],
    condensed_adj: &Array2<f64>,
    condensed_labels: &[usize],
) -> QualityMetrics {
    QualityMetrics {
        degree_distribution_distance: degree_distribution_distance(orig_adj, condensed_adj),
        spectral_distance: spectral_distance(orig_adj, condensed_adj),
        label_coverage: label_coverage(orig_labels, condensed_labels),
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute a normalised degree histogram.
///
/// Returns a vector of probabilities where index `d` is the fraction of
/// nodes with degree `d`. The histogram is padded to a common length.
fn degree_histogram(adj: &Array2<f64>) -> Vec<f64> {
    let n = adj.nrows();
    if n == 0 {
        return vec![1.0]; // trivial distribution
    }

    let degrees: Vec<usize> = (0..n)
        .map(|i| {
            let mut deg = 0usize;
            for j in 0..adj.ncols() {
                if adj[[i, j]].abs() > 1e-12 && i != j {
                    deg += 1;
                }
            }
            deg
        })
        .collect();

    let max_deg = degrees.iter().copied().max().unwrap_or(0);
    let mut hist = vec![0.0f64; max_deg + 1];
    for &d in &degrees {
        hist[d] += 1.0;
    }

    // Normalise
    let total: f64 = hist.iter().sum();
    if total > 0.0 {
        for h in &mut hist {
            *h /= total;
        }
    }

    hist
}

/// KL divergence D_KL(P || Q) with smoothing.
fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    let eps = 1e-10;
    let max_len = p.len().max(q.len());

    let mut kl = 0.0;
    for i in 0..max_len {
        let pi = p.get(i).copied().unwrap_or(0.0) + eps;
        let qi = q.get(i).copied().unwrap_or(0.0) + eps;

        // Renormalise locally is not needed because we add eps to both
        kl += pi * (pi / qi).ln();
    }

    // Subtract the contribution of pure eps-eps terms to get a cleaner value
    // (they contribute eps * ln(1) = 0, but due to asymmetric padding
    //  we just clamp to non-negative)
    kl.max(0.0)
}

/// Approximate the top eigenvalues of the graph Laplacian using power iteration.
///
/// Returns up to `min(n, 10)` eigenvalues sorted in descending order.
fn approximate_eigenvalues(adj: &Array2<f64>) -> Vec<f64> {
    let n = adj.nrows();
    if n == 0 {
        return Vec::new();
    }

    // Construct Laplacian L = D - A
    let mut laplacian = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let deg: f64 = adj.row(i).sum();
        laplacian[[i, i]] = deg;
        for j in 0..n {
            laplacian[[i, j]] -= adj[[i, j]];
        }
    }

    // Power iteration with deflation for top-k eigenvalues
    let k = n.min(10);
    let mut eigenvalues = Vec::with_capacity(k);
    let mut work_matrix = laplacian;

    for _ in 0..k {
        let eig = power_iteration(&work_matrix, n, 100);
        eigenvalues.push(eig.0);

        // Deflate: M = M - lambda * v * v^T
        let lambda = eig.0;
        let v = &eig.1;
        for i in 0..n {
            for j in 0..n {
                work_matrix[[i, j]] -= lambda * v[i] * v[j];
            }
        }
    }

    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

/// Power iteration to find the dominant eigenvalue and eigenvector.
///
/// Returns `(eigenvalue, eigenvector)`.
fn power_iteration(matrix: &Array2<f64>, n: usize, max_iter: usize) -> (f64, Vec<f64>) {
    if n == 0 {
        return (0.0, Vec::new());
    }

    // Initialise with a vector that is not orthogonal to the dominant eigenvector
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    // Break symmetry
    for i in 0..n {
        v[i] += (i as f64) * 1e-6;
    }
    normalise_vector(&mut v);

    let mut eigenvalue = 0.0;

    for _ in 0..max_iter {
        // w = M * v
        let mut w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += matrix[[i, j]] * v[j];
            }
        }

        // Rayleigh quotient
        let mut dot_wv = 0.0;
        let mut dot_vv = 0.0;
        for i in 0..n {
            dot_wv += w[i] * v[i];
            dot_vv += v[i] * v[i];
        }
        eigenvalue = if dot_vv.abs() > 1e-15 {
            dot_wv / dot_vv
        } else {
            0.0
        };

        normalise_vector(&mut w);
        v = w;
    }

    (eigenvalue, v)
}

/// Normalise a vector to unit length in place.
fn normalise_vector(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small triangle graph (3 nodes, fully connected).
    fn triangle_graph() -> Array2<f64> {
        Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
            .expect("valid shape for triangle graph")
    }

    /// Build a chain graph: 0-1-2-3.
    fn chain_graph(n: usize) -> Array2<f64> {
        let mut adj = Array2::<f64>::zeros((n, n));
        for i in 0..(n - 1) {
            adj[[i, i + 1]] = 1.0;
            adj[[i + 1, i]] = 1.0;
        }
        adj
    }

    // -----------------------------------------------------------------------
    // degree_distribution_distance tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_degree_distribution_distance_identical() {
        let adj = triangle_graph();
        let dist = degree_distribution_distance(&adj, &adj);
        assert!(
            dist.abs() < 1e-8,
            "degree distribution distance of identical graphs should be ~0, got {dist}"
        );
    }

    #[test]
    fn test_degree_distribution_distance_different() {
        let complete = triangle_graph();
        let chain = chain_graph(3);

        let dist = degree_distribution_distance(&complete, &chain);
        assert!(
            dist > 0.0,
            "degree distribution distance of different graphs should be positive, got {dist}"
        );
    }

    #[test]
    fn test_degree_distribution_distance_empty_vs_nonempty() {
        let empty = Array2::<f64>::zeros((3, 3));
        let chain = chain_graph(3);

        let dist = degree_distribution_distance(&empty, &chain);
        assert!(
            dist > 0.0,
            "empty vs chain should have positive distance, got {dist}"
        );
    }

    // -----------------------------------------------------------------------
    // spectral_distance tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_spectral_distance_identical() {
        let adj = triangle_graph();
        let dist = spectral_distance(&adj, &adj);
        assert!(
            dist.abs() < 1e-6,
            "spectral distance of identical graphs should be ~0, got {dist}"
        );
    }

    #[test]
    fn test_spectral_distance_different_structure() {
        let complete = triangle_graph();
        let chain = chain_graph(3);

        let dist = spectral_distance(&complete, &chain);
        assert!(
            dist > 0.0,
            "spectral distance of structurally different graphs should be positive, got {dist}"
        );
    }

    #[test]
    fn test_spectral_distance_empty_graph() {
        let empty = Array2::<f64>::zeros((3, 3));
        let dist = spectral_distance(&empty, &empty);
        assert!(
            dist.abs() < 1e-6,
            "spectral distance of two empty graphs should be ~0, got {dist}"
        );
    }

    // -----------------------------------------------------------------------
    // label_coverage tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_label_coverage_all_present() {
        let orig = [0, 1, 2, 0, 1, 2];
        let cond = [2, 0, 1];
        let cov = label_coverage(&orig, &cond);
        assert!(
            (cov - 1.0).abs() < 1e-12,
            "all labels present => coverage should be 1.0, got {cov}"
        );
    }

    #[test]
    fn test_label_coverage_partial() {
        let orig = [0, 1, 2];
        let cond = [0, 1]; // missing class 2
        let cov = label_coverage(&orig, &cond);
        assert!(
            (cov - 2.0 / 3.0).abs() < 1e-12,
            "2 of 3 labels present => coverage should be ~0.667, got {cov}"
        );
    }

    #[test]
    fn test_label_coverage_none_present() {
        let orig = [0, 1, 2];
        let cond = [3, 4]; // completely disjoint labels
        let cov = label_coverage(&orig, &cond);
        assert!(
            cov.abs() < 1e-12,
            "no labels present => coverage should be 0.0, got {cov}"
        );
    }

    #[test]
    fn test_label_coverage_empty_original() {
        let orig: [usize; 0] = [];
        let cond = [0, 1];
        let cov = label_coverage(&orig, &cond);
        assert!(
            (cov - 1.0).abs() < 1e-12,
            "empty original => coverage should be 1.0, got {cov}"
        );
    }

    // -----------------------------------------------------------------------
    // evaluate_condensation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_evaluate_condensation_identical() {
        let adj = triangle_graph();
        let labels = [0, 1, 2];

        let metrics = evaluate_condensation(&adj, &labels, &adj, &labels);

        assert!(
            metrics.degree_distribution_distance.abs() < 1e-8,
            "degree dist should be ~0 for identical graphs"
        );
        assert!(
            metrics.spectral_distance.abs() < 1e-6,
            "spectral dist should be ~0 for identical graphs"
        );
        assert!(
            (metrics.label_coverage - 1.0).abs() < 1e-12,
            "label coverage should be 1.0 for identical labels"
        );
    }

    #[test]
    fn test_evaluate_condensation_returns_valid_metrics() {
        let orig = chain_graph(6);
        let orig_labels = [0, 0, 1, 1, 2, 2];

        let condensed = chain_graph(3);
        let cond_labels = [0, 1, 2];

        let metrics = evaluate_condensation(&orig, &orig_labels, &condensed, &cond_labels);

        assert!(
            metrics.degree_distribution_distance >= 0.0,
            "degree distribution distance should be non-negative"
        );
        assert!(
            metrics.spectral_distance >= 0.0,
            "spectral distance should be non-negative"
        );
        assert!(
            (0.0..=1.0).contains(&metrics.label_coverage),
            "label coverage should be in [0, 1], got {}",
            metrics.label_coverage
        );
        assert!(
            (metrics.label_coverage - 1.0).abs() < 1e-12,
            "all 3 classes present => coverage = 1.0"
        );
    }
}
