//! Comprehensive cluster evaluation metrics
//!
//! This module provides an extended set of clustering evaluation metrics beyond
//! the basics already in `metrics/`. These include external (ground-truth
//! required), internal, and graph-based measures.
//!
//! # Metrics
//!
//! - **Purity**: Fraction of correctly assigned points (max overlap with true classes)
//! - **F-measure**: Harmonic mean of precision/recall per cluster and overall
//! - **Completeness / Homogeneity (entropy-based)**: How well each cluster (class) maps to a single class (cluster)
//! - **Variation of Information (VI)**: Information-theoretic distance between clusterings
//! - **Normalized Cut**: Graph-based cut cost normalised by cluster volumes
//! - **Modularity**: Newman-Girvan modularity for graph-based clustering

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Contingency matrix helper
// ---------------------------------------------------------------------------

/// Build a contingency matrix from true and predicted labels.
///
/// Returns (matrix, true_labels_sorted, pred_labels_sorted).
fn contingency_matrix(
    true_labels: &[i32],
    pred_labels: &[i32],
) -> Result<(Vec<Vec<usize>>, Vec<i32>, Vec<i32>)> {
    if true_labels.len() != pred_labels.len() {
        return Err(ClusteringError::InvalidInput(
            "true_labels and pred_labels must have the same length".into(),
        ));
    }
    let n = true_labels.len();
    if n == 0 {
        return Err(ClusteringError::InvalidInput("Empty labels".into()));
    }

    // Collect unique labels (sorted)
    let mut true_set: Vec<i32> = true_labels.iter().copied().collect();
    true_set.sort();
    true_set.dedup();
    let mut pred_set: Vec<i32> = pred_labels.iter().copied().collect();
    pred_set.sort();
    pred_set.dedup();

    let true_idx: HashMap<i32, usize> = true_set.iter().enumerate().map(|(i, &l)| (l, i)).collect();
    let pred_idx: HashMap<i32, usize> = pred_set.iter().enumerate().map(|(i, &l)| (l, i)).collect();

    let n_true = true_set.len();
    let n_pred = pred_set.len();
    let mut mat = vec![vec![0usize; n_pred]; n_true];

    for i in 0..n {
        if let (Some(&ti), Some(&pi)) =
            (true_idx.get(&true_labels[i]), pred_idx.get(&pred_labels[i]))
        {
            mat[ti][pi] += 1;
        }
    }

    Ok((mat, true_set, pred_set))
}

// ---------------------------------------------------------------------------
// Purity
// ---------------------------------------------------------------------------

/// Compute clustering purity.
///
/// Purity measures what fraction of the total data points are correctly
/// classified when each cluster is assigned to its majority true class.
///
/// purity = (1/n) * sum_k max_j |c_k ∩ t_j|
///
/// Range: [0, 1]. Higher is better but biased toward many clusters (k = n → purity = 1).
pub fn purity(true_labels: &[i32], pred_labels: &[i32]) -> Result<f64> {
    let (mat, _, _) = contingency_matrix(true_labels, pred_labels)?;
    let n = true_labels.len();

    let sum_max: usize = (0..mat[0].len())
        .map(|j| (0..mat.len()).map(|i| mat[i][j]).max().unwrap_or(0))
        .sum();

    Ok(sum_max as f64 / n as f64)
}

// ---------------------------------------------------------------------------
// F-measure
// ---------------------------------------------------------------------------

/// Per-cluster and overall F-measure result.
#[derive(Debug, Clone)]
pub struct FMeasureResult {
    /// Per-cluster F1 scores (indexed by predicted cluster label).
    pub per_cluster_f1: Vec<f64>,
    /// Overall (weighted) F-measure.
    pub overall_f1: f64,
    /// Per-cluster precision.
    pub per_cluster_precision: Vec<f64>,
    /// Per-cluster recall.
    pub per_cluster_recall: Vec<f64>,
}

/// Compute the F-measure (F1) for clustering.
///
/// For each predicted cluster, finds the best-matching true class and
/// computes precision and recall. The overall F1 is a weighted average.
pub fn f_measure(true_labels: &[i32], pred_labels: &[i32]) -> Result<FMeasureResult> {
    let (mat, true_set, pred_set) = contingency_matrix(true_labels, pred_labels)?;
    let n = true_labels.len() as f64;
    let n_pred = pred_set.len();
    let n_true = true_set.len();

    // Column sums (per predicted cluster size)
    let pred_sizes: Vec<usize> = (0..n_pred)
        .map(|j| (0..n_true).map(|i| mat[i][j]).sum())
        .collect();

    // Row sums (per true class size)
    let true_sizes: Vec<usize> = (0..n_true)
        .map(|i| (0..n_pred).map(|j| mat[i][j]).sum())
        .collect();

    let mut per_cluster_f1 = vec![0.0f64; n_pred];
    let mut per_cluster_precision = vec![0.0f64; n_pred];
    let mut per_cluster_recall = vec![0.0f64; n_pred];

    for j in 0..n_pred {
        let pred_size = pred_sizes[j] as f64;
        if pred_size == 0.0 {
            continue;
        }

        // Find best matching true class
        let mut best_f1 = 0.0f64;
        let mut best_prec = 0.0f64;
        let mut best_rec = 0.0f64;

        for i in 0..n_true {
            let overlap = mat[i][j] as f64;
            let true_size = true_sizes[i] as f64;
            if true_size == 0.0 || pred_size == 0.0 {
                continue;
            }

            let precision = overlap / pred_size;
            let recall = overlap / true_size;
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            if f1 > best_f1 {
                best_f1 = f1;
                best_prec = precision;
                best_rec = recall;
            }
        }

        per_cluster_f1[j] = best_f1;
        per_cluster_precision[j] = best_prec;
        per_cluster_recall[j] = best_rec;
    }

    // Weighted average F1
    let overall_f1 = if n > 0.0 {
        (0..n_pred)
            .map(|j| per_cluster_f1[j] * pred_sizes[j] as f64)
            .sum::<f64>()
            / n
    } else {
        0.0
    };

    Ok(FMeasureResult {
        per_cluster_f1,
        overall_f1,
        per_cluster_precision,
        per_cluster_recall,
    })
}

// ---------------------------------------------------------------------------
// Entropy-based: Completeness & Homogeneity
// ---------------------------------------------------------------------------

/// Result of entropy-based completeness and homogeneity computation.
#[derive(Debug, Clone)]
pub struct EntropyMetrics {
    /// Homogeneity: each cluster contains only members of a single class.
    pub homogeneity: f64,
    /// Completeness: all members of a class are assigned to the same cluster.
    pub completeness: f64,
    /// V-measure: harmonic mean of homogeneity and completeness.
    pub v_measure: f64,
    /// Entropy of the true labels H(C).
    pub entropy_true: f64,
    /// Entropy of the predicted labels H(K).
    pub entropy_pred: f64,
    /// Conditional entropy H(C|K).
    pub conditional_entropy_ck: f64,
    /// Conditional entropy H(K|C).
    pub conditional_entropy_kc: f64,
}

/// Compute entropy-based homogeneity, completeness, and V-measure.
///
/// - Homogeneity = 1 - H(C|K) / H(C)
/// - Completeness = 1 - H(K|C) / H(K)
/// - V-measure = harmonic mean of homogeneity and completeness
///
/// Where H(C|K) is the conditional entropy of classes given clusters.
pub fn entropy_metrics(true_labels: &[i32], pred_labels: &[i32]) -> Result<EntropyMetrics> {
    let (mat, _, _) = contingency_matrix(true_labels, pred_labels)?;
    let n = true_labels.len() as f64;
    let n_true = mat.len();
    let n_pred = if n_true > 0 { mat[0].len() } else { 0 };

    // Marginals
    let true_sums: Vec<f64> = (0..n_true)
        .map(|i| (0..n_pred).map(|j| mat[i][j] as f64).sum())
        .collect();
    let pred_sums: Vec<f64> = (0..n_pred)
        .map(|j| (0..n_true).map(|i| mat[i][j] as f64).sum())
        .collect();

    // H(C) - entropy of true labels
    let entropy_true = entropy_from_counts(&true_sums, n);

    // H(K) - entropy of predicted labels
    let entropy_pred = entropy_from_counts(&pred_sums, n);

    // H(C|K) - conditional entropy of classes given clusters
    let mut h_ck = 0.0f64;
    for j in 0..n_pred {
        if pred_sums[j] == 0.0 {
            continue;
        }
        for i in 0..n_true {
            let nij = mat[i][j] as f64;
            if nij > 0.0 {
                h_ck -= nij / n * (nij / pred_sums[j]).ln();
            }
        }
    }

    // H(K|C) - conditional entropy of clusters given classes
    let mut h_kc = 0.0f64;
    for i in 0..n_true {
        if true_sums[i] == 0.0 {
            continue;
        }
        for j in 0..n_pred {
            let nij = mat[i][j] as f64;
            if nij > 0.0 {
                h_kc -= nij / n * (nij / true_sums[i]).ln();
            }
        }
    }

    let homogeneity = if entropy_true.abs() < 1e-15 {
        1.0
    } else {
        1.0 - h_ck / entropy_true
    };

    let completeness = if entropy_pred.abs() < 1e-15 {
        1.0
    } else {
        1.0 - h_kc / entropy_pred
    };

    let v_measure = if homogeneity + completeness > 0.0 {
        2.0 * homogeneity * completeness / (homogeneity + completeness)
    } else {
        0.0
    };

    Ok(EntropyMetrics {
        homogeneity,
        completeness,
        v_measure,
        entropy_true,
        entropy_pred,
        conditional_entropy_ck: h_ck,
        conditional_entropy_kc: h_kc,
    })
}

/// Shannon entropy from a vector of counts.
fn entropy_from_counts(counts: &[f64], total: f64) -> f64 {
    if total <= 0.0 {
        return 0.0;
    }
    let mut h = 0.0f64;
    for &c in counts {
        if c > 0.0 {
            let p = c / total;
            h -= p * p.ln();
        }
    }
    h
}

// ---------------------------------------------------------------------------
// Variation of Information (VI)
// ---------------------------------------------------------------------------

/// Compute Variation of Information between two clusterings.
///
/// VI(U, V) = H(U|V) + H(V|U) = H(U) + H(V) - 2 * I(U;V)
///
/// Lower is better; VI = 0 when clusterings are identical.
/// Returns (vi, normalised_vi) where normalised_vi = VI / log(n).
pub fn variation_of_information(labels_u: &[i32], labels_v: &[i32]) -> Result<(f64, f64)> {
    let (mat, _, _) = contingency_matrix(labels_u, labels_v)?;
    let n = labels_u.len() as f64;
    let n_u = mat.len();
    let n_v = if n_u > 0 { mat[0].len() } else { 0 };

    let u_sums: Vec<f64> = (0..n_u)
        .map(|i| (0..n_v).map(|j| mat[i][j] as f64).sum())
        .collect();
    let v_sums: Vec<f64> = (0..n_v)
        .map(|j| (0..n_u).map(|i| mat[i][j] as f64).sum())
        .collect();

    // H(U|V)
    let mut h_uv = 0.0f64;
    for j in 0..n_v {
        if v_sums[j] == 0.0 {
            continue;
        }
        for i in 0..n_u {
            let nij = mat[i][j] as f64;
            if nij > 0.0 {
                h_uv -= (nij / n) * (nij / v_sums[j]).ln();
            }
        }
    }

    // H(V|U)
    let mut h_vu = 0.0f64;
    for i in 0..n_u {
        if u_sums[i] == 0.0 {
            continue;
        }
        for j in 0..n_v {
            let nij = mat[i][j] as f64;
            if nij > 0.0 {
                h_vu -= (nij / n) * (nij / u_sums[i]).ln();
            }
        }
    }

    let vi = h_uv + h_vu;
    let normalised = if n > 1.0 { vi / n.ln() } else { 0.0 };

    Ok((vi, normalised))
}

// ---------------------------------------------------------------------------
// Normalized Cut
// ---------------------------------------------------------------------------

/// Compute the normalised cut of a clustering given an affinity/weight matrix.
///
/// NCut = sum_k [ cut(C_k, complement) / vol(C_k) ]
///
/// where cut is the sum of edge weights crossing the boundary and vol is the
/// sum of all edge weights incident to vertices in the cluster.
///
/// Lower values indicate better clustering.
pub fn normalized_cut<F: Float + FromPrimitive + Debug>(
    affinity: ArrayView2<F>,
    labels: &[i32],
) -> Result<f64> {
    let n = affinity.shape()[0];
    if n != affinity.shape()[1] {
        return Err(ClusteringError::InvalidInput(
            "Affinity matrix must be square".into(),
        ));
    }
    if labels.len() != n {
        return Err(ClusteringError::InvalidInput(
            "labels length must match affinity dimension".into(),
        ));
    }

    // Find unique clusters
    let mut cluster_set: Vec<i32> = labels.iter().copied().filter(|&l| l >= 0).collect();
    cluster_set.sort();
    cluster_set.dedup();

    if cluster_set.is_empty() {
        return Ok(0.0);
    }

    let mut ncut = 0.0f64;

    for &ci in &cluster_set {
        let in_cluster: Vec<usize> = (0..n).filter(|&i| labels[i] == ci).collect();
        let out_cluster: Vec<usize> = (0..n).filter(|&i| labels[i] != ci).collect();

        // cut(C_k, complement)
        let mut cut_val = 0.0f64;
        for &i in &in_cluster {
            for &j in &out_cluster {
                cut_val += affinity[[i, j]].to_f64().unwrap_or(0.0);
            }
        }

        // vol(C_k) = sum of all edge weights incident to C_k
        let mut vol = 0.0f64;
        for &i in &in_cluster {
            for j in 0..n {
                vol += affinity[[i, j]].to_f64().unwrap_or(0.0);
            }
        }

        if vol > 1e-15 {
            ncut += cut_val / vol;
        }
    }

    Ok(ncut)
}

// ---------------------------------------------------------------------------
// Modularity
// ---------------------------------------------------------------------------

/// Compute Newman-Girvan modularity for a graph clustering.
///
/// Q = (1/2m) * sum_ij [ A_ij - k_i*k_j/(2m) ] * delta(c_i, c_j)
///
/// where A is the adjacency matrix, k_i is the degree of node i, m is the
/// total number of edges, and delta checks if i and j are in the same cluster.
///
/// Range: [-0.5, 1]. Higher is better; > 0.3 is considered significant structure.
pub fn modularity<F: Float + FromPrimitive + Debug>(
    adjacency: ArrayView2<F>,
    labels: &[i32],
) -> Result<f64> {
    let n = adjacency.shape()[0];
    if n != adjacency.shape()[1] {
        return Err(ClusteringError::InvalidInput(
            "Adjacency matrix must be square".into(),
        ));
    }
    if labels.len() != n {
        return Err(ClusteringError::InvalidInput(
            "labels length must match adjacency dimension".into(),
        ));
    }

    // Compute degrees and total weight
    let mut degrees = vec![0.0f64; n];
    let mut two_m = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let w = adjacency[[i, j]].to_f64().unwrap_or(0.0);
            degrees[i] += w;
            two_m += w;
        }
    }

    if two_m.abs() < 1e-15 {
        return Ok(0.0);
    }

    let mut q = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            if labels[i] == labels[j] && labels[i] >= 0 {
                let a_ij = adjacency[[i, j]].to_f64().unwrap_or(0.0);
                let expected = degrees[i] * degrees[j] / two_m;
                q += a_ij - expected;
            }
        }
    }
    q /= two_m;

    Ok(q)
}

// ---------------------------------------------------------------------------
// Aggregate quality report
// ---------------------------------------------------------------------------

/// Comprehensive clustering quality report.
#[derive(Debug, Clone)]
pub struct ClusterQualityReport {
    /// Purity score.
    pub purity: f64,
    /// F-measure result.
    pub f_measure: FMeasureResult,
    /// Entropy-based metrics (homogeneity, completeness, V-measure).
    pub entropy_metrics: EntropyMetrics,
    /// Variation of Information (raw, normalised).
    pub variation_of_information: (f64, f64),
}

/// Compute a comprehensive quality report for a clustering against ground truth.
pub fn cluster_quality_report(
    true_labels: &[i32],
    pred_labels: &[i32],
) -> Result<ClusterQualityReport> {
    let p = purity(true_labels, pred_labels)?;
    let fm = f_measure(true_labels, pred_labels)?;
    let em = entropy_metrics(true_labels, pred_labels)?;
    let vi = variation_of_information(true_labels, pred_labels)?;

    Ok(ClusterQualityReport {
        purity: p,
        f_measure: fm,
        entropy_metrics: em,
        variation_of_information: vi,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    // Perfect clustering: pred == true
    fn perfect_labels() -> (Vec<i32>, Vec<i32>) {
        let true_l = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let pred_l = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        (true_l, pred_l)
    }

    // Partially wrong clustering
    fn partial_labels() -> (Vec<i32>, Vec<i32>) {
        let true_l = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let pred_l = vec![0, 0, 1, 1, 1, 2, 2, 2, 0]; // 3 errors
        (true_l, pred_l)
    }

    // -- Contingency matrix tests --

    #[test]
    fn test_contingency_matrix() {
        let (t, p) = perfect_labels();
        let (mat, _, _) = contingency_matrix(&t, &p).expect("failed");
        // 3x3 diagonal
        assert_eq!(mat.len(), 3);
        assert_eq!(mat[0].len(), 3);
        assert_eq!(mat[0][0], 3);
        assert_eq!(mat[1][1], 3);
        assert_eq!(mat[2][2], 3);
    }

    #[test]
    fn test_contingency_mismatch_lengths() {
        let t = vec![0, 1];
        let p = vec![0];
        assert!(contingency_matrix(&t, &p).is_err());
    }

    #[test]
    fn test_contingency_empty() {
        let t: Vec<i32> = vec![];
        let p: Vec<i32> = vec![];
        assert!(contingency_matrix(&t, &p).is_err());
    }

    // -- Purity tests --

    #[test]
    fn test_purity_perfect() {
        let (t, p) = perfect_labels();
        let pu = purity(&t, &p).expect("failed");
        assert!((pu - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_purity_partial() {
        let (t, p) = partial_labels();
        let pu = purity(&t, &p).expect("failed");
        assert!(pu > 0.0 && pu < 1.0);
    }

    #[test]
    fn test_purity_all_same_cluster() {
        let t = vec![0, 0, 1, 1];
        let p = vec![0, 0, 0, 0];
        let pu = purity(&t, &p).expect("failed");
        assert!((pu - 0.5).abs() < 1e-10);
    }

    // -- F-measure tests --

    #[test]
    fn test_f_measure_perfect() {
        let (t, p) = perfect_labels();
        let fm = f_measure(&t, &p).expect("failed");
        assert!((fm.overall_f1 - 1.0).abs() < 1e-10);
        for &f in &fm.per_cluster_f1 {
            assert!((f - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_f_measure_partial() {
        let (t, p) = partial_labels();
        let fm = f_measure(&t, &p).expect("failed");
        assert!(fm.overall_f1 > 0.0 && fm.overall_f1 < 1.0);
    }

    #[test]
    fn test_f_measure_dimensions() {
        let (t, p) = partial_labels();
        let fm = f_measure(&t, &p).expect("failed");
        assert_eq!(fm.per_cluster_f1.len(), fm.per_cluster_precision.len());
        assert_eq!(fm.per_cluster_f1.len(), fm.per_cluster_recall.len());
    }

    // -- Entropy metrics tests --

    #[test]
    fn test_entropy_perfect() {
        let (t, p) = perfect_labels();
        let em = entropy_metrics(&t, &p).expect("failed");
        assert!((em.homogeneity - 1.0).abs() < 1e-10);
        assert!((em.completeness - 1.0).abs() < 1e-10);
        assert!((em.v_measure - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_partial() {
        let (t, p) = partial_labels();
        let em = entropy_metrics(&t, &p).expect("failed");
        assert!(em.homogeneity >= 0.0 && em.homogeneity <= 1.0);
        assert!(em.completeness >= 0.0 && em.completeness <= 1.0);
        assert!(em.v_measure >= 0.0 && em.v_measure <= 1.0);
    }

    #[test]
    fn test_entropy_single_cluster() {
        let t = vec![0, 0, 1, 1];
        let p = vec![0, 0, 0, 0]; // all in one cluster
        let em = entropy_metrics(&t, &p).expect("failed");
        // Completeness should be 1 (all members of each class in same cluster)
        assert!((em.completeness - 1.0).abs() < 1e-10);
        // Homogeneity should be < 1
        assert!(em.homogeneity < 1.0);
    }

    #[test]
    fn test_entropy_from_counts_fn() {
        let counts = vec![5.0, 5.0];
        let h = entropy_from_counts(&counts, 10.0);
        assert!((h - 2.0f64.ln()).abs() < 1e-10); // ln(2)
    }

    // -- Variation of Information tests --

    #[test]
    fn test_vi_identical() {
        let (t, p) = perfect_labels();
        let (vi, nvi) = variation_of_information(&t, &p).expect("failed");
        assert!(vi.abs() < 1e-10);
        assert!(nvi.abs() < 1e-10);
    }

    #[test]
    fn test_vi_different() {
        let (t, p) = partial_labels();
        let (vi, nvi) = variation_of_information(&t, &p).expect("failed");
        assert!(vi > 0.0);
        assert!(nvi >= 0.0);
    }

    #[test]
    fn test_vi_symmetric() {
        let (t, p) = partial_labels();
        let (vi1, _) = variation_of_information(&t, &p).expect("failed");
        let (vi2, _) = variation_of_information(&p, &t).expect("failed");
        assert!((vi1 - vi2).abs() < 1e-10);
    }

    // -- Normalized Cut tests --

    #[test]
    fn test_ncut_perfect_separation() {
        // Two disconnected components
        let affinity = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ],
        )
        .expect("shape");
        let labels = vec![0, 0, 1, 1];
        let nc = normalized_cut(affinity.view(), &labels).expect("failed");
        assert!(nc.abs() < 1e-10); // Perfect cut = 0
    }

    #[test]
    fn test_ncut_bad_cut() {
        // Fully connected graph, arbitrary split
        let affinity = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            ],
        )
        .expect("shape");
        let labels = vec![0, 0, 1, 1];
        let nc = normalized_cut(affinity.view(), &labels).expect("failed");
        assert!(nc > 0.0);
    }

    #[test]
    fn test_ncut_invalid_input() {
        let affinity = Array2::<f64>::zeros((3, 4));
        let labels = vec![0, 0, 0];
        assert!(normalized_cut(affinity.view(), &labels).is_err());
    }

    // -- Modularity tests --

    #[test]
    fn test_modularity_basic() {
        // Two-community graph
        let adj = Array2::from_shape_vec(
            (6, 6),
            vec![
                0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.5,
                0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 0.0,
            ],
        )
        .expect("shape");
        let labels = vec![0, 0, 0, 1, 1, 1];
        let q = modularity(adj.view(), &labels).expect("failed");
        assert!(
            q > 0.0,
            "Expected positive modularity for clear communities"
        );
    }

    #[test]
    fn test_modularity_single_cluster() {
        let adj = Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
            .expect("shape");
        let labels = vec![0, 0, 0];
        let q = modularity(adj.view(), &labels).expect("failed");
        // Single cluster modularity should be 0
        assert!(q.abs() < 1e-10);
    }

    #[test]
    fn test_modularity_invalid() {
        let adj = Array2::<f64>::zeros((3, 4));
        let labels = vec![0, 0, 0];
        assert!(modularity(adj.view(), &labels).is_err());
    }

    // -- Comprehensive report tests --

    #[test]
    fn test_quality_report_perfect() {
        let (t, p) = perfect_labels();
        let report = cluster_quality_report(&t, &p).expect("failed");
        assert!((report.purity - 1.0).abs() < 1e-10);
        assert!((report.f_measure.overall_f1 - 1.0).abs() < 1e-10);
        assert!((report.entropy_metrics.v_measure - 1.0).abs() < 1e-10);
        assert!(report.variation_of_information.0.abs() < 1e-10);
    }

    #[test]
    fn test_quality_report_partial() {
        let (t, p) = partial_labels();
        let report = cluster_quality_report(&t, &p).expect("failed");
        assert!(report.purity > 0.0);
        assert!(report.f_measure.overall_f1 > 0.0);
        assert!(report.entropy_metrics.v_measure > 0.0);
    }

    // -- Edge cases --

    #[test]
    fn test_two_elements() {
        let t = vec![0, 1];
        let p = vec![0, 1];
        let pu = purity(&t, &p).expect("failed");
        assert!((pu - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_element() {
        let t = vec![0];
        let p = vec![0];
        let pu = purity(&t, &p).expect("failed");
        assert!((pu - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_negative_labels() {
        // Noise label -1 should still work
        let t = vec![0, 0, 1, 1, -1];
        let p = vec![0, 0, 1, 1, -1];
        let pu = purity(&t, &p).expect("failed");
        assert!((pu - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_modularity_empty_graph() {
        let adj = Array2::<f64>::zeros((4, 4));
        let labels = vec![0, 0, 1, 1];
        let q = modularity(adj.view(), &labels).expect("failed");
        assert!(q.abs() < 1e-10);
    }
}
