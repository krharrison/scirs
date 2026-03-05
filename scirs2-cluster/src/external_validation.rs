//! External cluster validation indices
//!
//! Metrics for comparing a clustering against ground-truth labels.
//!
//! # Indices
//!
//! - **Adjusted Rand Index (ARI)**: Chance-corrected measure of pairwise agreement
//! - **Normalized Mutual Information (NMI)**: Information-theoretic similarity
//! - **Fowlkes-Mallows Index (FMI)**: Geometric mean of precision and recall
//! - **V-measure**: Harmonic mean of homogeneity and completeness
//! - **Adjusted Mutual Information (AMI)**: Chance-corrected MI

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Contingency matrix builder
// ---------------------------------------------------------------------------

/// Build a contingency matrix from two label vectors.
///
/// Returns `(contingency, row_labels, col_labels)`.
fn contingency_matrix(
    labels_true: &[usize],
    labels_pred: &[usize],
) -> Result<(Array2<usize>, Vec<usize>, Vec<usize>)> {
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "label vectors must have equal length".into(),
        ));
    }
    if labels_true.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "label vectors must be non-empty".into(),
        ));
    }

    let rows: BTreeSet<usize> = labels_true.iter().copied().collect();
    let cols: BTreeSet<usize> = labels_pred.iter().copied().collect();
    let row_vec: Vec<usize> = rows.into_iter().collect();
    let col_vec: Vec<usize> = cols.into_iter().collect();

    let r = row_vec.len();
    let c = col_vec.len();

    let row_idx: BTreeMap<usize, usize> =
        row_vec.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let col_idx: BTreeMap<usize, usize> =
        col_vec.iter().enumerate().map(|(i, &v)| (v, i)).collect();

    let mut mat = Array2::<usize>::zeros((r, c));
    for (&t, &p) in labels_true.iter().zip(labels_pred.iter()) {
        let ri = row_idx[&t];
        let ci = col_idx[&p];
        mat[[ri, ci]] += 1;
    }

    Ok((mat, row_vec, col_vec))
}

fn comb2(n: usize) -> usize {
    if n < 2 {
        0
    } else {
        n * (n - 1) / 2
    }
}

// ---------------------------------------------------------------------------
// Adjusted Rand Index
// ---------------------------------------------------------------------------

/// Adjusted Rand Index (ARI).
///
/// Measures the similarity between two clusterings adjusted for chance.
/// Values range from -1 (worse than random) to 1 (perfect agreement).
/// Expected value for random labelling is 0.
pub fn adjusted_rand_index_ext<F>(labels_true: &[usize], labels_pred: &[usize]) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let (cont, _, _) = contingency_matrix(labels_true, labels_pred)?;
    let n = labels_true.len();
    let n2 = comb2(n);

    let sum_comb_nij: usize = cont.iter().map(|&v| comb2(v)).sum();
    let sum_comb_a: usize = cont.sum_axis(Axis(1)).iter().map(|&v| comb2(v)).sum();
    let sum_comb_b: usize = cont.sum_axis(Axis(0)).iter().map(|&v| comb2(v)).sum();

    let expected = F::from(sum_comb_a).ok_or_else(|| conv_err())?
        * F::from(sum_comb_b).ok_or_else(|| conv_err())?
        / F::from(n2).ok_or_else(|| conv_err())?;

    let max_index = (F::from(sum_comb_a).ok_or_else(|| conv_err())?
        + F::from(sum_comb_b).ok_or_else(|| conv_err())?)
        / from_f64::<F>(2.0)?;
    let index = F::from(sum_comb_nij).ok_or_else(|| conv_err())?;

    let denom = max_index - expected;
    if denom.abs() < F::epsilon() {
        return Ok(F::zero());
    }

    Ok((index - expected) / denom)
}

// ---------------------------------------------------------------------------
// Mutual Information helpers
// ---------------------------------------------------------------------------

fn entropy_from_counts<F: Float + FromPrimitive>(counts: &[usize], n: usize) -> Result<F> {
    let n_f = F::from(n).ok_or_else(|| conv_err())?;
    let mut h = F::zero();
    for &c in counts {
        if c > 0 {
            let p = F::from(c).ok_or_else(|| conv_err())? / n_f;
            h = h - p * p.ln();
        }
    }
    Ok(h)
}

fn mutual_information_from_cont<F: Float + FromPrimitive>(
    cont: &Array2<usize>,
    n: usize,
) -> Result<F> {
    let n_f = F::from(n).ok_or_else(|| conv_err())?;
    let row_sums = cont.sum_axis(Axis(1));
    let col_sums = cont.sum_axis(Axis(0));

    let mut mi = F::zero();
    for i in 0..cont.nrows() {
        for j in 0..cont.ncols() {
            let nij = cont[[i, j]];
            if nij == 0 {
                continue;
            }
            let nij_f = F::from(nij).ok_or_else(|| conv_err())?;
            let ni = F::from(row_sums[i]).ok_or_else(|| conv_err())?;
            let nj = F::from(col_sums[j]).ok_or_else(|| conv_err())?;
            mi = mi + nij_f / n_f * (nij_f * n_f / (ni * nj)).ln();
        }
    }
    Ok(mi)
}

// ---------------------------------------------------------------------------
// Normalized Mutual Information
// ---------------------------------------------------------------------------

/// Normalized Mutual Information (NMI).
///
/// Normalised by the arithmetic mean of the two entropies.
/// Values range from 0 (no mutual information) to 1 (perfect correlation).
pub fn normalized_mutual_info_ext<F>(labels_true: &[usize], labels_pred: &[usize]) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let (cont, _, _) = contingency_matrix(labels_true, labels_pred)?;
    let n = labels_true.len();

    let row_sums: Vec<usize> = cont.sum_axis(Axis(1)).to_vec();
    let col_sums: Vec<usize> = cont.sum_axis(Axis(0)).to_vec();

    let h_true: F = entropy_from_counts(&row_sums, n)?;
    let h_pred: F = entropy_from_counts(&col_sums, n)?;

    if h_true.abs() < F::epsilon() && h_pred.abs() < F::epsilon() {
        return Ok(F::one());
    }

    let mi: F = mutual_information_from_cont(&cont, n)?;
    let denom = (h_true + h_pred) / from_f64::<F>(2.0)?;
    if denom.abs() < F::epsilon() {
        return Ok(F::zero());
    }

    Ok(mi / denom)
}

// ---------------------------------------------------------------------------
// Fowlkes-Mallows Index
// ---------------------------------------------------------------------------

/// Fowlkes-Mallows Index.
///
/// The geometric mean of pairwise precision and recall.
/// Values range from 0 to 1, where 1 indicates perfect agreement.
pub fn fowlkes_mallows_index<F>(labels_true: &[usize], labels_pred: &[usize]) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let (cont, _, _) = contingency_matrix(labels_true, labels_pred)?;

    let tp: usize = cont.iter().map(|&v| comb2(v)).sum();

    let row_sums = cont.sum_axis(Axis(1));
    let col_sums = cont.sum_axis(Axis(0));

    let sum_comb_rows: usize = row_sums.iter().map(|&v| comb2(v)).sum();
    let sum_comb_cols: usize = col_sums.iter().map(|&v| comb2(v)).sum();

    if sum_comb_rows == 0 || sum_comb_cols == 0 {
        return Ok(F::zero());
    }

    let tp_f = F::from(tp).ok_or_else(|| conv_err())?;
    let p = tp_f / F::from(sum_comb_rows).ok_or_else(|| conv_err())?;
    let r = tp_f / F::from(sum_comb_cols).ok_or_else(|| conv_err())?;

    Ok((p * r).sqrt())
}

// ---------------------------------------------------------------------------
// V-measure (homogeneity + completeness)
// ---------------------------------------------------------------------------

/// V-measure result.
#[derive(Debug, Clone, Copy)]
pub struct VMeasureResult<F: Float> {
    /// Homogeneity score in [0, 1].
    pub homogeneity: F,
    /// Completeness score in [0, 1].
    pub completeness: F,
    /// V-measure (harmonic mean) in [0, 1].
    pub v_measure: F,
}

/// V-measure with configurable beta.
///
/// `beta > 1` weights completeness higher; `beta < 1` weights homogeneity higher.
/// When `beta = 1` this is the standard V-measure (harmonic mean).
pub fn v_measure_ext<F>(
    labels_true: &[usize],
    labels_pred: &[usize],
    beta: F,
) -> Result<VMeasureResult<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let (cont, _, _) = contingency_matrix(labels_true, labels_pred)?;
    let n = labels_true.len();
    let n_f = F::from(n).ok_or_else(|| conv_err())?;

    let row_sums: Vec<usize> = cont.sum_axis(Axis(1)).to_vec();
    let col_sums: Vec<usize> = cont.sum_axis(Axis(0)).to_vec();

    let h_true: F = entropy_from_counts(&row_sums, n)?;
    let h_pred: F = entropy_from_counts(&col_sums, n)?;

    // Conditional entropy H(C|K) -- classes given clusters
    let h_c_given_k: F = conditional_ent(&cont, &col_sums, n)?;
    // Conditional entropy H(K|C)
    let h_k_given_c: F = conditional_ent_transpose(&cont, &row_sums, n)?;

    let homogeneity = if h_true.abs() < F::epsilon() {
        F::one()
    } else {
        F::one() - h_c_given_k / h_true
    };

    let completeness = if h_pred.abs() < F::epsilon() {
        F::one()
    } else {
        F::one() - h_k_given_c / h_pred
    };

    let v_measure = if homogeneity + completeness <= F::zero() {
        F::zero()
    } else {
        let beta_sq = beta * beta;
        (F::one() + beta_sq) * homogeneity * completeness / (beta_sq * homogeneity + completeness)
    };

    Ok(VMeasureResult {
        homogeneity,
        completeness,
        v_measure,
    })
}

/// Standard V-measure (beta = 1).
pub fn v_measure_standard<F>(
    labels_true: &[usize],
    labels_pred: &[usize],
) -> Result<VMeasureResult<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    v_measure_ext(labels_true, labels_pred, F::one())
}

// ---------------------------------------------------------------------------
// Adjusted Mutual Information
// ---------------------------------------------------------------------------

/// Adjusted Mutual Information (AMI).
///
/// Mutual information adjusted for chance, using the expected MI under
/// a hypergeometric model.
pub fn adjusted_mutual_info_ext<F>(labels_true: &[usize], labels_pred: &[usize]) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let (cont, _, _) = contingency_matrix(labels_true, labels_pred)?;
    let n = labels_true.len();

    let row_sums: Vec<usize> = cont.sum_axis(Axis(1)).to_vec();
    let col_sums: Vec<usize> = cont.sum_axis(Axis(0)).to_vec();

    let h_true: F = entropy_from_counts(&row_sums, n)?;
    let h_pred: F = entropy_from_counts(&col_sums, n)?;

    let mi: F = mutual_information_from_cont(&cont, n)?;

    // Expected MI under the hypergeometric model (approximation for large n)
    let emi: F = expected_mutual_info(&row_sums, &col_sums, n)?;

    let max_h = h_true.max(h_pred);
    let denom = max_h - emi;

    if denom.abs() < F::epsilon() {
        return Ok(F::zero());
    }

    Ok((mi - emi) / denom)
}

/// Compute E[MI] under the hypergeometric model.
///
/// Uses the exact formula from Vinh, Epps & Bailey (2010), which sums
/// over all possible nij values.
fn expected_mutual_info<F: Float + FromPrimitive>(a: &[usize], b: &[usize], n: usize) -> Result<F> {
    let n_f = F::from(n).ok_or_else(|| conv_err())?;

    // Pre-compute log-factorials for the exact sum
    let max_val = n + 1;
    let log_fact = log_factorials::<F>(max_val)?;

    let mut emi = F::zero();

    for &ai in a {
        for &bj in b {
            let nij_min = if ai + bj > n { ai + bj - n } else { 0 };
            let nij_max = ai.min(bj);

            for nij in nij_min..=nij_max {
                if nij == 0 {
                    continue;
                }
                let nij_f = F::from(nij).ok_or_else(|| conv_err())?;

                // Term: nij/n * log(n * nij / (ai * bj))
                let ai_f = F::from(ai).ok_or_else(|| conv_err())?;
                let bj_f = F::from(bj).ok_or_else(|| conv_err())?;
                let log_term = (n_f * nij_f / (ai_f * bj_f)).ln();

                // Probability: hypergeometric pmf
                // P(nij) = C(ai, nij) * C(n-ai, bj-nij) / C(n, bj)
                let log_p = log_fact[ai] - log_fact[nij] - log_fact[ai - nij] + log_fact[n - ai]
                    - log_fact[bj - nij]
                    - log_fact[n - ai - (bj - nij)]
                    - log_fact[n]
                    + log_fact[bj]
                    + log_fact[n - bj];

                let p = log_p.exp();
                emi = emi + nij_f / n_f * log_term * p;
            }
        }
    }

    Ok(emi)
}

fn log_factorials<F: Float + FromPrimitive>(n: usize) -> Result<Vec<F>> {
    let mut lf = vec![F::zero(); n + 1];
    for i in 2..=n {
        lf[i] = lf[i - 1] + F::from(i).ok_or_else(|| conv_err())?.ln();
    }
    Ok(lf)
}

// ---------------------------------------------------------------------------
// Conditional entropy helpers
// ---------------------------------------------------------------------------

/// H(C|K) from contingency where rows = classes, cols = clusters
fn conditional_ent<F: Float + FromPrimitive>(
    cont: &Array2<usize>,
    col_sums: &[usize],
    n: usize,
) -> Result<F> {
    let n_f = F::from(n).ok_or_else(|| conv_err())?;
    let mut h = F::zero();
    for j in 0..cont.ncols() {
        let nj = col_sums[j];
        if nj == 0 {
            continue;
        }
        let nj_f = F::from(nj).ok_or_else(|| conv_err())?;
        for i in 0..cont.nrows() {
            let nij = cont[[i, j]];
            if nij == 0 {
                continue;
            }
            let nij_f = F::from(nij).ok_or_else(|| conv_err())?;
            h = h - nij_f / n_f * (nij_f / nj_f).ln();
        }
    }
    Ok(h)
}

/// H(K|C) from contingency transposed
fn conditional_ent_transpose<F: Float + FromPrimitive>(
    cont: &Array2<usize>,
    row_sums: &[usize],
    n: usize,
) -> Result<F> {
    let n_f = F::from(n).ok_or_else(|| conv_err())?;
    let mut h = F::zero();
    for i in 0..cont.nrows() {
        let ni = row_sums[i];
        if ni == 0 {
            continue;
        }
        let ni_f = F::from(ni).ok_or_else(|| conv_err())?;
        for j in 0..cont.ncols() {
            let nij = cont[[i, j]];
            if nij == 0 {
                continue;
            }
            let nij_f = F::from(nij).ok_or_else(|| conv_err())?;
            h = h - nij_f / n_f * (nij_f / ni_f).ln();
        }
    }
    Ok(h)
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn conv_err() -> ClusteringError {
    ClusteringError::ComputationError("float conversion failed".into())
}

fn from_f64<F: Float + FromPrimitive>(v: f64) -> Result<F> {
    F::from(v).ok_or_else(|| conv_err())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Perfect agreement
    fn perfect() -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 1, 1, 2, 2], vec![0, 0, 1, 1, 2, 2])
    }

    // Permuted labels (still perfect up to relabelling)
    fn permuted() -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 1, 1, 2, 2], vec![2, 2, 0, 0, 1, 1])
    }

    // Random-ish labels
    fn partial() -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 0, 1, 1, 1], vec![0, 0, 1, 1, 1, 0])
    }

    // All same
    fn trivial() -> (Vec<usize>, Vec<usize>) {
        (vec![0, 0, 0, 0], vec![0, 0, 0, 0])
    }

    #[test]
    fn test_ari_perfect() {
        let (t, p) = perfect();
        let ari: f64 = adjusted_rand_index_ext(&t, &p).expect("ari");
        assert!((ari - 1.0).abs() < 1e-10, "ARI perfect: {}", ari);
    }

    #[test]
    fn test_ari_permuted() {
        let (t, p) = permuted();
        let ari: f64 = adjusted_rand_index_ext(&t, &p).expect("ari");
        assert!((ari - 1.0).abs() < 1e-10, "ARI permuted: {}", ari);
    }

    #[test]
    fn test_ari_partial() {
        let (t, p) = partial();
        let ari: f64 = adjusted_rand_index_ext(&t, &p).expect("ari");
        assert!(ari > -1.0 && ari < 1.0, "ARI partial: {}", ari);
    }

    #[test]
    fn test_nmi_perfect() {
        let (t, p) = perfect();
        let nmi: f64 = normalized_mutual_info_ext(&t, &p).expect("nmi");
        assert!((nmi - 1.0).abs() < 1e-6, "NMI perfect: {}", nmi);
    }

    #[test]
    fn test_nmi_permuted() {
        let (t, p) = permuted();
        let nmi: f64 = normalized_mutual_info_ext(&t, &p).expect("nmi");
        assert!((nmi - 1.0).abs() < 1e-6, "NMI permuted: {}", nmi);
    }

    #[test]
    fn test_nmi_trivial() {
        let (t, p) = trivial();
        let nmi: f64 = normalized_mutual_info_ext(&t, &p).expect("nmi");
        assert!((nmi - 1.0).abs() < 1e-6, "NMI trivial: {}", nmi);
    }

    #[test]
    fn test_fowlkes_mallows_perfect() {
        let (t, p) = perfect();
        let fmi: f64 = fowlkes_mallows_index(&t, &p).expect("fmi");
        assert!((fmi - 1.0).abs() < 1e-10, "FMI perfect: {}", fmi);
    }

    #[test]
    fn test_fowlkes_mallows_permuted() {
        let (t, p) = permuted();
        let fmi: f64 = fowlkes_mallows_index(&t, &p).expect("fmi");
        assert!((fmi - 1.0).abs() < 1e-10, "FMI permuted: {}", fmi);
    }

    #[test]
    fn test_fowlkes_mallows_partial() {
        let (t, p) = partial();
        let fmi: f64 = fowlkes_mallows_index(&t, &p).expect("fmi");
        assert!(fmi >= 0.0 && fmi <= 1.0, "FMI in range: {}", fmi);
    }

    #[test]
    fn test_v_measure_perfect() {
        let (t, p) = perfect();
        let v: VMeasureResult<f64> = v_measure_standard(&t, &p).expect("v_measure");
        assert!((v.homogeneity - 1.0).abs() < 1e-6, "H: {}", v.homogeneity);
        assert!((v.completeness - 1.0).abs() < 1e-6, "C: {}", v.completeness);
        assert!((v.v_measure - 1.0).abs() < 1e-6, "V: {}", v.v_measure);
    }

    #[test]
    fn test_v_measure_partial() {
        let (t, p) = partial();
        let v: VMeasureResult<f64> = v_measure_standard(&t, &p).expect("v_measure");
        assert!(v.homogeneity >= 0.0 && v.homogeneity <= 1.0);
        assert!(v.completeness >= 0.0 && v.completeness <= 1.0);
        assert!(v.v_measure >= 0.0 && v.v_measure <= 1.0);
    }

    #[test]
    fn test_v_measure_beta() {
        let (t, p) = partial();
        let v1: VMeasureResult<f64> = v_measure_ext(&t, &p, 0.5).expect("v beta 0.5");
        let v2: VMeasureResult<f64> = v_measure_ext(&t, &p, 2.0).expect("v beta 2.0");
        // Both should be valid
        assert!(v1.v_measure >= 0.0 && v1.v_measure <= 1.0);
        assert!(v2.v_measure >= 0.0 && v2.v_measure <= 1.0);
    }

    #[test]
    fn test_ami_perfect() {
        let (t, p) = perfect();
        let ami: f64 = adjusted_mutual_info_ext(&t, &p).expect("ami");
        assert!((ami - 1.0).abs() < 1e-4, "AMI perfect: {}", ami);
    }

    #[test]
    fn test_ami_permuted() {
        let (t, p) = permuted();
        let ami: f64 = adjusted_mutual_info_ext(&t, &p).expect("ami");
        assert!((ami - 1.0).abs() < 1e-4, "AMI permuted: {}", ami);
    }

    #[test]
    fn test_ami_partial() {
        let (t, p) = partial();
        let ami: f64 = adjusted_mutual_info_ext(&t, &p).expect("ami");
        assert!(ami >= -1.0 && ami <= 1.0, "AMI partial in range: {}", ami);
    }

    #[test]
    fn test_contingency_matrix_basic() {
        let t = vec![0, 0, 1, 1];
        let p = vec![0, 0, 1, 1];
        let (cont, rows, cols) = contingency_matrix(&t, &p).expect("cont");
        assert_eq!(cont.nrows(), 2);
        assert_eq!(cont.ncols(), 2);
        assert_eq!(cont[[0, 0]], 2);
        assert_eq!(cont[[1, 1]], 2);
    }

    #[test]
    fn test_contingency_matrix_error_mismatch() {
        let t = vec![0, 1];
        let p = vec![0, 1, 2];
        assert!(contingency_matrix(&t, &p).is_err());
    }

    #[test]
    fn test_contingency_matrix_error_empty() {
        let t: Vec<usize> = vec![];
        let p: Vec<usize> = vec![];
        assert!(contingency_matrix(&t, &p).is_err());
    }

    #[test]
    fn test_ari_error_mismatch() {
        let t = vec![0, 1];
        let p = vec![0, 1, 2];
        assert!(adjusted_rand_index_ext::<f64>(&t, &p).is_err());
    }

    #[test]
    fn test_nmi_range() {
        // NMI is always in [0, 1]
        let (t, p) = partial();
        let nmi: f64 = normalized_mutual_info_ext(&t, &p).expect("nmi");
        assert!(nmi >= 0.0 && nmi <= 1.0 + 1e-10, "NMI range: {}", nmi);
    }
}
