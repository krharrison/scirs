//! Information Retrieval ranking metrics with simple slice-based API.
//!
//! These functions accept plain Rust slices rather than ndarray arrays,
//! providing a simpler interface for common IR evaluation tasks.

use crate::error::{MetricsError, Result};

/// Computes Average Precision for a single query.
///
/// AP = sum_k P(k) * rel(k) / n_relevant
///
/// where P(k) is precision at position k and rel(k) is 1 if the item at
/// position k is relevant after ranking by `y_score` in descending order.
///
/// # Arguments
/// * `y_true` — relevance labels (true if relevant)
/// * `y_score` — predicted relevance scores (higher = more relevant)
///
/// # Returns
/// Average Precision in [0, 1].
pub fn average_precision(y_true: &[bool], y_score: &[f64]) -> Result<f64> {
    if y_true.len() != y_score.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true ({}) and y_score ({}) have different lengths",
            y_true.len(),
            y_score.len()
        )));
    }
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty input arrays".to_string(),
        ));
    }

    let n_relevant = y_true.iter().filter(|&&r| r).count();
    if n_relevant == 0 {
        return Ok(0.0);
    }

    // Sort indices by score descending
    let mut indices: Vec<usize> = (0..y_true.len()).collect();
    indices.sort_by(|&a, &b| {
        y_score[b]
            .partial_cmp(&y_score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ap = 0.0_f64;
    let mut n_hits = 0_usize;
    for (rank, &idx) in indices.iter().enumerate() {
        if y_true[idx] {
            n_hits += 1;
            ap += n_hits as f64 / (rank + 1) as f64;
        }
    }
    Ok(ap / n_relevant as f64)
}

/// Mean Average Precision over a collection of queries.
///
/// # Arguments
/// * `queries` — slice of `(y_true, y_score)` pairs, one per query
///
/// # Returns
/// MAP in [0, 1].
pub fn mean_average_precision(queries: &[(Vec<bool>, Vec<f64>)]) -> Result<f64> {
    if queries.is_empty() {
        return Err(MetricsError::InvalidInput(
            "No queries provided".to_string(),
        ));
    }
    let mut sum = 0.0_f64;
    for (y_true, y_score) in queries {
        sum += average_precision(y_true, y_score)?;
    }
    Ok(sum / queries.len() as f64)
}

/// Normalized Discounted Cumulative Gain at k.
///
/// DCG@k = sum_{i=1}^{k} (2^rel_i - 1) / log2(i+1)
/// NDCG@k = DCG@k / IDCG@k
///
/// # Arguments
/// * `y_true` — graded relevance scores (non-negative)
/// * `y_score` — predicted scores for ranking
/// * `k` — cutoff depth (0 means use all items)
///
/// # Returns
/// NDCG in [0, 1].
pub fn ndcg_at_k(y_true: &[f64], y_score: &[f64], k: usize) -> Result<f64> {
    if y_true.len() != y_score.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true ({}) and y_score ({}) have different lengths",
            y_true.len(),
            y_score.len()
        )));
    }
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty input arrays".to_string(),
        ));
    }

    let n = y_true.len();
    let cutoff = if k == 0 || k > n { n } else { k };

    // Sort by predicted score descending → DCG
    let mut pred_indices: Vec<usize> = (0..n).collect();
    pred_indices.sort_by(|&a, &b| {
        y_score[b]
            .partial_cmp(&y_score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let dcg: f64 = pred_indices[..cutoff]
        .iter()
        .enumerate()
        .map(|(i, &idx)| (2.0_f64.powf(y_true[idx]) - 1.0) / (i as f64 + 2.0).log2())
        .sum();

    // Sort by true relevance descending → IDCG
    let mut ideal_indices: Vec<usize> = (0..n).collect();
    ideal_indices.sort_by(|&a, &b| {
        y_true[b]
            .partial_cmp(&y_true[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let idcg: f64 = ideal_indices[..cutoff]
        .iter()
        .enumerate()
        .map(|(i, &idx)| (2.0_f64.powf(y_true[idx]) - 1.0) / (i as f64 + 2.0).log2())
        .sum();

    if idcg == 0.0 {
        return Ok(0.0);
    }
    Ok(dcg / idcg)
}

/// Reciprocal Rank: 1/rank of the first relevant document.
///
/// Documents are sorted by `y_score` in descending order. If no relevant
/// document is found, returns 0.
pub fn reciprocal_rank(y_true: &[bool], y_score: &[f64]) -> Result<f64> {
    if y_true.len() != y_score.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true ({}) and y_score ({}) have different lengths",
            y_true.len(),
            y_score.len()
        )));
    }
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty input arrays".to_string(),
        ));
    }

    let mut indices: Vec<usize> = (0..y_true.len()).collect();
    indices.sort_by(|&a, &b| {
        y_score[b]
            .partial_cmp(&y_score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (rank, &idx) in indices.iter().enumerate() {
        if y_true[idx] {
            return Ok(1.0 / (rank + 1) as f64);
        }
    }
    Ok(0.0)
}

/// Mean Reciprocal Rank over a collection of queries.
pub fn mean_reciprocal_rank(queries: &[(Vec<bool>, Vec<f64>)]) -> Result<f64> {
    if queries.is_empty() {
        return Err(MetricsError::InvalidInput(
            "No queries provided".to_string(),
        ));
    }
    let mut sum = 0.0_f64;
    for (y_true, y_score) in queries {
        sum += reciprocal_rank(y_true, y_score)?;
    }
    Ok(sum / queries.len() as f64)
}

/// Kendall's Tau rank correlation coefficient.
///
/// Tau = (concordant - discordant) / C(n, 2)
///
/// # Arguments
/// * `rankings1` — permutation of 0..n (rank of each item in list 1)
/// * `rankings2` — permutation of 0..n (rank of each item in list 2)
pub fn kendall_tau(rankings1: &[usize], rankings2: &[usize]) -> Result<f64> {
    if rankings1.len() != rankings2.len() {
        return Err(MetricsError::InvalidInput(format!(
            "rankings1 ({}) and rankings2 ({}) have different lengths",
            rankings1.len(),
            rankings2.len()
        )));
    }
    let n = rankings1.len();
    if n < 2 {
        return Err(MetricsError::InvalidInput(
            "Need at least 2 items to compute Kendall's Tau".to_string(),
        ));
    }

    let mut concordant = 0_i64;
    let mut discordant = 0_i64;

    for i in 0..n {
        for j in (i + 1)..n {
            let sign1 = (rankings1[j] as i64) - (rankings1[i] as i64);
            let sign2 = (rankings2[j] as i64) - (rankings2[i] as i64);
            let product = sign1 * sign2;
            if product > 0 {
                concordant += 1;
            } else if product < 0 {
                discordant += 1;
            }
            // ties contribute 0
        }
    }

    let total = n as f64 * (n as f64 - 1.0) / 2.0;
    Ok((concordant - discordant) as f64 / total)
}

/// Spearman's Footrule distance (normalized).
///
/// Normalized footrule = sum |pi1(i) - pi2(i)| / (n^2 / 2)
///
/// # Arguments
/// * `rankings1` — position of each item in ranking 1 (0-indexed)
/// * `rankings2` — position of each item in ranking 2 (0-indexed)
pub fn spearman_footrule(rankings1: &[usize], rankings2: &[usize]) -> Result<f64> {
    if rankings1.len() != rankings2.len() {
        return Err(MetricsError::InvalidInput(format!(
            "rankings1 ({}) and rankings2 ({}) have different lengths",
            rankings1.len(),
            rankings2.len()
        )));
    }
    let n = rankings1.len();
    if n == 0 {
        return Err(MetricsError::InvalidInput(
            "Empty rankings provided".to_string(),
        ));
    }

    let raw_sum: f64 = rankings1
        .iter()
        .zip(rankings2.iter())
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .sum();

    // Max footrule distance for n items is n^2/2 (for n even) or (n^2-1)/2 (odd)
    // Use n^2/2 as the normalization factor as per the specification
    let normalizer = (n * n) as f64 / 2.0;
    Ok(raw_sum / normalizer)
}

/// Precision at k.
///
/// Fraction of top-k results that are relevant.
///
/// # Arguments
/// * `y_true` — relevance labels
/// * `y_score` — predicted scores
/// * `k` — cutoff depth
pub fn precision_at_k(y_true: &[bool], y_score: &[f64], k: usize) -> Result<f64> {
    if y_true.len() != y_score.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true ({}) and y_score ({}) have different lengths",
            y_true.len(),
            y_score.len()
        )));
    }
    if k == 0 {
        return Err(MetricsError::InvalidInput(
            "k must be greater than 0".to_string(),
        ));
    }
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty input arrays".to_string(),
        ));
    }

    let mut indices: Vec<usize> = (0..y_true.len()).collect();
    indices.sort_by(|&a, &b| {
        y_score[b]
            .partial_cmp(&y_score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let cutoff = k.min(indices.len());
    let hits = indices[..cutoff].iter().filter(|&&idx| y_true[idx]).count();
    Ok(hits as f64 / cutoff as f64)
}

/// Recall at k.
///
/// Fraction of all relevant documents that appear in the top-k results.
///
/// # Arguments
/// * `y_true` — relevance labels
/// * `y_score` — predicted scores
/// * `k` — cutoff depth
pub fn recall_at_k(y_true: &[bool], y_score: &[f64], k: usize) -> Result<f64> {
    if y_true.len() != y_score.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true ({}) and y_score ({}) have different lengths",
            y_true.len(),
            y_score.len()
        )));
    }
    if k == 0 {
        return Err(MetricsError::InvalidInput(
            "k must be greater than 0".to_string(),
        ));
    }
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty input arrays".to_string(),
        ));
    }

    let n_relevant = y_true.iter().filter(|&&r| r).count();
    if n_relevant == 0 {
        return Ok(0.0);
    }

    let mut indices: Vec<usize> = (0..y_true.len()).collect();
    indices.sort_by(|&a, &b| {
        y_score[b]
            .partial_cmp(&y_score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let cutoff = k.min(indices.len());
    let hits = indices[..cutoff].iter().filter(|&&idx| y_true[idx]).count();
    Ok(hits as f64 / n_relevant as f64)
}

/// R-Precision: precision at R, where R is the number of relevant documents.
///
/// This equals the fraction of relevant documents in the top-R results.
///
/// # Arguments
/// * `y_true` — relevance labels
/// * `y_score` — predicted scores
pub fn r_precision(y_true: &[bool], y_score: &[f64]) -> Result<f64> {
    if y_true.len() != y_score.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true ({}) and y_score ({}) have different lengths",
            y_true.len(),
            y_score.len()
        )));
    }
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty input arrays".to_string(),
        ));
    }

    let n_relevant = y_true.iter().filter(|&&r| r).count();
    if n_relevant == 0 {
        return Ok(0.0);
    }

    precision_at_k(y_true, y_score, n_relevant)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_average_precision_perfect() {
        let y_true = vec![true, false, true, false];
        let y_score = vec![0.9, 0.4, 0.8, 0.2];
        let ap = average_precision(&y_true, &y_score).expect("should succeed");
        // Top order: idx0(0.9,rel), idx2(0.8,rel), idx1(0.4,not), idx3(0.2,not)
        // P@1=1/1, P@2=2/2 → AP = (1.0 + 1.0) / 2 = 1.0
        assert!((ap - 1.0).abs() < 1e-10, "AP should be 1.0, got {ap}");
    }

    #[test]
    fn test_average_precision_worst() {
        let y_true = vec![true, true, false, false];
        let y_score = vec![0.1, 0.2, 0.9, 0.8];
        let ap = average_precision(&y_true, &y_score).expect("should succeed");
        // Sorted: idx2, idx3, idx1, idx0 — relevant at positions 3 and 4
        // P@3 = 1/3, P@4 = 2/4 → AP = (1/3 + 1/2) / 2 = (5/6) / 2 ≈ 0.4167
        assert!(ap < 0.5, "AP should be low, got {ap}");
    }

    #[test]
    fn test_mean_average_precision() {
        let queries = vec![
            (vec![true, false, true], vec![0.9, 0.3, 0.8]),
            (vec![false, true, false], vec![0.1, 0.9, 0.2]),
        ];
        let map = mean_average_precision(&queries).expect("should succeed");
        // Query 1: AP = 1.0; Query 2: AP = 1.0 (relevant is at rank 1)
        assert!((map - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_at_k_perfect() {
        let y_true = vec![3.0, 2.0, 1.0, 0.0];
        let y_score = vec![0.9, 0.8, 0.5, 0.1];
        let ndcg = ndcg_at_k(&y_true, &y_score, 4).expect("should succeed");
        // Predicted order equals ideal order → NDCG = 1.0
        assert!((ndcg - 1.0).abs() < 1e-10, "NDCG should be 1.0, got {ndcg}");
    }

    #[test]
    fn test_ndcg_at_k_cutoff() {
        let y_true = vec![3.0, 0.0, 2.0, 1.0];
        let y_score = vec![0.9, 0.3, 0.8, 0.5];
        let ndcg = ndcg_at_k(&y_true, &y_score, 2).expect("should succeed");
        // top-2 predicted: idx0(3), idx2(2) → same as ideal → NDCG@2=1.0
        assert!((ndcg - 1.0).abs() < 1e-10, "NDCG@2 should be 1.0, got {ndcg}");
    }

    #[test]
    fn test_reciprocal_rank_first() {
        let y_true = vec![true, false, false];
        let y_score = vec![0.9, 0.5, 0.3];
        let rr = reciprocal_rank(&y_true, &y_score).expect("should succeed");
        assert!((rr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reciprocal_rank_second() {
        let y_true = vec![false, true, false];
        let y_score = vec![0.9, 0.5, 0.3];
        let rr = reciprocal_rank(&y_true, &y_score).expect("should succeed");
        // Sorted: idx0(not), idx1(rel) → rank 2 → RR = 0.5
        assert!((rr - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mean_reciprocal_rank() {
        let queries = vec![
            (vec![true, false], vec![0.9, 0.4]),
            (vec![false, true], vec![0.9, 0.4]),
        ];
        let mrr = mean_reciprocal_rank(&queries).expect("should succeed");
        // Q1: RR=1.0, Q2: RR=0.5 → MRR=0.75
        assert!((mrr - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_kendall_tau_identical() {
        let r1 = vec![0, 1, 2, 3];
        let r2 = vec![0, 1, 2, 3];
        let tau = kendall_tau(&r1, &r2).expect("should succeed");
        assert!((tau - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kendall_tau_reversed() {
        let r1 = vec![0, 1, 2, 3];
        let r2 = vec![3, 2, 1, 0];
        let tau = kendall_tau(&r1, &r2).expect("should succeed");
        assert!((tau - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_spearman_footrule_identical() {
        let r1 = vec![0, 1, 2, 3];
        let r2 = vec![0, 1, 2, 3];
        let dist = spearman_footrule(&r1, &r2).expect("should succeed");
        assert!((dist - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_at_k() {
        let y_true = vec![true, false, true, false, true];
        let y_score = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let p = precision_at_k(&y_true, &y_score, 3).expect("should succeed");
        // Top-3: idx0(rel), idx1(not), idx2(rel) → 2/3
        assert!((p - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_recall_at_k() {
        let y_true = vec![true, false, true, false, true];
        let y_score = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let r = recall_at_k(&y_true, &y_score, 3).expect("should succeed");
        // Top-3: idx0(rel), idx1(not), idx2(rel) → 2 hits / 3 total relevant
        assert!((r - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_r_precision() {
        // 2 relevant docs; top-2 both relevant → R-precision = 1.0
        let y_true = vec![true, false, true, false];
        let y_score = vec![0.9, 0.4, 0.8, 0.2];
        let rp = r_precision(&y_true, &y_score).expect("should succeed");
        assert!((rp - 1.0).abs() < 1e-10);
    }
}
