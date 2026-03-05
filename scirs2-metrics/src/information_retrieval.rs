//! Information Retrieval Metrics
//!
//! Standalone functions for evaluating information retrieval systems.
//! These metrics measure the quality of ranked result lists returned
//! by search engines, recommendation systems, and similar IR systems.
//!
//! # Metrics
//!
//! - **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks of first relevant result
//! - **Normalized Discounted Cumulative Gain (NDCG)**: Measures ranking quality with position discount
//! - **Mean Average Precision at K (MAP@K)**: Mean of average precisions across queries
//! - **Precision@K**: Fraction of relevant documents in top-K results
//! - **Recall@K**: Fraction of total relevant documents found in top-K results
//! - **Hit Rate**: Fraction of queries where at least one relevant result appears in top-K
//!
//! # Examples
//!
//! ```
//! use scirs2_metrics::information_retrieval::{
//!     mean_reciprocal_rank, ndcg, precision_at_k, recall_at_k, hit_rate,
//! };
//!
//! // Binary relevance labels for 3 queries (1.0 = relevant, 0.0 = not)
//! let relevance = vec![
//!     vec![0.0, 1.0, 0.0, 0.0, 1.0],
//!     vec![1.0, 0.0, 0.0, 1.0, 0.0],
//!     vec![0.0, 0.0, 1.0, 0.0, 0.0],
//! ];
//!
//! let mrr = mean_reciprocal_rank(&relevance).expect("Failed");
//! let ndcg_val = ndcg(&relevance, 5).expect("Failed");
//! let p_at_3 = precision_at_k(&relevance, 3).expect("Failed");
//! let r_at_3 = recall_at_k(&relevance, 3).expect("Failed");
//! let hr = hit_rate(&relevance, 3).expect("Failed");
//! ```

use crate::error::{MetricsError, Result};

/// Computes Mean Reciprocal Rank (MRR) across multiple queries.
///
/// For each query, the reciprocal rank is 1/rank where rank is the position
/// of the first relevant document (1-indexed). If no relevant document exists,
/// the reciprocal rank is 0.
///
/// # Arguments
///
/// * `ranked_relevance` - For each query, a vector of relevance scores in ranked order.
///   A value > 0.0 indicates relevance.
///
/// # Returns
///
/// The mean reciprocal rank across all queries, in [0, 1].
///
/// # Errors
///
/// Returns error if the input is empty.
pub fn mean_reciprocal_rank(ranked_relevance: &[Vec<f64>]) -> Result<f64> {
    if ranked_relevance.is_empty() {
        return Err(MetricsError::InvalidInput(
            "ranked_relevance must not be empty".to_string(),
        ));
    }

    let mut rr_sum = 0.0;
    for query_rel in ranked_relevance {
        let mut found = false;
        for (pos, &rel) in query_rel.iter().enumerate() {
            if rel > 0.0 {
                rr_sum += 1.0 / (pos as f64 + 1.0);
                found = true;
                break;
            }
        }
        if !found {
            // No relevant document: reciprocal rank is 0
        }
    }

    Ok(rr_sum / ranked_relevance.len() as f64)
}

/// Computes Discounted Cumulative Gain (DCG) for a single ranked list at position k.
///
/// DCG = sum_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)
fn dcg_at_k(relevance: &[f64], k: usize) -> f64 {
    let limit = k.min(relevance.len());
    let mut dcg = 0.0;
    for i in 0..limit {
        let gain = (2.0_f64).powf(relevance[i]) - 1.0;
        let discount = (i as f64 + 2.0).log2(); // log2(i+2) since i is 0-indexed
        dcg += gain / discount;
    }
    dcg
}

/// Computes Normalized Discounted Cumulative Gain (NDCG) across multiple queries.
///
/// NDCG normalizes DCG by the ideal DCG (IDCG), which is the DCG obtained
/// with an ideal ranking (sorted by relevance in descending order).
///
/// # Arguments
///
/// * `ranked_relevance` - For each query, a vector of relevance scores in ranked order.
/// * `k` - The cut-off position. Only the top-k results are considered.
///
/// # Returns
///
/// The mean NDCG across all queries, in [0, 1].
///
/// # Errors
///
/// Returns error if input is empty or k is 0.
pub fn ndcg(ranked_relevance: &[Vec<f64>], k: usize) -> Result<f64> {
    if ranked_relevance.is_empty() {
        return Err(MetricsError::InvalidInput(
            "ranked_relevance must not be empty".to_string(),
        ));
    }
    if k == 0 {
        return Err(MetricsError::InvalidInput(
            "k must be greater than 0".to_string(),
        ));
    }

    let mut ndcg_sum = 0.0;
    let mut valid_queries = 0;

    for query_rel in ranked_relevance {
        let dcg = dcg_at_k(query_rel, k);

        // Compute ideal DCG: sort relevance in descending order
        let mut ideal_rel = query_rel.clone();
        ideal_rel.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let idcg = dcg_at_k(&ideal_rel, k);

        if idcg > 0.0 {
            ndcg_sum += dcg / idcg;
            valid_queries += 1;
        } else {
            // No relevant documents for this query; NDCG is defined as 0 or skipped
            // We include it as 0.0 for consistency
            valid_queries += 1;
        }
    }

    if valid_queries == 0 {
        return Ok(0.0);
    }

    Ok(ndcg_sum / valid_queries as f64)
}

/// Computes NDCG for a single query (not averaged).
///
/// This is useful when you want per-query NDCG values.
///
/// # Arguments
///
/// * `relevance` - Relevance scores in ranked order for a single query.
/// * `k` - The cut-off position.
///
/// # Returns
///
/// The NDCG value for the single query, in [0, 1].
pub fn ndcg_single(relevance: &[f64], k: usize) -> Result<f64> {
    if relevance.is_empty() {
        return Err(MetricsError::InvalidInput(
            "relevance must not be empty".to_string(),
        ));
    }
    if k == 0 {
        return Err(MetricsError::InvalidInput(
            "k must be greater than 0".to_string(),
        ));
    }

    let dcg = dcg_at_k(relevance, k);

    let mut ideal_rel: Vec<f64> = relevance.to_vec();
    ideal_rel.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg = dcg_at_k(&ideal_rel, k);

    if idcg <= 0.0 {
        return Ok(0.0);
    }

    Ok(dcg / idcg)
}

/// Computes Mean Average Precision at K (MAP@K) across multiple queries.
///
/// Average Precision (AP) is the average of precision values at each relevant
/// position in the ranked list. MAP is the mean of AP values across queries.
///
/// # Arguments
///
/// * `ranked_relevance` - For each query, binary relevance scores (>0 = relevant) in ranked order.
/// * `k` - The cut-off position. Only the top-k results are considered.
///
/// # Returns
///
/// The MAP@K value in [0, 1].
///
/// # Errors
///
/// Returns error if input is empty or k is 0.
pub fn map_at_k(ranked_relevance: &[Vec<f64>], k: usize) -> Result<f64> {
    if ranked_relevance.is_empty() {
        return Err(MetricsError::InvalidInput(
            "ranked_relevance must not be empty".to_string(),
        ));
    }
    if k == 0 {
        return Err(MetricsError::InvalidInput(
            "k must be greater than 0".to_string(),
        ));
    }

    let mut ap_sum = 0.0;

    for query_rel in ranked_relevance {
        let limit = k.min(query_rel.len());
        let mut relevant_count = 0.0;
        let mut precision_sum = 0.0;

        for i in 0..limit {
            if query_rel[i] > 0.0 {
                relevant_count += 1.0;
                precision_sum += relevant_count / (i as f64 + 1.0);
            }
        }

        // Total relevant documents (could be beyond k)
        let total_relevant: f64 = query_rel.iter().filter(|&&r| r > 0.0).count() as f64;

        if total_relevant > 0.0 {
            ap_sum += precision_sum / total_relevant;
        }
    }

    Ok(ap_sum / ranked_relevance.len() as f64)
}

/// Computes Mean Average Precision across all positions (no cutoff).
///
/// This is equivalent to MAP@K where K equals the list length.
///
/// # Arguments
///
/// * `ranked_relevance` - For each query, binary relevance scores in ranked order.
///
/// # Returns
///
/// The MAP value in [0, 1].
pub fn mean_average_precision(ranked_relevance: &[Vec<f64>]) -> Result<f64> {
    if ranked_relevance.is_empty() {
        return Err(MetricsError::InvalidInput(
            "ranked_relevance must not be empty".to_string(),
        ));
    }

    // Use the maximum list length as k
    let max_k = ranked_relevance.iter().map(|q| q.len()).max().unwrap_or(0);

    if max_k == 0 {
        return Err(MetricsError::InvalidInput(
            "All query result lists are empty".to_string(),
        ));
    }

    map_at_k(ranked_relevance, max_k)
}

/// Computes Precision@K averaged across multiple queries.
///
/// Precision@K is the fraction of relevant documents in the top-K results.
///
/// # Arguments
///
/// * `ranked_relevance` - For each query, binary relevance scores in ranked order.
/// * `k` - The cut-off position.
///
/// # Returns
///
/// The mean Precision@K in [0, 1].
///
/// # Errors
///
/// Returns error if input is empty or k is 0.
pub fn precision_at_k(ranked_relevance: &[Vec<f64>], k: usize) -> Result<f64> {
    if ranked_relevance.is_empty() {
        return Err(MetricsError::InvalidInput(
            "ranked_relevance must not be empty".to_string(),
        ));
    }
    if k == 0 {
        return Err(MetricsError::InvalidInput(
            "k must be greater than 0".to_string(),
        ));
    }

    let mut prec_sum = 0.0;

    for query_rel in ranked_relevance {
        let limit = k.min(query_rel.len());
        let relevant_in_top_k = query_rel[..limit].iter().filter(|&&r| r > 0.0).count();
        prec_sum += relevant_in_top_k as f64 / k as f64;
    }

    Ok(prec_sum / ranked_relevance.len() as f64)
}

/// Computes Recall@K averaged across multiple queries.
///
/// Recall@K is the fraction of total relevant documents found in the top-K results.
///
/// # Arguments
///
/// * `ranked_relevance` - For each query, binary relevance scores in ranked order.
///   The full list should contain all candidate documents (relevant and non-relevant).
/// * `k` - The cut-off position.
///
/// # Returns
///
/// The mean Recall@K in [0, 1].
///
/// # Errors
///
/// Returns error if input is empty or k is 0.
pub fn recall_at_k(ranked_relevance: &[Vec<f64>], k: usize) -> Result<f64> {
    if ranked_relevance.is_empty() {
        return Err(MetricsError::InvalidInput(
            "ranked_relevance must not be empty".to_string(),
        ));
    }
    if k == 0 {
        return Err(MetricsError::InvalidInput(
            "k must be greater than 0".to_string(),
        ));
    }

    let mut recall_sum = 0.0;
    let mut valid_queries = 0;

    for query_rel in ranked_relevance {
        let total_relevant: usize = query_rel.iter().filter(|&&r| r > 0.0).count();
        if total_relevant == 0 {
            // Skip queries with no relevant documents
            continue;
        }

        let limit = k.min(query_rel.len());
        let relevant_in_top_k = query_rel[..limit].iter().filter(|&&r| r > 0.0).count();
        recall_sum += relevant_in_top_k as f64 / total_relevant as f64;
        valid_queries += 1;
    }

    if valid_queries == 0 {
        return Ok(0.0);
    }

    Ok(recall_sum / valid_queries as f64)
}

/// Computes Hit Rate at K across multiple queries.
///
/// Hit Rate@K measures the fraction of queries for which at least one relevant
/// document appears in the top-K results.
///
/// # Arguments
///
/// * `ranked_relevance` - For each query, binary relevance scores in ranked order.
/// * `k` - The cut-off position.
///
/// # Returns
///
/// The hit rate in [0, 1].
///
/// # Errors
///
/// Returns error if input is empty or k is 0.
pub fn hit_rate(ranked_relevance: &[Vec<f64>], k: usize) -> Result<f64> {
    if ranked_relevance.is_empty() {
        return Err(MetricsError::InvalidInput(
            "ranked_relevance must not be empty".to_string(),
        ));
    }
    if k == 0 {
        return Err(MetricsError::InvalidInput(
            "k must be greater than 0".to_string(),
        ));
    }

    let mut hits = 0;

    for query_rel in ranked_relevance {
        let limit = k.min(query_rel.len());
        let has_relevant = query_rel[..limit].iter().any(|&r| r > 0.0);
        if has_relevant {
            hits += 1;
        }
    }

    Ok(hits as f64 / ranked_relevance.len() as f64)
}

/// Computes R-Precision for each query and returns the mean.
///
/// R-Precision is Precision@R where R is the number of relevant documents
/// for that query. It equals the fraction of relevant documents found in
/// the top-R positions.
///
/// # Arguments
///
/// * `ranked_relevance` - For each query, binary relevance scores in ranked order.
///
/// # Returns
///
/// The mean R-Precision in [0, 1].
pub fn r_precision(ranked_relevance: &[Vec<f64>]) -> Result<f64> {
    if ranked_relevance.is_empty() {
        return Err(MetricsError::InvalidInput(
            "ranked_relevance must not be empty".to_string(),
        ));
    }

    let mut rp_sum = 0.0;
    let mut valid_queries = 0;

    for query_rel in ranked_relevance {
        let total_relevant: usize = query_rel.iter().filter(|&&r| r > 0.0).count();
        if total_relevant == 0 {
            continue;
        }

        let limit = total_relevant.min(query_rel.len());
        let relevant_in_top_r = query_rel[..limit].iter().filter(|&&r| r > 0.0).count();
        rp_sum += relevant_in_top_r as f64 / total_relevant as f64;
        valid_queries += 1;
    }

    if valid_queries == 0 {
        return Ok(0.0);
    }

    Ok(rp_sum / valid_queries as f64)
}

/// Computes the F1@K score, the harmonic mean of Precision@K and Recall@K.
///
/// # Arguments
///
/// * `ranked_relevance` - For each query, binary relevance scores in ranked order.
/// * `k` - The cut-off position.
///
/// # Returns
///
/// The mean F1@K in [0, 1].
pub fn f1_at_k(ranked_relevance: &[Vec<f64>], k: usize) -> Result<f64> {
    let p = precision_at_k(ranked_relevance, k)?;
    let r = recall_at_k(ranked_relevance, k)?;

    if p + r <= 0.0 {
        return Ok(0.0);
    }

    Ok(2.0 * p * r / (p + r))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- MRR tests ----

    #[test]
    fn test_mrr_perfect_ranking() {
        // First result is always relevant
        let rel = vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
        let mrr = mean_reciprocal_rank(&rel).expect("should succeed");
        assert!((mrr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mrr_second_position() {
        // Relevant at position 2 for all queries
        let rel = vec![vec![0.0, 1.0, 0.0], vec![0.0, 1.0, 0.0]];
        let mrr = mean_reciprocal_rank(&rel).expect("should succeed");
        assert!((mrr - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mrr_no_relevant() {
        let rel = vec![vec![0.0, 0.0, 0.0]];
        let mrr = mean_reciprocal_rank(&rel).expect("should succeed");
        assert!((mrr - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mrr_mixed() {
        // Query 1: relevant at position 1 -> RR=1.0
        // Query 2: relevant at position 3 -> RR=1/3
        let rel = vec![vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]];
        let mrr = mean_reciprocal_rank(&rel).expect("should succeed");
        let expected = (1.0 + 1.0 / 3.0) / 2.0;
        assert!((mrr - expected).abs() < 1e-10);
    }

    #[test]
    fn test_mrr_empty_input() {
        let rel: Vec<Vec<f64>> = vec![];
        assert!(mean_reciprocal_rank(&rel).is_err());
    }

    // ---- NDCG tests ----

    #[test]
    fn test_ndcg_perfect_binary() {
        // All relevant docs at top
        let rel = vec![vec![1.0, 1.0, 0.0, 0.0]];
        let val = ndcg(&rel, 4).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_reversed_binary() {
        // Relevant docs at bottom -> lower NDCG
        let rel = vec![vec![0.0, 0.0, 1.0, 1.0]];
        let val = ndcg(&rel, 4).expect("should succeed");
        assert!(val < 1.0);
        assert!(val > 0.0);
    }

    #[test]
    fn test_ndcg_graded_relevance() {
        // Perfect ranking for graded relevance: 3, 2, 1, 0
        let rel = vec![vec![3.0, 2.0, 1.0, 0.0]];
        let val = ndcg(&rel, 4).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_single_perfect() {
        let val = ndcg_single(&[3.0, 2.0, 1.0, 0.0], 4).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_no_relevant() {
        let rel = vec![vec![0.0, 0.0, 0.0]];
        let val = ndcg(&rel, 3).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_k_zero() {
        let rel = vec![vec![1.0, 0.0]];
        assert!(ndcg(&rel, 0).is_err());
    }

    // ---- MAP@K tests ----

    #[test]
    fn test_map_at_k_perfect() {
        // Two relevant docs at positions 1 and 2
        let rel = vec![vec![1.0, 1.0, 0.0, 0.0]];
        let val = map_at_k(&rel, 4).expect("should succeed");
        // AP = (1/1 + 2/2) / 2 = 1.0
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_map_at_k_spread() {
        // Relevant at positions 1 and 3
        let rel = vec![vec![1.0, 0.0, 1.0, 0.0]];
        let val = map_at_k(&rel, 4).expect("should succeed");
        // AP = (1/1 + 2/3) / 2 = (1.0 + 0.6667) / 2 = 0.8333
        assert!((val - 5.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_map_at_k_no_relevant() {
        let rel = vec![vec![0.0, 0.0, 0.0]];
        let val = map_at_k(&rel, 3).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_map_at_k_cutoff() {
        // Relevant at position 4, but k=2
        let rel = vec![vec![0.0, 0.0, 0.0, 1.0]];
        let val = map_at_k(&rel, 2).expect("should succeed");
        // No relevant in top-2, but total_relevant=1, so AP=0/1=0
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_map_at_k_empty_input() {
        let rel: Vec<Vec<f64>> = vec![];
        assert!(map_at_k(&rel, 3).is_err());
    }

    #[test]
    fn test_mean_average_precision() {
        let rel = vec![vec![1.0, 0.0, 1.0, 0.0], vec![0.0, 1.0, 0.0, 1.0]];
        let val = mean_average_precision(&rel).expect("should succeed");
        assert!(val > 0.0 && val <= 1.0);
    }

    // ---- Precision@K tests ----

    #[test]
    fn test_precision_at_k_all_relevant() {
        let rel = vec![vec![1.0, 1.0, 1.0]];
        let val = precision_at_k(&rel, 3).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_at_k_none_relevant() {
        let rel = vec![vec![0.0, 0.0, 0.0]];
        let val = precision_at_k(&rel, 3).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_at_k_partial() {
        // 2 out of 3 relevant in top-3
        let rel = vec![vec![1.0, 0.0, 1.0, 0.0]];
        let val = precision_at_k(&rel, 3).expect("should succeed");
        assert!((val - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_at_k_larger_than_list() {
        // k=5 but only 3 results
        let rel = vec![vec![1.0, 1.0, 0.0]];
        let val = precision_at_k(&rel, 5).expect("should succeed");
        // 2 relevant out of 5 (k)
        assert!((val - 2.0 / 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_at_k_multiple_queries() {
        let rel = vec![
            vec![1.0, 1.0, 0.0], // P@2 = 2/2 = 1.0
            vec![0.0, 0.0, 1.0], // P@2 = 0/2 = 0.0
        ];
        let val = precision_at_k(&rel, 2).expect("should succeed");
        assert!((val - 0.5).abs() < 1e-10);
    }

    // ---- Recall@K tests ----

    #[test]
    fn test_recall_at_k_all_found() {
        // 2 relevant total, both in top-3
        let rel = vec![vec![1.0, 0.0, 1.0, 0.0]];
        let val = recall_at_k(&rel, 3).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_recall_at_k_partial() {
        // 3 relevant total, 1 in top-2
        let rel = vec![vec![1.0, 0.0, 1.0, 1.0]];
        let val = recall_at_k(&rel, 2).expect("should succeed");
        assert!((val - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_recall_at_k_none_found() {
        // 2 relevant, but none in top-2
        let rel = vec![vec![0.0, 0.0, 1.0, 1.0]];
        let val = recall_at_k(&rel, 2).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_recall_at_k_no_relevant_docs() {
        // No relevant documents at all
        let rel = vec![vec![0.0, 0.0, 0.0]];
        let val = recall_at_k(&rel, 3).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_recall_at_k_multiple_queries() {
        let rel = vec![
            vec![1.0, 0.0, 1.0], // 2 relevant, 1 in top-1 -> recall=0.5
            vec![0.0, 1.0, 0.0], // 1 relevant, 0 in top-1 -> recall=0.0
        ];
        let val = recall_at_k(&rel, 1).expect("should succeed");
        assert!((val - 0.25).abs() < 1e-10);
    }

    // ---- Hit Rate tests ----

    #[test]
    fn test_hit_rate_all_hit() {
        let rel = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let val = hit_rate(&rel, 2).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hit_rate_no_hit() {
        let rel = vec![vec![0.0, 0.0, 1.0], vec![0.0, 0.0, 1.0]];
        let val = hit_rate(&rel, 2).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hit_rate_partial() {
        let rel = vec![
            vec![1.0, 0.0, 0.0], // hit in top-1
            vec![0.0, 0.0, 1.0], // no hit in top-1
        ];
        let val = hit_rate(&rel, 1).expect("should succeed");
        assert!((val - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hit_rate_k_larger_than_list() {
        let rel = vec![vec![0.0, 1.0]];
        let val = hit_rate(&rel, 10).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hit_rate_empty() {
        let rel: Vec<Vec<f64>> = vec![];
        assert!(hit_rate(&rel, 3).is_err());
    }

    // ---- R-Precision tests ----

    #[test]
    fn test_r_precision_perfect() {
        // 2 relevant, both in top-2
        let rel = vec![vec![1.0, 1.0, 0.0, 0.0]];
        let val = r_precision(&rel).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_r_precision_half() {
        // 2 relevant, 1 in top-2
        let rel = vec![vec![1.0, 0.0, 1.0, 0.0]];
        let val = r_precision(&rel).expect("should succeed");
        assert!((val - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_r_precision_none_at_top() {
        // 1 relevant, not in top-1
        let rel = vec![vec![0.0, 1.0]];
        let val = r_precision(&rel).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_r_precision_no_relevant() {
        let rel = vec![vec![0.0, 0.0, 0.0]];
        let val = r_precision(&rel).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    // ---- F1@K tests ----

    #[test]
    fn test_f1_at_k_perfect() {
        // All 2 relevant in top-2
        let rel = vec![vec![1.0, 1.0, 0.0]];
        let val = f1_at_k(&rel, 2).expect("should succeed");
        // P@2 = 1.0, R@2 = 1.0 -> F1 = 1.0
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_f1_at_k_zero() {
        let rel = vec![vec![0.0, 0.0, 0.0]];
        let val = f1_at_k(&rel, 2).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_f1_at_k_mixed() {
        // 2 relevant total, 1 in top-2
        let rel = vec![vec![1.0, 0.0, 1.0]];
        let val = f1_at_k(&rel, 2).expect("should succeed");
        // P@2 = 1/2 = 0.5, R@2 = 1/2 = 0.5 -> F1 = 0.5
        assert!((val - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_f1_at_k_k_zero() {
        let rel = vec![vec![1.0, 0.0]];
        assert!(f1_at_k(&rel, 0).is_err());
    }
}
