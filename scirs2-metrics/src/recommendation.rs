//! Recommendation System Metrics
//!
//! Standalone functions for evaluating recommendation systems beyond accuracy.
//! These metrics assess the quality, diversity, novelty, and fairness of
//! recommendations.
//!
//! # Metrics
//!
//! - **Catalog Coverage**: Fraction of catalog items that appear in recommendations
//! - **User Coverage**: Fraction of users who receive at least one recommendation
//! - **Intra-list Diversity**: Average pairwise dissimilarity within recommendation lists
//! - **Novelty**: Average self-information of recommended items (based on popularity)
//! - **Serendipity**: Unexpectedness of relevant recommendations
//! - **Beyond-accuracy Aggregate**: Combined multi-objective quality score
//!
//! # Examples
//!
//! ```
//! use scirs2_metrics::recommendation::{catalog_coverage, user_coverage, novelty};
//!
//! let recommendations = vec![
//!     vec![0, 1, 2],
//!     vec![1, 3, 4],
//!     vec![2, 4, 5],
//! ];
//! let total_items = 10;
//! let total_users = 5;
//!
//! let cc = catalog_coverage(&recommendations, total_items).expect("Failed");
//! let uc = user_coverage(&recommendations, total_users).expect("Failed");
//!
//! let popularity = vec![0.5, 0.3, 0.8, 0.1, 0.2, 0.05];
//! let nov = novelty(&recommendations, &popularity).expect("Failed");
//! ```

use crate::error::{MetricsError, Result};
use std::collections::{HashMap, HashSet};

/// Computes catalog coverage: the fraction of total catalog items that appear
/// in at least one recommendation list.
///
/// Higher coverage means the recommender explores more of the catalog.
///
/// # Arguments
///
/// * `recommendations` - For each user, a list of recommended item IDs.
/// * `total_items` - Total number of items in the catalog.
///
/// # Returns
///
/// Coverage ratio in [0, 1].
///
/// # Errors
///
/// Returns error if total_items is 0.
pub fn catalog_coverage(recommendations: &[Vec<usize>], total_items: usize) -> Result<f64> {
    if total_items == 0 {
        return Err(MetricsError::InvalidInput(
            "total_items must be > 0".to_string(),
        ));
    }

    let unique_items: HashSet<usize> = recommendations
        .iter()
        .flat_map(|items| items.iter().copied())
        .collect();

    Ok(unique_items.len() as f64 / total_items as f64)
}

/// Computes user coverage: the fraction of users who receive at least one
/// recommendation (non-empty list).
///
/// # Arguments
///
/// * `recommendations` - For each user, a list of recommended item IDs.
/// * `total_users` - Total number of users in the system.
///
/// # Returns
///
/// Coverage ratio in [0, 1].
///
/// # Errors
///
/// Returns error if total_users is 0.
pub fn user_coverage(recommendations: &[Vec<usize>], total_users: usize) -> Result<f64> {
    if total_users == 0 {
        return Err(MetricsError::InvalidInput(
            "total_users must be > 0".to_string(),
        ));
    }

    let users_with_recs = recommendations.iter().filter(|r| !r.is_empty()).count();
    Ok(users_with_recs as f64 / total_users as f64)
}

/// Computes intra-list diversity: average pairwise distance between items
/// within each recommendation list.
///
/// Uses a user-provided distance function to measure dissimilarity between
/// pairs of items.
///
/// # Arguments
///
/// * `recommendations` - For each user, a list of recommended item IDs.
/// * `distance_fn` - A function `(item_a, item_b) -> distance` returning
///   a non-negative dissimilarity between two items.
///
/// # Returns
///
/// The mean intra-list diversity across all users with >= 2 recommendations.
/// Higher values mean more diverse recommendation lists.
pub fn intra_list_diversity<F>(recommendations: &[Vec<usize>], distance_fn: F) -> Result<f64>
where
    F: Fn(usize, usize) -> f64,
{
    if recommendations.is_empty() {
        return Err(MetricsError::InvalidInput(
            "recommendations must not be empty".to_string(),
        ));
    }

    let mut diversity_sum = 0.0;
    let mut valid_users = 0;

    for rec_list in recommendations {
        if rec_list.len() < 2 {
            continue;
        }

        let mut pair_sum = 0.0;
        let mut pair_count = 0;

        for i in 0..rec_list.len() {
            for j in i + 1..rec_list.len() {
                pair_sum += distance_fn(rec_list[i], rec_list[j]);
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            diversity_sum += pair_sum / pair_count as f64;
            valid_users += 1;
        }
    }

    if valid_users == 0 {
        return Ok(0.0);
    }

    Ok(diversity_sum / valid_users as f64)
}

/// Computes intra-list diversity using a precomputed distance matrix.
///
/// # Arguments
///
/// * `recommendations` - For each user, a list of recommended item IDs.
/// * `distance_matrix` - A flat vector representing an N x N distance matrix
///   where element [i * n + j] is the distance between items i and j.
/// * `n` - The number of items (side length of the distance matrix).
///
/// # Returns
///
/// The mean intra-list diversity.
pub fn intra_list_diversity_matrix(
    recommendations: &[Vec<usize>],
    distance_matrix: &[f64],
    n: usize,
) -> Result<f64> {
    if n == 0 {
        return Err(MetricsError::InvalidInput("n must be > 0".to_string()));
    }
    if distance_matrix.len() != n * n {
        return Err(MetricsError::InvalidInput(format!(
            "distance_matrix length {} does not match n*n = {}",
            distance_matrix.len(),
            n * n
        )));
    }

    intra_list_diversity(recommendations, |a, b| {
        if a < n && b < n {
            distance_matrix[a * n + b]
        } else {
            0.0
        }
    })
}

/// Computes novelty of recommendations based on item popularity.
///
/// Novelty is measured as the average self-information (surprise) of
/// recommended items:
///   novelty = mean(-log2(popularity_i))
///
/// Less popular items contribute more novelty.
///
/// # Arguments
///
/// * `recommendations` - For each user, a list of recommended item IDs.
/// * `popularity` - A slice where `popularity[item_id]` is the popularity
///   score (probability of interaction) for each item. Values should be in (0, 1].
///
/// # Returns
///
/// The mean novelty across all recommendations. Higher means more novel.
///
/// # Errors
///
/// Returns error if recommendations or popularity is empty, or if item IDs
/// exceed popularity array bounds.
pub fn novelty(recommendations: &[Vec<usize>], popularity: &[f64]) -> Result<f64> {
    if recommendations.is_empty() {
        return Err(MetricsError::InvalidInput(
            "recommendations must not be empty".to_string(),
        ));
    }
    if popularity.is_empty() {
        return Err(MetricsError::InvalidInput(
            "popularity must not be empty".to_string(),
        ));
    }

    let mut novelty_sum = 0.0;
    let mut total_items = 0;

    for rec_list in recommendations {
        for &item_id in rec_list {
            if item_id >= popularity.len() {
                return Err(MetricsError::InvalidInput(format!(
                    "item_id {} exceeds popularity array length {}",
                    item_id,
                    popularity.len()
                )));
            }

            let pop = popularity[item_id];
            if pop > 0.0 {
                novelty_sum += -pop.log2();
            }
            // Items with 0 popularity are maximally novel (infinite self-information)
            // We skip them to avoid infinity in the average
            total_items += 1;
        }
    }

    if total_items == 0 {
        return Ok(0.0);
    }

    Ok(novelty_sum / total_items as f64)
}

/// Computes serendipity of recommendations.
///
/// Serendipity measures the unexpectedness of *relevant* recommendations.
/// An item is serendipitous if it is both relevant and unexpected (not predicted
/// by a simple baseline like popularity-based recommendation).
///
/// serendipity = (1/|relevant|) * sum_{i in relevant} unexpectedness(i)
///
/// # Arguments
///
/// * `recommendations` - For each user, a list of recommended item IDs.
/// * `relevance` - For each user, a set of relevant item IDs (ground truth).
/// * `expected_items` - A set of items the baseline model would recommend.
///   Items NOT in this set are considered unexpected.
///
/// # Returns
///
/// The mean serendipity across users.
pub fn serendipity(
    recommendations: &[Vec<usize>],
    relevance: &[HashSet<usize>],
    expected_items: &HashSet<usize>,
) -> Result<f64> {
    if recommendations.len() != relevance.len() {
        return Err(MetricsError::InvalidInput(
            "recommendations and relevance must have the same length".to_string(),
        ));
    }
    if recommendations.is_empty() {
        return Err(MetricsError::InvalidInput(
            "recommendations must not be empty".to_string(),
        ));
    }

    let mut serendipity_sum = 0.0;
    let mut valid_users = 0;

    for (rec_list, rel_set) in recommendations.iter().zip(relevance.iter()) {
        let mut unexpected_relevant = 0;
        let mut relevant_in_rec = 0;

        for &item_id in rec_list {
            if rel_set.contains(&item_id) {
                relevant_in_rec += 1;
                if !expected_items.contains(&item_id) {
                    unexpected_relevant += 1;
                }
            }
        }

        if relevant_in_rec > 0 {
            serendipity_sum += unexpected_relevant as f64 / relevant_in_rec as f64;
            valid_users += 1;
        }
    }

    if valid_users == 0 {
        return Ok(0.0);
    }

    Ok(serendipity_sum / valid_users as f64)
}

/// Computes the Gini index of the item recommendation distribution.
///
/// A Gini index of 0 means all items are recommended equally often.
/// A Gini index approaching 1 means recommendations are concentrated on few items.
///
/// # Arguments
///
/// * `recommendations` - For each user, a list of recommended item IDs.
/// * `total_items` - Total number of items in the catalog.
///
/// # Returns
///
/// The Gini index in [0, 1].
pub fn gini_index(recommendations: &[Vec<usize>], total_items: usize) -> Result<f64> {
    if total_items == 0 {
        return Err(MetricsError::InvalidInput(
            "total_items must be > 0".to_string(),
        ));
    }

    // Count item frequencies
    let mut item_counts = vec![0usize; total_items];
    for rec_list in recommendations {
        for &item_id in rec_list {
            if item_id < total_items {
                item_counts[item_id] += 1;
            }
        }
    }

    // Sort counts
    item_counts.sort_unstable();

    let n = item_counts.len();
    let total: usize = item_counts.iter().sum();

    if total == 0 {
        return Ok(0.0);
    }

    let mut gini_sum = 0.0;
    for (i, &count) in item_counts.iter().enumerate() {
        gini_sum += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * count as f64;
    }

    Ok(gini_sum / (n as f64 * total as f64))
}

/// Computes the Shannon entropy of the item recommendation distribution.
///
/// Higher entropy indicates more uniform distribution of recommendations.
///
/// # Arguments
///
/// * `recommendations` - For each user, a list of recommended item IDs.
///
/// # Returns
///
/// The entropy value (non-negative). Higher means more diverse.
pub fn recommendation_entropy(recommendations: &[Vec<usize>]) -> Result<f64> {
    if recommendations.is_empty() {
        return Err(MetricsError::InvalidInput(
            "recommendations must not be empty".to_string(),
        ));
    }

    let mut item_counts: HashMap<usize, usize> = HashMap::new();
    let mut total = 0usize;

    for rec_list in recommendations {
        for &item_id in rec_list {
            *item_counts.entry(item_id).or_insert(0) += 1;
            total += 1;
        }
    }

    if total == 0 {
        return Ok(0.0);
    }

    let mut entropy = 0.0;
    for &count in item_counts.values() {
        let p = count as f64 / total as f64;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    Ok(entropy)
}

/// Computes a beyond-accuracy aggregate score combining multiple quality dimensions.
///
/// This composite metric balances accuracy, diversity, novelty, and coverage
/// using configurable weights.
///
/// # Arguments
///
/// * `accuracy` - An accuracy metric value (e.g., NDCG, MAP) in [0, 1].
/// * `diversity` - A diversity metric value in [0, 1] (normalized).
/// * `novelty` - A novelty value (will be normalized by max_novelty).
/// * `coverage` - A coverage metric value in [0, 1].
/// * `weights` - Weights for [accuracy, diversity, novelty, coverage]. Must sum to 1.
///
/// # Returns
///
/// The weighted aggregate score.
pub fn beyond_accuracy_score(
    accuracy: f64,
    diversity: f64,
    novelty_val: f64,
    coverage: f64,
    weights: &[f64; 4],
) -> Result<f64> {
    // Validate weights sum to approximately 1
    let weight_sum: f64 = weights.iter().sum();
    if (weight_sum - 1.0).abs() > 1e-6 {
        return Err(MetricsError::InvalidInput(format!(
            "weights must sum to 1.0, got {}",
            weight_sum
        )));
    }

    // Validate ranges
    for &w in weights {
        if w < 0.0 {
            return Err(MetricsError::InvalidInput(
                "weights must be non-negative".to_string(),
            ));
        }
    }

    Ok(weights[0] * accuracy
        + weights[1] * diversity
        + weights[2] * novelty_val
        + weights[3] * coverage)
}

/// Computes popularity bias: the ratio of popular items in recommendations
/// versus in the catalog.
///
/// A value of 1.0 indicates no bias. Values > 1.0 indicate the recommender
/// favors popular items.
///
/// # Arguments
///
/// * `recommendations` - For each user, a list of recommended item IDs.
/// * `popularity` - `popularity[item_id]` is the popularity score for each item.
/// * `popularity_threshold` - Items with popularity >= threshold are "popular".
///
/// # Returns
///
/// The popularity bias ratio (recommended popular fraction / catalog popular fraction).
pub fn popularity_bias(
    recommendations: &[Vec<usize>],
    popularity: &[f64],
    popularity_threshold: f64,
) -> Result<f64> {
    if recommendations.is_empty() || popularity.is_empty() {
        return Err(MetricsError::InvalidInput(
            "inputs must not be empty".to_string(),
        ));
    }

    // Compute catalog popular fraction
    let catalog_popular = popularity
        .iter()
        .filter(|&&p| p >= popularity_threshold)
        .count();
    let catalog_popular_frac = catalog_popular as f64 / popularity.len() as f64;

    if catalog_popular_frac <= 0.0 {
        // No popular items in catalog, cannot compute bias ratio
        return Ok(1.0);
    }

    // Compute recommended popular fraction
    let mut total_recs = 0;
    let mut popular_recs = 0;

    for rec_list in recommendations {
        for &item_id in rec_list {
            total_recs += 1;
            if item_id < popularity.len() && popularity[item_id] >= popularity_threshold {
                popular_recs += 1;
            }
        }
    }

    if total_recs == 0 {
        return Ok(1.0);
    }

    let rec_popular_frac = popular_recs as f64 / total_recs as f64;

    Ok(rec_popular_frac / catalog_popular_frac)
}

/// Computes long-tail coverage: the fraction of "long-tail" (unpopular) items
/// that appear in recommendations.
///
/// # Arguments
///
/// * `recommendations` - For each user, a list of recommended item IDs.
/// * `popularity` - `popularity[item_id]` is the popularity score.
/// * `popularity_threshold` - Items with popularity < threshold are "long-tail".
///
/// # Returns
///
/// The long-tail coverage ratio in [0, 1].
pub fn long_tail_coverage(
    recommendations: &[Vec<usize>],
    popularity: &[f64],
    popularity_threshold: f64,
) -> Result<f64> {
    if popularity.is_empty() {
        return Err(MetricsError::InvalidInput(
            "popularity must not be empty".to_string(),
        ));
    }

    // Find long-tail items
    let long_tail_items: HashSet<usize> = popularity
        .iter()
        .enumerate()
        .filter(|(_, &p)| p < popularity_threshold)
        .map(|(i, _)| i)
        .collect();

    if long_tail_items.is_empty() {
        return Ok(0.0);
    }

    // Find recommended long-tail items
    let recommended_long_tail: HashSet<usize> = recommendations
        .iter()
        .flat_map(|rec| rec.iter().copied())
        .filter(|&item_id| long_tail_items.contains(&item_id))
        .collect();

    Ok(recommended_long_tail.len() as f64 / long_tail_items.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- catalog_coverage tests ----

    #[test]
    fn test_catalog_coverage_full() {
        let recs = vec![vec![0, 1, 2], vec![3, 4]];
        let val = catalog_coverage(&recs, 5).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_catalog_coverage_partial() {
        let recs = vec![vec![0, 1], vec![1, 2]];
        let val = catalog_coverage(&recs, 10).expect("should succeed");
        assert!((val - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_catalog_coverage_empty_recs() {
        let recs: Vec<Vec<usize>> = vec![vec![], vec![]];
        let val = catalog_coverage(&recs, 10).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_catalog_coverage_zero_items() {
        let recs = vec![vec![0]];
        assert!(catalog_coverage(&recs, 0).is_err());
    }

    // ---- user_coverage tests ----

    #[test]
    fn test_user_coverage_all() {
        let recs = vec![vec![0], vec![1], vec![2]];
        let val = user_coverage(&recs, 3).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_user_coverage_partial() {
        let recs = vec![vec![0], vec![], vec![2]];
        let val = user_coverage(&recs, 4).expect("should succeed");
        assert!((val - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_user_coverage_none() {
        let recs = vec![vec![], vec![]];
        let val = user_coverage(&recs, 5).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_user_coverage_zero_users() {
        let recs = vec![vec![0]];
        assert!(user_coverage(&recs, 0).is_err());
    }

    // ---- intra_list_diversity tests ----

    #[test]
    fn test_ild_uniform_distance() {
        let recs = vec![vec![0, 1, 2]];
        let val = intra_list_diversity(&recs, |_a, _b| 1.0).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ild_zero_distance() {
        let recs = vec![vec![0, 1, 2]];
        let val = intra_list_diversity(&recs, |_a, _b| 0.0).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ild_single_item() {
        let recs = vec![vec![0]]; // Only 1 item, no pairs
        let val = intra_list_diversity(&recs, |_a, _b| 1.0).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ild_matrix() {
        // 3 items, distance matrix
        let dist = vec![0.0, 0.5, 0.8, 0.5, 0.0, 0.3, 0.8, 0.3, 0.0];
        let recs = vec![vec![0, 1, 2]];
        let val = intra_list_diversity_matrix(&recs, &dist, 3).expect("should succeed");
        // Pairs: (0,1)=0.5, (0,2)=0.8, (1,2)=0.3 -> mean = 1.6/3
        assert!((val - 1.6 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ild_empty() {
        let recs: Vec<Vec<usize>> = vec![];
        assert!(intra_list_diversity(&recs, |_a, _b| 1.0).is_err());
    }

    // ---- novelty tests ----

    #[test]
    fn test_novelty_popular_items() {
        let recs = vec![vec![0, 1]];
        let popularity = vec![0.5, 0.5]; // fairly popular
        let val = novelty(&recs, &popularity).expect("should succeed");
        // -log2(0.5) = 1.0 for each item
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_novelty_unpopular_items() {
        let recs = vec![vec![0, 1]];
        let popularity = vec![0.125, 0.125]; // unpopular
        let val = novelty(&recs, &popularity).expect("should succeed");
        // -log2(0.125) = 3.0 for each item
        assert!((val - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_novelty_mixed() {
        let recs = vec![vec![0, 1]];
        let popularity = vec![0.5, 0.25]; // mixed
        let val = novelty(&recs, &popularity).expect("should succeed");
        // (-log2(0.5) + -log2(0.25)) / 2 = (1 + 2) / 2 = 1.5
        assert!((val - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_novelty_empty() {
        let recs: Vec<Vec<usize>> = vec![];
        let popularity = vec![0.5];
        assert!(novelty(&recs, &popularity).is_err());
    }

    #[test]
    fn test_novelty_out_of_bounds() {
        let recs = vec![vec![5]]; // item 5 exceeds popularity length
        let popularity = vec![0.5, 0.3];
        assert!(novelty(&recs, &popularity).is_err());
    }

    // ---- serendipity tests ----

    #[test]
    fn test_serendipity_all_unexpected() {
        let recs = vec![vec![0, 1, 2]];
        let relevance = vec![[0, 1, 2].iter().copied().collect::<HashSet<usize>>()];
        let expected: HashSet<usize> = HashSet::new(); // nothing expected
        let val = serendipity(&recs, &relevance, &expected).expect("should succeed");
        // All 3 relevant items are unexpected
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_serendipity_all_expected() {
        let recs = vec![vec![0, 1, 2]];
        let relevance = vec![[0, 1, 2].iter().copied().collect::<HashSet<usize>>()];
        let expected: HashSet<usize> = [0, 1, 2].iter().copied().collect();
        let val = serendipity(&recs, &relevance, &expected).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_serendipity_partial() {
        let recs = vec![vec![0, 1, 2]];
        let relevance = vec![[0, 1, 2].iter().copied().collect::<HashSet<usize>>()];
        let expected: HashSet<usize> = [0].iter().copied().collect(); // only item 0 expected
        let val = serendipity(&recs, &relevance, &expected).expect("should succeed");
        // 2 unexpected relevant out of 3 relevant
        assert!((val - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_serendipity_no_relevant() {
        let recs = vec![vec![0, 1, 2]];
        let relevance = vec![HashSet::new()];
        let expected: HashSet<usize> = HashSet::new();
        let val = serendipity(&recs, &relevance, &expected).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_serendipity_mismatched_len() {
        let recs = vec![vec![0]];
        let relevance = vec![];
        let expected = HashSet::new();
        assert!(serendipity(&recs, &relevance, &expected).is_err());
    }

    // ---- gini_index tests ----

    #[test]
    fn test_gini_perfectly_equal() {
        let recs = vec![vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]];
        let val = gini_index(&recs, 3).expect("should succeed");
        assert!(val.abs() < 0.01); // close to 0
    }

    #[test]
    fn test_gini_perfectly_unequal() {
        // Only item 0 ever recommended
        let recs = vec![vec![0, 0, 0], vec![0, 0], vec![0]];
        let val = gini_index(&recs, 3).expect("should succeed");
        // Very high Gini, but not exactly 1 due to formula specifics
        assert!(val > 0.5);
    }

    #[test]
    fn test_gini_empty() {
        let recs: Vec<Vec<usize>> = vec![vec![]];
        let val = gini_index(&recs, 5).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gini_zero_items() {
        let recs = vec![vec![0]];
        assert!(gini_index(&recs, 0).is_err());
    }

    // ---- recommendation_entropy tests ----

    #[test]
    fn test_entropy_uniform() {
        // All items equally likely
        let recs = vec![vec![0], vec![1], vec![2], vec![3]];
        let val = recommendation_entropy(&recs).expect("should succeed");
        // 4 items, each with probability 0.25 -> entropy = 2.0
        assert!((val - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_concentrated() {
        // Only one item
        let recs = vec![vec![0], vec![0], vec![0], vec![0]];
        let val = recommendation_entropy(&recs).expect("should succeed");
        // Entropy = 0 (no uncertainty)
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_binary() {
        let recs = vec![vec![0], vec![1]];
        let val = recommendation_entropy(&recs).expect("should succeed");
        // 2 items, each p=0.5 -> entropy = 1.0
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_empty() {
        let recs: Vec<Vec<usize>> = vec![];
        assert!(recommendation_entropy(&recs).is_err());
    }

    // ---- beyond_accuracy_score tests ----

    #[test]
    fn test_beyond_accuracy_equal_weights() {
        let val = beyond_accuracy_score(0.8, 0.6, 0.5, 0.7, &[0.25, 0.25, 0.25, 0.25])
            .expect("should succeed");
        let expected = 0.25 * 0.8 + 0.25 * 0.6 + 0.25 * 0.5 + 0.25 * 0.7;
        assert!((val - expected).abs() < 1e-10);
    }

    #[test]
    fn test_beyond_accuracy_accuracy_only() {
        let val = beyond_accuracy_score(0.9, 0.5, 0.3, 0.2, &[1.0, 0.0, 0.0, 0.0])
            .expect("should succeed");
        assert!((val - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_beyond_accuracy_invalid_weights() {
        assert!(beyond_accuracy_score(0.5, 0.5, 0.5, 0.5, &[0.5, 0.5, 0.5, 0.5]).is_err());
    }

    #[test]
    fn test_beyond_accuracy_negative_weights() {
        assert!(beyond_accuracy_score(0.5, 0.5, 0.5, 0.5, &[-0.1, 0.5, 0.3, 0.3]).is_err());
    }

    // ---- popularity_bias tests ----

    #[test]
    fn test_popularity_bias_no_bias() {
        // 2 out of 4 items are popular, 2 out of 4 recs are popular
        let recs = vec![vec![0, 1], vec![2, 3]];
        let popularity = vec![0.8, 0.2, 0.9, 0.1]; // items 0,2 are popular at threshold 0.5
        let val = popularity_bias(&recs, &popularity, 0.5).expect("should succeed");
        // Catalog popular: 2/4 = 0.5, Rec popular: 2/4 = 0.5 -> ratio = 1.0
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_popularity_bias_high_bias() {
        // All recs are popular items
        let recs = vec![vec![0, 0], vec![0, 0]];
        let popularity = vec![0.9, 0.1, 0.05, 0.02]; // only item 0 popular at 0.5
        let val = popularity_bias(&recs, &popularity, 0.5).expect("should succeed");
        // Catalog popular: 1/4 = 0.25, Rec popular: 4/4 = 1.0 -> ratio = 4.0
        assert!((val - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_popularity_bias_empty() {
        let recs: Vec<Vec<usize>> = vec![];
        let popularity = vec![0.5];
        assert!(popularity_bias(&recs, &popularity, 0.5).is_err());
    }

    #[test]
    fn test_popularity_bias_no_popular_items() {
        let recs = vec![vec![0, 1]];
        let popularity = vec![0.1, 0.2]; // none popular at threshold 0.5
        let val = popularity_bias(&recs, &popularity, 0.5).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10); // returns 1.0 when no popular items
    }

    // ---- long_tail_coverage tests ----

    #[test]
    fn test_long_tail_full() {
        // All long-tail items are recommended
        let recs = vec![vec![1, 2, 3]];
        let popularity = vec![0.8, 0.1, 0.05, 0.02]; // items 1,2,3 are long-tail at 0.5
        let val = long_tail_coverage(&recs, &popularity, 0.5).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_long_tail_none() {
        // No long-tail items recommended
        let recs = vec![vec![0]]; // only popular item
        let popularity = vec![0.8, 0.1, 0.05];
        let val = long_tail_coverage(&recs, &popularity, 0.5).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_long_tail_partial() {
        let recs = vec![vec![1]]; // only item 1 from long-tail
        let popularity = vec![0.8, 0.1, 0.05]; // items 1,2 are long-tail at 0.5
        let val = long_tail_coverage(&recs, &popularity, 0.5).expect("should succeed");
        assert!((val - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_long_tail_empty_popularity() {
        let recs = vec![vec![0]];
        let popularity: Vec<f64> = vec![];
        assert!(long_tail_coverage(&recs, &popularity, 0.5).is_err());
    }
}
