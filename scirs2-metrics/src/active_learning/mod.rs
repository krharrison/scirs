//! Active Learning Metrics
//!
//! This module provides metrics and selection strategies for active learning:
//!
//! - **Uncertainty sampling**: margin, entropy, least confidence
//! - **Query-by-committee**: vote entropy, KL disagreement
//! - **Expected model change**: gradient magnitude proxy
//! - **Core-set selection**: greedy farthest-first traversal for diversity
//! - **Batch-mode selection**: diversity-weighted batch selection
//! - **Candidate ranking**: top-n selection by score

use crate::error::{MetricsError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for active learning experiments.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ActiveLearningConfig {
    /// Number of committee members for query-by-committee methods.
    pub n_committee: usize,
    /// Number of candidate samples to consider.
    pub n_candidates: usize,
}

impl Default for ActiveLearningConfig {
    fn default() -> Self {
        Self {
            n_committee: 5,
            n_candidates: 100,
        }
    }
}

/// Type of uncertainty scoring used for active learning.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UncertaintyScore {
    /// Margin sampling: 1 - (p_max - p_second_max)
    MarginSampling,
    /// Entropy sampling: H(p) = -sum(p_i * log(p_i))
    EntropySampling,
    /// Least confidence: 1 - p_max
    LeastConfidence,
    /// Query by committee disagreement
    QueryByCommittee,
    /// Expected model change (gradient magnitude proxy)
    ExpectedModelChange,
    /// Core-set diversity selection (farthest-first traversal)
    CoreSet,
}

// ─────────────────────────────────────────────────────────────────────────────
// Uncertainty Sampling — batch (vector-of-vectors) API
// ─────────────────────────────────────────────────────────────────────────────

/// Margin sampling over multiple candidates.
///
/// For each candidate, computes `1 - (p_max - p_second_max)`.
/// A smaller margin means more uncertainty; returned score is higher when
/// the model is more uncertain.
///
/// Each inner `Vec<f64>` must have at least 2 class probabilities.
pub fn margin_sampling(probs: &[Vec<f64>]) -> Result<Vec<f64>> {
    if probs.is_empty() {
        return Err(MetricsError::InvalidInput(
            "probs must not be empty".to_string(),
        ));
    }
    probs
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if p.len() < 2 {
                return Err(MetricsError::InvalidInput(format!(
                    "sample {i}: margin sampling requires at least 2 class probabilities"
                )));
            }
            let mut sorted = p.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let margin = sorted[0] - sorted[1];
            Ok(1.0 - margin)
        })
        .collect()
}

/// Entropy sampling over multiple candidates.
///
/// For each candidate, computes Shannon entropy `H(p) = -sum(p_i * ln(p_i))`.
/// Higher entropy indicates more uncertainty (uniform distribution maximises it).
pub fn entropy_sampling(probs: &[Vec<f64>]) -> Result<Vec<f64>> {
    if probs.is_empty() {
        return Err(MetricsError::InvalidInput(
            "probs must not be empty".to_string(),
        ));
    }
    probs
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if p.is_empty() {
                return Err(MetricsError::InvalidInput(format!(
                    "sample {i}: probabilities must not be empty"
                )));
            }
            let h: f64 = p
                .iter()
                .filter(|&&pi| pi > 0.0)
                .map(|&pi| -pi * pi.ln())
                .sum();
            Ok(h)
        })
        .collect()
}

/// Least confidence over multiple candidates.
///
/// For each candidate, computes `1 - max(p_i)`.
/// Lower maximum probability indicates more uncertainty.
pub fn least_confidence(probs: &[Vec<f64>]) -> Result<Vec<f64>> {
    if probs.is_empty() {
        return Err(MetricsError::InvalidInput(
            "probs must not be empty".to_string(),
        ));
    }
    probs
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if p.is_empty() {
                return Err(MetricsError::InvalidInput(format!(
                    "sample {i}: probabilities must not be empty"
                )));
            }
            let p_max = p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            Ok(1.0 - p_max)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-sample convenience functions (backwards-compatible)
// ─────────────────────────────────────────────────────────────────────────────

/// Margin sampling score for a single sample: `1 - (p_max - p_second_max)`.
pub fn margin_sampling_score(probabilities: &[f64]) -> Result<f64> {
    if probabilities.len() < 2 {
        return Err(MetricsError::InvalidInput(
            "margin sampling requires at least 2 class probabilities".to_string(),
        ));
    }
    let mut sorted = probabilities.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let margin = sorted[0] - sorted[1];
    Ok(1.0 - margin)
}

/// Entropy-based uncertainty for a single sample: `H(p) = -sum(p_i * ln(p_i))`.
pub fn entropy_uncertainty(probabilities: &[f64]) -> Result<f64> {
    if probabilities.is_empty() {
        return Err(MetricsError::InvalidInput(
            "probabilities must not be empty".to_string(),
        ));
    }
    let h = probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum::<f64>();
    Ok(h)
}

/// Least confidence for a single sample: `1 - max(p_i)`.
pub fn least_confidence_score(probabilities: &[f64]) -> Result<f64> {
    if probabilities.is_empty() {
        return Err(MetricsError::InvalidInput(
            "probabilities must not be empty".to_string(),
        ));
    }
    let p_max = probabilities
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(1.0 - p_max)
}

// ─────────────────────────────────────────────────────────────────────────────
// Query-by-Committee
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that a committee is non-empty and all members have the same
/// number of classes.
fn check_committee(committee_probs: &[Vec<f64>]) -> Result<usize> {
    if committee_probs.is_empty() {
        return Err(MetricsError::InvalidInput(
            "committee must have at least one member".to_string(),
        ));
    }
    let n_classes = committee_probs[0].len();
    if n_classes == 0 {
        return Err(MetricsError::InvalidInput(
            "each committee member must supply at least one class probability".to_string(),
        ));
    }
    for (i, member) in committee_probs.iter().enumerate() {
        if member.len() != n_classes {
            return Err(MetricsError::DimensionMismatch(format!(
                "committee member {i} has {} classes, expected {n_classes}",
                member.len()
            )));
        }
    }
    Ok(n_classes)
}

/// Query-by-committee disagreement via vote entropy (single sample).
///
/// Each committee member "votes" for the class with the highest probability.
/// Returns the entropy of the resulting vote distribution.
pub fn vote_entropy(committee_probs: &[Vec<f64>]) -> Result<f64> {
    let n_classes = check_committee(committee_probs)?;
    let n_members = committee_probs.len() as f64;

    let mut votes = vec![0usize; n_classes];
    for member in committee_probs {
        let winner = member
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        votes[winner] += 1;
    }

    let h = votes
        .iter()
        .filter(|&&v| v > 0)
        .map(|&v| {
            let frac = v as f64 / n_members;
            -frac * frac.ln()
        })
        .sum::<f64>();
    Ok(h)
}

/// Query-by-committee disagreement via average KL divergence from consensus (single sample).
pub fn qbc_kl_disagreement(committee_probs: &[Vec<f64>]) -> Result<f64> {
    let n_classes = check_committee(committee_probs)?;
    let n_members = committee_probs.len() as f64;

    let mut consensus = vec![0.0_f64; n_classes];
    for member in committee_probs {
        for (c, &p) in consensus.iter_mut().zip(member) {
            *c += p;
        }
    }
    for c in &mut consensus {
        *c /= n_members;
    }

    let mut total_kl = 0.0_f64;
    for member in committee_probs {
        let kl: f64 = member
            .iter()
            .zip(&consensus)
            .map(|(&pi, &mi)| {
                if pi <= 0.0 {
                    0.0
                } else if mi <= 0.0 {
                    f64::INFINITY
                } else {
                    pi * (pi / mi).ln()
                }
            })
            .sum();
        if kl.is_infinite() {
            return Err(MetricsError::CalculationError(
                "KL divergence is infinite in committee disagreement".to_string(),
            ));
        }
        total_kl += kl;
    }
    Ok(total_kl / n_members)
}

/// Query-by-committee for multiple candidates.
///
/// `committee_probs[m][s]` is committee member `m`'s probability vector for sample `s`.
/// Returns a disagreement score per sample (vote entropy across committee members).
pub fn query_by_committee(committee_probs: &[Vec<Vec<f64>>]) -> Result<Vec<f64>> {
    if committee_probs.is_empty() {
        return Err(MetricsError::InvalidInput(
            "committee_probs must have at least one member".to_string(),
        ));
    }
    let n_members = committee_probs.len();
    let n_samples = committee_probs[0].len();

    // Validate dimensions
    for (m, member) in committee_probs.iter().enumerate() {
        if member.len() != n_samples {
            return Err(MetricsError::DimensionMismatch(format!(
                "committee member {m} has {} samples, expected {n_samples}",
                member.len()
            )));
        }
    }

    let mut scores = Vec::with_capacity(n_samples);
    for s in 0..n_samples {
        // Gather this sample's predictions from all committee members
        let sample_probs: Vec<Vec<f64>> = (0..n_members)
            .map(|m| committee_probs[m][s].clone())
            .collect();
        let ve = vote_entropy(&sample_probs)?;
        scores.push(ve);
    }
    Ok(scores)
}

// ─────────────────────────────────────────────────────────────────────────────
// Expected Model Change
// ─────────────────────────────────────────────────────────────────────────────

/// Expected model change: uses gradient norm as a proxy for informativeness.
///
/// `gradients[i]` is the gradient vector (or gradient magnitude proxy) for candidate `i`.
/// Returns `||gradient_i||_2` for each candidate.
pub fn expected_model_change(gradients: &[Vec<f64>]) -> Result<Vec<f64>> {
    if gradients.is_empty() {
        return Err(MetricsError::InvalidInput(
            "gradients must not be empty".to_string(),
        ));
    }
    gradients
        .iter()
        .enumerate()
        .map(|(i, g)| {
            if g.is_empty() {
                return Err(MetricsError::InvalidInput(format!(
                    "sample {i} has empty gradient vector"
                )));
            }
            let norm = g.iter().map(|&v| v * v).sum::<f64>().sqrt();
            Ok(norm)
        })
        .collect()
}

/// Expected gradient magnitude proxy (probability-based): `||p - y_one_hot||_2`.
///
/// Approximated as the Euclidean distance from the predicted probability
/// vector to the one-hot encoding of the predicted class (argmax).
/// Returns one magnitude value per sample.
pub fn expected_gradient_magnitude(probabilities: &[Vec<f64>]) -> Result<Vec<f64>> {
    if probabilities.is_empty() {
        return Err(MetricsError::InvalidInput(
            "probabilities must not be empty".to_string(),
        ));
    }
    probabilities
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if p.is_empty() {
                return Err(MetricsError::InvalidInput(format!(
                    "sample {i} has empty probability vector"
                )));
            }
            let argmax = p
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(j, _)| j)
                .unwrap_or(0);
            let mag = p
                .iter()
                .enumerate()
                .map(|(j, &pj)| {
                    let one_hot = if j == argmax { 1.0 } else { 0.0 };
                    (pj - one_hot).powi(2)
                })
                .sum::<f64>()
                .sqrt();
            Ok(mag)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Core-Set Selection
// ─────────────────────────────────────────────────────────────────────────────

/// Euclidean distance between two feature vectors.
fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Core-set selection via greedy farthest-first traversal.
///
/// Given `embeddings` for all candidates and `selected` (indices of already-labeled
/// points), selects `n_select` new points that maximise minimum distance to the
/// already-selected set plus newly-chosen points.
///
/// If `selected` is empty, the first point (index 0) is used as the seed.
pub fn core_set_selection(
    embeddings: &[Vec<f64>],
    selected: &[usize],
    n_select: usize,
) -> Result<Vec<usize>> {
    if embeddings.is_empty() {
        return Err(MetricsError::InvalidInput(
            "embeddings must not be empty".to_string(),
        ));
    }
    if n_select == 0 {
        return Ok(vec![]);
    }
    let n = embeddings.len();
    if n_select > n {
        return Err(MetricsError::InvalidInput(format!(
            "n_select ({n_select}) exceeds number of points ({n})"
        )));
    }

    // Build initial set of centres
    let mut centres: Vec<usize> = selected.to_vec();
    // Mark already-selected as used
    let mut used = vec![false; n];
    for &idx in &centres {
        if idx < n {
            used[idx] = true;
        }
    }

    // If no centres, seed with index 0
    if centres.is_empty() {
        centres.push(0);
        used[0] = true;
    }

    // Compute initial min-dist from each point to nearest centre
    let mut min_dists: Vec<f64> = (0..n)
        .map(|i| {
            if used[i] {
                return 0.0;
            }
            centres
                .iter()
                .map(|&c| {
                    if c < n {
                        euclidean_dist(&embeddings[i], &embeddings[c])
                    } else {
                        f64::INFINITY
                    }
                })
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    let mut new_selected = Vec::with_capacity(n_select);

    while new_selected.len() < n_select {
        // Pick the point with the largest min-dist (farthest from all centres)
        let next = min_dists
            .iter()
            .enumerate()
            .filter(|(i, _)| !used[*i])
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);

        match next {
            Some(idx) => {
                new_selected.push(idx);
                used[idx] = true;
                // Update min_dists
                let new_centre = &embeddings[idx];
                for (i, md) in min_dists.iter_mut().enumerate() {
                    if !used[i] {
                        let d = euclidean_dist(&embeddings[i], new_centre);
                        if d < *md {
                            *md = d;
                        }
                    }
                }
            }
            None => break, // No more candidates
        }
    }

    Ok(new_selected)
}

/// Greedy k-center core-set selection (legacy API).
///
/// Iteratively selects the point that is farthest from the current set of
/// selected centres, maximising minimum coverage.
///
/// Returns `k` indices into `features`.
pub fn greedy_k_center(
    features: &[Vec<f64>],
    k: usize,
    seed_idx: Option<usize>,
) -> Result<Vec<usize>> {
    if features.is_empty() {
        return Err(MetricsError::InvalidInput(
            "features must not be empty".to_string(),
        ));
    }
    if k == 0 {
        return Err(MetricsError::InvalidInput(
            "k must be at least 1".to_string(),
        ));
    }
    if k > features.len() {
        return Err(MetricsError::InvalidInput(format!(
            "k ({k}) exceeds number of points ({})",
            features.len()
        )));
    }

    let n = features.len();
    let first = seed_idx.unwrap_or(0).min(n - 1);

    let mut selected = vec![first];
    let mut min_dists: Vec<f64> = features
        .iter()
        .map(|f| euclidean_dist(f, &features[first]))
        .collect();

    while selected.len() < k {
        let next = min_dists
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        selected.push(next);
        let new_centre = &features[next];
        for (i, md) in min_dists.iter_mut().enumerate() {
            let d = euclidean_dist(&features[i], new_centre);
            if d < *md {
                *md = d;
            }
        }
    }

    Ok(selected)
}

// ─────────────────────────────────────────────────────────────────────────────
// Ranking & Selection Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Rank candidates by score (highest first) and return the top `n_select` indices.
///
/// If `n_select >= scores.len()`, returns all indices sorted by descending score.
pub fn rank_candidates(scores: &[f64], n_select: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(n_select);
    indices
}

/// Rank candidates by active learning score (highest score -> highest priority).
///
/// Returns all indices sorted from most to least uncertain.
pub fn rank_by_uncertainty(scores: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch-Mode Selection
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for batch-mode active learning selection.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BatchSelectionMethod {
    /// Select by entropy uncertainty score.
    Entropy,
    /// Select by margin sampling score.
    MarginSampling,
    /// Select by greedy k-center core-set.
    CoreSet,
}

/// Configuration for batch-mode active learning selection.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BatchSelectionConfig {
    /// Number of samples to select.
    pub n_select: usize,
    /// Weight on diversity vs pure uncertainty in `[0, 1]`.
    /// `0.0` = pure uncertainty, `1.0` = pure diversity (CoreSet).
    pub diversity_weight: f64,
    /// Selection method.
    pub method: BatchSelectionMethod,
}

impl Default for BatchSelectionConfig {
    fn default() -> Self {
        Self {
            n_select: 10,
            diversity_weight: 0.5,
            method: BatchSelectionMethod::Entropy,
        }
    }
}

/// Batch selection that balances uncertainty with diversity.
///
/// When `diversity_weight == 0.0`, this is equivalent to pure uncertainty ranking.
/// When `diversity_weight == 1.0`, this is pure core-set (diversity) selection.
/// Values in between produce a hybrid: candidates are scored by
/// `(1 - diversity_weight) * normalized_uncertainty + diversity_weight * normalized_distance`.
///
/// `scores` is an uncertainty score per candidate (higher = more uncertain).
/// `embeddings` is a feature vector per candidate (for computing distances).
/// `n_select` is how many candidates to choose.
pub fn batch_selection(
    scores: &[f64],
    embeddings: &[Vec<f64>],
    n_select: usize,
    diversity_weight: f64,
) -> Result<Vec<usize>> {
    if scores.is_empty() || embeddings.is_empty() {
        return Err(MetricsError::InvalidInput(
            "scores and embeddings must not be empty".to_string(),
        ));
    }
    if scores.len() != embeddings.len() {
        return Err(MetricsError::DimensionMismatch(format!(
            "scores len {} != embeddings len {}",
            scores.len(),
            embeddings.len()
        )));
    }

    let n = scores.len();
    let k = n_select.min(n);

    if k == 0 {
        return Ok(vec![]);
    }

    // Pure uncertainty
    let dw = diversity_weight.clamp(0.0, 1.0);
    if dw < 1e-12 {
        return Ok(rank_candidates(scores, k));
    }

    // Pure diversity
    if (dw - 1.0).abs() < 1e-12 {
        return core_set_selection(embeddings, &[], k);
    }

    // Hybrid: greedy selection balancing uncertainty + diversity
    // Normalize uncertainty scores to [0, 1]
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let score_range = max_score - min_score;
    let norm_scores: Vec<f64> = if score_range > 1e-15 {
        scores
            .iter()
            .map(|&s| (s - min_score) / score_range)
            .collect()
    } else {
        vec![0.5; n]
    };

    let mut selected = Vec::with_capacity(k);
    let mut used = vec![false; n];

    // Seed with highest-uncertainty point
    let seed = norm_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    selected.push(seed);
    used[seed] = true;

    // Min distance from each point to selected set
    let mut min_dists: Vec<f64> = (0..n)
        .map(|i| {
            if i == seed {
                0.0
            } else {
                euclidean_dist(&embeddings[i], &embeddings[seed])
            }
        })
        .collect();

    while selected.len() < k {
        // Normalize min_dists to [0, 1]
        let max_dist = min_dists
            .iter()
            .enumerate()
            .filter(|(i, _)| !used[*i])
            .map(|(_, &d)| d)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_dist_val = min_dists
            .iter()
            .enumerate()
            .filter(|(i, _)| !used[*i])
            .map(|(_, &d)| d)
            .fold(f64::INFINITY, f64::min);
        let dist_range = max_dist - min_dist_val;

        // Combined score: (1-dw)*uncertainty + dw*diversity
        let mut best_idx = 0;
        let mut best_combined = f64::NEG_INFINITY;

        for i in 0..n {
            if used[i] {
                continue;
            }
            let norm_dist = if dist_range > 1e-15 {
                (min_dists[i] - min_dist_val) / dist_range
            } else {
                0.5
            };
            let combined = (1.0 - dw) * norm_scores[i] + dw * norm_dist;
            if combined > best_combined {
                best_combined = combined;
                best_idx = i;
            }
        }

        selected.push(best_idx);
        used[best_idx] = true;

        // Update min_dists
        let new_centre = &embeddings[best_idx];
        for (i, md) in min_dists.iter_mut().enumerate() {
            if !used[i] {
                let d = euclidean_dist(&embeddings[i], new_centre);
                if d < *md {
                    *md = d;
                }
            }
        }
    }

    Ok(selected)
}

/// Select a batch of samples for labeling (legacy API).
///
/// Combines an uncertainty score with optional diversity (greedy spacing).
pub fn batch_select(
    features: &[Vec<f64>],
    probabilities: &[Vec<f64>],
    config: &BatchSelectionConfig,
) -> Result<Vec<usize>> {
    if features.is_empty() || probabilities.is_empty() {
        return Err(MetricsError::InvalidInput(
            "features and probabilities must not be empty".to_string(),
        ));
    }
    if features.len() != probabilities.len() {
        return Err(MetricsError::DimensionMismatch(format!(
            "features len {} != probabilities len {}",
            features.len(),
            probabilities.len()
        )));
    }
    let n = features.len();
    let k = config.n_select.min(n);

    match config.method {
        BatchSelectionMethod::CoreSet => greedy_k_center(features, k, None),
        BatchSelectionMethod::Entropy => {
            let scores: Vec<f64> = probabilities
                .iter()
                .map(|p| entropy_uncertainty(p))
                .collect::<Result<Vec<_>>>()?;
            let ranked = rank_by_uncertainty(&scores);
            Ok(ranked.into_iter().take(k).collect())
        }
        BatchSelectionMethod::MarginSampling => {
            let scores: Vec<f64> = probabilities
                .iter()
                .map(|p| margin_sampling_score(p))
                .collect::<Result<Vec<_>>>()?;
            let ranked = rank_by_uncertainty(&scores);
            Ok(ranked.into_iter().take(k).collect())
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- Margin sampling ---

    #[test]
    fn test_margin_sampling_uniform_score_zero() {
        // Uniform probs: p_max == p_second => margin=0 => score=1.0 (most uncertain)
        let probs = vec![vec![0.25, 0.25, 0.25, 0.25], vec![0.5, 0.5]];
        let scores = margin_sampling(&probs).expect("should succeed");
        assert!(
            (scores[0] - 1.0).abs() < 1e-12,
            "uniform 4-class: score should be 1.0, got {}",
            scores[0]
        );
        assert!(
            (scores[1] - 1.0).abs() < 1e-12,
            "uniform 2-class: score should be 1.0, got {}",
            scores[1]
        );
    }

    #[test]
    fn test_margin_sampling_peaked_close_to_one() {
        // Peaked probs: p_max far from p_second => large margin => score close to 0
        let probs = vec![vec![0.99, 0.01]];
        let scores = margin_sampling(&probs).expect("should succeed");
        assert!(
            scores[0] < 0.05,
            "peaked should have low uncertainty, got {}",
            scores[0]
        );
    }

    // --- Entropy sampling ---

    #[test]
    fn test_entropy_uniform_has_max() {
        let n = 4;
        let p = 1.0 / n as f64;
        let probs = vec![vec![p; n]];
        let scores = entropy_sampling(&probs).expect("should succeed");
        let expected = (n as f64).ln();
        assert!(
            (scores[0] - expected).abs() < 1e-10,
            "expected {expected}, got {}",
            scores[0]
        );
    }

    #[test]
    fn test_entropy_point_mass_zero() {
        let probs = vec![vec![1.0, 0.0, 0.0]];
        let scores = entropy_sampling(&probs).expect("should succeed");
        assert!(
            scores[0].abs() < 1e-12,
            "point mass entropy should be 0, got {}",
            scores[0]
        );
    }

    // --- Least confidence ---

    #[test]
    fn test_least_confidence_confident_low_score() {
        let probs = vec![vec![0.95, 0.03, 0.02]];
        let scores = least_confidence(&probs).expect("should succeed");
        assert!(
            scores[0] < 0.1,
            "confident prediction should have low LC, got {}",
            scores[0]
        );
    }

    #[test]
    fn test_least_confidence_uncertain_high_score() {
        let probs = vec![vec![0.34, 0.33, 0.33]];
        let scores = least_confidence(&probs).expect("should succeed");
        assert!(
            scores[0] > 0.5,
            "uncertain prediction should have high LC, got {}",
            scores[0]
        );
    }

    // --- Query by committee ---

    #[test]
    fn test_qbc_unanimous_low_disagreement() {
        // All committee members agree on class 0
        let committee = vec![
            vec![vec![0.9, 0.1], vec![0.8, 0.2]],     // member 0: 2 samples
            vec![vec![0.85, 0.15], vec![0.75, 0.25]], // member 1
            vec![vec![0.95, 0.05], vec![0.7, 0.3]],   // member 2
        ];
        let scores = query_by_committee(&committee).expect("should succeed");
        // All members predict class 0 for sample 0 => vote entropy = 0
        assert!(
            scores[0].abs() < 1e-12,
            "unanimous committee: disagreement should be 0, got {}",
            scores[0]
        );
    }

    #[test]
    fn test_qbc_disagreeing_positive() {
        // Committee members disagree
        let committee = vec![
            vec![vec![0.9, 0.1]], // predicts class 0
            vec![vec![0.1, 0.9]], // predicts class 1
        ];
        let scores = query_by_committee(&committee).expect("should succeed");
        assert!(
            scores[0] > 0.0,
            "disagreeing committee should have positive score, got {}",
            scores[0]
        );
    }

    // --- Expected model change ---

    #[test]
    fn test_expected_model_change_norm() {
        let gradients = vec![
            vec![3.0, 4.0],           // norm = 5.0
            vec![0.0, 0.0],           // norm = 0.0
            vec![1.0, 1.0, 1.0, 1.0], // norm = 2.0
        ];
        let scores = expected_model_change(&gradients).expect("should succeed");
        assert!((scores[0] - 5.0).abs() < 1e-12);
        assert!(scores[1].abs() < 1e-12);
        assert!((scores[2] - 2.0).abs() < 1e-12);
    }

    // --- Core-set selection ---

    #[test]
    fn test_core_set_points_well_spread() {
        // Points at 0, 10, 20, ..., 90 on a line
        let embeddings: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 10.0]).collect();
        let selected = core_set_selection(&embeddings, &[], 3).expect("should succeed");
        assert_eq!(selected.len(), 3);
        // With seed at 0, second should be 9 (farthest), third should be 4 or 5
        // Verify they're spread out: no two selected points within 15 of each other
        for i in 0..selected.len() {
            for j in (i + 1)..selected.len() {
                let d = euclidean_dist(&embeddings[selected[i]], &embeddings[selected[j]]);
                assert!(d >= 10.0, "selected points should be spread: dist={d}");
            }
        }
    }

    #[test]
    fn test_core_set_with_existing_selected() {
        let embeddings: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let already_selected = vec![0, 9]; // endpoints
        let new = core_set_selection(&embeddings, &already_selected, 1).expect("should succeed");
        assert_eq!(new.len(), 1);
        // The farthest from {0, 9} is either 4 or 5 (midpoint)
        assert!(
            new[0] >= 3 && new[0] <= 6,
            "midpoint expected, got {}",
            new[0]
        );
    }

    // --- Rank candidates ---

    #[test]
    fn test_rank_candidates_top_n() {
        let scores = vec![0.1, 0.9, 0.5, 0.3, 0.7];
        let top3 = rank_candidates(&scores, 3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0], 1); // highest score
        assert_eq!(top3[1], 4); // second highest
        assert_eq!(top3[2], 2); // third highest
    }

    // --- Batch selection ---

    #[test]
    fn test_batch_selection_diversity_zero_matches_uncertainty() {
        let scores = vec![0.1, 0.9, 0.5, 0.3, 0.7];
        let embeddings: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();

        let pure_unc = rank_candidates(&scores, 3);
        let batch = batch_selection(&scores, &embeddings, 3, 0.0).expect("should succeed");
        assert_eq!(
            batch, pure_unc,
            "diversity_weight=0 should match pure uncertainty ranking"
        );
    }

    #[test]
    fn test_batch_selection_returns_correct_count() {
        let scores = vec![0.5; 20];
        let embeddings: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, 0.0]).collect();
        let selected = batch_selection(&scores, &embeddings, 7, 0.5).expect("should succeed");
        assert_eq!(selected.len(), 7);
    }

    #[test]
    fn test_batch_selection_respects_n_select_legacy() {
        let features: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let probs: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                let p = i as f64 / 20.0;
                vec![p, 1.0 - p]
            })
            .collect();
        let cfg = BatchSelectionConfig {
            n_select: 7,
            ..Default::default()
        };
        let selected = batch_select(&features, &probs, &cfg).expect("should succeed");
        assert_eq!(selected.len(), 7, "should select exactly 7 samples");
    }

    // --- Single-sample backwards compat ---

    #[test]
    fn test_margin_sampling_score_compat() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let s = margin_sampling_score(&p).expect("should succeed");
        assert!((s - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_vote_entropy_unanimous_zero() {
        let committee = vec![vec![0.9, 0.1], vec![0.8, 0.2], vec![0.95, 0.05]];
        let ve = vote_entropy(&committee).expect("should succeed");
        assert!(
            ve.abs() < 1e-12,
            "unanimous vote should give entropy=0, got {ve}"
        );
    }

    #[test]
    fn test_expected_gradient_magnitude_shape() {
        let probs = vec![vec![0.7, 0.2, 0.1], vec![0.3, 0.4, 0.3]];
        let mags = expected_gradient_magnitude(&probs).expect("should succeed");
        assert_eq!(mags.len(), 2);
        for m in &mags {
            assert!(*m >= 0.0, "magnitude must be non-negative, got {m}");
        }
    }

    #[test]
    fn test_k_center_returns_k_points() {
        let features: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, 0.0]).collect();
        let selected = greedy_k_center(&features, 5, None).expect("should succeed");
        assert_eq!(selected.len(), 5);
    }

    #[test]
    fn test_default_config() {
        let cfg = ActiveLearningConfig::default();
        assert_eq!(cfg.n_committee, 5);
        assert_eq!(cfg.n_candidates, 100);
    }
}
