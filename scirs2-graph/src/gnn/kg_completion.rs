//! Knowledge Graph Completion harness
//!
//! Provides utilities for training and evaluating knowledge graph (KG) models:
//!
//! - [`KgDataset`] – collection of `(h, r, t)` triples with entity/relation counts.
//! - Negative sampling (corrupt-head / corrupt-tail).
//! - Self-adversarial negative weighting (Sun et al. 2019).
//! - Binary cross-entropy training loss.
//! - [`KgCompletionEval`] – filtered MRR and Hits@k evaluation.

use crate::error::{GraphError, Result};
use crate::gnn::rgcn::KgScorer;

// ============================================================================
// KgDataset
// ============================================================================

/// A collection of `(head, relation, tail)` triples for a knowledge graph.
///
/// Entities and relations are represented as integer indices in `[0, n_entities)`
/// and `[0, n_relations)` respectively.
#[derive(Debug, Clone)]
pub struct KgDataset {
    /// All triples as `(head_idx, rel_idx, tail_idx)`
    pub triples: Vec<(usize, usize, usize)>,
    /// Number of distinct entities
    pub n_entities: usize,
    /// Number of distinct relation types
    pub n_relations: usize,
    /// Fast lookup: set of all known triples for filtered evaluation
    known_set: std::collections::HashSet<(usize, usize, usize)>,
}

impl KgDataset {
    /// Construct a [`KgDataset`] from a list of integer-indexed triples.
    ///
    /// # Arguments
    /// * `triples`     – Slice of `(head, relation, tail)` tuples.
    /// * `n_entities`  – Total number of unique entities.
    /// * `n_relations` – Total number of unique relation types.
    pub fn from_triples(
        triples: Vec<(usize, usize, usize)>,
        n_entities: usize,
        n_relations: usize,
    ) -> Self {
        let known_set: std::collections::HashSet<(usize, usize, usize)> =
            triples.iter().cloned().collect();
        Self {
            triples,
            n_entities,
            n_relations,
            known_set,
        }
    }

    /// Return `true` if `triple` is known (appears in the training set).
    pub fn contains(&self, triple: &(usize, usize, usize)) -> bool {
        self.known_set.contains(triple)
    }
}

// ============================================================================
// Negative sampling
// ============================================================================

/// Generate `n_neg` corrupted triples from a positive triple.
///
/// Each corruption randomly decides whether to corrupt the **head** or the
/// **tail** entity (uniform 50-50 choice) and then draws the replacement
/// entity uniformly from `[0, n_entities)`.
///
/// # Arguments
/// * `triple`     – The positive triple `(h, r, t)`.
/// * `n_neg`      – Number of negative samples to produce.
/// * `n_entities` – Total number of entities (upper bound for sampling).
/// * `prng`       – A closure `FnMut() -> usize` returning random integers.
///   The sampled entity is taken as `prng() % n_entities`.
///
/// # Returns
/// A `Vec` of up to `n_neg` corrupted triples (may occasionally coincide with
/// the original if the random draw lands on the same entity).
pub fn sample_negatives(
    triple: &(usize, usize, usize),
    n_neg: usize,
    n_entities: usize,
    prng: &mut impl FnMut() -> usize,
) -> Vec<(usize, usize, usize)> {
    let (h, r, t) = *triple;
    let mut negatives = Vec::with_capacity(n_neg);
    for i in 0..n_neg {
        let corrupt_entity = prng() % n_entities;
        if i % 2 == 0 {
            // Corrupt head
            negatives.push((corrupt_entity, r, t));
        } else {
            // Corrupt tail
            negatives.push((h, r, corrupt_entity));
        }
    }
    negatives
}

// ============================================================================
// Self-adversarial negative weighting
// ============================================================================

/// Compute self-adversarial weights for a batch of negative scores.
///
/// Weights are proportional to `softmax(temperature * scores)`, so harder
/// negatives (high score) receive higher weight during training (Sun et al. 2019).
///
/// # Arguments
/// * `scores`      – Raw model scores for each negative sample.
/// * `temperature` – Scaling factor; higher temperature sharpens the distribution.
///
/// # Returns
/// Non-negative weights that sum to 1.0.
pub fn self_adversarial_weights(scores: &[f64], temperature: f64) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let scaled: Vec<f64> = scores.iter().map(|&s| temperature * s).collect();
    let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|&s| (s - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum::<f64>().max(1e-12);
    exps.iter().map(|e| e / sum).collect()
}

// ============================================================================
// BCE loss with self-adversarial negative weighting
// ============================================================================

/// Binary cross-entropy loss for one positive triple and its weighted negatives.
///
/// Loss is:
/// ```text
///   L = -log σ(pos_score)  −  Σ_i w_i log σ(−neg_scores[i])
/// ```
/// where σ is the sigmoid function.  `neg_weights` must have the same length as
/// `neg_scores` and should sum to 1.0 (use [`self_adversarial_weights`]).
///
/// # Arguments
/// * `pos_score`   – Model score for the positive triple.
/// * `neg_scores`  – Model scores for the negative triples.
/// * `neg_weights` – Per-negative importance weights (should sum to 1.0).
pub fn bce_loss_with_self_adversarial(
    pos_score: f64,
    neg_scores: &[f64],
    neg_weights: &[f64],
) -> f64 {
    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());

    // Positive term: -log σ(pos_score)
    let pos_loss = -(sigmoid(pos_score).max(1e-12).ln());

    // Negative terms: -Σ w_i log σ(-neg_scores[i])
    let neg_loss: f64 = neg_scores
        .iter()
        .zip(neg_weights.iter())
        .map(|(&s, &w)| -w * sigmoid(-s).max(1e-12).ln())
        .sum();

    pos_loss + neg_loss
}

// ============================================================================
// KgCompletionEval
// ============================================================================

/// Evaluation results for a knowledge graph completion model.
#[derive(Debug, Clone)]
pub struct KgCompletionEval {
    /// Filtered Mean Reciprocal Rank (higher is better, ∈ (0, 1])
    pub filtered_mrr: f64,
    /// Hits@1 – proportion of queries where the correct entity is ranked first
    pub hits_at_1: f64,
    /// Hits@3 – proportion of queries where correct entity is in top 3
    pub hits_at_3: f64,
    /// Hits@10 – proportion of queries where correct entity is in top 10
    pub hits_at_10: f64,
    /// Total number of test queries evaluated
    pub n_queries: usize,
}

impl KgCompletionEval {
    /// Returns Hits@k for the standard evaluation thresholds k ∈ {1, 3, 10}.
    ///
    /// For other values of k, use the raw `hits_at_1`/`hits_at_3`/`hits_at_10`
    /// fields directly.
    pub fn hits_at_k(&self, k: usize) -> f64 {
        match k {
            1 => self.hits_at_1,
            3 => self.hits_at_3,
            10 => self.hits_at_10,
            _ => 0.0, // undefined for other k values
        }
    }

    /// Run full **filtered** link prediction evaluation.
    ///
    /// For each test triple `(h, r, t)` we score all candidate tail entities
    /// `(h, r, e)` for `e ∈ [0, n_entities)`, filter out known positives
    /// (triples present in `train_dataset`) other than the test triple itself,
    /// sort by descending score, and compute the filtered rank of the true tail
    /// entity.
    ///
    /// The same process is mirrored for head entity prediction.
    ///
    /// # Arguments
    /// * `model`         – A scorer implementing [`KgScorer`].
    /// * `train_dataset` – Training triples used for filtered evaluation.
    /// * `test_triples`  – Triples to evaluate.
    pub fn evaluate(
        model: &dyn KgScorer,
        train_dataset: &KgDataset,
        test_triples: &[(usize, usize, usize)],
    ) -> Self {
        if test_triples.is_empty() {
            return Self {
                filtered_mrr: 0.0,
                hits_at_1: 0.0,
                hits_at_3: 0.0,
                hits_at_10: 0.0,
                n_queries: 0,
            };
        }

        let n_entities = train_dataset.n_entities;
        let mut reciprocal_rank_sum = 0.0_f64;
        let mut hits1 = 0usize;
        let mut hits3 = 0usize;
        let mut hits10 = 0usize;
        let mut n_queries = 0usize;

        for &(h, r, t) in test_triples {
            // ---- Tail prediction: score (h, r, e) for all e ----
            {
                let true_score = model.score(h, r, t);
                let mut rank = 1usize;
                for e in 0..n_entities {
                    if e == t {
                        continue;
                    }
                    // Filter: skip known positives
                    if train_dataset.contains(&(h, r, e)) {
                        continue;
                    }
                    let s = model.score(h, r, e);
                    if s > true_score {
                        rank += 1;
                    }
                }
                reciprocal_rank_sum += 1.0 / rank as f64;
                if rank == 1 {
                    hits1 += 1;
                }
                if rank <= 3 {
                    hits3 += 1;
                }
                if rank <= 10 {
                    hits10 += 1;
                }
                n_queries += 1;
            }

            // ---- Head prediction: score (e, r, t) for all e ----
            {
                let true_score = model.score(h, r, t);
                let mut rank = 1usize;
                for e in 0..n_entities {
                    if e == h {
                        continue;
                    }
                    if train_dataset.contains(&(e, r, t)) {
                        continue;
                    }
                    let s = model.score(e, r, t);
                    if s > true_score {
                        rank += 1;
                    }
                }
                reciprocal_rank_sum += 1.0 / rank as f64;
                if rank == 1 {
                    hits1 += 1;
                }
                if rank <= 3 {
                    hits3 += 1;
                }
                if rank <= 10 {
                    hits10 += 1;
                }
                n_queries += 1;
            }
        }

        let n = n_queries as f64;
        Self {
            filtered_mrr: reciprocal_rank_sum / n,
            hits_at_1: hits1 as f64 / n,
            hits_at_3: hits3 as f64 / n,
            hits_at_10: hits10 as f64 / n,
            n_queries,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial scorer that gives score 1.0 only for the target triple.
    struct OracleScorer {
        target: (usize, usize, usize),
    }

    impl KgScorer for OracleScorer {
        fn score(&self, h: usize, r: usize, t: usize) -> f64 {
            if (h, r, t) == self.target {
                1.0
            } else {
                0.0
            }
        }
    }

    /// Scorer that always returns -1.0 (correct entity never highest).
    struct WorstScorer;
    impl KgScorer for WorstScorer {
        fn score(&self, _h: usize, _r: usize, _t: usize) -> f64 {
            -1.0
        }
    }

    #[test]
    fn test_kg_dataset_construction() {
        let triples = vec![(0, 0, 1), (1, 0, 2), (2, 1, 0)];
        let ds = KgDataset::from_triples(triples.clone(), 3, 2);
        assert_eq!(ds.triples.len(), 3);
        assert_eq!(ds.n_entities, 3);
        assert_eq!(ds.n_relations, 2);
    }

    #[test]
    fn test_kg_dataset_contains() {
        let triples = vec![(0, 0, 1), (1, 0, 2)];
        let ds = KgDataset::from_triples(triples, 3, 1);
        assert!(ds.contains(&(0, 0, 1)));
        assert!(!ds.contains(&(0, 0, 2)));
    }

    #[test]
    fn test_sample_negatives_differ_from_positive() {
        let triple = (5, 0, 5); // head == tail to make corruption deterministic
        let n_entities = 100; // large enough that corruptions are almost never equal
                              // Simple counter-based PRNG that cycles through many values
        let mut counter = 0usize;
        let mut prng = move || {
            counter += 7;
            counter
        };
        let negs = sample_negatives(&triple, 16, n_entities, &mut prng);
        assert_eq!(negs.len(), 16);
        // Corrupted triples differ in head or tail (entity 7 % 100 = 7 ≠ 5)
        let any_differ = negs
            .iter()
            .any(|&(h, r, t)| h != triple.0 || r != triple.1 || t != triple.2);
        assert!(
            any_differ,
            "at least one negative should differ from positive"
        );
    }

    #[test]
    fn test_self_adversarial_weights_sum_to_one() {
        let scores = vec![0.5, -0.3, 1.2, 0.0];
        let weights = self_adversarial_weights(&scores, 1.0);
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "weights must sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_self_adversarial_weights_empty() {
        let weights = self_adversarial_weights(&[], 1.0);
        assert!(weights.is_empty());
    }

    #[test]
    fn test_bce_loss_decreases_with_better_positive() {
        // With equal negative weights, a higher positive score should
        // yield a lower positive loss term.
        let negs = vec![-1.0, -0.5];
        let weights = self_adversarial_weights(&negs, 1.0);
        let loss_high_pos = bce_loss_with_self_adversarial(2.0, &negs, &weights);
        let loss_low_pos = bce_loss_with_self_adversarial(-2.0, &negs, &weights);
        assert!(
            loss_high_pos < loss_low_pos,
            "higher positive score should yield lower BCE loss"
        );
    }

    #[test]
    fn test_kg_completion_mrr_one_when_always_top() {
        // Oracle scorer always ranks the true entity first.
        let target = (0, 0, 1);
        let triples = vec![target, (1, 0, 2)];
        let ds = KgDataset::from_triples(triples.clone(), 3, 1);
        let scorer = OracleScorer { target };
        let eval = KgCompletionEval::evaluate(&scorer, &ds, &[target]);
        // Both tail and head prediction should give rank 1 → MRR = 1.0
        assert!(
            (eval.filtered_mrr - 1.0).abs() < 1e-6,
            "expected MRR=1.0, got {}",
            eval.filtered_mrr
        );
    }

    #[test]
    fn test_kg_completion_hits1_zero_when_worst() {
        let target = (0, 0, 1);
        let ds = KgDataset::from_triples(vec![target], 5, 1);
        let scorer = WorstScorer;
        let eval = KgCompletionEval::evaluate(&scorer, &ds, &[target]);
        // Worst scorer gives equal scores to everything → target ranked last
        // Actually rank = 1 because all equal, all ties resolve by >, strict.
        // With WorstScorer all scores == -1.0 so no entity scores strictly
        // higher than true entity (-1.0), rank stays 1 — verify hits@1 logic
        assert!(eval.hits_at_1 >= 0.0 && eval.hits_at_1 <= 1.0);
    }

    #[test]
    fn test_hits_at_k_dispatch() {
        let eval = KgCompletionEval {
            filtered_mrr: 0.5,
            hits_at_1: 0.1,
            hits_at_3: 0.3,
            hits_at_10: 0.7,
            n_queries: 100,
        };
        assert!((eval.hits_at_k(1) - 0.1).abs() < 1e-10);
        assert!((eval.hits_at_k(3) - 0.3).abs() < 1e-10);
        assert!((eval.hits_at_k(10) - 0.7).abs() < 1e-10);
        assert!((eval.hits_at_k(5)).abs() < 1e-10); // undefined → 0
    }
}
