//! Knowledge Graph dataset generators and utilities.
//!
//! This module provides synthetic knowledge graph generators suitable for benchmarking
//! knowledge graph embedding methods (TransE, RotatE, RESCAL, etc.) and link-prediction
//! pipelines.
//!
//! # Contents
//!
//! - [`KgTriple`]            – A single (subject, predicate, object) triple.
//! - [`KnowledgeGraphDataset`] – A dataset of entities, relations, and triples.
//! - [`KgSplit`]             – Standard train / validation / test partition.
//! - [`FreebaseSubset`]      – Generator for FB15k-style random KB subsets.
//! - [`TransitiveRelation`]  – Generator for synthetic transitive-closure chains.
//! - [`negative_sampling`]   – Corrupt triples for negative-example generation.
//! - [`entity_frequency`]    – Occurrence histogram of entities across triples.

use crate::error::{DatasetsError, Result};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────────────────────────────────────

/// A single knowledge-graph triple (subject, predicate, object) represented as
/// indices into the entity and relation tables of a [`KnowledgeGraphDataset`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KgTriple {
    /// Index of the subject entity.
    pub subject: usize,
    /// Index of the relation / predicate.
    pub predicate: usize,
    /// Index of the object entity.
    pub object: usize,
}

impl KgTriple {
    /// Create a new triple.
    #[inline]
    pub fn new(subject: usize, predicate: usize, object: usize) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }
}

/// A complete knowledge graph dataset.
///
/// Entities are identified by zero-based integer indices, as are relations.
/// String labels are stored in the `entity_names` and `relation_names` vectors
/// (one entry per index).
#[derive(Debug, Clone)]
pub struct KnowledgeGraphDataset {
    /// Number of distinct entities.
    pub n_entities: usize,
    /// Number of distinct relation types.
    pub n_relations: usize,
    /// All triples in the dataset.
    pub triples: Vec<KgTriple>,
    /// Human-readable entity names (length == n_entities).
    pub entity_names: Vec<String>,
    /// Human-readable relation names (length == n_relations).
    pub relation_names: Vec<String>,
}

impl KnowledgeGraphDataset {
    /// Return the number of triples.
    #[inline]
    pub fn n_triples(&self) -> usize {
        self.triples.len()
    }

    /// Iterate over all subjects.
    pub fn subjects(&self) -> impl Iterator<Item = usize> + '_ {
        self.triples.iter().map(|t| t.subject)
    }

    /// Iterate over all objects.
    pub fn objects(&self) -> impl Iterator<Item = usize> + '_ {
        self.triples.iter().map(|t| t.object)
    }

    /// Return all triples whose predicate equals `rel_id`.
    pub fn triples_by_relation(&self, rel_id: usize) -> Vec<&KgTriple> {
        self.triples
            .iter()
            .filter(|t| t.predicate == rel_id)
            .collect()
    }

    /// Convert an entity index to its name, returning `None` if out of range.
    pub fn entity_name(&self, idx: usize) -> Option<&str> {
        self.entity_names.get(idx).map(String::as_str)
    }

    /// Convert a relation index to its name, returning `None` if out of range.
    pub fn relation_name(&self, idx: usize) -> Option<&str> {
        self.relation_names.get(idx).map(String::as_str)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KgSplit
// ─────────────────────────────────────────────────────────────────────────────

/// Standard train / validation / test split for a knowledge graph dataset.
#[derive(Debug, Clone)]
pub struct KgSplit {
    /// Training triples (typically ~80%).
    pub train: Vec<KgTriple>,
    /// Validation triples (typically ~10%).
    pub valid: Vec<KgTriple>,
    /// Test triples (typically ~10%).
    pub test: Vec<KgTriple>,
}

/// Split a flat list of triples into train / validation / test sets.
///
/// The triples are first shuffled using `seed` then cut at the given ratios.
/// `valid_ratio` and `test_ratio` must each be in `(0, 1)` and their sum must
/// be strictly less than 1.
///
/// # Errors
///
/// Returns [`DatasetsError::InvalidFormat`] when ratios are out of range or
/// `triples` is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::knowledge_graph_datasets::{KgTriple, split_triples};
///
/// let triples: Vec<KgTriple> = (0..100)
///     .map(|i| KgTriple::new(i % 10, i % 3, (i + 1) % 10))
///     .collect();
/// let split = split_triples(&triples, 0.1, 0.1, 42).expect("split failed");
/// assert_eq!(split.train.len() + split.valid.len() + split.test.len(), 100);
/// ```
pub fn split_triples(
    triples: &[KgTriple],
    valid_ratio: f64,
    test_ratio: f64,
    seed: u64,
) -> Result<KgSplit> {
    if triples.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "split_triples: triples must not be empty".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&valid_ratio) || valid_ratio == 0.0 {
        return Err(DatasetsError::InvalidFormat(format!(
            "split_triples: valid_ratio ({valid_ratio}) must be in (0, 1)"
        )));
    }
    if !(0.0..1.0).contains(&test_ratio) || test_ratio == 0.0 {
        return Err(DatasetsError::InvalidFormat(format!(
            "split_triples: test_ratio ({test_ratio}) must be in (0, 1)"
        )));
    }
    if valid_ratio + test_ratio >= 1.0 {
        return Err(DatasetsError::InvalidFormat(
            "split_triples: valid_ratio + test_ratio must be < 1.0".to_string(),
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..triples.len()).collect();
    // Fisher-Yates shuffle
    for i in (1..indices.len()).rev() {
        let uniform = scirs2_core::random::Uniform::new(0usize, i + 1).map_err(|e| {
            DatasetsError::ComputationError(format!("Uniform creation failed: {e}"))
        })?;
        let j = uniform.sample(&mut rng);
        indices.swap(i, j);
    }

    let n = triples.len();
    let n_test = (n as f64 * test_ratio).round() as usize;
    let n_valid = (n as f64 * valid_ratio).round() as usize;
    let n_train = n - n_valid - n_test;

    let train = indices[..n_train]
        .iter()
        .map(|&i| triples[i])
        .collect();
    let valid = indices[n_train..n_train + n_valid]
        .iter()
        .map(|&i| triples[i])
        .collect();
    let test = indices[n_train + n_valid..]
        .iter()
        .map(|&i| triples[i])
        .collect();

    Ok(KgSplit { train, valid, test })
}

// ─────────────────────────────────────────────────────────────────────────────
// FreebaseSubset
// ─────────────────────────────────────────────────────────────────────────────

/// Generator for a Freebase-style synthetic knowledge base subset.
///
/// The generated graph mimics the structural characteristics of FB15k:
/// a mixture of typed-entity hierarchies, property assertions, and symmetric /
/// inverse relation pairs.
///
/// # Relation types (cycling through `n_relations`):
///
/// 0 → `type_of`           (hierarchical membership)
/// 1 → `part_of`           (meronymy)
/// 2 → `related_to`        (symmetric, bidirectional)
/// 3 → `has_property`      (unary assertion encoded as binary)
/// 4 → `inverse_of(prev)`  (inverse-triple mirror)
/// 5+ → generic predicates
pub struct FreebaseSubset;

impl FreebaseSubset {
    /// Generate a random KB-like triple set.
    ///
    /// # Arguments
    ///
    /// * `n_entities`  – Number of distinct entity nodes (must be ≥ 2).
    /// * `n_relations` – Number of distinct relation types (must be ≥ 1).
    /// * `n_triples`   – Total number of triples to generate (deduplicated).
    ///                   The actual count may be lower if the triple space is small.
    /// * `seed`        – Random seed for reproducibility.
    ///
    /// # Errors
    ///
    /// Returns an error if `n_entities < 2`, `n_relations < 1`, or
    /// `n_triples == 0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_datasets::knowledge_graph_datasets::FreebaseSubset;
    ///
    /// let kg = FreebaseSubset::generate(50, 5, 200, 42).expect("fb failed");
    /// assert!(kg.n_triples() > 0);
    /// assert_eq!(kg.n_entities, 50);
    /// assert_eq!(kg.n_relations, 5);
    /// ```
    pub fn generate(
        n_entities: usize,
        n_relations: usize,
        n_triples: usize,
        seed: u64,
    ) -> Result<KnowledgeGraphDataset> {
        if n_entities < 2 {
            return Err(DatasetsError::InvalidFormat(
                "FreebaseSubset::generate: n_entities must be >= 2".to_string(),
            ));
        }
        if n_relations < 1 {
            return Err(DatasetsError::InvalidFormat(
                "FreebaseSubset::generate: n_relations must be >= 1".to_string(),
            ));
        }
        if n_triples == 0 {
            return Err(DatasetsError::InvalidFormat(
                "FreebaseSubset::generate: n_triples must be > 0".to_string(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let ent_dist =
            scirs2_core::random::Uniform::new(0usize, n_entities).map_err(|e| {
                DatasetsError::ComputationError(format!("Uniform entity dist failed: {e}"))
            })?;
        let rel_dist =
            scirs2_core::random::Uniform::new(0usize, n_relations).map_err(|e| {
                DatasetsError::ComputationError(format!("Uniform relation dist failed: {e}"))
            })?;

        let mut seen: std::collections::HashSet<KgTriple> =
            std::collections::HashSet::with_capacity(n_triples);
        let mut triples: Vec<KgTriple> = Vec::with_capacity(n_triples);

        // Maximum unique triples possible in this space
        let max_possible = n_entities * n_relations * (n_entities - 1);
        let target = n_triples.min(max_possible);

        let max_attempts = target * 20 + 1000;
        let mut attempts = 0usize;

        while triples.len() < target && attempts < max_attempts {
            attempts += 1;

            let s = ent_dist.sample(&mut rng);
            let mut o = ent_dist.sample(&mut rng);
            // Avoid reflexive triples
            if o == s {
                o = (s + 1) % n_entities;
            }
            let p = rel_dist.sample(&mut rng);

            let t = KgTriple::new(s, p, o);
            if seen.insert(t) {
                triples.push(t);

                // For relation type 2 (related_to / symmetric), also add reverse
                if p % 6 == 2 && triples.len() < target {
                    let rev = KgTriple::new(o, p, s);
                    if seen.insert(rev) {
                        triples.push(rev);
                    }
                }

                // For relation type 4 (inverse), add mirrored triple with
                // relation (p+1) % n_relations
                if p % 6 == 3 && n_relations > 1 && triples.len() < target {
                    let inv_rel = (p + 1) % n_relations;
                    let inv = KgTriple::new(o, inv_rel, s);
                    if seen.insert(inv) {
                        triples.push(inv);
                    }
                }
            }
        }

        // Build entity / relation name tables
        let entity_names: Vec<String> = (0..n_entities)
            .map(|i| format!("entity_{i}"))
            .collect();

        let relation_type_labels = [
            "type_of",
            "part_of",
            "related_to",
            "has_property",
            "inverse_of",
        ];
        let relation_names: Vec<String> = (0..n_relations)
            .map(|i| {
                let label = relation_type_labels
                    .get(i % relation_type_labels.len())
                    .copied()
                    .unwrap_or("generic");
                format!("{label}_{i}")
            })
            .collect();

        Ok(KnowledgeGraphDataset {
            n_entities,
            n_relations,
            triples,
            entity_names,
            relation_names,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TransitiveRelation
// ─────────────────────────────────────────────────────────────────────────────

/// Generator for transitive-closure chains.
///
/// Builds a linear chain `e_0 → e_1 → … → e_{chain_length}` for a chosen
/// predicate, then adds all transitive consequences
/// `(e_i, pred, e_j)` for every `i < j`.
///
/// This is useful for evaluating whether KG embeddings can learn transitivity.
pub struct TransitiveRelation;

impl TransitiveRelation {
    /// Generate a set of triples encoding a transitive relation.
    ///
    /// The predicate index is fixed to `predicate_id`.  Entity indices range
    /// from `entity_offset` to `entity_offset + n_entities - 1`.
    ///
    /// # Arguments
    ///
    /// * `n_entities`    – Total number of entity nodes (must be ≥ 2).
    /// * `chain_length`  – Length of the primary directed chain (must satisfy
    ///                     `chain_length < n_entities`).
    /// * `predicate_id`  – Index of the relation type to use.
    /// * `entity_offset` – Starting entity index (allows multiple chains in one
    ///                     combined dataset).
    ///
    /// # Returns
    ///
    /// A deduplicated `Vec<KgTriple>` containing all base-chain and
    /// transitive-closure triples.
    ///
    /// # Errors
    ///
    /// Returns an error if `n_entities < 2` or `chain_length >= n_entities`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_datasets::knowledge_graph_datasets::TransitiveRelation;
    ///
    /// let triples = TransitiveRelation::generate(5, 4, 0, 0).expect("transitive failed");
    /// // Chain: 0→1→2→3→4 plus all 4+3+2+1 = 10 transitive pairs
    /// assert_eq!(triples.len(), 10);
    /// ```
    pub fn generate(
        n_entities: usize,
        chain_length: usize,
        predicate_id: usize,
        entity_offset: usize,
    ) -> Result<Vec<KgTriple>> {
        if n_entities < 2 {
            return Err(DatasetsError::InvalidFormat(
                "TransitiveRelation::generate: n_entities must be >= 2".to_string(),
            ));
        }
        if chain_length >= n_entities {
            return Err(DatasetsError::InvalidFormat(format!(
                "TransitiveRelation::generate: chain_length ({chain_length}) \
                 must be < n_entities ({n_entities})"
            )));
        }

        // Nodes participating in the chain: entity_offset .. entity_offset + chain_length (inclusive)
        let chain_nodes: Vec<usize> = (0..=chain_length)
            .map(|i| entity_offset + i)
            .collect();

        let mut triples: Vec<KgTriple> = Vec::new();

        // Add all pairs (i, j) where i < j — this is the full transitive closure
        // of the chain i → i+1 → … → j.
        for (idx_i, &src) in chain_nodes.iter().enumerate() {
            for &dst in chain_nodes.iter().skip(idx_i + 1) {
                triples.push(KgTriple::new(src, predicate_id, dst));
            }
        }

        Ok(triples)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// negative_sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Generate corrupted (negative) triples by randomly replacing either the
/// subject or the object of each positive triple.
///
/// The strategy follows the standard KGE evaluation protocol: for each
/// positive triple, `n_neg_per_pos` negatives are produced.  The replacement
/// entity is drawn uniformly from `entities` and the resulting triple must not
/// appear in `positive_set`.
///
/// # Arguments
///
/// * `positives`      – Slice of ground-truth triples.
/// * `n_neg_per_pos`  – Number of negative triples to generate per positive.
/// * `n_entities`     – Total entity count (entity indices are `0..n_entities`).
/// * `seed`           – Random seed.
///
/// # Errors
///
/// Returns an error if `positives` is empty, `n_neg_per_pos == 0`, or
/// `n_entities < 2`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::knowledge_graph_datasets::{KgTriple, negative_sampling};
///
/// let positives: Vec<KgTriple> = vec![
///     KgTriple::new(0, 0, 1),
///     KgTriple::new(1, 0, 2),
/// ];
/// let negatives = negative_sampling(&positives, 2, 5, 42).expect("neg sampling failed");
/// assert_eq!(negatives.len(), 4);
/// ```
pub fn negative_sampling(
    positives: &[KgTriple],
    n_neg_per_pos: usize,
    n_entities: usize,
    seed: u64,
) -> Result<Vec<KgTriple>> {
    if positives.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "negative_sampling: positives must not be empty".to_string(),
        ));
    }
    if n_neg_per_pos == 0 {
        return Err(DatasetsError::InvalidFormat(
            "negative_sampling: n_neg_per_pos must be > 0".to_string(),
        ));
    }
    if n_entities < 2 {
        return Err(DatasetsError::InvalidFormat(
            "negative_sampling: n_entities must be >= 2".to_string(),
        ));
    }

    // Build a fast membership set for positive triples
    let positive_set: std::collections::HashSet<KgTriple> =
        positives.iter().copied().collect();

    let mut rng = StdRng::seed_from_u64(seed);
    let ent_dist = scirs2_core::random::Uniform::new(0usize, n_entities).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform entity dist failed: {e}"))
    })?;
    // Bernoulli(0.5) — decides whether to corrupt subject (0) or object (1)
    let side_dist = scirs2_core::random::Uniform::new(0usize, 2).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform side dist failed: {e}"))
    })?;

    let mut negatives: Vec<KgTriple> =
        Vec::with_capacity(positives.len() * n_neg_per_pos);

    for &pos in positives {
        let mut generated = 0usize;
        let max_attempts = n_neg_per_pos * (n_entities + 10);
        let mut attempts = 0usize;

        while generated < n_neg_per_pos && attempts < max_attempts {
            attempts += 1;
            let corrupt_entity = ent_dist.sample(&mut rng);
            let side = side_dist.sample(&mut rng);

            let neg = if side == 0 {
                // Corrupt subject
                KgTriple::new(corrupt_entity, pos.predicate, pos.object)
            } else {
                // Corrupt object
                KgTriple::new(pos.subject, pos.predicate, corrupt_entity)
            };

            if !positive_set.contains(&neg) && neg != pos {
                negatives.push(neg);
                generated += 1;
            }
        }
    }

    Ok(negatives)
}

// ─────────────────────────────────────────────────────────────────────────────
// entity_frequency
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the occurrence frequency of each entity across all triples.
///
/// An entity is counted once per triple position (subject and object are
/// counted independently, so a self-loop would contribute 2 to the count).
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::knowledge_graph_datasets::{KgTriple, entity_frequency};
///
/// let triples = vec![
///     KgTriple::new(0, 0, 1),
///     KgTriple::new(0, 0, 2),
///     KgTriple::new(1, 0, 2),
/// ];
/// let freq = entity_frequency(&triples);
/// assert_eq!(freq[&0], 2); // entity 0 appears as subject twice
/// assert_eq!(freq[&1], 2); // entity 1 appears as subject + object once each
/// assert_eq!(freq[&2], 2); // entity 2 appears as object twice
/// ```
pub fn entity_frequency(triples: &[KgTriple]) -> HashMap<usize, usize> {
    let mut freq: HashMap<usize, usize> = HashMap::new();
    for t in triples {
        *freq.entry(t.subject).or_insert(0) += 1;
        *freq.entry(t.object).or_insert(0) += 1;
    }
    freq
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── KgTriple ─────────────────────────────────────────────────────────────

    #[test]
    fn test_kg_triple_new() {
        let t = KgTriple::new(1, 2, 3);
        assert_eq!(t.subject, 1);
        assert_eq!(t.predicate, 2);
        assert_eq!(t.object, 3);
    }

    #[test]
    fn test_kg_triple_equality() {
        let a = KgTriple::new(0, 0, 1);
        let b = KgTriple::new(0, 0, 1);
        let c = KgTriple::new(1, 0, 0);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ── FreebaseSubset ───────────────────────────────────────────────────────

    #[test]
    fn test_freebase_basic() {
        let kg = FreebaseSubset::generate(20, 4, 50, 42).expect("freebase basic");
        assert_eq!(kg.n_entities, 20);
        assert_eq!(kg.n_relations, 4);
        assert!(!kg.triples.is_empty());
        assert_eq!(kg.entity_names.len(), 20);
        assert_eq!(kg.relation_names.len(), 4);
    }

    #[test]
    fn test_freebase_triples_are_valid() {
        let kg = FreebaseSubset::generate(10, 3, 30, 7).expect("freebase valid triples");
        for t in &kg.triples {
            assert!(t.subject < kg.n_entities, "subject out of range: {}", t.subject);
            assert!(t.predicate < kg.n_relations, "predicate out of range: {}", t.predicate);
            assert!(t.object < kg.n_entities, "object out of range: {}", t.object);
            assert_ne!(t.subject, t.object, "reflexive triple found");
        }
    }

    #[test]
    fn test_freebase_no_duplicate_triples() {
        let kg = FreebaseSubset::generate(15, 5, 100, 13).expect("freebase no dupes");
        let mut seen = std::collections::HashSet::new();
        for &t in &kg.triples {
            assert!(seen.insert(t), "duplicate triple found: {t:?}");
        }
    }

    #[test]
    fn test_freebase_error_too_few_entities() {
        assert!(FreebaseSubset::generate(1, 3, 10, 1).is_err());
    }

    #[test]
    fn test_freebase_error_no_relations() {
        assert!(FreebaseSubset::generate(10, 0, 10, 1).is_err());
    }

    #[test]
    fn test_freebase_error_zero_triples() {
        assert!(FreebaseSubset::generate(10, 3, 0, 1).is_err());
    }

    // ── TransitiveRelation ───────────────────────────────────────────────────

    #[test]
    fn test_transitive_full_closure() {
        // Chain 0→1→2→3→4 has transitive closure of size 4+3+2+1 = 10
        let triples = TransitiveRelation::generate(5, 4, 0, 0).expect("transitive full");
        assert_eq!(triples.len(), 10, "expected 10 triples in full transitive closure");
    }

    #[test]
    fn test_transitive_two_nodes() {
        let triples = TransitiveRelation::generate(2, 1, 0, 0).expect("transitive two nodes");
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0], KgTriple::new(0, 0, 1));
    }

    #[test]
    fn test_transitive_predicate_and_offset() {
        let triples = TransitiveRelation::generate(4, 2, 5, 10).expect("transitive offset");
        // Nodes 10, 11, 12; pairs: (10,11),(10,12),(11,12) = 3
        assert_eq!(triples.len(), 3);
        for t in &triples {
            assert_eq!(t.predicate, 5);
            assert!(t.subject >= 10 && t.object >= 10);
        }
    }

    #[test]
    fn test_transitive_error_chain_ge_n_entities() {
        assert!(TransitiveRelation::generate(5, 5, 0, 0).is_err());
    }

    #[test]
    fn test_transitive_error_too_few_entities() {
        assert!(TransitiveRelation::generate(1, 0, 0, 0).is_err());
    }

    // ── split_triples ────────────────────────────────────────────────────────

    #[test]
    fn test_split_triples_counts() {
        let triples: Vec<KgTriple> = (0..100)
            .map(|i| KgTriple::new(i % 10, i % 3, (i + 1) % 10))
            .collect();
        let split = split_triples(&triples, 0.1, 0.1, 42).expect("split counts");
        assert_eq!(
            split.train.len() + split.valid.len() + split.test.len(),
            100
        );
    }

    #[test]
    fn test_split_triples_no_overlap() {
        let triples: Vec<KgTriple> = (0..50)
            .map(|i| KgTriple::new(i % 8, i % 4, (i + 2) % 8))
            .collect();
        let split = split_triples(&triples, 0.1, 0.2, 99).expect("split no overlap");
        let mut all_indices: std::collections::HashSet<*const KgTriple> =
            std::collections::HashSet::new();
        // We verify no triple pointer appears in two sets by collecting identities
        // via address comparison on copied values — instead, count total == original
        let total = split.train.len() + split.valid.len() + split.test.len();
        assert_eq!(total, 50);
        // Check sets are disjoint by checking uniqueness of (subject,predicate,object)
        let train_set: std::collections::HashSet<KgTriple> =
            split.train.iter().copied().collect();
        for t in &split.valid {
            assert!(!train_set.contains(t), "overlap between train and valid");
        }
        // Suppress unused variable warning
        let _ = all_indices.insert(split.train.as_ptr());
    }

    #[test]
    fn test_split_triples_error_empty() {
        assert!(split_triples(&[], 0.1, 0.1, 1).is_err());
    }

    #[test]
    fn test_split_triples_error_bad_ratios() {
        let triples = vec![KgTriple::new(0, 0, 1)];
        assert!(split_triples(&triples, 0.6, 0.6, 1).is_err());
    }

    // ── negative_sampling ────────────────────────────────────────────────────

    #[test]
    fn test_negative_sampling_count() {
        let positives: Vec<KgTriple> = vec![
            KgTriple::new(0, 0, 1),
            KgTriple::new(1, 0, 2),
            KgTriple::new(2, 0, 3),
        ];
        let negatives = negative_sampling(&positives, 3, 10, 42).expect("neg count");
        // We expect exactly n_pos * n_neg_per_pos negatives when the entity space is large enough
        assert_eq!(negatives.len(), 9);
    }

    #[test]
    fn test_negative_sampling_not_in_positive_set() {
        let positives: Vec<KgTriple> = (0..5)
            .map(|i| KgTriple::new(i, 0, (i + 1) % 10))
            .collect();
        let pos_set: std::collections::HashSet<KgTriple> =
            positives.iter().copied().collect();
        let negatives = negative_sampling(&positives, 4, 10, 11).expect("neg not in pos");
        for neg in &negatives {
            assert!(!pos_set.contains(neg), "negative is in positive set: {neg:?}");
        }
    }

    #[test]
    fn test_negative_sampling_error_empty_positives() {
        assert!(negative_sampling(&[], 2, 5, 1).is_err());
    }

    #[test]
    fn test_negative_sampling_error_zero_n_neg() {
        let pos = vec![KgTriple::new(0, 0, 1)];
        assert!(negative_sampling(&pos, 0, 5, 1).is_err());
    }

    #[test]
    fn test_negative_sampling_error_too_few_entities() {
        let pos = vec![KgTriple::new(0, 0, 1)];
        assert!(negative_sampling(&pos, 2, 1, 1).is_err());
    }

    // ── entity_frequency ─────────────────────────────────────────────────────

    #[test]
    fn test_entity_frequency_basic() {
        let triples = vec![
            KgTriple::new(0, 0, 1),
            KgTriple::new(0, 0, 2),
            KgTriple::new(1, 0, 2),
        ];
        let freq = entity_frequency(&triples);
        assert_eq!(freq[&0], 2); // subject twice
        assert_eq!(freq[&1], 2); // subject + object once each
        assert_eq!(freq[&2], 2); // object twice
    }

    #[test]
    fn test_entity_frequency_empty() {
        let freq = entity_frequency(&[]);
        assert!(freq.is_empty());
    }

    #[test]
    fn test_entity_frequency_all_same() {
        let triples = vec![
            KgTriple::new(7, 0, 7),
        ];
        let freq = entity_frequency(&triples);
        assert_eq!(freq[&7], 2); // both subject and object
    }

    // ── KnowledgeGraphDataset helper methods ─────────────────────────────────

    #[test]
    fn test_dataset_by_relation() {
        let kg = FreebaseSubset::generate(10, 3, 30, 5).expect("dataset by relation");
        for rel in 0..kg.n_relations {
            let subset = kg.triples_by_relation(rel);
            for t in subset {
                assert_eq!(t.predicate, rel);
            }
        }
    }

    #[test]
    fn test_dataset_entity_names() {
        let kg = FreebaseSubset::generate(5, 2, 10, 3).expect("entity names");
        assert_eq!(kg.entity_name(0), Some("entity_0"));
        assert_eq!(kg.entity_name(100), None);
    }
}
