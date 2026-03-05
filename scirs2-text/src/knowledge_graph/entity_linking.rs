//! Named entity linking: resolving text mentions to knowledge graph entities.
//!
//! Provides:
//! 1. **Candidate generation** – surface-form lookup with alias expansion.
//! 2. **Entity disambiguation** – rank candidates by context similarity.
//! 3. **NIL detection** – identify mentions with no valid KG match.
//! 4. **Coreference-aware linking** – propagate links through coreference chains.

use std::collections::HashMap;

use crate::error::{Result, TextError};
use super::graph::{EntityId, KnowledgeGraph};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A mention of a potential entity in text.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityMention {
    /// The surface text of the mention (e.g., "Apple", "the company").
    pub surface: String,
    /// The character span of the mention.
    pub span: (usize, usize),
    /// Optionally, the NER type assigned by an upstream tagger.
    pub ner_type: Option<String>,
    /// Optional coreference chain identifier (for coreference-aware linking).
    pub coref_chain: Option<usize>,
}

/// The result of linking a mention to the knowledge graph.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkedMention {
    /// The original mention.
    pub mention: EntityMention,
    /// The resolved entity id (None → NIL).
    pub entity_id: Option<EntityId>,
    /// The canonical entity name (None → NIL).
    pub entity_name: Option<String>,
    /// Linking confidence in [0, 1].
    pub confidence: f64,
    /// Whether the mention was deemed NIL (no matching entity).
    pub is_nil: bool,
    /// Scored candidate list used for disambiguation.
    pub candidates: Vec<CandidateEntity>,
}

/// A single candidate entity with its ranking score.
#[derive(Debug, Clone, PartialEq)]
/// A single candidate entity with its ranking score.
pub struct CandidateEntity {
    /// The entity's numeric identifier in the knowledge graph.
    pub entity_id: EntityId,
    /// The canonical name of the entity.
    pub entity_name: String,
    /// Disambiguation score in [0, 1] (higher = more likely).
    pub score: f64,
}

// ---------------------------------------------------------------------------
// Alias table
// ---------------------------------------------------------------------------

/// An alias entry describing one surface form → entity mapping.
#[derive(Debug, Clone)]
struct AliasEntry {
    entity_id: EntityId,
    entity_name: String,
    /// Prior probability that this surface form refers to this entity.
    prior: f64,
}

// ---------------------------------------------------------------------------
// Entity Linker
// ---------------------------------------------------------------------------

/// Links entity mentions in text to knowledge graph entries.
///
/// # Example
/// ```rust
/// use scirs2_text::knowledge_graph::{KnowledgeGraph, entity_linking::EntityLinker};
///
/// let mut kg = KnowledgeGraph::new();
/// kg.add_entity("Apple Inc.", "Organization");
/// kg.add_entity("Apple", "Fruit");
///
/// let mut linker = EntityLinker::new();
/// linker.build_from_kg(&kg);
/// linker.add_alias("Apple", "Apple Inc.", 0.7);
/// linker.add_alias("Apple", "Apple", 0.3);
///
/// let mention = scirs2_text::knowledge_graph::entity_linking::EntityMention {
///     surface: "Apple".to_string(),
///     span: (0, 5),
///     ner_type: Some("ORG".to_string()),
///     coref_chain: None,
/// };
/// let linked = linker.link_mention(&mention, "Apple released a new iPhone.", &kg);
/// assert!(linked.entity_id.is_some());
/// ```
pub struct EntityLinker {
    /// surface_form (lowercase) → list of candidates
    alias_table: HashMap<String, Vec<AliasEntry>>,
    /// NIL threshold: if best score < this → NIL
    nil_threshold: f64,
    /// Maximum number of candidates to consider during disambiguation.
    max_candidates: usize,
}

impl EntityLinker {
    /// Create an empty linker.
    pub fn new() -> Self {
        EntityLinker {
            alias_table: HashMap::new(),
            nil_threshold: 0.15,
            max_candidates: 10,
        }
    }

    /// Set the NIL detection threshold (default: 0.15).
    pub fn with_nil_threshold(mut self, threshold: f64) -> Self {
        self.nil_threshold = threshold;
        self
    }

    /// Populate the alias table from all entity names in a knowledge graph.
    ///
    /// For each entity the canonical name and any registered type labels are
    /// inserted as aliases.
    pub fn build_from_kg(&mut self, kg: &KnowledgeGraph) {
        for name in kg.entities() {
            if let Some(id) = kg.entity_id(name) {
                // Canonical name
                self.insert_alias(name, id, name, 1.0);
                // Each word in the name as a short-form alias with lower prior
                for token in name.split_whitespace() {
                    if token.len() >= 3 {
                        self.insert_alias(token, id, name, 0.4);
                    }
                }
                // Initialism: "Apple Inc." → "AI"
                let initialism: String = name
                    .split_whitespace()
                    .filter_map(|w| w.chars().next())
                    .filter(|c| c.is_uppercase())
                    .collect();
                if initialism.len() >= 2 {
                    self.insert_alias(&initialism, id, name, 0.3);
                }
            }
        }
    }

    /// Add (or update) an explicit alias mapping.
    ///
    /// - `surface`: the mention surface form (case-insensitive).
    /// - `entity_name`: canonical entity name in the KG.
    /// - `prior`: prior probability for this alias interpretation.
    pub fn add_alias(&mut self, surface: &str, entity_name: &str, prior: f64) {
        // We don't have the id here; it gets resolved at link time.
        // Store with id=usize::MAX as a sentinel; resolved during linking.
        let entry = AliasEntry {
            entity_id: usize::MAX,
            entity_name: entity_name.to_string(),
            prior,
        };
        self.alias_table
            .entry(surface.to_lowercase())
            .or_default()
            .push(entry);
    }

    fn insert_alias(&mut self, surface: &str, id: EntityId, name: &str, prior: f64) {
        let entries = self.alias_table.entry(surface.to_lowercase()).or_default();
        if !entries.iter().any(|e| e.entity_id == id) {
            entries.push(AliasEntry {
                entity_id: id,
                entity_name: name.to_string(),
                prior,
            });
        }
    }

    // -----------------------------------------------------------------------
    // Candidate generation
    // -----------------------------------------------------------------------

    /// Generate candidate entities for a surface-form mention.
    pub fn generate_candidates(
        &self,
        surface: &str,
        kg: &KnowledgeGraph,
    ) -> Vec<CandidateEntity> {
        let key = surface.to_lowercase();
        let mut seen_ids: std::collections::HashSet<EntityId> =
            std::collections::HashSet::new();
        let mut candidates: Vec<CandidateEntity> = Vec::new();

        if let Some(entries) = self.alias_table.get(&key) {
            for entry in entries {
                // Resolve sentinel ids from the KG
                let id = if entry.entity_id == usize::MAX {
                    match kg.entity_id(&entry.entity_name) {
                        Some(id) => id,
                        None => continue,
                    }
                } else {
                    entry.entity_id
                };
                if seen_ids.insert(id) {
                    candidates.push(CandidateEntity {
                        entity_id: id,
                        entity_name: entry.entity_name.clone(),
                        score: entry.prior,
                    });
                }
            }
        }

        // Fuzzy fallback: prefix / substring match on entity names
        if candidates.is_empty() {
            let lower_surface = surface.to_lowercase();
            for name in kg.entities() {
                let lower_name = name.to_lowercase();
                if lower_name.starts_with(&lower_surface)
                    || lower_surface.starts_with(&lower_name)
                    || lower_name.contains(&lower_surface)
                {
                    if let Some(id) = kg.entity_id(name) {
                        if seen_ids.insert(id) {
                            let overlap = lower_name.len().min(lower_surface.len()) as f64
                                / lower_name.len().max(lower_surface.len()).max(1) as f64;
                            candidates.push(CandidateEntity {
                                entity_id: id,
                                entity_name: name.to_string(),
                                score: overlap * 0.5,
                            });
                        }
                    }
                }
            }
        }

        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(self.max_candidates);
        candidates
    }

    // -----------------------------------------------------------------------
    // Disambiguation
    // -----------------------------------------------------------------------

    /// Rank candidates by combining alias prior with context compatibility.
    ///
    /// Context scoring uses:
    /// - NER type compatibility with entity types (if available).
    /// - Simple token-overlap between the sentence context and entity
    ///   neighbour names in the KG.
    fn score_candidate(
        &self,
        candidate: &CandidateEntity,
        mention: &EntityMention,
        context: &str,
        kg: &KnowledgeGraph,
    ) -> f64 {
        let mut score = candidate.score;

        // NER type bonus
        if let Some(ref ner_type) = mention.ner_type {
            let entity_types = kg.entity_types(&candidate.entity_name);
            let ner_lower = ner_type.to_lowercase();
            for et in &entity_types {
                if et.to_lowercase().contains(&ner_lower)
                    || ner_lower.contains(&et.to_lowercase())
                {
                    score += 0.2;
                    break;
                }
            }
        }

        // Context token overlap with neighbours
        let ctx_tokens: std::collections::HashSet<String> = context
            .split_whitespace()
            .map(|t| t.to_lowercase().trim_matches(|c: char| !c.is_alphabetic()).to_string())
            .filter(|t| t.len() > 2)
            .collect();

        let neighbours: Vec<&super::graph::Triple> =
            kg.query_all(&candidate.entity_name);
        for nb in neighbours {
            if let Some(obj_name) = kg.entity_name(nb.object) {
                for tok in obj_name.split_whitespace() {
                    if ctx_tokens.contains(&tok.to_lowercase()) {
                        score += 0.05;
                    }
                }
            }
            if let Some(subj_name) = kg.entity_name(nb.subject) {
                for tok in subj_name.split_whitespace() {
                    if ctx_tokens.contains(&tok.to_lowercase()) {
                        score += 0.05;
                    }
                }
            }
        }

        score.min(1.0)
    }

    // -----------------------------------------------------------------------
    // Linking
    // -----------------------------------------------------------------------

    /// Link a single mention to the KG.
    pub fn link_mention(
        &self,
        mention: &EntityMention,
        context: &str,
        kg: &KnowledgeGraph,
    ) -> LinkedMention {
        let mut candidates = self.generate_candidates(&mention.surface, kg);

        // Re-score with context
        for c in candidates.iter_mut() {
            c.score = self.score_candidate(c, mention, context, kg);
        }
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let top = candidates.first();
        let (entity_id, entity_name, confidence, is_nil) = match top {
            Some(c) if c.score >= self.nil_threshold => {
                (Some(c.entity_id), Some(c.entity_name.clone()), c.score, false)
            }
            _ => (None, None, 0.0, true),
        };

        LinkedMention {
            mention: mention.clone(),
            entity_id,
            entity_name,
            confidence,
            is_nil,
            candidates,
        }
    }

    /// Link all mentions in a document; applies coreference-aware propagation.
    ///
    /// All mentions in the same coreference chain inherit the link of the
    /// highest-confidence non-NIL member of that chain.
    pub fn link_document(
        &self,
        mentions: &[EntityMention],
        document: &str,
        kg: &KnowledgeGraph,
    ) -> Vec<LinkedMention> {
        let mut linked: Vec<LinkedMention> = mentions
            .iter()
            .map(|m| self.link_mention(m, document, kg))
            .collect();

        // Coreference-aware propagation
        self.propagate_coref_links(&mut linked);
        linked
    }

    /// For each coreference chain, propagate the best link to NIL members.
    fn propagate_coref_links(&self, linked: &mut Vec<LinkedMention>) {
        // Build: coref_chain_id → best LinkedMention index
        let mut chain_best: HashMap<usize, usize> = HashMap::new();
        for (i, lm) in linked.iter().enumerate() {
            if let Some(chain_id) = lm.mention.coref_chain {
                if !lm.is_nil {
                    let entry = chain_best.entry(chain_id).or_insert(i);
                    if linked[i].confidence > linked[*entry].confidence {
                        *entry = i;
                    }
                }
            }
        }
        // Apply propagation
        for i in 0..linked.len() {
            if linked[i].is_nil {
                if let Some(chain_id) = linked[i].mention.coref_chain {
                    if let Some(&best_idx) = chain_best.get(&chain_id) {
                        let eid = linked[best_idx].entity_id;
                        let ename = linked[best_idx].entity_name.clone();
                        let conf = linked[best_idx].confidence * 0.8; // slight penalty
                        linked[i].entity_id = eid;
                        linked[i].entity_name = ename;
                        linked[i].confidence = conf;
                        linked[i].is_nil = false;
                    }
                }
            }
        }
    }
}

impl Default for EntityLinker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// NIL detection utilities
// ---------------------------------------------------------------------------

/// Standalone NIL detector: returns `true` when a mention cannot be resolved.
pub fn is_nil_mention(
    surface: &str,
    kg: &KnowledgeGraph,
    linker: &EntityLinker,
    threshold: f64,
) -> bool {
    let dummy = EntityMention {
        surface: surface.to_string(),
        span: (0, surface.len()),
        ner_type: None,
        coref_chain: None,
    };
    let result = linker.link_mention(&dummy, surface, kg);
    result.is_nil || result.confidence < threshold
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_kg() -> KnowledgeGraph {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity("Apple Inc.", "Organization");
        kg.add_entity("Steve Jobs", "Person");
        kg.add_entity("Cupertino", "Location");
        kg.add_relation("Apple Inc.", "founded_by", "Steve Jobs", 0.99);
        kg.add_relation("Apple Inc.", "headquartered_in", "Cupertino", 0.99);
        kg
    }

    #[test]
    fn test_build_from_kg() {
        let kg = sample_kg();
        let mut linker = EntityLinker::new();
        linker.build_from_kg(&kg);
        // "Apple Inc." should be in the alias table
        assert!(linker.alias_table.contains_key("apple inc."));
    }

    #[test]
    fn test_generate_candidates() {
        let kg = sample_kg();
        let mut linker = EntityLinker::new();
        linker.build_from_kg(&kg);
        let candidates = linker.generate_candidates("Apple Inc.", &kg);
        assert!(!candidates.is_empty());
        assert!(candidates.iter().any(|c| c.entity_name == "Apple Inc."));
    }

    #[test]
    fn test_link_known_entity() {
        let kg = sample_kg();
        let mut linker = EntityLinker::new();
        linker.build_from_kg(&kg);

        let mention = EntityMention {
            surface: "Apple Inc.".to_string(),
            span: (0, 10),
            ner_type: Some("Organization".to_string()),
            coref_chain: None,
        };
        let linked = linker.link_mention(&mention, "Apple Inc. was founded by Steve Jobs.", &kg);
        assert!(!linked.is_nil, "should link to known entity");
        assert_eq!(linked.entity_name.as_deref(), Some("Apple Inc."));
    }

    #[test]
    fn test_nil_for_unknown() {
        let kg = sample_kg();
        let mut linker = EntityLinker::new();
        linker.build_from_kg(&kg);

        let mention = EntityMention {
            surface: "Banana Corp".to_string(),
            span: (0, 11),
            ner_type: None,
            coref_chain: None,
        };
        let linked = linker.link_mention(&mention, "Banana Corp sells tropical fruit.", &kg);
        assert!(linked.is_nil, "unknown entity should be NIL");
    }

    #[test]
    fn test_coref_propagation() {
        let kg = sample_kg();
        let mut linker = EntityLinker::new();
        linker.build_from_kg(&kg);

        let mentions = vec![
            EntityMention {
                surface: "Apple Inc.".to_string(),
                span: (0, 10),
                ner_type: Some("Organization".to_string()),
                coref_chain: Some(0),
            },
            EntityMention {
                surface: "the company".to_string(),
                span: (12, 23),
                ner_type: None,
                coref_chain: Some(0), // same chain
            },
        ];
        let linked =
            linker.link_document(&mentions, "Apple Inc. the company was founded in 1976.", &kg);
        // The second mention ("the company") should be linked via coref
        assert!(!linked[1].is_nil, "coref mention should be propagated");
    }

    #[test]
    fn test_is_nil_mention() {
        let kg = sample_kg();
        let mut linker = EntityLinker::new();
        linker.build_from_kg(&kg);

        assert!(!is_nil_mention("Apple Inc.", &kg, &linker, 0.1));
        assert!(is_nil_mention("Unknown Corp XYZ", &kg, &linker, 0.1));
    }
}
