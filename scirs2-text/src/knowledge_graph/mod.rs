//! Knowledge Graph construction, relation extraction, entity linking, and embeddings.
//!
//! This module provides a full pipeline for building and querying typed knowledge graphs
//! from natural language text, including:
//!
//! - [`graph`]: Core `KnowledgeGraph` data structure with BFS shortest-path, subgraph
//!   extraction, and TransE-style entity embeddings.
//! - [`relation_extraction`]: Four complementary extraction strategies (pattern-based,
//!   template-based, Open IE, distant supervision).
//! - [`entity_linking`]: Named entity linking with candidate generation, disambiguation,
//!   NIL detection, and coreference-aware propagation.
//! - [`kg_embeddings`]: Full implementations of TransE, TransR, DistMult, and ComplEx.
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_text::knowledge_graph::{
//!     KnowledgeGraph,
//!     relation_extraction::RelationExtractionPipeline,
//!     entity_linking::EntityLinker,
//! };
//!
//! // Build a graph manually
//! let mut kg = KnowledgeGraph::new();
//! kg.add_entity("Alice", "Person");
//! kg.add_entity("Acme Corp", "Organization");
//! kg.add_relation("Alice", "works_at", "Acme Corp", 0.9);
//!
//! // Query
//! let triples = kg.query_relations("Alice");
//! assert_eq!(triples.len(), 1);
//!
//! // Find shortest path
//! kg.add_entity("New York", "Location");
//! kg.add_relation("Acme Corp", "located_in", "New York", 0.95);
//! let path = kg.shortest_path("Alice", "New York").unwrap();
//! assert_eq!(path.len(), 3);
//!
//! // Extract relations from text
//! let pipeline = RelationExtractionPipeline::new().unwrap();
//! let extracted = pipeline.extract("Google acquired YouTube in 2006.");
//! let _ = extracted; // non-empty for acquisition patterns
//!
//! // Link mentions to entities
//! let mut linker = EntityLinker::new();
//! linker.build_from_kg(&kg);
//! ```
//!
//! # Training KG Embeddings
//!
//! ```rust
//! use scirs2_text::knowledge_graph::kg_embeddings::{
//!     TransE, EmbeddingConfig, build_indices,
//! };
//!
//! let raw_triples = vec![
//!     ("Alice", "works_at", "Acme"),
//!     ("Acme", "located_in", "London"),
//! ];
//! let (emap, rmap, triples) = build_indices(&raw_triples);
//! let mut model = TransE::new(emap.len(), rmap.len(), EmbeddingConfig::default());
//! model.train(&triples);
//! let emb = model.entity_vector(0).unwrap();
//! assert_eq!(emb.len(), 64);
//! ```

pub mod entity_linking;
pub mod graph;
pub mod kg_embeddings;
pub mod relation_extraction;

// Re-export the most commonly used types at the module level for convenience.
pub use entity_linking::{EntityLinker, EntityMention, LinkedMention};
pub use graph::{EntityId, KnowledgeGraph, Triple};
pub use kg_embeddings::{
    build_indices, ComplEx, DistMult, EmbeddingConfig, EmbeddingModel, TransE, TransR,
    TrainingTriple,
};
pub use relation_extraction::{
    DistantSupervisionAligner, DistantSupervisionConfig, ExtractionStrategy, ExtractedRelation,
    OpenIeExtractor, PatternRelationExtractor, RelationExtractionPipeline,
    TemplateRelationExtractor,
};

#[cfg(test)]
mod tests {
    use super::*;
    use entity_linking::EntityMention;
    use kg_embeddings::EmbeddingConfig;
    use relation_extraction::RelationExtractionPipeline;

    /// End-to-end integration test: text → KG → embeddings → query.
    #[test]
    fn test_end_to_end_pipeline() {
        // 1. Build a small KG from a corpus
        let mut kg = KnowledgeGraph::new();
        let pipeline = RelationExtractionPipeline::new().expect("pipeline creation failed");
        let corpus = [
            "Google acquired YouTube in 2006.",
            "Apple is headquartered in Cupertino.",
            "Elon Musk founded Tesla.",
        ];
        for sentence in &corpus {
            pipeline.populate_kg(sentence, &mut kg);
        }
        // Also add a few known entities for linking
        kg.add_entity("Google", "Organization");
        kg.add_entity("YouTube", "Organization");
        kg.add_entity("Apple", "Organization");
        kg.add_entity("Cupertino", "Location");
        kg.add_relation("Google", "acquired", "YouTube", 0.99);
        kg.add_relation("Apple", "headquartered_in", "Cupertino", 0.99);

        // 2. Entity linking
        let mut linker = EntityLinker::new();
        linker.build_from_kg(&kg);
        let mention = EntityMention {
            surface: "Google".to_string(),
            span: (0, 6),
            ner_type: Some("Organization".to_string()),
            coref_chain: None,
        };
        let linked = linker.link_mention(
            &mention,
            "Google is a major tech company.",
            &kg,
        );
        assert!(!linked.is_nil, "Google should be linked");

        // 3. Train TransE embeddings
        let all_triples = kg.all_triples();
        let raw: Vec<(&str, &str, &str)> = all_triples
            .iter()
            .filter_map(|t| {
                let s = kg.entity_name(t.subject)?;
                let o = kg.entity_name(t.object)?;
                Some((s, t.predicate.as_str(), o))
            })
            .collect();

        if !raw.is_empty() {
            let (emap, rmap, triples) = build_indices(&raw);
            let mut model = TransE::new(
                emap.len(),
                rmap.len(),
                EmbeddingConfig { epochs: 10, dim: 16, ..Default::default() },
            );
            model.train(&triples);
            let emb = model.entity_vector(0).expect("embedding should exist");
            assert_eq!(emb.len(), 16);
        }
    }

    #[test]
    fn test_kg_subgraph_and_path() {
        let mut kg = KnowledgeGraph::new();
        for (s, r, o) in [
            ("A", "r1", "B"),
            ("B", "r2", "C"),
            ("C", "r3", "D"),
            ("D", "r4", "E"),
        ] {
            kg.add_entity(s, "Node");
            kg.add_entity(o, "Node");
            kg.add_relation(s, r, o, 1.0);
        }

        let path = kg.shortest_path("A", "E").expect("path must exist");
        assert_eq!(path.len(), 5, "path A→B→C→D→E has 5 nodes");

        let sub = kg.subgraph("C", 1);
        assert!(sub.num_entities() >= 3, "C, B, D expected in 1-hop subgraph");
    }

    #[test]
    fn test_all_embedding_models() {
        let triples = vec![
            TrainingTriple { head: 0, relation: 0, tail: 1 },
            TrainingTriple { head: 1, relation: 1, tail: 2 },
        ];
        let cfg = EmbeddingConfig { dim: 8, epochs: 5, ..Default::default() };

        let mut transe = TransE::new(3, 2, cfg.clone());
        transe.train(&triples);
        assert!(transe.entity_vector(0).is_some());

        let mut transr = TransR::new(3, 2, cfg.clone(), 4);
        transr.train(&triples);
        assert!(transr.entity_vector(0).is_some());

        let mut distmult = DistMult::new(3, 2, cfg.clone());
        distmult.train(&triples);
        assert!(distmult.entity_vector(0).is_some());

        let mut complex = ComplEx::new(3, 2, cfg);
        complex.train(&triples);
        assert!(complex.entity_vector(0).is_some());
    }
}
