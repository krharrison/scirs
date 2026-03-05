//! Dependency parsing: finding grammatical relations between words.
//!
//! # Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`graph`] | Core types: `DependencyGraph`, `DependencyArc`, `DepLabel` |
//! | [`arc_standard`] | Arc-Standard transition parser (Nivre 2004) |
//! | [`arc_eager`] | Arc-Eager transition parser (Nivre 2003) |
//! | [`projective`] | Eisner O(n³) projective parser + Chu-Liu/Edmonds arborescence |
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_text::dependency::{ArcStandardParser, DepLabel};
//!
//! let tokens  = vec!["The".to_string(), "cat".to_string(), "sat".to_string()];
//! let pos     = vec!["DT".to_string(), "NN".to_string(), "VBD".to_string()];
//! let parser  = ArcStandardParser::new();
//! let graph   = parser.parse(&tokens, &pos);
//!
//! // Every token must have exactly one head.
//! for i in 1..=graph.n_tokens {
//!     assert!(graph.head_of(i).is_some());
//! }
//!
//! // CoNLL-U output.
//! let conllu = graph.to_conllu();
//! assert!(conllu.contains("sat"));
//! ```

pub mod arc_eager;
pub mod arc_standard;
pub mod graph;
pub mod projective;

pub use arc_eager::ArcEagerParser;
pub use arc_standard::{ArcStandardConfig, ArcStandardParser, Transition};
pub use graph::{DepLabel, DependencyArc, DependencyGraph};
pub use projective::{ChuLiuEdmonds, EisnerParser, ScoreMatrix};

// ---------------------------------------------------------------------------
// Integration tests (module-level)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // DependencyGraph: projectivity
    // ------------------------------------------------------------------

    #[test]
    fn integration_projectivity_simple_tree() {
        // "The cat sat" — DT NN VBD
        // ROOT(0) → sat(3), sat(3) → cat(2), cat(2) → The(1)
        let tokens = vec!["The".into(), "cat".into(), "sat".into()];
        let pos    = vec!["DT".into(), "NN".into(), "VBD".into()];
        let mut g  = DependencyGraph::new(tokens, pos);
        g.add_arc(0, 3, DepLabel::Root, 1.0);
        g.add_arc(3, 2, DepLabel::Subj, 1.0);
        g.add_arc(2, 1, DepLabel::Det,  1.0);
        assert!(g.is_projective(), "simple left-branching tree should be projective");
    }

    // ------------------------------------------------------------------
    // ArcStandardParser on simple sentence
    // ------------------------------------------------------------------

    #[test]
    fn integration_arc_standard_three_tokens() {
        let tokens = vec!["The".into(), "cat".into(), "sat".into()];
        let pos    = vec!["DT".into(), "NN".into(), "VBD".into()];
        let parser = ArcStandardParser::new();
        let graph  = parser.parse(&tokens, &pos);

        assert_eq!(graph.n_tokens, 3);
        for i in 1..=3 {
            assert!(
                graph.head_of(i).is_some(),
                "arc-standard: token {} missing head", i
            );
        }
    }

    // ------------------------------------------------------------------
    // LAS / UAS computation
    // ------------------------------------------------------------------

    #[test]
    fn integration_las_uas() {
        let tokens = vec!["a".into(), "b".into(), "c".into()];
        let pos    = vec!["NN".into(); 3];

        // Build a gold graph.
        let mut gold = DependencyGraph::new(tokens.clone(), pos.clone());
        gold.add_arc(0, 3, DepLabel::Root, 1.0);
        gold.add_arc(3, 2, DepLabel::Subj, 1.0);
        gold.add_arc(2, 1, DepLabel::Det,  1.0);

        // Identical prediction.
        let pred = gold.clone();
        let las  = pred.las(&gold);
        let uas  = pred.uas(&gold);
        assert!((las - 1.0).abs() < 1e-9, "LAS should be 1.0, got {}", las);
        assert!((uas - 1.0).abs() < 1e-9, "UAS should be 1.0, got {}", uas);

        // Wrong label prediction (UAS should be 1.0, LAS < 1.0).
        let mut wrong_label = DependencyGraph::new(tokens.clone(), pos.clone());
        wrong_label.add_arc(0, 3, DepLabel::Dep, 1.0);
        wrong_label.add_arc(3, 2, DepLabel::Dep, 1.0);
        wrong_label.add_arc(2, 1, DepLabel::Dep, 1.0);
        let las2 = wrong_label.las(&gold);
        let uas2 = wrong_label.uas(&gold);
        assert!((uas2 - 1.0).abs() < 1e-9, "UAS with wrong labels should be 1.0");
        assert!(las2 < 1.0, "LAS with wrong labels should be < 1.0");
    }

    // ------------------------------------------------------------------
    // CoNLL-U output
    // ------------------------------------------------------------------

    #[test]
    fn integration_conllu_output() {
        let tokens = vec!["The".into(), "cat".into(), "sat".into()];
        let pos    = vec!["DT".into(), "NN".into(), "VBD".into()];
        let mut g  = DependencyGraph::new(tokens, pos);
        g.add_arc(0, 3, DepLabel::Root, 1.0);
        g.add_arc(3, 2, DepLabel::Subj, 1.0);
        g.add_arc(2, 1, DepLabel::Det,  1.0);
        let conllu = g.to_conllu();

        // Should contain all three tokens.
        assert!(conllu.contains("The"),  "CoNLL-U missing 'The'");
        assert!(conllu.contains("cat"),  "CoNLL-U missing 'cat'");
        assert!(conllu.contains("sat"),  "CoNLL-U missing 'sat'");
        // Token 1 (The): head = 2, label = det.
        assert!(conllu.contains("\t2\tdet\t"), "CoNLL-U wrong head/label for 'The'");
        // Token 3 (sat): head = 0, label = root.
        assert!(conllu.contains("\t0\troot\t"), "CoNLL-U wrong head/label for 'sat'");

        // Round-trip.
        let g2 = DependencyGraph::from_conllu(&conllu);
        assert_eq!(g2.n_tokens, g.n_tokens);
        assert_eq!(g2.tokens, g.tokens);
    }

    // ------------------------------------------------------------------
    // EisnerParser on a small sentence
    // ------------------------------------------------------------------

    #[test]
    fn integration_eisner_small_sentence() {
        let tokens  = vec!["The".into(), "cat".into(), "sat".into()];
        let pos     = vec!["DT".into(), "NN".into(), "VBD".into()];
        let parser  = EisnerParser::from_heuristic(tokens.len());
        let graph   = parser.parse_to_graph(tokens.clone(), pos.clone());

        assert_eq!(graph.n_tokens, 3);
        for i in 1..=3 {
            assert!(graph.head_of(i).is_some(), "Eisner: token {} missing head", i);
        }
        // Distance-heuristic always yields a projective tree.
        assert!(graph.is_projective());
    }

    // ------------------------------------------------------------------
    // ArcEagerParser
    // ------------------------------------------------------------------

    #[test]
    fn integration_arc_eager_three_tokens() {
        let tokens = vec!["The".into(), "cat".into(), "sat".into()];
        let pos    = vec!["DT".into(), "NN".into(), "VBD".into()];
        let parser = ArcEagerParser::new();
        let graph  = parser.parse(&tokens, &pos);

        assert_eq!(graph.n_tokens, 3);
        for i in 1..=3 {
            assert!(
                graph.head_of(i).is_some(),
                "arc-eager: token {} missing head", i
            );
        }
    }

    // ------------------------------------------------------------------
    // ChuLiuEdmonds
    // ------------------------------------------------------------------

    #[test]
    fn integration_chu_liu_edmonds() {
        let scores = ScoreMatrix::from_distance_heuristic(3);
        let tokens = vec!["The".into(), "cat".into(), "sat".into()];
        let pos    = vec!["DT".into(), "NN".into(), "VBD".into()];
        let graph  = ChuLiuEdmonds::parse_to_graph(&scores, tokens, pos);
        assert_eq!(graph.n_tokens, 3);
        for i in 1..=3 {
            assert!(graph.head_of(i).is_some(), "CLE: token {} missing head", i);
        }
    }
}
