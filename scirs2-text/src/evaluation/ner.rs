//! CoNLL-2003 NER evaluation protocol — span-level F1.
//!
//! This module implements the *exact match* span evaluation protocol used by
//! the CoNLL-2003 shared task.  Both span boundaries **and** the entity label
//! must agree for a prediction to count as a true positive.
//!
//! # BIO encoding
//!
//! Token labels follow the standard BIO(E) scheme:
//! - `B-<TYPE>` — beginning of a named entity of type `<TYPE>`
//! - `I-<TYPE>` — inside/continuation of a named entity of the same type
//! - `O` — outside any entity
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::evaluation::ner::{extract_spans_from_bio, evaluate_ner, NerSpan};
//!
//! let tokens = ["John", "Smith", "works", "at", "Google", "."];
//! let labels = ["B-PER", "I-PER", "O", "O", "B-ORG", "O"];
//!
//! let gold = extract_spans_from_bio(
//!     &tokens.iter().map(|s| *s).collect::<Vec<_>>(),
//!     &labels.iter().map(|s| *s).collect::<Vec<_>>(),
//! );
//! assert_eq!(gold.len(), 2);
//! ```

use crate::error::TextError;
use std::collections::HashMap;
use std::fmt;

// ─── Core data types ──────────────────────────────────────────────────────────

/// A contiguous named-entity span in a token sequence.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NerSpan {
    /// Inclusive start token index.
    pub start: usize,
    /// Exclusive end token index.
    pub end: usize,
    /// Entity type label (e.g. `"PER"`, `"ORG"`, `"LOC"`, `"MISC"`).
    pub label: String,
}

impl NerSpan {
    /// Construct a new span.
    pub fn new(start: usize, end: usize, label: impl Into<String>) -> Self {
        NerSpan {
            start,
            end,
            label: label.into(),
        }
    }
}

impl fmt::Display for NerSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}..{})", self.label, self.start, self.end)
    }
}

/// Per-class precision / recall / F1 metrics.
#[derive(Debug, Clone)]
pub struct ClassMetrics {
    /// Precision = TP / (TP + FP).
    pub precision: f64,
    /// Recall = TP / (TP + FN).
    pub recall: f64,
    /// F1 harmonic mean.
    pub f1: f64,
    /// Number of gold spans with this label.
    pub support: usize,
    /// True positives.
    pub tp: usize,
    /// False positives.
    pub fp: usize,
    /// False negatives.
    pub fn_: usize,
}

impl ClassMetrics {
    fn from_counts(tp: usize, fp: usize, fn_: usize) -> Self {
        let precision = if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let recall = if tp + fn_ == 0 {
            0.0
        } else {
            tp as f64 / (tp + fn_) as f64
        };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        ClassMetrics {
            precision,
            recall,
            f1,
            support: tp + fn_,
            tp,
            fp,
            fn_,
        }
    }
}

/// Overall NER evaluation result (CoNLL-style).
#[derive(Debug, Clone)]
pub struct NerEvaluationResult {
    /// Micro-averaged precision over all entity spans.
    pub precision: f64,
    /// Micro-averaged recall over all entity spans.
    pub recall: f64,
    /// Micro-averaged F1 (the primary CoNLL metric).
    pub f1: f64,
    /// Per-entity-type metrics.
    pub per_class: HashMap<String, ClassMetrics>,
    /// Exact match accuracy at the sentence level (fraction of sequences
    /// where all predicted spans are correct and no gold spans are missing).
    pub exact_match: f64,
}

// ─── BIO span extraction ──────────────────────────────────────────────────────

/// Extract named-entity [`NerSpan`]s from a BIO-tagged token sequence.
///
/// Handles:
/// - Standard `B-TYPE` / `I-TYPE` / `O` labels.
/// - Implicit entity boundary changes (a new `B-` tag after an `I-` tag of a
///   different type, or a `B-` of the same type, terminates the current span).
///
/// # Errors
///
/// Returns an empty `Vec` rather than an error on malformed label sequences
/// (e.g. a bare `I-TYPE` with no preceding `B-TYPE`) — the continuation token
/// is treated as a fresh span start.
pub fn extract_spans_from_bio(tokens: &[&str], labels: &[&str]) -> Vec<NerSpan> {
    if tokens.len() != labels.len() {
        return Vec::new();
    }

    let mut spans: Vec<NerSpan> = Vec::new();
    let mut current: Option<(usize, String)> = None; // (start, label)

    for (i, label) in labels.iter().enumerate() {
        let tag = *label;

        if tag == "O" || tag.is_empty() {
            // Close any open span.
            if let Some((start, lbl)) = current.take() {
                spans.push(NerSpan::new(start, i, lbl));
            }
        } else if let Some(rest) = tag.strip_prefix("B-") {
            // New entity begins — close previous span.
            if let Some((start, lbl)) = current.take() {
                spans.push(NerSpan::new(start, i, lbl));
            }
            current = Some((i, rest.to_string()));
        } else if let Some(rest) = tag.strip_prefix("I-") {
            // Continuation — only valid if entity type matches.
            match &current {
                Some((_, lbl)) if lbl == rest => {
                    // Same type: continue
                }
                _ => {
                    // Type mismatch or no open span: close old, start new.
                    if let Some((start, lbl)) = current.take() {
                        spans.push(NerSpan::new(start, i, lbl));
                    }
                    current = Some((i, rest.to_string()));
                }
            }
        } else {
            // Unknown scheme — treat the whole tag as a B-tag.
            if let Some((start, lbl)) = current.take() {
                spans.push(NerSpan::new(start, i, lbl));
            }
            current = Some((i, tag.to_string()));
        }
    }

    // Close trailing span.
    if let Some((start, lbl)) = current.take() {
        spans.push(NerSpan::new(start, labels.len(), lbl));
    }

    spans
}

// ─── Evaluation ──────────────────────────────────────────────────────────────

/// Compute CoNLL-2003 NER evaluation metrics.
///
/// Evaluation is performed at the **span level**: a prediction is a true
/// positive only when both token boundaries **and** the entity type label
/// exactly match a gold span.
///
/// Returns micro-averaged precision/recall/F1 plus per-class breakdowns.
pub fn evaluate_ner(predictions: &[Vec<NerSpan>], gold: &[Vec<NerSpan>]) -> NerEvaluationResult {
    if predictions.len() != gold.len() {
        // Return zeroed result for mismatched batch sizes.
        return NerEvaluationResult {
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
            per_class: HashMap::new(),
            exact_match: 0.0,
        };
    }

    // Per-class counters: label → (tp, fp, fn_)
    let mut class_tp: HashMap<String, usize> = HashMap::new();
    let mut class_fp: HashMap<String, usize> = HashMap::new();
    let mut class_fn: HashMap<String, usize> = HashMap::new();

    let mut exact_match_count = 0usize;

    for (pred_spans, gold_spans) in predictions.iter().zip(gold.iter()) {
        // Convert spans to HashSet for O(1) lookup.
        use std::collections::HashSet;
        let gold_set: HashSet<&NerSpan> = gold_spans.iter().collect();
        let pred_set: HashSet<&NerSpan> = pred_spans.iter().collect();

        for span in pred_spans {
            let label = &span.label;
            if gold_set.contains(span) {
                *class_tp.entry(label.clone()).or_insert(0) += 1;
            } else {
                *class_fp.entry(label.clone()).or_insert(0) += 1;
            }
        }

        for span in gold_spans {
            let label = &span.label;
            if !pred_set.contains(span) {
                *class_fn.entry(label.clone()).or_insert(0) += 1;
            }
        }

        // Exact match: predicted set equals gold set.
        if pred_set == gold_set {
            exact_match_count += 1;
        }
    }

    // Collect all known labels.
    let mut all_labels: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for k in class_tp
        .keys()
        .chain(class_fp.keys())
        .chain(class_fn.keys())
    {
        all_labels.insert(k.clone());
    }

    let mut per_class: HashMap<String, ClassMetrics> = HashMap::new();
    let mut total_tp = 0usize;
    let mut total_fp = 0usize;
    let mut total_fn = 0usize;

    for label in &all_labels {
        let tp = *class_tp.get(label).unwrap_or(&0);
        let fp = *class_fp.get(label).unwrap_or(&0);
        let fn_ = *class_fn.get(label).unwrap_or(&0);

        per_class.insert(label.clone(), ClassMetrics::from_counts(tp, fp, fn_));

        total_tp += tp;
        total_fp += fp;
        total_fn += fn_;
    }

    let overall = ClassMetrics::from_counts(total_tp, total_fp, total_fn);
    let n_sequences = predictions.len();
    let exact_match = if n_sequences == 0 {
        0.0
    } else {
        exact_match_count as f64 / n_sequences as f64
    };

    NerEvaluationResult {
        precision: overall.precision,
        recall: overall.recall,
        f1: overall.f1,
        per_class,
        exact_match,
    }
}

// ─── CoNLL-style report formatting ───────────────────────────────────────────

/// Format an [`NerEvaluationResult`] as a `conlleval`-style text report.
///
/// The output closely follows the style produced by the official `conlleval.pl`
/// Perl script distributed with the CoNLL-2003 shared task:
///
/// ```text
/// processed N tokens; found: K phrases; correct: J
/// accuracy: XX.XX%; precision: XX.XX%; recall: XX.XX%; FB1: XX.XX
///        LOC: precision XX.XX%; recall XX.XX%; FB1: XX.XX  N
/// ```
pub fn conll_format_report(result: &NerEvaluationResult) -> String {
    let mut lines: Vec<String> = Vec::new();

    // Compute aggregate counts from per-class metrics.
    let total_gold: usize = result.per_class.values().map(|m| m.support).sum();
    let total_pred: usize = result.per_class.values().map(|m| m.tp + m.fp).sum();
    let total_correct: usize = result.per_class.values().map(|m| m.tp).sum();

    lines.push(format!(
        "processed N tokens with {} phrases; found: {} phrases; correct: {}",
        total_gold, total_pred, total_correct,
    ));

    lines.push(format!(
        "accuracy:  N/A; precision:  {:6.2}%; recall:  {:6.2}%; FB1:  {:6.2}",
        result.precision * 100.0,
        result.recall * 100.0,
        result.f1 * 100.0,
    ));

    // Per-class rows sorted alphabetically.
    let mut labels: Vec<&String> = result.per_class.keys().collect();
    labels.sort();

    for label in labels {
        let m = &result.per_class[label];
        lines.push(format!(
            "  {:>8}: precision:  {:6.2}%; recall:  {:6.2}%; FB1:  {:6.2}  {}",
            label,
            m.precision * 100.0,
            m.recall * 100.0,
            m.f1 * 100.0,
            m.support,
        ));
    }

    lines.join("\n")
}

/// Convenience wrapper that returns an error instead of silently degrading when
/// input lengths are mismatched.
pub fn evaluate_ner_checked(
    predictions: &[Vec<NerSpan>],
    gold: &[Vec<NerSpan>],
) -> Result<NerEvaluationResult, TextError> {
    if predictions.len() != gold.len() {
        return Err(TextError::InvalidInput(format!(
            "Prediction batch size ({}) != gold batch size ({})",
            predictions.len(),
            gold.len()
        )));
    }
    Ok(evaluate_ner(predictions, gold))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BIO extraction ────────────────────────────────────────────────────

    #[test]
    fn test_bio_extraction_basic() {
        let tokens = vec!["John", "Smith", "works", "at", "Google", "."];
        let labels = vec!["B-PER", "I-PER", "O", "O", "B-ORG", "O"];
        let spans = extract_spans_from_bio(&tokens, &labels);

        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0], NerSpan::new(0, 2, "PER"));
        assert_eq!(spans[1], NerSpan::new(4, 5, "ORG"));
    }

    #[test]
    fn test_bio_extraction_single_token_entity() {
        let tokens = vec!["Paris", "is", "beautiful"];
        let labels = vec!["B-LOC", "O", "O"];
        let spans = extract_spans_from_bio(&tokens, &labels);

        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0], NerSpan::new(0, 1, "LOC"));
    }

    #[test]
    fn test_bio_extraction_all_outside() {
        let tokens = vec!["the", "cat", "sat"];
        let labels = vec!["O", "O", "O"];
        let spans = extract_spans_from_bio(&tokens, &labels);
        assert!(spans.is_empty());
    }

    #[test]
    fn test_bio_extraction_trailing_entity() {
        let tokens = vec!["Visit", "New", "York"];
        let labels = vec!["O", "B-LOC", "I-LOC"];
        let spans = extract_spans_from_bio(&tokens, &labels);

        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0], NerSpan::new(1, 3, "LOC"));
    }

    #[test]
    fn test_bio_extraction_nested_b_tags() {
        // Two consecutive B- tags without an I- between them.
        let tokens = vec!["Apple", "Inc", "Google", "LLC"];
        let labels = vec!["B-ORG", "I-ORG", "B-ORG", "I-ORG"];
        let spans = extract_spans_from_bio(&tokens, &labels);

        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0], NerSpan::new(0, 2, "ORG"));
        assert_eq!(spans[1], NerSpan::new(2, 4, "ORG"));
    }

    #[test]
    fn test_bio_extraction_type_change() {
        // I-ORG after B-PER: different type — creates a new implicit span.
        let tokens = vec!["John", "Google"];
        let labels = vec!["B-PER", "I-ORG"];
        let spans = extract_spans_from_bio(&tokens, &labels);

        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].label, "PER");
        assert_eq!(spans[1].label, "ORG");
    }

    #[test]
    fn test_bio_extraction_mismatched_lengths() {
        let tokens = vec!["John"];
        let labels = vec!["B-PER", "O"];
        let spans = extract_spans_from_bio(&tokens, &labels);
        assert!(spans.is_empty()); // graceful degradation
    }

    // ── NER evaluation ────────────────────────────────────────────────────

    #[test]
    fn test_ner_eval_perfect() {
        let gold = vec![vec![NerSpan::new(0, 2, "PER"), NerSpan::new(4, 5, "ORG")]];
        let pred = gold.clone();

        let result = evaluate_ner(&pred, &gold);
        assert!((result.precision - 1.0).abs() < 1e-9);
        assert!((result.recall - 1.0).abs() < 1e-9);
        assert!((result.f1 - 1.0).abs() < 1e-9);
        assert!((result.exact_match - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ner_eval_no_predictions() {
        let gold = vec![vec![NerSpan::new(0, 2, "PER")]];
        let pred = vec![vec![]];

        let result = evaluate_ner(&pred, &gold);
        assert!((result.precision - 0.0).abs() < 1e-9);
        assert!((result.recall - 0.0).abs() < 1e-9);
        assert!((result.f1 - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_ner_eval_partial_match() {
        // Prediction has wrong end boundary → FP + FN.
        let gold = vec![vec![NerSpan::new(0, 3, "PER")]];
        let pred = vec![vec![NerSpan::new(0, 2, "PER")]];

        let result = evaluate_ner(&pred, &gold);
        // 0 TP, 1 FP, 1 FN → precision=0, recall=0, f1=0
        assert!((result.precision - 0.0).abs() < 1e-9);
        assert!((result.recall - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_ner_eval_wrong_label() {
        // Same span boundaries but wrong label.
        let gold = vec![vec![NerSpan::new(0, 2, "PER")]];
        let pred = vec![vec![NerSpan::new(0, 2, "ORG")]];

        let result = evaluate_ner(&pred, &gold);
        assert!((result.f1 - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_ner_eval_per_class_metrics() {
        let gold = vec![vec![
            NerSpan::new(0, 2, "PER"),
            NerSpan::new(3, 5, "ORG"),
            NerSpan::new(6, 8, "LOC"),
        ]];
        let pred = vec![vec![
            NerSpan::new(0, 2, "PER"), // TP for PER
            // ORG missed → FN
            NerSpan::new(6, 8, "LOC"),  // TP for LOC
            NerSpan::new(9, 11, "ORG"), // FP for ORG
        ]];

        let result = evaluate_ner(&pred, &gold);

        let per = result.per_class.get("PER").expect("PER class");
        assert_eq!(per.tp, 1);
        assert_eq!(per.fp, 0);
        assert_eq!(per.fn_, 0);

        let org = result.per_class.get("ORG").expect("ORG class");
        // 0 TP, 1 FP (9-11), 1 FN (3-5)
        assert_eq!(org.tp, 0);
        assert_eq!(org.fp, 1);
        assert_eq!(org.fn_, 1);

        let loc = result.per_class.get("LOC").expect("LOC class");
        assert_eq!(loc.tp, 1);
        assert_eq!(loc.fp, 0);
        assert_eq!(loc.fn_, 0);
    }

    #[test]
    fn test_ner_eval_multiple_sequences() {
        let gold = vec![
            vec![NerSpan::new(0, 1, "PER")],
            vec![NerSpan::new(0, 1, "ORG")],
        ];
        let pred = vec![
            vec![NerSpan::new(0, 1, "PER")], // correct
            vec![NerSpan::new(0, 1, "PER")], // wrong label
        ];

        let result = evaluate_ner(&pred, &gold);
        // PER: 2 TP total? No — the second seq has pred PER but gold ORG.
        // Seq 1: PER TP=1. Seq 2: ORG FN=1, PER FP=1.
        assert!((result.precision - 0.5).abs() < 1e-9);
        assert!((result.recall - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_ner_eval_exact_match_rate() {
        let gold = vec![
            vec![NerSpan::new(0, 1, "PER")],
            vec![NerSpan::new(0, 1, "ORG")],
        ];
        let pred = vec![
            vec![NerSpan::new(0, 1, "PER")], // exact
            vec![NerSpan::new(0, 1, "PER")], // not exact
        ];

        let result = evaluate_ner(&pred, &gold);
        assert!((result.exact_match - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_conll_report_format() {
        let gold = vec![vec![NerSpan::new(0, 2, "PER")]];
        let pred = gold.clone();
        let result = evaluate_ner(&pred, &gold);
        let report = conll_format_report(&result);

        assert!(report.contains("precision"));
        assert!(report.contains("recall"));
        assert!(report.contains("FB1"));
        assert!(report.contains("PER"));
        assert!(report.contains("100.00"));
    }

    #[test]
    fn test_ner_checked_length_mismatch() {
        let gold = vec![vec![NerSpan::new(0, 1, "PER")]];
        let pred: Vec<Vec<NerSpan>> = vec![];
        assert!(evaluate_ner_checked(&pred, &gold).is_err());
    }

    #[test]
    fn test_bio_extraction_multiple_types() {
        let tokens = vec!["The", "EU", "said", "Angela", "Merkel", "from", "Germany"];
        let labels = vec!["O", "B-ORG", "O", "B-PER", "I-PER", "O", "B-LOC"];

        let spans = extract_spans_from_bio(&tokens, &labels);
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0], NerSpan::new(1, 2, "ORG"));
        assert_eq!(spans[1], NerSpan::new(3, 5, "PER"));
        assert_eq!(spans[2], NerSpan::new(6, 7, "LOC"));
    }
}
