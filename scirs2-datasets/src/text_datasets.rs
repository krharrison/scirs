//! Natural language / text dataset generators.
//!
//! This module provides synthetic text datasets for NLP tasks:
//! text classification, sentiment analysis, named entity recognition,
//! question answering, and language modelling.
//!
//! All generators are deterministic given a seed and require no external
//! dependencies beyond the standard library (RNG is a minimal Park-Miller LCG).

use crate::error::{DatasetsError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Minimal Park-Miller LCG (no external rand crate)
// ─────────────────────────────────────────────────────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 6364136223846793005 } else { seed })
    }

    /// Next pseudo-random u64.
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Uniform float in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform usize in [0, n).
    fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }

    /// Box-Muller Normal(0,1) sample.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TextDataset
// ─────────────────────────────────────────────────────────────────────────────

/// A generic text dataset holding raw text strings and optional class labels.
#[derive(Debug, Clone)]
pub struct TextDataset {
    /// The raw text samples.
    pub texts: Vec<String>,
    /// Optional integer class label per sample.
    pub labels: Option<Vec<usize>>,
    /// Human-readable label names (one per class).
    pub label_names: Option<Vec<String>>,
}

impl TextDataset {
    /// Construct an unlabelled dataset.
    pub fn new(texts: Vec<String>) -> Self {
        Self {
            texts,
            labels: None,
            label_names: None,
        }
    }

    /// Construct a labelled dataset.
    ///
    /// # Errors
    ///
    /// Returns an error if `texts.len() != labels.len()`.
    pub fn with_labels(
        texts: Vec<String>,
        labels: Vec<usize>,
        label_names: Vec<String>,
    ) -> Result<Self> {
        if texts.len() != labels.len() {
            return Err(DatasetsError::InvalidFormat(format!(
                "TextDataset: texts ({}) and labels ({}) must have the same length",
                texts.len(),
                labels.len()
            )));
        }
        Ok(Self {
            texts,
            labels: Some(labels),
            label_names: Some(label_names),
        })
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.texts.len()
    }

    /// Returns `true` if the dataset contains no samples.
    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Class-specific vocabulary templates
// ─────────────────────────────────────────────────────────────────────────────

/// Return a small per-class vocabulary biased towards class `class_id`.
fn class_vocab(class_id: usize, vocab_size: usize, rng: &mut Lcg) -> Vec<String> {
    // Bias words that are "characteristic" of the class.
    let bias: Vec<&str> = match class_id % 6 {
        0 => vec!["sports", "game", "team", "player", "score", "win", "match"],
        1 => vec![
            "politics",
            "government",
            "law",
            "election",
            "policy",
            "vote",
        ],
        2 => vec![
            "technology",
            "computer",
            "software",
            "data",
            "algorithm",
            "code",
        ],
        3 => vec![
            "science",
            "research",
            "study",
            "experiment",
            "result",
            "theory",
        ],
        4 => vec![
            "health",
            "medical",
            "patient",
            "treatment",
            "disease",
            "drug",
        ],
        _ => vec![
            "culture",
            "art",
            "music",
            "film",
            "book",
            "story",
            "character",
        ],
    };
    let fillers = [
        "the", "a", "is", "and", "in", "of", "to", "that", "with", "it", "this", "for", "on",
        "are", "was", "by", "an", "at", "be", "from", "as", "have", "has", "but", "not", "or",
        "they", "we", "our", "their",
    ];
    let mut vocab: Vec<String> = bias.iter().map(|w| w.to_string()).collect();
    // Fill remaining slots from fillers, cycling as needed.
    let need = vocab_size.saturating_sub(vocab.len());
    for i in 0..need {
        vocab.push(fillers[i % fillers.len()].to_string());
    }
    // Shuffle to vary word order per class.
    for i in (1..vocab.len()).rev() {
        let j = rng.next_usize(i + 1);
        vocab.swap(i, j);
    }
    vocab.truncate(vocab_size);
    vocab
}

/// Build one synthetic text sentence for a given class.
fn build_sentence(class_id: usize, vocab_size: usize, avg_words: usize, rng: &mut Lcg) -> String {
    // Word count varies around avg_words ± 30 %.
    let n_words = (avg_words as f64 * (0.7 + rng.next_f64() * 0.6)).round() as usize;
    let n_words = n_words.max(3);

    let vocab = class_vocab(class_id, vocab_size, rng);
    // First word capitalised.
    let first = &vocab[rng.next_usize(vocab.len())];
    let mut words: Vec<String> = vec![{
        let mut s = first.clone();
        if let Some(c) = s.get_mut(0..1) {
            c.make_ascii_uppercase();
        }
        s
    }];
    for _ in 1..n_words {
        words.push(vocab[rng.next_usize(vocab.len())].clone());
    }
    format!("{}.", words.join(" "))
}

// ─────────────────────────────────────────────────────────────────────────────
// make_text_classification
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a synthetic multi-class text classification dataset.
///
/// Each sample is a short pseudo-sentence whose vocabulary is biased towards its
/// class, making the task learnable by bag-of-words models.
///
/// # Arguments
///
/// * `n_samples`  – Total number of text samples.
/// * `n_classes`  – Number of distinct classes (1 – 32).
/// * `vocab_size` – Number of distinct word types per class (≥ 3).
/// * `avg_words`  – Average sentence length in words (≥ 3).
/// * `seed`       – Reproducibility seed.
///
/// # Errors
///
/// Returns an error if `n_classes == 0`, `vocab_size < 3`, or `avg_words < 3`.
pub fn make_text_classification(
    n_samples: usize,
    n_classes: usize,
    vocab_size: usize,
    avg_words: usize,
    seed: u64,
) -> Result<TextDataset> {
    if n_classes == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_text_classification: n_classes must be >= 1".to_string(),
        ));
    }
    if vocab_size < 3 {
        return Err(DatasetsError::InvalidFormat(
            "make_text_classification: vocab_size must be >= 3".to_string(),
        ));
    }
    if avg_words < 3 {
        return Err(DatasetsError::InvalidFormat(
            "make_text_classification: avg_words must be >= 3".to_string(),
        ));
    }
    if n_samples == 0 {
        let label_names: Vec<String> = (0..n_classes).map(|i| format!("class_{i}")).collect();
        return TextDataset::with_labels(vec![], vec![], label_names);
    }

    let mut rng = Lcg::new(seed);
    let mut texts = Vec::with_capacity(n_samples);
    let mut labels = Vec::with_capacity(n_samples);

    // Round-robin class assignment to ensure balance.
    for i in 0..n_samples {
        let class_id = i % n_classes;
        labels.push(class_id);
        texts.push(build_sentence(class_id, vocab_size, avg_words, &mut rng));
    }

    let label_names: Vec<String> = (0..n_classes).map(|i| format!("class_{i}")).collect();
    TextDataset::with_labels(texts, labels, label_names)
}

// ─────────────────────────────────────────────────────────────────────────────
// make_sentiment_dataset
// ─────────────────────────────────────────────────────────────────────────────

const POSITIVE_WORDS: &[&str] = &[
    "excellent",
    "great",
    "wonderful",
    "amazing",
    "fantastic",
    "love",
    "perfect",
    "brilliant",
    "superb",
    "outstanding",
    "impressive",
    "recommend",
    "enjoy",
    "positive",
    "good",
    "happy",
    "beautiful",
    "best",
    "helpful",
    "pleased",
];

const NEGATIVE_WORDS: &[&str] = &[
    "terrible",
    "awful",
    "horrible",
    "hate",
    "worst",
    "disappointing",
    "poor",
    "bad",
    "useless",
    "pathetic",
    "waste",
    "broken",
    "frustrating",
    "annoying",
    "mediocre",
    "failed",
    "refused",
    "ugly",
    "slow",
    "boring",
];

const NEUTRAL_WORDS: &[&str] = &[
    "the",
    "a",
    "is",
    "in",
    "of",
    "to",
    "it",
    "this",
    "for",
    "on",
    "was",
    "by",
    "product",
    "service",
    "experience",
    "time",
    "day",
    "place",
    "thought",
    "said",
    "made",
    "quite",
    "very",
    "really",
    "actually",
    "just",
];

fn build_sentiment_text(positive: bool, rng: &mut Lcg) -> String {
    let bias = if positive {
        POSITIVE_WORDS
    } else {
        NEGATIVE_WORDS
    };
    let n_bias = 2 + rng.next_usize(3); // 2-4 biased words
    let n_neutral = 5 + rng.next_usize(8); // 5-12 neutral words

    let mut words: Vec<String> = Vec::with_capacity(n_bias + n_neutral);
    for _ in 0..n_bias {
        words.push(bias[rng.next_usize(bias.len())].to_string());
    }
    for _ in 0..n_neutral {
        words.push(NEUTRAL_WORDS[rng.next_usize(NEUTRAL_WORDS.len())].to_string());
    }
    // Simple shuffle.
    for i in (1..words.len()).rev() {
        let j = rng.next_usize(i + 1);
        words.swap(i, j);
    }
    if let Some(w) = words.first_mut() {
        if let Some(c) = w.get_mut(0..1) {
            c.make_ascii_uppercase();
        }
    }
    format!("{}.", words.join(" "))
}

/// Generate a synthetic binary sentiment analysis dataset.
///
/// Label `0` = negative, label `1` = positive.  Texts are constructed from
/// domain-typical vocabulary so that simple bag-of-words classifiers can learn.
///
/// # Arguments
///
/// * `n_samples` – Total number of samples (balanced 50/50).
/// * `seed`      – Reproducibility seed.
///
/// # Errors
///
/// Returns an error only on internal construction failure.
pub fn make_sentiment_dataset(n_samples: usize, seed: u64) -> Result<TextDataset> {
    let mut rng = Lcg::new(seed);
    let mut texts = Vec::with_capacity(n_samples);
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let positive = i % 2 == 1;
        texts.push(build_sentiment_text(positive, &mut rng));
        labels.push(usize::from(positive));
    }

    let label_names = vec!["negative".to_string(), "positive".to_string()];
    TextDataset::with_labels(texts, labels, label_names)
}

// ─────────────────────────────────────────────────────────────────────────────
// NerDataset
// ─────────────────────────────────────────────────────────────────────────────

/// A Named-Entity Recognition dataset in CoNLL / IOB2 format.
#[derive(Debug, Clone)]
pub struct NerDataset {
    /// Tokenised sentences (one `Vec<String>` per sentence).
    pub sentences: Vec<Vec<String>>,
    /// IOB2 label sequence aligned with each token (`B-TYPE`, `I-TYPE`, `O`).
    pub labels: Vec<Vec<String>>,
    /// Sorted set of unique NER tags present in the dataset.
    pub tag_vocab: Vec<String>,
}

/// Entity definitions used by the NER generator.
const NER_ENTITIES: &[(&str, &[&str])] = &[
    (
        "PER",
        &[
            "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack",
            "Karen", "Leo",
        ],
    ),
    (
        "ORG",
        &[
            "Acme Corp",
            "Beta Inc",
            "Gamma Ltd",
            "Delta Systems",
            "Epsilon AI",
            "Zeta Labs",
            "Eta Solutions",
            "Theta Group",
        ],
    ),
    (
        "LOC",
        &[
            "Tokyo", "Paris", "Berlin", "New York", "Sydney", "London", "Toronto", "Beijing",
            "Cairo", "Oslo",
        ],
    ),
    (
        "DATE",
        &[
            "Monday",
            "Tuesday",
            "January 2026",
            "last year",
            "next week",
            "2024",
            "yesterday",
            "tomorrow",
        ],
    ),
];

const NER_FILLER: &[&str] = &[
    "the",
    "a",
    "and",
    "is",
    "was",
    "in",
    "on",
    "at",
    "for",
    "with",
    "announced",
    "said",
    "reported",
    "visited",
    "joined",
    "founded",
    "signed",
    "launched",
    "met",
    "worked",
];

fn tokenise(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|t| t.trim_matches(',').to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

fn build_ner_sentence(rng: &mut Lcg) -> (Vec<String>, Vec<String>) {
    // Template: FILLER ENTITY FILLER [ENTITY FILLER]*
    let mut tokens: Vec<String> = Vec::new();
    let mut labels: Vec<String> = Vec::new();

    let n_segments = 2 + rng.next_usize(3); // 2-4 entity mentions
    for _ in 0..n_segments {
        // 1-3 filler words
        let nf = 1 + rng.next_usize(3);
        for _ in 0..nf {
            tokens.push(NER_FILLER[rng.next_usize(NER_FILLER.len())].to_string());
            labels.push("O".to_string());
        }
        // One entity (might be multi-token like "Acme Corp")
        let (entity_type, names) = NER_ENTITIES[rng.next_usize(NER_ENTITIES.len())];
        let entity_text = names[rng.next_usize(names.len())];
        let entity_tokens = tokenise(entity_text);
        for (ti, tok) in entity_tokens.iter().enumerate() {
            tokens.push(tok.clone());
            if ti == 0 {
                labels.push(format!("B-{entity_type}"));
            } else {
                labels.push(format!("I-{entity_type}"));
            }
        }
    }

    // Capitalise first token.
    if let Some(t) = tokens.first_mut() {
        if let Some(c) = t.get_mut(0..1) {
            c.make_ascii_uppercase();
        }
    }

    (tokens, labels)
}

/// Generate a synthetic CoNLL-style NER dataset.
///
/// The dataset uses IOB2 labelling with four entity types: PER, ORG, LOC, DATE.
///
/// # Arguments
///
/// * `n_sentences` – Number of sentences to generate.
/// * `seed`        – Reproducibility seed.
///
/// # Errors
///
/// Returns an error if `n_sentences == 0`.
pub fn make_ner_dataset(n_sentences: usize, seed: u64) -> Result<NerDataset> {
    if n_sentences == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_ner_dataset: n_sentences must be >= 1".to_string(),
        ));
    }

    let mut rng = Lcg::new(seed);
    let mut sentences = Vec::with_capacity(n_sentences);
    let mut all_labels = Vec::with_capacity(n_sentences);

    for _ in 0..n_sentences {
        let (toks, lbls) = build_ner_sentence(&mut rng);
        sentences.push(toks);
        all_labels.push(lbls);
    }

    // Build sorted tag vocabulary.
    let mut tag_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    for lbls in &all_labels {
        for lbl in lbls {
            tag_set.insert(lbl.clone());
        }
    }
    let mut tag_vocab: Vec<String> = tag_set.into_iter().collect();
    tag_vocab.sort();

    Ok(NerDataset {
        sentences,
        labels: all_labels,
        tag_vocab,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// QaDataset
// ─────────────────────────────────────────────────────────────────────────────

/// A simple extractive question-answering dataset.
///
/// Each sample consists of a `context` paragraph, a `question`, and an
/// `answer` that is a substring of the context with its start character offset.
#[derive(Debug, Clone)]
pub struct QaDataset {
    /// Short paragraphs serving as reading context.
    pub contexts: Vec<String>,
    /// Questions answerable from the corresponding context.
    pub questions: Vec<String>,
    /// `(start_char_offset, answer_text)` pairs — answers are spans in `contexts`.
    pub answers: Vec<(usize, String)>,
}

/// Template contexts for QA generation.
const QA_TEMPLATES: &[(&str, &str, &str, usize)] = &[
    // (context, question, answer_text, answer_start)
    (
        "The Eiffel Tower is located in Paris, France. It was built in 1889.",
        "Where is the Eiffel Tower located?",
        "Paris, France",
        36,
    ),
    (
        "Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of relativity.",
        "When was Albert Einstein born?",
        "1879",
        47,
    ),
    (
        "The Amazon River is the largest river in the world by discharge volume.",
        "Which river has the largest discharge volume?",
        "Amazon River",
        4,
    ),
    (
        "Python was created by Guido van Rossum and first released in 1991.",
        "Who created Python?",
        "Guido van Rossum",
        18,
    ),
    (
        "The Great Wall of China stretches over 21,000 kilometres.",
        "How long is the Great Wall of China?",
        "21,000 kilometres",
        40,
    ),
    (
        "Marie Curie was the first woman to win a Nobel Prize, in Physics in 1903.",
        "What prize did Marie Curie win in 1903?",
        "Nobel Prize",
        43,
    ),
    (
        "The human genome contains approximately 3 billion base pairs.",
        "How many base pairs does the human genome contain?",
        "3 billion",
        41,
    ),
    (
        "Jupiter is the largest planet in the solar system.",
        "Which is the largest planet in the solar system?",
        "Jupiter",
        0,
    ),
];

/// Generate a synthetic extractive QA dataset by cycling template entries.
///
/// # Arguments
///
/// * `n_samples` – Number of QA pairs.
/// * `seed`      – Reproducibility seed (currently unused; kept for API consistency).
///
/// # Errors
///
/// Returns an error if `n_samples == 0`.
pub fn make_qa_dataset(n_samples: usize, seed: u64) -> Result<QaDataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_qa_dataset: n_samples must be >= 1".to_string(),
        ));
    }
    let _ = seed; // seed reserved for future stochastic augmentation

    let mut contexts = Vec::with_capacity(n_samples);
    let mut questions = Vec::with_capacity(n_samples);
    let mut answers = Vec::with_capacity(n_samples);

    let n_templates = QA_TEMPLATES.len();
    for i in 0..n_samples {
        let (ctx, q, ans, start) = QA_TEMPLATES[i % n_templates];
        contexts.push(ctx.to_string());
        questions.push(q.to_string());
        answers.push((start, ans.to_string()));
    }

    Ok(QaDataset {
        contexts,
        questions,
        answers,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// make_lm_dataset
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a synthetic language modelling dataset as tokenised integer sequences.
///
/// Each sequence of length `seq_len` is drawn from a bigram-like model: the next
/// token depends weakly on the previous one, creating mild local structure.
/// Token `0` is the BOS marker, token `vocab_size - 1` is the EOS marker.
///
/// # Arguments
///
/// * `n_samples`  – Number of sequences.
/// * `seq_len`    – Length of each sequence (including BOS + EOS tokens).
/// * `vocab_size` – Vocabulary size (must be ≥ 3).
/// * `seed`       – Reproducibility seed.
///
/// # Returns
///
/// A `Vec<Vec<usize>>` of shape `(n_samples, seq_len)`.
///
/// # Errors
///
/// Returns an error if `vocab_size < 3` or `seq_len < 2`.
pub fn make_lm_dataset(
    n_samples: usize,
    seq_len: usize,
    vocab_size: usize,
    seed: u64,
) -> Result<Vec<Vec<usize>>> {
    if vocab_size < 3 {
        return Err(DatasetsError::InvalidFormat(
            "make_lm_dataset: vocab_size must be >= 3".to_string(),
        ));
    }
    if seq_len < 2 {
        return Err(DatasetsError::InvalidFormat(
            "make_lm_dataset: seq_len must be >= 2 (BOS + at least one token)".to_string(),
        ));
    }

    let bos: usize = 0;
    let eos: usize = vocab_size - 1;
    // Interior tokens: 1 .. vocab_size-2.
    let n_interior = vocab_size.saturating_sub(2);

    let mut rng = Lcg::new(seed);
    let mut sequences = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let mut seq = Vec::with_capacity(seq_len);
        seq.push(bos);
        let mut prev = bos;
        for pos in 1..seq_len {
            if pos == seq_len - 1 {
                seq.push(eos);
            } else if n_interior == 0 {
                // Degenerate case: only BOS and EOS exist.
                seq.push(rng.next_usize(vocab_size));
            } else {
                // Bigram bias: with prob 0.4 continue near prev, else random.
                let tok = if rng.next_f64() < 0.4 && prev > 0 && prev < eos {
                    // Drift ±1 around prev, staying in interior range.
                    let delta = if rng.next_f64() < 0.5 {
                        1usize
                    } else {
                        n_interior.saturating_sub(1)
                    };
                    ((prev - 1 + delta) % n_interior) + 1
                } else {
                    rng.next_usize(n_interior) + 1
                };
                prev = tok;
                seq.push(tok);
            }
        }
        sequences.push(seq);
    }

    Ok(sequences)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_classification_basic() {
        let ds = make_text_classification(30, 3, 20, 10, 42).expect("text classification failed");
        assert_eq!(ds.len(), 30);
        assert!(!ds.is_empty());
        let labels = ds.labels.as_ref().expect("labels should be Some");
        assert_eq!(labels.len(), 30);
        for &l in labels {
            assert!(l < 3, "label out of range: {l}");
        }
        let label_names = ds.label_names.as_ref().expect("label names should be Some");
        assert_eq!(label_names.len(), 3);
    }

    #[test]
    fn test_text_classification_zero_samples() {
        let ds =
            make_text_classification(0, 2, 10, 5, 1).expect("empty classification should succeed");
        assert_eq!(ds.len(), 0);
        assert!(ds.is_empty());
    }

    #[test]
    fn test_text_classification_invalid() {
        assert!(make_text_classification(10, 0, 10, 5, 1).is_err());
        assert!(make_text_classification(10, 2, 2, 5, 1).is_err());
        assert!(make_text_classification(10, 2, 10, 2, 1).is_err());
    }

    #[test]
    fn test_sentiment_dataset() {
        let ds = make_sentiment_dataset(20, 7).expect("sentiment failed");
        assert_eq!(ds.len(), 20);
        let labels = ds.labels.as_ref().expect("labels Some");
        for &l in labels {
            assert!(l < 2);
        }
        let label_names = ds.label_names.as_ref().expect("label names Some");
        assert_eq!(label_names, &["negative", "positive"]);
    }

    #[test]
    fn test_ner_dataset() {
        let ds = make_ner_dataset(10, 99).expect("ner failed");
        assert_eq!(ds.sentences.len(), 10);
        assert_eq!(ds.labels.len(), 10);
        for (sent, lbls) in ds.sentences.iter().zip(ds.labels.iter()) {
            assert_eq!(sent.len(), lbls.len(), "token/label length mismatch");
        }
        assert!(!ds.tag_vocab.is_empty());
        // IOB2 check.
        for lbls in &ds.labels {
            for lbl in lbls {
                assert!(
                    lbl == "O" || lbl.starts_with("B-") || lbl.starts_with("I-"),
                    "unexpected NER tag: {lbl}"
                );
            }
        }
    }

    #[test]
    fn test_ner_empty_error() {
        assert!(make_ner_dataset(0, 1).is_err());
    }

    #[test]
    fn test_qa_dataset() {
        let ds = make_qa_dataset(16, 42).expect("qa failed");
        assert_eq!(ds.contexts.len(), 16);
        assert_eq!(ds.questions.len(), 16);
        assert_eq!(ds.answers.len(), 16);
        for ((ctx, _q), (start, ans)) in ds
            .contexts
            .iter()
            .zip(ds.questions.iter())
            .zip(ds.answers.iter())
        {
            // Answer must be a substring starting at the stated offset.
            assert!(
                ctx.len() >= start + ans.len(),
                "answer offset out of range in context"
            );
            assert_eq!(&ctx[*start..start + ans.len()], ans.as_str());
        }
    }

    #[test]
    fn test_qa_empty_error() {
        assert!(make_qa_dataset(0, 1).is_err());
    }

    #[test]
    fn test_lm_dataset() {
        let seqs = make_lm_dataset(50, 20, 100, 13).expect("lm failed");
        assert_eq!(seqs.len(), 50);
        for seq in &seqs {
            assert_eq!(seq.len(), 20);
            assert_eq!(seq[0], 0, "BOS must be 0");
            assert_eq!(seq[seq.len() - 1], 99, "EOS must be vocab_size-1");
            for &tok in seq {
                assert!(tok < 100, "token out of vocab");
            }
        }
    }

    #[test]
    fn test_lm_invalid() {
        assert!(make_lm_dataset(10, 20, 2, 1).is_err()); // vocab_size < 3
        assert!(make_lm_dataset(10, 1, 10, 1).is_err()); // seq_len < 2
    }

    #[test]
    fn test_reproducibility() {
        let a = make_text_classification(10, 2, 10, 5, 123).expect("a");
        let b = make_text_classification(10, 2, 10, 5, 123).expect("b");
        assert_eq!(a.texts, b.texts);
        let c = make_text_classification(10, 2, 10, 5, 456).expect("c");
        // Different seed → different texts (with high probability).
        assert_ne!(a.texts, c.texts);
    }
}
