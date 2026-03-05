//! Composable NLP Processing Pipelines
//!
//! This module provides a flexible, composable NLP pipeline system that lets you
//! chain text-processing steps together with a builder pattern.
//!
//! # Architecture
//!
//! ```text
//! PipelineBuilder  ──add_step()──►  NlpPipeline  ──process()──►  Vec<String>
//!                                       │
//!                                  BatchProcessor ──process_batch()──► Vec<Vec<String>>
//! ```
//!
//! # Available Steps
//!
//! | Step | Description |
//! |------|-------------|
//! | [`PipelineStep::Tokenize`] | Split text into word tokens |
//! | [`PipelineStep::Lowercase`] | Convert all tokens to lowercase |
//! | [`PipelineStep::RemoveStopwords`] | Drop common English stop-words |
//! | [`PipelineStep::RemovePunctuation`] | Strip punctuation-only tokens |
//! | [`PipelineStep::Stem`] | Porter-stem each token |
//! | [`PipelineStep::Lemmatize`] | WordNet-lemmatize each token |
//! | [`PipelineStep::NGrams`] | Expand unigrams into N-gram tokens |
//! | [`PipelineStep::Custom`] | Apply a user-supplied closure |
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::pipeline::{PipelineBuilder, PipelineStep};
//!
//! let pipeline = PipelineBuilder::new()
//!     .add_step(PipelineStep::Tokenize)
//!     .add_step(PipelineStep::Lowercase)
//!     .add_step(PipelineStep::RemoveStopwords)
//!     .add_step(PipelineStep::Stem)
//!     .build();
//!
//! let tokens = pipeline.process("The quick brown foxes jumped over the lazy dogs").unwrap();
//! assert!(!tokens.is_empty());
//! ```
//!
//! # Batch Processing
//!
//! ```rust
//! use scirs2_text::pipeline::{BatchProcessor, PipelineBuilder, PipelineStep};
//!
//! let pipeline = PipelineBuilder::new()
//!     .add_step(PipelineStep::Tokenize)
//!     .add_step(PipelineStep::Lowercase)
//!     .add_step(PipelineStep::RemoveStopwords)
//!     .build();
//!
//! let docs = vec![
//!     "Hello world, how are you?",
//!     "The quick brown fox jumps.",
//! ];
//! let processor = BatchProcessor::new(pipeline);
//! let results = processor.process_batch(&docs).unwrap();
//! assert_eq!(results.len(), 2);
//! ```

use crate::error::{Result, TextError};
use crate::lemmatization::Lemmatizer;
use crate::lemmatization::WordNetLemmatizer;
use crate::stemming::{PorterStemmer, Stemmer};
use scirs2_core::parallel_ops::*;
use std::collections::HashSet;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Built-in English stop-word list (top ~175 words)
// ─────────────────────────────────────────────────────────────────────────────

fn default_stopwords() -> HashSet<&'static str> {
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "can",
        "could",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "not",
        "only",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "because",
        "as",
        "until",
        "while",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "any",
        "also",
        "if",
        "though",
        "although",
        "because",
        "since",
        "unless",
        "whether",
        "nor",
        "neither",
        "either",
        "both",
        "like",
        "across",
        "among",
        "along",
        "around",
        "near",
        "within",
        "without",
        "toward",
        "towards",
        "via",
        "per",
        "upon",
        "onto",
        "beside",
        "besides",
        "behind",
    ]
    .iter()
    .copied()
    .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline step definitions
// ─────────────────────────────────────────────────────────────────────────────

/// A single text-processing step in an [`NlpPipeline`].
#[derive(Clone)]
pub enum PipelineStep {
    /// Tokenize the raw text into word tokens.
    ///
    /// This step must be first (or operate on a single-element `Vec<String>`
    /// that contains the full sentence).  It applies simple whitespace /
    /// punctuation-aware splitting.
    Tokenize,

    /// Convert every token to lowercase.
    Lowercase,

    /// Remove tokens that appear in the built-in English stop-word list.
    RemoveStopwords,

    /// Remove tokens consisting entirely of punctuation characters.
    RemovePunctuation,

    /// Porter-stem each token.
    Stem,

    /// WordNet-lemmatize each token (noun form, then verb fallback).
    Lemmatize,

    /// Expand the current token list into N-grams of size `n`.
    ///
    /// For example, with `n = 2` the list `["quick", "brown", "fox"]`
    /// becomes `["quick_brown", "brown_fox"]`.
    NGrams(usize),

    /// Apply a custom transformation closure to the token list.
    ///
    /// The closure receives a `Vec<String>` and returns a new `Vec<String>`.
    Custom(Arc<dyn Fn(Vec<String>) -> Vec<String> + Send + Sync>),
}

impl std::fmt::Debug for PipelineStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineStep::Tokenize => write!(f, "Tokenize"),
            PipelineStep::Lowercase => write!(f, "Lowercase"),
            PipelineStep::RemoveStopwords => write!(f, "RemoveStopwords"),
            PipelineStep::RemovePunctuation => write!(f, "RemovePunctuation"),
            PipelineStep::Stem => write!(f, "Stem"),
            PipelineStep::Lemmatize => write!(f, "Lemmatize"),
            PipelineStep::NGrams(n) => write!(f, "NGrams({n})"),
            PipelineStep::Custom(_) => write!(f, "Custom(..)"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NLP Pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// An ordered sequence of [`PipelineStep`]s that transforms raw text into
/// a list of processed tokens.
///
/// Create instances with [`PipelineBuilder`].
pub struct NlpPipeline {
    steps: Vec<PipelineStep>,
    stopwords: HashSet<&'static str>,
    stemmer: PorterStemmer,
    lemmatizer: WordNetLemmatizer,
}

impl std::fmt::Debug for NlpPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NlpPipeline")
            .field("steps", &self.steps)
            .finish()
    }
}

impl NlpPipeline {
    /// Create a new pipeline from the given steps.
    ///
    /// Prefer using [`PipelineBuilder`] which provides a more ergonomic API.
    pub fn new(steps: Vec<PipelineStep>) -> Self {
        Self {
            steps,
            stopwords: default_stopwords(),
            stemmer: PorterStemmer::new(),
            lemmatizer: WordNetLemmatizer::new(),
        }
    }

    /// Process a single text string, returning the resulting tokens.
    ///
    /// If no [`PipelineStep::Tokenize`] step is present, the entire text is
    /// treated as a single token.
    pub fn process(&self, text: &str) -> Result<Vec<String>> {
        // Start with the full text as a single token; the Tokenize step
        // will split it into individual words.
        let mut tokens: Vec<String> = vec![text.to_string()];

        for step in &self.steps {
            tokens = self.apply_step(step, tokens)?;
        }

        Ok(tokens)
    }

    /// Return an immutable view of the configured pipeline steps.
    pub fn steps(&self) -> &[PipelineStep] {
        &self.steps
    }

    // ── Internal step dispatch ──────────────────────────────────────────────

    fn apply_step(&self, step: &PipelineStep, tokens: Vec<String>) -> Result<Vec<String>> {
        match step {
            PipelineStep::Tokenize => self.step_tokenize(tokens),
            PipelineStep::Lowercase => Ok(Self::step_lowercase(tokens)),
            PipelineStep::RemoveStopwords => Ok(self.step_remove_stopwords(tokens)),
            PipelineStep::RemovePunctuation => Ok(Self::step_remove_punctuation(tokens)),
            PipelineStep::Stem => self.step_stem(tokens),
            PipelineStep::Lemmatize => self.step_lemmatize(tokens),
            PipelineStep::NGrams(n) => Self::step_ngrams(tokens, *n),
            PipelineStep::Custom(f) => Ok(f(tokens)),
        }
    }

    /// Tokenize: split each current token further on whitespace / word
    /// boundaries.  After this step every entry in the vector is a single word.
    fn step_tokenize(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let mut out = Vec::new();
        for tok in tokens {
            // Simple unicode-aware word extraction
            let words = extract_words(&tok);
            out.extend(words);
        }
        Ok(out)
    }

    fn step_lowercase(tokens: Vec<String>) -> Vec<String> {
        tokens.into_iter().map(|t| t.to_lowercase()).collect()
    }

    fn step_remove_stopwords(&self, tokens: Vec<String>) -> Vec<String> {
        tokens
            .into_iter()
            .filter(|t| !self.stopwords.contains(t.to_lowercase().as_str()))
            .collect()
    }

    fn step_remove_punctuation(tokens: Vec<String>) -> Vec<String> {
        tokens
            .into_iter()
            .filter(|t| t.chars().any(|c| c.is_alphanumeric()))
            .collect()
    }

    fn step_stem(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        tokens
            .into_iter()
            .map(|t| {
                self.stemmer
                    .stem(&t)
                    .map_err(|e| TextError::ProcessingError(e.to_string()))
            })
            .collect()
    }

    fn step_lemmatize(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        tokens
            .into_iter()
            .map(|t| {
                self.lemmatizer
                    .lemmatize(&t, None)
                    .map_err(|e| TextError::ProcessingError(e.to_string()))
            })
            .collect()
    }

    fn step_ngrams(tokens: Vec<String>, n: usize) -> Result<Vec<String>> {
        if n == 0 {
            return Err(TextError::InvalidInput("NGrams n must be >= 1".to_string()));
        }
        if n == 1 {
            return Ok(tokens);
        }
        if tokens.len() < n {
            return Ok(Vec::new());
        }

        let grams = tokens.windows(n).map(|window| window.join("_")).collect();

        Ok(grams)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline Builder
// ─────────────────────────────────────────────────────────────────────────────

/// Fluent builder for constructing an [`NlpPipeline`].
///
/// # Example
///
/// ```rust
/// use scirs2_text::pipeline::{PipelineBuilder, PipelineStep};
///
/// let pipeline = PipelineBuilder::new()
///     .add_step(PipelineStep::Tokenize)
///     .add_step(PipelineStep::Lowercase)
///     .add_step(PipelineStep::RemovePunctuation)
///     .add_step(PipelineStep::RemoveStopwords)
///     .add_step(PipelineStep::Lemmatize)
///     .build();
///
/// let tokens = pipeline.process("The running dogs are playing").unwrap();
/// assert!(!tokens.is_empty());
/// ```
#[derive(Debug, Default)]
pub struct PipelineBuilder {
    steps: Vec<PipelineStep>,
}

impl PipelineBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Append a pipeline step.
    pub fn add_step(mut self, step: PipelineStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Construct the [`NlpPipeline`].
    pub fn build(self) -> NlpPipeline {
        NlpPipeline::new(self.steps)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch Processor
// ─────────────────────────────────────────────────────────────────────────────

/// Processes multiple documents through an [`NlpPipeline`] in parallel using
/// the `scirs2-core` parallel abstractions.
///
/// # Example
///
/// ```rust
/// use scirs2_text::pipeline::{BatchProcessor, PipelineBuilder, PipelineStep};
///
/// let pipeline = PipelineBuilder::new()
///     .add_step(PipelineStep::Tokenize)
///     .add_step(PipelineStep::Lowercase)
///     .add_step(PipelineStep::RemoveStopwords)
///     .build();
///
/// let docs = vec![
///     "The quick brown fox",
///     "A lazy dog sleeps",
///     "Hello world",
/// ];
/// let processor = BatchProcessor::new(pipeline);
/// let results = processor.process_batch(&docs).unwrap();
/// assert_eq!(results.len(), 3);
/// ```
pub struct BatchProcessor {
    pipeline: Arc<NlpPipeline>,
    /// Minimum documents per thread for parallel dispatch.
    parallel_threshold: usize,
}

impl BatchProcessor {
    /// Create a new `BatchProcessor` wrapping the given pipeline.
    pub fn new(pipeline: NlpPipeline) -> Self {
        Self {
            pipeline: Arc::new(pipeline),
            parallel_threshold: 32,
        }
    }

    /// Set the minimum number of documents required before parallel processing
    /// is used.  Defaults to 32.
    pub fn with_parallel_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    /// Process a slice of text documents and return one token list per document.
    pub fn process_batch(&self, documents: &[&str]) -> Result<Vec<Vec<String>>> {
        if documents.len() < self.parallel_threshold {
            // Sequential for small batches (avoids thread-spawn overhead)
            documents
                .iter()
                .map(|doc| self.pipeline.process(doc))
                .collect()
        } else {
            // Parallel processing via scirs2-core parallel_ops
            let pipeline = Arc::clone(&self.pipeline);
            let results: Vec<Result<Vec<String>>> = documents
                .par_iter()
                .map(|doc| pipeline.process(doc))
                .collect();

            results.into_iter().collect()
        }
    }

    /// Process documents and also return any per-document errors instead of
    /// short-circuiting on the first failure.
    pub fn process_batch_tolerant(
        &self,
        documents: &[&str],
    ) -> Vec<std::result::Result<Vec<String>, TextError>> {
        if documents.len() < self.parallel_threshold {
            documents
                .iter()
                .map(|doc| self.pipeline.process(doc))
                .collect()
        } else {
            let pipeline = Arc::clone(&self.pipeline);
            documents
                .par_iter()
                .map(|doc| pipeline.process(doc))
                .collect()
        }
    }

    /// Return a reference to the inner pipeline.
    pub fn pipeline(&self) -> &NlpPipeline {
        &self.pipeline
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Predefined pipeline factories
// ─────────────────────────────────────────────────────────────────────────────

/// Create a standard tokenization + lowercasing + stop-word removal pipeline.
pub fn basic_pipeline() -> NlpPipeline {
    PipelineBuilder::new()
        .add_step(PipelineStep::Tokenize)
        .add_step(PipelineStep::Lowercase)
        .add_step(PipelineStep::RemovePunctuation)
        .add_step(PipelineStep::RemoveStopwords)
        .build()
}

/// Create a pipeline that tokenizes, lowercases, removes stop-words, then
/// Porter-stems.
pub fn stemming_pipeline() -> NlpPipeline {
    PipelineBuilder::new()
        .add_step(PipelineStep::Tokenize)
        .add_step(PipelineStep::Lowercase)
        .add_step(PipelineStep::RemovePunctuation)
        .add_step(PipelineStep::RemoveStopwords)
        .add_step(PipelineStep::Stem)
        .build()
}

/// Create a pipeline that tokenizes, lowercases, removes stop-words, then
/// WordNet-lemmatizes.
pub fn lemmatization_pipeline() -> NlpPipeline {
    PipelineBuilder::new()
        .add_step(PipelineStep::Tokenize)
        .add_step(PipelineStep::Lowercase)
        .add_step(PipelineStep::RemovePunctuation)
        .add_step(PipelineStep::RemoveStopwords)
        .add_step(PipelineStep::Lemmatize)
        .build()
}

/// Create an N-gram feature-extraction pipeline.  Returns bigrams (n=2) by
/// default; pass `n` to override.
pub fn ngram_pipeline(n: usize) -> NlpPipeline {
    PipelineBuilder::new()
        .add_step(PipelineStep::Tokenize)
        .add_step(PipelineStep::Lowercase)
        .add_step(PipelineStep::RemovePunctuation)
        .add_step(PipelineStep::RemoveStopwords)
        .add_step(PipelineStep::NGrams(n))
        .build()
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: word extraction
// ─────────────────────────────────────────────────────────────────────────────

/// Extract alphabetic/numeric word tokens from `text`, stripping surrounding
/// punctuation from each token.
fn extract_words(text: &str) -> Vec<String> {
    // Split on whitespace first, then strip leading/trailing non-alphanumeric
    text.split_whitespace()
        .filter_map(|raw| {
            // Strip leading and trailing non-alphanumeric characters
            let trimmed: String = raw.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PipelineBuilder ─────────────────────────────────────────────────────

    #[test]
    fn test_builder_creates_pipeline() {
        let pipeline = PipelineBuilder::new()
            .add_step(PipelineStep::Tokenize)
            .add_step(PipelineStep::Lowercase)
            .build();
        assert_eq!(pipeline.steps().len(), 2);
    }

    #[test]
    fn test_tokenize_step() {
        let pipeline = PipelineBuilder::new()
            .add_step(PipelineStep::Tokenize)
            .build();
        let tokens = pipeline.process("hello world foo").unwrap();
        assert_eq!(tokens, vec!["hello", "world", "foo"]);
    }

    #[test]
    fn test_lowercase_step() {
        let pipeline = PipelineBuilder::new()
            .add_step(PipelineStep::Tokenize)
            .add_step(PipelineStep::Lowercase)
            .build();
        let tokens = pipeline.process("Hello World FOO").unwrap();
        assert_eq!(tokens, vec!["hello", "world", "foo"]);
    }

    #[test]
    fn test_remove_punctuation_step() {
        let pipeline = PipelineBuilder::new()
            .add_step(PipelineStep::Tokenize)
            .add_step(PipelineStep::RemovePunctuation)
            .build();
        let tokens = pipeline.process("Hello, world! This is a test.").unwrap();
        // Punctuation-only tokens should be gone; "Hello," becomes "Hello"
        // because extract_words strips surrounding punctuation already.
        assert!(tokens
            .iter()
            .all(|t| t.chars().any(|c| c.is_alphanumeric())));
    }

    #[test]
    fn test_remove_stopwords_step() {
        let pipeline = PipelineBuilder::new()
            .add_step(PipelineStep::Tokenize)
            .add_step(PipelineStep::Lowercase)
            .add_step(PipelineStep::RemoveStopwords)
            .build();
        let tokens = pipeline
            .process("the quick brown fox is a fast animal")
            .unwrap();
        // "the", "is", "a" should be removed
        assert!(!tokens.contains(&"the".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
        // Content words should remain
        assert!(tokens.contains(&"quick".to_string()));
        assert!(tokens.contains(&"brown".to_string()));
        assert!(tokens.contains(&"fox".to_string()));
    }

    #[test]
    fn test_stem_step() {
        let pipeline = PipelineBuilder::new()
            .add_step(PipelineStep::Tokenize)
            .add_step(PipelineStep::Lowercase)
            .add_step(PipelineStep::Stem)
            .build();
        let tokens = pipeline.process("running dogs are jumping").unwrap();
        // Porter stems
        assert!(tokens.contains(&"run".to_string()), "tokens: {tokens:?}");
        assert!(tokens.contains(&"dog".to_string()), "tokens: {tokens:?}");
        assert!(tokens.contains(&"jump".to_string()), "tokens: {tokens:?}");
    }

    #[test]
    fn test_lemmatize_step() {
        let pipeline = PipelineBuilder::new()
            .add_step(PipelineStep::Tokenize)
            .add_step(PipelineStep::Lowercase)
            .add_step(PipelineStep::Lemmatize)
            .build();
        let tokens = pipeline.process("The cats went to the mice").unwrap();
        // WordNet lemmatizer should resolve irregular forms
        assert!(
            tokens.contains(&"cat".to_string()) || tokens.contains(&"cats".to_string()),
            "tokens: {tokens:?}"
        );
    }

    #[test]
    fn test_ngrams_step() {
        let pipeline = PipelineBuilder::new()
            .add_step(PipelineStep::Tokenize)
            .add_step(PipelineStep::Lowercase)
            .add_step(PipelineStep::NGrams(2))
            .build();
        let tokens = pipeline.process("quick brown fox").unwrap();
        assert_eq!(tokens, vec!["quick_brown", "brown_fox"]);
    }

    #[test]
    fn test_ngrams_step_trigram() {
        let pipeline = PipelineBuilder::new()
            .add_step(PipelineStep::Tokenize)
            .add_step(PipelineStep::NGrams(3))
            .build();
        let tokens = pipeline.process("a b c d").unwrap();
        assert_eq!(tokens, vec!["a_b_c", "b_c_d"]);
    }

    #[test]
    fn test_ngrams_invalid_n() {
        let pipeline = PipelineBuilder::new()
            .add_step(PipelineStep::Tokenize)
            .add_step(PipelineStep::NGrams(0))
            .build();
        let result = pipeline.process("hello world");
        assert!(result.is_err());
    }

    #[test]
    fn test_ngrams_too_short() {
        let pipeline = PipelineBuilder::new()
            .add_step(PipelineStep::Tokenize)
            .add_step(PipelineStep::NGrams(5))
            .build();
        let tokens = pipeline.process("hi").unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_custom_step() {
        let pipeline = PipelineBuilder::new()
            .add_step(PipelineStep::Tokenize)
            .add_step(PipelineStep::Custom(Arc::new(|tokens| {
                tokens.into_iter().filter(|t| t.len() > 3).collect()
            })))
            .build();
        let tokens = pipeline.process("I am the quick brown fox").unwrap();
        assert!(tokens.iter().all(|t| t.len() > 3));
    }

    #[test]
    fn test_empty_input() {
        let pipeline = PipelineBuilder::new()
            .add_step(PipelineStep::Tokenize)
            .add_step(PipelineStep::Lowercase)
            .build();
        let tokens = pipeline.process("").unwrap();
        // Empty string — no tokens
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_full_pipeline() {
        let pipeline = PipelineBuilder::new()
            .add_step(PipelineStep::Tokenize)
            .add_step(PipelineStep::Lowercase)
            .add_step(PipelineStep::RemovePunctuation)
            .add_step(PipelineStep::RemoveStopwords)
            .add_step(PipelineStep::Stem)
            .build();
        let tokens = pipeline
            .process("The quick brown foxes are jumping over the lazy dogs!")
            .unwrap();
        // Stop-words removed, content words stemmed
        assert!(!tokens.contains(&"the".to_string()));
        assert!(!tokens.contains(&"are".to_string()));
        // "foxes" → "fox" → Porter stem
        assert!(tokens.iter().any(|t| t == "fox" || t.starts_with("fox")));
        assert!(!tokens.is_empty());
    }

    // ── BatchProcessor ──────────────────────────────────────────────────────

    #[test]
    fn test_batch_processor_basic() {
        let pipeline = basic_pipeline();
        let processor = BatchProcessor::new(pipeline);
        let docs = vec![
            "The quick brown fox",
            "Hello world this is a test",
            "Machine learning is fascinating",
        ];
        let results = processor.process_batch(&docs).unwrap();
        assert_eq!(results.len(), 3);
        for (doc, tokens) in docs.iter().zip(results.iter()) {
            assert!(!tokens.is_empty(), "expected tokens for doc: {doc}");
        }
    }

    #[test]
    fn test_batch_processor_parallel() {
        // Force parallel path by setting threshold=0
        let pipeline = stemming_pipeline();
        let processor = BatchProcessor::new(pipeline).with_parallel_threshold(0);

        let docs: Vec<&str> = (0..100)
            .map(|_| "running foxes jumping over lazy dogs")
            .collect();

        let results = processor.process_batch(&docs).unwrap();
        assert_eq!(results.len(), 100);
        // Every result should contain "fox" (stem of "foxes")
        for tokens in &results {
            assert!(
                tokens.iter().any(|t| t == "fox"),
                "expected 'fox' in {tokens:?}"
            );
        }
    }

    #[test]
    fn test_batch_processor_tolerant() {
        let pipeline = basic_pipeline();
        let processor = BatchProcessor::new(pipeline);
        let docs = vec!["hello world", "the quick brown fox"];
        let results = processor.process_batch_tolerant(&docs);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[test]
    fn test_batch_processor_empty_doc() {
        let pipeline = basic_pipeline();
        let processor = BatchProcessor::new(pipeline);
        let docs = vec!["", "hello world"];
        let results = processor.process_batch(&docs).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].is_empty());
        assert!(!results[1].is_empty());
    }

    // ── Predefined pipelines ────────────────────────────────────────────────

    #[test]
    fn test_basic_pipeline_factory() {
        let pipeline = basic_pipeline();
        let tokens = pipeline.process("The fox is quick and agile").unwrap();
        assert!(!tokens.contains(&"the".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
    }

    #[test]
    fn test_stemming_pipeline_factory() {
        let pipeline = stemming_pipeline();
        let tokens = pipeline.process("The dogs are running fast").unwrap();
        assert!(!tokens.contains(&"the".to_string()));
        assert!(tokens.contains(&"dog".to_string()));
        assert!(tokens.contains(&"run".to_string()));
    }

    #[test]
    fn test_lemmatization_pipeline_factory() {
        let pipeline = lemmatization_pipeline();
        let tokens = pipeline.process("The mice went to sleep").unwrap();
        assert!(!tokens.contains(&"the".to_string()));
        // "mice" → "mouse" via WordNet
        assert!(tokens.contains(&"mouse".to_string()));
        // "went" → "go" via WordNet exception
        assert!(tokens.contains(&"go".to_string()));
    }

    #[test]
    fn test_ngram_pipeline_factory() {
        let pipeline = ngram_pipeline(2);
        let tokens = pipeline.process("quick brown fox").unwrap();
        // All tokens should be bigrams (joined with _)
        for tok in &tokens {
            assert!(tok.contains('_'), "expected bigram, got: {tok}");
        }
    }

    // ── Extract words utility ───────────────────────────────────────────────

    #[test]
    fn test_extract_words_strips_punctuation() {
        let words = extract_words("Hello, world! Foo-bar.");
        assert!(words.contains(&"Hello".to_string()), "{words:?}");
        assert!(words.contains(&"world".to_string()), "{words:?}");
        assert!(words.contains(&"Foo-bar".to_string()), "{words:?}");
    }

    #[test]
    fn test_extract_words_empty() {
        let words = extract_words("   ");
        assert!(words.is_empty());
    }
}
