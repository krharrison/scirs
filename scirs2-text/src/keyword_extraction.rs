//! Keyword extraction module
//!
//! This module provides multiple algorithms for extracting keywords from text:
//!
//! - **TF-IDF**: Term Frequency-Inverse Document Frequency scoring
//! - **TextRank**: Graph-based ranking using co-occurrence networks
//! - **RAKE**: Rapid Automatic Keyword Extraction using phrase delimiters
//!
//! Each method can be used independently or through the unified `extract_keywords` function.

use crate::error::{Result, TextError};
use crate::tokenize::{Tokenizer, WordTokenizer};
use crate::vectorize::{TfidfVectorizer, Vectorizer};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single extracted keyword together with its relevance score.
#[derive(Debug, Clone)]
pub struct Keyword {
    /// The keyword text (may be a single word or a multi-word phrase).
    pub text: String,
    /// Relevance score (higher is more relevant). Scale depends on the method.
    pub score: f64,
}

/// Method used for keyword extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeywordMethod {
    /// TF-IDF based extraction.
    TfIdf,
    /// TextRank graph-based extraction.
    TextRank,
    /// Rapid Automatic Keyword Extraction.
    Rake,
}

// ---------------------------------------------------------------------------
// Unified API
// ---------------------------------------------------------------------------

/// Extract keywords from `text` using the specified `method`.
///
/// Returns up to `top_k` keywords sorted by descending score.
///
/// # Errors
///
/// Returns an error if the text cannot be tokenized or if the chosen method
/// encounters an internal error.
pub fn extract_keywords(text: &str, method: KeywordMethod, top_k: usize) -> Result<Vec<Keyword>> {
    match method {
        KeywordMethod::TfIdf => {
            let extractor = TfIdfKeywordExtractor::new();
            extractor.extract(text, top_k)
        }
        KeywordMethod::TextRank => {
            let extractor = TextRankKeywordExtractor::new();
            extractor.extract(text, top_k)
        }
        KeywordMethod::Rake => {
            let extractor = RakeKeywordExtractor::new();
            extractor.extract(text, top_k)
        }
    }
}

// ---------------------------------------------------------------------------
// TF-IDF keyword extraction
// ---------------------------------------------------------------------------

/// Extracts keywords by splitting the text into pseudo-documents (sentences)
/// and computing per-term TF-IDF scores, then averaging across sentences.
pub struct TfIdfKeywordExtractor {
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
    /// Minimum token length to consider.
    min_token_len: usize,
}

impl TfIdfKeywordExtractor {
    /// Create a new TF-IDF keyword extractor with default settings.
    pub fn new() -> Self {
        Self {
            tokenizer: Box::new(WordTokenizer::default()),
            min_token_len: 2,
        }
    }

    /// Set the minimum token length (tokens shorter than this are ignored).
    pub fn with_min_token_len(mut self, len: usize) -> Self {
        self.min_token_len = len;
        self
    }

    /// Extract `top_k` keywords from `text`.
    pub fn extract(&self, text: &str, top_k: usize) -> Result<Vec<Keyword>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        // Split into sentences as pseudo-documents.
        let sentences = split_sentences(text);
        if sentences.is_empty() {
            return Ok(Vec::new());
        }

        let sentence_refs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();

        let mut vectorizer = TfidfVectorizer::default();
        vectorizer.fit(&sentence_refs)?;
        let tfidf_matrix = vectorizer.transform_batch(&sentence_refs)?;

        // Build vocabulary from the tokenizer so we can map column index -> term.
        let vocab = build_vocabulary(&sentences, &*self.tokenizer)?;

        // Average the TF-IDF scores across sentences for each term.
        let n_terms = tfidf_matrix.ncols();
        let n_docs = tfidf_matrix.nrows();
        if n_terms == 0 || n_docs == 0 {
            return Ok(Vec::new());
        }

        let mut avg_scores: Vec<f64> = Vec::with_capacity(n_terms);
        for col_idx in 0..n_terms {
            let col_sum: f64 = (0..n_docs).map(|row| tfidf_matrix[[row, col_idx]]).sum();
            avg_scores.push(col_sum / n_docs as f64);
        }

        // Map scores to terms.
        let mut keyword_scores: Vec<Keyword> = Vec::new();
        for (idx, &score) in avg_scores.iter().enumerate() {
            if score <= 0.0 {
                continue;
            }
            if let Some(term) = vocab.get(&idx) {
                if term.len() >= self.min_token_len {
                    keyword_scores.push(Keyword {
                        text: term.clone(),
                        score,
                    });
                }
            }
        }

        keyword_scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        keyword_scores.truncate(top_k);
        Ok(keyword_scores)
    }
}

impl Default for TfIdfKeywordExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TextRank keyword extraction
// ---------------------------------------------------------------------------

/// TextRank-based keyword extraction.
///
/// Constructs a co-occurrence graph of words within a sliding window and
/// applies the PageRank algorithm to score each word. Multi-word keywords
/// are formed by merging adjacent high-scoring words.
pub struct TextRankKeywordExtractor {
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
    /// Window size for co-occurrence.
    window_size: usize,
    /// Damping factor for PageRank.
    damping: f64,
    /// Maximum PageRank iterations.
    max_iterations: usize,
    /// Convergence threshold.
    convergence_threshold: f64,
    /// Minimum token length.
    min_token_len: usize,
}

impl TextRankKeywordExtractor {
    /// Create a new TextRank keyword extractor with default parameters.
    pub fn new() -> Self {
        Self {
            tokenizer: Box::new(WordTokenizer::default()),
            window_size: 4,
            damping: 0.85,
            max_iterations: 100,
            convergence_threshold: 1e-5,
            min_token_len: 2,
        }
    }

    /// Set the co-occurrence window size.
    pub fn with_window_size(mut self, size: usize) -> Result<Self> {
        if size < 2 {
            return Err(TextError::InvalidInput(
                "Window size must be at least 2".to_string(),
            ));
        }
        self.window_size = size;
        Ok(self)
    }

    /// Set the damping factor.
    pub fn with_damping(mut self, d: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&d) {
            return Err(TextError::InvalidInput(
                "Damping factor must be between 0 and 1".to_string(),
            ));
        }
        self.damping = d;
        Ok(self)
    }

    /// Extract `top_k` keywords from `text`.
    pub fn extract(&self, text: &str, top_k: usize) -> Result<Vec<Keyword>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let tokens = self.tokenizer.tokenize(text)?;
        let filtered: Vec<String> = tokens
            .into_iter()
            .filter(|t| t.len() >= self.min_token_len && !is_stopword(t))
            .collect();

        if filtered.is_empty() {
            return Ok(Vec::new());
        }

        // Build co-occurrence graph (adjacency list with weights).
        let mut graph: HashMap<String, HashMap<String, f64>> = HashMap::new();
        for window in filtered.windows(self.window_size) {
            for i in 0..window.len() {
                for j in (i + 1)..window.len() {
                    let a = &window[i];
                    let b = &window[j];
                    *graph
                        .entry(a.clone())
                        .or_default()
                        .entry(b.clone())
                        .or_insert(0.0) += 1.0;
                    *graph
                        .entry(b.clone())
                        .or_default()
                        .entry(a.clone())
                        .or_insert(0.0) += 1.0;
                }
            }
        }

        // PageRank-style scoring.
        let nodes: Vec<String> = graph.keys().cloned().collect();
        let n = nodes.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let node_idx: HashMap<&str, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();

        let mut scores = vec![1.0 / n as f64; n];

        // Pre-compute out-degree sums.
        let out_sums: Vec<f64> = nodes
            .iter()
            .map(|node| {
                graph
                    .get(node)
                    .map(|neighbors| neighbors.values().sum::<f64>())
                    .unwrap_or(0.0)
            })
            .collect();

        for _ in 0..self.max_iterations {
            let mut new_scores = vec![(1.0 - self.damping) / n as f64; n];

            for (j, node_j) in nodes.iter().enumerate() {
                if out_sums[j] <= 0.0 {
                    continue;
                }
                if let Some(neighbors) = graph.get(node_j) {
                    for (neighbor, weight) in neighbors {
                        if let Some(&i) = node_idx.get(neighbor.as_str()) {
                            new_scores[i] += self.damping * (weight / out_sums[j]) * scores[j];
                        }
                    }
                }
            }

            let diff: f64 = scores
                .iter()
                .zip(new_scores.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            scores = new_scores;
            if diff < self.convergence_threshold {
                break;
            }
        }

        // Collect single-word scores.
        let mut word_scores: HashMap<String, f64> = HashMap::new();
        for (i, node) in nodes.iter().enumerate() {
            word_scores.insert(node.clone(), scores[i]);
        }

        // Merge adjacent high-scoring words into multi-word keywords.
        let all_tokens = self.tokenizer.tokenize(text)?;
        let keywords = merge_adjacent_keywords(&all_tokens, &word_scores, top_k);

        Ok(keywords)
    }
}

impl Default for TextRankKeywordExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Merge adjacent tokens that are both in the scored map into multi-word
/// keywords, summing their scores.
fn merge_adjacent_keywords(
    tokens: &[String],
    word_scores: &HashMap<String, f64>,
    top_k: usize,
) -> Vec<Keyword> {
    let mut phrases: Vec<(Vec<String>, f64)> = Vec::new();
    let mut current_phrase: Vec<String> = Vec::new();
    let mut current_score: f64 = 0.0;

    for token in tokens {
        if let Some(&score) = word_scores.get(token) {
            current_phrase.push(token.clone());
            current_score += score;
        } else {
            if !current_phrase.is_empty() {
                phrases.push((current_phrase.clone(), current_score));
                current_phrase.clear();
                current_score = 0.0;
            }
        }
    }
    if !current_phrase.is_empty() {
        phrases.push((current_phrase, current_score));
    }

    // Deduplicate phrases.
    let mut seen: HashSet<String> = HashSet::new();
    let mut keywords: Vec<Keyword> = Vec::new();
    for (words, score) in phrases {
        let phrase_text = words.join(" ");
        if seen.contains(&phrase_text) {
            continue;
        }
        seen.insert(phrase_text.clone());
        keywords.push(Keyword {
            text: phrase_text,
            score,
        });
    }

    keywords.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    keywords.truncate(top_k);
    keywords
}

// ---------------------------------------------------------------------------
// RAKE keyword extraction
// ---------------------------------------------------------------------------

/// RAKE (Rapid Automatic Keyword Extraction).
///
/// The algorithm:
/// 1. Split text into candidate phrases using stopword delimiters.
/// 2. Build a word co-occurrence matrix from those phrases.
/// 3. Score each word as `degree(word) / frequency(word)`.
/// 4. Score each phrase as the sum of its word scores.
pub struct RakeKeywordExtractor {
    /// Minimum phrase length in words.
    min_phrase_len: usize,
    /// Maximum phrase length in words.
    max_phrase_len: usize,
    /// Minimum word length.
    min_word_len: usize,
}

impl RakeKeywordExtractor {
    /// Create a new RAKE extractor with default settings.
    pub fn new() -> Self {
        Self {
            min_phrase_len: 1,
            max_phrase_len: 4,
            min_word_len: 2,
        }
    }

    /// Set the minimum phrase length (in words).
    pub fn with_min_phrase_len(mut self, len: usize) -> Self {
        self.min_phrase_len = len;
        self
    }

    /// Set the maximum phrase length (in words).
    pub fn with_max_phrase_len(mut self, len: usize) -> Self {
        self.max_phrase_len = len;
        self
    }

    /// Extract `top_k` keywords from `text`.
    pub fn extract(&self, text: &str, top_k: usize) -> Result<Vec<Keyword>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        // 1. Generate candidate phrases by splitting on stopwords and punctuation.
        let candidates = self.generate_candidates(text);
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // 2. Compute word frequency and word degree.
        let mut word_freq: HashMap<String, f64> = HashMap::new();
        let mut word_degree: HashMap<String, f64> = HashMap::new();

        for phrase in &candidates {
            let words: Vec<&str> = phrase
                .split_whitespace()
                .filter(|w| w.len() >= self.min_word_len)
                .collect();
            let degree = words.len() as f64;
            for word in &words {
                let w = word.to_lowercase();
                *word_freq.entry(w.clone()).or_insert(0.0) += 1.0;
                *word_degree.entry(w).or_insert(0.0) += degree;
            }
        }

        // 3. Compute word scores: degree(w) / freq(w).
        let mut word_scores: HashMap<String, f64> = HashMap::new();
        for (word, freq) in &word_freq {
            let degree = word_degree.get(word).copied().unwrap_or(0.0);
            if *freq > 0.0 {
                word_scores.insert(word.clone(), degree / freq);
            }
        }

        // 4. Score each candidate phrase.
        let mut phrase_scores: Vec<Keyword> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for phrase in &candidates {
            let normalized = phrase.to_lowercase();
            if seen.contains(&normalized) {
                continue;
            }
            seen.insert(normalized.clone());

            let words: Vec<&str> = normalized
                .split_whitespace()
                .filter(|w| w.len() >= self.min_word_len)
                .collect();
            if words.is_empty() {
                continue;
            }

            let score: f64 = words
                .iter()
                .map(|w| word_scores.get(*w).copied().unwrap_or(0.0))
                .sum();

            phrase_scores.push(Keyword {
                text: normalized,
                score,
            });
        }

        phrase_scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        phrase_scores.truncate(top_k);
        Ok(phrase_scores)
    }

    /// Split text into candidate keyword phrases by using stopwords and
    /// punctuation as delimiters.
    fn generate_candidates(&self, text: &str) -> Vec<String> {
        let lower = text.to_lowercase();
        // Split on punctuation and stopwords.
        let mut candidates: Vec<String> = Vec::new();
        let mut current_phrase: Vec<String> = Vec::new();

        for word in lower.split(|c: char| !c.is_alphanumeric() && c != '\'') {
            let trimmed = word.trim();
            if trimmed.is_empty() {
                if !current_phrase.is_empty() {
                    self.add_candidate(&mut candidates, &current_phrase);
                    current_phrase.clear();
                }
                continue;
            }

            if is_stopword(trimmed) {
                if !current_phrase.is_empty() {
                    self.add_candidate(&mut candidates, &current_phrase);
                    current_phrase.clear();
                }
            } else {
                current_phrase.push(trimmed.to_string());
            }
        }

        if !current_phrase.is_empty() {
            self.add_candidate(&mut candidates, &current_phrase);
        }

        candidates
    }

    fn add_candidate(&self, candidates: &mut Vec<String>, phrase_words: &[String]) {
        if phrase_words.len() < self.min_phrase_len || phrase_words.len() > self.max_phrase_len {
            return;
        }
        let phrase = phrase_words.join(" ");
        if phrase
            .split_whitespace()
            .any(|w| w.len() >= self.min_word_len)
        {
            candidates.push(phrase);
        }
    }
}

impl Default for RakeKeywordExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Naive sentence splitter (splits on `.`, `!`, `?`).
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }
    sentences
}

/// Build a mapping from column index -> term, mirroring the vocabulary the
/// TF-IDF vectorizer builds internally.  We re-tokenize identically.
fn build_vocabulary(
    sentences: &[String],
    tokenizer: &dyn Tokenizer,
) -> Result<HashMap<usize, String>> {
    let mut term_to_idx: HashMap<String, usize> = HashMap::new();
    let mut next_idx: usize = 0;

    for sentence in sentences {
        let tokens = tokenizer.tokenize(sentence)?;
        for token in tokens {
            if let std::collections::hash_map::Entry::Vacant(e) = term_to_idx.entry(token) {
                e.insert(next_idx);
                next_idx += 1;
            }
        }
    }

    let idx_to_term: HashMap<usize, String> =
        term_to_idx.into_iter().map(|(t, i)| (i, t)).collect();
    Ok(idx_to_term)
}

/// Minimal English stopword list used by TextRank and RAKE.
fn is_stopword(word: &str) -> bool {
    const STOPWORDS: &[&str] = &[
        "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had", "do",
        "does", "did", "will", "would", "shall", "should", "may", "might", "must", "can", "could",
        "not", "no", "nor", "so", "than", "that", "this", "these", "those", "it", "its", "i", "me",
        "my", "we", "us", "our", "you", "your", "he", "him", "his", "she", "her", "they", "them",
        "their", "what", "which", "who", "whom", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such", "only", "own", "same",
        "also", "just", "about", "above", "after", "again", "against", "any", "because", "before",
        "below", "between", "during", "further", "here", "into", "once", "out", "over", "then",
        "there", "through", "under", "until", "up", "very", "while",
    ];
    STOPWORDS.contains(&word.to_lowercase().as_str())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- TF-IDF tests ----

    #[test]
    fn test_tfidf_extracts_keywords() {
        let text = "Machine learning is a powerful tool. \
                    Machine learning algorithms process data efficiently. \
                    Deep learning extends machine learning with neural networks.";
        let keywords = extract_keywords(text, KeywordMethod::TfIdf, 5)
            .expect("TF-IDF extraction should succeed");
        assert!(!keywords.is_empty());
        assert!(keywords.len() <= 5);
        // Scores should be in descending order.
        for pair in keywords.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
    }

    #[test]
    fn test_tfidf_empty_text() {
        let result =
            extract_keywords("", KeywordMethod::TfIdf, 5).expect("Empty text should not error");
        assert!(result.is_empty());
    }

    #[test]
    fn test_tfidf_single_sentence() {
        let result = extract_keywords(
            "Rust programming language is fast and safe.",
            KeywordMethod::TfIdf,
            3,
        )
        .expect("Single sentence should succeed");
        // Should still return some keywords.
        assert!(!result.is_empty());
    }

    #[test]
    fn test_tfidf_respects_top_k() {
        let text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa. \
                    Alpha beta gamma delta epsilon zeta eta theta iota kappa.";
        let result =
            extract_keywords(text, KeywordMethod::TfIdf, 3).expect("Extraction should succeed");
        assert!(result.len() <= 3);
    }

    #[test]
    fn test_tfidf_min_token_len() {
        let extractor = TfIdfKeywordExtractor::new().with_min_token_len(5);
        let text = "AI and ML are big. Artificial intelligence is growing.";
        let result = extractor
            .extract(text, 10)
            .expect("Extraction should succeed");
        // No token shorter than 5 chars.
        for kw in &result {
            for word in kw.text.split_whitespace() {
                assert!(word.len() >= 5, "Word '{}' is too short", word);
            }
        }
    }

    // ---- TextRank tests ----

    #[test]
    fn test_textrank_extracts_keywords() {
        let text = "Natural language processing enables computers to understand human language. \
                    Text mining and information retrieval are subfields of natural language processing. \
                    Sentiment analysis determines the emotional tone of text.";
        let keywords = extract_keywords(text, KeywordMethod::TextRank, 5)
            .expect("TextRank extraction should succeed");
        assert!(!keywords.is_empty());
        assert!(keywords.len() <= 5);
    }

    #[test]
    fn test_textrank_empty_text() {
        let result =
            extract_keywords("", KeywordMethod::TextRank, 5).expect("Empty text should not error");
        assert!(result.is_empty());
    }

    #[test]
    fn test_textrank_scores_descending() {
        let text = "Graph algorithms are fundamental in computer science. \
                    PageRank is a famous graph algorithm. \
                    Many applications use graph-based methods.";
        let keywords = extract_keywords(text, KeywordMethod::TextRank, 10)
            .expect("TextRank extraction should succeed");
        for pair in keywords.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
    }

    #[test]
    fn test_textrank_window_size() {
        let extractor = TextRankKeywordExtractor::new()
            .with_window_size(2)
            .expect("Window size 2 should be valid");
        let text = "Alpha beta gamma delta epsilon. Alpha beta gamma delta.";
        let result = extractor
            .extract(text, 5)
            .expect("Extraction should succeed");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_textrank_invalid_window() {
        let result = TextRankKeywordExtractor::new().with_window_size(0);
        assert!(result.is_err());
    }

    // ---- RAKE tests ----

    #[test]
    fn test_rake_extracts_keywords() {
        let text =
            "Compatibility of systems of linear constraints over the set of natural numbers. \
                    Criteria of compatibility of a system of linear Diophantine equations.";
        let keywords =
            extract_keywords(text, KeywordMethod::Rake, 5).expect("RAKE extraction should succeed");
        assert!(!keywords.is_empty());
        assert!(keywords.len() <= 5);
    }

    #[test]
    fn test_rake_empty_text() {
        let result =
            extract_keywords("", KeywordMethod::Rake, 5).expect("Empty text should not error");
        assert!(result.is_empty());
    }

    #[test]
    fn test_rake_phrase_scoring() {
        let text = "Machine learning algorithms are important. \
                    Deep learning algorithms are even more powerful. \
                    Algorithms drive modern artificial intelligence.";
        let keywords = extract_keywords(text, KeywordMethod::Rake, 10)
            .expect("RAKE extraction should succeed");
        // Multi-word phrases should generally score higher than single words
        // because RAKE sums word scores.
        assert!(!keywords.is_empty());
        for pair in keywords.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
    }

    #[test]
    fn test_rake_stopword_splitting() {
        let text = "The quick brown fox and the lazy dog.";
        let extractor = RakeKeywordExtractor::new();
        let candidates = extractor.generate_candidates(text);
        // "the", "and" are stopwords, so they should not appear in candidates.
        for candidate in &candidates {
            for word in candidate.split_whitespace() {
                assert!(!is_stopword(word), "'{}' is a stopword", word);
            }
        }
    }

    #[test]
    fn test_rake_max_phrase_len() {
        let extractor = RakeKeywordExtractor::new().with_max_phrase_len(2);
        let text = "Advanced machine learning algorithms improve natural language processing.";
        let result = extractor
            .extract(text, 10)
            .expect("Extraction should succeed");
        for kw in &result {
            let word_count = kw.text.split_whitespace().count();
            assert!(word_count <= 2, "Phrase '{}' exceeds max length", kw.text);
        }
    }

    // ---- Cross-method tests ----

    #[test]
    fn test_all_methods_non_empty_for_real_text() {
        let text = "Rust is a systems programming language focused on safety and performance. \
                    The Rust compiler prevents data races and memory errors at compile time. \
                    Many developers choose Rust for building reliable software.";
        for method in &[
            KeywordMethod::TfIdf,
            KeywordMethod::TextRank,
            KeywordMethod::Rake,
        ] {
            let keywords = extract_keywords(text, *method, 5).expect("Extraction should succeed");
            assert!(
                !keywords.is_empty(),
                "Method {:?} returned empty for real text",
                method
            );
        }
    }

    #[test]
    fn test_all_methods_handle_whitespace_only() {
        for method in &[
            KeywordMethod::TfIdf,
            KeywordMethod::TextRank,
            KeywordMethod::Rake,
        ] {
            let result =
                extract_keywords("   \t\n  ", *method, 5).expect("Whitespace should not error");
            assert!(result.is_empty());
        }
    }
}
