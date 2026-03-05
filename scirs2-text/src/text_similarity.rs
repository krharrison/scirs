//! Text similarity module
//!
//! Provides multiple methods for computing similarity between text strings:
//!
//! - **TF-IDF Cosine**: Cosine similarity over TF-IDF document vectors
//! - **Jaccard**: Token-level and character n-gram Jaccard similarity
//! - **BM25**: Okapi BM25 relevance scoring
//! - **Edit Distance**: Normalised Levenshtein similarity
//!
//! All methods are available through the unified [`text_similarity`] function
//! or through dedicated structs for finer-grained configuration.

use crate::error::{Result, TextError};
use crate::tokenize::{Tokenizer, WordTokenizer};
use crate::vectorize::{TfidfVectorizer, Vectorizer};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Similarity method selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityMethod {
    /// Cosine similarity on TF-IDF vectors.
    TfIdfCosine,
    /// Token-level Jaccard similarity.
    Jaccard,
    /// Character n-gram Jaccard similarity.
    CharNgramJaccard {
        /// N-gram size (e.g. 3 for trigrams).
        n: usize,
    },
    /// BM25 relevance score (text2 is treated as the "document", text1 as the "query").
    Bm25,
    /// Normalised Levenshtein similarity (1 - normalised_distance).
    EditDistance,
}

/// Result of a similarity computation.
#[derive(Debug, Clone)]
pub struct SimilarityResult {
    /// The computed similarity score. Range depends on the method:
    /// - Cosine, Jaccard, EditDistance: [0, 1]
    /// - BM25: [0, +inf)
    pub score: f64,
    /// The method that was used.
    pub method: SimilarityMethod,
}

// ---------------------------------------------------------------------------
// Unified API
// ---------------------------------------------------------------------------

/// Compute similarity between `text1` and `text2` using the specified method.
///
/// # Errors
///
/// Returns an error when tokenization fails or the method parameters are
/// invalid (e.g. a zero n-gram size).
pub fn text_similarity(
    text1: &str,
    text2: &str,
    method: SimilarityMethod,
) -> Result<SimilarityResult> {
    let score = match method {
        SimilarityMethod::TfIdfCosine => tfidf_cosine_similarity(text1, text2)?,
        SimilarityMethod::Jaccard => jaccard_token_similarity(text1, text2)?,
        SimilarityMethod::CharNgramJaccard { n } => char_ngram_jaccard_similarity(text1, text2, n)?,
        SimilarityMethod::Bm25 => bm25_score(text1, text2)?,
        SimilarityMethod::EditDistance => edit_distance_similarity(text1, text2),
    };
    Ok(SimilarityResult { score, method })
}

// ---------------------------------------------------------------------------
// TF-IDF cosine similarity
// ---------------------------------------------------------------------------

/// Compute cosine similarity between two texts via TF-IDF vectors.
///
/// Both texts are treated as a mini-corpus (2 documents) to derive IDF weights.
pub fn tfidf_cosine_similarity(text1: &str, text2: &str) -> Result<f64> {
    if text1.trim().is_empty() && text2.trim().is_empty() {
        return Ok(1.0);
    }
    if text1.trim().is_empty() || text2.trim().is_empty() {
        return Ok(0.0);
    }

    let docs = vec![text1, text2];
    let mut vectorizer = TfidfVectorizer::default();
    let matrix = vectorizer.fit_transform(&docs)?;

    let n = matrix.ncols();
    if n == 0 {
        return Ok(0.0);
    }

    let mut dot = 0.0_f64;
    let mut norm1_sq = 0.0_f64;
    let mut norm2_sq = 0.0_f64;

    for col in 0..n {
        let a = matrix[[0, col]];
        let b = matrix[[1, col]];
        dot += a * b;
        norm1_sq += a * a;
        norm2_sq += b * b;
    }

    let norm1 = norm1_sq.sqrt();
    let norm2 = norm2_sq.sqrt();

    if norm1 == 0.0 || norm2 == 0.0 {
        return Ok(0.0);
    }

    Ok(dot / (norm1 * norm2))
}

/// Configurable TF-IDF cosine similarity calculator.
pub struct TfIdfCosineSimilarity {
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
}

impl TfIdfCosineSimilarity {
    /// Create a new calculator with the default word tokenizer.
    pub fn new() -> Self {
        Self {
            tokenizer: Box::new(WordTokenizer::default()),
        }
    }

    /// Compute the similarity between two texts.
    pub fn similarity(&self, text1: &str, text2: &str) -> Result<f64> {
        tfidf_cosine_similarity(text1, text2)
    }

    /// Compute pairwise similarities for a corpus.
    /// Returns a flat vector of (i, j, score) for all i < j.
    pub fn pairwise(&self, texts: &[&str]) -> Result<Vec<(usize, usize, f64)>> {
        if texts.len() < 2 {
            return Ok(Vec::new());
        }

        let mut vectorizer = TfidfVectorizer::default();
        let matrix = vectorizer.fit_transform(texts)?;
        let n = texts.len();
        let cols = matrix.ncols();
        let mut results = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let mut dot = 0.0_f64;
                let mut n1 = 0.0_f64;
                let mut n2 = 0.0_f64;
                for c in 0..cols {
                    let a = matrix[[i, c]];
                    let b = matrix[[j, c]];
                    dot += a * b;
                    n1 += a * a;
                    n2 += b * b;
                }
                let sim = if n1 == 0.0 || n2 == 0.0 {
                    0.0
                } else {
                    dot / (n1.sqrt() * n2.sqrt())
                };
                results.push((i, j, sim));
            }
        }

        Ok(results)
    }
}

impl Default for TfIdfCosineSimilarity {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Jaccard similarity
// ---------------------------------------------------------------------------

/// Token-level Jaccard similarity.
pub fn jaccard_token_similarity(text1: &str, text2: &str) -> Result<f64> {
    let tokenizer = WordTokenizer::default();
    let tokens1 = tokenizer.tokenize(text1)?;
    let tokens2 = tokenizer.tokenize(text2)?;

    if tokens1.is_empty() && tokens2.is_empty() {
        return Ok(1.0);
    }

    let set1: HashSet<&str> = tokens1.iter().map(|s| s.as_str()).collect();
    let set2: HashSet<&str> = tokens2.iter().map(|s| s.as_str()).collect();

    let intersection = set1.intersection(&set2).count();
    let union = set1.union(&set2).count();

    if union == 0 {
        return Ok(1.0);
    }

    Ok(intersection as f64 / union as f64)
}

/// Character n-gram Jaccard similarity.
///
/// Computes Jaccard similarity over the sets of character n-grams derived
/// from both texts.
pub fn char_ngram_jaccard_similarity(text1: &str, text2: &str, n: usize) -> Result<f64> {
    if n == 0 {
        return Err(TextError::InvalidInput(
            "N-gram size must be at least 1".to_string(),
        ));
    }

    let ngrams1 = char_ngrams(text1, n);
    let ngrams2 = char_ngrams(text2, n);

    if ngrams1.is_empty() && ngrams2.is_empty() {
        return Ok(1.0);
    }

    let set1: HashSet<&str> = ngrams1.iter().map(|s| s.as_str()).collect();
    let set2: HashSet<&str> = ngrams2.iter().map(|s| s.as_str()).collect();

    let intersection = set1.intersection(&set2).count();
    let union = set1.union(&set2).count();

    if union == 0 {
        return Ok(1.0);
    }

    Ok(intersection as f64 / union as f64)
}

/// Extract character n-grams from text.
fn char_ngrams(text: &str, n: usize) -> Vec<String> {
    let lower = text.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    if chars.len() < n {
        return Vec::new();
    }
    chars.windows(n).map(|w| w.iter().collect()).collect()
}

// ---------------------------------------------------------------------------
// BM25
// ---------------------------------------------------------------------------

/// BM25 (Okapi Best Matching 25) scoring parameters.
#[derive(Debug, Clone)]
pub struct Bm25Config {
    /// Term frequency saturation parameter (typical: 1.2 - 2.0).
    pub k1: f64,
    /// Length normalization parameter (typical: 0.75).
    pub b: f64,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self { k1: 1.5, b: 0.75 }
    }
}

/// Compute BM25 relevance of `document` with respect to `query`.
///
/// The function treats the query and document each as a single text.
/// IDF is estimated from a synthetic 2-document corpus for simplicity;
/// for a full corpus-aware BM25, use [`Bm25Scorer`].
pub fn bm25_score(query: &str, document: &str) -> Result<f64> {
    let scorer = Bm25Scorer::new(Bm25Config::default());
    scorer.score_single(query, document)
}

/// A BM25 scorer that can score a query against an entire corpus.
pub struct Bm25Scorer {
    config: Bm25Config,
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
}

impl Bm25Scorer {
    /// Create a new BM25 scorer with the given configuration.
    pub fn new(config: Bm25Config) -> Self {
        Self {
            config,
            tokenizer: Box::new(WordTokenizer::default()),
        }
    }

    /// Score a single `query` against a single `document`.
    pub fn score_single(&self, query: &str, document: &str) -> Result<f64> {
        let query_tokens = self.tokenizer.tokenize(query)?;
        let doc_tokens = self.tokenizer.tokenize(document)?;

        if query_tokens.is_empty() || doc_tokens.is_empty() {
            return Ok(0.0);
        }

        // Document term frequencies.
        let mut doc_tf: HashMap<String, f64> = HashMap::new();
        for token in &doc_tokens {
            *doc_tf.entry(token.clone()).or_insert(0.0) += 1.0;
        }

        let doc_len = doc_tokens.len() as f64;
        // For a single-document scenario we use avgdl = doc_len.
        let avgdl = doc_len;
        // Total docs = 1, containing term => df = 1 for matching terms.
        let n_docs = 1.0_f64;

        let mut score = 0.0_f64;
        let query_terms: HashSet<String> = query_tokens.into_iter().collect();

        for term in &query_terms {
            let tf = doc_tf.get(term).copied().unwrap_or(0.0);
            if tf == 0.0 {
                continue;
            }
            let df = 1.0_f64; // the document contains the term
            let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln();
            let numerator = tf * (self.config.k1 + 1.0);
            let denominator =
                tf + self.config.k1 * (1.0 - self.config.b + self.config.b * doc_len / avgdl);
            score += idf * numerator / denominator;
        }

        Ok(score)
    }

    /// Score a query against every document in a corpus.
    /// Returns a vector of (document_index, score) sorted by score descending.
    pub fn score_corpus(&self, query: &str, corpus: &[&str]) -> Result<Vec<(usize, f64)>> {
        let query_tokens = self.tokenizer.tokenize(query)?;
        if query_tokens.is_empty() {
            return Ok(Vec::new());
        }

        let n_docs = corpus.len() as f64;

        // Tokenize all documents and compute term frequencies.
        let mut all_doc_tokens: Vec<Vec<String>> = Vec::with_capacity(corpus.len());
        let mut all_doc_tf: Vec<HashMap<String, f64>> = Vec::with_capacity(corpus.len());
        let mut total_len: f64 = 0.0;

        for doc in corpus {
            let tokens = self.tokenizer.tokenize(doc)?;
            total_len += tokens.len() as f64;
            let mut tf_map: HashMap<String, f64> = HashMap::new();
            for token in &tokens {
                *tf_map.entry(token.clone()).or_insert(0.0) += 1.0;
            }
            all_doc_tokens.push(tokens);
            all_doc_tf.push(tf_map);
        }

        let avgdl = if corpus.is_empty() {
            1.0
        } else {
            total_len / n_docs
        };

        // Compute document frequency for each query term.
        let query_terms: HashSet<String> = query_tokens.into_iter().collect();
        let mut df: HashMap<String, f64> = HashMap::new();
        for term in &query_terms {
            let count = all_doc_tf
                .iter()
                .filter(|tf_map| tf_map.contains_key(term))
                .count();
            df.insert(term.clone(), count as f64);
        }

        // Score each document.
        let mut results: Vec<(usize, f64)> = Vec::with_capacity(corpus.len());
        for (idx, doc_tf) in all_doc_tf.iter().enumerate() {
            let doc_len = all_doc_tokens[idx].len() as f64;
            let mut score = 0.0_f64;

            for term in &query_terms {
                let tf = doc_tf.get(term).copied().unwrap_or(0.0);
                if tf == 0.0 {
                    continue;
                }
                let term_df = df.get(term).copied().unwrap_or(0.0);
                let idf = ((n_docs - term_df + 0.5) / (term_df + 0.5) + 1.0).ln();
                let numerator = tf * (self.config.k1 + 1.0);
                let denominator =
                    tf + self.config.k1 * (1.0 - self.config.b + self.config.b * doc_len / avgdl);
                score += idf * numerator / denominator;
            }

            results.push((idx, score));
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Edit distance similarity
// ---------------------------------------------------------------------------

/// Normalised Levenshtein similarity: `1.0 - (edit_distance / max_len)`.
///
/// Returns 1.0 for identical strings and 0.0 for completely different strings.
pub fn edit_distance_similarity(text1: &str, text2: &str) -> f64 {
    if text1.is_empty() && text2.is_empty() {
        return 1.0;
    }

    let distance = levenshtein(text1, text2);
    let max_len = std::cmp::max(text1.chars().count(), text2.chars().count()) as f64;

    if max_len == 0.0 {
        return 1.0;
    }

    1.0 - (distance as f64 / max_len)
}

/// Levenshtein distance between two strings (O(n*m) DP).
fn levenshtein(s1: &str, s2: &str) -> usize {
    let a: Vec<char> = s1.chars().collect();
    let b: Vec<char> = s2.chars().collect();
    let m = a.len();
    let n = b.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use a single-row DP to save memory.
    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for j in 0..=n {
        prev[j] = j;
    }

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = std::cmp::min(
                std::cmp::min(prev[j] + 1, curr[j - 1] + 1),
                prev[j - 1] + cost,
            );
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- TF-IDF cosine ----

    #[test]
    fn test_tfidf_cosine_identical() {
        let score = tfidf_cosine_similarity("hello world", "hello world").expect("Should succeed");
        assert!(
            (score - 1.0).abs() < 1e-6,
            "Identical texts should have similarity ~1.0, got {}",
            score
        );
    }

    #[test]
    fn test_tfidf_cosine_different() {
        let score = tfidf_cosine_similarity("the cat sat", "purple dinosaur robot")
            .expect("Should succeed");
        assert!(
            score < 0.5,
            "Very different texts should have low similarity, got {}",
            score
        );
    }

    #[test]
    fn test_tfidf_cosine_empty() {
        assert!(
            (tfidf_cosine_similarity("", "").expect("ok") - 1.0).abs() < 1e-6,
            "Both empty -> 1.0"
        );
        assert!(
            tfidf_cosine_similarity("hello", "").expect("ok").abs() < 1e-6,
            "One empty -> 0.0"
        );
    }

    #[test]
    fn test_tfidf_cosine_symmetric() {
        let ab = tfidf_cosine_similarity("machine learning", "deep learning").expect("ok");
        let ba = tfidf_cosine_similarity("deep learning", "machine learning").expect("ok");
        assert!((ab - ba).abs() < 1e-10, "Cosine should be symmetric");
    }

    #[test]
    fn test_tfidf_cosine_partial_overlap() {
        let score =
            tfidf_cosine_similarity("the quick brown fox", "the quick red car").expect("ok");
        assert!(score > 0.0 && score < 1.0);
    }

    #[test]
    fn test_tfidf_pairwise() {
        let calc = TfIdfCosineSimilarity::new();
        let texts = vec!["hello world", "hello there", "goodbye moon"];
        let pairs = calc.pairwise(&texts).expect("ok");
        assert_eq!(pairs.len(), 3); // C(3,2) = 3
        for (i, j, s) in &pairs {
            assert!(*i < *j);
            assert!(*s >= 0.0);
        }
    }

    // ---- Jaccard ----

    #[test]
    fn test_jaccard_identical() {
        let score = jaccard_token_similarity("hello world", "hello world").expect("ok");
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let score = jaccard_token_similarity("alpha beta", "gamma delta").expect("ok");
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_partial() {
        let score = jaccard_token_similarity("cat dog bird", "cat dog fish").expect("ok");
        // intersection={cat, dog}=2, union={cat, dog, bird, fish}=4 -> 0.5
        assert!((score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_empty() {
        let score = jaccard_token_similarity("", "").expect("ok");
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_symmetric() {
        let ab = jaccard_token_similarity("a b c", "b c d").expect("ok");
        let ba = jaccard_token_similarity("b c d", "a b c").expect("ok");
        assert!((ab - ba).abs() < 1e-10);
    }

    // ---- Char n-gram Jaccard ----

    #[test]
    fn test_char_ngram_identical() {
        let score = char_ngram_jaccard_similarity("hello", "hello", 3).expect("ok");
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_char_ngram_different() {
        let score = char_ngram_jaccard_similarity("abcdef", "uvwxyz", 3).expect("ok");
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn test_char_ngram_partial() {
        let score = char_ngram_jaccard_similarity("abcde", "abcfg", 2).expect("ok");
        // ab, bc common; cd, de vs cf, fg unique
        assert!(score > 0.0 && score < 1.0);
    }

    #[test]
    fn test_char_ngram_zero_n() {
        let result = char_ngram_jaccard_similarity("a", "b", 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_char_ngram_empty() {
        let score = char_ngram_jaccard_similarity("", "", 2).expect("ok");
        assert!((score - 1.0).abs() < 1e-6);
    }

    // ---- BM25 ----

    #[test]
    fn test_bm25_matching_query() {
        let score = bm25_score(
            "machine learning",
            "machine learning is a subset of artificial intelligence",
        )
        .expect("ok");
        assert!(score > 0.0);
    }

    #[test]
    fn test_bm25_no_match() {
        let score = bm25_score("quantum physics", "the cat sat on the mat").expect("ok");
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn test_bm25_empty() {
        let score = bm25_score("", "anything").expect("ok");
        assert!(score.abs() < 1e-6);
        let score = bm25_score("anything", "").expect("ok");
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn test_bm25_corpus_scoring() {
        let scorer = Bm25Scorer::new(Bm25Config::default());
        let corpus = vec![
            "machine learning algorithms",
            "deep learning neural networks",
            "cooking recipes for dinner",
        ];
        let results = scorer
            .score_corpus("learning algorithms", &corpus)
            .expect("ok");
        assert_eq!(results.len(), 3);
        // First result should be the most relevant document.
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_bm25_corpus_empty_query() {
        let scorer = Bm25Scorer::new(Bm25Config::default());
        let results = scorer.score_corpus("", &["a", "b"]).expect("ok");
        assert!(results.is_empty());
    }

    // ---- Edit distance ----

    #[test]
    fn test_edit_distance_identical() {
        let score = edit_distance_similarity("kitten", "kitten");
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_edit_distance_completely_different() {
        let score = edit_distance_similarity("abc", "xyz");
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_edit_distance_known_value() {
        // "kitten" -> "sitting" = distance 3, max_len 7 => similarity = 4/7
        let score = edit_distance_similarity("kitten", "sitting");
        let expected = 1.0 - 3.0 / 7.0;
        assert!(
            (score - expected).abs() < 1e-6,
            "Expected {}, got {}",
            expected,
            score
        );
    }

    #[test]
    fn test_edit_distance_empty() {
        assert!((edit_distance_similarity("", "") - 1.0).abs() < 1e-6);
        assert!(edit_distance_similarity("abc", "").abs() < 1e-6);
    }

    #[test]
    fn test_edit_distance_symmetric() {
        let ab = edit_distance_similarity("abc", "aec");
        let ba = edit_distance_similarity("aec", "abc");
        assert!((ab - ba).abs() < 1e-10);
    }

    // ---- Unified API ----

    #[test]
    fn test_unified_api_all_methods() {
        let methods = vec![
            SimilarityMethod::TfIdfCosine,
            SimilarityMethod::Jaccard,
            SimilarityMethod::CharNgramJaccard { n: 3 },
            SimilarityMethod::Bm25,
            SimilarityMethod::EditDistance,
        ];
        for method in methods {
            let result =
                text_similarity("hello world", "hello there", method).expect("Should succeed");
            assert!(
                result.score >= 0.0,
                "Score should be non-negative for {:?}",
                method
            );
        }
    }

    #[test]
    fn test_unified_api_returns_correct_method() {
        let result = text_similarity("a", "b", SimilarityMethod::Jaccard).expect("ok");
        assert_eq!(result.method, SimilarityMethod::Jaccard);
    }
}
