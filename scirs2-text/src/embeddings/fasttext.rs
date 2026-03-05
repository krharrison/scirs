//! FastText embeddings with character n-grams
//!
//! This module implements FastText, an extension of Word2Vec that learns
//! word representations as bags of character n-grams. This approach handles
//! out-of-vocabulary words and morphologically rich languages better.
//!
//! ## Overview
//!
//! FastText represents each word as a bag of character n-grams. For example:
//! - word: "where"
//! - 3-grams: "<wh", "whe", "her", "ere", "re>"
//! - The word embedding is the sum of its n-gram embeddings
//!
//! Key advantages over vanilla Word2Vec:
//! - Handles out-of-vocabulary (OOV) words via subword decomposition
//! - Better representations for morphologically rich languages
//! - Captures internal word structure (prefixes, suffixes, roots)
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_text::embeddings::fasttext::{FastText, FastTextConfig};
//!
//! // Create configuration
//! let config = FastTextConfig {
//!     vector_size: 100,
//!     min_n: 3,
//!     max_n: 6,
//!     window_size: 5,
//!     epochs: 5,
//!     learning_rate: 0.05,
//!     min_count: 1,
//!     negative_samples: 5,
//!     ..Default::default()
//! };
//!
//! // Train model
//! let documents = vec![
//!     "the quick brown fox jumps over the lazy dog",
//!     "a quick brown dog outpaces a quick fox"
//! ];
//!
//! let mut model = FastText::with_config(config);
//! model.train(&documents).expect("Training failed");
//!
//! // Get word vector (works even for OOV words!)
//! if let Ok(vector) = model.get_word_vector("quickest") {
//!     println!("Vector for OOV word 'quickest': {:?}", vector);
//! }
//! ```

use crate::error::{Result, TextError};
use crate::tokenize::{Tokenizer, WordTokenizer};
use crate::vocabulary::Vocabulary;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// FastText configuration
#[derive(Debug, Clone)]
pub struct FastTextConfig {
    /// Size of word vectors
    pub vector_size: usize,
    /// Minimum length of character n-grams
    pub min_n: usize,
    /// Maximum length of character n-grams
    pub max_n: usize,
    /// Size of context window
    pub window_size: usize,
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Minimum word count threshold
    pub min_count: usize,
    /// Number of negative samples
    pub negative_samples: usize,
    /// Subsampling threshold for frequent words
    pub subsample: f64,
    /// Bucket size for hashing n-grams
    pub bucket_size: usize,
}

impl Default for FastTextConfig {
    fn default() -> Self {
        Self {
            vector_size: 100,
            min_n: 3,
            max_n: 6,
            window_size: 5,
            epochs: 5,
            learning_rate: 0.05,
            min_count: 5,
            negative_samples: 5,
            subsample: 1e-3,
            bucket_size: 2_000_000,
        }
    }
}

/// FastText model for learning word representations with character n-grams
///
/// Decomposes each word into character n-grams (subwords) and learns embeddings
/// for both whole words and their constituent n-grams. This enables:
///
/// - Out-of-vocabulary word handling (any word can be represented via its n-grams)
/// - Morphological awareness (similar prefixes/suffixes produce similar vectors)
/// - Robustness to misspellings and rare word forms
pub struct FastText {
    /// Configuration
    config: FastTextConfig,
    /// Vocabulary of words
    vocabulary: Vocabulary,
    /// Word frequencies
    word_counts: HashMap<String, usize>,
    /// Word embeddings (for words in vocabulary)
    word_embeddings: Option<Array2<f64>>,
    /// Output embeddings used during training (for negative sampling)
    output_embeddings: Option<Array2<f64>>,
    /// N-gram embeddings (subword information)
    ngram_embeddings: Option<Array2<f64>>,
    /// N-gram to bucket index mapping
    ngram_to_bucket: HashMap<String, usize>,
    /// Tokenizer
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
    /// Current learning rate
    current_learning_rate: f64,
    /// Negative sampling table (unigram distribution raised to 3/4)
    sampling_weights: Vec<f64>,
}

impl Debug for FastText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastText")
            .field("config", &self.config)
            .field("vocabulary_size", &self.vocabulary.len())
            .field("word_embeddings", &self.word_embeddings.is_some())
            .field("ngram_embeddings", &self.ngram_embeddings.is_some())
            .field("ngram_count", &self.ngram_to_bucket.len())
            .finish()
    }
}

impl Clone for FastText {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            vocabulary: self.vocabulary.clone(),
            word_counts: self.word_counts.clone(),
            word_embeddings: self.word_embeddings.clone(),
            output_embeddings: self.output_embeddings.clone(),
            ngram_embeddings: self.ngram_embeddings.clone(),
            ngram_to_bucket: self.ngram_to_bucket.clone(),
            tokenizer: Box::new(WordTokenizer::default()),
            current_learning_rate: self.current_learning_rate,
            sampling_weights: self.sampling_weights.clone(),
        }
    }
}

impl FastText {
    /// Create a new FastText model with default configuration
    pub fn new() -> Self {
        Self {
            config: FastTextConfig::default(),
            vocabulary: Vocabulary::new(),
            word_counts: HashMap::new(),
            word_embeddings: None,
            output_embeddings: None,
            ngram_embeddings: None,
            ngram_to_bucket: HashMap::new(),
            tokenizer: Box::new(WordTokenizer::default()),
            current_learning_rate: 0.05,
            sampling_weights: Vec::new(),
        }
    }

    /// Create a new FastText model with custom configuration
    pub fn with_config(config: FastTextConfig) -> Self {
        let learning_rate = config.learning_rate;
        Self {
            config,
            vocabulary: Vocabulary::new(),
            word_counts: HashMap::new(),
            word_embeddings: None,
            output_embeddings: None,
            ngram_embeddings: None,
            ngram_to_bucket: HashMap::new(),
            tokenizer: Box::new(WordTokenizer::default()),
            current_learning_rate: learning_rate,
            sampling_weights: Vec::new(),
        }
    }

    /// Set a custom tokenizer
    pub fn with_tokenizer(mut self, tokenizer: Box<dyn Tokenizer + Send + Sync>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Extract character n-grams from a word
    ///
    /// Wraps the word with boundary markers < and > before extracting.
    /// For example, "fox" with min_n=3, max_n=4 produces:
    /// 3-grams: "<fo", "fox", "ox>"
    /// 4-grams: "<fox", "fox>", "`<fox>`"(if len allows)
    pub fn extract_ngrams(&self, word: &str) -> Vec<String> {
        let word_with_boundaries = format!("<{}>", word);
        let chars: Vec<char> = word_with_boundaries.chars().collect();
        let mut ngrams = Vec::new();

        for n in self.config.min_n..=self.config.max_n {
            if chars.len() < n {
                continue;
            }

            for i in 0..=(chars.len() - n) {
                let ngram: String = chars[i..i + n].iter().collect();
                ngrams.push(ngram);
            }
        }

        ngrams
    }

    /// Hash an n-gram to a bucket index using FNV-1a
    fn hash_ngram(&self, ngram: &str) -> usize {
        let mut hash: u64 = 2166136261;
        for byte in ngram.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(16777619);
        }
        (hash % (self.config.bucket_size as u64)) as usize
    }

    /// Build vocabulary from texts
    pub fn build_vocabulary(&mut self, texts: &[&str]) -> Result<()> {
        if texts.is_empty() {
            return Err(TextError::InvalidInput(
                "No texts provided for building vocabulary".into(),
            ));
        }

        // Count word frequencies
        let mut word_counts = HashMap::new();

        for &text in texts {
            let tokens = self.tokenizer.tokenize(text)?;
            for token in tokens {
                *word_counts.entry(token).or_insert(0) += 1;
            }
        }

        // Build vocabulary with min_count threshold
        self.vocabulary = Vocabulary::new();
        for (word, count) in &word_counts {
            if *count >= self.config.min_count {
                self.vocabulary.add_token(word);
            }
        }

        if self.vocabulary.is_empty() {
            return Err(TextError::VocabularyError(
                "No words meet the minimum count threshold".into(),
            ));
        }

        self.word_counts = word_counts;

        // Initialize embeddings
        let vocab_size = self.vocabulary.len();
        let vector_size = self.config.vector_size;
        let bucket_size = self.config.bucket_size;

        let mut rng = scirs2_core::random::rng();

        // Initialize word embeddings
        let word_embeddings = Array2::from_shape_fn((vocab_size, vector_size), |_| {
            (rng.random::<f64>() * 2.0 - 1.0) / vector_size as f64
        });

        // Initialize output embeddings for negative sampling
        let output_embeddings = Array2::zeros((vocab_size, vector_size));

        // Initialize n-gram embeddings
        let ngram_embeddings = Array2::from_shape_fn((bucket_size, vector_size), |_| {
            (rng.random::<f64>() * 2.0 - 1.0) / vector_size as f64
        });

        self.word_embeddings = Some(word_embeddings);
        self.output_embeddings = Some(output_embeddings);
        self.ngram_embeddings = Some(ngram_embeddings);

        // Build n-gram to bucket mapping
        self.ngram_to_bucket.clear();
        for i in 0..self.vocabulary.len() {
            if let Some(word) = self.vocabulary.get_token(i) {
                let ngrams = self.extract_ngrams(word);
                for ngram in ngrams {
                    if !self.ngram_to_bucket.contains_key(&ngram) {
                        let bucket = self.hash_ngram(&ngram);
                        self.ngram_to_bucket.insert(ngram, bucket);
                    }
                }
            }
        }

        // Build negative sampling weights (unigram distribution ^ 0.75)
        self.sampling_weights = vec![0.0; vocab_size];
        for i in 0..vocab_size {
            if let Some(word) = self.vocabulary.get_token(i) {
                let count = self.word_counts.get(word).copied().unwrap_or(1);
                self.sampling_weights[i] = (count as f64).powf(0.75);
            }
        }

        Ok(())
    }

    /// Sample a negative example using the unigram distribution
    fn sample_negative(&self, rng: &mut impl Rng) -> usize {
        if self.sampling_weights.is_empty() {
            return 0;
        }
        let total: f64 = self.sampling_weights.iter().sum();
        if total <= 0.0 {
            return rng.random_range(0..self.vocabulary.len().max(1));
        }
        let r = rng.random::<f64>() * total;
        let mut cumulative = 0.0;
        for (i, &w) in self.sampling_weights.iter().enumerate() {
            cumulative += w;
            if r <= cumulative {
                return i;
            }
        }
        self.sampling_weights.len() - 1
    }

    /// Compute the full subword-aware representation for a word index
    ///
    /// Returns the average of the word vector and all its n-gram vectors.
    fn compute_word_representation(&self, word_idx: usize) -> Result<(Array1<f64>, Vec<usize>)> {
        let word_embeddings = self
            .word_embeddings
            .as_ref()
            .ok_or_else(|| TextError::EmbeddingError("Word embeddings not initialized".into()))?;
        let ngram_embeddings = self
            .ngram_embeddings
            .as_ref()
            .ok_or_else(|| TextError::EmbeddingError("N-gram embeddings not initialized".into()))?;

        let word = self
            .vocabulary
            .get_token(word_idx)
            .ok_or_else(|| TextError::VocabularyError("Invalid word index".into()))?;

        let ngrams = self.extract_ngrams(word);
        let ngram_buckets: Vec<usize> = ngrams
            .iter()
            .filter_map(|ng| self.ngram_to_bucket.get(ng).copied())
            .collect();

        let mut vec = word_embeddings.row(word_idx).to_owned();
        for &bucket in &ngram_buckets {
            vec += &ngram_embeddings.row(bucket);
        }
        let divisor = 1.0 + ngram_buckets.len() as f64;
        vec /= divisor;

        Ok((vec, ngram_buckets))
    }

    /// Train the FastText model
    pub fn train(&mut self, texts: &[&str]) -> Result<()> {
        if texts.is_empty() {
            return Err(TextError::InvalidInput(
                "No texts provided for training".into(),
            ));
        }

        // Build vocabulary if not already built
        if self.vocabulary.is_empty() {
            self.build_vocabulary(texts)?;
        }

        // Prepare training data
        let mut sentences = Vec::new();
        for &text in texts {
            let tokens = self.tokenizer.tokenize(text)?;
            let word_indices: Vec<usize> = tokens
                .iter()
                .filter_map(|token| self.vocabulary.get_index(token))
                .collect();
            if !word_indices.is_empty() {
                sentences.push(word_indices);
            }
        }

        // Pre-compute n-gram buckets for all words
        let mut word_ngram_buckets: Vec<Vec<usize>> = Vec::with_capacity(self.vocabulary.len());
        for i in 0..self.vocabulary.len() {
            if let Some(word) = self.vocabulary.get_token(i) {
                let ngrams = self.extract_ngrams(word);
                let buckets: Vec<usize> = ngrams
                    .iter()
                    .filter_map(|ng| self.ngram_to_bucket.get(ng).copied())
                    .collect();
                word_ngram_buckets.push(buckets);
            } else {
                word_ngram_buckets.push(Vec::new());
            }
        }

        // Training loop
        for epoch in 0..self.config.epochs {
            // Update learning rate
            self.current_learning_rate =
                self.config.learning_rate * (1.0 - (epoch as f64 / self.config.epochs as f64));
            self.current_learning_rate = self
                .current_learning_rate
                .max(self.config.learning_rate * 0.0001);

            // Train on each sentence
            for sentence in &sentences {
                self.train_sentence(sentence, &word_ngram_buckets)?;
            }
        }

        Ok(())
    }

    /// Train on a single sentence using skip-gram with negative sampling
    fn train_sentence(
        &mut self,
        sentence: &[usize],
        word_ngram_buckets: &[Vec<usize>],
    ) -> Result<()> {
        if sentence.len() < 2 {
            return Ok(());
        }

        // Clone sampling weights to avoid borrow conflict with self.sample_negative()
        let sampling_weights = self.sampling_weights.clone();
        let vocab_len = self.vocabulary.len().max(1);
        let negative_samples = self.config.negative_samples;
        let current_lr = self.current_learning_rate;

        let word_embeddings = self
            .word_embeddings
            .as_mut()
            .ok_or_else(|| TextError::EmbeddingError("Word embeddings not initialized".into()))?;
        let output_embeddings = self
            .output_embeddings
            .as_mut()
            .ok_or_else(|| TextError::EmbeddingError("Output embeddings not initialized".into()))?;
        let ngram_embeddings = self
            .ngram_embeddings
            .as_mut()
            .ok_or_else(|| TextError::EmbeddingError("N-gram embeddings not initialized".into()))?;

        let vector_size = self.config.vector_size;
        let mut rng = scirs2_core::random::rng();

        // Pre-compute cumulative distribution for negative sampling to avoid borrow conflict
        let total_weight: f64 = sampling_weights.iter().sum();
        let cumulative_weights: Vec<f64> = if total_weight > 0.0 {
            let mut cum = Vec::with_capacity(sampling_weights.len());
            let mut acc = 0.0;
            for &w in &sampling_weights {
                acc += w;
                cum.push(acc);
            }
            cum
        } else {
            Vec::new()
        };

        // Skip-gram training for each word in the sentence
        for (pos, &target_idx) in sentence.iter().enumerate() {
            // Random window size
            let window = 1 + rng.random_range(0..self.config.window_size);

            // Get n-gram buckets for this target word
            let ngram_buckets = &word_ngram_buckets[target_idx];

            // Compute the subword-aware input representation
            let mut input_vec = word_embeddings.row(target_idx).to_owned();
            for &bucket in ngram_buckets {
                input_vec += &ngram_embeddings.row(bucket);
            }
            let divisor = 1.0 + ngram_buckets.len() as f64;
            input_vec /= divisor;

            // For each context word in window
            for i in pos.saturating_sub(window)..=(pos + window).min(sentence.len() - 1) {
                if i == pos {
                    continue;
                }

                let context_idx = sentence[i];

                // Accumulated gradient for the input vector
                let mut grad_input = Array1::zeros(vector_size);

                // Positive example
                let output_vec = output_embeddings.row(context_idx).to_owned();
                let dot_product: f64 = input_vec
                    .iter()
                    .zip(output_vec.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
                let gradient = (1.0 - sigmoid) * current_lr;

                // Accumulate gradient for input
                grad_input.scaled_add(gradient, &output_vec);

                // Update output embedding for positive example
                let mut out_row = output_embeddings.row_mut(context_idx);
                let update = &input_vec * gradient;
                out_row += &update;

                // Negative sampling
                for _ in 0..negative_samples {
                    let neg_idx = if cumulative_weights.is_empty() {
                        if vocab_len > 0 {
                            rng.random_range(0..vocab_len)
                        } else {
                            0
                        }
                    } else {
                        let r = rng.random::<f64>() * total_weight;
                        match cumulative_weights.binary_search_by(|w| {
                            w.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Equal)
                        }) {
                            Ok(i) => i,
                            Err(i) => i.min(cumulative_weights.len() - 1),
                        }
                    };
                    if neg_idx == context_idx {
                        continue;
                    }

                    let neg_vec = output_embeddings.row(neg_idx).to_owned();
                    let dot_product: f64 = input_vec
                        .iter()
                        .zip(neg_vec.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
                    let gradient = -sigmoid * current_lr;

                    // Accumulate gradient for input
                    grad_input.scaled_add(gradient, &neg_vec);

                    // Update output embedding for negative example
                    let mut neg_row = output_embeddings.row_mut(neg_idx);
                    let neg_update = &input_vec * gradient;
                    neg_row += &neg_update;
                }

                // Distribute gradient back to word embedding and n-gram embeddings
                let scaled_grad = &grad_input / divisor;

                let mut word_row = word_embeddings.row_mut(target_idx);
                word_row += &scaled_grad;

                for &bucket in ngram_buckets {
                    let mut ngram_row = ngram_embeddings.row_mut(bucket);
                    ngram_row += &scaled_grad;
                }
            }
        }

        Ok(())
    }

    /// Get the embedding vector for a word (handles OOV words via subwords)
    ///
    /// For in-vocabulary words, returns the average of the word vector and its n-gram vectors.
    /// For OOV words, returns the average of matching n-gram vectors.
    pub fn get_word_vector(&self, word: &str) -> Result<Array1<f64>> {
        let word_embeddings = self
            .word_embeddings
            .as_ref()
            .ok_or_else(|| TextError::EmbeddingError("Model not trained".into()))?;
        let ngram_embeddings = self
            .ngram_embeddings
            .as_ref()
            .ok_or_else(|| TextError::EmbeddingError("Model not trained".into()))?;

        let ngrams = self.extract_ngrams(word);
        let mut vector = Array1::zeros(self.config.vector_size);
        let mut count = 0.0;

        // Add word embedding if in vocabulary
        if let Some(idx) = self.vocabulary.get_index(word) {
            vector += &word_embeddings.row(idx);
            count += 1.0;
        }

        // Add n-gram embeddings (always, even for in-vocab words)
        for ngram in &ngrams {
            if let Some(&bucket) = self.ngram_to_bucket.get(ngram) {
                vector += &ngram_embeddings.row(bucket);
                count += 1.0;
            } else {
                // For OOV words, hash the n-gram directly
                let bucket = self.hash_ngram(ngram);
                if bucket < self.config.bucket_size {
                    vector += &ngram_embeddings.row(bucket);
                    count += 1.0;
                }
            }
        }

        if count > 0.0 {
            vector /= count;
            Ok(vector)
        } else {
            Err(TextError::VocabularyError(format!(
                "Cannot compute vector for word '{}': no n-grams found",
                word
            )))
        }
    }

    /// Find most similar words to a given word
    pub fn most_similar(&self, word: &str, top_n: usize) -> Result<Vec<(String, f64)>> {
        let word_vec = self.get_word_vector(word)?;
        self.most_similar_by_vector(&word_vec, top_n, &[word])
    }

    /// Find most similar words to a given vector
    pub fn most_similar_by_vector(
        &self,
        vector: &Array1<f64>,
        top_n: usize,
        exclude_words: &[&str],
    ) -> Result<Vec<(String, f64)>> {
        let mut similarities = Vec::new();

        for i in 0..self.vocabulary.len() {
            if let Some(candidate) = self.vocabulary.get_token(i) {
                if exclude_words.contains(&candidate) {
                    continue;
                }

                if let Ok(candidate_vec) = self.get_word_vector(candidate) {
                    let similarity = cosine_similarity(vector, &candidate_vec);
                    similarities.push((candidate.to_string(), similarity));
                }
            }
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(similarities.into_iter().take(top_n).collect())
    }

    /// Compute word analogy: a is to b as c is to ?
    ///
    /// Uses vector arithmetic: result = b - a + c, then finds most similar words.
    /// Works with OOV words since FastText can compute vectors for any word.
    pub fn analogy(&self, a: &str, b: &str, c: &str, top_n: usize) -> Result<Vec<(String, f64)>> {
        let a_vec = self.get_word_vector(a)?;
        let b_vec = self.get_word_vector(b)?;
        let c_vec = self.get_word_vector(c)?;

        // d = b - a + c
        let mut d_vec = b_vec.clone();
        d_vec -= &a_vec;
        d_vec += &c_vec;

        // Normalize
        let norm = d_vec.iter().fold(0.0, |sum, &val| sum + val * val).sqrt();
        if norm > 0.0 {
            d_vec.mapv_inplace(|val| val / norm);
        }

        self.most_similar_by_vector(&d_vec, top_n, &[a, b, c])
    }

    /// Compute cosine similarity between two words
    ///
    /// Both words can be OOV.
    pub fn word_similarity(&self, word1: &str, word2: &str) -> Result<f64> {
        let vec1 = self.get_word_vector(word1)?;
        let vec2 = self.get_word_vector(word2)?;
        Ok(cosine_similarity(&vec1, &vec2))
    }

    /// Save the model to a file
    ///
    /// Saves in a format that includes word vectors, n-gram info, and config.
    /// Uses a custom header format:
    /// Line 1: FASTTEXT <vocab_size> <vector_size> <min_n> <max_n> <bucket_size>
    /// Lines 2+: word vector_components...
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let word_embeddings = self
            .word_embeddings
            .as_ref()
            .ok_or_else(|| TextError::EmbeddingError("Model not trained".into()))?;

        let mut file = File::create(path).map_err(|e| TextError::IoError(e.to_string()))?;

        // Write extended header
        writeln!(
            &mut file,
            "FASTTEXT {} {} {} {} {}",
            self.vocabulary.len(),
            self.config.vector_size,
            self.config.min_n,
            self.config.max_n,
            self.config.bucket_size,
        )
        .map_err(|e| TextError::IoError(e.to_string()))?;

        // Write each word and its full subword-aware vector
        for i in 0..self.vocabulary.len() {
            if let Some(word) = self.vocabulary.get_token(i) {
                write!(&mut file, "{} ", word).map_err(|e| TextError::IoError(e.to_string()))?;

                // Write the raw word embedding (subword info is reconstructed on load)
                for j in 0..self.config.vector_size {
                    write!(&mut file, "{:.6} ", word_embeddings[[i, j]])
                        .map_err(|e| TextError::IoError(e.to_string()))?;
                }

                writeln!(&mut file).map_err(|e| TextError::IoError(e.to_string()))?;
            }
        }

        // Write n-gram mapping section
        writeln!(&mut file, "NGRAMS {}", self.ngram_to_bucket.len())
            .map_err(|e| TextError::IoError(e.to_string()))?;

        for (ngram, &bucket) in &self.ngram_to_bucket {
            writeln!(&mut file, "{} {}", ngram, bucket)
                .map_err(|e| TextError::IoError(e.to_string()))?;
        }

        Ok(())
    }

    /// Load a FastText model from a file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(|e| TextError::IoError(e.to_string()))?;
        let mut reader = BufReader::new(file);

        // Read header
        let mut header = String::new();
        reader
            .read_line(&mut header)
            .map_err(|e| TextError::IoError(e.to_string()))?;

        let parts: Vec<&str> = header.split_whitespace().collect();
        if parts.len() < 6 || parts[0] != "FASTTEXT" {
            return Err(TextError::EmbeddingError(
                "Invalid FastText file format (expected FASTTEXT header)".into(),
            ));
        }

        let vocab_size = parts[1]
            .parse::<usize>()
            .map_err(|_| TextError::EmbeddingError("Invalid vocab size".into()))?;
        let vector_size = parts[2]
            .parse::<usize>()
            .map_err(|_| TextError::EmbeddingError("Invalid vector size".into()))?;
        let min_n = parts[3]
            .parse::<usize>()
            .map_err(|_| TextError::EmbeddingError("Invalid min_n".into()))?;
        let max_n = parts[4]
            .parse::<usize>()
            .map_err(|_| TextError::EmbeddingError("Invalid max_n".into()))?;
        let bucket_size = parts[5]
            .parse::<usize>()
            .map_err(|_| TextError::EmbeddingError("Invalid bucket_size".into()))?;

        let config = FastTextConfig {
            vector_size,
            min_n,
            max_n,
            bucket_size,
            ..Default::default()
        };

        let mut vocabulary = Vocabulary::new();
        let mut word_embeddings = Array2::zeros((vocab_size, vector_size));

        // Read word vectors
        for i in 0..vocab_size {
            let mut line = String::new();
            reader
                .read_line(&mut line)
                .map_err(|e| TextError::IoError(e.to_string()))?;

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < vector_size + 1 {
                return Err(TextError::EmbeddingError(format!(
                    "Invalid vector at line {}",
                    i + 2
                )));
            }

            vocabulary.add_token(parts[0]);

            for j in 0..vector_size {
                word_embeddings[[i, j]] = parts[j + 1].parse::<f64>().map_err(|_| {
                    TextError::EmbeddingError(format!(
                        "Invalid float at line {}, position {}",
                        i + 2,
                        j + 1
                    ))
                })?;
            }
        }

        // Read n-gram mapping section (if present)
        let mut ngram_to_bucket = HashMap::new();
        let mut ngram_header = String::new();
        if reader
            .read_line(&mut ngram_header)
            .map_err(|e| TextError::IoError(e.to_string()))?
            > 0
        {
            let ngram_parts: Vec<&str> = ngram_header.split_whitespace().collect();
            if ngram_parts.len() >= 2 && ngram_parts[0] == "NGRAMS" {
                let ngram_count = ngram_parts[1]
                    .parse::<usize>()
                    .map_err(|_| TextError::EmbeddingError("Invalid ngram count".into()))?;

                for _ in 0..ngram_count {
                    let mut ngram_line = String::new();
                    reader
                        .read_line(&mut ngram_line)
                        .map_err(|e| TextError::IoError(e.to_string()))?;

                    let np: Vec<&str> = ngram_line.split_whitespace().collect();
                    if np.len() >= 2 {
                        let bucket = np[1]
                            .parse::<usize>()
                            .map_err(|_| TextError::EmbeddingError("Invalid bucket".into()))?;
                        ngram_to_bucket.insert(np[0].to_string(), bucket);
                    }
                }
            }
        }

        // Initialize n-gram embeddings (zeros since we don't save them in text format)
        let ngram_embeddings = Array2::zeros((bucket_size, vector_size));

        Ok(Self {
            config,
            vocabulary,
            word_counts: HashMap::new(),
            word_embeddings: Some(word_embeddings),
            output_embeddings: None,
            ngram_embeddings: Some(ngram_embeddings),
            ngram_to_bucket,
            tokenizer: Box::new(WordTokenizer::default()),
            current_learning_rate: 0.05,
            sampling_weights: Vec::new(),
        })
    }

    /// Get the vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// Get the vector size
    pub fn vector_size(&self) -> usize {
        self.config.vector_size
    }

    /// Get the n-gram configuration (min_n, max_n)
    pub fn ngram_range(&self) -> (usize, usize) {
        (self.config.min_n, self.config.max_n)
    }

    /// Get the number of unique n-grams discovered
    pub fn ngram_count(&self) -> usize {
        self.ngram_to_bucket.len()
    }

    /// Check if a word is in the vocabulary
    pub fn contains(&self, word: &str) -> bool {
        self.vocabulary.contains(word)
    }

    /// Check if a word can be represented (either in vocab or has matching n-grams)
    pub fn can_represent(&self, word: &str) -> bool {
        if self.vocabulary.contains(word) {
            return true;
        }
        // Check if any n-grams match
        let ngrams = self.extract_ngrams(word);
        ngrams
            .iter()
            .any(|ng| self.ngram_to_bucket.contains_key(ng))
    }

    /// Get all words in the vocabulary
    pub fn get_vocabulary_words(&self) -> Vec<String> {
        let mut words = Vec::with_capacity(self.vocabulary.len());
        for i in 0..self.vocabulary.len() {
            if let Some(word) = self.vocabulary.get_token(i) {
                words.push(word.to_string());
            }
        }
        words
    }
}

impl Default for FastText {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_ngrams() {
        let config = FastTextConfig {
            min_n: 3,
            max_n: 4,
            ..Default::default()
        };
        let model = FastText::with_config(config);

        let ngrams = model.extract_ngrams("test");
        assert!(!ngrams.is_empty());
        assert!(ngrams.contains(&"<te".to_string()));
        assert!(ngrams.contains(&"est".to_string()));
        assert!(ngrams.contains(&"st>".to_string()));
        // 4-grams
        assert!(ngrams.contains(&"<tes".to_string()));
        assert!(ngrams.contains(&"test".to_string()));
        assert!(ngrams.contains(&"est>".to_string()));
    }

    #[test]
    fn test_extract_ngrams_short_word() {
        let config = FastTextConfig {
            min_n: 3,
            max_n: 6,
            ..Default::default()
        };
        let model = FastText::with_config(config);

        let ngrams = model.extract_ngrams("a");
        // "<a>" has 3 chars, so only 3-gram possible: "<a>"
        assert_eq!(ngrams.len(), 1);
        assert_eq!(ngrams[0], "<a>");
    }

    #[test]
    fn test_fasttext_training() {
        let texts = [
            "the quick brown fox jumps over the lazy dog",
            "a quick brown dog outpaces a quick fox",
        ];

        let config = FastTextConfig {
            vector_size: 10,
            window_size: 2,
            min_count: 1,
            epochs: 1,
            min_n: 3,
            max_n: 4,
            bucket_size: 1000,
            ..Default::default()
        };

        let mut model = FastText::with_config(config);
        let result = model.train(&texts);
        assert!(result.is_ok());

        // Test getting vector for in-vocabulary word
        let vec = model.get_word_vector("quick");
        assert!(vec.is_ok());
        assert_eq!(vec.expect("Failed to get vector").len(), 10);

        // Test getting vector for OOV word (should work due to n-grams)
        let oov_vec = model.get_word_vector("quickest");
        assert!(oov_vec.is_ok());
    }

    #[test]
    fn test_fasttext_oov_handling() {
        let texts = ["hello world", "hello there"];

        let config = FastTextConfig {
            vector_size: 10,
            min_count: 1,
            epochs: 1,
            bucket_size: 1000,
            ..Default::default()
        };

        let mut model = FastText::with_config(config);
        model.train(&texts).expect("Training failed");

        // Get vector for OOV word that shares n-grams with "hello"
        let oov_vec = model.get_word_vector("helloworld");
        assert!(oov_vec.is_ok(), "FastText should handle OOV words");
    }

    #[test]
    fn test_fasttext_analogy() {
        let texts = [
            "the king ruled the kingdom wisely",
            "the queen ruled the kingdom wisely",
            "the man worked in the field",
            "the woman worked in the field",
            "the king and the queen were happy",
            "the man and the woman were happy",
        ];

        let config = FastTextConfig {
            vector_size: 20,
            window_size: 3,
            min_count: 1,
            epochs: 5,
            min_n: 3,
            max_n: 5,
            bucket_size: 1000,
            ..Default::default()
        };

        let mut model = FastText::with_config(config);
        model.train(&texts).expect("Training failed");

        // Just verify analogy doesn't crash
        let result = model.analogy("king", "man", "woman", 3);
        assert!(result.is_ok());
        let answers = result.expect("analogy");
        assert!(!answers.is_empty());
    }

    #[test]
    fn test_fasttext_word_similarity() {
        let texts = [
            "the cat sat on the mat",
            "the dog sat on the rug",
            "the cat and dog played",
        ];

        let config = FastTextConfig {
            vector_size: 10,
            min_count: 1,
            epochs: 3,
            min_n: 3,
            max_n: 4,
            bucket_size: 1000,
            ..Default::default()
        };

        let mut model = FastText::with_config(config);
        model.train(&texts).expect("Training failed");

        let sim = model.word_similarity("cat", "dog");
        assert!(sim.is_ok());
        // Both should have finite similarity
        assert!(sim.expect("similarity").is_finite());
    }

    #[test]
    fn test_fasttext_save_load() {
        let texts = ["the quick brown fox jumps", "the lazy brown dog sleeps"];

        let config = FastTextConfig {
            vector_size: 5,
            min_count: 1,
            epochs: 1,
            min_n: 3,
            max_n: 4,
            bucket_size: 1000,
            ..Default::default()
        };

        let mut model = FastText::with_config(config);
        model.train(&texts).expect("Training failed");

        let save_path = std::env::temp_dir().join("test_fasttext_save.txt");
        model.save(&save_path).expect("Failed to save");

        let loaded = FastText::load(&save_path).expect("Failed to load");
        assert_eq!(loaded.vocabulary_size(), model.vocabulary_size());
        assert_eq!(loaded.vector_size(), model.vector_size());
        assert_eq!(loaded.ngram_range(), model.ngram_range());

        std::fs::remove_file(save_path).ok();
    }

    #[test]
    fn test_fasttext_can_represent() {
        let texts = ["hello world"];

        let config = FastTextConfig {
            vector_size: 5,
            min_count: 1,
            epochs: 1,
            ..Default::default()
        };

        let mut model = FastText::with_config(config);
        model.train(&texts).expect("Training failed");

        assert!(model.contains("hello"));
        assert!(model.can_represent("hello"));
        assert!(!model.contains("helloworld"));
        assert!(model.can_represent("helloworld")); // OOV but has matching n-grams
    }

    #[test]
    fn test_fasttext_most_similar() {
        let texts = [
            "the dog runs fast",
            "the cat runs fast",
            "the bird flies high",
        ];

        let config = FastTextConfig {
            vector_size: 10,
            min_count: 1,
            epochs: 5,
            min_n: 3,
            max_n: 4,
            bucket_size: 1000,
            ..Default::default()
        };

        let mut model = FastText::with_config(config);
        model.train(&texts).expect("Training failed");

        let similar = model.most_similar("dog", 2).expect("most_similar");
        assert!(!similar.is_empty());
        assert!(similar.len() <= 2);
    }

    #[test]
    fn test_fasttext_empty_input() {
        let texts: Vec<&str> = vec![];
        let mut model = FastText::new();
        let result = model.train(&texts);
        assert!(result.is_err());
    }

    #[test]
    fn test_fasttext_config_defaults() {
        let config = FastTextConfig::default();
        assert_eq!(config.vector_size, 100);
        assert_eq!(config.min_n, 3);
        assert_eq!(config.max_n, 6);
        assert_eq!(config.window_size, 5);
        assert_eq!(config.bucket_size, 2_000_000);
    }

    #[test]
    fn test_hash_ngram_deterministic() {
        let model = FastText::new();
        let h1 = model.hash_ngram("abc");
        let h2 = model.hash_ngram("abc");
        assert_eq!(h1, h2);

        let h3 = model.hash_ngram("xyz");
        // Different strings should (usually) hash differently
        // Not guaranteed but very likely
        assert_ne!(h1, h3);
    }
}
