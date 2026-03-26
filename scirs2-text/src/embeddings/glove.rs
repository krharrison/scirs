//! GloVe (Global Vectors for Word Representation) embeddings
//!
//! This module provides functionality to both load pre-trained GloVe embeddings
//! and train GloVe embeddings from scratch using co-occurrence statistics.
//!
//! ## Overview
//!
//! GloVe is an unsupervised learning algorithm for obtaining vector representations
//! for words. Training is performed on aggregated global word-word co-occurrence
//! statistics from a corpus, and the resulting representations capture interesting
//! linear substructures of the word vector space.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use scirs2_text::embeddings::glove::GloVe;
//!
//! // Load pre-trained GloVe embeddings
//! let glove = GloVe::load("path/to/glove.6B.100d.txt")
//!     .expect("Failed to load GloVe embeddings");
//!
//! // Get word vector
//! if let Ok(vector) = glove.get_word_vector("king") {
//!     println!("Vector for 'king': {:?}", vector);
//! }
//!
//! // Find similar words
//! let similar = glove.most_similar("king", 5).expect("Failed to find similar words");
//! for (word, similarity) in similar {
//!     println!("{}: {:.4}", word, similarity);
//! }
//!
//! // Analogy: king - man + woman = ?
//! let analogy = glove.analogy("king", "man", "woman", 5)
//!     .expect("Failed to compute analogy");
//! println!("Analogy results: {:?}", analogy);
//! ```
//!
//! ## Training from Scratch
//!
//! ```rust
//! use scirs2_text::embeddings::glove::{GloVeTrainer, GloVeTrainerConfig};
//!
//! let documents = vec![
//!     "the quick brown fox jumps over the lazy dog",
//!     "the dog chased the fox around the yard",
//!     "a quick brown dog outpaces a lazy fox",
//! ];
//!
//! let config = GloVeTrainerConfig {
//!     vector_size: 50,
//!     window_size: 5,
//!     min_count: 1,
//!     epochs: 25,
//!     learning_rate: 0.05,
//!     x_max: 100.0,
//!     alpha: 0.75,
//! };
//!
//! let mut trainer = GloVeTrainer::with_config(config);
//! let glove = trainer.train(&documents).expect("Training failed");
//!
//! // Use the trained model
//! let vec = glove.get_word_vector("fox");
//! assert!(vec.is_ok());
//! ```

use crate::error::{Result, TextError};
use crate::tokenize::{Tokenizer, WordTokenizer};
use crate::vocabulary::Vocabulary;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// GloVe embeddings model
///
/// Provides functionality to load, train, and use GloVe word embeddings.
/// Supports both loading pre-trained vectors and training from scratch
/// using co-occurrence statistics.
#[derive(Debug, Clone)]
pub struct GloVe {
    /// Word vocabulary
    vocabulary: Vocabulary,
    /// Word to index mapping for fast lookup
    word_to_idx: HashMap<String, usize>,
    /// Main embedding matrix where each row is a word vector
    embeddings: Array2<f64>,
    /// Context embedding matrix (used during training, summed with main for final vectors)
    context_embeddings: Option<Array2<f64>>,
    /// Dimensionality of the embeddings
    vector_size: usize,
}

impl GloVe {
    /// Create a new empty GloVe model
    pub fn new() -> Self {
        Self {
            vocabulary: Vocabulary::new(),
            word_to_idx: HashMap::new(),
            embeddings: Array2::zeros((0, 0)),
            context_embeddings: None,
            vector_size: 0,
        }
    }

    /// Create a GloVe model from main and context embedding matrices
    ///
    /// The final word vectors are computed as the sum of the main and context vectors,
    /// following the original GloVe paper's recommendation.
    pub fn from_trained(
        vocabulary: Vocabulary,
        word_to_idx: HashMap<String, usize>,
        main_embeddings: Array2<f64>,
        context_embeddings: Array2<f64>,
        vector_size: usize,
    ) -> Result<Self> {
        if main_embeddings.nrows() != context_embeddings.nrows() {
            return Err(TextError::EmbeddingError(
                "Main and context embedding matrices must have the same number of rows".into(),
            ));
        }
        if main_embeddings.ncols() != vector_size || context_embeddings.ncols() != vector_size {
            return Err(TextError::EmbeddingError(
                "Embedding matrices must have vector_size columns".into(),
            ));
        }

        // Sum main + context for the final embeddings
        let combined = &main_embeddings + &context_embeddings;

        Ok(Self {
            vocabulary,
            word_to_idx,
            embeddings: combined,
            context_embeddings: Some(context_embeddings),
            vector_size,
        })
    }

    /// Load GloVe embeddings from a file
    ///
    /// The file should be in the standard GloVe format:
    /// ```text
    /// word1 0.123 0.456 0.789 ...
    /// word2 0.234 0.567 0.890 ...
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(|e| TextError::IoError(e.to_string()))?;
        let reader = BufReader::new(file);

        let mut words = Vec::new();
        let mut vectors = Vec::new();
        let mut vector_size = 0;

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result.map_err(|e| TextError::IoError(e.to_string()))?;
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.is_empty() {
                continue;
            }

            // First part is the word, rest are vector components
            let word = parts[0].to_string();
            let vector_parts = &parts[1..];

            // Set vector size from first line
            if line_num == 0 {
                vector_size = vector_parts.len();
                if vector_size == 0 {
                    return Err(TextError::EmbeddingError(
                        "Invalid GloVe file: no vector components found".into(),
                    ));
                }
            }

            // Verify vector size consistency
            if vector_parts.len() != vector_size {
                return Err(TextError::EmbeddingError(format!(
                    "Inconsistent vector size at line {}: expected {}, got {}",
                    line_num + 1,
                    vector_size,
                    vector_parts.len()
                )));
            }

            // Parse vector components
            let vector: Result<Vec<f64>> = vector_parts
                .iter()
                .map(|&s| {
                    s.parse::<f64>().map_err(|_| {
                        TextError::EmbeddingError(format!(
                            "Failed to parse float at line {}: '{}'",
                            line_num + 1,
                            s
                        ))
                    })
                })
                .collect();

            words.push(word);
            vectors.push(vector?);
        }

        if words.is_empty() {
            return Err(TextError::EmbeddingError(
                "No embeddings loaded from file".into(),
            ));
        }

        // Build vocabulary
        let mut vocabulary = Vocabulary::new();
        let mut word_to_idx = HashMap::new();

        for (idx, word) in words.iter().enumerate() {
            vocabulary.add_token(word);
            word_to_idx.insert(word.clone(), idx);
        }

        // Create embedding matrix
        let vocab_size = words.len();
        let mut embeddings = Array2::zeros((vocab_size, vector_size));

        for (idx, vector) in vectors.iter().enumerate() {
            for (j, &val) in vector.iter().enumerate() {
                embeddings[[idx, j]] = val;
            }
        }

        Ok(Self {
            vocabulary,
            word_to_idx,
            embeddings,
            context_embeddings: None,
            vector_size,
        })
    }

    /// Save GloVe embeddings to a file
    ///
    /// Saves in the standard GloVe format compatible with load()
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut file = File::create(path).map_err(|e| TextError::IoError(e.to_string()))?;

        // Write in index order for deterministic output
        for idx in 0..self.vocabulary.len() {
            if let Some(word) = self.vocabulary.get_token(idx) {
                write!(&mut file, "{}", word).map_err(|e| TextError::IoError(e.to_string()))?;

                for j in 0..self.vector_size {
                    write!(&mut file, " {:.6}", self.embeddings[[idx, j]])
                        .map_err(|e| TextError::IoError(e.to_string()))?;
                }

                writeln!(&mut file).map_err(|e| TextError::IoError(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Get the vector for a word
    pub fn get_word_vector(&self, word: &str) -> Result<Array1<f64>> {
        match self.word_to_idx.get(word) {
            Some(&idx) => Ok(self.embeddings.row(idx).to_owned()),
            None => Err(TextError::VocabularyError(format!(
                "Word '{}' not in vocabulary",
                word
            ))),
        }
    }

    /// Find the most similar words to a given word
    pub fn most_similar(&self, word: &str, top_n: usize) -> Result<Vec<(String, f64)>> {
        let word_vec = self.get_word_vector(word)?;
        self.most_similar_by_vector(&word_vec, top_n, &[word])
    }

    /// Find the most similar words to a given vector
    pub fn most_similar_by_vector(
        &self,
        vector: &Array1<f64>,
        top_n: usize,
        exclude_words: &[&str],
    ) -> Result<Vec<(String, f64)>> {
        // Create set of excluded indices
        let exclude_indices: Vec<usize> = exclude_words
            .iter()
            .filter_map(|&word| self.word_to_idx.get(word).copied())
            .collect();

        // Calculate cosine similarity for all words
        let mut similarities = Vec::new();

        for (word, &idx) in &self.word_to_idx {
            if exclude_indices.contains(&idx) {
                continue;
            }

            let word_vec = self.embeddings.row(idx).to_owned();
            let similarity = cosine_similarity(vector, &word_vec);
            similarities.push((word.clone(), similarity));
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N
        Ok(similarities.into_iter().take(top_n).collect())
    }

    /// Compute word analogy: a is to b as c is to ?
    ///
    /// Uses vector arithmetic: result = b - a + c
    pub fn analogy(&self, a: &str, b: &str, c: &str, top_n: usize) -> Result<Vec<(String, f64)>> {
        // Get vectors
        let a_vec = self.get_word_vector(a)?;
        let b_vec = self.get_word_vector(b)?;
        let c_vec = self.get_word_vector(c)?;

        // Compute: b - a + c
        let mut result_vec = b_vec.clone();
        result_vec -= &a_vec;
        result_vec += &c_vec;

        // Normalize the result vector
        let norm = result_vec.iter().fold(0.0, |sum, &x| sum + x * x).sqrt();
        if norm > 0.0 {
            result_vec.mapv_inplace(|x| x / norm);
        }

        // Find most similar words to the result vector
        self.most_similar_by_vector(&result_vec, top_n, &[a, b, c])
    }

    /// Get the vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.word_to_idx.len()
    }

    /// Get the vector dimensionality
    pub fn vector_size(&self) -> usize {
        self.vector_size
    }

    /// Check if a word is in the vocabulary
    pub fn contains(&self, word: &str) -> bool {
        self.word_to_idx.contains_key(word)
    }

    /// Get all words in the vocabulary
    pub fn get_words(&self) -> Vec<String> {
        self.word_to_idx.keys().cloned().collect()
    }

    /// Get the embeddings matrix (for advanced use cases)
    pub fn get_embeddings(&self) -> &Array2<f64> {
        &self.embeddings
    }

    /// Check if context embeddings are available (only from training)
    pub fn has_context_embeddings(&self) -> bool {
        self.context_embeddings.is_some()
    }
}

impl Default for GloVe {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Co-occurrence matrix ────────────────────────────────────────────────────

/// A sparse co-occurrence matrix for GloVe training
#[derive(Debug, Clone)]
pub struct CooccurrenceMatrix {
    /// Sparse entries: (row, col) -> count (with distance weighting)
    entries: HashMap<(usize, usize), f64>,
    /// Vocabulary size
    vocab_size: usize,
}

impl CooccurrenceMatrix {
    /// Create a new empty co-occurrence matrix
    pub fn new(vocab_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            vocab_size,
        }
    }

    /// Add a co-occurrence with distance-weighted count
    pub fn add(&mut self, i: usize, j: usize, weight: f64) {
        *self.entries.entry((i, j)).or_insert(0.0) += weight;
    }

    /// Get the co-occurrence count for a word pair
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.entries.get(&(i, j)).copied().unwrap_or(0.0)
    }

    /// Get the number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Iterate over all non-zero entries
    pub fn iter(&self) -> impl Iterator<Item = (&(usize, usize), &f64)> {
        self.entries.iter()
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

// ─── GloVe Trainer Configuration ─────────────────────────────────────────────

/// Configuration for GloVe training
#[derive(Debug, Clone)]
pub struct GloVeTrainerConfig {
    /// Dimensionality of word vectors
    pub vector_size: usize,
    /// Context window size (symmetric)
    pub window_size: usize,
    /// Minimum word frequency to include in vocabulary
    pub min_count: usize,
    /// Number of training epochs
    pub epochs: usize,
    /// Initial learning rate (AdaGrad is used, so this is the base)
    pub learning_rate: f64,
    /// Maximum co-occurrence count for weighting function (x_max in paper)
    pub x_max: f64,
    /// Exponent for the weighting function (alpha in paper, typically 0.75)
    pub alpha: f64,
}

impl Default for GloVeTrainerConfig {
    fn default() -> Self {
        Self {
            vector_size: 50,
            window_size: 5,
            min_count: 5,
            epochs: 25,
            learning_rate: 0.05,
            x_max: 100.0,
            alpha: 0.75,
        }
    }
}

// ─── GloVe Trainer ───────────────────────────────────────────────────────────

/// GloVe trainer that builds co-occurrence matrices and trains embeddings
///
/// Implements the weighted least squares objective from
/// Pennington, Socher, Manning (2014) "GloVe: Global Vectors for Word Representation"
pub struct GloVeTrainer {
    /// Training configuration
    config: GloVeTrainerConfig,
    /// Tokenizer for text processing
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
}

impl std::fmt::Debug for GloVeTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GloVeTrainer")
            .field("config", &self.config)
            .finish()
    }
}

impl GloVeTrainer {
    /// Create a new GloVe trainer with default configuration
    pub fn new() -> Self {
        Self {
            config: GloVeTrainerConfig::default(),
            tokenizer: Box::new(WordTokenizer::default()),
        }
    }

    /// Create a new GloVe trainer with custom configuration
    pub fn with_config(config: GloVeTrainerConfig) -> Self {
        Self {
            config,
            tokenizer: Box::new(WordTokenizer::default()),
        }
    }

    /// Set a custom tokenizer
    pub fn with_tokenizer(mut self, tokenizer: Box<dyn Tokenizer + Send + Sync>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Build the co-occurrence matrix from a corpus
    ///
    /// Uses a symmetric window with distance-based weighting:
    /// weight = 1 / distance
    pub fn build_cooccurrence(
        &self,
        texts: &[&str],
        vocabulary: &Vocabulary,
    ) -> Result<CooccurrenceMatrix> {
        let vocab_size = vocabulary.len();
        let mut matrix = CooccurrenceMatrix::new(vocab_size);

        for &text in texts {
            let tokens = self.tokenizer.tokenize(text)?;

            // Convert tokens to indices (skip unknown words)
            let indices: Vec<usize> = tokens
                .iter()
                .filter_map(|t| vocabulary.get_index(t))
                .collect();

            // For each word, count co-occurrences within the window
            for (pos, &center_idx) in indices.iter().enumerate() {
                let window = self.config.window_size;
                let start = pos.saturating_sub(window);
                let end = (pos + window).min(indices.len() - 1);

                for context_pos in start..=end {
                    if context_pos == pos {
                        continue;
                    }
                    let context_idx = indices[context_pos];
                    // Distance-based weighting: closer words get higher weight
                    let distance = context_pos.abs_diff(pos);
                    let weight = 1.0 / distance as f64;

                    matrix.add(center_idx, context_idx, weight);
                }
            }
        }

        Ok(matrix)
    }

    /// Build vocabulary from corpus with frequency filtering
    pub fn build_vocabulary(&self, texts: &[&str]) -> Result<(Vocabulary, HashMap<String, usize>)> {
        let mut word_counts: HashMap<String, usize> = HashMap::new();

        for &text in texts {
            let tokens = self.tokenizer.tokenize(text)?;
            for token in tokens {
                *word_counts.entry(token).or_insert(0) += 1;
            }
        }

        let mut vocabulary = Vocabulary::new();
        let mut word_to_idx = HashMap::new();

        // Sort by frequency descending for deterministic ordering
        let mut sorted_words: Vec<(String, usize)> = word_counts
            .into_iter()
            .filter(|(_, count)| *count >= self.config.min_count)
            .collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        for (word, _) in &sorted_words {
            let idx = vocabulary.len();
            vocabulary.add_token(word);
            word_to_idx.insert(word.clone(), idx);
        }

        if vocabulary.is_empty() {
            return Err(TextError::VocabularyError(
                "No words meet the minimum count threshold".into(),
            ));
        }

        Ok((vocabulary, word_to_idx))
    }

    /// The GloVe weighting function: f(x) = (x / x_max)^alpha if x < x_max, else 1
    fn weighting_function(&self, x: f64) -> f64 {
        if x < self.config.x_max {
            (x / self.config.x_max).powf(self.config.alpha)
        } else {
            1.0
        }
    }

    /// Train GloVe embeddings on a corpus
    ///
    /// Steps:
    /// 1. Build vocabulary from texts
    /// 2. Construct co-occurrence matrix
    /// 3. Train with weighted least squares (AdaGrad)
    /// 4. Return GloVe model with combined main + context vectors
    pub fn train(&mut self, texts: &[&str]) -> Result<GloVe> {
        if texts.is_empty() {
            return Err(TextError::InvalidInput(
                "No texts provided for training".into(),
            ));
        }

        // Step 1: Build vocabulary
        let (vocabulary, word_to_idx) = self.build_vocabulary(texts)?;
        let vocab_size = vocabulary.len();

        // Step 2: Build co-occurrence matrix
        let cooccurrence = self.build_cooccurrence(texts, &vocabulary)?;

        // Step 3: Initialize parameters
        let vector_size = self.config.vector_size;
        let mut rng = scirs2_core::random::rng();

        // Main word vectors (W) and context word vectors (W_tilde)
        let mut w_main = Array2::from_shape_fn((vocab_size, vector_size), |_| {
            (rng.random::<f64>() * 2.0 - 1.0) * 0.1
        });
        let mut w_context = Array2::from_shape_fn((vocab_size, vector_size), |_| {
            (rng.random::<f64>() * 2.0 - 1.0) * 0.1
        });

        // Bias terms
        let mut b_main =
            Array1::from_shape_fn(vocab_size, |_| (rng.random::<f64>() * 2.0 - 1.0) * 0.1);
        let mut b_context =
            Array1::from_shape_fn(vocab_size, |_| (rng.random::<f64>() * 2.0 - 1.0) * 0.1);

        // AdaGrad accumulators
        let mut grad_sq_w_main: Array2<f64> = Array2::from_elem((vocab_size, vector_size), 1.0);
        let mut grad_sq_w_context: Array2<f64> = Array2::from_elem((vocab_size, vector_size), 1.0);
        let mut grad_sq_b_main: Array1<f64> = Array1::from_elem(vocab_size, 1.0);
        let mut grad_sq_b_context: Array1<f64> = Array1::from_elem(vocab_size, 1.0);

        let learning_rate = self.config.learning_rate;

        // Step 4: Training loop
        // Collect entries for iteration (shuffling each epoch)
        let mut entries: Vec<((usize, usize), f64)> = cooccurrence
            .iter()
            .map(|(&(i, j), &x)| ((i, j), x))
            .collect();

        for _epoch in 0..self.config.epochs {
            // Shuffle entries each epoch
            entries.as_mut_slice().shuffle(&mut rng);

            for &((i, j), x_ij) in &entries {
                if x_ij <= 0.0 {
                    continue;
                }

                let log_x = x_ij.ln();
                let f_x = self.weighting_function(x_ij);

                // Compute the inner product w_i . w_j + b_i + b_j
                let mut dot = 0.0;
                for k in 0..vector_size {
                    dot += w_main[[i, k]] * w_context[[j, k]];
                }
                dot += b_main[i] + b_context[j];

                // Cost derivative: f(x_ij) * (w_i . w_j + b_i + b_j - log(x_ij))
                let diff = dot - log_x;
                let fdiff = f_x * diff;

                // Clamp to prevent exploding gradients
                let fdiff = fdiff.clamp(-10.0, 10.0);

                // Update main word vector and context word vector
                for k in 0..vector_size {
                    let grad_main = fdiff * w_context[[j, k]];
                    let grad_context = fdiff * w_main[[i, k]];

                    // AdaGrad update
                    w_main[[i, k]] -= learning_rate * grad_main / grad_sq_w_main[[i, k]].sqrt();
                    w_context[[j, k]] -=
                        learning_rate * grad_context / grad_sq_w_context[[j, k]].sqrt();

                    grad_sq_w_main[[i, k]] += grad_main * grad_main;
                    grad_sq_w_context[[j, k]] += grad_context * grad_context;
                }

                // Update biases
                b_main[i] -= learning_rate * fdiff / grad_sq_b_main[i].sqrt();
                b_context[j] -= learning_rate * fdiff / grad_sq_b_context[j].sqrt();

                grad_sq_b_main[i] += fdiff * fdiff;
                grad_sq_b_context[j] += fdiff * fdiff;
            }
        }

        // Step 5: Build final GloVe model (main + context vectors)
        GloVe::from_trained(vocabulary, word_to_idx, w_main, w_context, vector_size)
    }
}

impl Default for GloVeTrainer {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate cosine similarity between two vectors
pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
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
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_glove_load_save() {
        // Create a temporary GloVe file
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        writeln!(temp_file, "king 0.1 0.2 0.3").expect("Failed to write");
        writeln!(temp_file, "queen 0.15 0.25 0.35").expect("Failed to write");
        writeln!(temp_file, "man 0.05 0.1 0.15").expect("Failed to write");
        writeln!(temp_file, "woman 0.08 0.13 0.18").expect("Failed to write");
        temp_file.flush().expect("Failed to flush");

        // Load the GloVe model
        let glove = GloVe::load(temp_file.path()).expect("Failed to load GloVe");

        // Check vocabulary size and vector size
        assert_eq!(glove.vocabulary_size(), 4);
        assert_eq!(glove.vector_size(), 3);

        // Check word lookup
        assert!(glove.contains("king"));
        assert!(glove.contains("queen"));
        assert!(!glove.contains("prince"));

        // Get word vector
        let king_vec = glove.get_word_vector("king").expect("Failed to get vector");
        assert_eq!(king_vec.len(), 3);
        assert!((king_vec[0] - 0.1).abs() < 1e-6);
        assert!((king_vec[1] - 0.2).abs() < 1e-6);
        assert!((king_vec[2] - 0.3).abs() < 1e-6);

        // Test save and reload
        let save_path = std::env::temp_dir().join("test_glove_save_v2.txt");
        glove.save(&save_path).expect("Failed to save");

        let reloaded = GloVe::load(&save_path).expect("Failed to reload");
        assert_eq!(reloaded.vocabulary_size(), glove.vocabulary_size());
        assert_eq!(reloaded.vector_size(), glove.vector_size());

        // Cleanup
        std::fs::remove_file(save_path).ok();
    }

    #[test]
    fn test_glove_similarity() {
        // Create a temporary GloVe file with similar vectors
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        writeln!(temp_file, "king 1.0 0.0 0.0").expect("Failed to write");
        writeln!(temp_file, "queen 0.9 0.1 0.0").expect("Failed to write");
        writeln!(temp_file, "man 0.5 0.5 0.0").expect("Failed to write");
        writeln!(temp_file, "woman 0.4 0.6 0.0").expect("Failed to write");
        writeln!(temp_file, "cat 0.0 0.0 1.0").expect("Failed to write");
        temp_file.flush().expect("Failed to flush");

        let glove = GloVe::load(temp_file.path()).expect("Failed to load GloVe");

        // Find similar words to "king"
        let similar = glove
            .most_similar("king", 2)
            .expect("Failed to find similar");

        // "queen" should be most similar to "king"
        assert_eq!(similar.len(), 2);
        assert_eq!(similar[0].0, "queen");
        assert!(similar[0].1 > 0.9);
    }

    #[test]
    fn test_cosine_similarity_fn() {
        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);

        let a = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let b = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![-1.0, 0.0, 0.0]);
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    // ─── Co-occurrence tests ─────────────────────────────────────────────

    #[test]
    fn test_cooccurrence_matrix_basic() {
        let mut matrix = CooccurrenceMatrix::new(5);
        matrix.add(0, 1, 1.0);
        matrix.add(0, 1, 0.5);
        matrix.add(1, 2, 1.0);

        assert!((matrix.get(0, 1) - 1.5).abs() < 1e-10);
        assert!((matrix.get(1, 2) - 1.0).abs() < 1e-10);
        assert!((matrix.get(2, 3) - 0.0).abs() < 1e-10);
        assert_eq!(matrix.nnz(), 2);
        assert_eq!(matrix.vocab_size(), 5);
    }

    #[test]
    fn test_build_cooccurrence_from_text() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("the");
        vocab.add_token("fox");
        vocab.add_token("dog");

        let texts = vec!["the fox dog"];
        let trainer = GloVeTrainer::with_config(GloVeTrainerConfig {
            window_size: 2,
            ..Default::default()
        });

        let matrix = trainer
            .build_cooccurrence(&texts, &vocab)
            .expect("Failed to build cooccurrence");

        // "the" and "fox" are distance 1 apart
        let the_idx = vocab.get_index("the").expect("the not in vocab");
        let fox_idx = vocab.get_index("fox").expect("fox not in vocab");
        let dog_idx = vocab.get_index("dog").expect("dog not in vocab");

        assert!(matrix.get(the_idx, fox_idx) > 0.0);
        assert!(matrix.get(fox_idx, dog_idx) > 0.0);
        assert!(matrix.get(the_idx, dog_idx) > 0.0); // distance 2, so weight 0.5
    }

    // ─── GloVe trainer config tests ──────────────────────────────────────

    #[test]
    fn test_glove_trainer_config_defaults() {
        let config = GloVeTrainerConfig::default();
        assert_eq!(config.vector_size, 50);
        assert_eq!(config.window_size, 5);
        assert_eq!(config.min_count, 5);
        assert_eq!(config.epochs, 25);
        assert!((config.learning_rate - 0.05).abs() < 1e-10);
        assert!((config.x_max - 100.0).abs() < 1e-10);
        assert!((config.alpha - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_weighting_function() {
        let trainer = GloVeTrainer::with_config(GloVeTrainerConfig {
            x_max: 100.0,
            alpha: 0.75,
            ..Default::default()
        });

        // Below x_max: f(x) = (x / x_max)^alpha
        let w1 = trainer.weighting_function(50.0);
        let expected = (50.0 / 100.0_f64).powf(0.75);
        assert!((w1 - expected).abs() < 1e-10);

        // At x_max: f(x) = 1.0
        let w2 = trainer.weighting_function(100.0);
        assert!((w2 - 1.0).abs() < 1e-10);

        // Above x_max: f(x) = 1.0
        let w3 = trainer.weighting_function(200.0);
        assert!((w3 - 1.0).abs() < 1e-10);

        // Near zero: small weight
        let w4 = trainer.weighting_function(1.0);
        assert!(w4 > 0.0 && w4 < 0.1);
    }

    // ─── GloVe training tests ────────────────────────────────────────────

    #[test]
    fn test_glove_training_basic() {
        let documents = vec![
            "the quick brown fox jumps over the lazy dog",
            "the dog chased the fox around the yard",
            "a quick brown dog outpaces a lazy fox",
            "the lazy dog sleeps all day long",
            "the fox runs quickly through the forest",
        ];

        let config = GloVeTrainerConfig {
            vector_size: 10,
            window_size: 3,
            min_count: 1,
            epochs: 10,
            learning_rate: 0.05,
            x_max: 10.0,
            alpha: 0.75,
        };

        let mut trainer = GloVeTrainer::with_config(config);
        let glove = trainer.train(&documents).expect("Training failed");

        assert!(glove.vocabulary_size() > 0);
        assert_eq!(glove.vector_size(), 10);

        // Should be able to get vectors for words in corpus
        let fox_vec = glove.get_word_vector("fox");
        assert!(fox_vec.is_ok());
        assert_eq!(fox_vec.expect("get vec failed").len(), 10);

        // Should have context embeddings from training
        assert!(glove.has_context_embeddings());
    }

    #[test]
    fn test_glove_training_produces_meaningful_vectors() {
        // Corpus designed so "dog" and "cat" appear in very similar contexts,
        // while "bird" appears in distinct contexts. Repeated sentences
        // reinforce co-occurrence statistics for the small corpus.
        let documents = vec![
            "the dog runs fast",
            "the cat runs fast",
            "the dog sleeps well",
            "the cat sleeps well",
            "the dog eats food",
            "the cat eats food",
            "the dog plays outside",
            "the cat plays outside",
            "the dog is friendly",
            "the cat is friendly",
            // Repeat to reinforce dog-cat co-occurrence patterns
            "the dog runs fast",
            "the cat runs fast",
            "the dog sleeps well",
            "the cat sleeps well",
            "the dog eats food",
            "the cat eats food",
            // bird in distinct contexts
            "the bird flies high",
            "the bird sings loudly",
            "the bird builds nests",
            "the bird migrates south",
        ];

        let config = GloVeTrainerConfig {
            vector_size: 20,
            window_size: 3,
            min_count: 1,
            epochs: 100,
            learning_rate: 0.05,
            x_max: 10.0,
            alpha: 0.75,
        };

        let mut trainer = GloVeTrainer::with_config(config);
        let glove = trainer.train(&documents).expect("Training failed");

        // dog and cat should have non-trivial vectors
        let dog_vec = glove.get_word_vector("dog").expect("dog vector");
        let cat_vec = glove.get_word_vector("cat").expect("cat vector");
        let bird_vec = glove.get_word_vector("bird").expect("bird vector");

        let dog_cat_sim = cosine_similarity(&dog_vec, &cat_vec);
        let dog_bird_sim = cosine_similarity(&dog_vec, &bird_vec);

        // dog/cat share more context than dog/bird
        assert!(
            dog_cat_sim > dog_bird_sim,
            "Expected dog-cat ({}) > dog-bird ({})",
            dog_cat_sim,
            dog_bird_sim
        );
    }

    #[test]
    fn test_glove_training_empty_corpus() {
        let documents: Vec<&str> = vec![];
        let mut trainer = GloVeTrainer::new();
        let result = trainer.train(&documents);
        assert!(result.is_err());
    }

    #[test]
    fn test_glove_build_vocabulary() {
        let texts = vec![
            "the quick brown fox",
            "the lazy brown dog",
            "quick quick quick",
        ];

        let trainer = GloVeTrainer::with_config(GloVeTrainerConfig {
            min_count: 2,
            ..Default::default()
        });

        let (vocab, word_to_idx) = trainer.build_vocabulary(&texts).expect("build vocab");

        // "the" appears 2x, "quick" 4x, "brown" 2x
        // "fox", "lazy", "dog" each appear 1x -> filtered by min_count=2
        assert!(vocab.contains("the"));
        assert!(vocab.contains("quick"));
        assert!(vocab.contains("brown"));
        assert!(!vocab.contains("fox"));
        assert!(!vocab.contains("lazy"));
        assert_eq!(vocab.len(), word_to_idx.len());
    }

    #[test]
    fn test_glove_save_load_trained() {
        let documents = vec![
            "the quick brown fox jumps",
            "the lazy brown dog sleeps",
            "the fox and the dog run",
        ];

        let config = GloVeTrainerConfig {
            vector_size: 5,
            window_size: 2,
            min_count: 1,
            epochs: 5,
            learning_rate: 0.05,
            x_max: 10.0,
            alpha: 0.75,
        };

        let mut trainer = GloVeTrainer::with_config(config);
        let glove = trainer.train(&documents).expect("Training failed");

        let save_path = std::env::temp_dir().join("test_glove_trained_save.txt");
        glove.save(&save_path).expect("Failed to save");

        let loaded = GloVe::load(&save_path).expect("Failed to load");
        assert_eq!(loaded.vocabulary_size(), glove.vocabulary_size());
        assert_eq!(loaded.vector_size(), glove.vector_size());

        // Vectors should match closely (save uses 6 decimal places)
        for word in glove.get_words() {
            let orig = glove.get_word_vector(&word).expect("orig vec");
            let load = loaded.get_word_vector(&word).expect("loaded vec");
            for (a, b) in orig.iter().zip(load.iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "Mismatch for word '{}': {} vs {}",
                    word,
                    a,
                    b
                );
            }
        }

        std::fs::remove_file(save_path).ok();
    }

    #[test]
    fn test_glove_analogy_with_trained() {
        let documents = vec![
            "the quick brown fox jumps over the lazy dog",
            "a quick brown dog outpaces a lazy fox",
        ];

        let config = GloVeTrainerConfig {
            vector_size: 10,
            window_size: 3,
            min_count: 1,
            epochs: 5,
            ..Default::default()
        };

        let mut trainer = GloVeTrainer::with_config(config);
        let glove = trainer.train(&documents).expect("Training failed");

        // Just verify analogy doesn't crash (small corpus won't give meaningful results)
        let result = glove.analogy("the", "fox", "dog", 2);
        assert!(result.is_ok());
        let answers = result.expect("analogy failed");
        assert!(!answers.is_empty());
    }

    #[test]
    fn test_glove_most_similar_trained() {
        let documents = vec![
            "the dog runs in the park",
            "the cat runs in the yard",
            "the dog plays in the park",
            "the cat plays in the yard",
        ];

        let config = GloVeTrainerConfig {
            vector_size: 10,
            window_size: 3,
            min_count: 1,
            epochs: 30,
            ..Default::default()
        };

        let mut trainer = GloVeTrainer::with_config(config);
        let glove = trainer.train(&documents).expect("Training failed");

        let similar = glove.most_similar("dog", 3).expect("most_similar failed");
        assert!(!similar.is_empty());
        // Just verify we get results; small corpus won't always rank perfectly
    }

    #[test]
    fn test_glove_from_trained_validation() {
        let vocab = Vocabulary::new();
        let word_to_idx = HashMap::new();
        let main_emb = Array2::zeros((3, 5));
        let ctx_emb = Array2::zeros((4, 5)); // Mismatched rows

        let result = GloVe::from_trained(vocab, word_to_idx, main_emb, ctx_emb, 5);
        assert!(result.is_err());
    }
}
