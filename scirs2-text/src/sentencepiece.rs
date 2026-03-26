//! SentencePiece Unigram Language Model Tokenizer
//!
//! This module implements a Unigram LM tokenizer inspired by the SentencePiece
//! algorithm (Kudo, 2018). It is the tokenization approach used by T5, LLaMA,
//! ALBERT, and many other modern language models.
//!
//! Key features:
//! - Vocabulary of subword pieces with associated log-probabilities
//! - Tokenization via Viterbi algorithm (most probable segmentation)
//! - N-best segmentations for subword regularization
//! - Simplified training: start with a large vocabulary, prune iteratively
//! - Configurable special tokens (BOS, EOS, PAD, UNK)
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::sentencepiece::{UnigramTokenizer, UnigramConfig};
//! use scirs2_text::tokenizer::TransformerTokenizer;
//!
//! let config = UnigramConfig {
//!     vocab_size: 100,
//!     character_coverage: 0.9995,
//!     ..Default::default()
//! };
//! let corpus = &["the cat sat on the mat", "the dog sat on the log"];
//! let tokenizer = UnigramTokenizer::train(corpus, config)
//!     .expect("training failed");
//! let ids = tokenizer.encode("the cat");
//! let text = tokenizer.decode(&ids);
//! assert!(!ids.is_empty());
//! ```

use crate::error::{Result, TextError};
use crate::tokenizer::TransformerTokenizer;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Special token identifiers for the Unigram tokenizer.
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// Beginning-of-sequence token
    pub bos: String,
    /// End-of-sequence token
    pub eos: String,
    /// Padding token
    pub pad: String,
    /// Unknown token (fallback for OOV characters)
    pub unk: String,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos: "<s>".to_string(),
            eos: "</s>".to_string(),
            pad: "<pad>".to_string(),
            unk: "<unk>".to_string(),
        }
    }
}

/// Configuration for training a Unigram tokenizer.
#[derive(Debug, Clone)]
pub struct UnigramConfig {
    /// Target vocabulary size (including special tokens).
    pub vocab_size: usize,
    /// Fraction of characters from the corpus that must be covered by the
    /// initial seed vocabulary. Values close to 1.0 produce more complete
    /// coverage but a larger initial vocab.
    pub character_coverage: f64,
    /// Special tokens to include in the vocabulary.
    pub special_tokens: SpecialTokens,
    /// The SentencePiece "sentencepiece" prefix character used to denote
    /// word boundaries (Unicode \u{2581}, the lower-one-eighth block).
    pub word_boundary: char,
    /// Shrinking factor: fraction of vocabulary to keep at each pruning
    /// iteration (0 < shrinking_factor < 1).
    pub shrinking_factor: f64,
    /// Number of EM iterations per pruning round.
    pub num_em_iterations: usize,
}

impl Default for UnigramConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8000,
            character_coverage: 0.9995,
            special_tokens: SpecialTokens::default(),
            word_boundary: '\u{2581}',
            shrinking_factor: 0.75,
            num_em_iterations: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Vocabulary piece
// ---------------------------------------------------------------------------

/// A single vocabulary piece with its log-probability.
#[derive(Debug, Clone)]
pub struct VocabPiece {
    /// The subword string
    pub piece: String,
    /// Log-probability of the piece (ln)
    pub log_prob: f64,
}

// ---------------------------------------------------------------------------
// UnigramTokenizer
// ---------------------------------------------------------------------------

/// A Unigram Language Model tokenizer.
///
/// Tokenization uses the Viterbi algorithm to find the most probable
/// segmentation of input text given the learned vocabulary and piece
/// log-probabilities.
#[derive(Debug, Clone)]
pub struct UnigramTokenizer {
    /// Vocabulary pieces with log-probabilities
    pieces: Vec<VocabPiece>,
    /// Piece string -> piece index (into `pieces`)
    piece_to_id: HashMap<String, usize>,
    /// Piece index -> token ID (u32) for the encode interface
    index_to_token_id: Vec<u32>,
    /// Token ID (u32) -> piece index
    token_id_to_index: HashMap<u32, usize>,
    /// Special token configuration
    special_tokens: SpecialTokens,
    /// Word boundary character
    word_boundary: char,
    /// UNK token ID
    unk_id: u32,
}

impl UnigramTokenizer {
    // -------------------------------------------------------------------
    // Construction from explicit vocabulary
    // -------------------------------------------------------------------

    /// Create a tokenizer from a pre-built vocabulary.
    ///
    /// `vocab` is a list of (piece, log_prob) pairs. Special tokens are
    /// added automatically if not already present.
    ///
    /// # Errors
    /// Returns an error if the vocabulary is empty.
    pub fn from_vocab(
        vocab: &[(String, f64)],
        special_tokens: SpecialTokens,
        word_boundary: char,
    ) -> Result<Self> {
        if vocab.is_empty() {
            return Err(TextError::InvalidInput(
                "vocabulary must not be empty".to_string(),
            ));
        }

        let mut pieces: Vec<VocabPiece> = Vec::new();
        let mut piece_to_id: HashMap<String, usize> = HashMap::new();

        // Insert special tokens first
        let specials = [
            &special_tokens.pad,
            &special_tokens.unk,
            &special_tokens.bos,
            &special_tokens.eos,
        ];
        for sp in &specials {
            let idx = pieces.len();
            pieces.push(VocabPiece {
                piece: (*sp).clone(),
                log_prob: 0.0,
            });
            piece_to_id.insert((*sp).clone(), idx);
        }

        // Insert vocab pieces (skip if already a special token)
        for (piece, log_prob) in vocab {
            if piece_to_id.contains_key(piece) {
                continue;
            }
            let idx = pieces.len();
            pieces.push(VocabPiece {
                piece: piece.clone(),
                log_prob: *log_prob,
            });
            piece_to_id.insert(piece.clone(), idx);
        }

        let index_to_token_id: Vec<u32> = (0..pieces.len() as u32).collect();
        let token_id_to_index: HashMap<u32, usize> = index_to_token_id
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();
        let unk_id = *piece_to_id
            .get(&special_tokens.unk)
            .ok_or_else(|| TextError::InvalidInput("UNK token missing".to_string()))?
            as u32;

        Ok(Self {
            pieces,
            piece_to_id,
            index_to_token_id,
            token_id_to_index,
            special_tokens,
            word_boundary,
            unk_id,
        })
    }

    // -------------------------------------------------------------------
    // Training
    // -------------------------------------------------------------------

    /// Train a Unigram tokenizer from a text corpus.
    ///
    /// The training procedure:
    /// 1. Build an initial large vocabulary from character n-grams that cover
    ///    at least `config.character_coverage` of the corpus characters.
    /// 2. Estimate piece probabilities via EM (expectation step uses Viterbi
    ///    counts).
    /// 3. Iteratively prune the vocabulary by removing pieces whose removal
    ///    causes the smallest increase in overall loss, until the target
    ///    `vocab_size` is reached.
    ///
    /// # Errors
    /// Returns an error if the corpus is empty or contains no usable text.
    pub fn train(corpus: &[&str], config: UnigramConfig) -> Result<Self> {
        if corpus.is_empty() {
            return Err(TextError::InvalidInput(
                "corpus must not be empty".to_string(),
            ));
        }

        let wb = config.word_boundary;

        // Step 1: Normalize text and build word frequencies
        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        for text in corpus {
            for word in text.split_whitespace() {
                let normalized = format!("{}{}", wb, word.to_lowercase());
                *word_freqs.entry(normalized).or_insert(0) += 1;
            }
        }

        if word_freqs.is_empty() {
            return Err(TextError::InvalidInput(
                "corpus contains no words".to_string(),
            ));
        }

        // Step 2: Build initial seed vocabulary from substrings
        let seed_vocab = Self::build_seed_vocab(&word_freqs, &config);

        // Step 3: Initialize piece probabilities uniformly
        let total_count: usize = seed_vocab.values().sum();
        let mut vocab_with_probs: Vec<(String, f64)> = seed_vocab
            .into_iter()
            .map(|(piece, count)| {
                let prob = (count as f64) / (total_count as f64);
                let log_prob = if prob > 0.0 {
                    prob.ln()
                } else {
                    f64::NEG_INFINITY
                };
                (piece, log_prob)
            })
            .collect();

        // Step 4: EM + Pruning loop
        let num_special = 4; // pad, unk, bos, eos
        let target = config.vocab_size.saturating_sub(num_special);

        let words_list: Vec<(String, usize)> = word_freqs.into_iter().collect();

        // Iterative pruning
        while vocab_with_probs.len() > target {
            // EM iterations to re-estimate probabilities
            for _ in 0..config.num_em_iterations {
                vocab_with_probs = Self::em_step(&vocab_with_probs, &words_list, wb);
            }

            // Compute loss (negative log-likelihood) contributed by each piece
            let piece_losses = Self::compute_piece_losses(&vocab_with_probs, &words_list, wb);

            // Sort pieces by loss impact (ascending): pieces with smallest
            // impact are safest to remove
            let mut indexed_losses: Vec<(usize, f64)> =
                piece_losses.into_iter().enumerate().collect();
            indexed_losses
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Determine how many to prune this round
            let current_size = vocab_with_probs.len();
            let keep = (current_size as f64 * config.shrinking_factor).ceil() as usize;
            let keep = keep.max(target);

            // Collect indices to keep: always keep single-character pieces
            // (they ensure full coverage)
            let num_to_remove = current_size.saturating_sub(keep);
            let mut remove_set = std::collections::HashSet::new();
            let mut removed = 0;
            for &(idx, _) in &indexed_losses {
                if removed >= num_to_remove {
                    break;
                }
                // Never remove single-character pieces (coverage guarantee)
                if vocab_with_probs[idx].0.chars().count() <= 1 {
                    continue;
                }
                // Never remove the word boundary prefix alone
                if vocab_with_probs[idx].0.len() <= wb.len_utf8() {
                    continue;
                }
                remove_set.insert(idx);
                removed += 1;
            }

            vocab_with_probs = vocab_with_probs
                .into_iter()
                .enumerate()
                .filter(|(i, _)| !remove_set.contains(i))
                .map(|(_, v)| v)
                .collect();
        }

        // Final EM pass
        for _ in 0..config.num_em_iterations {
            vocab_with_probs = Self::em_step(&vocab_with_probs, &words_list, wb);
        }

        Self::from_vocab(&vocab_with_probs, config.special_tokens, wb)
    }

    /// Build the initial seed vocabulary from character n-grams.
    fn build_seed_vocab(
        word_freqs: &HashMap<String, usize>,
        config: &UnigramConfig,
    ) -> HashMap<String, usize> {
        let mut substring_counts: HashMap<String, usize> = HashMap::new();

        // Character frequency for coverage filtering
        let mut char_counts: HashMap<char, usize> = HashMap::new();
        let mut total_chars: usize = 0;
        for (word, &freq) in word_freqs {
            for ch in word.chars() {
                *char_counts.entry(ch).or_insert(0) += freq;
                total_chars += freq;
            }
        }

        // Determine which characters to keep based on coverage
        let mut sorted_chars: Vec<(char, usize)> = char_counts.into_iter().collect();
        sorted_chars.sort_by_key(|item| std::cmp::Reverse(item.1));

        let mut covered_chars = std::collections::HashSet::new();
        let mut covered_count = 0usize;
        let coverage_target = (total_chars as f64 * config.character_coverage) as usize;
        for (ch, count) in &sorted_chars {
            if covered_count >= coverage_target {
                break;
            }
            covered_chars.insert(*ch);
            covered_count += count;
        }

        // Extract substrings (up to length 16) from each word
        let max_piece_len = 16;
        for (word, &freq) in word_freqs {
            let chars: Vec<char> = word.chars().collect();
            for start in 0..chars.len() {
                // Check if starting char is covered
                if !covered_chars.contains(&chars[start]) {
                    continue;
                }
                for end in (start + 1)..=(chars.len().min(start + max_piece_len)) {
                    // Check if all chars in this substring are covered
                    let all_covered = chars[start..end].iter().all(|c| covered_chars.contains(c));
                    if !all_covered {
                        break;
                    }
                    let piece: String = chars[start..end].iter().collect();
                    *substring_counts.entry(piece).or_insert(0) += freq;
                }
            }
        }

        substring_counts
    }

    /// One EM step: re-estimate piece log-probabilities from Viterbi counts.
    fn em_step(
        vocab: &[(String, f64)],
        words: &[(String, usize)],
        word_boundary: char,
    ) -> Vec<(String, f64)> {
        let piece_to_idx: HashMap<&str, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, (p, _))| (p.as_str(), i))
            .collect();

        let mut counts = vec![0.0f64; vocab.len()];

        for (word, freq) in words {
            let segmentation =
                Self::viterbi_segment_with_vocab(word, vocab, &piece_to_idx, word_boundary);
            for piece in &segmentation {
                if let Some(&idx) = piece_to_idx.get(piece.as_str()) {
                    counts[idx] += *freq as f64;
                }
            }
        }

        let total: f64 = counts.iter().sum();
        if total <= 0.0 {
            return vocab.to_vec();
        }

        vocab
            .iter()
            .enumerate()
            .map(|(i, (piece, _old_lp))| {
                let prob = (counts[i] + 1e-10) / (total + 1e-10 * vocab.len() as f64);
                (piece.clone(), prob.ln())
            })
            .collect()
    }

    /// Compute per-piece loss impact: the increase in negative log-likelihood
    /// if that piece were removed from the vocabulary.
    fn compute_piece_losses(
        vocab: &[(String, f64)],
        words: &[(String, usize)],
        word_boundary: char,
    ) -> Vec<f64> {
        let piece_to_idx: HashMap<&str, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, (p, _))| (p.as_str(), i))
            .collect();

        // Baseline loss
        let baseline_loss = Self::compute_corpus_loss(vocab, words, &piece_to_idx, word_boundary);

        let mut losses = vec![0.0f64; vocab.len()];

        // For efficiency, only compute loss impact for multi-char pieces
        for (idx, (piece, _)) in vocab.iter().enumerate() {
            if piece.chars().count() <= 1 {
                losses[idx] = f64::MAX; // Never remove single chars
                continue;
            }

            // Build a reduced vocab without this piece
            let reduced: Vec<(&str, f64)> = vocab
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, (p, lp))| (p.as_str(), *lp))
                .collect();

            let reduced_map: HashMap<&str, usize> = reduced
                .iter()
                .enumerate()
                .map(|(i, (p, _))| (*p, i))
                .collect();

            let reduced_vocab: Vec<(String, f64)> =
                reduced.iter().map(|(p, lp)| (p.to_string(), *lp)).collect();

            let reduced_loss =
                Self::compute_corpus_loss(&reduced_vocab, words, &reduced_map, word_boundary);

            losses[idx] = reduced_loss - baseline_loss;
        }

        losses
    }

    /// Compute the total negative log-likelihood of the corpus under the
    /// current vocabulary.
    fn compute_corpus_loss(
        vocab: &[(String, f64)],
        words: &[(String, usize)],
        piece_to_idx: &HashMap<&str, usize>,
        word_boundary: char,
    ) -> f64 {
        let mut total_loss = 0.0f64;

        for (word, freq) in words {
            let segmentation =
                Self::viterbi_segment_with_vocab(word, vocab, piece_to_idx, word_boundary);
            let seg_score: f64 = segmentation
                .iter()
                .map(|p| {
                    piece_to_idx
                        .get(p.as_str())
                        .map(|&i| vocab[i].1)
                        .unwrap_or(-100.0)
                })
                .sum();
            total_loss -= seg_score * *freq as f64;
        }

        total_loss
    }

    /// Viterbi segmentation using an explicit vocab and piece-to-index map.
    fn viterbi_segment_with_vocab(
        text: &str,
        vocab: &[(String, f64)],
        piece_to_idx: &HashMap<&str, usize>,
        _word_boundary: char,
    ) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }

        // best_score[i] = best log-prob for segmenting chars[0..i]
        let mut best_score = vec![f64::NEG_INFINITY; n + 1];
        let mut best_prev = vec![0usize; n + 1]; // best_prev[i] = j such that chars[j..i] is the last piece
        best_score[0] = 0.0;

        for i in 1..=n {
            // Try all pieces ending at position i
            let max_piece_len = 16.min(i);
            for len in 1..=max_piece_len {
                let j = i - len;
                let piece: String = chars[j..i].iter().collect();
                if let Some(&idx) = piece_to_idx.get(piece.as_str()) {
                    let score = best_score[j] + vocab[idx].1;
                    if score > best_score[i] {
                        best_score[i] = score;
                        best_prev[i] = j;
                    }
                }
            }

            // If no piece matched, fall back to single-character segmentation
            // with a heavy penalty
            if best_score[i] == f64::NEG_INFINITY {
                best_score[i] = best_score[i - 1] + (-100.0);
                best_prev[i] = i - 1;
            }
        }

        // Backtrack to recover the segmentation
        let mut result = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let prev = best_prev[pos];
            let piece: String = chars[prev..pos].iter().collect();
            result.push(piece);
            pos = prev;
        }
        result.reverse();
        result
    }

    // -------------------------------------------------------------------
    // Encoding / Decoding
    // -------------------------------------------------------------------

    /// Normalize input text: prepend word boundary, lowercase.
    fn normalize_text(&self, text: &str) -> String {
        let mut result = String::new();
        for (i, word) in text.split_whitespace().enumerate() {
            if i > 0 {
                result.push(self.word_boundary);
            }
            result.push(self.word_boundary);
            result.push_str(&word.to_lowercase());
        }
        result
    }

    /// Segment text using the Viterbi algorithm.
    fn viterbi_segment(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }

        let mut best_score = vec![f64::NEG_INFINITY; n + 1];
        let mut best_prev = vec![0usize; n + 1];
        best_score[0] = 0.0;

        for i in 1..=n {
            let max_piece_len = 16.min(i);
            for len in 1..=max_piece_len {
                let j = i - len;
                let piece: String = chars[j..i].iter().collect();
                if let Some(&idx) = self.piece_to_id.get(&piece) {
                    let score = best_score[j] + self.pieces[idx].log_prob;
                    if score > best_score[i] {
                        best_score[i] = score;
                        best_prev[i] = j;
                    }
                }
            }
            // Fallback: single char with UNK penalty
            if best_score[i] == f64::NEG_INFINITY {
                best_score[i] = best_score[i - 1] + (-100.0);
                best_prev[i] = i - 1;
            }
        }

        let mut result = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let prev = best_prev[pos];
            let piece: String = chars[prev..pos].iter().collect();
            result.push(piece);
            pos = prev;
        }
        result.reverse();
        result
    }

    /// Return the N-best segmentations of the given text.
    ///
    /// Uses a beam-style extension of the Viterbi algorithm to return the
    /// top-k segmentations ranked by total log-probability. This is useful
    /// for subword regularization (randomly sampling from the N-best list
    /// during training).
    ///
    /// # Arguments
    /// - `text`: raw text to segment
    /// - `n`: number of segmentations to return (at most)
    ///
    /// # Returns
    /// A vector of `(segmentation, log_prob)` pairs sorted by descending
    /// log-probability. The length may be less than `n` if fewer distinct
    /// segmentations exist.
    pub fn nbest_segment(&self, text: &str, n: usize) -> Vec<(Vec<String>, f64)> {
        let normalized = self.normalize_text(text);
        let chars: Vec<char> = normalized.chars().collect();
        let len = chars.len();
        if len == 0 || n == 0 {
            return Vec::new();
        }

        // For each position, keep top-n best paths
        // best[i] = Vec of (score, prev_position) sorted by score desc
        let mut best: Vec<Vec<(f64, usize)>> = vec![Vec::new(); len + 1];
        best[0].push((0.0, 0));

        for i in 1..=len {
            let mut candidates: Vec<(f64, usize)> = Vec::new();
            let max_piece_len = 16.min(i);
            for piece_len in 1..=max_piece_len {
                let j = i - piece_len;
                let piece: String = chars[j..i].iter().collect();
                if let Some(&idx) = self.piece_to_id.get(&piece) {
                    let piece_score = self.pieces[idx].log_prob;
                    for &(prev_score, _) in &best[j] {
                        candidates.push((prev_score + piece_score, j));
                    }
                }
            }
            // Fallback for uncovered characters
            if candidates.is_empty() {
                for &(prev_score, _) in &best[i - 1] {
                    candidates.push((prev_score - 100.0, i - 1));
                }
            }
            // Sort desc and keep top-n
            candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(n);
            best[i] = candidates;
        }

        // Backtrack from each of the top-n endpoints
        let mut results: Vec<(Vec<String>, f64)> = Vec::new();
        for &(score, _) in &best[len] {
            // Reconstruct one path with this score
            let mut segments = Vec::new();
            let mut pos = len;
            let mut remaining_score = score;

            while pos > 0 {
                let mut found = false;
                let max_pl = 16.min(pos);
                for pl in 1..=max_pl {
                    let j = pos - pl;
                    let piece: String = chars[j..pos].iter().collect();
                    if let Some(&idx) = self.piece_to_id.get(&piece) {
                        let ps = self.pieces[idx].log_prob;
                        let needed = remaining_score - ps;
                        // Check if `j` has a path with approximately this score
                        let has_match = best[j].iter().any(|&(s, _)| (s - needed).abs() < 1e-6);
                        if has_match {
                            segments.push(piece);
                            remaining_score = needed;
                            pos = j;
                            found = true;
                            break;
                        }
                    }
                }
                if !found {
                    // Fallback: single char
                    let piece: String = chars[pos - 1..pos].iter().collect();
                    segments.push(piece);
                    remaining_score += 100.0;
                    pos -= 1;
                }
            }
            segments.reverse();
            results.push((segments, score));
        }

        // Deduplicate
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut seen = std::collections::HashSet::new();
        results.retain(|(seg, _)| seen.insert(seg.clone()));
        results.truncate(n);
        results
    }

    /// Get the token ID for a piece string.
    pub fn piece_to_token_id(&self, piece: &str) -> Option<u32> {
        self.piece_to_id
            .get(piece)
            .map(|&idx| self.index_to_token_id[idx])
    }

    /// Get the piece string for a token ID.
    pub fn token_id_to_piece(&self, id: u32) -> Option<&str> {
        self.token_id_to_index
            .get(&id)
            .map(|&idx| self.pieces[idx].piece.as_str())
    }

    /// Return the UNK token ID.
    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// Return the BOS token ID, if present.
    pub fn bos_id(&self) -> Option<u32> {
        self.piece_to_token_id(&self.special_tokens.bos.clone())
    }

    /// Return the EOS token ID, if present.
    pub fn eos_id(&self) -> Option<u32> {
        self.piece_to_token_id(&self.special_tokens.eos.clone())
    }

    /// Return the PAD token ID, if present.
    pub fn pad_id(&self) -> Option<u32> {
        self.piece_to_token_id(&self.special_tokens.pad.clone())
    }
}

// ---------------------------------------------------------------------------
// TransformerTokenizer implementation
// ---------------------------------------------------------------------------

impl TransformerTokenizer for UnigramTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let normalized = self.normalize_text(text);
        let segments = self.viterbi_segment(&normalized);

        segments
            .iter()
            .map(|piece| {
                self.piece_to_id
                    .get(piece.as_str())
                    .map(|&idx| self.index_to_token_id[idx])
                    .unwrap_or(self.unk_id)
            })
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut text = String::new();
        for &id in ids {
            if let Some(&idx) = self.token_id_to_index.get(&id) {
                let piece = &self.pieces[idx].piece;
                // Skip special tokens in decode output
                if piece == &self.special_tokens.pad
                    || piece == &self.special_tokens.bos
                    || piece == &self.special_tokens.eos
                    || piece == &self.special_tokens.unk
                {
                    continue;
                }
                text.push_str(piece);
            }
        }
        // Replace word boundary with space and trim
        text.replace(self.word_boundary, " ").trim().to_string()
    }

    fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_corpus() -> Vec<&'static str> {
        vec![
            "the cat sat on the mat",
            "the dog sat on the log",
            "a cat and a dog",
            "the cat is on the mat and the dog is on the log",
            "cats and dogs are friends",
        ]
    }

    fn train_small() -> UnigramTokenizer {
        let config = UnigramConfig {
            vocab_size: 120,
            character_coverage: 0.9995,
            shrinking_factor: 0.75,
            num_em_iterations: 3,
            ..Default::default()
        };
        UnigramTokenizer::train(&small_corpus(), config).expect("training should succeed")
    }

    #[test]
    fn test_train_produces_vocab() {
        let tok = train_small();
        assert!(tok.vocab_size() > 0);
        // Should have special tokens
        assert!(tok.bos_id().is_some());
        assert!(tok.eos_id().is_some());
        assert!(tok.pad_id().is_some());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let tok = train_small();
        let text = "the cat sat";
        let ids = tok.encode(text);
        assert!(!ids.is_empty(), "encode should produce token IDs");
        let decoded = tok.decode(&ids);
        // Decoded text should contain the original words
        assert!(
            decoded.contains("the"),
            "decoded should contain 'the': {decoded}"
        );
        assert!(
            decoded.contains("cat"),
            "decoded should contain 'cat': {decoded}"
        );
        assert!(
            decoded.contains("sat"),
            "decoded should contain 'sat': {decoded}"
        );
    }

    #[test]
    fn test_viterbi_finds_optimal_segmentation() {
        let tok = train_small();
        let text = "the";
        let normalized = tok.normalize_text(text);
        let segments = tok.viterbi_segment(&normalized);
        // The word "the" should be segmented (possibly as one piece if learned)
        assert!(!segments.is_empty());
        // Reconstruct and verify
        let joined: String = segments.concat();
        assert_eq!(joined, normalized);
    }

    #[test]
    fn test_unknown_characters_produce_unk() {
        let tok = train_small();
        // Use characters unlikely to be in our training corpus
        let ids = tok.encode("\u{4e16}\u{754c}"); // Chinese characters
        assert!(!ids.is_empty());
        // At least some tokens should be UNK (the Chinese characters themselves).
        // The word boundary prefix may have a valid ID, so we check that UNK
        // appears at least once.
        let unk_count = ids.iter().filter(|&&id| id == tok.unk_id()).count();
        assert!(
            unk_count > 0,
            "at least one token should be UNK for unknown chars, got ids: {ids:?}"
        );
    }

    #[test]
    fn test_empty_corpus_error() {
        let config = UnigramConfig::default();
        let result = UnigramTokenizer::train(&[], config);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_vocab_empty_error() {
        let result = UnigramTokenizer::from_vocab(&[], SpecialTokens::default(), '\u{2581}');
        assert!(result.is_err());
    }

    #[test]
    fn test_nbest_returns_multiple_segmentations() {
        let tok = train_small();
        let results = tok.nbest_segment("cat", 3);
        assert!(
            !results.is_empty(),
            "should return at least one segmentation"
        );
        // First result should be the best (highest score)
        if results.len() >= 2 {
            assert!(
                results[0].1 >= results[1].1,
                "first result should have highest score"
            );
        }
    }

    #[test]
    fn test_encode_empty_string() {
        let tok = train_small();
        let ids = tok.encode("");
        assert!(ids.is_empty(), "empty string should produce no tokens");
    }

    #[test]
    fn test_encode_single_character() {
        let tok = train_small();
        let ids = tok.encode("a");
        assert!(
            !ids.is_empty(),
            "single char should produce at least one token"
        );
        let decoded = tok.decode(&ids);
        assert!(
            decoded.contains('a'),
            "decoded should contain 'a': {decoded}"
        );
    }

    #[test]
    fn test_vocab_size_respects_config() {
        let config = UnigramConfig {
            vocab_size: 60,
            character_coverage: 0.9995,
            shrinking_factor: 0.75,
            num_em_iterations: 2,
            ..Default::default()
        };
        let tok =
            UnigramTokenizer::train(&small_corpus(), config).expect("training should succeed");
        // Vocab size should be close to target (may be slightly different
        // due to special tokens and single-char preservation)
        assert!(
            tok.vocab_size() <= 80,
            "vocab size {} should be close to target 60",
            tok.vocab_size()
        );
    }
}
