//! Batch Tokenization with Padding, Truncation, and Attention Masks
//!
//! This module provides utilities for tokenizing multiple texts efficiently,
//! producing padded/truncated sequences with attention masks suitable for
//! batch inference with transformer models.
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::batch_tokenizer::{batch_encode, PaddingStrategy, TruncationStrategy, BatchConfig};
//! use scirs2_text::tokenizer::{BPETokenizer, TransformerTokenizer};
//!
//! let corpus = &["the cat sat on the mat", "the dog sat on the log"];
//! let tokenizer = BPETokenizer::train(corpus, 100).expect("train failed");
//!
//! let texts = &["the cat", "the dog sat"];
//! let config = BatchConfig {
//!     max_length: Some(10),
//!     padding: PaddingStrategy::LongestInBatch,
//!     truncation: TruncationStrategy::Right,
//!     pad_token_id: 0,
//! };
//! let batch = batch_encode(texts, &tokenizer, &config);
//! assert_eq!(batch.input_ids.len(), 2);
//! assert_eq!(batch.attention_mask.len(), 2);
//! ```

use crate::tokenizer::TransformerTokenizer;

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Strategy for padding sequences to the same length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// No padding: sequences are returned as-is (varying lengths).
    NoPadding,
    /// Pad all sequences to `max_length` (requires `max_length` to be set).
    MaxLength,
    /// Pad all sequences to the length of the longest sequence in the batch.
    LongestInBatch,
}

/// Strategy for truncating sequences that exceed `max_length`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// No truncation: sequences are returned at full length.
    NoTruncation,
    /// Truncate from the right (keep the beginning).
    Right,
    /// Truncate from the left (keep the end).
    Left,
}

/// Side on which to add padding tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingSide {
    /// Add padding tokens on the right (default, used by most models).
    Right,
    /// Add padding tokens on the left (used by some decoder-only models).
    Left,
}

/// Configuration for batch encoding.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum sequence length. When `None`, no length limit is imposed
    /// (but padding to longest may still apply).
    pub max_length: Option<usize>,
    /// Padding strategy.
    pub padding: PaddingStrategy,
    /// Truncation strategy.
    pub truncation: TruncationStrategy,
    /// Token ID to use for padding.
    pub pad_token_id: u32,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_length: None,
            padding: PaddingStrategy::LongestInBatch,
            truncation: TruncationStrategy::Right,
            pad_token_id: 0,
        }
    }
}

/// Extended configuration with padding side.
#[derive(Debug, Clone)]
pub struct BatchConfigExt {
    /// Base configuration.
    pub base: BatchConfig,
    /// Side on which to add padding.
    pub padding_side: PaddingSide,
}

impl Default for BatchConfigExt {
    fn default() -> Self {
        Self {
            base: BatchConfig::default(),
            padding_side: PaddingSide::Right,
        }
    }
}

// ---------------------------------------------------------------------------
// BatchEncoding output
// ---------------------------------------------------------------------------

/// The result of batch-encoding a set of texts.
///
/// All inner vectors have the same outer length (= number of texts). When
/// padding is enabled, all inner `Vec<u32>` have the same length (= padded
/// sequence length).
#[derive(Debug, Clone)]
pub struct BatchEncoding {
    /// Token IDs for each input text, possibly padded and/or truncated.
    pub input_ids: Vec<Vec<u32>>,
    /// Attention mask: `1` for real tokens, `0` for padding tokens.
    pub attention_mask: Vec<Vec<u32>>,
    /// Original (pre-padding, pre-truncation) lengths of each sequence.
    pub lengths: Vec<usize>,
}

impl BatchEncoding {
    /// Number of sequences in the batch.
    pub fn batch_size(&self) -> usize {
        self.input_ids.len()
    }

    /// Length of the (padded) sequences. Returns 0 for an empty batch.
    pub fn seq_length(&self) -> usize {
        self.input_ids.first().map_or(0, |v| v.len())
    }

    /// Return total number of real (non-padding) tokens across the batch.
    pub fn total_real_tokens(&self) -> usize {
        self.attention_mask
            .iter()
            .flat_map(|mask| mask.iter())
            .filter(|&&v| v == 1)
            .count()
    }
}

// ---------------------------------------------------------------------------
// Batch encoding functions
// ---------------------------------------------------------------------------

/// Truncate a sequence according to the given strategy and max_length.
fn truncate(ids: &[u32], strategy: TruncationStrategy, max_length: usize) -> Vec<u32> {
    if ids.len() <= max_length {
        return ids.to_vec();
    }
    match strategy {
        TruncationStrategy::NoTruncation => ids.to_vec(),
        TruncationStrategy::Right => ids[..max_length].to_vec(),
        TruncationStrategy::Left => ids[ids.len() - max_length..].to_vec(),
    }
}

/// Pad a sequence to `target_length` with `pad_id`, adding padding on the right.
fn pad_right(ids: &[u32], target_length: usize, pad_id: u32) -> (Vec<u32>, Vec<u32>) {
    let real_len = ids.len();
    if real_len >= target_length {
        let truncated = &ids[..target_length];
        let mask = vec![1u32; target_length];
        return (truncated.to_vec(), mask);
    }
    let mut padded = ids.to_vec();
    let mut mask = vec![1u32; real_len];
    let pad_count = target_length - real_len;
    padded.extend(std::iter::repeat_n(pad_id, pad_count));
    mask.extend(std::iter::repeat_n(0u32, pad_count));
    (padded, mask)
}

/// Pad a sequence to `target_length` with `pad_id`, adding padding on the left.
fn pad_left(ids: &[u32], target_length: usize, pad_id: u32) -> (Vec<u32>, Vec<u32>) {
    let real_len = ids.len();
    if real_len >= target_length {
        let start = real_len - target_length;
        let truncated = &ids[start..];
        let mask = vec![1u32; target_length];
        return (truncated.to_vec(), mask);
    }
    let pad_count = target_length - real_len;
    let mut padded: Vec<u32> = std::iter::repeat_n(pad_id, pad_count).collect();
    let mut mask: Vec<u32> = std::iter::repeat_n(0u32, pad_count).collect();
    padded.extend_from_slice(ids);
    mask.extend(std::iter::repeat_n(1u32, real_len));
    (padded, mask)
}

/// Batch-encode multiple texts using a tokenizer.
///
/// Each text is independently encoded, then optionally truncated and padded
/// according to the provided [`BatchConfig`].
///
/// Padding is added on the **right** side (standard for most models).
/// For left-padding, use [`batch_encode_ext`].
///
/// # Arguments
/// - `texts`: slice of input strings
/// - `tokenizer`: any tokenizer implementing [`TransformerTokenizer`]
/// - `config`: batch configuration (max_length, padding, truncation, pad token)
pub fn batch_encode<T: TransformerTokenizer>(
    texts: &[&str],
    tokenizer: &T,
    config: &BatchConfig,
) -> BatchEncoding {
    if texts.is_empty() {
        return BatchEncoding {
            input_ids: Vec::new(),
            attention_mask: Vec::new(),
            lengths: Vec::new(),
        };
    }

    // Step 1: Encode all texts
    let mut encoded: Vec<Vec<u32>> = texts.iter().map(|t| tokenizer.encode(t)).collect();
    let original_lengths: Vec<usize> = encoded.iter().map(|v| v.len()).collect();

    // Step 2: Truncation
    if let Some(max_len) = config.max_length {
        if config.truncation != TruncationStrategy::NoTruncation {
            for seq in &mut encoded {
                *seq = truncate(seq, config.truncation, max_len);
            }
        }
    }

    // Step 3: Determine target length for padding
    let target_length = match config.padding {
        PaddingStrategy::NoPadding => {
            // No padding: return as-is with per-sequence masks
            let attention_mask: Vec<Vec<u32>> =
                encoded.iter().map(|seq| vec![1u32; seq.len()]).collect();
            return BatchEncoding {
                input_ids: encoded,
                attention_mask,
                lengths: original_lengths,
            };
        }
        PaddingStrategy::MaxLength => config
            .max_length
            .unwrap_or_else(|| encoded.iter().map(|s| s.len()).max().unwrap_or(0)),
        PaddingStrategy::LongestInBatch => {
            let longest = encoded.iter().map(|s| s.len()).max().unwrap_or(0);
            // If max_length is set, cap at max_length
            match config.max_length {
                Some(ml) => longest.min(ml),
                None => longest,
            }
        }
    };

    // Step 4: Pad sequences
    let mut input_ids = Vec::with_capacity(encoded.len());
    let mut attention_mask = Vec::with_capacity(encoded.len());

    for seq in &encoded {
        let (padded, mask) = pad_right(seq, target_length, config.pad_token_id);
        input_ids.push(padded);
        attention_mask.push(mask);
    }

    BatchEncoding {
        input_ids,
        attention_mask,
        lengths: original_lengths,
    }
}

/// Batch-encode with extended configuration (including padding side).
///
/// Same as [`batch_encode`] but allows choosing left or right padding.
pub fn batch_encode_ext<T: TransformerTokenizer>(
    texts: &[&str],
    tokenizer: &T,
    config: &BatchConfigExt,
) -> BatchEncoding {
    if texts.is_empty() {
        return BatchEncoding {
            input_ids: Vec::new(),
            attention_mask: Vec::new(),
            lengths: Vec::new(),
        };
    }

    // Step 1: Encode all texts
    let mut encoded: Vec<Vec<u32>> = texts.iter().map(|t| tokenizer.encode(t)).collect();
    let original_lengths: Vec<usize> = encoded.iter().map(|v| v.len()).collect();

    // Step 2: Truncation
    if let Some(max_len) = config.base.max_length {
        if config.base.truncation != TruncationStrategy::NoTruncation {
            for seq in &mut encoded {
                *seq = truncate(seq, config.base.truncation, max_len);
            }
        }
    }

    // Step 3: Determine target length
    let target_length = match config.base.padding {
        PaddingStrategy::NoPadding => {
            let attention_mask: Vec<Vec<u32>> =
                encoded.iter().map(|seq| vec![1u32; seq.len()]).collect();
            return BatchEncoding {
                input_ids: encoded,
                attention_mask,
                lengths: original_lengths,
            };
        }
        PaddingStrategy::MaxLength => config
            .base
            .max_length
            .unwrap_or_else(|| encoded.iter().map(|s| s.len()).max().unwrap_or(0)),
        PaddingStrategy::LongestInBatch => {
            let longest = encoded.iter().map(|s| s.len()).max().unwrap_or(0);
            match config.base.max_length {
                Some(ml) => longest.min(ml),
                None => longest,
            }
        }
    };

    // Step 4: Pad sequences
    let pad_fn = match config.padding_side {
        PaddingSide::Right => pad_right,
        PaddingSide::Left => pad_left,
    };

    let mut input_ids = Vec::with_capacity(encoded.len());
    let mut attention_mask = Vec::with_capacity(encoded.len());

    for seq in &encoded {
        let (padded, mask) = pad_fn(seq, target_length, config.base.pad_token_id);
        input_ids.push(padded);
        attention_mask.push(mask);
    }

    BatchEncoding {
        input_ids,
        attention_mask,
        lengths: original_lengths,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::BPETokenizer;

    fn train_tokenizer() -> BPETokenizer {
        let corpus = &[
            "the cat sat on the mat",
            "the dog sat on the log",
            "cats and dogs",
            "the quick brown fox",
        ];
        BPETokenizer::train(corpus, 100).expect("training should succeed")
    }

    #[test]
    fn test_batch_encode_basic() {
        let tok = train_tokenizer();
        let texts = &["the cat", "the dog sat"];
        let config = BatchConfig {
            padding: PaddingStrategy::LongestInBatch,
            ..Default::default()
        };
        let batch = batch_encode(texts, &tok, &config);
        assert_eq!(batch.batch_size(), 2);
        // Both sequences should be padded to the same length
        assert_eq!(batch.input_ids[0].len(), batch.input_ids[1].len());
        assert_eq!(batch.attention_mask[0].len(), batch.attention_mask[1].len());
    }

    #[test]
    fn test_padding_adds_correct_tokens() {
        let tok = train_tokenizer();
        let texts = &["the", "the cat sat on the mat"];
        let config = BatchConfig {
            padding: PaddingStrategy::LongestInBatch,
            pad_token_id: 0,
            ..Default::default()
        };
        let batch = batch_encode(texts, &tok, &config);

        // The shorter sequence should have padding
        let shorter_len = batch.lengths[0];
        let padded_len = batch.input_ids[0].len();
        if shorter_len < padded_len {
            // Padding tokens (0) should appear at the end
            for i in shorter_len..padded_len {
                assert_eq!(
                    batch.input_ids[0][i], 0,
                    "padding token should be 0 at position {i}"
                );
            }
        }
    }

    #[test]
    fn test_attention_mask_correct() {
        let tok = train_tokenizer();
        let texts = &["the", "the cat sat"];
        let config = BatchConfig {
            padding: PaddingStrategy::LongestInBatch,
            pad_token_id: 0,
            ..Default::default()
        };
        let batch = batch_encode(texts, &tok, &config);

        // For the shorter sequence, mask should be 1 for real tokens, 0 for padding
        let shorter_len = batch.lengths[0];
        for i in 0..shorter_len.min(batch.attention_mask[0].len()) {
            assert_eq!(
                batch.attention_mask[0][i], 1,
                "real token at {i} should have mask 1"
            );
        }
        for i in shorter_len..batch.attention_mask[0].len() {
            assert_eq!(
                batch.attention_mask[0][i], 0,
                "padding at {i} should have mask 0"
            );
        }
    }

    #[test]
    fn test_truncation_right() {
        let tok = train_tokenizer();
        let texts = &["the cat sat on the mat"];
        let config = BatchConfig {
            max_length: Some(3),
            padding: PaddingStrategy::NoPadding,
            truncation: TruncationStrategy::Right,
            pad_token_id: 0,
        };
        let batch = batch_encode(texts, &tok, &config);
        assert!(
            batch.input_ids[0].len() <= 3,
            "truncated length should be <= 3, got {}",
            batch.input_ids[0].len()
        );
    }

    #[test]
    fn test_truncation_left() {
        let tok = train_tokenizer();
        let texts = &["the cat sat on the mat"];
        let config = BatchConfig {
            max_length: Some(3),
            padding: PaddingStrategy::NoPadding,
            truncation: TruncationStrategy::Left,
            pad_token_id: 0,
        };
        let batch = batch_encode(texts, &tok, &config);
        assert!(
            batch.input_ids[0].len() <= 3,
            "truncated length should be <= 3"
        );
    }

    #[test]
    fn test_no_padding_varying_lengths() {
        let tok = train_tokenizer();
        let texts = &["the", "the cat sat"];
        let config = BatchConfig {
            padding: PaddingStrategy::NoPadding,
            truncation: TruncationStrategy::NoTruncation,
            ..Default::default()
        };
        let batch = batch_encode(texts, &tok, &config);
        // Sequences should have different lengths
        assert_eq!(batch.input_ids[0].len(), batch.lengths[0]);
        assert_eq!(batch.input_ids[1].len(), batch.lengths[1]);
    }

    #[test]
    fn test_max_length_padding() {
        let tok = train_tokenizer();
        let texts = &["the"];
        let config = BatchConfig {
            max_length: Some(10),
            padding: PaddingStrategy::MaxLength,
            truncation: TruncationStrategy::Right,
            pad_token_id: 0,
        };
        let batch = batch_encode(texts, &tok, &config);
        assert_eq!(
            batch.input_ids[0].len(),
            10,
            "should be padded to max_length"
        );
    }

    #[test]
    fn test_empty_input() {
        let tok = train_tokenizer();
        let texts: &[&str] = &[];
        let config = BatchConfig::default();
        let batch = batch_encode(texts, &tok, &config);
        assert_eq!(batch.batch_size(), 0);
    }

    #[test]
    fn test_empty_string_in_batch() {
        let tok = train_tokenizer();
        let texts = &["", "the cat"];
        let config = BatchConfig {
            padding: PaddingStrategy::LongestInBatch,
            pad_token_id: 0,
            ..Default::default()
        };
        let batch = batch_encode(texts, &tok, &config);
        assert_eq!(batch.batch_size(), 2);
        // Empty string should produce length 0
        assert_eq!(batch.lengths[0], 0);
    }

    #[test]
    fn test_left_padding() {
        let tok = train_tokenizer();
        let texts = &["the", "the cat sat"];
        let config = BatchConfigExt {
            base: BatchConfig {
                padding: PaddingStrategy::LongestInBatch,
                pad_token_id: 0,
                ..Default::default()
            },
            padding_side: PaddingSide::Left,
        };
        let batch = batch_encode_ext(texts, &tok, &config);

        // For the shorter sequence, padding should be at the beginning
        let shorter_len = batch.lengths[0];
        let total_len = batch.input_ids[0].len();
        if shorter_len < total_len {
            let pad_count = total_len - shorter_len;
            for i in 0..pad_count {
                assert_eq!(
                    batch.attention_mask[0][i], 0,
                    "left padding mask at {i} should be 0"
                );
                assert_eq!(
                    batch.input_ids[0][i], 0,
                    "left padding token at {i} should be pad_id"
                );
            }
            for i in pad_count..total_len {
                assert_eq!(
                    batch.attention_mask[0][i], 1,
                    "real token mask at {i} should be 1"
                );
            }
        }
    }

    #[test]
    fn test_total_real_tokens() {
        let tok = train_tokenizer();
        let texts = &["the cat", "the"];
        let config = BatchConfig {
            padding: PaddingStrategy::LongestInBatch,
            pad_token_id: 0,
            ..Default::default()
        };
        let batch = batch_encode(texts, &tok, &config);
        let total = batch.total_real_tokens();
        let expected: usize = batch.lengths.iter().sum();
        assert_eq!(
            total, expected,
            "total real tokens should equal sum of lengths"
        );
    }

    #[test]
    fn test_truncation_with_padding() {
        let tok = train_tokenizer();
        let texts = &["the cat sat on the mat", "the"];
        let config = BatchConfig {
            max_length: Some(4),
            padding: PaddingStrategy::MaxLength,
            truncation: TruncationStrategy::Right,
            pad_token_id: 0,
        };
        let batch = batch_encode(texts, &tok, &config);
        // All sequences should be exactly max_length
        for seq in &batch.input_ids {
            assert_eq!(seq.len(), 4, "all sequences should be padded to max_length");
        }
    }
}
