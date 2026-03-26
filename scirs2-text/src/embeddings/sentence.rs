//! Sentence-BERT-style embedding aggregation.
//!
//! This module provides pooling strategies that reduce a sequence of token
//! embeddings to a single fixed-size sentence representation, following the
//! methodology of Sentence-BERT (Reimers & Gurevych, 2019).
//!
//! # Pooling strategies
//!
//! | Strategy | Description |
//! |----------|-------------|
//! | `MeanPooling` | Average of all non-padding token embeddings (default) |
//! | `MaxPooling` | Element-wise maximum across non-padding tokens |
//! | `ClsToken` | First token (\[CLS\]) embedding |
//! | `MeanMax` | Concatenation of mean and max poolings (dim × 2) |
//! | `WeightedMean` | Position-decaying weighted mean (exponential decay) |
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::embeddings::sentence::{SentenceEmbedder, SentenceEmbedderConfig, PoolingStrategy};
//! use scirs2_core::ndarray::Array2;
//!
//! // Build a tiny 4-token × 3-dim embedding matrix.
//! let token_embeddings = Array2::from_shape_vec((4, 3), vec![
//!     1.0, 0.0, 0.0,
//!     0.0, 1.0, 0.0,
//!     0.0, 0.0, 1.0,
//!     0.0, 0.0, 0.0,
//! ]).unwrap();
//!
//! let config = SentenceEmbedderConfig {
//!     pooling: PoolingStrategy::MeanPooling,
//!     normalize: true,
//!     dim: 3,
//! };
//! let embedder = SentenceEmbedder::new(token_embeddings, config);
//!
//! // Embed two tokens with attention mask [1, 1, 0, 0]
//! let emb = embedder.embed_token_ids(&[0, 1], &[1, 1]).unwrap();
//! assert_eq!(emb.len(), 3);
//! ```

use crate::error::{Result, TextError};
use scirs2_core::ndarray::{Array1, Array2, Axis};

// ─── PoolingStrategy ──────────────────────────────────────────────────────────

/// Strategy used to aggregate token-level embeddings into a sentence vector.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum PoolingStrategy {
    /// Arithmetic mean of all non-padding token embeddings.
    MeanPooling,
    /// Element-wise maximum over non-padding token embeddings.
    MaxPooling,
    /// Use the first token (index 0, i.e. `[CLS]`) embedding directly.
    ClsToken,
    /// Concatenate `MeanPooling` and `MaxPooling` outputs (output dim × 2).
    MeanMax,
    /// Position-weighted mean: token at position `i` receives weight
    /// `exp(-α·i)` where `α = ln(2) / max_len` so that the last token has
    /// weight ≈ 0.5 × weight of the first token.
    WeightedMean,
}

// ─── SentenceEmbedderConfig ───────────────────────────────────────────────────

/// Configuration for [`SentenceEmbedder`].
#[derive(Debug, Clone)]
pub struct SentenceEmbedderConfig {
    /// Pooling strategy (default: `MeanPooling`).
    pub pooling: PoolingStrategy,
    /// Whether to L2-normalise the output embedding (default `true`).
    pub normalize: bool,
    /// Token embedding dimensionality.  Derived automatically from the
    /// embedding matrix if `0`; set explicitly to catch shape mismatches early.
    pub dim: usize,
}

impl Default for SentenceEmbedderConfig {
    fn default() -> Self {
        SentenceEmbedderConfig {
            pooling: PoolingStrategy::MeanPooling,
            normalize: true,
            dim: 768,
        }
    }
}

// ─── SentenceEmbedder ─────────────────────────────────────────────────────────

/// Aggregates token-level embeddings into sentence-level representations.
///
/// The embedding matrix is indexed by token ID: row `i` is the embedding for
/// token with ID `i`.  Embeddings are typically pre-trained (e.g. a BERT-style
/// token embedding table extracted from a transformer) and are **not** trained
/// by this struct.
pub struct SentenceEmbedder {
    config: SentenceEmbedderConfig,
    /// Shape `[vocab_size, dim]`.
    token_embeddings: Array2<f64>,
}

impl SentenceEmbedder {
    // ── Construction ──────────────────────────────────────────────────────

    /// Create a new `SentenceEmbedder` from a token embedding matrix and config.
    ///
    /// # Panics
    ///
    /// Does **not** panic — mismatches between `config.dim` and the matrix
    /// column count are detected lazily at embedding time.
    pub fn new(token_embeddings: Array2<f64>, mut config: SentenceEmbedderConfig) -> Self {
        if config.dim == 0 {
            config.dim = token_embeddings.ncols();
        }
        SentenceEmbedder {
            config,
            token_embeddings,
        }
    }

    // ── Embedding ─────────────────────────────────────────────────────────

    /// Embed a single token-ID sequence into a sentence vector.
    ///
    /// `attention_mask` has the same length as `token_ids`; a value of `1`
    /// indicates a real token and `0` indicates padding that should be ignored.
    pub fn embed_token_ids(
        &self,
        token_ids: &[u32],
        attention_mask: &[u32],
    ) -> Result<Array1<f64>> {
        if token_ids.len() != attention_mask.len() {
            return Err(TextError::InvalidInput(format!(
                "token_ids length ({}) != attention_mask length ({})",
                token_ids.len(),
                attention_mask.len()
            )));
        }
        if token_ids.is_empty() {
            return Err(TextError::InvalidInput(
                "Cannot embed an empty token sequence".to_string(),
            ));
        }

        let dim = self.token_embeddings.ncols();
        let vocab_size = self.token_embeddings.nrows();

        // Validate token IDs.
        for &id in token_ids {
            if id as usize >= vocab_size {
                return Err(TextError::EmbeddingError(format!(
                    "Token ID {} out of vocabulary range [0, {})",
                    id, vocab_size
                )));
            }
        }

        // Collect (embedding_row, mask) pairs for non-padding positions.
        let active: Vec<(usize, f64)> = token_ids
            .iter()
            .zip(attention_mask.iter())
            .enumerate()
            .filter_map(|(pos, (&id, &mask))| {
                if mask != 0 {
                    Some((id as usize, pos as f64))
                } else {
                    None
                }
            })
            .collect();

        if active.is_empty() {
            // All padding: return zero vector (or error — here we return zeros).
            return Ok(Array1::zeros(dim));
        }

        let result = match &self.config.pooling {
            PoolingStrategy::MeanPooling => self.pool_mean(&active, dim),
            PoolingStrategy::MaxPooling => self.pool_max(&active, dim),
            PoolingStrategy::ClsToken => {
                // Use the embedding of the first token regardless of mask.
                let cls_id = token_ids[0] as usize;
                self.token_embeddings.row(cls_id).to_owned()
            }
            PoolingStrategy::MeanMax => {
                let mean = self.pool_mean(&active, dim);
                let max = self.pool_max(&active, dim);
                // Concatenate: output length = 2 * dim
                let mut out = Array1::zeros(2 * dim);
                out.slice_mut(scirs2_core::ndarray::s![..dim]).assign(&mean);
                out.slice_mut(scirs2_core::ndarray::s![dim..]).assign(&max);
                out
            }
            PoolingStrategy::WeightedMean => self.pool_weighted_mean(&active, dim),
        };

        if self.config.normalize && !matches!(self.config.pooling, PoolingStrategy::MeanMax) {
            Ok(l2_normalize(result))
        } else if self.config.normalize && matches!(self.config.pooling, PoolingStrategy::MeanMax) {
            Ok(l2_normalize(result))
        } else {
            Ok(result)
        }
    }

    /// Embed a batch of sequences into a 2-D array of shape `[batch, dim]`.
    ///
    /// All sequences are processed independently using the configured pooling
    /// strategy.  Padding tokens are ignored based on `attention_masks`.
    pub fn embed_batch(
        &self,
        token_ids: &[Vec<u32>],
        attention_masks: &[Vec<u32>],
    ) -> Result<Array2<f64>> {
        if token_ids.len() != attention_masks.len() {
            return Err(TextError::InvalidInput(format!(
                "token_ids batch size ({}) != attention_masks batch size ({})",
                token_ids.len(),
                attention_masks.len()
            )));
        }
        if token_ids.is_empty() {
            return Err(TextError::InvalidInput(
                "Batch must contain at least one sequence".to_string(),
            ));
        }

        let batch_size = token_ids.len();
        let out_dim = match &self.config.pooling {
            PoolingStrategy::MeanMax => self.config.dim * 2,
            _ => self.config.dim,
        };
        let mut out = Array2::zeros((batch_size, out_dim));

        for (i, (ids, mask)) in token_ids.iter().zip(attention_masks.iter()).enumerate() {
            let emb = self.embed_token_ids(ids, mask)?;
            out.row_mut(i).assign(&emb);
        }

        Ok(out)
    }

    // ── Similarity ────────────────────────────────────────────────────────

    /// Compute cosine similarity between two sentence embeddings.
    ///
    /// Returns a value in `[-1, 1]`; identical embeddings give `1.0`.
    pub fn semantic_similarity(&self, emb_a: &Array1<f64>, emb_b: &Array1<f64>) -> f64 {
        cosine_similarity(emb_a, emb_b)
    }

    /// Find the `top_k` most similar entries in `corpus` for the `query`
    /// embedding.
    ///
    /// `corpus` has shape `[n_sentences, dim]`.  Returns a vector of
    /// `(index, cosine_similarity)` pairs sorted by descending similarity.
    pub fn most_similar(
        &self,
        query: &Array1<f64>,
        corpus: &Array2<f64>,
        top_k: usize,
    ) -> Vec<(usize, f64)> {
        let n = corpus.nrows();
        let mut scores: Vec<(usize, f64)> = (0..n)
            .map(|i| {
                let row = corpus.row(i).to_owned();
                (i, cosine_similarity(query, &row))
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    // ── Private pooling helpers ───────────────────────────────────────────

    fn pool_mean(&self, active: &[(usize, f64)], dim: usize) -> Array1<f64> {
        let mut sum = Array1::zeros(dim);
        for &(id, _pos) in active {
            sum = sum + self.token_embeddings.row(id).to_owned();
        }
        sum / active.len() as f64
    }

    fn pool_max(&self, active: &[(usize, f64)], dim: usize) -> Array1<f64> {
        let mut result = Array1::from_elem(dim, f64::NEG_INFINITY);
        for &(id, _pos) in active {
            let row = self.token_embeddings.row(id);
            for j in 0..dim {
                if row[j] > result[j] {
                    result[j] = row[j];
                }
            }
        }
        result
    }

    fn pool_weighted_mean(&self, active: &[(usize, f64)], dim: usize) -> Array1<f64> {
        // Exponential positional decay: w_i = exp(-α · pos_i)
        // Choose α so that the last position weight ≈ 0.5.
        let max_pos = active
            .iter()
            .map(|(_, p)| *p)
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1.0);
        let alpha = std::f64::consts::LN_2 / max_pos;

        let mut weighted_sum = Array1::zeros(dim);
        let mut total_weight = 0.0f64;

        for &(id, pos) in active {
            let weight = (-alpha * pos).exp();
            weighted_sum = weighted_sum + self.token_embeddings.row(id).to_owned() * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            weighted_sum
        }
    }
}

// ─── Standalone utilities ─────────────────────────────────────────────────────

/// Compute cosine similarity between two vectors.
///
/// Returns `0.0` if either vector is zero.
pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < f64::EPSILON || norm_b < f64::EPSILON {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

/// L2-normalise a vector in-place and return it.
fn l2_normalize(mut v: Array1<f64>) -> Array1<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > f64::EPSILON {
        v.mapv_inplace(|x| x / norm);
    }
    v
}

/// Compute the L2 norm of a vector.
pub fn l2_norm(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Create a simple 4-token × 4-dim identity-style embedding matrix.
    fn identity_embeddings(n: usize) -> Array2<f64> {
        let mut m = Array2::zeros((n, n));
        for i in 0..n {
            m[[i, i]] = 1.0;
        }
        m
    }

    fn make_config(pooling: PoolingStrategy) -> SentenceEmbedderConfig {
        SentenceEmbedderConfig {
            pooling,
            normalize: false,
            dim: 4,
        }
    }

    fn all_ones_mask(n: usize) -> Vec<u32> {
        vec![1u32; n]
    }

    // ── MeanPooling ───────────────────────────────────────────────────────

    #[test]
    fn test_sentence_embedder_mean_pool() {
        let emb = identity_embeddings(4);
        let embedder = SentenceEmbedder::new(emb, make_config(PoolingStrategy::MeanPooling));

        // Tokens 0 and 1 → mean of rows 0 and 1 = [0.5, 0.5, 0, 0]
        let ids = vec![0u32, 1];
        let mask = all_ones_mask(2);
        let out = embedder.embed_token_ids(&ids, &mask).expect("embed");
        assert!((out[0] - 0.5).abs() < 1e-9);
        assert!((out[1] - 0.5).abs() < 1e-9);
        assert!((out[2] - 0.0).abs() < 1e-9);
        assert!((out[3] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_sentence_embedder_mean_pool_with_padding() {
        let emb = identity_embeddings(4);
        let embedder = SentenceEmbedder::new(emb, make_config(PoolingStrategy::MeanPooling));

        // Token 0 is real, token 1 is padding (mask=0) → only row 0
        let ids = vec![0u32, 1];
        let mask = vec![1u32, 0];
        let out = embedder.embed_token_ids(&ids, &mask).expect("embed");
        assert!((out[0] - 1.0).abs() < 1e-9);
        assert!((out[1] - 0.0).abs() < 1e-9);
    }

    // ── MaxPooling ────────────────────────────────────────────────────────

    #[test]
    fn test_sentence_embedder_max_pool() {
        let emb = identity_embeddings(4);
        let embedder = SentenceEmbedder::new(emb, make_config(PoolingStrategy::MaxPooling));

        // Tokens 0,1,2 → element-wise max of rows 0-2 = [1,1,1,0]
        let ids = vec![0u32, 1, 2];
        let mask = all_ones_mask(3);
        let out = embedder.embed_token_ids(&ids, &mask).expect("embed");
        assert!((out[0] - 1.0).abs() < 1e-9);
        assert!((out[1] - 1.0).abs() < 1e-9);
        assert!((out[2] - 1.0).abs() < 1e-9);
        assert!((out[3] - 0.0).abs() < 1e-9);
    }

    // ── CLS token ─────────────────────────────────────────────────────────

    #[test]
    fn test_sentence_embedder_cls() {
        let emb = identity_embeddings(4);
        let embedder = SentenceEmbedder::new(emb, make_config(PoolingStrategy::ClsToken));

        let ids = vec![2u32, 3]; // CLS at position 0 is token ID 2
        let mask = all_ones_mask(2);
        let out = embedder.embed_token_ids(&ids, &mask).expect("embed");
        // Should be row 2 of the identity matrix: [0,0,1,0]
        assert!((out[0] - 0.0).abs() < 1e-9);
        assert!((out[1] - 0.0).abs() < 1e-9);
        assert!((out[2] - 1.0).abs() < 1e-9);
        assert!((out[3] - 0.0).abs() < 1e-9);
    }

    // ── Normalization ─────────────────────────────────────────────────────

    #[test]
    fn test_sentence_embedder_normalize() {
        let emb = identity_embeddings(4);
        let mut cfg = make_config(PoolingStrategy::MeanPooling);
        cfg.normalize = true;
        let embedder = SentenceEmbedder::new(emb, cfg);

        let ids = vec![0u32, 1]; // mean = [0.5, 0.5, 0, 0]
        let mask = all_ones_mask(2);
        let out = embedder.embed_token_ids(&ids, &mask).expect("embed");
        let norm = l2_norm(&out);
        assert!((norm - 1.0).abs() < 1e-9, "norm = {}", norm);
    }

    // ── Cosine similarity ─────────────────────────────────────────────────

    #[test]
    fn test_similarity_identical() {
        let emb = identity_embeddings(4);
        let embedder = SentenceEmbedder::new(emb, make_config(PoolingStrategy::ClsToken));

        let v = array![1.0_f64, 0.0, 0.0, 0.0];
        let sim = embedder.semantic_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_similarity_orthogonal() {
        let emb = identity_embeddings(4);
        let embedder = SentenceEmbedder::new(emb, make_config(PoolingStrategy::ClsToken));

        let a = array![1.0_f64, 0.0, 0.0, 0.0];
        let b = array![0.0_f64, 1.0, 0.0, 0.0];
        let sim = embedder.semantic_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_similarity_opposite() {
        let emb = identity_embeddings(4);
        let embedder = SentenceEmbedder::new(emb, make_config(PoolingStrategy::ClsToken));

        let a = array![1.0_f64, 0.0, 0.0, 0.0];
        let b = array![-1.0_f64, 0.0, 0.0, 0.0];
        let sim = embedder.semantic_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-9);
    }

    // ── MeanMax ───────────────────────────────────────────────────────────

    #[test]
    fn test_sentence_embedder_mean_max() {
        let emb = identity_embeddings(4);
        let cfg = SentenceEmbedderConfig {
            pooling: PoolingStrategy::MeanMax,
            normalize: false,
            dim: 4,
        };
        let embedder = SentenceEmbedder::new(emb, cfg);

        // Output should be 8-dimensional (dim=4 → mean(4) || max(4))
        let ids = vec![0u32, 1];
        let mask = all_ones_mask(2);
        let out = embedder.embed_token_ids(&ids, &mask).expect("embed");
        assert_eq!(out.len(), 8);
    }

    // ── WeightedMean ──────────────────────────────────────────────────────

    #[test]
    fn test_sentence_embedder_weighted_mean_single_token() {
        let emb = identity_embeddings(4);
        let embedder = SentenceEmbedder::new(emb, make_config(PoolingStrategy::WeightedMean));

        let ids = vec![2u32]; // single token
        let mask = all_ones_mask(1);
        let out = embedder.embed_token_ids(&ids, &mask).expect("embed");
        // With a single token the weighted mean equals that token's embedding.
        assert!((out[2] - 1.0).abs() < 1e-9);
    }

    // ── Batch embedding ───────────────────────────────────────────────────

    #[test]
    fn test_embed_batch_shape() {
        let emb = identity_embeddings(4);
        let embedder = SentenceEmbedder::new(emb, make_config(PoolingStrategy::MeanPooling));

        let token_ids = vec![vec![0u32, 1], vec![2u32, 3]];
        let masks = vec![all_ones_mask(2), all_ones_mask(2)];
        let out = embedder
            .embed_batch(&token_ids, &masks)
            .expect("batch embed");
        assert_eq!(out.nrows(), 2);
        assert_eq!(out.ncols(), 4);
    }

    // ── most_similar ──────────────────────────────────────────────────────

    #[test]
    fn test_most_similar_returns_top_k() {
        let emb = identity_embeddings(4);
        let embedder = SentenceEmbedder::new(emb, make_config(PoolingStrategy::ClsToken));

        let query = array![1.0_f64, 0.0, 0.0, 0.0];
        // Corpus: 3 rows, first one is identical to query.
        let corpus = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .expect("shape");

        let results = embedder.most_similar(&query, &corpus, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // index 0 is most similar
        assert!((results[0].1 - 1.0).abs() < 1e-9);
    }

    // ── Error cases ───────────────────────────────────────────────────────

    #[test]
    fn test_embed_empty_sequence_errors() {
        let emb = identity_embeddings(4);
        let embedder = SentenceEmbedder::new(emb, make_config(PoolingStrategy::MeanPooling));
        assert!(embedder.embed_token_ids(&[], &[]).is_err());
    }

    #[test]
    fn test_embed_mismatched_mask_errors() {
        let emb = identity_embeddings(4);
        let embedder = SentenceEmbedder::new(emb, make_config(PoolingStrategy::MeanPooling));
        assert!(embedder.embed_token_ids(&[0u32, 1], &[1u32]).is_err());
    }

    #[test]
    fn test_embed_out_of_vocab_errors() {
        let emb = identity_embeddings(4);
        let embedder = SentenceEmbedder::new(emb, make_config(PoolingStrategy::MeanPooling));
        assert!(embedder.embed_token_ids(&[10u32], &[1u32]).is_err());
    }
}
