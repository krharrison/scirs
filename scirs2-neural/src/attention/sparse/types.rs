//! Types for sparse attention patterns (BigBird / Longformer-style).
//!
//! This module defines the enumerations and configuration structs that govern
//! which attention patterns are used when operating on long sequences.

use std::fmt;

/// Sparse-attention pattern variants.
///
/// Controls which query–key pairs are evaluated during attention.  All
/// patterns give sub-quadratic complexity for long sequences.
///
/// ```rust
/// use scirs2_neural::attention::sparse::SparsePattern;
/// let p = SparsePattern::LocalWindow;
/// assert_eq!(p, SparsePattern::LocalWindow);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparsePattern {
    /// Sliding-window local attention: each token attends to the `window_size`
    /// tokens immediately before and after it (Longformer local attention).
    LocalWindow,

    /// Global + local attention: a set of designated *global* tokens attend to
    /// all positions and all positions attend to global tokens; non-global
    /// tokens additionally use a local window (Longformer full model).
    GlobalLocal,

    /// Uniform-random attention pattern: each token attends to `n_random`
    /// randomly chosen positions in addition to its local window.  Used in the
    /// BigBird architecture.
    Random,

    /// Block-sparse attention (BigBird): the sequence is divided into blocks of
    /// size `block_size` and each block attends to: itself, `n_random_blocks`
    /// randomly chosen blocks, and any designated global tokens.
    BlockSparse,

    /// Alias for [`LocalWindow`](SparsePattern::LocalWindow) — a sliding window
    /// with no global tokens.
    Sliding,
}

impl fmt::Display for SparsePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SparsePattern::LocalWindow => write!(f, "LocalWindow"),
            SparsePattern::GlobalLocal => write!(f, "GlobalLocal"),
            SparsePattern::Random => write!(f, "Random"),
            SparsePattern::BlockSparse => write!(f, "BlockSparse"),
            SparsePattern::Sliding => write!(f, "Sliding"),
        }
    }
}

/// Configuration for sparse attention.
///
/// # Defaults
///
/// ```rust
/// use scirs2_neural::attention::sparse::{SparseAttentionConfig, SparsePattern};
/// let cfg = SparseAttentionConfig::default();
/// assert_eq!(cfg.pattern, SparsePattern::LocalWindow);
/// assert_eq!(cfg.window_size, 64);
/// assert_eq!(cfg.n_global_tokens, 0);
/// assert_eq!(cfg.n_random, 3);
/// assert_eq!(cfg.block_size, 64);
/// assert_eq!(cfg.n_heads, 8);
/// assert_eq!(cfg.head_dim, 64);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SparseAttentionConfig {
    /// Which attention pattern to use.
    pub pattern: SparsePattern,

    /// Half-width of the local attention window (tokens on each side).
    ///
    /// Token `i` attends to positions `max(0, i−w)..=min(n−1, i+w)`.
    pub window_size: usize,

    /// Number of *global* token slots prepended to every sequence.
    ///
    /// Global tokens attend to **all** positions and all positions attend back
    /// to them (used for CLS tokens, task prefixes, etc.).
    pub n_global_tokens: usize,

    /// Number of uniformly-random extra positions each token attends to
    /// (BigBird random component).
    pub n_random: usize,

    /// Block size for BigBird block-sparse patterns.
    ///
    /// The sequence is partitioned into non-overlapping blocks of this size.
    pub block_size: usize,

    /// Number of attention heads.
    pub n_heads: usize,

    /// Dimensionality of each attention head.
    pub head_dim: usize,
}

impl Default for SparseAttentionConfig {
    fn default() -> Self {
        Self {
            pattern: SparsePattern::LocalWindow,
            window_size: 64,
            n_global_tokens: 0,
            n_random: 3,
            block_size: 64,
            n_heads: 8,
            head_dim: 64,
        }
    }
}

impl fmt::Display for SparseAttentionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SparseAttentionConfig(pattern={}, window={}, n_global={}, n_random={}, block={}, heads={}, head_dim={})",
            self.pattern,
            self.window_size,
            self.n_global_tokens,
            self.n_random,
            self.block_size,
            self.n_heads,
            self.head_dim,
        )
    }
}

/// Precomputed sparse-attention mask.
///
/// `attend_to[i]` is a sorted, deduplicated list of position indices that
/// query `i` is allowed to attend to.
#[derive(Debug, Clone)]
pub struct SparseAttentionMask {
    /// Length of the sequence this mask was built for.
    pub seq_len: usize,

    /// For each query position `i`, the set of key positions `i` attends to.
    ///
    /// Entries are sorted in ascending order and deduplicated.
    pub attend_to: Vec<Vec<usize>>,
}

impl SparseAttentionMask {
    /// Total number of attended pairs across all query positions.
    pub fn n_pairs(&self) -> usize {
        self.attend_to.iter().map(|v| v.len()).sum()
    }

    /// Fraction of attended pairs relative to the fully dense case `seq_len²`.
    pub fn density(&self) -> f64 {
        if self.seq_len == 0 {
            return 0.0;
        }
        self.n_pairs() as f64 / (self.seq_len * self.seq_len) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_attention_config_default() {
        let cfg = SparseAttentionConfig::default();
        assert_eq!(cfg.pattern, SparsePattern::LocalWindow);
        assert_eq!(cfg.window_size, 64);
        assert_eq!(cfg.n_global_tokens, 0);
        assert_eq!(cfg.n_random, 3);
        assert_eq!(cfg.block_size, 64);
        assert_eq!(cfg.n_heads, 8);
        assert_eq!(cfg.head_dim, 64);
    }

    #[test]
    fn sparse_pattern_display() {
        assert_eq!(SparsePattern::LocalWindow.to_string(), "LocalWindow");
        assert_eq!(SparsePattern::BlockSparse.to_string(), "BlockSparse");
    }

    #[test]
    fn mask_density_empty() {
        let mask = SparseAttentionMask {
            seq_len: 0,
            attend_to: Vec::new(),
        };
        assert!((mask.density() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn mask_density_full() {
        // Every position attends to every other: density == 1.
        let seq_len = 4;
        let attend_to = vec![vec![0, 1, 2, 3]; seq_len];
        let mask = SparseAttentionMask { seq_len, attend_to };
        assert!((mask.density() - 1.0).abs() < 1e-10);
    }
}
