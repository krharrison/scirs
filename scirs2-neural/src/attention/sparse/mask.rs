//! Sparse-attention mask generation.
//!
//! Constructs [`SparseAttentionMask`] instances for the patterns defined in
//! [`super::types::SparsePattern`].  All patterns produce a mask as a
//! `Vec<Vec<usize>>` where entry `i` contains the (sorted, deduplicated) key
//! positions that query `i` is allowed to attend to.

use super::types::{SparseAttentionConfig, SparseAttentionMask, SparsePattern};

/// Builder for sparse attention masks.
///
/// Construct via [`AttentionMaskBuilder::new`], then call
/// [`AttentionMaskBuilder::build`] with the desired sequence length and global
/// token indices.
///
/// ```rust
/// use scirs2_neural::attention::sparse::{
///     AttentionMaskBuilder, SparseAttentionConfig, SparsePattern,
/// };
///
/// let mut cfg = SparseAttentionConfig::default();
/// cfg.pattern = SparsePattern::LocalWindow;
/// cfg.window_size = 2;
/// let builder = AttentionMaskBuilder::new(cfg);
/// let mask = builder.build(8, &[]);
///
/// // Token 0 with window=2 attends to tokens 0,1,2.
/// assert!(mask.attend_to[0].contains(&0));
/// assert!(mask.attend_to[0].contains(&1));
/// assert!(mask.attend_to[0].contains(&2));
/// // Token 0 should NOT attend to token 3 with window=2.
/// assert!(!mask.attend_to[0].contains(&3));
/// ```
pub struct AttentionMaskBuilder {
    config: SparseAttentionConfig,
}

impl AttentionMaskBuilder {
    /// Create a builder with the given configuration.
    pub fn new(config: SparseAttentionConfig) -> Self {
        Self { config }
    }

    /// Build a sparse attention mask for a sequence of length `seq_len`.
    ///
    /// `global_indices` specifies additional positions (0-based) that should
    /// act as global tokens, attending to all positions and receiving attention
    /// from all positions.  This parameter is ignored for patterns that have no
    /// global component (e.g., [`SparsePattern::LocalWindow`]).
    pub fn build(&self, seq_len: usize, global_indices: &[usize]) -> SparseAttentionMask {
        if seq_len == 0 {
            return SparseAttentionMask {
                seq_len: 0,
                attend_to: Vec::new(),
            };
        }

        let mut attend_to = match self.config.pattern {
            SparsePattern::LocalWindow | SparsePattern::Sliding => {
                Self::local_window_pattern(seq_len, self.config.window_size)
            }
            SparsePattern::GlobalLocal => {
                let mut pat = Self::local_window_pattern(seq_len, self.config.window_size);
                // Wire in the global tokens (from config.n_global_tokens).
                let cfg_globals: Vec<usize> =
                    (0..self.config.n_global_tokens.min(seq_len)).collect();
                Self::add_global(&mut pat, &cfg_globals, seq_len);
                Self::add_global(&mut pat, global_indices, seq_len);
                pat
            }
            SparsePattern::Random => {
                let mut pat = Self::local_window_pattern(seq_len, self.config.window_size);
                Self::add_random(
                    &mut pat,
                    self.config.n_random,
                    seq_len,
                    0xDEAD_BEEF_1234_5678,
                );
                pat
            }
            SparsePattern::BlockSparse => {
                let mut pat = Self::block_sparse_pattern(
                    seq_len,
                    self.config.block_size,
                    self.config.n_random,
                );
                let cfg_globals: Vec<usize> =
                    (0..self.config.n_global_tokens.min(seq_len)).collect();
                Self::add_global(&mut pat, &cfg_globals, seq_len);
                pat
            }
        };

        // Always wire in caller-supplied global indices.
        if !global_indices.is_empty() && self.config.pattern != SparsePattern::GlobalLocal {
            Self::add_global(&mut attend_to, global_indices, seq_len);
        }

        // Sort and deduplicate every row.
        for row in &mut attend_to {
            row.sort_unstable();
            row.dedup();
        }

        SparseAttentionMask { seq_len, attend_to }
    }

    // -----------------------------------------------------------------------
    // Pattern builders
    // -----------------------------------------------------------------------

    /// Local-window pattern: token `i` attends to `[max(0,i-w)..=min(n-1,i+w)]`.
    fn local_window_pattern(seq_len: usize, window: usize) -> Vec<Vec<usize>> {
        (0..seq_len)
            .map(|i| {
                let lo = i.saturating_sub(window);
                let hi = (i + window).min(seq_len - 1);
                (lo..=hi).collect()
            })
            .collect()
    }

    /// Add global tokens to an existing pattern.
    ///
    /// For each index in `global_indices`:
    /// * that position attends to **all** other positions, and
    /// * every other position also attends to that global position.
    fn add_global(pattern: &mut [Vec<usize>], global_indices: &[usize], seq_len: usize) {
        for &g in global_indices {
            if g >= seq_len {
                continue;
            }
            // The global token attends to everything.
            let all: Vec<usize> = (0..seq_len).collect();
            pattern[g] = all;

            // Everyone else attends back to the global token.
            for (i, row) in pattern.iter_mut().enumerate() {
                if i != g {
                    row.push(g);
                }
            }
        }
    }

    /// Add `n_random` uniformly-random additional positions to each token's
    /// attention set.  `seed` initialises the PRNG.
    fn add_random(pattern: &mut [Vec<usize>], n_random: usize, seq_len: usize, seed: u64) {
        if n_random == 0 || seq_len == 0 {
            return;
        }
        let mut rng = Xorshift64::new(seed);
        for row in pattern.iter_mut().take(seq_len) {
            for _ in 0..n_random {
                let j = (rng.next_u64() as usize) % seq_len;
                row.push(j);
            }
        }
    }

    /// BigBird block-sparse pattern.
    ///
    /// Each block of size `block_size` attends to itself and
    /// `n_random_blocks` randomly chosen other blocks.
    fn block_sparse_pattern(
        seq_len: usize,
        block_size: usize,
        n_random_blocks: usize,
    ) -> Vec<Vec<usize>> {
        let bs = block_size.max(1);
        let n_blocks = seq_len.div_ceil(bs);
        let mut pattern: Vec<Vec<usize>> = vec![Vec::new(); seq_len];
        let mut rng = Xorshift64::new(0xC0FF_EEFA_CADE_1234);

        for b in 0..n_blocks {
            let b_start = b * bs;
            let b_end = (b_start + bs).min(seq_len);

            // Tokens in block `b` attend within their own block.
            for row in pattern.iter_mut().take(b_end).skip(b_start) {
                for j in b_start..b_end {
                    row.push(j);
                }
            }

            // Also attend to `n_random_blocks` random other blocks.
            let n_other = n_blocks.saturating_sub(1);
            let n_pick = n_random_blocks.min(n_other);
            if n_pick > 0 {
                for _ in 0..n_pick {
                    let other_b = (rng.next_u64() as usize) % n_blocks;
                    let o_start = other_b * bs;
                    let o_end = (o_start + bs).min(seq_len);
                    for row in pattern.iter_mut().take(b_end).skip(b_start) {
                        for j in o_start..o_end {
                            row.push(j);
                        }
                    }
                }
            }
        }

        pattern
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Fraction of non-attended pairs (1 − density).
    ///
    /// A value of 1.0 means no attention is computed at all; a value close to
    /// 0.0 means the pattern is close to full dense attention.
    pub fn sparsity_ratio(&self, mask: &SparseAttentionMask) -> f64 {
        1.0 - mask.density()
    }

    /// Average number of key positions each query attends to.
    pub fn average_attend_count(&self, mask: &SparseAttentionMask) -> f64 {
        if mask.seq_len == 0 {
            return 0.0;
        }
        mask.n_pairs() as f64 / mask.seq_len as f64
    }
}

// ---------------------------------------------------------------------------
// Minimal PRNG (no external dependency)
// ---------------------------------------------------------------------------

struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x1234_5678_ABCD_EF01
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::sparse::SparseAttentionConfig;
    use crate::attention::sparse::SparsePattern;

    fn make_builder(pattern: SparsePattern, window: usize) -> AttentionMaskBuilder {
        let cfg = SparseAttentionConfig {
            pattern,
            window_size: window,
            ..Default::default()
        };
        AttentionMaskBuilder::new(cfg)
    }

    // ---- LocalWindow -------------------------------------------------------

    #[test]
    fn local_window_attend_count() {
        // With window=2 and seq_len=8:
        // Token 0 attends to 0,1,2 → 3 tokens.
        // Token 4 attends to 2,3,4,5,6 → 5 tokens.
        let builder = make_builder(SparsePattern::LocalWindow, 2);
        let mask = builder.build(8, &[]);

        assert_eq!(mask.attend_to[0].len(), 3); // max(0,0-2)=0, 0+2=2 → [0,1,2]
        assert_eq!(mask.attend_to[4].len(), 5); // [2,3,4,5,6]
                                                // Edge: token 7, window=2: [5,6,7] → 3 tokens.
        assert_eq!(mask.attend_to[7].len(), 3);
    }

    #[test]
    fn local_window_sorted_no_dups() {
        let builder = make_builder(SparsePattern::LocalWindow, 3);
        let mask = builder.build(10, &[]);
        for row in &mask.attend_to {
            let sorted_copy = {
                let mut v = row.clone();
                v.sort_unstable();
                v.dedup();
                v
            };
            assert_eq!(row, &sorted_copy, "row not sorted/deduplicated");
        }
    }

    #[test]
    fn sliding_same_as_local_window() {
        let cfg_lw = SparseAttentionConfig {
            pattern: SparsePattern::LocalWindow,
            window_size: 3,
            ..Default::default()
        };
        let cfg_sl = SparseAttentionConfig {
            pattern: SparsePattern::Sliding,
            window_size: 3,
            ..Default::default()
        };
        let m1 = AttentionMaskBuilder::new(cfg_lw).build(10, &[]);
        let m2 = AttentionMaskBuilder::new(cfg_sl).build(10, &[]);
        assert_eq!(m1.attend_to, m2.attend_to);
    }

    // ---- Global tokens -----------------------------------------------------

    #[test]
    fn global_token_attends_to_all() {
        let cfg = SparseAttentionConfig {
            pattern: SparsePattern::LocalWindow,
            window_size: 1,
            ..Default::default()
        };
        let builder = AttentionMaskBuilder::new(cfg);
        let seq_len = 6;
        let mask = builder.build(seq_len, &[0]); // token 0 is global

        // Token 0 should attend to all positions.
        assert_eq!(mask.attend_to[0].len(), seq_len);
        for i in 0..seq_len {
            assert!(
                mask.attend_to[0].contains(&i),
                "global token 0 should attend to position {i}"
            );
        }
    }

    #[test]
    fn all_tokens_attend_to_global() {
        let cfg = SparseAttentionConfig {
            pattern: SparsePattern::LocalWindow,
            window_size: 1,
            ..Default::default()
        };
        let builder = AttentionMaskBuilder::new(cfg);
        let seq_len = 6;
        let mask = builder.build(seq_len, &[0]); // token 0 is global

        for i in 1..seq_len {
            assert!(
                mask.attend_to[i].contains(&0),
                "token {i} should attend to global token 0"
            );
        }
    }

    // ---- Random attention --------------------------------------------------

    #[test]
    fn random_pattern_adds_extra_positions() {
        let base_builder = make_builder(SparsePattern::LocalWindow, 1);
        let base_mask = base_builder.build(16, &[]);
        let base_count: usize = base_mask.attend_to.iter().map(|v| v.len()).sum();

        let cfg_rand = SparseAttentionConfig {
            pattern: SparsePattern::Random,
            window_size: 1,
            n_random: 3,
            ..Default::default()
        };
        let rand_mask = AttentionMaskBuilder::new(cfg_rand).build(16, &[]);
        let rand_count: usize = rand_mask.attend_to.iter().map(|v| v.len()).sum();

        // Random should add (possibly deduplicated) extra attention pairs.
        assert!(rand_count >= base_count, "random should not reduce pairs");
    }

    // ---- Sparsity / density ------------------------------------------------

    #[test]
    fn sparsity_ratio_local_window_less_than_one() {
        let cfg = SparseAttentionConfig {
            pattern: SparsePattern::LocalWindow,
            window_size: 4,
            ..Default::default()
        };
        let builder = AttentionMaskBuilder::new(cfg);
        let mask = builder.build(256, &[]);
        let sparsity = builder.sparsity_ratio(&mask);
        // For seq=256 and window=4, density ≈ (2*4+1)/256 ≈ 0.035 → sparsity ≈ 0.965
        assert!(sparsity > 0.9, "sparsity={sparsity:.4} should be > 0.9");
    }

    #[test]
    fn average_attend_count_matches_window() {
        // Window=2 on seq=10: interior tokens attend to 5; edge tokens to 3–4.
        let cfg = SparseAttentionConfig {
            pattern: SparsePattern::LocalWindow,
            window_size: 2,
            ..Default::default()
        };
        let builder = AttentionMaskBuilder::new(cfg);
        let mask = builder.build(10, &[]);
        let avg = builder.average_attend_count(&mask);
        // The average should be between 3 and 5.
        assert!((3.0..=5.0).contains(&avg), "avg={avg:.2}");
    }

    // ---- BlockSparse -------------------------------------------------------

    #[test]
    fn block_sparse_within_block_attended() {
        let cfg = SparseAttentionConfig {
            pattern: SparsePattern::BlockSparse,
            block_size: 4,
            n_random: 0, // no random blocks for determinism
            ..Default::default()
        };
        let builder = AttentionMaskBuilder::new(cfg);
        let mask = builder.build(8, &[]);

        // Token 0 (block 0) must attend to tokens 1, 2, 3 (also in block 0).
        for j in 1..4_usize {
            assert!(
                mask.attend_to[0].contains(&j),
                "token 0 should attend to token {j} in same block"
            );
        }
        // Token 4 (block 1) must attend to tokens 5, 6, 7.
        for j in 5..8_usize {
            assert!(
                mask.attend_to[4].contains(&j),
                "token 4 should attend to token {j} in same block"
            );
        }
    }

    // ---- Edge cases --------------------------------------------------------

    #[test]
    fn empty_sequence_produces_empty_mask() {
        let builder = make_builder(SparsePattern::LocalWindow, 3);
        let mask = builder.build(0, &[]);
        assert_eq!(mask.seq_len, 0);
        assert!(mask.attend_to.is_empty());
    }

    #[test]
    fn single_token_attends_to_itself() {
        let builder = make_builder(SparsePattern::LocalWindow, 5);
        let mask = builder.build(1, &[]);
        assert_eq!(mask.attend_to[0], vec![0]);
    }
}
