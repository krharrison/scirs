//! Sparse attention forward computation.
//!
//! Computes scaled dot-product attention over the positions specified by a
//! precomputed [`SparseAttentionMask`], avoiding the O(n²) full-attention
//! matrix for long sequences.
//!
//! ## Layout convention
//!
//! All Q / K / V tensors are stored as flat `f64` slices in **row-major** order
//! with logical shape `[seq_len, n_heads, head_dim]`.  Element at position
//! `(s, h, d)` lives at flat index `s * n_heads * head_dim + h * head_dim + d`.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_neural::attention::sparse::{
//!     SparseAttention, SparseAttentionConfig, SparsePattern,
//! };
//!
//! let mut cfg = SparseAttentionConfig::default();
//! cfg.pattern = SparsePattern::LocalWindow;
//! cfg.window_size = 2;
//! cfg.n_heads = 2;
//! cfg.head_dim = 4;
//! let seq_len = 6;
//! let n = seq_len * cfg.n_heads * cfg.head_dim;
//! let q: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
//! let k = q.clone();
//! let v = q.clone();
//!
//! let sa = SparseAttention::new(cfg);
//! let out = sa.forward(&q, &k, &v, seq_len, &[]);
//! assert_eq!(out.len(), n);
//! ```

use super::mask::AttentionMaskBuilder;
use super::types::{SparseAttentionConfig, SparseAttentionMask};

/// Sparse scaled dot-product attention operator.
pub struct SparseAttention {
    config: SparseAttentionConfig,
    mask_builder: AttentionMaskBuilder,
}

impl SparseAttention {
    /// Create a new operator with the given configuration.
    pub fn new(config: SparseAttentionConfig) -> Self {
        let mask_builder = AttentionMaskBuilder::new(config.clone());
        Self {
            config,
            mask_builder,
        }
    }

    /// Compute the sparse attention output.
    ///
    /// # Arguments
    ///
    /// * `q` — query tensor, flat `f64` slice with shape `[seq_len, n_heads, head_dim]`.
    /// * `k` — key tensor, same shape.
    /// * `v` — value tensor, same shape.
    /// * `seq_len` — number of sequence positions.
    /// * `global_indices` — additional positions acting as global tokens
    ///   (passed through to [`AttentionMaskBuilder::build`]).
    ///
    /// # Returns
    ///
    /// Output tensor with the same flat shape `[seq_len, n_heads, head_dim]`.
    pub fn forward(
        &self,
        q: &[f64],
        k: &[f64],
        v: &[f64],
        seq_len: usize,
        global_indices: &[usize],
    ) -> Vec<f64> {
        let n_heads = self.config.n_heads;
        let head_dim = self.config.head_dim;
        let total = seq_len * n_heads * head_dim;

        if total == 0 || q.len() < total || k.len() < total || v.len() < total {
            return vec![0.0; total];
        }

        let mask = self.mask_builder.build(seq_len, global_indices);
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut output = vec![0.0f64; total];

        for i in 0..seq_len {
            let allowed = &mask.attend_to[i];
            if allowed.is_empty() {
                continue;
            }

            for h in 0..n_heads {
                // Slice Q for query position i, head h.
                let q_offset = i * n_heads * head_dim + h * head_dim;
                let q_vec: &[f64] = &q[q_offset..q_offset + head_dim];

                // Gather K and V vectors for allowed positions.
                let k_positions: Vec<Vec<f64>> = allowed
                    .iter()
                    .map(|&j| {
                        let k_off = j * n_heads * head_dim + h * head_dim;
                        k[k_off..k_off + head_dim].to_vec()
                    })
                    .collect();
                let v_positions: Vec<Vec<f64>> = allowed
                    .iter()
                    .map(|&j| {
                        let v_off = j * n_heads * head_dim + h * head_dim;
                        v[v_off..v_off + head_dim].to_vec()
                    })
                    .collect();

                // Compute attention weights.
                let weights = Self::softmax_attend(q_vec, &k_positions, scale);

                // Weighted sum of value vectors.
                let mut attn_out = vec![0.0f64; head_dim];
                for (w, v_vec) in weights.iter().zip(v_positions.iter()) {
                    for d in 0..head_dim {
                        attn_out[d] += w * v_vec[d];
                    }
                }

                // Write to output.
                let out_offset = i * n_heads * head_dim + h * head_dim;
                output[out_offset..out_offset + head_dim].copy_from_slice(&attn_out);
            }
        }

        output
    }

    /// Compute the attention output for a single query position `i` over
    /// a set of allowed key/value positions (multi-head, concatenated).
    ///
    /// This is a higher-level wrapper that processes all heads at once.
    ///
    /// # Arguments
    ///
    /// * `q_i`          — query vector for position `i`, shape `[n_heads * head_dim]`.
    /// * `k_positions`  — list of key vectors at allowed positions, each `[n_heads * head_dim]`.
    /// * `v_positions`  — list of value vectors at allowed positions, same shape.
    /// * `scale`        — attention scale factor (`1/sqrt(head_dim)`).
    ///
    /// # Returns
    ///
    /// Output vector of shape `[n_heads * head_dim]`.
    pub fn attend_to_positions(
        q_i: &[f64],
        k_positions: &[Vec<f64>],
        v_positions: &[Vec<f64>],
        scale: f64,
    ) -> Vec<f64> {
        if k_positions.is_empty() || q_i.is_empty() {
            return vec![0.0; q_i.len()];
        }
        let n_heads_dim = q_i.len();
        // Compute attention weights over all allowed positions (treating the
        // full concatenated multi-head vector as one large head for simplicity
        // when called as a standalone helper).
        let weights = Self::softmax_attend(q_i, k_positions, scale);
        let mut out = vec![0.0f64; n_heads_dim];
        for (w, v_vec) in weights.iter().zip(v_positions.iter()) {
            let d = v_vec.len().min(n_heads_dim);
            for i in 0..d {
                out[i] += w * v_vec[i];
            }
        }
        out
    }

    /// Compute softmax attention weights for a single query over `keys`.
    ///
    /// # Arguments
    ///
    /// * `q`     — query vector `[D]`.
    /// * `keys`  — list of key vectors, each `[D]`.
    /// * `scale` — scalar multiplied by each dot product before softmax.
    ///
    /// # Returns
    ///
    /// Attention weight vector of length `keys.len()`.  Values are
    /// non-negative and sum to 1.0 (or 0.0 when `keys` is empty).
    pub fn softmax_attend(q: &[f64], keys: &[Vec<f64>], scale: f64) -> Vec<f64> {
        if keys.is_empty() {
            return Vec::new();
        }
        // Compute raw scores.
        let scores: Vec<f64> = keys
            .iter()
            .map(|k| {
                let dot: f64 = q.iter().zip(k.iter()).map(|(&qi, &ki)| qi * ki).sum();
                dot * scale
            })
            .collect();

        // Numerically stable softmax.
        let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum <= 0.0 || sum.is_nan() {
            let u = 1.0 / keys.len() as f64;
            return vec![u; keys.len()];
        }
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Borrow the configuration.
    pub fn config(&self) -> &SparseAttentionConfig {
        &self.config
    }

    /// Build the attention mask for the given sequence length (useful for
    /// inspecting the sparsity pattern without running the full forward pass).
    pub fn build_mask(&self, seq_len: usize, global_indices: &[usize]) -> SparseAttentionMask {
        self.mask_builder.build(seq_len, global_indices)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::sparse::{SparseAttentionConfig, SparsePattern};

    fn make_sa(
        pattern: SparsePattern,
        window: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> SparseAttention {
        let cfg = SparseAttentionConfig {
            pattern,
            window_size: window,
            n_heads,
            head_dim,
            ..Default::default()
        };
        SparseAttention::new(cfg)
    }

    fn zeros(seq_len: usize, n_heads: usize, head_dim: usize) -> Vec<f64> {
        vec![0.0; seq_len * n_heads * head_dim]
    }

    // ---- Output shape ------------------------------------------------------

    #[test]
    fn forward_output_shape_correct() {
        let sa = make_sa(SparsePattern::LocalWindow, 2, 4, 8);
        let seq_len = 10;
        let n = seq_len * 4 * 8;
        let q: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();
        let k = q.clone();
        let v = q.clone();
        let out = sa.forward(&q, &k, &v, seq_len, &[]);
        assert_eq!(out.len(), n, "output length should equal seq * heads * dim");
    }

    #[test]
    fn forward_empty_sequence_returns_empty() {
        let sa = make_sa(SparsePattern::LocalWindow, 2, 2, 4);
        let out = sa.forward(&[], &[], &[], 0, &[]);
        assert!(out.is_empty());
    }

    // ---- Attention weights -------------------------------------------------

    #[test]
    fn softmax_attend_weights_sum_to_one() {
        let q = vec![1.0, 0.0, -1.0];
        let keys = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let weights = SparseAttention::softmax_attend(&q, &keys, 1.0);
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "attention weights should sum to 1.0, got {sum}"
        );
        for &w in &weights {
            assert!(w >= 0.0, "weights must be non-negative");
        }
    }

    #[test]
    fn softmax_attend_no_keys_returns_empty() {
        let q = vec![1.0, 2.0];
        let weights = SparseAttention::softmax_attend(&q, &[], 1.0);
        assert!(weights.is_empty());
    }

    #[test]
    fn softmax_attend_single_key_weight_is_one() {
        let q = vec![0.5, -0.5];
        let keys = vec![vec![1.0, 0.5]];
        let weights = SparseAttention::softmax_attend(&q, &keys, 0.5);
        assert_eq!(weights.len(), 1);
        assert!((weights[0] - 1.0).abs() < 1e-10);
    }

    // ---- Attend to positions -----------------------------------------------

    #[test]
    fn attend_to_positions_output_length() {
        let dim = 6;
        let q = vec![0.1f64; dim];
        let k = vec![vec![0.2f64; dim], vec![0.3f64; dim]];
        let v = k.clone();
        let out = SparseAttention::attend_to_positions(&q, &k, &v, 1.0);
        assert_eq!(out.len(), dim);
    }

    // ---- Forward integration -----------------------------------------------

    #[test]
    fn forward_v_zeros_output_zeros() {
        let (seq_len, n_heads, head_dim) = (6, 2, 4);
        let sa = make_sa(SparsePattern::LocalWindow, 1, n_heads, head_dim);
        let n = seq_len * n_heads * head_dim;
        let q: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let k = q.clone();
        let v = zeros(seq_len, n_heads, head_dim);
        let out = sa.forward(&q, &k, &v, seq_len, &[]);
        for &x in &out {
            assert!(x.abs() < 1e-12, "V=0 → output should be 0, got {x}");
        }
    }

    #[test]
    fn forward_single_token_equal_to_v() {
        // Single query token: attention over window collapses to a weighted
        // sum; with Q==K==V and one allowed position, output == V[0].
        let (n_heads, head_dim) = (1, 4);
        let seq_len = 1;
        let sa = make_sa(SparsePattern::LocalWindow, 1, n_heads, head_dim);
        let v: Vec<f64> = (0..head_dim).map(|i| (i + 1) as f64).collect();
        let out = sa.forward(&v, &v, &v, seq_len, &[]);
        assert_eq!(out.len(), n_heads * head_dim);
        for (a, b) in out.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-9, "single token: out={a}, v={b}");
        }
    }

    // ---- Config / mask integration -----------------------------------------

    #[test]
    fn build_mask_local_window() {
        let sa = make_sa(SparsePattern::LocalWindow, 2, 2, 4);
        let mask = sa.build_mask(8, &[]);
        // Token 0 should attend to tokens 0,1,2 (window=2).
        assert_eq!(mask.attend_to[0], vec![0, 1, 2]);
    }

    #[test]
    fn sparse_attention_config_default_works() {
        let cfg = SparseAttentionConfig::default();
        let _sa = SparseAttention::new(cfg);
        // Constructing from default config should not panic.
    }
}
