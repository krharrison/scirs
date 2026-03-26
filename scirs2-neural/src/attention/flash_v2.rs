//! Flash Attention 2 — tiled online-softmax attention.
//!
//! Flash Attention 2 (Dao, 2023) improves on the original Flash Attention by
//! parallelising over the **query** dimension (rows of the attention matrix)
//! rather than only over batch and head dimensions.  The key algorithmic idea
//! is an *online softmax* that processes the attention matrix in tiles and
//! updates running statistics (`m` = running max, `l` = running sum of
//! exponentials) so that the final softmax is never materialised in full.
//!
//! ## Layout convention
//!
//! All Q / K / V slices are stored **flat** in *row-major* order with logical
//! shape `[seq_len, head_dim]` for a single head.  Multi-head batching is
//! handled by the caller.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_neural::attention::flash_v2::FlashAttentionV2;
//!
//! let seq_len = 8;
//! let head_dim = 4;
//! let n = seq_len * head_dim;
//!
//! // Simple arithmetic tensors — no rand / ndarray needed.
//! let q: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
//! let k = q.clone();
//! let v = q.clone();
//!
//! let fa2 = FlashAttentionV2::new(head_dim, /*causal=*/ false, /*block_size=*/ 4);
//! let out = fa2.forward(&q, &k, &v, seq_len);
//! assert_eq!(out.len(), n);
//! ```

/// Flash Attention 2 operator.
///
/// Computes scaled dot-product attention via tiled online-softmax with O(n)
/// memory (in tile size) rather than the O(n²) required by the naïve algorithm.
#[derive(Debug, Clone)]
pub struct FlashAttentionV2 {
    /// Dimensionality of each head.
    head_dim: usize,
    /// Scale factor applied before softmax: `1 / sqrt(head_dim)` by default.
    scale: f64,
    /// Whether to apply a causal mask (upper-triangular zeroed).
    causal: bool,
    /// Tile / block size along the key dimension.
    block_size: usize,
}

impl FlashAttentionV2 {
    /// Create a new Flash Attention 2 operator.
    ///
    /// # Arguments
    ///
    /// * `head_dim`   — feature dimension per head.
    /// * `causal`     — enable causal masking.
    /// * `block_size` — key-dimension tile size.  Larger tiles use more
    ///   temporary memory but reduce loop overhead.  Must be ≥ 1.
    pub fn new(head_dim: usize, causal: bool, block_size: usize) -> Self {
        let scale = if head_dim > 0 {
            1.0 / (head_dim as f64).sqrt()
        } else {
            1.0
        };
        Self {
            head_dim,
            scale,
            causal,
            block_size: block_size.max(1),
        }
    }

    /// Create with an explicit scale factor.
    pub fn with_scale(head_dim: usize, scale: f64, causal: bool, block_size: usize) -> Self {
        Self {
            head_dim,
            scale,
            causal,
            block_size: block_size.max(1),
        }
    }

    /// Compute Flash Attention 2 forward pass for a single head.
    ///
    /// # Arguments
    ///
    /// * `q`       — query tensor, flat `[seq_len * head_dim]`.
    /// * `k`       — key tensor, flat `[seq_len * head_dim]`.
    /// * `v`       — value tensor, flat `[seq_len * head_dim]`.
    /// * `seq_len` — number of query / key positions.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[seq_len * head_dim]`.
    pub fn forward(&self, q: &[f64], k: &[f64], v: &[f64], seq_len: usize) -> Vec<f64> {
        let d = self.head_dim;
        let expected = seq_len * d;

        if seq_len == 0 || d == 0 || q.len() < expected || k.len() < expected || v.len() < expected
        {
            return vec![0.0; expected];
        }

        // Running statistics per query row: m[i] = max score, l[i] = ∑ exp.
        let mut m = vec![f64::NEG_INFINITY; seq_len];
        let mut l = vec![0.0f64; seq_len];
        let mut out = vec![0.0f64; expected];

        let n_blocks = seq_len.div_ceil(self.block_size);

        for block_idx in 0..n_blocks {
            let k_start = block_idx * self.block_size;
            let k_end = (k_start + self.block_size).min(seq_len);
            let k_block = &k[k_start * d..k_end * d];
            let v_block = &v[k_start * d..k_end * d];

            // Process all query rows against this key block.
            for qi in 0..seq_len {
                let q_row = &q[qi * d..(qi + 1) * d];

                // Compute raw scores qi·kj for j in [k_start, k_end).
                let block_len = k_end - k_start;
                let mut scores = Vec::with_capacity(block_len);

                for (bj, kj) in (k_start..k_end).enumerate() {
                    // Causal mask: only allow kj ≤ qi.
                    if self.causal && kj > qi {
                        scores.push(f64::NEG_INFINITY);
                        continue;
                    }
                    let k_row = &k_block[bj * d..(bj + 1) * d];
                    let dot: f64 = q_row
                        .iter()
                        .zip(k_row.iter())
                        .map(|(&qi_val, &ki_val)| qi_val * ki_val)
                        .sum();
                    scores.push(dot * self.scale);
                }

                // Online softmax update for this block.
                // new_max = max(m[qi], max(scores))
                let block_max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                if block_max == f64::NEG_INFINITY {
                    // All positions in this block are masked — skip.
                    continue;
                }

                let new_m = m[qi].max(block_max);

                // Rescale the existing running sum and output to the new max.
                let rescale = (m[qi] - new_m).exp();
                l[qi] *= rescale;
                let out_row = &mut out[qi * d..(qi + 1) * d];
                for x in out_row.iter_mut() {
                    *x *= rescale;
                }

                // Accumulate new block contribution.
                for (bj, &s) in scores.iter().enumerate() {
                    if s == f64::NEG_INFINITY {
                        continue;
                    }
                    let exp_s = (s - new_m).exp();
                    l[qi] += exp_s;

                    let v_row = &v_block[bj * d..(bj + 1) * d];
                    let out_row = &mut out[qi * d..(qi + 1) * d];
                    for (o, &vv) in out_row.iter_mut().zip(v_row.iter()) {
                        *o += exp_s * vv;
                    }
                }

                m[qi] = new_m;
            }
        }

        // Final normalisation: divide by l[i] for each query row.
        for qi in 0..seq_len {
            let li = l[qi];
            if li > 0.0 && li.is_finite() {
                let out_row = &mut out[qi * d..(qi + 1) * d];
                for x in out_row.iter_mut() {
                    *x /= li;
                }
            }
            // If li == 0 (all positions masked), output row stays zero.
        }

        out
    }

    /// Compute attention for a **tile** of queries against a **block** of keys
    /// and values, updating online-softmax statistics in-place.
    ///
    /// This lower-level function is exposed for composability (e.g. when
    /// integrating with a custom query-tiling loop).
    ///
    /// # Arguments
    ///
    /// * `q_tile`  — query tile, flat `[q_tile_len * head_dim]`.
    /// * `k_block` — key block, flat `[k_block_len * head_dim]`.
    /// * `v_block` — value block, flat `[k_block_len * head_dim]`.
    /// * `q_offset`— starting query index in the full sequence (for causal masking).
    /// * `k_offset`— starting key index in the full sequence (for causal masking).
    /// * `m`       — running max statistics, length `q_tile_len` (updated in-place).
    /// * `l`       — running sum statistics, length `q_tile_len` (updated in-place).
    /// * `out`     — accumulation buffer, flat `[q_tile_len * head_dim]` (updated in-place).
    /// * `scale`   — attention scale factor.
    /// * `causal`  — whether to apply causal masking.
    #[allow(clippy::too_many_arguments)]
    pub fn tile_attention_block(
        q_tile: &[f64],
        k_block: &[f64],
        v_block: &[f64],
        q_offset: usize,
        k_offset: usize,
        m: &mut [f64],
        l: &mut [f64],
        out: &mut [f64],
        scale: f64,
        causal: bool,
        head_dim: usize,
    ) {
        if head_dim == 0 {
            return;
        }
        let q_tile_len = q_tile.len() / head_dim;
        let k_block_len = k_block.len() / head_dim;

        for qi_local in 0..q_tile_len {
            let qi = q_offset + qi_local;
            let q_row = &q_tile[qi_local * head_dim..(qi_local + 1) * head_dim];

            let mut block_max = f64::NEG_INFINITY;
            let mut block_scores = Vec::with_capacity(k_block_len);

            for kj_local in 0..k_block_len {
                let kj = k_offset + kj_local;
                if causal && kj > qi {
                    block_scores.push(f64::NEG_INFINITY);
                    continue;
                }
                let k_row = &k_block[kj_local * head_dim..(kj_local + 1) * head_dim];
                let dot: f64 = q_row.iter().zip(k_row.iter()).map(|(&a, &b)| a * b).sum();
                let s = dot * scale;
                block_scores.push(s);
                if s > block_max {
                    block_max = s;
                }
            }

            if block_max == f64::NEG_INFINITY {
                continue;
            }

            let new_m = m[qi_local].max(block_max);
            let rescale = (m[qi_local] - new_m).exp();
            l[qi_local] *= rescale;

            let out_row = &mut out[qi_local * head_dim..(qi_local + 1) * head_dim];
            for x in out_row.iter_mut() {
                *x *= rescale;
            }

            for (kj_local, &s) in block_scores.iter().enumerate() {
                if s == f64::NEG_INFINITY {
                    continue;
                }
                let exp_s = (s - new_m).exp();
                l[qi_local] += exp_s;
                let v_row = &v_block[kj_local * head_dim..(kj_local + 1) * head_dim];
                let out_row = &mut out[qi_local * head_dim..(qi_local + 1) * head_dim];
                for (o, &vv) in out_row.iter_mut().zip(v_row.iter()) {
                    *o += exp_s * vv;
                }
            }

            m[qi_local] = new_m;
        }
    }

    /// Return the head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Return the configured block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Return the scale factor.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Return whether causal masking is enabled.
    pub fn causal(&self) -> bool {
        self.causal
    }
}

// ---------------------------------------------------------------------------
// Helper: naive reference attention (used in tests)
// ---------------------------------------------------------------------------

/// Naïve O(n²) scaled dot-product attention — reference implementation for
/// testing Flash Attention 2 correctness.
pub fn naive_attention(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    seq_len: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f64> {
    if seq_len == 0 || head_dim == 0 {
        return vec![];
    }
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut out = vec![0.0f64; seq_len * head_dim];

    for qi in 0..seq_len {
        let q_row = &q[qi * head_dim..(qi + 1) * head_dim];

        // Raw scores.
        let mut scores = vec![0.0f64; seq_len];
        for kj in 0..seq_len {
            if causal && kj > qi {
                scores[kj] = f64::NEG_INFINITY;
                continue;
            }
            let k_row = &k[kj * head_dim..(kj + 1) * head_dim];
            let dot: f64 = q_row.iter().zip(k_row.iter()).map(|(&a, &b)| a * b).sum();
            scores[kj] = dot * scale;
        }

        // Stable softmax.
        let max_s = scores
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores
            .iter()
            .map(|&s| {
                if s.is_finite() {
                    (s - max_s).exp()
                } else {
                    0.0
                }
            })
            .collect();
        let sum_e: f64 = exps.iter().sum();
        let weights: Vec<f64> = if sum_e > 0.0 {
            exps.iter().map(|&e| e / sum_e).collect()
        } else {
            vec![1.0 / seq_len as f64; seq_len]
        };

        // Weighted sum of values.
        let out_row = &mut out[qi * head_dim..(qi + 1) * head_dim];
        for kj in 0..seq_len {
            let v_row = &v[kj * head_dim..(kj + 1) * head_dim];
            for (o, &vv) in out_row.iter_mut().zip(v_row.iter()) {
                *o += weights[kj] * vv;
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn arithmetic_tensor(seq_len: usize, head_dim: usize) -> Vec<f64> {
        let n = seq_len * head_dim;
        (0..n).map(|i| (i as f64 + 1.0) * 0.01).collect()
    }

    #[test]
    fn test_flash_v2_output_matches_standard_attention() {
        let seq_len = 8;
        let head_dim = 4;
        let q = arithmetic_tensor(seq_len, head_dim);
        let k = arithmetic_tensor(seq_len, head_dim);
        let v = arithmetic_tensor(seq_len, head_dim);

        let fa2 = FlashAttentionV2::new(head_dim, false, 4);
        let fa_out = fa2.forward(&q, &k, &v, seq_len);
        let naive_out = naive_attention(&q, &k, &v, seq_len, head_dim, false);

        assert_eq!(fa_out.len(), naive_out.len());
        for (a, b) in fa_out.iter().zip(naive_out.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "FA2 vs naive mismatch: {a:.6} vs {b:.6}"
            );
        }
    }

    #[test]
    fn test_flash_v2_causal_mask() {
        // With causal masking and constant Q=K=V, each output row should equal
        // that value (since weights are uniform over past positions, all equal).
        let seq_len = 6;
        let head_dim = 4;

        // Constant q, k — all values 0.1.
        let q = vec![0.1f64; seq_len * head_dim];
        let k = q.clone();

        // V has different values per row so we can verify which are attended.
        let mut v = vec![0.0f64; seq_len * head_dim];
        for i in 0..seq_len {
            for d in 0..head_dim {
                v[i * head_dim + d] = (i + 1) as f64;
            }
        }

        let fa2 = FlashAttentionV2::new(head_dim, true, 4);
        let naive_out = naive_attention(&q, &k, &v, seq_len, head_dim, true);
        let fa_out = fa2.forward(&q, &k, &v, seq_len);

        for (a, b) in fa_out.iter().zip(naive_out.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "causal FA2 vs naive mismatch: {a:.6} vs {b:.6}"
            );
        }
    }

    #[test]
    fn test_flash_v2_online_softmax_numerically_stable() {
        // Create scores with large differences that would cause NaN in
        // non-stable implementations.
        let seq_len = 4;
        let head_dim = 2;

        // Amplified values so raw scores differ by ~700 (would overflow without
        // numerical stability).
        let q = vec![100.0_f64, 0.0, -100.0, 0.0, 100.0, 0.0, -100.0, 0.0];
        let k = vec![1.0_f64, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let v = vec![1.0f64; seq_len * head_dim];

        let fa2 = FlashAttentionV2::new(head_dim, false, 2);
        let out = fa2.forward(&q, &k, &v, seq_len);

        for &x in &out {
            assert!(!x.is_nan(), "output contains NaN — numerical instability");
            assert!(x.is_finite(), "output is not finite");
        }
    }

    #[test]
    fn test_flash_v2_batch_consistency() {
        // Two identical single-head sequences should produce identical outputs.
        let seq_len = 8;
        let head_dim = 8;

        let q = arithmetic_tensor(seq_len, head_dim);
        let k = arithmetic_tensor(seq_len, head_dim);
        let v = arithmetic_tensor(seq_len, head_dim);

        let fa2 = FlashAttentionV2::new(head_dim, false, 4);
        let out1 = fa2.forward(&q, &k, &v, seq_len);
        let out2 = fa2.forward(&q, &k, &v, seq_len);

        assert_eq!(
            out1, out2,
            "identical inputs must produce identical outputs"
        );
    }

    #[test]
    fn test_fa2_tile_sizes_same_result() {
        // block_size=16 and block_size=32 should produce the same output.
        let seq_len = 16;
        let head_dim = 4;
        let q = arithmetic_tensor(seq_len, head_dim);
        let k = arithmetic_tensor(seq_len, head_dim);
        let v = arithmetic_tensor(seq_len, head_dim);

        let fa2_small = FlashAttentionV2::new(head_dim, false, 4);
        let fa2_large = FlashAttentionV2::new(head_dim, false, 16);

        let out_small = fa2_small.forward(&q, &k, &v, seq_len);
        let out_large = fa2_large.forward(&q, &k, &v, seq_len);

        for (a, b) in out_small.iter().zip(out_large.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "tile size should not change output: {a:.8} vs {b:.8}"
            );
        }
    }

    #[test]
    fn test_flash_v2_empty_sequence() {
        let fa2 = FlashAttentionV2::new(4, false, 4);
        let out = fa2.forward(&[], &[], &[], 0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_flash_v2_single_token() {
        // With a single token the output is just the value.
        let head_dim = 4;
        let q = vec![1.0f64, 0.0, 0.0, 0.0];
        let k = vec![1.0f64, 0.0, 0.0, 0.0];
        let v = vec![2.0f64, 3.0, 4.0, 5.0];

        let fa2 = FlashAttentionV2::new(head_dim, false, 4);
        let out = fa2.forward(&q, &k, &v, 1);

        assert_eq!(out, v, "single token: output must equal value");
    }

    #[test]
    fn test_tile_attention_block_updates_running_stats() {
        let head_dim = 2;
        let q_tile = vec![1.0f64, 0.0];
        let k_block = vec![1.0f64, 0.0, 0.5, 0.0];
        let v_block = vec![1.0f64, 2.0, 3.0, 4.0];

        let mut m = vec![f64::NEG_INFINITY];
        let mut l = vec![0.0f64];
        let mut out = vec![0.0f64; head_dim];

        FlashAttentionV2::tile_attention_block(
            &q_tile, &k_block, &v_block, 0, 0, &mut m, &mut l, &mut out, 1.0, false, head_dim,
        );

        // l must be > 0 after processing two tokens.
        assert!(l[0] > 0.0, "running sum should be positive");
        // m must be finite after processing.
        assert!(m[0].is_finite(), "running max should be finite");
    }
}
