//! Sliding-window attention (Longformer / Mistral style).
//!
//! Instead of computing the full O(n²) attention matrix, each token attends
//! only to a symmetric local window of `window_size` tokens on each side.
//! Optionally, a set of *global* tokens (e.g. the `[CLS]` token) attend to
//! **all** positions and all positions attend back to them.
//!
//! Complexity: O(n × w) where `w = window_size`, compared to O(n²) for dense
//! attention.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_neural::attention::sliding_window::{SlidingWindowAttention, SlidingWindowConfig};
//!
//! let cfg = SlidingWindowConfig::default(); // window_size=512, global_tokens=0
//! let swa = SlidingWindowAttention::new(cfg);
//!
//! let seq_len = 8;
//! let head_dim = 4;
//! let n = seq_len * head_dim;
//! let q: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
//! let out = swa.forward(&q, &q, &q, seq_len, head_dim);
//! assert_eq!(out.len(), n);
//! ```

/// Configuration for sliding-window attention.
///
/// # Defaults
///
/// ```rust
/// use scirs2_neural::attention::sliding_window::SlidingWindowConfig;
/// let cfg = SlidingWindowConfig::default();
/// assert_eq!(cfg.window_size, 512);
/// assert_eq!(cfg.global_tokens, 0);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Half-width of the local attention window.  Token `i` attends to
    /// positions `max(0, i - w)..=min(n-1, i + w)`.
    pub window_size: usize,

    /// Number of leading tokens (0, 1, …, `global_tokens - 1`) that attend to
    /// **all** positions and from which all positions attend back.
    pub global_tokens: usize,
}

impl Default for SlidingWindowConfig {
    fn default() -> Self {
        Self {
            window_size: 512,
            global_tokens: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// SlidingWindowAttention
// ---------------------------------------------------------------------------

/// Sliding-window scaled dot-product attention operator.
#[derive(Debug, Clone)]
pub struct SlidingWindowAttention {
    config: SlidingWindowConfig,
}

impl SlidingWindowAttention {
    /// Create a new operator with the given configuration.
    pub fn new(config: SlidingWindowConfig) -> Self {
        Self { config }
    }

    /// Compute the sliding-window attention output.
    ///
    /// # Layout convention
    ///
    /// All Q / K / V tensors are stored as flat `f64` slices in row-major
    /// order with logical shape `[seq_len, head_dim]` for a **single head**.
    /// Multi-head batching is handled by the caller.
    ///
    /// # Arguments
    ///
    /// * `q`        — query tensor, flat `[seq_len * head_dim]`.
    /// * `k`        — key tensor, flat `[seq_len * head_dim]`.
    /// * `v`        — value tensor, flat `[seq_len * head_dim]`.
    /// * `seq_len`  — number of sequence positions.
    /// * `head_dim` — feature dimension per position.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[seq_len * head_dim]`.
    pub fn forward(
        &self,
        q: &[f64],
        k: &[f64],
        v: &[f64],
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f64> {
        let expected = seq_len * head_dim;
        if seq_len == 0 || head_dim == 0 || q.len() < expected {
            return vec![0.0; expected];
        }

        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut out = vec![0.0f64; expected];
        let window = self.config.window_size;
        let global = self.config.global_tokens;

        for qi in 0..seq_len {
            // Global tokens attend to all positions; others use local window.
            let (k_start, k_end) = if qi < global {
                (0, seq_len)
            } else {
                (qi.saturating_sub(window), (qi + window + 1).min(seq_len))
            };

            let v_out = self.compute_local_attention(
                qi, k_start, k_end, q, k, v, head_dim, scale, global, seq_len,
            );

            let out_offset = qi * head_dim;
            out[out_offset..out_offset + head_dim].copy_from_slice(&v_out);
        }

        out
    }

    /// Compute the attention output for a single query position `qi`.
    ///
    /// This function collects the key positions in `[k_start, k_end)` plus any
    /// global tokens that fall outside that range, runs softmax, and returns
    /// the weighted value sum.
    ///
    /// # Arguments
    ///
    /// * `q_pos`     — query position index.
    /// * `k_start`   — inclusive start of local key range.
    /// * `k_end`     — exclusive end of local key range.
    /// * `q`         — full query tensor, flat `[seq_len * head_dim]`.
    /// * `k`         — full key tensor, flat `[seq_len * head_dim]`.
    /// * `v`         — full value tensor, flat `[seq_len * head_dim]`.
    /// * `head_dim`  — feature dimension per position.
    /// * `scale`     — attention scale factor.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_local_attention(
        &self,
        q_pos: usize,
        k_start: usize,
        k_end: usize,
        q: &[f64],
        k: &[f64],
        v: &[f64],
        head_dim: usize,
        scale: f64,
        global: usize,
        seq_len: usize,
    ) -> Vec<f64> {
        let q_row = &q[q_pos * head_dim..(q_pos + 1) * head_dim];

        // Build sorted, deduplicated list of attended key positions.
        let mut key_positions: Vec<usize> = (k_start..k_end).collect();

        // Add global tokens that are outside the local window.
        for g in 0..global.min(seq_len) {
            if g < k_start || g >= k_end {
                key_positions.push(g);
            }
        }
        key_positions.sort_unstable();
        key_positions.dedup();

        if key_positions.is_empty() {
            return vec![0.0f64; head_dim];
        }

        // Compute raw scores.
        let mut scores: Vec<f64> = key_positions
            .iter()
            .map(|&kj| {
                let k_row = &k[kj * head_dim..(kj + 1) * head_dim];
                let dot: f64 = q_row
                    .iter()
                    .zip(k_row.iter())
                    .map(|(&qi, &ki)| qi * ki)
                    .sum();
                dot * scale
            })
            .collect();

        // Stable softmax.
        let max_s = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for s in &mut scores {
            *s = (*s - max_s).exp();
        }
        let sum_s: f64 = scores.iter().sum();
        if sum_s > 0.0 {
            for s in &mut scores {
                *s /= sum_s;
            }
        } else {
            let u = 1.0 / key_positions.len() as f64;
            scores.iter_mut().for_each(|s| *s = u);
        }

        // Weighted sum of values.
        let mut out = vec![0.0f64; head_dim];
        for (&weight, &kj) in scores.iter().zip(key_positions.iter()) {
            let v_row = &v[kj * head_dim..(kj + 1) * head_dim];
            for (o, &vv) in out.iter_mut().zip(v_row.iter()) {
                *o += weight * vv;
            }
        }
        out
    }

    /// Return the configuration.
    pub fn config(&self) -> &SlidingWindowConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn arithmetic_qkv(seq_len: usize, head_dim: usize) -> Vec<f64> {
        let n = seq_len * head_dim;
        (0..n).map(|i| (i as f64 + 1.0) * 0.05).collect()
    }

    /// Full-dense attention (no window restriction) — reference.
    fn dense_attention(
        q: &[f64],
        k: &[f64],
        v: &[f64],
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f64> {
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut out = vec![0.0f64; seq_len * head_dim];

        for qi in 0..seq_len {
            let q_row = &q[qi * head_dim..(qi + 1) * head_dim];
            let mut scores: Vec<f64> = (0..seq_len)
                .map(|kj| {
                    let k_row = &k[kj * head_dim..(kj + 1) * head_dim];
                    let dot: f64 = q_row.iter().zip(k_row.iter()).map(|(a, b)| a * b).sum();
                    dot * scale
                })
                .collect();
            let max_s = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for s in &mut scores {
                *s = (*s - max_s).exp();
            }
            let sum_s: f64 = scores.iter().sum();
            for s in &mut scores {
                *s /= sum_s;
            }
            let out_row = &mut out[qi * head_dim..(qi + 1) * head_dim];
            for kj in 0..seq_len {
                let v_row = &v[kj * head_dim..(kj + 1) * head_dim];
                for (o, &vv) in out_row.iter_mut().zip(v_row.iter()) {
                    *o += scores[kj] * vv;
                }
            }
        }
        out
    }

    #[test]
    fn test_sliding_window_vs_full_for_short_seq() {
        // When seq_len <= window_size, sliding-window == full attention.
        let seq_len = 6;
        let head_dim = 4;
        let window_size = 8; // larger than seq_len

        let q = arithmetic_qkv(seq_len, head_dim);
        let k = arithmetic_qkv(seq_len, head_dim);
        let v = arithmetic_qkv(seq_len, head_dim);

        let cfg = SlidingWindowConfig {
            window_size,
            global_tokens: 0,
        };
        let swa = SlidingWindowAttention::new(cfg);
        let swa_out = swa.forward(&q, &k, &v, seq_len, head_dim);
        let dense_out = dense_attention(&q, &k, &v, seq_len, head_dim);

        for (a, b) in swa_out.iter().zip(dense_out.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "sliding-window with large window should match dense: {a:.8} vs {b:.8}"
            );
        }
    }

    #[test]
    fn test_sliding_window_attends_only_to_neighbors() {
        // Use V with distinct values per row so we can detect which positions
        // contributed to the output.
        let seq_len = 8;
        let head_dim = 2;
        let window_size = 1; // only immediate neighbors

        // q and k are all-ones (uniform attention).
        let q = vec![1.0f64; seq_len * head_dim];
        let k = q.clone();

        // V row i has values [i+1, i+1].
        let v: Vec<f64> = (0..seq_len)
            .flat_map(|i| vec![(i + 1) as f64; head_dim])
            .collect();

        let cfg = SlidingWindowConfig {
            window_size,
            global_tokens: 0,
        };
        let swa = SlidingWindowAttention::new(cfg);
        let out = swa.forward(&q, &k, &v, seq_len, head_dim);

        // For position i=4 with window=1, the attended positions are {3,4,5}.
        // With uniform attention the output should be the average of V[3], V[4], V[5]
        // = (4+5+6)/3 = 5.
        let pos4_val = out[4 * head_dim];
        let expected = (4.0 + 5.0 + 6.0) / 3.0;
        assert!(
            (pos4_val - expected).abs() < 1e-9,
            "pos4 should attend to neighbors: got {pos4_val:.6}, expected {expected:.6}"
        );
    }

    #[test]
    fn test_sliding_window_boundary_positions() {
        // Position 0 can only attend backward within window (just itself and
        // the right neighbor if window=1).
        let seq_len = 4;
        let head_dim = 2;
        let window_size = 1;

        let q = vec![1.0f64; seq_len * head_dim];
        let k = q.clone();
        let v: Vec<f64> = (0..seq_len)
            .flat_map(|i| vec![(i + 1) as f64; head_dim])
            .collect();

        let cfg = SlidingWindowConfig {
            window_size,
            global_tokens: 0,
        };
        let swa = SlidingWindowAttention::new(cfg);
        let out = swa.forward(&q, &k, &v, seq_len, head_dim);

        // Position 0: attends to {0, 1} → average of V[0], V[1] = (1+2)/2 = 1.5
        let pos0_val = out[0];
        assert!(
            (pos0_val - 1.5).abs() < 1e-9,
            "boundary pos0: got {pos0_val:.6}"
        );

        // Position 3 (last): attends to {2, 3} → (3+4)/2 = 3.5
        let pos3_val = out[3 * head_dim];
        assert!(
            (pos3_val - 3.5).abs() < 1e-9,
            "boundary pos3: got {pos3_val:.6}"
        );
    }

    #[test]
    fn test_sliding_window_global_tokens() {
        // With global_tokens=1 the first token attends to ALL positions.
        let seq_len = 6;
        let head_dim = 2;
        let window_size = 1;

        let q = vec![1.0f64; seq_len * head_dim];
        let k = q.clone();
        let v: Vec<f64> = (0..seq_len)
            .flat_map(|i| vec![(i + 1) as f64; head_dim])
            .collect();

        let cfg = SlidingWindowConfig {
            window_size,
            global_tokens: 1,
        };
        let swa = SlidingWindowAttention::new(cfg);
        let out = swa.forward(&q, &k, &v, seq_len, head_dim);

        // Position 0 (global): attends to all 6 positions → average = (1+2+3+4+5+6)/6 = 3.5
        let pos0_val = out[0];
        let expected = (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0) / 6.0;
        assert!(
            (pos0_val - expected).abs() < 1e-9,
            "global pos0 should attend all: got {pos0_val:.6}, expected {expected:.6}"
        );
    }

    #[test]
    fn test_sliding_window_global_token_included_by_all() {
        // With global_tokens=1 every position should include token 0 in its
        // attended set, even if 0 is outside its local window.
        let seq_len = 8;
        let head_dim = 2;
        let window_size = 1; // only immediate neighbors
        let global_tokens = 1;

        // Q = K = ones for uniform attention.
        let q = vec![1.0f64; seq_len * head_dim];
        let k = q.clone();

        // V: row 0 has value 100, all others have value 1.
        let mut v = vec![1.0f64; seq_len * head_dim];
        v[0] = 100.0;
        v[1] = 100.0;

        let cfg = SlidingWindowConfig {
            window_size,
            global_tokens,
        };
        let swa = SlidingWindowAttention::new(cfg);
        let out = swa.forward(&q, &k, &v, seq_len, head_dim);

        // For position 5 (window [4,5,6] + global [0]) the global token with
        // V=100 should pull the output well above 1.
        let pos5 = out[5 * head_dim];
        assert!(
            pos5 > 1.0,
            "pos5 should be above 1.0 due to global token: got {pos5:.4}"
        );
    }

    #[test]
    fn test_sliding_window_output_shape() {
        let seq_len = 10;
        let head_dim = 8;
        let q = arithmetic_qkv(seq_len, head_dim);
        let cfg = SlidingWindowConfig::default();
        let swa = SlidingWindowAttention::new(cfg);
        let out = swa.forward(&q, &q, &q, seq_len, head_dim);
        assert_eq!(out.len(), seq_len * head_dim);
    }

    #[test]
    fn test_sliding_window_empty_sequence() {
        let cfg = SlidingWindowConfig::default();
        let swa = SlidingWindowAttention::new(cfg);
        let out = swa.forward(&[], &[], &[], 0, 4);
        assert!(out.is_empty() || out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_sliding_window_config_default() {
        let cfg = SlidingWindowConfig::default();
        assert_eq!(cfg.window_size, 512);
        assert_eq!(cfg.global_tokens, 0);
    }
}
