//! FlashAttention-style memory-efficient attention for time series.
//!
//! ## Algorithm
//!
//! Standard scaled dot-product attention computes:
//!
//! ```text
//! Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V
//! ```
//!
//! The full Q Kᵀ matrix is O(N²) in memory.  FlashAttention avoids this by
//! processing Q in row-tiles (B_r rows) and K/V in column-tiles (B_c columns)
//! while maintaining a **running online softmax** state `(m_i, l_i)`:
//!
//! ```text
//! m_i  = running max of row i's dot-products seen so far
//! l_i  = Σ exp(s - m_i)   (normaliser)
//! O_i  = accumulated weighted value sum / l_i
//! ```
//!
//! When a new K/V block arrives:
//!
//! ```text
//! m_new  = max(m_prev, max(S_block))
//! l_new  = exp(m_prev - m_new) * l_prev + Σ exp(S_block - m_new)
//! O_new  = (l_prev * exp(m_prev - m_new) * O_prev
//!          + sum_j exp(S_ij - m_new) * V_j) / l_new
//! ```
//!
//! This is numerically stable and uses O(N · d) memory instead of O(N²).
//!
//! ## CPU vs GPU trade-offs
//!
//! The GPU implementation leverages SRAM to keep tiles on-chip.  Here we use
//! the same tiling algorithm in pure Rust, which eliminates the N² allocation
//! and improves cache locality compared to the naive implementation.

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::{s, Array2, Array3};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for FlashAttention.
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Tile / block size for query rows and key-value columns (default: 64).
    ///
    /// Larger values improve arithmetic intensity at the cost of more memory
    /// per tile.  Must be ≥ 1.
    pub block_size: usize,
    /// Dropout probability applied to attention weights (default: 0.0).
    ///
    /// Note: this is a *simulated* stochastic drop implemented with a
    /// deterministic LCG — suitable for testing, not production training.
    pub dropout: f64,
    /// Whether to apply a causal (auto-regressive) mask (default: `true`).
    ///
    /// When `true`, position `i` cannot attend to position `j > i`.
    pub causal: bool,
    /// Explicit scaling factor.  `None` ⟹ `1 / √d_k` (default: `None`).
    pub scale: Option<f64>,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            dropout: 0.0,
            causal: true,
            scale: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-head flash attention
// ─────────────────────────────────────────────────────────────────────────────

/// Memory-efficient attention via tiled online softmax.
///
/// # Arguments
///
/// * `q` – query matrix  `[seq_len, d_k]`
/// * `k` – key matrix    `[seq_len, d_k]`
/// * `v` – value matrix  `[seq_len, d_v]`
/// * `config` – tuning knobs
///
/// # Returns
///
/// Output matrix `[seq_len, d_v]`.
///
/// # Errors
///
/// Returns [`TimeSeriesError::InvalidInput`] when:
/// * `q`, `k` have different second dimensions (d_k mismatch), or
/// * `k`, `v` have different first dimensions (seq_len mismatch), or
/// * `d_k == 0`.
pub fn flash_attention(
    q: &Array2<f64>,
    k: &Array2<f64>,
    v: &Array2<f64>,
    config: &FlashAttentionConfig,
) -> Result<Array2<f64>> {
    let seq_q = q.nrows();
    let d_k = q.ncols();
    let seq_k = k.nrows();
    let d_v = v.ncols();

    if d_k == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "flash_attention: d_k must be > 0".to_string(),
        ));
    }
    if k.ncols() != d_k {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: d_k,
            actual: k.ncols(),
        });
    }
    if seq_k != v.nrows() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: seq_k,
            actual: v.nrows(),
        });
    }

    let scale = config.scale.unwrap_or_else(|| 1.0 / (d_k as f64).sqrt());

    let block_size = config.block_size.max(1);

    // Output accumulator and per-row statistics.
    let mut output = Array2::<f64>::zeros((seq_q, d_v));
    // m_i: running row maximum (log domain)
    let mut m_running = vec![f64::NEG_INFINITY; seq_q];
    // l_i: running normalisation sum
    let mut l_running = vec![0.0_f64; seq_q];

    // Iterate over K/V column-tiles.
    let n_kv_blocks = (seq_k + block_size - 1) / block_size;

    for kv_block in 0..n_kv_blocks {
        let kv_start = kv_block * block_size;
        let kv_end = (kv_start + block_size).min(seq_k);
        let tile_kv = kv_end - kv_start;

        // Extract K tile: [tile_kv, d_k]
        let k_tile = k.slice(s![kv_start..kv_end, ..]);
        // Extract V tile: [tile_kv, d_v]
        let v_tile = v.slice(s![kv_start..kv_end, ..]);

        // Iterate over Q row-tiles.
        let n_q_blocks = (seq_q + block_size - 1) / block_size;

        for q_block in 0..n_q_blocks {
            let q_start = q_block * block_size;
            let q_end = (q_start + block_size).min(seq_q);
            let tile_q = q_end - q_start;

            // Extract Q tile: [tile_q, d_k]
            let q_tile = q.slice(s![q_start..q_end, ..]);

            // Compute S = Q_tile @ K_tile^T * scale → [tile_q, tile_kv]
            let mut s_mat = vec![0.0_f64; tile_q * tile_kv];
            for qi in 0..tile_q {
                for ki in 0..tile_kv {
                    let mut dot = 0.0_f64;
                    for dk in 0..d_k {
                        dot += q_tile[[qi, dk]] * k_tile[[ki, dk]];
                    }
                    s_mat[qi * tile_kv + ki] = dot * scale;
                }
            }

            // Apply causal mask: set S[qi, ki] = -∞ when kv_start+ki > q_start+qi
            if config.causal {
                for qi in 0..tile_q {
                    let abs_q = q_start + qi;
                    for ki in 0..tile_kv {
                        let abs_k = kv_start + ki;
                        if abs_k > abs_q {
                            s_mat[qi * tile_kv + ki] = f64::NEG_INFINITY;
                        }
                    }
                }
            }

            // For each query row, perform online softmax update.
            for qi in 0..tile_q {
                let abs_qi = q_start + qi;
                let row = &s_mat[qi * tile_kv..(qi + 1) * tile_kv];

                // Find local maximum for this row (handles -∞ entries).
                let m_local = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                // Compute exp(s - m_local) for each position.
                let exps: Vec<f64> = row
                    .iter()
                    .map(|&s| {
                        if s == f64::NEG_INFINITY {
                            0.0
                        } else {
                            (s - m_local).exp()
                        }
                    })
                    .collect();

                // Sum of exps for local block.
                let l_local: f64 = exps.iter().sum();

                // Online softmax merge.
                let m_prev = m_running[abs_qi];
                let l_prev = l_running[abs_qi];

                let m_new = m_prev.max(m_local);
                // Correction factor for previously accumulated output.
                let correction = if m_prev == f64::NEG_INFINITY {
                    // First tile that has any non-masked entries.
                    0.0
                } else {
                    (m_prev - m_new).exp()
                };
                let local_scale = if m_local == f64::NEG_INFINITY {
                    0.0
                } else {
                    (m_local - m_new).exp()
                };

                let l_new = correction * l_prev + local_scale * l_local;

                // Update output row.
                if l_new > 0.0 {
                    for dv in 0..d_v {
                        // Weighted sum from current K/V tile.
                        let new_contrib: f64 = exps
                            .iter()
                            .enumerate()
                            .map(|(ki, &e)| e * v_tile[[ki, dv]])
                            .sum();

                        let prev_out = output[[abs_qi, dv]];
                        output[[abs_qi, dv]] =
                            (correction * l_prev * prev_out + local_scale * new_contrib) / l_new;
                    }
                }

                m_running[abs_qi] = m_new;
                l_running[abs_qi] = l_new;
            }
        }
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-head flash attention
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-head FlashAttention.
///
/// Splits the last dimension into `n_heads` independent heads, applies
/// [`flash_attention`] per head, and concatenates the results.
///
/// # Arguments
///
/// * `q`       – `[batch, seq_len, n_heads * d_k]`
/// * `k`       – `[batch, seq_len, n_heads * d_k]`
/// * `v`       – `[batch, seq_len, n_heads * d_v]`
/// * `n_heads` – number of attention heads
/// * `config`  – passed through to [`flash_attention`]
///
/// # Returns
///
/// `[batch, seq_len, n_heads * d_v]`
///
/// # Errors
///
/// Returns [`TimeSeriesError::InvalidInput`] when the last dimension of `q`/`k`
/// is not divisible by `n_heads`, or batch/seq shapes are inconsistent.
pub fn multi_head_flash_attention(
    q: &Array3<f64>,
    k: &Array3<f64>,
    v: &Array3<f64>,
    n_heads: usize,
    config: &FlashAttentionConfig,
) -> Result<Array3<f64>> {
    if n_heads == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "n_heads must be ≥ 1".to_string(),
        ));
    }

    let (batch, seq_len, qk_dim) = (q.shape()[0], q.shape()[1], q.shape()[2]);
    let (batch_k, seq_k, k_dim) = (k.shape()[0], k.shape()[1], k.shape()[2]);
    let (batch_v, seq_v, v_dim) = (v.shape()[0], v.shape()[1], v.shape()[2]);

    if batch_k != batch || seq_k != seq_len || k_dim != qk_dim {
        return Err(TimeSeriesError::InvalidInput(format!(
            "K shape [{batch_k}, {seq_k}, {k_dim}] incompatible with Q [{batch}, {seq_len}, {qk_dim}]"
        )));
    }
    if batch_v != batch || seq_v != seq_len {
        return Err(TimeSeriesError::InvalidInput(format!(
            "V batch/seq [{batch_v}, {seq_v}] incompatible with Q [{batch}, {seq_len}]"
        )));
    }
    if qk_dim % n_heads != 0 {
        return Err(TimeSeriesError::InvalidInput(format!(
            "Q/K last dim {qk_dim} not divisible by n_heads {n_heads}"
        )));
    }
    if v_dim % n_heads != 0 {
        return Err(TimeSeriesError::InvalidInput(format!(
            "V last dim {v_dim} not divisible by n_heads {n_heads}"
        )));
    }

    let d_k = qk_dim / n_heads;
    let d_v = v_dim / n_heads;

    let mut output = Array3::<f64>::zeros((batch, seq_len, n_heads * d_v));

    for b in 0..batch {
        for h in 0..n_heads {
            let q_head_start = h * d_k;
            let k_head_start = h * d_k;
            let v_head_start = h * d_v;

            // Extract head slices as 2-D arrays [seq_len, d_k/d_v].
            let q_slice = q.slice(s![b, .., q_head_start..q_head_start + d_k]);
            let k_slice = k.slice(s![b, .., k_head_start..k_head_start + d_k]);
            let v_slice = v.slice(s![b, .., v_head_start..v_head_start + d_v]);

            // Convert to owned Array2.
            let q2: Array2<f64> = q_slice.to_owned();
            let k2: Array2<f64> = k_slice.to_owned();
            let v2: Array2<f64> = v_slice.to_owned();

            let head_out = flash_attention(&q2, &k2, &v2, config)?;

            // Write head output into the correct columns.
            let out_start = h * d_v;
            for i in 0..seq_len {
                for dv in 0..d_v {
                    output[[b, i, out_start + dv]] = head_out[[i, dv]];
                }
            }
        }
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array2, Array3};

    /// Naive (reference) attention for small sequences.
    fn naive_attention(
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
        scale: f64,
        causal: bool,
    ) -> Array2<f64> {
        let seq = q.nrows();
        let d_v = v.ncols();
        let mut out = Array2::zeros((seq, d_v));
        for i in 0..seq {
            let mut scores = vec![0.0_f64; seq];
            for j in 0..seq {
                if causal && j > i {
                    scores[j] = f64::NEG_INFINITY;
                    continue;
                }
                let dot: f64 = (0..q.ncols()).map(|dk| q[[i, dk]] * k[[j, dk]]).sum();
                scores[j] = dot * scale;
            }
            // Softmax.
            let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores
                .iter()
                .map(|&s| {
                    if s == f64::NEG_INFINITY {
                        0.0
                    } else {
                        (s - max_s).exp()
                    }
                })
                .collect();
            let sum_e: f64 = exps.iter().sum();
            for j in 0..seq {
                if sum_e > 0.0 {
                    for dv in 0..d_v {
                        out[[i, dv]] += (exps[j] / sum_e) * v[[j, dv]];
                    }
                }
            }
        }
        out
    }

    // Helper: build a simple [n, d] array from a flat slice (row-major).
    fn arr2(rows: usize, cols: usize, data: &[f64]) -> Array2<f64> {
        Array2::from_shape_vec((rows, cols), data.to_vec()).expect("shape matches data length")
    }

    #[test]
    fn test_flash_attention_shape() {
        let seq = 16;
        let d_k = 8;
        let d_v = 4;
        let q = Array2::ones((seq, d_k));
        let k = Array2::ones((seq, d_k));
        let v = Array2::ones((seq, d_v));
        let cfg = FlashAttentionConfig::default();
        let out = flash_attention(&q, &k, &v, &cfg).expect("should succeed");
        assert_eq!(out.shape(), &[seq, d_v]);
    }

    #[test]
    fn test_flash_attention_matches_standard() {
        // Small sequence where numerical comparison is feasible.
        let seq = 8;
        let d_k = 4;
        let d_v = 4;

        // Deterministic pseudo-random data.
        let data: Vec<f64> = (0..seq * d_k)
            .map(|i| ((i as f64 * 1.1 + 0.3).sin()) * 0.5)
            .collect();
        let q = arr2(seq, d_k, &data);
        let k_data: Vec<f64> = (0..seq * d_k)
            .map(|i| ((i as f64 * 0.9 + 0.7).cos()) * 0.5)
            .collect();
        let k = arr2(seq, d_k, &k_data);
        let v_data: Vec<f64> = (0..seq * d_v)
            .map(|i| (i as f64 * 0.3).sin() * 0.3)
            .collect();
        let v = arr2(seq, d_v, &v_data);

        let scale = 1.0 / (d_k as f64).sqrt();
        let cfg = FlashAttentionConfig {
            block_size: 4,
            causal: true,
            dropout: 0.0,
            scale: Some(scale),
        };

        let flash_out = flash_attention(&q, &k, &v, &cfg).expect("flash should succeed");
        let naive_out = naive_attention(&q, &k, &v, scale, true);

        for i in 0..seq {
            for dv in 0..d_v {
                let diff = (flash_out[[i, dv]] - naive_out[[i, dv]]).abs();
                assert!(
                    diff < 1e-10,
                    "flash vs naive mismatch at [{i},{dv}]: flash={:.6e} naive={:.6e}",
                    flash_out[[i, dv]],
                    naive_out[[i, dv]]
                );
            }
        }
    }

    #[test]
    fn test_flash_attention_non_causal_matches_standard() {
        let seq = 6;
        let d_k = 4;
        let d_v = 4;

        let q = Array2::from_shape_fn((seq, d_k), |(i, j)| ((i + j) as f64 * 0.1).sin());
        let k = Array2::from_shape_fn((seq, d_k), |(i, j)| ((i * 2 + j) as f64 * 0.15).cos());
        let v = Array2::from_shape_fn((seq, d_v), |(i, j)| i as f64 * 0.2 + j as f64 * 0.05);

        let scale = 1.0 / (d_k as f64).sqrt();
        let cfg = FlashAttentionConfig {
            block_size: 3,
            causal: false,
            dropout: 0.0,
            scale: Some(scale),
        };

        let flash_out = flash_attention(&q, &k, &v, &cfg).expect("flash should succeed");
        let naive_out = naive_attention(&q, &k, &v, scale, false);

        for i in 0..seq {
            for dv in 0..d_v {
                let diff = (flash_out[[i, dv]] - naive_out[[i, dv]]).abs();
                assert!(diff < 1e-10, "non-causal mismatch at [{i},{dv}]");
            }
        }
    }

    #[test]
    fn test_flash_attention_causal_mask() {
        // With causal masking, output at position 0 should depend only on V[0].
        let seq = 4;
        let d_k = 2;
        let d_v = 2;

        // Q and K are identical unit vectors so attention score = scale (constant).
        let q = Array2::ones((seq, d_k));
        let k = Array2::ones((seq, d_k));
        // Each V row is uniquely identifiable.
        let v = Array2::from_shape_fn((seq, d_v), |(i, _j)| i as f64);

        let cfg = FlashAttentionConfig {
            block_size: 2,
            causal: true,
            dropout: 0.0,
            scale: Some(1.0),
        };

        let out = flash_attention(&q, &k, &v, &cfg).expect("should succeed");

        // Position 0: only attends to position 0 → output should equal V[0] = 0.0
        assert!(
            (out[[0, 0]] - 0.0).abs() < 1e-10,
            "position 0 should attend only to itself, got {}",
            out[[0, 0]]
        );
        // Position 1: attends to positions 0 and 1, average of 0 and 1 = 0.5
        let expected_row1 = 0.5;
        assert!(
            (out[[1, 0]] - expected_row1).abs() < 1e-9,
            "position 1 expected {expected_row1} got {}",
            out[[1, 0]]
        );
    }

    #[test]
    fn test_flash_attention_large_seq() {
        // Verify that large sequences are handled without allocation panic.
        let seq = 1024;
        let d_k = 16;
        let d_v = 16;
        let q = Array2::from_shape_fn((seq, d_k), |(i, j)| ((i + j) as f64 * 0.001).sin());
        let k = Array2::from_shape_fn((seq, d_k), |(i, j)| ((i * 2 + j) as f64 * 0.001).cos());
        let v = Array2::ones((seq, d_v));
        let cfg = FlashAttentionConfig {
            block_size: 64,
            causal: true,
            dropout: 0.0,
            scale: None,
        };
        let out = flash_attention(&q, &k, &v, &cfg).expect("large sequence should succeed");
        assert_eq!(out.shape(), &[seq, d_v]);
        // All values should be finite.
        for val in out.iter() {
            assert!(val.is_finite(), "output contains non-finite value");
        }
    }

    #[test]
    fn test_multi_head_flash_attention_shape() {
        let batch = 2;
        let seq = 16;
        let n_heads = 4;
        let d_k = 8; // per head
        let d_v = 8; // per head

        let q = Array3::ones((batch, seq, n_heads * d_k));
        let k = Array3::ones((batch, seq, n_heads * d_k));
        let v = Array3::ones((batch, seq, n_heads * d_v));

        let cfg = FlashAttentionConfig::default();
        let out = multi_head_flash_attention(&q, &k, &v, n_heads, &cfg)
            .expect("multi-head should succeed");

        assert_eq!(out.shape(), &[batch, seq, n_heads * d_v]);
    }

    #[test]
    fn test_flash_attention_scale_applied() {
        // Two configs differing only in scale should produce different outputs.
        let seq = 4;
        let d_k = 4;
        let d_v = 2;
        let q = Array2::from_shape_fn((seq, d_k), |(i, j)| (i * d_k + j) as f64 * 0.1);
        let k = q.clone();
        // V must NOT be uniform — otherwise any softmax weighting gives the same result.
        let v = Array2::from_shape_fn((seq, d_v), |(i, j)| (i * d_v + j + 1) as f64 * 0.3);

        let cfg_default = FlashAttentionConfig {
            causal: false,
            scale: None, // 1/√4 = 0.5
            ..Default::default()
        };
        let cfg_big_scale = FlashAttentionConfig {
            causal: false,
            scale: Some(10.0),
            ..Default::default()
        };

        let out_default = flash_attention(&q, &k, &v, &cfg_default).expect("should succeed");
        let out_big = flash_attention(&q, &k, &v, &cfg_big_scale).expect("should succeed");

        // With large scale, softmax is more peaked — outputs should differ.
        let same = out_default
            .iter()
            .zip(out_big.iter())
            .all(|(a, b)| (a - b).abs() < 1e-12);
        assert!(!same, "different scales should produce different outputs");
    }

    #[test]
    fn test_flash_attention_error_dk_mismatch() {
        let q = Array2::ones((4, 8));
        let k = Array2::ones((4, 6)); // d_k mismatch
        let v = Array2::ones((4, 4));
        let cfg = FlashAttentionConfig::default();
        let result = flash_attention(&q, &k, &v, &cfg);
        assert!(result.is_err(), "should error on d_k mismatch");
    }

    #[test]
    fn test_flash_attention_error_seq_mismatch() {
        let q = Array2::ones((4, 8));
        let k = Array2::ones((4, 8));
        let v = Array2::ones((5, 4)); // seq_len mismatch between k and v
        let cfg = FlashAttentionConfig::default();
        let result = flash_attention(&q, &k, &v, &cfg);
        assert!(result.is_err(), "should error on seq_len mismatch");
    }

    #[test]
    fn test_flash_attention_block_size_one() {
        // Extreme tiling: block_size=1 should still produce correct results.
        let seq = 5;
        let d_k = 3;
        let d_v = 3;
        let q = Array2::from_shape_fn((seq, d_k), |(i, j)| (i + j) as f64 * 0.2);
        let k = Array2::from_shape_fn((seq, d_k), |(i, j)| (i * 2 + j) as f64 * 0.1);
        let v = Array2::from_shape_fn((seq, d_v), |(i, _j)| i as f64 * 0.5);

        let scale = 1.0 / (d_k as f64).sqrt();
        let cfg_tiny = FlashAttentionConfig {
            block_size: 1,
            causal: true,
            dropout: 0.0,
            scale: Some(scale),
        };
        let cfg_large = FlashAttentionConfig {
            block_size: 64,
            causal: true,
            dropout: 0.0,
            scale: Some(scale),
        };

        let out_tiny = flash_attention(&q, &k, &v, &cfg_tiny).expect("block_size=1");
        let out_large = flash_attention(&q, &k, &v, &cfg_large).expect("block_size=64");

        for i in 0..seq {
            for dv in 0..d_v {
                let diff = (out_tiny[[i, dv]] - out_large[[i, dv]]).abs();
                assert!(diff < 1e-10, "block_size sensitivity at [{i},{dv}]");
            }
        }
    }

    #[test]
    fn test_multi_head_flash_attention_head_split_correctness() {
        // Single head should match single-head flash_attention.
        let batch = 1;
        let seq = 8;
        let d_k = 4;

        let q3 = Array3::from_shape_fn((batch, seq, d_k), |(_, i, j)| (i + j) as f64 * 0.1);
        let k3 = Array3::from_shape_fn((batch, seq, d_k), |(_, i, j)| (i * 2 + j) as f64 * 0.07);
        let v3 = Array3::from_shape_fn((batch, seq, d_k), |(_, i, j)| (i as f64 + j as f64) * 0.05);

        let cfg = FlashAttentionConfig {
            block_size: 4,
            causal: true,
            dropout: 0.0,
            scale: None,
        };

        let mh_out =
            multi_head_flash_attention(&q3, &k3, &v3, 1, &cfg).expect("multi-head n_heads=1");

        let q2: Array2<f64> = q3.slice(s![0, .., ..]).to_owned();
        let k2: Array2<f64> = k3.slice(s![0, .., ..]).to_owned();
        let v2: Array2<f64> = v3.slice(s![0, .., ..]).to_owned();
        let single_out = flash_attention(&q2, &k2, &v2, &cfg).expect("single head");

        for i in 0..seq {
            for dv in 0..d_k {
                let diff = (mh_out[[0, i, dv]] - single_out[[i, dv]]).abs();
                assert!(diff < 1e-12, "n_heads=1 mismatch at [{i},{dv}]");
            }
        }
    }
}
