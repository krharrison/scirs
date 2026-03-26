//! Hierarchical sparse attention for ultra-long time series sequences.
//!
//! Standard self-attention is O(N²) in time and memory.  This module implements
//! a three-level hierarchical decomposition that achieves sub-quadratic complexity
//! while preserving both local fine-grained and global coarse-grained dependencies:
//!
//! | Level | Mechanism | Complexity |
//! |-------|-----------|------------|
//! | 0 (fine) | Local window attention | O(N · W) |
//! | 1 (medium) | Strided pooling + full attention | O(N/s · N/s) |
//! | 2 (coarse) | Global CLS-like tokens | O(G · N) |
//!
//! The outputs from all three levels are combined with a fixed mixing weight.
//!
//! ## Key functions
//!
//! - [`hierarchical_attention`] — top-level entry point
//! - [`local_window_attention`] — level-0 fine attention
//! - [`pooled_attention`] — level-1 medium attention with average pooling
//! - [`global_token_attention`] — level-2 coarse global-token attention

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the hierarchical sparse attention module.
#[derive(Debug, Clone)]
pub struct HierarchicalAttnConfig {
    /// Width of the local attention window (level 0).
    pub window_size: usize,
    /// Stride used when pooling for medium-level attention (level 1).
    pub stride: usize,
    /// Number of hierarchical levels to compute (1–3).
    pub n_levels: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// Number of global (CLS-like) tokens for coarse attention (level 2).
    pub global_tokens: usize,
}

impl Default for HierarchicalAttnConfig {
    fn default() -> Self {
        Self {
            window_size: 64,
            stride: 32,
            n_levels: 3,
            n_heads: 4,
            head_dim: 32,
            global_tokens: 4,
        }
    }
}

/// A single sparse attention level (stored for introspection / debugging).
#[derive(Debug, Clone)]
pub struct SparseAttentionLevel {
    /// Window size used at this level.
    pub window_size: usize,
    /// Stride used at this level.
    pub stride: usize,
    /// Query vectors (shape: seq_len × model_dim).
    pub queries: Vec<Vec<f64>>,
    /// Key vectors.
    pub keys: Vec<Vec<f64>>,
    /// Value vectors.
    pub values: Vec<Vec<f64>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: scaled dot-product attention for a slice of Q/K/V
// ─────────────────────────────────────────────────────────────────────────────

/// Compute softmax over a mutable slice (in-place, numerically stable).
fn softmax_inplace(v: &mut [f64]) {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0f64;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 1e-30 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// Single-head scaled dot-product attention.
///
/// `queries`: Q_len × d_k
/// `keys`:    KV_len × d_k
/// `values`:  KV_len × d_v
///
/// Returns output of shape Q_len × d_v.
fn scaled_dot_product(
    queries: &[Vec<f64>],
    keys: &[Vec<f64>],
    values: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let q_len = queries.len();
    let kv_len = keys.len();
    if q_len == 0 || kv_len == 0 {
        return queries.to_vec();
    }
    let d_k = queries[0].len().max(1);
    let d_v = if values.is_empty() || values[0].is_empty() {
        1
    } else {
        values[0].len()
    };
    let scale = (d_k as f64).sqrt().max(1e-8);

    let mut output = vec![vec![0.0f64; d_v]; q_len];
    let mut scores = vec![0.0f64; kv_len];

    for (qi, q) in queries.iter().enumerate() {
        // Compute raw scores
        for (ki, k) in keys.iter().enumerate() {
            let dot: f64 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
            scores[ki] = dot / scale;
        }
        // Softmax
        softmax_inplace(&mut scores[..kv_len]);
        // Weighted sum of values
        for (ki, &w) in scores[..kv_len].iter().enumerate() {
            if ki < values.len() {
                for (d, val) in values[ki].iter().enumerate() {
                    if d < output[qi].len() {
                        output[qi][d] += w * val;
                    }
                }
            }
        }
    }
    output
}

/// Project a sequence of vectors by taking every `head_dim` elements
/// starting at `offset` for the given head index.
fn extract_head(x: &[Vec<f64>], head: usize, head_dim: usize) -> Vec<Vec<f64>> {
    let start = head * head_dim;
    x.iter()
        .map(|row| {
            let end = (start + head_dim).min(row.len());
            if start < row.len() {
                row[start..end].to_vec()
            } else {
                vec![0.0; head_dim]
            }
        })
        .collect()
}

/// Merge per-head outputs back into a single sequence by concatenation.
fn merge_heads(head_outputs: Vec<Vec<Vec<f64>>>, seq_len: usize) -> Vec<Vec<f64>> {
    let n_heads = head_outputs.len();
    if n_heads == 0 || seq_len == 0 {
        return vec![];
    }
    let head_dim = head_outputs[0].first().map(|v| v.len()).unwrap_or(0);
    let total_dim = n_heads * head_dim;
    let mut merged = vec![vec![0.0f64; total_dim]; seq_len];
    for (h, head_out) in head_outputs.iter().enumerate() {
        for (i, row) in head_out.iter().enumerate() {
            if i < seq_len {
                for (d, val) in row.iter().enumerate() {
                    merged[i][h * head_dim + d] = *val;
                }
            }
        }
    }
    merged
}

// ─────────────────────────────────────────────────────────────────────────────
// Level 0: Local window attention
// ─────────────────────────────────────────────────────────────────────────────

/// Level-0 local window attention.
///
/// For each position `i`, attends only to tokens in the window
/// `[max(0, i - window/2), min(n, i + window/2)]`.
/// Complexity: O(N · W) where W = `window`.
///
/// Supports multi-head by splitting the feature dimension.
pub fn local_window_attention(
    x: &[Vec<f64>],
    window: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<Vec<f64>> {
    let seq_len = x.len();
    if seq_len == 0 {
        return vec![];
    }
    let n_heads = n_heads.max(1);
    let half_w = window / 2;

    let mut head_outputs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_heads);

    for h in 0..n_heads {
        let head_x = extract_head(x, h, head_dim);
        let mut head_out = vec![vec![0.0f64; head_dim]; seq_len];

        for i in 0..seq_len {
            let lo = i.saturating_sub(half_w);
            let hi = (i + half_w + 1).min(seq_len);

            let q_row = &head_x[i];
            let keys_slice = &head_x[lo..hi];
            let vals_slice = &head_x[lo..hi];

            let scores: Vec<f64> = {
                let d_k = q_row.len().max(1) as f64;
                let scale = d_k.sqrt();
                keys_slice
                    .iter()
                    .map(|k| q_row.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f64>() / scale)
                    .collect()
            };
            let mut weights = scores;
            softmax_inplace(&mut weights);

            let out_row = &mut head_out[i];
            for (j, &w) in weights.iter().enumerate() {
                for (d, val) in vals_slice[j].iter().enumerate() {
                    if d < out_row.len() {
                        out_row[d] += w * val;
                    }
                }
            }
        }
        head_outputs.push(head_out);
    }

    merge_heads(head_outputs, seq_len)
}

// ─────────────────────────────────────────────────────────────────────────────
// Level 1: Pooled attention
// ─────────────────────────────────────────────────────────────────────────────

/// Level-1 strided pooling + full attention + upsample.
///
/// 1. Average-pool `x` from length N down to `ceil(N/pool_stride)`.
/// 2. Run full multi-head attention on the pooled sequence.
/// 3. Upsample back to N by repeating each pooled output `pool_stride` times.
///
/// Complexity: O((N/s)²).
pub fn pooled_attention(
    x: &[Vec<f64>],
    pool_stride: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<Vec<f64>> {
    let seq_len = x.len();
    if seq_len == 0 {
        return vec![];
    }
    let stride = pool_stride.max(1);
    let feat_dim = x[0].len();

    // 1. Average pool
    let pooled_len = (seq_len + stride - 1) / stride;
    let mut pooled = vec![vec![0.0f64; feat_dim]; pooled_len];
    let mut counts = vec![0usize; pooled_len];
    for (i, row) in x.iter().enumerate() {
        let pool_idx = i / stride;
        if pool_idx < pooled_len {
            for (d, v) in row.iter().enumerate() {
                if d < feat_dim {
                    pooled[pool_idx][d] += v;
                }
            }
            counts[pool_idx] += 1;
        }
    }
    for (row, &cnt) in pooled.iter_mut().zip(counts.iter()) {
        if cnt > 1 {
            for v in row.iter_mut() {
                *v /= cnt as f64;
            }
        }
    }

    // 2. Full multi-head attention on pooled
    let n_heads = n_heads.max(1);
    let mut attended = vec![vec![0.0f64; n_heads * head_dim]; pooled_len];

    let mut head_outputs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_heads);
    for h in 0..n_heads {
        let head_pooled = extract_head(&pooled, h, head_dim);
        let head_attn = scaled_dot_product(&head_pooled, &head_pooled, &head_pooled);
        head_outputs.push(head_attn);
    }
    let attended_full = merge_heads(head_outputs, pooled_len);
    for (i, row) in attended_full.into_iter().enumerate() {
        if i < attended.len() {
            attended[i] = row;
        }
    }

    // 3. Upsample back to seq_len
    let out_dim = attended[0].len();
    let mut output = vec![vec![0.0f64; out_dim]; seq_len];
    for (i, out_row) in output.iter_mut().enumerate() {
        let pool_idx = (i / stride).min(pooled_len.saturating_sub(1));
        *out_row = attended[pool_idx].clone();
    }
    output
}

// ─────────────────────────────────────────────────────────────────────────────
// Level 2: Global token attention
// ─────────────────────────────────────────────────────────────────────────────

/// Level-2 global-token (CLS-like) attention.
///
/// `n_global` fixed query tokens attend to the full sequence.
/// Each position in the sequence then also attends to all global tokens
/// (bidirectional broadcast).
///
/// Complexity: O(G · N).
pub fn global_token_attention(
    x: &[Vec<f64>],
    n_global: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<Vec<f64>> {
    let seq_len = x.len();
    if seq_len == 0 || n_global == 0 {
        return x.to_vec();
    }
    let n_heads = n_heads.max(1);
    let feat_dim = x[0].len();

    // Create global query tokens as the mean of evenly-spaced sequence positions
    let mut global_queries: Vec<Vec<f64>> = Vec::with_capacity(n_global);
    for g in 0..n_global {
        let idx = (g * seq_len) / n_global;
        global_queries.push(x[idx.min(seq_len - 1)].clone());
    }

    // 1. Global queries attend to full sequence → global context vectors
    let n_heads_clamped = n_heads.max(1);
    let mut global_context = vec![vec![0.0f64; n_heads_clamped * head_dim]; n_global];
    let mut g_head_outputs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_heads_clamped);
    for h in 0..n_heads_clamped {
        let q_h = extract_head(&global_queries, h, head_dim);
        let k_h = extract_head(x, h, head_dim);
        let v_h = extract_head(x, h, head_dim);
        g_head_outputs.push(scaled_dot_product(&q_h, &k_h, &v_h));
    }
    let global_full = merge_heads(g_head_outputs, n_global);
    for (i, row) in global_full.into_iter().enumerate() {
        if i < global_context.len() {
            global_context[i] = row;
        }
    }

    // 2. Each sequence position attends to global context tokens
    let mut seq_head_outputs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_heads_clamped);
    for h in 0..n_heads_clamped {
        let q_h = extract_head(x, h, head_dim);
        let k_h = extract_head(&global_context, h, head_dim);
        let v_h = extract_head(&global_context, h, head_dim);
        seq_head_outputs.push(scaled_dot_product(&q_h, &k_h, &v_h));
    }
    merge_heads(seq_head_outputs, seq_len)
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level: hierarchical attention
// ─────────────────────────────────────────────────────────────────────────────

/// Combine outputs from multiple levels using fixed mixing weights.
///
/// Weights: level-0 = 0.5, level-1 = 0.3, level-2 = 0.2 (normalised to sum 1).
fn mix_levels(outputs: &[Vec<Vec<f64>>], seq_len: usize) -> Vec<Vec<f64>> {
    if outputs.is_empty() || seq_len == 0 {
        return vec![];
    }
    // Fixed mixing weights (sum = 1.0)
    let weights = [0.5f64, 0.3, 0.2];
    let sum_w: f64 = weights[..outputs.len()].iter().sum();

    let out_dim = outputs[0].first().map(|r| r.len()).unwrap_or(0);
    let mut mixed = vec![vec![0.0f64; out_dim]; seq_len];

    for (level_idx, level_out) in outputs.iter().enumerate() {
        let w = weights[level_idx.min(weights.len() - 1)] / sum_w;
        for (i, row) in level_out.iter().enumerate() {
            if i < seq_len {
                for (d, val) in row.iter().enumerate() {
                    if d < out_dim {
                        mixed[i][d] += w * val;
                    }
                }
            }
        }
    }
    mixed
}

/// Pad or truncate each row of a sequence to `target_dim`.
fn pad_or_truncate(x: Vec<Vec<f64>>, target_dim: usize) -> Vec<Vec<f64>> {
    x.into_iter()
        .map(|mut row| {
            row.resize(target_dim, 0.0);
            row
        })
        .collect()
}

/// Multi-level hierarchical sparse attention.
///
/// # Arguments
/// * `x` — input sequence of shape `(seq_len, model_dim)`
/// * `config` — attention configuration
///
/// # Returns
/// Output sequence of shape `(seq_len, n_heads * head_dim)`.
pub fn hierarchical_attention(x: &[Vec<f64>], config: &HierarchicalAttnConfig) -> Vec<Vec<f64>> {
    let seq_len = x.len();
    if seq_len == 0 {
        return vec![];
    }
    let out_dim = config.n_heads * config.head_dim;

    let mut level_outputs: Vec<Vec<Vec<f64>>> = Vec::new();
    let n_levels = config.n_levels.clamp(1, 3);

    // Level 0: local window
    if n_levels >= 1 {
        let window = config.window_size.max(1);
        let l0 = local_window_attention(x, window, config.n_heads, config.head_dim);
        level_outputs.push(pad_or_truncate(l0, out_dim));
    }

    // Level 1: pooled
    if n_levels >= 2 {
        let stride = config.stride.max(1);
        let l1 = pooled_attention(x, stride, config.n_heads, config.head_dim);
        level_outputs.push(pad_or_truncate(l1, out_dim));
    }

    // Level 2: global tokens
    if n_levels >= 3 {
        let l2 = global_token_attention(x, config.global_tokens, config.n_heads, config.head_dim);
        level_outputs.push(pad_or_truncate(l2, out_dim));
    }

    mix_levels(&level_outputs, seq_len)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sequence(seq_len: usize, feat_dim: usize) -> Vec<Vec<f64>> {
        (0..seq_len)
            .map(|i| {
                (0..feat_dim)
                    .map(|d| (i * feat_dim + d) as f64 / 100.0)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_local_window_attention_shape() {
        let seq_len = 20;
        let feat_dim = 8;
        let x = make_sequence(seq_len, feat_dim);
        let out = local_window_attention(&x, 4, 2, 4);
        assert_eq!(
            out.len(),
            seq_len,
            "output sequence length should match input"
        );
        assert_eq!(
            out[0].len(),
            2 * 4,
            "output dim should be n_heads * head_dim"
        );
    }

    #[test]
    fn test_pooled_attention_shape() {
        let seq_len = 16;
        let feat_dim = 8;
        let x = make_sequence(seq_len, feat_dim);
        let out = pooled_attention(&x, 4, 2, 4);
        assert_eq!(
            out.len(),
            seq_len,
            "pooled attention should restore seq length"
        );
        assert_eq!(
            out[0].len(),
            2 * 4,
            "output dim should be n_heads * head_dim"
        );
    }

    #[test]
    fn test_global_token_attention_shape() {
        let seq_len = 12;
        let feat_dim = 8;
        let x = make_sequence(seq_len, feat_dim);
        let out = global_token_attention(&x, 2, 2, 4);
        assert_eq!(
            out.len(),
            seq_len,
            "global token attention output length should match input"
        );
        assert_eq!(
            out[0].len(),
            2 * 4,
            "output dim should be n_heads * head_dim"
        );
    }

    #[test]
    fn test_hierarchical_attention_default_config() {
        let seq_len = 32;
        let config = HierarchicalAttnConfig {
            window_size: 8,
            stride: 4,
            n_levels: 3,
            n_heads: 2,
            head_dim: 4,
            global_tokens: 2,
        };
        let x = make_sequence(seq_len, config.n_heads * config.head_dim);
        let out = hierarchical_attention(&x, &config);
        assert_eq!(out.len(), seq_len);
        assert_eq!(out[0].len(), config.n_heads * config.head_dim);
    }

    #[test]
    fn test_hierarchical_attention_empty_input() {
        let config = HierarchicalAttnConfig::default();
        let out = hierarchical_attention(&[], &config);
        assert!(out.is_empty(), "empty input should produce empty output");
    }

    #[test]
    fn test_hierarchical_attention_single_level() {
        let seq_len = 10;
        let config = HierarchicalAttnConfig {
            n_levels: 1,
            window_size: 4,
            n_heads: 1,
            head_dim: 4,
            ..Default::default()
        };
        let x = make_sequence(seq_len, 4);
        let out = hierarchical_attention(&x, &config);
        assert_eq!(out.len(), seq_len);
    }
}
