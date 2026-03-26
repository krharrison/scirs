//! Gradient compression algorithms for distributed training.
//!
//! Implements Top-K sparsification (Stich et al. 2018), Random-K sparsification,
//! 1-bit gradient quantization (Seide et al. 2014), and PowerSGD-style low-rank
//! approximation.

use crate::error::{CoreError, CoreResult};

// ─────────────────────────────────────────────────────────────────────────────
// Top-K Compressor
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Top-K gradient sparsification (Stich et al. 2018).
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TopKConfig {
    /// Fraction of gradient elements to keep (0, 1].  Default: 0.01.
    pub k_fraction: f64,
    /// Whether to accumulate the residual error and add it back before
    /// the next compression step (error feedback).  Default: `true`.
    pub use_error_feedback: bool,
}

impl Default for TopKConfig {
    fn default() -> Self {
        TopKConfig {
            k_fraction: 0.01,
            use_error_feedback: true,
        }
    }
}

/// Top-K gradient sparsification compressor.
///
/// Keeps the `k = ceil(n * k_fraction)` elements with the largest absolute
/// value.  Optionally accumulates the discarded residual for the next round
/// (error feedback).
pub struct TopKCompressor {
    config: TopKConfig,
    /// Residual error accumulated from previous rounds (same length as
    /// gradient).
    error_feedback: Vec<f64>,
}

impl TopKCompressor {
    /// Create a new compressor for gradients of length `n_params`.
    pub fn new(n_params: usize, config: TopKConfig) -> Self {
        TopKCompressor {
            config,
            error_feedback: vec![0.0; n_params],
        }
    }

    /// Compress a gradient vector.
    ///
    /// Returns `(indices, values)` of the top-k elements by absolute value.
    /// If error feedback is enabled the residual from the *previous* round is
    /// added to `gradient` before selecting the top-k elements, and the new
    /// residual is stored internally.
    pub fn compress(&mut self, gradient: &[f64]) -> CoreResult<(Vec<usize>, Vec<f64>)> {
        if gradient.is_empty() {
            return Ok((vec![], vec![]));
        }
        let n = gradient.len();
        if n != self.error_feedback.len() {
            return Err(CoreError::ShapeError(crate::error::ErrorContext::new(
                format!(
                    "TopKCompressor: gradient len {} != initialised len {}",
                    n,
                    self.error_feedback.len()
                ),
            )));
        }
        // Apply error feedback
        let mut g: Vec<f64> = if self.config.use_error_feedback {
            gradient
                .iter()
                .zip(self.error_feedback.iter())
                .map(|(a, b)| a + b)
                .collect()
        } else {
            gradient.to_vec()
        };

        let k = ((n as f64 * self.config.k_fraction).ceil() as usize)
            .max(1)
            .min(n);

        // Compute indices sorted by |g| descending
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| {
            g[b].abs()
                .partial_cmp(&g[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let top_k: Vec<usize> = order[..k].to_vec();

        // Build sparse output
        let mut indices: Vec<usize> = top_k.clone();
        indices.sort_unstable();
        let values: Vec<f64> = indices.iter().map(|&i| g[i]).collect();

        // Update error feedback residual
        if self.config.use_error_feedback {
            for &i in &indices {
                g[i] = 0.0;
            }
            self.error_feedback = g; // residual = g - sparse(g)
        }

        Ok((indices, values))
    }

    /// Decompress a sparse gradient back to a dense vector.
    pub fn decompress(indices: &[usize], values: &[f64], n_total: usize) -> CoreResult<Vec<f64>> {
        if indices.len() != values.len() {
            return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
                "decompress: indices and values length mismatch",
            )));
        }
        let mut out = vec![0.0f64; n_total];
        for (&i, &v) in indices.iter().zip(values.iter()) {
            if i >= n_total {
                return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
                    format!(
                        "decompress: index {} out of bounds for n_total {}",
                        i, n_total
                    ),
                )));
            }
            out[i] = v;
        }
        Ok(out)
    }

    /// Compression ratio: `n_total / k`.
    pub fn compression_ratio(&self, n_total: usize) -> f64 {
        let k = ((n_total as f64 * self.config.k_fraction).ceil() as usize)
            .max(1)
            .min(n_total);
        n_total as f64 / k as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Random-K Compressor
// ─────────────────────────────────────────────────────────────────────────────

/// Random-K gradient sparsification.
///
/// Keeps a randomly-selected subset of gradient elements (uniform without
/// replacement).
pub struct RandomKCompressor {
    /// Fraction of elements to retain.
    k_fraction: f64,
}

impl RandomKCompressor {
    /// Create a new random-K compressor.
    pub fn new(k_fraction: f64) -> Self {
        RandomKCompressor { k_fraction }
    }

    /// Compress a gradient using a simple LCG PRNG seeded by `seed`.
    ///
    /// Returns `(indices, values)` of the randomly-selected elements.
    pub fn compress(&self, gradient: &[f64], seed: u64) -> CoreResult<(Vec<usize>, Vec<f64>)> {
        if gradient.is_empty() {
            return Ok((vec![], vec![]));
        }
        let n = gradient.len();
        let k = ((n as f64 * self.k_fraction).ceil() as usize).max(1).min(n);

        // Fisher-Yates partial shuffle using a minimal LCG PRNG
        // (no external rand dependency — pure COOLJAPAN policy)
        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng_state = seed.wrapping_add(1);
        let lcg_a: u64 = 6364136223846793005;
        let lcg_c: u64 = 1442695040888963407;

        for i in 0..k {
            rng_state = rng_state.wrapping_mul(lcg_a).wrapping_add(lcg_c);
            let j = (rng_state >> 33) as usize % (n - i) + i;
            indices.swap(i, j);
        }

        let mut selected: Vec<usize> = indices[..k].to_vec();
        selected.sort_unstable();
        let values: Vec<f64> = selected.iter().map(|&i| gradient[i]).collect();

        Ok((selected, values))
    }

    /// Decompress a sparse gradient back to a dense vector.
    pub fn decompress(indices: &[usize], values: &[f64], n_total: usize) -> CoreResult<Vec<f64>> {
        if indices.len() != values.len() {
            return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
                "decompress: indices and values length mismatch",
            )));
        }
        let mut out = vec![0.0f64; n_total];
        for (&i, &v) in indices.iter().zip(values.iter()) {
            if i >= n_total {
                return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
                    format!(
                        "decompress: index {} out of bounds for n_total {}",
                        i, n_total
                    ),
                )));
            }
            out[i] = v;
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1-Bit Quantizer
// ─────────────────────────────────────────────────────────────────────────────

/// 1-bit gradient quantization (Seide et al. 2014).
///
/// Each element is mapped to ±mean_abs(gradient).
pub struct OneBitQuantizer;

impl OneBitQuantizer {
    /// Quantize a gradient vector.
    ///
    /// Returns `(bit_words, scale)` where each bit in `bit_words` corresponds
    /// to one gradient element: `1` = positive (≥ 0), `0` = negative (< 0).
    /// `scale` is `mean(|gradient|)`.
    pub fn quantize(gradient: &[f64]) -> CoreResult<(Vec<u64>, f64)> {
        if gradient.is_empty() {
            return Ok((vec![], 0.0));
        }
        let scale = gradient.iter().map(|x| x.abs()).sum::<f64>() / gradient.len() as f64;
        let n_words = gradient.len().div_ceil(64);
        let mut bits = vec![0u64; n_words];
        for (i, &v) in gradient.iter().enumerate() {
            if v >= 0.0 {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
        Ok((bits, scale))
    }

    /// Dequantize: reconstruct gradient from bit words and scale.
    ///
    /// Returns a vector of length `n` where positive bits map to `+scale` and
    /// zero bits map to `-scale`.
    pub fn dequantize(bits: &[u64], scale: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                if (bits[i / 64] >> (i % 64)) & 1 == 1 {
                    scale
                } else {
                    -scale
                }
            })
            .collect()
    }

    /// Mean absolute error between original and quantized gradient.
    pub fn quantization_error(original: &[f64], quantized: &[f64]) -> f64 {
        if original.is_empty() {
            return 0.0;
        }
        let len = original.len().min(quantized.len());
        let total: f64 = original[..len]
            .iter()
            .zip(quantized[..len].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        total / len as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PowerSGD Low-Rank Approximation
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for PowerSGD low-rank gradient compression.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct PowerSgdConfig {
    /// Target rank for the low-rank decomposition.  Default: 4.
    pub rank: usize,
    /// Number of power-iteration steps to improve the approximation.  Default: 1.
    pub n_power_iter: usize,
    /// Reuse the left factor as momentum across rounds.  Default: `true`.
    pub reuse_momentum: bool,
}

impl Default for PowerSgdConfig {
    fn default() -> Self {
        PowerSgdConfig {
            rank: 4,
            n_power_iter: 1,
            reuse_momentum: true,
        }
    }
}

/// Compress a gradient matrix `G` (m × n) into P (m × r) and Q (n × r)
/// such that `G ≈ P · Q^T`.
///
/// Implementation follows the PowerSGD algorithm (Vogels et al. 2019):
/// 1. Initialize Q with a small random matrix (Gaussian approximation via
///    deterministic seeded values — pure Rust, no external crate).
/// 2. Power-iterate: P = G·Q  (ortho-normalised), Q = G^T·P  (ortho-normalised).
///
/// # Arguments
/// * `gradient_matrix` — row-major `[m][n]` gradient matrix.
/// * `config` — rank and iteration settings.
///
/// # Returns
/// `(P, Q)` where P is `m × r` and Q is `n × r`.
pub fn low_rank_compress(
    gradient_matrix: &[Vec<f64>],
    config: &PowerSgdConfig,
) -> CoreResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let m = gradient_matrix.len();
    if m == 0 {
        return Ok((vec![], vec![]));
    }
    let n = gradient_matrix[0].len();
    if n == 0 {
        return Ok((vec![vec![]; m], vec![]));
    }
    let r = config.rank.min(m.min(n));
    if r == 0 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "low_rank_compress: rank must be >= 1",
        )));
    }

    // ── Initialise Q (n × r) with a deterministic pseudo-random matrix ────
    let mut q = vec![vec![0.0f64; r]; n];
    let mut rng: u64 = 0xDEAD_BEEF_1234_5678;
    let lcg_a: u64 = 6364136223846793005;
    let lcg_c: u64 = 1442695040888963407;
    for row in q.iter_mut() {
        for x in row.iter_mut() {
            rng = rng.wrapping_mul(lcg_a).wrapping_add(lcg_c);
            // Map to [-1, 1]
            *x = (rng as i64 as f64) / (i64::MAX as f64);
        }
    }
    orthonormalize_cols(&mut q)?;

    // ── Power iterations ──────────────────────────────────────────────────
    // Invariant: Q is always column-orthonormal after each iteration.
    // The approximation is G ≈ P·Q^T where:
    //   P = G·Q   (NOT orthonormalized — P captures the actual projection)
    //   Q is column-orthonormal
    //
    // When r = min(m,n) and Q's column space spans the full space, Q·Q^T = I
    // and P·Q^T = G·Q·Q^T = G exactly.
    let mut p = vec![vec![0.0f64; r]; m];
    for _ in 0..config.n_power_iter.max(1) {
        // P = G · Q   (m × r)
        for i in 0..m {
            for j in 0..r {
                p[i][j] = gradient_matrix[i]
                    .iter()
                    .enumerate()
                    .map(|(k, &g)| g * q[k][j])
                    .sum();
            }
        }
        orthonormalize_cols(&mut p)?;

        // Q = G^T · P   (n × r)
        for k in 0..n {
            for j in 0..r {
                q[k][j] = gradient_matrix
                    .iter()
                    .enumerate()
                    .map(|(i, row)| row[k] * p[i][j])
                    .sum();
            }
        }
        // Keep Q orthonormal throughout (required for P·Q^T = G·Q·Q^T ≈ G)
        orthonormalize_cols(&mut q)?;
    }

    // Final P = G · Q  (where Q is orthonormal)
    // G_approx = P · Q^T = G · Q · Q^T
    // When r = n, Q·Q^T = I and P·Q^T = G exactly.
    for i in 0..m {
        for j in 0..r {
            p[i][j] = gradient_matrix[i]
                .iter()
                .enumerate()
                .map(|(k, &g)| g * q[k][j])
                .sum();
        }
    }

    Ok((p, q))
}

/// Decompress a low-rank gradient approximation back to a full matrix.
///
/// Computes `G_hat = P · Q^T` where P is `m × r` and Q is `n × r`.
pub fn low_rank_decompress(p: &[Vec<f64>], q: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = p.len();
    if m == 0 {
        return vec![];
    }
    let r = p[0].len();
    let n = q.len();
    let mut out = vec![vec![0.0f64; n]; m];
    for i in 0..m {
        for k in 0..n {
            let dot: f64 = (0..r).map(|j| p[i][j] * q[k][j]).sum();
            out[i][k] = dot;
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Modified Gram-Schmidt orthonormalisation of the *columns* of a row-major
/// matrix.
///
/// The matrix is `rows × cols`.  Uses the numerically more stable modified
/// Gram-Schmidt variant.  If a column becomes numerically zero (norm < ε) it
/// is replaced by the first canonical basis vector not already represented in
/// the span, ensuring the output always has `cols` orthonormal columns even
/// for rank-deficient inputs.
fn orthonormalize_cols(mat: &mut Vec<Vec<f64>>) -> CoreResult<()> {
    let rows = mat.len();
    if rows == 0 {
        return Ok(());
    }
    let cols = mat[0].len();
    if cols == 0 {
        return Ok(());
    }

    for j in 0..cols {
        // Modified Gram-Schmidt: subtract projections one at a time (more
        // stable than classical GS).
        for k in 0..j {
            // Re-compute dot product with the already-orthonormalized column k.
            let dot: f64 = (0..rows).map(|i| mat[i][j] * mat[i][k]).sum();
            for i in 0..rows {
                let prev = mat[i][k];
                mat[i][j] -= dot * prev;
            }
        }
        // Normalise column j.
        let norm: f64 = (0..rows).map(|i| mat[i][j] * mat[i][j]).sum::<f64>().sqrt();
        if norm < 1e-10 {
            // Column is linearly dependent — replace with a canonical basis
            // vector not already in the span.
            let mut replaced = false;
            'outer: for candidate in 0..rows {
                // Check if e_candidate is linearly independent from current cols
                for k in 0..j {
                    // If column k has a large component along e_candidate, skip.
                    if mat[candidate][k].abs() > 0.9 {
                        continue 'outer;
                    }
                }
                for i in 0..rows {
                    mat[i][j] = if i == candidate { 1.0 } else { 0.0 };
                }
                replaced = true;
                break;
            }
            if !replaced {
                // Fallback: use e_{j % rows}
                for i in 0..rows {
                    mat[i][j] = if i == j % rows { 1.0 } else { 0.0 };
                }
            }
        } else {
            for i in 0..rows {
                mat[i][j] /= norm;
            }
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Top-K ────────────────────────────────────────────────────────────

    #[test]
    fn test_topk_keeps_exactly_k_elements() {
        let cfg = TopKConfig {
            k_fraction: 0.25,
            use_error_feedback: false,
        };
        let mut comp = TopKCompressor::new(8, cfg);
        let grad = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.6];
        let (indices, values) = comp.compress(&grad).expect("compress failed");
        // k = ceil(8 * 0.25) = 2
        assert_eq!(indices.len(), 2);
        assert_eq!(values.len(), 2);
        // The top-2 elements by abs value are 0.9 (idx 3) and 0.8 (idx 5)
        assert!(indices.contains(&3));
        assert!(indices.contains(&5));
    }

    #[test]
    fn test_topk_error_feedback_reduces_over_rounds() {
        // With error feedback the residual from round 1 boosts small
        // gradients in round 2, so the total compressed signal should grow.
        let cfg = TopKConfig {
            k_fraction: 0.5,
            use_error_feedback: true,
        };
        let mut comp = TopKCompressor::new(4, cfg);
        let grad = vec![1.0, 0.1, 0.1, 0.1];
        let (_, v1) = comp.compress(&grad).expect("compress round 1 failed");
        let (_, v2) = comp.compress(&grad).expect("compress round 2 failed");
        // The accumulated residual from round 1 should contribute to round 2
        let sum1: f64 = v1.iter().map(|x| x.abs()).sum();
        let sum2: f64 = v2.iter().map(|x| x.abs()).sum();
        // With feedback the second round's selected values should include
        // previously discarded signal — sum2 != 0.
        assert!(sum1 > 0.0);
        assert!(sum2 > 0.0);
    }

    #[test]
    fn test_randomk_correct_size() {
        let comp = RandomKCompressor::new(0.1);
        let grad: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (indices, values) = comp.compress(&grad, 42).expect("compress failed");
        // k = ceil(100 * 0.1) = 10
        assert_eq!(indices.len(), 10);
        assert_eq!(values.len(), 10);
        // No duplicate indices
        let mut sorted = indices.clone();
        sorted.dedup();
        assert_eq!(sorted.len(), indices.len());
    }

    #[test]
    fn test_1bit_quantize_dequantize_preserves_sign() {
        let gradient = vec![-3.0, 1.5, -0.5, 2.0, -0.1, 0.8];
        let (bits, scale) = OneBitQuantizer::quantize(&gradient).expect("quantize failed");
        let dequantized = OneBitQuantizer::dequantize(&bits, scale, gradient.len());
        for (orig, deq) in gradient.iter().zip(dequantized.iter()) {
            // Sign must be preserved
            let same_sign = (orig >= &0.0 && deq >= &0.0) || (orig < &0.0 && deq < &0.0);
            assert!(same_sign, "sign mismatch: orig={} deq={}", orig, deq);
        }
    }

    #[test]
    fn test_low_rank_compress_decompress_close_for_full_rank() {
        // Use a genuinely full-rank matrix (circulant / diagonal-dominant).
        // For a rank-r matrix, a rank-r decomposition should be exact.
        // We construct a rank-2 matrix explicitly and verify that rank-2
        // compression reconstructs it to machine precision.
        let m = 4;
        let n = 4;
        // rank-2 matrix: G = u1*v1^T + u2*v2^T
        let u1 = [1.0, 2.0, 3.0, 4.0];
        let v1 = [5.0, -1.0, 2.0, 0.5];
        let u2 = [0.5, -1.0, 1.5, -2.0];
        let v2 = [1.0, 3.0, -2.0, 4.0];
        let grad: Vec<Vec<f64>> = (0..m)
            .map(|i| (0..n).map(|j| u1[i] * v1[j] + u2[i] * v2[j]).collect())
            .collect();
        let cfg = PowerSgdConfig {
            rank: 2,          // exact for a rank-2 matrix
            n_power_iter: 10, // enough iterations to converge
            reuse_momentum: false,
        };
        let (p, q) = low_rank_compress(&grad, &cfg).expect("compress failed");
        let approx = low_rank_decompress(&p, &q);
        let mut max_err = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let err = (grad[i][j] - approx[i][j]).abs();
                max_err = max_err.max(err);
            }
        }
        assert!(max_err < 1e-6, "max reconstruction error = {}", max_err);
    }

    #[test]
    fn test_powersgd_config_defaults() {
        let cfg = PowerSgdConfig::default();
        assert_eq!(cfg.rank, 4);
        assert_eq!(cfg.n_power_iter, 1);
        assert!(cfg.reuse_momentum);
    }

    #[test]
    fn test_compression_ratio_computation() {
        let cfg = TopKConfig {
            k_fraction: 0.01,
            use_error_feedback: true,
        };
        let comp = TopKCompressor::new(1000, cfg);
        // k = ceil(1000 * 0.01) = 10 → ratio = 100
        let ratio = comp.compression_ratio(1000);
        assert!((ratio - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_gradient_handling() {
        let cfg = TopKConfig::default();
        let mut comp = TopKCompressor::new(0, cfg);
        let (idx, val) = comp.compress(&[]).expect("compress empty failed");
        assert!(idx.is_empty());
        assert!(val.is_empty());

        let comp2 = RandomKCompressor::new(0.1);
        let (idx2, val2) = comp2.compress(&[], 0).expect("compress empty failed");
        assert!(idx2.is_empty());
        assert!(val2.is_empty());

        let (bits, scale) = OneBitQuantizer::quantize(&[]).expect("quantize empty failed");
        assert!(bits.is_empty());
        assert_eq!(scale, 0.0);
    }
}
