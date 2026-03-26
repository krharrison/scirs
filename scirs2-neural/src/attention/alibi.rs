//! ALiBi — Attention with Linear Biases.
//!
//! ALiBi (Press et al., 2021, "Train Short, Test Long: Attention with Linear
//! Biases Enables Input Length Extrapolation") replaces positional encodings
//! with a simple bias added to the attention logits before softmax:
//!
//! ```text
//! score(q_i, k_j)  =  q_i · k_j / sqrt(d)  −  m_h * |i − j|
//! ```
//!
//! where `m_h` is a head-specific slope. The slopes follow a geometric
//! sequence: `m_h = 2^{-8h/H}` for `h = 1 … H`.
//!
//! Because the bias depends only on the *distance* `|i − j|`, the model can
//! extrapolate to longer sequences at inference time without any change to the
//! weights.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_neural::attention::alibi::{AlibiConfig, AlibiSlopes};
//!
//! let cfg = AlibiConfig::default(); // num_heads=8, max_seq_len=2048
//! let slopes = AlibiSlopes::new(cfg);
//!
//! let seq_len = 4;
//! let num_heads = 8;
//! let bias = slopes.get_bias_matrix(seq_len);
//! assert_eq!(bias.len(), num_heads);
//! assert_eq!(bias[0].len(), seq_len * seq_len);
//! ```

/// Configuration for ALiBi position encoding.
///
/// # Defaults
///
/// ```rust
/// use scirs2_neural::attention::alibi::AlibiConfig;
/// let cfg = AlibiConfig::default();
/// assert_eq!(cfg.num_heads, 8);
/// assert_eq!(cfg.max_seq_len, 2048);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct AlibiConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Maximum sequence length (used for pre-allocation hints).
    pub max_seq_len: usize,
}

impl Default for AlibiConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            max_seq_len: 2048,
        }
    }
}

// ---------------------------------------------------------------------------
// AlibiSlopes
// ---------------------------------------------------------------------------

/// Precomputed ALiBi slopes and bias matrices.
///
/// Construct with [`AlibiSlopes::new`], then call [`AlibiSlopes::get_bias_matrix`]
/// to obtain the head-indexed bias matrix for a given sequence length, or call
/// [`AlibiSlopes::apply_to_attention_scores`] to add the bias to a flat scores
/// buffer in-place.
#[derive(Debug, Clone)]
pub struct AlibiSlopes {
    config: AlibiConfig,
    /// Precomputed slopes `m_h` for each head.
    slopes: Vec<f64>,
}

impl AlibiSlopes {
    /// Compute ALiBi head slopes.
    ///
    /// Uses the formula `m_h = 2^{-8h/H}` for `h = 1 … H`, where `H` is the
    /// number of heads.
    ///
    /// ```rust
    /// use scirs2_neural::attention::alibi::AlibiSlopes;
    /// let slopes = AlibiSlopes::compute_slopes(4);
    /// assert_eq!(slopes.len(), 4);
    /// // slopes[0] = 2^{-2} = 0.25
    /// assert!((slopes[0] - 0.25).abs() < 1e-10);
    /// ```
    pub fn compute_slopes(num_heads: usize) -> Vec<f64> {
        if num_heads == 0 {
            return Vec::new();
        }
        (1..=num_heads)
            .map(|h| 2.0_f64.powf(-8.0 * h as f64 / num_heads as f64))
            .collect()
    }

    /// Create a new [`AlibiSlopes`] from a configuration.
    pub fn new(config: AlibiConfig) -> Self {
        let slopes = Self::compute_slopes(config.num_heads);
        Self { config, slopes }
    }

    /// Return the precomputed slope for head `h` (0-indexed).
    ///
    /// Returns `None` when `h >= num_heads`.
    pub fn slope_at(&self, h: usize) -> Option<f64> {
        self.slopes.get(h).copied()
    }

    /// Return a reference to all slopes.
    pub fn slopes(&self) -> &[f64] {
        &self.slopes
    }

    /// Compute the ALiBi bias matrix for all heads.
    ///
    /// For each head `h`, the bias at entry `(i, j)` is:
    ///
    /// ```text
    /// bias[h][i * seq_len + j] = -slope_h * |i - j|
    /// ```
    ///
    /// # Returns
    ///
    /// A `Vec` of length `num_heads`, each element being a flat
    /// `[seq_len * seq_len]` row-major bias matrix for one head.
    pub fn get_bias_matrix(&self, seq_len: usize) -> Vec<Vec<f64>> {
        self.slopes
            .iter()
            .map(|&slope| {
                let mut mat = vec![0.0f64; seq_len * seq_len];
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let dist = i.abs_diff(j);
                        mat[i * seq_len + j] = -slope * dist as f64;
                    }
                }
                mat
            })
            .collect()
    }

    /// Add ALiBi biases to a flat attention scores tensor in-place.
    ///
    /// # Arguments
    ///
    /// * `scores`    — flat scores tensor of shape `[num_heads * seq_len * seq_len]`
    ///   in head-major row-major layout.
    /// * `seq_len`   — sequence length.
    /// * `num_heads` — number of heads.  Must match `slopes.len()`.
    ///
    /// If `scores.len() != num_heads * seq_len * seq_len` the function is a
    /// no-op to avoid out-of-bounds access.
    pub fn apply_to_attention_scores(&self, scores: &mut [f64], seq_len: usize, num_heads: usize) {
        if seq_len == 0 || num_heads == 0 {
            return;
        }
        let expected = num_heads * seq_len * seq_len;
        if scores.len() < expected {
            return;
        }

        for h in 0..num_heads {
            let slope = match self.slopes.get(h).copied() {
                Some(s) => s,
                None => continue,
            };
            let head_offset = h * seq_len * seq_len;
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let dist = i.abs_diff(j);
                    scores[head_offset + i * seq_len + j] -= slope * dist as f64;
                }
            }
        }
    }

    /// Return the configuration.
    pub fn config(&self) -> &AlibiConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alibi_config_default() {
        let cfg = AlibiConfig::default();
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.max_seq_len, 2048);
    }

    #[test]
    fn test_alibi_slopes_geometric_sequence() {
        // For H=8 heads the slopes must form a geometric sequence with ratio
        // 2^{-1} = 0.5: m_1=0.5, m_2=0.25, …
        let slopes = AlibiSlopes::compute_slopes(8);
        assert_eq!(slopes.len(), 8);
        // m_1 = 2^{-1} = 0.5
        assert!((slopes[0] - 0.5).abs() < 1e-12, "m_1 should be 0.5");
        // m_2 = 2^{-2} = 0.25
        assert!((slopes[1] - 0.25).abs() < 1e-12, "m_2 should be 0.25");
        // Slopes are strictly decreasing.
        for i in 1..slopes.len() {
            assert!(slopes[i] < slopes[i - 1], "slopes should be decreasing");
        }
    }

    #[test]
    fn test_alibi_num_heads_8() {
        let cfg = AlibiConfig {
            num_heads: 8,
            max_seq_len: 64,
        };
        let alibi = AlibiSlopes::new(cfg);
        let slopes = alibi.slopes();
        assert_eq!(slopes.len(), 8);
        // All slopes in (0, 1).
        for &s in slopes {
            assert!(s > 0.0 && s < 1.0, "slope out of range: {s}");
        }
    }

    #[test]
    fn test_alibi_slopes_four_heads() {
        // For H=4: m_h = 2^{-2h} → m_1=0.25, m_2=0.0625, m_3=0.015625, m_4=~3.9e-3
        let slopes = AlibiSlopes::compute_slopes(4);
        assert!((slopes[0] - 0.25).abs() < 1e-10);
        assert!((slopes[1] - 0.0625).abs() < 1e-10);
    }

    #[test]
    fn test_alibi_bias_triangular_structure() {
        // bias[0][i,j] == -slope * |i-j|.
        let cfg = AlibiConfig {
            num_heads: 2,
            max_seq_len: 8,
        };
        let alibi = AlibiSlopes::new(cfg);
        let seq_len = 4;
        let bias = alibi.get_bias_matrix(seq_len);

        assert_eq!(bias.len(), 2);
        assert_eq!(bias[0].len(), seq_len * seq_len);

        let slope0 = alibi.slopes()[0];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let dist = i.abs_diff(j);
                let expected = -slope0 * dist as f64;
                let got = bias[0][i * seq_len + j];
                assert!(
                    (got - expected).abs() < 1e-12,
                    "bias mismatch at ({i},{j}): got {got:.8}, expected {expected:.8}"
                );
            }
        }
    }

    #[test]
    fn test_alibi_diagonal_is_zero() {
        // bias[h][i,i] = 0 for all h, i (distance is 0).
        let cfg = AlibiConfig {
            num_heads: 4,
            max_seq_len: 16,
        };
        let alibi = AlibiSlopes::new(cfg);
        let seq_len = 5;
        let bias = alibi.get_bias_matrix(seq_len);

        for head_bias in &bias[..4] {
            for i in 0..seq_len {
                let diag = head_bias[i * seq_len + i];
                assert!(diag.abs() < 1e-12, "diagonal should be 0, got {diag}");
            }
        }
    }

    #[test]
    fn test_alibi_applied_to_scores() {
        let num_heads = 2;
        let seq_len = 3;
        let cfg = AlibiConfig {
            num_heads,
            max_seq_len: 8,
        };
        let alibi = AlibiSlopes::new(cfg);

        // Create flat scores tensor, all zeros.
        let mut scores = vec![0.0f64; num_heads * seq_len * seq_len];
        alibi.apply_to_attention_scores(&mut scores, seq_len, num_heads);

        // After applying: diagonal entries should still be 0 (distance 0).
        let slope0 = alibi.slopes()[0];
        for i in 0..seq_len {
            let diag = scores[i * seq_len + i];
            assert!(diag.abs() < 1e-12, "diagonal should remain 0");
            // Off-diagonal: score[i][j] = -slope * |i-j|.
            for j in 0..seq_len {
                if i != j {
                    let dist = i.abs_diff(j);
                    let expected = -slope0 * dist as f64;
                    let got = scores[i * seq_len + j];
                    assert!(
                        (got - expected).abs() < 1e-12,
                        "scores[{i},{j}] head0: got {got:.8} expected {expected:.8}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_alibi_symmetry() {
        // bias[h][i,j] == bias[h][j,i] (symmetric around diagonal).
        let cfg = AlibiConfig {
            num_heads: 3,
            max_seq_len: 16,
        };
        let alibi = AlibiSlopes::new(cfg);
        let seq_len = 5;
        let bias = alibi.get_bias_matrix(seq_len);

        for (h, head_bias) in bias[..3].iter().enumerate() {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let ij = head_bias[i * seq_len + j];
                    let ji = head_bias[j * seq_len + i];
                    assert!(
                        (ij - ji).abs() < 1e-12,
                        "alibi bias should be symmetric: bias[{h}][{i},{j}]={ij} vs [{j},{i}]={ji}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_alibi_zero_heads_no_panic() {
        let slopes = AlibiSlopes::compute_slopes(0);
        assert!(slopes.is_empty());
    }
}
