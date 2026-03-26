//! Rotary Position Embedding (RoPE).
//!
//! RoPE (Su et al., 2021, "RoFormer: Enhanced Transformer with Rotary Position
//! Embedding") encodes absolute position information into query and key vectors
//! by rotating pairs of features by a position-dependent angle.  Because the
//! resulting inner product depends only on the *relative* position between two
//! tokens, RoPE naturally generalises to sequences longer than those seen
//! during training.
//!
//! ## Mathematical definition
//!
//! For position `m` and feature dimension pair `(i, i + d/2)` the rotation is:
//!
//! ```text
//! x'[i]     = x[i] * cos(m * θ_i) - x[i+d/2] * sin(m * θ_i)
//! x'[i+d/2] = x[i] * sin(m * θ_i) + x[i+d/2] * cos(m * θ_i)
//! ```
//!
//! where `θ_i = base^{-2i/d}` (default `base = 10000`).
//!
//! ## Example
//!
//! ```rust
//! use scirs2_neural::attention::rope::{RopeConfig, RopeEmbedding};
//!
//! // Use Default and override fields via the builder pattern.
//! let cfg = RopeConfig::default(); // dim=0 (no-op), base=10000, max_seq_len=4096
//! let rope = RopeEmbedding::new(cfg);
//!
//! let seq_len = 4;
//! let dim = 8;
//! let mut q: Vec<f64> = (0..seq_len * dim).map(|i| i as f64 * 0.1).collect();
//! // With dim=0 this is a no-op; the buffer is unchanged.
//! rope.apply(&mut q, seq_len, dim);
//! assert_eq!(q.len(), seq_len * dim);
//! ```

use std::f64::consts::PI;

/// Configuration for Rotary Position Embedding.
///
/// # Defaults
///
/// ```rust
/// use scirs2_neural::attention::rope::RopeConfig;
/// let cfg = RopeConfig::default();
/// assert_eq!(cfg.base, 10_000.0);
/// assert_eq!(cfg.dim, 0);
/// assert_eq!(cfg.max_seq_len, 4096);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct RopeConfig {
    /// Frequency base `θ_base`.  Standard value is `10000.0` (GPT-NeoX,
    /// LLaMA, etc.).  Larger values shift frequencies to lower bands.
    pub base: f64,

    /// Number of feature dimensions to rotate.  Must be even.  When `0` the
    /// embedding is a no-op (useful as a placeholder).
    pub dim: usize,

    /// Maximum sequence length for which sin/cos tables are precomputed.
    pub max_seq_len: usize,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            base: 10_000.0,
            dim: 0,
            max_seq_len: 4096,
        }
    }
}

// ---------------------------------------------------------------------------
// RopeEmbedding
// ---------------------------------------------------------------------------

/// Precomputed Rotary Position Embedding tables.
///
/// Construct once with [`RopeEmbedding::new`], then call
/// [`RopeEmbedding::apply`] to rotate query or key vectors in-place.
#[derive(Debug, Clone)]
pub struct RopeEmbedding {
    config: RopeConfig,
    /// `cos_table[m][i]` = cos(m * θ_i) for position m, half-dim i.
    cos_table: Vec<Vec<f64>>,
    /// `sin_table[m][i]` = sin(m * θ_i) for position m, half-dim i.
    sin_table: Vec<Vec<f64>>,
}

impl RopeEmbedding {
    /// Create a new [`RopeEmbedding`] by precomputing sin/cos tables.
    ///
    /// If `config.dim == 0` or `config.max_seq_len == 0` the tables are
    /// empty and [`apply`] becomes a no-op.
    pub fn new(config: RopeConfig) -> Self {
        let d = config.dim;
        let max_seq = config.max_seq_len;

        if d == 0 || max_seq == 0 {
            return Self {
                config,
                cos_table: Vec::new(),
                sin_table: Vec::new(),
            };
        }

        // Half-dimension (we rotate pairs).
        let half_d = d / 2;

        // Precompute base frequencies θ_i = base^{-2i/d}.
        let thetas: Vec<f64> = (0..half_d)
            .map(|i| {
                let exp = -2.0 * i as f64 / d as f64;
                config.base.powf(exp)
            })
            .collect();

        let mut cos_table = Vec::with_capacity(max_seq);
        let mut sin_table = Vec::with_capacity(max_seq);

        for m in 0..max_seq {
            let m_f = m as f64;
            let cos_row: Vec<f64> = thetas.iter().map(|&theta| (m_f * theta).cos()).collect();
            let sin_row: Vec<f64> = thetas.iter().map(|&theta| (m_f * theta).sin()).collect();
            cos_table.push(cos_row);
            sin_table.push(sin_row);
        }

        Self {
            config,
            cos_table,
            sin_table,
        }
    }

    /// Apply RoPE rotation to a flat tensor in-place.
    ///
    /// The tensor is assumed to have logical shape `[seq_len, head_dim]`.
    /// Only the first `dim` features (matching the config) are rotated; the
    /// rest are left untouched.  If `seq_len > max_seq_len` positions beyond
    /// the table length are left unrotated (safe, if suboptimal).
    ///
    /// # Arguments
    ///
    /// * `x`        — flat buffer `[seq_len * head_dim]`, modified in-place.
    /// * `seq_len`  — number of token positions.
    /// * `head_dim` — feature dimension per position.
    pub fn apply(&self, x: &mut [f64], seq_len: usize, head_dim: usize) {
        let d = self.config.dim.min(head_dim);
        if d == 0 || !d.is_multiple_of(2) || seq_len == 0 {
            return;
        }
        let half_d = d / 2;

        for m in 0..seq_len {
            if m >= self.cos_table.len() {
                // Beyond precomputed range — compute on the fly.
                self.apply_position_runtime(x, m, head_dim, half_d);
                continue;
            }
            let cos_row = &self.cos_table[m];
            let sin_row = &self.sin_table[m];
            let offset = m * head_dim;

            for i in 0..half_d {
                let x_i = x[offset + i];
                let x_j = x[offset + i + half_d];
                let c = cos_row[i];
                let s = sin_row[i];
                x[offset + i] = x_i * c - x_j * s;
                x[offset + i + half_d] = x_i * s + x_j * c;
            }
        }
    }

    /// Compute and apply rotation for a position beyond the precomputed table.
    fn apply_position_runtime(&self, x: &mut [f64], pos: usize, head_dim: usize, half_d: usize) {
        let d = half_d * 2;
        let offset = pos * head_dim;
        let m_f = pos as f64;

        for i in 0..half_d {
            let exp = -2.0 * i as f64 / d as f64;
            let theta = self.config.base.powf(exp);
            let angle = m_f * theta;
            let c = angle.cos();
            let s = angle.sin();

            let x_i = x[offset + i];
            let x_j = x[offset + i + half_d];
            x[offset + i] = x_i * c - x_j * s;
            x[offset + i + half_d] = x_i * s + x_j * c;
        }
    }

    /// Compute the "half-rotation" permutation of a slice: `[-x[d/2:], x[:d/2]]`.
    ///
    /// This helper is used internally to verify the rotation identity:
    /// `rotate(x, m) = x * cos(m*θ) + half_rotate(x) * sin(m*θ)`.
    ///
    /// # Arguments
    ///
    /// * `x_slice` — slice of length `2k` (even).
    pub fn apply_half_rotation(x_slice: &[f64]) -> Vec<f64> {
        let n = x_slice.len();
        if n == 0 || !n.is_multiple_of(2) {
            return x_slice.to_vec();
        }
        let half = n / 2;
        let mut out = vec![0.0f64; n];
        for i in 0..half {
            out[i] = -x_slice[i + half];
            out[i + half] = x_slice[i];
        }
        out
    }

    /// Precomputed cos values for position `m` (length `dim/2`).
    /// Returns an empty slice if `m >= max_seq_len` or `dim == 0`.
    pub fn cos_at(&self, m: usize) -> &[f64] {
        self.cos_table.get(m).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Precomputed sin values for position `m` (length `dim/2`).
    /// Returns an empty slice if `m >= max_seq_len` or `dim == 0`.
    pub fn sin_at(&self, m: usize) -> &[f64] {
        self.sin_table.get(m).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Return the base frequency.
    pub fn base(&self) -> f64 {
        self.config.base
    }

    /// Return the feature dimension.
    pub fn dim(&self) -> usize {
        self.config.dim
    }

    /// Return the precomputed table length (max sequence length).
    pub fn table_len(&self) -> usize {
        self.cos_table.len()
    }

    /// The `RopeConfig` used to construct this embedding.
    pub fn config(&self) -> &RopeConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rope(dim: usize) -> RopeEmbedding {
        let cfg = RopeConfig {
            dim,
            ..Default::default()
        };
        RopeEmbedding::new(cfg)
    }

    /// Euclidean norm of a slice.
    fn norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    #[test]
    fn test_rope_config_default() {
        let cfg = RopeConfig::default();
        assert_eq!(cfg.base, 10_000.0);
        assert_eq!(cfg.dim, 0);
        assert_eq!(cfg.max_seq_len, 4096);
    }

    #[test]
    fn test_rope_rotation_orthogonal() {
        // RoPE is an orthogonal transform: ||rotate(x)|| == ||x||.
        let dim = 8;
        let rope = make_rope(dim);
        let seq_len = 4;
        let mut x: Vec<f64> = (0..seq_len * dim).map(|i| (i as f64 + 1.0) * 0.1).collect();
        let orig_norm = norm(&x);

        rope.apply(&mut x, seq_len, dim);

        let rotated_norm = norm(&x);
        assert!(
            (orig_norm - rotated_norm).abs() < 1e-9,
            "norms should be equal: {orig_norm:.8} vs {rotated_norm:.8}"
        );
    }

    #[test]
    fn test_rope_different_positions_differ() {
        // Applying rope to position 0 and position 1 should give different results
        // (unless the input happens to be invariant, which is very unlikely).
        let dim = 8;
        let seq_len = 2;
        let base: Vec<f64> = (0..dim).map(|i| (i as f64 + 1.0) * 0.3).collect();

        // Two tokens with same feature vector.
        let mut x = vec![];
        x.extend_from_slice(&base);
        x.extend_from_slice(&base);

        let rope = make_rope(dim);
        rope.apply(&mut x, seq_len, dim);

        // pos 0 and pos 1 outputs should differ.
        let pos0 = &x[..dim];
        let pos1 = &x[dim..];
        let diff: f64 = pos0
            .iter()
            .zip(pos1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "different positions should produce different outputs"
        );
    }

    #[test]
    fn test_rope_relative_shift_property() {
        // RoPE satisfies: <rotate(q, m), rotate(k, n)> = f(m-n, q, k)
        // We verify this by checking two pairs with the same relative offset
        // have the same dot product.
        let dim = 8;
        let rope = make_rope(dim);

        let q: Vec<f64> = (0..dim).map(|i| (i as f64 + 1.0) * 0.1).collect();
        let k: Vec<f64> = (0..dim).map(|i| ((dim - i) as f64) * 0.2).collect();

        // Pair (m=1, n=0) and pair (m=3, n=2) — same relative offset.
        let seq_len = 4;
        let mut buf = vec![0.0f64; seq_len * dim];

        // Rotate q at position 1.
        buf[dim..2 * dim].copy_from_slice(&q);
        let mut q1 = buf[dim..2 * dim].to_vec();
        let cfg = RopeConfig {
            dim,
            ..Default::default()
        };
        let rope2 = RopeEmbedding::new(cfg);
        // Apply rope to a 2-element sequence starting from position 1 by
        // placing q at row 1 in a 2-row buffer.
        let mut buf2 = vec![0.0f64; 2 * dim];
        buf2[dim..].copy_from_slice(&q);
        rope2.apply(&mut buf2, 2, dim);
        let q_rot_1 = buf2[dim..].to_vec();

        // Rotate q at position 3.
        let mut buf3 = vec![0.0f64; 4 * dim];
        buf3[3 * dim..].copy_from_slice(&q);
        rope2.apply(&mut buf3, 4, dim);
        let q_rot_3 = buf3[3 * dim..].to_vec();

        // Rotate k at position 0 and 2.
        let mut k_rot_0 = k.clone();
        rope2.apply(&mut k_rot_0, 1, dim); // position 0 rotation.

        let mut buf4 = vec![0.0f64; 3 * dim];
        buf4[2 * dim..].copy_from_slice(&k);
        rope2.apply(&mut buf4, 3, dim);
        let k_rot_2 = buf4[2 * dim..].to_vec();

        let dot_10: f64 = q_rot_1.iter().zip(k_rot_0.iter()).map(|(a, b)| a * b).sum();
        let dot_32: f64 = q_rot_3.iter().zip(k_rot_2.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (dot_10 - dot_32).abs() < 1e-8,
            "RoPE relative-shift: dot_10={dot_10:.8} dot_32={dot_32:.8}"
        );
    }

    #[test]
    fn test_rope_apply_inplace() {
        let dim = 4;
        let rope = make_rope(dim);
        let seq_len = 3;
        let mut x: Vec<f64> = (0..seq_len * dim).map(|i| i as f64 * 0.5).collect();
        let original = x.clone();
        rope.apply(&mut x, seq_len, dim);
        // Verify position 0 rotation: at m=0 all angles are 0, so cos=1, sin=0
        // → x unchanged for the first row.
        let pos0_unchanged = x[..dim]
            .iter()
            .zip(original[..dim].iter())
            .all(|(a, b)| (a - b).abs() < 1e-12);
        assert!(
            pos0_unchanged,
            "at position 0 RoPE should not change values"
        );
        // Verify that later positions ARE changed.
        let pos2_changed = x[2 * dim..3 * dim]
            .iter()
            .zip(original[2 * dim..3 * dim].iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(pos2_changed, "position 2 should be rotated");
    }

    #[test]
    fn test_rope_base_frequencies() {
        // Verify that precomputed cos values match hand-computed θ_i.
        let base = 10_000.0_f64;
        let dim = 4;
        let cfg = RopeConfig {
            base,
            dim,
            max_seq_len: 8,
        };
        let rope = RopeEmbedding::new(cfg);

        // Half-dim = 2, thetas: θ_0 = 1.0, θ_1 = 10000^{-1} = 0.0001.
        // At m=1: cos_0 = cos(1.0), cos_1 = cos(0.0001)
        let cos_m1 = rope.cos_at(1);
        assert_eq!(cos_m1.len(), 2);

        let theta_0 = base.powf(-0.0 / dim as f64); // =1.0
        let theta_1 = base.powf(-2.0 / dim as f64);
        let expected_cos_0 = (1.0_f64 * theta_0).cos();
        let expected_cos_1 = (1.0_f64 * theta_1).cos();

        assert!((cos_m1[0] - expected_cos_0).abs() < 1e-12);
        assert!((cos_m1[1] - expected_cos_1).abs() < 1e-12);
    }

    #[test]
    fn test_rope_half_rotation() {
        let x = vec![1.0f64, 2.0, 3.0, 4.0];
        let rotated = RopeEmbedding::apply_half_rotation(&x);
        // Expected: [-x[2], -x[3], x[0], x[1]] = [-3, -4, 1, 2]
        assert!((rotated[0] - (-3.0)).abs() < 1e-12);
        assert!((rotated[1] - (-4.0)).abs() < 1e-12);
        assert!((rotated[2] - 1.0).abs() < 1e-12);
        assert!((rotated[3] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_rope_zero_dim_is_noop() {
        let rope = make_rope(0);
        let mut x = vec![1.0f64, 2.0, 3.0];
        let orig = x.clone();
        rope.apply(&mut x, 1, 3);
        assert_eq!(x, orig, "dim=0 should be a no-op");
    }

    #[test]
    fn test_rope_position_zero_identity() {
        // cos(0) = 1, sin(0) = 0 for all θ_i → position 0 is identity.
        let dim = 8;
        let rope = make_rope(dim);
        let seq_len = 3;
        let mut x: Vec<f64> = (0..seq_len * dim).map(|i| (i as f64) * 0.7 + 1.0).collect();
        let original = x.clone();
        rope.apply(&mut x, seq_len, dim);

        for i in 0..dim {
            assert!(
                (x[i] - original[i]).abs() < 1e-12,
                "position 0 identity failed at i={i}"
            );
        }
    }
}
