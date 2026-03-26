//! Time encoding for Temporal Graph Neural Networks.
//!
//! Implements learnable (TGAT-style) and fixed positional time encodings.
//!
//! ## TGAT Time Encoding (Xu et al. 2020)
//!
//! The functional time encoding maps a scalar `t` into a `d`-dimensional
//! feature vector using sinusoidal basis functions with learnable frequencies:
//!
//! ```text
//! φ(t) = [cos(ω₁·t), sin(ω₁·t), cos(ω₂·t), sin(ω₂·t), ..., cos(ω_{d/2}·t), sin(ω_{d/2}·t)]
//! ```
//!
//! Frequencies are initialised with geometric spacing (similar to Transformer PE):
//!
//! ```text
//! ω_i = 1 / 10000^(2i / d)
//! ```
//!
//! This ensures high-frequency components capture short-term interactions while
//! low-frequency components capture long-range temporal dependencies.

use crate::error::{GraphError, Result};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// TimeEncode
// ─────────────────────────────────────────────────────────────────────────────

/// Learnable functional time encoder (TGAT, Xu et al. 2020).
///
/// Encodes a scalar timestamp into a `time_dim`-dimensional vector via
/// sinusoidal basis functions with learnable frequency weights `ω`.
///
/// `time_dim` must be even; each pair `(cos(ω_i·t), sin(ω_i·t))` uses one
/// frequency.
#[derive(Debug, Clone)]
pub struct TimeEncode {
    /// Frequency weights ω (length = time_dim / 2)
    pub omega: Vec<f64>,
    /// Total output dimension (even, = 2 * omega.len())
    pub time_dim: usize,
}

impl TimeEncode {
    /// Create a `TimeEncode` with geometric frequency spacing.
    ///
    /// Frequencies are initialised as `ω_i = 1 / 10000^(2i / time_dim)`.
    ///
    /// # Errors
    /// Returns `Err` if `time_dim` is 0 or odd.
    pub fn new(time_dim: usize) -> Result<Self> {
        if time_dim == 0 {
            return Err(GraphError::InvalidParameter {
                param: "time_dim".to_string(),
                value: "0".to_string(),
                expected: "positive even integer".to_string(),
                context: "TimeEncode::new".to_string(),
            });
        }
        if time_dim % 2 != 0 {
            return Err(GraphError::InvalidParameter {
                param: "time_dim".to_string(),
                value: format!("{}", time_dim),
                expected: "even integer".to_string(),
                context: "TimeEncode::new".to_string(),
            });
        }
        let half = time_dim / 2;
        let omega: Vec<f64> = (0..half)
            .map(|i| {
                let exponent = (2 * i) as f64 / time_dim as f64;
                1.0 / 10000_f64.powf(exponent)
            })
            .collect();
        Ok(TimeEncode { omega, time_dim })
    }

    /// Encode a scalar timestamp `t` into a `time_dim`-dimensional vector.
    ///
    /// Output layout: `[cos(ω₁t), sin(ω₁t), cos(ω₂t), sin(ω₂t), ...]`
    pub fn encode(&self, t: f64) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.time_dim);
        for &w in &self.omega {
            out.push((w * t).cos());
            out.push((w * t).sin());
        }
        out
    }

    /// Encode the time *difference* Δt = t_query − t_edge.
    ///
    /// This is equivalent to `encode(delta_t)` but semantically clearer.
    pub fn encode_delta(&self, t_query: f64, t_edge: f64) -> Vec<f64> {
        self.encode(t_query - t_edge)
    }

    /// Update frequencies via a gradient step (for gradient-based learning).
    ///
    /// `grad` must have the same length as `self.omega`.
    pub fn update_omega(&mut self, grad: &[f64], lr: f64) -> Result<()> {
        if grad.len() != self.omega.len() {
            return Err(GraphError::InvalidParameter {
                param: "grad".to_string(),
                value: format!("len={}", grad.len()),
                expected: format!("len={}", self.omega.len()),
                context: "TimeEncode::update_omega".to_string(),
            });
        }
        for (w, g) in self.omega.iter_mut().zip(grad.iter()) {
            *w -= lr * g;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PositionalTimeEncoding
// ─────────────────────────────────────────────────────────────────────────────

/// Positional time encoding combining absolute and relative time features.
///
/// Produces a concatenation of:
/// - Absolute time encoding: `TimeEncode::encode(t)` — captures the absolute
///   position in the timeline.
/// - Relative time encoding: `TimeEncode::encode(t - t_ref)` — captures the
///   lag from a reference (e.g., graph start time or node's last interaction).
///
/// Output dimensionality: `2 * time_dim`.
#[derive(Debug, Clone)]
pub struct PositionalTimeEncoding {
    /// Underlying sinusoidal encoder
    encoder: TimeEncode,
    /// Reference time (e.g., graph start time)
    pub t_ref: f64,
}

impl PositionalTimeEncoding {
    /// Create a new positional encoding with the given `time_dim` and reference time.
    ///
    /// # Errors
    /// Propagates errors from `TimeEncode::new`.
    pub fn new(time_dim: usize, t_ref: f64) -> Result<Self> {
        let encoder = TimeEncode::new(time_dim)?;
        Ok(PositionalTimeEncoding { encoder, t_ref })
    }

    /// Encode timestamp `t` into a `2 * time_dim`-dimensional vector.
    ///
    /// Layout: `[abs_enc(t) || rel_enc(t - t_ref)]`
    pub fn encode(&self, t: f64) -> Vec<f64> {
        let mut out = self.encoder.encode(t);
        let rel = self.encoder.encode(t - self.t_ref);
        out.extend(rel);
        out
    }

    /// Total output dimensionality.
    pub fn output_dim(&self) -> usize {
        2 * self.encoder.time_dim
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Compute scaled dot-product attention scores between query `q` and keys `K`.
///
/// Returns unnormalised attention logits of shape `[n_keys]`.
/// `scale = 1 / sqrt(d_k)`.
pub(crate) fn scaled_dot_product(q: &[f64], keys: &[Vec<f64>]) -> Vec<f64> {
    let d_k = q.len().max(1);
    let scale = 1.0 / (d_k as f64).sqrt();
    keys.iter()
        .map(|k| k.iter().zip(q.iter()).map(|(ki, qi)| ki * qi).sum::<f64>() * scale)
        .collect()
}

/// Numerically stable softmax over a slice.
pub(crate) fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|x| (x - max_val).exp()).collect();
    let sum = exps.iter().sum::<f64>().max(1e-12);
    exps.iter().map(|e| e / sum).collect()
}

/// Matrix-vector product: `W x` where `W` is `(out × in)` and `x` is `(in,)`.
pub(crate) fn matvec(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum())
        .collect()
}

/// Sigmoid activation.
pub(crate) fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Element-wise sigmoid over a vector.
pub(crate) fn sigmoid_vec(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| sigmoid(x)).collect()
}

/// Element-wise tanh over a vector.
pub(crate) fn tanh_vec(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| x.tanh()).collect()
}

/// ReLU activation.
pub(crate) fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Element-wise ReLU over a vector.
pub(crate) fn relu_vec(v: &[f64]) -> Vec<f64> {
    v.iter().map(|&x| relu(x)).collect()
}

/// Initialise a weight matrix `(rows × cols)` using Xavier uniform.
///
/// Fan-in = `cols`, fan-out = `rows`.  Uses a pure-Rust LCG for reproducibility.
pub(crate) fn xavier_init(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed.wrapping_add(1);
    let limit = (6.0 / (rows + cols).max(1) as f64).sqrt();

    let mut w = vec![vec![0.0f64; cols]; rows];
    for row in w.iter_mut() {
        for v in row.iter_mut() {
            // LCG: x_{n+1} = a * x_n + c  (mod 2^64)
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let frac = (state >> 11) as f64 / (1u64 << 53) as f64; // uniform [0,1)
            *v = frac * 2.0 * limit - limit; // uniform [-limit, limit)
        }
    }
    w
}

/// Concatenate two slices into a new `Vec<f64>`.
pub(crate) fn concat(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_encode_dimension() {
        let enc = TimeEncode::new(8).expect("should create encoder");
        let v = enc.encode(1.5);
        assert_eq!(v.len(), 8, "output must have time_dim elements");
    }

    #[test]
    fn test_time_encode_at_zero() {
        let enc = TimeEncode::new(8).expect("encoder");
        let v = enc.encode(0.0);
        // At t=0: cos(ω_i * 0) = 1.0, sin(ω_i * 0) = 0.0
        for i in 0..4 {
            assert!((v[2 * i] - 1.0).abs() < 1e-12, "cos(0) must be 1");
            assert!(v[2 * i + 1].abs() < 1e-12, "sin(0) must be 0");
        }
    }

    #[test]
    fn test_time_encode_different_times() {
        let enc = TimeEncode::new(8).expect("encoder");
        let v1 = enc.encode(1.0);
        let v2 = enc.encode(2.0);
        // Different timestamps must produce different encodings
        let diff: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "encode(1.0) must differ from encode(2.0)");
    }

    #[test]
    fn test_time_encode_odd_dim_returns_error() {
        let result = TimeEncode::new(7);
        assert!(result.is_err(), "odd time_dim must return error");
    }

    #[test]
    fn test_positional_time_encoding_dim() {
        let enc = PositionalTimeEncoding::new(8, 0.0).expect("positional encoder");
        let v = enc.encode(3.0);
        assert_eq!(v.len(), 16, "positional encoding must be 2*time_dim");
        assert_eq!(enc.output_dim(), 16);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "softmax must sum to 1");
    }

    #[test]
    fn test_softmax_empty() {
        let probs = softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_xavier_init_shape() {
        let w = xavier_init(4, 8, 42);
        assert_eq!(w.len(), 4);
        assert_eq!(w[0].len(), 8);
    }

    #[test]
    fn test_sigmoid_range() {
        let vals: Vec<f64> = vec![-100.0, -1.0, 0.0, 1.0, 100.0];
        for v in vals {
            let s = sigmoid(v);
            assert!(s >= 0.0 && s <= 1.0, "sigmoid must be in [0,1]");
        }
    }
}
