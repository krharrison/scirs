//! NeRF MLP architecture (Mildenhall et al. 2020).
//!
//! Architecture summary
//! --------------------
//! Geometry network (8 layers, 256-wide, ReLU activations):
//!   - Input: γ(pos)       [3 × (1 + 2·L_pos)] features
//!   - Layer 4: γ(pos) is concatenated again (skip connection)
//!   - Output: volume density σ (ReLU≥0) + 256-dim feature vector
//!
//! Radiance network (2 layers):
//!   - Input: feature_256 ∥ γ(dir)   [256 + 3 × (1 + 2·L_dir)] features
//!   - Output: RGB colour (sigmoid in \[0,1\]^3)

use super::positional_encoding::{encode_direction, encode_position};
use crate::error::{Result, VisionError};

// ── LCG parameters (Knuth) ─────────────────────────────────────────────────
const LCG_A: u64 = 6_364_136_223_846_793_005;
const LCG_C: u64 = 1_442_695_040_888_963_407;

/// Minimal linear congruential pseudo-random number generator.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }
    /// Return a uniform sample in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(LCG_A).wrapping_add(LCG_C);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Return a sample from a zero-mean unit-variance distribution
    /// (Box–Muller transform, deterministic).
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ── Activation functions ───────────────────────────────────────────────────

#[inline]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ── Linear layer helpers ───────────────────────────────────────────────────

/// Forward pass of a single linear layer: `y = W x + b`.
fn linear_forward(
    weights: &[Vec<f64>], // [out × in]
    bias: &[f64],         // [out]
    input: &[f64],        // [in]
) -> Vec<f64> {
    let out_dim = weights.len();
    let mut out = Vec::with_capacity(out_dim);
    for (row, b) in weights.iter().zip(bias.iter()) {
        let mut acc = *b;
        for (w, x) in row.iter().zip(input.iter()) {
            acc += w * x;
        }
        out.push(acc);
    }
    out
}

// ── NerfMlp ───────────────────────────────────────────────────────────────

/// The canonical NeRF two-branch MLP.
///
/// # Fields
///
/// * `geo_weights` / `geo_biases`   – geometry network layers (8 + 2 dense).
/// * `rgb_weights` / `rgb_biases`   – radiance network layers (2 dense).
/// * `n_freq_pos`                   – positional-encoding frequencies for xyz.
/// * `n_freq_dir`                   – positional-encoding frequencies for direction.
pub struct NerfMlp {
    geo_weights: Vec<Vec<Vec<f64>>>, // [n_layers+1][out][in]
    geo_biases: Vec<Vec<f64>>,       // [n_layers+1][out]
    rgb_weights: Vec<Vec<Vec<f64>>>, // [2][out][in]
    rgb_biases: Vec<Vec<f64>>,       // [2][out]
    /// Number of positional-encoding frequency bands for 3-D position.
    pub n_freq_pos: usize,
    /// Number of positional-encoding frequency bands for view direction.
    pub n_freq_dir: usize,
    hidden_dim: usize,
}

impl NerfMlp {
    /// Construct and Xavier-initialise a new NeRF MLP.
    ///
    /// # Arguments
    ///
    /// * `n_freq_pos` – positional-encoding bands for position (default 10).
    /// * `n_freq_dir` – positional-encoding bands for direction (default 4).
    /// * `hidden_dim` – width of hidden layers (default 256).
    /// * `seed`       – PRNG seed for weight initialisation.
    pub fn new(n_freq_pos: usize, n_freq_dir: usize, hidden_dim: usize, seed: u64) -> Self {
        let mut rng = Lcg::new(seed);

        let pos_enc_dim = 3 * (1 + 2 * n_freq_pos); // γ(pos) dimension
        let dir_enc_dim = 3 * (1 + 2 * n_freq_dir); // γ(dir) dimension

        // ── Geometry network ───────────────────────────────────────────────
        // Layer 0: pos_enc_dim → hidden_dim
        // Layers 1..3: hidden_dim → hidden_dim
        // Layer 4 (skip): hidden_dim + pos_enc_dim → hidden_dim
        // Layers 5..7: hidden_dim → hidden_dim
        // Layer 8 (density head): hidden_dim → 1 + hidden_dim  (density + feature)
        let geo_in_dims = {
            let mut dims = vec![pos_enc_dim];
            dims.extend(std::iter::repeat_n(hidden_dim, 3)); // layers 1..3
            dims.push(hidden_dim + pos_enc_dim); // layer 4 skip
            dims.extend(std::iter::repeat_n(hidden_dim, 3)); // layers 5..7
            dims.push(hidden_dim); // layer 8 (density head input)
            dims
        };
        let geo_out_dims = {
            let mut dims: Vec<usize> = std::iter::repeat_n(hidden_dim, 8).collect();
            dims.push(1 + hidden_dim); // density (1) + feature (hidden_dim)
            dims
        };

        let n_geo = geo_in_dims.len();
        let mut geo_weights = Vec::with_capacity(n_geo);
        let mut geo_biases = Vec::with_capacity(n_geo);
        for (in_d, out_d) in geo_in_dims.iter().zip(geo_out_dims.iter()) {
            let scale = (2.0 / *in_d as f64).sqrt(); // He / Xavier initialisation
            let w: Vec<Vec<f64>> = (0..*out_d)
                .map(|_| (0..*in_d).map(|_| rng.next_normal() * scale).collect())
                .collect();
            let b = vec![0.0_f64; *out_d];
            geo_weights.push(w);
            geo_biases.push(b);
        }

        // ── Radiance (RGB) network ─────────────────────────────────────────
        // Layer 0: (hidden_dim + dir_enc_dim) → hidden_dim/2  (128)
        // Layer 1: hidden_dim/2 → 3  (RGB)
        let rgb_in_d0 = hidden_dim + dir_enc_dim;
        let rgb_h = (hidden_dim / 2).max(1);
        let rgb_in_dims = [rgb_in_d0, rgb_h];
        let rgb_out_dims = [rgb_h, 3_usize];

        let mut rgb_weights = Vec::with_capacity(2);
        let mut rgb_biases = Vec::with_capacity(2);
        for (&in_d, &out_d) in rgb_in_dims.iter().zip(rgb_out_dims.iter()) {
            let scale = (2.0 / in_d as f64).sqrt();
            let w: Vec<Vec<f64>> = (0..out_d)
                .map(|_| (0..in_d).map(|_| rng.next_normal() * scale).collect())
                .collect();
            let b = vec![0.0_f64; out_d];
            rgb_weights.push(w);
            rgb_biases.push(b);
        }

        Self {
            geo_weights,
            geo_biases,
            rgb_weights,
            rgb_biases,
            n_freq_pos,
            n_freq_dir,
            hidden_dim,
        }
    }

    /// Run forward inference.
    ///
    /// Returns `(density, rgb)` where `density >= 0` and each RGB component in \[0,1\].
    ///
    /// # Arguments
    ///
    /// * `pos` – 3-D world-space position.
    /// * `dir` – unit view direction.
    pub fn forward(&self, pos: &[f64; 3], dir: &[f64; 3]) -> Result<(f64, [f64; 3])> {
        let pos_enc = encode_position(pos, self.n_freq_pos);
        let dir_enc = encode_direction(dir, self.n_freq_dir);

        // ── Geometry branch ────────────────────────────────────────────────
        let mut h = pos_enc.clone();
        for (layer_idx, (w, b)) in self
            .geo_weights
            .iter()
            .zip(self.geo_biases.iter())
            .enumerate()
        {
            // Skip connection at layer 4: concatenate γ(pos) again
            if layer_idx == 4 {
                let mut skip_h = h.clone();
                skip_h.extend_from_slice(&pos_enc);
                h = skip_h;
            }

            let raw = linear_forward(w, b, &h);

            // After layer 8 (density head): no activation yet — handled below
            if layer_idx == self.geo_weights.len() - 1 {
                h = raw;
            } else {
                h = raw.into_iter().map(relu).collect();
            }
        }

        // Split final output: h[0] = density, h[1..] = feature
        if h.len() != 1 + self.hidden_dim {
            return Err(VisionError::InvalidInput(
                "geometry network output shape mismatch".to_string(),
            ));
        }
        let density = relu(h[0]);
        let feature = &h[1..];

        // ── Radiance branch ────────────────────────────────────────────────
        let mut rgb_in: Vec<f64> = Vec::with_capacity(self.hidden_dim + dir_enc.len());
        rgb_in.extend_from_slice(feature);
        rgb_in.extend_from_slice(&dir_enc);

        // Layer 0 with ReLU
        let rgb_h = {
            let raw = linear_forward(&self.rgb_weights[0], &self.rgb_biases[0], &rgb_in);
            raw.into_iter().map(relu).collect::<Vec<_>>()
        };

        // Layer 1 → sigmoid
        let rgb_raw = linear_forward(&self.rgb_weights[1], &self.rgb_biases[1], &rgb_h);
        if rgb_raw.len() != 3 {
            return Err(VisionError::InvalidInput(
                "radiance network output shape mismatch".to_string(),
            ));
        }
        let rgb = [
            sigmoid(rgb_raw[0]),
            sigmoid(rgb_raw[1]),
            sigmoid(rgb_raw[2]),
        ];

        Ok((density, rgb))
    }

    /// Input dimensionality of the geometry skip-connection at layer 4.
    ///
    /// Used in tests to verify architecture correctness.
    pub fn skip_layer_input_dim(&self) -> usize {
        // layer 4 weights shape is [hidden_dim][hidden_dim + pos_enc_dim]
        if self.geo_weights.len() > 4 {
            self.geo_weights[4][0].len()
        } else {
            0
        }
    }

    /// Expected input dimension at layer 4: hidden_dim + pos_enc_dim.
    pub fn expected_skip_input_dim(&self) -> usize {
        self.hidden_dim + 3 * (1 + 2 * self.n_freq_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_mlp() -> NerfMlp {
        // Small model for fast tests
        NerfMlp::new(4, 2, 64, 42)
    }

    #[test]
    fn test_nerf_mlp_output_range() {
        let mlp = build_mlp();
        let pos = [0.1, 0.2, 0.3];
        let dir = [0.0, 0.0, 1.0];
        let (density, rgb) = mlp.forward(&pos, &dir).expect("forward pass");
        assert!(
            density >= 0.0,
            "density must be non-negative, got {density}"
        );
        for (c, ch) in rgb.iter().enumerate() {
            assert!(
                *ch >= 0.0 && *ch <= 1.0,
                "rgb[{c}] must be in [0,1], got {ch}"
            );
        }
    }

    #[test]
    fn test_nerf_skip_connection() {
        let mlp = build_mlp();
        let actual = mlp.skip_layer_input_dim();
        let expected = mlp.expected_skip_input_dim();
        assert_eq!(
            actual, expected,
            "skip-layer input dim: expected {expected}, got {actual}"
        );
    }
}
