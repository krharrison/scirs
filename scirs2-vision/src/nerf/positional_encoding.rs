//! Positional (Fourier) encoding for NeRF inputs.
//!
//! Implements the sinusoidal positional encoding from Mildenhall et al. 2020
//! ("NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis").
//!
//! For a scalar `x` and `L` frequency levels, the encoding is:
//!
//! ```text
//! γ(x) = [x, sin(2⁰ π x), cos(2⁰ π x), sin(2¹ π x), cos(2¹ π x), …,
//!            sin(2^{L-1} π x), cos(2^{L-1} π x)]
//! ```
//!
//! giving `1 + 2L` output features per scalar.

use std::f64::consts::PI;

/// Encode a single scalar value with `n_freq` frequency bands.
///
/// Output length: `1 + 2 * n_freq`.
///
/// # Arguments
///
/// * `x`      – scalar input value.
/// * `n_freq` – number of frequency bands L.
pub fn positional_encode(x: f64, n_freq: usize) -> Vec<f64> {
    let out_len = 1 + 2 * n_freq;
    let mut out = Vec::with_capacity(out_len);
    out.push(x);
    for k in 0..n_freq {
        let freq = (1u64 << k) as f64 * PI; // 2^k · π
        out.push((freq * x).sin());
        out.push((freq * x).cos());
    }
    out
}

/// Encode a 3-D position vector with `n_freq` frequency bands per component.
///
/// Each of the three components is independently encoded; the results are
/// concatenated:
///
/// ```text
/// output = [γ(x), γ(y), γ(z)]
/// ```
///
/// Output length: `3 * (1 + 2 * n_freq)`.
///
/// # Arguments
///
/// * `pos`    – 3-D world-space position `[x, y, z]`.
/// * `n_freq` – number of frequency bands L.
pub fn encode_position(pos: &[f64; 3], n_freq: usize) -> Vec<f64> {
    let component_len = 1 + 2 * n_freq;
    let mut out = Vec::with_capacity(3 * component_len);
    for &coord in pos.iter() {
        out.extend_from_slice(&positional_encode(coord, n_freq));
    }
    out
}

/// Encode a unit viewing-direction vector with `n_freq` frequency bands per component.
///
/// Identical in structure to [`encode_position`] but intended for the lower-frequency
/// direction branch of NeRF (typically `n_freq` = 4).
///
/// Output length: `3 * (1 + 2 * n_freq)`.
///
/// # Arguments
///
/// * `dir`    – unit-length view direction `[dx, dy, dz]`.
/// * `n_freq` – number of frequency bands L.
pub fn encode_direction(dir: &[f64; 3], n_freq: usize) -> Vec<f64> {
    encode_position(dir, n_freq)
}

/// Compute the expected output dimensionality for [`encode_position`] /
/// [`encode_direction`] with a given number of frequency bands.
#[inline]
pub fn encoding_dim(n_freq: usize) -> usize {
    3 * (1 + 2 * n_freq)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positional_encode_dim() {
        for n_freq in [1_usize, 4, 10] {
            let enc = positional_encode(0.5, n_freq);
            assert_eq!(
                enc.len(),
                1 + 2 * n_freq,
                "n_freq={n_freq}: expected dim {}, got {}",
                1 + 2 * n_freq,
                enc.len()
            );
        }
    }

    #[test]
    fn test_encode_position_dim() {
        for n_freq in [1_usize, 4, 10] {
            let enc = encode_position(&[0.1, 0.2, 0.3], n_freq);
            assert_eq!(enc.len(), 3 * (1 + 2 * n_freq));
        }
    }

    #[test]
    fn test_positional_encode_zero() {
        // γ(0) = [0, sin(0), cos(0), sin(0), cos(0), …]
        //       = [0, 0, 1, 0, 1, …]
        let enc = positional_encode(0.0, 3);
        assert!((enc[0] - 0.0).abs() < 1e-12); // identity
        for k in 0..3 {
            let sin_idx = 1 + 2 * k;
            let cos_idx = 2 + 2 * k;
            assert!(
                (enc[sin_idx] - 0.0).abs() < 1e-12,
                "sin at k={k} should be 0, got {}",
                enc[sin_idx]
            );
            assert!(
                (enc[cos_idx] - 1.0).abs() < 1e-12,
                "cos at k={k} should be 1, got {}",
                enc[cos_idx]
            );
        }
    }

    #[test]
    fn test_encoding_dim_helper() {
        assert_eq!(encoding_dim(10), 3 * 21);
        assert_eq!(encoding_dim(4), 3 * 9);
    }
}
