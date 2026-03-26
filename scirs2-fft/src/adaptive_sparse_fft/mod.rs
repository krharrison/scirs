//! Adaptive Sparse FFT with parameter-free sparsity estimation.
//!
//! This module provides an adaptive sparse FFT algorithm that automatically
//! estimates signal sparsity without requiring the user to specify the number
//! of non-zero frequency components in advance.
//!
//! # Algorithm Overview
//!
//! 1. **Sparsity estimation** — an energy-based knee-point / elbow method
//!    determines the number of dominant frequency components `k`.
//! 2. **Hash-based recovery** — the signal is permuted and sub-sampled in
//!    multiple iterations to localise frequency components with high probability.
//! 3. **Iterative refinement** — each iteration refines the sparsity estimate
//!    using the residual energy from the previous round.
//!
//! # Examples
//!
//! ```
//! use scirs2_fft::adaptive_sparse_fft::{AdaptiveSparseFft, AdaptiveSfftConfig};
//! use std::f64::consts::PI;
//!
//! let n = 256;
//! let signal: Vec<f64> = (0..n)
//!     .map(|i| {
//!         (2.0 * PI * 8.0 * i as f64 / n as f64).sin()
//!         + 0.5 * (2.0 * PI * 20.0 * i as f64 / n as f64).sin()
//!     })
//!     .collect();
//!
//! let config = AdaptiveSfftConfig::default();
//! let solver = AdaptiveSparseFft::new();
//! let result = solver.compute(&signal, &config).expect("recovery should succeed");
//!
//! println!(
//!     "Estimated sparsity: {}, iterations: {}",
//!     result.estimated_sparsity, result.iterations
//! );
//! ```

pub mod estimation;
pub mod recovery;
pub mod types;

pub use estimation::{estimate_sparsity, SparsityEstimator};
pub use recovery::{adaptive_sparse_fft_auto, AdaptiveSparseFft};
pub use types::{AdaptiveSfftConfig, AdaptiveSfftResult};

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Helper: generate a multi-tone signal with `k` tones.
    fn make_multi_tone(n: usize, freq_bins: &[usize]) -> Vec<f64> {
        (0..n)
            .map(|i| {
                freq_bins
                    .iter()
                    .map(|&f| (2.0 * PI * f as f64 * i as f64 / n as f64).sin())
                    .sum::<f64>()
            })
            .collect()
    }

    #[test]
    fn test_estimate_sparsity_single_tone() {
        let n = 256;
        let signal = make_multi_tone(n, &[10]);
        let k = estimate_sparsity(&signal).expect("should succeed");
        // A pure single tone should give a very small sparsity estimate
        assert!(k <= 4, "Expected k <= 4 for single tone, got {k}");
    }

    #[test]
    fn test_estimate_sparsity_multi_tone() {
        let n = 256;
        let signal = make_multi_tone(n, &[5, 13, 27, 55]);
        let k = estimate_sparsity(&signal).expect("should succeed");
        // k should be small (the elbow is typically at or near 4)
        assert!(k >= 1, "k must be at least 1, got {k}");
    }

    #[test]
    fn test_adaptive_sfft_config_default() {
        let config = AdaptiveSfftConfig::default();
        assert_eq!(config.max_sparsity, 64);
        assert!((config.confidence - 0.95).abs() < 1e-10);
        assert_eq!(config.filter_width, 8);
        assert_eq!(config.max_iterations, 5);
    }

    #[test]
    fn test_adaptive_sfft_single_tone_finds_frequency() {
        let n = 256;
        let freq_bin = 10_usize;
        let signal = make_multi_tone(n, &[freq_bin]);
        let config = AdaptiveSfftConfig::default();
        let solver = AdaptiveSparseFft::new();
        let result = solver.compute(&signal, &config).expect("should succeed");

        assert!(
            !result.frequencies.is_empty(),
            "Should find at least one frequency"
        );
        assert!(
            result.estimated_sparsity >= 1,
            "Estimated sparsity must be >= 1"
        );
    }

    #[test]
    fn test_adaptive_sfft_result_structure() {
        let n = 128;
        let signal = make_multi_tone(n, &[3, 7, 11, 23]);
        let config = AdaptiveSfftConfig::default();
        let solver = AdaptiveSparseFft::new();
        let result = solver.compute(&signal, &config).expect("should succeed");

        // Frequencies and coefficients must have the same length
        assert_eq!(result.frequencies.len(), result.coefficients.len());
        // Iterations must be at least 1
        assert!(result.iterations >= 1);
        // Captured fraction must be in [0, 1]
        assert!(result.captured_energy_fraction >= 0.0);
        assert!(result.captured_energy_fraction <= 1.0 + 1e-9);
    }

    #[test]
    fn test_sparsity_estimator_noisy_signal() {
        let n = 512;
        // Pseudo-noise via simple LCG — no external rand dependency needed
        let mut state: u64 = 0xdeadbeef_cafebabe;
        let noise: Vec<f64> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as f64 / (u32::MAX as f64) * 2.0 - 1.0
            })
            .collect();
        let estimator = SparsityEstimator::new(128);
        let k = estimator
            .estimate(&noise)
            .expect("should succeed for noise");
        assert!(k >= 1);
    }

    #[test]
    fn test_adaptive_sfft_auto_convenience() {
        let n = 128;
        let signal = make_multi_tone(n, &[7]);
        let result = adaptive_sparse_fft_auto(&signal).expect("should succeed");
        assert!(!result.frequencies.is_empty());
    }
}
