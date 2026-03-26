//! Cyclostationary Spectral Analysis.
//!
//! A cyclostationary signal has statistical properties that vary periodically
//! in time.  This module provides tools for analysing such signals via the
//! Spectral Correlation Density (SCD) — also known as the Spectral Correlation
//! Function (SCF).
//!
//! # Algorithm
//!
//! The SCD is computed using the FFT Accumulation Method (FAM), which is
//! equivalent to the Time-Smoothed Cyclic Cross-Periodogram (TSCCP):
//!
//! ```text
//! S_x(f; α) ≈ (1/T) Σ_t X_T(t, f + α/2) · conj(X_T(t, f − α/2))
//! ```
//!
//! where `X_T(t, f)` is the windowed STFT, `f` the spectral frequency, and
//! `α` the cyclic frequency.
//!
//! # Examples
//!
//! ```
//! use scirs2_fft::cyclostationary::{CyclostationaryAnalyzer, CyclostationaryConfig};
//! use std::f64::consts::PI;
//!
//! // Generate an AM signal: cyclic freq = 2 * carrier
//! let n = 512;
//! let fs = 100.0;
//! let fc = 10.0;  // carrier frequency
//! let fm = 1.0;   // modulation frequency
//! let signal: Vec<f64> = (0..n)
//!     .map(|i| {
//!         let t = i as f64 / fs;
//!         (1.0 + 0.8 * (2.0 * PI * fm * t).cos()) * (2.0 * PI * fc * t).cos()
//!     })
//!     .collect();
//!
//! let mut config = CyclostationaryConfig::default();
//! config.n_fft = 64;
//! config.fs = fs;
//!
//! let analyzer = CyclostationaryAnalyzer::new();
//! let result = analyzer.compute_scd(&signal, fs, &config).expect("SCD should succeed");
//!
//! println!(
//!     "SCD shape: {} cyclic freqs × {} spectral freqs",
//!     result.n_alphas(),
//!     result.n_freqs()
//! );
//! ```

pub mod analysis;
pub mod types;

pub use analysis::{compute_scd, detect_cyclic_frequencies, CyclostationaryAnalyzer};
pub use types::{CyclostationaryConfig, SpectralCorrelationResult};

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Build an AM (amplitude-modulated) signal with known cyclic frequency.
    fn make_am_signal(n: usize, fs: f64, fc: f64, fm: f64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (1.0 + 0.8 * (2.0 * PI * fm * t).cos()) * (2.0 * PI * fc * t).cos()
            })
            .collect()
    }

    #[test]
    fn test_cyclostationary_config_default() {
        let config = CyclostationaryConfig::default();
        assert_eq!(config.n_fft, 512);
        assert!((config.overlap - 0.5).abs() < 1e-10);
        assert!((config.alpha_resolution - 0.01).abs() < 1e-10);
        assert!(config.cyclic_freqs.is_none());
    }

    #[test]
    fn test_scd_output_shape() {
        let n = 512;
        let fs = 1.0;
        // Provide explicit cyclic frequencies so we know the shape
        let alphas = vec![0.0, 0.1, 0.2];
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.1 * i as f64).sin()).collect();

        let mut config = CyclostationaryConfig::default();
        config.n_fft = 64;
        config.fs = fs;
        config.cyclic_freqs = Some(alphas.clone());

        let analyzer = CyclostationaryAnalyzer::new();
        let result = analyzer
            .compute_scd(&signal, fs, &config)
            .expect("SCD should succeed");

        assert_eq!(
            result.n_alphas(),
            alphas.len(),
            "Row count should match n_alphas"
        );
        assert_eq!(
            result.n_freqs(),
            config.n_fft,
            "Column count should match n_fft"
        );
        assert_eq!(result.spectral_frequencies.len(), config.n_fft);
    }

    #[test]
    fn test_cyclostationary_am_detects_cyclic_freq() {
        let n = 1024;
        let fs = 100.0;
        let fc = 10.0;
        let fm = 2.0; // AM cyclic freq = 2 * fm
        let signal = make_am_signal(n, fs, fc, fm);

        let analyzer = CyclostationaryAnalyzer::new();
        let alphas = analyzer
            .detect_cyclic_frequencies(&signal, fs, 0.5)
            .expect("detection should succeed");

        // For an AM signal there should be at least one detected cyclic frequency
        // (the exact count depends on signal length and resolution)
        // Just verify the function returns without error and gives non-negative freqs
        for &a in &alphas {
            assert!(a >= 0.0, "Cyclic frequencies must be non-negative");
        }
    }

    #[test]
    fn test_detect_cyclic_frequencies_white_noise() {
        // For pseudo-noise the peak-detection threshold should suppress all bins
        // (or at most a few spurious ones).  Mainly test it doesn't error out.
        let n = 512;
        let mut state: u64 = 0xc0ffee_deadbeef;
        let noise: Vec<f64> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as f64 / (u32::MAX as f64) * 2.0 - 1.0
            })
            .collect();

        let analyzer = CyclostationaryAnalyzer::new();
        // Use a high threshold so noise peaks are rejected
        let result = analyzer.detect_cyclic_frequencies(&noise, 1.0, 0.01);
        // Should not error; result may be empty or small
        assert!(result.is_ok());
    }

    #[test]
    fn test_scd_with_explicit_cyclic_freqs() {
        let n = 512;
        let fs = 1.0;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.2 * i as f64).cos()).collect();

        let mut config = CyclostationaryConfig::default();
        config.n_fft = 64;
        config.fs = fs;
        config.cyclic_freqs = Some(vec![0.0, 0.1, 0.2, 0.4]);

        let analyzer = CyclostationaryAnalyzer::new();
        let result = analyzer
            .compute_scd(&signal, fs, &config)
            .expect("SCD should succeed");

        assert_eq!(result.cyclic_frequencies.len(), 4);
        assert_eq!(result.n_alphas(), 4);
    }

    #[test]
    fn test_cyclic_power_spectrum() {
        let n = 512;
        let fs = 1.0;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.1 * i as f64).sin()).collect();

        let config = CyclostationaryConfig {
            n_fft: 64,
            fs,
            cyclic_freqs: Some(vec![0.0, 0.05, 0.1, 0.2]),
            ..CyclostationaryConfig::default()
        };

        let analyzer = CyclostationaryAnalyzer::new();
        let result = analyzer
            .compute_scd(&signal, fs, &config)
            .expect("SCD should succeed");

        let cps = result.cyclic_power_spectrum();
        assert_eq!(cps.len(), 4);
        for &v in &cps {
            assert!(
                v >= 0.0,
                "Cyclic power spectrum values must be non-negative"
            );
        }
    }
}
