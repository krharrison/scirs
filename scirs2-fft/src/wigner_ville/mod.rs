//! Wigner-Ville Distribution (WVD) and Pseudo-WVD (PWVD).
//!
//! The Wigner-Ville distribution provides an ideal time-frequency representation
//! for mono-component (single-tone or chirp) signals.  For multi-component signals
//! it exhibits cross-terms (interference terms), which can be suppressed by using
//! the Pseudo-WVD (PWVD) with Gaussian smoothing.
//!
//! # Examples
//!
//! ## Pure-tone WVD
//!
//! ```
//! use scirs2_fft::wigner_ville::{WignerVille, WvdConfig};
//! use std::f64::consts::PI;
//!
//! let n = 64;
//! let fs = 64.0;
//! let freq = 8.0; // Hz
//! let signal: Vec<f64> = (0..n)
//!     .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
//!     .collect();
//!
//! let wv = WignerVille::new();
//! let config = WvdConfig::default();
//! let result = wv.compute_wvd(&signal, fs, &config).expect("WVD should succeed");
//!
//! // WVD output shape
//! assert_eq!(result.n_times(), n);
//! assert_eq!(result.n_freqs(), n);
//! ```
//!
//! ## Pseudo-WVD (reduced cross-terms)
//!
//! ```
//! use scirs2_fft::wigner_ville::{WignerVille, WvdConfig};
//! use std::f64::consts::PI;
//!
//! let n = 64;
//! let fs = 64.0;
//! let signal: Vec<f64> = (0..n)
//!     .map(|i| {
//!         (2.0 * PI * 8.0 * i as f64 / fs).sin()
//!         + (2.0 * PI * 20.0 * i as f64 / fs).sin()
//!     })
//!     .collect();
//!
//! let wv = WignerVille::new();
//! let mut config = WvdConfig::default();
//! config.smooth_window = 8;
//!
//! let result_pwvd = wv.compute_pwvd(&signal, fs, &config).expect("PWVD should succeed");
//! assert_eq!(result_pwvd.n_times(), n);
//! ```

pub mod distribution;
pub mod types;

pub use distribution::{compute_pwvd, compute_wvd, WignerVille};
pub use types::{WvdConfig, WvdResult};

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_pure_tone(n: usize, freq_hz: f64, fs: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f64 / fs).sin())
            .collect()
    }

    #[test]
    fn test_wvd_config_default() {
        let config = WvdConfig::default();
        assert!(config.analytic);
        assert_eq!(config.smooth_window, 0);
        assert!(config.n_freqs.is_none());
    }

    #[test]
    fn test_wvd_output_shape() {
        let n = 32;
        let fs = 32.0;
        let signal = make_pure_tone(n, 4.0, fs);
        let wv = WignerVille::new();
        let config = WvdConfig::default();
        let result = wv
            .compute_wvd(&signal, fs, &config)
            .expect("WVD should succeed");

        // Shape: [n_time][n_freq]
        assert_eq!(
            result.wvd.len(),
            n,
            "WVD row count must equal signal length"
        );
        for row in &result.wvd {
            assert_eq!(row.len(), n, "Each WVD row must have n_freqs columns");
        }
        assert_eq!(result.times.len(), n);
        assert_eq!(result.frequencies.len(), n);
    }

    #[test]
    fn test_wvd_pure_tone_energy_at_correct_frequency() {
        let n = 64;
        let fs = 64.0;
        let freq_hz = 8.0;
        let signal = make_pure_tone(n, freq_hz, fs);
        let wv = WignerVille::new();
        let config = WvdConfig::default();
        let result = wv
            .compute_wvd(&signal, fs, &config)
            .expect("WVD should succeed");

        // Sum energy over all time frames for each frequency bin to get the
        // integrated time-frequency energy profile.
        let n_freqs = result.n_freqs();
        let mut freq_energy = vec![0.0_f64; n_freqs];
        // Skip boundary frames (first and last quarter) to avoid edge artefacts
        let t_start = n / 4;
        let t_end = 3 * n / 4;
        for t in t_start..t_end {
            for (f, &val) in result.wvd[t].iter().enumerate() {
                freq_energy[f] += val.abs();
            }
        }

        // Find the bin with maximum integrated energy (excluding DC bin 0)
        let peak_bin = freq_energy[1..]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i + 1)
            .unwrap_or(0);

        // For the WVD of an analytic signal z(t) = e^{j2πf₀t}, the autocorrelation
        // R(t,τ) = z(t+τ)·z*(t-τ) = e^{j4πf₀τ}, so the WVD peaks at 2*f₀ in the
        // DFT output.
        let expected_bin = (2.0 * freq_hz * n as f64 / fs).round() as usize % n;
        let nominal_bin = (freq_hz * n as f64 / fs).round() as usize;

        // Accept peak near the WVD-doubled bin OR the nominal bin (±2 tolerance)
        let near_expected = (peak_bin as i64 - expected_bin as i64).unsigned_abs() <= 2;
        let near_nominal = (peak_bin as i64 - nominal_bin as i64).unsigned_abs() <= 2;

        assert!(
            near_expected || near_nominal,
            "Peak bin {peak_bin} should be near WVD expected bin {expected_bin} or nominal {nominal_bin}"
        );
    }

    #[test]
    fn test_wvd_result_is_real_valued() {
        // The WVdResult.wvd contains f64 (real-valued) by construction.
        // Just verify the data type compiles and values are finite.
        let n = 16;
        let fs = 16.0;
        let signal = make_pure_tone(n, 2.0, fs);
        let wv = WignerVille::new();
        let config = WvdConfig::default();
        let result = wv
            .compute_wvd(&signal, fs, &config)
            .expect("WVD should succeed");

        for row in &result.wvd {
            for &val in row {
                assert!(val.is_finite(), "WVD values should all be finite");
            }
        }
    }

    #[test]
    fn test_pwvd_output_shape() {
        let n = 32;
        let fs = 32.0;
        let signal = make_pure_tone(n, 4.0, fs);
        let wv = WignerVille::new();
        let mut config = WvdConfig::default();
        config.smooth_window = 5;
        let result = wv
            .compute_pwvd(&signal, fs, &config)
            .expect("PWVD should succeed");

        assert_eq!(result.wvd.len(), n);
        assert_eq!(result.n_freqs(), n);
    }

    #[test]
    fn test_wvd_convenience_function() {
        let n = 32;
        let signal = make_pure_tone(n, 3.0, n as f64);
        let result = compute_wvd(&signal, n as f64).expect("should succeed");
        assert_eq!(result.wvd.len(), n);
    }
}
