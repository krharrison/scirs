// Batched Welch power-spectral-density estimator
//
// Implements the Welch method for multiple channels simultaneously.
// Uses scirs2-fft's `rfft` for the per-segment FFT.
//
// Output shape: [n_channels, nfft / 2 + 1]

use crate::error::{SignalError, SignalResult};
use crate::gpu::batched_stft::{extract_frames, make_window, BatchWindowType};
use scirs2_core::ndarray::Array2;

// ---------------------------------------------------------------------------
// Scaling mode
// ---------------------------------------------------------------------------

/// Welch PSD normalisation strategy.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WelchScaling {
    /// Power spectral density (V²/Hz).  The result is normalised by
    /// `fs * window_power`, where `window_power = sum(w²)`.
    Density,
    /// Power spectrum (V²).  The result is normalised by `window_power` only.
    Spectrum,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`batched_welch_psd`].
#[derive(Debug, Clone)]
pub struct BatchedWelchConfig {
    /// Samples per segment. Default: 256.
    pub nperseg: usize,
    /// Number of overlapping samples between consecutive segments.
    /// Default: 128 (50 % overlap).
    pub noverlap: usize,
    /// FFT size (zero-pad if > `nperseg`). Default: 256.
    pub nfft: usize,
    /// Window function. Default: [`BatchWindowType::Hann`].
    pub window_type: BatchWindowType,
    /// PSD normalisation. Default: [`WelchScaling::Density`].
    pub scaling: WelchScaling,
    /// Sample rate in Hz, used for [`WelchScaling::Density`]. Default: 1.0.
    pub fs: f32,
}

impl Default for BatchedWelchConfig {
    fn default() -> Self {
        Self {
            nperseg: 256,
            noverlap: 128,
            nfft: 256,
            window_type: BatchWindowType::Hann,
            scaling: WelchScaling::Density,
            fs: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Main function
// ---------------------------------------------------------------------------

/// Compute the Welch PSD for a batch of signals.
///
/// # Arguments
///
/// * `signals` – `[n_channels, signal_length]`
/// * `config`  – See [`BatchedWelchConfig`]
///
/// # Returns
///
/// `[n_channels, nfft / 2 + 1]` array of PSD estimates.
pub fn batched_welch_psd(
    signals: &Array2<f32>,
    config: &BatchedWelchConfig,
) -> SignalResult<Array2<f32>> {
    // ---- Validate config ---------------------------------------------------
    if config.nperseg < 2 {
        return Err(SignalError::InvalidArgument(
            "nperseg must be at least 2".into(),
        ));
    }
    if config.nfft < config.nperseg {
        return Err(SignalError::InvalidArgument(
            "nfft must be >= nperseg".into(),
        ));
    }
    if config.noverlap >= config.nperseg {
        return Err(SignalError::InvalidArgument(
            "noverlap must be strictly less than nperseg".into(),
        ));
    }
    if config.fs <= 0.0 {
        return Err(SignalError::InvalidArgument("fs must be positive".into()));
    }

    let n_channels = signals.nrows();
    let signal_length = signals.ncols();
    let n_freq = config.nfft / 2 + 1;
    let hop = config.nperseg - config.noverlap;

    if n_channels == 0 {
        return Err(SignalError::InvalidArgument(
            "signals must have at least one channel".into(),
        ));
    }

    // Pre-compute window (for nperseg length)
    let window = make_window(config.window_type, config.nperseg);
    // Window power for normalisation
    let win_power: f32 = window.iter().map(|&w| w * w).sum();

    // Normalisation denominator
    let norm_denom = match config.scaling {
        WelchScaling::Density => win_power * config.fs,
        WelchScaling::Spectrum => win_power,
    };

    let mut psd = Array2::<f32>::zeros((n_channels, n_freq));

    for ch in 0..n_channels {
        let channel: Vec<f32> = signals.row(ch).to_vec();

        // Frame the signal (no centre-padding for Welch)
        let frames = extract_frames(&channel, config.nperseg, hop, false);
        let n_segments = frames.len();

        if n_segments == 0 {
            // Leave PSD as zeros; signal is too short
            continue;
        }

        let mut accum = vec![0.0f64; n_freq];

        for frame in &frames {
            // Zero-pad frame to nfft and apply window
            let mut padded = vec![0.0f64; config.nfft];
            for (i, (&s, &w)) in frame.iter().zip(window.iter()).enumerate() {
                padded[i] = (s * w) as f64;
            }

            // RFFT
            let spectrum = scirs2_fft::rfft(&padded, None)
                .map_err(|e| SignalError::ComputationError(format!("rfft error in Welch: {e}")))?;

            // Accumulate |FFT|^2
            for (k, c) in spectrum.iter().take(n_freq).enumerate() {
                accum[k] += c.re * c.re + c.im * c.im;
            }
        }

        // Average over segments and apply normalisation
        let scale = 1.0 / (n_segments as f64 * norm_denom as f64);

        // Handle the DC (k=0) and Nyquist (k=nfft/2) bins: they appear only
        // once in the one-sided spectrum so they should NOT be doubled.
        // Interior bins represent both positive and negative frequencies, so
        // we multiply by 2.
        let nyquist_bin = config.nfft / 2; // = n_freq - 1

        for k in 0..n_freq {
            let two_sided_factor = if k == 0 || k == nyquist_bin {
                1.0f64
            } else {
                2.0f64
            };
            psd[[ch, k]] = (accum[k] * scale * two_sided_factor) as f32;
        }
    }

    Ok(psd)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array2;
    use std::f32::consts::PI;

    fn sine_batch(freq: f32, fs: f32, n: usize, n_ch: usize) -> Array2<f32> {
        let sig: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / fs).sin())
            .collect();
        Array2::from_shape_fn((n_ch, n), |(_, j)| sig[j])
    }

    #[test]
    fn test_batched_welch_output_shape() {
        let config = BatchedWelchConfig {
            nperseg: 64,
            noverlap: 32,
            nfft: 64,
            ..Default::default()
        };
        let signals = Array2::zeros((3, 512));
        let psd = batched_welch_psd(&signals, &config).expect("welch");
        assert_eq!(psd.shape(), &[3, 33]); // nfft/2+1 = 33
    }

    #[test]
    fn test_batched_welch_single_channel_reasonable_psd() {
        // White noise: PSD should be roughly flat and positive
        let n = 1024;
        let config = BatchedWelchConfig {
            nperseg: 128,
            noverlap: 64,
            nfft: 128,
            fs: 1000.0,
            scaling: WelchScaling::Density,
            ..Default::default()
        };
        // Use a deterministic signal: square wave (energy spread across odd harmonics)
        let sq: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let signals = Array2::from_shape_fn((1, n), |(_, j)| sq[j]);
        let psd = batched_welch_psd(&signals, &config).expect("welch");

        // All PSD values should be non-negative
        for &v in psd.iter() {
            assert!(v >= 0.0, "Negative PSD value: {v}");
        }

        // Total power estimate should be > 0
        let total: f32 = psd.iter().sum();
        assert!(total > 0.0, "PSD sums to zero");
    }

    #[test]
    fn test_batched_welch_scaling() {
        // For a pure tone with amplitude 1, Welch Density PSD peak ≈ 0.5/Δf,
        // Welch Spectrum peak ≈ 0.5.  Here we just verify that Density and
        // Spectrum give different magnitudes.
        let fs = 1024.0;
        let config_density = BatchedWelchConfig {
            nperseg: 64,
            noverlap: 32,
            nfft: 64,
            fs,
            scaling: WelchScaling::Density,
            ..Default::default()
        };
        let config_spectrum = BatchedWelchConfig {
            scaling: WelchScaling::Spectrum,
            ..config_density.clone()
        };
        let signals = sine_batch(100.0, fs, 512, 1);

        let psd_d = batched_welch_psd(&signals, &config_density).expect("density");
        let psd_s = batched_welch_psd(&signals, &config_spectrum).expect("spectrum");

        let sum_d: f32 = psd_d.iter().sum();
        let sum_s: f32 = psd_s.iter().sum();

        // They should differ by approximately fs (the bin-width factor)
        // sum_density / sum_spectrum ≈ 1/fs * (nfft/2+1) when averaged
        // Simply check they are different:
        assert!(
            (sum_d - sum_s).abs() > 1e-6,
            "Density and Spectrum scalings should differ"
        );
    }

    #[test]
    fn test_batched_welch_zero_signal_zero_psd() {
        let config = BatchedWelchConfig::default();
        let signals = Array2::<f32>::zeros((2, 1024));
        let psd = batched_welch_psd(&signals, &config).expect("welch");
        for &v in psd.iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-10);
        }
    }
}
