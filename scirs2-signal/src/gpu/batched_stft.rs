// Batched Short-Time Fourier Transform (STFT)
//
// Implements a batch-oriented STFT suitable for multi-channel audio / sensor
// data.  All FFT work is delegated to scirs2-fft (which wraps OxiFFT) so the
// heavy lifting is done in optimised pure-Rust code.
//
// Output shape convention: [n_channels, n_freq_bins, n_frames]
//   n_freq_bins = fft_size / 2 + 1   (one-sided spectrum for real input)
//   n_frames    = (padded_length - fft_size) / hop_size + 1

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::num_complex::Complex32;
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Window type
// ---------------------------------------------------------------------------

/// Window functions available for the batched STFT.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchWindowType {
    /// Hann (von Hann) window – good sidelobe attenuation
    Hann,
    /// Hamming window – slightly lower sidelobe vs. Hann
    Hamming,
    /// Blackman window – very low sidelobe at the cost of wider main lobe
    Blackman,
    /// Rectangular (boxcar) window – no windowing
    Rectangular,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`BatchedStftEngine`].
#[derive(Debug, Clone)]
pub struct BatchedStftConfig {
    /// FFT size in samples. Default: 512.
    pub fft_size: usize,
    /// Hop (stride) between successive frames. Default: 128.
    pub hop_size: usize,
    /// Window function applied to each frame. Default: [`BatchWindowType::Hann`].
    pub window_type: BatchWindowType,
    /// When `true` the input signal is zero-padded by `fft_size / 2` on each
    /// side so that the first frame is centred on sample 0. Default: `true`.
    pub center: bool,
}

impl Default for BatchedStftConfig {
    fn default() -> Self {
        Self {
            fft_size: 512,
            hop_size: 128,
            window_type: BatchWindowType::Hann,
            center: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Batch-optimised STFT engine.
///
/// Create once with [`BatchedStftEngine::new`], then call
/// [`BatchedStftEngine::compute_batch`] for each batch of signals.
pub struct BatchedStftEngine {
    config: BatchedStftConfig,
    /// Pre-computed window coefficients (length = `config.fft_size`)
    window: Array1<f32>,
}

impl BatchedStftEngine {
    /// Construct a new engine, pre-computing the window function.
    pub fn new(config: BatchedStftConfig) -> SignalResult<Self> {
        if config.fft_size < 2 {
            return Err(SignalError::InvalidArgument(
                "fft_size must be at least 2".into(),
            ));
        }
        if config.hop_size == 0 {
            return Err(SignalError::InvalidArgument(
                "hop_size must be at least 1".into(),
            ));
        }
        let window = make_window(config.window_type, config.fft_size);
        Ok(Self { config, window })
    }

    /// Compute the number of one-sided frequency bins.
    #[inline]
    pub fn n_freq_bins(&self) -> usize {
        self.config.fft_size / 2 + 1
    }

    /// Compute the number of STFT frames for a signal of the given length.
    pub fn n_frames(&self, signal_length: usize) -> usize {
        let pad = if self.config.center {
            self.config.fft_size
        } else {
            0
        };
        let padded = signal_length + pad;
        if padded < self.config.fft_size {
            return 0;
        }
        (padded - self.config.fft_size) / self.config.hop_size + 1
    }

    /// Compute the batched STFT.
    ///
    /// # Arguments
    ///
    /// * `signals` – 2-D array with shape `[n_channels, signal_length]`.
    ///
    /// # Returns
    ///
    /// 3-D array with shape `[n_channels, n_freq_bins, n_frames]`.
    pub fn compute_batch(&self, signals: &Array2<f32>) -> SignalResult<Array3<Complex32>> {
        let n_channels = signals.nrows();
        let signal_length = signals.ncols();

        if n_channels == 0 {
            return Err(SignalError::InvalidArgument(
                "signals must have at least one channel".into(),
            ));
        }

        let n_frames = self.n_frames(signal_length);
        let n_freq = self.n_freq_bins();

        if n_frames == 0 {
            return Err(SignalError::InvalidArgument(format!(
                "Signal length {signal_length} is too short for fft_size {}",
                self.config.fft_size
            )));
        }

        // Allocate output: [n_channels, n_freq, n_frames]
        let mut output = Array3::<Complex32>::zeros((n_channels, n_freq, n_frames));

        for ch in 0..n_channels {
            let channel_slice: Vec<f32> = signals.row(ch).to_vec();
            let frames = extract_frames(
                &channel_slice,
                self.config.fft_size,
                self.config.hop_size,
                self.config.center,
            );

            for (frame_idx, frame) in frames.iter().enumerate() {
                // Apply window
                let windowed: Vec<f64> = frame
                    .iter()
                    .zip(self.window.iter())
                    .map(|(&s, &w)| (s * w) as f64)
                    .collect();

                // FFT via scirs2-fft (returns Complex64, length = fft_size/2 + 1)
                let spectrum = scirs2_fft::rfft(&windowed, None)
                    .map_err(|e| SignalError::ComputationError(format!("rfft error: {e}")))?;

                // Store as Complex32
                for (freq_idx, c) in spectrum.iter().take(n_freq).enumerate() {
                    output[[ch, freq_idx, frame_idx]] = Complex32::new(c.re as f32, c.im as f32);
                }
            }
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Build a window of `size` samples for the given window type.
pub(crate) fn make_window(window_type: BatchWindowType, size: usize) -> Array1<f32> {
    let mut w = vec![0.0f32; size];
    let n = size as f32;
    match window_type {
        BatchWindowType::Hann => {
            for (i, v) in w.iter_mut().enumerate() {
                *v = 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1.0)).cos());
            }
        }
        BatchWindowType::Hamming => {
            for (i, v) in w.iter_mut().enumerate() {
                *v = 0.54 - 0.46 * (2.0 * PI * i as f32 / (n - 1.0)).cos();
            }
        }
        BatchWindowType::Blackman => {
            for (i, v) in w.iter_mut().enumerate() {
                let t = i as f32;
                *v = 0.42 - 0.5 * (2.0 * PI * t / (n - 1.0)).cos()
                    + 0.08 * (4.0 * PI * t / (n - 1.0)).cos();
            }
        }
        BatchWindowType::Rectangular => {
            for v in w.iter_mut() {
                *v = 1.0;
            }
        }
    }
    Array1::from_vec(w)
}

/// Extract overlapping frames from `signal`.
///
/// If `center` is `true`, the signal is zero-padded by `frame_size / 2` on
/// each side before framing, so that the first frame is centred on sample 0.
pub(crate) fn extract_frames(
    signal: &[f32],
    frame_size: usize,
    hop: usize,
    center: bool,
) -> Vec<Vec<f32>> {
    let padded: Vec<f32> = if center {
        let pad = frame_size / 2;
        let mut p = vec![0.0f32; pad];
        p.extend_from_slice(signal);
        p.extend(vec![0.0f32; pad]);
        p
    } else {
        signal.to_vec()
    };

    if padded.len() < frame_size {
        return Vec::new();
    }

    let n_frames = (padded.len() - frame_size) / hop + 1;
    let mut frames = Vec::with_capacity(n_frames);

    for i in 0..n_frames {
        let start = i * hop;
        let end = start + frame_size;
        frames.push(padded[start..end].to_vec());
    }

    frames
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

    fn sine_signal(freq_hz: f32, sample_rate: f32, n_samples: usize) -> Vec<f32> {
        (0..n_samples)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn test_batched_stft_output_shape_single_channel() {
        let config = BatchedStftConfig {
            fft_size: 64,
            hop_size: 16,
            window_type: BatchWindowType::Hann,
            center: false,
        };
        let engine = BatchedStftEngine::new(config.clone()).expect("engine");
        let n_samples = 256;
        let signals = Array2::from_shape_fn((1, n_samples), |(_, j)| j as f32 * 0.01);
        let result = engine.compute_batch(&signals).expect("stft");

        let expected_frames = engine.n_frames(n_samples);
        let expected_freq = engine.n_freq_bins();
        assert_eq!(result.shape(), &[1, expected_freq, expected_frames]);
    }

    #[test]
    fn test_batched_stft_output_shape_multi_channel() {
        let config = BatchedStftConfig {
            fft_size: 32,
            hop_size: 8,
            window_type: BatchWindowType::Hamming,
            center: true,
        };
        let engine = BatchedStftEngine::new(config.clone()).expect("engine");
        let n_channels = 4;
        let n_samples = 128;
        let signals = Array2::zeros((n_channels, n_samples));
        let result = engine.compute_batch(&signals).expect("stft");

        assert_eq!(result.shape()[0], n_channels);
        assert_eq!(result.shape()[1], engine.n_freq_bins());
        assert_eq!(result.shape()[2], engine.n_frames(n_samples));
    }

    #[test]
    fn test_batched_stft_n_freq_bins() {
        for fft_size in [16, 32, 64, 128, 256, 512] {
            let config = BatchedStftConfig {
                fft_size,
                hop_size: fft_size / 4,
                ..Default::default()
            };
            let engine = BatchedStftEngine::new(config).expect("engine");
            assert_eq!(engine.n_freq_bins(), fft_size / 2 + 1);
        }
    }

    #[test]
    fn test_batched_stft_pure_tone_peak() {
        // A 1 kHz tone sampled at 8 kHz should have energy at bin ~8
        let fs = 8000.0f32;
        let tone_freq = 1000.0f32;
        let fft_size = 256;
        let n_samples = 2048;

        let signal = sine_signal(tone_freq, fs, n_samples);
        let config = BatchedStftConfig {
            fft_size,
            hop_size: fft_size / 2,
            window_type: BatchWindowType::Hann,
            center: false,
        };
        let engine = BatchedStftEngine::new(config).expect("engine");
        let signals = Array2::from_shape_fn((1, n_samples), |(_, j)| signal[j]);
        let stft = engine.compute_batch(&signals).expect("stft");

        // Expected bin for tone_freq: round(tone_freq * fft_size / fs)
        let expected_bin = ((tone_freq * fft_size as f32 / fs).round()) as usize;

        // Pick the middle frame
        let n_frames = stft.shape()[2];
        let mid_frame = n_frames / 2;

        // Find the maximum magnitude bin
        let magnitudes: Vec<f32> = (0..engine.n_freq_bins())
            .map(|b| {
                let c = stft[[0, b, mid_frame]];
                (c.re * c.re + c.im * c.im).sqrt()
            })
            .collect();

        let peak_bin = magnitudes
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("cmp"))
            .map(|(i, _)| i)
            .expect("peak");

        // Allow ±1 bin tolerance
        assert!(
            peak_bin.abs_diff(expected_bin) <= 1,
            "peak bin {peak_bin} not near expected {expected_bin}"
        );
    }

    #[test]
    fn test_frame_signals_hop_size() {
        let signal: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let frames = extract_frames(&signal, 10, 5, false);
        // n_frames = (100 - 10) / 5 + 1 = 19
        assert_eq!(frames.len(), 19);
        assert_eq!(frames[0].len(), 10);
        assert_abs_diff_eq!(frames[0][0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(frames[1][0], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_batched_stft_zero_signal_zero_output() {
        let config = BatchedStftConfig {
            fft_size: 32,
            hop_size: 8,
            center: false,
            ..Default::default()
        };
        let engine = BatchedStftEngine::new(config).expect("engine");
        let signals = Array2::zeros((2, 128));
        let stft = engine.compute_batch(&signals).expect("stft");
        for &v in stft.iter() {
            assert_abs_diff_eq!(v.re, 0.0, epsilon = 1e-6);
            assert_abs_diff_eq!(v.im, 0.0, epsilon = 1e-6);
        }
    }
}
