//! Streaming Short-Time Fourier Transform processor.
//!
//! Maintains a ring buffer of incoming samples.  Each time the ring buffer
//! accumulates `hop_size` new samples beyond the previous analysis point, it
//! extracts the most recent `window_size` samples, applies a window function,
//! runs a radix-2 FFT (zero-padded to the next power of two if necessary), and
//! returns a [`SpectralFrame`].
//!
//! The radix-2 Cooley–Tukey implementation is self-contained (no external FFT
//! crate) and supports any input length by zero-padding to the next power of
//! two.

use std::f32::consts::PI;

// ------------------------------------------------------------------
// Configuration types
// ------------------------------------------------------------------

/// Window functions available for STFT analysis.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowFn {
    /// Rectangular (no windowing).
    Rectangular,
    /// Hann (raised cosine).
    #[default]
    Hann,
    /// Hamming.
    Hamming,
    /// Blackman.
    Blackman,
}

/// Configuration for the streaming FFT processor.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct StreamingFftConfig {
    /// FFT analysis window size in samples (must be a power of two).
    /// Default: `512`.
    pub window_size: usize,
    /// Number of new samples between successive analysis frames.
    /// Default: `256` (50 % overlap).
    pub hop_size: usize,
    /// Window function to apply before the FFT.  Default: [`WindowFn::Hann`].
    pub window_fn: WindowFn,
    /// Number of audio/sensor channels to process simultaneously.
    /// Default: `1`.
    pub n_channels: usize,
}

impl Default for StreamingFftConfig {
    fn default() -> Self {
        Self {
            window_size: 512,
            hop_size: 256,
            window_fn: WindowFn::Hann,
            n_channels: 1,
        }
    }
}

// ------------------------------------------------------------------
// SpectralFrame
// ------------------------------------------------------------------

/// One analysis frame produced by the streaming FFT processor.
#[derive(Debug, Clone)]
pub struct SpectralFrame {
    /// Magnitude spectrum (linear).  Length: `window_size / 2 + 1` bins.
    pub magnitudes: Vec<f32>,
    /// Phase spectrum in radians.  Length: `window_size / 2 + 1` bins.
    pub phases: Vec<f32>,
    /// Position (in samples) of the first sample of the window that produced
    /// this frame, measured from the start of the stream.
    pub timestamp: usize,
}

// ------------------------------------------------------------------
// StreamingFftProcessor
// ------------------------------------------------------------------

/// Incremental FFT processor that emits [`SpectralFrame`]s as samples arrive.
///
/// # Example
///
/// ```
/// use scirs2_wasm::streaming_fft::{StreamingFftConfig, StreamingFftProcessor};
///
/// let mut proc = StreamingFftProcessor::new(StreamingFftConfig::default());
/// // Push one full window + one hop worth of samples.
/// let samples: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).sin()).collect();
/// let frames = proc.push_samples(&samples);
/// assert!(!frames.is_empty());
/// ```
pub struct StreamingFftProcessor {
    config: StreamingFftConfig,
    /// Ring buffer holding the most recent `window_size` samples.
    ring_buffer: Vec<f32>,
    /// Write position (next slot to fill) in the ring buffer.
    buffer_pos: usize,
    /// Total samples consumed so far.
    sample_count: usize,
    /// How many new samples have arrived since the last analysis frame.
    samples_since_last_frame: usize,
}

impl StreamingFftProcessor {
    /// Construct a new processor from the given `config`.
    ///
    /// # Panics (debug only)
    /// If `config.window_size` is not a power of two.
    pub fn new(config: StreamingFftConfig) -> Self {
        debug_assert!(
            config.window_size.is_power_of_two(),
            "window_size must be a power of two, got {}",
            config.window_size
        );
        let ring_buffer = vec![0.0_f32; config.window_size];
        Self {
            ring_buffer,
            buffer_pos: 0,
            sample_count: 0,
            samples_since_last_frame: 0,
            config,
        }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Push a batch of new samples.
    ///
    /// Returns all [`SpectralFrame`]s that became ready during this call.
    /// A frame is emitted each time `hop_size` new samples have arrived since
    /// the last emission.
    pub fn push_samples(&mut self, samples: &[f32]) -> Vec<SpectralFrame> {
        let mut frames = Vec::new();
        let hop = self.config.hop_size;

        for &sample in samples {
            // Write sample into ring buffer (modular).
            self.ring_buffer[self.buffer_pos % self.config.window_size] = sample;
            self.buffer_pos += 1;
            self.sample_count += 1;
            self.samples_since_last_frame += 1;

            if self.samples_since_last_frame >= hop {
                // Only emit a frame once the ring buffer is fully populated.
                if self.sample_count >= self.config.window_size {
                    let window_data = self.extract_window();
                    let frame = self.process_window(&window_data);
                    frames.push(frame);
                }
                self.samples_since_last_frame = 0;
            }
        }

        frames
    }

    /// Number of new samples needed to trigger the next frame.
    pub fn pending_samples(&self) -> usize {
        self.config
            .hop_size
            .saturating_sub(self.samples_since_last_frame)
    }

    /// Reset the processor to its initial state (clears the ring buffer).
    pub fn reset(&mut self) {
        self.ring_buffer.fill(0.0);
        self.buffer_pos = 0;
        self.sample_count = 0;
        self.samples_since_last_frame = 0;
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Extract `window_size` samples from the ring buffer in chronological order.
    fn extract_window(&self) -> Vec<f32> {
        let ws = self.config.window_size;
        let mut out = Vec::with_capacity(ws);
        // The oldest sample sits at `buffer_pos % ws` (oldest write position).
        let start = self.buffer_pos % ws;
        for i in 0..ws {
            out.push(self.ring_buffer[(start + i) % ws]);
        }
        out
    }

    /// Run the FFT pipeline on one window of samples and produce a frame.
    fn process_window(&self, window_data: &[f32]) -> SpectralFrame {
        let timestamp = self.sample_count.saturating_sub(self.config.window_size);

        let windowed = self.apply_window(window_data);
        let spectrum = self.fft_radix2(&windowed);

        // Only keep the positive-frequency bins (DC … Nyquist).
        let n_bins = self.config.window_size / 2 + 1;
        let mut magnitudes = Vec::with_capacity(n_bins);
        let mut phases = Vec::with_capacity(n_bins);

        for (re, im) in spectrum.iter().take(n_bins) {
            magnitudes.push((re * re + im * im).sqrt());
            phases.push(im.atan2(*re));
        }

        SpectralFrame {
            magnitudes,
            phases,
            timestamp,
        }
    }

    /// Apply the configured window function to `data`.
    fn apply_window(&self, data: &[f32]) -> Vec<f32> {
        let n = data.len();
        match self.config.window_fn {
            WindowFn::Rectangular => data.to_vec(),
            WindowFn::Hann => data
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let w = 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos());
                    x * w
                })
                .collect(),
            WindowFn::Hamming => data
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let w = 0.54 - 0.46 * (2.0 * PI * i as f32 / (n - 1) as f32).cos();
                    x * w
                })
                .collect(),
            WindowFn::Blackman => data
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let phase = 2.0 * PI * i as f32 / (n - 1) as f32;
                    let w = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
                    x * w
                })
                .collect(),
        }
    }

    /// Radix-2 Cooley–Tukey FFT (decimation-in-time, iterative).
    ///
    /// If `data.len()` is not a power of two the input is **zero-padded** to the
    /// next power of two before the transform.
    ///
    /// Returns a `Vec<(f32, f32)>` of `(real, imag)` pairs, length equal to the
    /// (possibly padded) transform size.
    pub(crate) fn fft_radix2(&self, data: &[f32]) -> Vec<(f32, f32)> {
        let n_orig = data.len();
        // Zero-pad to next power of two.
        let n = if n_orig.is_power_of_two() {
            n_orig
        } else {
            n_orig.next_power_of_two()
        };

        // Build complex array (re, im) — padded with zeros.
        let mut a: Vec<(f32, f32)> = data.iter().map(|&x| (x, 0.0)).collect();
        a.resize(n, (0.0, 0.0));

        // Bit-reversal permutation.
        let log2_n = n.trailing_zeros() as usize;
        for i in 0..n {
            let j = bit_reverse(i, log2_n);
            if j > i {
                a.swap(i, j);
            }
        }

        // Cooley–Tukey butterfly stages.
        let mut len = 2_usize;
        while len <= n {
            let half = len / 2;
            let angle = -2.0 * PI / len as f32;
            let w_base = (angle.cos(), angle.sin()); // principal root of unity

            for start in (0..n).step_by(len) {
                let mut wr = 1.0_f32;
                let mut wi = 0.0_f32;
                for k in 0..half {
                    let u = a[start + k];
                    let v = a[start + k + half];
                    // v * w
                    let vw = (v.0 * wr - v.1 * wi, v.0 * wi + v.1 * wr);

                    a[start + k] = (u.0 + vw.0, u.1 + vw.1);
                    a[start + k + half] = (u.0 - vw.0, u.1 - vw.1);

                    // Advance twiddle factor.
                    let new_wr = wr * w_base.0 - wi * w_base.1;
                    let new_wi = wr * w_base.1 + wi * w_base.0;
                    wr = new_wr;
                    wi = new_wi;
                }
            }
            len <<= 1;
        }

        a
    }
}

/// Reverse the lowest `bits` bits of `x`.
#[inline]
fn bit_reverse(x: usize, bits: usize) -> usize {
    let mut result = 0_usize;
    let mut v = x;
    for _ in 0..bits {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_fft_config_default() {
        let cfg = StreamingFftConfig::default();
        assert_eq!(cfg.window_size, 512);
        assert_eq!(cfg.hop_size, 256);
        assert_eq!(cfg.window_fn, WindowFn::Hann);
        assert_eq!(cfg.n_channels, 1);
    }

    #[test]
    fn test_push_samples_returns_frames_when_enough_data() {
        let cfg = StreamingFftConfig {
            window_size: 64,
            hop_size: 32,
            ..Default::default()
        };
        let mut proc = StreamingFftProcessor::new(cfg);
        // Push exactly window_size + hop_size samples → should get at least 1 frame.
        let samples: Vec<f32> = (0..96).map(|i| (i as f32 * 0.1).sin()).collect();
        let frames = proc.push_samples(&samples);
        assert!(!frames.is_empty(), "expected at least one frame");
    }

    #[test]
    fn test_spectral_frame_magnitude_length() {
        let cfg = StreamingFftConfig {
            window_size: 64,
            hop_size: 32,
            ..Default::default()
        };
        let mut proc = StreamingFftProcessor::new(cfg);
        let samples: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
        let frames = proc.push_samples(&samples);
        assert!(!frames.is_empty());
        for frame in &frames {
            assert_eq!(
                frame.magnitudes.len(),
                33, // window_size/2 + 1 = 64/2 + 1
                "magnitudes length wrong"
            );
        }
    }

    #[test]
    fn test_fft_pure_tone_peak_at_correct_bin() {
        // Generate a pure tone at bin k_tone = 4 of a 64-point FFT.
        let n = 64_usize;
        let k_tone = 4_usize;
        let data: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * k_tone as f32 * i as f32 / n as f32).cos())
            .collect();

        let cfg = StreamingFftConfig {
            window_size: n,
            ..Default::default()
        };
        let proc = StreamingFftProcessor::new(cfg);
        // Use a rectangular window for exact frequency recovery.
        let spectrum = proc.fft_radix2(&data);
        let magnitudes: Vec<f32> = spectrum
            .iter()
            .take(n / 2 + 1)
            .map(|(re, im)| (re * re + im * im).sqrt())
            .collect();

        let peak_bin = magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        assert_eq!(peak_bin, k_tone, "peak at wrong bin: {peak_bin}");
    }

    #[test]
    fn test_reset_clears_buffer() {
        let cfg = StreamingFftConfig {
            window_size: 64,
            hop_size: 32,
            ..Default::default()
        };
        let mut proc = StreamingFftProcessor::new(cfg);
        let samples: Vec<f32> = vec![1.0; 96];
        let _ = proc.push_samples(&samples);
        assert!(proc.sample_count > 0);
        proc.reset();
        assert_eq!(proc.sample_count, 0);
        assert!(proc.ring_buffer.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_hann_window_applied_correctly() {
        let cfg = StreamingFftConfig {
            window_size: 8,
            window_fn: WindowFn::Hann,
            ..Default::default()
        };
        let proc = StreamingFftProcessor::new(cfg);
        let data = vec![1.0_f32; 8];
        let windowed = proc.apply_window(&data);
        // Hann window at i=0 and i=N-1 should be ~0.
        assert!(windowed[0].abs() < 1e-6, "Hann w[0] should be ~0");
        assert!(windowed[7].abs() < 1e-6, "Hann w[N-1] should be ~0");
        // Interior values should be non-zero.
        assert!(
            windowed[4] > 0.9,
            "Hann w[4] should be near 1.0: {}",
            windowed[4]
        );
    }

    #[test]
    fn test_fft_radix2_handles_non_power_of_two_via_padding() {
        let cfg = StreamingFftConfig::default();
        let proc = StreamingFftProcessor::new(cfg);
        // 10-element input → padded to 16.
        let data = vec![1.0_f32; 10];
        let result = proc.fft_radix2(&data);
        assert_eq!(result.len(), 16, "should be padded to next power of two");
    }

    #[test]
    fn test_pending_samples_decreases_as_samples_arrive() {
        let cfg = StreamingFftConfig {
            window_size: 64,
            hop_size: 32,
            ..Default::default()
        };
        let mut proc = StreamingFftProcessor::new(cfg);
        let initial_pending = proc.pending_samples();
        // Push some samples (fewer than hop_size).
        let samples = vec![0.0_f32; 10];
        let _ = proc.push_samples(&samples);
        // pending_samples should be less than before.
        assert!(
            proc.pending_samples() < initial_pending || proc.pending_samples() == 0,
            "pending_samples did not decrease"
        );
    }
}
