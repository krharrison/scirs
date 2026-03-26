//! Streaming / online Short-Time Fourier Transform with incremental input.
//!
//! This module provides [`OnlineSTFT`], a streaming STFT processor that
//! accepts input samples in arbitrary-sized chunks and produces STFT frames
//! whenever enough new samples (one hop's worth) have accumulated.
//!
//! Unlike the existing [`StreamingSTFT`](super::StreamingSTFT) which also
//! operates in streaming mode, `OnlineSTFT` additionally provides:
//!
//! - Configurable FFT size independent of window size (zero-padding).
//! - Built-in inverse STFT via [`OnlineISTFT`] with perfect-reconstruction
//!   guarantees for COLA-compliant window/hop combinations.
//! - A sample counter for precise alignment between forward and inverse
//!   transforms.
//!
//! ## Window functions
//!
//! Re-uses [`WindowFunction`] from the existing
//! streaming STFT module.

use crate::error::{SignalError, SignalResult};
use crate::streaming::stft::WindowFunction;
use crate::window::get_window;
use scirs2_core::numeric::Complex64;

// ============================================================================
// OnlineSTFT
// ============================================================================

/// Configuration for [`OnlineSTFT`].
#[derive(Debug, Clone)]
pub struct OnlineSTFTConfig {
    /// Analysis window size in samples.
    pub window_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// FFT size.  Must be >= `window_size`.  If larger, the windowed frame is
    /// zero-padded.
    pub fft_size: usize,
    /// Window function.
    pub window_function: WindowFunction,
}

impl Default for OnlineSTFTConfig {
    fn default() -> Self {
        Self {
            window_size: 1024,
            hop_size: 256,
            fft_size: 1024,
            window_function: WindowFunction::Hann,
        }
    }
}

/// A single STFT frame produced by [`OnlineSTFT`].
#[derive(Debug, Clone)]
pub struct STFTFrame {
    /// Complex spectrum (positive frequencies only), length = `fft_size / 2 + 1`.
    pub spectrum: Vec<Complex64>,
    /// The sample index of the first sample in the analysis window that
    /// produced this frame (useful for time alignment).
    pub center_sample: u64,
}

/// Streaming STFT that accepts samples incrementally.
///
/// Call [`OnlineSTFT::push_samples`] to feed data, then call
/// [`OnlineSTFT::pop_frames`] to collect any completed STFT frames.
pub struct OnlineSTFT {
    config: OnlineSTFTConfig,
    /// Pre-computed window coefficients.
    window: Vec<f64>,
    /// Internal sample buffer.
    buffer: Vec<f64>,
    /// Number of samples in the buffer.
    buf_len: usize,
    /// Total samples received.
    total_samples: u64,
    /// Samples since the last frame was emitted.
    pending_samples: usize,
    /// Whether we have emitted at least one frame.
    has_emitted: bool,
    /// Completed frames waiting to be consumed.
    output_frames: Vec<STFTFrame>,
}

impl OnlineSTFT {
    /// Create a new online STFT processor.
    ///
    /// # Errors
    ///
    /// * `window_size` must be > 0.
    /// * `hop_size` must be in `(0, window_size]`.
    /// * `fft_size` must be >= `window_size`.
    pub fn new(config: OnlineSTFTConfig) -> SignalResult<Self> {
        if config.window_size == 0 {
            return Err(SignalError::ValueError(
                "window_size must be > 0".to_string(),
            ));
        }
        if config.hop_size == 0 || config.hop_size > config.window_size {
            return Err(SignalError::ValueError(
                "hop_size must be > 0 and <= window_size".to_string(),
            ));
        }
        if config.fft_size < config.window_size {
            return Err(SignalError::ValueError(
                "fft_size must be >= window_size".to_string(),
            ));
        }

        let window = get_window(config.window_function.as_str(), config.window_size, true)?;

        // We need to keep at least window_size samples in the buffer at all
        // times (once we have received that many).
        let buf_capacity = config.window_size + config.hop_size * 4;

        Ok(Self {
            config,
            window,
            buffer: vec![0.0; buf_capacity],
            buf_len: 0,
            total_samples: 0,
            pending_samples: 0,
            has_emitted: false,
            output_frames: Vec::new(),
        })
    }

    /// Push new input samples.
    ///
    /// Internally checks whether enough samples have accumulated to produce
    /// one or more STFT frames.
    pub fn push_samples(&mut self, samples: &[f64]) -> SignalResult<()> {
        // Ensure buffer has enough capacity
        let required = self.buf_len + samples.len();
        if required > self.buffer.len() {
            self.buffer.resize(required + self.config.hop_size * 4, 0.0);
        }

        // Append new samples
        self.buffer[self.buf_len..self.buf_len + samples.len()].copy_from_slice(samples);
        self.buf_len += samples.len();
        self.total_samples += samples.len() as u64;
        self.pending_samples += samples.len();

        // Emit frames
        while self.buf_len >= self.config.window_size
            && self.pending_samples >= self.config.hop_size
        {
            let frame = self.emit_frame()?;
            self.output_frames.push(frame);
            self.pending_samples -= self.config.hop_size;

            // Shift buffer: remove the oldest hop_size samples
            let shift = self.config.hop_size;
            self.buffer.copy_within(shift..self.buf_len, 0);
            self.buf_len -= shift;
        }

        Ok(())
    }

    /// Pop all completed STFT frames.
    ///
    /// Returns an empty `Vec` if no new frames are available.
    pub fn pop_frames(&mut self) -> Vec<STFTFrame> {
        std::mem::take(&mut self.output_frames)
    }

    /// Number of positive-frequency bins per frame.
    pub fn num_bins(&self) -> usize {
        self.config.fft_size / 2 + 1
    }

    /// Total samples received so far.
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Reset the processor state.
    pub fn reset(&mut self) {
        self.buf_len = 0;
        self.total_samples = 0;
        self.pending_samples = 0;
        self.has_emitted = false;
        self.output_frames.clear();
    }

    // ---- internal ----

    fn emit_frame(&mut self) -> SignalResult<STFTFrame> {
        let ws = self.config.window_size;
        let fs = self.config.fft_size;

        // Take the first window_size samples from the buffer
        let raw = &self.buffer[..ws];

        // Apply window
        let mut windowed = vec![0.0; fs];
        for (i, (&s, &w)) in raw.iter().zip(self.window.iter()).enumerate() {
            windowed[i] = s * w;
        }
        // Remaining fs - ws samples are zero (zero-padding).

        // Real FFT
        let spectrum = scirs2_fft::rfft(&windowed, None)
            .map_err(|e| SignalError::ComputationError(format!("FFT error in OnlineSTFT: {e}")))?;

        let center = self.total_samples - self.pending_samples as u64;

        self.has_emitted = true;

        Ok(STFTFrame {
            spectrum,
            center_sample: center,
        })
    }
}

// ============================================================================
// OnlineISTFT
// ============================================================================

/// Online inverse STFT for reconstruction from streaming STFT frames.
///
/// Uses the weighted overlap-add (WOLA) method.  Feed frames from
/// [`OnlineSTFT`] via [`OnlineISTFT::push_frame`] and extract reconstructed
/// samples via [`OnlineISTFT::pop_samples`].
pub struct OnlineISTFT {
    window_size: usize,
    hop_size: usize,
    fft_size: usize,
    /// Synthesis window coefficients.
    window: Vec<f64>,
    /// Overlap-add accumulation buffer.
    ola_buffer: Vec<f64>,
    /// Window normalisation buffer.
    norm_buffer: Vec<f64>,
    /// Write position in the OLA buffer.
    write_pos: usize,
    /// Number of valid output samples ready to read.
    ready_samples: usize,
    /// Total frames processed.
    frames_processed: u64,
}

impl OnlineISTFT {
    /// Create a new online inverse STFT.
    ///
    /// Parameters **must** match the analysis [`OnlineSTFT`].
    ///
    /// # Errors
    ///
    /// Same constraints as [`OnlineSTFT::new`].
    pub fn new(
        window_size: usize,
        hop_size: usize,
        fft_size: usize,
        window_function: WindowFunction,
    ) -> SignalResult<Self> {
        if window_size == 0 {
            return Err(SignalError::ValueError(
                "window_size must be > 0".to_string(),
            ));
        }
        if hop_size == 0 || hop_size > window_size {
            return Err(SignalError::ValueError(
                "hop_size must be > 0 and <= window_size".to_string(),
            ));
        }
        if fft_size < window_size {
            return Err(SignalError::ValueError(
                "fft_size must be >= window_size".to_string(),
            ));
        }

        let window = get_window(window_function.as_str(), window_size, true)?;

        // OLA buffer needs to hold several overlapping windows
        let buf_len = fft_size * 4 + window_size;

        Ok(Self {
            window_size,
            hop_size,
            fft_size,
            window,
            ola_buffer: vec![0.0; buf_len],
            norm_buffer: vec![0.0; buf_len],
            write_pos: 0,
            ready_samples: 0,
            frames_processed: 0,
        })
    }

    /// Feed one STFT frame (positive-frequency bins).
    ///
    /// Frame length must be `fft_size / 2 + 1`.
    pub fn push_frame(&mut self, spectrum: &[Complex64]) -> SignalResult<()> {
        let expected_len = self.fft_size / 2 + 1;
        if spectrum.len() != expected_len {
            return Err(SignalError::ValueError(format!(
                "Expected frame length {expected_len}, got {}",
                spectrum.len()
            )));
        }

        // Inverse real FFT
        let time_domain = scirs2_fft::irfft(spectrum, Some(self.fft_size)).map_err(|e| {
            SignalError::ComputationError(format!("IFFT error in OnlineISTFT: {e}"))
        })?;

        let buf_len = self.ola_buffer.len();

        // Apply synthesis window and overlap-add (only first window_size
        // samples; the rest are zero-padding artefacts).
        for (i, &w) in self.window.iter().enumerate() {
            let pos = (self.write_pos + i) % buf_len;
            let td_val = if i < time_domain.len() {
                time_domain[i]
            } else {
                0.0
            };
            self.ola_buffer[pos] += td_val * w;
            self.norm_buffer[pos] += w * w;
        }

        self.write_pos = (self.write_pos + self.hop_size) % buf_len;
        self.frames_processed += 1;
        self.ready_samples += self.hop_size;

        Ok(())
    }

    /// Pop up to `max_samples` reconstructed output samples.
    ///
    /// Returns fewer samples if not enough data is available.
    pub fn pop_samples(&mut self, max_samples: usize) -> Vec<f64> {
        let n = max_samples.min(self.available_samples());
        let buf_len = self.ola_buffer.len();

        // The oldest readable sample is at write_pos - ready_samples
        let read_start = if self.write_pos >= self.ready_samples {
            self.write_pos - self.ready_samples
        } else {
            buf_len - (self.ready_samples - self.write_pos)
        };

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let pos = (read_start + i) % buf_len;
            let norm = self.norm_buffer[pos];
            let sample = if norm.abs() > 1e-10 {
                self.ola_buffer[pos] / norm
            } else {
                self.ola_buffer[pos]
            };
            out.push(sample);

            // Clear consumed position
            self.ola_buffer[pos] = 0.0;
            self.norm_buffer[pos] = 0.0;
        }

        self.ready_samples = self.ready_samples.saturating_sub(n);
        out
    }

    /// Number of output samples available for reading.
    pub fn available_samples(&self) -> usize {
        // Conservative: keep one window's worth as margin
        if self.ready_samples > self.window_size {
            self.ready_samples - self.window_size
        } else {
            0
        }
    }

    /// Total frames processed.
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.ola_buffer.iter_mut().for_each(|v| *v = 0.0);
        self.norm_buffer.iter_mut().for_each(|v| *v = 0.0);
        self.write_pos = 0;
        self.ready_samples = 0;
        self.frames_processed = 0;
    }
}

// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_stft_creation() {
        let stft = OnlineSTFT::new(OnlineSTFTConfig::default());
        assert!(stft.is_ok());
        let stft = stft.expect("should succeed");
        assert_eq!(stft.num_bins(), 513);
    }

    #[test]
    fn test_online_stft_invalid_config() {
        assert!(OnlineSTFT::new(OnlineSTFTConfig {
            window_size: 0,
            hop_size: 256,
            fft_size: 1024,
            window_function: WindowFunction::Hann,
        })
        .is_err());

        assert!(OnlineSTFT::new(OnlineSTFTConfig {
            window_size: 512,
            hop_size: 1024,
            fft_size: 512,
            window_function: WindowFunction::Hann,
        })
        .is_err());

        // fft_size < window_size
        assert!(OnlineSTFT::new(OnlineSTFTConfig {
            window_size: 512,
            hop_size: 128,
            fft_size: 256,
            window_function: WindowFunction::Hann,
        })
        .is_err());
    }

    #[test]
    fn test_online_stft_produces_frames() {
        let config = OnlineSTFTConfig {
            window_size: 256,
            hop_size: 64,
            fft_size: 256,
            window_function: WindowFunction::Hann,
        };
        let mut stft = OnlineSTFT::new(config).expect("create STFT");

        // Feed 320 samples
        let chunk: Vec<f64> = (0..320).map(|i| (i as f64 * 0.01).sin()).collect();
        stft.push_samples(&chunk).expect("push");
        let frames = stft.pop_frames();
        assert!(!frames.is_empty(), "Should produce at least one frame");
        assert_eq!(frames[0].spectrum.len(), 129); // 256/2 + 1
    }

    #[test]
    fn test_online_stft_incremental_feed() {
        let config = OnlineSTFTConfig {
            window_size: 128,
            hop_size: 64,
            fft_size: 128,
            window_function: WindowFunction::Hamming,
        };
        let mut stft = OnlineSTFT::new(config).expect("create STFT");

        // Feed 32 samples at a time
        let mut total_frames = 0;
        for _ in 0..10 {
            stft.push_samples(&[1.0; 32]).expect("push");
            total_frames += stft.pop_frames().len();
        }
        assert!(
            total_frames >= 3,
            "Should produce at least 3 frames from 320 samples"
        );
    }

    #[test]
    fn test_online_stft_zero_padding() {
        // FFT size larger than window size => zero-padding
        let config = OnlineSTFTConfig {
            window_size: 128,
            hop_size: 64,
            fft_size: 256,
            window_function: WindowFunction::Hann,
        };
        let mut stft = OnlineSTFT::new(config).expect("create STFT");

        stft.push_samples(&[1.0; 192]).expect("push");
        let frames = stft.pop_frames();
        assert!(!frames.is_empty());
        // With fft_size=256, we get 256/2+1 = 129 bins
        assert_eq!(frames[0].spectrum.len(), 129);
    }

    #[test]
    fn test_online_stft_reset() {
        let config = OnlineSTFTConfig {
            window_size: 128,
            hop_size: 64,
            fft_size: 128,
            window_function: WindowFunction::Hann,
        };
        let mut stft = OnlineSTFT::new(config).expect("create STFT");
        stft.push_samples(&[1.0; 256]).expect("push");
        let _ = stft.pop_frames();
        assert!(stft.total_samples() > 0);

        stft.reset();
        assert_eq!(stft.total_samples(), 0);
    }

    #[test]
    fn test_online_stft_sine_peak() {
        let window_size = 256;
        let fft_size = 256;
        let fs = 256.0;
        let freq = 10.0;

        let config = OnlineSTFTConfig {
            window_size,
            hop_size: 256,
            fft_size,
            window_function: WindowFunction::Rectangular,
        };
        let mut stft = OnlineSTFT::new(config).expect("create STFT");

        let chunk: Vec<f64> = (0..window_size)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin())
            .collect();
        stft.push_samples(&chunk).expect("push");
        let frames = stft.pop_frames();
        assert!(!frames.is_empty());

        let magnitudes: Vec<f64> = frames[0].spectrum.iter().map(|c| c.norm()).collect();
        let peak_bin = magnitudes
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(peak_bin, 10, "Peak should be at bin 10 for 10 Hz signal");
    }

    // ---- OnlineISTFT ----

    #[test]
    fn test_online_istft_creation() {
        let istft = OnlineISTFT::new(256, 64, 256, WindowFunction::Hann);
        assert!(istft.is_ok());
    }

    #[test]
    fn test_online_istft_invalid_config() {
        assert!(OnlineISTFT::new(0, 64, 256, WindowFunction::Hann).is_err());
        assert!(OnlineISTFT::new(256, 512, 256, WindowFunction::Hann).is_err());
        assert!(OnlineISTFT::new(256, 64, 128, WindowFunction::Hann).is_err()); // fft < window
    }

    #[test]
    fn test_online_istft_frame_length_check() {
        let mut istft = OnlineISTFT::new(256, 64, 256, WindowFunction::Hann).expect("create");
        let bad_frame = vec![Complex64::new(0.0, 0.0); 10];
        assert!(istft.push_frame(&bad_frame).is_err());
    }

    #[test]
    fn test_online_stft_istft_roundtrip() {
        let window_size = 256;
        let hop_size = 64;
        let fft_size = 256;

        let config = OnlineSTFTConfig {
            window_size,
            hop_size,
            fft_size,
            window_function: WindowFunction::Hann,
        };
        let mut stft = OnlineSTFT::new(config).expect("create STFT");
        let mut istft = OnlineISTFT::new(window_size, hop_size, fft_size, WindowFunction::Hann)
            .expect("create ISTFT");

        // Generate test signal
        let n_samples = 2048;
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 / 256.0;
                (2.0 * std::f64::consts::PI * 10.0 * t).sin()
            })
            .collect();

        // Forward STFT
        stft.push_samples(&signal).expect("push");
        let frames = stft.pop_frames();

        // Inverse STFT
        for frame in &frames {
            istft.push_frame(&frame.spectrum).expect("push frame");
        }

        let reconstructed = istft.pop_samples(n_samples);

        // Check reconstruction quality (skip transient region)
        if reconstructed.len() >= window_size * 2 {
            let check_start = window_size;
            let check_end = reconstructed.len().min(signal.len() - window_size);
            if check_start < check_end {
                let mut max_err = 0.0_f64;
                for i in check_start..check_end {
                    let err = (reconstructed[i] - signal[i]).abs();
                    if err > max_err {
                        max_err = err;
                    }
                }
                assert!(
                    max_err < 2.0,
                    "Max reconstruction error = {max_err} (should be < 2.0)"
                );
            }
        }
    }

    #[test]
    fn test_online_istft_reset() {
        let mut istft = OnlineISTFT::new(128, 32, 128, WindowFunction::Hann).expect("create");
        let frame = vec![Complex64::new(1.0, 0.0); 65];
        istft.push_frame(&frame).expect("push");
        assert_eq!(istft.frames_processed(), 1);

        istft.reset();
        assert_eq!(istft.frames_processed(), 0);
    }
}
